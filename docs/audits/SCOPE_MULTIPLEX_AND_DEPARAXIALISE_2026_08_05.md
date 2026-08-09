# Scoping — multiplexed passes, and removing the remaining paraxial steps (2026-08-05)

> ## MEASURED (2026-08-05, same day) — two items below are now settled, and one of my own recommendations is RETRACTED
>
> Design-121 8-group chain, N=2048 (the shipped converged grid), `rs=4`,
> `nfc=4096`, `wf=2.0`. Memory-guarded; **peak RSS 3.48 GB of 34.3 GB**.
>
> ### (A) Profile — the batched-K-stack item is DEAD
>
> | category | self time | share |
> |---|---|---|
> | **ray trace** | **154.4 s** | **70.9 %** |
> | other | 40.3 s | 18.5 % |
> | FFT / spectral | 19.4 s | 8.9 % |
> | elementwise numpy | 1.8 s | 0.8 % |
> | sag / phase screen | 1.7 s | 0.8 % |
>
> (217.7 s wall under `cProfile` at `n_workers=4`; the 104 s production figure is
> un-profiled at `n_workers=8`. Shares are what matter, not the absolute.)
>
> **The chain is ray-trace bound, not FFT bound.** Batching FFTs across K
> congruences therefore addresses ~9 % of the time — an Amdahl ceiling of ~1.1×
> even if batching were free. **Do not build the batched K-stack.** Corollary:
> the only large per-run lever is the 71 % ray-trace block, i.e. isoplanatic
> screen reuse — which is approximation-bearing and therefore ruled out by the
> "no paraxial/approximation" criterion. **`congruence_workers` remains the
> answer for multi-beam cost**, and it is already shipped.
>
> Incidental: `scipy.ndimage.geometric_transform` is 12.1 s / 72 calls (the
> remap interpolation) and in-glass `angular_spectrum_propagate` is 12.7 s / 11
> calls — the largest single non-ray items, if anyone wants a smaller target.
>
> ### (B) Hand-off `R_out` — **I over-claimed; P2 is downgraded**
>
> | group | R_par (mm) | R_best-fit (mm) | rel err | residual defocus |
> |---|---|---|---|---|
> | Lens S3-S4 | 143.37 | 142.11 | 8.8e-03 | 3.7 rad |
> | Lens S5-S7 | 703591 | 10776 | 9.9e-01 | 6.4 rad |
> | Lens S14-S15 | 703613 | 10111 | 9.9e-01 | 6.8 rad |
> | Lens S16-S17 | 703620 | 10248 | 9.9e-01 | 6.8 rad |
> | Lens S18-S20 | −263.19 | −277.91 | 5.6e-02 | 14.0 rad |
> | Lens S21-S22 | −60.15 | −61.27 | 1.9e-02 | 16.5 rad |
> | Lens S23-S24 | −24.46 | −24.67 | 8.7e-03 | 14.0 rad |
> | Lens S25-S27 | −7.71 | −7.88 | 2.2e-02 | 83.4 rad |
>
> The paraxial `R_out` *does* differ from the best-fit quadratic — up to 98.6 %
> in `R` on the near-collimated mid-chain groups, and 3.7-83.4 rad (≈0.6-13
> waves) of residual quadratic phase left in the envelope. My scoping called this
> "likely the biggest accuracy win." **That was wrong, and the measurement says
> so.** Three reasons:
>
> 1. **It is a representation choice, not an accuracy defect.** The field is
>    `E = env·exp(ikS(R))`. If `R` is the paraxial value rather than the best-fit
>    one, **the envelope absorbs the difference exactly** — the total field is
>    unchanged. It only becomes an *error* through the paraxial frame's
>    sensitivity to envelope angular content, or through sampling.
> 2. **That sensitivity is measured, and negligible.** The same run's
>    `gap_env_theta` reads **1.2e-04 - 4.7e-03 rad** — consistent with the
>    defocus figures (a few rad over a ~5 mm beam ⇒ ~1e-3 rad of angle), and
>    implying a frame-dropped term of order **1e-07 rad**, i.e. ~6 orders under
>    arm C's 0.30 rad tripwire.
> 3. **The chain already hits the ideal-field ceiling with the paraxial `R_out`**
>    (best-focus FWHM 3.450 µm against a measured ideal ceiling of 3.45-3.55 µm,
>    EE6 99.6 % — `AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md` closure
>    block). An `R_out` that was costing real accuracy could not coexist with
>    that.
>
> Also, a large part of the headline "98.6 %" is an artefact of the regime, not a
> defect: at R ≈ 7e5 mm the carrier is essentially collimated, so an
> ~1-wave curvature difference maps to a huge *relative* change in `R`. In
> curvature terms it is ~1 wave of defocus, which is ordinary residual aberration
> of a real group against its paraxial idealisation — exactly what one should
> expect, and what the envelope exists to carry.
>
> **Revised status of P2:** a **conditioning nicety, low priority** — it would
> make the envelope carry less phase (marginally better sampling headroom and a
> better-conditioned frame), not fix an accuracy problem. The 83 rad on the fast
> final group is the one row worth a second look, since that is ~13 waves for a
> "slowly varying" envelope to carry, but that group already routes through the
> exact final leg.
>
> **Net effect on the priority order:** P3 (anisotropic tilt stretch — 7.1e-05
> waves worst leg, tilt-scaled so it hits the off-axis DOE orders) and P1
> (`gap_kernel='exact'` — free) are now the top two paraxial items. P2 drops
> below them. The batched-K-stack is removed.

Two consumer questions, answered together because the answer to the first
reorders the second:

1. **Is a multiplexed pass (all K congruences in one traversal) feasible?**
2. **"I don't like using paraxial anywhere — even if it's N runs, I'd prefer ASM
   where possible for accuracy per run."**

Short answers: **(1) not as a free algorithmic win — linearity makes K passes the
*correct* decomposition, and the ray/OPL cost is irreducibly O(K) without
introducing an approximation.** **(2) The remaining paraxial surface is smaller
than expected — two of the four candidate sites are already exact — and the two
that remain are both cheap to address.** Detail below, with the paraxial work
scoped as the actionable path since it is what you actually asked for.

---

## Part 0 — Verified paraxial inventory (what is actually left)

Measured against the tree at v5.32.1, not from documentation:

| # | step | status | where |
|---|---|---|---|
| 1 | **in-glass propagation** | **ALREADY EXACT** — `get_default_wave_propagator()` returns `'asm'`, dispatching to band-limited `angular_spectrum_propagate` | `_lens_real.py:1689` |
| 2 | **final leg to the image** | **ALREADY EXACT** — `final_leg='exact'`/`'auto'` (R9), exact-sphere reference + band-limited Bluestein ASM | `carrier.py`, niche R9 |
| 3 | **gap KERNEL** | **PARAXIAL — Fresnel TF**, and not overridable today (`fresnel_tf_propagate`, 8 call sites) | `_carrier_step_fast`, `carrier.py:532` |
| 4 | **gap FRAME** | **PARAXIAL** — Sziklas-Siegman coordinate scaling `m = R_out/R`, reduced leg `z_eff = z·R/R_out`, derived from the paraxial wave equation | `carrier.py:513-558` |
| 5 | **hand-off carrier R_out** | **PARAXIAL** — ABCD Möbius `R_out = (A·R_in+B)/(C·R_in+D)` | `_paraxial_group_r_out`, `carrier.py:3349` |
| 6 | anisotropic tilt stretch (approx. "C") | **UNIMPLEMENTED** — documented in `_tilt_obliquity`, transport uses `z` not `z/(1−L²)^{3/2}`; priced 7.1e-05 waves rms worst leg / 2.2e-03 summed on 121 | `carrier.py:3444` |

So: **#3, #5, #6 are the removable paraxial steps; #4 is the structural one.**
That is a materially smaller surface than the spec implies, and #5 in particular
has not been discussed anywhere in the spec.

---

## Part 1 — Is a multiplexed pass feasible?

### 1.1 The structural answer: K passes is the correct decomposition, not waste

The chain represents the field as `E = env · exp(i k S(r; R))` — **one** carrier.
A multiplexed field (32 DOE orders) has 32 simultaneous local wavevectors, so:

* no single sphere de-chirps all of them — dividing by any one carrier leaves the
  other 31 as fast beat structure, and the "envelope" is no longer
  slowly-varying, which is the entire premise of the small grid;
* the ray launch needs a **single-valued** local wavevector; a multi-valued field
  has no single-valued eikonal. This is the multi-valued/caustic problem, and the
  library already detects it (`on_multi_congruence`, `multi_congruence_threshold`).

**The key point, and the reason "multiplexing" cannot buy what one hopes:** the
optics are **linear**, so through a lens group congruence *k* maps to exit
congruence *k* and they **never mix**. Superposition only needs re-forming at the
readout. There is therefore no cross-congruence redundancy for an algorithm to
exploit — the K ray traces are genuinely K different computations (different ray
paths through the glass), not K copies of one.

This is why `propagate_traced_carrier_chain_multi` (niche D2) is architecturally
right as K independent chains, and why the O(K) cost is honest rather than a
missed optimisation.

### 1.2 What sharing IS available, and its exact limit

`PreparedTracedLens` already exists and already exploits the one real
redundancy: the entire expensive traced leg (ray trace, Chebyshev/spline fit,
Newton inversion, `phase_analytic_lens`, OPL map, masks) depends only on
`(prescription, wavelength, dx, N, carrier)` — **not** on the envelope. So one
screen serves many *fields*.

**The blocker for DOE orders is that `carrier` is in that tuple.** Orders differ
by tilt (`TiltedCarrier`), the tilt enters the element (`centre=(x_c,y_c)`,
`tilt=(L,M)` at `carrier.py:6550`), and a tilted bundle traverses different ray
paths through the glass — so the OPL map is genuinely different, not a shifted
copy. **No exact screen reuse across orders.**

There is an *approximate* version — isoplanatism: for small tilts the screen is
approximately the untilted screen shifted. That would collapse K traced legs to
one plus K shifts, which is the single largest possible win in this whole
document. **But it is an approximation**, exactly the kind you have said you do
not want, so it is listed below as opt-in-and-measured, never default.

### 1.3 Realistically available gains, ranked

| lever | mechanism | expected gain | approximation? |
|---|---|---|---|
| **`congruence_workers`** (shipped, niche D8) | processes; FP-identical to serial; ~1.5 GB/worker at N=2048 by the measured `22·N²·16 B` model | near-linear in workers — 32 orders ≈ 7-10 min at 8 workers vs 0.92 h serial | **none** |
| **Batched K-stack** | carry `(K, N, N)` envelopes, batch the FFTs and amortise per-call Python/dispatch over K | **unknown until profiled** — same FLOPs; helps only in proportion to how much of the 104 s/run is FFT + overhead rather than ray/Newton work | none |
| Geometry-only caches | cos-grid cache, sag reuse | already shipped (R1/B1) | none |
| Isoplanatic screen reuse | one traced screen + K shifts | potentially **~K×** on the traced leg | **YES** — field-dependent aberration dropped; must be measured and gated |
| GBD | natively multi-valued (beamlets carry their own directions) | would be a true single multiplexed pass | measured **2.4 % chain power** at 121 conditions — not currently viable |

**Prerequisite for the batching estimate: a profile.** Nobody has recorded where
the 104 s/run at N=2048 goes. If it is ray-trace/Newton bound, batching buys
little and only isoplanatism moves the needle; if it is FFT/wave bound, batching
is worth real multiples. **This is ~1 hour of work and it decides whether the
batched-K-stack item is worth building at all** — the same "measure before
building" discipline applied elsewhere in this campaign.

---

## Part 2 — Removing the paraxial steps (the actionable path)

### P1 — `gap_kernel='exact'`: swap the Fresnel TF for the exact ASM TF · **cheapest real win**

`_carrier_step_fast` propagates the envelope over the reduced leg with
`fresnel_tf_propagate`. The exact kernel `exp(i k z √(1−(λf)²))` is the same
FFT → multiply → IFFT with a different exponent, and `angular_spectrum_propagate`
(band-limited, Matsushima) already exists.

* **Cost: essentially zero.** Same FFT count; the only extra work is a `sqrt` over
  the frequency grid, which is per-`(z, grid)` and cacheable.
* **Effect:** removes paraxial step #3 outright.
* **Expected magnitude on 121-class geometry: small but real.** The spec's own S2
  measurement (verified) records the swap moving the end-to-end answer by
  relL2 3.56e-07 / +0.0000 EE3 points, with a closed-form dropped term of
  9.4e-07 waves rms worst leg. It is *free accuracy*, not a big accuracy gain —
  worth taking precisely because it costs nothing.
* **Honest caveat that must be measured, not assumed.** The SS change of
  variables was *derived* from the paraxial equation, and `carrier.py:3742-3745`
  records that SS-with-Fresnel reproduces a plain Fresnel TF on the full field to
  3e-8. Substituting the exact kernel into a paraxially-derived frame gives
  "exact kernel in an approximate frame" — plausibly closer to truth, but **not
  provably so**. It needs a full-field exact-ASM oracle on one leg to confirm the
  change moves *toward* truth rather than merely *away from Fresnel*. Do not ship
  the default flip without that oracle.
* **Shape:** `gap_kernel='fresnel'` (default, FP-identical) | `'exact'`. Effort: S
  (hours, incl. the oracle).

### P2 — exact (real-ray) hand-off `R_out` instead of the paraxial ABCD · **likely the biggest accuracy win here**

Not mentioned anywhere in the spec, and probably more consequential than the
kernel. The carrier radius handed between groups comes from the **paraxial** ABCD
Möbius law (`_paraxial_group_r_out`). On a real group with spherical aberration
the true exit wavefront curvature differs from the paraxial one.

**Why this matters more than it looks, and why it is now measurable:** if the
handed-off `R` mismatches the beam's actual curvature, the difference does not
vanish — it becomes **residual non-spherical content carried by the envelope**,
which is precisely the quantity the new Stage-0 observable measures
(`gap_env_theta`), and precisely what degrades the SS frame (#4). So a paraxial
`R_out` *manufactures* frame error. The two issues are coupled, and P2 attacks
the cause rather than the symptom.

* **Mechanism:** the traced leg already computes the exit OPL map; fit its
  curvature (a low-order term of the existing Chebyshev fit, already available)
  and hand *that* to the next gap instead of the ABCD value.
* **Immediate diagnostic, before writing any code:** on the 121 chain, compare
  `_paraxial_group_r_out`'s value against the traced exit curvature per group, and
  read `gap_env_theta` on each leg. If the paraxial `R_out` is already accurate,
  `gap_env_theta` stays at the ~1e-3 rad floor and P2 is unnecessary; if it is
  off, the observable will show it. **~1-2 hours, and it decides the item.**
* **Shape:** `handoff_curvature='paraxial'` (default) | `'traced'`. Effort: M.
* **Risk:** the traced curvature is a fitted quantity with its own noise; a noisy
  `R` could be worse than a smooth-but-biased one. Must be gated on the
  observable improving, not on the curvature "looking right."

### P3 — implement approximation "C", the anisotropic tilt stretch

Documented in `_tilt_obliquity` but not applied: under tilt the envelope's own
diffraction wants `z/(1−L²)^{3/2}` along the tilt and `z/(1−L²)^{1/2}` across it
(+0.32 % / +0.11 % at 46 mrad). Priced on 121 at 7.1e-05 waves rms worst leg,
2.2e-03 summed — i.e. ~75× larger than the kernel term (P1), and the largest
*quantified* paraxial residual in the gap.

* **Mechanism:** the separable per-axis SS step already exists
  (`_carrier_step_fast_1d`, astigmatic path), so this is an anisotropic `z_eff`
  rather than new machinery.
* **Relevance to you specifically:** it scales with tilt, so it matters *most*
  for the off-axis DOE orders — the multiplexed study is exactly where it bites.
* Effort: S-M. **Recommend doing this before P1**, since it is priced ~75× larger
  and the mechanism is already present.

### P4 — the SS frame itself (#4)

The structural one, and the spec's M1/M2 target. Position unchanged from
`AUDIT_GAP_FRAME_OBSERVABLE_2026_08_05.md` §4: **do not build until the new
observable says it matters on a real design.** Current measurements — 1.9e-04 rad
implied frame drop on a benign relay, 8.8e-04 on a faster one, against arm C's
0.30 rad tripwire — say it does not, yet. M2 is structurally unlikely to close
(the exact operator breaks the dilation symmetry SS relies on); M1 trades the
paraxial-frame error for a **Debye** error of O(1/N) in the Fresnel number
`N = w²/(λ|R|)`, which *degrades* for small intermediate beams — so on a
"no approximations" criterion M1 is not obviously an upgrade, just a different
approximation. If #4 must go, the honest route is full-field exact ASM on that
leg at λ/(2NA) sampling, accepting the grid cost for the legs that need it —
which is a *hybrid*, not a chain replacement.

---

## Part 3 — Recommended order

1. **Profile one chain run** (~1 h). Decides whether batched-K is worth building,
   and gives everything else a cost baseline. Nothing else here is blocked by it.
2. **P2 diagnostic** (~1-2 h): paraxial vs traced `R_out` per group, read against
   the new `gap_env_theta`. Highest information-per-hour in this document,
   because it tests whether a paraxial step is *manufacturing* the frame error.
3. **P3** (anisotropic tilt stretch): largest quantified paraxial residual,
   mechanism already present, and it is tilt-scaled so it hits the DOE orders.
4. **P1** (`gap_kernel='exact'`): free, but land it *with* the full-field oracle
   that confirms the direction of the change.
5. **`congruence_workers`** for the 32-order study now — it is shipped, exact,
   and turns 0.92 h into ~7-10 min. No new code.
6. **P4 / M1** only if the observable, on a real design, says the frame error is
   live. Then with an explicit error budget for whatever replaces it.
7. Isoplanatic screen reuse only if profiling shows the traced leg dominates AND
   you decide the field-dependence approximation is acceptable — measured and
   opt-in, never default.

## Part 4 — What I would not do

* **Build M1 now.** Trades one approximation for another, is unproven in the
  regime it targets, and the observable currently says the error it fixes is
  ~3 orders under the tripwire.
* **Chase multiplexing as a cost fix.** Linearity means the O(K) ray cost is
  real. Parallelism already recovers most of it; the only large win is
  approximation-bearing.
* **Flip any default without a full-field oracle**, including P1's free swap. The
  reason is on the record: this codebase has already shipped one measured-null
  "improvement" that turned out to *regress* the target when finally measured
  end-to-end (`AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` §4b).
