# Gap-transport frame observable — implementation record & spec assessment (2026-08-05)

**Trigger:** `SPEC_EXACT_SPHERE_GAP_TRANSPORT_2026_08_05.md` (consumer-authored),
which asks for the paraxial Sziklas-Siegman inter-group transport to be replaced
by an "exact sphere-referenced" transport (its **M1**).

**Outcome in one line:** the spec's Stage 0 instinct was right, for a **narrower
reason than it argues**; its two motivating claims (§4 "no guard covers B", §8
"the chain is slower than the ASM it replaced") **do not survive verification**;
Stages 0-1 are implemented in corrected form; **Stages 2-4 (M1) are declined
pending the measurement Stage 0 now makes possible** — which is the spec's own
stated discipline.

Method: direct code reading, plus two independent Opus verification passes (one
adversarially reviewing *my* assessment, one verifying the spec's quantitative
and cost claims against the repo's recorded evidence). All numbers below are
either measured this session or cited to a repo file:line / doc section.

---

## 1. What was implemented

### 1.1 Stage 0 — the frame observable (`_gap_envelope_angular_spread`)

The chain's justification for a **paraxial frame** is stated at the top of
`propagators/carrier.py`: "the envelope's residual angular content (after the
carrier is divided out)" is small. **That premise was never measured at runtime.**
Arms A/B of `_check_gap_paraxial` are both computed from the *carrier's*
geometry. This function measures the premise directly.

Published on **every** inter-group leg, in `stages[i]`, whether or not anything
fires:

| key | meaning |
|---|---|
| `gap_env_theta` | envelope's residual angular spread, rad |
| `gap_env_nyq_frac` | that spread ÷ the grid Nyquist tilt — **read this before trusting the value** |
| `gap_env_phi_drop` | implied frame-dropped quartic `k·|z_eff|·θ⁴/8`, rad — same convention as the existing `gap_phi_drop`, so the two are directly comparable |
| `gap_z_eff` | the reduced transport distance, m |
| `gap_env_spectral` | which estimator ran (see below) |

**Two estimators, better one preferred.** The adversarial review correctly noted
that `validation/repro_traced_carrier_121/approx_leg_budget_121.py` *already*
measures this quantity offline by FFT power-percentile bandwidth, and that a
wrapped nearest-neighbour difference is the weaker instrument (the library
documents its blindness in three places). So:

* grids ≤ `_GAP_ENV_SPECTRAL_MAX_N` (4096) take the **spectral** route — the
  99.9 % power radius of the envelope's angular spectrum, matching the offline
  script's definition;
* larger grids fall back to the **amplitude-weighted wrapped-difference** rms,
  scanned in 128-row bands so no `(Ny,Nx)` temporary forms. This matters: at the
  design-121 production N=28672 a complex FFT workspace is 12.25 GB, and the
  sibling `_gap_amp_radius` was deliberately written to avoid exactly that.

**Honest limit, applying to both routes.** Neither can see angular content above
the grid's own Nyquist tilt — that is a property of the *sampling*. The
difference route additionally **folds** over-Nyquist content back to a small
reading, which is why `nyq_frac` is published and why a multi-scale (stride-1 vs
stride-2) cross-check raises it on disagreement. Measured residual limit: an
*exactly commensurate* tilt (e.g. precisely 1.5× Nyquist) folds to the same
magnitude at both strides and still evades — no first-difference family can
close that case. This is the "you cannot measure aliasing with the aliased
gradient" lesson from `AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` §4b.3,
applied rather than re-learned.

### 1.2 Stage 1 — arm C of the gap guard, with its **own** knob

New kwargs on both chain entry points: `on_gap_frame` (`'warn'` default) and
`gap_env_phi_tol` (0.30 rad default; `0` = report-without-tripping).

Arm C fires when the **measured** frame drop exceeds `gap_env_phi_tol`, **or**
when `nyq_frac` exceeds `_GAP_ENV_NYQUIST_FRAC` (0.5) — because past that point
the reading is a lower bound and the honest response is to say so, not to report
a confident small number.

**Threshold provenance, stated plainly:** 0.30 rad is `gap_sag_tol`'s value
carried across by **dimensional analogy** (both are radians of dropped quartic),
**not** an independent end-to-end calibration of the frame axis. Arm B's NA table
*is* ASM-calibrated, but along the proxy axis. Producing the frame-axis
calibration is what this observable exists for; the warning text says so
verbatim. Measured margins make the analogy safe in practice: healthy legs sit
~3 orders below the threshold and the pathological case ~2 orders above (§3).

---

## 2. Two defects in my own first implementation, found by adversarial review

Recording these because both were real and neither was in the spec:

1. **A collimated leg (R = inf) took no gap arm at all.** The whole guard was
   gated at the call site on `isfinite(R)`. That is exactly backwards for the
   frame arm: with no co-moving reduction `z_eff = z`, its **largest** possible
   value, so `k|z_eff|θ⁴/8` is *maximal* on precisely the legs that were
   unguarded — and roadmap P8 names "a fast final group after a collimated
   space" as **the most common free-space relay architecture**. My first version
   even computed the spread there and discarded it. Fixed: the guard is now
   called for any scalar `R`, and arms A/B self-silence on a collimated leg
   (`phi_drop = 0`, `na = 0`) so the change is inert for them.
2. **Arm C initially shared `on_gap_paraxial`.** Since arm C's threshold is
   uncalibrated, a noisy trip invites a global `'ignore'` — which would also
   have silenced the two *calibrated* carrier-geometry arms. The spec's §6 asked
   for a separate `on_gap_frame`; my implementation had diverged. Fixed, and
   pinned by `test_arm_c_has_its_own_knob_and_does_not_silence_arms_a_b`.

Two of my own test expectations were also wrong and were corrected rather than
worked around: a "collimated" fixture whose gap actually sat *after* a +200 mm
singlet (so R ≈ −198 mm, `z_eff` 0.0501 not 0.040), and a grid whose half-width
(768 µm) was **smaller than the beam** (w0 = 1500 µm), so the truncation edge
legitimately produced large spectral content. Both were harness errors that the
new observable correctly exposed.

---

## 3. Measured results

Estimator validation (spectral route, N=256):

* real-valued Gaussian envelope → θ = 4.0e-03 rad, i.e. the beam's **own
  diffractive floor** (λ/πw = 2.1e-03 for w=200 µm; the 99.9 % radius sits a few
  × above it). A phase-difference estimator reads 0.0 here; the spectral value is
  the physically correct one, because that content is real and does drive the
  frame term.
* monotone in injected quartic residual; flags its own undersampling at 0.9×
  Nyquist; difference-fallback recovers a known 5e-03 rad tilt to 2 %.

Chain-level (self-contained relays, no `.zmx`):

| case | carrier NA (arm B) | measured envelope θ | implied frame drop | arm B | arm C |
|---|---|---|---|---|---|
| benign long-gap relay (121-like) | 0.0067 | 9.0e-03 rad | **1.9e-04 rad** | silent | silent |
| faster short-conjugate relay | 0.0724 | 1.2e-02 rad | **8.8e-04 rad** | silent | silent |
| **carrier-mismatched envelope** | **0.0067** | **0.22 rad** | **73 rad** | **silent** | **FIRES** |

**That third row is the whole justification for the work.** The carrier-NA proxy
reads 0.0067 — 90× below arm B's 0.60 threshold, entirely silent — while the
envelope's own angular content is 0.22 rad and the implied frame drop is 73 rad.
The blind spot the spec's Stage 0 was reaching for is real, is now instrumented,
and is now guarded.

Equally important, the first two rows: on healthy relays the frame term is
**1.9e-04 – 8.8e-04 rad**, i.e. ~3 orders below the 0.30 threshold. Combined with
§4's finding that the repo has *already* measured this axis null on 121's six
legs (Fresnel rms 8.8e-09 → 9.4e-07 waves, envelope NA 0.00012 → 0.01198), the
evidence says **approximation B is not a live problem in the validated regime** —
which is the central reason M1 was declined.

**No regressions.** `test_niche_c3_gap_paraxial_guard` + `test_niche_p2_guards` +
`test_carrier_referenced`: 48 passed. `test_niche_d5_dx_flatness_gate` +
`test_niche_p2_design_battery` (17-case singlet/doublet/triplet/relay × NA ×
aperture:beam): **30 passed, zero frame warnings** — no false alarms on real
designs. New file `tests/unit/test_niche_gap_frame_observable.py`: 12 passed.

---

## 4. What was declined, and why

**Stages 2-4 (M1, the exact sphere-referenced transport) — NOT implemented.**
Four independent reasons, in order of weight:

1. **The spec's own staging says measure first**, and the measurement did not
   previously exist at runtime. It does now (§1.1). Building the fix before the
   observable is the failure mode this repo has already paid for once: the Part-E
   `wavefront_aware` launch was implemented, measured to *regress* the target
   (EE6 50 %→10 %), and reverted — recorded in
   `AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` §4b.
2. **The premise "B is unpriced" is substantially false.** Arm B is
   ASM-calibrated along the NA axis (0.000 EE points to NA 0.45), and
   `approx_leg_budget_121.py` has already measured the envelope-spectrum axis on
   121's six legs. `carrier.py` goes further and records a measurement that for
   a non-crossing leg the shipped step "reproduces a plain Fresnel transfer
   function on the FULL field to 3e-8 of peak, so SS-vs-ASM **IS**
   Fresnel-vs-exact and nothing else" — i.e. B largely *reduces to* A there, and
   A is measured null at 3.56e-07 relL2. The spec's clean A/B split double-counts
   one θ-expansion evaluated at two different θ.
3. **M1 may be worse than what it replaces, in exactly the regime it targets.**
   Its exactness rests on the sphere↔plane-wave (Debye/stationary-phase)
   projection, whose error is O(1/N) in the **Fresnel number** N = w²/(λ|R|) —
   not O(1/kR). Since N = NA·(w/λ), it *degrades* for small intermediate beams,
   which is the short-conjugate/fast-intermediate case §7.1(b) names as
   acceptance. The spec sets no error budget for M1's own Debye + resampling
   error, so Stage 3's tolerance would be unanchored.
4. **The spec cannot yet name a design where Sziklas-Siegman fails.** §7.1
   requires one as *acceptance*, but §6 lets Stages 0-2 proceed without it. That
   inverts the risk. Given (2), the cheapest decisive next step is not M1
   plumbing — it is running the **existing** `approx_leg_budget_121.py` against a
   short-conjugate / high-intermediate-NA prescription and seeing whether the
   dropped term actually becomes large. Hours of work; it either justifies the
   whole spec or kills it.

**M2 (exact-eikonal scaled frame) — recommend dropping, with a reason the spec
does not give.** Sziklas-Siegman is an instance of the dilation symmetry of the
*paraxial* (Schrödinger) equation: `∇⊥²` is homogeneous of degree −2, so
`x → x/m(z)` is absorbable. The exact one-way operator
`∂_z = i√(k² + ∇⊥²)` is **not homogeneous** — dilation sends `∇⊥² → ∇⊥²/m²` but
leaves the fixed scale `k` alone, breaking the symmetry SS depends on. Any
"exact-eikonal scaled frame" must therefore truncate (back to paraxial), carry an
infinite series in `∇⊥²/k²`, or go nonlocal — which matches the literature
(wide-angle BPM is Padé/series). Strong structural argument, not a proof of
impossibility. A better candidate for that slot: keep the exact kernel and do the
coordinate scaling with a **chirp-z/Bluestein** transform, which the library
already runs on the exact final leg — held to the same "show it closes first"
standard.

---

## 5. Spec claims that did not survive verification

Full detail is in the response block prepended to the spec itself. Summary:

| § | Claim | Verdict |
|---|---|---|
| §3 | four Fresnel-kernel null figures | **VERIFIED exactly** (`APPROXIMATION_AUDIT_TRACED_2026_07_30.md`) |
| §4 | "No guard covers B" | **FALSE** — arm B is the frame arm, ASM-calibrated; C3 shipped it in response to the same P7 sentence quoted as unfulfilled |
| §4 | "4.08x margin" | 4.06 (code says 4.1); and it is arm B's number |
| §4 | roadmap quote 1 | verbatim but **superseded** 6 days before the spec |
| §4 | roadmap quote 2 | **not verbatim anywhere**; qualifier dropped |
| §8 | "chain slower than the ASM it replaced" | **INVERTS at the converged grid** — 0.92 h vs 2.5 h at N=2048 (2.7× faster); loses only at the grid the repo says buys zero accuracy. **But the O(N_beams) structural half STANDS** — see §5.1 |
| §8 | the ~2.5 h reference run | **denominator sound** (author-confirmed exact-ASM run at N=28672); an earlier draft of this audit wrongly inferred otherwise — see §5.1 |
| §8 | BPM "F/1.6" justification | **withdrawn** by the repo ("it was wrong"; beam runs f/26-f/44) |
| §8 vs §5 | "gaps are where ASM is exact and cheap" vs M3 rejected because ASM on a gap *is* the fixed-grid run | **internally inconsistent** |
| §5 | M1 "exact by construction" | **overstated**; exactness rests on a Debye step, error O(1/N) not O(1/kR), and N degrades where M1 is most wanted |
| §2 | "verified to 2.2e-16" | **unsupported** (recorded 1.8e-03→4.9e-05 rad; likely a typo for 2.2e-04) |
| §2 | "0.000 EE pts across NA 0.35-0.45" | merges two sweeps; "0.35" appears nowhere |
| §2 | C9 "must not be re-litigated" | understates scope — C9 is **+1.4011 EE3 points** on a tilted congruence, not a null |
| §7.2 | every traced acceptance is monochromatic | **VERIFIED** |

### 5.1 Correction to this audit, and the part of §8 that stands (2026-08-05)

**Correction.** An earlier draft of this audit asserted that §8's ~2.5 h figure
"is the traced chain at N=28672, not an ASM run, and that run formed no focus."
That over-extended the evidence. What the verification actually established is
narrower: the *only 2.5 h figure documented in the repo*
(`AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md`:13) describes a **traced** run at
that grid which formed no focus. Inferring from that document that §8's figure
was therefore not an ASM run does not follow — the two runs are distinct. The
spec author confirms (2026-08-05) that **an exact-ASM run at N=28672 did complete
in ~2.5 h**, carrying physical inaccuracies that were being addressed on a
separate axis. §8's denominator is therefore sound, which makes the
converged-grid inversion **firmer** than this audit originally recorded, not
weaker. The BPM/"F/1.6" withdrawal is a separate and unaffected finding.

**The structural half of §8 stands, and it is the more important half.** Cost
scales differently in beam count for the two methods:

* the traced chain is **O(N_beams)** — the traced element model cannot represent
  a multi-valued field, so every congruence is an independent pass;
* exact ASM is **O(1) in beam count** — one multiplexed field, one pass.

Measured crossover at the converged grid (chain 104 s/run at N=2048; ASM 2.5 h
for all beams): **~86 beams serial**, or ~700 with 8 `congruence_workers`
(niche D8; process-parallel and FP-identical to serial; ~1.5 GB/worker at
N=2048 from the measured `22·N²·16 B` model, so 8 workers is comfortable on a
34 GB box). At the 32-order study the chain has real headroom — 32 × 104 s =
0.92 h serial, ~7-10 min at 8 workers, against 2.5 h. Well beyond ~86 orders the
comparison reverses and stays reversed.

**Consequence for this spec: M1 does not address that axis.** The N-fold cost
lives in the *elements*, not the gaps, so an exact gap transport leaves it
exactly where it was — and §8 itself pitches M1 on *exactness* ("reserve the ray
machinery for the glass"), not on collapsing N passes into one. The levers for
large beam counts are, in order: `congruence_workers` today, and a
multiplexing-capable element model later. Neither is this spec, and neither is
blocked by it.

---

## 6. Pre-existing defect found, NOT fixed

`tests/unit/test_niche_d3_guards.py:171` documents its `'aberrated'` fixture as a
"smooth r^4, ~8 rad p-v" residual, but the literal
`(X²+Y²)²/(8*0.05**3)` yields **6.2e-08 rad** at r = w (≈ 5.6e-07 rad p-v across
the bright mask) — eight orders short. For ~8 rad the radius literal needs to be
≈ 9.9e-05 m. Consequences: the P3 evidence row "Gaussian × smooth r^4 residual
(8 rad p-v) → A 0.0000 / B 0.000000" recorded in `carrier.py` is **vacuous**, and
`test_single_valued_inputs_sit_far_below_the_multivalued_cutoff('aberrated')`
passes trivially. Net effect: the library had **no** measurement of what its
wrapped-increment estimator does on a genuinely aberrated envelope — precisely
the input class this work instruments. Left for the owner because it changes a
documented evidence row in someone else's niche, but it is a real hole and the
new `test_niche_gap_frame_observable.py` now covers that input class directly.

---

## 7. Recommended next steps, in order

1. **Run `approx_leg_budget_121.py` on a short-conjugate / high-intermediate-NA
   prescription** (spec §7.1(b)/(c)). Cheapest decisive experiment available; it
   either establishes that the frame error exists somewhere real or retires the
   spec. Do this **before** any M1 work.
2. **Calibrate `gap_env_phi_tol` on the frame axis** with an EE-point sweep, the
   way niche C3 earned arm B's threshold, and replace the analogy-based 0.30.
3. Only if (1) finds a real failure: **Stage 2 (M1 behind a flag)**, with an
   explicit error budget for M1's own Debye + resampling error, and Stage 3's
   negative control as the acceptance gate.
4. Fix the §6 `test_niche_d3_guards.py` fixture and re-measure the P3 row it
   supports.
5. Consider promoting `approx_leg_budget_121.py`'s spectral estimator into a
   supported diagnostic entry point, since it is now half-shipped (the spectral
   route in §1.1 uses its definition).
