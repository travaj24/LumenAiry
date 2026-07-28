# Design 121 (full configuration) — what the library still needs

Consumer-side requirements note, written against **v5.31.0** (verified at that
tag, not inferred). Companion to `AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md`
(F-A/F-C/F-D shipped; see §7 on F-B's record) and
`AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md`.

**Premise: the refractive half is done.** The v5.29 default flip
(`carrier_reference='sphere'` + `preserve_input_phase='remap'` +
`amplitude_model='ray_density'` + `remap_sampling='full'`) took the design-121
8-group relay from EE6 69.7% to **EE6 99.6 / EE3 88.8 / FWHM 3.450 um** at
N=2048, dx0=1.0 um — at the measured ideal-field ceiling for that readout.
Nothing below asks for more accuracy on that path.

**What this note is about:** design 121 is not a bare relay. The shipping
device is a **Dammann DOE feeding an 8x4 order fan (480 um pitch, +-46 mrad)
from an emitter array**. Every item below is a blocker or a guard rail for
running the *full* configuration, and none of them moved in v5.30 or v5.31.

Priorities are consumer-side (what blocks the study), not library-internal.

---

## P1 — A first-class "N independent congruences through one chain" facility

**This is the single blocker.** It has two faces that are the same feature:

* **Per-order (DOE).** A post-DOE field superposes 32 comparable-power beams at
  well-separated angles. `apply_real_lens_traced`'s entrance->exit map assumes
  one congruence per exit pixel, so the fan cannot go through the chain
  multiplexed — the library says so itself (`_lens_traced.py:2170-2179`, which
  names "comparable-power beams at well-separated angles (post-DOE at large
  split)" as the excluded case).
* **Per-emitter (array).** Design 121 is a 4x4 / 8x8 emitter array. The
  existing multi-emitter entry point `apply_real_lens_traced_multi` **cannot
  express the validated configuration**: it fixes `preserve_input_phase=True`,
  and `amplitude_model='ray_density'` / `fit_radius_beam_factor` now *raise* on
  the default `reuse_prepared=True` (`_lens_traced.py:5111-5137`). So the array
  case is currently locked out of the v5.29 physics that made the single-beam
  case work.

Both want: **run K clean single-congruence chains and recombine coherently at
the image plane.**

### What's missing mechanically

1. **The chain carrier is scalar-only.** `propagate_traced_carrier_chain`
   reduces every hand-off to `R = float(R_carrier)` (`carrier.py:703`, `:933`),
   so a *tilted* carrier `W = S(r) + (L x + M y)` is not expressible chain-level
   — even though the ELEMENT already accepts an arbitrary `ndarray` `W(x,y)`
   via `carrier=` (`_lens_traced.py:2156-2157`). A +-46 mrad order therefore has
   no way to be carried as its own congruence through the chain.
2. **No per-order/segmented route exists from the chain.** `segmented`,
   `multivalued`, `per_order` and `order_split` appear **zero** times in
   `propagators/carrier.py` (grep, v5.31). `apply_real_lens_traced_segmented`
   exists but is element-level only.
3. **`max_segments: int = 32`** (`_lens_traced.py:5428`) is *exactly* saturated
   by an 8x4 fan, leaving no headroom for zero-order leak or a stray order —
   and its multi-segment path routes through `_multi`, i.e. back into the
   contract in the bullet above.

### Requested

Either of these unblocks the study; the first is smaller, the second is what a
consumer actually wants to call:

* **(a) Tilted-carrier hand-offs** — let the chain carry `(R, L, M)` (sphere +
  linear tilt) instead of scalar `R`. Minimal, and makes per-order runs
  first-class since each order becomes a clean congruence.
* **(b) A per-congruence orchestrator** — e.g.
  `propagate_traced_carrier_chain_multi(fields_or_orders, groups, ..., recombine='coherent')`
  that runs each congruence through the shipped-default chain and sums complex
  amplitudes on a common image grid. Serves per-order and per-emitter with one
  implementation.

**Preferred variant for 121:** decenter each order to on-axis for its own chain
run and re-offset at the image plane — that keeps the +-46 mrad out of the
residual entirely, so each run sits inside the validated envelope rather than
2.3x outside it (see P3). If (b) does the decentre/re-offset internally, the
consumer never has to think about it.

**Acceptance:** the 32-order 121 fan reconstructs with per-frame power matching
the Dammann design uniformity (design 2.78%/frame, uniformity ~0.996) instead
of the 0.47 +/- 0.51%/frame scramble measured when the fan was pushed through
the chain multiplexed at v5.28.

---

## P2 — Let the full design be expressed as ONE object

`DGRATING` surfaces are imported as **flat optical surfaces** — the diffractive
behaviour is dropped (`io/prescriptions_zemax.py:829`). Consequence: the 121
prescription that the chain sees has never contained the DOE, and the consumer
must hand-build the grating, hand-split the chain at the DOE plane, and
hand-fold the 51.539 mm DOE gap into a neighbouring group's gap. That manual
fold is exactly the kind of bookkeeping that produced a wrong answer once
already in this study.

**Requested:** import `DGRATING` parameters (period, orders, and the existing
`PARM 1` lines/um already parsed at `:474`) into the prescription, and let the
chain's `groups` list accept a DOE entry between refractive groups. The
*propagation* can still be per-order under P1; this is about the design being
expressible and the gaps being bookkept by the library.

Related, already fixed and worth a consumer note: `bba1bc4` closed a
`create_periodic_phase_mask` cell-lookup bug (`clip` where a modulo was needed)
that put a full 0<->pi flip on 2816/65536 mask pixels in the measured case,
leaking 11.3% of power off the order lattice with 2.95% into nominally
forbidden even orders. **Any DOE fan built through that helper before v5.30
should be re-measured before its non-uniformity is attributed to physics.**

---

## P3 — Refuse (or shout about) multi-congruence input

The highest-value guard here, because the failure mode is a *plausible-looking
wrong answer*: at v5.28 the 32-order fan went through the chain and produced a
populated, credible-looking frame lattice whose per-frame power was scrambled
(0.47 +/- 0.51% vs a design 2.78%). Nothing raised, nothing warned.

The library already has the threshold — residual transverse angular spread
above **~0.02 rad** is documented as invalidating the carrier-referenced traced
correction (`_lens_traced.py:1032-1038`). A +-46 mrad fan is ~2.3x outside it on
a single order and worse on the fan rms.

**Requested:** have `propagate_traced_carrier_chain` measure the input's
angular spread / multi-valuedness at entry and, above the documented envelope,
either raise or emit a `RuntimeWarning` that names the multi-congruence route
(P1). Note the measurement caveat already recorded in the wavefront-aware
audit: a wrapped nearest-neighbour gradient estimator **under-reports** when the
content aliases, so the detector should not be built on the aliased gradient
alone. `apply_real_lens_universal` already does multi-valuedness routing
(`fga.py:2859-2878`) — the chain could reuse that measurement.

---

## P4 — Close the validation envelope where production actually runs

* **dx-flatness is published only over N = 1024-4096** (dx0 2.0 -> 0.5 um). The
  original F-B evidence matrix's *worst* row — dx0 = 0.25 um / N = 8192, which
  read EE6 46.5% pre-flip — has never been re-published under the shipped
  defaults. Closing that one row would make the convergence claim airtight
  against its own original counter-evidence.
* **Nothing above N=4096 has been re-validated post-flip at all**, while the
  pre-flip N=28672 runs are what produced both the >100% energy reading (F-A)
  and the divergence. If large-N is intended to be supported, it needs a row;
  if it is not, saying so explicitly would be just as useful — the consumer
  lesson from this study is that grid size stopped being the accuracy lever, and
  that is worth stating where people will read it.
* **No CI gate asserts design-121 dx-flatness** (the `.zmx` can't ship). The
  existing `self_check='dx'` gate runs on one synthetic N=512 singlet with a
  single sqrt(2) step at 5% tolerance on window-power/peak/r50 — it would catch
  a gross regression but, by my estimate, not a subtle one, and it does not
  cover EE3/EE6/FWHM. A synthetic multi-group stand-in with the 121's NA
  progression would close this without shipping the prescription.

---

## P5 — Guard rails on the two silent-degradation paths

* **`na_exact_threshold = 0.15` vs design 121's exit NA 0.152** — a **1.3%
  margin** (`carrier.py:2629`). `final_leg='auto'` routes correctly today, but
  one beam-size change drops this design silently onto the paraxial readout,
  which is ~200 rad wrong at this NA. Requested: warn when the exit NA lands
  within ~20% of the threshold, so the near-miss is visible. (Consumer-side
  mitigation is `final_leg='exact'` explicitly, which the 121 runners should
  do regardless — but the trap is set for the next design.)
* **RAM-capped readout** currently emits a `RuntimeWarning` and continues with
  a metric computed on a degraded grid, correctly labelled
  "RESOLUTION-LIMITED (non-converged)". For unattended/batch production runs
  that warning is easy to lose. Requested: an `on_ram_cap='error'` option so a
  production run fails loudly rather than reporting a degraded number.
* **`rs_fine` clamp degenerate corner** — when the memory/Nyquist-capped
  `dx_fine` is coarser than `ray_subsample * cur_dx`, the rescale clamps to
  `rs_fine = 1` and the F-C pitch-preservation contract silently stops holding
  (measured 5.25x mismatch at the N=28672 / `n_fine_cap`=16384 121 condition).
  Warn-only today; an opt-in strict mode would be better for production.

---

## P6 — Doc hygiene (cheap, and it actively misleads today)

`AUDIT_WAVEFRONT_AWARE_RAY_LAUNCH_2026_07_23.md` still carries **"F-B
(dx-scaling audit, still open)"** at `:89` and "Left open" at `:191`. F-B was
in fact root-caused and closed by the frozen-amplitude work
(`AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` §2-§4, §6.7) plus the
`0a743a6` upsample-lattice fix — one day *after* that audit was written. It was
never closed under the label "F-B", and `AUDIT_TRACED_CHAIN_DX_SCALING_2026_07_22.md`
was never amended, so both docs still read as open. Its §"Validity map" also
still says re-attributing the DOE fan scramble "needs the F-B fix first" — that
prerequisite is now met, and the re-attribution lands on P1/P2 instead.

Also stale, same class: `Migration-Guide.md:770-775` says the P5 return-contract
transition "remains scheduled for v5.32" (it executed in v5.30);
`carrier.py:2856-2861` calls `remap_sampling='full'` "opt-in only for
byte-compatibility" (it is a chain default since `a9dc454`); and
`validation/repro_traced_carrier_121/focus_scan_121.py`'s footer prints the
pre-S12 acceptance (3.550 / 88.4 / 99.3) as the shipping-default line.

---

## P7 — Known-and-accepted, listed so it isn't re-discovered

* **Inter-group transport is still paraxial** (Sziklas-Siegman). Under
  `'sphere'` the (S - parabola) term rides inside the transported envelope
  (~7 rad at r=w on the 121 final gap); measured end-to-end cost <= 0.2 EE3
  points, core agreement 0.019 rad rms vs an exact-ASM prototype. Fine for 121
  — but there is **no high-NA-gap guard**, so the next design finds the edge
  the hard way.
* `window_factor` is consumed twice on the exact path and compounds below
  wf ~ 3 (at wf=2, 8.6 EE3 points lost). Keeping the 121 acceptance's wf=4.0 is
  fine; a note at the call site would save the next consumer the bisection.

---

## Summary

| # | Item | Blocks | Size |
|---|------|--------|------|
| P1 | Per-congruence chain (tilted carrier and/or multi orchestrator) | **the full 121 config: DOE fan AND emitter array** | large |
| P2 | DGRATING import + DOE as a chain group | manual, error-prone bookkeeping | medium |
| P3 | Multi-congruence refusal/warning at chain entry | silent wrong answers | small |
| P4 | dx-flat row at N=8192 + a 121-class CI gate | confidence at production pitch | small-medium |
| P5 | NA-threshold proximity warning, `on_ram_cap='error'`, strict `rs_fine` | silent degradation in batch runs | small |
| P6 | Stale F-B / migration / footer text | actively misleads readers today | trivial |

P1 and P3 together are what turn "design 121 runs" into "design 121 runs
*smoothly, with the DOE, and tells you when it can't*". P3 is small and would
have saved this study a full experiment cycle.
