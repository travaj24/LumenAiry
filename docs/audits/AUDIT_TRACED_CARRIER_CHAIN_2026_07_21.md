# Audit — traced-group fidelity on strongly-diverging input + the carrier-referenced chain (design-121 validation set)

**Date:** 2026-07-21 · **Library state:** v5.27.0, working tree `6e34208-dirty` (cache-dev WIP present; none of the findings touch the dirty files' code paths)
**Scope:** `apply_real_lens_traced` (carrier handling, thick-group exit wavefront), `propagate_carrier_referenced` + envelope/reconstruct hand-offs, chain composition.
**Repro:** `validation/repro_traced_carrier_121/{traced_group_oracle.py, carrier_chain_121.py}` — both run in <2 min at N=2048, no Zemax needed. Oracle is lumenairy-free (exact meridional raytrace + eikonal).

## Context (why this audit exists)

The 121 relay (8 groups, 15 refracting surfaces, emitter NA 0.104, per-group carrier R_in from +47.9 mm to -21.1 mm on 3-6 mm beams) is the production validation set for real-surface propagation. Reference truth: Zemax POP images the 4 um-waist emitter to a 2.736 um waist; lumenairy's conjugate-stigmatic thin chain reproduces this on the same pipeline (2.97 um at the recorded plane, EE6 = 100%). Full-scale traced runs of the real chain (`carrier='auto'`, N=28672, v5.24.4 exp22 and v5.27.0 exp24) conserve power (0.9998+) but form NO focus — energy smeared over +-1.8 mm.

To isolate the failure, the 121 chain was rebuilt on the NEW carrier-referenced composition (P8-capstone style): analytic pilot legs (`propagate_carrier_referenced`), reconstruct at each group's front vertex, `apply_real_lens_traced(carrier=R_in)`, re-envelope with the ABCD R_out. Result:

- **The carrier infrastructure is excellent.** Whole chain in 27-56 s at N=2048 (vs ~2.5 h at N=28672). Carrier R tracks the ABCD q-trace to 3-6 digits at all 8 hand-offs, beam width matches the analytic relay profile at every plane, power is conserved BIT-EXACT through 7 of 8 groups, the through-focus split auto-zooms the grid onto the image (dx -> 15 nm), and the 121's real apertures finally fit inside the (34-52 mm) co-moving grids.
- **The image still does not form**: a sub-diffraction coherent spike (FWHM ~1.3 um < the 2.74 um limit — the classic speckle-peak signature) holding ~0.1% of the power, on a window-filling pedestal. With every leg analytic and lossless, the defect is isolated to the eight `apply_real_lens_traced` calls.

## Method (per-group oracle)

For each group in isolation, at its ACTUAL chain conditions (w, R_in from the physical q-trace): input `E = exp(-r^2/w^2) * exp(i k S(R_in))`, `S(R) = sign(R)(sqrt(r^2+R^2)-|R|)`, on an N=2048 grid spanning 8.4 w. Apply the traced group. Compare pointwise against the exact meridional oracle (rays from the carrier point source, exact multi-surface Snell + eikonal to the exit vertex plane; signed-t continuation for concave-front surfaces whose cap lies behind the vertex plane). Metrics over r < w, piston removed: total rms residual; the r^2 (exit-curvature) component; the r^4+ (aberration) component; high-frequency rms ("scramble", residual minus 21-px smoothing).

Oracle self-checks: the two flat windows come out EXACT (0.000 rad), the weak doublet S21-S22 near-exact (0.080 rad) — conventions (vertex planes, piston, entrance eikonal) are agreed between oracle and library on benign cases.

## Findings

### F1 (P1) — `carrier='auto'` does not engage on a clean spherical input

On S5-S7 (pure spherical carrier, R_in = +153.37 mm, w = 6 mm):

| variant | rms residual (rad) | r^4+ (rad) |
|---|---|---|
| `carrier=+153.37e-3` (explicit) | 1.072 | 0.005 |
| `carrier='auto'` | 1.635 | 0.588 |
| **no carrier (plane-wave ref)** | **1.643** | **0.588** |

`'auto'` is statistically IDENTICAL to no-carrier — the auto-fit fails to detect the carrier (fits ~inf) on exactly the input class it exists for. This is why the full-scale chain runs with `carrier='auto'` (exp22 v5.24.4, exp24 v5.27.0) behave as if H6 never landed. H6's machinery works — when handed an explicit R (r^4 drops 100x, 0.588 -> 0.005) — but the 'auto' path never hands it one.
**Fix:** make the auto-fit robustly recover a spherical carrier from a smooth diverging/converging input (e.g. fit 1/R from the unwrapped radial phase of the low-pass field, or from the Husimi mean-slope map). Acceptance: `'auto'` == explicit R on this exact repro (both tables in `traced_group_oracle.py`).

### F2 (P1) — thick groups leave a SMOOTH exit-wavefront error (curvature-dominated) even with the correct explicit carrier

Full table, default knobs (`carrier=R_in`, `ray_subsample=4`), rad over r < w:

| group | R_in (mm) | w (mm) | rms res | curvature error | r^4+ | hf |
|---|---|---|---|---|---|---|
| S3-S4 (doublet) | +47.9 | 5.0 | 0.264 | negligible | 0.013 | 0.013 |
| S5-S7 (triplet) | +153.4 | 6.0 | 1.072 | **+0.050 /m spurious defocus** (exit ~collimated) | 0.005 | 0.002 |
| S14-S15 (window) | inf | 6.4 | **0.000** | — | 0 | 0 |
| S16-S17 (window) | inf | 6.4 | **0.000** | — | 0 | 0 |
| S18-S20 (triplet) | inf | 6.4 | 1.810 | 122% of 1/R_out | 0.296 | 0.015 |
| S21-S22 (doublet) | -230.7 | 5.2 | 0.080 | 0.02% | 0.028 | 0.029 |
| S23-S24 (doublet) | -51.5 | 4.1 | 1.810 | 25% | 0.215 | 0.034 |
| S25-S27 (triplet) | -21.1 | 3.1 | 1.802 | 18% | **2.741** | 0.143 |

Signature: the error is SMOOTH (hf ~ 0 — not sampling, not interpolation noise; `ray_subsample` 16 vs 4 changes nothing), dominated by an exit-CURVATURE (defocus) term, worst for TRIPLETS and for steep input curvature, with a large genuine r^4 residual only at the steepest group. Windows are exact and the weak doublet is clean, so this is not a plane/piston convention error. Accumulated over the chain: ~6.8 rad rms -> Strehl ~ e^-46 ~ 0 -> the observed pedestal, in both the carrier chain (N=2048) and the fixed-grid production run (exp24, N=28672 — same model, same failure, 300x the cost).
**Hypothesis for the implementer:** the ENTRANCE eikonal is carrier-referenced (H6), but the INTRA-group reference between surfaces (the amp/amp(pw) screens and/or the entrance->exit map inversion) still assumes a near-collimated reference through the glass, so the error grows with (group thickness) x (wavefront curvature inside the group) — consistent with triplet >> doublet and with windows being exact. The `E_analytic_pw` plane-wave pass visible in the runtime logs is a natural suspect.
**Fix acceptance:** per-group rms < 0.1 rad on ALL 8 groups at the table's conditions (one command: `python traced_group_oracle.py`) -- **DONE (R7, commit 14d737e):** the current table reads S3-S4 0.015 / S5-S7 0.008 / windows 0.000 / S18-S20 0.004 / S21-S22 0.004 / S23-S24 0.012 / **S25-S27 0.023** (all < 0.1). The end-to-end EE6 >= 99% target is treated separately in **R9 below** (the final high-NA leg is now exact, but end-to-end is blocked by a *different*, upstream mechanism -- see R9).

### F3 (P2) — `tilt_aware_rays=True` degrades this class

Same S5-S7 case: 1.723 rad rms and 5x the curvature error of the default path. The N5 entrance-eikonal fix made tilt-aware collimated-safe, but on a steep spherical carrier it is currently worse than the default; either fix alongside F2 or guard/document.

### F4 (P3) — composition ergonomics (feature requests, all proven by the repro scripts)

1. **Chain orchestrator**: the hand-off pattern (carrier leg -> reconstruct -> element(carrier=R) -> re-envelope(R_out)) is ~30 lines of user code per chain and needs the element's own exit curvature; ship it as an API (element supplies R_out; today it must come from an external ABCD q-trace).
2. **Near-focus landing**: landing a leg within ~10 um of the carrier focus collapses the co-moving window (31 um) and clips the halo, producing spurious sub-lambda "spots". The working pattern (stop 0.5 mm short + fine Bluestein zoom to the target plane) should be packaged, or the focus-crossing bridge should accept a target-plane readout.
3. Positive finding, worth a pinned test: carrier hand-offs at all 8 groups track ABCD to 3-6 digits, power bit-exact through 7 groups, and the co-moving grids ELIMINATE the "aperture exceeds grid" truncation that affects every fixed-grid 121 production run.

## Impact / priority

F1+F2 are jointly the LAST blocker to a real-surface production model for corrected relays: with them fixed, the carrier-referenced traced chain delivers Zemax-class absolute fidelity at N=2048 in under a minute — ~300x cheaper than the fixed-grid runs, and cheaper than every currently-valid alternative. (For context, the other real-surface candidates are all closed off at 121 conditions: analytic phase screens carry ~25 rad of structural sag-projection error post-H1; GBD with `reexpand='auto'` recovers only 2.4% chain power — per-group completeness 0.90/0.99 at some groups but 0.17-0.43 at S5-S7/S23-S24/S25-S27; FGA production re-test pending H8.)

## Repro commands

```
python validation/repro_traced_carrier_121/traced_group_oracle.py            # F2 table
python validation/repro_traced_carrier_121/traced_group_oracle.py variants   # F1/F3 table (S5-S7)
python validation/repro_traced_carrier_121/carrier_chain_121.py              # end-to-end acceptance
```

All three need the 121 .zmx (path at the top of each script) and the runner's Sellmeier registrations (parsed automatically from `run_poc_119_120_v518.py`). Grids are N=2048; total runtime < 5 min CPU.

---

## R9 addendum (2026-07-22) — the EXACT high-NA final leg, and why the end-to-end 121 still does not reach EE6 >= 99%

**Shipped.** `lumenairy.carrier_referenced_exact_focus_readout` + `propagate_traced_carrier_chain(final_leg='auto'|'exact', na_exact_threshold=0.15)` + the fine-trace helper `_fine_trace_group_exit`. Tests: `tests/unit/test_niche_r9_highna_final_leg.py` (self-contained, no .zmx). Low-NA / default chain paths are byte-identical (the two R8 orchestrator-vs-manual tests still pass unchanged).

### The final-leg mechanism (validated, correct)

R7 solved the PER-GROUP wavefront (all 8 groups < 0.023 rad vs the exact ray oracle). But the **last** leg is at NA ~ 0.46 (R_out = -7.71 mm, w = 3.53 mm), and there the paraxial carrier is structurally wrong: it references the wavefront to a quadratic PARABOLA, whose r^4 deviation from the true EXACT sphere `S(R)=sign(R)(sqrt(r^2+R^2)-|R|)` reaches **~200 rad** at the beam edge. Re-enveloping the last leg with the paraxial ABCD `R_out` dumps that r^4 onto the paraxial envelope, which paraxial carrier propagation cannot focus.

Fix (Option A): reference the field to its own EXACT sphere (the two steep phases alias-cancel pointwise on the shared grid, leaving a smooth envelope), resample onto a grid that Nyquist-samples the exact sphere (`dx <= lambda/(2 NA)`), reconstruct, and propagate to the image with the EXISTING band-limited ASM Bluestein zoom (`angular_spectrum_propagate_mft`) — no paraxial magnification/curvature. **Validated on a clean synthetic NA-0.30 / 0.46 exact-sphere Gaussian: the exact leg reaches EE-in-2w0 = 99.8% (matching a fully-resolved fine-grid ground truth to <3%, FWHM ~ lambda/(2 NA)), where the paraxial carrier path gives ~1%** (test `test_r9_exact_leg_focuses_highna_sphere`).

A second, coupled fix is required for the last leg to help at all end-to-end: the last group's exit (NA 0.46) also ALIASES on the co-moving grid (dx=16.6 um, exit sphere ~33 rad/pixel; the coarse Newton/poly OPL fit then aliases high-order aberration into defocus, `_lens_traced.py:3055` warns of exactly this). So the exact final leg first **re-traces the last group on a grid that Nyquist-samples its exit** (`_fine_trace_group_exit`, dx_fine ~ 1.08 um), then does the exact readout.

### End-to-end 121 (`carrier_chain_121` conditions, N=2048 chain, exact final leg)

| leg model | EE3 | EE6 | EE12 | FWHM | best plane |
|---|---|---|---|---|---|
| BEFORE (paraxial final leg) | 5.3% | **7.3%** | 14.6% | 4.75 um | +100 um |
| AFTER (exact final leg, R9) | 52.1% | **69.7%** | 73.6% | 4.05 um | MSoP (dz=0) |

(References: Zemax POP 2.736 um; stigmatic chain 2.97 um.) The exact leg is a ~10x EE6 improvement and moves best focus onto the MSoP — but **EE6 = 99% is NOT reached (69.7%).**

### The REAL remaining blocker (upstream, NOT the final leg)

The field ENTERING the high-NA tail already carries **~1.68 rad RMS wavefront aberration** (a4·w^4 ~ +32 rad), measured at the S21-S22 / S23-S24 / S25-S27 front vertices — and it stays ~1.7 rad through the tail. This is NOT the parabola/sphere r^4 (negligible at the front's large R), NOT the paraxial gap transport, and NOT the last group's sampling:

* Exact-sphere reconstruct/envelope (replacing the parabola everywhere) changes it by < 3% (1.69 -> 1.64 rad).
* The whole tail on ONE fine grid with **fully-exact** ASM gaps (no paraxial carrier, no downsample) plateaus at the **same** EE6 ~ 69% — so the tail handling is not the limiter.
* Yet the per-group oracle shows every front group is clean (< 0.02 rad) given a clean sphere input.

So the ~1.68 rad is **accumulated by the traced-carrier CHAIN through the (low-NA, individually-clean) FRONT groups S3-S18**. Mechanism: `apply_real_lens_traced(carrier=R)` launches rays along the carrier SPHERE gradient, so a beam whose intermediate wavefront is NOT a clean sphere (the pre-correction aberration a corrected relay legitimately carries at mid-relay planes) is propagated as if it were spherical — the tail's designed cancellation of that pre-aberration is lost, and small per-group mismatches accumulate. Reaching EE6 >= 99% needs a **wavefront-aware (not sphere-referenced) ray launch through the whole chain** (`tilt_aware_rays` is the closest existing lever, but F3 showed it degrades on steep carriers and it is guarded off for explicit carriers) — a model change well beyond R9's final-leg scope.

**Verdict:** the exact high-NA final leg is the correct, necessary, and now-shipped fix for the *paraxial-focus* half of the problem (validated to the diffraction limit on clean input; +10x on the real 121). The end-to-end EE6 >= 99% target is blocked by a *separate* upstream limitation — the traced-carrier model's inability to carry a non-spherical intermediate wavefront through a corrected relay — which is the next item to close.
