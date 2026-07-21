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
**Fix acceptance:** per-group rms < 0.1 rad on ALL 8 groups at the table's conditions (one command: `python traced_group_oracle.py`), then end-to-end: `carrier_chain_121.py` must produce EE6 >= 99% at the MSoP with a ~2.9 um spot (references: Zemax 2.736 um; stigmatic chain 2.97 um at the plane).

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
