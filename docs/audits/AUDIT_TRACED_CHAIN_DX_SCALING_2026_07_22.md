# Traced carrier chain: grid-pitch (dx) scaling audit -- 2026-07-22

Addendum to `AUDIT_TRACED_CARRIER_CHAIN_2026_07_21.md`. Applies to the
v5.28.0 R8/R9 stack (`propagate_traced_carrier_chain` +
`carrier_referenced_exact_focus_readout`). All findings were produced on the
design-121 8-group chain (Tx02-MSOP16, lam = 1.31 um, 4 um-waist launch,
exact final leg at NA ~ 0.46) by sweeping the launch grid at production
scale (N = 28672 requested by the design study).

> **STATUS (amended 2026-07-27, v5.31.0): ALL FOUR FINDINGS CLOSED.**
>
> * **F-A / F-C / F-D** — fixed by `7189dfd`; `_fourier_upsample_crop` gained a
>   real band-limited downsample branch (shape invariant now holds in both
>   directions).
> * **F-B** — root-caused and closed, but never under the label "F-B", which is
>   why this document read as open for five days. It was NOT the parabola
>   hand-off: see `AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24.md` §2-§4 (frozen
>   intra-group amplitude + `preserve_input_phase=True` analytic-pair phase
>   corruption, both reproduced in ONE group with no chain machinery) and §6.7
>   (dx-flat by configuration), plus `0a743a6` (coarse->fine upsample lattice
>   bug). Shipped as the v5.29 chain default flip (`455be4a`, `a9dc454`).
>   Design-121 acceptance is now EE6 99.6 / EE3 88.8 / FWHM 3.450 um at the SAME
>   N=2048 / dx0=1.0 um where this audit measured 69.7%.
> * Caveats carried forward into
>   `ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27.md` §P4: dx-flatness is
>   published only over N=1024-4096, so the worst row of the matrix below
>   (dx0=0.25 um / N=8192, EE6 46.5% pre-flip) has never been re-published under
>   the shipped defaults.
> * The "Validity map" section below is superseded on one point: re-attributing
>   the DOE fan scramble no longer "needs the F-B fix first" (that prerequisite
>   is met). It lands on the multi-congruence gap instead — same roadmap, §P1.

Summary: one hard bug (F-A, energy non-conservation at chain N > 16384), one
first-order accuracy defect (F-B, the chain's absolute EE numbers are NOT
dx-converged -- no plateau anywhere in the tested range), and two smaller
robustness items (F-C, F-D). Ray/OPL sampling density was explicitly CLEARED
(F-0).

## Evidence matrix

Design-121 chain, `ray_subsample` chosen per row, exact final leg,
readout 1024 x 0.05 um. "window" = total power inside the readout window /
launch power (must be <= ~0.96 after stop losses; > 1.0 is impossible).

| N     | dx0 (launch) | ray pitch | FWHM    | EE3   | EE6    | EE12   | window  |
|-------|--------------|-----------|---------|-------|--------|--------|---------|
| 1024  | 2.0 um       | 4 um      | 3.75 um | 65.8% | 82.8%  | 84.7%  | 87.6%   |
| 2048  | 1.0 um (ref) | 4 um      | 4.05 um | 52.4% | 69.7%  | 73.5%  | 78.7%   |
| 2048  | 1.0 um, rs=8 | 8 um      | 4.05 um | 52.4% | 69.7%  | 73.5%  | 78.7%   |
| 4096  | 0.5 um, rs=8 | 4 um      | 4.35 um | 39.8% | 54.6%  | 67.2%  | 77.5%   |
| 8192  | 0.25 um      | 1 um      | 8.85 um | 19.1% | 46.5%  | 59.2%  | 75.2%   |
| 28672 | 0.071 um     | 0.29 um   | 7.05 um | 50.0% | 102.3% | 128.2% | 130.8%  |

The 2048 row is the config the 2026-07-21 audit's R9 table was produced at
(4.05 um / 52.1 / 69.7 / 73.6 -- reproduced here to the decimal).

## F-0 (cleared): ray/OPL sampling density is NOT the driver

`rs=8` at N = 2048 (2x sparser OPL sampling, 8 um physical pitch) is
byte-identical to the `rs=4` reference in every metric, and the N = 4096 row
holds the PHYSICAL ray pitch fixed at the reference 4 um yet still degrades.
Per-stage envelope power is constant to 7 digits at N <= 2048. The
degradation tracks the per-stage grid pitch alone.

## F-A (P0): n_fine cap makes the exact final leg pitch-inconsistent at chain N > 16384

Consequence: energy non-conservation plus core blur.

`_fine_trace_group_exit` (propagators/carrier.py):

```python
n_fine = int(2 ** int(np.ceil(np.log2(max(win / dx_fine, n_crop)))))
n_fine = int(min(n_fine, n_fine_cap))          # cap = 16384
dx_fine = win / n_fine                          # pitch claimed downstream
env_f = _fourier_upsample_crop(env, n_crop, n_fine)
```

but `_fourier_upsample_crop` short-circuits when asked to DOWNSAMPLE:

```python
if n_fine <= n_crop:
    return ec                                   # raw n_crop x n_crop crop!
```

When the chain grid N > 16384 and the window spans the grid (`n_crop` = N >
cap), the helper returns a 28672^2 array while the caller proceeds with
`dx_fine = win/16384` -- every downstream consumer (carrier reconstruct, the
traced group's stop geometry and OPL fit, the exact-sphere readout and its
power normalisation) runs with the pixel pitch wrong by `n_crop/n_fine`
(1.75x at N = 28672). Observed: readout window carries 130.8% of the launch
power (EE6 = 102.3%, EE12 = 128.2%) plus core blur. Trigger is exactly and
only `n_crop > n_fine_cap`: the N = 8192 row (n_crop < cap) shows no
inflation.

Fix (mechanical): give `_fourier_upsample_crop` a true band-limited
downsample branch -- k-space TRUNCATION to `n_fine` with the same
value-preserving scale `(n_fine/n_crop)**2` -- so its documented contract
("returns the envelope on the n_fine grid spanning the same window") holds
in both directions; assert `env_f.shape[-1] == n_fine` at the call site.
The envelope is smooth by construction (carrier divided out), so truncation
is the correct operation.

## F-B (P1): absolute focal metrics are not dx-converged (no plateau; onset at stage 1)

At fixed physical ray pitch and conserved energy, EE6 falls 82.8 -> 69.7 ->
54.6 -> 46.5% as the launch pitch halves 2 -> 1 -> 0.5 -> 0.25 um (the
per-stage co-moving pitch scales with it, 48 -> 24 -> 12 -> 6 um through the
front groups). There is no converged region: the 2026-07-21 audit's
end-to-end EE6 = 69.7% is a POINT ON A SCALE-DEPENDENT CURVE, not a model
prediction. ~15 EE6 points per octave of dx around production settings.

Localisation so far:

* Onset is the FIRST traced group and it compounds: at dx0 = 0.25 um the
  stage-exit envelope radius is already +3.4% at S3-S4 and +11% by S16-S17
  vs the reference run; per-stage power stays conserved (pure phase/blur
  pollution, invisible to the stage w/power stats at milder dx).
* Not ray density (F-0), not the final leg (final-leg dx_fine is capped to
  ~1.5 um for this design at every N, i.e. near-identical across rows).
* Suspects (unresolved, in likelihood order): pixel-unit tilt-field
  smoothing (`smooth_sigma_px = 4` -- physical smoothing length shrinks
  linearly with dx); entrance-tilt finite differencing vs the per-pixel
  carrier phase step (12 rad/px at the reference pitch vs 0.86 rad/px at
  dx0 = 0.071 um -- the estimator operates in completely different regimes);
  Newton/Cheb OPL-fit conditioning as the fit point count grows (N/rs)^2.

Recommended library action: add a dx-convergence gate to the chain test
suite (same chain at dx0 and dx0/2 must agree in EE6 to a tolerance);
instrument the per-group phase residual vs the meridional oracle
(`validation/repro_traced_carrier_121/traced_group_oracle.py`) as a function
of grid pitch to find which leg diverges; then fix the pixel-unit scaling.
Until then, treat traced-chain ABSOLUTE EE/FWHM as carrying the ~15-pt/octave
dx systematic, and only compare LIKE-dx runs.

## F-C (P2): fine final leg inherits chain ray_subsample in PIXEL units

`_fine_trace_group_exit` passes the chain-level `ray_subsample` straight to
the traced re-run on the n_fine grid. The physical ray pitch on that grid is
then `rs * dx_fine`, unrelated to the chain's. At `rs = 1` the Cheb2D
evaluator attempted a (28, 20151, 20151) float64 design matrix = 84.7 GiB ->
MemoryError. Fix: rescale on entry (`rs_fine = max(1, round(rs * cur_dx /
dx_fine))`) to preserve the CHAIN's physical ray pitch, and cap the Cheb fit
point count independently.

## F-D (P2): n_fine cap under-samples the exit sphere even when it "works"

For design-121-class exit NA (0.46), the target pitch is lam/(3 NA) =
0.95 um but the cap delivers win/16384 ~ 1.50 um at EVERY chain N (window =
7w with w ~ 3.5 mm). That is below even the 2-point Nyquist margin for the
beam-edge sphere frequency (1.42 um). The runs "work" because `bandlimit=
True` masks the aliased corner, at the cost of silently discarding the
outer-NA content. Options: raise `n_fine_cap` when RAM permits (16384^2
complex128 = 4 GiB is conservative on the boxes this runs on); shrink the
default `window_factor` (7w spends most of n_fine on empty guard band -- 4w
suffices for the envelope); or emit a warning when `dx_fine` lands above
lam/(2 NA).

## Validity map for existing design-121 results

* v5.28 traced Run A/B at N = 2048/4096 (2026-07-22 study): internally
  consistent, energy-conserving, and cross-validated per-group -- but their
  ABSOLUTE EE numbers inherit the F-B systematic at their respective launch
  pitches (1.0 um / 0.4 um). Cross-model comparisons at like dx remain valid.
* The DOE run's fan-uniformity scramble (0.47 +/- 0.51%/frame) can no longer
  be attributed purely to the single-sphere-carrier model limitation: its
  0.4 um launch pitch sits partway down the F-B curve, so part of the
  scramble is numerical. Re-attribution needs the F-B fix first.
* Production-size grids (N = 28672) are ONLY valid pitch-preserving (pin
  dx0 at the reference pitch; N buys guard band). Extent-preserving
  refinement is invalid: F-B degradation plus, beyond N = 16384, the F-A
  energy bug.

## Repro

Design-121 chain runner with env knobs
(`validation/repro_traced_carrier_121/repro_dx_scaling.py`):
`RN` = chain N, `DX0` = launch pitch (omit for extent-preserving
`1.0e-6 * 2048/RN`), `RS` = ray_subsample. The table above is
`(RN, DX0, RS)` = (1024, 2e-6, 2), (2048, 1e-6, 4), (2048, 1e-6, 8),
(4096, .5e-6, 8), (8192, .25e-6, 4), (28672, extent-preserving, 4).
