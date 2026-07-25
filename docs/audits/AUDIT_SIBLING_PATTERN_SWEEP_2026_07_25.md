# Audit — sibling-pattern sweep of the traced-campaign bug classes (2026-07-25)

After the traced-carrier campaign closed (`AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24`,
commits through `4b514a4`), its bug classes were swept across the whole library by two
Opus subagents with disjoint territories (A: raytrace/, analysis/, propagators/ except
carrier.py; B: carrier.py, elements/, validation/).  Discipline: measure-first (every
CONFIRMED has a repro with numbers, prediction-vs-measurement where possible),
bit-identical fixes wherever the old path was correct, a regression pin per fix,
explicit CLEAN verdicts.  **16 confirmed fixes landed** (commits `17c0ad7` sweep A,
`fd78ad2` sweep B; 77 new pins; combined-tree gates 180 pins green + the design-121
acceptance unchanged at best-focus FWHM 3.550 µm / EE3 88.2 / EE6 99.3, on-axis).
The full per-site tables with repro numbers are in the two commit messages and the
S9/S10 test files' docstrings.  This document records what is NOT in the commits:
the measured REPORT-ONLY findings awaiting deliberate decisions, and the
verified-clean map.

## 1. Report-only findings (measured; each needs a deliberate decision)

| site | finding (measured) | why not fixed now |
|---|---|---|
| `raytrace/seidel.py:122` | A FLAT fold mirror skips the parity/index-sign flip (`elif surf.is_mirror and np.isfinite(R)` gates R-independent bookkeeping): powered mirror EFL +0.5 m stays +0.5 after a flat fold (curved fold correctly flips to −0.5); every Seidel S1–S5 downstream of a flat fold uses the wrong n. | Substantive physics change to a heavily-pinned module; needs its own gated pass. |
| `raytrace/_conic_core.py:183/208` | Out-of-domain conic sag clamps to a FINITE 0.0 (JAX gradient-safety convention) while the NumPy twin returns NaN: at h beyond the oblate limit, `conic_sag`=+0.0, derivs ~6e12, `conic_sag(nan)`=0.0 — a jax-traced Newton can keep a ray alive at a phantom vertex. | The zeroing is a documented JAX design convention; changing it is a cross-backend design decision. |
| `raytrace/intersection.py:280` | NaN-position ray survives the NumPy aperture gate as `RAY_OK` (`h_sq > sd²` → False for NaN); JAX twin has the safe polarity.  Also the only kill site missing the `== RAY_OK` first-failure-wins guard its own comment promises. | One-line flip but could move pinned vignetting counts; do with a survey of the pins. |
| `raytrace/from_field.py:606` | Ray placement is edge-anchored (`linspace(0, N-1, n)`), not cell-centred as documented: `n_rays=1`/`4` RAISE on a valid centred Gaussian ("no pixels survived"); 9→1 ray, 25→6. | Cell-centred placement changes results at every n_rays (not bit-compatible). |
| `analysis/ao.py:1227` | `noise_sigma_pixels` is applied in pre-rescale slope units (missing `/slope_scale`): measured effect 3.7e-5 relative — the knob is effectively INERT. | Fixing changes the meaning of a published parameter; decide desired semantics first. |
| `analysis/detector.py:162-185` | Explicit `n_pixels` bins by `Ny/n_pixels` field samples but labels the axis with `pixel_pitch` spacing — a factor `(N·dx_field)/(n_pixels·pixel_pitch)` inconsistency (2× in the probe), unvalidated; `detector_pixels_per_lenslet` is dead. | Inspection-grade repro only (no photon-scale end-to-end); wants a clean repro then a contract decision. |
| `carrier.py` `rs_fine` clamp | At N=28672/nfc=16384 the F-C pitch-preserving rescale rounds to 0→1, so the retrace ray pitch is 5.25× the chain's (contract gap, forced by the grid). | Documented; a real fix needs a finer retrace grid policy (memory trade). |
| `_lens_traced.py` `smooth_sigma_px=4` | REFUTED as the F-B driver (unreachable from the chain; core tilt dx-invariant to 6 digits; physical-σ variant no better multi-mode). Pixel-unit dependence is real but consequence-free on reachable paths. | Nothing to fix; recorded so it is not re-chased. |
| `propagators/mft.py` `resample_field` | `N_out=None` with `dx_out ≫ N·dx_in` computes round()→0 and silently returns a (0,0) array; `order` unvalidated.  (Centering itself verified CLEAN to 5e-15 across all parities.) | Guard is trivial but was out of both agents' final scopes; small follow-up. |
| `analysis/` minor flags | `plotting.py:1443` (N−1)/2-vs-N/2 half-pixel anchor; `aberration.py:562` first-gap dedup threshold on non-uniform axes; `through_focus.py:741` cutoff comment/code mismatch + unguarded `nanmax`; `opd.py:535` row Ny//2 labelled Nx/2; `ao.py:448` n_lenslets=1 → 0/0; `ray_fan.py:249` inf·tan(0)=NaN telecentric fan; `from_field.py:496/691` N//2-vs-N/2 and edge one-sided diffs; `ray_fan.py:714`/`layout.py:60` degenerate-input crashes; `fga.py:2136` unweighted meridional caustic metric. | Individually small; batch into a hygiene pass. |

## 2. Verified-clean map (measured, incl. counterfactual sensitivity checks)

- **GBD node upsample** (`gbd.py:2772`) — the reference lattice bug's direct sibling
  is CORRECT (`arange(N)/step` vs the actual `arange(0,N,step)` lattice): ≤5.7e-14 px
  ramp error and exact node reproduction at divisor AND non-divisor strides; the
  counterfactual `ii·Ns/N` shifts +17.2 px on the same probe.
- **FGA fractional-index interp** (`fga.py:1096`) — exact vs the actual coarse axis
  (`np.interp` construction), 0–6.2e-14 px; one flagged rim caveat (last short cell,
  masked region).  **FGA Gabor normalisation** carries its cell area correctly.
- **`resample_field` centering** (all parities + non-integer scales), **GBD windowed
  reconstruct** (shared anchor; physical window units), **Bluestein/CZT**, **sas/rs
  padding**, **fresnel.py**, **subaperture.py**, **memory.py**.
- **carrier.py inf-handling** (`propagate_carrier_referenced`, `_carrier_step_fast`,
  `_radial/_build_carrier_phase`, `_sphere_parab_conversion`, `_rereference`,
  Möbius `R_out(inf)=A/C` exact), **Newton iters/tol** (insensitive to 2.1e-15 across
  iters 3–48 and tol ×1e-2–×1e4), **`sag_chunk_rows` in `apply_real_lens`**
  (bit-identical bands), **`_fourier_upsample_crop` in-contract branches** (Parseval
  1.00000000, both directions), **segmented partitions** (partition of unity holds).

## 3. Cross-cutting lessons added to the house discipline

- `isinf(x)` fast paths must check EVERY quantity that can carry the discarded
  physics (the flat-path/cylinder bug: x-radius flat ≠ surface flat).
- "Byte-identical" performance paths (banding/chunking) must inherit EVERY accuracy
  option of the path they shadow (the `sag_chunk_rows` order-1 downgrade silently
  disabled R7 for large-N carrier calls).
- Shape asserts do not validate PITCH: a resampler can return the right shape from
  the wrong window (both F-A and the new n_crop>input branch).
- Harness knobs must ERROR on unrecognised values (two false campaign results came
  from silent fall-throughs).
