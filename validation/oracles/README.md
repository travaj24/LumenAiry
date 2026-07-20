# Independent real-lens oracles (lumenairy-free)

Standalone cross-check tools for the real-lens propagator family. **Not** part
of pytest collection (see `conftest.py`) -- they are CLI scripts run against a
JSON job, independent of lumenairy, Zemax, and any FFT grid.

## `debye_oracle_v3.py` (P0 / niche N0.1)

Congruence-fixed Debye/Huygens oracle: exact meridional raytrace through a
rotationally-symmetric conic + even-aspheric surface list, launched along a
signed input congruence `R_in`, producing **both**

- **(a) a geometric transverse-aberration spot** (ray-density r2m / EE), and
- **(b) a congruence-fixed ring-Huygens diffraction integral** to the image
  plane.

It fixes the two bugs that made `debye_oracle2.py` unusable for a non-collimated
congruence:

1. **entrance eikonal** `W_in = h^2/(2 R_in)` restored in the phase (the
   hammer-H6 class omission), and
2. **exit ring measure** with the energy-conserving Jacobian
   `A_env(h) * sqrt(h * y_exit * dy_exit/dh)` instead of the entrance measure
   `h dh` (equal only in the collimated limit `y_exit ~ h`).

Run:

```
python debye_oracle_v3.py <job.json> [z_img_offset_mm]
```

The geometric spot (a) is the ray-density metric; near a strong reconvergence
caustic it OVER-estimates the true wave spot by up to ~2x, so **(b) is the
diffraction-faithful ground truth** for wave-model comparison. See
`docs/audit_real_lens_displaced_2026_07_19.md` (P2 section) for the validation
matrix and the measured wave/geometric split.

The `evaluate(job, dz_off_mm)` function is importable for in-process use by
`tests/unit/test_niche_p2_*.py`.

## `geom_spot_decenter_oracle.py` (P3 / niche N2)

Lumenairy-free 3-D geometric spot-diagram oracle for **decentered / tilted**
conic + even-aspheric elements -- the independent ground truth for the
decenter/tilt-induced **centroid shift** and **comatic spot** that a
rotationally-symmetric oracle structurally cannot represent.

* `geom_spot(job, ...)` -- traces a 2-D Gaussian-apodized ray bundle through a
  surface list where each surface may carry `decenter_mm=[dx,dy]` and
  small-angle `tilt_mrad=[tx,ty]` (the **field-frame linear-ramp** tilt
  convention the lumenairy displaced model uses), returning the geometric spot
  centroid (um), RMS radius, and EE(about-centroid).
* `rigid_tilt_centroid(job, surf_index, ...)` -- an INDEPENDENT **rigid-body
  full-rotation** tilt reference (the Zemax / optomechanical convention,
  `R = Rx(tx) @ Ry(ty)` about the vertex).  Its centroid magnitude cross-checks
  the linear-ramp centroid to leading order (they agree to <0.5% at moderate
  tilt; opposite sign by the differing "positive tilt" definition) -- so the
  linear ramp is a validated tilt model for the centroid, not a coincidence.

Near a strong reconvergence caustic the geometric ray-density spot
over-estimates the true wave spot (P2 note); for the decenter/tilt centroid and
the coma flare DIRECTION it is geometric-exact.  Imported in-process by
`tests/unit/test_niche_p3_*.py`.

## `caustic_fold_truth.py` (P5 / niche N0.3)

Multi-valued **FOLD-CAUSTIC** ground-truth oracle -- the reference the FGA
caustic-specialty validation (niche N6a) compares against.  Builds the wave field
at a THROUGH-FOCUS plane of a strongly-aberrated singlet where the geometric ray
map genuinely **folds** (multi-valued: two ray branches reach the same transverse
radius, forming a bright caustic ring with Airy-type fringes inside and an
exponential tail outside) by a **dense, DIRECT Rayleigh-Sommerfeld / Huygens ring
integral** of the exact exit field -- **no stationary phase, no ray-branch
assumptions** enter the propagation, so the caustic interference emerges on its
own.

* `evaluate(job, save_prefix=None)` -- traces a dense collimated fan through the
  conic/aspheric surface list, builds the energy-conserving exit field
  `E_exit(y) = A_env(h) sqrt(h/(y J)) exp(i k opl)`, then the direct RS ring
  integral to the observation plane; returns the grid-free **radial** metrics
  (`huy_*`), the 2-D **axis-centred** grid metrics on the propagators' grid
  (`grid_*`), and the self-verification fields.
* Self-verification (all lens-model-independent): **grid convergence**
  (`convergence_r2m_frac` -- doubling the fan/rho sampling changes windowed r2m
  by < 0.5%; measured 1.3e-5), **energy closure** (`energy_closure` -- full-plane
  propagated power / launched power; measured 0.999), and **multi-valuedness**
  (`n_branches_at_caustic` >= 2, `fold_radius_um` the caustic ring).
* Metric convention: for a rotationally-symmetric ring caustic the peak sits on
  a bright fringe OFF-AXIS, so the in-library **peak-centred** `_wave_metrics`
  inflates r2m ~12%; `win_metrics_2d(..., center='axis')` (the default here)
  measures from the optical axis and reproduces the grid-free radial metric to
  <1%.  STEP B compares axis-centred metrics for all models.

Installed case + reference: `caustic_fold_case.json` (plano-convex convex-first
f~5.2 mm, f/3.65, N_F~82, lambda 1.31 um; observation plane 1.0 LSA short of the
marginal focus -> a single fold ring at 14.26 um) and `caustic_fold_ref.npz`
(radial + 2-D reference field + grid, regenerable via
`python caustic_fold_truth.py caustic_fold_case.json --save caustic_fold_ref`).
Ground-truth axis-centred metrics: **r2m 11.22, EE50 10.05, EE80 13.50 um**.  See
`docs/audit_real_lens_hammer_2026_07_19.md` (N6a section) for the FGA vs GBD vs
traced comparison against this reference.

## ZOS-API oracle: POP + Huygens PSF (`scratchpad/zos_oracle.py`, P8 / N0.2)

The Zemax OpticStudio ZOS-API oracle (dispersionless model glass, one connection
at a time) lives in the session scratchpad, NOT under version control (it drives
a licensed external tool).  It is a JSON-job CLI run with the py3.13 zospy venv:

```text
D:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics\OPDPy_Lumenairy_Crosscheck\.venv-zemax\Scripts\python.exe \
    <scratchpad>\zos_oracle.py <job.json> <out_prefix>
```

The job may carry a `"pop"` block (Gaussian-pilot Physical Optics Propagation --
the historical mode) and/or, since P8, a `"huygens_psf"` block:

```json
{ "wavelength_um": 1.31, "aperture_mm": 12.0,
  "object_thickness_mm": 0.0,           // omit/<=0 => infinite conjugate (point src)
  "surfaces": [ {"radius_mm": 51.68, "thickness_mm": 5.0, "index": 1.5168,
                 "semi_diameter_mm": 6.0},
                {"radius_mm": -51.68, "thickness_mm": 49.162, "index": "air"} ],
  "huygens_psf": {"pupil_sampling": "512x512", "image_sampling": "256x256",
                  "image_delta_um": 1.0, "field": 1, "wavelength_index": 1,
                  "normalize": true} }
```

The Huygens-PSF mode (uniform, pupil-limited point source at finite/infinite
conjugate) returns, under `metrics["huygens"]`, the Strehl ratio, the PSF pixel
pitch (um), the centroid, r2m + EE50/80/95 (about centroid AND about peak, um),
and the radial first-dark-ring radius (`first_zero_um`); the PSF grid is saved to
`<out_prefix>_huygens.npz`.

**Validated (N0.2 acceptance):**
* *Unaberrated (Airy).*  A slow equiconvex singlet (f~100 mm, EPD 4 mm, f/25) at
  infinite conjugate returns Strehl `1.000` and `first_zero_um = 40.32` vs the
  analytic Airy first dark ring `1.22*lambda*F/# = 40.15 um` (**0.4%**).
* *Aberrated vs `debye_oracle_v3` (two independent Huygens integrals).*  The f/2
  equiconvex singlet stopped to f/4 (aperture 12 mm), UNIFORM pupil, at the same
  paraxial-focus plane and a MATCHED 110-um metric window, agrees with
  `debye_oracle_v3` (large-`w0` uniform pupil) to **EE50 1.9% / EE80 0.72% /
  EE95 0.17%**.  Two caveats, both measured and documented:
  (1) the EE-about-centroid metric renormalizes to captured energy, so the metric
  window MUST match between the two tools -- an un-matched window (Zemax +-64 um
  vs the ~105-um spot) spuriously halved the Zemax EE (the P8 "0.6x" red herring);
  (2) a heavily-aberrated PSF needs both an adequate image window (>= the EE95
  radius) AND high ZOS pupil sampling (the full-f/2 aperture-24 mm case is still
  pupil-sampling-limited at 512x512: EE80 climbs 203->306->376 um across
  256/512/1024 pupil samples, converging toward the Debye 353 um whose ring-Huygens
  J0 far-field kernel itself carries a ~0.7-rad edge-phase error at f/2, so f/4 is
  the clean cross-check regime, not full f/2).
