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
