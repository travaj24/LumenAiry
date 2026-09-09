# probe_pmm2d_staggered_slant

Prototype + measurements for **Phase D** of `docs/PMM_ROADMAP.md`: a NATIVE
SLANT (constant x-z / y-z coordinate shear -- tilted-axis pillars) for the PURE
staggered 2-D PMM (`lumenairy/elements/pmm/twod_staggered.py`).

Verdict document: `docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md`.

Nothing under `lumenairy/` or `tests/` is modified.  `slant_lib.py` imports the
shipped private helpers rather than copying them, and asserts on import that
`lumenairy.__file__` resolves inside this worktree.

## Run

All scripts, from the worktree root, single-threaded:

```bash
cd /c/tmp/lum_slantp && PYTHONPATH=/c/tmp/lum_slantp \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_pmm2d_staggered_slant/<script>.py
```

| script | what it produces | doc section |
|---|---|---|
| `slant_lib.py` | the prototype: the covariant slant congruence, the slanted first-order staggered generator (`SlantSolver`), the mode split, and a `PMM2DStackPure.solve` clone with per-layer slant | S1, S2 |
| `g0_reduction.py` | G0a/G0b/G0c -- slant-0 BIT-IDENTITY to the shipped `_assemble_oop` and to `PMM2DStackPure.solve` | S3 |
| `m1_null.py` | M1 -- uniform isotropic / anisotropic layer, any slant, is a no-op | S4.1 |
| `m1b_frame_phase.py` | M1b -- the frame-anchor phase on the TRANSMITTED amplitudes, pinned two-sided | S4.1 |
| `m1c_null_ladder.py` | M1c -- the null residual's M-ladder (discretization, not conditioning) | S4.1 |
| `m2_dispersion.py` | M2 -- sheared-frame uniform-slab dispersion vs exact quartic roots; four gauge arms + two ablation controls | S4.2 |
| `m3_stripe_vs_1d.py` | M3 -- y-uniform slanted stripe vs the 1-D slant oracle, per order; the SLANT-SIGN arbitration | S4.3 |
| `m3b_oracle_drift.py` | M3b -- the 1-D scalar slant oracle's OWN degree drift | S4.3 |
| `m3c_tm_spectral_oracle.py` | M3c -- the same, against the CONVERGED covariant 1-D oracle | S4.3 |
| `m3d_tm_deep_ladder.py` | M3d -- TM/TE M-ladder to M = 10, slant 0 vs 35 deg | S4.3 |
| `m4_pillar.py` | M4 -- a genuinely 2-D slanted pillar: prototype vs the hybrid slant metric vs a PURE z-staircase | S4.4 |
| `m5_census_cascade.py` | M5 -- spurious census, cascade closure vs depth, layer-split identity | S4.5 |
| `m6_cost.py` | M6 -- per-region and end-to-end cost; the staircase-equivalence arithmetic (reads `results/m4_pillar.json`) | S4.6 |
| `m7_slant_x_aniso.py` | M7 -- slant x anisotropy, incl. the slant x OUT-OF-PLANE combination the hybrid refuses | S4.7 |

Order matters only for `m6_cost.py`, which reads `results/m4_pillar.json`.

JSON outputs land in `results/`.  Every number quoted in the verdict document
comes from one of these files.

## Build pin

CPython 3.14.6, numpy 2.4.4, scipy 1.17.1 (scipy-openblas), Windows 11,
tesla-ryzen; `lumenairy` 5.43.0 + `fb3fd93`, worktree `C:/tmp/lum_slantp`,
branch `probe/pmm2d-staggered-slant`.  Single-build campaign -- per
`docs/TESTING_STANDARDS.md` rule 5 no test bar may be derived from these
numbers until they are re-measured on a second build; they are used here to
adjudicate a GO/NO-GO, which is a decision about SHAPE (convergence direction,
null exactness, sign uniqueness, two-sided controls), not about a reading.
