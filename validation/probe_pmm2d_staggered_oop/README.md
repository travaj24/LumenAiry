# Stage-B probe: out-of-plane anisotropy for the PURE staggered 2-D PMM

Prototype only -- **no library code is changed by this branch**
(`probe/pmm2d-staggered-oop`, off `main` f70628d = lumenairy 5.42.1 + BOR SEM).
The write-up with every table is
`docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_2026_09_09.md`.

## How to run

Every script pins `PYTHONPATH` to this worktree and asserts
`lumenairy.__file__` starts with it (`probe_common.assert_worktree`).  The
OMP / OpenBLAS / MKL caps are exported **before** python starts.

```bash
cd /c/tmp/lum_aniso_oop
export PYTHONPATH=/c/tmp/lum_aniso_oop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

python validation/probe_pmm2d_staggered_oop/m0_convention.py
python validation/probe_pmm2d_staggered_oop/m1_dispersion.py
python validation/probe_pmm2d_staggered_oop/m2_berreman.py
python validation/probe_pmm2d_staggered_oop/m3_inplane_limit.py
python validation/probe_pmm2d_staggered_oop/m4_stripe_1d.py
python validation/probe_pmm2d_staggered_oop/m5_pillar_2d.py
python validation/probe_pmm2d_staggered_oop/m6_cascade.py
python validation/probe_pmm2d_staggered_oop/m7_cost.py
python validation/probe_pmm2d_staggered_oop/m8_nx3_grid.py
python validation/probe_pmm2d_staggered_oop/m9_corner_control.py
```

or all of them:

```bash
bash validation/probe_pmm2d_staggered_oop/run_all.sh m0 m1 m2 m3 m4 m5 m6 m7 m8 m9
```

Logs land in `validation/probe_pmm2d_staggered_oop/logs/`, JSON summaries in
`validation/probe_pmm2d_staggered_oop/results/`.  Total wall time on
tesla-ryzen (single-threaded): ~90 min, the M2, M5 and M8 ladders
dominating (the 2-D Fourier oracles at high truncation are most of it).

## Files

| file | what |
|---|---|
| `probe_common.py` | the prototype: staggered basis (copied verbatim from `twod_staggered.py`), tensor-weighted Galerkin assembly, **candidate (a)** first-order `4q^2` generator, **candidate (d)** `6q^2` quadratic pencil, the in-plane E-form rebuilt in the probe's sign convention, exact quartic `kz` roots, Gram-weighted modal flux, the far-field projection, and a single-layer R/T/Jones driver |
| `m0_convention.py` | M0: Bloch sign of `Basis1D`, exact-root solver validation, probe E-form vs the shipped `Granet2DTransverseE`, strong vs Eq.-25 H-partner |
| `m1_dispersion.py` | M1: uniform-slab dispersion, M-ladder, resolvable window, assignment census, (a)-vs-(d) spectra, convention arbitration |
| `m2_berreman.py` | M2: uniform OOP slab R/T/Jones vs `berreman_jones_1d`, normal / oblique / conical, M-ladder |
| `m3_inplane_limit.py` | M3: the in-plane limit -- spectra, observables, and the algebraic reduction to the shipped E-form |
| `m4_stripe_1d.py` | M4: y-uniform OOP stripe grating vs `pmm_jones_1d` + `rcwa_jones_1d`, per order, plus y-momentum conservation |
| `m5_pillar_2d.py` | M5: 2-D OOP pillar vs the hybrid `pmm_jones_2d`, with the hybrid's own floor measured and the no-floor property two-sided |
| `m6_cascade.py` | M6: cascade stability on a depth ladder (0.25 / 1 / 3 wavelengths) + energy closure |
| `m7_cost.py` | M7: dimension and eig wall time vs the in-plane path; the normal-incidence anti-commuting-involution structure |
| `m8_nx3_grid.py` | M8: a (3,3) grid with a non-centro-symmetric OOP feature vs both 2-D oracles, plus position invariance |
| `m9_corner_control.py` | M9: attributing M8's disagreement -- the same cells with the OOP entries zeroed (so the staggered arm is the SHIPPED in-plane discretization), and a convex feature as the second arm |
| `run_all.sh` | driver |

## Mount

Windows 11, python 3.14.6, numpy 2.4.4, scipy 1.17.1, tesla-ryzen.
`lumenairy 5.42.1` from `C:\tmp\lum_aniso_oop\lumenairy\__init__.py`.
