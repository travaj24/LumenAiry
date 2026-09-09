# Probes -- MAGNETIC (permeability-tensor) anisotropy for the pure staggered 2-D PMM

Every measurement quoted in
`docs/audits/BUILD_PMM2D_STAGGERED_MAGNETIC_2026_09_10.md` and in the bar
comments of `tests/unit/test_pmm2d_staggered_magnetic.py` comes from one of
these scripts.  They are small and self-contained; none writes data files.

Run any of them from the worktree root:

```
PYTHONPATH=/c/tmp/lum_mag OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_pmm2d_staggered_magnetic/<file>.py
```

Each asserts `lumenairy.__file__` starts with the worktree path, so a probe can
never silently measure an installed wheel.

| probe | build-doc table | what it measures |
|---|---|---|
| `m2_analytic_slab.py` | M2 | G2: a uniform isotropic magnetic slab vs the ANALYTIC Airy / characteristic-matrix formula with the wave impedance -- convention pinned on the nonmagnetic arm first, then lossless and LOSSY magnetic arms, M ladder, TE and TM, R / T / Jones |
| `m3_duality.py` | M3 | G3: electromagnetic duality `(eps, mu) <-> (mu, eps)` in the (s, p) frame -- per-order R/T and the order-0 Jones, M = 5..8, normal / oblique / conical, with the no-rotation control |
| `m4_1d_dual_stripe.py` | M4 | G4: the 1-D ENGINE CENSUS, and a y-uniform MAGNETIC stripe against `pmm_jones_1d` + `rcwa_jones_1d` through duality, per order, plus the y-forbidden leak |
| `m5_closure_symmetry.py` | M5, M6, M9 | G5 lossless closure with a Hermitian mu, the lossy-mu deficit, the tripwire predicate; G9 the `layer_absorption` budget with a lossy magnetic layer; G6 the x<->y transpose with the mu blocks swapped + the m12/m21 placement control |
| `m6_failbefore_guards_cost.py` | M1b, M7, M8 | the four fail-before knockouts (incl. the R-vs-Gram trap), the guard census, and the cost of the magnetic assembly vs the eps-only one at (3,3) M=8 |
