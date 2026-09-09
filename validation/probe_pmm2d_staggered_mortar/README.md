# probe -- per-layer grids (L2 mortar) for the PURE staggered 2-D PMM

Research probe for the GO/NO-GO in
`docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md`.
Nothing here is library code; nothing under `lumenairy/` or `tests/` is touched.

Every script asserts, before importing anything else, that `lumenairy` resolves
inside this worktree (`mortar2d.guard()`), and every number in the experiment
doc comes from one of the commands below.

## Environment

```
cd /c/tmp/lum_mortarp
export PYTHONPATH=/c/tmp/lum_mortarp OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

Measured on: Windows 11, python 3.14.6, numpy 2.4.4 + MKL, scipy 1.17.1,
threads pinned to 1 (all timings are single-threaded).

## Files

| file | what it is |
|---|---|
| `mortar2d.py` | the prototype: 1-D cross-mass between two modified-Legendre partitions, the separable (Kronecker) apply, the SQUARE and GENERALIZED 2-D mortar interfaces, and `MortarStack2D` -- the per-layer-grid twin of `PMM2DStackPure` |
| `m0_smoke.py` | algebra smoke: cross-mass reduces to the mass; `kron_apply` is exact; the V1/V2 Grams built from 1-D factors ARE the eigensolver's `-Rmat` blocks |
| `m0b_diag.py` | the two controls that separate "the mortar is wrong" from "the coarse layer is under-resolved": a single-layer h-refinement control, and the TRANSPARENT-INTERFACE probe against analytic Fresnel |
| `m0c_gram_cond.py` | `cond_2` of the V1/V2 block Grams -- the round-off scale that DERIVES the M1 bar |
| `m1_conforming.py` | M1 -- conforming identity: same grid through the mortar path vs the shipped union-grid cascade |
| `m2_nested.py`, `m2b_converge.py` | M2 -- nested grids, and the per-layer-`M` convergence ladder + the `H_BLOCK_SWAP` fail-before control |
| `m3_nonconforming.py` | M3 -- four grid pairs against ONE union-grid reference on the common refinement, scalar and Hermitian-tensor, two-sided closure |
| `m4_stripe_1d.py` | M4 -- y-uniform stripe stack, different duty per layer, per order against the exact 1-D `PMMStack` oracle |
| `m4b_perlayer_M.py` | M4b -- the per-layer modal count as the lever, priced in wall time and eig-work |
| `m4c_equal_dof.py` | M4c -- the DECISIVE comparison: union vs mortar at EQUAL degrees of freedom (identical eigenproblem sizes) |
| `m5_oop_mixed.py` | M5 -- generalized (out-of-plane) mortar twin: split uniform OOP slab vs `berreman_jones_1d`, then patterned OOP-over-scalar vs the union-grid generalized cascade |
| `m6_staircase.py` | M6 -- z-staircase taper at equal modal count, per-slice grids vs the one-fine-lattice union |
| `m6c_staircase_1d.py` | M6c -- the same staircase made y-uniform so the exact 1-D oracle applies, at EQUAL DOF, incl. the LCM=12 case where the union lattice has a DOF floor |
| `m7_conditioning.py` | M7 -- conditioning census in M1's instruments + separable-vs-dense cross-mass |

## Commands (exactly as run)

```
python validation/probe_pmm2d_staggered_mortar/m0_smoke.py
python validation/probe_pmm2d_staggered_mortar/m0b_diag.py
python validation/probe_pmm2d_staggered_mortar/m0c_gram_cond.py
python validation/probe_pmm2d_staggered_mortar/m1_conforming.py
python validation/probe_pmm2d_staggered_mortar/m2_nested.py 4 5 6 7
python validation/probe_pmm2d_staggered_mortar/m2b_converge.py 6 5 7 9 11 13
python validation/probe_pmm2d_staggered_mortar/m3_nonconforming.py 4 5
python validation/probe_pmm2d_staggered_mortar/m4_stripe_1d.py
python validation/probe_pmm2d_staggered_mortar/m4b_perlayer_M.py
python validation/probe_pmm2d_staggered_mortar/m4c_equal_dof.py 4 5 6
python validation/probe_pmm2d_staggered_mortar/m5_oop_mixed.py 4 5
python validation/probe_pmm2d_staggered_mortar/m6_staircase.py 4 5
python validation/probe_pmm2d_staggered_mortar/m6c_staircase_1d.py
python validation/probe_pmm2d_staggered_mortar/m7_conditioning.py
```

`m2_nested.py` and `m4_stripe_1d.py` take the modal ladder as positional
arguments; `m2b_converge.py` takes `M_B` first, then the `M_A` ladder.
JSON results sit beside each script (`m*.json`) and are committed.

`logs/` holds the verbatim stdout of the runs whose numbers reach the
experiment document but whose JSON is partial (a run stopped after its
informative points, to free the box for the next one): `m0b_diag.log`,
`m2b_converge.log`, `m4c_equal_dof.log`, `m5_oop_mixed.log`,
`m6c_staircase_1d.log`, `m7_conditioning.log`.

Cost note: `m2_nested.py`, `m4_stripe_1d.py`, `m4c_equal_dof.py`,
`m6c_staircase_1d.py` and `m3_nonconforming.py` at their upper rungs run region
eigenproblems of dimension 1800-2600 and take 3-13 minutes per point
single-threaded. Start at the low rungs.
