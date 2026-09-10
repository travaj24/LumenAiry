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

`logs/` (`.txt`, because the repo gitignores `*.log`) holds the verbatim stdout
of the runs whose numbers reach the
experiment document but whose JSON is partial (a run stopped after its
informative points, to free the box for the next one): `m0b_diag.txt`,
`m2b_converge.txt`, `m3_nonconforming.txt`, `m4b_perlayer_M.txt`,
`m4c_equal_dof.txt`, `m5_oop_mixed.txt`,
`m6c_staircase_1d.txt`, `m7_conditioning.txt`.

Cost note: `m2_nested.py`, `m4_stripe_1d.py`, `m4c_equal_dof.py`,
`m6c_staircase_1d.py` and `m3_nonconforming.py` at their upper rungs run region
eigenproblems of dimension 1800-2600 and take 3-13 minutes per point
single-threaded. Start at the low rungs.

---

## FOLLOW-UP 2026-09-09 -- F1..F5 (open items O-1, O-2, O-6, O-10; roadmap N-1)

Numbers in the experiment document's `FOLLOW-UP 2026-09-09` section.

| file | what it is |
|---|---|
| `f1_crossbuild.py` | F1 (O-1) -- the six decisive tables (M1 identity, the isolated-mortar Fresnel error, M4 vs the 1-D oracle, M4c equal-DOF, the OOP twin, the conditioning census) in ONE process, so every build measures the same fixtures.  Writes `f1_crossbuild_<tag>.json` |
| `f1_compare.py` | diffs the arms and reports the per-quantity CROSS-BUILD SPREAD and the spread envelope per family |
| `f2_device_regime.py` | F2 (O-2) -- the corner-dominated 2-D pillar pair (own `N = 2` and `N = 3`, union `N = 6`, conical), equal-DOF mortar vs union against the union grid at the top of its own documented ladder; scalar and IN-PLANE LC TENSOR arms |
| `f3_perlayer_M_recipe.py` | F3 (O-10) -- the two-knob `M_A` x `M_B` convergence SURFACE on the F2 pair, each layer's own single-layer residual, the greedy rule (6/9) and the FLOOR rule (15/16) |
| `f4_uniform_oblique.py` | F4 (O-6) -- a UNIFORM layer on `N = 1` at oblique/conical: isolated vs the analytic Fresnel slab and vs `berreman_jones_1d`, then INSIDE a mortar cascade against the exact 1-D `PMMStack`, then against the shared-grid path |
| `nonuniform.py` | F5 (roadmap N-1) -- the PROTOTYPE: `Basis1DNU` (Granet Eq. 31 with per-segment jacobians), `Granet2DTransverseE_NU`, `MortarStackNU` (per-layer own-WALLS grids through the mortar cascade) |
| `f5_nonuniform.py` | F5 gates (a) bit-identity on uniform walls, (b) 2 non-uniform segments == 3 uniform segments, (c)/(c2) arbitrary walls vs the two hybrid oracles, (c3) arbitrary walls vs the EXACT 1-D oracle, (d) a 4-slice taper vs the hybrid staircase, (d2) the same taper vs the EXACT 1-D oracle |
| `f5d_diag.py` | attributes gate (d2)'s `M = 5..7` plateau: the mortar-free control, the FORCED-mortar identity on non-uniform grids, the slice-count scaling, the per-order breakdown |
| `f5e_nearwall.py` | the wall-separation sweep with the conditioning census, and the deep `M = 11` taper rung.  Its "explosion" reading is SUPERSEDED by `f5f_attrib.py` |
| `f5f_attrib.py` | measures the 1-D `PMMStack` oracle's OWN degree-12-vs-14 self-gap alongside every comparison, and so attributes the near-coincident-wall blow-up to the ORACLE (open item O-11), not to the non-uniform mortar |

### Commands (exactly as run)

```
# F1 -- three builds.  Windows:
python validation/probe_pmm2d_staggered_mortar/f1_crossbuild.py win
# the same OpenBLAS forced onto its SSE kernels (a genuinely different code path):
OPENBLAS_CORETYPE=NEHALEM python validation/probe_pmm2d_staggered_mortar/f1_crossbuild.py win_nehalem
# WSL / gcc / glibc / python 3.12:
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_mortarp && PYTHONPATH=/mnt/c/tmp/lum_mortarp \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/lumvenv/bin/python validation/probe_pmm2d_staggered_mortar/f1_crossbuild.py wsl"
python validation/probe_pmm2d_staggered_mortar/f1_compare.py          # win wsl win_nehalem

python validation/probe_pmm2d_staggered_mortar/f2_device_regime.py scalar tensor
python validation/probe_pmm2d_staggered_mortar/f3_perlayer_M_recipe.py
python validation/probe_pmm2d_staggered_mortar/f4_uniform_oblique.py

python validation/probe_pmm2d_staggered_mortar/f5_nonuniform.py a b
python validation/probe_pmm2d_staggered_mortar/f5_nonuniform.py c d
python validation/probe_pmm2d_staggered_mortar/f5_nonuniform.py c3 d2
python validation/probe_pmm2d_staggered_mortar/f5d_diag.py
python validation/probe_pmm2d_staggered_mortar/f5e_nearwall.py sweep
python validation/probe_pmm2d_staggered_mortar/f5e_nearwall.py deep
python validation/probe_pmm2d_staggered_mortar/f5f_attrib.py
```

Cost note for this round: `f2_device_regime.py`'s union reference ladder runs
to `q = 36` (eig 2592, 1028 s for that one point) and the whole script is
~45 min; `f5e_nearwall.py deep` is ~20 min; everything else is minutes.
`f4_uniform_oblique.py` deliberately does NOT sweep `grid = 6` in part (ii)
(`q = 6 (M_u - 1)` reaches an eig of 5832 at `M_u = 10`, an hour per point) --
the shared-grid arm is measured in part (iii) instead.

Two traps this round produced, recorded so they are not re-produced:

* **Do not score a conical solve against `fresnel_slab_te`.**  At `phi != 0`
  the incident `E_y` row is not s-polarized, so the scalar Fresnel formula
  reads a spurious constant error (2.4e-02 FLAT in every `(N, M)` cell) that
  looks exactly like a solver defect.  Use `berreman_jones_1d` for conical.
* **Report the oracle's own self-gap next to every comparison.**  The
  near-coincident-wall "explosion" of `f5e_nearwall.py` was the 1-D oracle
  losing convergence (self-gap 4.8e-01), and at one `delta` the oracle's two
  degrees agreed on a wrong answer -- invisible without the self-gap column.
