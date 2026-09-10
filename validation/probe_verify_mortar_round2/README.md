# VERIFY probes -- PMM2D staggered per-layer mortar, ROUND 2

Independent re-measurement suite for
`docs/audits/VERIFY_PMM2D_MORTAR_ROUND2_2026_09_11.md`, which verifies
`docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md` (defects D1, D2, D3).

Every script asserts which `lumenairy` it imported and writes its JSON beside
itself.  Nothing here reads a number from the fix doc: the fixtures, the
oracles and the bars are all built here.  Run from the worktree root:

```
PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_verify_mortar_round2/<script>.py [sections]
```

WSL:

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vmortar2 && \
  PYTHONPATH=/mnt/c/tmp/lum_vmortar2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  V5_TAG=wsl ~/lumvenv/bin/python validation/probe_verify_mortar_round2/v5_d3.py"
```

| script | sections | what it measures |
|---|---|---|
| `v1_bitid.py` | -- | 30 fixtures / 165 sha256 hashes PLUS the warning set: shared-grid (scalar, tensor, out-of-plane, magnetic, slanted, `retain_internal` + `layer_absorption`, per-order amplitudes, `jones=False`), per-layer (conforming, non-conforming, nested, mixed, per-layer `n_modes`, both taper builders + `rule='bottom'`, tensor, slant, magnetic, `retain_internal`, uniform-ARRAY and INTEGER spellings), four 1-D `PMMStack` arms, and the projector directly on both global stencil sets |
| `v1_compare.py` | -- | compares two `v1_bitid` runs hash by hash and warning set by warning set |
| `v2_d1.py` | `floor allhost spectrum exponents` | D1 re-derived on an INDEPENDENT fixture: the `M`-ladder floor against the exact 1-D `PMMStack`; three ALL-HOST layers on three grids against the ANALYTIC Airy slab; the free spurious-spectrum predictor and its constant `c` (incl. a cross-fixture check against the builder's own two fixtures); the fitted exponents in `1/delta` |
| `v3_guard.py` | `census boundary exempt shared falsepos degraded nomortar_fp` | BOTH break attempts on the width contract: the false-positive census read off the shipped `_STAG_SEG_CENSUS` hook; the 1e-9 slack boundary; the integer exemption; the shared path's refusal; the closing taper's crossing slice; the SILENT degraded band above the bar; and the conforming-stack false positive |
| `v4_d2.py` | `bitid pop refuse warnerr plain1d` | D2: `lu_solve(lu_factor(A), B)` vs `np.linalg.solve` on the REAL captured mortar operands; the healthy `rcond` population per SITE; whether the backstop fires on a one-axis sliver; `LinAlgWarning` under `-W error`; the `~1850` plain 1-D decision numbers |
| `v5_d3.py` | `need rule kernel integer candidate nqmap` | D3: the measured quadrature requirement and its slope predictor; the shipped rule's margin; the kernel ladder BEFORE/AFTER; the 720-cell integer claim; the rejected candidate's 457; and which ordinary geometries the rule moves |
| `v6_d3_impact.py` | (`compare`) | how much D3 moves an ordinary per-layer FAR FIELD, `with` vs `24651c8` |
| `v7_generalized.py` | -- | the THIRD guarded site (`_interface_smatrix_general_mortar_2d`): its healthy `rcond` population over 24 ordinary out-of-plane / slanted / mixed per-layer stacks.  **This is where DEFECT V1 was found.** |
| `v8_oop_regression.py` | (`compare`) | DEFECT V1's two-way reproducer: ONE device built per-layer (takes the generalized mortar) and on the common refinement (does not), on both trees and both builds |

## The `without` arm

```
git worktree add --detach C:/tmp/lum_prem2 24651c8
V1_TAG=pre PYTHONPATH=/c/tmp/lum_prem2 V1_EXPECT_ROOT=/c/tmp/lum_prem2 \
  python <this dir>/v1_bitid.py
python <this dir>/v1_compare.py with pre
```

The `V*_EXPECT_ROOT` assertion is not decoration: a probe launched with the
wrong `PYTHONPATH` silently measures the wrong tree.

## Two traps recorded here because they produce FALSE PASSES

1. **`np.asarray` of a closure or a dict is a 0-d object array whose
   `tobytes()` is a POINTER.**  `_stag_fourier_projection` returns a closure and
   `per_order_amplitudes` returns a dict; hashing them directly hashes process
   addresses.  `v1_bitid.py` applies the closure to `basis.B` / `basis.Btilde`
   and hashes each dict entry.
2. **`stack.py` imports `_interface_smatrix` BY NAME at module level**, so
   patching `lumenairy.elements.pmm._core._interface_smatrix` alone intercepts
   nothing (measured: 0 calls).  `v4_d2.py plain1d` patches both modules.
