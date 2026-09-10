# INDEPENDENT verification of ROUND 4 -- the per-AXIS band warning

Report: `docs/audits/VERIFY_PMM2D_MORTAR_ROUND4_2026_09_11.md`.
Under verification: `docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md`
(commit `b7239bf`, parent `15af675`).

Every fixture here is this verification's own -- different period (1.24e-6 m),
wavelength (0.905e-6 m), incidence, substrate, contrast, out-of-plane and
magnetic tensors, wall positions, slant and thicknesses from both the fix's
probes and the round-3 verification's.  **Units are METRES throughout.**

| file | what it measures |
|---|---|
| `_path.py` | pins the tree under measurement (`VMORTAR4_TREE`) and REFUSES to run if `lumenairy` resolves elsewhere; the resolved path is recorded in every JSON |
| `_vfix4.py` | the fixtures: 32 per-layer stacks, 5 patterned ladders, 6 uniform-slab ladders, the promotion population, the 1-D-oracle ladder, and the FIX's own DEFECT-2 device |
| `v1_axis_identity.py` | task A -- the family solved and hashed, the geometry read per axis, and `window_halfwidth` |
| `v2_bars.py` | task C -- the SPREAD population (bar 1) and the `M` ladder against an exact 1-D oracle (bar 2) |
| `v3_ladder.py` | task B -- is a band-width segment on a NON-mortared axis actually harmless?  Patterned (deciding), uniform-slab, and the fix's own device swept on BOTH axes |
| `v4_compare.py` | flattens two `v1` trees and classifies every differing leaf as answer / warning / other |

## Runs

```sh
# ONE BLAS thread on the COMMAND LINE; a module's own os.environ.setdefault is
# a no-op in a multi-file run.  Every command starts with the cd.

# task A -- the SAME probe against BOTH trees, in separate interpreters
cd /c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VMORTAR4_TREE=C:/tmp/lum_vmortar4 python \
  validation/probe_verify_mortar_round4/v1_axis_identity.py --tag win_post \
  --parts fgw
cd /c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VMORTAR4_TREE=C:/tmp/lum_vmortar4_pre python \
  validation/probe_verify_mortar_round4/v1_axis_identity.py --tag win_pre \
  --parts fgw
cd /c/tmp/lum_vmortar4 && python \
  validation/probe_verify_mortar_round4/v4_compare.py --pre win_pre \
  --post win_post

# task B -- the ladders
cd /c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VMORTAR4_TREE=C:/tmp/lum_vmortar4 python \
  validation/probe_verify_mortar_round4/v3_ladder.py --tag win --Ms 4,6
cd /c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VMORTAR4_TREE=C:/tmp/lum_vmortar4 python \
  validation/probe_verify_mortar_round4/v3_ladder.py --tag win_fixdev \
  --Ms 4 --kinds "" --fixdev --trivial

# task C -- the two restated bars
cd /c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 VMORTAR4_TREE=C:/tmp/lum_vmortar4 python \
  validation/probe_verify_mortar_round4/v2_bars.py --tag win --parts pl \
  --ladder-M 5,6,7

# the WSL arm of every one of those
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VMORTAR4_TREE=/mnt/c/tmp/lum_vmortar4 ~/lumvenv/bin/python \
  validation/probe_verify_mortar_round4/v3_ladder.py --tag wsl --Ms 4,6'

# suites and lint
cd /c/tmp/lum_vmortar4 && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest \
  tests/unit/test_verify_pmm2d_mortar_round4.py \
  tests/unit/test_fix_pmm2d_mortar_round4.py \
  tests/unit/test_fix_pmm2d_mortar_round3.py \
  tests/unit/test_verify_pmm2d_mortar_round3.py \
  tests/unit/test_pmm2d_staggered_mortar.py \
  tests/unit/test_pmm2d_staggered_nonuniform.py -q -p no:randomly
wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmortar4 && ~/lumvenv/bin/ruff check \
  lumenairy/ tests/ validation/probe_verify_mortar_round4/'
```

## Logs kept beside the JSON

`_v1_*.txt`, `_v2_*.txt`, `_v3_*.txt` are the probe stdout;
`_run_5suites_{win,wsl}.txt` are the five suites the task names on each build,
`_run_6suites_win.txt` the same set with this verification's own file added.
