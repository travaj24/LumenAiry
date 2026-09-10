# probe_verify_oop_block_eig -- independent verification probes

Measurement scripts for
`docs/audits/VERIFY_PMM2D_STAGGERED_OOP_BLOCK_EIG_2026_09_10.md`, the
independent adversarial verification of the normal-incidence PARITY block
reduction (`lumenairy/elements/pmm/twod_staggered._stag_block_eig`) and of the
converged-reference study
`docs/audits/EXPERIMENT_PMM2D_STAGGERED_OOP_REFERENCE_2026_09_10.md`.

Nothing here reads the build's own probes.  Fixtures are written from the
physics in `vfix.py` (a different tensor zoo and a different mount from the
build's), each cell-construction helper ASSERTS whether the cell is its own
parity image, and where the library uses an index trick the probe recomputes
the same quantity by a second, independent route (an explicit dense `R`; a
re-assembly of the twelve pencil blocks from the public primitives; a
from-scratch re-implementation of the extrapolation).

Every script calls `vfix.assert_arm(...)` and refuses to run if the imported
`lumenairy` is not the copy that arm is supposed to measure.

## how to run

All commands from the worktree root, one BLAS thread:

```
cd /c/tmp/lum_vacc
export PYTHONPATH=/c/tmp/lum_vacc OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1
python validation/probe_verify_oop_block_eig/<script>.py
```

The one exception is the `fb3fd93` comparison arm, which must be launched from
a directory that is NOT the worktree (python puts the cwd first on `sys.path`,
which otherwise wins over `PYTHONPATH`):

```
cd <any other directory>
PYTHONPATH="D:/Metacept/.../Lumenairy" OMP_NUM_THREADS=1 ... \
  python /c/tmp/lum_vacc/validation/probe_verify_oop_block_eig/v1_default_on.py main
```

| script | what it measures | wall time |
|---|---|---|
| `vfix.py` | shared fixtures, hashing, metrics (not run directly) | -- |
| `v1_default_on.py tip \| main \| compare` | sha256 bit-identity across three arms (tip `auto`, tip `False`, `fb3fd93`) on scalar / in-plane / magnetic / refused-OOP fixtures | ~4 min + ~3 min |
| `v2_on_off.py` | structure residual `dA`/`dB` with an EXPLICIT dense `R`, on 13 carrying and 9 violating cells; then ON vs OFF observables, eigenvalue set and the `2q^2/2q^2` split | ~9 min |
| `v3_involution.py` | the 1-D parity maps, the twelve re-assembled pencil blocks, the two eliminations, all 64 (permutation, sign) patterns, 4 corruptions of the 1-D map | ~2 min |
| `v4_failbefore.py` | fail-before with the gate disarmed; a CONTINUOUS violation swept across `_STAG_BLOCK_TOL`; eigenpair backward error vs the dense `zgeev` | ~6 min |
| `v5_speed.py` | interleaved cold-subprocess whole-solve timings (`--whole` is the single-arm worker) and in-process region timings | ~25 min |
| `v6_refit.py` | an independent re-fit of the committed ladders: a scanned-rate power law AND a pure Aitken/Shanks transform; C3 agreement, C4 direction, C5 bound | ~3 min |
| `v6b_refit_detail.py` | the same on the no-Jones observable set (reproduces the study's own counts), the C4 "worst observable" selection, C1/S2 spot-checks | ~4 min |
| `v6c_rcwa_oop.py` | re-runs one staggered and one rcwa ladder rung against the committed JSON; zeroes `e13/e23/e31/e32` to confirm `rcwa_jones_2d` consumes them | ~1 min |
| `v7_test_durability.py` | every numeric constant of the two new test files, re-measured on those tests' own fixtures | ~2 min |

`results/` holds the JSON each script writes; `logs/` the console record.

## the reference-study data

`v6_refit.py` and `v6b_refit_detail.py` READ
`validation/probe_pmm2d_staggered_oop_reference/results/t1_{corner,chiral}.json`
-- every rung of every ladder is stored there, so no ladder needed re-running.
`v6c_rcwa_oop.py` establishes that those stored rungs are reproducible on this
build from an independently constructed fixture (worst disagreement 4.4e-16 on
the staggered `M = 7` rung, and exactly 0 on the rcwa `n = 5` rung).
