# probe_verify_mortar_round3

The INDEPENDENT re-measurement behind
`docs/audits/VERIFY_PMM2D_MORTAR_ROUND3_2026_09_11.md`.

Every fixture here is written in this directory and is deliberately disjoint
from `validation/probe_fix_mortar_round3/` and from
`tests/unit/test_fix_pmm2d_mortar_round3.py` in period, wavelength, incidence,
contrast, out-of-plane tensor, wall positions and slant.  Nothing is read from
the fix; where one of its numbers is quoted in the report it is labelled as
theirs and stands beside mine.

## Run them

Every command starts from the worktree root, with the thread variables on the
COMMAND LINE (a probe launched elsewhere silently measures a different
library; see `_path.py`).  `<tag>` is the build: `win` or `wsl`.

```sh
cd /c/tmp/lum_vmortar3
E="OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1"
env $E python -u validation/probe_verify_mortar_round3/v1_mechanism.py win
env $E python -u validation/probe_verify_mortar_round3/v2_populations.py \
    win_healthy healthy          # also: win_sliver sliver, win_broken broken
env $E python -u validation/probe_verify_mortar_round3/v3_cost.py gen
env $E python -u validation/probe_verify_mortar_round3/v3_cost.py bench \
    win_post C:/tmp/lum_vmortar3
env $E python -u validation/probe_verify_mortar_round3/v4_band.py win rest
env $E python -u validation/probe_verify_mortar_round3/v4_band.py win ladder
env $E python -u validation/probe_verify_mortar_round3/v5_falsepos.py win fp2
env $E python -u validation/probe_verify_mortar_round3/v6_bitid.py selfcheck \
    C:/tmp/lum_vmortar3
env $E python -u validation/probe_verify_mortar_round3/v7_convergence.py win
env $E python -u validation/probe_verify_mortar_round3/v8_hunt_ladder.py win

wsl -e bash -lc 'cd /mnt/c/tmp/lum_vmortar3 && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 ~/lumvenv/bin/python -u \
  validation/probe_verify_mortar_round3/v1_mechanism.py wsl'
```

## The files

| file | what it measures | report section |
|---|---|---|
| `_path.py` | puts THIS worktree first on `sys.path` and REFUSES to run if `lumenairy` resolves anywhere else | S2 |
| `_vfix.py` | the fixtures: mixed / control / adversarial per-layer stacks, in METRES | -- |
| `_capture.py` | records every mortar operand a solve builds, WITHOUT the guard, tagging each side PROMOTED structurally | -- |
| `v1_mechanism.py` | SVD, near-null participation, residual, range membership on mixed vs control operands | S3 |
| `v2_populations.py` | the HEALTHY / SLIVER / BROKEN residual populations, in three independent parts | S4, S5 |
| `v3_cost.py` | `gen` captures five real operands; `bench` times bare / gecon / guarded / residual / exact in each TREE against the same `.npz` | S0 row f |
| `v4_band.py` | the degradation ladder against the exact 1-D `PMMStack`; the ordinary-geometry census; the message, switch, and once-per-solve checks | S11 |
| `v5_falsepos.py` | two false-positive candidates for the band warning, and the silent-wrong-answer hunt | S8, S10 |
| `v6_bitid.py` | 37 fixtures / 120 hashes vs the PRE-fix tree, with a `selfcheck` mode that proves the harness reproduces its own hashes first | S7 |
| `v7_convergence.py` | do the un-refused mixed stacks converge to a mortar-free twin | S6 |
| `v8_hunt_ladder.py` | the hunt's two candidates taken to a convergence ladder | S10 |

`<name>_<tag>.json` beside each script is that script's output on that build.

## Three traps, each of which produced authoritative-looking wrong output

1. **`sys.path[0]` is the SCRIPT's directory.**  `import lumenairy` from a
   probe resolves to whatever is installed, not to the worktree.  `_path.py`
   exists for this and every JSON records the resolved path.
2. **`np.asarray` of a DICT hashes a POINTER.**  `per_order_amplitudes()`
   returns a dict; hashing it that way made the bit-identity battery report a
   HASH DIFF in a fixture the round-3 diff cannot touch.  Run
   `v6_bitid.py selfcheck` before believing any comparison.
3. **`np.linalg.solve` is not the shipped path.**  On an exactly singular
   operand `gesv` raises while `getrf` + `getrs` return a non-finite answer,
   and it is the second that `_guarded_mortar_solve` residuates.

## Units

`_vfix.py`, `v2`, `v3`, `v5`, `v7` and `v8` are in METRES (period 0.87e-6,
wavelength 0.73e-6).  `v4_band.py` is DIMENSIONLESS (period 1.07, wavelength
0.79) because it is scored against the 1-D `PMMStack`, whose `segments=` are
FRACTIONS of the period and must sum to 1 while the 2-D `x_walls=` are
absolute.  Mixing the two makes `k0` wrong by 1e6 and the per-segment
quadrature asks for a ~1e5-node Gauss-Legendre rule.
