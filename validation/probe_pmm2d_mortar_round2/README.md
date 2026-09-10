# Round-2 probes -- PMM2D staggered per-layer mortar (D1, D2, D3)

Reproducers and derivations for
`docs/audits/FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md`, which fixes the three
defects raised by `docs/audits/VERIFY_PMM2D_STAGGERED_MORTAR_2026_09_11.md`.

Every script asserts which `lumenairy` it imported and writes its JSON beside
itself.  Run from the worktree root:

```
PYTHONPATH=$PWD OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_pmm2d_mortar_round2/<script>.py [sections]
```

| script | sections | what it measures |
|---|---|---|
| `r1_onset.py` | `ladder cond shared` | the sliver's spurious spectrum vs the free predictor; `cond_2` of both mortar solve operators, the cross-mass factors and `W`/`V`, with the exponent in `1/delta` FITTED; and whether the SHARED (union pixel-lattice) path can be driven into the same band (it cannot -- `x_walls` is refused there) |
| `r2_quad.py` | `need rule ladder` | D3.  The smallest Gauss order reaching a refined rule's OWN floor over `(omega, M)` (the bar derived per point from the reference's 37-node self-drift); the candidate rule scored two-sided (clears the requirement; returns `2 M + 8` on every uniform lattice `M = 3..14 x N = 1..60`); and the kernel-error ladder |
| `r3_ladder.py` | `ladder healthy taper` | the D1 onset map on a y-UNIFORM stack scored against the exact 1-D `PMMStack`, with ordinary-wall controls |
| `r4_screen.py` | `pop` | D2.  106 mortar solves over a HEALTHY population (every grid pair the shipped fixtures build) and a SLIVER population, with the exact `cond_2`, the LAPACK `gecon` estimate, the free lower bound, and a bit-identity check of `lu_factor`+`lu_solve` against `np.linalg.solve` |
| `r5_conv.py` | `conv nomortar` | the DECIDING D1 measurement: the `M = 4..8` ladder at six wall separations -- does it stop converging? |
| `r6_bitid.py` + `r6_compare.py` | -- | 33 fixtures / 87 sha256 hashes, run twice (this tree and a pristine `git archive HEAD` tree) |
| `r7_allhost.py` | `pure nomortar taper` | the mortar's OWN error: three ALL-HOST layers on three different grids against the ANALYTIC slab, plus the no-mortar control and the narrowest segment `add_tapered_pillar` actually builds |

`r1_onset.json` is the `cond`/`shared` run; `r1_onset_ladder.json` is a
superseded first cut of the onset instrument (its reference's own `M`-ladder
self-gap, 9.2e-04, is the size of the effect -- which is WHY `r3`/`r5` score
against an exact oracle instead).  It is kept because the fix doc cites it.
