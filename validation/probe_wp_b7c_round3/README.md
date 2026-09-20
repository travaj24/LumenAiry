# WP-B7c round 3 -- probes

Everything behind a number in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7c_ROUND3_REPORT.md`.

Round 3 answers the follow-ups the round-2 VERIFICATION left
(`fixes/VERIFY_WP-B7c_ROUND2.md`, defects E1, E4, E5, E6, E7 and claims
1b / 1c / 5 / 6).

## What is re-used, and what is not

The three propagators of the verifier's oracle (`rs_j0`, `rs_exact`,
`asm_field`) and its scoring helpers are imported VERBATIM from
`validation/probe_verify_b7c_round2/vroracle.py`.  They were written
independently of the builder's oracle, convergence-controlled there, and
re-deriving them a third time would put a third bias between a round-3 number
and the round-2 number it is meant to correct.  What round 3 adds to the
oracle is an EVEN-ASPHERIC term in the ray trace (round 2's optics were
spherical or conic only) and the exact azimuthal quadrature applied at every
radius out to the grid corner rather than inside the energy core (E6).

The fixture modules of both earlier rounds are imported, so the round-3
population CONTAINS both -- including `P` (f/1.2), which round 2 excluded
because the shared oracle's Debye `J0` form does not hold at `y_max/z = 0.45`
and which the angular spectrum scores without an envelope argument.

## The files

| file | what | outputs |
|---|---|---|
| `r3oracle.py` | the verifier's three propagators + an aspheric-capable exact meridional trace + `exact_full_radius`; carries its own control that the aspheric extension is bit-inert on a spherical prescription | -- |
| `r3fixtures.py` | 16 distinct prescriptions: the builder's 8, `P`, the verifier's 5, and three new -- `AS` (a true even-ASPHERE with a non-monotone focal locus), `MC` (a CONCAVE-FIRST positive meniscus), `HN` (plano-convex at **NA 0.41**) | -- |
| `r3geom.py` | index control, NA, paraxial / marginal foci, the fold window per optic; the ladders are derived from its output and from nothing else | `geom_win.json` |
| `r3ladder.py` | the band / tail ladders, as functions of `geom_win.json` | -- |
| `r3scan.py` | one call per plane with both refusal bars patched to `inf` at run time, so the reading and the field it would have refused come from the SAME call; every diagnostic recorded whole | `band_*_win.json`, `fine_*_win.json` |
| `r3run.py` | the scan pool, one child process per fixture | `runsummary_*_win.json` |
| `r3fine.py` | the fine ladders: brackets the bar per optic from a coarser scan and re-scans between the straddling planes at 20-30x the step; run repeatedly to refine | `fine_*_win.json` |
| `r3join.py` | the populations, the gap, the TWO-SIDED margins and the bar-cost table by fidelity band; joins a post-fix READING column to a pre-fix FIDELITY column and checks the returned power is bit-equal before it does | `joined_*_win.json` |
| `r3control.py` | what a CONVERGED render reads: many optics far from every caustic, plus a grid ladder and a window ladder on the slow control | `control_*_win.json` |
| `r3oraclefloor.py` | E6: the Debye `J0` form against the exact azimuthal quadrature at FULL radius and against the angular spectrum, each with its own convergence control | `oraclefloor_win.json` |
| `r3fallback.py` | R3-4: does anything order the fallback route's fidelity?  Spearman + best-threshold separation for every candidate already in the diagnostics, plus two computed from the ray trace | `fallback_win.json` |
| `r3bitid.py` | bit identity ARCHIVE to ARCHIVE, child process per tree, `lumenairy.__file__` asserted, `LUMENAIRY_MEM_BUDGET_MB` pinned | `bitid_win.json` |
| `r3mutate.py` | the 17-mutation matrix (the verifier's 13 + round 3's 4) against the whole five-file gate | `mutation_*.json`, `mut_*.txt` |

## Reproducing

```
cd validation/probe_wp_b7c_round3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=<tree>

python r3geom.py geom_win.json                 # index control + the ladders' input
python r3run.py pre 6 --alt                    # the 469-plane band + tail scan
python r3fine.py f1 6 1.06 -- band_*_pre_win.json      # 501 planes, 20-30x finer
python r3fine.py f2 6 1.06 -- band_*_pre_win.json fine_*_f1_win.json
python r3run.py post 6 --alt --oracle none     # the readings again, post-fix
python r3join.py joined_post_win.json --readings band_*_post_win.json -- band_*_pre_win.json fine_*_win.json
python r3control.py control_pre_win.json
python r3oraclefloor.py oraclefloor_win.json
python r3fallback.py fallback_win.json joined_post_win.json
python r3bitid.py bitid_win.json <pre archive> <post archive>
MUT_TREE=C:/tmp/lum_mb3_mut python r3mutate.py mutation_win.json
```

The scans are the slow part (one angular-spectrum oracle per plane); every
`*_win.json` carries its rows, so `r3join.py`, `r3fallback.py` and the report
can be re-run without re-measuring.
