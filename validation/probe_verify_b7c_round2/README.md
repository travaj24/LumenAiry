# VERIFY-WP-B7c round 2 -- the independent verifier's probes

Everything behind a number in
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B7c_ROUND2.md`.

Nothing here shares code with `validation/oracles/caustic_fold_truth.py`,
`validation/probe_verify_b7c/` or `validation/probe_wp_b7c_round2/`, except
that `vfixtures.builder()` imports the builder's fixture module so that the
two claims stated ON the builder's geometries (claim 2 at `V` z = 1761 um,
claim 4 at `F_alt`) are re-measured on those geometries and not on lookalikes.

## The oracle

| file | what |
|---|---|
| `vroracle.py` | Sellmeier coefficients for ten Schott glasses typed here (control against `lumenairy.glass`: 2.2e-16 worst case); an exact sequential meridional CONIC ray trace; and THREE propagators of the traced exit field -- `rs_j0` (the shared oracle's Debye `J0` ring integral, re-implemented), `rs_exact` (the same integral with the azimuth kept EXACTLY at every radius), `asm_field` (a band-limited angular spectrum, exact for the scalar Helmholtz equation) |
| `vfixtures.py` | five optics the campaign has never run: `W` plano-first, `X` NA 0.3865, `Y` oblate-conic, `Z` cemented doublet, `C` the converged control -- plus `*_alt` grids and the two builder fixtures |
| `vgeom.py` | reconnaissance: index control, NA, marginal / paraxial foci, the z window with an interior fold |

## The measurements

| file | what | outputs |
|---|---|---|
| `vscan.py` | ONE call per plane on ONE tree with both refusal bars patched to `inf` at run time, so the reading and the field it would have refused come from the same call; the shipped decision is re-derived from the constants imported before the patch | `band_*_win.json`, `fine_*_win.json`, `find_*_win.json`, `log_*.txt` |
| `vjoin.py` | the populations, the gap and the confusion matrix at both accept criteria | `joined_win.json` |
| `vsum.py` | compact table of any scan log | -- |
| `vreadings.py` | the 23 readings every claim rests on, no oracle, run on BOTH builds | `readings_win.json`, `readings_wsl.json` |
| `vsweep.py` | is the band a property of the quantity or of the fixture set?  `ray_subsample` 1..8 and `dx` x0.5..x2 at healthy fold planes | `sweep_W_win.json` |
| `vbitid.py` | 57-case bit identity, child process per tree, `lumenairy.__file__` asserted, `LUMENAIRY_MEM_BUDGET_MB` PINNED | `bitid_win.json` |
| `vcost.py` | structural (call counts for the trace, the KMAH leg and the fold trace, with the arbiter on and off) and wall clock, round-1 tree against HEAD | `cost_win.json` (best of 3), `cost2_win.json` (best of 7) |
| `vmutate.py` | the 13-mutation matrix, each applied to a detached worktree and run against the three shipped decision files | `mutation*_win.json`, `mut_*_win.txt` |
| `voracle_check.py` | the `J0` form against BOTH exact propagators, each with its own convergence control, at every radius out to the grid corner | `oraclefloor_win.json`, `oraclefloor_builder_win.json` |
| `log_builder_oracle_cross_win.txt` | the builder's own `oracle_field(phi='exact')` at its own fixture and plane, beside my angular spectrum |

## Reproducing

```
cd validation/probe_verify_b7c_round2
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH=<tree>

python vgeom.py                                   # index control + geometry
python vscan.py X "940:990:6,991:999:9,1000" band_X_win.json
python vscan.py X "992.22:992.40:19" fine_X2_win.json   # the two false refusals
python vjoin.py joined_win.json log_band_*_win.txt log_fine_*_win.txt
python vreadings.py readings_win.json             # and again on the second build
python vbitid.py bitid_win.json <round-1 tree> <head tree>
python vcost.py cost_win.json <round-1 tree> <head tree>
python vmutate.py mutation_win.json               # needs MUT_TREE, a detached worktree
python voracle_check.py oraclefloor_win.json B:Q 5680 W 4900 Y 1980
```

The scans are the slow part (one angular-spectrum oracle per plane); the
`*_win.json` files carry every row so `vjoin.py` and `vsum.py` can be re-run
without re-measuring.
