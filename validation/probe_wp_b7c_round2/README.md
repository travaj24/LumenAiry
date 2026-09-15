# `probe_wp_b7c_round2` -- the pixel-halving arbiter

Probes, oracle driver and JSON behind
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7c_ROUND2_REPORT.md`.

Nothing here is imported by the library or by the test suite; the decision
tests (`tests/unit/test_audit2609_b7c2_pixel_halving_arbiter.py`) re-derive
what they assert on the running build.

## The measurement, in one line

Re-rasterise the SAME mapped triangles onto a grid of half the pitch over the
same physical window, from the same launch lattice, and compare the deposited
power.  A converged point-sampled quadrature reads 1 at any pitch; one that has
stopped being unbiased reads ~4 per halving.

## Files

| file | what it does |
|---|---|
| `fixtures.py` | eight optics -- VERIFY-B7b's own fold fixture, VERIFY-WP-B7c's four, and three new ones (an AIR-SPACED doublet, a CONIC-surfaced N-LAK22, an N-SF11 at NA 0.33).  The VERIFY module is loaded by PATH under a distinct name because the two files share a basename |
| `oracle.py` | the driver for `validation/oracles/caustic_fold_truth.py`, plus the EXACT azimuthal quadrature that measures that oracle's Debye `J0` NA ceiling.  Sellmeier coefficients are typed here from the Schott catalogue and the delta against the library's own dispersion is reported as a control, never used |
| `scan.py` | oracle-FREE plane scan on HEAD: the launched-power reading, the branch-sum continuity, the RETURNED field's continuity (read back from the refusal message where the plane is refused), the route and the branch count.  Seconds per plane, so it is what chooses the planes the scored probe runs on |
| `arbiter.py` | the scored probe: everything `scan.py` reports plus the uniform completion, the ray-to-wave hand-off and all of them against the oracle.  Run on the audit base `96cb2096` (which refuses nothing) for the classification |
| `summarise.py` | joins HEAD's readings to the base tree's oracle scores on `(fixture, z)` and DERIVES the bar: the populations, the criterion-free separation, and the confusion matrix at two accept criteria |
| `oraclefloor.py` | the oracle's own floor and the NA ceiling (D5): the Debye `J0` phase error, the difference against the exact azimuthal quadrature, and the exact arm's convergence in its two knobs |
| `bitid.py` | 30 fixtures, SHA-256 over `tobytes()`, in a child process per tree with `lumenairy.__file__` asserted under it |
| `cost.py` | the arbiter's price, timed on whichever tree it is run against |
| `explore0.py` / `explore1.py` / `explore2.py` | the exploratory passes kept for provenance: that the reading separates at all, that it is not an artefact of resampling the input, and the three candidate statistics (branch sum, completed field, bright side only) measured side by side -- which is how the shipped one was chosen |

## Running them

Every command from the worktree root, with
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on the command
line and `PYTHONPATH` pinning the tree.  The probes are scripts, not `-c`
snippets, so `sys.path[0]` is the probe directory and the tree really is the
one `PYTHONPATH` names -- each prints `lumenairy.__file__` first.

```
python validation/probe_wp_b7c_round2/scan.py F_alt --out head_F_alt_win.json 1069 1072 1074 1076 1080
PYTHONPATH=/c/tmp/lum_mb2_pre python validation/probe_wp_b7c_round2/arbiter.py F_alt band_F_alt_pre_win.json --noexternal 1069 1072 1074 1076 1080
python validation/probe_wp_b7c_round2/summarise.py joined_head_win.json head_*.json band_*_pre_win.json
python validation/probe_wp_b7c_round2/oraclefloor.py oraclefloor_win.json S:3214.78 G:950 P:888
python validation/probe_wp_b7c_round2/bitid.py C:/tmp/lum_mb2 bitid_head_win.json
```

`--phi exact` on `arbiter.py` scores against the exact azimuthal quadrature
instead of the shared oracle's `J0` form; it is what the NA 0.33 fixture is
scored with.
