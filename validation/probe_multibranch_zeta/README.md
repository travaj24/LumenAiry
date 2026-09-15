# WP-B7c probes -- the multibranch blow-up and the fold envelope

Everything behind
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-B7c_REPORT.md`
(handoff items 4.2 and 4.3).  Nothing here is imported by the library or by the
test suite; the JSON is the recorded measurement for both builds.

| file | what it does |
|---|---|
| `fixtures.py` | the five optics: `V` (VERIFY-B7b section 4.1's fold fixture, verbatim), `C` / `D` (two of WP-B7c's own), `W1` / `W2` (WP-B7b's own two, verbatim from `WP-B7b_REPORT.md` sections 3.1 and 3.2) |
| `oracle.py` | drives `validation/oracles/caustic_fold_truth.py` -- the lumenairy-free direct Rayleigh-Sommerfeld ring integral over an exact conic trace -- onto a fixture's own grid.  Schott Sellmeier coefficients are typed HERE; `index_control()` reports the delta against `get_glass_index` (0.0 on all four glasses) |
| `probe1_scan.py` | plane scan of the completion's diagnostics and two energy readings taken outside the module |
| `probe2_mechanism.py` | the three mechanism controls (`min_area_ratio`, `caustic_band`, `ray_subsample`) plus the mapped-triangle census in pixel units |
| `probe3_foldscan.py` | cheap fold scan (meridional tracer only): which planes carry a single fold ring and what `zeta_extrapolation` ladder they span |
| `probe4_ladder.py` | the scoring ladder: four members against the oracle, fidelity + power + the full diagnostics |
| `probe5_tail.py` | `zeta(r)` beyond the two-branch band (`kappa_eff` from the oracle's dark-side decay) and the cumulative dark-tail energy by depth |
| `probe6_ustar.py` | the band's own curvature bound `u* = 0.1 kappa / \|q\|` over every fixture and plane |
| `probe7_member.py` | the member ordering against beam width at a fixed optic and plane (the refuted aperture-truncation hypothesis) |
| `summarise.py` | the tables in the report, from the ladder JSON |
| `bitid.py` | the archive-to-archive bit-identity probe: run it INSIDE a tree, it asserts `lumenairy.__file__` lives there and digests 13 fixtures |

Every run carries `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
`*_win.json` is Windows py3.14.6 / numpy 2.4.4; `*_wsl.json` is WSL
py3.12.3 / numpy 2.4.6; `*_parent_*.json` was taken in a `git archive` tree of
the base commit `96cb2096`.
