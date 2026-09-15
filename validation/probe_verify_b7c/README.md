# VERIFY-WP-B7c probes

Independent re-derivation of WP-B7c's refusal band, its blow-up mechanism and
its two fold constants, on FIVE prescriptions WP-B7c never used and on two
builds.  Report:
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-B7c.md`.

Nothing here imports WP-B7c's own `validation/probe_multibranch_zeta/`.  The
shared piece is the ORACLE itself (`validation/oracles/caustic_fold_truth.py`,
the lumenairy-free direct Rayleigh-Sommerfeld ring integral the whole B7 arc
is scored against); `oracle.py` here is a separate driver with its own typed
Sellmeier table and its own multi-surface handling.

| file | what it does |
|---|---|
| `fixtures.py` | the optics: `Q` cemented doublet (overcorrected SA), `M` positive meniscus, `S` convex-first plano-convex at 532 nm, `F` fast f/2.0 singlet, `P` the same singlet at f/1.2 (kept to expose the oracle's NA ceiling), plus a second grid for each (`*_alt`) |
| `oracle.py` | drives `caustic_fold_truth.py`; Sellmeier typed here, delta against `get_glass_index` reported as a control |
| `probe0_scan.py` | cheap ladder scan (no oracle) -- locates the fold window, the fallback window and the blow-up transition per fixture |
| `probe1_band.py` | the oracle-scored ladder: branch sum, completion (or its `RuntimeError`), both hand-offs, all four scored against the oracle.  Run on BOTH trees; the base tree supplies the fidelity at the planes the head refuses |
| `probe2_mech.py` | the mechanism: independent sub-pixel area statistics off the oracle's own trace, the `ray_subsample` ladder, `caustic_band`, `min_area_ratio`, and the PIXEL-refinement control WP-B7c did not run |
| `probe3_tail.py` | `zeta_linear_range` reported / never applied (monkey-patched to absurd values, digest compared), and the cumulative dark-side energy by fill depth |
| `probe4_bracket.py` | the refusal's denominator: how far `power_ratio` and `power_ratio_triangles` separate, and where that separation exceeds the bar |
| `probe5_oraclefloor.py` | the oracle's own convergence, energy closure and Debye `J0` expansion parameter per fixture |
| `bitid.py` | archive-to-archive byte identity, 32 fixtures, child process per tree with `lumenairy.__file__` asserted |
| `summarise.py` | joins the head and base ladders on `z` and re-derives the band |
| `run_band.sh` | runs the four ladders of one tree in parallel |

`scan_*.txt` are `probe0_scan.py` ladders at 0.02-1 um steps through the
blow-up transition -- the evidence that the ratio interval WP-B7c reports as
empty is populated, and that the accept/refuse decision is not monotone in the
output plane distance.  `pytest_*.txt` are the run logs.

JSON naming: `<probe>_<fixture>_<build>.json` with build in
`{head_win, pre_win, head_wsl}`; `joined_head_win.json` is `summarise.py`'s
output; `mutations_win.txt` is the mutation matrix for the decision tests;
`failbefore_base_win.json` is the base tree's behaviour at the refused planes.

Every run used `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1` on
the command line and `PYTHONPATH` pinned to the tree under test; each JSON
records the `lumenairy.__file__` it was taken against.
