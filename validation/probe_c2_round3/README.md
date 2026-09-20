# `probe_c2_round3` -- WP-C2 round 3 (VERIFY-WP-C2 round 2 closures)

Every probe runs with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
MKL_NUM_THREADS=1` on the command line, on **Windows py3.14.6 / numpy
2.4.4** and **WSL py3.12.3 / numpy 2.4.6**, with the tree under test pinned
through `PYTHONPATH` and `lumenairy.__file__` recorded inside each JSON.
The PRE tree is this round's own `git archive 49ddf4bd`, unpacked at
`C:/tmp/lum_c2c_pre49` (`/mnt/c/tmp/lum_c2c_pre49` from WSL).

| probe | defect | what it measures |
|---|---|---|
| `r3_apply_real_lens_wayback.py` | VR2-D1 | SHA-256 of the field `apply_real_lens` returns, on three prescriptions x four call shapes, in one process per tree |
| `r3_wayback_compare.py` | VR2-D1 | PRE vs POST: the way back byte-identical, the default moved, the `seidel_correction=False` control identical with and without the keywords |
| `r3_census_alias_hop.py` | VR2-D1 | the entry-point census read three ways (names only / + import aliases / + one private-helper hop) over an arbitrary tree, source only, nothing imported |
| `r3_history_drift_envelope.py` | VR2-D2 | the history-bundle drift against `n eps` and `2 n eps` over 90 stack / glass / radius / field-angle combinations |
| `r3_w6a2_modes.py` | VR2-D5 | the `w6_a2` second-Newton-step ratio at the shipped root and at five deliberately unconverged ones |
| `r3_mutants.sh` / `r3_mutants_*.txt` | VR2-D3, VR2-D4 | the five mutants this round's new ids have to catch, each on a fresh `git archive` of the round-3 tip |

Outputs are `<probe>_{win,wsl}.json`; the mutation runs are `.txt` pytest
tails.
