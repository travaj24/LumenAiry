# `probe_fix_sliver_round3/` -- the RELATIVE closure (verification defect D-5)

Measurement probes for the round-3 repair of the `PMMStack` sliver guard's
ARBITER.  Round 2 asked the re-solve on the prescribed `min_feature` grid to
reach an ABSOLUTE `_SLIVER_ATTRIB_CLOSURE` = 1e-5; on a stack whose
SLIVER-FREE truncation super-unity already sits above that bar the criterion
can never be met, so the arbiter returns a wrong answer as `truncation`
(`docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md`, defect D-5).
Round 3 makes the closure RELATIVE, keeping the absolute value as the lower
arm of a `max`.  Findings:
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND3_2026_09_11.md`.

Every probe asserts which `lumenairy.__file__` it imported and pins one BLAS
thread before numpy is imported.  Run with

```
cd /c/tmp/lum_sliver3
PYTHONPATH=/c/tmp/lum_sliver3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_sliver_round3/<probe>.py out.json
```

and on the second build with

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_sliver3 && \
  PYTHONPATH=/mnt/c/tmp/lum_sliver3 \
  ~/lumvenv/bin/python validation/probe_fix_sliver_round3/<probe>.py out.json"
```

| file | what it measures |
|---|---|
| `f_fixtures.py` | the three families and the shared helpers.  The two ARBITER criteria are re-implemented here as pure functions of the recorded `(worst, su_snapped, move, w_wide)`, so one run scores BOTH the round-2 and the candidate round-3 verdict and the round-2 column can be checked against the library on a tree that still carries round 2 |
| `s1_census.py` | the 648-configuration false-positive census box and the 660-row false-negative grid, with the DROP factor added to every arbitrated row and both analytic verdicts alongside the library's actual decision |
| `s2_d5.py` | the three D-5 rows and one CORRECT control row: the sliver-free degree ladder, the drop factor, `move / w_wide`, the error before and after the snap, and what the library does |
| `s3_dropgap.py` | the DROP-factor populations over three families -- five ordinary staircases (continuity slopes 0.47 .. 31.4), five D-5 mounts, and four points on a guided-mode resonance flank where `dR/d(duty)` is 128-174 -- plus the FLIP census: which rows change verdict between round 2 and round 3, at five candidate fractions |
| `s4_lower_envelope.py` | the CORRECT population's drop envelope as a FAMILY property: a wide staircase box (three periods, two wavelengths, two superstrate indices, three lossy substrates, five angles, three degrees, three ridge permittivities, two slice counts, four wall steps) plus the D-5 devices themselves at coarse wall steps |

## Outputs committed here

| JSON | build | arm |
|---|---|---|
| `s1_census_BEFORE_win.json` / `_wsl.json` | Windows / WSL | the tree still carrying ROUND 2 -- the baseline, and the check that the analytic round-2 verdict equals the library's decision on all 291 + 447 arbitrated rows |
| `s1_census_AFTER_win.json` / `_wsl.json` | Windows / WSL | the shipped round-3 tree |
| `s2_d5_BEFORE_win.json`, `s2_d5_win.json` / `s2_d5_wsl.json` | Windows / WSL | the D-5 rows before and after |
| `s3_dropgap_win.json` / `_wsl.json` | Windows / WSL | version-independent (the guard is disarmed throughout and the verdicts are scored analytically) |
| `s4_lower_win.json` / `_wsl.json` | Windows / WSL | version-independent, same reason |
