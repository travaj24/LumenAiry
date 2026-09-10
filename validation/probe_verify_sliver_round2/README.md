# `probe_verify_sliver_round2/` -- the INDEPENDENT verification of round 2

Re-measurement probes for the round-2 `PMMStack` sliver-guard ARBITER
(`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md`), written
without reusing that round's probes.  Findings:
`docs/audits/VERIFY_PMMSTACK_SLIVER_ROUND2_2026_09_11.md`.

Every probe asserts which `lumenairy.__file__` it imported and pins one BLAS
thread before numpy is imported.  Run with

```
cd /c/tmp/lum_vsliver2
PYTHONPATH=/c/tmp/lum_vsliver2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_verify_sliver_round2/<probe>.py
```

and on the second build with

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vsliver2 && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_vsliver2 \
  ~/lumvenv/bin/python validation/probe_verify_sliver_round2/<probe>.py"
```

| file | what it measures |
|---|---|
| `w_fixtures.py` | 31 bit-identity fixtures + the sliver family, written from scratch (a different period, wavelength, angle, duty and permittivity pair from the fix's O-11 stack) |
| `w1_bitid.py` | sha256 of every array AND the warning set, this tree vs a read-only `bb0527a` worktree; `--compare A.json B.json` diffs two runs |
| `w2_populations.py` | the trigger / closure / move populations on 5 fixtures x 150 deltas x 3 degrees = 2,250 rows, chosen to span continuity slope 0.47 .. 31.4 |
| `w3_bars.py` | the three solve-free bars: the own-scale ratio population, the anti-Hermitian deadband (200k lossless + 50k lossy directors + a gain ladder), the q-excess populations and the liner ladder |
| `w4_side_effects.py` | the arbiter's side effects: no mutation (per-attribute, against a guard-disarmed control), no recursion, cost, firing rate on 600 converged rows, the `unknown` branch |
| `w5_census.py` | my own 648-configuration false-positive box and my own 660-row false-negative grid, round 1 vs round 2, plus bit-identity of every RETURNED row |
| `w6_resonant.py` | the R2-D counter-fixture: guided-mode-resonance and Fabry-Perot stacks with `dR/dx` dialed, hunting a false refusal and a false `truncation` |
| `w7_lc_within.py` | the anisotropic classes on MY tensors, the pol-0 evidence, the within-layer liner ladder, and whether an OWNED liner can be refused through the cross-layer path |
| `w8_lc_exact.py` | the round-2 report's OWN director convention, so its stated `move` = 38.9 / 290.4 / 22.2 / 316.5 and the 0.01x / 316x pol split are checked as numbers; plus a CO-OCCURRING cross-layer sliver and broken owned liner |

The `_wsl` JSONs are the second build's.  `w1_bitid_tip*.json` are the
pre-round-2 arm, produced from a read-only `git worktree add C:/tmp/lum_prer2
bb0527a`.
