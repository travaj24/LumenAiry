# `probe_pmmstack_sliver_round2` -- the O-11 sliver guard, ROUND 2

Every bar in `docs/audits/FIX_PMMSTACK_SLIVER_WALLS_ROUND2_2026_09_11.md` is
read off one of these scripts' JSON.  They are the round-2 counterpart of
`validation/probe_pmmstack_sliver/` (the round-1 fix) and
`validation/probe_verify_sliver/` (the verification that refuted its margins in
both directions), and they reuse those campaigns' fixtures and grids so the
populations are comparable row for row.

Run each from the repo root, one BLAS thread, with the worktree on
`PYTHONPATH`; every script asserts which `lumenairy.__file__` it imported and
writes a JSON next to itself (`*_wsl.json` for the WSL arm):

```
cd /c/tmp/lum_sliver2
PYTHONPATH=/c/tmp/lum_sliver2 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_pmmstack_sliver_round2/r1_populations.py
```

| script | what it measures |
|---|---|
| `r_fixtures.py` | the shared O-11 fixture, its exact `delta -> 0` reference, the unguarded solve, the continuity classification, and the two move statistics (polarization 1, which is the campaign's `err` convention, and BOTH polarizations, which is what the library computes) |
| `r1_populations.py` | both populations on the verification's 120-delta x 3-degree grid, the arbiter's decision on every row, and the trigger ladder 1e-2 .. 1e-4 |
| `r2_falseneg.py` | the FALSE-NEGATIVE census on the 60-delta x 5-degree quiet grid -- round 1's misses, and which of them round 2 catches |
| `r3_trigger.py` | the trigger bar as a FAMILY property: both populations on THREE fixtures (O-11, visible, telecom; continuity slopes 1.15 / 4.44 / 1.04) |
| `r4_falsepos.py` | the FALSE-POSITIVE census on the verification's 648 realistic staircase configurations, plus the arbiter's cost and how often it fires |
| `r5_within_layer.py` | the WITHIN-LAYER liner ladder (V-6) and the two populations of the q-excess predictor the warning is barred on |
| `r6_ratio.py` | conjunct (a)'s ratio bar: the M2 coated-taper class, whether a ratio-12 cell carries the defect, and the ordinary non-conforming population |
| `r7_anisotropic.py` | the LIQUID-CRYSTAL / anisotropic class (V-5): in-plane and out-of-plane rotated uniaxial directors, gyrotropic, and a NON-Hermitian control |
| `r8_bitid.py` | bit-identity of the round-1 18 + the verification's 21 fixtures against the pre-round-2 tip |
| `r9_paths.py` | the arbiter on the paths the verification could not check: the wavelength sweep, `prepare()`, conical, slant, `stabilize='slices'` |

`r1`-`r6` were first run against the PRE-change library (the arbiter
implemented inside the probe) so the bars could be chosen from the measurement
rather than assumed; they are re-run against the shipped library and read the
same numbers, which is the two-sided part.  `r7`-`r9` need the shipped code.
