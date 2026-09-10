# probe_fix_lens_5440 -- the seven follow-ups of the 5.44.0 traced-lens verification

Measurement for D1-D7 of
`docs/audits/VERIFY_LENS_BANDED_COMPLEX64_2026_09_10.md`; findings and tables
in `docs/audits/FIX_LENS_5440_FOLLOWUPS_2026_09_11.md`.

Every script prints `lumenairy.__file__` / `__version__` first (`_fixp.banner`,
which REFUSES to run outside the tree it names), and the JSON it writes carries
them, so an arm cannot be mistaken for another.  The fixtures of
`../probe_verify_lens_5440/_fix.py` are reused where an arm has to be
comparable with the verification's; that directory's own probes are re-run
UNCHANGED for every bit-identity claim, so identity is checked against the
verification's fixtures rather than against a re-derivation.

## Run

```
cd /c/tmp/lum_lensfix
PYTHONPATH=/c/tmp/lum_lensfix OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fix_lens_5440/<script>.py \
  validation/probe_fix_lens_5440/results/<out>.json

# second build
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_lensfix && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONPATH=/mnt/c/tmp/lum_lensfix \
  ~/lumvenv/bin/python validation/probe_fix_lens_5440/<script>.py <out>.json"
```

## Scripts

| script | follow-up | what it measures |
|---|---|---|
| `_fixp.py` | -- | banner with a tree assertion, array hasher, free-RAM reading |
| `p1_d2_transient.py` | D2 | whole-call `tracemalloc` peak of ONE `carrier_referenced_envelope` / `_reconstruct` call at N=2048 / 4096, scalar and astigmatic carrier, complex64 and complex128, with the returned field hashed so the fix can be shown value-preserving |
| `p2_d1_attr.py` | D1 | `w.filename` / `w.lineno` of every ray-density notice with all five thresholds driven over, at band heights 0 / 32 / 7, plus the field hash |
| `p3_d6_time.py` | D6 | wall time (best of N), `InverseCharacteristic.eval_into` call and channel counts, `map_coordinates` counts, and the field hash, for `{screen, ray_density} x {whole-grid, AUTO-banded, AUTO-banded with inverse_map=False}`.  The third arm is the CONTROL: it forces 5.44.0's banded call back onto the incumbent v5.43.0 used to select silently |
| `p6_d6_stages.py` | D6 | the same six routes split one level finer -- `eval_into`, **`domain_mask`** (calls AND pixels tested, in grids), `build_inverse_map`, `map_coordinates`, rest.  This is what shows the evaluator route's price is the domain test and not the 7/4 evaluation, and that the banded ray-density branch was testing 2.00 grids where the whole-grid arm tests 1.00 |
| `p4_d3_fft.py` | D3 | a real two-group traced carrier chain WITH an exact focus readout (`final_leg='exact'` -- the plain chain never reaches the crop) in three arms: complex64 with the shipped single-precision transform pair, complex64 with the pair forced to complex128 and narrowed once, and complex128; plus the crop's own cost at `n_fine` 256 .. 2048 |
| `splice_durations.py` | -- | line-based splice of pytest-split's measured durations into `.test_durations`, retained entries byte-identical (CHORE_TEST_HYGIENE_2026_08_16 (e)) |
| `run_d6.sh` / `run_d6_after.sh` | D6 | the sweep drivers (both builds; before / after the pass-2 mask fix) |

`results/` holds the JSON, the sweep logs and `_testfiles.txt` (the file list
of the final pytest run).  Re-runs of the verification's own probes are named
`v*_after.json` there.
