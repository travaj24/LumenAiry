# probe_verify_lens_5440 -- independent verification of the two 5.44.0 traced-lens changes

Adversarial re-measurement of the two `### Changed` entries in the 5.44.0
CHANGELOG block (library commit `4e8ea24`, released at `50824e9`):

1. the traced ROW-BAND assembly now also serves `amplitude_model='ray_density'`
   and the inverse-characteristic evaluator;
2. complex64 THROUGH the carrier chain.

Nothing here reads the shipped tests' conclusions: every fixture is built in
`_fix.py` (a different singlet, glass, wavelength, grid and carrier from the
ones `tests/unit/test_banded_ray_density_and_inverse_map.py` uses), except
`v9` and `v10`, which deliberately re-run the SHIPPED fixtures so the recorded
constants can be checked against what they actually measure.

Findings and verdicts: `docs/audits/VERIFY_LENS_BANDED_COMPLEX64_2026_09_10.md`.

## Arms

| arm | tree | assert |
|---|---|---|
| WITH | `C:/tmp/lum_vlens` @ `50824e9` (5.44.0) | `lumenairy.__file__` printed by `_fix.banner()` |
| WITHOUT | `C:/tmp/lum_v5430` @ `v5.43.0` | same |
| second build | WSL, `~/lumvenv` (py 3.12.3 / numpy 2.4.6) | same |

Every script prints `lumenairy.__file__` and `__version__` first; the JSON it
writes carries them too, so an arm can never be mistaken for the other.

## Run

```
# WITH arm
cd /c/tmp/lum_vlens
PYTHONPATH=/c/tmp/lum_vlens OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_verify_lens_5440/<script>.py \
  validation/probe_verify_lens_5440/results/<out>.json

# WITHOUT arm (same script, other tree)
cd /c/tmp/lum_v5430
PYTHONPATH=/c/tmp/lum_v5430 ... python \
  /c/tmp/lum_vlens/validation/probe_verify_lens_5440/<script>.py <out>.json
```

## Scripts

| script | what it measures |
|---|---|
| `_fix.py` | fixtures: two singlets (a moderate and a deliberately strong one), Gaussian / spherical-carrier fields, a hasher, a cold-cache runner |
| `v1_without_identity.py` | 14 fixtures x {`sag_chunk_rows=0`, `None`} hashed -- the non-banded traced calls that must not move across the release |
| `v2_banded_claim.py` | band heights {0, 7, 32, 128, N} (and {1, 3} in `--fold` / `--medianbite`) pairwise on field + diagnostics + warnings, both inversion routes.  `--forced` drives the three ray-density self-checks and the D9 origin verdict over their thresholds; `--fold` fires the caustic census; `--medianbite` puts the census MEDIAN into every pixel (floor 1.5x median) so a band-order difference in the median would be a FIELD difference |
| `v3_determinism.py` | one field hash per case at `OPENBLAS_NUM_THREADS` 1 / 2 / 4, each in its own subprocess (D14/D15) |
| `v4_evalcount_memory.py` | `InverseCharacteristic.eval_into` call / channel counts (the 7/4 claim) and whole-call `tracemalloc` peaks; `--peaks-only` for large N |
| `v5_c64_ladder.py` | the complex64 phasor error vs the phase argument, 2e+02 .. 2e+05 rad, per helper, against a float32-ARGUMENT control |
| `v6_c64_chain.py` | dtype at every hand-off in a real two-group chain, a six-leg synthetic chain, the crop's double-rounding, and the 4e-05 energy honesty bar |
| `v7_c64_memory.py` | does requesting complex64 actually save memory -- peak and the caller frame of every surviving full-grid complex128 phasor |
| `v8_failbefore.py` | banded vs whole-grid at the shipped default on both builds (`--auto` = the true `sag_chunk_rows=None` default at N=4096) |
| `v9_durability.py` | every constant the two new test files assert, re-measured on the SHIPPED fixtures (run on Windows and WSL) |
| `v10_s10_route.py` | the CHANGELOG's named 2.19e-02 S10 difference, reproduced on both builds |
| `v11_side_effects.py` | warning attribution (`stacklevel`), the D9 fraction, and wall time without `tracemalloc` |
| `v19_fold_warn_attr.py` | attribution of the fold-caustic warning after it moved into a shared closure (companion to `v11`) |
| `v12_d9_sum_order.py` | the one non-bit-identical quantity: the D9 origin sums, band-accumulated vs whole-grid |
| `v13_sign_scan.py` | the two-row-halo `det J` sign scan, both algorithms transcribed and run against each other on 489 fields x 12 band heights, engineered to put flips ON band boundaries |
| `v18_c15_probe_on_band.py` | whether the niche-C15 private `_imap_out['probe_rc']` diagnostic is filled on a banded call |
| `v14_c128_chain_identity.py` | the complex128 carrier chain, readout and crop hashed for the cross-release identity claim |

| `v20_whole_grid_rd_time.py` | isolates the whole-grid ray-density call's wall time in a fresh process (the V15 cross-build discrepancy turned out to be a run-ordering artifact) |

`results/` holds the JSON and the pytest / probe logs.
