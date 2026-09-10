# `validation/probe_verify_slant/` -- the INDEPENDENT verification probes

Scripts backing `docs/audits/VERIFY_PMM2D_STAGGERED_SLANT_2026_09_10.md`, the
adversarial verification of the native constant-shear SLANT on the PURE
staggered 2-D PMM (`feat/pmm2d-staggered-slant`, merged at `8b9af801`).

They are NOT the build's probes re-run.  Every fixture, oracle and reference
construction here is this verification's own, and the two claims the build
states as cross-engine facts (the SIGN and the FRAME-ANCHOR PHASE) are re-asked
in forms that do not depend on any other engine's convention: a `np.roll`
staircase for the geometry, and a lab-referenced staircase for the phase.

| script | what it measures |
|---|---|
| `_lib.py` | arm detection (`tip` = `C:/tmp/lum_vslant`, `base` = the 2efc7a2 main clone), JSON dump, hashing.  Every run stamps `lumenairy.__file__`; nothing is labelled by a flag |
| `v1_without_identity.py` + `v1_compare.py` | 18 fixtures x 7 spellings of "no slant" x 2 checkouts, sha256 on R/T/Jones -- the WITHOUT arm |
| `v2a_geometry_sign.py` | what the public `slant` DRAWS, by `np.roll` staircase, in both engines |
| `v2b_oracle_sign.py` | the stripe against `pmm_efficiency_1d_slanted` per order, both signs, plus the LOSSLESS TRAP (both signs' energy closure) |
| `v2c_chain_sign.py` | the sign chain to geometry: the 1-D `add_sheared_grating` centre law vs an explicit-centre staircase; 2-D vs 1-D cell-index and order-direction consistency; the hybrid ladder |
| `v3_frame_anchor_phase.py` | the anchor derived here and walked three ways (uniform null, pure staircase, chiral cell vs hybrid) |
| `v3b_anchor_patterned.py` | the anchor on a PATTERNED layer against a REFINABLE lab-referenced oracle (a 60-pixel hybrid staircase) |
| `v4a_null.py` | the uniform null: 5 tensors x 6 slants (to 60 deg) x 3 mounts, + the M-ladder |
| `v4b_dispersion.py` | the sheared-frame spectrum vs the EXACT quartic roots, four gauge arms, three ablations |
| `v4c_census.py` | the forward/backward split computed HERE (pre-rebalance), spectral radius, growth; incl. a LOSSY cell and a METAL |
| `v5_parity.py` | the parity question: structural residual, forced reduction, and the END-TO-END forced arm |
| `v6_refusals.py` | every documented refusal + the accepted shapes + the sneak attempts |
| `v6b_readings.py` | are `_check_stack_slant`'s two admitted frame-offset readings EXACT?  (an M-ladder on the one comparison that pins both) |
| `v7_wood.py` | the Wood-list decision, walked onto each candidate cut-off |
| `v8_hybrid_anchor.py` | the DEFECT this verification found on the OTHER engine, pinned PER ORDER |
| `v9_durability.py` | re-measures every numeric bar in `tests/unit/test_pmm2d_staggered_slant.py` on the file's own fixtures; run on both builds |

## Running

```
# Windows arm
cd validation/probe_verify_slant
PYTHONPATH=C:/tmp/lum_vslant OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -W ignore -u v9_durability.py

# WSL arm (writes results/<name>_tip_wsl.json)
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vslant/validation/probe_verify_slant && \
  PYTHONPATH=/mnt/c/tmp/lum_vslant VSLANT_TAG=wsl OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 ~/lumvenv/bin/python -W ignore -u v9_durability.py"

# the WITHOUT arm, against the main clone at 2efc7a2 (READ-ONLY)
PYTHONPATH="D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy" \
  python v1_without_identity.py && python v1_compare.py
```

Results land in `results/<probe>_<arm>_<tag>.json` with the interpreter, numpy,
scipy and thread caps recorded in `_env`.

## Two traps these probes hit, recorded so the next reader does not

* **A lexicographic sort is not a spectrum comparison.**  `v5`'s first
  spectrum-gap metric sorted both spectra with `np.sort_complex` and differenced
  them elementwise.  Most of a lossless region's eigenvalues are purely
  imaginary, so their real parts are `+/-` round-off and the sort scrambles
  them: two IDENTICAL spectra reported a gap of `1.2e+01`.  The VERTICAL
  control -- where both branches are shipped code and must agree -- is what
  caught it.  The metric is a Hausdorff distance now.
* **A mirror-symmetric-up-to-translation cell hides the slant sign in the
  zeroth-order Jones.**  `v2a`'s first fixture was a single block, which is its
  own mirror image up to a translation, so the `+t` and `-t` structures are
  mirror-related and their zeroth-order reflection Jones agree to `5e-15` while
  their per-order `T` differs by `2.6e-01`.  Use an x-ASYMMETRIC profile
  (three levels), and read PER-ORDER quantities.
