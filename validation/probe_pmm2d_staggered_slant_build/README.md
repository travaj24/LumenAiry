# Phase-D SLANT -- the BUILD's own measurements

Every script here measures the **SHIPPED** library
(`lumenairy.elements.pmm.twod_staggered` / `stack2d_pure`), never a prototype:
the build transplanted the prototype's formulation, so the numbers a test bar is
derived from must come out of the code that ships.

* Formulation and the GO decision:
  `docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md`
  (prototype in `validation/probe_pmm2d_staggered_slant/`).
* The build, both builds' tables and the open items:
  `docs/audits/BUILD_PMM2D_STAGGERED_SLANT_2026_09_10.md`.

`_lib.py` raises on import if `lumenairy` resolves outside the worktree.

## Running

Windows:

```
cd validation/probe_pmm2d_staggered_slant_build
set PYTHONPATH=C:\tmp\lum_slant
set OMP_NUM_THREADS=1 & set OPENBLAS_NUM_THREADS=1 & set MKL_NUM_THREADS=1
set SLANT_BUILD_TAG=win
python g1_hash_null.py
```

WSL (the second build; py3.12 / numpy 2.4.6 / scipy-openblas SkylakeX):

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_slant/validation/probe_pmm2d_staggered_slant_build \
  && SLANT_BUILD_TAG=wsl PYTHONPATH=/mnt/c/tmp/lum_slant \
     OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/lumvenv/bin/python g1_hash_null.py"
```

Each script writes `results/<name>_<SLANT_BUILD_TAG>.json` with the build pin
(interpreter, numpy, scipy, platform) embedded.

## The scripts

| script | gates | ~wall (WIN / WSL) |
|---|---|---|
| `g1_hash_null.py` | **B1** slant-0 bit-identity; **B2** the uniform-layer null + its M-ladder; **B3** the frame-anchor phase, three arms | 24 s / 23 s |
| `g2_dispersion.py` | **B4** sheared-frame dispersion vs the EXACT quartic roots, four gauge/shift arms, both ablations, the sum-of-roots discriminator, the M-ladder | 23 s / 23 s |
| `g3_stripe.py` | **B5** the y-uniform stripe vs `pmm_efficiency_1d_slanted` per order with the SIGN arm; **B6** slant x out-of-plane vs `pmm_jones_1d_slanted` + the hybrid's refusal | 64 s / 62 s |
| `g4_census.py` | **B7** the spurious census; **B8** cascade vs depth; **B9** the layer split; **B10** the no-floor property | 16 s / 15 s |
| `g5_parity_refusals_wood.py` | **B11** the parity refusal (with the geometry-shim fail-before), all the refusals and the accepted shapes, and the WOOD-list decision | 14 s / 2 s |
| `g6_cost_pillar.py` | COST at campaign sizes + the 2-D pillar three ways on `Nx = 8 / 12` (the expensive one -- ~25 min) | -- |
| `g7_testfixtures.py` | the SAME quantities at the sizes `tests/unit/test_pmm2d_staggered_slant.py` actually runs, so every bar in that file has a measured cross-build gap | 48 s / 45 s |

## Two traps worth knowing

1. **Do not monkeypatch `twod_staggered._slant_is_zero` to force the parity
   gauge onto a slanted pencil.** That same name is what
   `Granet2DTransverseE.__init__` reads to decide whether to apply the shear at
   all, so the patched arm silently builds the VERTICAL pencil and measures
   nothing. Use the geometry shim (`_Shim` in `g5`, `_GeometryShim` in the test
   file).
2. **The frame-anchor phase's formula is stated in the INTERNAL shear
   `t = -slant`.** Summing the PUBLIC slant instead inverts it, and the arm
   that then looks correct is the one that is twice as wrong as no correction
   at all. `g1`'s three-arm block is what catches it.
