# probe_wood_list -- Task G measurements (2026-09-10)

The probes behind `docs/audits/FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md` Task G
(one Wood-anomaly permittivity list for the pure staggered 2-D PMM).

Each takes ONE argument: the lumenairy root the arm must come from.  Every
probe asserts `lumenairy.__file__` against it, so a pre/post comparison cannot
silently read the wrong tree.

```
# post-fix arm (this branch)
PYTHONPATH=/c/tmp/lum_wood OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_wood_list/g1_hash.py /c/tmp/lum_wood

# pre-fix arm (a clone at the base commit, read-only)
PYTHONPATH=<main-clone> OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_wood_list/g1_hash.py <main-clone>
```

| probe | what it measures |
|---|---|
| `g1_hash.py` | sha256 of R/T plus every nudged wavelength, 11 scalar + tensor fixtures (Gate 1: `diff` the two arms' JSON) |
| `g2_oncut.py` | the verify report's reproducer: scalar entry vs tensor entry on a LAYER cut-off |
| `g2b_stack.py` | the same comparison inside ONE entry point (`PMM2DStackPure`, scalar cell vs `e*I`) |
| `g2c_uniform.py` | the nudge itself, spied on `_grazing_safe_wavelength`, incl. list length |
| `g2d_uniform2.py` | the `kind="uniform"` scalar-layer branch, on a layer cut-off no half-space shares |
| `g3_warn.py` | the Rayleigh-cut-off WARNING, fixture by fixture (Gate 3) |
| `g4_size.py` | the nudge's size and consequence; the bar derivation for the consequence test |
| `g5_perf.py` | deduplication: 46.9 ms/call raw (64x64) vs 0.02 ms, identical wavelength |
