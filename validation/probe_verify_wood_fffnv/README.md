# probe_verify_wood_fffnv -- INDEPENDENT verification, 2026-09-10

Re-measurements behind `docs/audits/VERIFY_WOOD_LIST_AND_FFFNV_2026_09_10.md`,
the adversarial verification of branch `fix/wood-list-fffnv` (Task G = one
Wood-anomaly permittivity list for the pure staggered 2-D PMM, Task H = the
`fff_nv` 1-D reduction fixture).  Fixtures are the verifier's own, not the
builder's, except `v3_builder_table.py`, whose whole purpose is to re-run the
builder's reported geometry and check their numbers.

Every probe takes the lumenairy root its arm must come from as `argv[1]` and
asserts `lumenairy.__file__` against it, so a pre/post comparison cannot
silently read the wrong tree.

```
# post-fix arm (this worktree)
PYTHONPATH=/c/tmp/lum_vwood OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_verify_wood_fffnv/v1_bitid.py \
  /c/tmp/lum_vwood out_post.json

# pre-fix arm: the read-only main clone at fb3fd93
PYTHONPATH=<main-clone> ... python <probe> <main-clone> out_pre.json

# WSL arm
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vwood && PYTHONPATH=/mnt/c/tmp/lum_vwood \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/lumvenv/bin/python <probe> /mnt/c/tmp/lum_vwood <tag>"
```

| probe | what it measures |
|---|---|
| `v1_bitid.py` | 14 OWN fixtures (TE/TM/oblique/conical/lossy/rect-periods/promotion/tensor/5 stacks): sha256 of R, T, Jones + every `_grazing_safe_wavelength` call's in/out wavelength and list length.  `diff` the two arms' JSON |
| `v2_oncut.py` | the on-cut-off reproducers, constructed through the public API: a LAYER cut-off no half-space shares, the same 1e-9 below it and far off, a HALF-SPACE cut-off (must be unchanged), the uniform-scalar-vs-uniform-tensor pair, and the guard's own nudge factor + trigger-window scan |
| `v3_builder_table.py` | the builder's G.1 / G.4 / G.7 rows on THEIR geometry, so their reported numbers can be checked digit for digit |
| `v4_warn.py` | the Rayleigh cut-off WARNING over 12 fixtures, plus an ADVERSARIAL scan of the 1e-4 warning boundary on a geometry that also sits on a layer cut-off |
| `v5_ladder.py` | the rigorous 1-D lossless-closure ladder (n_orders 11..41) for the `fff_nv` stripe at four groove permittivities |
| `v6_mech.py` | the MECHANISM: Hermiticity of the Li in-plane operator, `cond(W)`, `cond([W; V])`, `cond(a + b)` at BOTH interfaces, the layer<->region modal-eigenvalue gap and the count of exactly degenerate modes, and the defect-vs-detune law |
| `v7_bars.py` | every numeric bar asserted in `tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py`, measured |
| `v8_pmm1320.py` | `tests/unit/test_v5_20_13_pmm_jones_2d_fff_nv.py`'s RCWA and PMM ladders at the coincident and the detuned groove, plus that file's other bars |

`_out_v*_<tag>.json` are the recorded readings (`win1` = Windows py3.14.6 /
numpy 2.4.4 at 1 BLAS thread, `win4` = the same at 4, `wsl` = WSL py3.12.3 /
numpy 2.4.6 at 1).
