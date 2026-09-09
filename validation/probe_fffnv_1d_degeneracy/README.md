# probe_fffnv_1d_degeneracy -- Task H measurements (2026-09-10)

The probes behind `docs/audits/FIX_WOOD_LIST_AND_FFFNV_2026_09_10.md` Task H:
why `test_fff_nv_stripe_reduces_to_rigorous_1d` was a per-build test, and what
made its fixture one.

Each takes ONE argument, the lumenairy root the arm must come from, and
asserts `lumenairy.__file__` against it.

```
PYTHONPATH=/c/tmp/lum_wood OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_fffnv_1d_degeneracy/h1_ladder.py \
  /c/tmp/lum_wood

wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_wood && PYTHONPATH=/mnt/c/tmp/lum_wood \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ~/lumvenv/bin/python validation/probe_fffnv_1d_degeneracy/h1_ladder.py \
  /mnt/c/tmp/lum_wood"
```

| probe | what it measures |
|---|---|
| `h1_ladder.py` | the rigorous 1-D lossless-closure defect for n_orders 5..61 on the shipped fixture -- the ladder the test used to scan |
| `h2_isolate.py` | which property breaks the closure: rotated vs diagonal vs isotropic tensor, rotation angle, contrast |
| `h4_cond.py` | Hermiticity of the Li in-plane operator and the conditioning of the layer eigenproblem (rules amplification-by-conditioning OUT) |
| `h5_where.py` | uniform vs patterned, depth, half-spaces, other tensors -- isolates the index coincidence |
| `h6_detune.py` | the 1/detune law: detune the groove / the ordinary index / the substrate and watch the defect collapse to 1e-13 |
| `h7_candidate.py` | candidate non-degenerate grooves: closure ladder + the two ratios the test asserts |
| `h8_refconv.py` | convergence and cost of the 1-D reference vs n_orders (sets `_ONED_REF_ORDERS`) |
| `h9_testquant.py` | the exact quantities the hardened tests assert, clean and coincident arms |
