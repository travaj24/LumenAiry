# `probe_verify_sliver` — independent verification of the O-11 sliver fix

Re-measurement of every claim in
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md`, with its own fixtures
and its own scripts. Findings, verdicts and the two follow-ups that came out of
it are in `docs/audits/VERIFY_PMMSTACK_SLIVER_WALLS_2026_09_11.md`.

Nothing here imports `validation/probe_pmmstack_sliver/`. Every probe asserts
which `lumenairy.__file__` it loaded.

## How to run

Windows (this tree = the fix + the merged per-layer/mortar work):

```
cd /c/tmp/lum_vsliver
PYTHONPATH=/c/tmp/lum_vsliver OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_verify_sliver/v2_mech.py
```

WSL:

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_vsliver && \
  PYTHONPATH=/mnt/c/tmp/lum_vsliver OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ~/lumvenv/bin/python validation/probe_verify_sliver/v2_mech.py <out_wsl.json>"
```

The PRE-FIX arm is the READ-ONLY main clone; run the same script with

```
cd /c/tmp && PYTHONPATH="D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy" \
  python /c/tmp/lum_vsliver/validation/probe_verify_sliver/v1_bitid.py <out.json> "D:/Metacept"
```

(the second argument is the root the script asserts `lumenairy.__file__` starts
with — never let a two-arm comparison run twice against the same tree).

## The probes

| file | what it establishes | output |
|---|---|---|
| `v_fixtures.py` | 21 `PMMStack` fixtures of my own: shared / per-layer hw1+hw2, conical, slant, OOP tensor, in-plane xy tensor, lossy, absorbing superstrate, `solve_vs_wavelength`, `prepare()`, `stabilize='slices'`, `internal_field`, `layer_absorption`, `per_order_amplitudes`, Bragg, a taper with the snap ACTIVE and DORMANT, and `_pmm_union_grid`'s 2-tuple on six geometries | — |
| `v1_bitid.py` | sha256 over dtype/shape/buffer of every returned array; run against this tree and the main clone on both builds | `v1_bitid_{fix,main}[_wsl].json` |
| `v2_mech.py` | A: the `1/w` and `1/w²` exponents FITTED on my own fixture at 3 degrees × 4 widths, plus the `\|q\|` predictor constant; B: the O-11 self-gaps + closure + which warnings fire; C: per-layer window vs the union at 2/3/5 layers | `v2_mech[_wsl].json` |
| `v3_guard.py` | A: 120 dense δ × 3 degrees; A2: the fix's own 46-point grid; B: the false-negative walk; C: the false-positive battery; D: the M2 audit-class coated taper; E: each miss END TO END with the snapped remedy as witness; F: the quiet band's WIDTH in δ. Sections are selected by the second argument (`a,b`, `c`, `d`, `a2,e`, `f`) | `v3_guard_*.json` |
| `v4_remedy.py` | A: the prescribed `min_feature` on THREE fixtures × four degrees against the exact `δ → 0` limit; B: open item A reproduced and counted; C: a candidate fix scored | `v4_remedy.json` |
| `v5_deadband.py` | the 11,418-case two-arm gate for follow-up 1 | `v5_deadband_{fix,main}.json` |
| `v6_2d_mortar.py` | A: the two shipped 2-D claims; B: a sliver inside ONE layer's non-uniform grid; B2: the same through `add_tapered_pillar`; B3: two mortar-coupled sliver grids; C: the 1-D single-layer liner the ownership rule exempts | `v6_2d_mortar.json` |
| `v7_durability.py` | the quantity behind EVERY bar in `test_fix_pmmstack_sliver_walls.py`, both builds | `v7_durability[_wsl].json` |
| `v9_falsepos.py` | A: conjunct (b)'s premise attacked on 960 sliver-free passive stacks; B: the 648-configuration false-positive census; C: what the prescribed remedy actually does | `v9_falsepos[_wsl].json` |
| `v10_paths.py` | open item E: the onset mapped on the CONICAL and SLANT cascades against their own `δ → 0` limits | `v10_paths[_wsl].json` |
| `v11_discriminator.py` | a CANDIDATE fix for open item F, scored two-sided: does the super-unity survive the prescribed `min_feature`? | `v11_discriminator[_wsl].json` |

## Traps

* **A printed δ is not the δ.** In the hazard band the answer moves with the
  last bits of `delta`: a 5-significant-figure copy of a quiet row reads
  `R+T−1 = +1.17` where the exact float reads `+6.9e-03`. `v3_guard.py`
  section E therefore reads its candidates out of `v3_guard_ab.json`, never out
  of the log. Set `LUMV_AB_JSON` to point it at the WSL run's file.
* **`min_feature` is ABSOLUTE metres on `PMMStack` and a FRACTION inside
  `_pmm_union_grid` / `_cross_layer_sliver`.** Probes that want the raw hazard
  pass `min_feature = period * 1e-12` (below the function's own 1e-9 fractional
  dedup `tol`, so the snap branch is skipped entirely).
* **Order sets differ between geometries.** A 3-segment layer and its
  2-segment `d → 0` limit can retain different far-field order counts;
  intersect the order arrays before differencing (`np.intersect1d` +
  `np.searchsorted`), or the comparison raises a broadcast error.
* **`_warn_stack_energy` scores `max` over BOTH polarizations.** The fix's
  `p5_dense.py` scores polarization 1 only, which is why its "max \|R+T−1\|
  among correct" reads 4.125e-06 where the guard's own definition reads
  4.59e-06 on the same rows.
* **`PMM2DStackHybrid.solve()` takes no `jones=` keyword** (it always returns
  four values); `PMM2DStackPure.solve(jones=False)` returns three.
* **The main clone is READ-ONLY.** Run pre-fix arms with `cd /c/tmp` (or
  `/tmp` under WSL) so nothing is written into it, and always pass the
  `expect_root` argument.
