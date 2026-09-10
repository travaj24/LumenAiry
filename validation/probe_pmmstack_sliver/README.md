# `probe_pmmstack_sliver` — the O-11 near-coincident-wall (sliver) defect

Reproduction, mechanism and remedy for open item **O-11** of
`docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md`: two adjacent
`PMMStack` layers whose wall sets differ by `delta` of the period put a SLIVER
element of exactly that width on the shared union grid, and past a
degree-dependent onset the cascade returns a deterministic wrong answer.

The fix, every bar's derivation and the cross-build tables are in
`docs/audits/FIX_PMMSTACK_SLIVER_WALLS_2026_09_11.md`.  The shipped guard lives
in `lumenairy/elements/pmm/stack.py` (`_cross_layer_sliver`,
`_stack_provably_passive`, `_sliver_refusal`, `PMM_SLIVER_GUARD`).

## How to run

Windows:

```
cd /c/tmp/lum_sliver
PYTHONPATH=/c/tmp/lum_sliver:/c/tmp/lum_sliver/validation/probe_pmmstack_sliver \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_pmmstack_sliver/p1_repro.py
```

WSL:

```
wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_sliver && \
  PYTHONPATH=/mnt/c/tmp/lum_sliver:/mnt/c/tmp/lum_sliver/validation/probe_pmmstack_sliver \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ~/lumvenv/bin/python \
  validation/probe_pmmstack_sliver/p1_repro.py"
```

`p2`, `p3`, `p8`, `p9`, `p10` import `p1_repro` / `p2_mech` / `bitid_fixtures`,
hence the probe directory on `PYTHONPATH`.

## The probes

| file | what it establishes | output |
|---|---|---|
| `p1_repro.py` | REPRODUCES O-11 exactly as `f5f_attrib.py` did (oracle arm only): the degree-12-vs-14 self-gap, the shift from the `delta -> 0` reference, and — new — the stack's OWN energy closure | `p1_repro.json` |
| `p2_mech.py` | MECHANISM: union cell count and narrowest cell, `k0 J`, `cond(S0)`, `|Kx^2|`, the modal `|q|max` per layer, and the interface solve's conditioning and largest S-matrix entry | `p2_mech.json` |
| `p3_onset.py` | ONSET MAP over `delta` = 3e-3..3e-6 x degree 8/12/14/16/20 with the wall-snap disabled | `p3_onset.json` |
| `p4_census.py` | whether the SHIPPED T3-4 instruments (`n_grow`, `n_grow_post`, margin, `q_excess`) see it.  They do not separate | `p4_census.json` |
| `p5_dense.py` | DENSE 46-point sweep on three degrees: the right/wrong populations against every candidate discriminator.  This is where the guard's bar is calibrated | `p5_dense.json` |
| `p6_warn.py` | which shipped WARNINGS fire.  Corrects O-11's "energy-invisible" reading: the 1-D stack's own `R+T` reads 2.17 / 3.61 / 23.4 and it warns every time | `p6_warn.json` |
| `p7_2d.py` | the 2-D stacks (`PMM2DStackHybrid`, `PMM2DStackPure`) on the same hazard | `p7_2d.json` |
| `p8_guard.py` | the guard TWO-SIDED: refused inside the band, bit-identical outside, with `PMM_SLIVER_GUARD = False` as the pre-fix arm | `p8_guard.json` |
| `p9_remedy.py` | the PRESCRIBED `min_feature` scored against the exact `delta -> 0` reference | `p9_remedy.json` |
| `p10_bitid.py` + `bitid_fixtures.py` | 18 shipped-fixture hashes, run against this tree and against the read-only main clone | `p10_bitid_fix.json`, `p10_bitid_main.json` |

## Traps

* **`f5f_attrib.py` sets `warnings.simplefilter("ignore")` at module scope.**
  That is why O-11 was logged as silent: the 1-D stack warns
  `energy not conserved (max R+T = 2.17 > 1)` on every wrong row.  The
  1.6e-08 closure in the audit table is the 2-D MORTAR arm's, not the
  oracle's.
* **The library-default `min_feature` is `period * 1e-5`, and `delta` in this
  fixture is a FRACTION of the period**, so `delta = 1e-5` sits exactly on the
  snap threshold.  The float comparison then merges ONE of the two wall pairs
  and leaves the other — an asymmetric geometry nobody asked for.  Probes that
  want the raw hazard pass `min_feature = period * 1e-10` (below
  `_pmm_union_grid`'s own 1e-9 fractional dedup tol), which disables the snap.
* **`layer_grids='per-layer'` is NOT a second opinion on a 2-layer stack.**  At
  `window_halfwidth = 1` every window is the whole stack, so it rebuilds the
  same union grid and returns the same 16 digits.  It only helps above
  `2 * window_halfwidth + 1` layers.
