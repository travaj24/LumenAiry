# probe_verify_staggered_aniso -- INDEPENDENT verification of Stage A

Adversarial verification of the in-plane anisotropic staggered 2-D PMM
(branch `feat/pmm2d-staggered-anisotropic`), written 2026-09-09 by an agent
that did not build it.  Findings and verdicts:
`docs/audits/VERIFY_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md`.

Nothing here re-uses the build's probes.  Every script asserts
`lumenairy.__file__` is under the worktree (the main-clone arm asserts the
`D:/` path instead), and every one of them re-MEASURES rather than reading.

Run any of them as

```
cd /c/tmp/lum_aniso && PYTHONPATH=/c/tmp/lum_aniso OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python validation/probe_verify_staggered_aniso/<probe>.py [args]
```

| probe | what it measures | wall |
|---|---|---|
| `v1_scalar_identity.py <libroot> <out.json>` | the shipped SCALAR path across two CODE VERSIONS: 5 efficiency fixtures + 3 assembled-operator sets + 1 pure-stack Jones fixture, hashed.  Run once with the worktree on `PYTHONPATH` and once with the read-only main clone `D:/.../Lumenairy`, then diff the JSONs. | 8 s x2 |
| `v2_gates.py g1 g3 g4 g5 g6 g7 g8 g9` | gates G1, G3-G9 re-measured on FRESH fixtures (different tensors, periods, wavelengths, angles, modal counts) -- the orders of magnitude of the build doc's tables, from a different corner of the parameter space | 100 s total |
| `v3_jones.py` | cross-engine COMPLEX Jones (magnitudes and phases) vs `berreman_jones_1d`, `pmm_jones_2d` and `rcwa_jones_2d`, with the CONJUGATED and TRANSPOSED arms as fail-befores | 45 s |
| `v3b_jones_power.py` | `|jones|^2` <-> order-0 efficiency identity including the incident longitudinal component, at normal / oblique / strongly conical incidence, with the transposed-column arm as the fail-before | 12 s |
| `v4_g2_li2003.py` | **G2 reconciliation**: Li, J. Opt. A 5, 345 (2003), Example 1 + Table 1 -- the ORIGINAL of Granet's Fig. 4 grating.  M ladder, the pillar/host swap against Li's second row, position invariance, the hybrid engine on the same reading, and the readings the build tried as controls | 15 s |
| `v5_tripwire.py` | the lossless-closure tripwire: the (eps_pillar, M) sweep behind its 1.82x fail-before margin, plus two ENGINEERED breaks (a rescaled modal `V`, and a non-unitary `lam`) | 40 s |
| `v6_doc_numbers.py` | re-runs the BUILD DOC's own tables T3-T9 on the build's own fixtures, by importing the shipped test module's helpers ("right conclusion, wrong numbers" check) | 60 s |
| `v7_thread_spread.py <tag>` | every bar quantity in the test file, measured at `OPENBLAS_NUM_THREADS` 1 and then 4 -- a real (partial) probe of the last-bit envelope TESTING_STANDARDS rule 5 asks for.  Run twice with different caps and diff | 25 s x2 |

`out_*.json` are the recorded readings of the 2026-09-09 run on
tesla-ryzen (Windows 11, CPython 3.14.6, NumPy 2.4.4, scipy-openblas
0.3.31.188.0).  `regress_staggered.log` is the shipped-suite regression run
whose zero tripwire firings are the guard's must-not-fire evidence.
