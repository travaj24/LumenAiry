# Probes -- in-plane anisotropy for the pure staggered 2-D PMM (Stage A)

Every measurement quoted in
`docs/audits/BUILD_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` and in the bar
comments of `tests/unit/test_pmm2d_staggered_anisotropic.py` comes from one of
these scripts.  They are small and self-contained; none writes data files.

Run any of them from the worktree root:

```
PYTHONPATH=/c/tmp/lum_aniso OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python validation/probe_pmm2d_staggered_aniso/<file>.py
```

Each asserts `lumenairy.__file__` starts with the worktree path, so a probe
can never silently measure an installed wheel.

| probe | build-doc table | what it measures |
|---|---|---|
| `p1_reduction.py` | T1 | G1: sha256 + max-diff of `Lmat`/`Rmat`/`Stt`/`Schur`/`Et_blocks` for the scalar arm vs the tensor `e*I` arm, and the two public entries' R/T |
| `p2_granet_table2.py` | T2 | G2: Granet Fig.4 grating under four transmitted-efficiency definitions x two axis assignments x two M |
| `p3_three_engine.py` | T2, T5 | the Granet geometry through the staggered, hybrid and RCWA engines (is the disagreement with the paper in the assembly, or in the reading?) |
| `p4_g2_search.py` | T2 | the 32-combination reading sweep (axis / pillar / conjugation / substrate sign) x three definitions |
| `p5_g2_diag.py` | T2 | full order spectrum, both polarizations, `(2,2)`-corner vs `(4,4)`-centred position invariance, M ladder, and five discrete host/pillar readings |
| `p6_g2_substrate.py` | T2 | the substrate reading (permittivity vs index vs vacuum vs glass) |
| `p7_berreman.py` | T3 | G3: uniform in-plane tensor slab vs `berreman_jones_1d`, 2 tensors x 2 grids x 3 incidences x 2 M |
| `p8_1d_reduction.py` | T4 | G4: y-uniform anisotropic stripe vs `pmm_jones_1d` and `rcwa_jones_1d`, per order, plus the y-forbidden leak |
| `p9_three_engine_2d.py` | T5 | G5: 2-D anisotropic cell across three engines + the `n_orders` 4->8 no-floor comparison |
| `p10_sym_stack.py` | T6-T9 | G6 closure, G7 symmetries + the swapped-block control, G8 stack, G9 absorption |
| `p11_tol_derivation.py` | T11 | the 40 lossless SCALAR configurations the closure tripwire must not fire on |
| `p12_cost.py` | T12 | wall time, `tracemalloc` peak and the retained-operator inventory, scalar vs tensor at `(3,3)`/M=8 |
| `p13_tripwire.py`, `p13b_tripwire.py`, `p13c_tripwire.py` | T11 | the search for an engineered LOSSLESS closure violation (contrast ladder, period/depth sweep, near-Rayleigh-cutoff route) |
| `p14_test_numbers.py` | T2-T4, T7, T8, T10 | recomputes exactly the quantities the test file's bars cite, by importing the test module's own helpers -- so the doc and the assertions cannot drift |
| `p15_g8a_g9.py`, `p16_g9_ladder.py`, `p17_mirror.py` | T8, T9, T7 | the stack / absorption / mirror numbers at the test file's exact parameters |
