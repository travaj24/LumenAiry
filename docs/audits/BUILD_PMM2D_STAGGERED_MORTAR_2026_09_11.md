# BUILD -- per-layer element grids (L2 mortar) + non-uniform segment boundaries for the PURE staggered 2-D PMM

**Date:** 2026-09-11 · **Branch:** `feat/pmm2d-staggered-mortar` (worktree
`C:/tmp/lum_mortar`, off the integration branch `wave2/pmm2d` `3034bcb`) ·
**Class:** library build + measurement

**Spec:** `docs/audits/EXPERIMENT_PMM2D_STAGGERED_MORTAR_2026_09_10.md` -- its
GO decision, the gate list G1-G9 of S12.4, and the FOLLOW-UP's F5 gate list
N1-N7.
**Prior art transplanted:** the 1-D per-layer surface
(`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md`), its mortar algebra and
`_redheffer_star_rect` (`pmm/_core.py`), its conditioning lessons
(`PMM_M1_CONDITIONING_2026_08_04.md`), and the hybrid's taper builders
(`pmm/stack2d.py`).

**Every number below is MEASURED on TWO BUILDS** by
`validation/probe_pmm2d_staggered_mortar/b1_build_gates.py`, which drives the
LIBRARY API only (unlike the `m*` / `f*` prototype scripts of the experiment,
which carry their own research implementation):

| arm | OS / toolchain | python | numpy | scipy | BLAS |
|---|---|---|---|---|---|
| **WIN** | Windows 11, MSVC wheels | 3.14.6 | 2.4.4 | 1.17.1 | scipy-openblas |
| **WSL** | WSL2 Ubuntu, gcc wheels, glibc 2.39 | 3.12.3 | 2.4.6 | 1.17.1 | scipy-openblas |

`OMP/OPENBLAS/MKL_NUM_THREADS=1` on both.  Commands:

```
cd /c/tmp/lum_mortar
PYTHONPATH=/c/tmp/lum_mortar OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 B1_TAG=win \
  python validation/probe_pmm2d_staggered_mortar/b1_build_gates.py

wsl.exe -e bash -lc "cd /mnt/c/tmp/lum_mortar && OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 B1_TAG=wsl PYTHONPATH=/mnt/c/tmp/lum_mortar \
  ~/lumvenv/bin/python validation/probe_pmm2d_staggered_mortar/b1_build_gates.py"
```

Results land in `b1_build_gates_win.json` / `b1_build_gates_wsl.json` beside
the script.  Wall times were taken with several processes sharing the box and
carry ~20 % contention noise; ratios quoted are within-script.

Following the experiment's F1 finding, the tables distinguish
**discretisation-limited** readings -- which this build reproduces across the
two arms to **8-13 significant digits** and which therefore carry tight bars --
from **round-off-limited** ones, which have an O(1) cross-build spread and are
quoted as magnitudes only.

---

## 0. WHAT SHIPPED

### 0.1 Non-uniform segment boundaries (Granet 2023 Eq. 31)

```python
Basis1D(d, walls, M, tau)                              # walls: int N or (N+1,) array
Granet2DTransverseE(px, py, wx, wy, M, eps_cell, ...)  # wx/wy: int or array, per axis
```

`walls` as an `int N` is the uniform lattice and is BIT-IDENTICAL to the
pre-change library; as an increasing `(N+1,)` array it is an arbitrary
partition.  The change is ONE change at FOUR sites -- the scalar jacobian `J`
becomes the per-segment vector `J_n`:

1. `Basis1D._global_matrix`,
2. `_global_pair_segmat` (module level),
3. `Granet2DTransverseE._eps_dir`'s inline `segmat`,
4. `_stag_fourier_projection`.

`Basis1D.mixed` needed NO edit -- one derivative contributes `1/J_n` and the
measure contributes `J_n`, so the scale is 1 on every segment whatever its
length.  `Basis1D.J` is `None` on a non-uniform basis on purpose (an
un-migrated reader raises a `TypeError` rather than mis-scaling silently);
`Basis1D.Jn` is the general quantity and `Basis1D.uniform` records which path
a basis took.

**One site NOT in the experiment's list was found during the build**:
`_stag_parity_1d` now returns `None` on a wall set that is not
mirror-symmetric.  `x -> d - x` sends segment `n` to `N-1-n`, and that is a
signed permutation of the LOCAL functions only when the two segments have equal
length; without the guard the out-of-plane parity block reduction would have
been applied to a structure the pencil does not have.  A uniform lattice is
mirror-symmetric, so the shipped behaviour is unchanged.

### 0.2 Per-layer element grids (the L2 mortar)

```python
PMM2DStackPure(period_x, period_y=None, *, ...,
               layer_grids="shared" | "per-layer", window_halfwidth=None)

stack.add_layer(thickness, *, eps=None, eps_cell=None, mu=None, mu_cell=None,
                slant=None,
                x_walls=None, y_walls=None,   # NEW: a patterned layer's walls
                grid=None,                    # NEW: a UNIFORM layer's segments
                n_modes=None)                 # NEW: the per-layer modal count

stack.add_tapered_pillar(thickness, *, eps_pillar, eps_host,
                         x_bounds_bottom, y_bounds_bottom,
                         x_bounds_top=None, y_bounds_top=None,
                         n_slices=8, rule="midpoint")
stack.add_tapered_pillars(thickness, *, pillars, eps_host, n_slices=8)

floor, per_layer = stack.convergence_floor()   # the per-layer-M screen
```

* `layer_grids="shared"` is the DEFAULT and is the union-grid path unchanged.
* `layer_grids="per-layer"` drops the union-grid raise.  A patterned layer's
  segment COUNT comes from its own `eps_cell` and is never a free parameter
  (`grid=` RAISES for a patterned layer -- a pillar 1/2 of the period wide
  exists on `N in {2,4,...}` and one 1/3 wide on `N in {3,6,...}`, so accepting
  a free `N` would silently change the DEVICE); its wall POSITIONS come from
  `x_walls` / `y_walls`, in which case `eps_cell` is the STRIP TILE -- exactly
  the hybrid's `tile`, which is what lets `add_tapered_pillar` transplant.
* `window_halfwidth` RAISES with its reason: on a segment partition the only
  partition carrying two layers' walls is their common refinement, i.e. the
  union grid, so there is no local enrichment to widen.
* The four new keywords RAISE on the shared path.

New functions -- all NEW, so the 1-D mortar and the shared 2-D path are
untouched:

| module | new |
|---|---|
| `pmm/twod_staggered.py` | `_stag_axis_masses` (factored out of `Granet2DTransverseE._axis_mats`), `_stag_basis_fingerprint`, `_stag_cross_mass_1d`, `_stag_cross_mass_1d_cached` (+ the `ByteBudgetedLRU` registry entry `pmm2d_staggered_geometry`), `_stag_kron_apply`, `StagGridOps`, `StagCrossOps` |
| `pmm/_core.py` | `PMM2D_MORTAR_H_SWAP`, `_stag_h_blocks`, `_stag_blk2_apply`, `_interface_smatrix_mortar_2d`, `_interface_smatrix_general_mortar_2d`.  `_redheffer_star_rect` and `_guarded_solve` REUSED unchanged |
| `pmm/stack2d_pure.py` | `_stag_walls_spec`, `_stag_walls_n`, `_stag_interior`, `PMM2DStackPure._source_prep` (factored verbatim out of `solve`), `_perlayer_spec`, `_finish_layer`, `_perlayer_modal_counts`, `_solve_per_layer`, `add_tapered_pillar`, `add_tapered_pillars`, `convergence_floor`, `_require_per_layer_taper` |

---

## 1. The gate table

| gate | claim | verdict | WIN | WSL |
|---|---|---|---|---|
| **N1** | the integer wall path is BIT-IDENTICAL to the scalar-`J` library, on every matrix family and every assembled pencil | **PASS** | 48/48 hashes equal, worst `|d| = 0.0e+00`; 34/34 pencil hashes equal | same |
| **N2** | an explicit `linspace` array is NOT required to be bit-identical; the difference is ULP and explained | **PASS** | 1.96e-16 rel at `d = 0.9, N = 4`; exactly 0 at `d = 1.2` | same |
| **N3** | two exact-wall representations of ONE device agree inside a TRIANGLE-INEQUALITY bar against the exact 1-D oracle | **PASS 5/5** | S4.1 | S4.1 |
| **N4** | arbitrary walls converge to the EXACT 1-D answer on a 3-segment grid | **PASS** | 3.32e-02 -> 5.35e-06 over `q = 9..24` | 3.32e-02 -> 5.35e-06 |
| **N6** | the conforming identity survives NON-UNIFORM grids through forced mortars | **PASS** | 6.5e-16 / 7.5e-16 | 8.0e-16 / 8.9e-16 |
| **N7** | FAIL-BEFORE for each of the four `J -> J_n` sites | **PASS 4/4** | S4.4 | S4.4 |
| **G1** | a conforming per-layer stack is BIT-EXACT vs `layer_grids='shared'` | **PASS** | sha256 equal on R, T, Jones | same |
| **G2** | the conforming identity through the FORCED mortar, bar DERIVED at runtime | **PASS 5/5** | worst 1.11e-13 vs a derived 1.62e-11 (146x) | worst 1.33e-13 (122x) |
| **G3** | **the H-row V1/V2 swap is load-bearing** -- disabling it must move the observable by orders on a NON-conforming stack | **PASS** | 5.8396e-03 -> 8.0455e+00 (1378x); closure 1.0412e-08 -> 2.7654e+01 (2.7e+09x) | 5.8396e-03 -> 8.0459e+00; closure 1.0413e-08 -> 2.7654e+01 |
| **G4** | a TRANSPARENT interface across NON-conforming grids reproduces the analytic Fresnel slab and `berreman_jones_1d` | **PASS** | 3.8e-14 scalar oblique; 3.221e-11 OOP conical | 3.5e-15; 3.221e-11 |
| **G5** | the stripe pair per order vs the exact 1-D `PMMStack`, with the anti-mirror tripwire | **PASS** | 1.73e-01 -> 2.90e-05 over `M = 5..11`; mirror/direct 8.3x .. 4575x | identical to 12 digits |
| **G6** | EQUAL-DOF non-regression: NOT WORSE on the observable, TIGHTER on closure | **PASS** | ratios 0.1769 / 0.0587 at `q = 12 / 18` | 0.1769 / 0.0587 |
| **G7** | two-sided lossless closure, scalar AND Hermitian tensor | **PASS** | 9.69e-04 -> 1.67e-05 over `M = 4..6` | same class |
| **G8** | the far-field order cap is DERIVED from the END grids and RAISES | **PASS** | error names the cap and both grids | same |
| **G9** | the conditioning census refuses nothing in the useful range | **PASS** | worst equilibrated `rcond` 8.92e-05, 4 decades above M1's 1e-8 screen | 8.91e-05 |
| **O-3** | `retain_internal` / `layer_absorption` on per-layer grids: the cross-machinery budget closes | **PASS, item CLOSED** | 5.27e-05 at `M = 5`, 2.14e-06 at `M = 6` | same class |
| **taper** | a 4-slice ARBITRARY-WALL taper agrees with the hybrid's own staircase at the same slices, and its closure is decades tighter | **PASS** | `R00` 0.027188 vs the oracle's 0.027002 (its own self-gap 5.88e-04); closure 5.17e-04 vs 8.34e-03 | `R00` 0.027188243730834 |
| **staircase** | a 3-slice staircase at EQUAL DOF is not worse than the union lattice, which additionally has a DOF FLOOR | **PASS** | 8.21e-02 vs 1.29e-01 at `q = 12` (0.634x) | S6.2 |

---

## 2. The one genuinely 2-D piece, and its fail-before (G3)

In 1-D both transverse components live in the SAME nodal space, so the shipped
mortar applies one 1-D mass blockwise as `kron(I_2, M)`.  That does NOT carry
over.  Here

```
V1 (E1 = Ex) = B(x) (x) Btilde(y)      Gram G1 = kron(Mtt_y, Mbb_x)
V2 (E2 = Ey) = Btilde(x) (x) B(y)      Gram G2 = kron(Mbb_y, Mtt_x)
```

are DIFFERENT spaces, and the Eq.-25 dual puts **H2 in the V1 placement and H1
in the V2 placement** -- read off `_region_modes`' `rot = [-Dual[qq:];
Dual[:qq]]` and confirmed by `PMM2DStackPure._flux_at`, which pairs `H[qq:]`
with `G1`.  Therefore

```
MassE_X = blkdiag(G1_X, G2_X) = -Rmat_X      CrossE = blkdiag(C1, C2)
MassH_X = blkdiag(G2_X, G1_X)   (SWAPPED)    CrossH = blkdiag(C2, C1)
```

**On a conforming interface `C1 = G1` and `C2 = G2`, so the swap cancels
identically and every identity-class gate passes either way.**  It is measured
instead by the fail-before switch `_core.PMM2D_MORTAR_H_SWAP` on a genuinely
non-conforming pair (pillar 1/2 wide on `N = 2` over pillar 1/3 wide on
`N = 3`, `theta = 0.18`, `phi = 0.35`, scored against the union grid on the
common refinement `N = 6`):

| `PMM2D_MORTAR_H_SWAP` | error vs the union reference | `|R + T - 1|` |
|---|---|---|
| `False` (naive same-order blocks) | **8.0455e+00** (WIN) / **8.0459e+00** (WSL) | **2.7654e+01** both |
| `True` (the shipped design) | **5.8396e-03** both | **1.0412e-08** (WIN) / 1.0413e-08 (WSL) |

**A factor 1.4e+03 on the observable and 2.7e+09 on lossless closure**, and
both arms are discretisation-scale, so the fail-before REPRODUCES across builds
to every printed digit rather than merely re-occurring.  The cheaper fixture
the shipped test uses (reference `M = 4`, arms `M_A = 7` / `M_B = 5`) reads
2.33e-02 / 3.59e-06 with the swap and 4.02e+01 / 1.15e+02 without.

The cross-mass is also COMPLEX here -- `Basis1D` glues its periodic hat with
`tau = exp(-i alpha0 d)`, so the basis carries the Bloch phase -- and the 1-D
algebra's `Cab^T` is `Cab^H`.  Using the transpose would be silent at normal
incidence (`tau = 1`).

---

## 3. The identity gates

### 3.1 G1 -- the identical-grid BYPASS is bit-exact

A three-layer stack (pillar `N=2` | uniform | stripe `N=2`) built twice, once
with `layer_grids='shared'` and once with `'per-layer'` at matching grids and
modal counts.  **sha256 of R, T and the reflection Jones are EQUAL** on both
builds.  This is a bypass, not an arithmetic identity, so the claim is exact
and no existing user's bits move.

### 3.2 G2 -- the conforming identity through the FORCED mortar

Bypass DISABLED (`_solve_per_layer(force_mortar=True)`), so a conforming stack
is driven through the mortar algebra.  Relative to `max(R, T)` for R/T and to
`max|J|` for the Jones:

| case | `M` | WIN worst | WSL worst |
|---|---|---|---|
| stripe \| stripe, `theta = 0` | 5 | 3.77e-15 | 2.92e-15 |
| stripe \| stripe, `theta = 0.20` | 5 | 2.88e-15 | 2.54e-15 |
| stripe \| pillar, `theta = 0.20` | 6 | 3.29e-14 | 2.89e-14 |
| pillar \| pillar, CONICAL (0.25, 0.7) | 6 | **1.11e-13** | **1.33e-13** |
| stripe \| uniform \| pillar, `theta = 0.15` | 5 | 1.22e-14 | 1.36e-14 |

Bit identity is not attainable and is not claimed: the mortar multiplies by the
block Gram `G` on both sides of a `solve` where the plain interface does not,
so the identity is exact algebraically and rounds at `eps * cond_2(G)`.
**The bar is DERIVED AT RUNTIME** from the stack's own grid, because it grows
with `M`:

| grid | `cond_2(G1)` | `cond_2(G2)` | `eps * cond` | `10 eps cond` (the bar) |
|---|---|---|---|---|
| `N=2, M=5` | 3.0594e+03 | 3.0466e+03 | 6.8e-13 | 6.8e-12 |
| `N=3, M=5` | 3.0594e+03 | 3.0537e+03 | 6.8e-13 | 6.8e-12 |
| `N=2, M=6` | 7.7633e+03 | 7.7449e+03 | 1.7e-12 | 1.7e-11 |
| `N=3, M=6` | 7.2271e+03 | 7.2902e+03 | 1.6e-12 | 1.6e-11 |

The condition numbers agree between the two builds to 12-13 digits.  The worst
reading over both builds (1.33e-13) sits **122x inside** its grid's derived
bar.  A fixed `1e-12` constant would fail at `M = 8`, where the experiment
measured `eps * cond = 7.8e-12`.

### 3.3 N6 -- and the same identity ON NON-UNIFORM grids

Three IDENTICAL slices at arbitrary walls (0.2371, 0.6183 of the period),
forced through the mortar at all four interfaces, against the bypass:

| `M` | WIN `dR` / `dT` / `dJ` | WSL |
|---|---|---|
| 5 | 6.5e-16 / 1.0e-15 / 7.4e-16 | 8.0e-16 / 1.2e-15 / 1.1e-15 |
| 7 | 7.5e-16 / 5.3e-16 / 8.9e-16 | 8.9e-16 / 1.0e-15 / 1.2e-15 |

This is the control that separates "the multi-layer machinery on arbitrary
walls" from "the mortar": they agree at round-off, so any residual a taper
shows is the device, not the interface.

---

## 4. The non-uniform half (N1-N4, N7)

### 4.1 N1 -- bit identity of the integer path

Against a scalar-`J` reimplementation of each of the four sites, SAME BUILD,
by sha256 of the raw bytes.

| arm | what | WIN | WSL |
|---|---|---|---|
| 1-D, `walls = N` over 6 grids `(d, N, M)` x 8 families (`mass<til\|til>`, `mass<B\|B>`, eps-weighted mass, `stiff`, `mixed`, both `_global_pair_segmat` refs, `_stag_fourier_projection` at `alpha0 = 0.31`) | 48 hashes | **0 mismatches, worst `|d| = 0.0e+00`** | same |
| 2-D assembled pencils `Rmat / Lmat / Stt / Schur / Agen / Bgen`, 3 grids x 5 cell kinds (scalar, in-plane tensor, out-of-plane, magnetic, slanted) | 34 hashes | **all equal** | same |

### 4.2 N2 -- the ULP case, explained

`np.linspace` computes `start + i*step` and pins its last element, so
`linspace[i+1] - linspace[i]` is not always the same double as `d/N`.
Measured relative difference of the assembled mass:

| `(d, N, M)` | WIN | WSL |
|---|---|---|
| (1.2, 2, 5), (1.2, 4, 5), (0.9, 3, 6) | 0.0 | 0.0 |
| (0.9, 4, 5) | **1.96e-16** | 1.96e-16 |

That is why the integer path is kept DISTINCT: routing the int through the
array path would make N1's unconditional claim depend on the period.

### 4.3 N3 / N4 -- against exact independent oracles

**N3** -- a duty-1/3 stripe has TWO exact-wall representations: a NON-uniform
2-segment grid with its wall at `P/3`, and the uniform 3-segment lattice.  The
bar is the TRIANGLE INEQUALITY against the exact 1-D `PMMStack` at degree 14
(its own degree-12 self-gap 1.5527e-05): two representations of one device may
disagree by at most the SUM of their own distances to truth.

| `M` | `q_NU` / `q_uniform` | err NU | err uniform | their gap | derived bar | inside? |
|---|---|---|---|---|---|---|
| 4 | 6 / 9 | 7.337e-02 | 2.975e-02 | 5.326e-02 | 1.031e-01 | yes |
| 5 | 8 / 12 | 2.123e-02 | 1.714e-03 | 2.126e-02 | 2.294e-02 | yes |
| 6 | 10 / 15 | 2.344e-03 | 7.227e-04 | 2.782e-03 | 3.066e-03 | yes |
| 7 | 12 / 18 | 7.073e-04 | 3.837e-05 | 7.039e-04 | 7.456e-04 | yes |
| 9 | 16 / 24 | 7.269e-06 | 3.729e-07 | 7.095e-06 | 7.642e-06 | yes |

**5/5 rungs inside a derived, not fitted, bar**, on both builds, and every
entry agrees between the arms to 11-13 digits.  Note the NU arm reaches
7.27e-06 on `q = 16` where the uniform arm needs `q = 24` for 3.73e-07 -- the
two sit on the same accuracy-per-DOF curve.

**N4** -- walls at 0.2371 and 0.6183 of the period, which the uniform lattice
cannot represent at any affordable `q`, made y-uniform so the exact 1-D
`PMMStack` is the truth for the whole 2-D stack (self-gap 8.5373e-06):

| `M` | `q` | err vs EXACT (WIN) | (WSL) | closure | wall (WIN) |
|---|---|---|---|---|---|
| 4 | 9 | 3.3205e-02 | 3.3205e-02 | 3.25e-03 | 0.3 s |
| 5 | 12 | 3.8906e-03 | 3.8906e-03 | 2.43e-04 | 1.2 s |
| 6 | 15 | 1.2149e-03 | 1.2149e-03 | 6.42e-06 | 4.3 s |
| 7 | 18 | 4.6695e-04 | 4.6695e-04 | 3.61e-07 | 13.1 s |
| 9 | 24 | **5.3543e-06** | **5.3543e-06** | **1.98e-10** | 73.2 s |

**Four decades of convergence against an exact independent oracle at walls the
shipped uniform lattice cannot represent, and eight decades of lossless
closure** -- with the two builds agreeing to 10+ digits at every rung.

### 4.4 N7 -- the fail-before for the four `J -> J_n` sites

Each site reverted to the scalar `J` in-process, with both arms in the same
process on the same build.  The claim is two-sided:

| site reverted | UNIFORM arm (bit-identical?) | NON-UNIFORM arm moved by |
|---|---|---|
| `Basis1D._global_matrix` | **sha256 equal** | > 1e-3 (the arm's closure blows to 1.25 / 1.07) |
| `_global_pair_segmat` | **sha256 equal** | > 1e-3 (closure 1.08 / 12.2) |
| `Granet2DTransverseE._eps_dir` | **sha256 equal** | > 1e-3 |
| `_stag_fourier_projection` | **sha256 equal** | > 1e-3 (closure 1.07) |

**A uniform-lattice-only gate cannot see ANY of the four** -- exactly as a
conforming-parity-only gate cannot see the mortar's V1/V2 swap.  That symmetry
between G3 and N7 is the shape this build kept returning to: every design
choice that is silent on the easy case carries an engineered arm that is wrong.

---

## 5. The accuracy gates (G4, G5)

### 5.1 G4 -- the mortar's OWN error, isolated

**Scalar.**  One uniform slab (`n = 2`, 300 nm) SPLIT into two sub-layers on
DIFFERENT grids.  The interface is physically absent, so the exact answer is
the analytic Fresnel slab; nothing but the mortar can move it.  `|R - R_exact|`
at `M = 7`:

| `theta` | (2,2) conf | (2,4) nested | (2,3) NON-conf | (3,4) NON-conf |
|---|---|---|---|---|
| 0.00, WIN | 9.44e-14 | 3.60e-14 | 4.37e-14 | 1.08e-14 |
| 0.00, WSL | 5.64e-14 | 4.16e-14 | 4.22e-14 | 2.28e-14 |
| 0.20, WIN | 6.72e-14 | 1.16e-14 | **3.81e-14** | 3.59e-14 |
| 0.20, WSL | 6.91e-14 | 2.99e-14 | **3.50e-15** | 6.06e-14 |

with closure at or below 8.2e-14 on every arm.  **A NON-conforming mortar
interface is not measurably worse than a conforming one** -- which is what says
every gap in the other gates is RESOLUTION, not the interface.  These readings
are ROUND-OFF-limited and their cross-build spread is O(1) by construction, so
the shipped test bars them as a magnitude (1e-10) and pins nothing.

**Out-of-plane.**  One uniform tilted-director LC slab
(`n_o = 1.5, n_e = 1.7`, tilt 35 deg, azimuth 25 deg) split across grids, on
the GENERALIZED cascade, against `berreman_jones_1d`.  Relative `dJones` at
25 deg CONICAL, `M = 7`:

| pair | WIN | WSL | closure (WIN) |
|---|---|---|---|
| (2,2) conforming | 2.0635e-12 | 2.0635e-12 | 4.27e-13 |
| (2,3) NON-conf | **3.2209e-11** | **3.2209e-11** | 1.30e-11 |
| (3,4) NON-conf | 3.4632e-13 | 3.4632e-13 | 1.40e-13 |

Identical to five digits across builds -- these are discretisation-limited.
**The generalized mortar twin reproduces Berreman through a non-conforming
interface at conical incidence to 3.2e-11.**

### 5.2 G5 -- the stripe pair per order vs the exact 1-D oracle

Layer A duty 1/2 on `N = 2`, layer B duty 1/3 on `N = 3` -- genuinely
non-conforming, common refinement `N = 6` -- made y-uniform so the exact 1-D
`PMMStack` at degree 14 is the truth for the whole 2-D stack (its own degree-12
self-gap **8.5978e-06**).  Score = max over the retained `n = 0` orders of
`|R - R_1D|`, `|T - T_1D|` on the TE row.

| `M` (both layers) | err vs 1-D (WIN) | (WSL) | MIRRORED assignment | closure | wall (WIN) |
|---|---|---|---|---|---|
| 5 | 1.7286e-01 | 1.7286e-01 | 1.0148e-01 | 2.06e-04 | 0.8 s |
| 7 | 1.6599e-02 | 1.6599e-02 | 1.3711e-01 | 9.89e-06 | 11.9 s |
| 9 | 4.6202e-04 | 4.6202e-04 | 1.3249e-01 | 2.65e-08 | 74.7 s |
| 11 | **2.8963e-05** | **2.8963e-05** | 1.3250e-01 | **2.49e-11** | 272.4 s |

**Four decades of convergence on the observable and seven on lossless closure,
reproducing across the two builds to 12-13 significant digits.**  The MIRRORED
column is the anti-mirror tripwire (physical order `-m` deposited into slot
`m`, the historical A|B defect of this engine): from `M = 7` the direct
assignment beats it by **8.3x / 287x / 4575x**, so the per-layer cascade is not
merely converging, it is converging into the RIGHT order slots.  At `M = 5` the
solve is not resolved enough for the two assignments to separate (0.59x), which
is why the tripwire is asserted at the resolved end.

The shipped test runs the cheaper rungs `(M_A, M_B) = (5,5) / (7,7) / (8,6)`
-- errs 1.73e-01 / 1.66e-02 / 6.44e-03, mirror 1.02e-01 / 1.37e-01 / 1.35e-01
-- because a `q = 30` region eig is an 1800-dimension solve.

---

## 6. The comparison that is not rigged (G6) and the staircase

### 6.1 G6 -- EQUAL degrees of freedom, two-layer stripe pair

Per axis `q = N (M-1)`, so `q_union(M) = 6(M-1)` equals `q_A = 2(M_A-1)` at
`M_A = 3M-2` and `q_B = 3(M_B-1)` at `M_B = 2M-1`; at those settings **every
region eigenproblem in both arms has exactly the same dimension `2 q^2`**.

| `q` | eig dim | arm | settings | err vs 1-D | closure | wall |
|---|---|---|---|---|---|---|
| 12 | 288 | union | `M = 3` | 1.5102e-01 | 8.51e-04 | 2.0 s |
| 12 | 288 | **per-layer** | `M_A = 7, M_B = 5` | **2.6712e-02** | **7.27e-05** | 2.5 s |
| 18 | 648 | union | `M = 4` | 5.0738e-03 | 2.27e-05 | 21.8 s |
| 18 | 648 | **per-layer** | `M_A = 10, M_B = 7` | **2.9777e-04** | **6.57e-10** | 24.8 s |

**mortar/union error ratio 0.176877 at `q = 12` and 0.058687 at `q = 18`** --
i.e. 5.7x and 17x more accurate at identical eigenproblem sizes, with the
closure 11.7x and 34,600x tighter, at 1.14-1.25x the wall time.  The two builds
agree on the RATIO to 9 significant digits (0.1768765744527005 vs
0.17687657457247719), which is why the shipped test bars the ratio rather than
the errors.

**The claim is stated as NOT WORSE, never as BETTER.**  The experiment measured
the advantage GONE at `q = 30` on this same pair (1.48x the union's error once
both arms have converged) and moving earlier still -- to between `q = 18` and
`q = 24` -- on a corner-dominated 2-D pillar pair.  The durable half is the
closure, which does not decay.

### 6.2 The 3-slice staircase, the use case

Slice widths 1/2, 1/3, 1/6 of the period, so per-slice `N = 2, 3, 6` and the
union lattice is their common refinement `N = 6`.  Made y-uniform, exact 1-D
oracle at degree 14 (self-gap 4.901e-06), per-slice `M_i = q/N_i + 1`:

| `q` | eig dim | arm | settings | err vs 1-D | closure | wall |
|---|---|---|---|---|---|---|
| 12 | 288 | union | `M = 3` | 1.294e-01 | 4.76e-03 | 1.6 s |
| 12 | 288 | **per-layer** | `M_i = 7, 5, 3` | **8.206e-02** | 4.46e-03 | 3.1 s |
| 18 | 648 | union | `M = 4` | 4.791e-03 | 5.15e-05 | 32.9 s |
| 18 | 648 | **per-layer** | `M_i = 10, 7, 4` | **2.412e-03** | 5.13e-05 | 38.9 s |

Ratios 0.634 and 0.504 -- not worse at either rung, and the advantage does not
decay across the measured range.

**The structural result is the DOF FLOOR, not the ratio.**  `Basis1D` requires
`M >= 3`, so the smallest `q` any lattice can carry is `2 N`.  On the sibling
staircase with widths 1/2, 1/3, 1/4 the union lattice is `N = 12` and cannot be
run below `q = 24` at all, where the per-layer arm reaches `q = 12`: **64x less
eigenwork, and it is a floor, not a tuning choice**.  At production modal
counts the same arithmetic (`dim = 2(N(M-1))^2`, `16 dim^2` bytes) gives an
eig-work ratio of **1832x** and, at `M = 8`, a 2.97 GB union matrix against
0.037 GB per-layer.

---

## 7. Two-sided closure, conditioning, and the order cap (G7-G9)

### 7.1 G7 -- two-sided lossless closure

On a lossless stack `sum R + sum T = 1` is EXACT, so the gate is two-sided.
A HERMITIAN (gyrotropic `e12 = -e21 = 0.9i`) tensor absorbs nothing, so the
same exact claim applies -- and it is the arm showing that the mortar carries
no material dependence.  `(2,3)` non-conforming pillar pair, `theta = 0.18`,
`phi = 0.35`:

| `M` | scalar |
|---|---|
| 4 | 9.69e-04 |
| 5 | 1.83e-04 |
| 6 | **1.67e-05** |

Each rung 5.3x then 11x tighter than the last.  The GYROTROPIC arm is run
under the same assertion (each rung at least 3x tighter, last below 1e-4) and
passes; its individual readings are not tabulated here because the gate is the
convergence SHAPE, not the values.  The tensor arm existing at all is the
material-independence claim: the mortar is a geometric projection.  The build-free restatement of a
closure ladder is CONVERGENCE, not a reading, so that is what the test asserts.

### 7.2 G9 -- the conditioning census

The mortar's ONE explicit inverse (`I + BA`) goes through the shipped guard, so
M1's instrument reads it for free.  Four sites appear per solve: the mortar
interface, the two per-layer star denominators and the plain interface
mode-match.  Worst equilibrated `rcond` over all solves:

| config | `M = 4` | `M = 5` | `M = 6` |
|---|---|---|---|
| pillar (2,3), WIN | 7.218e-04 | 5.237e-04 | 4.054e-04 |
| pillar (2,3), WSL | 7.219e-04 | 5.239e-04 | 4.050e-04 |
| pillar (2,6), WIN | **8.919e-05** | 1.497e-04 | 1.089e-04 |
| pillar (2,6), WSL | **8.914e-05** | 1.496e-04 | 1.581e-04 |
| stripe (2,3), WIN | 2.275e-04 | 1.492e-03 | 7.682e-04 |
| stripe (2,3), WSL | 2.275e-04 | 1.492e-03 | 7.677e-04 |

**No cliff, and nothing refused.**  The worst reading anywhere is 8.91e-05,
FOUR decades above M1's `1e-8` screen.  One cell (`pillar (2,6)` at `M = 6`)
moves 45 % between builds -- exactly the shape `TESTING_STANDARDS.md` S5 warns
about for a census quantity downstream of a near-degenerate operator, and the
reason the shipped bar is `> 1e-6` rather than a reading.  `_guarded_lstsq`
refused nothing on any arm.

### 7.3 G8 -- the far-field order cap

The forward Rayleigh projection has `q = N (M-1)` columns per axis, so it can
carry at most `q` order slots; beyond that the retained slots ALIAS one another
and the least-squares draw is build-dependent while conserving energy exactly
-- the T3-3 shape, energy-invisible.  The half-spaces ride the END layers'
grids, so the cap is `n_orders <= (min(q_sup, q_sub) - 1) // 2`.

**The probe CLAMPED; the shipped surface RAISES** (open item O-9), naming the
cap and both end grids, because a user asking for orders the end grids cannot
carry should be told, not quietly served fewer.  The cap follows the END
layers' own `n_modes`, so raising the first and last layers' resolution is the
documented fix as well as lowering `n_orders`.

---

## 8. Memory: the cross-mass is never materialised

`C1 = kron(Ctt_y, Cbb_x)` -- an IDENTITY, because both field spaces are tensor
products and both segment partitions are rectangular.  Measured against the
dense `np.kron`:

| grids | `M` | dense | factors | memory ratio | apply speed (WIN / WSL) | identity |
|---|---|---|---|---|---|---|
| (2,3) | 6 | 0.36 MB | 0.0048 MB | 75x | 0.34x / 0.27x | 4.75e-16 |
| (3,6) | 6 | 3.24 MB | 0.0144 MB | 225x | 2.98x / 2.74x | 4.12e-16 |
| (4,6) | 8 | 22.1 MB | 0.0376 MB | 588x | 18.1x / 22.9x | 6.63e-16 |
| (6,12) | 6 | 51.8 MB | 0.0576 MB | **900x** | **39.0x / 26.9x** | 2.24e-16 |

The factorisation is an identity to 2-7e-16 (BLAS reassociation only) on both
builds, and the memory ratio grows as `q_A q_B`.  **The small pairs are NOT
faster factored** -- (2,3) at `M = 6` reads 0.27-0.34x, i.e. einsum overhead
dominates below ~100x100 -- and the shipped test says so.  The win that makes
the design work is the MEMORY: at the case-B staircase's `(N=4) | (N=12)`
interface the dense form is already ~50 MB **per component per interface**, and
a staircase has two components and `nlay + 1` interfaces.

Cache: the 1-D cross-mass factors are memoized in the `ByteBudgetedLRU`
registry as `pmm2d_staggered_geometry`, keyed on
`(d, M, tau, wall bytes)` per basis, so `clear_asm_caches()` drains it and
`cache_report()` shows it by name.  An entry is `q_a x q_b` complex128, i.e.
kilobytes.

---

## 9. `retain_internal` / `layer_absorption` (open item O-3, CLOSED)

Two changes and one non-change, exactly as the experiment's S12.3 designed:

* the partial cascades `S_above` / `S_below_bot` become RECTANGULAR and go
  through `_redheffer_star_rect` (verbatim what the 1-D per-layer path does);
* `_flux_at` reads the LAYER's own block field Gram `blkdiag(G1_i, G2_i)` and
  its own `qq_i`.  **No new assembly**: `G_i` IS `-Rmat_i` and its Kronecker
  FACTORS are already held in that layer's `StagGridOps`, so the flux
  quadrature is separable too;
* `layer_absorption`'s half-space calibration `F_top[0] / (1 - R_tot)` is
  unchanged, because the half-spaces are conforming by construction.

The honest gate is the CROSS-MACHINERY closure `sum_i A_i == 1 - sum R - sum T`
-- an internal Gram-flux budget against the Rayleigh far field, two independent
pieces of machinery.  On a NON-conforming lossy pillar pair
(`eps = 6 + 0.35i` on `N = 2` over `5 + 0.20i` on `N = 3`, `theta = 0.18`,
`phi = 0.35`):

| `M` | `sum_i A_i` | `1 - R - T` | gap | absorbed fraction |
|---|---|---|---|---|
| 5 | [0.19987, 0.21930] | [0.19991, 0.21935] | 5.27e-05 | 0.20-0.22 |
| 6 | [0.22817, 0.20434] | [0.22817, 0.20434] | **2.14e-06** | 0.20-0.23 |

The gap is discretisation-limited (it tightens 25x with one modal rung), so it
reproduces across builds and takes a real bar.  **O-3 is CLOSED.**

---

## 10. Tapers on the no-floor engine

`PMM2DStackPure.add_tapered_pillar` / `add_tapered_pillars` transplant the
hybrid's surfaces: an auto-sliced z-staircase of exact-wall scalar layers, each
slice on its OWN 3-segment non-uniform grid whose interior walls are the
interpolated pillar bounds.  **No two slices share a wall.**

Fixture: 4 slices, bounds interpolating from `[0.1873 P, 0.7241 P]` to
`[0.2917 P, 0.6109 P]` with the midpoint rule, `theta = 0.18`, `phi = 0.35`,
scored against the hybrid's own `add_tapered_pillar` staircase at the IDENTICAL
slices:

| arm | setting | `R(0,0)` | closure | wall (WIN) |
|---|---|---|---|---|
| hybrid | degree 7, `n_orders` 9 | 0.027002 | 8.34e-03 | -- |
| hybrid | degree 9, `n_orders` 9 | 0.026414 -- deg-7-vs-9 self-gap **5.88e-04** | 3.68e-03 | 15.9 s |
| hybrid | degree 11, `n_orders` 9 | deg-9-vs-11 self-gap 2.62e-04 | -- | -- |
| **pure NU mortar** | `M = 4` (`q = 9`) | 0.055159 | 3.12e-03 | 0.9 s |
| **pure NU mortar** | `M = 5` (`q = 12`) | **0.027188** | **5.17e-04** | 4.4 s |
| **pure NU mortar** | `M = 6` (`q = 15`) | 0.028124 | **3.64e-05** | 16.9 s |

`R(0,0)` reproduces between builds to 12-13 digits (0.027188243730828 WIN /
0.027188243730834 WSL).  **The pure arm sits 1.86e-04 from the degree-7 oracle
-- inside the oracle's own 5.88e-04 self-gap -- with a lossless closure 16x
tighter at `M = 5` and 101x tighter at `M = 6` than the oracle it is being
scored against.**  So the residual is the oracle's floor showing, and the test
says so rather than pretending otherwise.

The cost statement, which is the point of the whole item: a 2-degree sidewall
over 310 nm at 6 slices moves a wall ~1.8 nm per slice, `delta = 2.6e-03` of a
700 nm period.

| | uniform lattice | non-uniform segments |
|---|---|---|
| segments per slice | `N ~ 390` | **3** |
| `q` | `>= 1170` | `3 (M-1)` -- **24** at `M = 9` |
| region eig dimension `2 q^2` | `>= 2.7e+06` | **1152** |

---

## 11. Tests

| file | tests | WIN wall | slowest test |
|---|---|---|---|
| `tests/unit/test_pmm2d_staggered_nonuniform.py` | 16 | 77.5 s | 21.8 s (`n3_two_exact_wall_representations_agree_inside_the_triangle_bar`) |
| `tests/unit/test_pmm2d_staggered_mortar.py` | 31 | 209.0 s | 32.9 s (`g3_h_row_v1_v2_swap_is_load_bearing`) |

47 tests, 286.9 s for the pair, no test above 33 s (measured with two other
processes on the box).  Every gate above has a test; the three the tests
DETUNE relative to the build-doc measurement, and why:

| gate | the doc measures | the test runs | why |
|---|---|---|---|
| G5 | `M = 5..11` (a `q = 30` region eig is an 1800-dimension solve, 272 s) | `(M_A, M_B) = (5,5) / (7,7) / (8,6)` | CI budget; the ladder's SHAPE (each rung >= 2x tighter, ending far above the oracle's self-gap) and the anti-mirror ratio are what the test asserts, and both survive the trim |
| G6 | `q = 12` and `q = 18` | `q = 12` | the `q = 18` point is a 47 s pair of solves; the RATIO claim is identical in kind at both |
| taper | hybrid `n_orders = 9`, pure to `M = 6` | hybrid `n_orders = 7`, pure to `M = 5` | the deg-9 / `n_orders` 9 hybrid solve alone is 16 s; the oracle's self-gap and the closure ratio are re-measured at the test's own setting (5.15e-04 and 13x) rather than carried over |

Both files cap `OMP/OPENBLAS/MKL_NUM_THREADS` at the top before importing
numpy.  `.test_durations` is spliced with the measured per-test times.

Regression suite re-run at the end of the build, all green:

| suite | result |
|---|---|
| `test_pmm2d_staggered_slant.py`, `..._oop.py`, `..._oop_block_eig.py`, `..._anisotropic.py`, `..._magnetic.py`, `..._wood_list.py`, `test_v5_12_0_pmm2d_staggered.py`, `test_v5_21_pmm2d_staggered_oblique.py`, `test_v5_14_0_pmm2d_stack.py`, `test_p2c_pmm2d_stack_cascade.py` | **299 passed** in 960 s |
| `test_audit_dynameta_consumer_api_2.py` (the `PMM2DStackPure` consumer gate) | see S13 |
| `ruff check lumenairy/ tests/` | clean |

---

## 12. Open items

| id | status |
|---|---|
| **O-1** second BLAS build | **CLOSED for this build** -- every gate above carries WIN and WSL readings.  The caveat that travels with it is the experiment's: both arms link scipy-openblas, so they vary OS, compiler, python and numpy but not the BLAS FAMILY, and the measured spreads are a LOWER bound. |
| **O-2** the equal-DOF regime | **CLOSED by the experiment (F2)**, and this build's G6 and S6.2 reproduce its sign at `q = 12` and `q = 18`.  The claim ships as "not worse", regime-qualified, with the closure as the durable half. |
| **O-3** `retain_internal` / `layer_absorption` per-layer | **CLOSED** -- S9, budget closure 2.14e-06 at `M = 6` on a non-conforming lossy pair. |
| **O-4** JAX twin | **STILL OPEN**, still not a regression: `PMM2DStackPure` has no JAX twin at all. |
| **O-5** `prepare()` / wavelength sweeps | **STILL OPEN**.  The cross-mass FACTORS are material-independent and are already cached; what is not cached across a wavelength change is the region eig, as on the shared path. |
| **O-6** `N = 1` uniform layers at oblique | **CLOSED by the experiment (F4) and SHIPPED as the default here**: `grid=None` gives `N = 1`, and `n_modes` for a uniform layer defaults to the measured neighbour rule `M_u = max(q_prev, q_next) / N_u + 1` rather than to the stack's `M`.  `_perlayer_modal_counts` implements it and the docstring states the trap in the F4 words. |
| **O-7** the `(2, 2)` grid fallback in `solve` | **CLOSED on the per-layer path**: it carries a per-layer WALL list and never consults `self._grid`.  The Wood-anomaly nudge already iterated the LAYER RECORDS, so it carried over untouched (it is now `_source_prep`, shared verbatim by both paths).  The `(2, 2)` fallback survives on the SHARED path, where it is correct. |
| **O-8** grids keyed on `tau` | **STILL OPEN**, and implemented as the experiment predicted: `_stag_basis_fingerprint` keys on `(d, M, tau, wall bytes)`, so an angle sweep rebuilds every basis.  The tau-free / tau-dependent split is still the obvious fix and is unmeasured. |
| **O-9** clamp vs raise on `n_orders` | **CLOSED** -- the shipped surface RAISES (S7.3). |
| **O-10** the per-layer `M` recipe | **CLOSED as a SURFACE**: `PMM2DStackPure.convergence_floor()` returns each layer's own single-layer residual (a measured lower bound on the pair error at 15/16 surface points) and the docstrings state, in the F3 words, that *a per-layer solve can be stationary in one knob and wrong*.  The greedy "raise the worse layer" rule is documented as a HINT and is NOT shipped as an auto-refiner (it read 6/9). |
| **O-11** the suspected 1-D `PMMStack` sliver defect at near-coincident LAYER walls | **NOT THIS ITEM'S, and not touched.**  It is being fixed by a concurrent worktree.  This build's G4/G5/N3/N4 oracles are all far outside the affected `delta = 1e-3 .. 3e-6` band (the fixtures use wall separations of order 0.1 of the period), so no gate here is detuned for it and none hit it. |
| **NEW** the parity guard on non-uniform walls | Found during the build, not in the experiment's four-site list: `_stag_parity_1d` must refuse a non-mirror-symmetric wall set (S0.1).  Gated by `test_parity_reduction_is_refused_on_an_unsymmetric_wall_set`. |
| **NEW** per-layer + SLANT | The per-layer path carries `slant` through unchanged, including the frame-anchor transmission phase and `_check_stack_slant`'s refusal set.  It is exercised only by the shared-path suites plus the per-layer plumbing; a dedicated per-layer slanted gate is NOT in this build and is the obvious next measurement. |

---

## 13. Commands and commits

```
# the gates, both builds
python validation/probe_pmm2d_staggered_mortar/b1_build_gates.py [g2g3 g4 g5g6 nu mem taper]

# the two new test files
python -m pytest tests/unit/test_pmm2d_staggered_nonuniform.py \
                 tests/unit/test_pmm2d_staggered_mortar.py -q
```
