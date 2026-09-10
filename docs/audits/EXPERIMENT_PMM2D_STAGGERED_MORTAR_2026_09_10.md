# EXPERIMENT -- per-layer element grids (L2 mortar) for the PURE staggered 2-D PMM

**Date:** 2026-09-10 · **Branch:** `probe/pmm2d-staggered-mortar` (worktree
`C:/tmp/lum_mortarp`, off `main` `fb3fd93`) · **Class:** prototype + measurement,
**no library code**
**Target:** `PMM2DStackPure` (`lumenairy/elements/pmm/stack2d_pure.py`), whose
union-grid constraint is roadmap item §1 of
`docs/audits/ROADMAP_PMM_PER_LAYER_GRIDS_2026_07_28.md`
**Prior art transplanted:** the 1-D per-layer surface
(`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md`), its mortar algebra
(`_core.py` `_interface_smatrix_mortar`, `_interface_smatrix_general_mortar`,
`_redheffer_star_rect`), and its conditioning lessons
(`PMM_M1_CONDITIONING_2026_08_04.md`)
**Everything below is measured** by a committed script in
`validation/probe_pmm2d_staggered_mortar/` (commands in that directory's
`README.md`). Evidence tags: **[M]** measured, **[A]** arithmetic on the tree
as it stands, **[H]** hypothesis, flagged.

**Build.** Windows 11, python 3.14.6, numpy 2.4.4 + MKL, scipy 1.17.1,
`OMP/OPENBLAS/MKL_NUM_THREADS=1`. Every wall time is single-threaded and was
taken with several probe processes sharing the box, so timings carry ~10-20 %
contention noise; the RATIOS quoted are within-script (both arms measured in
the same process, back to back) except where stated. **No second BLAS build was
run** -- see §13, open item O-1.

---

## 0. VERDICT: GO -- with a narrower scope and a different mechanism than the roadmap assumed

The mortar is **correct** (§3, §4), **convergent on genuinely non-conforming
grids** (§7), **serves the generalized / out-of-plane cascade** (§8), and
**carries no conditioning cliff** in the useful range (§10). At **equal
degrees of freedom** -- the only comparison that is not rigged -- it is
**7.6x / 7.3x more accurate** than the union grid in the under-converged
regime and a **wash (1.48x) once both arms converge** (§7.3, three points
against an exact independent oracle), while its lossless closure is 170x-3300x
tighter throughout. On a 3-slice staircase it is 2.3x / 4.8x / 2.6x more
accurate at identical eigenproblem sizes, and on the LCM = 12 staircase it runs
at a `q` the union lattice **cannot reach at all** (§9).

Three findings reshape the item, and two of them contradict the roadmap:

1. **The roadmap's central worry does not exist, and its central prerequisite
   does not either.** The roadmap (§1.2 item 1, and correction S-1) says the
   2-D per-layer grid needs "its own resolution PLUS the two neighbours' -- the
   interface-conforming window, the 1-D lesson that own-walls-only FAILS".
   **In this basis a window is unrepresentable**: Granet's segmentation is a
   UNIFORM lattice (`Basis1D.xb = linspace`), so the only lattice containing
   two layers' walls is the **LCM** lattice -- which is the union grid itself.
   There is no local enrichment, so per-layer grids here are necessarily
   own-walls-only. **And own-walls-only WORKS** (§7.3): the 1-D failure was a
   nodal-SEM boundary-layer defect at ARBITRARY wall positions, and it does not
   transfer.
2. **What the union grid actually buys is h-refinement, not conformity** --
   and per degree of freedom, p-refinement on the own-walls grid buys MORE
   while the answer is still converging and the SAME once it has converged
   (§7.3; the `q = 30` point that bounds this is reported there, not hidden).
   The eig dimension is `2 (N (M-1))^2`, a function of the PRODUCT
   `q = N (M-1)` alone, so "per-layer grids" do not save work by themselves:
   they save work only because a layer with few walls needs a smaller `q` than
   the LCM lattice forces on it.
3. **The 1-D mortar's `kron(I_2, .)` block operator is WRONG here**, and
   silently so on symmetric cells. In 2-D the two transverse components live in
   DIFFERENT tensor-product spaces and the Eq.-25 H partner SWAPS them, so the
   H row must be tested with the V1/V2 blocks exchanged (§2.2). This is the one
   piece of the transplant that is genuinely new physics-of-the-basis rather
   than transcription.

**What is NOT delivered by this item, and must not be claimed for it:** the
roadmap's headline use case, a *tapered pillar* whose walls move by ~1.8 nm per
slice, remains **unreachable** -- not because of the mortar but because a
1.8 nm wall offset on a 700 nm period needs `N ~ 390` on a uniform lattice, and
per-layer grids do not make an unrepresentable wall representable (§1.3). This
item unlocks staircases whose slice widths are **low-order rationals of the
period** (the LCM stays small per layer even when it is large across the
stack), which is the affordable-taper class, not the arbitrary one. The
enabling change for arbitrary tapers is still non-uniform segment boundaries in
`Basis1D` (campaign item N-1), exactly as roadmap correction S-1 states.

---

## 1. What "per-layer grids" can mean in this basis

### 1.1 The lattice is uniform [A]

`Basis1D.__init__` sets `self.xb = np.linspace(0.0, self.d, self.N + 1)`
(`twod_staggered.py:174`, comment *"segment boundaries on the eps walls
(Eq.31, uniform)"*) with a single scalar jacobian `self.J = 0.5 * self.h`.
There are no per-layer *wall positions* in the pure stack -- only a per-layer
*segment count* `N` on a uniform lattice, and `PMM2DStackPure` additionally
requires `Nx == Ny` (`stack2d_pure.py` `_validate_stag_cell`, and a bare
`assert` in `Granet2DTransverseE.__init__`).

Consequences that shape everything else:

* **A per-layer grid is one integer `N_i`** (plus, if the API allows it, a
  per-layer modal count `M_i`).
* **The common refinement of two grids is `LCM(N_a, N_b)`**, and it is the only
  lattice that carries both wall sets. So the union-grid reference for a
  non-conforming pair is exact and computable -- which is what makes every
  measurement below possible -- and it is also why *window enrichment has no
  analogue*: widening layer `i`'s grid to contain its neighbours' walls means
  `LCM(N_{i-1}, N_i, N_{i+1})`, i.e. the union.
* **Which per-layer grids are ADMISSIBLE is a geometry question, not a knob.**
  A pillar of width `1/2` exists on `N in {2,4,6,...}` and one of width `1/3` on
  `N in {3,6,...}`. Asking for the `1/2` pillar "on N=3" silently changes the
  device (it becomes `2/3` wide). The API must derive `N_i` from the layer's
  own `eps_cell`, never accept it as a free parameter for a patterned layer.
  (This probe made exactly that mistake in a first cut of §6 and the resulting
  arm was withdrawn.)

### 1.2 Cost is a function of `q = N (M-1)` alone [A]

`_region_modes` runs `sla.eig(L, G)` on a `2 q^2` pencil with `q = N (M-1)`
(`twod_staggered.py`: `Basis1D.dim = N*(M-1)`, `Granet2DTransverseE.q = bx.dim`,
`dimtot = 2*q*q`). The out-of-plane generator is `4 q^2` but Cholesky-whitened
to a standard eig. So

| `q` | eig dim `2q^2` | GB / matrix (`16 n^2`) | flop (`~30 n^3`) |
|---|---|---|---|
| 12 | 288 | 0.001 | 7.2e8 |
| 18 | 648 | 0.005 | 8.2e9 |
| 24 | 1152 | 0.017 | 4.6e10 |
| 30 | 1800 | 0.043 | 1.7e11 |
| 42 | 3528 | 0.166 | 1.3e12 |
| 60 | 7200 | 0.691 | 1.1e13 |

**Per-layer grids buy nothing by being per-layer.** They buy exactly one thing:
the freedom for layer `i` to choose its own `q_i`, instead of every layer paying
`q = N_LCM (M-1)`. Whether that is a win is an accuracy-per-`q` question, and
§7.3 answers it.

### 1.3 What this item does NOT unlock [A]

Roadmap correction S-1's arithmetic stands unchanged: a 2-degree sidewall over
310 nm at `n_slice = 6` moves a wall ~1.8 nm per slice, which on a 700 nm
period needs `N ~ 390` to land on a uniform lattice -- `q = 390 (M-1)`,
i.e. impossible at any `M`. **Per-layer grids do not make an unrepresentable
wall representable.** They make a staircase affordable when each slice's own
width is a low-order rational of the period (§9), and they make the LCM across
the stack irrelevant. Arbitrary tapers still need non-uniform `Basis1D`
segments (campaign item N-1).

---

## 2. Design

Notation: layers `A` (above) and `B` (below) at one interface, on grids
`(N_A, M_A)` and `(N_B, M_B)`; `q_X = N_X (M_X - 1)` per axis and
`qq_X = q_X^2` per component. Field spaces (Granet Eq. 34, and the module's
own block comment):

```
V1 (E1 = Ex) = B(x1)      (x) Btilde(x2)      Gram G1 = kron(Mtt_y, Mbb_x)
V2 (E2 = Ey) = Btilde(x1) (x) B(x2)           Gram G2 = kron(Mbb_y, Mtt_x)
```

with the eigensolver's index convention `I = jx + qx*jy`, i.e. `np.kron(Ky, Kx)`
(y slow, x fast). `G = -Rmat = blkdiag(G1, G2)`.

### 2.1 The mortar space and the pairing

**Choice: the 1-D convention, unchanged.** Tangential **E** is tested against
grid **B**'s trace space and tangential **H** against grid **A**'s -- the
classic mode-matching pairing, which keeps the system square for
`n_A != n_B` and is what `_interface_smatrix_mortar` already implements. No
third (union) mortar space is introduced: a union trace space would need the
LCM lattice at every interface, which is precisely the cost the item exists to
avoid, and it would make the interface system rectangular in both directions.

With `<u, v> = INT conj(u) v` (the inner product `Basis1D._global_matrix`
implements -- it conjugates the LEFT coefficients):

```
MassE_B  W_B (cb+ + cb-) = CrossE^H W_A (ca+ + ca-)      [2 qq_B equations]
MassH_A  V_A (ca+ - ca-) = CrossH   V_B (cb+ - cb-)      [2 qq_A equations]
```

and then, verbatim from the 1-D mortar with `A = (MassE_B W_B)^-1 CrossE^H W_A`
and `B = (MassH_A V_A)^-1 CrossH V_B`:

```
S11 = (I + BA)^-1 (I - BA)      S12 = 2 (I + BA)^-1 B
S21 = A (I + S11)               S22 = A S12 - I
```

`A` is `(2 qq_B) x (2 qq_A)`, `B` is `(2 qq_A) x (2 qq_B)`, `BA` is square --
the same shape the 1-D cascade already carries, so
`_core._redheffer_star_rect` is reused **unchanged**.

### 2.2 The block operator is NOT `kron(I_2, .)` -- this is the new physics

In 1-D both transverse components live in the SAME nodal space, which is why
the shipped mortar applies one 1-D mass blockwise as `kron(I_2, M)`
(`_kron2_apply`). **That does not carry over.** Here `E1 in V1` and `E2 in V2`
are different spaces, and the Eq.-25 dual puts **H2 in the V1 placement and H1
in the V2 placement** -- read off `_region_modes` (`rot = [-Dual[qq:];
Dual[:qq]]`, so `V[:qq]` holds V2-space coefficients and `V[qq:]` V1-space
ones) and confirmed by `PMM2DStackPure._flux_at`, which pairs `H[qq:]` with
`G1` and `H[:qq]` with `G2`. Therefore

```
MassE_X = blkdiag(G1_X, G2_X) = -Rmat_X      CrossE = blkdiag(C1, C2)
MassH_X = blkdiag(G2_X, G1_X)   (SWAPPED)    CrossH = blkdiag(C2, C1)
```

**The swap is silent on a conforming interface**: when the grids are identical
`C1 = G1` and `C2 = G2`, so `A = W_B^-1 W_A` and `B = V_A^-1 V_B` whichever
order the H blocks are written in -- the M1-class identity test **cannot see
this bug**. It is measured instead by a fail-before switch
(`mortar2d.H_BLOCK_SWAP`) on a genuinely non-conforming pair (§5.3).

### 2.3 The cross-mass, and its exact Kronecker factorization

`C1[i,j] = INTINT conj(phi^A_{1,i}) phi^B_{1,j} dx dy` and likewise `C2`. Both
factor **exactly** -- both spaces are tensor products and both partitions are
rectangular:

```
C1 = kron( Ctt_y(A,B), Cbb_x(A,B) )        C2 = kron( Cbb_y(A,B), Ctt_x(A,B) )
```

where `Cbb`/`Ctt` are 1-D cross-masses between the SAME named global set
(`B` / `Btilde`) on the two partitions, computed by Gauss-Legendre on the
**union of the two segment partitions** -- exact, because on each union
sub-interval both sides are polynomials of degree `<= M-1`, so `M_A + M_B + 2`
points integrate the product exactly. (Granet 2023 §3E makes the same
observation for non-aligned material boundaries: *"well-known Gauss-Legendre
quadrature formulas are used"*.) Near-coincident walls of the two lattices
appear only in this **integration mesh**, never as spectral elements -- the
property the 1-D `_sem_cross_mass` docstring calls "the decisive difference
from the shared union grid".

Two 2-D-specific facts:

* **The cross-mass is COMPLEX**, unlike the 1-D nodal SEM's. `Basis1D` glues
  its periodic hat with `tau = exp(-i alpha0 d)` (Eq. 33), so the basis itself
  carries the Bloch phase and both the mass and the cross-mass are complex
  Hermitian-consistent. `Cab^T` in the 1-D algebra becomes `Cab^H` here. Using
  the transpose would be a silent error at normal incidence (`tau = 1`).
* **The dense form is a rejected design** (roadmap §1.2 item 2, and it is worse
  than the roadmap estimated in 2-D): a dense `C1` is `q_A^2 x q_B^2`. Measured
  (§10.2): 49.4 MB at `N=(6,12), M=6` against 0.055 MB for its two factors --
  900x -- and the separable apply is exact to 3.8e-16 and 49x faster. The
  prototype's `kron_apply` is the production form.

### 2.4 The generalized twin (out-of-plane / slant)

Any out-of-plane layer returns DISTINCT forward/backward sets
(`_region_modes_oop`) and puts the whole stack on the generalized cascade, so
the mortar needs the twin of `_interface_smatrix_general_mortar`:

```
[ CrossE^H Wb_a  -MassE_B Wf_b ] [ca-]   [ -CrossE^H Wf_a  MassE_B Wb_b ] [ca+]
[ MassH_A  Vb_a  -CrossH  Vf_b ] [cb+] = [ -MassH_A  Vf_a  CrossH  Vb_b ] [cb-]
```

`2 qq_B + 2 qq_A` rows against `m_a + m_b = 2 qq_A + 2 qq_B` unknowns -- square
for both the in-plane pencil (`m = 2 qq` by the `[W; -V] <-> -lam` symmetry,
written out by `_modes_as_general`) and the `4 q^2` out-of-plane generator
after its flux split. Same V1/V2 swap on the H rows. Measured in §8.

### 2.5 Half-spaces, far field, order capacity

**Choice: each half-space is built on the grid of the layer it touches** -- the
1-D convention (`stack.py:_solve_vertical_perlayer`), and it is more strongly
motivated here than there:

* both end interfaces become PLAIN square matches, so **no mortar sits where the
  far field is projected**, and the M1 identity of §3 is exact rather than
  approximate at the ends;
* the eps-free geometric eig cache (`_homog_geom_cache`) is per grid and is what
  makes the two half-spaces free; a separate half-space grid would need its own
  cache AND add two mortar interfaces at the two most observable planes;
* `_far_projector_2d` is then built per END grid (two projectors instead of
  one), which costs nothing measurable.

**The consequence must be designed for, not discovered.** The far-field
Rayleigh capacity is set by the END layers' grids: the per-axis projection has
`q = N_end (M_end - 1)` columns, so `n_orders <= (q - 1) // 2`. A stack whose
first layer sits on `N=2, M=4` has `q = 6` and can retain only `|m| <= 2`. The
prototype clamps to `min` over the two end grids and reports
`n_orders_used`; this is the **T3-3 lesson applied before it bites** -- T3-3 was
exactly a cap computed from a union while the half-spaces lived on window grids,
and its failure mode (a rank-deficient `Hsup`, a build-dependent minimum-norm
draw, energy-invisible) is available here verbatim. In this probe the clamp
fired on every arm whose first layer was coarse, and `_guarded_lstsq` refused
nothing.

### 2.6 Conditioning: what M1 says, and what it does not

M1's conclusions transfer without change of instrument:

* a backward-stable `solve` needs no guard (LAPACK `gesv`'s residual is ~eps
  whatever the conditioning), so the two mortar `solve`s stay `solve`s;
* the exposure is the explicit inverse, i.e. the cascade denominator `I + BA`,
  and one guard there covers the function;
* the discriminator is the **equilibrated** reciprocal condition
  `_rcond_1_equilibrated`, free once the inverse exists;
* and M1's **withdrawn refusal stays withdrawn** -- §10 reports the census and
  refuses nothing, because M1 S2.7 established that the 2-D methods sit inside
  the 1-D "broken" band while being right. Nothing in this probe reopens that.

The two star denominators are already guarded inside `_redheffer_star_rect`,
which this design reuses.

---

## 3. M1 -- the conforming identity [M]

`m1_conforming.py`. Two- and three-region stacks on ONE grid, driven through
the mortar path with the identical-grid bypass DISABLED (`force_mortar=True`),
against the shipped `PMM2DStackPure` union-grid cascade. Relative to
`max(R, T)` for R/T and to `max|J|` for the Jones.

| case | `dR` | `dT` | `dJones` | closure, union | closure, mortar |
|---|---|---|---|---|---|
| stripe A \| stripe B, M=5, theta=0 | 3.26e-15 | 5.66e-15 | 5.40e-15 | 6.9e-05 | 6.9e-05 |
| stripe A \| stripe B, M=5, theta=0.20 | 1.27e-15 | 1.15e-14 | 4.97e-15 | 1.8e-05 | 1.8e-05 |
| stripe \| pillar, M=6, theta=0.20 | 1.41e-13 | 1.67e-13 | 2.31e-13 | 1.7e-06 | 1.7e-06 |
| pillar \| pillar, M=6, CONICAL (0.25, 0.7) | 8.75e-15 | 2.92e-14 | 5.95e-14 | 9.0e-07 | 9.0e-07 |
| stripe \| uniform \| pillar, M=5, theta=0.15 | 1.00e-14 | 2.25e-14 | 3.92e-14 | 1.8e-05 | 1.8e-05 |

**Worst over the five: 2.31e-13 relative.** The mortar reduces to the plain
square modal match. Bit identity is not attainable and is not claimed: the
mortar multiplies by the block Gram `G` on both sides of a `solve` where the
plain interface does not, so the identity is exact algebraically and rounds at
`eps * cond_2(G)`. Measured (`m0c_gram_cond.py`):

| grid | `cond_2(G1)` | `cond_2(G2)` | `eps * cond` |
|---|---|---|---|
| N=2, M=5 | 3.06e+03 | 3.04e+03 | 6.8e-13 |
| N=3, M=6 | 7.23e+03 | 7.31e+03 | 1.6e-12 |
| N=6, M=4 | 8.85e+02 | 8.85e+02 | 2.0e-13 |
| N=3, M=8 | 3.50e+04 | 3.53e+04 | 7.8e-12 |

The worst M1 reading, 2.31e-13 on a `N=3, M=6` stack, sits **7x inside** that
grid's `eps * cond` -- so the bar is derived, not fitted, and it must be
derived AT RUNTIME (it grows with `M`; a fixed `1e-12` constant would fail at
`M = 8`). A three-region pillar/pillar case scored the same way in §6 reads
**2.73e-14** on `(R, T)` and 2.39e-13 on the Jones.

**And the identical-grid BYPASS is bit-exact**: with `force_mortar` off, a
per-layer stack whose grids happen to coincide takes the shipped
`_interface_smatrix` and returns `0.000e+00` against `PMM2DStackPure`
(`m2b_converge.py`, "CONFORMING control"). That is the 1-D contract
(`test_identical_wall_stack_is_bit_exact_vs_shared`) reproduced in 2-D, and it
is what would make a `grid_mode='per_layer'` switch safe to expose without
moving any existing user's bits.

## 4. The two controls that make every later gap readable [M]

`m0_smoke.py` and `m0b_diag.py`. Written first, because without them a
resolution gap and a mortar defect are indistinguishable -- and the first
reading of §5 looked exactly like a defect.

### 4.1 Algebra smoke

| check | worst relative |
|---|---|
| `cross_mass_1d(b, b, set)` vs `Basis1D.mass(set, set)` (both sets, N=2/3/4) | **9.80e-16** |
| `kron_apply(Ky, Kx, X)` vs the dense `np.kron(Ky, Kx) @ X` | **3.22e-16** |
| V1 / V2 Grams built from the 1-D factors vs the eigensolver's `-Rmat` blocks | **0.0** (bit-identical) |

The last row is worth stating plainly: the mortar's E-row mass operator is not
merely *like* the eigensolver's block field Gram, it **is** it, bit for bit --
so an implementation reads it off `-Rmat` instead of rebuilding it.

### 4.2 The h-refinement control -- the SHIPPED engine only, no mortar anywhere

A single stripe layer (`eps` 2 / 6, duty 1/2, `theta = 0.20`) drawn on `N=2` and
on its EXACT `N=4` refinement, both through `pmm_efficiency_2d_staggered`:

| M | rel `dR` (N=2 vs N=4) | rel `dT` | `sum R` at N=2 | at N=4 |
|---|---|---|---|---|
| 4 | 3.59e-01 | 6.13e-01 | 0.300360 | 0.338079 |
| 5 | 4.21e-01 | 1.13e-01 | 0.448572 | 0.338711 |
| 6 | 4.65e-02 | 5.22e-02 | 0.324702 | 0.340008 |
| 7 | 6.35e-03 | 1.37e-02 | 0.340920 | 0.340044 |

**Two lattices that represent the SAME geometry exactly disagree by 36-42 % at
M=4-5 with no mortar in sight.** Any per-layer-vs-union comparison at equal `M`
inherits this. §7.3 is the comparison that does not.

### 4.3 The transparent-interface probe -- the mortar's own error, isolated

One uniform slab (`n = 2`, 300 nm) SPLIT into two sub-layers on DIFFERENT
grids. The interface is physically absent, so the exact answer is the analytic
Fresnel slab; there is no geometry to resolve, so nothing but the mortar can
move the answer. TE (incident `E_y`), `|R - R_exact|`:

| theta | M | grids (2,2) | (2,4) nested | (2,3) NON-conforming | (3,4) NON-conforming |
|---|---|---|---|---|---|
| 0.00 | 5 | 2.8e-30 | 7.9e-29 | 1.7e-28 | 1.7e-28 |
| 0.00 | 7 | 1.6e-28 | 2.8e-27 | 7.5e-27 | 6.6e-27 |
| 0.20 | 5 | 2.7e-11 | 1.9e-11 | 1.9e-11 | 7.1e-13 |
| 0.20 | 7 | 8.8e-17 | 1.9e-16 | **4.7e-16** | 2.7e-16 |

and the energy closure on the same runs at `theta = 0.20, M = 7`:
3.9e-15 / 9.1e-14 / 3.3e-14 / 5.2e-14.

**A non-conforming mortar interface reproduces the analytic Fresnel answer to
4.7e-16 and conserves energy to 3.3e-14, and is not measurably worse than the
conforming one.** The mortar adds no leak of its own; every gap in §5-§9 is
resolution. (The `M = 5` row is the uniform layer's own degree-limited Bloch
phase at oblique -- the caveat `stack2d_pure`'s docstring already carries -- and
it moves the conforming pair by the same 2.7e-11.)

## 5. M2 -- nested grids

`m2_nested.py`, `m2b_converge.py`. Layer A a duty-1/2 stripe on `N=2`; layer B a
2-D pillar cell that needs `N=4`; the union reference is A re-expressed exactly
on `N=4`. `theta = 0.20`, `n_orders` 2 (the mortar's SUP half-space rides layer
0's `N=2` grid, capacity `q = 2(M-1)` -- §2.5).

### 5.1 Joint modal ladder (both arms at the same M)

| M | `dR` | `dT` | `dJones` | closure ref | closure mortar | t ref | t mortar |
|---|---|---|---|---|---|---|---|
| 4 | 2.91e-01 | 3.46e-01 | 2.83e-01 | 2.8e-04 | 2.8e-03 | 1.8 s | 0.7 s |
| 5 | 6.28e-01 | 5.76e-01 | 5.78e-01 | 4.1e-06 | 3.8e-04 | 7.0 s | 4.6 s |
| 6 | 3.41e-01 | 5.08e-01 | 4.24e-01 | 1.8e-07 | 5.8e-05 | 22.6 s | 13.9 s |
| 7 | 1.86e-01 | 2.40e-01 | 2.36e-01 | 3.7e-09 | 9.9e-07 | 68.8 s | 45.0 s |

Read with §4.2 in hand, this table says nothing about the mortar: layer A on
`N=2` is not resolved below `M ~ 7`, and the union arm hands it `N=4`.

### 5.2 The per-layer modal count IS the answer

Hold layer B at `(N=4, M=6)` and walk `M_A` on A's own `N=2` grid -- something
the union grid structurally cannot do (it carries one `M` for the whole stack):

| `M_A` | vs union ref M=6 | vs union ref M=8 | closure | wall |
|---|---|---|---|---|
| 5 | 5.67e-01 | 5.72e-01 | 3.8e-04 | 23.5 s |
| 7 | 2.45e-01 | 2.49e-01 | 8.7e-07 | 26.0 s |
| 9 | 1.33e-02 | 3.26e-02 | 1.8e-07 | 36.5 s |
| 11 | 1.85e-02 | 3.71e-02 | 1.8e-07 | 47.0 s |
| 13 | 1.10e-02 | 2.11e-02 | 1.8e-07 | 102.9 s |

**Honest limit of this measurement:** the reference's OWN convergence is
`2.84e-02` between M=6 and M=8, so from `M_A = 9` onward the mortar arm and the
reference agree **to within the reference's own uncertainty**, and this fixture
cannot resolve further. §7 replaces it with an exact independent oracle. What
M2 does establish is the collapse from 5.7e-01 to ~1e-02 driven by `M_A` alone,
at 1.6x the wall time -- the gap is A's resolution, bought off on A's own small
grid.

### 5.3 FAIL-BEFORE: the V1/V2 swap on the H row is load-bearing [M]

Same stack, `M_A = 11`, scored against the union reference at M=8:

| `H_BLOCK_SWAP` | error vs reference | `abs(R+T-1)` |
|---|---|---|
| `False` (the naive 1-D-looking same-order blocks) | **7.28e+01** | **8.22e+01** |
| `True` (the design of §2.2) | **3.71e-02** | **1.75e-07** |

Three orders on the observable and nine on energy closure. Note that **§3's
identity test passes either way**: on identical grids `C1 = G1`, `C2 = G2` and
the swap cancels (§2.2). A conforming-parity gate alone would have shipped this
bug -- which is the single most important gate lesson this probe produced.

## 6. M3 -- non-conforming grids, four pairs against ONE reference [M]

`m3_nonconforming.py`. Pillar pair, `theta = 0.18`, `phi = 0.35`, periods
1.2 um, `wl` 0.85 um. Layer A's pillar is 1/2 wide (`N in {2,6}`), layer B's is
1/3 wide (`N in {3,6}`), so the union reference is `N=6` and four grid pairs
score against it. `(3,6)` is NOT in the table because a 1/2-wide pillar does not
exist on `N=3` -- see §1.1; a first cut of this probe ran it and was silently
measuring a different device.

`M = 4`; union reference 19.4 s, its own `R+T-1 = -3.5e-06`:

| grids (A,B) | kind | `d(R,T)` | `dJones` | `R+T-1` (two-sided) | wall | speedup |
|---|---|---|---|---|---|---|
| (6,6) | conforming, mortar FORCED | **2.73e-14** | 2.39e-13 | [-1.3e-05, -3.5e-06] | 22.1 s | 0.88x |
| (2,6) | nested, A coarse | 2.87e-01 | 1.29e+00 | [-4.8e-03, -8.1e-04] | 12.8 s | 1.52x |
| (6,3) | nested, B coarse | 2.76e-01 | 9.20e-01 | [+2.5e-05, +3.0e-03] | 13.7 s | 1.42x |
| (2,3) | NON-conforming, both coarse | 2.50e-01 | 1.35e+00 | [-3.4e-03, -1.8e-03] | **0.2 s** | **97.7x** |

The reading is the important part. **Coarsening EITHER layer costs ~28 %, and
the fully non-conforming pair costs no more than either nesting alone**
(2.50e-01 against 2.87e-01 and 2.76e-01) while running 98x faster. There is no
"non-conformity penalty" stacked on top of the resolution cost -- consistent
with §4.3, where the mortar's own error is 1e-16.

`M = 5` on the same fixture (union reference 171.8 s, `R+T-1 = +1.5e-07`):

| grids (A,B) | `d(R,T)` | `dJones` | `R+T-1` | wall | speedup |
|---|---|---|---|---|---|
| (6,6) forced mortar | **2.94e-14** | 9.72e-14 | [+6.7e-08, +1.5e-07] | 158.1 s | 1.09x |
| (2,6) | 2.15e-01 | 7.80e-01 | [-9.8e-05, -3.1e-05] | 108.1 s | 1.59x |
| (6,3) | 1.02e-01 | 1.92e-01 | [-5.4e-05, -4.2e-05] | 129.6 s | 1.32x |
| (2,3) | 2.28e-01 | 8.86e-01 | [-1.2e-04, -4.9e-05] | **1.2 s** | **146x** |

Closure is scored TWO-SIDED (deficit and excess are equally defects on a
lossless stack). It tightens by ~1.5 orders from `M=4` to `M=5` on every arm
and reaches 6.9e-12 by `M = 11` in §7 -- i.e. the closure defect tracks the
coarse layers' resolution and decays with it, not with the interface kind.

### 6.1 The same ladder with a HERMITIAN (lossless anisotropic) tensor

Layer A's pillar replaced by a GYROTROPIC tensor
`[[6, 0.9i, 0], [-0.9i, 6, 0], [0, 0, 5]]` -- Hermitian, hence lossless, hence
`R + T = 1` is still exact and the two-sided gate still applies. `M = 4`, union
reference 32.7 s, `R+T-1 = +1.8e-05`:

| grids (A,B) | `d(R,T)` | `dJones` | `R+T-1` | wall | speedup |
|---|---|---|---|---|---|
| (6,6) forced mortar | **1.41e-14** | 7.98e-14 | [-2.9e-06, +1.8e-05] | 37.5 s | 0.87x |
| (2,6) | 2.29e-01 | 3.26e-01 | [-2.8e-04, +1.4e-03] | 20.0 s | 1.64x |
| (6,3) | 2.45e-01 | 2.90e-01 | [-2.8e-04, +6.8e-04] | 20.6 s | 1.59x |
| (2,3) | 2.06e-01 | 4.69e-01 | [+3.9e-04, +2.4e-03] | **0.2 s** | **141x** |

and at `M = 5` (union reference 203.9 s, `R+T-1 = +1.1e-07`):

| grids (A,B) | `d(R,T)` | `dJones` | `R+T-1` | wall | speedup |
|---|---|---|---|---|---|
| (6,6) forced mortar | **1.38e-14** | 2.93e-14 | [+3.5e-08, +1.1e-07] | 166.4 s | 1.23x |
| (2,6) | 3.95e-01 | 4.42e-01 | [-5.5e-05, -4.2e-05] | 89.4 s | 2.28x |
| (6,3) | 2.12e-02 | 4.12e-02 | [-2.6e-05, +4.2e-05] | 84.0 s | 2.43x |
| (2,3) | 3.86e-01 | 4.46e-01 | [-8.3e-05, -5.0e-05] | **1.0 s** | **212x** |

The conforming identity holds on a tensor layer to **1.41e-14 / 1.38e-14**, the
two-sided closure tightens by 1.5 orders from `M=4` to `M=5` on every arm, and
the non-conforming arms behave exactly as the scalar ones -- the mortar is a
geometric projection and carries no material dependence, which is the same
reason the 1-D mortar is "geometry-level and gauge-free"
(`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` §5.1). Note the gyrotropic
`M=5` arms are NOT monotone in `M` (the (2,6) arm reads 2.29e-01 at `M=4` and
3.95e-01 at `M=5`); at these coarse resolutions the answer is not yet in its
asymptotic regime, exactly as the scalar h-refinement control of §4.2 shows for
the shipped engine with no mortar present.

## 7. M4 -- the stripe pair against an EXACT INDEPENDENT ORACLE

`m4_stripe_1d.py`, `m4b_perlayer_M.py`, `m4c_equal_dof.py`. A y-uniform stripe
stack with a DIFFERENT duty per layer: layer A duty 1/2 (`N=2`), layer B duty
1/3 (`N=3`) -- genuinely non-conforming, common refinement `N=6`. Because the
stack is y-uniform, the **exact 1-D pure PMM** (`PMMStack`, degree 14, no
Fourier floor, no corner residual) is the oracle for the whole stack, so both
arms are scored against TRUTH instead of against each other. `theta = 0.20`,
period 0.9 um, `wl` 0.6 um; the score is `max` over the retained `n = 0` orders
of `|R - R_1D|`, `|T - T_1D|` on the TE row. Oracle self-gap degree 12 vs 14:
**2.92e-08**, i.e. four decades below every entry in the tables.

### 7.1 The mortar converges to the exact answer

| M (both layers) | error vs 1-D | mirrored assignment | closure | wall |
|---|---|---|---|---|
| 5 | 3.95e-01 | 3.95e-01 | 4.3e-04 | 0.8 s |
| 7 | 2.99e-02 | 1.41e-01 | 7.7e-07 | 7.7 s |
| 9 | 1.73e-03 | 1.13e-01 | 4.0e-09 | 47.6 s |
| 11 | **3.33e-04** | 1.11e-01 | **6.9e-12** | 246.6 s |

Three decades of convergence on the observable and eight on lossless closure.
The "mirrored assignment" column is the shipped suite's anti-mirror tripwire
(`test_staggered_per_order_orientation_vs_1d`): from `M >= 7` the direct
assignment beats the mirrored one by 4.7x / 65x / 333x, so the per-layer
cascade is not merely converging, it is converging in the RIGHT order slots.

Union grid on the common refinement `N=6`, same oracle:

| M | error vs 1-D | closure | wall |
|---|---|---|---|
| 4 | 7.01e-03 | 9.8e-06 | 16.1 s |
| 5 | 5.56e-04 | 8.4e-08 | 141.4 s |

### 7.2 The per-layer modal count, priced

Layer B held at `(N=3, M=7)`, `M_A` walked on A's own `N=2` grid
(`m4b_perlayer_M.py`); eig-work is `sum_regions dim^3`:

| `M_A` | error vs 1-D | wall | eig-work |
|---|---|---|---|
| 7 | 2.99e-02 | 12.5 s | 2.96e+08 |
| 9 | 2.15e-03 | 16.1 s | 4.06e+08 |
| 11 | 8.99e-04 | 29.1 s | 7.84e+08 |
| 13 | 6.45e-04 | 99.0 s | 1.80e+09 |
| 15 | 5.91e-04 | 291.3 s | 4.13e+09 |

and the union arm measured in the SAME process for a fair wall-time comparison:

| M | error vs 1-D | wall | eig-work |
|---|---|---|---|
| 4 | 7.01e-03 | 28.0 s | 5.44e+08 |
| 5 | 5.56e-04 | 196.1 s | 3.06e+09 |
| 6 | 2.17e-05 | 561.4 s | 1.17e+10 |

Two readings, and the second is the more useful one:

* **In the working band the per-layer curve sits below the union's**: at ~16 s
  the per-layer arm reads 2.15e-03 against the union's 7.01e-03 at 28 s, and at
  29 s it reads 8.99e-04 where the union needs 196 s to reach 5.56e-04.
* **But it PLATEAUS at 5.9e-04**, because `M_B = 7` on layer B's `N=3` grid is
  now the limiting error and no amount of `M_A` touches it. The union arm,
  which raises both layers together, walks past it to 2.17e-05.
  **The per-layer lever must be applied to the LIMITING layer**, which means a
  per-layer convergence check (raise each `M_i` in turn and watch the answer),
  not one global `M`. That is a real usability cost of the API and it belongs
  in the docstring, not in a footnote: the union grid's single `M` is
  *convenient* precisely because it cannot be mis-set per layer.

### 7.3 THE DECISIVE MEASUREMENT: equal degrees of freedom

The union grid does not merely "cost more". It spends its extra degrees of
freedom **h-refining on the LCM lattice**, and the per-layer route spends its
own **p-refining on each layer's own lattice**. The staggered basis makes the
comparison exact: per axis `q = N (M-1)`, so `q_union(M) = 6(M-1)` equals
`q_A(M_A) = 2(M_A - 1)` at `M_A = 3M - 2` and `q_B(M_B) = 3(M_B - 1)` at
`M_B = 2M - 1` -- and at those settings **every region eigenproblem in both arms
has exactly the same dimension `2 q^2`**.

| `q` | eig dim | arm | settings | error vs 1-D | closure | wall | ratio mortar/union |
|---|---|---|---|---|---|---|---|
| 18 | 648 | union | `N=6, M=4` | 7.01e-03 | 6.3e-05 | 23.6 s | |
| 18 | 648 | **mortar** | `A: N=2, M=10` / `B: N=3, M=7` | **9.17e-04** | **3.7e-07** | 29.2 s | **0.13x** |
| 24 | 1152 | union | `N=6, M=5` | 5.56e-04 | 2.1e-07 | 191.2 s | |
| 24 | 1152 | **mortar** | `A: N=2, M=13` / `B: N=3, M=9` | **7.58e-05** | **2.9e-10** | 218.8 s | **0.14x** |
| 30 | 1800 | union | `N=6, M=6` | **2.17e-05** | 1.0e-08 | 592.7 s | |
| 30 | 1800 | mortar | `A: N=2, M=16` / `B: N=3, M=11` | 3.22e-05 | **3.0e-12** | 749.3 s | **1.48x** |

**At the two working `q` the mortar is 7.6x and 7.3x MORE accurate, and its
lossless closure is 170x and 720x tighter, at 1.15-1.24x the wall time. At
`q = 30` the advantage is GONE on the observable (1.48x the union's error) --
and this third point is reported because it bounds the claim, not despite
bounding it.** By `q = 30` both arms are within 3e-05 of the exact answer,
i.e. both are essentially converged and the residual is no longer the
discretisation this comparison is about; the mortar's energy closure is still
3300x tighter there (3.0e-12 vs 1.0e-08), so the two arms are not converging
to different places -- the observable ordering at the converged end is a
coin-toss between two right answers.

**The defensible statement, therefore:** *at equal DOF the mortar is several
times more accurate in the UNDER-CONVERGED regime -- which is the regime a
staircase campaign actually runs in -- and a wash (within 1.5x) once both arms
converge.* The "h-vs-p penalty" the roadmap's window-enrichment argument
predicts is **absent**; "reversed" is true only where it matters and is not
claimed beyond that.

Why, stated as physics rather than as a ratio: the tangential trace at the
interface is `C^0` with kinks at BOTH layers' walls, and h-refinement at the
NEIGHBOUR's walls (all the union grid adds) buys only the kink; p-refinement on
the own-walls grid buys the kink algebraically AND the smooth interior
spectrally, and the interior is where the modal content lives -- until both are
resolved, at which point neither distribution matters. The 1-D own-walls-only
failure (75-83 % spread,
`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` §3) was a nodal-SEM
boundary-layer defect in the wall-normal `Ex` channel at ARBITRARY wall
positions; on Granet's uniform lattice with a modified-Legendre basis it does
not reproduce. **[H, flagged]** the mechanism above is an interpretation, not a
proof; the numbers are the claim (open item O-2).

## 8. M5 -- mixed kinds through the GENERALIZED mortar twin [M]

`m5_oop_mixed.py`. An out-of-plane tensor layer (tilted-director LC,
`uniaxial_tensor(1.5, 1.7, 35 deg, phi=25 deg)`) takes the first-order `4 q^2`
staggered generator, returns DISTINCT forward/backward sets and puts the whole
stack on the generalized cascade -- so the mortar needs the twin of §2.4.

### 8.1 Independent oracle: a uniform OOP slab SPLIT across grids

One uniform OOP slab (350 nm, `n_sup` 1.0, `n_sub` 1.5) cut into two halves on
different grids. The interface is physically absent, so `berreman_jones_1d` is
the exact answer, and nothing about it may depend on the split. Relative
`dJones`:

| theta | M | (2,2) | (2,3) NON-conf | (3,4) NON-conf |
|---|---|---|---|---|
| 0 deg | 5 | 2.34e-14 | 1.49e-13 | 1.56e-13 |
| 0 deg | 7 | 8.62e-14 | 1.57e-13 | 3.54e-13 |
| 25 deg (conical, phi = 40 deg) | 5 | 2.73e-09 | 4.10e-08 | 1.84e-09 |
| 25 deg (conical, phi = 40 deg) | 7 | 1.37e-13 | 2.18e-13 | **3.23e-13** |

`|R+T-1|` on the same runs is 1.1e-13 or better at `M=7` on every grid pair.
**The generalized mortar reproduces Berreman through a non-conforming interface
to 3.2e-13 at conical incidence** -- the same statement §4.3 makes for the
square twin, on the harder cascade.

### 8.2 Patterned OOP-over-scalar against the union-grid generalized cascade

Layer A an OOP pillar (`eps_host` 2.25, LC core), layer B a scalar patterned
cell; `theta = 15 deg`, `phi = 30 deg`; relative `dJones` vs the union arm:

| pair | M | `dJones` vs union | closure union | closure mortar | t union | t mortar | speedup |
|---|---|---|---|---|---|---|---|
| nested (2,4) -> union 4 | 4 | 3.21e-01 | 6.5e-04 | 8.4e-03 | 1.9 s | 1.0 s | 1.95x |
| nested (2,4) -> union 4 | 5 | 6.15e-02 | 3.6e-05 | 8.8e-05 | 10.7 s | 6.5 s | 1.65x |
| NON-conf (2,3) -> union 6 | 4 | 2.23e-01 | 2.8e-05 | 7.7e-04 | 18.3 s | 0.2 s | **78.4x** |
| NON-conf (2,3) -> union 6 | 5 | 3.24e-02 | 4.0e-07 | 1.4e-04 | 158.2 s | 1.2 s | **133x** |

Converging at the equal-`M` (unequal-DOF) rate §5 and §6 established for the
scalar cascade, with no additional penalty for the generalized form, and the
non-conforming pair runs two orders faster because its LCM reference is `N=6`
while its own grids are `N=2` and `N=3`.

## 9. M6 -- the z-staircase, the use case

`m6_staircase.py` (equal `M`), `m6c_staircase_1d.py` (equal DOF, exact oracle).
A 3-slice pillar taper. Because Granet's lattice is uniform, a slice of pillar
width `1/N` lives on `N` and on every multiple of `N`, and the union stack pays
`LCM` for **every** slice:

* case **A**: widths 1/2, 1/3, 1/6 -> per-slice `N = 2, 3, 6`, `LCM = 6`
  (the union reference is computable, so accuracy can be scored);
* case **B**: widths 1/2, 1/3, 1/4 -> per-slice `N = 2, 3, 4`, `LCM = 12`
  (the case per-layer grids exist for).

### 9.1 Equal modal count -- and why that framing understates the result

Case A, 2-D pillars, against the union-grid staircase on `N=6`:

| M | `d(R,T)` | `dJones` | closure union | closure mortar | t union | t mortar | speedup |
|---|---|---|---|---|---|---|---|
| 4 | 2.06e-01 | 2.73e-01 | 1.6e-05 | 1.8e-04 | 23.0 s | 9.1 s | 2.52x |
| 5 | 2.90e-01 | 9.10e-02 | 1.6e-07 | 3.3e-04 | 169.9 s | 94.6 s | 1.80x |

At equal `M` the union slice `N=6` carries `q = 6(M-1)` while the `N=2` slice
carries `2(M-1)` -- three times the DOF -- so this table measures the DOF
difference, exactly as §5.1 did. It is reported because it is the naive
comparison a user would make, and the answer to it is: **don't**.

### 9.2 Equal DOF, against the exact 1-D oracle

The same staircase made y-uniform (stripes), so the exact 1-D `PMMStack` at
degree 14 is the oracle for the whole 3-slice stack. Parameterised by the
per-axis DOF `q`, at which **every region eigenproblem in both arms is the same
`2 q^2`**; per-slice `M_i = q / N_i + 1`:

**Case A** (`N = 2, 3, 6`, `LCM = 6`):

| `q` | eig dim | arm | settings | error vs 1-D | closure | wall |
|---|---|---|---|---|---|---|
| 12 | 288 | union | `M = 3` | 1.56e-01 | 2.3e-03 | 2.1 s |
| 12 | 288 | **per-layer** | `M_i = 7, 5, 3` | **6.89e-02** | 2.4e-03 | 2.5 s |
| 18 | 648 | union | `M = 4` | 1.84e-02 | 3.3e-05 | 41.4 s |
| 18 | 648 | **per-layer** | `M_i = 10, 7, 4` | **3.84e-03** | 3.1e-05 | 46.4 s |
| 24 | 1152 | union | `M = 5` | 1.04e-03 | 2.2e-07 | 206.5 s |
| 24 | 1152 | **per-layer** | `M_i = 13, 9, 5` | **3.93e-04** | 2.2e-07 | 276.3 s |

**2.3x / 4.8x / 2.6x more accurate at identical eigenproblem sizes**,
reproducing §7.3's sign on a three-region staircase -- and here the advantage
does NOT decay across the measured range (ratios 0.44 / 0.21 / 0.38), where the
two-layer stripe pair's did by `q = 30`. Neither range reaches the converged
regime the stripe pair's `q = 30` point sits in, so this is not evidence
against that point; it is evidence that the useful-`q` advantage survives a
third region and two mortar interfaces.

**Case B** (`N = 2, 3, 4`, `LCM = 12`) -- and here the union arm runs out of
lattice before it runs out of budget:

| `q` | eig dim | arm | settings | error vs 1-D | closure | wall |
|---|---|---|---|---|---|---|
| 12 | 288 | union | -- | **UNREACHABLE** (`M >= 3` forces `q >= 24` on `N=12`) | -- | -- |
| 12 | 288 | **per-layer** | `M_i = 7, 5, 4` | 4.94e-02 | 4.3e-04 | **2.3 s** |
| 24 | 1152 | **per-layer** | `M_i = 13, 9, 7` | **1.79e-04** | 6.7e-10 | 261.1 s |

### 9.3 The structural result: the union lattice has a DOF FLOOR [A + M]

`Basis1D` requires `M >= 3`, so the smallest `q` any lattice can carry is
`2 N`. On the union lattice that is `q >= 2 N_LCM`; per-layer it is
`q_i >= 2 N_i`. For case B (`N = 2, 3, 4`, `LCM = 12`):

| | smallest attainable `q` | eig dim per slice | eig work, 3 slices (`sum dim^3`) |
|---|---|---|---|
| union `N=12` | **24** | 1152 | 4.59e+09 |
| per-layer `N = 2, 3, 4` | **12** | 288 | 7.17e+07 |

**64x less eig work per slice, and it is a floor, not a tuning choice** -- the
union grid simply cannot be run at `q = 12` on this stack. The per-layer arm
reaches its `q = 12` point in a couple of seconds; the union arm's cheapest
possible point is already the per-layer arm's `q = 24` point.

The same arithmetic at production modal counts, `dim = 2 (N (M-1))^2`,
`16 dim^2` bytes per matrix (case B, 3 slices):

| M | per-layer dims (`N = 2,3,4`) | `sum dim^3` | union `N=12` dim | `sum dim^3` (x3) | GB / matrix | eig-work ratio |
|---|---|---|---|---|---|---|
| 4 | 72 / 162 / 288 | 2.85e+07 | 2592 | 5.22e+10 | 0.100 | **1832x** |
| 6 | 200 / 450 / 800 | 6.11e+08 | 7200 | 1.12e+12 | 0.772 | **1832x** |
| 8 | 392 / 882 / 1568 | 4.60e+09 | 14112 | 8.43e+12 | 2.97 | **1832x** |

At `M = 8` the union arm needs a **2.97 GB** matrix and `scipy.linalg.eig`
holds several live; the per-layer arm's largest is 0.037 GB. **This is the
memory-budget acceptance gate roadmap correction S-3 asked for, and per-layer
grids clear it by three orders.**

## 10. M7 -- conditioning census, and the separable cross-mass [M]

`m7_conditioning.py`. Four non-conforming configurations (`M3 pillar (2,3)`,
`M3 pillar (2,6)`, `M4 stripe (2,3)`, `M6 taper (2,3,6)`) at `M = 4, 5, 6`,
instrumented at the three mortar sites with M1's own free instrument,
`_rcwa_core._rcond_1_equilibrated` (the equilibrated reciprocal 1-condition,
exact and computed from the inverse that already exists).

### 10.1 The census

| config | `M = 4` | `M = 5` | `M = 6` | mortar calls / solve |
|---|---|---|---|---|
| M3 pillar (2,3) | 4.57e-05 | 1.48e-05 | 5.29e-06 | 3 |
| M3 pillar (2,6) | 1.28e-05 | 2.90e-06 | 1.08e-06 | 3 |
| M4 stripe (2,3) | 2.01e-05 | 3.30e-05 | 1.90e-05 | 3 |
| M6 taper (2,3,6) | 3.27e-05 | 1.09e-05 | 4.39e-06 | 6 |

(worst equilibrated `rcond` over all sites in that solve; 3 sites per mortar
interface, so a 2-layer stack has 3 calls and the 3-slice taper has 6.)

Per site, over all 15 solves:

| site | calls | dim | equilibrated `rcond` | below M1's `1e-8` screen |
|---|---|---|---|---|
| mortar E-row `MassE_B W_B` (a `solve`) | 15 | 128-1800 | [1.08e-06, 1.01e-04] | **0** |
| mortar H-row `MassH_A V_A` (a `solve`) | 15 | 128-1800 | [7.00e-06, 3.88e-04] | **0** |
| **mortar interface `I + BA`** (the explicit inverse) | 15 | 128-1800 | [3.42e-04, 1.85e-02] | **0** |

**No cliff.** The worst reading anywhere is 1.08e-06, two orders above M1's
screen, and it *improves* with `M` on three of the four configurations (the
stripe pair is flat) -- the opposite of a cliff, which is what one expects when
the conditioning is set by the basis's own Gram rather than by the grid
mismatch.

Three readings worth stating explicitly:

* **The guarded site is the best-conditioned of the three.** `I + BA` reads
  [3.4e-04, 1.85e-02] while the two `solve` operands read decades worse -- the
  SAME ordering M1 measured in 1-D (`M_b W_b` cond 3.4e+02..3.8e+06,
  `M_a V_a` 7.3e+05..2.1e+07, `I + BA` 1.3e+01..6.8e+03). The 2-D transplant
  therefore inherits M1's conclusion unchanged: **keep the two `solve`s as
  `solve`s** (LAPACK `gesv` is backward stable, so a residual screen on them
  measures nothing) and guard `I + BA`, which carries the compounded exposure
  of both.
* **Nothing is refused, and nothing should be.** M1's inverse refusal was
  WITHDRAWN (M1 S2.7) because correct 2-D methods read inside the 1-D broken
  band; this probe does not reopen that, and the census is an instrument here,
  not a gate.
* **`_guarded_lstsq` refused nothing** across every run in this document, once
  the far-field order cap is derived from the END grids (§2.5). It is the
  T3-3 class and the clamp is what keeps it quiet -- an uncapped `n_orders` on
  a coarse end grid is exactly the rank-deficient minimum-norm draw T3-3
  documented, and it is energy-invisible.

### 10.2 The cross-mass must never be materialised [M]

The design claim of §2.3, priced. `C1 = kron(Ctt_y, Cbb_x)`; the "dense" column
is `np.kron(...)`'s footprint, the "factors" column is the two 1-D
cross-masses:

| grids (A,B) | M | dense `kron` | factors | memory | apply speed | identity |
|---|---|---|---|---|---|---|
| (2,3) | 6 | 0.34 MB | 0.005 MB | 75x | 2.0x | 3.2e-16 |
| (3,6) | 6 | 3.09 MB | 0.014 MB | 225x | 6.8x | 3.7e-16 |
| (4,6) | 8 | 21.10 MB | 0.036 MB | 588x | 36.4x | 4.3e-16 |
| (6,12) | 6 | 49.44 MB | 0.055 MB | **900x** | **49.2x** | 3.8e-16 |

The factorisation is an IDENTITY (3-4e-16, i.e. BLAS reassociation only), the
memory ratio grows as `q_A q_B` and the apply grows with it. At the case-B
staircase's `(N=4) | (N=12)` interface the dense form is already 49 MB **per
component per interface**; there are two components and `nlay + 1` interfaces.
This is not an optimisation, it is the difference between the design working
and not.

## 11. The verdict against the rule

The brief's rule: **GO only if M1, M3, M4 pass at derived bars with a bounded
non-conforming remainder and no conditioning cliff in the useful range.**

| requirement | reading | verdict |
|---|---|---|
| **M1** conforming identity | worst 2.31e-13 relative over 5 stacks incl. conical and 3-region (§3); the identical-grid BYPASS is bit-exact (0.0) | **PASS** -- and the bar is DERIVED: `eps * cond_2(G)` is 1.6e-12 on that grid (measured, §3), so the reading sits 7x inside it |
| **M3** non-conforming vs the common refinement | (2,3) at `M=4` is 2.50e-01 from the union reference, and that is RESOLUTION, not the mortar: the same coarsening applied to EITHER layer alone costs the same 2.8e-01 (§6), the mortar's own error on a transparent interface is 4.7e-16 (§4.3), and the observable converges 3.95e-01 -> 3.33e-04 with closure 4.3e-04 -> 6.9e-12 against an EXACT oracle (§7.1) | **PASS** |
| **non-conforming remainder bounded** | it is not merely bounded, it is not resolvable above round-off: 1e-28 (normal) / 4.7e-16 (oblique) scalar, 3.2e-13 conical out-of-plane, and a non-conforming pair is never worse than a conforming one at the same DOF | **PASS** |
| **M4** vs the validated 1-D oracle, per order | 3.33e-04 at `M=11` with the anti-mirror tripwire at 333x (§7.1); at EQUAL DOF the mortar beats the union grid 7.6x / 7.3x under-converged and loses by 1.48x once converged, with closure 170x-3300x tighter at every point (§7.3) | **PASS** -- the requirement is a bounded remainder, and the bound is now two-sided |
| **no conditioning cliff** | equilibrated `rcond` IMPROVES with `M` on 3 of 4 configurations and is flat on the fourth; worst reading anywhere 1.08e-06, two orders above M1's `1e-8` screen; 0 of 45 site-calls screened in, nothing refused, and `_guarded_lstsq` refused nothing once the far-field cap is derived from the END grids (§10) | **PASS** |
| M2 nested | converges with `M_A` to inside the reference's OWN uncertainty (§5.2) | **PASS**, weakly -- the fixture cannot resolve further; §7 supersedes it |
| M5 generalized / out-of-plane | Berreman to 3.2e-13 through a non-conforming interface at conical incidence (§8.1) | **PASS** |
| M6 staircase | 2.3x / 4.8x / 2.6x more accurate at identical eigenproblem sizes (`q` = 12 / 18 / 24); a 64x eig-work FLOOR advantage on the LCM=12 case, where the union lattice cannot be run at all below `q = 24`, and 1832x at production `M` (§9) | **PASS** |
| M7 conditioning census | the guarded site (`I + BA`) is the BEST-conditioned of the three, reproducing M1's 1-D ordering, so M1's "keep the solves as solves, guard the explicit inverse" transplants unchanged (§10.1) | **PASS** |

**GO.** With the scope statement of §0: this delivers **low-order-rational
staircases and mixed-resolution stacks**, not arbitrary tapers.

## 12. Integration route

### 12.1 API -- mirror the 1-D spelling exactly

`PMMStack` spells it `layer_grids="shared" | "per-layer"` -- a HYPHEN, not an
underscore, validated at `stack.py:184-187` -- with `window_halfwidth=1`
(`stack.py:176`, which itself raises when set with `layer_grids='shared'`,
`stack.py:199-204`). Mirror the first, **refuse the second**:

```python
PMM2DStackPure(period_x, period_y=None, *, n_superstrate=1.0, n_substrate=1.0,
               n_modes=8, degree=None, n_orders=7,
               layer_grids="shared")          # NEW: "shared" | "per-layer"

stack.add_layer(thickness, *, eps=None, eps_cell=None,
                grid=None,                    # NEW: uniform layers only
                n_modes=None)                 # NEW: per-layer modal count
```

* `layer_grids="shared"` is the default and is **today's code path unchanged**
  -- the union-grid `raise` in `add_layer` stays exactly as it is.
* `layer_grids="per-layer"` **drops the union-grid raise**. A patterned layer's
  `N` comes from its own `eps_cell` (already validated square by
  `_validate_stag_cell`); it is NEVER a free parameter (§1.1 -- accepting one
  silently changes the device).
* `grid=` applies to UNIFORM layers only, which have no walls of their own;
  default to the previous layer's `N` (never smaller than 1). `N = 1` is legal
  and is the cheapest region in the library -- but see open item O-6.
* `n_modes=` per layer is **the lever that makes the whole item pay** (§7.2,
  §7.3, §9.2). Default to the stack's `n_modes`.
* `window_halfwidth=` must **raise** with the reason: on a uniform lattice a
  window is `LCM(N_{i-1}, N_i, N_{i+1})`, i.e. the union grid -- there is no
  local enrichment to widen (§1.1). This mirrors how the 1-D path raises on
  `stabilize='slices'` with its reason stated.

### 12.2 Functions

In `twod_staggered.py`, beside `Basis1D` (they are basis facts, not stack
facts):

| new | what |
|---|---|
| `_stag_cross_mass_1d(ba, bb, which)` | the 1-D cross-mass of §2.3. ~35 lines. Memoize on a geometry fingerprint exactly as `_core._sem_cross_mass_cached` does (`_geo_fingerprint` -> `(d, N, M, tau)` here), through the same `ByteBudgetedLRU` registry so `clear_asm_caches` / `cache_report` see it. |
| `_stag_grid_ops(px, py, N, M, taux, tauy)` | per-grid bases + the four 1-D masses. **These already exist** as `Granet2DTransverseE._axis_mats`' `Mtt_x/Mbb_x/Mtt_y/Mbb_y`, and their krons ARE `-Rmat`'s blocks bit-for-bit (§4.1) -- so factor `_axis_mats` out rather than duplicating it. |
| `_stag_kron_apply(Ky, Kx, X)` | the separable apply of §2.3. ~8 lines. |

In `pmm/_core.py`, beside the 1-D mortar (so the two live together and the
1-D's conditioning comment block covers both):

| new | what |
|---|---|
| `_interface_smatrix_mortar_2d(...)` | §2.1 + §2.2. ~30 lines; the only structural difference from its 1-D sibling is that `_kron2_apply` becomes a two-DIFFERENT-operators block apply, with the H row swapped. |
| `_interface_smatrix_general_mortar_2d(...)` | §2.4. ~30 lines. |
| `_redheffer_star_rect` | **REUSED UNCHANGED** -- it is already dimension-agnostic and already carries M1's star-denominator guards. |

In `stack2d_pure.py`: the driver changes are the per-grid geometric-eig cache
(keyed `(N, M)` instead of once), the interface dispatch (plain when
`(N_a, M_a) == (N_b, M_b)`, mortar otherwise), the two per-end far-field
projectors, and the order cap. ~80 lines. The prototype
`validation/probe_pmm2d_staggered_mortar/mortar2d.py::MortarStack2D` is the
working shape.

### 12.3 `retain_internal` / `layer_absorption`

Both work, with one real change and one non-change:

* **`_internal_amplitudes` is grid-independent** -- it reads only the retained
  partial cascades, `lam_f/lam_b` and thicknesses -- but `S_above` /
  `S_below_bot` become RECTANGULAR, so their recurrences must go through
  `_redheffer_star_rect` instead of `_redheffer_star`. This is verbatim what
  `stack.py:1800-1810` already does on the 1-D per-layer path.
* **`_flux_at` must become per-layer.** It currently reads one shared Gram
  `d["G"]` and one `d["qq"]`; per-layer they are `G_i = blkdiag(G1_i, G2_i)`
  and `qq_i`. **No new assembly**: `G_i` is `-Rmat_i`, and its factors are in
  the layer's `GridOps` already (§4.1). The flux form itself
  (`Re(h2^H G1 e1 - h1^H G2 e2)`) is unchanged.
* `layer_absorption`'s calibration (`F_top[0] / (1 - R_tot)`) is unchanged --
  it is a half-space flux ratio and the half-spaces are conforming by
  construction.
* The honest cross-machinery check `sum_i A_i == 1 - sum R - sum T` becomes
  the natural gate, exactly as on the 1-D path (which measured 5e-3
  non-conforming against <1e-10 conforming).

### 12.4 Gates for the build agent

G1-G2 and G4-G5 transplant the 1-D set; **G3 is new and is the one that matters
most**.

| gate | claim | measured here |
|---|---|---|
| G1 | conforming per-layer stack is BIT-EXACT vs `PMM2DStackPure` (identical grids bypass the mortar) | 0.000e+00 (§3) |
| G2 | conforming identity through the FORCED mortar path, bar DERIVED AT RUNTIME as a small multiple of `eps * cond_2(G)` measured on the stack's own grids -- NOT a fixed constant (it grows with `M`: 6.8e-13 at `M=5`, 7.8e-12 at `M=8`) | worst 2.31e-13, 7x inside its grid's `eps * cond` (§3) |
| **G3** | **the H-row V1/V2 swap is load-bearing**: disabling it on a NON-conforming stack must move the observable by >= 2 orders | 7.28e+01 vs 3.71e-02; closure 8.22e+01 vs 1.75e-07 (§5.3). **G1/G2 cannot see this** -- a conforming-parity-only gate would ship the bug |
| G4 | transparent interface across NON-conforming grids reproduces the analytic Fresnel slab and `berreman_jones_1d` | 4.7e-16 scalar oblique; 3.2e-13 out-of-plane conical (§4.3, §8.1) |
| G5 | stripe stack per order vs the exact 1-D `PMMStack`, with the anti-mirror tripwire | 3.33e-04 direct vs 1.11e-01 mirrored at `M=11` (§7.1) |
| G6 | EQUAL-DOF non-regression: at matched `q` the per-layer arm is **within a small factor** of the union arm on the observable and **not worse** on lossless closure. The observable half must NOT be pinned as "better" -- it is 7.6x / 7.3x better at `q` = 18 / 24 and 1.48x worse at `q` = 30 (§7.3). The closure half is the durable one (170x-3300x tighter, no decay) | §7.3, §9.2 |
| G7 | two-sided lossless closure, scalar AND Hermitian tensor | §6, §10 |
| G8 | far-field order cap derived from the END grids, with a fail-before switch reproducing the unclamped draw | the T3-3 pattern; clamp implemented and exercised in the probe (§2.5) |
| G9 | conditioning census recorded per site; nothing refused in the useful range | §10 |

**Cost of the build: 3-5 days**, not the roadmap's 2-4 weeks. The reduction is
real and has a reason: the roadmap budgeted for per-layer *wall sets*, union
bookkeeping, window enrichment and a symmetry-fold interaction, and roadmap
corrections S-1 and S-2 already deleted two of those. On a uniform lattice a
"grid" is one integer, the cross-mass is two small 1-D matrices, and
`_redheffer_star_rect` already exists.

## 13. Open items

| id | item |
|---|---|
| **O-1** | **No second BLAS build.** Everything here is Windows/MKL. Before any bar in this document becomes a test constant, re-measure on the WSL/OpenBLAS CI proxy -- `TESTING_STANDARDS.md` rule 5 ("bars need a gap on both sides", with the measured cross-build envelope) is not satisfied by one build, and this campaign's own history (M1 S2.7) is the reason. |
| **O-2** | **The equal-DOF advantage is regime-dependent and the regime is not mapped.** Stripe pair: 0.13x / 0.14x / **1.48x** at `q` = 18 / 24 / 30; staircase: 0.44x / 0.21x / 0.38x at `q` = 12 / 18 / 24. The advantage is real where the answer is not yet converged and gone where it is, on the one pair measured to convergence. Two things are unmeasured: (a) a corner-dominated 2-D pillar pair, where the corner cap makes convergence ALGEBRAIC and h-refinement should do relatively better -- this is the case most likely to reverse the sign at working `q`, and it is the campaign's actual device class; (b) whether the closure advantage (170x-3300x, and it does NOT decay) is the more durable signal. Do (a) before quoting any accuracy ratio in a campaign. |
| **O-3** | `retain_internal` / `layer_absorption` per-layer is DESIGNED (§12.3) but **not prototyped and not measured**. The per-layer Gram is the only real change and it is free, but the closure `sum A_i = 1 - R - T` must be measured on a non-conforming lossy stack before the surface is claimed. |
| **O-4** | JAX twin: not attempted, and not a regression -- `PMM2DStackPure` has no JAX twin today. |
| **O-5** | `prepare()` / wavelength sweeps: the geometry (bases, Grams, cross-mass FACTORS) is material-independent and cacheable forever, and the 1-D roadmap §3 sketch applies verbatim -- **cheaper here**, because what is cached per interface is two small 1-D matrices rather than a dense cross-mass. |
| **O-6** | A UNIFORM layer at `N = 1` is the cheapest region the engine can express, but a uniform layer at oblique incidence is degree-limited (`stack2d_pure`'s own caveat: its Bloch phase must be resolved in the modified-Legendre basis), and §4.3 shows that cost directly -- 2.7e-11 at `M=5` against 8.8e-17 at `M=7`, on grids of every kind. Measure `N=1` uniform layers at oblique before defaulting to them. |
| **O-7** | The Wood-anomaly nudge (`_grazing_safe_wavelength`) and the tensor-diagonal gather in `PMM2DStackPure.solve` iterate the layer list, not a union cell, so they carry over -- but the `Nx, Ny = self._grid` fallback (`(2, 2)` when no patterned layer exists) has no per-layer meaning and must be replaced by the per-layer grid list. |
| **O-8** | Grids are held per `(N, M)` AND per Bloch phase (`tau = exp(-i alpha0 d)`), so an angle sweep re-builds every basis. The 1-D geometry cache is keyed on content and has the same property; if angle sweeps become hot, split `Basis1D` into a tau-free part (the elementary matrices, which are tau-independent) and the tau-dependent glue. Not measured. |
| **O-10** | **Per-layer `M` needs a per-layer convergence procedure.** §7.2 measures the failure mode directly: walking `M_A` alone plateaus at 5.9e-04 because layer B's own `M_B = 7` is the limiting error, while the union grid's single `M` walks past it to 2.17e-05. The build must ship a documented recipe (raise each `M_i` in turn, take the answer stationary in ALL of them) and the docstring must say that a per-layer solve can be stationary in one knob and wrong. |
| **O-9** | The probe's `MortarStack2D` clamps `n_orders` silently and reports `n_orders_used`. The shipped version should RAISE above capacity (the classical/JAX siblings' behaviour) rather than clamp, so a user asking for orders the end grids cannot carry is told, not quietly served fewer. |

---

# FOLLOW-UP 2026-09-09 -- open items O-1, O-2, O-10, O-6 closed; roadmap item N-1 (non-uniform segments) GO

**Class:** prototype + measurement, **no library code**.  Everything in this
section comes from a committed script in
`validation/probe_pmm2d_staggered_mortar/` (commands in that directory's
`README.md`).  Evidence tags as above.

**BUILD CORRECTION [M].**  The Build note at the top of this document says
"numpy 2.4.4 + MKL".  That is **wrong** and it is corrected here: the Windows
arm's numpy and scipy are both linked against **scipy-openblas 0.3.31.188.0**
(`np.__config__.CONFIG['Build Dependencies']['blas']`, measured
2026-09-09 by `f1_crossbuild.py`).  No MKL was ever in this campaign.  The
consequence is stated in F1 below: the second arm had to be chosen to differ in
something real.

## F1 (O-1) -- the second and third build, and the spread envelope [M]

`f1_crossbuild.py` runs the six decisive tables in ONE process so every arm
measures exactly the same fixtures; `f1_compare.py` diffs them.  Three arms:

| arm | OS / toolchain | python | numpy | BLAS |
|---|---|---|---|---|
| `win` | Windows 11, MSVC-built wheels | 3.14.6 | 2.4.4 | scipy-openblas 0.3.31, DYNAMIC_ARCH (Haswell build target) |
| `wsl` | WSL2 Ubuntu, gcc 14.2 wheels, glibc 2.39 | 3.12.3 | 2.4.6 | scipy-openblas 0.3.31, DYNAMIC_ARCH (SkylakeX build target) |
| `win_nehalem` | Windows 11, `OPENBLAS_CORETYPE=NEHALEM` | 3.14.6 | 2.4.4 | the SAME library forced onto its SSE kernels instead of the CPU's AVX2 ones (1.5x slower on a 600-dim complex eig, measured -- i.e. a genuinely different code path, not a relabel) |

**Why three.**  Both PyPI wheel families link scipy-openblas, and OpenBLAS's
`DYNAMIC_ARCH` picks its kernels from the CPU -- which is the SAME CPU under
WSL.  So `win` vs `wsl` varies OS, libm, compiler, python and numpy but very
likely NOT the BLAS kernel; `win_nehalem` varies the kernel and nothing else.
Between them the three arms do vary every layer, but this is still not an MKL /
Accelerate / reference-LAPACK arm, so **every spread below is a LOWER BOUND on
the true cross-build envelope** and a bar derived from it needs the usual
decades on top, not a factor of two.

### F1.1 The envelope, by family

93 quantities compared.

| family | n | worst spread | median spread | reading |
|---|---|---|---|---|
| M1 conforming identity (`dR`/`dT`/`dJones`) | 16 | **8.04e-01** | 4.35e-01 | pure round-off; the VALUE is meaningless across builds |
| isolated mortar vs analytic Fresnel (`dR`, closure) | 24 | **9.51e-01** | 6.16e-01 | same |
| OOP twin vs `berreman_jones_1d` (`dJones`, closure) | 16 | **8.46e-01** | 5.31e-01 | same |
| **M4 vs the exact 1-D oracle** (err, mirror, closure) | 12 | **8.37e-05** | 3.73e-10 | DISCRETISATION, not round-off -- reproducible to 4-9 decades |
| **M4c equal-DOF** (errors, closures, ratio) | 10 | **2.35e-03** | 2.20e-08 | same; the one 2.35e-03 is the `q=24` mortar CLOSURE at 2.85e-10, i.e. round-off again |
| conditioning `rcond` (per config, per site) | 14 | **1.06e-01** | 3.60e-11 | build-sensitive at the 10 % level on ONE configuration |

**The dividing line is not the quantity, it is what limits it.**  Anything whose
magnitude is set by DISCRETISATION reproduces across all three arms to
1e-4..1e-10 relative.  Anything whose magnitude is set by ROUND-OFF has an O(1)
cross-build spread -- 3.9e-15 vs 3.3e-16 vs 2.7e-15 for one closure reading, a
factor of 12.  That is not a defect; it is what a residual at `eps` does.

### F1.2 What this means for each gate bar in S12.4

| gate | quantity | measured spread | bar that survives it |
|---|---|---|---|
| G1 identical-grid bypass | `0.000e+00` on every arm | exact | **`== 0` stands** -- it is a bypass, not an arithmetic claim |
| G2 forced-mortar identity | worst reading 2.31e-13 / 1.07e-13 / 1.36e-13 (spread 5.4e-01) | O(1) | a MAGNITUDE bar only, and it must be **derived at runtime** from `eps * cond_2(G)` as S3 already says: the bar is 1.6e-12 on that grid, the worst reading over three builds is 2.3e-13, so the gap is 7x. That is ONE decade, not the usual three -- **raise the multiplier to `10 * eps * cond_2(G)`** and the gap becomes 70x with the reading's own build spread (2.2x) inside it |
| G3 H-row swap fail-before | 7.28e+01 vs 3.71e-02 | the claim is ">= 2 orders"; both numbers are discretisation-scale | **stands** |
| G4 transparent interface | 4.7e-16 / 3.2e-13 class | O(1) spread | magnitude bar with decades; do NOT pin 4.7e-16 |
| G5 M4 per order vs the 1-D oracle | 3.33e-04, spread 8.4e-05 relative across arms at the `M=9` rung | 1e-4 relative | **a tight bar is sound here** -- e.g. `err(M=9) < 3e-03` has 3 decades of gap on both sides |
| G6 equal-DOF non-regression | ratios 1.3078e-01 / 1.3078e-01 / 1.3078e-01 at `q=18` (spread 5.2e-09) | 1e-8 | **the RATIO is the most reproducible quantity in the whole document** -- bar it, not the errors |
| G7 two-sided closure | closure readings, spread up to 9.1e-01 | O(1) | magnitude bar; the CONVERGENCE (each rung tighter than the last by >= 1 decade) is the build-free restatement |
| G9 conditioning census | `M4 stripe (2,3)` M=4: 2.0097e-05 / 2.0100e-05 / **2.2434e-05** (spread 1.06e-01) | 10 % | M1's `1e-8` screen sits **2.0 decades** below the worst reading anywhere (1.08e-06); with a 10 % build spread that margin is real but it is now MEASURED rather than assumed |

**The one number that moved materially between arms is a conditioning
reading**, and it moved on the NEHALEM-kernel arm, not between operating
systems -- exactly the shape `TESTING_STANDARDS.md` S5 warns about (a census
quantity downstream of a near-degenerate operator).


## F2 (O-2) -- THE DEVICE REGIME: a corner-dominated 2-D pillar pair [M]

`f2_device_regime.py`.  O-2 named the one case most likely to REVERSE the
equal-DOF sign: a crossed 2-D pillar, whose four corners cap the convergence at
ALGEBRAIC, so h-refinement (all the union grid buys) should do relatively
better than p-refinement (all the own-walls grid buys).  Fixture: layer A a
pillar 1/2 of the period wide on its own `N = 2`, layer B a pillar 1/3 wide on
`N = 3`, common refinement `N = 6`; CONICAL incidence (`theta = 0.18`,
`phi = 0.35`), period 1.2 um, `wl` 0.85 um.

### F2.1 The reference and its own uncertainty

Because there is no exact oracle for a crossed 2-D pillar pair, the reference
is the union grid at the top of its OWN ladder, and that ladder is reported so
every statement below is bounded by it:

| union `M` | `q` | eig dim | `R(0,0)` | gap vs previous | closure | wall |
|---|---|---|---|---|---|---|
| 3 | 12 | 288 | 0.002640980 | -- | 5.6e-04 | 2.2 s |
| 4 | 18 | 648 | 0.004992238 | 2.12e-01 | 1.3e-05 | 14.6 s |
| 5 | 24 | 1152 | 0.005324533 | 2.75e-02 | 1.5e-07 | 86.0 s |
| 6 | 30 | 1800 | 0.005346876 | 8.71e-04 | 2.3e-09 | 376.7 s |
| **7** | **36** | **2592** | **0.005351023** | **4.45e-04** | 1.7e-11 | 1028.3 s |

**Reference = union `M = 7`; its own uncertainty is 4.45e-04.**  The ladder is
algebraic, as the corner cap requires -- the gap falls by 8.5x, 32x, 2.0x per
rung, nothing like the stripe pair's spectral collapse.

### F2.2 Equal DOF, scalar pillar pair

Per axis `q = 6(M-1)` on the union and `q_A = 2(M_A-1)`, `q_B = 3(M_B-1)` on
the own-walls grids, so `M_A = 3M-2`, `M_B = 2M-1` puts every region
eigenproblem in both arms at the same `2 q^2`:

| `q` | eig dim | union err | union closure | mortar err | mortar closure | mortar/union | readable? |
|---|---|---|---|---|---|---|---|
| 12 | 288 | 2.24e-01 | 5.6e-04 | **1.19e-01** | **5.6e-05** | **0.529x** | yes |
| 18 | 648 | 2.68e-02 | 1.3e-05 | **4.92e-03** | **5.4e-08** | **0.183x** | yes |
| 24 | 1152 | **1.15e-03** | 1.5e-07 | 2.75e-03 | **2.9e-11** | 2.399x | **NO** -- 1.15e-03 is 2.6x the reference's own 4.45e-04 uncertainty |

Wall time 1.7 / 18.0 / 103.6 s (mortar) against 2.2 / 14.6 / 86.0 s (union).

### F2.3 Equal DOF with an IN-PLANE LC TENSOR in layer A

Layer A's pillar replaced by a rotated-uniaxial director lying IN the plane
(`uniaxial_tensor(1.5, 1.9, theta = pi/2, phi = 35 deg)`, so `e13 = e23 = 0`
and the block-form Eq.-7 path runs).  Union ladder gaps 8.02e-02, 4.95e-03,
1.27e-03; reference = union `M = 6`, uncertainty **1.27e-03**:

| `q` | union err | union closure | mortar err | mortar closure | mortar/union |
|---|---|---|---|---|---|
| 12 | 8.91e-02 | 1.9e-04 | **4.40e-02** | **2.2e-05** | **0.494x** |
| 18 | 6.21e-03 | 2.7e-06 | **4.10e-03** | **1.7e-08** | **0.660x** |

### F2.4 VERDICT on O-2

**The equal-DOF advantage SURVIVES the corner cap in the under-converged
regime, and the crossing point moves EARLIER.**

* Scalar 2-D pillar pair: **0.53x at `q = 12`, 0.18x at `q = 18`** -- the
  mortar is 1.9x and 5.5x more accurate at identical eigenproblem sizes.  The
  `q = 24` point reads 2.40x the other way, but it is **not resolvable against
  this reference** (both arms sit inside 2.6x of the reference's own
  uncertainty), so the honest statement is *the advantage is gone somewhere
  between `q = 18` and `q = 24`*, against `q ~ 30` on the stripe pair (S7.3).
  The corner cap does what O-2 predicted -- it brings the crossing forward --
  but it does **not reverse the sign at working `q`**.
* Tensor arm: **0.49x and 0.66x**, same sign, and the advantage decays with `q`
  the same way.  The mortar carries no material dependence, as S6.1 already
  measured for the Hermitian gyrotropic cell.
* **The closure advantage does not decay and is LARGER here than on stripes**:
  10x / **249x** / **5142x** (scalar) and 8.6x / **160x** (tensor).  S7.3's
  "the closure half is the durable one" reproduces on the device class.

**So the accuracy ratio quoted in a campaign must be regime-qualified**: *below
`q ~ 18` on a corner-dominated 2-D pillar pair the mortar is 2-5x more accurate
at equal DOF; by `q ~ 24` the two arms are within a factor of a few of each
other and of the reference, and only the lossless closure still separates them
(by 3-4 decades).*  O-2 is CLOSED with that qualification.

## F3 (O-10) -- the per-layer `M` recipe, as a surface [M]

`f3_perlayer_M_recipe.py`, on the F2 pillar pair, scored against the union grid
at `M = 6` (`q = 30`, its own uncertainty 8.71e-04).

### F3.1 Each layer's OWN single-layer residual

The quantity a recipe would actually have a user compute: the layer ALONE
between the half-spaces on its own grid, against the same layer alone at the
top of its own ladder (`M_A = 13`, `M_B = 9`, both `q = 24`).

| layer A (`N=2`) | `q_A` | own residual | layer B (`N=3`) | `q_B` | own residual |
|---|---|---|---|---|---|
| `M_A = 5` | 8 | 2.03e-01 | `M_B = 4` | 9 | 5.18e-02 |
| `M_A = 7` | 12 | 5.25e-02 | `M_B = 5` | 12 | 2.42e-03 |
| `M_A = 9` | 16 | 8.55e-04 | `M_B = 6` | 15 | 1.84e-03 |
| `M_A = 11` | 20 | 8.49e-05 | `M_B = 7` | 18 | 7.30e-05 |

### F3.2 The two-knob surface (pair error vs the reference)

| | `M_B = 4` | `M_B = 5` | `M_B = 6` | `M_B = 7` |
|---|---|---|---|---|
| **`M_A = 5`** | 3.42e-01 | 2.28e-01 | 2.18e-01 | 2.14e-01 |
| **`M_A = 7`** | 2.82e-01 | 1.19e-01 | 5.43e-02 | 4.24e-02 |
| **`M_A = 9`** | 2.68e-01 | 8.92e-02 | 1.60e-02 | **4.90e-03** |
| **`M_A = 11`** | 2.79e-01 | 9.02e-02 | 1.80e-02 | 6.81e-03 |

**The plateau is real in BOTH directions, and it is the whole finding:**

* hold `M_B = 4`, walk `M_A` 5 -> 11: 3.42e-01, 2.82e-01, 2.68e-01, 2.79e-01
  -- **stationary at ~2.7e-01, and NOT MONOTONE**, so a user watching only
  `M_A` sees a converged answer that is wrong by 27 %;
* hold `M_A = 5`, walk `M_B` 4 -> 7: 3.42e-01, 2.28e-01, 2.18e-01, 2.14e-01
  -- stationary at ~2.1e-01, the mirror failure;
* only the DIAGONAL descends: (5,4) 3.42e-01 -> (7,5) 1.19e-01 -> (9,6)
  1.60e-02 -> (11,7) 6.81e-03.

The `(11,7)` cell reading 6.81e-03 against `(9,7)`'s 4.90e-03 is a genuine
non-monotonicity (both are 6-8x the reference's uncertainty), consistent with
S6.1: at corner-capped resolutions this basis is not yet in its asymptotic
regime and neither arm is monotone.

### F3.3 The RULE, tested two ways

**Candidate rule (greedy):** *raise `M` on the layer whose own single-layer
residual is larger.*  Tested at the 9 interior points by comparing the error
drop from one rung of `M_A` against one rung of `M_B`: **6 hits / 9**.  All
three misses are cells where layer A's own residual is larger but raising `M_B`
helps more -- at `(M_A=7, M_B=4)` the two own residuals are 5.25e-02 and
5.18e-02, a tie the rule cannot resolve, and the two rungs are not DOF-matched
(one rung of `M_B` on `N=3` is `+3` in `q`, one rung of this `M_A` ladder is
`+4`).  **A greedy rule on the raw residual is NOT reliable enough to ship.**

**What IS reliable -- the FLOOR rule:**
`pair_error >= max(own_residual_A, own_residual_B)` holds at **15 of 16**
surface points; the single exception (`M_A=7, M_B=7`: 4.24e-02 against
5.25e-02) misses by 1.2x.  The ratio `pair_error / max(own residual)` runs
1.03-80 and is smallest exactly where the plateau is (1.05, 1.07, 1.12 down the
`M_A = 5` row -- there the pair error IS layer A's own residual).

### F3.4 VERDICT on O-10: the recipe the build must ship

1. **The stopping criterion is stationarity in EVERY `M_i`, never in one.**
   Measured failure: stationary to 4 % across four rungs of `M_A` while 27 %
   wrong (F3.2, the `M_B = 4` column).  Stationarity in one knob is not
   evidence.
2. **Screen with the per-layer own residual first, because it is a measured
   LOWER BOUND on the pair error (15/16) and it costs one single-layer solve
   per layer per rung** -- one region eig instead of the stack's.  A layer whose
   own residual is 5e-02 makes a 1e-03 stack answer impossible, and that is
   knowable before the stack is ever assembled.
3. **Do NOT ship the greedy "raise the worse layer" rule as an auto-refiner**
   (6/9).  Ship it as a HINT and the floor as the gate.
4. The docstring must say, in these words, that *a per-layer solve can be
   stationary in one knob and wrong*, and must point at the floor screen.

## F4 (O-6) -- a UNIFORM layer on `N = 1` at oblique / conical [M]

`f4_uniform_oblique.py`.  O-6 asked whether a uniform layer may default to the
cheapest grid the engine can express.  The worry was real -- S4.3 measured a
uniform layer's own Bloch-phase error at 2.7e-11 (`M=5`) against 8.8e-17
(`M=7`) at `theta = 0.20` -- so the question is whether `N = 1` makes it worse.

### F4.1 ISOLATED: one uniform slab against the analytic Fresnel slab

`n = 2`, 300 nm, vacuum half-spaces, `phi = 0` (so the incident `E_y` row IS
s-polarized).  `|R - R_exact|`.  The exact answer does not depend on `N` at
all, so every entry is the basis's own error.

| `theta` | `N` | `M=3` | `M=4` | `M=5` | `M=6` | `M=7` | `M=8` | `M=9` |
|---|---|---|---|---|---|---|---|---|
| 0.00 | 1 | 1.1e-16 | 1.1e-16 | 3.2e-15 | 0.0e+00 | 3.9e-16 | 3.1e-14 | 6.1e-14 |
| 0.00 | 2 | 6.7e-16 | 5.0e-16 | 2.1e-15 | 1.8e-14 | 1.1e-15 | 4.5e-14 | 1.5e-14 |
| 0.00 | 3 | 1.2e-15 | 1.6e-15 | 1.4e-14 | 1.8e-14 | 2.7e-14 | 5.9e-14 | 2.9e-15 |
| 0.20 | 1 | 1.0e-04 | 1.9e-06 | 6.2e-08 | 2.2e-10 | 1.2e-12 | 6.9e-14 | 1.0e-13 |
| 0.20 | 2 | 6.8e-06 | 3.4e-08 | 1.2e-10 | 2.5e-13 | 4.7e-14 | 9.1e-14 | 1.1e-14 |
| 0.20 | 3 | 1.4e-06 | 3.3e-09 | 4.6e-12 | 1.8e-15 | 2.8e-14 | 2.3e-14 | 7.1e-14 |
| 0.40 | 1 | 5.2e-03 | 4.1e-04 | 1.6e-05 | 4.1e-07 | 2.6e-09 | 2.4e-10 | 3.1e-12 |
| 0.40 | 2 | 2.7e-04 | 9.4e-06 | 9.9e-08 | 7.6e-10 | 4.1e-12 | 3.1e-14 | 4.8e-14 |
| 0.40 | 3 | 7.9e-05 | 6.5e-07 | 4.2e-09 | 1.4e-11 | 4.5e-14 | 1.3e-13 | 3.1e-13 |
| 0.60 | 1 | 5.5e-02 | 6.9e-02 | 6.4e-04 | 4.5e-05 | 1.1e-06 | 8.0e-08 | 2.1e-09 |
| 0.60 | 2 | 4.5e-03 | 1.8e-04 | 4.5e-06 | 7.4e-08 | 8.5e-10 | 6.9e-12 | 8.3e-14 |
| 0.60 | 3 | 6.8e-04 | 1.7e-05 | 2.0e-07 | 1.4e-09 | 7.0e-12 | 1.1e-13 | 7.2e-15 |

CONICAL (`theta = 0.35`, `phi = 0.6`) against `berreman_jones_1d`, relative
`dJones` -- the scalar Fresnel formula does NOT apply at `phi != 0` (see the
trap note in the README):

| `N` | `M=3` | `M=4` | `M=5` | `M=6` | `M=7` | `M=8` | `M=9` |
|---|---|---|---|---|---|---|---|
| 1 | 2.9e-03 | 1.3e-04 | 5.1e-06 | 5.5e-08 | 6.2e-10 | 5.1e-12 | 9.3e-14 |
| 2 | 2.1e-04 | 2.4e-06 | 1.5e-08 | 6.0e-11 | 1.8e-13 | 1.0e-13 | 2.6e-13 |
| 3 | 4.2e-05 | 2.1e-07 | 5.9e-10 | 1.0e-12 | 2.4e-14 | 1.3e-13 | 8.6e-14 |

**At equal `M` a coarse grid looks worse; at equal DOF it is the other way
round, by decades.**  Re-read by `q = N (M-1)`:

| `theta = 0.60`, matched `q` | `N = 1` | `N = 2` | `N = 3` | `N=1` advantage |
|---|---|---|---|---|
| `q = 4` | **6.4e-04** | 4.5e-03 | -- | 7x |
| `q = 6` | **1.1e-06** | 1.8e-04 | 6.8e-04 | 164x / 618x |
| `q = 8` | **2.1e-09** | 4.5e-06 | -- | 2140x |

| CONICAL, matched `q` | `N = 1` | `N = 2` | `N = 3` | `N=1` advantage |
|---|---|---|---|---|
| `q = 4` | **5.1e-06** | 2.1e-04 | -- | 41x |
| `q = 6` | **6.2e-10** | 2.4e-06 | 4.2e-05 | 3900x / 68000x |
| `q = 8` | **9.3e-14** | 1.5e-08 | -- | 160000x |

**Reason, and it is the opposite of the worry:** the field in a uniform region
is ONE analytic plane wave, so the error is spectral in the polynomial degree
and the cheapest way to buy degree is to spend the whole `q` budget on ONE
element.  `N = 1, M = 9` and `N = 3, M = 3` cost `q = 8` and `q = 6`
respectively; the first is 2.1e-09 and the second 6.8e-04.  **Isolated, a
uniform layer should be on `N = 1`.**

### F4.2 IN A CASCADE, against the exact 1-D oracle

The isolated result is not the whole answer, because inside a cascade the
uniform layer's trace space must also carry the NEIGHBOURS' modal content --
which has kinks at THEIR walls, and is not a single plane wave.  Fixture: a
y-uniform stripe stack `A(duty 1/2, N=2, M_A=8) | uniform eps=2.25 (grid, M_u)
| B(duty 1/3, N=3, M_B=7)`, so the exact 1-D `PMMStack` at degree 14 is the
truth for the whole stack (self-gaps 2.32e-07 / 6.42e-07 / 2.00e-07).  Only the
uniform layer varies.

| `theta` | grid | `M_u=3` | 4 | 5 | 6 | 7 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|---|---|
| 0.00 | 1 | 5.7e-01 | 3.6e-01 | 1.4e-01 | 2.5e-01 | 5.0e-02 | 3.1e-02 | 1.3e-02 | **4.4e-03** |
| 0.00 | 2 | 9.1e-02 | 5.1e-02 | 5.7e-02 | 9.0e-03 | 1.0e-02 | 4.2e-03 | 3.5e-03 | **3.5e-03** |
| 0.00 | 3 | 2.0e-02 | 4.3e-02 | 6.3e-03 | 4.8e-03 | 3.4e-03 | 3.5e-03 | 3.5e-03 | **3.5e-03** |
| 0.20 | 1 | 3.2e-01 | 2.9e-01 | 1.5e-01 | 1.6e-01 | 6.2e-02 | 1.4e-01 | 4.8e-02 | **2.0e-02** |
| 0.20 | 2 | 1.7e-01 | 9.9e-02 | 7.8e-02 | 5.1e-02 | 3.5e-02 | 1.7e-02 | 1.2e-02 | **1.2e-02** |
| 0.20 | 3 | 7.4e-02 | 7.5e-02 | 3.1e-02 | 1.6e-02 | 1.2e-02 | 1.2e-02 | 1.2e-02 | **1.2e-02** |
| 0.40 | 1 | 4.3e-01 | 3.9e-01 | 2.8e-01 | 2.1e-01 | 2.1e-01 | 1.5e-01 | 7.1e-02 | **3.1e-02** |
| 0.40 | 2 | 2.6e-01 | 1.6e-01 | 3.7e-01 | 1.0e-01 | 8.5e-02 | 3.0e-02 | 2.5e-02 | **2.5e-02** |
| 0.40 | 3 | 1.1e-01 | 2.7e-01 | 4.8e-02 | 3.0e-02 | 2.5e-02 | 2.5e-02 | 2.5e-02 | **2.5e-02** |

Three readings:

1. **Every grid reaches the SAME floor** -- 3.5e-03 / 1.2e-02 / 2.5e-02 at
   `theta = 0 / 0.20 / 0.40 -- and that floor is NOT the uniform layer: it is
   layers A and B at their fixed `M_A = 8`, `M_B = 7`.  Once `M_u` is adequate
   the uniform layer is never the limiter, on any grid.
2. **`N = 1` needs a much higher `M_u` to get there.**  `grid = 3` is at the
   floor by `M_u = 7`, `grid = 2` by `M_u = 8-10`, `grid = 1` still 1.3-1.7x
   above it at `M_u = 12`.  **At the stack's default `M` (7-8 here), `grid = 1`
   is 3-6x worse than `grid = 2` or `3`** -- and that is precisely the
   configuration the natural API default would produce.
3. **Per DOF `N = 1` is still competitive and often cheapest**: at
   `theta = 0.40`, `grid = 1` reaches 3.1e-02 on `q = 11`, where `grid = 2`
   needs `q = 14` for 3.0e-02 and `grid = 3` needs `q = 15`.

### F4.3 Against the SHARED-GRID path

The same stack on the union lattice `N = 6` with one global `M`, same oracle:

| `theta` | union `M=4` (`q=18`) | union `M=5` (`q=24`) |
|---|---|---|
| 0.00 | 7.4e-03 (11.2 s) | 4.5e-04 (68.5 s) |
| 0.20 | 2.8e-02 (11.5 s) | 3.5e-03 (70.0 s) |
| 0.40 | 3.9e-02 (11.5 s) | 2.8e-03 (69.6 s) |

The union arm at `q = 24` beats the per-layer arm's floor -- **because it also
raises A and B**, which the per-layer arm held fixed at `q_A = 14`, `q_B = 18`
by construction.  The eig work is `sum dim^3`: the per-layer arm at
`(grid=1, M_u=12)` is `392^3 + 242^3 + 648^3 = 3.5e+08`; the union arm at
`M = 5` is `3 x 1152^3 = 4.6e+09`, **13x more**.  This part of the table is not
a per-layer-vs-union comparison (S7.3 and F2 are); it is the calibration that
says the floors above are the neighbours' and nothing else.

### F4.4 VERDICT on O-6

**A uniform layer may go on `N = 1`, and it should -- but its `M_u` must NOT
default to the stack's `M`.**

* Isolated, `N = 1` is 1-5 DECADES better than `N = 2` or `N = 3` at equal DOF
  at every angle measured, including conical (F4.1).  The field is one plane
  wave; degree is the right currency and one element buys the most of it.
* In a cascade the uniform layer must additionally represent the NEIGHBOURS'
  traces, and there `N = 1` pays: it needs `M_u ~ 12` where `N = 3` needs
  `M_u ~ 7` to reach the same floor.  At a shared default `M` it reads 3-6x
  worse (F4.2).
* **The API rule this yields, MEASURED** (`f4b_mu_rule.py`): `n_modes` for a
  uniform layer must default not to the stack's `M` but to the value that
  matches its NEIGHBOURS' `q` -- `M_u = max(q_prev, q_next) / grid + 1`.  On
  the F4.2 fixture (`q_A = 14`, `q_B = 18`) that is `M_u = 19` on `grid = 1`,
  `M_u = 10` on `grid = 2`, `M_u = 7` on `grid = 3`, all three at `q_u = 18`
  and an eig dimension of `2 * 18^2 = 648`:

  | `theta` | `grid=1, M_u=7` (the DEFAULT, `q_u=6`) | `grid=1, M_u=19` | `grid=2, M_u=10` | `grid=3, M_u=7` |
  |---|---|---|---|---|
  | 0.00 | 5.00e-02 | **3.46e-03** | 3.54e-03 | 3.45e-03 |
  | 0.20 | 6.23e-02 | **1.21e-02** | 1.24e-02 | 1.18e-02 |
  | 0.40 | 2.13e-01 | **2.50e-02** | 2.53e-02 | 2.47e-02 |

  **At matched `q_u` the grid choice does not matter at all** -- the three
  columns agree to 3 %, and all three sit on the neighbours' floor.  The
  stack-default `M_u` is 14x / 5x / 8.5x worse.  So `grid = 1` is the right
  default because it is the cheapest basis to build, not because it is more
  accurate; the accuracy lever is `q_u`.  **`N = 1` is cheap, `N = 1` at the
  stack's `M` is a trap**, and the docstring must say so.
* S2.5's far-field caveat is unchanged and now doubly load-bearing: an `N = 1`
  END layer caps the Rayleigh orders at `(M-2)//2`.  With the `M_u` rule above
  that cap is generous; with a defaulted `M_u` it is not.

## F5 (roadmap item N-1) -- NON-UNIFORM SEGMENT BOUNDARIES: **GO**

`nonuniform.py` (the generalized basis), `f5_nonuniform.py` (gates a-d, c3, d2),
`f5d_diag.py`, `f5e_nearwall.py`, `f5f_attrib.py`.

This is the item S0 named as the one thing this experiment does NOT deliver:
*"the enabling change for arbitrary tapers is still non-uniform segment
boundaries in `Basis1D` (campaign item N-1)"*.  It turns out to be a
**four-line-family change**, and the paper already contains it.

### F5.0 The paper has non-uniform segments; the implementation chose uniform [A]

Granet 2023 Eq. 31, verbatim from page 655:

> *"consider a line `I` of length `d` divided by `N` adjacent segments
> `I_n = [x_n, x_{n+1}]`, `n = 1, 2, ... N`, and `I = U I_n`.  Each segment is
> mapped to the reference interval `[-1, 1]` by the change of variable*
> `x = 0.5 (x_{n+1} - x_n) u + 0.5 (x_{n+1} + x_n)`."

Nothing there requires `x_{n+1} - x_n` to be constant.  `Basis1D.__init__`
(`twod_staggered.py:324-327`) chooses it to be:

```python
self.h  = self.d / self.N            # ONE segment length
self.J  = 0.5 * self.h               # ONE scalar jacobian
self.xb = np.linspace(0.0, self.d, self.N + 1)
```

and that single choice is what forces S1.1's conclusion (*"the only lattice
containing two layers' walls is the LCM lattice"*) and S1.3's (*"a 1.8 nm wall
offset on a 700 nm period needs `N ~ 390`"*).  **Remove the choice and both
conclusions go away**: a taper slice with two walls is THREE segments wherever
those walls are.

**What has to change is exactly one thing, applied in four places: the scalar
`J` becomes a per-segment `J_n`.**

| # | site | change |
|---|---|---|
| 1 | `Basis1D._global_matrix` | mass `* J` -> `* J_n`; stiffness `/ J` -> `/ J_n`; **mixed unchanged** |
| 2 | `_global_pair_segmat` (module level) | the same, per segment |
| 3 | `Granet2DTransverseE._eps_dir`'s inline `segmat` | the same (`op == 'm'` only) |
| 4 | `_stag_fourier_projection` | `xphys = mid_n + J_n * u` and the `J_n / d` weight |

**And what does NOT change, which is why this is cheap:**

* `Basis1D.mixed` was **already correct**: one derivative contributes `1/J_n`
  and the measure contributes `J_n`, so the scale is `1` on every segment
  regardless of the partition.  It needs no edit at all.
* `_build_elementary` -- `m_ref`, `s_ref`, `c_ref` live on the reference
  interval and never saw `J`.
* `_build_sets` -- the hats (Eq. 32) glue `Ltilde_2` of one segment to
  `Ltilde_1` of the next **by value** (`Ltilde_1(-1) = Ltilde_2(+1) = 1`,
  `Ltilde_1(+1) = Ltilde_2(-1) = 0`), which is a statement about the reference
  interval alone; the Bloch periodic hat (Eq. 33, the `tau` glue) is likewise
  unchanged.  Segment lengths never enter.
* **The de Rham property `d(Btilde) subset span(B)`** -- the thing that makes
  this basis spurious-free -- is per-segment and scale-free: on segment `n` a
  `Btilde` member is a polynomial of degree `<= M-1`, its derivative has degree
  `<= M-2`, and `B`'s local span IS every polynomial of degree `<= M-2`
  (two half-hats spanning degree `<= 1` plus bubbles `2..M-2`).  Multiplying by
  `1/J_n` does not leave that span. **[A]**
* **The mortar cross-mass `cross_mass_1d` is already general**: it integrates on
  the UNION of the two partitions and maps each union sub-interval into each
  side's own segment with that segment's own affine map.  Not one character
  changes.

### F5.1 GATE (a) -- uniform boundaries reproduce the shipped basis BIT-IDENTICALLY [M]

`f5_nonuniform.py a`.  `Basis1DNU(d, walls, M, tau)` accepts `walls` as an
`int N` (the uniform lattice) or an `(N+1,)` boundary array.

| arm | what is compared | result |
|---|---|---|
| 1-D, `walls = N` (int), `N, M` = (2,5), (3,6), (4,4), (6,4) | `mass<til\|til>`, `mass<B\|B>`, eps-weighted mass, `stiff`, `mixed<B\|d til>`, `_global_pair_segmat(m_ref)`, `_stag_fourier_projection` at `alpha0 = 0.31` -- **7 matrix families x 4 grids** | **BIT-IDENTICAL, worst `\|d\| = 0.0e+00`** |
| 2-D scalar `3x3` cell | `Rmat`, `Lmat`, `Stt`, `Schur` | **BIT-IDENTICAL** |
| 2-D IN-PLANE TENSOR `2x2` cell (gyrotropic pillar in a biaxial host) | `Rmat`, `Lmat`, `Stt`, `Schur` | **BIT-IDENTICAL** |
| 2-D OUT-OF-PLANE `2x2` cell (tilted LC director) | `Agen`, `Bgen` (the `4 q^2` first-order generator) | **BIT-IDENTICAL** |

The ULP question the gate had to answer: `walls` given EXPLICITLY as
`np.linspace(0, d, N+1)` is bit-identical too at `d = 1.2 um` for
`N = 2, 3, 4, 6` -- but at `d = 0.9 um, N = 4` it differs by **1.96e-16
relative**.  The reason is arithmetic, not physics: `np.linspace` computes
`start + i*step` and pins the last element, so `linspace[i+1] - linspace[i]` is
not always the same double as `d/N`.  **Therefore the integer path must stay a
distinct path in the implementation** (`walls = N` -> `h = d/N`,
`J_n = 0.5*h` for all `n`), which is what makes the bit-identity claim
unconditional and what a `G1`-class gate would assert.

**GATE (a): PASS**, unconditionally on the integer path, at ULP level
(1.96e-16, explained) on the explicit-boundary path.

### F5.2 GATE (b) -- 2 non-uniform segments == 3 uniform segments [M]

Two exact-wall representations of ONE device (a pillar occupying `[0, P/3]` per
axis): a NON-uniform 2-segment grid with its wall at `P/3`, and the uniform
3-segment grid.  They must converge to the same answer.

**(b1) 2-D pillar**, `theta = 0.18`, single layer, gap relative to `max(R, T)`:

| `M` | NU 2 seg, `q = 2(M-1)` | uniform 3 seg, `q = 3(M-1)` | relative gap | closure NU / uniform |
|---|---|---|---|---|
| 4 | 6 | 9 | 9.34e-02 | 1.1e-02 / 4.4e-04 |
| 5 | 8 | 12 | 2.18e-02 | 1.4e-03 / 3.6e-05 |
| 6 | 10 | 15 | 4.67e-02 | 2.5e-04 / 1.3e-06 |
| 7 | 12 | 18 | **3.82e-03** | 3.6e-05 / 2.3e-08 |

**(b2) the STRIPE twin, both arms against the EXACT 1-D `PMMStack`** (degree 14;
its own degree-12-vs-14 self-gap **1.39e-06**, three decades below every entry
below).  The derived bar is the triangle inequality: two representations of one
device may disagree by at most the SUM of their own distances to truth.

| `M` | NU (`q`) err vs EXACT | uniform (`q`) err vs EXACT | their gap | **derived bar** = sum | inside? |
|---|---|---|---|---|---|
| 4 | 1.02e-01 (6) | 2.50e-01 (9) | 1.48e-01 | 3.52e-01 | yes |
| 5 | 1.27e-02 (8) | 1.17e-02 (12) | 9.59e-04 | 2.44e-02 | yes |
| 6 | 1.44e-02 (10) | 1.36e-02 (15) | 7.53e-04 | 2.80e-02 | yes |
| 7 | 1.55e-04 (12) | 5.90e-05 (18) | 9.56e-05 | 2.14e-04 | yes |
| 9 | 1.65e-05 (16) | 1.98e-06 (24) | 1.46e-05 | 1.85e-05 | yes |

**GATE (b): PASS at all five rungs against a derived, not fitted, bar** -- and
note the NU arm reaches 1.65e-05 on `q = 16` where the uniform arm needs
`q = 24` for 1.98e-06, i.e. the two are on the same accuracy-per-DOF curve.

### F5.3 GATE (c) -- ARBITRARY walls on their own `3x3` non-uniform grid [M]

Three sub-gates, because the two hybrid oracles have a floor on this cell class
and the third sub-gate does not.

**(c1) walls at `19/80` and `49/80` of the period** (exactly representable on an
80-pixel cell, so BOTH hybrids are exact-wall too; utterly unreachable for the
uniform lattice, which would need `N = 80`, `q >= 240`).  `theta = 0.18`,
`phi = 0.35`.  The two oracles' OWN ladders first:

| oracle | setting | `R(0,0)` | own self-gap |
|---|---|---|---|
| hybrid PMM | degree 11, `n_orders` 7 | 0.107991438 | -- |
| hybrid PMM | degree 11, `n_orders` 11 | 0.105229909 | 7.87e-03 |
| hybrid PMM | degree 13, `n_orders` 11 | 0.106776687 | 1.55e-03 |
| RCWA | `n_orders` 11 | 0.122451909 | -- |
| RCWA | `n_orders` 15 | 0.119468096 | 5.03e-03 |

**The two oracles disagree with each other by 1.27e-02 on `R(0,0)`** -- the
corner-dominated crossed pillar is exactly the regime `twod_staggered`'s own
docstring flags (*"neither Fourier arm is converged there"*).  Against that:

| pure NU `3x3` | `q` | vs hybrid PMM | vs RCWA | own closure | wall |
|---|---|---|---|---|---|
| `M = 4` | 9 | 2.04e-01 | 2.16e-01 | 1.1e-04 | 0.2 s |
| `M = 5` | 12 | 8.30e-02 | 7.48e-02 | 2.0e-05 | 0.9 s |
| `M = 6` | 15 | 7.12e-03 | 1.62e-02 | 1.9e-06 | 3.2 s |
| `M = 7` | 18 | **4.29e-03** | 1.07e-02 | **4.2e-08** | 10.7 s |

By `M = 7` the pure non-uniform arm sits **inside the two oracles' own mutual
spread** (4.3e-03 and 1.07e-02 against their 1.27e-02 disagreement) with an
energy closure of 4.2e-08 -- four decades tighter than either oracle's
convergence gap.  That is agreement AT the oracles' floor, which is all this
sub-gate can assert.

**(c2) walls at 0.2371 and 0.6183** -- genuinely arbitrary, driven through the
hybrid's `_pmm2d_solve_core` at those EXACT walls (its own degree-11-to-13
self-gap 1.50e-03): pure NU reads 1.79e-01 / 7.81e-02 / 9.29e-03 / 9.68e-03 at
`M = 4..7`, closure 2.2e-04 -> **4.0e-08**.  Same reading, same floor.

**(c3) THE DECISIVE ONE -- the same arbitrary walls made y-uniform, so the
EXACT 1-D `PMMStack` applies** (it takes arbitrary segment widths natively;
degree-12-vs-14 self-gap **4.87e-08**):

| pure NU `3x3` @ walls (0.2371, 0.6183) | `q` | **err vs the EXACT answer** | closure | wall |
|---|---|---|---|---|
| `M = 4` | 9 | 4.48e-02 | 3.2e-03 | 0.2 s |
| `M = 5` | 12 | 2.36e-02 | 3.6e-05 | 0.8 s |
| `M = 6` | 15 | 3.14e-03 | 1.4e-05 | 3.1 s |
| `M = 7` | 18 | 2.67e-03 | 5.4e-08 | 10.5 s |
| `M = 9` | 24 | **4.05e-05** | **2.0e-10** | 56.4 s |

**Three decades of convergence against an exact independent oracle, at walls
that the shipped uniform lattice cannot represent at any affordable `q`, and
seven decades of lossless closure.**  The oracle's own self-gap (4.87e-08) sits
three decades below the last reading, so the measurement is readable
throughout.

**GATE (c): PASS.**

### F5.4 GATE (d) -- a 4-SLICE TAPER as a per-layer non-uniform MORTAR cascade [M]

Four slices, walls interpolated between `x in [0.1873 P, 0.7241 P]` at the
bottom and `[0.2917 P, 0.6109 P]` at the top with the hybrid's own midpoint
rule, so the four wall pairs are
`0.2787-0.6250`, `0.2525-0.6533`, `0.2264-0.6816`, `0.2004-0.7099` -- **no two
slices share a single wall**, and each slice sits on its own 3-segment
non-uniform grid.  Five mortar interfaces, every one of them non-conforming.

**(d1) 2-D pillar taper vs the hybrid's `add_tapered_pillar` staircase** with
the identical slices:

| arm | setting | err vs hybrid | closure | wall |
|---|---|---|---|---|
| hybrid staircase | degree 9, `n_orders` 9 | -- | 4.1e-03 | 10.9 s |
| hybrid staircase | degree 11, `n_orders` 9 | (self-gap 8.69e-03) | 2.4e-03 | 11.3 s |
| NU mortar cascade | `M = 4` (`q = 9`) | 2.78e-01 | 3.4e-03 | 0.5 s |
| NU mortar cascade | `M = 5` (`q = 12`) | 1.43e-01 | 3.5e-04 | 2.7 s |
| NU mortar cascade | `M = 6` (`q = 15`) | 4.16e-02 | 1.3e-05 | 10.4 s |
| NU mortar cascade | `M = 7` (`q = 18`) | **3.18e-02** | **2.1e-06** | 30.1 s |

The hybrid's own degree self-gap is 8.69e-03 and its lossless closure is
2.4e-03 -- **the pure arm's closure is 1100x tighter than the oracle it is
being scored against**, so 3.18e-02 is the oracle's floor showing, not the pure
arm's error.

**(d2) THE DECISIVE ONE -- the same taper made y-uniform, exact 1-D oracle**
(self-gap 2.89e-08):

| NU mortar cascade, 4 slices, 5 non-conforming interfaces | `q` | **err vs the EXACT answer** | closure | wall |
|---|---|---|---|---|
| `M = 4` | 9 | 3.13e-01 | 2.0e-02 | 0.5 s |
| `M = 5` | 12 | 1.08e-01 | 4.2e-04 | 2.5 s |
| `M = 6` | 15 | 1.19e-01 | 1.0e-05 | 8.8 s |
| `M = 7` | 18 | 1.24e-01 | 7.8e-06 | 34.3 s |
| `M = 9` | 24 | 1.15e-02 | 2.9e-08 | 188.4 s |
| **`M = 11`** | **30** | **1.20e-04** | **4.9e-11** | 716.8 s |

The `M = 5..7` PLATEAU at ~1.2e-01 is real and it is alarming until it is
attributed, so it was: **it is the device, not the mortar** (F5.5).

### F5.5 The plateau, attributed -- three controls and one oracle defect [M]

`f5d_diag.py`, `f5f_attrib.py`.

| control | what it isolates | reading |
|---|---|---|
| **[A]** 4 IDENTICAL slices (a straight pillar at the same arbitrary walls) -- all four grids equal, so the identical-grid BYPASS fires and **no mortar exists in the run** | the multi-layer machinery, mortar-free | 3.08e-01, 2.73e-02, **2.51e-02**, 1.45e-04, 2.00e-06, 6.48e-07 at `M = 4,5,6,7,9,11` -- **the SAME non-monotone plateau shape** |
| **[B]** the same 4 identical slices with `force_mortar=True` -- 5 FORCED mortar interfaces between identical NON-UNIFORM grids | the M1 conforming identity, on non-uniform grids, through five interfaces | bypass vs forced differ by **2.08e-16** (`M=5`) and **3.61e-16** (`M=7`); closures identical to 2 digits |
| **[C]** the taper truncated to 1 / 2 / 3 / 4 slices | how the error scales with the number of non-conforming interfaces | at `M = 9`: 1.13e-06, 1.90e-05, 2.74e-04, 1.15e-02 -- it grows with the STACK, and the 1-slice arm is at 1e-06 |
| **[E]** per-order breakdown at `M = 7` | whether one order carries it | spread across `m = -1, 0, +1` (`dT` = -1.24e-01, -1.80e-02, +5.44e-02), i.e. under-resolution, not an order-slot artefact |

**[B] is the load-bearing one: the S3 conforming identity holds on NON-UNIFORM
grids to 2e-16 through five forced mortar interfaces.**  Combined with [A] --
which reproduces the plateau with no mortar in the process at all -- the
plateau is the four-slice device's corner-capped convergence, and `M = 11`
walks past it to 1.20e-04.

**And one finding that is NOT about this item.**  A wall-separation sweep (two
slices whose wall sets differ by `delta` as a fraction of the period) appeared
to show the non-uniform mortar EXPLODING at `delta = 1e-4 .. 1e-5`, with the
lossless closure staying at 1.6e-08 -- the energy-invisible shape.  It is the
**1-D `PMMStack` ORACLE**, not the mortar (`f5f_attrib.py`, which measures the
oracle's own degree-12-vs-14 self-gap alongside every comparison):

| `delta` (fraction of period) | 1-D oracle's OWN self-gap | oracle(delta) - oracle(0) | pure NU `M=7` - oracle(0) |
|---|---|---|---|
| 1.00e-02 | 2.84e-08 | 9.26e-03 | 9.32e-03 |
| 2.60e-03 | 3.53e-08 | 2.83e-03 | 2.86e-03 |
| 1.00e-03 | 4.34e-08 | 1.13e-03 | 1.18e-03 |
| 3.00e-04 | 6.43e-08 | 3.44e-04 | 4.99e-04 |
| **1.00e-04** | **4.79e-01** | 4.79e-01 | 3.03e-04 |
| **3.00e-05** | **8.17e+00** | 8.62e+00 | 2.34e-04 |
| **1.00e-05** | 1.18e-04 | **9.30e-01** | 2.15e-04 |
| 3.00e-06 | 5.47e-08 | 3.46e-06 | 2.08e-04 |
| 1.00e-06 | 5.47e-08 | 1.15e-06 | 2.06e-04 |
| 0 | 5.47e-08 | 0 | 2.05e-04 |

The last column is the whole answer: **the pure non-uniform mortar arm is
smooth and MONOTONE in `delta` all the way to zero** (9.32e-03 -> 2.05e-04,
no discontinuity anywhere), while the oracle's own self-gap blows to 4.8e-01
and 8.2e+00 at `delta = 1e-4` and `3e-5`, and at `delta = 1e-5` **both of its
degrees agree on an answer 9.3e-01 away from the `delta -> 0` limit** -- a
converged-looking wrong answer.  Both spellings of the 1-D oracle
(`layer_grids` default and `'per-layer'`) read identically.  This is a
suspected sliver-element defect in the shipped 1-D `PMMStack` at near-coincident
LAYER walls; it is logged as **O-11** below and it is not this item's.

The taper regime the roadmap actually cares about is far from it: roadmap
correction S-1's 2-degree sidewall moves a wall 1.8 nm on a 700 nm period,
`delta = 2.57e-03` of the period, where the sweep reads 2.46e-04 at `M = 7`
and everything is monotone.

**GATE (d): PASS** -- 1.20e-04 against an exact oracle on a four-slice
arbitrary-wall taper, with the plateau attributed to the device by two
controls and the conforming identity re-established on non-uniform grids at
2e-16.

### F5.6 What the arithmetic becomes

Roadmap correction S-1, restated with non-uniform segments:

| | uniform lattice (today) | non-uniform segments |
|---|---|---|
| a 2-deg sidewall over 310 nm, 6 slices, 700 nm period (wall moves 1.8 nm/slice) | `N ~ 390` per slice -> `q = 390 (M-1) >= 1170`, eig `2 q^2 >= 2.7e+06` | **3 segments** per slice -> `q = 3 (M-1)`; at `M = 9`, `q = 24`, eig **1152** |
| the measured 4-slice arbitrary-wall taper | unreachable | **1.20e-04** vs the exact oracle at `M = 11` (717 s), 1.15e-02 at `M = 9` (188 s) |

**S0's scope statement is superseded**: with non-uniform segments this item
delivers the ARBITRARY taper, not only the low-order-rational staircase.

### F5.7 VERDICT: **GO**, and the integration route

The change composes with the mortar build rather than competing with it -- **a
per-layer own-walls NON-UNIFORM grid is exactly what the mortar cascade was
built to carry**, and every mortar-side ingredient (the cross-mass, the
separable apply, the V1/V2 swap, `_redheffer_star_rect`) is untouched.

**The API.**  Today `eps_cell`'s grid IS the wall set, which is why arbitrary
walls are inexpressible.  Add the wall set as an input, spelled as the hybrid
already spells it (`PMM2DStackHybrid._append_patterned(kind, t, xw, yw, tile)`
and `pmm_efficiency_2d_cell`'s internal `x_walls, y_walls, tile =
_cell_to_walls_tile(...)`):

```python
Basis1D(d, walls, M, tau)          # walls: int N (uniform, BIT-IDENTICAL) or
                                   #        an (N+1,) increasing array
Granet2DTransverseE(px, py, wx, wy, M, eps_cell, ...)   # wx/wy: int or array

PMM2DStackPure.add_layer(thickness, *, eps=None, eps_cell=None,
                         x_walls=None, y_walls=None,     # NEW
                         grid=None, n_modes=None)
```

* `x_walls` / `y_walls` are the layer's own INTERIOR wall positions in metres
  (a full `0..period` boundary array is also accepted).  **`None` is today's
  behaviour, bit-identical**: the uniform lattice implied by `eps_cell.shape`.
* With walls given, `eps_cell` becomes the STRIP TILE of shape
  `(len(x_walls)+1, len(y_walls)+1)` -- literally the hybrid's `tile`, so the
  two 2-D stacks take the same geometry description and
  `add_tapered_pillar` (`stack2d.py:605-646`) transplants **verbatim**, its
  `_append_patterned("scalar", dz, xw, yw, tile)` becoming
  `add_layer(dz, eps_cell=tile, x_walls=xw, y_walls=yw)`.
* Keep `Nx == Ny`.  That constraint is `bx.dim == by.dim`, i.e. equal SEGMENT
  COUNTS per axis; the wall POSITIONS may differ freely between the axes (gate
  (c) and (d) both exercise that).
* `Basis1D.J` must be `None` (not a mean, not a NaN) on a non-uniform basis, so
  any un-migrated reader raises a `TypeError` immediately instead of applying
  one segment's scaling to all of them.  The prototype does this and it caught
  nothing after the four sites above were changed -- which is itself the
  evidence that there are only four.
* Validation: strictly increasing, `walls[0] == 0`, `walls[-1] == period`, and
  `eps_cell.shape[:2] == (len(wx)-1, len(wy)-1)`.

**Cost of the build: 1-2 days on top of the mortar build**, and the two should
ship together -- the mortar without non-uniform segments serves low-order
rationals only (S0), and non-uniform segments without the mortar cannot cascade
two slices with different walls.

**Gates for the non-uniform half** (in the S12.4 numbering):

| gate | claim | measured here |
|---|---|---|
| **N1** | `walls = N` (int) reproduces the shipped `Basis1D` and `Granet2DTransverseE` BIT-FOR-BIT, on the scalar, in-plane-tensor AND out-of-plane paths | `0.0e+00` over 7 matrix families x 4 grids and 3 cell kinds (F5.1) |
| **N2** | an explicit `linspace` wall array is NOT required to be bit-identical; the difference is ULP-level and explained | 1.96e-16 relative at `(d = 0.9 um, N = 4)`; 0.0 at `d = 1.2 um` (F5.1) |
| **N3** | two exact-wall representations of one device converge to the same answer, bounded by the TRIANGLE INEQUALITY against an exact oracle -- not by a fitted constant | 5/5 rungs inside `err_NU + err_uniform` (F5.2) |
| **N4** | a cell with arbitrary walls converges to the EXACT 1-D answer on its own 3-segment grid | 4.48e-02 -> **4.05e-05** over `q = 9..24`, oracle self-gap 4.87e-08 (F5.3 c3) |
| **N5** | a multi-slice arbitrary-wall taper converges through NON-CONFORMING mortar interfaces | **1.20e-04** at `q = 30`, closure 4.9e-11 (F5.4 d2) |
| **N6** | the conforming identity survives non-uniform grids: forcing the mortar on identical NON-UNIFORM grids reproduces the bypass | **2.08e-16 / 3.61e-16** through 5 forced interfaces (F5.5 [B]) |
| **N7** | FAIL-BEFORE for the four `J_n` sites: reverting any one of them to the scalar `J` must break a NON-uniform solve while leaving every uniform solve bit-identical | not run; the four sites are identified and each is 1-3 lines. The build must run it -- a uniform-only gate cannot see any of them, exactly as G3 cannot see the V1/V2 swap |

## F6 -- open items after this round

| id | status |
|---|---|
| **O-1** second BLAS build | **CLOSED** by F1, with a caveat that must travel with it: the three arms vary OS, compiler, python, numpy and the BLAS KERNEL, but not the BLAS FAMILY (all three are scipy-openblas 0.3.31).  The measured spreads are a LOWER bound; the doc's original "numpy + MKL" build note was wrong and is corrected in the FOLLOW-UP header. |
| **O-2** the equal-DOF advantage's regime | **CLOSED** by F2.  The corner-dominated 2-D pillar pair does not reverse the sign at working `q`; it moves the crossing from `q ~ 30` (stripes) to between `q = 18` and `q = 24`.  Same sign with an in-plane LC tensor.  The closure advantage is larger on the device class than on stripes (up to 5142x) and does not decay. |
| **O-6** `N = 1` uniform layers at oblique | **CLOSED** by F4.  `N = 1` is right, by 1-5 decades per DOF isolated; the trap is its `n_modes`, which must scale to the NEIGHBOURS' `q`, not to the stack's `M`. |
| **O-10** the per-layer `M` recipe | **CLOSED** by F3.  Ship stationarity-in-every-knob as the criterion and the per-layer own-residual FLOOR (15/16) as the cheap screen; do NOT ship the greedy rule (6/9). |
| **N-1** non-uniform segments | **GO** (F5).  Four sites, `J -> J_n`; bit-identical on uniform walls including the out-of-plane generator; 4.05e-05 vs the exact oracle at arbitrary walls; 1.20e-04 on a four-slice arbitrary-wall taper.  Ship it WITH the mortar build (1-2 days on top). |
| **O-3** `retain_internal` / `layer_absorption` per-layer | **STILL OPEN** -- designed (S12.3), not prototyped, not measured.  Unchanged by this round. |
| **O-4** JAX twin | **STILL OPEN**, still not a regression. |
| **O-5** `prepare()` / wavelength sweeps | **STILL OPEN**; non-uniform segments do not change the argument (the cached objects are still two small 1-D matrices per interface, now keyed on the wall array instead of on `N`). |
| **O-7** the `(2, 2)` grid fallback in `PMM2DStackPure.solve` | **STILL OPEN**; with F5 it must become a per-layer WALL list, not a per-layer `N` list. |
| **O-8** grids keyed on `tau` | **STILL OPEN** and slightly worse with F5: the cache key becomes the wall ARRAY plus `tau`, so an angle sweep over a taper rebuilds every slice's basis.  The tau-free/tau-dependent split S O-8 proposes is now the obvious fix. |
| **O-9** clamp vs raise on `n_orders` | **STILL OPEN**; F4.4 makes it sharper (an `N = 1` end layer caps at `(M-2)//2`). |
| **O-11** NEW -- **a suspected sliver-element defect in the shipped 1-D `PMMStack` at near-coincident LAYER walls** | Two layers whose wall sets differ by `delta = 1e-4 .. 1e-5` of the period: the solver's own degree-12-vs-14 self-gap reads 4.79e-01 and 8.17e+00, and at `delta = 1e-5` BOTH degrees agree on an answer 9.30e-01 away from the `delta -> 0` limit -- a converged-looking wrong answer, with the lossless closure sitting at 1.6e-08 throughout (**energy-invisible**).  Both `layer_grids` spellings read the same.  Reproducer: `f5f_attrib.py`.  Outside `1e-3 .. 3e-6` the solver is smooth and correct, and a 2-degree sidewall taper (`delta = 2.6e-03`) is nowhere near it -- but a `stabilize`-style degree consensus would NOT catch this, which is what makes it worth a ticket. |

**Nothing in this round reopens the GO of S0.**  It narrows two claims (the
equal-DOF ratio is regime-bounded, F2.4; the per-layer `M` needs a documented
recipe, F3.4), removes the scope limitation that S0 stated as permanent (F5.6:
arbitrary tapers are reachable), and replaces one assumption with a measurement
(F1: which bars are build-free and which are not).
