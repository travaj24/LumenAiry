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

Against the union curve of §7.1: at ~16 s the per-layer arm reads 2.15e-03 and
the union 7.01e-03 (3.3x better); at ~29 s the per-layer arm reads 8.99e-04
where the union needs 141 s to reach 5.56e-04. **The per-layer route's
error-vs-wall-time curve sits below the union's throughout the measured band**,
and the lever that puts it there -- a different `M` per layer -- does not exist
on the union grid.

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

`PMMStack` spells it `layer_grids="shared" | "per-layer"` (a hyphen, not an
underscore; `stack.py:172`) with `window_halfwidth=1`. Mirror the first,
**refuse the second**:

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
| **O-9** | The probe's `MortarStack2D` clamps `n_orders` silently and reports `n_orders_used`. The shipped version should RAISE above capacity (the classical/JAX siblings' behaviour) rather than clamp, so a user asking for orders the end grids cannot carry is told, not quietly served fewer. |
