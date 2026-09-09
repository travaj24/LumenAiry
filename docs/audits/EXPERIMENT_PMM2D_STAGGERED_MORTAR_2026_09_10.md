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
run** -- see §12, open item O-1.

---

## 0. VERDICT: **GO**, with a scope that is narrower and a mechanism that is
## different from what the roadmap assumed

The mortar is **correct** (§3, §4), **convergent on genuinely non-conforming
grids** (§7), **serves the generalized / out-of-plane cascade** (§8), and
**carries no conditioning cliff** in the useful range (§10). At **equal
degrees of freedom** it is not merely as good as the union grid, it is
**7.6x more accurate** on the stripe pair with an exact independent oracle
(§7.3), and on a 3-slice staircase it reaches the oracle at wall times the
union grid cannot (§9).

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
   and p-refinement on the own-walls grid buys more, per degree of freedom
   (§7.3). The eig dimension is `2 (N (M-1))^2`, a function of the PRODUCT
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
  (§10.2): 46 MB at `N=(6,12), M=6` against 0.03 MB for its two factors, and
  the separable apply is exact to 3e-16 and ~30x faster. The prototype's
  `kron_apply` is the production form.

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
mortar multiplies by `G` on both sides of a `solve` where the plain interface
does not, so the identity is exact algebraically and rounds differently
(`cond_2(G1)` at `N=3, M=6` is ~3e+02, and `2.3e-13 ~ eps * cond` is where that
lands). A three-region pillar/pillar case scored the same way in §6 reads
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

Closure is scored TWO-SIDED (deficit and excess are equally defects on a
lossless stack) and stays inside [-4.8e-03, +3.0e-03] at `M=4` on every arm
against the union's -3.5e-06 -- i.e. the closure defect tracks the coarse
layers' resolution and decays with it, to 6.9e-12 by `M = 11` in §7.

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

| `q` | eig dim | arm | settings | error vs 1-D | closure | wall |
|---|---|---|---|---|---|---|
| 18 | 648 | union | `N=6, M=4` | 7.01e-03 | 6.3e-05 | 23.6 s |
| 18 | 648 | **mortar** | `A: N=2, M=10` / `B: N=3, M=7` | **9.17e-04** | **3.7e-07** | 29.2 s |
| 24 | 1152 | union | `N=6, M=5` | 5.56e-04 | 2.1e-07 | 191.2 s |
| 24 | 1152 | **mortar** | `A: N=2, M=13` / `B: N=3, M=9` | **7.58e-05** | **2.9e-10** | 218.8 s |

**At equal DOF the mortar is 7.6x and 7.3x MORE accurate, and its lossless
closure is 170x and 720x tighter, at 1.15-1.24x the wall time.** The
"h-vs-p penalty" the roadmap's window-enrichment argument predicts is not
merely absent -- its sign is reversed.

Why, stated as physics rather than as a ratio: the tangential trace at the
interface is `C^0` with kinks at BOTH layers' walls, and h-refinement at the
NEIGHBOUR's walls (all the union grid adds) buys only the kink; p-refinement on
the own-walls grid buys the kink algebraically AND the smooth interior
spectrally, and the interior is where the modal content lives. The 1-D
own-walls-only failure (75-83 % spread,
`AUDIT_PMM_PER_LAYER_GRIDS_IMPL_2026_07_28.md` §3) was a nodal-SEM
boundary-layer defect in the wall-normal `Ex` channel at ARBITRARY wall
positions; on Granet's uniform lattice with a modified-Legendre basis it does
not reproduce. **[H, flagged]** the sign reversal itself is not proved here, only
measured on this pair at two `q`; it should be re-measured on a second geometry
class before it is quoted as a general property (open item O-2).

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
