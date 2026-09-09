# EXPERIMENT -- a NATIVE SLANT for the PURE staggered 2-D PMM (roadmap Phase D)

**Date:** 2026-09-10 - **Branch:** `probe/pmm2d-staggered-slant` off `main` @ `fb3fd93`
(the commit after the 5.43.0 release; ancestry verified) - **Worktree:** `C:/tmp/lum_slantp`
**Scope:** prototype + verdict + integration route.  **No library code was written.**
Everything lives under `validation/probe_pmm2d_staggered_slant/`.
**Binding law:** `docs/TESTING_STANDARDS.md`.

Tags: **[A]** analysis/derivation, no run - **[M]** measured on the build in S0 -
**[H]** hypothesis consistent with the evidence but not established.

---

## VERDICT UP FRONT -- **GO**

A constant x-z / y-z shear is a **small, structured addition** to the
first-order out-of-plane staggered generator that shipped on 2026-09-09: a
**pointwise 3x3 congruence** on the cell tensor plus **six extra Galerkin
blocks** built from per-axis matrices the basis already supplies.  It needs no
new basis code, no new cascade, no new far field, and no magnetic-anisotropy
generalization.

Every gate passes, each two-sided:

| gate | claim | result |
|---|---|---|
| **G0** | slant 0 reduces to the shipped path | **PASS** -- the pencil is **BIT-IDENTICAL** to `_assemble_oop` (0 differing bytes in `A` and `B`); the driver reproduces `PMM2DStackPure.solve` to `<= 4.2e-15` (and bit-identically on an out-of-plane cell) |
| **M1** | a uniform layer at any slant is a no-op | **PASS at machine precision** -- `2.2e-16` (R) / `2.4e-14` (Jones) at M=8, converging **spectrally** (`6.6e-08 -> 2.2e-16` over M=4..8) |
| **M1b** | the frame-anchor phase is real, analytic and unique | **PASS, two-sided** -- `exp(-i alpha_m . t d)` collapses the transmission Jones from `1.0e-01..6.2e-01` to `<= 9.3e-08`; the `+i` arm is 2x WORSE than no correction |
| **M2** | sheared-frame dispersion == exact quartic roots | **PASS** -- `1.3e-14 .. 3.6e-14` on the physical arm; the three wrong gauge/shift arms sit at `1.3e-02 .. 1.3e-01`; **dropping the six blocks** gives `8.6e-04..4.8e-02` and **dropping the congruence** `8.6e-04..3.8e-02` -- both halves of the formulation are load-bearing and each is measured |
| **M3** | y-uniform slanted stripe == the validated 1-D slant oracle, per order | **PASS** -- slant sign pinned uniquely (`t = -tan(phi)`: `1.2e-03` vs `4.2e-01`); TE `2.1e-07` at M=8 at slant 0/10/20/35, TM tracks (in fact beats) the vertical control at every M |
| **M4** | a genuinely 2-D slanted pillar, three ways | **PASS** -- see S4.4 |
| **M5** | census + cascade | **PASS** -- see S4.5 |
| **M6** | cost | see S4.6 |
| **M7** | slant x anisotropy | **PASS**, and it CLOSES a coverage gap -- see S4.7 |

**The formulation choice is the finding.**  The 1-D library carries the slant as
a lab-Cartesian **convection** (`d_z -> i q - t . grad` on the four diagonal
generator blocks).  That form **cannot be transplanted** into the staggered
basis: it needs `<B | d/dx | B>`, the derivative of the DISCONTINUOUS staggered
set, which is not a mimetic bracket there -- and, physically, the lab `E_z`
JUMPS across a slanted wall while `V3` is C0, so the Cartesian placement is
non-conformal at any `t != 0`.  The **covariant** frame (Granet 2017 Eq. 5 /
Li 1999) is conformal by construction: the covariant components have EXACTLY the
vertical continuity structure in the sheared frame (`E_1` jumps across
`u = const`; `E_2` and `E_3cov` are continuous), which is the same de Rham
placement `twod_staggered` already implements.  So in this basis the choice that
is HARD in 1-D is the EASY one, and the choice that ships in 1-D is the one that
does not fit.

**The 1-D V-partner wall (roadmap S8.5) does not arise.**  It was a
second-order-operator problem: a good pure-E operator whose reconstructed
magnetic partner `V = QW/lambda` drifted `1.0 -> 0.54 -> -0.18` over 0-60 deg.
Here `H` is part of the state -- the generator is first order on
`[E1; E2; G1; G2]` -- so there is nothing to reconstruct.  Measured, not
assumed: M2's uniform-slab dispersion is exact to `1e-14` at 45 deg slant (a
wrong partner cannot produce the right four roots), M5's cascade closes to
`1e-13` at three wavelengths with a forward growth factor of exactly `1.0`, and
M5's split is exactly `2q^2 / 2q^2` at every slant up to 60 deg.

---

## 0.  Build pin

| | |
|---|---|
| interpreter | CPython 3.14.6 |
| numpy / scipy | 2.4.4 / 1.17.1 (scipy-openblas) |
| `lumenairy.__file__` | `C:\tmp\lum_slantp\lumenairy\__init__.py` -- **asserted on import** by `slant_lib.py` |
| version | 5.43.0 + `fb3fd93` |
| thread caps | `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` set before python |
| host | Windows 11, tesla-ryzen |

Single-build campaign.  Per `TESTING_STANDARDS.md` rule 5 **no test bar may be
derived from these numbers** until they are re-measured on a second build.  They
adjudicate a GO/NO-GO, which is a decision about *shape* -- convergence
direction, null exactness, sign uniqueness, and the failure of the deliberately
wrong arms -- not about a reading.

Scripts and their commands: `validation/probe_pmm2d_staggered_slant/README.md`.
JSON in that directory's `results/`.

---

## 1.  THE FORMULATION

### 1.0  The papers, identified (the roadmap's Phase D citation, resolved)

`PMM_Papers/` holds three candidates for what the roadmap calls
"Edee-Granet 2024 ~3.4x vs FMM".  Read, they separate cleanly:

| file | what it is | relevance |
|---|---|---|
| **`josaa-41-9-1803.pdf`** | **Edee & Granet, "Polynomial modal method for crossed slanted gratings", JOSA A 41(9) 1803 (2024)** | **THE Phase D paper** -- the 2-D slanted PMM, in the covariant frame, on the first-order `[F1; F2; G1; G2]` generator |
| `josaa-35-4-608.pdf` | Edee, Plumey, Moreau & Guizal, "Matched coordinates ... for complex metasurface modeling", JOSA A 35(4) 608 (2018) | a DIFFERENT technique (a contour-following coordinate for non-axis-aligned patterns), not the slant; the roadmap S5 already records matched coordinates as a dead end in a wall-fitted PMM |
| `josaa-38-1-52.pdf` | Granet, "Spectral element method with modified Legendre polynomials for modal analysis of lamellar gratings", JOSA A 38(1) 52 (2021) | the 1-D modified-Legendre SEM basis itself -- what `Basis1D` implements |
| `josaa-34-6-975.pdf` | Granet, Randriamihaja & Raniriharinosy, "Polynomial modal analysis of slanted lamellar gratings", JOSA A 34(6) 975 (2017) | the 1-D covariant slant the library's 1-D path implements |
| `josaa-40-4-652.pdf` | Granet, JOSA A 40(4) 652 (2023) | the 2-D staggered basis `twod_staggered` implements.  Its Eq. 38 (`x = x1 + sin(eta) x2`, `y = x2 cos(eta)`) is a TILTED UNIT CELL -- an IN-PLANE `x-y` shear for a non-rectangular lattice -- **not** the `x-z` sidewall slant this document is about.  Do not reuse it for a slant |

**[M] The "3.4x" is an S-MATRIX SIZE ratio at matched accuracy, not a wall
clock.**  josaa-41-9-1803 S4.B: on a 2-D slanted grating, "with `N_b = 55`
layers, the FMM reaches ... an S-matrix size of 6724, whereas the PMM shows
convergence curves leveling off ... around an S-matrix size of 2000"
(`6724 / 2000 = 3.36`).  The same section states the honest cost structure the
present campaign re-measures in M6: the PMM's slanted eigenproblem is
`4(2N)(2N)` -- "twice as large as ... the FMM ... in each layer" -- but it
replaces `N_b` of them.

**[A] Three points where the papers confirm this derivation textually**, which
is why S1.2-S1.4 below reads as a re-derivation rather than an invention:

* josaa-34-6-975 Eq. 8 puts `sqrt(g) g^{ij}` on **both** `D^i = eps_0 eps
  sqrt(g) g^{ij} E_j` and `B^i = mu_0 sqrt(g) g^{ij} H_j` -- i.e. the shear
  makes the medium MAGNETICALLY anisotropic too, exactly as S1.3 finds;
* josaa-34-6-975 Eq. 6 gives `A_1 = A_x`, `A_3 = A_z`, `A_2 = tan(phi) A_x +
  A_y` (their propagation axis is `y`) with the remark that the interface
  planes "are common to all coordinate systems and ... the only Cartesian
  component ... affected by the change of coordinates is `A_y`" -- the
  roadmap's C-FRAME identity match, published;
* josaa-41-9-1803 S3 assigns its expansion spaces as `F_1, G_2, D^2` in
  `|P> (x) |Q~>` (discontinuous in `x^1`, continuous in `x^2`), `F_2, G_1, D^1`
  in `|P~> (x) |Q>`, and `D^3` in `|P> (x) |Q>` -- which is **exactly the
  staggered de Rham placement `twod_staggered` already implements**
  (`V1 = B (x) Btilde`, `V2 = Btilde (x) B`, `Vw = B (x) B`), stated for the
  SLANTED case.

**[A] One convention DIFFERS from the 2024 paper, in the library's favour.**  Its
Eq. 16 scales the longitudinal coordinate (`z = x^3 cos(theta_s)`), so
`sqrt(g) = 1/cos(theta_s) != 1` (its Eq. 18) and its `mu^33 = cos(theta_s)`;
its `G_3` elimination therefore divides by `mu^33` (its Eq. 11).  The library's
1-D convention -- and this campaign's -- is the volume-preserving one of
josaa-34-6-975 Eq. 2 (`w = z`, `det J = 1`), where **`mu^33 = g^33 = 1
exactly**, so the shipped STRONG `G_3` elimination through the curl survives
untouched, and `eps^33 = eps_zz` is unchanged by the congruence, so the shipped
POINTWISE `e33`-Schur survives untouched too.  Both are load-bearing for
"nothing downstream changes" (S1.4).

### 1.1  Conventions -- and which of the library's two they follow

Frame anchored at the layer **TOP**, matching the shipped 1-D
(`stack.py:690-692`) and hybrid (`BUILD_PMM2D_SLANT_METRIC_2026_08_16` S1.2)
convention:

```
  u = x - t_x w ,   v = y - t_y w ,   w = z ,      t = (t_x, t_y)

           d(x,y,z)     [ 1  0  t_x ]                    [ 1  0  -t_x ]
    A  =  ---------- =  [ 0  1  t_y ]  ,     A^-1  =     [ 0  1  -t_y ]
           d(u,v,w)     [ 0  0   1  ]                    [ 0  0    1  ]
```

`det A = 1` EXACTLY, for any slant vector, in 2-D as in 1-D (`A` is unit
upper-triangular).  So `sqrt(g) = 1`: the mass matrices are untouched, there is
no dilation generator, and `w = z` means the constant-`w` surfaces ARE the lab's
constant-`z` surfaces.  The three properties that make the shear tractable and
the taper not (`BUILD_PMM2D_SLANT_METRIC_2026_08_16` S1.1) carry over verbatim;
**this document is the shear.  A taper is a different geometry and is out of
scope** (M5's own measurement in that document: a shear contributes `1.00x` to a
taper).

Covariant / contravariant metrics `g_ij = A^T A`:

```
            [  1    0    t_x  ]                 [ 1+t_x^2   t_x t_y   -t_x ]
  g_ij  =   [  0    1    t_y  ]      g^ij  =    [ t_x t_y   1+t_y^2   -t_y ]
            [ t_x  t_y  1+|t|^2]                [  -t_x      -t_y       1  ]
```

**[M] The SIGN of `t` relative to the library's `slant_angle` is measured, not
inherited.**  The shipped 1-D tree carries two mutually inconsistent slant sign
conventions (`BUILD_PMM2D_SLANT_METRIC_2026_08_16` S1.5 records both), so M3
arbitrates it against `pmm_efficiency_1d_slanted` by running BOTH: the winner is
`t_x = -tan(slant_angle)` (`1.2e-03` vs `4.2e-01`, three decades, S4.3).

### 1.2  Why COVARIANT and not the 1-D CONVECTION form

Both forms are algebraically the same operator split two ways
(`BUILD_PMM2D_SLANT_METRIC_2026_08_16` S1.5), and in the 1-D **nodal** basis --
where every field component lives in ONE C0 space -- the convection form is the
one that ships, because it is robust and because there the two forms have the
same conformity.  In the **staggered** basis they do not.

**[A] The lab-Cartesian convection does not fit this basis, for two independent
reasons.**

1. **A bracket the basis does not have.**  With `d_z|_x = d_w|_u - t . grad`,
   the term `-i (t . D)` lands on each of the four diagonal blocks of the state
   `[E_1; E_2; G_1; G_2]`.  `E_1` lives in `V1 = B(x) (x) Btilde(y)` and `G_2`
   likewise, so `t_x D_1 E_1` needs `<B | d/dx | B>` -- the derivative of the
   DISCONTINUOUS set, whose element-wise Galerkin form silently drops the jump
   deltas.  (`<Btilde | d B>` IS available, as `-<B | d Btilde>^H`, and the
   shipped assembly already uses exactly that for `CwE1`/`CwE2`; `<B | d B>` is
   not.)  This is the 2-D face of the roadmap's **C-CONV** warning that the
   naive convection "converges cleanly to a wrong, energy-violating answer".
2. **Non-conformal placement.**  Across a slanted wall the outward normal is
   `(1, 0, -t_x)/|.|`, so the continuous quantities are `E_y` and
   `t_x E_x + E_z` -- NOT `E_z`.  The lab `E_z` jumps by `-t_x [E_x]`, while
   `V3 = Btilde (x) Btilde` is C0.  The Cartesian placement is therefore wrong
   by `O(t [E_x])` at any nonzero slant, independent of the bracket problem.

**[A] The covariant components fix both at once.**  Define

```
  E_cov = A^T E_lab :   (E_1, E_2, E_3) = ( E_x , E_y , t_x E_x + t_y E_y + E_z )
  G_cov = A^T G_lab :   (G_1, G_2, G_3) = ( G_x , G_y , t_x G_x + t_y G_y + G_z )
  G^3   = (A^-1 G_lab)_3 = G_z            (the CONTRAVARIANT longitudinal one)
```

with `G = i Z0 H` the shipped generator's magnetic state.  `E_3` is exactly the
combination that IS continuous across the slanted wall (this is the roadmap's
**C-FRAME** formula `Az_cov = tan(phi) Ax + Az`), and `E_1` is exactly the one
that jumps.  **The continuity structure in the frame is identical to the
vertical case**, so the shipped de Rham placement -- `E1` in `V1`, `E2` in `V2`,
`E3` in `V3`, `G1` in `V2`, `G2` in `V1`, `G3` in `Vw` -- is conformal for the
covariant components at any slant, with nothing moved.

### 1.3  The sheared-frame system, derived

Write the lab Maxwell system in normalized form (`D_j = d_j / k0`,
`D x E = G`, `D x G = eps E`), substitute `d_x = d_u`, `d_y = d_v`,
`d_z = d_w - t_x d_u - t_y d_v`, take `d_w -> i q k0`, and eliminate the lab
`E_z = E_3 - t_x E_1 - t_y E_2` in favour of the covariant `E_3`.  Two
cancellations do all the work:

* in the `E_1` row the leftover convection collapses onto `D_1 E_2 - D_2 E_1`,
  which IS `G^3` -- so `-t_x D_1 E_1` becomes `+t_y G^3`, and the offending
  bracket never appears;
* in the `G_1` row the same substitution collapses the convection onto
  `D_1 G_3cov`, i.e. the SHIPPED single-derivative term with `G^3` replaced by
  the covariant `G_3 = G^3 + t_x G_1 + t_y G_2`.

The result is the vertical system with two rigid substitutions:

```
 (3)  D1 E2 - D2 E1 = G^3                                      [Vw,  STRONG]
 (2)  q E1 = -i G2 - i D1 E3          + i t_y G^3              [V1]
 (1)  q E2 = +i G1 - i D2 E3          - i t_x G^3              [V2]
 (6)  (eps E)_3 = D1 G2 - D2 G1                                [V3,  WEAK]
 (5)  q G1 = -i (eps E)_2 - i D1 ( G^3 + t_x G1 + t_y G2 )     [V2]
 (4)  q G2 = +i (eps E)_1 - i D2 ( G^3 + t_x G1 + t_y G2 )     [V1]
```

with, throughout,

```
    eps^{lm}  =  A^-1  eps_lab  A^-T          (sqrt g = 1)
```

**[A] Consistency check (done symbolically, and it is what pins the six blocks).**
The system above is exactly

```
    curl'(E_cov) = mu^{lm} G_cov ,   curl'(G_cov) = eps^{lm} E_cov ,
    mu^{lm} = g^{lm} = A^-1 A^-T
```

-- i.e. the standard covariant Maxwell pair (josaa-34-6-975 Eqs. 7-8;
josaa-41-9-1803 Eqs. 4-6, whose `F_m = E_m`, `G_m = i Z H_m` is the state the
shipped generator already carries).  Every row of the six was verified against
that pair term by term.  The 2024 paper reaches the same first-order
`-i k gamma [F1; F2; G1; G2] = L [F1; F2; G1; G2]` (its Eq. 15) by eliminating
`F_3` and `G_3`; what is added here is the placement of that system on the
SHIPPED staggered de Rham spaces, and the observation that with `det J = 1` the
two eliminations reduce to the ones already in `_assemble_oop` (S1.0).

**[A] This is why the state and the cascade are untouched.**  The retained
transverse state `[E_1; E_2; G_1; G_2]` is the covariant tangential state, and
the covariant TANGENTIAL components EQUAL the lab-Cartesian ones (only the
NORMAL component is altered by a shear).  So the modal matrix `W`, the partner
`V`, the flux form, the forward/backward selector, `_OOP_H_GAUGE`, the
generalized S-matrix and the far-field projector all mean exactly what they
meant before.

**[A] And it is why `mu != I` costs nothing.**  A naive covariant
implementation would have to carry the magnetic anisotropy `mu^{lm} = g^{lm}`
as a new tensor in the generator (the shipped one assumes `mu = I`).  The
derivation above shows the whole of `mu^{lm}`'s content is (i) the two `t G^3`
terms in the E rows, where `G^3` is ALREADY eliminated strongly, and (ii) the
`t_x G_1 + t_y G_2` inside `G_3cov` in the G rows.  No magnetic-tensor machinery
is needed.

### 1.4  What the assembly actually gains -- the whole of it

Against `Granet2DTransverseE._assemble_oop` (`twod_staggered.py:803`):

**(a) one pointwise congruence.**  `eps_cell -> A^-1 eps_cell A^-T`, a
`(Nx, Ny, 3, 3)` einsum.  Nothing downstream changes: the nine eps-weighted
masses `A11..A33`, the `e33`-Schur (`E3S`), the `Vw` elimination (`G3S`) and
their set pairs are all as shipped.  An ISOTROPIC slanted cell becomes an
out-of-plane tensor cell in the frame (`eps^{13} = -t_x eps`), which is exactly
why the slant belongs on the out-of-plane generator and not on the `2q^2`
in-plane pencil.

**(b) six extra Galerkin blocks**, each a Kronecker product of per-axis matrices
the basis already builds:

| block | meaning | built from |
|---|---|---|
| `<V1\|Vw>` | `+i t_y G^3` in row 0 | `kron(<til\|B>_y, <B\|B>_x)` |
| `<V2\|Vw>` | `-i t_x G^3` in row 1 | `kron(<B\|B>_y, <til\|B>_x)` |
| `<V2\|D1\|V2>` | `t_x G_1` in `G_3cov`, row 2 | `kron(<B\|B>_y, Ctt_x/k0)` |
| `<V2\|D1\|V1>` | `t_y G_2` in `G_3cov`, row 2 | `kron(<B\|til>_y, -(dbt_x)^H)` |
| `<V1\|D2\|V2>` | `t_x G_1` in `G_3cov`, row 3 | `kron(-(dbt_y)^H, <B\|til>_x)` |
| `<V1\|D2\|V1>` | `t_y G_2` in `G_3cov`, row 3 | `kron(Ctt_y/k0, <B\|B>_x)` |

`Ctt` is already in `_axis_mats`; `-(dbt)^H` is the SAME
integration-by-parts identity the shipped assembly uses for `CwE1`/`CwE2`
(and it is the distributionally exact `<Btilde | d B>`, deltas included).  The
two cross masses `<til|B>` / `<B|til>` are one `Basis1D.mass` call each.

**(c) the rotation gauge.**  `_OOP_ROT_SIGN` is a 180-degree rotation about `z`
between the basis and the `eps_cell` / far-field indexing.  Under it
`t -> -t` as well as `e13, e23, e31, e32 -> -`, so the prototype applies BOTH
once, up front, and builds in the rotated gauge.  **[A] The two are consistent:**
`eps^{lm}(R eps R, -t) = R eps^{lm}(eps, t) R`, verified symbolically and
confirmed by M2's gauge arms.

### 1.5  Interfaces -- and the ONE thing a shear does add

**[A] The tangential match is the IDENTITY**, exactly as the roadmap's C-FRAME
contract states: `det g = 1`, the interface `z = const` is a coordinate surface
common to both frames, and the only Cartesian component the shear alters is the
NORMAL one, which is not matched.  Isotropic half-spaces, uniform regions and
vertical layers enter unchanged.

**[A] The frame anchor, however, is physical bookkeeping and must be paid.**
With the frame anchored at the layer top, the layer's BOTTOM face sits at
`u = x - t d`.  For a region bounded BELOW by a HOMOGENEOUS medium that lateral
offset is a **gauge** -- a translation maps a homogeneous half-space to itself
-- so it changes nothing in `S11` and appears in `S21` as ONE unimodular
diagonal phase per order:

```
    T_lab(m)  =  exp( -i alpha_m . t d )  T_frame(m)
```

`alpha_m` is real for every order, propagating or evanescent, so the factor is
always unimodular: **efficiencies are untouched and the REFLECTION Jones is
exact; only the TRANSMISSION Jones carries it.**  That is also why neither
paper mentions it -- josaa-34-6-975 and josaa-41-9-1803 both report
EFFICIENCIES, where a unimodular per-order factor is invisible; this library
returns Jones matrices, where it is not.  M1 measures its size
(`0.10 .. 0.62` on a uniform null at 35 deg slant and 25 deg incidence -- large
enough that omitting it would look like a formulation error) and M1b pins the
sign two-sided (S4.1).

**[A] The scope limit that follows.**  In a stack, the offset ACCUMULATES as
`sum_j t_j d_j`.  Between two regions with the SAME slant the identity match is
exact (one global frame; the shift is carried by the far field once).  Between
regions with DIFFERENT slants -- including a slanted layer above a VERTICAL
PATTERNED one -- the offset is a real lateral translation of one pattern
relative to the other, and in a nodal SEM basis that is neither diagonal nor
exactly representable unless it is a whole number of grid cells.  So:

* slanted layer(s) + homogeneous regions: **exact**;
* a stack in which every patterned layer shares one slant: **exact**;
* mixed slants between PATTERNED layers: **out of scope** (see S6, "what the
  build must refuse").

---

## 2.  The prototype

`validation/probe_pmm2d_staggered_slant/slant_lib.py`:

* `cov_tensor(eps, tx, ty)` -- the pointwise congruence `A^-1 eps A^-T`;
* `SlantSolver` -- `Granet2DTransverseE` with `_assemble_slant()`, which is the
  shipped `_assemble_oop` body with `rot` folded in up front plus the six blocks
  of S1.4(b).  It **always** takes the first-order generator (a sheared cell is
  an out-of-plane cell in the frame) and at `t = 0` is BIT-IDENTICAL to the
  shipped assembly;
* `slant_region_modes` -- `_region_modes_oop` verbatim (Cholesky whitening, the
  flux split with the deep-decay override, `_OOP_H_GAUGE`), returning the
  generalized 6-tuple plus the raw spectrum for the census;
* `solve_slant_stack` -- a `PMM2DStackPure.solve` clone with a per-layer
  `slant`, including the frame-anchor phase of S1.5.

It imports the shipped private helpers rather than copying them, and raises on
import if `lumenairy.__file__` is not inside the worktree.  **No file under
`lumenairy/` or `tests/` was modified.**

---

## 3.  [M] G0 -- the reduction gates

`g0_reduction.py`.

**G0c (algebra).**  `A^-1 (A eps A^T) A^-T == eps` to `1.2e-16` on a random
complex `(3,3)`; `cov_tensor(eps, 0, 0) == eps` exactly; `det A - 1 = 0`.

**G0a (the pencil).**  Out-of-plane cell (`uniaxial(1.5, 1.7, tilt 35, azim 25)`
in one of four cells), `px = py = 0.9`, `M = 5`, conical Bloch shift.  The
prototype's `(A, B)` vs the shipped `_assemble_oop`'s:

| | differing bytes | max abs diff |
|---|---|---|
| `Agen` | **0** | `0.0` |
| `Bgen` | **0** | `0.0` |

**G0b (the driver).**  `solve_slant_stack` at `slant = 0` vs
`PMM2DStackPure.solve`, `M = 5`, `n_orders = 4`:

| cell | mount | dR | dT | dJones |
|---|---|---|---|---|
| scalar pillar | normal | 2.98e-16 | 1.44e-15 | 4.22e-15 |
| scalar pillar | conical 0.25/0.6 | 5.28e-16 | 3.86e-15 | 1.86e-15 |
| in-plane tensor pillar | normal | 1.63e-16 | 6.11e-16 | 9.21e-16 |
| in-plane tensor pillar | conical | 1.63e-16 | 3.11e-15 | 8.54e-16 |
| **out-of-plane tensor pillar** | normal | **0.0** | **0.0** | **0.0** |
| **out-of-plane tensor pillar** | conical | **0.0** | **0.0** | **0.0** |

The scalar / in-plane rows are not bit-identical only because the prototype
drives the GENERALIZED cascade for every layer while the library uses the square
one there; the size (`<= 4.2e-15`) matches the shipped measurement of that same
substitution (`7.7e-14`, `BUILD_PMM2D_STAGGERED_OOP_2026_09_09` S11).  The
out-of-plane rows, which run the same cascade on both sides, are exact.

---

## 4.  [M] The measurements

### 4.1  M1 -- the NULL TEST

`m1_null.py`, `m1b_frame_phase.py`, `m1c_null_ladder.py`.  A shear of a
HOMOGENEOUS medium is a pure coordinate change, so a uniform layer at any slant
must return the unslanted answer.  Five tensors (isotropic, in-plane uniaxial,
OUT-OF-PLANE uniaxial, gyrotropic, lossy out-of-plane) x three mounts x four
slants (`x10`, `x35`, `y35`, `diag35`) = 60 rows, `M = 5`.

**[M] Worst residual over all 60 rows** (reference: the same cell at slant 0):

| quantity | worst overall | worst at oblique / conical only |
|---|---|---|
| `R` per order | 3.05e-09 | 3.05e-09 |
| `T` per order | 2.40e-08 | 2.40e-08 |
| reflection Jones | 1.60e-08 | 1.60e-08 |
| non-(0,0) order leak | 1.45e-28 | 1.45e-28 |
| **transmission Jones (no S1.5 phase)** | **6.21e-01** | **6.21e-01** |

At NORMAL incidence every row is `<= 3.0e-14` and the leak is exactly `0.0`
(`k_t = 0` kills the shear's coupling; the oblique rows are the informative
ones, as in the hybrid's A2).  A `y`-only shear at `phi_inc = 0` is likewise
`<= 2.9e-14` -- shearing along a direction that carries no transverse momentum
is a no-op, which tests the VECTOR structure of the six blocks and has no 1-D
analogue.

**[M] M1c -- the residual is DISCRETIZATION, and it is SPECTRAL.**  Worst M1 row
(isotropic uniform, oblique 25, slant 35 deg), Jones columns carrying the S1.5
phase:

| M | dim | dR | dT | dJones(R) | dJones(T) |
|---|---|---|---|---|---|
| 4 | 144 | 6.62e-08 | 1.50e-06 | 2.05e-06 | 1.64e-05 |
| 5 | 256 | 1.79e-09 | 7.76e-09 | 9.82e-09 | 9.28e-08 |
| 6 | 400 | 2.28e-12 | 3.38e-11 | 2.50e-11 | 3.45e-10 |
| 7 | 576 | 3.87e-15 | 3.32e-14 | 5.34e-14 | 9.05e-13 |
| 8 | 784 | **2.22e-16** | 4.04e-14 | **2.09e-15** | **2.40e-14** |

Eight decades over four steps of `M`, ending at machine precision.  **M1
PASSES at machine precision**, and the M1 table's `1e-09` figures are simply the
`M = 5` rung of this ladder.

**[M] M1b -- the frame-anchor phase, pinned two-sided.**  Uniform null,
transmission Jones, three arms:

| tensor | mount | slant | NO correction | `exp(-i a.t d)` | `exp(+i a.t d)` |
|---|---|---|---|---|---|
| isotropic | oblique 25 | x10 | 1.35e-01 | **5.85e-09** | 2.70e-01 |
| isotropic | oblique 25 | x35 | 5.29e-01 | **9.28e-08** | 1.00e+00 |
| isotropic | conical 25/40 | x35 | 3.97e-01 | **1.10e-08** | 7.70e-01 |
| isotropic | conical 25/40 | diag35 | 6.20e-01 | **1.10e-08** | 1.15e+00 |
| out-of-plane uniaxial | oblique 25 | x35 | 5.29e-01 | **7.65e-08** | 1.00e+00 |
| out-of-plane uniaxial | conical 25/40 | diag35 | 6.20e-01 | **8.97e-09** | 1.14e+00 |

Worst `-i` arm `9.28e-08` (the `M = 5` discretization floor of that row, per
M1c); best `none` arm `1.01e-01`; best `+i` arm `2.02e-01`.  **The `+i` arm is
twice as wrong as doing nothing** -- the correction is not a fudge that could
absorb any residual.  The measured sizes also match the analytic prediction
`|exp(i alpha_0 . t d) - 1| . |T|` to the printed digits (x10 at oblique 25:
`2 sin(pi . 0.4226 . 0.1763 . 0.35) = 0.164`, times `|T| ~ 0.83` = `0.136`, vs
`1.35e-01` measured).

### 4.2  M2 -- SHEARED-FRAME DISPERSION (the sign / factor gate)

`m2_dispersion.py`.  For a UNIFORM cell the generator's eigenvalues must be

```
   q(m,n) = kz_root(eps_lab; u, v)  +  t_x u + t_y v          (k0 units)
```

with `kz_root` the four EXACT roots of `det(k k^T - |k|^2 I + eps) = 0` in exact
polynomial arithmetic (`probe_pmm2d_staggered_oop/probe_common.exact_kz_roots`).
The shear translates each harmonic's four roots RIGIDLY -- no tensor error can
mimic that, which is what makes this the sign / factor gate.  `M = 8`, `(2,2)`
grid, `px = py = 0.9`.  Reported: max over the fundamental's four exact roots of
the distance to the nearest generator eigenvalue.

**[M] The four gauge / shift arms** (`a+/a-` = transverse gauge `+alpha` /
`-alpha`; `s+/s-` = shift `+t.alpha` / `-t.alpha`):

| tensor | mount | slant | **a+s+** | a+s- | a-s+ | a-s- |
|---|---|---|---|---|---|---|
| uniaxial tilt35 azim25 | conical 25/40 | none | **3.15e-14** | 3.15e-14 | 9.16e-02 | 9.16e-02 |
| uniaxial tilt35 azim25 | conical 25/40 | x20 | **2.72e-14** | 1.00e-01 | 1.31e-02 | 9.16e-02 |
| uniaxial tilt35 azim25 | conical 25/40 | diag35 | **3.62e-14** | 9.88e-02 | 1.05e-01 | 9.16e-02 |
| non-reciprocal `e13 = -conj(e31)` | conical 25/40 | x20 | **2.05e-14** | 5.68e-02 | 5.68e-02 | 2.05e-14 |
| non-reciprocal | conical 25/40 | diag35 | **1.60e-14** | 4.57e-02 | 4.57e-02 | 1.61e-14 |
| isotropic 2.25 | conical 25/40 | diag35 | **1.81e-08** | 1.26e-01 | 1.26e-01 | 1.81e-08 |

At NORMAL incidence all four arms coincide (`alpha = 0` kills both the gauge and
the shift), exactly as the shipped Stage-B rotation gate found; the conical rows
are the discriminator.  The isotropic rows sit at `1.8e-08` **at every slant,
INCLUDING zero** -- that is the double-root conditioning of an isotropic tensor
(`sqrt(eps_machine)`), not a slant effect.  The `a-s-` arm passes only where the
tensor is `alpha -> -alpha` symmetric, i.e. it is not a real second solution.

**[M] The two ABLATION CONTROLS -- both halves of the formulation are
load-bearing, and each is measured:**

| tensor | mount | slant | full | six blocks REMOVED | congruence REMOVED |
|---|---|---|---|---|---|
| uniaxial | conical 25/40 | x20 | **2.72e-14** | 3.88e-02 | 1.87e-02 |
| uniaxial | conical 25/40 | diag35 | **3.62e-14** | 3.19e-02 | 2.86e-02 |
| isotropic | conical 25/40 | x20 | **1.81e-08** | 1.85e-02 | 1.85e-02 |
| isotropic | conical 25/40 | diag35 | **1.81e-08** | 8.56e-04 | 8.56e-04 |
| non-reciprocal | conical 25/40 | x20 | **2.05e-14** | 4.19e-02 | 3.58e-02 |
| non-reciprocal | conical 25/40 | diag35 | **1.60e-14** | 4.82e-02 | 3.79e-02 |

**[M] The sum-of-roots discriminator** (the `AUDIT_OOP_GENERATOR_FACTOR_I`
asymmetric-pair test, re-asked with a shear): the fundamental's four roots sum
to `4 t.alpha` above their vertical value.  Generator vs exact, uniaxial at
conical 25/40 -- `none`: `-0.091625` / `-0.091625`; `x20`: `+0.379709` /
`+0.379709`; `diag35`: `+1.347728` / `+1.347728`.

**[M] M-ladder** (uniaxial, conical 25/40, diag35): `1.40e-06` (M=4),
`4.65e-09`, `9.98e-12`, `2.18e-14`, `3.62e-14`, `1.33e-14` (M=9) -- spectral to
the eigenvalue-conditioning floor.  **M2 PASSES.**

### 4.3  M3 -- the y-uniform SLANTED STRIPE vs the 1-D oracles

`m3_stripe_vs_1d.py`, `m3b_oracle_drift.py`, `m3c_tm_spectral_oracle.py`,
`m3d_tm_deep_ladder.py`.  Cell: `px = py = 0.75 lam`, binary x-grating
`n = 2 / 1`, duty 0.5, `depth = 0.30 lam`, `n_sub = 1.5`, `wl = 1`.  Comparison
is per order over `m = -2..2`, on both R and T; the 2-D rows use incident `Ex`
(= TM at `phi_inc = 0`) and `Ey` (= TE).

**[M] T3a -- the slant sign, arbitrated by the oracle** (`phi = 35 deg`,
`M = 7`; "ctrl" is the same 2-D cell at slant 0 against `pmm_efficiency_1d`):

| mount | pol | vertical ctrl | `t = +tan(phi)` | `t = -tan(phi)` |
|---|---|---|---|---|
| normal | TE | 1.07e-07 | 4.15e-01 | **4.28e-06** |
| normal | TM | 1.14e-03 | 2.24e-01 | **1.22e-03** |
| oblique 25 | TE | 1.79e-06 | 2.02e-02 | **1.44e-06** |
| oblique 25 | TM | 1.25e-04 | 1.27e-01 | **2.24e-04** |

`t = -tan(slant_angle)`, uniquely: the wrong sign is off by two to five decades
and does not track the control.  A slanted grating is not x-mirror symmetric, so
`R_{+1} != R_{-1}` and the sign is observable PER ORDER -- which is why this,
and not an energy check, is the place to pin it.

**[M] T3b -- the per-order ladder** (winning sign; `yleak` = worst amplitude
scattered into any `n != 0` order, which a mis-placed `t_y` block would break):

| phi | mount | M | TE | TM | y-leak | closure |
|---|---|---|---|---|---|---|
| 0 | normal | 8 | 1.07e-07 | 4.90e-04 | 5.3e-26 | 5.24e-09 |
| 10 | normal | 8 | 1.15e-07 | 4.88e-04 | 8.8e-26 | 4.42e-09 |
| 20 | normal | 8 | 2.00e-07 | 7.69e-04 | 9.9e-26 | 3.39e-09 |
| 35 | normal | 8 | 2.65e-07 | 1.27e-03 | 8.7e-26 | 7.66e-09 |
| 0 | oblique 25 | 8 | 1.71e-07 | 8.35e-05 | 1.2e-26 | 5.21e-11 |
| 10 | oblique 25 | 8 | 1.19e-07 | 1.30e-04 | 6.8e-26 | 1.18e-10 |
| 20 | oblique 25 | 8 | 9.94e-08 | 1.60e-04 | 6.1e-26 | 1.39e-10 |
| 35 | oblique 25 | 8 | 2.11e-07 | 1.48e-04 | 6.8e-26 | 7.41e-11 |

TE reaches `1e-07` at EVERY slant, tracking the vertical control; the lossless
closure at slant 35 (`7.4e-11`) is as good as at slant 0; y-momentum is
conserved to `1e-26`.  (Closure is reported as CONTEXT, never as a pass
criterion -- the lossless trap is a named hazard for slant work.)

**[M] T3b's TM column is ORACLE-limited, and M3b proves it.**  The 1-D scalar
slant oracle's OWN per-order drift on the same cell (its own degree ladder,
reference degree 34):

| phi | mount | pol | deg 18 | deg 22 | deg 26 | deg 30 |
|---|---|---|---|---|---|---|
| 0 | normal | TM | 2.53e-05 | 1.18e-05 | 6.49e-06 | 2.37e-06 |
| 20 | normal | TM | 8.08e-04 | 4.66e-04 | 2.50e-04 | 1.04e-04 |
| 35 | normal | TM | 1.13e-03 | 6.51e-04 | 3.49e-04 | 1.45e-04 |
| 35 | normal | TE | 1.93e-10 | 4.16e-11 | 1.02e-11 | 3.34e-12 |

The oracle's TM falls ALGEBRAICALLY and is itself `6.5e-04` at the degree T3b
uses -- the documented `~1e-4` wall-normal floor of the 1-D metric route.  Its
TE is `4e-11`, i.e. converged.  So T3b's TE column is a real measurement of the
2-D slanted arm and its TM column is not.

**[M] M3c -- against a CONVERGED TM oracle.**
`pmm_jones_1d_slanted(factorization='covariant')` (the Li-1999 oblique-coordinate
1-D path) driven with isotropic tensors has its own TM drift of `1.2e-06 ..
3.6e-05` at degree 26-30, and differs from the scalar oracle by `2.7e-04 ..
8.1e-04` on TM at normal incidence (and by exactly `0.00e+00` at oblique, where
`pmm_efficiency_1d_slanted` routes into the same solver).  Against it, at
`M = 8`: TM `4.53e-04` (phi 10, normal), `3.84e-04` (20), `1.66e-04` (35),
`1.49e-04` (35, oblique 25); TE `1.15e-07 / 2.00e-07 / 2.65e-07 / 2.11e-07`.

**[M] M3d -- the decisive control: SLANTED vs VERTICAL, same cell, deep ladder**
(normal incidence, against the converged covariant / vertical 1-D Jones oracle
at degree 30):

| M | dim | slant 0: TM / TE | slant 35 deg: TM / TE |
|---|---|---|---|
| 5 | 256 | 2.12e-03 / 1.26e-04 | 7.66e-04 / 2.76e-03 |
| 6 | 400 | 1.13e-03 / 1.26e-04 | 3.72e-04 / 3.04e-05 |
| 7 | 576 | 1.13e-03 / 1.07e-07 | 3.32e-04 / 4.28e-06 |
| 8 | 784 | 4.80e-04 / 1.07e-07 | 1.66e-04 / 2.65e-07 |
| 9 | 1024 | 4.80e-04 / 8.74e-09 | 1.44e-04 / 1.04e-07 |
| 10 | 1296 | **2.46e-04 / 8.74e-09** | **8.50e-05 / 4.26e-08** |

The slanted arm's TM residual is **smaller than the vertical arm's at every
M**, and both fall algebraically -- i.e. the `~1e-4` TM level is the SHIPPED
staggered basis's own wall-normal behaviour on this cell, and the shear does not
touch it.  TE is spectral on both arms, within a factor of five of each other.

**M3 PASSES:** the slanted arm tracks the vertical control everywhere the
oracles can resolve it, and the sign is pinned two-sided.
