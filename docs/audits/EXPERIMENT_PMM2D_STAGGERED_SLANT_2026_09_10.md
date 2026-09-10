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
| **M4** | a genuinely 2-D slanted pillar, three ways | **PASS** -- the INDEPENDENT hybrid agrees to `5.4e-03` (normal) with its OWN truncation step at `4.4e-03`, and walks toward the prototype monotonically at conical (`5.28e-02 -> 1.70e-02` over `n_orders` 5..11); the corrected-direction pure staircase converges `1.40e-01 -> 4.71e-02 -> 2.72e-02 -> 1.48e-02`; the prototype is grid- and degree-invariant to `2.1e-04 .. 6.1e-04` across four `(Nx, M)` settings |
| **M5** | census + cascade + no-floor | **PASS** -- split exactly `2q^2/2q^2` in all 20 census rows to slant 60 deg (scalar, out-of-plane, `eps`=12, lossy); forward growth exactly `1.0000e+00` on every lossless row; one layer == two half-layers to `1.3e-15`; `n_orders` 3->8 moves the answer `2e-16` (normal) / `3e-06` (conical) |
| **M6** | cost | **the slant is FREE**: `0.86x .. 1.07x` the vertical OUT-OF-PLANE region solve it has to use anyway; `1.37x .. 1.41x` end to end vs a vertical scalar solve |
| **M7** | slant x anisotropy | **PASS**, and it CLOSES a coverage gap -- slant x OUT-OF-PLANE matches `pmm_jones_1d_slanted` at `1.7e-05 .. 2.8e-05` per order while `PMM2DStackHybrid` raises `NotImplementedError` on that combination (measured, S4.7) |

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
assumed: M2's uniform-slab dispersion is exact to `1.3e-14` at 35 deg slant and
conical incidence (a wrong partner cannot produce the right four roots), M5's
forward growth factor is exactly `1.0000e+00` at three wavelengths with closure
that does not grow with depth, and M5's forward/backward split is exactly
`2q^2 / 2q^2` in all 20 census rows at slants to 60 degrees.

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

### 4.4  M4 -- a genuinely 2-D SLANTED PILLAR, three ways

`m4_pillar.py`, `m4a_timing_probe.py`, `m4b_staircase_ladder.py`,
`m4c_hybrid_ladder.py`, `m4d_arbiter_xslant.py`.
Pillar `[0.3, 0.6] x [0.3, 0.6]` (a quarter-period
square) in `px = py = 1.2 lam`, `eps` 4 / 1, `depth = 0.8 lam`, `n_sub = 1.5`,
`wl = 1`.  The pillar translates by `t . depth = 0.6` = HALF a period over the
layer -- a steep case, chosen because every staircase wall then lands on the
uniform `Nx = Ny = 8` grid.  Comparison is per order on `(0,0)`, `(+-1, 0)`,
`(0, +-1)`, `(+-1, +-1)`, both incident polarizations, on R and T.

**[M] (a) The prototype metric layer -- one solve, and it is GRID-INVARIANT.**
The same physical layer solved on two different union grids and two modal
counts:

| block | `Nx=4, M=5` | `Nx=4, M=6` | `Nx=8, M=3` | `Nx=8, M=4` (reference) |
|---|---|---|---|---|
| normal, diagonal slant -- self-move vs the reference | 1.82e-04 | 1.40e-04 | 5.23e-04 | -- |
| normal -- cost / closure | 8.1 s / 5.6e-08 | 28.4 s / 4.8e-10 | 7.0 s / 1.6e-05 | 98.2 s / 1.2e-07 |
| conical 20/35, x slant -- self-move | 4.43e-04 | 3.13e-04 | 5.38e-03 | -- |
| conical -- cost / closure | 8.8 s / 9.0e-06 | 38.6 s / 2.2e-07 | 9.2 s / 7.1e-04 | 133.9 s / 5.5e-06 |

The `Nx = 4` and `Nx = 8` grids describe the SAME pillar (the extra walls are
redundant), so the `1.4e-04 .. 4.4e-04` spread is the solver's own
discretization, not a slant artifact -- and it is what the reference row below
is worth.

**[M] (b) The shipped HYBRID slant metric -- an INDEPENDENT formulation, and the
sign, two-sided.**  `PMM2DStackHybrid.add_layer(slant=...)`: a Fourier basis, a
tensor fold, and a lab-Cartesian convection on its 4N generator -- nothing in
common with the prototype except the physics.  Distance to the prototype's
`Nx=8, M=4` reference:

| block | `t_hybrid = +t_probe` | `t_hybrid = -t_probe` |
|---|---|---|
| normal, diagonal, `n_orders = 5` | 1.811e-01 | **1.196e-02** |
| normal, diagonal, `n_orders = 7` | 1.761e-01 | **5.436e-03** |
| conical 20/35, x, `n_orders = 5` | 3.189e-01 | **5.270e-02** |
| conical 20/35, x, `n_orders = 7` | 3.190e-01 | **4.230e-02** |

**The hybrid's public `slant` is the NEGATIVE of the prototype's internal `t`**
(S6.1) -- and the wrong sign does not merely sit further away, it does not
IMPROVE with `n_orders` (1.811e-01 -> 1.761e-01, 3.189e-01 -> 3.190e-01) while
the right one does (1.196e-02 -> 5.436e-03).  That is the two-sidedness: a
truncation ladder separates a discretization gap from a wrong structure.

**[M] (c) A z-STAIRCASE from the PURE solver itself -- and the DIRECTION arm
that had to be added.**

The first pass built the staircase marching the pillar in `+x` while the
prototype's internal `t` marches it in `-x` (S6.1), and it did not converge:
`2.415e-01 / 1.649e-01 / 1.685e-01` at `n = 1 / 2 / 4` on the normal-diagonal
block, `2.599e-01 / 2.942e-01 / 3.070e-01` at conical -- i.e. WORSE than not
slanting at all past `n = 1`.  That is a probe-side error, not a formulation
one, and it is exactly what a two-sided scan is for, so the staircase's marching
direction is now a SCANNED arm (`m4b_staircase_ladder.py`; the first pass is
retained as `results/m4b_v1_wrongwalk.json`).  With the direction scanned, on
the M4b geometry (same pillar, `t = (0.75, 0)`, `Nx = Ny = 8`, `M = 3`,
reference = the metric layer on the SAME grid so the discretization is
common-mode):

| mount | direction | n=1 | n=2 | n=4 |
|---|---|---|---|---|
| normal | **WITH the slant** | 1.399e-01 | **4.670e-02** | **2.076e-02** |
| normal | AGAINST | 1.399e-01 | 2.064e-01 | 2.068e-01 |
| conical 20/35 | **WITH the slant** | 2.616e-01 | **2.055e-01** | **8.058e-02** |
| conical 20/35 | AGAINST | 2.616e-01 | 2.888e-01 | 3.016e-01 |

**The correct direction converges toward the metric layer** (3.0x then 2.2x at
normal; 3.2x over the ladder at conical), and the wrong one gets monotonically
worse -- which is the roadmap's A5 gate, passed, plus a third independent
pinning of the slant sign.  At `n = 1` both directions coincide, because the
`n = 1` "staircase" is just the VERTICAL pillar: `m4d` measures a vertical
pillar at **1.396e-01** from the slanted answer, which is the `n = 1` entry to
three digits.

**[M] The `M = 4` CONTROL** (on M4's own normal/diagonal block, wrong-direction
arm) rules out an `M = 3` artifact completely -- every rung reproduces its
`M = 3` value to four digits, at 5-10x the cost:

| n | `M = 3` | `M = 4` | `M = 3` cost | `M = 4` cost |
|---|---|---|---|---|
| 1 | 2.415e-01 | **2.415e-01** | 8.8 s | 104.8 s |
| 2 | 1.649e-01 | **1.649e-01** | 13.6 s | 146.2 s |
| 4 | 1.685e-01 | **1.685e-01** | 22.3 s | 213.3 s |

So the staircase is fully converged in the modal count; what moves it is the
geometry alone.

Two incidental confirmations fall out.  MIDPOINT and LEADING-EDGE sampling give
IDENTICAL efficiencies at every `n` where both are admissible (`2.415e-01` and
`1.649e-01` in both columns of M4) -- the two differ by a RIGID lateral
translation of the whole stack, and the pure staggered solver's position
invariance makes that a no-op.  And closure is excellent throughout
(`8.3e-05 .. 9.1e-04`) on BOTH direction arms, including the one that is
`2.1e-01` wrong: **the lossless trap, reproduced on this cascade.**  Energy is
reported here as context and is never a pass criterion.

**[M] Why the ladder stops at n = 4, and why that is the finding.**  `Basis1D`'s
segments are a `linspace`, so every slice must place its walls on ONE uniform
union grid of spacing `h = px / Nx`.  Two consequences:

1. the admissible slice counts are exactly the DIVISORS of `S / h` (`S` = the
   total lateral walk).  Here `S = px/2 = 4h`, so `n in {1, 2, 4}` and nothing
   else -- the ladder is fixed by the geometry and the grid, not chosen;
2. the smallest non-zero per-slice lateral STEP is `h` itself.  At `Nx = 8` that
   is `0.15`, i.e. **half the pillar width**.  A staircase of this pillar can
   therefore never have a step finer than half the feature until the union grid
   is refined -- and refining it costs `(Nx (M-1))^6` in the region eig.

`m4a_timing_probe.py` prices that: one region solve, `px = py = 1.2`, same cell.

| grid | in-plane `2q^2` (QZ) | slanted `4q^2` (whitened) |
|---|---|---|
| `Nx=4, M=5` (`q^2 = 256`) | 512, **3.51 s** | 1024, **4.78 s** |
| `Nx=4, M=6` (`q^2 = 400`) | 800, **13.55 s** | 1600, **13.74 s** |
| `Nx=8, M=3` (`q^2 = 256`) | 512, **3.34 s** | 1024, **3.76 s** |
| `Nx=8, M=4` (`q^2 = 576`) | 1152, **52.73 s** | 2304, **40.05 s** |

Note the last row: the SLANTED `4q^2` solve is *faster* than the vertical
in-plane `2q^2` one at the same `q`, because the slanted pencil is
Cholesky-whitened to a standard eig while the in-plane path pays a QZ.  Doubling
`Nx` from 4 to 8 at fixed `M` multiplies the region solve by ~8x per axis pair;
that is the price of every extra staircase rung.

**[M] (d) M4c -- does the hybrid WALK TOWARD the prototype as its truncation is
lifted?**  The hybrid has a Fourier floor the pure solver does not, so the
question that matters is direction.  Reference: the prototype metric layer at
`Nx = 4, M = 6` (its distance to the `Nx = 8, M = 4` rung is `1.4e-04` /
`3.1e-04`, two to three decades below the hybrid's own step, so it is a fixed
target on this scale).

| mount | n_orders | vs prototype | the hybrid's OWN step | cost |
|---|---|---|---|---|
| normal, diagonal | 5 | 1.205e-02 | -- | 0.4 s |
| normal, diagonal | 7 | 5.534e-03 | 6.631e-03 | 2.3 s |
| normal, diagonal | 9 | 7.228e-03 | 3.073e-03 | 8.5 s |
| normal, diagonal | 11 | **5.391e-03** | 4.426e-03 | 25.8 s |
| conical 20/35, x | 5 | 5.278e-02 | -- | 0.5 s |
| conical 20/35, x | 7 | 4.198e-02 | 1.568e-02 | 2.4 s |
| conical 20/35, x | 9 | 2.261e-02 | 2.770e-02 | 9.1 s |
| conical 20/35, x | 11 | **1.700e-02** | 5.606e-03 | 27.4 s |

At NORMAL incidence the gap to the prototype (`5.4e-03`) is the SAME SIZE as the
hybrid's own `n_orders` step (`4.4e-03`) -- i.e. the two engines agree to within
the hybrid's own convergence, which is the strongest statement this comparison
can make.  At CONICAL the hybrid is still walking: `5.28e-02 -> 1.70e-02`
monotonically as `n_orders` goes 5 -> 11, moving TOWARD the prototype by a
factor of 3.1, with its own step still `5.6e-03`.  Neither row shows the
prototype sitting outside where the hybrid is heading.

**[M] (e) M4d -- THE ARBITER, on the x-slant / normal-incidence cell where the
first-pass staircase plateaued.**  Same geometry as M4b, three questions at once:

*Is the prototype itself converged?*  Its self-move across FOUR grids and modal
counts, against `Nx = 8, M = 4`:

| `Nx=4, M=5` | `Nx=4, M=6` | `Nx=8, M=3` | `Nx=12, M=3` |
|---|---|---|---|
| 3.07e-04 | 2.10e-04 | 6.12e-04 | 2.70e-04 |

Grid- and degree-invariant to `~3e-04`, i.e. two to three decades below
everything it is being compared against.

*Does the INDEPENDENT hybrid walk toward it?*  Yes, monotonically:

| n_orders | vs prototype | the hybrid's OWN step | cost |
|---|---|---|---|
| 5 | 1.656e-02 | -- | 0.4 s |
| 7 | 1.541e-02 | 1.176e-02 | 2.4 s |
| 9 | 9.058e-03 | 1.203e-02 | 8.8 s |
| 11 | **5.891e-03** | 7.569e-03 | 26.9 s |

The gap closes 2.8x over the ladder and ends BELOW the hybrid's own step
(`5.9e-03` vs `7.6e-03`) -- the two engines agree to within the hybrid's own
convergence.  (The hybrid raises its own energy warning, `max R+T = 1.01`, at
`n_orders >= 9` on this steep cell; recorded, not used.)

*And what is the slant worth at all?*  A VERTICAL pillar sits **1.396e-01** from
the slanted answer -- so on this cell the slant is a `1.4e-01` effect, the
correct-direction staircase reaches `2.1e-02` at `n = 4`, and the two engines
that solve the slanted layer directly agree at `5.9e-03`.

**[M] (f) M4b -- the staircase's CONVERGENCE RATE, and what a rung costs.**
Same cell, `t = (0.75, 0)`, normal incidence, LEADING-EDGE sampling marching
WITH the slant, on two union grids.  Reference is the metric layer on the SAME
grid, so the discretization is common-mode and the number is the staircase's
geometric error alone.

| grid (`h`) | n | per-slice step | vs metric | cost vs the metric layer |
|---|---|---|---|---|
| `Nx=8` (`h = 0.150`) | 1 | 2.00 widths | 1.399e-01 | 0.83x |
| `Nx=8` | 2 | 1.00 | 4.670e-02 | 1.24x |
| `Nx=8` | 4 | 0.50 | **2.076e-02** | 2.44x |
| `Nx=12` (`h = 0.100`) | 1 | 2.00 | 1.397e-01 | 1.13x |
| `Nx=12` | 2 | 1.00 | 4.707e-02 | 1.44x |
| `Nx=12` | 3 | 0.67 | 2.716e-02 | 1.82x |
| `Nx=12` | 6 | 0.33 | **1.478e-02** | 3.22x |

Two things this settles.  First, **the error is set by the per-slice STEP, not
by the union grid**: at a matched step of 1.00 widths the two grids read
`4.670e-02` and `4.707e-02`, and the ladder falls as `~1/n` on both.  Second,
**the union grid is what caps the ladder**: a step of `1/3` of the pillar width
needs `Nx = 12` and 6 slices, costs `3.22x` one slanted solve, and still sits
`1.5e-02` away -- while the INDEPENDENT hybrid sits `5.9e-03` away (S4.4(e)).
Halving the step again means `n = 12` on `Nx = 24`, i.e. a region eig `(24/12)^6
= 64x` larger per layer, twelve times over.

**That is the honest form of the speed win.**  Not "one solve replaces N
slices" at a fixed price, but: the staircase's cost to reach a given accuracy
grows with BOTH the slice count and the union grid it forces, and in the pure
staggered solver the two are locked together.

### 4.5  M5 -- SPURIOUS CENSUS, CASCADE STABILITY, and the NO-FLOOR property

`m5_census_cascade.py`.  The 1-D convection route's known cost is an
"advection spurious" sea whose `|Re q|` grows with slant, and the roadmap's
V-partner wall showed up as an energy blow-up.  Both are census questions, and
both are asked here against the SAME cell at slant 0 -- the shipped
out-of-plane generator's own census is the control.

**[M] T5a -- CENSUS.**  `px = py = 1.2 lam`, conical 20/35, `M = 6`, `(2,2)`
grid, `dim = 4q^2 = 400`.  "above-band" counts modes with `|q| > 3 sqrt(max eps)`
-- the polynomial basis's own unresolved harmonics, which exist at slant 0 too.

| cell | slant | split (want 200/200) | `min Re(lam_f)` | `max abs(q)` | above-band |
|---|---|---|---|---|---|
| scalar pillar `eps` 4 | vertical | **200/200** | -5.3e-15 | 9.63 | 98 (24.5%) |
| scalar pillar | x 20 deg | **200/200** | -3.6e-15 | 9.73 | 102 (25.5%) |
| scalar pillar | x 45 deg | **200/200** | -6.5e-15 | 11.23 | 128 (32.0%) |
| scalar pillar | diag 45 deg | **200/200** | -5.2e-15 | 13.24 | 160 (40.0%) |
| scalar pillar | x 60 deg | **200/200** | -2.4e-15 | 14.20 | 186 (46.5%) |
| out-of-plane uniaxial pillar | vertical | **200/200** | -9.0e-15 | 9.56 | 144 (36.0%) |
| out-of-plane uniaxial pillar | x 60 deg | **200/200** | -3.3e-15 | 14.10 | 222 (55.5%) |
| high contrast `eps` 12 | vertical | **200/200** | -3.7e-15 | 9.56 | 0 |
| high contrast `eps` 12 | x 60 deg | **200/200** | -6.7e-15 | 13.41 | 34 (8.5%) |
| LOSSY pillar `4 + 0.6i` | vertical | **200/200** | **+4.9e-02** | 9.63 | 96 (24.0%) |
| LOSSY pillar | x 60 deg | **200/200** | **+3.3e-02** | 14.22 | 185 (46.2%) |

Three readings:

* the forward/backward split is **exactly `2q^2 / 2q^2` in all 20 rows**, BEFORE
  the selector's defensive rebalance, at slants up to 60 degrees, on a high-
  contrast cell and on a lossy one -- the contract `_region_modes_oop` pins;
* `min Re(lam_f)` is `>= -2.3e-14` on every lossless row (no growing mode
  classified forward) and strictly POSITIVE on every lossy row (forward modes
  decay), which is the sign a broken magnetic partner would break;
* `max abs(q)` grows from `9.6` to `14.2` between slant 0 and 60 degrees --
  a factor of `1.48`, i.e. `sec(60 deg) = 2` bounded, and NOT the `~210` a
  from-scratch convection form produces in 1-D.  The above-band population is
  the polynomial basis's own unresolved harmonics (24.5% ALREADY at slant 0),
  growing with the same `sec` factor; the flux selector absorbs them, which is
  what the exact split says.

**[M] T5b -- CASCADE vs DEPTH.**  `M = 5`, depths 0.25 / 1 / 3 wavelengths;
`max fwd growth` is `max exp(-Re(lam_f) k0 L)` at 3 wavelengths (any value above
1 means a growing mode was classified forward).

| cell | slant | mount | 0.25 lam | 1 lam | 3 lam | max fwd growth |
|---|---|---|---|---|---|---|
| scalar pillar | vertical | normal | 1.70e-04 | 2.07e-04 | 2.83e-04 | **1.0000e+00** |
| scalar pillar | vertical | conical 20/35 | 1.63e-03 | 7.54e-03 | 1.48e-02 | **1.0000e+00** |
| scalar pillar | x 36.9 deg | conical 20/35 | 2.12e-03 | 3.92e-03 | 4.53e-03 | **1.0000e+00** |
| scalar pillar | diag 45 deg | normal | 4.05e-05 | 2.12e-04 | 7.59e-05 | **1.0000e+00** |
| scalar pillar | diag 45 deg | conical 20/35 | 1.44e-03 | 8.40e-04 | 5.76e-04 | **1.0000e+00** |
| out-of-plane pillar | x 36.9 deg | normal | 6.03e-05 | 1.73e-04 | 3.87e-04 | **1.0000e+00** |
| out-of-plane pillar | diag 45 deg | conical 20/35 | 1.82e-04 | 6.89e-04 | 7.35e-04 | **1.0000e+00** |
| LOSSY pillar | x 36.9 deg | normal | 0.134 | 0.639 | 0.937 | 3.18e-01 |
| LOSSY pillar | diag 45 deg | conical 20/35 | 0.167 | 0.589 | 0.869 | 4.79e-01 |

Forward growth is **exactly `1.0000e+00`** on every lossless row and strictly
below 1 on every lossy one.  Closure does NOT grow with depth on the slanted
rows -- and note the sharpest comparison in the table: the VERTICAL cell at
conical incidence degrades `1.63e-03 -> 1.48e-02` over the depth ladder while
the SLANTED cell on the same geometry improves, `2.12e-03 -> 4.53e-03`.  The
lossy rows' absorption rises monotonically and stays in `[0, 1]`.

**[M] T5c -- LAYER SPLIT.**  One slanted layer of depth `d` versus two stacked
slanted layers of `d/2` at the same slant (the same cell in both, which is the
correct construction: the identity interface match carries the frame offset,
S1.5):

| cell | slant | mount | dR | dT | dJones |
|---|---|---|---|---|---|
| scalar pillar | x 36.9 deg | normal | 2.57e-16 | 5.55e-16 | 6.02e-16 |
| scalar pillar | diag 45 deg | conical 20/35 | 1.95e-16 | 5.55e-16 | 1.26e-15 |
| out-of-plane pillar | x 36.9 deg | conical 20/35 | 5.90e-17 | 9.99e-16 | 6.32e-16 |
| out-of-plane pillar | diag 45 deg | normal | 1.46e-16 | 4.72e-16 | 7.08e-16 |

Worst over all 8 rows: **1.28e-15**.  Machine precision -- so the propagator,
the internal interface and the frame anchor compose exactly, which is the
strongest single check on S1.5's interface argument.

**[M] T5d -- the NO-FLOOR property survives the slant.**  Movement of the
per-order result and the reflection Jones when the far-field order half-width
goes `3 -> 5 -> 8`:

| cell | slant | mount | move 3->5 | move 3->8 |
|---|---|---|---|---|
| scalar pillar | x 36.9 deg | normal | 2.78e-16 | 3.89e-16 |
| scalar pillar | diag 45 deg | normal | 6.11e-16 | 1.17e-15 |
| out-of-plane pillar | x 36.9 deg | normal | 8.88e-16 | 1.94e-16 |
| scalar pillar | x 36.9 deg | conical 20/35 | 1.81e-06 | 2.45e-06 |
| scalar pillar | diag 45 deg | conical 20/35 | 3.33e-06 | 4.87e-06 |
| out-of-plane pillar | diag 45 deg | conical 20/35 | 2.29e-06 | 2.58e-06 |

Machine precision at normal incidence; `~3e-06` at conical.  **[H]** the conical
residue is very likely the `_grazing_safe_wavelength` nudge, which is a function
of the ORDER SET and so changes slightly with `n_orders` -- it is absent at
normal incidence, where no order is near a cutoff.  Either way it is three
decades below the Fourier hybrid's own `E_z`-rule spread (`7.7e-04`), so the
no-floor property is intact.

**M5 PASSES on every arm.**

### 4.6  M6 -- COST

`m6_cost.py` (plus `m4a_timing_probe.py`, S4.4).

**[M] T6a -- per-region assembly + eig**, `Nx = Ny = 2`, conical Bloch shift,
median of 2-5 repeats:

| M | `q^2` | in-plane `2q^2` (QZ) | vertical OOP `4q^2` | SLANTED `4q^2` | slant / OOP | slant / in-plane |
|---|---|---|---|---|---|---|
| 4 | 36 | 8.9 ms | 31.6 ms | 27.3 ms | **0.86x** | 3.15x |
| 5 | 64 | 41.4 ms | 92.1 ms | 92.8 ms | **1.01x** | 2.01x |
| 6 | 100 | 126.0 ms | 242.1 ms | 227.9 ms | **0.94x** | 1.80x |
| 7 | 144 | 339.6 ms | 690.6 ms | 742.2 ms | **1.07x** | 2.18x |

**The slant itself is FREE.**  Against the vertical out-of-plane path -- which
is where a slanted cell has to live anyway, because its covariant tensor has
out-of-plane entries -- the congruence and the six blocks cost `0.86x .. 1.07x`.
The `1.8x .. 3.2x` against the in-plane `2q^2` path is the price of needing the
first-order generator at all, and it is the SHIPPED Stage-B number (`1.33-2.03x`
measured there), not something the slant adds.

And at larger `q` the slanted solve is outright FASTER than the vertical
in-plane one, because it is a Cholesky-whitened standard eig while the in-plane
path pays a QZ -- `m4a` at `Nx = 8, M = 4`: in-plane `2q^2 = 1152` takes
**52.73 s**, slanted `4q^2 = 2304` takes **40.05 s**.

**[M] T6b -- end-to-end single-layer solve** (scalar pillar, conical 20/35,
`n_orders = 3`), slanted vs vertical:

| M | vertical | slanted | ratio |
|---|---|---|---|
| 5 | 0.128 s | 0.181 s | **1.41x** |
| 6 | 0.327 s | 0.448 s | **1.37x** |
| 7 | 0.918 s | 1.280 s | **1.39x** |

**[M] T6c -- the staircase arithmetic.**  See S4.4: on the M4b geometry the
staircase's ladder is fixed by the union grid, its per-slice lateral step cannot
go below one grid cell, and a rung costs `0.8x .. 2.9x` a slanted solve.  The
practical statement is not "one slanted solve replaces N slices" but the
stronger one: **at the slice counts a given union grid admits, there may be no
rung that reaches the slanted answer at all** -- refining the step means
refining `Nx`, which multiplies the region eig by `(Nx (M-1))^6`.

### 4.7  M7 -- SLANT x ANISOTROPY, and a coverage gap this CLOSES

`m7_slant_x_aniso.py`, `m7b_hybrid_oop_slant.py`, `m7c_hybrid_stripe_anomaly.py`.
The covariant congruence `eps -> A^-1 eps A^-T` is tensor-agnostic, so slant x
anisotropy is not a separate feature in this formulation -- it is the same line
of code.  That matters because the shipped 2-D engines do not cover it.

**[M] T7a/T7b -- the COVERAGE GAP is real, and it raises at SOLVE, not at
`add_layer`.**  `PMM2DStackHybrid.add_layer(eps_tensor_cell=..., slant=...)`
ACCEPTS an out-of-plane tensor; the refusal comes later, from

```
NotImplementedError: _layer_eigenmodes_tensor: a SLANTED layer with OUT-OF-PLANE
coupling (eps_xz/yz/zx/zy) is not supported.  The 2-D slant metric is validated
for IN-PLANE tensors only ...
```

So the restriction is genuine (the method's docstring is right about the
outcome) but is enforced one call later than the docstring reads -- worth a
line in the build's own validation.  **No 2-D engine in the suite covers slant x
out-of-plane today.**

*(An incidental note, resolved: `m7b`'s in-plane arm reported a `TypeError` --
that was the PROBE's own result unpacking, not the library.  `m7c` drives all
four `{y-uniform stripe, pillar} x {slant, no slant}` in-plane tensor
configurations through `PMM2DStackHybrid.solve()` and all four succeed.)*

**[M] T7b -- the prototype DOES cover it, validated against the 1-D engine that
also does.**  A y-uniform SLANTED OUT-OF-PLANE stripe (ridge =
`uniaxial(1.5, 1.7, tilt 35 deg, azim 25 deg)`, groove = air, `px = 0.75 lam`,
`depth = 0.30 lam`, duty 0.5) against `pmm_jones_1d_slanted` at degree 30,
per order over `m = -2..2`, both incident polarizations:

| phi | mount | M | incident Ex | incident Ey | oracle's own drift | y-leak | closure |
|---|---|---|---|---|---|---|---|
| 0 | normal | 8 | 2.68e-05 | 1.23e-06 | 3.1e-06 | **0.0** | 4.97e-10 |
| 20 | normal | 8 | 2.71e-05 | 1.75e-06 | 5.5e-06 | **0.0** | 9.11e-10 |
| 35 | normal | 8 | **1.66e-05** | 1.96e-06 | 6.8e-06 | **0.0** | 2.30e-09 |
| 0 | oblique 25 | 8 | 1.91e-05 | 9.35e-07 | 3.3e-06 | **0.0** | 4.10e-11 |
| 20 | oblique 25 | 8 | 2.75e-05 | 3.79e-07 | 1.4e-05 | **0.0** | 2.04e-11 |
| 35 | oblique 25 | 8 | **2.76e-05** | 6.93e-07 | 2.1e-05 | **0.0** | 8.56e-11 |

The slanted rows are as good as the vertical (`phi = 0`) row -- in fact slightly
better at 35 degrees normal -- the `Ey` channel is AT the oracle's own drift,
y-momentum is conserved EXACTLY (`0.0`, not merely small), and closure reaches
`1e-10`.  `m7b` reproduces the `phi = 35`, oblique-25 row independently at
`Ex 2.76e-05 / Ey 6.93e-07`.

**[M] T7c -- a genuinely 2-D slanted IN-PLANE tensor pillar vs the hybrid.**
`px = py = 1.2 lam`, quarter-period pillar of `uniaxial(1.5, 1.7, tilt 90 deg,
azim 25 deg)`, `depth = 0.8 lam`, `t = 0.75`:

| mount | prototype self-move M5->M6 | hybrid `+t`, n=5 / n=7 | hybrid `-t`, n=5 / n=7 |
|---|---|---|---|
| normal | 1.39e-04 | 4.78e-02 / 4.48e-02 | **1.04e-02 / 4.24e-03** |
| conical 20/35 | 9.56e-05 | 5.40e-02 / 4.85e-02 | **7.84e-03 / 3.57e-03** |

The sign is pinned a third time, on a TENSOR cell; the right arm halves with the
hybrid's truncation while the wrong arm does not move; and the prototype's own
`(M)` self-move is two decades below the gap, so the residual is the hybrid's
Fourier floor.  **M7 PASSES**, and the slant x out-of-plane combination becomes
available for the first time in a 2-D engine.

---

## 5.  What this does NOT cover

* **TAPER.**  A shear is a tilted axis with a CONSTANT cross-section.  No shear
  absorbs a dilation -- measured at `1.00x` in
  `BUILD_PMM2D_SLANT_METRIC_2026_08_16` M5, on the one-wall taper carrying the
  maximum shear content a taper can have.  The taper's `sqrt(g)` is
  z-dependent, which brings back the dilation generator, a non-normal
  `q -> -conj(q)` pencil with no valid mode selector, and a distorted far field.
  None of that arises here, and none of it is solved here.
* **Mixed slants between PATTERNED layers** (S1.5): the frame offset between two
  differently-sheared patterned regions is a real lateral translation of one
  nodal grid relative to the other.  Exact only when the offset is a whole
  number of grid cells; otherwise it needs an interpolation the pure basis does
  not have.  Homogeneous / uniform neighbours are unaffected (the offset is a
  gauge).  **The build must refuse this**, loudly.
* **Curved walls** -- Phase E, unchanged by this work.
* **`retain_internal` / `layer_absorption` below a slanted layer** were not
  measured.  The frame-anchor phase is a per-order far-field correction here;
  an internal-field probe evaluated at a plane inside or below a sheared layer
  is in the frame, not the lab, and needs the same treatment.  The first build
  should either carry it through `_flux_at` or refuse `retain_internal` on a
  slanted stack.

---

## 6.  THE INTEGRATION ROUTE

Everything below is a change to two files.  No new module, no new basis
function, no new cascade, no new far field.

### 6.1  `lumenairy/elements/pmm/twod_staggered.py`

**(1) `Granet2DTransverseE.__init__(..., slant=(0.0, 0.0))`.**

```
  # a sheared cell is an OUT-OF-PLANE cell in the frame, always
  if slant != (0.0, 0.0):
      cell33 = promote_scalar_to_tensor(eps_cell)       # (Nx,Ny) -> (Nx,Ny,3,3)
      rot    = _OOP_ROT_SIGN
      cell33 = flip_offplane(cell33, rot)               # e13,e23,e31,e32 *= rot
      tx, ty = rot * slant[0], rot * slant[1]           # t rotates WITH them
      self.eps_cell = cov_congruence(cell33, tx, ty)    # A^-1 eps A^-T
      self.offplane = True
  else:  # unchanged dispatch, byte for byte
```

The rot-on-`t` is not cosmetic: `eps^{lm}(R eps R, -t) = R eps^{lm}(eps, t) R`
is what makes the two consistent (S1.4(c)), and M2's `a-` arms measure the
failure of getting it wrong (`1.3e-02 .. 1.3e-01`).

**(2) `_assemble_oop` gains one guarded block.**  Six `np.kron`s and four
row additions, exactly as in `slant_lib.SlantSolver._assemble_slant`:

| new per-axis matrix | call |
|---|---|
| `Mtb_x`, `Mtb_y` | `b.mass(b.Btilde, b.B)` |
| `Mbt_x`, `Mbt_y` | `b.mass(b.B, b.Btilde)` |
| `dtb_x`, `dtb_y` | `-(dbt).conj().T` -- the integration-by-parts identity the assembly already uses for `CwE1`/`CwE2`; the ELEMENTWISE `b.mixed(b.Btilde, b.B)` would silently drop the jump deltas and is NOT the same matrix |
| `Ctt_x`, `Ctt_y` | already in `_axis_mats` |

`if tx == 0 and ty == 0` skips the whole block, which is what makes the slant-0
path bit-identical (G0a: 0 differing bytes).

**(3) `_region_modes_oop`: UNCHANGED.**  The retained state is the covariant
tangential state = the lab-Cartesian tangential state, so the Cholesky
whitening, the flux-based split with the deep-decay override, `_OOP_H_GAUGE`,
and the `2q^2 / 2q^2` contract all keep their meanings.  M5 measures each.

**(4) `pmm_efficiency_2d_staggered` / `pmm_jones_2d_staggered`: add `slant=` and
forward.**  Match `PMM2DStackHybrid.add_layer`'s signature exactly
(`(t_x, t_y)` or a bare scalar `t_x`, a TANGENT, cross-section at the layer's
TOP) so a caller can move a cell between the two 2-D engines without a
convention change.

**THE SIGN, measured twice and stated as a relation.**  The prototype's internal
`t` (the `t` of `x = u + t w` in S1.1) satisfies

```
    t_probe   =  - t_hybrid_public        (M4, 5.4e-03 vs 1.8e-01 / 3.2e-01)
    t_probe   =  - tan(slant_angle_1D)    (M3, 1.2e-03 vs 4.2e-01)
    => t_hybrid_public = tan(slant_angle_1D)     -- the hybrid and the 1-D
       scalar entry already share ONE public convention.
```

So the build takes the PUBLIC `slant` and passes `-slant` into the congruence
and the six blocks, and the three engines then agree.  Which way a positive
`slant` physically tilts the wall is a library convention this campaign did not
independently re-derive -- M3 and M4 pin the RELATION, which is what an
implementation needs, and each pins it against an engine validated elsewhere.

### 6.2  `lumenairy/elements/pmm/stack2d_pure.py`

**(5) `PMM2DStackPure.add_layer(..., slant=None)`.**  A slanted layer sets
`any_oop = True` (already the flag that promotes the stack to the GENERALIZED
cascade -- no cascade work at all), and the `eig_cache` key must include the
slant.

**(6) THE FRAME-ANCHOR PHASE -- the one genuinely new line of physics.**
In `solve`, after `kxv`/`kyv` are built:

```
  shx = sum(L["slant"][0] * L["thickness"] for L in self._layers)
  shy = sum(L["slant"][1] * L["thickness"] for L in self._layers)
  tphase = exp(-1j * k0 * (kxv * shx + kyv * shy))
  tx_ord *= tphase ;  ty_ord *= tphase          # TRANSMITTED only
```

`S11` and the reflection Jones need NOTHING (S1.5).  Omitting it leaves R, T and
the reflection Jones exactly right and the transmission Jones wrong by up to
`6.2e-01` -- a silent-wrong of the worst class, so its gate (M1b, both signs on
a uniform null at oblique) is not optional.

**(7) REFUSALS, all raising with the reason:**
* a slanted PATTERNED layer whose neighbour is a PATTERNED layer with a
  DIFFERENT slant (a VERTICAL patterned layer included) -- S1.5;
* dispersive (callable) and traced (JAX) layers -- mirror the hybrid;
* `retain_internal` on a slanted stack, unless `_flux_at` is taught the frame
  (S5).

**(8) [H] The Wood-anomaly nudge list.**  `solve` feeds `_grazing_safe_wavelength`
the tensor layers' diagonals.  A slanted SCALAR layer's covariant diagonal is
`eps (1 + t_x^2)` etc., so its modal spectrum reaches higher than `eps`; whether
it should be added to `_eps_gr` is NOT settled by this campaign and the build
should measure it (a slanted cell walked onto a Rayleigh cutoff).

### 6.3  Gates for the build (every one of them measured here)

| gate | assertion | measured here |
|---|---|---|
| B1 | `slant = 0` is BYTE-IDENTICAL to the pre-slant library (pencil AND end-to-end) | G0a / G0b |
| B2 | uniform layer + any slant is a no-op, M-ladder to machine precision | M1 / M1c |
| B3 | the frame-anchor phase, BOTH signs, on the transmission Jones at oblique | M1b |
| B4 | sheared-frame dispersion == `exact_kz_roots + t.alpha`, PLUS the two ablations (drop the six blocks / drop the congruence) | M2 |
| B5 | y-uniform stripe == `pmm_efficiency_1d_slanted` per order, both pols, normal + oblique, slant 10/20/35 -- and the SIGN arm | M3 |
| B6 | slant x OUT-OF-PLANE stripe == `pmm_jones_1d_slanted` per order | M7 |
| B7 | census: split exactly `2q^2/2q^2`, `min Re(lam_f) >= 0`, no band-exceeding mode carrying flux | M5 |
| B8 | closure does not grow with depth (0.25 / 1 / 3 lam) and `max fwd growth == 1.0` | M5 |
| B9 | one layer of depth `d` == two of `d/2` at the same slant | M5 |
| B10 | NO-FLOOR survives: the answer does not move with `n_orders` | M5 |
| B11 | the mixed-slant refusal fires | (new) |

B4 and B5 are the ones that cannot be replaced by an energy check: the lossless
trap is the named hazard for slant work, and M4's own numbers show a staircase
that conserves energy while sitting decades from the truth.

### 6.4  Cost of the build

| piece | size | risk |
|---|---|---|
| the congruence + the six blocks in `_assemble_oop` | ~40 lines | LOW -- every bracket already exists; `slant = 0` is bit-identical, which is a byte-level regression gate |
| `slant=` through `add_layer` / the two entries + cache key | ~30 lines | LOW -- `any_oop` already promotes the cascade |
| the frame-anchor phase in `solve` | ~6 lines | MEDIUM -- silent-wrong if omitted or mis-signed; gate B3 is mandatory |
| the refusals | ~25 lines | LOW |
| gates B1-B11 as tests | the eleven above | MEDIUM -- B4/B5/B6 need the oracles this probe already drives |

No new module, no new basis function, no cascade change, no far-field change.

---

## 7.  Follow-ups this campaign did NOT do

* `docs/PMM_ROADMAP.md` Phase D is left untouched (parallel agents are working
  in other worktrees); the build should update Section 1's capability matrix
  row "2-D slanted" and Section 4's Phase D entry, and record that the
  "3.4x vs FMM" figure is an S-matrix-size ratio (S1.0), not a wall clock.
* The Wood-anomaly `_eps_gr` question (S6.2 item 8).
* `retain_internal` / `layer_absorption` under a slanted layer (S5).
* A second BUILD for the numbers here, per `TESTING_STANDARDS.md` rule 5,
  before any of them becomes a test bar.
