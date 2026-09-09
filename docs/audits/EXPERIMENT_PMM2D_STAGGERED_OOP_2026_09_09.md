# Out-of-plane anisotropy in the PURE staggered 2-D PMM -- PROTOTYPE VERDICT, 2026-09-09

Branch `probe/pmm2d-staggered-oop` off `main` f70628d (lumenairy 5.42.1 + the
three BOR SEM commits).  Stage B of
`docs/audits/PLAN_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md` Section 4.
**No library code is changed by this branch.**  Everything lives in
`validation/probe_pmm2d_staggered_oop/` (scripts + a `README.md` with the exact
commands) and in the tables below; every number in this document is printed by
one of those scripts.

**Mount.**  Windows 11, python 3.14.6, numpy 2.4.4, scipy 1.17.1,
scipy-openblas, tesla-ryzen (24 cores / 128 GB).  Every probe exported
`OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = MKL_NUM_THREADS = 1` **before**
python started and asserted `lumenairy.__file__` --
`C:\tmp\lum_aniso_oop\lumenairy\__init__.py`, version 5.42.1.

---

## VERDICT

PLACEHOLDER-VERDICT

---

## S0.  Conventions, and the one sign the shipped module cannot see

The shipped `twod_staggered.py` is written end to end in the PUBLIC
`exp(-i w t)` gauge and needs no conjugation bridge.  It also happens to be
BLIND to the sign of its own transverse derivative, and that blindness ends
the moment an out-of-plane tensor entry appears.  This has to be settled
before anything is derived, because it is exactly the class of defect that
`AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14` documents across six copies.

**Where the blindness comes from.**  In the E-formulation every operator that
carries a transverse derivative carries an EVEN number of them:
`S_tt = -Curl^H Gw^-1 Curl` (two), and `K_tz eps33^-1 K_zt` (one on each
side).  Flipping `d/dx -> -d/dx` therefore leaves `Lmat` invariant, so the
module's choice `tau = exp(-i alpha0 p)` -- a basis whose plane-wave content
runs as `exp(-i alpha0 x)` while its longitudinal factor is
`exp(+i gamma z)` -- is unobservable there.  It is NOT unobservable in an
out-of-plane generator, where `eps_xz` multiplies ONE derivative against ONE
field, and where the physical discriminator (the asymmetric extraordinary
pair) is precisely odd in `k_t`.

**What the basis actually carries (measured, M0.1).**  `Basis1D`'s periodic
hat puts `Ltilde_1` on segment 0 and `tau * Ltilde_2` on segment `N-1`, i.e.
`f(d) = tau f(0)`.  Read off directly: `hat(0) = 1`,
`hat(d)/hat(0) = tau` to `0.0e+00`; and the lowest eigenfunction of the 1-D
Galerkin Laplacian on that basis has measured phase slope `+1.234000` against
an exact `K = +1.234000` (error `4.4e-13`), with `mu_0 = 1.52276` against
`K^2 = 1.52276` (error `2.2e-13`).  So the basis carries `exp(+i K x)` with
`tau = exp(+i K d)`.

**The probe's choice.**  Pass `tau = exp(+i k0 kx0 px)`, so the m-th harmonic
is the PHYSICAL `exp(+i(k0 kx0 + m G) x)` and `D_1 -> +i kx/k0` throughout.
The far-field kernel is then `exp(-i(K + mG)x)` (the probe writes its own;
the shipped `_stag_fourier_projection` uses the opposite sign consistently
with the opposite `tau`).  A library integration may keep the shipped `tau`
provided it flips the sign of every SINGLE-derivative operator in the new
blocks -- but the honest recommendation is to flip `tau` and the projection
kernel together and remove the trap.

**Normalized Maxwell.**  With `D_j = (1/k0) d/dx_j`, `D_3 -> i q`,
`q = gamma / k0`, and the normalized magnetic field `G = i Z0 H` (the same
state the shipped 1-D `_build_generator_metric` uses -- `[Ex; Ey; iZHx; iZHy]`
-- and the same one `rcwa._core._select_forward_flux` reads,
`Hx = Vfull[2N:3N]/1j`), Maxwell becomes REAL-coefficient:

```
    D x E = G ,        D x G = eps E .
```

Componentwise, with `E3`/`G3` still present:

```
 (1)  D2 E3 - i q E2 = G1              (4)  D2 G3 - i q G2 = (eps E)_1
 (2)  i q E1 - D1 E3 = G2              (5)  i q G1 - D1 G3 = (eps E)_2
 (3)  D1 E2 - D2 E1 = G3               (6)  D1 G2 - D2 G1 = (eps E)_3
```

---

## S1.  The discrete de Rham structure the staggered basis already has

Granet's Eq. 34 places `E1 in V1 = B(x)(x)Btil(y)`,
`E2 in V2 = Btil(x)(x)B(y)`, `E3 in V3 = Btil(x)(x)Btil(y)`; the shipped
`_assemble` adds `Vw = B(x)(x)B(y)` as the curl's own space.  Writing
`H1 in V2`, `H2 in V1`, `H3 in Vw` (which is what `_region_modes`'s Eq.-25
partner already does -- its `V` holds V2-space coefficients on top and
V1-space on the bottom), the four spaces form a MATCHED PAIR of complexes:

```
   primal (E-type):    V3  --grad-->  (V1, V2)  --curl-->  Vw
   dual   (G-type):    Vw  <--        (G1,G2) = (V2, V1)  --div-->  V3
```

Both arrows of the primal complex are EXACT in this basis, because
`d/dx Btil` lies in `span(B)`:

* `D1 V3 = (dBtil)(x) (x) Btil(y) in B(x)Btil = V1`, `D2 V3 in V2`
  -- the gradient lands strongly in `[V1; V2]` (this is the shipped `K_tz`);
* `D2 E1 = B (x) (dBtil) in Vw` and `D1 E2 in Vw`
  -- the curl lands strongly in `Vw` (this is the shipped `Curl`, and the
  module's own comment says testing it in `Vw` rather than `V3` is what makes
  the discretization spurious-free).

The dual complex is exact in the same way, with the roles swapped:
`div(G1, G2) = D1 G1 + D2 G2` lands strongly in `Vw`, while
`curl(G) = D1 G2 - D2 G1` is WEAK and is tested in `V3` with the derivative
moved onto the (continuous, Bloch-periodic) `V3` test function -- which is
exactly the `_eps_dir(..., "dL", ...)` device the shipped `K_zt` uses.

**Consequence.**  Equations (1), (2), (3) are STRONG (their right-hand sides
land exactly in the space the unknown lives in), and (4), (5), (6) are WEAK
with one integration by parts each onto a continuous test function.  Nothing
about that placement is new -- it is the shipped isotropic structure read as a
first-order system.  Both candidates below respect it.

Notation for the assembled blocks (all built by
`probe_common.StaggeredCell`, `k0`-normalized where a derivative appears):

| block | definition |
|---|---|
| `M1, M2, M3, Mw` | Grams of `V1, V2, V3, Vw` |
| `CwE1 = <Vw\|D2\|V1>`, `CwE2 = <Vw\|D1\|V2>` | the strong curl (shipped `Cw_E1`, `Cw_E2`) |
| `P13 = <V1\|D1\|V3>`, `P23 = <V2\|D2\|V3>` | the strong mimetic gradient (shipped `-Ktz`) |
| `Aab = <Va\| e_ab \|Vb>`, a,b in 1..3 | eps-weighted component masses (Appendix-A Eq. 40/41, plus the FOUR NEW out-of-plane blocks `A13, A23, A31, A32`) |
| `Kab = <D_a V3\| e_ab \|Vb>` | the eps-weighted div-D blocks with the derivative on the V3 TEST (Appendix-A Eq. 44, plus the NEW third column `K13, K23`) |
| `Sxx = CwE1^H Mw^-1 CwE1`, `Syy`, `Sxy`, `Syx` | the curl-curl, `= -Stt` blockwise |

---

## S2.  Candidate (a) -- the first-order staggered generator, `4 q^2`

Eliminate `G3` STRONGLY from (3) and `E3` WEAKLY from (6) (tested in `V3`,
derivative on the test), leaving the state `x = [e1; e2; g1; g2]`:

```
    g3 = Mw^-1 ( CwE2 e2 - CwE1 e1 )
    e3 = A33^-1 ( P23^H g1 - P13^H g2 - A31 e1 - A32 e2 )
```

`A33` is invertible exactly when `e33 != 0` -- the same
`_require_nonzero_ezz` precondition the hybrid already documents.  The four
remaining rows, each tested in its own space, give the pencil
`A x = q B x` with `B = blkdiag(M1, M2, M2, M1)`:

```
  (II)  q M1 e1 = -i M1 g2 - i P13 e3
  (I)   q M2 e2 = +i M2 g1 - i P23 e3
  (V)   q M2 g1 = -i [ A21 e1 + A22 e2 + A23 e3 - CwE2^H g3 ]
  (IV)  q M1 g2 = +i [ A11 e1 + A12 e2 + A13 e3 + CwE1^H g3 ]
```

Substituting `e3` and `g3` produces the block structure of Li 2003 /
`rcwa._core._layer_eigenmodes_tensor`'s `G = [[A, P], [Q, B]]`, transplanted
into the staggered spaces:

* `e13, e23` (the `A13`, `A23` masses) feed the `G`-rows -- the RCWA `B` block;
* `e31, e32` (the `A31`, `A32` masses, inside `e3`) feed the `E`-rows through
  `P13, P23` -- the RCWA `A` block;
* the `e33`-Schur is POINTWISE per cell (it is `A33^-1` applied to the
  eps-weighted masses), never a product of separately discretized factors --
  the ordering the 1-D `gen2` prototype got wrong (`_build_generator_metric`'s
  own note).

**Where the factors of `i` are.**  Every `i` above comes from `D_3 = i q` and
nothing else; the transverse derivative operators carry no explicit `i` (they
are `<.|D|.>` Galerkin blocks, and the basis supplies `exp(+i k x)`).  This
is why the probe's sign convention (S0) is load-bearing: with the shipped
`tau` sign the same expressions would need the OOP blocks negated.  M1 T3
measures that: negating the OOP entries in the assembly moves the dispersion
by 4.7e-2 while the reference sits at 1e-14.

**Reduction to the shipped E-form.**  With `e13 = e23 = e31 = e32 = 0`,
`e3 = A33^-1(P23^H g1 - P13^H g2)`, which is the div(D)=0 slaving written
through the curl-H row.  M3 measures whether that reproduces the shipped
`Lmat` spectrum.

---

## S3.  Candidate (d) -- the quadratic E-form that KEEPS div(D) = 0, `6 q^2`

Take the curl-curl form `D x (D x E) = eps E` for the two TRANSVERSE
components, and use `div(D) = 0` for the third equation.  This is legitimate
for `q != 0`: `D . (D x D x E) == 0`, so `D . r = -D . (eps E) = 0` whenever
the constraint holds; with `r1 = r2 = 0` that leaves `i q r3 = 0`.

```
 (H1)  q^2 E1 - D2^2 E1 + D1 D2 E2 + i q D1 E3 = (eps E)_1
 (H2)  q^2 E2 - D1^2 E2 + D1 D2 E1 + i q D2 E3 = (eps E)_2
 (*)   i q [e31 E1 + e32 E2 + e33 E3] + D1 (eps E)_1 + D2 (eps E)_2 = 0
```

Discretized with (H1) tested in `V1`, (H2) in `V2`, (*) in `V3` with the
derivative on the test, this is the quadratic pencil
`q^2 P2 x + q P1 x + P0 x = 0` on `x = [e1; e2; e3]`:

```
  P2 = blkdiag(M1, M2, 0)

  P1 = i * [[  0 ,   0 , P13 ],
            [  0 ,   0 , P23 ],
            [ A31, A32, A33  ]]

  P0 =     [[ Sxx - A11      , -Sxy - A12      , -A13          ],
            [ -Syx - A21     , Syy - A22       , -A23          ],
            [ -(K11 + K21)   , -(K12 + K22)    , -(K13 + K23)  ]]
```

**It reduces to the shipped E-form ALGEBRAICALLY, not merely numerically.**
Set the OOP entries to zero.  Row 3 becomes
`i q A33 e3 = (K11 + K21) e1 + (K12 + K22) e2 =: Kdiv e_t`, so
`i q e3 = A33^-1 Kdiv e_t`; substituting into rows 1 and 2,

```
  q^2 blkdiag(M1, M2) e_t
      = [[A11 - Sxx , A12 + Sxy], [A21 + Syx, A22 - Syy]] e_t
        - [P13; P23] A33^-1 Kdiv e_t
      = ( [eps_t] + S_tt - K_tz eps33^-1 K_zt ) e_t
      = Lmat e_t                      (Granet Eq. 23-24, term for term)
```

with `blkdiag(M1, M2) = -Rmat`.  That identification is measured in M3 T3
(and the probe's E-form is measured against the SHIPPED
`Granet2DTransverseE` in M0.3).

`P2` is singular by construction (`q^2` does not multiply `E3`), so the
`6 q^2` linearization
`[[0, I], [-P0, -P1]] y = q [[I, 0], [0, P2]] y` is a genuinely singular
pencil and needs a QZ.  The eigenvalue budget it produces is a MEASUREMENT,
not an assumption -- see M1 T2b.

---

## S4.  M0 -- conventions and self-checks

`validation/probe_pmm2d_staggered_oop/m0_convention.py`.

| check | measured | what it establishes |
|---|---|---|
| 0.1 Bloch sign, read off the hat | `hat(d)/hat(0) - exp(+iKd)` = **0.0e+00** | the basis carries `exp(+iKx)` |
| 0.1 Bloch sign, phase-slope fit of the lowest eigenfunction | slope `+1.234000` vs exact `+1.234000` (err **4.4e-13**); `mu_0 = 1.52276` vs `K^2 = 1.52276` (err **2.2e-13**) | independent confirmation, not a re-reading of the same line |
| 0.2 exact quartic roots, ISOTROPIC (double roots) | **2.3e-08** | the degeneracy's `sqrt(eps_mach)` conditioning, NOT the solver -- so the dispersion gate must be run on a NON-degenerate tensor |
| 0.2 exact quartic roots, uniaxial optic-axis-z (simple roots, closed form o/e) | **9.3e-15** | the root solver is machine-exact where it is used |
| 0.3 probe E-form vs shipped `Granet2DTransverseE` | rel `\|L - Lmat\|` = **5.86e-15** (normal AND oblique); rel `\|G + Rmat\|` = **0.0** / 4.8e-17 | the probe's assembly IS the shipped isotropic discretization -- everything downstream is a generalization of it, not a re-implementation |
| 0.4 strong H-partner vs shipped Eq.-25 partner | ratio = **-1.000000000i** exactly, spread **6.5e-12** over 200 modes | the two H conventions differ by ONE global constant, which cancels in every S-matrix; the probe's strong form is usable everywhere |

The 0.2 row changed the design of M1: the plan's tilted-uniaxial probe is
non-degenerate (o and e distinct), so the root solver is exact there; but any
gate written on an ISOTROPIC reference tensor would sit on a 1e-8 floor that
is nothing to do with the solver.

---

## S5.  M1 -- uniform-slab dispersion

Probe cell: `(2,2)` grid, `px = py = 0.9 lam`, tilted uniaxial
`no = 1.5, ne = 1.7, tilt 35 deg` --

```
 eps = [[2.46055, 0, 0.30070], [0, 2.25, 0], [0.30070, 0, 2.67945]]
```

plus an azimuth-40 variant (`e23 != 0` as well) and a lossy variant
(`+0.08i` on the principal values).

### T1  per-harmonic residual (max over the harmonic's four exact roots of the
### distance to the nearest generator eigenvalue, relative)

Candidate (a), `dim = 4 q^2`, `q = 2(M-1)`:

| M | dim | eig [s] | (0,0) | (+1,0) | (0,+1) | (-1,0) | (+1,+1) | (+2,0) |
|---|---|---|---|---|---|---|---|---|
| \[normal\] 5 | 256 | 0.06 | 1.2e-14 | 1.3e-04 | 8.9e-06 | 1.3e-04 | 3.5e-04 | 3.8e-02 |
| 6 | 400 | 0.17 | 1.6e-14 | 3.5e-06 | 1.6e-07 | 3.5e-06 | 9.0e-06 | 4.3e-03 |
| 7 | 576 | 0.49 | 6.4e-15 | 6.3e-08 | 2.1e-09 | 6.3e-08 | 1.6e-07 | 3.4e-04 |
| 8 | 784 | 1.00 | 1.5e-14 | 8.2e-10 | 2.1e-11 | 8.2e-10 | 2.0e-09 | 1.9e-05 |
| 9 | 1024 | 2.04 | 4.7e-14 | 8.1e-12 | 1.5e-13 | 8.1e-12 | 2.0e-11 | 7.9e-07 |
| 10 | 1296 | 3.68 | 2.4e-14 | 1.3e-13 | 5.2e-14 | 1.4e-13 | 2.0e-13 | 2.6e-08 |
| \[oblique 25/40\] 5 | 256 | 0.07 | 5.3e-10 | 5.6e-03 | 2.8e-03 | 4.2e-06 | 2.8e-03 | 6.4e-02 |
| 6 | 400 | 0.18 | 1.1e-12 | 2.5e-04 | 1.2e-04 | 5.4e-08 | 1.2e-04 | 9.6e-03 |
| 7 | 576 | 0.52 | 1.0e-14 | 7.6e-06 | 3.3e-06 | 4.7e-10 | 3.6e-06 | 1.0e-03 |
| 8 | 784 | 1.02 | 2.8e-14 | 1.7e-07 | 6.7e-08 | 3.0e-12 | 7.7e-08 | 7.9e-05 |
| 9 | 1024 | 2.12 | 3.8e-14 | 2.8e-09 | 1.0e-09 | 3.9e-14 | 1.2e-09 | 4.5e-06 |
| 10 | 1296 | 3.70 | 3.6e-14 | 3.6e-11 | 1.3e-11 | 4.0e-14 | 1.6e-11 | 1.9e-07 |

Candidate (d), `dim = 6 q^2` (of which `q^2` are infinite), reproduces
candidate (a) column for column to the printed digits:

| M | dim | n_inf | eig [s] | (0,0) | (+1,0) | (0,+1) | (+2,0) |
|---|---|---|---|---|---|---|---|
| \[normal\] 5 | 320 | 64 | 0.53 | 1.1e-14 | 1.3e-04 | 8.9e-06 | 3.8e-02 |
| 6 | 500 | 100 | 1.76 | 8.6e-15 | 3.5e-06 | 1.6e-07 | 4.3e-03 |
| 7 | 720 | 144 | 5.51 | 1.1e-14 | 6.3e-08 | 2.1e-09 | 3.4e-04 |
| 8 | 980 | 196 | 14.29 | 4.8e-15 | 8.2e-10 | 2.1e-11 | 1.9e-05 |
| \[oblique 25/40\] 7 | 720 | 144 | 5.76 | 7.4e-15 | 7.6e-06 | 3.3e-06 | 1.0e-03 |
| 8 | 980 | 196 | 15.46 | 1.3e-14 | 1.7e-07 | 6.7e-08 | 7.9e-05 |

**Reading.**  The FUNDAMENTAL harmonic is reproduced at **1e-14 -- machine
precision -- by BOTH candidates, at normal AND at oblique incidence, with an
OUT-OF-PLANE tensor.**  That is the factor-i gate: the (0,0) roots here are
`{-1.542955, -1.439234, +1.439234, +1.470290}` -- an asymmetric extraordinary
pair whose sum is `-0.0727`, and a generator with the wrong relative `i` on
the OOP blocks cannot produce it (T3).

Every other harmonic converges SPECTRALLY in `M` (three to four decades per
two degrees) toward the same machine precision.  This refines a premise of the
plan: the staggered basis is POLYNOMIAL, so "the spectrum equals the union of
the exact roots to 1e-12" is a CONVERGENCE statement per harmonic, not an
identity -- unlike a Fourier basis, where it is an identity.

### T1b  how many harmonics does the basis actually carry?

Discrete 1-D Bloch wavenumbers (the generalized eig of `<dBtil|dBtil>` against
`<Btil|Btil>`) against the exact `|kx_m|/k0`, `Nx = 2`:

| M | q | max abs error over the q lowest | resolved to < 1e-8 (normal) | resolved (oblique) |
|---|---|---|---|---|
| 5 | 8 | 2.45 | 1 / 8 | 1 / 8 |
| 6 | 10 | 4.06 | 1 / 10 | 1 / 10 |
| 7 | 12 | 6.11 | 2 / 12 | 2 / 12 |
| 8 | 14 | 8.60 | 3 / 14 | 2 / 14 |
| 9 | 16 | 11.5 | 2 / 16 | 3 / 16 |
| 10 | 18 | 14.9 | 4 / 18 | 4 / 18 |

The top of the 1-D spectrum is O(1) wrong, as in every spectral-element
method.  This is a property of the SHIPPED basis, unchanged by anything here,
and it is what sets the "off-branch" population in T2 below -- those modes are
UNRESOLVED HARMONICS, not spurious modes.

### T2  assignment census (optimal one-to-one matching of the WHOLE spectrum)

Two root sets are used, and the difference between them is the whole point.
`discrete` evaluates the quartic on the DISCRETE channel wavenumbers the basis
carries (a uniform cell block-diagonalizes into `(jx, jy)` channels, so if the
generator holds the exact plane-wave symbol its spectrum IS this set,
regardless of resolution); `exact` uses the physical harmonics and therefore
adds the T1b resolution error on top.

| cand | case | root set | dim | n_inf | median dist | max dist | off (>0.3) | max relative flux of off | min abs q of off | min decay Re(lam) of off |
|---|---|---|---|---|---|---|---|---|---|---|
| (a) M=8 | normal / tilt35 | discrete | 784 | 0 | 6.0e-04 | 1.85 | 134 | 6.0e-15 | 5.77 | 5.75 |
| (a) | normal / tilt35 | exact | 784 | 0 | 2.8e-01 | 13.7 | 375 | 1.1e-14 | 5.15 | 5.15 |
| (a) | obl25 / tilt35 | discrete | 784 | 0 | 1.2e-04 | 1.29 | 84 | 2.1e-14 | 7.64 | 7.62 |
| (a) | obl25 / azim40 | discrete | 784 | 0 | 1.3e-03 | 1.88 | 128 | 2.1e-14 | 7.46 | 7.45 |
| (a) | obl25 / lossy | discrete | 784 | 0 | 1.2e-04 | 1.29 | 84 | 1.9e-02 | 7.62 | 7.60 |
| (d) M=7 | normal / tilt35 | discrete | 720 | 144 | 4.5e-04 | 1.45 | 220 | 2.5e-02 | **1.3e-16** | **2.1e-17** |
| (d) | obl25 / tilt35 | discrete | 720 | 144 | 1.9e-04 | 0.96 | 192 | **1.6e-01** | 2.1e-16 | 1.1e-17 |
| (d) | obl25 / azim40 | discrete | 720 | 144 | 1.4e-03 | 1.41 | 218 | 7.0e-02 | 2.6e-16 | 5.4e-17 |
| (d) | obl25 / lossy | discrete | 720 | 144 | 1.9e-04 | 0.96 | 192 | 4.9e-02 | 3.1e-16 | 3.8e-19 |

**Candidate (a): the spurious census is EMPTY in the operational sense.**
Every mode that is not on a physical branch is DEEPLY EVANESCENT
(`min |q| >= 5.77`, `min Re(lam) >= 5.75`, i.e. `exp(-5.75 * k0 z)` -- the mode
has decayed by `e^-36` in one wavelength) and carries **relative flux
<= 2.1e-14** on a lossless cell (1.9e-2 on a lossy one, where "flux" is not a
conserved label anyway).  This is exactly the population that
`rcwa._core._select_forward_flux`'s DEEP-DECAY (`|Re gam| > 0.5`) override was
written for, and its bar of 0.5 is beaten by an order of magnitude.

**Candidate (d): the census is NOT empty.**  Besides the deep-evanescent
population it carries `q^2` modes at `q = 0` with relative flux up to
**1.6e-01** -- flux-carrying by the selector's measure, and NOT decay-
classified (`Re(lam) ~ 1e-17`).  They must be removed explicitly before the
forward/backward split; the probe does that with a `|q| > 1e-6 max|q|` filter
(S3 / `probe_common._NULL_TOL`).  The gap is 12 decades wide at these `M`
(largest null `4.3e-13` at M=7 against a smallest physical `|q| ~ 1.4`), so
the filter is safe here -- but it is a filter on the eigenvalue MAGNITUDE, and
it is exactly a physical `|q| -> 0` (a Rayleigh cutoff) that would collide
with it.  That is the price, and it is a permanent one.

### T2b  are (a) and (d) the same discretization?

| case | M | dim (a) | dim (d), finite | `max_a min_d |qa - qd|` | (d)-only modes | their abs q |
|---|---|---|---|---|---|---|
| normal | 5 | 256 | 320 | 4.9e-13 | 64 (= q^2) | 5.5e-17 .. 2.4e-14 |
| normal | 6 | 400 | 500 | 2.5e-11 | 100 (= q^2) | 8.0e-18 .. 8.2e-14 |
| normal | 7 | 576 | 720 | 1.5e-11 | 144 (= q^2) | 1.3e-16 .. 4.3e-13 |
| oblique | 5 | 256 | 320 | 7.4e-13 | 64 | 1.9e-16 .. 1.4e-14 |
| oblique | 6 | 400 | 500 | 1.7e-12 | 100 | 6.6e-17 .. 6.2e-14 |
| oblique | 7 | 576 | 720 | 3.0e-12 | 144 | 2.1e-16 .. 1.6e-13 |

**Candidate (d)'s spectrum is candidate (a)'s spectrum (to 3e-11) plus exactly
`q^2` null modes at `q = 0`, plus `q^2` eigenvalues at infinity.**  The two
formulations are the SAME discretization; (d) simply carries the static null
space of the div(D) = 0 constraint explicitly and pays a QZ for it.  Everything
downstream in M2, M4, M5 and M6 confirms this: the two candidates agree to the
printed digits on every observable.

### T3  convention arbitration -- what the dispersion gate can and cannot see

Discriminator: the sum of the four (0,0) roots.  It is exactly zero for any
in-plane tensor and non-zero only through the out-of-plane coupling; a
generator with the legacy real-coefficient OOP blocks produces the artificially
symmetric pair and hence a zero sum (this is
`AUDIT_OOP_GENERATOR_FACTOR_I_2026_07_14`'s anchor 1).  The exact value on the
lossy tilted probe is `-0.072600 + 0.002168i`.

| assembly variant | cand | h(0,0) residual | h(+1,0) | `\|sum(matched q) - sum(exact)\|` |
|---|---|---|---|---|
| **reference** | (a) | **2.8e-14** | 1.5e-07 | **3.9e-14** |
| **reference** | (d) | **6.5e-15** | 6.9e-06 | **9.6e-15** |
| negate OOP (`e13,e23,e31,e32 -> -`) | (a) | 4.7e-02 | 6.3e-02 | 4.2e-02 |
| negate OOP | (d) | 4.7e-02 | 6.3e-02 | 4.2e-02 |
| drop OOP entirely | (a)/(d) | 1.9e-02 | 1.2e-01 | 2.4e-03 |
| conjugate OOP | (a)/(d) | 2.8e-14 | 1.5e-07 | 3.9e-14 |
| transpose OOP (`e13 <-> e31`) | (a)/(d) | 2.8e-14 | 1.5e-07 | 3.9e-14 |

Repeated on an ASYMMETRIC probe (`e13 = 0.3007 + 0.11i`,
`e31 = 0.3007 - 0.11i`) the same pattern holds: negate 4.2e-02, drop 2.4e-03,
reference / conjugate / transpose all 1e-14.

**The gate is sharp for the sign (12 decades of separation) and BLIND to the
transpose** -- and that blindness is a theorem, not a weakness of the probe:
`det(k k^T - |k|^2 I + eps)` is invariant under `eps -> eps^T`, so no
dispersion measurement can see a swapped `e13 / e31` placement.  Only the
FIELDS can.  M2 therefore carries a non-reciprocal (`e13 = conj(e31)`,
Hermitian, still lossless) tensor and a transpose negative control.

---

## S6.  M2 -- Berreman 4x4 on a uniform out-of-plane slab

PLACEHOLDER-M2

---

## S7.  M3 -- the in-plane limit

PLACEHOLDER-M3

---

## S8.  M4 -- y-uniform out-of-plane stripe grating

Stripe along `x` (`eps` varies in `x` only), `px = py = 1.2 lam`,
`depth = 0.4 lam`, duty 0.5, `n_sup = 1`, `n_sub = 1.5`, classical mount.
Solved on the 2-D staggered cell and compared PER ORDER against
`rcwa_jones_1d(n_orders = 61)` and `pmm_jones_1d(degree = 18,
far_field_orders = 31, stabilize = False)`.  A lateral shift of the ridge
multiplies order `m` by a phase, so the comparison is on the shift-invariant
quantities: the per-order efficiencies and the order-0 Jones.

**The oracles' own spread is the bar** (they are two independent engines):

| case | theta | rcwa vs pmm1d, max abs dR | max abs dT | max abs dJones |
|---|---|---|---|---|
| ridge = tilt35, groove = air | 0 | 7.5e-07 | 6.6e-06 | 3.8e-06 |
| ridge = tilt35, groove = air | 25 | 1.6e-06 | 4.1e-06 | 5.6e-06 |
| ridge = tilt35 lossy, groove = 2.25 | 0 | 2.4e-08 | 2.4e-07 | 7.9e-08 |
| ridge = tilt35 lossy, groove = 2.25 | 25 | 2.5e-08 | 1.5e-06 | 6.8e-08 |

### ridge = tilted-35 uniaxial, groove = air (a hard DIELECTRIC CORNER cell)

| cand | M | dim | max abs dR vs rcwa | max abs dT | max abs dJones | y-leak (R) | y-leak (T) | abs(R+T-1) |
|---|---|---|---|---|---|---|---|---|
| \[theta = 0\] (a) | 5 | 128 | 1.29e-04 | 8.42e-04 | 6.95e-04 | 1.8e-30 | 3.0e-29 | 2.74e-04 |
| (a) | 6 | 200 | 4.20e-05 | 1.99e-04 | 2.24e-04 | 1.2e-29 | 1.3e-28 | 7.10e-06 |
| (a) | 7 | 288 | 2.03e-05 | 8.82e-05 | 2.10e-04 | 2.8e-29 | 1.6e-27 | 2.88e-07 |
| (a) | 8 | 392 | 1.38e-05 | 8.11e-05 | 9.27e-05 | 1.9e-28 | 2.3e-27 | 1.20e-08 |
| (d) | 7 | 288 | 2.03e-05 | 8.82e-05 | 2.10e-04 | 1.1e-29 | 6.0e-29 | 2.88e-07 |
| \[theta = 25\] (a) | 5 | 128 | 7.49e-04 | 6.19e-03 | 2.64e-03 | 5.3e-30 | 3.6e-29 | 1.49e-03 |
| (a) | 6 | 200 | 1.71e-05 | 2.66e-04 | 2.42e-04 | 1.5e-28 | 8.0e-28 | 7.39e-05 |
| (a) | 7 | 288 | 1.26e-05 | 1.24e-04 | 1.25e-04 | 3.1e-28 | 1.1e-27 | 5.19e-06 |
| (a) | 8 | 392 | 8.30e-06 | 9.83e-05 | 8.56e-05 | 5.5e-28 | 1.0e-26 | 1.17e-07 |
| (d) | 7 | 288 | 1.26e-05 | 1.24e-04 | 1.25e-04 | 1.9e-28 | 2.8e-28 | 5.19e-06 |

### ridge = tilted-35 LOSSY uniaxial, groove = 2.25 (a weak-contrast cell)

| cand | M | dim | max abs dR vs rcwa | max abs dT | max abs dJones | y-leak (R) | abs(R+T-1) = absorbed |
|---|---|---|---|---|---|---|---|
| \[theta = 0\] (a) | 5 | 128 | 6.83e-06 | 1.83e-05 | 1.94e-05 | 3.3e-31 | 6.01e-02 |
| (a) | 7 | 288 | 3.05e-06 | 1.37e-06 | 9.86e-06 | 1.6e-30 | 6.01e-02 |
| (a) | 8 | 392 | 2.71e-06 | 3.18e-06 | 9.33e-06 | 6.8e-30 | 6.01e-02 |
| (d) | 7 | 288 | 3.05e-06 | 1.37e-06 | 9.86e-06 | 1.5e-30 | 6.01e-02 |
| \[theta = 25\] (a) | 8 | 392 | 1.67e-06 | 8.30e-06 | 7.87e-06 | 3.9e-29 | 6.39e-02 |
| (d) | 7 | 288 | 1.95e-06 | 8.35e-06 | 9.59e-06 | 1.0e-29 | 6.39e-02 |

**Reading.**

* Both candidates converge monotonically to the 1-D engines.  On the
  weak-contrast cell they reach **2.7e-06 / 9.3e-06** at M = 8, within a
  factor of ~100 of the two oracles' own disagreement.  On the air-groove
  cell -- which has four DIELECTRIC CORNERS, where the shipped module
  documents ALGEBRAIC (not spectral) convergence -- they reach 1.4e-05 at
  M = 8 and are still falling; that is the CORNER CAP of the shipped basis,
  not an out-of-plane defect (the same cell without OOP entries caps the same
  way, and the numbers at M = 5..8 fall as `M^-3`-ish, exactly the documented
  behaviour).
* **Y-MOMENTUM CONSERVATION is exact to 1e-30 .. 1e-26.**  A y-invariant cell
  scatters NOTHING into a `(m, n != 0)` order in either candidate.  This is a
  strong statement about the assembly: the OOP blocks `A23`, `A32`, `K23`
  carry a `y` derivative and a `y`-mixed mass, and a mis-placed one would
  break exactly this.
* Candidates (a) and (d) agree to the printed digits in every row.
* Lossless closure falls to **1.2e-08** at M = 8; the lossy rows' 6.0e-02 is
  the ABSORBED fraction, not a closure failure (no unity is claimed there).

---

## S9.  M5 -- a genuinely 2-D out-of-plane pillar

PLACEHOLDER-M5

---

## S10.  M6 -- cascade stability and energy closure

PLACEHOLDER-M6

---

## S11.  M7 -- cost, and the normal-incidence block structure

PLACEHOLDER-M7

---

## S12.  Scored against the plan's verdict rule

PLACEHOLDER-SCORE

---

## S13.  Recommended integration route

PLACEHOLDER-ROUTE

---

## S14.  Files and commands

PLACEHOLDER-FILES
