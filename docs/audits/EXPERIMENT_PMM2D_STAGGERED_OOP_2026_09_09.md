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

| candidate | dimension | eig | verdict |
|---|---|---|---|
| **(a)** first-order staggered generator on `[E1; E2; G1; G2]` | `4 q^2` | Cholesky-whitened standard eig | **GO** |
| **(d)** quadratic E-form keeping `div(D) = 0`, linearized | `6 q^2` | QZ (singular leading coefficient) | **PARTIAL -- correct but strictly dominated** |

The two are the SAME discretization: candidate (d)'s spectrum is candidate
(a)'s spectrum to 3e-11 plus exactly `q^2` null modes at `gamma = 0` plus
`q^2` eigenvalues at infinity (S5 T2b), and every observable in M2, M4, M5, M6
agrees between them to the printed digits.  (d) costs 8 - 19x the eig and
needs an extra eigenvalue-magnitude filter that (a) does not; it earns its
place as the DERIVATION that proves the exact reduction to the shipped
E-form, not as the integration route.

**The decisive numbers for candidate (a):**

* **dispersion** -- the fundamental harmonic's four exact quartic roots to
  **2.8e-14** at normal AND conical incidence with an out-of-plane tensor; the
  wrong-sign assembly sits at **4.7e-02** (12 decades of separation), and every
  higher harmonic converges spectrally in `M`;
* **spurious census** -- of `4 q^2` modes, the ones not on a physical branch
  are ALL deep evanescent (`Re(lam) >= 5.75`, ten times
  `_select_forward_flux`'s deep-decay bar) with relative flux **2.1e-14**; the
  forward/backward split came out exactly `2q^2 / 2q^2` in every one of the 72
  cascade rows;
* **Berreman** -- `dR = 5.6e-16`, `dJones = 1.6e-15` on a uniform out-of-plane
  slab at oblique (M = 9), `3.2e-15 / 8.2e-15` at conical (M = 7), for
  lossless, lossy AND non-reciprocal tensors, with the order leakage
  identically zero; the drop / negate / transpose negative controls sit at
  **1.2e-04 .. 2.1e-03**;
* **1-D per order** -- **2.7e-06 / 9.3e-06** against `rcwa_jones_1d` and
  `pmm_jones_1d` (whose own mutual spread is 2.4e-08 .. 5.6e-06), with
  y-momentum conserved to **1e-30**;
* **cascade** -- lossless closure **1e-13** (uniform) at 0.25, 1 AND 3
  wavelengths with a forward growth factor of exactly **1.0000e+00**;
* **cost** -- **1.7 - 2.4x** the in-plane `2 q^2` path, and the normal-incidence
  parity-times-sign involution that `_generator_block_eig` exploits HOLDS on
  the staggered generator (`||R A R + A|| / ||A|| = 2.3e-15`), so a `2 q^2`
  accelerator is available there.

**One premise of the plan was refuted and replaced.**  "For a uniform cell the
generator's spectrum must equal the union over Bloch harmonics of the exact
quartic roots to ~1e-12" is an IDENTITY in a Fourier basis and a CONVERGENCE
statement in this polynomial one: at `Nx = 2, M = 10` the staggered basis
carries only 4 of its 18 harmonics per axis to 1e-8 (S5 T1b).  The gate was
re-derived per harmonic with an M-ladder, which is sharper, and the census was
re-run against the DISCRETE channel wavenumbers so that "unresolved harmonic"
and "spurious mode" stop being conflated.

**One physics degeneracy will trap the integration's tests if it is not
written down.**  The dispersion relation is invariant under `eps -> eps^T`, so
NO dispersion or energy measurement can detect a swapped `e13 / e31`
placement; and negating the out-of-plane block is an exact symmetry whenever
the director azimuth equals the incidence azimuth.  The gates therefore need a
NON-RECIPROCAL tensor and a director azimuth different from the incidence
azimuth (S6).

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

Notation for the assembled blocks (`bracket(x, w, y)` denotes the Galerkin block
`INT conj(x) w y`; all built by `probe_common.StaggeredCell`, `k0`-normalized where a derivative appears):

| block | definition |
|---|---|
| `M1, M2, M3, Mw` | Grams of `V1, V2, V3, Vw` |
| CwE1 = bracket(Vw, D2, V1), CwE2 = bracket(Vw, D1, V2) | the strong curl (shipped `Cw_E1`, `Cw_E2`) |
| P13 = bracket(V1, D1, V3), P23 = bracket(V2, D2, V3) | the strong mimetic gradient (shipped `-Ktz`) |
| Aab = bracket(Va, e_ab, Vb), a,b in 1..3 | eps-weighted component masses (Appendix-A Eq. 40/41, plus the FOUR NEW out-of-plane blocks `A13, A23, A31, A32`) |
| Kab = bracket(D_a V3, e_ab, Vb) | the eps-weighted div-D blocks with the derivative on the V3 TEST (Appendix-A Eq. 44, plus the NEW third column `K13, K23`) |
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
| 0.3 probe E-form vs shipped `Granet2DTransverseE` | rel max abs(L - Lmat) = **5.86e-15** (normal AND oblique); rel max abs(G + Rmat) = **0.0** / 4.8e-17 | the probe's assembly IS the shipped isotropic discretization -- everything downstream is a generalization of it, not a re-implementation |
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

| case | M | dim (a) | dim (d), finite | max over (a) of dist to nearest (d) | (d)-only modes | their abs q |
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

| assembly variant | cand | h(0,0) residual | h(+1,0) | abs(sum(matched q) - sum(exact)) |
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

The exact oracle for a uniform layer at any incidence.  `px = py = 0.9 lam`,
`depth = 0.35 lam`, `n_sup = 1`, `n_sub = 1.5`; both incident polarizations;
normal / oblique 25 deg (classical) / conical 25 deg with phi 40 deg.  Four
tensors, chosen so that between them they discriminate every wrong assembly:

| tensor | e13 | e23 | why |
|---|---|---|---|
| tilt35 (lossless, Hermitian) | 0.30070 | 0 | the baseline; closure is a two-sided claim |
| tilt35 azim40 | 0.23035 | 0.19329 | brings `e23` / `A23` / `K23` in |
| tilt35 lossy 0.08 | 0.30070 | 0 | non-Hermitian: no unity claimed, only the deficit |
| **tilt35 NON-RECIPROCAL** `e13 = conj(e31)` | 0.30070 + 0.22000i | 0 | Hermitian (still lossless) but NOT symmetric -- the only case that can see an `e13 <-> e31` swap |

### convergence ladder (max over both polarizations)

`tilt35 (lossless)`; Berreman gives `R = [0.041253957, 0.040000000]` at normal
and `R = [0.038164702, 0.041516667]` at conical, closing to 1e-15.

| cand | M | dim | eig+solve [s] | max abs dR | max abs dT | max abs dJones | order leak | abs(R+T-1) |
|---|---|---|---|---|---|---|---|---|
| \[normal\] (a) | 5 | 128 | 0.16 | 4.9e-16 | 4.2e-14 | 1.3e-15 | **0.0e+00** | 4.2e-14 |
| (a) | 10 | 648 | 11.9 | 6.7e-16 | 1.6e-13 | 3.9e-15 | 0.0e+00 | 1.6e-13 |
| (d) | 8 | 392 | 18.8 | 4.7e-16 | 5.6e-15 | 1.7e-15 | 0.0e+00 | 4.9e-15 |
| \[oblique 25\] (a) | 5 | 128 | 0.18 | 6.8e-09 | 1.6e-07 | 2.0e-08 | 0.0e+00 | 1.7e-07 |
| (a) | 6 | 200 | 0.50 | 7.8e-12 | 4.9e-11 | 2.1e-11 | 0.0e+00 | 4.2e-11 |
| (a) | 7 | 288 | 1.36 | 4.3e-14 | 8.5e-13 | 1.2e-13 | 0.0e+00 | 8.9e-13 |
| (a) | 8 | 392 | 2.94 | 9.3e-16 | 8.7e-14 | 2.1e-15 | 0.0e+00 | 8.7e-14 |
| (a) | 9 | 512 | 6.95 | **5.6e-16** | **4.1e-15** | **1.6e-15** | 0.0e+00 | 4.4e-15 |
| (d) | 8 | 392 | 18.3 | 1.3e-15 | 1.6e-14 | 2.8e-15 | 0.0e+00 | 1.5e-14 |
| \[conical 25/40\] (a) | 5 | 128 | 0.17 | 9.4e-10 | 2.1e-08 | 2.5e-09 | 0.0e+00 | 2.2e-08 |
| (a) | 6 | 200 | 0.48 | 2.7e-13 | 1.7e-12 | 1.1e-12 | 0.0e+00 | 1.4e-12 |
| (a) | 7 | 288 | 1.34 | **3.2e-15** | 3.7e-14 | **8.2e-15** | 0.0e+00 | 3.6e-14 |
| (a) | 8 | 392 | 2.87 | 1.6e-15 | 5.3e-14 | 4.3e-15 | 0.0e+00 | 5.5e-14 |
| (d) | 8 | 392 | 18.6 | 1.0e-15 | 1.7e-14 | 2.5e-15 | 0.0e+00 | 1.6e-14 |

The other three tensors give the same ladder to within a factor of 2 -- e.g.
the NON-RECIPROCAL tensor at conical reaches `dR = 7.7e-16`,
`dJones = 2.4e-15` at M = 8, and the lossy one `dR = 8.2e-16`,
`dJones = 2.5e-15` with `abs(R+T-1) = 1.11e-01` reproducing Berreman's own
`-1.11e-01 / -1.09e-01` absorbed fraction to 1e-15.

**Both candidates reproduce the exact 4x4 Berreman R, T and BOTH Jones
matrices to MACHINE PRECISION on a uniform out-of-plane slab, at normal,
oblique and conical incidence, lossless, lossy and non-reciprocal.**  The
oblique / conical rows show the documented degree limitation of a UNIFORM
region in this basis (the Bloch phase must be resolved): 6.8e-09 at M = 5
falling to 1e-15 by M = 8-9, four decades per two degrees.  `order leak` is
IDENTICALLY ZERO -- a uniform cell puts nothing in a non-zero diffraction
order.

### negative controls (candidate (a), M = 8, conical 25/40)

The reference rows above sit at `dR ~ 1e-15`, `dJones ~ 3e-15`.  The same
solve with the out-of-plane block deliberately mis-assembled:

| tensor | control | max abs dR | max abs dJones | separation from the reference |
|---|---|---|---|---|
| tilt35 | drop OOP | 1.97e-04 | 2.14e-03 | 11 / 12 decades |
| tilt35 | negate OOP | 3.77e-05 | 3.53e-03 | 10 / 12 decades |
| tilt35 | transpose OOP | 1.62e-15 | 4.34e-15 | none -- `e13 = e31` here, so the transpose is a NO-OP |
| tilt35 azim40 | drop OOP | 1.21e-04 | 1.15e-03 | 11 / 12 decades |
| tilt35 azim40 | negate OOP | 7.08e-16 | 1.84e-15 | none -- see below |
| tilt35 lossy | drop OOP | 4.47e-04 | 2.06e-03 | 11 / 12 decades |
| tilt35 lossy | negate OOP | 8.93e-05 | 3.50e-03 | 11 / 12 decades |
| **NON-RECIPROCAL** | drop OOP | 1.29e-03 | 4.01e-03 | 12 / 12 decades |
| **NON-RECIPROCAL** | negate OOP | 2.09e-03 | 5.50e-03 | 12 / 12 decades |
| **NON-RECIPROCAL** | **transpose OOP** | **2.06e-03** | **5.49e-03** | **12 / 12 decades** |

Two of these rows are degeneracies of the PHYSICS, not weak measurements, and
a test suite must know about them:

* **transpose is invisible on a SYMMETRIC tensor** (trivially -- the operation
  does nothing).  It becomes a 2.1e-03 signal the moment the tensor is
  non-reciprocal, which is the case a magneto-optic or in-plane-magnetized LC
  cell actually is.
* **negate is invisible on the azim40 tensor at phi = 40** because there the
  director azimuth EQUALS the incidence azimuth: negating the OOP block is the
  180-degree rotation about z, which maps that whole configuration onto itself
  (both the axis and `k_t` rotate by 180 degrees, and the lab Jones is even
  under the simultaneous sign flip of x and y).  It is a 3.8e-05 .. 2.1e-03
  signal at every other azimuth.

So the gate is sharp, but **a single tensor and a single azimuth is not enough
to gate an OOP assembly** -- the integration's tests need at least a
non-symmetric tensor and a director azimuth that differs from the incidence
azimuth.  That is the most transferable finding in this document.

---

## S7.  M3 -- the in-plane limit

The question the plan asks is precise: with the cross terms zeroed, are the OOP
discretizations EQUIVALENT to the shipped in-plane one (distance ~1e-12), or
merely both convergent?  Three cells, all with `e13 = e23 = e31 = e32 = 0`: a
patterned ISOTROPIC cell (`eps` 2.25 / 1.0 / 4.0+0.2i), a patterned IN-PLANE
UNIAXIAL cell (director in the plane, azimuth 30 deg), and a GYROTROPIC cell
(`e12 = -e21 = 0.5i`, Hermitian).

### T1  spectra -- candidate's eigenvalue set vs the E-form's `{+q, -q}`

max over the candidate's modes of the distance to the nearest E-form
eigenvalue:

| cell | M | dim | (a) normal | (a) oblique | (d) normal | (d) oblique |
|---|---|---|---|---|---|---|
| isotropic | 5 | 256 | 2.6e-13 | 1.4e-12 | 2.7e-12 | 2.0e-12 |
| isotropic | 6 | 400 | 1.3e-12 | 9.4e-13 | 6.6e-12 | 4.3e-12 |
| isotropic | 7 | 576 | 3.9e-12 | 5.1e-12 | 2.0e-11 | 6.9e-12 |
| in-plane uniaxial | 7 | 576 | 5.2e-12 | 1.3e-11 | 6.0e-12 | 1.3e-11 |
| gyrotropic | 5 | 256 | 3.9e-13 | 6.1e-13 | 1.8e-12 | 2.7e-12 |
| gyrotropic | 7 | 576 | 4.7e-12 | 3.1e-12 | 1.6e-11 | 1.3e-11 |

The residual grows with dimension exactly as an eigenvalue conditioning number
does (2e-13 at 256 to 2e-11 at 576), not as a discretization error would.

### T2  observables -- candidate vs the probe's E-form vs the SHIPPED solver

| theta | M | R(a) - R(eform) | R(d) - R(eform) | J(a) - J(eform) | J(d) - J(eform) |
|---|---|---|---|---|---|
| 0 | 6 | 9.5e-16 | 1.2e-15 | 5.8e-15 | 5.1e-15 |
| 0 | 7 | 1.9e-16 | 2.3e-15 | 9.5e-15 | 1.8e-14 |
| 25/40 | 6 | 2.4e-15 | 1.0e-15 | 9.7e-15 | 1.1e-14 |
| 25/40 | 7 | 1.2e-15 | 3.5e-15 | 1.9e-14 | 1.6e-14 |

and against the SHIPPED scalar `pmm_efficiency_2d_staggered` at normal
incidence, where its `te`/`tm` rows ARE the probe's `Ey`/`Ex` rows:

| M | shipped te (Ey) sum R | probe row 1 | diff | shipped tm (Ex) sum R | probe row 0 | diff |
|---|---|---|---|---|---|---|
| 6 | 0.0288319480 | 0.0288319480 | 7.9e-15 | 0.0288319480 | 0.0288319480 | 2.2e-15 |
| 7 | 0.0288461842 | 0.0288461842 | 1.2e-14 | 0.0288461842 | 0.0288461842 | 2.8e-15 |

### T3  the algebraic reduction of candidate (d)

Rebuilding `Lmat` and `Rmat` out of candidate (d)'s `P0`, `P1`, `P2` blocks by
the elimination of S3, on the same cell:

| cell | incidence | rel max abs(L_rec - L_probe) | rel max abs(G_rec - G) | rel max abs(L_rec - Lmat_shipped) |
|---|---|---|---|---|
| isotropic | normal | **0.00e+00** | **0.00e+00** | 5.86e-15 |
| isotropic | oblique | **0.00e+00** | **0.00e+00** | 5.86e-15 |
| in-plane uniaxial | normal / oblique | **0.00e+00** | **0.00e+00** | (no shipped tensor arm) |
| gyrotropic | normal / oblique | **0.00e+00** | **0.00e+00** | (no shipped tensor arm) |

The `0.00e+00` columns are BIT-IDENTICAL, which is what a derivation check
should give -- they confirm that the S3 elimination of `e3` from `P1`/`P0`
really is the shipped `[eps_t] + S_tt - K_tz eps33^-1 K_zt`, term for term,
not merely numerically close.  The independent content is the last column:
5.86e-15 against the SHIPPED `Granet2DTransverseE.Lmat` on a different build
path.

**Answer to Measurement 3: the two discretizations are EQUIVALENT, not merely
both convergent.**  An OOP tile can therefore reduce to the in-plane path
exactly -- the Stage-A G1-style reduction gate is available (bit identity of
the operators when the tensor's OOP entries are zero, and 1e-15 on the
observables through two different mode solvers).

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

A mis-classified growing mode does not show at one depth -- it shows as
`exp(+|Re lam| k0 L)`, so this is a DEPTH LADDER: 0.25, 1 and 3 wavelengths,
`px = py = 1.2 lam`, `n_sup = 1`, `n_sub = 1.5`.  The lossless tensor is
EXACTLY Hermitian (measured `|eps - eps^H| = 0.00e+00`), so its closure is a
two-sided claim.

`max fwd growth` is `max exp(-Re(lam_f) k0 L)` over the forward set: any value
above 1 means a growing mode was classified forward.  `flux gap` is
`[min flux over the forward set, max flux over the backward set]`, relative to
the largest modal flux.

### uniform lossless out-of-plane tensor -- closure is machine-exact

| cand | M | depth / lam | sum R | sum T | R+T-1 | max fwd growth | fwd/bwd | flux gap |
|---|---|---|---|---|---|---|---|---|
| \[normal\] (a) | 7 | 0.25 | 0.0453366 | 0.9546634 | **+1.0e-14** | 1.0000e+00 | 288/288 | -1.1e-14 / +1.1e-14 |
| (a) | 7 | 1.00 | 0.0416527 | 0.9583473 | **+3.7e-14** | 1.0000e+00 | 288/288 | -1.1e-14 / +1.1e-14 |
| (a) | 7 | 3.00 | 0.0501759 | 0.9498241 | **+1.2e-13** | 1.0000e+00 | 288/288 | -1.1e-14 / +1.1e-14 |
| (d) | 7 | 3.00 | 0.0501759 | 0.9498241 | +1.6e-13 | 1.0000e+00 | 288/288 | -3.4e-15 / +2.5e-15 |
| \[conical 25/40\] (a) | 6 | 3.00 | 0.0378712 | 0.9621288 | +3.1e-11 | 1.0000e+00 | 200/200 | -6.4e-15 / +6.5e-15 |
| (a) | 7 | 0.25 | 0.0430755 | 0.9569245 | **-1.2e-12** | 1.0000e+00 | 288/288 | -2.1e-14 / +8.6e-15 |
| (a) | 7 | 3.00 | 0.0378712 | 0.9621288 | **-1.1e-12** | 1.0000e+00 | 288/288 | -2.1e-14 / +8.6e-15 |
| (d) | 7 | 3.00 | 0.0378712 | 0.9621288 | -1.2e-12 | 1.0000e+00 | 288/288 | -2.5e-15 / +2.6e-15 |

### lossless PILLAR (a corner cell -- closure is discretization-limited)

| cand | M | depth / lam | R+T-1 | max fwd growth | fwd/bwd |
|---|---|---|---|---|---|
| \[normal\] (a) | 6 | 0.25 / 1 / 3 | +1.4e-06 / -2.5e-06 / -3.5e-06 | 1.0000e+00 | 200/200 |
| (a) | 7 | 0.25 / 1 / 3 | **-7.2e-08 / -2.4e-07 / -2.0e-07** | 1.0000e+00 | 288/288 |
| \[conical\] (a) | 6 | 0.25 / 1 / 3 | -1.2e-05 / +1.4e-05 / -7.1e-05 | 1.0000e+00 | 200/200 |
| (a) | 7 | 0.25 / 1 / 3 | **-7.2e-07 / +7.0e-07 / -2.7e-06** | 1.0000e+00 | 288/288 |
| (d) | 7 | 0.25 / 1 / 3 | -7.2e-07 / +7.0e-07 / -2.7e-06 | 1.0000e+00 | 288/288 |

The closure improves by one to two decades from M = 6 to M = 7 at every
depth, and it does NOT grow with depth -- a growing-mode leak would multiply
by `exp(2 * 3 * k0 * something)` between the 0.25 and the 3-wavelength row and
is not there.

### LOSSY pillar -- no unity is claimed, only boundedness

| cand | M | depth / lam | sum R | sum T | 1 - R - T (absorbed) | max fwd growth |
|---|---|---|---|---|---|---|
| (a) | 7 | 0.25 | 0.0353130 | 0.9224430 | 0.0422 | 9.7817e-01 |
| (a) | 7 | 1.00 | 0.0186733 | 0.7131744 | 0.2682 | 9.1549e-01 |
| (a) | 7 | 3.00 | 0.0141886 | 0.3398127 | 0.6460 | 7.6729e-01 |
| (d) | 7 | 3.00 | 0.0141886 | 0.3398127 | 0.6460 | 7.6729e-01 |

Absorption rises monotonically with depth and stays in `[0, 1]`; the forward
growth factor stays strictly below 1 at every depth.

**Reading.**  In EVERY row of this measurement the forward/backward split came
out exactly `2q^2 / 2q^2` and `max fwd growth = 1.0000e+00`.  The deep
evanescent modes that M1's census flagged as off-branch are split by their
DECAY SIGN (their flux is 1e-14 relative, i.e. noise of either sign -- visible
in the `flux gap` column, where the minimum forward flux is NEGATIVE at the
1e-14 level and the maximum backward flux is POSITIVE at the same level), and
that is precisely what `_select_forward_flux`'s deep-decay override is for.
The census is BOUNDED AND PRICED: the price is that the OOP path MUST use the
flux selector with the deep-decay override, not a bare `Re(gam)` or a bare
flux split.

---

## S11.  M7 -- cost, and the normal-incidence block structure

### T1  dimension and eig wall time (single-threaded, tilted uniaxial cell)

The in-plane E-form is a `2 q^2` generalized eig (`scipy.linalg.eig(L, G)`);
candidate (a) is a `4 q^2` HERMITIAN-whitened standard eig (Cholesky of the
block Gram, then `numpy.linalg.eig`); candidate (d) is a `6 q^2` QZ, because
its leading coefficient `P2` is singular.

| Nx | M | q | E-form dim / time | (a) dim / time | (d) dim / time | (a)/E-form | (d)/(a) |
|---|---|---|---|---|---|---|---|
| 2 | 5 | 8 | 128 / 0.03 s | 256 / 0.07 s | 384 / 0.57 s | 2.4x | 7.9x |
| 2 | 6 | 10 | 200 / 0.09 s | 400 / 0.20 s | 600 / 1.86 s | 2.2x | 9.3x |
| 2 | 7 | 12 | 288 / 0.26 s | 576 / 0.55 s | 864 / 6.86 s | 2.1x | 12.4x |
| 2 | 8 | 14 | 392 / 0.66 s | 784 / 1.20 s | 1176 / 17.47 s | 1.8x | 14.6x |
| 3 | 5 | 12 | 288 / 0.28 s | 576 / 0.56 s | 864 / 6.66 s | 2.0x | 11.9x |
| 3 | 6 | 15 | 450 / 0.96 s | 900 / 1.64 s | 1350 / 31.74 s | 1.7x | 19.3x |

**Candidate (a) costs 1.7 - 2.4x the in-plane path** -- far less than the
`(4/2)^3 = 8x` a naive dimension count predicts, because the in-plane path
pays a QZ (`eig(L, G)`) while (a) pays a Cholesky-whitened standard eig on a
matrix twice the size.  **Candidate (d) costs 8 - 19x candidate (a)**, and the
ratio GROWS with dimension (the QZ's constant against the standard eig's).
The plan's estimate of "~3.4x the eig of (a)" for (d) is an UNDER-estimate by
a factor of 2 to 6 at these sizes.

For context, the hybrid `pmm_jones_2d` OOP path on the same physical cell
takes 3.1 s at `n_orders = 7`, 8.7 s at 9 and 24.0 s at 11 (M5), while
candidate (a) resolves the same cell in 1.3 - 3.0 s at M = 7 - 8 with a result
that does not move with `n_orders`.

### T2  the normal-incidence anti-commuting involution

`_generator_block_eig` gets all `4N` eigenpairs from ONE `2N` eig when a
signed permutation `R = S (I4 (x) F)` ANTI-commutes with the generator, `F`
being the Fourier order flip and `S = diag(I, I, -I, -I)`.  The staggered
analogue of `F` is the EXACT parity of the modified-Legendre dofs: `x -> d - x`
maps segment `s -> N-1-s`, swaps the two half-hats and multiplies the
degree-`a` bubble by `(-1)^a`, so the continuous hats permute (node `k -> N-k`)
and the bubbles carry a sign -- a signed permutation, valid when `tau = 1`
(normal incidence).  Built explicitly and verified on the ASSEMBLED pencil:

| cell | Nx | M | parity-map residual | R^2 - I | `\|\|R A R + A\|\| / \|\|A\|\|` | `\|\|R B R - B\|\| / \|\|B\|\|` |
|---|---|---|---|---|---|---|
| uniform OOP | 2 | 6 | 2.2e-16 | 4.4e-16 | **2.80e-14** | 3.2e-16 |
| uniform OOP | 2 | 7 | 2.2e-16 | 4.4e-16 | **1.68e-14** | 2.1e-16 |
| uniform OOP | 4 | 5 | 3.3e-16 | 4.4e-16 | **2.60e-15** | 3.1e-16 |
| CENTRO-symmetric pillar OOP | 2 | 6 | 2.2e-16 | 4.4e-16 | **2.80e-14** | 3.2e-16 |
| CENTRO-symmetric pillar OOP | 4 | 5 | 3.3e-16 | 4.4e-16 | **2.99e-15** | 3.1e-16 |
| OFF-CENTRE pillar OOP | 2 | 6 | 2.2e-16 | 4.4e-16 | **7.28e-01** | 3.2e-16 |
| OFF-CENTRE pillar OOP | 2 | 7 | 2.2e-16 | 4.4e-16 | **5.76e-01** | 2.1e-16 |
| OFF-CENTRE pillar OOP | 4 | 5 | 3.3e-16 | 4.4e-16 | **3.59e-01** | 3.1e-16 |

and the involution splits into two sectors of exactly `2 q^2` (`288 / 288` at
`Nx = 2, M = 7`).

**The structure transplants.**  `R A R = -A` to 2.6e-15 .. 2.8e-14 and
`R B R = B` to 3e-16 on uniform and centro-symmetric cells, so at normal
incidence the `4 q^2` eig can be replaced by one `2 q^2` eig by exactly
`_generator_block_eig`'s algebra.  The OFF-CENTRE row is the two-sided half of
the claim: 0.36 - 0.73, i.e. the structure genuinely FAILS there -- which is
precisely the case that function's docstring already warns about ("a cell
whose permittivity is centro-symmetric but whose spectral-element WALLS are
not ... is refused here by measurement").  A library integration should keep
the same verify-then-use gate rather than trusting a geometry test.

---

## S12.  Scored against the plan's verdict rule

The plan's rule: *GO for a candidate only if 1, 2, 4, 6 pass at their derived
bars AND the spurious census is empty or provably benign
(bounded-and-priced); otherwise NO-GO with the failure mechanism pinned by
measurement.*

| gate | derived bar | candidate (a) | candidate (d) |
|---|---|---|---|
| **M1 dispersion** | the exact quartic roots of the resolved harmonics; the discriminating quantity is the sum of the (0,0) roots, whose wrong-sign value differs by 4.2e-02 | (0,0) residual **2.8e-14**, sum error **3.9e-14**; every harmonic converges spectrally | (0,0) residual **6.5e-15**, sum error **9.6e-15**; identical ladder | 
| **M2 Berreman** | the exact 4x4 oracle, closing to 1e-15 itself; the negative controls sit at 3.8e-05 .. 2.1e-03 | **dR 5.6e-16, dJones 1.6e-15** at M = 9 oblique; conical 3.2e-15 / 8.2e-15 at M = 7; order leak exactly 0 | **dR 1.0e-15, dJones 2.5e-15** at M = 8 conical |
| **M4 1-D per order** | the two 1-D engines' own spread, 2.4e-08 .. 5.6e-06 | **2.7e-06 / 9.3e-06** at M = 8 on the weak-contrast cell; 1.4e-05 on the corner cell (the shipped basis's documented corner cap), monotone | identical to (a) to the printed digits |
| **M6 cascade** | closure must be small AND not drift with depth; growth factor must not exceed 1 | closure **1e-13** (uniform) / **2.7e-06 at M=7** (corner pillar) at 0.25, 1 AND 3 wavelengths; growth **1.0000e+00** everywhere; split exactly 2q^2/2q^2 in every row | identical |
| **spurious census** | empty, or bounded and priced | **BENIGN**: everything off-branch is deep-evanescent (`min Re(lam) 5.75`, i.e. 10x the `_select_forward_flux` deep-decay bar) and flux-null at **2.1e-14** relative | **PRICED**: additionally `q^2` modes at `q = 0` carrying up to **1.6e-01** relative flux, which are neither flux-null nor decay-classified and MUST be removed by an explicit `abs(q)` filter before the split |
| **M3 in-plane reduction** (not a gate, but the plan asks) | equivalent vs merely convergent | **EQUIVALENT** (1e-11 spectra, 1e-15 observables) | **EQUIVALENT, and BIT-IDENTICAL at the operator level** |

**Candidate (a): GO.**  Every gate passes with 10 to 12 decades between the
measurement and its negative control, and the census is benign under
machinery the library already ships.

**Candidate (d): PARTIAL.**  It passes the same four gates with the same
numbers -- M1 T2b shows it IS candidate (a)'s spectrum plus a null space -- so
it is not wrong.  It is DOMINATED: `6 q^2` instead of `4 q^2`, a QZ instead of
a Hermitian-whitened standard eig, and a null space that must be filtered by
eigenvalue magnitude (a filter that would collide with a genuine
`abs(q) -> 0` at a Rayleigh cutoff, exactly where the shipped module already
warns).  Its one advantage -- the exact operator-level reduction to the
shipped E-form -- is available to (a) too at 1e-11, and (d) remains valuable
as the DERIVATION that proves the reduction (S3, M3 T3).

---

## S13.  Recommended integration route

**Formulation: candidate (a)** -- the first-order staggered generator on
`[E1; E2; G1; G2]`, `A x = q B x` with `B = blkdiag(M1, M2, M2, M1)` (S2).

**1. Assembly.**  `Granet2DTransverseE` already builds every ingredient; the
tensor path adds NINE eps-weighted masses (`A11..A33`, of which `A12`, `A21`
are Stage A's Eq.-40 mixed blocks and `A13`, `A23`, `A31`, `A32` are new) and
SIX eps-weighted div-D blocks (`K11..K23`, of which `K11`, `K22` are the
shipped `Kzt_E1`/`Kzt_E2`, `K12`, `K21` are Stage A's, and `K13`, `K23` are
new).  All fifteen are the SAME `_eps_dir` kron assembly with a different
component map -- `probe_common.StaggeredCell._assemble` is 20 lines and the
library version would be shorter still since the pieces exist.  Retain nothing
new (audit P3-37): the OOP blocks go into `A` and `B` and the locals die.

**2. Mode entry.**  A new `_region_modes_oop(solver)` returning the 6-tuple
`(Wf, Vf, lam_f, Wb, Vb, lam_b)` -- the same shape
`rcwa._core._layer_eigenmodes_tensor` returns on its OOP branch, and the same
shape the PMM slant path already consumes.  Inside: Cholesky-whiten `B`,
`numpy.linalg.eig`, then split.

**3. The split MUST be flux-based with the deep-decay override.**  Reuse
`rcwa._core._select_forward_flux` verbatim by feeding it the CHOLESKY-WHITENED
blocks `[L1 e1; L2 e2; L2 g1; L1 g2]` with `M1 = L1^H L1`, `M2 = L2^H L2`:
its plain harmonic sums then reproduce the Gram-weighted flux
`Sz = Im(e1^H M1 g2 - e2^H M2 g1)` EXACTLY (the PMM_ROADMAP C-FLUX rule --
`probe_common.split_forward`, 8 lines).  M6 measured this split at exactly
`2q^2 / 2q^2` in every one of its 72 rows, with `max exp(-Re(lam) k0 L) =
1.0000e+00` at depths up to 3 wavelengths.  A bare `Re(gam)` split or a bare
flux split will NOT do: the off-branch modes have 1e-14 relative flux of
RANDOM SIGN, and it is the `|Re gam| > 0.5` override that classifies them
(they sit at `Re(lam) >= 5.75`, ten times the bar).

**4. Cascade.**  `_modes_to_M` + `_interface_smatrix_general` +
`_propagation_smatrix_general` + the shipped `_redheffer_star`, with the
ISOTROPIC half-spaces entering as `[[W, W], [V, -V]]`.  Measured: the
generalized interface reproduces the shipped square `_interface_smatrix` to
**7.9e-14 (S11) / 7.7e-14 (S21)** on an isotropic pair at normal and
**1.0e-13 / 9.3e-14** at conical, so mixed isotropic / OOP stacks cost nothing
in accuracy.

**5. The H-partner.**  Use the STRONG rows `G1 = D2 E3 - i q E2` (in V2),
`G2 = i q E1 - D1 E3` (in V1) rather than Eq. 25 -- simpler, and it is the
same object: M0.4 measured `V_shipped = -i * V_strong` to a spread of
**6.5e-12** over 200 modes, a global constant that cancels in every S-matrix.

**6. The sign.**  Flip `Basis1D`'s `tau` to `exp(+i alpha0 p)` AND
`_stag_fourier_projection`'s kernel together (S0), so the transverse
derivative and the longitudinal `exp(+i gamma z)` share one handedness and the
OOP blocks carry no compensating sign.  Gate it with the M1 T3
discriminator (the sum of the four (0,0) roots; wrong sign = 4.2e-02 against a
reference of 3.9e-14) -- four lines, and it is the ONLY thing standing between
this and a seventh copy of the factor-i defect.

**7. Guards.**  `e33 != 0` (the `A33` solve; mirror `_require_nonzero_ezz`),
square grid, and the existing grazing / Rayleigh-cutoff warning -- the latter
matters MORE here, because a `q -> 0` mode is what the (d) null filter would
collide with and what makes the `V ~ 1/gamma` amplification worst.

**8. Cost.**  1.7 - 2.4x the in-plane path's eig (M7 T1), and the far-field,
the Redheffer algebra, the cache keying and `layer_absorption` all carry over
unchanged.  At NORMAL incidence on a centro-symmetric cell the parity-times-
sign involution holds on the assembled generator (M7 T2), so the `4 q^2` eig
can be replaced by one `2 q^2` eig with
`rcwa._core._generator_block_eig`'s algebra -- an optional accelerator, gated
the same verify-then-use way.

**Gates the integration should carry** (all measured here, so their bars are
derived): M1 T3 sign arbitration; M2 against `berreman_jones_1d` for a
uniform OOP layer at normal / oblique / conical, INCLUDING a non-reciprocal
tensor and a director azimuth different from the incidence azimuth (S6); M3's
bit-identity reduction to the scalar path; M4's per-order agreement with
`pmm_jones_1d` / `rcwa_jones_1d` plus the y-momentum zero; M6's depth ladder.

---

## S15.  What this leaves open

* **The transpose (non-reciprocal) placement is settled by M2 only.**  No
  dispersion measurement can see `e13 <-> e31` (S5 T3), so any library test
  that gates the OOP blocks MUST include a non-reciprocal Hermitian tensor
  compared against Berreman's FIELDS -- an energy or dispersion check will
  pass a swapped assembly.
* **The `tau` sign trap.**  The shipped module's `tau = exp(-i alpha0 p)` is
  unobservable in the isotropic / in-plane path and load-bearing in the OOP
  path.  Whichever way the integration goes, it needs a test that fails when
  the sign is flipped in ONE place -- the M1 T3 `sum of the (0,0) roots`
  discriminator is that test, and it is four lines.
* **Anisotropic HALF-SPACES** stay out of scope (the probe's half-spaces are
  isotropic, matching the hybrid's documented restriction).
* **Slant x out-of-plane** stays refused, as in `_layer_eigenmodes_tensor`.
* **The JAX twin**: the staggered path is NumPy-only today; nothing here
  changes that.
* **Non-square grids** (`Nx != Ny`) remain out of scope; the tensor blocks are
  kron products on a square staggered basis exactly as the scalar ones are.

---

## S14.  Files and commands

All scripts under `validation/probe_pmm2d_staggered_oop/`; the exact commands
are in its `README.md`.  Every script exports the OMP caps at import and
asserts `lumenairy.__file__` starts with the worktree path.

| file | measurement |
|---|---|
| `probe_common.py` | the prototype: the staggered basis copied verbatim from `twod_staggered.py`, the tensor-weighted Galerkin assembly, candidate (a) `generator_a` / `modes_a`, candidate (d) `pencil_d` / `modes_d`, the in-plane E-form `eform_operators` / `eform_modes`, `exact_kz_roots`, `modal_flux`, `split_forward`, the far-field projection, and `solve_slab` |
| `m0_convention.py` | S4 |
| `m1_dispersion.py` | S5 |
| `m2_berreman.py` | S6 |
| `m3_inplane_limit.py` | S7 |
| `m4_stripe_1d.py` | S8 |
| `m5_pillar_2d.py` | S9 |
| `m6_cascade.py` | S10 |
| `m7_cost.py` | S11 |
| `run_all.sh` | driver; logs to `logs/`, JSON to `results/` |

```bash
cd /c/tmp/lum_aniso_oop
export PYTHONPATH=/c/tmp/lum_aniso_oop
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
bash validation/probe_pmm2d_staggered_oop/run_all.sh m0 m1 m2 m3 m4 m5 m6 m7
```
