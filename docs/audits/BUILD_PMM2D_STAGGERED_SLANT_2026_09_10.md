# BUILD -- a NATIVE constant-shear SLANT for the PURE staggered 2-D PMM (roadmap Phase D)

**Date:** 2026-09-10 - **Branch:** `feat/pmm2d-staggered-slant` off the
integration branch `wave2/pmm2d` (`9203843` = 5.43.0 + the Wood-list
unification + magnetic anisotropy + the out-of-plane parity accelerator;
`wave2/pmm2d` merged back in at `99b4377`, which is what brought the prototype
into the tree). **Worktree:** `C:/tmp/lum_slant`.
**Spec:** `docs/audits/EXPERIMENT_PMM2D_STAGGERED_SLANT_2026_09_10.md`
(the formulation, the GO decision, the integration route and its gates B1-B11).
**Prototype:** `validation/probe_pmm2d_staggered_slant/`.
**Binding law:** `docs/TESTING_STANDARDS.md`.

Tags: **[A]** analysis, no run - **[M]** measured on BOTH builds in S0 -
**[H]** hypothesis consistent with the evidence but not established.

---

## VERDICT UP FRONT -- **SHIPPED**, and one of the spec's premises is REFUTED

The transplant is exact: the shipped assembly's `(A, B)` is **BIT-IDENTICAL to
the prototype's** -- 0 differing bytes over 18 cell x slant x mount
combinations -- once the public-to-internal sign relation is applied. Every
gate B1-B11 passes on two builds, and two things the spec did not have came out
of the build:

1. **The frame-anchor phase's sign was wrong on the first write**, because the
   spec states the formula in the INTERNAL shear `t` while the public API takes
   `slant = -t`. The two-sided B3 gate caught it immediately (the shipped arm
   read `2.9e-01 .. 1.3e+00`, i.e. *worse than no correction*, which is exactly
   the signature the spec predicts for the wrong arm). Fixed; the shipped arm
   now reads `9.9e-08 .. 2.6e-05`.
2. **The parity accelerator's structural residual does NOT catch a shear.**
   The spec's instruction assumed it would fire "decades above
   `_STAG_BLOCK_TOL`". Measured, on a centro-symmetric slanted cell at normal
   incidence it reads `2.31e-15` against a `1e-10` bar -- **five decades
   BELOW** -- and, forced onto that pencil, the reduction reproduces the dense
   spectrum to `1.0e-12` and satisfies the original pencil to `2.8e-14`. `R` is
   a 180-degree ROTATION about `z`, not a mirror, and a rotation carries the
   sheared cell's covariant tensor AND its slant vector consistently, so the
   `+/-q` pairing survives the shear. **The explicit refusal is therefore the
   only gate there is**, and it is shipped as deliberate conservatism, not as a
   correctness fix. The library comment says so, in those words.

| gate | claim | verdict (WIN / WSL) |
|---|---|---|
| **B1** | `slant = 0` is BYTE-IDENTICAL, pencil and end to end | **PASS** -- 5 spellings, 0 differing bytes; and the slanted pencil is bit-identical to the prototype's |
| **B2** | a uniform layer at any slant is a no-op | **PASS** -- worst of 60 rows `4.10e-06` (M=5, both builds), ladder `1.34e-04 -> 2.03e-12` over M 4..8 |
| **B3** | the frame-anchor phase, both signs | **PASS two-sided** -- shipped `9.86e-08 .. 2.64e-05`, none `1.43e-01 .. 7.42e-01`, conjugate `2.86e-01 .. 1.33e+00` (~2x "none") |
| **B4** | sheared-frame dispersion == exact roots, + both ablations | **PASS** -- physical arm `1.4e-14 .. 6.2e-14` (WIN) / `2.5e-14 .. 3.7e-14` (WSL); wrong arms `3.5e-02 .. 1.1e-01`; six-blocks-removed `5.5e-02 .. 9.9e-02`; congruence-removed `7.4e-02 .. 8.8e-02` |
| **B5** | y-uniform stripe == `pmm_efficiency_1d_slanted` per order | **PASS, sign pinned** -- TE `2.65e-07` at M=8 at slants 0/10/20/35 (vertical control `1.71e-07`); wrong sign `4.15e-01` |
| **B6** | slant x OUT-OF-PLANE == `pmm_jones_1d_slanted` | **PASS + coverage gap closed** -- Ex `2.76e-05` slanted vs `2.68e-05` vertical control; the hybrid RAISES on the combination |
| **B7** | census: split exactly `2q^2/2q^2` | **PASS** -- exact in all 17 rows to slant 60 deg; `min Re(lam_f) >= -1.9e-14` lossless, `+3.11e-02` lossy; `max|q|` 9.56 -> 14.42 (1.51x) |
| **B8** | closure does not run away; forward growth == 1 | **PASS** -- growth `1.0000e+00` on every lossless row at three depths; `3.17e-01` lossy |
| **B9** | one layer of `d` == two of `d/2` | **PASS** -- worst `1.67e-15` (WIN) / `1.47e-15` (WSL) |
| **B10** | NO-FLOOR survives | **PASS** -- `1.55e-15` normal, `2.76e-06` conical |
| **B11** | the parity accelerator refuses; the refusals fire | **PASS** -- and see the refuted premise above |
| **M4** | a genuinely 2-D slanted pillar, three ways | **PASS** -- the INDEPENDENT hybrid walks toward the pure answer `2.69e-02 -> 1.23e-02` with the correct sign and does NOT with the wrong one (`2.84e-01 -> 2.79e-01`); a pure staircase converges only marching WITH the slant |
| **COST** | the slant is free | **PASS** -- `0.87x .. 1.07x` the vertical out-of-plane region solve |

---

## 0. Build pin -- TWO builds

`TESTING_STANDARDS.md` rule 5 forbids a bar without a measured cross-build gap.
The prototype campaign was single-build and explicitly said so; every bar in
`tests/unit/test_pmm2d_staggered_slant.py` is derived from the two builds below.

| | **WIN** | **WSL** |
|---|---|---|
| OS | Windows 11, tesla-ryzen | Ubuntu (WSL2) on the same host |
| interpreter | CPython 3.14.6 | CPython 3.12.3 |
| numpy / scipy | 2.4.4 / 1.17.1 | 2.4.6 / 1.17.1 |
| BLAS | scipy-openblas | scipy-openblas 0.3.31.188.0, USE64BITINT, SkylakeX |
| thread caps | `OMP` = `OPENBLAS` = `MKL` = 1 | `OMP` = `OPENBLAS` = 1 |

`lumenairy.__file__` is asserted to be inside the worktree on import by
`validation/probe_pmm2d_staggered_slant_build/_lib.py`; every probe writes
`results/<name>_<tag>.json` with `SLANT_BUILD_TAG` in `{win, wsl}`.

**The cross-build spread, in one sentence:** every quantity in this document
that is not at machine precision agrees between the two builds **to three
significant digits or better** -- the residuals here are deterministic
discretization, not round-off. The only quantities that move are the
machine-precision ones (`1.7e-14` vs `1.9e-14`, `1.55e-15` vs `9.4e-16`), and
every bar sits decades from both.

---

## 1. What shipped

Two files, ~150 lines of library code, no new module, no new basis function, no
new cascade, no new far field.

### 1.1 `lumenairy/elements/pmm/twod_staggered.py`

* **`_slant_congruence(eps33, tx, ty)`** -- the pointwise
  `eps^{lm} = A^-1 eps_lab A^-T` with `A = [[1,0,tx],[0,1,ty],[0,0,1]]`.
  `det A = 1` exactly at any slant, so `sqrt(g) = 1` and `eps^33 = eps_zz` is
  unchanged. `t = 0` returns the input object.
* **`_slant_rot_gauge(eps33, tx, ty)`** -- the congruence taken in the
  `_OOP_ROT_SIGN` gauge, applying the 180-degree rotation to BOTH the four
  out-of-plane tensor entries AND the slant vector. The consistency identity
  `eps^{lm}(R eps R, -t) = R eps^{lm}(eps, t) R` is what makes the two one
  operation; getting the `t` half wrong is the `a-` arm of B4
  (`5.8e-02 .. 1.1e-01`).
* **`Granet2DTransverseE(..., slant=None)`** -- normalises through the SAME
  `_norm_slant_pair` helper the hybrid and the 1-D entries use (so `None`,
  `0.0`, a scalar `t_x`, a 2-sequence and a numpy pair are one contract),
  promotes a scalar cell to `e * I`, gates `e33 != 0`, applies the rotated
  congruence, sets `offplane = True` unconditionally, records
  `self.slant` (public), `self.slanted`, `self._slant_rot` (internal, rotated)
  and `self.eps_lab`, and refuses `mu_cell`.
* **`_assemble_oop`** -- `rot` becomes `1.0` when the cell arrived pre-rotated
  (applying it twice would undo it), and gains SIX guarded blocks:

  | block | meaning | built from |
  |---|---|---|
  | `<V1\|Vw>` | `+i t_y G^3`, row 0 | `kron(mass(til,B)_y, Mbb_x)` |
  | `<V2\|Vw>` | `-i t_x G^3`, row 1 | `kron(Mbb_y, mass(til,B)_x)` |
  | `<V2\|D1\|V2>` | `t_x G_1` in `G_3cov`, row 2 | `kron(Mbb_y, Ctt_x/k0)` |
  | `<V2\|D1\|V1>` | `t_y G_2` in `G_3cov`, row 2 | `kron(mass(B,til)_y, -(dbt_x)^H)` |
  | `<V1\|D2\|V2>` | `t_x G_1` in `G_3cov`, row 3 | `kron(-(dbt_y)^H, mass(B,til)_x)` |
  | `<V1\|D2\|V1>` | `t_y G_2` in `G_3cov`, row 3 | `kron(Ctt_y/k0, Mbb_x)` |

  `-(dbt)^H` is the integration-by-parts identity the shipped assembly already
  uses for `CwE1`/`CwE2`, and it is the distributionally exact
  `<Btilde | d B>`. The element-wise `b.mixed(b.Btilde, b.B)` -- i.e. the
  `self.Ctb_*` already sitting in `_axis_mats` -- is NOT that matrix: it drops
  the jump deltas. **The build used `-(dbt)^H`, and the bit-identity against
  the prototype (S2) is the check that it did.**
* **`_region_modes_oop`** -- UNCHANGED, exactly as the spec requires.
* **`_stag_parity_gauge`** -- REFUSES a slanted solver (S5).
* **`pmm_jones_2d_staggered(..., slant=None)`** -- forwards.
* **`pmm_efficiency_2d_staggered(..., slant=None)`** -- accepts the keyword
  ONLY to raise `NotImplementedError`. A sheared scalar cell is an
  out-of-plane cell in the frame (`eps^{13} = -t_x eps`), so the two incident
  polarizations mix and that entry's SINGLE-polarization efficiencies are not
  what it would be returning -- the same argument it already uses to refuse a
  `(Nx, Ny, 3, 3)` cell. Accepting-and-raising is the loud form: a keyword the
  entry silently dropped would be a silent-wrong.

### 1.2 `lumenairy/elements/pmm/stack2d_pure.py`

* **`add_layer(..., slant=None)`** on every layer kind; `mu` + slant raises at
  `add_layer`.
* Per-layer routing: a slanted layer takes its own region eig with
  `slant=` passed through, and a slanted **UNIFORM** layer is materialised as a
  constant cell on the union grid (it cannot ride the shared eps-free geometric
  eig, because in the frame it is an out-of-plane region). It is still a
  physical no-op -- that is B2.
* `any_oop |= slanted`, so the stack takes the generalized cascade. **No
  cascade code changed.**
* The eig-cache key carries the slant.
* **The frame-anchor phase**, in `solve`, on the TRANSMITTED amplitudes only.
* **`_check_stack_slant`** and **`_layer_is_patterned`** -- the refusals (S4).
* `retain_internal` + slant raises.

### 1.3 The SIGN, as shipped

```
    slant_public  =  -t_internal          (the t of x = u + t w)
    slant_public  =  +tan(slant_angle_1D) =  t_hybrid_public
```

pinned FOUR times in this build, each against an independently validated
engine: B5 against `pmm_efficiency_1d_slanted` (`1.22e-03` vs `4.15e-01`), M4
against the hybrid metric (`1.23e-02` vs `2.79e-01`, and only the right arm
improves with truncation), M4c against a pure-solver staircase (converges only
marching WITH the slant), and B4 against the exact quartic roots (`1.4e-14` vs
`3.5e-02`).

---

## 2. [M] B1 -- the reduction gates

`g1_hash_null.py` (B1 block), `cmp_proto` (scratch).

**B1a -- the pencil.** `slant` in `{None, 0.0, (0,0), [0,0], np.zeros(2)}` on a
scalar, an in-plane tensor and an out-of-plane tensor cell, at normal and
conical incidence: **one sha256 per row, on both builds.** End to end likewise.

**B1b -- against the PROTOTYPE.** The shipped `Granet2DTransverseE(..., slant=s)`
pencil vs `slant_lib.SlantSolver(..., slant=-s)`:

| cell | slants | mounts | differing bytes `A` / `B` | max abs diff |
|---|---|---|---|---|
| scalar, in-plane tensor, out-of-plane tensor | `(0.3,0)`, `(0,0.45)`, `(0.5,-0.25)` | normal, conical | **0 / 0** in all 18 rows | **0.0** |

The transplant is exact, and the public-to-internal sign relation is what makes
it exact.

**B1c -- the algebra.** `A^-1 (A eps A^-T)^-1 ...` round-trip `<= 1e-15 |eps|`;
`cov(eps, 0, 0) is eps`; `det A = 1.0` exactly; `eps^33` unchanged bit for bit
at every slant (`A^-1`'s third row is `(0, 0, 1)`, so the products with zero
are exact).

---

## 3. [M] B2 / B3 -- the null test and the frame anchor

`g1_hash_null.py`. Five tensors (isotropic, in-plane uniaxial, OUT-OF-PLANE
uniaxial, gyrotropic, lossy out-of-plane) x three mounts x four slants (x10,
x35, y35, diag35) = 60 rows at M = 5.

**Table B2 -- worst over all 60 rows** (reference: the same cell at slant 0):

| quantity | WIN | WSL |
|---|---|---|
| `R` per order | 1.588e-06 | 1.588e-06 |
| `T` per order | 1.057e-06 | 1.057e-06 |
| reflection Jones | 4.098e-06 | 4.098e-06 |
| non-(0,0) order leak | 1.766e-07 | 1.766e-07 |
| worst at NORMAL incidence (`R`) | 5.29e-15 | 2.97e-15 |
| leak at NORMAL incidence | 1.95e-28 | 1.59e-28 |

At normal incidence `k_t = 0` kills the shear's coupling, so the oblique /
conical rows are the informative ones. A `y`-only shear at `phi = 0` is
likewise a no-op -- shearing along a direction that carries no transverse
momentum -- which has no 1-D analogue and tests the VECTOR structure of the six
blocks.

**Table B2c -- the residual is DISCRETIZATION, and it is SPECTRAL** (worst null
row: isotropic uniform, oblique 25, slant 35 deg; Jones columns carry the
frame-anchor phase):

| M | dim | dR | dT | dJones(R) |
|---|---|---|---|---|
| 4 | 144 | 1.39e-05 | 2.47e-04 | 1.34e-04 |
| 5 | 256 | 8.68e-07 | 3.46e-07 | 2.95e-06 |
| 6 | 400 | 1.32e-08 | 1.04e-08 | 3.93e-08 |
| 7 | 576 | 1.15e-10 | 1.12e-10 | 3.39e-10 |
| 8 | 784 | **6.91e-13** (WIN) / 6.89e-13 (WSL) | 6.91e-13 / 6.92e-13 | **2.03e-12** / 2.02e-12 |

Eight decades over four steps of `M`; the two builds agree to three digits at
every rung.

**Table B3 -- the frame-anchor phase, three arms** (uniform null, transmission
Jones, M = 5; the two builds are identical to every printed digit):

| tensor | mount | slant | NO correction | **shipped `exp(-i a.t d)`** | conjugate `+i` |
|---|---|---|---|---|---|
| isotropic | oblique 25 | x10 | 1.93e-01 | **8.58e-07** | 3.84e-01 |
| isotropic | oblique 25 | x35 | 7.42e-01 | **1.66e-05** | 1.33e+00 |
| isotropic | oblique 25 | diag35 | 5.34e-01 | **7.90e-06** | 1.01e+00 |
| isotropic | conical 25/40 | x10 | 1.44e-01 | **9.86e-08** | 2.87e-01 |
| isotropic | conical 25/40 | x35 | 5.61e-01 | **1.91e-06** | 1.05e+00 |
| isotropic | conical 25/40 | diag35 | 7.19e-01 | **9.10e-07** | 1.29e+00 |
| out-of-plane uniaxial | oblique 25 | x35 | 7.38e-01 | **2.64e-05** | 1.32e+00 |
| out-of-plane uniaxial | conical 25/40 | diag35 | 7.15e-01 | **1.59e-06** | 1.28e+00 |

The conjugate arm is about **twice as wrong as doing nothing** in every row --
the correction is not a fudge that could absorb an arbitrary residual. And R,
T and the REFLECTION Jones are exact on every one of these rows without it,
which is the whole reason omission is silent.

**[M] The first implementation had this sign inverted** (it summed the PUBLIC
slant where the formula takes the internal shear) and the "shipped" column read
`2.87e-01 .. 1.33e+00` while the `+i` column read `9.86e-08 .. 2.64e-05`. The
gate was written before the code was believed, and it is what found it.

---

## 4. [M] B4 -- sheared-frame dispersion

`g2_dispersion.py`. `M = 8`, `(2,2)` grid, `px = py = 0.9`, conical Bloch shift
`(kx0, ky0) = (0.25, 0.18)`. For a UNIFORM cell the generator's eigenvalues
must be `kz_root(eps_lab; u, v) + t . (u, v)` with `t` the INTERNAL shear and
`kz_root` the four EXACT roots in exact polynomial arithmetic.

**Table B4a -- the four gauge / shift arms** (`a+/a-` = transverse gauge
`+alpha`/`-alpha`; `s+/s-` = shift `+t.alpha`/`-t.alpha`, `t` internal):

| tensor | slant | **a+s+** (WIN / WSL) | a+s- | a-s+ | a-s- |
|---|---|---|---|---|---|
| uniaxial tilt35 azim25 | none | **6.22e-14 / 3.67e-14** | same | 6.79e-02 | 6.79e-02 |
| uniaxial | x20 | **2.58e-14 / 3.35e-14** | 5.87e-02 | 5.81e-02 | 3.53e-02 |
| uniaxial | diag35 | **1.38e-14 / 2.46e-14** | 1.12e-01 | 1.12e-01 | 6.79e-02 |
| non-reciprocal `e13 != e31` | x20 | **4.31e-14 / 3.05e-14** | 8.02e-02 | 1.07e-01 | 4.88e-02 |
| non-reciprocal | diag35 | **1.75e-14 / 2.82e-14** | 1.13e-01 | 1.07e-01 | 7.40e-02 |
| isotropic 2.25 | diag35 | **2.26e-08 / 2.15e-08** | 1.12e-01 | 1.12e-01 | 2.26e-08 |

The isotropic row sits at `sqrt(eps_machine)` **at every slant, including
zero** -- the double-root conditioning of an isotropic tensor, not a slant
effect -- and its `a-s-` arm passes only because an isotropic tensor IS
`alpha -> -alpha` symmetric, i.e. it is the same solution relabelled, not a
second one. Both facts are handled by CONSTRUCTION in the test, not by a
tolerance.

**Table B4b -- the two ABLATIONS, each driven through the shipped assembly**
(identical on both builds to three digits):

| tensor | slant | full | six blocks REMOVED | congruence REMOVED |
|---|---|---|---|---|
| uniaxial | x20 | **2.6e-14 / 3.4e-14** | 6.39e-02 | 7.37e-02 |
| uniaxial | diag35 | **1.4e-14 / 2.5e-14** | 9.40e-02 | 8.60e-02 |
| non-reciprocal | x20 | **4.3e-14 / 3.1e-14** | 5.47e-02 | 8.28e-02 |
| non-reciprocal | diag35 | **1.8e-14 / 2.8e-14** | 9.85e-02 | 8.80e-02 |
| isotropic | diag35 | **2.3e-08 / 2.2e-08** | 1.58e-03 | 1.58e-03 |

Both halves of the formulation are load-bearing, and each is measured.

**Table B4c -- the sum-of-roots discriminator** (uniaxial, conical 25/40): the
fundamental's four roots sum to `4 t.alpha` above their vertical value.
Generator / exact -- `none`: `-0.067930 / -0.067930`; `x20`:
`-0.431900 / -0.431900`; `diag35`: `-0.919539 / -0.919539`. Identical on both
builds; the slant moves the sum by 0.36 and 0.85.

**Table B4d -- the M-ladder** (uniaxial, diag35):

| M | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|
| WIN | 3.55e-07 | 6.35e-10 | 7.56e-13 | 1.35e-14 | 1.38e-14 |
| WSL | 3.55e-07 | 6.35e-10 | 7.66e-13 | 2.18e-14 | 2.46e-14 |

Spectral to the eigenvalue-conditioning floor; the last two rungs are where the
two builds separate, which is why the test's bar is at M = 6.

---

## 5. [M] B5 / B6 -- against the 1-D oracles

`g3_stripe.py`, `g7_testfixtures.py`. Cell: `px = py = 0.75 lam`, binary
x-grating `n = 2/1`, duty 0.5, `depth = 0.30 lam`, `n_sub = 1.5`, `wl = 1`.
Comparison is per order, on R and T; row 0 = incident `Ex` (= TM at `phi = 0`),
row 1 = `Ey` (= TE).

**Table B5a -- the SIGN, arbitrated by the oracle** (`phi = 35 deg`, `M = 7`;
"ctrl" is the same 2-D cell at slant 0 against `pmm_efficiency_1d`). Identical
on both builds:

| mount | pol | vertical ctrl | `slant = +tan` | `slant = -tan` |
|---|---|---|---|---|
| normal | TE | 1.07e-07 | **4.28e-06** | 4.15e-01 |
| normal | TM | 1.14e-03 | **1.22e-03** | 2.24e-01 |
| oblique 25 | TE | 1.79e-06 | **1.44e-06** | 2.02e-02 |
| oblique 25 | TM | 1.25e-04 | **2.24e-04** | 1.27e-01 |

A slanted grating is not x-mirror symmetric, so `R_{+1} != R_{-1}` and the sign
is observable PER ORDER -- which is why this, not an energy check, is where it
is pinned. **The wrong-sign arm conserves energy to `3.850e-07` -- and so does the RIGHT
one, the same number to four digits** -- while the two sit `4.2e-01` apart per
order. Energy carries literally ZERO information about the sign on this cell.
That is asserted in the test file (`test_b5_wrong_slant_sign_conserves_energy`),
so the lossless trap is demonstrated rather than merely written down.

**Table B5b -- the per-order ladder at M = 8** (winning sign; `yleak` = worst
amplitude into any `n != 0` order):

| phi | mount | TE | TM | y-leak | closure |
|---|---|---|---|---|---|
| 0 | normal | 1.07e-07 | 4.90e-04 | 5.3e-26 | 5.24e-09 |
| 10 | normal | 1.15e-07 | 4.88e-04 | 8.8e-26 | 4.42e-09 |
| 20 | normal | 2.00e-07 | 7.69e-04 | 9.9e-26 | 3.39e-09 |
| 35 | normal | 2.65e-07 | 1.27e-03 | 8.7e-26 | 7.66e-09 |
| 0 | oblique 25 | 1.71e-07 | 8.35e-05 | 1.2e-26 | 5.21e-11 |
| 35 | oblique 25 | 2.11e-07 | 1.48e-04 | 6.8e-26 | 7.41e-11 |

TE reaches `1e-07` at EVERY slant, tracking the vertical control;
y-momentum is conserved to `1e-26`. **The TM column is ORACLE-limited**: the
1-D scalar slant oracle's own per-order drift at this degree is itself
`~6.5e-04` (the documented wall-normal floor of the 1-D metric route), which is
why the test's TM bar is stated as tracking the VERTICAL control rather than as
an accuracy claim.

**Table B6 -- slant x OUT-OF-PLANE vs `pmm_jones_1d_slanted`** (ridge =
`uniaxial(1.5, 1.7, tilt 35, azim 25)`, groove = air, `M = 8`, degree-30
oracle):

| phi | mount | incident Ex | incident Ey | oracle's own drift | y-leak | closure |
|---|---|---|---|---|---|---|
| 0 | normal | 2.68e-05 | 1.23e-06 | 1.9e-06 | 9.7e-26 | 4.97e-10 |
| 20 | normal | 2.71e-05 | 1.75e-06 | 3.3e-06 | 9.8e-26 | 9.11e-10 |
| 35 | normal | **1.66e-05** | 1.96e-06 | 4.0e-06 | 9.0e-26 | 2.30e-09 |
| 0 | oblique 25 | 1.91e-05 | 9.35e-07 | 1.6e-06 | 5.3e-26 | 4.10e-11 |
| 35 | oblique 25 | **2.76e-05** | 6.93e-07 | 9.1e-06 | 5.2e-26 | 8.56e-11 |

The slanted rows are as good as the vertical row -- better at 35 degrees normal
-- and `PMM2DStackHybrid` raises on the same combination:

```
NotImplementedError: _layer_eigenmodes_tensor: a SLANTED layer with OUT-OF-PLANE
coupling (eps_xz/yz/zx/zy) is not supported.  The 2-D slant metric is validated
for IN-PLANE tensors only ...
```

**This build closes that gap.**

---

## 6. [M] B7-B10 -- census, cascade, split, no floor

`g4_census.py`. `px = py = 1.2 lam`, conical 20/35, `M = 6`, `(2,2)` grid,
`dim = 4q^2 = 400`.

**Table B7 -- CENSUS** (17 rows; the VERTICAL control is the out-of-plane
tensor cell, because a vertical SCALAR cell has no `4 q^2` pencil at all -- it
runs the `2 q^2` in-plane path, which is the B1 bit-identity statement):

| cell | slant | split (want 200/200) | `min Re(lam_f)` | `max abs(q)` | above-band |
|---|---|---|---|---|---|
| out-of-plane uniaxial pillar | vertical | **200/200** | -9.0e-15 | 9.56 | 148 |
| scalar pillar `eps` 4 | x20 | **200/200** | -5.1e-15 | 9.73 | 102 |
| scalar pillar | x45 | **200/200** | -2.9e-15 | 11.23 | 128 |
| scalar pillar | diag45 | **200/200** | -5.4e-15 | 11.44 | 130 |
| scalar pillar | x60 | **200/200** | -9.4e-15 | 14.20 | 186 |
| out-of-plane pillar | x60 | **200/200** | -2.3e-15 | 14.42 | 234 |
| high contrast `eps` 12 | x60 | **200/200** | -1.2e-14 | 13.41 | 34 |
| LOSSY pillar `4 + 0.6i` | x20 | **200/200** | **+7.5e-02** | 9.74 | 101 |
| LOSSY pillar | x60 | **200/200** | **+3.1e-02** | 14.22 | 186 |

* the split is **exactly `2q^2 / 2q^2` in all 17 rows, BEFORE the selector's
  defensive rebalance** (which would make a misclassification silent), on both
  builds;
* `min Re(lam_f) >= -1.86e-14` on every lossless row and strictly POSITIVE on
  every lossy one;
* `max abs(q)` grows `9.56 -> 14.42` between vertical and 60 degrees, a factor
  **1.51**, i.e. `sec(60) = 2` bounded -- **not** the `~210` a from-scratch
  convection form produces in 1-D. The above-band population is the polynomial
  basis's own unresolved harmonics (already 148 at slant 0 on the out-of-plane
  cell), growing with the same `sec` factor.

**Table B8 -- CASCADE vs DEPTH** (`M = 5`, 0.25 / 1 / 3 wavelengths):

| cell | slant | mount | 0.25 lam | 1 lam | 3 lam | max fwd growth |
|---|---|---|---|---|---|---|
| scalar | vertical | conical | 1.63e-03 | 7.54e-03 | 1.48e-02 | **1.0000e+00** |
| scalar | x36.9 | conical | 1.90e-03 | 1.69e-02 | 1.43e-02 | **1.0000e+00** |
| scalar | diag45 | normal | 8.22e-05 | 2.27e-04 | 2.86e-04 | **1.0000e+00** |
| out-of-plane | x36.9 | normal | 5.91e-05 | 1.79e-04 | 1.91e-04 | **1.0000e+00** |
| out-of-plane | diag45 | conical | 1.35e-04 | 3.53e-04 | 2.59e-04 | **1.0000e+00** |
| LOSSY | x36.9 | normal | 0.134 | 0.639 | 0.937 | 3.17e-01 |

Forward growth is exactly `1.0000e+00` (the largest reading is `1 + 1.5e-13`,
i.e. round-off) on every lossless row and strictly below 1 on every lossy one.
The lossy rows' absorption rises monotonically and stays in `[0, 1]`.

**Table B9 -- LAYER SPLIT** (one slanted layer of `d` vs two of `d/2`):

| worst over the 6 rows | WIN | WSL |
|---|---|---|
| `max(dR, dT, dJones)` | **1.67e-15** | **1.47e-15** |

Machine precision: the propagator, the internal patterned-to-patterned
interface and the accumulated frame anchor compose exactly.

**Table B10 -- NO FLOOR** (`n_orders` 3 -> 5 -> 8):

| mount | WIN | WSL |
|---|---|---|
| normal | 1.55e-15 | 9.44e-16 |
| conical | 2.758e-06 | 2.758e-06 |

**[H]** the conical residue is the `_grazing_safe_wavelength` nudge, which is a
function of the ORDER SET and so moves slightly with `n_orders` -- it is absent
at normal incidence, where no order is near a cutoff, and the two builds agree
on it to seven digits (a deterministic nudge, not round-off). Either way it is
three decades below the Fourier hybrid's own `E_z`-rule spread of `7.7e-04`.

---

## 7. [M] B11 -- the parity accelerator, and a REFUTED premise

`g5_parity_refusals_wood.py`.

**Table B11a -- the structural residual does NOT catch a shear.**
Centro-symmetric out-of-plane cell (its own parity image on a mirror-symmetric
wall layout), NORMAL incidence, `M = 6`. The gauge is forced onto the REAL
slanted pencil through a GEOMETRY SHIM (see the note below):

| slant | shipped gauge | forced `max\|R A R + A\|/max\|A\|` | vs `_STAG_BLOCK_TOL` | forced reduction runs? | forced spectrum gap | forced pencil residual |
|---|---|---|---|---|---|---|
| vertical | given | 2.31e-15 | `1e-10` -- **PASSES** | yes | 7.4e-13 / 7.8e-13 | 2.4e-14 / 2.6e-14 |
| x35 | **REFUSED** | 2.31e-15 | **PASSES** | yes | 9.7e-13 / 9.5e-13 | 2.8e-14 / 2.5e-14 |
| diag35 | **REFUSED** | 2.31e-15 | **PASSES** | yes | 5.2e-13 / 5.2e-13 | 1.9e-14 / 2.8e-14 |
| centro SCALAR cell, x35 | **REFUSED** | 1.29e-15 | **PASSES** | yes | 7.5e-13 | 1.2e-13 |

**The spec's premise was that a shear breaks the mirror symmetry and the
structure test would fire "decades above the bar". It does not.** `R` is a
180-degree ROTATION about `z` (both axes flip), not a mirror; a rotation
carries the sheared cell's covariant tensor AND its slant vector consistently,
so the `+/-q` pairing the reduction rests on survives the shear -- and forced
onto the slanted pencil the reduction is, on these cells, CORRECT
(`2.8e-14` in the original pencil). So:

* the explicit refusal in `_stag_parity_gauge` is the **ONLY** gate;
* it costs an accelerator opportunity rather than covering a known wrong
  answer;
* it ships anyway, because the shear is a new geometry whose forward/backward
  split, gauge and reconstruction are validated only on the dense branch, and
  because the precedent for not letting a slanted answer ride on a tolerance is
  the hybrid's normal-incidence silent-wrong
  (`BUILD_PMM2D_SLANT_METRIC_2026_08_16.md` S8: its even-parity fold returned
  the VERTICAL answer, wrong by 2.5e-01 at 35 degrees, with energy conserved
  and nothing warned, while every OBLIQUE test passed);
* the library comment states all of this, so nothing false is shipped.
* **Open item:** re-enabling the reduction for slanted cells at normal
  incidence is a follow-up that needs its own two-sided gate.

**[M] A methodological trap, recorded because it wasted a measurement.**
Monkeypatching `twod_staggered._slant_is_zero` to force the gauge does NOT
work: the same name is what `Granet2DTransverseE.__init__` reads to decide
whether to apply the shear at all, so the patched arm silently builds the
VERTICAL pencil. The first "forced" run reported a `2.11e-02` disagreement
that was in fact the vertical-vs-slanted difference. The geometry shim
(`_GeometryShim` in the test file, `_Shim` in the probe) is the fix, and both
carry the story in their docstrings.

**Table B11a2 -- end to end** (identical on both builds): on the SLANTED cell
`symmetry='auto'` is **sha256-IDENTICAL** to `symmetry=False`; on the VERTICAL
control it is NOT (the reduction engages there, so the two take different code
paths -- which is what proves the slanted equality is not vacuous); and the
slanted answer differs from the vertical one by `dJones = 2.11e-02`, so the
cell can see a silently-dropped slant.

**Table B11b -- the refusals** (all fire, both builds):

| construction | raises |
|---|---|
| two PATTERNED layers at different slants | `NotImplementedError: ... MIXED SLANTS between PATTERNED layers ...` |
| a vertical uniform film BETWEEN two slanted patterned layers | `NotImplementedError: ... the layers ABOVE patterned layer 3 carry a MIX of vertical and slanted regions ...` |
| `mu` / `mu_cell` + slant | `NotImplementedError` (at `add_layer`) |
| `retain_internal=True` + slant | `NotImplementedError` (at `solve`) |
| `pmm_efficiency_2d_staggered(..., slant=)` | `NotImplementedError` pointing at `pmm_jones_2d_staggered` |
| `slant=(0.1, 0.2, 0.3)` | `ValueError` from the shared `_norm_slant_pair` |

and the three ACCEPTED shapes, which must not raise: ONE slanted patterned
layer among VERTICAL uniform films; TWO patterned layers at the SAME slant;
UNIFORM-only layers at different slants.

### 7.1 [A] The refusal rule, derived

Each slanted region is solved in its OWN frame, anchored at that layer's TOP
face, and the interface match is the IDENTITY (`det g = 1`, `w = z`, and the
covariant TANGENTIAL components equal the lab-Cartesian ones). So what the
cascade actually solves places each layer's cell displaced laterally by the
ACCUMULATED offset of everything above it, `Sh_i = sum_{j<i} t_j d_j`. Two
readings of that are exact, and each is measured:

* `Sh_i = 0` -- every layer above the pattern is vertical, so the cell sits
  where the caller put it. **The workhorse**, and the case B5/B6/M4 measure;
* `Sh_i = t_i Z_i` with every layer above at the SAME slant -- one global shear
  of the whole stack, the frame simply continuing. **The layer-split identity**,
  measured at `1.5e-15` (B9).

Anything else puts one nodal grid at a real lateral translation relative to
another, which a nodal SEM basis cannot represent exactly unless the offset is
a whole number of grid cells. `_check_stack_slant` admits exactly the two
readings and refuses the rest, naming the offending layer indices.

---

## 8. [M] The WOOD-ANOMALY decision (the spec's open [H])

**DECISION: a slanted layer contributes its LAB permittivities, exactly as a
vertical layer does. The covariant diagonal `eps (1 + t^2)` is NOT added.**

That is the reading which leaves the shipped `_wood_eps_reals` call sites
untouched, so a slant-0 layer is bit-identical by construction. It is also the
one the measurement supports.

**Table W -- a slanted scalar layer (`eps = 4`, `t = 0.6`) walked onto each
candidate cut-off** (identical on both builds):

| candidate | value | does the nudge fire? (lab list / covariant list) | the layer's actual `min abs(q)` |
|---|---|---|---|
| LAB `eps` | 4.0000 | **yes** / yes | **7.672e-03** -- nearly a null mode |
| covariant `e11 = eps (1 + t^2)` | 5.4400 | no / yes | **1.200e+00** -- not a cut-off at all |
| `abs(alpha)^2 + (t.alpha)^2` | 1.3600 | -- | 1.704e-01 |

Three reasons, in order of weight:

1. **[M] The lab value is the coincidence that matters.** At `wl = px sqrt(4)`
   the slanted layer's spectrum comes within `7.7e-03` of a null mode -- which
   is what crashes the interface S-matrix and what the nudge exists to avoid.
   At the covariant coincidence the layer's spectrum is two decades away:
   listing `eps (1 + t^2)` would nudge on a NON-EVENT.
2. **[A] The covariant diagonal is not the slanted layer's null condition
   anyway.** In the frame the mode condition is
   `kz_root(eps; alpha_m) = -t . alpha_m`, i.e.
   `eps_eff(m) = abs(alpha_m)^2 + (t . alpha_m)^2` -- ORDER-DEPENDENT, so no
   single scalar in a flat list expresses it. `eps (1 + t^2)` is the covariant
   tensor's largest diagonal entry, which is a statement about the operator's
   coefficients, not about where a mode goes grazing.
3. **[A] It keeps the nudge INVARIANT under adding a slant.** With the lab
   reading, sweeping `t` at fixed wavelength never changes which wavelength is
   solved; with the covariant reading it would jump discontinuously as
   `eps (1 + t^2)` crossed a `kt^2`, which is exactly the kind of
   discretization-visible discontinuity a parameter sweep must not have.

Over-listing is numerically inert off an EXACT coincidence (the guard's trigger
band is `abs(eps - kt^2) <= 1e-9`), so this is a robustness/consistency
decision rather than an accuracy one -- but it is a decision, and it is
measured.

---

## 9. [M] M4 -- a genuinely 2-D slanted pillar, three ways

`g6_cost_pillar.py` (full campaign sizes) and `g7_testfixtures.py` (the test's
own sizes). Pillar `[0.3, 0.6] x [0.3, 0.6]` in `px = py = 1.2 lam`, `eps`
4/1, `depth = 0.8 lam`, `t = 0.75` -- the cross-section translates by HALF a
period over the layer, so every staircase wall lands on the uniform grid.

**Table M4a -- the shipped metric layer is grid- and degree-invariant**
(vs `Nx = 8, M = 4`): `9.66e-05 .. 3.08e-03` across `(Nx, M)` in
`{(4,5), (4,6), (8,3), (12,3)}` -- two to three decades below everything it is
compared against.

**Table M4b -- the INDEPENDENT hybrid metric** (`PMM2DStackHybrid.add_layer(
slant=)`: a Fourier basis, a tensor fold and a LAB-CARTESIAN CONVECTION on its
4N generator -- nothing in common with this formulation but the physics):

| mount | `n_orders` | correct sign | wrong sign | the hybrid's OWN step |
|---|---|---|---|---|
| normal | 5 | 1.611e-02 | 2.222e-01 | -- |
| normal | 7 | 1.888e-02 | 2.106e-01 | 1.31e-02 / 1.16e-02 |
| normal | 9 | **6.773e-03** | 2.121e-01 | 1.26e-02 / 1.30e-02 |
| conical 20/35 | 5 | 3.893e-02 | 2.288e-01 | -- |
| conical 20/35 | 9 | **2.497e-02** | 2.535e-01 | 1.73e-02 / 2.65e-02 |

The correct sign ends BELOW the hybrid's own `n_orders` step -- the two engines
agree to within the hybrid's own convergence, which is the strongest statement
this comparison can make -- while the wrong sign does not move at all. That is
the two-sidedness: a truncation ladder separates a discretization gap from a
wrong structure.

**Table M4c -- a z-STAIRCASE from the PURE solver, with the marching DIRECTION
scanned** (reference = the metric layer on the SAME grid and degree, so the
discretization is common-mode; `Nx = 8`, `M = 3`):

| mount | direction | n=1 | n=2 | n=4 |
|---|---|---|---|---|
| normal | **WITH the slant** | 1.399e-01 | **4.670e-02** | **2.076e-02** |
| normal | AGAINST | 1.399e-01 | 2.064e-01 | 2.053e-01 |
| conical | **WITH the slant** | 2.127e-01 | **7.523e-02** | **6.175e-02** |
| conical | AGAINST | 2.127e-01 | 8.693e-02 | 1.962e-01 |

The correct direction converges toward the metric layer (3.0x then 2.2x at
normal) and the wrong one gets monotonically worse. At `n = 1` both directions
coincide, because the `n = 1` "staircase" IS the vertical pillar. These
numbers reproduce the prototype's to four digits (`1.399e-01 / 4.670e-02 /
2.076e-02`), which is an independent check that the transplant did not move the
physics.

**Why the ladder stops at `n = 4`, and why that is the finding.** `Basis1D`'s
segments are a `linspace`, so every slice must place its walls on ONE uniform
union grid of spacing `h = px/Nx`: the admissible slice counts are exactly the
DIVISORS of `S/h`, and the smallest non-zero per-slice step is `h` itself
(half the pillar width at `Nx = 8`). Refining the step means refining `Nx`,
which multiplies the region eig by `(Nx (M-1))^6`. **The practical statement is
not "one slanted solve replaces N slices" but the stronger one: at the slice
counts a given union grid admits, there may be no rung that reaches the slanted
answer at all.**

**Table M4d -- the test's own sizes** (a (3,3) grid, pillar = one grid cell,
`t = 1.0` so the walk is exactly two grid cells; identical on both builds):

| mount | hybrid `+t`: n=3/5/7 | hybrid `-t`: n=3/5/7 | staircase WITH: n=1/2 | AGAINST: n=1/2 | a VERTICAL pillar |
|---|---|---|---|---|---|
| normal | 2.688e-02 / 1.926e-02 / **1.234e-02** | 2.843e-01 / 2.856e-01 / 2.787e-01 | 3.698e-01 / **1.324e-01** | 3.698e-01 / 3.236e-01 | 3.753e-01 |
| conical | 3.924e-02 / 2.630e-02 / **1.787e-02** | 3.730e-01 / 3.756e-01 / 3.683e-01 | 4.869e-01 / **2.205e-01** | 4.869e-01 / 4.800e-01 | 4.626e-01 |

The slant is a `4e-01` effect on this cell and the two engines agree to
`1.2e-02`. The pure solver's own `M` self-move at these sizes is `5.0e-03 ..
6.0e-03` -- the SAME order as the hybrid gap, not decades below it, which is
why the test asserts only DECISIONS (monotone improvement with the right sign,
no improvement with the wrong one) and never a reading.

---

## 10. [M] COST

`g6_cost_pillar.py`, `g7_testfixtures.py`.

**Table C1 -- per-region assembly + eig**, `Nx = Ny = 2`, conical Bloch shift,
median of three (WIN):

| M | `q^2` | in-plane `2q^2` (QZ) | vertical OOP `4q^2` | SLANTED `4q^2` | **slant / OOP** | slant / in-plane |
|---|---|---|---|---|---|---|
| 4 | 36 | 9.5 ms | 28.7 ms | 28.7 ms | **1.00x** | 3.03x |
| 5 | 64 | 38.2 ms | 96.1 ms | 83.6 ms | **0.87x** | 2.19x |
| 6 | 100 | 119.1 ms | 249.6 ms | 235.6 ms | **0.94x** | 1.98x |
| 7 | 144 | 337.0 ms | 688.3 ms | 642.3 ms | **0.93x** | 1.91x |

At the test's sizes the WSL arm reads `0.90x` / `0.93x` (M = 5 / 6) against
WIN's `0.97x` / `0.96x`, so the ratio's cross-build spread is ~7% and the
test's bar is set at 2.0x.

**THE SLANT ITSELF IS FREE.** Against the vertical out-of-plane path -- which
is where a slanted cell has to live anyway, because its covariant tensor has
out-of-plane entries -- the congruence and the six blocks cost `0.87x .. 1.07x`
(including the prototype's own reading). The `1.9x .. 3.0x` against the
in-plane `2q^2` path is the price of needing the first-order generator at all,
and it is the SHIPPED Stage-B number.

**Table C2 -- end to end** (scalar pillar, conical 20/35, `n_orders = 3`):
`1.55x` (M=5), `1.47x` (M=6), `1.49x` (M=7) vs a vertical scalar solve.

**The "3.4x vs FMM" figure, corrected in the roadmap.** Edee-Granet 2024
S4.B's number is an **S-MATRIX SIZE ratio at matched accuracy, not a wall
clock**: with `N_b = 55` layers the FMM reaches an S-matrix size of 6724 while
the PMM levels off around 2000 (`6724/2000 = 3.36`). The same section states
the honest cost structure -- the PMM's slanted eigenproblem is `4(2N)(2N)`,
twice the FMM's per layer, but it replaces `N_b` of them.
`docs/PMM_ROADMAP.md` Phase D now says this.

---

## 11. Tests

`tests/unit/test_pmm2d_staggered_slant.py` -- **68 tests**.

| build | wall | slowest test |
|---|---|---|
| WIN | **84.0 s** | 5.69 s (`test_b2_null_residual_is_discretization_and_spectral`) |
| WSL | **77.1 s** | 5.44 s (`test_m4_slanted_pillar_vs_independent_hybrid_metric[normal]`) |

Within the plan's test-cost rule (file < 4 min, no test > 40 s, grids
`<= (3,3)`, `M <= 8`, OMP caps at the file top). `.test_durations` spliced by
dict union (68 new entries, nothing else touched).

The suite's shape, in the order it is written:

| test(s) | gate |
|---|---|
| `test_b1_slant_zero_is_bit_identical` (6), `test_b1_congruence_is_the_identity_at_zero_slant` | B1 |
| `test_b2_uniform_layer_at_any_slant_is_a_noop` (5), `test_b2_null_residual_is_discretization_and_spectral` | B2 |
| `test_b3_frame_anchor_phase_two_sided` (4) | B3 |
| `test_b4_sheared_frame_dispersion_matches_exact_roots` (5), `test_b4_both_halves_of_the_formulation_are_load_bearing` (5), `test_b4_sum_of_roots_and_m_ladder` | B4 |
| `test_b5_slanted_stripe_matches_1d_oracle_per_order` (6), `test_b5_wrong_slant_sign_conserves_energy` | B5 |
| `test_b6_slant_times_out_of_plane_matches_1d_jones` (4), `test_b6_hybrid_refuses_slant_times_out_of_plane` | B6 |
| `test_b7_forward_backward_split_is_exactly_half` (4) | B7 |
| `test_b8_no_forward_mode_grows_and_closure_does_not_run_away` (4) | B8 |
| `test_b9_one_layer_equals_two_half_layers` (4) | B9 |
| `test_b10_no_floor_survives_the_slant` (4) | B10 |
| `test_b11_parity_accelerator_refuses_a_slanted_cell`, `..._symmetry_auto_on_a_slanted_cell_is_identical_to_false`, four refusal tests, `test_b11_slant_is_in_the_eig_cache_key` | B11 |
| `test_m4_slanted_pillar_vs_independent_hybrid_metric` (2), `test_m4_pure_staircase_converges_toward_the_metric_layer` (2) | M4 |
| `test_cost_slant_is_free_against_the_out_of_plane_solve_it_must_use` | COST |

Every numeric bar states BOTH builds' readings and the table above it comes
from; every "this is right" claim is paired with an engineered arm that is
wrong, separated by decades; bit-identity claims are same-build, two-arm,
sha256.

**Regression suite re-run green** (`test_pmm2d_staggered_oop.py`,
`test_pmm2d_staggered_oop_block_eig.py`, `test_pmm2d_staggered_anisotropic.py`,
`test_pmm2d_staggered_magnetic.py`, `test_pmm2d_staggered_wood_list.py`,
`test_v5_12_0_pmm2d_staggered.py`, `test_v5_21_pmm2d_staggered_oblique.py`,
`test_p2c_pmm2d_stack_cascade.py`, `test_v5_14_0_pmm2d_stack.py`):
**228 passed in 435.6 s** on WIN, after the `wave2/pmm2d` merge.

`ruff check lumenairy/ tests/` clean.

---

## 12. Commits

| commit | what |
|---|---|
| `96374af` | `feat(pmm2d slant)`: the congruence, the rotation gauge, the six blocks, the parity refusal, the two entry points |
| `e6ee0da` | `feat(pmm2d slant)`: `slant=` through `PMM2DStackPure`, the frame-anchor phase, the refusals, the Wood decision |
| `0b62943` | `test(pmm2d slant)`: gates B1-B11, bars from two builds, + `validation/probe_pmm2d_staggered_slant_build/` |
| `f23b0a3` | merge `wave2/pmm2d` (brings the prototype into the tree + the magnetic Wood `eps*mu` follow-up) |
| (this doc) | `docs(pmm2d slant)`: the build doc, the module docstrings, `PMM_ROADMAP` Phase D -> shipped, `CHANGELOG` |

---

## 13. Open items

1. **The parity accelerator on slanted cells.** Measured admissible on the
   cells here (`2.8e-14` in the original pencil) but refused by construction.
   Re-enabling it needs its own two-sided gate covering the forward/backward
   split and the reconstruction at slant, not merely the structural residual.
2. **Mixed slants between PATTERNED layers** -- refused. The exact fix is a
   lateral translation between two nodal grids, which is exact only on a whole
   number of grid cells; a general one needs an interpolation the pure basis
   does not have.
3. **`retain_internal` / `layer_absorption` under a slanted layer** -- refused.
   `_flux_at` would have to be taught the frame (the same per-order phase, at
   the probe plane's own depth).
4. **TAPERS** are still a z-staircase, and are a different geometry: a taper's
   `sqrt(g)` is z-dependent, which brings back a dilation generator, a
   non-normal `q -> -conj(q)` pencil with no valid mode selector and a
   distorted far field. None of that arises here and none of it is solved here.
5. **The TM `~1e-4` wall-normal level** on a stripe is the shipped staggered
   basis's own behaviour and the 1-D oracle's own floor; the shear does not
   touch it (B5's TM column tracks the vertical control at every `M`). The
   proper fix remains the roadmap's DG/mortar S-matrix.
6. **`pmm_efficiency_2d_staggered` with a slant** raises rather than driving
   both polarizations internally. If a single-polarization slanted efficiency
   is ever wanted, the honest form is a thin wrapper over
   `pmm_jones_2d_staggered` that says which incident polarization it drove.
7. **A slanted layer's own Rayleigh cut-off** is order-dependent
   (`eps_eff(m) = abs(alpha_m)^2 + (t.alpha_m)^2`, S8) and the flat nudge list
   cannot express it. The lab reading catches the coincidence that matters;
   a per-order guard would be a separate, larger change to
   `_grazing_safe_wavelength`.
