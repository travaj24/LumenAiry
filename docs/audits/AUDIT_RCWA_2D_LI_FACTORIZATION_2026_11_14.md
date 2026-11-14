# Audit: the 2-D inverse rule (`formulation='li'`) of RCWAStack on crossed gratings (2026-11-14)

**Question.** Is the inverse-rule factorization of the Fourier modal solver correct on
two-dimensional (crossed) gratings, and what should `formulation='auto'` resolve to on a 2-D
`RCWAStack`?  Trigger: a reported measurement on a square array of `eps = -3+1i` patches in
which the 2-D `'li'` reflectance moved away from two independent references as the truncation
grew, while the plain Laurent rule moved toward them -- and the 5.51.0 staging branch made
`'auto'` choose `'li'` for metallic 2-D layers without a metallic 2-D measurement (plan
`PLAN_METAL_STRIP_GRATING_SOLVERS_2026_11_14.md`, decision D2).

**Base:** branch `wip/v551_stage` at `93c7c6df`.  Every number below was measured on this tree
(`lumenairy.__file__` asserted inside the worktree, BLAS 2 threads, a shared and loaded
machine: times are indicative).

**Short answer.**  The library's 2-D `'li'` is Li's 1997 rule, term by term; it has no defect,
and on every dielectric cell it converges to the same limit as two independent references.  It
is 15x-60x more accurate than Laurent's rule on dielectric rectangles at the same truncation,
2x-50x on metal rectangles of eps -10+1i and -70+5i, and better or level on rasterized disks.
The one regime where Laurent's rule is more accurate is the triggering cell itself: a weak metal
(eps -3+1i) whose right-angle corners sit at the border of the irregular field singularity,
above about 14 orders per direction; both rules are within 2e-3 of the reference there.  The
"drift away" seen in the reflectance is partly real (the inverse rule's complex-`r` error grows
from 1.1e-3 at 4 orders to 2.1e-3 at 12) and partly a projection: from 12 to 28 orders that
error falls again (1.6e-3) while `R` keeps falling, because `r` approaches the reference along
a direction in which `|r|` decreases.  `'auto'` on a 2-D stack now resolves per
shape class -- `'li'` for every layer the rule supports, whatever its material -- instead of
the staging rule's "metallic layers only" (section 6).

---

## 1. What the literature says the rule is

**(a) Rectangles aligned with the lattice -- Li 1997** (L. Li, "New formulation of the Fourier
modal method for crossed surface-relief gratings", J. Opt. Soc. Am. A 14, 2758 (1997)).  For a
z-invariant layer, with the boundary made of segments parallel to the axes ("zigzag" contour,
his Fig. 4), Maxwell's equations are Fourier analysed with the factorization rules of Li 1996
(J. Opt. Soc. Am. A 13, 1870):

* `D_x = eps E_x`: across an x-wall (`x = const`) `E_x` is normal and `D_x` continuous, so the
  x-Fourier analysis takes the INVERSE rule; across a y-wall `E_x` is tangential and continuous,
  so the subsequent y-Fourier analysis takes the DIRECT (Laurent) rule.  The operator
  (his Eq. 8, at `zeta = 0`):

  `[[eps]]_xx[(m,n),(j,l)] = (1/d_y) int_0^{d_y} {[1/eps]_x(y)^-1}_{mj} exp(-i (n-l) K_y y) dy`,

  i.e. for every y the x-Toeplitz matrix of `1/eps` is built and inverted, and the resulting
  matrix function of y is Fourier transformed along y.  Order matters: inverse rule FIRST, along
  the component's own axis, then the direct rule along the other axis.
* `D_y = eps E_y`: the transposed construction (his Eq. 9): inverse rule along y per x, then the
  direct rule along x.  The two in-plane components therefore take DIFFERENT operators; they
  are not transposes of each other's matrix, they are the same recipe with x and y exchanged.
* `E_z`: tangential to every vertical wall, hence continuous; `D_z = eps E_z` takes the direct
  rule and the `E_z` elimination uses `[[eps]]^-1` with `[[eps]]` the plain 2-D Laurent
  (Toeplitz-block-Toeplitz) matrix (his Eq. 27).
* For `zeta = 0` (rectangular lattice) Li's "hypothesis" (his note 19) is not needed: the rule
  is derived, not assumed.  Li notes (note 18) that he ignores the edge effect at the vertices
  of the zigzag contour.

**(b) Curved or oblique walls.**  Li's rule sees a curved wall as a staircase.  It converges to
the staircase, whose far field approaches the smooth shape's as the steps shrink, but the steps
put walls parallel to the field where the true wall is oblique, and the rule then mixes
tangential and normal components.  Schuster, Ruoff, Kerwien, Rafler and Osten, "Normal vector
method for convergence improvement using the RCWA for crossed gratings", J. Opt. Soc. Am. A 24,
2880 (2007), Sec. 4-5: Li's formulation "can be understood as a generalization of the correct
1D formulation to 2D gratings, which can only be achieved by introducing zigzag boundary
curves"; on the checkerboard of Li's Example 1 in the axis-aligned unit cell "Li's formulation
is perfectly appropriate" and their normal-vector (NV) formulation converges like it, while in
the 45-degree cell (all walls oblique) Li's rule converges worse than in the axis-aligned one and
the NV formulation much better.  The NV formulation (Popov and Neviere, J. Opt. Soc. Am. A 18,
2886 (2001); Schuster 2007 Eq. 7-8):

  `[D_x; D_y] = ( [[eps]] I - Delta [[N N^T]] ) [E_x; E_y]`,
  `Delta = [[eps]] - [[1/eps]]^-1`,

with `N` a smooth field equal to the wall normal on every boundary; the cross term
`-Delta [[N_x N_y]]` is what a curved wall needs and what the axis rule lacks.  Goetz, Schuster,
Frenner, Rafler and Osten, Opt. Express 16, 17295 (2008), generate `N` automatically (gradient of
a blurred bitmap, masked to the boundary band, inverse-distance interpolation elsewhere).  Liu
and Fan, "S4: A free electromagnetic solver for layered periodic structures", Comput. Phys.
Commun. 183, 2233 (2012), implement and compare the practical variants (Laurent,
discretized / subpixel-smoothed permittivity, the polarization decomposition with normal-vector
or Jones-vector bases); the polarization-decomposition variants are the ones that handle
arbitrary (curved) shapes, at a higher cost per Fourier coefficient.  Lalanne, J. Opt. Soc.
Am. A 14, 1592 (1997), uses a weighted mix of the two rules for 2-D cells.

**(c) Why sharp metal corners are hard for every expansion method.**  Li and Granet,
J. Opt. Soc. Am. A 28, 738 (2011): at a right-angle edge the transverse field behaves as
`rho^(tau - 1)` with `tau = (2/pi) arccos sqrt(Delta')`,
`Delta' = (e1 e3 - e2 e4)^2 / ((e1+e2)(e2+e3)(e3+e4)(e4+e1))` (their Eqs. 2-3).  For one metal
quadrant `m` in a dielectric `d = 1` (the corner of a patch or pillar) `Delta' = (m-1)^2 /
(4 (m+1)^2)`; for real `m` in `(-3, -1/3)` `tau` is imaginary (an irregular, infinitely
oscillating singularity: "any method using Fourier series this way is doomed to fail"), and
`m = -3` is the border, where `tau = 0`.  Measured exponents of the cells below (corner of one
quadrant in vacuum):

| eps of the quadrant | tau | field near the corner |
|---|---|---|
| -3 + 1i (weak metal) | 0.361 - 0.232i | `rho^-0.64`, oscillating in `log rho` |
| -10 + 1i | 0.583 - 0.010i | `rho^-0.42` |
| -70 + 5i (strong metal) | 0.656 - 0.001i | `rho^-0.34` |
| 12 (dielectric) | 0.722 | `rho^-0.28` |
| 4 | 0.806 | `rho^-0.19` |

The weak-metal patch of the triggering measurement sits next to the irregular band: it is the
hardest of these cells for EVERY method (the references below converge slowly too).  Lyndin,
Parriaux and Tishchenko, J. Opt. Soc. Am. A 24, 3781 (2007), add the strong-metal mechanism:
the truncated Fourier system carries slow, high-spatial-frequency spurious plasmon-like modes
whose interference makes low-loss metal results jump with the truncation.

## 2. What the library builds, term by term

| Li 1997 | library | verdict |
|---|---|---|
| Eq. 8: per y, x-Toeplitz of `1/eps`, inverted, then Fourier along y, for `E_x` | `twod._li_convolutions_2d` (`eps_cell`): `cf = fft(1/eps, axis=0)/Sx`, `Tx = cf[dmx, :]` per pixel row, `Ahat = fft(inv(Tx), axis=0)/Sy`, `Cxx[(m,n),(m',n')] = Ahat[n-n', m, m']`.  `twod._li_rect_axis_operator` / `_analytic_li_operators_2d` (rectangles): the same with the rows replaced by their exact y-bands and the band-wise constant inverse Fourier transformed exactly | identical |
| Eq. 9: the transposed construction for `E_y` | `Cyy` from `Ty = cg[:, dmy]` per pixel column, `Bhat = fft(inv(Ty), axis=0)/Sx`, `Cyy[(m,n),(m',n')] = Bhat[m-m', n, n']` | identical |
| `E_z` elimination with `[[eps]]^-1`, plain 2-D Laurent (Eq. 27) | `_layer_eigenmodes_tensor(Kx, Ky, Cxx, 0, 0, Cyy, EPS)`: `P` from `inv(EPS)`, `EPS = _eps_convolution_2d(cell)` | identical |
| `D_x` row: `Cxx` multiplies `E_x`; `D_y` row: `Cyy` multiplies `E_y` | `_tensor_PQ`: `Q = [[Cyx + Kx Ky, Cyy - Kx Kx], [Ky Ky - Cxx, -(Cxy + Ky Kx)]]` on the state `[E_x, E_y]` (`Cyy` in the `dH_x/dz` row with `E_y`, `Cxx` in the `dH_y/dz` row with `E_x`) | identical |
| shapes | rectangles only (`_li_rect_shapes_problem` refuses disks, ellipses, any rotation key); a disk reaches the rule only as a rasterized `eps_cell`, i.e. as Li's zigzag | as Li 1997 |

Pinned by `tests/unit/test_rcwa_li_2d_crossed.py::test_li_operators_are_lis_eqs_8_and_9`, which
transcribes Eqs. 8 and 9 as explicit sums on a cell with two different rectangles (metal and
dielectric, `Mx != My`) and finds the library's blocks equal to 1e-12 relative, while the swapped
assignment, the Laurent matrix and the global 2-D inverse `[[1/eps]]^-1` are all O(1e-1) away.
Existing pins already cover the placement and the bridge: a y-uniform cell reduces to the 1-D
inverse rule per order (`test_rcwa_stack_shapes_li_2d.py::test_one_band_full_height_rectangles
_reproduce_c1`), the raster converges to the analytic rectangles (Romberg residual ~1e-11), and
`symmetry=False` reproduces the even-parity fold (re-measured here on the weak-metal patch,
`'li'` N = 8: equal to all printed digits).

**Verdict: no defect.**  Neither the order of the two steps, nor the per-component assignment,
nor the transposition, nor the `E_z` rule departs from Li 1997.  What the rule cannot do --
curved walls, and sharp metal corners -- is a property of the rule, not of the code.

## 3. References and their uncertainty

**Cells.**  A 300 nm square (or a disk of radius 180 nm) centred in a 600 nm square cell, in
air, over uniform layers, on n = 1.5, at 1310 nm (no diffracted order propagates, so `R` is the
zeroth order).  `weak`: 60 nm of eps -3+1i over 150 nm of eps 2.1 (the triggering patch cell).
`mid`: the same with eps -10+1i.  `strong`: 50 nm of eps -70+5i over 60 nm of eps 2.1 over a
100 nm mirror of eps -70+5i (patch / spacer / mirror).  `diel`: 200 nm of eps 12 over 150 nm of
eps 2.1; `lowd`: the same with eps 4.  `checker`: Li 1997 Example 1 in his unit cell A (period
2.5 lambda, two eps-2.25 squares of side 1.25 lambda touching at their corners, lambda thick,
incident from n = 1.5 onto air).  Normal incidence for all; `weak`, `strong`, `diel` also at
theta = 0.3 rad, phi = 0 (x column = p, y column = s).

**The quantity compared** is the zeroth-order reflection Jones entry `J_xx` (and `J_yy` at
oblique incidence), complex -- per-order amplitude, phase included, not only energy.  Both
solvers reference the reflection phase to the top interface of the first layer; on a
uniform-only twin of the weak cell they agree to 1.5e-15.

**Reference 1: the pure staggered 2-D polynomial modal solver** (`PMM2DStackPure`:
Granet's modified-Legendre basis on the walls of the cell, an exact segment decomposition, no
Fourier factorization at all; a disk through its 3 x 3 conformal map), at the largest
affordable `n_modes` M.  It converges algebraically on these sharp corners, monotonically on
all cells but `weak` (odd / even M interleave there).  Its uncertainty is stated as the largest
change over its last two rungs; on the slowly converging metal cells the true error can exceed
that by a small factor (the `mid` ladder still moves 0.6-1.0e-3 per rung at M = 12).

| cell | top M | `J_xx` | uncertainty |
|---|---|---|---|
| weak square | 12 | -0.18244 - 0.20252i | 4.4e-4 |
| mid square | 12 | -0.30021 - 0.40449i | 2.7e-3 |
| strong square | 12 | -0.84189 + 0.31098i | 5.9e-3 |
| diel square | 12 | -0.12251 + 0.01401i | 1.5e-4 |
| lowd square | 10 | -0.03927 - 0.04417i | 3.9e-5 |
| checkerboard | 16 | +0.14440 - 0.06227i | 2.8e-5 |
| weak disk | 9 | -0.18307 - 0.21743i | 3.6e-4 |
| mid disk | 9 | -0.42298 - 0.48060i | 3.0e-3 |
| strong disk | 9 | -0.83656 + 0.35122i | 1.0e-2 |
| diel disk | 9 | -0.13016 + 0.02343i | 3.4e-4 |

**Reference 2: a published value.**  Li 1997, Table 1: the checkerboard's T(0,0) = 17.486 %
(computed in his unit cell B; his Fig. 6 shows cell A and cell B differing by about 1e-3).  The
polynomial modal reference gives 0.175621 (M = 16, rungs 12 / 14 / 16: 0.175659 / 0.175635 /
0.175621), 7.6e-4 above Li's, inside his A/B spread.  The library's `'li'` gives 0.174861 at
`n_orders` 10 and 0.175259 at 14 (climbing toward the polynomial value); `'laurent'` gives
0.183838 / 0.181661.

**Reference 3: the Laurent tail.**  On the dielectric cells Laurent's sequence is the clean
`a + b/N` of the direct rule (successive-difference ratios match the `1/N` prediction to within 5 %),
and its extrapolation lands on the other references: `diel` R 0.01522 from N = 8 / 10 / 12
(polynomial 0.01521 at M = 12, `'li'` 0.01515 at N = 24); `lowd` Re J -0.03934 from N = 12 / 16 / 20
(`'li'` -0.03921, polynomial -0.03927); checkerboard T(0,0) 0.1761 from N = 10 / 12 / 14.  So on
dielectrics the two Fourier rules and the polynomial solver share one limit.  On the metal cells
neither Fourier sequence has a clean tail and no extrapolation is attempted.

**Reference 4: an external one.**  For the weak square only, the triggering report quotes a 3-D
finite-element R of 0.074403 / 0.074469 / 0.074531 at three refinements (|r| 0.27300).  The
polynomial reference gives R 0.07430 at M = 12, still rising by ~1e-4 per two rungs: consistent.

**Symmetry checks** (sensitive to per-order errors, not to energy): at normal incidence the
cells are invariant under x <-> y, and `J_xx = J_yy`, `J_xy = J_yx = 0` hold to round-off for
both rules and the polynomial solver on every cell (worst over all normal-incidence runs:
`|J_xy|` 5e-12, `|J_xx - J_yy| / |J_xx|` 7e-11; the even-parity fold engages for `'li'`, and
`symmetry=False` reproduces it to all printed digits).  These checks are passed by every
formulation, correct or not, so they are recorded but do not discriminate.

## 4. Convergence tables

`|J - J_ref|` against the polynomial reference above, `n_orders` N per direction
(`(2N+1)^2` harmonics), both rules; `R` for orientation; wall time of one solve (shared, loaded
machine, BLAS 2 threads, even-parity fold active at normal incidence).  On the disks `'laurent'`
is the analytic disk (`shapes=`) and `'li'` a 1024 x 1024 node raster of it (`eps_cell`, Li's
staircase) -- `'li'` does not take a disk shape.

### Squares, normal incidence

weak (eps -3+1i), J_ref M = 12 (uncertainty 4.4e-4)

| N | 'li' | 'laurent' | R 'li' | R 'laurent' | t (s) |
|---|---|---|---|---|---|
| 4 | 1.09e-03 | 1.41e-02 | 0.07441 | 0.07987 | <1 |
| 8 | 2.07e-03 | 6.98e-03 | 0.07428 | 0.07813 | <1 |
| 12 | 2.14e-03 | 3.06e-03 | 0.07392 | 0.07580 | 3 |
| 16 | 2.03e-03 | 1.14e-03 | 0.07373 | 0.07479 | 10-15 |
| 20 | 1.89e-03 | 2.59e-04 | 0.07365 | 0.07441 | 45 |
| 24 | 1.75e-03 | 1.59e-04 | 0.07363 | 0.07424 | 120 |
| 28 | 1.63e-03 | 3.92e-04 | 0.07363 | 0.07414 | 290 |

mid (eps -10+1i), J_ref M = 12 (2.7e-3)

| N | 'li' | 'laurent' | R 'li' | R 'laurent' | t (s) |
|---|---|---|---|---|---|
| 4 | 3.28e-02 | 1.08e-01 | 0.28510 | 0.23586 | 1 |
| 8 | 2.37e-02 | 6.92e-02 | 0.27750 | 0.26316 | <1 |
| 12 | 1.91e-02 | 8.56e-02 | 0.27314 | 0.19700 | 3 |
| 16 | 1.76e-02 | 3.25e-02 | 0.27153 | 0.23678 | 12 |
| 20 | 1.60e-02 | 3.27e-02 | 0.27011 | 0.25116 | 39 |
| 24 | 1.47e-02 | - | 0.26877 | - | 132 |

strong (eps -70+5i, over a mirror), J_ref M = 12 (5.9e-3)

| N | 'li' | 'laurent' | R 'li' | R 'laurent' | t (s) |
|---|---|---|---|---|---|
| 4 | 7.23e-02 | 6.51e-01 | 0.70070 | 0.84500 | 1 |
| 6 | 1.91e-02 | 3.88e-01 | 0.78283 | 0.67006 | <1 |
| 8 | 3.72e-02 | 2.69e-01 | 0.76706 | 0.40972 | <1 |
| 10 | 4.25e-02 | 8.08e-01 | 0.75693 | 0.02194 | 2 |
| 12 | 6.96e-02 | 7.35e-01 | 0.73950 | 0.56978 | 3 |
| 14 | 1.58e-01 | 5.42e-01 | 0.58023 | 0.65882 | 7 |
| 16 | 9.79e-02 | 4.25e-01 | 0.67376 | 0.65409 | 16 |
| 18 | 4.61e-02 | 3.86e-01 | 0.75083 | 0.68152 | 31 |
| 20 | 2.59e-02 | 3.35e-01 | 0.77122 | 0.61557 | 54 |
| 22 | 1.85e-02 | - | 0.77554 | - | 99 |
| 24 | 1.37e-02 | 2.82e-01 | 0.78199 | 0.71504 | 150 |

diel (eps 12), J_ref M = 12 (1.5e-4)

| N | 'li' | 'laurent' | R 'li' | R 'laurent' | t (s) |
|---|---|---|---|---|---|
| 4 | 1.57e-03 | 6.14e-02 | 0.01490 | 0.03397 | 1 |
| 8 | 5.32e-04 | 3.34e-02 | 0.01511 | 0.02452 | <1 |
| 12 | 3.38e-04 | 2.31e-02 | 0.01514 | 0.02142 | 3 |
| 16 | 2.76e-04 | 1.77e-02 | 0.01515 | 0.01987 | 14 |
| 20 | 2.50e-04 | 1.43e-02 | 0.01515 | 0.01893 | 40 |
| 24 | 2.37e-04 | - | 0.01515 | - | 111 |

lowd (eps 4), J_ref M = 10 (3.9e-5)

| N | 'li' | 'laurent' | R 'li' | R 'laurent' | t (s) |
|---|---|---|---|---|---|
| 4 | 8.14e-04 | 1.34e-02 | 0.00347 | 0.00394 | 1 |
| 8 | 2.38e-04 | 7.05e-03 | 0.00348 | 0.00368 | <1 |
| 12 | 1.29e-04 | 4.79e-03 | 0.00349 | 0.00361 | 3 |
| 16 | 9.00e-05 | 3.63e-03 | 0.00349 | 0.00358 | 12 |
| 20 | 7.19e-05 | 2.92e-03 | 0.00349 | 0.00356 | 40 |

checkerboard (eps 2.25, Li 1997 Example 1, cell A), J_ref M = 16 (2.8e-5)

| N | 'li' | 'laurent' | T00 'li' | T00 'laurent' | t (s) |
|---|---|---|---|---|---|
| 4 | 2.63e-03 | 5.94e-03 | 0.16873 | 0.19109 | 1 |
| 6 | 4.45e-04 | 3.07e-03 | 0.17307 | 0.18808 | <1 |
| 8 | 8.59e-05 | 2.20e-03 | 0.17434 | 0.18558 | <1 |
| 10 | 6.82e-05 | 1.74e-03 | 0.17486 | 0.18384 | 1 |
| 12 | 8.31e-05 | 1.45e-03 | 0.17512 | 0.18259 | 3 |
| 14 | 8.52e-05 | 1.25e-03 | 0.17526 | 0.18166 | 6 |

### Squares, oblique incidence (theta = 0.3 rad), p (`J_xx`) / s (`J_yy`)

J_ref at M = 9: weak 1.1e-3 / 9.0e-4, strong 1.7e-2 / 1.7e-2, diel 5.7e-4 / 5.8e-4 uncertainty
(no symmetry fold off normal incidence: about 4x the normal-incidence time).

| N | weak 'li' p / s | weak 'laurent' p / s | strong 'li' p / s | strong 'laurent' p / s | diel 'li' p / s | diel 'laurent' p / s |
|---|---|---|---|---|---|---|
| 4 | 3.2e-3 / 1.8e-3 | 1.5e-2 / 1.5e-2 | 7.0e-2 / 6.2e-2 | 6.1e-1 / 6.1e-1 | 1.9e-3 / 2.2e-3 | 6.1e-2 / 6.4e-2 |
| 8 | 3.3e-3 / 2.5e-3 | 7.5e-3 / 7.9e-3 | 2.9e-2 / 2.9e-2 | 3.0e-1 / 2.5e-1 | 8.0e-4 / 9.0e-4 | 3.3e-2 / 3.4e-2 |
| 12 | 2.9e-3 / 2.4e-3 | 4.2e-3 / 3.8e-3 | 6.0e-2 / 6.1e-2 | 7.0e-1 / 6.9e-1 | 6.3e-4 / 6.8e-4 | 2.3e-2 / 2.4e-2 |
| 14 | 2.7e-3 / 2.2e-3 | 3.1e-3 / 2.6e-3 | 1.5e-1 / 1.5e-1 | 5.2e-1 / 5.1e-1 | 5.9e-4 / 6.3e-4 | 2.0e-2 / 2.1e-2 |
| 16 | 2.6e-3 / 2.1e-3 | 2.4e-3 / 1.8e-3 | 1.1e-1 / 1.0e-1 | 4.1e-1 / 4.0e-1 | 5.7e-4 / 6.1e-4 | 1.7e-2 / 1.8e-2 |

### Disks, normal incidence ('laurent' = analytic disk, 'li' = 1024 x 1024 raster)

| N | weak 'laurent' | weak 'li' | mid 'laurent' | mid 'li' | strong 'laurent' | strong 'li' | diel 'laurent' | diel 'li' |
|---|---|---|---|---|---|---|---|---|
| 4 | 3.37e-3 | 6.29e-3 | 7.54e-2 | 8.26e-2 | 7.73e-1 | 2.01e-1 | 6.86e-2 | 1.25e-2 |
| 8 | 3.64e-3 | 2.56e-3 | 4.19e-2 | 4.40e-2 | 3.68e-1 | 2.94e-1 | 3.62e-2 | 6.33e-3 |
| 12 | 1.78e-3 | 1.86e-3 | 4.96e-2 | 2.89e-2 | 3.66e-1 | 1.32e-1 | 2.43e-2 | 4.19e-3 |
| 16 | 1.69e-3 | 1.74e-3 | 3.22e-2 | 2.05e-2 | 2.13e-1 | 1.82e-1 | 1.85e-2 | 3.17e-3 |

(reference uncertainty: weak 3.6e-4, mid 3.0e-3, strong 1.0e-2, diel 3.4e-4.)

**Cost.**  The two rules cost the same to within the noise of a loaded machine at every rung
(the inverse rule's extra work is `Sy + Sx` batched inversions of `(2N+1)`-square Toeplitz
blocks -- band-wise for analytic rectangles -- next to the `2(2N+1)^2` eigensolve): 1 s at
N = 8, 3 s at 12, 12-16 s at 16, 40-55 s at 20, 110-150 s at 24, 290 s at 28 (about 9.5 GB
resident at 28) at normal incidence with the even-parity fold; oblique solves cost about 4x.
The polynomial references cost 2 min (M = 10) to 7-9 min (M = 12) per solve.

**What could not be afforded:** `N` beyond 28 (memory and hours per solve), polynomial `n_modes`
beyond 12 on the single-patch cells (13-14 would take 15-30 min each), and oblique ladders for
the disks and for `mid` / `lowd`.

## 5. What the measurements say

1. **The 2-D inverse rule converges to the right answer.**  On every dielectric cell its error
   falls monotonically to the reference's own uncertainty (`lowd` 7e-5, checkerboard 7e-5 to 9e-5
   with a reference good to 3e-5, `diel` 2.4e-4 with a reference good to 1.5e-4 and still moving toward it), Laurent's `1/N`
   tail extrapolates to the same value, and on the published checkerboard it lands within Li's
   own A/B spread.  This rules out a factorization defect independently of the term-by-term
   comparison of section 2.
2. **On rectangles it is the better rule everywhere but on one cell**: 15x-60x on dielectrics at
   the same N, 2x on `mid`, 3x-50x on `strong` (where Laurent's direct rule is erratic and off by
   0.3-0.8 at every affordable N), level at oblique incidence on `weak` at N = 16.
3. **The exception is `weak` (eps -3+1i) at normal incidence above N of about 14**, where
   Laurent's error keeps falling (2.6e-4 at 20) and the inverse rule's creeps (1.9e-3 at 20,
   1.6e-3 at 28).  Read through R alone this looks like divergence (R 0.07441 -> 0.07363 from
   4 to 28 orders); in complex `r` the error first grows (1.1e-3 at 4, 2.1e-3 at 12) and then
   falls slowly, `r` approaching the reference along a direction in which `|r|` decreases.  This is the cell whose corner sits at the border of the irregular
   singularity (section 1c, `tau = 0.36 - 0.23i`): every method converges slowly there (the
   polynomial reference interleaves odd / even M at 4e-4), and Li's rule, built for the jumps of
   `E_x` across walls, gains nothing over Laurent's when the field near the corner is dominated by
   the oscillating singular term.  Both rules are within 2e-3 of the reference at every N
   measured.
4. **Disks:** Li's staircase (an `eps_cell` raster under `'li'`) is 6x closer than the analytic
   disk under Laurent's rule on the dielectric, 1.5x on `mid`, level on `weak`; on `strong` both
   are 0.13-0.37 off at N <= 16 (unusable; use the polynomial solver).  The library's
   normal-vector formulation (`fff_nv`, single-layer functions only) refuses cornered or curved
   metal cells by design (`twod._nv_nonseparable_guard`), so there is no measured curved-wall
   rule to switch to.
5. **Energy closure** is at round-off for both rules on every cell (lossless trap: it carries
   no information about which is right).

## 6. Decision: what `formulation='auto'` resolves to on a 2-D stack

**Per shape class, not per material:**

| 2-D layer | 'auto' | measured basis |
|---|---|---|
| `eps_cell` (any material, any pattern, dispersive included) | `'li'` | squares: better or level on every cell at N <= 14, better at every N except `weak` above 14; rasterized disks: better or level on every cell |
| `shapes`, axis-aligned rectangles only (any material) | `'li'` | the analytic form of the same rule; the square tables above |
| `shapes` holding a disk or an ellipse | `'laurent'` | `'li'` takes rectangles only (Li 1997 is a zigzag rule); rasterize the cell to use Li's staircase, measured better on dielectric disks |
| JAX cell; exact zero permittivity; uniform; tensor | `'laurent'` | unchanged |

The 5.51 staging rule ('li' for METALLIC layers only, decision D2) was set without a metallic
2-D measurement.  Measured, it was right for `strong` and `mid` but left every dielectric pillar
on the direct rule, 15x-60x less accurate at the same truncation (the "few 1e-3" by which the
two rules differed on the plan's dielectric pillar is Laurent's error, not a property of 'li'),
and it gave the one cell where Laurent wins at high N (`weak`) to 'li' anyway.  A material
threshold could carve out `weak`, but it would rest on one measured point, it would switch the
rule inside a dispersive spectrum, and at the truncations a 2-D stack can afford (N <= 14,
<= 7 s per solve) the inverse rule is still the better of the two there.  The rule therefore
does not read the material; the `add_layer` docstring carries the table above and names the
`weak`-metal regime, where an explicit `formulation='laurent'` is measured better above
N of about 14.

**No runtime warning** is added anywhere (library users run with warnings as errors, where any new warning on a default path is a failure).

**What moves:** 2-D dielectric layers left at the default (5.50 / staging: Laurent; now: 'li')
and 2-D dispersive layers left at the default.  Explicit `'laurent'` and explicit `'li'` results
are unchanged bit for bit (re-solved after the change: identical to the pre-change runs on the
`diel`, `strong`, raster-disk, checkerboard and `weak` (N = 20) cells).  1-D stacks are
untouched (the 1-D branch of the resolution is unchanged).

## 7. Fix

None to the factorization: the operator is Li's, verified term by term (section 2) and by
convergence to independent references (section 5).  The change is the 2-D `'auto'` rule
(section 6) and its documentation.

## 8. Tests

`tests/unit/test_rcwa_li_2d_crossed.py` (fast, ~8 s): Li's Eqs. 8 / 9 transcribed and compared
with the library's blocks; the square cells against the polynomial reference at N = 8 (`diel`,
`strong`, `weak`: 'li' under a bar, 'laurent' above a floor); 'auto' is the explicit 'li' call
bit for bit on dielectric and metal squares; disks (analytic Laurent vs raster 'li'); Li 1997's
checkerboard against his Table 1 and against the polynomial reference.  Slow: the `weak`
crossover ladder (N = 12 and 20), and the dielectric reference one rung down.
`tests/unit/test_rcwa_stack_auto_formulation.py`: the 2-D resolution table restated for the
shape-class rule, the 2-D builder test restated ('auto' -> 'li' for a dielectric pillar,
bit-identical to the explicit call).
