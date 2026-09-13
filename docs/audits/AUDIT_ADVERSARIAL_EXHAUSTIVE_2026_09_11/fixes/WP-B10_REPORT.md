# WP-B10 -- a disc-orthogonal (Zernike) basis for the traced lens's ray fits: shipped OPT-IN, because it cannot change a fit and does change a conditioning

Subject: audit sec. 15.9 -- "Zernike (disc-orthogonal) basis for the traced fits: the entire fit-radius /
arbiter / predictor apparatus exists because a square Chebyshev basis couples marginal rays into defocus on
a disc."

Everything below is a MEASUREMENT taken on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, Python 3.14.6, one process at a time.  The
two oracles are niche D7's and they share no code with the element: the closed-form Fermat exit sphere of
the `K = -n^2` conic stand-in, and an inline exact conic raytrace (flat entrance, exact even-conic sag,
vector Snell on the gradient normal), both REBUILT here rather than imported from the test module.  The
default basis's byte identity was proved against `git archive` extractions in child processes with
`lumenairy.__file__` asserted, never through pytest.  Nothing was checked out, stashed or written to the
repository; no git write command was issued.

---

## 1. Summary

| finding | status | files:lines | tests | oracle | measured before -> after |
|---|---|---|---|---|---|
| **B10-1** the audit's premise -- that the square basis is what couples marginal rays into defocus, so a disc basis would leave the fit-radius arbiter / predictor with nothing to arbitrate | **MEASURED AND REFUSED.  A change of basis inside one span cannot move a least-squares answer** | -- (this is the finding, not a change) | `test_audit2609_b10_zernike_fit_basis.py::test_changing_the_basis_does_not_change_the_fitted_polynomial`, `::test_the_fitted_opl_misses_the_oracle_by_the_same_amount_on_both_bases`, `::test_the_arbiter_has_exactly_the_same_thing_to_arbitrate`, `::test_the_two_bases_span_the_same_space`, `::test_the_two_oracles_agree_with_each_other` | the analytic Fermat sphere and the inline exact conic trace (against each other first), plus rank / projection of the two design matrices | the element's decentred exit slope is the SAME NUMBER on both bases at every order 6..20 and both decentres (233.859 / 85.207 / 44.457 / 31.556 / 10.841 / 11.829 / 3.718 / 5.419 / **2.371 / 1.683** / 1.044 / 0.655 / 0.321 / 0.301 urad); the niche-C11 arbiter's two candidate residuals agree to all 7 printed digits (1.073452e-07 / 1.735703e-07 m at 0.5 w), so its verdict cannot move |
| **B10-2** marginal rays DO couple into defocus -- and it is a property of the WEIGHTED PROBLEM and of the ORDER, not of the basis | **MEASURED, and it is WP-A26's finding restated in a disc-orthogonal coordinate system** | -- | `::test_the_marginal_rays_couple_into_defocus_in_both_bases_alike`, `::test_the_coupling_falls_with_the_order_and_not_with_the_basis` | the exact conic exit map, projected onto the disc basis by exact quadrature | D1's skirt moves the fitted map's `Z(2,0)` on the fit disc by **2.083e-10 m** at order 6, **2.317e-11** at 10 and **1.934e-13** at the shipped 16 -- a factor of 1077 bought by the ORDER -- and the two bases report the same shift to **1.3e-10** of it at order 6 and 4.7e-09 at order 10 |
| **B10-3** what the disc basis DOES buy: the conditioning of the solve, wherever the retained samples are the disc | **SHIPPED OPT-IN** -- `fit_basis='chebyshev' \| 'zernike'`, default byte-identical | `lumenairy/elements/_lens_traced.py:3691-3897` (the basis), `:3299-3345` (the design), `:3621-3660` (the evaluation), `:7811` / `:8490-8534` / `:9152-9177` (the keyword, its documentation and its gate), `:11792-11800` / `:12008-12029` (the applied disc and the fit site) | `::test_a_disc_shaped_fit_is_conditioned_by_the_disc_basis`, `::test_the_disc_basis_advantage_decays_with_the_order_on_a_square_of_data`, `::test_the_basis_is_orthonormal_on_the_unit_disc` | the library's own `_gram_rcond` (diagonally equilibrated, so this is not a column-units artefact) and the C13 census | ON AXIS, where the restriction is a hard NaN mask and the samples ARE the disc: Gram rcond **1.415e-11 -> 9.785e-01** and the C13 step-down fires **3 of 3 solves -> 0 of 3**, with the returned field's exit slope unchanged at 41.089 urad.  DECENTRED, where D1's skirt hands the fit the whole launch square: **1.70e-10 -> 1.31e-01** at order 6, the advantage decaying a measured 0.95 decades per degree to **1.03e-11 -> 3.20e-10** at order 14 and crossing over by the shipped order 16 |
| **B10-4** the cost of the opt-in path | **MEASURED AND STATED** | `lumenairy/elements/_lens_traced.py:3621-3660` | -- (no wall clock is asserted anywhere; the tests count step-down firings instead) | interleaved medians, same call | **412.5 -> 463.9 ms, 1.12x** on WP-A26's own cost fixture (N = 512, dx = 8 um, `ray_subsample=8`, order 16); 1.69x on the `ray_subsample=1` ladder, where the per-pixel evaluation dominates.  The cause is that the Zernike evaluation has no numba kernel |
| **B10-5** the inverse-characteristic model's exit fits and niche C12's box spectrum stay Chebyshev | **DEFERRED, with the reason** (section 13) | -- | -- | -- | both are fits over a DIFFERENT domain than the ray-fit disc; C12's `s^n` inflation law is a property of the box-normalised degree-graded basis and would have to be re-derived |

**Headline.**  The audit asked for a disc-orthogonal basis on the grounds that the square one is what
couples marginal rays into defocus.  It is not, and the reason is one line of linear algebra that niche D7
already wrote down for the affine case: least squares depends on the SPAN, not on the basis.  The Zernike
set of total degree `<= order` and the tensor-Chebyshev total-degree set are two bases of the same
`(order+1)(order+2)/2`-dimensional space, so the same samples with the same weights return the same
polynomial -- measured here as the same exit slope to the digit at every order on both fixtures, and the
same arbiter residuals to seven digits.  The coupling the audit names is real, but what produces it is D1's
weighted skirt (the fit is handed every sample out to 4.01 fit-disc radii) against an order that cannot
follow it, which is exactly what WP-A26 re-derived 10 -> 16.

What the disc basis does buy is conditioning, and it buys a lot of it where the retained samples really are
a disc: the concentric branch's Gram goes from numerically singular to essentially the identity and the
niche-C13 step-down stops firing altogether.  On the decentred branch the same advantage is large at low
order and decays by a measured 0.95 decades per degree -- because a disc-normalised column grows as
`(r/R)^n` over a square of data that reaches 4.01 disc radii -- until it crosses the square basis's flat
`~1e-11` around the shipped order 16.  That decay is A26-7 ("the launch lattice is a SQUARE inside a DISC-shaped domain") made
quantitative.  So the basis ships opt-in, with the table, and the default does not move.

---

## 2. What the two bases are, and what a change of basis can and cannot do

`_Cheb2DEvaluator` fits `values` on the launch lattice against the tensor-Chebyshev columns
`T_kx(u) T_ky(v)` with `kx + ky <= order`, `u, v` normalised to the launch square.  `fit_basis='zernike'`
replaces those columns with `Z_n^m` of total degree `n <= order`, orthonormal under the mean over the
RAY-FIT DISC -- beam-centred on the decentred branch, concentric otherwise -- and nothing else changes: the
same samples, the same `_FIT_DISC_OUTSIDE_WEIGHT_REL` weights, the same `_DECENTRED_FIT_POLY_ORDER` raise
and its sample-count step-down, the same niche-C11 arbiter, the same solver.

Both sets have `(order+1)(order+2)/2` members (a Zernike degree-`n` shell holds `n+1` terms, `m` from `-n`
to `n` in steps of two; a Chebyshev total-degree shell holds `n+1`), and they span the same space.  Measured
on a 61x61 lattice, rank of the Chebyshev design, of the Zernike design and of their concatenation, and the
residual of projecting every Zernike column onto the Chebyshev columns:

| order | terms | rank cheb | rank zern | rank [cheb\|zern] | max relative projection residual |
|---|---|---|---|---|---|
| 6 | 28 | 28 | 28 | 28 | 5.354e-15 |
| 12 | 91 | 91 | 91 | 91 | 3.882e-15 |
| 16 | 153 | 153 | 153 | 153 | 3.913e-15 |

A weighted least-squares solution is the minimiser of `||W (A c - b)||` over the SPAN of `A`'s columns.
Two bases of one span give `A_z = A_c M` with `M` invertible, so `c_z = M^-1 c_c` and the fitted FUNCTION is
identical in exact arithmetic.  Niche D7 refused the affine re-map of the Chebyshev domain on exactly this
ground and pinned it (`test_the_basis_domain_is_affine_invariant_so_remapping_it_is_a_no_op`); a rotation of
the basis inside the span is the same statement with `M` not diagonal.

Consequences, all of them measured below rather than argued:

* the fit-radius arbiter cannot lose its job, because its two candidates are two different DISCS -- different
  retained samples and different weights, i.e. two different problems -- and no change of basis inside one
  span equalises two problems;
* "the marginal rays couple into defocus" cannot be a statement about the basis.  It is a statement about
  which samples the fit is required to follow, and about how many terms it has to follow them with;
* what a change of basis CAN move is the conditioning of the numerical solve, which is a joint statement
  about the basis and the sample measure -- and that is the whole of what this work package delivers.

---

## 3. The design as shipped

`apply_real_lens_traced(..., fit_basis='chebyshev' | 'zernike')` and the same keyword on
`prepare_real_lens_traced`.

* **Default `'chebyshev'`, byte-identical** (section 7).  The default path passes no new keyword to
  `_Cheb2DEvaluator` at all -- the three shipped fit calls at `:12019-12029` are spelled exactly as they
  were, which is also what keeps the fit-order spies in `test_niche_c1_consolidation` and
  `test_niche_d7_decentred_fit` (they replace `__init__` with a fixed signature) working.
* **The disc.**  `_fit_basis_disc` is resolved beside the restriction, AFTER the niche-C11 arbiter has
  chosen which disc is applied (`:11792`), and only when the restriction holds enough samples to be applied
  at all: `(bcx, bcy, _beam_fit_radius)` on the off-centre branch -- the beam's own disc, since the
  geometric intersection only trims the side away from the beam -- and `(0, 0, _fit_r_max)` concentric.
  With no restriction resolved, the fit site (`:12011`) falls back to the launch lattice's circumscribing
  disc, the smallest one that keeps every sample at `rho <= 1`.
* **The weighted restriction is untouched.**  D1's skirt is what keeps the fitted map single-valued outside
  the disc (WP-A26 section 3.1 measured that removing it folds the map), and none of the three cures A26
  measured and refused -- bounding the sample set at the clear aperture, a two-class skirt, a uniformly
  weaker `_FIT_DISC_OUTSIDE_WEIGHT_REL` -- is repeated here.  `fit_basis` changes columns, never rows.
* **Evaluation.**  The Zernike branch (`:3621`) is one implementation on every backend: the numba Chebyshev
  kernel has no Zernike counterpart, so there is no second floating-point order for a Newton pool worker to
  disagree with its parent about, and `backend` is inert there by construction rather than by policy.  The
  pool's fit state carries the basis and the disc, and ONLY on the opt-in basis, so the default payload is
  the nine keys `test_the_shipped_state_carries_the_fit_and_not_the_grids` pins.
* **Refusals.**  `fit_basis='zernike'` with `newton_fit='spline'` or `inversion_method != 'newton'` raises,
  gated for every call rather than only for the calls that build a polynomial fit: a knob that is silently
  inert is the defect this module's own `on_fit_domain_basis` ledger exists to record.
* **Numerics.**  The radial factor is the Jacobi form `R_n^m(rho) = rho^m P_{(n-m)/2}^{(0,m)}(2 rho^2 - 1)`
  by three-term recurrence, not by the monomial sum: `R_16^0`'s monomial coefficients alternate and reach
  84 084 against a polynomial bounded by 1 on the disc.  The azimuthal factor is `Re`/`Im` of
  `(u + i v)^|m|` -- no trigonometry, and no `rho = 0` case to special-case in the gradient.

---

## 4. The d7 Fermat singlet: the ladder, orders 6..20, both bases

Niche D7's fixture rebuilt (N-BK7, f = 3 mm, 3.40 mm aperture, 0.60 mm beam, `ray_subsample=1`, 405 769
rays), exit-slope rms over the beam core against the analytic decentre-invariant sphere, at 0.5 and 1.0 beam
radii of decentre.  `rcond` is the smallest equilibrated Gram reciprocal condition number over the call's
five least-squares solves; `sing` counts how many of them screen singular, i.e. how often the niche-C13
step-down engages.

| order | terms | 0.5 w urad | 1.0 w urad | rcond 0.5 w cheb -> zern | sing 0.5 w | rcond 1.0 w cheb -> zern | sing 1.0 w |
|---|---|---|---|---|---|---|---|
| 6 | 28 | **233.859 / 233.859** | **354.413 / 354.413** | 1.696e-10 -> **1.306e-01** | 5/5 -> **0/5** | 1.925e-10 -> **6.148e-02** | 4/5 -> **0/5** |
| 8 | 45 | **85.207 / 85.207** | **90.432 / 90.432** | 5.733e-11 -> **8.284e-04** | 5/5 -> **0/5** | 5.801e-11 -> **1.900e-04** | 4/5 -> **0/5** |
| 10 | 66 | **44.457 / 44.457** | **31.556 / 31.556** | 2.525e-11 -> **5.084e-06** | 5/5 -> **0/5** | 2.710e-11 -> **1.058e-06** | 4/5 -> **0/5** |
| 12 | 91 | **10.841 / 10.841** | **11.829 / 11.829** | 1.499e-11 -> **3.750e-08** | 5/5 -> **0/5** | 1.665e-11 -> 7.349e-09 | 4/5 -> 4/5 |
| 14 | 120 | **3.718 / 3.718** | **5.419 / 5.419** | 1.025e-11 -> 3.203e-10 | 5/5 -> 4/5 | 1.116e-11 -> 3.662e-11 | 4/5 -> 4/5 |
| **16 (shipped)** | **153** | **2.371 / 2.371** | **1.683 / 1.683** | 7.682e-12 -> 2.978e-12 | 5/5 -> 4/5 | 8.465e-12 -> 2.517e-13 | 4/5 -> 4/5 |
| 18 | 190 | **1.044 / 1.044** | **0.655 / 0.655** | 6.224e-12 -> 2.690e-14 | 5/5 -> 4/5 | 6.968e-12 -> 1.375e-15 | 4/5 -> 4/5 |
| 20 | 231 | **0.321 / 0.321** | **0.301 / 0.301** | 5.262e-12 -> 2.598e-16 | 5/5 -> 4/5 | 5.951e-12 -> **0.000e+00** | 4/5 -> 4/5 |

Each accuracy cell is `chebyshev / zernike`.  **They are the same number in every cell**, to the three
decimals the estimator resolves, at eight orders and two decentres -- and the Chebyshev column reproduces
WP-A26's own ladder (44.457 / 31.556 at 10, 2.371 / 1.683 at 16, 1.044 / 0.655 at 18) digit for digit,
which is what says this rebuilt fixture is that fixture.

On axis, order 6, the branch this work package does not touch: **41.089 urad on both bases**, with the
census reading `rcond 1.415e-11, 3 of 3 solves singular` on Chebyshev and `9.785e-01, 0 of 3` on Zernike.

Three more readings from the same runs:

* **the returned FIELD, not just the figure.**  At the fixture's default `ray_subsample=2`, where the
  inverse-characteristic model is engaged, `apply_real_lens_traced(..., fit_basis='zernike')` returns a
  field that is `np.array_equal` to the Chebyshev one, on axis and at 1.0 w alike -- the model supplies the
  OPL, the entrance coordinates and `det J` per pixel, so the forward fit's basis does not reach the
  returned field at all, exactly as fix D5 / `FIX_G8_PROBE` record for its ORDER.  At `ray_subsample=1`,
  where it does reach the field, the two bases agree to the estimator's three decimals at every order
  above.
* **the least-squares answer itself.**  `||b - A x||` of the returned coefficients against an independent
  QR solve of the same system reads **1.000000 in every row of the table**, on both bases -- with one
  exception, order 20 at 1.0 w on the Zernike basis, where the Gram loses positive-definiteness outright
  (`rcond 0.000e+00`) and the answer comes back at 1.000004.  Section 6 measures how much of that is the
  C13 step-down: below order 20, none of it.
* **the fit itself, held out from its own lattice and scored on the oracle.**  Both bases fit the exact
  conic OPL under D1's skirt weights and are then evaluated at points that are NOT fit samples, inside the
  beam disc, against the closed-form sphere: rms **4.0349e-08 m** at order 6 and **1.8857e-10 m** at the
  shipped 16, with the two bases agreeing to 2.3e-12 and 1.2e-10 OF THAT ERROR.  The element's exit-slope
  estimator is this quantity with a Newton inversion and a phase unwrap in front of it, and it says the
  same thing.
* **the niche-C11 arbiter.**  Its two candidate residuals, measured on both bases in the same call, at
  0.5 w: `1.073452e-07 m off-centre against 1.735703e-07 m concentric` at order 6 -- identical to all seven
  printed digits across the bases, and still agreeing to five digits at order 20 (9.845194e-10 against
  9.844878e-10) where the solve itself is near-singular.  The verdict is the same at every order.  The
  audit's expectation that a disc basis would leave it "nothing to arbitrate" does not survive: what it
  arbitrates is two DIFFERENT DISCS, and a change of basis inside one span cannot equalise two different
  weighted problems.

---

## 5. D1's adversarial ghost geometry

The fold regularisation's own fixture (a weak R = 32 mm singlet, 12 mm aperture, 0.40 mm beam at 5.6 mm of
decentre, `amplitude_model='ray_density'`, `preserve_input_phase='remap'`), scored the way
`test_no_fold_and_no_ghost_across_the_adversarial_geometries` scores it: fold-caustic warnings, the
off-beam lobe as a fraction of the on-beam peak, the off-beam power fraction, and sign changes of
`d(x_out)/dx` of the APPLIED forward-map fit over the whole launch lattice.

| order | basis | folds | off-beam / peak | off-beam power | sign changes of `d(x_out)/dx` | min rcond | step-down |
|---|---|---|---|---|---|---|---|
| 6 | chebyshev | 0 | 1.698e-04 | 1.295e-08 | 0 | 1.802e-14 | 7 of 8 |
| 6 | zernike | 0 | 1.698e-04 | 1.295e-08 | 0 | 1.802e-14 | **2 of 8** |
| 8 | chebyshev | 0 | 1.698e-04 | 1.295e-08 | 0 | 1.802e-14 | 7 of 8 |
| 8 | zernike | 0 | 1.698e-04 | 1.295e-08 | 0 | 1.802e-14 | **2 of 8** |
| 10 | chebyshev | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 7 of 8 |
| 10 | zernike | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 6 of 8 |
| 12 | chebyshev | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 7 of 8 |
| 12 | zernike | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 6 of 8 |
| 14 | chebyshev | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 7 of 8 |
| 14 | zernike | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.246e-16 | 6 of 8 |
| **16** | chebyshev | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 6 of 7 |
| **16** | zernike | 0 | 1.756e-04 | 1.346e-08 | 0 | 0.000e+00 | 5 of 7 |
| 18 | chebyshev | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 6 of 7 |
| 18 | zernike | 0 | 1.756e-04 | 1.346e-08 | 0 | 0.000e+00 | 5 of 7 |
| 20 | chebyshev | 0 | 1.756e-04 | 1.346e-08 | 0 | 1.802e-14 | 6 of 7 |
| 20 | zernike | 0 | 1.756e-04 | 1.346e-08 | 0 | 0.000e+00 | 5 of 7 |

**Nothing D1 cares about moves, on either basis, at any order**: zero fold-caustic warnings, zero sign
changes of the applied forward map's `d(x_out)/dx` over the whole launch lattice, and an off-beam lobe at
1.698e-04 (orders 6-8) / 1.756e-04 (10-20) of the on-beam peak either way -- the same 1.76e-04 WP-A26
recorded across its own order ladder.  The off-beam figures move with the ORDER and not with the basis,
which is the same statement the d7 table makes.

The `min rcond` column reads 1.802e-14 for Chebyshev at every order because the worst solve of a ghost call
is NOT the forward fit: it is the inverse-characteristic model's own total-degree-14 exit fit over the exit
coordinates, which this work package deliberately leaves on the Chebyshev basis (section 13).  WP-A26
measured the same 1.802e-14 and attributed it the same way.  So on this fixture the observable that moves
is the step-down COUNT, and it moves the way the d7 fixture's rcond does: **7 of 8 solves screen singular
on the square basis and 2 of 8 on the disc basis at orders 6-8**, the advantage decaying with order until
only the imap's own Chebyshev solves are left firing.

---

## 6. The conditioning census, and the law the advantage decays by

The census is the one niche C13's tests use: for every least-squares solve the call makes, the equilibrated
Gram reciprocal condition number `_gram_rcond(A^T A)` -- which is what the step-down screens on, and which
is diagonally equilibrated, so none of this is a statement about column UNITS -- and `||b - A x||` of the
returned coefficients against an independent QR solve of the same system.

**The residual ratio is 1.000000 in every row of every table above, on both bases, at every order from 6 to
20** -- one exception, order 20 at 1.0 w on the Zernike basis, at 1.000004.  Both bases return the
least-squares answer.

And that is NOT the C13 step-down carrying it, which is worth measuring rather than assuming.  Driven with
`LSTSQ_CONDITIONING_STEPDOWN` forced both ways on the decentred fit of the test file's own fixture:

| order | basis | rcond | residual ratio vs QR, step-down ON | OFF |
|---|---|---|---|---|
| 10 | zernike | 1.041e-06 | 1.000000 | 1.000000 |
| 14 | zernike | 3.563e-11 | 1.000000 | 1.000000 |
| 16 | zernike | 2.359e-13 | 1.000000 | 1.000000 |
| 18 | zernike | 1.260e-15 | 1.000000 | 1.000000 |
| 20 | zernike | 1.624e-17 | **1.000003** | **1.000188** |
| 10-20 | chebyshev | 1.3e-11 .. 3.4e-11 | 1.000000 | 1.000000 |

So the normal equations answer the Zernike system correctly even where its equilibrated Gram screens at
1e-15, and C13 is a no-op below order 20 and worth 1.9e-04 of the residual there.  Nothing in this basis
is relying on the step-down to be safe -- which matters, because niche D7's own hard-mask arm shows what
relying on it looks like: a returned fit that misses the least-squares residual by 3.7e+04x
(`_DRAW_RESID_RATIO` = 10 is the bar between the two regimes, and 1.000188 is four decades under it).

**Where the samples are a disc.**  Three readings, all from calls above:

| fit | samples | terms | Chebyshev rcond | Zernike rcond | step-down |
|---|---|---|---|---|---|
| the on-axis call's three applied fits (hard NaN mask at r <= 0.90 mm) | 39 565 | 28 | 1.415e-11 | **9.785e-01** | 3 of 3 solves -> **0 of 3** |
| the C11 arbiter's CONCENTRIC trial fit inside the decentred call (hard mask) | 118 717 | 28 | 3.108e-08 | **9.863e-01** | fires -> does not |
| a disc-masked order-12 fit of the exact conic OPL (the test file's fixture) | 1 606 | 91 | **0.000e+00** | **3.326e-01** | fires -> does not |

A Gram of 0.98 is the identity to within the lattice's own discretisation of the disc: the basis is
orthonormal under the measure the samples actually carry, which is the entire design claim.

**Where they are a square.**  The decentred branch's three applied fits, whose rows are the whole launch
lattice at D1's two weights.  Same table as section 4, read as the ratio `rcond_zernike / rcond_chebyshev`
-- i.e. how many times better conditioned the disc basis is:

| order | 0.5 w | 1.0 w |
|---|---|---|
| 6 | **7.7e+08 x better** | 3.2e+08 x |
| 8 | 1.4e+07 x | 3.3e+06 x |
| 10 | 2.0e+05 x | 3.9e+04 x |
| 12 | 2.5e+03 x | 4.4e+02 x |
| 14 | 31 x | 3.3 x |
| **16** | **0.39 x (worse)** | **0.030 x** |
| 18 | 4.3e-03 x | 2.0e-04 x |
| 20 | 4.9e-05 x | 0 (the Gram loses positive-definiteness) |

The advantage is 7.7e+08 at order 6 and falls by a measured factor of ~80 every two degrees (8.9x per
degree), crossing 1.0 between orders 14 and 16 on the 0.5 w arm and between 12 and 14 on the 1.0 w arm.

**The law.**  A degree-`n` Zernike column normalised to a disc of radius `R` grows as `(r/R)^n` outside it;
this fixture's launch lattice reaches `r/R = 4.01` at the corner and D1's skirt keeps every one of those
rows at weight `w_out = 3.288e-05`.  The skirt's share of the Gram therefore rises by `(r/R)^2 = 16.1` per
degree while the disc's does not move, so the basis stops being orthogonal under the sample measure at a
rate of about 1.2 decades per degree.  Measured, the advantage decays a little slower than that (0.95
decades per degree) because the square basis's own conditioning also drifts down across the ladder
(1.70e-10 at order 6 to 5.26e-12 at order 20) -- the crossover is where the two curves meet, and it is a
property of `R_data / R_disc`, not of the element.

That is A26-7 -- "the launch lattice is a SQUARE inside a DISC-shaped domain" -- priced.  It is also why
the default cannot move on this evidence: at the SHIPPED `_DECENTRED_FIT_POLY_ORDER = 16` the decentred
branch is past the crossover, and what a disc-orthogonal basis buys there is negative.

---

## 7. Byte identity of the default basis

Two library trees, ONE variable.  `git archive HEAD lumenairy` was extracted twice into the scratch
directory; in the second extraction, and only there, `lumenairy/elements/_lens_traced.py` was replaced by
the working copy.  So the other engineers' uncommitted edits in the working tree cannot enter the
comparison, and the only difference between the two trees is this work package.  Each tree was driven by
the same probe in its own CHILD PROCESS with `cwd` and `PYTHONPATH` set to that tree and the
editable-install finder removed from `sys.meta_path`, with `lumenairy.__file__` asserted to be inside the
extraction before anything was measured.  Never through pytest.  `git archive` is the only git command
used and it only reads.

Fifteen configurations, md5 of the returned field's bytes plus `np.array_equal`:

```
concentric              md5 SAME  array_equal True  max|d| 0.000e+00
decentred_1w            md5 SAME  array_equal True  max|d| 0.000e+00
decentred_pre_d7        md5 SAME  array_equal True  max|d| 0.000e+00
decentred_order20       md5 SAME  array_equal True  max|d| 0.000e+00
spline                  md5 SAME  array_equal True  max|d| 0.000e+00
inverse_map_off         md5 SAME  array_equal True  max|d| 0.000e+00
fwd_pre_d7              md5 SAME  array_equal True  max|d| 0.000e+00
fwd_order20             md5 SAME  array_equal True  max|d| 0.000e+00
fwd_concentric          md5 SAME  array_equal True  max|d| 0.000e+00
sub1_concentric         md5 SAME  array_equal True  max|d| 0.000e+00
sub1_decentred          md5 SAME  array_equal True  max|d| 0.000e+00
sub1_decentred_pre_d7   md5 SAME  array_equal True  max|d| 0.000e+00
inversion_fit           md5 SAME  array_equal True  max|d| 0.000e+00
ghost_ray_density       md5 SAME  array_equal True  max|d| 0.000e+00
prepared_screen         md5 SAME  array_equal True  max|d| 0.000e+00
MISMATCHES: 0
```

**And the battery is not vacuous**, which is the half of a byte-identity claim that usually goes unstated:
the same fifteen cases return **12 DISTINCT fields**, and three of the distinct ones differ ONLY in the
forward fit's order on the path where that fit reaches the returned field (`fwd_pre_d7`, `inverse_map_off`
at the shipped order, `fwd_order20` -- three different md5s; likewise `sub1_decentred` against
`sub1_decentred_pre_d7`).  A change that moved the default fit by one ULP would show here.

The four cases that share an md5 -- `decentred_1w`, `decentred_pre_d7`, `decentred_order20`, `spline` --
share it on BOTH trees, and that is fix D5 / `FIX_G8_PROBE`'s finding rather than a defect in the probe:
with the inverse-characteristic model engaged the forward fit's order (and backend, and now basis) does not
reach the returned field at all.  It is why the battery also carries the `fwd_*` and `sub1_*` cases.

---

## 8. Cost

Medians of 7 INTERLEAVED runs of one decentred `apply_real_lens_traced` (N = 512, dx = 8 um,
`ray_subsample=8`, the shipped order 16) -- the same call shape WP-A26 priced its order raise with, on a
box shared with other jobs:

| basis | median | min |
|---|---|---|
| chebyshev | **412.5 ms** | 396.9 ms |
| zernike | **463.9 ms** | 433.2 ms |

**1.12x**, paid only when asked for.  On the `ray_subsample=1` ladder above -- 405 769 rays, the whole call
with the census instrument attached (which doubles every solve with a QR, on both bases alike) -- the same
comparison reads 42.3 s against 71.6 s at order 16 and 58.6 s against 104.0 s at order 20, i.e. **1.69x /
1.77x**: there the per-pixel Newton evaluation of the fits is a much larger share of the call.  Both
numbers are the same cause, which is that the Zernike evaluation has no numba kernel (section 13,
B10-D2) -- the Chebyshev path drops into a `@njit(parallel=True, fastmath=True)` recurrence per sample and
the Zernike path runs a chunked column generator in NumPy.

No test asserts either number.  What the test file pins instead is the step-down firing count, which is
what the conditioning claim is actually about.

---

## 9. One defect this work package introduced, and how it was found

Between the first implementation and this report, the new keyword reached `_decentred_fit_score`
UNCONDITIONALLY -- `basis=_fit_basis, basis_disc=...` on every call, including the default one.  Niche C12
wraps that function with `def s_spy(*a)`, a positional-only spy, so
`test_niche_c12_physics_fit_selection.py::test_the_two_selectors_agree_on_the_slow_fixture` failed with
`TypeError: s_spy() got an unexpected keyword argument 'basis'`.  A verifier measured it on the live tree
and reported it.

Fixed the way the evaluator's own call site already was: the keywords are assembled into a dict that is
EMPTY on the default basis (`:11661-11672`), so every default call is spelled exactly as it was.  That is
the same discipline the three shipped `_Cheb2DEvaluator` calls follow, and for the same reason -- niches
C1, C12 and D7 all replace one of these functions with a fixed-signature spy, and a byte-identity contract
that only holds for the ARGUMENT VALUES is not the contract this module's tests rely on.  The whole
verification set below was run after the fix; `test_niche_c12_physics_fit_selection.py` is green in it.

---

## 10. Files touched

| file | what |
|---|---|
| `lumenairy/elements/_lens_traced.py` | the basis (`:3691-3897`), the evaluator's design and evaluation branches (`:3299-3345`, `:3455-3460`, `:3533-3534`, `:3621-3660`), the pool state (`:2281-2284`), the arbiter's per-candidate basis (`:4566-4600`, `:11661-11682`), the applied disc (`:11792-11800`), the fit site (`:12008-12029`), the keyword, its documentation and its gate on both traced entry points (`:7811`, `:8490-8534`, `:9152-9177`, `:15067`, `:15223`) |
| `docs/history/lumenairy.elements._lens_traced.md` | re-recorded fingerprints (`scripts/record_history_fingerprints.py`) |
| `tests/unit/test_audit2609_b10_zernike_fit_basis.py` | new |
| `docs/audits/.../fixes/WP-B10_REPORT.md`, `WP-B10_CHANGELOG.md` | new |

---

## 11. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, `-p no:randomly`, one process at a
time, on the working tree (which also carries other engineers' in-progress edits to
`lumenairy/raytrace/`, `carrier.py`, `_lens_real.py`, `rcwa/` -- the byte-identity proof in section 7 is
what isolates this work package from them).

| command | result | duration |
|---|---|---|
| `pytest tests/unit/test_audit2609_b10_zernike_fit_basis.py` (new) | **30 passed** | 4.6 s |
| `pytest` `test_niche_c11_decentred_fit_arbiter` + `c12_physics_fit_selection` + `c13_lstsq_conditioning` + `c6_fit_guard` + `fix_d5_fit_domain_basis` | **119 passed** | 42.2 s |
| `pytest test_fix_newton_pool_memory.py test_niche_newton_pool_both_fits.py` | **78 passed, 1 skipped** (numexpr) | 79.2 s |
| `pytest` `a3_traced_lens` + `a3_caustic_siblings` + `a3_verify_traced` + `a26_decentred_exit_reference` + `a17_history_lint` + `a17_history_relocation` | **832 passed, 2 failed** -> the fingerprint drift, cured below | 113.0 s |
| `pytest` the four `a16_*` files | **164 passed, 2 failed** -> the KNOWN RED of section 12 | 15.6 s |
| `pytest tests/unit/test_niche_d1_tilted_carrier.py` | **33 passed** | 73.9 s |
| `pytest tests/unit -k real_lens` | **158 passed, 3 skipped, 2 failed** (the same two a16 ids, which match on the entry-point name) | 262.6 s |
| `pytest tests/unit/test_niche_d7_decentred_fit.py` (slow) | **38 passed** -- the default is still 2.371 / 1.683 urad | 297.0 s |
| `pytest tests/unit/test_niche_d6_exact_tilted_leg.py` (slow) | **38 passed** | 171.3 s |
| `python validation/run_all.py test_lenses` | **ALL 1 files passed** | 25.4 s |
| `ruff check` on `_lens_traced.py` + the new test file | **All checks passed!** | 1 s |
| `python scripts/record_history_fingerprints.py lumenairy/elements/_lens_traced.py --reason ...` | re-recorded (drift shown, then written) | 3 s |
| `python scripts/record_history_fingerprints.py --check` | **`OK lumenairy.elements._lens_traced.md`** | 4 s |
| `pytest test_audit2609_a17_history_lint.py` | **5 passed** -- the narrative ratchet did not move | 8.8 s |

**Two reds, both named before the run, neither mine to fix:**

1. `test_audit2609_a16_lens_config_round_trip.py::test_every_keyword_is_classified_as_field_contract_or_documented_exclusion[apply_real_lens_traced]` and `[prepare_real_lens_traced]` -- the new keyword is not yet classified in `lens_config.py`, which is section 12's request.  It is the gate working: a new keyword on a config-carrying entry point must be classified by someone in writing.
2. `test_audit2609_a17_history_relocation.py` -- BEFORE the re-record, two ids for `lumenairy.elements._lens_traced`; those are green after it.  What is still red in that file afterwards is `lumenairy.propagators.system` (ast + token) and `lumenairy.elements._lens_real` (token), i.e. two OTHER modules another engineer has open -- `--check` prints `OK lumenairy.elements._lens_traced.md` beside their `DRIFT` lines.

Everything else is green.  Nothing in the verification set was skipped, and no test was modified to make it
pass.

---

## 12. Requested changes outside my ownership

`fit_basis` is a new keyword on two entry points, and
`test_audit2609_a16_lens_config_round_trip.py::test_every_keyword_is_classified_as_field_contract_or_documented_exclusion`
is the gate that refuses to let a keyword exist without someone deciding, in writing, whether it is
configurable.  It is configurable: it is a HOW-is-this-discretised setting whose effect on the answer is
bounded by the fit's own conditioning, which is exactly `LensNumerics`' charter.  A verifier is working in
`lens_config.py`, so the four edits below are requested rather than taken.  Until they are applied, that
one test fails for exactly two ids and for no other reason -- section 11 records it as a KNOWN RED with the
exact message:

```
FAILED test_audit2609_a16_lens_config_round_trip.py::
    test_every_keyword_is_classified_as_field_contract_or_documented_exclusion[apply_real_lens_traced]
FAILED test_audit2609_a16_lens_config_round_trip.py::
    test_every_keyword_is_classified_as_field_contract_or_documented_exclusion[prepare_real_lens_traced]
AssertionError: prepare_real_lens_traced: ['fit_basis'] are keyword parameters that lens_config neither
carries as a field nor documents as deliberately keyword-only.
```

The census edit (12.3) is NOT needed yet and must land WITH the field, not before it: the count only grows
when `LensNumerics` grows, and `test_to_kwargs_emits_only_requests_unless_asked_for_defaults` is green on
this tree today.

### 12.1 `lumenairy/elements/lens_config.py` -- the field

In `LensNumerics`' docstring, immediately after the `newton_poly_order` entry:

```rst
    fit_basis : str, default 'chebyshev'
        Design basis of the traced ray fits: ``'chebyshev'`` (tensor Chebyshev
        on the launch square) or ``'zernike'`` (orthonormal on the ray-fit
        disc).  The two span the same polynomial space at the same total
        degree, so this changes the CONDITIONING of the fit's least squares and
        not the fit; ``'zernike'`` requires ``newton_fit='polynomial'`` and
        ``inversion_method='newton'``.  The membership check lives in the entry
        point (see the module docstring).  Accepted by:
        ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
```

and, in the field block, immediately after `newton_poly_order: int = 6`:

```python
    fit_basis: str = 'chebyshev'
```

No `__post_init__` clause: `newton_fit`, `inversion_method` and `amplitude_model` are the three siblings
with exactly this shape (a string whose vocabulary the entry point owns), and
`test_post_init_accepts_every_documented_legal_value` would then also need the new value.  If the verifier
prefers a `_require_choice(fn, 'fit_basis', self.fit_basis, _vocab('_VALID_FIT_BASES'))`, the vocabulary
tuple is `lumenairy/elements/_lens_traced.py:3711` and is exported for exactly that purpose.

### 12.2 `lumenairy/elements/lens_config.py` -- the two table entries

```python
_NUMERICS_FOR['apply_real_lens_traced']:   'fit_basis': 'fit_basis',
_NUMERICS_FOR['prepare_real_lens_traced']: 'fit_basis': 'fit_basis',
```

(added beside `'newton_poly_order': 'newton_poly_order'` in both dicts).

### 12.3 `tests/unit/test_audit2609_a16_lens_config_round_trip.py:274` -- the census

```python
-    assert len(everything) == n_fields == 39, (
+    assert len(everything) == n_fields == 40, (
```

one field more on `LensNumerics`.

### 12.4 `docs/lens_configuration.md` -- the field table

The `LensNumerics` heading at `:264` reads `(18 fields)` and becomes `(19 fields)`; the row goes after
`newton_poly_order`:

```
| `fit_basis` | `'chebyshev'` | -- | Y | Y | -- | -- | -- | -- |
```

Nothing else in that document needs to change: the worked examples do not enumerate the fields.

---

## 13. Deferred

**B10-D1 -- the inverse-characteristic model's exit fits, and niche C12's box spectrum, stay Chebyshev.**
`fit_basis` selects the basis of the ENTRANCE-plane fits, whose disc is the ray-fit disc.
`_lens_imap.build_inverse_map`'s total-degree-14 solves fit over the EXIT coordinates -- a different
domain, whose natural disc is the exit hull and not this one -- and `inversion_method='fit'` does the same,
which is why that combination is refused rather than silently half-applied.  `_decentred_fit_spectrum`
stays Chebyshev for a stronger reason: niche C12's disc-inflation law (`restricting to a concentric
sub-domain of relative radius s scales the degree-n contribution by s^n`) is a property of the
box-normalised degree-graded basis and would have to be re-derived shell by shell for a disc-normalised
one.  Both are real extensions; neither is a regression fix.

**B10-D2 -- the Zernike evaluation has no numba kernel.**  The cost in section 8 is almost entirely that:
the Chebyshev path drops into a `@njit(parallel=True)` kernel that runs the recurrence inline per sample,
and the Zernike path runs a chunked generator over `n_terms` columns in NumPy.  A kernel for it is
mechanical (the Jacobi recurrence has the same shape as the Chebyshev one) and would need the same
backend-pinning treatment `_resolved_cheb_backend` gives the existing one, because it would introduce the
second floating-point order that the pool's bit-identity contract is about.  Not attempted here: the basis
ships opt-in and the cost is stated.

**B10-D3 -- the crossover order is a property of the FIXTURE, and making the default move would need the
design-121 surface.**  What sets it is `R_data / R_disc` -- 4.01 on the D7 singlet -- through the
`(R_data/R_disc)^2`-per-degree decay measured in section 6.  A wider beam, a tighter launch square or a
disc-shaped launch lattice all move it, and on a geometry whose retained samples really are the fit disc
there is no crossover at all (the concentric branch, section 4).  A default change would have to be
measured on design 121's own chain, which is local-only (`validation/repro_traced_carrier_121/`), and it is
not proposed.

**B10-D4 -- A26-7 is the change that would make this basis the right default, and it is not a basis
change.**  The element's forward-map domain is a DISC of radius `~launch_radius` (`bound`,
`out_of_domain`), but the launch lattice is the SQUARE that circumscribes it, so the fit's worst data sits
1.41x beyond its own domain and, on the decentred branch, at 4.01 fit-disc radii.  That is what makes a
disc-orthogonal basis orthogonal under the wrong measure at high order.  Trimming the corners is NOT free
-- WP-A26 section 3.1 measured 11.934 / 18.920 urad for it and doing it by deletion folds the map -- so
this stays what A26 recorded it as: a real inconsistency with no cheap cure.  What B10 adds is the price of
it, in decades of conditioning per degree.

**B10-D5 -- A26-6 is untouched, and this basis cannot touch it.**  The weighted restriction's strength is
blind to the skirt's residual MAGNITUDE; that is a statement about ROWS (which samples, at which weight),
and `fit_basis` changes only COLUMNS.  Recorded here so that the two are not confused later.
