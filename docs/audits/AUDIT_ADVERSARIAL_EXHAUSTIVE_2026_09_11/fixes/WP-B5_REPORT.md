# WP-B5 report -- RCWA / EME / BOR: the three items WP-A14 deferred with designs

Audit `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`, `fixes/WP-A14_REPORT.md` section 6 items
D1, D2, D3 (findings H4 and H3).  Branch `audit-fixes-2026-09`, base HEAD `284daccc`
(WP-B3); the branch advanced to `8dab7de5` (WP-B1 / B2 / B6) while this work package ran,
and `git diff 284daccc..8dab7de5 -- elements/rcwa elements/eme elements/bor
elements/berreman.py` is EMPTY, so the `284daccc` archive is still the exact pre-change
state of every file measured here.  Every measurement below was taken on this machine with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`, one process at a time.

All three items were re-confirmed on the current HEAD before anything was changed; the line
numbers WP-A14 cited had moved (`oned.py:131 -> :135`, `:652 -> :708`) with no change of
substance.

---

## 1. Summary

| # | Status | Files:lines | Tests | Oracle | Measured before -> after |
|---|---|---|---|---|---|
| **D2** (H4, perf) | **fixed** | `rcwa/_core.py:3343` (new `_redheffer_star_rt`), `:2893`, `:5067`; `oned.py:721,:756,:1153`; `twod.py:1253,:1377,:2035,:2418` | `test_audit2609_b5_rcwa_eme_bor.py::test_d2_closed_form_matches_the_independent_star_oracle`, `::..._on_the_metallic_chain`, `::test_d2_a_single_layer_solve_records_one_star_inverse_not_two`, `::test_d2_the_retained_denominator_is_the_tighter_reading`, `::test_d2_zero_block_shortcuts_take_no_inverse_at_all`, `::test_d2_energy_closes_through_the_closed_form`, `::test_d2_the_multi_layer_even_fold_still_assembles_its_star`, `::test_d2_two_d_entry_points_still_close` | the Redheffer composition from its DEFINING coupled system, solved whole as one dense `2n` system (written in the test file, no star algebra) | 1-D Ag TM solve **16.8 -> 14.0 / 80.3 -> 65.4 / 514.8 -> 377.1 / 7533 -> 3973 ms** at `n_orders` 50/100/200/400 (**1.20x / 1.23x / 1.37x / 1.90x**); the star's share **20.2/18.7/21.6/70.8 % -> 5.0/6.8/6.7/45.5 %**; `np.linalg.inv` per solve **6 -> 5**; guarded-inverse census rows per 1-D single-layer solve **4 -> 3**.  Answer moves **<= 1.665e-15 absolute / 3.114e-15 relative** over 382 arrays. |
| **D3** (H3) | **fixed** | `rcwa/twod.py:542` (`_li_convolutions_2d_tensor_full`, symmetrized), `:585` (`_li_tensor_full_l2l1`), `:1977` | `::test_d3_offplane_fff_nv_keeps_the_cells_own_mirror`, `::test_d3_offplane_fff_nv_is_mirror_covariant`, `::test_d3_uniform_rotated_director_matches_the_berreman_oracle`, `::test_d3_separable_stripe_is_unchanged_by_the_symmetrisation`, `::test_d3_reduces_exactly_to_the_in_plane_operator_it_generalizes`, `::test_d3_energy_closes_on_the_off_plane_path` | the x<->y MIRROR of Maxwell's equations (the cell's own, and the general covariance `J -> P J P`); a conical Berreman 4x4 on a uniform rotated-director cell | `\|Jxx - Jyy\|` on a mirror-symmetric cell: square **9.10e-04 -> 2.4e-15** (M = 3), **3.32e-04 -> 8.5e-15** (M = 6); disk **4.71e-03 -> 3.6e-15**, **2.35e-03 -> 2.3e-15**.  Mirror covariance **2.03e-04 .. 5.01e-03 -> 5.1e-15 .. 1.6e-14**.  Berreman agreement 1.1e-15 .. 7.1e-15, unchanged. |
| **D1** (H4, perf) | **measured, REFUSED at both sites** (nothing shipped) | -- (`oned.py:135`, `:708` unchanged) | `::test_d1_levinson_lands_outside_what_the_package_closure_admits`, `::test_d1_the_inverse_rule_toeplitz_is_not_the_ill_conditioned_matrix`, `::test_d1_the_two_sites_still_form_the_explicit_inverse`, `::test_d1_guarded_inverse_is_untouched_by_this_work_package` | the row-equilibrated backward residual of the system each candidate solves -- the same instrument `_guarded_inverse` scores an inverse on | `solve_toeplitz` costs **12x .. 20x** the shipped composite (449 ms vs 37.8 at `n_orders` 200; 3918 vs 192 at 400) and lands **two decades** further from its own equation (4.15e-13 vs 6.22e-15), moving the answer **4.1e-12** relative -- above the package's 1.4e-13 closure.  `cond([[1/eps]])` over the whole metallic ladder is **2.51e+02**, not the 1e13 the deferral attributed to it. |

---

## 2. Per item

### D2 (H4, perf) -- the two-interface closed form

**What was there.**  Every single-layer entry point builds
`interface(sup|layer) -> propagation -> interface(layer|sub)` and ends in
`_redheffer_star`, whose zero-block fast path cannot fire (both `A22` and `B11` are
non-zero).  It assembled all four `2N x 2N` output blocks -- twelve matrix products and two
`_guarded_inverse` calls -- and every caller then read exactly `S11 @ cinc` and
`S21 @ cinc`, one or two columns, discarding `S12` and `S22` entirely.

Re-confirmed on HEAD before changing anything (`scratchpad/b5/prof_sites.py`, Ag TM ladder,
best of three interleaved runs):

| `n_orders` | `N` | whole solve | final star | 2 interfaces | 6 x `inv` |
|---|---|---|---|---|---|
| 50 | 101 | 16.8 ms | 3.4 ms (20.2 %) | 4.1 ms | 2.9 ms |
| 100 | 201 | 80.3 ms | 15.0 ms (18.7 %) | 27.8 ms | 18.3 ms |
| 200 | 401 | 514.8 ms | 111.2 ms (21.6 %) | 189.6 ms | 112.7 ms |
| 400 | 801 | 7533 ms | 5337 ms (70.8 %) | 950 ms | 2072 ms |

WP-A14 costed this site at ~8 %, which is the two INVERSES alone; the twelve products are
the larger half and were not counted.  The `n_orders = 400` row is worth its own sentence:
the same star on SYNTHETIC operands of the same size takes 681 ms, so 4.7 s of the 5.3 is
SUBNORMAL arithmetic -- `A22 = X S22 X` (`_propagation_star`) underflows into subnormals in
the evanescent tails, where `exp(-lam k0 L)` reaches ~1e-273 and its square is below the
normal range.  Primitive costs at one BLAS thread for scale: at `N = 801`, `gemm` 51.8 ms,
`inv` 100.4 ms, `solve` (N right-hand sides) 98.6 ms, one mat-vec 0.30 ms.

**What I changed and why.**  New `_redheffer_star_rt(SA, SB, cinc)` in `_core.py:3343`
returns `(S11 @ cinc, S21 @ cinc)` without assembling the star.  With
`D = (I - B11 A22)^-1`, `u = A21 c`, `w = B11 u` and `z = D w`, the push-through identity
`(I - A22 B11)^-1 = I + A22 D B11` (equivalently `F A22 = A22 D`) removes the second inverse
outright:

    S11 c = A11 c + A12 z              S21 c = B21 (u + A22 z)

-- seven mat-vecs per source column, ONE `2N` product (`B11 @ A22`) and ONE inverse, against
twelve products and two inverses.  It is wired at the eight sites that destructure
`S11, _S12, S21, _S22` and use nothing else: `oned.py:721` (the planar TE/TM fast path),
`oned.py:756` (the 2N path of `rcwa_efficiency_1d`), `oned.py:1153`
(`_jones_1d_from_profiles`, serving `rcwa_jones_1d` and `rcwa_jones_1d_segments`, both
polarizations as one two-column block, in-plane and full-3x3), `twod.py:1253`
(`rcwa_efficiency_2d`), `twod.py:1377` (`PreparedRCWA2D.solve`), `twod.py:2035`
(`rcwa_jones_2d`, both branches), `twod.py:2418` (`rcwa_efficiency_2d_shapes`) and
`_core.py:2893` (`_symmetric_solve_rt`, the single-layer even-parity fold).

**The guard is preserved on the inverse that remains, and this was checked rather than
assumed.**  `I - B11 A22` goes through `_guarded_inverse` under the same site string
`"rcwa Redheffer star (I - B11 A22)"`, so the M1 census records it identically.  A 1-D
single-layer solve censuses three guarded inverses instead of four (measured: `x2 rcwa
interface mode-match (a+b)` + `x1 rcwa Redheffer star (I - B11 A22)`, against the same plus
`x1 ... (I - A22 B11)` before).  Three things make the dropped row lossless:

* the two matrices are SIMILAR (`(I - A22 B11) A22 = A22 (I - B11 A22)`) and share a
  spectrum;
* what the census actually reads is the EQUILIBRATED `rcond`, which similarity does not
  preserve -- so it was measured.  On the metallic ladder the RETAINED matrix is the
  tighter reading at every rung: 0.340 / 0.167 / 0.108 / 0.0695 against the dropped one's
  0.523 / 0.670 / 0.671 / 0.671 at `n_orders` 11 / 50 / 100 / 200, and it is the one that
  keeps falling with truncation.  On the thin-grating family the M1 census was taken on
  (`period` 10 um, `dn` 0.05, `n_orders` 4..48, both polarizations) the two agree to within
  6 % and both sit at 0.87 .. 1.00, with equilibrated residuals 1.1e-16 .. 3.2e-16 -- the
  guard is inert there, on both;
* neither star inverse can REFUSE: `rcond_refuse` is `None` at both call sites (it is armed
  only on `_interface_smatrix_general`'s `T22`), so the refusal path is untouched.  A
  fixture that trips the guard today trips it identically after.

The two zero-block shortcuts `_redheffer_star` takes are taken here too, on the same
concrete `.any()` tests behind the same `is_jax_array` backend guard, so a chain that pays
no star inverse today still pays none (asserted, with an empty census).

**NOT applied to `_symmetric_cascade_rt`** (`_core.py:2932`), deliberately:
`elements/pmm/stack2d.py:1576` and `pmm/twod_jones.py:996` fold their own cascades through
it, and closing its last star on the sources would move the PMM engines' last bits from
inside the RCWA package -- another work package's numbers, changed by me, in a release where
`pmm/` is being edited concurrently.  A comment says so at the site and a test pins the
decision.  `_symmetric_solve_rt`, its single-layer twin, has no such caller and does use the
closed form.

**How I verified.**

*Correctness, against an oracle the library did not produce.*  `_star_oracle` in the test
file composes the two S-matrices from their DEFINITION -- the gap's up/down amplitudes
solved as one dense `2n` system `[[I, -A22], [-B11, I]] [u; v] = [A21 c; 0]` -- with no
star algebra, no block inverse and no push-through identity.  Relative agreement:

| operands | assembled star | closed form | ratio |
|---|---|---|---|
| random `n` = 12 / 24 / 48 | 1.20e-15 / 4.39e-15 / 3.94e-14 | 1.55e-15 / 4.63e-15 / 1.56e-14 | 1.29 / 1.05 / 0.40 |
| captured Ag chains, `n_orders` 11 (TE/TM) / 50 / 100 | 1.60e-16 / 1.95e-16 / 5.26e-16 / 2.56e-16 | 1.13e-16 / 1.43e-16 / 1.28e-16 / 2.56e-16 | 0.71 / 0.74 / 0.24 / 1.00 |

The two formulations sit at the same floor; neither is systematically the worse.

*The default movement, against the pre-change tree.*  `git archive HEAD lumenairy` extracted
READ-ONLY into `scratchpad/b5/base/`, imported from a child process whose cwd and
`PYTHONPATH` are the archive (`lumenairy.__file__` asserted and printed on both sides,
never through pytest).  382 arrays over: 192 1-D efficiency configurations (Ag / Au /
dielectric x TE/TM x `'li'`/`'laurent'` x `n_orders` 11/50/100/200 x normal and 12 deg), the
1-D Jones family including the full-3x3 out-of-plane branch, 2-D efficiency and Jones on two
cells x three formulations x two angles, the out-of-plane 2-D Jones, analytic shapes with
and without the even fold, a prepared wavelength sweep, and the must-not-move set:

| group | arrays | byte-identical | max abs | max rel |
|---|---|---|---|---|
| `berreman_jones_1d` | 4 | **4** | 0 | 0 |
| `elements/bor` (`radial_spectrum`, `guided_modes`) | 3 | **3** | 0 | 0 |
| `elements/eme` (`strip_x_modes`, `layer_modes`) | 2 | **2** | 0 | 0 |
| `RCWAStack` 1-D, 1 and 2 layers, `symmetry` off and auto | 12 | **12** | 0 | 0 |
| `rcwa_efficiency_1d` | 192 | 13 | 3.886e-16 | 1.377e-15 |
| `rcwa_jones_1d` (+ full-3x3) | 24 | 0 | 3.331e-16 | 6.930e-16 |
| `rcwa_efficiency_2d` | 48 | 18 | 1.554e-15 | 2.496e-15 |
| `rcwa_jones_2d` (+ out-of-plane) | 84 | 38 | 1.665e-15 | 3.114e-15 |
| `rcwa_efficiency_2d_shapes` | 8 | 0 | 2.220e-16 | 1.647e-15 |
| `PreparedRCWA2D.solve` | 4 | 0 | 4.996e-16 | 1.165e-15 |

Two ladders were run deeper, both base vs live:

* the metallic convergence ladder at `n_orders` 100 / 200 / 300 / 400 x Ag and Au x both
  polarizations: worst movement **3.331e-16 absolute / 1.650e-15 relative**, and
  `sum R + T` identical to twelve printed digits on all sixteen rungs;
* the library's own documented instability class (`period` 10 um, `n_ridge` 1.55 /
  `n_groove` 1.5 on `n_sub = n_sup = 1.5`, `n_orders` 4..48 step 2, both polarizations, 46
  configurations): worst **1.110e-15 / 1.124e-15**, every rung closing at 0 .. 1.1e-15.

*Energy.*  Lossless 1-D closure 4.4e-16 .. 2.2e-13 over `n_orders` 11/51/101 x both
polarizations; 2-D `rcwa_efficiency_2d` 2.2e-16 .. 3.2e-15 and `rcwa_jones_2d` 6.7e-16 ..
6.9e-15 on `'laurent'` / `'li'` / `'fff_nv'`; the out-of-plane path 1.4e-14 / 1.8e-14.

*Cost.*  Medians of three interleaved runs, reported not asserted (the test file gates the
OPERATION COUNT instead):

| `n_orders` | `N` | whole solve before -> after | star share | `np.linalg.inv` |
|---|---|---|---|---|
| 50 | 101 | 16.8 -> 14.0 ms (1.20x) | 20.2 % -> 5.0 % | 6 -> 5 |
| 100 | 201 | 80.3 -> 65.4 ms (1.23x) | 18.7 % -> 6.8 % | 6 -> 5 |
| 200 | 401 | 514.8 -> 377.1 ms (1.37x) | 21.6 % -> 6.7 % | 6 -> 5 |
| 400 | 801 | 7533 -> 3973 ms (1.90x) | 70.8 % -> 45.5 % | 6 -> 5 |

After the change the two `_interface_smatrix` builds are the dominant `O(N^3)` term
(41.7 % at `n_orders` 200) -- see section 6.

**Residual risk, stated exactly.**  The closed form is a RE-ASSOCIATION, and there is one
regime where a re-association is not neutral: if a LAYER mode carries an exponentially
GROWING propagator, `A22` has entries far above 1, `I - B11 A22` is near-singular, and the
answer is a difference of huge terms that no association computes reliably.
`_sqrt_decay`'s `Re(lam) >= 0` branch is exactly what makes that unreachable from the public
API (`|X| <= 1` always).  It IS reachable by monkeypatching that branch, and one existing
test does so -- see section 5, where the measurement and the requested restatement are
recorded.  The regime is documented in `_redheffer_star_rt`'s docstring.

### D3 (H3) -- the off-plane (full 3x3) `fff_nv` operator

**What was wrong.**  H3 symmetrized the in-plane 2x2 Li-2003 operator over its two
factorization orders and explicitly left the full-3x3 one (`_li_convolutions_2d_tensor_full`,
which `rcwa_jones_2d(formulation='fff_nv')` uses for an out-of-plane tensor cell) on the
fixed `L2 L1` order.  The same defect survives there.  Re-confirmed on HEAD at the OPERATOR
level, before any entry point is involved (`max|L2L1 - L1L2|` relative to the operator's own
scale; uniaxial pillar `n_o = 1.5` / `n_e = 1.7` at 40 deg polar in a background of
`eps = 2.25`, 96x96 cell, `n_orders` 3 / 4 / 6):

| cell | director azimuth | relative order difference |
|---|---|---|
| rectangle | 20 deg | 1.67e-04 / 1.62e-04 / 1.56e-04 |
| square | 20 deg | 2.14e-04 / 2.08e-04 / 2.00e-04 |
| disk | 20 deg | 2.92e-04 / 2.76e-04 / 2.60e-04 |
| square or disk | **45 deg** (the mirror plane) | 5.8e-16 / 5.8e-16 / 5.7e-16 |
| disk, ISOTROPIC `eps = 6.25` | -- | 2.15e-02 / 2.02e-02 / 1.88e-02 |

Two rows earn their place.  The isotropic disk is the in-plane H3 number (2.06e-02 at M = 4
there) reproduced THROUGH the 3x3 path, which is the first evidence that this is the same
defect and not a second one.  The 45-degree row is a trap I walked into and report so the
next reader does not: at that azimuth the tensor itself satisfies `P eps P = eps`, and with
a low-contrast pillar (`eps` 2.38 in 2.25, the first fixture tried) the whole order
difference collapses to 5.8e-16 -- not because the operator is symmetric but because there
is nothing for it to be asymmetric ABOUT.  The Jones fixtures below therefore use a
HIGH-contrast pillar (`n_o = 2.0` / `n_e = 2.6` in air), where the same 45-degree cell shows
the artefact at 9e-04 .. 5e-03.

**What I changed and why.**  `_li_convolutions_2d_tensor_full` (`twod.py:542`) returns the
mean of the two orders; the single order is split out as `_li_tensor_full_l2l1`
(`twod.py:585`) and stays reachable through `symmetrize=False`, which is what the
fail-before arms of the new tests use.  The 3x3 transpose is the in-plane argument with the
component permutation `(x, y, z) -> (y, x, z)`: `exx<->eyy`, `exy<->eyx`, `exz<->eyz`,
`ezx<->ezy`, `ezz` alone, together with the transposed pixel grid and the swapped
order-label columns, so `T P L2L1(eps) P T = L1L2(P eps^T P)` and the nine blocks come back
in the same retained-order basis with only their component labels to swap.  The mean is
taken on the RAW `ehat` blocks, so the caller's `l3-` `E_z` fold (`twod.py:1977`) runs AFTER
it -- the mean of two Schur complements is not the Schur complement of the mean.  The
function gained `n_orders_y` (the transposed run needs it as its `n_orders_x`) and a
keyword-only `symmetrize=True`; it has exactly one caller.

**How I verified** (`scratchpad/b5/d3_symm.py`).  Three oracles, none of them the library's
own answer for the quantity being gated.

*(a) The cell's own mirror.*  Director azimuth 45 deg puts the optic axis in the `x = y`
plane, so `P eps P = eps` componentwise, and a transpose-symmetric pattern makes the whole
problem mirror-invariant: `J = P J P`, i.e. `Jxx == Jyy` and `Jxy == Jyx` exactly at normal
incidence.  Jones scale 0.225 (square) / 0.112 (disk):

| cell | `\|Jxx - Jyy\|`, M = 3 / 4 / 5 / 6 |
|---|---|
| square, before | 9.099e-04 / 8.333e-04 / 3.680e-04 / 3.316e-04 |
| square, after | **2.43e-15 / 1.07e-14 / 1.10e-14 / 8.52e-15** |
| disk, before | 4.709e-03 / 3.523e-03 / 2.802e-03 / 2.349e-03 |
| disk, after | **3.57e-15 / 7.61e-15 / 1.52e-14 / 2.32e-15** |

i.e. 0.4 % to 4 % of spurious form birefringence on a cell that has none, removed to this
path's own arithmetic floor -- which the other three arms independently put at 5.4e-16 ..
2.6e-14 on the same solver (the off-diagonal probe below, the separable stripe, and the
Berreman arm).  `|Jxy - Jyx|` is 5.4e-16 .. 1.3e-14 before and 2.9e-15 .. 1.6e-14 after on the
same eight readings: on these cells the artefact lands on the DIAGONAL, which is why the
diagonal probe is the one that carries a fail-before and the off-diagonal one is asserted
but not claimed as an improvement.  Test (b) below is the probe that sees the whole
matrix.

*(b) Mirror covariance on cells with NO symmetry of their own* (director azimuth 20 deg):
mirror the cell and `J` must come back as `P J P`.  Jones scale 0.114 .. 0.184:

| cell | M = 3 / 4 / 5, before | after |
|---|---|---|
| rectangle | 2.945e-04 / 4.836e-04 / 2.030e-04 | **5.98e-15 / 9.18e-15 / 1.62e-14** |
| disk | 5.008e-03 / 3.743e-03 / 3.117e-03 | **5.15e-15 / 1.30e-14 / 1.25e-14** |

This is the property that holds for EVERY cell, so it cannot be satisfied by a fixture that
happens to be symmetric.

*(c) A conical Berreman 4x4 solve* on a UNIFORM rotated-director uniaxial cell
(`elements/berreman.py`, which I did not change; it takes the tensor directly and carries no
Fourier factorization at all, which is the entire object of D3).  `max|dJ|` against a Jones
of scale 0.239 .. 0.268, at `n_orders` 1 and 2:

| polar / azimuth | before | after |
|---|---|---|
| 0 / 0 deg | 2.06e-15 / 4.11e-15 | 2.06e-15 / 4.11e-15 |
| 14 / 29 deg | 1.08e-15 / 5.74e-15 | 1.08e-15 / 5.74e-15 |
| 25 / 45 deg | 2.18e-15 / 7.13e-15 | 2.18e-15 / 7.13e-15 |

Unchanged, as it must be: on a laterally uniform cell the two factorization orders coincide
analytically, so this arm gates the off-plane path's ABSOLUTE correctness and the
symmetrisation's inertness at the same time.

*(d) The separable limit.*  A y-uniform stripe: `max|J_sym - J_L2L1|` = 7.77e-15 / 1.32e-14
/ 2.56e-14 at M = 3 / 5 / 7, against the stripe's GENUINE form birefringence
`|Jxx - Jyy| = 0.327` -- thirteen decades of signal above the change.

*(e) The in-plane path.*  `_li_convolutions_2d_tensor` (the H3 2x2 operator) is untouched:
no line of it is in the diff.  Stronger, and asserted: on a cell with NO off-plane
components the symmetrized 3x3 operator's four in-plane blocks are **bit-identical (0.0)**
to the symmetrized in-plane 2x2 operator at `n_orders` 3 and 5, and its four off-plane
blocks are exactly zero.  The in-plane ENTRY POINT's numbers move only through D2 (<=
1.665e-15, table above).

*Energy.*  The off-plane path closes at 1.4e-14 (M = 2) and 1.8e-14 (M = 3) on a lossless
rotated-director square pillar at 14 / 29 deg conical.

**Residual risk.**  The symmetrisation doubles the (scalar-pivot) factorization pass of the
3x3 operator, which is the cheap half of the build; the `O(N^3)` block inversions inside it
are also doubled, so an out-of-plane `fff_nv` build is measurably more expensive at large
`n_orders` -- the same trade H3 already took in-plane.  It is not on the eigensolve's
critical path and was not separately optimised.  `rcwa_jones_2d(formulation='fff_nv')` on an
out-of-plane cell returns different numbers; the migration note is in the changelog.

### D1 (H4, perf) -- the two Toeplitz inverses: measured, and both sites left alone

**The sites.**  `oned.py:135` (`EPS_II = inv(_toeplitz_1d(inv_c, n_orders))`, the Li
inverse-rule matrix) and `oned.py:708` (`EPS_inv1 = inv(EPS)` in the planar TM fast path).
Measured share of a 1-D Ag TM solve at `n_orders` 200, BEFORE D2: the convolution builder
including its inverse is 18.3 ms of 514.8 (3.5 %) and the fast path's `inv(EPS)` another
~17 ms (3.4 %) -- together ~7 %, consistent with WP-A14's ~8 %.  At `n_orders` 400 the pair
is 1.3 % + 4.6 % of 7533 ms.  D2 shrinks the denominator, so AFTER it the same two inverses
are 4.4 % + ~4.2 % at `n_orders` 200 and 2.5 % + 8.7 % at 400 -- which is why they were
measured against their alternatives on this tree rather than on the pre-D2 one.

**What the design asked for, and what the sites actually look like.**  "Wherever the inverse
is immediately MULTIPLIED, replace `inv(T) @ X` by `scipy.linalg.solve_toeplitz` (or a
`solve`)."  Enumerated:

* **Site A** is multiplied twice in the planar TM fast path (`A @ EPS_II` to build `M1`, and
  `EPS_II @ Wl1` to build `V`), so the candidate composites are `inv` + 2 products, two
  independent solves, or one `lu_factor` + two `lu_solve`.  It is NOT multiplied on the
  general 2N path: `_layer_Q_matrix` uses `EPS_normal` in a SUBTRACTION (`Ky Ky -
  EPS_normal`), which needs the explicit matrix, so that path cannot lose the inverse at
  all.
* **Site B** is not multiplied by a matrix at any site: its only consumer is the elementwise
  `kx[:, None] * EPS_inv1 * kx[None, :]`, which is `O(N^2)`.  The only candidate is `inv`
  against a single `solve` with a diagonal right-hand side.

**Cost** (`scratchpad/b5/d1_sites.py`, the library's own matrices, Ag at 633 nm, duty 0.5,
one BLAS thread, medians of three):

| `n_orders` | `N` | site A: ships | 2 x `solve` | `lu_factor` + 2 `lu_solve` | 2 x `solve_toeplitz` |
|---|---|---|---|---|---|
| 50 | 101 | 0.75 ms | 1.03 | 0.99 | **9.84** |
| 100 | 201 | 5.08 ms | 6.48 | 5.19 | **66.5** |
| 200 | 401 | 37.80 ms | 40.32 | 29.85 | **449.4** |
| 400 | 801 | 192.5 ms | 201.1 | 170.1 | **3918** |

| `n_orders` | `N` | site B: ships (`inv`) | `solve(EPS, diag)` | `solve_toeplitz` |
|---|---|---|---|---|
| 50 | 101 | 0.79 ms | 0.73 | **3.40** |
| 100 | 201 | 3.11 ms | 4.09 | **35.7** |
| 200 | 401 | 17.55 ms | 16.96 | **204.4** |
| 400 | 801 | 114.96 ms | 120.09 | **1946** |

`solve_toeplitz` is `O(N^2)` per right-hand side and both sites need `N` of them, so its
Cython Levinson recursion is `O(N^3)` in scalar code against LAPACK's blocked `O(N^3)`:
12x .. 20x at site A, 4x .. 17x at site B.  The plain `solve` route is a wash (site B, 3 %
either way, inside the noise) or slightly worse (site A, 7 %, because two independent
triangular sweeps cost what one inverse plus two products cost).

**Accuracy** (row-equilibrated backward residual of `T Y = X` with `X = diag(kx)`, the same
instrument `_guarded_inverse` scores an inverse on):

| matrix | `n_orders` | `inv` then multiply | LU `solve` | `solve_toeplitz` | rel \|Levinson - shipped\| |
|---|---|---|---|---|---|
| Ag | 50 | 6.61e-15 | 8.87e-15 | **4.69e-13** | 5.27e-12 |
| Ag | 200 | 6.22e-15 | 6.50e-15 | **4.15e-13** | 4.08e-12 |
| Au | 50 | 2.61e-15 | 2.48e-15 | **1.25e-13** | 6.32e-13 |
| Au | 200 | 6.43e-15 | 5.00e-15 | **8.85e-13** | 1.80e-11 |
| W | 50 | 4.18e-15 | 3.27e-15 | 5.30e-15 | 2.97e-15 |
| W | 200 | 4.08e-15 | 3.84e-15 | 1.17e-14 | 6.33e-15 |

Levinson's answer moves the operator by 6.3e-13 .. 1.8e-11 relative on the two noble metals
-- ABOVE the 1.4e-13 closure this package holds, so it is a tolerance the H2 / M1 census does
not admit.  (The LU `solve` moves it by 2.9e-15 .. 1.1e-14, a decade below.)  On tungsten,
whose `[[1/eps]]` is better conditioned, Levinson is fine -- which is the point: the
tolerance would have to be set by the worst metal on the ladder, and that is 1.8e-11.

**A correction of record.**  WP-A14 deferred D1 partly because `[[1/eps]]` for a metallic
grating "is exactly the matrix the M1 conditioning census found reaching `cond ~1e13`".  It
is not; that reading belongs to the interface mode-match `a + b`, which
`_interface_smatrix`'s own docstring records and the M1 census measured at 3.1e16.  Over the
metallic ladder (Ag / Au / Al / W at 633 nm x duty 0.1 / 0.5 / 0.9 x `n_orders` 50 / 200)
the worst `cond([[1/eps]])` is **2.51e+02** and the worst `cond([[eps]])` is **2.51e+02**,
both at Ag duty 0.1 / 0.9 and `n_orders` 50.  So the stability worry that motivated the
deferral does not apply to this matrix; the refusal rests on cost and on Levinson's own
backward error, and a future build that changes either re-opens it, which is what the two
D1 gates are for.

**The one route that does pay, and why it is not taken.**  `lu_factor` once + two
`lu_solve` at site A is 29.85 ms against 37.80 at `n_orders` 200 (1.27x) and 170.1 against
192.5 at 400 (1.13x) -- **2.1 % / 0.6 % of the whole solve** after D2.  It is not shipped:
it buys ~2 % on ONE of the 1-D paths (the general 2N path cannot use it at all), it would
cost a second default-moving change in the same release on top of D2's, and it would have
to thread an LU factorization through `_binary_grating_convolutions`' `(EPS, EPS_II)` return
contract, which is public enough to be in `oned.__all__` and read by tests.  Recorded in
section 6 as available, with its numbers.

---

## 3. Files touched

Source (all within WP-B5 ownership):

* `lumenairy/elements/rcwa/_core.py` -- new `_redheffer_star_rt` (`:3343`) and its
  `__all__` entry (`:5067`); `_symmetric_solve_rt` closes its last star on the source
  (`:2893`); `_symmetric_cascade_rt` gains the comment recording why it does NOT
  (`:3014`).  `_redheffer_star`, `_interface_smatrix`, `_guarded_inverse`,
  `_propagation_star` and every constant are unchanged.
* `lumenairy/elements/rcwa/oned.py` -- the import swap (`:41`) and the three closed-form
  call sites (`:721`, `:756`, `:1153`); the `rcwa_jones_1d` core now builds both
  polarization sources before the branch and indexes the two returned columns.
* `lumenairy/elements/rcwa/twod.py` -- the import swap (`:38`) and four closed-form call
  sites (`:1253`, `:1377`, `:2035`, `:2418`); D3's `_li_convolutions_2d_tensor_full`
  (`:542`) + `_li_tensor_full_l2l1` (`:585`) + the call site (`:1977`).

Tests:

* NEW `tests/unit/test_audit2609_b5_rcwa_eme_bor.py` (38 tests, including the independent
  dense-system star oracle).

Docs:

* `docs/audits/.../fixes/WP-B5_REPORT.md` (this file), `.../fixes/WP-B5_CHANGELOG.md`.

**History documents: none to re-record.**  `docs/history/` carries
`lumenairy.elements.rcwa.stack.md` for this package and nothing for `_core.py`, `oned.py` or
`twod.py`; `stack.py` is not in the diff (`ls docs/history | grep -i rcwa` returns that one
file).  `python scripts/record_history_fingerprints.py --check` reads
`OK lumenairy.elements.rcwa.stack.md` and `OK` on every `eme` and `bor` document; the ONLY
`DRIFT` line in the whole tree is `lumenairy.elements._lens_traced.md`, which is another
work package's in-flight edit and is also the root of three of the five suite failures.

No edits to `CHANGELOG.md`, `README.md`, `Migration-Guide.md`, `CONVENTIONS.md`,
`pyproject.toml`, `lumenairy/__init__.py`, `rcwa/stack.py`, `elements/eme/*`,
`elements/bor/*`, `berreman.py`, `pmm/*`, or any propagator / lens module.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`-q --no-header -p no:cacheprovider`.  The box was shared with three other work packages
throughout, so the DURATIONS below are upper bounds, not clean timings; the performance
numbers in section 2 were taken separately, interleaved, with nothing else of mine running.

| command (test files) | result | duration |
|---|---|---|
| `tests/unit/test_audit2609_b5_rcwa_eme_bor.py` (new) | **38 passed** | 17 s |
| `test_audit2609_a14_rcwa_eme_bor.py test_audit2609_a14_verify.py` (the WP-A14 gates + VERIFY-A14) | **59 passed** | 122 s |
| `test_v5_20_13_pmm_jones_2d_fff_nv.py test_v5_20_12_rcwa_jones_2d_fff_nv.py test_v5_11_0_rcwa_fff_nv_2d.py test_m1_conditioning_guard.py` | 52 passed, **1 failed**, 1 skipped | 379 s |
| `pytest tests/unit -k "rcwa or eme or bor"` (2043 selected, 13801 deselected) | **2030 passed, 5 failed, 8 skipped, 1 xfailed** | 4660 s (1:17:39) |
| the same selection, run a SECOND time independently | **2030 passed, 5 failed, 8 skipped, 1 xfailed** -- the same five | 4853 s (1:20:53) |
| `python validation/run_all.py test_rcwa` | **PASS** (1 of 1 file) | 1.5 s |
| `ruff check lumenairy/elements/rcwa lumenairy/elements/eme lumenairy/elements/bor tests/unit/test_audit2609_b5_rcwa_eme_bor.py` | **All checks passed** | < 1 s |
| `python scripts/record_history_fingerprints.py --check` | `lumenairy.elements.rcwa.stack.md` **OK**, every `eme` / `bor` document **OK**; the tree's ONLY drift is `lumenairy.elements._lens_traced.md` (not mine) | 3 s |
| `tests/unit/test_audit2609_a17_history_lint.py` | **5 passed** | 9 s |

The selection is the only one with failures, it was run twice on two independent processes
and both agree on the same five, and they split three / two:

**Not this work package (3).**  All three are one other engineer's in-flight edit to
`lumenairy/elements/_lens_traced.py`, which the fingerprint check independently names as the
tree's only drift:
`test_audit2609_a17_history_relocation.py::test_the_module_ast_is_unchanged_since_the_history_move[lumenairy.elements._lens_traced]`,
its `..._token_stream_...` twin, and
`test_niche_c12_physics_fit_selection.py::test_a_disagreement_is_never_silent_and_names_both_scores`,
which fails with `TypeError: _picks_concentric.<locals>.s_spy() got an unexpected keyword
argument 'basis'` raised at `_lens_traced.py:11660` -- a monkeypatched spy whose signature
has not caught up with its call site.  None of the three touches `rcwa` / `eme` / `bor`.

**This work package (2), and they are ONE defect class, not two.**
`test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy`
(arm d) and
`test_audit_s1_2_rcwa_lossless_tripwire.py::test_the_pre_round_one_branch_reopens_it_and_the_message_names_the_cause`
are the same construction in two files: both install the ENGINEERED pre-round-1
`_sqrt_decay` body, both then drive `rcwa_jones_1d_segments` over `n_orders` 11..41, and both
read the closure with no `except`.  Both now meet an `_EnergyError` (`sum R + T = 2.312e+00`,
the same rung) where they used to read a warning.  Their SHIPPED arms are unchanged.  The
full two-sided measurement and the exact patch for each are in section 5(a).

The skip count is the pre-existing set (the `test_m1_conditioning_guard.py:1354` "premise
absent on this arm" skip and the LAPACK-build skips in `test_v5_6_rcwa_convergence.py` and
friends); the `xfail` is the documented `test_verify_bor_guards_round3.py` D-V1 one.  None
of them is this work package's.

Measurement scripts (not tests, run once each): `scratchpad/b5/prof_before.py`,
`prof_sites.py`, `micro.py`, `d1_toeplitz.py`, `d1_sites.py`, `d2_census.py`, `d2_thin.py`,
`d3_symm.py`, `ladder.py`, `thin_ladder.py`, `armd.py`, `armd2.py`, `cmp.py`, and the read-only
pre-change archive under `scratchpad/b5/base/`.

---

## 5. Requested changes outside my ownership

**(a) The two ENGINEERED pre-round-1 arms.**  These are the two reds in the verification set
that belong to this work package, and they are one defect class in two files:

* `tests/unit/test_v5_20_12_rcwa_jones_2d_fff_nv.py::test_stripe_fixture_is_free_of_the_mode_match_degeneracy`,
  arm (d);
* `tests/unit/test_audit_s1_2_rcwa_lossless_tripwire.py::test_the_pre_round_one_branch_reopens_it_and_the_message_names_the_cause`.

Both install the pre-round-1 `_sqrt_decay` body (the exact `Re(r) == 0` pin, which an `eig`
output never satisfies), both drive `rcwa_jones_1d_segments` over `n_orders` 11..41, and
both compute the closure with no `except`.  The measurements are below in full; the patches
are one `try` each, and the orchestrator applies them.

**A1 -- `test_v5_20_12_rcwa_jones_2d_fff_nv.py`, arm (d).**

The test's arms (a)-(c) -- the SHIPPED behaviour -- are unchanged:

| arm | before | after |
|---|---|---|
| POST clean (`eps_groove` 2.10), worst `\|sum R + T - 2\|` over `n_orders` 11..41 | 1.1591e-13, 0 warnings | **1.1546e-13, 0 warnings** |
| POST index-coincident (2.25) | 2.2893e-13, 0 warnings | **2.2982e-13, 0 warnings** |

Arm (d) installs a deliberately WRONG `_sqrt_decay` body (the pre-round-1 exact `Re(r) == 0`
pin) so that a propagating layer mode keeps the incoming root -- i.e. an exponentially
GROWING propagator, which the shipped branch cut makes unreachable.  In that regime
`A22 = X S22 X` has entries far above 1, `I - B11 A22` is near-singular, and the answer is a
difference of huge terms; the assembled star and the closed form are then two different
associations of a quantity no build agrees on, and they disagree by O(1):

| arm (d) | before | after |
|---|---|---|
| PRE clean (2.10) | 1.3611e-13 | 3.1830e-12 |
| PRE index-coincident (2.25) | 5.5775e-03, 15 warnings of 16 rungs | **finite worst 4.28e-01 over the 14 rungs that return, `_EnergyError` on the other 2** |

The `_EnergyError` is the gross tripwire `tot > 1.05 * n_states` in `_check_energy` firing
on `sum R + T = 2.312` -- the library REFUSING a garbage number rather than warning about
it, which on a deliberately-broken solver is the guard working.  `worst()` in the test has
no `except`, so the raise propagates and the test errors before it can evaluate either of
its two assertions -- and both would still hold: `pre_warned` = 14 against a `>= 1` bar,
and `pre_degen / pre_clean` = 4.28e-01 / 3.183e-12 = **1.3e+11** against a 1e+05 bar on the
FINITE rungs alone (with the two raising rungs folded in as `inf` it is unbounded).

I checked that this is not a numerical weakness of the closed form anywhere reachable: the
association is not the lever (the alternative `B21 @ u + B21 @ (A22 @ z)` gives the same two
raises), and in every well-posed regime the two formulations are indistinguishable -- the
independent-oracle table in section 2, the 382-array ladder at <= 1.7e-15, the deep metallic
ladder at 3.3e-16, and the library's own instability class at 1.1e-15 over 46 rungs.

Requested edit, in `worst()` (that file is not mine to change):

```python
     def worst(eps_groove):
         """(worst |sum R + sum T - 2| over the ladder, closure warnings)."""
         eg = np.diag([eps_groove] * 3).astype(complex)
         out, warned = 0.0, 0
         for n in ladder:
             with warnings.catch_warnings(record=True) as rec:
                 warnings.simplefilter("always")
-                _o, R1, T1, _J = rcwa_jones_1d_segments(
-                    PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL,
-                    theta=0.0, n_orders=n)
+                try:
+                    _o, R1, T1, _J = rcwa_jones_1d_segments(
+                        PX, [(0.5, er), (0.5, eg)], 1.5, 1.0, DEPTH, WL,
+                        theta=0.0, n_orders=n)
+                    d = abs(float(np.sum(R1) + np.sum(T1) - 2.0))
+                except _EnergyError:
+                    # The GROSS tripwire (tot > 1.05 * n_states) is the defect
+                    # manifesting, not an absent measurement: the solver is
+                    # refusing a number no build agrees on.  Reached only on the
+                    # ENGINEERED pre-round-1 arm, where a layer mode carries a
+                    # GROWING propagator (WP-B5 D2: the two-interface closed
+                    # form and the assembled star are two associations of a
+                    # difference of huge terms, and disagree by O(1) there --
+                    # 5.6e-03 vs 4.3e-01 on this fixture).  The POST arms assert
+                    # `< _ONED_SOUND_CLOSURE`, so an inf there still fails.
+                    d = float("inf")
             warned += sum(1 for w in rec if "lossless energy closure violated"
                           in str(w.message))
-            out = max(out, abs(float(np.sum(R1) + np.sum(T1) - 2.0)))
+            out = max(out, d)
         return out, warned
```

with `from lumenairy.elements.rcwa._core import _EnergyError` added to the existing
`from lumenairy.elements.rcwa...` imports.  This is strictly TIGHTER than what it replaces:
the POST arms still assert `< _ONED_SOUND_CLOSURE`, so an `inf` there fails; arm (d)'s two
claims are unchanged; and the arm now records the stronger of the two signals the reopened
defect can produce instead of crashing on it.

**A2 -- `test_audit_s1_2_rcwa_lossless_tripwire.py::test_the_pre_round_one_branch_reopens_it_and_the_message_names_the_cause`.**

Same construction, different fixture (`period` 0.7 um, a 35-degree rotated director whose
ordinary `no^2 = 2.25` coincides with the groove AND with `n_substrate^2`, depth 0.5 um,
`lambda` 1 um), and the same helper shape: `_s1_2_closure` calls `rcwa_jones_1d_segments`
with no `except`.  Measured base vs live (`scratchpad/b5/armd2.py`, worst over
`n_orders` 11..41, both trees, each in a child process with `lumenairy.__file__` asserted):

| arm | before (`284daccc`) | after |
|---|---|---|
| SHIPPED coincident | 2.2893e-13, 0 warnings, 0 raises | **2.2982e-13, 0 warnings, 0 raises** |
| SHIPPED detuned control (+1e-3) | 3.8680e-13 | **3.8636e-13** |
| PRE-round-1 coincident | 5.5775e-03, 15 warnings, 0 raises | **4.2752e-01 finite worst, 14 warnings, 2 raises of 16 rungs** |
| PRE-round-1 detuned control | 3.9968e-13 | **1.5091e-11** |

Its two numeric assertions still hold after the patch, with their gaps measured:
`worst_bad > 1e3 * worst_good` reads **2.833e+10** (before: 1.395e+10) against a 1e+03 bar,
and `worst_good < _S1_2_CLOSURE_BAR` reads **1.5091e-11** against the file's own 1e-09 bar,
1.8 decades inside it.  Its four message-text assertions read the same 14 warnings.

Requested edit, in `_s1_2_closure` (that file is not mine to change):

```python
 def _s1_2_closure(er, eg, n_orders):
     """``(|sum R + sum T - 2|, an _EnergyWarning fired)`` at one truncation."""
     with warnings.catch_warnings(record=True) as rec:
         warnings.simplefilter("always")
-        out = rcwa_jones_1d_segments(0.7e-6, [(0.5, er), (0.5, eg)], 1.5, 1.0,
-                                     0.5e-6, 1.0e-6, angle=0.0,
-                                     n_orders=n_orders)
+        try:
+            out = rcwa_jones_1d_segments(0.7e-6, [(0.5, er), (0.5, eg)], 1.5,
+                                         1.0, 0.5e-6, 1.0e-6, angle=0.0,
+                                         n_orders=n_orders)
+            miss = abs(float(np.sum(out[1]) + np.sum(out[2])) - 2.0)
+        except _EnergyError:
+            # The GROSS tripwire (tot > 1.05 * n_states) is the defect
+            # manifesting, not an absent measurement: the solver is refusing a
+            # number no build agrees on.  Reached only on the ENGINEERED
+            # pre-round-1 arm, where a layer mode carries a GROWING propagator
+            # (WP-B5 D2: the two-interface closed form and the assembled star
+            # are two associations of a difference of huge terms and disagree
+            # by O(1) there -- 5.6e-03 against 4.3e-01 on this fixture).  The
+            # SHIPPED arms assert `< _S1_2_CLOSURE_BAR`, so an inf there still
+            # fails.
+            miss = float("inf")
     fired = [w for w in rec if isinstance(w.message, _EnergyWarning)]
-    return (abs(float(np.sum(out[1]) + np.sum(out[2])) - 2.0), fired)
+    return (miss, fired)
```

with `_EnergyError` added to the existing
`from lumenairy.elements.rcwa._core import _EnergyWarning` line.

**Why the fixture itself does NOT have to change, with the numbers.**  The question the
brief poses is whether the shipped code or the engineered arm moved.  Three independent
measurements say the arm:

1. the SHIPPED arms of both tests are unchanged at the 1e-15 level (1.1591e-13 -> 1.1546e-13
   and 2.2893e-13 -> 2.2982e-13, 0 warnings and 0 raises on both trees);
2. the regime the arm creates is unreachable from the public API -- it requires a layer mode
   with `Re(lam) < 0`, i.e. `|exp(-lam k0 L)| > 1`, which `_sqrt_decay`'s branch forbids and
   which the round-3 branch-cut fix exists to forbid.  Inside it `I - B11 A22` is
   near-singular and the answer is a difference of huge terms, so ANY re-association moves
   it by O(1): the alternative association `B21 @ u + B21 @ (A22 @ z)`, which defers the
   cancellation past the `B21` projection, was built and measured on the A1 fixture and
   raises on the SAME two rungs (finite worst 4.58e-01 against variant A's 4.28e-01) -- so
   the choice of association is not the lever, and the shipped one is the cheaper;
3. in every reachable regime the two formulations are indistinguishable -- the
   independent-oracle table in section 2 (ratio 0.24 .. 1.29), the 382-array ladder at
   <= 1.665e-15, the deep metallic ladder at 3.331e-16, and the library's own instability
   class at 1.110e-15 over 46 rungs.

So the fixtures stay; what changes is that each helper records the gross-tripwire REFUSAL as
the defect manifesting rather than crashing on it -- which is the stronger of the two signals
those arms exist to observe.

**(b) `CONVENTIONS.md` section 11 (`formulation='fff_nv'` is entry-point-specific).**  One
sentence: `rcwa_jones_2d(formulation='fff_nv')` on an OUT-OF-PLANE (full 3x3) tensor cell is
symmetrized over the two Li-2003 factorization orders, as the in-plane 2x2 operator already
is, so the x <-> y mirror is exact there (a single factorization order breaks it by
2e-04 .. 5e-03).  Not my file.

**(c) Nothing else.**  No change is requested in `pmm/`, `berreman.py`, `pyproject.toml` or
any propagator / lens module.  WP-A14's outstanding request (a) -- `threadpoolctl` as a hard
dependency -- is still open and still worth doing; it is unrelated to this pass, and it is
still not installed on this box (the H5 warning fires in the suite by design).

---

## 6. Deferred, with measurements

**B5-D1a -- the shared-LU route at site A.**  `lu_factor` once + two `lu_solve` instead of
`inv` + two products in the planar TM fast path: 29.85 ms against 37.80 at `n_orders` 200,
170.1 against 192.5 at 400, i.e. **2.1 % / 0.6 % of the whole solve**.  Not taken this pass
(section 2, D1).  If it is ever taken it needs the `(EPS, EPS_II)` return contract of
`_binary_grating_convolutions` widened, and its own before/after ladder, since it moves the
default by ~1e-14 relative.

**B5-D2a -- `_interface_smatrix` is now the dominant `O(N^3)` term.**  After D2 the two
interface builds are 41.7 % of a 1-D Ag TM solve at `n_orders` 200 (157 ms of 377) and
23.0 % at 400.  Each pays two `solve`s, one `_guarded_inverse` and three products.  The closed form never
reads `B12` or `B22` from the SECOND interface, and `S21` there costs a product only because
`S22` is built first -- so a `want=` selector on `_interface_smatrix` would drop one product
of three on one of the two interfaces, ~5 % of the solve.  Not attempted this pass: it is a
signature change on a helper with 26 call sites inside `rcwa/` (including six in `stack.py`,
whose history document would then have to be re-recorded for a pure-performance edit), and
it wants to be done together with B5-D2b rather than twice.  `elements/pmm/_core.py:1954`
has its own twin and `berreman.py` uses only the GENERALIZED interface, so nothing outside
`rcwa/` is affected either way.

**B5-D2b -- the subnormal tail of `_propagation_star`.**  `A22 = X S22 X` underflows into
subnormals at `n_orders >= ~300`, and a single `2N` product against it costs ~33x a normal
one on this box (1.7 s vs 52 ms at `N = 801`).  D2 removes eleven of the twelve such
products; the twelfth (`B11 @ A22`, needed to form the guarded inverse's argument) still
pays it, and it is 45 % of the solve at `n_orders` 400.  Flushing those entries to zero
would change the default; measuring what it changes, and whether the propagating block is
provably unaffected, is a pass of its own.  Effort: 0.5 day.

**D4 / D5 (WP-A14) -- the `_branchcut` and forward-selector consolidations.**  Untouched, as
the brief directs: two of the four sites live in `pmm/`.

**Not reproducible: none.**  Both D2's and D3's defects reproduced on the current HEAD, and
D1's two sites are exactly where WP-A14 left them.

---

## Addendum after VERIFY-B5 (orchestrator, 2026-09-13)

* **Section 2, "Residual risk, stated exactly" is superseded by this paragraph.**  The closed form is a RE-ASSOCIATION, and it is
  neutral exactly while `I - B11 A22` is well conditioned.  Where that denominator is near-singular the answer is a difference of terms
  far larger than the difference, and both formulations carry their own `cond * eps`.  Two things reach that regime.  An exponentially
  GROWING layer propagator does (`A22` far above 1) and `_sqrt_decay`'s `Re(lam) >= 0` branch keeps it off the public API (`|X| <= 1`
  always); it IS reachable by monkeypatching that branch, and the two engineered test arms of section 5 do so.  A HIGH-Q CAVITY RESONANCE
  does it with `|X| <= 1` throughout and IS reachable: on a weakly modulated high-index slab whose `+-1` order is evanescent in both
  half-spaces, `cond(I - B11 A22)` reaches 1.75e+13 from `rcwa_efficiency_1d` alone, the closed form and the assembled star land 4.2e-04
  apart and EQUALLY far (9.7e-04 each) from the coupled system solved whole, and the shipped per-order efficiency moves up to 6.4e-06
  between them (measured by VERIFY-B5 with a double-double reference).  The `<= 1.665e-15 / 3.114e-15` envelope of section 2 is a statement
  about the well-conditioned population it was measured on, not a bound on the entry points.  Both regimes are documented in
  `_redheffer_star_rt`'s docstring and gated by
  `test_audit2609_b5_rcwa_eme_bor.py::test_d2_a_near_singular_star_denominator_is_reachable_and_neither_form_is_better`.
* **Section 5(a), point 2** reads with this scope: the regime the engineered arm creates -- a layer mode with `Re(lam) < 0` -- is unreachable
  from the public API; the near-singular denominator it produces is reachable by other means (a cavity resonance) and is documented in
  `_redheffer_star_rt`, but not with `|A22|` far above 1, which is what makes THAT arm's disagreement O(1).  The conclusion (the fixtures
  stay; the arms record the refusal) is unaffected.
* **Section 2, the 382-array movement table**: the row `rcwa_jones_2d (+ out-of-plane) | 84 | 38 | 1.665e-15 | 3.114e-15` covers the
  `'laurent'` / `'li'` formulations on the out-of-plane cell; `'fff_nv'` on an out-of-plane cell is D3's intended change (1.08e-03 measured)
  and is covered by the migration note, not by that row.
* **Section 2, D3 "Residual risk"**: the symmetrisation doubles the operator build, which is 7.7 % of the whole solve at `n_orders` 4 and
  falls to 2.9 % by 8 -- the eigensolve outgrows it; "measurably more expensive at large `n_orders`" has the direction backwards.
* **Section 5(c), correction of record**: `threadpoolctl` IS a declared dependency (`pyproject.toml`, `requirements.txt`, `>= 3.1`) and
  3.6.0 is installed on this box; the skip list reads "threadpoolctl installed: the cap is effective here".  WP-A14's request is closed.
