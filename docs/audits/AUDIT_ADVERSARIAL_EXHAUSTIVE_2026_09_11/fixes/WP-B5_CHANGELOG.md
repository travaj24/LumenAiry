# WP-B5 changelog text -- RCWA (audit 2026-09-11, the H4 / H3 items WP-A14 deferred)

Release 5.47.0.  Three items from `fixes/WP-A14_REPORT.md` section 6: D1 (the two Toeplitz
inverses -- MEASURED and refused, with the numbers), D2 (the two-interface closed form) and
D3 (the off-plane `fff_nv` symmetrisation).

### Performance -- RCWA: the last star of a single-layer cascade is applied to the source instead of assembled (H4, deferred item D2)

Every single-layer entry point ends its cascade with
`_redheffer_star(interface -> propagation -> interface, layer|substrate)` and then reads
exactly two things out of the result: `S11 @ cinc` and `S21 @ cinc`.  The star's zero-block
fast path cannot fire there (both `A22` and `B11` are non-zero), so it was assembling four
`2N x 2N` blocks -- twelve matrix products and TWO `_guarded_inverse` calls -- to use two of
them on one or two columns.

New `_redheffer_star_rt` (`lumenairy/elements/rcwa/_core.py:3343`) computes those two
products directly.  With `D = (I - B11 A22)^-1`, `u = A21 c` and `z = D B11 u`, the
push-through identity `(I - A22 B11)^-1 = I + A22 D B11` removes the second inverse
outright and the star reduces to

    S11 c = A11 c + A12 z        S21 c = B21 (u + A22 z)

-- seven mat-vecs, ONE `2N` product and ONE inverse.  Wired at
`oned.py:721` (planar TE/TM fast path), `oned.py:756` (the 2N path of `rcwa_efficiency_1d`),
`oned.py:1153` (`rcwa_jones_1d` / `rcwa_jones_1d_segments`, both polarizations in one
block), `twod.py:1253` (`rcwa_efficiency_2d`), `twod.py:1377` (`PreparedRCWA2D.solve`),
`twod.py:2035` (`rcwa_jones_2d`, in-plane and full-3x3), `twod.py:2418`
(`rcwa_efficiency_2d_shapes`) and `_core.py:2893` (`_symmetric_solve_rt`, the single-layer
even-parity fold).

Measured on the 1-D metallic ladder (Ag `n = 0.135 + 3.99j` at 633 nm, period 1 um, depth
0.25 um, duty 0.5, `n_sub = 1.5`, `formulation='li'`, TM), medians of three interleaved
runs at one BLAS thread:

| `n_orders` | whole solve | star's share |
|---|---|---|
| 50 | 16.8 -> **14.0 ms** (1.20x) | 20.2 % -> 5.0 % |
| 100 | 80.3 -> **65.4 ms** (1.23x) | 18.7 % -> 6.8 % |
| 200 | 514.8 -> **377.1 ms** (1.37x) | 21.6 % -> 6.7 % |
| 400 | 7533 -> **3973 ms** (1.90x) | 70.8 % -> 45.5 % |

The `n_orders = 400` row is superlinear in both columns for a reason worth recording: at
`N = 801` the propagation-scaled `A22 = X S22 X` underflows into SUBNORMALS in the
evanescent tails, and one 801-wide product against it costs ~1.7 s where a normal one costs
52 ms.  Eleven of the twelve products no longer touch it.  `np.linalg.inv` calls per 1-D
solve: 6 -> 5.

THE ANSWER MOVES, in the last bits only, and this is a deliberate default change.  The same
terms are summed in a different order (mat-vec instead of mat-mat).  Measured against the
pre-change tree over 382 arrays -- 192 1-D efficiency configurations (Ag / Au / dielectric x
both polarizations x `'li'` / `'laurent'` x `n_orders` 11..200 x normal and 12 deg), the
1-D Jones family including the full-3x3 branch, 2-D efficiency / Jones / analytic-shape /
prepared-sweep, the deep metallic ladder at `n_orders` 100..400 and the library's own
instability class (`period` 10 um, `dn` 0.05, 46 rungs):

* worst absolute movement **1.665e-15**, worst relative **3.114e-15**;
* deep metallic ladder 3.331e-16 / 1.650e-15; instability class 1.110e-15 / 1.124e-15;
* energy closure unchanged to twelve printed digits on every rung.

`RCWAStack`, `berreman_jones_1d`, the EME layer solvers and the BOR radial solvers are
BYTE-IDENTICAL (proved against a `git archive` of the pre-change tree, imported from a
child process whose `PYTHONPATH` is the archive).

THE CONDITIONING GUARD IS PRESERVED ON THE ONE REMAINING INVERSE.  `I - B11 A22` still goes
through `_guarded_inverse` under the same site string, so the M1 census records it
identically; a 1-D single-layer solve now censuses three inverses (two interfaces + one
star) instead of four.  Dropping `I - A22 B11` hides nothing: the two are similar
(`(I - A22 B11) A22 = A22 (I - B11 A22)`), and the retained one is the TIGHTER equilibrated
`rcond` of the pair wherever they differ -- 0.340 / 0.167 / 0.108 / 0.0695 against
0.523 / 0.670 / 0.671 / 0.671 at `n_orders` 11 / 50 / 100 / 200 of the metallic ladder, and
equal to within 6 % on the thin-grating family the M1 census was taken on.  Neither star
inverse was ever the refusing one (`rcond_refuse` is armed only on
`_interface_smatrix_general`'s `T22`), so no refusal path changes, and the two zero-block
shortcuts are taken on the same concrete tests as before -- a chain that paid no star
inverse still pays none.

DELIBERATELY NOT APPLIED to `_core.py:2932` `_symmetric_cascade_rt`: `elements/pmm/stack2d.py`
and `pmm/twod_jones.py` fold their own cascades through it, and closing its last star on the
sources would move the PMM engines' last bits from inside the RCWA package.  Pinned by a
test.

Files: `lumenairy/elements/rcwa/_core.py:3343` (new `_redheffer_star_rt`), `:2893`,
`:5067` (`__all__`); `oned.py:41,:721,:756,:1153`; `twod.py:38,:1253,:1377,:2035,:2418`.
Tests: `tests/unit/test_audit2609_b5_rcwa_eme_bor.py::test_d2_closed_form_matches_the_independent_star_oracle`,
`::test_d2_closed_form_matches_the_oracle_on_the_metallic_chain`,
`::test_d2_a_single_layer_solve_records_one_star_inverse_not_two`,
`::test_d2_the_retained_denominator_is_the_tighter_reading`,
`::test_d2_zero_block_shortcuts_take_no_inverse_at_all`,
`::test_d2_energy_closes_through_the_closed_form`,
`::test_d2_the_multi_layer_even_fold_still_assembles_its_star`,
`::test_d2_two_d_entry_points_still_close`.

### Fixed -- RCWA: the OFF-PLANE (full 3x3) `fff_nv` operator broke the cell's x<->y mirror (H3, deferred item D3)

The in-plane 2x2 Li-2003 operator was symmetrized over its two factorization orders in
5.46.0 (H3).  The full-3x3 operator that `rcwa_jones_2d(formulation='fff_nv')` uses for an
OUT-OF-PLANE tensor cell was left on the fixed `L2 L1` order, so the same defect survived
there: a cell whose geometry AND director are invariant under the x <-> y mirror -- where
the mirror forces `Jxx == Jyy` and `Jxy == Jyx` at normal incidence -- came back with a
difference, and mirroring any cell did not mirror its Jones matrix.

`_li_convolutions_2d_tensor_full` (`lumenairy/elements/rcwa/twod.py:542`) now returns the
mean of the two orders, with the single order split out as `_li_tensor_full_l2l1`
(`twod.py:585`) and still reachable through `symmetrize=False`.  The 3x3 transpose is the
in-plane argument with the component permutation `(x, y, z) -> (y, x, z)`: `exx<->eyy`,
`exy<->eyx`, `exz<->eyz`, `ezx<->ezy`, `ezz` alone, the pixel grid transposed and the
order-label columns swapped, so that `T P L2L1(eps) P T = L1L2(P eps^T P)` and the nine
blocks come back in the same retained-order basis.  The mean is taken on the RAW `ehat`
blocks, which puts the caller's `l3-` `E_z` fold after it (`twod.py:1977`) -- the mean of
two Schur complements is not the Schur complement of the mean.

Measured on a uniaxial pillar (`n_o = 2.0`, `n_e = 2.6`, director polar 40 deg) in air,
period 0.5 um, depth 0.3 um, 633 nm, 96x96 cell, normal incidence, director azimuth 45 deg
so the cell is its own mirror:

| cell | `\|Jxx - Jyy\|`, M = 3 / 4 / 5 / 6 |
|---|---|
| square, before | 9.10e-04 / 8.33e-04 / 3.68e-04 / 3.32e-04 |
| square, after | **2.4e-15 / 1.1e-14 / 1.1e-14 / 8.5e-15** |
| disk, before | 4.71e-03 / 3.52e-03 / 2.80e-03 / 2.35e-03 |
| disk, after | **3.6e-15 / 7.6e-15 / 1.5e-14 / 2.3e-15** |

on a Jones matrix whose own scale is 0.11 (disk) to 0.23 (square), i.e. 0.4 % to 4 % of
spurious form birefringence on a cell that has none.  The general property -- mirror the
cell and the Jones matrix must come back as `P J P` -- goes from 2.03e-04 .. 5.01e-03 to
5.1e-15 .. 1.6e-14 on cells with no symmetry of their own.  At the operator level the two
factorization orders differed by a relative 1.6e-04 .. 2.9e-04 on these anisotropic cells
(and 1.9e-02 .. 2.2e-02 on an isotropic disk, which is the in-plane H3 number reproduced
through the 3x3 path).

Three things are unchanged and are asserted, not assumed: a UNIFORM rotated-director cell
still matches a conical Berreman 4x4 solve to 1.1e-15 .. 7.1e-15 at 0 / 14 / 25 deg polar
(both before and after -- the two orders coincide on a laterally uniform cell, which is
what makes Berreman a clean oracle for the rest of the path); a y-uniform SEPARABLE stripe,
where the two orders coincide analytically, moves by 7.8e-15 .. 2.6e-14 against a genuine
form birefringence of 0.327; and on a cell with NO off-plane components the symmetrized 3x3
operator's four in-plane blocks are BIT-IDENTICAL to the symmetrized in-plane 2x2 operator
(and its off-plane blocks exactly zero), which is the sense in which this is the
generalization of the shipped in-plane fix rather than a second, different one.

The in-plane `_li_convolutions_2d_tensor` is untouched.  The extra cost is one more
scalar-pivot factorization pass, the cheap half of the build.

Files: `lumenairy/elements/rcwa/twod.py:542` (`_li_convolutions_2d_tensor_full`, now
symmetrized, `+ n_orders_y` and `symmetrize=`), `:585` (`_li_tensor_full_l2l1`), `:1977`
(the call site and its comment).
Tests: `tests/unit/test_audit2609_b5_rcwa_eme_bor.py::test_d3_offplane_fff_nv_keeps_the_cells_own_mirror`,
`::test_d3_offplane_fff_nv_is_mirror_covariant`,
`::test_d3_uniform_rotated_director_matches_the_berreman_oracle`,
`::test_d3_separable_stripe_is_unchanged_by_the_symmetrisation`,
`::test_d3_reduces_exactly_to_the_in_plane_operator_it_generalizes`,
`::test_d3_energy_closes_on_the_off_plane_path`.

### Unchanged (measured) -- RCWA: the two Toeplitz inverses keep their explicit inverse (H4, deferred item D1)

WP-A14 deferred a Levinson / Gohberg-Semencul route for the two genuinely Toeplitz inverses
of the 1-D solve -- `inv([[1/eps]])` (`lumenairy/elements/rcwa/oned.py:135`) and
`inv([[eps]])` in the planar TM fast path (`oned.py:708`) -- and named a cheaper 80 %:
`scipy.linalg.solve_toeplitz` wherever the inverse is immediately multiplied.  Both sites
were measured against that route and both KEEP the explicit inverse.  Nothing in the
library changed; what follows is the evidence, so the decision is re-openable rather than
folklore.

**Cost.**  `solve_toeplitz` is `O(N^2)` per right-hand side, and both sites need `N` of
them, so its Cython Levinson recursion runs `O(N^3)` in scalar code against LAPACK's
blocked `O(N^3)`.  Measured on the library's own matrices (Ag, duty 0.5, one BLAS thread),
the composite each site actually needs:

| `n_orders` | site A: `inv` + 2 products | 2 x `solve` | `lu_factor` + 2 `lu_solve` | 2 x `solve_toeplitz` |
|---|---|---|---|---|
| 50 | 0.75 ms | 1.03 | 0.99 | **9.84** |
| 100 | 5.08 ms | 6.48 | 5.19 | **66.5** |
| 200 | 37.8 ms | 40.3 | 29.9 | **449** |
| 400 | 192 ms | 201 | 170 | **3918** |

Site B consumes its inverse ONLY in the elementwise `kx[:, None] * EPS_inv * kx[None, :]`
-- there is no matrix product to fold a solve into -- so the candidates are `inv` against a
single `solve`: 0.79 / 3.11 / 17.6 / 115 ms against 0.73 / 4.09 / 17.0 / 120 ms, a wash,
and `solve_toeplitz` at 3.40 / 35.7 / 204 / 1946 ms.

**Accuracy.**  On the row-equilibrated backward residual of the system each candidate
claims to solve, Levinson lands two decades further out than the LU inverse it would
replace -- 4.69e-13 / 4.15e-13 (Ag, `n_orders` 50 / 200) and 1.25e-13 / 8.85e-13 (Au)
against 6.61e-15 / 6.22e-15 and 2.61e-15 / 6.43e-15 -- and its ANSWER differs from the
shipped one by 6.3e-13 .. 1.8e-11 relative, i.e. ABOVE the 1.4e-13 closure this package
holds.  An LU `solve` differs by 2.9e-15 .. 1.1e-14, a decade below it.  The documented
tolerance the design asked for is therefore not one the H2 / M1 census admits.

**A correction of record.**  The deferral said `[[1/eps]]` for a metallic grating "is
exactly the matrix the M1 conditioning census found reaching `cond ~1e13`".  It is not:
that reading belongs to the interface mode-match `a + b`, which `_interface_smatrix`'s own
docstring records.  Measured over the metallic ladder (Ag / Au / Al / W at 633 nm x duty
0.1 / 0.5 / 0.9 x `n_orders` 50 / 200) the worst `cond([[1/eps]])` is **2.51e+02** and the
worst `cond([[eps]])` **2.51e+02**.  D1's refusal rests on cost and on Levinson's own
backward error, not on this matrix being near-singular.

Files: none.
Tests: `tests/unit/test_audit2609_b5_rcwa_eme_bor.py::test_d1_levinson_lands_outside_what_the_package_closure_admits`,
`::test_d1_the_inverse_rule_toeplitz_is_not_the_ill_conditioned_matrix`,
`::test_d1_the_two_sites_still_form_the_explicit_inverse`,
`::test_d1_guarded_inverse_is_untouched_by_this_work_package`.

### Migration notes

* `rcwa_efficiency_1d`, `rcwa_jones_1d`, `rcwa_jones_1d_segments`, `rcwa_efficiency_2d`,
  `rcwa_jones_2d`, `rcwa_efficiency_2d_shapes` and `PreparedRCWA2D.solve` return values that
  differ from 5.46.0 in the last bits (worst measured 1.7e-15 absolute / 3.1e-15 relative).
  A test that pins one of these to more than ~13 significant figures will need its value
  re-recorded.  The envelope is the well-conditioned population's: at a high-Q cavity
  resonance, where `I - B11 A22` reaches `cond` 1e13, the two formulations differ by up to
  6.4e-06 and neither is the better one -- see `_redheffer_star_rt`.  `RCWAStack`, `berreman_jones_1d`, `elements/eme`,
  `elements/bor` and `elements/pmm` are byte-identical.
* `rcwa_jones_2d(formulation='fff_nv')` on an OUT-OF-PLANE (full 3x3) tensor cell returns a
  different, x<->y-symmetric answer; the change is 2e-04 .. 5e-03 on the Jones matrix of a
  patterned cell and zero on a uniform or separable one.
* `_li_convolutions_2d_tensor_full` (private) takes `n_orders_y` as a new fourth positional
  argument and a keyword-only `symmetrize=True`.
* Two test helpers that ENGINEER the pre-round-1 `_sqrt_decay` body need a
  `try / except _EnergyError` around their `rcwa_jones_1d_segments` call
  (`test_v5_20_12_rcwa_jones_2d_fff_nv.py`'s `worst()` and
  `test_audit_s1_2_rcwa_lossless_tripwire.py`'s `_s1_2_closure`).  In that regime a layer
  mode carries an exponentially GROWING propagator, the cascade is a difference of huge
  terms, and the closed form's garbage crosses the gross `R + T > 1.05 n_states` tripwire
  where the assembled star's stayed under it (5.6e-03 -> 4.3e-01 on both fixtures).  Both
  tests' own numeric claims still hold with the raise folded in as `inf`; the shipped arms
  of both are unchanged (1.1591e-13 -> 1.1546e-13 and 2.2893e-13 -> 2.2982e-13, zero
  warnings on either tree).  Patches in `fixes/WP-B5_REPORT.md` section 5(a).
