# WP-A14 changelog text -- RCWA / EME / BOR (audit 2026-09-11, findings H1-H6, G11)

### Fixed -- BOR: `guided_modes` returned an EMPTY list for every weakly-guiding fiber (H1)

The guided-window guard band was `5e-3 * k0` per side -- a fraction of `k0`, not of the
`(n_core - n_clad) * k0`-wide window it guards -- so nothing at all was admitted unless
`n_core - n_clad > 0.01`.  Standard telecom SMF (`dn ~ 0.005`) and the textbook `V = 2.4`
fiber (1.45 / 1.44, `dn = 0.010` exactly, where the admissible interval collapses to a
single point) both returned `[]` at every `(Rbig, N)` tried, with no warning and no way to
tell the result from a genuinely cut-off structure -- while `radial_coupled_modes` held
the correct HE11 all along.  The band is now `max(1e-6 * k0, 1e-3 * (qhi - qlo))`, which is
invariant under BOTH a unit rescale (the W6-B2 fix) and an index-contrast rescale; the
real-axis tolerance is clamped to half the window, which only binds where it previously
exceeded the whole guided band.  A window narrower than twice the band now RAISES
`ValueError` naming both permittivities, and an empty result that had candidates inside the
band WARNS naming the closest one.

Measured (`lambda = 1.55 um`, `V = 2.4`, `Rbig = 6a`, `N = 150`) against the exact hybrid
HE11 oracle: `dn = 0.010` 0 modes -> 1 mode at `n_eff = 1.445294274` (exact 1.445293173,
err +1.10e-06); `dn = 0.005` 0 modes -> 1 mode at `n_eff = 1.447648919` (exact 1.447648366,
err +5.53e-07).  The four existing `guided_modes` fixtures all use `dn = 1.04`, 100x the old
band, which is why CI was green.

Files: `lumenairy/elements/bor/coupled_radial_eigensolver.py:566` (re-anchored 2026-09-12 after the history relocation moved lines).
Tests: `tests/unit/test_audit2609_a14_rcwa_eme_bor.py::test_h1_weakly_guiding_fiber_is_not_an_empty_list`,
`::test_h1_margin_scales_with_the_window_not_with_k0`,
`::test_h1_degenerate_window_raises_instead_of_returning_empty`.

### Fixed -- RCWA: the Rayleigh-anomaly wavelength nudge was silent, one-sided and non-monotone (H2)

When a diffracted order sits EXACTLY at cut-off the solver substitutes
`wavelength * (1 + 1e-7)` because the half-space mode basis is genuinely degenerate there
(`kz = 0` gives two coincident modes and a singular interface match -- confirmed:
`LinAlgError: Singular matrix` from `_interface_smatrix` for 1-D TE and for both 2-D
polarizations at the canonical mount).  The substitution was completely silent, and the
one-sided answer was 4.05e-04 relative wrong AND sat above BOTH of its neighbours in a
wavelength sweep -- a one-point spike, not a limit.

Three changes: (1) the nudge now emits a `WoodNudgeWarning` (a new public category so a
deliberate on-anomaly sweep can filter it, or a strict caller can promote it to an error)
naming the requested wavelength, the effective one and the relative shift; (2) the RCWA
entry points report the SYMMETRIC AVERAGE of two solves bracketing the requested wavelength
(`wavelength * (1 -/+ rel)`, exactly symmetric), on the NARROWEST bracket that clears the
detection threshold -- `rel` grows geometrically from a new RCWA-local
`_WOOD_PAIR_STEP_REL = 1e-9`, two decades narrower than the shared one-sided step; (3)
`Efficiency2D.wl_eff` and `RCWAResult.wl_eff` carry the wavelength(s) actually
solved -- a float normally, the `(lo, hi)` pair when the answer is an average.

Measured at the Moharam mount `Lambda = lambda = 1 um` (n_ridge 2.04, d = 1 um, duty 0.5,
normal incidence, n_orders 21) against an independent direct-4N-boundary-match oracle at the
EXACT wavelength (0.155845785342, its own closure 1.8e-15):

* TM R0: 0.155908839054 (+4.05e-04 rel) -> 0.155846827297 (+6.69e-06 rel), a factor **60.5**;
* TE R0: 0.040265175260 (+1.36e-05 rel) -> 0.040264572250 (-1.42e-06 rel), a factor 9.5;
* over the whole order array, TM max|dR| 6.31e-05 -> 2.34e-06, TE 5.46e-07 -> 3.96e-07.
  The TRANSMITTED orders are the larger residual and were not quoted at first: TM
  max|dT| is **1.45e-05** after the fix, which is the largest single deviation
  anywhere in the 24-configuration oracle sweep (VERIFY-A14 V12);
* warnings emitted: 0 -> 1;
* the TM R0 sweep over `delta = -3e-7 .. +3e-7` is now strictly monotone (smallest forward
  difference +5.23e-06); before, the value AT the anomaly equalled its `+1e-7` neighbour and
  exceeded its `+3e-8` one, so the sequence decreased twice;
* energy closure through the average: -3.75e-14 (TM) / +1.87e-14 (TE);
* the audit's own 24-configuration oracle sweep still agrees to <= 1.5e-13 at the 22 off-anomaly
  configurations (its <= 2.0e-13 bar), and the 12-configuration energy sweep to <= 1.03e-13.

NOTE ON THE AUDIT'S RATIONALE: the symmetric average is NOT "the continuous limit to
O(delta^2)".  A Wood anomaly is a SQUARE-ROOT branch point in the wavelength, so both the
one-sided and the averaged error fall as `sqrt(delta)` -- measured over five decades
(`delta` 1e-6 .. 3e-9, ratio 1.732 per 3x in delta) and re-confirmed independently on
FOUR mounts x two polarizations (per-decade ratio 3.15-3.18 against `sqrt(10)` = 3.162).

WHICH LEVER PAYS (re-measured, VERIFY-A14): two things changed at once -- the bracket is
100x narrower than the shared one-sided step, and the two sides are averaged.  Isolated at
a MATCHED `delta = 1e-7`, the AVERAGE helps on 3 of those 8 mount/polarization arms and
HURTS on 5, by up to 10x (lossy Ag TM: one-sided +8.0e-05 against average -7.7e-04); the
6.05x it buys on the Moharam TM arm is its best case, not its behaviour.  The reliable
lever is the NARROWER BRACKET, which the `sqrt(shift)` law turns into a guaranteed ~10x on
every mount, and which is only possible because the solve stays clean far below it (closure
2e-14 .. 5e-14 and the physical `sqrt(shift)` agreement measured all the way down to a
1e-13 shift).  What the average buys UNCONDITIONALLY is CONTINUITY of the wavelength sweep
-- the property no choice of one-sided step can have, and the one an optimiser differencing
through the anomaly actually trips over.

HOW ACCURATE THE RESULT IS, ACROSS MOUNTS (not just the fixture below): the post-fix
residual against an exact-wavelength oracle is 6.7e-06 relative on the Moharam mount but
reaches **2.9e-04** on a `n_sub = 1.5` substrate anomaly, and the gain over the one-sided
nudge ranges **1.05x .. 60.5x** over the eight arms.  The finding is MITIGATED and now
ANNOUNCED; it is not eliminated.

The SHARED one-sided nudge is numerically unchanged (`1e-7` per iteration, detection threshold
`1e-9`): `_grazing_safe_wavelength` is imported by 12 sites in `elements/pmm`, whose staggered
engine degrades like `1/sqrt(distance)` TOWARD a cut-off, so a narrower step would help RCWA and
hurt PMM.  The narrower bracket and the averaging are both local to the RCWA path.

ONE THING THE AVERAGE GETS WRONG, stated exactly: an order that is EXACTLY grazing at the
requested wavelength carries no z-directed power (a theorem; the oracle returns exactly 0).  The
one-sided nudge happened to agree -- at `+delta` those orders are evanescent -- while the average
does not, because at `-delta` they are propagating and carry ~`sqrt(delta)` of power, half of
which survives the mean.  Measured at the shipped bracket, m = +/-1: TM R 0.0 -> 2.344e-06, TE
R 0.0 -> 3.963e-07 on the Moharam mount, and 10x smaller than at a 1e-7 bracket.  Relative to
the specular order of the same port that is 4.8-5.0 decades HERE, but the census over four
mounts x two polarizations puts the worst at **3.25 decades** (lossy Ag TM transmission) --
the bound that generalises is the bracket's own `sqrt(2 * 1e-9) = 4.5e-05` times an O(1)
coefficient, and the measured worst is **3.81e-05 absolute / 5.67e-04 relative**.  The power
comes OUT of the specular order, so the closure stays exact; zeroing those orders afterwards
would leave the closure short by exactly that amount and trip the library's own 1e-6 lossless
clause, so it was rejected.  Pinned by a dedicated test against those derived bars.

The TRACED (JAX) path does NOT carry this artefact: with a traced wavelength the anomaly
cannot be detected host-side, so the grazing mode is regularised in place at the `+rel` leg's
own offset (`_traced_grazing_floor`) and stays EVANESCENT -- it carries exactly zero power.
That path used to return all-NaN under `jax.jit`; see the H2 note below.

Migration: `rcwa_efficiency_1d`, `rcwa_jones_1d`, `rcwa_jones_1d_segments`,
`rcwa_efficiency_2d`, `rcwa_efficiency_2d_shapes`, `RCWA2DPrepared.solve`, `rcwa_jones_2d`
and `RCWAStack.solve` return DIFFERENT numbers at an exact Wood anomaly than before (closer
to the exact-wavelength answer), and now warn there.  Off-anomaly solves are bit-identical
and unwarned.  `RCWAStack.solve(retain_internal=True)` keeps the one-sided nudge (the
retained per-layer partial S-matrices are not linear in the field, so averaging them is not
defined) and says so in its warning.  To restore silence,
`warnings.filterwarnings('ignore', category=lumenairy.elements.rcwa.WoodNudgeWarning)`.

Files: `lumenairy/elements/rcwa/_core.py:1651-1962` (the warning category, the narrow symmetric
bracket `_WOOD_PAIR_STEP_REL`, the `_WoodAnomaly` control-flow signal, the `_wood_symmetric`
decorator and the result combiner), `lumenairy/elements/rcwa/oned.py:603, :1365, :1575`, `lumenairy/elements/rcwa/twod.py:1093, :1297, :1813, :2313`,
`lumenairy/elements/rcwa/stack.py:642, :647, :2982`.
Tests: `tests/unit/test_audit2609_a14_rcwa_eme_bor.py::test_h2_the_nudge_announces_itself`,
`::test_h2_exact_wood_point_beats_the_one_sided_nudge`,
`::test_h2_wavelength_sweep_is_monotone_through_the_anomaly`,
`::test_h2_exactly_grazing_orders_stay_under_the_sqrt_bracket_bound`,
`::test_h2_wl_eff_is_on_the_result`,
`::test_h2_off_anomaly_solves_are_untouched_and_unwarned`.

### Fixed -- RCWA: `rcwa_jones_2d(formulation='fff_nv')` manufactured form birefringence on a symmetric cell (H3)

On the Jones entry `fff_nv` selects the Li-2003 successive `L2 L1` tensor factorization,
which factorizes x first and y second.  That order is not x<->y symmetric, so a cell that is
(`np.array_equal(cell, cell.T)`) came back with `Jxx != Jyy` at normal incidence, where the
cell's own C4 / C-infinity symmetry makes them identical.  The in-plane operator is now the
symmetric mean `(L2 L1 + L1 L2) / 2`, which is exactly symmetric for a transpose-symmetric
cell (`T L2L1(eps) T = L1L2(eps^T)`) at the cost of one extra scalar-pivot factorization.

Measured `|Jxx - Jyy|` (period 0.5 um, depth 0.3 um, lambda 0.633 um, eps 6.25 in 2.25,
96x96 cell, normal incidence), `n_orders` 4 -> 12:

| cell | before | after | `laurent` / `li` reference |
|---|---|---|---|
| square (C4) | 9.04e-04 -> 1.06e-04 | 4.1e-14 -> 1.4e-13 | 5e-15 .. 3e-13 |
| disk (C-inf) | 2.06e-02 -> 5.75e-03 | 5.4e-15 -> 4.5e-14 | 1e-15 .. 2e-13 |

A y-uniform (separable) stripe is unchanged (the two orders coincide analytically there):
max|dJ| 8.7e-14 / 4.5e-14 / 1.3e-12 at `n_orders` 4 / 8 / 12; the same holds on an
x-UNIFORM stripe (8.7e-14), which only the transposed leg can get right, and on a
non-square raster with `n_orders_x != n_orders_y`.

MIGRATION, stated for the general cell (VERIFY-A14 V10): a cell that is NOT
transpose-symmetric also returns DIFFERENT numbers than before -- measured max|dJ| between
the symmetrized and the single-order operator of **8.3e-04 at `n_orders` 4 falling to
1.2e-04 at 12** on three axis-aligned fixtures (an off-centre rectangle, a 12x72
rectangle, a two-bar union).  That is inside the truncation error and the two converge to
the SAME limit (both track `li` at `n_orders` 16 to 7.5e-05 .. 2.1e-04 at 12, with the mean
equal or slightly closer on every rung), but a user pinning `fff_nv` Jones values will see
the change.  Only a SEPARABLE cell is bit-unchanged.

Second half of the finding: `rcwa_efficiency_2d(formulation='fff_nv')` REFUSES a curved cell
while the Jones entry accepted the same disk silently.  `rcwa_jones_2d` now emits a
validated-scope `UserWarning` naming the diagonal-boundary fraction (18% on the disk, 0 on
the square), with a new `allow_nonseparable_nv=False` kwarg to silence it.  It WARNS rather
than raising because the Li-2003 failure mode here is a convergence rate, not the
normal-vector method's ~50% absorptance mis-split.

`RCWAStack`'s `_li_blocks` passes `symmetrize=False`: that call wants the Li-1997 PER-AXIS
rule, whose documented exact reduction to `_li_convolutions_2d` the mean would break.

Files: `lumenairy/elements/rcwa/twod.py:410-539` (the symmetrized entry plus the split-out
single-order `_li_tensor_l2l1`), `:726-772` (`_li_tensor_scope_notice`), `:1861`;
`lumenairy/elements/rcwa/stack.py:2654-2661`.
Tests: `tests/unit/test_audit2609_a14_rcwa_eme_bor.py::test_h3_fff_nv_keeps_the_cells_own_symmetry`,
`::test_h3_separable_stripe_is_unchanged_by_the_symmetrisation`,
`::test_h3_curved_cell_gets_the_validated_scope_notice`.

### Performance -- RCWA / BOR (H4)

* **The even-parity fold now covers every in-plane formulation on `rcwa_jones_2d`.**  It was
  gated on `formulation == 'laurent'` even though it acts on the `(P, Q)` generator and is
  indifferent to how the permittivity operators were factorized, so `'li'` and `'fff_nv'`
  users always ran the full `2N` solve.  The operator set is now built once, before the
  attempt, and reused by whichever path runs.  Measured on a 96x96 square cell, 5 interleaved
  medians, `OPENBLAS_NUM_THREADS=1`: `'li'` 3.01x (`n_orders` 6) / 3.21x (9), `'fff_nv'`
  2.47x / 3.15x, against 1.00x before; `'laurent'` unchanged at 2.71x / 3.31x and
  bit-identical (it used the same direct-rule operators either way).  Agreement with the
  full solve: max|dJ| 5.3e-14 .. 3.4e-13, inside the documented "~1e-12, not bit-identical"
  even-basis contract.
* **`_inplane_ops` builds the Li operators once for an isotropic cell.**
  `_li_convolutions_2d` computes BOTH `Cxx` and `Cyy` on every call and the two-call form
  discarded half of each.  Guarded on array identity then value equality (skipped on JAX,
  which cannot branch on data): 1 call for an isotropic cell, still 2 for a genuinely
  anisotropic one, retained blocks bit-identical (0.0 / 0.0).
* **The BOR SEM pencils use `eigh(A, M)`.**  `radial_spectrum` formed `M^-1 A` explicitly and
  ran a non-symmetric `eig` on what is a symmetric-definite pair, then truncated a complex
  spectrum with `.real`.  Accuracy against the Bessel zeros IMPROVES 2-4x (degree 8,
  12 elements): m = 0/1/3 Dirichlet 3.11e-13/1.78e-13/5.42e-14 -> 1.19e-13/9.26e-14/1.38e-14,
  m = 1/3 Neumann 7.04e-13/1.83e-13 -> 1.74e-13/3.81e-14.  Whole-call speed 1.03x / 0.97x /
  1.70x at n = 97 / 241 / 481 -- below the audit's structural 3-5x estimate because the
  Python element-assembly loop dominates under n ~ 250.  `return_modes=True` eigenvectors are
  now `M`-ORTHONORMAL (`INT r psi^2 dr = 1`) instead of unit 2-norm: a scale change, and the
  natural normalisation for this weak form.  Both existing eigenvector gates normalise, so
  they are unaffected.

Files: `lumenairy/elements/rcwa/twod.py:1839-1899`; `lumenairy/elements/bor/radial_eigensolver.py:166-178`.
Tests: `tests/unit/test_audit2609_a14_rcwa_eme_bor.py::test_h4_even_parity_fold_covers_every_in_plane_formulation`,
`::test_h4_bor_pencil_eigh_accuracy`, `::test_h4_isotropic_cell_builds_the_li_operators_once`.

### Changed -- RCWA: the inert-BLAS-cap warning now quantifies what it costs (H5)

`set_blas_threads` / `rcwa_blas_threads` / the `@_with_blas_limit` wrapper on every public
entry point are inert without `threadpoolctl`, which is not a declared dependency.  The
existing once-per-process warning said the cap was inert; it now says what that costs,
measured: `inv()` of a 163x163 complex matrix 2.29 s unpinned vs 0.0057 s at one thread
(400x) on a 24-thread Windows OpenBLAS 0.3.31 build, and a 1-D TM solve at `n_orders = 81`
18.2 s instead of 0.13 s (140x) -- with both remedies spelled out.

The dependency declaration itself is requested from the tests/CI work package (exact line in
`WP-A14_REPORT.md` section 5).

Files: `lumenairy/elements/rcwa/_core.py:249-263`.

### Fixed -- RCWA / EME: documentation-vs-behaviour and aliasing hygiene (H6)

* **`RCWAResult.per_order_amplitudes()` now copies every array it hands out.**  `kz` was
  explicitly copied "so the public dict keeps its writable-array contract", but
  `Ex`/`Ey`/`kx`/`ky`/`orders` were handed out by reference into the result's own modal dict,
  so `amp['Ex'][:] = 0` made the NEXT call on the SAME result return `max|Ex| = 0.0`
  (`to_numpy` is a no-op view on a NumPy backend, which is how the alias survived the W7-B
  pass).  One dict of copies per call is ~5 x (2, N) complex, negligible beside the solve.
* **`_check_energy` gained a passive-structure one-sided bar.**  The tight lossless closure
  clause is disarmed by ANY complex permittivity, so a metal grating in air -- where
  `R + T <= 1` is a THEOREM because the incidence medium is lossless and nothing has gain --
  could return `R + T` anywhere in `(1, 1.05]` with no signal.  The new clause (armed by
  `_passive_media`: exactly lossless incidence, no gain anywhere, symmetric tensors) warns
  above a `1e-6` one-sided excess, ~7 decades above the 1.4e-13 closure clean solves hold and
  3.7 decades below the 1.05 hard tripwire that was the only guard before.
* **The M8 "`n_orders_y = 0` reproduces `rcwa_efficiency_1d` per order to ~5e-15" claim is
  restated with its rasterisation scope.**  The 1-D core builds exact analytic step
  coefficients while the 2-D core FFTs a rasterised cell, so the gap is a rasterisation
  error, clean `O(1/Sx^2)`: 5.16e-04 at `Sx = 64` down to 1.26e-07 at 4096, and ~1e-3 at the
  minimum sampling `_validate_cell_sampling` allows.  The ~5e-15 is the y-harmonic claim
  (`N_y = 0` vs `N_y = 1`), not a 1-D/2-D equivalence.
* **EME `eme_2d.strip_x_modes`' dead `np.isrealobj` arm removed** -- `A` is built
  `dtype=complex` unconditionally, so the `.real` arm never ran.
* **EME `ref_2d_modes` uses `np.conj(px)` / `np.conj(py)` for the Bloch wrap**, the same rule
  `strip_x_modes` states at `:83-86` and for the same reason (for `|p| = 1` the two agree
  analytically but `1/p` carries a roundoff error that makes the operator only
  APPROXIMATELY Hermitian -- and this is the FD oracle that Hermitian operator is compared
  against).  Bit-identical at `kx0 = ky0 = 0`.

Files: `lumenairy/elements/rcwa/stack.py:736-745, :2337-2353`;
`lumenairy/elements/rcwa/_core.py:932-1026, :1801-1831 (docstring)`; `lumenairy/elements/rcwa/oned.py:761`; `lumenairy/elements/rcwa/twod.py:1235, :1480, :2025, :2389`;
`lumenairy/elements/eme/eme_2d.py:90-96, :451-455`.
Tests: `tests/unit/test_audit2609_a14_rcwa_eme_bor.py::test_h6_per_order_amplitudes_hands_out_copies`,
`::test_h6_passive_media_predicate`, `::test_h6_passive_bound_is_armed_on_a_lossy_cell`.

### Fixed -- RCWA: `_sqrt_decay`'s on-cut predicate flipped near-ZERO evanescent modes (G11)

The pin tested `|Re r| <= band * scale`, i.e. proximity to the ORIGIN on the SPECTRUM's
scale, which a deeply evanescent root also satisfies once its own magnitude has collapsed:
`_sqrt_decay([1e-20 - 1e-30j])` returned `-1e-10 + 5e-21j`, handing a genuinely DECAYING
mode back with `Re(lam) < 0` -- the `exp(+|gamma| k0 L)` growth the `Re >= 0` rule exists to
prevent.  Because the scale is the array maximum, a spectrum with a large top could pull an
ordinary near-cutoff evanescent mode in too, with a worst price
`|X| = exp(band * max|lam| * k0 L)` = 1.059 / 302 / 1e248 for spectrum scales 1 / 1e2 / 1e4
at the docstring's own worst `k0 L = 1.14e7`.  The predicate gains a third conjunct
`Im(r)^2 > Re(r)^2` -- "nearer the IMAGINARY axis than the real one", which is what "on the
cut" means and is scale free -- so the docstring's "a flipped mode is by construction a
PROPAGATING one" becomes a theorem rather than a census.

Inert on every population the band was derived against: the worst `|Re r| / |r|` ever
flipped there is 2.0751e-03, so `Im^2 / Re^2 > 2e5`, five decades clear of the new edge.
All RCWA / PMM branch-cut, even-sector and round-2/3 verification gates pass unchanged.

Files: `lumenairy/elements/rcwa/_core.py:1564-1579` and the docstring's PRICE paragraph.
Tests: `tests/unit/test_audit2609_a14_rcwa_eme_bor.py::test_g11_near_zero_evanescent_mode_is_not_flipped`,
`::test_g11_scale_relative_band_cannot_flip_an_evanescent_mode`,
`::test_g11_genuine_on_cut_propagating_mode_still_flips`.

### Added -- BOR: the propagating S-matrix's SUPERPOSITION energy closure is now gated

The audit measured that the staggered BOR modal basis is flux-orthonormal and the
propagating S-matrix is unitary, so energy closes for ARBITRARY excitation -- a strictly
stronger statement than the per-channel `energy` array (which is exactly the DIAGONAL of
`U^H U`) can make, and nothing in the suite pinned it.  Gate added on an index-matched
lossless ring grating (Rbig 12 um, m = 1, one 0.5 um ring layer, lambda 1 um, 78 propagating
channels): Gram max|offdiag| 2.50e-12 and worst closure 1.67e-12 over 300 random unit-norm
multi-channel inputs, against a 1e-9 bar.

Tests: `tests/unit/test_audit2609_a14_rcwa_eme_bor.py::test_bor_propagating_smatrix_is_unitary_and_closes_on_superpositions`.

### Fixed -- RCWA: `jax.jit` returned all-NaN at an exact Wood anomaly (H2 follow-up, VERIFY-A14 V5)

The host-side Rayleigh-anomaly nudge needs a CONCRETE wavelength -- it compares
`|eps - kt^2|` against a threshold and moves the wavelength if any order is at cut-off.
Under `jax.jit` every constant built inside the traced function is a `DynamicJaxprTracer`,
so `geom_concrete` was False, the whole guard block was SKIPPED and the solve ran AT the
anomaly, where the half-space basis has two coincident modes.  Measured on the canonical
`Lambda = lambda = 1 um` mount: `jax.jit(rcwa_efficiency_1d)` returned `NaN` for every
order and `jax.grad` through it was NaN too, for TE and TM alike -- while the same call
EAGERLY (concrete arrays, host-side nudge) returned 0.040057645.  Forcing the PRE-2026-09-12
one-sided wavelength reproduced the NaN, so this predates the symmetric bracket.

The traced path now regularises the grazing mode in place instead, at the `+rel` leg's own
offset: an order whose `|kz^2|` is below `2 * _WOOD_PAIR_STEP_REL * |eps|` is pushed onto
the EVANESCENT side at exactly `|kz| = sqrt(2 * rel * |eps|)`, and the SAME shifted `kz^2`
builds the mode matrix `Q` -- which is the load-bearing half, because for a uniform
half-space `det Q = eps^N prod(kz^2)`, so `V = Q diag(1/lam)` is exactly rank-deficient
whenever any `kz = 0` no matter how `1/lam` is floored (flooring `lam` alone was measured
and left the NaN in place).

Measured after: finite for TE and TM, closure `|sum R + sum T - 1| <= 4.4e-16`, agreement
with the eager (bracket-mean) answer **9.4e-08 (TE) / 5.3e-06 (TM)** -- inside the same
`sqrt(delta) ~ 4.5e-05` band the bracket itself carries -- and `jax.grad` finite (0.4397).
The floored order stays EVANESCENT, so unlike the symmetric average the traced path leaves
an exactly grazing order at EXACTLY zero power.  OFF the anomaly the floor never binds and
the traced path is BIT-IDENTICAL with it and without it (measured 0.0 over TE/TM x five
wavelengths).  The NumPy and concrete-JAX paths are untouched (`grazing_floor=None`).

The 2-D entry points are unaffected: with a traced wavelength they refuse LOUDLY
(`TracerArrayConversionError`) before reaching the solver, and with a concrete wavelength
and a traced cell they take the host-side nudge as before (verified finite and closing at
1e-16 at the anomaly).

Files: `lumenairy/elements/rcwa/_core.py` (`_traced_grazing_floor`, `grazing_floor=` on
`_homogeneous_eigenmodes` and `_layer_eigenmodes`), `oned.py`.
Tests: `tests/unit/test_rcwa.py::test_jax_wood_anomaly_no_nan` (extended to run under
`jit` with a traced layer index, both polarizations, with the off-anomaly bit-identity arm).

### Fixed -- BOR: a SHORT `guided_modes` result is no longer silent either (H1 follow-up, VERIFY-A14)

The 2026-09-12 H1 work made an EMPTY list audible through every filter.  A list that was
merely SHORT stayed quiet: Si/SiO2 (`dn = 2.04`) at V = 4.0 returned ONE mode while the
exact hybrid characteristic equation has THREE (`n_eff` = 3.038391486 / 1.566626407 /
1.440009396), with no signal at all.  Every call now also counts the roots of that exact
equation for the requested azimuthal order -- a sign-change scan of `fiber_oracle.fiber_det`,
the 4x4 Bessel boundary-match determinant, which shares no code with the finite-difference
vector eigensolver -- and WARNS, naming both counts and the order, when the solver returns
fewer.

The census is a scan, not a solve: 2001 samples, no bisection, MEASURED 49-72 ms against
0.72 s / 17.9 s / 67.8 s for the FD eigensolve at N = 150 / 400 / 600, i.e. 9.9% of the
call at the smallest grid anyone uses and 0.1-0.4% at the grids real work runs.  Its
resolution is `(n_core - n_clad) / 2000` in `n_eff`, so two roots closer than one cell (a
near-degenerate HE/EH pair, or a tangential double root) are counted once -- an error that
is ONE-SIDED in the safe direction, so the notice can miss a shortfall but can never invent
one.  Counts verified identical to the BISECTING `fiber_modes` on twelve fixtures spanning
m = 0..5, V = 1.8..9 and three index systems.  `census=False` skips the scan.

Files: `lumenairy/elements/bor/coupled_radial_eigensolver.py`
(`_step_index_root_census`, `_CENSUS_SCAN`, `census=` on `guided_modes`).
Tests: `tests/unit/test_audit2609_a14_verify.py::test_h1_a_short_guided_mode_list_is_never_silent`.

### Fixed -- RCWA: the `fff_nv` validated-scope notice now reaches the OUT-OF-PLANE path (H3 follow-up, VERIFY-A14 V6)

`_li_tensor_scope_notice` was reachable only from the IN-PLANE branch of
`rcwa_jones_2d`, so an out-of-plane (full 3x3) tensor cell with a CURVED pattern got
`formulation='fff_nv'` with no scope signal at all -- while the identical in-plane cell
warned.  The off-plane path runs the same Li-2003 staircase factorization (and, per the
deferred D3, the un-symmetrized one), so it carries the same notice now: measured 1 notice
on an out-of-plane disk, 0 on an out-of-plane square, 0 for `laurent` / `li` on either, and
silenced by `allow_nonseparable_nv=True`.

Files: `lumenairy/elements/rcwa/twod.py`.
Tests: `tests/unit/test_audit2609_a14_verify.py::test_h3_offplane_fff_nv_gets_the_scope_notice_too`.
