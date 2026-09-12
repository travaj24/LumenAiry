# WP-A14 report -- RCWA, EME and BOR (`elements/rcwa`, `elements/eme`, `elements/bor`)

Audit: `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` section 13 (H1-H6), the G11 row of section 12,
partition report `RCWA-EME-BOR.md`.  Branch `audit-fixes-2026-09`, base HEAD `437c1b06`.
Every measurement below was taken on this machine with `OPENBLAS_NUM_THREADS=1`
(`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).

All findings were re-confirmed on the current HEAD before any change.  The BOR/EME modules had
moved since the audit revision (rounds 2/3 of the BOR multilayer guards, 2026-09-12), so every
citation was re-located by content; the H1 site had moved from `:514-520` to `:495-529` and the
EME `ref_2d_modes` site from `:439-441` to `:450-454`, with no change of substance.

---

## 1. Summary

| # | Status | Files:lines | Tests | Oracle | Measured before -> after |
|---|---|---|---|---|---|
| **H1** (P1) | **fixed** | `bor/coupled_radial_eigensolver.py:495-586` | `test_audit2609_a14_rcwa_eme_bor.py::test_h1_weakly_guiding_fiber_is_not_an_empty_list`, `::test_h1_margin_scales_with_the_window_not_with_k0`, `::test_h1_degenerate_window_raises_instead_of_returning_empty` | exact hybrid HE11 4x4 boundary-match determinant (`fiber_oracle`) | `dn = 0.010`: **0 modes -> 1 mode**, `n_eff` 1.445294274 vs exact 1.445293173 (err +1.10e-06). `dn = 0.005`: **0 -> 1**, 1.447648919 vs 1.447648366 (+5.53e-07). Warnings 0 -> the degenerate case now raises. |
| **H2** (P1) | **fixed** | `rcwa/_core.py:1635-1930`; `oned.py:568,:1311,:1521`; `twod.py:1094,:1298,:1814,:2304`; `stack.py:642,:647,:2977` | `::test_h2_the_nudge_announces_itself`, `::test_h2_exact_wood_point_beats_the_one_sided_nudge`, `::test_h2_wavelength_sweep_is_monotone_through_the_anomaly`, `::test_h2_exactly_grazing_orders_stay_four_decades_below_the_specular`, `::test_h2_wl_eff_is_on_the_result`, `::test_h2_off_anomaly_solves_are_untouched_and_unwarned` | independent direct 4N boundary-match 1-D RCWA (re-derived inside the test file) at the EXACT wavelength | TM R0 at `Lambda = lambda = 1 um`: **0.155908839054 (+4.05e-04 rel) -> 0.155846827297 (+6.69e-06 rel)**, a factor 60.5 (TE 9.5x). Warnings **0 -> 1**. The `lambda` sweep through the anomaly is **non-monotone -> strictly monotone**. `wl_eff` now on `Efficiency2D` / `RCWAResult`. |
| **H3** (P2) | **fixed** | `rcwa/twod.py:411-540`, `:726-772`, `:1861`; `stack.py:2654-2661` | `::test_h3_fff_nv_keeps_the_cells_own_symmetry`, `::test_h3_separable_stripe_is_unchanged_by_the_symmetrisation`, `::test_h3_curved_cell_gets_the_validated_scope_notice` | the cell's own C4 / C-infinity symmetry (`Jxx == Jyy` exactly at normal incidence) | `|Jxx - Jyy|` square **9.04e-04 -> 4.1e-14** (M = 4), **1.06e-04 -> 1.4e-13** (M = 12); disk **2.06e-02 -> 5.4e-15** (M = 4), **5.75e-03 -> 4.5e-14** (M = 12). Curved-cell notices on the Jones entry **0 -> 1**. |
| **H4** (P2, perf) | **partially fixed** (3 of 5 items; 2 deferred with designs) | `rcwa/twod.py:1840-1900`; `bor/radial_eigensolver.py:165-177` | `::test_h4_even_parity_fold_covers_every_in_plane_formulation`, `::test_h4_bor_pencil_eigh_accuracy`, `::test_h4_isotropic_cell_builds_the_li_operators_once` | full `2N` solve (`symmetry=False`); Bessel zeros `j_{m,n}` / `j'_{m,n}` | even fold on `rcwa_jones_2d`: `'li'` **1.00x -> 3.01x / 3.21x**, `'fff_nv'` **1.00x -> 2.47x / 3.15x** (`n_orders` 6 / 9), agreement 5.3e-14..3.4e-13. `_inplane_ops` Li builds per isotropic solve **2 -> 1**, bit-identical. BOR `eigh(A,M)`: relative error vs Bessel zeros **3.11e-13/1.78e-13/5.42e-14/7.04e-13/1.83e-13 -> 1.19e-13/9.26e-14/1.38e-14/1.74e-13/3.81e-14**; whole-call 1.03x / 0.97x / **1.70x** at n = 97 / 241 / 481. |
| **H5** (P3) | **fixed** (code half) + **requested** (dependency line) | `rcwa/_core.py:233-247` | -- (the existing `test_niche_audit_m4_m5_m6_rcwa.py` gate covers the inert path) | measured 400x / 140x | warning text now carries the measured cost and both remedies. The `pyproject.toml` / `requirements.txt` lines are in section 5 below (not my files). |
| **H6** (P3) | **fixed** (5 items) + **deferred** (3 code-organization items) | `rcwa/stack.py:736-745,:2349-2365`; `_core.py:916-1010`, M8 docstring; `oned.py:761`; `twod.py:1236,:1481,:2016,:2380`; `eme/eme_2d.py:90-96,:450-454` | `::test_h6_per_order_amplitudes_hands_out_copies`, `::test_h6_passive_media_predicate`, `::test_h6_passive_bound_is_armed_on_a_lossy_cell` | direct mutation probe; the three passivity clauses | `amp['Ex'][:] = 0` then re-read: **max\|Ex\| = 0.0 -> unchanged**. Passive `R+T <= 1` bar: **unarmed (1.05 only) -> 1e-6 one-sided**. M8 claim restated with its `O(1/Sx^2)` rasterisation scope. |
| **G11** (P3) | **fixed** | `rcwa/_core.py:1548-1563` + docstring | `::test_g11_near_zero_evanescent_mode_is_not_flipped`, `::test_g11_scale_relative_band_cannot_flip_an_evanescent_mode`, `::test_g11_genuine_on_cut_propagating_mode_still_flips`; `test_fix_branch_cut_round2.py::test_the_shared_body_narrows_the_round_one_SELECTOR_to_the_cut` (restated) | the involution itself: a flipped root must have `Re(lam^2) < 0` | `_sqrt_decay([1e-20 - 1e-30j])`: **-1e-10 + 5e-21j -> +1e-10 - 5e-21j** (principal root kept). Worst `\|X\|` at `k0 L = 1.14e7` for spectrum scales 1 / 1e2 / 1e4: **1.059 / 302 / 1e248 -> 1.0 / 1.0 / 1.0**. |
| **extra** | **added** | -- | `::test_bor_propagating_smatrix_is_unitary_and_closes_on_superpositions` | unitarity of the propagating S-matrix | Gram max\|offdiag\| 2.50e-12, superposition closure 1.67e-12 over 300 random unit-norm inputs (nothing gated this before). |

---

## 2. Per finding

### H1 (P1) -- `guided_modes` returned `[]` for every weakly-guiding fiber

**What was wrong.**  `qlo + q_margin < q.real < qhi - q_margin` with `q_margin = 5e-3 * k0`.  The
guided window is `(n_core - n_clad) * k0` wide, so a band that is a fraction of `k0` -- not of the
window -- admits nothing unless `dn > 0.01`.  The W6-B2 fix had made the margin unit-invariant and
left it contrast-dependent, which is the same bug in the other axis.

Re-confirmed on HEAD before changing anything (`scratchpad/wpa14/h1_before.py`): `lambda = 1.55 um`,
`V = 2.4`, `m = 1`, `(Rbig, N)` in `{6a} x {300, 600}` -- `guided_modes` returned **0 modes and 0
warnings** for both `dn = 0.010` (1.45 / 1.44) and `dn = 0.005` (1.45 / 1.445), while
`radial_coupled_modes` on the same geometry held the HE11 at `n_eff = 1.445293453` (err +2.80e-07
against the exact oracle's 1.445293173) with `reldiv = 1.98e-03`, i.e. inside every other filter.
The window/band arithmetic: window `= 0.010000 k0` against `2 * q_margin = 0.010000 k0`.

**What I changed and why.**  `q_margin = max(1e-6 * k0, 1e-3 * (qhi - qlo))` -- a fraction of the
WINDOW (contrast-invariant) floored on `k0` (unit-invariant), so both halves of the invariance
statement now hold.  `imag_tol` is clamped to `0.5 * window`, which is never looser than the old
`5e-4 * k0` and only binds where the old tolerance exceeded the whole guided band.  A window
narrower than twice the band raises `ValueError` naming both permittivities (nothing can be
admitted there, and saying so is the one thing an empty list cannot).  An empty result that HAD a
candidate inside the band now warns naming the closest one, so the "silent `[]`" mode cannot return
through a different door.

**How I verified.**  `scratchpad/wpa14/h1_after.py`, against `fiber_oracle`'s exact hybrid HE11:

| fixture | `N` = 150 | 200 | 300 |
|---|---|---|---|
| `dn = 0.010`, exact 1.445293173 | 1.445294274 (+1.10e-06) | 1.445233747 (-5.94e-05) | 1.445293453 (+2.80e-07) |
| `dn = 0.005`, exact 1.447648366 | 1.447648919 (+5.53e-07) | 1.447618708 (-2.97e-05) | 1.447648507 (+1.41e-07) |
| legacy `dn = 1.04` (`e1=6, e2=2`) | 1 mode | 1 mode | (0 -- unchanged from before, the reldiv/tail filters) |

Exactly one mode is returned in every weakly-guiding case, and the legacy `dn = 1.04` fixtures that
the four existing call sites use are unaffected (the new band is *looser* there: `1e-3 * 1.035 k0`
vs `5e-3 k0`).  `tests/unit/test_coupled_eigensolver.py` and `test_niche_audit_w6_bor.py` pass.

**Residual risk.**  The convergence is not monotone in `N` (the `N = 200` row is 50x worse than
`N = 150` and `N = 300`), consistent with the audit's unverified suspicion that the staggered/FD
order drops to first at a material interface that does not land on a face.  The test bar (2e-4) is
set from the measured family, not from an assumed order.  The `imag_tol` clamp is a tightening; it
cannot admit anything the old rule rejected.

### H2 (P1) -- the Rayleigh-anomaly nudge was silent, one-sided and non-monotone

**What was wrong.**  `_grazing_safe_wavelength` substitutes `wl * (1 + 1e-7)` (up to 8 times)
whenever any diffracted order sits exactly at cut-off, and returned that answer with no warning and
no record.  Re-confirmed on HEAD (`scratchpad/wpa14/h2_before.py`), Moharam mount
`Lambda = lambda = 1 um`, `n_ridge = 2.04`, `d = 1 um`, duty 0.5, normal, `n_orders = 21`,
`formulation='li'`:

* TM R0 **0.155908839054** against my independent boundary-match oracle's **0.155845785342** at the
  exact wavelength -- 6.31e-05 absolute, **4.05e-04 relative**; TE 5.46e-07 (1.36e-05 rel);
* `warnings.catch_warnings(record=True)` around the solve: **empty**;
* `_grazing_safe_wavelength(1e-6, ...)` returns `1.0000001e-06`, confirming the substitution;
* the value at `delta = 0` (0.155908839054) sits ABOVE both neighbours (0.155822748627 at -3e-8,
  0.155880240631 at +3e-8) -- a one-point spike, not a limit.

**Two things I established that the audit did not, and that shaped the fix:**

1. **The nudge is genuinely load-bearing for most mounts, not a defensive habit.**  With the nudge
   monkeypatched to the identity, the exact-wavelength solve raises
   `LinAlgError: Singular matrix` from `_interface_smatrix` (`b = solve(Vb, Va)`) for 1-D **TE** at
   `n_substrate = 1` and for **both** 2-D polarizations.  1-D TM happens to survive, and where it
   does it reproduces the oracle to **1.06e-13 at every truncation** (M = 11/21/31/41/61).  So the
   exact problem is well posed and this formulation's half-space mode basis is what is defective at
   `kz = 0` (two coincident modes, `_inv_lam` floors `1/lam`).  "Just solve at the exact wavelength"
   is therefore not available as a general default.
2. **The symmetric average is NOT accurate to `O(delta^2)`** as the audit's fix note and the WP
   brief state.  A Wood anomaly is a SQUARE-ROOT branch point in the wavelength, so both the
   one-sided and the averaged error fall as `sqrt(delta)`.  Measured over five decades against the
   oracle (TM R0):

   | `delta` | one-sided `+delta` | symmetric average |
   |---|---|---|
   | 1e-06 | +2.02e-04 | +3.30e-05 |
   | 1e-07 (the OLD nudge) | +6.31e-05 | +1.04e-05 |
   | 1e-08 | +1.99e-05 | +3.30e-06 |
   | 1e-09 (the shipped bracket) | +6.28e-06 | **+1.04e-06** |
   | 3e-09 | +1.09e-05 | +1.81e-06 |

   Ratio 1.732 = `sqrt(3)` per 3x in `delta`, on both columns.  The average buys a factor 6.05 on
   the COEFFICIENT, not an order -- so the other lever is the bracket WIDTH, and that is the one
   that pays.  I report this rather than quoting the brief.

3. **The solve stays clean far below the shipped bracket.**  With the nudge disabled and the
   wavelength stepped manually, the 1-D TM closure is 2e-14 .. 5e-14 and the oracle agreement is
   the physical `sqrt(delta)` all the way down to `delta = 1e-13` (where the grazing mode's `kz` is
   4.5e-07 and `_inv_lam` is five decades above its 1e-12 floor).  Conditioning is therefore NOT
   what sets the bracket width; the `1e-9` DETECTION threshold is.

**What I changed and why.**

* **The nudge announces itself.**  New public warning category
  `lumenairy.elements.rcwa.WoodNudgeWarning` (a dedicated class so a deliberate on-anomaly sweep can
  filter it and a strict caller can promote it to an error), naming the requested wavelength, the
  effective one(s) and the relative shift.  This is the core of the P1 and it costs nothing.
* **The RCWA entry points report the symmetric average, on the NARROWEST bracket that clears the
  detection threshold.**  `_grazing_safe_wavelength_pair` returns `wavelength * (1 -/+ rel)` --
  exactly symmetric, so the two solves bracket the request and their mean is the request to
  rounding -- growing `rel` geometrically from `_WOOD_PAIR_STEP_REL = 1e-9` until BOTH sides clear
  `_WOOD_DETECT`.  That base step is two decades narrower than the shared one-sided
  `_WOOD_STEP_REL = 1e-7`, which is the point: the error falls as `sqrt(shift)`, so a 100x narrower
  bracket is a 10x better answer, and this constant is local to the RCWA path so PMM keeps its own
  (see below).  Its reach is `1e-9 * 10^7 = 1e-2` relative at the default `max_iter` -- four decades
  LONGER than the one-sided search's `8e-7` -- and if no symmetric bracket is reachable it falls
  back to the shared one-sided nudge.  Each entry point raises a private `_WoodAnomaly` the moment
  it detects an anomaly; its own `@_wood_symmetric` wrapper re-enters it twice with a private
  keyword-only `_wl_eff` and averages the two results.  A control-flow exception plus
  `inspect.signature().bind` was chosen over the obvious alternative (each entry point re-listing
  its ~20 arguments to call itself), which is exactly the site where a later kwarg silently fails to
  be forwarded.  Off-anomaly the wrapper is a bare call: no signature work, no second solve, no
  warning, bit-identical numbers.
* **`wl_eff` is on the result.**  `Efficiency2D.wl_eff` and `RCWAResult.wl_eff` carry the wavelength
  actually solved -- a float normally, the `(lo, hi)` pair when the answer is an average.  The
  1-D entries return bare tuples and cannot carry a field; for them the warning is the channel.
* **The SHARED step is deliberately unchanged.**  `_grazing_safe_wavelength` is imported by 12 sites
  in `elements/pmm/`, whose staggered engine degrades like `1/sqrt(distance)` TOWARD a cut-off
  (`test_pmm2d_staggered_wood_list.py` header) -- the opposite sign from RCWA, where the exact
  wavelength is the target.  Narrowing the shared step would have bought RCWA 10x and cost PMM,
  across a work package I do not own; the RCWA side takes the same 10x through its own
  `_WOOD_PAIR_STEP_REL`, which only `_grazing_safe_wavelength_pair` reads.  Both the narrower
  bracket and the averaging are local to the RCWA entry points, and PMM's numbers are untouched.

Covered entry points: `rcwa_efficiency_1d`, `rcwa_jones_1d`, `rcwa_jones_1d_segments`,
`rcwa_efficiency_2d`, `RCWA2DPrepared.solve`, `rcwa_jones_2d`, `rcwa_efficiency_2d_shapes`,
`RCWAStack._solve_once`.  `RCWAStack.solve(retain_internal=True)` keeps the one-sided nudge by
design (the retained per-layer partial S-matrices are not linear in the field) and says so.

**How I verified** (`scratchpad/wpa14/h2_after1d.py`, `h2_all_entries.py`; all eight entries):

* TM R0 4.05e-04 -> **6.69e-06** relative (absolute 6.31e-05 -> 1.04e-06, a factor 60.5);
  TE 1.36e-05 -> **1.42e-06** (5.46e-07 -> 5.72e-08, a factor 9.5);
* over the whole order array: TM max|dR| 6.31e-05 -> **2.34e-06**, TE 5.46e-07 -> **3.96e-07**;
* exactly one `WoodNudgeWarning` per entry point, naming both wavelengths;
* energy closure through the average 4.24e-14 (1-D) / 1.000000000000 (2-D, all three 2-D entries);
* the TM R0 sweep over `delta = -3e-7 .. +3e-7` is now **strictly monotone** (smallest forward
  difference +5.23e-06) and the anomaly value lies strictly between its neighbours;
* `wl_eff = (9.99999999e-07, 1.000000001e-06)` on `Efficiency2D` and `RCWAResult`, mean `1e-06` to
  rounding; a plain float off-anomaly;
* the audit's own 24-configuration oracle sweep (`repro/RCWA-EME-BOR/p02_oracle.py`) re-run on the
  fixed HEAD: **22 of 24 configurations agree to <= 1.5e-13** (the audit's <= 2.0e-13 "keep intact"
  bar), and the two that do not are exactly the two Wood rows -- which are now 27x (TM) and 1.4x
  (TE) closer than before on max|dR|.  `p01_energy.py` re-run: closure <= 1.03e-13 over all 12
  configurations (audit bar 1.4e-13);
* off-anomaly (`lambda = 0.9 um`) the library still matches the oracle to 3e-15 with no warning.

**The one thing the average gets wrong, stated exactly.**  An order that is EXACTLY grazing at the
requested wavelength carries no z-directed power -- a theorem, and the oracle returns exactly 0 for
it.  The one-sided nudge happened to agree (at `+delta` the m = +/-1 orders of this mount are
EVANESCENT); the average does not, because on the `-delta` side those orders are PROPAGATING and
carry ~`sqrt(delta)` of power, half of which survives the mean.  Measured at the shipped bracket,
m = +/-1: TM R 0.0 -> 2.344e-06, TE R 0.0 -> 3.963e-07 -- 10x smaller than at a 1e-7 bracket, as
`sqrt(delta)` requires, and 4-5 decades below the specular order of the same port.  The power is not
invented: it comes out of the specular order, so the closure stays exact (-3.75e-14 TM, +1.87e-14
TE).  Zeroing those orders afterwards was considered and REJECTED -- it would leave the closure
short by exactly that amount, which is a worse violation of a theorem than a per-order value four
decades below its neighbours, and it would trip the library's own 1e-6 lossless-closure warning.
The artifact is pinned by `::test_h2_exactly_grazing_orders_stay_four_decades_below_the_specular`
so it cannot grow silently.

**Residual risk.**  (a) The average changes the returned numbers at exact Wood anomalies -- a
deliberate default change, with the migration note in the changelog.  Two existing tests exercise
an exact anomaly (`test_rcwa.py::test_jax_wood_anomaly_no_nan`,
`test_v5_11_0_rcwa_fff_nv_2d.py::test_laurent_li_bit_identical_after_integration`); both pass, since
neither pins the value.  (b) The nudge now warns from the SHARED helper, so PMM entry points warn
too at an exact anomaly.  This is informative, not numeric; the PMM suite passes
(`test_pmm2d_staggered_wood_list.py`, `test_audit_w3_pmm_jax_guards.py`).  I flag it for the PMM WP
in section 5.  (c) `_wood_mean` refuses (raises `ValueError`) if the two sides retained different
order sets -- possible only if the mirror crossed an order-count boundary, which the `1e-9`
detection threshold makes ~1e-7 away.

### H3 (P2) -- `fff_nv` on `rcwa_jones_2d` broke the cell's own symmetry and had no scope guard

**What was wrong.**  On the Jones entry `fff_nv` means the Li-2003 successive `L2 L1`
factorization, which factorizes x first and y second; the order is not x<->y symmetric, so a
transpose-symmetric cell at normal incidence -- where `Jxx == Jyy` is forced by the cell's own
symmetry -- came back with a difference.  Re-confirmed on HEAD through the new private
`symmetrize=False` switch (`scratchpad/wpa14/h3_before.py`), 96x96 cell, period 0.5 um, depth
0.3 um, `lambda` 0.633 um, eps 6.25 in 2.25:

| cell | M=4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| square (C4) | 9.04e-04 | 4.29e-04 | 2.46e-04 | 1.56e-04 | 1.06e-04 |
| disk (C-inf) | 2.06e-02 | 1.31e-02 | 9.36e-03 | 7.30e-03 | 5.75e-03 |

(The audit's numbers on its own square/disk are 3x smaller -- a slightly different fill fraction --
but the ~1/M scaling and the square-vs-disk ratio reproduce exactly.)  `laurent` and `li` on the
same cells sit at 1e-15..3e-13 throughout.

**What I changed and why.**  The in-plane operator is the symmetric mean
`(L2 L1 + L1 L2) / 2`.  `L1 L2(eps)` is computed by running the same body on the x<->y TRANSPOSED
problem (swap the component labels, transpose the pixel arrays, swap the order-label columns): under
the transpose `T`, `T L2L1(eps) T = L1L2(eps^T)`, and the retained-order ROWS are untouched so the
four blocks come back in the same basis and only their component labels swap back.  The single-order
body is split out as `_li_tensor_l2l1` and is still reachable through `symmetrize=False`, which is
what `RCWAStack._li_blocks` passes -- that call wants the Li-1997 PER-AXIS rule, whose documented
exact reduction to `_li_convolutions_2d` the mean would break.

For the second half of the finding I added `_li_tensor_scope_notice`, which mirrors
`_nv_nonseparable_guard`'s diagnostic (the curved-wall fraction from `_nv_curved_wall_fraction`, run
on the in-plane trace `(exx + eyy)/2`) onto `rcwa_jones_2d`, with a new
`allow_nonseparable_nv=False` kwarg.  It WARNS rather than raising, deliberately: on this entry
`fff_nv` is a different algorithm from the one `rcwa_efficiency_2d` refuses, and its failure mode on
a curved wall is a staircase convergence rate, not the normal-vector method's ~50% absorptance
mis-split.  Raising would have been a gratuitous break of a path that, after the symmetrisation,
tracks `li` to 2.5e-03 at M = 12 on the disk.

**How I verified** (`scratchpad/wpa14/h3.py`):

| cell | M=4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| square, `fff_nv` after | 4.06e-14 | 1.61e-14 | 2.13e-14 | 1.97e-14 | 1.40e-13 |
| disk, `fff_nv` after | 5.38e-15 | 8.63e-14 | 1.31e-13 | 1.73e-13 | 4.52e-14 |

-- the same level as `laurent` (9.75e-15..2.6e-13) and `li` (1.9e-14..2.1e-13) on the same cells.
Separable stripe: `max|J_sym - J_L2L1|` = 8.71e-14 / 4.51e-14 / 1.34e-12 at M = 4/8/12 (the two
orders coincide analytically there, so this is the extra arithmetic's rounding), and the stripe's
GENUINE form birefringence `|Jxx - Jyy| = 1.33` is unchanged.  The scope notice fires once on the
disk (18% diagonal boundary) and not at all on the square; `allow_nonseparable_nv=True` silences it.

**Residual risk.**  The off-plane (full 3x3) `fff_nv` path
(`_li_convolutions_2d_tensor_full`) is NOT symmetrised -- the audit measured only the in-plane 2x2
operator, and the 3x3 transpose argument needs `ezz`/`exz`/`ezx` handled too.  Recorded as deferred
(section 6).  The symmetrisation doubles the (cheap, scalar-pivot) factorization; measured cost is
inside the noise of the eigensolve it feeds.

### H4 (P2, perf) -- three of five items implemented, two deferred

**(a) The even-parity fold now covers every in-plane formulation on `rcwa_jones_2d`.**  It was gated
on `formulation == 'laurent'` and built its own direct-rule operator set, even though the fold acts
on the `(P, Q)` generator and is indifferent to the factorization -- so `'li'` and `'fff_nv'` users
always ran the full `2N` solve (`rcwa_efficiency_2d` had already closed this for `'li'`).  The
operator set is now built once, before the attempt, and reused by whichever path runs; the symmetry
PROBE stays the direct-rule `xx` convolution for every formulation so centre detection is
formulation-independent and the `'laurent'` path is bit-identical (it used the same matrix either
way).  Measured (`scratchpad/wpa14/h4_fold.py`, 5 interleaved medians, 96x96 square cell):

| formulation | M | max\|J_sym - J_full\| | R00 agreement | t_sym | t_full | speedup |
|---|---|---|---|---|---|---|
| laurent | 6 / 9 | 2.23e-13 / 3.38e-13 | 5.7e-14 / 8.3e-14 | 142 / 809 ms | 386 / 2681 ms | 2.71x / 3.31x |
| li | 6 / 9 | 1.08e-13 / 7.36e-14 | 2.5e-14 / 1.5e-14 | 133 / 857 ms | 400 / 2750 ms | **3.01x / 3.21x** |
| fff_nv | 6 / 9 | 1.18e-13 / 2.16e-13 | 2.8e-14 / 4.9e-14 | 163 / 975 ms | 403 / 3070 ms | **2.47x / 3.15x** |

`li` and `fff_nv` were 1.00x before (no fold).  Agreement is inside the documented "~1e-12, not
bit-identical" even-basis contract.  The regression test asserts the fold ENGAGES (by counting
`_symmetric_cascade_rt` and requiring a non-`None` return), not just that the numbers agree -- a
silent fall-through would otherwise pass.

**(b) `_inplane_ops` builds the Li operators once for an isotropic cell.**
`_li_convolutions_2d` computes BOTH `Cxx` and `Cyy` on every call (`Sy` batched inversions of
`(2Mx+1)^3` plus `Sx` of `(2My+1)^3`) and the two-call form discarded half of each.  Guarded on
array identity then value equality, skipped on JAX (which cannot branch on data).  Measured: **2 -> 1**
call for an isotropic cell, still 2 for a genuinely anisotropic one, retained blocks bit-identical
(0.0 / 0.0).  Note for reviewers: `_is_traced` is a SCALAR predicate (it tries `complex(v)`) and is
True for every array -- my first attempt used it as the JAX test and silently disabled the
optimisation; the shipped guard uses the entry point's own `is_jax`.

**(c) BOR symmetric-definite pencils use `eigh(A, M)`.**  `radial_spectrum` formed `M^-1 A`
explicitly and ran a non-symmetric `eig` on a symmetric-definite pair, then truncated a complex
spectrum with `.real`.  Measured (`scratchpad/wpa14/h4_eigh.py`, degree 8 / 12 elements, against
Bessel zeros):

| case | legacy rel. err | `eigh` rel. err |
|---|---|---|
| m = 0 Dirichlet | 3.113e-13 | **1.191e-13** |
| m = 1 Dirichlet | 1.777e-13 | **9.259e-14** |
| m = 3 Dirichlet | 5.418e-14 | **1.377e-14** |
| m = 1 Neumann | 7.041e-13 | **1.741e-13** |
| m = 3 Neumann | 1.831e-13 | **3.808e-14** |

Whole-call medians of 5 interleaved runs: 17.35 vs 17.95 ms (n ~ 97), 51.89 vs 50.44 ms (241),
112.62 vs 190.91 ms (481) -- **1.03x / 0.97x / 1.70x**, not the audit's structural 3-5x, because the
Python element-assembly double loop dominates under n ~ 250.  I report the measured figure.
`return_modes=True` eigenvectors are now `M`-orthonormal (`INT r psi^2 dr = 1`) rather than unit
2-norm; both existing eigenvector gates normalise, and the axis-regularity assertion
(`vecs[0, :] == 0.0` exactly for m != 0) still holds because row 0 is outside `keep`.

**(d, e) Deferred: Levinson for the two Toeplitz inverses; the single-layer two-interface closed
form.**  See section 6, with the measured cost share (each ~8% of a 1-D metallic solve at
`n_orders = 200`, not the ~13% the audit estimated on a slower measurement) and a design.

### H5 (P3) -- the inert BLAS cap

The once-per-process warning existed and was correct but did not say what the inert path costs.  It
now carries the measured numbers (`inv()` of a 163x163 complex matrix 2.29 s unpinned vs 0.0057 s at
one thread, 400x; a 1-D TM solve at `n_orders = 81` 18.2 s vs 0.13 s, 140x) and both remedies
(`pip install threadpoolctl`, or the environment variables before importing numpy).  Confirmed live
in the test output: `test_rcwa.py::test_set_blas_threads_numerically_equivalent` emits it.
`threadpoolctl` is in NEITHER `pyproject.toml` nor `requirements.txt` (the audit said "optional group
only"; it is in no group at all) -- the exact lines are in section 5.

### H6 (P3) -- five items fixed

* **`RCWAResult.per_order_amplitudes()` copies every array.**  `kz` was explicitly copied for the
  writable-array contract, but `Ex`/`Ey`/`kx`/`ky`/`orders` were handed out by reference into the
  result's own modal dict (`to_numpy` is a no-op view on a NumPy backend, which is how the alias
  survived the W7-B pass).  Verified by mutation: `amp[k][...] = 0` for every array key, then a
  second call returns the original arrays bit for bit.
* **A passive-structure one-sided energy bar.**  `_passive_media(eps_sup, eps_sub, *eps)` is True
  when the incidence half-space is exactly lossless, nothing has gain, and every `(3,3)` tensor is
  symmetric -- the conditions under which `R + T <= 1` is a theorem.  `_check_energy` then warns
  (`_EnergyWarning`, so the `stabilize=` ladders treat it as a failed rung, consistent with the
  lossless clause) above a `1e-6` one-sided excess.  Derivation in `_PASSIVE_EXCESS_BAR`: ~7 decades
  above the 1.4e-13 closure clean solves hold in this package, 3.7 decades below the 1.05 tripwire
  that was the only guard before.  The three exclusion clauses are each a measured exception (lossy
  incidence +2.3% at `Im(n_sup) = 0.1`; gain; a non-reciprocal asymmetric tensor) and are gated in
  the test as a truth table.
* **The M8 "~5e-15" claim restated with its rasterisation scope**, carrying the measured
  `O(1/Sx^2)` table (5.16e-04 at Sx = 64 down to 1.26e-07 at 4096) and the ~1e-3 figure at the
  minimum sampling `_validate_cell_sampling` allows.  The ~5e-15 is the y-harmonic claim
  (`N_y = 0` vs `N_y = 1`), not a 1-D/2-D equivalence.
* **EME `strip_x_modes`' dead `np.isrealobj` arm removed** (`A` is built `dtype=complex`
  unconditionally, so the `.real` arm never ran).
* **EME `ref_2d_modes` uses `np.conj(px)` / `np.conj(py)`** for the Bloch wrap -- the rule
  `strip_x_modes` states at `:83-86` and the inconsistency that comment exists to prevent, in its own
  twin.  Bit-identical at `kx0 = ky0 = 0` (where `p = 1`); at non-zero Bloch phase it makes the FD
  oracle EXACTLY Hermitian, as the operator it validates already is.  All 154 EME + radial tests pass.

### G11 (P3) -- `_sqrt_decay`'s on-cut predicate

**What was wrong.**  `flip = (|Re r| <= band * scale) & (Im r < 0)` measures proximity to the
ORIGIN on the SPECTRUM's scale.  A deeply evanescent root satisfies that too once its own magnitude
has collapsed, and flipping one hands a DECAYING mode back with `Re(lam) < 0` -- the
`exp(+|gamma| k0 L)` growth the `Re >= 0` rule exists to prevent.  Re-confirmed on HEAD:
`_sqrt_decay([1e-20 - 1e-30j])` -> `-1e-10 + 5e-21j`.

**What I changed.**  A third conjunct `Im(r)^2 > Re(r)^2` -- "nearer the IMAGINARY axis than the
real one", which is what "on the cut" means and is scale free.  With it, a flipped root has
`|Re r| < |Im r|`, hence `Re(lam^2) < 0`, hence the mode IS propagating: the docstring's
"a flipped mode is by construction a PROPAGATING one" becomes a theorem instead of a census, and
the price bound `|X| - 1 <= exp(band * scale * k0 L) - 1` follows.

**How I verified.**  The counterexample now keeps the principal root; the scale-relative
counterexample (`[big^2, (0.5e-8 big)^2 - 1e-30j]`) no longer flips for spectrum scales 1 / 1e2 /
1e4, so `|X|` at `k0 L = 1.14e7` is 1.0 instead of 1.059 / 302 / 1e248.  Genuine on-cut propagating
modes (`lam^2 = -s + i eta`, `eta` at backward-error level) still flip to the outgoing root, and a
mode whose `eta` puts `|Re r|` above the band still does not.  On the round-2 verification's own
4,010-value fixture the new conjunct drops 14 of 37 flips at the one array size where the band fires
(n = 243); the CLOSEST dropped entry has `|Re r| / |Im r| = 1.251` (evanescent side of the
45-degree line) while the closest KEPT entry has `|Im r| / |Re r| = 1.007` -- so nothing on the cut
was lost.  On the populations `_CUT_BAND_REL` was derived against, the worst `|Re r| / |r|` ever
flipped is 2.0751e-03, five decades clear of the new edge, so the conjunct is inert there:
`test_fix_branch_cut_round2.py`, `test_verify_branch_cut_round2.py`,
`test_fix_rcwa_even_sector_wsl.py`, `test_verify_rcwa_even_sector.py` all pass.

JAX parity: the flip SETS are identical between the NumPy and `jax.numpy` bodies over 503 values
(40 flips each); the values differ by 8.04e-14, which is XLA's `sqrt` differing from NumPy's in the
last bits and is pre-existing (the same 8.04e-14 separates the raw `sqrt`s).  `jit` and `grad` both
still work -- the predicate is a piecewise-constant boolean multiplying by a real +/-1, exactly as
before.  CuPy is not installed on this machine; the change is backend-generic
(`xp.imag`/`xp.real` arithmetic only, no host branch) and was desk-checked against
`array_namespace` dispatch.

**A test that pinned the defect.**
`test_fix_branch_cut_round2.py::test_the_shared_body_keeps_the_round_one_SELECTOR_and_changes_only_the_value`
asserted, in its own words, that "THE SELECTOR IS UNCHANGED ... decided by exactly the round-1
predicate" -- i.e. it pinned exactly the predicate G11 says is wrong, and it failed on my change
(37 vs 23 flips at n = 243).  I restated it as
`test_the_shared_body_narrows_the_round_one_SELECTOR_to_the_cut` with three strictly stronger
claims: (a1) the flip set is exactly the published round-4 predicate; (a2) it is a SUBSET of the
round-1 selector and every dropped entry has `Re(r)^2 >= Im(r)^2` (so nothing on the cut was lost,
and the round-1/round-2 population measurements carry forward as an upper bound); (b) where both
fire the values differ by exactly `2 Re(r)`, and off the round-4 flip set the body is the principal
root bit for bit.  A final assertion fails if the new conjunct ever stops dropping anything on that
fixture, so (a2) cannot go vacuous.

---

## 3. Files touched

Source (all within WP-A14 ownership):

* `lumenairy/elements/rcwa/_core.py` -- G11 predicate + docstring; `WoodNudgeWarning`,
  `_grazing_safe_wavelength` (warning + `fn_name`/`warn` kwargs, numerics unchanged),
  `_grazing_safe_wavelength_pair`, `_WoodAnomaly`, `_wood_signature`, `_wood_mean_leaf`,
  `_wood_mean`, `_wood_symmetric`; `Efficiency2D.wl_eff`; `_passive_media`,
  `_PASSIVE_EXCESS_BAR`, `_check_energy(passive=)`; the M8 docstring; the H5 warning text;
  `__all__` additions.
* `lumenairy/elements/rcwa/oned.py` -- H2 on `rcwa_efficiency_1d`, `rcwa_jones_1d`,
  `rcwa_jones_1d_segments`; H6 `passive=` at the 1-D energy guard.
* `lumenairy/elements/rcwa/twod.py` -- H2 on `rcwa_efficiency_2d`, `RCWA2DPrepared.solve`,
  `rcwa_jones_2d`, `rcwa_efficiency_2d_shapes`, plus `wl_eff` on all three `Efficiency2D`
  constructions; H3 symmetrised `_li_convolutions_2d_tensor` + `_li_tensor_l2l1` +
  `_li_tensor_scope_notice` + `allow_nonseparable_nv` on `rcwa_jones_2d`; H4 `_inplane_ops`
  single-call and the generalized even-parity fold; H6 `passive=` at four energy guards;
  docstrings.
* `lumenairy/elements/rcwa/stack.py` -- H2 on `RCWAStack._solve_once` + `RCWAResult.wl_eff` +
  `RCWAResult._wood_mean_with`; H3 `symmetrize=False` in `_li_blocks`; H6
  `per_order_amplitudes` copies and `_stack_passive`.
* `lumenairy/elements/bor/coupled_radial_eigensolver.py` -- H1.
* `lumenairy/elements/bor/radial_eigensolver.py` -- H4 `eigh(A, M)`.
* `lumenairy/elements/eme/eme_2d.py` -- H6 dead branch + `conj(px)`/`conj(py)`.

Tests:

* NEW `tests/unit/test_audit2609_a14_rcwa_eme_bor.py` (32 tests, incl. an embedded independent
  direct-boundary-match 1-D RCWA oracle).
* `tests/unit/test_fix_branch_cut_round2.py` -- restated the G11-pinning test (see above).
* `tests/unit/test_niche_audit_m4_m5_m6_rcwa.py`,
  `tests/unit/test_audit_s1_2_rcwa_lossless_tripwire.py` -- their `_check_energy` stubs had a fixed
  signature and broke on the new `passive=` kwarg; both now take `**kw` and the lossless-tripwire
  one RECORDS what it forwards, so a future guard clause extends the gate instead of breaking it.

Docs:

* `docs/audits/.../fixes/WP-A14_REPORT.md` (this file), `.../fixes/WP-A14_CHANGELOG.md`.

No edits to `CHANGELOG.md`, `README.md`, `Migration-Guide.md`, `CONVENTIONS.md`, `pyproject.toml`,
`lumenairy/__init__.py`, `pmm/`, or any propagator/lens module.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`,
`-q --no-header -p no:cacheprovider`.

| command (test files) | result | duration |
|---|---|---|
| `tests/unit/test_audit2609_a14_rcwa_eme_bor.py` (new) | **32 passed** | 30 s |
| `test_rcwa.py test_v5_20_12_rcwa_jones_2d_fff_nv.py test_v5_20_6_rcwa_jones_2d_li.py test_v5_11_0_rcwa_fff_nv_2d.py test_v5_6_1_rcwa_symmetry.py` | **133 passed** | 622 s |
| `test_radial_eigensolver.py test_eme_2d.py test_eme_2d_vector.py test_eme_diffraction.py test_eme_census_determinacy.py test_eme_jax_modes.py test_fix_eme_branch_cut.py test_audit_w6_eme.py test_niche_audit_w6_eme.py` | **154 passed** | 1008 s |
| `test_fix_branch_cut_round2.py test_verify_branch_cut_round2.py test_fix_rcwa_even_sector_wsl.py test_verify_rcwa_even_sector.py test_v5_11_0_rcwa_internal_field.py test_niche_audit_w7_rcwa.py test_v5_21_delta_audit.py test_v5_14_1_rcwa_audit_fixes.py` | 1 failed -> **fixed** -> 159 passed | 127 s + 69 s |
| `test_niche_audit_m4_m5_m6_rcwa.py ... test_audit_w6_bor.py` (13 files incl. the PMM wood-list and PMM JAX guards) | 2 failed -> **fixed** -> 180 passed | 50 s + 4 s |
| `test_v5_10_3_rcwa_2d_autodiff.py test_v5_20_1_rcwa_2d_oop_jax.py test_v5_20_3_rcwa_1d_oop_jax.py test_bor_sem_jax.py test_v5_20_11_bor_jax.py` (JAX) | **29 passed** | 168 s |
| `python validation/run_all.py test_rcwa --quiet` | **PASS** | 2.5 s |
| BOR batch (`test_coupled_eigensolver.py test_niche_audit_w6_bor.py test_bor_sem.py test_bor_solve.py test_audit_bor_grazing_cutoff.py test_fix_bor_multilayer_guards.py test_verify_bor_multilayer_guards.py test_fix_bor_guards_round2.py test_verify_bor_guards_round2.py test_verify_bor_guards_round3.py`) | **237 passed, 1 xfailed** | 1835 s |
| RCWA batch 2 (`test_v5_13_0_rcwa2d_sweep.py test_v5_11_0_rcwa_stack_stabilize.py test_v5_10_0_rcwa_builders.py test_v5_9_0_rcwa_quickwins.py test_v5_20_8_rcwa_threaded_sweep.py test_doe_rcwa.py test_v5_11_0_rcwa_segments.py test_v5_7_0_rcwa_asr.py test_v5_6_rcwa_convergence.py test_v5_14_1_rcwa_deferred.py test_v5_11_1_rcwa_lowerpri.py test_v5_11_0_rcwa_full_tensor.py test_v5_11_0_rcwa_device_helpers.py`) | **226 passed, 2 skipped** | 649 s |
| after narrowing the Wood bracket to 1e-9: `test_rcwa.py test_v5_11_0_rcwa_fff_nv_2d.py test_v5_21_delta_audit.py test_pmm2d_staggered_wood_list.py test_audit_w3_pmm_jax_guards.py` | **156 passed** | 466 s |
| `repro/RCWA-EME-BOR/p02_oracle.py` (24 configurations) and `p01_energy.py` (12) | agreement <= 1.5e-13 off-anomaly; closure <= 1.03e-13 | 40 s |
| FINAL consolidated re-run after narrowing the bracket (`test_audit2609_a14_rcwa_eme_bor.py test_v5_13_0_rcwa2d_sweep.py test_v5_11_0_rcwa_stack_stabilize.py test_v5_11_0_rcwa_internal_field.py test_v5_20_12_rcwa_jones_2d_fff_nv.py test_v5_20_6_rcwa_jones_2d_li.py test_v5_6_1_rcwa_symmetry.py test_niche_audit_w7_rcwa.py test_v5_14_1_rcwa_audit_fixes.py test_fix_branch_cut_round2.py test_verify_branch_cut_round2.py`) | **194 passed** | 303 s |

**Three failures found, all fixed, none pre-existing and unrelated:**

1. `test_fix_branch_cut_round2.py::test_the_shared_body_keeps_the_round_one_SELECTOR_and_changes_only_the_value`
   -- a test that PINNED the G11 defect (it asserted the round-1 selector was unchanged).  Restated
   with three stronger claims; see the G11 section.
2. `test_niche_audit_m4_m5_m6_rcwa.py::test_m5_laurent_stabilize_still_burns_the_ladder_on_closure`
   and 3. `test_audit_s1_2_rcwa_lossless_tripwire.py::test_rcwastack_solve_forwards_lossless_flag`
   -- both monkeypatch `_check_energy` with a fixed `(fn_name, R, T, lossless=False)` signature, so
   the new `passive=` kwarg raised `TypeError`.  Both stubs now take `**kw`; the lossless-tripwire
   one also records what it forwards, which strengthens it.

The BOR batch's single `xfailed` is the pre-existing, documented
`test_verify_bor_guards_round3.py::test_the_liner_verdict_agrees_at_all_three_positions_at_a_second_rbig`
(VERIFY round 3, D-V1 -- the `_BOR_FRAC_DEADBAND` width question), unrelated to this WP.
**No other pre-existing failures were observed** in any file I ran.  Two environment notes: the library
was briefly un-importable mid-session because another agent had `pmm/twod.py` in an intermediate
state (`AttributeError: module ... has no attribute 'pmm_2d_order_drift'`); it resolved on its own
and is not related to this WP.  `threadpoolctl` is not installed here, so the H5 warning fires in
the suite (by design) and `test_niche_audit_m4_m5_m6_rcwa.py` skips its "cap is effective" branch.

---

## 5. Requested changes outside my ownership

**(a) `pyproject.toml` -- declare `threadpoolctl` as a HARD dependency (H5).**  Requested from the
tests/CI work package.  Exact change, at `pyproject.toml:81-86`:

```toml
dependencies = [
    "numpy>=2.0",
    "scipy>=1.13",
    "matplotlib>=3.5",
    "psutil>=5.0",
    # v5.45.2 (audit H5): threadpoolctl is what makes set_blas_threads /
    # rcwa_blas_threads / the @_with_blas_limit wrapper on every public RCWA
    # entry point ACTUALLY apply a cap.  Without it they are inert (they warn),
    # and on an oversubscribed many-core box that is catastrophic, not a
    # micro-optimisation: MEASURED on a 24-thread Windows OpenBLAS 0.3.31
    # build, inv() of a 163x163 complex matrix takes 2.29 s unpinned against
    # 0.0057 s at one thread (400x), and a 1-D TM RCWA solve at n_orders=81
    # takes 18.2 s instead of 0.13 s (140x).  It is tiny and pure Python (and
    # is already a transitive dependency of scikit-learn, so most environments
    # have it), so the library's own remedy should work out of the box.
    "threadpoolctl>=3.1",
]
```

and the mirror in `requirements.txt`, in the "Required" block after `psutil>=5.0`:

```
threadpoolctl>=3.1        # makes set_blas_threads / rcwa_blas_threads effective (inert without it: 400x on a 24-thread OpenBLAS box)
```

Why `>=3.1`: `ThreadpoolController` (the cached-enumeration path `_get_blas_controller` prefers)
landed in threadpoolctl 3.0, and 3.1 is the first release with the Windows OpenBLAS 0.3.x detection
this box needs.  `_core.py` already degrades correctly to the legacy `threadpool_limits` and to a
no-op, so nothing breaks if the floor is set lower.

**(b) Heads-up for the PMM work package (H2, no action required unless you disagree).**
`_grazing_safe_wavelength` lives in `rcwa/_core.py` (mine) and is imported by 12 sites in
`elements/pmm/`.  Its NUMERICS are unchanged (same `1e-7` step, same `1e-9` detection threshold);
the only change PMM sees is that it now emits a `WoodNudgeWarning` whenever it nudges -- and the
PMM suites I ran
(`test_pmm2d_staggered_wood_list.py`, `test_audit_w3_pmm_jax_guards.py`,
`test_niche_audit_w3_rcwa_pmm.py`, `test_audit_w5_pmm_rcwa.py`, `test_audit_w6_pmm_rcwa.py`) pass.
I deliberately did NOT shrink the SHARED step, even though a narrower shift is worth 10x to RCWA,
because the staggered PMM engine degrades like `1/sqrt(distance)` TOWARD a cut-off and a narrower
step would make it worse.  The RCWA side takes its 10x through its own
`_WOOD_PAIR_STEP_REL = 1e-9`, which only `_grazing_safe_wavelength_pair` reads.  If PMM would prefer
a different step of its own, `_grazing_safe_wavelength` now takes `fn_name` and `warn` kwargs and a
per-caller step would be a one-line addition -- and the measurement that would justify it is the
one in `_WOOD_PAIR_STEP_REL`'s note (conditioning is clean down to a 1e-13 shift on the RCWA side;
PMM's own floor has not been measured).

**(c) `CONVENTIONS.md` Section 11 (`formulation='fff_nv'` is entry-point-specific).**  Worth one
sentence recording that `rcwa_jones_2d(formulation='fff_nv')` now emits a validated-scope warning on
a curved cell (the Jones counterpart of `rcwa_efficiency_2d`'s refusal) and takes
`allow_nonseparable_nv`.  Not my file.

---

## 6. Deferred items (with designs)

**D1 -- H4: Levinson / Gohberg-Semencul for the two genuinely Toeplitz inverses.**
Sites: `oned.py:131` (`inv(Toeplitz(1/eps))` in `_binary_grating_convolutions`) and `oned.py:652`
(`inv(EPS)` for the P block).  Measured cost share on this machine (`scratchpad/wpa14/h4_perf.py`,
1-D TM Ag grating, `formulation='li'`): at `n_orders = 200` (N = 401) the whole solve is **1.081 s**,
of which 6 `inv` calls total **0.259 s**, 4 `solve` calls 0.070 s and 1 `eig` 0.206 s.  The two
Toeplitz inverses are ~2/6 of the inverse time, **~0.086 s = 8%** of the solve -- a third of the
audit's 0.29 s / 13% estimate, which was taken on a slower measurement.

Design: two Levinson-Durbin recursions (forward and backward Yule-Walker solves) give the first and
last columns of the inverse; Gohberg-Semencul then expresses `T^-1` as a difference of two products
of triangular Toeplitz matrices, each applicable in `O(N log N)` by FFT or `O(N^2)` directly.

Why deferred: classical Levinson is only WEAKLY stable, and provably so only for Hermitian positive
definite Toeplitz.  `[[1/eps]]` for a metallic grating is a general complex Toeplitz (for a lossy
`eps` the coefficients do not satisfy `c_{-k} = conj(c_k)`) and is exactly the matrix the M1
conditioning census found reaching `cond ~1e13`.  The verification bar here is "bit-identical or
documented tolerance", and a documented tolerance on this matrix would need its own two-sided census
across the metallic convergence ladder -- comfortably more work than the 8% is worth in this pass.
Effort estimate: 1-1.5 days including the census.  A cheaper 80% of the win, if someone wants it:
`scipy.linalg.solve_toeplitz` for the two places where the inverse is immediately MULTIPLIED by
something (which is most of them), avoiding the explicit inverse entirely.

**D2 -- H4: a direct two-interface closed form for a single-layer 1-D stack.**  The
`interface -> propagation -> interface` chain ends in a `_redheffer_star` whose zero-block fast path
(`_core.py:2748-2756`) cannot fire because both `A22` and `B11` are non-zero, so it pays two
`_guarded_inverse` calls.  Measured share ~0.086 s of 1.081 s at N = 401, again ~8%.
Design: Moharam-1995b enhanced transmittance, or the algebraic closed form for
`S_a * P * S_b` in which the only inverse is `(I - B11 A22)^-1` -- one inverse instead of three.
Why deferred: `_guarded_inverse` carries the M1 conditioning census, its residual probe and its
refusal path, and a closed form would have to reproduce all three or lose the guard on the default
path of every single-layer solve.  Effort estimate: 1 day including a bit-tolerance sweep over the
metallic ladder.

**D3 -- H3: symmetrise the OFF-PLANE (full 3x3) `fff_nv` operator.**
`_li_convolutions_2d_tensor_full` still uses the fixed `L2 L1` order.  The audit measured only the
in-plane 2x2 path, and the in-plane fix leaves the off-plane path exactly as it was.  Design: the
same transpose argument extends, but the 3x3 transpose must permute `(x, y, z) -> (y, x, z)` and swap
`exz <-> eyz`, `ezx <-> ezy`, leaving `ezz` alone; the `l3-` Schur fold then has to be applied after
the mean, not before, since the mean of two Schur complements is not the Schur complement of the
mean.  Needs a conical Berreman oracle on a ROTATED-director cell to gate it.  Effort: 0.5 day plus
the oracle.

**D4 -- H6: one `lumenairy/_branchcut.py` carrying the band SHAPE with per-caller scales.**
Three independent bands share the `1e-8` constant with different scales
(`rcwa._core._CUT_BAND_REL`, `eme._branch._EME_CUT_BAND_REL`, `pmm._core._forward_branch_flip`'s),
plus `bor._orient.orient_band_scale`.  The brief's condition is "if you can prove bit-identity",
and two of the four live in `pmm/`, which is another WP's ownership this round.  Design: a
`band_mask(r, *, scale, band)` returning the boolean, with each caller keeping its own derived
constant and its own documented population; the three call sites then differ only in the scale they
pass.  Bit-identity is provable per site (the expression is the same), but it must be checked
against all four populations on both builds.  Effort: 0.5 day, best done in a pass that owns
`pmm/` too.

**D5 -- H6: one forward/backward mode selector.**  `_select_forward_flux` (rcwa),
`_strip_split_forward` (eme vector), `forward_decaying_root` (eme scalar + diffraction) and
`bor._orient.forward_orient` all implement "flux sign, falling back to decay sign, with a relative
band".  Two of the four are mine; consolidating only those would leave the duplication that matters.
Same recommendation as D4: do it in a pass that owns all four.  Effort: 1 day.

**D6 -- code organization (audit's "observations", not a finding).**  `_core.py` is 4,415 lines with
88 `__all__` names of which 83 are private; the `STAGGERED_WALL_ANCHOR` constant carries ~100 lines
of measurement prose that belongs in `docs/audits/`.  Both are worth doing and neither is safe to do
in the same pass as a behaviour change this size -- a file split makes every other WP's diff
unreviewable.  Recommend a dedicated, test-only-diff commit after this remediation lands.

**Not-reproducible: none.**  Every finding assigned to WP-A14 reproduced on the current HEAD.

---

## 7. Changelog

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A14_CHANGELOG.md`
