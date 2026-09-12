# WP-A13 — PMM 2-D (hybrid, staggered, stacks, JAX twins) — remediation report

Findings **G5–G13** of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md`
§12, partition report
`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/PMM-2D.md`.
Branch `audit-fixes-2026-09`.  Every number below was re-measured on this
machine (Windows 11, CPython 3.14, numpy 2.4.6, jax 0.10.1 CPU,
`OPENBLAS_NUM_THREADS=1`), before and after, against the audit's own repro
scripts where they exist.

---

## 1. Summary

| ID | Status | Files:sites | Tests | Oracle | Measured before → after |
|---|---|---|---|---|---|
| **G5** (P1) | **fixed** | `pmm/stack2d.py:_symmetric_layer_specs` (li branch), `:_build_layer_modes` (scalar branch); `pmm/_jax_stack2d.py:_modes_projected` + its concrete-scalar call; `pmm/twod.py:_layer_modes_projected` docstring | `test_audit2609_a13_stack2d.py::test_g5_li_is_90deg_rotation_invariant_on_the_stack` (×6), `::test_g5_one_layer_stack_equals_the_single_cell_entry` (×2), `::test_g5_the_routed_pair_is_what_the_stack_now_passes`, `test_audit2609_a13_jax.py::test_g5_jax_stack_twin_routes_the_per_slot_li_operators`, `::test_g5_jax_stack_twin_matches_the_numpy_stack` (×2) | 90° rotation invariance of one physical grating (an exact symmetry of Maxwell's equations, not produced by the code); `pmm_efficiency_2d_cell` as the documented same-physics sibling | `max|ΔT|` 4.977e-03 → **3.39e-14**; T00 Δ 1.231e-03 (5.3 % of T00) → **0.0**; `|ΔJ|` 2.318e-02 → **7.81e-14**; stack vs cell entry 1.23e-03 → 3.8e-14 |
| **G6** (P2) | **fixed** (default deliberately kept — see §2.2) | `pmm/twod_jones.py` module docstring, `formulation` docstring + `'auto'` resolution, fold gate `formulation != "fff_nv"` removed | `test_audit2609_a13_twod.py::test_g6_fff_nv_beats_laurent_beats_li_on_a_separable_cell`, `::test_g6_auto_picks_fff_nv_on_separable_and_laurent_otherwise`, `::test_g6_fff_nv_takes_the_even_parity_fold` | `rcwa_jones_1d(n_orders=80, 'li')` — an independent 1-D engine | `\|Jxx−ref\|` at n_orders 11: fff_nv 2.72e-03 / laurent 3.02e-02 / li **6.90e-02** (ranking reproduced); fff_nv fold now runs: `ΔJ` 1.49e-12, **3.96×** (7.24 s → 1.83 s) |
| **G7** (P2) | **fixed** (advisory + drift check; `_PASSIVE_TOL_2D` deliberately NOT tightened — see §2.3) | `pmm/twod.py`: new `pmm_2d_order_drift`, `_ADVISORY_TOL_2D`, `energy_tol=` on both scalar entries, `_warn_lossless_energy_2d(tol=)`; docstring pointers in `twod_jones.py` | `test_audit2609_a13_twod.py::test_g7_closure_improves_while_the_per_order_split_moves`, `::test_g7_a_converged_truncation_passes_the_drift_check_silently`, `::test_g7_energy_tol_can_tighten_the_tripwire` | the physical requirement that the answer stop moving with truncation | closure 9.69e-03 → 3.63e-03 → **4.35e-04** (improving) while T00 0.2715167 → 0.2373950 → **0.1542801** (−35 %) at n_orders 5/9/11 — now caught: `max_drift` 8.31e-02 vs the 1e-2 bar, `converged=False` |
| **G8** (P2) | **fixed** | `pmm/twod.py`: `PreparedPMM2D.__slots__`/`solve`/docstring, `_prepare_pmm2d_core`, `prepare_pmm_2d`, `prepare_pmm_2d_cell`, `pmm_efficiency_2d_vs_wavelength`, `pmm_efficiency_2d_cell_vs_wavelength` | `test_audit2609_a13_twod.py::test_g8_prepared_is_bit_identical_to_the_direct_entry` (×4), `::test_g8_the_sweep_path_runs_the_lossless_tripwire`, `::test_g8_truncation_reaches_the_prepared_and_sweep_paths` | `pmm_efficiency_2d[_cell]` itself, at matching settings | prepared vs direct at the DEFAULT `symmetry='auto'`: `max|dR|` 6.33e-14 / `max|dT|` 1.08e-13 → **0.000e+00** (bit-identical); tripwire 0 warnings → 1 on both the prepared and the sweep path at the identical `sum(R+T) = 1.054125`; `truncation='circular'` `TypeError` → 29 retained orders matching the direct entry bit-for-bit |
| **G9** (P2) | **fixed** | `pmm/twod_staggered.py`: `_MAX_STAG_PENCIL_DOF`, `_stag_merged_segments`, `_validate_stag_cost`, `max_pencil_dof=` on `pmm_efficiency_2d_staggered` / `pmm_jones_2d_staggered` + docstrings; `pmm/stack2d_pure.py`: `add_layer(max_pencil_dof=)` + the guard on both the plain and magnetic branches + docstring; `pmm/twod.py`: PIXEL-vs-SEGMENT cross-references | `test_audit2609_a13_staggered_cost.py` (8 tests, 0.12 s) | `_cell_to_walls_tile`'s own merge rule, applied to the same array | the audit's two calls (498 s / 8.4 GB and 1096 s / 3.9 GB) produced **no raise, no warning, nothing**; now a redundancy warning naming `12×12 segments and only 3×3 DISTINCT strips … expressible on the uniform 4×4 lattice … 729× less QZ time, 81× less memory`, plus an absolute `max_pencil_dof` refusal |
| **G10** (P2, perf) | **fixed** except the tensor-`lops` cache (deferred, §6.1) and the `cascade` default (measured, NOT adopted, §2.6) | `pmm/_jax_twod.py` (`_axis_projector_pair`, `_proj_sandwich_jnp`, `_static_prep`, `_static_prep_cell`, `_scalar_jax_tail`), `pmm/_jax_twod_jones.py::_proj`, `pmm/_jax_stack2d.py` (`_layer_static_traced`, traced branch), `pmm/twod.py::_axis_ops_1d`, `pmm/twod_jones.py` separable `_mass`, `pmm/twod_jones.py` + `pmm/stack2d.py` circular truncation, `pmm/stack2d.py::_warn_eig_cache_refusals` | `test_audit2609_a13_jax.py::test_g10_*` (5), `test_audit2609_a13_twod.py::test_g10_*` (4), `test_audit2609_a13_stack2d.py::test_g10_*` (3) | the pre-fix bodies, written out in the tests as bit-identity oracles; `_scalar_projected_ops` for the JAX constants | dense Kronecker pair 7.2 / 22.2 / 43.2 MiB at 12/16/32 strips-per-axis → **0** (2.4 GiB at the documented ceiling); JAX/NumPy `T` parity 8.47e-11 → **1.63e-13**; `_axis_ops_1d` **3.7×**, separable `_mass` **4.8×**, both bit-identical; circular truncation 121 → 81 orders, 0.216 s → 0.068 s (stack) and 0.200 s → 0.082 s (`pmm_jones_2d`, `symmetry=False`); eig-cache refusal: silent → one warning per instance |
| **G11** (P3) | **landed in WP-A14** (which owns `rcwa/_core.py`); PMM-2D-side contract pinned here | — (no source change of mine) | `test_audit2609_a13_twod.py::test_g11_no_layer_mode_grows_beyond_the_branch_cut_band` | the fixed predicate on the audit's counterexample, AND the analytic bound `\|X\| ≤ exp(band·max\|λ\|·k0·L)` | `_sqrt_decay([1e-20 − 1e-30j])` **−1e-10 + 5e-21j → +1e-10 − 5e-21j** (the evanescent mode is no longer flipped); the PMM-2D-side bound is `1 + 5.6e-07` in this regime and holds either way |
| **G12** (P3) | **fixed** | `pmm/stack2d.py`: `_mode_key` (+ `symmetry`, + `truncation`), `period_x`/`period_y` properties + `_set_period`, class docstring Notes | `test_audit2609_a13_stack2d.py::test_g12_symmetry_is_in_the_modal_cache_key`, `::test_g12_periods_are_frozen_after_the_first_layer` | a FRESH object at the new setting | `symmetry` mutation: reused answer BIT-IDENTICAL to the pre-mutation one (`|J_b − J_a| = 0.0`) → now equals the fresh object to 2.2e-16 and differs from the stale one; period mutation `|ΔJ| = 4.390e-01` → raises |
| **G13** (P3) | **fixed** | `pmm/twod_jones.py`: `return_jones_transmission=`, `_pmm_jones_2d_at` 5-tuple + slant frame anchor, Returns/Notes seam note; `pmm/twod.py::_assemble_2d` reference note; dead `noqa` re-exports deleted in `_jax_twod.py` / `_jax_twod_jones.py` | `test_audit2609_a13_twod.py::test_g13_transmission_jones_matches_the_stack_and_the_qwp_reference`, `::test_g13_the_four_tuple_contract_is_unchanged`, `::test_g13_the_transmission_jones_carries_the_slant_frame_anchor` | `rcwa_jones_1d(n_orders=60, 'li')` retardance **+100.066°**; `PMM2DStackHybrid.jones_transmission()` | no transmission Jones at all → +100.12 / +100.23 / +99.90° at n_orders 5/9/15 (`fff_nv`), i.e. inside **0.17°**; bit-identical to the stack accessor incl. the slant anchor |

CONVENTIONS §7.1's "the 1-D solvers return te/tm" wording (the tail of G13) is
in `CONVENTIONS.md`, which COMMON.md forbids me to edit — requested in §5.

---

## 2. Per finding

### 2.1 G5 (P1) — `PMM2DStackHybrid` never routed the per-slot Li operators

**What was wrong.** Audit P3-33 split the hybrid's `formulation='li'` eps
operator into a per-slot pair: `EpnxF` (the inverse rule along x) on the `Ex`
slot and `EpnyF` on the `Ey` slot, so that a **y**-patterned cell gets the
inverse rule on its wall-normal component.  `twod._pmm2d_solve_core` passes
them.  `PMM2DStackHybrid` did not — both `_symmetric_layer_specs` (the
even-parity fold) and `_build_layer_modes` (the full solve) kept the legacy
`(Ex ← EpnF, Ey ← EpsF)` assignment.  For a y-patterned layer
`_scalar_projected_ops` returns `EpnF = kron(Ty @ Epn @ Typ, Ix)` — the
**y**-axis inverse-rule operator — and the legacy line put it on the **Ex**
slot while `Ey` kept Laurent, i.e. BOTH slots anti-Li: exactly the case
`twod.py`'s own branch comment describes as fixed.

The **JAX twin carried the same defect**: `_jax_stack2d._modes_projected`
spelled the identical legacy assignment, so the differentiable stack broke the
same symmetry (COMMON rule 9).

**What I changed.**  Two lines mirroring `twod.py`'s solve core — `EPS_nx,
EPS_ny = lops["EpnxF"], lops["EpnyF"]` under `li` in `_symmetric_layer_specs`,
and `EpnxF=lops["EpnxF"], EpnyF=lops["EpnyF"]` on the `_layer_modes_projected`
call in `_build_layer_modes` — plus the jnp twin's `_modes_projected`, which
now takes the same optional pair and defaults to the legacy assignment for the
traced branch (a traced cell is patterned on BOTH axes, where the two
assignments coincide because no per-axis Fourier rule exists in that
representation).  `_layer_modes_projected`'s docstring, which named the gap,
is corrected.  No cache-key change was needed.

**How I verified.**  The audit's own repro scripts, re-run before and after:

* `repro/PMM-2D/q1b_stackli.py` — before: x-pat/y-pat T00 `0.0231919632` vs
  `0.0229566480` (n_orders 5) and `0.0334736588` vs `0.0322424723`
  (n_orders 9); after: **identical at every n_orders 3/5/7/9** (both
  formulations).
* `repro/PMM-2D/q1c.py` — before, the stack's y-orientation missed the
  single-cell entry by 1.23e-03; after, all four columns read
  `0.02319196` (n = 5) and `0.03347366` (n = 9).
* `repro/PMM-2D/p3b.py` — before `max|ΔT| = 4.977e-03`, `|ΔJ| = 2.318e-02`;
  after `3.385e-14` / `7.805e-14` (`symmetry='auto'`) and `1.139e-13` /
  `2.337e-13` (`symmetry=False`), against `laurent`'s unchanged 1.07e-14.

My own grid (degree 7/11 × n_orders 3/5/9 × symmetry auto/False): worst
`max|ΔT|` **1.14e-13**, worst `|ΔJ|` **1.51e-12**.  JAX twin: NumPy parity
`max|dR| ≤ 2.4e-14` / `max|dT| ≤ 2.5e-14` / `max|dJ| ≤ 1.0e-13` on BOTH
orientations, and the twin's own two-orientation residual is 3.06e-15.
`formulation='laurent'` is bit-for-bit unchanged (control: T00
`0.022958587327` before and after).

**Residual risk.**  The doubly-patterned branch is untouched by design — it has
no per-axis Fourier rule, so `EpnxF`/`EpnyF` there are `EpnF`/`EpsF` and the
historical assignment is preserved byte for byte.  Users who tuned a design
against the old y-patterned answer will see it move (by up to 5.3 % of T00);
the changelog carries the migration note.

### 2.2 G6 (P2) — the `formulation` docstring, `'auto'`, and the `fff_nv` fold

**What was wrong.**  On the Jones entry `formulation` selects only the `E_z`
elimination rule; the in-plane block is Laurent either way.  The docstring
presented `'li'` as "the hybrid's *validated* inverse-rule elimination", which
reads as "at least as good", when it is measurably the worst of the three and
its error is in the PHASE.  Separately, the even-parity fold was gated off for
`fff_nv` alone, so the best formulation was also the slowest.

**What I changed.**  (a) The `formulation` docstring and the module docstring
now carry the measurement (table below) and the ordering `fff_nv > laurent >
li`.  (b) `formulation='auto'` resolves to `'fff_nv'` on a SEPARABLE in-plane
cell and `'laurent'` otherwise (on the JAX path always `'laurent'`, since
`'fff_nv'` is NumPy only).  (c) The `formulation != "fff_nv"` fold gate is
removed.

**Why the DEFAULT is unchanged.**  The WP allows the switch only "if all
existing tests still pass".  They do not, and the reason is a documented
contract, not an incidental fixture: `pmm_jones_2d`'s module docstring and
`formulation='laurent'` entry both promise that a scalar cell reduces
**EXACTLY** to `pmm_efficiency_2d_cell(formulation='laurent')`, and
`tests/unit/test_v5_14_0_pmm_jones_2d.py::test_scalar_tensor_reduces_to_scalar_laurent`
pins it at 1e-11 using the DEFAULT formulation.  An auto-`fff_nv` default would
make that contract false for exactly the separable cells the LC-QWP work uses
(the whole point of `fff_nv` is that it differs from Laurent), and silently.
So the default stays and `'auto'` is the documented, additive, recommended
opt-in.

**How I verified.**  Audit fixture `repro/PMM-2D/q3_fffnv.py` (separable Si
stripe, eps 12.25, duty 1/2, Λ = 0.47 µm, λ = 1 µm, d = 0.3 µm, n_sub = 1.5,
degree 11), `|Jxx − rcwa_jones_1d(n_orders=80, 'li')|`:

| n_orders | `fff_nv` | `laurent` | `li` | `rcwa_jones_2d li` |
|---|---|---|---|---|
| 3 | 5.99e-02 | 1.92e-01 | 3.51e-01 | 3.58e-02 |
| 5 | 4.91e-02 | 1.38e-01 | 1.81e-01 | 7.31e-03 |
| 9 | 1.88e-02 | 4.20e-02 | 1.14e-01 | 2.69e-04 |
| 11 | **2.72e-03** | 3.02e-02 | **6.90e-02** | 2.95e-04 |

— the audit's ranking reproduced (it reported 2.7e-03 / 2.90e-02 / 6.85e-02 /
2.7e-04 at n = 11).  The fold, now enabled for all three rules on a centred
stripe at degree 11 / n_orders 11: `max|ΔJ(fold − full)|` = **1.485e-12**
(`fff_nv`), 4.915e-13 (`laurent`), 5.023e-13 (`li`) — the same decade — for
**3.96×** / 3.89× / 3.80× wall time.  QWP retardance by formulation
(Λ/λ = 0.2, reference +100.066°): `fff_nv` +0.056 / +0.167 / −0.162° at
n_orders 5/9/15 against `laurent` −5.374 / −1.129 / −0.433° and `li` −3.337 /
−0.030 / −0.027°.

**Residual risk.**  `'auto'` changes nothing by default.  Enabling the fold for
`fff_nv` moves `fff_nv` answers at normal incidence on a centro-symmetric cell
by ~1e-12 (the fold's documented level), which is the same treatment the other
two rules have had since audit F2.

### 2.3 G7 (P2) — closure is not a convergence proof

**What was wrong.**  The hybrid's lossless closure plateaus at 1e-4…1e-2, is
non-monotone in `n_orders`, sits far inside `_PASSIVE_TOL_2D = 5e-2` — and on
one fixture moves OPPOSITE to the per-order error it stands in for.

**What I changed.**  (a) `pmm_2d_order_drift(solve_at_n_orders, n_orders)` — a
new public helper on `twod.py` that re-solves at `n_orders − step`, compares
per-order efficiencies on the orders both solves retain, and warns naming
whichever of the two bars was missed (`_PER_ORDER_TOL_2D` = 1e-2, the constant
the `stabilize=True` consensus already uses, and `_ADVISORY_TOL_2D` = 3e-3, the
measured clean closure floor).  (b) `_ADVISORY_TOL_2D` as a documented
constant, and `energy_tol=` on `pmm_efficiency_2d` / `pmm_efficiency_2d_cell`
so a caller can run the tighter gate.  (c) `_warn_lossless_energy_2d`'s message
now says closure is not a convergence proof and names the helper.  (d)
`pmm_jones_2d`'s module docstring now points energy-critical work at
`pmm_jones_2d_staggered` / `PMM2DStackPure`, as `pmm_efficiency_2d`'s already
did, with the measured `n_orders`-invariance and its wall-time price.

**Why `_PASSIVE_TOL_2D` is NOT tightened.**  The WP offers "tighten … **or**
add an advisory threshold".  Tightening is the wrong half here for two
measured reasons.  First, the constant is shared with the `_stabilize_scalar` /
`_stabilize_jones` passive gates, so tightening it would make those REJECT
solves they currently accept — including legitimately floor-limited ones (the
audit's own L-shaped chiral cell reads `ΣR+ΣT = 1.0112` at n_orders 5, which a
3e-3 gate would call a failure).  Second, and decisively: on the fixture that
matters the closure is ANTI-correlated with the per-order error, so no value of
this tolerance turns it into a convergence signal.  The advisory constant plus
the drift check address the finding without a silent default change.

**How I verified.**  The audit's `q14_stagtime.py` fixture reproduced exactly:

| `pmm_efficiency_2d_cell` | ΣR+ΣT | T00 |
|---|---|---|
| n_orders = 5 | 1.009688038044 | 0.2715166794 |
| n_orders = 9 | 1.003634425845 | 0.2373949696 |
| n_orders = 11 | 0.999564765274 | **0.1542801483** |

`pmm_2d_order_drift(solve_at, 11)` returns `max_drift = 8.311e-02`,
`drift_00 = 8.311e-02`, `closure = 4.352e-04`, `closure_prev = 3.634e-03`,
`converged = False` and warns — i.e. it catches precisely the −35 % swing the
energy test calls an improvement.  On a low-contrast (eps 2.25) pillar at the
same settings it returns `converged=True` and is silent, so it is a signal
rather than noise.  `energy_tol=_ADVISORY_TOL_2D` fires on a 1e-2-class
closure the default gate lets through.

**Residual risk.**  The helper costs one extra (cheaper) solve; it is opt-in.
The plateau itself is unchanged — it is a scope property of the engine, and the
docstrings now say so and name the no-floor alternative.

### 2.4 G8 (P2) — `PreparedPMM2D.solve`

**What was wrong.**  Three silent differences from `pmm_efficiency_2d` on the
path every `*_vs_wavelength` sweep takes: no even-parity fold (which is what
the docstring's "~1e-13 division reorder" actually was), no `truncation`, and
no `_warn_lossless_energy_2d`.

**What I changed.**  `symmetry`, `truncation` and `energy_tol` are parameters
of `prepare_pmm_2d`, `prepare_pmm_2d_cell` and both `*_vs_wavelength` helpers,
carrying the SAME defaults as the direct entries.  `truncation` is applied in
`_prepare_pmm2d_core` (the order set and the projected operators are both
wavelength-free, so the circular restriction happens once).  `solve` runs
`_symmetric_solve_2d` on the same gate `_pmm2d_solve_core` uses, and calls
`_warn_lossless_energy_2d` on the stored `eps_reals`.  The class docstring's
"only delta" sentence is replaced by the measurement.

**How I verified.**  Prepared vs direct is now **bit-identical**
(`max|dR| = max|dT| = 0.000e+00`) at `symmetry` ∈ {`'auto'`, `False`} and
(degree, n_orders) ∈ {(9, 4), (7, 3)} — pre-fix it was 6.33e-14 / 1.08e-13
against the default.  The audit's `q13_warn.py` configuration (degree 7,
n_orders 2, lossless eps-12.25 pillar) gives `sum(R+T) = 1.054125` on the
direct, prepared and sweep paths alike and now produces **one warning on each**
(pre-fix: 1 / 0 / 0).  `truncation='circular'` retains 29 orders on all three
paths against 49 rectangular, bit-identical to the direct entry.

**Residual risk.**  A prepared sweep at the defaults now takes the fold, so its
numbers move by ~1e-13 — onto the direct entry's default answer.  Migration
note in the changelog (`symmetry=False` restores the old bits).

### 2.5 G9 (P2) — the staggered `eps_cell` cost cliff

**What was wrong.**  `eps_cell` is a PIXEL grid in the hybrid family (redundant
rows merged for free, cost-guarded) and a SEGMENT grid in the staggered family
(every row an element, pencil `2·Nx(M−1)·Ny(M−1)`, **no cost guard at all**).
The parameter name and the sibling's behaviour actively invite the mistake, and
it is a ~1000× cliff.

**What I changed.**  `_stag_merged_segments` (the `_cell_to_walls_tile` merge
rule, applied jointly to `eps_cell` and `mu_cell`), `_validate_stag_cost`, and
`_MAX_STAG_PENCIL_DOF = 12 000`, wired into `pmm_efficiency_2d_staggered`,
`pmm_jones_2d_staggered` (forwarded) and `PMM2DStackPure.add_layer` (both the
plain and the magnetic branch), all exposing `max_pencil_dof=`.  Plus
PIXEL-vs-SEGMENT cross-references in both families' docs.

Two tiers, because one cannot do the job of the other: an ABSOLUTE cap cannot
separate the pathological 12-segment grid at M = 5 (pencil 4608) from a
legitimate 8-segment grid at M = 8 (pencil 6272, larger), and a redundancy test
cannot bound an already-minimal but enormous grid.  So the redundancy test
WARNS (splitting a region into more segments is a legal h-refinement) and the
absolute cap RAISES.

**Calibration.**  The cap admits every grid in the shipped staggered suite
(largest 8 segments/axis at M = 8 → 6272; the documented M = 10 / 3-segment
solve → 1458) and projects ~28 GB at the cap (measured RSS ≈ 10–12 × dof²·16 B:
3.9 GB at 4608, 8.4 GB at 7200).

**The suggestion is NOT the distinct-strip count, and running the suite is what
proved it.**  A first-cut predicate that suggested the merged strip count
produced 25 warnings on the shipped staggered suite, and inspecting them
surfaced two ways that number is WRONG ADVICE:

* **the walls must survive.**  The audit's own 12×12 half-fill pillar has 3
  distinct strips, but its walls sit at indices 3 and 9 — at 1/4 and 3/4 —
  and this family's default path pins walls to the uniform lattice
  `i·P/N`.  A 3-segment lattice puts them at 1/3 and 2/3, so "re-express it on
  a 3×3 grid" would silently change the DUTY CYCLE: the same failure mode as
  the sibling `period_x` finding in this very WP.  The reducible factor is
  `g = gcd(N, every wall index on either axis)` and the answer is `N/g` = **4**,
  not 3.  (The audit's "450×450 suffices" carries the same slip; the honest
  figure at M = 6 is 800×800.)
* **the grid must stay SQUARE.**  A 1-D stripe merges per axis to `(2, 1)`,
  which the family's `Nx == Ny` guard rejects outright — hence one `g` over the
  JOINT wall set of both axes.

A third class is exempt rather than wrong: a cell that reduces to **1×1** is
UNIFORM, and tiling a uniform axis into equal segments is the documented way to
satisfy that same square contract (`PMM2DStackPure.add_layer(grid=)` exists
precisely to h-refine a uniform region).

With `_stag_minimal_uniform_segments` implementing the gcd rule, the audit's
own case still warns.

**CORRECTION (VERIFY-A13, 2026-09-12).**  This paragraph originally read "the
shipped staggered suite is SILENT".  Re-measured, it was not: **6** cost-guard
warnings remained, all in `tests/unit/test_pmm2d_staggered_mortar.py` — a file
this WP re-ran only in its earlier "remainder" group, BEFORE the gcd rule
landed, so the re-run that produced the "silent" reading never covered it.  The
arithmetic was right; the ADVICE was not followable.  All six sites are
`layer_grids='shared'` union-grid arms where a 2x2 and a 3x3 pillar are
deliberately tiled onto one common 6x6 lattice because the shared contract
requires it, and each layer was told to use its own minimum — measured, obeying
either one made the OTHER layer's `add_layer` raise `all patterned layers must
share ONE common (Nx, Ny) grid`.  The union grid is a property of the STACK, so
the advice is too: the followable number is the lcm of the per-layer minima
(`lcm(2, 3) = 6`, the grid the caller already passed).  VERIFY-A13 implemented
that joint rule — `PMM2DStackPure.solve` now takes the warn arm ONCE over every
patterned layer, `add_layer` on a shared stack keeps only the absolute refusal,
and `layer_grids='per-layer'` is unchanged.  Re-measured after the change:
`test_pmm2d_staggered_mortar.py` emits **0**, and the audit's single-layer 12x12
case still warns naming its 4x4 lattice.

**How I verified.**  Guard-only, so nothing expensive runs: the 12×12 array has
3×3 distinct strips and a minimal uniform lattice of 4;
`pmm_efficiency_2d_staggered(degree=6)` names a **7200×7200** pencil and
`PMM2DStackPure(n_modes=5).add_layer` a **4608×4608** one — exactly the audit's
two readings; at the default cap the latter warns with *"12×12 segments and
only 3×3 DISTINCT strips … the SAME geometry — the same walls, to the segment —
is expressible on the uniform 4×4 lattice at pencil 512, 729× less QZ time and
81× less memory"*; a 3×3 centred pillar, a 3×3 stripe, a 6×6 single-corner cell
and any uniform grid are silent; the same 12×12 array through
`pmm_efficiency_2d_cell` still takes 0.7 s.

**Residual risk.**  A user deliberately h-refining a PATTERNED region now gets
one warning per `add_layer`; `max_pencil_dof=` is the documented acknowledgement.

### 2.6 G10 (P2, performance)

Six sub-items.  Five done, one deferred, one measured and deliberately not
adopted.

**(a) The dense Kronecker projector pair — done.**  `_jax_twod._static_prep`,
`_static_prep_cell` and `_jax_stack2d._layer_static_traced` built
`Tp = kron(Ty, Tx)` + `pinv(Tp)` and then rebuilt the per-axis projectors three
lines later; `_static_prep` also formed a dense `N×N` `diag(1/Mdiag)` and two
dense `Minv @ kron(...)` products.  All three now build `Tx/Txp/Ty/Typ` once
(`_axis_projector_pair`) and every jnp sandwich is `_proj_sandwich_jnp` — the
two per-axis einsum contractions of `twod._sandwich_factorized`.  MEASURED: the
dropped pair alone is 7.2 / 22.2 / 43.2 MiB at 12 / 16 / 32 strips per axis
(degree 9, n_orders 7 / 11 / 11) against 2.5 / 13.1 / 13.7 MiB retained by the
whole prep, and **2.4 GiB** at `_MAX_NODAL_DOF = 150 000` with n_orders 11 —
plus one `O(N·Nf²)` pinv and two redundant per-axis pinvs in every case.
It is also a PARITY win: the dense `Tp @ Gx0 @ Tpinv` spelling carried an extra
`Ty Typ` factor `_scalar_projected_ops` does not, so the frozen constants are
now bit-identical to the NumPy path's and the pillar entry's NumPy/JAX `T`
parity improves from RMS relative **8.47e-11** (the audit's reading) to
**1.63e-13**.  AD-vs-central-FD stays clean at 1.04e-07 against the 1e-4 gate.

**(b) Diagonal GLL masses — done, bit-identical.**  Verified first that the
masses are EXACTLY diagonal (`count_nonzero(M − diag(diag(M))) == 0` at degree
5/7/11 × 1–3 elements per strip, for `M` and every `Mtile`), then that LAPACK's
`inv` of such a matrix is exactly `1/d` and the dense products add only exact
zeros — so `twod._axis_ops_1d`'s four operators are `np.array_equal` to the
pre-fix body at every setting measured, at **3.7×** (200 calls 29.2 → 7.8 ms at
n = 33).  Same treatment in `twod_jones`'s separable `_mass`: **4.8×** (400
calls 41.7 → 8.7 ms), also bit-identical.  I deliberately did NOT collapse
`T1 @ diag(v) @ T1p` to `(T1 * v) @ T1p` there: measured, that is 1 ULP
different (1.75e-16 relative), and bit-identity is worth more than the
remaining `O(nO·n²)` on a validated path.

**Control for (b).**  Because "bit-identical" is the whole claim, I also
checked it end to end rather than only on the operators: with the
`twod_jones` half of the change REVERTED in place, the audit's `p1cde.py`
90-degree-covariance probe prints `9.948e-14 / 1.323e-13 / 3.055e-13 /
1.675e-13` — the SAME digits as with the change in.  (Those differ from the
audit's own reading of `1.96e-14 / 7.85e-14 / 3.30e-14 / 5.44e-14` by a factor
of a few in both directions; the control shows that gap is the two machines'
BLAS, not this WP.)

**(c) `cascade='fused'` as the default — measured, NOT adopted.**  The audit's
premise is that it is "free" because `'tree'` measured bit-identical to
`'fast'`.  On my fixtures it is not: over three oblique multilayer stacks at
n_orders 4 and 7, `'fused'` vs `'fast'` gives `max|dR| ≤ 2.0e-14`,
`max|dT| ≤ 1.0e-13`, `max|dJ| ≤ 8.7e-14` and is NEVER bit-identical (`'tree'`
likewise, 1.3e-15…1.1e-14), and the whole-solve median over 5 interleaved runs
is 1.04× / 1.15× / 1.10×.  A few per cent does not buy a silent 1e-13 move of
every user's bits (COMMON rule 8), so the default stays `'fast'` and the
measurement is now in the `cascade` docstring so the choice is informed.

**(d) Caching the k0-free projected TENSOR operators — deferred**, see §6.1.

**(e) Circular truncation on `pmm_jones_2d`, `PMM2DStackHybrid` and the sweeps
— done.**  The projected operators are functions of the order LIST, so
restricting them with `np.ix_(keep, keep)` IS the operator built on the
circular list — the same restriction `_pmm2d_solve_core` already applied to
`lops`.  `_tensor_layer_modes` gained a `keep=` mask applied to
`GxF/GyF/Cxx/Cxy/Cyx/Cyy/EZZ` and the out-of-plane blocks after assembly (the
kron-factored branches must be assembled on the full box), and the stack
restricts its cached full-box `lops` at use, so ONE `_geom_cache` entry serves
both truncations.  MEASURED at degree 9 / n_orders 5: 121 → 81 orders,
0.216 → 0.068 s (scalar stack layer); 81 → 49 orders, 0.069 → 0.034 s
(in-plane tensor) and 0.117 → 0.051 s (out-of-plane tensor);
0.200 → 0.082 s on `pmm_jones_2d` with `symmetry=False`.  The tensor→scalar
reduction contract still holds under the circular set (3.9e-14).
`'rectangular'` is the default and takes every branch bit-for-bit unchanged.
Both JAX dispatches refuse `'circular'` loudly rather than silently returning
the rectangular answer.

**(f) The eig-cache refusal counter — done.**  `solve` snapshots
`_eig_cache.n_refused` around `_layer_mode_sets` and warns ONCE per instance
when it moves, naming the retained bytes, the budget and the per-entry cost.
Verified with `cache_max_bytes=1` at oblique incidence (normal incidence on a
centro-symmetric cell takes the even-parity fold, which bypasses the modal
cache entirely, so there is nothing to refuse there): `refused = 2`, exactly
one warning, none on the second solve, none at the default budget — and the
answer is bit-identical with and without the budget, which is the
refuse-never-degrade contract.

### 2.7 G11 (P3) — the branch cut

`rcwa/_core.py` is WP-A14's file and I did not touch it.  I reproduced the
finding on the then-current HEAD (`_sqrt_decay(np.array([1e-20 - 1e-30j])) →
-1.0e-10 + 5.0e-21j`, a flipped EVANESCENT mode) and wrote the PMM-2D-side gate
the WP asked for as the analytic BOUND rather than as the current output,
precisely so that it would survive the predicate change:

```
|exp(-lam k0 L)|  <=  exp(band * max|lam| * k0 * L),   band = 1e-8
```

asserted on a real PMM-2D layer spectrum at the entries' own regime (a 0.3 µm
layer at λ = 1 µm: bound `1 + 5.6e-07`, measured growth inside it).

WP-A14 then landed `& (Im(r)**2 > Re(r)**2)` mid-session, so the test now also
asserts the FIXED contract directly, as the coordinator asked: the audit's
counterexample must NOT flip (`Re(lam) >= 0`), while a genuinely ON-CUT root
(near the imaginary axis) still does.  MEASURED across that commit:
`-1.000000e-10 + 5.000000e-21j` → `+1.000000e-10 - 5.000000e-21j`.  Keeping
the bound half means the file still says something true if the predicate is
ever re-tuned.

### 2.8 G12 (P3) — cache key and attribute hardening

`symmetry` (and `truncation`, for the same reason) added to `_mode_key`'s
`common`.  `period_x`/`period_y` are now read-only properties once
`self._layers` is non-empty; before the first `add_layer` they are still
settable (nothing derived exists yet).  `_set_period`'s message carries the
measurement.  The class docstring gained a Notes section saying which
attributes are mutable between solves and why the periods are the exception.

Verified: the `symmetry` mutation is now honoured (pre-fix the re-solve was
BIT-IDENTICAL to the pre-mutation answer; now it differs from that and matches
a fresh object to 2.2e-16), and both period setters raise while leaving the
value unchanged.  `_materialized_layers`'s `probe.__dict__.update(self.__dict__)`
clone and `solve_vs_wavelength`'s `copy.copy` clone both still work
(`_period_x` rides in `__dict__`; smoke-tested on a dispersive 3-wavelength
sweep and a threaded 2-worker Jones sweep).

### 2.9 G13 (P3) — the transmission Jones and the documentation tail

`return_jones_transmission=True` adds a 5th return built from the `tx`/`ty`
already computed in `_pmm_jones_2d_at`, in the SAME layout
`PerOrderAmplitudesMixin.jones_transmission` uses.  A subtlety the audit did
not have to consider: `pmm_jones_2d` takes `slant=`, and a slanted patterned
cell is solved in a sheared frame whose exit plane sits a lateral `t · depth`
from the lab one — so the transmitted amplitudes need the same unimodular
per-order anchor `exp(+i k0 (α_m·t) d)` that `PMM2DStackHybrid.solve` applies
(and whose absence there was invisible to every energy check until 2026-09-10).
I added it, with `_slanted_cell_is_a_frame_noop` as the same exemption test the
stack uses, and verified it against the stack: **bit-identical** (`0.000e+00`)
for both the vertical and the `slant=(0.4, 0)` case, and the anchor is not a
no-op on that cell (`max|ΔJ_t| > 1e-3` between them).

QWP verification (Λ/λ = 0.2, duty 0.5, d = 208.14 nm): retardance +100.12 /
+100.23 / +99.90° at n_orders 5 / 9 / 15 with `fff_nv` against the
`rcwa_jones_1d(n_orders=60, 'li')` reference **+100.066°**; POSITIVE on the
SLOW axis, i.e. `exp(+i·retardance)`, exactly CONVENTIONS §7; the S₃ = −1 loop
closes (`diag(1, e^{+iπ/2})` with the FAST axis at +45° on x-pol gives
`S = (1, 0, 0, −1)` under the library's own `S3 = −2 Im(Ex conj Ey)`).  The
4-tuple contract is untouched when the flag is off (`np.array_equal` on all
four elements).  `stabilize=True` returns the transmission Jones of the degree
the consensus picked (matched by object identity on the reflection Jones the
consensus returns).  The JAX path refuses.

Also in G13's tail: the `_assemble_2d` "reference implementation" note, the two
dead `# noqa: F401` re-exports deleted, and the Notes seam note pointing at the
two stacks' `jones_transmission()`.

---

## 3. Files touched

Source (all within the WP's ownership list):

* `lumenairy/elements/pmm/twod.py` — `_ADVISORY_TOL_2D`, `pmm_2d_order_drift`,
  `_warn_lossless_energy_2d(tol=)`, `_axis_ops_1d` (diagonal masses),
  `_assemble_2d` note, `_layer_modes_projected` docstring,
  `_cell_to_walls_tile` PIXEL/SEGMENT note, `energy_tol=` on both scalar
  entries, `PreparedPMM2D` (slots/solve/docstring), `_prepare_pmm2d_core`,
  `prepare_pmm_2d`, `prepare_pmm_2d_cell`, both `*_vs_wavelength` helpers.
* `lumenairy/elements/pmm/twod_jones.py` — module docstring, `formulation`
  docstring + `'auto'`, `truncation=`, `return_jones_transmission=`,
  `_tensor_layer_modes(keep=)`, the separable `_mass` diagonal-mass rewrite,
  the `fff_nv` fold gate, `_pmm_jones_2d_at` 5-tuple + slant frame anchor.
* `lumenairy/elements/pmm/stack2d.py` — per-slot Li routing (2 sites),
  `_mode_key` (+`symmetry`, +`truncation`), read-only period properties,
  `_order_keep_mask` / `_restrict_lops` + circular threading,
  `_warn_eig_cache_refusals`, `cache_stats` docstring, `cascade` docstring
  measurement, class docstring Notes, JAX circular refusal.
* `lumenairy/elements/pmm/_jax_stack2d.py` — `_modes_projected` per-slot pair,
  `_layer_static_traced` factorized projectors, the traced branch's sandwiches.
* `lumenairy/elements/pmm/_jax_twod.py` — `_axis_projector_pair`,
  `_proj_sandwich_jnp`, `_static_prep`, `_static_prep_cell`,
  `_scalar_jax_tail`, dead re-export removed.
* `lumenairy/elements/pmm/_jax_twod_jones.py` — `_proj` factorized, dead
  re-export removed.
* `lumenairy/elements/pmm/twod_staggered.py` — `_MAX_STAG_PENCIL_DOF`,
  `_stag_merged_segments`, `_validate_stag_cost`, `max_pencil_dof=` on both
  public entries + docstrings.
* `lumenairy/elements/pmm/stack2d_pure.py` — `add_layer(max_pencil_dof=)` +
  the guard on the plain and magnetic branches + docstring, import.

All eight modules additionally pass `fn_name=` to WP-A14's newly-announcing
`_grazing_safe_wavelength`, so a Wood-anomaly substitution names the entry that
took it (`pmm_efficiency_2d_cell: a diffracted order sits EXACTLY at
cut-off ...`) instead of the helper.

Tests — new:

* `tests/unit/test_audit2609_a13_stack2d.py` (14 tests, 17 s)
* `tests/unit/test_audit2609_a13_twod.py` (20 tests, 29 s)
* `tests/unit/test_audit2609_a13_staggered_cost.py` (8 tests, 0.12 s)
* `tests/unit/test_audit2609_a13_jax.py` (8 tests, 17 s)

Tests — edited in place (one mechanical rename, my module):

* `tests/unit/test_audit_w4_jax_static_caches.py` — `_PREP_KEYS` and the two
  `c0["Tp"]` reads now name the per-axis projectors, since `Tp`/`Tpinv` no
  longer exist.  The cache contract the file pins (bounded, LRU, frozen,
  byte-identical after eviction) is unchanged and still passes (11 tests).

No test in the suite asserted the OLD (wrong) behaviour of any finding, so
nothing had to be un-pinned.

---

## 4. Tests run

All with `OPENBLAS_NUM_THREADS=1`, `-q --no-header -p no:cacheprovider`.

| command | result |
|---|---|
| `pytest tests/unit/test_audit2609_a13_stack2d.py` | **14 passed, 17.4 s** |
| `pytest tests/unit/test_audit2609_a13_twod.py` | **20 passed, 28.9 s** |
| `pytest tests/unit/test_audit2609_a13_staggered_cost.py` | **8 passed, 0.12 s** |
| `pytest tests/unit/test_audit2609_a13_jax.py` | **8 passed, 17.1 s** |
| `pytest tests/unit/test_audit_w4_jax_static_caches.py` | **11 passed, 1.9 s** |
| staggered suite (9 files): `test_pmm2d_staggered_{anisotropic,magnetic,nonuniform,oop,oop_block_eig,slant,wood_list}.py test_v5_12_0_pmm2d_staggered.py test_audit_p1_staggered_guard.py` | **269 passed, 0 failed, 767 s** (first-cut G9 predicate, 25 new warnings) |
| staggered re-run after the exemptions (8 of those files) | **234 passed, 0 failed, 605 s**, new warnings 25 → 5 |
| staggered re-run after the gcd rule (same 8 files) | **234 passed, 0 failed, 713.8 s**, new warnings 5 → **0** |
| staggered / mortar remainder (9 files): `test_pmm2d_staggered_oop.py test_pmm2d_staggered_mortar.py test_{fix,verify}_pmm2d_mortar_round*.py test_v5_21_pmm2d_staggered_oblique.py test_pmm2d_staggered_oop_corner_convergence.py` | **115 passed, 1 skipped, 1 FAILED, 829.6 s** — the failure is `test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`, diagnosed in §5 item 4 as WP-A12's 1-D `PMMStack` union-grid warning, not this WP |
| all four new files + the edited cache test, re-run after the `fn_name` and G11 changes | **61 passed, 62.2 s** |
| post-`fn_name` regression over the Wood-sensitive PMM 2-D files (14 files: the hybrid entries, `PreparedPMM2D`, the Jones entry, the internal field, `fff_nv`, the lossless tripwires, and the staggered Wood-list / oblique files) | **134 passed, 0 failed, 875.7 s** |
| hybrid suite (16 files): `test_v5_11_0_pmm2d.py test_v5_14_0_pmm2d_{cell,stack,conical,oop,stabilize}.py test_v5_14_0_pmm_jones_2d.py test_v5_12_0_pmm2d_loss.py test_v5_13_0_pmm2d_hybrid_sweep.py test_v5_20_13_pmm_jones_2d_fff_nv.py test_p2c_pmm2d_stack_cascade.py test_p2t_pmm2d_tree_cascade.py test_pmm2d_lossless_closure_two_sided.py test_pmm2d_oop_block_eig.py test_audit_s1_3_pmm2d_lossless_tripwire.py test_v5_14_3_pmm_internal_field.py` | **208 passed, 0 failed, 738.8 s** |
| slant / autodiff / misc: `test_pmm2d_slant_metric.py test_fix_hybrid_slant_transmission_anchor.py test_verify_pmm2d_perlayer_slant.py test_fix_slant_anchor_v1_v2_o2.py test_verify_slant_anchor_v1_v2_o2.py test_v5_14_0_pmm2d_autodiff.py test_v5_20_2_pmm_jones_2d_jax.py test_v5_14_2_jax_stacks.py test_niche_audit_w7_pmm.py test_v5_14_0_pmm_audit_fixes.py test_audit_w3_pmm_jax_guards.py test_v5_21_pmm_threaded_sweep.py test_audit_s5_4_standalone_jones_transmission.py test_v5_18_1_jones_shared_eig.py test_audit_w4_jax_static_caches.py test_fix_branch_cut_round2.py test_verify_branch_cut_round2.py` | **339 passed, 0 failed, 427.4 s** |

**Baseline (pre-change), same machine:** the first four hybrid files were
`39 passed in 327.8 s`.

**Pre-existing / cross-WP failures, and environment notes.**

* `test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`
  **FAILS**, on `assert nwarn_ok == 0`.  My judgement: **not related to this
  WP.**  The one warning is `_pmm_union_grid: snapped 2 pair(s) of
  NEAR-COINCIDENT cross-layer walls closer than min_feature ...`, a new
  diagnostic in `lumenairy/elements/pmm/stack.py` (the 1-D `PMMStack`,
  WP-A12's file), and the fixture is a 1-D two-layer stack whose walls differ
  by `1e-4` of the period — exactly what the new warning is for.  Reproduced
  standalone outside pytest: one warning at both wall separations, closures
  `1.0000000000000562` / `1.0000000000000628`.  Nothing in WP-A13 is on that
  call path (this WP owns `twod*.py`, `stack2d*.py`, `_jax_*2d*.py`; the test
  drives `pmm/stack.py` and `pmm/_core.py::_interface_smatrix`).  Handed to
  WP-A12 in §5, item 4.
* One background run of the secondary hybrid files aborted at import with
  `SyntaxError: source code string cannot contain null bytes` from
  `lumenairy/io/prescriptions_zemax.py` — a transient mid-edit state of another
  agent's file, unrelated to this WP; the same command succeeded on re-run.
* No `validation/` topic file covers the 2-D PMM
  (`validation/elements/test_rcwa.py` has zero PMM-2D call sites), so
  `validation/run_all.py` has nothing to run for this area.
* **Mid-session dependency: WP-A14 landed** (`e1bf79be`) while this WP was in
  flight, adding `WoodNudgeWarning` to `_grazing_safe_wavelength` and the
  `Im(r)^2 > Re(r)^2` term to `_sqrt_decay`.  Both were absorbed: all eight
  PMM 2-D Wood call sites now pass `fn_name=` so the message names the entry
  rather than the helper, the G11 test was strengthened to the fixed
  predicate, and every PMM 2-D suite above was re-run or ran after that commit
  with no filter needed.

**Pre-fix confirmation of the new tests.**  I reverted the three G5/G12 source
edits in place (a scripted patch and restore, no git writes) and re-ran
`test_audit2609_a13_stack2d.py`: **10 failed, 4 passed** — all six
rotation-invariance parametrisations, both same-physics parametrisations, the
structural routing gate and the `symmetry`-cache-key test.  Restored:
14 passed.  The remaining 4 test new API (the period refusal, the refusal
warning, circular truncation) and have no pre-fix form.

### 4.1 "Keep intact" — the audit's verified-correct list, re-measured

The WP names the properties that must survive.  Every one was re-measured on
the audit's OWN repro scripts after the changes (mine → the audit's reading):

| property | probe | after | audit |
|---|---|---|---|
| Jones basis / columns / sign vs an independent 3-medium TMM, θ = 0° and 30° | `p1b.py` | `\|Jxx−r_x\|` **6.21e-17** / 1.42e-16, `\|Jyy−r_s\|` 6.21e-17 / 1.24e-16, `\|Jxy\|=\|Jyx\|=0` | 6.21e-17 / 1.42e-16 / 1.24e-16 — identical |
| φ-covariance of the uniform limit (φ = 45°) | `p1b.py` | off-diagonals ≤ 2.0e-16 | ≤ 2e-16 |
| 90° rotation covariance of `pmm_jones_2d` ITSELF | `p1cde.py` (b) | 9.95e-14 / 1.32e-13 / 3.06e-13 / 1.68e-13 against `\|J\| ≈ 0.99` | 1.96e-14 / 7.85e-14 / 3.30e-14 / 5.44e-14 (same decade; the control in §2.6(b) shows the gap is the two machines' BLAS, not this WP — the digits are unchanged by my edits) |
| mirror symmetry → zero cross-pol | `p1cde.py` (c) | `\|Jxy\| = 3.32e-14`, `\|Jyx\| = 4.18e-15` | 2.61e-14 / 1.32e-15 |
| even-parity fold vs the full solve | `p1cde.py` (c) | 3.21e-13 | 1.75e-13 |
| C4 → `Jxx = Jyy` | `p1cde.py` (d) | 1.80e-14 (laurent) / 1.47e-13 (li), `\|Jxy\| ≤ 2.3e-14` | 2.42e-13 / 8.25e-14 |
| `fff_nv` crossed-cell raise + separable reduction | `test_audit2609_a13_twod.py::test_g6_auto_*` | raises with the documented message; the reduction reproduces the audit's convergence table | ✓ |
| cache keys (`formulation`, `degree`, `grade`, `n_orders`, walls, tiles) | existing suite + `test_g12_*` | unchanged, plus `symmetry`/`truncation` now covered | ✓ |
| JAX x64 enforcement / NumPy parity / AD-vs-FD | `test_audit_w3_pmm_jax_guards.py`, `test_v5_20_2_pmm_jones_2d_jax.py`, `test_audit2609_a13_jax.py` | x64 raise unchanged; parity **improved** (pillar `T` 8.47e-11 → 1.63e-13); AD-vs-FD 1.04e-07 | ✓ |
| frame-anchor gating, slant refusals | `test_fix_hybrid_slant_transmission_anchor.py`, `test_verify_pmm2d_perlayer_slant.py`, `test_pmm2d_slant_metric.py`, `test_fix_slant_anchor_v1_v2_o2.py` | pass; the new `pmm_jones_2d` transmission Jones reproduces the stack's anchor BIT-IDENTICALLY | ✓ |
| the staggered engine's `n_orders` invariance | the staggered suite (269 tests) | pass | ✓ |

The staggered solves in my own tests are guard-only (no eigensolve), and the
heaviest staggered fixture I ran is the shipped suite's own (M ≤ 8 in every new
test), per the WP's wall-time instruction.

---

## 5. Requested changes outside my ownership

1. **`lumenairy/elements/pmm/__init__.py` and `lumenairy/__init__.py`** — add
   `pmm_2d_order_drift` to the re-export lists.  It is in `twod.__all__`, so
   `from lumenairy.elements.pmm import pmm_2d_order_drift` already works, but
   `lumenairy.pmm_2d_order_drift` does not and the name is missing from
   `pmm.__all__`.  Exact change: add the string `"pmm_2d_order_drift"` to
   `lumenairy/elements/pmm/__init__.py::__all__` (beside
   `"pmm_efficiency_2d_cell_vs_wavelength"`), and to whichever PMM block of
   `lumenairy/__init__.py` re-exports `prepare_pmm_2d`.  *Why:* it is the
   convergence signal G7 asks users to reach for; a helper you cannot import
   from the package root will not be used.

2. **`CONVENTIONS.md` §7.1** — the tail of G13.  The text "the 1-D solvers
   return `te`/`tm` (`s`/`p`)" invites a te-first reading, which silently
   transposes a retardance sign.  MEASURED (re-confirmed here): for the
   Λ/λ = 0.2 form-birefringent grating, `rcwa_jones_1d`'s `J[0,0]` matches
   `PMM2DStackHybrid`'s `Jxx` and `J[1,1]` matches `Jyy` — index 0 is
   **x ≡ tm**, index 1 is **y ≡ te**, which is also what `rcwa_jones_1d`'s own
   Returns block says.  Exact change: state the ORDER explicitly ("index 0 =
   tm ≡ x, index 1 = te ≡ y at φ = 0").  `pmm_jones_2d`'s Returns block
   already cross-references §7.1 and now also documents the transmission
   convention, so the 2-D side degrades gracefully until this lands.

3. **`CHANGELOG.md`** — assembled by the orchestrator from
   `WP-A13_CHANGELOG.md` (this directory).

4. **`tests/unit/test_fix_pmm2d_mortar_round2.py::test_the_plain_1d_interface_solve_is_left_unguarded_and_this_is_why`
   — FAILING, and not mine to fix.**  Its `assert nwarn_ok == 0` now sees ONE
   warning: `_pmm_union_grid: snapped 2 pair(s) of NEAR-COINCIDENT cross-layer
   walls closer than min_feature ...`, a new diagnostic in
   `lumenairy/elements/pmm/stack.py` (the 1-D `PMMStack` union grid — WP-A12's
   file).  The fixture is built from exactly such walls (a 1-D two-layer stack
   whose walls differ by `delta = 1e-4` of the period), so the warning is
   correct and the assertion is stale.  Reproduced standalone: 1 warning at
   both `delta = 1e-4` and `1e-5`, closure `1.0000000000000562` /
   `1.0000000000000628`.  Nothing in this WP can reach it — my files are
   `twod*.py` / `stack2d*.py` / `_jax_*2d*.py` and this test drives
   `pmm/stack.py` + `pmm/_core.py::_interface_smatrix`.  Suggested fix (WP-A12's
   call): either scope the union-grid snap warning so it does not fire on a
   snap the caller cannot avoid, or narrow the assertion to
   "no warning from THIS site" by filtering that message.

5. *(informational, now satisfied)* my
   `test_g11_no_layer_mode_grows_beyond_the_branch_cut_band` was written to
   assert the `_sqrt_decay` BOUND rather than its output so that WP-A14's
   predicate change could not break it; that change has landed and the test now
   asserts the fixed contract as well.  Both halves pass.

---

## 6. Deferred

### 6.1 G10(d) — caching the k0-free projected TENSOR operators

**Not done.**  `PMM2DStackHybrid._build_layer_modes` passes `lops=None` for
tensor layers, so `_geom_cache` stores `(ax, ay, None)`: the nodal axis build
IS already reused across a sweep, but `_tensor_layer_modes` rebuilds the
per-axis projections (`_axis_projection` + `pinv`, `O(nq·degree²)` plus a pinv)
and the `_proj` sandwiches at every wavelength and every angle, where the
scalar branch caches them.

**Why deferred.**  The k0 split is BRANCH-DEPENDENT inside a 220-line function
carrying four discretization branches, the out-of-plane Schur fold, the
`return_ops` fold gate and the `block_eig` gauge: the uniform branch's `GxF` is
`diag(kxv)` (no k0-free part at all), the separable branch's is
`kron(Iy, g1)/k0 + kx0·kron(Iy, ip1)` on the patterned axis and `diag(kyv)` on
the other, and only the crossed branch has the scalar branch's clean
`Gx0F/k0 + kx0·Ip` shape.  Splitting assembly from eigen-dispatch correctly is
the right fix (the audit's own code-organisation note says so) but it is a
refactor of validated physics, and the measured share is small — the audit
timed `_proj` at 0.47 s of a ~20 s solve.

**Concrete design.**  Extract
`_tensor_projected_ops(ax, ay, x_walls, y_walls, tile_i, ox, oy, formulation)
-> dict(kind, Gx0F|None, IpxF|None, Gy0F|None, IpyF|None, Cxx, Cxy, Cyx, Cyy,
EZZ, oop)` by MOVING the three assembly branches verbatim, with `kind` naming
which axes are k0-free.  `_tensor_layer_modes` keeps its signature, gains
`ops=None`, and when given one rebuilds `GxF`/`GyF` from `kind` (`diag(kxv)`,
or `Gx0F/k0 + kx0·IpxF`) and goes straight to the fold / eig.  `stack2d`'s
`_geom_key` then caches `ops` for tensor layers exactly as it caches `lops` for
scalar ones (no key change: the key already covers walls, tile bytes, degree,
grade, periods and `n_orders`).  Gate: the moved code must be BIT-IDENTICAL —
assert `np.array_equal` on all seven operators against the current function for
uniform / separable / crossed / out-of-plane / `fff_nv` cells before and after,
and re-run `test_v5_14_0_pmm_jones_2d.py`, `test_v5_14_0_pmm2d_oop.py`,
`test_pmm2d_oop_block_eig.py`, `test_v5_20_13_pmm_jones_2d_fff_nv.py`.
**Effort: ~3–4 h** including the bit-identity harness.

### 6.2 G10(c) — `cascade='fused'` as the default

Measured and deliberately not adopted (§2.6): not bit-identical to `'fast'`
(1.5e-15 … 1.0e-13 across three fixtures at two truncations) for a whole-solve
median of 1.04×–1.15×.  Adopt it only together with a deliberate
"cascade default moves" entry and a regression sweep; the measurement is now in
the `cascade` docstring either way.  **Effort if adopted: ~1 h** plus whatever
the suite's bit-pinned cascade tests need.

### 6.3 Not attempted (out of scope for this WP)

The audit's alternative-algorithm recommendation — matched-coordinate /
adaptive-spatial-resolution FMM (Granet 1999, Weiss 2009), which is the
standard cure for the Laurent floor on crossed high-contrast cells and would
lift the crossed-`fff_nv` refusal — is a new solver, not a remediation.  It
remains the highest-value algorithmic addition for this partition.

---

## 7. Changelog

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A13_CHANGELOG.md`
