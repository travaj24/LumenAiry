# WP-A13 changelog text -- PMM 2-D (hybrid / staggered / stacks / JAX twins)

Findings G5-G13 of `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` §12
(partition report `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/PMM-2D.md`).

### Fixed -- PMM 2-D stack: `formulation='li'` no longer depends on which axis you drew the grating along

`PMM2DStackHybrid` (and its JAX twin) never routed the per-slot Li operators
`EpnxF`/`EpnyF` that `twod._pmm2d_solve_core` has passed since audit P3-33, so
under the class's **default** `formulation='li'` a y-patterned layer received
the y-axis inverse-rule operator on the **Ex** slot and Laurent on Ey -- both
slots anti-Li -- while an x-patterned layer was routed correctly.  One physical
grating therefore gave two different answers depending on the axis it was drawn
along, and a one-layer stack disagreed with `pmm_efficiency_2d_cell`, which is
documented as the same physics.  Nothing warned; `formulation='laurent'` was
symmetric throughout, which is what isolates the cause to the routing.
(`lumenairy/elements/pmm/stack2d.py` `_symmetric_layer_specs` /
`_build_layer_modes`, `lumenairy/elements/pmm/_jax_stack2d.py`
`_modes_projected`; finding **G5**, P1.)

MEASURED on one Si grating (eps 12.25/1.0, duty 1/2, Λ = 0.47 µm, λ = 1 µm,
d = 0.3 µm, n_sub = 1.5, normal incidence) drawn along x and along y:

| quantity | before | after |
|---|---|---|
| T00, n_orders 5 | 0.0231919632 vs 0.0229566480 (Δ 2.35e-04) | identical |
| T00, n_orders 9 | 0.0334736588 vs 0.0322424723 (Δ **1.23e-03**, 5.3 % of T00) | identical |
| `max|ΔT|` (degree 7, n_orders 5) | 4.977e-03 | **3.39e-14** |
| `|ΔJ|` reflection Jones | 2.318e-02 | **7.81e-14** |
| one-layer stack vs `pmm_efficiency_2d_cell` | 1.23e-03 | 3.8e-14 … 8.2e-13 |

Both `symmetry='auto'` and `symmetry=False` carried the defect; both are fixed.
No cache-key change was needed (`formulation` was already in `_mode_key`).
Gate: `tests/unit/test_audit2609_a13_stack2d.py` (six parametrisations of
`test_g5_li_is_90deg_rotation_invariant_on_the_stack`,
`test_g5_one_layer_stack_equals_the_single_cell_entry`,
`test_g5_the_routed_pair_is_what_the_stack_now_passes`) and
`tests/unit/test_audit2609_a13_jax.py::test_g5_jax_stack_twin_routes_the_per_slot_li_operators`
-- all ten fail on the pre-fix code.

### Fixed -- PMM 2-D sweeps: `PreparedPMM2D.solve` now folds, truncates and warns like the direct entry

`PreparedPMM2D.solve` -- the path every `pmm_efficiency_2d[_cell]_vs_wavelength`
sweep takes -- silently differed from `pmm_efficiency_2d` in three ways
(`lumenairy/elements/pmm/twod.py`; finding **G8**, P2):

* **no even-parity fold.**  `prepare_pmm_2d` took no `symmetry` argument at all
  and `solve` never called `_symmetric_solve_2d`, so the class docstring's
  "reproduces `pmm_efficiency_2d(...)` to ~1e-13 (the only delta: `Gx0F/k0`
  reorders one division)" was describing the MISSING FOLD: measured against
  `symmetry=False` it was exactly `0.000e+00`, and against the entry's DEFAULT
  `symmetry='auto'` `max|dR| = 6.33e-14` / `max|dT| = 1.08e-13`.  `symmetry` is
  now a parameter of `prepare_pmm_2d` / `prepare_pmm_2d_cell` and of both
  `*_vs_wavelength` helpers, defaulting to `'auto'` as on the direct entries,
  and the two paths are now **bit-identical** (`max|dR| = max|dT| = 0.000e+00`
  at both settings, degree 9 / n_orders 4 and degree 7 / n_orders 3).
* **no `truncation`.**  `pmm_efficiency_2d_cell(truncation='circular')`
  retained 29 orders on the audit's fixture while the prepared and sweep paths
  rejected the keyword with a `TypeError` and swept rectangular (49).
  `truncation` is now threaded and applied at prepare time (it is
  wavelength-free), and the retained sets and efficiencies match the direct
  entry exactly.
* **no lossless tripwire.**  `_warn_lossless_energy_2d` ran only from the
  `stabilize=False` arm of the direct entries.  At degree 7 / n_orders 2 on a
  provably lossless eps-12.25 pillar all three paths return
  `sum(R+T) = 1.054125` -- identical to the last digit -- and only the direct
  one warned, so a whole wavelength sweep could sit at a 5.4 % energy excess
  with no signal.  `solve` now runs it (`eps_reals` was already stored).

**Migration.** A prepared sweep built at the defaults now takes the even-parity
fold, so its numbers move by ~1e-13 (they move ONTO `pmm_efficiency_2d`'s
default answer).  Pass `symmetry=False` to `prepare_pmm_2d*` /
`*_vs_wavelength` for the previous bits.  A lossless sweep in an
ill-conditioned `(degree, n_orders)` corner will now emit the closure warning
it always should have.
Gate: `tests/unit/test_audit2609_a13_twod.py::test_g8_*`.

### Added -- `pmm_2d_order_drift`: the convergence signal lossless closure cannot give you

On the Fourier-projected hybrid, `sum(R)+sum(T)` is not a proxy for the
per-order error, and the two can move in OPPOSITE directions.  MEASURED on one
fixed cell (12×12 pixel grid, eps 12.25 pillar at duty 1/2, Px = Py = 0.9 µm,
λ = 1 µm, d = 0.3 µm, n_sub = 1.45, degree 11): closure improves monotonically
9.69e-03 → 3.63e-03 → 4.35e-04 at `n_orders` 5 / 9 / 11 while `T00` goes
0.2715167 → 0.2373950 → 0.1542801, i.e. **−35 %** between the last two with no
sign of settling -- so a user watching energy alone picks the worst of the
three.  `lumenairy.elements.pmm.twod.pmm_2d_order_drift(solve_at_n_orders,
n_orders)` re-solves at `n_orders - step` and reports `max_drift` / `drift_00`
/ `closure` / `closure_prev` / `converged`, warning when either bar is missed.
The module docstrings of `pmm_jones_2d` and `pmm_efficiency_2d[_cell]` now
state the measured plateau and point energy-critical work at the no-floor
staggered engine, as `pmm_efficiency_2d`'s already did.
(Finding **G7**, P2; gate `tests/unit/test_audit2609_a13_twod.py::test_g7_*`.)

### Added -- `energy_tol=` on the scalar 2-D entries, and `_ADVISORY_TOL_2D`

`_PASSIVE_TOL_2D = 5e-2` is a CATASTROPHE gate: it sits 2.5 decades above the
~3e-3 a clean hybrid solve reaches (measured 1.5e-03 / 1.4e-04 / 5.9e-04 at
`n_orders` 5 / 9 / 11 on a lossless Si pillar), so it cannot see a
1e-3-class regression.  It is NOT tightened -- the plateau is a documented
property of a Fourier-truncated engine (an L-shaped chiral cell legitimately
reads 1.1e-02 at n_orders 5) and closure is anti-correlated with the per-order
error anyway -- but `pmm_efficiency_2d` / `pmm_efficiency_2d_cell` now take
`energy_tol=` so a caller can ask for the tighter gate on a geometry they know
is clean, and `_ADVISORY_TOL_2D` (3e-3) documents the measured clean floor.
(Finding **G7**, P2.)

### Added -- `pmm_jones_2d(return_jones_transmission=True)`

`pmm_jones_2d` returned no TRANSMISSION Jones -- the observable for a
transmissive metasurface QWP -- and neither the module docstring nor the
Returns section pointed at `PMM2DStackHybrid/Pure.jones_transmission()`, the
only route that produced one.  The 5th return uses the same convention as those
accessors (rows `[E_x; E_y]`, columns = incident `E_x`/`E_y`, public
`exp(-iωt)`) and carries the slanted-frame anchor, so it is **bit-identical**
to `PMM2DStackHybrid.jones_transmission()` on the same slanted or vertical
layer.  `stabilize=True` returns the transmission Jones of the degree the
consensus picked; the JAX path refuses loudly (the jnp twin keeps no
amplitudes).  MEASURED on the form-birefringent Si/air QWP at Λ/λ = 0.2
(duty 0.5, d = λ/4/(n∥ − n⊥) = 208.14 nm): the retardance
`wrap(arg(J^t_yy) − arg(J^t_xx))` is POSITIVE on the SLOW axis --
`exp(+i·retardance)`, exactly CONVENTIONS §7 -- and reads +100.12 / +100.23 /
+99.90° at `n_orders` 5 / 9 / 15 with `formulation='fff_nv'` against the
`rcwa_jones_1d(n_orders=60, 'li')` reference **+100.066°**, i.e. inside 0.17°
at every truncation tested (`'laurent'` is 5.37° out at n_orders 5 and a
0th-order Rytov EMT slab is 4.19° out).  The Returns block and the Notes now
document the cross-engine seam.
(Finding **G13**, P3; gate `tests/unit/test_audit2609_a13_twod.py::test_g13_*`.)

### Changed -- `pmm_jones_2d`: the formulation docstring is now the measurement, and `formulation='auto'` is added

On this entry `'li'` and `'laurent'` differ ONLY in the `E_z` rule (the
in-plane block is Laurent either way), so `'li'` is not the wall-normal inverse
rule it is on the scalar entries -- and the docstring's "the hybrid's
*validated* inverse-rule elimination" read as "at least as good" when it is
measurably the worst of the three.  MEASURED on a separable high-contrast Si
stripe (degree 11) against `rcwa_jones_1d(n_orders=80, 'li')`, `|Jxx − ref|`:

| n_orders | `fff_nv` | `laurent` | `li` | `rcwa_jones_2d li` |
|---|---|---|---|---|
| 5 | 4.91e-02 | 1.38e-01 | 1.81e-01 | 7.31e-03 |
| 11 | **2.72e-03** | 3.02e-02 | **6.90e-02** | 2.95e-04 |

and on a C4 Si pillar `'li'` keeps the WRONG SIGN on `Im(Jxx)` at every
affordable truncation (`arg(Jxx)` 8.9° out at n_orders 11 against 1.2° for
`'laurent'`) while energy closes to 1e-5 on both, so no tripwire fires.  The
docstring now carries that table and the ordering.

`formulation='auto'` is added: `'fff_nv'` on a SEPARABLE in-plane cell (where
it is both available and best) and `'laurent'` otherwise; on the JAX path it
resolves to `'laurent'` (`'fff_nv'` is NumPy only).  The DEFAULT stays
`'laurent'` -- making `'fff_nv'` the default would silently break the
documented EXACT reduction of a scalar cell to
`pmm_efficiency_2d_cell(formulation='laurent')` for precisely the separable
cells the LC-QWP work uses -- so `'auto'` is opt-in and documented as the
recommended setting for new code.
(Finding **G6**, P2; gate `tests/unit/test_audit2609_a13_twod.py::test_g6_*`.)

### Performance -- `pmm_jones_2d`: the even-parity fold now runs for `fff_nv` (3.96×)

The fold was gated off for `fff_nv` alone (`twod_jones.py`
`formulation != "fff_nv"`), so the most accurate formulation was also the
slowest.  `fff_nv` is reachable only on a SEPARABLE cell, where every operator
the fold touches is a 1-D projected mass kron'd with an identity -- exactly the
shape the other two rules hand over -- and the fold's own precondition (a
centro-symmetric cell at normal incidence) is unchanged and still
auto-detected.  MEASURED at degree 11 / n_orders 11 on a centred Si stripe:
fold-vs-full `max|ΔJ| = 1.49e-12` (against 4.92e-13 / 5.02e-13 for
`laurent` / `li` on the same cell) for **3.96×** wall time (1.83 s vs 7.24 s).
(Finding **G6**, P2.)

### Performance -- PMM 2-D: exactly diagonal GLL masses are no longer inverted with LAPACK

`_build_axis` assembles `M` and every `Mtile` from `np.diag(w·J)` element
blocks, so `count_nonzero(M − diag(diag(M))) == 0` exactly (verified at degree
5/7/11 and 1-3 elements per strip).  The crossed-cell branch of
`_scalar_projected_ops` has exploited that since v5.14; the SEPARABLE branches
-- the ones a 1-D grating layer in a 2-D stack takes, i.e. the LC-QWP geometry
-- still ran `np.linalg.inv(M)`, `Minv @ P` and `np.linalg.solve(P_inv, M)`.
Replaced by reciprocals and nodal-vector accumulations, **bit-identical**
(`np.array_equal` on all four operators of `twod._axis_ops_1d` at every setting
measured, and on `twod_jones`'s separable `_mass`): `twod._axis_ops_1d`
**3.7×** (200 calls 29.2 → 7.8 ms at n = 33), `twod_jones._tensor_layer_modes`'s
separable component mass **4.8×** (400 calls 41.7 → 8.7 ms), both growing as
O(n³) vs O(n).  (Findings **G10** / G3, P2;
gate `tests/unit/test_audit2609_a13_twod.py::test_g10_diagonal_mass_identities_are_exact`.)

### Performance -- PMM 2-D JAX twins: the dense Kronecker projector pair is gone

`_jax_twod._static_prep` / `_static_prep_cell` and
`_jax_stack2d._layer_static_traced` still built `Tp = kron(Ty, Tx)` and
`pinv(Tp)` -- the `(Nf, N)` pair the NumPy F5 audit deleted -- and then
rebuilt the per-axis projectors three lines later; `_static_prep` additionally
formed a dense `N×N` `diag(1/Mdiag)` and two dense `Minv @ kron(...)` products.
The three preps now build `Tx/Txp/Ty/Typ` once and every jnp sandwich
(`_jax_twod._scalar_jax_tail`, `_jax_twod_jones._proj`,
`_jax_stack2d`'s traced branch) is the two per-axis einsum contractions of
`twod._sandwich_factorized`.  MEASURED: the dropped pair alone is 7.2 / 22.2 /
43.2 MiB at (12, 16, 32) strips per axis (degree 9, n_orders 7/11/11) against
2.5 / 13.1 / 13.7 MiB retained by the whole prep, and **2.4 GiB** at the
documented `_MAX_NODAL_DOF = 150 000` ceiling with n_orders 11 -- plus one
`O(N·Nf²)` pinv and two redundant per-axis pinvs in every case.

It is also a PARITY improvement: the dense `Tp @ Gx0 @ Tpinv` spelling carried
an extra `Ty Typ` factor the NumPy path does not, so the frozen constants are
now **bit-identical** to `_scalar_projected_ops`'s and the pillar entry's
NumPy/JAX `T` parity improves from RMS relative **8.47e-11** to **1.63e-13**
(`R` 2.94e-14; Jones 2.70e-12; AD-vs-central-FD 1.04e-07 against the 1e-4
gate).  `_static_prep*` no longer expose `Tp`/`Tpinv`
(`tests/unit/test_audit_w4_jax_static_caches.py` renamed its key list
accordingly; the cache contract it pins is unchanged).
(Finding **G10**, P2; gate `tests/unit/test_audit2609_a13_jax.py::test_g10_*`.)

### Added -- circular (Lalanne-1997) truncation on `pmm_jones_2d` and `PMM2DStackHybrid`

Circular truncation existed only on `pmm_efficiency_2d[_cell]`.  The projected
operators are functions of the order LIST, so restricting them to the circular
subspace IS the operator built on that subspace; both entries now take
`truncation='circular'` (and so do `prepare_pmm_2d*` and both
`*_vs_wavelength` helpers -- see G8).  MEASURED at n_orders 5, degree 9:
121 → 81 retained orders and 0.216 → 0.068 s on a scalar stack layer,
81 → 49 orders and 0.069 → 0.034 s (in-plane tensor) / 0.117 → 0.051 s
(out-of-plane tensor), 0.200 → 0.082 s on `pmm_jones_2d` with
`symmetry=False`; the tensor→scalar reduction contract still holds under the
circular set (3.9e-14).  `'rectangular'` remains the default and is
bit-for-bit unchanged.  NumPy only -- both JAX dispatches refuse loudly.
(Finding **G10**, P2.)

### Added -- PMM 2-D stack: the eig-cache refusal is surfaced

`LayerCache` is refuse-never-degrade: at its byte budget it returns the modal
set and does not retain it, so the physics is untouched and NOTHING said the
sweep had just lost all modal reuse -- `cache_stats()['eig']['refused']` was
the only signal.  `PMM2DStackHybrid.solve` now warns once per instance when
that counter moves, naming the retained bytes, the budget and the per-entry
cost (~3.7 MB at n_orders 6, ~36 MB at n_orders 11, so a few hundred distinct
wavelengths at production truncation walk past the 5 %-of-RAM budget).
(Finding **G10**, P2; gate
`tests/unit/test_audit2609_a13_stack2d.py::test_g10_eig_cache_refusal_is_surfaced`.)

### Fixed -- staggered PMM 2-D: `eps_cell` is a SEGMENT grid, and an unguarded ~1000× cost cliff now signals

`eps_cell` means two incompatible things across the two 2-D PMM families.  The
HYBRID family takes a PIXEL grid whose redundant rows/columns
`_cell_to_walls_tile` merges away for free, and it has had a cost guard
(`max_nodal_dof`) all along.  The STAGGERED family
(`pmm_efficiency_2d_staggered`, `pmm_jones_2d_staggered`,
`PMM2DStackPure.add_layer`) takes a SEGMENT grid where every row and column IS
an element and the generalized pencil is `2·Nx(M−1)·Ny(M−1)` -- and it had NO
cost guard at all.  MEASURED on the same 12×12 half-fill-pillar array (3
distinct strips per axis, expressible on a 4-segment lattice):
`pmm_efficiency_2d_staggered(degree=6)`
**498 s CPU / 8.4 GB** (a 7200×7200 QZ pencil where 800×800 suffices),
`PMM2DStackPure(n_modes=5).add_layer` **1096 s / 3.9 GB**, against **0.22 s /
< 1 GB** for the same array through `pmm_efficiency_2d_cell` -- and neither
staggered call raised, warned or printed anything.

Added (`lumenairy/elements/pmm/twod_staggered.py`,
`lumenairy/elements/pmm/stack2d_pure.py`):

* `max_pencil_dof` (default 12 000, ~28 GB projected) on all three surfaces,
  the SEGMENT-grid sibling of the hybrid's `max_nodal_dof`.  It refuses with
  the pencil dimension, the projected footprint and the PIXEL-vs-SEGMENT
  distinction in the message.  Every grid in the shipped staggered suite is far
  below it (largest: 8 segments/axis at M = 8, pencil 6272; the documented
  M = 10 / 3-segment solve is 1458).
* a REDUNDANCY warning naming the merged strip count (computed with
  `_cell_to_walls_tile`'s own merge rule) AND the grid the caller should
  actually pass.  Those are not the same number, and that distinction is the
  care this guard needed: the audit's 12×12 pillar has 3 distinct strips but
  its walls sit at indices 3 and 9, i.e. at 1/4 and 3/4, which a 3-segment
  uniform lattice (walls at 1/3, 2/3) does NOT contain — "re-express it on a
  3×3 grid" would silently change the DUTY CYCLE, the same failure mode as the
  sibling `period_x` finding.  The suggestion is therefore the smallest SQUARE
  UNIFORM lattice that holds every wall on either axis,
  `N / gcd(N, all wall indices)` = **4**, and the message reads "your grid has
  12×12 segments and only 3×3 DISTINCT strips … the SAME geometry — the same
  walls, to the segment — is expressible on the uniform 4×4 lattice at pencil
  512, 729× less QZ time and 81× less memory".  It warns rather than refuses
  because splitting a region into more segments is a legal h-refinement.  A
  grid that reduces to 1×1 is exempt: tiling a uniform axis into equal segments
  is the documented way to satisfy the `Nx == Ny` contract.
* PIXEL-vs-SEGMENT cross-references in both families' `eps_cell` docs
  (`twod._cell_to_walls_tile`, `pmm_efficiency_2d_cell`,
  `pmm_efficiency_2d_staggered`, `pmm_jones_2d_staggered`,
  `PMM2DStackPure.add_layer`), the way CONVENTIONS §11 cross-references
  `fff_nv`.

(Finding **G9**, P2; gate `tests/unit/test_audit2609_a13_staggered_cost.py`,
8 tests, 0.12 s -- every assertion is on the guard, never on a solve.)

### Fixed -- PMM 2-D stack: `symmetry` is in the modal cache key, and the lattice periods are frozen

Two cache/attribute gaps the W7 A11 hardening missed
(`lumenairy/elements/pmm/stack2d.py`; finding **G12**, P3):

* `symmetry` was absent from `_mode_key` although `_build_layer_modes` passes
  `block_eig=self.symmetry` to `_tensor_layer_modes`, where it selects between
  the parity-sign block reduction and the dense 4Nf zgeev.  Mutating it between
  solves served the stale modal set: measured on an out-of-plane tensor pillar,
  the re-solve came back BIT-IDENTICAL to the pre-mutation answer
  (`|J_b − J_a| = 0.0`) while a fresh object differed by 7.16e-16.  (`truncation`
  is in the key for the same reason.)
* `period_x` / `period_y` were plain attributes, but `add_layer` has already
  frozen the walls in METRES -- so a later period change re-solved the OLD
  walls at a NEW period, i.e. a silently different DUTY CYCLE: measured
  `|J_reused − J_fresh(same duty)| = 4.390e-01` against the parameter's true
  sensitivity 6.492e-01, while every other public attribute in the same sweep
  read 0.0.  They are now read-only properties once a layer exists (assignment
  raises with the measurement); before the first `add_layer` they are still
  settable.  The class docstring documents which attributes are mutable
  between solves and why.

Gate: `tests/unit/test_audit2609_a13_stack2d.py::test_g12_*`.

### Changed -- documentation only

* `twod._assemble_2d` is labelled a KEPT REFERENCE IMPLEMENTATION, on no live
  solve path, naming the two files that use it as the dense-kron oracle
  (finding **G13**).
* `twod._layer_modes_projected`'s docstring no longer says "the PMM2DStack path
  keeps the legacy assignment" -- every in-tree caller now routes the pair.
* the two dead `# noqa: F401` re-exports of `is_jax_array` in
  `_jax_twod.py` / `_jax_twod_jones.py` are deleted (the dispatch imports it
  from `...backend` directly) (finding **G13**).
* `PMM2DStackHybrid`'s `cascade` docstring carries the measured `'fused'`
  comparison (agreement `max|dR| ≤ 2.0e-14` / `max|dT| ≤ 1.0e-13` /
  `max|dJ| ≤ 8.7e-14`, never bit-identical; whole-solve median 1.04×-1.15× over
  5 interleaved runs on three fixtures).  The default stays `'fast'`: a 1e-13
  move of every user's bits is not paid for by a few per cent (finding
  **G10**, measured and NOT adopted).

### Added -- PMM 2-D side of the branch-cut bound (G11 landed in WP-A14)

`rcwa/_core._sqrt_decay`'s on-cut predicate tested proximity of `Re(r)` to the
ORIGIN rather than to the IMAGINARY AXIS, so it could flip a near-zero
EVANESCENT mode and return `Re(lam) < 0` -- a forward propagator that grows.
The predicate is WP-A14's and now carries `& (Im(r)^2 > Re(r)^2)`;
`tests/unit/test_audit2609_a13_twod.py::test_g11_no_layer_mode_grows_beyond_the_branch_cut_band`
pins the PMM-2D side of it in two layers: the FIXED predicate asserted directly
on the audit's counterexample (`_sqrt_decay([1e-20 - 1e-30j])` read
`-1.000000e-10 + 5.000000e-21j` before and `+1.000000e-10 - 5.000000e-21j`
after, so `Re(lam) >= 0`), AND the analytic bound
`|exp(-lam k0 L)| <= exp(band * max|lam| * k0 * L)` (~ 1 + 5.6e-7 in the regime
these entries run) measured on a real PMM-2D layer spectrum -- which held
before the predicate change and still holds after, so the file keeps saying
something true if the predicate is ever re-tuned.

### Changed -- PMM 2-D: every Wood-anomaly nudge now names the entry that took it

WP-A14 made `_grazing_safe_wavelength` announce its substitution through a
`lumenairy.elements.rcwa.WoodNudgeWarning` and accept `fn_name=`.  All eight
PMM 2-D call sites now pass it, so the message reads
`pmm_efficiency_2d_cell: a diffracted order sits EXACTLY at cut-off ...`
instead of `_grazing_safe_wavelength: ...`:
`pmm_efficiency_2d[_cell]` (through `_pmm2d_solve_core`'s own `fn_name`),
`prepare_pmm_2d[_cell](...).solve`, `pmm_jones_2d`, `PMM2DStackHybrid.solve`,
`PMM2DStack.solve` (the jnp twin), `pmm_efficiency_2d_staggered`,
`PMM2DStackPure.solve`, and the JAX host guard (which forwards its caller's
name).  Numerics are unchanged -- the same 1e-7 relative step on the same 1e-9
detection.  A test that deliberately solves ON an anomaly should filter that
category; no PMM 2-D test needed one (the whole PMM-2D suite passes with the
warning live).
