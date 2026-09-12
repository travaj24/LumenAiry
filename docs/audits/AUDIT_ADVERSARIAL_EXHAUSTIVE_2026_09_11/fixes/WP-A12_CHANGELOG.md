# WP-A12 changelog text (PMM 1-D / stack / core + CONVENTIONS §7.1)

Assembled from findings G1-G4 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` Section 11 and the
partition report `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/PMM-1D.md`.  Every number below was
re-measured on this branch: the "before" readings come from running the audit's own repro scripts
(`repro/PMM-1D/p11_jax_guards.py`, `p3e_band2.py`, `p10e_arb_count.py`, `p12_jones_basis.py`) on
HEAD before the change, and the "after" readings from re-running them.  Performance claims are
deterministic **solve counts** and **bit-identity A/B**, never wall-clock: this workstation was
running ~20 sibling agents throughout and the audit records three failed timing probes to say so.

---

### Fixed -- pmm: the differentiable `PMMStack.solve` twin returned before EVERY guard, and returned silently-wrong answers where NumPy raises (G1, P1)

`PMMStack.solve` dispatched to `pmm/_jax_stack.py` and **returned** before
`_require_propagating_incidence`, before the cross-layer sliver screen and before
`_warn_stack_energy`.  `_jax_stack.py` contained zero occurrences of any of those names; its only
guard was a grazing check on `|kz_inc| < 1e-9`.  Measured on the audit's two-layer Si/SiO2 fixture,
routing to the twin by making one layer's `eps` a `jnp` array while every other input stays a plain
Python value:

| case | NumPy branch | JAX branch, before | JAX branch, after |
|---|---|---|---|
| gain superstrate `n_sup = 1 - 1e-3j` (fully **concrete**) | `ValueError: gain incidence medium ...` | returns `R+T = [-0.848, -0.863]` -- **negative efficiencies, silently** | the **identical** `ValueError`, string for string |
| manufactured sliver `s = 1.5e-5` of the period, degree 14 / 16 / 18 (`min_feature` PINNED at `period * 1e-5`, see below) | `ValueError` (sliver refusal) | `T0(E_y) = 1.3417`, `max R+T = 8.35` -- a 735 % energy violation, **no warning** | same number, now under **two** warnings: the geometric sliver screen and `energy not conserved (max R+T = 8.35 > 1)` |
| same, degree 12 / 20 | returns 0.76587 | returns 0.76589 | returns 0.76589 under the geometric screen warning alone |

The sliver rows are measured with `min_feature` **pinned at `period * 1e-5`**, the pre-2026-09-12
default.  That has to be said because the two findings interact: at the default this release raises
to `period * 1e-3` (G2 below) the same `s = 1.5e-5` collision is SNAPPED AWAY, the fixture carries
no sliver at all, and the audit's reproducer reads `T0 = 0.7658976`, `tot = 1.000000` silently on
both branches at every degree 12-20.  That is the G2 cure acting on the G1 reproducer, not the G1
guard standing down; the guard itself is exercised at the old threshold, above and in the
regression tests.

The gain case is **exactly the audit-M3 2026-07-25 defect** the NumPy path was fixed for, still
alive on the twin -- and `n_superstrate` is fully concrete there, so the documented "a TRACED value
skips the guard" carve-out never covered it.  The sliver case is worse than the gain case because it
is *degree-dependent*: it passes any spot check the user makes at a neighbouring degree.  This is the
differentiable design loop -- the LC-QWP inverse-design path this library exists for.

Three things now run on both branches:

* the concrete incidence guard (`_jpmm_concrete_incidence_guard`) is hoisted **above** the traced
  dispatch, so a gain / evanescent / metallic incidence medium raises identically on both;
* the **pure-geometry** sliver screen runs before the twin and WARNS.  It reads only wall
  coordinates, `min_feature`, `degree` and `period`, every one of which is a concrete host value on
  the differentiable path (the twin freezes the geometry), so it is trace-safe;
* `_warn_stack_energy` runs on the returned arrays whenever they are **concrete** -- an ordinary
  eager `solve()` -- raising on a non-finite or negative total and warning above the super-unity bar.

What still does not run, and is now stated in the `PMMStack.solve` docstring rather than nowhere:
under `jit`/`grad` the outputs are Tracers, so the energy tripwire is skipped (its comparison cannot
be taken without severing the trace); a *traced* `n_superstrate`/`angle` skips the incidence raise
for the same reason; and the three-solve sliver **arbiter** -- the only thing that can turn a screen
hit into the NumPy path's `ValueError` -- is never available under a trace, so a sliver stack WARNS
here where NumPy refuses.  The docstring previously asserted the opposite (*"same physics ... 
forward-identical to NumPy at ~1e-15"*) and listed only slant / out-of-plane / `stabilize` /
`retain_internal` / sweep as raising.

Parity and gradients are unaffected (they were never the problem): NumPy-vs-JAX per-order
efficiencies agree to 1.2e-15 on the audit's own over-capacity ladder, a clean stack fires **no**
warning, and `jax.grad` w.r.t. `eps_ridge` matches a central difference to **2.0e-8 relative**
(AD -0.02222695101, FD -0.02222695056).

Tests: `tests/unit/test_audit2609_a12_pmm1d.py::test_g1_*` (4 tests, including a trace-safety test
that the screen still fires under `jax.grad` while the tripwire correctly stands down).

---

### Changed -- pmm: `PMMStack`'s default `min_feature` is now `period * 1e-3` (was `period * 1e-5`), which is ABOVE the measured sliver hazard band instead of at its bottom (G2, P2)

The union-grid wall snap is the only thing that removes a manufactured cross-layer sliver before it
reaches the solve, and the sliver pathology has a *measured width* that is ABSOLUTE rather than a
multiple of the knob: an unsnapped collision of size `s` corrupts the answer for `s` at roughly
**1e-5 .. 1e-4 of a PERIOD** (both fixtures below, degrees 10-26) and is harmless outside that.
That is what makes raising the threshold a cure at all -- a band that scaled WITH `min_feature`
could never be cleared by raising it, and the ladder below shows it does not, because at
`period * 1e-3` the rungs at 1x .. 8x of the threshold are clean where at `period * 1e-5` they were
the whole hazard.  The old default therefore snapped away only the collisions that were already
harmless and left the entire dangerous decade exposed -- and with the refusal armed (the shipped
default) a staircased stack at `s = 1.5e-5` of the period was **REFUSED at every degree from 10 to
24**, i.e. could not be solved at all at the library default.

Re-measured on the audit's second, independent fixture (TiO2-like 2.35/1.46, 0.55 um pitch, 0.70 um,
31 deg, two layers, refusal disarmed), sweeping `s` over a 0.3x..100x ladder of `min_feature` at
degrees 10/14/18/22/26 and scoring **degree-SCATTER at fixed `s`** (a smooth drift with `s` is a
genuinely different geometry and is correct physics; an answer that jumps between branches as
`degree` changes is the pathology).  Reference `T0 = 0.199229666`:

| `min_feature` | rungs of 11 showing degree-scatter |
|---|---|
| `period * 1e-5` (the old default) | **6 of 11** -- every rung from 1x to 8x; `T0` reads 0.2645 / 0.3082 / 0.1939, up to 55 % wrong and scattering by +-5 % between ADJACENT degrees |
| `period * 1e-4` | **1 of 11** (only the 1.0x rung, and at one degree of five) |
| `period * 1e-3` (the new default) | **0 of 11** -- every rung degree-independent to 7 digits |

and where two settings both leave a collision unsnapped they agree **exactly**: `s = 1.5e-4` reads
0.19839028 under `1e-5` and `1e-4`, `s = 3e-4` reads 0.19755266 under both, `s = 1e-3` reads
0.19366045 under `1e-5` and `1e-3`.  Raising the knob does not perturb the cases it does not touch --
it only removes cells the union manufactured.  On the audit's first fixture (Si/SiO2, 1.0 um pitch,
1.55 um, 12 deg) the same band reaches `T0 = 22.38` and `147.67` against a correct 0.7577.

`1e-3` is also the scale the sliver refusal itself already prescribes as its first remedy
(`mf_fix = 2 * w_wide * P`).

**Migration.**  A caller who relied on the old value -- for example to keep a deliberate sub-nm
cross-layer offset in the grid, or to reproduce a number measured before this release -- gets it
back with `min_feature=period*1e-5`.  The snap moves walls by at most `min_feature/2`, so a stack
**with** colliding cross-layer walls can now see its solved geometry differ from the requested one
by up to 5e-4 of a period; a stack **without** them is byte-identical (no pair inside the threshold
-> identical grid -> identical answer).  The snap has always been cross-layer-pairs-only: a close
pair a single layer owns (an intentional thin liner) is never thinned, whatever this is set to
(verified across liner widths 1e-2 .. 1e-6 of a period: byte-identical union grids at the old and
the new default).

**What that costs in ANSWERS, and what it now says out loud.**  Where a collision exists and is
snapped, the number moves -- measured on the audit's second fixture at `s = 3e-4` of the period,
`T0 = 0.19829790` on the snapped grid against `0.19755266` unsnapped, i.e. **3.8e-3 relative**.  And
because the threshold is two decades higher, `_pmm_union_grid` now WARNS
(`_pmm_union_grid: snapped N pair(s) of NEAR-COINCIDENT cross-layer walls closer than
min_feature=... (period fractions): ...`) on stacks that were previously silent -- any stack whose
cross-layer walls sit between 1e-5 and 1e-3 of a period apart.  That warning is not new code; it is
the existing snap notice reaching a population it did not reach before, and it is the intended
signal: it names the pairs and the maximum wall displacement, so the caller can see how far the
solved geometry has drifted from the requested one.  A caller who needs the requested geometry
verbatim passes `min_feature=period*1e-5` (or smaller); a caller who wants the snap but not the
notice should NOT filter it blind -- check first that the displacement it reports is below the
accuracy they need.

`PMMStack.__init__` now documents `min_feature` in its `Parameters` section -- it had no entry at
all -- and states the **collision-scale rule** the caller should actually reason with: the threshold
must sit ~10x ABOVE the geometry's own cross-layer collision scale, and that scale is not a period
fraction (for a taper it is `(thickness / n_slices) * tan(sidewall)`, in nanometres, independent of
the period).  The period-scaled default is a convenience, not a derivation.

Tests: `tests/unit/test_audit2609_a12_pmm1d.py::test_g2_*` (4 tests: the value, the geometric band
sweep, the numeric degree-scatter cure on the second fixture, and the bit-identity of what the
larger snap does not touch).  Three existing fixtures that inherited the old default and *need* the
sliver to survive now pin `min_feature` explicitly and say why:
`tests/unit/test_fix_pmmstack_sliver_walls.py` (the O-11 hazard-band fixture),
`tests/unit/test_pmm_m2_window_contract.py::_mk` (whose T3-1 measurement asserts the snap is inert)
and `tests/unit/test_pmm_m3_efficiency.py`'s T3-4 sweep-ordering fixture.

---

### Performance -- pmm: the sliver arbiter's extra solves are lazy; the exactly-diagonal GLL masses take an O(n) path; `Q @ W2` is built once (G3, P2)

All four changes are **bit-identical**, verified by an in-process A/B over 75 arrays spanning single-
layer Jones (normal / oblique / metal), the scalar `te`/`tm` entries, conical, four multilayer stacks
(shared grid, anisotropic, lossy, per-layer, normal-incidence and slanted) and the modal kernels
themselves: **0 of 75 arrays differ, worst |A - B| = 0.000e+00**.

**The sliver arbiter: 4 solves -> 2 on a `truncation` verdict.**  Counted deterministically by
wrapping `PMMStack.solve` itself (not a `_core` helper -- `stack.py` binds those names at import,
which is the instrumentation bug the auditor caught and recorded):

| n_slices | sliver `s` | solves before | solves after | guard OFF | verdict |
|---|---|---|---|---|---|
| 4 | 3e-4 | 4 | **2** | 1 | truncation |
| 8 | 3e-4 | 4 | **2** | 1 | truncation |
| 8 | 1e-4 | 4 | 4 | 1 | refused (needs `d12`) |
| 16 | 5e-5 | 4 | 4 | 1 | refused (needs `d12`) |
| 4 | 0 | 1 | 1 | 1 | *control -- no sliver* |
| 1 | 0 | 1 | 1 | 1 | *control -- single layer* |

`d12` -- the device's own measured sensitivity to where the contested wall sits -- is read only on
the `'sliver'`/`'wall'` fork, and that fork is reached only after `d0` has cleared the geometric
floor, so the two `_sliver_collapse_solve` calls are now paid only when they are used.  On a
`'truncation'` verdict `ev['d12']`, `ev['d0_over_d12']` and `ev['closed_super_unity']` are `None`:
they were not measured, and reporting a number nobody computed would be worse than reporting none.
No verdict and no returned number changes -- only how many solves are paid to reach them.

`PMM_SLIVER_ARBITER_LAZY` (module-level, default `True`) is the fail-before switch: `False` restores
the eager form bit for bit -- all three solves, every evidence field populated -- for an A/B or for a
diagnostic that wants the wall sensitivity on a row the criterion did not need it for.  Three tests
whose SUBJECT is that denominator now ask for it through the switch and keep every number they
measured (`test_fix_pmmstack_sliver_walls_round2.py` x2, `test_fix_pmmstack_sliver_round4.py` x2),
and the arbiter's cost test is restated to the per-verdict contract with the pre-lazy count
re-derived through the switch instead of quoted.

*Correcting the audit on two points.*  (1) The finding says memoizing the verdict means "a sweep pays
it once".  It does not and cannot: `wl` is part of the determinant of what the re-solves read, so
each point of `solve_vs_wavelength` is genuinely different physics and would miss by construction.
(2) A memo keyed on that determinant WAS implemented and then **removed**: its only remaining
beneficiary was a repeated *identical* `solve()` (measured 10 -> 6 and 20 -> 8 invocations over five
repeats), and against that it made the arbiter's re-solves invisible to the three shipped contracts
that instrument `_sliver_probe_solve` to assert WHICH source it is handed -- a worse trade than the
saving.  See the WP report for the design a future attempt should use instead.

**Diagonal GLL masses.**  GLL mass lumping makes every nodal mass operator structurally diagonal
(`Mloc = diag(ref_w * J)`; the only overlaps are shared element-boundary nodes), and the geometric
mass `S0` additionally real and positive.  `_safe_inv`, `_safe_solve` and the seven `inv(S0) @ X`
products inside `_sem_modes_tensor` now take an `O(n)` reciprocal / row scale on an exactly-REAL
diagonal, and the dense LAPACK path otherwise.  The restriction to *real* diagonals is measured, not
cautious: over 1,000 random diagonals (positive; mixed-sign; `float64` and `complex128`; magnitudes
spanning 16 decades including the 1e-14 sliver-element regime) `inv(D)` vs `diag(1/d)`,
`solve(D, B)` vs `B / d[:, None]` and `diag(1/d) @ B` vs `(1/d)[:, None] * B` differ in **0 of
1,000** trials -- while on genuinely COMPLEX diagonals they differ in **1,000 of 1,000** by 2-4e-16
relative, because LAPACK's complex division and NumPy's are not the same last bit.  A lossy `1/eps`
mass therefore stays on the dense path and keeps its exact shipped arithmetic.

**`Q @ W2` once.**  `_sem_modes_tensor` built the same `(2n)^3` product twice -- for the probe
partner `V0` whose flux picks the forward set, and again for the returned `V2`, which differs only
by the per-column branch flip already folded into the divisor.  It is now named once.  The trailing
`@ np.diag(...)` stay gemms deliberately: `A @ np.diag(v)` and `A * v[None, :]` agree only to ~2e-16
relative for complex `v` (measured), and `lam` is complex, so collapsing them would have traded
bit-identity for one column scale.

**`_sem_fourier_projection`** scatters with `np.add.at` instead of a Python loop over
`degree + 1` global nodes.  `l2g` repeats a global index at the periodic wrap, so the accumulation
order matters; `np.add.at` accumulates in index order -- the same order the loop used -- and is
bit-identical (0 of 200 random layouts differ) and **1.45-2.6x** faster over
`(degree, n_el) = (12, 4) .. (32, 20)`.  The JAX twin's static copy takes the same change.

**`_grazing_safe_wavelength`** (WP-A14's change) now emits a `WoodNudgeWarning` whenever it nudges
a wavelength off an exact Rayleigh cut-off.  Every 1-D PMM caller of it now passes `fn_name=`, so the
warning names the entry point the user actually called (`pmm_efficiency_1d`, `pmm_jones_1d`,
`pmm_efficiency_1d_segments`, `pmm_jones_1d_segments`, `pmm_jones_1d_conical`,
`pmm_jones_1d_conical_tensor`, `PMMStack.solve (conical)` and the shared `_conical_nodal_solve`
label) rather than a private helper.  Numerics unchanged -- the nudge fires only on a wavelength
within ~1e-9 of a cut-off, and away from one the call is silent and returns the wavelength unchanged.

**`_GEO_EIG_CACHE`** is keyed on a 32-byte blake2b digest instead of the full operator bytes -- at a
production `n_glob` = 300 the old key was ~1.4 MB of complex128 *retained per entry*, beside a
~1.4 MB value, plus an `O(n^2)` copy on every lookup -- and is now a `ByteBudgetedLRU`
(`pmm_geometric_eig`) bounded by `LUMENAIRY_CACHE_BUDGET_MB` and drained by `clear_asm_caches()`,
as its sibling `_PERLAYER_GEO_CACHE` already was.  The 64-entry count cap is gone: bytes, not
entries, are the resource being protected.

**Speed, reported honestly.**  On a 4-layer fixture the deterministic call counts read **22 -> 14**
dense `np.linalg.inv` per solve (`solve` unchanged at 10, `eig` at 5), worth 0.77 % of the
inv/solve/eig flop total; the gemm side (7*n^3 of row scales per patterned layer plus 8*n^3 from the
`Q@W2` reuse, against ~344 n^3 per layer) is ~4 % by the flop model and is not separately measurable
at run time (the `@` operator does not route through `np.matmul`).  An **interleaved** min/median-of-9
wall-clock A/B on this box reads OFF/ON = 0.985-1.004 (min) and 0.948-1.182 (median) across degree
10/14/20/26 -- indistinguishable from contention noise -- and `tracemalloc` peaks are identical to
the byte (13.01 MB / 44.74 MB at degree 14 / 26).  **No speedup is claimed**: what is established is
bit-identity, 8 fewer dense inverses per solve, 2 fewer whole-stack solves on a truncation verdict,
and a cache key that no longer retains megabytes.

Tests: `tests/unit/test_audit2609_a12_pmm1d.py::test_g3_*` (7 tests -- the solve counts with the
pre-lazy control flow re-derived in-process through the fail-before switch, the digest key, the
diagonal kernels against LAPACK, an end-to-end fail-before A/B with the fast path neutralised, the
projection scatter, the no-sliver scope control, and the quadratic-in-wavelength pencil).

---

### Changed -- pmm: `PMMStack.internal_field(pol=...)` takes the family's `'te'`/`'tm'`/`'s'`/`'p'` spellings (G4, P3)

`internal_field(..., pol='x')` raised `ValueError: pol must be 0 or 1`.  CONVENTIONS §7 pins that the
`s`/`te` and `p`/`tm` aliases are "accepted everywhere (case-insensitive)"; this was the one place in
the PMM surface that was an index into the Jones columns instead.  `'tm'`/`'p'` now select the
incident `E_x` row and `'te'`/`'s'` the incident `E_y` row -- the same rows as `R_eff[0]`/`R_eff[1]`
-- case-insensitively and whitespace-tolerantly; the integers `0`/`1` still work unchanged; an
unknown spelling raises with the §2 `fn_name: ` prefix and names every accepted value.

### Changed -- pmm: `set_source(angle=A, theta=T)` with `A != T` is no longer silent (G4, P3)

The "alias wins, no equality check" rule is a deliberate, test-pinned cross-suite contract and does
**not** change -- `theta` still wins, and the resolved number is unchanged.  What changes is that two
*different* non-zero angles in one call, which has no legitimate reading, now emits a warning naming
both values and the one that won.  The warning is gated on `angle != 0` because an ordinary
`theta=`-only call leaves `angle` at its `0.0` default and is indistinguishable from an explicit
zero.  (A raise would be strictly safer and is what the audit recommends, but it has to land in the
RCWA resolver at the same time or the two suites stop agreeing -- which is the property
`test_v5_12_0_naming_aliases` exists to pin.  Recorded as a requested cross-suite change.)

### Fixed -- pmm: `PMMStack.prepare().solve()` silently dropped propagating orders (G4, P3)

The far-field order-budget block was copy-pasted **13 times** across `_core.py`, `stack.py` and the
JAX twin -- the exact multi-copy shape this codebase's own audits blame for its three worst recent
defects (the six-copy factor-i defect, the six-copy `_sqrt_decay` branch-cut defect, and the T3-3
conical order-cap defect, which was *one copy of this very block* capping from the wrong grid).  All
thirteen are now one `_core._farfield_order_set(...)` helper, and consolidating them surfaced a
divergent copy: `_PreparedPMMStack.solve` clamped the projector to the nodal capacity but **never
refused** when the propagating orders did not fit inside it, so the prepared path returned a far
field with orders missing -- sub-unity power, which the one-sided energy tripwire cannot see.  It now
raises like every sibling path.  The refusal wording is unified across the thirteen sites (the four
minority spellings differed only in prose; none was asserted anywhere).

### Fixed -- docs: three PMM 1-D performance / convergence claims that the measurements contradict (G4, P3)

* `_core.py`'s headline *"the dominant cost of a PMM solve is the dense `np.linalg.eig` ... (~85% of
  runtime)"* is replaced by what is actually known: a cProfile of an 8-layer degree-24
  `PMMStack.solve` counts **42 `np.linalg.inv` / 18 `np.linalg.solve` / 8 `np.linalg.eig`** calls,
  and an arithmetic flop model puts the eig at ~58 %, the interface at ~14 % and the Redheffer star
  at ~19 % per layer -- i.e. a third of the work is in the interface and star **inverses**.  The note
  now also says, in as many words, that the empirical split is **unmeasured**: every wall-clock
  attempt was destroyed by machine contention (a 144x144 complex eig timed at 490 ms against a ~5 ms
  expectation; `inv(384)` timed *faster* than `inv(320)`), so the percentages are a flop model and
  should be re-measured on a quiet machine before anyone optimises against them.
* `elements_per_region > 1` + `grade=True` was documented as *"the speed lever for TM
  (hp-refinement)"*.  Measured on the Au (0.18 + 3.43j) / air corner case it is advertised for,
  against an `R_inf + c/n` extrapolation of the library's own RCWA at `n_orders` 401/601/801/1201,
  single-element p-refinement beats it by **2.5-4x at matched DOF** (48 DOF: 4.36e-5 vs 1.07e-4;
  96: ~5.6e-6 vs 2.00e-5; 192: 9.90e-7 vs 1.23e-6), and the eig cost goes as DOF^3.  Grading does
  beat non-grading at fixed element count (1.5-3x), which is why it stays the default *when the knob
  is raised at all* -- but uniform-degree grading is not hp-refinement, and the docstring now says
  so and carries the ladder.
* `pmm_jones_1d` claimed *"Converges SPECTRALLY in the polynomial `degree` with no accuracy floor"*
  without qualification -- and `pmm_jones_1d` is the physics `PMMStack` runs.  True for `E_y` (TE):
  9.4e-6 / 2.6e-7 / 1.8e-8 / 4.9e-9 at degree 8/12/16/20 on a metal lamellar cell, a local rate ~9
  and rising.  False for `E_x` (TM), where the wall corner -- not the metal -- limits it:
  5.4e-4 / 2.2e-4 / 1.1e-4 / 4.4e-5 / 2.1e-5 / 8.0e-6 / 1.9e-6 at degree 8/12/16/24/32/44/60, and on
  a *lossless* high-contrast n = 3.48/1 cell a flat `O(N^-2.7)` out to degree 44 with no acceleration
  at all.  At degree 28 the TM order-0 error is ~2.4e-6 where TE is ~2.8e-11 -- five orders apart on
  the same cell.  Both ladders are now in the docstring, with the same caveat added to `PMMStack`'s
  class docstring.
* `pmm_efficiency_1d_slanted` / `pmm_jones_1d_slanted` now record the slanted-TM **energy-closure
  floor**: on a *lossless* slanted cell the inclined-coordinate generator reads `ΣR + ΣT` =
  1.0000908 / 1.0000604 / 1.0000432 at degree 16 / 22 / 28, super-unity by 4-9e-5 and *decreasing*
  with degree (so it converges and is not an instability), where the vertical cascade on the same
  solid reads 1.0000000000.  Anyone using `|ΣR + ΣT - 1|` as a tripwire on a slanted TM stack must
  calibrate against that floor rather than against 1e-10.  TE is unaffected (`|tot - 1| <= 1.5e-8`).

### Fixed -- CONVENTIONS §7.1 named the wrong basis for the 1-D Jones, and the mislabel hid a sign (G4, P3)

§7.1 said *"The 1-D solvers return `te`/`tm` (`s`/`p`)"* and that at `phi = 0` the bases coincide "up
to the `tm` <-> `x`, `te` <-> `y` identification".  Measured on a uniform slab (n = 2.1, d = 0.32 um,
lambda = 0.55 um, n_sup = 1, n_sub = 1.5) against an independent analytic TMM:

| angle | `pmm J[0,0] / r_p` | `pmm J[1,1] / r_s` | `rcwa J / pmm J` |
|---|---|---|---|
| 0 deg | **-1.000000 + 0.000000j** | +1.000000 | +1.000000 |
| 30 deg | **-1.000000 - 0.000000j** | +1.000000 | +1.000000 |
| 60 deg | **-1.000000 - 0.000000j** | +1.000000 | +1.000000 |

Both 1-D Jones solvers return the **lab Cartesian `(E_x, E_y)`** basis (their own docstrings say so),
whose `xx` entry is `-r_p` in the standard Fresnel p convention; at normal incidence they correctly
give `J_xx = J_yy` (lab-frame isotropy), which a true `te`/`tm` matrix would not.  The library is
internally consistent -- PMM and RCWA agree to 1e-15 -- but §7.1 is the declared source of truth and
said something different, so a consumer taking it literally picked up a sign on the p row/column at
`phi = 0`.  §7.1 now states, for 1-D **and** 2-D, that the returned Jones is the lab Cartesian basis
with index 0 = `x` and index 1 = `y`, that at `phi = 0` the `x` column is the p channel *up to the
sign of the p unit vector* (`J[0, 0] = -r_p`), and that at conical incidence the `te`/`tm` matrix is
the lab one conjugated by the rotation.  This is the same §7.1 defect the PMM-2-D partition found
from the other side (G13).

Tests: `tests/unit/test_audit2609_a12_pmm1d.py::test_g4_*` (6 tests, including the §7.1 statement
checked against an analytic three-medium TMM written in the test rather than against library code,
and a DISCOVERED sweep that fails if copy fourteen of the order-budget block ever ships).
