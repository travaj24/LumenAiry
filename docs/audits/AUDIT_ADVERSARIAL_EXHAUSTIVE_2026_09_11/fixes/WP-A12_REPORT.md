# WP-A12 report — PMM 1-D / stack / core (`elements/pmm/_core.py`, `stack.py`, `oned.py`, `conical.py`, `_jax_stack.py`) + CONVENTIONS §7.1

Branch `audit-fixes-2026-09`.  Findings G1–G4 of `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11` §11 and the
partition report `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/PMM-1D.md`.

Every number below was re-measured on this branch.  "Before" readings come from running the audit's
own repro scripts on the pre-change tree; "after" from re-running them.  Performance is reported as
**deterministic solve / call counts** and **bit-identity**, never as wall clock: this workstation was
running ~20 sibling agents throughout, and the audit records three failed timing probes to say why.

---

## 1. Summary

| ID | Status | Files : lines | Tests | Oracle | Measured before → after |
|---|---|---|---|---|---|
| **G1** (P1) | **fixed** | `pmm/stack.py` : `solve` dispatch (`_jpmm_concrete_incidence_guard` hoisted above `_holds_traced()`; `_warn_jax_stack_sliver` + `_warn_stack_energy_concrete` added), `_sliver_screen(allow_traced=)` | `tests/unit/test_audit2609_a12_pmm1d.py::test_g1_*` (4) | the NumPy branch (same stack, only the eps dtype differs); `jax.grad` vs central FD | gain `n_sup = 1−1e-3j`: JAX returned `R+T = [−0.848, −0.863]` **silently** → **identical `ValueError`**, string for string.  Sliver `s = 1.5e-5`: `T0 = 1.3417`, `max R+T = 8.35`, **no warning** → same number under **two** warnings (geometric screen + `energy not conserved`).  Clean stack: 0 warnings, NumPy-vs-JAX `≤ 1.2e-15`; `grad` vs FD `2.0e-8` rel. |
| **G2** (P2) | **fixed** | `pmm/stack.py` : `_MIN_FEATURE_DEFAULT_FRAC = 1e-3`, `PMMStack.__init__` | `::test_g2_*` (4) | the shipped `_cross_layer_sliver` screen; a degree-scatter sweep on the audit's *second* fixture | degree-scattering rungs of 11: `1e-5·P` **6** → `1e-3·P` **0**; where both settings leave a collision unsnapped they agree **exactly** (bit-identical at `s` = 2e-3, 5e-3) |
| **G3** (P2) | **fixed**, one sub-item **not reproducible as proposed** and one **implemented then withdrawn** | `pmm/_core.py` : `_real_diagonal`, `_row_scale_apply`, `_safe_inv`, `_safe_solve`, `_sem_modes_tensor`, `_uniform_geo_eig`, `_scalar_uniform_geo_eig`, `_geo_eig_key`, `_cached_geo_eig`, `_GEO_EIG_CACHE`, `_sem_fourier_projection`, `_jpmm_fourier_projection`; `pmm/stack.py` : `_sliver_arbiter`, `PMM_SLIVER_ARBITER_LAZY` | `::test_g3_*` (7) | deterministic solve counts (wrapping `PMMStack.solve`); LAPACK on the same matrix; an in-process fail-before A/B | arbiter on a `truncation` verdict **4 → 2** solves (4 → 4 where the verdict genuinely needs `d12`); 8 of 22 dense `inv` calls removed per 4-layer solve; geo-eig key `n_glob²·16 B` → **32 B** + byte-budgeted; **0 of 75** A/B arrays differ; wall clock indistinguishable from noise (see §2 G3) |
| **G4** (P3) | **fixed** (one item **partially**, see §5) | `pmm/_core.py` : `_farfield_order_set`, `_resolve_incidence`, the perf note; `pmm/stack.py` : `_resolve_internal_pol`, `internal_field`, 7 order-budget sites, `PMMStack` docstring; `pmm/oned.py` : `pmm_jones_1d`, `pmm_efficiency_1d`, both slant entries; `CONVENTIONS.md` §7.1 | `::test_g4_*` (6) | an analytic three-medium TMM written in the test; a measured TE/TM convergence-rate ladder; a DISCOVERED source sweep | order-budget copies **13 → 1**; `internal_field(pol='tm')` raised → works; `angle=0.9, theta=0.25` silent → warns; `prepare().solve()` silently dropped propagating orders → refuses; §7.1 now states `J[0,0] = −r_p`, measured `−1.000000` at 0°/30°/60° |
| G3 sub-item: **dimensionless-`kx0` pencil** | **not reproducible / infeasible**, documented | `pmm/_core.py::_uniform_geo_eig` (derivation in the comment) | `::test_g3_the_oblique_geometric_eig_genuinely_changes_with_wavelength` | the pencil's own algebra + cache-entry counts | the rewrite cannot exist: the non-dimensionalised pencil is **quadratic in the wavelength** (`a²L̃ − i·a·kxn(C̃−C̃ᵀ) + kxn²S̃0` with `a = 1/(P k0)`).  Normal incidence: 2 wavelengths → **1** cache entry.  Oblique: 2 wavelengths → **2** entries, operators differing by `O(1)` relative |
| Alternative algorithms (Gegenbauer basis / true hp mesh) | **deferred**, design + estimate in §6 | — | — | — | — |

---

## 2. Per finding

### G1 (P1) — the differentiable `PMMStack.solve` twin returned before every guard

**What was wrong.**  `PMMStack.solve` dispatched to `pmm/_jax_stack.py` and *returned* before
`_require_propagating_incidence`, before the cross-layer sliver screen and before
`_warn_stack_energy`.  `_jax_stack.py` contained zero occurrences of any of those names; its only
guard was a grazing check on `|kz_inc| < 1e-9`.  Two consequences, both reproduced on the current
tree before the change with `repro/PMM-1D/p11_jax_guards.py`:

```
gain n_sup   numpy: RAISE ValueError :: PMMStack.solve: gain incidence medium ...
             jax  : tot=[-0.84773 -0.86262]                    <-- silent, negative
deg= 14      numpy: RAISE ValueError        jax: T0(Ey)=1.3416682 tot=8.348260
deg= 16      numpy: RAISE ValueError        jax: T0(Ey)=1.3416312 tot=8.349067
deg= 18      numpy: RAISE ValueError        jax: T0(Ey)=1.3416253 tot=8.348532
deg= 12 / 20 numpy: 0.7658701 / —           jax: 0.7658918 / 0.7659174   (agrees)
```

The gain case is exactly the audit-M3 2026-07-25 defect the NumPy path was fixed for, still alive on
the twin — and `n_superstrate` is fully concrete there, so the documented "a TRACED value skips the
guard" carve-out never covered it.  The sliver case is worse because it is *degree-dependent*: it
passes any spot check at a neighbouring degree.

**What I changed, and why.**

1. `_jpmm_concrete_incidence_guard("PMMStack.solve", self.n_sup, self._src["angle"])` is hoisted to
   immediately **above** the `self._holds_traced()` dispatch (and below the conical dispatch, which
   already guards itself, so nothing double-raises).  It already degrades correctly on a traced
   value.  For a NumPy stack it is two extra float comparisons before the existing call.
2. `_sliver_screen` gained `allow_traced=False`.  The traced exclusion exists to protect the
   *arbiter* (whose three re-solves are meaningless under a trace), but the screen itself reads only
   wall coordinates, `min_feature`, `degree` and `period` — all concrete host values on the
   differentiable path, because the twin freezes the geometry.  `_warn_jax_stack_sliver` calls it
   with `allow_traced=True` before the twin runs and WARNS; the arbiter never passes the flag.
3. `_warn_stack_energy_concrete` runs the NumPy tripwire on the returned arrays when they are
   concrete, detected with a typed `isinstance(x, jax.core.Tracer)` — **not** a concretization
   `try/except`, so the non-`ui` broad-except budget is untouched (my files add **0**; verified by
   count against `HEAD`).

The refusal cannot be made complete under a trace: the arbiter is what turns a screen hit into the
NumPy `ValueError`, and it needs three re-solves.  That is now stated in the `PMMStack.solve`
docstring, which previously asserted the opposite (*"same physics … forward-identical to NumPy at
~1e-15"*) and listed only slant / out-of-plane / `stabilize` / `retain_internal` / sweep as raising.

**How I verified.**  `p11_jax_guards.py` re-run: the gain row now reads `RAISE ValueError ::
PMMStack.solve: gain incidence medium …` on *both* branches, byte-identical strings.  A dedicated
probe confirms the warning structure per degree:

```
deg= 12  jax: T0(Ey)=0.7658918 tot=1.000019   warn=['SLIVER-SCREEN']
deg= 14  jax: T0(Ey)=1.3416682 tot=8.348260   warn=['SLIVER-SCREEN', 'ENERGY']
deg= 16  jax: T0(Ey)=1.3416653 tot=8.349067   warn=['SLIVER-SCREEN', 'ENERGY']
deg= 18  jax: T0(Ey)=1.3416225 tot=8.348532   warn=['SLIVER-SCREEN', 'ENERGY']
deg= 20  jax: T0(Ey)=0.7659349 tot=1.000026   warn=['SLIVER-SCREEN']
```

i.e. the geometric screen fires at every degree (it is geometry, not the answer) and the energy
tripwire fires at exactly the corrupt ones.  Clean-stack control: **no** warnings, eager value
0.9970291411, `jax.grad` = −0.02222695101 against a central FD of −0.02222695056 (**2.0e-8**
relative, the audit's 5.8e-8 class).  Under `jax.grad` on the sliver stack the screen still fires and
the gradient is finite (−0.245428+2.78943j) while the tripwire correctly stands down.

**Residual risk.**  Inside `jit`/`grad` the energy tripwire cannot run, and a traced
`n_superstrate`/`angle` still skips the incidence raise — both are trace contracts, not oversights,
and both are now in the docstring.  A user who only ever differentiates a sliver stack gets a warning
where NumPy refuses; the docstring names the remedy (solve once with NumPy inputs first).

---

### G2 (P2) — the default `min_feature` sat at the bottom of the hazard band

**What was wrong.**  `min_feature` defaulted to `period · 1e-5`, and the sliver hazard band is
`s ∈ [1, 8] · min_feature`.  The default therefore snapped away only the collisions that were
already harmless and left the whole dangerous decade exposed; with the refusal armed, a staircased
stack at `s = 1.5e-5` of the period was refused at **every degree from 10 to 24**.

**What I changed.**  `_MIN_FEATURE_DEFAULT_FRAC = 1.0e-3`, a named constant carrying the measurement,
with the collision-scale rule and the migration note in `PMMStack.__init__`, and a `min_feature`
entry added to the class `Parameters` section (it had none).  The stale sub-claim in the old comment
("on a 700 nm pitch the default is 0.007 nm, ~200× too small") is rewritten as a statement about the
mechanism rather than about a default that no longer exists.

**How I verified.**  `repro/PMM-1D/p3e_band2.py` re-run in full on the audit's second, independent
fixture (TiO₂-like 2.35/1.46, Λ = 0.55 µm, λ = 0.70 µm, 31°), reference `T0 = 0.199229666`:

| `min_feature` | rungs of 11 with degree-scatter | worst reading in the band |
|---|---|---|
| `1e-5·P` | **6** (1× … 8×) | 0.30817 / 0.26455 / 0.19388 vs 0.19923 |
| `1e-4·P` | **1** (1× only, at one degree of five) | 0.19377 |
| `1e-3·P` | **0** — every rung degree-independent to 7 digits | — |

and the non-perturbation claim reproduces exactly: `s = 1.5e-4` reads 0.19839028 under `1e-5` and
`1e-4`; `s = 3e-4` reads 0.19755266 under both; `s = 1e-3` reads 0.19366045 under `1e-5` and `1e-3`.
The regression test additionally pins bit-identity at `s` = 2e-3 and 5e-3 (both above the new
threshold, so neither setting snaps them).

**Test fallout, and what I did about it.**  Three existing fixtures inherited the old default *and
needed the sliver to survive*; at the new default their geometry carries no sliver at all and they
measure nothing.  They now pin `min_feature` explicitly and say why:

* `tests/unit/test_fix_pmmstack_sliver_walls.py` — the O-11 hazard-band fixture (6 construction
  sites + the band comment).  5 failures → 19 passed, 1 dynamic skip.
* `tests/unit/test_pmm_m2_window_contract.py::_mk` — its T3-1 measurement *asserts* the snap is
  inert at `PERIOD*1e-5`, so without the pin the screen and the solve disagreed about the geometry.
* `tests/unit/test_pmm_m3_efficiency.py` — the T3-4 sweep-ordering fixture needs a device that trips
  the classification guard, and the sliver is the mechanism.

No test asserted the *old default value*; all three were using it as a convenient way to get a
sliver-carrying stack.

**Residual risk.**  This is a **default change** and is called out as such in the changelog with a
migration note: a stack *with* colliding cross-layer walls can now see its solved geometry differ
from the requested one by up to `5e-4` of a period (the snap moves walls by ≤ `min_feature/2`); a
stack *without* them is byte-identical.  The snap has always been cross-layer-pairs-only, so an
intentional single-layer thin feature is never touched.

---

### G3 (P2) — cost

**(a) The sliver arbiter: the two collapse solves are lazy.**  `d12` — the device's own
sensitivity to where the contested wall sits — is read only on the `'sliver'`/`'wall'` fork, which is
reached only after `d0` has cleared the geometric floor.  The two `_sliver_collapse_solve` calls are
now paid only there, and `ev['d12']` / `ev['d0_over_d12']` / `ev['closed_super_unity']` are `None` on
the `'truncation'` branch — they were not measured, and reporting a number nobody computed would be
worse than reporting none.  Counted deterministically by wrapping `PMMStack.solve` itself (not a
`_core` helper — `stack.py` binds those names at import, the instrumentation bug the auditor caught
and recorded):

| n_slices | `s` | before | after | guard OFF | verdict |
|---|---|---|---|---|---|
| 4 | 3e-4 | 4 | **2** | 1 | truncation |
| 8 | 3e-4 | 4 | **2** | 1 | truncation |
| 8 | 1e-4 | 4 | 4 | 1 | refused (needs `d12`) |
| 16 | 5e-5 | 4 | 4 | 1 | refused (needs `d12`) |
| 4 | 0 | 1 | 1 | 1 | *control — no sliver* |
| 1 | 0 | 1 | 1 | 1 | *control — single layer* |

`PMM_SLIVER_ARBITER_LAZY` (default `True`) is the fail-before switch, in the module's own idiom:
`False` restores the eager form bit for bit.  The "before" column is re-derived through it
in-process by the regression test, not quoted.  No verdict and no returned number changes.

**(a′) The verdict MEMO: implemented, measured, and WITHDRAWN.**  The finding asks for the verdict to
be memoized "so a sweep pays it once".  I implemented it — a blake2b digest of the complete
determinant of what the re-solves read (geometry, every segment permittivity, both half-spaces, the
source, the discretisation knobs and the four module-level arithmetic switches), in a
registry-enrolled `ByteBudgetedLRU` — and then removed it, for two measured reasons:

1. **The sweep benefit does not exist.**  `wl` is part of that determinant, so every point of
   `solve_vs_wavelength` is genuinely different physics and misses by construction.  The only
   remaining beneficiary is a repeated *identical* `solve()`, measured at 10 → 6 invocations (5
   repeats, n_slices 8) and 20 → 8 (n_slices 16).
2. **It cost more than that.**  Three shipped contracts instrument `_sliver_probe_solve` /
   `_sliver_collapse_solve` to assert *which source* the arbiter is handed
   (`test_the_sweep_arbitrates_at_its_own_wavelength_not_a_stale_set_source`,
   `test_the_sweep_arbitrates_the_same_way_at_any_worker_count`,
   `test_the_prepared_path_arbitrates_at_the_wavelength_it_was_given`).  A memo *around* those
   helpers makes the second and later calls invisible to the spy, so the contract stops being
   checkable — and a cache that hides the thing it caches from the tests that guard it is a worse
   trade than the saving it buys.  Removing it took the sliver-corpus failures from 12 to 6 (the
   remaining 6 were the laziness, and are the restatements described below).

The design a future attempt should use is in §6: memoize *inside* `_sliver_probe_solve` /
`_sliver_collapse_solve`, where a spy that replaces the whole function still observes every call.

**(b) Diagonal masses, `Q @ W2`, `np.add.at`, the geo-eig key.**  `_safe_inv`, `_safe_solve` and the
seven `inv(S0) @ X` products in `_sem_modes_tensor` take an `O(n)` reciprocal / row scale on an
exactly-**real** diagonal (`_real_diagonal` / `_row_scale_apply`), and the dense LAPACK path
otherwise.  The restriction to real diagonals is measured, not cautious — over 1,000 random diagonals
(positive; mixed-sign; `float64` and `complex128`; 16 decades of magnitude including the 1e-14
sliver-element regime) `inv(D)` vs `diag(1/d)`, `solve(D,B)` vs `B/d[:,None]` and `diag(1/d)@B` vs
`(1/d)[:,None]*B` differ in **0 of 1,000** trials, while on genuinely *complex* diagonals they differ
in **1,000 of 1,000** by 2–4e-16 relative.  `Q @ W2` is named once instead of twice (the trailing
`@ np.diag(…)` stay gemms deliberately: `A @ diag(v)` and `A * v[None,:]` agree only to ~2e-16
relative for complex `v`, measured, and `lam` is complex).  `_sem_fourier_projection` and its JAX
static twin scatter with `np.add.at` (bit-identical — `l2g` repeats at the periodic wrap and
`np.add.at` accumulates in the same index order the loop did; 0 of 200 random layouts differ — and
1.45–2.6× faster over `(degree, n_el)` = (12,4)…(32,20)).  `_GEO_EIG_CACHE` is keyed on a 32-byte
blake2b digest instead of `n_glob²` complex128 of operator bytes (~1.4 MB *per key* at a production
`n_glob` = 300, plus an `O(n²)` copy per lookup) and is now a `ByteBudgetedLRU`.

**Bit-identity A/B.**  75 arrays across single-layer Jones (normal / oblique / Au), the scalar
`te`/`tm` entries, conical, four multilayer stacks (shared, anisotropic + lossy, per-layer,
normal-incidence, slanted) and the modal kernels themselves, captured before the change and again
after: **0 of 75 differ, worst |A−B| = 0.000e+00.**

**Speed, reported honestly.**  On a 4-layer fixture the deterministic call counts read **22 → 14**
dense `np.linalg.inv` per solve (`solve` unchanged at 10, `eig` at 5), worth 0.77 % of the
inv/solve/eig flop total; the gemm side (7·n³ of row scales per patterned layer plus 8·n³ from the
`Q@W2` reuse, against ≈344 n³ per layer) is ~4 % by the flop model and is **not** separately
measurable at run time — the `@` operator does not route through `np.matmul`, so it cannot be
counted.  An **interleaved** min/median-of-9 wall-clock A/B on this box reads OFF/ON = 0.985–1.004
(min) and 0.948–1.182 (median) across degree 10/14/20/26, i.e. **indistinguishable from contention
noise**, and `tracemalloc` peaks are identical to the byte (13.01 MB / 44.74 MB at degree 14 / 26).
I am claiming **no speedup**: what is established is bit-identity, 8 fewer dense inverses per solve,
2 fewer whole-stack solves on a truncation verdict, and a cache key that no longer retains megabytes.
The audit's own 1.02–1.14× was measured on a different fixture; it is not contradicted here, it is
below this box's noise floor.

**(d) The Wood-anomaly nudge's new warning (WP-A14's change, coordinated).**
`rcwa/_core::_grazing_safe_wavelength` now emits a `WoodNudgeWarning` when it nudges, and takes
`fn_name=`.  Every 1-D PMM caller now passes it, so the warning names the entry point the user
actually called — `pmm_efficiency_1d`, `pmm_jones_1d`, `pmm_efficiency_1d_segments`,
`pmm_jones_1d_segments` (through `_wood_safe_wl_1d`, which gained an `fn_name` parameter),
`pmm_jones_1d_conical`, `pmm_jones_1d_conical_tensor`, `PMMStack.solve (conical)`, and the shared
`_conical_nodal_solve` label.  Verified: solving at `wl = period` on an air-clad cell now reads
`pmm_efficiency_1d: a diffracted order sits EXACTLY at cut-off …` and
`pmm_jones_1d: …` respectively.  Numerics unchanged; the 2-D PMM callers belong to WP-A13 and were
not touched.  No PMM 1-D test of mine sits on an anomaly, so no category filter was needed; one
pre-existing test (`test_v5_20_7_pmm_geo_eig_cache`) now emits the warning and still passes.  I have
no measurement that would justify a per-caller step, so I am not requesting one.

**Residual risk.**  The laziness is the only behaviour-visible change here, and it is a cost
change: no verdict and no returned number moves, and the switch restores the eager form bit for bit.
What it does change is the *evidence dict* on a `'truncation'` verdict, where three fields are now
`None`; five tests read them and were restated (four through the switch, keeping every number they
measured; one — the arbiter's own cost test — restated to the per-verdict contract with the pre-lazy
count re-derived through the switch).  Nothing in the library reads those fields on that branch.

---

### G4 (P3) — API and documentation

* **`internal_field(pol=…)`** now takes `'tm'`/`'p'` (→ row 0, incident `E_x`) and `'te'`/`'s'`
  (→ row 1, incident `E_y`), case-insensitively, through a `_resolve_internal_pol` helper; the
  integers `0`/`1` are unchanged; an unknown spelling raises with the §2 `fn_name: ` prefix naming
  every accepted value.  The mapping is tested against the index spelling it must reproduce, so a
  transposed alias table cannot pass.
* **`angle=A, theta=T` with `A ≠ T`** now warns, naming both values and the winner.  The
  resolution is unchanged — `theta` still wins — because that is a deliberate cross-suite contract
  pinned by `test_v5_12_0_naming_aliases`, which asserts PMM and RCWA agree about it.  See §5: the
  raise the audit recommends needs a coordinated change in the RCWA resolver.
* **The far-field order budget**: 13 near-verbatim copies (6 in `_core.py`, 7 in `stack.py`, counted
  at `HEAD`) replaced by one `_core._farfield_order_set(period, wl, n_max, ffo, n_glob, label, *,
  degree, kx0, k0, when)`.  Consolidating surfaced a **divergent copy**: `_PreparedPMMStack.solve`
  clamped the projector to the nodal capacity but never refused when the propagating orders did not
  fit, so the prepared path returned a far field with orders *missing* — sub-unity power, invisible
  to the one-sided energy tripwire.  It now refuses like every sibling.  The four minority refusal
  wordings are unified (none was asserted anywhere; checked).
* **The "~85 % of runtime" note** is replaced by the deterministic call counts (42 `inv` / 18
  `solve` / 8 `eig` on an 8-layer degree-24 stack) and the flop model (eig ≈ 58 %, interface ≈ 14 %,
  star ≈ 19 % per layer), with an explicit statement that the *empirical* split is **unmeasured** and
  why (contention: a 144×144 complex eig timed at 490 ms; `inv(384)` timed faster than `inv(320)`).
* **`elements_per_region > 1, grade=True`** loses "the speed lever for TM (hp-refinement)" and gains
  the matched-DOF ladder that contradicts it (2.5–4× worse than single-element p-refinement at equal
  DOF), plus the reason (uniform-degree grading is not hp).
* **The TM convergence caveat** is added to `pmm_jones_1d` (with both measured ladders: TE rate ≈ 9
  and rising, TM `O(N^-2.7)` flat on a *lossless* cell) and to `PMMStack`'s class docstring.  The
  regression test measures the *rates* on this build rather than asserting the prose.
* **The slanted-TM super-unity floor** (1 + 4e-5 … 9e-5, decreasing with degree, against 1e-10 for
  the vertical cascade) is recorded in both slant entry docstrings as a calibration floor.
* **CONVENTIONS §7.1** now says every solver returns the lab Cartesian `(E_x, E_y)` Jones, index
  0 = x / 1 = y, and that at `φ = 0` the x column is the p channel *up to the sign of the p unit
  vector* (`J[0,0] = −r_p`).  Re-measured with `repro/PMM-1D/p12_jones_basis.py`: `pmm/analytic`
  ratio `xx = −1.000000`, `yy = +1.000000` at 0°/30°/60°, `rcwa/pmm = +1.000000` on both.  The
  regression test re-derives the analytic TMM inside the test rather than calling library code.

---

## 3. Files touched

Source (all within the WP's ownership):

* `lumenairy/elements/pmm/_core.py`
* `lumenairy/elements/pmm/stack.py`
* `lumenairy/elements/pmm/oned.py`
* `CONVENTIONS.md` (§7.1 only)

* `lumenairy/elements/pmm/conical.py` (the `fn_name=` wiring for WP-A14's `WoodNudgeWarning` only)

Not touched, though listed as owned: `lumenairy/elements/pmm/_jax_stack.py` and
`lumenairy/elements/pmm/__init__.py` — G1 is fixed entirely on the dispatcher side, which is where
the guards belong and where they degrade correctly on a trace.  `conical.py`'s order budget is a
different, 2-D-style block that the consolidation does not cover.

Tests:

* **new** `tests/unit/test_audit2609_a12_pmm1d.py` (21 tests)
* `tests/unit/test_fix_pmmstack_sliver_walls.py` — `min_feature` pinned at the fixture's own value
  (6 sites) + the band comment corrected
* `tests/unit/test_pmm_m2_window_contract.py` — `_mk` pins `min_feature`
* `tests/unit/test_pmm_m3_efficiency.py` — the T3-4 sweep-ordering fixture pins `min_feature`
* `tests/unit/test_fix_pmmstack_sliver_walls_round2.py` — an `_eager_arbiter()` context manager
  (the new fail-before switch) for the two tests whose subject is `d12`; the arbiter's cost test
  restated to the per-verdict contract, with the pre-lazy count re-derived through the switch
* `tests/unit/test_fix_pmmstack_sliver_round4.py` — the same `_eager_arbiter()` for its two
  denominator tests

Docs: `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A12_REPORT.md` (this file) and
`WP-A12_CHANGELOG.md`.

---

## 4. Tests run

Every command was run with `OPENBLAS_NUM_THREADS=1` and
`python -m pytest … -q --no-header -p no:cacheprovider`.

| what | result | time |
|---|---|---|
| **the new file** `tests/unit/test_audit2609_a12_pmm1d.py` | **21 passed** | 43 s |
| **fail-before** — the same file with every WP-A12 change reverted in-process (a pytest plugin patching `_MIN_FEATURE_DEFAULT_FRAC`, `_real_diagonal`, `PMM_SLIVER_ARBITER_LAZY`, `_warn_jax_stack_sliver`, `_warn_stack_energy_concrete`, `_jpmm_concrete_incidence_guard`, `_resolve_incidence`, `_resolve_internal_pol`) | **10 failed**, 11 passed | 37 s |
| **batch 1** — 22 PMM / cache-registry files + the new one | **545 passed, 1 failed** (pre-existing, below) | 336 s |
| **batch 2** — the 7-file sliver corpus + 18 further PMM 1-D / alias / conventions files | **546 passed, 1 failed** (pre-existing, below), 1 dynamic skip | 458 s |
| the sliver corpus alone, after the arbiter work (7 files + the new one) | **108 passed, 1 failed** (pre-existing, below), 1 dynamic skip | 247 s |

Across the two batches: **1 091 passed, 2 failed, 1 skipped** — and both failures are pre-existing
(reproduced with every WP-A12 change neutralised in-process; see below).
| `ruff check lumenairy/elements/pmm/ tests/unit/test_audit2609_a12_pmm1d.py …` | **All checks passed** | |

**Fail-before, in detail.**  10 of the 21 new tests fail with the changes reverted in-process.  The
other 11 cannot be reverted that way, and their fail-before is *structural* — verified against `HEAD`
rather than asserted: `_geo_eig_key`, `_farfield_order_set`, `_real_diagonal`, `_row_scale_apply`,
`_resolve_internal_pol`, `_warn_jax_stack_sliver` and `_sliver_probe_key` are all absent at `HEAD`
(so those tests error there); the order-budget parity trim occurs **13 times** at `HEAD` and once
now; `CONVENTIONS.md` at `HEAD` contains the old *"The 1-D solvers return te/tm"* sentence and not
`J[0, 0] = -r_p`; `oned.py` at `HEAD` carries the unqualified *"Converges SPECTRALLY … with no
accuracy floor"*; and `PMMStack.prepare().solve` has no capacity refusal at `HEAD`.

**Audit "keep intact" re-checks** — the audit's own repro scripts, re-run on this tree:

| property | audit | re-measured |
|---|---|---|
| energy, lossless Si/SiO₂ (`p5`) | ≤ 3.1e-12 | **3.11e-12** worst, median ~2e-14 |
| reciprocity TM `m = −1` (`p5`) | 1.8e-11 | **1.775e-11** |
| unpatterned vs an independent TMM (`p6`) | Δ\|r\| ≤ 2.3e-14, Δphase ≤ 1.7e-13 | **2.30e-14** / **2.5e-15**; cross-pol ≤ 9.1e-45 |
| PMM vs the library's RCWA (`p5b`) | 4.6e-8 / 7.5e-8 / 1.2e-6 (TE 0/17/45°) | **4.62e-8 / 7.53e-8 / 1.16e-6** |
| mortar → interface S-matrix on identical grids (`p3c`) | ≤ 4.3e-14 | **≤ 4.31e-14**; `hw ≥ nlay−1` reproduces shared at **0.0** |
| cache keyed on operator bytes, shuffled replay (`p9`) | 0 mismatches | **0 mismatches**, twice |
| unit invariance over 13 decades (`p9d`) | R₀ = 0.08536814608764 ± 6e-14 | **±1e-13** across km → Å |
| conical: `pmm_jones_1d_conical` vs `PMMStack(phi=π/2)` (`p7`) | bit-identical | **max\|ΔJ\| = 0.0**; \|tot−1\| ≤ 2.6e-11 |
| `layer_absorption` on a lossless stack (`p14`) | −4.6e-14 / −1.6e-12 | **−4.6e-14 / −1.1e-12** |
| `internal_field` continuity (`p14d`) | Ex 4.4e-3 rel, Ez stays put | **4.398e-4 rel** at this Δz, Ez **5.615e-1** — the audit's table |
| `stabilize=True` never materially worse (`p13`) | 0.40–1.00 in 47/48 | **0.71–1.00** on the re-run cells |
| round-4 sliver guard: false negatives (`p3d`, `min_feature` pinned at the census's own value) | 0 in 88 cells | **0 in 96 cells**; every returned value 0.7576–0.7578 |
| G2 hazard band (`p3e`) | 6 / 1 / 0 of 11 rungs at `1e-5` / `1e-4` / `1e-3` | **6 / 1 / 0**, and the unsnapped rungs agree exactly |
| G1 JAX guards (`p11`) | gain silent, sliver 8.35 silent | **gain refuses, sliver warns twice** |
| §7.1 Jones basis (`p12`) | `J[0,0]/r_p = −1.000000` | **−1.000000** at 0°/30°/60°; `rcwa/pmm = +1.000000` |

**Pre-existing failures found (not mine; each confirmed by re-running with every WP-A12 change
neutralised in-process):**

1. `tests/unit/test_pmm_m2_window_contract.py::test_halfwidth_2_moves_the_answer_only_inside_the_mortar_band`
   — fails identically with all my changes reverted.  Its own screen reports every cell as
   classification-unsound (`n_growing > 0` on the halfwidth-2 runs), which is the build-fragile class
   its docstring already documents.  A BLAS/NumPy-build property of this box, not a library change.
   I did pin `min_feature` in its `_mk` regardless, because without it the test's own
   `assert merged == 0` screen and its solves would have been looking at different geometries.
2. `tests/unit/test_verify_pmmstack_sliver_walls.py::test_the_pure_stacks_shared_grid_cannot_express_a_sliver`
   — raises from `lumenairy/elements/pmm/twod_staggered.py` (`PMM2DStackPure.add_layer`'s pencil-DOF
   refusal), a PMM 2-D file owned by WP-A13 and being edited concurrently (the message wording
   changed between two of my runs).  Fails with all my changes reverted.
3. `tests/unit/test_audit_except_budget.py` — 55 broad `except Exception:` against a budget of 48;
   every added clause is in another WP's file (see §5.5).

**No `validation/test_*.py` topic file covers the PMM 1-D surface** (checked by grep for `PMMStack` /
`pmm_jones_1d` / `pmm_efficiency_1d` across `validation/**/test_*.py`), so `validation/run_all.py`
has nothing to run for this area.

---

## 5. Requested changes outside my ownership

1. **`lumenairy/elements/rcwa/oned.py::_resolve_incidence`** (WP owning `rcwa/`, i.e. WP-A14) —
   mirror the `angle`/`theta` mismatch diagnostic I added to `pmm/_core.py::_resolve_incidence`
   (warn when both are given, both are numeric, `angle != 0` and `angle != theta`; still resolve to
   `theta`).  **Why:** `tests/unit/test_v5_12_0_naming_aliases.py::
   test_set_source_theta_wins_consistent_across_suites` pins that the two suites behave identically,
   so a PMM-only change would be a divergence.  Once both sides warn, the audit's preferred fix —
   *raising* on a mismatch — becomes a single coordinated change to both resolvers plus an update to
   that test file (which is cross-suite and therefore nobody's sole property); I did not make it
   unilaterally.
2. **`lumenairy/elements/rcwa/stack.py:2293`** — same call site, same reasoning.
3. **`tests/unit/test_v5_12_0_naming_aliases.py`** — if (1)/(2) land as a *raise*, this file's
   `angle=0.9, theta=_TH` / `angle=0.0, theta=0.25` rows need updating to `pytest.raises`.  It is a
   cross-suite test file, not a PMM one, so I left it alone.
4. **`CHANGELOG.md`** — my changelog text is in
   `docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A12_CHANGELOG.md` for the
   orchestrator to assemble, per COMMON.md §2.
5. Note for the orchestrator, not a request: `tests/unit/test_audit_except_budget.py` FAILS on this
   branch — 55 broad `except Exception:` clauses against a budget of 48, and rising as sibling WPs
   land (it read 53 earlier in this session).  Attributed by `git diff`, every added clause is in
   another work package's file: `lumenairy/elements/_lens_jax.py` ×2,
   `lumenairy/elements/_lens_real.py` ×2, `lumenairy/elements/_lens_imap.py` ×1, and earlier
   `lumenairy/io/prescriptions_code_v.py` / `lumenairy/glass.py`.  **My files add zero** — per-file
   counts are byte-identical to `HEAD` (`_core.py` 5, `stack.py` 3, `oned.py` 2, `conical.py` 0,
   `_jax_stack.py` 3), which is why `_warn_stack_energy_concrete` detects a Tracer with a typed
   `isinstance(x, jax.core.Tracer)` rather than the usual concretization `try/except`.  Either the
   new sites need narrowing or the budget needs a documented bump.

---

## 6. Deferred, with designs

1. **Raise (rather than warn) on an `angle`/`theta` mismatch.**  Design: add the raise to both
   resolvers behind one shared predicate; update the three alias tests.  Effort: ~1 h including the
   cross-suite test edit.  Blocked only on ownership (§5).
2. **`_resolve_order_count(far_field_orders, n_orders)` has the same "alias wins, no equality check"
   shape.**  I did not touch it: unlike `angle`, its default varies per entry point (21, 11, …), so
   "the caller passed a non-default explicitly" is not decidable at the resolver.  Design: thread a
   sentinel default (`_ORDERS_UNSET`) through the ~12 entry points so the resolver can tell "unset"
   from "explicitly 21".  Effort: ~2–3 h, touching every 1-D and 2-D entry signature — a mechanical
   but wide change that would collide with three other work packages this week.
3. **Gegenbauer / ultraspherical basis for the TM wall corner** (the audit's alternative (c), and the
   highest-value algorithmic move in this partition).  Design: `_gll_nodes_weights(degree)` and
   `_lagrange_derivative_matrix(nodes)` are the only two places the basis enters; an ultraspherical
   Gauss–Lobatto rule with parameter λ (λ = 1/2 recovers today's Legendre/GLL) gives a one-parameter
   family, and the element mass stays diagonal only for λ = 1/2 — so `_build_sem*` must stop assuming
   a lumped mass (`_real_diagonal` already returns `None` for a non-diagonal `S0`, so my G3 fast
   paths degrade correctly and nothing else in the assembly hard-codes diagonality).  Gate: the
   auditor's Au/air TM fixture with the extrapolated RCWA oracle — the change is worth taking only if
   the measured local rate leaves the `O(N^-2.7)` regime.  Effort: ~1–2 days including the
   convergence study, and it must be opt-in (a `basis=` kwarg) until the whole 1-D corpus is
   re-measured against it, because it changes every number.
4. **A genuine hp mesh** (geometric grading σ ≈ 0.15 *with a linearly decreasing degree toward the
   corner*, Babuška–Guo — the audit's (c′)).  `_graded_boundaries` already produces the mesh; what is
   missing is a per-element degree, which today is a single scalar threaded through `_build_sem*`,
   `_l2g_periodic`, `_sem_fourier_projection` and every `mats["degree"]` reader.  Effort: ~2 days,
   and it is the larger change of the two for the same target, so (3) should be measured first.
5. **Eigen-recycling across an oblique sweep** (Jacobi–Davidson or Newton-from-previous-point), the
   only route to the cross-wavelength reuse §G3(c) shows a rescale cannot give.  Design and prior art
   are in `docs/audits/EXPERIMENT_PMM2D_EIG_RECYCLE_2026_08_16.md` and apply unchanged in 1-D.
   Effort: ~2–3 days, and it needs its own accuracy gate (a seeded solver can converge to a
   *different* mode).
6. **`PMMStack.solve` is still ~470 lines.**  The order-budget extraction removed one of the eight
   duplicated cascade tails; the grid build, the eig memo and the Redheffer loop are still
   re-implemented in eight places.  Effort: ~1–2 days, and it should be done as its own work package
   with a bit-identity gate, not folded into a defect fix.
7. **Memoizing the sliver arbiter's re-solves** (implemented, measured, withdrawn — §2 G3(a′)).  The
   design a second attempt should use: put the memo **inside** `_sliver_probe_solve` and
   `_sliver_collapse_solve`, keyed on the complete determinant (the `_sliver_probe_key` digest I
   wrote is a working starting point: period / degree / n_el / grade / ffo / min_feature /
   layer_grids / window_halfwidth / factorization / both half-spaces / wl / angle / phi / the four
   arithmetic switches / every layer's thickness, slant and per-segment `(width, eps)` bytes; a
   non-materialisable eps yields `None` and disables the memo for that call).  Inside the helpers, a
   spy that replaces the whole function still observes every call, so the three
   "which source was the arbiter handed" contracts stay checkable.  Measured benefit: 10 → 6 and
   20 → 8 `PMMStack.solve` invocations over five repeated identical solves.  Effort: ~2 h plus a
   stale-key test per arithmetic switch.  Worth doing only alongside a use case that actually
   re-solves the same point (a stabilize ladder or a consensus check), because a wavelength sweep
   provably cannot hit it.

---

## 7. Changelog text

`docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/WP-A12_CHANGELOG.md`
