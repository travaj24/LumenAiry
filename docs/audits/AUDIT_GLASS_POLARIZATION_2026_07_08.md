# Glass + Polarization Audit — 2026-07-08

Scope: full line-level reads of `glass.py` (1,677 — every trace,
Seidel sum, coating, and chromatic analysis resolves indices through
it; zero prior deep-audit findings) and `elements/polarization.py`
(1,190 — the Jones-field layer and its convention-bearing element
family).  Continuation of the under-covered-subsystem sweep after the
raytrace completion tranche.  Read-only single-context pass; dispersion
values and Jones algebra re-derived by hand.

---

## 1. `glass.py` — verdict: excellent

The dispersion layer is the most defensively engineered module
audited so far.  Independently verified:

* **Sellmeier evaluator** — 3-term form with resonance guard and
  negative-radicand raise; hand-computed N-BK7 at the d-line from the
  bundled coefficients: n_d = 1.5168 ✓.
* **Polynomial (formula-3) evaluator** — hand-computed Sumita K-BK7
  at the d-line: n_d = 1.51633, exactly the catalogue value ✓.
  Scalar fast-path / array path parity; non-positive-n² raise.
* **`_guard_wavelength`** — the Sellmeier(sign-symmetric: warn+abs) vs
  polynomial (raise) distinction is correct reasoning; all *bundled*
  polynomial exponents happen to be even, so the raise is
  conservatively future-proof for the general formula-3 contract.
* **Dispatch ladder** (`get_glass_index`) — air → unknown-name
  (substring + difflib suggestions) → callable → validity warn →
  `__sellmeier__` → `__polynomial__` → `__thin_lens__` → user-fixed
  sentinel → refractiveindex tuple, with the minimal-install fallback
  chain (bundled Sellmeier → bundled polynomial → stub
  NotImplementedError → ImportError).  Every arm reachable, every
  error message actionable.
* **Import-time consistency check** — six-way
  registry↔coefficients↔validity↔stub cross-validation converts the
  whole "orphan row / dangling sentinel" bug class into a load-time
  failure.  This is the pattern other registries in the library
  should copy.
* **Cache discipline** (P3-40/P2-41 remediations verified) —
  LRU-bounded value cache keyed at picometre resolution, lock held
  only around bookkeeping (compute outside), user-fixed entries
  preserved by the registry-driven clearer, surgical
  `_invalidate_glass_name` on re-registration.
* `get_glass_index_complex` — the refractiveindex-unavailable tuple
  path lands on `_glass_cache[name]` → `KeyError` → caught by the
  (correctly broad) except → warn + κ=0.  Intentional-looking and
  safe.

### GL-1 (P4) — the missing-κ warning advertises a remediation that does not exist
`_warn_missing_kappa_once` tells the user to "supply kappa explicitly
via register_fixed_glass" — but `register_fixed_glass(name, n: float)`
accepts a real index only, and the `_FixedIndex` shim it stores
exposes only `get_refractive_index` (no extinction).  A user following
the advice ends up back at the same warning.  The working remediation
is registering a **complex-returning callable** (which
`get_glass_index_complex` honours).  Fix the message, or add a κ
parameter to `register_fixed_glass` / `_FixedIndex`.

### GL-2 (P4) — `trace._register_fixed_index` mutates the glass value-cache without the lock
`raytrace/trace.py:_register_fixed_index` purges
`_glass_value_cache` entries by iterating and deleting **without
taking `_GLASS_CACHE_LOCK`** — a concurrent `get_glass_index` doing
an LRU `move_to_end` can raise "OrderedDict mutated during
iteration", and the torn read-modify-write is exactly what the P3-40
lock exists to serialise.  glass.py already ships the lock-correct
helper for this exact job (`_invalidate_glass_name`, P2-41) — the
trace-side copy predates it and was never routed through.  One-line
fix: call `_invalidate_glass_name(name)` before the re-register (it
also pops `_glass_cache`, which the immediate overwrite makes
harmless).

### Nits (glass)
* `'SILICA'` is registry-aliased to Malitson fused silica but has no
  bundled Sellmeier row, unlike its three siblings (`SiO2`,
  `F_SILICA`, `FUSED_SILICA`) — on a minimal install the alias raises
  ImportError while the siblings resolve.  Copy the row.
* Array wavelengths crash in `_maybe_warn_outside_validity`
  (`float(array)`) for any validity-tracked glass, even though the
  evaluators and the value cache both handle arrays.  Out of the
  documented scalar contract, but the module is one `np.any` away
  from honest array support.
* The tuple-path `_glass_cache` insert (RefractiveIndexMaterial
  construction) is unlocked — benign double-build, consistent with
  the documented cache discipline; noted for completeness.
* The Sellmeier resonance guard triggers only within 1e-12 µm² of a
  pole; near-resonance (anomalous-dispersion) wavelengths return
  physically huge indices with only the validity warning as signal —
  acceptable given the validity table covers every bundled glass.

---

## 2. `elements/polarization.py` — verdict: clean

All element and analysis physics verified by hand:

* **Waveplate Jones matrix** — expanded
  `R(θ)·diag(1, e^{+iφ})·R(−θ)` term-by-term; matches the code.  The
  P2-15 slow-axis-positive-phase sign (post-v5.17.0 realignment to
  the Berreman/RCWA solver Jones) is self-consistent end-to-end:
  worked example, QWP fast axis +45° on x̂ gives `Ey/Ex = −i`,
  `S3 = −2 Im(Ex Ey*) = −1` — exactly the documented 'left' branch of
  `create_circular_polarized`, whose 'right' (1, +i)/√2 gives
  S3 = +1 under the same formula ✓.  The three-way convention story
  (elements ↔ solvers ↔ `vector_diffraction`) now genuinely closes.
* **PBS** — `Jt = R·diag(a,b)·R⁻¹`, `Jr` with a,b swapped;
  `Jt†Jt + Jr†Jr = (a²+b²)·I = I` — power conserved *exactly* for
  every extinction ratio, per-pixel; each port operates on an
  independent copy (the in-place-mutation trap is handled).
* **Elliptical source** — `R(ψ)·(cos χ, i sin χ)` expansion matches.
* **Stokes / DOP / ellipse** — the `S3 = −2 Im(Ex Ey*)` convention
  applied uniformly; `tan 2ψ = S2/S1` via arctan2; `sin 2χ`
  clipped before arcsin.
* **`jones_pupil_to_stokes_unpolarized`** — correct Mueller-column-0
  (unpolarized-input average) with the ½ factor:
  S0 = ½Tr(JJ†), S2/S3 from `J00 J10* + J01 J11*` ✓.
* **JonesField** — grid convention `(arange(N) − N/2)·dx` in the
  spatially-varying Jones path; dtype-follows-input; anamorphic `dy`
  forwarded everywhere (the P2-4 fixes verified in place);
  batch-vs-sequential ASM dispatch sound; `propagate_fresnel` reads
  the input pitch for both components before mutating it.
* The per-component-dispatch validity caveats (P3-18/19/20/21,
  P2-3 s/p-averaging warning) are honestly documented and warn where
  they should.

### Nits (polarization)
* `JonesField.apply_spherical_lens` reaches for
  `kwargs['wavelength']` — omitting it raises a bare `KeyError`
  instead of a TypeError naming the missing argument (every sibling
  takes `wavelength` positionally).
* Five helper docstrings still say `angle_deg` "takes precedence
  over" `angle` — stale pre-v4.14.3 wording; the implemented (and
  correct) behaviour is raise-on-disagreement, as the same
  docstrings' Raises sections state.
* `apply_jones_matrix` requires callables to return per-pixel
  matrices; a constant-(2,2)-returning callable errors.  Documented,
  but a cheap broadcast-accept would be kinder.

---

## 3. Coverage statement

Every line of `glass.py` and `elements/polarization.py` read this
pass, plus the `_register_fixed_index` / `register_fixed_glass`
call sites in `raytrace/trace.py` and `user_library.py` (for GL-1/
GL-2).  Not audited here: the remaining `elements/` non-PMM modules
(`doe.py`, `elements.py`, `thin_grating.py`, `bsdf.py`,
`coatings.py`, `freeform.py`, `eme/`), `io/`, and `optimize/` — the
natural next tranches.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-08.
Companion docs: `AUDIT_RAYTRACE_CORE_2026_07_08.md`,
`AUDIT_PROPAGATORS_KERNELS_2026_07_07.md`,
`AUDIT_ANALYSIS_METRIC_CORE_2026_07_07.md`,
`AUDIT_SOURCES_CORE_2026_07_07.md`, `AUDIT_V5_21_DELTA_2026_07_07.md`.*
