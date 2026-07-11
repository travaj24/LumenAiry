# Optimize Second-Pass Audit — 2026-07-09

Scope: a second, deeper pass over `optimize/` at the user's request,
targeting everything the first-pass tranches covered only structurally:
`driver.py`'s scipy-method dispatch tail (1028-1413 — the LM /
differential-evolution / basin-hopping / dual-annealing / Newton paths
and final result assembly) and setup head (438-560 — precision guard,
P2#14 warning, need_ray/need_wave), `wrapper_merits.py`'s cache
infrastructure (132-310), and `context.py`'s `Constraint` (400-591) —
plus adversarial re-verification of the first-pass conclusions
(OPT-1/OPT-2 directions, the `jac='auto'` partition).  Read-only.

---

## 1. Re-verification of first-pass conclusions

* **OPT-1 stands.**  Adversarially re-checked from the driver side:
  `_sum_merits` adds every term and every scipy path *minimises*
  `merit_fn`; nothing anywhere in the driver negates or inverts a
  JaxMeritTerm's contribution.  The JAX LG merit's `Σ|Strehl|²` really
  is minimised toward zero Strehl.
* **OPT-2 stands** (nothing on the driver side back-fills the
  ToleranceAwareMerit sub-context).
* The `jac='auto'` disjoint partition, the FD stencils, and the
  checkpoint logic re-confirmed as first-pass documented.

## 2. Newly verified (previously structural-only)

* **LM path** — the residual vector `√(mᵢ + 1e-30)` makes
  `least_squares` minimise exactly `Σ mᵢ` (the scalar merit sum) with
  the sqrt-at-zero non-differentiability floored; the P1-DEEP-1-2
  None-endpoint bounds fix, the 3-tuple length guard, and the loud
  lm→trf override warning are all in place and correct.
* **Global methods** — DE/dual-annealing require bounds (raise
  otherwise); all four cancellation callbacks (incl. the P1-NEW-L
  dual-annealing fix) poll `is_cancelled` and return True to stop.
* **Newton path** — the FD-Hessian (central FD of the gradient,
  outer step 1e-4 balancing the inner O(h), bounds-clipped stencil,
  symmetrised `½(H+Hᵀ)`) is correct; the >30-var cost warning and the
  `hess='fd'`/callable dispatch check out.
* **Generic minimize path** — P2-24 bounds forwarding to every
  bounds-capable method with a loud drop-warning otherwise; the TNC
  `maxiter→maxfun` mapping; constraint threading; the v5.18 removal of
  the scipy-1.18-rejected `disp` option.
* **Setup head** — the precision knob's try/finally + `__del__`-backed
  dtype restore guard is sound; the **P2#14 propagator-mismatch
  warning is present and fires** (closing the cross-reference asserted
  by the wrapper-merit docstrings).
* **Wrapper-merit cache** — exemplary: the P2-25 two-level design
  (aperture-independent grids shared by reference across per-aperture
  entries, so an aperture-as-free-variable FD sweep rebuilds only the
  boolean mask), correct lock discipline (build outside the lock,
  double-check publish, counter incremented under the lock), the
  content-hashed ndarray aperture key, the `_ZERO_APERTURE_MASK`
  semantics, and both LRU bounds.
* **`Constraint`** — the v4.16.2/3 pickle-probe (catching closures and
  partials the old `__name__` heuristic missed, `Exception`-wide for
  hostile `__reduce__`), the opt-in `validate()` scalar-shape check
  with its actionable error, and the `to_scipy` ±inf translation are
  all correct.

## 3. Findings (second pass)

### OPT-3 (P4) — the `method='lm'` path bypasses `merit_fn`, silently losing checkpointing and telemetry
The LM branch drives `least_squares` with its own `residuals(x)`
closure, which re-implements the eval counter and progress emission but
**not** the rest of `merit_fn`'s bookkeeping: no best-merit tracking,
no history rows, no rolling `_state_save()` (checkpoints), and no
`plane_logger` callback.  Consequences: a multi-hour `method='lm'` run
with `state_file=` set writes **no checkpoint until the final
force-save** — a crash mid-run loses everything, defeating the v4.16
resume feature the user explicitly opted into — and per-eval telemetry
consumers silently receive nothing.  Every other method routes through
`merit_fn` and gets all four behaviours.  **Fix**: have `residuals`
delegate the bookkeeping (or call `merit_fn` once per eval and reuse
its ctx), or document the LM exclusions on the `state_file` /
`plane_logger` parameters.

### Nits
* The `Constraint.__post_init__` transitional `DeprecationWarning`
  ("scheduled for removal in v5.0") still fires — once per process, on
  the first Constraint construction, for **every** user including ones
  who never saw v4.16.1 — twenty-one minor versions past its removal
  date.  Stale-deprecation class; delete it.
* DE / basin-hopping / dual-annealing hard-code `seed=42` with no
  user-facing seed parameter: runs are reproducible but two "independent"
  global-search attempts are identical, and there is no way to
  restart the stochastic search with a different seed short of
  reparameterising the problem.
* `DesignResult.converged = getattr(res, 'success', True)` defaults to
  **True** when the scipy result lacks `success` (basin-hopping's
  result object) — an optimistic report of convergence.
* `DesignResult.iterations = call_count[0]` is the merit-**eval**
  count, not the iteration count (`iter_count` exists but isn't
  returned) — mislabeled field.

## 4. Coverage statement

With this pass, `optimize/` is line-audited end to end: the only
remaining structural-only corners are `merit_terms.py`'s
`MatchIdealThinLens`/`MatchIdealSystem` construction plumbing
(`_make_source`/`_build_real_elements`/`_propagate`, ~210-640 — thin
orchestration over already-audited sources/propagators, with the metric
kernels themselves verified in the first pass) and `multi_objective`'s
pymoo `_evaluate` body.  Subsystem findings ledger: OPT-1 (P3, JAX
Strehl merit direction), OPT-2 (P3, ToleranceAwareMerit sub-context),
OPT-3 (P4, LM checkpoint bypass), plus the nits here and in the three
first-pass docs.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion docs: `AUDIT_OPTIMIZE_MERITS_2026_07_09.md`,
`AUDIT_OPTIMIZE_DRIVER_2026_07_09.md`,
`AUDIT_OPTIMIZE_WRAPPERS_2026_07_09.md`,
`AUDIT_OPTIMIZE_TAIL_2026_07_09.md`.*
