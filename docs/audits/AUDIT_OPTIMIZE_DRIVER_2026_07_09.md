# Optimize Driver + Parameterizations Audit — 2026-07-09

Scope: the orchestration core of `optimize/` — `driver.py` (1,413 —
`design_optimize`, the FD-gradient helpers, the `jac='auto'` JAX+FD
routing, the wave-propagator registry, checkpoint/resume) and
`parameterizations.py` (473 — `DesignParameterization`,
`RawParameterization`, `MultiPrescriptionParameterization`, the path
read/write and scale-floor classification).  Second `optimize/`
tranche, after the merit terms.  Read-only; the FD stencils and the
gradient assembly re-derived.

---

## 1. Verdict

**Clean — no findings above nit level.**  The driver is robust and
correct, with the one real optimize bug being OPT-1 (in the merit, not
here).  Verified this pass:

* **`_fd_grad_pure`** — central (`(f₊−f₋)/2h`) and forward
  (`(f₊−f₀)/h`) schemes; per-variable relative step
  `eps·max(|xᵢ|, scale_floorᵢ)`; the P3-47 bounds-aware stencil
  clipping (clip legs to the box, use the *actual* shrunken span, fall
  back to a one-sided difference pointing inward when a variable is
  pinned at a bound, zero on a degenerate `lb==ub`) — all correct, and
  the unclipped path stays byte-identical to the historical
  `(fp−fm)/(2·step)`.  The P2-16 `validate_f0` stale-cache opt-in is a
  sound guard on the forward path's `f0==f(x)` contract.
* **`_merit_jac_auto`** (the `jac='auto'` path) — the partition into
  `jax_grad_terms` (JaxMeritTerm with `build_args`) and `other_terms`
  is **disjoint** (`other_terms = [m for m in merit_terms if m not in
  jax_grad_terms]`), so summing `Σ t.gradient_at_x(x)` (analytic) +
  forward-FD of the remaining terms reconstructs the total merit's
  gradient with no double-counting; the cached `f0_other` (P2-11 perf)
  is evaluated at the same `x`.  `use_analytic_jac` correctly falls
  back to scipy's own numdiff when no JAX merit is present, and a
  user-callable `jac` is forwarded as-is.
* **`evaluate(x)`** — the ray leg (efl/bfl/seidel with every degenerate
  path caught and capped at the `1e9` sentinel), the wave leg
  (propagator-registry dispatch, through-focus scan, best-focus Strehl,
  OPD map for the Zernike merits), and the `need_ray`/`need_wave`
  gating are all sound.  The 4.10 consistent `(value, ctx)` tuple
  return (verified in the out-of-range-BFL early return), the F-5
  NaN-safe `nanargmax` for best-focus, and the seidel
  `(per_surf, totals)` sum-over-surfaces extraction all check out.
* **`RawParameterization` guard** — a `needs_ray=True` merit paired
  with a template-free (`no 'surfaces'`) build raises a clear error up
  front rather than silently degenerating to `efl/bfl = 1e9` (the
  clean fail the jax_merits `needs_ray=False` contract relies on).
* **Parameterizations** — `_read_path`/`_write_path` tuple traversal;
  the P3-50 / P3-19 duplicate-`free_vars` guard (prevents dead
  over-parameterised slots); `scale_floor` resolution (None →
  path-classified, scalar/array → broadcast with length check);
  deep-copy-on-`build`.
* **Checkpoint/resume** — JSON persist with atomic temp-write +
  `os.replace`, shape-guarded resume (`x_best.shape != (n_params,)` →
  silent fresh start), graceful I/O-failure warn-and-continue, and the
  1000-row history cap.  The best-merit tracking is monotonic.
* Constraint routing (method-support gate), progress monotonic clamp,
  and the per-iteration telemetry logging are all sound.

## 2. Findings

**None above nit level.**

### Nits
* `_classify_path_to_floor` classifies any `('surfaces', i, key)` whose
  `key` merely **starts with `'a'`** as aspheric (`key_lc.startswith('a')`),
  alongside the intended `alpha*`/`aspheric*`/`A4`.  A future non-aspheric
  surface field beginning with `a` would silently get the dimensionless
  `1e-3` FD floor.  Only affects FD step conditioning (not the result),
  and no such field exists today — latent.
* The `scale_floor` infrastructure (and the whole
  `_classify_path_to_floor` table) is **inert on the default
  `jac='auto'`-without-JAX path**: there `final_jac is None` and scipy
  estimates the gradient with its own 2-point numdiff, which never sees
  `scale_floor`.  This is documented in `_fd_grad_pure`'s F-20 note, so
  it's a disclosed limitation rather than a defect — but it means the
  per-variable floor only bites on the JAX-combined or `method='newton'`
  paths, which is easy to miss.
* `RawParameterization.build` returns `{'_raw_params': x,
  'aperture_diameter': None}`; the `None` aperture is handled by the
  P3-48 guard in `MaxFNumberMerit` and by the wave leg's
  `pres.get('aperture_diameter') or (0.4*N*dx)` fallback — consistent,
  noted for completeness.

## 3. Coverage statement

Deep-read: `parameterizations.py` in full (all three parameterization
classes + path/scale-floor helpers); `driver.py`'s FD-gradient core
(`_fd_bounds_arrays`, `_fd_grad_pure`, `_fd_grad_for`), the `jac='auto'`
assembly (`_merit_jac_auto`), the `evaluate(x)` ray+wave orchestration,
`merit_fn` state tracking, and the checkpoint/resume (`_state_load`/
`_state_save`).  Structurally covered: `design_optimize`'s scipy-method
dispatch tail (the actual `minimize` / global-optimiser calls and the
final-result assembly, ~1030-1413) and the wave-propagator registry
shims (`_wave_real_lens`/`_wave_gbd`/`_wave_hf`/`_wave_hfpi`/
`_wave_asymptotic`, which forward to already-audited propagators).
**Not audited**: `core.py` (419, re-exports + json/os aliases),
`context.py` (591, `EvaluationContext` computed properties like
`rms_wavefront_waves` — thin wrappers over already-audited
`analysis/` functions), `wrapper_merits.py` (992),
`multi_objective.py` (396), `multiconfig.py` (440), `_merit_jit.py`
(263), and the `io/` siblings — the remaining ground.

---

*Audit performed single-context against lumenairy v5.21, 2026-07-09.
Companion doc: `AUDIT_OPTIMIZE_MERITS_2026_07_09.md` (OPT-1, the
merit-sign bug this driver faithfully differentiates).*
