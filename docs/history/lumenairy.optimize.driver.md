<!-- lumenairy-history-doc
module: lumenairy/optimize/driver.py
ast_sha256: 12007fb56de1eb90c27b13035eeca2a53e9939133007a51385e3d6ed329cc68c
token_sha256: d4d34e8b2e3b7a11dd662b170b12c34b2ff2f3515a2aa5b292f00d6b3a9e95f8
pre_relocation_lines: 1674
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
-->


# Version history -- `lumenairy/optimize/driver.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/driver.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

`driver.py` is the scipy dispatch layer, so most of its comments are of the
form "scipy behaves like X; here is what we do about it".  Those are why-
comments and stayed.  What moved is the release framing wrapped around them --
`vN.N (audit …)` prefixes, and the "pre-fix the whitelist was …" / "pre-fix
scipy warned …" clauses that turn a live hazard into a changelog entry.  Every
one of those hazards is still live: a shorter bounds whitelist still drops
bounds silently, `'maxiter'` still does nothing on TNC, `b[0] if b else` still
misreads a `(None, ub)` tuple.  They are now stated as what WOULD happen,
which is the form that still helps the next reader.

One block was deliberately left almost intact: the `.. versionchanged:: 5.30`
on `design_optimize` documenting that `wave_traced` now raises `TypeError`,
with its migration recipe and the grep-verification that made the removal
safe.  That is a live instruction to an upgrading caller, not narrative; only
the "deprecated earlier in v5.30, W5 shim-removal wave" phrase moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L4-4 | `<module> docstring` | the release/agent split tag |
| L8-8 | `<module> docstring` | "for bit-for-bit public-API preservation" |
| L19-21 | `logger import` | the release / roadmap-sweep tag |
| L126-132 | `_wave_real_lens` | the release/wave tag and the removal narrative for the `wave_traced` branch |
| L178-183 | `_wave_asymptotic` | the release/audit tag and "the previous `+0.5`" |
| L284-284 | `_fd_grad_pure` | the release and audit tag |
| L303-304 | `_fd_grad_pure` | "this is the historical default and preserves bit-identical gradient values with pre-v4.13.0 behaviour" |
| L380-380 | `_fd_grad_pure` | the release and audit tag |
| L502-504 | `design_optimize` | "the body lives here post-v5.1.0 split.  Parameter and behaviour contracts are unchanged." |
| L507-508 | `design_optimize versionchanged` | "(deprecated earlier in v5.30, W5 shim-removal wave)" |
| L521-526 | `design_optimize` | the release/audit tag and "the historical hard-coded value -- so existing callers see byte-identical behaviour" |
| L558-558 | `design_optimize / _RestoreDtype` | the release and audit tag |
| L595-599 | `design_optimize` | the release/audit tag and "threading the propagator through the sub-merit is a v4.14+ feature" |
| L633-633 | `design_optimize` | the release / roadmap tag |
| L661-661 | `design_optimize` | the release / roadmap tag |
| L764-765 | `design_optimize` | the release / roadmap-sweep tag |
| L810-811 | `design_optimize` | the release / roadmap-sweep tag |
| L896-896 | `design_optimize` | the release/wave tag and "dropped from opts with the kwarg" |
| L956-956 | `design_optimize / evaluate` | the release and audit tag |
| L1016-1017 | `_fd_grad_for` | "preserve bit-identical gradient values with pre-v4.13.0 behaviour" |
| L1055-1055 | `_combined_jac` | the release and audit tag |
| L1088-1092 | `_post_eval_bookkeeping` | the "pre-fix that path re-implemented only the eval counter" framing |
| L1105-1110 | `_post_eval_bookkeeping` | the release/roadmap tag and "so pre-v4.16 callers see byte-identical behaviour" |
| L1166-1166 | `design_optimize callbacks` | the release and audit tag |
| L1196-1201 | `_dual_annealing_cb` | the release/audit tag and the "was an inline lambda" framing |
| L1217-1217 | `design_optimize dispatch` | the release and audit tag |
| L1277-1289 | `_resolve_bound` | the release/audit tag and the "Pre-v4.16.1 used ..." framing |
| L1290-1295 | `_resolve_bound` | the release/audit tag and the "Pre-v4.16.2 the helper silently picked" framing |
| L1317-1324 | `design_optimize / lm` | the release/audit tag and the "Pre-v4.16.2 the override was invisible" framing |
| L1367-1369 | `design_optimize / basinhopping` | "Pre-fix the local search always finite-differenced" |
| L1449-1449 | `design_optimize / newton` | the release / roadmap tag |
| L1567-1567 | `design_optimize / minimize` | the release / roadmap tag |
| L1572-1582 | `design_optimize / minimize` | the release/audit tag and the "Pre-fix the whitelist was ..." framing |
| L1599-1604 | `design_optimize / minimize` | the release/audit tag and the "Pre-fix scipy warned" framing |
| L1606-1614 | `design_optimize / minimize` | the release tag on the 'disp' removal |
| L1637-1641 | `design_optimize` | the release/roadmap tag and "so pre-v4.16 callers see byte-identical behaviour" |
| L1654-1654 | `design_optimize finally` | the release and audit tag |

---

### L4-4 -- `<module> docstring` -- the release/agent split tag

*Left in the source:* the split relationship.

```text
v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
```

### L8-8 -- `<module> docstring` -- "for bit-for-bit public-API preservation"

*Left in the source:* the re-export fact.

```text
``optimize/core.py`` for bit-for-bit public-API preservation.
```

### L19-21 -- `logger import` -- the release / roadmap-sweep tag

*Left in the source:* what the logger is for and the default-quiet guarantee.

```text
# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
# Module-level logger for design_optimize entry + per-scipy-iteration
# progress.  Default-quiet via the lumenairy root logger's NullHandler.
```

### L126-132 -- `_wave_real_lens` -- the release/wave tag and the removal narrative for the `wave_traced` branch

*Left in the source:* the live rule -- register a propagator, do not gate on a boolean -- the reason (a flag that mutates the meaning of another argument), and the whole copy-paste recipe that follows.

```text
    # v5.30 (W5 shim-removal wave): the ``opts['wave_traced']`` branch that
    # routed to ``apply_real_lens_traced`` is REMOVED with the
    # ``design_optimize(wave_traced=)`` flag that was its only gate (R-17
    # grep-verified zero callers repo-wide, so CI never covered it).  To
    # drive ``apply_real_lens_traced`` from a design run, register a
    # propagator -- one dispatch mechanism instead of a boolean that
    # mutates the meaning of ``ray_subsample``::
```

### L178-183 -- `_wave_asymptotic` -- the release/audit tag and "the previous `+0.5`"

*Left in the source:* the convention, the library-wide list it matches, and the half-pixel consequence of getting it wrong -- re-stated as a hazard rather than as a past defect.

```text
    # v4.12.1 (B1-10): pixel-centred `(arange(N) - N/2)*dx`, matches the
    # library-wide convention (ASM, Fresnel, RS, sources).  Merit
    # functions that compare wave-leg fields across propagator
    # families need a single shared grid convention; the previous
    # `+0.5` produced a half-pixel offset between the asymptotic leg
    # and the ASM / GBD / HF legs.
```

### L284-284 -- `_fd_grad_pure` -- the release and audit tag

*Left in the source:* the whole note about which paths actually consume `scale_floor`, which is the part a caller tuning it needs.

```text
        / thicknesses).  v5.4.6 (audit F-20): note that ``scale_floor``
```

### L303-304 -- `_fd_grad_pure` -- "this is the historical default and preserves bit-identical gradient values with pre-v4.13.0 behaviour"

*Left in the source:* that central differences are the default, their eval count and their truncation order.

```text
        error); this is the historical default and preserves bit-
        identical gradient values with pre-v4.13.0 behaviour.
```

### L380-380 -- `_fd_grad_pure` -- the release and audit tag

*Left in the source:* what the check is and why the tolerance is tight-but-not-exact.

```text
            # v4.14 (audit P2 #16): opt-in stale-cache check.  Tight
```

### L502-504 -- `design_optimize` -- "the body lives here post-v5.1.0 split.  Parameter and behaviour contracts are unchanged."

*Left in the source:* where the canonical docstring is and that the body lives here.

```text
    See ``lumenairy.optimize.core.design_optimize`` for the canonical
    docstring; the body lives here post-v5.1.0 split.  Parameter and
    behaviour contracts are unchanged.
```

### L507-508 -- `design_optimize versionchanged` -- "(deprecated earlier in v5.30, W5 shim-removal wave)"

*Left in the source:* the rest of the versionchanged block IN FULL -- the removal, the grep-verification that made it safe, the TypeError, and the migration recipe.  A live instruction to an upgrading caller.

```text
       ``wave_traced`` is **REMOVED** (deprecated earlier in v5.30, W5
       shim-removal wave).  R-17
```

### L521-526 -- `design_optimize` -- the release/audit tag and "the historical hard-coded value -- so existing callers see byte-identical behaviour"

*Left in the source:* what `seed` controls, its default, and both alternatives.

```text
    v5.24.x (audit S4-18): ``seed`` controls the RNG of the stochastic
    global methods (``differential_evolution`` / ``basin_hopping`` /
    ``dual_annealing``).  Defaults to ``42`` -- the historical hard-coded
    value -- so existing callers see byte-identical behaviour; pass a
    different int for an independent stochastic restart, or ``None`` to
    let scipy draw from the unseeded global RNG.
```

### L558-558 -- `design_optimize / _RestoreDtype` -- the release and audit tag

*Left in the source:* the entire argument for try/finally over `__del__`, including the CPython/PyPy refcount caveat and the `_restored` double-fire guard.

```text
    # v4.14 (audit P2 #10): the dominant restore path is now an
```

### L595-599 -- `design_optimize` -- the release/audit tag and "threading the propagator through the sub-merit is a v4.14+ feature"

*Left in the source:* the condition for the warning and the live reason it exists: the propagator is not threaded through the sub-merit.

```text
    # v4.14 (audit P2 #14): warn if the user selected a non-default
    # wave_propagator (e.g. 'gbd') AND any of the three Merit classes
    # that hard-code apply_real_lens for off-nominal legs is in use.
    # Threading the propagator through the sub-merit is a v4.14+
    # feature; this warning surfaces the silent inconsistency.
```

### L633-633 -- `design_optimize` -- the release / roadmap tag

*Left in the source:* the whole constraint-validation contract and the SLSQP / trust-constr restriction.

```text
    # v4.16 (ROADMAP #9): hard-constraint validation + method-compat
```

### L661-661 -- `design_optimize` -- the release / roadmap tag

*Left in the source:* the entire checkpoint/resume contract.

```text
    # v4.16 (ROADMAP #10): state-file checkpoint/resume.  Persist the
```

### L764-765 -- `design_optimize` -- the release / roadmap-sweep tag

*Left in the source:* what the entry log carries and why.

```text
    # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
    # Entry log -- method + free-param count + merit-term count + iter
```

### L810-811 -- `design_optimize` -- the release / roadmap-sweep tag

*Left in the source:* the per-iteration record and its parity with the progress callback.

```text
        # v5.3.2 (ROADMAP logging adoption sweep -- per-iteration
        # telemetry): one INFO record per scipy iteration -- mirrors
```

### L896-896 -- `design_optimize` -- the release/wave tag and "dropped from opts with the kwarg"

*Left in the source:* the live fact that `opts` carries no `wave_traced`, and that `ray_subsample` is the documented channel for a user-registered traced propagator.

```text
            # v5.30 (W5): ``wave_traced`` dropped from ``opts`` with the
```

### L956-956 -- `design_optimize / evaluate` -- the release and audit tag

*Left in the source:* the NaN-argmax hazard and the wrapper-merit parity note.

```text
                # v5.4.6 (audit F-5): NaN-safe argmax -- a single NaN
```

### L1016-1017 -- `_fd_grad_for` -- "preserve bit-identical gradient values with pre-v4.13.0 behaviour"

*Left in the source:* the eval-count trade-off between central and forward differences.

```text
            (the default) preserve bit-identical gradient values with
            pre-v4.13.0 behaviour at 2N evaluations per gradient.
```

### L1055-1055 -- `_combined_jac` -- the release and audit tag

*Left in the source:* the entire f0-reuse argument, which is why forward-FD is chosen here.

```text
        # v4.14 (audit P2 #11): switch to forward-FD with a cached
```

### L1088-1092 -- `_post_eval_bookkeeping` -- the "pre-fix that path re-implemented only the eval counter" framing

*Left in the source:* the full consequence -- no checkpoint until the final force-save, no per-eval telemetry -- as what a partial re-implementation costs.

```text
        ``method='lm'`` ``residuals`` path gets ALL of it too: pre-fix that
        path re-implemented only the eval counter + progress, so a multi-hour
        LM run with ``state_file=`` set wrote no checkpoint until the final
        force-save (a mid-run crash lost everything) and per-eval telemetry
        consumers silently received nothing."""
```

### L1105-1110 -- `_post_eval_bookkeeping` -- the release/roadmap tag and "so pre-v4.16 callers see byte-identical behaviour"

*Left in the source:* the monotonic-improvement argument and the reason the bookkeeping is gated on state_file.

```text
        # v4.16 (ROADMAP #10): track best-merit-seen for checkpoint /
        # resume.  The optimiser is guaranteed to call merit_fn at the
        # actual converged x_opt at the end (in the final
        # evaluate() block), so x_best is monotonic-improving.
        # Gated on state_file being non-None so pre-v4.16 callers see
        # byte-identical behaviour (no per-eval bookkeeping cost).
```

### L1166-1166 -- `design_optimize callbacks` -- the release and audit tag

*Left in the source:* the whole cancellation protocol and which scipy methods honour a True return.

```text
    # v4.14 (audit P2 #13): honour the progress cancellation protocol
```

### L1196-1201 -- `_dual_annealing_cb` -- the release/audit tag and the "was an inline lambda" framing

*Left in the source:* why the callback is named and what a non-polling callback costs -- a Qt Stop press silently ignored.

```text
    # v4.13.2 (P1-NEW-L): dual_annealing's callback was an inline
    # lambda that did NOT poll ``is_cancelled(progress)`` -- a Qt
    # ``Stop`` press during a dual_annealing run was silently
    # ignored.  Promote to a named callback matching the pattern of
    # the other three scipy callbacks; returning True asks
    # dual_annealing to terminate the run.
```

### L1217-1217 -- `design_optimize dispatch` -- the release and audit tag

*Left in the source:* the try/finally rationale and the __del__ safety-net note.

```text
    # v4.14 (audit P2 #10): wrap the dispatch + final evaluation in
```

### L1277-1289 -- `_resolve_bound` -- the release/audit tag and the "Pre-v4.16.1 used ..." framing

*Left in the source:* the entire truthiness trap, the leak into np.array, and scipy's bounds spec -- all still live reasons the explicit check exists.

```text
            # v4.16.1 (AUDIT_V4_16_0_DEEP P1-DEEP-1-2): explicit
            # ``None``-aware unpacking.  Pre-v4.16.1 used
            # ``b[0] if b else -np.inf``, where ``b`` is a 2-tuple
            # ``(lb_i, ub_i)``; a non-empty tuple is ALWAYS truthy in
            # Python, so the conditional never fired even when either
            # endpoint was ``None`` (the "no bound on this side"
            # idiom).  ``None`` then leaked into ``np.array(...)``,
            # producing an object-dtype array that scipy's
            # ``least_squares`` rejects with an opaque downstream
            # error.  scipy's ``bounds=(lb, ub)`` spec requires each
            # endpoint to be either a finite float or +/-inf, not
            # ``None``.  Explicit per-endpoint check below honours the
            # idiom and produces a clean float64 array.
```

### L1290-1295 -- `_resolve_bound` -- the release/audit tag and the "Pre-v4.16.2 the helper silently picked" framing

*Left in the source:* the 3-tuple mistake the guard catches and the formats a user confuses.

```text
            # v4.16.2 (audit P3-NEW-F1-3): length guard.  Pre-v4.16.2
            # the helper silently picked ``b[0]`` / ``b[1]`` from
            # ANY indexable -- so a 3-tuple ``(lb, ub, extra)`` (a
            # genuine user mistake, e.g. mixing up scipy bounds /
            # least_squares bounds / DE bounds formats) would parse
            # cleanly with the 3rd element dropped.  Raise instead
```

### L1317-1324 -- `design_optimize / lm` -- the release/audit tag and the "Pre-v4.16.2 the override was invisible" framing

*Left in the source:* scipy's lm/bounds contract, the silent switch to trf, and the test-naming evidence of how invisible it is.

```text
            # v4.16.2 (audit P3-NEW-F1-8): scipy's least_squares
            # contract is that ``method='lm'`` does NOT accept
            # bounds; passing both forces a silent switch to
            # ``'trf'``.  Pre-v4.16.2 the override was invisible to
            # the user: test names of the form
            # ``test_bug4_lm_bounds_*`` documented "lm" while the
            # production path actually ran "trf".  Warn at the
            # override point so the user knows.
```

### L1367-1369 -- `design_optimize / basinhopping` -- "Pre-fix the local search always finite-differenced"

*Left in the source:* the rule and its cost, stated as what happens without the forward.

```text
            # user-supplied ``jac`` callable).  Pre-fix the local search
            # always finite-differenced even when an exact gradient was
            # in hand.
```

### L1449-1449 -- `design_optimize / newton` -- the release / roadmap tag

*Left in the source:* the whole Newton-step rationale and its problem-size guidance.

```text
            # v4.16 (ROADMAP #12): Hessian / Newton-step.  For small
```

### L1567-1567 -- `design_optimize / minimize` -- the release / roadmap tag

*Left in the source:* the constraint translation and the note that compatibility was validated up front.

```text
            # v4.16 (ROADMAP #9): thread hard constraints through
```

### L1572-1582 -- `design_optimize / minimize` -- the release/audit tag and the "Pre-fix the whitelist was ..." framing

*Left in the source:* the whole argument including the measured probe (bounded Powell converging to x=2.0 with bounds [(0, 1)]) and the warn-rather-than-drop rule.

```text
            # v5.17.x (AUDIT_V5_17_0 P2-24): forward ``bounds`` for
            # EVERY minimize method that honours them.  Pre-fix the
            # whitelist was ('L-BFGS-B', 'SLSQP', 'trust-constr'), so
            # user-supplied bounds were SILENTLY dropped for Powell /
            # Nelder-Mead / TNC / COBYLA / COBYQA -- methods scipy
            # box-constrains natively -- and the optimizer freely
            # walked outside the user's stated box (probe: bounded
            # Powell converged to x=2.0 with bounds [(0, 1)]).  For
            # methods that truly cannot handle bounds we keep passing
            # None but warn loudly (parity with the method='lm'
            # bounds warning above) instead of dropping silently.
```

### L1599-1604 -- `design_optimize / minimize` -- the release/audit tag and the "Pre-fix scipy warned" framing

*Left in the source:* the TNC option-name fact and the silent-ineffectiveness it causes, as a live hazard.

```text
            # v5.17.x (AUDIT_V5_17_0 wave-5 follow-up): TNC does not
            # take a 'maxiter' option -- its evaluation budget is
            # 'maxfun'.  Pre-fix scipy warned 'Unknown solver
            # options: maxiter' and ran with its DEFAULT budget, so
            # ``max_iter`` was silently ineffective for TNC.  Map the
            # option name per-method.
```

### L1606-1614 -- `design_optimize / minimize` -- the release tag on the 'disp' removal

*Left in the source:* the whole scipy 1.18 compatibility argument -- which is why 'disp' must NOT be re-added -- plus the pointer to this file.

```text
            # v5.18.0: 'disp' is NO LONGER passed as a solver option.
            # scipy 1.18.0 tightened per-method option validation and now
            # rejects 'disp' for L-BFGS-B (and likely other methods),
            # emitting ``OptimizeWarning: Unknown solver options: disp``
            # under every generic-minimize call (scipy <= 1.17 accepted
            # it).  The driver already prints its own iteration progress
            # from the merit callback when ``verbose`` is set, so scipy's
            # internal ``disp`` was redundant; dropping it keeps the
            # generic path option-clean on scipy 1.17 AND 1.18.
```

### L1637-1641 -- `design_optimize` -- the release/roadmap tag and "so pre-v4.16 callers see byte-identical behaviour"

*Left in the source:* why the final state is force-saved and why it is gated on state_file.

```text
        # v4.16 (ROADMAP #10): also force-save the final state so the
        # checkpoint file reflects the converged solution, regardless
        # of where state_save_every left the rolling-save counter.
        # Gated on state_file so pre-v4.16 callers see byte-identical
        # behaviour.
```

### L1654-1654 -- `design_optimize finally` -- the release and audit tag

*Left in the source:* the restore contract on every exit path.

```text
        # v4.14 (audit P2 #10): explicit deterministic restore.  Runs on
```

