<!-- lumenairy-history-doc
module: lumenairy/optimize/context.py
ast_sha256: 13e850f4b689f3a92332efb9b27f922c2342feb6599ea3b9c9a4f711bfd7c541
token_sha256: 5b35469bc6d5e5b25b9b676829d68b4278995cae377a976261059270c1728602
pre_relocation_lines: 605
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/optimize/context.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/context.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

Almost all of it is one story told four times: **the `_ZERO_APERTURE_MASK`
semantics flip and the sentinel-plumbing consolidation that followed it.**  A
scalar `aperture_diameter = 0` once produced an all-False boolean mask, one
release collapsed that branch into `mask = None`, and the two mean opposite
things.  The RULE (they must not be conflated, and what each means) is live and
stayed in the source at every site; the release pair moved here.

The second thread is the `Constraint` auto-probe removal.  That one is a
**live migration statement** -- a caller who relied on the automatic
`fun(np.zeros(1))` shape check has to call `.validate()` now, and a one-cycle
`DeprecationWarning` tells them so -- so the whole rationale (why the probe was
expensive, why its swallowed exceptions made the waste invisible, what to call
instead) stayed with the warning.  Only the release attributions moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L4-8 | `<module> docstring` | the release/agent tag, "Pre-v5.1.0 everything lived in core.py", and "the split is mechanical (no public API change)" |
| L98-106 | `_ZERO_APERTURE_MASK` | the release pair that introduced and undid the zero-aperture semantics flip |
| L108-115 | `_ZeroApertureMaskSentinel` | the release/agent tag and the three-way duplication that predated the shared base class |
| L129-134 | `sentinel promotion note` | the release/agent/audit tag and the "Pre-v4.15.2 these were bare scalar fallbacks" framing |
| L155-160 | `_InvalidFocalLengthSentinel` | the release tags around the scalar-then-singleton change |
| L182-188 | `_FailedScanStrehlSentinel` | the release tags |
| L204-213 | `the absent perturbed-ABCD sentinel` | that the class and its singleton once existed as dead code, and which release deleted them |
| L378-388 | `_CONSTRAINT_AUTOPROBE_DEPRECATION_WARNED` | the three release tags tracing the auto-probe's introduction, silent removal and the latch that announced it |
| L419-427 | `Constraint docstring` | the release/audit tag and "has been REMOVED" |
| L429-430 | `Constraint docstring` | the release pair in the section heading |
| L436-438 | `Constraint docstring` | the release tags around the heuristic-to-probe change |
| L467-469 | `Constraint docstring example` | the release/audit tag and "so copy-paste users don't trigger the v4.16.1 lambda warning" |
| L494-506 | `Constraint.__post_init__` | the release/audit tags and the "instead of the v4.16.1 heuristic" framing |
| L508-513 | `Constraint.__post_init__` | the release/audit tag and "widen the catch list" |
| L538-545 | `Constraint.__post_init__` | "the v4.16.1 auto-probe used to live here -- removed because" |
| L550-554 | `Constraint.validate` | "Pre-v4.16.2 this ran automatically" |

---

### L4-8 -- `<module> docstring` -- the release/agent tag, "Pre-v5.1.0 everything lived in core.py", and "the split is mechanical (no public API change)"

*Left in the source:* what the module hosts and the live re-export fact -- both import paths work.

```text
v5.1.0 split (Agent E): this module hosts the data containers + sentinel
plumbing that the merit-term hierarchy and ``design_optimize`` depend
on.  Pre-v5.1.0 everything lived in ``lumenairy/optimize/core.py``; the
split is mechanical (no public API change) and ``optimize/core.py`` now
re-exports every previously-public name from this module.
```

### L98-106 -- `_ZERO_APERTURE_MASK` -- the release pair that introduced and undid the zero-aperture semantics flip

*Left in the source:* the distinction itself and the full consequence of conflating the two, re-stated as a present-tense hazard, plus the `is`-comparison contract callers use.

```text
# v4.14.1 (P1-NEW-1): sentinel meaning "aperture explicitly zero, block
# all light."  Distinguished from ``mask is None`` ("no aperture
# specified, use full grid").  Pre-v4.14.0 a scalar
# ``aperture_diameter=0`` produced an all-False boolean mask, which
# downstream apply_real_lens treated as "block all light"; v4.14.0
# collapsed that branch into ``mask=None``, flipping the semantics so
# ``aperture_diameter=0`` instead produced a grid-filling plane wave.
# Callers compare ``mask is _ZERO_APERTURE_MASK`` to detect the
# deliberate-zero case and zero their field accordingly.
```

### L108-115 -- `_ZeroApertureMaskSentinel` -- the release/agent tag and the three-way duplication that predated the shared base class

*Left in the source:* why the base class is inherited -- the pickle-safe `__reduce__` and what its absence does to `is`-identity across a process boundary -- plus the pointer to this file.

```text
# v4.15.1 (Agent E): now inherits from ``_deprecation._Sentinel`` to
# share the singleton-name registry + pickle-safe ``__reduce__``
# protocol.  Pre-v4.15.1 this class duplicated the singleton plumbing
# in 3 places (here, ``_AngleUnsetSentinel`` in ``polarization.py``,
# and ``_Sentinel`` in ``_deprecation.py``); none carried a
# ``__reduce__``, so pickling a sentinel produced a NEW instance on
# the receiving side and broke ``is``-identity checks in distributed
# merit evaluation / joblib caches.
```

### L129-134 -- `sentinel promotion note` -- the release/agent/audit tag and the "Pre-v4.15.2 these were bare scalar fallbacks" framing

*Left in the source:* the whole live contract: scalar storage preserved at the call sites, registry membership for identity checks, the `.value` attribute, the `__bool__ -> False` inheritance and the naming convention.

```text
# v4.15.2 (Agent E, AUDIT_V4_15_1 P2): three additional pre-existing
# sentinel patterns in this module are promoted to ``_Sentinel``
# subclasses for pickle-safety + ``is``-identity discoverability.  Pre-
# v4.15.2 these were bare scalar fallbacks (``1e9`` for invalid focal
# length, ``0.0`` for failed-scan Strehl, and a "fall-back-to-nominal"
# marker for perturbed-ABCD failures).  Scalar storage is preserved at
```

### L155-160 -- `_InvalidFocalLengthSentinel` -- the release tags around the scalar-then-singleton change

*Left in the source:* the complete dual contract -- the branch still writes 1e9, ctx_is_valid still recovers the semantics by magnitude, and the singleton is there for a strict identity check.

```text
    Used at the wave-leg ABCD failure branch.  Pre-v4.15.2 that branch
    wrote a bare scalar ``efl = bfl = 1e9``; the magnitude-check
    downstream (``ctx_is_valid``) recovered the "invalid" semantics by
    comparing ``abs(v) >= _INVALID_FL_SENTINEL * 0.5``.  v4.15.2 keeps
    the scalar write (arithmetic stability) and adds this singleton so
    a future caller wanting a strict identity check
```

### L182-188 -- `_FailedScanStrehlSentinel` -- the release tags

*Left in the source:* why 0.0 is the right scalar fallback (the optimizer reads it as "very bad design" without dragging the parameter vector past the dispatcher's adaptive-step safeguards) and the dual scalar/singleton contract.

```text
    Used at the through-focus-scan exception branches.  Pre-v4.15.2 the
    branch wrote ``sub_ctx.strehl_best = 0.0``; the optimizer treats
    ``0.0`` as "very bad design" so the merit-leg contribution sinks
    into the noise floor without dragging the parameter vector further
    than the dispatcher's adaptive-step safeguards allow.  v4.15.2
    keeps the scalar write and adds this singleton for identity
    discoverability.
```

### L204-213 -- `the absent perturbed-ABCD sentinel` -- that the class and its singleton once existed as dead code, and which release deleted them

*Left in the source:* the live reason no such sentinel exists: the branch writes a 2-tuple, and wrapping it in one singleton would break downstream unpacking.  Stated as a standing prohibition rather than as a deletion record.

```text
# v4.15.4 (audit AUDIT_V4_15_3 P2-NEW-F1-B option a): the previously
# defined ``_PerturbedABCDFallbackSentinel`` class and its singleton
# ``_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ`` were dead code -- never
# wired at the intended callsite (the tolerance-perturbation ABCD
# failure branch at ``ToleranceAwareMerit.evaluate``).  The branch
# writes a 2-tuple fallback ``(efl_p, bfl_p) = (ctx.efl, ctx.bfl)``
# rather than a single scalar, and wrapping the tuple in a single
# sentinel singleton would break downstream unpacking.  v4.15.4 deletes
# the class + singleton outright; see the historical comment in the
# v4.15.4 release notes.
```

### L378-388 -- `_CONSTRAINT_AUTOPROBE_DEPRECATION_WARNED` -- the three release tags tracing the auto-probe's introduction, silent removal and the latch that announced it

*Left in the source:* the entire migration rationale -- it is a LIVE DeprecationWarning, so why the probe is gone, what replaces it, and why the latch is at module level all stayed.

```text
# v4.16.3 (audit P2-NEW-F1-1): one-cycle DeprecationWarning latched at
# module level, pattern parity with v4.16.2 MultiWavelengthMerit.
# v4.16.1 shipped a ``Constraint.__post_init__`` auto-probe that called
# ``fun(np.zeros(1))`` to shape-check the return.  v4.16.2 silently
# removed it (the probe was expensive for BFL-style ``fun`` callables
# that internally ran a full ray-trace) and moved the contract to an
# opt-in :meth:`Constraint.validate` method.  Emit a one-cycle
# DeprecationWarning so callers that came to rely on the v4.16.1
# auto-probe notice the change and call ``.validate()`` explicitly.
# Latched at module level so an optimisation loop that builds many
# ``Constraint(...)`` objects doesn't flood the warning channel.
```

### L419-427 -- `Constraint docstring` -- the release/audit tag and "has been REMOVED"

*Left in the source:* the whole cost argument as a present-tense statement of why there is no probe, and the instruction to call validate() explicitly.

```text
    v4.16.2 (audit P2-NEW-F1-1): the automatic ``fun(np.zeros(1))``
    probe formerly run in :meth:`__post_init__` has been REMOVED.
    For a BFL-style ``fun`` that internally calls
    :func:`system_abcd` / :func:`apply_real_lens`, the probe ran an
    entire trace on every ``Constraint(...)`` instantiation (e.g.
    once per optimisation set-up and once per parallel-worker fork).
    Caught exceptions were swallowed silently so users couldn't even
    see the wasted work.  Call :meth:`validate` explicitly after
    construction if you want the (best-effort) shape check.
```

### L429-430 -- `Constraint docstring` -- the release pair in the section heading

*Left in the source:* the heading and the entire pickle-safety contract under it.

```text
    Pickle-safety contract (v4.16.1 / v4.16.2)
    -------------------------------------------
```

### L436-438 -- `Constraint docstring` -- the release tags around the heuristic-to-probe change

*Left in the source:* why a pickle probe beats a `__name__` heuristic, and the full list of callables each catches.

```text
    v4.16.2 (audit P2-NEW-F1-2) replaces the v4.16.1
    ``getattr(fun, '__name__', None) == '<lambda>'`` heuristic with
    a direct :func:`pickle.dumps` probe so closures
```

### L467-469 -- `Constraint docstring example` -- the release/audit tag and "so copy-paste users don't trigger the v4.16.1 lambda warning"

*Left in the source:* why the example is written with a module-level function.

```text
    >>> # Require sum(x) <= 1 exactly.  v4.16.2 (audit P3-NEW-F1-7):
    >>> # docstring example now uses a module-level function so
    >>> # copy-paste users don't trigger the v4.16.1 lambda warning.
```

### L494-506 -- `Constraint.__post_init__` -- the release/audit tags and the "instead of the v4.16.1 heuristic" framing

*Left in the source:* the entire argument for the probe, including the three callable patterns a `__name__` check misses and the parallel-eval failure they produce.

```text
        # v4.16.2 (audit P2-NEW-F1-2): pickle-probe instead of the
        # v4.16.1 ``__name__ == '<lambda>'`` heuristic.  The
        # ``__name__`` check missed closures (``def inner(x): ...``
        # has ``__name__ == 'inner'``) and
        # ``functools.partial(lambda x: ..., ...)`` (whose
        # ``__name__`` attribute does not exist at all) -- both are
        # genuinely unpicklable and both will fail under
        # ``differential_evolution(workers>1)`` / joblib-parallelised
        # FD-gradients with PicklingError at the first parallel
        # eval, which is exactly the failure mode the v4.16.1 closure
        # was meant to prevent.  A direct ``pickle.dumps(self.fun)``
        # probe catches all three patterns at construction time
        # cheaply (most ``fun`` callables are tens of bytes pickled).
```

### L508-513 -- `Constraint.__post_init__` -- the release/audit tag and "widen the catch list"

*Left in the source:* why the catch is `Exception` -- the four exception classes the narrow tuple misses, the best-effort framing, and the deliberate BaseException carve-out.

```text
        # v4.16.3 (audit P2-NEW-F1-2): widen the catch list from
        # ``(pickle.PicklingError, AttributeError, TypeError)`` to
        # ``Exception``.  The narrow tuple missed ``RecursionError``
        # (deep object graph), ``RuntimeError`` (raised by a custom
        # ``__reduce__``), ``MemoryError`` (huge object), and arbitrary
        # exceptions from ``__reduce__`` / ``__getstate__``.  A ``fun``
```

### L538-545 -- `Constraint.__post_init__` -- "the v4.16.1 auto-probe used to live here -- removed because"

*Left in the source:* the same cost argument as a standing note on why no probe runs here, and the pointer to validate().

```text
        # v4.16.2 (audit P2-NEW-F1-1): the v4.16.1 ``fun(np.zeros(1))``
        # auto-probe used to live here -- removed because for a
        # BFL-style ``fun`` that calls ``system_abcd(...)`` internally
        # it would run an entire trace on every ``Constraint(...)``
        # instantiation, swallowing the exception silently.  Users
        # who want the shape check can call ``self.validate()``
        # explicitly after construction; see :meth:`validate` for
        # the same scalar-only contract.
```

### L550-554 -- `Constraint.validate` -- "Pre-v4.16.2 this ran automatically"

*Left in the source:* why it is opt-in: the probe is expensive for the canonical BFL / EFL pattern and its caught exceptions are invisible to the caller.

```text
        Pre-v4.16.2 this ran automatically in :meth:`__post_init__`
        but the probe was expensive for the canonical BFL / EFL
        constraint pattern (runs a full ray-trace) and the caught
        exceptions were swallowed silently.  Now opt-in -- call
        explicitly after construction if you want the shape check.
```

