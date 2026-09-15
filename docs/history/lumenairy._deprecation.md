<!-- lumenairy-history-doc
module: lumenairy/_deprecation.py
ast_sha256: de1c52354641ed24fbb08c622ba620b05187b22e9e8aa2e8375d50170724d3c9
token_sha256: 5f7feeb45778e60bffba2c9730451d5dc7a1844ec1687c306ff60a3eb70cbc03
pre_relocation_lines: 688
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/_deprecation.py`

This file holds the version-history narrative that used to live in
`lumenairy/_deprecation.py`.  Each block is reproduced **verbatim** under the
source line it came from in the pre-relocation file.

This module is the clearest instance in the sweep of the pattern the audit's
sec. 14 V6 names: **each release appended a new entry to a running log inside
the source, and none of them updated the entry above.**  Three logs had grown
here:

1. a five-entry **horizon-slip log** on `NEXT_REMOVAL_VERSION` (v5.32.0,
   v5.36.1, v5.40.0, v5.43.0, v5.45.0), each recording that the horizon came
   due with both registries empty and was advanced by one line;
2. a **W5 execution log** listing every shim the v5.30 wave removed, with the
   per-module old-form -> new-form inventory the CHANGELOG's `### Removed`
   section already carries;
3. two **tombstones** -- one retired `REMOVAL_SCHEDULE` entry and one retired
   `API_TRANSITION_VERSION` entry -- each explaining what the executed entry
   used to say.

All three moved here in full.  What stayed in the source is everything that is
still true of the code: the MEASURED defect the registry exists to prevent
(`will be removed in v5.27` emitted from a v5.29.0 library; ten of twelve live
deprecations past their stated horizon), the structural reason a per-call-site
interpolation rots, the four bullets describing the live mechanism, and -- the
one piece of the tombstone that is a live invariant -- the rule that an
executed entry is DELETED rather than kept, because
`check_removal_schedule` invariant 2 requires every value to lie in the
future.

Nothing the interpreter executes changed in the move; `NEXT_REMOVAL_VERSION`
is still `'5.48'`.  The header above records the SHA-256 of (a) the module's
AST with every docstring removed and source positions ignored, and (b) its
`tokenize` stream reduced to NAME/OP/NUMBER/STRING with comments and
docstrings dropped -- both taken from the file as it stood BEFORE the
relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L11-12 | `<module> docstring` | the release / roadmap tag on the orphan-helpers note |
| L50-51 | `__all__` | the release/audit tag on the registry group |
| L57-57 | `__all__` | the release/agent tag on the sentinel helpers |
| L67-78 | `<module> removal-schedule registry` | the release/audit tag and the past-tense framing of the measured defect |
| L80-80 | `<module> removal-schedule registry` | "The fix keeps" framing |
| L102-127 | `<module> removal-schedule registry` | the W5 execution log -- every shim the wave removed, per module, and the API-transition execution note |
| L132-154 | `NEXT_REMOVAL_VERSION` | the five-entry horizon-slip log (v5.32.0 / v5.36.1 / v5.40.0 / v5.43.0 / v5.45.0), including the gating-matrix lesson the v5.36.0 tag paid for |
| L163-174 | `REMOVAL_SCHEDULE` | the tombstone of the one retired entry ('5.27' -> '5.32', the source-factory kwarg deprecations) |
| L193-219 | `API_TRANSITION_VERSION` | the tombstone of the executed propagate() return-contract flip, its reasoning and its retired warning |
| L301-302 | `_format_removal` | the "pre-v5.30 each interpolated ... which is how four independent copies of the same rot survived" framing |
| L366-366 | `_Sentinel` | the release/agent tag |
| L406-406 | `_SENTINEL_REGISTRY` | the release/agent tag |
| L421-428 | `_sentinel_unpickle` | the "pre-v4.15.2 fallback path" framing |
| L453-459 | `_NoDefaultSentinel` | the release/agent tag and the "Pre-v4.15.2 _NO_DEFAULT was a bare _Sentinel" note |

---

### L11-12 -- `<module> docstring` -- the release / roadmap tag on the orphan-helpers note

*Left in the source:* the entire keep-them rationale: no telemetry on external callers, and they document the canonical message format.

```text
v5.2 (ROADMAP opportunistic item -- "_deprecation.py orphan helpers"):
``warn_deprecated_kwarg``, ``warn_renamed_function``, and
```

### L50-51 -- `__all__` -- the release/audit tag on the registry group

*Left in the source:* the group label.

```text
    # v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A
    # "deprecation registry rot"): the removal-schedule registry.
```

### L57-57 -- `__all__` -- the release/agent tag on the sentinel helpers

*Left in the source:* the pickle protocol requirement that the unpickler be importable by name.

```text
    # v4.15.1 (Agent E): pickle-safe sentinel helpers; the unpickler
```

### L67-78 -- `<module> removal-schedule registry` -- the release/audit tag and the past-tense framing of the measured defect

*Left in the source:* the measured defect itself (v5.27 banner from a v5.29.0 library, ten of twelve past their horizon, eight saying v5.0) and the structural argument, both re-stated as what happens WITHOUT the registry, plus the pointer to this file.

```text
# v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25, Territory A
# "deprecation registry rot").  MEASURED defect: the removed-in banner
# emitted ``will be removed in v5.27`` from a v5.29.0 library -- i.e. the
# message advertised a horizon the release had already blown through.  Ten
# of twelve live deprecations were past their stated removal version (eight
# said v5.0).
#
# The bug is structural, not a typo: the four message builders below
# interpolated ``version_removed`` verbatim, so NOTHING in the library ever
# compared a stated horizon against the running ``__version__``.  Every
# call site was free to rot independently, and CI could not see it (the
# pins assert the version STRING appears, which a stale string does).
```

### L80-80 -- `<module> removal-schedule registry` -- "The fix keeps" framing

*Left in the source:* all four mechanism bullets unchanged.

```text
# The fix keeps the mechanics in exactly one place:
```

### L102-127 -- `<module> removal-schedule registry` -- the W5 execution log -- every shim the wave removed, per module, and the API-transition execution note

*Left in the source:* the live state in four lines: both registries are empty, every scheduled removal and the one API transition are executed, and the module still works for the next cycle.

```text
# v5.30 (W5 shim-removal wave): the owner EXECUTED the overdue removals
# rather than slipping them again -- see the CHANGELOG's ``### Removed``
# section for the full old-form -> new-form table.  The retired shims were
# the ones this registry existed to track:
#
#   * ``sources/core.py`` -- ``seed=`` / ``sigma=`` kwargs, the five
#     ``Source.*`` legacy positional overloads, ``create_led_source``'s
#     positional overload, and the Schell ``return_kind`` sentinel
#     (``_RETURN_KIND_UNSET`` + ``_warn_schell_return_kind_default``).
#   * ``elements/doe.py`` -- ``makedammann2d(_legacy_units='auto')``.
#   * ``propagators/gbd.py`` / ``propagators/hf.py`` -- the inert
#     ``wavelength=`` keywords.
#   * ``optimize/`` -- the ``wave_traced`` / ``use_traced_lens`` /
#     ``focus_search`` zero-caller flags.
#
# The same wave then EXECUTED the one **API transition** on the books -- the
# P5 / roadmap-F1 ``propagate()`` return-contract flip (see
# :data:`API_TRANSITION_VERSION`).  Same reasoning, one release earlier in its
# own cycle: the announcement and the flip had not yet shipped in a release, so
# waiting would have shipped a warning about a change no caller could see yet.
#
# What that leaves live here: :data:`REMOVAL_SCHEDULE` is empty, the four
# message builders are unchanged, and :data:`API_TRANSITION_VERSION` schedules
# nothing (its one entry is a tombstone).  The module stays fully functional --
# the next deprecation cycle, or the next API transition, registers here as
# before.
```

### L132-154 -- `NEXT_REMOVAL_VERSION` -- the five-entry horizon-slip log (v5.32.0 / v5.36.1 / v5.40.0 / v5.43.0 / v5.45.0), including the gating-matrix lesson the v5.36.0 tag paid for

*Left in the source:* the constant's own docstring: what the horizon is for, and that bumping it is a deliberate one-line slip recorded in the CHANGELOG.  The slip log itself is a per-release record, which is what the CHANGELOG is for.

```text
# v5.32.0 release: the horizon below had come due with NOTHING scheduled
# (both registries are empty tombstones -- every removal and the one API
# transition were EXECUTED early, in v5.30).  Advancing it is therefore
# the documented deliberate one-line slip, slipping no actual removal.
# Recorded in the CHANGELOG's 5.32.0 block.
# v5.36.1: same situation, same one-line slip -- v5.36.0 shipped and the
# horizon came due with both registries still empty tombstones.  Caught
# by the release verify shard (the schedule tests assert the horizon is
# future); the gating matrix ran pre-version-bump and could not see it.
# Recorded in the CHANGELOG's 5.36.1 block.
# v5.40.0: third deliberate one-line slip (v5.32.0 and v5.36.1
# precedents) -- the horizon came due with both registries still
# empty tombstones.  Applied PROACTIVELY in the release commit this
# time: the gating matrix runs pre-version-bump and structurally
# cannot see horizon collisions (the v5.36.0 tag paid for that
# lesson).  Recorded in the CHANGELOG's 5.40.0 block.
# v5.43.0: fourth proactive one-line slip (v5.32.0 / v5.36.1 / v5.40.0
# precedents) -- 5.43.0 ships one minor below the horizon with both
# registries still empty tombstones; slipped now so the NEXT release
# cannot collide at tag-verify.  Recorded in the CHANGELOG's 5.43.0 block.
# v5.45.0 (2026-09-10): fifth proactive slip, 5.46 -> 5.48 -- 5.45.0 ships
# one minor below the horizon with both registries still empty, and a
# 5.45.1 is scheduled behind it.  Recorded in the CHANGELOG's 5.45.0 block.
```

### L163-174 -- `REMOVAL_SCHEDULE` -- the tombstone of the one retired entry ('5.27' -> '5.32', the source-factory kwarg deprecations)

*Left in the source:* the live invariant the tombstone explained: an executed entry is DELETED, because check_removal_schedule invariant 2 requires every value to lie in the future.

```text
#: Tombstone, v5.30 (W5 shim-removal wave, owner decision: remove now
#: rather than wait for v5.32):
#:
#: * ``'5.27' -> '5.32'`` (added v5.30) -- the v5.25 ``seed=`` -> ``rng=``
#:   and ``sigma=`` -> ``w0=`` source-factory kwarg deprecations
#:   (``sources/core.py``'s ``_DEPRECATION_VERSION_REMOVED``).  The kwargs
#:   are GONE in v5.30; the entry is retired with them.  Entries are
#:   deleted rather than kept-as-history because
#:   :func:`check_removal_schedule` invariant 2 requires every value to
#:   lie in the future -- a completed removal cannot satisfy that and
#:   would turn the self-check permanently red.  The history lives here
#:   as a comment and in the CHANGELOG's ``### Removed`` section.
```

### L193-219 -- `API_TRANSITION_VERSION` -- the tombstone of the executed propagate() return-contract flip, its reasoning and its retired warning

*Left in the source:* the same delete-not-keep invariant, and the deliberately-NOT-scheduled PropagationResult.__iter__ decision with its full argument -- that one is a standing design commitment, not a record.

```text
#: Tombstone, v5.30 (roadmap ``docs/roadmap_deferred_2026_07_21.md`` Part F1,
#: audit P5 -- owner decision: flip now rather than wait for v5.32):
#:
#: * :func:`lumenairy.propagators.dispatch.propagate` -- the DEFAULT return is
#:   a :class:`~lumenairy.propagators.PropagationResult` for every method
#:   (roadmap F1 option 4, the option costed as least-breaking).  **Done in
#:   v5.30**, in the same release that announced it: the transition
#:   ``DeprecationWarning`` and its ``_caller_is_internal`` external-caller
#:   predicate are retired with it, because a warning saying "the default will
#:   become a PropagationResult in vX" cannot outlive the version that makes it
#:   one.  ``return_result=False`` keeps the legacy bare-ndarray /
#:   ``(E, dx_out, dy_out)`` shapes permanently and un-deprecated;
#:   ``return_result=True`` is unchanged.  As with the
#:   :data:`REMOVAL_SCHEDULE` entries above, the completed entry is recorded
#:   here as prose and in the CHANGELOG rather than left in a live registry
#:   field -- an executed transition cannot satisfy a "lies in the future"
#:   invariant.
#:
#: NOT scheduled here (decided against in the same pass, and NOT changed by the
#: flip):
#:
#: * ``PropagationResult.__iter__`` -- stays **2-item** ``(field,
#:   intermediates)`` permanently (audit P16).  Option 4 keeps
#:   ``return_result=False`` available, so 3-tuple unpackers migrate by
#:   naming the legacy contract instead of by us re-arity-ing iteration --
#:   which would break the ``E, inter = propagate_through_system(...,
#:   return_result=True)`` callers that the 2-item form exists for.
```

### L301-302 -- `_format_removal` -- the "pre-v5.30 each interpolated ... which is how four independent copies of the same rot survived" framing

*Left in the source:* the single-source rule and the rot mechanism, in the present tense.

```text
    below (pre-v5.30 each interpolated ``version_removed`` itself, which
    is how four independent copies of the same rot survived).  ``verb`` is
```

### L366-366 -- `_Sentinel` -- the release/agent tag

*Left in the source:* the whole pickle-safe singleton contract and why it matters across a process boundary.

```text
    v4.15.1 (Agent E): pickle-safe singleton via a name-keyed registry
```

### L406-406 -- `_SENTINEL_REGISTRY` -- the release/agent tag

*Left in the source:* what the registry is for and why it is module-level rather than class-level.

```text
# v4.15.1 (Agent E): name-keyed registry of every ``_Sentinel`` instance
```

### L421-428 -- `_sentinel_unpickle` -- the "pre-v4.15.2 fallback path" framing

*Left in the source:* the whole isinstance-downgrade hazard and the distributed-pipeline scenarios it appears in.

```text
    fresh base :class:`_Sentinel`.  The pre-v4.15.2 fallback path
    produced a *base* ``_Sentinel`` that compared ``False`` under
    ``isinstance`` checks against the original subclass (e.g.
    ``_ZeroApertureMaskSentinel``), silently downgrading caller
    semantics on receivers where the subclass-defining module had not
    yet been imported.  The audit (AUDIT_V4_15_1, P2) flagged this as
    a latent bug in distributed pipelines with delayed imports
    (joblib workers, dask distributed, multiprocessing Pool workers
```

### L453-459 -- `_NoDefaultSentinel` -- the release/agent tag and the "Pre-v4.15.2 _NO_DEFAULT was a bare _Sentinel" note

*Left in the source:* why the dedicated subclass exists and the registry key it uses.

```text
    v4.15.2 (Agent E, P3): dedicated subclass for consistency with
    :class:`_ZeroApertureMaskSentinel` and :class:`_AngleUnsetSentinel`.
    Pre-v4.15.2 ``_NO_DEFAULT`` was a bare ``_Sentinel('NO_DEFAULT')``
    instance, which differed cosmetically from the other two sentinels.
    No behaviour change: the new subclass overrides nothing and the
    singleton instance is still keyed by the ``'NO_DEFAULT'`` registry
    name.
```

