<!-- lumenairy-history-doc
module: lumenairy/sources/core.py
ast_sha256: 12f6ee71a7f7d4b0cc5187d9fb07849a83d6b1e9e7986f6482e9929e22469104
token_sha256: cd6e494df98d87072856c9908d76b2be728845febf6e35d35c6e30df5d7efeeb
pre_relocation_lines: 3381
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-B8 (audit Z3 / WP-A11 sec. 6.1-6.2): _schell_phase_realizations gains generator='fft'|'modes' with Gori pseudo-modes and an n_pseudo_modes heuristic, forwarded by create_gaussian_schell_source and create_schell_model_source; create_gaussian_beam gains the geometry_dtype= opt-in.  Both defaults are byte-identical
-->

# Version history -- `lumenairy/sources/core.py`

This file holds the version-history narrative that used to live in
`lumenairy/sources/core.py` -- the "vN.N (audit X): pre-fix this did A, which
was wrong because B, now it does C" passages.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

Almost all of it is one shape: a **deprecation chronology**.  This module
carried three removal waves (v4.11.2 kwarg forwarding, v4.14.2 / v4.15
positional-overload retirement, v5.25 kwarg renames) and the v5.30 "W5"
shim-removal wave that executed them, and every site recorded which release
deprecated a shape, which horizon it announced, how often that horizon slipped
and which release finally removed it.  None of that describes what the code
does.  What DOES describe the code -- that a legacy call shape is still
DETECTED so the `TypeError` can name the exact canonical form, rather than
degrading to Python's generic arity message -- stayed in the source at every
one of the four collector sites.

Three statements were not history but were **wrong**, and were corrected
rather than moved (they are recorded here with the correction noted):
`PartialCoherenceMCF`'s "deferred to v4.16+", `Source.gaussian_schell`'s
"MCF-aware downstream propagators are not in v4.15.x scope", and the promise
of "a future `Source.realizations()` per-realization iterator ... in scope for
v4.16+".  The library is at v5.46; none of the three shipped, and
`lumenairy/_validation.py` still refuses a `PartialCoherenceMCF` at every
propagator entry point.

Measured derivations of live constants stayed in the source, as
`docs/TESTING_STANDARDS.md` S5 requires: `_SCHELL_PAD_SIGMA`'s
`exp(-(2p)^2 / (2 sigma_g^2))` residual, the audit-Z2 anti-wrap argument in
`_schell_phase_realizations`, and the `pad_sigma=0.0` bit-for-bit
reproduction clause.

Nothing the interpreter executes changed in the move.  The header above
records the SHA-256 of (a) the module's AST with every docstring removed and
source positions ignored, and (b) its `tokenize` stream reduced to
NAME/OP/NUMBER/STRING with comments and docstrings dropped -- both taken from
the file as it stood BEFORE the relocation.
`tests/unit/test_audit2609_a17_history_relocation.py` re-computes both from
the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L49-74 | `<module>` -- shim-removal preamble | the v5.25 / v4.14.2 / v4.15 deprecation timeline and the v5.30 (W5) removal wave |
| L91-95 | `_coerce_source_rng` | the v5.25 -> v5.29 `seed=` deprecation and its v5.30 removal |
| L131-138 | `_resolve_gaussian_width` | the v5.25 `sigma` deprecation, its slipped v5.27 horizon and the v5.30 removal |
| L287-295 | `_validate_grid_params` | what the pre-v4.15 check accepted before `bool` was rejected explicitly |
| L471-487 | `create_gaussian_beam` | the pre-4.7 positional ordering, the v5.25 sigma -> w0 migration and the full v5.30 removal note |
| L1249-1251 | `create_top_hat_beam` | the pre-4.7 positional ordering |
| L1402-1406 | `create_fiber_mode` -- `dy` | that pre-v4.13.2 a user-supplied `dy=` was silently squared through `factory_kwargs` |
| L1456-1462 | `_reject_led_legacy_positional` | the v4.14.2 deprecation, the `version_removed='5.0'` banner that shipped to v5.29 and the v5.30 execution |
| L1530-1532 | `create_led_source` -- `wavelength` | the `since v4.14.2` slot-move tag |
| L1539-1541 | `create_led_source` -- `dy` | when the `dy` kwarg was added and which release family it caught up with |
| L1555-1558 | `create_led_source` -- `source_angles` | a comment correcting an earlier COMMENT: the docstring used to say '~21 samples' |
| L1564-1571 | `create_led_source` -- Notes | the pre-v4.14.2 positional ordering and which convention it broke |
| L1573-1590 | `create_led_source` -- `.. versionchanged:: 5.30` | the 29-release deprecation slip and the old-form migration example |
| L1592 | `create_led_source` body | the `v5.30 (W5)` removal tag on the live guard |
| L1821-1826 | `PartialCoherenceMCF` -- Notes | the 'deferred to v4.16+' scope promise (CORRECTED: still unimplemented at v5.46) |
| L2059-2092 | `<module>` -- `return_kind` sentinel preamble | the whole v4.15.2 -> v4.15.5 -> v4.16.1 -> v5.30 life of the `_RETURN_KIND_UNSET` sentinel |
| L2144-2149 | `_schell_phase_realizations` | the vestigial `N` keyword removed in v5.24.x and why it was dead |
| L2324-2325 | `create_gaussian_schell_source` | the `v4.15.1 (P0-NEW-2): redesigned` framing |
| L2374-2378 | `create_gaussian_schell_source` -- `sigma_g` | what the pre-v5.46 periodised kernel did (0.27-of-peak error, 99 %-coherent opposite edges) |
| L2398-2409 | `create_gaussian_schell_source` -- `return_kind` | the retired default-path DeprecationWarning and the v5.30 sentinel removal |
| L2456-2461 | `create_gaussian_schell_source` body | the v4.16.1 warning retirement and the v5.30 sentinel-branch removal |
| L2516-2517 | `create_schell_model_source` | the `v4.15.1 (P0-NEW-2): redesigned` framing |
| L2537-2539 | `create_schell_model_source` -- `coherence_length` | that pre-v5.46 the realised kernel was the grid-PERIODISED Gaussian |
| L2550-2552 | `create_schell_model_source` -- `return_kind` | the retired default-path DeprecationWarning |
| L2584-2586 | `create_schell_model_source` body | the v4.16.1 warning retirement and the v5.30 sentinel-branch removal |
| L2629-2630 | `create_annular_incoherent_source` -- summary | the roadmap/redesign tags in the one-line summary |
| L2650-2653 | `create_annular_incoherent_source` | the `no longer collapses the ensemble` framing |
| L2677-2679 | `create_annular_incoherent_source` -- `return_kind` | the retired default-path DeprecationWarning |
| L2711-2713 | `create_annular_incoherent_source` body | the v4.16.1 warning retirement and the v5.30 sentinel-branch removal |
| L2911-2917 | `Source` -- factory-kwargs preamble | what pre-4.11.2 did with an unforwarded `dy=` / `dtype=` |
| L2919-2953 | `Source` -- size-arg normalisation preamble | the five pre-v4.15 positional orders and the v4.15 -> v5.30 removal chronology |
| L3224-3227 | `Source` -- partial-coherence section banner | the roadmap item numbers the two factories landed under |
| L3244-3247 | `Source.gaussian_schell` -- `.. versionchanged:: 5.30` | the v5.25 deprecation and its stated v5.27 horizon |
| L3249-3264 | `Source.gaussian_schell` | the pre-v4.15.2 Source-wrapped 3-D ensemble, and the v4.16.1 / v5.30 warning-and-sentinel retirement |
| L3287-3288 | `Source.gaussian_schell` -- inconsistency note | the 'not in v4.15.x scope' horizon (CORRECTED: still unimplemented at v5.46) |
| L3306-3307 | `Source.gaussian_schell` -- invariant-break note | the 'a future Source.realizations() is in scope for v4.16+' promise (CORRECTED: never shipped; v5.46 has no such API) |
| L3309-3311 | `Source.gaussian_schell` body | the v4.16.1 warning retirement and the v5.30 sentinel removal |
| L3352-3357 | `Source.schell_model` | the v4.16.1 warning retirement and the v5.30 sentinel removal |
| L3363-3365 | `Source.schell_model` -- invariant-break note | the pointer to the 'v4.16+ Source.realizations() plan' (CORRECTED: never shipped) |
| L3367-3369 | `Source.schell_model` body | the v4.16.1 warning retirement and the v5.30 sentinel removal |

---

### L49-74 -- `<module>` -- shim-removal preamble -- the v5.25 / v4.14.2 / v4.15 deprecation timeline and the v5.30 (W5) removal wave

*Left in the source:* which call shapes are rejected today, that the rejection is permanent, and WHY the legacy shape is still detected rather than left to Python's arity error

```text
# v5.30 (W5 shim-removal wave -- honest break, precedent:
# ``propagators/system.py`` ``_reject_legacy`` / ``analysis/detector.py``
# ``cosmic_ray_rate``).  Two families of shim are REMOVED from this module:
#
# 1. The v5.25 kwarg renames (audit S3-16 / B1), stated horizon v5.27,
#    re-scheduled to v5.32 in v5.30 and then executed here instead of
#    slipping a third time:
#      * Schell-family ``seed=<int>``     ->  ``rng=<int>``   (exactly
#        equivalent: ``seed`` was forwarded verbatim to ``rng``).
#      * ``create_gaussian_beam(sigma=s)`` ->  ``w0=s*sqrt(2)``  (``w0`` is
#        the 1/e^2 intensity radius, ``sigma`` was the field std-dev).
#    Both are plain signature removals, so the old form now raises
#    ``TypeError: ... unexpected keyword argument`` -- the same shape as the
#    v5.0 ``cosmic_ray_rate`` retirement.
#
# 2. The v4.14.2 / v4.15 legacy POSITIONAL overloads (``create_led_source``,
#    ``Source.gaussian`` / ``plane_wave`` / ``point_source`` / ``top_hat`` /
#    ``fiber_mode``) and the v4.15.1 Schell ``return_kind`` sentinel, all of
#    which advertised ``version_removed='5.0'`` while shipping through v5.29
#    (R-18 re-scheduled the banner to v5.32; v5.30 executes it).  These
#    intercepted VALUES, so -- following ``system.py``'s ``_reject_legacy``
#    precedent -- the legacy shape is still DETECTED and rejected with an
#    actionable ``TypeError`` naming the canonical form.  That rejection is
#    permanent: it schedules nothing and adds no new deprecation debt.
#
# See the CHANGELOG ``### Removed`` section for the full migration table.
```

### L91-95 -- `_coerce_source_rng` -- the v5.25 -> v5.29 `seed=` deprecation and its v5.30 removal

*Left in the source:* the one-line migration (``rng=<int>`` reproduces ``seed=<int>`` bit-for-bit), which a caller porting old code still needs

```text
    v5.30: the deprecated ``seed=`` spelling (v5.25 -> v5.29) is REMOVED.
    ``rng=<int>`` reproduces the old ``seed=<int>`` stream bit-for-bit --
    ``seed`` was forwarded verbatim into ``rng`` -- so the migration is a
    pure rename.  Passing ``seed=`` now raises ``TypeError`` from the
    factory signature.
```

### L131-138 -- `_resolve_gaussian_width` -- the v5.25 `sigma` deprecation, its slipped v5.27 horizon and the v5.30 removal

*Left in the source:* the migration factor.  The live ``TypeError`` below states it too, so the docstring keeps only the conversion.

```text
    v5.30: the legacy ``sigma`` kwarg (the field *standard deviation* -- a
    misnomer relative to the library-wide waist convention) is REMOVED.
    It was deprecated in v5.25 with a stated v5.27 horizon that shipped
    unremoved through v5.29.  Migration: ``sigma=s`` -> ``w0=s*sqrt(2)``
    (equivalently ``w0=w`` reproduces ``sigma=w/sqrt(2)`` bit-for-bit --
    that direction is exact because it is the division this function
    performs).  Passing ``sigma=`` now raises ``TypeError`` from the
    :func:`create_gaussian_beam` signature.
```

### L287-295 -- `_validate_grid_params` -- what the pre-v4.15 check accepted before `bool` was rejected explicitly

*Left in the source:* the whole hazard argument -- it is the reason the explicit ``bool`` branch exists and must stay next to it.  Only the 'the pre-v4.15 check accepted' framing moved.

```text
    # v4.15 (P2-VAL-1 / v4.14.2 carryover): explicitly reject ``bool``.
    # ``isinstance(True, (int, np.integer))`` returns True so the
    # pre-v4.15 check accepted ``N=True`` / ``N=False`` as 1 / 0.
    # ``N=False`` then hit the ``int(N) <= 0`` guard with a confusing
    # "N=0" error; ``N=True`` (a Boolean grid size, plainly wrong)
    # silently produced a 1x1 grid.  Boolean ``N`` is almost certainly
    # a caller bug (passing ``N=large_grid_flag and 1024`` -> 1024 only
    # when flag is truthy; ``N=use_gpu and 256`` -> ``False`` when GPU
    # is off; etc.), so the loudest correct action is a TypeError.
```

### L471-487 -- `create_gaussian_beam` -- the pre-4.7 positional ordering, the v5.25 sigma -> w0 migration and the full v5.30 removal note

*Left in the source:* the live signature and the slot convention, plus a condensed ``.. versionchanged:: 5.30`` carrying the migration formula

```text
    Signature is ``(N, dx, wavelength, *, w0, ...)`` since 4.7 (the width
    argument is keyword-only).  Prior to 4.7 the ordering was
    ``(N, dx, sigma, wavelength=None, ...)`` with positional ``sigma``;
    the new style places ``wavelength`` at the third positional slot
    (matching every other source factory).

    v5.25 (audit S3-16): the width argument migrated from ``sigma`` (the
    field standard deviation -- a misnomer relative to the library-wide
    waist convention) to the canonical ``w0`` (the 1/e^2 intensity
    radius).

    .. versionchanged:: 5.30
        The deprecated ``sigma`` kwarg is **removed** (deprecated v5.25,
        stated horizon v5.27, shipped unremoved through v5.29).  Migrate
        ``sigma=s`` -> ``w0=s*sqrt(2)``; equivalently ``w0=w`` reproduces
        the old ``sigma=w/sqrt(2)`` field bit-for-bit.  Passing ``sigma=``
        raises ``TypeError``.
```

### L1249-1251 -- `create_top_hat_beam` -- the pre-4.7 positional ordering

*Left in the source:* the live signature and the slot convention

```text
    Signature is ``(N, dx, wavelength, *, diameter, ...)`` since 4.7.
    Prior to 4.7 the ordering was
    ``(N, dx, diameter, wavelength=None, ...)``.
```

### L1402-1406 -- `create_fiber_mode` -- `dy` -- that pre-v4.13.2 a user-supplied `dy=` was silently squared through `factory_kwargs`

*Left in the source:* what ``dy`` does now and who threads it

```text
        Grid spacing in y [m].  Defaults to ``dx``.  v4.13.2 added the
        keyword so :meth:`Source.fiber_mode` can pass an anamorphic
        ``dy`` through to the underlying Gaussian without erroring
        (the historical signature only accepted ``dx``, so a user-
        supplied ``dy=`` was silently squared via ``factory_kwargs``).
```

### L1456-1462 -- `_reject_led_legacy_positional` -- the v4.14.2 deprecation, the `version_removed='5.0'` banner that shipped to v5.29 and the v5.30 execution

*Left in the source:* the rejected shape itself, and the whole 'why keep an always-raising collector' rationale below it

```text
    v5.30 (W5 shim-removal wave): the positional form
    ``(N, dx, diameter, divergence_angle, wavelength, x0, y0, dtype)`` is
    REMOVED.  It was deprecated in v4.14.2 with ``version_removed='5.0'``
    and kept shipping through v5.29 (R-18 re-scheduled the banner to
    v5.32; the owner executed the removal at v5.30 rather than slipping a
    third time).

```

### L1530-1532 -- `create_led_source` -- `wavelength` -- the `since v4.14.2` slot-move tag

*Left in the source:* the slot convention itself

```text
        Vacuum wavelength.  Now in the canonical 3rd positional slot
        (since v4.14.2) -- matches every other ``create_*`` factory in
        ``sources/core.py``.
```

### L1539-1541 -- `create_led_source` -- `dy` -- when the `dy` kwarg was added and which release family it caught up with

*Left in the source:* that anamorphic grids thread a distinct y-pitch through this factory

```text
        v4.14.2 added the kwarg so anamorphic grids can thread a
        distinct y-pitch through this factory like the rest of the
        v4.13.0+ Source family.
```

### L1555-1558 -- `create_led_source` -- `source_angles` -- a comment correcting an earlier COMMENT: the docstring used to say '~21 samples'

*Left in the source:* the live count (37) and the instruction to size output arrays to it

```text
        + 6 + 12 + 18 = 37 for ``n_ring = 3``).  4.10: docstring
        previously said "~21 samples" which underspecified the actual
        count; downstream callers allocating output arrays should size
        them at 37, not 21.
```

### L1564-1571 -- `create_led_source` -- Notes -- the pre-v4.14.2 positional ordering and which convention it broke

*Left in the source:* the live signature and the convention it follows

```text
    Signature is ``(N, dx, wavelength, *, diameter, divergence_angle,
    dy=None, x0=0, y0=0, dtype=None)`` since v4.14.2.  Pre-v4.14.2 the
    ordering was ``(N, dx, diameter, divergence_angle, wavelength,
    x0=0, y0=0, dtype=None)`` -- ``diameter`` and ``divergence_angle``
    were positional and the function neither accepted ``dy=`` nor a
    ``*`` keyword-only separator.  This broke the post-v4.7 convention
    of keyword-only physical parameters with ``wavelength`` in the
    canonical 3rd positional slot.
```

### L1573-1590 -- `create_led_source` -- `.. versionchanged:: 5.30` -- the 29-release deprecation slip and the old-form migration example

*Left in the source:* the directive itself and the canonical call, condensed -- the live ``TypeError`` already names the remapping

```text
    .. versionchanged:: 5.30
        The legacy positional form is **removed** (deprecated v4.14.2,
        stated horizon v5.0, shipped unremoved through v5.29 -- 29 minor
        releases past its own date; R-18 re-scheduled the banner to v5.32
        and v5.30 executes it).  Any positional surplus past ``wavelength``
        now raises ``TypeError`` naming the canonical form, following the
        ``propagators/system.py`` ``_reject_legacy`` precedent: the legacy
        SHAPE is still detected so the diagnostic stays actionable instead
        of degrading to Python's generic arity message.  Migration::

            # Old (removed in v5.30 -- raises TypeError)
            E, angles, x, y = create_led_source(
                64, 16e-6, 100e-6, 0.3, 1.31e-6)

            # New (canonical)
            E, angles, x, y = create_led_source(
                64, 16e-6, 1.31e-6,
                diameter=100e-6, divergence_angle=0.3)
```

### L1592 -- `create_led_source` body -- the `v5.30 (W5)` removal tag on the live guard

*Left in the source:* the pointer to the helper that carries the rationale

```text
    # v5.30 (W5): legacy positional form REMOVED -- see the helper.
```

### L1821-1826 -- `PartialCoherenceMCF` -- Notes -- the 'deferred to v4.16+' scope promise (CORRECTED: still unimplemented at v5.46)

*Left in the source:* the live limitation, restated without the stale horizon and pointed at the guard that enforces it

```text
    MCF-aware downstream propagators (Koehler-/Hopkins-style coherent-
    mode propagation, MCF transport through the system) are NOT in
    v4.15.1 scope and are deferred to v4.16+.  The current
    :class:`PartialCoherenceMCF` is consumable for inspection and
    analysis only (intensity, two-point coherence, coherent-mode
    extraction).
```

### L2059-2092 -- `<module>` -- `return_kind` sentinel preamble -- the whole v4.15.2 -> v4.15.5 -> v4.16.1 -> v5.30 life of the `_RETURN_KIND_UNSET` sentinel

*Left in the source:* the live consequence -- no sentinel, no warning, and why no bespoke rejection branch is needed here (it is what distinguishes this removal from the positional ones)

```text
# v4.15.2 (P0-NEW-1) - v4.15.5: the 3 Schell factories changed default
# return shape from ``(E_2d, x, y)`` (v4.15.0) to
# ``(ensemble_3d, dx, dy, wavelength)`` (v4.15.1).  A one-release
# ``DeprecationWarning`` was emitted on the default path so pre-v4.15.0
# callers doing ``E, x, y = create_gaussian_schell_source(...)`` would
# see a loud heads-up rather than a propagation-time wrong-shape
# failure.  Detection used a per-module sentinel
# (``_RETURN_KIND_UNSET``, a ``_SchellReturnKindUnsetSentinel``): when
# the caller left ``return_kind`` unset, the factory saw the sentinel and
# warned; explicit ``return_kind='ensemble'`` / ``'mcf'`` was silent.
#
# v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6) retired the warning itself --
# the transition had had five releases of exposure -- leaving the kwarg
# default at plain ``'ensemble'`` and the sentinel branch at each of the
# five call sites as a pure no-op.
#
# v5.30 (W5 shim-removal wave): the whole sentinel apparatus is REMOVED --
# ``_SchellReturnKindUnsetSentinel``, the ``_RETURN_KIND_UNSET`` singleton,
# the ``_warn_schell_return_kind_default`` helper, and the five no-op
# ``if return_kind is _RETURN_KIND_UNSET`` branches.  It advertised
# ``version_removed='5.0'`` while shipping through v5.29 (R-18 re-scheduled
# the banner to v5.32; the owner executed it at v5.30).  The helper had
# ZERO production call sites (pinned in
# ``tests/unit/test_niche_audit_w3_ui_deprecation.py``), so nothing on the
# modern path changes.
#
# Old form -> new form: ``return_kind=_RETURN_KIND_UNSET`` (or omitted)
# -> omit it, or pass ``return_kind='ensemble'`` explicitly.  A caller who
# still holds a reference to the old sentinel object now gets the existing
# ``ValueError`` from :func:`_validate_return_kind` ("return_kind must be
# 'ensemble' or 'mcf'"), which already names the modern values -- so no
# bespoke rejection branch is needed here (contrast the positional
# overloads above, where the legacy shape carried no such self-describing
# validator).
```

### L2144-2149 -- `_schell_phase_realizations` -- the vestigial `N` keyword removed in v5.24.x and why it was dead

*Left in the source:* the live statement of what specifies the grid

```text
    v5.24.x (audit S3-16): the vestigial ``N`` keyword was removed.  The
    grid is fully specified by ``Ny`` / ``Nx``; the old ``N`` argument
    duplicated ``Nx`` (callers passed ``N=Ny=Nx`` for the square-grid
    case) and was never read in the body -- a dead parameter that only
    invited an inconsistent (``N`` != ``Nx``) call.

```

### L2324-2325 -- `create_gaussian_schell_source` -- the `v4.15.1 (P0-NEW-2): redesigned` framing

*Left in the source:* the return contract, which is the whole of the live statement

```text
    v4.15.1 (P0-NEW-2): redesigned to deliver actual partial coherence.
    Returns either the raw ``(n_realizations, Ny, Nx)`` complex
```

### L2374-2378 -- `create_gaussian_schell_source` -- `sigma_g` -- what the pre-v5.46 periodised kernel did (0.27-of-peak error, 99 %-coherent opposite edges)

*Left in the source:* nothing here -- the live statement ('the realised kernel is now the documented Gaussian to ~1e-14 at any sigma_g') is already made in the Grid-constraint paragraph above

```text
        ``w0``.  v5.46 (audit Z2): the realised kernel is the
        documented Gaussian at any ``sigma_g``; pre-v5.46 it was the
        grid-PERIODISED Gaussian, which at ``sigma_g = L/3`` differed
        by 0.27 of peak and made opposite edges of the grid 99 %
        coherent.
```

### L2398-2409 -- `create_gaussian_schell_source` -- `return_kind` -- the retired default-path DeprecationWarning and the v5.30 sentinel removal

*Left in the source:* the live default and the fact that the call is silent

```text
        v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        ``DeprecationWarning`` (v4.15.2 -> v4.15.5) flagging the
        v4.15.0 -> v4.15.1 return-shape change is retired now that
        the new ensemble contract has had multiple releases of
        exposure.  The default is ``'ensemble'`` and the call is
        silent.

        .. versionchanged:: 5.30
            The ``_RETURN_KIND_UNSET`` sentinel (and the
            ``_warn_schell_return_kind_default`` helper) are **removed**
            -- both had been no-ops since v4.16.1.  Omit ``return_kind``
            or pass ``'ensemble'`` / ``'mcf'`` explicitly.
```

### L2456-2461 -- `create_gaussian_schell_source` body -- the v4.16.1 warning retirement and the v5.30 sentinel-branch removal

*Left in the source:* what the live call does

```text
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
    # ``DeprecationWarning`` (v4.15.2 -> v4.15.5) was retired once the
    # v4.15.0 return-shape change had multiple releases of exposure.
    # v5.30 (W5): the no-op ``_RETURN_KIND_UNSET`` sentinel branch is
    # removed with the rest of the shim; the kwarg default is plain
    # ``'ensemble'`` and validation happens in one place.
```

### L2516-2517 -- `create_schell_model_source` -- the `v4.15.1 (P0-NEW-2): redesigned` framing

*Left in the source:* the MCF factorisation, which is the live statement

```text
    v4.15.1 (P0-NEW-2): redesigned to deliver actual partial coherence.
    The MCF factorises as
```

### L2537-2539 -- `create_schell_model_source` -- `coherence_length` -- that pre-v5.46 the realised kernel was the grid-PERIODISED Gaussian

*Left in the source:* the live statement that the kernel is the documented Gaussian at any coherence length, now pointed at the generator that makes it true

```text
        v5.46 (audit Z2): the realised kernel is the documented Gaussian
        at any coherence length; pre-v5.46 it was the grid-PERIODISED
        Gaussian.
```

### L2550-2552 -- `create_schell_model_source` -- `return_kind` -- the retired default-path DeprecationWarning

*Left in the source:* the live default and the cross-reference

```text
        See :func:`create_gaussian_schell_source`.  v4.16.1 (audit
        item 6): the default-path ``DeprecationWarning`` is retired;
        the default is plain ``'ensemble'`` and the call is silent.
```

### L2584-2586 -- `create_schell_model_source` body -- the v4.16.1 warning retirement and the v5.30 sentinel-branch removal

*Left in the source:* what the live call does

```text
    # v4.16.1: default-path warning retired.  v5.30 (W5): the no-op
    # sentinel branch is removed (see
    # :func:`create_gaussian_schell_source`).
```

### L2629-2630 -- `create_annular_incoherent_source` -- summary -- the roadmap/redesign tags in the one-line summary

*Left in the source:* what the source IS

```text
    """Annular (ring) source -- spatially-incoherent at the source
    plane (v4.15, ROADMAP v4.16 #11; v4.15.1 P0-NEW-2 redesign).
```

### L2650-2653 -- `create_annular_incoherent_source` -- the `no longer collapses the ensemble` framing

*Left in the source:* the live return contract, stated positively

```text
    v4.15.1 (P0-NEW-2): the factory no longer collapses the ensemble
    into a single complex field; it returns either the raw ensemble or
    a :class:`PartialCoherenceMCF` (diagonal MCF reflecting the
    incoherent-source character).
```

### L2677-2679 -- `create_annular_incoherent_source` -- `return_kind` -- the retired default-path DeprecationWarning

*Left in the source:* the live default and the cross-reference

```text
        See :func:`create_gaussian_schell_source`.  v4.16.1 (audit
        item 6): the default-path ``DeprecationWarning`` is retired;
        the default is plain ``'ensemble'`` and the call is silent.
```

### L2711-2713 -- `create_annular_incoherent_source` body -- the v4.16.1 warning retirement and the v5.30 sentinel-branch removal

*Left in the source:* what the live call does

```text
    # v4.16.1: default-path warning retired.  v5.30 (W5): the no-op
    # sentinel branch is removed (see
    # :func:`create_gaussian_schell_source`).
```

### L2911-2917 -- `Source` -- factory-kwargs preamble -- what pre-4.11.2 did with an unforwarded `dy=` / `dtype=`

*Left in the source:* what the forwarding does now

```text
    # 4.11.2 (audit round-3): the classmethod factories below pass
    # ``**factory_kwargs`` through to the underlying ``create_*`` calls
    # so callers can configure ``dy=``, ``dtype=``, ``normalize=``,
    # ``use_gpu=``, etc. without having to call the bare function
    # directly.  Pre-4.11.2 these kwargs were not propagated, so
    # anamorphic grids and single-precision fields silently fell back
    # to the create_*'s defaults.
```

### L2919-2953 -- `Source` -- size-arg normalisation preamble -- the five pre-v4.15 positional orders and the v4.15 -> v5.30 removal chronology

*Left in the source:* the canonical order, and why three of the five keep an always-raising collector

```text
    # -----------------------------------------------------------------
    # v4.15 (ROADMAP v4.15 #2): size-arg normalisation on the 5
    # Source.* factory classmethods.
    #
    # Pre-v4.15 the 5 factories had inconsistent positional order:
    #   - ``Source.gaussian(w0, N, dx, wavelength)``  -- size first
    #   - ``Source.plane_wave(N, dx, wavelength)``    -- N first
    #   - ``Source.point_source(N, dx, wavelength)``  -- N first
    #   - ``Source.top_hat(diameter, N, dx, wavelength)`` -- size first
    #   - ``Source.fiber_mode(mfd, N, dx, wavelength)``  -- size first
    #
    # v4.15 picks the canonical order
    # ``Source.method(*, N, dx, wavelength, <size_kwargs>)`` (kwarg-only
    # with the ``*`` separator).
    #
    # v5.30 (W5 shim-removal wave): the legacy positional form is REMOVED
    # from all five.  It was deprecated in v4.15 with
    # ``version_removed='5.0'`` and kept shipping through v5.29 (R-18
    # re-scheduled the banner to v5.32; the owner executed the removal at
    # v5.30).  Each classmethod keeps an always-raising
    # ``*_legacy_positional`` collector so the legacy SHAPE is still
    # detected and the ``TypeError`` can name the exact canonical
    # signature -- the ``propagators/system.py`` ``_reject_legacy``
    # precedent.  This matters most for ``gaussian`` / ``top_hat`` /
    # ``fiber_mode``, where the legacy order put the SIZE argument first,
    # so a positional caller has every quantity one slot out; a bare
    # arity error would not say that.  The rejection is permanent and
    # schedules nothing.
    #
    # The three already-kwarg-only factories (``plane_wave``,
    # ``point_source``) keep their existing signature; the only change
    # for them is that they now appear under the canonical
    # ``Source.method(*, N, dx, wavelength, ...)`` umbrella in the
    # docs and the factory-validation parametrize list.
    # -----------------------------------------------------------------
```

### L3224-3227 -- `Source` -- partial-coherence section banner -- the roadmap item numbers the two factories landed under

*Left in the source:* what the section holds

```text
    # -----------------------------------------------------------------
    # v4.15 (ROADMAP v4.16 #9, #11): two new partial-coherence factories
    # for the Schell-model family and the annular-incoherent source.
    # -----------------------------------------------------------------
```

### L3244-3247 -- `Source.gaussian_schell` -- `.. versionchanged:: 5.30` -- the v5.25 deprecation and its stated v5.27 horizon

*Left in the source:* the directive and the exact-equivalence migration

```text
        .. versionchanged:: 5.30
            The legacy ``seed=`` kwarg is **removed** (deprecated v5.25,
            stated horizon v5.27).  ``rng=<int>`` reproduces the old
            ``seed=<int>`` stream bit-for-bit.
```

### L3249-3264 -- `Source.gaussian_schell` -- the pre-v4.15.2 Source-wrapped 3-D ensemble, and the v4.16.1 / v5.30 warning-and-sentinel retirement

*Left in the source:* the live return type and the fact that the default path is silent.  The full contract follows in its own section.

```text
        v4.15.2 (Agent E, AUDIT_V4_15_1 P2): the return type now
        matches the top-level factory's return-type convention
        verbatim -- ``return_kind='ensemble'`` yields the raw
        ``(ensemble, dx, dy, wavelength)`` 4-tuple (NOT a
        :class:`Source`-wrapped 3-D ensemble).  Pre-v4.15.2 this
        classmethod wrapped the 3-D ensemble inside a :class:`Source`
        whose ``E`` was 3-D, breaking the :class:`Source` contract
        (every other ``Source.*`` classmethod produces a 2-D field)
        and surprising downstream ``src.intensity()`` callers with
        broadcasting axis mismatches.

        v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        ``DeprecationWarning`` (v4.15.2 -> v4.15.5) for the v4.15.0
        -> v4.15.1 return-shape change is retired.  The default is
        plain ``'ensemble'`` and the call is silent.  v5.30 (W5): the
        ``_RETURN_KIND_UNSET`` sentinel + helper are removed.
```

### L3287-3288 -- `Source.gaussian_schell` -- inconsistency note -- the 'not in v4.15.x scope' horizon (CORRECTED: still unimplemented at v5.46)

*Left in the source:* the live limitation

```text
        inspection / analysis only; MCF-aware downstream propagators
        are not in v4.15.x scope.
```

### L3306-3307 -- `Source.gaussian_schell` -- invariant-break note -- the 'a future Source.realizations() is in scope for v4.16+' promise (CORRECTED: never shipped; v5.46 has no such API)

*Left in the source:* the live statement -- there is no iterator, unpack explicitly

```text
        A future ``Source.realizations()`` per-realization iterator
        is in scope for v4.16+ but is NOT shipped in v4.15.3.
```

### L3309-3311 -- `Source.gaussian_schell` body -- the v4.16.1 warning retirement and the v5.30 sentinel removal

*Left in the source:* what the live call does

```text
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        # DeprecationWarning is retired.  v5.30 (W5): the no-op sentinel
        # pass-through is removed with the rest of the shim.
```

### L3352-3357 -- `Source.schell_model` -- the v4.16.1 warning retirement and the v5.30 sentinel removal

*Left in the source:* the live default and the silence

```text
        v4.16.1 (audit AUDIT_V4_16_0_DEEP item 6): the default-path
        ``DeprecationWarning`` is retired in line with
        :meth:`Source.gaussian_schell`.  The default is plain
        ``'ensemble'`` and the call is silent.  v5.30 (W5): the sentinel
        is removed (see :meth:`Source.gaussian_schell`).

```

### L3363-3365 -- `Source.schell_model` -- invariant-break note -- the pointer to the 'v4.16+ Source.realizations() plan' (CORRECTED: never shipped)

*Left in the source:* the cross-reference to the full rationale

```text
        single-source.  See :meth:`Source.gaussian_schell` docstring
        for the full rationale and the v4.16+ ``Source.realizations()``
        per-realization-iterator plan.
```

### L3367-3369 -- `Source.schell_model` body -- the v4.16.1 warning retirement and the v5.30 sentinel removal

*Left in the source:* what the live call does

```text
        # v4.16.1: default-path DeprecationWarning retired.  v5.30 (W5):
        # the no-op sentinel pass-through is removed (see
        # Source.gaussian_schell).
```
