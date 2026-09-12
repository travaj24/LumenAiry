<!-- lumenairy-history-doc
module: lumenairy/glass.py
ast_sha256: 630f81ef4f97e45b0b20250def034268308147a0a82f34cc5a976c98d37c77a7
token_sha256: c01964bc4d03b8a91a12d0c0f469679eff1ab97998bd5c88080932c3c4cc3acf
pre_relocation_lines: 2160
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/glass.py`

This file holds the version-history narrative that used to live in
`lumenairy/glass.py` -- the "vX.Y (audit Z): pre-fix this did A, which was wrong
because B, now it does C" blocks, the release-tag prefixes on section headers,
and the notes a later revision wrote to correct an earlier *comment*.  Each
block is reproduced **verbatim** under the source line it came from in the
pre-relocation file, so `git log -S` on any phrase here still lands on the
commit that wrote it.

`glass.py` is a catalogue module, and almost all of its history is of one
shape: **a bundled dispersion row, or a consistency check, annotated with the
release that added it and with what the library did before it existed.**  The
catalogue *provenance* (which vendor YAML a row was copied from, and the
measured n_d / V_d residual against it) is NOT history -- it is the derivation
`docs/TESTING_STANDARDS.md` S5 requires a numeric row to carry, and it stayed
in the source.  What moved is the release framing around it.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment; those are noted per block below as *Left in the
source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L79-83 | `<module> imports` | why `OrderedDict` is imported un-aliased |
| L87-88 | `<module> imports` | which release moved numpy to module scope |
| L195-217 | `SELLMEIER_COEFFICIENTS -- Ohara S-LAH` | the mis-attributed in-code coefficients (n_d off by 0.058 / 0.118) and the two round trips that followed |
| L240-244 | `SELLMEIER_COEFFICIENTS -- fused silica` | what a minimal install did before the silica rows were bundled |
| L266-266 | `SELLMEIER_COEFFICIENTS -- CDGM header` | the release tag on the section header |
| L342-342 | `POLYNOMIAL_COEFFICIENTS header` | the release tag on the section header |
| L352-359 | `POLYNOMIAL_COEFFICIENTS header` | the three-release rollout of the formula-3 path (entries in one release, evaluator in the next, coefficients in a third) |
| L373-373 | `POLYNOMIAL_COEFFICIENTS header` | which releases the 5e-5 cross-check methodology came from |
| L376-377 | `POLYNOMIAL_COEFFICIENTS` | the release tag on the ingestion note |
| L515-540 | `_POLYNOMIAL_STUB_NAMES` | the generic-ImportError path that predated the stub manifest, and the release that emptied the manifest |
| L545-549 | `_guard_wavelength` | the release that consolidated three copies of the wavelength guard |
| L602-609 | `_polynomial_index` | which release added array support and which contract it was matching |
| L656-672 | `_sellmeier_index` | the opaque `math domain error` the resonance guard replaced, and the two exceptions an array input used to raise |
| L1004-1009 | `_validity_warned` | that the dispatcher emitted no signal at all before the validity warning existed |
| L1049-1055 | `_GLASS_VALIDITY_REGISTRY_EXEMPTIONS` | a note correcting this list's own earlier description |
| L1335-1338 | `_check_glass_registry_consistency` | which release added each of the six structural checks |
| L1350-1388 | `_check_glass_registry_consistency` | the per-check release tags and the pre-v4.15 silent-dead-code narrative |
| L1400-1401 | `_check_glass_registry_consistency` | the release tag on the polynomial forward check |
| L1411-1415 | `_check_glass_registry_consistency` | the release tag on the reverse check |
| L1428-1428 | `_check_glass_registry_consistency` | the release tag on the polynomial reverse check |
| L1446-1447 | `_check_glass_registry_consistency` | the release tag on the stub-manifest well-formedness check |
| L1474-1474 | `_check_glass_registry_consistency` | the release tag on the GLASS_VALIDITY -> GLASS_REGISTRY check |
| L1492-1495 | `_check_glass_registry_consistency` | the release tag on the tuple well-formedness check |
| L1505-1509 | `_check_glass_registry_consistency` | the release that started accepting numpy scalars and the release whose check users were tripping |
| L1582-1582 | `_GLASS_CACHE_LOCK` | the release tag |
| L1598-1615 | `_glass_value_cache` | a correction of this comment's own earlier "femtometre" wording, and the unbounded plain dict the LRU replaced |
| L1618-1618 | `_GLASS_VALUE_CACHE_SIZE` | the release tag on the bound rationale |
| L1635-1635 | `_cached_glass_value` | the release tag |
| L1657-1658 | `_invalidate_glass_name` | the release tag |
| L1695-1698 | `_require_finite_catalogue_index` | the pre-fix wording ("used to hand back nan") |
| L1824-1824 | `get_glass_index` | the release tag |
| L1832-1838 | `get_glass_index` | the inverted "POLYNOMIAL -> SELLMEIER" wording that drifted into a release note, and the audit that reconciled it |
| L1850-1856 | `get_glass_index` | that the bundled polynomial evaluator was once reachable only on the refractiveindex-unavailable fallback |
| L1906-1911 | `get_glass_index` | a comment that CONTRADICTED the code: it said POLYNOMIAL_COEFFICIENTS was empty and that ingestion was still staged |
| L1918-1919 | `get_glass_index` | the release tag |
| L2012-2013 | `get_glass_index_complex` | which release added __polynomial__ alongside __sellmeier__ |
| L2105-2109 | `<module> cache-registry enrollment` | that clear_asm_caches never drained these four caches before the enrollment landed, and the case-sensitive filter that hid them |

---

### L79-83 -- `<module> imports` -- why `OrderedDict` is imported un-aliased

*Left in the source:* the whole rule -- the walker matches the literal RHS shape, so an alias hides the cache.  Only the release tag and the "audit P2-42 just closed this for lower-case names" aside moved.

```text
# NOTE: imported UN-aliased on purpose -- the v4.16.1 cache-enrollment
# meta-pin's AST walker recognises a cache declaration by the literal
# ``OrderedDict(...)`` / ``dict(...)`` / ``{}`` RHS shape; an aliased
# ``_OrderedDict()`` would hide ``_glass_value_cache`` from discovery
# (the exact blindness audit P2-42 just closed for lower-case names).
```

### L87-88 -- `<module> imports` -- which release moved numpy to module scope

*Left in the source:* the live reason: the import-time GLASS_VALIDITY check accepts numpy scalars, so `np` must be bound before it runs.

```text
# v4.16.2 (audit P3-NEW-F1-4): import numpy at module scope to support
# numpy-scalar acceptance in GLASS_VALIDITY well-formedness check.
```

### L195-217 -- `SELLMEIER_COEFFICIENTS -- Ohara S-LAH` -- the mis-attributed in-code coefficients (n_d off by 0.058 / 0.118) and the two round trips that followed

*Left in the source:* the provenance and the measured residuals of the rows the table holds today, plus a one-line warning not to restore the misattributed pair.  The 4.11.2 removal and the v4.15 re-bundle narratives moved.

```text
    # 4.11.2: the previously hard-coded Sellmeier coefficients for these
    # two glasses produced n_d = 1.8458 (S-LAH64) and 1.8853 (S-LAH79)
    # vs Ohara catalog n_d = 1.78800 and 2.00330 respectively -- off by
    # 0.058 and 0.118.  The in-code coefficients appear to be misattri-
    # buted from a different glass.  Removed from the in-code Sellmeier
    # table and routed through the authoritative refractiveindex.info
    # lookup via the '__sellmeier__' sentinel.  Requires ``pip install
    # refractiveindex`` -- without it, a glass lookup for these names
    # will fail with a clear error rather than silently returning a
    # ~3% wrong index.  Caught by AUDIT_ROUND3_2026_05_16.md (CRIT-1).
    #
    # v4.15 (P1-GL-1): re-bundle these as a fallback path.  v4.11.2 left
    # tuple-registered S-LAH64 / S-LAH79 with NO Sellmeier coefficients,
    # so a minimal install (without the ``refractiveindex`` Python pkg)
    # hit the dispatcher's ``ImportError`` branch on every lookup.
    # Bundling the OHARA Sellmeier table (sourced from the
    # refractiveindex.info-database YAMLs, OHARA Zemax 2017-11-30
    # catalog) restores the in-process fallback while still matching
    # ``refractiveindex`` to within 1e-9 (S-LAH64) / 4e-7 (S-LAH79) at
    # n_d.  Verified to ~5e-5 across 488 / 532 / 633 / 1064 / 1310 /
    # 1550 nm; rms residual relative to the refractiveindex.info
    # tabulated values stays below 1e-5 across the catalogued
    # 0.32-2.4 um (LAH64) / 0.37-2.4 um (LAH79) bands.
```

### L240-244 -- `SELLMEIER_COEFFICIENTS -- fused silica` -- what a minimal install did before the silica rows were bundled

*Left in the source:* the Malitson 1965 citation, the um^2 convention and the measured residual -- the S5 derivation of the rows.

```text
    # v4.15 (P1-GL-1): bundled Sellmeier fallback for tuple-registered
    # fused-silica entries.  Pre-4.15 these were registered as
    # (main, SiO2, Malitson) tuples in GLASS_REGISTRY but absent from
    # SELLMEIER_COEFFICIENTS, so a minimal install raised ImportError
    # on every silica lookup.  Malitson 1965 (J. Opt. Soc. Am.
```

### L266-266 -- `SELLMEIER_COEFFICIENTS -- CDGM header` -- the release tag on the section header

*Left in the source:* the section header and the whole formula / provenance / cross-check block under it.

```text
    # v4.16.0 (ROADMAP #13) -- CDGM Sellmeier bundled fallback
```

### L342-342 -- `POLYNOMIAL_COEFFICIENTS header` -- the release tag on the section header

*Left in the source:* the header and the formula-3 definition.

```text
# v4.16.2 (pre-v5.0 prep) -- formula-3 (polynomial) bundled evaluator
```

### L352-359 -- `POLYNOMIAL_COEFFICIENTS header` -- the three-release rollout of the formula-3 path (entries in one release, evaluator in the next, coefficients in a third)

*Left in the source:* the formula, the storage layout, the add-an-entry recipe and the 5e-5 cross-check bar.

```text
# v4.16.0 introduced the catalogue entries (Hikari E-/J-, Sumita K-, 4
# CDGM polynomial glasses) but routed them exclusively through the
# optional ``refractiveindex`` package.  v4.16.2 lands the bundled
# evaluator infrastructure so minimal installs no longer fail-fast on
# these 26 entries; per-glass coefficient ingestion is staged for
# v5.0 (one-shot import from the refractiveindex.info YAML dataset
# requires a non-trivial vendor-source review for each catalogue).
#
```

### L373-373 -- `POLYNOMIAL_COEFFICIENTS header` -- which releases the 5e-5 cross-check methodology came from

*Left in the source:* the bar itself (5e-5 against refractiveindex.info's tabulated n_d) and the fact that it is the same bar the Sellmeier rows are held to.

```text
# to 5e-5 (matches the v4.14.2 / v4.16.0 cross-check methodology).
```

### L376-377 -- `POLYNOMIAL_COEFFICIENTS` -- the release tag on the ingestion note

*Left in the source:* everything else: the count, the per-vendor sources, the bit-for-bit transcription rule and the stub-removal protocol.

```text
    # v5.2.3 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
    # 24 formula-3 catalogue entries (4 CDGM + 10 Hikari + 10 Sumita)
```

### L515-540 -- `_POLYNOMIAL_STUB_NAMES` -- the generic-ImportError path that predated the stub manifest, and the release that emptied the manifest

*Left in the source:* what the set is for (typo vs known-but-unbundled), that it is empty today, and why the empty frozenset is kept rather than deleted.

```text
# v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
# manifest of formula-3 polynomial glass names that are present in
# GLASS_REGISTRY (as ``(shelf, book, page)`` tuples for the optional
# refractiveindex live lookup) but NOT yet ingested into
# POLYNOMIAL_COEFFICIENTS for the bundled-evaluator fallback.
#
# This set lets the dispatcher distinguish "user typo / unknown
# glass" (ValueError with suggestions) from "known formula-3 glass
# without bundled coefficients" (NotImplementedError directing the
# user to install the [glass] extra or open a coefficient-ingestion
# request).  Pre-v5.2 the latter path raised a generic ImportError
# that conflated "package not installed" with "package installed but
# this glass not yet covered", which made the v5.2.1 ingestion gap
# invisible to users.
#
# v5.2.3 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
# all 24 formula-3 catalogue entries (4 CDGM + 10 Hikari + 10 Sumita)
# have now been ingested into POLYNOMIAL_COEFFICIENTS verbatim from
# the refractiveindex.info YAML dataset, so the stub manifest is
# empty.  The frozenset is intentionally retained (rather than
# deleted) so that any future formula-3 catalogue additions land
# here as a stub before their coefficient row, and the migration-
# message dispatch arm in get_glass_index remains reachable.
#
# Catalogue and validity ranges for each entry are recorded in
# GLASS_REGISTRY / GLASS_VALIDITY (search by name).
```

### L545-549 -- `_guard_wavelength` -- the release that consolidated three copies of the wavelength guard

*Left in the source:* the whole rule, including the reason the helper is shared (a fix landing in one evaluator cannot silently skip the others) and the sign-symmetric / not-sign-symmetric split.

```text
    """v5.4.6 (audit P3-4 / P3-7): shared negative / NaN wavelength guard
    for the dispersion evaluators, so ``glass._sellmeier_index``,
    ``glass._polynomial_index`` and ``coatings._coating_sellmeier`` all
    handle bad wavelengths identically (a fix landing in one no longer
    silently skips the others).
```

### L602-609 -- `_polynomial_index` -- which release added array support and which contract it was matching

*Left in the source:* the scalar-or-array contract itself, the return types and the numpy-convention note.

```text
    v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
    accepts either a Python scalar (returns float, matching the v4.16.2
    contract and the ``_sellmeier_index`` sibling) or an array-like
    ``wavelength_m`` (returns an ndarray with the same shape).  Array
    inputs follow the numpy convention -- JAX / CuPy callers can pass
    their own arrays through and will receive numpy back; downstream
    consumers in ``get_glass_index`` always call this with a scalar
    so the contract stays narrow.
```

### L656-672 -- `_sellmeier_index` -- the opaque `math domain error` the resonance guard replaced, and the two exceptions an array input used to raise

*Left in the source:* what the guard does now and what it raises; the scalar-or-array contract; and the measured 0-ULP agreement between the two paths.

```text
    4.10: validates that the wavelength does not coincide with a
    Sellmeier resonance (``lam² ≈ C_i``) and that the radicand stays
    positive.  Pre-4.10 a wavelength near a resonance raised an opaque
    ``math domain error``; this version raises ``ValueError`` with the
    glass name and the offending wavelength.

    R-12 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): accepts either a
    Python scalar (returns ``float``, the historical contract, on a pure
    ``math`` fast path) or an array-like ``wavelength_m`` (returns an
    ndarray of the same shape) -- mirroring the ``_polynomial_index``
    sibling, whose docstring already claimed the two were at parity.
    Pre-fix an array input died on ``abs(lam2 - ci) < 1e-12`` with
    numpy's opaque "truth value of an array with more than one element
    is ambiguous", and a list died with "can't multiply sequence by
    non-int of type 'float'".  The scalar path is bit-identical (same
    ``_math.sqrt`` of the same float expression); the vector path agrees
    with a scalar loop to 0 ULP.
```

### L1004-1009 -- `_validity_warned` -- that the dispatcher emitted no signal at all before the validity warning existed

*Left in the source:* what the memo is, its key, and the live reason for warning (an extrapolated index is still a number, and it means nothing).

```text
# v4.16.0: one-shot warn-once memo for out-of-range Sellmeier
# extrapolations.  Pre-v4.16 the dispatcher emitted no signal when
# the user asked for a wavelength outside the documented Sellmeier
# fit band (e.g. ``get_glass_index('N-BK7', 200e-9)``), silently
# returning an extrapolated number that has no physical meaning.
# v4.16 warns the first time per (glass, wavelength_nm) pair.
```

### L1049-1055 -- `_GLASS_VALIDITY_REGISTRY_EXEMPTIONS` -- a note correcting this list's own earlier description

*Left in the source:* the corrected description -- the four bullets saying what each exempt name actually is, which is the part a reader needs.

```text
# v4.16.1 (audit P1-NEW-F2-1 / C.3): names exempt from the
# GLASS_VALIDITY -> GLASS_REGISTRY direction of the consistency check.  A
# GLASS_VALIDITY row for one of these must NOT be treated as drift.
#
# Note what these names actually are today, which is not what this list's
# earlier description claimed:
#
```

### L1335-1338 -- `_check_glass_registry_consistency` -- which release added each of the six structural checks

*Left in the source:* the count and the names of the six checks, plus the opt-in seventh.

```text
    Six structural checks (v4.14.2 forward + v4.15 reverse + v4.16.1
    GLASS_VALIDITY -> GLASS_REGISTRY + v4.16.1 tuple well-formedness +
    v4.16.3 polynomial forward/reverse), plus an opt-in seventh VALUE
    check.
```

### L1350-1388 -- `_check_glass_registry_consistency` -- the per-check release tags and the pre-v4.15 silent-dead-code narrative

*Left in the source:* every check's rule and its reachability argument -- an orphan row is unreachable because the dispatcher never consults the table unless the registry routes there.  That argument is why the check exists; only the "pre-v4.15" framing moved.

```text
    * **Forward** (v4.14.2): every ``'__sellmeier__'``-flagged
      registry entry must have a coefficient row.
    * **Polynomial forward** (v4.16.3, audit P3-NEW-F1-1): every
      ``'__polynomial__'``-flagged registry entry must have a
      :data:`POLYNOMIAL_COEFFICIENTS` row.  Sibling to the Sellmeier
      forward check.
    * **Polynomial reverse** (v4.16.3, audit P3-NEW-F1-1): every row
      in :data:`POLYNOMIAL_COEFFICIENTS` must appear in
      :data:`GLASS_REGISTRY`.  Sibling to the Sellmeier reverse check.
    * **Reverse** (v4.15, P2): every row in
      :data:`SELLMEIER_COEFFICIENTS` must appear in
      :data:`GLASS_REGISTRY`.  Pre-v4.15 a coefficient row added
      without a corresponding registry entry was silent dead code
      (the dispatcher never consulted ``SELLMEIER_COEFFICIENTS``
      unless the registry first routed there) -- the reverse check
      surfaces such an orphan immediately at import time.

      For tuple-style entries (where the registry routes to
      refractiveindex.info first), the coefficient row is a legal
      fallback for minimal installs and not an orphan.  We accept
      the reverse-direction membership regardless of whether the
      registry entry is ``'__sellmeier__'`` or a tuple -- both
      paths consult ``SELLMEIER_COEFFICIENTS`` (the tuple path only
      when ``_REFRACTIVEINDEX_AVAILABLE`` is False).
    * **GLASS_VALIDITY -> GLASS_REGISTRY** (v4.16.1 audit P1-NEW-F2-1
      / C.3): every key in :data:`GLASS_VALIDITY` must appear in
      :data:`GLASS_REGISTRY`.  A validity entry without a registry
      entry is unreachable -- the validity warn helper looks up by
      registry name, so an orphan GLASS_VALIDITY row never fires
      its warning.  (The reverse direction
      GLASS_REGISTRY -> GLASS_VALIDITY is intentionally NOT enforced
      as a hard requirement -- missing validity defaults to the
      no-warning sentinel ``(0.0, inf)``.  Users registering a custom
      callable glass typically do not declare a wavelength range.)
    * **Tuple well-formedness** (v4.16.1 audit P1-NEW-F2-1 / C.3):
      each GLASS_VALIDITY entry must be a 2-tuple
      ``(lambda_min, lambda_max)`` with ``lambda_min < lambda_max``
      and both finite-non-negative.  A malformed tuple would silently
      pass or always-fire on the dispatch path.
```

### L1400-1401 -- `_check_glass_registry_consistency` -- the release tag on the polynomial forward check

*Left in the source:* the check's description and its sibling relationship to the Sellmeier forward check.

```text
    # v4.16.3 (audit P3-NEW-F1-1): Forward for __polynomial__ flag ->
    # row must exist.  Sibling to the __sellmeier__ forward check above.
```

### L1411-1415 -- `_check_glass_registry_consistency` -- the release tag on the reverse check

*Left in the source:* the unreachability argument in full.

```text
    # Reverse (v4.15, P2): row -> registry entry must exist.  Without
    # a registry entry the row is unreachable: ``get_glass_index``
    # raises ``ValueError`` before it inspects SELLMEIER_COEFFICIENTS,
    # so adding coefficients but forgetting the registry pointer was
    # a silent no-op pre-v4.15.
```

### L1428-1428 -- `_check_glass_registry_consistency` -- the release tag on the polynomial reverse check

*Left in the source:* the rest of the comment unchanged.

```text
    # v4.16.3 (audit P3-NEW-F1-1): Reverse for POLYNOMIAL_COEFFICIENTS.
```

### L1446-1447 -- `_check_glass_registry_consistency` -- the release tag on the stub-manifest well-formedness check

*Left in the source:* the disjoint-state invariant and both of its failure modes.

```text
    # v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
    # _POLYNOMIAL_STUB_NAMES well-formedness.  Every stub manifest entry
```

### L1474-1474 -- `_check_glass_registry_consistency` -- the release tag on the GLASS_VALIDITY -> GLASS_REGISTRY check

*Left in the source:* the rest of the comment unchanged.

```text
    # v4.16.1 (audit P1-NEW-F2-1 / C.3): GLASS_VALIDITY -> GLASS_REGISTRY.
```

### L1492-1495 -- `_check_glass_registry_consistency` -- the release tag on the tuple well-formedness check

*Left in the source:* the tuple shape and the ordering / finiteness requirement.

```text
    # v4.16.1 (audit P1-NEW-F2-1 / C.3): tuple well-formedness for every
    # GLASS_VALIDITY entry.  Each must be a 2-tuple
    # (lambda_min, lambda_max) with lmin < lmax and both finite,
    # non-negative.
```

### L1505-1509 -- `_check_glass_registry_consistency` -- the release that started accepting numpy scalars and the release whose check users were tripping

*Left in the source:* the live trap: `isinstance(np.int32(0), (int, float))` is False on Python 3.10+, which is the only reason the tuple is widened.

```text
        # v4.16.2 (audit P3-NEW-F1-4): accept numpy scalars in addition
        # to native Python ``int`` / ``float``.  ``isinstance(np.int32(0),
        # (int, float))`` is False on Python 3.10+; users passing
        # ``GLASS_VALIDITY['X'] = (np.int32(300e-9), np.float32(700e-9))``
        # are tripping the v4.16.1 well-formedness check.
```

### L1582-1582 -- `_GLASS_CACHE_LOCK` -- the release tag

*Left in the source:* everything the comment says about the lock: what it covers, the precedent it follows, and the never-held-while-calling-out rule that keeps it out of any lock-order cycle.

```text
# v5.17.1 (audit P3-40): lock guarding ``_glass_value_cache`` LRU
```

### L1598-1615 -- `_glass_value_cache` -- a correction of this comment's own earlier "femtometre" wording, and the unbounded plain dict the LRU replaced

*Left in the source:* what the cache holds, its key (now stated in picometres with the expression that produces them), the sentinel-confirmed consultation rule that makes a stale hit impossible, and the eviction-safety argument.

```text
# v5.6: value cache for the IMMUTABLE-catalogue dispatch branches only
# (__sellmeier__, __polynomial__ and the refractiveindex-unavailable
# Sellmeier / polynomial fallback).  Keyed on (glass_name, wavelength in
# picometres) -> float index.  (v5.17.1 audit P3-40: the historical
# "femtometre" wording was wrong -- ``round(wavelength * 1e12)`` is
# picometre resolution, still far below any optical relevance.)  It is
# consulted ONLY inside those branches (after the entry sentinel is
# confirmed), so re-registering a name under a different dispatch (a
# callable, a tuple, register_fixed_glass) can never serve a stale
# value; ``register_fixed_glass`` clears it as well.  Array wavelengths
# bypass it (not hashable / not the hot scalar path).
#
# v5.17.1 (audit P3-40): LRU-bounded OrderedDict (was an unbounded plain
# dict) + enrolled in the central cache registry as ``'glass_caches'``
# (see ``_clear_glass_caches`` at the bottom of this module).  Values
# are immutable floats, so eviction can never mutate a value a caller
# already holds, and the recompute is a pure Sellmeier / polynomial
# evaluation -- byte-identical on hit, miss, and post-eviction.
```

### L1618-1618 -- `_GLASS_VALUE_CACHE_SIZE` -- the release tag on the bound rationale

*Left in the source:* the whole derivation.  TESTING_STANDARDS S5 requires a numeric bar to carry its sizing argument, and this one is the bar's only derivation.

```text
# v5.17.1 (audit P3-40) bound rationale: one entry costs ~100 B (tuple
```

### L1635-1635 -- `_cached_glass_value` -- the release tag

*Left in the source:* the LRU bound, the lock discipline and the compute-outside-the-lock race note.

```text
    v5.17.1 (audit P3-40): LRU-bounded at ``_GLASS_VALUE_CACHE_SIZE``
```

### L1657-1658 -- `_invalidate_glass_name` -- the release tag

*Left in the source:* what the function drops and why, and the sibling invalidation it mirrors.

```text
    """Drop every cached resolution for ``glass_name`` (v5.17.1, audit
    P2-41): the ``_glass_cache`` object (stale ``_FixedIndex`` or a
```

### L1695-1698 -- `_require_finite_catalogue_index` -- the pre-fix wording ("used to hand back nan")

*Left in the source:* the same hazard, stated as what happens without the guard: an out-of-range page interpolates to NaN rather than raising, and one multiplication later the NaN is across the whole field.

```text
    wavelength INTERPOLATES TO NaN instead of raising, so a lookup outside
    the page's range used to hand back ``nan`` -- ``get_glass_index('SILICON',
    633e-9)`` and, through it, ``get_glass_index_complex`` returning
    ``nan + 0j`` -- with only a validity *warning* to show for it.  One
```

### L1824-1824 -- `get_glass_index` -- the release tag

*Left in the source:* the ordering rule (warn before the lookup) and the list of entry kinds it skips.

```text
    # v4.16.0 (ROADMAP #14): validity-range warning.  Emitted before
```

### L1832-1838 -- `get_glass_index` -- the inverted "POLYNOMIAL -> SELLMEIER" wording that drifted into a release note, and the audit that reconciled it

*Left in the source:* the actual order (SELLMEIER -> POLYNOMIAL) and the disjointness argument that makes the order cosmetic for correctness.  `tests/unit/test_v4_16_3_agent_d.py::test_glass_dispatch_order_comment_matches_code` pins the literal `SELLMEIER -> POLYNOMIAL`, which is kept.

```text
    # v4.16.3 (audit P3-NEW-V3-1): dispatch order is SELLMEIER -> POLYNOMIAL,
    # NOT the "POLYNOMIAL -> SELLMEIER" wording that drifted into the
    # v4.16.2 CHANGELOG / release notes.  Both sentinels are disjoint at
    # the GLASS_REGISTRY level (a name carries one or the other) and the
    # consistency check enforces the row exists, so the order is
    # cosmetic for correctness; it matters only for documentation
    # truthfulness.
```

### L1850-1856 -- `get_glass_index` -- that the bundled polynomial evaluator was once reachable only on the refractiveindex-unavailable fallback

*Left in the source:* what the sentinel does now -- resolve via POLYNOMIAL_COEFFICIENTS regardless of whether refractiveindex is installed -- and the sibling parity note.

```text
    # v4.16.3 (audit P3-NEW-F1-1): __polynomial__ sentinel parallel to
    # __sellmeier__.  Pre-v4.16.3 the bundled formula-3 polynomial
    # evaluator was reachable ONLY via the refractiveindex-unavailable
    # fallback below -- which means a user who followed the
    # ``pip install lumenairy[glass]`` recommendation in the README
    # could never hit the bundled polynomial path.  The sentinel makes
    # the polynomial dispatch first-class: any glass registered as
```

### L1906-1911 -- `get_glass_index` -- a comment that CONTRADICTED the code: it said POLYNOMIAL_COEFFICIENTS was empty and that ingestion was still staged

*Left in the source:* a corrected statement of what the arm covers: all 24 formula-3 catalogue entries are bundled, so the arm resolves every registered formula-3 glass on a minimal install.

```text
        # v4.16.2 (pre-v5.0 prep): formula-3 polynomial fallback for
        # glasses whose coefficients have been ingested into
        # POLYNOMIAL_COEFFICIENTS.  Empty at v4.16.2 ship; populating
        # the 24 catalogue entries is staged for v5.2.1 (per-glass
        # vendor-source review against refractiveindex.info YAML +
        # 5e-5 n_d cross-check).
```

### L1918-1919 -- `get_glass_index` -- the release tag

*Left in the source:* the whole reason the arm raises NotImplementedError rather than ImportError -- the two remediations are different.

```text
        # v5.2 (ROADMAP v5.1 formula-3 polynomial coefficients ingestion):
        # known-but-stubbed formula-3 glass.  Raise NotImplementedError
```

### L2012-2013 -- `get_glass_index_complex` -- which release added __polynomial__ alongside __sellmeier__

*Left in the source:* the rule: these paths have no extinction data, so kappa = 0 is returned explicitly.

```text
    # return kappa = 0 explicitly.  v4.16.3 (audit P3-NEW-F1-1):
    # __polynomial__ added parallel to __sellmeier__.
```

### L2105-2109 -- `<module> cache-registry enrollment` -- that clear_asm_caches never drained these four caches before the enrollment landed, and the case-sensitive filter that hid them

*Left in the source:* what the enrollment buys, and the live warning that the meta-pin's discovery is name-shape sensitive.

```text
# v5.17.1 (audit P3-40): central-registry enrollment for this module's
# four caches.  Pre-v5.17.1 ``clear_asm_caches`` /
# ``lumenairy_context(clear_caches_on_exit=True)`` never drained them
# (the enrollment meta-pin's case-sensitive ``endswith('_CACHE')``
# filter missed the lower-case names -- audit P2-42).
```

