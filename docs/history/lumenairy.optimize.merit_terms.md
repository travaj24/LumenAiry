<!-- lumenairy-history-doc
module: lumenairy/optimize/merit_terms.py
ast_sha256: 5193e38d3913a76ea592de50f0078fbeebd6afbf45e7a82fa7ab430d13c29409
token_sha256: 9367c850e66293a96f63b0a1d7afdb7d75d858e57d8f6a484fd1ae382da26193
pre_relocation_lines: 2002
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/optimize/merit_terms.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/merit_terms.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

**Two things about this module's relocation are deliberate and worth stating.**

First, the `.. versionchanged:: 5.30` block on `MatchIdealSystem` was NOT
touched.  It documents five kwargs that were REMOVED and now raise `TypeError`,
and it carries the migration recipe for each.  That is a live instruction to a
user upgrading, not narrative -- the same line part 1 drew around
`set_pyfftw_planner`'s migration note -- so it stays in the source in full,
grep-verification sentence included (that sentence is the evidence the removal
was safe).

Second, every edit ABOVE line 600 is line-count neutral, on purpose.
`CHANGELOG.md` cites `optimize/merit_terms.py:638` for
`MatchIdealSystem._make_source`'s `ap > 0` branch, and
`tests/unit/test_v4_15_agent_f.py` re-derives that line by anchor string and
requires the citation to sit within +/- 5 of it.  Holding the anchor at its
existing line keeps that pin green without anyone having to re-cite a line
number -- which is exactly the drift class the pin exists to catch, and not a
thing a comment relocation should be spending.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L4-4 | `<module> docstring` | the release and agent that split this module out of core.py |
| L14-15 | `<module> docstring` | "no public-API behaviour changes" -- a statement about the split, not about the module |
| L160-166 | `StrehlMerit.evaluate` | the release and audit tag |
| L178-178 | `RMSWavefrontMerit` | the release and audit tag |
| L408-409 | `MatchIdealSystem` | the release tag and "is no longer reachable" |
| L582-597 | `MatchIdealSystem._make_source` | the release/audit tag and the "Pre-v4.14.2 the ap > 0 check silently fell through ... the exact bug v4.14.1 fixed in the wrapper-merit cache but missed at this pre-existing site" narrative |
| L605-607 | `MatchIdealSystem._make_source` | the release/audit tag and the release that ran the dtype sweep |
| L636-639 | `MatchIdealSystem._expand_prescription_placeholder` | the release / wave tag and "is removed" |
| L716-721 | `MatchIdealSystem.evaluate` | the release / wave tag and the removal narrative (the flag, its sole caller, and the deleted helper) |
| L1234-1234 | `LGAberrationMerit.evaluate` | the release and audit tag |
| L1299-1299 | `LGAberrationMerit.evaluate` | the release and audit tag |
| L1384-1391 | `LGAberrationMerit.evaluate` | "the old ``|L|^2`` drove the design toward |Strehl| = 0" and "this is verbatim the fix the JAX twin already carries" |
| L1393-1403 | `LGAberrationMerit.evaluate` | the release tag and the "Before v5.46 it was the bare |L|^2" framing |
| L1475-1475 | `NormalizedMerit` | the release and audit tag |
| L1486-1490 | `NormalizedMerit` | "byte-identical to their historical behaviour" |
| L1711-1712 | `MinThicknessMerit.evaluate` | the release/audit tag and "byte-identical to the pre-dedup inline loop" |
| L1722-1727 | `MaxThicknessMerit` | the release/audit tag and the "Pre-fix this iterated EVERY entry" narrative |
| L1735-1736 | `MaxThicknessMerit` | "restore the pre-fix behaviour" as the description of include_air |

---

### L4-4 -- `<module> docstring` -- the release and agent that split this module out of core.py

*Left in the source:* the split relationship, which is live.

```text
v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
```

### L14-15 -- `<module> docstring` -- "no public-API behaviour changes" -- a statement about the split, not about the module

*Left in the source:* the re-export fact and that core.py remains the documented import path.

```text
public names are re-exported from ``optimize/core.py``; no public-API
behaviour changes.
```

### L160-166 -- `StrehlMerit.evaluate` -- the release and audit tag

*Left in the source:* the whole coercion contract: why float() is used, which classes write the sentinel, and that an identity check is available before the cast.

```text
        # v4.15.3 (P1-NEW-F1-3): coerce ``ctx.strehl_best`` via
        # ``float()`` so the
        # ``_FAILED_SCAN_STREHL_SENTINEL_OBJ`` singleton written by
        # ``MultiFieldMerit`` / ``ToleranceAwareMerit`` collapses to
        # its scalar fallback (0.0) for arithmetic.  Identity-check
        # is available before the cast for callers that want to
        # distinguish a real-zero Strehl from a failed-scan zero.
```

### L178-178 -- `RMSWavefrontMerit` -- the release and audit tag

*Left in the source:* the entire OSA-index derivation of the default -- which modes index 4 drops and keeps, why contiguous slicing cannot remove defocus while keeping both astigmatism orientations, and what a genuinely defocus-insensitive variant would need.  That is the derivation of a live default, which TESTING_STANDARDS S5 keeps with the value.

```text
    v5.4.6 (audit F-4): the default ``exclude_low_order=4`` drops OSA
```

### L408-409 -- `MatchIdealSystem` -- the release tag and "is no longer reachable"

*Left in the source:* the live fact: the traced variant is not reachable through a flag and must be written out explicitly.

```text
        ``'real_lens'`` element.  (v5.30: the ``'real_lens_traced'``
        variant is no longer reachable through a flag -- write that
```

### L582-597 -- `MatchIdealSystem._make_source` -- the release/audit tag and the "Pre-v4.14.2 the ap > 0 check silently fell through ... the exact bug v4.14.1 fixed in the wrapper-merit cache but missed at this pre-existing site" narrative

*Left in the source:* all three branches and the full consequence of dropping the `<= 0` arm, re-stated as a present-tense hazard, with the cross-reference to the wrapper-merit sentinel branch kept.

```text
            # v4.14.2 (P1-NEW-1): three branches matching the canonical
            # _ZERO_APERTURE_MASK semantics shared by
            # ``MultiWavelengthMerit.evaluate``,
            # ``MultiFieldMerit.evaluate``, and
            # ``ToleranceAwareMerit.evaluate``:
            #   * ``ap`` finite and > 0   -> circular boolean mask.
            #   * ``ap`` finite and <= 0  -> deliberate-zero aperture;
            #     block all light by zeroing E entirely.  Pre-v4.14.2
            #     the ``ap > 0`` check silently fell through to the
            #     ``else`` branch and produced a full-grid plane wave,
            #     which apply_real_lens would then propagate as a
            #     bright on-axis "source" -- the exact bug v4.14.1
            #     fixed in the wrapper-merit cache but missed at this
            #     pre-existing site.
            #   * ``ap`` is None or non-finite -> no aperture specified;
            #     full-grid plane wave (unchanged behaviour).
```

### L605-607 -- `MatchIdealSystem._make_source` -- the release/audit tag and the release that ran the dtype sweep

*Left in the source:* the dtype hazard itself (a 0.0+0.0j literal upcasts complex64) and the list of sibling sites that share the rule.

```text
                # v4.14.2 (P1-NEW-4): dtype-aware zero so a complex64
                # cdtype is not silently upcast to complex128 by the
                # 0.0+0.0j literal.  Mirrors the v4.13.2 sweep at
```

### L636-639 -- `MatchIdealSystem._expand_prescription_placeholder` -- the release / wave tag and "is removed"

*Left in the source:* what the placeholder expands to today and what a caller who wants the traced propagator must write instead.

```text
        # v5.30 (W5): ``use_traced_lens`` is removed, so the placeholder
        # always expands to the default ``'real_lens'`` element.  A caller
        # who wants the traced propagator writes that element out in
        # ``real_elements`` explicitly (with its own ``ray_subsample``).
```

### L716-721 -- `MatchIdealSystem.evaluate` -- the release / wave tag and the removal narrative (the flag, its sole caller, and the deleted helper)

*Left in the source:* the live guidance: there is no focus search, and here is the explicit propagate element that replaces it.

```text
        # v5.30 (W5): the optional axial focus search is REMOVED with the
        # ``focus_search`` flag that was its only gate, and
        # ``_focus_search_penalty`` is deleted with it (grep-verified:
        # ``self.focus_search`` here was the sole caller).  To decouple
        # "correct focal plane" from "aberration quality", put an explicit
        # ``{'type': 'propagate', 'z': dz}`` offset in ``ideal_elements``.
```

### L1234-1234 -- `LGAberrationMerit.evaluate` -- the release and audit tag

*Left in the source:* the entire caching argument -- that the reference is a pure function of the listed inputs, and the per-term cost it saves on a CompositeMerit.

```text
        # v5.46 (VERIFY-A4 follow-up, O-3): the aberration-free reference is a
```

### L1299-1299 -- `LGAberrationMerit.evaluate` -- the release and audit tag

*Left in the source:* the dimensional-cancellation argument, which is why the reference is evaluated with the SAME w_o.

```text
                # v5.46 (audit Y2 follow-up): the same evaluation on the
```

### L1384-1391 -- `LGAberrationMerit.evaluate` -- "the old ``|L|^2`` drove the design toward |Strehl| = 0" and "this is verbatim the fix the JAX twin already carries"

*Left in the source:* the sign argument in full (design_optimize MINIMISES, so the (0,0) channel must be a deficit), the JAX-twin agreement, and the rule that every other channel keeps |L|^2.

```text
                    # MINIMISES the weighted merit sum, so the old ``|L|^2``
                    # drove the design toward |Strehl| = 0 (MAXIMUM
                    # aberration).  This is verbatim the fix the JAX twin
                    # ``make_lg_aberration_merit_jax`` already carries
                    # (jax_merits.py: ``piston_weight * (1 - |res|^2)``);
                    # the two now agree numerically on the (0, 0) target.
                    # Every OTHER (p, ell) channel keeps ``|L|^2`` -- driving
                    # a named aberration channel to zero IS the intent there.
```

### L1393-1403 -- `LGAberrationMerit.evaluate` -- the release tag and the "Before v5.46 it was the bare |L|^2" framing

*Left in the source:* the normalisation and both measured numbers (4.79e+14 and 3.2e-03) as evidence that a bare |L|^2 is not a Strehl ratio, plus the dimensional-accident explanation.  Those numbers are the derivation of the normalisation and stay with it.

```text
                    # v5.46 (audit Y2 follow-up): ``mag_sq`` is now
                    # ``|L|^2 / |L_ref(0,0)|^2``, dimensionless and exactly
                    # 1.0 on an aberration-free optic, so ``1 - mag_sq`` is a
                    # real deficit in [0, 1] on the default 'sigma' branch.
                    # Before v5.46 it was the bare ``|L|^2`` -- 4.79e+14 on
                    # the stock singlet after the Van Vleck normalisation
                    # (audit Y2) and 3.2e-03 before it, neither of which is a
                    # Strehl ratio; the old number only LOOKED like one
                    # because the missing lambda*sqrt(|det J|) (a LENGTH)
                    # cancelled the closed-form branch's 1/length^2 by
                    # dimensional accident.
```

### L1475-1475 -- `NormalizedMerit` -- the release and audit tag

*Left in the source:* the whole scale survey -- dimensionless vs m^2 vs dioptre^2, the ~1e12 weight pre-compensation it forces -- which is the argument for the wrapper existing.

```text
    v5.25 (audit S4-18 / B3): the built-in merit families evaluate on
```

### L1486-1490 -- `NormalizedMerit` -- "byte-identical to their historical behaviour"

*Left in the source:* the opt-in rule and the re-tune-your-weights instruction, stated as a property of the current default path.

```text
    **This is strictly OPT-IN.**  Unwrapped merit terms are byte-identical
    to their historical behaviour (the ``design_optimize`` default path is
    unchanged); wrapping is a deliberate choice that re-bases the weight
    calibration onto the common dimensionless scale, so re-tune weights
    when you adopt it.
```

### L1711-1712 -- `MinThicknessMerit.evaluate` -- the release/audit tag and "byte-identical to the pre-dedup inline loop"

*Left in the source:* that the classification helper is shared with the sibling class.

```text
                # v5.24.x (audit S4-18): shared glass/air classification;
                # behaviour byte-identical to the pre-dedup inline loop.
```

### L1722-1727 -- `MaxThicknessMerit` -- the release/audit tag and the "Pre-fix this iterated EVERY entry" narrative

*Left in the source:* the rule and the reason for it -- an object- or image-space air gap is not a manufacturability constraint on the glass -- re-stated as what counting every entry WOULD do, plus the pointer to this file.

```text
    v5.24.x (audit S4-18): only glass thicknesses count; air gaps are
    skipped, matching the documented intent ("glass thickness") and the
    :class:`MinThicknessMerit` sibling.  Pre-fix this iterated EVERY
    entry in ``prescription['thicknesses']`` including air gaps -- so a
    large object/image-space air gap (which is not a manufacturability
    constraint on the glass) was penalised contra the docstring.
```

### L1735-1736 -- `MaxThicknessMerit` -- "restore the pre-fix behaviour" as the description of include_air

*Left in the source:* what the flag does now and its default.

```text
        Set True to restore the pre-fix behaviour and also penalise
        large air gaps.  Default False.
```

