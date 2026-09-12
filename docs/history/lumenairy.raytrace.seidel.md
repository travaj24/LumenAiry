<!-- lumenairy-history-doc
module: lumenairy/raytrace/seidel.py
ast_sha256: 7984f9551684714d0f80cb0edbfd4a68c7b6be9d917d2c8389ad3883a44ede30
token_sha256: c5fd346826c05552072d63ca6dcbbbf618efce1e53c5f8c72819636eddf2fd58
pre_relocation_lines: 1913
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/seidel.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/seidel.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

This module carries more MEASURED EVIDENCE per comment than anything else in
the partition -- exact ray-trace oracles, per-design percentage errors, bit
identities.  Almost none of it moved: `docs/TESTING_STANDARDS.md` S5 wants a
numeric claim to carry its oracle, and here the oracle IS the argument for a
sign convention.  What moved is the tense and the attribution: `pre-fix the
branch read ...`, `this line used to read ...`, `Pre-4.10 used ...` are now
stated as what the WRONG form does, which is the thing a future editor tempted
to simplify the branch needs to read.

Two blocks are of a rarer kind and moved in full: notes in which a comment
corrects an **earlier draft of itself** -- "an earlier draft of this fix
applied the sign here and broke seven S11-1 pins; that was a category error,
retracted" and "an earlier draft of this note wrongly called ``fnum``
defective on that basis; the measurement retracts it".  Those are the
documentation's own history.  The source now states the settled rule once.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L29-30 | `<module> docstring` | the split-provenance note |
| L206-208 | `_unfolded_bfl conversion note` | a note correcting an EARLIER DRAFT OF ITSELF -- the sign applied in the wrong frame, seven broken S11-1 pins, and the retraction |
| L229-231 | `EFL CONVENTION note` | a second self-correction -- an earlier draft of this note calling ``fnum`` defective, and the retraction |
| L315-322 | `system_abcd mirror branch` | "Pre-fix the branch read ``elif surf.is_mirror and np.isfinite(R)``" |
| L327-329 | `system_abcd mirror branch` | "The old gating was also DISCONTINUOUS in R" |
| L833-834 | `_pre_stop_abcd` | "Before AUDIT_... R-1 the two built it independently and DISAGREED" |
| L884-886 | `_pre_stop_abcd notes` | "pre-fix the flat-fold system returned" |
| L931-936 | `_post_stop_abcd` | "R-1 follow-on: ``compute_pupils`` used to supply that leg as a dummy air-to-air Surface" |
| L1034-1042 | `compute_pupils notes` | "this line used to read ``ep_z = -B / A``" and "the old expression gives -t" |
| L1085-1087 | `compute_pupils` | "the two used to disagree" |
| L1101-1106 | `compute_pupils` | the release/audit tag, "Pre-fix this line was a bare expression" and "was ``-B/A``" / "was reduced" |
| L1131-1132 | `compute_pupils` | "pre-fix this line read ``-B/D``" |
| L1191-1195 | `compute_pupils` | "Pre-4.10 used ``stop_radius * D_post``" |
| L1396-1401 | `seidel_coefficients` | "Bit-for-bit identical to the previous inline construction here." |
| L1540-1544 | `_aspheric_seidel` | "The per-surface loop used to read only ..." and the ``grep`` that found zero hits |
| L1636-1640 | `seidel_coefficients mirror branch` | "Pre-fix a FLAT fold mirror fell into the flat-REFRACTOR branch" |
| L1759-1763 | `seidel_coefficients flat branch` | "Pre-4.9 zeroed S1/S2/S3 here" and "that the old branch dropped silently" |

---

### L29-30 -- `<module> docstring` -- the split-provenance note

*Left in the source:* the bit-for-bit claim, re-stated as a property of this module's contents.

```text
No physics change: contents are bit-for-bit copies of the original
implementations.
```

### L206-208 -- `_unfolded_bfl conversion note` -- a note correcting an EARLIER DRAFT OF ITSELF -- the sign applied in the wrong frame, seven broken S11-1 pins, and the retraction

*Left in the source:* the whole two-way measurement (10 designs, <= 1.7e-11 agreement, the concave-mirror worked example) and the settled rule, plus a one-line warning that applying the sign here is the wrong frame.

```text
    focus at -100.000000 mm and unfolded ``bfl`` +100.000000 mm.  An
    earlier draft of this fix applied the sign here and broke seven
    S11-1 pins; that was a category error, retracted.
```

### L229-231 -- `EFL CONVENTION note` -- a second self-correction -- an earlier draft of this note calling ``fnum`` defective, and the retraction

*Left in the source:* both formulas, both measured values (11.264729957829 vs 17.086342788618) and the conclusion that ``fnum`` is correct as written.

```text
      on that design and is the AIR-ONLY formula.  An earlier draft of
      this note wrongly called ``fnum`` defective on that basis; the
      measurement retracts it.
```

### L315-322 -- `system_abcd mirror branch` -- "Pre-fix the branch read ``elif surf.is_mirror and np.isfinite(R)``"

*Left in the source:* the Welford argument, the wrong-gating consequence and every measured number, re-stated as what that gating does.

```text
        # ``u' = nu'/n' = nu/(-n) = -u`` when c = 0.  Pre-fix the branch
        # read ``elif surf.is_mirror and np.isfinite(R)``, so a FLAT fold
        # mirror fell into the powerless ``np.eye(2)`` branch and skipped
        # the bookkeeping entirely, leaving every downstream leg with the
        # wrong index sign AND the wrong ray slope.  Measured on
        # ``[flat fold(t=0.2), concave R=-1]``: EFL/BFL +0.500000 where
        # the exact 3-D trace (raytrace.trace, no shared code) gives
        # -0.500000 and the R = -1e9 curved fold gives -0.500000; on
```

### L327-329 -- `system_abcd mirror branch` -- "The old gating was also DISCONTINUOUS in R"

*Left in the source:* the discontinuity and both measured EFLs.

```text
        # [12.700, 17.780, 12.700] mm).  The old gating was also
        # DISCONTINUOUS in R: R = -1e12 gave EFL -0.1985770345 while
        # R = inf gave +0.1985770345 on the same folded singlet.
```

### L833-834 -- `_pre_stop_abcd` -- "Before AUDIT_... R-1 the two built it independently and DISAGREED"

*Left in the source:* the single-source rule and the fact that independent construction is what makes the two disagree.

```text
    conditions).  Before AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 R-1 the
    two built it independently and DISAGREED (see below).  Identity when
```

### L884-886 -- `_pre_stop_abcd notes` -- "pre-fix the flat-fold system returned"

*Left in the source:* the whole exactness signature -- bit-identical pupils to the mirrorless control, the zero-gap vanishing, the even-parity exactness -- and all four measured error figures above it.

```text
    The signature is exact: pre-fix the flat-fold system returned
    BIT-IDENTICAL pupils to the mirrorless control (the fold's only
    paraxial effect on this leg IS the sign), the error vanishes iff the
```

### L931-936 -- `_post_stop_abcd` -- "R-1 follow-on: ``compute_pupils`` used to supply that leg as a dummy air-to-air Surface"

*Left in the source:* the prohibition and all four measured errors (+3.8% / +7.9% xp_z, +1.9% / +2.5% xp_radius).

```text
    R-1 follow-on: ``compute_pupils`` used to supply that leg as a dummy
    air-to-air ``Surface``, so it was always evaluated in AIR.  When the
    stop's image-side medium is glass (a front stop declared on a lens
    surface, or a stop inside a cemented block) the leg was short by a
    factor ``n``: measured ``xp_z`` error +3.8% (stop at a BK7 surface),
    +7.9% (stop inside BK7); ``xp_radius`` +1.9% / +2.5%.
```

### L1034-1042 -- `compute_pupils notes` -- "this line used to read ``ep_z = -B / A``" and "the old expression gives -t"

*Left in the source:* the whole sign derivation, the exact-real-ray discriminator and both masking conditions -- as an argument about which expression is right.

```text
    R-1 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): this line used to read
    ``ep_z = -B / A``, i.e. it returned the object DISTANCE (positive to
    the left) under the name of a SIGNED COORDINATE -- the mirror image
    of the true pupil plane.  Exact-real-ray discriminator: with a
    powerless pre-stop leg (flat dummy, gap ``t``, then the stop) the EP
    *is* the stop, at ``z_ep = +t``; the old expression gives ``-t``.
    The defect was masked at ``stop_index == 0`` (``z_ep = 0``) and, for
    every other system, by the missing pre-stop transfer above, which
    left ``B`` too small to notice.
```

### L1085-1087 -- `compute_pupils` -- "the two used to disagree"

*Left in the source:* the single-source rule and the failure independent construction produces.

```text
        # R-1: the pre-stop sub-system comes from the SINGLE shared
        # builder that ``seidel_coefficients`` also uses -- the two used
        # to disagree (this side was missing the final leg to the stop).
```

### L1101-1106 -- `compute_pupils` -- the release/audit tag, "Pre-fix this line was a bare expression" and "was ``-B/A``" / "was reduced"

*Left in the source:* the assignment requirement and the UnboundLocalError it prevents, plus both convention notes.

```text
            # v5.4.6 (audit F-2): ``ep_z`` MUST be assigned here.  Pre-fix
            # this line was a bare expression whose value was discarded,
            # leaving ``ep_z`` unbound on every non-front-stop system ->
            # UnboundLocalError at the ``return PupilInfo(...)`` line.
            # R-1: signed coordinate ``+B/A`` (see Notes) -- was ``-B/A``.
            # W4: ``* n_obj`` -- was reduced (bit-identical in air).
```

### L1131-1132 -- `compute_pupils` -- "pre-fix this line read ``-B/D``"

*Left in the source:* the whole Welford-frame argument for the ``n_out`` factor.

```text
        # W3-T2 (mirror parity): pre-fix this line read ``-B/D``, i.e. it
        # dropped ``n_out`` entirely.  ``_post_stop_abcd`` works in the
```

### L1191-1195 -- `compute_pupils` -- "Pre-4.10 used ``stop_radius * D_post``"

*Left in the source:* the magnification algebra and the 1/D^2 consequence of the angular form, as a live warning.

```text
            # (AD−BC)/D = 1/D for air-to-air systems (det M = 1).  Pre-
            # 4.10 used `stop_radius * D_post` (the angular magnification,
            # not transverse) — every XP-radius downstream consumer
            # (vignetting, f/#, Seidel) was wrong by 1/D² for non-trivial
            # post-stop systems.
```

### L1396-1401 -- `seidel_coefficients` -- "Bit-for-bit identical to the previous inline construction here."

*Left in the source:* the single-source rule and the R-1 finding that motivates it.

```text
    # shared source for this sub-system: R-1 of
    # AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 found ``compute_pupils``
    # building the same split WITHOUT the final leg, so the two
    # disagreed on the pre-stop system (and hence on the entrance
    # pupil) for every non-front-stop design.  Bit-for-bit identical to
    # the previous inline construction here.
```

### L1540-1544 -- `_aspheric_seidel` -- "The per-surface loop used to read only ..." and the ``grep`` that found zero hits

*Left in the source:* the whole base-sphere failure mode and its measurement (the k = -1 parabola reported as aberrated while measuring aberration-free to 1.4e-17 m).

```text
        loop used to read only ``radius`` / glasses / ``thickness`` /
        ``is_mirror``: ``grep -n 'conic\\|aspheric' seidel.py`` returned
        ZERO hits, so a conic or aspheric surface silently reported the
        sums of its BASE SPHERE, with no warning and no docstring note.
        Measured pre-fix: a mirror R = -200 mm, h = 25 mm reported
```

### L1636-1640 -- `seidel_coefficients mirror branch` -- "Pre-fix a FLAT fold mirror fell into the flat-REFRACTOR branch"

*Left in the source:* the branch-ordering rule and the full consequence of the wrong branch, in the present tense.

```text
        # note in :func:`system_abcd`.  Pre-fix a FLAT fold mirror fell
        # into the flat-REFRACTOR branch below, which does not set
        # ``n2 = -n1`` and does not flip ``mirror_parity``, so every
        # surface downstream of a flat fold was evaluated at the wrong
        # effective index sign and the wrong marginal/chief ray height.
```

### L1759-1763 -- `seidel_coefficients flat branch` -- "Pre-4.9 zeroed S1/S2/S3 here" and "that the old branch dropped silently"

*Left in the source:* the physics -- a flat surface inside a stack does contribute -- and the plano-convex hand calc that shows it.

```text
            # Pre-4.9 zeroed S1/S2/S3 here -- but a flat surface inside
            # a stack contributes to spherical / coma / astigmatism
            # exactly as the audit's plano-convex hand calc showed:
            # the R2=∞ surface of a plano-convex singlet has a real
            # S1 contribution that the old branch dropped silently.
```

