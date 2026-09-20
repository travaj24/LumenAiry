<!-- lumenairy-history-doc
module: lumenairy/analysis/image_plane_wfe.py
ast_sha256: 540048fdfe24b40a46ba60f650573d06fddeee50909061eaca6f615bb2b5b34e
token_sha256: bdf4951e165d69177385d5b486bdd02c3522b58f14277cfb53487c6a9585eb5d
pre_relocation_lines: 1225
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-20 -- WP-C2 round 2 (VERIFY-WP-C2 D4): this module's exported internally-tracing entry point(s) take the tracer's own sphere_normal= / renormalize= keywords (default None, which stamps nothing) and forward them verbatim to the trace call, so the pre-WP-C2 arithmetic is one keyword away; 742/742 arrays byte-identical archive to archive on both builds
-->

# Version history -- `lumenairy/analysis/image_plane_wfe.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/image_plane_wfe.py`.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

Every block here is the same shape, and it is the shape the WP-A17 SWEEP-1
follow-up pass was asked to close: a **live guard whose comment explained
itself by naming the release that added it** ("Pre-fix X happened", "Pre-4.12
the dispatcher only passed ...").  The hazard X is still reachable -- the guard
is the only thing preventing it -- so the source now states X in the present
tense, as what goes wrong WITHOUT the guard, together with every measurement
that sizes it.  What moved is the release attribution and the
"bit-identical to pre-fix" reassurance that travelled with it.

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
| L326-328 | `wavefront_error_map` -- `sphere_tangent='vertex'` | the `pre-3.8.2 versions of this function` attribution |
| L515-524 | `_conjugate_distances` -- the index-threaded Gauss equation | the `Pre-fix, measured against an exact real-ray oracle` framing |
| L596-599 | `_pupil_aim` -- the exit-pupil aim point | the `Pre-4.10 always aimed at z=0` framing |
| L696-701 | `_chief_index` -- the ALIVE constraint | the `Pre-4.10 could pick a dead vignetted on-axis ray` framing |
| L732-737 | `_reference_sphere` -- the 1/N_chief path length | the `Pre-4.12.0 only the chief-image landing got the factor (the v4.11.2 fix)` framing |
| L747-752 | `_reference_sphere` -- the folded frame | the `Pre-fix, on an air singlet + flat fold` framing |
| L853-857 | `_best_rms_defocus` -- the sphere radius | the `pre-4.10 always used 1/img_d_m` framing |

---

### L326-328 -- `wavefront_error_map` -- `sphere_tangent='vertex'` -- the `pre-3.8.2 versions of this function` attribution

*Left in the source:* the convention and its live sibling

```text
          Radius = ``img_d_m``.  Simplest convention; what
          ``conv_a_to_rs_opd`` and pre-3.8.2 versions of this
          function used.
```

### L515-524 -- `_conjugate_distances` -- the index-threaded Gauss equation -- the `Pre-fix, measured against an exact real-ray oracle` framing

*Left in the source:* the whole oracle comparison: both glass cases with their percentage errors and wave counts, the accuracy the threaded form reaches, and the warning that the two errors are NOT a common factor

```text
        # Pre-fix, measured against an exact real-ray oracle (a ray from
        # the axial object point, axis crossing read past the last
        # surface): an N-BK7 IMAGE space gave img_d_m = +41.569590 mm vs
        # +70.557940 mm exact (-41.1%, and the resulting misplaced
        # reference sphere reported 114.8 waves PV); an N-BK7 OBJECT
        # space gave +34.023215 mm vs +35.549003 mm (-4.3%, 292.8 waves
        # PV).  The index-threaded form below matches that oracle to
        # <= 5.3e-12 on both, and to 1.8e-12 on the air control.  Note
        # the two errors are NOT a common factor -- n_obj and n_img enter
        # differently -- which is why both must be threaded.
```

### L596-599 -- `_pupil_aim` -- the exit-pupil aim point -- the `Pre-4.10 always aimed at z=0` framing

*Left in the source:* where the aim point is and what aiming at the vertex costs a mid-stop system

```text
    # from surface 0 with radius fod.ep_radius.  Pre-4.10 always aimed
    # at z=0 with the full aperture radius, so off-axis fields with a
    # mid-stop system landed at the wrong pupil position and reported
    # wrong WFE.
```

### L696-701 -- `_chief_index` -- the ALIVE constraint -- the `Pre-4.10 could pick a dead vignetted on-axis ray` framing

*Left in the source:* the selection rule, the NaN poisoning the constraint prevents, and the fallback

```text
    # 4.10: identify chief = ALIVE ray closest to (0, 0) in pupil
    # coords.  Pre-4.10 could pick a dead vignetted on-axis ray (rare
    # but possible for systems where the chief is geometrically
    # blocked), which NaN-poisoned every downstream OPL calculation
    # via opl[chief] = NaN.  Fall back to the unconstrained nearest
    # if no rays survived (caller will see the all-NaN result anyway).
```

### L732-737 -- `_reference_sphere` -- the 1/N_chief path length -- the `Pre-4.12.0 only the chief-image landing got the factor (the v4.11.2 fix)` framing

*Left in the source:* what goes wrong if the sphere radius is left at the axial distance, and the on-axis degeneracy

```text
    # Pre-4.12.0 only the chief-image landing got the 1/N_chief factor
    # (the v4.11.2 fix); the sphere radius was left at the axial
    # ``img_d_m``, so for off-axis fields the sphere no longer passed
    # through the chief and the resulting quadratic shape error was
    # absorbed as phantom defocus by ``best_rms``.  On-axis (N=1) this
    # is a no-op.
```

### L747-752 -- `_reference_sphere` -- the folded frame -- the `Pre-fix, on an air singlet + flat fold` framing

*Left in the source:* the measured consequence of dropping ``_fold_sign`` -- a NEGATIVE sphere radius and 321 waves PV on a few-wave system -- and the guarantee that it is an IEEE no-op unfolded

```text
    # z).  Pre-fix, on an air singlet + flat fold, ``1/N_chief`` inverted
    # every arc-length factor and the sphere was centred on the wrong side:
    # ``r_sphere_m`` came back NEGATIVE (-4.288895e-02 m) and the WFE read
    # 321.00 waves PV / 100.17 waves RMS for a system that is a few waves
    # unfolded.  ``_fold_sign`` is exactly +1 for every unfolded system
    # (and every even mirror count), so all of this is an IEEE no-op there.
```

### L853-857 -- `_best_rms_defocus` -- the sphere radius -- the `pre-4.10 always used 1/img_d_m` framing

*Left in the source:* which radius the closed form needs and which branch ``1/img_d_m`` is wrong on

```text
                # 4.10: closed-form best-RMS uses the SPHERE radius
                # (which depends on sphere_tangent), not img_d_m
                # directly.  For 'exit_pupil' the sphere radius is
                # img_d_m - fod.xp_z; pre-4.10 always used 1/img_d_m
                # which is wrong for that branch.
```
