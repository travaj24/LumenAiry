<!-- lumenairy-history-doc
module: lumenairy/analysis/strehl.py
ast_sha256: 2d243f0c01da7bdb4b0b0cfff3efc9e0d677829ab2c4b38b601352d522e9ceca
token_sha256: 7a2f5316b75ef0658ec60d2b867bc863f70e5c7859d43e4831639a09f9e21594
pre_relocation_lines: 557
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/strehl.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/strehl.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

Two blocks, both `_check_2d_scalar_field` guard comments.

This module also carries one or more `_check_2d_scalar_field` guard comments
written as "Previously an MCF / 3-D ensemble input failed at X".  What each
guard PREVENTS is live -- a 3-D ensemble does not error, it produces a wrong
answer -- so those sentences were rewritten in the present tense rather than
moved, and the block below records the original wording.

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
| L75-80 | `strehl_ratio` -- the 2-D guards | the `Previously` framing |
| L271-278 | `coupling_efficiency` -- the 2-D guards | the `Previously` framing |

---

### L75-80 -- `strehl_ratio` -- the 2-D guards -- the `Previously` framing

*Left in the source:* what the guards prevent on both arguments and the declared input kind

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guards via the shared
    # ``_check_2d_scalar_field`` helper on BOTH input fields.
    # Previously an MCF / 3-D ensemble input (either ``E`` or
    # ``E_ref``) failed downstream at ``xp.abs(...)`` with an
    # unhelpful Python TypeError.  Input kind: 'field' (both args
    # are 2-D scalar complex amplitudes).
```

### L271-278 -- `coupling_efficiency` -- the 2-D guards -- the `Previously` framing

*Left in the source:* both failure modes -- including the silently wrong 3-D overlap -- and the declared input kind

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guards via the shared
    # ``_check_2d_scalar_field`` helper on both fields.  Previously
    # an MCF / 3-D ensemble input failed at the ``.shape`` attribute
    # access (for MCF) or produced a wrong (3-D) overlap (for an
    # ensemble).  The V6 walker discovers this entry via the first-
    # positional-name ``E``; the inline guard routes both failure
    # modes to the canonical v4.16 message.  Input kind: 'field'
    # (both args are 2-D scalar complex amplitudes).
```
