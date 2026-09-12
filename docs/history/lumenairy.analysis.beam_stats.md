<!-- lumenairy-history-doc
module: lumenairy/analysis/beam_stats.py
ast_sha256: 456c3f8947d79f92503122d1d48bd31a32d7a13d3b56aaa8674708cffa77f30c
token_sha256: 36213881473012758847f462ccbf3017d519e949e046507ab4bda241ab5d800e
pre_relocation_lines: 772
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/beam_stats.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/beam_stats.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

Three blocks, none of them large.  One records that `_xp_of` used to be a
4-line wrapper duplicated across five files; the other two are
`_check_2d_scalar_field` guard comments.

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
| L35-40 | `<module>` -- the `_xp_of` alias | that this used to be a 4-line wrapper duplicated in five files |
| L355-363 | `beam_d4sigma` -- the 2-D guard | the `Previously` framing |
| L544-550 | `M2` -- the 2-D guard | the `Previously` framing |

---

### L35-40 -- `<module>` -- the `_xp_of` alias -- that this used to be a 4-line wrapper duplicated in five files

*Left in the source:* what the alias is and why it exists

```text
# v5.2 (ROADMAP "Duplicate `_xp_of`" cleanup):  this used to be a
# 4-line wrapper duplicated in 5 files (elements/elements.py,
# elements/freeform.py, analysis/beam_stats.py, analysis/strehl.py,
# analysis/psf_mtf_otf.py).  Consolidated to the canonical backend
# helper; the underscore-prefixed alias preserves the existing
# in-module references without touching call sites.
```

### L355-363 -- `beam_d4sigma` -- the 2-D guard -- the `Previously` framing

*Left in the source:* both failure modes -- the MCF TypeError and, worse, the silently wrong 3-D broadcast variance -- and the declared input kind

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously a
    # ``PartialCoherenceMCF`` input failed downstream at
    # ``np.abs(E)`` with ``TypeError: bad operand type for abs()``;
    # a 3-D ensemble would have produced a wrong (3-D) variance
    # estimate via NumPy broadcasting.  The V6 walker now discovers
    # this entry via first-positional-name ``E``; the inline guard
    # routes both failure modes to the canonical v4.16 message.
    # Input kind: 'field' (2-D scalar complex amplitude).
```

### L544-550 -- `M2` -- the 2-D guard -- the `Previously` framing

*Left in the source:* both failure modes and the declared input kind

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously an MCF / 3-D
    # ensemble input failed at ``E.shape`` unpacking with
    # ``ValueError: too many values to unpack`` (for 3-D) or at the
    # ``.shape`` attribute access (for MCF).  Routes both failure
    # modes to the canonical v4.16 message via the V6 walker.
    # Input kind: 'field'.
```
