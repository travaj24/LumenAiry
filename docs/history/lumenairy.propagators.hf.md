<!-- lumenairy-history-doc
module: lumenairy/propagators/hf.py
ast_sha256: f6d5d2e97c686e003e8c57c46c87f91ebdb5167c08e328b44b6bc83624447026
token_sha256: 75d2e2418b9bffd92a1707f741cd90cbf2e649dd4a5bdec9c680f16695283f6e
pre_relocation_lines: 1009
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/hf.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/hf.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

One block: a comment correcting an earlier COMMENT.  `propagate_hf_freespace`'s
summary line used to claim the path applies "the standard `1/(i lambda z)` Van
Vleck factor", and audit K15 replaced it with a paragraph explaining that
there is no Van Vleck factor here and naming the kernel that IS applied.  The
naming stayed -- a reader needs to know the kernel is the RS-I Green's
function, whose leading term is `cos(theta)/(i lambda r)` -- while the record
of what the line used to say is here.

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
| L249-254 | `propagate_hf_freespace` | a comment correcting an earlier COMMENT: the pre-v5.46 summary line's `standard 1/(i lambda z) Van Vleck factor` claim |

---

### L249-254 -- `propagate_hf_freespace` -- a comment correcting an earlier COMMENT: the pre-v5.46 summary line's `standard 1/(i lambda z) Van Vleck factor` claim

*Left in the source:* the kernel that IS applied and its leading term

```text
    K15 (audit 2026-09-11): the pre-v5.46 summary line claimed "the
    standard ``1/(i lambda z)`` Van Vleck factor".  There is no Van
    Vleck factor on this path, and the kernel actually applied is the
    RS-I Green's function ``(z/(2 pi r^2))(1/r - ik) exp(ikr)``, whose
    leading term is ``cos(theta)/(i lambda r)`` -- not
    ``1/(i lambda z)``.
```
