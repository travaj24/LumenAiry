<!-- lumenairy-history-doc
module: lumenairy/elements/bsdf.py
ast_sha256: cfe811076550f82fe31d6992761b5f3eed0f51f721dc4ca93bc2be249d816370
token_sha256: bd897d07a2841772569117a8c4e7e65430c7b66a2948bd1b0f2107b84921e23f
pre_relocation_lines: 810
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/bsdf.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bsdf.py` -- the clause recording what an unconsumed spec key
used to do.  The block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

What did NOT move: the alias contract and the both-spellings-is-an-error rule.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment plus a pointer to this file; those are noted per block
below as *Left in the source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L686-687 | `_reject_unknown_keys` docstring | "used to leave the corresponding parameter at its default with no diagnostic" |

---

### L686-687 -- `_reject_unknown_keys` docstring -- "used to leave the corresponding parameter at its default with no diagnostic"

*Left in the source:* the same failure as the reason for the refusal, with the worked example.  See docs/history/lumenairy.elements.bsdf.md.

```text
    A spec key that no constructor argument consumes used to leave the
    corresponding parameter at its default with no diagnostic, so a
```
