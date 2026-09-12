<!-- lumenairy-history-doc
module: lumenairy/elements/bor/coupled_radial_eigensolver.py
ast_sha256: 4970550710109548a8e60c6d5adbcc2b819f058a25a881ad2101999ce8d7b960
token_sha256: a7f5e14b4d04b62b100f988c72d7a9a2e5503087757427e2bb1467d3101a918a
pre_relocation_lines: 716
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/bor/coupled_radial_eigensolver.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/coupled_radial_eigensolver.py` -- the W6-B3 clause
recording what an unrecognised `wall` value used to do.  The block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

What did NOT move: the basis defaults, the `reldiv` tagging contract and the
measured divergence populations.

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
| L158-159 | `_check_wall` docstring | "An unrecognized value used to fall through to ``'natural'`` silently" |

---

### L158-159 -- `_check_wall` docstring -- "An unrecognized value used to fall through to ``'natural'`` silently"

*Left in the source:* the same failure as the reason for the refusal -- a typo buying open-boundary physics -- plus the pointer to this file.

```text
    Dirichlet wall on the staggered path.  An unrecognized value used to fall
    through to ``'natural'`` silently -- a typo bought open-boundary physics."""
```
