<!-- lumenairy-history-doc
module: lumenairy/elements/segment_geometry.py
ast_sha256: 85027c1fbcabddfdad159b7024faa061fed78946c6ce6a16ab8ddf2732093918
token_sha256: 3c6aba27608a91d6510d99b86f9dc51554ab78a2036b7d19ff716c6a327bcbae
pre_relocation_lines: 574
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/segment_geometry.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/segment_geometry.py` -- the two P3-39 clauses recording that
a too-thin carved-side band used to be a silent no-op.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

What did NOT move: the geometric contract of `coat` / `line_interface` and the
warn-once bookkeeping.

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
| L271-275 | `SegmentStackGeometry.line_interface` docstring | "(previously a silent no-op leaving an inconsistent partial liner)" |
| L344-346 | `SegmentStackGeometry.line_interface`, the thin-band warn | "previously a silent no-op that left an inconsistent partial liner" |

---

### L271-275 -- `SegmentStackGeometry.line_interface` docstring -- "(previously a silent no-op leaving an inconsistent partial liner)"

*Left in the source:* what happens now and why a warning is the right answer, since the alternative is an inconsistent partial liner.

```text
        v5.17 (audit P3-39): when the carved-side band at a horizontal
        interface is THINNER than ``t`` the horizontal liner cannot be
        carved and is omitted there; a ``UserWarning`` is emitted
        (previously a silent no-op leaving an inconsistent partial
        liner)."""
```

### L344-346 -- `SegmentStackGeometry.line_interface`, the thin-band warn -- "previously a silent no-op that left an inconsistent partial liner"

*Left in the source:* the same outcome as the reason for the warning: vertical walls lined, this horizontal interface not.

```text
                    # below can never fire -- previously a silent no-op
                    # that left an inconsistent partial liner (vertical
                    # walls lined, this horizontal interface not).
```
