<!-- lumenairy-history-doc
module: lumenairy/raytrace/surface.py
ast_sha256: e545dade6b06e974a3cf511197608b08abc7daec8e797524c44b29b1334cac47
token_sha256: 2404a25514aad91db31ee9801dcd3aa5470917239fb9fe84c7c5bfb3f22ca7b9
pre_relocation_lines: 751
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/surface.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/surface.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks.  The `clone`-field hazard is live -- a hand-rolled field list
still drops the coord-break and world-frame blocks if anyone writes one --
so it stayed, re-stated in the present tense.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L11-12 | `<module> docstring` | the split-provenance claim |
| L681-684 | `Surface.clone` | the release/audit tag and the "pre-fix" framing |

---

### L11-12 -- `<module> docstring` -- the split-provenance claim

*Left in the source:* the bit-for-bit statement.

```text
No physics change: contents are bit-for-bit copies of the original
implementations.
```

### L681-684 -- `Surface.clone` -- the release/audit tag and the "pre-fix" framing

*Left in the source:* the hazard, which is why the clone must enumerate fields from the dataclass.

```text
    v5.17.1 (audit P3-60): pre-fix the hand-rolled field list dropped
    the coord-break and world-frame blocks, so a cloned coord-break
    Surface silently became a regular flat refracting surface and a
    cloned world-frame surface lost its frame.
```

