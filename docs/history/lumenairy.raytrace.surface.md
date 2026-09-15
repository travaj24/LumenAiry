<!-- lumenairy-history-doc
module: lumenairy/raytrace/surface.py
ast_sha256: 296c0892a536d44ebea0803b96b1523870ccf553cf62c5d4b75992fed0737388
token_sha256: d65270f03a000ae51ff2382388a6231e57f9fcd0299988b61dc7116c58cadb12
pre_relocation_lines: 751
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B9 item 2: closed-form sphere normal _sphere_normal + the shared is_pure_spherical predicate; _surface_normal gains analytic_sphere= (default False, generic route unchanged)
re_recorded: 2026-09-13 -- WP-B9 item 2: closed-form sphere normal _sphere_normal + the shared _is_pure_spherical predicate; _surface_normal gains analytic_sphere= (default False, generic route unchanged)
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
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

