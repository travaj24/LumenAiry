<!-- lumenairy-history-doc
module: lumenairy/cache.py
ast_sha256: a9893b7c5a149ffc0ebeed2824a6bb977bc61d35d4c3297fc54aa7175beeb9f5
token_sha256: fbd00cd03e208445d633a192a6028aa41f6372799813c4bbe0696a968e618653
pre_relocation_lines: 676
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/cache.py`

This file holds the version-history narrative that used to live in
`lumenairy/cache.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

One block, and it is the clearest measured-derivation case in the sweep: the
two accounting errors `deep_nbytes` must not re-introduce, one in each
direction, with the numbers that show why the fail-safe direction is the safe
one.  The whole argument stayed in the source; only the release tag and the
past tense moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L267-275 | `deep_nbytes` | the release/audit tag and the "used to be wrong" framing |

---

### L267-275 -- `deep_nbytes` -- the release/audit tag and the "used to be wrong" framing

*Left in the source:* both failure directions with their measured magnitudes, and the fail-safe argument for the residual under-count.

```text
    v5.29.1 (audit A-5): both halves of this used to be wrong in
    OPPOSITE directions.  A view was charged its slice size (measured: 16 B
    for a 16-byte window on a 4 MiB base), so a view-heavy cache under a
    1 MiB ceiling accounted 256 B while genuinely retaining 64 MiB -- 64x
    over budget, and eviction never fired.  Meanwhile repeated arrays were
    double-counted because the ``nbytes`` shortcut returned before the
    ``_seen`` check (``deep_nbytes((a, a))`` charged 8000 B twice).
    Charging the base buffer once per unique owner fixes the dangerous
    direction without re-introducing the double count.  Residual
```

