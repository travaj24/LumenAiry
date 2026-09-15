<!-- lumenairy-history-doc
module: lumenairy/_cache_registry.py
ast_sha256: eb9e6e03f453f48298d048f643cff0340d6795c35b870a8efa715ac57b200ffc
token_sha256: a9722cc34783cbd489bab60fe64e54f7439943733c59fe64ceff01c538f20e69
pre_relocation_lines: 292
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/_cache_registry.py`

This file holds the version-history narrative that used to live in
`lumenairy/_cache_registry.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks, both describing the "fix N, miss N+1" meta-pattern this registry
exists to retire.  The pattern is live -- it is what a hand-threaded
lazy-import block would bring back -- so it stayed in the present tense; the
per-release count of how many caches had accumulated moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L10-13 | `<module> docstring` | the "Pre-v4.16" framing and the per-release cache count |
| L125-126 | `register_cache_clearer` | the "pre-v4.16" framing |

---

### L10-13 -- `<module> docstring` -- the "Pre-v4.16" framing and the per-release cache count

*Left in the source:* the meta-pattern and the threading burden it removes.

```text
pattern in the cache-clear domain.  Pre-v4.16 every new cache had to
remember to thread a new lazy-import + try/except block into
``clear_asm_caches``; v4.14.3 added the 8th cache
(``_lg_polynomial_items``) and the v4.14.2 audit found the meta-
```

### L125-126 -- `register_cache_clearer` -- the "pre-v4.16" framing

*Left in the source:* what the registration replaces.

```text
    block that pre-v4.16 ``clear_asm_caches`` accumulated as new
    caches were added.
```

