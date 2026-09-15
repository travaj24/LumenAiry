<!-- lumenairy-history-doc
module: lumenairy/ui/layout_2d.py
ast_sha256: 8a9349f9b0e778cfb12c3a92ffb5db432aab947e9ff97129d824125a2be97997
token_sha256: f31a272f4efbaa7f8ea3c4c4c4519e92cd7d0e8caa5b14ed8214e914c4123363
pre_relocation_lines: 1118
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/ui/layout_2d.py`

This file holds the version-history narrative that used to live in
`lumenairy/ui/layout_2d.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Four blocks.  Each records a rendering decision and the wrong alternative
that motivated it; in every case the wrong alternative is one line away, so
the argument stayed and only the past-tense framing moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L145-148 | `Layout2DDock view size policy` | "was wrong" |
| L613-614 | `Layout2DDock emitter array` | "The old nested ix loop" |
| L639-640 | `Layout2DDock preview rays` | "The original (3.6.1 first cut)" |
| L785-786 | `Layout2DDock world-frame draw` | "previously the entire ray fan was squished" |

---

### L145-148 -- `Layout2DDock view size policy` -- "was wrong"

*Left in the source:* the policy choice and exactly what ``Ignored`` costs.

```text
        # ``Ignored`` here was wrong: it told Qt the view didn't
        # care about size, so the dock area allocated minimal
        # space and the layout never reached a useful default
        # size on first launch.
```

### L613-614 -- `Layout2DDock emitter array` -- "The old nested ix loop"

*Left in the source:* the projection argument and the duplicate-dot failure.

```text
            # drawn.  The old nested ix loop multiplied its own offset
            # by 0.0 and redrew nx identical overlapping dots.
```

### L639-640 -- `Layout2DDock preview rays` -- "The original (3.6.1 first cut)"

*Left in the source:* the direction rule and the opt-in default.

```text
        # visually unambiguous.  The original (3.6.1 first cut)
        # drew them upstream of the source which read backward.
```

### L785-786 -- `Layout2DDock world-frame draw` -- "previously the entire ray fan was squished"

*Left in the source:* the gap accounting and the visual failure it prevents.

```text
        source-to-first-surface air gap; previously the entire ray
        fan was squished into the first ~10 mm of the system.
```

