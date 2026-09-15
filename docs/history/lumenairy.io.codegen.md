<!-- lumenairy-history-doc
module: lumenairy/io/codegen.py
ast_sha256: 42cd063faa6ad18a81e035a4589414bf9e4a01f00e6c3b878d07663cd7dd814f
token_sha256: 0af715c0e4b3597f00611d9c3645ecb6f87bc049ac4607c2b9dabfb4ead1a036
pre_relocation_lines: 1221
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/io/codegen.py`

This file holds the version-history narrative that used to live in
`lumenairy/io/codegen.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks.  The escaping rule is a security invariant -- every string this
module interpolates comes from a file the user did not write -- so the whole
argument stayed in the source, re-stated as what an unescaped interpolation
does rather than as what one release did.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L52-55 | `<module> escaping note` | the "Pre-fix those were interpolated" framing |
| L195-195 | `generate_simulation_script` | the release note on the removed 1.31e-6 default |

---

### L52-55 -- `<module> escaping note` -- the "Pre-fix those were interpolated" framing

*Left in the source:* the whole injection argument, including the worked payload.

```text
# Pre-fix those were interpolated into CODE positions of the generated
# script -- ``la.GLASS_REGISTRY['{g}'] = ...``, ``print("Applying {label}
# ...")``, ``print('Running: {sys_name}')`` -- with no escaping, so a
# whitespace-free ``GLAS`` token such as ``X'];<payload>;#`` became live code
```

### L195-195 -- `generate_simulation_script` -- the release note on the removed 1.31e-6 default

*Left in the source:* the live rule: neither supplied means ValueError, not a silent NIR conversion.

```text
        (v4.13.0 onward; pre-v4.13.0 this defaulted to 1.31e-6).
```

