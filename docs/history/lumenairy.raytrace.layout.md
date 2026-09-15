<!-- lumenairy-history-doc
module: lumenairy/raytrace/layout.py
ast_sha256: 9ec337cda3725e080b5b62bd88b1ffaf80c0444a867ee37dda55d3f5fe503c31
token_sha256: ed32dcd4d1857e026a71efa41440650b25002e56d7ecee6c2234dc76c4a7232a
pre_relocation_lines: 190
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/raytrace/layout.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/layout.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks, both of the same shape: a bare built-in exception replaced by one
that names the function and the offending value.  The bare exception is what
the guard prevents, so it stayed.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L60-61 | `trace_summary units guard` | "used to surface" |
| L76-78 | `trace_summary empty-bundle guard` | "used to die here" |

---

### L60-61 -- `trace_summary units guard` -- "used to surface"

*Left in the source:* the diagnostic rule and the bare KeyError it replaces.

```text
    # function and the offending value.  A wrong ``units`` string used to
    # surface as a bare ``KeyError: 'cm'``.
```

### L76-78 -- `trace_summary empty-bundle guard` -- "used to die here"

*Left in the source:* the same rule for the empty bundle.

```text
    # S11-6d: an EMPTY bundle (n_total == 0) used to die here with a bare
    # ``ZeroDivisionError: division by zero`` from ``n_alive / n_total``,
    # naming neither the function nor the cause.
```

