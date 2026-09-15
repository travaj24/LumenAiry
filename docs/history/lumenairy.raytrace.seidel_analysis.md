<!-- lumenairy-history-doc
module: lumenairy/raytrace/seidel_analysis.py
ast_sha256: e23d3ab2c0ed41719d8e1f8a3da3f3615acc645d82f0b57e25fc7896bde27061
token_sha256: 662f13f5d518cbe015bd0f087512e90807e48a884eb20fb56f37756998ee9e6b
pre_relocation_lines: 405
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/raytrace/seidel_analysis.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/seidel_analysis.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

One block.  The sign convention and its exact-trace oracle stayed -- they are
what a caller adding `W` to a pupil phase depends on -- and so did the
consumer-impact list, because it is still the answer to "who cares about the
sign?".  The "pre-fix the expansion was composed out of ..." framing moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L246-252 | `seidel_wavefront_expansion` | the "pre-fix the expansion was composed" framing and "unaffected by the fix" |

---

### L246-252 -- `seidel_wavefront_expansion` -- the "pre-fix the expansion was composed" framing and "unaffected by the fix"

*Left in the source:* the sign convention, the exact-trace oracle numbers and the full list of consumers the sign matters to.

```text
    therefore has ``W(rho = 1) < 0``.  R-2
    (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): pre-fix the expansion was
    composed directly out of the ``-S_Welford`` sums and so returned
    ``-W``; measured against an exact-trace wavefront oracle the ratio
    was ``-0.9975 ... -0.9998`` on four singlets over ``rho in
    [0.3, 1]``, and ``-1.000`` term by term for the ``rho^2``, ``rho^3``
    and ``rho^4`` terms.  Magnitude-only consumers (RMS / PV WFE, Strehl
```

