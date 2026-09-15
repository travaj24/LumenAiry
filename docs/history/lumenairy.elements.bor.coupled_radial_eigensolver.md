<!-- lumenairy-history-doc
module: lumenairy/elements/bor/coupled_radial_eigensolver.py
ast_sha256: 4970550710109548a8e60c6d5adbcc2b819f058a25a881ad2101999ce8d7b960
token_sha256: e1f9448017be5b5e0134e70fca30832ecc75de904a04c8b26a5bf7a774a4a6b0
pre_relocation_lines: 716
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
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
