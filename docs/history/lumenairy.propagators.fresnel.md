<!-- lumenairy-history-doc
module: lumenairy/propagators/fresnel.py
ast_sha256: f91e2a541c81d45798c2a0edb4abe725282fcc138e7a76ffd7296f0b94f77753
token_sha256: 3fc098d5b1c43a290551f72dc2abd0b93e91bf41c3210c1584531c74303baeb2
pre_relocation_lines: 593
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/fresnel.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/fresnel.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

One block, and only its framing moved.  The odd-N half-sample derivation is
live -- it is the algebra the implementation follows -- and its measurement
(intensity centroid -0.5000 px, rel err 1.7e-1 against `fresnel_propagate_mft`
at N=257, against 3.2e-14 at even N) is what shows the correction matters.
The source now attributes that error to the ABSENCE of the correction rather
than to a release that lacked it.

The dtype-resolution and copy-avoidance comments further down carry the same
`Pre-fix` phrasing but describe hazards that are still reachable, so they were
left as live why-comments.

Nothing the interpreter executes changed in the move.  The header above
records the SHA-256 of (a) the module's AST with every docstring removed and
source positions ignored, and (b) its `tokenize` stream reduced to
NAME/OP/NUMBER/STRING with comments and docstrings dropped -- both taken from
the file as it stood BEFORE the relocation.
`tests/unit/test_audit2609_a17_history_relocation.py` re-computes both from
the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L110-114 | `fresnel_propagate` -- the odd-N anchor | the `pre-fix the returned field was` framing |

---

### L110-114 -- `fresnel_propagate` -- the odd-N anchor -- the `pre-fix the returned field was` framing

*Left in the source:* the whole derivation and its measurement, re-attributed to the absence of the correction

```text
    planes, which is not a mere relabelling: pre-fix the returned field
    was half an output pixel off its own grid (measured intensity
    centroid -0.5000 px) with a residual phase ramp, rel err 1.7e-1 vs
    ``fresnel_propagate_mft`` on the same output grid at N=257
    (even N: 3.2e-14).
```
