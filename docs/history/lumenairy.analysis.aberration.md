<!-- lumenairy-history-doc
module: lumenairy/analysis/aberration.py
ast_sha256: d1ef9c09b05b39e848fca5654c1f6b51f8a57b797e88bd4f9c52f4b1c22d08cf
token_sha256: 6a901ad6ecdd71ec71cd34f97ac806f8edc4bc36849a2f7782a151b11652725f
pre_relocation_lines: 692
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-20 -- WP-C2 round 2 (VERIFY-WP-C2 D4): this module's exported internally-tracing entry point(s) take the tracer's own sphere_normal= / renormalize= keywords (default None, which stamps nothing) and forward them verbatim to the trace call, so the pre-WP-C2 arithmetic is one keyword away; 742/742 arrays byte-identical archive to archive on both builds
-->

# Version history -- `lumenairy/analysis/aberration.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/aberration.py`.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

Every block here is the same shape, and it is the shape the WP-A17 SWEEP-1
follow-up pass was asked to close: a **live guard whose comment explained
itself by naming the release that added it** ("Pre-fix X happened", "Pre-4.12
the dispatcher only passed ...").  The hazard X is still reachable -- the guard
is the only thing preventing it -- so the source now states X in the present
tense, as what goes wrong WITHOUT the guard, together with every measurement
that sizes it.  What moved is the release attribution and the
"bit-identical to pre-fix" reassurance that travelled with it.

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
| L125-127 | `aberration_summary` -- `_maybe_warn_glass` | the `previously got buried in notes` framing |

---

### L125-127 -- `aberration_summary` -- `_maybe_warn_glass` -- the `previously got buried in notes` framing

*Left in the source:* what the helper does and the failure it prevents -- a system with an unknown glass reading as diffraction-limited

```text
    # Helper: glass-catalog failures previously got buried in `notes`
    # while Seidel returned zeros -- making a system with an unknown
    # glass look "diffraction-limited".  Bubble them up as warnings.
```
