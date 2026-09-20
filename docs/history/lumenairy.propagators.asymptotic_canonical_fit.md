<!-- lumenairy-history-doc
module: lumenairy/propagators/asymptotic_canonical_fit.py
ast_sha256: 48eaa202c1dec68574e5e2111b5de8929f210f61dc0acf2da823acfdb0c2a92f
token_sha256: de32f25746dd51bf69712f361c42cf82face350fa9a0157bd7d3b96556ef5a55
pre_relocation_lines: 1354
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-B7 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, WP-A4 sec. 6 items 3-8 + VERIFY-B1 F1/F2): the Y4 fused basis evaluation and hoisted Newton factor, the aberration_tensor mode/waist caches, the S6 gate's k1-slope statistic and mean-plus-spread chart sizing, the JAX screen's chief-ray displacement term, and the S9 FFT kernel clip
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-20 -- WP-C2 round 2 (VERIFY-WP-C2 D4): this module's exported internally-tracing entry point(s) take the tracer's own sphere_normal= / renormalize= keywords (default None, which stamps nothing) and forward them verbatim to the trace call, so the pre-WP-C2 arithmetic is one keyword away; 742/742 arrays byte-identical archive to archive on both builds
-->

# Version history -- `lumenairy/propagators/asymptotic_canonical_fit.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/asymptotic_canonical_fit.py`.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

One block, a comment correcting an earlier COMMENT: the Gauss-Newton Hessian
note used to claim the dropped term is "exact at the stationary point", and
audit Y5 recorded both the retraction and the proof.  The proof is live -- it
is what tells a reader that dropping the term costs convergence RATE and not
the root -- so it stayed; the note that the comment (and its two siblings)
used to say otherwise is here.

Everything else the loose classifier flags in this module is a live guard
whose comment explains what it prevents (the auto-poly-order gate, the W6-A16
validity guards, the non-zero-initialised output array), and none of it moved.

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
| L728-732 | `solve_envelope_stationary` -- the Gauss-Newton Hessian | a comment correcting an earlier COMMENT: that this comment used to claim the dropped term is exact at the stationary point, and that its two siblings still did |

---

### L728-732 -- `solve_envelope_stationary` -- the Gauss-Newton Hessian -- a comment correcting an earlier COMMENT: that this comment used to claim the dropped term is exact at the stationary point, and that its two siblings still did

*Left in the source:* the proof that the dropped term does not vanish there, which is the part that keeps the Gauss-Newton note honest

```text
        # Y5 (audit): it is NOT "exact at the stationary point", as this
        # comment used to claim (and its two siblings still did).  At the
        # stationary point J^T(s_1 - s_src)/w_s^2 = -(v - v_c)/w_p^2,
        # which is not zero, so s_1 - s_src does not vanish there and
        # neither does the dropped term; what vanishes is the RESIDUAL.
```
