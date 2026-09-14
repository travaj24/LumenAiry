<!-- lumenairy-history-doc
module: lumenairy/propagators/asymptotic_canonical_fit.py
ast_sha256: d36204f01b8a593b77b15b74aa7896a6ca75a1c216660afef977a747526d7219
token_sha256: 27f528cd966695b45c11da59135966e5ad9344560e7e58c63a73a1363e5a7b93
pre_relocation_lines: 1354
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-B7 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, WP-A4 sec. 6 items 3-8 + VERIFY-B1 F1/F2): the Y4 fused basis evaluation and hoisted Newton factor, the aberration_tensor mode/waist caches, the S6 gate's k1-slope statistic and mean-plus-spread chart sizing, the JAX screen's chief-ray displacement term, and the S9 FFT kernel clip
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
