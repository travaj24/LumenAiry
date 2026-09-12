<!-- lumenairy-history-doc
module: lumenairy/analysis/coronagraph.py
ast_sha256: 4b42c372c877bc466ca4d20a69b10ba89ca2a2dc92087fdb04a9ada8d6fd8c42
token_sha256: cc16013d5e4249cef0c4052694b7d39868b1031b65cdfc86533b0971554a3719
pre_relocation_lines: 198
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/coronagraph.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/coronagraph.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

One block: a comment correcting an earlier COMMENT.  The `metric` parameter's
description recorded that the docstring used to mis-describe `'rms'` as the
1-sigma speckle-noise metric.  The correction itself is live -- `'rms'`
includes any non-zero residual bias, which is exactly what a caller choosing
between `'rms'` and `'std'` needs -- so the source now states it as a property
of `'rms'` rather than as a retraction.

The `pix_per_lam_over_D = N` paragraph did NOT move: that assumption is still
reachable (it is the documented fallback when `pupil_diameter_m` is omitted,
with a `RuntimeWarning`), so describing it and its failure mode is a live
contract.

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
| L79-81 | `contrast_curve` -- `metric` | a comment correcting an earlier COMMENT: that the docstring used to mis-describe `'rms'` as the 1-sigma metric |

---

### L79-81 -- `contrast_curve` -- `metric` -- a comment correcting an earlier COMMENT: that the docstring used to mis-describe `'rms'` as the 1-sigma metric

*Left in the source:* the live distinction between the two metrics, restated as a property of ``'rms'``

```text
        speckle-noise floor metric.  4.10: pre-4.10 mis-described
        ``'rms'`` as the 1-sigma metric (it includes any non-zero
        residual bias).
```
