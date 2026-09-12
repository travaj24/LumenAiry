<!-- lumenairy-history-doc
module: lumenairy/analysis/interferometry.py
ast_sha256: 0ed47b7c6111489ae64d46b9932ddc551ccb7e6404f25c090ac09732b0678045
token_sha256: 9fb9f86f5caeaf9270e758d3ba721bbadcc18b875b94d460541c6b3748efbf15
pre_relocation_lines: 246
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/interferometry.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/interferometry.py`.  Each block is reproduced **verbatim** under the source line it came
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
| L81-87 | `simulate_fringes` -- the fringe model | the `Pre-4.10 used background + 0.5 * visibility * cos(phase)` framing |
| L149-158 | `phase_shifting_interferometry` -- the LSQ estimator | the `The previous correlation estimator ... the old form was biased` framing |

---

### L81-87 -- `simulate_fringes` -- the fringe model -- the `Pre-4.10 used background + 0.5 * visibility * cos(phase)` framing

*Left in the source:* the model, the contrast it delivers, and the alternative form that breaks the round-trip claim

```text
    # 4.10: classic Michelson fringe is
    #   I = background * (1 + visibility * cos(phase))
    # which produces Michelson contrast V = (Imax - Imin) / (Imax + Imin)
    # = visibility (matching the kwarg semantics).  Pre-4.10 used
    # `background + 0.5 * visibility * cos(phase)`, which produced
    # contrast 0.5 even with visibility=1, breaking the docstring's
    # round-trip claim.
```

### L149-158 -- `phase_shifting_interferometry` -- the LSQ estimator -- the `The previous correlation estimator ... the old form was biased` framing

*Left in the source:* the model, the design matrix, the exact condition under which the correlation estimator coincides with it, and the fact that equispaced callers get the same answer

```text
    # v5.4.6 (audit F-13): GENERAL least-squares extraction valid for
    # ARBITRARY (non-equispaced) shifts.  Model each frame as
    #   I_k = a + A*cos(s_k) + B*sin(s_k),  A = b*cos(phi), B = b*sin(phi)
    # and solve the linear LSQ for (a, A, B) per pixel via the design
    # matrix S = [1, cos(s), sin(s)].  The previous correlation estimator
    # atan2(sum(I*sin), sum(I*cos)) is the LSQ solution ONLY when the S
    # columns are orthogonal (equispaced full-period shifts); for that
    # case S^T S = diag(n, n/2, n/2) and this reduces to exactly the old
    # result (so equispaced callers are bit-preserved), but for arbitrary
    # shifts the old form was biased.
```
