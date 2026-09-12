<!-- lumenairy-history-doc
module: lumenairy/analysis/coherence.py
ast_sha256: b73d888f0f6b9b03418b78cdb71fc8077961a65d4fdd307db683e7af6c638e34
token_sha256: 318c055ce37a74011a1dba5c51c2f13690321ec2ac4a2367bcef54296f90044d
pre_relocation_lines: 223
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/coherence.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/coherence.py`.  Each block is reproduced **verbatim** under the source line it came
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
| L78-84 | `koehler_image` -- the 2-D guard | the `Pre-v4.15.5` framing |
| L102-108 | `partial_coherent_image` -- obliquity weighting | the `(the old behaviour)` framing on the small-NA limit |
| L214-220 | `mutual_coherence_1d` -- the operand order | the `pre-4.10 used rows.T.conj() @ rows` framing |

---

### L78-84 -- `koehler_image` -- the 2-D guard -- the `Pre-v4.15.5` framing

*Left in the source:* both failure modes -- including the 3-D one that returns a meaningful but WRONG N -- and the declared input kind

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Pre-v4.15.5 an MCF / 3-D
    # ensemble object failed at ``object_field.shape[0]`` indexing
    # (3-D returned a meaningful but wrong N) or attribute access
    # (MCF).  Routes both to the canonical v4.16 message via the V6
    # walker.  Input kind: 'field' (the object transmission is a
    # 2-D scalar complex field on the object plane).
```

### L102-108 -- `partial_coherent_image` -- obliquity weighting -- the `(the old behaviour)` framing on the small-NA limit

*Left in the source:* the model, why a bare count-average is wrong, and the regime where the correction actually bites

```text
            # v5.4.6 (audit P3-12): obliquity / solid-angle weighting.  A
            # bare count-average over the Cartesian (ax, ay) grid treats
            # every direction as equally bright and over-weights high-angle
            # directions.  Weight each contribution by cos(theta) (a
            # uniform-radiance / Lambertian condenser model) and normalise
            # by sum(w).  At small condenser_NA cos(theta) -> 1 (the old
            # behaviour); the correction matters near the 0.999 NA clamp.
```

### L214-220 -- `mutual_coherence_1d` -- the operand order -- the `pre-4.10 used rows.T.conj() @ rows` framing

*Left in the source:* which quantity each operand order computes, and why getting it wrong is SILENT (Hermiticity survives) but flips the off-diagonal sign for every phase-sensitive consumer

```text
    # Gamma[i, j] = < E(x_i) conj(E(x_j)) > over the ensemble.  4.10:
    # pre-4.10 used rows.T.conj() @ rows which produces
    # < conj(E(x_i)) E(x_j) > -- the complex conjugate of the
    # documented quantity.  Hermiticity is still preserved (Gamma is
    # always Hermitian), so the bug was silent, but any phase-sensitive
    # consumer (degree of coherence, Wolf-Mandel imaging) saw the
    # off-diagonals with flipped sign.
```
