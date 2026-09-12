<!-- lumenairy-history-doc
module: lumenairy/optimize/multi_objective.py
ast_sha256: d2d728362820929922a0d49c5ba884d3eb458cb29c93553613011aaa13fe2116
token_sha256: cbeeaf989d1e374dfde2da370901af0a6d7fc66e59302f03e052dae060821fc3
pre_relocation_lines: 450
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/optimize/multi_objective.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/multi_objective.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks, both the same shape: a docstring promise that the code did not
keep, and the release that closed the gap.  The PROMISE is live and stayed
in the source; the "the check did not exist" clause moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L158-160 | `design_optimize_multi_objective` | the pre-v5.46 "only n_params was read and the documented check did not exist" clause |
| L206-210 | `design_optimize_multi_objective` | the "pre-v5.46 this returned" framing |
| L260-265 | `design_optimize_multi_objective` | the "but the check did not exist" clause |

---

### L158-160 -- `design_optimize_multi_objective` -- the pre-v5.46 "only n_params was read and the documented check did not exist" clause

*Left in the source:* what ``x0`` is used for and the warning it raises.

```text
        an ``x0`` outside the box raises a :class:`UserWarning` (I8;
        pre-v5.46 only ``n_params`` was read and the documented check did
        not exist).
```

### L206-210 -- `design_optimize_multi_objective` -- the "pre-v5.46 this returned" framing

*Left in the source:* the whole failure mode as what happens WITHOUT the guard, including the ``np.asarray(None)`` -> ``array(nan)`` mechanism that makes it silent.

```text
        sets ``Result.X`` to ``None`` and pre-v5.46 this returned a 0-d
        NaN array AS the Pareto front (``np.asarray(None, np.float64)``
        is ``array(nan)``, ``ndim == 0``, so the 1-D normalisation guard
        never fired), or raised ``IndexError`` when ``progress`` was
        supplied (I7).
```

### L260-265 -- `design_optimize_multi_objective` -- the "but the check did not exist" clause

*Left in the source:* the docstring promise this code keeps, and the reason the warning is a warning rather than an error.

```text
    # I8 (AUDIT_ADVERSARIAL_EXHAUSTIVE 2026-09-11): the docstring says ``x0``
    # is "used to infer n_params and as a sanity check against bounds", but
    # the check did not exist -- only ``n_params`` was read.  NSGA-II samples
    # its own population, so an out-of-box ``x0`` does not break the run; it
    # does mean the caller's starting design is outside the box they think
    # they are searching, which is worth saying once.
```

