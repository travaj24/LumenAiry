<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/twod.py
ast_sha256: 4620d14d47bffcecfe7441dbdb41c78c427681d3b279ff9a3f6cd97515162ca8
token_sha256: 666adcd158721f2b2a4c491df99bd49d875e68de29f51231ce459fffda61ccc4
pre_relocation_lines: 2029
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/pmm/twod.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/pmm/twod.py` -- the clauses describing what the closure
predicate, the order-cap ceiling and the JAX pillar-bounds dispatch USED TO do.
Each block is reproduced **verbatim** under the source line it came from in the
pre-relocation file.

What did NOT move: the measured 10.5 % energy-loss case that motivates the
two-sided closure, the `_MAX_NODAL_DOF` sizing, the `.. note:: API change (v5.11
-> v5.12)` migration note on `pmm_efficiency_2d`, and the SEGMENT-vs-PIXEL grid
statements `tests/unit/test_audit2609_a13_staggered_cost.py` reads out of
`_cell_to_walls_tile.__doc__`.

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
| L333-336 | `_warn_closure` docstring | "This predicate was previously the passivity window" |
| L340 | `_warn_closure` docstring, the measured case | "sits inside the old window" |
| L513 | `_MAX_NODAL_DOF` | "than the old dense-kron path's 4000" |
| L764 | `_projected_ops_2d` | "the old _assemble_2d path performed" |
| L1320-1322 | `pmm_efficiency_2d`, the pillar-bounds contract | "previously the JAX branch returned first" |

---

### L333-336 -- `_warn_closure` docstring -- "This predicate was previously the passivity window"

*Left in the source:* why a passivity window cannot express a closure test here, which is the whole point of the predicate.

```text
    defect as an excess.  This predicate was previously the passivity window
    ``-tol <= tot <= 1+tol`` inherited from the siblings -- but those siblings
    never establish losslessness, so they cannot use the deficit side, whereas
    this one can and must.  The gap was not academic: the same near-singular
```

### L340 -- `_warn_closure` docstring, the measured case -- "sits inside the old window"

*Left in the source:* the measured 10.5 % loss and the window that admits it, in present tense.

```text
    warning, because 0.8953 sits inside the old window (measured: duty-0.25
```

### L513 -- `_MAX_NODAL_DOF` -- "than the old dense-kron path's 4000"

*Left in the source:* the comparison itself, without attributing it to a removed implementation.

```text
# (O(Nf^2 N) flops) -- so the ceiling is ~40x higher than the old dense-kron
```

### L764 -- `_projected_ops_2d` -- "the old _assemble_2d path performed"

*Left in the source:* what the separable form avoids, as a property of the alternative rather than of a past revision.

```text
        # OF A DIAGONAL MATRIX the old _assemble_2d path performed are never
```

### L1320-1322 -- `pmm_efficiency_2d`, the pillar-bounds contract -- "previously the JAX branch returned first"

*Left in the source:* the same silent-wrong outcome as the reason the check runs before the dispatch.

```text
    # inverted / degenerate bounds must raise here too; previously the JAX
    # branch returned first and silently built a negative-width strip (an
    # energy-conserving but geometrically WRONG answer -- the lossless trap).
```
