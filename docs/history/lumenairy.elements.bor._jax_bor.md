<!-- lumenairy-history-doc
module: lumenairy/elements/bor/_jax_bor.py
ast_sha256: 5379d6ad83a711ff9116676b4e012cbdda9192a16f84d99e1ded6d89396cb7c9
token_sha256: adb26cd73305d5b02a328f92e19bb71611c023541b9736bb626a0f1c05a600b4
pre_relocation_lines: 206
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/bor/_jax_bor.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/_jax_bor.py` -- the clause naming the superseded 0.05
angular cutoff.  The block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

What did NOT move: the real-axis floor, the S1-16 shared-core statement and the
deliberate absence of the `reldiv` leg.

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
| L185-187 | `_mask`, the propagating floor | "The old 0.05 was an angular cutoff" |

---

### L185-187 -- `_mask`, the propagating floor -- "The old 0.05 was an angular cutoff"

*Left in the source:* what the floor is and what it guards, with the ANGULAR alternative named as the thing it is not.

```text
        # 1e-6 (guards only the q ~ 0 degenerate point).  The old 0.05 was an
        # angular cutoff that dropped genuinely propagating near-grazing
        # orders.
```
