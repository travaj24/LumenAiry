<!-- lumenairy-history-doc
module: lumenairy/elements/_berreman_jax.py
ast_sha256: 627458e44b53c31ee8bbd8e92622811dfdfbda678a5335172b033035aaa6c9a3
token_sha256: 64653d665913a043977de506be7a82a9a59f61e12693ca51310545d19dbbcdc0
pre_relocation_lines: 592
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/_berreman_jax.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/_berreman_jax.py` -- the clauses recording where the two
partitions used to fork and what the JAX path returned before the gauge fix.
Each block is reproduced **verbatim** under the source line it came from in the
pre-relocation file.

What did NOT move: the measured twin divergence (T = 0 against the NumPy
cascade's T = 0.988) and the statement that the two paths now partition modes
byte-for-byte.

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
| L71-72 | `_layer_modes_jax` docstring | "where they previously forked -- numpy ranked by decay" |
| L452-454 | `_farfield_generalized_jax`, the gauge bridge | "so pre-fix the JAX path returned T = 0" |

---

### L71-72 -- `_layer_modes_jax` docstring -- "where they previously forked -- numpy ranked by decay"

*Left in the source:* the one partition rule that would fork, as the reason both paths share this one.

```text
    modes byte-for-byte in every case (including degenerate bianisotropic
    inputs, where they previously forked -- numpy ranked by decay)."""
```

### L452-454 -- `_farfield_generalized_jax`, the gauge bridge -- "so pre-fix the JAX path returned T = 0"

*Left in the source:* the measured twin divergence as the reason the conjugation is here, in present tense.

```text
    # routes here at EVERY incidence (no obliqueness test), so pre-fix the JAX
    # path returned T = 0 on n_sub = 1.5+0.3j even at NORMAL incidence, where the
    # NumPy native cascade gives T = 0.988 (a 0.988 twin divergence).
```
