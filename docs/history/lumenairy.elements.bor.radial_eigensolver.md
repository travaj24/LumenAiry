<!-- lumenairy-history-doc
module: lumenairy/elements/bor/radial_eigensolver.py
ast_sha256: 660b4065f1f6116e5e64d201f99c673d8acfdd336a6c6a2cc993beba0738db24
token_sha256: 0fcebde377c6d4ff1a4d553978ad5368c7676cbe3cf584f468f45233c28fd796
pre_relocation_lines: 178
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/bor/radial_eigensolver.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/radial_eigensolver.py` -- the W6-B12 clause recording
what ``R <= 0`` and ``n_el < 1`` used to do.  The block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

What did NOT move: the sign-flip argument (Jacobian, measure and stiffness flip
together, so ``R = -1`` returns the ``|R| = 1`` spectrum), which is the reason
the guard cannot be left to a later numerical failure.

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
| L101-106 | `radial_modes` docstring, Domain validation | "``R <= 0`` used to be accepted ... died with a bare IndexError" |

---

### L101-106 -- `radial_modes` docstring, Domain validation -- "``R <= 0`` used to be accepted ... died with a bare IndexError"

*Left in the source:* both failure modes as the reason for the guard, and the sibling it matches.

```text
    Domain validation (audit W6-B12): ``R <= 0`` used to be accepted -- the
    element Jacobian, the ``r`` measure and the ``1/r`` stiffness all flip sign
    together, so ``R = -1`` silently returned the ``|R| = 1`` spectrum -- and
    ``n_el < 1`` died with a bare ``IndexError`` from the local->global map.
    ``BORStack`` has guarded its own ``Rbig > 0`` since P3-10; this is the
    sibling gap.
```
