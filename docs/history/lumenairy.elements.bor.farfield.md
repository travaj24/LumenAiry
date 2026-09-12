<!-- lumenairy-history-doc
module: lumenairy/elements/bor/farfield.py
ast_sha256: a5f1b4a82b63c19d8e623b1d69cb57bc1f42db296a0b71d6433c3de75e9c1a93
token_sha256: 353b1cd2353ef86c8717ea6f6131c8421ba4e9e9d432c595aa157323908acb63
pre_relocation_lines: 136
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/bor/farfield.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/farfield.py` -- the W6-B10 clause recording what a
complex ``eps`` used to do to the propagating mask and to ``theta``.  The block
is reproduced **verbatim** under the source line it came from in the
pre-relocation file.

What did NOT move: the convention statement (the angle is taken in
``Re sqrt(eps)``, matching ``BORStack.solve``).

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
| L96-101 | `far_field_angles` docstring | W6-B10 -- "a complex ``eps`` used to make ``s`` complex" |

---

### L96-101 -- `far_field_angles` docstring -- W6-B10 -- "a complex ``eps`` used to make ``s`` complex"

*Left in the source:* the whole failure chain as the reason the real part is taken, since ``order_power_fractions`` passes ``eps`` straight through.

```text
    ``angles`` (``eps_sup.real``).  Audit W6-B10: a complex ``eps`` used to make
    ``s`` complex, so the propagating mask fell back to numpy's LEXICOGRAPHIC
    complex comparison and ``theta`` was filled from a complex ``arcsin`` whose
    imaginary part was dropped with only a ``ComplexWarning`` -- silently wrong
    angles for any lossy half-space (``order_power_fractions`` passes ``eps``
    straight through).
```
