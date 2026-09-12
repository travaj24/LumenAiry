<!-- lumenairy-history-doc
module: lumenairy/elements/bor/zcascade.py
ast_sha256: 93dafad6ad97775ca9da0b9aa8cb1a839982a0feea73a4e830de4e301e97994b
token_sha256: 76e5b104ff674104fb4d735f5145a40ecbd7ae8104e3587f2dbe260200dc6532
pre_relocation_lines: 279
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/bor/zcascade.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/zcascade.py` -- the S1-18 clause on the duplicated
eigensolve and the W6-B3/B4 clause on the silently-ignored `wall` value.  Each
block is reproduced **verbatim** under the source line it came from in the
pre-relocation file.

What did NOT move: the dedupe contract (the two nodal paths assemble
byte-identical `K`/`B` and `reldiv` is q-orientation invariant) and the
staggered-path rejection rule.

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
| L133-135 | `zcascade_modes` docstring | S1-18 -- that ``build_layer(basis='nodal')`` used to run a second, byte-identical eigensolve for this tag |
| L149-154 | `zcascade_modes` docstring, the ``wall`` note | "used to fall through to the leaky ``'natural'`` wall silently" |

---

### L133-135 -- `zcascade_modes` docstring -- S1-18 -- that ``build_layer(basis='nodal')`` used to run a second, byte-identical eigensolve for this tag

*Left in the source:* what harvesting the tag here buys and why it is safe, which is the contract.

```text
    S1-18: ``bor_solve.build_layer(basis='nodal')`` used to get this tag from a
    SECOND, byte-identical ``radial_coupled_modes`` eigensolve; harvesting it
    here dedupes that ~2x cost with NO change to the tag (the two nodal paths
```

### L149-154 -- `zcascade_modes` docstring, the ``wall`` note -- "used to fall through to the leaky ``'natural'`` wall silently"

*Left in the source:* both rules -- validation and the staggered-path rejection -- with the measured bit-identity that makes silent acceptance dangerous.

```text
    .. note:: (audit W6-B3/B4) ``wall`` is validated -- an unrecognized value
       used to fall through to the leaky ``'natural'`` wall silently -- and the
       staggered path REJECTS the nodal-only ``R_pml`` / ``wall='natural'``
       rather than ignoring them (measured bit-identical output, so a caller
       asking for an open radial boundary got the closed Dirichlet wall with no
       signal).  ``wall=None`` means "this basis's default": ``'natural'``
```
