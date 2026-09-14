<!-- lumenairy-history-doc
module: lumenairy/elements/eme/_branch.py
ast_sha256: b5f5e0175ecca61567149ff1f73564c1fa9e72541c6ccd37ef67bc5d6dd63401
token_sha256: ea411656d86c31d12f988318afea0fa3e144c362818625a472f7c4444be2da91
pre_relocation_lines: 161
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- the on-cut band comparison and the forward selector moved to the shared lumenairy/_branchcut.py leaf; each engine keeps its own derived scale (bit-identical, WP-B11a item 1)
-->

# Version history -- `lumenairy/elements/eme/_branch.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/eme/_branch.py` -- the `ROUND 2 (D13)` passage, which quotes
and refutes `cut_band`'s own earlier justifying comment.  Each block is
reproduced **verbatim** under the source line it came from in the pre-relocation
file.

What did NOT move: the dimensional argument, the measured um-vs-nm census and
the `T00` divergence it produced, and the rule that a dimensioned literal is not
allowed anywhere.

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
| L47-58 | `<module>` docstring | ROUND 2 (D13) -- the LITERAL 1.0 floor and the "``ky`` here is DIMENSIONLESS" comment that justified it, quoted and refuted |
| L84-85 | `_CUT_BAND_REL` | "what ROUND 2 changed is the SCALE it multiplies" |
| L119-120 | `cut_band` docstring | "and what ROUND 2 removed" |

---

### L47-58 -- `<module>` docstring -- ROUND 2 (D13) -- the LITERAL 1.0 floor and the "``ky`` here is DIMENSIONLESS" comment that justified it, quoted and refuted

*Left in the source:* the dimensional argument itself, which is what makes a literal floor wrong, and the failure mode it produces.

```text
ROUND 2 (D13) -- THE FLOOR WAS UNIT-DEPENDENT AND IS NOW ``k0``.  As first
built, :func:`cut_band` floored the spectrum scale at a LITERAL 1.0, justified
in its own comment by "``ky`` here is DIMENSIONLESS (the EME modules work in
``k0``-normalized units)".  That is false.  ``strip_x_modes`` assembles
``d2/dx2 + eps k0^2`` on a spacing ``Lx / Nx``, so ``lam`` carries
1/length^2 and ``ky`` carries 1/length; ``mode_match`` forms
``exp(i qz depth)``, which is dimensionless only because ``qz`` is 1/length.
``k0`` is a FREE ARGUMENT carrying units, not a normalisation.  A literal 1.0
therefore engages whenever ``max|ky| < 1`` in the caller's units -- the ordinary
case for a sub-micron cell written in nanometres -- and the BRANCH DECISION
moves with the unit system, which the pre-5.45.1 exact-zero pin (having no
scale at all) did not.
```

### L84-85 -- `_CUT_BAND_REL` -- "what ROUND 2 changed is the SCALE it multiplies"

*Left in the source:* which scale the factor multiplies, stated directly.

```text
#: since it was written; what ROUND 2 changed is the SCALE it multiplies (see
#: the module docstring's D13 paragraph and :func:`cut_band`).
```

### L119-120 -- `cut_band` docstring -- "and what ROUND 2 removed"

*Left in the source:* the prohibition itself and the measured cost of breaking it.

```text
    NOT allowed on any path, and what ROUND 2 removed, is a DIMENSIONED LITERAL
    (the floor was 1.0): that made the band 5.2x wider in physical terms for a
```
