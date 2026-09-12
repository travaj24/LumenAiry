<!-- lumenairy-history-doc
module: lumenairy/elements/eme/eme_diffraction.py
ast_sha256: 8f4d17a552379a9018e09f5f37d193cf94fe21e475f7d6030579a843b5595b37
token_sha256: c19a7fc198dde6a2cfca0d38fdc929cea88b8d33411f01eea52c5899dcedf28a
pre_relocation_lines: 299
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/eme/eme_diffraction.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/eme/eme_diffraction.py` -- the W6 "used to / previously"
clauses on the backward-amplitude reference and the lossy-layer spectrum.  Each
block is reproduced **verbatim** under the source line it came from in the
pre-relocation file.

What did NOT move: the measured `cond(A)` ladder, the analytic lossy-Airy
comparison, and the statement that a REAL `eps_xy` takes the identical
byte-for-byte path.

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
| L147 | `mode_match` docstring, STABILITY | "the backward amplitudes ``c-`` used to be referenced at ``z = 0``" |
| L235-237 | `eme_diffraction_orders` docstring, LOSSY layers | "It previously took the real part" |

---

### L147 -- `mode_match` docstring, STABILITY -- "the backward amplitudes ``c-`` used to be referenced at ``z = 0``"

*Left in the source:* the same reference choice as the reason, in present tense, with the measured conditioning ladder below it untouched.

```text
    the backward amplitudes ``c-`` used to be referenced at ``z = 0``, which put
```

### L235-237 -- `eme_diffraction_orders` docstring, LOSSY layers -- "It previously took the real part"

*Left in the source:* what the complex spectrum buys, with the measured lossy-Airy comparison that follows it.

```text
    LOSSY layers (AUDIT W6 fix).  A complex ``eps_xy`` now keeps the COMPLEX
    ``qz^2`` (``ref_2d_modes(return_complex=True)``), so absorption is modelled.
    It previously took the real part, which made an absorbing slab behave as a
```
