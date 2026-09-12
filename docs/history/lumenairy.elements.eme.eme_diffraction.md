<!-- lumenairy-history-doc
module: lumenairy/elements/eme/eme_diffraction.py
ast_sha256: 9acc98446b594cb8358f8b3f3671fd1b6886535e19da46d99f000006560d0467
token_sha256: fa23038e05adcedf6eb1a58707dc4fd7e42ab5464b8b873d5aeaedbfde3866f3
pre_relocation_lines: 299
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- zero-norm refusal message reworded to present tense; the retired wording is recorded in this document at L167-169 (WP-A22 follow-up)
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
| L167-169 | `mode_match`, the zero-norm refusal MESSAGE | "this used to surface as an opaque 'SVD did not converge' LinAlgError" |
| L235-237 | `eme_diffraction_orders` docstring, LOSSY layers | "It previously took the real part" |

---

### L147 -- `mode_match` docstring, STABILITY -- "the backward amplitudes ``c-`` used to be referenced at ``z = 0``"

*Left in the source:* the same reference choice as the reason, in present tense, with the measured conditioning ladder below it untouched.

```text
    the backward amplitudes ``c-`` used to be referenced at ``z = 0``, which put
```

### L167-169 -- `mode_match`, the zero-norm refusal MESSAGE -- "this used to surface as an opaque 'SVD did not converge' LinAlgError"

This one was inside a STRING the interpreter executes -- the text of the
`ValueError` a user sees -- so it was out of scope for a documentation-only
sweep (both fingerprints move when it changes) and was rewritten separately,
by WP-A22, with the fingerprints re-recorded in the same commit.

*Left in the source:* the connection to the `LinAlgError`, in present tense --
"Unguarded, such a column reaches the least-squares solve and fails there as an
opaque 'SVD did not converge' LinAlgError that names neither Psi nor the
column."  That is a statement about what the guard prevents, which a reader who
removes it or who has an old traceback still needs; the retired half was the
claim about when the library changed.  The `zero-norm or non-finite` phrase is
unchanged: `test_niche_audit_w6_eme.py:783` matches on it.

```text
        raise ValueError(
            "mode_match: Psi has a zero-norm or non-finite mode column, so it "
            "cannot be normalised (this used to surface as an opaque "
            "'SVD did not converge' LinAlgError from the least-squares solve).")
```

### L235-237 -- `eme_diffraction_orders` docstring, LOSSY layers -- "It previously took the real part"

*Left in the source:* what the complex spectrum buys, with the measured lossy-Airy comparison that follows it.

```text
    LOSSY layers (AUDIT W6 fix).  A complex ``eps_xy`` now keeps the COMPLEX
    ``qz^2`` (``ref_2d_modes(return_complex=True)``), so absorption is modelled.
    It previously took the real part, which made an absorbing slab behave as a
```
