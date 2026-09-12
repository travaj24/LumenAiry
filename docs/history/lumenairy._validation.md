<!-- lumenairy-history-doc
module: lumenairy/_validation.py
ast_sha256: c833dde82cd90c8dc16b1a824e12d853c03d4a05039b85fab8326dd837a250bf
token_sha256: 6712f57a993bd273cb66c0b9a054cd975fe5deccd87eb56de62778b53e94b070
pre_relocation_lines: 274
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/_validation.py`

This file holds the version-history narrative that used to live in
`lumenairy/_validation.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Three blocks.  This module is one long guard, and each block records which
audit widened it.  The GUARD's contract stayed at every site; the audit
attributions and the "the message promised X but enforced Y" framing moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L104-110 | `validate_field_input` | the release/audit tags and the "parameterised after the v4.15.4 audit noted" framing |
| L146-149 | `validate_field_input Raises` | the release/audit tag and the "promised ... since v4.15.2 but enforced" framing |
| L205-208 | `validate_field_input 3-D branch` | the release/audit tag |
| L247-248 | `validate_field_input` | the release/audit tag and the past-tense framing |

---

### L104-110 -- `validate_field_input` -- the release/audit tags and the "parameterised after the v4.15.4 audit noted" framing

*Left in the source:* the vocabulary, the Raises cross-reference and the per-site declaration rule.

```text
        "field" string.  v4.15.5 (P2-NEW-F1-3): parameterised after
        the v4.15.4 audit noted that the vector-diffraction sites
        took a pupil, not a field, but the error message hardcoded
        "field".  v5.31 (audit A-9): the value must be a member of
        the closed :data:`_INPUT_KINDS` vocabulary
        (``'field'`` / ``'psf'`` / ``'pupil'``) -- see the Raises
        section -- and is now declared explicitly at every wired
```

### L146-149 -- `validate_field_input Raises` -- the release/audit tag and the "promised ... since v4.15.2 but enforced" framing

*Left in the source:* the promise-vs-enforcement gap as a live statement about what the guard must check.

```text
        offender).  v5.46 (audit Z4): the guard's message promised
        "2-D complex" since v4.15.2 but enforced only ``ndim == 2``, so
        both of these reached the kernels and produced a plausible
        finite result computed by the wrong arithmetic --
```

### L205-208 -- `validate_field_input 3-D branch` -- the release/audit tag

*Left in the source:* the hint's content and the note that the bare iterate is still valid.

```text
            # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b follow-up):
            # the ensemble hint now points at the
            # :func:`propagate_ensemble` helper rather than the bare
            # iterate-pattern.  The bare iterate still works (and is
```

### L247-248 -- `validate_field_input` -- the release/audit tag and the past-tense framing

*Left in the source:* what the two guards below catch.

```text
    # v5.46 (audit Z4): the two 2-D inputs that used to pass this guard and
    # then compute something plausible but wrong.
```

