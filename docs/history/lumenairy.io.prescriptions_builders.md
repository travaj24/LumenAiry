<!-- lumenairy-history-doc
module: lumenairy/io/prescriptions_builders.py
ast_sha256: d8936ee9103c6a8118c622db5af8c1c8ab14d67a99447e38ca28373c7aac4972
token_sha256: b53193614552828f43a0468ebd9136b77219fe8d40af60bd802364c14db77bf6
pre_relocation_lines: 660
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/io/prescriptions_builders.py`

This file holds the version-history narrative that used to live in
`lumenairy/io/prescriptions_builders.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Five blocks, all about the OAP factory's angle convention.  The CONVENTION
(`off_axis_angle` is the surface-normal angle, half the chief-ray fold) and
its derivation `h = 2 f tan(alpha)` stayed in the source -- they are what a
caller has to get right.  What moved is the record of which release
reconciled the docstring with the formulas, and what the factory accepted
before the validation existed.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L251-256 | `make_oap docstring` | the release tags and the "the docstring described this argument as the chief-ray fold angle" reconciliation note |
| L274-276 | `make_oap docstring` | the "pre-v4.15.1 factory accepted ... silently" framing |
| L324-325 | `make_oap docstring` | "in contrast to the pre-v4.15.1" |
| L405-409 | `make_oap` | the "Pre-v4.15.1 the code used" framing |
| L472-475 | `CATALOG['LA1509-C']` | a note correcting an EARLIER FIGURE IN THIS SAME COMMENT ("the earlier '199.68' ... was wrong in its last two digits") |

---

### L251-256 -- `make_oap docstring` -- the release tags and the "the docstring described this argument as the chief-ray fold angle" reconciliation note

*Left in the source:* the convention itself and the geometric test that pins it.

```text
        (Pre-v4.15.1 the docstring described this argument as the
        chief-ray fold angle, but the surrounding formulas already
        assumed the surface-normal convention; v4.15.1 reconciles
        the docstring with the audited geometric derivation below
        and with the chief-ray geometric test ``decenter =
        2*f*tan(alpha)`` at ``alpha = pi/4`` giving ``2 f`` rather
```

### L274-276 -- `make_oap docstring` -- the "pre-v4.15.1 factory accepted ... silently" framing

*Left in the source:* what the validation rejects and the surface it would otherwise produce.

```text
        the pre-v4.15.1 factory accepted ``vertex_radius=0`` and
        ``vertex_radius=-1`` silently, producing a flat or
        oppositely-curved surface).
```

### L324-325 -- `make_oap docstring` -- "in contrast to the pre-v4.15.1"

*Left in the source:* the worked 90-deg case and the divergence the other reading produces.

```text
    finite and physical, in contrast to the pre-v4.15.1
    ``f tan(theta) = f tan(pi/2) = inf``.
```

### L405-409 -- `make_oap` -- the "Pre-v4.15.1 the code used" framing

*Left in the source:* the derivation and the factor-of-two hazard, as a live warning against the other form.

```text
    # angle ``2 * alpha``.  Pre-v4.15.1 the code used
    # ``h = f tan(alpha)`` -- off by a factor of two and divergent
    # at alpha approaching pi/2.  Worst case: a 90-deg-fold OAP
    # (alpha = pi/4) wants h = 2 f, not the pre-fix h = f tan(pi/4)
    # = f.
```

### L472-475 -- `CATALOG['LA1509-C']` -- a note correcting an EARLIER FIGURE IN THIS SAME COMMENT ("the earlier '199.68' ... was wrong in its last two digits")

*Left in the source:* the catalogue values and the measured EFL that verifies them -- the S5 derivation of the row.

```text
    # (measured EFL 99.652 mm @ 587.6 nm; was 199.8652 mm with R1 = 103.29 mm
    #  -- 2.0056x the corrected value.  Re-measured VERIFY-A10 2026-09-12 by
    #  an independent 2x2 ABCD product; the earlier "199.68" in this comment
    #  was wrong in its last two digits.)
```

