<!-- lumenairy-history-doc
module: lumenairy/propagators/vectorial_hfpi.py
ast_sha256: df3e28e6c26dc290a30dfcd566f811cb648f488e5c891ee9575776995ae65417
token_sha256: 5e1bfea1132ffa2bda9014ead15c95c8ab91d831fcc1367893881365d66b11dc
pre_relocation_lines: 763
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/vectorial_hfpi.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/vectorial_hfpi.py`.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

Two of the three blocks are comments correcting earlier COMMENTS, and they are
the sharpest example in this sweep of why that class has to go somewhere.
Both paragraphs existed to retract an appeal to "the full m-theory dipole
formalism" -- a formalism that does not exist in this module or anywhere else
in the library.  The retraction was the right thing for the audit to write and
the wrong thing to leave in the source: what a reader needs is the physics that
IS implemented (a straight-line advance rotates nothing; the direction change
is handled by `_rigid_rotate` and the obliquity by the symmetric scalar
Kirchhoff factor), and both sites say exactly that in the paragraph above the
retraction.  The module docstring's own `versionchanged:: 5.46` note, which
those paragraphs pointed at, is untouched.

Nothing the interpreter executes changed in the move.  The header above
records the SHA-256 of (a) the module's AST with every docstring removed and
source positions ignored, and (b) its `tokenize` stream reduced to
NAME/OP/NUMBER/STRING with comments and docstrings dropped -- both taken from
the file as it stood BEFORE the relocation.
`tests/unit/test_audit2609_a17_history_relocation.py` re-computes both from
the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L333-337 | `advance_vector_paths` | a comment correcting an earlier COMMENT: that this paragraph used to attribute the neglected rotation to 'the full m-theory dipole formalism' |
| L392-397 | `apply_vector_aperture_diffraction` | a comment correcting an earlier COMMENT: that this paragraph used to say the Jones vector is multiplied by `cos(theta_new)` for 'the m-theory dipole obliquity' |
| L481-485 | `apply_vector_aperture_diffraction` -- the re-emission measure | the `V1 (verify pass)` framing |

---

### L333-337 -- `advance_vector_paths` -- a comment correcting an earlier COMMENT: that this paragraph used to attribute the neglected rotation to 'the full m-theory dipole formalism'

*Left in the source:* the physics that is implemented -- no rotation on a straight-line advance, all rotation at emission and re-emission

```text

    (Audit K15/K17: this paragraph used to attribute the neglected
    rotation to "the full m-theory dipole formalism".  No such formalism
    exists in this module or anywhere else in the library -- see the
    module docstring's ``versionchanged:: 5.46`` note.)
```

### L392-397 -- `apply_vector_aperture_diffraction` -- a comment correcting an earlier COMMENT: that this paragraph used to say the Jones vector is multiplied by `cos(theta_new)` for 'the m-theory dipole obliquity'

*Left in the source:* the live composition -- rigid rotation (orthogonal, no amplitude factor) times the symmetric scalar Kirchhoff obliquity and the ``1/(i lambda) dOmega`` prefactor

```text

    (Audit K15/K17: this paragraph used to say the Jones vector is
    "multiplied by ``cos(theta_new)`` to account for the m-theory dipole
    obliquity".  There is no such tensor -- see the module docstring --
    and since v5.46 the obliquity is the symmetric scalar above while the
    direction change is handled by the rotation, not by a scale factor.)
```

### L481-485 -- `apply_vector_aperture_diffraction` -- the re-emission measure -- the `V1 (verify pass)` framing

*Left in the source:* which helper carries the derivation and what the legacy measure costs a chain

```text
    # V1 (verify pass, 2026-09-12): the MEASURE is the shared
    # :func:`~lumenairy.propagators.hfpi._reemission_measure` -- see its
    # docstring for the derivation and for what the pre-fix factor cost
    # (cascaded amplitudes low by ``n_paths * r_in``, measured identical
    # in this module and in the scalar twin).
```
