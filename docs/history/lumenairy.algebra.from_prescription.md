<!-- lumenairy-history-doc
module: lumenairy/algebra/from_prescription.py
ast_sha256: 807eb930f7d40fb1afe4df0437dfa385eaf726bece15f0e4f62481637e37e2f6
token_sha256: 4f21ae1c890b3dd14f0ac0ab4a44d78e64b6965f611f746210ee5e2f2aed4644
pre_relocation_lines: 230
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/algebra/from_prescription.py`

This file holds the version-history narrative that used to live in
`lumenairy/algebra/from_prescription.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

One block: a note about a previous cross-layer AGREEMENT on a wrong answer.
The physics rule (a flat fold still reverses propagation, so the parity
toggle is R-independent) stayed; the record of which release made the two
layers agree wrongly moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L182-190 | `prescription_to_operators mirror branch` | the release/audit tag and the "had made this layer skip" narrative |

---

### L182-190 -- `prescription_to_operators mirror branch` -- the release/audit tag and the "had made this layer skip" narrative

*Left in the source:* the physics rule, the cross-layer agreement hazard and the bit-identity guarantee.

```text
            # v4.15.2 (audit P1-NEW-B) had made this layer skip the
            # parity toggle for FLAT mirrors specifically to match
            # ``raytrace.system_abcd``, whose own ``elif surf.is_mirror
            # and np.isfinite(R)`` gating dropped it.  That made the two
            # layers agree on the WRONG answer -- ``system_abcd`` is now
            # fixed (see the S11-1 note in raytrace/seidel.py, with the
            # exact-3-D-trace oracle numbers), so this twin follows it
            # back to the R-independent form.  Bit-identical for curved
            # mirrors and for every mirror-free prescription.
```

