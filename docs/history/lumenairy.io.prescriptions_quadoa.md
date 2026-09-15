<!-- lumenairy-history-doc
module: lumenairy/io/prescriptions_quadoa.py
ast_sha256: 3118a33a2c664cb7241ec2f54985fb641e718959f806a6c7751812ce1d9e6d16
token_sha256: 9c92f6d765fbd4222190344f293d5d61e0b4e32087beef2da1cdaf8502ad9a9e
pre_relocation_lines: 440
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/io/prescriptions_quadoa.py`

This file holds the version-history narrative that used to live in
`lumenairy/io/prescriptions_quadoa.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Four blocks.  Two are a LIVE READER CONTRACT -- a legacy file really does
carry `[4.0, 6.0, ...]` where coefficients belong, and the loader still has
to cope -- so those statements stayed; what moved is the attribution of which
release wrote the bad files.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L49-51 | `_quadoa_serialize_aspheric` | the "Pre-v4.11.2 this iterated dict keys as if they were values" framing |
| L60-62 | `_quadoa_serialize_aspheric` | the "Pre-fix the coefficients were written unscaled" framing |
| L91-93 | `_quadoa_parse_aspheric` | the release attribution on the legacy serializer |
| L177-181 | `export_quadoa_json` | the "pre-v5.46 this fell back to 0" framing |

---

### L49-51 -- `_quadoa_serialize_aspheric` -- the "Pre-v4.11.2 this iterated dict keys as if they were values" framing

*Left in the source:* the rule and the exact wrong output it guards against.

```text
    keys).  ``None`` -> ``None``.  Pre-v4.11.2 this iterated dict keys
    as if they were values, writing the powers [4.0, 6.0, ...] instead
    of the coefficients.
```

### L60-62 -- `_quadoa_serialize_aspheric` -- the "Pre-fix the coefficients were written unscaled" framing

*Left in the source:* the scale law, the inconsistency it prevents and the no-op guarantee for metre exports.

```text
    Pre-fix the coefficients were written unscaled, so a MM file carried
    radii in mm but aspheres in per-meter -- an internally inconsistent
    prescription for any external Quadoa reader.  ``scale=1.0`` (the
```

### L91-93 -- `_quadoa_parse_aspheric` -- the release attribution on the legacy serializer

*Left in the source:* the reader's live coping rule for a legacy list.

```text
      (the pre-v4.11.2 serializer wrote ``[4.0, 6.0, ...]`` -- those
      values are uninterpretable, so a legacy list is read at face
      value as coefficients starting from power=4).
```

### L177-181 -- `export_quadoa_json` -- the "pre-v5.46 this fell back to 0" framing

*Left in the source:* the whole invented-stop hazard as what the fallback would do.

```text
            # I8 (AUDIT_ADVERSARIAL_EXHAUSTIVE 2026-09-11): pre-v5.46 this
            # fell back to 0, and line ~211 then wrote ``is_stop = (i == 0)``
            # on every surface -- so a prescription with NO declared stop
            # round-tripped as one with a stop at surface 0, INVENTING an
            # aperture stop the design never had.  Keep None and write no
```

