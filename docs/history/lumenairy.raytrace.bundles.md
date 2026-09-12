<!-- lumenairy-history-doc
module: lumenairy/raytrace/bundles.py
ast_sha256: 1e558e7d45fe3f5f03acb73df77f3d8884c39af932cf8c08e2dbeea86fd37411
token_sha256: 06931463d3f6f7c415ab14583d46dac63ea6af2c4160df99e2cef046fcb57f82
pre_relocation_lines: 227
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/bundles.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/bundles.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks about one default: `ray_to_beamlets` folds `opd` (piston phase)
and `alive` into the beamlet amplitude.  Both the reason and the escape hatch
are live and stayed.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L152-157 | `ray_to_beamlets` | the "Pre-fix (audit P2-33) the default silently dropped both" and "the pre-fix escape hatch" framing |
| L170-171 | `ray_to_beamlets` | "keep the old all-ones default" |

---

### L152-157 -- `ray_to_beamlets` -- the "Pre-fix (audit P2-33) the default silently dropped both" and "the pre-fix escape hatch" framing

*Left in the source:* the whole argument for folding, and the explicit-amplitude escape hatch.

```text
    Pre-fix (audit P2-33) the default silently dropped both: every
    beamlet got phase 0 -- zeroing all inter-beamlet piston phases,
    the exact quantity coherent recombination interferes on -- and
    dead/TIR rays contributed full amplitude 1.  An explicitly passed
    ``amplitude`` is used verbatim (no opd/alive folding), preserving
    the pre-fix escape hatch where callers folded the phase themselves.
```

### L170-171 -- `ray_to_beamlets` -- "keep the old all-ones default"

*Left in the source:* the getattr fallbacks and their result.

```text
        # alive -> all-True), so schema-less bundles keep the old
        # all-ones default.
```

