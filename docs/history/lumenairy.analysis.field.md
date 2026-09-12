<!-- lumenairy-history-doc
module: lumenairy/analysis/field.py
ast_sha256: a31a56bbb8964307ba48696bc8a76f264f9131a39e7c0f76d28229f53dde39bf
token_sha256: c7e0ae3ea542c1be05e6710c794fef9c5b7deca629384e0a02f21a98e461e606
pre_relocation_lines: 1492
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/field.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/field.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

Two small blocks.  `append_image_plane`'s W4d paragraph is a do-not-simplify
note with a measured discriminator and stayed almost intact -- only its
opening, which described the WORLD-frame branch by what it "used to" do, is
here; the ambiguity argument that makes the current placement necessary (a
folded world list ending AT a mirror carries the pre-fold frame, one ending
after a re-aligning coord-break carries the post-fold frame, and no property
of a raw prescription tells them apart) is the reason the code cannot go back
to `last.world_R[:, 2]`, so it stayed in full.  The W4c mirror-parity block
above it, with its measured 185x spot error, likewise stayed.

The second is provenance: `sensitivity_ranking` recorded which GUI dock the
algorithm was lifted out of.

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
| L208-210 | `append_image_plane` -- the world-frame branch | the `used to place the plane along last.world_R[:, 2] unconditionally` framing |
| L1403-1405 | `sensitivity_ranking` | which GUI dock the algorithm was lifted out of |

---

### L208-210 -- `append_image_plane` -- the world-frame branch -- the `used to place the plane along last.world_R[:, 2] unconditionally` framing

*Left in the source:* the whole ambiguity argument, the resolution, the bit-identity guarantee and the measured discriminator

```text
    W4d (closes W4c's flag F1).  The WORLD-frame branch used to place the
    plane along ``last.world_R[:, 2]`` unconditionally, which is right for
    some world lists and wrong for others -- and the AMBIGUITY IS REAL:
```

### L1403-1405 -- `sensitivity_ranking` -- which GUI dock the algorithm was lifted out of

*Left in the source:* that the same algorithm is callable from a script

```text
    Lifted-from-GUI helper: the same algorithm previously buried in
    ``ui/sensitivity_dock.SensitivityDock._run_ranking`` is now
    callable from any script.
```
