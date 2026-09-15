<!-- lumenairy-history-doc
module: lumenairy/optimize/jax_merits.py
ast_sha256: 3821a4fe12c45154d28f3530632572cc83c5c24f82b84b46f589405f701ef277
token_sha256: 0b871f7853d721c1293d01bb6a8f1dc722b0f969127fa6ddc45ac81d196c04a5
pre_relocation_lines: 806
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/optimize/jax_merits.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/jax_merits.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two threads: the `jax_enable_x64` constructor side effect (which is now
explicit and warns), and the `w_o` convention on the JAX sigma branch.  Both
arguments are live -- an unconditional process-wide `jax.config.update` is
still undefined behaviour mid-trace, and the convention still has to cancel
against the aberration-free reference -- so both stayed in the source.  The
release tags and the "used to call" / "previously a SILENT" clauses moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L29-36 | `_ensure_jax_x64` | the "used to call ... unconditionally" framing |
| L40-40 | `_ensure_jax_x64` | "no longer silent" |
| L222-222 | `the JAX sigma-branch w_o convention` | the release/audit tag |
| L445-449 | `make_lg_aberration_merit_jax` | "(previously a SILENT global side effect)" |
| L486-486 | `make_lg_aberration_merit_jax` | the release and audit tag |
| L582-582 | `make_lg_aberration_merit_jax` | the release/audit tag |
| L596-596 | `make_lg_aberration_merit_jax` | the release/audit tag |

---

### L29-36 -- `_ensure_jax_x64` -- the "used to call ... unconditionally" framing

*Left in the source:* the entire hazard as a standing statement about what an unconditional update does, including the mid-trace undefined behaviour.

```text
    S3-14 (audit AUDIT_V5_24_2): :func:`make_lg_aberration_merit_jax` and
    :func:`optimize_traced_geometry` used to call
    ``jax.config.update('jax_enable_x64', True)`` unconditionally as a
    CONSTRUCTOR SIDE EFFECT -- silently flipping a PROCESS-WIDE global.  A
    caller depending on JAX's default float32 elsewhere in the same
    process was switched to float64 with no signal, and (worse) the flip
    is undefined behaviour if it happens mid-trace under an outer
    ``jax.jit``.  This helper makes the requirement explicit:
```

### L40-40 -- `_ensure_jax_x64` -- "no longer silent"

*Left in the source:* the rule and the warning.

```text
      ``RuntimeWarning`` so the global mutation is no longer silent.
```

### L222-222 -- `the JAX sigma-branch w_o convention` -- the release/audit tag

*Left in the source:* the whole convention-not-measurement argument and the cancellation it relies on.

```text
# v5.46 (VERIFY-A4 follow-up, O-5): fraction of the fit's ``s2`` half-range
```

### L445-449 -- `make_lg_aberration_merit_jax` -- "(previously a SILENT global side effect)"

*Left in the source:* both modes of the flag and what each does.

```text
    requirement is met (S3-14): the default enables x64 process-wide if
    it is off, now with a ``RuntimeWarning`` (previously a SILENT global
    side effect); pass ``enable_x64=False`` to instead REQUIRE x64 be set
    already and raise a clear ``RuntimeError`` otherwise, mutating no
    global state.
```

### L486-486 -- `make_lg_aberration_merit_jax` -- the release and audit tag

*Left in the source:* the restriction, the missing general tensor, and why non-(0,0) targets are rejected at construction.

```text
    # v4.13.2 (C-P0-1): aberration_tensor_lg00_jax only computes the
```

### L582-582 -- `make_lg_aberration_merit_jax` -- the release/audit tag

*Left in the source:* the dimensionless-ratio argument and the w_o = 1.0 pinning on both calls.

```text
            # v5.46 (audit Y2 follow-up): the same coefficient on the
```

### L596-596 -- `make_lg_aberration_merit_jax` -- the release/audit tag

*Left in the source:* what makes the ratio a Strehl and the cancellation note.

```text
                # v5.46 (VERIFY-A4 follow-up, O-5): the sigma-grid OVERLAP,
```

