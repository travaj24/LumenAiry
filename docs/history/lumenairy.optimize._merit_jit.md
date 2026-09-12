<!-- lumenairy-history-doc
module: lumenairy/optimize/_merit_jit.py
ast_sha256: 3e1d1b9c131e1ffb4f9c5759e440fdf57bcc2be8d99b720ecc71597c9dbbd7c1
token_sha256: 74c1120f2e99e60380770f2617bf6d3806ea6fe9863052020a1a45409a03f327
pre_relocation_lines: 273
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
-->


# Version history -- `lumenairy/optimize/_merit_jit.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/_merit_jit.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

This module exists only because of one change (the fused Numba kernel), so
its release tag appears in the module docstring, the import block, the public
helper's docstring and the fallback comment.  The CONTRACT -- that the NumPy
fallback computes the same expression, and that the call site selects on
`_NUMBA_AVAILABLE` -- is live and stayed.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L7-9 | `<module> docstring` | the release / roadmap tag and the "pre-v5.3 NumPy path" framing |
| L34-36 | `<module> docstring` | "is exactly the pre-v5.3 path" |
| L38-38 | `<module> docstring` | the release / roadmap tag on the author line |
| L48-48 | `<module> Numba probe` | the release / roadmap tag |
| L176-176 | `_multi_field_tilt_phasor_masked` | the release / roadmap tag |
| L217-217 | `_multi_field_jit` | the release and audit tag |
| L270-271 | `_multi_field_jit` | "exactly the pre-v5.3 path that this module supersedes" |

---

### L7-9 -- `<module> docstring` -- the release / roadmap tag and the "pre-v5.3 NumPy path" framing

*Left in the source:* the temporary-count argument that is the whole reason for the kernel.

```text
the masked tilted plane wave on each per-field leg.  v5.3 (ROADMAP
v5.3 horizon -- MultiFieldMerit JIT): the pre-v5.3 NumPy path
materialises three N x N temporaries per field
```

### L34-36 -- `<module> docstring` -- "is exactly the pre-v5.3 path"

*Left in the source:* the equivalence guarantee and the selection rule.

```text
The NumPy fallback (when Numba is unavailable) is exactly the
pre-v5.3 path -- the call site picks which to invoke based on the
module-level ``_NUMBA_AVAILABLE`` flag.
```

### L38-38 -- `<module> docstring` -- the release / roadmap tag on the author line

*Left in the source:* the author line.

```text
Author: Andrew Traverso -- v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT).
```

### L48-48 -- `<module> Numba probe` -- the release / roadmap tag

*Left in the source:* the measured reason the probe is lazy (~1.8 s of cold-start import) and the shared-probe note.

```text
# v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT): Numba probe, LAZY (audit
```

### L176-176 -- `_multi_field_tilt_phasor_masked` -- the release / roadmap tag

*Left in the source:* the whole contract: the expression it returns, the JIT gate and the size threshold.

```text
    v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT): single-call
```

### L217-217 -- `_multi_field_jit` -- the release and audit tag

*Left in the source:* the hazard.

```text
    # v5.4 (audit P3): defensive dtype check -- complex256 would silently downgrade
```

### L270-271 -- `_multi_field_jit` -- "exactly the pre-v5.3 path that this module supersedes"

*Left in the source:* the equivalence statement and why the fallback is kept.

```text
    # Pure-NumPy fallback.  Exactly the pre-v5.3 path that this
    # module supersedes (kept for parity + correctness pinning).
```

