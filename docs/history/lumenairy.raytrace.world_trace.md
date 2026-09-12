<!-- lumenairy-history-doc
module: lumenairy/raytrace/world_trace.py
ast_sha256: bc86743128dac35d879098d43ee5e1ac918e61a31cea618342dee150ceb15220
token_sha256: 5fbec0e52f6a07744cbdcd9ae83bb5c6eaf776b409ece49151cb24001f1c3629
pre_relocation_lines: 250
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/world_trace.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/world_trace.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Three blocks, all twins of sites in `trace.py`.  The DOE guard and the
grating `1 / n2` factor are live physics and stayed, with their measured
ratio; the "pre-fix this site divided unguarded" framing moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L13-14 | `<module> docstring` | the split-provenance claim |
| L174-178 | `trace_world DOE branch` | "Pre-fix this site divided unguarded" |
| L186-186 | `trace_world DOE branch` | "pre-fix" on the measured ratio |

---

### L13-14 -- `<module> docstring` -- the split-provenance claim

*Left in the source:* the bit-for-bit statement.

```text
No physics change: contents are bit-for-bit copies of the original
implementations.
```

### L174-178 -- `trace_world DOE branch` -- "Pre-fix this site divided unguarded"

*Left in the source:* the guard's contract, the JAX-path citation and both failure modes.

```text
            # the sibling numpy loop (``trace.py``) now enforces.  Pre-fix
            # this site divided unguarded, so ``period=0.0`` raised
            # ``ZeroDivisionError`` mid-trace and ``period=nan`` silently
            # NaN-poisoned (L, M).  ``inf`` already gave 0.0 by IEEE
            # division, so that case is bit-identical.
```

### L186-186 -- `trace_world DOE branch` -- "pre-fix" on the measured ratio

*Left in the source:* the physics, the measured ratio and the index-independent OPL rule.

```text
            # exactly n(N-BK7) = 1.503583 pre-fix).  The OPL term keeps the
```

