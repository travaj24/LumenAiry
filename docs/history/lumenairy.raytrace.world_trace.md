<!-- lumenairy-history-doc
module: lumenairy/raytrace/world_trace.py
ast_sha256: 56192a5caf96875f27aaa048de18ea946f40f41b06f953e5712e96201399ffc0
token_sha256: 7e5d517429caf4ff2889351ce87c44ca6b9f554bd5c71bb1e51695bab3a39c81
pre_relocation_lines: 250
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B9 items 1-2: trace_world() gains renormalize= and sphere_normal=, defaults unchanged, matching trace()
re_recorded: 2026-09-13 -- WP-B9 items 1-2: trace_world() gains renormalize= and sphere_normal=, defaults unchanged, matching trace()
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-20 -- WP-C2 (5.49.0): trace_world() defaults to sphere_normal='analytic', matching trace(); 'generic' is the byte-identical way back
re_recorded: 2026-09-20 -- WP-C2 (5.49.0): trace_world() defaults to renormalize='exit', matching trace(); 'surface' is the byte-identical way back
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

