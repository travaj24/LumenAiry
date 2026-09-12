<!-- lumenairy-history-doc
module: lumenairy/analysis/zernike.py
ast_sha256: d6d07f65c285b06977c29e568d81314b35f57d2c503f104526b10a413a76a10b
token_sha256: c66645e08137b0a4d9cfad0c944a1156a15962d5a867400553302488865d644a
pre_relocation_lines: 876
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/zernike.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/zernike.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

One block, and only its framing moved.  The cached Zernike basis is returned
read-only, and the comment explained that by saying the arrays "used to come
back writable".  What makes the freeze necessary is live and unchanged -- a
single stray in-place write poisons the cache for every later consumer in the
process, measured -- so the source now states that as the reason for the
freeze rather than as a release note.

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
| L362-366 | `zernike_basis_matrix` -- the read-only cache | the `they used to come back writable` framing |

---

### L362-366 -- `zernike_basis_matrix` -- the read-only cache -- the `they used to come back writable` framing

*Left in the source:* the measured poisoning hazard that makes the freeze necessary, and the note that every in-library consumer is read-only so the freeze costs nothing

```text
    ``basis.copy()`` if you need a mutable copy.  v5.29.1 (audit A-13ish):
    they used to come back writable, so a single stray in-place write
    silently poisoned the basis for every later consumer in the process
    (measured: a value written into ``basis[0, 0]`` was still there on the
    next call, and the LRU key never noticed).  Every in-library consumer
```
