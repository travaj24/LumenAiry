<!-- lumenairy-history-doc
module: lumenairy/analysis/zernike.py
ast_sha256: 658f5e715e0433f5467c4879f54f207d5a38434ed66b3543df2f7a7de61a3db9
token_sha256: 8640108bea4a0a353ef4ab8289e5070ca84f8ed51bcd75a5abbdabe3af6200c5
pre_relocation_lines: 876
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B8 (audit A6.3): the basis build shares one rho**k memo and one hoisted pupil mask across the modes (1.58-3.57x, bit-identical), and _zernike_radial hands orders n >= 22 to the Kintner recurrence, which is where the factorial sum's relative error against an exact rational oracle first exceeds 1e-9
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
