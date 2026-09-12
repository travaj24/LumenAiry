<!-- lumenairy-history-doc
module: lumenairy/analysis/opd.py
ast_sha256: 7f684df0c053dbe2152be4c7a81cab8eeeb02309d1d3513b207547fafd48832b
token_sha256: 6823f40239948aa918b5d107e59684a59007e1aa78ed1f2e3508e3753cf079fc
pre_relocation_lines: 1257
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/opd.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/opd.py`.  Each block is reproduced **verbatim** under the
source line it came from in the pre-relocation file.

Exactly one block moved, and it is a single sentence: a comment correcting an
earlier COMMENT, recording that `opd_profile`'s docstring used to claim the
cut sits at exactly `y = 0` / `x = 0`.  The S11-6e argument that explains the
live behaviour -- the index is the floor `N // 2` while the returned `coord`
axis is centred with the float `N / 2`, so an odd-`N` cut sits at `-d / 2`,
which is not a bug because a centred odd grid has no sample at zero -- stayed
in full.

Everything else the loose classifier flags in this module is measurement that
bounds live behaviour and was not touched: the AN-2 interior-zero unwrap
hazard, the piston-anchor measurements on decentred pupils (+1.0000 and
+2.0000 waves), and the residue-versus-sampling split with its measured
0.0000-waves residue for a radially symmetric aliased wavefront.

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
| L971-973 | `opd_profile` -- `axis` | a comment correcting an earlier COMMENT: that the docstring used to claim the cut is at exactly `y = 0` / `x = 0` |

---

### L971-973 -- `opd_profile` -- `axis` -- a comment correcting an earlier COMMENT: that the docstring used to claim the cut is at exactly `y = 0` / `x = 0`

*Left in the source:* the whole S11-6e argument, which explains where the cut actually sits and why that is correct

```text
        nearest samples.  The docstring (which used to claim the cut is
        at exactly ``y = 0`` / ``x = 0``) is what was wrong; the code and
        the returned ``coord`` are unchanged.
```
