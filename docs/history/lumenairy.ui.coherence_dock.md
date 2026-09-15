<!-- lumenairy-history-doc
module: lumenairy/ui/coherence_dock.py
ast_sha256: 26889acadd9714b4409aaecd70144091a148adc61ec26417b90cd3b482deda55
token_sha256: 52ba746acd1da7dc6fd8c05b40ce471c08eb373d178a24e467f7f3d7034a2b18
pre_relocation_lines: 872
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/ui/coherence_dock.py`

This file holds the version-history narrative that used to live in
`lumenairy/ui/coherence_dock.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks, both from the same finding: Tab 1's "Source shape" combo was an
inert control.  The module-scope placement of the shape -> source-point map,
and the reason for it (it must be exercisable on a box with no PySide6), are
live design constraints and stayed.

This module's ~200 sibling release TAGS -- the `v5.4.3 (audit GUI-resize)` /
`v5.4.4 (audit GUI-resize round 2)` boilerplate repeated across the dock
family -- were stripped in place rather than recorded here: they are one-line
labels on an otherwise-live why-comment, with no narrative attached.  The
full before/after list is in the WP-A17 SWEEP-3 report.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L30-37 | `<module> source-pupil geometry` | the release/audit tag and the "had NEVER been wired" framing |
| L44-45 | `<module> source-pupil geometry` | "the pre-fix Tab 3 helper" |

---

### L30-37 -- `<module> source-pupil geometry` -- the release/audit tag and the "had NEVER been wired" framing

*Left in the source:* the inert-control hazard as a live statement, and the whole module-scope / Qt-free design constraint.

```text
# Source-pupil geometry.  v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25,
# Territory A follow-up): Tab 1's "Source shape" combo (Circular / Annular /
# Dipole / Quadrupole) had NEVER been wired -- ``_run`` built its params dict
# without reading ``combo_shape`` at all, so all four entries produced the
# identical filled-disk image (a live instance of the audit's inert-control
# pattern).  The shape -> source-point mapping lives HERE, at module scope
# and free of any Qt dependency, so it is exercisable (and pinned) on a box
# with no PySide6 installed; the docks only call it.
```

### L44-45 -- `<module> source-pupil geometry` -- "the pre-fix Tab 3 helper"

*Left in the source:* the bit-preservation guarantee, which is the property a reader needs.

```text
#     the count is ~(fill factor) * n**2.  Bit-preserved from the pre-fix
#     Tab 3 helper -- Tab 3's numbers do not move.
```

