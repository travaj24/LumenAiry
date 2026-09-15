<!-- lumenairy-history-doc
module: lumenairy/ui/main_window.py
ast_sha256: 9a5bb39141721e20b1b39ec110a6dea18462f505402161ca2d0e710ee3eaf9e3
token_sha256: 91263ec5ee8ffc91686d12b041e62cf0cd3535f9a67b40ada428d496520c714c
pre_relocation_lines: 3645
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/ui/main_window.py`

This file holds the version-history narrative that used to live in
`lumenairy/ui/main_window.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks.  The high-index glass table's contents are a MEASUREMENT (which
glasses fall back to the N-BK7 estimate and by how much the EFL is off), so
they stayed; the "used to list only" framing moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L475-475 | `MainWindow materials dock` | "the old split layout" |
| L1659-1663 | `MainWindow catalogue menu` | "The table used to list only" |

---

### L475-475 -- `MainWindow materials dock` -- "the old split layout"

*Left in the source:* the reason both docks stay alive and in the View menu.

```text
        # View menu) for users who prefer the old split layout.
```

### L1659-1663 -- `MainWindow catalogue menu` -- "The table used to list only"

*Left in the source:* the measured consequence of a short table, which is why the list must stay complete.

```text
                    # menu-sort heuristic.  The table used to list only
                    # ('S-LAH64', 'N-SF11') -- so Thorlabs parts carrying
                    # N-LASF9 or S-NPH1 fell back to the n_d=1.5168 N-BK7
                    # estimate and sorted alongside crown singlets despite
                    # having ~10-15% shorter EFLs at the same radii.
```

