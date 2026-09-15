<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/stack2d_pure.py
ast_sha256: f87cd4e43438e60f24d69ec66ef9e66955762a00987023af1dcfe8fd3afb900a
token_sha256: 099b8f246d8109e5f24d43b7a1960d47b15b543676afd18fe2c842fcd8c270cf
pre_relocation_lines: 2190
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- deferred shared-grid advisory passes stacklevel=4 so it reports at the caller's solve(), not at this file (WP-A22 item 8b)
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/elements/pmm/stack2d_pure.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/pmm/stack2d_pure.py` -- the `ROUND 2` / `ROUND 3` / `ROUND 4`
headings on the two per-layer guards and the sliver-band warning.  Each block is
reproduced **verbatim** under the source line it came from in the pre-relocation
file.

What did NOT move: the two guards themselves, the conforming-stack exemption and
the per-axis rule, all of which describe what the code does now.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment plus a pointer to this file; those are noted per block
below as *Left in the source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L1778-1780 | `PMM2DStackPure.solve` docstring | the ROUND 2 heading on the two per-layer guards |
| L1824-1833 | `PMM2DStackPure.solve`, the sliver-band warning | the ROUND 3 / ROUND 4 chronology and what round 3 asked of the stack instead of per axis |

---

### L1778-1780 -- `PMM2DStackPure.solve` docstring -- the ROUND 2 heading on the two per-layer guards

*Left in the source:* that the path carries two guards the shared path does not need, and that they are layered on purpose.

```text
        ROUND 2 (2026-09-11, ``FIX_PMM2D_MORTAR_ROUND2_2026_09_11.md``).  This
        path carries TWO guards that the shared path does not need, and they
        are layered on purpose:
```

### L1824-1833 -- `PMM2DStackPure.solve`, the sliver-band warning -- the ROUND 3 / ROUND 4 chronology and what round 3 asked of the stack instead of per axis

*Left in the source:* both rules -- the warning is conditioned on a real cross-grid interface, and the test is PER AXIS -- with the reason for each.

```text
        # ROUND 3 (VERIFY S5.4): the band ABOVE the width contract is accepted
        # and measurably degraded, and until now SILENTLY.  This is the one
        # place where the NEIGHBOURS are known, so the warning is conditioned
        # on the stack actually building a cross-grid interface -- a fully
        # CONFORMING per-layer stack takes the plain square match everywhere
        # and is measured delta-independent, so it must not warn.
        # ROUND 4 (VERIFY round 3, DEFECT 2): that test is PER AXIS.  Round 3
        # asked it of the STACK and then scanned both axes, so layers differing
        # on x while sharing the y wall array warned about a narrow y segment
        # on which the mortar is the identity.
```
