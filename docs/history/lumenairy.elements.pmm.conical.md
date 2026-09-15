<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/conical.py
ast_sha256: dd5305a98365e8d9312a0bd763b30d27af3d20f7314f5cca5991b24a8428f6ec
token_sha256: 10d850e5b6d1220f425203d2de7538274780affd6f5de2ad9697d6cfaab5e6f1
pre_relocation_lines: 693
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/elements/pmm/conical.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/pmm/conical.py` -- the T3-3 note recording where the
per-layer window-grid block used to sit, and the "Historical note" retracting an
earlier revision of the same docstring's accuracy claim.  Each block is
reproduced **verbatim** under the source line it came from in the pre-relocation
file.

What did NOT move: the rank-deficiency mechanism, the sibling survey that makes
conical the outlier, and the corrected `~1e-15` agreement figure.

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
| L220-222 | `_pmm_jones_conical_core`, the window-grid ordering | T3-3 -- "This block used to sit AFTER the order cap" |
| L577-582 | `_pmm_jones_conical_core` docstring | a "Historical note" retracting this docstring's own earlier few-percent OOP residual claim |
| L610-611 | `_pmm_jones_conical_core`, the tensor gate | "the old path returned silently wrong numbers for it" |

---

### L220-222 -- `_pmm_jones_conical_core`, the window-grid ordering -- T3-3 -- "This block used to sit AFTER the order cap"

*Left in the source:* the ordering requirement and the whole silent-failure chain it prevents.

```text
    # T3-3 (M1, 2026-08-04).  This block used to sit AFTER the order cap, and
    # the cap was computed from ``nU`` -- the FULL-UNION cell count -- on both
    # paths.  On the per-layer path the half-spaces live on the WINDOW grids,
```

### L577-582 -- `_pmm_jones_conical_core` docstring -- a "Historical note" retracting this docstring's own earlier few-percent OOP residual claim

*Left in the source:* the corrected agreement figure and the three solvers it covers.

```text
    incidence -- normal, planar-oblique, AND conical.  (Historical note: the
    docstring here previously reported a "few-percent OOP-at-conical residual vs
    Berreman"; that was an artifact of a BUG in the ``berreman_jones_1d`` S-matrix
    oracle it was graded against -- fixed 2026-07-05 -- NOT of this generator.
    With the corrected oracle the singular-value agreement is ``~1e-15``; this
    solver, :func:`pmm_jones_2d`, and ``rcwa_jones_2d`` were all correct.)
```

### L610-611 -- `_pmm_jones_conical_core`, the tensor gate -- "the old path returned silently wrong numbers for it"

*Left in the source:* the refusal and the silent-wrong alternative it replaces.

```text
    # out-of-plane cell is rejected loudly (the old path returned silently
    # wrong numbers for it); a UNIFORM cell of any tensor keeps the exact
```
