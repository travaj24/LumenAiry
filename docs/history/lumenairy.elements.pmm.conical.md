<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/conical.py
ast_sha256: 34a489d5c1b65f66c1ec8e8eef6c70504ab4950c7bc815e872481bd0d1731ed2
token_sha256: 803d51a8632d583d4a0f703eaafc13eb5ce46d0a0647a9aa989aa1e420f065d6
pre_relocation_lines: 693
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-11-14 -- grade normalized at the 1-D conical entry points; geometric grading refused at conical incidence (_refuse_geometric_conical, measured inaccurate there)
re_recorded: 2026-11-14 -- 2026-11-14 -- the nodal conical cascade takes its modes from _conical_modes (structured uniform / classical-basis layer solves); geometric grading refused only with PMM_CONICAL_STRUCTURED=False; wavelength passed to the grade normalizer
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

### 2026-11-14 -- the conical cascade's modal solves are structured; geometric grading accepted at conical incidence

`_conical_nodal_solve` now takes every medium's modes from `_conical_modes`:
exactly structured modes for laterally uniform isotropic media
(`_core._sem_modes_uniform_conical`; the classical shared-eig modes at
``ky0 = 0``) and the classical-basis solve for every other layer
(`_core._sem_modes_conical`).  The refusal `_refuse_geometric_conical` added
the same day now fires only with the fail-before switch
`PMM_CONICAL_STRUCTURED = False`; its docstring states why the dense solve
cannot take a geometric layout (the measurement narrative it first carried was
removed from the source).  The 1-D conical entry points pass their wavelength
to the grade normalizer (the default geometric floor then includes the
wavelength part, as on every other 1-D entry point).  On Chebyshev layouts the
results move at the old path's round-off (largest measured: 8.0e-10 on one
rotated tensor layer, 2.2e-8 on two patterned layers over a metal film, where
the discretization errors are 3e-4 .. 2e-2), with the round-off scatter under
a 1e-10 change of phi down from 3.0e-9 to 4.2e-11 on the strip / spacer /
mirror stack of the tests.
