<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/oned.py
ast_sha256: 535a5f73a140596da713c48865892410f3c3c3a3b66fe005804d9c02b655ebff
token_sha256: 2c44d68dd138f78ddb6c9d84e371ea732152649791438123c651d01b49dad569
pre_relocation_lines: 1864
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/elements/pmm/oned.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/pmm/oned.py` -- the clauses recording what `set_source` used
to bypass, what two marginal degrees could previously corroborate, what the
formerly-dense normal-incidence resonances were, and what the pre-fix
covariant-for-OOP routing was validated against.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

What did NOT move: the CROSS-ENGINE SEAM note, the TM / wall-corner convergence
statement and the far-field order-budget wording that
`tests/unit/test_audit2609_a12_pmm1d.py` asserts on, and the measured
factorization defect that sizes the OOP routing.

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
| L46 | `<module>`, the re-exported resolver | "``set_source`` previously bypassed it" |
| L413-414 | `pmm_jones_1d` docstring, ``stabilize`` | "two marginal degrees could previously corroborate each other ~1e-3 off" |
| L416-421 | `pmm_jones_1d` docstring, ``stabilize`` | "NB v5.14 also fixed the root cause of the formerly-DENSE normal-incidence resonances" |
| L1151-1153 | `pmm_jones_1d_slanted`, the OOP routing | "The pre-fix 'covariant-for-OOP-too' routing was validated in a world where all three engines shared the factor-i defect" |
| L1683 | `pmm_efficiency_vs_wavelength` docstring | "the PMM family previously had no dispersive-material sweep" |

---

### L46 -- `<module>`, the re-exported resolver -- "``set_source`` previously bypassed it"

*Left in the source:* the audit id and the reason the re-export exists.

```text
# (audit S1-7 -- ``set_source`` previously bypassed it).  Re-exported here for
```

### L413-414 -- `pmm_jones_1d` docstring, ``stabilize`` -- "two marginal degrees could previously corroborate each other ~1e-3 off"

*Left in the source:* the failure the energy-cleanest tie-break prevents, in present tense.

```text
        member preferred on lossless structures (v5.14: two marginal degrees
        could previously corroborate each other ~1e-3 off).  The consensus is
```

### L416-421 -- `pmm_jones_1d` docstring, ``stabilize`` -- "NB v5.14 also fixed the root cause of the formerly-DENSE normal-incidence resonances"

*Left in the source:* the live consequence -- ``stabilize=False`` conserves energy at every probed degree, so the consensus is a safety net rather than a necessity.

```text
        NOT strictly monotone in the requested degree.  NB v5.14 also fixed
        the root cause of the formerly-DENSE normal-incidence resonances (a
        noise-sensitive legacy forward-mode branch); with that fix
        ``stabilize=False`` conserves energy at every probed degree, and the
        consensus is a safety net rather than a necessity.  Set ``False`` to
        solve at exactly ``degree``.
```

### L1151-1153 -- `pmm_jones_1d_slanted`, the OOP routing -- "The pre-fix 'covariant-for-OOP-too' routing was validated in a world where all three engines shared the factor-i defect"

*Left in the source:* the routing rule and the documented limitation; the measured defect stays above it.

```text
        # The pre-fix 'covariant-for-OOP-too' routing was validated in a world
        # where all three engines shared the factor-i defect and agreed on the
        # same symmetrized wrong answer.  Explicit 'covariant' still solves
```

### L1683 -- `pmm_efficiency_vs_wavelength` docstring -- "the PMM family previously had no dispersive-material sweep"

*Left in the source:* the audit id that added it.

```text
    audit: the PMM family previously had no dispersive-material sweep).
```
