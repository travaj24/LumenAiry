<!-- lumenairy-history-doc
module: lumenairy/elements/elements.py
ast_sha256: 6943dd547be12c6dd1396cb36968727185fc208e3e7ff02a1ef52b6aa64fbf96
token_sha256: 72333d83f9b248dc393cfd43f65df9c238fdf3c9bb91f68fdf99dc5d738d7cbf
pre_relocation_lines: 1402
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-20 -- WP-C1: apply_aperture's edge default flips 'hard' -> 'gray' (the signature's default token)
re_recorded: 2026-09-20 -- VERIFY-C1 D1: the two rim-keyword guards move out of apply_aperture's body into the shared _validate_edge_kwargs, which the chain's element reader also calls, so all three chain routes refuse identically.  VERIFY-C1 D3: the docstring's 'no convergence order at all' is restated as 'first order at best with erratic step orders', with the independent optic's mean orders.
-->

# Version history -- `lumenairy/elements/elements.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/elements.py` -- the v5.2 `_xp_of` consolidation note and the
E-L19 retraction of the `'gaussian'` apodisation bullet's own earlier text.
Each block is reproduced **verbatim** under the source line it came from in the
pre-relocation file.

What did NOT move: the S5-9 shared-LayerSpec note in the module docstring
(`tests/unit/test_g10_s5_9_layerspec_tracked.py` reads it), the OSA/Noll
statement on `zernike`, and the apodisation formulas.

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
| L25-28 | `<module>`, the ``_xp_of`` import | "this used to be a 4-line wrapper duplicated in 5 files" |
| L1069-1070 | `apply_apodized_pupil` docstring, the ``'gaussian'`` bullet | "this bullet used to declare sigma mandatory, which contradicted both the ``sigma`` entry below and the code" |

---

### L25-28 -- `<module>`, the ``_xp_of`` import -- "this used to be a 4-line wrapper duplicated in 5 files"

*Left in the source:* what the alias is for -- keeping in-module references working without touching call sites -- and the audit that consolidated it.

```text
# v5.2 (ROADMAP "Duplicate `_xp_of`" cleanup): this used to be a
# 4-line wrapper duplicated in 5 files; consolidated to the canonical
# backend helper.  The underscore-prefixed alias preserves existing
# in-module references without touching call sites.
```

### L1069-1070 -- `apply_apodized_pupil` docstring, the ``'gaussian'`` bullet -- "this bullet used to declare sigma mandatory, which contradicted both the ``sigma`` entry below and the code"

*Left in the source:* the default itself, which is the contract; the bullet no longer has an earlier version to contradict.

```text
          E-L19: this bullet used to declare sigma mandatory, which
          contradicted both the ``sigma`` entry below and the code).
```
