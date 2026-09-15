<!-- lumenairy-history-doc
module: lumenairy/propagators/sas.py
ast_sha256: 9e2418e8813271e5af722ad2f4234330996a6560f8e017e2b5f15a189bb5a8b7
token_sha256: 373ceb831de963b19b8ce42488786b4403929582ed3f110500fd55b17916347d
pre_relocation_lines: 393
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- SAS gains the near-field chirp-sampling guard (z < N dx^2 / lambda), the complement of the paper's far-field z_limit: a RuntimeWarning in fresnel_propagate's K1 shape, values unchanged.
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/sas.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/sas.py`.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

Every block here is the same shape, and it is the shape the WP-A17 SWEEP-1
follow-up pass was asked to close: a **live guard whose comment explained
itself by naming the release that added it** ("Pre-fix X happened", "Pre-4.12
the dispatcher only passed ...").  The hazard X is still reachable -- the guard
is the only thing preventing it -- so the source now states X in the present
tense, as what goes wrong WITHOUT the guard, together with every measurement
that sizes it.  What moved is the release attribution and the
"bit-identical to pre-fix" reassurance that travelled with it.

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
| L223-227 | `_sas_kernels` -- the dtype note | the `are no longer built in the caller's real dtype` framing |
| L260-274 | `_sas_kernels` -- the float64 frequency axes | the `Pre-fix these were float32 for a complex64 caller` framing |

---

### L223-227 -- `_sas_kernels` -- the dtype note -- the `are no longer built in the caller's real dtype` framing

*Left in the source:* the rule and the pointer to the derivation below

```text
    # K3 (audit 2026-09-11): the kernel PHASE ARGUMENTS are no longer
    # built in the caller's real dtype -- see the float64 note at the
    # frequency-axis construction below.  Only the finished complex
    # kernels are cast to ``target_cdtype``; the stored / returned dtype
    # is unchanged.
```

### L260-274 -- `_sas_kernels` -- the float64 frequency axes -- the `Pre-fix these were float32 for a complex64 caller` framing

*Left in the source:* the whole cancellation argument and its measurement -- this is what stops someone building the axes at the output dtype to 'save memory'

```text
        # K3 (audit 2026-09-11): these axes and every kernel PHASE
        # ARGUMENT below are built in float64 regardless of the output
        # dtype; only the finished complex kernels are cast to
        # ``target_cdtype``.  This is the "f64-carrier-then-cast" recipe
        # ``fresnel.py`` already uses for its quadratic carrier (v5.17.x
        # P2-29) and that the ASM transfer function uses via its mod-2*pi
        # fold.  Pre-fix these were float32 for a complex64 caller, and
        # the ``h_AS - h_Fr`` difference below is a near-1 cancellation
        # whose absolute error is ~eps REGARDLESS of how small the
        # difference is -- then multiplied by ``k*z``: measured
        # max|Delta(h_AS - h_Fr)| = 9.091e-08 float32 vs float64
        # (N_new = 1024, dx = 1 um, lambda = 633 nm), i.e. 2.9e-3 rad of
        # phase error at z = 3.24 mm and 0.902 rad at z = 1 m -- and long
        # distance is SAS's whole reason to exist.  The axes are 1-D and
        # field-independent, so the float64 build costs nothing.
```
