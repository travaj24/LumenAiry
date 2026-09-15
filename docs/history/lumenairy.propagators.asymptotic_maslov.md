<!-- lumenairy-history-doc
module: lumenairy/propagators/asymptotic_maslov.py
ast_sha256: b62534cbe4794d121f73147d1793fe523d53e17c9243287cafbb885215df8638
token_sha256: d2ae3314a1d5dceab27552905ce1174df10a48594f7030cdbb4fc0f3740bbd15
pre_relocation_lines: 756
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-B7 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, WP-A4 sec. 6 items 3-8 + VERIFY-B1 F1/F2): the Y4 fused basis evaluation and hoisted Newton factor, the aberration_tensor mode/waist caches, the S6 gate's k1-slope statistic and mean-plus-spread chart sizing, the JAX screen's chief-ray displacement term, and the S9 FFT kernel clip
re_recorded: 2026-09-13 -- WP-B7: Optional type hints on the two new keyword parameters (_phi_v2_hessian_batch's T12_rows, _solve_envelope_stationary_batch's scale_relative_stop)
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/asymptotic_maslov.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/asymptotic_maslov.py`.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

Four blocks moved.  The one worth reading is `van_vleck_weight`'s audit-Y1/Y2
paragraph: it records that the pre-fix integrand carried `|det J|` to the
first power with no `1/lambda`, so the returned field was
`i * lambda * sqrt(|det J|)` times the true one -- measured
`E_code / E_true = 2.0000000000256e-08 j` on an exact free-space chart, i.e.
exactly `i * lambda * z`.  That is the scale a caller holding archived results
needs, and `AberrationTensorResult.van_vleck_weight` still carries the algebra
for converting between the two; only the account of which release changed it
is here.

The W6-A2 **scale-relative convergence verdict** did NOT move.  It is the
derivation of a live contract -- the residual is dimensional, its natural size
is O(1e7) on the library's own default waists, so an absolute `tol=1e-12` was
unreachable and `converged` was False for 288 of 289 pixels that had converged
to machine precision -- and it states explicitly that the ITERATION still keys
off the absolute test, so every returned `v2*` is unchanged.  Deleting that
would invite someone to "simplify" the verdict back.

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
| L45-48 | `<module>` -- Chebyshev alias import | which release moved the helpers and which roadmap item it was |
| L129-139 | `van_vleck_weight` | the pre-fix integrand, the measured `i * lambda * z` ratio, and the disagreement with `propagate_hf_chebyshev_quadrature` |
| L630-637 | `_solve_envelope_stationary_batch` -- the dropped Hessian term | a comment correcting an earlier COMMENT: that the pre-v5.46 wording claimed the term vanishes at the stationary point |
| L639-646 | `_solve_envelope_stationary_batch` -- the `converged` flag | the pre-v4.14.1 line that flagged stalled and singular pixels as successes |

---

### L45-48 -- `<module>` -- Chebyshev alias import -- which release moved the helpers and which roadmap item it was

*Left in the source:* what the aliases are for

```text
# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# Chebyshev helpers moved to lumenairy._math.chebyshev; binding the
# new public names to the legacy underscore-prefixed locals keeps the
# existing call sites in this module unchanged.
```

### L129-139 -- `van_vleck_weight` -- the pre-fix integrand, the measured `i * lambda * z` ratio, and the disagreement with `propagate_hf_chebyshev_quadrature`

*Left in the source:* the full derivation of the LIVE weight, which is the two paragraphs above it

```text

    Audit Y1/Y2: the pre-fix integrand used ``|det J|`` to the FIRST power
    with no ``1/lambda``, so the returned field was
    ``i * lambda * sqrt(|det J|)`` times the true one -- a factor that is
    wavelength- AND field-point-dependent, not the documented "arbitrary
    constant".  Measured on an exact free-space chart
    (``s1 = s2 - z v2``, ``z = 20 mm``, ``lambda = 1 um``):
    ``E_code / E_true = 2.0000000000256e-08 j`` pre-fix, i.e. exactly
    ``i * lambda * z``; the sibling ``propagate_hf_chebyshev_quadrature``
    already carried the correct ``-1j * sqrt(|det d2Phi/ds1 ds2|)``, so the
    two families disagreed by exactly this factor.
```

### L630-637 -- `_solve_envelope_stationary_batch` -- the dropped Hessian term -- a comment correcting an earlier COMMENT: that the pre-v5.46 wording claimed the term vanishes at the stationary point

*Left in the source:* the live statement -- both solvers drop the same term, it changes the rate and not the root, and it does NOT vanish at the stationary point

```text
    All pixels iterate in lockstep starting from ``(v_cx, v_cy)``;
    converged pixels still consume CPU on subsequent iterations but
    that cost is amortised across the batch.  The math matches the
    scalar Gauss-Newton-like solver bit-for-bit -- both drop the same
    ``sum_k (s_1 - s_src)_k d^2 s_1_k / dv2 dv2`` term from the Hessian
    MODEL, which changes the convergence rate and not the root (Y5:
    that term does not vanish at the stationary point, contrary to the
    pre-v5.46 wording -- only the residual does).
```

### L639-646 -- `_solve_envelope_stationary_batch` -- the `converged` flag -- the pre-v4.14.1 line that flagged stalled and singular pixels as successes

*Left in the source:* the live bookkeeping rule and the exact test ``converged`` reports

```text
    v4.14.1 (P1-NEW-3):  prior to v4.14.1 the loop set
    ``converged[idx[done & ~is_conv]] = True`` to drop stalled /
    singular pixels from the active set, which silently flagged
    failures as successes -- contrary to the documented contract.
    The function now uses a separate ``finished`` mask for the
    active-set bookkeeping and writes ``True`` to ``converged`` only
    for genuinely-converged pixels (``rn < tol`` pre-v5.30,
    ``rn < tol * max(r0, 1)`` from v5.30 -- see above).
```
