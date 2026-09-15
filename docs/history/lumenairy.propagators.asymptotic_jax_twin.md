<!-- lumenairy-history-doc
module: lumenairy/propagators/asymptotic_jax_twin.py
ast_sha256: 6c9e1419f0baf53eb77fadd98d9498d4032747a96d3868e52719a5cd0c947ccb
token_sha256: 6f938a27998a7bdad9322b3f498963ed9f62ddde2327d3477a4de420981e9382
pre_relocation_lines: 1216
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- P1-NEW-4: safe_bquad and safe_phi take dtype-matched zeros((), x.dtype) fills instead of the 0.0+0.0j literal, which promoted the real phi_star to complex (WP-A22 follow-up)
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/asymptotic_jax_twin.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/asymptotic_jax_twin.py`.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

Ten small blocks moved; every one of them is release provenance wrapped around
a statement that is simply true of the code (`v5.1.0 file-split (Agent D)`,
`v5.2 (ROADMAP ... extraction)`, `Pre-fix, ... auto-enabled x64`, `Pre-fix it
was NOT registered with the ... registry`).  The statements stayed; the
provenance is here.

The one substantive split is in `_differentiable_lstsq`.  Its "WHY NOT THE
NORMAL EQUATIONS" block is a live do-not-reinstate argument with a measured
table, and both the table and the two failure mechanisms stayed in the source.
What moved is (a) the "this function used to solve ... documented as shifting
the solution by ~1e-10 relative ... that claim was wrong by eight orders"
framing -- a comment correcting an earlier COMMENT -- and (b) the paragraph
measuring how wrong the SUPERSEDED estimator was on an independent ray set,
which describes an estimator this function no longer offers.

Two long blocks deliberately did NOT move: the audit-Y3 / W3-T3b note on
`_lg00_sampling_waist` (why the closed-form `lambda_max` replaces
`jnp.linalg.eigvalsh`, why `w_o` is a convention rather than a length here,
why its dependence on `M` is `stop_gradient`-ed, and the finite-difference
agreements that establish all three), and the W6-A10 vignetting-parity table
in `fit_canonical_polynomials_jax`.  Both are the derivation of live
behaviour, and both carry explicit "change one and you must change the other"
instructions.

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
| L3-5 | `<module>` | the v5.1.0 file-split provenance |
| L65-76 | `_require_jax_x64` | the pre-fix account of `fit_canonical_polynomials_jax` auto-enabling x64 by mutating a global mid-call |
| L109-116 | `_chebyshev_vandermonde_xp` | the `not the prior list-of-arrays form` comparison |
| L129-133 | `_evaluate_polynomial_4d_xp` | that a previous ``xp.stack(T)`` step is gone |
| L687-695 | `clear_jax_ift_solver_cache` | that the singleton was NOT registered with the central cache-clearer registry before S4-15 |
| L893-900 | `_differentiable_lstsq` -- why not the normal equations | the `This function used to solve` framing and a comment correcting an earlier COMMENT (the documented '~1e-10 relative' shift that was wrong by eight orders) |
| L903 | `_differentiable_lstsq` -- the estimator table | the `(old)` tag and the `<-- the CI/local red` annotation on the first row |
| L909-916 | `_differentiable_lstsq` -- the two failures | the past-tense framing (`the floor was`, `it did not`, `could never have reached`) |
| L918-930 | `_differentiable_lstsq` -- physical size of the old error | the whole paragraph: it measures the SUPERSEDED estimator's accuracy against the current one |
| L967-971 | `fit_canonical_polynomials_jax` -- Limitations | the `the former auto-enable` framing |

---

### L3-5 -- `<module>` -- the v5.1.0 file-split provenance

*Left in the source:* the inventory of what the module holds

```text
v5.1.0 file-split (Agent D):  extracted from
``lumenairy.propagators.asymptotic`` with NO public-API or physics
change.  Holds:
```

### L65-76 -- `_require_jax_x64` -- the pre-fix account of `fit_canonical_polynomials_jax` auto-enabling x64 by mutating a global mid-call

*Left in the source:* why float64 is required (measured: ~5 % coefficient error and NaN gradients), and why this RAISES instead of auto-enabling -- which is the instruction a future editor needs

```text
    v5.17.x (audit P3-52): replicates
    :func:`lumenairy.elements.rcwa._core._require_jax_x64` (kept local
    to avoid a propagators -> elements.rcwa import edge).  The
    asymptotic twins need float64: single-precision ``lstsq`` gives
    ~5% coefficient error and NaN gradients in the canonical fit, and
    the Newton-IFT / Strehl-coefficient evaluations degrade silently.
    Pre-fix, :func:`fit_canonical_polynomials_jax` auto-enabled x64 by
    mutating the global ``jax_enable_x64`` MID-CALL -- unsafe when the
    caller jits the surrounding computation (JAX documents mid-trace
    config mutation as undefined behaviour) -- while the other three
    twins had no x64 handling at all.  A one-line caller setup replaces
    both: raise with instructions instead."""
```

### L109-116 -- `_chebyshev_vandermonde_xp` -- the `not the prior list-of-arrays form` comparison

*Left in the source:* what the shim forwards to, the return contract, and why it stays JAX-traceable

```text
    v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
    back-compat shim -- forwards to
    :func:`lumenairy._math.chebyshev.chebyshev_vandermonde` with the
    ``xp`` kwarg.  The returned array is now the same stacked-array
    contract as the NumPy helper (shape ``(max_k + 1,) + u.shape``),
    not the prior list-of-arrays form -- still JAX-traceable because
    the canonical implementation uses functional construction +
    ``xp.stack`` for non-NumPy backends.
```

### L129-133 -- `_evaluate_polynomial_4d_xp` -- that a previous ``xp.stack(T)`` step is gone

*Left in the source:* what the helper returns, which is why no stack is needed

```text
    # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
    # ``_chebyshev_vandermonde_xp`` (now a shim into _math.chebyshev)
    # already returns the stacked array directly, so the previous
    # ``xp.stack(T)`` step is gone.  3.5.6 vectorised-across-basis
    # construction is preserved.
```

### L687-695 -- `clear_jax_ift_solver_cache` -- that the singleton was NOT registered with the central cache-clearer registry before S4-15

*Left in the source:* what the singleton pins and which entry points now reclaim it

```text
    v5.24.x (audit S4-15): the ``_JAX_IFT_SOLVER_CACHE`` singleton pins
    the compiled XLA executable for the decorated solver for the life of
    the process.  Pre-fix it was NOT registered with the central
    cache-clearer registry, so ``clear_asm_caches`` /
    ``clear_all_registered_caches`` left the compiled solver (and its
    XLA device memory) resident even when the caller explicitly asked to
    drain every cache.  Registering this clearer lets a
    ``lumenairy_context(clear_caches_on_exit=True)`` or an explicit
    ``clear_asm_caches()`` reclaim it.
```

### L893-900 -- `_differentiable_lstsq` -- why not the normal equations -- the `This function used to solve` framing and a comment correcting an earlier COMMENT (the documented '~1e-10 relative' shift that was wrong by eight orders)

*Left in the source:* the alternative it rules out, the fit it was measured on, and the full conditioning figures

```text
    v5.29 (audit W4-T3) -- WHY NOT THE NORMAL EQUATIONS.  This function
    used to solve ``(A^H A + floor·I) x = A^H b`` with
    ``floor = 1e-12·(trace(A^H A)/n + 1)``, documented as shifting the
    solution by "~1e-10 relative".  MEASURED on the validation harness's
    own fit (singlet R1 = 20 mm, ``n_field=4, n_pupil=8, poly_order=4``:
    ``A`` is 1024x70, full rank 70/70, ``sigma_max = 47.68``,
    ``sigma_min = 7.29e-6``, so ``cond(A) = 6.54e+06`` and
    ``cond(A^H A) = 4.28e+13``) that claim was wrong by eight orders:
```

### L903 -- `_differentiable_lstsq` -- the estimator table -- the `(old)` tag and the `<-- the CI/local red` annotation on the first row

*Left in the source:* the row itself: the measured error of that floor

```text
        normal eq, floor 2.257e-10 (old)   4.4898e-02   <-- the CI/local red
```

### L909-916 -- `_differentiable_lstsq` -- the two failures -- the past-tense framing (`the floor was`, `it did not`, `could never have reached`)

*Left in the source:* both failure mechanisms and their numbers -- this is the argument that stops the normal equations being reinstated

```text
    Two independent failures compounded.  (1) The floor was 4.25x LARGER
    than ``sigma_min^2 = 5.31e-11``, so it did not "keep the solve finite"
    -- it DOMINATED the two smallest singular directions (the spectrum
    drops 2.48e-3 -> 8.01e-6 -> 7.29e-6, i.e. two near-null directions).
    (2) Even with the floor removed entirely, squaring a 6.5e+06 condition
    number leaves the normal equations ~1e-3 short of ``lstsq``, so
    shrinking the floor could never have reached the 1e-5 the validation
    check asks for.  QR fixes both at the same cost class.
```

### L918-930 -- `_differentiable_lstsq` -- physical size of the old error -- the whole paragraph: it measures the SUPERSEDED estimator's accuracy against the current one

*Left in the source:* where the discrepancy lives, which is the part that still explains the numbers in the table above

```text
    Physical size of the old error, measured the same way (see
    ``validation/propagators/test_asymptotic.py::
    t_fit_canonical_polynomials_jax_matches_numpy``): the two coefficient
    vectors fitted the TRAINING samples equally well (1.01e-07 waves apart,
    RMS residual 2.6128e-07 vs 2.6250e-07) and agreed to 1.15e-05 waves on
    an INDEPENDENT physically-reachable ray set -- the 4.5e-02 lived almost
    entirely in the near-null directions, i.e. the corners of the
    normalised box that the trace never reaches, where both fits are pure
    extrapolation (uniform-random box points: 1.87e-01 waves apart).  So
    the old behaviour was a small real accuracy loss (truth error on the
    independent set 1.21e-05 waves against NumPy's 9.81e-07) wearing a
    scary-looking coefficient number.  Post-fix the coefficients agree to
    ~1e-10 and the distinction is moot.
```

### L967-971 -- `fit_canonical_polynomials_jax` -- Limitations -- the `the former auto-enable` framing

*Left in the source:* the requirement, the one-line setup, and why it is not done automatically

```text
    * Requires ``jax_enable_x64`` (raises ``RuntimeError`` otherwise).
      v5.17.x (audit P3-52): the former auto-enable mutated the global
      ``jax_enable_x64`` MID-CALL, which is unsafe inside ``jax.jit``
      (undefined behaviour per the JAX docs).  Enable it once at
      import: ``jax.config.update('jax_enable_x64', True)``.
```
