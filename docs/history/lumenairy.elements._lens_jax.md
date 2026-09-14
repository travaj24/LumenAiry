<!-- lumenairy-history-doc
module: lumenairy/elements/_lens_jax.py
ast_sha256: 31f68294c61090e11b7e7b9c1aafe4fd4809878dcbcb36087aed126d4f3334d6
token_sha256: 2c31b2db8f3bbe7c7ec4077e68e38722b0504dff736ed662b117d8b41ea99103
pre_relocation_lines: 1021
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B7 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, WP-A4 sec. 6 items 3-8 + VERIFY-B1 F1/F2): the Y4 fused basis evaluation and hoisted Newton factor, the aberration_tensor mode/waist caches, the S6 gate's k1-slope statistic and mean-plus-spread chart sizing, the JAX screen's chief-ray displacement term, and the S9 FFT kernel clip
-->

# Version history -- `lumenairy/elements/_lens_jax.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/_lens_jax.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks, the comments that corrected earlier
comments, and the per-release chronologies that had accumulated on constants
whose CURRENT value is what the source now states.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

What did NOT move: the parameter contracts and accuracy statements
of the public JAX entry points (the `cheb_order` / `newton_iters` residual
figures, the `amplitude='input'`/`'analytic'` gradient-flow contract, the
geometry-gradient requirements, and the "this path is not row-banded, budget
for the monolithic cost" memory note) -- those describe what the code does now.


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
| L305-309 | `_resolve_complex_dtype` | "Pre-fix users who set ``set_default_complex_dtype(np.complex128)`` could still hit complex64 here" |
| L420-424 | `apply_real_lens_traced_jax` docstring, ``prescription`` | the parenthetical recording that this parameter was DOCUMENTED as ``lens_prescription`` before v5.30 and that the documented call form raised ``TypeError`` |
| L513-517 | the mirror-in-surfaces guard (traced JAX entry) | "Pre-fix a hand-built prescription with ``surfaces[i]['is_mirror']=True`` would slip past" |
| L666-673 | the tracer-safe initial-guess magnification | "keeps the geometry gradient identical to the pre-fix ``_diff_geom`` branch" -- the S7 fix written against the state it replaced |
| L699-707 | the complex-dtype resolution at the OPL/amplitude combine | two stacked release notes (v4.13.0 L2 then v4.14.0) on the same resolver call, the second recording the silent complex64 -> complex128 upcast the first shipped with |

---

### L305-309 -- `_resolve_complex_dtype` -- "Pre-fix users who set ``set_default_complex_dtype(np.complex128)`` could still hit complex64 here"

*Left in the source:* the rule -- one knob, the library-wide default -- and the divergence reading `jax_enable_x64` instead produces

```text
    # v4.13.0 (audit L2): unify on the library-wide default dtype rather
    # than reading ``jax.config.jax_enable_x64``.  Pre-fix users who set
    # ``set_default_complex_dtype(np.complex128)`` could still hit
    # complex64 here when ``jax_enable_x64`` was left at its default
    # False -- inconsistent with the NumPy-side ``apply_real_lens``.
```

### L420-424 -- `apply_real_lens_traced_jax` docstring, ``prescription`` -- the parenthetical recording that this parameter was DOCUMENTED as ``lens_prescription`` before v5.30 and that the documented call form raised ``TypeError``

*Left in the source:* the parameter's own one-line description

```text
        Same format as :func:`apply_real_lens`.  (Documented as
        ``lens_prescription`` before v5.30 -- audit
        AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 Territory A: that name is
        the function's INTERNAL alias, never an accepted keyword, so the
        documented call form raised ``TypeError``.)
```

### L513-517 -- the mirror-in-surfaces guard (traced JAX entry) -- "Pre-fix a hand-built prescription with ``surfaces[i]['is_mirror']=True`` would slip past"

*Left in the source:* the hazard itself, present tense

```text
    # v4.13.0 (audit L4a): port the explicit mirror-in-surfaces guard
    # from ``apply_real_lens_traced``.  Pre-fix a hand-built prescription
    # with ``surfaces[i]['is_mirror']=True`` would slip past, and the
    # ray-traced OPD leg would silently treat the mirror as a refractor
    # with the wrong sign.
```

### L666-673 -- the tracer-safe initial-guess magnification -- "keeps the geometry gradient identical to the pre-fix ``_diff_geom`` branch" -- the S7 fix written against the state it replaced

*Left in the source:* why the guess is taken tracer-safe on BOTH branches, and why `stop_gradient` costs nothing

```text
    # S7 (audit): the initial-guess magnification is taken tracer-safe on BOTH
    # branches.  The ``float(x_out_grid[...])`` the static branch used raised
    # ConcretizationTypeError under ``jax.jit``, so the DEFAULT path of this
    # function -- the one the docstring advertises as "vmap+JIT replaces the
    # [NumPy] pool" -- could not be jitted at all.  The Newton root does not
    # depend on the starting point, so ``stop_gradient`` keeps the geometry
    # gradient identical to the pre-fix ``_diff_geom`` branch, and the same
    # float64 arithmetic makes the static branch's guess unchanged.
```

### L699-707 -- the complex-dtype resolution at the OPL/amplitude combine -- two stacked release notes (v4.13.0 L2 then v4.14.0) on the same resolver call, the second recording the silent complex64 -> complex128 upcast the first shipped with

*Left in the source:* both live rules: one knob for the dtype decision, and the input dtype is honoured

```text
    # v4.13.0 (audit L2): unify on the library-wide default dtype
    # (``set_default_complex_dtype``) rather than reading the JAX
    # global ``jax_enable_x64``.  Two different knobs for the same
    # decision is exactly the divergence audit L2 flagged.
    # v4.14.0: pass ``E_in.dtype`` so the input dtype is honoured.
    # Pre-v4.14 the resolver returned the library default, silently
    # upcasting a complex64 input to complex128 whenever
    # ``jax_enable_x64=True``.  Caught by parametrized dispatcher pin
    # in v4.14.0 Agent 6.
```
