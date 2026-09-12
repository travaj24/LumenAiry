<!-- lumenairy-history-doc
module: lumenairy/analysis/phase_retrieval.py
ast_sha256: e9a4bb26daae96d6440a83423fba33f47bd8d6bcaa754673c07290e7e5c14db7
token_sha256: 354319b21a0d5e42f54f37404872c83ecdc8e073435825d8ab9c23d04587e88c
pre_relocation_lines: 1068
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/phase_retrieval.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/phase_retrieval.py`.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

Every block here is the same shape, and it is the shape the WP-A17 SWEEP-1
follow-up pass was asked to close: a **live guard whose comment explained
itself by naming the release that added it** ("Pre-fix X happened", "Pre-4.12
the dispatcher only passed ...").  The hazard X is still reachable -- the guard
is the only thing preventing it -- so the source now states X in the present
tense, as what goes wrong WITHOUT the guard, together with every measurement
that sizes it.  What moved is the release attribution and the
"bit-identical to pre-fix" reassurance that travelled with it.

Two of the blocks are parameter documentation rather than guards, and they
are the V6 pattern in its purest form: `seed` and `dtype` each carried a
paragraph describing what the parameter did BEFORE it worked, inside the
entry that documents what it does.  A caller reading `seed : int, optional`
met three sentences about v4.11.2 before reaching the one that says what
passing an int does.

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
| L132-146 | `gerchberg_saxton` -- the JAX dispatch | the `Pre-4.12 the dispatcher only passed n_iter` and `Pre-fix the dispatcher silently dropped return_history` framings |
| L317-326 | `error_reduction` -- the `initial_guess` refusal | the `Pre-fix the kwarg was silently dropped, so two calls ... produced identical trajectories` framing |
| L368-370 | `error_reduction` -- `seed` | the `pre-4.11.2 the kwarg didn't exist on this path` parenthetical |
| L390-394 | `error_reduction` -- the 1e-30 epsilon | the `the old path would have produced` framing |
| L549-550 | `hybrid_input_output` -- `seed` / `dtype` | the `pre-4.11.2 the kwargs didn't exist on this path` parenthetical |
| L601-607 | `<module>` -- the module-scope kernel cache | the `Pre-4.12 the iteration body lived inside a closure` framing |
| L694-696 | `gerchberg_saxton_jax` -- the final error | the `pre-fix the metric used the far field carried out of the loop` framing |
| L768-774 | `gerchberg_saxton_jax` -- `seed` | the `4.11.2: now actually seeds ... Pre-4.11.2 the kwarg was accepted but ignored (``_ = seed`` with no consumer)` paragraph |
| L780-792 | `gerchberg_saxton_jax` -- `dtype` | the parenthetical describing the pre-fix ``TypeError: float() argument ... not 'complex'``, and the `Before this it was float32 unconditionally` paragraph |
| L856-860 | `gerchberg_saxton_jax` -- the complex-dtype pairing | the `pre-fix this hard-cast to jnp.complex64 silently demoted` framing |
| L914-916 | `hybrid_input_output_jax` -- `seed` / `dtype` | the `(was hard-coded to seed=0 pre-4.10)` parenthetical |

---

### L132-146 -- `gerchberg_saxton` -- the JAX dispatch -- the `Pre-4.12 the dispatcher only passed n_iter` and `Pre-fix the dispatcher silently dropped return_history` framings

*Left in the source:* which kwargs must be forwarded and why, and the whole ``return_history`` contract -- the warning, the synthesised empty history, and the instruction to use ``backend='numpy'`` for real history

```text
        # 4.12.0 (audit round-4 B2-6): forward all reproducibility /
        # precision kwargs to the JAX path.  Pre-4.12 the dispatcher
        # only passed ``n_iter``, silently dropping ``seed``,
        # ``initial_phase``, and ``dtype``.  Function-level kwargs on
        # gerchberg_saxton_jax were wired correctly internally; the
        # unified front door just didn't forward them.
        #
        # v4.13.0 (audit L4c): the JAX twin returns ``(phase, err)`` --
        # it does not support host-side per-iteration error capture
        # (``return_history=True``).  Pre-fix the dispatcher silently
        # dropped ``return_history``, so a user expecting a 3-tuple
        # received a 2-tuple with no warning.  Now we emit a
        # ``RuntimeWarning`` and synthesise an empty history list so
        # the return shape always matches the NumPy API.  Users who
        # need real history must use ``backend='numpy'``.
```

### L317-326 -- `error_reduction` -- the `initial_guess` refusal -- the `Pre-fix the kwarg was silently dropped, so two calls ... produced identical trajectories` framing

*Left in the source:* why the conversion is lossy and why this refuses rather than demoting -- the reason the ``NotImplementedError`` is correct

```text
        # v4.13.0 (audit L4b): the NumPy API takes ``initial_guess``
        # (a complex object-field starting point); the JAX twin
        # ``error_reduction_jax`` takes ``init_phase`` (a real-valued
        # initial Fourier-phase array).  Converting between them is
        # lossy: phase = np.angle(initial_guess) discards amplitude
        # info and changes the iteration trajectory.  Rather than
        # silently demote, raise ``NotImplementedError`` so the user
        # makes an explicit choice.  Pre-fix the kwarg was silently
        # dropped, so two calls with different ``initial_guess``
        # produced identical (random-init) trajectories.
```

### L368-370 -- `error_reduction` -- `seed` -- the `pre-4.11.2 the kwarg didn't exist on this path` parenthetical

*Left in the source:* what the kwarg buys and the manual alternative

```text
    # 4.11.2: honour `seed` for reproducibility (pre-4.11.2 the kwarg
    # didn't exist on this path; users wanting deterministic runs had
    # to construct `initial_guess` manually).
```

### L390-394 -- `error_reduction` -- the 1e-30 epsilon -- the `the old path would have produced` framing

*Left in the source:* the algebraic identity, what the epsilon preserves, and the one pathological coincidence where the two forms differ

```text
    # measured_amplitude > 0.  In that pathological case the old path
    # would have produced ``measured_amplitude * (1+0j)`` (because
    # np.angle(0+0j) == 0) -- but we accept the new ``~0`` here because
    # |F| < 1e-30 means the iterate has functionally collapsed; both
    # behaviours are valid limits.  JAX path untouched.
```

### L549-550 -- `hybrid_input_output` -- `seed` / `dtype` -- the `pre-4.11.2 the kwargs didn't exist on this path` parenthetical

*Left in the source:* what the kwargs control

```text
    # 4.11.2: honour `seed` / `dtype` for reproducibility + precision
    # control (pre-4.11.2 the kwargs didn't exist on this path).
```

### L601-607 -- `<module>` -- the module-scope kernel cache -- the `Pre-4.12 the iteration body lived inside a closure` framing

*Left in the source:* what the cache buys and the cost of a per-call closure, which is the reason not to inline the kernels again

```text
# v4.12 perf: each outer driver below builds its iteration kernel at
# module scope (parameterised via a small cache keyed on n_iter and any
# scalar Python knobs).  Pre-4.12 the iteration body lived inside a
# closure that was re-created on every call -- the ``lax.fori_loop``
# inside was jit-traced by JAX, but the outer wrapper paid a fresh
# dispatch each invocation.  With the module-scope cache, repeated
# calls with the same n_iter reuse the same compiled XLA executable.
```

### L694-696 -- `gerchberg_saxton_jax` -- the final error -- the `pre-fix the metric used the far field carried out of the loop` framing

*Left in the source:* which iterate the error is measured on, and the mistake that phrasing guards against

```text
        # Final error from the far field of the FINAL iterate, matching
        # the NumPy path's post-loop re-transform (pre-fix the metric
        # used the far field carried out of the loop = previous iterate).
```

### L768-774 -- `gerchberg_saxton_jax` -- `seed` -- the `4.11.2: now actually seeds ... Pre-4.11.2 the kwarg was accepted but ignored (``_ = seed`` with no consumer)` paragraph

*Left in the source:* what each value of the parameter does

```text
        4.11.2: now actually seeds the random initial-phase draw.
        Pre-4.11.2 the kwarg was accepted but ignored (``_ = seed``
        with no consumer), so two calls with different seeds produced
        the same trajectory.  Pass ``None`` (default) for a uniformly-
        zero initial phase (matches the historical deterministic
        behaviour); pass an int to draw an i.i.d. uniform initial
        phase that randomises the iteration start.
```

### L780-792 -- `gerchberg_saxton_jax` -- `dtype` -- the parenthetical describing the pre-fix ``TypeError: float() argument ... not 'complex'``, and the `Before this it was float32 unconditionally` paragraph

*Left in the source:* the live rule for a complex dtype, the x64 convention, the measured precision gap that makes it matter, and how to pin float32

```text
        always a real float.  (Pre-fix a complex ``dtype`` fell through
        to the defensive branch, made ``src``/``tgt`` complex, and the
        call died in ``float(err)`` with
        ``TypeError: float() argument must be ... not 'complex'``.)

        ``None`` (default) follows JAX's own x64 convention: float64
        when ``jax.config.jax_enable_x64`` is enabled, float32 otherwise.
        Before this it was float32 unconditionally, so a caller who had
        turned x64 on still got a single-precision answer and a ~1e-6
        error floor while the NumPy twin reached ~1e-14 -- the two
        backends documented "the same physics" and did not agree to more
        than six digits. Pass an explicit ``np.float32`` to pin the
        historical behaviour regardless of the global flag.
```

### L856-860 -- `gerchberg_saxton_jax` -- the complex-dtype pairing -- the `pre-fix this hard-cast to jnp.complex64 silently demoted` framing

*Left in the source:* the pairing rule and what a hard cast would cost a caller who asked for float64

```text
    # v4.13.0 (audit L2): pre-fix this hard-cast to ``jnp.complex64``
    # silently demoted the iteration state to single precision even when
    # the user passed ``dtype=np.float64`` for ground-truth comparison.
    # The complex dtype is paired with the real ``dtype``: complex64
    # for float32, complex128 for float64.
```

### L914-916 -- `hybrid_input_output_jax` -- `seed` / `dtype` -- the `(was hard-coded to seed=0 pre-4.10)` parenthetical

*Left in the source:* what each kwarg selects

```text
    4.10: ``seed`` controls the random initial phase (was hard-coded
    to seed=0 pre-4.10).  ``dtype`` selects float32 (default) or
    float64 (matches NumPy precision).
```
