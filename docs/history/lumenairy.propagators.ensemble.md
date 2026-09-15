<!-- lumenairy-history-doc
module: lumenairy/propagators/ensemble.py
ast_sha256: ca6b386818504085a9fc0b305b2765933f1f12721f2ffd0d4fe21abff0af4569
token_sha256: d6c93fb6b56c90536ac0177760bc631640e01299058f0b896abd221dbc0ba5a2
pre_relocation_lines: 440
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/ensemble.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/ensemble.py`.  Each block is reproduced **verbatim** under the source line it came
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
| L87-92 | `_unwrap_field` | the `v4.16.2 ... Pre-fix every branch coerced through np.asarray(...)` framing |
| L261-266 | `propagate_ensemble` -- the kwargs-collision check | the `Pre-fix a caller passing ... triggered` framing |
| L277-285 | `propagate_ensemble` -- backend-preserving dispatch | the `Pre-fix the duck-typed np.asarray(ensemble) fallback below silently transferred` framing |
| L315-320 | `propagate_ensemble` -- the empty-ensemble guard | the `pre-fix the downstream I_acc / float(0) raised` framing |
| L348-358 | `_resolve_accumulator_dtype` | the `Pre-v4.16.3 the earlier gate ... was structurally dead` framing |

---

### L87-92 -- `_unwrap_field` -- the `v4.16.2 ... Pre-fix every branch coerced through np.asarray(...)` framing

*Left in the source:* the contract (the backend is preserved) and what coercing here would cost -- a silent host transfer of a CuPy / JAX return

```text
    v4.16.2 (audit P1-NEW-F1-2): preserve the backend of the returned
    field.  Pre-fix every branch coerced through ``np.asarray(...)``,
    silently transferring CuPy / JAX returns to host NumPy.  Now we
    return the underlying array as-is (NumPy, CuPy, or eager
    ``jax.Array``); the per-realisation accumulator runs through the
    matching ``xp.*`` namespace.
```

### L261-266 -- `propagate_ensemble` -- the kwargs-collision check -- the `Pre-fix a caller passing ... triggered` framing

*Left in the source:* what the check replaces and why the bare Python error is not good enough

```text
    # v4.16.2 (audit P3-NEW-F1-2): explicit kwargs-collision check.
    # Pre-fix a caller passing ``propagator_kwargs={'dx': ...}`` plus a
    # positional ``dx=...`` triggered the bare Python ``TypeError: got
    # multiple values for keyword argument 'dx'`` -- correct but
    # confusing.  Raise a domain-level ``ValueError`` with a clear
    # remediation pointer.
```

### L277-285 -- `propagate_ensemble` -- backend-preserving dispatch -- the `Pre-fix the duck-typed np.asarray(ensemble) fallback below silently transferred` framing

*Left in the source:* the dispatch rule, what the naive fallback would defeat, and the one input class that DOES still take it

```text
    # v4.16.2 (audit P1-NEW-F1-2): backend-preserving dispatch.
    # Pre-fix the duck-typed ``np.asarray(ensemble)`` fallback below
    # silently transferred CuPy ensembles to host NumPy and forced
    # concretisation of JAX arrays, defeating any GPU / autodiff
    # workflow the user built upstream.  Now we detect the backend via
    # ``array_namespace`` and run the accumulator on the same xp.
    # Inputs that aren't a NumPy / CuPy / JAX array (e.g. a Python
    # list of 2-D arrays) fall back to ``np.asarray``; the warning
    # below documents the coercion.
```

### L315-320 -- `propagate_ensemble` -- the empty-ensemble guard -- the `pre-fix the downstream I_acc / float(0) raised` framing

*Left in the source:* why the ndim check is not enough and both ways an empty ensemble fails without this guard

```text
    # v4.16.2 (audit P3-NEW-F1-1): reject empty ensembles cleanly.
    # ``shape=(0, Ny, Nx)`` passes the ndim check above; pre-fix the
    # downstream ``I_acc / float(0)`` raised an opaque
    # ``ZeroDivisionError`` (or NaN-poisoned the result depending on
    # the dtype).  Surface the empty-ensemble case as a domain-level
    # ``ValueError`` at the entry point.
```

### L348-358 -- `_resolve_accumulator_dtype` -- the `Pre-v4.16.3 the earlier gate ... was structurally dead` framing

*Left in the source:* why the knob is on the canonical path rather than in the ``except`` branch, and what the ``try/except`` is still for

```text
    # v4.16.3 (audit P2-NEW-F1-3): re-shape the fallback so
    # ``get_default_real_dtype()`` is the canonical ``in_dtype is None``
    # path rather than an unreachable ``except`` branch.  Pre-v4.16.3
    # the earlier ``hasattr(ensemble, 'dtype')`` gate + ``np.asarray``
    # coercion (lines ~287-302) guaranteed ``getattr(ensemble, 'dtype',
    # None)`` always returned a valid numpy dtype by the time control
    # reached this site, so the ``except (TypeError, ValueError)``
    # branch was structurally dead and the ``set_default_real_dtype``
    # knob had no reachable consumer library-wide.  The ``try/except``
    # is retained as a belt-and-suspenders guard for the exotic-dtype
    # case (e.g. a future numpy dtype that doesn't expose ``.real``).
```
