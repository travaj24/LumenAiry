<!-- lumenairy-history-doc
module: lumenairy/propagators/result.py
ast_sha256: 2ca9b8b0c05951c407739ac660a9693f26347808fca5532a4cc3a27a96f126a4
token_sha256: 3a2232e4f1b03a4b441d56e194916dbe48c755990affc89e4b05a571b30d9c96
pre_relocation_lines: 244
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/result.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/result.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

Three small blocks moved, all of them the reasoning that SETTLED the P16
2-item-iteration question rather than the settlement itself.  The settlement
stayed in all three places it appears (the class warning, the `.. note::`, and
`__iter__`'s own docstring), because it is a live, permanent contract and the
worked example of what each unpacking form does is exactly what a caller
needs.  What moved is the account of why the roadmap's F1 decision forced the
question in the same pass.

The module docstring's statement that `lumenairy.propagate` returns a
`PropagationResult` BY DEFAULT, with the native shapes still available
bit-identically through `return_result=False`, is a live contract and stayed
verbatim.

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
| L62-78 | `PropagationResult` -- the P16 note | the account of the deferred F1 decision and why settling P16 in the same pass was required |
| L90-93 | `PropagationResult.dy` | the `no longer silently discards` framing |
| L175-178 | `PropagationResult.__iter__` | the `did NOT change when propagate()'s default return became a PropagationResult` framing |

---

### L62-78 -- `PropagationResult` -- the P16 note -- the account of the deferred F1 decision and why settling P16 in the same pass was required

*Left in the source:* the decision itself, that nothing schedules a change, and both halves of the rationale (what re-arity-ing would break, and why no arity change is needed)

```text
       **P16 resolved (v5.30, roadmap Part F1).**  The deferred F1 decision
       -- ``propagate()``'s default return becoming a ``PropagationResult``
       for every method, EXECUTED in v5.30 -- was required to settle P16 in
       the same pass, because the wrapper's 2-item iteration and the
       kernels' 3-item tuple cannot both be "the" unpacking contract once
       the wrapper is the default.

       **Decision: iteration stays 2-item, permanently.**  It is NOT
       scheduled to become ``(field, dx_out, dy_out)`` at the flip, and no
       registry entry schedules such a change.  Rationale: the chosen F1
       option keeps ``return_result=False`` available past the flip, so a
       caller who unpacks ``E, dxo, dyo`` migrates by naming that contract
       -- no arity change needed.  Re-arity-ing :meth:`__iter__` would
       instead break the ``E, intermediates = propagate_through_system(...,
       return_result=True)`` callers this method exists for, i.e. trade one
       breakage for a new one, which is exactly what the least-breaking
       option was chosen to avoid.
```

### L90-93 -- `PropagationResult.dy` -- the `no longer silently discards` framing

*Left in the source:* which kernels report a distinct ``dy_out`` and what threads it here

```text
        Anamorphic Fresnel / Fraunhofer / SAS kernels return a
        distinct ``dy_out``; v4.13.0 (audit L3) threads that value
        through :func:`_coerce_field` so the wrapped result no longer
        silently discards the y-axis pitch.
```

### L175-178 -- `PropagationResult.__iter__` -- the `did NOT change when propagate()'s default return became a PropagationResult` framing

*Left in the source:* that the arity is not scheduled to change and what a 3-tuple unpacker passes

```text
        v5.30 (roadmap Part F1): this arity did NOT change when
        ``propagate()``'s default return became a ``PropagationResult``,
        and is NOT scheduled to change -- see the P16 note on the class.
        3-tuple unpackers pass ``return_result=False``.
```
