<!-- lumenairy-history-doc
module: lumenairy/raytrace/differential.py
ast_sha256: a145518952f85d14350db8b28dd3e1df14ba11958cc9209ddbceafe3c0412efa
token_sha256: 69e733620bc218c598099db436317e5631657340a34961a036cd4d405716e7ca
pre_relocation_lines: 1078
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B9 item 6: _adrt_step handles even-power aspheres (conic seed + 6-step differentiated Newton in _adrt_aspheric_intersect, polynomial gradient in the normal); the analytic Jacobian no longer raises for aspheric_coeffs; the numba kernel is excluded for aspheres
re_recorded: 2026-09-14 -- WP-B12: ray_transfer_jacobian / _analytic take reference='surface'|'exit_vertex'; one shared projection (_project_to_exit_vertex_plane) re-references the base-ray state AND the 4x4 Jacobian onto the last surface's vertex plane, with a structural flat-surface short-circuit so a flat last surface stays bit-identical
-->


# Version history -- `lumenairy/raytrace/differential.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/differential.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Three blocks.  The NaN-propagation rule for the numba dual-number sqrt is
live and load-bearing -- clamping a NaN radicand makes the numba and NumPy
backends disagree about which rays are faulted -- so the whole argument
stayed, re-stated as what the clamping form does.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L361-361 | `_apply_coord_break_dual` | "the measured pre-fix numbers" |
| L670-676 | `_dual_sqrt_numba` | the "pre-fix ternary" / "Measured pre-fix" framing |
| L1044-1048 | `differential_ray_trace_jax / _full` | the release/audit tag and "this path used to walk" |

---

### L361-361 -- `_apply_coord_break_dual` -- "the measured pre-fix numbers"

*Left in the source:* the transpose convention, the KB citation and the pointer to the sibling derivation.

```text
        # the derivation and the measured pre-fix numbers).
```

### L670-676 -- `_dual_sqrt_numba` -- the "pre-fix ternary" / "Measured pre-fix" framing

*Left in the source:* the whole NaN-propagation requirement and the backend-disagreement it prevents, with both measured values.

```text
        # is False, so the pre-fix ternary clamped a NaN radicand to
        # ``vc = 0.0`` and returned a perfectly finite ``0.0`` value with
        # a huge-but-finite tangent -- while ``_dual_sqrt``'s
        # ``np.maximum(nan, 0.0)`` is ``nan`` (numpy's maximum
        # propagates NaN), giving ``nan`` value AND ``nan`` tangent
        # (``d / (2 * np.maximum(nan, 1e-300))``).  Measured pre-fix:
        # numpy nan vs numba 0.0.  A silent 0.0 turns an already-faulted
```

### L1044-1048 -- `differential_ray_trace_jax / _full` -- the release/audit tag and "this path used to walk"

*Left in the source:* the single-pass argument, which is the reason for the aux plumbing.

```text
        R7 (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11): this path used to
        walk the whole prescription TWICE -- ``jax.jacfwd(_state)`` for
        the Jacobian and a second ``vmap(_full)`` for the state and OPL.
        ``jax.jacfwd(..., has_aux=True)`` returns the primal outputs of
        the same forward pass alongside the Jacobian, so the second walk
```

