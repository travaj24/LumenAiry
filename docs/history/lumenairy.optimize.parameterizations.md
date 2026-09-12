<!-- lumenairy-history-doc
module: lumenairy/optimize/parameterizations.py
ast_sha256: 2125df6287c984b5ae3d094a3eb211e5782dddc9e2ff783aa96783f12fa04835
token_sha256: 52a7cf64fe9fd1e93dd8c8594c4c494d6115c72b24029c7660f06640e987e39b
pre_relocation_lines: 479
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/optimize/parameterizations.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/parameterizations.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Every block here is the same release tag -- `v5.2 (AUDIT_V4_13_1 P1-1
closure)` -- stamped on the per-variable FD scale-floor machinery at nine
sites.  The machinery is entirely live: the floors, the path-classification
table and the broadcast rules all stayed in the source.  Only the tag moved,
plus one clause describing the predicate that preceded the current regex.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L21-22 | `_DEFAULT_SCALE_FLOORS` | the release/audit tag |
| L49-49 | `_classify_path_to_floor` | the release/audit tag |
| L92-96 | `_classify_path_to_floor` | the audit tag and the "the old startswith('a')" framing |
| L134-134 | `PrescriptionParameterization` | the release/audit tag |
| L144-145 | `PrescriptionParameterization.__post_init__` | two nested release/audit tags |
| L178-178 | `PrescriptionParameterization.__post_init__` | the release/audit tag |
| L185-185 | `PrescriptionParameterization._resolve_scale_floor` | the release/audit tag |
| L378-378 | `MultiPrescriptionParameterization` | the release/audit tag |
| L397-397 | `MultiPrescriptionParameterization.__post_init__` | the release/audit tag |
| L428-428 | `MultiPrescriptionParameterization.__post_init__` | the release/audit tag |
| L435-435 | `MultiPrescriptionParameterization._resolve_scale_floor` | the release/audit tag |

---

### L21-22 -- `_DEFAULT_SCALE_FLOORS` -- the release/audit tag

*Left in the source:* the whole reason the floors exist -- a near-zero parameter collapses ``x_scale[i]`` and the optimiser's step-size logic divides through to a sub-eps relative step.

```text
# v5.2 (AUDIT_V4_13_1 P1-1 closure): per-variable-type scale floors used
# by the driver's finite-difference Hessian estimator.  Without an
```

### L49-49 -- `_classify_path_to_floor` -- the release/audit tag

*Left in the source:* the mapping contract and every accepted path form.

```text
    """v5.2 (AUDIT_V4_13_1 P1-1 closure): map a free-var path tuple to
```

### L92-96 -- `_classify_path_to_floor` -- the audit tag and the "the old startswith('a')" framing

*Left in the source:* the false-positive hazard in full, re-stated as what a bare ``startswith('a')`` WOULD match -- which is why the regex must not be loosened.

```text
        # (``A4`` / ``a_8`` / ``a12``).  Nit (AUDIT_OPTIMIZE_DRIVER): the old
        # ``key_lc.startswith('a')`` matched ANY key beginning with 'a' (a
        # future ``'axis'`` / ``'angle'`` surface field would silently pick up
        # the dimensionless aspheric FD floor); ``re.fullmatch(r'a_?\d+')``
        # keeps the intended A-coefficient names without the false positives.
```

### L134-134 -- `PrescriptionParameterization` -- the release/audit tag

*Left in the source:* the whole field contract, including the driver read-site and the auto-fill rule.

```text
    # v5.2 (AUDIT_V4_13_1 P1-1 closure): per-parameter absolute scale
```

### L144-145 -- `PrescriptionParameterization.__post_init__` -- two nested release/audit tags

*Left in the source:* the guard and its entire failure description -- separate x[i] slots writing one field, last-write-wins, a dead variable, and a split FD gradient.

```text
        # v5.17.x (AUDIT_V5_17_0 P3-50): mirror the v4.14 (audit P3
        # #19) duplicate-free_vars guard from
```

### L178-178 -- `PrescriptionParameterization.__post_init__` -- the release/audit tag

*Left in the source:* the resolution rule for None / scalar / array.

```text
        # v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve ``scale_floor`` to a
```

### L185-185 -- `PrescriptionParameterization._resolve_scale_floor` -- the release/audit tag

*Left in the source:* the contract.

```text
        """v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve a user-supplied
```

### L378-378 -- `MultiPrescriptionParameterization` -- the release/audit tag

*Left in the source:* the auto-fill rule and the inner-path convention.

```text
    # v5.2 (AUDIT_V4_13_1 P1-1 closure): per-parameter absolute scale
```

### L397-397 -- `MultiPrescriptionParameterization.__post_init__` -- the release/audit tag

*Left in the source:* the whole over-parameterisation argument.

```text
        # v4.14 (audit P3 #19): duplicate (prescription_index, *path)
```

### L428-428 -- `MultiPrescriptionParameterization.__post_init__` -- the release/audit tag

*Left in the source:* the inner-path resolution rule.

```text
        # v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve ``scale_floor`` to
```

### L435-435 -- `MultiPrescriptionParameterization._resolve_scale_floor` -- the release/audit tag

*Left in the source:* the contract.

```text
        """v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve a user-supplied
```

