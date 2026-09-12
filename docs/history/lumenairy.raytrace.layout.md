<!-- lumenairy-history-doc
module: lumenairy/raytrace/layout.py
ast_sha256: 9ec337cda3725e080b5b62bd88b1ffaf80c0444a867ee37dda55d3f5fe503c31
token_sha256: 78dd068b25b8998243594f8deb170065d3ff82df0676faf664bd691703973db1
pre_relocation_lines: 190
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/layout.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/layout.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Two blocks, both of the same shape: a bare built-in exception replaced by one
that names the function and the offending value.  The bare exception is what
the guard prevents, so it stayed.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L60-61 | `trace_summary units guard` | "used to surface" |
| L76-78 | `trace_summary empty-bundle guard` | "used to die here" |

---

### L60-61 -- `trace_summary units guard` -- "used to surface"

*Left in the source:* the diagnostic rule and the bare KeyError it replaces.

```text
    # function and the offending value.  A wrong ``units`` string used to
    # surface as a bare ``KeyError: 'cm'``.
```

### L76-78 -- `trace_summary empty-bundle guard` -- "used to die here"

*Left in the source:* the same rule for the empty bundle.

```text
    # S11-6d: an EMPTY bundle (n_total == 0) used to die here with a bare
    # ``ZeroDivisionError: division by zero`` from ``n_alive / n_total``,
    # naming neither the function nor the cause.
```

