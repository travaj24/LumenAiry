<!-- lumenairy-history-doc
module: lumenairy/elements/eme/eme_2d.py
ast_sha256: d29860b978f3916ae627112a7a8d9781852425d03d61cba1af4352ab50993299
token_sha256: f6c86f480a7584f9c6dd2f29cd2d02d0e760cce49f83ed05e14c4dfb18861c0a
pre_relocation_lines: 503
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/eme/eme_2d.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/eme/eme_2d.py` -- the two W6 clauses recording what the
pre-fix `eig` route and the inert `sigma` argument did.  Each block is
reproduced **verbatim** under the source line it came from in the pre-relocation
file.

What did NOT move: the measured garbage census (68 modes returned, 0/3 real
modes recovered), the growing-propagator argument, and the refusal message.

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
| L65-66 | `layer_modes` docstring | "(the pre-fix behaviour)" and the past tense around it |
| L414-416 | `ref_2d_modes`, the ``sigma`` guard | "sigma was silently INERT" |

---

### L65-66 -- `layer_modes` docstring -- "(the pre-fix behaviour)" and the past tense around it

*Left in the source:* the whole damage argument as the reason real ``eps`` at ``kx0 != 0`` does not go through ``eig``.

```text
    Sending real ``eps`` at ``kx0 != 0`` through ``eig`` (the pre-fix behaviour)
    was doubly damaging and made ``layer_modes`` return pure garbage there
```

### L414-416 -- `ref_2d_modes`, the ``sigma`` guard -- "sigma was silently INERT"

*Left in the source:* the inertness as a present fact, with the measurement and the consequence for the caller.  See docs/history/lumenairy.elements.eme.eme_2d.md.

```text
        # AUDIT W6: sigma was silently INERT without k (the dense path ignores it
        # entirely -- measured bit-identical results for sigma=1e9), so a caller
        # asking for modes near a shift got the whole dense spectrum instead.
```
