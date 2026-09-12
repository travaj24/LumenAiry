<!-- lumenairy-history-doc
module: lumenairy/elements/bor/bor_stack.py
ast_sha256: 6fbc5434ec50b0349770400fa695a05dcecff912e54da9b72c8db874c5425d7e
token_sha256: 63e2a92215dbd93b0658e7d9659db1788326365720afe58cd2016c27fc377b4c
pre_relocation_lines: 1031
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
-->

# Version history -- `lumenairy/elements/bor/bor_stack.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/bor/bor_stack.py` -- the "previously mixed" note on the
mode/order terminology, the W6-B7 clause on the two-source-argument guard, and
the AUDIT_BOR_PROPAGATING_CUTOFF paragraph recording the superseded ANGULAR
cutoff constant.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

What did NOT move: the terminology block itself (`tests/unit/
test_audit_v5_24_2_b2_bor_exports.py` reads `azimuthal order`, `mode`, `order`
and `S5-6` out of the class docstring), and the measured energy leak that sizes
the real-axis floor.

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
| L86 | `BORStack` docstring, the terminology block | "previously mixed" |
| L502 | `BORStack.set_source`, the source guard | "the wavelength used to be silently discarded" |
| L673-681 | `BORStack`, the propagating-order floor | the superseded 0.05 ANGULAR cutoff and the bit-identity it was chosen for |

---

### L86 -- `BORStack` docstring, the terminology block -- "previously mixed"

*Left in the source:* the terminology contract and its audit id, which is what the exported-API test pins.

```text
    API (audit S5-10 / B2 -- previously mixed):
```

### L502 -- `BORStack.set_source`, the source guard -- "the wavelength used to be silently discarded"

*Left in the source:* the same failure as the reason both arguments are rejected together.

```text
        # W6-B7: give BOTH and the wavelength used to be silently discarded.
```

### L673-681 -- `BORStack`, the propagating-order floor -- the superseded 0.05 ANGULAR cutoff and the bit-identity it was chosen for

*Left in the source:* why the floor is on the REAL AXIS and not an angle, and the measured energy leak an angular cutoff produces.

```text
            # AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13: the original
            # P2-06 constant (0.05, chosen to keep k0=2.0 bit-identity with
            # the pre-fix absolute threshold) was an ANGULAR cutoff -- it
            # dropped genuinely propagating near-grazing orders (theta up to
            # 88 deg in n=1.41), silently biasing per-order R/T low and
            # leaking energy (2.28e-2 on the ring-grating reproducer).  A
            # propagating mode is real-q up to the q ~ 0 degenerate point,
            # so the real-axis floor guards ONLY that point (1e-6).  This
            # floor is compatible with the flux normalizer's fallback branch
```
