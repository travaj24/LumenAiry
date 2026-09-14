<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/stack2d.py
ast_sha256: 42ed5d704f6e95f29a9f2b2abdaa95dca2bfdaf2da8e5448b467648fab303629
token_sha256: 739b4a6036051436b3a88fefc24791651a73f08d1fb510f49b248aeece0aac62
pre_relocation_lines: 2261
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B6 (audit 2026-09-11 G10(d)): _geom_cache carries a tensor layer's k0-free projected operators (_tensor_projected_ops) beside the scalar lops, and _geom_key gains the formulation the cached EZZ rule depends on; every operator is bit-identical with and without the cache
re_recorded: 2026-09-13 -- WP-B11a: the Collins readout's K1 applicability window stated on the public transport docstring (item 20); PMM2DStackHybrid's formulation/cascade/symmetry refuse an out-of-vocabulary assignment (item 19); sampling= on the free-space HFPI pair (item 12)
re_recorded: 2026-09-14 -- PMM2DStackHybrid.truncation joins formulation/cascade/symmetry as a guarded property, sharing _check_truncation with __init__.
-->

# Version history -- `lumenairy/elements/pmm/stack2d.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/pmm/stack2d.py`: the W7 A11 measured pre-fix table behind
the `_geom_key` cache key, and the two "bit-identical to the pre-fix library"
clauses.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

What did NOT move: the frame-anchor derivation, the mortar and sliver contracts,
and the measured bars they carry.

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
| L132-133 | `_slant_frame_offset` docstring | "stay bit-identical to the pre-fix path" |
| L490-505 | `PMM2DStack._geom_key` docstring | W7 A11 -- what the key used to carry and the measured stale-build drifts that followed |
| L1832-1835 | `PMM2DStack.solve`, the public-gauge amplitudes | "stay BIT-IDENTICAL to the pre-fix library" |

---

### L132-133 -- `_slant_frame_offset` docstring -- "stay bit-identical to the pre-fix path"

*Left in the source:* what the sentinel means, stated as a property of the code rather than of an earlier release.

```text
    ``(0.0, 0.0)`` for a stack with no sheared region, which is the signal to
    skip the anchor entirely and stay bit-identical to the pre-fix path."""
```

### L490-505 -- `PMM2DStack._geom_key` docstring -- W7 A11 -- what the key used to carry and the measured stale-build drifts that followed

*Left in the source:* the rule (the key must carry the five SOLVER parameters too) and why it is not optional: the cache outlives `solve()` and the attributes have no property guard.

```text
        W7 A11 (2026-07-26): the key used to carry the LAYER geometry only
        (kind / tile bytes+shape / walls / element counts).  But the cached
        value is produced by ``_build_axis(self.period_*, ..., self.degree,
        ..., self.grade)`` and ``_scalar_projected_ops(..., self.period_x,
        self.period_y)`` over the ``self.n_orders`` order set -- five SOLVER
        parameters that were absent from the key while ``_geom_cache``
        persists across ``solve()`` and is dropped only by ``add_layer``.
        They are plain public attributes with no property guard, so mutating
        one after a solve served the STALE build with no signal.  Measured
        pre-fix (4x4 cell + uniform film, ``n_orders=2``): ``degree`` 5 -> 9
        returned ``sum(R) = 0.237212592`` where a fresh object gives
        ``0.243068009`` (8.58e-03); ``degree`` 5 -> 7 and 5 -> 11 came back
        BIT-IDENTICAL to the degree-5 answer (6.24e-03 / 9.48e-03);
        ``grade`` False -> True drifted 2.03e-02.  Clearing ``_geom_cache``
        by hand made every one of them bit-identical to the fresh object --
        the build was right, only the key was wrong."""
```

### L1832-1835 -- `PMM2DStack.solve`, the public-gauge amplitudes -- "stay BIT-IDENTICAL to the pre-fix library"

*Left in the source:* which quantities the frame anchor touches and which it does not, which is the contract.

```text
            # pair additionally carries the frame anchor derived above.  R, T,
            # rz/tz and the reflection Jones are computed from the untouched
            # cascade output, so a slanted patterned layer's efficiencies and
            # reflection stay BIT-IDENTICAL to the pre-fix library.
```
