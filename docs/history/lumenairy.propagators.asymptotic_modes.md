<!-- lumenairy-history-doc
module: lumenairy/propagators/asymptotic_modes.py
ast_sha256: 4305e20e6744d29928121290c9321e4f0863864d7bf7f714907a1e6fbfe9525e
token_sha256: 8dc7e53fbb11f5b5e0c196a18db49b81d9e3bbbbd19d7edc7180102293a6bce4
pre_relocation_lines: 893
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/asymptotic_modes.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/asymptotic_modes.py`.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

`_lg_mode_conj_stack` is the archetype of the audit's V6 finding: its cache
key was widened three times (v4.14.1 added the pitch, v5.17.1 the origin,
v5.30 the three-corner fingerprint) and each round APPENDED a paragraph
instead of updating the one above it.  By the end the docstring opened with a
summary that named the ORIGIN as the key -- two paragraphs before the
paragraph that says the origin is not enough.  That summary was therefore
**wrong about the live code**, and it is corrected rather than moved: the
source now states the key once, with the `indexing='xy'` / `indexing='ij'`
collision that forced the corner fingerprint and the measurement that found
it.  `_hg_mode_conj_stack` carried the same three paragraphs and got the same
treatment.

Two smaller blocks were comments correcting earlier COMMENTS -- `decompose_lg`
recording that its own text used to say "trapezoidal quadrature" and
"(Nx, Ny)".  The live caveat that the rectangle rule matters for a field with
support at the boundary, with its 5.0e-15 measurement, stayed.

`_grid_corner_fingerprint`'s own docstring did NOT move: the proof that shape
+ pitch + origin cannot separate the two meshgrid orientations, and the
8.232e+00 worst relative error it produced, is the derivation of a live cache
key.

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
| L331-357 | `_lg_mode_conj_stack` -- cache key | three appended widening paragraphs (v4.14.1 pitch, v5.17.1 origin, v5.30 corners), each recording what the previous key collided on |
| L415-426 | `_hg_mode_conj_stack` | the same three appended widening paragraphs |
| L759 | `_meshgrid_axis_step` | the `pre-fix idiom, unchanged values` tag |
| L783-787 | `decompose_lg` -- the quadrature rule | a comment correcting an earlier COMMENT: that this line used to say 'trapezoidal quadrature', which the code has never done |
| L796 | `decompose_lg` -- `field` shape | a comment correcting an earlier COMMENT: that the docstring used to say ``(Nx, Ny)`` |

---

### L331-357 -- `_lg_mode_conj_stack` -- cache key -- three appended widening paragraphs (v4.14.1 pitch, v5.17.1 origin, v5.30 corners), each recording what the previous key collided on

*Left in the source:* the live key -- corrected, since the summary above those paragraphs still named the ORIGIN -- together with the collision that forced the corner fingerprint and its measured size

```text
    Cache key includes the grid shape, the physical pitch ``(dx, dy)``,
    the grid origin ``(X[0, 0], Y[0, 0])``, all basis parameters, and
    the dtype of the (X, Y) sample arrays so cached entries are only
    reused when the result would be bit-equal.

    v4.14.1 (P0-NEW-1):  ``dx, dy`` are included in the cache key.
    Pre-v4.14.1 keys captured only ``(Ny, Nx)``, so two calls with the
    same shape but different physical pitch (e.g. ``dx=1e-6`` then
    ``dx=2e-6`` at N=256) collided on the cache and the second call
    silently received the first call's modes evaluated against the
    second call's field.  Thread-safe via ``_LG_MODE_STACK_LOCK``.

    v5.17.1 (audit P1-06):  the grid origin ``(X[0, 0], Y[0, 0])`` is
    included in the cache key.  Shape + pitch alone do not pin the
    physical sample positions, so two same-shape/same-pitch grids at
    different offsets (e.g. a shifted ROI) collided and the second call
    silently received modes evaluated at the first grid's coordinates.

    v5.30 (audit W6-A11):  the origin is not enough either -- an
    ``indexing='xy'`` grid and an ``indexing='ij'`` grid built from the
    same two equal-length axes share shape, pitch AND origin, so they
    collided and the second caller received the first caller's
    TRANSPOSED stack (measured worst relative error 8.232e+00 against an
    independent overlap oracle, vs 5.2e-15 with the cache cleared in
    between).  The key now carries the three corners per axis --
    :func:`_grid_corner_fingerprint` -- which pin a rectilinear grid
    completely.
```

### L415-426 -- `_hg_mode_conj_stack` -- the same three appended widening paragraphs

*Left in the source:* the live key and the cross-reference

```text
    v4.14.1 (P0-NEW-1):  ``dx, dy`` are included in the cache key for
    the same reason as the LG variant -- same shape at different
    physical pitch must not collide.  Thread-safe via
    ``_HG_MODE_STACK_LOCK``.

    v5.17.1 (audit P1-06):  the grid origin ``(X[0, 0], Y[0, 0])`` is
    included in the cache key; see :func:`_lg_mode_conj_stack`.

    v5.30 (audit W6-A11):  the three corners per axis replace the bare
    origin so an ``indexing='xy'`` grid cannot collide with the
    ``indexing='ij'`` grid built from the same axes; see
    :func:`_grid_corner_fingerprint`.
```

### L759 -- `_meshgrid_axis_step` -- the `pre-fix idiom, unchanged values` tag

*Left in the source:* what the branch detects

```text
        # indexing='xy' orientation (pre-fix idiom, unchanged values)
```

### L783-787 -- `decompose_lg` -- the quadrature rule -- a comment correcting an earlier COMMENT: that this line used to say 'trapezoidal quadrature', which the code has never done

*Left in the source:* the live rule and the measured size of the difference, which is the caveat a caller with boundary support needs

```text
    v5.30 (audit W6-A14): pre-fix this line said "trapezoidal
    quadrature", which the code has never done.  It is immaterial when
    the field is contained inside the grid (measured 5.0e-15 relative
    difference between the two rules on an LG_{0,0} over +-4w) but it is
    NOT immaterial for a field with support at the boundary.
```

### L796 -- `decompose_lg` -- `field` shape -- a comment correcting an earlier COMMENT: that the docstring used to say ``(Nx, Ny)``

*Left in the source:* the live shape, stated on the line above

```text
        (v5.30 audit W6-A14: the pre-fix docstring said ``(Nx, Ny)``.)
```
