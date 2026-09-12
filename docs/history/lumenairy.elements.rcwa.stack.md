<!-- lumenairy-history-doc
module: lumenairy/elements/rcwa/stack.py
ast_sha256: 8d711be52b2b11b235608870b89c36fda8bb3ab38d9c4c0be29347c4f7cdb4d7
token_sha256: 450ffb77577a61f1ceebc4d50f5776360fbb1a9ce4189cbdfa8f60c81c96b2c5
pre_relocation_lines: 3372
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/rcwa/stack.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/rcwa/stack.py` -- chiefly the W8 boundary-coincidence
before/after table on `add_tapered_grating`, and the "formerly / previously /
the old X did A" clauses on the layer, gauge and render paths.  Each block is
reproduced **verbatim** under the source line it came from in the pre-relocation
file.

What did NOT move: the PIXEL CELL CONTRACT block, the `raster` convergence
ladders and their RECOMMENDATION / REJECTED verdicts, the
`RCWAYAverageWarning` measurements, and the `eps_cell_normal` companion-pair
contract -- `tests/unit/test_niche_audit_w9_raster_harmonic.py` reads several of
those strings out of the module source and out of
`RCWAStack.add_tapered_grating.__doc__` / `add_layer.__doc__`.

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
| L776-779 | `RCWAResult.apply_transmission` docstring | that the field helpers historically exposed only the reflection port |
| L978-980 | `RCWAResult._layer_amplitudes` docstring | "the old top-referenced ``c- exp(+lam k0 z)``" framing |
| L1065-1071 | `RCWAResult.internal_field`, the incident-gauge bridge | what a complex incident Jones "previously returned" |
| L1465-1467 | `RCWAStack.layers` docstring | that reverse translators "previously had to" read the private slot |
| L1523-1526 | `RCWAStack.add_layer` docstring, the ``eps`` bullet | "a bare ``(3, 3)`` formerly crashed at solve" |
| L1586-1593 | `RCWAStack.add_layer`, the uniform-tensor entry | that a bare ``(3, 3)`` "was formerly accepted" into the scalar slot |
| L1902-1927 | `RCWAStack.add_tapered_grating` docstring, "BOUNDARY COINCIDENCE" | the W8 pre-fix / post-fix efficiency ladder and the measured duty quantisation it corrects |
| L2250-2254 | `RCWAStack.plot_geometry`, the pixel test | "the old ``|xs - cx| < wx/2``" framing |
| L2415-2419 | `RCWAStack._materialized_layers`, the tensor contract | "must not re-impose the old in-plane-only restriction" |
| L2666-2669 | `RCWAStack._li_tensor_convolutions` docstring | "it is the historical operator BIT for BIT (the pre-2026-09-12 body is now ...)" |

---

### L776-779 -- `RCWAResult.apply_transmission` docstring -- that the field helpers historically exposed only the reflection port

*Left in the source:* why the transmissive counterpart exists, and the copy semantics.

```text
        counterpart of :meth:`apply_reflection` (audit S5-11: a transmissive
        metasurface's observable is the transmitted field, but the RCWAResult
        field helpers historically defaulted to / only exposed the reflection
        port).  Operates on a COPY so the caller's incident field is preserved.
```

### L978-980 -- `RCWAResult._layer_amplitudes` docstring -- "the old top-referenced ``c- exp(+lam k0 z)``" framing

*Left in the source:* the overflow hazard itself, in present tense -- it is the reason the backward amplitude is referenced at the layer BOTTOM.

```text
        ``exp(+lam k0 z)`` (the old top-referenced ``c- exp(+lam k0 z)`` blew up
        to NaN through high-loss metal layers, silently zeroing
        :meth:`layer_absorption`).  Math-identical: ``c- = X c-_bot`` with
```

### L1065-1071 -- `RCWAResult.internal_field`, the incident-gauge bridge -- what a complex incident Jones "previously returned"

*Left in the source:* the gauge rule and the failure it prevents, in present tense, with the oracle that found it.

```text
        # The cascade runs in the INTERNAL (conjugate) gauge and the output
        # is conjugated back, so a PUBLIC incident Jones (ex0, ey0) must enter
        # CONJUGATED: conj(S(conj(inc))) = ex0 F_pub_x + ey0 F_pub_y -- the
        # public-linear superposition.  Real incidents are unchanged; complex
        # (circular/elliptical) drives previously returned the field of the
        # conjugated incident, i.e. the OPPOSITE handedness (bug found by the
        # PMM internal-field co-registration oracle, 2026-06-11).
```

### L1465-1467 -- `RCWAStack.layers` docstring -- that reverse translators "previously had to" read the private slot

*Left in the source:* what the accessor is for, in present tense.

```text
        :meth:`add_layer` stored (AUDIT_DYNAMETA_CONSUMER_API_GAPS A2;
        reverse translators previously had to read the private ``_layers``
        slot under a version ceiling).
```

### L1523-1526 -- `RCWAStack.add_layer` docstring, the ``eps`` bullet -- "a bare ``(3, 3)`` formerly crashed at solve"

*Left in the source:* what the ``(3, 3)`` spelling means and where it is routed, which is the contract.

```text
        * ``eps`` (scalar) -- uniform isotropic spacer; a ``(3, 3)`` array is a
          spatially-uniform ANISOTROPIC slab (audit S1-14 / S5-11: expanded to a
          uniform tensor cell and solved on the tensor path -- the clean
          uniform-tensor entry; a bare ``(3, 3)`` formerly crashed at solve);
```

### L1586-1593 -- `RCWAStack.add_layer`, the uniform-tensor entry -- that a bare ``(3, 3)`` "was formerly accepted" into the scalar slot

*Left in the source:* the same opaque failure as the reason for the expansion, plus the ordering requirement relative to the one-of count.

```text
        # Uniform ANISOTROPIC entry (audit S1-14 / S5-11): a (3, 3) permittivity
        # tensor passed as ``eps`` is a spatially-uniform tensor layer.  A bare
        # (3, 3) was formerly accepted into the scalar "uniform" slot and then
        # crashed OPAQUELY at solve (``complex()`` of a (3, 3) array).  Expand it
        # to a minimally-sampled uniform tensor CELL and route it through the
        # validated tensor eigenmode path (its only Fourier harmonic is DC, so it
        # is exactly a homogeneous anisotropic slab).  Kept BEFORE the one-of
        # count so the layer registers as an ``eps_tensor_cell``.
```

### L1902-1927 -- `RCWAStack.add_tapered_grating` docstring, "BOUNDARY COINCIDENCE" -- the W8 pre-fix / post-fix efficiency ladder and the measured duty quantisation it corrects

*Left in the source:* the rule (the wall test is HALF-OPEN) and the coincidence FAMILY that makes it matter, plus the statement that non-coincident rasters are unaffected.

```text
        BOUNDARY COINCIDENCE (audit W8, fixed v5.31).  The wall test used to be
        the SYMMETRIC ``|x - centre| < duty/2``, which excludes BOTH walls, so a
        wall landing EXACTLY on a pixel centre lost that pixel.  At
        ``shear = 0.5, duty = 0.5`` the lower wall of slice ``k`` sits at
        ``lo = (k + 0.5)/(2 n_slices)``, which hits a pixel centre
        ``(i + 0.5)/n_x`` for EVERY slice whenever ``n_x == 2 n_slices`` -- a
        whole coincidence FAMILY, not one unlucky point.  Measured pre-fix at
        ``n_slices = 128, n_x = 256``: all 128 slices realised duty
        ``0.49609375 = 127/256`` (-3.906e-03).

        The physics, on a clean-closure case (``P = 1 um``, ``wl = 633 nm``,
        ``d = 300 nm``, ``eps_ridge = 4``, ``M = 7``, ``n_slices = 64``, so the
        coincidence is at ``n_x = 128``; realised duty ``63/128 = 0.4921875``).
        ``(R0_TE, R0_TM, T0_TE, T0_TM)`` versus ``n_slices`` at ``n_x = 128``::

            n_slices | pre-fix                                | post-fix
                  16 | 0.067642 0.135155 0.167056 0.571015    | identical
                  32 | 0.067392 0.135313 0.168182 0.569711    | identical
                  64 | 0.070403 0.125244 0.161805 0.587247 <- | 0.067328 0.135350 0.168465 0.569385
                 128 | 0.067270 0.135390 0.168769 0.569466    | identical

        The ``n_slices = 64`` row is the OUTLIER; every other row is BIT-identical
        pre and post (the fix touches only coincident pixels).  Against the
        ``n_x = 1024`` answer the coincident point was off by 1.802e-02 and is
        now off by 1.552e-04 -- the ordinary ``O(1/n_x)`` quantisation, a 116x
        improvement, achieved AT ``n_x = 128`` rather than by refining.
```

### L2250-2254 -- `RCWAStack.plot_geometry`, the pixel test -- "the old ``|xs - cx| < wx/2``" framing

*Left in the source:* both failure modes as the reason the render test is half-open and wrap-aware.

```text
                            # HALF-OPEN + wrap-aware (audit W8, PIXEL CELL
                            # CONTRACT): the old ``|xs - cx| < wx/2`` excluded
                            # BOTH walls (an edge exactly on a pixel centre
                            # dropped the pixel) and silently dropped any
                            # rectangle crossing the cell edge from the render.
```

### L2415-2419 -- `RCWAStack._materialized_layers`, the tensor contract -- "must not re-impose the old in-plane-only restriction"

*Left in the source:* the contract the dispersive path must mirror and the one guard that does apply.

```text
                # Mirror the STATIC add_layer tensor contract (audit P3-38):
                # out-of-plane tensors are supported since v5.14.1, so the
                # materialised dispersive cell must not re-impose the old
                # in-plane-only restriction -- only the nonzero-e_zz guard
                # (the pointwise ezz-Schur fold divides by it) applies.
```

### L2666-2669 -- `RCWAStack._li_tensor_convolutions` docstring -- "it is the historical operator BIT for BIT (the pre-2026-09-12 body is now ...)"

*Left in the source:* what the flag does and where the L2L1 operator lives, which is what a reader following the two rules needs.

```text
        ``symmetrize=False`` below is what keeps this call on the per-axis
        rule, and it is the historical operator BIT for BIT (the pre-2026-09-12
        ``_li_convolutions_2d_tensor`` body is now ``_li_tensor_l2l1``,
        unchanged)."""
```
