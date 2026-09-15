<!-- lumenairy-history-doc
module: lumenairy/analysis/detector.py
ast_sha256: 3cbadf0e6352825613b1de677453fc9c24df1ab4e3f18583eb907c6a75a730cc
token_sha256: 57210402c83251b0b69ea99869be73cf8bba33801fe64b6421485da2f06be3c0
pre_relocation_lines: 960
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/analysis/detector.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/detector.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

One block here is the audit's §15.7 case -- a comment that states the OPPOSITE
of the code -- and it is the clearest example in this sweep of how that
happens.  The 33-line banner above the integration block had been appended to
three times without ever being edited, so it described three different
algorithms in sequence:

1. "The old approach used integer truncation ..."  (gone);
2. "Here we use scipy.ndimage.zoom to resample to the detector pitch with
   proper anti-aliased integration ..."  (**false** -- `scipy.ndimage` is not
   imported by this module and `zoom` appears nowhere outside that comment);
3. "4.10: proper area integration ...  Pre-4.10 used
   scipy.ndimage.zoom(order=1) ..."  (contradicting 2);
4. "For non-integer ratios first uniform-filter to anti-alias, then sample at
   the new pixel centers, scaled by pixel_pitch^2 ..."  (**false** -- the
   live branch assigns each field sample's energy to the pixel containing its
   physical centre, with no filter and no `pixel_pitch**2`).

A reader following the comment would have believed the module interpolates.
The source now states what the two live branches do, and keeps the
dimensional argument against point-sampling interpolation as a
do-not-do-this rather than as a description.

Two further blocks were comments correcting earlier COMMENTS: the `x_det`
"center coordinates" retraction, and `shack_hartmann`'s `input_kind` gloss,
which argued at length against its own v4.15.5 conclusion.  Both source sites
now state the live convention once.

What stayed: the live `cosmic_ray_rate` -> `cosmic_ray_rate_per_m2_per_s`
migration (a user still needs it), the S11-4 error law and its measured 16x
size, the flux-conservation argument in the non-integer branch, and
`_path_integrate_slopes`'s note that averaging two one-sided integrals halves
every separable wavefront -- a live do-not-do-this with its own measurement.

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
| L124-132 | `apply_detector` -- `x_det` / `y_det` | a comment correcting an earlier COMMENT: that the pre-fix docstring called these 'center coordinates', which they have never been |
| L137-168 | `apply_detector` -- the S11-4 pixel-area contract | the 5-row photon-scale before/after table, whose post-fix column is 1.0000 throughout, and the note that the two branches disagreed with each other |
| L195-199 | `apply_detector` -- the `n_pixels` guard | the `used to reach ... and die` framing |
| L209-241 | `apply_detector` -- the integration banner (CONTRADICTED THE CODE) | three successive descriptions of how this block integrates, newest last, two of which name a `scipy.ndimage` path this module has not used since 4.10 |
| L561-587 | `shack_hartmann` -- the 2-D guard and `input_kind` | a comment correcting an earlier COMMENT: the v4.15.5 gloss that declared `input_kind='pupil'` here, and the argument that talked itself out of its own conclusion |
| L693-710 | `shack_hartmann` -- the reference-centroid slice | the pre-4.10 raw-centroid slope, and the account of the pre-v5.30 transform mismatch (introduced with the v4.10 reference pass, hoisted out of the loop by v4.13.0 without being flagged) |

---

### L124-132 -- `apply_detector` -- `x_det` / `y_det` -- a comment correcting an earlier COMMENT: that the pre-fix docstring called these 'center coordinates', which they have never been

*Left in the source:* the live convention, the binning expression that proves it, the instruction for getting centres, and the deliberate deferral of the half-pixel-anchor decision

```text
        ``[x_det[j], x_det[j] + pixel_pitch)``.  (S11-4 note: the
        pre-fix docstring called these "center coordinates", which they
        have never been -- both binning branches map a field sample at
        ``x`` to ``floor(x / pixel_pitch + n_pixels / 2)``.  The values
        are left as-is rather than shifted by half a pixel because that
        is a separate half-pixel-anchor decision of the same class as
        ``analysis/plotting.py``'s ``(N-1)/2`` anchor, deliberately
        deferred by AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1.  Add
        ``pixel_pitch / 2`` if you need centres.)
```

### L137-168 -- `apply_detector` -- the S11-4 pixel-area contract -- the 5-row photon-scale before/after table, whose post-fix column is 1.0000 throughout, and the note that the two branches disagreed with each other

*Left in the source:* the contract itself, the error law an explicit ``n_pixels`` would introduce, its measured size, and the exact condition under which the block-sum fast path is provably identical to the physical binning

```text
    (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1).  Pre-fix, an explicit
    ``n_pixels`` silently redefined the pixel area: the integer fast
    path block-summed ``Ny / n_pixels`` FIELD SAMPLES per detector pixel
    while the returned axis was spaced by ``pixel_pitch``, so the
    per-pixel signal was wrong by
    ``((N * dx_field) / (n_pixels * pixel_pitch))**2`` and the whole
    field's flux was crammed into (or smeared across) the declared
    detector.  Measured at photon scale on a uniform
    ``I0 = 1e18 /m^2/s`` field, 64x64 @ 1 um, QE 1, 1 s, no noise
    (expected per-pixel electrons ``I0 * pixel_pitch**2``):

    ======================== ============= =============
    (n_pixels, pixel_pitch)  measured/exp  measured/exp
    \\                        (pre-fix)     (post-fix)
    ======================== ============= =============
    (16, 4 um)  matched       1.0000        1.0000
    (16, 2 um)                4.0000        1.0000
    (16, 8 um)                0.2500        1.0000
    (8,  2 um)               16.0000        1.0000
    (16, 2.5 um)              2.5600        1.0000
    ======================== ============= =============

    The non-integer branch (v5.4.6 F-10) was already correct at every
    combination -- it bins by physical position against ``pixel_pitch``
    -- so the two branches also disagreed with each other.  The integer
    block-sum fast path is now taken ONLY when it is provably identical
    to the physical binning: ``pixel_pitch / dx_field`` an integer
    ``s >= 1`` AND ``n_pixels * s == Nx == Ny`` (the detector exactly
    tiling the field, which is what the default ``n_pixels`` produces).
    Every other combination routes through the flux-conserving
    physical-position branch.  Bit-identical for the default
    ``n_pixels`` on a square field.
```

### L195-199 -- `apply_detector` -- the `n_pixels` guard -- the `used to reach ... and die` framing

*Left in the source:* the two numpy messages the guard replaces, which are the reason it exists

```text
        # S11-4: a non-positive explicit count used to reach the binning
        # block and die inside numpy ("zero-size array to reduction
        # operation maximum" for 0, "can only specify one unknown
        # dimension" for -2), naming neither this function nor the
        # argument.
```

### L209-241 -- `apply_detector` -- the integration banner (CONTRADICTED THE CODE) -- three successive descriptions of how this block integrates, newest last, two of which name a `scipy.ndimage` path this module has not used since 4.10

*Left in the source:* the imbalance that rules out index truncation, the two branches that are actually taken, and the dimensional argument against point-sampling interpolation -- restated as a do-not-do-this

```text
    # ---- Area-weighted integration onto the detector grid --------------
    # The old approach used integer truncation of the per-field-sample
    # index into the detector pixel grid, which gave non-uniform per-
    # pixel sample counts when (pixel_pitch / dx_field) wasn't an exact
    # integer aligned with the grid.  That imbalance dominated the
    # Poisson statistics (std was 20x sqrt(mean)).
    #
    # Here we use scipy.ndimage.zoom to resample to the detector pitch
    # with proper anti-aliased integration, then multiply by dx_field^2
    # to turn the re-sampled intensity (per unit area) into a per-pixel
    # integrated signal.  For integer ratios this agrees with block-sum
    # reshape to machine precision; for non-integer ratios it
    # interpolates cleanly.
    # 4.10: proper area integration of the intensity field onto the
    # detector grid.  Pre-4.10 used scipy.ndimage.zoom(order=1), which
    # is BILINEAR INTERPOLATION (point-sample at the new pixel
    # centers), NOT area integration.  Multiplying that by pixel_pitch^2
    # is dimensionally pixel_pitch^2 * intensity, NOT
    # integral_over_pixel(intensity) * dx_field^2 -- so photon
    # conservation fails for non-integer pixel_pitch/dx_field ratios
    # and shot-noise calibration loses meaning.
    #
    # For integer ratios use block-sum via np.add.reduceat.  For
    # non-integer ratios first uniform-filter to anti-alias, then
    # sample at the new pixel centers, scaled by pixel_pitch^2 so the
    # integral over each detector pixel is correctly represented.
    # (P4: removed a leftover compute-and-discard expression here.)
    #
    # Per-detector-pixel area in field samples.  S11-4
    # (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): this MUST come from
    # ``pixel_pitch``, the physical pixel size, not from
    # ``Ny / n_pixels``.  See the Notes block in the docstring for the
    # photon-scale before/after table (up to 16x per-pixel error).
```

### L561-587 -- `shack_hartmann` -- the 2-D guard and `input_kind` -- a comment correcting an earlier COMMENT: the v4.15.5 gloss that declared `input_kind='pupil'` here, and the argument that talked itself out of its own conclusion

*Left in the source:* what the guard prevents, the library-wide rule that decides the noun, and the physical argument that this entry point takes a metrically-scaled field

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Pre-v4.15.5 an MCF / 3-D
    # ensemble input failed at ``E.shape[0]`` (3-D returned a wrong
    # ``N``) or attribute access (MCF), then propagated wrong slopes
    # / centroids through the lenslet loop.  Routes both to the
    # canonical v4.16 message via the V6 walker.
    #
    # v5.32 (audit A-9 handoff): ``input_kind='field'``, superseding this
    # comment's own v4.15.5 claim of "Input kind: 'pupil' (the SH-WFS
    # measures a complex pupil-plane field)".  That gloss argued itself
    # out of its own conclusion -- "pupil-plane" names the PLANE, while
    # the thing this argument carries is a FIELD, which is exactly what
    # the Parameters entry above says ("E : ndarray, complex, shape
    # (N, N) / Input field at the lenslet array plane").  ``input_kind``
    # picks the noun in the rejection message, so it must match the
    # argument the caller actually passed: declaring 'pupil' here would
    # tell someone who passed ``E`` that a "2-D complex pupil" was
    # expected -- re-creating, in mirror image, the very A-9 defect
    # (``compute_psf(pupil, ...)`` reporting "field") that this rollout
    # exists to close.  The library-wide convention after A-9 is that
    # only the three sites whose parameter IS named ``pupil``
    # (``compute_psf``, ``richards_wolf_focus``, ``debye_wolf_psf``)
    # declare 'pupil'.  Physically consistent too: this entry point
    # takes ``dx`` and ``wavelength`` and propagates each sub-aperture
    # with the bandlimited angular-spectrum kernel at
    # ``z = lenslet_focal``, which needs a metrically-scaled field, not
    # a dimensionless pupil function.
```

### L693-710 -- `shack_hartmann` -- the reference-centroid slice -- the pre-4.10 raw-centroid slope, and the account of the pre-v5.30 transform mismatch (introduced with the v4.10 reference pass, hoisted out of the loop by v4.13.0 without being flagged)

*Left in the source:* why a reference is subtracted at all, why ONE slice serves every lenslet, and the requirement that the reference go through bit-identically the same transform as the measurement

```text
        # 4.10: pre-4.10 reported raw centroid / lenslet_focal as the slope,
        # baking in any per-lenslet centring bias from sa_pixels rounding
        # / x0 offset as a fake tilt in EVERY measurement.  Compute the
        # zero-slope reference centroid from a unit-amplitude flat field
        # and subtract.  A flat (ones) field produces an IDENTICAL
        # sub-aperture for every lenslet (``ones * lenslet_phase ==
        # lenslet_phase``), so ONE reference slice serves every lenslet.
        #
        # v5.30 (S12-1): the reference is slice 0 of the SAME batch that
        # carries the measurements, so it goes through bit-identically the
        # same transform.  Pre-v5.30 it was propagated by a BARE
        # ``fftshift(fft2(ifftshift(...)))`` while the measurement used the
        # bandlimited angular-spectrum kernel at ``z = lenslet_focal`` -- a
        # mismatch present since the v4.10 reference pass was introduced
        # and merely hoisted out of the loop by v4.13.0 (whose comment
        # above names the two different transforms without flagging the
        # asymmetry).  See the ``Notes`` block in this function's docstring
        # for the measured before/after table.
```
