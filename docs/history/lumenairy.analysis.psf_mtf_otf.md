<!-- lumenairy-history-doc
module: lumenairy/analysis/psf_mtf_otf.py
ast_sha256: 9b1cddc0e769c5ddb48a57475595fdf5bc384cc51b358f8ea40dcc385feaea13
token_sha256: 60dae352e8ce16025fec376da99d81fc20263f1f652f1fdaef00393701af87d9
pre_relocation_lines: 1579
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/psf_mtf_otf.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/psf_mtf_otf.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

Five blocks moved, of three kinds:

* **Two fail-before / fix-after tables.**  `rayleigh_resolution` and
  `fwhm_resolution` each carried a two-column accuracy table comparing the
  v5.29 integer-pixel radial binning against the v5.30 sub-pixel profile.
  The fix-after column is the measured accuracy of the code that ships, so it
  stayed (`docs/TESTING_STANDARDS.md` S5); the v5.29 column is the defect the
  A-1 finding closed, so it is here.  Nothing is lost by that split: the
  argument that binning is wrong -- lopsided small-r shells pulling the
  azimuthal mean down, with its own measurements -- stays in full in
  `_resolution_profile`'s Rationale, which is where a future editor tempted
  to re-bin would look.
* **A comment correcting an earlier COMMENT.**  `compute_otf`'s Returns
  section recorded that the docstring used to claim `otf[0, 0]` was the DC
  element, contradicting the `fftshift` the implementation has always
  applied.  The correct convention is stated plainly above it; the note
  about the incorrect one is here.
* **A comment about a comment's own life cycle.**  `compute_psf`'s guard
  recorded that a v4.15.5 marker comment deferred to a helper landing in a
  parallel branch, outlived it by five minor releases, and in the meantime
  described a pupil as a "field".

What each guard PREVENTS stayed in both places -- an ndim-3 input reaching
`xp.fft.fft2` is silently wrong, not an error, and that is the reason the
guard exists.

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
| L123-134 | `compute_psf` -- the 2-D guard | a comment about the comment's own history: a v4.15.5 marker that deferred to a helper in a parallel branch, outlived it by five minor releases and left this guard calling a pupil a 'field' until v5.31 A-9 |
| L228-234 | `compute_otf` -- the OTF lattice convention | a comment correcting an earlier COMMENT: that the docstring used to claim ``otf[0, 0]`` was the DC = 1 element, contradicting the ``fftshift`` the implementation has always applied |
| L242-250 | `compute_otf` -- the 2-D guard | the `v4.15.5 (P1-NEW-2WAY-1)` / `Previously` framing |
| L1130-1151 | `rayleigh_resolution` -- Accuracy | the two-column v5.29-binned / v5.30-sub-pixel comparison, i.e. the fail-before column of the A-1 fix |
| L1507-1526 | `fwhm_resolution` -- Accuracy | the two-column v5.29-binned / v5.30-sub-pixel comparison |

---

### L123-134 -- `compute_psf` -- the 2-D guard -- a comment about the comment's own history: a v4.15.5 marker that deferred to a helper in a parallel branch, outlived it by five minor releases and left this guard calling a pupil a 'field' until v5.31 A-9

*Left in the source:* what the guard prevents and which ``input_kind`` it declares

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously an MCF / 3-D
    # ensemble pupil failed downstream at ``pupil.ndim`` /
    # ``pupil.shape`` and produced an unhelpful TypeError /
    # ValueError instead of the canonical v4.16 message.  Input
    # kind: 'pupil' (the function consumes a 2-D pupil amplitude *
    # phase product and does a single Fraunhofer FT to the PSF
    # plane).  v5.31 (audit A-9): ``input_kind='pupil'`` is now
    # actually passed.  v4.15.5 landed the parameterised helper in a
    # parallel branch and left a marker comment here deferring to it;
    # the marker outlived the thing it was waiting for by five minor
    # releases, so this guard described a pupil as a "field".
```

### L228-234 -- `compute_otf` -- the OTF lattice convention -- a comment correcting an earlier COMMENT: that the docstring used to claim ``otf[0, 0]`` was the DC = 1 element, contradicting the ``fftshift`` the implementation has always applied

*Left in the source:* the convention itself (stated in the two paragraphs above) and the matching frequency axes

```text
        v5.29.1 (audit A-7): this docstring previously claimed
        ``otf[0, 0]`` was the DC = 1 element, contradicting the
        ``fftshift`` the implementation has always applied.  Behaviour is
        unchanged -- only the documented convention is corrected.  The
        matching spatial-frequency axes are
        ``fftshift(fftfreq(N, d=dx_psf))``; see :func:`mtf_radial`, which
        already assumes the centred layout.
```

### L242-250 -- `compute_otf` -- the 2-D guard -- the `v4.15.5 (P1-NEW-2WAY-1)` / `Previously` framing

*Left in the source:* the whole hazard -- an ndim-3 input would have been FFT'd along its last two axes, silently wrong -- and why a real intensity PSF is accepted by a helper named for scalar fields

```text
    # v4.15.5 (P1-NEW-2WAY-1): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  Previously an MCF / 3-D
    # ensemble psf failed downstream at ``xp.fft.fft2`` (which would
    # FFT along the last two axes of a 3-D stack -- silently wrong
    # output shape).  Input kind: 'psf' (a real-valued intensity
    # PSF; the helper still accepts it because the only invariant
    # checked is ``.ndim == 2`` plus the MCF rejection).  Routes
    # both failure modes to the canonical v4.16 message via the V6
    # walker.
```

### L1130-1151 -- `rayleigh_resolution` -- Accuracy -- the two-column v5.29-binned / v5.30-sub-pixel comparison, i.e. the fail-before column of the A-1 fix

*Left in the source:* the measured accuracy of the LIVE implementation (the fix-after column), the sampling floor below which the metric is meaningless, and the name of the profile it measures.  The evidence that integer-pixel binning is wrong stays in full in :func:`_resolution_profile`'s Rationale.

```text
    Accuracy (v5.30, audit
    AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 finding A-1):
    ``axis='radial'`` measures the sub-pixel azimuthally-averaged
    profile (:func:`_radial_profile_subpixel`, the same one
    :func:`sparrow_resolution` uses) instead of the integer-pixel
    radial binning it used through v5.29.  Measured on an analytic
    Airy PSF (600 nm, f/4) against ``1.22 lambda f/#``:

    ======================  ===============  ==============
    samples / first zero    v5.29 (binned)   v5.30 (sub-px)
    ======================  ===============  ==============
    19.5                    -0.12%           +0.02%
    9.8                     +1.54%           +0.10%
    4.9                     +6.76%           +0.18%
    2.4                     NaN + warning    +3.26%
    ======================  ===============  ==============

    Below ~3 samples per first zero the ring is barely resolved and
    the residual error grows quickly (a few percent at 2.4, tens of
    percent at 1.2); the metric is only meaningful on a PSF the grid
    actually resolves.  ``axis='x'`` / ``axis='y'`` take pixel-aligned
    cuts through the peak and are unchanged.
```

### L1507-1526 -- `fwhm_resolution` -- Accuracy -- the two-column v5.29-binned / v5.30-sub-pixel comparison

*Left in the source:* the measured accuracy of the LIVE implementation, the name of the profile it measures, and why the binned profile biased the crossing inward

```text
    Accuracy (v5.30, audit
    AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 finding A-1):
    ``axis='radial'`` measures the sub-pixel azimuthally-averaged
    profile (:func:`_radial_profile_subpixel`, the same one
    :func:`sparrow_resolution` uses) instead of the integer-pixel
    radial binning it used through v5.29, whose lopsided small-r
    shells biased the half-max crossing sharply inward.  Measured on
    an analytic Airy PSF (600 nm, f/4) against ``1.029 lambda f/#``:

    ======================  ===============  ==============
    samples / first zero    v5.29 (binned)   v5.30 (sub-px)
    ======================  ===============  ==============
    19.5                    -0.12%           -0.00%
    9.8                     -1.92%           +0.01%
    4.9                     -8.04%           +0.01%
    2.4                     -21.08%          -1.03%
    ======================  ===============  ==============

    ``axis='x'`` / ``axis='y'`` take pixel-aligned cuts through the
    peak and are unchanged (+0.1 to +1.4% over the same sweep).
```
