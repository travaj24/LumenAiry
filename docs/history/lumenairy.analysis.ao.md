<!-- lumenairy-history-doc
module: lumenairy/analysis/ao.py
ast_sha256: 720f352dc377c9fea2675fcebb0f77f1aa246c340482188685256841fbaa4d60
token_sha256: 459fb3449702bbf6df66f0b540883ccbaf176940183637d7a9e54c6ebd11f038
pre_relocation_lines: 1355
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/analysis/ao.py`

This file holds the version-history narrative that used to live in
`lumenairy/analysis/ao.py`.  Each block is reproduced **verbatim** under the
source line it came from in the pre-relocation file.

Only two blocks moved.  `ao.py`'s long comments are calibration physics -- the
SH simulator's own centroid response curve, the two-step reference/scale
calibration and why it is cached per lenslet geometry -- and they stayed
untouched, as did the joint `gain` / `leak` semantics table in
`ao_closed_loop`'s Notes, which is a live contract a caller gets wrong by
intuition.

What moved is the account of *why* the WFS adapter was written (a GUI combo
box with nothing behind it) and the measured evidence that
`noise_sigma_pixels` was inert before the injection point moved upstream of
the calibration rescale.  The reason the injection point matters -- the
algebraic identity, the `slope_scale` units, the RNG-order guarantee -- stayed
in the source: it is what stops someone moving it back.

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
| L960-969 | `make_shack_hartmann_wfs` -- why the adapter exists | the account of the v5.3.2 GUI's 'WFS type' combo having nothing to wire |
| L1190-1223 | `make_shack_hartmann_wfs` -- centroid-noise placement | the measured pre-fix inertness of the knob (4.83e-5 and 4.81e-7 relative perturbation, i.e. nothing) and the `Documented effect, now true` framing |

---

### L960-969 -- `make_shack_hartmann_wfs` -- why the adapter exists -- the account of the v5.3.2 GUI's 'WFS type' combo having nothing to wire

*Left in the source:* what the factory builds and what consumes it

```text
# v5.4 (AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A wfs adapter): the
# ``ao_closed_loop`` helper accepts an arbitrary callable
# ``wfs(residual_phase) -> measured_phase``, but until v5.4 the only
# canonical option was ``None`` (ideal phase sensing) or a hand-rolled
# closure -- the v5.3.2 GUI ``ao_dock`` ships a "WFS type" combo
# (``shack_hartmann`` / ``pyramid`` / ``curvature``) that had nothing
# to wire because no library-side adapter wrapped ``shack_hartmann``
# (the spot-displacement simulator) and ``slope_to_modal`` (the
# slope-to-Zernike reconstructor) into a single closure.  This
# factory closes that gap so the dock can call a real WFS path.
```

### L1190-1223 -- `make_shack_hartmann_wfs` -- centroid-noise placement -- the measured pre-fix inertness of the knob (4.83e-5 and 4.81e-7 relative perturbation, i.e. nothing) and the `Documented effect, now true` framing

*Left in the source:* where the noise is injected and WHY it must be injected there, the algebraic identity that makes the two placements equivalent up to a scale, the RNG-order guarantee, the quantified effect of the knob, and the NaN-sentinel note

```text
        # 2b. Optional centroid noise.  S11-5
        # (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): this block used to
        # sit AFTER the ``/ slope_scale`` calibration rescale below, so a
        # sigma quoted in RAW SH slope units (m of OPD per m of pupil)
        # was added to slopes already converted to rad/m of PHASE -- a
        # ``slope_scale`` (~1e-6 .. 1e-7, essentially lambda/2pi times the
        # SH response) mis-scaling that made the knob inert.  Measured on
        # a 64x64 defocus residual, ``subaperture_grid=8``,
        # ``lenslet_focal=5e-3``: at ``dx_pupil = 1e-4`` m,
        # ``noise_sigma_pixels = 1`` perturbed the reconstruction by
        # 4.83e-5 relative (and 4.81e-7 at ``dx_pupil = 1e-5`` m) -- i.e.
        # nothing.
        #
        # The fix is to inject the noise where it physically belongs:
        # on the RAW measurement, upstream of the calibration.  That is
        # algebraically identical to dividing the sigma by
        # ``slope_scale`` after the rescale
        # (``(s + n - ref)/ss == (s - ref)/ss + n/ss``) but needs no
        # extra division, cannot blow up on a degenerate ``slope_scale``
        # any worse than the signal path already does, and keeps the RNG
        # draw count and order unchanged -- so a given ``rng_seed``
        # produces the identical noise SEQUENCE, only correctly applied.
        # The ``_noise == 0`` path is untouched and bit-identical.
        #
        # Documented effect, now true: ``noise_sigma_pixels = s`` injects
        # an independent Gaussian centroid error of ``s * dx_pupil``
        # metres per sub-aperture, i.e. a wavefront-slope error of
        # ``s * dx_pupil / lenslet_focal`` radians of tilt.  Note this is
        # a LARGE error for a coarse pupil grid: one whole pixel of
        # centroid error on a 100 um pupil pitch with a 5 mm lenslet is
        # 20 mrad of tilt.  Real SH systems sit at s ~ 0.01 - 0.1.
        #
        # NaN sentinels (out-of-bounds lenslets) stay NaN: finite noise
        # added to NaN is NaN, so the ``good`` mask below is unchanged.
```
