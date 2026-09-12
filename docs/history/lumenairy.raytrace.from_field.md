<!-- lumenairy-history-doc
module: lumenairy/raytrace/from_field.py
ast_sha256: 3adb4d02e5387f5ae39423f234654218350619a5fbdee72e9f201149b39b6dbd
token_sha256: 17d8eeccc024675619a25f4666b3c46ed9b71b91f07c162a7aa03cbfb84497ff
pre_relocation_lines: 943
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/from_field.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/from_field.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

This module's history is one convention decision told six times: the
intensity-threshold comparison is INCLUSIVE (`>=`) in all three placement
modes.  The convention is live and stayed at every site; what moved is the
per-site record of which mode used `>` before and which release changed it.
The measured evidence for the phase-gradient fixes (the wrapped-L table, the
interior-vs-edge 0.050000 / 0.025000 ratio) stayed with its rule.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L69-74 | `<module> docstring` | the release tags and the per-mode before-state |
| L153-155 | `place_rays_from_field` | the release tags and "pre-v4.15.3 used strict >" |
| L189-192 | `place_rays_from_field` | the release tag and the per-mode before-state |
| L493-497 | `_place_cdf` | the release/audit tag and "Previously the threshold was applied" |
| L501-505 | `_place_cdf` | the release tag and "Pre-v4.15.3 used strict >" |
| L572-579 | `_place_rejection` | the release tag and the per-mode before-state |
| L627-635 | `_place_uniform` | "used to produce duplicate ... we now dedupe" and the release tag |
| L766-769 | `_angles_from_phase_gradient` | "the old claim" |
| L799-800 | `_angles_from_phase_gradient` | "the clipped self-reference used to halve" |
| L856-859 | `_angles_from_phase_gradient` | "Including it used to halve" |

---

### L69-74 -- `<module> docstring` -- the release tags and the per-mode before-state

*Left in the source:* the convention itself and why boundary-exact pixels matter.

```text
v4.15.3 -- Agent D: intensity-threshold comparison made consistent
across the 3 placement modes; all three now use the inclusive
``|E|^2 / max(|E|^2) >= intensity_threshold`` convention.  Pre-v4.15.3
only ``_place_rejection`` used ``>=``; ``_place_cdf`` and
``_place_uniform`` used strict ``>``, dropping boundary-exact pixels.
This is a numerical behaviour change for inputs at the exact threshold
```

### L153-155 -- `place_rays_from_field` -- the release tags and "pre-v4.15.3 used strict >"

*Left in the source:* the pixel-wise-before-marginal rule and the convention.

```text
          the CDF.  (v4.15.2 fix; see P1-NEW-D in the v4.15.1 audit.
          v4.15.3 -- ``>=`` is the canonical convention; pre-v4.15.3
          used strict ``>``.)  Use for visualisation.
```

### L189-192 -- `place_rays_from_field` -- the release tag and the per-mode before-state

*Left in the source:* the inclusive convention and the threshold's second role.

```text
        retained.  v4.15.3 -- the three modes are now consistent;
        pre-v4.15.3 only ``_place_rejection`` used the inclusive
        comparison while ``_place_cdf`` and ``_place_uniform`` used
        strict ``>``.  For ``angle_method='complex_gradient'`` the
```

### L493-497 -- `_place_cdf` -- the release/audit tag and "Previously the threshold was applied"

*Left in the source:* the ordering rule and the noise-accumulation failure it prevents.

```text
    v4.15.2 fix (P1-NEW-D in the v4.15.1 audit) -- threshold is now
    applied pixel-wise BEFORE the marginal sums.  Previously the
    threshold was applied to the marginal sums themselves
    (``Ix.sum(axis=0) > threshold * Ix.max()``), which let sub-
    threshold background noise accumulate across rows / columns and
```

### L501-505 -- `_place_cdf` -- the release tag and "Pre-v4.15.3 used strict >"

*Left in the source:* the convention and the siblings it matches.

```text
    v4.15.3 -- threshold comparison is inclusive (``>=``) so pixels at
    exactly ``intensity_threshold`` are retained, matching the
    canonical "pixel intensity meets threshold" convention used in
    ``_place_rejection`` and now also ``_place_uniform``.  Pre-v4.15.3
    used strict ``>``.  Documented in the v4.15.3 release notes.
```

### L572-579 -- `_place_rejection` -- the release tag and the per-mode before-state

*Left in the source:* the convention.

```text
    v4.15.3 -- threshold comparison is inclusive (``>=``) here as in
    all three placement modes; the canonical convention is "pixel
    intensity meets threshold" so a pixel whose normalised intensity
    is exactly ``intensity_threshold`` is retained.  Pre-v4.15.3 only
    this mode used ``>=`` (which became the chosen convention); the
    other two modes (``_place_cdf``, ``_place_uniform``) used strict
    ``>``.  The behaviour change for inputs at the exact threshold
    boundary is documented in the v4.15.3 release notes.
```

### L627-635 -- `_place_uniform` -- "used to produce duplicate ... we now dedupe" and the release tag

*Left in the source:* the dedupe rule and the convention.

```text
    number of unique pixels in the grid, the sub-grid pixelisation
    used to produce duplicate ``(iy, ix)`` entries; we now dedupe in
    a stable manner.

    v4.15.3 -- threshold comparison is inclusive (``>=``) so pixels at
    exactly ``intensity_threshold`` are retained, matching the
    canonical "pixel intensity meets threshold" convention used in
    ``_place_rejection`` and now also ``_place_cdf``.  Pre-v4.15.3
    used strict ``>``.  Documented in the v4.15.3 release notes.
```

### L766-769 -- `_angles_from_phase_gradient` -- "the old claim"

*Left in the source:* the whole wrapping measurement and the falsification, as a standing statement.

```text
    +0.050, 0.490 as -0.010).  That also falsified the old claim that
    the form "can detect evanescent rays whose tangential k exceeds
    :math:`\\pi/\\Delta x`": an evanescent ``L = 0.49`` came back as a
    benign ``L = -0.01``, so no evanescent ray was ever flagged.
```

### L799-800 -- `_angles_from_phase_gradient` -- "the clipped self-reference used to halve"

*Left in the source:* the hazard as a present-tense property.

```text
    # below (R6: the clipped self-reference used to halve the baseline
    # while the divisor stayed at 2 dx).
```

### L856-859 -- `_angles_from_phase_gradient` -- "Including it used to halve"

*Left in the source:* the boundary argument and the measured interior/edge ratio.

```text
    # is a real positive number carrying no phase.  Including it used to
    # halve the recovered direction cosine exactly -- measured on a
    # uniform-amplitude tilted plane wave with L_true = 0.05 on a 64x64
    # grid: interior pixels +0.050000, edge columns +0.025000, ratio
```

