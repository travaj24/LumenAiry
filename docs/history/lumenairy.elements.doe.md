<!-- lumenairy-history-doc
module: lumenairy/elements/doe.py
ast_sha256: 02873e2fd0767e4f7a521bfcc8dfbd0405879d63ea4f5636c0df928b3b3f16cc
token_sha256: 3d409a8175caa828f2fb36d8a3c30ec8c028f1ff65973983fe6d309ad01265c7
pre_relocation_lines: 1258
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- WP-B11b item 8: every warning in the two lens bodies asks _lens_kernels.caller_stacklevel for its level, so it names the first frame outside the package whatever wrapper / configuration re-entry reached it; doe.py's zone-plate fill takes T's own dtype.
-->

# Version history -- `lumenairy/elements/doe.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/doe.py`: the three stacked `.. versionchanged::` blocks on
`makedammann2d` (4.14.2 / 4.14.3 / 5.30) that trace the micrometre-to-SI unit
migration and the rise and removal of the `_legacy_units='auto'` heuristic, the
measured pre-fix occupancy table behind the `% cell_N` lattice close, and the
smaller "X used to ..." clauses.  Each block is reproduced **verbatim** under the
source line it came from in the pre-relocation file.

What did NOT move: the live `4.14.2` migration recipe (multiply pre-4.14.2
values by `1e-6`), the `5.30` statement of what `_legacy_units` accepts today and
why `'auto'` raises, and the reasons the guards are shaped as they are.

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
| L151-153 | `create_periodic_phase_mask` docstring, Notes | "it used to clamp with ``clip``" |
| L179-192 | `create_periodic_phase_mask`, the cell-pixel lookup | E-M7 -- the measured per-cell-pixel occupancy, the 2816 wrong mask pixels and the 11.3 % off-lattice leak the clamp produced |
| L505-507 | `create_fresnel_zone_plate` docstring, Notes | the 4.7 ``_m``-suffix rename |
| L522-526 | `create_fresnel_zone_plate`, the focal-length guard | "Negative focal_length used to silently strip the sign" |
| L581-607 | `makedammann2d` docstring, the 4.14.2 / 4.14.3 ``versionchanged`` blocks | the micrometre heuristic, its DeprecationWarning, and the ``_legacy_units='auto'`` mode that no longer exists |
| L608-628 | `makedammann2d` docstring, the 5.30 ``versionchanged`` block | the measured 1e-6-wrong THz design and the argument for retiring the shim rather than deprecating it again |
| L669-670 | `makedammann2d` docstring, ``cell_pixels`` | "keeps the historical behaviour" |
| L740-756 | `makedammann2d`, the ``_legacy_units`` dispatch | the W5 shim-removal narrative around the retired ``'auto'`` mode |
| L988-990 | `makedammann2d`, the progress plot | "match the historical user expectation" |
| L1145-1148 | `load_fits_field`, the split amp/phase auto-detect | "Pre-fix the default save -> default load round-trip ... SILENTLY dropped all phase" |

---

### L151-153 -- `create_periodic_phase_mask` docstring, Notes -- "it used to clamp with ``clip``"

*Left in the source:* the periodic-close rule and the failure the clamp produces, in present tense.

```text
    the tiling is exactly periodic (v5.30, audit E-M7 -- it used to clamp
    with ``clip``, which folded the last half-pixel of each cell onto the
    last column and injected spurious diffraction orders).
```

### L179-192 -- `create_periodic_phase_mask`, the cell-pixel lookup -- E-M7 -- the measured per-cell-pixel occupancy, the 2816 wrong mask pixels and the 11.3 % off-lattice leak the clamp produced

*Left in the source:* why the close must be modular rather than clamped, which is the rule a later editor must not undo.

```text
    # v5.30 (audit E-M7): close the lattice with ``% cell_N``, NOT
    # ``clip(..., 0, cell_N - 1)``.  ``in_cell`` lives in ``[0, cell_extent)``
    # so ``round(in_cell / cell_pixel_size)`` reaches ``cell_N`` for any sample
    # in the last HALF pixel of the cell -- and the nearest cell pixel there is
    # the WRAPPED one, index 0, not the clipped ``cell_N - 1``.  Clipping folded
    # that whole half-pixel into the last column, breaking the periodicity the
    # function's own docstring promises: measured on the grid-native
    # 8-pixel/cell case (N=256, dx = cell_pixel_size) the per-cell-pixel
    # occupancy was [21, 32, 32, 32, 32, 32, 32, 43] (spread 22 of 32) instead
    # of a uniform 32, 2816 of 65536 mask pixels carried the wrong phase
    # (max|dt| = 2.0, a full 0<->pi flip), and the tiled 0/pi 50 %-duty binary
    # grating leaked 11.3 % of its power OFF the order lattice with 2.95 %
    # landing in the EVEN (nominally forbidden) orders -- both exactly 0 with
    # the modulo close.
```

### L505-507 -- `create_fresnel_zone_plate` docstring, Notes -- the 4.7 ``_m``-suffix rename

*Left in the source:* nothing -- the rename predates every supported version and the parameter names are documented above.

```text

    4.7 dropped the historical ``_m`` suffix from ``dx`` /
    ``focal_length`` / ``wavelength``.
```

### L522-526 -- `create_fresnel_zone_plate`, the focal-length guard -- "Negative focal_length used to silently strip the sign"

*Left in the source:* the same failure as the reason for the guard, plus the recipe for a diverging zone plate.

```text
    # 4.10: enforce positive focal length so the zone-plate behaves
    # as the docstring says ("Positive = converging").  Negative
    # focal_length used to silently strip the sign and produce an
    # identical converging FZP; users wanting a diverging FZP can flip
    # the sign of the applied phase elsewhere.
```

### L581-607 -- `makedammann2d` docstring, the 4.14.2 / 4.14.3 ``versionchanged`` blocks -- the micrometre heuristic, its DeprecationWarning, and the ``_legacy_units='auto'`` mode that no longer exists

*Left in the source:* the 4.14.2 unit migration and its recipe, which a pre-4.14.2 call site still needs; the 4.14.3 block described guards that the 5.30 block below supersedes entirely.

```text
    .. versionchanged:: 4.14.2
        Units are now SI metres throughout (was micrometres).  ``periodx``,
        ``periody`` and ``waveln`` are expected in **metres**; the returned
        ``cell_pixel_size`` is in metres and matches them directly (no
        internal ``* 1e-6`` rescale).  Pre-4.14.2 callers passing
        ``periodx=61.0`` / ``waveln=1.31`` (micrometres) hit a thousand-
        fold drift in ``samplingx`` that was silently masked by an
        output ``* 1e-6`` rescale.  v4.14.2 emits a ``DeprecationWarning``
        when ``periodx``, ``periody`` or ``waveln`` look like micrometres
        (heuristic: ``periodx > 1e-3 m`` or ``waveln > 1e-3 m``, i.e.
        larger than 1 mm) and treats the inputs as legacy micrometre
        values; remove the warning by converting your call sites to SI
        (multiply old values by ``1e-6``).

    .. versionchanged:: 4.14.3
        The ``> 1e-3`` legacy-um heuristic silently miscompiled THz / MMW
        designs where SI-correct ``periodx`` / ``waveln`` legitimately
        exceed 1 mm (e.g. 5 mm grating period at 1.1 mm far-IR
        wavelength).  Two guards added: (1) inputs above 1 m are
        rejected as unambiguously wrong (``ValueError``); (2) an
        explicit ``_legacy_units`` kwarg (``'auto'`` / ``'um'`` /
        ``'SI'``) lets THz / MMW users bypass the heuristic.  Pass
        ``_legacy_units='SI'`` to opt out of the auto rescale and
        accept mm-scale inputs as SI metres.  Supported wavelength
        range: 10 nm (``1e-8 m``) -- 1 mm (``1e-3 m``) under the
        ``'auto'`` mode; up to 1 m under ``'SI'`` mode.
```

### L608-628 -- `makedammann2d` docstring, the 5.30 ``versionchanged`` block -- the measured 1e-6-wrong THz design and the argument for retiring the shim rather than deprecating it again

*Left in the source:* what ``_legacy_units`` accepts NOW, that ``'auto'`` raises, and the supported migration path for genuine micrometre call sites -- the whole of what a caller has to act on.

```text
    .. versionchanged:: 5.30
        ``_legacy_units`` default flipped ``'auto'`` -> ``'SI'`` **and the
        micrometre auto-detect mode was removed entirely** (audit E-H11,
        ``AUDIT_ADVERSARIAL_CODEBASE_2026_07_25``, executed in the W5
        shim-removal wave).  The ``'auto'`` heuristic silently multiplied
        any ``periodx`` / ``periody`` / ``waveln`` above 1 mm by ``1e-6``,
        so a physically correct SI THz / MMW design (8 mm period at 1.1 mm
        wavelength) came back with 5e-10 m cells -- a factor 1e-6 wrong --
        and the only diagnostic was a ``DeprecationWarning``, which is
        suppressed by default outside ``__main__``.  SI metres are now
        taken at face value: no rescale, no warning.  A shim that
        SILENTLY REWRITES physical inputs cannot be left reachable once
        it is known wrong for a legitimate design regime, so ``'auto'``
        was retired rather than given another cycle: passing
        ``_legacy_units='auto'`` now raises ``ValueError`` naming the two
        surviving modes.  Explicit ``_legacy_units='um'`` is unchanged and
        is the supported migration path for genuine micrometre call
        sites: ``'auto'``'s per-parameter rescale is reproduced exactly by
        ``'um'`` whenever every parameter was micrometre-valued (its own
        documented use case), and a hybrid call must state which values
        are which rather than have a magnitude heuristic guess.
```

### L669-670 -- `makedammann2d` docstring, ``cell_pixels`` -- "keeps the historical behaviour"

*Left in the source:* what the default actually is, which is spelled out on the next line.

```text
        ``wavsamp``-derived grid size.  ``None`` (default) keeps the
        historical behaviour: the cell is
```

### L740-756 -- `makedammann2d`, the ``_legacy_units`` dispatch -- the W5 shim-removal narrative around the retired ``'auto'`` mode

*Left in the source:* the two live modes, the reason there is no third, and the precedent for spelling the rejection out.

```text
    # v4.14.3: dispatch on ``_legacy_units``.  Two modes (v5.30):
    #
    #   'SI'   -- pass-through, accept mm-scale inputs as SI metres
    #             (intended for THz / MMW designs).  DEFAULT since
    #             v5.30 (audit E-H11).
    #   'um'   -- explicit legacy micrometres; rescale unconditionally
    #             with NO warning.
    #
    # v5.30 (W5 shim-removal wave): the v4.14.2 ``'auto'`` heuristic is
    # REMOVED.  It rescaled by 1e-6 whenever a value exceeded 1 mm, which
    # is exactly wrong for a physical THz / MMW design; retiring it to an
    # explicit opt-in (v5.30 audit E-H11) was the first step, deleting it
    # the second.  A shim that silently rewrites physical inputs is not
    # given another cycle.  Precedent for the explicit named rejection:
    # ``propagators/system.py`` ``_reject_legacy`` (v5.0 aperture schema)
    # -- the mode name INTERCEPTED VALUES, so a bare "not valid" message
    # would leave a migrating caller without the recipe.
```

### L988-990 -- `makedammann2d`, the progress plot -- "match the historical user expectation"

*Left in the source:* what the rescale is for: the axes are drawn in micrometres.

```text
            # v4.14.2: ``rxscal``/``ryscal`` are now in SI metres
            # (post-SI conversion); scale to micrometres for display
            # so the plot axes match the historical user expectation.
```

### L1145-1148 -- `load_fits_field`, the split amp/phase auto-detect -- "Pre-fix the default save -> default load round-trip ... SILENTLY dropped all phase"

*Left in the source:* the same round-trip as the reason the auto-detect exists.

```text
            # tagged ``EXTNAME='PHASE'`` / ``BUNIT='radians'``).  Pre-fix the
            # default save -> default load (``hdu_phase=None``) round-trip
            # took this "amplitude only" branch and SILENTLY dropped all
            # phase (mirroring the real/imag-stack auto-detection above).
```
