<!-- lumenairy-history-doc
module: lumenairy/elements/_lens_real.py
ast_sha256: f9d76940ddabdd9c5060b9b3c74dc52f33c1901dca0654e76789a7530f782bf9
token_sha256: 70f5975ca289211a7fbc3b27ddb0155ebe8ac5a5f01109d3921ed75d6b1612e0
pre_relocation_lines: 8117
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
-->

# Version history -- `lumenairy/elements/_lens_real.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/_lens_real.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks, the comments that corrected earlier
comments, and the per-release chronologies that had accumulated on constants
whose CURRENT value is what the source now states.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

What did NOT move: the measured derivations of the live constants and
defaults (`docs/TESTING_STANDARDS.md` S5) -- the `_GLASS_VALUE_CACHE` sizing,
the `_NUMEXPR_MIN_SIZE` crossover, the displaced-model oracle tables, the
remap fold/pull-back guard bars, the tangent-facet derivation and the
byte-identity statements the banded paths rest on.  The live migration
statements (`surface_frame`, `carrier=`, `surface_model=`) stayed as well.


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
| L137-141 | `_NUMEXPR_MIN_SIZE` | the v5.30 E-L5 record that this comment used to point at a dead twin in `lenses.py` |
| L1833-1836 | `_demodulated_input` docstring | "used to be thrown away with the ``np.abs(E_in)`` sampling" and the two measurements of that loss |
| L1979-1981 | `_warn_if_remap_lattice_smooths` docstring | "-- and nothing used to say so", the reason the warning was added |
| L3723-3726 | `_REMAP_SUPPORT_AMP_FRAC` | "a converging beam's perfectly ordinary 8x pad used to REFUSE on dark corner pixels" written as the defect that prompted the constant |
| L5014-5019 | `apply_real_lens` docstring, ``seidel_poly_order`` | the v5.30 E-M1 `.. note::` recording that this entry used to read "default 8" while the signature shipped 6 |
| L6241-6243 | `_obl_gap_advance` docstring | "pre-v5.35.3 they ``continue``d past it, which is precisely why ``carrier=`` had to disqualify them" |
| L6642-6649 | the row-band qualification test, ``freeform_type`` | "Pre-fix the band loop silently dropped the departure for non-Q types with no diagnostic" |
| L7072-7077 | the Forbes Q-bfs / Q-con freeform arm | the v4.15.1 P3-NEW-A framing ("is explicitly delegated to this module for closure") on a live dispatch |
| L7221-7224 | the surface-normal gradient | "4.10: pass dy ... pre-4.10 used dx for both" -- the anamorphic defect as a release note |
| L7239-7247 | the cosine-clamp warning | the v5.25.0 H1 note that the clamp is no longer harmful for the OPD, appended under the round-2 M-LR note that said it was |
| L7528-7535 | the Fresnel amplitude transmission | "4.10: ... Pre-4.10 used 0.5*(t_s+t_p) which only matches 45-deg linear polarisation at low AOI" |
| L7558-7566 | the TIR mask | "audit #3.5: was inside `if fresnel:` pre-4.9" and what that left slant_correction users with |
| L7588-7597 | the aperture stop at a decentred surface | a three-release chronology on one four-line lookup: 4.10.2's patch, 4.11.1's correction of it (wrong accessor, wrong keys) and v4.13.2's dtype-aware zero |
| L7917-7927 | `PreparedRealLens` docstring | "Pre-v5.29.1 the class hard-coded ASM / ``dy = dx`` / float64 geometry and never consulted the defaults at all", with the measured 49.6 divergence |
| L7971-7977 | `PreparedRealLens.__call__`, the gap propagation | "(this used to hard-code ASM)" |
| L8053-8058 | `prepare_real_lens`, the default resolution | "Pre-fix this function hard-coded ASM / dy=dx / float64 geometry and never consulted the defaults", with the measured 49.6 divergence |

---

### L137-141 -- `_NUMEXPR_MIN_SIZE` -- the v5.30 E-L5 record that this comment used to point at a dead twin in `lenses.py`

*Left in the source:* that this is the only live copy and that the propagators keep their own constant

```text
# while the benefit scales with the array size.  This is the ONLY live copy of
# the constant (v5.30, audit E-L5: the dead twin in ``lenses.py`` -- which this
# comment used to point at for the rationale -- has been deleted; the rationale
# now lives here, next to its three readers below).  The propagators keep their
# own ``asm._NE_MIN_SIZE``, deliberately separate.
```

### L1833-1836 -- `_demodulated_input` docstring -- "used to be thrown away with the ``np.abs(E_in)`` sampling" and the two measurements of that loss

*Left in the source:* what the demodulation carries; the bit-identity statement for the collimated case stays below

```text
    ``conjugate=None``) the whole of its curvature -- is not in the trace and
    used to be thrown away with the ``np.abs(E_in)`` sampling: measured
    identical output (4.7e-16) for a flat and a 35-wave-defocused input, and a
    150 mm diverging source focusing at the COLLIMATED 21 mm instead of 25 mm.
```

### L1979-1981 -- `_warn_if_remap_lattice_smooths` docstring -- "-- and nothing used to say so", the reason the warning was added

*Left in the source:* what the warning is about and the measured contrast loss

```text
    pitch is not propagated, it is SMOOTHED AWAY -- and nothing used to say so.
    Measured: a ripple at 2.2 launch samples per period comes back at 0.51 of
    its input contrast where the (field-grid) screen path resolves it at 1.26.
```

### L3723-3726 -- `_REMAP_SUPPORT_AMP_FRAC` -- "a converging beam's perfectly ordinary 8x pad used to REFUSE on dark corner pixels" written as the defect that prompted the constant

*Left in the source:* why the guards are scored over the support, with the same measured numbers, in the present tense

```text
#: beam's perfectly ordinary 8x pad used to REFUSE on dark corner pixels while
#: the illuminated pupil sat at ``det = 0.9986``, three orders inside the bar.
#: Scoring the guards over the support fixes the misdiagnosis (the message said
#: "change model"; the actual remedy was "shrink the grid").  1e-6 of peak
```

### L5014-5019 -- `apply_real_lens` docstring, ``seidel_poly_order`` -- the v5.30 E-M1 `.. note::` recording that this entry used to read "default 8" while the signature shipped 6

*Left in the source:* the parameter text itself, which now states the shipped default once

```text

        .. note::
           v5.30 (audit E-M1): this entry read "default 8" while the
           signature has shipped ``seidel_poly_order=6`` (and the UI's
           lens-options dialog defaults to 6).  The DOC was wrong -- the
           behaviour is unchanged.
```

### L6241-6243 -- `_obl_gap_advance` docstring -- "pre-v5.35.3 they ``continue``d past it, which is precisely why ``carrier=`` had to disqualify them"

*Left in the source:* why the advance is factored out: both row-banded paths must reach it

```text
        Factored out of the whole-grid surface body so the two row-banded
        paths reach it too -- pre-v5.35.3 they ``continue``d past it, which is
        precisely why ``carrier=`` had to disqualify them."""
```

### L6642-6649 -- the row-band qualification test, ``freeform_type`` -- "Pre-fix the band loop silently dropped the departure for non-Q types with no diagnostic"

*Left in the source:* the live rule -- why ANY freeform_type falls through to the whole-grid path, stated for both the Q and the non-Q families

```text
            # v5.17.1 (audit P2-04): ANY freeform_type falls through to the
            # whole-grid path -- Q-bfs / Q-con so their departure IS
            # computed there, and the non-Q types (zernike / xy_polynomial
            # / chebyshev) so the whole-grid path's "freeform departure is
            # NOT included" RuntimeWarning keeps firing on the (default)
            # banded path.  Pre-fix the band loop silently dropped the
            # departure for non-Q types with no diagnostic.  Outputs are
            # unchanged (the departure was dropped on both paths).
```

### L7072-7077 -- the Forbes Q-bfs / Q-con freeform arm -- the v4.15.1 P3-NEW-A framing ("is explicitly delegated to this module for closure") on a live dispatch

*Left in the source:* what the arm does and why -- the Q families' departure is a 2-D scalar phase contribution that is added to the base conic sag

```text
        # v4.15.1 (P3-NEW-A): Forbes Q-bfs / Q-con sag is a 2-D scalar
        # phase contribution exactly analogous to the (forthcoming)
        # xy-polynomial / Zernike / Chebyshev wave-optics paths and is
        # explicitly delegated to this module for closure -- so for
        # ``freeform_type in ('q_bfs', 'q_con')`` we compute the
        # freeform departure here and ADD it to the base conic sag.
```

### L7221-7224 -- the surface-normal gradient -- "4.10: pass dy ... pre-4.10 used dx for both" -- the anamorphic defect as a release note

*Left in the source:* the rule: ``np.gradient`` takes the spacing in array-axis order, and both axes must be passed

```text
            # 4.10: pass dy for the y-axis spacing -- pre-4.10 used dx
            # for both, which gave the wrong surface-normal direction on
            # anamorphic grids (dx != dy).  np.gradient takes the spacing
            # in the same order as the array axes (y, x).
```

### L7239-7247 -- the cosine-clamp warning -- the v5.25.0 H1 note that the clamp is no longer harmful for the OPD, appended under the round-2 M-LR note that said it was

*Left in the source:* the live scope of the warning: the *cos form cannot diverge, so the clamp only matters for the Fresnel legs, which still divide by cos

```text
            # 4.10: warn the FIRST time per call we clamp a real ray's
            # cosine.  The 1e-3 floor (≈89.94°) was previously silent;
            # for steep aspheres or strongly tilted bundles it acts on
            # physical (non-TIR) rays before the TIR mask fires, and
            # the historical /cos slant OPD blew up ~1000x per
            # clamped pixel (round-2 audit M-LR).  v5.25.0 (H1): the
            # corrected *cos form cannot diverge, so the clamp is now
            # harmless for the OPD -- the warning is kept for the
            # Fresnel-coefficient legs, which still divide by cos.
```

### L7528-7535 -- the Fresnel amplitude transmission -- "4.10: ... Pre-4.10 used 0.5*(t_s+t_p) which only matches 45-deg linear polarisation at low AOI"

*Left in the source:* the rule and the reason -- average the INTENSITY coefficients, because t_s and t_p have different phases at high AOI

```text
            # 4.10: average the INTENSITY coefficients for unpolarised
            # scalar throughput, not the amplitude coefficients.  At
            # Brewster's angle (or any high AOI), t_s and t_p have
            # different phases; their amplitude sum can cancel where
            # sqrt(0.5*(|t_s|^2+|t_p|^2)) correctly captures the
            # incoherent average power.  Pre-4.10 used 0.5*(t_s+t_p)
            # which only matches 45-deg linear polarisation at low AOI.
            # For polarised inputs route through the Jones pipeline.
```

### L7558-7566 -- the TIR mask -- "audit #3.5: was inside `if fresnel:` pre-4.9" and what that left slant_correction users with

*Left in the source:* the rule -- the mask must fire whenever ``sin2_tt`` was computed -- and the unphysical residual amplitude that follows if it does not

```text
        # ---- TIR mask (audit #3.5: was inside `if fresnel:` pre-4.9) --
        # Suppress regions that went into total internal reflection.
        # This must fire whenever ``sin2_tt`` was computed -- i.e. for
        # both ``fresnel=True`` and ``slant_correction=True`` paths,
        # since the slant OPD divides by ``cos_tt_safe`` which is
        # ill-defined where ``sin2_tt > 1``.  Pre-4.9 only ran this
        # inside the Fresnel block, leaving slant_correction=True +
        # fresnel=False users with unphysical residual field amplitude
        # in TIR regions.
```

### L7588-7597 -- the aperture stop at a decentred surface -- a three-release chronology on one four-line lookup: 4.10.2's patch, 4.11.1's correction of it (wrong accessor, wrong keys) and v4.13.2's dtype-aware zero

*Left in the source:* the rule -- respect the stop surface's own decenter, read from the ``decenter`` key's ``(dx, dy)`` tuple, with a dtype-aware zero

```text
            # 4.10.2: respect the stop surface's decenter/displacement
            # if any.  Pre-4.10.2 always used h_sq_axis (centred at the
            # optical axis), so a decentered stop was modelled at the
            # wrong location and clipped the wrong region of the beam.
            # 4.11.1: the 4.10.2 patch used ``getattr(surf, ...)`` on a
            # dict (always returns the default 0.0), and looked up the
            # wrong keys (``decenter_x_m`` / ``decenter_y_m``).  The
            # surface dict's actual key is ``decenter`` and the value
            # is a ``(dx, dy)`` tuple -- mirror line 520 above.
            # v4.13.2 (audit C-P1-4): dtype-aware zero for both branches.
```

### L7917-7927 -- `PreparedRealLens` docstring -- "Pre-v5.29.1 the class hard-coded ASM / ``dy = dx`` / float64 geometry and never consulted the defaults at all", with the measured 49.6 divergence

*Left in the source:* what a prepared lens freezes and what a caller must do to pick up a changed default

```text
    A prepared lens FREEZES the settings that were live when it was prepared
    (v5.29.1; audit E-H3).  ``wave_propagator``, ``sag_dtype`` and ``_dy`` hold
    the values :func:`prepare_real_lens` resolved from the process-wide
    defaults (:func:`set_default_wave_propagator` /
    :func:`set_lens_sag_dtype` / :func:`set_default_dy`), so a prepared object
    keeps reproducing the field it was built for even if a global default is
    flipped afterwards -- rebuild it to pick up new settings.  Pre-v5.29.1 the
    class hard-coded ASM / ``dy = dx`` / float64 geometry and never consulted
    the defaults at all, so after ``set_default_wave_propagator('fresnel')``
    the prepared object diverged from :func:`apply_real_lens` by 49.6 on a
    singlet with no diagnostic.
```

### L7971-7977 -- `PreparedRealLens.__call__`, the gap propagation -- "(this used to hard-code ASM)"

*Left in the source:* the live rule: dispatch on the propagator frozen at prepare time, through the same helper `apply_real_lens` uses

```text
                # v5.29.1 (audit E-H3): dispatch on the propagator FROZEN at
                # prepare time via the same helper apply_real_lens uses, so
                # the two agree for every propagator (this used to hard-code
                # ASM).  ``lam_med`` is already the in-medium wavelength, so
                # the helper's ``wavelength / n_medium_r`` reduces to it with
                # ``n_medium_r=1.0``; absorption is off here (the factory
                # rejects it), which makes the ``kappa`` / ``k0`` args inert.
```

### L8053-8058 -- `prepare_real_lens`, the default resolution -- "Pre-fix this function hard-coded ASM / dy=dx / float64 geometry and never consulted the defaults", with the measured 49.6 divergence

*Left in the source:* the live rule: resolve at prepare time, explicit kwargs win, freeze the resolved values -- and the pointer to this document

```text
    # v5.29.1 (audit E-H3): resolve the process-wide defaults AT PREPARE TIME
    # (explicit kwargs win, exactly as in apply_real_lens) and freeze the
    # resolved values on the returned object.  Pre-fix this function hard-coded
    # ASM / dy=dx / float64 geometry and never consulted the defaults, so a
    # later set_default_wave_propagator('fresnel') desynchronised the prepared
    # object from apply_real_lens by 49.6 with no diagnostic.
```
