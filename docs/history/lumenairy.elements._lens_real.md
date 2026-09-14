<!-- lumenairy-history-doc
module: lumenairy/elements/_lens_real.py
ast_sha256: 78c106c88ebba4c7de44092921427d5db501abe18b8e794ca5f6e9e292ec5f02
token_sha256: a89f4581bf67954d704bfd01361abddd5aa996919d5cf50b1a019b3826dc9fab
pre_relocation_lines: 8117
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-B2 (audit 2026-09-11 L9): the 2-D displaced remap's launch->exit map is inverted on its own structured launch grid instead of Delaunay-triangulating the scattered exit points (the scattered backend is retained as its oracle); the carried input envelope is cut at the largest centred window the field grid holds, which is what actually made the launch lattice reflection-unstable; the lattice is raised 181 -> 257 and exposed as a validated displaced_n_side keyword
re_recorded: 2026-09-13 -- WP-B3b (K6): _propagate_through_glass's two resample_field calls gate method= on whether the lens grid's window fits inside one chirp-Z reconstruction period; in glass lam_medium = wavelength/n puts these legs on the spline side far more often than the free-space chain
re_recorded: 2026-09-13 -- VERIFY-WP-B2 (audit 2026-09-11 L9 re-verification): the 1-D symmetric remap _apply_displaced_remap cuts its carried envelope at the largest centred window the field grid holds, closing the same input-window asymmetry WP-B2 fixed in the 2-D remap and deferred here; _warn_if_remap_lattice_smooths quotes the pitch the trace actually uses (the fan is _DISP_REMAP_2D_FAN_FACTOR wider than the aperture) so the displaced_n_side it names really clears the field-pitch bar
re_recorded: 2026-09-13 -- VERIFY-B3b V5 (orchestrator): the in-glass 'fresnel' gap leg refuses an anamorphic pitch and a non-square grid, as the 'sas' branch does -- the resample back reads one pitch and one N_out, so the y axis was scaled by the x ratio
re_recorded: 2026-09-13 -- the row-band schedule the chunked surface paths iterate is one generator (_row_bands / _band_in_halo); four copies of the halo arithmetic removed, 56/56 bit-identical (WP-B11a item 3)
re_recorded: 2026-09-13 -- the grid-versus-aperture bookkeeping moved to the new elements/_lens_kernels.py leaf and is re-exported; _lens_traced reads the leaf, closing its module-level 2-cycle with the lenses facade (WP-B11a item 4)
re_recorded: 2026-09-14 -- apply_real_lens gains physics=LensPhysics, the fourth configuration object: the nine analytic-screen model-term switches become config fields, purely additive.
re_recorded: 2026-09-14 -- WP-B11b item 8: every warning in the two lens bodies asks _lens_kernels.caller_stacklevel for its level, so it names the first frame outside the package whatever wrapper / configuration re-entry reached it; doe.py's zone-plate fill takes T's own dtype.
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

---

## WP-B2 (2026-09-13) -- the 2-D displaced remap's launch lattice and its inversion

Recorded here rather than in the source, per `CONTRIBUTING.md`: the source says
what the code does now and why, and this is the "what it used to do, what that
was thought to be, and what it turned out to be" that would otherwise
accumulate on `_DISP_REMAP_2D_N_SIDE` and `_apply_displaced_remap_2d`.

### The finding

Audit `AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11.md` L9 (P2): *"2-D displaced
remap resolution fixed by `n_side=181` and Delaunay."*  WP-A2 fixed the
silence -- `_warn_if_remap_lattice_smooths` -- but left the lattice at 181,
because raising it alone was a measured regression:

> Scored by the mirror residual of the image-plane intensity on
> `test_niche_p10_transverse_walk_remap.py`'s fixture (N = 512, f/5 singlet,
> d = 0.6 mm): 7.9e-14 at n_side = 181, 5.5e-14 at 257, 4.1e-14 at 513 -- but
> **4.1e-03 at 512 and 7.4e-03 at 1025**.  The instability is the Delaunay
> backend's, not the resolution's: a denser scattered set hands QHull more
> near-degenerate cells to resolve arbitrarily, and which way it resolves them
> is not reflection-stable.

That reading reproduces exactly on this build (relL2 7.927e-14 / 5.498e-14 /
6.288e-03 / 4.124e-14 / 7.409e-03 at 181 / 257 / 512 / 513 / 1025; the "4.1e-03
at 512" WP-A2 quoted is the EE80 column of the same sweep, 4.082e-03).

### What it actually was

The attribution to QHull was wrong.  Running the same sweep through the new
structured inversion, with no triangulation anywhere, gave 6.807e-03 at 512 and
3.888e-03 at 1025 -- the same instability, same lattices.  The cause is one
line upstream of both backends:

```python
    cx = X0.ravel() / dx + Nx / 2.0
    amp_in = map_coordinates(_F.real, [cy, cx], order=1,
                             mode='constant', cval=0.0) + 1j * (...)
```

The field axis `(arange(N) - N/2) * dx` runs from `-(N/2) dx` to `+(N/2 - 1) dx`
-- one whole sample further on the `-x` side.  A ray launched in the band
`(x[-1], x[-1] + dx]` therefore samples off the grid and carries nothing, while
its mirror between `x[0] - dx` and `x[0]` carries the full envelope.  On the p10
fixture that is 0.63 of the peak, and whether any ray lands in the band is
decided by the launch pitch:

| n_side | dstep [um] | a ray in the band? | mirror-asymmetric launch samples | max amp asymmetry |
|---|---|---|---|---|
| 181 | 57.22 | no | 0 | 4.4e-16 |
| 257 | 40.23 | no | 0 | 3.3e-16 |
| 512 | 20.16 | **yes** | 406 | 6.281e-01 |
| 513 | 20.12 | no | 0 | 3.3e-16 |
| 1025 | 10.06 | **yes** | 812 | 6.292e-01 |

which is the whole of the "raising the lattice breaks the mirror symmetry"
effect, and why it looked like a property of the lattice.  `_apply_displaced_
remap_2d` now cuts the carried envelope at the largest CENTRED window the grid
holds (`|x| <= x[-1]`, `|y| <= y[-1]`), and the mirror residual reads 2.9e-14
to 8.5e-12 across 181, 257, 512, 513, 1025 and 2049, on both backends -- nine
decades below the readings it replaced.

### The backend that was replaced

`_apply_displaced_remap_2d` rebuilt the exit field with a single
`LinearNDInterpolator` over the scattered exit points (one 3-column
triangulation shared by the complex amplitude and the OPL -- the K3 / N15
perf work).  That code is not gone: it is `_remap2d_interp_delaunay`, reachable
as `interp_method='delaunay'`, and it is the oracle the structured inversion is
checked against in
`tests/unit/test_audit2609_b2_displaced_remap_inversion.py`.

Byte-identity between the two was never achievable and was not claimed:
barycentric interpolation over the exit triangulation and bilinear
interpolation in launch space are different second-order approximations of the
same map.  Measured against a ray-exact oracle (the same fan, each field
point's launch coordinate found by Newton on the TRUE trace, so the oracle has
no lattice at all), peak-relative over the illuminated core of a 0.7 mm beam at
N = 512:

| n_side | delaunay \|E\| rms | structured \|E\| rms | delaunay phase | structured phase |
|---|---|---|---|---|
| 181 | 9.213e-04 | 8.722e-04 | 4.715e-02 | 4.715e-02 |
| 257 | 4.571e-04 | 4.251e-04 | 2.410e-02 | 2.410e-02 |
| 513 | 1.156e-04 | 1.079e-04 | 5.882e-03 | 5.883e-03 |
| 1025 | 2.918e-05 | 2.743e-05 | 1.439e-03 | 1.439e-03 |

-- second order in the launch pitch for both, with the structured backend 5-6 %
closer in amplitude and identical in phase.  Where the two genuinely differ is
a truncated pupil: the hull ends at the outermost retained exit point, so the
scattered backend leaves a ring of exactly-zero pixels inside the illuminated
region (61 to 360 of 3782 sampled core points, lattice-dependent) and loses the
power in it (0.97720 against 0.98658 of the input at n_side = 181).

---

## VERIFY-WP-B2 (2026-09-13) -- the same window, in the 1-D remap

The WP-B2 pass above recorded the 1-D symmetric remap's identical input-window
asymmetry as deferred work, on the reading that "it is rotationally symmetric,
so no fixture in the suite exercises a mirror pair through it".  Re-measured
independently, the exposure is larger than that: the asymmetry shows up with a
CENTRED input through a ROTATIONALLY SYMMETRIC element, which is the simplest
call the path has.

`_apply_displaced_remap` reads the input at the ENTRANCE height
`X * scale`, `scale = h_in / r_out`.  For a converging element the ray walks
inward, so `scale > 1` and the read runs off the `+x` end of the axis
`(arange(N) - N/2) * dx` while its mirror -- one whole sample further out on
`-x` -- is still on the grid and returns the full envelope.  Measured on a
symmetric f/5-class singlet (`R = 42.5 / -63 mm`, 4.2 mm of n = 1.5093 glass),
`displaced_mode='remap'`, before the fix:

| grid | paired mirror relL2 of \|E\| | pixels off by > 1e-9 of peak | worst pixel |
|---|---|---|---|
| N = 640, dx = 6.5 um, w0 = 2.4 mm | 3.301e-02 | 1236 | 0.489 of peak against an exact 0 on its mirror |
| N = 512, dx = 8 um, w0 = 3.0 mm | 4.586e-02 | 988 | 0.650 of peak against an exact 0 on its mirror |

-- a crescent of dead pixels on `+x` only, from a rotationally symmetric system
on a rotationally symmetric input.  The same rule the 2-D remap uses now
applies here: the carried envelope is cut at the largest CENTRED window the
grid holds (`|X * scale| <= x[-1]`, `|Y * scale| <= y[-1]`), and the two
readings above become 1.18e-16 and 1.01e-16 with zero pixels off.  The price is
the outermost ring of the input on BOTH sides instead of one side.

The deferral's stated blocker -- that the fix "moves the byte-identity pin"
`test_niche_p10_...::test_symmetric_remap_is_the_p2_1d_remap_byte_identical` --
does not hold: that pin compares `apply_real_lens(displaced_mode='remap')`
against a direct call to `_apply_displaced_remap`, so both sides move together
and the pin still passes.

## VERIFY-WP-B2 (2026-09-13) -- the launch pitch the caller is told about

`_warn_if_remap_lattice_smooths` scored the launch pitch as
`2 * r_aperture / (n_side - 1)`, but the fan is thrown
`_DISP_REMAP_2D_FAN_FACTOR = 1.03` wider than the aperture, so the pitch the
trace actually uses is 3 % coarser.  The consequence was in the message's own
advice: at a 10 mm aperture and dx = 8 um it named `displaced_n_side=626`,
whose real pitch is 16.48 um against the 16.00 um bar the message says that
value clears (dx = 4 um: it named 1251, real pitch 8.24 um against 8.00 um).
Both the quoted pitch and the named lattice are now computed from the fan the
trace throws, and the fan factor is one constant read by the builder and the
warning, so the same two calls name 645 and 1289 -- lattices whose real pitch
(15.99 um and 8.00 um) does clear the bar.
