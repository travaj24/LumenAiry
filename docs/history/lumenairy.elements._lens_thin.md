<!-- lumenairy-history-doc
module: lumenairy/elements/_lens_thin.py
ast_sha256: 3c29e2404db6f77845ca493ec19350197cd32c7c51457e60556cf95be6a710fb
token_sha256: 65006ad5ac3c80038a967b778d82a2675f3efc47aeb474ec40e0b3c7771aa74d
pre_relocation_lines: 1431
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- CuPy handles taken from backend._optional instead of via .lenses, breaking the _lens_thin <-> lenses module-level 2-cycle; outputs bit-identical (WP-A22 item 8a)
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
-->

# Version history -- `lumenairy/elements/_lens_thin.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/_lens_thin.py` -- the `vN.NN (audit X): pre-guard this
returned A, which was wrong because B` blocks in front of the input guards, and
the two passages that retract an earlier revision of their own docstring
(`'local_only'`, and the SA-nulling-conic guidance on
`apply_aspheric_lens`).  Each block is reproduced **verbatim** under the source
line it came from in the pre-relocation file.

This module is a thin-ELEMENT module (`apply_thin_lens`,
`apply_spherical_lens`, `apply_aspheric_lens`, `apply_cylindrical_lens`,
`apply_grin_lens`); it is not part of the traced/analytic lens family that
WP-A17 part 2 covers.

What did NOT move: the measured VALIDITY BOUNDARIES (the screen-vs-trace PV
ladder, the 503 um reference-plane offset, the orientation-blindness numbers,
the SA-nulling conic measurements, the GRIN pitch errors) -- those describe what
the shipped models do today, and `tests/unit/test_audit2609_a8_thin_elements.py`
and `tests/unit/test_niche_audit_w3_elements.py` read several of them out of the
docstrings.

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
| L111-116 | `apply_thin_lens` docstring, ``'nonparaxial'`` | thin-lens audit bug 1 -- the historical unsigned form and the wrong sign it gave for ``f < 0`` |
| L119-128 | `apply_thin_lens` docstring, ``'aplanatic'`` | thin-lens audit bug 2 -- the historical profile's wrong quartic sign, its ray mapping, and the measured focus shift |
| L143-144 | `apply_thin_lens` docstring, ``'local_only'`` | that this entry's own pre-v5.29.1 text described the opposite of what the model does |
| L161 | `apply_thin_lens` docstring, ``'local_only'`` | "the old text promised" -- a reference to the retracted wording |
| L183-208 | `apply_thin_lens` docstring, ``Raises`` | the full pre-guard split of ``f=nan`` / ``f=+-inf`` / ``f=0`` across the five lens models |
| L252-306 | `apply_thin_lens`, the ``f`` guard | the two measured pre-guard tables (non-finite ``f`` across three models; ``f = 0`` across five) and the W4-2 'recorded-not-fixed' decision |
| L363-371 | `apply_thin_lens`, the ``'nonparaxial'`` branch | bug 1 -- the historical unsigned form, and the measured f = -30 mm focus it produced at +30 mm |
| L389-402 | `apply_thin_lens`, the ``'aplanatic'`` branch | bug 2 -- the historical profile's quartic sign, its implied ray mapping, and the measured 9.1 / 7.7 / 4.27 um focus ladder |
| L481-490 | `_sphere_phase`, the ``'local_only'`` branch | that the comment which used to sit here was backwards |
| L612-614 | `apply_spherical_lens` docstring, ``aperture_diameter`` | E-M8 -- the clamped finite sag those pixels used to come back with |
| L717-732 | `_surface_sag`, the out-of-domain branch | E-M8 -- that this function used to CLAMP ``h_sq``, and the 20916 measured pixels it returned at |E| = 1 |
| L862-867 | `apply_aspheric_lens` docstring, "SA-nulling conics" | that this docstring used to prescribe the ``-n**2`` conic on the curved first surface, and the retraction of that guidance |
| L901-912 | `apply_aspheric_lens`, the odd-power guard | R-8 / E-L7 -- the measured pre-guard bit-identity of ``A1={5: 1e6}`` and ``A1={4: 1e6}`` |
| L953-970 | `_aspheric_sag`, the conic-domain gate | the stacked 4.10 and v5.30 entries, and the 352 shell pixels that used to leave this function finite |
| L1068-1072 | `apply_cylindrical_lens` docstring, ``Raises`` | the "Pre-guard, ..." before/after listing |
| L1089-1108 | `apply_cylindrical_lens`, the ``f`` guard | the v5.31 / v5.32 pair of entries and their measured pre-guard behaviour |
| L1128-1132 | `apply_cylindrical_lens`, the CuPy dispatch | that the three sibling functions had no ``use_gpu`` path before C-P1-6 |
| L1223 | `apply_grin_lens` docstring, ``thin_form`` | that ``thin_form=True`` exists to reproduce the PRE-FIX screen |
| L1280-1282 | `apply_grin_lens`, the CuPy dispatch | "(was previously numpy-only)" |

---

### L111-116 -- `apply_thin_lens` docstring, ``'nonparaxial'`` -- thin-lens audit bug 1 -- the historical unsigned form and the wrong sign it gave for ``f < 0``

*Left in the source:* what the ``sign(f)`` factor is FOR, which is the thing a reader must not 'simplify' away, plus the validity statement.

```text
            For f > 0 this equals the historical k*(f - sqrt(f**2+r**2))
            byte-for-byte; for f < 0 the historical form had the WRONG
            SIGN (it converged exactly like its +|f| twin -- thin-lens
            audit 2026-07-18, bug 1).  Exact for collimated->focus or
            focus->collimated; at FINITE conjugates it over-corrects
            (use ``'stigmatic'``).
```

### L119-128 -- `apply_thin_lens` docstring, ``'aplanatic'`` -- thin-lens audit bug 2 -- the historical profile's wrong quartic sign, its ray mapping, and the measured focus shift

*Left in the source:* the quartic sign this model carries and why the opposite one is wrong, so the profile cannot be 'restored' by a later reader.

```text
            condition domain r < |f| (unit phase outside).  The
            historical profile -k*f*(1 - sqrt(1 - r**2/f**2)) had the
            WRONG quartic sign (-k r**4/8f**3 where a converging sphere
            needs +k r**4/8f**3), so it DOUBLED paraxial's spherical
            aberration instead of removing it (thin-lens audit
            2026-07-18, bug 2; measured 9.1 um focus vs paraxial 7.7 um
            vs correct 4.27 um at NA~0.1).  Its ray mapping was
            sin(theta) = tan(asin(r/f)) -- neither the sine condition
            (whose pure-phase-screen profile is the PARAXIAL quadratic)
            nor the tangent/stigmatic condition (the spherical phase).
```

### L143-144 -- `apply_thin_lens` docstring, ``'local_only'`` -- that this entry's own pre-v5.29.1 text described the opposite of what the model does

*Left in the source:* the deprecation itself and the corrected algebra below it.

```text
            **Deprecated since v5.29.1 (audit E-H7) -- and its pre-v5.29.1
            docstring described the OPPOSITE of what it does.**  It is a
```

### L161 -- `apply_thin_lens` docstring, ``'local_only'`` -- "the old text promised" -- a reference to the retracted wording

*Left in the source:* which model actually has zero gradient at the lenslet centre, which is what a micro-lens array needs.

```text
            The NO-STEER model the old text promised is the plain decentered
```

### L183-208 -- `apply_thin_lens` docstring, ``Raises`` -- the full pre-guard split of ``f=nan`` / ``f=+-inf`` / ``f=0`` across the five lens models

*Left in the source:* the contract -- which inputs raise and why -- and the two live consequences: the Jones method is covered by the same guard, and nearby small ``f`` is untouched bit-for-bit.

```text
        If ``f`` is not finite.  v5.31 (audit W4-1, sibling of the v5.29
        W3-T4 polarization guards): pre-fix ``f=np.nan`` returned a field
        whose every pixel was ``nan+nanj`` under ``'paraxial'`` /
        ``'nonparaxial'`` -- and, under ``'aplanatic'``, silently applied
        NO lens at all (the ``r < |f|`` domain test is False everywhere for
        NaN, so the unit-phase sentinel covers the whole grid).  ``f=+-inf``
        inverted that split: a flat no-op under ``'paraxial'``, all-NaN
        under the two sqrt models.  Nothing was raised in any case, and
        :meth:`~lumenairy.elements.polarization.JonesField.apply_thin_lens`
        -- which routes both components through here and mutates itself in
        place -- propagated the poison into the caller's own object.
    ValueError
        If ``f == 0``.  v5.32 (audit W5-2) closes the inconsistency W4-2
        measured and recorded but deliberately did not fix: ``f = 0`` is
        physically meaningless for a thin lens (an infinite-power surface
        with no realisation), yet the five ``lens_model`` branches split
        three ways -- ``'paraxial'`` / ``'stigmatic'`` / ``'local_only'``
        raised a bare ``ZeroDivisionError('division by zero')`` naming
        neither this function nor the argument, while ``'nonparaxial'``
        and ``'aplanatic'`` returned ``E_in`` EXACTLY UNCHANGED (no lens
        applied at all -- silently for the former; with two bare numpy
        ``RuntimeWarning``\\ s for the latter).  The guard sits above the
        ``lens_model`` dispatch, so all five now raise identically, and
        :func:`apply_cylindrical_lens` was given the matching guard in the
        same change.  Nearby small-``f`` values are untouched
        bit-for-bit -- only exact zero is rejected.
```

### L252-306 -- `apply_thin_lens`, the ``f`` guard -- the two measured pre-guard tables (non-finite ``f`` across three models; ``f = 0`` across five) and the W4-2 'recorded-not-fixed' decision

*Left in the source:* why the guard sits above the dispatch, what it protects (including the Jones method), and the note that an infinite focal length means 'no lens' rather than something to reduce.

```text
    # v5.31 (audit W4-1, sibling of the v5.29 W3-T4 polarization guards): a
    # non-finite ``f`` used to sail straight through into the phase, and the
    # three lens models disagreed about the damage -- so the SAME bad input was
    # a NaN field, a silent no-op, or nothing at all depending on
    # ``lens_model``.  Measured pre-guard (N = 32, dx = 5 um, lambda = 1.55 um,
    # fraction of non-finite output pixels):
    #
    #   f          paraxial   nonparaxial   aplanatic
    #   nan        1.000      1.000         0.000  (silent NO-OP: the
    #                                              ``r**2/f**2 < 1`` domain
    #                                              test is False everywhere, so
    #                                              the unit-phase sentinel wins
    #                                              and NO lens is applied)
    #   +-inf      0.000      1.000         1.000  (paraxial's ``k/(2f) -> 0``
    #                                              is a flat no-op; the two
    #                                              sqrt models hit inf - inf)
    #
    # ``JonesField.apply_thin_lens`` (polarization.py) routes BOTH components
    # through here and mutates itself in place while returning ``self``, so
    # ``f=nan`` poisoned the caller's own object: every pixel of Ex AND Ey came
    # back ``nan+nanj``, ``degree_of_polarization`` and every Stokes parameter
    # read all-NaN, and nothing was raised.  Rejecting non-finite ``f`` here
    # fixes the scalar entry point and the Jones method together.  Note ``f``
    # is NOT periodic, so there is nothing for the caller to reduce -- an
    # infinite focal length means "no lens", which is what omitting the call
    # expresses; pass a large finite ``f`` if you want the residual power.
    #
    # v5.32 (audit W5-2): the guard now also rejects ``f == 0``, which
    # W4-2 measured and deliberately left alone as "recorded-not-fixed"
    # (a pre-existing inconsistency needing its own decision).  That
    # decision: ``f = 0`` is physically meaningless for a thin lens --
    # an infinite-power surface with no realisation -- so it raises,
    # like every other unusable ``f``.  Measured pre-guard (N = 16,
    # dx = 5 um, lambda = 1.55 um, f = 0.0), the FIVE ``lens_model``
    # branches split THREE ways:
    #
    #   lens_model    f = 0 pre-v5.32
    #   paraxial      ZeroDivisionError('division by zero')  <- k/(2f)
    #   nonparaxial   silent NO-OP, exactly E_in back, no warning
    #   aplanatic     silent NO-OP, exactly E_in back, + 2 bare
    #                 RuntimeWarnings (invalid value / divide by zero
    #                 from ``r_sq / f**2``)
    #   stigmatic     ZeroDivisionError                      <- 1.0/f
    #   local_only    ZeroDivisionError
    #
    # The two sqrt models no-op because ``sqrt(0 + r_sq) - |0| = r``
    # times ``sign(0) = 0`` is a zero phase, and 'aplanatic' additionally
    # fails its ``r**2/f**2 < 1`` domain test everywhere so the
    # unit-phase sentinel covers the whole grid.  So the SAME meaningless
    # input was a bare uninformative exception naming neither this
    # function nor the argument, OR the caller's field handed straight
    # back with no lens applied and (in one arm) two anonymous numpy
    # warnings.  Guarding here -- above the ``lens_model`` dispatch --
    # makes all five agree, and covers ``JonesField.apply_thin_lens``
    # for f=0 exactly as it already does for non-finite f.
```

### L363-371 -- `apply_thin_lens`, the ``'nonparaxial'`` branch -- bug 1 -- the historical unsigned form, and the measured f = -30 mm focus it produced at +30 mm

*Left in the source:* the sign-safe profile and what the ``sign(f)`` factor is FOR, immediately above the RATIONALIZED cancellation note (which stays in full).

```text
        # v5.25.0 (thin-lens audit 2026-07-18, bug 1): the historical
        # ``exp(1j*k*(f - sqrt(f**2 + r_sq)))`` expands, for f < 0, to a
        # CONVERGING quadratic -k r**2/(2|f|) -- a diverging lens that
        # focused identically to its +|f| twin (measured: f = -30 mm
        # produced the same z = +30 mm focus and peak as f = +30 mm).
        # The sign-safe stigmatic sphere is
        # ``phi = -sign(f) * k * (sqrt(r**2 + f**2) - |f|)``, which is
        # byte-identical to the historical form for f > 0 (IEEE negation
        # of an exact subtraction) and correctly DIVERGES for f < 0.
```

### L389-402 -- `apply_thin_lens`, the ``'aplanatic'`` branch -- bug 2 -- the historical profile's quartic sign, its implied ray mapping, and the measured 9.1 / 7.7 / 4.27 um focus ladder

*Left in the source:* what the branch computes and the apodization warning, which is the part a caller can act on.

```text
        # v5.25.0 (thin-lens audit 2026-07-18, bug 2): the historical
        # profile ``-k*f*(1 - sqrt(1 - r**2/f**2))`` expands to
        # ``-k r**2/2f - k r**4/8f**3`` -- the WRONG quartic sign (a
        # converging sphere needs ``+k r**4/8f**3``), so it carried 2x
        # paraxial's spherical-aberration error in the SAME direction
        # and focused WORSE than paraxial (9.1 um vs 7.7 um vs correct
        # 4.27 um at NA~0.1).  Its implied ray mapping,
        # sin(theta) = tan(asin(r/f)), is neither the Abbe sine
        # condition (whose pure-phase-screen profile is the PARAXIAL
        # quadratic) nor the stigmatic tangent condition.  The corrected
        # phase is the exact stigmatic sphere restricted to the
        # sine-condition domain r < |f|; the sqrt(cos theta) pupil
        # APODIZATION that distinguishes a true aplanat is an AMPLITUDE
        # factor a pure phase mask must not apply (see docstring).
```

### L481-490 -- `_sphere_phase`, the ``'local_only'`` branch -- that the comment which used to sit here was backwards

*Left in the source:* the corrected algebra and the deprecation policy, stated once.

```text
        # DEPRECATED (v5.29.1, audit E-H7).  The comment that used to sit here
        # -- "the standard decentered quadratic minus the linear tilt that
        # would otherwise steer the beam" -- was backwards, as was the
        # docstring: the sum below expands to an ORIGIN-centred parabola plus a
        # constant piston, so the local gradient at (xc, yc) is -k*xc/f and the
        # sub-beam is steered ONTO THE AXIS.  The zero-gradient (no-steer)
        # model is the plain decentered 'paraxial' lens.  Behaviour is
        # deliberately unchanged (0 callers in-repo, but it is a public enum
        # value -- deprecate, don't break); see the docstring for the algebra
        # and the measured -20.0 mrad steer.
```

### L612-614 -- `apply_spherical_lens` docstring, ``aperture_diameter`` -- E-M8 -- the clamped finite sag those pixels used to come back with

*Left in the source:* the NaN convention itself, which is the contract, and the library-wide siblings it matches.

```text
        come back **NaN** (v5.30, audit E-M8 -- they used to come back
        with a clamped, finite sag of ``0.99 R`` and ``|E| = 1``, a
        transmission through a nonexistent surface).  NaN is the
```

### L717-732 -- `_surface_sag`, the out-of-domain branch -- E-M8 -- that this function used to CLAMP ``h_sq``, and the 20916 measured pixels it returned at |E| = 1

*Left in the source:* the rule and the hazard in present tense (clamping imprints a phase screen for a surface that is not there), with the measured pixel count kept as the size of the error, and the note that the ``aperture_diameter=None`` branch is unaffected.

```text
        # v5.30 (audit E-M8): a sphere of radius R simply does not exist
        # beyond ``h = |R|``, so return NaN there and let the aperture mask
        # zero those pixels -- exactly what the aspheric sibling
        # ``apply_aspheric_lens._aspheric_sag`` and the canonical
        # ``lenses.surface_sag_general`` / ``raytrace.conic_sag`` helpers do.
        # This function used to CLAMP ``h_sq`` to ``0.9999 R**2``, which
        # saturates the sag at a finite ``0.99 R`` and therefore imprinted a
        # unit-magnitude phase screen for a surface that is not there.  With
        # ``aperture_diameter`` larger than ``2|R|`` those pixels survived the
        # aperture: measured at R = 10 mm, aperture_diameter = 28 mm, N = 256,
        # dx = 120 um, 20916 out-of-domain pixels left the function with
        # |E| = 1.000000 and a bogus sag of 9.9 mm, while the aspheric sibling
        # on the SAME geometry returned NaN at exactly those 20916 pixels.
        # The ``aperture_diameter=None`` branch below already zeroes
        # ``h_sq >= 0.9999 * min(R1**2, R2**2)``, so this changes nothing
        # there (``where`` selects the zero, not the NaN).
```

### L862-867 -- `apply_aspheric_lens` docstring, "SA-nulling conics" -- that this docstring used to prescribe the ``-n**2`` conic on the curved first surface, and the retraction of that guidance

*Left in the source:* the corrected statement -- two different conics, and the ``-n**2`` value belongs to neither of these two cases -- ahead of the two measured bullets, which stay.

```text
    Two DIFFERENT conics are involved and this docstring used to conflate
    them: it prescribed the ``-n**2`` conic on the CURVED FIRST surface of a
    plano-convex lens as a third-order-SA null for collimated input.  That
    guidance is RETRACTED -- it names the wrong surface for a real lens AND
    the wrong value for this screen (adversarial audit 2026-07-25, finding
    E-C1).  Measured with an exact meridional Snell + eikonal trace
```

### L901-912 -- `apply_aspheric_lens`, the odd-power guard -- R-8 / E-L7 -- the measured pre-guard bit-identity of ``A1={5: 1e6}`` and ``A1={4: 1e6}``

*Left in the source:* the mechanism (``h_sq ** (power // 2)`` floors an odd power to the next even one with no diagnostic), the size of the error, and why the checker import is function-local.

```text
    # v5.31 (audit R-8 / E-L7 residual): reject ODD aspheric powers on BOTH
    # surfaces up front.  ``_aspheric_sag`` below evaluates
    # ``coeff * h_sq ** (power // 2)`` on its flat AND its curved branch, so an
    # odd power silently floors to the next-lower EVEN one and the screen
    # imprints a DIFFERENT surface with no diagnostic.  Measured pre-guard
    # (N = 64, dx = 5 um, lambda = 1.55 um): ``A1={5: 1e6}`` returned a field
    # BIT-identical to ``A1={4: 1e6}`` on both branches (and likewise for
    # ``A2``), the underlying sag being 100x the true ``h**5`` value at
    # h = 10 mm.  Same shared checker (and message) as
    # ``lenses.surface_sag_general`` / ``raytrace.conic_sag``; the import is
    # function-local because ``lumenairy.raytrace.__init__`` imports
    # ``raytrace.surface``, which imports ``elements.lenses``.
```

### L953-970 -- `_aspheric_sag`, the conic-domain gate -- the stacked 4.10 and v5.30 entries, and the 352 shell pixels that used to leave this function finite

*Left in the source:* the NaN convention, the ``norm < 0.9999`` gate and why it is not ``norm < 1.0``, the measured shell population, and the note that the two halves of the function now agree.

```text
        # 4.10: clamp invalid (outside-conic-domain) pixels to NaN so a
        # downstream aperture mask explicitly zeros them, rather than
        # silently extrapolating a 1e-12 floor that produced
        # near-singular sag (1e6 m for typical optics) outside the
        # surface domain.
        # v5.30 (audit E-L8): gate on ``norm < 0.9999`` like every sibling
        # (``lenses.surface_sag_general``, ``lenses._conic_sag_xp``,
        # ``elements.apply_mirror``'s inline sag, ``raytrace.conic_sag``)
        # rather than on ``denom_arg > 0`` (i.e. ``norm < 1.0``).  The two
        # differ on the thin shell ``0.9999 <= norm < 1.0``, where the conic
        # denominator ``1 + sqrt(denom_arg)`` is within 1e-2 of its vertical
        # tangent and the sag is numerically meaningless; the siblings NaN it.
        # Measured (R = 10 mm sphere, N = 2048, dx = 10 um): 352 pixels land
        # in that shell and used to leave this function finite (|E| = 1.000000)
        # while ``surface_sag_general`` returned NaN for all 352.  The
        # ``aperture_diameter=None`` branch below already uses the matching
        # ``h_sq < max_h_sq * 0.9999`` cut, so the two halves of this function
        # now agree on where the surface stops existing.
```

### L1068-1072 -- `apply_cylindrical_lens` docstring, ``Raises`` -- the "Pre-guard, ..." before/after listing

*Left in the source:* the same three failure modes as the reason the guard exists, in present tense, and the mirroring rule.

```text
        (v5.32, audit W5-2).  Pre-guard, ``f=nan`` returned a field whose
        every pixel was ``nan+nanj`` with nothing raised, ``f=+-inf``
        collapsed ``k/(2f)`` to zero and silently applied NO lens, and
        ``f=0`` raised a bare ``ZeroDivisionError('division by zero')``
        that named neither this function nor the argument.  Both guards
```

### L1089-1108 -- `apply_cylindrical_lens`, the ``f`` guard -- the v5.31 / v5.32 pair of entries and their measured pre-guard behaviour

*Left in the source:* that this is the sibling of the ``apply_thin_lens`` guard, the three failure modes it closes, and why the two must answer an unusable ``f`` identically.

```text
    # v5.31 (audit W4-2): the SIBLING of the ``apply_thin_lens`` non-finite-f
    # guard -- the only other ``f``-taking entry point in this module, with the
    # identical failure.  Measured pre-guard (N = 16, dx = 5 um): ``f=nan``
    # returned a field whose every pixel was ``nan+nanj`` with nothing raised;
    # ``f=+-inf`` collapsed ``k/(2f)`` to zero and silently applied NO lens.
    # Guarded here rather than left for a later sweep so the two cannot drift.
    #
    # v5.32 (audit W5-2): extended to ``f == 0`` alongside its
    # ``apply_thin_lens`` sibling, for the same reason and in the same
    # commit so the pair still cannot drift.  This function has NO
    # ``lens_model`` knob -- it is unconditionally the paraxial
    # ``k/(2f) x**2`` form -- so, measured pre-guard (N = 16, dx = 5 um,
    # both ``axis='x'`` and ``axis='y'``), ``f=0`` raised
    # ``ZeroDivisionError('division by zero')`` here on EVERY path: there
    # was no silent-no-op arm to unify away, only a bare exception naming
    # neither this function nor the argument.  Post-guard it raises the
    # same ``ValueError`` as ``apply_thin_lens``, so the two entry points
    # now answer f=0 identically -- which is the whole point, since
    # 'paraxial' ``apply_thin_lens`` and this function compute the same
    # quadratic and previously disagreed with each other's *other* models.
```

### L1128-1132 -- `apply_cylindrical_lens`, the CuPy dispatch -- that the three sibling functions had no ``use_gpu`` path before C-P1-6

*Left in the source:* what the dispatch does and why ``cp`` is resolved through the ``_lenses_module`` lazy slot rather than a bare global.

```text
    # v4.13.2 (audit C-P1-6): dispatch through CuPy when use_gpu=True
    # or E_in is already a CuPy array.  Resolve ``cp`` via the
    # _lenses_module lazy slot rather than a bare global (which is
    # not bound in this module's namespace).  Pre-fix the three
    # sibling functions had no use_gpu path at all.
```

### L1223 -- `apply_grin_lens` docstring, ``thin_form`` -- that ``thin_form=True`` exists to reproduce the PRE-FIX screen

*Left in the source:* that it is the short-rod approximation and is provided for back-compatibility, with the measured error ladder above it.

```text
        reproduce the pre-fix screen.
```

### L1280-1282 -- `apply_grin_lens`, the CuPy dispatch -- "(was previously numpy-only)"

*Left in the source:* the dispatch and the pointer to the sibling's resolution rationale.

```text
    # v4.13.2 (audit C-P1-6): CuPy dispatch (was previously numpy-only).
    # See apply_cylindrical_lens above for the _lenses_module.cp
    # resolution rationale.
```
