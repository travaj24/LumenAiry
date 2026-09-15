<!-- lumenairy-history-doc
module: lumenairy/elements/polarization.py
ast_sha256: 1a97668f3a64829e0933ed70903aa573791daba90007637fdc6a914612112fb6
token_sha256: dcd372620dcbffb9916ae5644b6d2331ec3e4508831ee8c736fa56ec0e5e4adf
pre_relocation_lines: 1838
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B8 (audit Z3 / WP-A11 sec. 6.3): apply_jones_matrix's 2x2 mix moves into _jones_mix_2x2 and accumulates through one shared scratch buffer (4.00 -> 3.00 full-grid complex arrays, the floor), bit-identical and gated on the dtypes already agreeing so a mixed-precision JonesField keeps the original expressions
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/elements/polarization.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/polarization.py`: the `vN.NN (audit X): pre-fix this
returned A` passages in front of the v5.29 input guards, the 4.10 / 4.11.1
handedness flip-and-restore, the post-v5.17.0 retarder-sign BEHAVIOR CHANGE
essay, and the docstrings that retract an earlier revision of themselves
(`degree_of_polarization`, `propagate_fresnel`, `propagate_fraunhofer`, the
batched-FFT threshold comment).  Each block is reproduced **verbatim** under the
source line it came from in the pre-relocation file.

What did NOT move: the circular-polarization sign convention block in the
module docstring (`tests/unit/test_niche_audit_e_polarization_inputs.py` reads
`IEEE`, `right-hand-rule` and `Born & Wolf` out of it), the Mueller-block sign
warning, the retarder-sign CONSEQUENCES a caller acts on, and the measured
numbers that size the live guards.

One correction, not a move: the module docstring said "CONVENTIONS.md section 7
still calls this row `Born-Wolf` and should be relabelled".  `CONVENTIONS.md`
line 159 already reads "IEEE / right-hand-rule", so the sentence was the
opposite of the truth and now says the two agree.

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
| L45-48 | `<module>` docstring, the circular-polarization convention | the E-M13 relabelling narrative, and a claim about CONVENTIONS.md that is no longer true |
| L142-148 | `JonesField.__init__` | E-L15 -- the AttributeError the pre-coercion shape guard raised |
| L245-247 | `JonesField._BATCH_MIN_N` | E-L12 -- that the pre-fix comment pointed at a setter that does not exist |
| L325-329 | `JonesField.propagate_fresnel` docstring | that this docstring used to promise a return it never had |
| L345-348 | `JonesField.propagate_fraunhofer` docstring | the same retracted "Returns new grid spacings" wording |
| L782-788 | `apply_waveplate` docstring, ``Raises`` | W3-T4 -- what an unguarded NaN retardance returned |
| L807-832 | `apply_waveplate` docstring, "BEHAVIOR CHANGE" | the v4.7-v5.17.0 EE-convention sign, the Berreman comparison that exposed it, and the superseded v5.4.6 "DECOUPLED" note |
| L1010-1015 | `apply_polarizing_beam_splitter` docstring, ``Raises`` | E-H9 -- the measured port swap the pre-fix guard allowed |
| L1026-1031 | `apply_polarizing_beam_splitter`, the ER guard | E-H9 -- what the pre-fix guard rejected |
| L1094-1098 | `create_linear_polarized` docstring, ``Raises`` | W3-T4 -- what an unguarded NaN angle returned |
| L1114-1124 | `_HANDEDNESS` | E-H8 -- the ``startswith('r')`` parse this table replaces |
| L1166-1169 | `create_circular_polarized` docstring, ``handedness`` | "pre-fix it gave 'right'" -- the P2-15 before/after |
| L1180-1184 | `create_circular_polarized` docstring, ``Raises`` | E-H8 -- the pre-fix parse, stated as history |
| L1190-1203 | `create_circular_polarized` docstring, ``Notes`` | the 4.10 handedness flip and its 4.11.1 restoration, plus a second copy of the P2-15 retarder-sign note |
| L1254-1257 | `create_elliptical_polarized` docstring, ``Raises`` | W3-T4 -- what an unguarded NaN orientation returned |
| L1263-1268 | `create_elliptical_polarized` docstring, the chi domain | the "Measured pre-fix" framing on the round-trip evidence |
| L1380-1389 | `degree_of_polarization` docstring | E-L17 -- a retraction of this docstring's own earlier "depolarization" text |
| L1404-1411 | `degree_of_polarization`, the background cut | E-L13 -- the absolute ``S0 > 1e-30`` cut this replaced |
| L1526-1537 | `jones_pupil_to_stokes_unpolarized` docstring, ``J`` | E-H10 -- that there was no layout guard at all before |
| L1584-1588 | `stokes_to_dop` docstring | "is now RELATIVE ... instead of the absolute" -- the before/after framing |
| L1642-1652 | `_order_power_scale`, the grazing-order branch | E-L14 -- the silent 1.0 divisor substitution and the measured 7.1e12 error it produced |
| L1670-1675 | `_plane_wave_carrier` docstring | Z4 -- that this site used the integer ``N // 2`` |

---

### L45-48 -- `<module>` docstring, the circular-polarization convention -- the E-M13 relabelling narrative, and a claim about CONVENTIONS.md that is no longer true

*Left in the source:* the measurement and the self-consistency statement; the stale 'should be relabelled' sentence is CORRECTED, not just moved -- CONVENTIONS.md section 7 already carries the IEEE label.

```text
(audit E-M13, 2026-07-25).  Nothing in the code changed for E-M13 -- the
whole element/solver family is self-consistent to 1e-16; only the label
was wrong.  CONVENTIONS.md section 7 still calls this row "Born-Wolf"
and should be relabelled "IEEE / right-hand-rule".
```

### L142-148 -- `JonesField.__init__` -- E-L15 -- the AttributeError the pre-coercion shape guard raised

*Left in the source:* the rule (coerce first) and the two reasons for it, in present tense, plus the no-op note for the ndarray path.

```text
        # v5.29 (audit E-L15): coerce FIRST.  Pre-fix the shape guard read
        # ``Ex.shape`` on the raw argument, so a nested list (the natural
        # hand-written input) died with ``AttributeError: 'list' object has
        # no attribute 'shape'`` instead of the documented ValueError -- and
        # a 1-D list never reached the 2-D check at all.  ``np.asarray`` is
        # a no-op for arrays, so the ndarray path is unchanged (``self.Ex``
        # is still the caller's object when it is already complex).
```

### L245-247 -- `JonesField._BATCH_MIN_N` -- E-L12 -- that the pre-fix comment pointed at a setter that does not exist

*Left in the source:* how to move the threshold, which is the only thing a reader needs.

```text
    # attribute directly.  v5.29 (audit E-L12): the pre-fix comment
    # pointed at a setter function that does not exist anywhere in the
    # library.
```

### L325-329 -- `JonesField.propagate_fresnel` docstring -- that this docstring used to promise a return it never had

*Left in the source:* what the method returns and where the new pitch is written, which is the contract.

```text
            ``self``.  v5.46 (audit Z4): this docstring used to read
            "Returns new grid spacings", which it never did -- the new
            spacings are written onto ``self.dx`` / ``self.dy`` (Fresnel is
            pitch-CHANGING: ``dx_out = lambda*z/(N*dx)``) and the method
            returns ``self``.  Read the new pitch off the returned object.
```

### L345-348 -- `JonesField.propagate_fraunhofer` docstring -- the same retracted "Returns new grid spacings" wording

*Left in the source:* the contract, pointing at the sibling method.

```text
            ``self``.  v5.46 (audit Z4): as for :meth:`propagate_fresnel`,
            the new spacings are written onto ``self.dx`` / ``self.dy``
            rather than returned; the old "Returns new grid spacings"
            wording described a return this method never had.
```

### L782-788 -- `apply_waveplate` docstring, ``Raises`` -- W3-T4 -- what an unguarded NaN retardance returned

*Left in the source:* the mechanism (``exp(+1j*nan)`` is NaN) and why the failure is invisible downstream, which is why it is rejected here.

```text
        Also if ``retardance`` (or either angle) is not finite.  v5.29
        (audit W3-T4, sibling of the E-L16 ellipticity guard): pre-fix
        ``retardance=np.nan`` returned a field whose every pixel was
        ``nan+nanj`` -- ``exp(+1j*nan)`` is NaN -- with nothing raised,
        so the NaN only surfaced far downstream (or not at all, since
        :func:`degree_of_polarization` reads NaN as NaN and intensity
        plots show blank).
```

### L807-832 -- `apply_waveplate` docstring, "BEHAVIOR CHANGE" -- the v4.7-v5.17.0 EE-convention sign, the Berreman comparison that exposed it, and the superseded v5.4.6 "DECOUPLED" note

*Left in the source:* a ``versionchanged`` directive naming the change and the sign that ships, plus the CONSEQUENCES a caller has to act on -- which QWP orientation gives which S3, and that HWPs and all intensity results are unaffected.

```text
    BEHAVIOR CHANGE (audit P2-15, post-v5.17.0): from v4.7 through
    v5.17.0 this function used ``exp(-i*retardance)`` on the slow axis
    -- the ``exp(+i omega t)`` (EE-convention) sign -- with a docstring
    that incorrectly attributed it to ``exp(-i omega t)``.  That made
    the Jones-element family the CONJUGATE of the library's own
    rigorous solver Jones: ``berreman_jones_1d`` on a uniaxial
    quarter-wave slab (``eps = diag(no^2, ne^2, no^2)``,
    ``d = lambda/(4 (ne - no))``, index-matched half-spaces) returns
    transmission Jones ``diag(e^{i k0 no d}, e^{i k0 ne d})`` --
    slow-relative-fast phase ``+pi/2`` -- and the same slab with its
    fast axis at +45 deg maps x-pol to ``Ey/Ex = -i`` (S3 = -1),
    while the pre-fix ``apply_waveplate`` gave ``Ey/Ex = +i``
    (S3 = +1): circular handedness flipped between the element and
    solver families for the same physical device.  The element sign
    now matches the solver family (``berreman_jones_1d`` /
    ``BerremanStack`` / ``rcwa_jones_1d``), so solver-derived Jones
    matrices drop into JonesField pipelines without conjugation.

    Consequences: a QWP with fast axis at +45 deg on x-pol now yields
    S3 = -1 (``create_circular_polarized``'s 'left'); use fast axis
    at -45 deg for S3 = +1 ('right').  Half-wave plates are unaffected
    (``exp(+-i pi) = -1`` either way), as are all
    retardance-magnitude / intensity results.  The v5.4.6 (P3-22)
    "DECOUPLED, mutually consistent" note predates the Berreman /
    RCWA retarder Jones (v5.14.4) and is superseded by this
    cross-family alignment; see CONVENTIONS.md section 7.
```

### L1010-1015 -- `apply_polarizing_beam_splitter` docstring, ``Raises`` -- E-H9 -- the measured port swap the pre-fix guard allowed

*Left in the source:* what is rejected and the consequence of not rejecting it, with the measured swap kept as the size of the error.

```text
        ``extinction_ratio`` is NaN or ``< 1``.  v5.29 (audit E-H9): the
        pre-fix guard rejected only ``extinction_ratio <= 0``, so a value
        in ``(0, 1)`` was accepted and silently SWAPPED the two output
        ports (``ER=0.1`` on an x-polarized input put 0.909 of the power
        in the "reflected" port and 0.091 in the "transmitted" one, with
        power still conserved so nothing flagged it).
```

### L1026-1031 -- `apply_polarizing_beam_splitter`, the ER guard -- E-H9 -- what the pre-fix guard rejected

*Left in the source:* the definition of ER, the hazard in present tense, and the house style the refusal follows.

```text
        # v5.29 (audit E-H9): ER is defined as the wanted:unwanted POWER
        # ratio, hence >= 1.  The pre-fix guard rejected only ER <= 0, so
        # ER in (0, 1) sailed through and inverted the two ports (leak >
        # 0.5 makes the "wanted" amplitude ``a`` the SMALLER one) -- power
        # conserved, no warning.  Reject instead of quietly supporting it,
        # matching the sibling guard's raise-on-bad-input style.
```

### L1094-1098 -- `create_linear_polarized` docstring, ``Raises`` -- W3-T4 -- what an unguarded NaN angle returned

*Left in the source:* the guard, its sibling, and the failure it prevents.

```text
        If ``angle`` is not finite.  v5.29 (audit W3-T4): sibling of the
        :func:`create_elliptical_polarized` ``orientation`` guard -- this
        is the same major-axis angle, and pre-fix ``angle=np.nan``
        returned a field whose every pixel was ``nan+nanj`` with nothing
        raised.
```

### L1114-1124 -- `_HANDEDNESS` -- E-H8 -- the ``startswith('r')`` parse this table replaces

*Left in the source:* what the table IS, why an explicit table rather than a prefix test, and why the rotation-sense names are excluded.

```text
# v5.29 (audit E-H8): the accepted ``handedness`` spellings for
# :func:`create_circular_polarized`, mapped to the sign of Ey's imaginary
# unit (== the sign of the resulting S3).  Pre-fix the parse was
# ``handedness.lower().startswith('r')`` with NO else-branch, so every
# unrecognised string ('cw', 'ccw', 'clockwise', 'linear', '', and any
# typo that does not begin with 'r') silently produced LEFT circular,
# while the typo 'rihgt' silently produced RIGHT.  Deliberately excluded:
# 'cw' / 'ccw' / 'clockwise' / 'counterclockwise'.  Rotation-sense names
# are ambiguous without also stating the viewing direction (from the
# source vs into the beam), which is exactly the axis on which the
# IEEE and Born & Wolf conventions disagree -- see the module docstring.
```

### L1166-1169 -- `create_circular_polarized` docstring, ``handedness`` -- "pre-fix it gave 'right'" -- the P2-15 before/after

*Left in the source:* the live cross-reference: which QWP orientation gives which handedness under the shipped retarder sign.

```text
        branch.  (Audit P2-15, post-v5.17.0: ``apply_waveplate`` was
        realigned to the Berreman/RCWA solver Jones, so a QWP with
        fast axis at **+45 deg** on x-pol now gives 'left' (S3 = -1);
        pre-fix it gave 'right'.)
```

### L1180-1184 -- `create_circular_polarized` docstring, ``Raises`` -- E-H8 -- the pre-fix parse, stated as history

*Left in the source:* the same failure as the reason the strict table exists, in present tense.

```text
        v5.29 (audit E-H8): pre-fix the parse was
        ``handedness.lower().startswith('r')`` with no else-branch, so
        ``'cw'``, ``'ccw'``, ``'clockwise'``, ``'linear'``, ``''`` and
        every typo not beginning with 'r' silently returned LEFT
        circular (and the typo ``'rihgt'`` silently returned RIGHT).
```

### L1190-1203 -- `create_circular_polarized` docstring, ``Notes`` -- the 4.10 handedness flip and its 4.11.1 restoration, plus a second copy of the P2-15 retarder-sign note

*Left in the source:* the two live facts: ``'right'`` obeys ``S3 > 0`` and agrees with ``vector_diffraction.py``, and the QWP recipe that reproduces it is fast axis at -45 deg.

```text
    4.11.1: the 4.10 "fix" to this function flipped the handedness
    branches so that 'right' produced ``(1, -i)/sqrt(2)``, which gave
    ``S3 = -1`` under the library's own Stokes formula and contradicted
    the hard-coded right-circular Jones vector in
    ``vector_diffraction.py``.  4.11.1 restores the pre-4.10 form
    where 'right' obeys ``S3 > 0``.

    Audit P2-15 (post-v5.17.0): ``apply_waveplate``'s retarder sign
    was flipped to match the Berreman/RCWA solver Jones (slow axis
    ``exp(+i*phi)``), so the QWP recipe that reproduces
    ``create_circular_polarized('right')`` on x-pol is now fast axis
    at **-45 deg** (pre-fix: +45 deg).  This function's own Jones
    vectors and its agreement with ``vector_diffraction.py`` are
    unchanged.
```

### L1254-1257 -- `create_elliptical_polarized` docstring, ``Raises`` -- W3-T4 -- what an unguarded NaN orientation returned

*Left in the source:* the guard and the sibling gap it closes.

```text
        If ``orientation`` is not finite.  v5.29 (audit W3-T4): pre-fix
        ``orientation=np.nan`` (or ``+-inf``) returned a field whose
        every pixel was ``nan+nanj`` with nothing raised -- the sibling
        gap left by the E-L16 ``ellipticity`` guard below.
```

### L1263-1268 -- `create_elliptical_polarized` docstring, the chi domain -- the "Measured pre-fix" framing on the round-trip evidence

*Left in the source:* the whole measurement, as what happens when the domain restriction is lifted -- it is the reason the restriction exists.

```text
        state silently comes back with a DIFFERENT (chi, psi).  Measured
        pre-fix: ``chi = 0.9`` round-tripped through
        :func:`polarization_ellipse` as ``chi = 0.6708,
        psi = pi/2`` (axes swapped, orientation rotated 90 deg), and
        ``chi = pi/2`` -- a perfectly reasonable-looking "circular"
        request -- came back LINEAR (chi = 0) at psi = pi/2.  Reduce
```

### L1380-1389 -- `degree_of_polarization` docstring -- E-L17 -- a retraction of this docstring's own earlier "depolarization" text

*Left in the source:* the corrected statement: a DOP below 1 is an output this container cannot produce, with the measurement and the pointer to :func:`stokes_to_dop`.

```text
    is dark).  v5.29 (audit E-L17): the pre-fix docstring's "values less
    than 1 indicate depolarization ... partially coherent / incoherent
    sources or through depolarizing elements" describes an output this
    container CANNOT produce -- there is no API to inject a partially
    polarized ``S = (1, 0, 0, 0)`` state, and no element in this module
    is depolarizing (measured: DOP = 1 to 4e-16 over every element and
    500 random pure states).  Partial polarization has to come in as a
    Stokes/Mueller quantity; use :func:`stokes_to_dop` for that.  The
    useful content here is therefore the MASK of illuminated pixels plus
    a numerical self-check of the Stokes algebra.
```

### L1404-1411 -- `degree_of_polarization`, the background cut -- E-L13 -- the absolute ``S0 > 1e-30`` cut this replaced

*Left in the source:* why the cut must be relative (DOP is scale-invariant) and the noise argument that sizes it.

```text
    # v5.29 (audit E-L13): the background cut used to be the ABSOLUTE
    # ``S0 > 1e-30``, which has nothing to do with the field's own scale:
    # a perfectly polarized 1e-15 V/m field (S0 = 1e-30) reported DOP =
    # 0.0, as did EVERY pixel of any field weaker than that, and a NaN
    # field reported 0.0 rather than NaN.  DOP is scale-invariant, so the
    # only defensible "this pixel is dark" test is relative to the
    # brightest pixel: amplitude rounding noise is ~eps of the peak
    # amplitude, i.e. ~eps^2 of the peak INTENSITY.
```

### L1526-1537 -- `jones_pupil_to_stokes_unpolarized` docstring, ``J`` -- E-H10 -- that there was no layout guard at all before

*Left in the source:* which layouts are accepted and what an unguarded transposed pupil silently returns, which is why the guard is there.

```text
        exactly this layout).  v5.29 (audit E-H10): BOTH layouts are
        accepted, matching :func:`apply_jones_matrix` -- the trailing-2x2
        ``(..., 2, 2)`` pupil layout above and the ``(2, 2, Ny, Nx)``
        layout that ``apply_jones_matrix`` calls canonical (the latter is
        moved to trailing axes internally via ``np.moveaxis``).  Anything
        else raises ``ValueError``.  Pre-fix there was no guard at all:
        a ``(2, 2, Ny, Nx)`` pupil was silently indexed as if its first
        two axes were spatial, returning ``(2, 2)``-shaped Stokes maps
        whose values were wrong by O(0.5) -- and shapes with no 2x2 block
        anywhere (e.g. ``(3, 3)``, ``(Ny, Nx, 3, 3)``) were accepted too.
        A ``(2, 2, 2, 2)`` input is ambiguous and is read as the
        documented trailing-2x2 pupil.
```

### L1584-1588 -- `stokes_to_dop` docstring -- "is now RELATIVE ... instead of the absolute" -- the before/after framing

*Left in the source:* the rule and the failure the absolute cut produces, in present tense.

```text
    Background pixels are set to 0.  v5.29 (audit E-L13, sibling of
    :func:`degree_of_polarization`): "background" is now RELATIVE to the
    brightest pixel (``S0 <= eps^2 max(S0)``) instead of the absolute
    ``S0 <= 1e-30``, which reported every pixel of a uniformly weak but
    perfectly polarized Stokes map as DOP 0.  ``NaN`` propagates.
```

### L1642-1652 -- `_order_power_scale`, the grazing-order branch -- E-L14 -- the silent 1.0 divisor substitution and the measured 7.1e12 error it produced

*Left in the source:* why a grazing order has no finite efficiency limit, the measured size of the substitution error, and which caller can actually reach the branch.

```text
    # v5.29 (audit E-L14): a GRAZING order (|kz| -> 0) is as unusable as an
    # evanescent one -- ``az = -(kx ax + ky ay)/kz`` diverges and the true
    # efficiency ``flux (tang + |az|^2) ~ |kz| / |kz|^2`` has no finite
    # limit.  Pre-fix the divisor was silently SUBSTITUTED with 1.0 there,
    # which kept az at its (tiny) tangential scale: measured at
    # |kz| = 1e-13 the returned amplitude scale was 4.47e-7 against the
    # honest continuation's 3.16e+6, a factor 7.1e12 too small.  Return 0
    # like the evanescent branch instead.  (``jones_field_from_orders``
    # already filters ``Re(kz) > 1e-12`` before calling; the reachable
    # caller is ``RCWAResult._order_power_scale``, which passes the raw
    # port kz.)
```

### L1670-1675 -- `_plane_wave_carrier` docstring -- Z4 -- that this site used the integer ``N // 2``

*Left in the source:* the centring rule and the half-pixel offset the integer form produces on ODD grids, which is why the rule is spelled out.

```text
    factory and every ``elements/elements.py`` grid.  v5.46 (audit Z4): this
    site used the integer ``N // 2``, which agrees for even ``N`` but puts the
    origin half a pixel off for ODD ``N``, so a :class:`JonesField` built by
    :func:`jones_field_from_orders` on an odd grid was offset by ``dx/2``
    relative to every element applied to it afterwards (apertures,
    spatially-varying Jones callables).
```
