<!-- lumenairy-history-doc
module: lumenairy/propagators/carrier_field.py
ast_sha256: 6fc5905777971218087ce405f041d80b4a14b34daf4a839c923e9de20ccc5138
token_sha256: f08b25bbf206e6dd5f3cbfb83808c1f470b26b7ab9683f0a154bc582a3d3c167
pre_relocation_lines: 1857
recorded_by: WP-A17 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/carrier_field.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/carrier_field.py` -- the "pre-fix this did A, which was
wrong because B, now it does C" passages.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

`carrier_field.py` is a young module and carries much less history than its
siblings: the audit's token census scored it 20.2 % only because the loose
"mentions an audit or a version" heuristic also catches its DERIVATIONS -- the
measured cliffs and enclosed-power calibrations that `docs/TESTING_STANDARDS.md`
S5 requires a numeric bar to carry.  Those stayed in the source.  What moved
here is the pre-fix evidence: what the guard accepted before the band term
existed, and what `_enclosed_power_radius` returned before a non-finite sample
raised.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L60-65 | `<module> docstring` | what leaving the envelope band out of the bound cost (P0-1 / P0-2) |
| L197-210 | `_BAND_HEADROOM` | the 24-fixture bisection behind the 2.5 multiplier |
| L312-317 | `CarrierSpec.piston` | which fix put an absolute optical path on the traced exit field |
| L902-909 | `_enclosed_power_radius` | P1-3 -- what a non-finite sample used to return |
| L1106-1112 | `carrier_difference_nyquist` | the 38 %-wrong round trip the band-less guard accepted (P0-1 / P0-2) |

---

### L60-65 -- `<module> docstring` -- what leaving the envelope band out of the bound cost (P0-1 / P0-2)

*Left in the source:* the aliasing argument itself, and the fact that an energy ledger cannot see it.

```text
   not ``ramp + band`` aliases the envelope's skirt into the answer.  Left
   out, that is a 38 %-wrong round trip accepted at the guard's own default
   (VERIFY_ARCHITECTURE P0-1/P0-2), and it is invisible to an energy ledger
   because aliasing conserves power.  :func:`carrier_difference_nyquist`
   takes the band MEASURED off the envelope (:meth:`CarrierField.band_slope`)
   and adds ``_BAND_HEADROOM`` times it to both bounds.
```

### L197-210 -- `_BAND_HEADROOM` -- the 24-fixture bisection behind the 2.5 multiplier

*Left in the source:* the calibration in condensed form -- the two coordinates, both spreads and both ends of the bracket.  TESTING_STANDARDS S5 requires a numeric bar to carry its derivation, so this one stays in the source; only the pointer to the fail-before / fix-after tables moves.

```text
#: HOW IT WAS DERIVED (docs/audits/FIX_VERIFY_ARCH_2026_08_12.md S1).  The
#: round-trip cliff was bisected over 24 fixtures -- beam widths 25..200 um,
#: ramps 0.02..0.10 rad, finite-R and collimated carriers.  Expressed as
#: ``(lambda/2dx - ramp) / band``, the cliff sits at 1.666..2.092, a 1.26x
#: spread over the whole matrix.  Expressed instead as a ``nyquist_margin``
#: on the ramp-only bound -- the coordinate the guard used to cut in -- the
#: SAME cliff runs 1.087..4.478, a 4.12x spread, which is the proof that no
#: choice of margin default can express this boundary and that the missing
#: BAND TERM is what the guard was short by.
#:
#: 2.5 is the measured worst case (2.092) plus 1.20x headroom.  The upper
#: end is set by over-refusal: 3.0 starts refusing pitches measured clean at
#: 3.4e-10 relative.  Verified in both directions -- see the fail-before /
#: fix-after tables in that document.
```

### L312-317 -- `CarrierSpec.piston` -- which fix put an absolute optical path on the traced exit field

*Left in the source:* what the attribute IS and why it is carried, which is the contract.

```text
        Constant optical path (m), EXPLICIT.  This is the term
        ``FIX_TILT_QUADRATIC_OPL_2026_08_11`` restores to
        ``apply_real_lens_traced``'s exit field, and carrying it here is what
        lets a field re-referenced onto another carrier keep a meaningful
        ABSOLUTE optical path.  Contributes ``exp(i k0 piston)`` -- a global
        unit phasor, intensity-blind by construction.
```

### L902-909 -- `_enclosed_power_radius` -- P1-3 -- what a non-finite sample used to return

*Left in the source:* why 0.0 is the WORST failure here rather than a safe one, and the ``nan > 0.0 is False`` trap that produces it -- both are live hazards for anyone editing this reduction.

```text
    A NON-FINITE sample RAISES.  It used to fall through the ``tot > 0.0``
    test -- ``nan > 0.0`` is ``False`` -- and return 0.0, which is not a
    conservative failure but the LEAST conservative one available: a support
    radius of zero collapses every maximum-over-the-disc in this module to
    the chief ray alone, where a concentric sphere-difference ramp is
    identically zero, so the Nyquist guard SILENTLY ACCEPTED calls it
    correctly refuses on the same field when clean (VERIFY_ARCHITECTURE
    P1-3).  A guard whose own input is NaN has to say so."""
```

### L1106-1112 -- `carrier_difference_nyquist` -- the 38 %-wrong round trip the band-less guard accepted (P0-1 / P0-2)

*Left in the source:* the ``env_band`` default and who always passes it, which is the contract a caller needs.

```text
    Leaving ``band`` out is what let the guard accept a **38 %-wrong**
    round trip at its own default (VERIFY_ARCHITECTURE P0-1/P0-2): with the
    pitch and the ramp both frozen at an ACCEPTED margin of 1.023, shrinking
    the beam 200 -> 10 um drove the round trip to rel L2 1.0055 while the
    reported margin never moved.  ``env_band`` defaults to 0.0 so the
    arithmetic of a caller who has no field in hand is unchanged, but
    :func:`re_reference` and :func:`aggregate` always measure and pass it.
```

