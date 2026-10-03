<!-- lumenairy-history-doc
module: lumenairy/propagators/carrier_field.py
ast_sha256: 30e58f95ccbbb6ed84f604b41668db2bfd00db3e1590a099530cbbfd65249e94
token_sha256: 3037cd63a48c957338801a7d2d839445553d23b8a6253f1ea93bee67885b84ce
pre_relocation_lines: 1857
recorded_by: WP-A17 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-14 -- Wave 5 item D (handoff 4.4): the warning chain is swept onto lumenairy.elements._lens_kernels.caller_stacklevel().  Every warnings.warn literal stacklevel and every threaded literal below it is retargeted to the computed level, which walks out to the first frame outside the package, so the attribution no longer depends on how deep the warn site sits.  MEASURED before the sweep (validation/probe_known_reds/probe_carrier_attribution.py): the tilt-inert notice named the caller when propagate_carrier_referenced was called directly and named library source when the identical site was reached through carrier_referenced_focus_readout -- 2 of 4 emissions misattributed, 0 of 4 after.  No physics changed.
re_recorded: 2026-09-20 -- WP-C4 round 2 (VERIFY-WP-C4 D2): mft_method= added and threaded to the MFT call, so the shape rule's default flip keeps a one-keyword way back at every public entry point; None stamps nothing and no answer moves
re_recorded: 2026-10-03 -- 5.50.0 removal: CarrierField is @dataclass(frozen=True) (assignment deprecated in 5.46, horizon 5.48 slipped once to 5.50); the __setattr__ warning shim, the _built gate, _CARRIER_FIELD_FROZEN_SINCE / _CARRIER_FIELD_FROZEN_IN and the two imports only the shim used are deleted; __hash__ = None keeps the class unhashable as before.  re_reference / aggregate / replace / pickle outputs byte-identical before/after on both builds
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

## Removed in 5.50.0 -- the `CarrierField` assignment-warning shim (deprecated 5.46)

*Not a relocation block: a record of code that was DELETED.*  5.46.0 (audit
finding C5) announced that `CarrierField` would become a frozen dataclass like
its `CarrierSpec` and `FieldGrid` members: a `__setattr__` override warned on
every assignment to a BUILT field (armed by a private `_built` flag set last in
`__post_init__`) and still let it through, with the horizon held in
`_CARRIER_FIELD_FROZEN_SINCE = '5.46'` / `_CARRIER_FIELD_FROZEN_IN = '5.48'`
and resolved through `_deprecation.resolve_removal_version`.  5.48.0 slipped the
horizon once to 5.50; 5.50.0 executed it: the decorator is
`@dataclass(frozen=True)`, the `__setattr__` override, the `_built` flag, both
constants and the two imports only the shim used (`warnings`,
`caller_stacklevel`) are deleted, and `__hash__ = None` is set explicitly so the
class stays unhashable as it was while mutable (`frozen=True` with `eq=True`
would otherwise generate a field-tuple hash for a value whose envelope array is
still writable in place).  `__post_init__` already wrote through
`object.__setattr__`, so construction, `dataclasses.replace`, `with_provenance`,
`copy.deepcopy`, `pickle` and the zarr round trip needed no change; MEASURED
2026-10-03, re_reference / aggregate / full_field / replace / pickle outputs of
a fixed fixture are byte-identical before and after on both the Windows py3.14
and the WSL py3.12 build.

*Left in the source:* the class docstring states the class is FROZEN, why, and
the routes to a changed field; a one-line comment points here.

The shim as it stood in 5.49.0 (the warning text abridged at "..."):

```text
    def __setattr__(self, name, value):
        """Announce a post-construction field assignment (deprecated).

        The assignment still happens -- this is the announcement half of the
        cycle, not the removal.  See the class docstring for the migration
        and :data:`_CARRIER_FIELD_FROZEN_IN` for the horizon."""
        if getattr(self, '_built', False):
            from .._deprecation import resolve_removal_version
            warnings.warn(
                f"CarrierField.{name}: assigning to a built CarrierField is "
                f"deprecated since v{_CARRIER_FIELD_FROZEN_SINCE} and will "
                f"raise in v"
                f"{resolve_removal_version(_CARRIER_FIELD_FROZEN_IN)} (the "
                f"class becomes frozen, like CarrierSpec and FieldGrid).  ..."
                DeprecationWarning, stacklevel=_caller_stacklevel())
        object.__setattr__(self, name, value)
```
