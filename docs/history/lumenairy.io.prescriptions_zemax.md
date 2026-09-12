<!-- lumenairy-history-doc
module: lumenairy/io/prescriptions_zemax.py
ast_sha256: f7024711a11c4cb8eaae610eaac39d9beaf2b226def1be3db87931696317e538
token_sha256: 1e20cdee5a3e898103b722ae369ab8661da51f270db63bd6083c8d5baa9fd20f
pre_relocation_lines: 2830
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/io/prescriptions_zemax.py`

This file holds the version-history narrative that used to live in
`lumenairy/io/prescriptions_zemax.py` -- the "vX.Y (audit Z): pre-fix this did
A, which was wrong because B, now it does C" blocks.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

This module is a format translator, and nearly every comment in it exists
because a specific Zemax record was once read or written wrongly.  That makes
the line between *history* and *why* unusually thin here, and the rule this
relocation applied is: **a failure mode that the current code still has to
avoid stays in the source, re-stated in the present tense; the release that
fixed it, and the description of the code that used to be there, move here.**
So "the pre-fix loader used `power = 2 + 2*parm_num`, which shifted every
coefficient up one even power" became "getting this wrong by one even power
shifts every coefficient up one power", and the measured consequence
(+2.8 mm defocus on poc1-19/20) stayed with it.

Two things were deliberately NOT touched:

* The audit identifiers used as *provenance* on a measurement -- `I2`, `I7`,
  `I8`, `ZX-3`, `ZX-nit`, `niche C1 item 3`, `P3-42`, `F-29`.  These name the
  study that produced a number the comment quotes, which is what
  `docs/TESTING_STANDARDS.md` S5 asks a numeric claim to carry.
* The self-quotation at the `niche C1 item 3` aperture-span block, which
  quotes the P3-42 comment sixty lines above it verbatim.  Rewording the
  quoted comment would have broken the quotation, so the P3-42 comment kept
  its wording and only lost its release tag.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L63-79 | `<module> DGRATING note` | the pre-v5.32 drop of PARM 1 / PARM 2, and the "the geometry import is UNCHANGED / what is new is" framing that described the release rather than the code |
| L81-81 | `<module> DGRATING note` | the date the two safety rules were learned |
| L123-128 | `_ZEMAX_AIR_POWERED_TYPES` | the "pre-fix" framing and the note that the v5.32 DGRATING fix was the same repair applied to one type |
| L580-582 | `load_zemax_zmx` | "Since v5.32 ... no longer discarded" |
| L597-601 | `load_zemax_zmx` | the "pre-fix a UTF-16-BE file decoded under latin-1" framing |
| L611-614 | `load_zemax_zmx` | the old "any supported encoding" message and what it misattributed |
| L675-675 | `load_zemax_zmx` | the release and audit tag |
| L747-752 | `load_zemax_zmx / PARM` | the pre-v4.15.1 non-zero-only store and the `parm_num >= 1` filter that dropped Q-type PARM 0 |
| L805-807 | `load_zemax_zmx` | "the pre-D4 window" / "the historical behaviour" phrasing |
| L813-817 | `load_zemax_zmx` | the release and niche tag |
| L832-832 | `load_zemax_zmx` | the release tag inside the auto-detect comment |
| L834-841 | `load_zemax_zmx` | the release tag and the past-tense framing of the window clip |
| L856-856 | `load_zemax_zmx` | the release and audit tag |
| L863-866 | `load_zemax_zmx` | the release tag on the DGRATING half of the +1 rule |
| L882-882 | `load_zemax_zmx` | the release number in the back-reference |
| L902-902 | `load_zemax_zmx` | the release and audit tag |
| L997-1001 | `load_zemax_zmx / Q-type` | that the loader once had no QBFS/QCON branch and what a Q-type prescription degraded to |
| L1010-1013 | `load_zemax_zmx / DGRATING` | the release and niche tag and the "pre-fix unknown-SURFTYPE branch" framing |
| L1061-1071 | `load_zemax_zmx / unknown SURFTYPE` | the release/audit tag and the "Pre-fix, every non-QBFS/QCON type fell into ..." framing |
| L1112-1118 | `load_zemax_zmx / EVENASPH` | the pre-fix `power = 2 + 2*parm_num` loader |
| L1131-1131 | `load_zemax_zmx` | the release and audit tag |
| L1139-1139 | `load_zemax_zmx` | the release and niche tag |
| L1277-1279 | `to_prescription` | the past-tense framing of the F-29 stop loss |
| L1283-1287 | `to_prescription` | the release/audit tag and the pre-v4.15.1 silent degrade |
| L1319-1319 | `load_zemax_zmx` | the release and audit tag |
| L1390-1392 | `load_zemax_zmx` | the "Pre-fix a zoom / thermal file imported as config 1" framing |
| L1749-1752 | `load_zemax_txt` | the "Pre-fix this side admitted glass and mirrors only" framing |
| L1763-1766 | `load_zemax_txt` | the release/audit tags on both halves of the +1 rule |
| L1785-1785 | `load_zemax_txt` | the release and audit tag |
| L1794-1794 | `load_zemax_txt` | the release and audit tag |
| L1965-1965 | `load_zemax_txt` | the release and audit tag |
| L2132-2132 | `export_zemax_lens_data` | the release and audit tag |
| L2185-2190 | `export_zemax_lens_data` | the release tag and "was previously mislabelled STANDARD" |
| L2255-2262 | `_warn_dropped_qtype` | the release/audit tag and the "Pre-fix both writers silently dropped these keys" paragraph |
| L2341-2343 | `_zmx_full_body` | "the new keys" / "the pre-3.7 lens-only path" |
| L2345-2349 | `_zmx_full_body` | the release/audit tag and "instead of the historical hardcoded 0" |
| L2422-2424 | `_zmx_full_body` | "Pre-v4.11.2 it drove a thickness sign-flip; that flip was removed" |
| L2461-2464 | `_zmx_full_body` | "Pre-v4.11.2 this compared the global surf_counter" |
| L2506-2511 | `_zmx_full_body` | the release tag and the "Pre-fix code converted ..." framing |
| L2527-2527 | `_zmx_full_body` | the release and audit tag |
| L2545-2553 | `_zmx_full_body` | the release/audit tag, the "Pre-fix only the refractive branch" framing and the v5.16.1 attribution of the power convention |
| L2666-2666 | `export_zemax_zmx` | the release and audit tag |
| L2766-2766 | `export_zemax_zmx` | the release and audit tag |

---

### L63-79 -- `<module> DGRATING note` -- the pre-v5.32 drop of PARM 1 / PARM 2, and the "the geometry import is UNCHANGED / what is new is" framing that described the release rather than the code

*Left in the source:* everything the note is actually for: what a DGRATING imports as, where the payload is attached, which consumer reads it, and the consequence of dropping it (re-stated in the present tense) with a pointer to this file.

```text
# Pre-v5.32 a ``DGRATING`` surface fell into the generic unknown-SURFTYPE
# branch: its GEOMETRY was imported correctly (flat / base conic) but PARM 1
# (lines per micrometre) and PARM 2 (the design diffraction order) were
# DROPPED with a warning.  The consequence recorded in the roadmap is that the
# prescription the wave chain sees "has never contained the DOE", so a
# consumer had to hand-build the grating, hand-split the chain at the DOE
# plane, and hand-fold the DOE's axial gaps into a neighbouring group -- the
# manual fold that produced a wrong answer once in the design-121 study.
#
# The geometry import is UNCHANGED (a DGRATING is still a flat/conic optical
# surface, and its PARM table is still NOT aspheric coefficients).  What is
# new is that the diffractive payload is ATTACHED: per surface on
# ``elements[i]['diffractive']``, and collected at the top level under
# ``'diffractives'`` together with the axial gaps to the neighbouring optical
# elements, which is what lets
# :func:`lumenairy.propagate_traced_carrier_chain` bookkeep those gaps itself
# (its ``groups`` list accepts ``{'doe': rx['diffractives'][k]}``).
```

### L81-81 -- `<module> DGRATING note` -- the date the two safety rules were learned

*Left in the source:* both rules in full -- they are the reason the window predicate and the gap convention look the way they do.

```text
# Two rules make that drop-in safe, both learned the hard way (2026-07-28):
```

### L123-128 -- `_ZEMAX_AIR_POWERED_TYPES` -- the "pre-fix" framing and the note that the v5.32 DGRATING fix was the same repair applied to one type

*Left in the source:* the failure mode, re-stated as what a glass/mirror-only predicate does, and the measured fixture that demonstrates it (PARAXIAL f=100 mm + STOP -> singlet alone, stop_index=None, zero warnings).

```text
# phase/hologram surface.  They never enter the glass span, so the pre-fix
# glass/mirror/DGRATING window auto-detect deleted them (and their STOP flag)
# BEFORE the unsupported-SURFTYPE branch could warn.  Measured: a ``PARAXIAL
# f=100 mm`` + STOP ahead of a glass singlet imported as the singlet alone,
# ``stop_index=None``, zero warnings.  The v5.32 DGRATING fix was exactly this
# repair applied to one type only.
```

### L580-582 -- `load_zemax_zmx` -- "Since v5.32 ... no longer discarded"

*Left in the source:* the contract: geometry on `'surfaces'`, payload on `'diffractives'`, and why the lens-only list stays free of diffractive keys.

```text
    paths keep seeing exactly the flat surface they did before.  Since v5.32
    the grating data is no longer discarded, though: it is attached to
    ``'diffractives'`` / ``elements[i]['diffractive']`` (with a
```

### L597-601 -- `load_zemax_zmx` -- the "pre-fix a UTF-16-BE file decoded under latin-1" framing

*Left in the source:* the whole encoding order and the reason for it, re-stated as what happens without BOM-sniffing first -- the misattributed "not a Zemax .zmx lens file" error.

```text
    # Read file.  I8: BOM-sniffing 'utf-16' FIRST so a big-endian export is
    # decoded by its BOM -- pre-fix a UTF-16-BE file decoded under latin-1
    # without a readable 'SURF' and was reported as "not a Zemax .zmx lens
    # file", pointing at the wrong cause.  Then UTF-16-LE (Zemax's own
    # BOM-less default), UTF-8, latin-1.
```

### L611-614 -- `load_zemax_zmx` -- the old "any supported encoding" message and what it misattributed

*Left in the source:* why reaching that branch means the file is not a Zemax lens file at all, which is what the live message says.

```text
        # ZX-nit (AUDIT_IO_ZEMAX): latin-1 always decodes, so reaching here
        # means the file WAS readable but carried no ``SURF`` record -- i.e.
        # it is not a Zemax .zmx lens file.  The old "any supported encoding"
        # message misattributed that to an encoding failure.
```

### L675-675 -- `load_zemax_zmx` -- the release and audit tag

*Left in the source:* the whole reason for the wrapper: a clear ValueError naming file, line and text instead of a bare IndexError from inside a token handler.

```text
        # v5.17.1 (audit P3-41): wrap the per-line keyword dispatch so a
```

### L747-752 -- `load_zemax_zmx / PARM` -- the pre-v4.15.1 non-zero-only store and the `parm_num >= 1` filter that dropped Q-type PARM 0

*Left in the source:* the Forbes/Zemax citation and the live rule -- PARM 0 is stored unconditionally and interpreted per surface -- plus what a `>= 1` filter would cost.

```text
                    # QBFS QCON docs).  Pre-v4.15.1 the loader only
                    # stored non-zero values and the parm_num >= 1
                    # filter further dropped any PARM 0 sourced from a
                    # Q-type freeform; v4.15.1 stores PARM 0
                    # unconditionally and decides per-surface how to
                    # consume it (Q-type r_max vs EVENASPH ignore).
```

### L805-807 -- `load_zemax_zmx` -- "the pre-D4 window" / "the historical behaviour" phrasing

*Left in the source:* what the span is and what `None` means, which is the contract the aperture fallback below reads.

```text
    # to read (the GLASS/MIRROR span, i.e. the pre-D4 window).  ``None`` means
    # "the whole imported window", which is the historical behaviour and what
    # every non-diffractive file gets.  See the aperture block below.
```

### L813-817 -- `load_zemax_zmx` -- the release and niche tag

*Left in the source:* the rule and the reason for the warning, including the quoted roadmap phrase that names the state it prevents.

```text
        # v5.32 (niche D4): an EXPLICIT surface_range is the caller's own
        # window, so it is honoured as given -- but a DGRATING it excludes is
        # dropped from 'diffractives' too, which is exactly the "the design
        # the chain sees has never contained the DOE" state roadmap P2 exists
        # to end.  Say so instead of dropping it silently.
```

### L832-832 -- `load_zemax_zmx` -- the release tag inside the auto-detect comment

*Left in the source:* the predicate's definition.

```text
        # ray -- glass, a mirror, or (v5.32, niche D4) a diffractive.
```

### L834-841 -- `load_zemax_zmx` -- the release tag and the past-tense framing of the window clip

*Left in the source:* the failure mode in full, re-stated in the present tense, including both DOE layouts it hit and why design 121 never showed it.

```text
        # v5.32 (niche D4): a DGRATING is an air-to-air flat, so the pre-fix
        # glass/mirror-only ``active`` list clipped the window to the glass
        # span and DISCARDED any DGRATING outside it -- with no warning, and
        # before ``_collect_diffractives`` ever ran, so ``'diffractives'``
        # came back EMPTY for two perfectly ordinary DOE layouts (a fan-out
        # behind a collimator; a fan-out at the output, behind the last
        # glass).  Design 121 never saw it because both its DGRATINGs sit
        # between glass surfaces.
```

### L856-856 -- `load_zemax_zmx` -- the release and audit tag

*Left in the source:* the whole +1 rule and the terminal-mirror consequence.  The wording is quoted verbatim by the niche C1 block sixty lines below, so it was kept exactly.

```text
        # v5.17.1 (audit P3-42): only extend the range by +1 when the
```

### L863-866 -- `load_zemax_zmx` -- the release tag on the DGRATING half of the +1 rule

*Left in the source:* the rule itself -- an air-to-air DGRATING has no exit surface, so the test is "does the last active surface carry glass" -- and the argument that this is the same test as the glass-only one for every non-diffractive file.

```text
        # v5.32: an air-to-air DGRATING has no exit surface either, so the
        # test is now "does the last active surface carry glass" -- which is
        # the SAME test as before for every non-diffractive file (a mirror is
        # either GLAS MIRROR, caught by is_mirror, or MIRR 1 with no glass).
```

### L882-882 -- `load_zemax_zmx` -- the release number in the back-reference

*Left in the source:* the back-reference itself (`the P3-42 failure three lines above`) and the quotation it carries, plus both measured pollution figures (12.000 -> 13.000 mm, and 12.000 -> 100.000 mm).

```text
        # DOE and the glass -- which is verbatim the v5.17.1 (P3-42) failure
```

### L902-902 -- `load_zemax_zmx` -- the release and audit tag

*Left in the source:* why a single terminal mirror is legitimate and why >= 2 still applies to refractive selections.

```text
    # v5.17.1 (audit P3-42): a single terminal mirror is a legitimate
```

### L997-1001 -- `load_zemax_zmx / Q-type` -- that the loader once had no QBFS/QCON branch and what a Q-type prescription degraded to

*Left in the source:* the whole PARM-to-coefficient mapping above it, and the pointer to the canonical consumer.

```text
        # Pre-v4.15.1 the loader had no QBFS/QCON branch, so any
        # Q-type prescription silently degraded to base conic plus
        # an EVENASPH-mis-interpreted PARM table.  See
        # ``lumenairy.elements.freeform.surface_sag_q_bfs`` for the
        # canonical coefficient consumer.
```

### L1010-1013 -- `load_zemax_zmx / DGRATING` -- the release and niche tag and the "pre-fix unknown-SURFTYPE branch" framing

*Left in the source:* the rule: geometry exactly as the generic branch imports it, payload attached rather than dropped.

```text
            # v5.32 (niche D4): keep the flat/conic geometry EXACTLY as the
            # pre-fix unknown-SURFTYPE branch imported it (PARM is still not
            # aspheric), and attach the diffractive payload instead of
            # dropping it.  See the module note above _dgrating_surface_data.
```

### L1061-1071 -- `load_zemax_zmx / unknown SURFTYPE` -- the release/audit tag and the "Pre-fix, every non-QBFS/QCON type fell into ..." framing

*Left in the source:* the entire argument, re-stated as what happens WITHOUT the branch, including the per-type PARM-meaning table and the measured TOROIDAL example (PARM 1 100.0 -> a_2 = 1e5 1/m -> 0.625 m of sag at r = 2.5 mm).

```text
            # v5.17.1 (audit P2-19): unknown SURFTYPE.  Pre-fix, every
            # non-QBFS/QCON type fell into the EVENASPH branch below,
            # which interpreted its PARM table as even-asphere
            # coefficients.  For sibling Zemax types the PARM slots
            # mean something entirely different (TOROIDAL PARM 1 =
            # radius of rotation in mm; ODDASPHE PARM n = coefficient
            # of r^n; DGRATING PARM 1 = grating lines/um; PARAXIAL
            # PARM 1 = focal length; ...), so the prescription silently
            # acquired enormous fake aspheric sag (a TOROIDAL
            # ``PARM 1 100.0`` became a_2 = 1e5 1/m -> 0.625 m of sag
            # at r = 2.5 mm).  Import unknown types as the plain base
```

### L1112-1118 -- `load_zemax_zmx / EVENASPH` -- the pre-fix `power = 2 + 2*parm_num` loader

*Left in the source:* the same arithmetic as a live hazard (getting the power wrong by one shifts and inflates every coefficient), with both measured consequences kept: the 1e6 inflation and the +2.8 mm defocus on poc1-19/20.

```text
            # The pre-fix loader used power = 2 + 2*parm_num, which shifted
            # every coefficient UP one even power (r^4 -> r^6, ...) AND --
            # via the unit_scale**(power-1) rescale -- inflated each value
            # by unit_scale**2 = 1e6.  On a real Zemax import this turned a
            # ~few-um asphere into a ~tens-of-um monster on the wrong order,
            # destroying the traced-lens wavefront (observed: +2.8 mm image
            # defocus + smeared spots on the poc1-19/20 designs).
```

### L1131-1131 -- `load_zemax_zmx` -- the release and audit tag

*Left in the source:* what `q_extra` is and where it is spread.

```text
        # v4.15.1 (P1-NEW-E): pack the Forbes Q-type freeform keys
```

### L1139-1139 -- `load_zemax_zmx` -- the release and niche tag

*Left in the source:* the rule that the payload never rides on the lens-only `surfaces` entries, and why.

```text
        # v5.32 (niche D4): the diffractive payload rides on the ELEMENT only
```

### L1277-1279 -- `to_prescription` -- the past-tense framing of the F-29 stop loss

*Left in the source:* the whole consequence, re-stated as what would happen without `is_stop` / `semi_diameter` on the lens-only surfaces.

```text
            # ``prescription['surfaces'][i].get('is_stop')`` -- fell through to
            # STOP=surface-0 on every LOADED file (relocating the declared
            # stop on re-export), and the tracer lost the explicit stop.
```

### L1283-1287 -- `to_prescription` -- the release/audit tag and the pre-v4.15.1 silent degrade

*Left in the source:* what is forwarded, to which consumers, and what their absence costs.

```text
        # v4.15.1 (P1-NEW-E): forward Forbes Q-type freeform keys so
        # the lens-only prescription consumed by apply_real_lens_traced
        # / surface_sag_freeform sees the coefficients and r_max
        # (pre-v4.15.1 the Q-bfs / Q-con SURFTYPE was silently dropped
        # to base conic).
```

### L1319-1319 -- `load_zemax_zmx` -- the release and audit tag

*Left in the source:* why the category and stacklevel are explicit -- the warning must point at the caller.

```text
        # v4.16.1 (audit ORG-2 / C.7): explicit UserWarning category +
```

### L1390-1392 -- `load_zemax_zmx` -- the "Pre-fix a zoom / thermal file imported as config 1" framing

*Left in the source:* the same fact as the live reason for the diagnostic.

```text
    # say out loud that only the base (LDE) state was imported.  Pre-fix a
    # zoom / thermal file imported as config 1 with no signal that the other
    # N-1 positions existed.
```

### L1749-1752 -- `load_zemax_txt` -- the "Pre-fix this side admitted glass and mirrors only" framing

*Left in the source:* the shared predicate and the failure it prevents, re-stated in the present tense, with the I2 reference intact.

```text
        # uses.  Pre-fix this side admitted glass and mirrors only, so a
        # ``PARAXIAL`` / ``ABCD`` / phase row outside the glass span -- and
        # its STOP flag -- was deleted here before any diagnostic could fire,
        # exactly the I2 failure, in the twin the fix had not reached.
```

### L1763-1766 -- `load_zemax_txt` -- the release/audit tags on both halves of the +1 rule

*Left in the source:* the rule, the terminal-mirror case, the air-to-air powered case and the cross-reference to the .zmx twin.

```text
        # v5.17.1 (audit P3-42): only extend the range by +1 when the
        # last active surface is refractive glass (the +1 captures its
        # exit surface).  A terminal MIRROR has no exit surface -- see
        # the matching fix in load_zemax_zmx.  v5.46: an air-to-air powered
```

### L1785-1785 -- `load_zemax_txt` -- the release and audit tag

*Left in the source:* the rule and its cross-reference.

```text
    # v5.17.1 (audit P3-42): allow a single terminal mirror (see
```

### L1794-1794 -- `load_zemax_txt` -- the release and audit tag

*Left in the source:* the whole reason for the per-surface warning -- the SUMMARY table carries no coefficients, so a non-STANDARD surface would degrade silently.

```text
    # v5.17.1 (audit P3-43): the SURFACE DATA SUMMARY table carries no
```

### L1965-1965 -- `load_zemax_txt` -- the release and audit tag

*Left in the source:* why the category and stacklevel are explicit.

```text
        # v4.16.1 (audit ORG-2 / C.7): explicit UserWarning category +
```

### L2132-2132 -- `export_zemax_lens_data` -- the release and audit tag

*Left in the source:* the resolution order for the default stop.

```text
    # v5.4.6 (audit F-29): default stop_surface to the prescription's own
```

### L2185-2190 -- `export_zemax_lens_data` -- the release tag and "was previously mislabelled STANDARD"

*Left in the source:* the rule and the live limitation it works around: the paste table cannot carry aspheric coefficients, so those surfaces get a footnote.

```text
        # v5.18.1: reflect the surface's ACTUAL type in the TYPE column instead
        # of hardcoding STANDARD -- an aspheric or freeform surface was
        # previously mislabelled STANDARD in the paste table (the export-side
        # sibling of the P3-43 .txt-loader drop).  The paste table's columns
        # cannot carry the aspheric a4/a6/... coefficients themselves, so record
        # such surfaces for a footnote pointing at the lossless .zmx export.
```

### L2255-2262 -- `_warn_dropped_qtype` -- the release/audit tag and the "Pre-fix both writers silently dropped these keys" paragraph

*Left in the source:* what the function warns about and the live consequence of NOT warning -- a Zemax cross-check run against the wrong surface.

```text
    """v5.17.1 (audit P2-20): warn LOUDLY when a Forbes Q-type freeform
    surface (``freeform_type`` = ``'q_bfs'`` / ``'q_con'`` with its
    ``q_bfs_coeffs`` / ``q_con_coeffs`` + ``r_max`` keys) is exported by
    a ``.zmx`` writer that has no QBFS/QCON emission path.

    Pre-fix both writers silently dropped these keys, so an exported
    Q-type surface degraded to its base conic with no diagnostic and a
    Zemax cross-check compared against the WRONG surface.
```

### L2341-2343 -- `_zmx_full_body` -- "the new keys" / "the pre-3.7 lens-only path"

*Left in the source:* which writer is used when, and that the lens-only path is the fallback.

```text
    Used by :func:`export_zemax_zmx` when the prescription dict
    carries the new keys; the pre-3.7 lens-only path remains the
    fallback.
```

### L2345-2349 -- `_zmx_full_body` -- the release/audit tag and "instead of the historical hardcoded 0"

*Left in the source:* the resolution rule and why it is duplicated here (self-consistency for a direct call).

```text
    # v5.4.7 (audit AUDIT_V5_4_6 #6): resolve the aperture-stop index from
    # the prescription when not given, instead of the historical hardcoded
    # 0 (first refractive surface).  The public ``export_zemax_zmx`` already
    # passes a resolved value (F-29), so this only changes a hypothetical
    # direct call -- but it makes the internal writer self-consistent.
```

### L2422-2424 -- `_zmx_full_body` -- "Pre-v4.11.2 it drove a thickness sign-flip; that flip was removed"

*Left in the source:* the live invariant: mirror_count is diagnostics-only and must NOT drive a sign flip, because the canonical thicknesses are already Zemax-signed.

```text
    # mirror_count is retained for diagnostics only.  Pre-v4.11.2 it
    # drove a thickness sign-flip; that flip was removed because the
    # canonical thicknesses are already Zemax-signed.
```

### L2461-2464 -- `_zmx_full_body` -- "Pre-v4.11.2 this compared the global surf_counter"

*Left in the source:* the documented meaning of `stop_surface` and why a global counter would put STOP on the wrong row in a folded design.

```text
    # of the aperture stop **among refracting surfaces**".  Pre-v4.11.2
    # this compared the global ``surf_counter`` (which includes
    # coord-breaks and mirrors) so folded designs placed STOP on the
    # wrong row.
```

### L2506-2511 -- `_zmx_full_body` -- the release tag and the "Pre-fix code converted ..." framing

*Left in the source:* the whole argument as a live prohibition: the loader stores raw Zemax-signed DISZ and the GUI keeps it canonical, so a parity flip destroys mirror DISZ on round-trip.

```text
        # v4.11.2: no mirror-parity flip.  Pre-fix code converted
        # "physical-positive" back to Zemax-signed by negating every
        # thickness after each mirror; but the loader stores raw
        # Zemax-signed DISZ (no conversion) and the GUI (v3.7.4+)
        # keeps Zemax-signed canonical, so the flip was always
        # spurious here and destroyed mirror DISZ on round-trip.
```

### L2527-2527 -- `_zmx_full_body` -- the release and audit tag

*Left in the source:* the rule and the cross-verification workflow it protects.

```text
        # v5.17.1 (audit P2-20): Forbes Q-type freeform keys have no
```

### L2545-2553 -- `_zmx_full_body` -- the release/audit tag, the "Pre-fix only the refractive branch" framing and the v5.16.1 attribution of the power convention

*Left in the source:* the rule, the round-trip identity requirement, and the full PARM mapping (parm_idx = power // 2, unit conversion).

```text
            # v5.17.1 (audit P2-20): emit even-aspheric coefficients on
            # mirrors too.  Pre-fix only the refractive branch below had
            # the EVENASPH switch + PARM emission, so an aspherized
            # mirror (e.g. an aspherized OAP) silently degraded to its
            # base conic on export and load->export->load was not
            # identity.  Same PARM mapping as refractives:
            # parm_idx = power // 2 (v5.16.1 power = 2*parm_num
            # convention), coefficient converted 1/m^(power-1) ->
            # 1/mm^(power-1).
```

### L2666-2666 -- `export_zemax_zmx` -- the release and audit tag

*Left in the source:* the resolution order and the round-trip property it preserves.

```text
    # v5.4.6 (audit F-29): default stop_surface to the prescription's own
```

### L2766-2766 -- `export_zemax_zmx` -- the release and audit tag

*Left in the source:* the rule.

```text
        # v5.17.1 (audit P2-20): warn loudly instead of silently
```

