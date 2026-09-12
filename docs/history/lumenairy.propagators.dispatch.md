<!-- lumenairy-history-doc
module: lumenairy/propagators/dispatch.py
ast_sha256: 49e2718d8cae3a8ab540e9870fe9e5f0cc76e2d43988b4b0a4e50a3c96b3add2
token_sha256: 82667cdd472e30f98dcf6e5a2907c87073edee8feb9f523d87e84f3d826b1da6
pre_relocation_lines: 1698
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/dispatch.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/dispatch.py` -- the "vN.N (audit X): pre-fix this did A,
which was wrong because B, now it does C" passages.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

Two audits dominate it.  **P5 / roadmap F1** flipped the default return of
`propagate` from the chosen kernel's native shape to a `PropagationResult`,
and the module recorded the whole decision: four costed options, the choice to
execute rather than announce, and a tombstone listing every symbol retired with
the transition `DeprecationWarning`.  **W9** (items 1-10) fixed ten dispatcher
defects in one pass, and every fix wrote its measured pre-fix evidence into the
comment above the guard it added.  Neither narrative describes what the code
does; both are here.

What stayed is the part a future editor cannot work without: the FALSY
`_NO_DEFAULT` sentinel warning (`if not return_result` would silently un-flip
the contract), the reason the sentinel rather than `True` is the parameter
default, and at each guard the rule it enforces together with the measurement
that names the members it points callers at (`gbd` / `hf` / `hfpi` do honour an
output-grid request -- measured shape 64->96, dx 40->80 um -- which is why the
diagnostic can name them).

`_auto_select_method`'s "CANONICAL free-space regime logic" paragraph also
stayed in full, including its measured 0.41-fidelity citation: it is the reason
`_select_asm_variant` delegates rather than carrying its own thresholds, and
deleting it would invite the duplicate-trip-point defect straight back.  The
4-row fidelity table that settled the bands, which lived in
`_select_asm_variant`'s `.. versionchanged:: 5.31` block as fail-before /
fix-after evidence, is reproduced below.

Nothing the interpreter executes changed in the move -- including the string
literals inside `propagate`'s `ValueError` messages, which still narrate
"pre-v5.31 this path silently produced `PropagationResult(field=None)`":
those are executable text, not comments, and the token fingerprint pins them.
The header above records the SHA-256 of (a) the module's AST with every
docstring removed and source positions ignored, and (b) its `tokenize` stream
reduced to NAME/OP/NUMBER/STRING with comments and docstrings dropped -- both
taken from the file as it stood BEFORE the relocation.
`tests/unit/test_audit2609_a17_history_relocation.py` re-computes both from the
live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L41-52 | `_GRID_CHANGING_METHODS` | what a pre-v5.30 `auto` caller silently got, and which audit recorded it |
| L55-107 | `<module>` -- the return-contract flip | the F1 four-option costing, the announce-then-execute decision, and the tombstone list of everything retired with the transition warning |
| L126-131 | `propagate` -- `return_result` parameter | the `v5.30 (audit P5 / roadmap F1, EXECUTED)` framing |
| L143-171 | `propagate` -- return-contract warning | what the pre-v5.30 default was, that `return_result=False` is bit-for-bit the pre-flip default, the option-4 reasoning for not re-arity-ing `PropagationResult`, and the retirement of the transition DeprecationWarning |
| L195-199 | `propagate` -- `sas` row of the auto table | what selecting `sas` for an output-grid caller used to raise |
| L228-236 | `propagate` -- the bit-identity note | that the tabled shapes are bit-for-bit the pre-v5.30 defaults, and the pointer to the four-option costing |
| L250-252 | `propagate` -- `method` parameter | the `bit-for-bit the pre-v5.31 behaviour` tag |
| L292-309 | `propagate` -- `output_grid` / `output_dx` | what each of the three W9 fixes changed, and what the pre-fix behaviour was in each case |
| L314-317 | `propagate` -- `output_grid` shortcut note | what the pre-4.12 ASM family did with a dropped output-grid request |
| L330-342 | `propagate` -- `return_result` parameter | that `False` is bit-for-bit the pre-flip default, and the 4.12 B1-7 tuple-unpacking defect |
| L358-364 | `propagate` -- the unpacking warning | that the 2-item iteration was re-decided at the flip rather than left as a collision |
| L376-378 | `propagate` -- Warns | that pre-flip the warning also fired on the default path |
| L411-419 | `propagate` -- resolving `wrap` | the `EXECUTED` flip framing and the bit-for-bit-as-before claim |
| L431-442 | `propagate` -- the legacy-contract warning | the `since the v5.30 flip` / `post-flip` framing |
| L450-462 | `propagate` -- `_requested_output_dx` | the measured pre-fix mislabelling (`result.dx == 4e-05` on a 96^2 / 80 um resample) |
| L468-472 | `propagate` -- the coerce branch | the 4.12 B1-7 pre-fix tuple path |
| L482-492 | `propagate` -- the null-field refusal | the measured `PropagationResult(field=None)` silence |
| L507-511 | `propagate` -- `out_dy` | that the y-pitch used to be discarded for anamorphic calls |
| L569-576 | `_coerce_field` | the v4.13.0 anamorphic info-loss closure and the 4.12 B1-7 `field=None` defect, both as pre-fix narrative |
| L701-717 | `_NO_OUTPUT_GRID_METHODS` | the measured pre-fix silent-drop evidence for `maslov` |
| L728-741 | `_REQUIRED_METHOD_KWARGS` | that two of the twelve VALID_METHODS were unusable through this entry point before the table existed |
| L835-850 | `_auto_select_method` -- `output_requested` | the measured pre-fix ValueError at z=1e-3 and the 'bit-for-bit unchanged' tail |
| L853-887 | `_auto_select_method` -- the removed DOE branch | the tombstone of the deleted `events_json` branch: that it could not fire, that forcing it raised, and the grating-deflection measurement that found no basis for routing to hfpi |
| L1133-1141 | `_dispatch_to_method` -- the DOE-kwarg refusal | the measured pre-fix `apply_real_lens_maslov() got an unexpected keyword argument` death and the dead `events_json` check |
| L1159-1164 | `_dispatch_to_method` -- the output-grid refusal | the pre-fix silent drop and the mislabelled wrapper |
| L1260-1272 | `_dispatch_to_method` -- the hfpi free-space branch | what the old precondition check named, and which release fixed the through-prescription path first |
| L1430-1486 | `_select_asm_variant` -- `.. versionchanged:: 5.31` | the three W9 fixes written as pre-fix/post-fix narrative, with the measured back-propagation raises, the bit-identical dropped-tilt measurement and the 4-row fidelity table that settled the regime bands |
| L1515-1522 | `_select_asm_variant` -- the delegation | this selector's former `Q > 20` / `Q > 2` thresholds |
| L1590-1593 | `which_propagator` -- the reason strings | that the reasons no longer quote this module's former ASM-family-only multiples |

---

### L41-52 -- `_GRID_CHANGING_METHODS` -- what a pre-v5.30 `auto` caller silently got, and which audit recorded it

*Left in the source:* what the set IS and what it gates today

```text
# v5.30 (audit P5): the bare-grid kernels whose native return is the
# ``(E, dx_out, dy_out)`` triple at a kernel-chosen output pitch rather
# than a bare ndarray at the input pitch.  ``method='auto'`` can pick
# any of these, so a pre-v5.30 ``auto`` caller who read the return as an
# ndarray silently got a tuple -- and at a different sampling.  That is the
# instability recorded in
# ``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`` P5 and closed by
# the flip described in the block below: the DEFAULT return is now the
# shape-stable ``PropagationResult`` for every method.  This set stays as-is
# because the native shapes stay reachable -- it gates the shape-instability
# ``UserWarning``, which post-flip can only fire on the explicit
# ``return_result=False`` (legacy-contract) path.
```

### L55-107 -- `<module>` -- the return-contract flip -- the F1 four-option costing, the announce-then-execute decision, and the tombstone list of everything retired with the transition warning

*Left in the source:* the three-value contract itself, the reason the sentinel (not ``True``) is the parameter default, and the FALSY-sentinel warning to future editors -- the one note that stops this contract being silently un-flipped

```text
# ---------------------------------------------------------------------------
# audit P5 / roadmap Part F1 -- the return-contract flip, EXECUTED (v5.30)
# ---------------------------------------------------------------------------
# The F1 decision (four costed options in
# ``docs/roadmap_deferred_2026_07_21.md``) landed as **option 4**: make the
# shape-stable :class:`~lumenairy.propagators.PropagationResult` the DEFAULT
# return while keeping the kernels' native shapes reachable behind an explicit
# ``return_result=False``.  v5.30 first shipped only the announcement (a
# registry-scheduled ``DeprecationWarning`` plus a falsy sentinel default);
# the owner then chose to EXECUTE the flip in the same release rather than
# ship a warning about a change nobody could yet see -- the same call the W5
# shim-removal wave made (see ``lumenairy/_deprecation.py``'s tombstones and
# the CHANGELOG's ``### Changed (BREAKING)`` section).
#
# As shipped:
#
#   * ``return_result`` UNSET -- a ``PropagationResult`` for **every** method.
#     ``.field`` / ``.dx`` / ``.dy`` are defined whichever kernel ran, so the
#     return shape no longer depends on ``z``.  That is the P5 finding closed.
#   * ``return_result=False`` -- the kernels' native shapes (bare ndarray OR
#     ``(E, dx_out, dy_out)``), bit-for-bit as before the flip.  A PERMANENT,
#     documented escape hatch: the migration path for ``E, dxo, dyo``
#     unpackers (``PropagationResult`` iteration stays 2-item, audit P16) and
#     for wrapper-free fast loops.  It is not deprecated and nothing is
#     scheduled against it.
#   * ``return_result=True`` -- unchanged.
#
# Retired WITH the flip (tombstone, v5.30): the transition
# ``DeprecationWarning`` (``_p5_transition_message``), its external-caller
# predicate (``_caller_is_internal``), ``_P5_DEPRECATED_SINCE`` /
# ``_P5_UNSTABLE_RETURN_TYPES``, and the ``API_TRANSITION_VERSION`` /
# ``resolve_removal_version`` imports that resolved its horizon.  A warning
# whose text is "the default WILL become a PropagationResult in vX" cannot
# outlive the version that makes it a PropagationResult; leaving it would
# advertise a future change that has already happened -- the exact
# registry-rot class the horizon mechanism exists to prevent.  Its purpose is
# served by the decision record that replaces it: this block, ``propagate``'s
# docstring, the roadmap's F1 EXECUTED entry, and the CHANGELOG.  Nothing
# remains to warn a *caller* about: the default is now the stable contract and
# the alternative is an explicit, supported argument.
#
# The ``_NO_DEFAULT`` sentinel STAYS as the parameter default (rather than
# becoming a literal ``True``) because the two are different statements:
# ``True`` says "this caller asked for the wrapper", the sentinel says "this
# caller did not choose, so the library's stable contract applies".  Keeping it
# means the distinction the transition measured stays available to any future
# contract decision, and ``inspect.signature(propagate)`` stays honest about
# which values are a *choice*.
#
# WARNING for future edits: the sentinel is FALSY.  ``if not return_result``
# would route it to the legacy contract -- i.e. silently un-flip this change.
# :func:`propagate` therefore resolves it ONCE, up front, into a local ``wrap``
# flag and routes on that; do the same in any new branch.
```

### L126-131 -- `propagate` -- `return_result` parameter -- the `v5.30 (audit P5 / roadmap F1, EXECUTED)` framing

*Left in the source:* the whole rationale -- it is the note that stops the falsy sentinel being routed on directly.  Only the version framing and the block name moved.

```text
    # v5.30 (audit P5 / roadmap F1, EXECUTED): the default is the "not passed"
    # sentinel, NOT a literal ``True``, so "the library's stable contract
    # applies" stays distinguishable from "this caller asked for the wrapper".
    # It resolves to the STABLE contract (a PropagationResult).  ``_NO_DEFAULT``
    # is FALSY, so it must never be routed on directly -- it is resolved once
    # into ``wrap`` below.  See the flip block at the top of this module.
```

### L143-171 -- `propagate` -- return-contract warning -- what the pre-v5.30 default was, that `return_result=False` is bit-for-bit the pre-flip default, the option-4 reasoning for not re-arity-ing `PropagationResult`, and the retirement of the transition DeprecationWarning

*Left in the source:* the contract as it stands, that ``False`` is a permanent escape hatch, and the 2-item iteration hazard

```text
       **Return contract, settled in v5.30 (audit P5, roadmap Part F1
       option 4 -- EXECUTED).**  The DEFAULT return is a
       :class:`~lumenairy.propagators.PropagationResult` for **every**
       method: ``.field`` / ``.dx`` / ``.dy`` are defined whichever kernel
       ran, and ``np.asarray(result)`` yields the field, so the return
       shape no longer depends on ``z``.  Pre-v5.30 the default was the
       chosen kernel's native shape -- a bare ``ndarray`` *or* an
       ``(E, dx_out, dy_out)`` triple, see the table below -- which is the
       instability P5 raised.

       * ``return_result=False`` -- the native shapes, bit-for-bit as
         before the flip.  A **permanent, supported escape hatch**, not a
         deprecated one: it is the migration for code that unpacks
         ``E, dxo, dyo`` and for fast loops that want no wrapper
         allocation.
       * ``return_result=True`` -- unchanged, and now the same contract the
         default hands back.

       ``PropagationResult`` iteration did **not** move to 3 items at the
       flip (audit P16): it stays ``(field, intermediates)``, which is what
       ``E, inter = propagate_through_system(..., return_result=True)``
       needs.  Re-arity-ing it would have traded one breakage for another,
       and option 4 does not require it -- ``return_result=False`` is the
       migration path for 3-tuple unpackers.

       The transition :class:`DeprecationWarning` that announced this flip
       while it was still scheduled is retired with the flip itself: the
       default it pointed callers away from no longer exists.

```

### L195-199 -- `propagate` -- `sas` row of the auto table -- what selecting `sas` for an output-grid caller used to raise

*Left in the source:* the live routing rule and the exact kernel it promotes to

```text
      v5.31 (audit W9-1): with ``output_grid`` / ``output_dx`` given, that
      band selects ``asm`` instead (SAS has no output-grid path, so
      selecting it raised a ``ValueError`` naming a kernel the caller
      never wrote); ``asm`` auto-promotes to the exact
      :func:`angular_spectrum_propagate_mft`.
```

### L228-236 -- `propagate` -- the bit-identity note -- that the tabled shapes are bit-for-bit the pre-v5.30 defaults, and the pointer to the four-option costing

*Left in the source:* the live statement: ``return_result`` is the only knob and nothing is scheduled against either value

```text
    .. note::
       The shapes in the table above are bit-for-bit what pre-v5.30
       releases returned by default; v5.30 changed *which of them you get
       without asking*, not what any of them contain.  The deferred
       four-option costing that chose this route (option 4) is recorded in
       ``docs/roadmap_deferred_2026_07_21.md`` Part F1 (audit P5), with the
       bit-identity evidence for both explicit modes.  ``return_result``
       is the only knob: there is no version-scheduled behaviour left here
       and no deprecation attached to either value.
```

### L250-252 -- `propagate` -- `method` parameter -- the `bit-for-bit the pre-v5.31 behaviour` tag

*Left in the source:* the rule and its definition of 'any other situation'

```text
        * unset in any other situation -- ``'auto'``, bit-for-bit the
          pre-v5.31 behaviour.  "Any other situation" is: the knob still holds
          its shipped value, **or** a ``prescription`` was supplied.  The knob
```

### L292-309 -- `propagate` -- `output_grid` / `output_dx` -- what each of the three W9 fixes changed, and what the pre-fix behaviour was in each case

*Left in the source:* the three live rules

```text
        - SAS / RS do not support arbitrary output-grid sampling and
          raise ``ValueError`` (pointing at the ASM-MFT entry point)
          if ``output_grid`` / ``output_dx`` is passed.  Since v5.31
          ``method='auto'`` no longer *selects* SAS when an output grid is
          requested (audit W9-1), so that raise is reachable only by
          naming ``method='sas'`` yourself.
        - Maslov / asymptotic / MHS do not thread the request to their
          kernels and raise ``ValueError`` naming the members that do
          (v5.31, audit W9-4).  Pre-v5.31 the request was silently
          dropped -- and with the ``output_dx`` shortcut the returned
          ``PropagationResult.dx`` reported the requested pitch while the
          field was still at the input pitch.
        - The pitch reported on the result honours **either** form: since
          v5.31 ``output_grid=(N_out, dx_out)`` sets
          ``PropagationResult.dx`` to ``dx_out`` (audit W9-5); pre-fix
          only the ``output_dx`` shortcut did, so an ``output_grid`` call
          came back labelled with the input pitch even though the field
          had genuinely been resampled.
```

### L314-317 -- `propagate` -- `output_grid` shortcut note -- what the pre-4.12 ASM family did with a dropped output-grid request

*Left in the source:* what the two spellings mean

```text
        the input ``N``).  Pre-4.12 the ASM family silently dropped
        these kwargs and returned a bare-grid output at the input
        pitch -- a quiet wrong-physics path that audit round-4 B1-8
        flagged.
```

### L330-342 -- `propagate` -- `return_result` parameter -- that `False` is bit-for-bit the pre-flip default, and the 4.12 B1-7 tuple-unpacking defect

*Left in the source:* that ``False`` is permanent and un-deprecated, why the sentinel default is kept, and which dx the wrapper reports for tuple-returning kernels

```text
        **``False`` is permanent and un-deprecated**, and is exactly how
        pre-v5.30 callers keep their shapes: ``propagate(...,
        return_result=False)`` is bit-for-bit the pre-flip default.  The
        sentinel default is kept (rather than a literal ``True``) so the
        library can still tell "did not choose -- give me the stable
        contract" apart from "this caller asked for the wrapper"; both
        resolve to the same return today.

        4.12: for tuple-returning kernels (Fresnel / Fraunhofer / SAS
        return ``(E, dx_out, dy_out)``) the wrapped result now reports
        the kernel's **output** dx, not the input dx.  Pre-4.12 audit
        round-4 B1-7: tuple unpacking silently failed, ``field`` was
        ``None``, and ``dx`` was the input pitch.
```

### L358-364 -- `propagate` -- the unpacking warning -- that the 2-item iteration was re-decided at the flip rather than left as a collision

*Left in the source:* the instruction (read the attributes) and the migration

```text
           Read the attributes (``.field``, ``.dx_out``, ``.dy_out``)
           instead of unpacking.  The 2-item iteration is pinned
           behaviour and did not change **at the F1 flip either**, which
           was decided explicitly in the same pass rather than left as a
           collision (audit P16).  ``return_result=False`` is the supported
           migration for ``E, dxo, dyo`` unpackers -- and since v5.30 they
           must pass it, because the bare 3-tuple is no longer the default.
```

### L376-378 -- `propagate` -- Warns -- that pre-flip the warning also fired on the default path

*Left in the source:* when the warning fires and when it cannot

```text
        the shape-stable wrapper, so the diagnostic has nothing to report
        (v5.30: pre-flip this also fired on the default path, which was
        then the legacy contract).
```

### L411-419 -- `propagate` -- resolving `wrap` -- the `EXECUTED` flip framing and the bit-for-bit-as-before claim

*Left in the source:* the resolution rule and the FALSY-sentinel warning, which is the load-bearing half

```text
    # v5.30 (audit P5 / roadmap F1 option 4, EXECUTED): resolve the return
    # contract ONCE, here, and route on ``wrap`` alone below.  ``_NO_DEFAULT``
    # ("caller did not choose") resolves to the STABLE contract -- that is the
    # flip.  Every other value keeps its truthiness, so an explicit
    # ``return_result=False`` (and any other falsy value a pre-flip caller
    # passed) still selects the kernels' native shapes, bit-for-bit as before.
    # Do NOT test ``return_result`` directly in new code: the sentinel is
    # falsy, so ``if not return_result`` would silently restore the pre-flip
    # default and un-do this change.
```

### L431-442 -- `propagate` -- the legacy-contract warning -- the `since the v5.30 flip` / `post-flip` framing

*Left in the source:* the whole reason the warning exists and why it is now the only one

```text
        # The legacy contract, reached only by asking for it
        # (``return_result=False``) since the v5.30 flip.  v5.30 (audit P5):
        # the auto-selector can hand back a bare ndarray at the input pitch OR
        # an ``(E, dx_out, dy_out)`` triple at a kernel-chosen pitch, decided
        # purely by ``z`` -- and the caller has no way to know which without
        # re-running the selector.  This warning says so, out loud, exactly
        # when it bites: ``auto`` chose a grid-changing kernel and the caller
        # opted out of the shape-stable wrapper.  The reported pitch is read
        # off the kernel's own return, so no formula is duplicated here.
        # Post-flip this is the ONLY place either shape-warning survives: the
        # default and ``return_result=True`` both return the wrapper, whose
        # shape does not depend on z, so there is nothing to warn about.
```

### L450-462 -- `propagate` -- `_requested_output_dx` -- the measured pre-fix mislabelling (`result.dx == 4e-05` on a 96^2 / 80 um resample)

*Left in the source:* that both spellings must be read here, and what goes wrong downstream if they are not

```text
    # v5.31 (audit W9-5): the requested output pitch can arrive EITHER as the
    # ``output_dx`` shortcut OR as the second element of the canonical
    # ``output_grid = (N_out, dx_out)`` tuple / ``{'N': ..., 'dx': ...}`` dict.
    # Pre-fix only the shortcut was read here, so an ``output_grid=(96, 80e-6)``
    # call -- which every honouring kernel really does resample to 80 um (the
    # field is bit-identical to additionally passing ``output_dx=80e-6``) --
    # came back labelled with the INPUT pitch.  MEASURED on a 64^2 / dx=40 um
    # probe: ``propagate(..., output_grid=(96, 80e-6))`` returned
    # ``field.shape == (96, 96)`` with ``result.dx == 4e-05`` for asm (via the
    # MFT promotion) and for gbd / hf / hfpi (which forward the request), i.e.
    # the wrapper's own sampling metadata was wrong by 2x on the DEFAULT
    # (post-P5-flip) contract, for every downstream coordinate / plot / store
    # consumer.  Kernels that report their own ``dx_out`` still win below.
```

### L468-472 -- `propagate` -- the coerce branch -- the 4.12 B1-7 pre-fix tuple path

*Left in the source:* what the branch does

```text
    # pitch when present.  4.12 fix (audit round-4 B1-7): kernels like
    # fresnel_propagate / fraunhofer_propagate / scalable_angular_spectrum_propagate
    # return ``(E, dx_out, dy_out)``; pre-4.12 the tuple path went
    # through _coerce_field which silently dropped to None and reported
    # the INPUT dx instead of the kernel's output dx.
```

### L482-492 -- `propagate` -- the null-field refusal -- the measured `PropagationResult(field=None)` silence

*Left in the source:* why the refusal exists and the one kernel/flag combination that reaches it

```text
    # v5.31 (audit W9-6): the whole point of the P5 flip is that ``.field`` is
    # defined "whichever kernel ran".  ``_coerce_field`` has a ``(None, None,
    # None)`` sentinel for returns it cannot read, and pre-fix that sentinel was
    # wrapped as-is -- so the flipped contract handed back a
    # ``PropagationResult(field=None)`` in complete silence.  MEASURED:
    # ``propagate(E, method='mhs', subdomains=[...], return_intermediate=True)``
    # -> ``PropagationResult`` with ``field is None``, no warning.  (MHS's
    # native shape there is a ``list`` of ``(HuygensSurface, ndarray)`` pairs,
    # and ``return_intermediate=True`` is MhsPipeline.run's OWN default -- the
    # dispatcher merely defaults it to False.)  A wrapper that cannot honour its
    # own contract must say so rather than emit a null field.
```

### L507-511 -- `propagate` -- `out_dy` -- that the y-pitch used to be discarded for anamorphic calls

*Left in the source:* what the fallback does and which calls need it

```text
    # v4.13.0 (audit L3): thread the kernel-reported ``dy_out`` onto
    # the wrapped result.  For square-grid kernels that only return
    # ``dx_out`` (or a bare ndarray) ``dy`` falls back to ``out_dx``,
    # preserving back-compat.  Pre-fix the y-pitch was silently
    # discarded for anamorphic Fresnel / Fraunhofer / SAS calls.
```

### L569-576 -- `_coerce_field` -- the v4.13.0 anamorphic info-loss closure and the 4.12 B1-7 `field=None` defect, both as pre-fix narrative

*Left in the source:* the two live bullets above them -- what the triple means and when it is ``None``

```text
    * v4.13.0 (audit L3): the triple-return is the closure for the
      anamorphic Fresnel info-loss bug -- pre-fix ``_coerce_field``
      ignored the third tuple element, silently discarding the y-axis
      pitch for any anamorphic Fresnel / Fraunhofer / SAS propagation.
    * 4.12 fix (audit round-4 B1-7): pre-4.12 the tuple-returning
      propagators (fresnel/fraunhofer/SAS) silently yielded
      ``field=None`` and ``dx=<input pitch>`` instead of the kernel's
      real output.
```

### L701-717 -- `_NO_OUTPUT_GRID_METHODS` -- the measured pre-fix silent-drop evidence for `maslov`

*Left in the source:* what the set is for, the measured evidence that gbd / hf / hfpi DO honour the request (which is what the diagnostic names), and why raising beats switching kernels

```text
# v5.31 (audit W9-4): the methods whose dispatcher branch does NOT thread
# ``output_grid`` / ``output_dx`` to its kernel.  ``maslov`` is the default
# ``method='auto'`` choice for any prescription without aspherics, so this was
# the most-travelled silent-drop path in the dispatcher.  MEASURED pre-fix on a
# 64^2 / dx=40 um singlet probe:
#
#   propagate(E, prescription=rx, output_dx=80e-6)
#     -> field BIT-IDENTICAL to the no-request call (still 40 um sampling)
#        but PropagationResult.dx reported 8e-05  <-- wrong metadata
#   propagate(E, prescription=rx, output_grid=(96, 80e-6))
#     -> shape (64, 64), dx 4e-05: the request vanished entirely, silently
#
# ``gbd`` / ``hf`` / ``hfpi`` all honour both forms (measured: shape 64->96 and
# dx 40->80 um), so the diagnostic names them.  Raising here is the 4.12 B1-8
# treatment already given to ``sas`` / ``rs``; the alternative (silently
# switching ``auto`` to ``gbd``) would trade a wrong answer for an unannounced
# 100x slowdown and a different physics model.
```

### L728-741 -- `_REQUIRED_METHOD_KWARGS` -- that two of the twelve VALID_METHODS were unusable through this entry point before the table existed

*Left in the source:* what the table is, the two concrete TypeErrors it prevents, and the deliberate refusal to invent defaults

```text
# v5.31 (audit W9-10): keyword-only arguments the kernel behind each method
# REQUIRES and cannot default.  Pre-fix the dispatcher forwarded ``**kwargs``
# blind and the caller got a raw ``TypeError`` naming a function they never
# called -- e.g. ``propagate(method='hfpi', prescription=rx)`` ->
# ``TypeError: propagate_hfpi_through_prescription() missing 1 required
# keyword-only argument: 'n_paths'`` and
# ``propagate(method='asymptotic', prescription=rx)`` ->
# ``TypeError: propagate_modal_asymptotic() missing 2 required keyword-only
# arguments: 's2_grid_x' and 's2_grid_y'``.  Two of the twelve VALID_METHODS
# were therefore unusable through this entry point as documented.  Deliberately
# NO invented defaults: ``n_paths`` is a Monte-Carlo budget and ``s2_grid_*``
# are output-plane grids: any value the dispatcher picked would be a silent
# accuracy decision.  The 4.12 B1-6 rule -- raise from the dispatcher, naming
# :func:`propagate` and everything that is missing.
```

### L835-850 -- `_auto_select_method` -- `output_requested` -- the measured pre-fix ValueError at z=1e-3 and the 'bit-for-bit unchanged' tail

*Left in the source:* the live routing rule, the remedy it applies automatically, and the B1-6 principle behind it

```text
        True when the caller passed ``output_grid`` / ``output_dx`` to
        :func:`propagate`.  v5.31 (audit W9-1): ``sas`` has no output-grid
        path -- :func:`_dispatch_bare_grid_with_output` raises for it -- so
        selecting it for a caller who asked for one produced a ``ValueError``
        naming a kernel the caller never wrote, decided purely by ``z``.
        MEASURED pre-fix at N=64, dx=2 um, lambda=633 nm: ``output_dx=3e-6``
        succeeded at ``z=1e-4`` (asm) and ``z=5`` (fraunhofer) and raised
        ``"propagate(method='sas', ...): SAS does not support arbitrary
        output-grid sampling"`` at ``z=1e-3``.  With ``output_requested`` the
        ``Q > 1`` band selects ``asm`` instead, which auto-promotes to the
        EXACT :func:`angular_spectrum_propagate_mft` -- precisely the remedy
        that SAS error message recommends, applied automatically.  This is the
        4.12 B1-6 rule ("never route the user into a hard-raise from a kernel
        they did not pick by name") applied to the B1-8 feature.  ``fraunhofer``
        is left alone: it has an MFT variant.  Routing with no output-grid
        request is bit-for-bit unchanged.
```

### L853-887 -- `_auto_select_method` -- the removed DOE branch -- the tombstone of the deleted `events_json` branch: that it could not fire, that forcing it raised, and the grating-deflection measurement that found no basis for routing to hfpi

*Left in the source:* the live fact that makes the ABSENCE of a DOE branch correct -- there is no prescription-embedded DOE representation, the declaration is a kwarg -- and the pointer to where that is enforced

```text
        # v5.31 (audit W9-9): the DOE -> 'hfpi' branch that used to sit here is
        # GONE.  It keyed on ``prescription['events_json']``, a key that at
        # HEAD occurs exactly ONCE in the repository -- in this file.  No
        # loader (``load_zemax_zmx``) and no factory (``make_singlet`` /
        # ``make_doublet`` / ...) has ever emitted it, so the branch could not
        # fire; and when forced (by hand-injecting the key) it routed to a call
        # that immediately raised
        # ``TypeError: propagate_hfpi_through_prescription() missing 1 required
        # keyword-only argument: 'n_paths'``.  Dead AND broken.
        #
        # It could not be repaired by pointing at a different key either:
        # MEASURED, this library has NO prescription-embedded DOE
        # representation at all.  Diffractive information travels as the
        # ``surface_diffraction`` / ``diffracting_surfaces`` KWARGS
        # (``{surf_index: (m_x, m_y, period_x, period_y)}``), accepted by
        # ``propagate_hfpi_through_prescription`` and
        # ``fit_canonical_polynomials`` and by nothing else --
        # ``apply_real_lens_maslov`` has no DOE parameter and raises on one.
        # There is therefore nothing on the prescription to detect.
        #
        # Nor was there a measured case for routing to ``hfpi`` automatically.
        # On the one analytic oracle available (a thin air-to-air grating, exit
        # centroid at ``t*tan(asin(m*lambda/Lambda))``), hfpi WITH
        # ``surface_diffraction`` missed the order-1 deflection by 85-97%
        # (period 40/20 um) -- no better than maslov's 100% -- so it is not
        # measurably the better automatic choice.  (That hfpi result is a
        # separate, undiagnosed finding recorded for an HFPI-interiors audit;
        # it is NOT evidence about routing beyond "no basis to prefer it".)
        #
        # What replaces the branch is honesty at the point of use: DOE kwargs
        # handed to a member that cannot accept them now raise a
        # dispatcher-level error naming the members that can, instead of the
        # raw ``apply_real_lens_maslov() got an unexpected keyword argument
        # 'surface_diffraction'`` a caller used to get from a kernel they never
        # named.  See ``_DOE_KWARGS`` in :func:`_dispatch_to_method`.
```

### L1133-1141 -- `_dispatch_to_method` -- the DOE-kwarg refusal -- the measured pre-fix `apply_real_lens_maslov() got an unexpected keyword argument` death and the dead `events_json` check

*Left in the source:* why the check has to live here: the declaration is a kwarg, so ``method='auto'`` cannot see it

```text
    # v5.31 (audit W9-9): DOE / grating kwargs handed to a member that cannot
    # accept them.  MEASURED pre-fix: ``propagate(E, prescription=rx,
    # surface_diffraction={0: (1, 0, 20e-6, 20e-6)})`` -- the library's own way
    # to declare a grating -- auto-selected ``maslov`` and died with
    # ``TypeError: apply_real_lens_maslov() got an unexpected keyword argument
    # 'surface_diffraction'``, from a kernel the caller never named.  The
    # ``events_json`` prescription check that was meant to catch this could
    # never fire (audit W9-9, see ``_auto_select_method``); the declaration the
    # library actually uses is a kwarg, and it is visible right here.
```

### L1159-1164 -- `_dispatch_to_method` -- the output-grid refusal -- the pre-fix silent drop and the mislabelled wrapper

*Left in the source:* the rule, the measured list of members that DO honour the request, and the precedent

```text
    # v5.31 (audit W9-4): ``maslov`` / ``asymptotic`` / ``mhs`` never thread
    # ``output_grid`` / ``output_dx`` to their kernels, and pre-fix the request
    # was dropped in silence -- with the ``output_dx`` shortcut the wrapper even
    # LABELLED the un-resampled field with the requested pitch.  Say so, out
    # loud, and name the members that do honour it (measured: gbd / hf / hfpi
    # all resample).  Same treatment ``sas`` / ``rs`` got in 4.12 (B1-8).
```

### L1260-1272 -- `_dispatch_to_method` -- the hfpi free-space branch -- what the old precondition check named, and which release fixed the through-prescription path first

*Left in the source:* the full list of required kwargs, where they are checked, and that the output-grid request is threaded here too

```text
            # v5.31 (audit W9-10): the precondition check that used to live
            # here named ONLY ``aperture_radius`` -- "needs at least an
            # aperture geometry" -- while the kernel also requires
            # ``z_to_aperture``, ``z_aperture_to_output`` and ``n_paths``, so
            # supplying just the advertised one still produced
            # ``TypeError: propagate_hfpi_freespace_aperture() missing 3
            # required keyword-only arguments`` (MEASURED).  All four are now
            # checked together, up front, by
            # ``_check_required_method_kwargs`` via ``_REQUIRED_METHOD_KWARGS``.
            # v5.2.5 (AUDIT_V5_2_3 P2-F1-1): thread the resolved
            # ``output_grid``/``output_dx`` through the freespace
            # branch too.  v5.2.3 fixed the through-prescription path
            # but the freespace branch silently dropped them.
```

### L1430-1486 -- `_select_asm_variant` -- `.. versionchanged:: 5.31` -- the three W9 fixes written as pre-fix/post-fix narrative, with the measured back-propagation raises, the bit-identical dropped-tilt measurement and the 4-row fidelity table that settled the regime bands

*Left in the source:* the three rules as statements of what the selector does now, plus a pointer to the table

```text
    .. versionchanged:: 5.31
       Two audit-W9 fixes, both bringing this selector in line with rules its
       twin :func:`_auto_select_method` has carried since 4.12:

       * **Back-propagation (W9-2).**  ``z < 0`` can no longer select ``sas`` /
         ``fraunhofer``.  Those kernels are forward-only and raised on the sign
         of ``z``, so :func:`asm_propagate` -- which runs whatever this returns
         -- crashed for any back-propagation past ``2 * L^2/(N*lambda)``.
         MEASURED pre-fix at N=64, dx=2 um, lambda=633 nm (threshold
         4.0442e-4 m): ``z=-1.2133e-3`` -> ``sas`` ->
         ``"scalable_angular_spectrum_propagate: z must be > 0"``;
         ``z=-1.2133e-2`` -> ``fraunhofer`` -> the analogous raise.  Every
         ASM-family member (``asm`` / ``asm_tilted`` / ``asm_mft``) accepts
         either sign, so the negative-``z`` case now stays inside that set --
         the 4.12 B1-6 guard, ported.
       * **Dropped tilt (W9-3).**  The ``asm_mft`` branch sits ABOVE the tilt
         branch and :func:`angular_spectrum_propagate_mft` has no ``tilt_x`` /
         ``tilt_y`` parameter, so a tilt passed alongside ``output_dx``
         vanished in complete silence: MEASURED bit-identical output
         (``max|difference| = 0.0``) for ``tilt_x=0.05`` versus ``tilt_x=0.0``
         at N=64, dx=2 um, z=5e-4, output_dx=3e-6.  The precedence is kept
         (there is no tilted-MFT kernel to route to) but the collision now
         emits a :class:`UserWarning` -- the same call v5.30 made for the
         sibling case, the legacy ``'propagate_tilted'`` element ignoring
         ``elem['method']``.
       * **Far-field trip re-based on the canonical criterion (W9-7).**  The
         free-space regime decision is now DELEGATED to
         :func:`_auto_select_method`, which is the library's canonical regime
         logic; this selector no longer carries thresholds of its own.  Pre-fix
         it tripped to ``'fraunhofer'`` at ``|z| > 20 * L^2/(N*lambda)``, i.e.
         at grid-Fresnel ratio ``Q > 20``.  Because ``N_F * Q = N/4``, that trip
         sits at aperture Fresnel number ``N_F = N/80`` -- it grows LINEARLY
         with the grid, so the branch fires further inside the near field the
         bigger the grid gets.  MEASURED just above the old trip
         (``z = 20.05 * L^2/(N*lambda)``, hard circular aperture filling half
         the grid, dx = 2 um, lambda = 633 nm), complex overlap fidelity against
         a pad-converged EXACT ``angular_spectrum_propagate_mft`` on the central
         8x8 patch of each candidate's own output grid:

         ======  ========  ==================  =============
         N       N_F(ap)   fid('fraunhofer')   fid('sas')
         ======  ========  ==================  =============
         128     0.399     0.9516              1.00000
         256     0.798     0.8185              1.00000
         512     1.596     0.4111              1.00000
         1024    3.192     0.4241              1.00000
         ======  ========  ==================  =============

         The canonical rule (``N_F < 0.1`` -> fraunhofer) keeps ``'sas'`` there,
         which is exact.  The SAS boundary moves with it, from ``Q > 2`` to the
         canonical ``Q > 1``; MEASURED in the newly-``'sas'`` band
         ``1 < Q <= 2`` on the same probe, both members are exact and NEITHER
         warns -- ``asm`` 0.99997-1.00000 vs ``sas`` 1.00000 at
         Q = 1.05 / 1.5 / 2.0 for N = 256 and 512 -- while at ``Q = 0.5`` (still
         ``'asm'`` under both rules) ``sas`` is the worse of the two
         (0.9955 / 0.9964), confirming ``Q > 1`` is the right place for it to
         start.
```

### L1515-1522 -- `_select_asm_variant` -- the delegation -- this selector's former `Q > 20` / `Q > 2` thresholds

*Left in the source:* what the delegation does and why ``prescription=None`` is correct here

```text
    # v5.31 (audit W9-7): delegate the free-space regime decision to
    # ``_auto_select_method``, which is the CANONICAL regime logic (see its
    # docstring).  Pre-fix this selector carried its own thresholds --
    # ``Q > 20`` -> fraunhofer, ``Q > 2`` -> sas, where
    # ``Q = lambda|z|/(N dx^2)`` -- and they disagreed with the canonical rule
    # in both bands.  See the ``versionchanged`` note above for the measured
    # fidelity table; ``prescription=None`` because every ASM-family member is
    # a bare-grid free-space kernel.
```

### L1590-1593 -- `which_propagator` -- the reason strings -- that the reasons no longer quote this module's former ASM-family-only multiples

*Left in the source:* which criteria the reasons quote and where they come from

```text
    # v5.31 (audit W9-7): the sas / fraunhofer reasons quote the CANONICAL
    # criteria (``Q > 1`` / ``N_F < 0.1``, both from ``_auto_select_method``),
    # not this module's former ASM-family-only ``L^2/(N*lambda)`` multiples,
    # which no longer decide anything.
```
