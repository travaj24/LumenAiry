<!-- lumenairy-history-doc
module: lumenairy/propagators/mhs.py
ast_sha256: bf7b57878e8216a12a8afa19bd13549b62ecd1bef5edc74ad489f426e4d29503
token_sha256: 32aed24747af4dfb0beab107f92d067f0afd3f92b2cb1ad4e8a08b6d87a3f8c1
pre_relocation_lines: 755
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/mhs.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/mhs.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

The first block is the most interesting one in this sweep, because it is a
comment that existed only to retract another comment.  `mhs.py`'s module
docstring used to open by describing ray bundles propagating geometrically
between Huygens surfaces and being converted to fields by a Huygens-surface
integral -- none of which this module implements.  Audit K15/K24 replaced that
paragraph with an accurate one and then, correctly for an audit, left a
paragraph explaining what had been removed and why.  That explanation is here;
the accurate description, and the live clarification that the IEEE 2023 paper
is BACKGROUND rather than a description of the code, stayed.

The `.. note::` about the two constructors disagreeing on the default method
(`dispatcher_subdomain` defaults to `'maslov'`, `MhsPipeline.from_prescription`
to `'gbd'`) also stayed in full: it is a live hazard with an explicit
instruction, not history.

Nothing the interpreter executes changed in the move.  The header above
records the SHA-256 of (a) the module's AST with every docstring removed and
source positions ignored, and (b) its `tokenize` stream reduced to
NAME/OP/NUMBER/STRING with comments and docstrings dropped -- both taken from
the file as it stood BEFORE the relocation.
`tests/unit/test_audit2609_a17_history_relocation.py` re-computes both from
the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L13-21 | `<module>` | a comment correcting an earlier COMMENT: the opening paragraph that described ray bundles and Huygens-surface integrals this module does not implement |
| L176-180 | `MhsPipeline._validate` -- surface matching | the measured pre-fix acceptance (a 50 um and a 1 mm centre offset both ran to completion with no diagnostic) |
| L559-600 | `dispatcher_subdomain` -- three stacked `versionchanged` directives | the v5.2 raise-then-v5.2.3-resample sequence for `method='maslov'`, and the flip-day account of the v5.30 return-contract change |
| L641-648 | `dispatcher_subdomain` -- the maslov call | the `flip-day migration` framing and the roadmap's not-flip-safe inventory entry |
| L671-679 | `dispatcher_subdomain` -- the resample re-normalisation | that the block used to be duplicated verbatim here and in `hf.py` |
| L712-716 | `dispatcher_subdomain` -- the non-maslov call | the `flip-day migration` framing |

---

### L13-21 -- `<module>` -- a comment correcting an earlier COMMENT: the opening paragraph that described ray bundles and Huygens-surface integrals this module does not implement

*Left in the source:* the accurate description (the paragraph above it) and the live clarification that the IEEE 2023 paper is background, not a description of the code

```text
K15/K24 (audit 2026-09-11): the paragraph that used to open this
docstring said the module "splits the propagation volume into
subdomains... within each subdomain, rays propagate geometrically...
at each Huygens surface, the ray bundle is converted to a complex field
via a Huygens-surface integral".  None of that is implemented here; the
text further down (from "This module provides the structural
framework") always was the accurate description.  The IEEE 2023 Multiple
Huygens Surface paper is BACKGROUND -- the framework this module's
composition API is shaped after -- not a description of the code.
```

### L176-180 -- `MhsPipeline._validate` -- surface matching -- the measured pre-fix acceptance (a 50 um and a 1 mm centre offset both ran to completion with no diagnostic)

*Left in the source:* why ``centre`` is part of the grid, which is the reason the check includes it

```text
                # different coordinate systems -- and the pre-fix check
                # accepted them, silently discarding the transverse jump
                # (measured: a 50 um offset, and a 1 mm one, both ran to
                # completion with no diagnostic while each propagator
                # worked on its own in_surface).
```

### L559-600 -- `dispatcher_subdomain` -- three stacked `versionchanged` directives -- the v5.2 raise-then-v5.2.3-resample sequence for `method='maslov'`, and the flip-day account of the v5.30 return-contract change

*Left in the source:* what the maslov branch does now, the non-resampling alternative, the narrow retained raise, and the ``return_result=False`` contract with the reason it is not forwardable

```text
    .. versionchanged:: 5.2
        v5.2 (AUDIT_V4_13_1 Part 2 P1-C, option a -- raise) raised
        ``ValueError`` at subdomain construction time when
        ``method='maslov'`` was paired with an ``out_surface`` declaring
        a grid that differed from ``in_surface``.  Pre-v5.2 the request
        was silently dropped (the maslov dispatcher branch does not
        forward ``output_grid`` / ``output_dx`` to
        :func:`apply_real_lens_maslov`) and downstream MHS pipeline
        stitching saw a mis-shaped intermediate field.

    .. versionchanged:: 5.2.3
        v5.2.3 (AUDIT_V4_13_1 P1-C substantive closure): the maslov
        branch now **actually resamples** onto ``out_surface`` instead
        of raising.  The propagation runs natively on the input grid
        (which is all the maslov kernel supports), then a one-step
        :func:`resample_field` lands the output on ``out_surface``.
        Total power is re-normalised across the resample so the
        resampling step preserves L2 energy to within the bicubic
        interpolator's numerical precision.

        For a non-resampling alternative -- when the resample step's
        bicubic interpolation error at the new grid's edges is not
        acceptable -- pass ``method='gbd'`` / ``'hfpi'`` / ``'hf'``
        (their underlying propagators sample the output grid natively
        via Bluestein chirp-Z or HFPI integration).

        The narrow retained-raise corner case: the maslov kernel itself
        requires a SQUARE input field (``E_in.shape[0] == E_in.shape[1]``)
        and SQUARE pixels (``dx == dy``).  If the caller hands the
        subdomain a non-square input grid or a non-square output grid,
        we raise at construction time (rather than letting the kernel
        raise mid-pipeline).

    .. versionchanged:: 5.30
        Both dispatcher calls now pass ``return_result=False`` explicitly
        (audit P5 / roadmap Part F1): v5.30 flipped
        :func:`~lumenairy.propagators.dispatch.propagate`'s default return to
        a :class:`~lumenairy.propagators.PropagationResult`, and this
        subdomain must hand a bare **field** to the MHS pipeline.  Outputs are
        bit-identical to pre-v5.30.  A ``return_result`` passed through
        ``**method_kwargs`` is therefore ignored rather than forwarded -- the
        subdomain's own return contract is the pipeline's, not the caller's.
```

### L641-648 -- `dispatcher_subdomain` -- the maslov call -- the `flip-day migration` framing and the roadmap's not-flip-safe inventory entry

*Left in the source:* the contract and the reason it has to be named at the call site

```text
            # v5.30 (audit P5 / roadmap F1, flip-day migration): pass
            # ``return_result=False`` explicitly.  This function's contract is
            # to hand a bare FIELD back to the MHS pipeline (it is measured
            # with ``np.abs`` / ``.dtype`` just below and stitched into the
            # next subdomain), and since v5.30 the dispatcher's DEFAULT return
            # is a ``PropagationResult``.  The roadmap's F1 inventory listed
            # this site as NOT flip-safe for exactly that reason; naming the
            # legacy contract keeps the output bit-identical.
```

### L671-679 -- `dispatcher_subdomain` -- the resample re-normalisation -- that the block used to be duplicated verbatim here and in `hf.py`

*Left in the source:* what the shared helper restores and the measured cost of the alternative (a window holding 67.27 % of the power returned carrying 100.00 %) -- which is what stops someone re-normalising to the full source power again

```text
            # K11 (audit 2026-09-11): this block used to be duplicated
            # verbatim here and in ``hf.py``, and BOTH renormalised to
            # the FULL source power -- which fabricates energy whenever
            # the declared out_surface window is smaller than the source
            # extent (measured: a window genuinely holding 67.27 % of the
            # power was returned carrying 100.00 %, amplitudes 1.219x).
            # One shared helper now restores the interpolation drift
            # only, measuring its reference power inside the target
            # window on the source grid, and warns on a real crop.
```

### L712-716 -- `dispatcher_subdomain` -- the non-maslov call -- the `flip-day migration` framing

*Left in the source:* the reason the legacy contract is named here

```text
        # v5.30 (audit P5 / roadmap F1, flip-day migration): ``return_result``
        # named explicitly for the same reason as the maslov branch above --
        # this return goes straight into the MHS pipeline as a field, so it
        # must stay the kernel's bare output now that the dispatcher default
        # is a ``PropagationResult``.
```
