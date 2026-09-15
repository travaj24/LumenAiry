<!-- lumenairy-history-doc
module: lumenairy/ui/model.py
ast_sha256: fd10117de41fa3cdcaba8b6248f6d5d0ef059d204763d4e1e42ac3cc75ed6848
token_sha256: e017cbbd70cf7b9b6ebc88e38118815309feecc73f2c787a799738941ab4178f
pre_relocation_lines: 3522
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/ui/model.py`

This file holds the version-history narrative that used to live in
`lumenairy/ui/model.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Nine blocks, all of the same shape: a defect the UI model once had, described
in the past tense next to the code that prevents it.  Every one of those
defects is still reachable if the code is changed back, so the HAZARD stayed
in the source, re-stated as what the wrong form does; the release attribution
moved.  The `run_optimization(method='Nelder-Mead')` back-compat rationale is
NOT here: it explains why a live default is what it is.

This module's ~200 sibling release TAGS -- the `v5.4.3 (audit GUI-resize)` /
`v5.4.4 (audit GUI-resize round 2)` boilerplate repeated across the dock
family -- were stripped in place rather than recorded here: they are one-line
labels on an otherwise-live why-comment, with no narrative attached.  The
full before/after list is in the WP-A17 SWEEP-3 report.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L278-282 | `SourceDefinition.to_source` | the release tags and the "the release that introduced the deprecation shim didn't migrate its own internal UI consumers" aside |
| L969-972 | `SystemModel.set_element_distance` | the release/audit tag |
| L1018-1021 | `SystemModel._sync_source_wavelength` | "Nothing used to keep the two equal" |
| L1887-1890 | `SystemModel.to_dict / enc_source` | "The pre-audit list omitted ..." |
| L2352-2356 | `SystemModel.world_surfaces` | "the Detector branch used to be tested first" |
| L2779-2780 | `SystemModel._ray_fan_directions` | "(the pre-fix form)" |
| L3062-3065 | `SystemModel.run_optimization` | the "Pre-v5.4 the call was hardcoded" framing |
| L3076-3077 | `SystemModel.run_optimization` | "so the pre-v5.4 default remains byte-identical" |
| L3087-3087 | `SystemModel.run_optimization` | "the pre-v5.24.4 write-back" |

---

### L278-282 -- `SourceDefinition.to_source` -- the release tags and the "the release that introduced the deprecation shim didn't migrate its own internal UI consumers" aside

*Left in the source:* the canonical call form and the DeprecationWarning that enforces it.

```text
        # Pre-v4.15.1 these 7 callsites used the legacy positional form
        # (e.g. ``Source.gaussian(w0, N, dx, wavelength)``) and so
        # emitted ``DeprecationWarning`` at v4.15.0 startup -- the
        # release that introduced the deprecation shim didn't migrate
        # its own internal UI consumers.
```

### L969-972 -- `SystemModel.set_element_distance` -- the release/audit tag

*Left in the source:* the frame agreement and the single-source-of-truth routing rule below it.

```text
            # Absolute mode: convert to relative.  v4.15 (P1-UI-4):
            # the previous-element back vertex is now expressed in the
            # SAME world-frame coords the absolute display column uses
            # (``element_z_positions_mm``, i.e. ``Element.origin[2]``).
```

### L1018-1021 -- `SystemModel._sync_source_wavelength` -- "Nothing used to keep the two equal"

*Left in the source:* the whole hazard, re-stated as what happens without the sync.

```text
        propagates it at the MODEL's wavelength.  Nothing used to keep
        the two equal, so a point-source / fiber-mode / tilted field was
        launched with the spherical phase and carrier tilt of whatever
        wavelength the source happened to be built at.
```

### L1887-1890 -- `SystemModel.to_dict / enc_source` -- "The pre-audit list omitted ..."

*Left in the source:* the round-trip requirement and the silent-revert failure a short list produces.

```text
            # Every constructor kwarg round-trips.  The pre-audit list
            # omitted polarization, the top-hat diameter and the two
            # fiber-mode fields, so restoring a saved session silently
            # reverted them to SourceDefinition's defaults.
```

### L2352-2356 -- `SystemModel.world_surfaces` -- "the Detector branch used to be tested first"

*Left in the source:* the precedence rule and the overlay-on-the-wrong-plane consequence of reversing it.

```text
        for the focal plane; the Detector branch used to be tested
        first, so those callers silently got the detector plane and
        then drew focal-plane overlays (Airy radius, distortion grid)
        on it.  Pass ``None`` -- the default -- to keep the Detector
        preference.
```

### L2779-2780 -- `SystemModel._ray_fan_directions` -- "(the pre-fix form)"

*Left in the source:* the polar decomposition and the sqrt(2) error the wrong assignment produces.

```text
            # BOTH L and M (the pre-fix form) put every ray on the x = y
            # diagonal and made the marginal ray sqrt(2) too steep.
```

### L3062-3065 -- `SystemModel.run_optimization` -- the "Pre-v5.4 the call was hardcoded" framing

*Left in the source:* the default and the back-compat guarantee that motivates it.

```text
        # dock dropdown choice through to scipy.minimize.  Pre-v5.4
        # the call was hardcoded to Nelder-Mead; we keep that as the
        # default so callers that don't pass ``method=`` see byte-
        # identical behaviour.
```

### L3076-3077 -- `SystemModel.run_optimization` -- "so the pre-v5.4 default remains byte-identical"

*Left in the source:* the bounded-methods-only rule and the unbounded default.

```text
        # (and Powell / CG / BFGS ...) is LEFT UNBOUNDED so the pre-v5.4
        # default remains byte-identical.
```

### L3087-3087 -- `SystemModel.run_optimization` -- "the pre-v5.24.4 write-back"

*Left in the source:* the whole apply_result contract and the data race it closes.

```text
        # ``apply_result=True`` and see the pre-v5.24.4 write-back.
```

