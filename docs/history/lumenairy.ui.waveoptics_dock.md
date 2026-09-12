<!-- lumenairy-history-doc
module: lumenairy/ui/waveoptics_dock.py
ast_sha256: 68bd80bfc918099c4021a37c06cf1556790270e2770af4dac06b9e2632b4a658
token_sha256: b73b2c2cbc3af6a1ff83bfe3ceb2167b2c02c88f1da28245ad11acc5ebe244d6
pre_relocation_lines: 3324
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/ui/waveoptics_dock.py`

This file holds the version-history narrative that used to live in
`lumenairy/ui/waveoptics_dock.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Eleven blocks.  This dock accumulated a long run of "the control was inert"
findings -- a combo never read, an import that never resolved, a checkbox that
was a no-op -- each recorded next to its fix in the past tense.  The MEASURED
evidence for each (the ImportError text, the ModuleNotFoundError, the ragged
3-tuple ValueError) stayed in the source, because it is what tells a reader
the control is genuinely wired now; the release attribution moved.

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
| L117-117 | `cost model` | "the old ... overestimate" |
| L228-229 | `cost model` | "the old spline path" |
| L236-236 | `cost model` | "the old spline path which had" |
| L392-394 | `_filter_wave_optics_surfaces` | "the two legs that used to meet" |
| L612-613 | `WaveOpticsWorker snapshot` | "pre-edit and post-edit" / "the old design" |
| L797-799 | `WaveOpticsWorker.run` | "Both locals used to be read from cfg and then never referenced" |
| L938-942 | `WaveOpticsWorker.run` | "The router used to swallow every exception" |
| L1074-1078 | `WaveOpticsWorker.run` | the release/audit tag and the past-tense framing |
| L1333-1334 | `WaveOpticsWorker.run / detector` | the release/audit tag |
| L2540-2545 | `WaveOpticsDock._on_mhs_finished` | the release/audit tag and "used to sit here" |
| L2663-2665 | `WaveOpticsDock._open_mft_options_dialog` | "now guards" |
| L3192-3193 | `WaveOpticsDock._on_progress` | "the old inline path" |

---

### L117-117 -- `cost model` -- "the old ... overestimate"

*Left in the source:* the cost decomposition.

```text
      NOT the old "6 FFTs per surface" overestimate.
```

### L228-229 -- `cost model` -- "the old spline path"

*Left in the source:* the measured speed ratio.

```text
        # and runs ~2-3x faster than the old RectBivariateSpline path
        # (with combined value+gradient eval + optional Numba jit).
```

### L236-236 -- `cost model` -- "the old spline path which had"

*Left in the source:* the comparison and the measured base cost.

```text
        # Smaller than the old spline path which had a ~0.15 s base.
```

### L392-394 -- `_filter_wave_optics_surfaces` -- "the two legs that used to meet"

*Left in the source:* the gap-carrying rule in full.

```text
    # the PREVIOUS kept surface: that merges the two legs that used to
    # meet at the dropped surface into the single leg of the unfolded
    # equivalent, in the medium they both live in.  Letting it vanish
```

### L612-613 -- `WaveOpticsWorker snapshot` -- "pre-edit and post-edit" / "the old design"

*Left in the source:* the snapshot rule and the mixed-state hazard it prevents.

```text
        # state (e.g. trace surfaces from the old design but the
        # prescription exported from the new one) or race a list
```

### L797-799 -- `WaveOpticsWorker.run` -- "Both locals used to be read from cfg and then never referenced"

*Left in the source:* the inert-control hazard, re-stated as what the unwired form does.

```text
        # range.  Both locals used to be read from cfg and then never
        # referenced, so the user restricted the run and silently got
        # the full system.  The span map is built on the GUI thread by
```

### L938-942 -- `WaveOpticsWorker.run` -- "The router used to swallow every exception"

*Left in the source:* the silent-fallback hazard and why the three ``lens_model_*`` result keys exist.

```text
            # dict.  The router used to swallow every exception and drop
            # through to the per-surface ASM loop in silence, so a
            # folded design -- which ``apply_real_lens`` refuses BY
            # DESIGN -- produced a thin-screen PSF labelled with the
            # analytic/traced/Maslov model the user picked.
```

### L1074-1078 -- `WaveOpticsWorker.run` -- the release/audit tag and the past-tense framing

*Left in the source:* the import rule and the measured ImportError it prevents.

```text
                # v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25,
                # Territory A UI pass): these four imports named
                # ``propagators.propagation``, the v5.1.0 re-export shell
                # for the ASM/Fresnel/RS/SAS/MFT family only -- it has
                # never exported the whole-prescription propagators.  Every
```

### L1333-1334 -- `WaveOpticsWorker.run / detector` -- the release/audit tag

*Left in the source:* all three defects and their measured signatures.

```text
                # v5.30 (audit AUDIT_ADVERSARIAL_CODEBASE_2026_07_25,
                # Territory A UI pass): three defects, all silent.
```

### L2540-2545 -- `WaveOpticsDock._on_mhs_finished` -- the release/audit tag and "used to sit here"

*Left in the source:* the prohibition and the NameError it prevents.

```text
        # v5.17 audit wave-5 (F821): a stray paste-duplicate of the
        # _on_save_toggle body used to sit here, referencing the
        # undefined name `checked` -- a NameError on every successful
        # MHS pipeline run.  The save-toggle sync belongs (and remains)
        # in _on_save_toggle below; running the pipeline must not
        # touch the save-planes state.
```

### L2663-2665 -- `WaveOpticsDock._open_mft_options_dialog` -- "now guards"

*Left in the source:* the guard and the whole destroyed-parent scenario.

```text
        The post-exec re-parent back to the original
        parent now guards against the parent having been destroyed
        while the dialog was open.  If the user closed the parent dock
```

### L3192-3193 -- `WaveOpticsDock._on_progress` -- "the old inline path"

*Left in the source:* the fallback rule.

```text
        # If fine progress is never emitted (e.g. the old inline path
        # with no sub-stages), approximate from step/total.
```

