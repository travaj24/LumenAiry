<!-- lumenairy-history-doc
module: lumenairy/ui/optimizer_dock.py
ast_sha256: efbdd1913076d309f6e46d6258f7e3d67175162a7cbd88d9bf13eefacd8cf425
token_sha256: eb895bcd12050ae81c882f4f5ffb303ac769e05124a4a0b8605f716b0c384cb2
pre_relocation_lines: 2030
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/ui/optimizer_dock.py`

This file holds the version-history narrative that used to live in
`lumenairy/ui/optimizer_dock.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Three blocks.  The repeated "Nelder-Mead because `model.run_optimization()`
hardcoded it pre-v5.4" statements were NOT moved: they are the reason a live
default is what it is, which a reader changing that default has to know.

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
| L167-169 | `OptimizeWorker.run` | "which used to set nothing this worker read" |
| L494-497 | `OptimizerDock._refresh_variables` | "the grid used to be sized from the unfiltered list" |
| L920-924 | `OptimizerDock._on_finished` | the release/audit tag and the "no longer writes" framing |

---

### L167-169 -- `OptimizeWorker.run` -- "which used to set nothing this worker read"

*Left in the source:* both cancellation channels and the process-abort consequence of reading neither.

```text
            # calls Qt's requestInterruption() on close -- which used to
            # set nothing this worker read, so the 2 s wait timed out and
            # Qt aborted the process mid-run.
```

### L494-497 -- `OptimizerDock._refresh_variables` -- "the grid used to be sized from the unfiltered list"

*Left in the source:* the sizing rule and the blank-row failure it prevents.

```text
        # ``get_variable_values()`` -- the grid used to be sized from
        # the unfiltered list, so a variable whose element had been
        # deleted left a blank row that did not correspond to any
        # value the optimizer would actually move.
```

### L920-924 -- `OptimizerDock._on_finished` -- the release/audit tag and the "no longer writes" framing

*Left in the source:* the whole thread-safety contract: the worker restores and exposes, the GUI thread applies.

```text
        # v5.24.4 (audit S4-7): the background OptimizeWorker no longer
        # writes its solution into the live model off-thread -- it restored
        # the model to its pre-run state and exposed the solution vector on
        # ``worker.result_x``.  Apply it HERE, on the GUI thread, so
        # self.elements is mutated and the rebuild signal is emitted from
```

