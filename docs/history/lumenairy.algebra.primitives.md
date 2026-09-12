<!-- lumenairy-history-doc
module: lumenairy/algebra/primitives.py
ast_sha256: 64f2ab776747b4c8387c7d030dfd9542f254d465a952b8e4315fffcd7d73f47e
token_sha256: c3fe0bf95ce274d5243649a9e9c6dd882de09eda5f1707984eb96851b3816414
pre_relocation_lines: 836
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/algebra/primitives.py`

This file holds the version-history narrative that used to live in
`lumenairy/algebra/primitives.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Four blocks.  The `.. versionchanged:: 5.46` default flip on `FreeSpace`
was NOT moved: it is a live migration statement with a measured before/after
a user needs when their chain's output pitch changes.  What moved is the
"pre-fix the call dropped dy" framing and the crash it names.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L175-175 | `FreeSpace._maybe_warn_far_field` | "a pre-v5.46 process" |
| L235-236 | `FreeSpace far-field warning` | "Before v5.46 that cell was silent in BOTH directions -- the old default" |
| L296-299 | `FreeSpace._apply` | the "Pre-fix the call dropped dy" framing |
| L813-815 | `Lens 3-stage chain` | "no longer triggers ... that the v4.15.2 closure exposed" |

---

### L175-175 -- `FreeSpace._maybe_warn_far_field` -- "a pre-v5.46 process"

*Left in the source:* the getattr rule and its reason.

```text
        # pre-v5.46 process has no such attribute.
```

### L235-236 -- `FreeSpace far-field warning` -- "Before v5.46 that cell was silent in BOTH directions -- the old default"

*Left in the source:* both measured cells (0.096 vs 0.9998 retained power, the 0.07 % width error) and the reason the warning exists.

```text
        directions -- the old default gave the right answer and warned about
        an unrelated return contract.
```

### L296-299 -- `FreeSpace._apply` -- the "Pre-fix the call dropped dy" framing

*Left in the source:* the anamorphic rule and the silent dy = dx fallback it prevents.

```text
        # correct y-axis grid pitch.  Pre-fix the call dropped
        # ``dy`` and the underlying kernel silently defaulted to
        # ``dy = dx``.  ``propagate`` forwards ``**method_kwargs``
        # to the chosen kernel; ASM / Fresnel / Fraunhofer / RS all
```

### L813-815 -- `Lens 3-stage chain` -- "no longer triggers ... that the v4.15.2 closure exposed"

*Left in the source:* the composition argument and the exact crash it rules out.

```text
        # grid no longer triggers the SAS-anamorphic
        # ``TypeError: sas_propagate() got an unexpected keyword
        # argument 'dy'`` crash that the v4.15.2 closure exposed.
```

