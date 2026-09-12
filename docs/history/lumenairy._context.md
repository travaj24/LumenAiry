<!-- lumenairy-history-doc
module: lumenairy/_context.py
ast_sha256: 7f580564d74cad6444ebbeb5d6a2ada18ea5bae4056b4ee3c1d6d8ee531ff0b9
token_sha256: 401ffd6d0ee935d94eb8e7bf65d38bb809dbce894b80946eec3df461f9bfaf42
pre_relocation_lines: 365
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/_context.py`

This file holds the version-history narrative that used to live in
`lumenairy/_context.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Four blocks.  The `lumenairy_context` entry-failure restore and the
registry-driven cache fan-out are both live invariants -- the "fix N, miss
N+1" drift the registry retires is exactly what a hand-listed fallback would
bring back -- so both stayed in the present tense.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L243-245 | `lumenairy_context` | the "Pre-v5.4.6 this call sat outside the try/finally" framing |
| L269-273 | `lumenairy_context cache fan-out` | the release/audit tag and the "Pre-fix it enumerated only 7 siblings" framing |
| L333-336 | `_install_atexit_restore` | the release/audit tag and the "renamed from ..." note |
| L359-364 | `install_atexit_restore alias` | the release tags on the rename |

---

### L243-245 -- `lumenairy_context` -- the "Pre-v5.4.6 this call sat outside the try/finally" framing

*Left in the source:* the restore invariant and what an unguarded call costs.

```text
    # before the ``with`` was attempted.  Pre-v5.4.6 this call sat
    # outside the try/finally, so a mid-apply exception stranded the
    # knobs that had already been set.
```

### L269-273 -- `lumenairy_context cache fan-out` -- the release/audit tag and the "Pre-fix it enumerated only 7 siblings" framing

*Left in the source:* the prohibition, the full list of what a hand-list omits, and the drift pattern it names.

```text
                # v5.24.x (audit S4-15): the fallback below no longer
                # hand-lists a subset of clearers.  Pre-fix it enumerated
                # only 7 siblings and OMITTED berreman/pmm/rcwa/glass/
                # wrapper_merit/eme_jax/bluestein -- the exact "fix N,
                # miss N+1" drift the registry was built to retire.  If
```

### L333-336 -- `_install_atexit_restore` -- the release/audit tag and the "renamed from ..." note

*Left in the source:* why the name is underscore-prefixed and when the helper runs.

```text
    Private bootstrap helper.  v5.2.5 (AUDIT_V5_2_3 P3-F4): renamed
    from ``install_atexit_restore`` to the underscore-prefixed
    form; the helper is called exactly once at the end of
    :mod:`lumenairy.__init__` during library import and has no
```

### L359-364 -- `install_atexit_restore alias` -- the release tags on the rename

*Left in the source:* the live back-compat contract: the old name keeps working, new code uses the new one.

```text
# back-compat alias.  Pre-v5.2.5 the function was named
# ``install_atexit_restore`` (no leading underscore) which made it
# look like a user-facing API surface despite being a private
# bootstrap helper.  External callers that imported it by the old
# name continue to work; new code should use the underscore-
# prefixed canonical name.
```

