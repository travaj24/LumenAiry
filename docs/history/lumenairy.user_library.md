<!-- lumenairy-history-doc
module: lumenairy/user_library.py
ast_sha256: fb41a8208a1e2d3965e5eabf24e216a0dc47e5361ccc24ce33d11caa21293107
token_sha256: 11d57554edd8b2ec12bfa2ea452f0c3f098c1f7834bc2505324504cab502e8f1
pre_relocation_lines: 1271
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/user_library.py`

This file holds the version-history narrative that used to live in
`lumenairy/user_library.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Six blocks.  Two threads: the stale-glass-cache invalidation (a re-pointed
registry entry must drop its cached resolution, or the OLD index is served
forever) and the swallowed-exception hygiene on the on-disk store.  Both
hazards are live, so both stayed in the source in the present tense; the
release attributions moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L147-150 | `<module> expression-mask note` | "This used to be dispatched through" |
| L219-222 | `_EXPR_MAX_* guards` | the release/audit tag |
| L519-525 | `load_material` | the "Pre-v5.17.1 a name that previously resolved" framing |
| L593-595 | `register_fixed_glass` | the audit id and the "pre-v4.14.3 the overwrite was silent" framing |
| L597-600 | `register_fixed_glass` | the release/audit tag and the "Pre-v4.14.3 accepted" framing |
| L646-647 | `register_fixed_glass` | the release/audit tag |
| L693-696 | `_decode_infinities` | "Previously this only handled" |
| L894-895 | `_check_thickness_carry` | "the old exporter" |
| L1223-1228 | `load_user_library` | the release/audit tag and the "Pre-fix this was a bare except" framing |
| L1257-1261 | `load_user_library` | the release/audit tag and the "say so instead of swallowing it" framing |

---

### L147-150 -- `<module> expression-mask note` -- "This used to be dispatched through"

*Left in the source:* the whole code-execution argument, which is why the AST interpreter exists.

```text
# (e.g. ``atan2(Y, X) * 3``) evaluated on the (X, Y) grid.  This used to
# be dispatched through the built-in ``eval()`` with the whole ``np``
# module exposed -- a code-execution risk if anyone can write to the
# library JSON (``eval("__import__('os').system(...)")`` etc.).  The
```

### L219-222 -- `_EXPR_MAX_* guards` -- the release/audit tag

*Left in the source:* the whole resource-exhaustion argument and its measured sizes -- the derivation of the guard's bar.

```text
#: v5.46 (audit Z4).  The allowlist AST interpreter below is a genuine
#: sandbox -- the auditor could not escape it -- but it left one
#: resource-exhaustion path open: CPython's ``int`` is arbitrary precision, so
#: ``2**(10**9)`` asks for a 125 MB integer (and ``1 << (10**9)`` the same)
```

### L519-525 -- `load_material` -- the "Pre-v5.17.1 a name that previously resolved" framing

*Left in the source:* the invalidation rule and the serve-the-old-index-forever hazard it prevents.

```text
        # any stale cached resolution for this name.  Pre-v5.17.1 a name
        # that previously resolved through ``_glass_cache`` (a user-fixed
        # ``_FixedIndex`` or a ``RefractiveIndexMaterial`` for a
        # different catalogue page) kept serving the OLD index forever:
        # ``get_glass_index``'s tuple branch trusts ``_glass_cache``
        # unconditionally.  Mirrors ``register_fixed_glass``'s hygiene
        # (which overwrites the cache entry and clears the value cache)
```

### L593-595 -- `register_fixed_glass` -- the audit id and the "pre-v4.14.3 the overwrite was silent" framing

*Left in the source:* the warning and the clobber hazard that motivates it.

```text
        -- but the audit P1-GL-2 noted that pre-v4.14.3 the overwrite
        was silent and could clobber catalog glasses like ``'N-BK7'``
        without warning).
```

### L597-600 -- `register_fixed_glass` -- the release/audit tag and the "Pre-v4.14.3 accepted" framing

*Left in the source:* everything the validation rejects and why.

```text
    # v4.14.3 (P1-GL-2): input validation.  Pre-v4.14.3 accepted any
    # name string (including ``''``) and any ``n`` (including
    # ``n < 1.0``, which is unphysical for ordinary materials), and
    # silently clobbered existing registry entries.
```

### L646-647 -- `register_fixed_glass` -- the release/audit tag

*Left in the source:* the lock rule and the reason for it.

```text
    # rare) and fully safe.  v5.17.1 (audit P3-40): mutations go under the
    # glass cache lock now that the value cache is a shared LRU OrderedDict.
```

### L693-696 -- `_decode_infinities` -- "Previously this only handled"

*Left in the source:* the full-tree rule and the TypeError it prevents.

```text
    ``float('-inf')``.  Previously this only handled the
    ``surfaces[i]['radius']`` slot, so any other field containing
    infinity (thickness, conic constant, aperture) came back as a
    string and caused downstream ``TypeError`` surprises.
```

### L894-895 -- `_check_thickness_carry` -- "the old exporter"

*Left in the source:* the gating condition and its reason.

```text
    has coordinate breaks, because that is the only way the old exporter
    could lose a gap.
```

### L1223-1228 -- `load_user_library` -- the release/audit tag and the "Pre-fix this was a bare except" framing

*Left in the source:* the skip-and-warn rule and the vanishing-glass failure it prevents.

```text
    not cost the user every other saved glass -- but v5.29.1 (audit A-8)
    also emits a :class:`UserWarning` naming the entry, the file on disk,
    and the exception.  Pre-fix this was a bare ``except ...: pass`` at
    both levels, so a user glass that failed to load simply vanished:
    every later ``get_glass_index('MyGlass', ...)`` raised
    "unknown glass" with nothing anywhere pointing at the real cause.
```

### L1257-1261 -- `load_user_library` -- the release/audit tag and the "say so instead of swallowing it" framing

*Left in the source:* the two-level error contract.

```text
    # underlying issue.  v5.29.1 (audit A-8): say so instead of
    # swallowing it silently.  This outer handler now only fires for
    # STORE-level failures (unreadable library directory) -- per-entry
    # failures are warned about and skipped inside
    # ``load_all_materials``.
```

