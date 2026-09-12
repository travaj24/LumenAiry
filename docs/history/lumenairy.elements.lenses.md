<!-- lumenairy-history-doc
module: lumenairy/elements/lenses.py
ast_sha256: b04dab0471eb3648b5dce36f04c78fd9ea99c411bcf0c3e58fd723956a7ce0f3
token_sha256: d7d07d7d1b7df91463173d0be2f475ed0c20ae71f145fa1ed07229ca148686f6
pre_relocation_lines: 1171
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/lenses.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/lenses.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks, the comments that corrected earlier
comments, and the per-release chronologies that had accumulated on constants
whose CURRENT value is what the source now states.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

What did NOT move: the measured odd-power-rejection derivations (the `{5: 1e6}` / `{3: 1e4}` 100x sag figures), the conic-domain NaN contract and the mypy-strict re-export rationale for the 31 `X as X` lines, which is why they are spelled that way.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment plus a pointer to this file; those are noted per block
below as *Left in the source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L120-132 | `<module>`, above `_is_cupy_array` | the v5.30 E-L4/E-L5 tombstone for the two module constants deleted here, including the note that one of them documented the OPPOSITE of shipped behaviour |
| L266 | `surface_sag_general`, outside the conic domain | "Pre-4.10 silently returned 0 sag there" |
| L887-892 | the shared-Chebyshev-helpers banner | "The three Chebyshev Vandermonde helpers that used to live here (originally inlined in v3.2.2, kept here in v3.5.5 because ...) have moved to ..." |

---

### L120-132 -- `<module>`, above `_is_cupy_array` -- the v5.30 E-L4/E-L5 tombstone for the two module constants deleted here, including the note that one of them documented the OPPOSITE of shipped behaviour

*Left in the source:* the live do-not-delete note on the numexpr scaffold that follows it

```text
# v5.30 (audit E-L4/E-L5): two module constants deleted here as dead.
#
#   * ``_NEWTON_MAX_ITERS = 8`` -- 0 readers (grep-verified repo-wide:
#     the only live definition is ``_lens_traced._NEWTON_MAX_ITERS = 12``,
#     which is what ``apply_real_lens_traced`` actually uses and what
#     ``tests/unit/test_niche_audit_e_prepared_and_enums.py`` pins).  Its
#     comment additionally documented the OPPOSITE of shipped behaviour
#     ("was 12, dropped to 8 in v3.5.5"), so a reader who found it here
#     would conclude the Newton cap is 8.
#   * ``_NUMEXPR_MIN_SIZE = 1 << 20`` -- 0 readers here; the live copy
#     lives in ``_lens_real.py``, which now carries the rationale text
#     that used to sit next to this definition.
#
```

### L266 -- `surface_sag_general`, outside the conic domain -- "Pre-4.10 silently returned 0 sag there"

*Left in the source:* the rule and the artefact a silent 0 sag produces

```text
        # is not defined.  Pre-4.10 silently returned 0 sag there,
```

### L887-892 -- the shared-Chebyshev-helpers banner -- "The three Chebyshev Vandermonde helpers that used to live here (originally inlined in v3.2.2, kept here in v3.5.5 because ...) have moved to ..."

*Left in the source:* where the helpers live and what the aliases below are for

```text
# v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
# The three Chebyshev Vandermonde helpers that used to live here
# (originally inlined in v3.2.2, kept here in v3.5.5 because
# propagators/asymptotic.py imports them) have moved to
# ``lumenairy/_math/chebyshev.py``.  The underscore-prefixed aliases
# below preserve every existing internal call site in this module --
```
