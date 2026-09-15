<!-- lumenairy-history-doc
module: lumenairy/_math/chebyshev.py
ast_sha256: 469a8bddcdde98c07c16d415a43da80f5fa465314f1e9b0ebaafc474f11a23bd
token_sha256: cfad968fe14458e96720b37b5f0a226a4fb769e84b13d7c20414bd72a5e88243
pre_relocation_lines: 456
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->


# Version history -- `lumenairy/_math/chebyshev.py`

This file holds the version-history narrative that used to live in
`lumenairy/_math/chebyshev.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

All of this module's history is about the EXTRACTION that created it: the
helpers were inlined in `elements/lenses.py`, `propagators/` reached into
`elements/` for a math primitive, and the extraction inverted that.  The
dependency RULE -- this is a propagator-free math root -- is live and stayed;
the move narrative and the per-branch "moved verbatim from X" provenance
moved here.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L8-18 | `<module> docstring` | the release / roadmap tag and the move narrative |
| L78-79 | `chebyshev_vandermonde` | the release tag and "moved verbatim from elements/lenses.py" |
| L89-91 | `chebyshev_vandermonde` | the release tag and the "moved verbatim from ... (formerly _chebyshev_vandermonde_xp)" provenance |
| L127-128 | `chebyshev_derivative_vandermonde` | the release tag and the move provenance |
| L145-145 | `chebyshev_derivative_vandermonde` | the release/audit tag |
| L204-205 | `chebyshev_second_derivative_vandermonde` | the release tag and the move provenance |
| L219-219 | `chebyshev_second_derivative_vandermonde` | the release/audit tag |
| L302-303 | `fit_chebyshev_2d` | "The round-trip claim used to be stated unconditionally" |
| L326-326 | `fit_chebyshev_2d` | the release/phase tag |

---

### L8-18 -- `<module> docstring` -- the release / roadmap tag and the move narrative

*Left in the source:* the inverted-dependency reason for the module's existence, the import rule and the back-compat aliases.

```text
the Maslov machinery used to live there; the asymptotic propagator
then imported them from ``elements/``, creating an inverted
dependency where ``propagators/`` reached into ``elements/`` for a
math primitive.

v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
The three helpers move here, into a propagator-free math root.
``elements/lenses.py`` and every ``propagators/asymptotic*.py``
consumer now imports from ``lumenairy._math.chebyshev`` directly.
Underscore-prefixed back-compat aliases are preserved at the old
import site so external callers keep working.
```

### L78-79 -- `chebyshev_vandermonde` -- the release tag and "moved verbatim from elements/lenses.py"

*Left in the source:* nothing but the branch label -- the path speaks for itself.

```text
        # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
        # original NumPy path, moved verbatim from elements/lenses.py.
```

### L89-91 -- `chebyshev_vandermonde` -- the release tag and the "moved verbatim from ... (formerly _chebyshev_vandermonde_xp)" provenance

*Left in the source:* the shape contract and the jit/grad-traceability requirement.

```text
    # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
    # xp-dispatched path, moved verbatim from propagators/asymptotic_jax_twin.py
    # (formerly ``_chebyshev_vandermonde_xp``).  Returns a stacked array
```

### L127-128 -- `chebyshev_derivative_vandermonde` -- the release tag and the move provenance

*Left in the source:* the branch label.

```text
        # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
        # original NumPy path, moved verbatim from elements/lenses.py.
```

### L145-145 -- `chebyshev_derivative_vandermonde` -- the release/audit tag

*Left in the source:* the whole construction rule and its jit/grad rationale.

```text
    # v5.2.5 (AUDIT_V5_2_3 P3-F1-3 chebyshev derivative xp dispatch):
```

### L204-205 -- `chebyshev_second_derivative_vandermonde` -- the release tag and the move provenance

*Left in the source:* the branch label.

```text
        # v5.2 (ROADMAP v5.1 shared Chebyshev helpers extraction):
        # original NumPy path, moved verbatim from elements/lenses.py.
```

### L219-219 -- `chebyshev_second_derivative_vandermonde` -- the release/audit tag

*Left in the source:* the construction and the recurrence identity with the NumPy branch.

```text
    # v5.2.5 (AUDIT_V5_2_3 P3-F1-3 chebyshev derivative xp dispatch):
```

### L302-303 -- `fit_chebyshev_2d` -- "The round-trip claim used to be stated unconditionally"

*Left in the source:* the shift argument and the warning, which is the live contract.

```text
        round-trip claim used to be stated unconditionally; an
        off-centre grid whose fit has any NON-CONSTANT term now emits a
```

### L326-326 -- `fit_chebyshev_2d` -- the release/phase tag

*Left in the source:* why the fit lives here rather than in the UI dock.

```text
    # v5.4 Phase 5: lstsq-based 2-D Chebyshev fit promoted out of the
```

