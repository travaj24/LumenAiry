<!-- lumenairy-history-doc
module: lumenairy/backend/fft.py
ast_sha256: df8331594011ee7a56c05409f667b0da0c395fbaf6b02609ccfbef01f861bc17
token_sha256: eaf7382315b336ea13c0e458e9b304a8efe1c686f3784426d44f92ac50546a5a
pre_relocation_lines: 253
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/backend/fft.py`

This file holds the version-history narrative that used to live in
`lumenairy/backend/fft.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Three blocks.  The dependency direction (`backend/fft.py` imports
`propagators/fft_infra`, never the `propagation` shell) is a live rule and
stayed; the three-release story of how the inversion was removed moved here.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L29-36 | `<module> docstring` | the release / roadmap tags and the three-release move narrative |
| L53-56 | `<module> _jnp accessor note` | "Dead code removed in v5.29.1" |
| L63-63 | `_jnp_required` | the release/audit tag |
| L112-113 | `<module> zero-copy note` | "a previously-returned field" |

---

### L29-36 -- `<module> docstring` -- the release / roadmap tags and the three-release move narrative

*Left in the source:* the import rule and both reasons for it.

```text
v5.2 (ROADMAP "backend/fft.py -> propagators/propagation.py
inversion" cleanup): pre-v5.1, the FFT infra lived inside
``propagators/propagation.py`` and ``backend/fft.py`` had to import
through that monolith.  v5.1 lifted the infra to ``fft_infra.py``;
v5.2 routes ``backend/fft.py`` directly through ``fft_infra``
(``from ..propagators import fft_infra as _prop``) so the inversion
through the propagation shell is removed and ``__getattr__``
forwarding (PEP-562) no longer sits in the hot FFT path.
```

### L53-56 -- `<module> _jnp accessor note` -- "Dead code removed in v5.29.1"

*Left in the source:* the standing statement that no such helper exists and which accessor is live.

```text
# Dead code removed in v5.29.1 (audit A-9..A-14): ``_jnp_or_none()`` had
# zero references repo-wide (grep-verified over every .py/.pyi/.md/.cfg/
# .toml in the tree -- the only hits were its own ``def`` line and the
# audit report naming it).  ``_jnp_required`` below is the live accessor;
```

### L63-63 -- `_jnp_required` -- the release/audit tag

*Left in the source:* the centralised None-narrow rationale.

```text
    installed).  v5.2 (AUDIT_V5_1_0 P2-NEW-F2-2 mypy strict closure):
```

### L112-113 -- `<module> zero-copy note` -- "a previously-returned field"

*Left in the source:* the whole aliasing hazard and the two in-library sites that hit it.

```text
      previously-returned field silently becoming byte-identical to a
      later leg's result.
```

