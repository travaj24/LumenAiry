<!-- lumenairy-history-doc
module: lumenairy/optimize/multiconfig.py
ast_sha256: df08d2bc34706cb8d65d8e9ab245eed31250a3f8f7a017b74cab37cfc9846e90
token_sha256: 3d276d334b842777026e047c3a23c699a1032f9fd821e9220fb293e17991b83e
pre_relocation_lines: 483
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/optimize/multiconfig.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/multiconfig.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

One physics bug told three times: the thin-lens lensmaker formula was
evaluated with a hardcoded `n = 1.5` regardless of the glass, which puts the
surface radii ~17 % off for a high-index glass.  The MEASUREMENT (N-LASF9 at
n ~ 1.85, 17 % radius error, and the `_zero_C_air_gap` correction masking it
by making the system afocal from wrong focal lengths) is what tells a future
editor why the lookup cannot be shortcut, so it stayed in the source --
re-stated as what hardcoding WOULD cost.  The release tags moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L27-34 | `_resolve_lens_glass_index` | the release/agent tag and the "pre-v4.14.3 both ... hardcoded" framing |
| L59-65 | `_resolve_lens_glass_index` | the release/agent tag and the "Pre-v4.15.1 this bounds check was exclusive" framing |
| L317-317 | `_zero_C_air_gap` | the release and audit tag |
| L370-375 | `beam_expander_prescription` | the release/agent tag and the past-tense framing |
| L380-382 | `beam_expander_prescription` | the "(The previous version used ...)" parenthetical |
| L447-447 | `keplerian_telescope` | the release and agent tag |

---

### L27-34 -- `_resolve_lens_glass_index` -- the release/agent tag and the "pre-v4.14.3 both ... hardcoded" framing

*Left in the source:* the measured consequence in full, which is the argument for the helper existing.

```text
    v4.14.3 (P1-MC / Agent B): pre-v4.14.3 both
    :func:`beam_expander_prescription` and :func:`keplerian_telescope`
    hardcoded ``n=1.5`` in the thin-lens lensmaker formula
    ``R = f*(n-1)*2``.  For ``glass='N-LASF9'`` (n ~ 1.85 at 587.6 nm)
    that produced surface radii 17% off from the requested focal
    length; the downstream ``_zero_C_air_gap`` correction made the
    system afocal but the focal lengths feeding it were wrong.  Real
    physics error -- this helper centralises the canonical lookup.
```

### L59-65 -- `_resolve_lens_glass_index` -- the release/agent tag and the "Pre-v4.15.1 this bounds check was exclusive" framing

*Left in the source:* both bounds and the reason for each -- the vacuum/air inclusive case and the Si/Ge upper edge -- as live constraints on the range.

```text
        # v4.15.1 (P3-1 / Agent E): consistency with
        # :func:`user_library.register_fixed_glass`, which accepts
        # ``n=1.0`` inclusively (vacuum / air; the canonical zero-
        # phase reference).  Pre-v4.15.1 this bounds check was
        # exclusive (``1.0 < n < 5.0``), so an "air" Sellmeier entry
        # at n=1.0 exactly was rejected with a misleading "outside
        # expected range" message.  Upper bound widened to 4.0 to
```

### L317-317 -- `_zero_C_air_gap` -- the release and audit tag

*Left in the source:* the whole degenerate-geometry argument and the note that both callers already catch RuntimeError.

```text
        # v4.13.2 (P1-NEW-G): a silent ``return g1`` here disguised
```

### L370-375 -- `beam_expander_prescription` -- the release/agent tag and the past-tense framing

*Left in the source:* the measured error and the reason the correction downstream cannot rescue it.

```text
    # v4.14.3 (P1-MC / Agent B): use the prescription's actual glass
    # at its design wavelength rather than the hardcoded ``n=1.5``
    # approximation.  For ``glass='N-LASF9'`` (n ~ 1.85 at 587.6 nm)
    # the hardcoded value put the lensmaker formula off by ~17% in
    # surface radius; the downstream ``_zero_C_air_gap`` correction
    # could not recover the underlying focal-length error.
```

### L380-382 -- `beam_expander_prescription` -- the "(The previous version used ...)" parenthetical

*Left in the source:* the shape rule and exactly what the wrong shape costs -- re-stated as a hazard rather than as a past defect.

```text
    # it has the correct focal length.  (The previous version used
    # [R_eye, inf] which halved the eyepiece focal length, giving a
    # beam expander whose magnification was half the requested M.)
```

### L447-447 -- `keplerian_telescope` -- the release and agent tag

*Left in the source:* the cross-reference and the rationale summary.

```text
    # v4.14.3 (P1-MC / Agent B): see ``beam_expander_prescription``
```

