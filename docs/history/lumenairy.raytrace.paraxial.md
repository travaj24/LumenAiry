<!-- lumenairy-history-doc
module: lumenairy/raytrace/paraxial.py
ast_sha256: 9065a4a64c9b653df35955a0f88143ea0056713f36fa6dc08aaa6974cfce18a4
token_sha256: d680af7d9577f8a2670edf41f919d6ec96b9a1464dcd2a53a96c49a605c9304a
pre_relocation_lines: 306
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/paraxial.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/paraxial.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Three blocks.  All three are convention corrections, and in each case the
WRONG form is still reachable -- one of them survives deliberately as a
warned fallback -- so the argument stayed in the source.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L74-85 | `field_of_view` | the "used to return" / "The old expression" framing |
| L162-164 | `hyperfocal-style helper` | the release/audit tag and "docstring formula corrected to match the code" |
| L223-227 | `f-number helper` | the "Pre-fix this function returned the SIGNED ratio" framing |

---

### L74-85 -- `field_of_view` -- the "used to return" / "The old expression" framing

*Left in the source:* the whole convention argument, the imaging relation, and the reason the aperture form is kept as a warned fallback.

```text
    branch used to return ``arctan((aperture/2) / object_distance)`` --
    the object-space APERTURE half-angle, which is independent of the
    sensor and is therefore not a field of view at all (it is the
    marginal-ray cone, i.e. the numerical aperture).  It is now computed
    from the transverse magnification,
    ``m = -image_distance / object_distance`` via the Newtonian /
    Gaussian imaging relation, giving
    ``h_obj_max = sensor_half_height / |m|`` and
    ``theta_max = arctan(h_obj_max / object_distance)``.  The old
    expression survives only as the explicitly-warned fallback when the
    caller supplies no sensor size, because removing it outright would
    break callers that relied on the (mislabelled) return value.
```

### L162-164 -- `hyperfocal-style helper` -- the release/audit tag and "docstring formula corrected to match the code"

*Left in the source:* the correct formula and the specific wrong form it must not drift back to.

```text
    .. v5.4.6 (audit F-25): docstring formula corrected to match the
        code ``H = (D/2) * (h/efl)``; the prior ``h*D/(4*(f/#)*f)`` form
        carried a spurious extra division by ``2*(f/#)``.
```

### L223-227 -- `f-number helper` -- the "Pre-fix this function returned the SIGNED ratio" framing

*Left in the source:* the convention and the three siblings it must agree with.

```text
    ``abs(EFL) / D``.  Pre-fix this function returned the SIGNED ratio,
    so a diverging prescription read ``f/-9.97`` while all three
    siblings that compute the same quantity -- ``raytrace.layout``
    (``abs(efl) / ap``), ``optimize.merit_terms.MaxFNumberMerit``
    (``abs(ctx.efl) / ap``) and ``seidel.compute_pupils`` -- reported
```

