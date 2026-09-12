<!-- lumenairy-history-doc
module: lumenairy/raytrace/intersection.py
ast_sha256: 866962a5e8a4ef97250d9320a96dd7f2afbac8843f23987893dc465c17ba66d1
token_sha256: 38fc4b57e4a39e4cef958818fa7f05808c89dde7adafa8fc2d8fb20f935c933f
pre_relocation_lines: 843
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/intersection.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/intersection.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Every block is the same shape: a defect that is still REACHABLE if the code
is changed back, recorded in the past tense.  The hazard stayed in the source,
re-stated as what the wrong form does; the release attribution and the
"pre-fix" framing moved.  The measured evidence -- the Cassegrain chief ray
landing 20 cm past the secondary vertex, the +/-4.121516 deg pure-tilt oracle,
the immortal-phantom bundle state -- stayed with the rule it justifies.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L13-14 | `<module> docstring` | the split-provenance claim |
| L154-158 | `_intersect_surface notes` | "A v4.12.0 attempt that also switched ..." |
| L224-227 | `_intersect_surface flat branch` | "Pre-fix it stayed alive with RAY_OK" |
| L268-271 | `_intersect_surface spherical branch` | the release/audit tag and the "replaces the prior" framing |
| L275-276 | `_intersect_surface spherical branch` | "the original workaround" |
| L279-279 | `_intersect_surface spherical branch` | the release tag |
| L400-403 | `_intersect_surface conic branch` | the release/audit tag and the "replaces the prior" framing |
| L497-500 | `_apply_aperture` | "The pre-fix line promised this in its comment but wrote" |
| L556-559 | `_refract TIR branch` | "the ``np.where`` below used to be UNCONDITIONAL" |
| L659-665 | `_transfer` | the "pre-fix" framing on both halves |
| L776-776 | `coord-break tilt block` | "Measured pre-fix on" |

---

### L13-14 -- `<module> docstring` -- the split-provenance claim

*Left in the source:* the bit-for-bit statement, as a property of the contents.

```text
No physics change: contents are bit-for-bit copies of the original
implementations.
```

### L154-158 -- `_intersect_surface notes` -- "A v4.12.0 attempt that also switched ..."

*Left in the source:* the measured 1.17e-3 cross-backend error, as the reason not to make that switch.

```text
    pre-v4.12.1.  A v4.12.0 attempt that also switched the spherical
    normal to the analytic ``(x/R, y/R, (z-R)/R)`` form (matching
    :mod:`jax_trace`) compounded a 1.17e-3 cross-backend rel error in
    the Maslov asymptotic test -- this conservative variant avoids
    that drift.
```

### L224-227 -- `_intersect_surface flat branch` -- "Pre-fix it stayed alive with RAY_OK"

*Left in the source:* the phantom-ray failure mode and the JAX-parity note.

```text
        # above must not be reported as a hit.  Pre-fix it stayed alive with
        # RAY_OK -- an IMMORTAL PHANTOM that walked a 4-flat stack accruing
        # opd = 0.0 and was still counted in the alive/centroid/RMS-spot
        # summary.  Both JAX kernels already kill it (_intersect_jax pure-flat
```

### L268-271 -- `_intersect_surface spherical branch` -- the release/audit tag and the "replaces the prior" framing

*Left in the source:* the direction-aware rule and the backward-ray failure it prevents.

```text
        # the ray's current direction).  v5.4.1 (audit P1): replaces
        # the prior direction-blind ``t = t1 if R > 0 else t2`` which
        # produced wrong-side-of-sphere results for any backward-
        # propagating ray (N=-1 after a reflection).  Audit reproducer
```

### L275-276 -- `_intersect_surface spherical branch` -- "the original workaround"

*Left in the source:* the cross-reference and the measured Cassegrain reproducer above it.

```text
        # analysis/ghost.py:_ghost_intersect for the original
        # workaround (now a thin alias).
```

### L279-279 -- `_intersect_surface spherical branch` -- the release tag

*Left in the source:* the tangency semantics and why disc == 0 is accepted.

```text
        # disc < 0: ray entirely misses the sphere.  v5.4.6 (audit P3-3):
```

### L400-403 -- `_intersect_surface conic branch` -- the release/audit tag and the "replaces the prior" framing

*Left in the source:* the same rule and failure, at the conic seed.

```text
            # the ray's current direction).  v5.4.1 (audit P1): replaces
            # the prior direction-blind ``t = t1 if R > 0 else t2`` which
            # produced wrong-side-of-sphere results for any backward-
            # propagating ray (N=-1 after a reflection) -- the Newton
```

### L497-500 -- `_apply_aperture` -- "The pre-fix line promised this in its comment but wrote"

*Left in the source:* the first-failure-wins rule and exactly what an unconditional write costs.

```text
                # pre-fix line promised this in its comment but wrote
                # ``np.where(clipped, ...)`` unconditionally, so a ray
                # already carrying an earlier diagnosis (e.g. RAY_TIR from
                # this surface's refraction on a previous pass, or a code
```

### L556-559 -- `_refract TIR branch` -- "the ``np.where`` below used to be UNCONDITIONAL"

*Left in the source:* the rule, the cross-reference to the sibling site, and the invariant argument.

```text
        # below used to be UNCONDITIONAL, i.e. it relabelled any code a
        # ray was already carrying -- the exact defect the aperture block
        # 50 lines down documents having fixed.  Harmless while the
        # ``alive => error_code == RAY_OK`` invariant holds (``newly_tir``
```

### L659-665 -- `_transfer` -- the "pre-fix" framing on both halves

*Left in the source:* the teleport hazard and the measured bundle state, plus the note that the state is reachable by design through trace's DOE branch.

```text
    pre-fix ``t`` was masked to 0 for them but ``rays.z`` was reset to the
    next vertex plane UNCONDITIONALLY, so a ray parallel to the axis-normal
    planes was TELEPORTED one gap downstream with zero OPL and stayed
    ``alive=True, error_code=0`` -- the "immortal phantom" that R-4 removed
    from ``_intersect_surface``'s flat branch but not from here.  Measured
    pre-fix on a bundle at ``z = 1e-4`` with ``N = 0``:
    ``_transfer(10 mm, n=1)`` returned ``z=[0 0], alive=[T T], opd=[0 0],
```

### L776-776 -- `coord-break tilt block` -- "Measured pre-fix on"

*Left in the source:* the whole two-renderer sign argument and both measured angles.

```text
        # and to ``trace_world``.  Measured pre-fix on a pure-tilt oracle
```

