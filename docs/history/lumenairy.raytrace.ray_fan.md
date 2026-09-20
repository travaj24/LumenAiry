<!-- lumenairy-history-doc
module: lumenairy/raytrace/ray_fan.py
ast_sha256: d34ab63465a8bc62e05e206d021199a934c64f00da5c7773907a82a07f4c8d25
token_sha256: c2719cdbe76128e54c58a50036711ffcc7328e4c6521a78992191723f7b72327
pre_relocation_lines: 1074
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B9 item 3: both ray fans and both OPD fans issue ONE concatenated trace via the new _trace_fan_set / _bundle_slice helpers instead of four; through_focus_rms gains pattern= for the area-uniform pupil
re_recorded: 2026-09-13 -- VERIFY-WP-B9: _trace_fan_set reads each input bundle's own error_code (the np.zeros stand-in would relabel a dead ray RAY_OK) and names the absolute Newton tolerance
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-20 -- WP-C2 round 2 (VERIFY-WP-C2 D4): this module's exported internally-tracing entry point(s) take the tracer's own sphere_normal= / renormalize= keywords (default None, which stamps nothing) and forward them verbatim to the trace call, so the pre-WP-C2 arithmetic is one keyword away; 742/742 arrays byte-identical archive to archive on both builds
-->


# Version history -- `lumenairy/raytrace/ray_fan.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/ray_fan.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Five blocks.  Every one is a measured defect whose hazard is still live: the
telecentric `ep_z = inf` NaN fan, the chief/fan pupil-zone mismatch, the
missing reference sphere, the empty-`focus_shifts` IndexError, and the
`__all__` asymmetry that the walker symmetry test depends on.  All the
arguments stayed; the past-tense framing moved.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L58-61 | `_ep_centring_offset` | the past-tense framing on the NaN propagation |
| L236-236 | `opd_fan_data helper` | "matching the pre-R1 contract" |
| L477-478 | `ray_fan_data` | the release tag and the "The 4.11.2 fix moved only" framing |
| L713-717 | `opd_fan_data` | the "Pre-fix this function returned" framing |
| L1005-1008 | `through_focus_rms` | "used to fall through" |
| L1064-1068 | `__all__` | the "pre-v5.1.0" framing |

---

### L58-61 -- `_ep_centring_offset` -- the past-tense framing on the NaN propagation

*Left in the source:* the whole telecentric case and its consequence, as a live hazard.

```text
    at the pre-stop group's rear focal plane, so ``A_pre = 0``): on-axis
    that is ``inf * tan(0) = inf * 0 = NaN``, and the NaN then propagated
    into every launched ray height, so ``ray_fan_data`` /
    ``opd_fan_data`` returned all-NaN fans with no diagnostic.  An
```

### L236-236 -- `opd_fan_data helper` -- "matching the pre-R1 contract"

*Left in the source:* the dead-ray convention.

```text
    surfaces.  Dead rays come back NaN, matching the pre-R1 contract.
```

### L477-478 -- `ray_fan_data` -- the release tag and the "The 4.11.2 fix moved only" framing

*Left in the source:* the pupil-zone argument and the ``ey(0) == ex(0) == 0`` property it buys.

```text
    # fan no longer passed through zero at py=0 (``ey(0)`` read the launch-
    # convention offset instead of 0).  We shift each fan's LAUNCH heights
```

### L713-717 -- `opd_fan_data` -- the "Pre-fix this function returned" framing

*Left in the source:* the first-order error term, which is the reason the reference sphere is required.

```text
    R1 (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11).  Pre-fix this function
    returned ``(img.opd - opd_chief) / wavelength``, i.e. the OPL to each
    ray's OWN intercept -- the reference sphere was missing entirely.
    That differs from the wavefront error at FIRST order in the
    transverse aberration: ``W_plane - W_true = eps * sin(theta')``.
```

### L1005-1008 -- `through_focus_rms` -- "used to fall through"

*Left in the source:* the exact IndexError the guard replaces with a named error.

```text
    # ``focus_shifts`` used to fall through the whole sweep and die at the
    # ``focus_shifts[best_idx]`` return with a bare
    # ``IndexError: index 0 is out of bounds for axis 0 with size 0``,
    # naming neither this function nor the offending argument.
```

### L1064-1068 -- `__all__` -- the "pre-v5.1.0" framing

*Left in the source:* the whole asymmetry rule, which the v4.16.0 walker symmetry test depends on.

```text
    # ``__all__`` -- pre-v5.1.0 they were importable from
    # ``lumenairy.raytrace`` (via an explicit re-export in
    # ``raytrace/__init__.py``) but were NOT in
    # ``lumenairy.raytrace.__all__`` (the advertised public
    # surface).  Keeping them off this submodule's ``__all__``
```

