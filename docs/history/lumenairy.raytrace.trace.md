<!-- lumenairy-history-doc
module: lumenairy/raytrace/trace.py
ast_sha256: 60b057d51a6f7efa2640f0f9fd09addcdd42a66ea4d16d5fa69c744259c38b1d
token_sha256: bd4f86c56973473300b359937a0f7e6de771963f612a61365a8947647b9fdaa4
pre_relocation_lines: 1902
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B9 items 1, 2, 5: trace() gains renormalize= and sphere_normal= (defaults unchanged); make_rings gains pattern='vogel' area-uniform sampling with the 'rings' default unchanged; ray_pattern='vogel' threaded through trace_prescription / raytrace_system
re_recorded: 2026-09-13 -- WP-B9 items 1, 2, 5: trace() gains renormalize= and sphere_normal= (defaults unchanged); make_rings gains pattern='vogel' area-uniform sampling with the 'rings' default unchanged; ray_pattern='vogel' threaded through trace_prescription / raytrace_system
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-20 -- WP-C2 (5.49.0): trace() defaults to sphere_normal='analytic' -- the closed-form pure-sphere normal; 'generic' is the byte-identical way back
re_recorded: 2026-09-20 -- WP-C2 (5.49.0): trace() defaults to renormalize='exit' -- one exit-plane rescale instead of one per surface; 'surface' is the byte-identical way back, and the history-drift bound is restated as n_surfaces * eps
re_recorded: 2026-09-20 -- WP-C2 round 2: trace_prescription and raytrace_system take the two way-back keywords (D4); the new _way_back_kwargs and _library_trace_default helpers (D4, D5); the sphere_normal docstring restated to the exact-input oracle (D1) and the renormalize docstring to the n*eps bound with the seventh-surface crossing (D3)
-->


# Version history -- `lumenairy/raytrace/trace.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/trace.py`.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

Nine blocks.  The grating `1 / n2` factor accounts for three of them: the
physics (tangential-wavevector conservation) and the MEASURED magnitude of
omitting it (0.26200000 vs 0.17425045 into N-BK7 -- a 50 % direction error)
stayed in the source at every site, because `n_medium=1.0` still defaults to
the air-correct behaviour and a caller has to know when to pass the glass
index.  What moved is the "pre-fix all four sites" framing.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L21-22 | `<module> docstring` | the split-provenance claim |
| L207-210 | `trace DOE branch` | "Pre-fix this site divided unguarded" |
| L219-221 | `trace DOE branch` | "Pre-fix all four sites applied" |
| L600-603 | `surfaces_from_prescription` | the "Pre-v4.15.1" framing |
| L1260-1265 | `apply_doe_phase_traced` | "reproduces the pre-fix behaviour exactly" and "where the pre-fix kick was high" |
| L1280-1284 | `apply_doe_phase_traced notes` | a note correcting an EARLIER DOCSTRING's claim about this same function |
| L1342-1346 | `apply_doe_phase_traced` | the "Pre-v5.2 ... always returned a positive N_new" framing |
| L1652-1655 | `_register_fixed_index` | the past-tense framing of the id()-derived name |
| L1870-1873 | `raytrace_system` | the release/audit tag on the in-place-write prohibition |

---

### L21-22 -- `<module> docstring` -- the split-provenance claim

*Left in the source:* the bit-for-bit statement.

```text
No physics change: contents are bit-for-bit copies of the original
implementations.
```

### L207-210 -- `trace DOE branch` -- "Pre-fix this site divided unguarded"

*Left in the source:* the guard's contract, the JAX-twin parity reference and both measured failure modes.

```text
            # ``period`` is non-finite or zero").  Pre-fix this site
            # divided unguarded: ``period=0.0`` raised
            # ``ZeroDivisionError`` mid-trace and ``period=nan`` silently
            # NaN-poisoned (L, M) (measured: numpy (nan, nan) vs jax
```

### L219-221 -- `trace DOE branch` -- "Pre-fix all four sites applied"

*Left in the source:* the physics, the error mode and the measured figures below it.

```text
            # Pre-fix all four sites applied ``m lambda / Lambda`` directly
            # to (L, M) AFTER refracting into ``glass_after``: exact in air
            # (n2 = 1) but high by exactly n2 at any interface into glass.
```

### L600-603 -- `surfaces_from_prescription` -- the "Pre-v4.15.1" framing

*Left in the source:* the gather rule and the silent-no-op it prevents.

```text
            # dataclass.  Pre-v4.15.1 the dispatcher routed the
            # freeform_type correctly but the coefficient list was
            # silently dropped, making Forbes Q a no-op on flat-keys
            # prescriptions (the unified-dict shape worked).
```

### L1260-1265 -- `apply_doe_phase_traced` -- "reproduces the pre-fix behaviour exactly" and "where the pre-fix kick was high"

*Left in the source:* the parameter's meaning, its default, when to override it, and the measured 50 % error.

```text
        reproduces the pre-fix behaviour exactly and is correct for a
        grating in air; pass the glass index for a grating on the glass
        side of an interface, where the pre-fix kick was high by exactly
        ``n2`` (measured 0.26200000 vs 0.17425045 into N-BK7 at
        Lambda = 5 um, lambda = 1.31 um, m = 1 -- a 50 % direction
        error).  The in-trace kicks (``trace`` / ``trace_world``'s
```

### L1280-1284 -- `apply_doe_phase_traced notes` -- a note correcting an EARLIER DOCSTRING's claim about this same function

*Left in the source:* the corrected statement: the direction-cosine form is exact in plane, and the one approximation is the 1/n2 factor.

```text
    diffraction this form is EXACT -- the pre-R5 docstring's claim that
    it "neglects the cosine factor that distinguishes ``sin`` from the
    direction cosine" was itself inaccurate; the real approximation the
    function made was the missing ``1 / n2``, now exposed as
    ``n_medium``.  The remaining idealisation is the thin-screen model:
```

### L1342-1346 -- `apply_doe_phase_traced` -- the "Pre-v5.2 ... always returned a positive N_new" framing

*Left in the source:* the sign rule and the sibling it must match.

```text
    # z is unchanged.  Pre-v5.2 ``apply_doe_phase_traced`` always
    # returned a positive ``N_new`` while the inline DOE kick in
    # :func:`trace` correctly preserved the sign (see line ~193:
    # ``r.N = np.where(r.N < 0, -_N_new, _N_new)``).  Match the inline
    # site so reverse-traced bundles (``N < 0``) keep their direction.
```

### L1652-1655 -- `_register_fixed_index` -- the past-tense framing of the id()-derived name

*Left in the source:* the id-recycling hazard and the idempotence argument for content-derived names.

```text
            # registration retargeted previously built surface lists
            # to the wrong index (trace() resolves glass at trace
            # time).  Content-derived names are idempotent (same
            # content -> same name -> bounded registry growth) and
```

### L1870-1873 -- `raytrace_system` -- the release/audit tag on the in-place-write prohibition

*Left in the source:* the prohibition and both caller shapes it protects.

```text
    # from in v4.13.2 (audit P1-NEW-J).  A caller that passes a
    # hand-built element list containing a shared ``Surface`` -- or that
    # calls ``raytrace_system`` twice on the same converted list --
    # otherwise sees its prescription silently re-thicknessed.
```

