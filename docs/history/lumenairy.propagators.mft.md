<!-- lumenairy-history-doc
module: lumenairy/propagators/mft.py
ast_sha256: 0f8c96a6815238955619bd4c543e8e12ffed6b947bff6b1cb3110b3c38bd5b7d
token_sha256: 8cb4155677b9b6fb7f9d30c8fd9aa9f6a7af334b91fb4d40bbe06e5386a40e8b
pre_relocation_lines: 1073
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B3 (audit K6 second half): resample_field gains method='chirpz', the band-limited interpolant evaluated through _bluestein_centred_2d, with the replica guard _warn_mft_output_window taught a per-axis N_out_y; the default 'spline' leg and all three MFT propagators byte-identical
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/mft.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/mft.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

Four blocks moved and they are small; `mft.py`'s long comments are mostly
algebra (the Bluestein expansion, the periodicity of each transform's
reconstruction) and those stayed untouched, as did the K6 resampling-MTF
table and the `_bluestein_separable` cost/accuracy note.

One of the four is worth calling out because it is the duplication pattern
the audit's V6 finding names.  The H-build comment stated the open-interval
band limit **twice** -- "4.12.0: both backends now use `fx < fx_max` ...;
pre-4.12 the NumPy branch used `<=`" near the top, and "4.12.0: strict `<`
band-limit (Matsushima-Shimobaba open interval; matches plain ASM)" fourteen
lines later -- because each audit round appended to the block instead of
editing it.  The source now states the interval once.

Nothing the interpreter executes changed in the move.  The header above
records the SHA-256 of (a) the module's AST with every docstring removed and
source positions ignored, and (b) its `tokenize` stream reduced to
NAME/OP/NUMBER/STRING with comments and docstrings dropped -- both taken from
the file as it stood BEFORE the relocation.
`tests/unit/test_audit2609_a17_history_relocation.py` re-computes both from
the live file on every run.

## Contents

| original line | site | what the block records |
|---|---|---|
| L256-258 | `angular_spectrum_propagate_mft` -- Raises | what a negative or NaN `dx_out` used to return |
| L367-385 | `angular_spectrum_propagate_mft` -- the H build | that the NumPy branch used `<=` before 4.12, one bin off from JAX; the same open-interval statement appeared TWICE in this one comment block |
| L455-458 | `angular_spectrum_propagate_mft` -- the odd-N origin offset | the pre-fix error at N=257 (rel err 1.5e-1, centroid -3.39 px) |
| L548-552 | `resample_field` -- the measured resampling MTF | a comment correcting an earlier COMMENT: that this note claimed '< 0.1 %' loss before v5.46 |

---

### L256-258 -- `angular_spectrum_propagate_mft` -- Raises -- what a negative or NaN `dx_out` used to return

*Left in the source:* what the validator rejects

```text
        ``N_out < 1`` (v5.30, audit P11 -- previously ``dx_out < 0`` was
        accepted and returned a finite field on a silently mirrored grid,
        and ``dx_out = nan`` returned an all-NaN field).
```

### L367-385 -- `angular_spectrum_propagate_mft` -- the H build -- that the NumPy branch used `<=` before 4.12, one bin off from JAX; the same open-interval statement appeared TWICE in this one comment block

*Left in the source:* the live interval, the whole S2-3 host-build rationale (including the ~26 dB float32 phase loss it prevents and what it costs in gradients), and the P12 asymptote note

```text
    # plain ASM.  4.12.0: both backends now use ``fx < fx_max`` (open
    # interval, matching the Matsushima-Shimobaba paper and plain ASM);
    # pre-4.12 the NumPy branch used `<=` (one-bin off from JAX).
    # v5.24.4 (audit S2-3): H is FIELD-INDEPENDENT (input geometry,
    # wavelength and z only), so build it on the HOST in float64 and
    # cache it, then move it onto the active backend.  Building it under
    # the JAX tracer instead silently evaluated the kernel argument
    # ``kz * z`` (up to ~1e6 rad) in float32 whenever ``jax_enable_x64``
    # is off (the JAX default) -- ``jnp.arange(dtype=float64)`` truncates
    # to float32 there -- losing ~26 dB of phase accuracy vs the NumPy
    # contract.  Host-building keeps the field gradient intact (H does
    # not depend on the field); only concrete-float geometry gradients
    # are foregone.  NumPy / CuPy paths are byte-identical to before,
    # and JAX now shares the same cached, f64-built H.  4.12.0: strict
    # `<` band-limit (Matsushima-Shimobaba open interval; matches plain
    # ASM).  v5.30 (audit P12): the cutoff below is the z -> infinity
    # ASYMPTOTE ``L / (2*lambda*|z|)`` of the paper's exact
    # local-frequency limit, not that limit -- strictly larger, so it
    # never over-filters.  See ``fft_infra._get_or_make_bandlimit``.
```

### L455-458 -- `angular_spectrum_propagate_mft` -- the odd-N origin offset -- the pre-fix error at N=257 (rel err 1.5e-1, centroid -3.39 px)

*Left in the source:* the correction itself, the accuracy it achieves, and the even-N no-op guarantee

```text
    # output centre is exact and keeps the declared grid: post-fix
    # ASM-MFT reproduces angular_spectrum_propagate on the same grid to
    # 4.4e-14 at N=257 (pre-fix: rel err 1.5e-1, centroid -3.39 px).
    # ``off_in`` is exactly 0.0 for even N_in -> bit-identical.
```

### L548-552 -- `resample_field` -- the measured resampling MTF -- a comment correcting an earlier COMMENT: that this note claimed '< 0.1 %' loss before v5.46

*Left in the source:* the measured MTF table above it and the quantified loss at the docstring's own 4-pixel case

```text
      The docstring's own case -- "features sampled at >= 4 pixels",
      i.e. 0.25 cycles/pixel -- sits between the 0.20 and 0.30 rows, so
      the loss there is between 0.9 % and 6.8 %, NOT the "< 0.1 %" this
      note claimed before v5.46.  ``dx_out == dx_in`` is the exact
      identity (rel L2 2.5e-16), so all of it is resampling MTF.
```
