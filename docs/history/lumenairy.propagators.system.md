<!-- lumenairy-history-doc
module: lumenairy/propagators/system.py
ast_sha256: 92c78bf52e3a76f6536c9d898254b83f2cc7ed555e1172c4595cb15f6dd7e729
token_sha256: 5f4ba024d2310d6b188ce7e463fbd139334111e795b0bb4ee470185df36334c5
pre_relocation_lines: 1919
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B3b (K6): the 'fresnel' leg evaluates fresnel_propagate_mft straight onto the chain grid instead of propagating to the single-FFT natural grid and resampling back (the crop and the interpolator MTF both go), and the 'sas' leg's surviving resample_field gates method= on whether the chain window fits inside one chirp-Z reconstruction period
re_recorded: 2026-09-13 -- VERIFY-B3b (K6): the 'fresnel' leg carries _warn_system_fresnel_window -- retiring the resample retired _warn_system_resample_crop with it, and fresnel_propagate_mft's faithful-zone warning is a disjoint condition on the chain grid (it reduces to z < N dx^2/lambda, the K1 band), so a beam that outgrows the chain window above that bound had no diagnostic at all
-->

# Version history -- `lumenairy/propagators/system.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/system.py`.  Each block is reproduced **verbatim**
under the source line it came from in the pre-relocation file.

Eight blocks moved and every one is the same shape: a live guard whose comment
explained itself by narrating the silent fall-through it closed.  The guards
and their measurements stayed -- "measured 0.0 relative difference against
`method='asm'`" is what tells a future editor that the fall-through really was
silent, and it is the reason the validation cannot be relaxed -- while the
`Pre-v5.30 ...` framing is here.

Two things deliberately did NOT move:

* the **W9-12 `ray_subsample` derivation**, which sets a live default against
  the first instinct to align it with its two siblings, and carries both arms
  of the measurement (Strehl 0.9994 / 0.9993 / 0.9974 at 1 / 4 / 8, and the
  `min_coarse_samples_per_aperture=32` breakage that raising the divisor would
  cause on coarse grids).  `docs/TESTING_STANDARDS.md` S5 requires it.  Only
  its opening -- a note that the element's DOCSTRING used to recommend against
  the code beneath it -- moved.
* the **v5.0 aperture-schema migration** recipe, which a caller on the legacy
  JAX schema still needs.

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
| L229-248 | `_TRACED_ELEMENT_*` -- the traced element's kwarg surface | the measured pre-v5.31 evidence that nine named keys and an outright typo were all bit-identical to omitting them |
| L328-335 | `_TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT` | a comment correcting an earlier COMMENT -- the docstring that recommended 4 while the code beneath it used 1 -- and the note that the docstring was the thing that was wrong |
| L756-763 | `propagate_through_system` -- method validation | the `Pre-v5.30 every unrecognised method fell through` framing |
| L791-795 | `propagate_through_system` -- per-element override | the `took the same silent fall-through` framing |
| L976-985 | `'propagate_tilted'` element | the `Pre-v5.30 a method key here was dropped in complete silence` framing |
| L1715-1722 | `propagate_through_system_jax` -- method validation | the `Pre-v5.30 method was accepted and never read` framing |
| L1774-1792 | `propagate_through_system_jax` -- dtype resolution | two stacked release paragraphs: the v4.13.0 hard-cast to `jnp.complex64` and the v4.16.1 `np.iscomplexobj(np.asarray(E_in))` probe |
| L1807-1811 | `propagate_through_system_jax` -- the cache key | the `Pre-fix the cache key omitted dtype entirely` framing |

---

### L229-248 -- `_TRACED_ELEMENT_*` -- the traced element's kwarg surface -- the measured pre-v5.31 evidence that nine named keys and an outright typo were all bit-identical to omitting them

*Left in the source:* the live rule (everything is forwarded, anything unaccepted RAISES) and the consequence that makes it matter -- the VALIDATED traced configuration was unreachable through this API

```text
# ---------------------------------------------------------------------------
# v5.31 (audit W9-11): the ``'real_lens_traced'`` element's kwarg surface
# ---------------------------------------------------------------------------
# Pre-v5.31 this element handler hard-coded FOUR arguments (``prescription``,
# ``bandlimit``, ``ray_subsample``, ``progress``) and every other key on the
# element dict was DROPPED IN SILENCE.  MEASURED: output bit-identical to
# omitting the key for all nine of ``amplitude_model``,
# ``preserve_input_phase``, ``remap_sampling``, ``fit_radius_beam_factor``,
# ``carrier``, ``on_undersample``, ``n_workers``, ``traced_kwargs`` -- and for
# an outright typo key.  The consequence the W9 audit was asked about: the
# v5.29 + S12 VALIDATED traced configuration (``amplitude_model='ray_density'``
# + ``preserve_input_phase='remap'`` + ``remap_sampling='full'`` +
# ``fit_radius_beam_factor=2.0``, the shipping defaults of
# ``propagate_traced_carrier_chain``) was UNREACHABLE through this chain API,
# and a caller who wrote those keys got the legacy configuration with no
# diagnostic.
#
# The keys are now forwarded, and anything the element does not accept RAISES.
# Silent fall-through on an unrecognised key is the class the v5.30 P6 twin fix
# closed for ``method``; this closes it for the traced element's parameters.
```

### L328-335 -- `_TRACED_ELEMENT_RAY_SUBSAMPLE_DEFAULT` -- a comment correcting an earlier COMMENT -- the docstring that recommended 4 while the code beneath it used 1 -- and the note that the docstring was the thing that was wrong

*Left in the source:* the three-way disagreement itself and BOTH measured arms of the argument for keeping this element's value at 1 (no fidelity to gain; real breakage to pay).  TESTING_STANDARDS S5: a numeric default carries its derivation.

```text
# v5.31 (audit W9-12): one number, three values, for the same physics --
# ``apply_real_lens_traced`` defaults ``ray_subsample=8``,
# ``propagate_traced_carrier_chain`` defaults 4 (its VALIDATED value), and this
# chain element hard-codes 1 while its docstring used to say "default 1; 4 is
# the recommended production value" -- a docstring recommending against the code
# directly beneath it.  The DOCSTRING is what was wrong, and it is fixed; the
# VALUE stays 1, against the first instinct to align it on the chain's 4,
# because the measurement argues the other way:
```

### L756-763 -- `propagate_through_system` -- method validation -- the `Pre-v5.30 every unrecognised method fell through` framing

*Left in the source:* what the validation prevents, the measurement that proves it was silent, and the twin that matches it

```text
    # v5.30 (audit: the NumPy twin of P6, recorded as a new measured
    # finding in AUDIT_ADVERSARIAL_CODEBASE_2026_07_25).  Pre-v5.30 every
    # unrecognised ``method`` -- 'gbd', 'fraunhofer', 'ASM', outright junk
    # -- fell through the ``if/elif`` chain below to the ``else: # Default:
    # ASM`` branch and silently returned the ASM field (measured 0.0
    # relative difference vs method='asm').  Validate at ENTRY against the
    # honoured set; the JAX twin (v5.30, audit P6) already does the same
    # with the same wording.
```

### L791-795 -- `propagate_through_system` -- per-element override -- the `took the same silent fall-through` framing

*Left in the source:* why an entry-only check is not enough

```text
            # v5.30 (NumPy twin of audit P6): the per-element override took
            # the same silent fall-through to ASM as the entry-point kwarg
            # (measured 0.0 rel diff vs method='asm' for
            # ``{'method': 'not_a_method'}``), so it needs the identical
            # validation -- an entry-only check would leave the hole open.
```

### L976-985 -- `'propagate_tilted'` element -- the `Pre-v5.30 a method key here was dropped in complete silence` framing

*Left in the source:* that the handler is ASM-only, the measurement, and the reason it WARNS rather than raises -- which is the decision a future editor would otherwise reopen

```text
            # v5.30 (flagged in 3f22778): this handler is ASM-ONLY -- it goes
            # straight to ``angular_spectrum_propagate_tilted`` and never reads
            # ``elem['method']``.  Pre-v5.30 a ``method`` key here was dropped
            # in complete silence: MEASURED 0.0 relative difference (and
            # bit-identical output) for ``method='fresnel'``, ``'sas'`` AND
            # ``'not_a_method'`` versus omitting the key -- i.e. not even
            # validated, unlike the ``'propagate'`` element which raises on an
            # unrecognised value.  Warn rather than raise: raising would be a
            # new breakage class for a legacy alias that has silently accepted
            # the key for many releases.
```

### L1715-1722 -- `propagate_through_system_jax` -- method validation -- the `Pre-v5.30 method was accepted and never read` framing

*Left in the source:* what this path implements, how it differs from the NumPy twin, and the ``method=None`` contract

```text
    # Pre-v5.30 ``method`` was accepted and never read: every value --
    # including 'fresnel', 'sas' and outright junk -- returned the ASM
    # field (bit-identical), while the NumPy twin
    # ``propagate_through_system`` honours 'fresnel' / 'sas' (and rejects
    # 'rs').  Silent fall-through is the forbidden class, so the JAX path
    # now names what it implements: ASM.  ``method=None`` is accepted as
    # an explicit "use this entry point's default" (it does NOT resolve
    # ``set_default_wave_propagator()`` -- see the docstring).
```

### L1774-1792 -- `propagate_through_system_jax` -- dtype resolution -- two stacked release paragraphs: the v4.13.0 hard-cast to `jnp.complex64` and the v4.16.1 `np.iscomplexobj(np.asarray(E_in))` probe

*Left in the source:* the live resolution rule and the reason the probe must duck-type -- ``np.asarray`` on a tracer raises, and jit is exactly what this entry point is for

```text
    # v4.13.0 (audit L2): pre-fix this hard-cast to ``jnp.complex64``
    # silently overrode ``set_default_complex_dtype(np.complex128)``.
    # Now goes through ``_resolve_jax_complex_dtype`` which reads the
    # library-wide default.  Complex inputs honour their own dtype; only
    # real inputs fall back to the configured default.
    #
    # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 7 / P3 / DEEP-4 MEDIUM-2):
    # the pre-v4.16.1 dtype probe used ``np.iscomplexobj(np.asarray(E_in))``
    # which raises ``jax.errors.TracerArrayConversionError`` when this
    # function is wrapped in ``jax.jit`` or ``jax.grad`` -- but this
    # function is explicitly named ``propagate_through_system_jax`` and
    # the docstring touts end-to-end jit'd caching.  The fix mirrors
    # the v4.15.3 ``_check_2d_scalar_field`` JAX-safe ``getattr(E,
    # 'attr', None)`` idiom: read the dtype via duck-typing (works for
    # NumPy arrays, JAX tracers, CuPy arrays, and the typed scalars
    # that the JAX cast accepts).  When the dtype attribute is missing
    # or non-complex, fall back to the library-default complex dtype
    # for the cast -- the real-input branch is exercised by tests that
    # build a real Gaussian envelope and rely on the autopromote.
```

### L1807-1811 -- `propagate_through_system_jax` -- the cache key -- the `Pre-fix the cache key omitted dtype entirely` framing

*Left in the source:* why dtype is in the key and what goes wrong without it

```text
        # v4.13.0 (audit L2): include dtype in the cache key so calls
        # at complex64 and complex128 don't share the same compiled XLA
        # kernel.  Pre-fix the cache key omitted dtype entirely, so the
        # first call to win the race fixed the kernel precision for
        # every subsequent call regardless of the active default.
```
