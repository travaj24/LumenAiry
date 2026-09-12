<!-- lumenairy-history-doc
module: lumenairy/optimize/wrapper_merits.py
ast_sha256: c09c446059138b4520ae70f1820fe4ef535150c2cf5c4854780408409dc5daff
token_sha256: 69b4b870ccae2a65b1bd1e7e977691b1cb88f1394f3357fac8c1a20a7c2ca278
pre_relocation_lines: 1048
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/optimize/wrapper_merits.py`

This file holds the version-history narrative that used to live in
`lumenairy/optimize/wrapper_merits.py` -- the "vX.Y (audit Z): pre-fix this did
A, which was wrong because B, now it does C" blocks.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file.

Three threads of history run through this module and they are the bulk of what
is recorded here:

* **the `_ZERO_APERTURE_MASK` semantics flip** (`aperture_diameter = 0`
  collapsing to `None` and so becoming a full-grid plane wave) -- noted at
  three separate sites, each naming the release that introduced the flip and
  the one that undid it.  The *rule* (the three branches must stay distinct,
  and why) is live and stayed in the source; the release pair moved.
* **the `MultiWavelengthMerit` SUM -> AVG change** -- noted at four sites.  The
  one-cycle `FutureWarning` it installed is a **live migration aid**, so the
  warning's whole rationale (a user's weight calibration sees a 3x drop on a
  3-wavelength configuration) stayed in the source next to the latch; only the
  release attribution moved.
* **the aperture-free grid cache** -- the measured retention figures
  (32 x 57 B/px = 7.6 GB at N = 2048, ~20x lower after) are the sizing
  derivation for `_WRAPPER_MERIT_GRID_CACHE_SIZE`, which
  `docs/TESTING_STANDARDS.md` S5 asks a bar to carry, so they stayed --
  re-stated as what duplicating the arrays WOULD cost rather than as what a
  past release did cost.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

**Line-citation note.**  `CHANGELOG.md` cites
`optimize/wrapper_merits.py:987` for the `_ZERO_APERTURE_MASK` sentinel
branch, and `tests/unit/test_v4_15_agent_f.py` re-derives that line by
anchor string and requires the citation to sit within +/- 5 of it.  This
relocation moves the anchor, so that citation needs refreshing; the WP report
gives the exact new line.

## Contents

| original line | site | what the block records |
|---|---|---|
| L4-5 | `<module> docstring` | the release and agent that split this module out of core.py |
| L17-22 | `<module> docstring -- cache layout` | the release/audit tag on the grid-cache split |
| L26-35 | `<module> docstring -- lookup contract` | the pre-v5.1.0 import shape, the "after the v5.1.0 split" framing, and "preserves the v4.15.3 mock.patch contract bit-for-bit" |
| L54-58 | `_merit_jit import` | the release / roadmap tag and "matches the pre-v5.3 path bit-for-bit" |
| L66-66 | `<module> meshgrid-cache section header` | the release tag on the section header |
| L94-104 | `_WRAPPER_MERIT_GRID_CACHE` | the release/audit tag and the "pre-v5.17.1 they were duplicated" framing |
| L118-118 | `_WRAPPER_MERIT_CACHE_LOCK alias` | the release tag on the dispatcher-pin convention |
| L122-122 | `_WRAPPER_MERIT_CACHE_LOCK` | the release and audit tag |
| L128-128 | `_WRAPPER_MERIT_GRID_CACHE_LOCK` | the release and audit tag |
| L191-198 | `_get_wrapper_merit_cache` | the release/audit tag and "values are byte-identical to the pre-split build" |
| L287-295 | `_get_wrapper_merit_cache` | the release tags on both sides of the zero-aperture semantics flip |
| L298-301 | `_get_wrapper_merit_cache` | the release/audit tag and the "same payload keys as pre-split" framing |
| L312-312 | `_get_wrapper_merit_cache` | the release and audit tag |
| L323-331 | `_clear_wrapper_merit_cache` | the release tags and the lazy-import-then-registry rollout narrative |
| L336-336 | `_clear_wrapper_merit_cache` | the release and audit tag |
| L343-345 | `<module> registry enrollment` | the release / roadmap tag and "now walks the registry" |
| L364-374 | `_MULTIWL_AVG_WARNED` | the release that introduced the transition and the release that added the latch |
| L384-389 | `MultiWavelengthMerit docstring` | "v4.16.1 closes the SUM-vs-AVG discrepancy ... pre-v4.16.1 this class summed" |
| L463-464 | `MultiWavelengthMerit.evaluate` | the release and audit tag |
| L480-480 | `MultiWavelengthMerit.evaluate` | the release and audit tag |
| L565-565 | `MultiWavelengthMerit.evaluate` | the release and audit tag |
| L579-598 | `MultiWavelengthMerit.evaluate` | the two release/audit tags and the "Pre-v4.16.1 this method silently summed" narrative |
| L599-600 | `MultiWavelengthMerit.evaluate` | the "v5.1.0 split (Agent E)" tag and "historical reset-fixtures" |
| L676-676 | `MultiFieldMerit.__init__` | the release and audit tag |
| L735-764 | `MultiFieldMerit.evaluate` | four stacked release tags (pre-4.10 clipping, v4.13.2 X-term, 4.11.1 precision knob, v4.14.1 three branches, v5.3 JIT) and their pre-fix clauses |
| L776-776 | `MultiFieldMerit.evaluate` | the release and audit tag |
| L941-951 | `ToleranceAwareMerit.evaluate` | the release tags and the note that a later release deleted the defunct `_PerturbedABCDFallbackSentinel` class |
| L968-968 | `ToleranceAwareMerit.evaluate` | the release and audit tag |
| L990-990 | `ToleranceAwareMerit.evaluate` | the release and audit tag |
| L1023-1023 | `ToleranceAwareMerit.evaluate` | the release and audit tag |
| L1030-1032 | `ToleranceAwareMerit.evaluate` | the "pre-fix they degenerated to inf / silently-inert" framing |

---

### L4-5 -- `<module> docstring` -- the release and agent that split this module out of core.py

*Left in the source:* the split relationship itself, which is live: core.py re-exports these names.

```text
v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
Hosts the three "wrapper" merit terms that sweep an inner sub-merit
```

### L17-22 -- `<module> docstring -- cache layout` -- the release/audit tag on the grid-cache split

*Left in the source:* the whole layout contract: which arrays are aperture-independent, the two keys, and the sharing-by-reference property that makes an aperture FD sweep cheap.

```text
tilt-phase construction.  v5.17.1 (audit P2-25): the aperture-
INDEPENDENT arrays live in the sibling ``_WRAPPER_MERIT_GRID_CACHE``
(keyed on ``(Ny, Nx, dx, dtype_str)`` only) and are shared by
reference across per-aperture entries, so a free ``aperture_diameter``
FD sweep no longer duplicates six N x N arrays per perturbed value.
See ``_get_wrapper_merit_cache`` for the detailed contract.
```

### L26-35 -- `<module> docstring -- lookup contract` -- the pre-v5.1.0 import shape, the "after the v5.1.0 split" framing, and "preserves the v4.15.3 mock.patch contract bit-for-bit"

*Left in the source:* the entire contract, re-stated as a rule: names are read through `optimize.core` by lazy module-attribute lookup because that is the binding `mock.patch` rebinds.  Losing this is how the patches would stop reaching these bodies.

```text
Pre-v5.1.0, every name used in the wrapper-merit bodies lived in
``lumenairy/optimize/core.py`` and was imported at the top of that
file via ``from ..raytrace import system_abcd`` etc.  Tests that
monkey-patch ``lumenairy.optimize.core.system_abcd`` (using
:func:`unittest.mock.patch`) target THAT binding -- not the original
``lumenairy.raytrace.system_abcd``.  After the v5.1.0 split, the
wrapper-merit class bodies still need to honour those patches, so
each call site reads the function via :mod:`lumenairy.optimize.core`
(``_core.system_abcd(...)``) via a lazy module-attribute lookup.  This
preserves the v4.15.3 mock.patch test contract bit-for-bit.
```

### L54-58 -- `_merit_jit import` -- the release / roadmap tag and "matches the pre-v5.3 path bit-for-bit"

*Left in the source:* what the kernel does and the bit-for-bit NumPy fallback guarantee, which is the property a caller relies on.

```text
# v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT): fused Numba
# kernel that builds the masked tilted plane wave in one parallel
# pass; falls back to a pure-NumPy implementation that matches the
# pre-v5.3 path bit-for-bit when Numba is unavailable.  See
# ``lumenairy/optimize/_merit_jit.py`` for the contract.
```

### L66-66 -- `<module> meshgrid-cache section header` -- the release tag on the section header

*Left in the source:* the header and the whole per-leg rebuild-cost argument under it.

```text
)
```

### L94-104 -- `_WRAPPER_MERIT_GRID_CACHE` -- the release/audit tag and the "pre-v5.17.1 they were duplicated" framing

*Left in the source:* the entire sizing argument, re-stated as what duplication WOULD cost -- 32 x 57 B/px = 7.6 GB at N = 2048 against a 1 B/px mask, ~20x lower retention now.  TESTING_STANDARDS S5 keeps a bar's derivation with the bar.

```text
# v5.17.1 (audit P2-25): aperture-FREE sibling cache.  Six of the seven
# payload arrays (X, Y, Y_factor, X_factor, r_squared, E_ones) depend
# only on (N, dx, dtype), yet pre-v5.17.1 they were duplicated into
# every per-aperture ``_WRAPPER_MERIT_CACHE`` entry.  With
# ``aperture_diameter`` as a free optimisation variable every FD
# perturbation minted a distinct key, so the 32-slot LRU retained up to
# 32 full 57 B/px payloads (7.6 GB at N=2048) of which everything but
# the 1 B/px mask was identical.  The grid arrays now live HERE, keyed
# on ``(Ny, Nx, dx, dtype_str)`` only, and the per-aperture entries
# hold references to the shared arrays plus their own boolean mask --
# worst-case retention drops ~20x (one 56 B/px grid set + 32 masks).
```

### L118-118 -- `_WRAPPER_MERIT_CACHE_LOCK alias` -- the release tag on the dispatcher-pin convention

*Left in the source:* the convention and the fact that the alias is the SAME lock object, which is the part that stops a second lock being introduced.

```text
# lock-order bugs).  The alias satisfies the v4.14.2 cache<->lock
```

### L122-122 -- `_WRAPPER_MERIT_CACHE_LOCK` -- the release and audit tag

*Left in the source:* what the lock covers, the precedent it follows, and the torn-OrderedDict failure it prevents.

```text
# v4.14.1 (P2-1): guard concurrent get / move_to_end / __setitem__ /
```

### L128-128 -- `_WRAPPER_MERIT_GRID_CACHE_LOCK` -- the release and audit tag

*Left in the source:* that it is the same lock object and the pointer to the note that says why.

```text
# v5.17.1 (audit P2-25): companion-name alias for the grid sibling --
```

### L191-198 -- `_get_wrapper_merit_cache` -- the release/audit tag and "values are byte-identical to the pre-split build"

*Left in the source:* the whole sharing contract and the aperture-sweep consequence, which is what a caller reasoning about rebuild cost needs.

```text
    v5.17.1 (audit P2-25): the six aperture-independent arrays are
    memoised separately in ``_WRAPPER_MERIT_GRID_CACHE`` keyed on
    ``(Ny, Nx, dx, dtype_str)`` and SHARED by reference across every
    per-aperture entry, so an aperture-sweeping run (e.g.
    ``aperture_diameter`` as a free FD variable) rebuilds only the
    cheap boolean mask, and the LRU retains one grid set instead of
    32 duplicates.  Values are byte-identical to the pre-split build
    (same construction, same float64 comparison for the mask).
```

### L287-295 -- `_get_wrapper_merit_cache` -- the release tags on both sides of the zero-aperture semantics flip

*Left in the source:* the rule and the full reason the three branches must stay distinct -- collapsing zero to None turns "block all light" into "full plane wave".

```text
            # v4.14.1 (P1-NEW-1): aperture explicitly <= 0 means
            # "block all light."  Distinct from the ``None`` branch
            # above (which means "no aperture specified, use full
            # grid").  Pre-v4.14.0 a scalar 0 produced an all-False
            # boolean mask; v4.14.0 erroneously collapsed it to None
            # and the downstream callers then treated the deliberate
            # zero as "no aperture -> full plane wave," flipping the
            # semantics.  Use a sentinel so callers can detect this
            # case via ``is`` and zero their fields explicitly.
```

### L298-301 -- `_get_wrapper_merit_cache` -- the release/audit tag and the "same payload keys as pre-split" framing

*Left in the source:* the ownership rule (shared references vs the entry's own mask) and the unchanged-keys guarantee the identity pins rely on.

```text
    # v5.17.1 (audit P2-25): the six aperture-independent arrays are
    # REFERENCES into the shared grid-cache payload; only ``mask`` is
    # owned by this entry.  Same payload keys as pre-split, so callers
    # and the eval-count / identity pins are unaffected.
```

### L312-312 -- `_get_wrapper_merit_cache` -- the release and audit tag

*Left in the source:* the reason the increment is inside the lock.

```text
        # v5.4 (audit P3): increment inside lock to preserve cache-build-counter invariant
```

### L323-331 -- `_clear_wrapper_merit_cache` -- the release tags and the lazy-import-then-registry rollout narrative

*Left in the source:* who calls it, through what, and the reverse-direction-dependency argument that keeps optimize/core free of import-time propagation side-effects.

```text
    v4.14.1 (P2-3): invoked from
    :func:`lumenairy.propagators.propagation.clear_asm_caches`.  Pre-
    v4.16 this was a lazy import inside ``clear_asm_caches``; v4.16
    routes the call through the central cache-clearer registry (see
    ``_cache_registry.py``).  Either way the reverse-direction
    dependency keeps optimize/core free of propagation-layer side-
    effects at import time while still leaving both caches pristine
    on a single ``clear_asm_caches()`` call.  Also callable directly
    from tests.
```

### L336-336 -- `_clear_wrapper_merit_cache` -- the release and audit tag

*Left in the source:* why one registered clearer covers both caches.

```text
        # v5.17.1 (audit P2-25): drain the aperture-free grid sibling
```

### L343-345 -- `<module> registry enrollment` -- the release / roadmap tag and "now walks the registry"

*Left in the source:* what the enrollment does and the late-binding-closure note that preserves mock.patch.object semantics.

```text
# v4.16.0 (ROADMAP #15): register the wrapper-merit clearer with the
# central registry at module-import time.  ``clear_asm_caches`` now
# walks the registry rather than enumerating clear calls by hand.
```

### L364-374 -- `_MULTIWL_AVG_WARNED` -- the release that introduced the transition and the release that added the latch

*Left in the source:* the entire migration rationale -- it is a LIVE user-facing warning, so why it fires, what a user should do about it, and why it is latched all stayed.

```text
# v4.16.2 (audit P1-NEW-F1-3): one-cycle FutureWarning latch for the
# MultiWavelengthMerit SUM->AVG transition introduced in v4.16.1.  The
# new AVG semantics are CORRECT (match the docstring and the sibling
# MultiFieldMerit / ToleranceAwareMerit classes which already divide by
# their loop length), but existing user weight-calibrations tuned
# against the pre-v4.16.1 SUM behaviour silently see a 3x drop on a
# 3-wavelength configuration.  Emit a one-shot FutureWarning the first
# time any MultiWavelengthMerit.evaluate runs with >1 wavelength so
# users notice the change and can re-scale weights if needed.  Latched
# at module level so optimisation loops (which call evaluate() many
# times per process) don't flood the warning channel.
```

### L384-389 -- `MultiWavelengthMerit docstring` -- "v4.16.1 closes the SUM-vs-AVG discrepancy ... pre-v4.16.1 this class summed"

*Left in the source:* the averaging contract and the consequence of summing instead, stated as a property rather than as a release note.

```text
    n_wavelengths).  v4.16.1 closes the SUM-vs-AVG discrepancy at
    the return: pre-v4.16.1 this class summed rather than averaged,
    silently tripling the chromatic merit contribution for a
    3-wavelength configuration vs a 1-wavelength one.  Matches the
    sibling :class:`MultiFieldMerit` and :class:`ToleranceAwareMerit`
    averaging shape.
```

### L463-464 -- `MultiWavelengthMerit.evaluate` -- the release and audit tag

*Left in the source:* the sentinel's purpose, the `is`-identity contract and the float() magnitude fallback.

```text
                # branch downstream.  v4.15.3 (P1-NEW-F1-3): wire the
                # ``_INVALID_FL_SENTINEL_OBJ`` singleton so a
```

### L480-480 -- `MultiWavelengthMerit.evaluate` -- the release and audit tag

*Left in the source:* the guard and why the identity test is needed before np.isfinite / abs.

```text
            # v4.15.3 (P1-NEW-F1-3): the sentinel form of ``bfl``
```

### L565-565 -- `MultiWavelengthMerit.evaluate` -- the release and audit tag

*Left in the source:* why ctx.x is threaded -- analytic gradients instead of an FD fallback.

```text
            # v4.13.2 (C-P1-2): thread ctx.x so JaxMeritTerm sub-
```

### L579-598 -- `MultiWavelengthMerit.evaluate` -- the two release/audit tags and the "Pre-v4.16.1 this method silently summed" narrative

*Left in the source:* the averaging rule with its sibling-class justification, the 3x consequence, and the whole FutureWarning latch rationale -- all as present-tense properties.

```text
        # v4.16.1 (AUDIT_V4_16_0_DEEP P1-DEEP-1-1): SUM -> AVG.
        # The docstring documents this class as "average" of the
        # sub-merit across wavelengths, and BOTH sibling classes
        # ``MultiFieldMerit`` and ``ToleranceAwareMerit`` divide by
        # ``len(...)`` at their return.  Pre-v4.16.1 this method
        # silently summed the per-wavelength contributions, so adding
        # a 3rd wavelength tripled the chromatic merit's weight
        # contribution relative to a 1-wavelength configuration.
        # Fix: divide by ``max(len(self.wavelengths), 1)`` to match
        # the documented behaviour and sibling-class averaging shape.
        #
        # v4.16.2 (audit P1-NEW-F1-3): emit a one-cycle FutureWarning
        # the first time evaluate() runs with >1 wavelength so users
        # tuning weight calibrations against the pre-v4.16.1 SUM
        # behaviour notice the silent 1/N drop in the merit
        # contribution.  Latched via the module-level
        # ``_MULTIWL_AVG_WARNED`` flag so optimisation loops don't
        # flood the warning channel.  Single-wavelength configurations
        # (len == 1) are unaffected by the SUM->AVG transition and
        # don't trigger the warning.
```

### L599-600 -- `MultiWavelengthMerit.evaluate` -- the "v5.1.0 split (Agent E)" tag and "historical reset-fixtures"

*Left in the source:* the dual-binding rule in full: read the alias first, write both on emission.

```text
        # v5.1.0 split (Agent E): the canonical latch lives on this
        # module, but historical reset-fixtures toggle
```

### L676-676 -- `MultiFieldMerit.__init__` -- the release and audit tag

*Left in the source:* the accepted input forms and the per-entry detection.

```text
        # v4.13.2 (C-P0-2): accept EITHER scalars (back-compat:
```

### L735-764 -- `MultiFieldMerit.evaluate` -- four stacked release tags (pre-4.10 clipping, v4.13.2 X-term, 4.11.1 precision knob, v4.14.1 three branches, v5.3 JIT) and their pre-fix clauses

*Left in the source:* every rule the stack encodes, re-stated in the present tense and separated into four paragraphs: why the plane wave is clipped, that the tilt is generic and honours the precision knob, why the three aperture branches must stay distinct, and the full Numba-kernel contract with its temporaries argument and fallback conditions.

```text
            # acceptance.  Pre-4.10 the unclipped grid-filling plane
            # wave fed every grid pixel through apply_real_lens, then
            # Strehl was computed against a "grid-filling" reference
            # which artificially lowered the value and biased the
            # optimizer toward apertures larger than designed.
            # v4.13.2 (C-P0-2): generic off-axis tilt with both X and
            # Y components.  Pre-fix the X term was silently dropped.
            # 4.11.1: honour precision knob (was hard-coded complex128
            # which silently demoted precision='single' configs).
            # v4.14.1 (P1-NEW-1): three branches -- None means "no
            # aperture specified, full grid"; _ZERO_APERTURE_MASK
            # means "aperture explicitly zero, block all light";
            # ndarray means "circular boolean mask."  Pre-v4.14.0 the
            # zero-diameter case was an all-False ndarray (correctly
            # zeroing the field); v4.14.0 collapsed it into the None
            # branch (full-grid plane wave), flipping the semantics.
            # v5.3 (ROADMAP v5.3 horizon -- MultiFieldMerit JIT):
            # the boolean-mask branch (the dominant hot path -- a
            # finite ``aperture_diameter`` is set on essentially
            # every real prescription) is now routed through the
            # fused Numba kernel in ``_merit_jit.py``.  The kernel
            # collapses ``sin(tx)*k_X + sin(ty)*k_Y -> exp(1j*phase)
            # -> where(mask, ..., 0)`` -- three N x N temporaries
            # per field in the legacy NumPy path -- into a single
            # parallel pass with zero temporaries.  The other two
            # branches (None / _ZERO_APERTURE_MASK) are already
            # NumPy-cheap and stay on the legacy path.  When Numba
            # is unavailable OR the grid is below the kernel-
            # overhead threshold, the helper falls back to the
            # legacy NumPy expression -- callers see no API change.
```

### L776-776 -- `MultiFieldMerit.evaluate` -- the release and audit tag

*Left in the source:* why ctx.x is threaded.

```text
            # Build sub-context.  v4.13.2 (C-P1-2): thread ctx.x so
```

### L941-951 -- `ToleranceAwareMerit.evaluate` -- the release tags and the note that a later release deleted the defunct `_PerturbedABCDFallbackSentinel` class

*Left in the source:* the live reason this branch is deliberately NOT wired to a sentinel singleton: the fallback is a two-float tuple and the consumer expects two floats.

```text
                # but is a stable sentinel.  v4.15.3 (P1-NEW-F1-3):
                # this branch is INTENTIONALLY left unwired.  The
                # fallback is a tuple-pattern ``(efl_p, bfl_p)`` not
                # a single scalar, and wrapping a tuple in a single
                # sentinel singleton would break downstream
                # ``sub_ctx.efl=efl_p``/``sub_ctx.bfl=bfl_p`` usage
                # (the consumer expects two floats, not a tuple).
                # v4.15.4 (AUDIT_V4_15_3 P2-NEW-F1-B option a) deleted
                # the defunct ``_PerturbedABCDFallbackSentinel`` class
                # outright; see the audit closure block in the v4.15.4
                # release notes.
```

### L968-968 -- `ToleranceAwareMerit.evaluate` -- the release and audit tag

*Left in the source:* the whole aperture-preservation argument and the note that apply_perturbations does not re-validate.

```text
            # v4.14.2 (P1-NEW-1): the perturbed prescription preserves
```

### L990-990 -- `ToleranceAwareMerit.evaluate` -- the release and audit tag

*Left in the source:* why ctx.x is threaded.

```text
            # v4.13.2 (C-P1-2): thread ctx.x so JaxMeritTerm sub-
```

### L1023-1023 -- `ToleranceAwareMerit.evaluate` -- the release and audit tag

*Left in the source:* the sentinel and the float()-coercion contract at the consumer.

```text
                    # v4.15.3 (P1-NEW-F1-3): wire the
```

### L1030-1032 -- `ToleranceAwareMerit.evaluate` -- the "pre-fix they degenerated to inf / silently-inert" framing

*Left in the source:* the same fact as a live consequence of not building the OPD map, including the contrast with MultiField / MultiWavelength.

```text
            # ZernikeCoefficient) see real data instead of ``None`` -- pre-fix
            # they degenerated to inf / silently-inert under this wrapper even
            # though they optimise fine under MultiField / MultiWavelength.
```

