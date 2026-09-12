<!-- lumenairy-history-doc
module: lumenairy/raytrace/jax_trace.py
ast_sha256: bd850b7885d2bc25659d943f39b681b6df95e0305269e1b5cde5d6ac33f5cb8c
token_sha256: 26316495716c3092b0f9729fbe89ffae55e776c54b29f10da642a27bdc920924
pre_relocation_lines: 1821
recorded_by: WP-A17 SWEEP-3 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->


# Version history -- `lumenairy/raytrace/jax_trace.py`

This file holds the version-history narrative that used to live in
`lumenairy/raytrace/jax_trace.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

This module is the JAX twin of the NumPy tracer, so most of its comments are
PARITY arguments -- "the NumPy path does X, so this kernel must do X too, and
here is the divergence that follows if it does not".  Those arguments are live
and stayed; what moved is the release framing around them, and the clauses that
name the specific past divergence (`pre-fix it stopped at 8`, `used to be
silently accepted`, `the old R_finite > 0 selector`) rather than the standing
rule.  In every case the same hazard is now stated as what WOULD happen, which
is what a future editor tempted to "simplify" the kernel needs to read.

Three measurements stayed in the source because they are the evidence for a
live decision, not a record of a past one:

* the RT-6 no-warning block's parity measurement (NA 0.64, ~2.4e-9 m, invariant
  across a 100x gap sweep) -- it is the argument for NOT re-adding the warning;
* the R5 grating `1 / n2` measurement (ratio 1.503583 == n(N-BK7), a 50 %
  direction error) -- the magnitude of the error the factor prevents;
* the `_ASPHERIC_NEWTON_ITERS = 10` alignment argument, including the float32
  alive-mask divergence a lower count produces.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.

## Contents

| original line | site | what the block records |
|---|---|---|
| L51-64 | `<module> RT-6 note` | "has been RETIRED" / "the old 0.95 gate" / "removed outright" |
| L73-77 | `_ASPHERIC_NEWTON_ITERS` | "Pre-fix it stopped at 8" |
| L84-84 | `_ASPHERIC_NEWTON_RESIDUAL_TOL` | the release and audit tag |
| L88-92 | `_ASPHERIC_NEWTON_RESIDUAL_TOL` | "an unconverged finite t used to be accepted silently" |
| L296-296 | `_intersect_jax` | the release and audit tag |
| L329-333 | `_intersect_jax` | the release/audit tags on both the JAX fix and the NumPy fix it mirrors |
| L358-361 | `_intersect_jax` | the release/audit tag and "used to be silently accepted" |
| L494-503 | `_apply_grating_kick_jax` | "Pre-fix all four sites in the library ... omitted it" and "reproduces the pre-fix (air-correct) behaviour" |
| L512-517 | `_apply_grating_kick_jax` | the release tags around the traced-period support |
| L564-564 | `_apply_grating_kick_jax` | the release and audit tag |
| L631-635 | `_transfer_jax` | "This path used to advance EVERY ray's position and OPL unconditionally" |
| L805-810 | `_reject_unsupported_jax_surfaces` | the release/audit tag and "previously it silently traced ... reproducing the pre-4.10 wrong-answer mode" |
| L880-884 | `_resolve_semi_diameters` | the release/audit tag and the "pre-fix the JAX builders read only" framing |
| L925-927 | `_build_jax_prescription` | "that the legacy ``trace_jax`` used to do inline on every call" |
| L1037-1037 | `_running_under_trace` | "the v4.12.0 failure mode" |
| L1061-1068 | `_trace_body_static` | the release/audit tag and the "former 4th parameter was DEAD and is gone" narrative |
| L1110-1111 | `_trace_body_traced` | the release/audit tag and "the dead parameter is gone" |
| L1154-1154 | `_TRACE_JAX_CACHE_LOCK` | the release/audit/agent tag |
| L1178-1180 | `<module> registry enrollment` | the release / roadmap tag and "now walks the registry" |
| L1201-1208 | `_make_jit_kernel` | the release/audit tag and the "both DEAD and are gone" / "the old docstring's ..." narrative |
| L1559-1561 | `_intersect_jax_param` | the release/audit tags |
| L1563-1568 | `_intersect_jax_param` | the release/audit tags and "The old ``R_finite > 0`` selector" |
| L1609-1609 | `_intersect_jax_param` | the release and audit tag |
| L1722-1727 | `trace_jax_with_params` | "so pre-fix it silently traced" |

---

### L51-64 -- `<module> RT-6 note` -- "has been RETIRED" / "the old 0.95 gate" / "removed outright"

*Left in the source:* the whole argument for there being no warning, including the ray-line invariance, the OPL telescoping identity and the full parity measurement.  Re-stated as a standing prohibition, plus the pointer to this file.

```text
# RT-6 (AUDIT_RAYTRACE_CORE_2026_07_08): the v4.16.1-4.16.3 high-NA
# "paraxial transfer" RuntimeWarning has been RETIRED.  It was built on a
# mischaracterisation: ``_transfer_jax``'s ``t ~= thickness`` step lands the
# ray at a point that is still EXACTLY on the ray line (only the parameter
# along the line differs from the vertex-plane value), and every downstream
# consumer -- the next surface intersect (a line/surface solve is invariant
# to which point on the line you start from) and the OPL accumulator (which
# telescopes: ``n*(thickness + t_int) == n*t_total`` with signed legs) -- is
# invariant to that choice.  A direct trace_jax-vs-NumPy-trace parity check
# at NA up to 0.64 (min|N|~0.77, well past the old 0.95 gate) agrees to
# ~2.4e-9 m, INVARIANT to the gap length across a 100x sweep (0.09 m -> 9 m);
# the residual is the surface-intersection solver tolerance, NOT a
# thickness-scaling paraxial error.  The warning therefore told users to
# distrust results that are correct to sub-ppm -- removed outright.
```

### L73-77 -- `_ASPHERIC_NEWTON_ITERS` -- "Pre-fix it stopped at 8"

*Left in the source:* the divergence a lower count produces, including the float32 alive-mask case, and the cost note -- all as a live argument for the value 10.

```text
# trace once).  Pre-fix it stopped at 8, so a marginal asphere whose
# Newton refinement needed the 9th/10th NumPy iteration could land at a
# slightly different ``t`` (and, in float32, alive-mask-diverge from the
# NumPy trace).  Matching the count (10) makes the two paths agree in the
# fully-converged regime; the quadratic tail costs ~2 extra evals but is
```

### L84-84 -- `_ASPHERIC_NEWTON_RESIDUAL_TOL` -- the release and audit tag

*Left in the source:* the NumPy acceptance criterion this mirrors and the whole residual-kill contract.

```text
# v5.17.1 (audit P3-59): post-Newton residual acceptance tolerance [m].
```

### L88-92 -- `_ASPHERIC_NEWTON_RESIDUAL_TOL` -- "an unconverged finite t used to be accepted silently"

*Left in the source:* the same hazard as what happens WITHOUT the residual check, and the mirroring of the NumPy 1e-12 criterion.

```text
# FIXED iteration count (no early exit -- required for jit/grad), so
# an unconverged finite t used to be accepted silently and the ray
# landed off-surface.  After the fixed iterations we evaluate the
# residual F(t) = z - sag(x, y) once and kill rays with |F| above this
# tolerance, mirroring the NumPy 1e-12 residual criterion.
```

### L296-296 -- `_intersect_jax` -- the release and audit tag

*Left in the source:* the disc >= 0 acceptance, the NumPy tangency parity and the strict disc > 0 sqrt guard for the gradient mask.

```text
        # v5.17.1 (audit P3-58): acceptance is disc >= 0, matching the
```

### L329-333 -- `_intersect_jax` -- the release/audit tags on both the JAX fix and the NumPy fix it mirrors

*Left in the source:* the direction-aware root argument and the Spencer-Murty near-root identity.

```text
            # v5.4.6 (audit P1-1): direction-AWARE root pick, mirroring the
            # v5.4.1 NumPy fix in intersection.py.  The Spencer-Murty
            # ``e/q`` form above IS that near root (|e/q| <= |q/a|), so a
            # backward-propagating ray (N < 0 after a mirror reflection)
            # no longer lands on the diametrically-opposite FAR root.
```

### L358-361 -- `_intersect_jax` -- the release/audit tag and "used to be silently accepted"

*Left in the source:* why a fixed iteration count needs a residual check, and the trace-safety note.

```text
            # v5.17.1 (audit P3-59): convergence check.  The fixed
            # iteration count has no early-exit/convergence tracking,
            # so a finite-but-unconverged t (steep asphere, grazing
            # incidence) used to be silently accepted where the NumPy
```

### L494-503 -- `_apply_grating_kick_jax` -- "Pre-fix all four sites in the library ... omitted it" and "reproduces the pre-fix (air-correct) behaviour"

*Left in the source:* the physics (tangential-wavevector conservation), the measured magnitude of the omission (1.503583 == n(N-BK7), 50 % direction error), the rule that the OPL term must NOT carry 1/n2, and the n_medium default.

```text
    post-refraction direction cosines carries a ``1 / n2``.  Pre-fix all
    four sites in the library (this one, ``trace``, ``trace_world``,
    ``apply_doe_phase_traced``) omitted it: exact in air, high by exactly
    ``n2`` into glass (measured ratio 1.503583 == n(N-BK7) at
    Lambda = 5 um, lambda = 1.31 um, m = 1 -- a 50 % direction error).
    The OPL term is the grating's own phase screen and must NOT carry the
    ``1 / n2``: its transverse gradient is exactly ``n2 L' - n1 L``.
    ``n_medium`` defaults to 1.0 so a caller that omits it reproduces the
    pre-fix (air-correct) behaviour; both trace bodies pass the surface's
    post-refraction index.
```

### L512-517 -- `_apply_grating_kick_jax` -- the release tags around the traced-period support

*Left in the source:* both failure modes of a ``float()`` cast -- the silent zero gradient and the TracerArrayConversionError -- as live reasons for the ``jnp.where`` form.

```text
    ``jax.grad`` w.r.t. grating period).  Pre-v4.12 used
    ``float(period_*)`` which stripped the JAX trace -> silent zero
    gradient under ``jax.grad``; ``np.isfinite`` on a traced value
    further raised ``TracerArrayConversionError``.  v4.12 keeps the
    trace alive via ``jnp.where`` whenever the period argument is
    JAX-traced.
```

### L564-564 -- `_apply_grating_kick_jax` -- the release and audit tag

*Left in the source:* the strict inequality, the NumPy expression it matches, and the grazing-order consequence.

```text
    # v5.17.1 (audit P3-58): evanescence is STRICTLY sumsq > 1.0,
```

### L631-635 -- `_transfer_jax` -- "This path used to advance EVERY ray's position and OPL unconditionally"

*Left in the source:* the whole backend-drift argument and the alive-mask scaling rule.

```text
    # This path used to advance EVERY ray's position and OPL unconditionally,
    # which is harmless for in-tree consumers (they all mask by ``alive``) but
    # leaves an unmasked ``jax_state_to_raybundle`` reader seeing
    # backend-dependent drift on the dead rows.  Scale the transfer leg by the
    # alive mask so a dead ray matches the NumPy backend (position + OPL frozen).
```

### L805-810 -- `_reject_unsupported_jax_surfaces` -- the release/audit tag and "previously it silently traced ... reproducing the pre-4.10 wrong-answer mode"

*Left in the source:* why the guard is a module-level helper and what the fast entry point does without it.

```text
    v5.17.x (audit P2-34): hoisted to a module-level helper so
    :func:`trace_jax_with_params` (which bypasses
    :func:`_build_jax_prescription` for perf) applies the SAME guard --
    previously it silently traced mirrors / coord-breaks / biconics /
    freeforms as flat refractives, reproducing the pre-4.10 wrong-answer
    mode for differentiable prescriptions.
```

### L880-884 -- `_resolve_semi_diameters` -- the release/audit tag and the "pre-fix the JAX builders read only" framing

*Left in the source:* the requirement to consult ``'elements'`` and the Zemax-prescription vignetting divergence it prevents.

```text
    v5.17.x (audit P2-35 residual): pre-fix the JAX builders read only
    the per-surface ``'semi_diameter'`` key and never consulted
    ``'elements'``, so a Zemax-loaded prescription (whose apertures
    live in ``'elements'``) vignetted under the NumPy trace but not
    under trace_jax / trace_jax_with_params.  Returns a list of Python
```

### L925-927 -- `_build_jax_prescription` -- "that the legacy ``trace_jax`` used to do inline on every call"

*Left in the source:* what the builder performs and the two uses of its output.

```text
    Performs the surface-kind validation, glass-index lookups, and
    semi-diameter resolution that the legacy ``trace_jax`` used to do
    inline on every call.  The output is suitable both for direct kernel
```

### L1037-1037 -- `_running_under_trace` -- "the v4.12.0 failure mode"

*Left in the source:* the whole nested-jit / dot_general NaN argument.

```text
    (the v4.12.0 failure mode).
```

### L1061-1068 -- `_trace_body_static` -- the release/audit tag and the "former 4th parameter was DEAD and is gone" narrative

*Left in the source:* the standing rule -- no such parameter, DOE kicks come from jp.aux[-1] -- and the RT-8 raise that enforces it.

```text
    v5.31 (audit R-18, verified): the former 4th parameter
    ``surface_diffraction`` was DEAD and is gone.  The DOE kicks are read from
    ``jp.aux[-1]`` (``diff_aux``), which :func:`_build_jax_prescription` folds
    in; the parameter was a vestige of the pre-``JaxPrescription`` signature
    and was never referenced in either trace body.  Passing a spec here could
    only mislead -- ``trace_jax`` raises (RT-8) if ``surface_diffraction`` is
    supplied alongside a pre-built prescription, precisely because this path
    cannot honour it.
```

### L1110-1111 -- `_trace_body_traced` -- the release/audit tag and "the dead parameter is gone"

*Left in the source:* the same standing rule and the cross-reference.

```text
    v5.31 (audit R-18, verified): the dead ``surface_diffraction`` parameter is
    gone -- see :func:`_trace_body_static`.
```

### L1154-1154 -- `_TRACE_JAX_CACHE_LOCK` -- the release/audit/agent tag

*Left in the source:* the torn-OrderedDict hazard and the precedent it follows.

```text
# v4.14.2 (P1-NEW-2 / Agent C): thread-safety lock for
```

### L1178-1180 -- `<module> registry enrollment` -- the release / roadmap tag and "now walks the registry"

*Left in the source:* what the enrollment does and the late-binding-closure note.

```text
# v4.16.0 (ROADMAP #15): register the raytrace-JAX clearer with the
# central registry at module-import time.  ``clear_asm_caches`` now
# walks the registry rather than enumerating clear calls by hand.
```

### L1201-1208 -- `_make_jit_kernel` -- the release/audit tag and the "both DEAD and are gone" / "the old docstring's ..." narrative

*Left in the source:* the standing rule about which values are closed over, where the cache key is built, and why wavelength is the one genuinely static value.

```text
    v5.31 (audit R-18, verified): the former ``jp_aux`` and
    ``surface_diffraction`` parameters were both DEAD and are gone.  ``jp_aux``
    was never referenced in the body -- the cache KEY is built by the caller
    (:func:`trace_jax`, ``cache_key = (jp.aux, wavelength, diff_aux)``), not
    here, so the old docstring's "keyed on ``jp_aux``" described the caller's
    job; the aux the kernel actually reads arrives with ``jp`` at call time.
    ``surface_diffraction`` was only forwarded to :func:`_trace_body_static`,
    which ignored it (the DOE kicks live in ``jp.aux[-1]``).  ``wavelength`` is
```

### L1559-1561 -- `_intersect_jax_param` -- the release/audit tags

*Left in the source:* the disc >= 0 acceptance and the strict-guard split.

```text
    # v5.17.1 (audit P3-58): acceptance is disc >= 0 (NumPy parity,
    # v5.4.6 audit P3-3 tangency semantics); the sqrt guard stays on
    # the strict disc > 0 for the H-RT-7 gradient mask.
```

### L1563-1568 -- `_intersect_jax_param` -- the release/audit tags and "The old ``R_finite > 0`` selector"

*Left in the source:* the direction-blindness hazard in full, as the reason the current selector must not be replaced.

```text
    # v5.4.6 (audit P3-1): direction-aware near-root pick (min |t|), matching
    # the v5.4.1 NumPy fix and the static-branch JAX kernel (P1-1).  The old
    # ``R_finite > 0`` selector is direction-blind and lands a backward leg
    # (post-mirror N<0) on the far root, corrupting jax.grad of mirror/folded
    # prescriptions.  The Spencer-Murty ``t = e/q`` form below is that near
    # root by construction (|e/q| <= |q/a|), with no vertex cancellation and
```

### L1609-1609 -- `_intersect_jax_param` -- the release and audit tag

*Left in the source:* the residual kill and its trace-safety note.

```text
    # v5.17.1 (audit P3-59): convergence check -- mirror the NumPy
```

### L1722-1727 -- `trace_jax_with_params` -- "so pre-fix it silently traced"

*Left in the source:* the guard's purpose and what the fast entry point does without it.

```text
    # Audit P2-34: apply the same unsupported-surface fail-loud guard
    # as trace_jax's builder.  This entry point bypasses
    # _build_jax_prescription for perf, so pre-fix it silently traced
    # mirrors / coord-breaks / biconics / freeforms as flat
    # refractives.  Static-fields-only, hence jit/grad trace-safe (see
    # _reject_unsupported_jax_surfaces).
```

