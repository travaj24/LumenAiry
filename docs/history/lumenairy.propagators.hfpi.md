<!-- lumenairy-history-doc
module: lumenairy/propagators/hfpi.py
ast_sha256: 00495d8e91f46de107768eb68b667dadfae0cae7c081eaec0aab832923b0714e
token_sha256: e0e94069cdd67d8055cca286c494980b931aea1edef237801f117c34e98f1ec2
pre_relocation_lines: 1658
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B3 (audit K13 + K22): propagate_hfpi_through_prescription gains z_output (a closing propagate_to_plane hop in the image-space medium) and normalisation='auto'; init_paths_stratified gains sampler='jittered'|'sobol', with the jittered draw moved verbatim into _jittered_cube_draw beside the new _sobol_cube_draw; every default byte-identical
re_recorded: 2026-09-13 -- WP-B3: the normalisation selector's refusal message states the free-space-legs half of the 'auto' rule that the docstring already carried
-->

# Version history -- `lumenairy/propagators/hfpi.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/hfpi.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

HFPI is a Monte-Carlo estimator, and its comments are unusually load-bearing:
most of what looks like history is the derivation of a live weight or a live
guard, and it stayed.  In particular these did NOT move:

* the **K18 source-area derivation** and its measured `sum(weights)/exact`
  table (1.000004 / 0.062140 / 0.003945 / 0.000260 tracking `1/N_pix`), which
  is why `init_paths_from_field` multiplies by `Ny*Nx*dx**2`;
* the **RS-I composition** in `_reemission_measure` -- the direction-variable
  rewrite, the invariant `W_m`, and why `1/cos(theta_in)` is bookkeeping
  rather than physics;
* the **W9-14 sampling-adequacy derivation** (occupancy 0.6 % / 2.1 % / 7.9 %
  at 20k / 80k / 320k paths, seed-to-seed fidelity 0.005 to 0.021, and the
  explicit statement that one path per pixel is necessary and nowhere near
  sufficient), which sets where the guard trips;
* the live `output_grid` -> `output_shape` deprecation, which still warns.

What moved is release chronology around those: what pre-4.10 omitted, what
`rng.spawn(i + 1)[-1]` did to the caller's generator, what the `wavelength=0.0`
default dropped, and the V1 transparent-plane verification of the re-emission
measure.

One comment appeared **three times, verbatim** -- the K19 `rng=None` note at
`init_paths_from_field`, `apply_aperture_diffraction` and
`propagate_hfpi_freespace_aperture`.  All three carried the same measured
pre-fix evidence; all three were condensed the same way, keeping the rule and
the reason (a fixed default seed makes a Monte-Carlo estimator's own error
estimate identically zero) and moving the evidence here once.

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
| L96-105 | `_spawn_rng` | the 4.11.2 account of HFPI walks passing one `rng` to every aperture |
| L117-126 | `_spawn_rng` -- the Generator branch | the measured pre-fix behaviour of ``rng.spawn(stream_index + 1)[-1]``: that it mutated the caller's generator and made the mapping order-dependent |
| L145-150 | `_spawn_rng` -- the JAX branch | that the except clause used to be a bare `except Exception: pass` |
| L243-254 | `init_paths_from_field` -- the rng default | the measured pre-fix evidence that the default was the fixed seed 0 (two default runs byte-identical, and identical to ``rng=0``) |
| L281-286 | `init_paths_from_field` -- Kirchhoff weighting | what pre-4.10 omitted and what that meant for existing relative-contrast results |
| L396-405 | `apply_aperture_diffraction` -- `.. versionchanged:: 5.46` | the full measured account of what the `wavelength=0.0` default dropped |
| L434-445 | `apply_aperture_diffraction` -- the rng default | the measured pre-fix evidence that the default was the fixed seed 0 (two default runs byte-identical, and identical to ``rng=0``) |
| L474-483 | `apply_aperture_diffraction` -- the re-emission prefactor | the pre-4.11.2 obliquity-only weights and the V1 pointer to the pre-fix measure |
| L542-558 | `_reemission_measure` -- the V1 verify note | the transparent-plane measurement that established the composition (amplitudes low by exactly ``n_paths * z1``, six ratios over a 4x range of z1 and n_paths) |
| L904-913 | `_promote_to_complex` | the v5.17.x account of `output_dtype=E_in.dtype` reaching `accumulate_to_grid`, and the pointer to the v4.10 twin fix |
| L1152-1163 | `propagate_hfpi_freespace_aperture` -- the rng default | the measured pre-fix evidence that the default was the fixed seed 0 (two default runs byte-identical, and identical to ``rng=0``) |
| L1383-1385 | `propagate_hfpi_through_prescription` -- sampling validation | that the parameter was documented but never dispatched |
| L1427-1431 | `propagate_hfpi_through_prescription` -- sampling dispatch | that the variance-reduction path was dead code |

---

### L96-105 -- `_spawn_rng` -- the 4.11.2 account of HFPI walks passing one `rng` to every aperture

*Left in the source:* the live hazard in full -- ``RandomState`` rebuilds a fresh ``default_rng(rng)`` per construction, so one int seed means identical draws at every aperture -- and what the function does about it

```text
    4.11.2: HFPI prescription walks previously called
    ``apply_aperture_diffraction(... rng=rng)`` with the same caller-
    supplied ``rng`` at every aperture.  Because
    :class:`lumenairy.backend.random.RandomState` rebuilds a fresh
    NumPy ``default_rng(rng)`` on every construction when ``rng`` is
    an integer (or None), passing the same int seed produced *the
    same uniform draws at every aperture* -- perfectly correlated
    diffraction events across the cascaded stack.  We derive a
    distinct child seed per stream index by hashing the parent seed
    with the stream counter.
```

### L117-126 -- `_spawn_rng` -- the Generator branch -- the measured pre-fix behaviour of ``rng.spawn(stream_index + 1)[-1]``: that it mutated the caller's generator and made the mapping order-dependent

*Left in the source:* the requirement that produces the current code -- a PURE function of ``(parent, stream_index)`` that does not mutate the caller's generator

```text
        # K24 (audit 2026-09-11): derive the child from the parent's
        # SeedSequence ENTROPY, exactly as the ``int`` branch above does,
        # so ``stream_index`` is a stable key.  The pre-fix
        # ``rng.spawn(stream_index + 1)[-1]`` MUTATED the caller's
        # generator (it advances ``n_children_spawned`` on every call)
        # and discarded ``stream_index`` children each time, so the
        # mapping was not a pure function of ``(parent, i)``: measured,
        # a fresh parent at the same stream index reproduced its draws,
        # but "stream 1 drawn AFTER stream 0" differed from "stream 1
        # drawn alone".  It also spawned i+1 children to use one.
```

### L145-150 -- `_spawn_rng` -- the JAX branch -- that the except clause used to be a bare `except Exception: pass`

*Left in the source:* why the clause must stay NARROW -- a swallowed exception hands both streams the identical key, which is the correlation this function exists to prevent

```text
    # K15/K24 (audit 2026-09-11): this used to be ``except Exception:
    # pass``, which fell through to "return as-is" and handed BOTH
    # streams the IDENTICAL key -- the exact correlation this function
    # exists to prevent -- with no diagnostic whatever.  Narrowed to the
    # import / attribute failures that can actually occur, and the
    # fall-through now says what it did.
```

### L243-254 -- `init_paths_from_field` -- the rng default -- the measured pre-fix evidence that the default was the fixed seed 0 (two default runs byte-identical, and identical to ``rng=0``)

*Left in the source:* the rule and the reason it matters -- a fixed default seed makes the estimator's own error estimate identically zero.  One of three identical copies of this comment; all three were condensed the same way.

```text
    # K19 (audit 2026-09-11): ``rng=None`` -- the DEFAULT on every HFPI
    # entry point -- must draw fresh system entropy, which is exactly
    # what ``RandomState(None)`` does (``np.random.default_rng(None)``).
    # Pre-fix this read ``rng if rng is not None else 0``, so the
    # default was the FIXED seed 0 and ``_spawn_rng``'s documented
    # "let each aperture pull from system entropy" branch was
    # unreachable.  Measured: two default runs byte-identical, and
    # identical to ``rng=0``.  HFPI is a 1/sqrt(N) Monte-Carlo
    # estimator sold on that convergence; the canonical way to see its
    # error is to re-run with a new seed, and on the default path that
    # error estimate was identically ZERO.  Pass an int (or a
    # Generator) for reproducibility.
```

### L281-286 -- `init_paths_from_field` -- Kirchhoff weighting -- what pre-4.10 omitted and what that meant for existing relative-contrast results

*Left in the source:* the live weighting formula above it and the whole K18 source-area derivation below it, measured table included

```text
    # Pre-4.10 omitted both 1/(i*lambda) and the solid-angle weight,
    # so absolute amplitudes were unphysical by ~10^6 per re-emission
    # at visible wavelengths.  Intensity ratios across paths were
    # unaffected (the missing factors are global), so existing relative-
    # contrast results still hold; absolute-photometry use is new.
    # K18 (audit 2026-09-11): the SOURCE-AREA factor.  The source pixel is
```

### L396-405 -- `apply_aperture_diffraction` -- `.. versionchanged:: 5.46` -- the full measured account of what the `wavelength=0.0` default dropped

*Left in the source:* the directive, the gate that made a zero default dangerous, the size of the error and the migration

```text
            No longer defaults to ``0.0`` (audit K12).  The
            ``1/(i*lambda)`` Kirchhoff prefactor was gated on
            ``wavelength > 0``, so omitting it silently DROPPED the
            prefactor: measured, every path weight came out wrong by a
            factor of exactly ``1/lambda`` = 1.5798e6 in magnitude AND
            by -90 degrees in phase, with zero warnings -- precisely the
            failure the v4.11.2 prefactor work fixed.  A physically
            meaningless default (lambda = 0) must not silently mean
            "skip the physics".  Migration: pass ``wavelength=`` (the
            library's own entry points always did).
```

### L434-445 -- `apply_aperture_diffraction` -- the rng default -- the measured pre-fix evidence that the default was the fixed seed 0 (two default runs byte-identical, and identical to ``rng=0``)

*Left in the source:* the rule and the reason it matters -- a fixed default seed makes the estimator's own error estimate identically zero.  One of three identical copies of this comment; all three were condensed the same way.

```text
    # K19 (audit 2026-09-11): ``rng=None`` -- the DEFAULT on every HFPI
    # entry point -- must draw fresh system entropy, which is exactly
    # what ``RandomState(None)`` does (``np.random.default_rng(None)``).
    # Pre-fix this read ``rng if rng is not None else 0``, so the
    # default was the FIXED seed 0 and ``_spawn_rng``'s documented
    # "let each aperture pull from system entropy" branch was
    # unreachable.  Measured: two default runs byte-identical, and
    # identical to ``rng=0``.  HFPI is a 1/sqrt(N) Monte-Carlo
    # estimator sold on that convergence; the canonical way to see its
    # error is to re-run with a new seed, and on the default path that
    # error estimate was identically ZERO.  Pass an int (or a
    # Generator) for reproducibility.
```

### L474-483 -- `apply_aperture_diffraction` -- the re-emission prefactor -- the pre-4.11.2 obliquity-only weights and the V1 pointer to the pre-fix measure

*Left in the source:* what the call does and where the exact intermediate-leg measure is derived

```text
    # 4.11.2: apply the Kirchhoff prefactor for each re-emission,
    # matching the convention applied in :func:`init_paths_from_field`.
    # Pre-4.11.2 the per-aperture re-emission left the weights with only
    # an obliquity factor, so each cascaded aperture under-weighted by
    # ~10^6 at visible wavelengths.
    #
    # V1 (verify pass, 2026-09-12): the MEASURE of that re-emission was
    # still wrong for a chain -- see :func:`_reemission_measure` for the
    # derivation, the pre-fix factor and the measured residual law
    # (amplitudes low by exactly ``n_paths * z_to_aperture``).
```

### L542-558 -- `_reemission_measure` -- the V1 verify note -- the transparent-plane measurement that established the composition (amplitudes low by exactly ``n_paths * z1``, six ratios over a 4x range of z1 and n_paths)

*Left in the source:* the full RS-I derivation above it, what ``'legacy'`` returns, and BOTH ways that factor is wrong for a chain -- which is the contract a caller choosing between the two modes needs

```text
    V1 (verify pass, 2026-09-12): the pre-fix factor was
    ``0.5*(cos_in + cos_out) * (1/(i lambda)) * Omega_out / n_paths``,
    which is wrong twice for a chain.  (a) The ``/ n_paths`` divides by
    the sample count a SECOND time -- a path is ONE sample of the joint
    (source pixel, direction_1, ..., direction_m) integral, so the
    ``1/n_paths`` belongs once and already lives in
    ``init_paths_from_field`` alongside ``A_src * Omega_1``.  (b) The
    intermediate leg carried no ``r``.  Measured on the oracle-free
    "an unobstructed aperture plane is transparent" property -- the
    two-leg walk over ``z1 + z2`` must equal the one-leg walk over the
    same total -- the returned amplitude was low by exactly
    ``n_paths * z1``: ``two/one * n_paths * z1`` = 1.098 / 1.046 /
    0.860 / 0.920 / 1.339 / 0.801 across z1 = 0.25 / 0.5 / 1.0 mm and
    n_paths = 0.5 / 2 M, i.e. 1.0 to Monte-Carlo scatter over a 4x range
    of each.  The symmetric Kirchhoff obliquity
    ``0.5(cos_in + cos_out)`` is a heuristic; the exact RS-I composition
    wants the outgoing cosine alone, which is what this returns.
```

### L904-913 -- `_promote_to_complex` -- the v5.17.x account of `output_dtype=E_in.dtype` reaching `accumulate_to_grid`, and the pointer to the v4.10 twin fix

*Left in the source:* the live hazard -- path weights are intrinsically complex, so a real buffer loses their imaginary part behind a suppressible ComplexWarning -- and the measured cost of it

```text
    v5.17.x (P2-32): the end-to-end helpers used to pass
    ``output_dtype=E_in.dtype`` straight into
    :func:`accumulate_to_grid`, so a REAL-dtype input field (e.g. a
    plain float aperture/amplitude mask) allocated a real output buffer
    and ``np.add.at`` silently discarded the imaginary part of every
    complex path weight (the weights are intrinsically complex: the
    ``1/(jλ)`` prefactor and every ``exp(j·k·Δs)`` leg).  Only a
    suppressible ComplexWarning was emitted; measured ~40% of the total
    intensity silently lost on a flat real source.  Mirrors the v4.10
    fix in ``hf.py`` for the same bug class.
```

### L1152-1163 -- `propagate_hfpi_freespace_aperture` -- the rng default -- the measured pre-fix evidence that the default was the fixed seed 0 (two default runs byte-identical, and identical to ``rng=0``)

*Left in the source:* the rule and the reason it matters -- a fixed default seed makes the estimator's own error estimate identically zero.  One of three identical copies of this comment; all three were condensed the same way.

```text
    # K19 (audit 2026-09-11): ``rng=None`` -- the DEFAULT on every HFPI
    # entry point -- must draw fresh system entropy, which is exactly
    # what ``RandomState(None)`` does (``np.random.default_rng(None)``).
    # Pre-fix this read ``rng if rng is not None else 0``, so the
    # default was the FIXED seed 0 and ``_spawn_rng``'s documented
    # "let each aperture pull from system entropy" branch was
    # unreachable.  Measured: two default runs byte-identical, and
    # identical to ``rng=0``.  HFPI is a 1/sqrt(N) Monte-Carlo
    # estimator sold on that convergence; the canonical way to see its
    # error is to re-run with a new seed, and on the default path that
    # error estimate was identically ZERO.  Pass an int (or a
    # Generator) for reproducibility.
```

### L1383-1385 -- `propagate_hfpi_through_prescription` -- sampling validation -- that the parameter was documented but never dispatched

*Left in the source:* why the check runs BEFORE the expensive parse, and what it prevents

```text
    # HFPI-2: validate the sampling selector up front (before the expensive
    # prescription parse / trace) -- it was previously documented but never
    # dispatched, so a typo silently ran uniform sampling.
```

### L1427-1431 -- `propagate_hfpi_through_prescription` -- sampling dispatch -- that the variance-reduction path was dead code

*Left in the source:* what the selector does

```text
    # HFPI-2: honour the ``sampling`` selector (validated up front).  Pre-fix
    # the parameter was documented (default ``'stratified'``) but never
    # dispatched -- every call used plain uniform ``init_paths_from_field``
    # regardless, so the variance-reduction path
    # (:func:`init_paths_stratified`) was dead.
```
