<!-- lumenairy-history-doc
module: lumenairy/propagators/gbd.py
ast_sha256: 904b6049fccac757a99dab51e0b1b0cd102efd5e07b1bfb4566f902f06c9bd45
token_sha256: c087abfe470aeabdc6465480dcd92fec96b8ec114e32be2b9cd88b33505d8375
pre_relocation_lines: 3845
recorded_by: WP-A17 SWEEP-4 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B7 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, WP-A4 sec. 6 items 3-8 + VERIFY-B1 F1/F2): the Y4 fused basis evaluation and hoisted Newton factor, the aberration_tensor mode/waist caches, the S6 gate's k1-slope statistic and mean-plus-spread chart sizing, the JAX screen's chief-ray displacement term, and the S9 FFT kernel clip
re_recorded: 2026-09-15 -- WP-B12b: the per-surface beamlet image leg consumes the shared exit-vertex projection (reference='exit_vertex') and the in-line conic-sag copy is deleted
-->

# Version history -- `lumenairy/propagators/gbd.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/gbd.py` -- the "vX.Y (audit Z): pre-fix this did A, which was
wrong because B, now it does C" blocks, the comments that corrected earlier
comments, and the per-release chronologies that had accumulated on constants
whose CURRENT value is what the source now states.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

What did NOT move: the measured derivations of this module's live
constants and defaults (`docs/TESTING_STANDARDS.md` S5) -- the beamlet-count /
waist sizing tables, the `sample_step` convergence recommender's own
measurements, the analytic-vs-FD Jacobian agreement figures, and the
truncation-free `'auto'` selection argument.  The live deprecation statements
(`output_grid`, `wavelength` on the recommender) stayed as statements about
what a caller should pass now.


Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment plus a pointer to this file; those are noted per block
below as *Left in the source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L192-200 | the ``output_grid`` / ``output_shape`` banner | "Pre-v5.2 the sub-propagators below interpreted ``output_grid`` as ``(Ny, Nx)``" and which release renamed what |
| L463-470 | the beamlet launch grid | "switch from cell-centred ... to pixel-centred" and "prior to this fix a coherent self-roundtrip walked the beamlet centres half a pixel" |
| L962-968 | the axial-phase sign in the beamlet transfer | "Pre-4.9 ``abs(t)`` accidentally took the complex conjugate of the axial phase on back-propagation" |
| L1030-1035 | the thin-lens kick on a beamlet bundle | "Pre-4.10.2 subtracted x/f directly from L" |
| L1361-1366 | the per-beamlet tilt ramp | "Pre-4.10 omitted this ramp" and what that cost |
| L1375-1388 | the fused-exponential reduction | "v4.13.1 perf: fuse the two ``xp.exp`` calls into one ... where pre-v4.13.1 evaluated ... separately" |
| L1419-1429 | the reconstruction phase argument | "The pre-fix ``-0.5*Q`` rendered the complex CONJUGATE of the propagated wavefront curvature" |
| L3195-3215 | the Collins amplitude factor | two stacked defect narratives -- "v5.17.x (P2-30) BEHAVIOUR CHANGE: pre-fix this applied ``Q_new / Q_old``" and "The pre-fix un-conjugated form carried the Gouy / Collins phase with the wrong sign" |
| L3755-3771 | the whole-system-ABCD axial OPL | a two-release narrative -- "Pre-4.11.1 ``axial_opl=`` was never populated" and "4.11.2: the v4.11.1 implementation here was dead-on-arrival" |

---

### L192-200 -- the ``output_grid`` / ``output_shape`` banner -- "Pre-v5.2 the sub-propagators below interpreted ``output_grid`` as ``(Ny, Nx)``" and which release renamed what

*Left in the source:* the two live contracts, which one is canonical, the failure a shape-reading sub-propagator produces, and the deprecation of the legacy kwarg

```text
# v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): output-grid kwarg semantics
# disambiguation.  Pre-v5.2 the sub-propagators below interpreted
# ``output_grid`` as ``(Ny, Nx)`` (shape only) while the dispatcher's
# ``propagate(output_grid=...)`` contract advertises ``(N_out, dx_out)``.
# The mismatch silently produced wrong-shape output arrays when the
# dispatcher forwarded the kwarg.  v5.2 keeps the dispatcher contract
# as canonical and renames the sub-propagator kwarg to ``output_shape``
# for the shape-only meaning.  The legacy ``output_grid`` kwarg on
# the sub-propagators is preserved with a ``DeprecationWarning``.
```

### L463-470 -- the beamlet launch grid -- "switch from cell-centred ... to pixel-centred" and "prior to this fix a coherent self-roundtrip walked the beamlet centres half a pixel"

*Left in the source:* which convention this grid uses, which siblings share it, and the phase error a cell-centred grid produces

```text
    # v4.12.1 (B1-10): switch from cell-centred `(arange(N) - N/2 + 0.5)*dx`
    # to pixel-centred `(arange(N) - N/2)*dx`, matching the library-wide
    # convention (ASM, Fresnel, RS, sources, ``apply_fresnel_curvature``).
    # ``reconstruct_field_from_beamlets`` (line ~264) already uses the
    # pixel-centred grid, so prior to this fix a coherent self-roundtrip
    # walked the beamlet centres half a pixel relative to the
    # reconstruction grid -- producing a `k_0 * dx / 2 * off-axis` phase
    # error that grew with NA and field angle.
```

### L962-968 -- the axial-phase sign in the beamlet transfer -- "Pre-4.9 ``abs(t)`` accidentally took the complex conjugate of the axial phase on back-propagation"

*Left in the source:* the rule and the failure it prevents, present tense

```text
    # 4.9 fix (audit #2.2): use raw ``t`` (signed) instead of
    # ``abs(t)``.  Under the exp(-iωt) time convention forward
    # propagation by distance z imparts exp(+i·k·z) -- correct for
    # both signs of z.  Pre-4.9 ``abs(t)`` accidentally took the
    # complex conjugate of the axial phase on back-propagation,
    # giving wrong sign on the propagated wavefront.  Forward
    # propagation was unaffected (because abs(positive) == positive).
```

### L1030-1035 -- the thin-lens kick on a beamlet bundle -- "Pre-4.10.2 subtracted x/f directly from L"

*Left in the source:* the rule -- the kick acts on paraxial SLOPES -- and the measured per-surface error of the direction-cosine form

```text
    # 4.10.2: the thin-lens kick acts on PARAXIAL SLOPES u = L/N, not on
    # direction cosines.  Pre-4.10.2 subtracted x/f directly from L,
    # which is only correct in the small-angle limit (N -> 1).  For
    # moderately non-paraxial bundles (N ~ 0.95-0.99) this introduces
    # a few-percent error per surface; for wide-angle fans the error
    # compounds.  Convert to slope, apply the kick, re-normalise.
```

### L1361-1366 -- the per-beamlet tilt ramp -- "Pre-4.10 omitted this ramp" and what that cost

*Left in the source:* what the ramp is for and when it is a no-op

```text
    # correctly off-chief-ray.  Pre-4.10 omitted this ramp; the
    # focal spot still focused correctly (the chief-ray phase is the
    # same), but off-chief-ray interference patterns and PSF wings
    # were degraded.  When the beamlets bundle was assembled with
    # ``directions = (0, 0)`` (the default for axial-input decompositions)
    # the ramp is zero so this fix is a no-op for that path.
```

### L1375-1388 -- the fused-exponential reduction -- "v4.13.1 perf: fuse the two ``xp.exp`` calls into one ... where pre-v4.13.1 evaluated ... separately"

*Left in the source:* what the fused form is, why it is exact to ulp, and why the einsum reduction replaces the intermediate

```text
    # v4.13.1 perf: fuse the two ``xp.exp`` calls into one (only on the
    # has_dirs branch, where pre-v4.13.1 evaluated ``exp(-i*k*Q*rho2/2)``
    # and ``exp(i*k*tilt)`` separately and multiplied them).  ``exp(A) *
    # exp(B) == exp(A + B)`` analytically; in complex128 the round-off
    # difference is ulp-level (<1e-15 relative) -- well within the
    # propagator accuracy budget.  This roughly halves the per-chunk
    # transcendental cost (exp dominates the inner-loop runtime on
    # moderate grids).  Also switches the per-chunk reduction from
    # ``out + sum(a_b * phase, axis=-1)`` to ``out += einsum('mnk,k->mn',
    # phase, a_b)`` -- the ``a_b * phase`` intermediate is the
    # largest 3-D buffer the loop allocates (chunk * Ny * Nx complex),
    # so dropping it shrinks the working set noticeably for the
    # default chunk_beamlets=4096 and saves one big allocation per
    # chunk on numpy.
```

### L1419-1429 -- the reconstruction phase argument -- "The pre-fix ``-0.5*Q`` rendered the complex CONJUGATE of the propagated wavefront curvature"

*Left in the source:* the convention chain, the trap, and why no intensity/focus test catches it

```text
            # Fused phase argument.  v5.4.6 (audit F-1): the stored Q uses
            # the engineering 1/q parameterisation (q_code = conj(q_physics));
            # the reconstructed FIELD must be expressed in the library's
            # exp(-i omega t) / forward exp(+ikz) convention, i.e. the
            # transverse curvature is exp(+i k rho^2 / (2 q_physics)) =
            # exp(+0.5j k conj(Q) rho^2).  The pre-fix ``-0.5*Q`` rendered the
            # complex CONJUGATE of the propagated wavefront curvature (the
            # |E| envelope and waist are sign-blind, so intensity/focus tests
            # never caught it).  conj(Q) has the same Im part, so the Gaussian
            # decay and the on-axis-waist (Re(Q)=0) reconstruction are
            # unchanged; only the off-waist phase sign is corrected.
```

### L3195-3215 -- the Collins amplitude factor -- two stacked defect narratives -- "v5.17.x (P2-30) BEHAVIOUR CHANGE: pre-fix this applied ``Q_new / Q_old``" and "The pre-fix un-conjugated form carried the Gouy / Collins phase with the wrong sign"

*Left in the source:* the factor, and both ways of getting it wrong stated as hazards with their measured consequences

```text
    # v5.17.x (P2-30) BEHAVIOUR CHANGE: pre-fix this applied
    # ``Q_new / Q_old`` = ``q_in / q_out`` = (C*q_in + D)/(A + B*Q_in),
    # i.e. the Collins factor times a spurious ``(C*q_in + D)``.  For any
    # focusing system (C != 0) that factor has non-unit modulus, so the
    # single-ABCD path disagreed with the sequential per-leg path
    # (``propagate_beamlets_freespace`` + ``apply_thin_lens_to_beamlets``
    # compose exactly to exp(ikL)/(A + B*Q_in), verified to 1e-15) in
    # both amplitude and piston phase -- e.g. a t1=20mm -> f=50mm ->
    # t2=30mm system came out |C*q_in+D| = 0.60x low in field amplitude
    # (0.36x in intensity) with a wrong piston, defeating the v4.11.1
    # axial_opl coherent-superposition fix.  Free-space-only ABCD (C=0,
    # D=1) is unchanged.  ``Q`` here is the module's engineering 1/q
    # parameterisation (Q = 1/q_code, q_code = conj(q_physics)); ABCD
    # elements are real so the Collins factor commutes with that
    # convention.
    # S5 (audit): the Collins amplitude is ``1/(A + B/q_phys)``; with the
    # module's engineering ``Q = 1/q_code = 1/conj(q_phys)`` and REAL ABCD
    # elements that is ``conj(1/(A + B Q))``.  The pre-fix un-conjugated form
    # carried the Gouy / Collins phase with the wrong sign -- near-global (and
    # therefore nearly invisible) at a lens exit plane, but not at
    # ``output_plane_distance != 0`` nor in any coherent combination.
```

### L3755-3771 -- the whole-system-ABCD axial OPL -- a two-release narrative -- "Pre-4.11.1 ``axial_opl=`` was never populated" and "4.11.2: the v4.11.1 implementation here was dead-on-arrival"

*Left in the source:* what the axial OPL is, which plane it reaches, what a missing one costs, and the dataclass-vs-dict trap the bare `except` hides

```text
    # 4.11.1 (H-AS-1): compute the axial OPL = sum_k n_k * t_k across
    # every glass/air segment of the prescription, out to the EXIT VERTEX
    # (this whole-system-ABCD path reconstructs THERE -- it does NOT add a
    # BFL / image-plane leg; that leg belongs to the per_surface=True
    # z_image path).  Pre-4.11.1 ``axial_opl=`` was never populated so the
    # per-beamlet
    # complex envelope lacked the system's axial phase reference and
    # multi-prescription reconstructions had the wrong piston relative
    # to ASM / Fresnel cross-checks.
    #
    # 4.11.2: the v4.11.1 implementation here was dead-on-arrival:
    # ``surfaces_from_prescription`` returns ``List[Surface]`` (Surface
    # is a @dataclass, not a dict), so ``_s.get('thickness', 0.0)``
    # raised AttributeError on the first iteration -- silently swallowed
    # by the surrounding bare ``except Exception`` and ``axial_opl``
    # always defaulted to None.  Switched to attribute access on the
    # Surface dataclass.  Caught by AUDIT_ROUND3_2026_05_16.md (CRIT-8).
```
