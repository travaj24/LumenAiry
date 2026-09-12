<!-- lumenairy-history-doc
module: lumenairy/elements/pmm/_core.py
ast_sha256: 94a17cad10eb2a98784d6f61d65b384d417207bc4871e0b3563ea08d5ffc5ae8
token_sha256: c528c9a8441739d4cca9e11b5cccbe2968a7ea627f2ed2d80b6a1024c8fc9b7b
pre_relocation_lines: 7690
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/pmm/_core.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/pmm/_core.py`: the dated `UPDATE` rounds of the T3-4
mode-cut guard, the "this docstring used to describe it as ..." corrections,
and the "the former X did A, which was wrong because B" notes at the far-field
and gauge sites.  Each block is reproduced **verbatim** under the source line it
came from in the pre-relocation file.

What did NOT move: every measured derivation of a live bar
(`_MORTAR_RESID_REFUSE`, `_MORTAR_RCOND_REFUSE`, `_MODE_CUT_MARGIN_WARN`,
`_MODE_GROWTH_REL`, the Rayleigh-projection least-squares bar, the union-grid
deadband), the calibration tables the shipped defaults are sized on, the
fail-before switches' descriptions, and the two wordings
`tests/unit/test_fix_pmm2d_mortar_round4.py` reads straight out of the source
("EXACTLY ONE side", "ASYMMETRIC", "BOTH sides promoted").

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
| L133-140 | `_resolve_incidence` docstring | G4 -- what a two-angle call resolved to before the warning |
| L174-175 | `_resolve_incidence_checked` docstring | S1-7 -- that ``set_source`` formerly bypassed this resolver |
| L775-777 | `_forward_branch_flip` docstring | that the selector was copy-pasted across five sites before the S1-8 consolidation |
| L787-792 | `_forward_branch_flip` docstring, the ``xp`` paragraph | that each branch reproduces its former per-site copy byte-for-byte |
| L807-813 | `_freeze_cached` docstring | W7 A13 -- the measured pre-fix poison census across four caches |
| L840-841 | `_mass_flux_cut` docstring | "The historical floor was ..." |
| L979-1041 | `PMM_MODE_CUT_GUARD`, the T3-4 block comment | the three dated UPDATE rounds (2026-08-05 / -06 / -08), including the one that retracts the bullet above it |
| L1630-1636 | `_mode_cut_verdict` docstring, channel A | what channel A read until 2026-08-08 and why it stopped |
| L1814-1816 | `_sem_modes` docstring, the ``robust`` parameter | that audit M10 corrected this docstring's own earlier claim |
| L2040-2045 | `_assemble_jones_farfield`, the incidence guard | audit M3 -- the ``abs(kz_inc) < 1e-9`` test this replaced, and the negative totals it let through |
| L2205-2215 | `_scalar_farfield_RT`, the absorbing-superstrate incident flux | W7 F-B -- the historical ``Re(kz_inc/eps_sup)`` recipe and the TE/TM drift it measured |
| L2223 | `_scalar_farfield_RT`, the gauge-corrected flux | the "Post-fix:" framing on the parity measurement |
| L2702-2708 | `_GEO_EIG_CACHE` | "It used to be a 64-entry ``OrderedDict`` keyed on the FULL operator bytes" and the ~180 MB it retained |
| L2722 | `_geo_eig_key` docstring | "in the same sense the old full-bytes key was" |
| L3211-3216 | `_stabilize_scalar` docstring, the lossless skip | "so the gate was contradicting the contract stated one call earlier.  Measured pre-fix, ..." |
| L3251 | `_stabilize_scalar`, the passive gate | "the historical one-sided ... test" |
| L3520-3526 | `_farfield_order_set` docstring, the traced-wavelength refusal | "the old code silently fell back" and "Measured pre-fix" |
| L5355-5358 | `_MORTAR_RESID_REFUSE` derivation | a ROUND 4 CORRECTION parenthetical about what this paragraph itself used to say |
| L5436-5441 | `_MORTAR_RESID_REFUSE` derivation, the scoping measurement | the "CORRECTION, ROUND 4 ... not merely to have a promoted side" framing |
| L5450-5451 | `_MORTAR_RESID_REFUSE` derivation, the closing sentence | "what changes is the prediction a later reader would make from the wording" -- a note about the documentation's own history |
| L6804-6808 | `_build_generator_metric_oop`, the factor-i fix | that the legacy real coefficients shared the rcwa generator's defect and agreed with the PRE-fix rcwa OOP results |

---

### L133-140 -- `_resolve_incidence` docstring -- G4 -- what a two-angle call resolved to before the warning

*Left in the source:* what the call does now and why the warning is gated on ``angle != 0``.

```text
    What IS done here (2026-09-12, audit finding G4) is to stop the SILENT
    half: two DIFFERENT non-zero angles in one call is a caller mistake with no
    legitimate reading, and it used to resolve to ``theta`` with no signal at
    all.  It now WARNS, naming both values and the one that won, while still
    resolving to ``theta`` so nothing downstream moves.  The warning is gated on
    ``angle != 0`` because a bare ``theta=...`` call leaves ``angle`` at its
    ``0.0`` default and is the ordinary, correct usage -- indistinguishable
    here from an explicit ``angle=0.0``."""
```

### L174-175 -- `_resolve_incidence_checked` docstring -- S1-7 -- that ``set_source`` formerly bypassed this resolver

*Left in the source:* the rule that matters now: every PMM entry point and source setter routes through this one resolver, so a back-side angle is rejected identically everywhere.

```text
    is rejected identically everywhere (audit S1-7: ``set_source`` formerly
    bypassed this and silently solved the supplementary front-side geometry)."""
```

### L775-777 -- `_forward_branch_flip` docstring -- that the selector was copy-pasted across five sites before the S1-8 consolidation

*Left in the source:* the provenance in one clause -- S1-8 consolidated five copies here, and the multi-copy pattern is what bred the factor-i defect -- because that is the reason not to inline it again.

```text
    branch -- the noise-robust selector formerly COPY-PASTED verbatim across the
    five scalar-vertical PMM generator sites (audit S1-8: the exact multi-copy
    pattern that bred the six-copy factor-i defect).
```

### L787-792 -- `_forward_branch_flip` docstring, the ``xp`` paragraph -- that each branch reproduces its former per-site copy byte-for-byte

*Left in the source:* the behavioural difference between the two branches, which is the contract a caller needs: NumPy materialises ``tol``, JAX keeps it traced so the angle derivative flows.

```text
    ``xp`` selects the array module: NumPy (default) materialises ``tol`` as a
    concrete Python float via ``max(float(...), 1.0)``, exactly as the historical
    NumPy copies did; passing ``jax.numpy`` keeps ``tol`` traced through
    ``jnp.maximum`` so the derivative w.r.t. the incidence angle still flows --
    reproducing the former JAX copy byte-for-byte.  This is a pure consolidation:
    every routed call site produces bit-identical output."""
```

### L807-813 -- `_freeze_cached` docstring -- W7 A13 -- the measured pre-fix poison census across four caches

*Left in the source:* the hazard and the four caches it covers, with the measurement stated as what happens when a caller writes into a shared array.

```text
    solve that hits the same key.  Measured pre-fix (mutate one entry by
    ``+= 1e-3``, then re-solve): ``PMM2DStackHybrid._geom_cache`` 21 of 23
    arrays writeable -> next solve drifts 1.543e-06;
    ``_PreparedPMMStack._eig_cache`` 12 of 12 -> 7.844e-07;
    ``stack2d._epsF_cache`` -> ``internal_field`` Ez drifts 1.377e-04; and
    ``_jax_twod._STATIC_CACHE`` 8 of 8 at MODULE scope, so the poison
    survives for the whole process.  No in-tree caller mutates them, so this
```

### L840-841 -- `_mass_flux_cut` docstring -- "The historical floor was ..."

*Left in the source:* why that floor cannot work -- the ``1.0`` carries length units -- which is the whole unit-safety argument.

```text
    historical floor was ``1e-9 * max(max|flux|, 1.0)`` -- and that ``1.0``
    has length units, pinning the cut at an ABSOLUTE ``1e-9``.  Harmless in
```

### L979-1041 -- `PMM_MODE_CUT_GUARD`, the T3-4 block comment -- the three dated UPDATE rounds (2026-08-05 / -06 / -08), including the one that retracts the bullet above it

*Left in the source:* what the two channels read TODAY, the coin-flip measurement that demoted ``spread``, the CI false alarm that moved channel A onto the residual, and why the default is still disarmed.

```text
# ---------------------------------------------------------------------------
# UPDATE 2026-08-05 (PMM_FOURNAME_ADJUDICATION_2026_08_05).  HALF of that is
# now closed, and the reason the other half is not is unchanged.
#
# The ``spread`` half of the conjunction was measured to be a COIN FLIP, not a
# property of the device: on the M3 cell ns = 6, degree = 10 -- 411 % wrong at
# |R+T-1| = 5.0e-07 -- the SAME wrong answer (0.5683670) reads spread = 1 at
# one BLAS thread and spread = 0 at N, because whether a round-off-flux mode
# lands above or below the cut depends on the reduction order.  The guard
# therefore spoke on that cell on one mount and was SILENT on the other.
#
# The replacement is a PHYSICAL invariant rather than a second statistic:
# :func:`_mode_cut_growth` asks whether the cut has put a GROWING mode in the
# forward set, which a passive layer cannot have at any thread count.  It reads
# >= 1 on every silent-wrong cell of the classical family on BOTH mounts and 0
# on every cured cell on both.  It is channel A of :func:`_mode_cut_verdict`;
# the old conjunction is retained as channel B so nothing goes quiet.
#
# The CONICAL false positive below is NOT closed by it: those three correct
# cells carry a growing forward mode too (1-3 of them, at 1.07-2.87 x the cut).
# So the default stays DISARMED, for the same reason and now on a second
# instrument.  The remaining lead is still the consensus probe.
#
# ---------------------------------------------------------------------------
# UPDATE 2026-08-06 (FIX_UNION_GRID_2THREAD_2026_08_06).  Channel A's invariant
# is now also a REPAIR -- :func:`_forward_growth_flip` -- so the condition this
# guard warns about is FIXED at the classification site rather than merely
# reported.  Two consequences a reader needs:
#
# * the census and the verdict deliberately still read the RAW ``prop``/``q``
#   (see :func:`_record_mode_cut`'s call sites), i.e. they report what the bare
#   selector WOULD have done.  So channel A still fires on cells the repair has
#   since made correct, and every calibration table in the M3 audit still
#   reads as written.  That is intentional: the instrument measures the
#   diagnosis, not the treatment;
# * which makes the DISARMED default less urgent rather than more.  The
#   conical false positive is unchanged and arming is still gated on the
#   consensus probe, but a growing forward mode is no longer a wrong answer
#   waiting to happen -- it is a repaired one that the instrument still names.
#
# ---------------------------------------------------------------------------
# UPDATE 2026-08-08 (FIX_CI_ROUND2_PMM_2026_08_08).  The first bullet above was
# WRONG, and ubuntu CI proved it: "the instrument still names it" is a FALSE
# ALARM once the repair exists.  On two images the armed guard reported
# "2 / 4 GROWING mode(s) in the FORWARD set ... within a factor 1.05 / 1.44 of
# the cut" on cells whose ANSWER was right (rel 0.0035 / 0.0057 against the
# RCWA anchor) -- because the repair had redirected exactly those modes, and
# the verdict was reading the pre-repair state.  Whether the pre-repair state
# is empty is a BLAS-reduction-order fact, so the old reading could not be
# asserted anywhere.  TWO changes, both measured in this file's docstrings:
#
# * :func:`_mode_cut_verdict` channel A now reads the RESIDUAL
#   (:func:`_mode_cut_growth_post`) -- what the SHIPPED forward set still
#   grows.  On the classical family that partitions the cells EXACTLY by the
#   RCWA-anchored error, which the raw reading never did;
# * :func:`_forward_growth_flip` drops the cut's decade on a PROVABLY PASSIVE
#   layer, where a growing forward mode is a contradiction at any distance.
#   That closes the two cells `FIX_UNION_GRID_2THREAD_2026_08_06` S9 item 3
#   left open (ns=2, degrees 18 and 20, survivors at 15.8-23.6 x the cut).
#
# The CENSUS still carries the raw ``n_grow`` next to ``n_grow_post``, so every
# calibration table in the M3 audit still reads as written.
# ---------------------------------------------------------------------------
```

### L1630-1636 -- `_mode_cut_verdict` docstring, channel A -- what channel A read until 2026-08-08 and why it stopped

*Left in the source:* why the residual and not the diagnosis, stated as a live property of the code, plus the whole measured table below it.

```text
    **A reads the RESIDUAL, not the DIAGNOSIS** (2026-08-08,
    ``FIX_CI_ROUND2_PMM_2026_08_08.md``).  Until then it read
    :func:`_mode_cut_growth` on the RAW ``prop``/``q`` -- what the BARE
    selector would have done -- which was right while the repair did not exist
    and became a FALSE ALARM once it did: the repair redirects those modes, so
    the answer is right and the guard was still telling the reader it might be
    "UNITARY BUT WRONG".  Measured on the M2 coated taper, guard armed,
```

### L1814-1816 -- `_sem_modes` docstring, the ``robust`` parameter -- that audit M10 corrected this docstring's own earlier claim

*Left in the source:* the live contract: ``robust`` is accepted and ignored, the selector has been unconditional since v5.14, and ``robust=False`` does NOT restore the legacy branch.

```text
    ``robust`` is ACCEPTED AND IGNORED (audit M10 2026-07-25 corrected this
    docstring, which used to describe it as selecting the branch): the
    NOISE-ROBUST forward selector has been UNCONDITIONAL since v5.14
```

### L2040-2045 -- `_assemble_jones_farfield`, the incidence guard -- audit M3 -- the ``abs(kz_inc) < 1e-9`` test this replaced, and the negative totals it let through

*Left in the source:* the same hazard in present tense, because a 'simplification' back to ``abs()`` re-opens it.

```text
    # TWO-SIDED + non-finite-aware (audit M3 2026-07-25): the former
    # ``abs(kz_inc) < 1e-9`` accepted a NEGATIVE kz_inc, which is exactly what a
    # GAIN superstrate produces (``_kz_forward`` takes its Re < 0 root), and
    # every efficiency is then silently NEGATED (measured tot = [-0.95, -0.82]
    # through the classical PMMStack cascade).  ``not (kz_inc > 1e-9)`` covers
    # grazing, negative AND NaN in one comparison, for all five callers.
```

### L2205-2215 -- `_scalar_farfield_RT`, the absorbing-superstrate incident flux -- W7 F-B -- the historical ``Re(kz_inc/eps_sup)`` recipe and the TE/TM drift it measured

*Left in the source:* why that recipe is wrong, in present tense (it is the flux of no wave, and it breaks TE == TM at normal incidence), with the measured drift kept as the size of the error.

```text
            # W7 F-B (2026-07-26): for an ABSORBING SUPERSTRATE the historical
            # ``Re(kz_inc/eps_sup)`` mixed gauges -- a REAL kz_inc (already
            # ``Re`` of the complex order-0 root) divided into a COMPLEX
            # eps_sup, which is the flux of no wave, while the NUMERATORS use
            # the full complex kz.  It broke the hardest symmetry there is: at
            # NORMAL incidence on an ISOTROPIC slab, TE and TM must be
            # identical, and they were not (measured T drift 1.4e-4 at
            # Im(n_sup)=0.01, 3.4e-3 at 0.05, 5.8e-2 at 0.2), so this ONE
            # recipe disagreed with every other far field in the family
            # (rcwa ``_project_efficiency``, ``_assemble_jones_farfield``, the
            # 2-D/conical sites) on the SAME physical problem.
```

### L2223 -- `_scalar_farfield_RT`, the gauge-corrected flux -- the "Post-fix:" framing on the parity measurement

*Left in the source:* the measurement itself, which is the live cross-engine check.

```text
            # Post-fix: rcwa parity 1.3e-14, TE == TM at normal 2.7e-15.
```

### L2702-2708 -- `_GEO_EIG_CACHE` -- "It used to be a 64-entry ``OrderedDict`` keyed on the FULL operator bytes" and the ~180 MB it retained

*Left in the source:* the sizing argument for the digest key and the byte budget, and the reason the entry COUNT is deliberately uncapped.

```text
#: It used to be a 64-entry ``OrderedDict`` keyed on the FULL operator bytes,
#: which at a production ``n_glob`` = 300 is a 1.4 MB complex128 KEY beside a
#: ~1.4 MB value -- up to ~180 MB retained with nothing bounding it, while the
#: sibling ``_PERLAYER_GEO_CACHE`` next door was enrolled.  A 32-byte digest
#: and the shared budget fix both halves; the entry COUNT is no longer capped
#: because bytes, not entries, are the resource being protected, and the
#: collective ceiling caps those.
```

### L2722 -- `_geo_eig_key` docstring -- "in the same sense the old full-bytes key was"

*Left in the source:* the exactness claim, stated against a full-bytes key in general.

```text
    the old full-bytes key was -- two pencils collide only on a 256-bit hash
```

### L3211-3216 -- `_stabilize_scalar` docstring, the lossless skip -- "so the gate was contradicting the contract stated one call earlier.  Measured pre-fix, ..."

*Left in the source:* the contradiction as a property of the gate, and the measured false refusal it produces.

```text
    construction ... treat the sums as indicative"), so the gate was
    contradicting the contract stated one call earlier.  Measured pre-fix,
    ``pmm_efficiency_1d`` raised ``RuntimeError: no resonance-free
    solve in degrees [10, 26); the requested degree sits in a high-degree
    resonance band`` for EVERY degree from ``Im(n_sup) = 0.01`` up, blaming
    the user's degree for a perfectly healthy solve (the ``stabilize=False``
```

### L3251 -- `_stabilize_scalar`, the passive gate -- "the historical one-sided ... test"

*Left in the source:* what a one-sided test certifies, as a property of that test.

```text
        # the historical one-sided ``tot <= 1 + tol`` test certified grossly
```

### L3520-3526 -- `_farfield_order_set` docstring, the traced-wavelength refusal -- "the old code silently fell back" and "Measured pre-fix"

*Left in the source:* the collapse and its silence as the reason for the refusal, with the measured table below untouched.

```text
    value, so the old code silently fell back to ``wl = inf`` -> ``m_prop = 0``
    -> the order set COLLAPSED to the bare ``far_field_orders`` floor, DROPPING
    propagating orders that the NumPy policy includes.

    It was silent in the worst way: un-jitted the value is concrete, so the
    forward answer was bit-exact and only the TRACED evaluation was wrong.
    Measured pre-fix (2-layer stack, degree 24/30, n_sub 1.5, wl 633 nm,
```

### L5355-5358 -- `_MORTAR_RESID_REFUSE` derivation -- a ROUND 4 CORRECTION parenthetical about what this paragraph itself used to say

*Left in the source:* the corrected claim, stated once and in place: the mechanism needs an ASYMMETRIC interface, and with BOTH sides promoted the operand is healthy by five decades (measured at the end of the block, which is where the numbers stay).

```text
#: ASYMMETRIC interface.  (ROUND 4 CORRECTION, 2026-09-11: this paragraph said
#: "whenever ONE side ... is promoted", which a reader takes as "either side".
#: With BOTH sides promoted the operand is HEALTHY -- see the correction at the
#: end of this docstring.)  MEASURED
```

### L5436-5441 -- `_MORTAR_RESID_REFUSE` derivation, the scoping measurement -- the "CORRECTION, ROUND 4 ... not merely to have a promoted side" framing

*Left in the source:* the scoping claim and the whole three-layer-fixture measurement beneath it -- `test_the_promoted_side_bar_is_scoped_to_the_asymmetric_interface` reads this text.

```text
#: **CORRECTION, ROUND 4 (2026-09-11, DEFECT 1 of**
#: ``docs/audits/VERIFY_PMM2D_MORTAR_ROUND3_2026_09_11.md`` **S8).**  The
#: mechanism needs the interface to be ASYMMETRIC -- EXACTLY ONE promoted side
#: -- not merely to have a promoted side.  With BOTH sides promoted (two
#: in-plane layers on different grids, an out-of-plane or slanted layer
#: ELSEWHERE in the stack putting the whole cascade on this form) the operand
```

### L5450-5451 -- `_MORTAR_RESID_REFUSE` derivation, the closing sentence -- "what changes is the prediction a later reader would make from the wording" -- a note about the documentation's own history

*Left in the source:* the statement that nothing shipped changes, and the two gates.

```text
#: What changes is the prediction a later reader would make from the wording,
#: which was wrong by five decades.  Gates:
```

### L6804-6808 -- `_build_generator_metric_oop`, the factor-i fix -- that the legacy real coefficients shared the rcwa generator's defect and agreed with the PRE-fix rcwa OOP results

*Left in the source:* what the assignment is, what it is pinned against, and the measured agreement bar.

```text
        # restored to the historical 1.5e-3 bar).  The legacy real
        # coefficients shared the rcwa generator's defect -- they agreed
        # with the PRE-fix rcwa OOP results for the same reason the
        # circular oracle did -- and gave the same artificially
        # +/- symmetric extraordinary dispersion.
```
