<!-- lumenairy-history-doc
module: lumenairy/propagators/carrier.py
ast_sha256: 7bfcf6241a72d82772548f551ec74b349590cd8b79f74ba3d94ed2e6e6531ac5
token_sha256: 0cba0d449e89b5777fcc2b1c64e5b8b3b0120257d037c923877bfb13e58ce1b7
pre_relocation_lines: 11602
recorded_by: WP-A17 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-12 -- WP-A24: the decentre_fit_frac warning quotes the 2026-09-12 six-point calibration and states the measured ordering (was the 2026-07-29 one, whose on-axis and 1.0 w rows have since crossed over)
re_recorded: 2026-09-12 -- ruff isort combine-as-imports (pyproject.toml, WP-A16 recommendation): aliased import statements from the same module merged into one; the set of bound names is unchanged
re_recorded: 2026-09-13 -- WP-A25: replica_fill={'repeat','zero'} on both focus readouts (what the window holds outside one Bluestein period), _period_out['faithful_samples'] / stage readout_faithful_samples, and the replica refusal's two measured regimes
re_recorded: 2026-09-13 -- WP-A25: replica_fill on both focus readouts, faithful-window publication, the replica refusal's two regimes; seven placeholder-free f-string prefixes removed from the new message
re_recorded: 2026-09-13 -- WP-B4: transport='collins' -- the Collins/ABCD-Fresnel carrier transport evaluated by a separable chirp-Z onto a freely chosen output pitch, its Kelly (2014) sampling guard, the complementary-quadrature selection and the Collins focus readout; the default transport is unchanged
re_recorded: 2026-09-13 -- VERIFY-WP-B4: transport='collins' -- a chain leg weighs the chirp-Z's output-period condition K3 (it has no on_replica of its own) and takes the transfer-function form wherever the chirp-Z is not representable on the lattice the leg returns; collins_k3 and collins_kernel published per stage
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
re_recorded: 2026-09-14 -- Wave 5 item D (handoff 4.4): the warning chain is swept onto lumenairy.elements._lens_kernels.caller_stacklevel().  Every warnings.warn literal stacklevel and every threaded literal below it is retargeted to the computed level, which walks out to the first frame outside the package, so the attribution no longer depends on how deep the warn site sits.  MEASURED before the sweep (validation/probe_known_reds/probe_carrier_attribution.py): the tilt-inert notice named the caller when propagate_carrier_referenced was called directly and named library source when the identical site was reached through carrier_referenced_focus_readout -- 2 of 4 emissions misattributed, 0 of 4 after.  No physics changed.
re_recorded: 2026-09-19 -- H2-2 (audit item 18): ONE xp threaded through the _collins_transport chain (_fft2_pair, _as_c_order, _is_traced; _collins_axis_chirp on bld, _collins_exact_kernel_correction through _tf_phase_to_H); NumPy path byte-identical (84/84 archive-to-archive, both builds)
re_recorded: 2026-09-19 -- Round 2 (VERIFY-WAVE5-HYGIENE2): _exact_dispersion_phase consolidates the three transcriptions of the non-paraxial dispersion (V-D22); _collins_carrier_leg and _collins_input_box run in the field's own namespace so the port reaches the public transport='collins' leg (V-D3); traced scalars, a closed-over jnp constant and the astigmatic 'auto' case are refused or documented by name (V-D11/V-D12/V-D13); the accuracy-keyed near-focus fallback is added behind _GAP_KERNEL_ACCURACY_TAU = None, off by default. NumPy path byte-identical archive-to-archive, 245/245 keys on both builds.
re_recorded: 2026-09-20 -- WP-C3: transport='collins' becomes the default on the three entry points that take it; the chain's focus readout resolves its quadrature with the new _collins_readout_k1 and falls back to the Sziklas readout above K1 = 1, publishing the route on its stage; the leg's transfer-function fallback now calls propagate_carrier_referenced(transport='sziklas') instead of _carrier_step_fast, which fixes an all-NaN collimated leg and gives an astigmatic carrier the fallback it was denied; the three internal call sites name their transport
re_recorded: 2026-09-20 -- WP-C3 round 2: the focus_readout stop-plane keys (standoff / on_focus_containment) SELECT the Sziklas readout instead of being refused on transport='collins', with _FOCUS_READOUT_STOP_PLANE_KEYS naming them and readout_route_reason='stop_plane_key' publishing the resolution; the leg's transfer-function fallback is opened to an inverted frame (A < 0), leaving only A == 0 and a resolved flat reference without one -- exactly the legs the Sziklas transport could never evaluate; the traced refusal names transport='sziklas' as its shortest way out
-->

# Version history -- `lumenairy/propagators/carrier.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/carrier.py` -- the "vX.Y (audit Z): pre-fix this did A,
which was wrong because B, now it does C" blocks.  Each block is reproduced
**verbatim** under the source line it came from in the pre-relocation file, so
`git log -S` on any phrase here still lands on the commit that wrote it.

The dominant pattern in this module was a constant whose comment carried EVERY
derivation it has ever had, newest last: `_FINE_GRID_WORK_ARRAYS` recorded four
(4 -> 16 -> 20 -> 22 -> 24), `_FOCUS_STANDOFF_*` three, `_PARAXIAL_BASE_BYTES`
two.  Only the last of each is the derivation of the value the code uses; the
earlier ones describe values the code no longer holds, and a reader cannot tell
which is which without reading all of them in order.  The source now carries the
CURRENT derivation -- `docs/TESTING_STANDARDS.md` S5 requires a numeric bar to
carry one -- and the superseded ones are here.

The second pattern was a comment correcting an earlier *comment*: "an earlier
cut of this note got wrong ... in both magnitude and DIRECTION", "the old
wording told callers the opposite", "an earlier revision of this note cited a
5.5x pitch split".  Those are notes about the documentation's own history; they
are here, and the source states the corrected rule once, plainly.

Nothing the interpreter executes changed in the move.  The header above records
the SHA-256 of (a) the module's AST with every docstring removed and source
positions ignored, and (b) its `tokenize` stream reduced to NAME/OP/NUMBER/
STRING with comments and docstrings dropped -- both taken from the file as it
stood BEFORE the relocation.  `tests/unit/test_audit2609_a17_history_relocation.py`
re-computes both from the live file on every run, so an edit that changes
behaviour while claiming to be history-only fails there.

Where the rationale is load-bearing for what the code does NOW, the source keeps
a condensed why-comment; those are noted per block below as *Left in the source*.

## Contents

| original line | site | what the block records |
|---|---|---|
| L181-189 | `_FOCUS_STANDOFF_* (fine-zoom leg)` | the two constant-zR defaults (6.0 zR, then 0.8 zR) the extent law replaced |
| L260-266 | `_FOCUS_STANDOFF_WAIST_GROWTH (small-extent branch)` | the 9-cell band where the derived law lost to both constants it replaced |
| L316-323 | `_FOCUS_STANDOFF_* (small-extent branch)` | the before/after matrix for the small-extent branch |
| L502-519 | `_freq_sq_1d_bld / _freq_1d_bld` | D7 -- what the half-integer axis measured before the fix |
| L652-663 | `_GAP_KERNELS` | D4 -- the pre-fix resolution table ('exsct' -> FRESNEL, ...) |
| L1194-1201 | `propagate_carrier_referenced` | D4 -- the collimated branch that called fresnel_tf_propagate unconditionally |
| L1357-1366 | `_carrier_step_fast` | why the private gap_kernel default was realigned to 'auto' |
| L1467-1474 | `_envelope_amp_centroid` | the grid-origin measurement the centroid replaced |
| L3935-3940 | `_tilt_exactness_phase` | the commit that rationalized one sphere and not the other |
| L3989-4019 | `SPHERE_PARAB_CONVERSION_EXACT` | the mis-cited "the untapered swap breaks a coarse chain", and the per-order EE3 table that replaced it |
| L4060-4068 | `_sphere_parab_conversion` | the refuted "while the untapered swap breaks a coarse chain" clause |
| L4070-4125 | `_sphere_parab_conversion` | the on-axis / off-axis taper sweeps and the 2026-08-02 decision to drop it |
| L4238-4248 | `_fourier_upsample_crop` | the raw-pocketfft site this function used to be, and what it cost |
| L4285-4301 | `_fourier_upsample_crop` | the numpy-1.x dtype-parity premise this promotion was written for |
| L4375-4382 | `_crop_about_centre` | "an earlier revision described THIS raise as the protection" |
| L4526-4533 | `_MULTI_CONGRUENCE_* (P3 gate)` | the v5.28 32-order fan that was multiplexed silently |
| L4599-4603 | `_MULTI_CONGRUENCE_* (P3 gate, B.1)` | what the pre-canonical (raw-dispersion) gate was silent on |
| L4687-4694 | `_MULTI_CONGRUENCE_* (P3 gate, floor)` | "an earlier cut of this note got wrong, in both magnitude and DIRECTION" |
| L4709-4719 | `_MULTI_CONGRUENCE_* (P3 gate, operational rule)` | the superseded "score a fan by its order spacing" rule |
| L4851-4953 | `_FINE_GRID_WORK_ARRAYS` | the three superseded work-array counts: 4 (census), 16, 20 and 22 |
| L4972-4993 | `_FINE_GRID_WORK_ARRAYS` | why slope 22 / floor 2.6 GB stopped being an upper bound |
| L5070-5086 | `_FINE_GRID_BASE_BYTES` | the 4.5 GB first cut, and the 2.6 GB that followed it |
| L5128-5135 | `_PARAXIAL_BASE_BYTES` | the first cut that charged the exact leg's floor to a paraxial worker |
| L5385-5395 | `carrier_referenced_exact_focus_readout` | what bounded this grid before n_fine_cap reached it |
| L5507-5516 | `carrier_referenced_exact_focus_readout` | the silent decentred-crop clamp and its measured cost |
| L5735-5748 | `carrier_referenced_exact_focus_readout` | what the readout grid had bounding it before the count cap |
| L6162-6168 | `_group_chief_transfer` | what the lumped-ABCD predictor left on the D1 relay |
| L6228-6241 | `_shift_envelope` | the raw-pocketfft site, and the lapsed numpy-1.x promotion premise |
| L7079-7086 | `DOE chain entries (niche D4)` | the hand-split / hand-fold workflow a DOE entry replaced |
| L7154-7161 | `DOE chain entries (niche D4)` | "an earlier revision of this note cited a 5.5x pitch split for design 121" |
| L8097-8104 | `propagate_traced_carrier_chain` | how the hand-written repro script relates to the orchestrator |
| L8945-8962 | `propagate_traced_carrier_chain` | the v5.29 default flip and the P2 aperture:beam cliff guard |
| L9732-9737 | `propagate_traced_carrier_chain_multi (niche D2)` | the v5.28 multiplexed-fan incident, again |
| L9781-9791 | `propagate_traced_carrier_chain_multi (niche D2)` | the two adversarial-pass corrections to the replica guard |
| L10423-10441 | `_multi_resolve_workers` | the 123-GB-of-a-127-GB-box over-subscription, and the 4.59x disagreement |

---

### L181-189 -- `_FOCUS_STANDOFF_* (fine-zoom leg)` -- the two constant-zR defaults (6.0 zR, then 0.8 zR) the extent law replaced

*Left in the source:* the invariant itself and the 3.4x measurement that refutes ANY constant multiple of the Rayleigh range -- that is what stops the next reader proposing a third one.

```text
# THE CONTROLLING INVARIANT IS THE GRID EXTENT, NOT NA (fix D2, 2026-08-06).
# Two earlier defaults were both CONSTANT multiples of the Rayleigh range --
# ``_BRIDGE_ZR_FACTOR`` (6.0 zR) and then 0.8 zR, the latter justified by an
# "the optimum is NA-independent" argument.  Both are refuted by measurement:
# the optimum moves by 3.4x with the INPUT GRID EXTENT at fixed NA, and at a
# grid half-extent of 2 beam radii the 0.8 zR default was measurably WORSE
# than the 6.0 zR one it replaced (FWHM error 10.58% vs 10.13% against an
# exact discrete paraxial focal-plane oracle; local optimum ~1.7 zR at 4.8%).
#
```

### L260-266 -- `_FOCUS_STANDOFF_WAIST_GROWTH (small-extent branch)` -- the 9-cell band where the derived law lost to both constants it replaced

*Left in the source:* the degeneracy itself -- that below 3.695 beam radii the extent-following law IS a constant -- which is the reason this branch exists at all.

```text
# for every extent and every NA.  So the "derived, extent-following" law was,
# under 3.695 beam radii of grid, one more CONSTANT multiple of the Rayleigh
# range -- 1.732 instead of 0.8 or 6.0 -- and it sat in a contiguous 9-cell
# band (NA >= 0.10, ext 1.5-2.0 on a 6 NA x 9 ext matrix) where it was worse
# than BOTH constants it replaced, by up to 5.1x against 0.8 zR and 3.9x
# against 6.0 zR, and where its worst cell (1.62e-1 at NA 0.05 / ext 1.5) lost
# to 6.0 zR's worst (1.26e-1) by 1.28x.
```

### L316-323 -- `_FOCUS_STANDOFF_* (small-extent branch)` -- the before/after matrix for the small-extent branch

*Left in the source:* nothing: the branch's own derivation is above it, and the acceptance matrix belongs with the fix document that owns it.

```text
# Measured after (6 NA x 9 ext, 6 w0 window, relL2 of |F| vs the oracle; full
# matrix in docs/audits/FIX_V1_V8_2026_08_06.md):
#     geomean  4.20e-3 -> 2.68e-3     worst  1.62e-1 -> 9.12e-2
#     against 0.8 zR   1.79e-2 / 1.07e+0   and 6.0 zR   1.47e-2 / 1.26e-1
# -- so the worst-case half of the headline, which V1 refuted, now holds on the
# EXTENDED grid too, and the 9-cell "worse than both" band drops to 7 cells
# whose worst loss is 1.8x (was 5.1x).  What remains is disclosed in the fix
# doc, not smoothed over.
```

### L502-519 -- `_freq_sq_1d_bld / _freq_1d_bld` -- D7 -- what the half-integer axis measured before the fix

*Left in the source:* the RULE and the N = 5 counter-example, which is what makes it checkable; the measured relL2 of the pre-fix build is the incident record.

```text
# THE OFFSET IS ``N // 2``, NOT ``N / 2`` (defect D7, REVIEW_TRACED_EXACT
# 2026-08-05; fixed 2026-08-06).  Both builders return the FFTSHIFTED
# (centred) frequency axis and every caller un-shifts it with ``ifftshift``,
# so the values must be exactly ``fftshift(fftfreq(N, d))`` -- i.e. the
# INTEGER bins ``(j - N//2)``.  The historical ``- N / 2`` is the same number
# for EVEN ``N`` (``N/2 == N//2`` exactly, so the even path -- the whole
# validated surface -- is bit-identical), but for ODD ``N`` it is
# half-integer, and ``ifftshift`` of a half-integer axis is NOT ``fftfreq``:
#
#     N = 5, d = 1:  ifftshift(old) = [-0.1, 0.1, 0.3, -0.5, -0.3]
#                    fftfreq(5)     = [ 0.0, 0.2, 0.4, -0.4, -0.2]
#
# so every transfer function built on it multiplied the wrong spectral bin.
# Measured consequence before the fix: ``_exact_tf_2d_xp`` vs the NumPy
# ``_exact_envelope_tf_step`` relL2 = 1.239 at N = 65 and 0.721 at N = 127
# (order unity), and ``_fresnel_tf_2d_xp`` vs ``fresnel_tf_propagate`` the
# same, against 4e-16 at N = 64 / 128.  ``N = 1`` was also wrong (``[-0.5]``
# for the single DC bin, which must be ``[0.0]``).
```

### L652-663 -- `_GAP_KERNELS` -- D4 -- the pre-fix resolution table ('exsct' -> FRESNEL, ...)

*Left in the source:* why the vocabulary is CHECKED rather than resolved by an if/elif chain: the catch-all last arm is the natural shape here and it is what buys the paraxial kernel back.

```text
# Defect D4 (REVIEW_TRACED_EXACT_2026_08_05; fixed 2026-08-06).  The kernel was
# resolved by an if/elif chain whose LAST arm was an unguarded catch-all, so
# every value that was not literally 'auto' or 'exact' selected the PARAXIAL
# kernel.  Measured before the fix, on ``propagate_carrier_referenced``:
#
#     'auto'    -> EXACT      'exsct'  -> FRESNEL   (dist_to_fresnel = 0.0)
#     'exact'   -> EXACT      'EXACT'  -> FRESNEL
#     'fresnel' -> FRESNEL    None / 1 / ''  -> FRESNEL
#
# i.e. a typo, a capitalisation, or an uninitialised variable silently bought
# back the paraxial gap transport this campaign exists to remove -- the same
# defect class as the ``on_readout_windo`` typo fixed under niche C1.
```

### L1194-1201 -- `propagate_carrier_referenced` -- D4 -- the collimated branch that called fresnel_tf_propagate unconditionally

*Left in the source:* the rule and why this branch is the worst place to be silently paraxial; the exact-vs-fresnel difference measured on the pre-fix branch is the incident record.

```text
    # D4 (2026-08-06): this branch used to call ``fresnel_tf_propagate``
    # UNCONDITIONALLY, so ``R = +/-inf`` ran the PARAXIAL kernel and dropped
    # ``tilt`` whatever ``gap_kernel`` said -- measured, the exact-vs-fresnel
    # difference on a collimated leg was exactly 0.000e+00 against 1.3e-05 on
    # the same leg at R = -0.2 m.  It is the worst place to be silently
    # paraxial: ``m == 1`` means NO frame rescaling, which is the one regime
    # where the exact kernel is genuinely exact (validated to 1e-12 against an
    # independent ASM oracle) and where it composes across splits perfectly.
```

### L1357-1366 -- `_carrier_step_fast` -- why the private gap_kernel default was realigned to 'auto'

*Left in the source:* the default and the fact that it matches every public entry point; the trap it closed is history.

```text
    ``gap_kernel`` DEFAULT: 'auto', matching every public entry point.  It was
    left at 'fresnel' when the public default flipped (2026-08-05), which made
    the private default silently PARAXIAL while every public path was exact --
    the same silent-fallback disease as D4, one level down.  Nothing in the
    library relied on it (all three call sites pass the argument explicitly), so
    aligning it changes no shipped physics; what it removes is the trap that a
    future internal call omitting the argument would quietly run paraxial.  It
    also restores the meaning of
    ``test_carrier_referenced::test_near_focus_landing_fast_path_unchanged``,
    which compares a DEFAULTED public call against a DEFAULTED private one and
```

### L1467-1474 -- `_envelope_amp_centroid` -- the grid-origin measurement the centroid replaced

*Left in the source:* what the centroid is for (a decentred beam reads its own width, not sqrt(2 x_c^2 + w^2)) and the sub-pixel snap contract.

```text
    Verifier round 2 (2026-08-06, sibling of V3): the standoff resolver and
    the near-focus bridge gate measured the beam about the GRID ORIGIN, so a
    decentred beam read ``sqrt(2 x_c^2 + w^2)`` -- 2.34x too wide at a 1.5 w
    decentre -- and resolved a 6.3x shorter leg (hence a 6.3x shorter
    Bluestein period).  Measuring about the centroid fixes that; the
    sub-pixel snap keeps every effectively-centred call on the exact
    ``centre == (0.0, 0.0)`` short-circuit of
    :func:`_envelope_amp_radius`, so the on-axis universe stays
```

### L3935-3940 -- `_tilt_exactness_phase` -- the commit that rationalized one sphere and not the other

*Left in the source:* the rule -- BOTH terms are rationalized -- and the measured cancellation floor a non-rationalized tilted term puts back.

```text
    # RATIONALIZED, both terms.  a185cfc removed ``sqrt(r^2+R^2) - |R|`` from
    # _exact_sphere_eikonal but not from here, so ANY nonzero tilt put the
    # whole k0*eps*|R| cancellation floor straight back into the carrier the
    # rationalization had just cleaned (VERIFY_ARCHITECTURE B14/P2-7:
    # measured 2.11e-11 rad at |R| = 50 mm, indistinguishable from the
    # pre-fix column, while the untilted path read 6.3e-17).
```

### L3989-4019 -- `SPHERE_PARAB_CONVERSION_EXACT` -- the mis-cited "the untapered swap breaks a coarse chain", and the per-order EE3 table that replaced it

*Left in the source:* the headline number and a pointer.  The paragraph above states why the taper was a Nyquist guard and not physics, which is the argument for the flag's default; the evidence for dropping it is a record.

```text
#:
#: **The counter-evidence on record was a mis-citation.**  This function's
#: docstring said "the untapered swap breaks a coarse chain", sourced to
#: ``AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24`` S6.6.  That audit measured the
#: opposite: *"The taper worked as designed -- stage traces identical to the
#: whole-grid swap to 4 digits, i.e. the guard band truly carries nothing -- so
#: the breakage is in-band and intrinsic, not an aliasing artifact."*  What
#: broke that chain was the CONVERSION ITSELF in the pre-``ray_density`` era
#: (its "window 77.5 % -> 7.1 %" is the spot walking out of a narrow readout,
#: which the same paragraph says), and the conversion has been the shipped
#: default since v5.29.  Re-derived in
#: ``docs/audits/D121_FINAL_CLOSURE_2026_08_02.md`` S4.
#:
#: Measured on design 121 (six post-DOE groups, ``ray_subsample=4``,
#: ``RN=1024``), EE3 against the exact-ray + Rayleigh-Sommerfeld oracle at the
#: chain's group-5 exit, read out against the exact eikonal:
#:
#: .. code-block:: text
#:
#:     order      taper ON   taper OFF   d      oracle (true ceiling)
#:     (0,0)       89.662     90.693   +1.032        90.742
#:     (-4,0)      89.385     90.342   +0.957        90.928
#:     (-4,-2)     88.904     89.900   +0.996        90.023
#:
#: and on the PRODUCTION path (``final_leg='exact'``, exact Bluestein readout,
#: N=2048/NFC=8192/WF=4.0), where the last group's own conversion happens on
#: the FINE retrace grid and is inert either way, the residual gain is small
#: but real and in the same direction: BEST-FOCUS[peak] ``dz=0``
#: **3.450 um / EE3 90.2 -> 3.350 um / EE3 90.3**, peak +0.8 %, no plane of the
#: +-80 um through-focus scan worse.
#:
```

### L4060-4068 -- `_sphere_parab_conversion` -- the refuted "while the untapered swap breaks a coarse chain" clause

*Left in the source:* what ``T(r)`` IS and the measurement that the guard band carries nothing. The clause that moved is a claim the flag note above re-derived as a mis-citation -- exactly the stale-comment class the audit flags in sec. 15.7.

```text
    ``T(r)`` is a ``cos^2`` roll-off from ``0.75*r_safe`` to
    ``r_safe = (|R|^3 * lambda / dx)^(1/3)`` -- the radius beyond which the
    DIFFERENCE term itself exceeds the grid's Nyquist slope, so a whole-grid
    swap would scatter aliased guard-band junk into the beam (measured: the
    tapered and whole-grid conversions agree to 4 digits on the design-121
    stages, i.e. the guard band truly carries nothing, while the untapered
    swap breaks a coarse chain).  ``w_beam`` (optional) enables a warning when
    the taper reaches into the beam (``r_safe < 2*w_beam``), where the
    representation would be mixed exactly where the amplitude matters.
```

### L4070-4125 -- `_sphere_parab_conversion` -- the on-axis / off-axis taper sweeps and the 2026-08-02 decision to drop it

*Left in the source:* the one fact a caller still needs -- the taper is off, and turning it back on costs a measured 1.41 EE3 points on a tilted congruence.

```text
    **The taper's mixed-convention skirt is a MEASURED NULL on design-121
    ON AXIS -- and a measured 1.41 EE3 POINTS on a tilted congruence.**  Read
    the two paragraphs below together; the first was validated on axis only and
    said so nowhere until 2026-07-31.

    *On axis (S12).*  Audit AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S8.6
    attributed "the residual 9 % of Strehl beyond r > 1.5 w" partly to this
    skirt; direct measurement refutes that.  Scaling ``r_safe`` by 1.5 and by
    INFINITY (T == 1, i.e. the whole-grid swap with no taper at all)
    reproduces the shipping design-121 result to the digit -- **at-plane
    3.650 um / 87.3 / 99.3 and best focus 3.550 um / EE3 89.57 / EE6 99.26,
    all of them ON-AXIS metrics** -- in all three runs.  Two reasons: (i) the
    conversion and its inverse are POINTWISE, so ``env = E*exp(-ikS)`` is exact
    at every grid point no matter how the phase slope compares with Nyquist --
    only FFT-based steps that see the RESULT care, and a wider taper makes the
    stored envelope smoother, not rougher; (ii) geometrically the taper barely
    reaches the beam on the planes that paragraph looked at -- the onset
    ``0.75*r_safe`` sits at 2.73 w (first entrance), 3.60 w (S21-S22 exit) and
    2.07 w (S23-S24 exit), and ``r_safe`` exceeds the whole grid on the fine
    retrace leg.  The S8.6 skirt was really the
    ``preserve_input_phase='remap'`` ray-lattice alias (see
    ``apply_real_lens_traced``'s ``remap_sampling``).

    *Off axis (2026-07-31, docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31
    S2).*  On design 121's WORST DOE order, (-4,-2) at 51.5 mrad, measured end
    to end through the exact readout against the landed niche-C6 launch:

        r_safe x 0.5     dEE3  -41.62   (EE3 46.15 %, P_tile -23.06, 2 fold
                                         caustic warnings)
        r_safe x 1       --     (shipped, EE3 87.771 %)
        r_safe x 2       dEE3   +1.4147
        no taper (T==1)  dEE3   +1.4147

    The response is MONOTONE and SATURATING: x2 and T==1 agree to four decimal
    places in EE3, EE6, ``P_tile`` and ``exit_power_above_nyquist``, i.e. at
    twice the radius the taper no longer touches anything the result depends
    on, and the optimum is NO TAPER.  So the taper is not doing something
    different off axis -- it is doing the SAME thing at a radius that is too
    small once the congruence is tilted, and the beam pays for the
    mixed-convention annulus.  The geometry: the onset sits at **1.64 w and
    1.63 w on the last two planes** with **5.0e-03 and 5.7e-03 of the envelope
    power beyond it** -- 25x the "~2e-4 of the power ever sees a mixed
    convention" the on-axis paragraph above concluded from a plane list that
    omits them.  The "at most ~2e-4" figure is retracted for tilted
    congruences.

    TAKEN, 2026-08-02 (niche C9): the default IS now ``T == 1``.  The one thing
    that stood against it -- "the untapered swap breaks a coarse chain" -- was
    re-derived and is a **mis-citation of a measurement that says the
    opposite**; see :data:`SPHERE_PARAB_CONVERSION_EXACT` for the source
    quotation, the per-call census that localises the effect, the per-order EE3
    table and the production acceptance.  The 2026-07-31 sweep above is
    reproduced by that work at the same sign and a comparable magnitude
    (+1.03 / +0.96 / +1.00 points at (0,0) / (-4,0) / (-4,-2) on the
    post-C8 tree), and the ``r_safe x 0.5`` cliff is unchanged -- it is the
    same monotone axis, read at its other end.
```

### L4238-4248 -- `_fourier_upsample_crop` -- the raw-pocketfft site this function used to be, and what it cost

*Left in the source:* which dispatcher it uses and why this site matters (twice per exact final leg at 8192-16384 square); the wall-clock share is a record.

```text
    FFT BACKEND (FIX_PERF_ROUND2_2026_08_10 item 1; AUDIT_TRACED_SPEED sec 5,
    row 5 of its ranked table).  The transform pair below used to be RAW
    ``np.fft.fft2`` / ``np.fft.ifft2``, i.e. single-threaded pocketfft, on the
    one shape in the whole chain where it matters -- this function runs TWICE
    per exact final leg at the FINE grid (retrace + readout), which is
    8192-16384 square.  Every other transform in the library goes through the
    :func:`_fft2` / :func:`_ifft2` dispatcher (pyFFTW with a cached plan and
    ``FFTW_THREADS`` threads, scipy.fft next, numpy last), so this site was the
    only large FFT paying a single core.  MEASURED on the design-121 fan order
    at ``n_fine_cap=8192``: the raw-pocketfft leaves under this function were
    2.51 % of the order's wall and the whole function 3.69 %.
```

### L4285-4301 -- `_fourier_upsample_crop` -- the numpy-1.x dtype-parity premise this promotion was written for

*Left in the source:* what the promotion DOES on numpy >= 2 -- it lands the non-complex64 dtypes on complex128 -- and the fact that both transforms of a complex64 pair run in single precision.  The superseded premise is the stale half.

```text
        # DTYPE PARITY with the raw ``np.fft`` this replaced.  Promote here so
        # a non-complex128 caller keeps the historical output dtype instead of
        # silently acquiring a narrower one (the shipped chain is complex128,
        # where ``asarray`` is a no-op and no copy is made).
        #
        # CORRECTION 2026-09-11 (VERIFY_LENS_BANDED_COMPLEX64_2026_09_10 D3):
        # the parity this was written for -- "numpy's FFT is double-only and
        # returns complex128 for EVERY input dtype, while the dispatcher's
        # pyFFTW / scipy backends preserve complex64" -- has not held since
        # numpy 2.0, which has a single-precision FFT: ``np.fft.fft2`` of a
        # complex64 array RETURNS complex64.  So on numpy >= 2 every backend
        # preserves complex64 and this promotion is what makes the OTHER
        # dtypes (real, float32, complex256) land on complex128, which is
        # still the historical answer for them.  The complex64 branch below is
        # therefore not a narrowing of a complex128 transform: BOTH transforms
        # of the pair run in single precision.  MEASURED and accepted -- see
        # the ``_cdt`` note.
```

### L4375-4382 -- `_crop_about_centre` -- "an earlier revision described THIS raise as the protection"

*Left in the source:* which guard a caller actually sees, and that this raise is a defensive invariant rather than that guard.

```text
    NOTE (2026-07-29 adversarial verification): that raise is a DEFENSIVE
    invariant, not the guard a caller sees.  Its only shipped caller,
    :func:`carrier_referenced_exact_focus_readout`, bounds ``n_crop`` by what
    fits at ``(x0, y0)`` BEFORE calling here, so this branch is unreachable
    from it; the user-visible guard for the same failure is that function's
    ``on_readout_window``, which measures the power the bound actually
    truncates.  An earlier revision described THIS raise as the protection,
    which it was not -- the clamp was silent.
```

### L4526-4533 -- `_MULTI_CONGRUENCE_* (P3 gate)` -- the v5.28 32-order fan that was multiplexed silently

*Left in the source:* the failure CLASS in one sentence -- a populated, credible, wrong answer -- because that is what the gate is calibrated against.  The per-frame numbers are the incident report.

```text
# The failure this exists for is a PLAUSIBLE-LOOKING WRONG ANSWER.  At v5.28
# the design-121 32-order Dammann fan was pushed through
# :func:`propagate_traced_carrier_chain` MULTIPLEXED and produced a populated,
# credible-looking frame lattice whose per-frame power was scrambled
# (0.47 +/- 0.51 % against a design 2.78 %/frame, uniformity ~0.996).  Nothing
# raised and nothing warned, even though ``apply_real_lens_traced``'s
# entrance->exit map names exactly that case -- "comparable-power beams at
# well-separated angles (post-DOE at large split)" -- as EXCLUDED.
```

### L4599-4603 -- `_MULTI_CONGRUENCE_* (P3 gate, B.1)` -- what the pre-canonical (raw-dispersion) gate was silent on

*Left in the source:* nothing here: the sqrt(dx) law above and the canonical table below are the derivation, and they say why the raw reading cannot be used.

```text
# The pre-canonical gate was therefore SILENT on design 121's own 32-order fan
# at dx0 = 0.25 um / N = 8192 -- the exact production condition roadmap P4
# names as the original F-B evidence matrix's worst row -- while the multiplexed
# answer stays 36-86 % wrong by the linearity oracle at every pitch.  Detector A
# is blind there by symmetry (residual 1.5e-16 rad).
```

### L4687-4694 -- `_MULTI_CONGRUENCE_* (P3 gate, floor)` -- "an earlier cut of this note got wrong, in both magnitude and DIRECTION"

*Left in the source:* the statement of what the floor is measured BETWEEN (interfering pairs) and the measured fan-to-pair mapping that follows it.

```text
# The floor is stated in the angle between INTERFERING PAIRS.  Mapping a FAN
# onto that pair scale is the part an earlier cut of this note got wrong, in
# both magnitude and DIRECTION: it claimed the score is "set by the finest
# fringes, i.e. by the nearest-neighbour order spacing", so that a dense fan
# would hide far below its span.  It does not.  Re-measured with the shipped
# helper (``_chain_entry_congruence_stats``; the harness reproduces the 8x8
# row below to 3 digits, so this is the same measurement, not a competing one):
#
```

### L4709-4719 -- `_MULTI_CONGRUENCE_* (P3 gate, operational rule)` -- the superseded "score a fan by its order spacing" rule

*Left in the source:* the corrected rule, its two concrete verdicts and the test that pins the boundary.

```text
# OPERATIONAL RULE, corrected: score a fan by its total span, derated ~20 %.
# A fan whose SPAN clears the ~19 mrad floor is caught even when its order
# spacing is far below the floor -- the old wording told callers the opposite,
# and was over-conservative rather than unsafe.  The two concrete verdicts it
# reported still stand on the re-measurement: the 8x8 +-23 fan sits ON the
# cutoff with no margin either way and is not reliably caught (though because
# its SPAN lands there, not its spacing), while the design-121 8x4 fan at
# +-46 / +-23 mrad reads 1.65e-2 / 1.82e-2 / 1.87e-2 and clears by ~2x at every
# pitch.  That boundary is pinned by
# ``test_the_documented_detection_floor_is_a_pinned_boundary`` so a future
# cutoff change cannot move it silently.
```

### L4851-4953 -- `_FINE_GRID_WORK_ARRAYS` -- the three superseded work-array counts: 4 (census), 16, 20 and 22

*Left in the source:* what the number IS (a constrained upper-bound envelope, not a decomposition) and the two method notes a re-measurement needs.  The derivation of the value the code actually holds is kept below.

```text
#
# v5.33.2 (docs/audits/AUDIT_TRACED_MEMORY_2026_08_09.md sec 2.3, 2.5 and
# row 9): this was 4 -- "the Fourier-upsample pad + its inverse transform,
# then the reconstructed field alongside the exact-sphere phasor, then the
# Bluestein zoom's own workspace" -- and that model is 4.0x OPTIMISTIC.  The
# leg does not hold four arrays.  MEASURED by a live big-ndarray census walked
# from ``sys._current_frames()`` at the peak plateau of one design-121 order
# (``RN=1024, RS=4, NFC=16384, WF=4.0, TILE=1024, DXO=0.2 um``, exact final
# leg, serial Newton, ``set_max_ram(105)`` so the grid choice is
# deterministic), at ``n_fine = 16384`` where one complex128 grid is 4.295 GB:
#
#   6 x 4.295 GB  complex128 (16384,16384)  _fine_trace_group_exit:
#                   env_f, E_full, _ph, _cf, _rp, _xf
#   5 x 4.295 GB  complex128 / float64      apply_real_lens_traced:
#                   _unit, E_out, _coords, E_analytic, _rd_resid_map
#  10 x 2.147 GB  float64   (16384,16384)   apply_real_lens_traced:
#                   _pip_remap_W, _ard, _absE, _nan_rd, _a_rd, ard_map,
#                   amp, _mag0, Y, X
#   1 x 0.268 GB  bool      (16384,16384)   apply_real_lens_traced: valid
#   --------------------------------------------------------------------
#   69.26 GB owned across 23 live full-grid arrays
#     = 69.26 / 4.295 = 16.1 complex128-equivalents IN FRAMES ALONE
#      (21.9 including the resident pyFFTW plan buffers; 23.0 against the
#       thread-free peak RSS of 98.85 GB, and 25.7 against the 110.55 GB
#       instrumented peak the census itself was taken inside -- the audit's
#       sec 4.5 observer artefact, which is why the frame-live count and not
#       an RSS ratio is what this constant carries).
#
# The consequence of the old 4 is the point, and it is measured: with
# ``frac = 0.5`` the 4-array model approves ``n_fine = 16384`` whenever
# ~34.4 GB is free, and the run then touches 98.85 GB -- 2.9x -- leaving a
# 137.4 GB box with 18.4 GB.  It also let ``_multi_resolve_workers`` approve
# SIX congruence workers (~484 GB) on a 128 GB box at that cap
# (AUDIT_TRACED_SPEED_2026_08_09.md sec 3.3).  The model being optimistic was
# the only reason the single-order run completed; that is the absence of a
# safety margin, not the presence of one.
#
# 16 was the FRAME-LIVE census rounded to an integer, deliberately NOT the
# 21.9 that includes the plan buffers (those are process-global and shared
# across the legs, so charging them per fine grid would double-count when two
# grids of different size are sized in the same process).  Whoever re-measures
# this: the census method is in the audit's sec 1 -- measure from OUTSIDE the
# process, an in-process sampler thread inflates peak working set by up to
# 2.5x on this workload.
#
# v5.33.3 (docs/audits/FIX_PERF_PARALLEL_2026_08_10.md sec 3) -- 16 -> 20,
# RE-DERIVED FROM A SCALING MEASUREMENT rather than from a census, because a
# census counts arrays at ONE grid and cannot separate what scales from what
# does not.  Peak RSS of a design-121 order was sampled at 1 Hz over the whole
# process at THREE fine grids on this branch, everything else pinned
# (``RN=1024 RS=4 NW=1 DXO=0.2 um NOUT=8192 TILE=1024 WF=4.0 LEG=auto``,
# ``ram_budget=inf`` so the grid choice is the one asked for):
#
#     n_fine    peak RSS      implied count at zero intercept
#      4096      7.123 GB              26.5
#      8192     23.968 GB              22.3
#     16384     84.589 GB              19.7
#
# The count FALLS with the grid, which is the signature of a fixed cost, not
# of a smaller array set; a straight line in ``n_fine ** 2`` fits all three to
# within 3.5 % and gives slope 305.9 B/px = **19.1 complex128-equivalents**
# and intercept **2.6 GB**.  Pairwise slopes are 18.8 / 19.2 / 20.9, so 20 is
# the round-up, and ``_FINE_GRID_BASE_BYTES`` carries the intercept.
#
# The direction matters: the shipped 16 was 1.20x OPTIMISTIC on the term that
# grows, which is the dangerous side of the trade -- ``_multi_resolve_workers``
# priced a NFC=8192 worker at 17.55 GB against a MEASURED 24.97 GB and
# approved FIVE workers on a box that holds three or four
# (AUDIT_TRACED_SPEED_2026_08_09 sec 3.4).
#
# v5.33.3 (VERIFY_PERF_BRANCH_2026_08_10 D4): 20 was the round-up of a
# THREE-POINT, TWO-ORDER, WHOLE-PROCESS fit.  It is not the envelope the
# clamp needs.  A congruence WORKER's own peak -- the quantity an OOM is
# measured against, and the only one observable at k > 1 -- sits ABOVE that
# line at 8192 (26.0 GB against the two-order 23.97), because a process's
# leg-local caches grow with the orders it runs.  Bounding the child from a
# 19.1-slope line therefore forced the INTERCEPT up (2.3 -> 4.5 GB), and an
# intercept is exactly the wrong lever: it over-prices the small end, where
# the whole peak IS the intercept.  At (20, 4.5 GB) the model read 1.476x the
# 4096 worker child measured below -- inside the 1.5x bar this file's test
# declares by 1.6 %, i.e. not reproducible.
#
# So the split is re-derived as what it actually is: a constrained UPPER-BOUND
# ENVELOPE over EVERY measured point (whole-process AND worker-child,
# two-order AND six-order, three grids), not a decomposition of where the
# bytes go.  Slope 22 and floor 2.6 GB is the pair that minimises the worst
# ratio subject to (a) bounding all eleven measured points, (b) keeping at
# least 2 % of margin over the 8192 worker child -- the row the clamp is
# actually decided by -- and (c) not pushing the 16384 price past what this
# box's own pre-flight will approve for one worker.  The floor is the
# three-grid fit's measured 2.3 GB intercept rounded up, and still clears the
# 1.75 GB interpreter-plus-import commit the Newton pool measured
# independently (``_lens_traced._NEWTON_WORKER_BASE_BYTES``).  Measured rows
# and ratios: ``tests/unit/test_niche_d8_congruence_workers.py``'s
# ``_MEASURED_PEAK_BYTES``; worst ratio 1.279 (was 1.476, on the 4096 child),
# tightest bound 1.023 on the 8192 child (was 1.013).
#
# A steeper slope prices the small end better still (23 / 2.0 GB reads 1.232
# worst) but takes ``n_fine = 16384`` from 97.5 GB to 101.2 GB per worker,
# which is where the runners' pre-flight stops approving a SINGLE 16384 worker
# on a ~105 GB-free box.  That trade was made deliberately and this is the
# note that says so.
#
```

### L4972-4993 -- `_FINE_GRID_WORK_ARRAYS` -- why slope 22 / floor 2.6 GB stopped being an upper bound

*Left in the source:* which row the pair is decided by, the two ratios, and the pointer to the measured rows in the test that carries them.

```text
# **The shipped (22, 2.6 GB) split is UNDER the 8192 k=3 worker child on this
# tree -- 26.591 modelled against 26.737 measured, 0.995x.**  It is not a loose
# upper bound any more; it is not an upper bound.  That is the OOM side of the
# trade, and it is why this moved rather than being left alone.
#
# The set did not move one way, which is what forced BOTH constants: the
# small-end child FELL 2.6 % while the binding 8192 child ROSE 2.9 %.  A
# 22-slope cannot absorb that pair -- bounding the k=3 child with 2 % of margin
# at slope 22 needs a 2.9 GB floor, and that floor prices the 4096 child at
# 1.35x, outside the 1.3x bar.  The feasible region starts at slope 23, and
# (24, 1.8 GB) is the integer pair in it that minimises the worst ratio:
# **worst 1.274, tightest 1.045**, an upper bound at all seven points.
#
# The floor is no longer carrying the child excess, and it is smaller for a
# MEASURED reason: a least-squares line through the four whole-process points
# now reads slope 320.5 B/px = 20.03 complex128-equivalents and intercept
# 2.102 GB, i.e. a per-process floor of 2.102 - 0.369 = **1.733 GB** once the
# ``_MULTI_WORKER_GRID_FACTOR`` term for that measurement's own 1024^2 input is
# removed.  1.8 GB is that rounded up, and it still clears the 1.75 GB
# interpreter-plus-import commit the Newton pool measured independently
# (``_lens_traced._NEWTON_WORKER_BASE_BYTES``) -- by 50 MB, which is thin and
# is stated rather than hidden.
```

### L5070-5086 -- `_FINE_GRID_BASE_BYTES` -- the 4.5 GB first cut, and the 2.6 GB that followed it

*Left in the source:* the measurement that decides the constant -- a real congruence worker at k > 1, which is the quantity an OOM is measured against -- and why it sits above a whole-process fit.

```text
# MEASURED THREE TIMES.  The three-grid fit above intercepts at 2.635 GB,
# less the 0.369 GB the ``_MULTI_WORKER_GRID_FACTOR`` term already charges for
# that measurement's own 1024^2 input = 2.3 GB -- but that fit is over runs of
# TWO orders in the PARENT.  A real congruence WORKER, which is what this
# constant is for, was then sampled directly at k > 1 (six orders, NFC 8192):
# largest single child **24.21 / 24.20 GiB = 26.0 GB**, i.e. above the
# two-order line.  The difference is the leg's own process-global caches,
# which grow with the number of orders a process runs.
#
# The first cut carried that difference entirely in this constant (4.5 GB),
# which bought a bounded child at the price of a 1.476x over-price at
# ``n_fine = 4096`` -- outside the 1.5x bar once the 4096 WORKER CHILD was
# measured (6.94 GB; VERIFY_PERF_BRANCH_2026_08_10 D4).  The slope carries it
# now (see ``_FINE_GRID_WORK_ARRAYS``), and this constant is back to the
# process floor it names.  What is in it: the interpreter-plus-import commit,
# the order table and chain-A output the process carries across chain B, and
# the process-global FFT plan / chirp caches the leg leaves behind (byte-capped
```

### L5128-5135 -- `_PARAXIAL_BASE_BYTES` -- the first cut that charged the exact leg's floor to a paraxial worker

*Left in the source:* the reason a SECOND floor exists: the exact leg's is a design-121-class exact-leg figure, and a paraxial worker builds no fine grid.

```text
# ``_FINE_GRID_BASE_BYTES``'s envelope note says in as many words that it is a
# design-121-class EXACT-leg figure.  The first cut charged it to every worker
# anyway, including ``final_leg='paraxial'`` workers, whose whole point is
# that no fine grid is built: on a box with 16 GB free that took a paraxial
# multi-congruence run from 21 approved workers to ONE, on the strength of a
# floor measured on a six-order exact-leg congruence.  A throughput
# regression, not a wrong answer -- but exactly the shape the envelope note
# exists to prevent.
```

### L5385-5395 -- `carrier_referenced_exact_focus_readout` -- what bounded this grid before n_fine_cap reached it

*Left in the source:* the quadratic-in-window_factor sizing and the measured 4x, which is what the cap is for.

```text
        v5.33.2, audit ``AUDIT_TRACED_MEMORY_2026_08_09`` row 10 -- one of that
        audit's two UNSAFE rows.  This grid's size is quadratic in
        ``window_factor`` (its window is ``window_factor * w_exit``) and until
        now NOTHING bounded it but the RAM clamp, whose cost model was 4.0x
        optimistic.  MEASURED on the design-121 production order: ``wf = 4``
        gives ``N_fine`` 8192 (4.295 GB/array) and ``wf = 7`` gives 16384 --
        4x the memory for the same physics.  ``propagate_traced_carrier_chain``
        forwards its ``focus_readout['n_fine_cap']`` (default 16384) here, so
        the production path is bound by the number that already bounds its
        re-trace leg; a DIRECT caller who passes nothing keeps the uncapped
        behaviour.
```

### L5507-5516 -- `carrier_referenced_exact_focus_readout` -- the silent decentred-crop clamp and its measured cost

*Left in the source:* the geometric bound and the measured decentred failure -- 0.279 of the power at 0.435 of the peak with an empty warning list -- because that is the bar ``readout_window_tol`` is set against.

```text
        The crop is necessarily bounded by what the grid holds -- at a chief
        ray ``(cx, cy)`` only ``N*dx - 2*max(|cx|, |cy|)`` is available -- and
        until v5.32.1 that bound was applied SILENTLY, so a decentred readout
        degraded with no symptom while the beam still sat comfortably on the
        grid.  Measured (1024 x 0.5 um grid, Gaussian ``w`` = 40 um,
        ``R`` = -400 um, ``z`` = 400 um, ``window_factor`` = 6) against a plain
        :func:`~lumenairy.propagators.mft.angular_spectrum_propagate_mft` on
        the same input grid: ``cx`` = 0 and 150 um agree to 3e-5 / 3e-4 of the
        peak, ``cx`` = 200 um returns 0.919 of the power at 0.906 of the peak,
        and ``cx`` = 230 um returns **0.279 of the power at 0.435 of the
```

### L5735-5748 -- `carrier_referenced_exact_focus_readout` -- what the readout grid had bounding it before the count cap

*Left in the source:* the cap ORDER (count cap, then RAM clamp) and that it matches _fine_trace_group_exit -- the two have to agree.

```text
    # v5.33.2 (AUDIT_TRACED_MEMORY_2026_08_09 row 10, one of the audit's two
    # UNSAFE rows): the COUNT cap the re-trace leg has always honoured, applied
    # here too and BEFORE the RAM clamp -- the same order as
    # ``_fine_trace_group_exit`` (``min(n_fine_req, n_fine_cap)`` then
    # ``_memory_bounded_n_fine``).
    #
    # Until now this grid had no count cap at all.  Its sizing is quadratic in
    # ``window_factor`` (the window is ``window_factor * w_exit``), so the ONLY
    # thing between it and an OOM was the RAM clamp -- whose cost model was
    # itself 4.0x optimistic (see ``_FINE_GRID_WORK_ARRAYS``).  MEASURED on the
    # design-121 production order: ``wf = 4`` lands N_fine = 8192 (4.295 GB per
    # working array) and ``wf = 7`` lands 16384, i.e. 4x the readout's memory
    # for the same physics, with nothing bounding it.  The exposure was latent
    # rather than realised at the two configurations the audit measured, which
```

### L6162-6168 -- `_group_chief_transfer` -- what the lumped-ABCD predictor left on the D1 relay

*Left in the source:* the current statement: the chief ray is TRACED, so the residual against an exact trace is machine zero at any angle.

```text
    So the predictor is not linearised at all any more: the chief ray is
    TRACED, through the group's own surfaces, with the same engine the tests
    use as their oracle.  Measured on that fixture the residual against the
    exact trace goes ``0.1214 um -> 0.0`` (machine precision).  It is exact at
    ANY angle, so the ``z L^3 / 2``-class error simply does not arise: on the
    D6 synthetic stand-in (``L = -0.20``) the old predictor sat 12.4 um from
    the Fermat focus while the exact leg's spot landed ON it.
```

### L6228-6241 -- `_shift_envelope` -- the raw-pocketfft site, and the lapsed numpy-1.x promotion premise

*Left in the source:* the dispatcher, the accuracy statement and the UNCONDITIONAL promotion, which is what this function actually does.

```text
    FFT BACKEND (FIX_PERF_ROUND2_2026_08_10 item 4a).  The transform pair was
    RAW ``np.fft``, i.e. single-threaded pocketfft, and it runs on the exact
    leg's FINE grid through :func:`_crop_about_centre` -- MEASURED at 1.37 % of
    a design-121 fan order's wall at ``n_fine_cap=8192``, which made it the
    second-largest raw-``np.fft`` site after
    :func:`_fourier_upsample_crop`.  Same dispatcher, same accuracy statement
    (bounded at FFT round-off, NOT bit-identical -- see that function's note
    and ``FIX_PERF_ROUND2_2026_08_10.md`` sec 5), and the same dtype-parity
    promotion, which here is UNCONDITIONAL: this transform pair always runs in
    complex128 and the result is narrowed back to the input's dtype on return.
    (2026-09-11, D3: the historical justification for the promotion -- that
    numpy's FFT was double-only -- lapsed at numpy 2.0, but the promotion
    itself is what this function does and is left as it is; the crop, which
    does NOT promote a complex64 input, carries the measured cost note.)
```

### L7079-7086 -- `DOE chain entries (niche D4)` -- the hand-split / hand-fold workflow a DOE entry replaced

*Left in the source:* why the fold is the error-prone step -- it is the reason ``gap_before`` is charged to the PRE-DOE angle and ``gap_after`` to the POST-DOE one, which is the bookkeeping this section goes on to define.

```text
# refractive halves.  Until v5.32 the DOE could not be part of the design the
# chain sees at all -- ``DGRATING`` surfaces imported as flat optical surfaces
# with their parameters dropped -- so a consumer had to hand-build the
# grating, hand-split the chain at the DOE plane, and hand-fold the DOE's
# 51.539 mm gap into a neighbouring group's ``gap_before``.  That manual fold
# is the error-prone step: it is only correct for an UNDEFLECTED order,
# because a fold transports the chief ray over the whole folded distance at
# the PRE-DOE angle.
```

### L7154-7161 -- `DOE chain entries (niche D4)` -- "an earlier revision of this note cited a 5.5x pitch split for design 121"

*Left in the source:* the corrected fact, stated once: design 121's DOE is in collimated space, so the near-focus corner does not apply to it.

```text
# NOTE, corrected 2026-07-28: design 121 is NOT such a design.  Its DOE sits
# in COLLIMATED space -- measured R = +703591.2 mm (703.6 m, diverging) at
# the pre-DOE group exit -- so one 58.5393 mm step and a 51.5393 + 7.0000 mm
# pair land on the SAME co-moving pitch (51.23386 um, ratio 1.000000) and
# agree to max|dE|/max|E| = 2.1e-11.  An earlier revision of this note cited
# a 5.5x pitch split for design 121's own leg; that number belongs to the
# near-focus corner in (2), not to this design.  For the 121 the operative
# reason is (1).
```

### L8097-8104 -- `propagate_traced_carrier_chain` -- how the hand-written repro script relates to the orchestrator

*Left in the source:* the fact that the two are different models under the shipping defaults, and which options reproduce the script.

```text
    ``validation/repro_traced_carrier_121/carrier_chain_121.py`` is the hand-
    written form of that pattern.  NOTE (v5.29): the two agree only with the
    LEGACY options -- ``carrier_reference='parabola'`` plus
    ``traced_kwargs={'amplitude_model': 'screen', 'preserve_input_phase':
    True}`` -- because the chain's defaults have since flipped to the validated
    carrier-regime configuration (see ``carrier_reference``).  With the shipping
    defaults this orchestrator is a DIFFERENT (and much more accurate) model
    than that script: design-121 best-focus EE6 79.7% -> 99.3%.
```

### L8945-8962 -- `propagate_traced_carrier_chain` -- the v5.29 default flip and the P2 aperture:beam cliff guard

*Left in the source:* what the chain defaults and the precedence rule (caller kwargs win), plus the measured cliff the fit-domain guard closes.

```text
    # v5.29 default flip (audit AUDIT_TRACED_FROZEN_AMPLITUDE_2026_07_24 S8):
    # the chain's per-group traced calls default to the validated
    # carrier-regime configuration -- the chain ALWAYS operates with its
    # carrier beyond the grid Nyquist, where the geometric (ray-density)
    # amplitude and the geometric residual carry are the correct physics,
    # not preferences.  Anything the caller passes in ``traced_kwargs`` (or a
    # group's own ``traced_kwargs``) WINS over these defaults; the standalone
    # ``apply_real_lens_traced`` element defaults are untouched.
    # P2 (audit AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24 §4): the chain also
    # defaults the APERTURE:BEAM CLIFF GUARD on -- the ray-fit domain is tied to
    # the beam, not to the (arbitrary, prescription-supplied) vignetting
    # aperture.  A chain is exactly the daily-driver case that receives
    # arbitrary apertures, and the cliff is silent: measured on the E4 corrected
    # relay, exit-wavefront Strehl 0.998 (6 mm aperture) -> 0.105 (7 mm) ->
    # 0.039 (10 mm) with no warning and no energy loss to show for it, recovered
    # to 0.9995 at every aperture by this default.  Fit-domain only: no field
    # energy is vignetted (measured identical exit power to 4 digits), and the
    # design-121 acceptance is unchanged.
```

### L9732-9737 -- `propagate_traced_carrier_chain_multi (niche D2)` -- the v5.28 multiplexed-fan incident, again

*Left in the source:* the one-line statement of why one congruence per chain is the contract; the incident numbers are recorded once, at the P3 gate.

```text
# comparable-power beams at well-separated angles.  Pushed through the chain
# MULTIPLEXED that fan produced a populated, credible-looking frame lattice
# whose per-frame power was scrambled (0.47 +/- 0.51 % against a design
# 2.78 %/frame) with nothing raised and nothing warned -- the element's
# entrance->exit map names exactly that case as excluded
# (``_lens_traced.py``, ``carrier``'s validity paragraph).
```

### L9781-9791 -- `propagate_traced_carrier_chain_multi (niche D2)` -- the two adversarial-pass corrections to the replica guard

*Left in the source:* both RULES, without the "an earlier cut did X" framing: sizing from min(period) over all K, and the K == 1 downgrade.

```text
#     ``filterwarnings('ignore')`` silences.  Two corrections a second
#     adversarial pass forced, both about WHOSE window is at risk:
#       - the shared 'auto' window is sized from min(period) over ALL K
#         congruences (measured in a cheap 16-px probe pass), not from
#         congruence 0.  Design-121's per-order periods span 1.8 %, so sizing
#         from the first congruence made the DEFAULT raise on the acceptance
#         config, and made "does it run at all" depend on list order.
#       - the guard is a MULTIPLEXING guard: at K = 1 there is no neighbouring
#         frame to contaminate and the answer is exactly the chain's, so it
#         downgrades to a warning and 'auto' keeps the requested field of view
#         (an earlier cut silently returned zeros over 55 % of a K=1 grid).
```

### L10423-10441 -- `_multi_resolve_workers` -- the 123-GB-of-a-127-GB-box over-subscription, and the 4.59x disagreement

*Left in the source:* both rules: the fine grid is a SECOND peak live at the same time, and the price is CALLED from the readout's own model rather than re-spelled.

```text
    # The EXACT final leg's fine grid is a SECOND peak, on top of the chain
    # working set and live at the same time.  Sizing workers from the chain
    # alone is how 3 workers each correctly decided they could afford a
    # 16384^2 fine grid (17.2 GB) and then collectively asked for 123 GB of a
    # 127 GB box -- MEASURED on design 121's fan, which died with 'Unable to
    # allocate 4.00 GiB for an array with shape (16384, 16384)' while 97 GB
    # still read free.
    #
    # ``_fine_grid_peak_bytes`` is the readout's OWN model (grid term + the
    # per-process floor), called rather than re-spelled: pricing a worker with
    # a second copy of the arithmetic is how this clamp and
    # ``_memory_bounded_n_fine`` came to disagree by 4.59x once already
    # (AUDIT_TRACED_SPEED_2026_08_09 sec 3.3).  ``n_fine_cap`` falsy =
    # ``final_leg='paraxial'``, which builds no fine grid and is therefore
    # priced with the MEASURED paraxial floor rather than the exact leg's --
    # v5.33.3, VERIFY_PERF_BRANCH_2026_08_10 D5: charging the exact leg's
    # design-121-class floor to a paraxial worker took a 16 GB-free box from
    # 21 approved workers to one, against a paraxial worker MEASURED at
    # 0.44-1.17 GB (1.17 being the design-121 fan's own, at k=2).
```

