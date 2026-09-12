<!-- lumenairy-history-doc
module: lumenairy/elements/berreman.py
ast_sha256: 6fbc34d8597f3841a3ad5f331915d33bdb63c8ffabea8a655a0d606899553de4
token_sha256: 1e239e73cf88e262a9b9ef5b84c8d242081cac076d304acf0c1fbd3ea2617d79
pre_relocation_lines: 1256
recorded_by: WP-A17 SWEEP-2 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/elements/berreman.py`

This file holds the version-history narrative that used to live in
`lumenairy/elements/berreman.py` -- chiefly the `HISTORY / CORRECTION` block in
front of the generalized S-matrix router, which retracts an earlier revision of
its own text, and the "measured pre-fix" / "this used to take" clauses in front
of the W6 guards.  Each block is reproduced **verbatim** under the source line
it came from in the pre-relocation file.

What did NOT move: the measured two-path agreement figures (3.8e-15 and
1.0e-11), the ACTIVE-MEDIA declined-with-measurement note, the contract-locked
gain behaviour, and the numbers that size the energy tripwire --
`tests/unit/test_niche_audit_w6_berreman.py` is the standing gate on several of
those.

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
| L180-186 | `_split_fwd_bwd` docstring | that the stable sort "reproduces the pre-fix physical partition" and where the two implementations previously forked |
| L216-219 | `_freeze` docstring | "measured pre-fix" on the cache-poisoning demonstration |
| L280-286 | `_checked_angle` docstring | "previously aliased" / "measured pre-fix" on the back-side angle |
| L298-306 | `_check_inputs` docstring | "the FUNCTIONAL entry silently accepted" / "Measured pre-fix" |
| L363-366 | `_check_energy_pols` docstring | "measured pre-fix it returned" on the missing tripwire |
| L492-497 | `_farfield_generalized` docstring | W6 F-4 -- the four inert parameters this function used to take, and the gauge its docstring mislabelled |
| L542-558 | `<module>`, the generalized S-matrix router | a HISTORY / CORRECTION block retracting this block's own earlier "the native cascade is subtly wrong, ~2% off" claim |
| L736-741 | `_farfield_generalized`, the kz gauge bridge | "Measured pre-fix" on the double-conjugation that zeroed T |

---

### L180-186 -- `_split_fwd_bwd` docstring -- that the stable sort "reproduces the pre-fix physical partition" and where the two implementations previously forked

*Left in the source:* the rule and the one place the two implementations could still diverge, which is why they share it.

```text
    keeps them in ascending index order; that reproduces the pre-fix
    physical partition exactly.  The two implementations previously forked
    ONLY in the degenerate (not-exactly-two-forward) fallback -- numpy
    ranked by decay (``argsort(Re gam)``) while JAX kept the flag-then-
    index order -- a divergence unreachable for the physical media tested
    but latent for degenerate bianisotropic inputs.  Aligning on the JAX
    rule removes the fork."""
```

### L216-219 -- `_freeze` docstring -- "measured pre-fix" on the cache-poisoning demonstration

*Left in the source:* the hazard and its demonstration in present tense, plus the precedent it follows.

```text
    enforced it, so a caller that wrote into a returned block silently poisoned
    every later solve (measured pre-fix:
    ``_layer_modes_cached(eps, Kx, Ky)[0][0, 0] = 999`` made the very next call
    return ``Wf[0, 0] = 999``).  No in-tree caller mutates them, so this closes a
```

### L280-286 -- `_checked_angle` docstring -- "previously aliased" / "measured pre-fix" on the back-side angle

*Left in the source:* the aliasing argument and the measured wrong answer, as the reason the guard exists.

```text
    ``Kx/Ky ~ sin(angle)``, so a back-side angle previously aliased BYTE-
    IDENTICALLY onto the supplementary front-side angle ``pi - angle`` and
    returned a plausible, energy-conserving answer for the WRONG geometry
    (measured pre-fix: ``angle = 2.0`` rad gave R = [0.01771, 0.19647], the
    ``pi - 2.0 = 1.1416`` rad answer, and ``angle = 10.0`` rad was accepted too;
    ``angle = pi/2`` and ``angle = nan`` raised a bare ``LinAlgError`` with no
    context).  A TRACED JAX angle skips the guard (the resolver's tracer
```

### L298-306 -- `_check_inputs` docstring -- "the FUNCTIONAL entry silently accepted" / "Measured pre-fix"

*Left in the source:* which guards run here and the measured energy violation the incidence screen prevents.

```text
    ``BerremanStack.add_layer`` already enforced (the FUNCTIONAL entry silently
    accepted ``t = 0`` and ``t < 0``), a finiteness screen on every material
    value (audit P3's NaN-index class), and
    :func:`_require_propagating_incidence` for an evanescent / metallic / GAIN
    incidence half-space.  Measured pre-fix: a metallic superstrate
    ``n_sup = 0.15+3.5j`` at ``theta = 0.5`` returned ``T = [30.78, 30.73]`` -- a
    3000% energy violation -- silently, where ``rcwa_jones_1d`` raises
    ``the incidence half-space is non-propagating``.  ``eps_sup`` is PUBLIC here;
    the guard takes the INTERNAL gauge, so it is conjugated on the way in."""
```

### L363-366 -- `_check_energy_pols` docstring -- "measured pre-fix it returned" on the missing tripwire

*Left in the source:* why the tripwire exists and the two silent answers it catches.

```text
    Berreman was the one solver in the family with NO tripwire: measured pre-fix
    it returned ``R + T = 1.086`` per pol for a lossy superstrate, and
    ``T = [30.78, 30.73]`` for a metallic one, silently, where
    ``rcwa_jones_1d`` raises.
```

### L492-497 -- `_farfield_generalized` docstring -- W6 F-4 -- the four inert parameters this function used to take, and the gauge its docstring mislabelled

*Left in the source:* the gauge statement itself, which is the contract, and the reason the extras are absent.

```text
    W6 F-4 (2026-07-26): this used to take ``(core, eps_sup, eps_sub, Kx, Ky)``
    and reference NONE of the four extras (AST-verified) -- inert parameters that
    invited the reader to think the far field re-derives the flux weights.  The
    docstring also mislabelled the return gauge as INTERNAL; it is PUBLIC (this
    cascade runs on raw public eps end to end, and the returned Jones is pinned
    bit-identical to ``rcwa_jones_1d``'s public-convention Jones)."""
```

### L542-558 -- `<module>`, the generalized S-matrix router -- a HISTORY / CORRECTION block retracting this block's own earlier "the native cascade is subtly wrong, ~2% off" claim

*Left in the source:* the measured two-path agreement, the factor-i correction that makes the native pairing valid, and the two reasons the router is kept -- plus the standing instruction to MEASURE rather than quote a figure from prose.

```text
# HISTORY / CORRECTION (W6 audit, 2026-07-26).  This block used to claim the
# NATIVE cascade above is "subtly wrong" for an out-of-plane tensor at oblique
# incidence -- that "the [W; -V] <-> -lam symmetry the native pairing implicitly
# relies on is BROKEN there, so the reflected amplitudes come out ~2% off".  That
# claim is REFUTED by measurement and is no longer true of this code: the native
# ``_solve_core`` / ``_farfield`` and this generalized cascade agree to 3.8e-15 on
# R, T, jones_r AND jones_t for rotx- / roty- / rotx-then-rotz-rotated biaxial
# slabs at theta in {0.2, 0.5, 0.9, 1.2} and phi in {0, 0.6, 2.4}, and to 1.0e-11
# worst over a 4000-config randomized sweep (1-3 fully-rotated biaxial layers,
# lossy layers, absorbing substrates, n_sup / n_sub in [1, 3.5], all angles and
# azimuths).  The pairing is restored by the 2026-07-14 FACTOR-i correction to
# :func:`_berreman_delta` -- which POST-DATES this router -- so the router is now
# a redundant (but independently cross-validated) second path, not an accuracy
# requirement.  It is KEPT because it is the path pinned against ``rcwa_jones_1d``
# and because the two-path agreement is itself a standing cross-family gate
# (``tests/unit/test_niche_audit_w6_berreman.py``).  Do NOT re-derive the "~2%"
# figure from this comment: measure it.
```

### L736-741 -- `_farfield_generalized`, the kz gauge bridge -- "Measured pre-fix" on the double-conjugation that zeroed T

*Left in the source:* the whole mechanism and the measured T = 0 against the stated reference, as the reason for the conjugation.

```text
    # this cascade holds the PUBLIC eps, so feeding it raw conjugated the value
    # TWICE and returned ``Re(kz) < 0`` for an absorbing half-space -- which the
    # ``Re(kz) > 0`` propagating mask below read as evanescent and SILENTLY
    # ZEROED.  Measured pre-fix: a tilted-director slab on
    # n_sub = 1.5+0.3j at theta = 0.3 returned T = 0.000000 where
    # ``rcwa_jones_1d`` (this path's own stated reference) gives T = 0.930316.
```
