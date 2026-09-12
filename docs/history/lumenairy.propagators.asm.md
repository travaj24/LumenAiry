<!-- lumenairy-history-doc
module: lumenairy/propagators/asm.py
ast_sha256: 5443d15c195cc9292768294fba219459559fdd111f3fef21f64d04498e7b29fb
token_sha256: a8262a4d4b4a7fe68d6ffa25b5c6825b5a7ffb7a2b73c2f4a690e9e2347f1cd8
pre_relocation_lines: 1405
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
-->

# Version history -- `lumenairy/propagators/asm.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/asm.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

Five blocks moved, and in four of them only the framing did: each was a live
hazard note written as "Pre-fix X happened".  The hazards are all still
reachable -- an uncast real field still latches the pyFFTW shape blacklist, a
complex128 carrier still upcasts the whole tilted pipeline -- so the source
now states them in the present tense, and what moved here is the release
attribution and the "bit-identical to pre-fix" reassurance that went with it.

Three long blocks deliberately did NOT move:

* the **2-shift fold derivation** at the top of the module, with the
  `S * S == 1` algebra, the odd-N measurement (2.1e-16 to 1.1e-15) that gates
  the fold on both axes being even, and the 1.52x / 1.40x timings;
* the **audit-P1 integer DC anchor** note, whose measured `shack_hartmann`
  centroid (-8.0874 px against -0.1535 px) is the reason the anchor is
  `N // 2` and not `N / 2`;
* the **H-cache** and **z == 0 identity** contracts.

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
| L312-324 | `_build_centered_asm_H` -- Notes | the `it was NOT` framing and the byte-identity survey that found the 1-ULP divergence |
| L871-878 | `angular_spectrum_propagate` -- the real-dtype cast | the `Pre-fix` framing |
| L1040-1047 | `apply_fresnel_curvature` -- dtype-follows-input | the `Pre-fix` framing and the byte-identity note for complex128 |
| L1129-1135 | `angular_spectrum_propagate_batch` -- the H fetch | what the pre-fix delegation cost (a wasted full-grid FFT+IFFT pair on garbage data on every batch call) |
| L1295-1304 | `angular_spectrum_propagate_tilted` -- the carrier dtype | the `Pre-fix` framing and the complex128 bit-identity note |

---

### L312-324 -- `_build_centered_asm_H` -- Notes -- the `it was NOT` framing and the byte-identity survey that found the 1-ULP divergence

*Left in the source:* the live instruction (multiply by the reciprocal, the same spelling the shared cache uses), the measured size of the divergence, and why bit-exactness is a contract on this path

```text
    K8 (audit 2026-09-11): it was NOT, for odd ``N`` with
    ``bandlimit=False``.  This builder formed the frequency axis by
    DIVISION, ``(arange(N) - N//2) / (N*dx)``, while
    ``_get_or_make_freq_grids`` multiplies by the reciprocal,
    ``(arange(N) - N//2) * (1.0/(N*dx))``; the two differ by up to 1 ULP
    whenever ``1/(N*dx)`` is not exactly representable.  Measured:
    byte-identical at N=64/dx=1 um and N=256/dx=0.5 um (both have an
    exact reciprocal) and at N=255/dx=0.5 um with ``bandlimit=True``, but
    ``max|dH| = 9.096e-13`` at N=255/dx=0.5 um with ``bandlimit=False``.
    Physically ~1e-12 rad and irrelevant, but "bit-exact" is a
    pinned-bits contract and this builder is the ``shack_hartmann``
    per-lenslet path, so the expression is now the SAME one the shared
    frequency-grid cache uses.
```

### L871-878 -- `angular_spectrum_propagate` -- the real-dtype cast -- the `Pre-fix` framing

*Left in the source:* the entire failure chain the cast prevents -- it is a latch, so the damage outlives the call that caused it

```text
        # v5.17.x (P2-26): cast the real-dtype field to the target complex
        # dtype BEFORE it reaches ``_fft2`` (mirrors the batch sibling).
        # Pre-fix, a real float32/float64 E_in was fed uncast into the
        # pyFFTW dispatcher, which rejects a real->complex in-place plan
        # with ``ValueError: Invalid direction``; the failure handler then
        # permanently blacklisted the bare SHAPE for ALL dtypes (so every
        # later complex128 call at that shape silently ran on scipy) and
        # emitted a misleading 'memory pressure' warning.
```

### L1040-1047 -- `apply_fresnel_curvature` -- dtype-follows-input -- the `Pre-fix` framing and the byte-identity note for complex128

*Left in the source:* the f64-carrier-then-cast recipe and the asymmetry it removes (the R=0 early return kept complex64 while R != 0 did not)

```text
    # v5.17.x (audit P3-51): honour dtype-follows-input.  The carrier
    # argument ``k*r2/(2R)`` is accumulated at float64 (r2 is built from
    # f64 grids above) and only the FINISHED phase factor is cast to
    # E's complex dtype before the multiply -- the P2-29 f64-carrier-
    # then-cast recipe.  Pre-fix a complex64 E was silently promoted to
    # complex128 whenever R != 0 (while the R=0 early-return above kept
    # complex64), contradicting the docstring's "same shape and dtype".
    # complex128 inputs are byte-identical (astype(copy=False) no-op).
```

### L1129-1135 -- `angular_spectrum_propagate_batch` -- the H fetch -- what the pre-fix delegation cost (a wasted full-grid FFT+IFFT pair on garbage data on every batch call)

*Left in the source:* the live instruction and the reason it matters

```text
    # v5.17.x (P2-27): fetch H directly through the shared cache/build
    # helper.  Pre-fix this delegated to the FULL scalar propagator on
    # an uninitialised ``xp.empty`` proxy field with
    # ``return_transfer_function=True``, paying a wasted full-grid
    # FFT+IFFT pair (plus an fftshift+copy of H) on garbage data on
    # EVERY batch call -- even on H-cache hits -- which made the batch
    # entry point measurably SLOWER than two scalar calls.
```

### L1295-1304 -- `angular_spectrum_propagate_tilted` -- the carrier dtype -- the `Pre-fix` framing and the complex128 bit-identity note

*Left in the source:* what building the carrier at the wrong dtype would cost, and the mod-2*pi fold that keeps the float32 path accurate

```text
    # v5.17.1: build the carrier AT the target dtype.  Pre-fix it was
    # unconditionally complex128 (``np.exp`` of a float64 phase), which
    # silently upcast the ENTIRE tilted pipeline (demod field, FFTs,
    # remodulated output) for complex64 inputs -- doubling the working
    # memory AND returning complex128, violating the dtype-follows-input
    # contract every other propagator honours.  For complex64 the carrier
    # phase is folded mod 2*pi in float64 BEFORE the float32 cast (the same
    # accuracy mitigation as the main ASM kernel), so the large carrier
    # argument (~1e5 rad across a big tilted grid) doesn't hit the float32
    # precision floor.  The complex128 path is bit-identical to pre-fix.
```
