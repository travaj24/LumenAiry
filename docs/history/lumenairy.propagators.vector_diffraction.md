<!-- lumenairy-history-doc
module: lumenairy/propagators/vector_diffraction.py
ast_sha256: e6eaebabe033bc939822b64b2ddf535dbcd820eb2dc100e84ed919de6d945323
token_sha256: acb719cdf0f7b3874e78b0dfca42998fa327c7e3709e3a6805763e0ee0419c27
pre_relocation_lines: 518
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-14 -- Wave-5 item D (CI run 34914295323): DIGEST-SCHEME change, not a code change -- token_fingerprint now feeds an f-string to the digest as ONE STRING record holding its exact source text instead of the running tokenizer's FSTRING_START/FSTRING_MIDDLE/FSTRING_END run, so the recorded value is a property of the file rather than of the interpreter that read it; PEP 701 made CPython 3.12 tokenize f-strings differently from 3.11, these digests were recorded on 3.12+, and all five py3.11 CI shards read a different token_sha256 for byte-identical sources (110 of 123 documents, measured).  The module source is unchanged and ast_sha256 is unchanged.
-->

# Version history -- `lumenairy/propagators/vector_diffraction.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/vector_diffraction.py`.  Each block is reproduced **verbatim** under the source line it came
from in the pre-relocation file.

Every block here is the same shape, and it is the shape the WP-A17 SWEEP-1
follow-up pass was asked to close: a **live guard whose comment explained
itself by naming the release that added it** ("Pre-fix X happened", "Pre-4.12
the dispatcher only passed ...").  The hazard X is still reachable -- the guard
is the only thing preventing it -- so the source now states X in the present
tense, as what goes wrong WITHOUT the guard, together with every measurement
that sizes it.  What moved is the release attribution and the
"bit-identical to pre-fix" reassurance that travelled with it.

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
| L79-80 | `richards_wolf_focus` -- `dx_pupil` | the `pre-v5.30 this was silent` tail |
| L134-136 | `richards_wolf_focus` -- precision | the `Pre-4.11.2 the input was always promoted` framing |
| L159-160 | `richards_wolf_focus` -- the focal pitch | the `(was a silent mismatch pre-4.10)` parenthetical |
| L197-202 | `richards_wolf_focus` -- the rim mask | the `Pre-4.11.1 clipped to sin(theta_max) *before* the mask was built` framing |
| L286-288 | `richards_wolf_focus` -- the apodisation | the `Pre-4.10 used only the aplanatic factor` framing |
| L410-426 | `richards_wolf_focus` -- the pad/crop registration anchor | the `Pre-S9 this used ...` framing and the `bit-identical` reassurance |
| L453-457 | `richards_wolf_focus` -- the exp(+i k f) sign | the `The pre-4.11.2 code used exp(-i k f)` framing |
| L468-471 | `richards_wolf_focus` -- the prefactor f-dependence | the `Pre-4.11.2 the prefactor was` framing |

---

### L79-80 -- `richards_wolf_focus` -- `dx_pupil` -- the `pre-v5.30 this was silent` tail

*Left in the source:* what the warning reports

```text
           ``dx_pupil`` / ``Np`` needed (P4, v5.30); pre-v5.30 this was
           silent.
```

### L134-136 -- `richards_wolf_focus` -- precision -- the `Pre-4.11.2 the input was always promoted` framing

*Left in the source:* the rule and the knob it would otherwise negate

```text
    # 4.11.2: honour precision='single' via the global default complex
    # dtype.  Pre-4.11.2 the input was always promoted to complex128,
    # silently negating ``set_default_complex_dtype(np.complex64)``.
```

### L159-160 -- `richards_wolf_focus` -- the focal pitch -- the `(was a silent mismatch pre-4.10)` parenthetical

*Left in the source:* the warning and the alternative backend

```text
    # if the user passed a different value (was a silent mismatch
    # pre-4.10).  A true free-pitch focal plane requires a chirp-z
```

### L197-202 -- `richards_wolf_focus` -- the rim mask -- the `Pre-4.11.1 clipped to sin(theta_max) *before* the mask was built` framing

*Left in the source:* the whole failure: an identically-True mask silently extending the exit pupil to the array and leaving the rim mask unenforced

```text
    # 4.11.1: build the rim mask from the UNCLIPPED sin_theta_raw so
    # the geometric pupil is honoured.  Pre-4.11.1 clipped to
    # sin(theta_max) *before* the mask was built, making the mask
    # ``sin_theta <= sin(theta_max)`` identically True for every grid
    # pixel and silently extending the exit pupil to the whole array
    # (Richards-Wolf rim mask was effectively unenforced).
```

### L286-288 -- `richards_wolf_focus` -- the apodisation -- the `Pre-4.10 used only the aplanatic factor` framing

*Left in the source:* the derivation above it, and what the wrong apodisation costs at the rim

```text
    # Pre-4.10 used only the aplanatic factor, so the effective
    # apodisation was cos^(3/2) θ instead of cos^(-1/2) θ -- biased
    # toward the centre, missing energy at the high-NA rim.
```

### L410-426 -- `richards_wolf_focus` -- the pad/crop registration anchor -- the `Pre-S9 this used ...` framing and the `bit-identical` reassurance

*Left in the source:* the whole do-not-revert argument: which parity combinations the naive expression breaks, the spurious linear phase it injects, both measured rates against the prediction, and which consumer is blind to it

```text
        # Pre-S9 this used ``(N_focal - Np)//2`` / ``(Np - N_focal)//2``, which
        # is the SAME integer whenever ``Np`` and ``N_focal`` share parity but
        # is off by exactly one index when they do not -- specifically for
        # (Np odd, N_focal even) on the pad branch and (Np even, N_focal odd)
        # on the crop branch.  That one-index slip translates the whole masked
        # + apodised pupil by one pupil pixel, so the returned COMPLEX focal
        # field picks up a spurious linear phase of ``2*pi/N_focal`` rad per
        # focal pixel: measured 0.098175 rad/px at (Np, N_focal) = (33, 64)
        # and 0.369599 rad/px at (32, 17), both matching the prediction
        # ``2*pi*delta/N_focal`` to 6 digits, with the two fields differing by
        # 145% in L2.  ``debye_wolf_psf`` (intensity) is blind to it; coherent
        # superposition with a reference arm is not -- the same failure class
        # the 4.11.2 ``exp(+i k f)`` sign fix addressed.  Both new expressions
        # are IDENTICAL integers to the old ones for every same-parity
        # (Np, N_focal) pair, so every even/even case -- i.e. every realistic
        # call, including the ``N_focal is None -> Np`` default -- is
        # bit-identical.
```

### L453-457 -- `richards_wolf_focus` -- the exp(+i k f) sign -- the `The pre-4.11.2 code used exp(-i k f)` framing

*Left in the source:* the library-wide sign convention and the spurious phase the wrong sign injects

```text
    # The pre-4.11.2 code used `exp(-i k f)`, which is opposite-sign to
    # every other forward-prop in the library (angular_spectrum_propagate,
    # fresnel_propagate, ... all use exp(+i k z) under exp(-iωt)).
    # Coherent superposition with a reference arm therefore picked up a
    # spurious exp(-2 i k f) phase mismatch.
```

### L468-471 -- `richards_wolf_focus` -- the prefactor f-dependence -- the `Pre-4.11.2 the prefactor was` framing

*Left in the source:* the wrong scaling and its measured consequence across two focal lengths

```text
    # Pre-4.11.2 the prefactor was (-i k f / 2π) · dx² · exp(-i k f),
    # which scaled as f (amplitude) → f² (intensity) — the WRONG sign of
    # the f-dependence: a 1 m focal length and a 1 cm focal length gave
    # Airy peak intensities differing by 10⁴ in the wrong direction.
```
