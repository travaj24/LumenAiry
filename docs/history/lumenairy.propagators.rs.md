<!-- lumenairy-history-doc
module: lumenairy/propagators/rs.py
ast_sha256: 286d131880517ae59a7cbe28858ffe2565d6168f2b171497e65639d6633f40c0
token_sha256: f4f42229338bc79fe0aa579212dc7ff32d07f96c4ac3545809c5d06f7c3e026c
pre_relocation_lines: 773
recorded_by: WP-A17 SWEEP-1 (audit AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11, finding P2-4 / sec. 14 V6)
checker: tests/unit/test_audit2609_a17_history_relocation.py
re_recorded: 2026-09-13 -- WP-B3 (audit K9 second half): kernel='spatial-integrated' -- the RS-I Green's function integrated over each pixel (Shen & Wang 2006) via a folded 6-node tensor Gauss-Legendre rule, its own H-cache tag, and the alias guard widened to both spatial kernels; 'auto' and 'spatial' byte-identical
-->

# Version history -- `lumenairy/propagators/rs.py`

This file holds the version-history narrative that used to live in
`lumenairy/propagators/rs.py`.  Each block is reproduced **verbatim** under
the source line it came from in the pre-relocation file.

`rs.py` is short and its docstring is nearly all measurement.  Most of what
the loose classifier flags is the derivation of a live choice and stayed: the
`bandlimit` warning (4.4e-8 against 1.8e-2 relative L2 -- five decades worse
with the mask ON, which is why the default is `False` on this propagator),
the `'transfer'` / `'spatial'` failure-mode contrast, the continuity table
across the switch, the H-caching contract, and the P12 note that the cutoff is
the paper's `z -> infinity` asymptote rather than the exact limit.

Four things moved: the pre-4.10 kernel sign and the 4.11.1 docstring
correction that followed it; the measured pre-v5.46 failure of the
point-sampled kernel (`P_out/P_in` up to 25.70) together with the byte-identity
re-measurement against the pre-v5.46 module; a comment correcting an earlier
COMMENT (the retracted "agree to machine precision" claim); and the `dropped
the dead conjunct` framing on the bandlimit guard.

The last one is worth a note, because only its framing moved.  The comment
records that RS carries no `and z != 0` conjunct while ASM and ASM-MFT do,
and WHY -- RS hard-raises on `z <= 0`, ASM accepts `z == 0` as the exact
identity.  That asymmetry is live, and a future editor "tidying" the two into
agreement would break `ASM(z=0)`.  It stayed; only "dropped the dead
conjunct" became "there is no conjunct here".

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
| L269-272 | `rayleigh_sommerfeld_propagate` -- the impulse response | that pre-4.10 the kernel used the negated form, and that the docstring was corrected in 4.11.1 to match -- a note about the docstring's own history |
| L359-379 | `rayleigh_sommerfeld_propagate` -- why `kernel='auto'` | the measured pre-v5.46 failure (``P_out/P_in`` 21.44 / 5.31 / 25.70 at relative L2 4.50 / 2.08 / 4.95) and the five-point byte-identity re-measurement against the pre-v5.46 module |
| L418-419 | `rayleigh_sommerfeld_propagate` -- Notes | how wrong the pre-v5.46 spatial kernel was on the same grid |
| L431-434 | `rayleigh_sommerfeld_propagate` -- Agreement with ASM | a comment correcting an earlier COMMENT: that the historical 'agree to machine precision' claim was never true, and what the pre-v5.46 kernel actually plateaued at |
| L683-688 | `rayleigh_sommerfeld_propagate` -- the bandlimit guard | the `dropped the dead conjunct` framing |

---

### L269-272 -- `rayleigh_sommerfeld_propagate` -- the impulse response -- that pre-4.10 the kernel used the negated form, and that the docstring was corrected in 4.11.1 to match -- a note about the docstring's own history

*Left in the source:* the Goodman 3-43 formula itself, which is what the code builds

```text
    Pre-4.10 the kernel used the negated ``(ik - 1/r)`` form, so
    superposing RS with ASM / Fresnel results was 180-degrees out of
    phase.  The docstring formula was updated in 4.11.1 to match the
    corrected code.
```

### L359-379 -- `rayleigh_sommerfeld_propagate` -- why `kernel='auto'` -- the measured pre-v5.46 failure (``P_out/P_in`` 21.44 / 5.31 / 25.70 at relative L2 4.50 / 2.08 / 4.95) and the five-point byte-identity re-measurement against the pre-v5.46 module

*Left in the source:* the aliasing criterion, the measured accuracy of the LIVE default, and the back-compat statement a caller needs (byte-identical at and above the threshold, FFT-floor agreement below it)

```text
        **Why the default changed.**  The point-sampled kernel's phase
        gradient ``k*sin(theta)*dx`` exceeds the ``pi``/pixel Nyquist
        limit whenever ``z < 2*N*dx**2/wavelength``, and nothing checked
        it.  Measured with all-default arguments against an exact Hankel
        angular-spectrum oracle (Gaussian w0 = 6 um, lambda = 633 nm,
        z = 50 um): ``P_out/P_in`` = 21.44 (N = 64, dx = 2 um), 5.31
        (N = 128, dx = 1 um), 25.70 (N = 128, dx = 2 um) with relative L2
        of 4.50 / 2.08 / 4.95.  The same grids under ``'auto'``:
        relative L2 5.3e-8 / 6.1e-8 / 5.3e-8 with ``P_out/P_in``
        1.000000.  At and above the threshold nothing moves at all: on
        the N = 128 / dx = 1 um probe ``z_crit`` is 404.4 um, and at
        z = 405 um and z = 1 mm the default output is BYTE-IDENTICAL to
        the pre-v5.46 kernel (re-measured against the pre-v5.46 module
        itself, five (N, dx, z) points including odd N = 65 and N = 100).
        Below the threshold the default now takes the transfer branch, so
        it is NOT byte-identical there -- but where the spatial kernel was
        still adequately sampled the two agree to round-off: relative L2
        between the pre-v5.46 output and the current default is 1.7e-13 at
        z = 200 um and 2.1e-13 at z = 300 um on the same probe, both at
        the FFT's own floor (each is ~2e-13 from an 8x-zero-padded
        reference).
```

### L418-419 -- `rayleigh_sommerfeld_propagate` -- Notes -- how wrong the pre-v5.46 spatial kernel was on the same grid

*Left in the source:* the live statement -- ASM is already exact in that regime, so RS is not the remedy

```text
    exact Hankel oracle -- i.e. ASM is already exact there, and the
    pre-v5.46 RS spatial kernel was 2.08 (208 %) wrong on the same grid.
```

### L431-434 -- `rayleigh_sommerfeld_propagate` -- Agreement with ASM -- a comment correcting an earlier COMMENT: that the historical 'agree to machine precision' claim was never true, and what the pre-v5.46 kernel actually plateaued at

*Left in the source:* the live agreement statement and the conditions under which it holds

```text
    is a real, quantified modelling difference, not round-off.  (The
    pre-v5.46 spatial kernel plateaued at 2.8e-2 against ASM for exactly
    the point-sampling reason above -- the historical "agree to machine
    precision" claim was never true.)
```

### L683-688 -- `rayleigh_sommerfeld_propagate` -- the bandlimit guard -- the `dropped the dead conjunct` framing

*Left in the source:* why RS has no ``z != 0`` conjunct while ASM / ASM-MFT keep theirs -- the asymmetry is the load-bearing part

```text
        # v5.30 (audit P13): dropped the dead ``and z != 0`` conjunct.  RS
        # is forward-only and the guard above hard-raises for ``z <= 0``
        # (measured: ``z=0`` and ``z=-1e-3`` both ValueError), so ``z``
        # here is always > 0 and the test could never be False.  ASM /
        # ASM-MFT keep THEIR ``z != 0`` conjuncts -- those DO accept
        # ``z == 0`` (the exact identity, audit S2-11) and rely on it.
```
