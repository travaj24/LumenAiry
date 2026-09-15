"""Final report additions: the fft_infra finding and the extra w3 red."""
import io
import sys

P = ('docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/'
     'WP-B14_KNOWN_REDS_REPORT.md')

ANCHOR = "## 7. Runs\n"

NEW = '''### 6.8 A LIBRARY DEFECT found while root-causing the a6 bit pin -- named, not fixed

Chasing the "bit-identical" transfer-function pin (6.7, third bullet) produced a
finding that is bigger than the test it came from, and it is recorded here
because it is the next wave's, not this one's.

**With the shipped pyFFTW two-buffer ping-pong ON, `_ifft2(_fft2(E) * H)` in
`lumenairy/propagators/fft_infra.py` is not a function of its input values
alone.**  Two call sites handed bit-identical `E` and bit-identical `H` return
results that differ.  Measured on WSL (Linux, numpy 2.4.6, pyFFTW 0.15.1), and
only at `n >= FFTW_MIN_SIZE` (256):

| mode | n = 128 | n = 256 | n = 512 |
|---|---|---|---|
| shipped (ping-pong on) | 0 | **1.448e-15 / 1.798e-15** (79 % of doubles differ) | **1.790e-15 / 1.897e-15** |
| `set_fft_double_buffer(False)` | 0 | **0** | **0** |
| `USE_PYFFTW=False` | 0 | 0 | 0 |

Identical at 1, 2 and 8 FFTW threads, under both ESTIMATE and MEASURE planning,
and across repeats; Windows reads exactly 0 in all three modes on all eight
ladder rungs.  Neither route is the "right" one -- both sit 2.5e-15 to 3.1e-15
from pocketfft -- so this is a reproducibility defect, not an accuracy one, and
the physics is unaffected at ~1 ULP.

What makes it a defect rather than a curiosity: **it contradicts
`set_fft_double_buffer`'s own docstring**, which states that values are
byte-identical either way.  A byte-identity contract that the shipped default
cannot honour is exactly the kind of claim the audit exists to find.

Not fixed here, and deliberately: the ping-pong is a documented 256 MB - 1 GB
per call saving with its own tests, there is no minimal correct edit visible
from outside the module, and the mechanism below the A/B was NOT established --
slot-plan parity, buffer stability across an intervening allocation, operand and
output alignment (swept 0 / 16 / 32 / 48 mod 64 on the real values), thread
count and planner effort each test clean in isolation.  Reproducer:
`validation/probe_known_reds/probe_c4_double_buffer.py`.  **Handed to the owner
of `fft_infra.py`.**

The a6 test itself was fixed without waiting for that: the byte claim now runs
where the change under test lives -- the whole shipped function against the
oracle with the ping-pong off, restored in a `finally` -- unconditionally on
every arm and every shape, measured 0.000e+00, with a second unconditional
claim on the shipped dispatch at a derived 1e-13 relative bar (250x over the
worst arm's 3.9e-16 and nine decades under the 1e-4 exact-versus-Fresnel kernel
gap).  No skip was needed.

### 6.9 One more BLAS-build classification, found by the wave's own ladder

`test_niche_audit_w3_oracles.py::test_w4_t1_pure_lg00_has_no_sigma_grid_and_is_unchanged`
was not in the CI failure set -- CI never reached it, because `--maxfail`
aborted first.  The kernel ladder run for the two briefed w3 pins found it: six
arms `181 passed`, **SANDYBRIDGE at one and at four threads `1 failed,
180 passed`**, on an exact-equality pin against the SAME frozen `L(0,0)`
literal as the briefed sibling, reading `rel 1.223e-08` against its own 1e-8.

The cause is the same and is worth stating plainly, because it is why an exact
pin was never going to hold: `L(0,0)` is a saddle amplitude with a stationary
phase of `|Phi| = 9.8883e+05` rad, so ONE ULP on a fit coefficient reaches the
answer as `eps |Phi| = 2.196e-10` -- and `md5(coef_phi)` differs on every
`OPENBLAS_CORETYPE` rung.  Measured `rel` against the frozen literal: HASWELL
1.838e-09, NEHALEM 6.653e-09, KATMAI 7.518e-09, **SANDYBRIDGE 1.223e-08**, WSL
4.398e-09, CI 8.190e-09 (the same digits on py3.14 and on the py3.12 JAX job,
so deterministic per build, not noise).  The constants were baked on one arm,
and the test was red on Windows at the 5.47.0 commit AND at the commit that
introduced the pin.

Both tests now read ONE shared bar derived once beside the constant
(`_Y2_L00_REL_BAR = 1e-6`, 82x over the worst arm), with the DERIVATION --
`want == old * (-1j) / (lambda sqrt|det J|)` -- kept as a separate
frozen-to-frozen claim at 1e-11 where the worst arm reads 9.86e-13.  Post-fix:
SANDYBRIDGE t1 and t4 and KATMAI t1 and t4 all `181 passed`.

That this was found at all is the argument for the ladder being mandatory: it
was invisible to CI (hidden behind `--maxfail`) and invisible on the default
local arm.

'''


def main():
    with io.open(P, encoding='utf-8', newline='') as fh:
        raw = fh.read()
    nl = '\r\n' if '\r\n' in raw else '\n'
    src = raw.replace('\r\n', '\n')
    if src.count(ANCHOR) != 1:
        print('anchor not unique:', src.count(ANCHOR))
        return 1
    src = src.replace(ANCHOR, NEW + ANCHOR)
    with io.open(P, 'w', encoding='utf-8', newline='') as fh:
        fh.write(src.replace('\n', nl))
    print('sections 6.8 and 6.9 written')
    return 0


if __name__ == '__main__':
    sys.exit(main())
