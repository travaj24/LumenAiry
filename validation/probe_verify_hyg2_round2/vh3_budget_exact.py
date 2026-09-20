"""VERIFY-WAVE5-HYGIENE2 round 2 -- the phase budget against an EXACT phase.

Why this file exists beside ``vh3_budget.py``.  The shipped derivation of
``_PHASE_BUDGET_MAX`` measures both chirp-Z routes AND the dense route against
a ``math.fsum`` reference.  ``fsum`` makes the SUMMATION correctly rounded; it
says nothing about the PHASE.  The dense route forms

    t = alpha * (n - cI) * (k - cO)       # two float64 roundings
    t = t - rint(t)                       # exact, by Sterbenz

and the second line is exact -- but the first has already thrown away the low
bits of a product whose exact value needs ~63 bits.  A reference that forms
``t`` the same way therefore agrees with the dense route by CONSTRUCTION at
every budget, which is exactly the reading the report quotes ("2.8e-16 ..
4.5e-16 at EVERY budget tested").

Here the phase is reduced modulo one turn EXACTLY, with
``fractions.Fraction``: ``alpha`` is a float64 and therefore an exact rational,
``n - cI`` and ``k - cO`` are exact half-integers, so the product and its
fractional part are exact and only ONE rounding (to float64) happens, at the
very end, on a number in [-1/2, 1/2).  The kernels are separable, so only
``Nx*Mx + Ny*My`` exact phases are needed.  The double sum is then accumulated
with ``math.fsum``, so the reference is correctly rounded in BOTH senses.

    PYTHONPATH=<tree> python vh3_budget_exact.py OUT.json
"""
import cmath
import json
import math
import sys
import warnings
from fractions import Fraction

import numpy as np

TAU = 2.0 * math.pi


def exact_kernel(alpha, n_in, n_out, cI, cO, sign):
    """``W[k, n] = exp(sign*2*pi*i*frac(alpha*(n-cI)*(k-cO)))``, exact phase.

    Every factor is an exact rational, so the product and its reduction
    modulo one turn are exact; the single float64 rounding is of the reduced
    fraction, which lies in ``[-1/2, 1/2)`` and therefore keeps all 53 bits
    of the phase whatever the budget.
    """
    fa = Fraction(alpha)
    W = np.empty((n_out, n_in), dtype=np.complex128)
    fI = Fraction(cI).limit_denominator(2 ** 20)
    fO = Fraction(cO).limit_denominator(2 ** 20)
    for k in range(n_out):
        fk = Fraction(k) - fO
        for n in range(n_in):
            t = fa * (Fraction(n) - fI) * fk
            t -= math.floor(t)                 # exact, in [0, 1)
            if t >= Fraction(1, 2):
                t -= 1                         # exact, in [-1/2, 1/2)
            W[k, n] = cmath.exp(sign * TAU * 1j * float(t))
    return W


def exact_sum(E, Wy, Wx):
    """``F = Wy . E . Wx^T`` with every entry accumulated by ``math.fsum``."""
    My, Ny = Wy.shape
    Mx, Nx = Wx.shape
    F = np.empty((My, Mx), dtype=np.complex128)
    for ky in range(My):
        wy = Wy[ky]
        for kx in range(Mx):
            wx = Wx[kx]
            re, im = [], []
            for ny in range(Ny):
                a = wy[ny]
                row = E[ny]
                for nx in range(Nx):
                    v = row[nx] * a * wx[nx]
                    re.append(v.real)
                    im.append(v.imag)
            F[ky, kx] = complex(math.fsum(re), math.fsum(im))
    return F


def relL2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / np.linalg.norm(np.asarray(b)))


def fixture(n, seed=20260920):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n))
            + 1j * rng.standard_normal((n, n))).astype(np.complex128)


def main(out_path):
    import lumenairy
    from lumenairy.propagators import _bluestein as BL

    eps = float(np.finfo(np.float64).eps)
    PB = float(BL._PHASE_BUDGET_MAX)
    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'platform': sys.platform,
           'PHASE_BUDGET_MAX': PB, 'eps': eps}

    budgets = (1e5, 1e7, 1e9, PB, 1e10, 1e11, 1e12, 1e14, 1e15)
    out = {}
    # The SHIPPED geometry (N=24, M=12) and a second one, so the reading is
    # not a property of one shape.
    for (nin, nout) in ((24, 12), (16, 8)):
        E = fixture(nin)
        cI, cO = nin / 2.0, nout / 2.0
        Nmax = max(nin, nout)
        rows = []
        for budget in budgets:
            alpha = budget / Nmax ** 2
            Wx = exact_kernel(alpha, nin, nout, cI, cO, -1)
            ref = exact_sum(E, Wx, Wx)
            row = {'budget': budget, 'alpha': alpha,
                   'eps_times_budget': eps * budget}
            for meth in ('bluestein', 'separable', 'direct'):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    F = BL._bluestein_centred_2d(
                        E, alpha, alpha, nout, nout, sign=-1, xp=np,
                        fft2=np.fft.fft2, ifft2=np.fft.ifft2, method=meth)
                    row[f'warned_{meth}'] = any(
                        issubclass(x.category, RuntimeWarning) for x in w)
                row[meth] = relL2(F, ref)
            # ... and the same measurement against a reference that forms the
            # phase the way the DENSE route does, to show the instrument.
            t_naive = np.empty((nout, nin))
            kk = np.arange(nout) - cO
            nn = np.arange(nin) - cI
            t_naive = float(alpha) * kk[:, None] * nn[None, :]
            Wn = np.exp(-1j * TAU * (t_naive - np.rint(t_naive)))
            refn = exact_sum(E, Wn, Wn)
            row['direct_vs_naive_reference'] = relL2(
                BL._bluestein_centred_2d(
                    E, alpha, alpha, nout, nout, sign=-1, xp=np,
                    fft2=np.fft.fft2, ifft2=np.fft.ifft2, method='direct'),
                refn)
            row['naive_reference_vs_exact'] = relL2(refn, ref)
            rows.append(row)
        out[f'{nin}x{nout}'] = rows
    res['rows'] = out

    fits = {}
    for key, rows in out.items():
        for meth in ('bluestein', 'direct'):
            sel = [r for r in rows if 1e-14 < r[meth] < 1e-2]
            if len(sel) >= 3:
                x = np.log10([r['budget'] for r in sel])
                y = np.log10([r[meth] for r in sel])
                fits[f'{key}:{meth}'] = {
                    'slope': float(np.polyfit(x, y, 1)[0]),
                    'n': len(sel), 'decades': float(x.max() - x.min())}
    res['slopes'] = fits

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    for key, rows in out.items():
        print(f"--- {key} (against an EXACT-phase fsum reference) ---")
        print("  budget      bluestein   direct      eps*budget  "
              "direct-vs-naive  naive-vs-exact  warned")
        for r in rows:
            print(f"  {r['budget']:.2e}  {r['bluestein']:.3e}   "
                  f"{r['direct']:.3e}   {r['eps_times_budget']:.3e}   "
                  f"{r['direct_vs_naive_reference']:.3e}        "
                  f"{r['naive_reference_vs_exact']:.3e}       "
                  f"{r['warned_bluestein']}")
    print("slopes:", {k: round(v['slope'], 4) for k, v in fits.items()})
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1])
