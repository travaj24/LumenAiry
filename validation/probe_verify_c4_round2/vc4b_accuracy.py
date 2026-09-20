"""VERIFY-WP-C4 ROUND 2, items 7 and 8 -- D6's phase-ratio derivation and N5's
two accuracy populations, against an oracle written from scratch.

THE ORACLE.  The reference here is computed ENTIRELY in ``mpmath`` at 40
decimal digits: the phase ``t = alpha*n*k`` is formed as an exact
:class:`fractions.Fraction` (``alpha`` is a float64 and therefore an exact
rational), reduced into ``[-1/2, 1/2)`` exactly, and only then handed to
``mp.exp``; both matrix products are accumulated in ``mpmath`` too.  Its own
error floor is ~1e-40, twenty-seven decades under anything measured, so the
routes' errors are resolved rather than compared against another float64 sum.
The shipped test's reference reduces the phase the same way -- there is no
other way -- but evaluates ``exp`` and sums in float64 with ``math.fsum``;
this one shares no line with it and its floor is derived rather than argued.

WHAT IS MEASURED

* **D6 (item 7).**  ``R_turns = (L - N_out)^2 / (2*(N-1)*(M-1))`` recomputed
  from the chirp's own padded index, against the shipped
  ``R = N_max^2/((N-1)(M-1))``; the measured gap ``err_chirp/err_dense``; the
  bar ``R/4``; and ``C_chirp/C_dense`` measured over ten decades of phase
  budget, which is the factor the "3.0x to 8.1x below the implied gap"
  statement rests on.  The bar's lower side is exercised by an IMPOSTOR: the
  separable chirp-Z arm answering in the dense arm's place, which must read a
  gap near 1.
* **N5 (item 8).**  The eight-plus shapes of section 4.3's ladder split by
  whether ``alpha*n*k`` is exactly representable in float64, measured here
  rather than argued from the odd part of ``N_max^2``, with six rows on each
  side instead of five and three.

    PYTHONPATH=<tree> python vc4b_accuracy.py <tree>

Author:  Andrew Traverso
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import vc4blib as L                                              # noqa: E402

DPS = 40

#: ``(N, M)`` -- section 4.3's dense-side ladder, extended by three shapes on
#: each side so both N5 populations carry six rows.
LADDER = [(64, 2), (96, 3), (128, 4), (160, 5), (192, 6), (224, 7),
          (256, 8), (128, 2), (288, 9), (320, 10), (384, 12), (448, 14)]

#: The geometry the ``C_chirp / C_dense`` envelope is measured at -- the same
#: small square fixture the hygiene-2 round the claim cites used, so the
#: envelope is comparable; the budgets are this probe's own.
C_GEOM = (24, 12)
C_BUDGETS = [1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15]


def _exact_frac_phase(alpha, n_in, n_out):
    """``frac(alpha*n*k)`` in ``[-1/2, 1/2)`` as exact Fractions."""
    fa = Fraction(alpha)
    rows = []
    for k in range(n_out):
        fk = Fraction(k)
        row = []
        for n in range(n_in):
            t = fa * fk * n
            t -= math.floor(t)
            if t >= Fraction(1, 2):
                t -= 1
            row.append(t)
        rows.append(row)
    return rows


def mp_reference(E, alpha, M, sign=-1):
    """The same sum, in mpmath at ``DPS`` digits, phase reduced exactly."""
    import mpmath as mp
    import numpy as np
    mp.mp.dps = DPS
    ny, nx = E.shape
    tau_i = mp.mpc(0, sign) * 2 * mp.pi
    Ty = _exact_frac_phase(alpha, ny, M)
    Tx = _exact_frac_phase(alpha, nx, M)
    Wy = [[mp.exp(tau_i * mp.mpf(t.numerator) / mp.mpf(t.denominator))
           for t in row] for row in Ty]
    Wx = [[mp.exp(tau_i * mp.mpf(t.numerator) / mp.mpf(t.denominator))
           for t in row] for row in Tx]
    Erows = [[mp.mpc(complex(v)) for v in row] for row in np.asarray(E)]
    # G[i][kx] = sum_n E[i][n] * Wx[kx][n]
    G = [[None] * M for _ in range(ny)]
    for i in range(ny):
        ei = Erows[i]
        for kx in range(M):
            wx = Wx[kx]
            s = mp.mpc(0)
            for n in range(nx):
                s += ei[n] * wx[n]
            G[i][kx] = s
    out = np.empty((M, M), dtype=np.complex128)
    for ky in range(M):
        wy = Wy[ky]
        for kx in range(M):
            s = mp.mpc(0)
            for i in range(ny):
                s += wy[i] * G[i][kx]
            out[ky, kx] = complex(s)
    return out


def alpha_is_exact(alpha, N, M):
    """Is ``alpha*k*n`` exact in float64 at every index pair?  Measured."""
    fa = Fraction(alpha)
    bad = 0
    for k in range(M):
        for n in range(N):
            if Fraction(alpha * k * n) != fa * k * n:
                bad += 1
    return bad == 0, bad


def main(tree):
    import numpy as np
    from scipy.fft import next_fast_len
    la = L.anchor(tree)
    L.single_thread_ffts()
    from lumenairy.propagators._bluestein import _bluestein_2d
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    eps = float(np.finfo(np.float64).eps)
    out = {'build': L.build(), 'python': sys.version.split()[0],
           'numpy': np.__version__, 'lumenairy_version': la.__version__,
           'oracle': 'mpmath dps=%d, exact Fraction phase' % DPS,
           'rows': [], 'c_ratio_rows': []}

    def run(E, a, M, **kw):
        return _bluestein_2d(E, a, a, M, M, sign=-1, xp=np, fft2=_fft2,
                             ifft2=_ifft2, **kw)

    # ---- items 7 + 8: the ladder at budget 1e3 --------------------------
    for (N, M) in LADDER:
        rng = np.random.default_rng(496)
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        a = 1.0e3 / float(max(N, M)) ** 2
        ref = mp_reference(E, a, M)
        dense = run(E, a, M, method='direct')
        chirp = run(E, a, M, method='bluestein')
        sep = run(E, a, M, method='separable')
        e_d = float(np.max(np.abs(dense - ref)))
        e_c = float(np.max(np.abs(chirp - ref)))
        e_s = float(np.max(np.abs(sep - ref)))
        Lpad = int(next_fast_len(N + M - 1))
        R = float(max(N, M)) ** 2 / (float(N - 1) * float(max(M - 1, 1)))
        R_turns = (float(Lpad - M) ** 2
                   / (2.0 * float(N - 1) * float(max(M - 1, 1))))
        exact, n_bad = alpha_is_exact(a, N, M)
        gap = e_c / e_d if e_d > 0 else float('inf')
        imp = e_c / e_s if e_s > 0 else float('inf')
        row = {'N': N, 'M': M, 'alpha': a, 'L_pad': Lpad,
               'alpha_n_k_exact_in_float64': exact,
               'inexact_index_pairs': n_bad,
               'err_dense': e_d, 'err_chirp': e_c, 'err_separable': e_s,
               'gap_chirp_over_dense': gap,
               'impostor_gap_chirp_over_separable': imp,
               'R_shipped': R, 'R_turns': R_turns,
               'R_over_R_turns': R / R_turns,
               'bar_R_over_4': R / 4.0,
               'R_turns_over_2': R_turns / 2.0,
               'bar_equals_R_turns_over_2': abs(R / 4.0 - R_turns / 2.0)
                                            / (R_turns / 2.0),
               'gap_over_bar': gap / (R / 4.0)}
        out['rows'].append(row)
        print("%4d->%-3d alpha_exact=%-5s err d=%.3e c=%.3e  gap=%9.2f  "
              "R=%8.2f R_turns=%8.2f  bar=R/4=%7.2f gap/bar=%6.2f  "
              "impostor=%.3f"
              % (N, M, exact, e_d, e_c, gap, R, R_turns, R / 4.0,
                 row['gap_over_bar'], imp), flush=True)

    # ---- item 7: C_chirp / C_dense over ten decades ---------------------
    N, M = C_GEOM
    rng = np.random.default_rng(496)
    E = (rng.standard_normal((N, N))
         + 1j * rng.standard_normal((N, N))).astype(np.complex128)
    S = float(np.sum(np.abs(E)))
    for budget in C_BUDGETS:
        a = budget / float(max(N, M)) ** 2
        ref = mp_reference(E, a, M)
        e_d = float(np.max(np.abs(run(E, a, M, method='direct') - ref)))
        e_c = float(np.max(np.abs(run(E, a, M, method='bluestein') - ref)))
        t_d = abs(a) * float(N - 1) * float(M - 1)
        t_c = abs(a) * float(max(N, M)) ** 2
        C_d = e_d / (eps * t_d * S) if t_d else float('nan')
        C_c = e_c / (eps * t_c * S) if t_c else float('nan')
        rec = {'budget': budget, 'err_dense': e_d, 'err_chirp': e_c,
               'max_t_dense': t_d, 'max_t_chirp': t_c,
               'C_dense': C_d, 'C_chirp': C_c,
               'C_chirp_over_C_dense': C_c / C_d if C_d else float('nan')}
        out['c_ratio_rows'].append(rec)
        print("budget %8.1e  err d=%.3e c=%.3e  C_d=%.4f C_c=%.4f  "
              "C_c/C_d=%.4f" % (budget, e_d, e_c, C_d, C_c,
                                rec['C_chirp_over_C_dense']), flush=True)

    ratios = [r['C_chirp_over_C_dense'] for r in out['c_ratio_rows']
              if math.isfinite(r['C_chirp_over_C_dense'])]
    out['C_ratio_envelope'] = [min(ratios), max(ratios)]
    ex = [r for r in out['rows'] if r['alpha_n_k_exact_in_float64']]
    nx_ = [r for r in out['rows'] if not r['alpha_n_k_exact_in_float64']]
    out['population_exact'] = {
        'n': len(ex), 'shapes': ["%d->%d" % (r['N'], r['M']) for r in ex],
        'gap_range': [min(r['gap_chirp_over_dense'] for r in ex),
                      max(r['gap_chirp_over_dense'] for r in ex)]}
    out['population_rounded'] = {
        'n': len(nx_), 'shapes': ["%d->%d" % (r['N'], r['M']) for r in nx_],
        'gap_range': [min(r['gap_chirp_over_dense'] for r in nx_),
                      max(r['gap_chirp_over_dense'] for r in nx_)]}
    out['bar_violations'] = [
        "%d->%d" % (r['N'], r['M']) for r in out['rows']
        if not r['gap_chirp_over_dense'] > r['bar_R_over_4']]
    out['impostor_gaps'] = [r['impostor_gap_chirp_over_separable']
                            for r in out['rows']]
    out['max_relative_difference_bar_vs_R_turns_over_2'] = max(
        r['bar_equals_R_turns_over_2'] for r in out['rows'])
    out['gap_over_bar_range'] = [min(r['gap_over_bar'] for r in out['rows']),
                                 max(r['gap_over_bar'] for r in out['rows'])]
    print()
    print("C_chirp/C_dense envelope        :", out['C_ratio_envelope'])
    print("implied gap / bar (2*C_c/C_d)   :",
          [2 * out['C_ratio_envelope'][0], 2 * out['C_ratio_envelope'][1]])
    print("population EXACT   (n=%d)       :" % len(ex),
          out['population_exact']['gap_range'])
    print("population ROUNDED (n=%d)       :" % len(nx_),
          out['population_rounded']['gap_range'])
    print("bar violations                  :", out['bar_violations'])
    print("impostor gaps                   :",
          [round(x, 3) for x in out['impostor_gaps']])
    print("max |R/4 - R_turns/2| / (R_turns/2):",
          out['max_relative_difference_bar_vs_R_turns_over_2'])
    L.write(out, os.path.join(HERE, "vc4b_accuracy_%s.json" % L.tag()))


if __name__ == '__main__':
    main(sys.argv[1])
