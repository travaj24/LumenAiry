"""VERIFY-WP-C4 claim 6 -- the accuracy claim, re-measured against MY OWN
exactly-reduced reference, with the two-sided bar RE-DERIVED from the code.

THE REFERENCE.  ``alpha`` is a float64 and therefore an exact rational, and the
indices are integers, so ``t = alpha*(n - cI)*(k - cO)`` is EXACT in
:class:`fractions.Fraction`.  Its fractional turn is reduced there -- exactly --
and only then rounded once to float64, which lands inside ``[-1/2, 1/2)``.  The
products ``E[ny,nx] * W`` are formed in float64 (one rounding each) and the
double sum is accumulated with :func:`math.fsum`, which is exact.  So the
reference commits ONE rounding per term and none in the summation: growth
factor 1, against the routes' ``sqrt(n)`` / ``3 log2 L^2``.

ITS OWN EXACTNESS IS GATED TWO-SIDEDLY, because a reference that agrees by
construction measures the instrument and not the route:

* at a DYADIC ``alpha`` the exact reduction and the route's ``t - rint(t)``
  must agree to the BIT (both are exact), and
* at a generic ``alpha`` and a large budget they must PART.

THE BAR, RE-DERIVED FROM THE CODE (not copied from the report).  Each output
point is a sum of ``n = Ny*Nx`` unit-modulus terms, so the absolute error has
two sources: the summation (growth ``g``) and the phase the route's float64
product has already lost.  ``_bluestein_2d`` builds
``exp(i*sign*pi*alpha*m^2)`` -- read off the source, ``m`` runs over the PADDED
kernel index, ``|m| <= L - N_out``, and ``L = next_fast_len(N_in+N_out-1)``, so
the chirp route's phase in TURNS reaches ``alpha*(L - N_out)^2 / 2`` (the
``pi`` is half a turn, which the report's ``alpha*N_max^2`` drops).
``_direct_matrix_2d`` builds ``exp(i*sign*2*pi*alpha*n*k)``, so the dense
route's phase in turns reaches ``alpha*(N_in-1)*(N_out-1)`` exactly.

    bar_route = (g_route + 1 + 2*pi*max|t|_route) * eps * sum|E|

and the ratio the shipped test's bar ``R/4`` is asserted against is re-derived
here as ``R_turns = max|t|_chirp / max|t|_dense``, which is HALF the report's
``R = N_max^2/((N-1)(M-1))``.  Both readings are printed.

    PYTHONPATH=<tree> python v4_accuracy.py <tree> [--budgets 1,1000]
"""
from __future__ import annotations

import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v4lib  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

# (tag, N_in, N_out).  Square, non-centred.  Cost of the reference is
# N^2 * M^2 fsum steps, so the ladder is capped near 4e6.
DENSE_SIDE = [('d_64_2', 64, 2), ('d_96_3', 96, 3), ('d_128_4', 128, 4),
              ('d_160_5', 160, 5), ('d_192_6', 192, 6), ('d_224_7', 224, 7),
              ('d_256_8', 256, 8), ('d_128_2', 128, 2), ('d_320_10', 320, 10)]
CHIRP_SIDE = [('c_32_2', 32, 2), ('c_64_4', 64, 4), ('c_96_6', 96, 6),
              ('c_128_8', 128, 8), ('c_64_8', 64, 8), ('c_24_12', 24, 12),
              ('c_32_16', 32, 16), ('c_48_24', 48, 24)]

#: A GENERIC multiplier on alpha.  With ``alpha = budget/N^2`` at a power-of-two
#: N the phase is dyadic, ``t - rint(t)`` is exact, the exact reference agrees
#: BY CONSTRUCTION and every route reads ~1e-16 -- the degeneracy the branch's
#: probe caught at ``N = 64 -> M = 2``.  Multiplying by an odd non-dyadic
#: rational kills it at every shape rather than at the ones that happen to have
#: a factor of 3 in ``N^2``.
GENERIC = 1.0 / 3.0


def exact_reference(np, E, alpha_x, alpha_y, M_y, M_x, sign):
    """The same sum, phase reduced EXACTLY and accumulated with ``fsum``."""
    Ny, Nx = E.shape
    ax, ay = Fraction(float(alpha_x)), Fraction(float(alpha_y))

    def kernel(a, n_in, n_out):
        W = np.empty((n_out, n_in), dtype=np.complex128)
        for k in range(n_out):
            for n in range(n_in):
                t = a * n * k
                f = t - (t.numerator // t.denominator)      # in [0, 1)
                if f >= Fraction(1, 2):
                    f -= 1                                   # into [-1/2, 1/2)
                ph = sign * 2.0 * math.pi * float(f)
                W[k, n] = complex(math.cos(ph), math.sin(ph))
        return W

    Wx = kernel(ax, Nx, M_x)
    Wy = kernel(ay, Ny, M_y)
    F = np.empty((M_y, M_x), dtype=np.complex128)
    Er, Ei = E.real, E.imag
    for ky in range(M_y):
        wy = Wy[ky]
        for kx in range(M_x):
            wx = Wx[kx]
            # outer product of the two kernels, formed once in float64
            Wr = wy.real[:, None] * wx.real[None, :] \
                - wy.imag[:, None] * wx.imag[None, :]
            Wi = wy.real[:, None] * wx.imag[None, :] \
                + wy.imag[:, None] * wx.real[None, :]
            pr = (Er * Wr - Ei * Wi).ravel()
            pi_ = (Er * Wi + Ei * Wr).ravel()
            F[ky, kx] = complex(math.fsum(pr.tolist()),
                                math.fsum(pi_.tolist()))
    return F


def derived_bars(np, N, M, alpha, sumabsE):
    """The two routes' bars, from the CODE's own phase construction."""
    from scipy.fft import next_fast_len
    eps = float(np.finfo(np.float64).eps)
    L = int(next_fast_len(int(N + M - 1)))
    m_max = max(L - M, M - 1, N - 1)
    t_chirp = abs(alpha) * float(m_max) ** 2 / 2.0        # turns
    t_dense = abs(alpha) * float(N - 1) * float(M - 1)    # turns
    g_chirp = 3.0 * math.log2(float(L) ** 2)
    g_dense = math.sqrt(float(N)) + math.sqrt(float(M))
    g_ref = 1.0
    return {
        'L': L, 'm_max': m_max,
        'max_t_turns_chirp': t_chirp, 'max_t_turns_dense': t_dense,
        'R_turns': (t_chirp / t_dense) if t_dense > 0 else float('inf'),
        'R_report': (float(max(N, M)) ** 2
                     / (float(N - 1) * float(M - 1))) if M > 1 else float('inf'),
        'bar_chirp': (g_chirp + g_ref + 2 * math.pi * t_chirp) * eps * sumabsE,
        'bar_dense': (g_dense + g_ref + 2 * math.pi * t_dense) * eps * sumabsE,
    }


def main(tree, budgets):
    import numpy as np
    from lumenairy.propagators._bluestein import (_auto_selects_direct,
                                                  _bluestein_2d)
    from lumenairy.propagators.fft_infra import _fft2, _ifft2
    v4lib.anchor(tree)
    out = {'build': v4lib.build_tag(), 'python': sys.version.split()[0],
           'numpy': np.__version__, 'budgets': budgets,
           'generic_multiplier': GENERIC, 'rows': [], 'controls': []}
    rng = np.random.default_rng(777)

    for budget in budgets:
        for tag, N, M in DENSE_SIDE + CHIRP_SIDE:
            E = (rng.standard_normal((N, N))
                 + 1j * rng.standard_normal((N, N))).astype(np.complex128)
            alpha = GENERIC * float(budget) / float(max(N, M)) ** 2
            ref = exact_reference(np, E, alpha, alpha, M, M, -1)
            kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
            Fc = _bluestein_2d(E, alpha, alpha, M, M, method='bluestein', **kw)
            Fs = _bluestein_2d(E, alpha, alpha, M, M, separable=True,
                               method='separable', **kw)
            Fd = _bluestein_2d(E, alpha, alpha, M, M, method='direct', **kw)
            Fa = _bluestein_2d(E, alpha, alpha, M, M, **kw)
            sumabs = float(np.abs(E).sum())
            bars = derived_bars(np, N, M, alpha, sumabs)
            nrm = float(np.linalg.norm(ref))

            def rd(F):
                return (float(np.max(np.abs(F - ref))),
                        float(np.linalg.norm(F - ref) / nrm))

            (mc, rc), (ms, rs), (md, rrd) = rd(Fc), rd(Fs), rd(Fd)
            ma, ra = rd(Fa)
            says = bool(_auto_selects_direct(N, N, M, M))
            row = {'budget': budget, 'tag': tag, 'N': N, 'M': M,
                   'ratio': M / N, 'alpha': alpha,
                   'actual_budget': alpha * float(max(N, M)) ** 2,
                   'rule_says_direct': says, 'sum_abs_E': sumabs, **bars,
                   'maxabs_chirp': mc, 'maxabs_sep': ms, 'maxabs_dense': md,
                   'maxabs_auto': ma,
                   'rel_chirp': rc, 'rel_sep': rs, 'rel_dense': rrd,
                   'rel_auto': ra,
                   'auto_bits_match_dense': bool(
                       Fa.tobytes() == Fd.tobytes()),
                   'auto_bits_match_chirp': bool(
                       Fa.tobytes() == Fc.tobytes()),
                   'chirp_inside_bar': bool(mc <= bars['bar_chirp']),
                   'sep_inside_bar': bool(ms <= bars['bar_chirp']),
                   'dense_inside_bar': bool(md <= bars['bar_dense']),
                   'room_chirp_decades': (math.log10(bars['bar_chirp'] / mc)
                                          if mc > 0 else None),
                   'room_dense_decades': (math.log10(bars['bar_dense'] / md)
                                          if md > 0 else None),
                   'gap_chirp_over_dense': (rc / rrd) if rrd > 0 else None,
                   'bar_R_over_4': bars['R_report'] / 4.0,
                   'bar_Rturns_over_4': bars['R_turns'] / 4.0}
            out['rows'].append(row)
            print(f"b={budget:<8g} {tag:9s} N={N:4d} M={M:3d} "
                  f"rule={'D' if says else 'c'} "
                  f"chirp={rc:.3e} dense={rrd:.3e} gap="
                  f"{row['gap_chirp_over_dense']:9.2f} "
                  f"R/4={row['bar_R_over_4']:7.2f} "
                  f"Rt/4={row['bar_Rturns_over_4']:7.2f} "
                  f"in_bars={row['chirp_inside_bar']}/{row['dense_inside_bar']}"
                  f" auto={'dense' if row['auto_bits_match_dense'] else ('chirp' if row['auto_bits_match_chirp'] else 'NEITHER')}",
                  flush=True)
            del E

    # ---- CONTROL 1: the reference's own exactness, two-sided -------------
    # dyadic alpha at a power-of-two N -> the exact reduction and the route's
    # t - rint(t) are BOTH exact and must agree to the bit; a generic alpha at
    # a high budget must make them PART.
    for label, N, M, alpha in (('dyadic_exact', 64, 2, 1.0e3 / 4096.0),
                               ('dyadic_exact_int', 64, 2, 1e15 / 4096.0),
                               ('generic_parts', 96, 3,
                                GENERIC * 1e12 / 96.0 ** 2)):
        fr = Fraction(float(alpha))
        diffs = []
        for k in range(M):
            for n in range(N):
                t = fr * n * k
                f = t - (t.numerator // t.denominator)
                if f >= Fraction(1, 2):
                    f -= 1
                route = float(alpha) * float(n) * float(k)
                route = route - float(np.rint(route))
                diffs.append(abs(float(f) - route))
        rec = {'control': label, 'N': N, 'M': M, 'alpha': alpha,
               'alpha_is_integer': bool(float(alpha).is_integer()),
               'max_frac_difference': max(diffs)}
        out['controls'].append(rec)
        print(f"CONTROL {label}: alpha={alpha!r} integer="
              f"{rec['alpha_is_integer']} max|frac diff|="
              f"{rec['max_frac_difference']:.3e}", flush=True)

    # ---- CONTROL 2: is the R/4 bar two-sided?  An IMPOSTOR dense arm ------
    # (the separable chirp-Z answer handed back as if it were the dense one)
    # must read a gap of ~1, i.e. FAIL the R/4 bar at every dense-side shape.
    for tag, N, M in DENSE_SIDE[:4]:
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        alpha = GENERIC * 1.0e3 / float(max(N, M)) ** 2
        ref = exact_reference(np, E, alpha, alpha, M, M, -1)
        kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
        Fc = _bluestein_2d(E, alpha, alpha, M, M, method='bluestein', **kw)
        Fs = _bluestein_2d(E, alpha, alpha, M, M, separable=True,
                           method='separable', **kw)
        nrm = float(np.linalg.norm(ref))
        rc = float(np.linalg.norm(Fc - ref) / nrm)
        rimp = float(np.linalg.norm(Fs - ref) / nrm)
        bars = derived_bars(np, N, M, alpha, float(np.abs(E).sum()))
        rec = {'control': 'impostor_dense_arm', 'tag': tag, 'N': N, 'M': M,
               'gap_with_impostor': rc / rimp,
               'bar_R_over_4': bars['R_report'] / 4.0,
               'impostor_refused': bool(rc / rimp < bars['R_report'] / 4.0)}
        out['controls'].append(rec)
        print(f"CONTROL impostor {tag}: gap={rec['gap_with_impostor']:.3f} "
              f"bar={rec['bar_R_over_4']:.2f} refused="
              f"{rec['impostor_refused']}", flush=True)
        del E

    ins = [r for r in out['rows']]
    out['summary'] = {
        'rows': len(ins),
        'all_inside_own_bar': all(
            (r['dense_inside_bar'] if r['rule_says_direct']
             else r['chirp_inside_bar']) for r in ins),
        'chirp_inside_bar': sum(1 for r in ins if r['chirp_inside_bar']),
        'sep_inside_bar': sum(1 for r in ins if r['sep_inside_bar']),
        'dense_inside_bar': sum(1 for r in ins if r['dense_inside_bar']),
        'auto_matched_neither': sum(
            1 for r in ins if not r['auto_bits_match_dense']
            and not r['auto_bits_match_chirp']),
        'auto_route_agrees_with_rule': sum(
            1 for r in ins
            if r['auto_bits_match_dense'] == r['rule_says_direct']),
        'min_room_decades': min(
            [r['room_dense_decades'] if r['rule_says_direct']
             else r['room_chirp_decades'] for r in ins
             if r['room_dense_decades'] is not None
             and r['room_chirp_decades'] is not None]),
    }
    print('SUMMARY', out['summary'], flush=True)
    v4lib.write_json(out, os.path.join(
        HERE, f"v4_accuracy_{v4lib.short_tag()}.json"))


def _arg(flag, default=None):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


if __name__ == '__main__':
    main(sys.argv[1],
         [float(x) for x in _arg('--budgets', '1,1000').split(',')])
