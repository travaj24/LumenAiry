"""Round 3 (VERIFY-WAVE5-HYGIENE2 round 2, D-1) -- the phase budget on the
SHIPPED id's own geometry, against a reference whose phase is EXACT.

Why this exists beside ``validation/probe_verify_hyg2_round2/
vh3_budget_exact.py``.  That probe drives ``_bluestein_centred_2d`` with
``cI = N/2``, ``cO = M/2``; the shipped id
``test_wave5_h2_mft_direct.py::_chirp_and_dense_at`` drives ``_bluestein_2d``
with ``cI = cO = 0`` and the library's own ``_fft2`` / ``_ifft2``.  The two
geometries put a different ``max |alpha n k|`` on the same budget, so the
constant in ``rel ~ C eps budget`` is not the same number and the id's bar has
to be derived on the id's own geometry.  Both are measured here.

The reference's phase is reduced modulo one turn with ``fractions.Fraction``:
``alpha`` is a float64 and therefore an exact rational and ``n``, ``k`` are
integers, so the product and its fractional part are exact and the single
float64 rounding lands on a number already inside ``[-1/2, 1/2)``.  The double
sum is accumulated with ``math.fsum``, so the reference is correctly rounded
in BOTH senses.

    PYTHONPATH=<tree> python r3_budget_exact.py <tree> OUT.json

The tree is named twice on purpose: ``sys.path[0]`` for a script is the
SCRIPT'S directory and not the working directory, so without the explicit
anchor below a probe run from the worktree silently binds whatever
``lumenairy`` is installed in site-packages.  MEASURED here 2026-09-20: the
box carries an installed 5.47.0 that predates the ``method=`` keyword, and
the first run of this file bound it.
"""

import json
import math
import os
import sys
import warnings
from fractions import Fraction

import numpy as np

TAU = 2.0 * math.pi


def anchor(tree):
    """Import lumenairy and REFUSE anything outside ``tree``."""
    import lumenairy
    got = os.path.realpath(lumenairy.__file__)
    want = os.path.realpath(tree)
    try:
        same = os.path.commonpath([got, want]) == want
    except ValueError:                      # different drives on Windows
        same = False
    if not same:
        raise SystemExit(f"WRONG TREE: {got!r} is not under {want!r}")
    return lumenairy


def exact_phase(alpha, n_in, n_out, c_in=0.0, c_out=0.0):
    """``frac(alpha (n-cI) (k-cO))`` in ``[-1/2, 1/2)``, reduced EXACTLY."""
    fa = Fraction(alpha)
    fI = Fraction(c_in).limit_denominator(1 << 20)
    fO = Fraction(c_out).limit_denominator(1 << 20)
    T = np.empty((n_out, n_in), dtype=np.float64)
    for k in range(n_out):
        fk = Fraction(k) - fO
        for n in range(n_in):
            t = fa * (Fraction(n) - fI) * fk
            t -= math.floor(t)
            if t >= Fraction(1, 2):
                t -= 1
            T[k, n] = float(t)
    return T


def naive_phase(alpha, n_in, n_out, c_in=0.0, c_out=0.0):
    """The same phase formed the way the ROUTE forms it: one float64 product,
    then ``t - rint(t)``."""
    n = np.arange(n_in, dtype=np.float64) - float(c_in)
    k = np.arange(n_out, dtype=np.float64) - float(c_out)
    t = float(alpha) * k[:, None] * n[None, :]
    return t - np.rint(t)


def exact_rows(alpha, n_in, n_out, c_in=0.0, c_out=0.0, sign=-1):
    """``W[k, n] = exp(sign 2 pi i frac(alpha (n-cI) (k-cO)))``, exact phase.

    ``np.exp`` of the exactly-reduced phase, i.e. the SAME final step
    :func:`naive_rows` takes, so what separates the two kernels is the
    reduction and nothing else.
    """
    return np.exp(1j * sign * TAU
                  * exact_phase(alpha, n_in, n_out, c_in, c_out))


def naive_rows(alpha, n_in, n_out, c_in=0.0, c_out=0.0, sign=-1):
    """The same kernel formed the way the ROUTE forms it."""
    return np.exp(1j * sign * TAU * naive_phase(alpha, n_in, n_out,
                                                c_in, c_out))


def fsum_sandwich(E, Wy, Wx):
    """``F = Wy . E . Wx^T`` with every entry accumulated by ``math.fsum``."""
    My = Wy.shape[0]
    Mx = Wx.shape[0]
    F = np.empty((My, Mx), dtype=np.complex128)
    for ky in range(My):
        wy = Wy[ky]
        for kx in range(Mx):
            T = E * (wy[:, None] * Wx[kx][None, :])
            F[ky, kx] = complex(math.fsum(T.real.ravel()),
                                math.fsum(T.imag.ravel()))
    return F


def rel(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def fit(budgets, vals):
    sel = [(b, v) for b, v in zip(budgets, vals) if 1e-14 < v < 1e-1]
    if len(sel) < 3:
        return None
    x = np.log10([s[0] for s in sel])
    y = np.log10([s[1] for s in sel])
    return {'slope': float(np.polyfit(x, y, 1)[0]), 'n': len(sel),
            'decades': float(x.max() - x.min())}


BUDGETS = (1e5, 1e7, 1e9, 1e10, 1e11, 1e12, 1e14, 1e15)


def main(tree, out_path):
    lumenairy = anchor(tree)
    from lumenairy.propagators._bluestein import (
        _bluestein_2d,
        _bluestein_centred_2d,
        _clear_h_fft_cache,
    )
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    eps = float(np.finfo(np.float64).eps)
    res = {'lumenairy_file': lumenairy.__file__,
           'lumenairy_version': lumenairy.__version__,
           'python': sys.version.split()[0], 'platform': sys.platform,
           'eps': eps}

    # --- the exact reduction's own exactness, both directions --------------
    # A DYADIC alpha makes alpha*n*k exactly representable in float64, so
    # there the two reductions must agree EXACTLY (modulo one whole turn --
    # the tie at |t| = 1/2 is resolved the other way by rint's ties-to-even,
    # and exp of either is the same complex number).
    dyad = {}
    for a in (0.125, 0.5, 2.0 ** -7, 3.0 * 2.0 ** -5):
        d = exact_phase(a, 24, 12) - naive_phase(a, 24, 12)
        dyad[repr(a)] = {
            'max_abs_fractional_part': float(np.max(np.abs(d - np.rint(d)))),
            'max_abs_diff': float(np.max(np.abs(d))),
        }
    res['dyadic_phase_agreement'] = dyad
    # ... and at a budget where the float64 product IS lossy they must part.
    a_big = 1e12 / 24.0 ** 2
    db = exact_phase(a_big, 24, 12) - naive_phase(a_big, 24, 12)
    res['lossy_phase_disagreement_max'] = float(
        np.max(np.abs(db - np.rint(db))))

    geoms = {
        # the SHIPPED id's geometry: _bluestein_2d, cI = cO = 0
        'plain_24x12': dict(n_in=24, n_out=12, c_in=0.0, c_out=0.0,
                            seed=5, centred=False),
        # the round-2 probe's geometry, for the cross-check
        'centred_24x12': dict(n_in=24, n_out=12, c_in=12.0, c_out=6.0,
                              seed=20260920, centred=True),
    }
    res['rows'] = {}
    res['slopes'] = {}
    for tag, g in geoms.items():
        rng = np.random.default_rng(g['seed'])
        n_in, n_out = g['n_in'], g['n_out']
        E = (rng.standard_normal((n_in, n_in))
             + 1j * rng.standard_normal((n_in, n_in))).astype(np.complex128)
        rows = []
        for budget in BUDGETS:
            alpha = budget / float(n_in) ** 2
            We = exact_rows(alpha, n_in, n_out, g['c_in'], g['c_out'])
            Wn = naive_rows(alpha, n_in, n_out, g['c_in'], g['c_out'])
            ref_e = fsum_sandwich(E, We, We)
            ref_n = fsum_sandwich(E, Wn, Wn)
            kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
            if g['centred']:
                fn = _bluestein_centred_2d
                kw.update(n_centre_in_x=g['c_in'], n_centre_in_y=g['c_in'],
                          k_centre_out_x=g['c_out'],
                          k_centre_out_y=g['c_out'])
            else:
                fn = _bluestein_2d
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter('always')
                _clear_h_fft_cache()
                c = fn(E, alpha, alpha, n_out, n_out, **kw)
                _clear_h_fft_cache()
                d = fn(E, alpha, alpha, n_out, n_out, method='direct', **kw)
                warned = [str(x.message) for x in w
                          if issubclass(x.category, RuntimeWarning)]
            rows.append({
                'budget': budget, 'alpha': alpha,
                'eps_budget': eps * budget,
                'chirp_vs_exact': rel(c, ref_e),
                'dense_vs_exact': rel(d, ref_e),
                'dense_vs_naive': rel(d, ref_n),
                'naive_vs_exact': rel(ref_n, ref_e),
                'C_dense': rel(d, ref_e) / (eps * budget),
                'C_chirp': rel(c, ref_e) / (eps * budget),
                'factor_chirp_over_dense': rel(c, ref_e) / rel(d, ref_e),
                'n_warn': len(warned),
            })
        res['rows'][tag] = rows
        for which in ('chirp_vs_exact', 'dense_vs_exact'):
            res['slopes'][f'{tag}:{which}'] = fit(
                [r['budget'] for r in rows], [r[which] for r in rows])

    # --- the THRESHOLD's own two rungs, on the shipped id's geometry -------
    from lumenairy.propagators._bluestein import _PHASE_BUDGET_MAX as PB
    rng = np.random.default_rng(5)
    E = (rng.standard_normal((24, 24))
         + 1j * rng.standard_normal((24, 24))).astype(np.complex128)
    thr = []
    for label, budget in (('0.9x', PB * 0.9), ('2.2x', PB * 2.2)):
        alpha = budget / 24.0 ** 2
        We = exact_rows(alpha, 24, 12)
        ref_e = fsum_sandwich(E, We, We)
        kw = dict(sign=-1, xp=np, fft2=_fft2, ifft2=_ifft2)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _clear_h_fft_cache()
            c = _bluestein_2d(E, alpha, alpha, 12, 12, **kw)
            nc = len([x for x in w if issubclass(x.category, RuntimeWarning)])
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            _clear_h_fft_cache()
            d = _bluestein_2d(E, alpha, alpha, 12, 12, method='direct', **kw)
            nd = len([x for x in w if issubclass(x.category, RuntimeWarning)])
        thr.append({'label': label, 'budget': budget,
                    'chirp_vs_exact': rel(c, ref_e),
                    'dense_vs_exact': rel(d, ref_e),
                    'factor': rel(c, ref_e) / rel(d, ref_e),
                    'chirp_warned': nc, 'dense_warned': nd})
    res['threshold_rungs'] = thr

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print(f"lumenairy.__file__ = {lumenairy.__file__}  "
          f"({res['platform']}, py{res['python']})")
    print("dyadic alpha (exact product): max |frac(t_exact - t_naive)| = "
          + ", ".join(f"{k}:{v['max_abs_fractional_part']:.1e}"
                      for k, v in res['dyadic_phase_agreement'].items()))
    print(f"lossy alpha (budget 1e12): max |frac(t_exact - t_naive)| = "
          f"{res['lossy_phase_disagreement_max']:.3e}")
    for tag, rows in res['rows'].items():
        print(f"--- {tag} (against an EXACT-phase fsum reference) ---")
        print("  budget     chirp/exact  dense/exact  eps*budget   "
              "C_dense  C_chirp  chirp/dense  dense/naive  naive/exact")
        for r in rows:
            print(f"  {r['budget']:.0e}   {r['chirp_vs_exact']:.3e}    "
                  f"{r['dense_vs_exact']:.3e}    {r['eps_budget']:.3e}   "
                  f"{r['C_dense']:7.4f}  {r['C_chirp']:7.4f}  "
                  f"{r['factor_chirp_over_dense']:9.3f}    "
                  f"{r['dense_vs_naive']:.3e}    {r['naive_vs_exact']:.3e}")
    print("--- threshold rungs (plain_24x12) ---")
    for r in res['threshold_rungs']:
        print(f"  {r['label']:>4} PB = {r['budget']:.4e}  chirp "
              f"{r['chirp_vs_exact']:.3e} (warn {r['chirp_warned']})  dense "
              f"{r['dense_vs_exact']:.3e} (warn {r['dense_warned']})  factor "
              f"{r['factor']:.3f}")
    print("slopes:", {k: (None if v is None else
                          (round(v['slope'], 4), v['decades']))
                      for k, v in res['slopes'].items()})
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
