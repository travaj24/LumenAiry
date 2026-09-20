"""VERIFY-WAVE5-HYGIENE2 round 2 -- the V-D5 chirp phase budget, re-measured.

Three readings on MY OWN grids, none of them the shipped fixture:

 B1  THE LAW.  For >= 6 budgets over >= 5 decades, the relative L2 of each
     chirp-Z route against a ``math.fsum`` CORRECTLY-ROUNDED evaluation of the
     same sum.  The reference is summed term by term in float64 complex with
     the real and imaginary parts accumulated by ``fsum`` separately, so it
     carries one rounding at the end and nothing else.
 B2  THE THRESHOLD, two-sided: at ``0.9 * _PHASE_BUDGET_MAX`` the primitive
     must be SILENT and the answer must still keep six figures; at
     ``1.1 *`` it must WARN.  A decade below the threshold it must still be
     silent -- otherwise the constant is not where its derivation puts it.
 B3  THE CONSTANT'S DERIVATION: ``_PHASE_BUDGET_MAX * eps == 1e-6`` exactly,
     and the warning names the error the budget implies.

    PYTHONPATH=<tree> python vh3_budget.py OUT.json
"""
import cmath
import json
import math
import sys
import warnings

import numpy as np


def fsum_reference(E, alpha_x, alpha_y, N_out_y, N_out_x, sign,
                   cIx, cIy, cOx, cOy):
    """The centred 2-D sum, accumulated by ``math.fsum``.

    ``fsum`` is correctly rounded, so each output entry carries exactly one
    rounding of the exact sum of the float64 terms.  The phase is reduced by
    ``t - rint(t)`` before ``cmath.exp`` for the same reason the dense route
    does it: otherwise the REFERENCE spends its own mantissa on the chirp and
    is as wrong as the thing it is measuring.
    """
    Ny, Nx = E.shape
    out = np.empty((N_out_y, N_out_x), dtype=np.complex128)
    tau = 2.0 * math.pi
    for ky in range(N_out_y):
        for kx in range(N_out_x):
            re, im = [], []
            for ny in range(Ny):
                ty = alpha_y * (ny - cIy) * (ky - cOy)
                ty -= round(ty)
                wy = cmath.exp(sign * tau * 1j * ty)
                for nx in range(Nx):
                    tx = alpha_x * (nx - cIx) * (kx - cOx)
                    tx -= round(tx)
                    w = wy * cmath.exp(sign * tau * 1j * tx)
                    v = complex(E[ny, nx]) * w
                    re.append(v.real)
                    im.append(v.imag)
            out[ky, kx] = complex(math.fsum(re), math.fsum(im))
    return out


def relL2(a, b):
    return float(np.linalg.norm(np.asarray(a) - np.asarray(b))
                 / np.linalg.norm(np.asarray(b)))


def fixture(n, seed=20260919):
    """A deterministic complex input -- not a Gaussian, so no entry is
    negligible and the reduction has real cancellation in it."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((n, n))
            + 1j * rng.standard_normal((n, n))).astype(np.complex128)


def main(out_path):
    import lumenairy
    from lumenairy.propagators import _bluestein as BL

    eps = float(np.finfo(np.float64).eps)
    res = {'lumenairy_file': lumenairy.__file__,
           'python': sys.version.split()[0], 'platform': sys.platform,
           'PHASE_BUDGET_MAX': float(BL._PHASE_BUDGET_MAX),
           'eps': eps,
           'B3_budget_times_eps': float(BL._PHASE_BUDGET_MAX) * eps}

    # --- B1: the law, on MY grids -----------------------------------------
    b1 = {}
    for (nin, nout) in ((20, 10), (28, 14)):
        E = fixture(nin)
        cI = nin / 2.0
        cO = nout / 2.0
        Nmax = max(nin, nout)
        rows = []
        for budget in (1e5, 1e7, 1e8, 1e9, 4.5036e9, 1e10, 1e11, 1e12,
                       1e13, 1e15, 1e16):
            alpha = budget / Nmax ** 2
            ref = fsum_reference(E, alpha, alpha, nout, nout, -1,
                                 cI, cI, cO, cO)
            row = {'budget': budget, 'alpha': alpha}
            for meth in ('bluestein', 'separable', 'direct'):
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    F = BL._bluestein_centred_2d(
                        E, alpha, alpha, nout, nout, sign=-1, xp=np,
                        fft2=np.fft.fft2, ifft2=np.fft.ifft2, method=meth)
                    row[f'warned_{meth}'] = any(
                        issubclass(x.category, RuntimeWarning) for x in w)
                row[meth] = relL2(F, ref)
            row['eps_times_budget'] = eps * budget
            rows.append(row)
        b1[f'{nin}x{nout}'] = rows
    res['B1'] = b1

    # the fitted slope of log10(rel) vs log10(budget) over the range where
    # the chirp error is above its own floor and below saturation
    fits = {}
    for key, rows in b1.items():
        sel = [r for r in rows if 1e-13 < r['bluestein'] < 1e-2]
        if len(sel) >= 3:
            x = np.log10([r['budget'] for r in sel])
            y = np.log10([r['bluestein'] for r in sel])
            fits[key] = {'slope': float(np.polyfit(x, y, 1)[0]),
                         'n_points': len(sel),
                         'decades': float(x.max() - x.min())}
    res['B1_slope'] = fits

    # --- B2: the threshold, two-sided and a decade below ------------------
    b2 = {}
    nin, nout = 20, 10
    E = fixture(nin)
    Nmax = max(nin, nout)
    PB = float(BL._PHASE_BUDGET_MAX)
    for tag, budget in (('a_decade_below', PB / 10.0),
                        ('just_below', 0.9 * PB),
                        ('just_above', 1.1 * PB),
                        ('a_decade_above', 10.0 * PB)):
        alpha = budget / Nmax ** 2
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            F = BL._bluestein_centred_2d(
                E, alpha, alpha, nout, nout, sign=-1, xp=np,
                fft2=np.fft.fft2, ifft2=np.fft.ifft2, method='bluestein')
            msgs = [str(x.message) for x in w
                    if issubclass(x.category, RuntimeWarning)]
        ref = fsum_reference(E, alpha, alpha, nout, nout, -1,
                             nin / 2.0, nin / 2.0, nout / 2.0, nout / 2.0)
        b2[tag] = {'budget': budget, 'warned': bool(msgs),
                   'rel': relL2(F, ref),
                   'keeps_six_figures': relL2(F, ref) < 1e-6,
                   'message': msgs[0] if msgs else None}
        # and the dense route on the same budget: it must never warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Fd = BL._bluestein_centred_2d(
                E, alpha, alpha, nout, nout, sign=-1, xp=np,
                fft2=np.fft.fft2, ifft2=np.fft.ifft2, method='direct')
            b2[tag]['dense_warned'] = any(
                issubclass(x.category, RuntimeWarning) for x in w)
        b2[tag]['dense_rel'] = relL2(Fd, ref)
    res['B2'] = b2

    # --- B3: the message names the implied error --------------------------
    msg = b2['just_above']['message'] or ''
    res['B3'] = {
        'derivation_exact': abs(res['B3_budget_times_eps'] - 1e-6) < 1e-21,
        'message_names_direct': "method='direct'" in msg,
        'message_names_budget': f"{PB:.1e}" in msg,
        'message_names_implied_error': (
            f"{1.1 * PB * eps:.1e}" in msg),
        'message': msg,
    }

    with open(out_path, 'w', encoding='cp1252') as fh:
        json.dump(res, fh, indent=1, sort_keys=True, default=str)
    print(f"lumenairy.__file__ = {lumenairy.__file__}")
    print(f"_PHASE_BUDGET_MAX = {PB:.6e}   x eps = "
          f"{res['B3_budget_times_eps']:.6e}")
    for key, rows in b1.items():
        print(f"--- {key} ---")
        print("  budget      bluestein  separable  direct     eps*budget  "
              "warned")
        for r in rows:
            print(f"  {r['budget']:.2e}  {r['bluestein']:.3e}  "
                  f"{r['separable']:.3e}  {r['direct']:.3e}  "
                  f"{r['eps_times_budget']:.3e}   "
                  f"{r['warned_bluestein']}")
    print("slopes:", {k: round(v['slope'], 4) for k, v in fits.items()})
    for tag, v in b2.items():
        print(f"B2 {tag:16s} budget {v['budget']:.3e}  warned "
              f"{v['warned']!s:5s}  rel {v['rel']:.3e}  6fig "
              f"{v['keeps_six_figures']!s:5s}  dense {v['dense_rel']:.3e} "
              f"(warned {v['dense_warned']})")
    print("B3:", {k: v for k, v in res['B3'].items() if k != 'message'})
    print(f"-> {out_path}")


if __name__ == '__main__':
    main(sys.argv[1])
