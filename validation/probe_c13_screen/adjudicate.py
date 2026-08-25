"""ITEM 2 -- is 1e-8 the right C13 screen for the traced fits?

For every captured fit: measure what the screen reads, what the three
candidate answers are, and how far each sits from an INDEPENDENT least-squares
oracle.  The screen's own stated requirement is the bar:

    "it must not skip a solve whose two candidates could differ by more than
     ``_LSTSQ_RESID_MARGIN``" (= 1e-6 relative, on ``||b - A x||``)

so the quantity that adjudicates it is the RESIDUAL EXCESS of the answer the
screen would let stand -- the plain normal-equations answer -- over the
attainable minimum, as a function of the rcond the screen reads.

Usage:  LUMENAIRY_ROOT=... python adjudicate.py <fits_dir> [--mp]
"""
import glob
import json
import os
import sys
import warnings

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy as la                                       # noqa: E402
from lumenairy.elements import _lens_traced as LT            # noqa: E402
from oracle import ls_oracle_mp, ls_oracle_refine            # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__


def _resid(A, b, x):
    return float(np.linalg.norm(np.asarray(b) - A @ np.asarray(x)))


def _plain_normal_equations(A, b, det):
    """The answer the screen LETS STAND when it does not fire: normal
    equations, Cholesky, no step-down, no refinement."""
    from scipy.linalg import cho_factor, cho_solve
    G, rhs = (LT._det_normal_equations(A, b) if det
              else (A.T @ A, A.T @ b))
    return cho_solve(cho_factor(G, check_finite=False), rhs,
                     check_finite=False), G


def adjudicate_one(A, b, det, want_mp=False):
    n, M = A.shape
    B = b if b.ndim == 2 else b[:, None]
    x_ne, G = _plain_normal_equations(A, b, det)
    rc = LT._gram_rcond(G)
    s = np.linalg.svd(A, compute_uv=False)
    condA = float(s[0] / s[-1])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        x_qr = LT._solve_lstsq_qr(A, b)
        x_ship = LT._solve_lstsq_thread_safe(A, b, deterministic=det)
    X_ne = x_ne if np.ndim(x_ne) == 2 else np.asarray(x_ne)[:, None]
    X_qr = x_qr if np.ndim(x_qr) == 2 else np.asarray(x_qr)[:, None]
    X_sh = x_ship if np.ndim(x_ship) == 2 else np.asarray(x_ship)[:, None]

    rows = []
    for c in range(B.shape[1]):
        bc = B[:, c]
        x_star, corr = ls_oracle_refine(A, bc)
        pk = float(np.max(np.abs(x_star))) or 1.0
        r_star = _resid(A, bc, x_star)
        rec = dict(
            n=int(n), M=int(M), det=bool(det), rhs=int(c),
            rcond=float(rc), condA=condA,
            screened=bool(rc < LT._LSTSQ_GRAM_RCOND_MIN),
            oracle_corr=[float(v) / pk for v in corr],
            r_star=r_star,
            err_ne=float(np.max(np.abs(X_ne[:, c] - x_star))) / pk,
            err_qr=float(np.max(np.abs(X_qr[:, c] - x_star))) / pk,
            err_ship=float(np.max(np.abs(X_sh[:, c] - x_star))) / pk,
            r_ne=_resid(A, bc, X_ne[:, c]),
            r_qr=_resid(A, bc, X_qr[:, c]),
            r_ship=_resid(A, bc, X_sh[:, c]),
        )
        for k in ('ne', 'qr', 'ship'):
            rec[f'excess_{k}'] = (rec[f'r_{k}'] - r_star) / r_star
        if want_mp and n <= 3000 and M <= 30:
            x_mp = ls_oracle_mp(A, bc)
            rec['oracle_gap_mp'] = float(
                np.max(np.abs(x_mp - x_star))) / pk
            x_dd, _ = ls_oracle_refine(A, bc, residual='dd')
            rec['oracle_gap_dd'] = float(
                np.max(np.abs(x_dd - x_star))) / pk
        rows.append(rec)
    return rows


def main():
    fits_dir = sys.argv[1]
    want_mp = '--mp' in sys.argv
    out = []
    for path in sorted(glob.glob(os.path.join(fits_dir, '*.npz'))):
        d = np.load(path)
        A, b, det = d['A'], d['b'], bool(d['det'][0])
        tag = os.path.splitext(os.path.basename(path))[0]
        for rec in adjudicate_one(A, b, det, want_mp=want_mp):
            rec['tag'] = tag
            out.append(rec)
            print(f"{tag:24s} n={rec['n']:7d} M={rec['M']:3d} "
                  f"rhs={rec['rhs']} rcond={rec['rcond']:.3e} "
                  f"condA={rec['condA']:.3e} screened={rec['screened']!s:5s} "
                  f"| err/pk ne={rec['err_ne']:.2e} qr={rec['err_qr']:.2e} "
                  f"ship={rec['err_ship']:.2e} "
                  f"| resid excess ne={rec['excess_ne']:+.3e} "
                  f"qr={rec['excess_qr']:+.3e} ship={rec['excess_ship']:+.3e}"
                  + (f" | mp-gap={rec['oracle_gap_mp']:.2e} dd-gap={rec['oracle_gap_dd']:.2e}"
                     if 'oracle_gap_mp' in rec else ''))
            sys.stdout.flush()
    with open(os.path.join(fits_dir, 'adjudication.json'), 'w',
              encoding='ascii') as fh:
        json.dump(out, fh, indent=1)
    print(f"\n{len(out)} solves adjudicated -> "
          f"{os.path.join(fits_dir, 'adjudication.json')}")


if __name__ == '__main__':
    main()
