"""ITEM 2 -- DERIVE the screen instead of reading one build's fits.

``_LSTSQ_GRAM_RCOND_MIN`` = 1e-8 is justified in the source by an argument, not
a measurement: "the normal equations lose ~cond(G) eps, so at the screen the
most a skipped solve can be off is ~1e-8, a hundredfold inside the
``_LSTSQ_RESID_MARGIN`` of 1e-6".  This measures that claim on a LADDER of
conditioning built from the traced fits' OWN design matrix, so the answer is a
property of this fit family rather than of whichever rcond one build happened
to produce.

Method (TESTING_STANDARDS rule 3 -- engineer the state, do not hope for it):
take a captured ``A``, SVD it, and replace its singular-value spectrum by a
geometric one with a PRESCRIBED ``cond(A)``.  The column geometry, the row
count and the right-hand side's residual fraction are the fit's own; only the
conditioning moves.  Then, at each rung, measure

  * what the screen reads (``_gram_rcond`` of the equilibrated Gram);
  * the RESIDUAL EXCESS of the plain normal-equations answer over the oracle
    minimum -- the quantity ``_LSTSQ_RESID_MARGIN`` is stated in;
  * the COEFFICIENT error of the same answer -- the quantity that reaches the
    Newton loop and the field.

The screen must fire at or above the rcond where the first of those crosses
the margin.  Where that crossing sits is the derived bar.

Usage:  LUMENAIRY_ROOT=... python ladder.py <A.npz> [out.json]
"""
import json
import os
import sys
import warnings

import numpy as np

_ROOT = os.environ['LUMENAIRY_ROOT']
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from oracle import ls_oracle_refine  # noqa: E402

import lumenairy as la  # noqa: E402
from lumenairy.elements import _lens_traced as LT  # noqa: E402

_want = os.path.realpath(os.path.join(_ROOT, 'lumenairy'))
assert os.path.realpath(os.path.dirname(la.__file__)) == _want, la.__file__


def _reconditioned(A, cond_target, rng):
    """``A`` with its singular spectrum replaced by a geometric one of the
    requested condition number; same U, same V, same shape."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    m = s.size
    s_new = s[0] * np.geomspace(1.0, 1.0 / cond_target, m)
    return (U * s_new[None, :]) @ Vt


def main():
    src = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else 'ladder.json'
    d = np.load(src)
    A0 = np.ascontiguousarray(d['A'], dtype=np.float64)
    b0 = np.asarray(d['b'], dtype=np.float64)
    if b0.ndim == 2:
        b0 = b0[:, 0]
    rng = np.random.default_rng(20260824)
    # the fit's OWN residual fraction, so the rung's b is as far from the
    # column space as the real one is (the refinement's convergence rate and
    # the residual's sensitivity both depend on it).
    x0, _ = ls_oracle_refine(A0, b0, steps=2)
    frac = float(np.linalg.norm(b0 - A0 @ x0) / np.linalg.norm(b0))
    print(f"source {os.path.basename(src)}  A={A0.shape}  "
          f"residual fraction {frac:.3e}")

    rows = []
    x_true = rng.standard_normal(A0.shape[1])
    for cond in np.geomspace(1e2, 1e11, 19):
        A = _reconditioned(A0, cond, rng)
        clean = A @ x_true
        noise = rng.standard_normal(A.shape[0])
        noise *= frac * np.linalg.norm(clean) / np.linalg.norm(noise)
        b = clean + noise
        G, rhs = LT._det_normal_equations(A, b)
        rc = float(LT._gram_rcond(G))
        s = np.linalg.svd(A, compute_uv=False)
        condA = float(s[0] / s[-1])
        x_star, corr = ls_oracle_refine(A, b, steps=4)
        pk = float(np.max(np.abs(x_star))) or 1.0
        r_star = float(np.linalg.norm(b - A @ x_star))
        from scipy.linalg import cho_factor, cho_solve
        try:
            x_ne = cho_solve(cho_factor(G, check_finite=False), rhs,
                             check_finite=False)
        except Exception as exc:                     # noqa: BLE001
            rows.append(dict(cond=float(cond), condA=condA, rcond=rc,
                             failed=f"{type(exc).__name__}: {exc}"))
            print(f"  cond(A)={condA:.2e} rcond={rc:.3e}  "
                  f"CHOLESKY FAILED ({type(exc).__name__})")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            x_qr = LT._solve_lstsq_qr(A, b)
        r_ne = float(np.linalg.norm(b - A @ x_ne))
        r_qr = float(np.linalg.norm(b - A @ x_qr))
        rec = dict(cond=float(cond), condA=condA, rcond=rc,
                   screened=bool(rc < LT._LSTSQ_GRAM_RCOND_MIN),
                   err_ne=float(np.max(np.abs(x_ne - x_star))) / pk,
                   err_qr=float(np.max(np.abs(x_qr - x_star))) / pk,
                   excess_ne=(r_ne - r_star) / r_star,
                   excess_qr=(r_qr - r_star) / r_star,
                   oracle_corr=[float(v) / pk for v in corr])
        rows.append(rec)
        print(f"  cond(A)={condA:.2e} rcond={rc:.3e} "
              f"screened={rec['screened']!s:5s} | err/pk ne={rec['err_ne']:.2e}"
              f" qr={rec['err_qr']:.2e} | resid excess ne="
              f"{rec['excess_ne']:+.3e} qr={rec['excess_qr']:+.3e} "
              f"| oracle last corr {rec['oracle_corr'][-1]:.1e}")
        sys.stdout.flush()

    with open(out_path, 'w', encoding='ascii') as fh:
        json.dump(dict(source=os.path.basename(src), shape=list(A0.shape),
                       residual_fraction=frac, rows=rows), fh, indent=1)
    print(f"-> {out_path}")


if __name__ == '__main__':
    main()
