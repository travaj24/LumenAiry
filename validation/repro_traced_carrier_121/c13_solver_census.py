# Niche C13 -- WHICH of a runner's least-squares solves the conditioning
# step-down actually touches, and by how much.
#
#   python c13_solver_census.py focus_scan_121.py
#   ORDERS='-1,0' python c13_solver_census.py rc_resdeg_121.py
#
# Wraps ``_solve_lstsq_thread_safe`` script-side (no library edit) and records,
# per call: the shape, the EQUILIBRATED Gram's reciprocal condition number (the
# screen ``_LSTSQ_GRAM_RCOND_MIN`` reads), and -- for the calls that screen in
# -- the two candidates' fit residuals, so the reroute decision is visible
# rather than asserted.
#
# WHY IT EXISTS.  The C13 claim "design 121's production acceptance is
# unchanged" needs a mechanism, not just a matching printout: the on-axis
# production route takes the CONCENTRIC hard-mask branch, whose fits are
# well conditioned, so the screen skips them all.  This is what measures that.
import hashlib
import os
import runpy
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

import _d121_common as _C                                      # noqa: E402,F401
import lumenairy                                               # noqa: E402
import lumenairy.elements._lens_traced as LT                   # noqa: E402

_ROWS = []
_REAL = LT._solve_lstsq_thread_safe


def _census(A, b):
    A64 = np.ascontiguousarray(A, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    G = A64.T @ A64
    rc = LT._gram_rcond(G)
    out = _REAL(A64, b64)
    row = [A64.shape[0], A64.shape[1], rc, np.nan, np.nan, 0]
    if rc < LT._LSTSQ_GRAM_RCOND_MIN and LT.LSTSQ_CONDITIONING_STEPDOWN:
        try:
            from scipy.linalg import cho_factor, cho_solve
            x_ne = cho_solve(cho_factor(G, check_finite=False), A64.T @ b64,
                             check_finite=False)
            row[3] = float(np.linalg.norm(b64 - A64 @ x_ne))
            row[4] = float(np.linalg.norm(b64 - A64 @ out))
            row[5] = int(not np.array_equal(out, x_ne))
        except Exception:
            pass
    _ROWS.append(row)
    return out


LT._solve_lstsq_thread_safe = _census

_h = hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]
print(f"[c13_solver_census] STEPDOWN={LT.LSTSQ_CONDITIONING_STEPDOWN}  "
      f"rcond_min={LT._LSTSQ_GRAM_RCOND_MIN:.1e}  "
      f"lumenairy {lumenairy.__version__}  _lens_traced {_h}", flush=True)

sys.argv = sys.argv[1:]
try:
    runpy.run_path(os.path.abspath(sys.argv[0]), run_name='__main__')
finally:
    A = np.array(_ROWS, dtype=float) if _ROWS else np.zeros((0, 6))
    n = len(A)
    scr = int((A[:, 2] < LT._LSTSQ_GRAM_RCOND_MIN).sum()) if n else 0
    rer = int(A[:, 5].sum()) if n else 0
    print(f"\n[c13_solver_census] {n} solves | {scr} screened in "
          f"({100.0 * scr / max(n, 1):.1f} %) | {rer} REROUTED "
          f"({100.0 * rer / max(n, 1):.1f} %)", flush=True)
    if n:
        print("[c13_solver_census] gram rcond quantiles "
              "(min/p10/median/p90/max): "
              + ' '.join('%.2e' % v for v in
                         np.quantile(A[:, 2], [0, .1, .5, .9, 1.0])))
        m = A[:, 5] > 0
        if m.any():
            g = A[m, 3] / np.where(A[m, 4] > 0, A[m, 4], np.nan)
            print("[c13_solver_census] rerouted calls, residual ratio "
                  "ne/qr  min %.4f  median %.4f  max %.4f"
                  % (np.nanmin(g), np.nanmedian(g), np.nanmax(g)))
            for shape in sorted({(int(r[0]), int(r[1])) for r in A[m]}):
                k = int(((A[:, 0] == shape[0]) & (A[:, 1] == shape[1])
                         & m).sum())
                print("[c13_solver_census]   %6d x %-4d rerouted %d times"
                      % (shape[0], shape[1], k))
