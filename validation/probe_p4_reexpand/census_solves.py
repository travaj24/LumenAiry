"""Census: which least-squares solves does the P4 GBD reexpand path make?

Refutes-or-confirms S14.6's "by construction" claim that the GBD re-expansion
cannot reach ``_solve_lstsq_thread_safe``.  Logs every call's shape, the Gram
rcond, which branch it takes, and the coefficient hash.

Usage:  LUMENAIRY_ROOT=... python census_solves.py [n_repeats]
"""
import hashlib
import sys

import numpy as np

import common as c
from lumenairy.elements import _lens_traced as lt

_LOG = []


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(a, dtype=np.float64)).tobytes()).hexdigest()[:12]


_orig_solve = lt._solve_lstsq_thread_safe
_orig_qr = lt._solve_lstsq_qr
_orig_refine = lt._det_refine
_state = {}


def _spy_qr(A, b):
    _state['qr'] = _state.get('qr', 0) + 1
    return _orig_qr(A, b)


def _spy_refine(A, b, x, small):
    out = _orig_refine(A, b, x, small)
    _state['refine'] = 'None' if out is None else 'applied'
    return out


def _spy_solve(A, b, deterministic=False):
    _state.clear()
    A = np.asarray(A)
    if deterministic:
        G, rhs = lt._det_normal_equations(A, b)
    else:
        G = A.T @ A
    rcond = lt._gram_rcond(G)
    out = _orig_solve(A, b, deterministic=deterministic)
    _LOG.append(dict(shape=tuple(A.shape), det=bool(deterministic),
                     rcond=float(rcond), qr=_state.get('qr', 0),
                     refine=_state.get('refine', '-'),
                     coef=_h(out), gram=_h(G)))
    return out


lt._solve_lstsq_thread_safe = _spy_solve
lt._solve_lstsq_qr = _spy_qr
lt._det_refine = _spy_refine

if __name__ == '__main__':
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    E = c.conv_input()
    presc = c.m5_biconcave()
    for it in range(reps):
        _LOG.clear()
        diag = {}
        E_a = c.gbd(E, presc, reexpand='auto', diagnostics=diag)
        n_a = len(_LOG)
        E_b = c.gbd(E, presc, reexpand='auto')
        print(f"--- iteration {it}: {len(_LOG)} solves "
              f"({n_a} diagnosed / {len(_LOG) - n_a} undiagnosed) ---")
        for i, r in enumerate(_LOG):
            print(f"  [{i}] shape={r['shape']} det={r['det']} "
                  f"rcond={r['rcond']:.3e} qr_calls={r['qr']} "
                  f"refine={r['refine']} gram={r['gram']} coef={r['coef']}")
        print(f"  equal={np.array_equal(E_a, E_b)} "
              f"hash_a={c.field_hash(E_a)} hash_b={c.field_hash(E_b)} "
              f"comp={diag['frame_completeness']!r} "
              f"reexp={diag['reexpanded']} nb={diag['n_beamlets']}")
