"""S14.6 stressor -- the ONE BLAS-adjacent step on the P4 path, at high count.

The P4 ``reexpand='auto'`` path makes exactly one least-squares solve per call:
``_compute_carrier``'s ``'auto'`` fit, an ``(82092, 5)`` weighted gradient fit,
routed through ``_solve_lstsq_thread_safe`` (census_solves.py).  Everything else
on the path is elementwise NumPy plus ``bincount``.  So this is the only step
whose bits could plausibly move with the BLAS thread count or with what else the
box is running -- and it is milliseconds, so it can be hammered tens of
thousands of times where the whole call can only be run hundreds.

Records, per iteration: the coefficient hash, the Gram hash, the equilibrated
Gram rcond (``_gram_rcond``'s ``eigvalsh``, the one unowned reduction the
5.42.0 audit named), and whether the C13 screen fired.

Usage:  python stress_fit.py <n_iters> <arm_id> <out.jsonl> [A.npz]
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

import common as c  # noqa: F401  (pins the tree + registers the model glass)
from lumenairy.elements import _lens_traced as lt


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def capture(path):
    """Run one P4 call with the solver spied on; save its (A, rhs)."""
    cap = {}
    orig = lt._solve_lstsq_thread_safe

    def spy(A, b, deterministic=False):
        cap.setdefault('A', np.array(A))
        cap.setdefault('b', np.array(b))
        cap.setdefault('det', bool(deterministic))
        return orig(A, b, deterministic=deterministic)

    lt._solve_lstsq_thread_safe = spy
    try:
        c.gbd(c.conv_input(), c.m5_biconcave(), reexpand='auto')
    finally:
        lt._solve_lstsq_thread_safe = orig
    np.savez(path, A=cap['A'], b=cap['b'], det=np.array([cap['det']]))
    print(f"captured A{cap['A'].shape} b{cap['b'].shape} "
          f"deterministic={cap['det']} -> {path}")


def main():
    n = int(sys.argv[1])
    arm = sys.argv[2]
    out = sys.argv[3]
    npz = sys.argv[4] if len(sys.argv) > 4 else 'fit_AB.npz'
    if not os.path.exists(npz):
        capture(npz)
    d = np.load(npz)
    A, b, det = d['A'], d['b'], bool(d['det'][0])
    h0 = None
    bad = 0
    with open(out, 'w', encoding='ascii') as fh:
        for it in range(n):
            t0 = time.perf_counter()
            G, rhs = (lt._det_normal_equations(A, b) if det
                      else (A.T @ A, A.T @ b))
            rc = lt._gram_rcond(G)
            x = lt._solve_lstsq_thread_safe(A, b, deterministic=det)
            hx = _h(x)
            if h0 is None:
                h0 = hx
            if hx != h0:
                bad += 1
            fh.write(json.dumps({
                'arm': arm, 'it': it, 'pid': os.getpid(),
                'omp': os.environ.get('OMP_NUM_THREADS', '(unset)'),
                'coef': hx, 'gram': _h(G), 'rhs': _h(rhs),
                'rcond': float(rc),
                'screened': bool(rc < lt._LSTSQ_GRAM_RCOND_MIN),
                'drift': bool(hx != h0),
                'dt': round(time.perf_counter() - t0, 4)}) + '\n')
    print(f"arm {arm} pid {os.getpid()}: {n} solves, {bad} coefficient drifts, "
          f"rcond={rc:.4e}")


if __name__ == '__main__':
    main()
