"""WP-C4 -- the dense route's bits MOVE with the BLAS kernel, and the DECISION
does not.

The dense route is two ``xp.matmul`` calls, so its last bits belong to whatever
GEMM the process linked and how many threads it used.  Nothing in this branch
pins a dense-route byte; this file is the measurement that says why, and its
other half is the measurement that says what is safe to pin instead.

Driven by ``c4_blas_sweep_run.py``, which re-execs this module once per
(``OPENBLAS_CORETYPE``, thread count) cell -- the kernel is chosen when OpenBLAS
is first loaded, so it cannot be changed inside a live process.

THREE THINGS ARE RECORDED PER CELL:

* ``dense_digest`` -- SHA-256 over the dense answer's raw bytes.  Expected to
  MOVE between cells; if it does not, the sweep proved nothing and says so.
* ``route`` -- which arm ``'auto'`` took, and whether it agrees with
  ``_auto_selects_direct``.  Must NOT move: the rule reads four integers.
* ``vs_exact`` -- the dense answer's distance from an exactly-reduced
  ``math.fsum`` reference, and the derived bar.  Must stay inside the bar in
  every cell: that is the form the accuracy claim is stated in, and it is
  stated that way BECAUSE the digest moves.

    PYTHONPATH=<tree> OPENBLAS_CORETYPE=<k> OPENBLAS_NUM_THREADS=<t> \\
        python c4_blas_sweep.py <tree> <label> OUT.json
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import c4lib  # noqa: E402

TAU = 2.0 * math.pi

#: two shapes the rule sends to the dense route, one it leaves on the chirp-Z
#: route -- the third is the control that shows the sweep is not simply moving
#: everything.
SHAPES = [(96, 3), (128, 4), (24, 12)]


def exact_rows(alpha, n_in, n_out, sign=-1):
    import numpy as np
    fa = Fraction(alpha)
    T = np.empty((n_out, n_in), dtype=np.float64)
    for k in range(n_out):
        fk = Fraction(k)
        for n in range(n_in):
            t = fa * fk * n
            t -= math.floor(t)
            if t >= Fraction(1, 2):
                t -= 1
            T[k, n] = float(t)
    return np.exp(1j * sign * TAU * T)


def fsum_reference(E, alpha, M, sign=-1):
    import numpy as np
    ny, nx = E.shape
    Wy = exact_rows(alpha, ny, M, sign)
    Wx = exact_rows(alpha, nx, M, sign)
    out = np.empty((M, M), dtype=np.complex128)
    for ky in range(M):
        wy = Wy[ky]
        for kx in range(M):
            T = E * (wy[:, None] * Wx[kx][None, :])
            out[ky, kx] = complex(math.fsum(T.real.ravel()),
                                  math.fsum(T.imag.ravel()))
    return out


def main(tree, label, out_path):
    lum = c4lib.anchor(tree)
    import numpy as np
    from lumenairy.propagators import _bluestein as B
    from lumenairy.propagators.fft_infra import _fft2, _ifft2

    eps = float(np.finfo(np.float64).eps)
    cfg = {}
    try:
        cfg = {k: v for k, v in np.show_config(mode='dicts').items()
               if 'Build Dependencies' in str(k) or 'blas' in str(k).lower()}
    except Exception as exc:                        # noqa: BLE001 -- recorded
        cfg = {'error': f"{type(exc).__name__}: {exc}"}

    out = {'label': label, 'build': c4lib.build_tag(),
           'lumenairy_file': lum.__file__,
           'env': {k: os.environ.get(k) for k in
                   ('OPENBLAS_CORETYPE', 'OPENBLAS_NUM_THREADS',
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
           'numpy': np.__version__, 'numpy_config': str(cfg)[:1200],
           'rows': []}
    for (N, M) in SHAPES:
        rng = np.random.default_rng(496)
        E = (rng.standard_normal((N, N))
             + 1j * rng.standard_normal((N, N))).astype(np.complex128)
        alpha = 1.0e3 / float(max(N, M)) ** 2
        says = bool(B._auto_selects_direct(N, N, M, M))
        c4lib.cold()
        auto = B._bluestein_2d(E, alpha, alpha, M, M, sign=-1, xp=np,
                               fft2=_fft2, ifft2=_ifft2)
        c4lib.cold()
        dense = B._bluestein_2d(E, alpha, alpha, M, M, sign=-1, xp=np,
                                fft2=_fft2, ifft2=_ifft2, method='direct')
        c4lib.cold()
        chirp = B._bluestein_2d(E, alpha, alpha, M, M, sign=-1, xp=np,
                                fft2=_fft2, ifft2=_ifft2, method='bluestein')
        ref = fsum_reference(E, alpha, M)
        g_dense = math.sqrt(float(E.size))
        max_t = abs(alpha) * float(N - 1) * float(max(M - 1, 1))
        bar_d = (g_dense + 1.0 + TAU * max_t) * eps * float(np.sum(np.abs(E)))
        dd = hashlib.sha256(np.ascontiguousarray(dense).tobytes()).hexdigest()
        cd = hashlib.sha256(np.ascontiguousarray(chirp).tobytes()).hexdigest()
        ad = hashlib.sha256(np.ascontiguousarray(auto).tobytes()).hexdigest()
        out['rows'].append({
            'N': N, 'M': M, 'rule_says_direct': says,
            'auto_digest': ad, 'dense_digest': dd, 'chirp_digest': cd,
            'route': ('direct' if ad == dd
                      else 'bluestein' if ad == cd else 'NONE'),
            'route_agrees_with_rule': ((ad == dd) == says),
            'dense_vs_exact_max_abs': float(np.max(np.abs(dense - ref))),
            'derived_bar_dense': bar_d,
            'dense_inside_bar': bool(np.max(np.abs(dense - ref)) < bar_d)})
        del E
    c4lib.write_json(out, out_path)
    for r in out['rows']:
        print(f"[{label}] N={r['N']:4d} M={r['M']:3d} rule="
              f"{'dense' if r['rule_says_direct'] else 'chirp':5s} route="
              f"{r['route']:9s} agrees={r['route_agrees_with_rule']} "
              f"dense_sha={r['dense_digest'][:12]} "
              f"vs_exact={r['dense_vs_exact_max_abs']:.4e} "
              f"bar={r['derived_bar_dense']:.4e} "
              f"inside={r['dense_inside_bar']}")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
