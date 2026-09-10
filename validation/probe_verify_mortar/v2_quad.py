"""V2c -- the far-field projector's QUADRATURE ORDER on non-uniform segments.

``_stag_fourier_projection`` integrates ``phi_a(x) e^{+i(mG + a0)x}`` with a
FIXED ``nq = 2 M + 8`` Gauss-Legendre rule PER SEGMENT.  On a uniform lattice a
segment is ``d / N`` long, so the phase a segment carries is at most
``2 pi m_max / N`` and the rule is sized for it.  With ARBITRARY walls a single
segment can be almost the whole period, and the phase it carries grows to
``2 pi m_max``.  The rule was NOT re-sized: this file measures whether that
matters, kernel-level and device-level.

``python v2_quad.py [kernel device]``
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import sys
import time
import warnings

import numpy as np
from numpy.polynomial.legendre import leggauss

import lumenairy
from lumenairy.elements.pmm import twod_staggered as ts
from lumenairy.elements.pmm.stack import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT)
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
RES = {}
WL = 1.0e-6
PX = 0.9e-6
PY = 0.9e-6

_ORIG_FOUR = ts._stag_fourier_projection


def refined_fourier(mult):
    """The SHIPPED projector with the per-segment quadrature order multiplied
    by ``mult`` -- everything else identical."""
    def _f(basis, orders, alpha0=0.0):
        d, N, M = basis.d, basis.N, basis.M
        G = 2.0 * np.pi / d
        xb = basis.xb
        xg, wg = leggauss(mult * (2 * M + 8))
        Vref, _ = ts._modleg_value_deriv(M, xg)
        orders = np.asarray(orders)
        T_local = np.zeros((len(orders), N, M), dtype=_C)
        for seg in range(N):
            J = basis.Jn[seg]
            xphys = 0.5 * (xb[seg] + xb[seg + 1]) + J * xg
            phase = np.exp(1j * np.outer(orders * G + alpha0, xphys))
            T_local[:, seg, :] = (J / d) * (phase * wg) @ Vref.T

        def _assemble(gs):
            return np.einsum("msa,jsa->mj", T_local, np.array(gs))
        return _assemble
    return _f


def sec_kernel():
    """Kernel-level: relative error of the SHIPPED rule vs a 8x-refined one,
    as a function of the longest segment and the highest order."""
    out = {}
    for Lmax, xb in ((1 / 3, np.array([0.0, 1 / 3, 2 / 3, 1.0])),
                     (0.50, np.array([0.0, 0.25, 0.75, 1.0])),
                     (0.62, np.array([0.0, 0.19, 0.81, 1.0])),
                     (0.80, np.array([0.0, 0.10, 0.90, 1.0])),
                     (0.91, np.array([0.0, 0.02, 0.93, 1.0])),
                     (0.96, np.array([0.0, 0.02, 0.98, 1.0]))):
        for M in (4, 5, 6, 8):
            for mmax in (3, 5, 7):
                b = ts.Basis1D(1.0, xb, M, np.exp(-1j * 0.31))
                orders = np.arange(-mmax, mmax + 1)
                lib = _ORIG_FOUR(b, orders, 0.31)(b.Btilde)
                ora = refined_fourier(8)(b, orders, 0.31)(b.Btilde)
                sc = max(float(np.max(np.abs(ora))), 1e-300)
                out[f"L{Lmax:.2f}_M{M}_m{mmax}"] = float(
                    np.max(np.abs(lib - ora)) / sc)
        print(f"[kernel] longest segment {Lmax:.2f} d: "
              + "  ".join(f"M{M}/m{m}={out[f'L{Lmax:.2f}_M{M}_m{m}']:.1e}"
                          for M in (4, 6) for m in (3, 7)), flush=True)
    RES["kernel"] = out


def _solve(walls, tile, M, n_orders, theta, phi=0.0, t=0.30e-6):
    st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=n_orders, layer_grids="per-layer")
    st.add_layer(t, eps_cell=tile, x_walls=[w * PX for w in walls],
                 y_walls=[w * PY for w in walls])
    st.set_source(WL, theta=theta, phi=phi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(jones=False)


def sec_device():
    """Device-level: does the projector's quadrature order move R/T, and by
    how much relative to the discretisation error against the EXACT 1-D
    oracle?"""
    out = {}
    walls = [0.02, 0.93]          # longest segment 0.91 of the period
    eps = [6.0, 2.25, 3.1]
    tile = np.zeros((3, 3), dtype=_C)
    for i, v in enumerate(eps):
        tile[i, :] = v
    segs = [(walls[0], eps[0]), (walls[1] - walls[0], eps[1]),
            (1.0 - walls[1], eps[2])]
    theta = 0.21
    ref = {}
    for deg in (12, 14):
        st = PMMStack(PX, n_superstrate=1.0, n_substrate=1.45, degree=deg,
                      n_orders=9)
        st.add_layer(0.30e-6, segments=segs)
        st.set_source(WL, theta=theta)
        o, R, T, _J = st.solve()
        ref[deg] = (np.asarray(o).ravel().astype(int), R, T)
    out["oracle_selfgap_12_14"] = float(max(
        np.max(np.abs(ref[12][1] - ref[14][1])),
        np.max(np.abs(ref[12][2] - ref[14][2]))))
    o1, R1, T1 = ref[14]
    idx1 = {int(m): i for i, m in enumerate(o1)}
    print(f"[device] exact 1-D oracle self-gap deg12-14 = "
          f"{out['oracle_selfgap_12_14']:.3e}", flush=True)
    lad = {}
    for M in (4, 5, 6, 7):
        n_ord = min(4, (3 * (M - 1) - 1) // 2)
        arms = {}
        for label, fn in (("shipped", _ORIG_FOUR),
                          ("quad_x4", refined_fourier(4)),
                          ("quad_x8", refined_fourier(8))):
            ts._stag_fourier_projection = fn
            try:
                arms[label] = _solve(walls, tile, M, n_ord, theta)
            finally:
                ts._stag_fourier_projection = _ORIG_FOUR
        o2, Rs, Ts = arms["shipped"]
        _o, R8, T8 = arms["quad_x8"]
        d_quad = float(max(np.max(np.abs(Rs - R8)), np.max(np.abs(Ts - T8))))
        _o4, R4, T4 = arms["quad_x4"]
        d_quad48 = float(max(np.max(np.abs(R4 - R8)),
                             np.max(np.abs(T4 - T8))))

        def _err(R, T):
            e = 0.0
            for k, (mx, my) in enumerate(o2):
                if int(my) != 0 or int(mx) not in idx1:
                    continue
                j = idx1[int(mx)]
                e = max(e, float(np.max(np.abs(R[:, k] - R1[:, j]))),
                        float(np.max(np.abs(T[:, k] - T1[:, j]))))
            return e
        e_s, e_8 = _err(Rs, Ts), _err(R8, T8)
        c_s = float(np.max(np.abs(Rs.sum(axis=1) + Ts.sum(axis=1) - 1.0)))
        c_8 = float(np.max(np.abs(R8.sum(axis=1) + T8.sum(axis=1) - 1.0)))
        lad[M] = {"n_orders": n_ord, "shipped_vs_x8": d_quad,
                  "x4_vs_x8": d_quad48, "err_shipped": e_s, "err_x8": e_8,
                  "closure_shipped": c_s, "closure_x8": c_8}
        print(f"[device] M={M} n_orders={n_ord}: shipped-vs-8x quadrature "
              f"{d_quad:.3e} (4x-vs-8x {d_quad48:.1e}) | err vs EXACT 1-D "
              f"{e_s:.3e} -> {e_8:.3e} | closure {c_s:.2e} -> {c_8:.2e}",
              flush=True)
    out["ladder"] = lad
    RES["device"] = out


SECTIONS = {"kernel": sec_kernel, "device": sec_device}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        t0 = time.time()
        SECTIONS[w]()
        print(f"--- {w} done in {time.time()-t0:.1f}s ---", flush=True)
    path = os.path.join(HERE, "v2_quad.json")
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:          # a partial write from a crashed run
            old = {}
    old.update(RES)
    with open(path, "w") as f:
        json.dump(old, f, indent=1, sort_keys=True, default=str)
    print("wrote", path)


if __name__ == "__main__":
    main()
