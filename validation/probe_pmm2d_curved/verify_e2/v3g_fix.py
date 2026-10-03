"""Sub-verifier item 3 fixtures (gates E2-2/3/4/6 on the VERIFIER's own
geometries).  Cell P = 1.1 (square), lambda = 0.95, air above, n = 1.5 below.
Nothing is imported from build_e2/."""
import time

import numpy as np
from _ve import PMM2DStackPure, solve  # noqa: F401  (asserts the tree)

from lumenairy.elements.pmm import _curvemortar as CMM

P = 1.1
WL = 0.95
N_SUP, N_SUB = 1.0, 1.5

# ---- recorder of every curved interface's adaptive node count -------------
NREC = []
_orig_init = CMM.StagCrossOpsMapped.__init__


def _rec_init(self, ga, gb, tol=None):
    t0 = time.perf_counter()
    _orig_init(self, ga, gb, tol=tol)
    NREC.append(dict(n=int(self.n), change=float(self.change),
                     Ma=int(ga.M), Mb=int(gb.M), Na=int(ga.bx.N),
                     Nb=int(gb.bx.N), wall=time.perf_counter() - t0,
                     mapped_a=ga.cmap is not None,
                     mapped_b=gb.cmap is not None))


CMM.StagCrossOpsMapped.__init__ = _rec_init


def stack(layers, M, per_layer=True, n_orders=3, sub=N_SUB, sup=N_SUP):
    kw = dict(n_superstrate=sup, n_substrate=sub, n_modes=M,
              n_orders=n_orders)
    if per_layer:
        kw["layer_grids"] = "per-layer"
    st = PMM2DStackPure(P, P, **kw)
    for t, shp, bg in layers:
        if shp is None:
            st.add_layer(t, eps=bg)
        else:
            st.add_layer(t, shapes=shp, background_eps=bg)
    return st


def jones_block(st, k):
    """Power-normalised 2x2 reflection Jones block of order k (own copy of
    the Phase C instrument; incidence from air)."""
    r = st._modal
    kx0, ky0, kzi = r["kx0"], r["ky0"], r["kz_inc"]
    A = np.array([[r["rx"][c][k] for c in (0, 1)],
                  [r["ry"][c][k] for c in (0, 1)]])
    kxo, kyo = r["kx"][k], r["ky"][k]
    kzo = complex(r["kz_ref"][k])
    if abs(kzo.imag) > 1e-12 or kzo.real <= 0:
        return None
    kzo = kzo.real
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, inv=False):
        w, V = np.linalg.eigh(S)
        return (V * w ** (-0.5 if inv else 0.5)) @ V.conj().T
    return msqrt(Wout) @ A @ msqrt(Gin, inv=True)


def reverse_angles(theta, phi, m, n):
    s = np.sin(theta)
    kx = -(s * np.cos(phi) + m * WL / P)
    ky = -(s * np.sin(phi) + n * WL / P)
    return float(np.arcsin(np.hypot(kx, ky))), float(np.arctan2(ky, kx))


def order_index(o, mn):
    return int(np.nonzero((o[:, 0] == mn[0]) & (o[:, 1] == mn[1]))[0][0])
