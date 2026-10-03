"""V6 -- the parity accelerator under a map or with mu (F-E1-5): its cost,
and whether its refusal is a CORRECTNESS decision or a COST / scope one.

usage: python v6_parity.py <M>

(i)   the shipped OOP path on a C2-symmetric UNMAPPED cell at normal
      incidence: region eig with the reduction (symmetry=True) and dense,
      timed (best of 3), outputs compared.
(ii)  the reduction LIFTED (the map / mu clause of _stag_parity_gauge
      bypassed by presenting the solver as unmapped and nonmagnetic to the
      gauge function only) on:
        a  the CENTRED 3 x 3 circle map (C2-symmetric), OOP disk;
        b  the OFF-CENTRE circle map (not C2-symmetric about the cell);
        c  an unmapped C2-symmetric cell with a LOSSLESS gyrotropic mu;
        d  the same with a LOSSY gyrotropic mu (non-Hermitian B);
        e  the centred circle map WITH the lossy mu in the disk;
      reading: whether _stag_block_eig accepts (structural residuals), the
      eigenvalue set against the dense branch, the full stack R / T / Jones
      against the dense stack, and the timing of the region eig.
"""
import sys
import time
import warnings

import _ve1common as V
import numpy as np

TS, CM = V.TS, V.CM
M = int(sys.argv[1])
P = 1.0
out = {}


def lifted(solver):
    cm, mg = solver.cmap, solver.magnetic
    solver.cmap, solver.magnetic = None, False
    try:
        return ORIG_GAUGE(solver)
    finally:
        solver.cmap, solver.magnetic = cm, mg


ORIG_GAUGE = TS._stag_parity_gauge


def residuals(s):
    g = lifted(s)
    if g is None:
        return None
    perm, r = g
    A, B = s.Agen, s.Bgen
    RAR = (r[:, None] * r[None, :]) * A[np.ix_(perm, perm)]
    RBR = (r[:, None] * r[None, :]) * B[np.ix_(perm, perm)]
    return (float(np.abs(RAR + A).max() / np.abs(A).max()),
            float(np.abs(RBR - B).max() / np.abs(B).max()))


def timed(fn, n=3):
    best, res = 1e30, None
    for _ in range(n):
        t0 = time.perf_counter()
        res = fn()
        best = min(best, time.perf_counter() - t0)
    return best, res


def eigdist(a, b):
    la, lb = np.concatenate([a[2], a[5]]), np.concatenate([b[2], b[5]])
    d = np.abs(la[:, None] - lb[None, :]).min(axis=1).max()
    return float(d / np.abs(lb).max())


def stack_cmp(cm, eps, mu, lift):
    res = []
    for sym in (False, True):
        st = V.PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.5,
                              n_modes=M, n_orders=2, cmap=cm, symmetry=sym)
        st.add_layer(0.4, eps_cell=eps, mu_cell=mu)
        st.set_source(1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if lift and sym:
                with V.patched(TS, "_stag_parity_gauge", lifted):
                    o, R, T, J = st.solve(jones=True)
            else:
                o, R, T, J = st.solve(jones=True)
        res.append((np.asarray(R), np.asarray(T), np.asarray(J),
                    V.jt_of(st)))
    return float(max(np.abs(x - y).max() for x, y in zip(*res)))


# (i) the shipped unmapped accelerator
e = np.broadcast_to(V.EYE, (3, 3, 3, 3)).copy()
e[1, 1] = V.DIRGEN
s = TS.Granet2DTransverseE(P, P, 3, 3, M, e)
fac = TS._stag_block_eig(s.Agen, s.Bgen, s.q * s.q, ORIG_GAUGE(s))
td, md = timed(lambda: TS._region_modes_oop(s, symmetry=False))
tr, mr = timed(lambda: TS._region_modes_oop(s, symmetry=True))
out["i_unmapped"] = dict(engaged=fac is not None, t_dense=td, t_reduced=tr,
                         speedup=td / tr, eig=eigdist(mr, md),
                         stack=stack_cmp(None, e, None, False))
print("i", out["i_unmapped"], flush=True)

# (ii) lifted
cmC = CM._circle_map_3x3(P, 0.3)[0]
cmO = CM._circle_map_3x3(P, 0.27, center=(0.41, 0.56))[0]
mug = np.broadcast_to(V.EYE, (3, 3, 3, 3)).copy()
mug[1, 1] = V.MU_GYRO
mul = np.broadcast_to(V.EYE, (3, 3, 3, 3)).copy()
mul[1, 1] = V.MU_GYRO_LOSSY
cases = {"a_centred_circle": (cmC, e, None),
         "b_offcentre_circle": (cmO, e, None),
         "c_unmapped_gyro_mu": (None, e, mug),
         "d_unmapped_lossy_mu": (None, e, mul),
         "e_centred_circle_lossy_mu": (cmC, e, mul)}
for name, (cm, eps, mu) in cases.items():
    kw = {} if cm is None else dict(cmap=cm)
    wx = 3 if cm is None else cm.u_walls
    wy = 3 if cm is None else cm.v_walls
    s = TS.Granet2DTransverseE(P, P, wx, wy, M, eps, mu_cell=mu, **kw)
    qq = s.q * s.q
    shipped_gauge = ORIG_GAUGE(s)
    res_ = residuals(s)
    g = lifted(s)
    fac = None if g is None else TS._stag_block_eig(s.Agen, s.Bgen, qq, g)
    td, md = timed(lambda: TS._region_modes_oop(s, symmetry=False), 2)
    rec = dict(shipped_refuses=shipped_gauge is None,
               bgen_hermitian=bool(s._bgen_hermitian),
               residual_A_B=res_, accepted=fac is not None, t_dense=td)
    if fac is not None:
        with V.patched(TS, "_stag_parity_gauge", lifted):
            # the shipped region eig, with the lifted gauge (it still takes
            # QZ for a non-Hermitian B unless the reduction accepts first)
            tr, mr = timed(lambda: TS._region_modes_oop(s, symmetry=True), 2)
        rec.update(t_reduced=tr, speedup=td / tr, eig=eigdist(mr, md))
        try:
            rec["stack"] = stack_cmp(cm, eps, mu, True)
        except Exception as exc:
            rec["stack"] = f"{type(exc).__name__}: {exc}"[:200]
    out[name] = rec
    print(name, rec, flush=True)
V.dump(f"v6_parity_M{M}.json", out)
