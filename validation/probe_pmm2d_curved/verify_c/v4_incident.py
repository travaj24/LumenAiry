"""V4 -- THE INCIDENT FIELD UNDER A MAP (F-B4 / D4), re-derived and measured.

Arms (the stack's ``_stag_incident_coeffs_mapped`` swapped per arm):
  l2norm   -- SHIPPED: exact L2 decomposition, renormalised on order 0;
  l2       -- bare L2 decomposition (H0 dropped);
  lstsq    -- the shipped least-squares overlap (Phase B behaviour);
  modepick -- (verifier's) the incident = the TWO discrete superstrate
              eigenmodes with the largest order-0 far field, combined so
              their order-0 far field is exactly the input: an EXACT discrete
              half-space mode, so a vacuum spacer can only phase it.

Kinds:
  film <map> <M> <th_deg> <ph_deg> : a uniform n = 2 film (depth 0.5) under
      map 'stretch' (SeparableStretch, sine stretches 0.10 / -0.07 on a 3 x 3
      grid) or 'circle' (the 3 x 3 circle map), max |R, T - Airy| (my own
      s / p Airy), every order, both inputs; + the n_orders 2 vs 5 change.
  spacer <map> <M> : pillar (circle map: the r = 0.36 disk; stretch map: a
      square pillar at the preimages of 0.4 / 0.8) with and without a 0.3
      VACUUM spacer on top, max |R, T change| per arm.
  recip <M> : reciprocity of an OFF-CENTRE disk (0.5, 0.7, r 0.3) at
      (20 deg, 30 deg), reflection channels (-1, 0) and (0, -1), singular
      values of the power-normalised Jones block vs its reversal.
Output: v4_<kind>_..._<build>.json
"""
import sys
import time
import warnings

import numpy as np
from _vc import BUILD, dump

from lumenairy.elements.pmm import (
    Circle,
    PMM2DStackPure,
    _curvemap as CM,
    stack2d_pure as SP,
    twod_staggered as TS,
)

warnings.simplefilter("ignore")
P, WL, DEPTH, NSUP, NSUB = 1.2, 1.0, 0.5, 1.0, 1.45
ORIG = TS._stag_incident_coeffs_mapped


def modepick(geom, bx, by, cmap, a0x, a0y, H0=None):
    nrm = np.linalg.norm(H0, axis=0)
    sel = np.argsort(nrm)[-2:]
    E = np.zeros((H0.shape[1], 2), complex)
    E[sel, [0, 1]] = 1.0
    return E @ np.linalg.inv(H0 @ E)


def set_arm(a):
    SP._stag_incident_coeffs_mapped = {
        "l2norm": ORIG,
        "l2": lambda *x, **k: ORIG(*x),
        "lstsq": lambda *x, **k: None,
        "modepick": modepick}[a]


def maps(name):
    if name == "circle":
        cm, _ = CM._circle_map_3x3(P, 0.36)
        return cm
    fx, fy = CM.SineStretch(0.10), CM.SineStretch(-0.07)
    return CM.SeparableStretch.from_physical_walls(
        np.array([0, 0.4, 0.8, P]), np.array([0, 0.4, 0.8, P]), fx=fx, fy=fy)


def solve(cmap, eps_cell, M, theta=0.0, phi=0.0, spacer=None, n_orders=3,
          eps_uniform=None):
    st = PMM2DStackPure(P, P, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=n_orders, cmap=cmap)
    if spacer:
        st.add_layer(float(spacer), eps=1.0)
    if eps_uniform is not None:
        st.add_layer(DEPTH, eps=eps_uniform)
    else:
        st.add_layer(DEPTH, eps_cell=eps_cell)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve()
    return st, np.asarray(o), np.asarray(R), np.asarray(T)


def airy(theta, phi, n2=2.0):
    """|r|^2 for input lab E_x (row 0) and E_y (row 1): split the incident
    transverse field into s and p (a film does not couple them)."""
    k = NSUP * np.sin(theta)
    kz = [np.sqrt(complex(n * n - k * k)) for n in (NSUP, n2, NSUB)]
    e = [NSUP ** 2, n2 ** 2, NSUB ** 2]
    k0 = 2 * np.pi / WL

    def slab(r1, r2):
        ph = np.exp(2j * kz[1] * k0 * DEPTH)
        return abs((r1 + r2 * ph) / (1 + r1 * r2 * ph)) ** 2
    rs = slab((kz[0] - kz[1]) / (kz[0] + kz[1]),
              (kz[1] - kz[2]) / (kz[1] + kz[2]))
    rp = slab((e[1] * kz[0] - e[0] * kz[1]) / (e[1] * kz[0] + e[0] * kz[1]),
              (e[2] * kz[1] - e[1] * kz[2]) / (e[2] * kz[1] + e[1] * kz[2]))
    # incident transverse E = (Ex, Ey); s-unit = (-sin phi, cos phi); the
    # p part's transverse projection is cos(theta) of its full amplitude
    out = []
    for ex, ey in ((1.0, 0.0), (0.0, 1.0)):
        a_s = -np.sin(phi) * ex + np.cos(phi) * ey
        a_p = (np.cos(phi) * ex + np.sin(phi) * ey) / np.cos(theta)
        out.append((a_s ** 2 * rs + a_p ** 2 * rp) / (a_s ** 2 + a_p ** 2))
    return np.array(out)


def idx(o, mn):
    return int(np.nonzero((o[:, 0] == mn[0]) & (o[:, 1] == mn[1]))[0][0])


ORD9 = [(m, n) for m in (-1, 0, 1) for n in (-1, 0, 1)]


def vec(o, R, T):
    i = [idx(o, mn) for mn in ORD9]
    return np.concatenate([R[:, i].ravel(), T[:, i].ravel()])


def film(mname, M, th, ph):
    cm = maps(mname)
    th, ph = np.deg2rad(th), np.deg2rad(ph)
    Rx = airy(th, ph)
    res = {"map": mname, "M": M, "theta": th, "phi": ph}
    for a in ("l2norm", "l2", "lstsq", "modepick"):
        set_arm(a)
        t0 = time.time()
        _st, o, R, T = solve(cm, None, M, th, ph, eps_uniform=4.0)
        i0 = idx(o, (0, 0))
        R2, T2 = R.copy(), T.copy()
        R2[:, i0] -= Rx
        T2[:, i0] -= 1.0 - Rx
        err = float(max(np.abs(R2).max(), np.abs(T2).max()))
        _st, o5, R5, T5 = solve(cm, None, M, th, ph, eps_uniform=4.0,
                                n_orders=5)
        _st, o2, R2_, T2_ = solve(cm, None, M, th, ph, eps_uniform=4.0,
                                  n_orders=2)
        win = float(np.max(np.abs(vec(o5, R5, T5) - vec(o2, R2_, T2_))))
        res[a] = {"airy_err": err, "n_orders_2_vs_5": win,
                  "wall_s": time.time() - t0}
        print(mname, M, a, res[a], flush=True)
    set_arm("l2norm")
    dump(f"v4_film_{mname}_M{M}_t{np.rad2deg(th):.0f}_p{np.rad2deg(ph):.0f}"
         f"_{BUILD}.json", res)


def pillar_cell(mname):
    e = np.ones((3, 3), complex)
    e[1, 1] = 4.0
    return e


def spacer(mname, M):
    cm = maps(mname)
    eps = pillar_cell(mname)
    res = {"map": mname, "M": M}
    for a in ("l2norm", "lstsq", "modepick"):
        set_arm(a)
        t0 = time.time()
        _s, o, R, T = solve(cm, eps, M)
        v0 = vec(o, R, T)
        out = {}
        for t in (0.3, 0.77):
            _s, o1, R1, T1 = solve(cm, eps, M, spacer=t)
            out[f"spacer_{t}"] = float(np.max(np.abs(vec(o1, R1, T1) - v0)))
        out["vec"] = v0.tolist()
        out["closure"] = float(np.max(np.abs(R.sum(1) + T.sum(1) - 1)))
        out["wall_s"] = time.time() - t0
        res[a] = out
        print(mname, M, a, {k: v for k, v in out.items() if k != "vec"},
              flush=True)
    set_arm("l2norm")
    dump(f"v4_spacer_{mname}_M{M}_{BUILD}.json", res)


def jones_block(md, k):
    """power-normalised 2 x 2 reflection Jones block of order index k"""
    A = np.array([[md["rx"][c][k] for c in (0, 1)],
                  [md["ry"][c][k] for c in (0, 1)]])
    kx0, ky0, kzi = md["kx0"], md["ky0"], md["kz_inc"]
    kxo, kyo, kzo = md["kx"][k], md["ky"][k], md["kz_ref"][k]
    if abs(np.imag(kzo)) > 1e-12 or np.real(kzo) <= 0:
        return None
    kzo = float(np.real(kzo))
    Gin = np.eye(2) + np.outer([kx0, ky0], [kx0, ky0]) / kzi ** 2
    Wout = (kzo / kzi) * (np.eye(2) + np.outer([kxo, kyo], [kxo, kyo])
                          / kzo ** 2)

    def msqrt(S, inv=False):
        w, V = np.linalg.eigh(S)
        return (V * w ** (-0.5 if inv else 0.5)) @ V.conj().T
    return msqrt(Wout) @ A @ msqrt(Gin, inv=True)


def recip(M, arms=("l2norm", "lstsq"), tag=""):
    th, ph = 20.0, 30.0
    res = {"M": M, "theta": th, "phi": ph}
    sh = [Circle(0.5, 0.7, 0.3, 4.0)]
    for a in arms:
        set_arm(a)
        rows = {}
        for m, n in ((-1, 0), (0, -1)):
            st0 = np.sin(np.deg2rad(th))
            kx = -(st0 * np.cos(np.deg2rad(ph)) + m * WL / P)
            ky = -(st0 * np.sin(np.deg2rad(ph)) + n * WL / P)
            tr = np.arcsin(np.hypot(kx, ky))
            pr = np.arctan2(ky, kx)
            sv = []
            for (t_, p_) in ((np.deg2rad(th), np.deg2rad(ph)), (tr, pr)):
                st = PMM2DStackPure(P, P, n_superstrate=NSUP,
                                    n_substrate=NSUB, n_modes=M, n_orders=3)
                st.add_layer(DEPTH, shapes=sh, background_eps=1.0)
                st.set_source(WL, theta=float(t_), phi=float(p_))
                o, R, T, J = st.solve()
                md = st._modal
                o = np.asarray(o)
                N = jones_block(md, idx(o, (m, n)))
                sv.append(np.linalg.svd(N, compute_uv=False))
            # wrong pairing: forward channel vs reverse specular
            rows[f"({m},{n})"] = float(np.max(np.abs(sv[0] - sv[1])))
        res[a] = rows
        print(M, a, rows, flush=True)
    set_arm("l2norm")
    dump(f"v4_recip{tag}_M{M}_{BUILD}.json", res)


if __name__ == "__main__":
    k = sys.argv[1]
    if k == "film":
        film(sys.argv[2], int(sys.argv[3]), float(sys.argv[4]),
             float(sys.argv[5]))
    elif k == "spacer":
        spacer(sys.argv[2], int(sys.argv[3]))
    elif k == "recipmp":
        recip(int(sys.argv[2]), arms=("modepick",), tag="mp")
    else:
        recip(int(sys.argv[2]))
