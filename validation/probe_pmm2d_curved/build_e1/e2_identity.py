"""E1-2 -- the IDENTITY map through the mapped out-of-plane generator is the
shipped out-of-plane generator to round-off: operators (A, B), the region
modes (eigenvalues, the forward / backward split, the modal fields) and the
full R / T / reflection and transmission Jones.

Cells (all through Granet2DTransverseE / PMM2DStackPure, with cmap=None vs an
explicit identity TransfiniteMap on the SAME walls):
  * L     -- the shipped G0 re-entrant-corner L cell (3 x 3) of a tilted LC;
  * nonrec-- a non-reciprocal OOP pillar on NON-UNIFORM walls;
  * slant -- a scalar eps-4 pillar slanted (0.25, -0.1) (public), 3 x 3;
  * slantoop -- the L cell slanted (0.2, 0.15);
at normal incidence and at the Bloch phases of an oblique / conical mount
(operators and modes), M = 5 and 7.

The full solves are compared at NORMAL incidence, where both incident
treatments are exact; at oblique incidence the mapped stack's exact modal
incident decomposition and the unmapped least-squares overlap differ at the
DISCRETISATION level even for an identity map (Phase C finding F-C3), so
those rows are recorded with the scalar in-plane difference next to them.
The non-uniform-wall cell is compared at the operator / mode level only (the
shared-grid stack takes integer walls).

usage: python e2_identity.py <M>      writes e2_identity_M<M>.json
"""
import sys
import time
import warnings

import _e1common as E
import numpy as np

CM, TS, PMM2DStackPure = E.CM, E.TS, E.PMM2DStackPure
P, WL, DEP, NSUB = 1.2, 1.0, 0.4, 1.5
EYE = np.eye(3, dtype=complex)
NW = np.array([0.0, 0.25, 0.85, P])
NWY = np.array([0.0, 0.3, 0.9, P])


def lcell(t):
    c = np.empty((3, 3, 3, 3), complex)
    c[:] = EYE
    for i, j in ((0, 0), (1, 0), (0, 1)):
        c[i, j] = t
    return c


def pill(t, host=EYE):
    c = np.empty((3, 3, 3, 3), complex)
    c[:] = host
    c[1, 1] = t
    return c


pil_sc = np.ones((3, 3), complex)
pil_sc[1, 1] = 4.0
CELLS = {
    "L": dict(eps=lcell(E.OOP), wx=3, wy=3, slant=None),
    "nonrec_nonuni": dict(eps=pill(E.NONREC, 2.25 * EYE), wx=NW, wy=NWY,
                          slant=None),
    "slant": dict(eps=pil_sc, wx=3, wy=3, slant=(0.25, -0.1)),
    "slantoop": dict(eps=lcell(E.OOP), wx=3, wy=3, slant=(0.2, 0.15)),
}
MOUNTS = {"normal": (0.0, 0.0), "oblique": (np.deg2rad(25.0), 0.0),
          "conical": (np.deg2rad(25.0), np.deg2rad(40.0))}


def idmap(wx, wy):
    u = np.linspace(0, P, wx + 1) if np.isscalar(wx) else np.asarray(wx)
    v = np.linspace(0, P, wy + 1) if np.isscalar(wy) else np.asarray(wy)
    return CM.TransfiniteMap(u, v)


def match_modes(a, b):
    """Max distance between two eigenvalue SETS (nearest neighbour), relative
    to the spectrum's scale."""
    a, b = np.asarray(a), np.asarray(b)
    d = np.abs(a[:, None] - b[None, :]).min(axis=1)
    return float(d.max() / np.abs(a).max())


def solver_arms(name, c, M, th, ph):
    a0x = np.sin(th) * np.cos(ph) * 2 * np.pi / WL
    a0y = np.sin(th) * np.sin(ph) * 2 * np.pi / WL
    kw = dict(alpha0x=a0x, alpha0y=a0y, k0=2 * np.pi / WL, slant=c["slant"])
    s0 = TS.Granet2DTransverseE(P, P, c["wx"], c["wy"], M, c["eps"], **kw)
    cm = idmap(c["wx"], c["wy"])
    s1 = TS.Granet2DTransverseE(P, P, cm.u_walls, cm.v_walls, M, c["eps"],
                                cmap=cm, **kw)
    sA = float(np.max(np.abs(s0.Agen)))
    sB = float(np.max(np.abs(s0.Bgen)))
    m0 = TS._region_modes_oop(s0)
    m1 = TS._region_modes_oop(s1)
    lam0 = np.concatenate([m0[2], m0[5]])
    lam1 = np.concatenate([m1[2], m1[5]])
    # the forward sets must be the same modes: match each forward lam of one
    # arm to the other's forward set (a split disagreement would show as a
    # forward mode matching nothing)
    fwd = match_modes(m0[2], m1[2])
    # modal fields: project arm-1 forward W onto arm-0's forward W for the
    # matched eigenvalue (non-degenerate ones), compare the H partners too
    i1 = np.argmin(np.abs(m0[2][:, None] - m1[2][None, :]), axis=1)
    W0, V0 = m0[0], m0[1]
    W1, V1 = m1[0][:, i1], m1[1][:, i1]
    gap = np.sort(np.abs(m0[2][:, None] - m0[2][None, :]), axis=1)[:, 1]
    ok = gap > 1e-6 * np.abs(m0[2]).max()
    ph1 = np.sum(np.conj(W0) * W1, axis=0) / np.sum(np.conj(W0) * W0, axis=0)
    dW = np.abs(W1 / ph1[None, :] - W0)[:, ok].max() / np.abs(W0).max()
    dV = np.abs(V1 / ph1[None, :] - V0)[:, ok].max() / np.abs(V0).max()
    return {"A_rel": float(np.max(np.abs(s1.Agen - s0.Agen)) / sA),
            "B_rel": float(np.max(np.abs(s1.Bgen - s0.Bgen)) / sB),
            "lam_all_rel": match_modes(lam0, lam1),
            "lam_forward_rel": fwd,
            "W_rel": float(dW), "V_rel": float(dV),
            "n_nondegenerate": int(ok.sum())}


def full(c, M, th, ph, cmap):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=NSUB, n_modes=M,
                        n_orders=2, cmap=cmap)
    st.add_layer(DEP, eps_cell=c["eps"], slant=c["slant"])
    st.set_source(WL, theta=th, phi=ph)
    o, R, T, J = st.solve(jones=True)
    return np.asarray(R), np.asarray(T), np.asarray(J), E.jones_t(st)


def main(M):
    warnings.simplefilter("ignore")
    out = {"M": M}
    for name, c in CELLS.items():
        for mname, (th, ph) in MOUNTS.items():
            t0 = time.perf_counter()
            r = solver_arms(name, c, M, th, ph)
            if not np.isscalar(c["wx"]):
                # the shared-grid stack takes integer walls only; the
                # non-uniform cell is gated at the operator / mode level
                r["t"] = time.perf_counter() - t0
                out[f"{name}_{mname}"] = r
                print(name, mname, r, flush=True)
                continue
            a = full(c, M, th, ph, None)
            cm = idmap(c["wx"], c["wy"])
            b = full(c, M, th, ph, cm)
            r.update({"dR": float(np.max(np.abs(a[0] - b[0]))),
                      "dT": float(np.max(np.abs(a[1] - b[1]))),
                      "dJr": float(np.max(np.abs(a[2] - b[2]))),
                      "dJt": float(np.max(np.abs(a[3] - b[3]))),
                      "t": time.perf_counter() - t0})
            out[f"{name}_{mname}"] = r
            print(name, mname, {k: (f"{v:.2e}" if isinstance(v, float)
                                    else v) for k, v in r.items()},
                  flush=True)
    # the scalar in-plane reference for the oblique incident-treatment gap
    for mname, (th, ph) in MOUNTS.items():
        a = full(dict(eps=pil_sc, wx=3, wy=3, slant=None), M, th, ph, None)
        b = full(dict(eps=pil_sc, wx=3, wy=3, slant=None), M, th, ph,
                 idmap(3, 3))
        out[f"scalar_inplane_{mname}"] = {
            "dR": float(np.max(np.abs(a[0] - b[0]))),
            "dT": float(np.max(np.abs(a[1] - b[1]))),
            "dJr": float(np.max(np.abs(a[2] - b[2])))}
        print("scalar", mname, out[f"scalar_inplane_{mname}"], flush=True)
    E.dump(f"e2_identity_M{M}.json", out)


if __name__ == "__main__":
    main(int(sys.argv[1]))
