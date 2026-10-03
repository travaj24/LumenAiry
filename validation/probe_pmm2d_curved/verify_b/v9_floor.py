"""V9 -- F-B4: the ~1e-9 floor of the mapped solve.  Is it the minimum-norm
DRAW of ``cinc = lstsq(Hsup, rhs)`` (under-determined: 2 (2 n_orders + 1)^2
equations, 2 q^2 half-space modes) or the CONDITIONING of Hsup under a map?

Arms (3 x 3 circle r = 0.36 unless stated, M = 6, normal incidence):
  svd        singular values of Hsup (n_orders 3 and 10), rank, cond on the
             row space; for c3, the identity map on the same walls, and a
             Phase-A SeparableStretch (sine 0.05 p) on the same walls
  null       R / T sensitivity to the NULL-SPACE component of cinc: add
             eps * (a random unit vector of null(Hsup)) to the min-norm cinc,
             eps = 1e-8 .. 1e-4: dR/T / eps.  O(1) => R / T are not a function
             of the window equations alone (a draw); ~0 => they are.
  rowpert    perturb Hsup's ROW-SPACE solve only (x = pinv(A + dA) b with dA
             a random 1e-15 relative perturbation) vs the null component:
             which one moves R / T?
  window     R / T vs n_orders = 2, 3, 5, 7, 8, 10, 12, 14 (the window
             dependence; overdetermined from 2 n + 1 >= 3 (M - 1))
Output: v9_floor_<build>.json
"""
import _vcommon as C
import numpy as np

CM, SP, TS = C.CM, C.SP, C.TS

cap = {}
orig = SP._guarded_lstsq


def hook(mode, eps=0.0, seed=0):
    rng = np.random.default_rng(seed)

    def f(A, b, site, hint=None):
        A = np.asarray(A)
        x = orig(A, b, site, hint)
        cap["A"], cap["b"], cap["x"] = A, b, x
        if mode == "null" and eps > 0:
            _u, s, vh = np.linalg.svd(A)
            rank = int(np.sum(s > s[0] * 1e-12))
            N = vh[rank:].conj().T
            z = N @ (rng.standard_normal(N.shape[1])
                     + 1j * rng.standard_normal(N.shape[1]))
            z /= np.linalg.norm(z)
            return x + eps * np.linalg.norm(x) * z
        if mode == "rowpert":
            dA = A * 1e-15 * rng.standard_normal(A.shape)
            return np.linalg.lstsq(A + dA, b, rcond=None)[0]
        return x
    return f


def solve(cm, eps, M, n_orders=3, mode=None, e=0.0, seed=0):
    SP._guarded_lstsq = hook(mode, e, seed) if mode else hook("plain")
    try:
        o, R, T, _s, _w = C.solve_map(cm, eps, M, n_orders=n_orders)
    finally:
        SP._guarded_lstsq = orig
    return C.vec(o, R, T)


def maps():
    c3, e3 = C.vcircle3(0.36)
    idm = CM.TransfiniteMap(c3.u_walls, c3.v_walls)
    sst = CM.SeparableStretch(c3.u_walls, c3.v_walls,
                              fx=CM.SineStretch(0.05 * C.P),
                              fy=CM.SineStretch(0.05 * C.P))
    return {"c3": (c3, e3), "identity": (idm, e3), "stretch0.05": (sst, e3)}


res = {}
M = 6
for name, (cm, eps) in maps().items():
    r = {}
    base = solve(cm, eps, M)
    for n_orders in (3, 10):
        solve(cm, eps, M, n_orders)
        A = cap["A"]
        s = np.linalg.svd(A, compute_uv=False)
        rank = int(np.sum(s > s[0] * 1e-12))
        r[f"svd_n{n_orders}"] = {"shape": A.shape, "rank": rank,
                                 "smax": float(s[0]),
                                 "smin_rowspace": float(s[rank - 1]),
                                 "cond": float(s[0] / s[rank - 1]),
                                 "resid": float(np.linalg.norm(
                                     A @ cap["x"] - cap["b"]))}
    r["null"] = {}
    for e in (1e-8, 1e-6, 1e-4):
        v = solve(cm, eps, M, 3, "null", e, seed=1)
        r["null"][e] = float(np.max(np.abs(v - base)) / e)
    v1 = solve(cm, eps, M, 3, "rowpert", seed=2)
    v2 = solve(cm, eps, M, 3, "rowpert", seed=3)
    r["rowpert_1e-15_dRT"] = float(max(np.max(np.abs(v1 - base)),
                                       np.max(np.abs(v2 - base))))
    r["window"] = {}
    vs = {}
    for n in (2, 3, 5, 7, 8, 10, 12, 14):
        vs[n] = solve(cm, eps, M, n)
    for n in vs:
        r["window"][n] = float(np.max(np.abs(vs[n] - vs[14])))
    res[name] = r
    print(name, {k: v for k, v in r.items()}, flush=True)
C.dump(f"v9_floor_{C.build_tag()}.json", res)
