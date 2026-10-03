"""B1 (second half) -- the TRANSFINITE class with no curve and identity
vertices is the identity map, and reproduces Phase A's identity numbers.

Fixture: Phase A's A2 fixture -- the 3 x 3 pillar on NON-UNIFORM walls
x in {0, 0.25, 0.85, 1.2}, y in {0, 0.30, 0.90, 1.2}, eps 4, M = 5 and 7.
Arms: no map (the shipped kron assembly), ``IdentityMap`` (Phase A), the
identity ``TransfiniteMap`` (Phase B) -- operators, the chosen node count,
the corner-rule cells (must be none) and the full solve R / T / Jones.
Also: a bilinear (straight-edged, non-identity) transfinite map against the
same map written as a closed-form callable (the blend formula on four
straight edges IS the bilinear map), as a check of the Gordon-Hall
evaluation and its analytic Jacobian.

  python validation/probe_pmm2d_curved/build_b/b2_identity.py
Output: b2_identity.json
"""
import _common as C
import numpy as np

XW = np.array([0.0, 0.25, 0.85, C.P])
YW = np.array([0.0, 0.30, 0.90, C.P])


def rel(a, b):
    s = float(np.max(np.abs(b)))
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / s


def main():
    k0 = 2 * np.pi / C.WL
    eps = np.ones((3, 3), complex)
    eps[1, 1] = C.EPS_P
    idm = C.CM.IdentityMap(XW, YW)
    tfm = C.CM.TransfiniteMap(XW, YW)
    res = {"env": C.env_record(), "fixture": "A2 pillar, non-uniform walls",
           "tf_singular_vertices": tfm.singular_vertices, "rows": []}
    for M in (5, 7):
        s0 = C.TS.Granet2DTransverseE(C.P, C.P, XW, YW, M, eps, k0=k0)
        s1 = C.TS.Granet2DTransverseE(C.P, C.P, XW, YW, M, eps, k0=k0,
                                      cmap=idm)
        s2 = C.TS.Granet2DTransverseE(C.P, C.P, XW, YW, M, eps, k0=k0,
                                      cmap=tfm)
        row = {"M": M, "nq_identity": s1._qrule.n, "nq_transfinite":
               s2._qrule.n, "corner_cells_transfinite":
               sorted(map(list, s2._qrule.points))}
        for nm in ("Rmat", "Lmat", "Stt", "Schur"):
            row[f"{nm}_tf_vs_kron"] = rel(getattr(s2, nm), getattr(s0, nm))
            row[f"{nm}_tf_vs_idmap"] = rel(getattr(s2, nm), getattr(s1, nm))
        o0, R0, T0 = C.solve_walls(XW[1:-1], YW[1:-1], eps, M)
        o1, R1, T1, J1 = C.solve(idm, eps, M)
        o2, R2, T2, J2 = C.solve(tfm, eps, M)
        v0, v1, v2 = C.vec(o0, R0, T0), C.vec(o1, R1, T1), C.vec(o2, R2, T2)
        row["RT_tf_vs_kron"] = float(np.max(np.abs(v2 - v0)))
        row["RT_tf_vs_idmap"] = float(np.max(np.abs(v2 - v1)))
        row["RT_idmap_vs_kron"] = float(np.max(np.abs(v1 - v0)))
        row["J_tf_vs_idmap"] = float(np.max(np.abs(np.asarray(J2)
                                                   - np.asarray(J1))))
        res["rows"].append(row)
        print(row, flush=True)
        C.dump("b2_identity.json", res)

    # bilinear check: move the four interior vertices, straight edges
    V = np.empty((4, 4, 2))
    V[..., 0] = XW[:, None]
    V[..., 1] = YW[None, :]
    V[1, 1] += (0.03, -0.02)
    V[2, 1] += (-0.01, 0.04)
    V[1, 2] += (0.02, 0.01)
    V[2, 2] += (0.05, -0.03)
    bil = C.CM.TransfiniteMap(XW, YW, V)
    err = 0.0
    for sx in range(3):
        for sy in range(3):
            U = np.linspace(XW[sx], XW[sx + 1], 9)[1:-1]
            W = np.linspace(YW[sy], YW[sy + 1], 7)[1:-1]
            s = (U - XW[sx]) / (XW[sx + 1] - XW[sx])
            t = (W - YW[sy]) / (YW[sy + 1] - YW[sy])
            S, T = np.meshgrid(s, t, indexing="ij")
            P00, P10, P01, P11 = V[sx, sy], V[sx + 1, sy], V[sx, sy + 1], \
                V[sx + 1, sy + 1]
            Xb = ((1 - S) * (1 - T))[..., None] * P00 + (S * (1 - T))[
                ..., None] * P10 + ((1 - S) * T)[..., None] * P01 + (
                S * T)[..., None] * P11
            Xs = ((1 - T)[..., None] * (P10 - P00) + T[..., None]
                  * (P11 - P01)) / (XW[sx + 1] - XW[sx])
            Xt = ((1 - S)[..., None] * (P01 - P00) + S[..., None]
                  * (P11 - P10)) / (YW[sy + 1] - YW[sy])
            g = bil.geom(sx, sy, U, W)
            for a, b in zip(g, (Xb[..., 0], Xb[..., 1], Xs[..., 0],
                                Xt[..., 0], Xs[..., 1], Xt[..., 1])):
                err = max(err, float(np.max(np.abs(a - b))))
    res["bilinear_closed_form_max_abs"] = err
    print("bilinear", err)
    C.dump("b2_identity.json", res)


if __name__ == "__main__":
    main()
