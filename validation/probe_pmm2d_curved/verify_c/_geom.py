"""Independent GEOMETRY oracle for a merged shape map (verifier, Phase C).

Given the merged wall grid / map and each layer's painted ``(u, v)`` cells,
and each layer's shapes (painted in order on a background), measure against
the shapes' ANALYTIC outlines -- nothing here reads the merge's own claims:

* ``paint``  : at 6 x 6 interior Gauss points of EVERY cell, the analytic
  material at the point's PHYSICAL image (``contains`` of the shapes, painted
  in order) must be the cell's eps.  Points within ``1e-9`` of an outline are
  skipped.  Reports the number of wrong points.
* ``on``     : every MATERIAL boundary of the painted grid (a cell edge with
  different eps on its two sides, incl. the periodic seams) is sampled at 200
  points through the map; at each the ANALYTIC material 1e-7 p either side
  along the image's normal must be the two cells' two (different) eps --
  counts the mismatches (a boundary 1e-7 p off the true outline fails).
* ``cover``  : 1000 points on every shape's outline (``boundary_points``);
  those that are a VISIBLE material boundary (the analytic material differs
  at +-1e-6 along the outline normal) must lie on a material-boundary image;
  max distance (segment projection onto 2000-point polylines; chord sag
  ~1e-8 for the radii used).
* ``detJ``   : min of det J at 12 x 12 interior Gauss nodes of each cell,
  relative to the cell's (u, v) area ratio, and the max / min spread of the
  largest singular value of J (steepness), over the non-singular cells.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def eps_key(e):
    a = np.asarray(e, dtype=complex)
    return tuple(np.round(a.ravel(), 12).tolist())


def painter(shapes, bg):
    def eps_at(x, y):
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        out = np.full(x.shape, complex(bg), dtype=complex)
        for sh in shapes:
            out[np.asarray(sh.contains(x, y), bool)] = complex(sh.eps)
        return out
    return eps_at


def min_outline_dist(shapes, x, y):
    d = np.full(np.shape(x), np.inf)
    for sh in shapes:
        d = np.minimum(d, np.abs(np.asarray(sh.signed_distance(x, y))))
    return d


def _seg_dist(pts, poly):
    """distance of each point to the polyline set (list of (n, 2))"""
    best = np.full(len(pts), np.inf)
    for P in poly:
        A, B = P[:-1], P[1:]
        AB = B - A
        L2 = np.maximum(np.sum(AB * AB, 1), 1e-300)
        for k0 in range(0, len(pts), 200):
            p = pts[k0:k0 + 200]
            t = np.clip(((p[:, None, :] - A[None]) * AB[None]).sum(-1)
                        / L2[None], 0.0, 1.0)
            C = A[None] + t[..., None] * AB[None]
            d = np.sqrt(((p[:, None, :] - C) ** 2).sum(-1)).min(1)
            best[k0:k0 + 200] = np.minimum(best[k0:k0 + 200], d)
    return best


def check(cmap, cells, layer_shapes, layer_bg, n_edge=200, n_poly=2000):
    U, V = cmap.u_bounds, cmap.v_bounds
    nx, ny = cmap.shape
    xg, _ = leggauss(6)
    s6 = 0.5 + 0.5 * xg
    sing = {(a, b) for a, b, _c, _d in cmap.singular_vertices}
    rep = []
    for li, (cell, shapes, bg) in enumerate(zip(cells, layer_shapes,
                                                layer_bg)):
        eps_at = painter(shapes, bg)
        ckey = [[eps_key(cell[i, j]) for j in range(ny)] for i in range(nx)]
        nbad, nchk = 0, 0
        for i in range(nx):
            for j in range(ny):
                Uq = U[i] + s6 * (U[i + 1] - U[i])
                Vq = V[j] + s6 * (V[j + 1] - V[j])
                X, Y = cmap.geom(i, j, Uq, Vq)[:2]
                far = min_outline_dist(shapes, X, Y) > 1e-9 if shapes else \
                    np.ones(X.shape, bool)
                e = eps_at(X, Y)
                bad = np.array([[eps_key(v) != ckey[i][j] for v in row]
                                for row in e]) & far
                nbad += int(bad.sum())
                nchk += int(far.sum())
        # material-boundary images
        polys = []
        s = np.linspace(0.0, 1.0, n_poly)
        se = np.linspace(0.0, 1.0, n_edge)
        on_bad, on_n = 0, 0
        for i in range(nx):
            for j in range(ny):
                ir = (i + 1) % nx
                jt = (j + 1) % ny
                for side, nb in (("r", (ir, j)), ("t", (i, jt))):
                    if ckey[i][j] == ckey[nb[0]][nb[1]]:
                        continue
                    if side == "r":
                        Uq = np.full(n_poly, U[i + 1])
                        Vq = V[j] + s * (V[j + 1] - V[j])
                    else:
                        Uq = U[i] + s * (U[i + 1] - U[i])
                        Vq = np.full(n_poly, V[j + 1])
                    X, Y = cmap.geom_points(i, j, Uq, Vq)[:2]
                    P = np.stack([X, Y], 1)
                    for sx_ in (-1, 0, 1):          # periodic copies
                        for sy_ in (-1, 0, 1):
                            polys.append(P + np.array(
                                [sx_ * cmap.period_x, sy_ * cmap.period_y]))
                    # 'on': the analytic material must CHANGE across the
                    # image, and match the two cells' eps
                    idx = np.round(se * (n_poly - 1)).astype(int)
                    idx = np.clip(idx, 1, n_poly - 2)
                    tg = P[idx + 1] - P[idx - 1]
                    tg /= np.maximum(np.hypot(tg[:, 0], tg[:, 1]),
                                     1e-300)[:, None]
                    nrm = np.stack([tg[:, 1], -tg[:, 0]], 1)
                    dl = 1e-7 * cmap.period_x
                    pa = P[idx] + dl * nrm
                    pb = P[idx] - dl * nrm
                    ea = eps_at(np.mod(pa[:, 0], cmap.period_x),
                                np.mod(pa[:, 1], cmap.period_y))
                    eb = eps_at(np.mod(pb[:, 0], cmap.period_x),
                                np.mod(pb[:, 1], cmap.period_y))
                    want = {ckey[i][j], ckey[nb[0]][nb[1]]}
                    for a, b in zip(ea, eb):
                        if not ({eps_key(a), eps_key(b)} == want
                                and eps_key(a) != eps_key(b)):
                            on_bad += 1
                    on_n += len(idx)
        # coverage of visible outlines
        cov_max, n_vis = 0.0, 0
        for sh in shapes:
            b = sh.boundary_points(1000)
            h = 1e-7
            gx = (sh.signed_distance(b[:, 0] + h, b[:, 1])
                  - sh.signed_distance(b[:, 0] - h, b[:, 1])) / (2 * h)
            gy = (sh.signed_distance(b[:, 0], b[:, 1] + h)
                  - sh.signed_distance(b[:, 0], b[:, 1] - h)) / (2 * h)
            g = np.hypot(gx, gy)
            g[g == 0] = 1.0
            nxv, nyv = gx / g, gy / g
            P = cmap.period_x
            ein = eps_at(np.mod(b[:, 0] - 1e-6 * nxv, P),
                         np.mod(b[:, 1] - 1e-6 * nyv, cmap.period_y))
            eout = eps_at(np.mod(b[:, 0] + 1e-6 * nxv, P),
                          np.mod(b[:, 1] + 1e-6 * nyv, cmap.period_y))
            vis = np.array([eps_key(a) != eps_key(c)
                            for a, c in zip(ein, eout)])
            n_vis += int(vis.sum())
            if vis.any():
                if not polys:
                    cov_max = np.inf
                else:
                    cov_max = max(cov_max, float(np.max(
                        _seg_dist(b[vis], polys))))
        rep.append(dict(layer=li + 1, paint_wrong=nbad, paint_checked=nchk,
                        boundary_side_mismatch=on_bad, boundary_points_checked=on_n,
                        outline_covered_max=cov_max, outline_visible=n_vis,
                        n_boundary_edges=len(polys)))
    # det J
    xg12, _ = leggauss(12)
    s12 = 0.5 + 0.5 * xg12
    rel_min, spread = np.inf, 1.0
    for i in range(nx):
        for j in range(ny):
            Uq = U[i] + s12 * (U[i + 1] - U[i])
            Vq = V[j] + s12 * (V[j + 1] - V[j])
            _X, _Y, xu, xv, yu, yv = cmap.geom(i, j, Uq, Vq)
            det = xu * yv - xv * yu
            rel_min = min(rel_min, float(det.min()))
            if not any((a, b) in sing for a, b in
                       ((i, j),)) and not any(
                    sv[0] == i and sv[1] == j for sv in
                    cmap.singular_vertices):
                Jm = np.stack([np.stack([xu, xv], -1),
                               np.stack([yu, yv], -1)], -2)
                sv = np.linalg.svd(Jm, compute_uv=False)[..., 0]
                spread = max(spread, float(sv.max() / sv.min()))
    return dict(layers=rep, detJ_min=rel_min, sigma_spread_max=spread,
                grid=[int(nx), int(ny)])


def verdict(g, tol=1e-7):
    ok = all(L["paint_wrong"] == 0 and L["boundary_side_mismatch"] == 0
             and L["outline_covered_max"] < tol for L in g["layers"])
    return "EXACT" if ok else "WRONG MAP"
