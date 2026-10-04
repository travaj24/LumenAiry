"""The verifier's own fixtures (different from the builder's 15 x 15 /
P 1.2 / 3 x 3 cell and its 1-D grating)."""
from _v import jnp, np

import lumenairy.elements.rcwa._core as RC
from lumenairy.elements.rcwa import rcwa_efficiency_2d, rcwa_jones_2d

# ---- RCWA: 24 x 24 pixel cell, square post (indices 7..16, symmetric about
# 11.5 in both axes and under x <-> y): C4v.
S = 24
PX = 0.9
WL = 1.0
NSUP, NSUB = 1.0, 1.52
DEPTH = 0.37
EB, EP = 1.8, 5.0
post = np.zeros((S, S))
post[7:17, 7:17] = 1.0
BASE = EB + (EP - EB) * post
# directions
D_X = np.zeros((S, S))            # widen the post along x: keeps both
D_X[6, 7:17] = D_X[17, 7:17] = 1  # mirrors, breaks the 90-degree rotation
D_CORNER = np.zeros((S, S))       # one pixel off a corner: breaks all
D_CORNER[6, 6] = 1.0
D_KEEP = post.copy()              # the whole post: keeps C4v
D_DIAG = np.zeros((S, S))         # two opposite corners: keeps the
D_DIAG[6, 6] = D_DIAG[17, 17] = 1  # diagonal mirror, breaks the rotation
RAND = np.random.default_rng(1234).uniform(-1.0, 1.0, (S, S))
DIRS = {"x": D_X, "corner": D_CORNER, "keep": D_KEEP, "diag": D_DIAG}


def rcwa_f(direction, pol, n_orders=3, loss=0.0, offset=0.0, theta=0.0,
           phi=0.0, formulation="laurent", base=None, symmetry=False):
    b = BASE if base is None else base
    lossy = 1j * loss * post
    D = DIRS[direction] if isinstance(direction, str) else direction

    def f(t, xp):
        eps = (xp.asarray(b.astype(complex) + lossy) + t * xp.asarray(D)
               + offset * xp.asarray(RAND)).astype(complex)
        kw = dict(theta=theta, phi=phi, polarization=pol,
                  n_orders_x=n_orders, n_orders_y=n_orders,
                  formulation=formulation)
        if xp is np:
            kw["symmetry"] = symmetry
        _o, R, T = rcwa_efficiency_2d(PX, PX, eps, NSUB, NSUP, DEPTH, WL,
                                      **kw)
        return xp.concatenate([R, T]).real
    return f


def omega2_numpy(eps_cell, n_orders=3, theta=0.0, phi=0.0,
                 formulation="laurent"):
    """The layer operator's eigenvalues as the NumPy path builds them
    (captured from the eig the solve actually calls)."""
    got = []
    orig = RC._eig_for

    def spy(xp):
        e = orig(xp)

        def g(A):
            out = e(A)
            got.append(np.asarray(out[0]))
            return out
        return g
    RC._eig_for = spy
    try:
        rcwa_efficiency_2d(PX, PX, eps_cell.astype(complex), NSUB, NSUP,
                           DEPTH, WL, theta=theta, phi=phi,
                           n_orders_x=n_orders, n_orders_y=n_orders,
                           formulation=formulation, symmetry=False)
    finally:
        RC._eig_for = orig
    return got[-1]


def clusters(lam, gap):
    lam = np.asarray(lam)
    s = np.max(np.abs(lam))
    n = lam.size
    close = np.abs(lam[:, None] - lam[None, :]) <= gap * s
    seen = np.zeros(n, bool)
    out = []
    for i in range(n):
        if seen[i]:
            continue
        stack, comp = [i], []
        seen[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for j in np.nonzero(close[k] & ~seen)[0]:
                seen[j] = True
                stack.append(j)
        if len(comp) > 1:
            out.append(sorted(comp))
    return out


# ---- 1-D PMM grating (different from the builder's P 1.2 / n 2 / 1 / duty
# 0.5 / depth 0.45 / n_sup 1.45 / n_sub 1 / degree 12)
G1 = dict(period=0.85, n_ridge=2.3, n_groove=1.35, n_sup=1.0, n_sub=1.6,
          depth=0.31, duty=0.4, wl=1.0, degree=10)


def pmm1d_f(pol, which="angle", loss=0.0):
    from lumenairy.elements.pmm import pmm_efficiency_1d
    g = G1

    def f(t, xp):
        nr = g["n_ridge"] + 1j * loss
        if which == "angle":
            ang, nrv = t, xp.asarray(nr + 0j)
        else:
            ang, nrv = 0.0, xp.asarray(nr + 0j) + t
        _o, R, T = pmm_efficiency_1d(g["period"], nrv, g["n_groove"],
                                     g["n_sub"], g["n_sup"], g["depth"],
                                     g["duty"], g["wl"], angle=ang,
                                     polarization=pol, degree=g["degree"],
                                     stabilize=False)
        return xp.concatenate([xp.asarray(R), xp.asarray(T)]).real
    return f


__all__ = ["rcwa_f", "omega2_numpy", "clusters", "pmm1d_f", "BASE", "DIRS",
           "post", "RAND", "jnp", "rcwa_jones_2d"]
