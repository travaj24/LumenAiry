"""R4 (round-2 item 2): a SYMMETRY-BREAKING gradient at a symmetric cell in
the library's OTHER JAX twins that share ``rcwa._jax_eig_stable`` -- AD vs a
premise-checked central-difference ladder of the NumPy solve.

    python r4_other_twins.py CASE

CASE
  rcwa2d   ``rcwa_efficiency_2d`` (JAX eps_cell), a 15 x 15 pixel cell: centre
           block eps 4 (5 x 5 pixels), its four side blocks eps 1.5, corners 1
           (four-fold symmetric); parameter t added to the two x-side blocks
           only (keeps both mirrors, breaks the 90-degree rotation: the 2-D
           analogue of a square pillar's width alone).  n_orders 3 x 3, TE/TM.
  pmm2d    ``pmm_efficiency_2d_cell`` (hybrid 2-D PMM, JAX eps_cell + the
           concrete region_layout), the same cell as 3 x 3 regions; degree 6,
           n_orders 3, TE/TM.
  pmm1d    ``pmm_efficiency_1d`` (JAX), a symmetric binary grating (duty 0.5),
           d/d(angle) at EXACTLY normal incidence of R and T in the +-1
           orders (the angle breaks the x-mirror; the 1-D twin traces no wall
           position, so the angle is its symmetry-breaking parameter).
  pmm2d_theta  ``pmm_efficiency_2d`` (hybrid pillar entry), a centred square
           pillar, d/d(theta) at theta = 0 of the (+-1, 0) orders.
  rcwa2d_theta ``rcwa_efficiency_2d``, the 15 x 15 cell, d/d(theta) at 0.
Quantities: the listed orders' R and T for each polarization; the error is
max |AD - FD| / max |FD| over them.  Controls: a symmetry-KEEPING direction
(t added to all four side blocks) for the cell cases.
"""
import sys

from _r2 import WL, P, dump, jax, jnp, ladder, np

from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_efficiency_2d
from lumenairy.elements.pmm.twod import pmm_efficiency_2d_cell
from lumenairy.elements.rcwa import rcwa_efficiency_2d

CASE = sys.argv[1]
out = {"case": CASE}
base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
xs3 = np.zeros((3, 3))
xs3[0, 1] = xs3[2, 1] = 1.0            # the two x-side blocks (eps[ix, iy])
all3 = np.zeros((3, 3))
all3[0, 1] = all3[2, 1] = all3[1, 0] = all3[1, 2] = 1.0
layout3 = np.array([[0, 2, 0], [3, 1, 3], [0, 2, 0]])


_IDX = {}


def pick(o, R, T, xp, want):
    # the order indices are read from the NumPy solve (the JAX path may hand
    # back traced order labels under jit) and reused for the JAX one
    key = (CASE, tuple(map(tuple, np.atleast_2d(np.asarray(want)))))
    if xp is jnp and key in _IDX:
        idx = _IDX[key]
        return xp.concatenate([xp.stack([R[i] for i in idx]),
                               xp.stack([T[i] for i in idx])])
    o = np.asarray(o)
    idx = [int(np.nonzero((o[:, 0] == a) & (o[:, 1] == b))[0][0])
           for a, b in want] if o.ndim == 2 else [
        int(np.nonzero(o == a)[0][0]) for a in want]
    _IDX[key] = idx
    return xp.concatenate([xp.stack([R[i] for i in idx]),
                           xp.stack([T[i] for i in idx])])


def make(case, direction):
    up = (lambda a: None if a is None else np.kron(a, np.ones((5, 5)))
          ) if case.startswith("rcwa") else (lambda a: a)
    base, dirn = up(base3), up(direction)
    want2 = [(0, 0), (1, 0), (-1, 0)]

    def f(t, xp, pol):
        if case == "rcwa2d":
            eps = (xp.asarray(base) + t * xp.asarray(dirn)).astype(complex)
            o, R, T = rcwa_efficiency_2d(P, P, eps, 1.45, 1.0, 0.45, WL,
                                         polarization=pol, n_orders_x=3,
                                         n_orders_y=3)
            return pick(o, R, T, xp, want2)
        if case == "rcwa2d_theta":
            eps = xp.asarray(base).astype(complex)
            o, R, T = rcwa_efficiency_2d(P, P, eps, 1.45, 1.0, 0.45, WL,
                                         theta=t, polarization=pol,
                                         n_orders_x=3, n_orders_y=3)
            return pick(o, R, T, xp, want2)
        if case == "pmm2d":
            eps = (xp.asarray(base) + t * xp.asarray(dirn)).astype(complex)
            kw = {"region_layout": layout3} if xp is jnp else {}
            o, R, T = pmm_efficiency_2d_cell(P, P, eps, 1.45, 1.0, 0.45, WL,
                                             degree=7, n_orders=3,
                                             polarization=pol, **kw)
            return pick(o, R, T, xp, want2)
        if case == "pmm2d_theta":
            e4 = xp.asarray(4.0 + 0j)
            o, R, T = pmm_efficiency_2d(P, P, e4, xp.asarray(1.0 + 0j),
                                        (0.3, 0.9), (0.3, 0.9), 1.45, 1.0,
                                        0.45, WL, degree=7, n_orders=3,
                                        polarization=pol, theta=t)
            return pick(o, R, T, xp, want2)
        if case == "pmm1d":
            o, R, T = pmm_efficiency_1d(P, xp.asarray(2.0 + 0j), 1.0, 1.45,
                                        1.0, 0.45, 0.5, WL, angle=t,
                                        polarization=pol, degree=12,
                                        stabilize=False)
            return pick(o, R, T, xp, [1, -1])
        raise ValueError(case)
    return f


dirs = {"breaking": xs3, "keeping": all3} if CASE in ("rcwa2d", "pmm2d") \
    else {"breaking": None}
for dname, d in dirs.items():
    f = make(CASE, d)
    for pol in ("te", "tm"):
        f(0.0, np, pol)                 # the order indices, from NumPy
        g = np.asarray(jax.jit(jax.jacrev(lambda t, pol=pol: f(
            t, jnp, pol)))(0.0))
        steps = [1e-3, 3e-4, 1e-4]
        _rows, fd, _ch, rat = ladder(lambda t, pol=pol: np.asarray(
            f(t, np, pol)), 0.0, steps, 1.0)
        sc = float(np.max(np.abs(fd)))
        rec = {"AD": g.tolist(), "FD_numpy": fd.tolist(),
               "premise": np.asarray(rat).ravel().round(3).tolist(),
               "scale": sc, "rel_err": float(np.max(np.abs(g - fd))) / sc}
        out[f"{dname}_{pol}"] = rec
        print(CASE, dname, pol, "rel err %.2e" % rec["rel_err"],
              "scale %.2e" % sc, "premise", rec["premise"][:6], flush=True)
print(dump(f"r4_other_twins_{CASE}.json", out))
