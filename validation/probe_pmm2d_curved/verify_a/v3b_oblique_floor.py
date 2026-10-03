"""V3b -- the sheared film at OBLIQUE incidence stalls near 4e-7 at M = 7..8
(v3_film_shear): is it a floor, and does it depend on n_orders (the
incident-overlap mechanism of v8c) or on the far projector's rule?"""
import _vcommon as C
import numpy as np

cm = C.make_shear_map(0.06, 0.05)
th = np.radians(25)
ex = C.airy(th)
rows = []
for M in (7, 8, 9, 10):
    for n_orders in (2, 3, 4):
        o, R, T, J, _ = C.stack_solve(cm, [C.cell("film", n=2)], M, theta=th,
                                      n_orders=n_orders)
        i0 = C.i00(o)
        e = []
        for row, pol in ((0, "p"), (1, "s")):
            e.append(float(max(abs(R[row, i0] - ex[pol][0]),
                               abs(T[row, i0] - ex[pol][1]))))
        r = dict(M=M, n_orders=n_orders, err_p=e[0], err_s=e[1],
                 closure=C.closure(R, T))
        rows.append(r)
        print(r, flush=True)
C.dump("v3b_oblique_floor", {"rows": rows})
