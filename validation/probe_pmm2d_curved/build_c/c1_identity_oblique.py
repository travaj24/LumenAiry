"""C1 side measurement -- the IDENTITY map (rectangles only) through the
mapped path against the shipped unmapped solver, at normal and oblique
incidence.  At normal incidence the plane wave is an exact discrete mode on
both paths (round-off agreement); at oblique incidence the two paths differ
by their INCIDENT treatment (unmapped: the least-squares overlap; mapped: the
exact renormalised L2 decomposition), a discretisation-level difference that
must fall with M.

  c1_identity_oblique.py <M>
Output: c1_identity_oblique_M<M>.json
"""
import sys

import _common as C
import numpy as np

from lumenairy.elements.pmm import Rect, compile_shapes  # noqa: E402


def run(M):
    shp = [Rect(0.55, 0.62, 0.5, 0.4, 4.0)]
    eps, _x, _y, cm = compile_shapes(C.P, C.P, shp, 1.0)
    res = {"env": C.env_record(), "M": M}
    for th, ph in ((0.0, 0.0), (0.3, 0.2)):
        out = []
        for route in ("shapes", "mapped_identity"):
            st = C.PMM2DStackPure(C.P, C.P, n_superstrate=C.N_SUP,
                                  n_substrate=C.N_SUB, n_modes=M, n_orders=3,
                                  cmap=None if route == "shapes" else cm)
            if route == "shapes":
                st.add_layer(C.DEPTH, shapes=shp, background_eps=1.0)
            else:
                st.add_layer(C.DEPTH, eps_cell=eps)
            st.set_source(C.WL, theta=th, phi=ph)
            o, R, T, J = st.solve()
            out.append((np.asarray(R), np.asarray(T), np.asarray(J)))
        res[f"theta{th}_phi{ph}"] = float(max(
            np.max(np.abs(a - b)) for a, b in zip(*out)))
    print(res, flush=True)
    C.dump(f"c1_identity_oblique_M{M}.json", res)


if __name__ == "__main__":
    run(int(sys.argv[1]))
