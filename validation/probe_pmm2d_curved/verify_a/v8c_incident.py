"""V8c -- why a vacuum spacer / n_orders move R / T under a non-polynomial
map: the superstrate's discrete incident decomposition.  The stack takes
cinc = lstsq(Hsup, delta_00) over the retained orders; under the identity map
the (0, 0) plane wave is an EXACT discrete mode, so cinc is one propagating
mode; under a stretch it is not, and cinc carries EVANESCENT modal content
whose effect depends on the reference plane.  Measured: the evanescent share
||cinc_evan|| / ||cinc|| and the residual ||Hsup cinc - delta||."""
import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import twod_staggered as TS
from lumenairy.elements.pmm._core import _guarded_lstsq
from lumenairy.elements.pmm._curvemap import IdentityMap

rows = []
for tag, cm in (("ident3", IdentityMap(3, 3, C.P, C.P)),
                ("sine0.02_u3", C.stretch_map(C.sine(0.02), np.linspace(0, 1, 4),
                                              np.linspace(0, 1, 4))),
                ("harm_asym_pre", C.stretch_map(C.HarmonicStretch(0.10, 0.04,
                                                                  0.9)))):
    for M in (5, 7):
        for n_orders in (2, 3):
            s = TS.Granet2DTransverseE(C.P, C.P, cm.u_walls, cm.v_walls, M,
                                       np.ones((3, 3), complex), k0=C.K0,
                                       cmap=cm)
            geom = TS._homog_geom_cache(s)
            W, V, lam = TS._homog_region_modes(geom, 1.0 + 0j)
            ox = np.arange(-n_orders, n_orders + 1)
            P1, P2, P12, P21 = TS._far_projector_2d(s.bx, s.by, ox, ox,
                                                    cmap=cm)
            qq = s.q * s.q
            H = TS._pmm2d_project_orders(P1, P2, W, qq, P12, P21)
            Nfo = ox.size ** 2
            p0 = Nfo // 2
            rhs = np.zeros(2 * Nfo, complex)
            rhs[p0] = 1.0
            c = _guarded_lstsq(H, rhs, "v8c")
            evan = np.abs(lam.real) > 1e-6        # decaying modes
            r = dict(map=tag, M=M, n_orders=n_orders,
                     evanescent_share=float(np.linalg.norm(c[evan])
                                            / np.linalg.norm(c)),
                     residual=float(np.linalg.norm(H @ c - rhs)))
            rows.append(r)
            print(r, flush=True)
C.dump("v8c_incident", {"rows": rows})
