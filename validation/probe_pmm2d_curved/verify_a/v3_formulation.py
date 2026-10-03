"""V3 -- the formulation, re-measured on a NON-SEPARABLE (sheared) map.

The build gated only the identity and SEPARABLE stretches, on which g12 = 0:
the mixed tensor entries (e12 = e21 = -eps g12 / sqrt g, chi12 = chi21 =
g12 / sqrt g) and the off-diagonal cofactor blocks (P12, P21) are identically
zero there, so neither their SIGN nor their PLACEMENT was ever exercised.
This probe drives them with the verifier's sheared map (``_vcommon.
make_shear_map``: straight walls at 0, p/2, p, every cell interior sheared,
no mirror symmetry):

  film   uniform eps under the shear vs the exact Airy slab (s / p), normal,
         oblique 25 deg (phi 0) and conical (25, 40 deg), M = 4..8;
  stripe the half-period ridge x in [0, p/2] under the shear vs
         pmm_efficiency_1d (TE / TM, normal), M = 4..9; plus the unmapped
         solve on the same 2 x 2 walls;
  proj   the four far-field projector arms on the film: the shipped
         COFACTOR det J J^-T, J^T, J^-T (no area element), identity;
         on the SEPARABLE stretch film at oblique incidence and on the shear.

usage: v3_formulation.py film|stripe|proj [M_lo M_hi]
"""
import sys

import _vcommon as C
import numpy as np

from lumenairy.elements.pmm import stack2d_pure as SP

ARM = sys.argv[1]
M_LO = int(sys.argv[2]) if len(sys.argv) > 2 else 4
M_HI = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SHEAR = (0.06, 0.05)


def film_err(o, R, T, theta):
    ex = C.airy(theta)
    i0 = C.i00(o)
    # row 0 = incident E_x, row 1 = incident E_y.  phi = 0: E_x is p, E_y s.
    # Conical (phi != 0): the polarisations mix, so compare the SUM over
    # orders of R and T against the polarisation-independent bound only
    # through the (0,0) order of the unmapped solve (see 'ref' arms).
    out = []
    for row, pol in ((0, "p"), (1, "s")):
        Rr = R[row].copy()
        Tr = T[row].copy()
        Rr[i0] -= ex[pol][0]
        Tr[i0] -= ex[pol][1]
        out.append(float(max(np.abs(Rr).max(), np.abs(Tr).max())))
    return out


def run_film():
    res = {"shear": SHEAR}
    cm = C.make_shear_map(*SHEAR)
    res["detJ_range"] = C.detj_range(cm)
    film = C.cell("film", n=2)
    for tag, th, ph in (("normal", 0.0, 0.0), ("oblique25", np.radians(25),
                                                0.0),
                        ("conical25_40", np.radians(25), np.radians(40))):
        rows = []
        for M in range(M_LO, M_HI + 1):
            o, R, T, J, _ = C.stack_solve(cm, [film], M, theta=th, phi=ph)
            o0, R0, T0, J0, _ = C.stack_solve(None, [film], M, theta=th,
                                              phi=ph)
            i0 = C.i00(o)
            # vs the UNMAPPED solve (exact on a film to round-off): every
            # order and the full zero-order Jones (rotation-invariant check
            # for conical incidence)
            d_eff = float(max(np.abs(R - R0).max(), np.abs(T - T0).max()))
            d_jones = float(np.abs(J - J0).max())
            row = dict(M=M, vs_unmapped_eff=d_eff, vs_unmapped_jones=d_jones,
                       closure=C.closure(R, T),
                       nonzero_orders=float(max(
                           np.delete(R, i0, axis=1).max(),
                           np.delete(T, i0, axis=1).max())))
            if ph == 0.0:
                row["vs_airy"] = film_err(o, R, T, th)
                row["unmapped_vs_airy"] = film_err(o0, R0, T0, th)
            rows.append(row)
            print(tag, row, flush=True)
        res[tag] = rows
    C.dump("v3_film_shear", res)


def run_stripe():
    res = {"shear": SHEAR}
    cm = C.make_shear_map(*SHEAR)
    st = np.ones((2, 2), complex)
    st[0, :] = C.EPS_F                  # ridge x in [0, p/2] (u-cell 0)
    refs = {pol: C.oracle_1d(pol, x_fill=0.5, degree=40) for pol in
            ("te", "tm")}
    gap = {pol: max(abs(a - b) for m in refs[pol]
                    for a, b in zip(refs[pol][m],
                                    C.oracle_1d(pol, 0.5, degree=48)[m]))
           for pol in refs}
    res["oracle_selfgap_40_48"] = gap
    rows = []
    for M in range(M_LO, M_HI + 1):
        o, R, T, J, _ = C.stack_solve(cm, [st], M)
        o0, R0, T0, J0, _ = C.stack_solve(None, [st], M)
        row = dict(M=M,
                   map_tm=C.stripe_err(o, R, T, 0, refs["tm"]),
                   map_te=C.stripe_err(o, R, T, 1, refs["te"]),
                   unm_tm=C.stripe_err(o0, R0, T0, 0, refs["tm"]),
                   unm_te=C.stripe_err(o0, R0, T0, 1, refs["te"]),
                   closure=C.closure(R, T))
        rows.append(row)
        print(row, flush=True)
    res["ladder"] = rows
    C.dump(f"v3_stripe_shear_M{M_LO}-{M_HI}", res)


class _View:
    """The far projector reads (X, Y, x_u, x_v, y_u, y_v) and forms the
    coefficient matrix [[y_v, -y_u], [-x_v, x_u]].  A view returns modified
    Jacobian entries so that matrix becomes the requested ARM."""

    def __init__(self, m, arm):
        self._m, self.arm = m, arm

    def geom(self, sx, sy, U, V):
        X, Y, xu, xv, yu, yv = self._m.geom(sx, sy, U, V)
        if self.arm == "JT":           # coefficients [[xu, yu], [xv, yv]]
            return X, Y, yv, -xv, -yu, xu
        if self.arm == "JinvT":        # cofactor / det J (no area element)
            d = xu * yv - xv * yu
            return X, Y, xu / d, xv / d, yu / d, yv / d
        if self.arm == "ident":
            return (X, Y, np.ones_like(xu), np.zeros_like(xv),
                    np.zeros_like(yu), np.ones_like(yv))
        raise ValueError(self.arm)


def run_proj():
    res = {}
    orig = SP._far_projector_2d
    cases = (("stretch0.12_oblique25", C.stretch_map(C.sine(0.12),
                                                      np.linspace(0, 1, 4),
                                                      np.linspace(0, 1, 4)),
              3, np.radians(25), 0.0),
             ("stretch0.12_normal", C.stretch_map(C.sine(0.12),
                                                  np.linspace(0, 1, 4),
                                                  np.linspace(0, 1, 4)),
              3, 0.0, 0.0),
             ("shear_oblique25", C.make_shear_map(*SHEAR), 2,
              np.radians(25), 0.0),
             ("shear_normal", C.make_shear_map(*SHEAR), 2, 0.0, 0.0))
    for tag, cm, n, th, ph in cases:
        film = C.cell("film", n=n)
        rows = {}
        for arm in ("cof", "JT", "JinvT", "ident"):
            if arm == "cof":
                SP._far_projector_2d = orig
            else:
                def patched(bx, by, ox, oy, a0x=0.0, a0y=0.0, cmap=None,
                            _arm=arm):
                    if cmap is not None:
                        cmap = _View(cmap, _arm)
                    return orig(bx, by, ox, oy, a0x, a0y, cmap=cmap)
                SP._far_projector_2d = patched
            out = []
            for M in (5, 7):
                o, R, T, J, _ = C.stack_solve(cm, [film], M, theta=th,
                                              phi=ph)
                out.append(dict(M=M, vs_airy=film_err(o, R, T, th),
                                closure=C.closure(R, T)))
            SP._far_projector_2d = orig
            rows[arm] = out
            print(tag, arm, out, flush=True)
        res[tag] = rows
    C.dump("v3_proj_arms", res)


{"film": run_film, "stripe": run_stripe, "proj": run_proj}[ARM]()
