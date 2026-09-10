"""V2a -- WHAT DOES THE PUBLIC ``slant`` MEAN GEOMETRICALLY?  A convention-free
arbiter built out of ``np.roll`` alone.

The build pins the sign against three engines (the 1-D slanted oracle, the
hybrid metric, a pure staircase).  All three are CROSS-ENGINE statements: they
say the keyword is consistent, not what it draws.  This probe asks the
independent question -- in the caller's OWN ``eps_cell`` index frame, which way
does the cross-section walk as depth increases? -- and answers it without any
sign convention at all, by building the same structure twice:

  * ONE slanted layer, ``slant=(+t, 0)`` and ``slant=(-t, 0)``;
  * a z-STAIRCASE of VERTICAL layers whose cells are ``np.roll(cell, +n, 0)``
    -- and ``np.roll(a, +n, axis=0)[i] = a[i - n]``, i.e. the pattern moves to
    HIGHER x index.  No slant keyword appears in the staircase at all.

The walk is a WHOLE period over the layer on a 6-cell grid, so every slice of
every ladder rung lands on the union grid exactly and the staircase is an EXACT
sampling of the sheared solid (``t = px/depth`` = 1.0, a 45-degree wall).
Ladder ``K = 1, 2, 3, 6`` (the divisors of the walk): the correct sign must
improve MONOTONICALLY along it, the wrong one must not.

Two notes the design had to account for, both measured:

* a TRAILING staircase (cell read at each slice's BOTTOM face) is the LEADING
  one rolled globally by one step -- a rigid lateral translation of the whole
  structure, which changes no efficiency and no zeroth-order Jones.  It is
  therefore not an independent arm and is not built.
* the discriminating observable is the PER-ORDER T (and R), not the
  zeroth-order reflection Jones: for a cell that is its own mirror image up to
  a translation the +/-t structures are mirror-related, so at normal incidence
  their zeroth-order Jones agree to 5e-15 while their per-order T differs by
  2.6e-01.  The cell used below is x-ASYMMETRIC ([4,4,2,1,1,1]) so that even
  the zeroth-order Jones separates.

Run on BOTH engines -- the pure staggered one (the thing under verification)
and the independent Fourier hybrid -- so "the two public conventions agree" is
measured rather than assumed.
"""
import numpy as np
from _lib import arm, dump, mx  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure

WL = 0.68e-6
PX = PY = 1.20e-6
DEP = 1.20e-6
NSUP, NSUB = 1.0, 1.5
NX = NY = 6
M = 3
NORD = 3
WALK_CELLS = 6                       # one whole period over the layer
T = (WALK_CELLS * (PX / NX)) / DEP   # = 1.0 exactly (45 degrees)
MOUNTS = {"normal": (0.0, 0.0),
          "oblique25": (np.deg2rad(25.0), 0.0),
          "conical": (np.deg2rad(25.0), np.deg2rad(40.0))}
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])   # x-ASYMMETRIC, 3 levels


def cell2d():
    c = np.ones((NX, NY), dtype=complex)
    c[:, 0:3] = XPROF[:, None]
    return c


def cell_stripe():
    c = np.empty((NX, NY), dtype=complex)
    c[:, :] = XPROF[:, None]
    return c


def pure_slant(cell, sl, theta, phi):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=NORD)
    st.add_layer(DEP, eps_cell=cell, slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T_, J = st.solve(jones=True)
    return dict(R=R, T=T_, J=J, clo=float(np.max(R.sum(1) + T_.sum(1))))


def pure_stair(cell, rolls, theta, phi):
    st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=NORD)
    d = DEP / len(rolls)
    for n in rolls:
        st.add_layer(d, eps_cell=np.roll(cell, int(n), axis=0))
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T_, J = st.solve(jones=True)
    return dict(R=R, T=T_, J=J, clo=float(np.max(R.sum(1) + T_.sum(1))))


def hyb_slant(cell, sl, theta, phi, nord=9):
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=nord)
    st.add_layer(DEP, eps_cell=cell, slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T_, J = st.solve()
    return dict(R=R, T=T_, J=J, clo=float(np.max(R.sum(1) + T_.sum(1))))


def hyb_stair(cell, rolls, theta, phi, nord=9):
    st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=nord)
    d = DEP / len(rolls)
    for n in rolls:
        st.add_layer(d, eps_cell=np.roll(cell, int(n), axis=0))
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T_, J = st.solve()
    return dict(R=R, T=T_, J=J, clo=float(np.max(R.sum(1) + T_.sum(1))))


def _d(a, b):
    return dict(dJ=mx(a["J"], b["J"]), dR=mx(a["R"], b["R"]),
                dT=mx(a["T"], b["T"]),
                dRT=max(mx(a["R"], b["R"]), mx(a["T"], b["T"])))


def rolls_leading(K, walk=WALK_CELLS):
    return [walk * k // K for k in range(K)]


LADDER = (1, 2, 3, 6)


def main():
    out = {"geometry": dict(PX=PX, PY=PY, DEP=DEP, WL=WL, NX=NX, M=M,
                            walk_cells=WALK_CELLS, t=T, n_orders=NORD,
                            xprofile=list(np.real(XPROF))),
           "pure": {}, "hybrid": {}}
    for cname, cell in (("pillar2d", cell2d()),
                        ("stripe_yuniform", cell_stripe())):
        for mname, (th, ph) in MOUNTS.items():
            key = f"{cname}/{mname}"
            sp = pure_slant(cell, (+T, 0.0), th, ph)
            sm = pure_slant(cell, (-T, 0.0), th, ph)
            vert = pure_slant(cell, None, th, ph)
            row = {"slant_effect_vs_vertical": _d(sp, vert),
                   "plus_vs_minus": _d(sp, sm),
                   "closure": dict(plus=sp["clo"], minus=sm["clo"],
                                   vertical=vert["clo"])}
            for K in LADDER:
                rl = rolls_leading(K)
                stp = pure_stair(cell, rl, th, ph)
                stm = pure_stair(cell, [-n for n in rl], th, ph)
                row[f"K{K}_plusroll_vs_slantPLUS"] = _d(stp, sp)
                row[f"K{K}_plusroll_vs_slantMINUS"] = _d(stp, sm)
                row[f"K{K}_minusroll_vs_slantMINUS"] = _d(stm, sm)
                row[f"K{K}_minusroll_vs_slantPLUS"] = _d(stm, sp)
                row[f"K{K}_closure"] = dict(plusroll=stp["clo"],
                                            minusroll=stm["clo"])
            out["pure"][key] = row
            lad_p = [row[f"K{K}_plusroll_vs_slantPLUS"]["dRT"] for K in LADDER]
            lad_m = [row[f"K{K}_plusroll_vs_slantMINUS"]["dRT"] for K in LADDER]
            print(f"[pure] {key}: +roll ladder vs +t "
                  f"{' '.join('%.3e' % v for v in lad_p)}  |  vs -t "
                  f"{' '.join('%.3e' % v for v in lad_m)}")

    for cname, cell in (("pillar2d", cell2d()),):
        for mname, (th, ph) in MOUNTS.items():
            key = f"{cname}/{mname}"
            sp = hyb_slant(cell, (+T, 0.0), th, ph)
            sm = hyb_slant(cell, (-T, 0.0), th, ph)
            row = {"plus_vs_minus": _d(sp, sm),
                   "closure": dict(plus=sp["clo"], minus=sm["clo"])}
            for K in LADDER:
                rl = rolls_leading(K)
                stp = hyb_stair(cell, rl, th, ph)
                row[f"K{K}_plusroll_vs_slantPLUS"] = _d(stp, sp)
                row[f"K{K}_plusroll_vs_slantMINUS"] = _d(stp, sm)
                row[f"K{K}_closure"] = dict(plusroll=stp["clo"])
            out["hybrid"][key] = row
            lad_p = [row[f"K{K}_plusroll_vs_slantPLUS"]["dRT"] for K in LADDER]
            lad_m = [row[f"K{K}_plusroll_vs_slantMINUS"]["dRT"] for K in LADDER]
            print(f"[hyb ] {key}: +roll ladder vs +t "
                  f"{' '.join('%.3e' % v for v in lad_p)}  |  vs -t "
                  f"{' '.join('%.3e' % v for v in lad_m)}")

    dump("v2a_geometry_sign", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
