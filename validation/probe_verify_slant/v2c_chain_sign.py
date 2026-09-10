"""V2c -- CLOSING THE SIGN CHAIN to actual GEOMETRY, in three independent links.

V2b showed the pure engine's public ``slant`` equals ``+tan(slant_angle)`` of
the 1-D oracle.  That is a cross-engine statement.  This probe turns it into a
statement about a DRAWING, and checks the third engine:

LINK 1 -- what does the 1-D ``slant_angle`` DRAW?  ``PMMStack.add_layer(
segments=..., slant_angle=s)`` versus a z-STAIRCASE built with
``PMMStack.add_layer(segments=...)`` alone, in which each slice's ridge is
placed EXPLICITLY at a centre that walks with depth (``_ridge_slice_segments``'
documented convention: the ridge is the period-1 interval
``[centre - duty/2, centre + duty/2)`` laid out from ``u = 0``, so a centre
that GROWS with depth is a walk toward +x).  Both signs, a slice ladder.

LINK 2 -- do the 2-D pure engine's CELL-INDEX and ORDER directions agree with
the 1-D engine's?  A three-level x-ASYMMETRIC y-uniform VERTICAL grating
([4,4,2,1,1,1] over six equal segments) at OBLIQUE incidence, per order, 2-D
versus 1-D.  If either the cell index or the order index ran backwards in the
2-D engine, this comparison would fail on the +/-m split -- and a slant-sign
error could then hide behind a double flip.  The MIRRORED profile is run as the
wrong arm.

LINK 3 -- the third engine.  ``PMM2DStackHybrid.add_layer(slant=)`` (a Fourier
basis and a lab-Cartesian convection: nothing in common with the pure
formulation but the physics) against the pure slanted layer, both signs, on an
``n_orders`` ladder, on a genuinely 2-D pillar.
"""
import numpy as np
from _lib import arm, dump  # noqa: I001

from lumenairy.elements.pmm import PMM2DStackHybrid, PMM2DStackPure, PMMStack
from lumenairy.elements.pmm.twod_staggered import pmm_jones_2d_staggered

WL = 0.68e-6
PER = 0.75 * WL
DEP = 0.30 * WL
NSUP, NSUB = 1.0, 1.5
DUTY = 0.5
EPS_R, EPS_G = 4.0, 1.0
DEG1D = 20
NORD1D = 7


# --------------------------------------------------------------- LINK 1
def _ridge_segments(centre, duty=DUTY, er=EPS_R, eg=EPS_G):
    """The consecutive ``(width_fraction, eps)`` list of ONE binary ridge whose
    centre sits at ``centre`` period-fractions, laid out from ``u = 0``.  The
    ridge is ``[centre - duty/2, centre + duty/2)`` reduced mod 1."""
    lo = (centre - duty / 2.0) % 1.0
    hi = lo + duty
    if hi <= 1.0:
        segs = [(lo, eg), (duty, er), (1.0 - hi, eg)]
    else:
        segs = [(hi - 1.0, er), (1.0 - lo, eg), (lo - (hi - 1.0), er)]
    return [(w, e) for w, e in segs if w > 1e-12]


def _stack1d(angle):
    return PMMStack(PER, n_substrate=NSUB, n_superstrate=NSUP, degree=DEG1D,
                    n_orders=NORD1D)


def one_d_slanted(shear, angle):
    """ONE slanted layer, ridge centre 0.5 at MID-depth, walking ``shear``
    periods top->bottom (the ``add_sheared_grating`` centre law)."""
    st = _stack1d(angle)
    st.add_sheared_grating(DEP, eps_ridge=EPS_R, eps_groove=EPS_G, duty=DUTY,
                           shear=float(shear), centre=0.5)
    st.set_source(WL, angle=angle)
    return st.solve()


def one_d_slant_angle(sa, angle, centre=0.5):
    """ONE slanted layer via the RAW ``slant_angle`` keyword (no builder)."""
    st = _stack1d(angle)
    st.add_layer(DEP, segments=_ridge_segments(centre),
                 slant_angle=float(sa))
    st.set_source(WL, angle=angle)
    return st.solve()


def one_d_stair(shear, angle, K):
    """The SAME solid as a K-slice vertical staircase, ridge centre placed
    explicitly at ``0.5 + shear * (zeta - 0.5)``."""
    st = _stack1d(angle)
    d = DEP / K
    for k in range(K):
        zeta = (k + 0.5) / K
        st.add_layer(d, segments=_ridge_segments(0.5 + shear * (zeta - 0.5)))
    st.set_source(WL, angle=angle)
    return st.solve()


def _rt(res):
    o, R, T, J = res
    return np.asarray(o), np.asarray(R), np.asarray(T)


def _cmp_rt(a, b):
    oa, Ra, Ta = _rt(a)
    ob, Rb, Tb = _rt(b)
    common = sorted(set(oa.tolist()) & set(ob.tolist()))
    ia = [int(np.where(oa == m)[0][0]) for m in common]
    ib = [int(np.where(ob == m)[0][0]) for m in common]
    return float(max(np.max(np.abs(Ra[:, ia] - Rb[:, ib])),
                     np.max(np.abs(Ta[:, ia] - Tb[:, ib]))))


# --------------------------------------------------------------- LINK 2
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
NXA = 6


def two_d_vertical(prof, theta, M=6, nord=3):
    cell = np.empty((NXA, NXA), dtype=complex)
    cell[:, :] = np.asarray(prof, dtype=complex)[:, None]
    o, R, T, J = pmm_jones_2d_staggered(
        PER, PER, cell, NSUB, NSUP, DEP, WL, degree=M, n_orders=nord,
        theta=theta, phi=0.0)
    o = np.asarray(o)
    sel = o[:, 1] == 0
    m = o[sel, 0]
    i = np.argsort(m)
    return m[i], R[:, sel][:, i], T[:, sel][:, i]


def one_d_vertical_profile(prof, theta):
    st = _stack1d(theta)
    st.add_layer(DEP, segments=[(1.0 / len(prof), float(e)) for e in prof])
    st.set_source(WL, angle=theta)
    return _rt(st.solve())


def _cmp_2d_1d(m2, R2, T2, o1, R1, T1):
    common = sorted(set(m2.tolist()) & set(o1.tolist()))
    i2 = [int(np.where(m2 == m)[0][0]) for m in common]
    i1 = [int(np.where(o1 == m)[0][0]) for m in common]
    # 2-D row 0 = incident Ex = TM at phi = 0 = 1-D row 0; row 1 = Ey = TE
    return float(max(np.max(np.abs(R2[:, i2] - R1[:, i1])),
                     np.max(np.abs(T2[:, i2] - T1[:, i1]))))


# --------------------------------------------------------------- LINK 3
PXB = PYB = 1.20e-6
DEPB = 0.80e-6
NXB = 3
TB = 0.75


def pillar_cell():
    c = np.ones((NXB, NXB), dtype=complex)
    c[1, 1] = 4.0
    return c


def pure_pillar(sl, theta, phi, M=5, nord=3):
    st = PMM2DStackPure(PXB, PYB, n_superstrate=NSUP, n_substrate=NSUB,
                        n_modes=M, n_orders=nord)
    st.add_layer(DEPB, eps_cell=pillar_cell(), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(jones=True)
    return np.asarray(o), R, T


def hyb_pillar(sl, theta, phi, nord):
    st = PMM2DStackHybrid(PXB, PYB, n_superstrate=NSUP, n_substrate=NSUB,
                          n_orders=nord)
    st.add_layer(DEPB, eps_cell=pillar_cell(), slant=sl)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve()
    return np.asarray(o), R, T


def _cmp_2d(a, b):
    oa, Ra, Ta = a
    ob, Rb, Tb = b
    ka = {tuple(int(v) for v in oa[i]): i for i in range(len(oa))}
    kb = {tuple(int(v) for v in ob[i]): i for i in range(len(ob))}
    common = sorted(set(ka) & set(kb))
    ia = [ka[k] for k in common]
    ib = [kb[k] for k in common]
    return float(max(np.max(np.abs(Ra[:, ia] - Rb[:, ib])),
                     np.max(np.abs(Ta[:, ia] - Tb[:, ib]))))


def main():
    out = {"link1_1d_geometry": {}, "link2_index_direction": {},
           "link3_hybrid": {}}

    # ---- LINK 1
    for aname, ang in (("normal", 0.0), ("oblique17", 0.17)):
        for shear in (0.25, 0.40):
            sa = float(np.arctan(shear * PER / DEP))
            row = {"shear": shear, "slant_angle_deg": float(np.rad2deg(sa))}
            sheared_p = one_d_slanted(+shear, ang)
            sheared_m = one_d_slanted(-shear, ang)
            raw_p = one_d_slant_angle(+sa, ang)
            raw_m = one_d_slant_angle(-sa, ang)
            row["sheared_plus_vs_rawangle_plus"] = _cmp_rt(sheared_p, raw_p)
            row["sheared_minus_vs_rawangle_minus"] = _cmp_rt(sheared_m, raw_m)
            row["sheared_plus_vs_rawangle_minus"] = _cmp_rt(sheared_p, raw_m)
            for K in (2, 4, 8, 16):
                stp = one_d_stair(+shear, ang, K)
                row[f"K{K}_walkPLUS_vs_shearPLUS"] = _cmp_rt(stp, sheared_p)
                row[f"K{K}_walkPLUS_vs_shearMINUS"] = _cmp_rt(stp, sheared_m)
            out["link1_1d_geometry"][f"{aname}/shear{shear}"] = row
            lad = [row[f"K{K}_walkPLUS_vs_shearPLUS"] for K in (2, 4, 8, 16)]
            ladm = [row[f"K{K}_walkPLUS_vs_shearMINUS"] for K in (2, 4, 8, 16)]
            print(f"[L1] {aname} shear {shear}: +walk vs +shear "
                  f"{' '.join('%.3e' % v for v in lad)} | vs -shear "
                  f"{' '.join('%.3e' % v for v in ladm)} | raw-angle "
                  f"{row['sheared_plus_vs_rawangle_plus']:.2e} / wrong "
                  f"{row['sheared_plus_vs_rawangle_minus']:.2e}")

    # ---- LINK 2
    for aname, ang in (("oblique25", np.deg2rad(25.0)),
                       ("oblique-25", -np.deg2rad(25.0))):
        m2, R2, T2 = two_d_vertical(XPROF, ang)
        o1, R1, T1 = one_d_vertical_profile(XPROF, ang)
        o1r, R1r, T1r = one_d_vertical_profile(XPROF[::-1], ang)
        row = dict(same_profile=_cmp_2d_1d(m2, R2, T2, o1, R1, T1),
                   mirrored_profile=_cmp_2d_1d(m2, R2, T2, o1r, R1r, T1r))
        out["link2_index_direction"][aname] = row
        print(f"[L2] {aname}: 2-D vs 1-D same profile {row['same_profile']:.3e}"
              f" | MIRRORED profile {row['mirrored_profile']:.3e}")

    # ---- LINK 3
    for mname, (th, ph) in (("normal", (0.0, 0.0)),
                            ("conical", (np.deg2rad(20.0),
                                         np.deg2rad(35.0)))):
        pp = pure_pillar((+TB, 0.0), th, ph)
        pm = pure_pillar((-TB, 0.0), th, ph)
        pv = pure_pillar(None, th, ph)
        row = {"pure_plus_vs_vertical": _cmp_2d(pp, pv),
               "pure_plus_vs_pure_minus": _cmp_2d(pp, pm)}
        prev = None
        for nord in (3, 5, 7, 9):
            hp = hyb_pillar((+TB, 0.0), th, ph, nord)
            hm = hyb_pillar((-TB, 0.0), th, ph, nord)
            row[f"n{nord}_hybPLUS_vs_purePLUS"] = _cmp_2d(hp, pp)
            row[f"n{nord}_hybMINUS_vs_purePLUS"] = _cmp_2d(hm, pp)
            row[f"n{nord}_hybPLUS_vs_pureMINUS"] = _cmp_2d(hp, pm)
            if prev is not None:
                row[f"n{nord}_hybrid_own_step"] = _cmp_2d(hp, prev)
            prev = hp
        out["link3_hybrid"][mname] = row
        lad = [row[f"n{n}_hybPLUS_vs_purePLUS"] for n in (3, 5, 7, 9)]
        ladw = [row[f"n{n}_hybMINUS_vs_purePLUS"] for n in (3, 5, 7, 9)]
        print(f"[L3] {mname}: hyb(+t) vs pure(+t) "
              f"{' '.join('%.3e' % v for v in lad)} | hyb(-t) vs pure(+t) "
              f"{' '.join('%.3e' % v for v in ladw)} | slant effect "
              f"{row['pure_plus_vs_vertical']:.3e}")

    dump("v2c_chain_sign", out)
    print("arm", arm())


if __name__ == "__main__":
    main()
