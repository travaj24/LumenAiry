"""TASK 4 -- O2: re-derive the ``T22`` bar on THIS verification's own
population.

TERMS.  ``_interface_smatrix_general`` is the GENERALIZED (4N) cascade's
layer<->region mode match; it inverts the ``T22`` block EXPLICITLY.
``_guarded_inverse``'s FREE instrument is the equilibrated reciprocal
1-condition estimate ``rcond(T22)``; its CONFIRMING instrument is the
equilibrated residual of ``A X = I``.  The shipped screen refuses when BOTH
miss (``rcond < 1e-10`` AND ``resid > 1e-8``).

A solve counts BROKEN here when, on a tree WITHOUT the guard, it returns (or
raises ``_EnergyError`` on) ``sum R + T`` above ``1.10`` per incident state --
the library's own energy criterion, which is not the instrument being
calibrated.  A solve counts REFUSED when, on the tree WITH the guard, it
raises ``_ConditioningError``.

The census is armed through ``rcwa._core._INV_CENSUS``, which
``_guarded_inverse`` reads as a module global at call time, so every consumer
that imported the interface function by name is instrumented from one place.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

if os.environ.get("LUM_ARM_TREE"):
    sys.path.insert(0, os.environ["LUM_ARM_TREE"])
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
import numpy as np  # noqa: E402

UM = 1e-6


# ---------------------------------------------------------------------------
# the fixture family
# ---------------------------------------------------------------------------

def _oop(exx=3.6, ezz=3.0, exz=0.7):
    m = np.eye(3, dtype=complex)
    m[0, 0], m[1, 1], m[2, 2] = exx, 2.4, ezz
    m[0, 2] = m[2, 0] = exz
    return m


def _tensor_cell(sx=6, sy=6, lo=1.0, hi=3.4, oop=False):
    c = np.zeros((sx, sy, 3, 3), dtype=complex)
    for i in range(sx):
        for j in range(sy):
            v = lo + (hi - lo) * (((i * 2 + j) % 5) / 4.0)
            c[i, j] = np.eye(3) * v
            if oop:
                c[i, j, 0, 2] = c[i, j, 2, 0] = 0.5 * ((i + j) % 2)
    return c


def fixtures():                                              # noqa: C901
    """``name -> callable`` over every consumer of the generalized
    interface."""
    from lumenairy.elements import berreman_jones_1d
    from lumenairy.elements.pmm import (
        PMM2DStackHybrid,
        PMM2DStackPure,
        PMMStack,
        pmm_jones_2d,
        pmm_jones_2d_staggered,
    )
    F = {}

    # -- 1-D PMMStack: convection SLANT, shared grid ---------------------
    def _s1d(shear, norders, degree, grids="shared", fac="convection",
             theta=25.0, oop=False, wl=0.62, per=1.10, nsub=1.5, nsup=1.0):
        def go():
            st = PMMStack(per * UM, n_substrate=nsub, n_superstrate=nsup,
                          degree=degree, n_orders=norders,
                          factorization=fac, layer_grids=grids)
            if shear:
                st.add_sheared_grating(0.40 * UM, eps_ridge=3.4,
                                       eps_groove=1.0, duty=0.45,
                                       shear=shear)
            if oop:
                st.add_layer(0.22 * UM,
                             segments=[(0.45, _oop()), (0.55, 1.2)])
            if not shear and not oop:
                st.add_layer(0.40 * UM, segments=[(0.45, 3.4), (0.55, 1.0)])
            st.set_source(wl * UM, angle=np.deg2rad(theta))
            return st.solve()
        return go

    for sh in (0.20, 0.35, 0.50, 0.70):
        for no in (3, 5, 7, 9, 11):
            F[f"pmm1d_slant{sh}_M{no}"] = _s1d(sh, no, 11)
    for no in (3, 5, 7, 9):
        F[f"pmm1d_oop_vert_M{no}"] = _s1d(0.0, no, 11, oop=True)
    F["pmm1d_oop_vert_perlayer_M5"] = _s1d(0.0, 5, 11, grids="per-layer",
                                           oop=True)
    F["pmm1d_slant0.35_perlayer_M5"] = _s1d(0.35, 5, 11, grids="per-layer")
    F["pmm1d_slant0.35_perlayer_M9"] = _s1d(0.35, 9, 11, grids="per-layer")
    F["pmm1d_slant0.35_cov_M7"] = _s1d(0.35, 7, 11, fac="covariant")
    F["pmm1d_slant0.35_deg7_M5"] = _s1d(0.35, 5, 7)
    F["pmm1d_slant0.35_deg15_M5"] = _s1d(0.35, 5, 15)
    F["pmm1d_slant0.35_grazing_M7"] = _s1d(0.35, 7, 11, theta=78.0)
    F["pmm1d_slant0.35_densesup_M7"] = _s1d(0.35, 7, 11, nsup=2.6, nsub=2.6)
    F["pmm1d_slant0.35_hicontrast_M7"] = _s1d(0.35, 7, 11, nsub=3.9)
    F["pmm1d_slant0.35_bigperiod_M11"] = _s1d(0.35, 11, 11, per=3.2)

    # -- 1-D conical (vertical, OOP permitted) ---------------------------
    def _con(norders, oop, phi=33.0):
        def go():
            st = PMMStack(1.10 * UM, n_substrate=1.5, degree=11,
                          n_orders=norders)
            if oop:
                st.add_layer(0.22 * UM,
                             segments=[(0.45, _oop()), (0.55, 1.2)])
            else:
                st.add_layer(0.40 * UM, segments=[(0.45, 3.4), (0.55, 1.0)])
            st.set_source(0.62 * UM, angle=np.deg2rad(25.0),
                          phi=np.deg2rad(phi))
            return st.solve()
        return go
    for no in (3, 5, 7):
        F[f"pmm1d_conical_M{no}"] = _con(no, False)
        F[f"pmm1d_conical_oop_M{no}"] = _con(no, True)

    # -- 2-D hybrid ------------------------------------------------------
    def _hyb(slant, norders, degree, oop=False, theta=25.0, per=1.20,
             wl=0.68, nsup=1.0, nsub=1.5, cascade="fast"):
        def go():
            st = PMM2DStackHybrid(per * UM, n_substrate=nsub,
                                  n_superstrate=nsup, degree=degree,
                                  n_orders=norders, cascade=cascade)
            st.add_layer(0.40 * UM,
                         eps_tensor_cell=_tensor_cell(oop=oop),
                         slant=slant)
            st.set_source(wl * UM, theta=np.deg2rad(theta))
            return st.solve()
        return go
    for sl in (0.25, 0.35, 0.50):
        for no in (3, 5, 7, 9):
            F[f"hyb_slant{sl}_M{no}"] = _hyb((sl, 0.0), no, 11)
    F["hyb_slant0.5_M11"] = _hyb((0.5, 0.0), 11, 11)
    for no in (3, 5, 7, 9):
        F[f"hyb_oop_vert_M{no}"] = _hyb(None, no, 11, oop=True)
    F["hyb_oop_vert_M11"] = _hyb(None, 11, 11, oop=True)
    F["hyb_slant0.5_deg7_M5"] = _hyb((0.5, 0.0), 5, 7)
    F["hyb_slant0.5_deg15_M5"] = _hyb((0.5, 0.0), 5, 15)
    F["hyb_slant0.5_tree_M5"] = _hyb((0.5, 0.0), 5, 11, cascade="tree")
    F["hyb_slant0.5_grazing_M7"] = _hyb((0.5, 0.0), 7, 11, theta=72.0)
    F["hyb_slant0.5_densesup_M7"] = _hyb((0.5, 0.0), 7, 11, nsup=2.6,
                                         nsub=2.6)
    F["hyb_slant0.5_hicontrast_M7"] = _hyb((0.5, 0.0), 7, 11, nsub=3.9)
    F["hyb_slant0.5_theta15_M5"] = _hyb((0.5, 0.0), 5, 11, theta=15.0)
    F["hyb_slant0.5_theta22_M5"] = _hyb((0.5, 0.0), 5, 11, theta=22.0)
    F["hyb_slant0.5_theta30_M5"] = _hyb((0.5, 0.0), 5, 11, theta=30.0)
    F["hyb_slanty0.5_M5"] = _hyb((0.0, 0.5), 5, 11)
    F["hyb_slantxy_M5"] = _hyb((0.35, 0.35), 5, 11)

    # -- pmm_jones_2d single-layer entry ---------------------------------
    def _j2d(slant, norders, degree=11, oop=False, theta=25.0):
        def go():
            return pmm_jones_2d(1.20 * UM, 1.20 * UM,
                                _tensor_cell(oop=oop), 1.5, 1.0,
                                0.40 * UM, 0.68 * UM,
                                theta=np.deg2rad(theta), degree=degree,
                                n_orders=norders, slant=slant)
        return go
    for no in (3, 5, 7, 9):
        F[f"j2d_slant0.5_M{no}"] = _j2d((0.5, 0.0), no)
    F["j2d_oop_vert_M5"] = _j2d(None, 5, oop=True)
    F["j2d_oop_vert_M9"] = _j2d(None, 9, oop=True)

    # -- 2-D PURE staggered ---------------------------------------------
    def _pure(slant, nmodes, norders, oop=False, mag=False):
        def go():
            st = PMM2DStackPure(1.20 * UM, n_substrate=1.5, n_modes=nmodes,
                                n_orders=norders)
            cell = np.full((4, 4), 1.0)
            cell[1:3, 1:3] = 3.4
            kw = {}
            if oop:
                tc = np.zeros((4, 4, 3, 3), dtype=complex)
                for i in range(4):
                    for j in range(4):
                        tc[i, j] = np.eye(3) * cell[i, j]
                        tc[i, j, 0, 2] = tc[i, j, 2, 0] = 0.4
                st.add_layer(0.40 * UM, eps_cell=tc, slant=slant, **kw)
            elif mag:
                st.add_layer(0.40 * UM, eps_cell=cell,
                             mu_cell=np.full((4, 4), 1.3))
            else:
                st.add_layer(0.40 * UM, eps_cell=cell, slant=slant)
            st.set_source(0.68 * UM, theta=np.deg2rad(25.0))
            return st.solve()
        return go
    F["pure_slant0.5_nm4_M2"] = _pure((0.5, 0.0), 4, 2)
    F["pure_slant0.5_nm6_M3"] = _pure((0.5, 0.0), 6, 3)
    F["pure_oop_vert_nm4_M2"] = _pure(None, 4, 2, oop=True)
    F["pure_magnetic_nm4_M2"] = _pure(None, 4, 2, mag=True)

    # -- Berreman 4x4 planar cascade -------------------------------------
    def _ber(theta, lossy=False, thick=False):
        def go():
            eps = _oop()
            if lossy:
                eps = eps + 0.35j * np.eye(3)
            d = (2.4 if thick else 0.30) * UM
            return berreman_jones_1d([(eps, d)], 1.5, 1.0, 0.62 * UM,
                                     theta=np.deg2rad(theta))
        return go
    for th in (0.0, 20.0, 45.0, 70.0):
        F[f"berreman_th{th:g}"] = _ber(th)
    F["berreman_lossy"] = _ber(35.0, lossy=True)
    F["berreman_thick"] = _ber(35.0, thick=True)

    # -- pure staggered single-layer entry -------------------------------
    def _stag(slant, nmodes, norders):
        def go():
            cell = np.full((4, 4), 1.0)
            cell[1:3, 1:3] = 3.4
            return pmm_jones_2d_staggered(1.20 * UM, 1.20 * UM, cell, 1.5,
                                          1.0, 0.40 * UM, 0.68 * UM,
                                          theta=np.deg2rad(25.0),
                                          n_modes=nmodes, n_orders=norders,
                                          slant=slant)
        return go
    F["stag_slant0.5_nm4_M2"] = _stag((0.5, 0.0), 4, 2)

    # -- the SCALAR-cell family that carries BOTH populations ------------
    # A HALF-CELL of uniform ground against a high-contrast x profile is
    # what excites the generalized generator's rank deficiency; the tensor
    # cells above never do, which is why the census had to be widened.
    PROF = {"A": [4.0, 4.0, 2.0, 1.0, 1.0, 1.0],
            "B": [9.0, 9.0, 4.0, 1.0, 1.0, 1.0],
            "C": [2.5, 1.0, 2.5, 1.0, 2.5, 1.0]}

    def _scell(prof, ground=1.0, n=6):
        c = np.full((n, n), ground, dtype=complex)
        c[:, :n // 2] = np.asarray(PROF[prof])[:, None] + (ground - 1.0)
        return c

    def _hybs(prof, slant, norders, degree=11, theta=25.0, ground=1.0,
              nsup=1.0, nsub=1.5, wl=0.68):
        def go():
            st = PMM2DStackHybrid(1.20 * UM, 1.20 * UM, n_superstrate=nsup,
                                  n_substrate=nsub, n_orders=norders,
                                  degree=degree)
            st.add_layer(0.50 * UM, eps_cell=_scell(prof, ground),
                         slant=(None if slant == 0 else (slant, 0.0)))
            st.set_source(wl * UM, theta=np.deg2rad(theta))
            return st.solve()
        return go
    for pk in PROF:
        for sl in (0.35, 0.50, 0.70):
            for no in (3, 5, 7):
                F[f"scal{pk}_sl{sl}_M{no}"] = _hybs(pk, sl, no)
    F["scalA_sl0.5_M9"] = _hybs("A", 0.50, 9)
    F["scalA_sl0.5_M11"] = _hybs("A", 0.50, 11)
    F["scalA_sl0.5_deg7_M5"] = _hybs("A", 0.50, 5, degree=7)
    F["scalA_sl0.5_deg15_M5"] = _hybs("A", 0.50, 5, degree=15)
    F["scalA_sl0.25_M5"] = _hybs("A", 0.25, 5)
    F["scalA_sl0.15_M5"] = _hybs("A", 0.15, 5)
    F["scalA_vert_M5"] = _hybs("A", 0.0, 5)
    F["scalA_sl0.5_normal_M5"] = _hybs("A", 0.50, 5, theta=0.0)
    F["scalA_sl0.5_th40_M5"] = _hybs("A", 0.50, 5, theta=40.0)
    F["scalA_sl0.5_th70_M5"] = _hybs("A", 0.50, 5, theta=70.0)
    F["scalA_sl0.5_th80_M7"] = _hybs("A", 0.50, 7, theta=80.0)
    F["scalA_sl0.5_densesup_M5"] = _hybs("A", 0.50, 5, nsup=2.6, nsub=2.6)
    F["scalA_sl0.5_hicontrast_M5"] = _hybs("A", 0.50, 5, nsub=3.9)
    F["scalA_sl0.5_ground2.1_M5"] = _hybs("A", 0.50, 5, ground=2.1)
    # a wavelength placed at a Rayleigh (Wood) cut-off of the substrate
    F["scalA_sl0.5_wood_M7"] = _hybs("A", 0.50, 7,
                                     wl=1.20 * (1.5 + np.sin(
                                         np.deg2rad(25.0))) / 2.0)
    F["scalA_sl0.5_woodx0.997_M7"] = _hybs(
        "A", 0.50, 7,
        wl=1.20 * (1.5 + np.sin(np.deg2rad(25.0))) / 2.0 * 0.997)

    def _j2ds(prof, slant, norders, degree=11):
        def go():
            c = _scell(prof)
            T = np.zeros(c.shape + (3, 3), dtype=complex)
            for i in range(3):
                T[..., i, i] = c
            return pmm_jones_2d(1.20 * UM, 1.20 * UM, T, 1.5, 1.0,
                                0.50 * UM, 0.68 * UM,
                                theta=np.deg2rad(25.0), degree=degree,
                                n_orders=norders,
                                slant=(None if slant == 0 else (slant, 0.0)))
        return go
    for no in (3, 5, 7, 9):
        F[f"j2dscalA_sl0.5_M{no}"] = _j2ds("A", 0.50, no)
    F["j2dscalA_vert_M5"] = _j2ds("A", 0.0, 5)
    return F


# ---------------------------------------------------------------------------


def run(census_armed=True):                                  # noqa: C901
    from lumenairy.elements.rcwa import _core as rc
    rows = {}
    F = fixtures()
    for name, fn in F.items():
        if census_armed:
            rc._INV_CENSUS = []
        t = time.time()
        rec = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                res = fn()
                rec["outcome"] = "SOLVED"
                o, R, T, J = res[0], res[1], res[2], res[3]
                Ra, Ta = np.asarray(R), np.asarray(T)
                if Ra.ndim == 1:
                    rec["sumRT"] = float(np.max(Ra + Ta))
                else:
                    rec["sumRT"] = float(np.max(np.sum(Ra, axis=1)
                                                + np.sum(Ta, axis=1)))
                rec["sha"] = _lib.sha(np.asarray(o), np.asarray(R),
                                      np.asarray(T), np.asarray(J))
            except Exception as exc:                        # noqa: BLE001
                rec["outcome"] = type(exc).__name__
                rec["message"] = str(exc)[:400]
                rec["sumRT"] = None
                m = str(exc)
                for tok in ("sum R + T = ", "total efficiency "):
                    if tok in m:
                        try:
                            rec["sumRT"] = float(
                                m.split(tok)[1].split()[0].rstrip(",;)"))
                        except Exception:                   # noqa: BLE001
                            pass
        rec["secs"] = round(time.time() - t, 2)
        rec["warned_energy"] = any("R + T" in str(w.message)
                                   or "efficiency" in str(w.message)
                                   for w in caught)
        rec["warnings"] = [str(w.message)[:160] for w in caught][:3]
        if census_armed:
            cen = [c for c in rc._INV_CENSUS
                   if "generalized interface" in c[0]]
            rec["n_interfaces"] = len(cen)
            if cen:
                rc_vals = [c[2] for c in cen]
                rs_vals = [c[3] for c in cen if c[3] is not None]
                rec["rcond_min"] = float(np.nanmin(rc_vals))
                rec["rcond_max"] = float(np.nanmax(rc_vals))
                rec["resid_max"] = (float(np.nanmax(rs_vals)) if rs_vals
                                    else None)
                rec["n_refused_rows"] = int(sum(1 for c in cen if c[4]))
                rec["sizes"] = sorted({int(c[1]) for c in cen})
            rc._INV_CENSUS = None
        rows[name] = rec
        print(f"  {name:34s} {rec['outcome']:22s} "
              f"RT={rec.get('sumRT')} rcmin={rec.get('rcond_min')}")
    return rows


def guard_state():
    from lumenairy.elements.rcwa import _core as rc
    return dict(
        has_rcond_refuse_kw=("rcond_refuse" in
                             rc._guarded_inverse.__code__.co_varnames),
        INV_T22_RCOND_REFUSE=getattr(rc, "_INV_T22_RCOND_REFUSE", None),
        INV_RESID_REFUSE=rc._INV_RESID_REFUSE,
        INV_RCOND_SCREEN=rc._INV_RCOND_SCREEN)


def unarmed_control():
    """The screen must be OFF for every caller that does not arm it: an
    UNARMED ``_guarded_inverse`` on a matrix the ARMED one refuses must return
    ``np.linalg.inv`` sha-identically."""
    from lumenairy.elements.rcwa import _core as rc
    rng = np.random.default_rng(20260911)
    n = 24
    U, _ = np.linalg.qr(rng.normal(size=(n, n))
                        + 1j * rng.normal(size=(n, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n))
                        + 1j * rng.normal(size=(n, n)))
    s = np.logspace(0, -20, n)
    A = U @ np.diag(s) @ V.conj().T
    out = {}
    plain = np.linalg.inv(A)
    out["plain_sha"] = _lib.sha(plain)
    try:
        X = rc._guarded_inverse(A, "verify unarmed")
        out["unarmed"] = "RETURNED"
        out["unarmed_sha"] = _lib.sha(X)
        out["unarmed_identical"] = bool(_lib.sha(X) == _lib.sha(plain))
    except Exception as exc:                                # noqa: BLE001
        out["unarmed"] = type(exc).__name__
    try:
        rc._guarded_inverse(A, "verify armed", rcond_refuse=1e-10)
        out["armed"] = "RETURNED"
    except TypeError as exc:
        out["armed"] = f"TypeError: {str(exc)[:120]}"
    except Exception as exc:                                # noqa: BLE001
        out["armed"] = type(exc).__name__
        out["armed_message"] = str(exc)[:500]
    return out


def main():
    t0 = time.time()
    out = dict(guard=guard_state(), unarmed_control=unarmed_control())
    out["census"] = run(census_armed=True)
    out["no_census"] = {k: {kk: vv for kk, vv in v.items()
                            if kk in ("outcome", "sumRT", "sha")}
                        for k, v in run(census_armed=False).items()}
    out["total_secs"] = round(time.time() - t0, 1)
    _lib.save("t4_o2_census", out)


if __name__ == "__main__":
    main()
