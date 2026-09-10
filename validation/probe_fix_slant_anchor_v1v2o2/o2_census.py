"""O2 -- the population census a ``T22`` refusal threshold must be derived from.

:func:`_interface_smatrix_general` inverts ``T22 = (inv(Mb) Ma)[2N:, 2N:]``
EXPLICITLY through :func:`_guarded_inverse`, whose screen is DORMANT by default
(``_INV_CENSUS = None``), so nothing screened ``cond(T22) = 4.5e+15`` on the
blow-ups the verification found.  This probe instruments EVERY consumer of the
generalized interface (the binding is imported by name into seven modules, so
each module's binding is patched) and records, per interface:

  * ``cond(T22)``      -- the SVD condition number the verification reported;
  * ``rcond_eq(T22)``  -- ``_rcond_1_equilibrated(T22, inv(T22))``, the FREE
    screen already in the library (two O(n^2) reductions, no factorisation);
  * ``resid_eq(T22)``  -- ``_equilibrated_inverse_residual(T22)``, the
    confirming instrument (one extra inverse).

Two populations: the BLOWN-UP class (slanted hybrid, shear >= 0.35, n_orders
3/5/7, degree >= 9) and the HEALTHY one (every other generalized-cascade
fixture the library ships: OOP hybrid, OOP pure, slanted pure, magnetic pure,
Berreman, the 1-D convection slant, the native-conical PMMStack cascade and the
``pmm_jones_2d`` single-layer entry).  The threshold is whatever the gap
between them supports, on BOTH builds.
"""
from __future__ import annotations

import math
import sys
import time
import warnings

import _lib as L
import numpy as np

# ---- the verification's EXACT blow-up fixture (S6 of the verify doc) -------
WL = 0.68e-6
PX = PY = 1.20e-6
D1, DF, FEPS = 0.50e-6, 0.25e-6, 3.6
NSUP, NSUB = 1.0, 1.5
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
MOUNTS = dict(normal=(0.0, 0.0),
              oblique25=(math.radians(25.0), 0.0),
              conical25_40=(math.radians(25.0), math.radians(40.0)),
              oblique40=(math.radians(40.0), 0.0))

_MODULES = ("lumenairy.elements.pmm.stack",
            "lumenairy.elements.pmm.stack2d",
            "lumenairy.elements.pmm.stack2d_pure",
            "lumenairy.elements.pmm.twod_jones",
            "lumenairy.elements.pmm.conical",
            "lumenairy.elements.pmm._core",
            "lumenairy.elements.berreman")


def cell(n=6, ground=1.0):
    """The verification's 6 x 6 blow-up cell.  ``ground`` is the value the
    x-profile sits on: ``1.0`` is the verification's own fixture (whose cell
    therefore contains the SUPERSTRATE's permittivity); a BENIGN ``2.1``
    removes that coincidence and is used for the healthy population."""
    c = np.full((n, n), ground, dtype=complex)
    c[:, 0:n // 2] = np.repeat(XPROF, n // 6)[:, None] + (ground - 1.0)
    return c


class T22Probe:
    """Patches every module-level binding of ``_interface_smatrix_general``
    and scores the ``T22`` block each call inverts."""

    def __init__(self):
        self.rows = []
        self._saved = {}

    def reset(self):
        self.rows = []

    def install(self):
        import importlib

        from lumenairy.elements.rcwa._core import (
            _equilibrated_inverse_residual,
            _interface_smatrix_general,
            _rcond_1_equilibrated,
        )

        def ifc(Ma, Mb):
            try:
                A = np.asarray(Ma)
                B = np.asarray(Mb)
                n2 = A.shape[0] // 2
                T = np.linalg.solve(B, A)
                T22 = T[n2:, n2:]
                X = np.linalg.inv(T22)
                self.rows.append(dict(
                    n=int(T22.shape[0]),
                    cond=float(np.linalg.cond(T22)),
                    rcond_eq=float(_rcond_1_equilibrated(T22, X)),
                    resid_eq=float(_equilibrated_inverse_residual(T22))))
            except Exception as e:                        # noqa: BLE001
                self.rows.append(dict(n=-1, cond=float("nan"),
                                      rcond_eq=float("nan"),
                                      resid_eq=float("nan"),
                                      exc=type(e).__name__))
            return _interface_smatrix_general(Ma, Mb)

        for name in _MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "_interface_smatrix_general"):
                self._saved[name] = mod._interface_smatrix_general
                mod._interface_smatrix_general = ifc

    def restore(self):
        import importlib
        for name, fn in self._saved.items():
            importlib.import_module(name)._interface_smatrix_general = fn
        self._saved = {}

    def summary(self):
        if not self.rows:
            return dict(n_interfaces=0)
        return dict(n_interfaces=len(self.rows),
                    max_cond=max(r["cond"] for r in self.rows),
                    min_rcond_eq=min(r["rcond_eq"] for r in self.rows),
                    max_resid_eq=max(r["resid_eq"] for r in self.rows),
                    sizes=sorted({r["n"] for r in self.rows}),
                    rows=[dict(r) for r in self.rows])


PR = T22Probe()


def scored(fn):
    """Run ``fn`` with the probe installed; return the record."""
    PR.reset()
    PR.install()
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out = fn()
        rec = dict(outcome="SOLVED", RT=out,
                   warned=sorted({str(x.message)[:60] for x in w}))
    except Exception as e:                                # noqa: BLE001
        rec = dict(outcome="RAISE", exc=type(e).__name__, msg=str(e)[:220])
    finally:
        PR.restore()
    rec["T22"] = PR.summary()
    return rec


#: The last SOLVE's raw 4-tuple, so ``o2_identity`` can hash the same returns
#: this module reduces to a number.  Set by :func:`_rt`, which every fixture
#: funnels through.
LAST = {}


def _rt(res):
    LAST["res"] = res
    R, T = np.asarray(res[1]), np.asarray(res[2])
    return float(np.max(R.sum(axis=-1) + T.sum(axis=-1)))


# ------------------------------------------------------------------ fixtures
def _oop_tensor(c, off=0.35):
    T = np.zeros(np.shape(c) + (3, 3), dtype=complex)
    for i in range(3):
        T[..., i, i] = c
    T[..., 0, 2] = T[..., 2, 0] = off * (np.asarray(c) - 1.0)
    return T


def hybrid(*, mount="oblique25", M=5, slant=0.5, degree=11, film=False,
           oop=False, ground=1.0, cascade=None):
    def go():
        from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
        kw = {} if cascade is None else dict(cascade=cascade)
        st = PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                              n_orders=M, degree=degree, **kw)
        sl = None if slant == 0 else (slant, 0.0)
        c = cell(ground=ground)
        if oop:
            st.add_layer(D1, eps_tensor_cell=_oop_tensor(c), slant=sl)
        else:
            st.add_layer(D1, eps_cell=c, slant=sl)
        if film:
            st.add_layer(DF, eps=FEPS)
        th, ph = MOUNTS[mount]
        st.set_source(WL, theta=th, phi=ph)
        return _rt(st.solve())
    return go


def pure(*, mount="oblique25", M=3, slant=0.5, n_modes=4, magnetic=False,
         oop=False, ground=1.0):
    def go():
        from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
        st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_modes=n_modes, n_orders=M)
        c = cell(ground=ground)
        kw = dict(eps_cell=(_oop_tensor(c) if oop else c))
        if magnetic:
            kw["mu_cell"] = 1.0 + 0.25 * (c.real - 1.0)
        st.add_layer(D1, slant=(None if slant == 0 else (slant, 0.0)), **kw)
        th, ph = MOUNTS[mount]
        st.set_source(WL, theta=th, phi=ph)
        return _rt(st.solve())
    return go


def oned(*, shear=0.30, theta=25.0, M=7, degree=12, oop=False):
    def go():
        from lumenairy.elements.pmm.stack import PMMStack
        st = PMMStack(0.80e-6, n_superstrate=1.0, n_substrate=1.6,
                      degree=degree, n_orders=M, factorization="convection")
        if oop:
            e = np.eye(3, dtype=complex) * 4.2
            e[0, 2] = e[2, 0] = 0.8
            st.add_layer(0.40e-6, segments=[(0.45, e),
                                            (0.55, np.eye(3) * 1.45)])
        else:
            st.add_sheared_grating(0.40e-6, eps_ridge=4.2, eps_groove=1.45,
                                   duty=0.45, shear=shear)
        st.set_source(0.55e-6, theta=math.radians(theta))
        return _rt(st.solve())
    return go


def conical():
    def go():
        from lumenairy.elements.pmm.stack import PMMStack
        st = PMMStack(0.80e-6, n_superstrate=1.0, n_substrate=1.6,
                      degree=12, n_orders=7)
        st.add_layer(0.40e-6, segments=[(0.45, 4.2), (0.55, 1.45)])
        st.set_source(0.55e-6, theta=math.radians(25.0),
                      phi=math.radians(35.0))
        return _rt(st.solve())
    return go


def berreman(*, theta=30.0, phi=20.0, wl=0.55e-6, thick=0.30e-6, lossy=False):
    def go():
        from lumenairy.elements.berreman import berreman_jones_1d
        e = np.eye(3, dtype=complex) * 2.9
        e[0, 2] = e[2, 0] = 0.6
        e[0, 1] = e[1, 0] = 0.3
        if lossy:
            e = e + 0.35j * np.eye(3)
        o = berreman_jones_1d([(e, thick), (np.eye(3) * 2.1, 0.20e-6)],
                              1.5, 1.0, wl, theta=math.radians(theta),
                              phi=math.radians(phi))
        # ``berreman_jones_1d`` returns (R, T, jones_r, jones_t) -- R and T are
        # already the FLUX-NORMALIZED power arrays (R + T = 1 for a lossless
        # stack), so they go straight into ``_rt``.  (Squaring them, as an
        # earlier version of this probe did, produces a number that is not a
        # closure at all and mis-bins these rows.)
        # ``_rt`` sums the LAST axis and maxes over the first, so R and T go in
        # as (2, 1) COLUMNS -- one incident state per row, exactly the (2, N)
        # shape the grating engines hand it.
        return _rt((np.zeros(1, dtype=int),
                    np.asarray(o[0], dtype=float).reshape(2, 1),
                    np.asarray(o[1], dtype=float).reshape(2, 1),
                    np.asarray(o[2])))
    return go


def conical_oop():
    def go():
        from lumenairy.elements.pmm.stack import PMMStack
        e = np.eye(3, dtype=complex) * 4.2
        e[0, 2] = e[2, 0] = 0.8
        st = PMMStack(0.80e-6, n_superstrate=1.0, n_substrate=1.6,
                      degree=12, n_orders=7)
        st.add_layer(0.40e-6, segments=[(0.45, e), (0.55, np.eye(3) * 1.45)])
        st.set_source(0.55e-6, theta=math.radians(25.0),
                      phi=math.radians(35.0))
        return _rt(st.solve())
    return go


def jones2d(*, slant=0.5, oop=False, M=3, degree=9, ground=1.0):
    def go():
        from lumenairy.elements.pmm.twod_jones import pmm_jones_2d
        c = cell(ground=ground)
        T = _oop_tensor(c, off=(0.35 if oop else 0.0))
        o = pmm_jones_2d(PX, PY, T, NSUB, NSUP, D1, WL,
                         theta=math.radians(25.0), phi=0.0, n_orders=M,
                         degree=degree,
                         slant=(None if slant == 0 else (slant, 0)))
        return _rt(o)
    return go


def build_cases():
    cases = {}
    # --- the BLOW-UP class and its cure ladder ------------------------------
    for M in (3, 5, 7, 9, 11):
        cases["hyb_sl0.50_dg11_ob25_M%d" % M] = hybrid(M=M)
        cases["hyb_sl0.50_dg11_ob25_film_M%d" % M] = hybrid(M=M, film=True)
    for sl in (0.0, 0.10, 0.25, 0.35, 1.00):
        cases["hyb_sl%.2f_dg11_ob25_M5" % sl] = hybrid(M=5, slant=sl)
    for dg in (7, 9, 13, 15):
        cases["hyb_sl0.50_dg%d_ob25_M5" % dg] = hybrid(M=5, degree=dg)
    for th in (15, 20, 22, 30, 35):
        cases["hyb_sl0.50_dg11_th%d_M5" % th] = hybrid(M=5, mount="th%d" % th)
    for mt in ("normal", "conical25_40", "oblique40"):
        cases["hyb_sl0.50_dg11_%s_M5" % mt] = hybrid(M=5, mount=mt)
    # --- HEALTHY generalized cascades elsewhere in the library --------------
    for M in (3, 5, 7, 9):
        cases["hyb_OOP_vert_ob25_M%d" % M] = hybrid(M=M, slant=0.0, oop=True)
        cases["hyb_OOP_vert_benign_M%d" % M] = hybrid(M=M, slant=0.0,
                                                      oop=True, ground=2.1)
        cases["hyb_sl0.50_benign_M%d" % M] = hybrid(M=M, ground=2.1)
    cases["hyb_OOP_sl0.50_ob25_M9"] = hybrid(M=9, slant=0.5, oop=True)
    cases["hyb_sl0.50_benign_tree_M9"] = hybrid(M=9, ground=2.1, film=True,
                                                cascade="tree")
    cases["hyb_sl0.50_benign_fused_M9"] = hybrid(M=9, ground=2.1, film=True,
                                                 cascade="fused")
    for M in (3, 5):
        cases["pure_sl0.50_ob25_M%d" % M] = pure(M=M)
        cases["pure_sl0.50_benign_M%d" % M] = pure(M=M, ground=2.1)
        cases["pure_OOP_vert_ob25_M%d" % M] = pure(M=M, slant=0.0, oop=True)
        cases["pure_OOP_vert_benign_M%d" % M] = pure(M=M, slant=0.0, oop=True,
                                                     ground=2.1)
    cases["pure_mag_vert_ob25_M3"] = pure(slant=0.0, magnetic=True)
    cases["pure_mag_OOP_ob25_M3"] = pure(slant=0.0, magnetic=True, oop=True)
    cases["pure_mag_sl0.50_ob25_M3"] = pure(magnetic=True)
    for sh in (0.15, 0.30, 0.60, 1.00):
        cases["oned_shear%.2f_ob25" % sh] = oned(shear=sh)
    for th in (0.0, 40.0, 60.0):
        cases["oned_shear0.30_th%.0f" % th] = oned(theta=th)
    cases["oned_OOP_vert_ob25"] = oned(oop=True)
    cases["pmmstack_conical"] = conical()
    cases["pmmstack_conical_OOP"] = conical_oop()
    for th in (0.0, 30.0, 60.0, 80.0):
        cases["berreman_oop_th%.0f" % th] = berreman(theta=th)
    cases["berreman_oop_lossy"] = berreman(lossy=True)
    cases["berreman_oop_thick"] = berreman(thick=3.0e-6)
    cases["jones2d_sl0.50_ob25"] = jones2d()
    for M in (3, 5, 7):
        cases["jones2d_OOP_vert_M%d" % M] = jones2d(slant=0.0, oop=True, M=M)
        cases["jones2d_OOP_benign_M%d" % M] = jones2d(slant=0.0, oop=True,
                                                      M=M, ground=2.1)
        cases["jones2d_sl0.50_benign_M%d" % M] = jones2d(M=M, ground=2.1)
    return cases


def main():
    t0 = time.time()
    for th in (15, 20, 22, 30, 35):
        MOUNTS["th%d" % th] = (math.radians(float(th)), 0.0)
    res = {}
    for name, fn in build_cases().items():
        rec = scored(fn)
        res[name] = rec
        s = rec["T22"]
        print("%-34s %-7s RT=%-12.6g nifc=%-3s cond=%-11.4g rcond=%-11.4g "
              "resid=%-11.4g" % (
                  name, rec["outcome"], rec.get("RT", float("nan")),
                  s.get("n_interfaces"), s.get("max_cond", float("nan")),
                  s.get("min_rcond_eq", float("nan")),
                  s.get("max_resid_eq", float("nan"))))
    # ---- classify BY MEASUREMENT, then read the gap off the two populations
    broken, healthy = [], []
    for name, rec in res.items():
        rows = rec.get("T22", {}).get("rows") or []
        if not rows:
            continue
        blown = (rec["outcome"] == "RAISE"
                 and "_EnergyError" in rec.get("exc", "")) or (
            rec["outcome"] == "SOLVED" and rec.get("RT", 0.0) > 1.10)
        (broken if blown else healthy).append((name, rows))

    def band(pop):
        rc = [r["rcond_eq"] for _n, rs in pop for r in rs]
        rd = [r["resid_eq"] for _n, rs in pop for r in rs]
        cd = [r["cond"] for _n, rs in pop for r in rs]
        return dict(n_solves=len(pop), n_interfaces=len(rc),
                    rcond_min=min(rc), rcond_max=max(rc),
                    resid_min=min(rd), resid_max=max(rd),
                    cond_min=min(cd), cond_max=max(cd))

    # A broken SOLVE is identified by its WORST interface; a healthy solve must
    # have EVERY interface above the bar.
    worst_broken = sorted((min(r["rcond_eq"] for r in rs), n)
                          for n, rs in broken)
    worst_healthy = sorted((min(r["rcond_eq"] for r in rs), n)
                           for n, rs in healthy)
    res["_populations"] = dict(
        broken=band(broken) if broken else None,
        healthy=band(healthy) if healthy else None,
        broken_worst_interface_rcond=worst_broken[:6],
        healthy_worst_interface_rcond=worst_healthy[:6],
        broken_names=sorted(n for n, _r in broken),
        healthy_names=sorted(n for n, _r in healthy))
    print("\n== POPULATIONS (per interface) ==")
    print("broken :", res["_populations"]["broken"])
    print("healthy:", res["_populations"]["healthy"])
    print("worst healthy solves:", worst_healthy[:6])
    print("best broken solves  :", sorted(worst_broken)[-6:])
    res["_seconds"] = round(time.time() - t0, 1)
    L.dump("o2_census", res, suffix=(sys.argv[1] if len(sys.argv) > 1 else ""))


if __name__ == "__main__":
    main()
