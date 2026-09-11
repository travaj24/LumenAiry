"""E-CASCADE -- class A (branch cut) + class B (interface conditioning) probes
for Berreman, RCWAStack, PMMStack, PMM2DStackHybrid, PMM2DStackPure and the
coatings characteristic-matrix chain.  MEASUREMENT ONLY.

CLASS-A FIXTURE (the RCWA killer shape, ported to each engine): a uniform layer
whose permittivity EXACTLY equals a neighbouring REGION's (and, where the
builder allows, an equal-eps SPACER as well), plus 1e-6 / -1e-6 relative detune
arms.  On a provably lossless stack the oracle is ``sum R + sum T = 1`` per
incident polarization, which needs no reference solve; the spread across
``OPENBLAS_NUM_THREADS`` and across the WIN / WSL builds is the determinism
instrument.

CLASS-B INSTRUMENT: ``lumenairy.elements.rcwa._core._INV_CENSUS`` (armed:
``(site, n, rcond_eq, resid_eq, refused)``) and
``lumenairy.elements.pmm._core._MORTAR_SOLVE_CENSUS``
(``(site, n, rcond, refused, residual)``).

Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=n MKL_NUM_THREADS=1 \
       PYTHONPATH=. python validation/probe_scope_bor_guards/e_cascade.py out.json
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e_lib as E  # noqa: E402

WL = 0.6e-6
PX = 0.5e-6
DEPTH = 0.2e-6
NS = 1.5                 # n_superstrate == n_substrate; eps == 2.25
HOST = 2.25              # the COINCIDENCE value
STRONG = 6.0


class Census:
    """Arm both census hooks around a solve and summarise the population."""

    def __enter__(self):
        from lumenairy.elements.pmm import _core as pc
        from lumenairy.elements.rcwa import _core as rc
        self.rc, self.pc = rc, pc
        self.old_i, self.old_m = rc._INV_CENSUS, pc._MORTAR_SOLVE_CENSUS
        rc._INV_CENSUS = []
        pc._MORTAR_SOLVE_CENSUS = []
        return self

    def __exit__(self, *a):
        self.inv = list(self.rc._INV_CENSUS or [])
        self.mor = list(self.pc._MORTAR_SOLVE_CENSUS or [])
        self.rc._INV_CENSUS, self.pc._MORTAR_SOLVE_CENSUS = (self.old_i,
                                                             self.old_m)
        return False

    def summary(self):
        def _agg(rows):
            by = {}
            for r in rows:
                d = by.setdefault(str(r[0]), dict(n_calls=0, widths=set(),
                                                  rconds=[], resids=[],
                                                  refused=0))
                d["n_calls"] += 1
                d["widths"].add(int(r[1]))
                if r[2] is not None and np.isfinite(r[2]):
                    d["rconds"].append(float(r[2]))
                if isinstance(r[3], (int, float)) and np.isfinite(r[3]):
                    d["resids"].append(float(r[3]))
                if r[4]:
                    d["refused"] += 1
            return {s: dict(n_calls=d["n_calls"], widths=sorted(d["widths"]),
                            refused=d["refused"],
                            rcond_min=min(d["rconds"]) if d["rconds"] else None,
                            rcond_max=max(d["rconds"]) if d["rconds"] else None,
                            resid_min=min(d["resids"]) if d["resids"] else None,
                            resid_max=max(d["resids"]) if d["resids"] else None)
                    for s, d in by.items()}
        mor = [(r[0], r[1], r[2], r[4], r[3]) for r in self.mor]
        return dict(inverse=_agg(self.inv), mortar=_agg(mor))


def _run(fn):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with Census() as c:
            try:
                res, err = fn(), None
            except Exception as exc:
                res, err = None, "%s: %s" % (type(exc).__name__,
                                             str(exc).replace("\n", " ")[:180])
        cen = c.summary()
    return res, err, cen, sorted({type(x.message).__name__ for x in w})


def _cl_rt(res):
    v = E.rt_per_pol(res)
    return dict(rt=[float(x) for x in v],
                closure=float(np.max(np.abs(v - 1.0))))


def _row(name, fn):
    res, err, cen, warns = _run(fn)
    d = dict(name=name, raised=err, warnings=warns, census=cen)
    if res is not None:
        d.update(_cl_rt(res))
    return d


# =========================================================== 1. BERREMAN 4x4
def berreman_rows():
    from lumenairy.elements.berreman import BerremanStack, berreman_jones_1d
    rows = []

    def planar(detune, theta, spacer=True):
        e = HOST * (1.0 + detune)
        lay = (([(e, 0.12e-6)] if spacer else []) + [(STRONG, 0.20e-6)]
               + ([(e, 0.12e-6)] if spacer else []))
        R, T, _Jr, _Jt = berreman_jones_1d(lay, NS, NS, WL, theta=theta)
        # (2,) per INCIDENT polarization -> (2, 1) so each row closes on its own
        return (None, np.asarray(R)[:, None], np.asarray(T)[:, None])

    for det in (0.0, 1e-6, -1e-6):
        for th in (0.0, 0.35):
            rows.append(_row("berreman_planar_det%.0e_th%.2f" % (det, th),
                             lambda d=det, t=th: planar(d, t)))
    rows.append(_row("berreman_planar_nospacer",
                     lambda: planar(0.0, 0.35, spacer=False)))

    def offplane(detune, theta):
        e = HOST * (1.0 + detune)
        t = np.diag([STRONG, STRONG, STRONG]).astype(complex)
        t[0, 2] = t[2, 0] = 0.6                       # exz -> out-of-plane
        lay = [(e * np.eye(3, dtype=complex), 0.12e-6), (t, 0.20e-6),
               (e * np.eye(3, dtype=complex), 0.12e-6)]
        R, T, _Jr, _Jt = berreman_jones_1d(lay, NS, NS, WL, theta=theta,
                                           phi=0.3)
        return (None, np.asarray(R)[:, None], np.asarray(T)[:, None])

    for det in (0.0, 1e-6):
        rows.append(_row("berreman_offplane_det%.0e" % det,
                         lambda d=det: offplane(d, 0.35)))

    def stack(detune):
        s = BerremanStack(n_superstrate=NS, n_substrate=NS)
        s.add_layer(0.12e-6, eps=HOST * (1.0 + detune))
        s.add_layer(0.20e-6, eps=STRONG)
        s.add_layer(0.12e-6, eps=HOST * (1.0 + detune))
        s.set_source(WL, theta=0.35)
        r = s.solve()                       # (R, T, jones) tuple
        return (None, np.asarray(r[0])[:, None], np.asarray(r[1])[:, None])

    for det in (0.0, 1e-6):
        rows.append(_row("berremanstack_det%.0e" % det, lambda d=det: stack(d)))
    return rows


# ============================================================ 2. RCWAStack
def rcwastack_rows():
    from lumenairy.elements.rcwa import RCWAStack
    rows = []

    def st(detune, spacer, n_orders=9, theta=0.0):
        e = HOST * (1.0 + detune)
        s = RCWAStack(PX, n_superstrate=NS, n_substrate=NS, n_orders=n_orders)
        if spacer:
            s.add_layer(0.12e-6, eps=e)
        cell = np.full(64, e, dtype=complex)
        cell[16:48] = e * (1.0 + 1e-6)      # WEAK modulation: no trunc. floor
        s.add_layer(DEPTH, eps_cell=cell)
        if spacer:
            s.add_layer(0.12e-6, eps=e)
        s.set_source(WL, theta=theta)
        o, R, T = s.solve().efficiencies()
        return (o, np.atleast_2d(R), np.atleast_2d(T))

    for det in (0.0, 1e-6):
        for sp in (True, False):
            rows.append(_row("rcwastack_det%.0e_sp%d" % (det, sp),
                             lambda d=det, s=sp: st(d, s)))
    rows.append(_row("rcwastack_oblique", lambda: st(0.0, True, theta=0.3)))
    return rows


# ============================================================ 3. PMMStack 1-D
def pmmstack_rows():
    from lumenairy import PMMStack
    rows = []

    def st(detune, spacer, degree=9, theta=0.0):
        e = HOST * (1.0 + detune)
        s = PMMStack(PX, n_superstrate=NS, n_substrate=NS, degree=degree)
        if spacer:
            s.add_layer(0.12e-6, segments=[(1.0, e)])
        s.add_layer(DEPTH, segments=[(0.3, e), (0.4, e * (1.0 + 1e-6)),
                                     (0.3, e)])
        if spacer:
            s.add_layer(0.12e-6, segments=[(1.0, e)])
        s.set_source(WL, theta=theta)
        o, R, T = s.solve()[:3]
        return (o, np.atleast_2d(R), np.atleast_2d(T))

    for det in (0.0, 1e-6):
        for sp in (True, False):
            rows.append(_row("pmmstack_det%.0e_sp%d" % (det, sp),
                             lambda d=det, s=sp: st(d, s)))
    rows.append(_row("pmmstack_oblique", lambda: st(0.0, True, theta=0.3)))
    return rows


# ====================================================== 4. PMM2DStackHybrid
def hybrid_rows():
    from lumenairy.elements.pmm import PMM2DStackHybrid
    rows = []

    def cell(host, pillar, S=32):
        e = np.full((S, S), host, dtype=complex)
        x = (np.arange(S) + 0.5) / S - 0.5
        m = (np.abs(x[:, None]) < 0.25) & (np.abs(x[None, :]) < 0.25)
        e[m] = pillar
        return e

    def st(detune, spacer, theta=0.0):
        e = HOST * (1.0 + detune)
        s = PMM2DStackHybrid(PX, PX, n_substrate=NS, n_superstrate=NS,
                             degree=7, n_orders=4, symmetry=False)
        if spacer:
            s.add_layer(0.1e-6, eps=e)
        s.add_layer(DEPTH, eps_cell=cell(e, e * (1.0 + 1e-6)))
        if spacer:
            s.add_layer(0.1e-6, eps=e)
        r = s.set_source(WL, theta=theta).solve()
        return (r[0], np.atleast_2d(r[1]), np.atleast_2d(r[2]))

    for det in (0.0, 1e-6):
        for sp in (True, False):
            rows.append(_row("hybrid_det%.0e_sp%d" % (det, sp),
                             lambda d=det, s=sp: st(d, s)))
    rows.append(_row("hybrid_oblique", lambda: st(0.0, True, theta=0.25)))
    return rows


# ======================================================== 5. PMM2DStackPure
def pure_rows():
    from lumenairy.elements.pmm import PMM2DStackPure
    rows = []

    def stagcell(host, pillar):
        e = np.full((4, 4), host, dtype=complex)
        e[1:3, 1:3] = pillar
        return e

    def st(detune, spacer, theta=0.0):
        e = HOST * (1.0 + detune)
        s = PMM2DStackPure(PX, PX, n_substrate=NS, n_superstrate=NS,
                           degree=6, n_orders=4)
        if spacer:
            s.add_layer(0.1e-6, eps=e)
        s.add_layer(DEPTH, eps_cell=stagcell(e, e * (1.0 + 1e-6)))
        if spacer:
            s.add_layer(0.1e-6, eps=e)
        r = s.set_source(WL, theta=theta).solve()
        return (r[0], np.atleast_2d(r[1]), np.atleast_2d(r[2]))

    for det in (0.0, 1e-6):
        for sp in (True, False):
            rows.append(_row("pure_det%.0e_sp%d" % (det, sp),
                             lambda d=det, s=sp: st(d, s)))
    rows.append(_row("pure_oblique", lambda: st(0.0, True, theta=0.25)))
    return rows


# ================================================= 6. coatings 2x2 Abeles chain
def coatings_rows():
    """``coating_reflectance`` cascades 2x2 CHARACTERISTIC matrices -- a
    TRANSFER-matrix chain, not an S-matrix -- with NO explicit inverse anywhere
    (class B unreachable) and NO grid (class C unreachable).  Its branch site is
    the local ``_cos_theta``: ``ct = sqrt(1 - (n0 sin0 / n)^2 + 0j)`` pinned by
    the EXACT ``(n_layer * ct).imag < 0.0`` and then FLOORED by the ABSOLUTE
    ``abs(ct) < 1e-12 -> 1e-12``.  Probed at, and either side of, the exact
    critical angle where ``ct == 0``."""
    from lumenairy.elements.coatings import coating_reflectance
    rows = []
    wl = np.array([0.6e-6])
    n_amb, n_lay = 1.5, 1.0
    th_c = float(np.arcsin(n_lay / n_amb))          # exact critical angle
    for tag, th in (("below", th_c * (1 - 1e-3)),
                    ("crit_1em9", th_c * (1 - 1e-9)),
                    ("crit_exact", th_c), ("above", th_c * (1 + 1e-3))):
        for d in (0.05e-6, 1.0e-6):
            try:
                out = coating_reflectance([(n_lay, d)], wl, n_substrate=1.5,
                                          n_ambient=n_amb, angle=th,
                                          polarization="both")
                R = float(np.asarray(out[0]).ravel()[0])
                T = float(np.asarray(out[1]).ravel()[0])
                rows.append(dict(name="coating_%s_d%.0e" % (tag, d), angle=th,
                                 R=R, T=T, RplusT=R + T))
            except Exception as exc:
                rows.append(dict(name="coating_%s_d%.0e" % (tag, d), angle=th,
                                 raised=repr(exc)[:160]))
    return rows


def main():
    E.pin_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "e_cascade.json"
    res = {}
    for key, fn in (("berreman", berreman_rows),
                    ("rcwastack", rcwastack_rows),
                    ("pmmstack", pmmstack_rows),
                    ("hybrid", hybrid_rows),
                    ("pure", pure_rows),
                    ("coatings", coatings_rows)):
        try:
            res[key] = fn()
        except Exception as exc:
            res[key] = dict(fatal=repr(exc)[:300])
        print("done", key, flush=True)
    E.dump(out, res)


if __name__ == "__main__":
    main()
