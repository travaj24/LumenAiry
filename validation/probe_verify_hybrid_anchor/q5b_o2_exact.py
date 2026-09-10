"""Q5b -- audit item O2 on the EXACT fixture the fix's own probes rejected.

``px = py = 1.20 um``, ``wl = 0.68 um``, a 6x6 cell whose x-profile is
``[4, 4, 2, 1, 1, 1]`` on a ground of ``1.0`` -- so the cell CONTAINS
``eps = 1.0``, the superstrate's own permittivity -- a slanted patterned layer
of ``d = 0.50 um`` at ``slant = (0.5, 0)`` over a ``0.25 um`` ``eps = 3.6``
film, ``n_sub = 1.5``.  (``validation/probe_fix_hybrid_slant_anchor``'s
``p1_census.py`` geometry constants, with ``p4_test_bars.py``'s film.)

Grid: ``n_orders`` 3..13 x four mounts, four stack shapes each, so the
blow-up's dependence on truncation, mount, the SLANT and the SECOND LAYER can
all be read off one table.  The mechanism instrumentation is q5's.
"""
from __future__ import annotations

import math
import time
import warnings

import _lib as L
import numpy as np
from q5_o2_blowup import Probe

WL = 0.68e-6
PX = PY = 1.20e-6
D1 = 0.50e-6
DF = 0.25e-6
FEPS = 3.6
NSUP, NSUB = 1.0, 1.5
TSL = 0.5                      # t d / px = 0.2083
XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])
NXC = 6
MOUNTS = dict(normal=(0.0, 0.0),
              oblique25=(math.radians(25.0), 0.0),
              conical25_40=(math.radians(25.0), math.radians(40.0)),
              oblique40=(math.radians(40.0), 0.0))
ORD_GRID = (3, 5, 7, 9, 11, 13)


def cell(n=NXC):
    c = np.ones((n, n), dtype=complex)
    c[:, 0:n // 2] = np.repeat(XPROF, n // NXC)[:, None]
    return c


def solve(mount, M, kind, probe=None):
    from lumenairy.elements.pmm import stack2d as S2
    st = S2.PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                             n_orders=M)
    sl = (TSL, 0.0) if kind.startswith("slant") else None
    st.add_layer(D1, eps_cell=cell(), slant=sl)
    if kind.endswith("over_film"):
        st.add_layer(DF, eps=FEPS)
    th, ph = MOUNTS[mount]
    st.set_source(WL, theta=th, phi=ph)
    if probe is not None:
        probe.reset()
        probe.install(S2)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            o, R, T, J = st.solve()
        rec = dict(outcome="SOLVED",
                   RT=float(np.max(np.asarray(R).sum(axis=1)
                                   + np.asarray(T).sum(axis=1))),
                   warn=[str(x.message)[:90] for x in w])
    except Exception as e:                              # noqa: BLE001
        rec = dict(outcome="RAISE", exc=type(e).__name__, msg=str(e)[:180])
    finally:
        if probe is not None:
            probe.restore(S2)
    if probe is not None:
        rec["probe"] = probe.summary()
    return rec


def main():
    t0 = time.time()
    res = {"fixture": dict(px=PX, wl=WL, d=D1, df=DF, feps=FEPS, tsl=TSL,
                           walk_um=TSL * D1 * 1e6,
                           walk_over_period=TSL * D1 / PX,
                           cell=cell().real.tolist())}
    grid = {}
    for kind in ("slant_over_film", "slant_only", "vertical_over_film",
                 "vertical_only"):
        for mount in MOUNTS:
            for M in ORD_GRID:
                r = solve(mount, M, kind)
                grid["%s|%s|M%d" % (kind, mount, M)] = dict(
                    outcome=r["outcome"], RT=r.get("RT"),
                    warned=bool(r.get("warn")), exc=r.get("exc"))
    res["grid"] = grid
    bad = {k: v for k, v in grid.items()
           if v["outcome"] == "RAISE" or (v["RT"] or 0.0) > 1.05}
    res["blowups"] = bad
    print("blowups: %d of %d" % (len(bad), len(grid)))
    for k in sorted(bad, key=lambda k: -(bad[k]["RT"] or 0.0)):
        print("   %-40s RT=%-14.6g warned=%s" % (k, bad[k]["RT"] or float("nan"),
                                                 bad[k]["warned"]))
    # silent-wrong check: is any blow-up UNWARNED?
    res["silent_blowups"] = {k: v for k, v in bad.items()
                             if not v["warned"] and v["outcome"] == "SOLVED"}
    print("SILENT (unwarned) blowups:", list(res["silent_blowups"]))

    # mechanism on the worst pair
    if bad:
        key = max(bad, key=lambda k: (bad[k]["RT"] or 0.0))
        kind, mount, M = key.split("|")
        M = int(M[1:])
        pr = Probe()
        rows = {}
        for kk in ("slant_over_film", "slant_only", "vertical_over_film",
                   "vertical_only"):
            rows[kk] = solve(mount, M, kk, probe=pr)
        rows["ladder"] = {("M%d" % m): solve(mount, m, kind)["RT"]
                          for m in ORD_GRID}
        # the near-cut-off order set, in the LAB metric
        th, ph = MOUNTS[mount]
        a0x = math.sin(th) * math.cos(ph)
        a0y = math.sin(th) * math.sin(ph)
        al = []
        for m in range(-M, M + 1):
            for n in range(-M, M + 1):
                al.append((m, n, math.hypot(a0x + m * WL / PX,
                                            a0y + n * WL / PY)))
        for cut, tag in ((1.0, "superstrate"), (1.5, "substrate")):
            near = sorted(al, key=lambda r: abs(r[2] - cut))[:4]
            rows["near_%s_cutoff_%.1f" % (tag, cut)] = [
                dict(m=m, n=n, alpha=round(a, 6), gap=round(a - cut, 6))
                for m, n, a in near]
        # the cell values against the half-space eps
        rows["cell_values"] = sorted({float(v.real)
                                      for v in cell().ravel()})
        rows["eps_sup"] = NSUP ** 2
        rows["eps_sub"] = NSUB ** 2
        # PURE engine on the same solid
        try:
            from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
            pu = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                                n_modes=6, n_orders=5)
            pu.add_layer(D1, eps_cell=cell(), slant=(TSL, 0.0))
            pu.add_layer(DF, eps=FEPS)
            pu.set_source(WL, theta=th, phi=ph)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                po = pu.solve()
            rows["pure_same_solid"] = dict(
                outcome="SOLVED",
                RT=float(np.max(np.asarray(po[1]).sum(axis=1)
                                + np.asarray(po[2]).sum(axis=1))),
                warn=[str(x.message)[:90] for x in w])
        except Exception as e:                          # noqa: BLE001
            rows["pure_same_solid"] = dict(outcome="RAISE",
                                           exc=type(e).__name__,
                                           msg=str(e)[:200])
        res["mechanism"] = dict(worst=key, rows=rows)
        print("-- mechanism at", key)
        for k, v in rows.items():
            print("   ", k, v)
    res["seconds"] = round(time.time() - t0, 1)
    L.dump("q5b_o2_exact", res)


if __name__ == "__main__":
    main()
