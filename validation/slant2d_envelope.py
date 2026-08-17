"""Phase B -- cross-build envelope for every numeric bar, plus the economics.

Run on BOTH builds; the bars written into
``tests/unit/test_pmm2d_slant_metric.py`` are derived from the WORSE of the two
readings, per docs/TESTING_STANDARDS.md rule 5 ("bars need a gap on both
sides", with the measurements and date recorded).

    PYTHONPATH=<repo> python validation/slant2d_envelope.py
"""
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm import pmm_jones_1d, pmm_jones_1d_slanted, pmm_jones_2d
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid

warnings.simplefilter("ignore")

P, WL, D, DUTY = 500e-9, 633e-9, 250e-9, 0.5
ER, EG = 2.25 + 0j, 1.0 + 0j
NSUB, NSUP = 1.5, 1.0
DEG = 11


def grating(nx=240, duty=DUTY, er=ER, eg=EG, shift=0):
    line = np.full(nx, eg, dtype=complex)
    line[:int(round(duty * nx))] = er
    line = np.roll(line, int(shift))
    c = np.zeros((nx, 1, 3, 3), dtype=complex)
    for i in range(3):
        c[:, :, i, i] = line[:, None]
    return c


def pillar(nx=120, er=ER, eg=EG, sx=0, sy=0, fx=(0.25, 0.60), fy=(0.30, 0.65)):
    g = np.full((nx, nx), eg, dtype=complex)
    g[int(fx[0] * nx):int(fx[1] * nx), int(fy[0] * nx):int(fy[1] * nx)] = er
    g = np.roll(np.roll(g, int(sx), axis=0), int(sy), axis=1)
    c = np.zeros((nx, nx, 3, 3), dtype=complex)
    for i in range(3):
        c[:, :, i, i] = g
    return c


def solve(cell, slant=None, theta=0.0, phi=0.0, n_orders=5):
    return pmm_jones_2d(P, P, cell, NSUB, NSUP, D, WL, theta=theta, phi=phi,
                        degree=DEG, n_orders=n_orders, slant=slant,
                        formulation="laurent")


def dmax(a, b):
    return max(float(np.max(np.abs(np.asarray(a[1]) - np.asarray(b[1])))),
               float(np.max(np.abs(np.asarray(a[2]) - np.asarray(b[2])))))


def cmp_1d(res2, res1):
    o2 = np.asarray(res2[0])
    keep = o2[:, 1] == 0
    m2 = o2[keep][:, 0]
    idx = np.argsort(m2)
    m2 = m2[idx]
    R2 = np.asarray(res2[1])[:, keep][:, idx]
    T2 = np.asarray(res2[2])[:, keep][:, idx]
    o1, R1, T1 = (np.asarray(res1[0]), np.asarray(res1[1]),
                  np.asarray(res1[2]))
    common = np.intersect1d(m2, o1)
    i2, i1 = np.searchsorted(m2, common), np.searchsorted(o1, common)
    d = 0.0
    for r in (0, 1):
        d = max(d, float(np.max(np.abs(R2[r][i2] - R1[r][i1]))))
        d = max(d, float(np.max(np.abs(T2[r][i2] - T1[r][i1]))))
    return d


def envelope():
    print("== BAR ENVELOPES ==")
    worst_uni = worst_inv = 0.0
    for theta in (0.0, 0.20):
        uni = grating(er=ER, eg=ER)
        b = solve(uni, theta=theta)
        for tdeg in (5.0, 20.0, 40.0):
            t = float(np.tan(np.radians(tdeg)))
            worst_uni = max(worst_uni,
                            dmax(solve(uni, (t, 0.0), theta), b),
                            dmax(solve(uni, (t, 0.7 * t), theta), b))
        cell = grating()
        bv = solve(cell, theta=theta)
        tx = float(np.tan(np.radians(20.0)))
        bx = solve(cell, (tx, 0.0), theta)
        for tydeg in (10.0, 30.0):
            ty = float(np.tan(np.radians(tydeg)))
            worst_inv = max(worst_inv,
                            dmax(solve(cell, (0.0, ty), theta), bv),
                            dmax(solve(cell, (tx, ty), theta), bx))
    print(f"  A2 uniform-slant no-op   worst = {worst_uni:.3e}   (bar 1e-11)")
    print(f"  A3 invariant-axis no-op  worst = {worst_inv:.3e}   (bar 1e-11)")

    worst_ratio = 0.0
    for theta in (0.0, 0.20):
        ctrl = cmp_1d(solve(grating(), None, theta, n_orders=7),
                      pmm_jones_1d(P, ER * np.eye(3), EG * np.eye(3), NSUB,
                                   NSUP, D, DUTY, WL, angle=theta, degree=16,
                                   far_field_orders=15))
        for phideg in (10.0, 20.0, 35.0):
            phi = np.radians(phideg)
            got = cmp_1d(
                solve(grating(), (float(np.tan(phi)), 0.0), theta, n_orders=7),
                pmm_jones_1d_slanted(P, ER * np.eye(3), EG * np.eye(3), NSUB,
                                     NSUP, D, DUTY, WL, phi, angle=theta,
                                     degree=16, far_field_orders=15,
                                     factorization="convection"))
            worst_ratio = max(worst_ratio, got / ctrl)
            print(f"  A4 theta={theta} phi={phideg:4.1f}: ctrl={ctrl:.3e} "
                  f"slant={got:.3e}  ratio={got/ctrl:.2f}")
    print(f"  A4 worst ratio = {worst_ratio:.2f}   (bar 3.0)")


def economics():
    """Metric layer vs staircase at MATCHED wall-clock and at typical practice.

    Interleaved (metric, staircase, metric, ...) so a machine-load drift
    cannot systematically favour one route.
    """
    print("\n== ECONOMICS (interleaved) ==")
    nx = 120
    cases = (("normal ", 0.0, 0.0), ("oblique", 0.20, 0.0),
             ("conical", 0.20, 0.60))
    # TWO realistic slanted-pillar cells: a moderate-index pillar with a 24 px
    # walk (t_x = 0.40), and a higher-index, narrower pillar with a 36 px walk
    # (t_x = 0.60) -- a steeper wall on a harder cell.
    cells = (("cellA eps2.25 t=0.40", 24, ER, (0.25, 0.60), (0.30, 0.65)),
             ("cellB eps4.00 t=0.60", 36, 4.0 + 0j, (0.30, 0.55), (0.32, 0.58)))
    for cname, walk_px, er, fx, fy in cells:
      tx = walk_px / nx * P / D
      print(f"\n  ##### {cname} (t_x={tx:.3f}) #####")
      for name, th, ph in cases:
        # interleave: time the metric layer around each staircase rung
        tm = []
        t0 = time.perf_counter()
        met = solve(pillar(nx, er=er, fx=fx, fy=fy), (tx, 0.0), th, ph)
        tm.append(time.perf_counter() - t0)
        rows = []
        for ns in (1, 2, 4, 8, 16):
            st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                                  degree=DEG, n_orders=5,
                                  formulation="laurent")
            for k in range(ns):
                st.add_layer(D / ns, eps_tensor_cell=pillar(
                    nx, er=er, fx=fx, fy=fy,
                    sx=int(round(walk_px * (k + 0.5) / ns))))
            st.set_source(WL, theta=th, phi=ph)
            t0 = time.perf_counter()
            out = st.solve()
            ts = time.perf_counter() - t0
            rows.append((ns, dmax(out, met), ts))
            t0 = time.perf_counter()
            solve(pillar(nx, er=er, fx=fx, fy=fy), (tx, 0.0), th, ph)
            tm.append(time.perf_counter() - t0)
        tmet = float(np.median(tm))
        bound = rows[-1][1]
        print(f"  --- {name} --- metric {tmet:.2f}s (median of {len(tm)}), "
              f"bound vs ns={rows[-1][0]} = {bound:.3e}")
        for ns, d, ts in rows:
            tag = "  <= matched" if ts <= tmet else ""
            print(f"      ns={ns:>2}: err={d:.3e}  {ts:6.2f}s "
                  f"({ts/tmet:.2f}x metric){tag}")


if __name__ == "__main__":
    print(f"lumenairy {lumenairy.__version__} @ {lumenairy.__file__}")
    print(f"python {sys.version.split()[0]} numpy {np.__version__}")
    envelope()
    economics()
