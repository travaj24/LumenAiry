"""F4 (open item O-6) -- a UNIFORM layer on ``N = 1`` at OBLIQUE / CONICAL
incidence.

``N = 1`` is the cheapest region the staggered engine can express (``q = M-1``
per axis, eig ``2 (M-1)^2``), and a uniform region has no walls of its own, so
the per-layer API's natural default for a uniform layer is ``grid = 1``.  But
``stack2d_pure``'s own docstring caveat and S4.3 of the mortar experiment say a
uniform region at OBLIQUE is DEGREE-limited: the physical field carries
``exp(-i alpha0 x)`` and the basis carries it only through the Bloch glue
``tau`` times a piecewise polynomial of degree ``M-1``, so the plane wave must
be RESOLVED -- and with one segment the polynomial has the whole period to
cover.

Three measurements:

(i)   ISOLATED, ANALYTIC: one uniform slab (n = 2) between vacuum half-spaces
      drawn on ``N = 1, 2, 3``, against the exact Fresnel slab (``phi = 0``, so
      the incident ``E_y`` row IS s-polarized) over an ``M`` ladder and four
      polar angles, and against ``berreman_jones_1d`` at CONICAL incidence
      (where the TE/TM split is not the incident basis and the scalar Fresnel
      formula does not apply -- a first cut of this script scored the conical
      row against ``fresnel_slab_te`` and read a spurious 2.4e-02 FLAT in every
      cell; that arm was withdrawn).
(ii)  IN A CASCADE, against an EXACT oracle: a y-uniform STRIPE stack
      ``A(duty 1/2, N=2, M=8) | uniform (grid, M_u) | B(duty 1/3, N=3, M=7)``,
      so the exact 1-D ``PMMStack`` at degree 14 is the truth for the whole
      stack.  Only the uniform layer's grid and modal count vary.
(iii) versus the SHARED-GRID path: the same stack on the union lattice
      ``N = 6`` at one global ``M``, scored against the same oracle -- so the
      per-layer arm's uniform-layer choice is priced against what a user gets
      today.
"""
import json
import os
import time
import warnings

import numpy as np
from mortar2d import guard, MortarStack2D, refine_cell
print("lumenairy:", guard(), flush=True)
from lumenairy import PMMStack, berreman_jones_1d
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
res = {}
OUT = os.path.join(HERE, "f4_uniform_oblique.json")


def dump():
    json.dump(res, open(OUT, "w"), indent=1)


PX = PY = 1.2e-6
WL = 0.85e-6
EPS_H, EPS_P = 2.25, 9.0


def fresnel_slab_te(n0, n1, n2, d, wl, theta):
    k0 = 2 * np.pi / wl
    s = n0 * np.sin(theta)
    kz = [k0 * np.sqrt(complex(n ** 2 - s ** 2)) for n in (n0, n1, n2)]
    r01 = (kz[0] - kz[1]) / (kz[0] + kz[1])
    r12 = (kz[1] - kz[2]) / (kz[1] + kz[2])
    t01 = 2 * kz[0] / (kz[0] + kz[1])
    t12 = 2 * kz[1] / (kz[1] + kz[2])
    ph = np.exp(2j * kz[1] * d)
    r = (r01 + r12 * ph) / (1 + r01 * r12 * ph)
    t = t01 * t12 * np.exp(1j * kz[1] * d) / (1 + r01 * r12 * ph)
    return float(abs(r) ** 2), float(abs(t) ** 2 * (kz[2] / kz[0]).real)


# =================================================================== (i)
print("\n=== (i) one uniform slab, ISOLATED, vs the analytic Fresnel slab ===",
      flush=True)
print("    The exact answer is N-independent, so every entry is the BASIS's "
      "own Bloch-phase error.", flush=True)
NSLAB, D = 2.0, 0.30e-6
MLAD = (3, 4, 5, 6, 7, 8, 9)
rows = []
for th in (0.00, 0.20, 0.40, 0.60):
    Rex, _T = fresnel_slab_te(1.0, NSLAB, 1.0, D, WL, th)
    print(f"  -- theta={th:.2f}, phi=0  (kx0 = {np.sin(th):.4f}, exact R = "
          f"{Rex:.9f}) --", flush=True)
    for N in (1, 2, 3):
        line = []
        for M in MLAD:
            s = MortarStack2D(PX, PY, n_modes=M, n_orders=0)
            s.add_layer(D, eps=NSLAB ** 2, grid=N)
            s.set_source(WL, theta=th, phi=0.0)
            o, R, T = s.solve(jones=False)
            R0, T0 = float(R[1].sum()), float(T[1].sum())
            rows.append(dict(part="i", theta=th, N=N, M=M, q=N * (M - 1),
                             dR=abs(R0 - Rex), closure=abs(R0 + T0 - 1.0)))
            line.append(f"M={M}:{abs(R0-Rex):8.1e}")
        print(f"     N={N}  " + "  ".join(line), flush=True)
res["i"] = rows
dump()

print("\n  -- CONICAL (theta=0.35 rad, phi=0.6 rad) vs berreman_jones_1d "
      "(relative dJones) --", flush=True)
rowsc = []
_Rb, _Tb, Jref, _ = berreman_jones_1d([(np.eye(3) * NSLAB ** 2, D)], 1.0, 1.0,
                                      WL, theta=0.35, phi=0.6)
for N in (1, 2, 3):
    line = []
    for M in MLAD:
        s = MortarStack2D(PX, PY, n_modes=M, n_orders=0)
        s.add_layer(D, eps=NSLAB ** 2, grid=N)
        s.set_source(WL, theta=0.35, phi=0.6)
        o, R, T, J = s.solve()
        dJ = float(np.max(np.abs(J - Jref))) / float(np.max(np.abs(Jref)))
        rowsc.append(dict(part="i-conical", N=N, M=M, q=N * (M - 1), dJ=dJ))
        line.append(f"M={M}:{dJ:8.1e}")
    print(f"     N={N}  " + "  ".join(line), flush=True)
res["i_conical"] = rowsc
dump()

# =================================================================== (ii)
print("\n=== (ii) the uniform layer INSIDE a mortar cascade, vs the EXACT "
      "1-D oracle ===", flush=True)
print("    stripe A(duty 1/2, N=2, M=8) | uniform eps=2.25 (grid, M_u) | "
      "stripe B(duty 1/3, N=3, M=7)", flush=True)
A2 = np.array([[EPS_P, EPS_P], [EPS_H, EPS_H]], complex)          # duty 1/2
B3 = np.array([[EPS_P] * 3, [EPS_H] * 3, [EPS_H] * 3], complex)   # duty 1/3
tA, tU, tB = 0.16e-6, 0.11e-6, 0.13e-6
MA, MB, NORD = 8, 7, 2


def oracle(theta, deg=14):
    s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=deg)
    s.add_layer(tA, segments=[(1 / 2, EPS_P), (1 / 2, EPS_H)])
    s.add_layer(tU, segments=[(1.0, EPS_H)])
    s.add_layer(tB, segments=[(1 / 3, EPS_P), (2 / 3, EPS_H)])
    s.set_source(WL, theta=theta)
    o, R, T = s.solve()[:3]
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return o[i], R[1][i], T[1][i]


def score(orders, R, T, MO, RO, TO):
    oo = np.asarray(orders)
    sel = oo[:, 1] == 0
    m = oo[sel, 0]
    j = np.argsort(m)
    m, r, t = m[j], np.asarray(R)[1][sel][j], np.asarray(T)[1][sel][j]
    keep = np.isin(MO, m)
    return max(float(np.abs(r - RO[keep]).max()),
               float(np.abs(t - TO[keep]).max()))


rows2 = []
for th in (0.00, 0.20, 0.40):
    MO, RO, TO = oracle(th)
    _m, R12, T12 = oracle(th, 12)
    sg = float(max(np.abs(RO - R12).max(), np.abs(TO - T12).max()))
    print(f"  -- theta={th:.2f}: 1-D oracle deg12-vs-deg14 self-gap "
          f"{sg:.2e} --", flush=True)
    rows2.append(dict(part="ii-oracle", theta=th, selfgap=sg))
    # grid = 6 (the union lattice) is NOT swept here: its q = 6 (M_u - 1)
    # reaches an eig of 2 q^2 = 5832 at M_u = 10, an hour per point.  The
    # shared-grid arm is measured properly in (iii) instead.
    for gu in (1, 2, 3):
        line = []
        for mu in (3, 4, 5, 6, 7, 8, 10, 12):
            s = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
            s.add_layer(tA, eps_cell=A2, n_modes=MA)
            s.add_layer(tU, eps=EPS_H, grid=gu, n_modes=mu)
            s.add_layer(tB, eps_cell=B3, n_modes=MB)
            s.set_source(WL, theta=th, phi=0.0)
            o, R, T = s.solve(jones=False)
            e = score(o, R, T, MO, RO, TO)
            rows2.append(dict(part="ii", theta=th, grid=gu, M_u=mu,
                              q=gu * (mu - 1), err=e,
                              closure=float(max(abs(R[p].sum() + T[p].sum() - 1)
                                                for p in (0, 1)))))
            line.append(f"M={mu}:{e:8.1e}")
        print(f"     grid={gu} (q=grid*(M_u-1))  " + "  ".join(line),
              flush=True)
        res["ii"] = rows2
        dump()

# =================================================================== (iii)
print("\n=== (iii) the SHARED-GRID path (union N=6, one global M), same "
      "oracle ===", flush=True)
rows3 = []
for th in (0.00, 0.20, 0.40):
    MO, RO, TO = oracle(th)
    line = []
    for M in (4, 5):
        t0 = time.perf_counter()
        u = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
        u.add_layer(tA, eps_cell=refine_cell(A2, 3))
        u.add_layer(tU, eps=EPS_H)
        u.add_layer(tB, eps_cell=refine_cell(B3, 2))
        u.set_source(WL, theta=th, phi=0.0)
        o, R, T = u.solve(jones=False)
        e = score(o, R, T, MO, RO, TO)
        dt = time.perf_counter() - t0
        rows3.append(dict(theta=th, M=M, q=6 * (M - 1), err=e, t=dt))
        line.append(f"M={M} (q={6*(M-1)}): {e:8.1e} [{dt:5.1f}s]")
    print(f"  theta={th:.2f}  " + "   ".join(line), flush=True)
    res["iii"] = rows3
    dump()

dump()
print("\nwrote", OUT, flush=True)
