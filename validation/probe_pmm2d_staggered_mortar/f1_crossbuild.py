"""F1 (open item O-1) -- SECOND BLAS BUILD.

Re-runs the DECISIVE quantities of the mortar experiment in ONE process so the
two builds measure exactly the same arms, and writes a tagged JSON that the
comparison script diffs:

  A  M1 conforming identity (mortar path vs the shipped union cascade)
  B  the ISOLATED-MORTAR Fresnel error (transparent interface, m0b(b))
  C  M4 vs the exact 1-D ``PMMStack`` oracle (mortar ladder + union arm)
  D  M4c EQUAL-DOF (union vs mortar at identical eigenproblem sizes)
  E  M5 OOP twin: uniform OOP slab SPLIT across grids vs ``berreman_jones_1d``
  F  M7 conditioning EXTREMES (worst/best equilibrated rcond per site)

The point is NOT to reproduce the Windows numbers: it is to measure the
CROSS-BUILD SPREAD of every quantity a future test bar might read, because
``docs/TESTING_STANDARDS.md`` rule 5 makes a bar sound only when the spread sits
decades below it.

Usage:  python f1_crossbuild.py [tag]      (tag defaults to win / wsl)
"""
import json
import os
import platform
import sys
import time
import warnings

import numpy as np
import mortar2d
from mortar2d import guard, MortarStack2D, refine_cell
print("lumenairy:", guard(), flush=True)
from lumenairy import PMMStack, berreman_jones_1d
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.rcwa._core import uniaxial_tensor

warnings.simplefilter("ignore")
TAG = sys.argv[1] if len(sys.argv) > 1 else ("win" if os.name == "nt" else "wsl")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, f"f1_crossbuild_{TAG}.json")

import scipy  # noqa: E402
try:
    _blas = np.__config__.CONFIG["Build Dependencies"]["blas"]["name"]
except Exception:                                          # noqa: BLE001
    _blas = "unknown"
BUILD = dict(tag=TAG, python=platform.python_version(), numpy=np.__version__,
             scipy=scipy.__version__, platform=platform.platform(), blas=_blas)
print("BUILD:", BUILD, flush=True)
res = dict(build=BUILD)

PX = PY = 0.9e-6
WL = 0.60e-6

# ======================================================================= A
print("\n[A] M1 conforming identity (forced mortar vs shipped union cascade)",
      flush=True)
A2s = np.array([[2.0] * 3, [4.0] * 3, [9.0] * 3], np.complex128)
B3s = np.array([[9.0, 2.0, 4.0]] * 3, np.complex128).T.copy()
Ppil = np.array([[6.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 3.5]],
                np.complex128)
tA, tB = 0.20e-6, 0.15e-6
CASES = [
    ("stripe|stripe  M=5 th=0.00", [A2s, B3s], 5, 0.00, 0.0),
    ("stripe|stripe  M=5 th=0.20", [A2s, B3s], 5, 0.20, 0.0),
    ("stripe|pillar  M=6 th=0.20", [A2s, Ppil], 6, 0.20, 0.0),
    ("pillar|pillar  M=6 conical", [Ppil, B3s], 6, 0.25, 0.7),
    ("3-layer + unif M=5 th=0.15", [A2s, None, Ppil], 5, 0.15, 0.4),
]
rowsA = []
for name, cells, M, th, ph in CASES:
    ref = PMM2DStackPure(PX, PY, n_modes=M, n_orders=5)
    mor = MortarStack2D(PX, PY, n_modes=M, n_orders=5)
    for c in cells:
        if c is None:
            ref.add_layer(0.08e-6, eps=2.25)
            mor.add_layer(0.08e-6, eps=2.25, grid=3)
        else:
            ref.add_layer(tA if c is cells[0] else tB, eps_cell=c)
            mor.add_layer(tA if c is cells[0] else tB, eps_cell=c)
    ref.set_source(WL, theta=th, phi=ph)
    mor.set_source(WL, theta=th, phi=ph)
    o0, R0, T0, J0 = ref.solve()
    o1, R1, T1, J1 = mor.solve(force_mortar=True)
    sc = max(float(np.max(R0)), float(np.max(T0)))
    dR = float(np.max(np.abs(R0 - R1))) / sc
    dT = float(np.max(np.abs(T0 - T1))) / sc
    dJ = float(np.max(np.abs(J0 - J1))) / float(np.max(np.abs(J0)))
    rowsA.append(dict(case=name, dR=dR, dT=dT, dJ=dJ))
    print(f"  {name:28s} dR {dR:9.2e} dT {dT:9.2e} dJ {dJ:9.2e}", flush=True)
res["A_m1"] = dict(rows=rowsA,
                   worst=max(max(r["dR"], r["dT"], r["dJ"]) for r in rowsA))

# ======================================================================= B
print("\n[B] isolated mortar: transparent interface vs analytic Fresnel",
      flush=True)


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


NSLAB, D = 2.0, 0.30e-6
rowsB = []
for theta in (0.0, 0.20):
    Rex, _Tex = fresnel_slab_te(1.0, NSLAB, 1.0, D, WL, theta)
    for M in (5, 7):
        for (Na, Nb) in ((2, 2), (2, 3), (3, 4)):
            s = MortarStack2D(PX, PY, n_modes=M, n_orders=2)
            s.add_layer(D / 2, eps=NSLAB ** 2, grid=Na)
            s.add_layer(D / 2, eps=NSLAB ** 2, grid=Nb)
            s.set_source(WL, theta=theta, phi=0.0)
            o, R, T = s.solve(jones=False)
            R0, T0 = float(R[1].sum()), float(T[1].sum())
            rowsB.append(dict(theta=theta, M=M, Na=Na, Nb=Nb,
                              dR=abs(R0 - Rex), closure=abs(R0 + T0 - 1.0)))
            print(f"  th={theta:.2f} M={M} ({Na},{Nb}) |dR| "
                  f"{abs(R0-Rex):9.2e}  |R+T-1| {abs(R0+T0-1):8.1e}",
                  flush=True)
res["B_fresnel"] = rowsB

# ======================================================================= C/D
print("\n[C] M4 vs the exact 1-D PMMStack oracle", flush=True)
A2 = np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128)
B3 = np.array([[9.0] * 3, [2.0] * 3, [4.0] * 3], np.complex128)
THETA, NORD = 0.20, 2


def oracle(degree):
    s = PMMStack(PX, n_superstrate=1.0, n_substrate=1.0, degree=degree)
    s.add_layer(tA, segments=[(1 / 2, 2.0), (1 / 2, 6.0)])
    s.add_layer(tB, segments=[(1 / 3, 9.0), (1 / 3, 2.0), (1 / 3, 4.0)])
    s.set_source(WL, theta=THETA)
    o, R, T = s.solve()[:3]
    o = np.asarray(o).ravel()
    i = np.argsort(o)
    return o[i], R[1][i], T[1][i]


MO, RO, TO = oracle(14)
_m12, R12, T12 = oracle(12)
res["C_oracle_selfgap"] = float(max(np.abs(RO - R12).max(),
                                    np.abs(TO - T12).max()))
print(f"  oracle deg12 vs deg14 self-gap {res['C_oracle_selfgap']:.3e}",
      flush=True)


def score(orders, R, T):
    oo = np.asarray(orders)
    sel = oo[:, 1] == 0
    m = oo[sel, 0]
    j = np.argsort(m)
    m, r, t = m[j], R[1][sel][j], T[1][sel][j]
    keep = np.isin(MO, m)
    direct = max(float(np.abs(r - RO[keep]).max()),
                 float(np.abs(t - TO[keep]).max()))
    mirror = max(float(np.abs(r - RO[keep][::-1]).max()),
                 float(np.abs(t - TO[keep][::-1]).max()))
    return direct, mirror


rowsC = []
for M in (5, 7, 9):
    t0 = time.perf_counter()
    s = MortarStack2D(PX, PY, n_modes=M, n_orders=NORD)
    s.add_layer(tA, eps_cell=A2)
    s.add_layer(tB, eps_cell=B3)
    s.set_source(WL, theta=THETA, phi=0.0)
    o, R, T = s.solve(jones=False)
    d, mi = score(o, R, T)
    clo = float(abs(R[1].sum() + T[1].sum() - 1.0))
    rowsC.append(dict(arm="mortar", M=M, err=d, mirror=mi, closure=clo,
                      t=time.perf_counter() - t0))
    print(f"  mortar M={M:2d} err {d:9.2e} mirror {mi:8.2e} closure {clo:8.1e}"
          f"  {rowsC[-1]['t']:6.1f}s", flush=True)
for M in (4,):
    t0 = time.perf_counter()
    s = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    s.add_layer(tA, eps_cell=refine_cell(A2, 3))
    s.add_layer(tB, eps_cell=refine_cell(B3, 2))
    s.set_source(WL, theta=THETA, phi=0.0)
    o, R, T = s.solve(jones=False)
    d, mi = score(o, R, T)
    clo = float(abs(R[1].sum() + T[1].sum() - 1.0))
    rowsC.append(dict(arm="union6", M=M, err=d, mirror=mi, closure=clo,
                      t=time.perf_counter() - t0))
    print(f"  union6 M={M:2d} err {d:9.2e} closure {clo:8.1e} "
          f" {rowsC[-1]['t']:6.1f}s", flush=True)
res["C_m4"] = rowsC

print("\n[D] M4c equal-DOF", flush=True)
rowsD = []
for M in (4, 5):
    q = 6 * (M - 1)
    MA, MB = 3 * M - 2, 2 * M - 1
    t0 = time.perf_counter()
    u = PMM2DStackPure(PX, PY, n_modes=M, n_orders=NORD)
    u.add_layer(tA, eps_cell=refine_cell(A2, 3))
    u.add_layer(tB, eps_cell=refine_cell(B3, 2))
    u.set_source(WL, theta=THETA, phi=0.0)
    ou, Ru, Tu = u.solve(jones=False)
    tu = time.perf_counter() - t0
    eu, _ = score(ou, Ru, Tu)
    cu = max(abs(Ru[p].sum() + Tu[p].sum() - 1.0) for p in (0, 1))
    t0 = time.perf_counter()
    mo = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
    mo.add_layer(tA, eps_cell=A2, n_modes=MA)
    mo.add_layer(tB, eps_cell=B3, n_modes=MB)
    mo.set_source(WL, theta=THETA, phi=0.0)
    om, Rm, Tm = mo.solve(jones=False)
    tm = time.perf_counter() - t0
    em, _ = score(om, Rm, Tm)
    cm = max(abs(Rm[p].sum() + Tm[p].sum() - 1.0) for p in (0, 1))
    rowsD.append(dict(q=q, err_union=eu, err_mortar=em, clo_union=float(cu),
                      clo_mortar=float(cm), ratio=em / eu, t_union=tu,
                      t_mortar=tm))
    print(f"  q={q:3d} union {eu:9.2e} (clo {cu:8.1e}) | mortar {em:9.2e} "
          f"(clo {cm:8.1e}) | ratio {em/eu:6.2f}x  {tu:6.1f}s/{tm:6.1f}s",
          flush=True)
res["D_equal_dof"] = rowsD

# ======================================================================= E
print("\n[E] M5 OOP twin: split uniform OOP slab vs berreman_jones_1d",
      flush=True)
PU, WLU = 0.9e-6, 1.0e-6
NSUP, NSUB = 1.0, 1.5
OOP = np.asarray(uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0),
                                 phi=np.deg2rad(25.0)), dtype=complex)
DEPU = 0.35e-6
rowsE = []
for (th, ph) in ((0.0, 0.0), (np.deg2rad(25.0), np.deg2rad(40.0))):
    Rb, Tb, Jrb, _ = berreman_jones_1d([(OOP, DEPU)], NSUB, NSUP, WLU,
                                       theta=th, phi=ph)
    for M in (5, 7):
        for (Na, Nb) in ((2, 2), (2, 3)):
            s = MortarStack2D(PU, PU, n_superstrate=NSUP, n_substrate=NSUB,
                              n_modes=M, n_orders=2)
            s.add_layer(DEPU / 2, eps=OOP, grid=Na)
            s.add_layer(DEPU / 2, eps=OOP, grid=Nb)
            s.set_source(WLU, theta=th, phi=ph)
            o, R, T, J = s.solve()
            dJ = float(np.max(np.abs(J - Jrb))) / float(np.max(np.abs(Jrb)))
            clo = max(abs(R[p].sum() + T[p].sum() - 1.0) for p in (0, 1))
            rowsE.append(dict(theta_deg=float(np.rad2deg(th)), M=M, Na=Na,
                              Nb=Nb, dJ=dJ, closure=float(clo)))
            print(f"  th={np.rad2deg(th):4.1f} M={M} ({Na},{Nb}) dJ {dJ:9.2e} "
                  f"clo {clo:8.1e}", flush=True)
res["E_oop"] = rowsE

# ======================================================================= F
print("\n[F] M7 conditioning extremes", flush=True)
PXC, WLC = 1.2e-6, 0.85e-6


def pillar(N, k, eh=2.25, ep=9.0):
    c = np.full((N, N), eh + 0j)
    c[:k, :k] = ep
    return c


def stripe(N, hi):
    c = np.full((N, N), 2.0 + 0j)
    c[:hi, :] = 6.0
    return c


CONFIGS = [
    ("M3 pillar (2,3)", [(0.16e-6, pillar(2, 1)), (0.13e-6, pillar(3, 1))],
     0.18, 0.35),
    ("M3 pillar (2,6)", [(0.16e-6, pillar(2, 1)), (0.13e-6, pillar(6, 2))],
     0.18, 0.35),
    ("M4 stripe (2,3)", [(0.20e-6, stripe(2, 1)), (0.15e-6, stripe(3, 1))],
     0.20, 0.0),
    ("M6 taper (2,3,6)", [(0.12e-6, pillar(2, 1)), (0.12e-6, pillar(3, 1)),
                          (0.12e-6, pillar(6, 1))], 0.18, 0.0),
]
by_site, rowsF = {}, []
for name, layers, th, ph in CONFIGS:
    for M in (4, 5):
        mortar2d.CENSUS = []
        s = MortarStack2D(PXC, PXC, n_modes=M, n_orders=2)
        for t, c in layers:
            s.add_layer(t, eps_cell=c)
        s.set_source(WLC, theta=th, phi=ph)
        s.solve(jones=False)
        for site, n, rc in mortar2d.CENSUS:
            by_site.setdefault(site, []).append(float(rc))
        worst = min(rc for _s, _n, rc in mortar2d.CENSUS)
        mortar2d.CENSUS = None
        rowsF.append(dict(config=name, M=M, worst_rcond=float(worst)))
        print(f"  {name:20s} M={M} worst rcond {worst:.3e}", flush=True)
res["F_cond"] = dict(rows=rowsF,
                     per_site={k: [min(v), max(v)] for k, v in by_site.items()})

json.dump(res, open(OUT, "w"), indent=1)
print("\nwrote", OUT, flush=True)
