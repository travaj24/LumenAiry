"""F4 (open item O-6) -- a UNIFORM layer on ``N = 1`` at OBLIQUE / CONICAL
incidence.

``N = 1`` is the cheapest region the staggered engine can express (``q = M-1``
per axis, eig ``2 (M-1)^2``), and a uniform region has no walls of its own, so
the per-layer API's natural default for a uniform layer is ``grid = 1``.  But
``stack2d_pure``'s own docstring caveat and S4.3 of the mortar experiment say a
uniform region at OBLIQUE is DEGREE-limited: the physical field carries
``exp(-i alpha0 x)``, the basis carries it only through the Bloch glue ``tau``
times a piecewise polynomial of degree ``M-1`` per segment, so the plane wave
must be RESOLVED -- and with one segment the polynomial has one period of the
Bloch phase to cover.

Two measurements:

(i)  ANALYTIC: one uniform slab (n = 2) between vacuum half-spaces, drawn on
     ``N = 1, 2, 3``, against the exact Fresnel slab, over an ``M`` ladder and
     four incidences.  The exact answer does not depend on ``N`` at all, so
     everything the table shows is the basis's own Bloch-phase error.
(ii) IN A CASCADE: ``pillar(N=2, M=8) | uniform(grid, M_u) | pillar(N=3, M=7)``
     with ONLY the uniform layer's grid and modal count varying, scored against
     the same stack with a CONVERGED uniform layer -- so the reading is that
     layer's own representation error inside a real mortar cascade, with the
     two patterned layers held fixed.
"""
import json
import os
import time
import warnings

import numpy as np
from mortar2d import guard, MortarStack2D
print("lumenairy:", guard(), flush=True)

warnings.simplefilter("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
res = {}

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
print("\n=== (i) one uniform slab vs the analytic Fresnel slab ===",
      flush=True)
print("    The exact answer is N-independent: every entry is the BASIS's own "
      "Bloch-phase error.", flush=True)
NSLAB, D = 2.0, 0.30e-6
rows = []
for (th, ph, tag) in ((0.00, 0.0, "normal"), (0.20, 0.0, "th=0.20"),
                      (0.40, 0.0, "th=0.40"), (0.35, 0.6, "conical 0.35/0.6")):
    Rex, Tex = fresnel_slab_te(1.0, NSLAB, 1.0, D, WL, th)
    print(f"  -- {tag}  (kx0 = {np.sin(th)*np.cos(ph):.4f}, "
          f"exact R = {Rex:.9f}) --", flush=True)
    for N in (1, 2, 3):
        line = []
        for M in (3, 4, 5, 6, 7, 8, 9):
            s = MortarStack2D(PX, PY, n_modes=M, n_orders=0)
            s.add_layer(D, eps=NSLAB ** 2, grid=N)
            s.set_source(WL, theta=th, phi=ph)
            o, R, T = s.solve(jones=False)
            R0, T0 = float(R[1].sum()), float(T[1].sum())
            e = abs(R0 - Rex)
            rows.append(dict(part="i", tag=tag, N=N, M=M, q=N * (M - 1),
                             dR=e, closure=abs(R0 + T0 - 1.0)))
            line.append(f"M={M}:{e:8.1e}")
        print(f"     N={N} (q=N(M-1))  " + "  ".join(line), flush=True)
res["i"] = rows

# =================================================================== (ii)
print("\n=== (ii) the same layer INSIDE a mortar cascade ===", flush=True)
print("    pillar(N=2, M=8) | uniform eps=2.25 (grid, M_u) | pillar(N=3, M=7)",
      flush=True)
A2 = np.full((2, 2), EPS_H + 0j)
A2[0, 0] = EPS_P
B3 = np.full((3, 3), EPS_H + 0j)
B3[0, 0] = EPS_P
tA, tU, tB = 0.16e-6, 0.11e-6, 0.13e-6
MA, MB, NORD = 8, 7, 2


def run(gu, mu, th, ph):
    s = MortarStack2D(PX, PY, n_modes=MB, n_orders=NORD)
    s.add_layer(tA, eps_cell=A2, n_modes=MA)
    s.add_layer(tU, eps=EPS_H, grid=gu, n_modes=mu)
    s.add_layer(tB, eps_cell=B3, n_modes=MB)
    s.set_source(WL, theta=th, phi=ph)
    return s.solve(jones=False)


rows2 = []
for (th, ph, tag) in ((0.00, 0.0, "normal"), (0.20, 0.0, "th=0.20"),
                      (0.40, 0.0, "th=0.40"), (0.35, 0.6, "conical 0.35/0.6")):
    t0 = time.perf_counter()
    oR, RR, TR = run(1, 16, th, ph)          # converged uniform layer, N=1
    oX, RX, TX = run(3, 10, th, ph)          # independent cross-check, N=3
    sc = max(float(np.max(RR)), float(np.max(TR)))
    xchk = max(float(np.abs(RR - RX).max()),
               float(np.abs(TR - TX).max())) / sc
    print(f"  -- {tag}: reference = uniform layer on (N=1, M=16); "
          f"cross-check (N=3, M=10) agrees to {xchk:.2e} --", flush=True)
    rows2.append(dict(part="ii-ref", tag=tag, cross_check=xchk,
                      t=time.perf_counter() - t0))
    for gu in (1, 2, 3):
        line = []
        for mu in (3, 4, 5, 6, 7, 8, 10, 12):
            o, R, T = run(gu, mu, th, ph)
            e = max(float(np.abs(R - RR).max()),
                    float(np.abs(T - TR).max())) / sc
            rows2.append(dict(part="ii", tag=tag, grid=gu, M_u=mu,
                              q=gu * (mu - 1), err=e,
                              closure=float(max(abs(R[p].sum() + T[p].sum() - 1)
                                                for p in (0, 1)))))
            line.append(f"M={mu}:{e:8.1e}")
        print(f"     grid={gu}  " + "  ".join(line), flush=True)
res["ii"] = rows2

json.dump(res, open(os.path.join(HERE, "f4_uniform_oblique.json"), "w"),
          indent=1)
print("\nwrote f4_uniform_oblique.json", flush=True)
