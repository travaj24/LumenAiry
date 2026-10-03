"""E2-H: ALL-HOST stacks through the curved mortar against the exact Airy
slab -- the per-layer mortar's own algebra test (shipped: three all-host
layers on three non-uniform grids reproduce the analytic slab to 1.2e-10 at
M = 5) with the layers on DIFFERENT maps.  Every layer is a homogeneous film
(eps 2.25, 1.7, 1.0), each carried on its own map; the device is a plain
three-film stack, so the exact answer is the 1-D transfer matrix.

arms (arg 1): 'sin_circ' (sinusoid map / circle map), 'circ_sin',
'three' (circle / sinusoid-x / sinusoid-y), 'stretch' (two separable
stretches); rung M (arg 2); optional theta phi (rad)."""
import sys
import time

import numpy as np
from _common import CM, N_SUB, N_SUP, R_CIRC, WL, P, PMM2DStackPure, dump

from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall, compile_shapes

arm, M = sys.argv[1], int(sys.argv[2])
th, ph = (float(sys.argv[3]), float(sys.argv[4])) if len(sys.argv) > 4 \
    else (0.0, 0.0)
_e, _x, _y, circ = compile_shapes(P, P, [Circle(0.6, 0.6, R_CIRC, 4.0)], 1.0)
_e, _x, _y, sinx = compile_shapes(P, P, [SinusoidalWall("x", 0.6, 0.12,
                                                        eps=2.25)], 1.0)
_e, _x, _y, siny = compile_shapes(P, P, [SinusoidalWall("y", 0.5, 0.1,
                                                        eps=2.25)], 1.0)
st1 = CM.SeparableStretch.from_physical_walls(
    np.array([0, 0.3, 0.9, P]), np.array([0, 0.3, 0.9, P]),
    fx=CM.SineStretch(0.06 * P))
st2 = CM.SeparableStretch.from_physical_walls(
    np.array([0, 0.5, 0.8, P]), np.array([0, 0.2, 0.7, P]),
    fy=CM.SineStretch(0.08 * P))
maps = {"sin_circ": [sinx, circ], "circ_sin": [circ, sinx],
        "sinx_siny": [sinx, siny], "circ_circ": [circ, circ],
        "sinx_id": [sinx, None], "circ_id": [circ, None],
        "three": [circ, sinx, siny], "stretch": [st1, st2],
        "shared_sinx": [None, None], "shared_circ": [None, None],
        "shared_siny": [None, None]}[arm]
import os

eps = ([1.0] * 3 if os.environ.get("E2H_VACUUM") else
       [2.25, 1.7, 1.0])[:len(maps)]
if os.environ.get("E2H_VACUUM"):
    N_SUB = 1.0
thick = [0.3, 0.25, 0.2][:len(maps)]


def airy(pol):
    """Exact reflectance / transmittance of the film stack (s or p)."""
    k0 = 2 * np.pi / WL
    kx = N_SUP * np.sin(th)
    ns = [N_SUP] + [np.sqrt(e) for e in eps] + [N_SUB]
    kz = [np.sqrt(complex(n * n - kx * kx)) for n in ns]
    if pol == "s":
        Y = kz
    else:
        Y = [kz[i] / ns[i] ** 2 for i in range(len(ns))]
    Mt = np.eye(2, dtype=complex)
    for i in range(1, len(ns) - 1):
        d = k0 * kz[i] * thick[i - 1]
        Mt = Mt @ np.array([[np.cos(d), -1j * np.sin(d) / Y[i]],
                            [-1j * Y[i] * np.sin(d), np.cos(d)]])
    Y0, Ys = Y[0], Y[-1]
    B, C = Mt @ np.array([1.0, Ys])
    r = (Y0 * B - C) / (Y0 * B + C)
    t = 2 * Y0 / (Y0 * B + C)
    R = abs(r) ** 2
    T = (Ys.real / Y0.real) * abs(t) ** 2
    return R, T


if arm.startswith("shared_"):
    cm1 = {"shared_sinx": sinx, "shared_circ": circ, "shared_siny": siny}[arm]
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=3, cmap=cm1)
    for t, e in zip(thick, eps):
        st.add_layer(t, eps=e)
else:
    st = PMM2DStackPure(P, P, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=3, layer_grids="per-layer")
    Ms = [int(v) for v in os.environ.get("E2H_MS", "").split(",") if v]
    for k, (t, e, cm) in enumerate(zip(thick, eps, maps)):
        st.add_layer(t, eps=e, cmap=cm, n_modes=Ms[k] if Ms else M)
st.set_source(WL, theta=th, phi=ph)
t0 = time.perf_counter()
o, R, T, J = st.solve()
dt = time.perf_counter() - t0
o = np.asarray(o)
p0 = int(np.nonzero((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
Rs, Ts = airy("s")
Rp, Tp = airy("p")
# input E along x at phi = 0 is p, E along y is s
err = max(abs(R[0, p0] - Rp), abs(T[0, p0] - Tp), abs(R[1, p0] - Rs),
          abs(T[1, p0] - Ts))
other = float(max(np.abs(np.delete(R, p0, axis=1)).max(),
                  np.abs(np.delete(T, p0, axis=1)).max()))
out = dict(arm=arm, M=M, Ms=os.environ.get("E2H_MS"), theta=th, phi=ph,
           err_vs_airy=float(err),
           nonspecular=other, wall=dt,
           closure=np.abs(R.sum(1) + T.sum(1) - 1.0))
print(out)
dump(f"e2_h_host_{arm}{'_vac' if os.environ.get('E2H_VACUUM') else ''}"
     f"_M{M}{('_Ms' + os.environ['E2H_MS'].replace(',', '-')) if os.environ.get('E2H_MS') else ''}_th{th}_ph{ph}.json", out)
