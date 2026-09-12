"""PROBE 9: caches and thread-safety.

(a) does the geometric-eig cache ever return a STALE answer when the
    wavelength / angle / eps changes (including a dispersive index resolving to
    a different value at the same wavelength key)?
(b) do concurrent solves in threads agree with serial solves?
(c) are the cached arrays write-protected?
(d) scale (unit) invariance of the whole 1-D path.
"""
import sys, warnings, threading
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc
from lumenairy.elements.pmm import pmm_efficiency_1d, pmm_jones_1d, PMMStack

print("=== 9a: geo-eig cache -- wavelength / angle / eps staleness ===")
per = 1.0e-6
cases = []
for wl in (1.55e-6, 1.30e-6):
    for angd in (0.0, 20.0):
        for nr in (3.48, 3.60):
            o, R, T = pmm_efficiency_1d(per, nr, 1.444, 1.444, 1.0, 0.4e-6,
                                        0.5, wl, angle=np.deg2rad(angd),
                                        polarization="tm", degree=18,
                                        far_field_orders=11, stabilize=False)
            cases.append(((wl, angd, nr), R.copy(), T.copy()))
# clear caches and redo -> must be identical
pc._clear_pmm_caches()
bad = 0
for (key, R0, T0) in cases:
    wl, angd, nr = key
    o, R, T = pmm_efficiency_1d(per, nr, 1.444, 1.444, 1.0, 0.4e-6, 0.5, wl,
                                angle=np.deg2rad(angd), polarization="tm",
                                degree=18, far_field_orders=11,
                                stabilize=False)
    d = max(np.max(np.abs(R - R0)), np.max(np.abs(T - T0)))
    if d > 0.0:
        bad += 1
        print(f"  MISMATCH after cache clear at {key}: {d:.3e}")
print(f"  {len(cases)} cases, {bad} cache-dependent mismatches "
      f"(0 = cache is content-keyed correctly)")
# and interleaved in a DIFFERENT order (cache warm with other entries)
import random
random.seed(0)
order = list(range(len(cases)))
random.shuffle(order)
bad2 = 0
for i in order:
    key, R0, T0 = cases[i]
    wl, angd, nr = key
    o, R, T = pmm_efficiency_1d(per, nr, 1.444, 1.444, 1.0, 0.4e-6, 0.5, wl,
                                angle=np.deg2rad(angd), polarization="tm",
                                degree=18, far_field_orders=11,
                                stabilize=False)
    if max(np.max(np.abs(R - R0)), np.max(np.abs(T - T0))) > 0.0:
        bad2 += 1
print(f"  shuffled replay: {bad2} mismatches")

print()
print("=== 9b: THREADED vs serial ===")
def job(i, out):
    wl = 1.2e-6 + 0.01e-6 * i
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = pmm_jones_1d(per, 3.48 ** 2 * np.eye(3),
                                  1.444 ** 2 * np.eye(3), 1.444, 1.0, 0.4e-6,
                                  0.5, wl, angle=0.25, degree=16,
                                  far_field_orders=11, stabilize=False)
    out[i] = (R.copy(), T.copy(), J.copy())

ser = {}
for i in range(16):
    job(i, ser)
pc._clear_pmm_caches()
par = {}
ths = [threading.Thread(target=job, args=(i, par)) for i in range(16)]
for t in ths:
    t.start()
for t in ths:
    t.join()
worst = 0.0
for i in range(16):
    worst = max(worst,
                np.max(np.abs(ser[i][0] - par[i][0])),
                np.max(np.abs(ser[i][1] - par[i][1])),
                np.max(np.abs(ser[i][2] - par[i][2])))
print(f"  16 threaded solves vs serial: max|d| = {worst:.3e}")

print()
print("=== 9c: cached arrays read-only? ===")
mats = pc._build_sem(per, 0.5 * per, 3.48 ** 2, 1.444 ** 2, 12, 1, 1, True)
nodes, w = pc._gll_nodes_weights(12)
D = pc._lagrange_derivative_matrix(nodes)
for nm, arr in (("gll nodes", nodes), ("gll weights", w),
                ("Dref", D)):
    try:
        arr[0] += 1.0
        print(f"  {nm}: WRITEABLE (cache poisoning possible)")
    except ValueError:
        print(f"  {nm}: read-only OK")
g = pc._scalar_uniform_geo_eig(mats, 2 * np.pi / 1.55e-6, 0.0)
for nm, arr in (("geo mu", g[0]), ("geo X", g[1])):
    try:
        arr[0] += 1.0
        print(f"  {nm}: WRITEABLE")
    except ValueError:
        print(f"  {nm}: read-only OK")

print()
print("=== 9d: UNIT (scale) invariance of the 1-D path ===")
base = dict(period=1.0e-6, depth=0.4e-6, wl=1.55e-6)
for scale, lbl in ((1.0, "metres"), (1e6, "micrometres"), (1e9, "nanometres"),
                   (1e10, "angstrom"), (1e-3, "kilometres")):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = pmm_efficiency_1d(base["period"] * scale, 3.48, 1.444, 1.444,
                                    1.0, base["depth"] * scale, 0.5,
                                    base["wl"] * scale, angle=0.25,
                                    polarization="tm", degree=18,
                                    far_field_orders=11, stabilize=False)
        o2, R2, T2, J2 = pmm_jones_1d(
            base["period"] * scale, 3.48 ** 2 * np.eye(3),
            1.444 ** 2 * np.eye(3), 1.444, 1.0, base["depth"] * scale, 0.5,
            base["wl"] * scale, angle=0.25, degree=18, far_field_orders=11,
            stabilize=False)
    i0 = int(np.where(o == 0)[0][0])
    print(f"  {lbl:14s}: scalar R0={R[i0]:.14f} sumRT={R.sum()+T.sum():.14f} | "
          f"jones J00={J2[0,0]:.12f}")
