"""PROBE 10: performance.

(a) are the SEM nodal MASS operators diagonal?  (they are built as sums of
    c*diag(w), so they should be) -- if so, _safe_inv(S0)/_safe_inv(Cinv_xx)
    and every iS0 @ M product are O(n^3) where O(n)/O(n^2) would do;
(b) profile a 1-D stack solve; report the split;
(c) measure the achievable saving.
"""
import sys, cProfile, pstats, io, time, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc
from lumenairy.elements.pmm import PMMStack

per, wl = 1.0e-6, 1.55e-6
k0 = 2 * np.pi / wl
kx0 = np.sin(np.deg2rad(13.0)) * k0
t3 = lambda e: pc._tensor3_dict(e * np.eye(3))

print("=== 10a: structure of the assembled operators ===")
for degree, nseg in ((16, 2), (24, 4)):
    w = list(np.diff(np.linspace(0, 1, nseg + 1)))
    eps = [3.48 ** 2 if i % 2 else 1.444 ** 2 for i in range(nseg)]
    m = pc._build_sem_tensor_segments(per, w, [t3(e) for e in eps], degree,
                                      1, True)
    n = m["n_glob"]
    def offd(M):
        return float(np.max(np.abs(M - np.diag(np.diag(M)))))
    print(f"  degree={degree} nseg={nseg} n_glob={n}")
    for k, M in m["mass"].items():
        print(f"    mass[{k:9s}]  max|offdiag| = {offd(M):.3e}   "
              f"nnz_offdiag = {int(np.count_nonzero(M - np.diag(np.diag(M))))}")
    for k, M in m["stiff"].items():
        print(f"    stiff[{k:9s}] max|offdiag| = {offd(M):.3e}   "
              f"density = {np.count_nonzero(M)/M.size:.3f}")
    for k, M in m["conv"].items():
        print(f"    conv[{k:9s}]  density = {np.count_nonzero(M)/M.size:.3f}")
    # scalar build too
    ms = pc._build_sem(per, 0.5 * per, 3.48 ** 2, 1.444 ** 2, degree, 1, 1,
                       True)
    for k in ("S0", "Peps", "Pinv"):
        print(f"    scalar {k:5s} max|offdiag| = {offd(ms[k]):.3e}")

print()
print("=== 10b: cost split of _sem_modes_tensor at n_glob ~ 100-200 ===")
for degree, nseg in ((16, 6), (24, 8), (32, 8)):
    w = list(np.diff(np.linspace(0, 1, nseg + 1)))
    eps = [3.48 ** 2 if i % 2 else 1.444 ** 2 for i in range(nseg)]
    m = pc._build_sem_tensor_segments(per, w, [t3(e) for e in eps], degree,
                                      1, True)
    n = m["n_glob"]
    S0 = m["S0"]
    t = time.perf_counter()
    for _ in range(5):
        iS0 = pc._safe_inv(S0)
    t_inv = (time.perf_counter() - t) / 5
    d = np.diag(S0)
    t = time.perf_counter()
    for _ in range(5):
        iS0d = np.diag(1.0 / d)
    t_invd = (time.perf_counter() - t) / 5
    print(f"  n_glob={n:4d}: _safe_inv(S0) = {t_inv*1e3:7.3f} ms   "
          f"diag inverse = {t_invd*1e3:7.3f} ms   "
          f"max|d| = {np.max(np.abs(iS0 - iS0d)):.2e}")
    t = time.perf_counter()
    for _ in range(3):
        W2, V2, lam, q = pc._sem_modes_tensor(m, k0, kx0)
    t_modes = (time.perf_counter() - t) / 3
    Mbig = np.random.rand(2 * n, 2 * n) + 1j * np.random.rand(2 * n, 2 * n)
    t = time.perf_counter()
    np.linalg.eig(Mbig)
    t_eig = time.perf_counter() - t
    print(f"             _sem_modes_tensor = {t_modes*1e3:8.2f} ms, "
          f"bare 2n eig = {t_eig*1e3:8.2f} ms "
          f"-> overhead = {(t_modes-t_eig)*1e3:8.2f} ms "
          f"({100*(t_modes-t_eig)/t_modes:.0f}%)")

print()
print("=== 10c: full stack solve profile (degree 24, 8 layers) ===")
st = PMMStack(per, n_substrate=1.444, n_superstrate=1.0, degree=24,
              far_field_orders=21)
for i in range(8):
    st.add_layer(0.06e-6, segments=[(0.3 + 0.05 * i, 3.48 ** 2),
                                    (0.7 - 0.05 * i, 1.444 ** 2)])
st.set_source(wl, angle=np.deg2rad(13.0))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    t = time.perf_counter()
    st.solve()
    print(f"  wall = {time.perf_counter()-t:.3f} s")
    pr = cProfile.Profile()
    pr.enable()
    st.solve()
    pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(22)
print(s.getvalue()[:4200])
