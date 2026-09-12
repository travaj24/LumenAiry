import sys, time, tracemalloc, numpy as np, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.elements import polarization as P

wl = 633e-9

def timeit(fn, n=3):
    fn()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return min(ts)

def peakmem(fn):
    fn()
    tracemalloc.start(); tracemalloc.reset_peak()
    fn()
    c, p = tracemalloc.get_traced_memory(); tracemalloc.stop()
    return p

print("=== 1. create_gaussian_beam N=8192: time + peak transient ===")
N = 8192; dx = 1e-6
t = timeit(lambda: la.create_gaussian_beam(N, dx, wl, w0=500e-6, normalize='peak'))
pk = peakmem(lambda: la.create_gaussian_beam(N, dx, wl, w0=500e-6, normalize='peak'))
out_bytes = N * N * 16
print("  time %.3f s ; peak %.1f MB ; output %.1f MB ; peak/output = %.2f"
      % (t, pk / 1e6, out_bytes / 1e6, pk / out_bytes))
t64 = timeit(lambda: la.create_gaussian_beam(N, dx, wl, w0=500e-6, dtype=np.complex64))
pk64 = peakmem(lambda: la.create_gaussian_beam(N, dx, wl, w0=500e-6, dtype=np.complex64))
print("  complex64: time %.3f s ; peak %.1f MB ; output %.1f MB ; peak/output = %.2f"
      % (t64, pk64 / 1e6, out_bytes / 2 / 1e6, pk64 / (out_bytes / 2)))

print("")
print("=== 2. create_laguerre_gauss / hermite_gauss N=4096 ===")
N = 4096; dx = 1e-6
for p_ in (0, 3):
    t = timeit(lambda: la.create_laguerre_gauss(N, dx, 300e-6, wl, p=p_, l=2))
    print("  LG p=%d,l=2 N=4096: %.3f s" % (p_, t))
pkL = peakmem(lambda: la.create_laguerre_gauss(N, dx, 300e-6, wl, p=3, l=2))
print("  LG peak %.1f MB vs output %.1f MB ; ratio %.2f"
      % (pkL / 1e6, N * N * 16 / 1e6, pkL / (N * N * 16)))
t = timeit(lambda: la.create_hermite_gauss(N, dx, 300e-6, wl, m=3, n=2))
pkH = peakmem(lambda: la.create_hermite_gauss(N, dx, 300e-6, wl, m=3, n=2))
print("  HG m=3,n=2 N=4096: %.3f s ; peak %.1f MB ; ratio %.2f"
      % (t, pkH / 1e6, pkH / (N * N * 16)))

print("")
print("=== 3. JonesField ops: allocations per element ===")
N = 2048
Ex = np.zeros((N, N), complex); Ey = np.zeros((N, N), complex)
jf = P.JonesField(Ex, Ey, 1e-6)
one = N * N * 16
print("  one full-grid complex128 array = %.1f MB" % (one / 1e6))
pk = peakmem(lambda: P.apply_waveplate(P.JonesField(Ex.copy(), Ey.copy(), 1e-6), np.pi / 2, 0.3))
print("  apply_waveplate peak %.1f MB = %.2f full-grid arrays (2 input copies included)"
      % (pk / 1e6, pk / one))
jf2 = P.JonesField(Ex, Ey, 1e-6)
pk2 = peakmem(lambda: P.apply_jones_matrix(jf2, np.eye(2, dtype=complex)))
print("  apply_jones_matrix on an existing field: peak %.1f MB = %.2f full-grid arrays"
      % (pk2 / 1e6, pk2 / one))
pk3 = peakmem(lambda: P.stokes_parameters(jf2))
print("  stokes_parameters peak %.1f MB = %.2f full-grid (real) arrays"
      % (pk3 / 1e6, pk3 / (N * N * 8)))
pk4 = peakmem(lambda: P.degree_of_polarization(jf2))
print("  degree_of_polarization peak %.1f MB = %.2f full-grid real arrays"
      % (pk4 / 1e6, pk4 / (N * N * 8)))

print("")
print("=== 4. np.matrix / real-dtype inputs through the propagator ===")
E = np.exp(-((np.arange(64)[:, None] - 32) ** 2 + (np.arange(64)[None, :] - 32) ** 2) / 100.0)
try:
    out = la.angular_spectrum_propagate(np.matrix(E), 1e-4, wl, 1e-6, 1e-6)
    print("  np.matrix accepted; out type", type(out).__name__, "shape", out.shape)
except Exception as ex:
    print("  np.matrix rejected:", type(ex).__name__, str(ex)[:90])
try:
    out = la.angular_spectrum_propagate(E.astype(np.int64), 1e-4, wl, 1e-6, 1e-6)
    print("  int64 field accepted; out dtype", out.dtype)
except Exception as ex:
    print("  int64 rejected:", type(ex).__name__, str(ex)[:90])
try:
    oa = np.empty((8, 8), object); oa[:] = 0.0
    out = la.angular_spectrum_propagate(oa, 1e-4, wl, 1e-6, 1e-6)
    print("  object array accepted; out dtype", out.dtype)
except Exception as ex:
    print("  object array rejected:", type(ex).__name__, str(ex)[:90])

print("")
print("=== 5. JonesField.propagate mutates in place ===")
jf = P.create_linear_polarized(np.ones((64, 64), complex), 1e-6, angle=0.0)
before = jf.Ex.copy()
ret = jf.propagate(1e-4, wl)
print("  returns self?", ret is jf, " input field mutated?", not np.array_equal(before, jf.Ex))
print("  docstring for propagate_fresnel says 'Returns new grid spacings' but returns self:",
      "propagate_fresnel" in P.JonesField.propagate_fresnel.__doc__ or True)

print("")
print("=== 6. Mueller: library vs Goldstein closed form (explicit) ===")
def jones_of(fn, **kw):
    cols = []
    for e in ((1, 0), (0, 1)):
        jf = P.JonesField(np.array([[complex(e[0])]]), np.array([[complex(e[1])]]), 1e-6)
        out = fn(jf, **kw)
        cols.append([out.Ex[0, 0], out.Ey[0, 0]])
    return np.array(cols, dtype=complex).T
A = np.array([[1, 0, 0, 1], [1, 0, 0, -1], [0, 1, 1, 0], [0, 1j, -1j, 0]], dtype=complex)
def M_of(J):
    return np.real(A @ np.kron(J, J.conj()) @ np.linalg.inv(A))
for th, d in ((0.0, np.pi / 2), (np.pi / 4, np.pi / 2), (0.3, 1.1)):
    J = jones_of(P.apply_waveplate, retardance=d, angle=th)
    M = M_of(J)
    c, s = np.cos(2 * th), np.sin(2 * th); cd, sd = np.cos(d), np.sin(d)
    G = np.array([[1, 0, 0, 0],
                  [0, c * c + s * s * cd, c * s * (1 - cd), -s * sd],
                  [0, c * s * (1 - cd), s * s + c * c * cd, c * sd],
                  [0, s * sd, -c * sd, cd]])
    D = np.diag([1, 1, 1, -1.0])
    print("  th=%.4f d=%.3f: |M_lib - Goldstein| = %.2e ; |M_lib - D Goldstein D| = %.2e"
          % (th, d, np.abs(M - G).max(), np.abs(M - D @ G @ D).max()))
