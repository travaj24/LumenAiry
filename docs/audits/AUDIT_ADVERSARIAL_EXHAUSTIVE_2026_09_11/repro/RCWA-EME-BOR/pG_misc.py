import numpy as np, warnings, time, threading
t0 = time.perf_counter()
from lumenairy.elements.rcwa import (rcwa_efficiency_1d, RCWAStack,
                                     set_blas_threads, rcwa_blas_threads)
from lumenairy.elements.rcwa import _core as RC
print("import %.1f s" % (time.perf_counter() - t0), flush=True)

print("\n=== Q) is the Wood/grazing wavelength nudge SILENT? ===", flush=True)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    o, R, T = rcwa_efficiency_1d(1.0e-6, 2.04, 1.0, 1.0, 1.0, 1.0e-6, 0.5,
                                 1.0e-6, polarization='tm', n_orders=21)
print("   warnings emitted:", [str(x.message)[:100] for x in w], flush=True)
print(f"   R0 at exactly lam=Lam: {R[21]:.12f}", flush=True)
print(f"   _grazing_safe_wavelength(1e-6,...) = "
      f"{RC._grazing_safe_wavelength(1e-6, 0.0, 0.0, np.arange(-21,22), np.zeros(43), 1e-6, 1.0, [1.0,1.0,2.04**2,1.0]):.15e}",
      flush=True)

print("\n=== Q2) same via RCWAStack (2-D engine) ===", flush=True)
for wl in (1.0e-6, 1.0000001e-6, 0.9999999e-6):
    st = RCWAStack(1.0e-6, period_y=1.0e-6, n_superstrate=1.0, n_substrate=1.0,
                   n_orders=6, n_orders_y=0)
    cell = np.where(np.arange(64)/64 < 0.5, 2.04**2, 1.0)[:, None]
    st.add_layer(1.0e-6, eps_cell=np.repeat(cell, 1, axis=1))
    st.set_source(wl, theta=0.0)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = st.solve()
    oo, RR, TT = r.efficiencies()
    print(f"   wl={wl:.10e}: R(x,0)={np.asarray(RR)[0].max():.12f} "
          f"warn={[str(x.message)[:60] for x in w]}", flush=True)

print("\n=== R) BLAS-thread global state / thread safety ===", flush=True)
print("   _get_blas_threads() initially:", RC._get_blas_threads(), flush=True)
seen = {}


def worker(n, tag):
    set_blas_threads(n)
    time.sleep(0.05)
    seen[tag] = RC._get_blas_threads()


ts = [threading.Thread(target=worker, args=(k, k)) for k in (1, 2, 4)]
for t in ts:
    t.start()
for t in ts:
    t.join()
print("   per-thread settings after concurrent set:", seen, flush=True)
print("   main thread setting:", RC._get_blas_threads(), flush=True)
with rcwa_blas_threads(2):
    print("   inside ctx:", RC._get_blas_threads(), flush=True)
print("   after ctx:", RC._get_blas_threads(), flush=True)

print("\n=== S) _HOMOG_CACHE identity / read-only protection ===", flush=True)
RC._clear_rcwa_caches()
st = RCWAStack(0.5e-6, period_y=0.5e-6, n_superstrate=1.0, n_substrate=1.5,
               n_orders=3, n_orders_y=3)
st.add_layer(0.2e-6, eps=4.0)
st.set_source(0.633e-6, theta=0.2)
r1 = st.solve()
print("   cache size after 1 solve:", len(RC._HOMOG_CACHE), flush=True)
amp = r1.per_order_amplitudes()
try:
    amp["kz"] *= 2.0
    print("   MUTATION SUCCEEDED (cache poisoning possible!)", flush=True)
except ValueError as e:
    print("   mutation refused (read-only) OK:", str(e)[:60], flush=True)
except Exception as e:
    print("   other:", type(e).__name__, str(e)[:80], flush=True)

print("\n=== T) lossy-structure energy guard window ===", flush=True)
# a passive lossy grating: R+T must be <= 1.  Check the guard's blind window.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    o, R, T = rcwa_efficiency_1d(1.2e-6, 1.5+0.05j, 1.0, 1.5, 1.0, 0.3e-6, 0.5,
                                 0.633e-6, polarization='tm', n_orders=21)
print(f"   lossy: sumR+T = {R.sum()+T.sum():.12f}  warn={[str(x.message)[:50] for x in w]}",
      flush=True)

print("\n=== U) 1-D TM solve timing vs N (single-thread BLAS) ===", flush=True)
args = dict(period=0.5e-6, n_ridge=0.135+3.99j, n_groove=1.0, n_substrate=1.5,
            n_superstrate=1.0, depth=0.2e-6, duty_cycle=0.5,
            wavelength=0.6328e-6)
rcwa_efficiency_1d(**args, polarization='tm', n_orders=11)
for M in (25, 50, 100, 200):
    t = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rcwa_efficiency_1d(**args, polarization='tm', n_orders=M,
                           formulation='li')
    print(f"   M={M:3d} (N={2*M+1:3d}): {time.perf_counter()-t:7.4f} s",
          flush=True)
print("   -- component costs at N=401 --", flush=True)
N = 401
A = (np.random.randn(N, N) + 1j*np.random.randn(N, N)) + N*np.eye(N)
for nm, fn in (("eig", lambda: np.linalg.eig(A)),
               ("inv", lambda: np.linalg.inv(A)),
               ("solve", lambda: np.linalg.solve(A, A))):
    t = time.perf_counter(); fn(); print(f"   {nm}: {time.perf_counter()-t:.4f} s",
                                         flush=True)
