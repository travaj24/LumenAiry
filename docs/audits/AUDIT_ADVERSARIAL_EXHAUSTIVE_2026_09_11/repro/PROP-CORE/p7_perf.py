import numpy as np, sys, time, threading, gc
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators import fft_infra as fi
from lumenairy.propagators.asm import angular_spectrum_propagate as asm
import scipy.fft as sfft

print("backends: PYFFTW_AVAILABLE=%s USE_PYFFTW=%s FFTW_THREADS=%d SCIPY_WORKERS=%s"
      % (fi.PYFFTW_AVAILABLE, fi.USE_PYFFTW, fi.FFTW_THREADS, fi.SCIPY_FFT_WORKERS))

def bench(fn, n=5):
    fn(); fn()
    ts=[]
    for _ in range(n):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return min(ts)

for N in (1024, 2048, 4096):
    x=(np.random.standard_normal((N,N))+1j*np.random.standard_normal((N,N)))
    t_pf = bench(lambda: fi._fft2(x))
    t_sp = bench(lambda: sfft.fft2(x, workers=-1))
    t_np = bench(lambda: np.fft.fft2(x)) if N<=2048 else float('nan')
    print(f"  N={N}: lumenairy _fft2 {t_pf*1e3:8.2f} ms | scipy workers=-1 {t_sp*1e3:8.2f} ms | numpy {t_np*1e3:8.2f} ms")
    del x; gc.collect()

print()
print("=== lock contention: 2 threads doing _fft2 at the SAME shape/dtype ===")
N=1024
X=[np.random.standard_normal((N,N))+1j*np.random.standard_normal((N,N)) for _ in range(2)]
def work(a, reps):
    for _ in range(reps): fi._fft2(a)
reps=8
t=time.perf_counter(); work(X[0],reps); work(X[1],reps); t_ser=time.perf_counter()-t
ths=[threading.Thread(target=work,args=(X[i],reps)) for i in range(2)]
t=time.perf_counter()
for th in ths: th.start()
for th in ths: th.join()
t_par=time.perf_counter()-t
print(f"  serial 2x{reps} = {t_ser*1e3:.1f} ms ; 2 threads = {t_par*1e3:.1f} ms ; speedup = {t_ser/t_par:.3f}x")
# different shapes -> different keys -> different locks
Y=[np.random.standard_normal((N,N))+1j*np.random.standard_normal((N,N)),
   np.random.standard_normal((N+2,N+2))+1j*np.random.standard_normal((N+2,N+2))]
t=time.perf_counter(); work(Y[0],reps); work(Y[1],reps); t_ser2=time.perf_counter()-t
ths=[threading.Thread(target=work,args=(Y[i],reps)) for i in range(2)]
t=time.perf_counter()
for th in ths: th.start()
for th in ths: th.join()
t_par2=time.perf_counter()-t
print(f"  DIFFERENT shapes: serial={t_ser2*1e3:.1f} ms ; 2 threads={t_par2*1e3:.1f} ms ; speedup={t_ser2/t_par2:.3f}x")

print()
print("=== ASM cold vs warm (H cache value) at N=2048 ===")
lam=1e-6; dx=0.5e-6
E=(np.random.standard_normal((2048,2048))+1j*np.random.standard_normal((2048,2048)))
fi.clear_asm_caches()
t=time.perf_counter(); asm(E,1e-4,lam,dx); t_cold=time.perf_counter()-t
t_warm=bench(lambda: asm(E,1e-4,lam,dx),3)
print(f"  cold {t_cold*1e3:.1f} ms ; warm {t_warm*1e3:.1f} ms ; H build share = {(t_cold-t_warm)/t_cold*100:.0f}%")
t_stream=bench(lambda: asm(E,1e-4,lam,dx,stream_transfer_function=True),3)
print(f"  streamed (never cached) {t_stream*1e3:.1f} ms  -> {t_stream/t_warm:.2f}x the warm plain path")
del E; gc.collect()

print()
print("=== freq-grid divide-vs-multiply: can it flip a band-limit bin? ===")
from lumenairy.propagators.fft_infra import _get_or_make_freq_grids, _get_or_make_bandlimit
flips=0; tested=0
rng=np.random.default_rng(3)
for _ in range(400):
    N=int(rng.integers(64,1025)); dxv=float(rng.uniform(0.1,5.0))*1e-6
    lamv=float(rng.uniform(0.4,1.6))*1e-6; zv=float(rng.uniform(1e-5,1e-2))
    kxs,_=_get_or_make_freq_grids(N,N,dxv,dxv,True)
    fx_mul=np.sqrt(kxs)/(2*np.pi)*np.sign(np.arange(N)-N//2)
    fx_div=(np.arange(N)-N//2)/(N*dxv)
    fmax=N*dxv/(2*lamv*zv)
    m1=np.abs(fx_div)<fmax
    m2=np.abs(np.sqrt(kxs)/(2*np.pi))<fmax
    tested+=1
    if not np.array_equal(m1,m2): flips+=1
print(f"  random (N,dx,lam,z): {flips}/{tested} cases where the mask from the DIVIDED fx differs")
# direct ULP comparison
N=1000; dxv=1.3e-6
a=(np.arange(N)-N//2)*(1.0/(N*dxv)); b=(np.arange(N)-N//2)/(N*dxv)
print(f"  N={N} dx={dxv}: max ULP diff between mul and div fx = {np.max(np.abs(a-b)/np.spacing(np.abs(b)+1e-300)):.1f} ULP")
