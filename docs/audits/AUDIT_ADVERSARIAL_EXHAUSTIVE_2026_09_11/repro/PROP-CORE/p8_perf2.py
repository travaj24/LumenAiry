import numpy as np, sys, time, threading, gc, os
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators import fft_infra as fi
print("cpu count:", os.cpu_count(), "load-sensitive: many other processes may be running")

N=1024
x=(np.random.standard_normal((N,N))+1j*np.random.standard_normal((N,N)))
y=fi._fft2(x)
print("bad shapes after one call:", fi._PYFFTW_BAD_SHAPES)
plan,buf,lock,nb=fi._get_or_make_plan('fwd',(N,N),np.dtype(np.complex128),fi.FFTW_THREADS)
plan2,buf2,lock2,nb2=fi._get_or_make_plan('fwd',(N,N),np.dtype(np.complex128),fi.FFTW_THREADS)
print(f"  n_bufs={nb}; slot buffers distinct: {buf is not buf2}; LOCK OBJECT SHARED: {lock is lock2}")
print(f"  => two threads at the same (dir,shape,dtype,threads) key serialise on one lock regardless of the ping-pong")

# instrumented contention: count max concurrency inside the critical region
import lumenairy.propagators.fft_infra as F
cnt={'cur':0,'max':0}
cl=threading.Lock()
orig=F._get_or_make_plan
class WrapLock:
    def __init__(self,l): self.l=l
    def __enter__(self):
        self.l.acquire()
        with cl:
            cnt['cur']+=1; cnt['max']=max(cnt['max'],cnt['cur'])
        return self
    def __exit__(self,*a):
        with cl: cnt['cur']-=1
        self.l.release()
def patched(direction,shape,dtype,threads):
    p,b,l,n=orig(direction,shape,dtype,threads)
    return p,b,WrapLock(l),n
F._get_or_make_plan=patched
def work(a,reps):
    for _ in range(reps): F._fft2(a)
X=[np.random.standard_normal((N,N))+1j*np.random.standard_normal((N,N)) for _ in range(4)]
ths=[threading.Thread(target=work,args=(X[i],6)) for i in range(4)]
for t in ths: t.start()
for t in ths: t.join()
print(f"  MEASURED max simultaneous threads inside the pyFFTW critical section (4 threads, same shape): {cnt['max']}")
F._get_or_make_plan=orig

print()
print("=== ASM cold breakdown at N=2048 (H build vs FFT plan build) ===")
from lumenairy.propagators.asm import angular_spectrum_propagate as asm, _get_asm_H_natural
lam=1e-6; dx=0.5e-6; M=2048
E=(np.random.standard_normal((M,M))+1j*np.random.standard_normal((M,M)))
fi.clear_asm_caches()
t=time.perf_counter(); fi._fft2(E); t_plan=time.perf_counter()-t   # plan build
t=time.perf_counter(); H=_get_asm_H_natural(M,M,dx,dx,lam,1e-4,True,np.dtype(np.complex128),np); t_H=time.perf_counter()-t
t=time.perf_counter(); fi._fft2(E); t_fft=time.perf_counter()-t
t=time.perf_counter(); _=fi._ifft2(fi._fft2(E)*H); t_pair=time.perf_counter()-t
print(f"  pyFFTW plan build (1st fft2) {t_plan*1e3:8.1f} ms")
print(f"  H build (cold)               {t_H*1e3:8.1f} ms")
print(f"  warm fft2                    {t_fft*1e3:8.1f} ms")
print(f"  fft2+multiply+ifft2          {t_pair*1e3:8.1f} ms")
print(f"  => H build is {t_H/(t_H+t_pair)*100:.0f}% of a COLD (plan-warm) ASM call; docstring claims 30-50%")
print(f"  H nbytes = {H.nbytes/1e6:.1f} MB (complex128); field is complex128 too")
del E,H; gc.collect()

print()
print("=== full-grid temporaries in the default ASM path ===")
import tracemalloc
M=2048
E=(np.random.standard_normal((M,M))+1j*np.random.standard_normal((M,M)))
fi.clear_asm_caches()
gc.collect(); tracemalloc.start()
_=asm(E,1e-4,lam,dx)
cur,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
one=M*M*16/1e6
print(f"  N={M} complex128: one full grid = {one:.1f} MB ; traced PEAK during a cold ASM call = {peak/1e6:.1f} MB = {peak/1e6/one:.2f} grids")
fi.clear_asm_caches(); gc.collect(); tracemalloc.start()
_=asm(E,1e-4,lam,dx,stream_transfer_function=True)
cur,peak2=tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"  streamed: traced PEAK = {peak2/1e6:.1f} MB = {peak2/1e6/one:.2f} grids")
