import numpy as np, sys, warnings, time, tracemalloc
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
warnings.simplefilter('ignore')
from lumenairy.elements.doe import create_kinoform, create_diffractive_lens
from lumenairy.elements.lenses import surface_sag_general
import lumenairy.elements.lenses as L

print("=== kinoform quantisation: floor vs round, blazed-grating efficiency ===", flush=True)
M=8192; per=1024
ramp = 2*np.pi*np.arange(M)/per            # +1-order blaze
for Lv in (2,4,8,16):
    step=2*np.pi/Lv
    phi=np.mod(ramp,2*np.pi)
    q_floor=np.floor(phi/step)*step
    q_round=np.mod(np.round(phi/step)*step, 2*np.pi)
    ef=abs(np.fft.fft(np.exp(1j*q_floor))[M//per])**2/M**2
    er=abs(np.fft.fft(np.exp(1j*q_round))[M//per])**2/M**2
    print(f"  L={Lv:2d}: eta1(floor)={ef:.6f}  eta1(round)={er:.6f}  sinc^2(1/L)={np.sinc(1.0/Lv)**2:.6f}", flush=True)

print("\n=== create_kinoform 1st-order efficiency at the design focus (2-D) ===", flush=True)
import lumenairy as la
N=1024; dx=2e-6; f=20e-3; wl=1e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); E0=np.exp(-(X**2+Y**2)/(0.8e-3)**2).astype(complex)
P_in = np.sum(np.abs(E0)**2)
for nl in (2,4,8,16,1024):
    T=create_kinoform(N,dx,f,wl,n_levels=nl)
    Ez=la.angular_spectrum_propagate(E0*T, f, wl, dx)
    I=np.abs(Ez)**2
    # power within 3 Airy radii of the axis
    R=np.hypot(X,Y); rr=3*1.22*wl*f/(2*0.8e-3)
    print(f"  n_levels={nl:5d}: on-axis Imax={I.max():.4g}  core power frac={I[R<rr].sum()/P_in:.4f}"
          f"  (sinc^2(1/L)={np.sinc(1.0/nl)**2:.4f})", flush=True)

print("\n=== PERF: surface_sag_general at N=4096, 4 aspheric terms ===", flush=True)
Ngrid=4096
x=np.linspace(-25e-3,25e-3,Ngrid); X,Y=np.meshgrid(x,x); hsq=(X*X+Y*Y)
co={4:1.0e3, 6:-2.0e6, 8:3.0e10, 10:-4.0e13}
def bench(fn, n=3):
    ts=[]
    for _ in range(n):
        t0=time.perf_counter(); fn(); ts.append(time.perf_counter()-t0)
    return min(ts)
surface_sag_general(hsq[:8,:8].copy(), 50e-3, -0.5, dict(co))   # warm numba
t_nb = bench(lambda: surface_sag_general(hsq, 50e-3, -0.5, dict(co)))
L._NUMBA_AVAILABLE=False
t_np = bench(lambda: surface_sag_general(hsq, 50e-3, -0.5, dict(co)))
L._NUMBA_AVAILABLE=True
print(f"  numba kernel : {t_nb*1e3:8.1f} ms", flush=True)
print(f"  numpy loop   : {t_np*1e3:8.1f} ms   speedup {t_np/t_nb:.2f}x", flush=True)
# Horner-on-h^2 alternative
def horner(hsq, co):
    ps=sorted(co); acc=np.zeros_like(hsq)
    # evaluate sum a_p (h^2)^(p//2) by Horner in h^2
    maxp=max(ps)//2
    c=[co.get(2*i,0.0) for i in range(maxp+1)]
    out=np.full_like(hsq, c[maxp])
    for i in range(maxp-1,-1,-1):
        out*=hsq; out+=c[i]
    return out*hsq**0   # (already includes const term)
t_h = bench(lambda: horner(hsq, co))
print(f"  Horner(h^2) aspheric-only reference: {t_h*1e3:8.1f} ms", flush=True)
tracemalloc.start()
L._NUMBA_AVAILABLE=False
_=surface_sag_general(hsq, 50e-3, -0.5, dict(co)); cur,pk=tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"  numpy-path peak traced alloc: {pk/2**20:.1f} MiB (one {Ngrid}^2 f64 grid = {Ngrid**2*8/2**20:.1f} MiB)", flush=True)
tracemalloc.start()
L._NUMBA_AVAILABLE=True
_=surface_sag_general(hsq, 50e-3, -0.5, dict(co)); cur,pk2=tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"  numba-path peak traced alloc: {pk2/2**20:.1f} MiB", flush=True)

print("\n=== PERF: create_microlens_array ===", flush=True)
from lumenairy.elements.doe import create_microlens_array
for Ng in (2048, 4096):
    t0=time.perf_counter(); _=create_microlens_array(Ng,1e-6,64,60e-6,2e-3,1e-6); t1=time.perf_counter()
    print(f"  N={Ng}: {t1-t0:.3f}s  ({Ng**2*16/2**20:.0f} MiB output)", flush=True)
tracemalloc.start()
_=create_microlens_array(2048,1e-6,64,60e-6,2e-3,1e-6)
cur,pk=tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"  N=2048 peak traced alloc: {pk/2**20:.1f} MiB (output alone = {2048**2*16/2**20:.1f} MiB)", flush=True)
