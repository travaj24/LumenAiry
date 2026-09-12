"""Probe 10 + 4 + 7: cross-model consistency and ENERGY conservation through
the public apply_real_lens at N=1024 on a biconvex N-BK7 singlet."""
import numpy as np, sys, time, tracemalloc
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import apply_real_lens

lam = 0.55e-6
N = 1024
dx = 4.0e-6
ap = 3.0e-3
rx = dict(surfaces=[
            dict(radius=+19.6e-3, conic=0.0, glass_before='AIR', glass_after='N-BK7'),
            dict(radius=-27.4e-3, conic=0.0, glass_before='N-BK7', glass_after='AIR')],
          thicknesses=[2.5e-3], aperture_diameter=ap)

axis = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(axis, axis)
w0 = ap/3.0
E0 = np.exp(-(X**2+Y**2)/w0**2).astype(np.complex128)
P0 = float(np.sum(np.abs(E0)**2))
# apertured input power (the models all aperture first)
Pap = float(np.sum(np.abs(np.where(X**2+Y**2 <= (ap/2)**2, E0, 0))**2))

res = {}
for name, kw in [('thin', {}),
                 ('displaced', dict(surface_model='displaced')),
                 ('disp_remap', dict(surface_model='displaced', displaced_mode='remap')),
                 ('disp_split', dict(surface_model='displaced', displaced_mode='split')),
                 ('tangent_facet', dict(surface_model='tangent_facet')),
                 ('tf_remap', dict(surface_model='tangent_facet_remap'))]:
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        E = apply_real_lens(E0.copy(), prescription=rx, wavelength=lam, dx=dx, **kw)
    except Exception as e:
        print(f"{name:14s} RAISED {type(e).__name__}: {str(e)[:160]}")
        tracemalloc.stop(); continue
    t1 = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    P = float(np.sum(np.abs(E)**2))
    res[name] = E
    print(f"{name:14s} t={t1-t0:7.3f}s peak={peak/2**20:8.1f}MiB  P/Pap={P/Pap:.6f}  "
          f"dtype={E.dtype}")

print()
print("pairwise exit-OPD rms difference (waves, piston+tilt removed, r<=ap/2*0.9):")
m = (X**2+Y**2) <= (0.9*ap/2)**2
names = list(res)
def unwrap_diff(a, b):
    ph = np.angle(a*np.conj(b))
    # both near-identical -> no unwrap needed if |ph|<pi
    f = ph/(2*np.pi)
    A = np.stack([np.ones(int(m.sum())), X[m], Y[m]], axis=1)
    c, *_ = np.linalg.lstsq(A, f[m], rcond=None)
    return float(np.sqrt(np.mean((f[m]-A@c)**2))), float(np.max(np.abs(ph[m])))
for i in range(len(names)):
    for j in range(i+1, len(names)):
        r, mx = unwrap_diff(res[names[i]], res[names[j]])
        # amplitude difference too
        a1 = np.abs(res[names[i]]); a2 = np.abs(res[names[j]])
        da = float(np.max(np.abs(a1-a2)[m])/max(np.max(a1[m]),1e-30))
        print(f"  {names[i]:14s} vs {names[j]:14s}  rms={r:10.3e} wv  maxphase={mx:7.3f} rad  dAmp_rel={da:.3e}")
