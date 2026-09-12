"""Fold-caustic comparison: independent ray-to-wave ASM oracle vs
apply_real_lens_traced_multibranch and apply_real_lens_traced_uniform.
Plano-convex, FLAT rear (so the missing exit-vertex correction is inert).
"""
import sys, time, warnings, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
import lumenairy
from lumenairy.glass import get_glass_index
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch as MB)
from lumenairy.elements._lens_traced_uniform import (
    apply_real_lens_traced_uniform as UNI)

lam = 1.0e-6; k = 2*np.pi/lam
ng = float(get_glass_index('N-BK7', lam))
R1, R2, d = 1.2e-3, float('inf'), 0.4e-3
LR = 0.6e-3
ap = 2*LR/0.98
D_OUT = 1.9e-3
N, dx = 4096, 0.35e-6
rx = lumenairy.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7', aperture=ap)
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x); r = np.hypot(X, Y)
r0, w = 0.56e-3, 30e-6
t_ = np.clip((r0+w-r)/w, 0, 1); amp = 0.5*(1-np.cos(np.pi*t_))
E_in = amp.astype(np.complex128)

# ---- oracle exit-pupil field (my own rays) ----
h = np.linspace(1e-12, LR, 600001)
xo0, opl0, L0, Nz0 = trace_singlet(h, R1, R2, d, ng, 0.0)
rho = np.abs(xo0)
assert np.all(np.diff(rho) > 0)
dh_drho = np.gradient(h, rho)
rad = x[N//2:]; amp_rad = amp[N//2, N//2:]
Ain = np.interp(h, rad, amp_rad, left=amp_rad[0], right=0.0)
Ap = Ain*np.sqrt(np.abs(h*dh_drho/np.maximum(rho, 1e-18)))
A_p = np.interp(r.ravel(), rho, Ap, left=0.0, right=0.0).reshape(N, N)
OPL_p = np.interp(r.ravel(), rho, opl0-opl0[0], left=0.0, right=opl0[-1]-opl0[0]).reshape(N, N)
E_pup = A_p*np.exp(1j*k*OPL_p)

def asm(E, dx, lam, z):
    n = E.shape[0]; fx = np.fft.fftfreq(n, dx)
    FX, FY = np.meshgrid(fx, fx)
    k2 = (1.0/lam)**2 - FX**2 - FY**2
    prop = np.where(k2 > 0, np.exp(2j*np.pi*np.sqrt(np.maximum(k2, 0.0))*z), 0.0)
    fl = (1.0/lam)/np.sqrt((2*z/(n*dx))**2 + 1.0)
    prop *= (np.abs(FX) <= fl) & (np.abs(FY) <= fl)
    return np.fft.ifft2(np.fft.fft2(E)*prop)

t0 = time.time(); E_ref = asm(E_pup, dx, lam, D_OUT); print('ASM %.1fs' % (time.time()-t0))
del A_p, OPL_p, E_pup

res = {}
for tag, fn, kw in (('multibranch', MB, {}), ('uniform', UNI, {})):
    t0 = time.time()
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        out = fn(E_in, prescription=rx, wavelength=lam, dx=dx,
                 output_plane_distance=D_OUT, ray_subsample=4,
                 return_diagnostics=True, **kw)
    E, dg = out
    res[tag] = np.asarray(E)
    print('%s %.1fs  n_branch max %d  reason=%s fell_back=%s'
          % (tag, time.time()-t0, dg['n_branch'].max(),
             dg.get('reason'), dg.get('fell_back')))
    for wv in ws: print('   WARN', str(wv.message)[:170])

row = N//2; sl = slice(N//2, N//2+120)
rr = x[sl]
Ir = np.abs(E_ref[row, sl])**2
out = {t_: np.abs(res[t_][row, sl])**2 for t_ in res}
print('\n r(um)     I_oracle   I_multibranch     I_uniform')
for i in range(0, 120, 3):
    print('%7.2f %12.5g %14.5g %13.5g' % (rr[i]*1e6, Ir[i],
          out['multibranch'][i], out['uniform'][i]))
pk = Ir.max()
for t_ in ('multibranch', 'uniform'):
    m = rr <= 21.9e-6
    print('%-12s bright-side (r<r_c) rms rel err %.4f ; all r<42um rms %.4f'
          % (t_, np.sqrt(np.mean(((out[t_][m]-Ir[m])/pk)**2)),
             np.sqrt(np.mean(((out[t_][rr <= 42e-6]-Ir[rr <= 42e-6])/pk)**2))))
print('power oracle %.6g  mb %.6g  uni %.6g  in %.6g'
      % (np.sum(np.abs(E_ref)**2), np.sum(np.abs(res['multibranch'])**2),
         np.sum(np.abs(res['uniform'])**2), np.sum(np.abs(E_in)**2)))
np.savez(r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/fold2.npz",
         rr=rr, Ir=Ir, Imb=out['multibranch'], Iun=out['uniform'])
