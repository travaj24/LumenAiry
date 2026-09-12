"""P1 repro: apply_real_lens_traced_uniform's dark-side fill is `rgrid > r_c`
UNCONDITIONALLY, where r_c is the |x_out| of the SIGNED turning point.  When the
observation plane is past the marginal focus the ray map re-crosses the axis, so
real single-branch rays land at |x_out| > r_c -- and the uniform fill ERASES that
bright annulus and replaces it with an exponentially decaying Airy tail.
"""
import sys, time, warnings, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
import lumenairy
from lumenairy.glass import get_glass_index
from lumenairy.elements._lens_traced_multibranch import apply_real_lens_traced_multibranch as MB
from lumenairy.elements._lens_traced_uniform import apply_real_lens_traced_uniform as UNI

lam = 1.0e-6; ng = float(get_glass_index('N-BK7', lam))
R1, R2, d = 1.2e-3, float('inf'), 0.4e-3
LR = 0.6e-3; ap = 2*LR/0.98
D_OUT = 2.0e-3                      # PAST the marginal focus
N, dx = 2048, 0.7e-6
rx = lumenairy.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7', aperture=ap)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); r = np.hypot(X, Y)
t_ = np.clip((0.56e-3+30e-6-r)/30e-6, 0, 1); amp = 0.5*(1-np.cos(np.pi*t_))
E_in = amp.astype(np.complex128)

h = np.linspace(1e-12, LR, 200001)
xo, opl, L, Nz = trace_singlet(h, R1, R2, d, ng, D_OUT)
i = np.argmax(xo)
print('signed turning point: x_out=%.3f um at h=%.4f mm' % (xo[i]*1e6, h[i]*1e3))
print('max |x_out| over all live rays: %.3f um (edge ray %.3f um)'
      % (np.abs(xo).max()*1e6, xo[-1]*1e6))

kw = dict(prescription=rx, wavelength=lam, dx=dx, output_plane_distance=D_OUT,
          ray_subsample=4, return_diagnostics=True)
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always')
    Emb, dmb = MB(E_in, **kw)
    t0 = time.time(); Eun, dun = UNI(E_in, **kw); t_uni = time.time()-t0
for w in ws: print('  WARN', str(w.message)[:150])
print('uniform: reason=%r fell_back=%r r_c=%.4g um kappa=%.4g  (%.1f s)'
      % (dun.get('reason'), dun.get('fell_back'),
         (dun.get('r_c') or 0)*1e6, dun.get('kappa') or 0, t_uni))
row = N//2
Imb = np.abs(np.asarray(Emb)[row, N//2:])**2
Iun = np.abs(np.asarray(Eun)[row, N//2:])**2
rr = x[N//2:]
print('\n r(um)    I_multibranch     I_uniform      ratio')
for rt in np.arange(0, 34, 2)*1e-6:
    j = int(np.argmin(np.abs(rr-rt)))
    print('%7.2f %14.5g %13.5g %10.3g' % (rr[j]*1e6, Imb[j], Iun[j],
          (Iun[j]/Imb[j]) if Imb[j] > 0 else np.nan))
band = (rr > (dun['r_c'] or 0)) & (rr < 23.5e-6)
print('\nannulus r_c < r < 23.5um  (real single-branch rays land here):')
print('   multibranch power in annulus %.6g   uniform %.6g   kept %.3g %%'
      % (Imb[band].sum(), Iun[band].sum(), 100*Iun[band].sum()/Imb[band].sum()))
print('   total grid power: mb %.6g  uniform %.6g  (in %.6g)'
      % (np.sum(np.abs(Emb)**2), np.sum(np.abs(Eun)**2), np.sum(np.abs(E_in)**2)))
