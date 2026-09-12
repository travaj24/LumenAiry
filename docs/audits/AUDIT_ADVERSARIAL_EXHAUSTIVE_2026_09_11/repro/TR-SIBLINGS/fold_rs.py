"""Independent ray-to-wave (Huygens/ASM) oracle for a fold caustic, compared
against apply_real_lens_traced_multibranch / _uniform.

Oracle: exit-pupil field built from MY OWN ray trace (geometric optics is exact
at the pupil, far from any caustic), then propagated by an exact
angular-spectrum integral I write here (no lumenairy propagator).
"""
import sys, time, warnings, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
import lumenairy
from lumenairy.glass import get_glass_index
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch as MB)

lam = 1.0e-6
k = 2*np.pi/lam
ng = float(get_glass_index('N-BK7', lam))
R1, R2, d, ap = 2.0e-3, float('inf'), 0.5e-3, 1.4e-3
D_OUT = 3.3e-3
N, dx = 4096, 0.6e-6
rx = lumenairy.make_singlet(R1=R1, R2=R2, d=d, glass='N-BK7', aperture=ap)

x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
r = np.hypot(X, Y)

# --- input: flat top r<=0.66mm with a 40 um raised-cosine edge -----------
r0, w = 0.66e-3, 40e-6
amp = np.clip((r0 + w - r)/w, 0.0, 1.0)
amp = 0.5*(1-np.cos(np.pi*amp))
E_in = amp.astype(np.complex128)

# --- ORACLE: exit-pupil field from my own ray trace ----------------------
h = np.linspace(0.0, 0.699e-3, 400001)
h[0] = 1e-12
xo0, opl0, L0, Nz0 = trace_singlet(h, R1, R2, d, ng, 0.0)   # to exit vertex
rho = np.abs(xo0)
assert np.all(np.diff(rho) > 0), 'exit height must be monotone at the pupil'
dh_drho = np.gradient(h, rho)
amp_p = np.interp(r.ravel(), rho, np.sqrt(np.abs(h*dh_drho/np.maximum(rho, 1e-15))),
                  left=np.nan, right=0.0).reshape(N, N)
amp_in = np.interp(r.ravel(), rho, np.interp(h, np.sort(r[N//2, N//2:]),
                   amp[N//2, N//2:]) if False else 0*h, left=0, right=0)  # placeholder
# input amplitude AT the launch height h (radial profile of `amp`)
rad = np.hypot(x[N//2:], 0.0)
amp_rad = amp[N//2, N//2:]
Ain_of_h = np.interp(h, rad, amp_rad, left=amp_rad[0], right=0.0)
A_p = np.interp(r.ravel(), rho, Ain_of_h*np.sqrt(np.abs(h*dh_drho/np.maximum(rho, 1e-15))),
                left=0.0, right=0.0).reshape(N, N)
OPL_p = np.interp(r.ravel(), rho, opl0, left=opl0[0], right=opl0[-1]).reshape(N, N)
E_pup = A_p*np.exp(1j*k*(OPL_p-opl0[0]))

# --- exact angular-spectrum propagation (written here) -------------------
def asm(E, dx, lam, z):
    n = E.shape[0]
    fx = np.fft.fftfreq(n, dx)
    FX, FY = np.meshgrid(fx, fx)
    k2 = (1.0/lam)**2 - FX**2 - FY**2
    kz = 2*np.pi*np.sqrt(np.maximum(k2, 0.0))
    prop = np.where(k2 > 0, np.exp(1j*kz*z), 0.0)
    # Matsushima band-limit
    fl = n*dx/(2.0*z*lam)*(1.0/lam)/np.sqrt((n*dx/(2*z))**2 + (1.0/lam)**2)
    prop = np.where((np.abs(FX) <= fl) & (np.abs(FY) <= fl), prop, 0.0)
    return np.fft.ifft2(np.fft.fft2(E)*prop)

t0 = time.time()
E_ref = asm(E_pup, dx, lam, D_OUT)
print('ASM oracle %.1f s' % (time.time()-t0))

# --- library multibranch --------------------------------------------------
t0 = time.time()
with warnings.catch_warnings(record=True) as ws:
    warnings.simplefilter('always')
    E_mb, diag = MB(E_in, prescription=rx, wavelength=lam, dx=dx,
                    output_plane_distance=D_OUT, ray_subsample=4,
                    return_diagnostics=True)
print('multibranch %.1f s' % (time.time()-t0))
for wv in ws:
    print('  WARN', str(wv.message)[:160])
print('n_branch max', diag['n_branch'].max(), 'kmah vals', np.unique(diag['kmah']))

np.savez(r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS/fold_fields.npz",
         E_ref=E_ref[N//2-200:N//2+200, N//2-200:N//2+200],
         E_mb=np.asarray(E_mb)[N//2-200:N//2+200, N//2-200:N//2+200],
         E_pup_row=E_pup[N//2], dx=dx, lam=lam, d_out=D_OUT)

# --- radial comparison ----------------------------------------------------
row = N//2
rr = x[N//2:N//2+120]
Ir = np.abs(E_ref[row, N//2:N//2+120])**2
Im = np.abs(np.asarray(E_mb)[row, N//2:N//2+120])**2
print('\n r(um)   I_oracle      I_multibranch   ratio')
for i in range(0, 120, 4):
    print('%7.2f  %12.5g  %12.5g  %8.3f' % (rr[i]*1e6, Ir[i], Im[i],
          Im[i]/Ir[i] if Ir[i] > 0 else np.nan))
m = Ir > 0.02*Ir.max()
print('\nwindowed (I>2%% peak) rms rel err %.4f ; peak oracle %.5g at %.2f um; '
      'peak mb %.5g at %.2f um'
      % (np.sqrt(np.mean(((Im[m]-Ir[m])/Ir.max())**2)),
         Ir.max(), rr[Ir.argmax()]*1e6, Im.max(), rr[Im.argmax()]*1e6))
p_ref = float(np.sum(np.abs(E_ref)**2)); p_mb = float(np.sum(np.abs(np.asarray(E_mb))**2))
p_pup = float(np.sum(np.abs(E_pup)**2)); p_in = float(np.sum(np.abs(E_in)**2))
print('power: in %.6g  pupil(oracle) %.6g  ASM out %.6g  multibranch out %.6g'
      % (p_in, p_pup, p_ref, p_mb))
