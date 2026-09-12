"""Probe 9: FGA free-space vs analytic Gaussian; through a lens vs ASM oracle."""
import numpy as np, sys, warnings, time
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.propagators.fga import apply_real_lens_fga
from lumenairy.propagators.asm import angular_spectrum_propagate
warnings.simplefilter("ignore")

def gap(z, ap):
    return {'name':'gap','aperture_diameter':ap,
            'surfaces':[{'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'},
                        {'radius':np.inf,'conic':0.,'glass_before':'air','glass_after':'air'}],
            'thicknesses':[z]}

lam = 1.0e-6; k = 2*np.pi/lam
N, dx = 96, 2.0e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
w0 = 25e-6; zR = np.pi*w0**2/lam
E0 = np.exp(-(X**2+Y**2)/w0**2).astype(complex)
print(f"zR = {zR*1e6:.1f} um")

def analytic(z):
    r2 = X**2+Y**2
    w = w0*np.sqrt(1+(z/zR)**2); psi = np.arctan2(z, zR)
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(k*z + k*r2*z/(2*(z**2+zR**2)) - psi))

print("\n### FGA free-space (flat air gap) vs analytic Gaussian, normalize_output='none'")
for z in (0.5*zR, 2*zR, 6*zR):
    t=time.perf_counter()
    E = np.asarray(apply_real_lens_fga(E0.copy(), prescription=gap(z, N*dx*0.9),
                                       wavelength=lam, dx=dx, normalize_output='none'))
    A = analytic(z)
    m = np.abs(A) > 0.05*np.abs(A).max()
    r = np.vdot(A[m], E[m])/np.vdot(A[m], A[m])
    fid = abs(np.vdot(A,E))**2/(np.vdot(A,A).real*np.vdot(E,E).real)
    pw = float(np.sum(np.abs(E)**2)/np.sum(np.abs(E0)**2))
    print(f"  z={z/zR:5.2f} zR: fidelity={fid:.6f}  complex scale={r:.5f} (|.|={abs(r):.5f},"
          f" arg={np.angle(r):+.4f})  power ratio={pw:.5f}   relL2(after scale)="
          f"{np.linalg.norm(E-r*A)/np.linalg.norm(r*A):.3e}  {time.perf_counter()-t:.1f}s")

print("\n### FGA through a singlet vs ASM oracle (exit plane)")
presc = la.make_singlet(3.0e-3, -3.0e-3, 0.5e-3, 'N-BK7', aperture=0.15e-3)
Ein = np.exp(-(X**2+Y**2)/(60e-6)**2).astype(complex)
Ef = np.asarray(apply_real_lens_fga(Ein.copy(), prescription=presc, wavelength=lam, dx=dx,
                                    normalize_output='none'))
Eg = np.asarray(la.apply_real_lens_gbd(Ein.copy(), prescription=presc, wavelength=lam, dx=dx))
Ea = np.asarray(la.apply_real_lens(Ein.copy(), prescription=presc, wavelength=lam, dx=dx))
for nm, E in (('fga', Ef), ('gbd', Eg)):
    fid = abs(np.vdot(Ea,E))**2/(np.vdot(Ea,Ea).real*np.vdot(E,E).real)
    print(f"  {nm}: fidelity vs analytic = {fid:.6f}  power ratio (vs input) = "
          f"{float(np.sum(np.abs(E)**2)/np.sum(np.abs(Ein)**2)):.5f}")
