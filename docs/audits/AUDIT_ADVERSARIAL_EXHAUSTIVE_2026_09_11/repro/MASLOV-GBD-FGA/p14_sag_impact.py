"""Probe 14: exit-plane phase of apply_real_lens_maslov vs apply_real_lens /
apply_real_lens_traced, for a FLAT vs CURVED last surface."""
import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
warnings.simplefilter("ignore")
lam, N, dx = 1.0e-6, 96, 6.0e-6
x=(np.arange(N)-N/2)*dx; X,Y=np.meshgrid(x,x); R2=X**2+Y**2
ap = 0.30e-3
E0 = ((R2 <= (0.5*ap)**2)*np.exp(-R2/(0.25e-3)**2)).astype(complex)

def sag_of(R, r2):
    if not np.isfinite(R): return np.zeros_like(r2)
    c=1.0/R
    return c*r2/(1+np.sqrt(np.maximum(1-c*c*r2,0.0)))

for label, R1, R2s in (("plano-convex, FLAT last surface ", 2.0e-3, np.inf),
                       ("plano-convex, CURVED last surf  ", np.inf, -2.0e-3)):
    presc = la.make_singlet(R1, R2s, 0.7e-3, 'N-BK7', aperture=ap)
    Em = np.asarray(la.apply_real_lens_maslov(E0.copy(), prescription=presc,
            wavelength=lam, dx=dx, normalize_output='power', collimated_input=True,
            integration_method='quadrature', n_v2=96, poly_order=5,
            ray_field_samples=12, ray_pupil_samples=12))
    Ea = np.asarray(la.apply_real_lens(E0.copy(), prescription=presc,
                                       wavelength=lam, dx=dx))
    m = (np.abs(Ea) > 0.05*np.abs(Ea).max()) & (np.abs(Em) > 0)
    d = np.angle(Em[m]*np.conj(Ea[m]))
    d = np.angle(np.exp(1j*(d - np.angle(np.vdot(Ea[m], Em[m])))))
    sg = sag_of(R2s, R2[m])
    print(f"{label}: phase resid (maslov - analytic) RMS = {np.sqrt(np.mean(d**2)):.4f} rad, "
          f"PV = {np.ptp(d):.4f} rad  |  k*|sag| over mask: RMS="
          f"{np.sqrt(np.mean((2*np.pi/lam*sg)**2)):.1f} rad PV={np.ptp(2*np.pi/lam*sg):.1f} rad")
    # correlation of the residual with the sag term (after unwrapping via cos)
    if np.isfinite(R2s):
        pred = np.angle(np.exp(1j*(2*np.pi/lam*(-sg))))
        pred = np.angle(np.exp(1j*(pred-np.mean(pred))))
        print(f"      corrcoef(resid, wrapped k*(-sag)) = "
              f"{np.corrcoef(np.cos(d), np.cos(pred))[0,1]:+.4f}")
