"""Probe 4: per-surface GBD Collins amplitude sign vs apply_real_lens_traced."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
import lumenairy.propagators.gbd as G

wl, dx, N = 1.31e-6, 8e-6, 256
xs = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(xs, xs)

def run(presc, E0, label):
    Et = np.asarray(la.apply_real_lens_traced(E0.copy(), prescription=presc,
                                              wavelength=wl, dx=dx, ray_subsample=2))
    Ea = np.asarray(la.apply_real_lens(E0.copy(), prescription=presc,
                                       wavelength=wl, dx=dx))
    out = {}
    for name in ('as-is','conj-collins'):
        if name == 'conj-collins':
            d0, e0 = G._det2x2, G._eigvals2x2
            G._det2x2 = lambda M, _d=d0: np.conj(_d(M))
            G._eigvals2x2 = lambda M, xp, _e=e0: np.conj(_e(M, xp))
        try:
            Eg = np.asarray(la.apply_real_lens_gbd(E0.copy(), prescription=presc,
                                                  wavelength=wl, dx=dx))
        finally:
            if name == 'conj-collins':
                G._det2x2, G._eigvals2x2 = d0, e0
        out[name] = Eg
    for name, Eg in out.items():
        for ref, rn in ((Et,'traced'), (Ea,'analytic')):
            m = np.abs(ref) > 0.05*np.abs(ref).max()
            ov = abs(np.vdot(ref[m], Eg[m]))**2/(np.vdot(ref[m],ref[m]).real*np.vdot(Eg[m],Eg[m]).real)
            ph = np.angle(Eg[m]*np.conj(ref[m]))
            ph = np.angle(np.exp(1j*(ph - np.angle(np.vdot(ref[m],Eg[m])))))
            print(f"  {label} {name:13s} vs {rn:9s}: overlap={ov:.6f}  "
                  f"resid phase RMS(after global)={np.sqrt(np.mean(ph**2)):.4f} rad  PV={ph.max()-ph.min():.4f}")
    return out

# slow lens, uniform disk input
presc = la.make_singlet(100e-3, -100e-3, 3e-3, 'N-BK7', aperture=1.5e-3)
E0 = ((X**2+Y**2) <= (0.7e-3)**2).astype(complex)
run(presc, E0, "F/67 ")

# faster lens
presc2 = la.make_singlet(12e-3, -12e-3, 3e-3, 'N-BK7', aperture=1.5e-3)
run(presc2, E0, "fast ")
