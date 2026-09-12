"""End-to-end: multibranch vs the single-valued traced path at the SAME
(exit-vertex) plane, for a flat rear surface and a curved rear surface."""
import warnings, numpy as np, lumenairy
from lumenairy.elements.lenses import apply_real_lens_traced
lam = 0.5876e-6; k0 = 2*np.pi/lam
N = 512; dx = 25e-6
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); r = np.hypot(X, Y)
E = np.exp(-(r/3.5e-3)**2).astype(np.complex128)

def run(R1, R2, tag):
    rx = lumenairy.make_singlet(R1=R1, R2=R2, d=5e-3, glass='N-BK7', aperture=12e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        A = np.asarray(apply_real_lens_traced(E, prescription=rx, wavelength=lam, dx=dx,
                                              amplitude_model='ray_density',
                                              on_undersample='silent'))
        B = np.asarray(apply_real_lens_traced(E, prescription=rx, wavelength=lam, dx=dx,
                                              caustic='multibranch', ray_subsample=2, amplitude_model='ray_density',
                                              on_undersample='silent'))
    m = (np.abs(A) > 0.02*np.abs(A).max()) & (np.abs(B) > 0)
    # phase difference with the global piston removed
    d = np.angle(B[m]*np.conj(A[m]))
    d = np.angle(np.exp(1j*(d - np.angle(np.sum(np.exp(1j*d))))))
    print('%-34s  pixels %6d   phase diff rms %8.4f rad (%.3f waves)  max %8.4f rad'
          % (tag, m.sum(), np.sqrt(np.mean(d**2)), np.sqrt(np.mean(d**2))/(2*np.pi),
             np.abs(d).max()))
    pA = float(np.sum(np.abs(A)**2)); pB = float(np.sum(np.abs(B)**2))
    print('%-34s  power single %.5g  multibranch %.5g  ratio %.4f'
          % ('', pA, pB, pB/pA))

run(25e-3, float('inf'), 'FLAT rear (R2=inf)')
run(float('inf'), -100e-3, 'CURVED rear (R1=inf,R2=-100mm)')
run(60e-3, -60e-3, 'CURVED rear (biconvex +-60mm)')
