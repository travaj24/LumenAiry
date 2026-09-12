"""RS / hf near-field kernel aliasing scan + bandlimit no-op proof."""
import sys
import numpy as np
from scipy.special import j0
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.asm import angular_spectrum_propagate
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate

lam = 633e-9; k = 2*np.pi/lam
N, dx, w0 = 128, 1e-6, 10e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x); Rr = np.hypot(X, Y)
E0 = np.exp(-(Rr/w0)**2).astype(np.complex128)
runif = np.unique(np.round(Rr.ravel(), 12))
inv = np.searchsorted(runif, np.round(Rr.ravel(), 12))


def oracle(z, nf=6000):
    fmax = min(8.0/(np.pi*w0), 0.999999/lam)
    fn, fw = np.polynomial.legendre.leggauss(nf)
    fv = 0.5*fmax*(fn+1.0); fwt = 0.5*fmax*fw
    wgt = (2*np.pi*np.pi*w0**2*np.exp(-(np.pi*w0*fv)**2)
           * np.exp(1j*k*z*np.sqrt(np.maximum(1-(lam*fv)**2, 0.0)))*fv*fwt)
    out = np.empty(runif.shape, dtype=complex)
    for i0 in range(0, runif.size, 400):
        out[i0:i0+400] = j0(2*np.pi*np.outer(runif[i0:i0+400], fv)) @ wgt
    return out[inv].reshape(N, N)


L2 = N*dx                     # padded half-extent (padded grid is 2N)
print(f"padded grid 2N={2*N}, half-extent L2/2 = N*dx = {L2*1e6:.0f} um")
print(f"grid Nyquist frequency = 1/(2 dx) = {1/(2*dx):.3e} 1/m")
print()
print(f"{'z (um)':>8} {'kernel rad/px':>14} {'RS bl=F':>11} {'RS bl=T':>11}"
      f" {'ASM bl=F':>11} {'bl cutoff 1/m':>14} {'mask all-pass?':>15}"
      f" {'bl=T==bl=F?':>12}")
for z in (25e-6, 50e-6, 100e-6, 200e-6, 300e-6, 400e-6, 500e-6, 700e-6, 1e-3, 3e-3):
    REF = oracle(z)
    a = rayleigh_sommerfeld_propagate(E0, z, lam, dx, bandlimit=False)
    b = rayleigh_sommerfeld_propagate(E0, z, lam, dx, bandlimit=True)
    c = angular_spectrum_propagate(E0, z=z, wavelength=lam, dx=dx, bandlimit=False)
    nn = np.linalg.norm(REF)
    step = k*(L2/np.hypot(L2, z))*dx
    cut = (2*N*dx)/(2*lam*z)
    print(f"{z*1e6:>8.0f} {step:>14.2f} "
          f"{np.linalg.norm(a-REF)/nn:>11.3e} {np.linalg.norm(b-REF)/nn:>11.3e}"
          f" {np.linalg.norm(c-REF)/nn:>11.3e} {cut:>14.3e}"
          f" {str(cut > 1/(2*dx)):>15s}"
          f" {str(np.array_equal(a, b)):>12s}")
