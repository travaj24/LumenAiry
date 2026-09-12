"""Confirm the RS near-field failure is spatial-domain kernel aliasing:
it scales with the PADDED GRID EXTENT at fixed z, and it breaks energy
conservation."""
import sys
import numpy as np
from scipy.special import j0
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
from lumenairy.propagators.asm import angular_spectrum_propagate

lam = 633e-9; k = 2*np.pi/lam
w0, z = 6e-6, 50e-6


def oracle(Rr, nf=6000):
    runif = np.unique(np.round(Rr.ravel(), 12))
    inv = np.searchsorted(runif, np.round(Rr.ravel(), 12))
    fmax = min(8.0/(np.pi*w0), 0.999999/lam)
    fn, fw = np.polynomial.legendre.leggauss(nf)
    fv = 0.5*fmax*(fn+1.0); fwt = 0.5*fmax*fw
    wgt = (2*np.pi*np.pi*w0**2*np.exp(-(np.pi*w0*fv)**2)
           * np.exp(1j*k*z*np.sqrt(np.maximum(1-(lam*fv)**2, 0.0)))*fv*fwt)
    out = np.empty(runif.shape, dtype=complex)
    for i0 in range(0, runif.size, 400):
        out[i0:i0+400] = j0(2*np.pi*np.outer(runif[i0:i0+400], fv)) @ wgt
    return out[inv].reshape(Rr.shape)


print(f"Gaussian w0={w0*1e6:.0f}um, z={z*1e6:.0f}um, lam={lam*1e9:.0f}nm")
print("Fixed physical content; only the grid EXTENT (via dx) changes.")
print(f"{'N':>5} {'dx(um)':>8} {'half-ext(um)':>13} {'kern rad/px':>12}"
      f" {'RS relL2':>11} {'ASM relL2':>11} {'RS power/Pin':>13}"
      f" {'ASM power/Pin':>14}")
for (N, dx) in ((64, 0.25e-6), (64, 0.5e-6), (64, 1e-6), (64, 2e-6),
                (128, 1e-6), (128, 2e-6)):
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x); Rr = np.hypot(X, Y)
    E0 = np.exp(-(Rr/w0)**2).astype(np.complex128)
    REF = oracle(Rr)
    a = rayleigh_sommerfeld_propagate(E0, z, lam, dx)          # defaults
    c = angular_spectrum_propagate(E0, z=z, wavelength=lam, dx=dx)  # defaults
    L = N*dx                       # padded half-extent
    step = k*(L/np.hypot(L, z))*dx
    nn = np.linalg.norm(REF)
    pin = float(np.sum(np.abs(E0)**2))
    trunc = 1 - np.exp(-2*(N*dx/2/w0)**2)   # ~fraction of Gaussian on-grid
    print(f"{N:>5} {dx*1e6:>8.2f} {N*dx/2*1e6:>13.1f} {step:>12.2f}"
          f" {np.linalg.norm(a-REF)/nn:>11.3e} {np.linalg.norm(c-REF)/nn:>11.3e}"
          f" {float(np.sum(np.abs(a)**2))/pin:>13.6f}"
          f" {float(np.sum(np.abs(c)**2))/pin:>14.6f}")
print("\n(kern rad/px = k*sin(theta_edge)*dx, the RS impulse-response phase")
print(" step at the padded-grid corner; > pi means the point-sampled kernel")
print(" h(x,y,z) is aliased before its FFT is ever taken.)")
