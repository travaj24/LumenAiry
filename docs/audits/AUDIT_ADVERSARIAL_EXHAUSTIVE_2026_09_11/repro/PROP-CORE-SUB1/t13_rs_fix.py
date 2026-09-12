"""Verify the concrete fix for the RS spatial-domain kernel aliasing:
mask h(x,y,z) to the radius whose propagation angle is representable on
the grid,  sin(theta_c) = lam/(2 dx)  ->  rho_c = z tan(theta_c).
This is Matsushima's criterion applied to the SPATIAL kernel (where RS
samples it) instead of to FFT(h) after the damage is done.
"""
import sys
import numpy as np
from scipy.special import j0
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
from lumenairy.propagators.asm import angular_spectrum_propagate

lam = 633e-9; k = 2*np.pi/lam
w0, z = 6e-6, 50e-6


def rs_fixed(E_in, z, lam, dx, mask=True):
    """rs.py:242-283 kernel build + convolution, with the spatial mask."""
    Ny, Nx = E_in.shape
    Ny2, Nx2 = 2*Ny, 2*Nx
    x = (np.arange(Nx2)-Nx2/2)*dx
    y = (np.arange(Ny2)-Ny2/2)*dx
    X, Y = np.meshgrid(x, y, indexing='xy')
    r = np.sqrt(X**2+Y**2+z**2)
    h = (z/(2*np.pi*r**2))*np.exp(1j*k*r)*(1.0/r-1j*k)*dx*dx
    if mask:
        s_c = lam/(2*dx)
        if s_c < 1.0:
            rho_c = z*s_c/np.sqrt(1-s_c**2)
            h = h*((X**2+Y**2) <= rho_c**2)
    H = np.fft.fft2(np.fft.ifftshift(h))
    Ep = np.zeros((Ny2, Nx2), complex)
    Ep[Ny//2:Ny//2+Ny, Nx//2:Nx//2+Nx] = E_in
    return np.fft.ifft2(np.fft.fft2(Ep)*H)[Ny//2:Ny//2+Ny, Nx//2:Nx//2+Nx]


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


print(f"Gaussian w0={w0*1e6:.0f}um, z={z*1e6:.0f}um")
print(f"{'N':>5} {'dx(um)':>7} {'rho_c(um)':>10} {'half-ext':>9}"
      f" {'RS as-shipped':>14} {'RS + spatial mask':>18} {'ASM':>11}"
      f" {'masked power':>13}")
for (N, dx) in ((64, 0.5e-6), (64, 1e-6), (64, 2e-6), (128, 1e-6), (128, 2e-6)):
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x); Rr = np.hypot(X, Y)
    E0 = np.exp(-(Rr/w0)**2).astype(np.complex128)
    REF = oracle(Rr); nn = np.linalg.norm(REF)
    a = rayleigh_sommerfeld_propagate(E0, z, lam, dx)
    b = rs_fixed(E0, z, lam, dx, mask=True)
    c = angular_spectrum_propagate(E0, z=z, wavelength=lam, dx=dx)
    s_c = lam/(2*dx)
    rho_c = z*s_c/np.sqrt(1-s_c**2) if s_c < 1 else np.inf
    pin = float(np.sum(np.abs(E0)**2))
    print(f"{N:>5} {dx*1e6:>7.2f} {rho_c*1e6:>10.1f} {N*dx/2*1e6:>9.1f}"
          f" {np.linalg.norm(a-REF)/nn:>14.3e}"
          f" {np.linalg.norm(b-REF)/nn:>18.3e}"
          f" {np.linalg.norm(c-REF)/nn:>11.3e}"
          f" {float(np.sum(np.abs(b)**2))/pin:>13.6f}")
