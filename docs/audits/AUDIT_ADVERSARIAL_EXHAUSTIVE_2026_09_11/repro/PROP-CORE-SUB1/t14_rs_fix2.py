"""The alias-free construction: build H in the FREQUENCY domain on the
same 2N zero-padded grid (the RS-I transfer function IS the ASM one),
instead of point-sampling h(x,y,z) in the spatial domain."""
import sys
import numpy as np
from scipy.special import j0
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate

lam = 633e-9; k = 2*np.pi/lam
w0, z = 6e-6, 50e-6


def rs_freqdomain(E_in, z, lam, dx):
    Ny, Nx = E_in.shape
    Ny2, Nx2 = 2*Ny, 2*Nx
    fx = np.fft.fftfreq(Nx2, d=dx)
    fy = np.fft.fftfreq(Ny2, d=dx)
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    arg = 1.0 - (lam*FX)**2 - (lam*FY)**2
    H = np.where(arg > 0, np.exp(1j*k*z*np.sqrt(np.maximum(arg, 0.0))), 0.0)
    Ep = np.zeros((Ny2, Nx2), complex)
    Ep[Ny//2:Ny//2+Ny, Nx//2:Nx//2+Nx] = E_in
    return np.fft.ifft2(np.fft.fft2(Ep)*H)[Ny//2:Ny//2+Ny, Nx//2:Nx//2+Nx]


def oracle(Rr, nf=6000):
    ru = np.unique(np.round(Rr.ravel(), 12))
    inv = np.searchsorted(ru, np.round(Rr.ravel(), 12))
    fmax = min(8.0/(np.pi*w0), 0.999999/lam)
    fn, fw = np.polynomial.legendre.leggauss(nf)
    fv = 0.5*fmax*(fn+1.0); fwt = 0.5*fmax*fw
    wgt = (2*np.pi*np.pi*w0**2*np.exp(-(np.pi*w0*fv)**2)
           * np.exp(1j*k*z*np.sqrt(np.maximum(1-(lam*fv)**2, 0.0)))*fv*fwt)
    out = np.empty(ru.shape, dtype=complex)
    for i0 in range(0, ru.size, 400):
        out[i0:i0+400] = j0(2*np.pi*np.outer(ru[i0:i0+400], fv)) @ wgt
    return out[inv].reshape(Rr.shape)


print(f"Gaussian w0={w0*1e6:.0f}um, z={z*1e6:.0f}um")
print(f"{'N':>5} {'dx(um)':>7} {'RS as-shipped':>14} {'power':>9}"
      f" {'RS freq-domain H':>18} {'power':>9}"
      f"    sampling guard  z < 2 N dx^2/lam ?")
for (N, dx) in ((64, 0.5e-6), (64, 1e-6), (64, 2e-6), (128, 1e-6), (128, 2e-6)):
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x); Rr = np.hypot(X, Y)
    E0 = np.exp(-(Rr/w0)**2).astype(np.complex128)
    REF = oracle(Rr); nn = np.linalg.norm(REF)
    a = rayleigh_sommerfeld_propagate(E0, z, lam, dx)
    b = rs_freqdomain(E0, z, lam, dx)
    pin = float(np.sum(np.abs(E0)**2))
    zc = 2*N*dx*dx/lam
    print(f"{N:>5} {dx*1e6:>7.2f} {np.linalg.norm(a-REF)/nn:>14.3e}"
          f" {float(np.sum(np.abs(a)**2))/pin:>9.4f}"
          f" {np.linalg.norm(b-REF)/nn:>18.3e}"
          f" {float(np.sum(np.abs(b)**2))/pin:>9.4f}"
          f"    z_crit={zc*1e6:8.1f}um  aliased={z < zc}")
