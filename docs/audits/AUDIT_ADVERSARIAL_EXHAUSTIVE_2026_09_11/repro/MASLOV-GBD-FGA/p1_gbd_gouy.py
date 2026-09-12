"""Probe: single-beamlet GBD free-space vs analytic Gaussian (amplitude+phase)."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.gbd import (BeamletBundle, propagate_beamlets_freespace,
                                       reconstruct_field_from_beamlets)

lam = 1.55e-6
k = 2*np.pi/lam
w0 = 30e-6
zR = np.pi*w0**2/lam
print(f"w0={w0*1e6:.2f} um  zR={zR*1e3:.4f} mm")

N = 128; dx = 2e-6
def analytic(z, Ny=N, Nx=N, dx=dx):
    x = (np.arange(Nx)-Nx/2)*dx
    X, Y = np.meshgrid(x, x)
    r2 = X**2+Y**2
    q = z - 1j*zR                      # exp(-i omega t) convention: q = z - i zR
    w = w0*np.sqrt(1+(z/zR)**2)
    psi = np.arctan2(z, zR)
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(k*z + k*r2*z/(2*(z**2+zR**2)) - psi))

# single beamlet at origin, waist w0, amplitude 1 on-axis
b = BeamletBundle(positions=np.array([[0.,0.,0.]]),
                  directions=np.array([[0.,0.,1.]]),
                  Q=np.array([-1j/zR], dtype=complex),
                  amplitude=np.array([1.0+0j]),
                  waist0=np.array([w0]))

for z in (0.0, 0.5*zR, 2.0*zR, 10.0*zR):
    bz = propagate_beamlets_freespace(b, z, lam)
    E = reconstruct_field_from_beamlets(bz, Ny=N, Nx=N, dx=dx, wavelength=lam, window=None)
    A = analytic(z)
    # on-axis (pixel N/2,N/2 is exactly rho=0)
    e0 = E[N//2, N//2]; a0 = A[N//2, N//2]
    ratio = e0/a0
    psi = np.arctan2(z, zR)
    print(f"z={z/zR:6.2f} zR:  |E|/|A| = {abs(ratio):.6f}   arg(E/A) = {np.angle(ratio):+.6f} rad"
          f"   2*psi = {2*psi:+.6f}   diff = {np.angle(ratio)-2*psi:+.3e}")
    # also full-field relative L2 after removing the ratio
    rel = np.linalg.norm(E - A)/np.linalg.norm(A)
    rel_c = np.linalg.norm(E/ratio - A)/np.linalg.norm(A)
    print(f"            rel L2 raw = {rel:.3e}   after removing scalar ratio = {rel_c:.3e}")
