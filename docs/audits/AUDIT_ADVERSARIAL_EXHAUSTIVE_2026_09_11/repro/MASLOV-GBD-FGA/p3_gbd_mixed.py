import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.propagators.gbd as G
from lumenairy.propagators.asm import angular_spectrum_propagate

lam = 1.0e-6
N = 256; dx = 2.0e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
E0 = np.exp(-(X**2+Y**2)/(40e-6)**2).astype(complex)
# add a sharp feature so the adaptive refinement engages strongly
E0 = E0 * (1.0 + 0.5*(np.abs(X) < 10e-6))

orig = G.propagate_beamlets_freespace
def patched(b, z_distance, wavelength, *, n_medium=1.0):
    out = orig(b, z_distance, wavelength, n_medium=n_medium)
    Nz = b.directions[...,2]; t = z_distance/np.where(np.abs(Nz)>1e-30, Nz, 1e-30)
    axial = np.exp(1j*(2*np.pi/wavelength*n_medium)*t)
    qr = out.amplitude/(b.amplitude*axial)
    return G.BeamletBundle(out.positions, out.directions, out.Q,
                           b.amplitude*np.conj(qr)*axial, out.waist0)

bA, st = G.decompose_field_adaptive(E0, dx, wavelength=lam, base_step=4,
                                    refine_step=1, refine_ratio=0.05,
                                    return_stats=True)
print("adaptive:", st, " waists(um):", np.unique(np.asarray(bA.waist0))*1e6)
zRf = np.pi*(3e-6)**2/lam; zRc = np.pi*(12e-6)**2/lam
print(f"zR fine={zRf*1e6:.1f} um, zR coarse={zRc*1e6:.1f} um")
for z in (2e-5, 5e-5, 2e-4, 1e-3):
    ref = angular_spectrum_propagate(E0, z, lam, dx)
    for name, fn in (("as-is", orig), ("conj-fix", patched)):
        b = fn(bA, z, lam)
        F = G.reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, wavelength=lam, window=5.0)
        phi = np.angle(np.vdot(ref, F))
        e_raw = np.linalg.norm(F-ref)/np.linalg.norm(ref)
        e_gl  = np.linalg.norm(F*np.exp(-1j*phi)-ref)/np.linalg.norm(ref)
        print(f"  z={z*1e6:7.1f}um {name:9s}  relL2 raw={e_raw:.4e}  best-global-phase-removed={e_gl:.4e}")
