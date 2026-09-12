"""Probe 2: full GBD pipeline vs ASM; and the NON-global consequence for a
mixed-waist (adaptive) bundle."""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.propagators.gbd as G
from lumenairy.propagators.asm import angular_spectrum_propagate

lam = 1.0e-6; k = 2*np.pi/lam
N = 256; dx = 2.0e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
w_in = 40e-6
E0 = np.exp(-(X**2+Y**2)/w_in**2).astype(complex)

z = 3e-3
ref = angular_spectrum_propagate(E0, z, lam, dx)

for wf in (1.0, 2.0, 3.0):
    b = G.decompose_field_to_beamlets(E0, dx, wavelength=lam, waist_factor=wf, sample_step=1)
    b = G.propagate_beamlets_freespace(b, z, lam)
    F = G.reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, wavelength=lam, window=5.0)
    ip = np.vdot(ref, F)
    phi = np.angle(ip)
    zR_b = np.pi*(wf*dx)**2/lam
    pred = 2*np.arctan(z/zR_b)
    rel_raw = np.linalg.norm(F-ref)/np.linalg.norm(ref)
    Fc = F*np.exp(-1j*phi)
    rel_fix = np.linalg.norm(Fc-ref)/np.linalg.norm(ref)
    print(f"wf={wf}: measured global phase={phi:+.6f}  library formula 2*atan(z/zRb)={pred:+.6f}"
          f"  relL2 raw={rel_raw:.3e}  after global-phase removal={rel_fix:.3e}")

# ---- now: patch the sign (conjugate qratio) and re-run ------------------
print("\n--- with qratio -> conj(qratio) (the proposed fix) ---")
orig = G.propagate_beamlets_freespace
def patched(beamlets, z_distance, wavelength, *, n_medium=1.0):
    out = orig(beamlets, z_distance, wavelength, n_medium=n_medium)
    # undo qratio, apply conj(qratio):  amp_new = amp_old*qratio*axial
    Nz = beamlets.directions[...,2]; t = z_distance/np.where(np.abs(Nz)>1e-30, Nz, 1e-30)
    kk = 2*np.pi/wavelength*n_medium
    axial = np.exp(1j*kk*t)
    qr = out.amplitude/(beamlets.amplitude*axial)
    amp = beamlets.amplitude*np.conj(qr)*axial
    return G.BeamletBundle(out.positions, out.directions, out.Q, amp, out.waist0)

for wf in (1.0, 2.0, 3.0):
    b = G.decompose_field_to_beamlets(E0, dx, wavelength=lam, waist_factor=wf, sample_step=1)
    b = patched(b, z, lam)
    F = G.reconstruct_field_from_beamlets(b, Ny=N, Nx=N, dx=dx, wavelength=lam, window=5.0)
    phi = np.angle(np.vdot(ref, F))
    rel_raw = np.linalg.norm(F-ref)/np.linalg.norm(ref)
    print(f"wf={wf}: residual global phase={phi:+.3e}  relL2 raw (NO phase fit)={rel_raw:.3e}")

# ---- the non-global case: mixed-waist bundle ---------------------------
print("\n--- mixed-waist (adaptive-style) bundle: error is NOT a global phase ---")
b_out, stats = G.decompose_field_adaptive(E0, dx, wavelength=lam, base_step=4,
                                          refine_step=1, refine_ratio=0.12,
                                          return_stats=True)
print("adaptive stats:", stats)
w0s = np.unique(np.asarray(b_out.waist0))
print("distinct beamlet waists (um):", w0s*1e6)
bp = G.propagate_beamlets_freespace(b_out, z, lam)
F = G.reconstruct_field_from_beamlets(bp, Ny=N, Nx=N, dx=dx, wavelength=lam, window=5.0)
phi = np.angle(np.vdot(ref, F))
print(f"raw relL2 = {np.linalg.norm(F-ref)/np.linalg.norm(ref):.4e}")
print(f"after best global-phase removal = {np.linalg.norm(F*np.exp(-1j*phi)-ref)/np.linalg.norm(ref):.4e}")
bp2 = patched(b_out, z, lam)
F2 = G.reconstruct_field_from_beamlets(bp2, Ny=N, Nx=N, dx=dx, wavelength=lam, window=5.0)
phi2 = np.angle(np.vdot(ref, F2))
print(f"FIXED: raw relL2 = {np.linalg.norm(F2-ref)/np.linalg.norm(ref):.4e}"
      f"  after global-phase removal = {np.linalg.norm(F2*np.exp(-1j*phi2)-ref)/np.linalg.norm(ref):.4e}")
