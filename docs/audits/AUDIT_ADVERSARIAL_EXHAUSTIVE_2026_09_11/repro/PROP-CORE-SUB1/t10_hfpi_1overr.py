"""Quantify the HFPI estimator bias: missing 1/r and missing output-pixel
Jacobian.  Analytic prediction

    E[HFPI(p)] / E_true(p)  ~  dx_out^2 * cos(theta) / (N_src_px * r)

so the HFPI/ASM amplitude ratio must HALVE when z doubles.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy.propagators.hfpi as H
from lumenairy.propagators.asm import angular_spectrum_propagate

lam = 633e-9
N, dx, w0 = 64, 2e-6, 12e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
Rr = np.hypot(X, Y)
E0 = np.exp(-(Rr/w0)**2).astype(np.complex128)
patch = Rr < 10e-6                      # central patch, well inside the beam
npaths = 24_000_000
cone = 0.05

print(f"grid {N}x{N} dx={dx*1e6:.0f}um, Gaussian w0={w0*1e6:.0f}um, "
      f"n_paths={npaths:,}, cone_half_angle={cone} rad")
print(f"{'z (mm)':>8} {'occupancy':>10} {'|HFPI|/|ASM| (patch)':>22} "
      f"{'predicted dx^2/(Nsrc*z)':>24} {'meas/pred':>10}")
vals = []
for z in (2e-3, 4e-3):
    p = H.init_paths_from_field(E0, dx, n_paths=npaths, wavelength=lam,
                                rng=11, cone_half_angle=cone)
    p = H.propagate_to_plane(p, z_target=z, wavelength=lam)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Eh = H.accumulate_to_grid(p, Ny=N, Nx=N, dx=dx,
                                  output_dtype=np.complex128,
                                  on_undersampled='silent')
    Ea = angular_spectrum_propagate(E0, z=z, wavelength=lam, dx=dx,
                                    bandlimit=False)
    occ = np.count_nonzero(Eh)/Eh.size
    # incoherent (shot-noise-robust) amplitude ratio over the patch
    rat = np.sqrt(np.mean(np.abs(Eh[patch])**2)
                  / np.mean(np.abs(Ea[patch])**2))
    pred = dx*dx/(N*N*z)
    vals.append(rat)
    print(f"{z*1e3:>8.1f} {occ:>10.3f} {rat:>22.5e} {pred:>24.5e} "
          f"{rat/pred:>10.4f}")
    del p

print(f"\nMEASURED ratio(z=2mm)/ratio(z=4mm) = {vals[0]/vals[1]:.4f}")
print("  -> 2.00 is the signature of the MISSING 1/r (truth scales 1/r,")
print("     the estimator scales 1/r^2 because the landed-path count per")
print("     output pixel already carries A_p cos(theta)/r^2).")
print("  -> 1.00 would mean the kernel is correct.")
