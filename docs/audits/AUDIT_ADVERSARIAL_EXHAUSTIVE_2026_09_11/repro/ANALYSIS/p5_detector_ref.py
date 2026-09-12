"""ANALYSIS probe 5: detector flux (high photon count), rectangular field,
   diffraction_limited_peak paraxial-reference bias (properly sampled)."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.detector import apply_detector
from lumenairy.analysis.through_focus import diffraction_limited_peak
from lumenairy.analysis.strehl import strehl_phase_integral
from lumenairy.propagators.propagation import angular_spectrum_propagate

print("=== A. apply_detector flux conservation (photon-rich, Poisson negligible) ===")
I0 = 1e20
Nf, dxf = 64, 1e-6
Ef = np.full((Nf, Nf), np.sqrt(I0), dtype=complex)
tot_field = I0*(Nf*dxf)**2
for (npx, pp) in ((16, 4e-6), (16, 2e-6), (16, 8e-6), (8, 2e-6), (16, 2.5e-6),
                  (32, 2e-6), (None, 4e-6)):
    img, xd, yd = apply_detector(Ef, dxf, pp, n_pixels=npx, seed=0)
    n = img.shape[0]
    expected = I0*pp*pp
    # only look at pixels fully inside the field
    span = n*pp
    ok = min(span, Nf*dxf)
    print(f"  n_pixels={str(npx):>4} pitch={pp*1e6:4.1f}um -> shape {img.shape}  "
          f"det span={span*1e6:5.1f}um field={Nf*dxf*1e6:.1f}um  "
          f"median/expected={np.median(img[img>0])/expected:.6f}  "
          f"sum/field_total={img.sum()/tot_field:.6f}")

print()
print("=== B. rectangular field (Ny != Nx) ===")
Er = np.full((32, 64), np.sqrt(I0), dtype=complex)
img, xd, yd = apply_detector(Er, dxf, 4e-6, seed=0)
print(f"  Er shape (32,64) dx=1um pitch=4um -> image {img.shape}; "
      f"sum/total_in = {img.sum()/(I0*32*64*dxf**2):.6f} "
      f"(expect 1 if all flux collected)")

print()
print("=== C. diffraction_limited_peak: paraxial-quadratic vs exact-sphere reference ===")
lam = 600e-9; k0 = 2*np.pi/lam
def W040_waves(D, f, lam):
    return (D/2)**4/(8*f**3*lam)
print("  (i) Fraunhofer/pupil-integral estimate of the reference's own Strehl loss")
for (D, f) in ((10e-3, 200e-3), (10e-3, 100e-3), (10e-3, 50e-3), (10e-3, 25e-3),
               (25.4e-3, 100e-3), (2e-3, 20e-3)):
    Ng = 512
    xg = np.linspace(-D/2, D/2, Ng)
    Xg, Yg = np.meshgrid(xg, xg)
    R2 = Xg**2+Yg**2
    ap = R2 <= (D/2)**2
    # phase error of the paraxial quadratic relative to the exact sphere
    dphi = (-k0*R2/(2*f)) - (-k0*(np.sqrt(R2+f**2)-f))
    P = ap*np.exp(1j*dphi)
    S_ref = strehl_phase_integral(P)
    print(f"   D={D*1e3:5.1f}mm f={f*1e3:6.1f}mm (f/{f/D:5.2f}): W040={W040_waves(D,f,lam):8.4f} waves"
          f"   reference-pupil Strehl = {S_ref:.5f}"
          f"  => computed Strehl inflated by {1/S_ref:7.3f}x")

print()
print("  (ii) direct ASM check at a scaled geometry the grid can resolve")
for (D, f, Ng, dxg) in ((307e-6, 1.23e-3, 3072, 0.15e-6),
                        (307e-6, 2.46e-3, 3072, 0.15e-6)):
    xg = (np.arange(Ng)-Ng/2)*dxg
    Xg, Yg = np.meshgrid(xg, xg)
    R2 = Xg**2+Yg**2
    ap = (R2 <= (D/2)**2).astype(float)
    E_sph = ap*np.exp(-1j*k0*(np.sqrt(R2+f**2)-f))
    pk_sph = float((np.abs(angular_spectrum_propagate(E_sph, f, lam, dxg, bandlimit=True))**2).max())
    pk_par = diffraction_limited_peak(ap.astype(complex), lam, f, dxg)
    dphi = (-k0*R2/(2*f)) - (-k0*(np.sqrt(R2+f**2)-f))
    S_pred = strehl_phase_integral((R2 <= (D/2)**2)*np.exp(1j*dphi))
    print(f"   D={D*1e6:.0f}um f={f*1e6:.0f}um (f/{f/D:.2f}) W040={W040_waves(D,f,lam):.4f} w: "
          f"ASM paraxial/sphere peak = {pk_par/pk_sph:.5f}  (pupil-integral predicts {S_pred:.5f})")
