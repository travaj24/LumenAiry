"""ANALYSIS probe 3: PSF/OTF/MTF vs analytic Airy; Strehl definitions;
   diffraction_limited_peak paraxial-reference bias."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from scipy.special import j1
from lumenairy.analysis.psf_mtf_otf import (compute_psf, compute_otf, compute_mtf,
    mtf_radial, encircled_energy_radius, rayleigh_resolution, fwhm_resolution,
    sparrow_resolution)
from lumenairy.analysis.strehl import strehl_ratio, strehl_marechal, strehl_phase_integral
from lumenairy.analysis.through_focus import diffraction_limited_peak

lam = 600e-9
print("=== A. compute_psf: Airy first zero + pitch + Parseval ===")
Np, dxp = 256, 100e-6           # pupil grid
D = 10e-3                        # aperture 10 mm
f = 40e-3                        # f/4
x = (np.arange(Np)-Np/2)*dxp
X, Y = np.meshgrid(x, x)
pup = ((X**2+Y**2) <= (D/2)**2).astype(complex)
for ov in (1, 2, 4):
    psf, dxpsf = compute_psf(pup, lam, f, dxp, oversample=ov)
    Npsf = psf.shape[0]
    pred = lam*f/(Npsf*dxp)
    # analytic Airy first zero
    r0 = 1.22*lam*f/D
    print(f"  oversample={ov}: N={Npsf} dx_psf={dxpsf*1e6:.5f} um (pred {pred*1e6:.5f}) "
          f"first-zero {r0*1e6:.4f} um = {r0/dxpsf:.2f} px")
    # Parseval check
    Ppup = np.sum(np.abs(pup)**2)*dxp**2
    Ppsf = np.sum(psf)*dxpsf**2
    print(f"      Parseval: pupil P={Ppup:.6e}  psf P={Ppsf:.6e}  ratio={Ppsf/Ppup:.9f}")

print()
print("=== B. PSF vs analytic Airy profile ===")
psf, dxpsf = compute_psf(pup, lam, f, dxp, oversample=4)
N = psf.shape[0]
xi = (np.arange(N)-N//2)*dxpsf
XI, YI = np.meshgrid(xi, xi)
Rr = np.sqrt(XI**2+YI**2)
v = np.pi*D*Rr/(lam*f)
airy = np.where(v == 0, 1.0, (2*j1(np.where(v == 0, 1, v))/np.where(v == 0, 1, v))**2)
airy = airy*psf.max()
sel = Rr < 10*lam*f/D
print(f"  max rel dev in r<10 lam f/D: {np.abs(psf[sel]-airy[sel]).max()/psf.max():.3e}")
print(f"  peak at centre index? argmax={np.unravel_index(psf.argmax(), psf.shape)} vs {(N//2,N//2)}")

print()
print("=== C. OTF from PSF vs pupil autocorrelation; MTF(0)=1 ===")
otf = compute_otf(psf)
mtf = np.abs(otf)
print(f"  mtf[N//2,N//2] = {mtf[N//2, N//2]:.12f}")
print(f"  mtf[0,0]       = {mtf[0, 0]:.3e}")
# analytic incoherent MTF for circular pupil
fc = D/(lam*f)                      # cutoff cyc/m
fx = np.fft.fftshift(np.fft.fftfreq(N, dxpsf))
nu = np.abs(fx)/fc
mtf_an = np.where(nu <= 1, (2/np.pi)*(np.arccos(np.clip(nu, 0, 1))
                                       - np.clip(nu, 0, 1)*np.sqrt(np.clip(1-nu**2, 0, None))), 0.0)
cut = mtf[N//2, :]
good = nu < 0.9
print(f"  max |MTF_num - MTF_analytic| for nu<0.9 : {np.abs(cut[good]-mtf_an[good]).max():.3e}")
# autocorrelation of pupil
pu = np.zeros((N, N), complex)
o = (N-Np)//2
pu[o:o+Np, o:o+Np] = pup
A = np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(pu))**2)).real
A = A/A.max()
print(f"  max |MTF - pupil autocorr| : {np.abs(mtf-A).max():.3e}")

print()
print("=== D. resolution metrics vs analytic ===")
print(f"  rayleigh (expect 1.22*lam*f#={1.22*lam*(f/D)*1e6:.4f} um): "
      f"{rayleigh_resolution(psf, dxpsf, lam)*1e6:.4f} um")
print(f"  fwhm     (expect 1.029*lam*f#={1.029*lam*(f/D)*1e6:.4f} um): "
      f"{fwhm_resolution(psf, dxpsf)*1e6:.4f} um")
print(f"  sparrow  (expect 0.947*lam*f#={0.947*lam*(f/D)*1e6:.4f} um): "
      f"{sparrow_resolution(psf, dxpsf)*1e6:.4f} um")
print(f"  EE84 radius (expect ~1.22 lam f# ={1.22*lam*(f/D)*1e6:.4f} um): "
      f"{encircled_energy_radius(np.sqrt(psf).astype(complex), dxpsf, threshold=0.838)*1e6:.4f} um")

print()
print("=== E. Strehl definitions: lambda/14 RMS wavefront ===")
rho = np.sqrt(X**2+Y**2)/(D/2); th = np.arctan2(Y, X)
inside = rho <= 1
for name, Zmode in (('defocus  Z(2,0)', np.sqrt(3)*(2*rho**2-1)),
                    ('astig    Z(2,2)', np.sqrt(6)*rho**2*np.cos(2*th)),
                    ('coma     Z(3,1)', np.sqrt(8)*(3*rho**3-2*rho)*np.cos(th)),
                    ('spher    Z(4,0)', np.sqrt(5)*(6*rho**4-6*rho**2+1))):
    sig = 1/14.0
    W = sig*Zmode*inside
    P_ab = inside*np.exp(2j*np.pi*W)
    psf_a, _ = compute_psf(P_ab.astype(complex), lam, f, dxp, oversample=4)
    psf_i, _ = compute_psf(inside.astype(complex), lam, f, dxp, oversample=4)
    S_peak = psf_a.max()/psf_i.max()
    S_int = strehl_phase_integral(P_ab)
    S_mar = strehl_marechal(sig)
    S_field = strehl_ratio(np.sqrt(psf_a).astype(complex), np.sqrt(psf_i).astype(complex), 1.0)
    print(f"  {name}: peak-ratio={S_peak:.5f}  phase_integral={S_int:.5f}  "
          f"marechal={S_mar:.5f}  strehl_ratio()={S_field:.5f}")

print()
print("  tilt-only wavefront (1 wave rms of Z(1,1)):")
W = 1.0*np.sqrt(4)*rho*np.cos(th)*inside
P_t = inside*np.exp(2j*np.pi*W)
psf_t, _ = compute_psf(P_t.astype(complex), lam, f, dxp, oversample=4)
psf_i, _ = compute_psf(inside.astype(complex), lam, f, dxp, oversample=4)
print(f"    peak-ratio Strehl = {psf_t.max()/psf_i.max():.5f} (expect ~1: tilt only shifts)")
print(f"    strehl_phase_integral = {strehl_phase_integral(P_t):.5f} (does NOT remove tilt)")

print()
print("=== F. diffraction_limited_peak: paraxial-reference bias ===")
from lumenairy.propagators.propagation import angular_spectrum_propagate
k0 = 2*np.pi/lam
for (Dd, ff) in ((2e-3, 50e-3), (5e-3, 50e-3), (10e-3, 50e-3), (10e-3, 20e-3)):
    Ng, dxg = 1024, 20e-6
    xg = (np.arange(Ng)-Ng/2)*dxg
    Xg, Yg = np.meshgrid(xg, xg)
    Rg2 = Xg**2+Yg**2
    ap = (Rg2 <= (Dd/2)**2).astype(float)
    # exact converging spherical wave (the TRUE diffraction-limited pupil)
    E_sph = ap*np.exp(-1j*k0*(np.sqrt(Rg2+ff**2)-ff))
    pk_sph = float((np.abs(angular_spectrum_propagate(E_sph, ff, lam, dxg, bandlimit=True))**2).max())
    pk_par = diffraction_limited_peak(ap.astype(complex), lam, ff, dxg)
    fnum = ff/Dd
    W40 = (Dd/2)**4/(8*ff**3*lam)
    print(f"  D={Dd*1e3:4.1f}mm f={ff*1e3:4.1f}mm (f/{fnum:.1f}): "
          f"paraxial-ref peak/true-sphere peak = {pk_par/pk_sph:.5f}  "
          f"=> Strehl inflated by {pk_sph/pk_par:.5f}x ; "
          f"4th-order term W040={W40:.4f} waves")
