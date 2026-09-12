"""ANALYSIS probe 4: SH wavefront recon, beam_stats, detector, GS error scale, EE."""
import sys, warnings, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.detector import shack_hartmann, apply_detector
from lumenairy.analysis.beam_stats import M2, beam_d4sigma, beam_centroid, beam_diameter
from lumenairy.analysis.psf_mtf_otf import encircled_energy_radius, encircled_energy_curve, compute_psf
from lumenairy.analysis.phase_retrieval import gerchberg_saxton, error_reduction

print("=== A. shack_hartmann: slope gain + wavefront-reconstruction scale ===")
lam = 632.8e-9; k0 = 2*np.pi/lam
N, dx = 256, 5e-6
pitch, focal = 32*dx, 5e-3
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)

for theta in (2e-4, 5e-4, 1e-3):
    W = theta*X                       # OPD [m] = theta * x  -> dW/dx = theta
    E = np.exp(1j*k0*W)
    sx, sy, wf, cx, cy = shack_hartmann(E, dx, lam, pitch, focal)
    g = np.nanmean(sx)/theta
    # reconstruct: expected wavefront = theta * (x - x0), step = sa_pixels*dx
    sa = int(round(pitch/dx)); p = sa*dx
    nl = sx.shape[0]
    j = np.arange(nl)
    expected = theta*p*j              # anchored at j=0
    got = wf[nl//2, :]
    ratio = np.polyfit(expected, got, 1)[0]
    print(f"  tilt {theta*1e3:.2f} mrad : slope gain = {g:.4f} ; "
          f"wavefront/expected slope ratio = {ratio:.4f} (should be ~1 x gain={g:.3f})")

print("  -> pure defocus:")
Wd = 1e-6*( (X**2+Y**2)/ (N*dx/2)**2 )       # 1 um at edge
E = np.exp(1j*k0*Wd)
sx, sy, wf, cx, cy = shack_hartmann(E, dx, lam, pitch, focal)
sa = int(round(pitch/dx)); p = sa*dx; nl = sx.shape[0]
jj = np.arange(nl)
xl = (jj - (nl-1)/2)*p
XL, YL = np.meshgrid(xl, xl)
Wtrue = 1e-6*((XL**2+YL**2)/(N*dx/2)**2)
Wtrue = Wtrue - Wtrue[0, 0]
m = np.isfinite(wf) & np.isfinite(Wtrue)
sc = np.polyfit(Wtrue[m].ravel(), wf[m].ravel(), 1)[0]
print(f"     reconstructed/true wavefront scale = {sc:.4f}  (expect ~1)")
print(f"     max|wf| = {np.nanmax(np.abs(wf)):.4e} m, max|true| = {np.nanmax(np.abs(Wtrue)):.4e} m")

print()
print("=== B. M2 on analytic beams ===")
for Ngrid, w0 in ((256, 40e-6), (512, 40e-6)):
    dxg = 4e-6 if Ngrid == 256 else 2e-6
    xg = (np.arange(Ngrid)-Ngrid/2)*dxg
    Xg, Yg = np.meshgrid(xg, xg)
    E = np.exp(-(Xg**2+Yg**2)/w0**2)
    m2 = M2(E.astype(complex), dxg, 633e-9)
    d4 = beam_d4sigma(E.astype(complex), dxg)
    print(f"  N={Ngrid} TEM00: M2={m2[0]:.6f},{m2[1]:.6f}  D4sigma={d4[0]*1e6:.4f} um "
          f"(2w0={2*w0*1e6:.4f})")
    # curved (non-waist) Gaussian: add quadratic phase
    Ec = E*np.exp(1j*2*np.pi/633e-9*(Xg**2+Yg**2)/(2*0.05))
    print(f"           curved:  M2={M2(Ec, dxg, 633e-9)[0]:.6f} (should stay 1)")
    # TEM01 (Hermite-Gauss n=1 in x): M2_x = 3
    E01 = (2*Xg/w0)*np.exp(-(Xg**2+Yg**2)/w0**2)
    print(f"           TEM10:   M2_x={M2(E01.astype(complex), dxg, 633e-9)[0]:.6f} (expect 3), "
          f"M2_y={M2(E01.astype(complex), dxg, 633e-9)[1]:.6f} (expect 1)")

print()
print("=== C. beam_diameter / encircled energy vs exact Gaussian ===")
Ngrid, dxg, w0 = 512, 1e-6, 40e-6
xg = (np.arange(Ngrid)-Ngrid/2)*dxg
Xg, Yg = np.meshgrid(xg, xg)
E = np.exp(-(Xg**2+Yg**2)/w0**2).astype(complex)
print(f"  beam_diameter 1/e^2 = {beam_diameter(E, dxg)*1e6:.4f} um (exact {2*w0*1e6:.4f})")
print(f"  EE radius @86.47%   = {encircled_energy_radius(E, dxg, threshold=1-np.exp(-2))*1e6:.4f} um "
      f"(exact {w0*1e6:.4f})")
print(f"  EE radius @50%      = {encircled_energy_radius(E, dxg, threshold=0.5)*1e6:.4f} um "
      f"(exact {w0*np.sqrt(np.log(2)/2)*1e6:.4f})")
r, ee = encircled_energy_curve(E, dxg, radii=np.array([0.5*w0, w0, 1.5*w0, 2*w0]))
exact = 1-np.exp(-2*(r/w0)**2)
print(f"  EE curve   : {np.array2string(ee, precision=6)}")
print(f"  exact      : {np.array2string(exact, precision=6)}")
print(f"  max abs err: {np.abs(ee-exact).max():.3e}")

print()
print("=== D. apply_detector: flux conservation + noise ===")
I0 = 1e12
Nf, dxf = 64, 1e-6
Ef = np.full((Nf, Nf), np.sqrt(I0), dtype=complex)
for (npx, pp) in ((16, 4e-6), (16, 2e-6), (16, 8e-6), (8, 2e-6), (16, 2.5e-6), (None, 4e-6)):
    img, xd, yd = apply_detector(Ef, dxf, pp, n_pixels=npx, quantum_efficiency=1.0,
                                 exposure_time=1.0, seed=0)
    npx_eff = img.shape[0]
    expected = I0*pp*pp
    inner = img[npx_eff//4:3*npx_eff//4, npx_eff//4:3*npx_eff//4]
    print(f"  n_pixels={str(npx):>4} pitch={pp*1e6:.1f}um -> shape {img.shape}, "
          f"mean(inner)/expected = {inner.mean()/expected:.6f}, "
          f"rel std = {inner.std()/inner.mean():.5f} (Poisson expect {1/np.sqrt(expected):.5f})")
# noise-free request?
img2, _, _ = apply_detector(Ef, dxf, 4e-6, quantum_efficiency=1.0, seed=1)
img3, _, _ = apply_detector(Ef, dxf, 4e-6, quantum_efficiency=1.0, seed=2)
print(f"  two different seeds identical? {np.array_equal(img2, img3)} "
      f"(Poisson is always applied -- no opt-out kwarg)")

print()
print("=== E. gerchberg_saxton reported error scale ===")
Ng = 64
xs = np.linspace(-1, 1, Ng)
Xs, Ys = np.meshgrid(xs, xs)
src = np.exp(-(Xs**2+Ys**2)/0.5**2)
rng = np.random.default_rng(0)
phi0 = rng.uniform(-np.pi, np.pi, (Ng, Ng))
F0 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(src*np.exp(1j*phi0))))
tgt = np.abs(F0)                              # EXACTLY achievable target
ph, err, hist = gerchberg_saxton(src, tgt, n_iter=50, initial_phase=phi0,
                                 return_history=True)
Fr = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(src*np.exp(1j*ph))))
print(f"  exact solution supplied as initial phase.")
print(f"  true |FFT| vs target max abs diff = {np.abs(np.abs(Fr)-tgt).max():.3e}")
print(f"  reported final error              = {err:.6e}")
print(f"  mean(target^2)                    = {np.mean(tgt**2):.6e}")
print(f"  history[0], history[-1]           = {hist[0]:.4e}, {hist[-1]:.4e}")
sp = np.sum(src**2); tp = np.sum(tgt**2)
print(f"  source_power={sp:.4e} target_power={tp:.4e}  ratio={tp/sp:.4e}  N^2={Ng**2}")

print()
print("=== F. error_reduction round trip (support-constrained) ===")
Ng = 64
obj = np.zeros((Ng, Ng), complex)
sup = np.zeros((Ng, Ng), bool)
sup[20:44, 20:44] = True
rng = np.random.default_rng(3)
obj[sup] = rng.random(sup.sum())*np.exp(1j*rng.uniform(-1, 1, sup.sum()))
meas = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(obj))))
rec, e, h = error_reduction(meas, sup, n_iter=400, seed=1, return_history=True)
Fr = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(rec))))
print(f"  final Fourier-magnitude error = {e:.4e} (history[0]={h[0]:.4e})")
print(f"  relative |F| mismatch          = {np.abs(Fr-meas).max()/meas.max():.4e}")
