import numpy as np, sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as lu
from lumenairy.sources.core import (create_gaussian_beam, create_hermite_gauss,
    create_laguerre_gauss, create_tilted_plane_wave, create_point_source,
    create_top_hat_beam, create_bessel_beam, create_fiber_mode,
    create_gaussian_schell_source, create_annular_incoherent_source)
from lumenairy.propagators.propagation import angular_spectrum_propagate

wl = 633e-9
print("=== 1. Gaussian beam: waist convention (1/e^2 intensity radius) ===")
N = 512; dx = 1e-6; w0 = 20e-6
E, x, y = create_gaussian_beam(N, dx, wl, w0=w0, normalize='peak')
I = np.abs(E)**2; I = I / I.max()
i0 = np.argmin(np.abs(x - w0))
print("  I(r=w0)/I(0) = %.6f  (expect exp(-2)=%.6f)" % (I[N//2, i0], np.exp(-2)))
print("  field amp at r=w0 = %.6f (expect exp(-1)=%.6f)" % (np.abs(E[N//2, i0]), np.exp(-1)))

print("")
print("=== 2. dtype preservation through normalize ===")
for mode in ('peak', 'power', 'none'):
    E, _, _ = create_gaussian_beam(64, dx, wl, w0=w0, normalize=mode, dtype=np.complex64)
    tag = "OK" if E.dtype == np.complex64 else "*** UPCAST ***"
    print("  create_gaussian_beam  normalize=%-6s dtype=%s  %s" % (mode, E.dtype, tag))
for mode in ('peak', 'power', 'none'):
    E, _, _ = create_hermite_gauss(64, dx, w0, wl, m=1, n=0, normalize=mode, dtype=np.complex64)
    tag = "OK" if E.dtype == np.complex64 else "*** UPCAST ***"
    print("  create_hermite_gauss  normalize=%-6s dtype=%s  %s" % (mode, E.dtype, tag))
E, _, _ = create_top_hat_beam(64, dx, wl, diameter=20e-6, normalize='power', dtype=np.complex64)
print("  create_top_hat_beam   normalize=power  dtype=%s  %s"
      % (E.dtype, "OK" if E.dtype == np.complex64 else "*** UPCAST ***"))

print("")
print("=== 3. HG/LG normalisation + orthogonality (N=512) ===")
N = 512; dx = 0.5e-6; w0 = 15e-6
modes = []; labels = []
for m in range(3):
    for n in range(3):
        if m + n <= 2:
            E, _, _ = create_hermite_gauss(N, dx, w0, wl, m=m, n=n, normalize='power')
            modes.append(E); labels.append("HG%d%d" % (m, n))
G = np.zeros((len(modes), len(modes)))
for i, a in enumerate(modes):
    for j, b in enumerate(modes):
        G[i, j] = abs(np.sum(a * np.conj(b)) * dx * dx)
print("  HG Gram diag:", np.round(np.diag(G), 10))
print("  HG max off-diagonal:", (G - np.diag(np.diag(G))).max())
modesL = []; labL = []
for p in range(2):
    for l in (-2, -1, 0, 1, 2):
        E, _, _ = create_laguerre_gauss(N, dx, w0, wl, p=p, l=l, normalize='power')
        modesL.append(E); labL.append("LG%d%+d" % (p, l))
GL = np.zeros((len(modesL), len(modesL)))
for i, a in enumerate(modesL):
    for j, b in enumerate(modesL):
        GL[i, j] = abs(np.sum(a * np.conj(b)) * dx * dx)
print("  LG Gram diag:", np.round(np.diag(GL), 10))
print("  LG max off-diagonal:", (GL - np.diag(np.diag(GL))).max())

print("")
print("=== 4. tilted plane wave: propagation direction (centroid drift after ASM) ===")
N = 512; dx = 2e-6; th = np.radians(3.0)
E, x, y = create_tilted_plane_wave(N, dx, wl, angle_x=th)
Eg, _, _ = create_gaussian_beam(N, dx, wl, w0=80e-6)
Ein = E * Eg
z = 3e-3
Eo = angular_spectrum_propagate(Ein, z, wl, dx, dx)
I = np.abs(Eo)**2
cx = np.sum(I * x[None, :]) / np.sum(I)
print("  angle_x=+3deg, z=%.1f mm -> centroid x = %+.3f um (expect %+.3f um)"
      % (z * 1e3, cx * 1e6, z * np.tan(th) * 1e6))
I0 = np.abs(angular_spectrum_propagate(Eg, z, wl, dx, dx))**2
print("  untilted control centroid = %+.3f um" % (np.sum(I0 * x[None, :]) / np.sum(I0) * 1e6))

print("")
print("=== 5. point source sign: z0>0 converging? ===")
N = 1024; dx = 1e-6; z0 = 2e-3
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    E, x, y = create_point_source(N, dx, wl, z0=z0)
    E2, _, _ = create_point_source(N, dx, wl, z0=-z0)
Ap = (np.sqrt(x[None, :]**2 + y[:, None]**2) < 200e-6)
Iin = np.abs(E * Ap)**2
Io = np.abs(angular_spectrum_propagate(E * Ap, z0, wl, dx, dx))**2
print("  z0>0 (documented converging) propagated +z0: peak=%.4e at %s (center=%d) ratio=%.2f"
      % (Io.max(), np.unravel_index(np.argmax(Io), Io.shape), N//2, Io.max()/Iin.max()))
I2 = np.abs(angular_spectrum_propagate(E2 * Ap, z0, wl, dx, dx))**2
print("  z0<0 (diverging) propagated +z0: peak=%.4e ratio=%.2f" % (I2.max(), I2.max()/Iin.max()))

print("")
print("=== 6. rng contract reproducibility ===")
kw = dict(N=32, dx=1e-6, wavelength=wl, w0=8e-6, sigma_g=4e-6, n_realizations=4)
a = create_gaussian_schell_source(rng=1234, **kw)[0]
b = create_gaussian_schell_source(rng=1234, **kw)[0]
c = create_gaussian_schell_source(rng=np.random.default_rng(1234), **kw)[0]
print("  int-seed reproducible:", np.array_equal(a, b), " Generator(1234) identical:", np.array_equal(a, c))
try:
    from lumenairy.backend.random import RandomState
    rs = RandomState(1234, backend='numpy')
    d = create_gaussian_schell_source(rng=rs, **kw)[0]
    print("  RandomState(numpy) accepted; equal to int-seed:", np.array_equal(a, d))
except Exception as ex:
    print("  RandomState:", type(ex).__name__, str(ex)[:150])

print("")
print("=== 7. GSM ensemble MCF vs analytic Schell kernel + Starikov-Wolf spectrum ===")
Ns = 24; dxs = 2e-6; w0s = 20e-6; sg = 12e-6
Ee, _, _, _ = create_gaussian_schell_source(N=Ns, dx=dxs, wavelength=wl, w0=w0s,
                                            sigma_g=sg, n_realizations=20000, rng=7)
Ef = Ee.reshape(Ee.shape[0], -1)
J = (Ef.conj().T @ Ef) / Ef.shape[0]
xs = (np.arange(Ns) - Ns / 2) * dxs
X, Y = np.meshgrid(xs, xs); P = np.stack([X.ravel(), Y.ravel()], axis=1)
Iprof = np.exp(-2 * (P[:, 0]**2 + P[:, 1]**2) / w0s**2)
D2 = ((P[:, None, :] - P[None, :, :])**2).sum(-1)
Jan = np.sqrt(Iprof[:, None] * Iprof[None, :]) * np.exp(-D2 / (2 * sg**2))
scale = np.trace(J).real / np.trace(Jan).real
print("  trace ratio (empirical/analytic) = %.6f" % scale)
print("  max relative MCF error (20000 realizations) = %.4f"
      % (np.abs(J - scale * Jan).max() / np.abs(scale * Jan).max()))
sigma_s = w0s / 2.0
aa = 1.0 / (4 * sigma_s**2); bb = 1.0 / (2 * sg**2); cc = np.sqrt(aa * aa + 2 * aa * bb)
lam1 = np.array([(np.pi / (aa + bb + cc))**0.5 * (bb / (aa + bb + cc))**n for n in range(12)])
lam2d = np.sort(np.array([l1 * l2 for l1 in lam1 for l2 in lam1]))[::-1]
lam2d = lam2d / lam2d.sum()
ev = np.linalg.eigvalsh(Jan)[::-1]; ev = ev / ev.sum()
print("  analytic Starikov-Wolf normalized eigenvalues[:6]:", np.round(lam2d[:6], 6))
print("  numeric from the analytic GSM J      [:6]:        ", np.round(ev[:6], 6))
print("  max abs diff over first 6:", np.abs(lam2d[:6] - ev[:6]).max())
evJ = np.linalg.eigvalsh((J + J.conj().T) / 2)[::-1]; evJ = evJ / evJ.sum()
print("  library-ensemble MCF eigenvalues     [:6]:        ", np.round(evJ[:6], 6))

print("")
print("=== 8. fiber mode MFD vs Marcuse ===")
V = 2.0; a_core = 4.0e-6
mfd = 2 * a_core * (0.65 + 1.619 * V**-1.5 + 2.879 * V**-6)
Ef2, xf, yf = create_fiber_mode(256, 0.5e-6, wl, mode_field_diameter=mfd)
If = np.abs(Ef2)**2; If /= If.max()
i = np.argmin(np.abs(If[128, 128:] - np.exp(-2)))
print("  Marcuse MFD(V=2,a=4um) = %.4f um; returned 1/e^2 radius = %.4f um (expect %.4f)"
      % (mfd * 1e6, xf[128 + i] * 1e6, mfd / 2 * 1e6))

print("")
print("=== 9. Bessel / top-hat sanity ===")
E, x, y = create_bessel_beam(256, 0.5e-6, wl, cone_angle=np.radians(5))
from scipy.special import jn_zeros
kr = 2 * np.pi / wl * np.sin(np.radians(5))
r1 = jn_zeros(0, 1)[0] / kr
print("  first J0 zero at r = %.4f um; sampled |E| at nearest pixel = %.4g"
      % (r1 * 1e6, np.abs(E[128, 128 + int(round(r1 / 0.5e-6))])))
E, x, y = create_top_hat_beam(256, 0.5e-6, wl, diameter=40e-6, normalize='power')
print("  top-hat integrated power:", np.sum(np.abs(E)**2) * 0.5e-6**2)

print("")
print("=== 10. grid centering convention across the module ===")
for fn, args, kw in ((create_gaussian_beam, (7, 1e-6, wl), dict(w0=3e-6)),
                     (create_tilted_plane_wave, (7, 1e-6, wl), {}),
                     (create_top_hat_beam, (7, 1e-6, wl), dict(diameter=3e-6))):
    E, x, y = fn(*args, **kw)
    print("  %-26s odd-N x = %s" % (fn.__name__, np.round(x * 1e6, 4)))
