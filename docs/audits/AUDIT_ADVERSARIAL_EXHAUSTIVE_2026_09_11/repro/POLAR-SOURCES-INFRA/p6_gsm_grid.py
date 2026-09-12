import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.sources.core import _schell_phase_realizations, create_gaussian_schell_source

print("=== GSM coherence kernel: realized mu(dx) vs documented Gaussian ===")
rng = np.random.default_rng(11)
for (N, dx, sg) in ((64, 1e-6, 20e-6), (64, 1e-6, 8e-6), (64, 1e-6, 2e-6)):
    L = N * dx
    phi = _schell_phase_realizations(Ny=N, Nx=N, dx=dx, dy=dx,
                                     coherence_length=sg,
                                     n_realizations=40000, rng=rng)
    # empirical mu along x through the middle row, referenced to pixel N//2
    ref = phi[:, N // 2, N // 2]
    row = phi[:, N // 2, :]
    mu = (np.mean(row * np.conj(ref)[:, None], axis=0)).real
    mu = mu / mu[N // 2]
    d = (np.arange(N) - N // 2) * dx
    gauss = np.exp(-d**2 / (2 * sg**2))
    # periodized (circular) Gaussian
    wrap = sum(np.exp(-(d + m * L)**2 / (2 * sg**2)) for m in range(-3, 4))
    wrap = wrap / wrap[N // 2]
    print("  sigma_g/L = %.3f" % (sg / L))
    print("    max |mu_emp - Gaussian|        = %.4f" % np.abs(mu - gauss).max())
    print("    max |mu_emp - wrapped Gaussian| = %.4f" % np.abs(mu - wrap).max())
    j = N // 2 + N // 4
    print("    mu at dx=L/4: empirical %.4f, Gaussian %.4f, wrapped %.4f"
          % (mu[j], gauss[j], wrap[j]))
    # mean intensity
    print("    E[<|phi|^2>] = %.6f (doc: 1.0)" % float(np.mean(np.abs(phi)**2)))

print("")
print("=== grid-centering convention: odd N ===")
from lumenairy.elements import polarization as P
import lumenairy.elements.elements as EL
nx = 7; dxg = 1e-6
print("  sources/elements convention (arange(N)-N/2):", np.round(((np.arange(nx) - nx / 2) * dxg) * 1e6, 3))
print("  polarization._plane_wave_carrier (arange-nx//2):", np.round(((np.arange(nx) - nx // 2) * dxg) * 1e6, 3))
c = P._plane_wave_carrier(0.5, 0.0, 1e-6, nx, nx, dxg, dxg)
print("  carrier phase along the row (deg):", np.round(np.angle(c[0]) * 180 / np.pi, 2))
# where is the zero-phase sample?
print("  zero-phase index:", int(np.argmin(np.abs(np.angle(c[0])))), " (nx//2 =", nx // 2, ")")
# apply_jones_matrix callable grid
jf = P.JonesField(np.ones((nx, nx), complex), np.zeros((nx, nx), complex), dxg)
seen = {}
def cb(X, Y):
    seen['x'] = X[0]
    J = np.zeros((2, 2, nx, nx), complex); J[0, 0] = 1; J[1, 1] = 1
    return J
P.apply_jones_matrix(jf, cb)
print("  apply_jones_matrix callable grid x (um):", np.round(seen['x'] * 1e6, 3))
