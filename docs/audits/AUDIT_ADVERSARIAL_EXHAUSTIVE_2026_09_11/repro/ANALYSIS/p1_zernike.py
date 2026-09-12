"""ANALYSIS probe 1: Zernike indexing, normalisation, orthogonality, fitting."""
import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.analysis.zernike import (
    zernike_index_to_nm, zernike_nm_to_index, zernike_polynomial,
    zernike_basis_matrix, zernike_decompose, zernike_reconstruct,
    _zernike_classical_name, astigmatism_mag_angle)

print("=== 1. OSA index -> (n,m) for j=0..37 ===")
bad = []
for j in range(38):
    n, m = zernike_index_to_nm(j)
    jr = zernike_nm_to_index(n, m)
    ok = (jr == j) and (abs(m) <= n) and ((n - abs(m)) % 2 == 0)
    if not ok:
        bad.append((j, n, m, jr))
    print(f"  j={j:2d} -> (n={n},m={m:+d})  roundtrip={jr:2d} {'OK' if ok else 'BAD'}  {_zernike_classical_name(n,m)}")
print("  BAD:", bad)

print()
print("=== 2. Gram matrix of first 28 modes on unit disc, N=512 ===")
N = 512
x = (np.arange(N) - N / 2 + 0.5) / (N / 2)   # symmetric cell-centred
X, Y = np.meshgrid(x, x)
R = np.sqrt(X**2 + Y**2)
TH = np.arctan2(Y, X)
mask = R <= 1.0
nmodes = 28
B = np.empty((mask.sum(), nmodes))
for j in range(nmodes):
    n, m = zernike_index_to_nm(j)
    B[:, j] = zernike_polynomial(n, m, R[mask], TH[mask])
G = (B.T @ B) / mask.sum()
off = G - np.diag(np.diag(G))
print(f"  max |diag - 1|     = {np.abs(np.diag(G)-1).max():.3e}")
print(f"  max |offdiag|      = {np.abs(off).max():.3e}")
i, k = np.unravel_index(np.abs(off).argmax(), off.shape)
print(f"  worst pair         = ({i},{k}) value {off[i,k]:.3e}")

print()
print("=== 3. RMS normalisation of each mode (should be 1.0) ===")
rms = np.sqrt((B**2).mean(axis=0))
print("  rms per mode:", np.array2string(rms, precision=5, max_line_width=200))
print(f"  max dev from 1 = {np.abs(rms-1).max():.3e}")

print()
print("=== 4. Fit round-trip: inject known coefficients, recover ===")
Ngrid, dx, ap = 256, 1e-4, 256 * 1e-4 * 0.8
c_true = np.zeros(21)
c_true[2] = 1.3e-6   # tilt X (OSA j=2 -> (1,1) cos -> x tilt)
c_true[1] = -0.7e-6  # tilt Y
c_true[4] = 2.0e-6   # defocus
c_true[7] = 0.5e-6   # vertical coma (3,-1)
c_true[12] = 0.9e-6  # primary spherical
opd = zernike_reconstruct(c_true, dx, (Ngrid, Ngrid), ap)
c_fit, names = zernike_decompose(opd, dx, ap, n_modes=21)
print("  max |c_fit - c_true| =", f"{np.abs(c_fit - c_true).max():.3e}")
for j in np.nonzero(c_true)[0]:
    print(f"   j={j:2d} {names[j]:28s} true={c_true[j]: .4e} fit={c_fit[j]: .4e}")

print()
print("=== 5. Tilt-X sign: does OSA j=2 vary along array COLUMNS (x)? ===")
c = np.zeros(6); c[2] = 1.0
m2 = zernike_reconstruct(c, dx, (64, 64), 64*dx*0.8)
print(f"  d/dcol (row 32): {m2[32, 40] - m2[32, 24]:+.4f}   (expect >0 for +x tilt)")
print(f"  d/drow (col 32): {m2[40, 32] - m2[24, 32]:+.4f}   (expect 0)")
c = np.zeros(6); c[1] = 1.0
m1 = zernike_reconstruct(c, dx, (64, 64), 64*dx*0.8)
print(f"  j=1 d/dcol: {m1[32, 40] - m1[32, 24]:+.4f} (expect 0)")
print(f"  j=1 d/drow: {m1[40, 32] - m1[24, 32]:+.4f} (expect >0 => row=+y)")

print()
print("=== 6. astigmatism_mag_angle ===")
for (c3, c5, lbl) in [(0, 1, 'pure vertical (0/90)'), (1, 0, 'pure oblique (45)'),
                      (1, 1, 'mix')]:
    cc = np.zeros(6); cc[3] = c3; cc[5] = c5
    mag, th = astigmatism_mag_angle(cc)
    print(f"  c3={c3} c5={c5} ({lbl}): mag={mag:.4f} theta={np.degrees(th):+8.3f} deg")

print()
print("=== 7. Cache-key soundness: does dy!=dx alias? ===")
from lumenairy.analysis import zernike as Z
Z.clear_zernike_basis_cache()
o1 = zernike_reconstruct(np.array([0., 0, 0, 0, 1.0]), 1e-4, (64, 64), 64e-4*0.8)
o2 = zernike_reconstruct(np.array([0., 0, 0, 0, 1.0]), 1e-4, (64, 64), 64e-4*0.8, dy=2e-4)
print(f"  identical when dy doubled? {np.allclose(o1, o2)}  (should be False)")
print(f"  max|o1-o2| = {np.abs(o1-o2).max():.4e}")

print()
print("=== 8. Annular / obscured pupil support? ===")
print("  functions exported:", Z.__all__)
