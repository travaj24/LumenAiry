"""Probe 14 -- recompute EXACTLY the quantities the test file's bars cite, so
the build doc's tables and the assertion comments carry the same numbers."""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

import tests.unit.test_pmm2d_staggered_anisotropic as t  # noqa: E402

print("== G2 (Granet sign discriminator, degree=6) ==")
tp, tm = t._granet(t._GRANET_B, t._GRANET_A)
up, um = t._granet(np.conj(t._GRANET_B), np.conj(t._GRANET_A))
cp, cm = t._granet(t._ISO, 2.25 * np.eye(3, dtype=complex))
print(f"  conjugated   T(1,1)={tp:.6f} T(-1,1)={tm:.6f}  diff={tp-tm:+.4e}")
print(f"  unconjugated T(1,1)={up:.6f} T(-1,1)={um:.6f}  diff={up-um:+.4e}")
print(f"  control(iso) T(1,1)={cp:.6f} T(-1,1)={cm:.6f}  diff={cp-cm:+.4e}")

print("== G3 (Berreman residual, R/T/Jones combined) ==")
for nm, t33 in (("lc", t._LC), ("gyro", t._GYRO)):
    for th, ph in ((0.0, 0.0), (25 * np.pi / 180, 0.0),
                   (25 * np.pi / 180, 40 * np.pi / 180)):
        for n in (2, 3):
            for M in (5, 7):
                r = t._g3_residual(t33, n, M, th, ph)
                print(f"  {nm:4s} n={n} M={M} theta={np.degrees(th):4.1f} "
                      f"phi={np.degrees(ph):4.1f}  {r:.3e}")

print("== G4 (1-D reduction ladder, theta=0.22) ==")
orc = t.pmm_jones_1d(t._G4_P, t._G4_RIDGE, t._G4_GROOVE, 1.5, 1.0, t._G4_DEP,
                     0.5, t._G4_WL, angle=0.22, degree=16, stabilize=False)
for M in (5, 6, 7, 8):
    dRT, dJ, forb = t._g4_residual(M, orc)
    print(f"  M={M}  max|dR|,|dT| = {dRT:.3e}   max|dJones| = {dJ:.3e}"
          f"   y-forbidden = {forb:.2e}")

print("== G7 (symmetry residuals) ==")
A = t._cell(t._LC, t._ISO)
print("  LC transpose        ", ["%.3e" % v for v in
                                 t._transpose_residual(A, t._transpose_cell(A))])
Ag = t._cell(t._GYRO, t._ISO)
Bg = t._transpose_cell(Ag)
Agw = Ag.copy()
Agw[..., 0, 1], Agw[..., 1, 0] = Ag[..., 1, 0].copy(), Ag[..., 0, 1].copy()
Aw = A.copy()
Aw[..., 0, 1], Aw[..., 1, 0] = A[..., 1, 0].copy(), A[..., 0, 1].copy()
print("  LC   swapped-block  ", ["%.3e" % v for v in
                                 t._transpose_residual(Aw, t._transpose_cell(A))])
print("  gyro transpose      ", ["%.3e" % v for v in
                                 t._transpose_residual(Ag, Bg)])
print("  gyro swapped-block  ", ["%.3e" % v for v in
                                 t._transpose_residual(Agw, Bg)])

print("== G8b ladder ==")
for M in (5, 7):
    print(f"  M={M}  {t._g8b_residual(M):.3e}")

print("== G10 off-plane noise of uniaxial_tensor(1.5,1.8,pi/2,0.55) ==")
off = float(np.abs(t._LC[[0, 1, 2, 2], [2, 2, 0, 1]]).max())
print(f"  max|off-plane| = {off:.3e}   scale = {float(np.abs(t._LC).max()):.3f}"
      f"   relative = {off/float(np.abs(t._LC).max()):.3e}"
      f"   floor = {1e-12*float(np.abs(t._LC).max()):.3e}")
