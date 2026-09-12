import sys, numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.algebra.from_prescription import from_prescription
from lumenairy.algebra.primitives import FreeSpace, ThinLens
from lumenairy.raytrace.seidel import system_abcd
from lumenairy.raytrace import surfaces_from_prescription

wl = 633e-9
print("=== A. from_prescription ABCD vs raytrace.system_abcd ===")
cases = {
    'singlet':  la.make_singlet(R1=50e-3, R2=-50e-3, d=5e-3, glass='N-BK7', aperture=25.4e-3),
    'meniscus': la.make_singlet(R1=30e-3, R2=80e-3, d=4e-3, glass='N-SF11', aperture=20e-3),
    'doublet':  la.make_doublet(R1=60e-3, R2=-40e-3, R3=-150e-3, d1=6e-3, d2=3e-3,
                                glass1='N-BK7', glass2='N-SF5', aperture=25e-3),
}
for name, rx in cases.items():
    try:
        op = from_prescription(rx, wl)
        surfs = surfaces_from_prescription(rx)
        M_ref = system_abcd(surfs, wl)[0]
        M_alg = op.abcd
        print("  %-9s max|dABCD| = %.3e" % (name, np.abs(np.asarray(M_alg) - np.asarray(M_ref)).max()))
        print("     algebra:", np.round(M_alg, 10).tolist())
        print("     raytrace:", np.round(np.asarray(M_ref), 10).tolist())
        print("     efl(algebra) = %.9f  efl(raytrace -1/C) = %.9f"
              % (op.efl, -1.0 / np.asarray(M_ref)[1, 0]))
    except Exception as ex:
        print("  %-9s FAILED: %s: %s" % (name, type(ex).__name__, str(ex)[:150]))

print("")
print("=== B. composition order A*B == 'apply B first' ===")
f1, d1 = 100e-3, 50e-3
A = ThinLens(f1); B = FreeSpace(d1)
M = (A * B).abcd
ref = np.array([[1, 0], [-1 / f1, 1]]) @ np.array([[1, d1], [0, 1]])
print("  (ThinLens*FreeSpace).abcd == L@D ?", np.allclose(M, ref), "  max diff", np.abs(M - ref).max())
print("  matrix:", np.round(M, 10).tolist(), " ref:", np.round(ref, 10).tolist())

print("")
print("=== C. thick lens: algebra chain vs analytic thick-lens ABCD ===")
n = la.get_glass_index('N-BK7', wl)
R1, R2, d = 50e-3, -50e-3, 5e-3
# analytic thick lens (air both sides), Welford/Hecht
P1 = (n - 1.0) / R1
P2 = (1.0 - n) / R2
M_an = (np.array([[1, 0], [-P2, 1]]) @ np.array([[1, d / n], [0, 1]])
        @ np.array([[1, 0], [-P1, 1]]))
op = from_prescription(cases['singlet'], wl)
print("  analytic thick-lens ABCD:", np.round(M_an, 10).tolist())
print("  algebra  chain      ABCD:", np.round(op.abcd, 10).tolist())
print("  max diff:", np.abs(np.asarray(op.abcd) - M_an).max())
f_an = -1.0 / M_an[1, 0]
print("  analytic EFL = %.9f m ; lensmaker 1/((n-1)(1/R1-1/R2+(n-1)d/(n R1 R2))) = %.9f m"
      % (f_an, 1.0 / ((n - 1) * (1 / R1 - 1 / R2 + (n - 1) * d / (n * R1 * R2)))))

print("")
print("=== D. anamorphic handling ===")
try:
    from lumenairy.algebra.primitives import CylindricalLens
    cl = CylindricalLens(f=100e-3, axis='x')
    print("  CylindricalLens is_anamorphic:", cl.is_anamorphic)
    print("  abcd_x:", np.round(cl.abcd_x, 8).tolist(), " abcd_y:", np.round(cl.abcd_y, 8).tolist())
    comp = FreeSpace(50e-3) * cl
    print("  composite anamorphic:", comp.is_anamorphic)
    try:
        comp.abcd
        print("  composite.abcd did NOT raise on an anamorphic system  <-- unexpected")
    except ValueError as ex:
        print("  composite.abcd raises for anamorphic (as documented):", str(ex)[:80])
except Exception as ex:
    print("  cylindrical probe:", type(ex).__name__, str(ex)[:160])

print("")
print("=== E. 4f system algebraic identity: FreeSpace(f) L(f) FreeSpace(2f) L(f) FreeSpace(f) ===")
f = 200e-3
S = (FreeSpace(f) * ThinLens(f) * FreeSpace(2 * f) * ThinLens(f) * FreeSpace(f))
print("  4f ABCD:", np.round(S.abcd, 12).tolist(), " (expect [[-1,0],[0,-1]])")
print("  max|M - (-I)| =", np.abs(np.asarray(S.abcd) + np.eye(2)).max())

print("")
print("=== F. algebra field application vs direct propagation (4f imaging) ===")
N, dx = 256, 8e-6
E, x, y = la.create_gaussian_beam(N, dx, wl, w0=100e-6)
out = S(E, dx=dx, wavelength=wl)
Eo = out[0] if isinstance(out, tuple) else getattr(out, 'E', out)
Eo = np.asarray(Eo)
I0 = np.abs(E) ** 2; I1 = np.abs(Eo) ** 2
print("  input power %.6f  output power %.6f  ratio %.4f"
      % (I0.sum() * dx * dx, I1.sum() * dx * dx, I1.sum() / I0.sum()))
print("  output peak at", np.unravel_index(np.argmax(I1), I1.shape), " centre =", N // 2)
