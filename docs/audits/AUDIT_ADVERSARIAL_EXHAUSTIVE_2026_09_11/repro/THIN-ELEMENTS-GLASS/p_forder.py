import numpy as np, sys
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.lenses import surface_sag_general
import lumenairy.elements.lenses as L

def ref(h_sq, R, k, co):
    c=1.0/R; n=(1+k)*h_sq/R**2
    s=np.where(n<0.9999, h_sq/(R*(1+np.sqrt(np.where(n<0.9999,1-n,0.01)))), np.nan)
    for p,a in co.items(): s = s + a*h_sq**(p//2)
    return s

x = np.linspace(-9e-3, 9e-3, 6); y = np.linspace(-4e-3, 4e-3, 4)
X, Y = np.meshgrid(x, y)                 # (4,6) C-order
hsq_C = X**2 + Y**2
co = {4: 1.0e3}
R, k = 50e-3, 0.0
print("numba available:", L._NUMBA_AVAILABLE)

for lbl, arr in [("C-contig (4,6)", hsq_C),
                 ("F-order  (4,6)", np.asfortranarray(hsq_C)),
                 ("transposed view (6,4)", hsq_C.T),
                 ("C-copy of transpose (6,4)", np.ascontiguousarray(hsq_C.T))]:
    got = surface_sag_general(arr, R, k, dict(co))
    want = ref(np.asarray(arr, dtype=float), R, k, co)
    err = np.max(np.abs(got - want))
    print(f"  {lbl:28s} flags C={arr.flags['C_CONTIGUOUS']} F={arr.flags['F_CONTIGUOUS']}"
          f"  max|err| = {err:.3e}   rel_to_asph_term = {err/np.max(np.abs(co[4]*np.asarray(arr)**2)):.3f}")

print("\nDetail for the F-order case:")
a = np.asfortranarray(hsq_C)
got = surface_sag_general(a, R, k, dict(co))
want = ref(np.asarray(a), R, k, co)
print("  got :\n", got)
print("  want:\n", want)
print("  (got-conic_only)/A4  vs  h_sq^2:")
conic = surface_sag_general(a, R, k, None)
print("   recovered h^4 :\n", (got-conic)/co[4])
print("   true      h^4 :\n", np.asarray(a)**2)
