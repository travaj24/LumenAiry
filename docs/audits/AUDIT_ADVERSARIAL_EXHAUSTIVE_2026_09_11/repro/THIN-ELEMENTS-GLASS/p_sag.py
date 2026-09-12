import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.lenses import surface_sag_general, surface_sag_biconic
import lumenairy.elements.lenses as L

def ref_conic(h, R, k):
    c = 1.0/R
    rad = 1 - (1+k)*c*c*h*h
    return np.where(rad>=0, c*h*h/(1+np.sqrt(np.maximum(rad,0))), np.nan)

print("=== surface_sag_general vs independent conic formula ===")
h = np.linspace(0, 9e-3, 11)
for R,k,lbl in [(50e-3,0.0,'sphere'),(50e-3,-1.0,'parabola'),(50e-3,-3.0,'hyperbola'),
                (50e-3,+2.0,'oblate ell.'),(-50e-3,0.0,'neg sphere')]:
    s = surface_sag_general(h**2, R, k)
    r = ref_conic(h, R, k)
    d = np.nanmax(np.abs(s-r))
    print(f"  {lbl:12s} R={R*1e3:+6.1f}mm k={k:+4.1f}  max|sag-ref| = {d:.3e}   sag(9mm)={s[-1]:.6e}")

print("\n=== behaviour beyond the conic domain (1-(1+k)c^2h^2 < 0) ===")
R, k = 10e-3, 3.0     # domain: h < R/sqrt(1+k) = 5 mm
hh = np.array([4.9e-3, 4.99e-3, 4.9995e-3, 5.0e-3, 5.1e-3, 20e-3])
s = surface_sag_general(hh**2, R, k)
print("  h[mm] :", hh*1e3)
print("  sag   :", s)
print("  ref   :", ref_conic(hh, R, k))
print("  -> library NaNs at norm>=0.9999 i.e. h >= %.6f mm (true limit %.6f mm)" %
      (1e3*R*np.sqrt(0.9999/(1+k)), 1e3*R/np.sqrt(1+k)))

print("\n=== R = inf / None / 0 ===")
for R in (np.inf, None, 0.0, -0.0):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        try:
            s = surface_sag_general(np.array([0.0, 1e-6]), R, 0.0)
            print(f"  R={R!r:8s} -> {s}  warnings={[str(x.message)[:50] for x in w]}")
        except Exception as e:
            print(f"  R={R!r:8s} -> {type(e).__name__}: {e}")

print("\n=== odd aspheric powers ===")
for coeffs in ({4:1e6}, {5:1e6}, {3:1e4}):
    try:
        s = surface_sag_general(np.array([1e-4]), np.inf, 0.0, coeffs)
        print(f"  {coeffs} -> {s}")
    except Exception as e:
        print(f"  {coeffs} -> {type(e).__name__}: {str(e)[:110]}")

print("\n=== numba kernel vs pure-numpy fallback (bitwise?) ===")
rng = np.random.default_rng(0)
hsq = (rng.random((256,256))*9e-3)**2
co = {4: 1.234e3, 6: -5.6e7, 8: 9.9e11, 10: -1.1e16}
sag_numba = surface_sag_general(hsq.copy(), 50e-3, -0.5, dict(co))
# force fallback
orig = L._NUMBA_AVAILABLE
L._NUMBA_AVAILABLE = False
sag_np = surface_sag_general(hsq.copy(), 50e-3, -0.5, dict(co))
L._NUMBA_AVAILABLE = orig
print("  numba avail:", orig, " max|diff| =", np.nanmax(np.abs(sag_numba-sag_np)),
      " bit-identical:", np.array_equal(sag_numba, sag_np, equal_nan=True))
print("  rel:", np.nanmax(np.abs(sag_numba-sag_np))/np.nanmax(np.abs(sag_np)))

print("\n=== numba kernel + NaN conic region (fastmath NaN propagation) ===")
hsq2 = np.array([[0.0, (4e-3)**2, (20e-3)**2]])   # last is outside domain for R=10mm k=3
s_nb = surface_sag_general(hsq2.copy(), 10e-3, 3.0, {4: 1e6})
L._NUMBA_AVAILABLE = False
s_np = surface_sag_general(hsq2.copy(), 10e-3, 3.0, {4: 1e6})
L._NUMBA_AVAILABLE = orig
print("  numba:", s_nb, "\n  numpy:", s_np)

print("\n=== F-order / non-contiguous h_sq into the numba path ===")
hF = np.asfortranarray(hsq[:8,:8])
try:
    a = surface_sag_general(hF, 50e-3, 0.0, {4:1e3})
    L._NUMBA_AVAILABLE = False
    b = surface_sag_general(np.ascontiguousarray(hF), 50e-3, 0.0, {4:1e3})
    L._NUMBA_AVAILABLE = orig
    print("  F-order OK, max|diff| vs C-order numpy =", np.max(np.abs(a-b)))
except Exception as e:
    L._NUMBA_AVAILABLE = orig
    print("  F-order ->", type(e).__name__, str(e)[:160])

print("\n=== float32 h_sq path ===")
h32 = (np.linspace(0,9e-3,5)**2).astype(np.float32)
s32 = surface_sag_general(h32, 50e-3, 0.0, {4:1e3})
s64 = surface_sag_general(h32.astype(np.float64), 50e-3, 0.0, {4:1e3})
print("  dtype out:", s32.dtype, " max|f32-f64| =", np.max(np.abs(s32.astype(np.float64)-s64)))

print("\n=== surface_sag_biconic: separable vs Zemax biconic ===")
X, Y = np.meshgrid(np.linspace(0,10e-3,5), np.linspace(0,10e-3,5))
Rx, Ry, kx, ky = 50e-3, 80e-3, -0.5, 0.3
sb = surface_sag_biconic(X, Y, R_x=Rx, R_y=Ry, conic_x=kx, conic_y=ky)
cx, cy = 1/Rx, 1/Ry
zem = (cx*X**2 + cy*Y**2)/(1+np.sqrt(1-(1+kx)*cx**2*X**2-(1+ky)*cy**2*Y**2))
print("  max|separable - Zemax| =", np.max(np.abs(sb-zem)), "  (max sag ~", np.max(np.abs(sb)), ")")
print("  on-axis y=0 agreement:", np.max(np.abs(sb[0,:]-zem[0,:])))
print("  corner (10,10)mm: sep=%.6e  zemax=%.6e  rel=%.3f%%" % (sb[-1,-1], zem[-1,-1], 100*abs(sb[-1,-1]-zem[-1,-1])/abs(zem[-1,-1])))
