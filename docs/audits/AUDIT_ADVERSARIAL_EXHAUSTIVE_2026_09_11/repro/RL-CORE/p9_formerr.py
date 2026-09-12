"""Probe 9: form_error handling (shape / dtype / decenter interplay),
sag_callable, biconic radius_y, and anamorphic dy."""
import sys, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import numpy as np
from lumenairy.elements._lens_real import apply_real_lens
lam = 632.8e-9; k0 = 2*np.pi/lam
N, dx = 128, 2e-5
E = np.ones((N, N), dtype=np.complex128)
def rx(**kw):
    s0 = dict(radius=50e-3, glass_before='AIR', glass_after='N-BK7'); s0.update(kw)
    return dict(surfaces=[s0, dict(radius=float('inf'), glass_before='N-BK7',
                                   glass_after='AIR')], thicknesses=[1e-3])

print("=== form_error: SHAPE mismatch ===")
for shp in ((N, N), (N//2, N//2), (N,), (N, N, 1)):
    fe = np.full(shp, 1e-7)
    try:
        Eo = apply_real_lens(E.copy(), prescription=rx(form_error=fe),
                             wavelength=lam, dx=dx)
        print(f"  shape {str(shp):<12} ACCEPTED -> out shape {Eo.shape}")
    except Exception as e:
        print(f"  shape {str(shp):<12} {type(e).__name__}: {str(e)[:70]}")

print()
print("=== form_error: value applied with the right sign / magnitude ===")
fe = np.full((N, N), 100e-9)
A = apply_real_lens(E.copy(), prescription=rx(), wavelength=lam, dx=dx)
B = apply_real_lens(E.copy(), prescription=rx(form_error=fe), wavelength=lam, dx=dx)
d = np.angle(B[N//2, N//2]/A[N//2, N//2])
n = 1.5150891983370924
print(f"  d(phase) = {d:+.6f} rad; expected -k0*(n2-n1)*100nm = "
      f"{-k0*(n-1)*100e-9:+.6f} rad")

print()
print("=== form_error + DECENTER: is the map shifted with the surface? ===")
fe2 = np.zeros((N, N)); fe2[N//2, N//2+10] = 1e-6
C = apply_real_lens(E.copy(), prescription=rx(form_error=fe2,
                                              decenter=(20*dx, 0.0)),
                    wavelength=lam, dx=dx)
D = apply_real_lens(E.copy(), prescription=rx(form_error=fe2),
                    wavelength=lam, dx=dx)
i1 = np.unravel_index(np.argmax(np.abs(np.angle(C/A))), C.shape)
i2 = np.unravel_index(np.argmax(np.abs(np.angle(D/A))), D.shape)
print(f"  peak of the form-error imprint: decentered {i1}, undecentered {i2}")
print("  (identical => form_error is in the FIELD frame, NOT carried with the "
      "surface decenter)")

print()
print("=== form_error dtype: float32 map with float64 geometry ===")
fe32 = np.full((N, N), 100e-9, dtype=np.float32)
F = apply_real_lens(E.copy(), prescription=rx(form_error=fe32),
                    wavelength=lam, dx=dx)
print(f"  out dtype {F.dtype};  max|d| vs float64 map = {np.abs(F-B).max():.3e}")

print()
print("=== anamorphic dy != dx ===")
try:
    G = apply_real_lens(E.copy(), prescription=rx(), wavelength=lam,
                        dx=dx, dy=dx/2)
    print(f"  dy=dx/2 accepted; peak |E| = {np.abs(G).max():.4f}")
except Exception as e:
    print(f"  {type(e).__name__}: {e}")

print()
print("=== biconic radius_y ===")
try:
    H = apply_real_lens(E.copy(), prescription=rx(radius_y=80e-3),
                        wavelength=lam, dx=dx)
    a = np.angle(H); 
    print(f"  radius_y accepted; phase at (x=1mm,y=0) {a[N//2, N//2+50]:+.4f} "
          f"vs (x=0,y=1mm) {a[N//2+50, N//2]:+.4f}  (should differ)")
except Exception as e:
    print(f"  {type(e).__name__}: {str(e)[:80]}")

print()
print("=== is_mirror / MIRROR surfaces ===")
for key in (dict(is_mirror=True), dict(glass_after='MIRROR')):
    try:
        apply_real_lens(E.copy(), prescription=rx(**key), wavelength=lam, dx=dx)
        print(f"  {key} ACCEPTED (should raise)")
    except Exception as e:
        print(f"  {key} -> {type(e).__name__}: {str(e)[:60]}")
