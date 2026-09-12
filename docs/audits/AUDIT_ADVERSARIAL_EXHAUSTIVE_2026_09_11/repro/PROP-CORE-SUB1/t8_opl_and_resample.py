"""(a) hf.propagate_huygens_fresnel_with_opl_callable vs exact Fresnel /
       exact Rayleigh-Sommerfeld-I quadrature (obliquity check)
   (b) the resample + Parseval-renormalisation energy falsification
       shared by hf.py:171-187 and mhs.py:621-641
   (c) timing / O(N^4) scaling of the OPL-callable path
"""
import sys, time, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.hf import (
    propagate_huygens_fresnel_with_opl_callable,
    propagate_huygens_fresnel_freespace,
)
from lumenairy.propagators.mft import resample_field

lam = 633e-9; k = 2*np.pi/lam
print("=" * 72)
print("(a) OPL-callable HF kernel vs exact quadrature")
print("=" * 72)
N, dx = 48, 2e-6
z = 200e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
E0 = np.exp(-(np.hypot(X, Y)/8e-6)**2).astype(np.complex128)
ox = (np.arange(9)-4)*dx
oy = (np.arange(9)-4)*dx


def opl(s1x, s1y, s2x, s2y):            # WAVES, exact spherical OPL
    return np.sqrt((s1x-s2x)**2 + (s1y-s2y)**2 + z*z)/lam


t0 = time.perf_counter()
out = propagate_huygens_fresnel_with_opl_callable(
    E0, opl_fn=opl, output_grid_x=ox, output_grid_y=oy,
    input_grid_dx=dx, apply_van_vleck=True)
t_opl = time.perf_counter()-t0

OX, OY = np.meshgrid(ox, oy)
# exact direct sums on the SAME discretisation
ref_fres = np.empty_like(out); ref_rs1 = np.empty_like(out)
ref_kirch = np.empty_like(out)
for i in range(9):
    for j in range(9):
        r = np.sqrt((X-OX[i, j])**2 + (Y-OY[i, j])**2 + z*z)
        ph = np.exp(1j*k*r)
        # Fresnel / paraxial Huygens: 1/(i lam z) exp(ikr)
        ref_fres[i, j] = np.sum(E0*ph)*dx*dx/(1j*lam*z)
        # Rayleigh-Sommerfeld I: (1/(i lam)) (z/r) exp(ikr)/r
        ref_rs1[i, j] = np.sum(E0*(z/r)*ph/r)*dx*dx/(1j*lam)
        # Kirchhoff: (1/(i lam)) (1+cos)/2 exp(ikr)/r
        ref_kirch[i, j] = np.sum(E0*0.5*(1+z/r)*ph/r)*dx*dx/(1j*lam)


def rel(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


print(f"  grid {N}x{N} -> 9x9 output, z={z*1e6:.0f}um, "
      f"max half-angle {np.degrees(np.arctan(N*dx/2/z)):.1f} deg")
print(f"  runtime {t_opl:.3f}s")
print(f"  relL2 vs Fresnel 1/(i lam z)          : {rel(out, ref_fres):.4e}")
print(f"  relL2 vs Rayleigh-Sommerfeld I cos/r  : {rel(out, ref_rs1):.4e}")
print(f"  relL2 vs Kirchhoff (1+cos)/2 /r       : {rel(out, ref_kirch):.4e}")
print(f"  (out/ref_fres) mean ratio             : "
      f"{np.mean(out/ref_fres):.6f}")
print(f"  -> the OPL-callable kernel carries NO obliquity and NO 1/r;")
print(f"     its amplitude is sqrt|det d2Phi/ds1 ds2| = 1/(lam z) exactly.")

# what does the Van Vleck density actually evaluate to?
h = 1e-6
s2x = s2y = 0.0
pxx = (opl(X+h, Y, s2x+h, s2y)-opl(X+h, Y, s2x-h, s2y)
       - opl(X-h, Y, s2x+h, s2y)+opl(X-h, Y, s2x-h, s2y))/(4*h*h)
pyy = (opl(X, Y+h, s2x, s2y+h)-opl(X, Y+h, s2x, s2y-h)
       - opl(X, Y-h, s2x, s2y+h)+opl(X, Y-h, s2x, s2y-h))/(4*h*h)
pxy = (opl(X+h, Y, s2x, s2y+h)-opl(X+h, Y, s2x, s2y-h)
       - opl(X-h, Y, s2x, s2y+h)+opl(X-h, Y, s2x, s2y-h))/(4*h*h)
pyx = (opl(X, Y+h, s2x+h, s2y)-opl(X, Y+h, s2x-h, s2y)
       - opl(X, Y-h, s2x+h, s2y)+opl(X, Y-h, s2x-h, s2y))/(4*h*h)
dens = np.sqrt(np.abs(pxx*pyy-pxy*pyx))
print(f"  sqrt|det| on axis = {dens[N//2, N//2]:.6e}   1/(lam z) = "
       f"{1/(lam*z):.6e}   ratio {dens[N//2, N//2]*lam*z:.6f}")
print(f"  sqrt|det| at corner = {dens[0, 0]:.6e}   "
      f"RS-I would need cos/(lam r) = "
      f"{(z/np.sqrt((X[0,0])**2+(Y[0,0])**2+z*z))/(lam*np.sqrt(X[0,0]**2+Y[0,0]**2+z*z)):.6e}"
      f"   ratio {dens[0,0]/((z/np.sqrt(X[0,0]**2+Y[0,0]**2+z*z))/(lam*np.sqrt(X[0,0]**2+Y[0,0]**2+z*z))):.4f}")

print()
print("  timing scaling (O(N_in^2 * N_out^2)):")
for (Ni, No) in ((32, 4), (32, 8), (64, 4), (64, 8)):
    xi = (np.arange(Ni)-Ni/2)*dx
    Xi, Yi = np.meshgrid(xi, xi)
    Ei = np.exp(-(np.hypot(Xi, Yi)/8e-6)**2).astype(np.complex128)
    go = (np.arange(No)-No/2)*dx
    t0 = time.perf_counter()
    propagate_huygens_fresnel_with_opl_callable(
        Ei, opl_fn=opl, output_grid_x=go, output_grid_y=go,
        input_grid_dx=dx, apply_van_vleck=True)
    dt = time.perf_counter()-t0
    print(f"    N_in={Ni:3d} N_out={No:2d}  {dt:7.3f}s  "
          f"-> extrapolated 128^2 -> 128^2: "
          f"{dt*(128/Ni)**2*(128/No)**2/3600:.2f} h")

print()
print("=" * 72)
print("(b) resample + Parseval renormalisation: energy falsification")
print("=" * 72)
N, dx = 64, 2e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
E = np.exp(-(np.hypot(X, Y)/20e-6)**2).astype(np.complex128)
p_full = float(np.sum(np.abs(E)**2))*dx**2
print(f"  input: N={N} dx={dx*1e6:.1f}um, window +-{N*dx/2*1e6:.0f}um, "
      f"Gaussian w0=20um, total power {p_full:.6e}")
for fac in (0.5, 0.25):
    dx_out = dx*fac                       # SHRINK the window by `fac`
    Er, dxr = resample_field(E, dx, dx_out, N)
    p_raw = float(np.sum(np.abs(Er)**2))*dx_out**2
    # truth: power actually inside the new, smaller window
    half = N*dx_out/2
    m = (np.abs(X) < half) & (np.abs(Y) < half)
    p_true = float(np.sum(np.abs(E[m])**2))*dx**2
    scale = np.sqrt(p_full/p_raw)
    print(f"  dx_out = {dx_out*1e6:.2f}um -> window +-{half*1e6:.0f}um")
    print(f"     true power inside new window   = {p_true:.6e}  "
          f"({100*p_true/p_full:.2f}% of input)")
    print(f"     resample_field raw power       = {p_raw:.6e}  "
          f"({100*p_raw/p_full:.2f}%)")
    print(f"     library renorm factor sqrt(p_in/p_out) = {scale:.6f}")
    print(f"     -> peak amplitude inflated by {scale:.4f}x "
          f"({100*(scale-1):.1f}% too bright) to fake energy conservation")

print()
print("  end-to-end via hf.propagate_huygens_fresnel_freespace(output_dx=...):")
Ef, dxf = propagate_huygens_fresnel_freespace(
    E, 1e-3, lam, dx, output_dx=dx*0.25)
Enat = propagate_huygens_fresnel_freespace(E, 1e-3, lam, dx)
p_nat = float(np.sum(np.abs(Enat)**2))*dx**2
p_out = float(np.sum(np.abs(Ef)**2))*dxf**2
half = N*dxf/2
m = (np.abs(X) < half) & (np.abs(Y) < half)
p_true = float(np.sum(np.abs(Enat[m])**2))*dx**2
print(f"     native-grid power {p_nat:.6e}; window +-{half*1e6:.1f}um holds "
      f"{100*p_true/p_nat:.2f}% of it")
print(f"     returned power    {p_out:.6e}  = {100*p_out/p_nat:.2f}% "
      f"of the native power  <-- should be {100*p_true/p_nat:.2f}%")
print(f"     return type: {type(Ef).__name__} + dx (tuple)   vs bare ndarray "
      f"when no output_dx is passed: {type(Enat).__name__}")
