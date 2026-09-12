import numpy as np, sys, warnings
sys.path.insert(0,r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
import lumenairy as la
from lumenairy.elements import apply_thin_lens, apply_spherical_lens, apply_grin_lens, apply_cylindrical_lens, apply_axicon
warnings.simplefilter('ignore')

wl = 1.0e-6; N = 2048; dx = 2.0e-6
x = (np.arange(N)-N/2)*dx
X, Y = np.meshgrid(x, x)
f = 20e-3
# Gaussian-apodised plane wave to avoid hard-edge ringing (w=1mm -> NA ~ 0.05)
w0 = 1.0e-3
E0 = np.exp(-(X**2+Y**2)/w0**2).astype(np.complex128)

def peak_z(E, zs):
    out=[]
    for z in zs:
        Ez = la.angular_spectrum_propagate(E, z, wl, dx)
        out.append(np.max(np.abs(Ez)**2))
    return np.array(out)

print("=== apply_thin_lens: converging sign under exp(+ikz) ===")
for model in ('paraxial','nonparaxial','stigmatic'):
    E = apply_thin_lens(E0, f=f, wavelength=wl, dx=dx, lens_model=model)
    zs = np.linspace(0.9*f, 1.1*f, 21)
    I = peak_z(E, zs)
    zf = zs[np.argmax(I)]
    # refine
    zs2 = np.linspace(zf-0.006*f, zf+0.006*f, 25); I2 = peak_z(E, zs2); zf2 = zs2[np.argmax(I2)]
    print(f"  {model:12s} peak at z = {zf2*1e3:.4f} mm   (f = {f*1e3:.4f} mm)  err = {(zf2-f)*1e6:+.2f} um  Imax={I2.max():.4g}")
# diverging f<0 must NOT focus downstream
for model in ('paraxial','nonparaxial'):
    E = apply_thin_lens(E0, f=-f, wavelength=wl, dx=dx, lens_model=model)
    zs = np.linspace(0.5*f, 1.5*f, 15); I = peak_z(E, zs)
    print(f"  f<0 {model:10s} max peak in [0.5f,1.5f] = {I.max():.4g}  (collimated Imax={np.max(np.abs(E0)**2):.4g}) -> diverging OK={I.max()<1.0}")

print("\n=== apply_thin_lens: aplanatic domain / clipping check ===")
Ea = apply_thin_lens(E0, f=f, wavelength=wl, dx=dx, lens_model='aplanatic')
print("  |E| preserved everywhere:", np.allclose(np.abs(Ea), np.abs(E0)))

print("\n=== apply_cylindrical_lens: line focus position ===")
Ec = apply_cylindrical_lens(E0, f=f, wavelength=wl, dx=dx, axis='x')
zs = np.linspace(0.95*f, 1.05*f, 21); I=peak_z(Ec, zs)
print(f"  x-axis line focus at z={zs[np.argmax(I)]*1e3:.4f} mm (f={f*1e3:.3f})")

print("\n=== apply_spherical_lens: thick singlet focus vs lensmaker / BFL ===")
n = 1.5168; R1, R2, d = 20e-3, -20e-3, 4e-3
inv_f = (n-1)*(1/R1 - 1/R2 + (n-1)*d/(n*R1*R2))
f_thick = 1/inv_f
f_thin  = 1/((n-1)*(1/R1 - 1/R2))
BFL = f_thick*(1 - (n-1)*d/(n*R1))
print(f"  thin-lens f = {f_thin*1e3:.5f} mm ; thick EFL = {f_thick*1e3:.5f} mm ; BFL(from back vertex) = {BFL*1e3:.5f} mm")
Es = apply_spherical_lens(E0, R1=R1, R2=R2, d=d, n_lens=n, wavelength=wl, dx=dx)
zs = np.linspace(0.97*f_thin, 1.02*f_thin, 31); I=peak_z(Es, zs); z0=zs[np.argmax(I)]
zs2 = np.linspace(z0-2e-5, z0+2e-5, 41); I2=peak_z(Es, zs2); z1=zs2[np.argmax(I2)]
print(f"  measured focus (from the screen plane) = {z1*1e3:.5f} mm")
print(f"    vs thin f : {(z1-f_thin)*1e6:+.2f} um ;  vs thick EFL: {(z1-f_thick)*1e6:+.2f} um ;  vs BFL: {(z1-BFL)*1e6:+.2f} um")
# d-independence claim
Es2 = apply_spherical_lens(E0, R1=R1, R2=R2, d=1e-9, n_lens=n, wavelength=wl, dx=dx)
print("  d=4mm vs d=1nm bit-identical:", np.array_equal(Es, Es2))

print("\n=== apply_grin_lens vs exact GRIN ABCD ===")
n0, g = 1.6, 300.0        # g in 1/m
for gd in (0.05, 0.3, np.pi/4, np.pi/2, np.pi*0.9):
    d_rod = gd/g
    f_code  = 1.0/(n0*g**2*d_rod)          # what the code's quadratic phase implements
    f_exact = 1.0/(n0*g*np.sin(g*d_rod))   # ABCD EFL of the GRIN rod
    print(f"  g*d={gd:5.3f}  f_code={f_code*1e3:9.4f} mm  f_exact={f_exact*1e3:9.4f} mm  ratio={f_code/f_exact:6.4f}  err={100*(f_code/f_exact-1):+7.2f}%")
# empirical: measure the focus the phase screen actually produces at quarter pitch
d_rod = (np.pi/2)/g
Eg = apply_grin_lens(E0, n0=n0, g=g, d=d_rod, wavelength=wl, dx=dx)
f_code = 1.0/(n0*g**2*d_rod); f_exact = 1.0/(n0*g*np.sin(g*d_rod))
zs = np.linspace(0.8*f_code, 1.3*f_code, 41); I = peak_z(Eg, zs)
print(f"  quarter-pitch: measured focus {zs[np.argmax(I)]*1e3:.4f} mm  vs f_code {f_code*1e3:.4f}  vs f_exact {f_exact*1e3:.4f} mm")

print("\n=== apply_axicon: Bessel cone angle ===")
alpha = np.deg2rad(1.0); na = 1.5
Ex = apply_axicon(E0, alpha, na, wl, dx)
beta = (na-1)*alpha
# Check the ring radius of the far-field: should be at k_r = k*beta
F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Ex)))
fx = np.fft.fftshift(np.fft.fftfreq(N, dx)); FX,FY = np.meshgrid(fx,fx)
FR = np.hypot(FX,FY); P = np.abs(F)**2
r_peak = FR.ravel()[np.argmax(P.ravel())]
print(f"  measured |f_r| at far-field peak = {r_peak:.1f} 1/m -> theta = {np.arcsin(r_peak*wl):.6f} rad ; expected (n-1)alpha = {beta:.6f} rad")
print(f"  Bessel zone z_max = w0/((n-1)alpha) = {w0/beta*1e3:.2f} mm")
