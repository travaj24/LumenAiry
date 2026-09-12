import numpy as np, sys
sys.path.insert(0, r'D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy')
from lumenairy.propagators.asm import angular_spectrum_propagate as asm
from lumenairy.propagators import fft_infra as fi

lam = 1.0e-6; k = 2*np.pi/lam

# ---------- 1(h): Gaussian beam vs analytic ----------
def gauss_analytic(N, dx, w0, z, lam):
    x = (np.arange(N)-N//2)*dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    zR = np.pi*w0**2/lam; k = 2*np.pi/lam
    w = w0*np.sqrt(1+(z/zR)**2)
    Rinv = z/(z**2+zR**2)
    gouy = np.arctan2(z, zR)
    r2 = X**2+Y**2
    return (w0/w)*np.exp(-r2/w**2)*np.exp(1j*(k*z + k*r2*Rinv/2 - gouy))

N=1024; dx=0.5e-6; w0=8e-6
zR = np.pi*w0**2/lam
print(f"zR = {zR*1e6:.3f} um ; N*dx = {N*dx*1e6} um")
for f in (0.5, 1.0, 3.0):
    z = f*zR
    E0 = gauss_analytic(N, dx, w0, 0.0, lam)
    Eex = gauss_analytic(N, dx, w0, z, lam)
    Enum = asm(E0.astype(np.complex128), z, lam, dx)
    err = np.linalg.norm(Enum-Eex)/np.linalg.norm(Eex)
    print(f"  ASM vs analytic Gaussian z={f}zR : relL2 = {err:.3e}")

# ---------- 1(h) plane wave exact phase ----------
for th_deg in (0.0, 5.0, 20.0):
    th = np.radians(th_deg)
    x = (np.arange(N)-N//2)*dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    fx0 = np.sin(th)/lam
    E0 = np.exp(2j*np.pi*fx0*X).astype(np.complex128)
    z = 20e-6
    Eo = asm(E0, z, lam, dx)
    # expected phase advance k z cos(theta)  (as a ratio at centre region)
    ratio = Eo[N//2, N//2]/E0[N//2, N//2]
    expected = np.exp(1j*k*z*np.cos(th))
    print(f"  plane wave th={th_deg}deg: |ratio|={abs(ratio):.6f} "
          f"phase_err={np.angle(ratio/expected):.3e} rad")

# ---------- 1(b): negative z, evanescent must not blow up ----------
# sub-wavelength grid so evanescent bins exist
N2=256; dx2=0.2e-6
rng = np.random.default_rng(0)
E = (rng.standard_normal((N2,N2))+1j*rng.standard_normal((N2,N2))).astype(np.complex128)
for z in (+5e-6, -5e-6):
    Eo = asm(E, z, lam, dx2)
    print(f"  z={z*1e6:+.1f}um sub-lambda grid: max|E_out|={np.abs(Eo).max():.4e} "
          f"finite={np.isfinite(Eo).all()}  P_out/P_in={np.sum(abs(Eo)**2)/np.sum(abs(E)**2):.6f}")

# round-trip forward then backward
Ef = asm(E, 5e-6, lam, dx2, bandlimit=False)
Eb = asm(Ef, -5e-6, lam, dx2, bandlimit=False)
print(f"  round-trip +z then -z (bandlimit off): relL2 = {np.linalg.norm(Eb-E)/np.linalg.norm(E):.3e}")

# ---------- 1(h) energy / Parseval, non-evanescent ----------
N3=512; dx3=1.0e-6
x = (np.arange(N3)-N3//2)*dx3
X,Y = np.meshgrid(x,x,indexing='xy')
E = np.exp(-(X**2+Y**2)/(2*(20e-6)**2)).astype(np.complex128)
for z in (1e-4, 1e-3, 1e-2):
    for bl in (True, False):
        Eo = asm(E, z, lam, dx3, bandlimit=bl)
        print(f"  energy z={z*1e3:.1f}mm bandlimit={bl}: P_out/P_in = "
              f"{np.sum(abs(Eo)**2)/np.sum(abs(E)**2):.8f}")
