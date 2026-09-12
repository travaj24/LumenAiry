"""RW test 3: sign / handedness of the Debye-Wolf transverse kernel.

(1) off-centre sub-aperture -> focal-plane linear phase ramp sign.
(2) coma pupil vs an INDEPENDENT (theta, phi) polar-quadrature
    Debye-Wolf oracle, evaluated at +P and -P.
"""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.vector_diffraction import richards_wolf_focus

lam = 633e-9
k = 2*np.pi/lam

# ---------------------------------------------------------------- (1)
print("=== (1) off-centre sub-aperture: sign of the focal phase ramp ===")
NA, f, Np, dxp, Nf = 0.3, 2e-3, 1024, 1.5e-6, 1024
x = (np.arange(Np)-Np/2)*dxp
X, Y = np.meshgrid(x, x)
xc = 0.5*f*NA                       # sub-aperture centre at +x
pupil = (np.hypot(X-xc, Y) <= 0.15*f*NA).astype(np.complex128)
Ex, Ey, Ez, xf, yf = richards_wolf_focus(pupil, lam, NA, f, dxp,
                                         N_focal=Nf, polarization='x')
c = Nf//2
cut = Ex[c, c-6:c+7]
ph = np.unwrap(np.angle(cut))
dxf = lam*f/(Nf*dxp)
slope = np.polyfit(np.arange(-6, 7)*dxf, ph, 1)[0]
pred = k*xc/f
print(f"  sub-aperture centre x_c = +{xc*1e6:.1f} um,  f = {f*1e3:.1f} mm")
print(f"  d(arg Ex)/dx_f  MEASURED = {slope:+.6e} rad/m")
print(f"  Debye-Wolf exp(+i k.r) predicts = {pred:+.6e} rad/m")
print(f"  ratio measured/predicted = {slope/pred:+.6f}   "
      f"-> {'CORRECT sign' if slope/pred > 0 else 'SIGN FLIPPED (mirrored focal field)'}")

# ---------------------------------------------------------------- (2)
print()
print("=== (2) coma pupil vs independent polar-quadrature Debye-Wolf oracle ===")
NA, f, Np, dxp, Nf = 0.3, 2e-3, 1024, 1.5e-6, 1024
a = f*NA
x = (np.arange(Np)-Np/2)*dxp
X, Y = np.meshgrid(x, x)
R = np.hypot(X, Y)
PH = np.arctan2(Y, X)
rho = R/a
W = 0.35                                   # waves of Zernike coma
Zc = (3*rho**3 - 2*rho)*np.cos(PH)
pupil = np.where(R <= a, np.exp(2j*np.pi*W*Zc), 0.0).astype(np.complex128)

Ex, Ey, Ez, xf, yf = richards_wolf_focus(pupil, lam, NA, f, dxp,
                                         N_focal=Nf, polarization='x')
dxf = lam*f/(Nf*dxp)
c = Nf//2


def oracle(xp_, yp_, zp_, nth=900, nph=1440):
    """Direct Debye-Wolf integral in (theta, phi), Gauss-Legendre x trapezoid."""
    th_max = np.arcsin(NA)
    tn, tw = np.polynomial.legendre.leggauss(nth)
    th = 0.5*th_max*(tn+1.0)
    thw = 0.5*th_max*tw
    ph = np.arange(nph)*(2*np.pi/nph)
    phw = 2*np.pi/nph
    TH, PHI = np.meshgrid(th, ph, indexing='ij')
    W2 = thw[:, None]*phw
    st, ct = np.sin(TH), np.cos(TH)
    cp, sp = np.cos(PHI), np.sin(PHI)
    # pupil amplitude at this (theta, phi): rho = f sin(theta)/a
    rr = f*st/a
    A = np.exp(2j*np.pi*W*(3*rr**3 - 2*rr)*cp)
    apod = np.sqrt(ct)                       # aplanatic (Abbe sine)
    Vx = cp**2*ct + sp**2
    Vy = cp*sp*(ct-1.0)
    Vz = -cp*st
    kern = np.exp(1j*k*(xp_*st*cp + yp_*st*sp + zp_*ct))
    base = A*apod*kern*st*W2
    pre = (-1j*k*f/(2*np.pi))*np.exp(1j*k*f)
    return (pre*np.sum(base*Vx), pre*np.sum(base*Vy), pre*np.sum(base*Vz))


print(f"  dx_focal = {dxf*1e9:.1f} nm")
print(f"  {'offset(px)':>10} {'|E_mod|/|E_orc|':>16} {'phase diff(rad)':>16}"
      f" {'rel err at +P':>14} {'rel err at -P':>14}")
for (dy_, dx_) in [(0, 2), (0, 4), (2, 3), (-3, 5), (0, 6)]:
    xP, yP = dx_*dxf, dy_*dxf
    ox, oy, oz = oracle(xP, yP, 0.0)
    mx, my, mz = Ex[c+dy_, c+dx_], Ey[c+dy_, c+dx_], Ez[c+dy_, c+dx_]
    nx, ny, nz = Ex[c-dy_, c-dx_], Ey[c-dy_, c-dx_], Ez[c-dy_, c-dx_]
    ov = np.array([ox, oy, oz])
    mv = np.array([mx, my, mz])
    # at -P the correct comparison flips the sign of Ez (odd in cos(phi))
    nv = np.array([nx, ny, -nz])
    e_p = np.linalg.norm(mv-ov)/np.linalg.norm(ov)
    e_m = np.linalg.norm(nv-ov)/np.linalg.norm(ov)
    print(f"  ({dy_:+d},{dx_:+d})".rjust(10)
          + f" {abs(mx)/abs(ox):>16.6f} {np.angle(mx/ox):>16.6f}"
          + f" {e_p:>14.4e} {e_m:>14.4e}")
print("  (rel err is the full (Ex,Ey,Ez) vector L2 error vs the oracle;")
print("   'at -P' compares the module value at the antipodal focal point.)")
