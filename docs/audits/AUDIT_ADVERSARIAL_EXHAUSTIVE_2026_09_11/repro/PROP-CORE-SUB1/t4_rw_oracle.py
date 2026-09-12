"""RW test 4: coma + Gaussian-apodised pupil vs an INDEPENDENT
(theta, phi) polar-quadrature Debye-Wolf oracle.

Compares the module value at +P and at -P against the oracle at +P.
"""
import sys
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.propagators.vector_diffraction import richards_wolf_focus

lam = 633e-9
k = 2*np.pi/lam
NA, f, Np, dxp, Nf = 0.3, 2e-3, 1024, 1.5e-6, 1024
a = f*NA
W = 0.35            # waves of Zernike coma
g = 0.45            # Gaussian pupil apodisation (in units of rho)


def pupil_fn(rho, phi):
    return np.exp(-(rho/g)**2) * np.exp(2j*np.pi*W*(3*rho**3 - 2*rho)*np.cos(phi))


x = (np.arange(Np)-Np/2)*dxp
X, Y = np.meshgrid(x, x)
R = np.hypot(X, Y); PH = np.arctan2(Y, X)
pupil = np.where(R <= a, pupil_fn(R/a, PH), 0.0).astype(np.complex128)

Ex, Ey, Ez, xf, yf = richards_wolf_focus(pupil, lam, NA, f, dxp,
                                         N_focal=Nf, polarization='x')
dxf = lam*f/(Nf*dxp)
c = Nf//2


def oracle(xP, yP, zP, nth=600, nph=1024):
    th_max = np.arcsin(NA)
    tn, tw = np.polynomial.legendre.leggauss(nth)
    th = 0.5*th_max*(tn+1.0); thw = 0.5*th_max*tw
    ph = np.arange(nph)*(2*np.pi/nph); phw = 2*np.pi/nph
    TH, PHI = np.meshgrid(th, ph, indexing='ij')
    Wt = thw[:, None]*phw
    st, ct = np.sin(TH), np.cos(TH)
    cp, sp = np.cos(PHI), np.sin(PHI)
    A = pupil_fn(f*st/a, PHI)
    base = A*np.sqrt(ct)*np.exp(1j*k*(xP*st*cp + yP*st*sp + zP*ct))*st*Wt
    pre = (-1j*k*f/(2*np.pi))*np.exp(1j*k*f)
    return np.array([pre*np.sum(base*(cp**2*ct + sp**2)),
                     pre*np.sum(base*(cp*sp*(ct-1.0))),
                     pre*np.sum(base*(-cp*st))])


print(f"dx_focal = {dxf*1e9:.1f} nm ; Airy radius 0.61 lam/NA = "
      f"{0.61*lam/NA*1e9:.0f} nm")
print(f"{'focal pt (py,px)':>18} {'|E_orc|':>11} "
      f"{'relL2 module(+P)':>18} {'relL2 module(-P)':>18}")
for (dy_, dx_) in [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (-2, 3), (2, -1), (0, -2)]:
    xP, yP = dx_*dxf, dy_*dxf
    ov = oracle(xP, yP, 0.0)
    mv = np.array([Ex[c+dy_, c+dx_], Ey[c+dy_, c+dx_], Ez[c+dy_, c+dx_]])
    nv = np.array([Ex[c-dy_, c-dx_], Ey[c-dy_, c-dx_], Ez[c-dy_, c-dx_]])
    n0 = np.linalg.norm(ov)
    print(f"  ({dy_:+d},{dx_:+d})".rjust(18)
          + f" {n0:>11.4e} {np.linalg.norm(mv-ov)/n0:>18.4e}"
          + f" {np.linalg.norm(nv-ov)/n0:>18.4e}")

# through-focus: does the +z defocus also flip?
print()
print("defocus z = +2 um (checks that the axial kernel is NOT flipped):")
Ex2, Ey2, Ez2, _, _ = richards_wolf_focus(pupil, lam, NA, f, dxp,
                                          N_focal=Nf, polarization='x',
                                          z_planes=[2e-6])
for (dy_, dx_) in [(0, 2), (1, -3)]:
    xP, yP = dx_*dxf, dy_*dxf
    ov = oracle(xP, yP, 2e-6)
    mv = np.array([Ex2[c+dy_, c+dx_], Ey2[c+dy_, c+dx_], Ez2[c+dy_, c+dx_]])
    nv = np.array([Ex2[c-dy_, c-dx_], Ey2[c-dy_, c-dx_], Ez2[c-dy_, c-dx_]])
    n0 = np.linalg.norm(ov)
    print(f"  ({dy_:+d},{dx_:+d})  relL2(+P) = "
          f"{np.linalg.norm(mv-ov)/n0:.4e}   relL2(-P) = "
          f"{np.linalg.norm(nv-ov)/n0:.4e}")
