"""Probe 1b/6/7: single SPHERICAL surface -- exact ray oracle vs
  * (T1) alone           = dz * sag
  * route 3 (T1+T2+T3)   = _tangent_facet_screen
  * remap  (R1)-(R5)     = _tangent_facet_remap_screen
Reports waves-rms residual at the exit vertex plane, piston+tilt removed.
"""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import (
    _tangent_facet_screen, _tangent_facet_remap_screen, _facet_axial_momenta)

lam = 0.55e-6
k0 = 2*np.pi/lam

def sag_sphere(r2, R):
    # conic k=0 sphere, sign convention: sag = r^2/(R(1+sqrt(1-r^2/R^2)))
    return r2/(R*(1.0+np.sqrt(np.maximum(1.0 - r2/R**2, 0.0))))

def exact_surface(x, y, R):
    r2 = x*x+y*y
    s = sag_sphere(r2, R)
    # grad analytic: d sag/dr = r/(R sqrt(1-r^2/R^2)) ; grad = (x,y)/ (R sqrt(...))
    den = R*np.sqrt(np.maximum(1.0-r2/R**2,1e-300))
    return s, x/den, y/den

def oracle(xg, yg, R, n1, n2, px, py):
    """Exact: hit point (Newton), exact Snell, exit eikonal referenced back to
    the vertex plane -> returns (OPD_target_at_pixel, W)."""
    pz1 = np.sqrt(n1**2 - px**2 - py**2)
    qx, qy = px/pz1, py/pz1
    s = np.zeros_like(xg)
    for _ in range(80):
        hx, hy = xg + s*qx, yg + s*qy
        f, gx, gy = exact_surface(hx, hy, R)
        # solve s = f(x+s q): Newton on F(s)=s-f
        Fp = 1.0 - (gx*qx + gy*qy)
        s = s - (s - f)/Fp
    hx, hy = xg + s*qx, yg + s*qy
    f, gx, gy = exact_surface(hx, hy, R)
    nu = np.stack([-gx, -gy, np.ones_like(gx)]); nu /= np.linalg.norm(nu, axis=0)
    p_in = np.stack([np.full_like(gx,px), np.full_like(gx,py), np.full_like(gx,pz1)])
    a = np.sum(p_in*nu, axis=0)
    Gam = -a + np.sqrt(n2**2 - n1**2 + a*a)
    p_out = p_in + Gam*nu
    pz2 = p_out[2]
    Wx = s*(qx - p_out[0]/pz2)
    Wy = s*(qy - p_out[1]/pz2)
    # S_out(x+W) - S_in(x) = s (n1^2/pz1 - n2^2/pz2)
    dS = s*(n1**2/pz1 - n2**2/pz2)
    # target OPD for a screen that imprints at the PIXEL x:
    #   S_model(x) = S_in(x) - OPD  ==  S_exact(x) = S_out(x+W) - p_out . W  (+O(W^2))
    #  -> OPD_screen = -dS + p_out_t . W     ... EXACT only to 1st order in W
    return dS, Wx, Wy, p_out[0], p_out[1], pz2, s

def rms_waves(f, m):
    # piston+tilt free rms over mask m
    X, Y = np.meshgrid(np.arange(f.shape[1]), np.arange(f.shape[0]))
    A = np.stack([np.ones(m.sum()), X[m], Y[m]], axis=1)
    c, *_ = np.linalg.lstsq(A, f[m], rcond=None)
    r = f[m] - A@c
    return float(np.sqrt(np.mean(r**2))/lam)

def run(R, n1, n2, theta_mrad, half_ap, N=513):
    dx = 2*half_ap/(N-1)
    ax = (np.arange(N)-N//2)*dx
    X, Y = np.meshgrid(ax, ax)
    px = n1*np.sin(theta_mrad*1e-3); py = 0.0
    pz1 = np.sqrt(n1**2-px**2)
    r2 = X*X+Y*Y
    sag = sag_sphere(r2, R)
    gy_, gx_ = np.gradient(sag, dx, dx)
    PX = np.full_like(sag, px); PY = np.full_like(sag, py)
    # ---- route 3
    opd3, ok3 = _tangent_facet_screen(sag, gx_, gy_, PX, PY, n1, n2, dx, dx, np)
    # ---- remap
    hxy, hxx = np.gradient(gx_, dx, dx)
    hyy, hyx = np.gradient(gy_, dx, dx)
    opdR, wx, wy, pox, poy, okR = _tangent_facet_remap_screen(
        sag, gx_, gy_, hxx, hxy, hyx, hyy, PX, PY, n1, n2, np)
    # ---- (T1)
    dz, _ = _facet_axial_momenta(PX, PY, gx_, gy_, n1, n2, np)
    opd1 = dz*sag
    # ---- oracle
    dS, Wx, Wy, pox_e, poy_e, pz2_e, s_e = oracle(X, Y, R, n1, n2, px, py)
    tgt_screen = -dS + (pox_e*Wx + poy_e*Wy)       # pixel-referenced target
    tgt_remap  = -dS                                # (R1) target
    m = (r2 <= (half_ap*0.92)**2)
    return (rms_waves(opd1-tgt_screen, m), rms_waves(opd3-tgt_screen, m),
            rms_waves(opdR-tgt_remap, m), np.max(np.abs(Wx[m])),
            rms_waves(wx-Wx, m)*lam, np.max(np.abs(s_e[m])))

print(" R[mm]  n1   n2   theta   half_ap    (T1)      route3     remap      max|W|um   dWx[m]rms  max_sag um")
for R, n1, n2 in [(19.6e-3, 1.0, 1.62), (-19.6e-3, 1.62, 1.0), (25e-3,1.0,1.8), (12e-3,1.0,1.5)]:
    for th in (0.0, 30.0, 80.0):
        for ha in (1.0e-3, 2.0e-3):
            a,b,c,w,dw,ms = run(R,n1,n2,th,ha)
            print(f"{R*1e3:7.1f} {n1:4.2f} {n2:4.2f} {th:6.1f} {ha*1e3:6.2f}  {a:10.3e} {b:10.3e} {c:10.3e}  {w*1e6:8.2f} {dw:10.3e} {ms*1e6:8.2f}")
