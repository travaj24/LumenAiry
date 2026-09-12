"""Probe 6/7 (corrected oracle): exit eikonal at the PIXEL, obtained by
inverting the exact ray landing map x -> x + W(x).  Compares
  (T1)  = dz*sag
  route3 = _tangent_facet_screen  (T1+T2+T3)
  remap  = _tangent_facet_remap_screen (R1)   [scored at the landing point]
"""
import numpy as np, sys
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements._lens_real import (
    _tangent_facet_screen, _tangent_facet_remap_screen, _facet_axial_momenta)
from scipy.interpolate import RectBivariateSpline

lam = 0.55e-6

def sag_sphere(r2, R):
    return r2/(R*(1.0+np.sqrt(np.maximum(1.0 - r2/R**2, 0.0))))

def exact_surface(x, y, R):
    r2 = x*x+y*y
    s = sag_sphere(r2, R)
    den = R*np.sqrt(np.maximum(1.0-r2/R**2,1e-300))
    return s, x/den, y/den

def ray(xg, yg, R, n1, n2, px, py):
    pz1 = np.sqrt(n1**2 - px**2 - py**2)
    qx, qy = px/pz1, py/pz1
    s = np.zeros_like(xg)
    for _ in range(14):
        hx, hy = xg + s*qx, yg + s*qy
        f, gx, gy = exact_surface(hx, hy, R)
        s = s - (s - f)/(1.0 - (gx*qx+gy*qy))
    hx, hy = xg + s*qx, yg + s*qy
    f, gx, gy = exact_surface(hx, hy, R)
    nu = np.stack([-gx, -gy, np.ones_like(gx)]); nu /= np.linalg.norm(nu, axis=0)
    p_in = np.stack([np.full_like(gx,px), np.full_like(gx,py), np.full_like(gx,pz1)])
    a = np.sum(p_in*nu, axis=0)
    Gam = -a + np.sqrt(n2**2 - n1**2 + a*a)
    p_out = p_in + Gam*nu
    pz2 = p_out[2]
    Wx = s*(qx - p_out[0]/pz2); Wy = s*(qy - p_out[1]/pz2)
    dS = s*(n1**2/pz1 - n2**2/pz2)
    return Wx, Wy, dS

def rms_waves(f, m):
    ys, xs = np.mgrid[0:f.shape[0], 0:f.shape[1]]
    A = np.stack([np.ones(int(m.sum())), xs[m], ys[m]], axis=1)
    c, *_ = np.linalg.lstsq(A, f[m], rcond=None)
    return float(np.sqrt(np.mean((f[m]-A@c)**2))/lam)

def run(R, n1, n2, theta_mrad, half_ap, N=257):
    dx = 2*half_ap/(N-1)
    ax = (np.arange(N)-N//2)*dx
    X, Y = np.meshgrid(ax, ax)
    px = n1*np.sin(theta_mrad*1e-3); py = 0.0
    pz1 = np.sqrt(n1**2-px**2)
    r2 = X*X+Y*Y
    sag = sag_sphere(r2, R)
    gy_, gx_ = np.gradient(sag, dx, dx)
    PX = np.full_like(sag, px); PY = np.full_like(sag, py)
    opd3, _ = _tangent_facet_screen(sag, gx_, gy_, PX, PY, n1, n2, dx, dx, np)
    hxy, hxx = np.gradient(gx_, dx, dx); hyy, hyx = np.gradient(gy_, dx, dx)
    opdR, wx, wy, pox, poy, _ = _tangent_facet_remap_screen(
        sag, gx_, gy_, hxx, hxy, hyx, hyy, PX, PY, n1, n2, np)
    dz, _ = _facet_axial_momenta(PX, PY, gx_, gy_, n1, n2, np)
    opd1 = dz*sag
    # ---- EXACT: invert x + W(x) = u by fixed point on the analytic ray map
    Ux = X.copy(); Uy = Y.copy()
    for _ in range(60):
        Wx, Wy, dS = ray(Ux, Uy, R, n1, n2, px, py)
        nUx = X - Wx; nUy = Y - Wy
        st = max(np.max(np.abs(nUx-Ux)), np.max(np.abs(nUy-Uy)))
        Ux, Uy = nUx, nUy
        if st < 1e-16: break
    Wx, Wy, dS = ray(Ux, Uy, R, n1, n2, px, py)
    # S_exact(pixel) = S_in(source x) + dS = px*Ux+py*Uy + dS
    S_ex = px*Ux + py*Uy + dS
    # model at pixel: S_in(pixel) - OPD(pixel)
    S_in_pix = px*X + py*Y
    m = (r2 <= (half_ap*0.9)**2)
    e1 = rms_waves((S_in_pix - opd1) - S_ex, m)
    e3 = rms_waves((S_in_pix - opd3) - S_ex, m)
    # remap: scored at the LANDING point.  S_model(x+W_model) = S_in(x)-opdR
    # exact at that same landing point = S_in(x) + dS_at_x
    Wxx, Wyy, dSx = ray(X, Y, R, n1, n2, px, py)
    eR = rms_waves(-opdR - dSx, m)
    walk_err = float(np.max(np.abs(wx-Wxx)[m]))
    return e1, e3, eR, float(np.max(np.abs(Wxx[m])))*1e6, walk_err*1e9

print("   R[mm] n1   n2  th[mrad] ap[mm] | (T1)wv     route3wv    remap(R1)wv | max|W|um  walkErr[nm]")
for R, n1, n2 in [(19.6e-3,1.0,1.62), (-19.6e-3,1.62,1.0), (12e-3,1.0,1.5)]:
    for th in (0.0, 30.0, 80.0):
        for ha in (1.0e-3, 2.0e-3):
            a,b,c,w,we = run(R,n1,n2,th,ha)
            print(f"{R*1e3:8.1f} {n1:4.2f} {n2:4.2f} {th:7.1f} {ha*1e3:6.2f} | {a:10.3e} {b:10.3e} {c:10.3e} | {w:8.2f} {we:10.3e}")
