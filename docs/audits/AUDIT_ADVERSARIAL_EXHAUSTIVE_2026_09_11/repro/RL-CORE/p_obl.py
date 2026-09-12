"""Probe: is the screen_obliquity correction WIRED correctly (does carrier=
reduce the error against a tilted-input ray oracle)?"""
import sys, os, warnings
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from oracle import sag_of, dsag_dh
from lumenairy.elements._lens_real import apply_real_lens
from lumenairy.glass import get_glass_index
lam = 632.8e-9; k0 = 2*np.pi/lam

def trace_tilted(rx, h0, theta):
    """Same oracle, but rays launched at angle theta in the x-z plane from the
    plane z=0, entering at x = h0."""
    S = rx['surfaces']; T = rx['thicknesses']
    nf = lambda g: float(get_glass_index(g, lam))
    x = np.array(h0, float).copy(); z = np.zeros_like(x)
    Lx = np.full_like(x, np.sin(theta)); Lz = np.full_like(x, np.cos(theta))
    opl = np.zeros_like(x)
    vz = [0.0]
    for t in T: vz.append(vz[-1] + float(t))
    for i, s in enumerate(S):
        R = s['radius']; k = s.get('conic', 0.0); a = s.get('aspheric_coeffs')
        zv = vz[i]; n1 = nf(s['glass_before']); n2 = nf(s['glass_after'])
        t = (zv - z)/Lz
        for _ in range(80):
            xx = x + t*Lx; zz = z + t*Lz
            f = zz - (zv + sag_of(xx*xx, R, k, a))
            df = Lz - dsag_dh(xx, R, k, a)*Lx
            st = f/df; t = t - st
            if np.max(np.abs(st)) < 1e-16: break
        xh = x + t*Lx; zh = z + t*Lz
        opl = opl + n1*t
        nx = -dsag_dh(xh, R, k, a); nz = np.ones_like(nx)
        nn = np.sqrt(nx*nx+nz*nz); nx, nz = nx/nn, nz/nn
        ci = -(Lx*nx + Lz*nz); sg_ = np.where(ci < 0, -1.0, 1.0)
        nx, nz = nx*sg_, nz*sg_; ci = -(Lx*nx + Lz*nz)
        mu = n1/n2; s2 = mu*mu*(1-ci*ci); ct = np.sqrt(np.maximum(1-s2, 0))
        Lx = mu*Lx + (mu*ci-ct)*nx; Lz = mu*Lz + (mu*ci-ct)*nz
        x, z = xh, zh; n_last = n2
    te = (vz[-1]-z)/Lz
    return x + te*Lx, opl + n_last*te

def bx(R=40e-3, t=4e-3, ap=3e-3):
    return dict(surfaces=[dict(radius=R, glass_before='AIR', glass_after='N-BK7'),
                          dict(radius=-R, glass_before='N-BK7', glass_after='AIR')],
                thicknesses=[t], aperture_diameter=ap)

from lumenairy import TiltedCarrier
rx = bx(); ap = rx['aperture_diameter']
for theta in (0.02, 0.05, 0.10):
    hf = np.linspace(-0.9*ap, 0.9*ap, 12001)
    xe, opl = trace_tilted(rx, hf, theta)
    # exit NA for sampling
    NA = 0.5*ap/ (abs(xe[-1]-xe[0])/ (hf[-1]-hf[0]) and 1) # crude
    dxg = 0.25*lam/ (np.sin(theta) + 0.5*ap/ (R_eff := 40e-3))
    N = int(2**np.ceil(np.log2(1.6*ap/dxg)))
    xg = (np.arange(N) - N/2)*dxg
    Xg, Yg = np.meshgrid(xg, xg)
    E = np.exp(1j*k0*np.sin(theta)*Xg).astype(np.complex128)
    o = np.argsort(xe)
    m = np.abs(xg) <= 0.45*ap
    res = {}
    for tag, kw in (('blind', {}),
                    ('carrier + correction',
                     dict(carrier=TiltedCarrier(R=1e9, L=np.sin(theta), M=0.0),
                          on_screen_obliquity='silent')),
                    ('carrier, correction OFF',
                     dict(carrier=TiltedCarrier(R=1e9, L=np.sin(theta), M=0.0),
                          screen_obliquity=False, on_screen_obliquity='silent'))):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Eo = apply_real_lens(E.copy(), prescription=rx, wavelength=lam,
                                 dx=dxg, **kw)
        ph = np.unwrap(np.angle(Eo[N//2]))
        W = ph[m]/k0
        d = W - np.interp(xg[m], xe[o], opl[o])
        # remove piston AND tilt (the tilt is an arbitrary reference choice)
        A = np.vstack([np.ones_like(xg[m]), xg[m]]).T
        d = d - A @ np.linalg.lstsq(A, d, rcond=None)[0]
        res[tag] = float(np.sqrt(np.mean(d**2)))*1e9
    print(f"theta={theta:.3f} rad  N={N} dx={dxg*1e6:.3f}um : "
          + '  '.join(f'{k}={v:.3f} nm' for k, v in res.items()))
