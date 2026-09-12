"""Probe 6: carrier handling -- float R, TiltedCarrier, 'auto', ndarray.
Oracle: independent exact-sphere trace launched along the carrier normals,
with the entrance eikonal W(x_in) added (H6)."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements._lens_traced import TiltedCarrier
WL = common.WL; k0 = 2*np.pi/WL
n_of = lambda g: la.get_glass_index(g, WL)

S = 200e-3                      # diverging point source 200 mm in front
AP = 6e-3; N = 768; dx = 1.35*AP/N
rx = common.plano_convex(R=100e-3, t=4e-3, ap=AP)
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x); R2 = X**2+Y**2
W_exact = np.sqrt(R2 + S*S) - S
m = R2 <= (0.40*AP)**2
w = 1.0e-3
E_in = (np.exp(-R2/w**2)*np.exp(1j*k0*W_exact)).astype(np.complex128)

def oracle_carrier_opl(Xq, Yq):
    """Newton-invert the ORACLE map (rays launched along grad W) to the exit
    grid; return W(x_in) + geometric OPL."""
    xe = Xq.copy(); ye = Yq.copy()
    def fwd(xa, ya):
        rho = np.sqrt(xa*xa+ya*ya+S*S)
        L = xa/rho; M = ya/rho
        xo, yo, opl, _ = common.oracle_trace(rx, xa, ya, L=L, M=M, n_of=n_of)
        return xo, yo, opl + (np.sqrt(xa*xa+ya*ya+S*S) - S)
    for _ in range(40):
        xo, yo, _ = fwd(xe, ye)
        rxr = xo-Xq; ryr = yo-Yq
        if np.nanmax(np.abs(rxr)) < 1e-13 and np.nanmax(np.abs(ryr)) < 1e-13: break
        h = 2e-7
        x1,y1,_ = fwd(xe+h, ye); x2,y2,_ = fwd(xe, ye+h)
        jxx=(x1-xo)/h; jyx=(y1-yo)/h; jxy=(x2-xo)/h; jyy=(y2-yo)/h
        det = jxx*jyy-jxy*jyx
        inv = np.where(np.abs(det)>1e-25, 1.0/det, 0.0)
        xe = xe - (jyy*rxr-jxy*ryr)*inv; ye = ye - (-jyx*rxr+jxx*ryr)*inv
    _,_,opl = fwd(xe, ye)
    return opl

opl_or = oracle_carrier_opl(X, Y)
kw = dict(prescription=rx, wavelength=WL, dx=dx, ray_subsample=8, n_workers=1,
          on_undersample='silent', min_coarse_samples_per_aperture=0,
          on_pool_memory='silent')
cases = [('carrier=None', None),
         ('carrier=+200mm (float)', S),
         ("carrier='auto'", 'auto'),
         ('carrier=ndarray(W_exact)', W_exact),
         ('TiltedCarrier(R=S,L=M=0)', TiltedCarrier(S, 0.0, 0.0, 0.0, 0.0))]
for tag, c in cases:
    with warnings.catch_warnings(record=True) as wl_:
        warnings.simplefilter('always')
        try:
            E = apply_real_lens_traced(E_in, carrier=c, on_noncollimated='off', **kw)
        except Exception as e:
            print(f"{tag:28s}: {type(e).__name__}: {e}"); continue
    r = np.angle(E*np.exp(-1j*k0*opl_or))[m]
    r = (r+np.pi) % (2*np.pi) - np.pi
    A = np.stack([np.ones(r.size), X[m], Y[m]], 1)
    cc,*_ = np.linalg.lstsq(A, r, rcond=None); rr = r-A@cc
    print(f"{tag:28s}: raw|resid|max={np.abs(r).max():.4e} rad  "
          f"rms(piston+tilt removed)={rr.std():.4e} rad "
          f"({rr.std()/k0*1e9:9.4f} nm)  piston={cc[0]:+.4e} rad")
