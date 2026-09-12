"""Probe 13: does the default newton_poly_order=6 Chebyshev forward fit carry
high-order aspheric (A8/A10) content?  Oracle = independent exact trace."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens_traced
from lumenairy.elements.lenses import surface_sag_general
WL = common.WL; k0 = 2*np.pi/WL
n_of = lambda g: la.get_glass_index(g, WL)

# --- extend the oracle to aspheric surfaces (Newton intersection) ---
def sag(h2, R, k, A):  return surface_sag_general(h2, R, k, A)
def dsag(h2, R, k, A, h=1e-9):
    return (sag(h2+2*np.sqrt(np.maximum(h2,0))*h+h*h, R, k, A) - sag(h2, R, k, A))/h

def oracle_asph(rx, xs, ys):
    surfs = rx['surfaces']
    xs = np.asarray(xs, float); ys = np.asarray(ys, float)
    p = np.stack([xs, ys, np.zeros_like(xs)], -1)
    d = np.zeros(xs.shape + (3,)); d[..., 2] = 1.0
    opl = np.zeros(xs.shape); zv = 0.0
    for s in surfs:
        R = float(s['radius']); k = float(s.get('conic', 0.0))
        A = s.get('aspheric_coeffs')
        n1 = n_of(s.get('glass_before','air')); n2 = n_of(s.get('glass_after','air'))
        t = (zv - p[...,2])/d[...,2]
        for _ in range(80):
            q = p + t[...,None]*d
            f = q[...,2] - (zv + sag(q[...,0]**2+q[...,1]**2, R, k, A))
            # d f / d t
            h = 1e-9
            q2 = p + (t+h)[...,None]*d
            f2 = q2[...,2] - (zv + sag(q2[...,0]**2+q2[...,1]**2, R, k, A))
            t = t - f*h/(f2-f)
            if np.nanmax(np.abs(f)) < 1e-15: break
        p = p + t[...,None]*d; opl = opl + n1*t
        # normal by central difference of sag
        hh = 1e-8
        z = lambda X, Y: sag(X*X+Y*Y, R, k, A)
        gx = (z(p[...,0]+hh, p[...,1]) - z(p[...,0]-hh, p[...,1]))/(2*hh)
        gy = (z(p[...,0], p[...,1]+hh) - z(p[...,0], p[...,1]-hh))/(2*hh)
        nv = np.stack([-gx, -gy, np.ones_like(gx)], -1)
        nv = nv/np.linalg.norm(nv, axis=-1, keepdims=True)
        mu = n1/n2
        ci = np.einsum('...i,...i->...', d, nv)
        ct = np.sqrt(np.maximum(0.0, 1-mu*mu*(1-ci*ci)))
        d = mu*d + (ct-mu*ci)[...,None]*nv
        zv = zv + float(s['thickness'])
    z_exit = sum(float(s['thickness']) for s in surfs)
    nl = n_of(surfs[-1].get('glass_after','air'))
    tt = (z_exit - p[...,2])/d[...,2]
    p = p + tt[...,None]*d; opl = opl + nl*tt
    return p[...,0], p[...,1], opl

def inv_opl(rx, X, Y):
    xe = X.copy(); ye = Y.copy()
    for _ in range(40):
        xo, yo, opl = oracle_asph(rx, xe, ye)
        rxr = xo-X; ryr = yo-Y
        if np.nanmax(np.abs(rxr))<1e-13 and np.nanmax(np.abs(ryr))<1e-13: break
        h = 2e-7
        x1,y1,_ = oracle_asph(rx, xe+h, ye); x2,y2,_ = oracle_asph(rx, xe, ye+h)
        jxx=(x1-xo)/h; jyx=(y1-yo)/h; jxy=(x2-xo)/h; jyy=(y2-yo)/h
        det = jxx*jyy-jxy*jyx
        inv = np.where(np.abs(det)>1e-20, 1.0/det, 0.0)
        xe = xe - (jyy*rxr-jxy*ryr)*inv; ye = ye - (-jyx*rxr+jxx*ryr)*inv
    _,_,opl = oracle_asph(rx, xe, ye)
    return opl

AP = 8e-3
A = {4: -1.0e2, 6: 5.0e4, 8: -2.0e7, 10: 8.0e9}   # strong A8/A10 content
rxa = {'wavelength': WL, 'aperture_diameter': AP,
       'surfaces': [
         {'radius': 60e-3, 'thickness': 4e-3, 'glass_before':'air',
          'glass_after':'_AUD_GLASS', 'conic': -0.6, 'aspheric_coeffs': A,
          'semi_diameter': AP/2},
         {'radius': -60e-3, 'thickness': 0.0, 'glass_before':'_AUD_GLASS',
          'glass_after':'air', 'semi_diameter': AP/2}],
       'thicknesses': [4e-3], 'stop_index': 0}
h = np.linspace(0, AP/2, 6)
print("aspheric departure over the pupil (um):",
      np.round(1e6*(sag(h**2, 60e-3, -0.6, A) - sag(h**2, 60e-3, 0.0, None)), 4))
N = 768; dx = 1.3*AP/N
x = (np.arange(N)-N/2)*dx; X, Y = np.meshgrid(x, x)
m = (X**2+Y**2) <= (0.45*AP)**2
opl_or = inv_opl(rxa, X, Y)
kw = dict(prescription=rxa, wavelength=WL, dx=dx, n_workers=1,
          on_undersample='silent', min_coarse_samples_per_aperture=0,
          on_pool_memory='silent')
for order in (6, 8, 10, 12):
    for sub, imap in ((1, True), (8, True), (8, False)):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = apply_real_lens_traced(np.ones((N,N), np.complex128),
                                       ray_subsample=sub, newton_poly_order=order,
                                       inverse_map=imap, **kw)
        r = np.angle(E*np.exp(-1j*k0*opl_or))[m]/k0
        A_ = np.stack([np.ones(r.size), X[m], Y[m]], 1)
        c,*_ = np.linalg.lstsq(A_, r, rcond=None); rr = r-A_@c
        print(f"  newton_poly_order={order:2d} sub={sub} imap={imap}: "
              f"raw|resid|max={np.abs(r).max()*1e9:10.4f} nm  "
              f"rms(piston+tilt removed)={rr.std()*1e9:10.4f} nm")
