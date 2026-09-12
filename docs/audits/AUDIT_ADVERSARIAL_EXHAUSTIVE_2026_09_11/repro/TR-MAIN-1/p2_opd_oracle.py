"""Probe 2: traced exit OPD vs an INDEPENDENT sequential exact-sphere ray
trace, plus the absolute-piston question."""
import warnings, numpy as np
import common
common.register_glass()
import lumenairy as la
from lumenairy.elements import apply_real_lens, apply_real_lens_traced

WL = common.WL
k0 = 2*np.pi/WL
n_of = lambda g: la.get_glass_index(g, WL)

def gauss(N, dx, w):
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X**2+Y**2)/w**2).astype(np.complex128), X, Y

def oracle_opd_on_exit_grid(rx, X, Y):
    """Solve the inverse map by Newton on the ORACLE trace: for each exit pixel
    find the entrance (xe,ye) landing there, return OPL."""
    xe = X.copy(); ye = Y.copy()
    for _ in range(60):
        xo, yo, opl, _ = common.oracle_trace(rx, xe, ye, n_of=n_of)
        rx_ = xo - X; ry_ = yo - Y
        if np.nanmax(np.abs(rx_)) < 1e-13 and np.nanmax(np.abs(ry_)) < 1e-13:
            break
        h = 1e-7
        x1,_,_,_ = common.oracle_trace(rx, xe+h, ye, n_of=n_of)
        _,y1,_,_ = common.oracle_trace(rx, xe, ye+h, n_of=n_of)
        x2,_,_,_ = common.oracle_trace(rx, xe, ye+h, n_of=n_of)
        _,y2,_,_ = common.oracle_trace(rx, xe+h, ye, n_of=n_of)
        jxx = (x1-xo)/h; jxy = (x2-xo)/h
        jyx = (y2-yo)/h; jyy = (y1-yo)/h
        det = jxx*jyy - jxy*jyx
        with np.errstate(divide='ignore', invalid='ignore'):
            inv = np.where(np.abs(det)>1e-20, 1.0/det, 0.0)
        xe = xe - (jyy*rx_ - jxy*ry_)*inv
        ye = ye - (-jyx*rx_ + jxx*ry_)*inv
    xo, yo, opl, _ = common.oracle_trace(rx, xe, ye, n_of=n_of)
    return opl, xe, ye

def report(tag, ph_traced, opl_oracle, mask, X, Y):
    """Remove piston + tilt, report RMS in nm."""
    d = ph_traced/k0 - opl_oracle            # metres of OPD difference
    d = d[mask]
    A = np.stack([np.ones(d.size), X[mask], Y[mask]], axis=1)
    c, *_ = np.linalg.lstsq(A, d, rcond=None)
    r = d - A@c
    print(f"  {tag}: RMS(after piston+tilt removal) = {r.std()*1e9:9.4f} nm   "
          f"PV = {(r.max()-r.min())*1e9:9.4f} nm   piston = {c[0]*1e9:.4e} nm")
    return r

for N in (512, 1024):
    dx = 18e-6*512/N if False else 20e-3/N*1.2   # grid = 1.2 x aperture
    rx = common.plano_convex(R=100e-3, t=4e-3, ap=20e-3)
    E_in = np.ones((N, N), dtype=np.complex128)
    x = (np.arange(N)-N/2)*dx
    X, Y = np.meshgrid(x, x)
    R2 = X**2+Y**2
    mask = R2 <= (0.45*20e-3)**2       # inside 90% of the aperture radius
    opl_or, xe, ye = oracle_opd_on_exit_grid(rx, X, Y)
    for sub in (1, 8):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E = apply_real_lens_traced(
                E_in, prescription=rx, wavelength=WL, dx=dx,
                ray_subsample=sub, n_workers=1, on_undersample='silent',
                min_coarse_samples_per_aperture=0, on_pool_memory='silent')
        ph = np.unwrap(np.unwrap(np.angle(E), axis=0), axis=1)
        # align the unwrapped branch to the oracle at the centre
        c = N//2
        ph = ph + k0*opl_or[c, c] - ph[c, c]
        report(f"N={N} dx={dx*1e6:.3f}um sub={sub}", ph, opl_or, mask, X, Y)
        # ABSOLUTE piston: phase of the traced field at centre vs k0*OPL(0)
        pw = np.angle(E[c, c])
        want = (k0*opl_or[c, c]) % (2*np.pi)
        if want > np.pi: want -= 2*np.pi
        dwaves = ((pw - want + np.pi) % (2*np.pi) - np.pi)/(2*np.pi)
        print(f"      abs phase at centre: traced {pw:+.6f} rad, "
              f"k0*OPL_oracle(0) mod 2pi {want:+.6f} rad, diff {dwaves:+.6e} waves")
        # apply_real_lens's own piston
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            Ea = apply_real_lens(E_in, prescription=rx, wavelength=WL, dx=dx)
        pa = np.angle(Ea[c, c])
        dd = ((pw - pa + np.pi) % (2*np.pi) - np.pi)/(2*np.pi)
        print(f"      piston(traced) - piston(apply_real_lens) = {dd:+.6f} waves")
