"""Probe 3/4: build_inverse_map cache key vs the D15 determinism flag, and
imap-vs-Newton accuracy on a real singlet congruence."""
import sys, time, warnings, numpy as np
sys.path.insert(0, r"docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/repro/TR-SIBLINGS")
from oracle import trace_singlet
import lumenairy
from lumenairy.glass import get_glass_index
from lumenairy.elements import _lens_imap as IM
from lumenairy.elements import _lens_traced as LT

lam = 0.5876e-6
ng = float(get_glass_index('N-BK7', lam))
R1, R2, d = 25e-3, float('inf'), 3e-3
LR = 3.0e-3
n_launch = 129
xs = np.linspace(-LR, LR, n_launch)
Xi, Yi = np.meshgrid(xs, xs, indexing='ij')
rho = np.hypot(Xi, Yi)

def trace2d(xin, yin, zout=0.0):
    """rot-sym exact map: radial trace + azimuth."""
    rr = np.hypot(xin, yin)
    xm, opl, L, Nz = trace_singlet(np.maximum(rr, 1e-15), R1, R2, d, ng, zout)
    scale = np.where(rr > 0, xm/np.maximum(rr, 1e-300), 0.0)
    # on-axis reference
    x0, o0, _, _ = trace_singlet(np.array([1e-12]), R1, R2, d, ng, zout)
    return xin*scale, yin*scale, opl-o0[0]

XO, YO, OP = trace2d(Xi, Yi)
amp = np.exp(-(rho/2.0e-3)**2)

# incumbent: cubic-spline interpolation of the forward map + Newton (a stand-in
# for _invert_newton; only its IDENTITY matters for the cache probe)
from scipy.interpolate import RectBivariateSpline
sx = RectBivariateSpline(xs, xs, XO); sy = RectBivariateSpline(xs, xs, YO)
so = RectBivariateSpline(xs, xs, OP)
def parity_invert(xq, yq, iters=8):
    xq = np.atleast_1d(np.asarray(xq, float)); yq = np.atleast_1d(np.asarray(yq, float))
    xe = xq.copy(); ye = yq.copy()
    for _ in range(iters):
        fx = sx.ev(xe, ye); fy = sy.ev(xe, ye)
        jxx = sx.ev(xe, ye, dx=1); jxy = sx.ev(xe, ye, dy=1)
        jyx = sy.ev(xe, ye, dx=1); jyy = sy.ev(xe, ye, dy=1)
        det = jxx*jyy - jxy*jyx; det = np.where(np.abs(det) > 1e-18, det, 1e-18)
        rx_, ry_ = fx-xq, fy-yq
        xe = np.clip(xe - (jyy*rx_-jxy*ry_)/det, -LR, LR)
        ye = np.clip(ye - (-jyx*rx_+jxx*ry_)/det, -LR, LR)
    return xe, ye, so.ev(xe, ye)
def probe_trace(px, py):
    return trace2d(np.asarray(px), np.asarray(py))

kw = dict(wavelength=lam, launch_radius=LR, census_amp=amp,
          parity_invert=parity_invert, parity_tag=('spline', 8),
          probe_trace=probe_trace)

print('== A) is DETERMINISTIC_TRACED_FIT in the cache key? ==')
IM.inverse_map_cache_clear()
old = LT.DETERMINISTIC_TRACED_FIT
try:
    LT.DETERMINISTIC_TRACED_FIT = True
    r1 = {}; m1 = IM.build_inverse_map(xs, XO, YO, OP, guard_record=r1, **kw)
    LT.DETERMINISTIC_TRACED_FIT = False
    r2 = {}; m2 = IM.build_inverse_map(xs, XO, YO, OP, guard_record=r2, **kw)
    LT.DETERMINISTIC_TRACED_FIT = True
    r3 = {}; m3 = IM.build_inverse_map(xs, XO, YO, OP, guard_record=r3, **kw)
finally:
    LT.DETERMINISTIC_TRACED_FIT = old
print('   refusals:', r1.get('refused'), r2.get('refused'), r3.get('refused'))
print('   cached flags:', r1.get('cached'), r2.get('cached'), r3.get('cached'))
print('   cache info:', IM.inverse_map_cache_info())
print('   key(det=True) == key(det=False)?', m1 is not None and m2 is not None
      and m1.key == m2.key)
print('   SAME OBJECT served for det=False?', m2 is m1)

# now force a cold build of the det=False arm to see whether the bits differ
IM.inverse_map_cache_clear()
try:
    LT.DETERMINISTIC_TRACED_FIT = False
    rc = {}; cold = IM.build_inverse_map(xs, XO, YO, OP, guard_record=rc, cache=False, **kw)
finally:
    LT.DETERMINISTIC_TRACED_FIT = old
if m1 is not None and cold is not None:
    dd = np.max(np.abs(m1.coef - cold.coef))
    rel = dd/np.max(np.abs(m1.coef))
    print('   coef(det=True) vs cold coef(det=False): max abs %.4g  rel %.4g  bit-identical=%s'
          % (dd, rel, bool(np.array_equal(m1.coef, cold.coef))))

print()
print('== B) LSTSQ_CONDITIONING_STEPDOWN in the key? ==')
IM.inverse_map_cache_clear()
oldc = LT.LSTSQ_CONDITIONING_STEPDOWN
try:
    LT.LSTSQ_CONDITIONING_STEPDOWN = True
    a = IM.build_inverse_map(xs, XO, YO, OP, **kw)
    LT.LSTSQ_CONDITIONING_STEPDOWN = False
    b = IM.build_inverse_map(xs, XO, YO, OP, **kw)
finally:
    LT.LSTSQ_CONDITIONING_STEPDOWN = oldc
print('   same key?', a is not None and b is not None and a.key == b.key,
      ' same object?', b is a, ' cache', IM.inverse_map_cache_info())

print()
print('== C) accuracy: imap model vs the spline-Newton incumbent, off lattice ==')
IM.inverse_map_cache_clear()
rec = {}
m = IM.build_inverse_map(xs, XO, YO, OP, guard_record=rec, **kw)
for kx in ('refused','n_fit_samples','n_terms','n_probe_traced','parity_map_opl_waves',
           'parity_incumbent_opl_waves','parity_ratio_opl','parity_map_pos_m',
           'parity_incumbent_pos_m','fit_resid_opl_waves','build_seconds'):
    print('   %-28s %s' % (kx, rec.get(kx)))
if m is not None:
    rng = np.random.default_rng(0)
    px = rng.uniform(-0.7*LR, 0.7*LR, 4000); py = rng.uniform(-0.7*LR, 0.7*LR, 4000)
    tx, ty, top = trace2d(px, py)
    xi, yi, op = m.eval(tx, ty, channels=(0, 1, 2))
    ix, iy, io = parity_invert(tx, ty)
    print('   MAP        : x_in rms %.4g m  OPL rms %.4g waves'
          % (np.sqrt(np.mean((xi-px)**2)), np.sqrt(np.mean((op-top)**2))/lam))
    print('   INCUMBENT  : x_in rms %.4g m  OPL rms %.4g waves'
          % (np.sqrt(np.mean((ix-px)**2)), np.sqrt(np.mean((io-top)**2))/lam))
    t0=time.time()
    for _ in range(3): m.eval(tx, ty, channels=(0,1,2))
    t_m=(time.time()-t0)/3
    t0=time.time()
    for _ in range(3): parity_invert(tx, ty)
    t_i=(time.time()-t0)/3
    print('   eval time 4000 pts: map %.4g s  incumbent(8 Newton) %.4g s' % (t_m, t_i))
