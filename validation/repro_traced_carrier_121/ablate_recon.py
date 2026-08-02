# Recon for the ABLATE study: what is in design 121's LAST post-DOE group,
# and is a per-surface ``decenter`` an EXACT rigid translation of the group?
#
# Nothing here is a finding -- it is the pre-flight check for
# ``ablate_recentre_121.py`` (the Option-A cheap probe).
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C                                        # noqa: E402
from lumenairy.raytrace import RayBundle, trace                  # noqa: E402
from lumenairy.raytrace.trace import surfaces_from_prescription  # noqa: E402

pre, post, gap, period = C.geometry()
print(f"n post groups = {len(post)}   doe period {period*1e6:.4f} um")
for i, g in enumerate(post):
    p = g['prescription']
    ap = p.get('aperture_diameter')
    sds = [s.get('semi_diameter') for s in p['surfaces']]
    print(f"grp {i}: gap_before {g['gap_before']*1e3:9.4f} mm  "
          f"nsurf {len(p['surfaces'])}  aperture_diameter "
          f"{None if ap is None else f'{ap*1e3:.4f} mm'}  "
          f"per-surf sd {sds}  keys {sorted(p.keys())}")
p5 = post[-1]['prescription']
print()
print("group 5 surfaces:")
for s in p5['surfaces']:
    print("   ", {k: v for k, v in s.items()
                  if k in ('radius', 'conic', 'glass_before', 'glass_after',
                           'semi_diameter', 'aspheric_coeffs', 'comment')})
print("   thicknesses", p5['thicknesses'])
print("   elements?", 'elements' in p5)

# ---- exactness of the rigid translation via per-surface 'decenter' ----------
# Trace a bundle through the group as-is, then through the group displaced by
# t with the bundle displaced by t.  Exit positions must differ by exactly t
# and the OPL must be identical.
LAM = C.LAM
tx, ty = 3.3723e-3, 0.0
for tag, tvec in (('t = (+3.3723, 0) mm', (tx, ty)),
                  ('t = (-3.3723, -1.6) mm', (-tx, -1.6e-3))):
    q = np.linspace(-1.2e-3, 1.2e-3, 9)
    X, Y = np.meshgrid(q, q)
    L0, M0 = 0.0461, 0.0230

    def _run(presc, ox, oy):
        s = surfaces_from_prescription(presc)
        n0 = np.sqrt(1 - L0 ** 2 - M0 ** 2)
        rb = RayBundle(x=X.ravel() + ox, y=Y.ravel() + oy,
                       z=np.zeros(X.size), L=np.full(X.size, L0),
                       M=np.full(X.size, M0), N=np.full(X.size, n0),
                       wavelength=LAM, alive=np.ones(X.size, bool),
                       opd=np.zeros(X.size))
        r = trace(rb, s, LAM, output_filter='last').image_rays
        return np.asarray(r.x), np.asarray(r.y), np.asarray(r.opd), \
            np.asarray(r.alive)

    p_inf = {**p5, 'surfaces': [{**s, 'semi_diameter': np.inf}
                                for s in p5['surfaces']]}
    p_dec = {**p_inf, 'surfaces': [{**s, 'decenter': tvec}
                                   for s in p_inf['surfaces']]}
    x0, y0, o0, a0 = _run(p_inf, 0.0, 0.0)
    x1, y1, o1, a1 = _run(p_dec, tvec[0], tvec[1])
    print(f"\nrigid-translation check, {tag}")
    print(f"   alive {a0.sum()}/{a0.size} vs {a1.sum()}/{a1.size}")
    print(f"   max |dx - tx| = {np.nanmax(np.abs(x1 - x0 - tvec[0])):.3e} m")
    print(f"   max |dy - ty| = {np.nanmax(np.abs(y1 - y0 - tvec[1])):.3e} m")
    print(f"   max |dOPL|    = {np.nanmax(np.abs(o1 - o0)):.3e} m "
          f"({np.nanmax(np.abs(o1 - o0)) / LAM:.3e} waves)")
