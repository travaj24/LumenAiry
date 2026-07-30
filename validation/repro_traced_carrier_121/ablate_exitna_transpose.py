# Is ``apply_real_lens_traced``'s MEASURED exit NA (_exit_na_out['na_exit'],
# and the "exit beam converges at NA_exit=..." Nyquist warning) reading the
# amplitude mask TRANSPOSED with respect to the ray order?
#
# The launch lattice is built with ``indexing='ij'``, so ray r = i*n + j sits at
# entrance (x = xs_in[i], y = xs_in[j]).  The significance mask is
# ``_amp = |E_in|[np.ix_(_ray_iy, _ray_ix)]``, whose element [a, b] is |E_in| at
# (y = xs_in[a], x = xs_in[b]) -- i.e. (y, x), the transpose of the ray order it
# is then ravelled against.  Invisible for any beam symmetric in x <-> y, which
# is why it has never shown.
#
# TEST: a BICONIC singlet (different x and y power) with a small collimated
# beam parked at (+a, 0) and then at (0, +a).  The exit NA of the two is
# genuinely different (a/f_x vs a/f_y).  If the mask is transposed the two
# REPORTED values are swapped.
#
# Uses only public API + the private ``_exit_na_out`` diagnostic dict.  No
# library edit.
import warnings

import numpy as np

warnings.filterwarnings('ignore')
import _d121_common as C                                        # noqa: E402,F401
from lumenairy.elements import apply_real_lens_traced           # noqa: E402
from lumenairy.raytrace import RayBundle, trace                 # noqa: E402
from lumenairy.raytrace.trace import surfaces_from_prescription  # noqa: E402

LAM = 1.31e-6
N = 256
DX = 30e-6
A = 2.0e-3

presc = {
    'surfaces': [
        {'radius': 0.020, 'radius_y': 0.040, 'conic': 0.0, 'conic_y': 0.0,
         'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': np.inf, 'radius_y': np.inf, 'conic': 0.0, 'conic_y': 0.0,
         'glass_before': 'N-BK7', 'glass_after': 'air'},
    ],
    'thicknesses': [3.0e-3],
    'aperture_diameter': 8.0e-3,
    'name': 'biconic',
}

x = (np.arange(N) - N / 2) * DX
X, Y = np.meshgrid(x, x)
w = 0.35e-3


def na_reported(cx, cy):
    E = np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / w ** 2).astype(complex)
    d = {}
    apply_real_lens_traced(
        E, prescription=presc, wavelength=LAM, dx=DX, ray_subsample=4,
        n_workers=1, on_undersample='silent', on_aperture_beam='silent',
        _exit_na_out=d)
    return d.get('na_exit', np.nan)


def na_traced(cx, cy):
    """Direct truth: max |(L, M)| over the e^-4-amplitude disc about the beam,
    traced with the same collimated launch the element uses."""
    surfs = surfaces_from_prescription(
        {k: v for k, v in presc.items() if k != 'aperture_diameter'})
    t = np.linspace(-2.0 * w, 2.0 * w, 41)
    U, V = np.meshgrid(t, t)
    xs, ys = (U + cx).ravel(), (V + cy).ravel()
    keep = (U ** 2 + V ** 2).ravel() <= (2.0 * w) ** 2
    xs, ys = xs[keep], ys[keep]
    rb = RayBundle(x=xs, y=ys, z=np.zeros_like(xs), L=np.zeros_like(xs),
                   M=np.zeros_like(xs), N=np.ones_like(xs), wavelength=LAM,
                   alive=np.ones(xs.size, bool), opd=np.zeros_like(xs))
    ir = trace(rb, surfs, LAM, output_filter='last').image_rays
    al = np.asarray(ir.alive)
    return float(np.hypot(np.asarray(ir.L)[al], np.asarray(ir.M)[al]).max())


for cx, cy in ((A, 0.0), (0.0, A)):
    rep = na_reported(cx, cy)
    tru = na_traced(cx, cy)
    print(f"beam at ({cx * 1e3:+.2f}, {cy * 1e3:+.2f}) mm:  "
          f"element na_exit = {rep:.5f}   direct trace = {tru:.5f}   "
          f"ratio {rep / tru:.4f}")
print()
print("If the two 'direct trace' numbers differ and the two reported ones are "
      "SWAPPED relative to them, the significance mask is transposed.")
