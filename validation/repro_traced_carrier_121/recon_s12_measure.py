# S12 ``remap_sampling`` on the settled tree: WHAT the numbers are now, and
# WHETHER niche C6 is what moved them.
#
# The S12 pins compare 'full' and 'lattice' against a REFERENCE built by
# ``ray_subsample = 1`` ("everything sampled at full wave-grid resolution").
# Niche C6 changed the launch direction from grad(W) to grad(W + a_fit), and
# ``a_fit`` is a polynomial fitted to the residual sampled ON THE RAY LATTICE
# -- so ``ray_subsample`` now moves the RAY MAP ITSELF, not only the
# resolution at which the transported residual is sampled.  That is a second,
# C6-only channel into every number in this file, and it is what has to be
# separated before any pin is re-priced.
#
# This runner prints the whole matrix in both library states:
#   C6 ON  = the settled tree (REMAP_STATIONARY_PHASE_LAUNCH = True)
#   C6 OFF = the fail-before switch, which the CHANGELOG pins bit-for-bit to
#            the pre-C6 library -- i.e. the state the existing numbers in the
#            test's docstring were measured in.
#
# usage:  python recon_s12_measure.py
import hashlib
import os
import sys
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
sys.path.insert(0, _ROOT)

import lumenairy as la                                     # noqa: E402,I001
import lumenairy.elements._lens_traced as LT               # noqa: E402

_WL = 1.31e-6
_N, _DX, _W, _A, _RC = 256, 4.0e-6, 200e-6, 6.0, -0.02


def _singlet(R1, R2, d, glass, ap, name='s'):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


def setup():
    x = (np.arange(_N) - _N // 2) * _DX
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    k = 2.0 * np.pi / _WL
    S = np.sign(_RC) * (np.sqrt(r2 + _RC ** 2) - abs(_RC))
    E = (np.exp(-r2 / _W ** 2) * np.exp(1j * k * S)
         * np.exp(1j * _A * (r2 / _W ** 2) ** 2)).astype(np.complex128)
    presc = _singlet(3.1e-3, -3.1e-3, 1.0e-3, 'N-BK7', 1.2e-3, 'strong')
    kw = dict(prescription=presc, wavelength=_WL, dx=_DX, carrier=_RC,
              amplitude_model='ray_density', preserve_input_phase='remap',
              parallel_amp=False, on_undersample='silent',
              on_noncollimated='silent')
    return E, kw, np.sqrt(r2)


def run(E, kw, **over):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(la.apply_real_lens_traced(E, **kw, **over))


def rms_phase_diff(a, b, mask=None):
    m = (np.abs(a) > 1e-3 * np.abs(a).max()) & (np.abs(b) > 0)
    if mask is not None:
        m = m & mask
    if not m.any():
        return 0.0
    d = np.angle(b[m] / a[m])
    wt = np.abs(a[m])
    return float(np.sqrt((wt * d ** 2).sum() / wt.sum()))


def r_alias(h, A=_A, w=_W):
    return (np.pi * w ** 4 / (4.0 * A * h)) ** (1.0 / 3.0)


def main():
    print("   lib sha256 %s" % hashlib.sha256(
        open(LT.__file__, 'rb').read()).hexdigest()[:16])
    E, kw, rr = setup()
    ra = r_alias(4 * _DX)
    print(f"S12 fixture: N={_N} dx={_DX*1e6:g} um w={_W*1e6:g} um A={_A:g} rad "
          f"carrier={_RC:g} m")
    print(f"  r_alias(h = 4 dx) = {ra*1e6:.2f} um = {ra/_W:.3f} w")
    print()

    for c6 in (True, False):
        old = LT.REMAP_STATIONARY_PHASE_LAUNCH
        LT.REMAP_STATIONARY_PHASE_LAUNCH = c6
        try:
            tag = 'C6 ON  (settled tree)' if c6 else 'C6 OFF (= pre-C6)'
            print(f"=== {tag} ===")
            # warm-up, as the test module's autouse fixture does
            for _ in range(2):
                run(E, kw, ray_subsample=4)
            ref = run(E, kw, ray_subsample=1, remap_sampling='lattice')
            ref_f = run(E, kw, ray_subsample=1, remap_sampling='full')
            print(f"  rs=1 lattice vs full byte-identical: "
                  f"{np.array_equal(ref, ref_f)}")
            print(f"  {'rs':>3} {'d(ref,lat)':>12} {'d(ref,ful)':>12} "
                  f"{'ratio':>9} {'d(lat,ful)':>12}")
            for rs in (2, 4, 8):
                lat = run(E, kw, ray_subsample=rs, remap_sampling='lattice')
                ful = run(E, kw, ray_subsample=rs, remap_sampling='full')
                dl = rms_phase_diff(ref, lat)
                df = rms_phase_diff(ref, ful)
                dlf = rms_phase_diff(lat, ful)
                print(f"  {rs:>3} {dl:>12.4e} {df:>12.4e} "
                      f"{(dl/max(df,1e-30)):>9.1f} {dlf:>12.4e}   "
                      f"identical={np.array_equal(lat, ful)}")
            # the alias-radius structure, at rs = 4
            rs = 4
            lat = run(E, kw, ray_subsample=rs, remap_sampling='lattice')
            ful = run(E, kw, ray_subsample=rs, remap_sampling='full')
            inner, outer = rr < 0.75 * ra, rr > 1.05 * ra
            print(f"  alias structure at rs=4 (inner r<{0.75*ra/_W:.2f} w, "
                  f"outer r>{1.05*ra/_W:.2f} w):")
            for nm, F in (('lattice', lat), ('full', ful)):
                di = rms_phase_diff(ref, F, inner)
                do = rms_phase_diff(ref, F, outer)
                print(f"    {nm:>8}: inner {di:.4e}  outer {do:.4e}  "
                      f"outer/inner {do/max(di,1e-30):.1f}")
            # power bookkeeping, so EE-blind halo/energy moves are visible
            for nm, F in (('rs1 ref', ref), ('lattice', lat), ('full', ful)):
                print(f"    {nm:>8}: P = {float((np.abs(F)**2).sum())*_DX*_DX:.6e}"
                      f"  peak |E| = {float(np.abs(F).max()):.6f}")
            print()
        finally:
            LT.REMAP_STATIONARY_PHASE_LAUNCH = old


if __name__ == '__main__':
    sys.exit(main())
