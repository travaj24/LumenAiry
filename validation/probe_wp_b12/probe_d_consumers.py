"""WP-B12 probe D -- the blast radius: every other consumer of the two
differential-transfer primitives.

``grep -rn ray_transfer_jacobian lumenairy/`` finds exactly two consumers that
CALL them: ``propagators/fga.py`` (four sites, which now ask for
``reference='exit_vertex'``) and ``propagators/gbd.py``
(``apply_prescription_persurface_to_beamlets``, which does NOT pass the keyword
and therefore keeps the last-surface default).  This probe measures what that
means:

1. GBD's field on a curved-last-surface prescription is unchanged by WP-B12 --
   asserted structurally (the default) and measured here against the same call
   with the primitives forced back onto ``reference='surface'``;
2. GBD carries its OWN in-line vertex correction (a v5.22 fix), written from
   the conic radius and conic constant only.  On a CONIC last surface it agrees
   with the shared projection; on an EVEN-ASPHERIC one it does not, because it
   drops the polynomial departure.  That gap is measured here and reported as
   an open item -- it is a pre-existing GBD defect, not one WP-B12 introduces.

Run with OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1 and
PYTHONPATH pointing at the tree under test.
"""
from __future__ import annotations

import copy
import hashlib
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b12_common import assert_tree, build_tag, dump  # noqa: E402

ROOT = os.environ.get('B12_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
_LAM = 1.064e-6


def _sha(a):
    return hashlib.sha256(np.ascontiguousarray(
        np.asarray(a)).tobytes()).hexdigest()[:24]


def _presc(asph=None, R=2.10e-3, t=0.70e-3, semi=0.20e-3, glass='N-BAF10'):
    s1 = {'radius': -R, 'conic': 0.0, 'thickness': 0.0, 'glass_before': glass,
          'glass_after': 'air', 'semi_diameter': semi}
    if asph:
        s1['aspheric_coeffs'] = dict(asph)
    return {'name': 'b12d', 'aperture_diameter': 2 * semi,
            'surfaces': [
                {'radius': R, 'conic': 0.0, 'thickness': t,
                 'glass_before': 'air', 'glass_after': glass,
                 'semi_diameter': semi}, s1],
            'thicknesses': [t], 'stop_index': 0}


def main():
    assert_tree(ROOT)
    import lumenairy as la
    from lumenairy.elements import apply_real_lens_gbd
    from lumenairy.raytrace import differential as D, surfaces_from_prescription

    out = {'build': build_tag(), 'version': la.__version__}

    # ---- 1. GBD is unchanged: the default reference is still the surface ----
    N, dx, w0 = 128, 3.6e-6, 80e-6
    xs = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(xs, xs)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)

    def _pin(base):
        def wrapped(*a, **kw):
            kw['reference'] = 'surface'
            return base(*a, **kw)
        return wrapped

    res = {}
    for tag, force in (('as_shipped', False), ('forced_surface', True)):
        saved = (D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic)
        try:
            if force:
                D.ray_transfer_jacobian = _pin(saved[0])
                D.ray_transfer_jacobian_analytic = _pin(saved[1])
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                f = np.asarray(apply_real_lens_gbd(
                    E, prescription=_presc(), wavelength=_LAM, dx=dx,
                    output_plane_distance=1.2e-3))
            res[tag] = _sha(f)
        finally:
            D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic = saved
    out['gbd_curved_last_surface'] = {
        'sha_as_shipped': res['as_shipped'],
        'sha_forced_surface': res['forced_surface'],
        'bit_identical': res['as_shipped'] == res['forced_surface'],
    }
    print('GBD curved-last-surface field bit-identical to the forced-surface '
          f'arm: {out["gbd_curved_last_surface"]["bit_identical"]}')

    # ---- 2. GBD's in-line vertex correction vs the shared projection -------
    # GBD folds ``-sag`` into its image leg with a conic-only sag (gbd.py's
    # ``_Rl`` / ``_kl`` block).  Reproduce that expression here and compare it
    # with the shared projection, on a conic and on an aspheric last surface.
    out['gbd_inline_sag_vs_shared'] = {}
    for tag, asph in (('conic', None), ('asphere_A4_A6',
                                        {4: 4.0e8, 6: -8.0e17})):
        p = _presc(asph=asph)
        surfs = [copy.copy(s) for s in surfaces_from_prescription(p)]
        surfs[-1].thickness = 0.0
        n = 401
        h = np.linspace(0.20e-3 / (2 * n), 0.20e-3 * 0.98, n)
        z = np.zeros(n)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dt = D.ray_transfer_jacobian(h.copy(), z.copy(), z.copy(),
                                         z.copy(), surfs, _LAM)
            dv = D.ray_transfer_jacobian(h.copy(), z.copy(), z.copy(),
                                         z.copy(), surfs, _LAM,
                                         reference='exit_vertex')
        ok = np.asarray(dt.alive, bool)
        _Rl = float(getattr(surfs[-1], 'radius', np.inf))
        _kl = float(getattr(surfs[-1], 'conic', 0.0) or 0.0)
        r2 = np.asarray(dt.x) ** 2 + np.asarray(dt.y) ** 2
        _cl = 1.0 / _Rl
        gbd_sag = _cl * r2 / (1.0 + np.sqrt(np.maximum(
            1.0 - (1.0 + _kl) * _cl * _cl * r2, 0.0)))
        shared_sag = np.asarray(dt.x) - np.asarray(dv.x)
        shared_sag = np.divide(shared_sag, np.asarray(dt.ux),
                               out=np.zeros_like(shared_sag),
                               where=np.abs(np.asarray(dt.ux)) > 1e-12)
        gap = float(np.abs((gbd_sag - shared_sag)[ok]).max())
        out['gbd_inline_sag_vs_shared'][tag] = {
            'max_sag_gap_m': gap,
            'max_sag_gap_waves': gap / _LAM,
            'max_sag_m': float(np.abs(shared_sag[ok]).max()),
        }
        print(f'GBD in-line sag vs the shared sag, {tag:14s}: '
              f'{gap:.3e} m ({gap / _LAM:.3f} waves) on a sag of '
              f'{float(np.abs(shared_sag[ok]).max()):.3e} m')
    dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f'probe_d_consumers_{sys.platform}_'
                      f'{sys.version_info.major}{sys.version_info.minor}.json'),
         out)


if __name__ == '__main__':
    main()
