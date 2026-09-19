"""VERIFY-WAVE5-E / E5 (O-3): the immersed-exit guard, both sides.

1. THE GUARD FIRES at each of the four FGA sites on a synthetic immersed exit,
   on MY prescription (different radii / glass / wavelength / aperture from the
   item-E fixture), and does NOT fire on the air-terminated twin.
2. THE TOLERANCE IS TWO-SIDED, swept as a function of the index rather than
   asserted at one point: ``resolve_exit_index`` is intercepted so the guard
   sees an exact index, and the refusal boundary is located by bisection at
   several image distances.  STP air (1.000277), n = 1.0001, water and oil are
   each read off that curve.
3. AIR IS EXACTLY 1.0 on this registry at every wavelength the library uses --
   the claim the guard's floor rests on.
"""
import copy
import json
import sys
import warnings

import numpy as np

import lumenairy as la
from lumenairy.propagators import fga as _fga
from lumenairy.raytrace import exit_vertex as _ev, surfaces_from_prescription

_LAM = 1.55e-6
_SEMI = 0.18e-3
_T = 0.70e-3
_GLASS = 'N-LAK22'


def _presc(glass_after):
    s1 = {'radius': np.inf, 'conic': 0.0, 'thickness': _T,
          'glass_before': 'air', 'glass_after': _GLASS,
          'semi_diameter': _SEMI}
    s2 = {'radius': -1.22e-3, 'conic': -0.25, 'thickness': 0.0,
          'glass_before': _GLASS, 'glass_after': glass_after,
          'semi_diameter': _SEMI}
    return {'name': 'v_o3', 'aperture_diameter': 2 * _SEMI,
            'surfaces': [s1, s2], 'thicknesses': [_T], 'stop_index': 0}


def _surfs(p):
    s = [copy.copy(x) for x in surfaces_from_prescription(p)]
    s[-1].thickness = 0.0
    return s


def _E(n=64):
    xs = (np.arange(n) - n // 2) * 7.0e-6
    X, Y = np.meshgrid(xs, xs)
    return np.exp(-(X ** 2 + Y ** 2) / (46e-6) ** 2).astype(np.complex128)


def _call(site, presc, z_img=0.35e-3):
    E = _E()
    kw = dict(prescription=presc, wavelength=_LAM, dx=7.0e-6,
              output_plane_distance=z_img)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if site == 'through_lens':
            return la.apply_real_lens_fga(E, **kw)
        if site == 'coarse':
            return la.apply_real_lens_fga(E, coarse_stride=3, **kw)
        if site == 'vector':
            return la.apply_real_lens_fga_vector(np.stack([E, E * 0.5]), **kw)
        return _fga._caustic_zone(E, 7.0e-6, presc, _LAM)


def _fires(site, presc):
    try:
        _call(site, presc)
    except NotImplementedError as exc:
        return dict(fired=True, immersed_in_message=('IMMERSED' in str(exc)),
                    message=str(exc)[:220])
    except Exception as exc:                                # noqa: BLE001
        return dict(fired=False, other_exception=type(exc).__name__,
                    message=str(exc)[:220])
    return dict(fired=False)


def _refuses_at(n_exit, z_img, surfs):
    """Would the guard refuse this exact index at this leg length?"""
    old = _ev.resolve_exit_index
    _ev.resolve_exit_index = lambda *a, **k: n_exit
    try:
        _fga._require_non_immersed_exit(surfs, _LAM, z_img, 'probe')
        return False
    except NotImplementedError:
        return True
    finally:
        _ev.resolve_exit_index = old


def _boundary(z_img, surfs, lo=0.0, hi=1.0):
    """Bisect for the smallest ``|n-1|`` the guard refuses."""
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if _refuses_at(1.0 + mid, z_img, surfs):
            hi = mid
        else:
            lo = mid
    return hi


def main():
    out = dict(lumenairy_file=la.__file__, python=sys.version.split()[0],
               platform=sys.platform, numpy=np.__version__,
               waves_budget=_fga._FGA_IMAGE_LEG_WAVE_BUDGET,
               noise_floor=_fga._FGA_EXIT_INDEX_NOISE_FLOOR,
               wavelength=_LAM)
    print('lumenairy.__file__ =', la.__file__, flush=True)

    # ---- air is exactly 1.0 on this registry ------------------------------
    from lumenairy.glass import get_glass_index
    out['air_index'] = {}
    for wl in (633e-9, 780e-9, 1030e-9, 1060e-9, 1310e-9, 1550e-9):
        v = float(get_glass_index('air', wl))
        out['air_index']['%.0fnm' % (wl * 1e9)] = v
    out['air_is_exactly_one'] = all(v == 1.0
                                    for v in out['air_index'].values())
    print('air exactly 1.0 at every wavelength:', out['air_is_exactly_one'],
          flush=True)

    # ---- the four sites, both prescriptions -------------------------------
    imm, air = _presc(_GLASS), _presc('air')
    out['exit_index_immersed'] = float(
        _ev.resolve_exit_index(_surfs(imm), _LAM, fn_name='probe'))
    out['exit_index_air'] = float(
        _ev.resolve_exit_index(_surfs(air), _LAM, fn_name='probe'))
    out['sites'] = {}
    for site in ('through_lens', 'coarse', 'vector', 'caustic_zone'):
        out['sites'][site] = dict(immersed=_fires(site, imm),
                                  air=_fires(site, air))
        print('%-14s immersed fired=%s  air fired=%s'
              % (site, out['sites'][site]['immersed']['fired'],
                 out['sites'][site]['air']['fired']), flush=True)
    out['guard_two_sided_at_sites'] = all(
        v['immersed']['fired'] and v['immersed'].get('immersed_in_message')
        and not v['air']['fired'] for v in out['sites'].values())

    # ---- the tolerance curve ---------------------------------------------
    surfs = _surfs(imm)
    out['tolerance'] = []
    for z in (0.0, 1e-6, 1.55e-6, 1e-5, 1e-4, 3.5e-4, 1e-3, 1e-2):
        b = _boundary(z, surfs)
        out['tolerance'].append(dict(
            z_image_m=z, refusal_boundary_abs_n_minus_1=b,
            predicted=max(_fga._FGA_EXIT_INDEX_NOISE_FLOOR,
                          _fga._FGA_IMAGE_LEG_WAVE_BUDGET * _LAM
                          / max(abs(z), _LAM)),
            refuses_stp_air=_refuses_at(1.000277, z, surfs),
            refuses_n_1p0001=_refuses_at(1.0001, z, surfs),
            refuses_water=_refuses_at(1.33, z, surfs),
            refuses_exact_one=_refuses_at(1.0, z, surfs)))
        r = out['tolerance'][-1]
        print('z=%9.3e  boundary |n-1| = %.4e (predicted %.4e)  '
              'STP air refused=%s  n=1.0001 refused=%s  water refused=%s'
              % (z, b, r['predicted'], r['refuses_stp_air'],
                 r['refuses_n_1p0001'], r['refuses_water']), flush=True)
    out['stp_air_refused_anywhere'] = any(
        r['refuses_stp_air'] for r in out['tolerance'])
    out['water_refused_everywhere'] = all(
        r['refuses_water'] for r in out['tolerance'])
    print(json.dumps(out, indent=1))


main()
