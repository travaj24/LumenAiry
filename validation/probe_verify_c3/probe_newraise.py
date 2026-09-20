"""VERIFY-WP-C3 -- does a PUBLIC chain call that worked at 49ddf4bd RAISE on
the branch's default, and on how many fixtures?

The CHANGELOG's Migration paragraph says:

    "No public call that worked on 5.48.1 raises on 5.49.0, with ONE
    exception, and it is a JAX one"

and the Migration-Guide says the readout's stop-plane keys and
``final_distance=0`` "keep working exactly as they did".  This probe sweeps a
GRID of ordinary chain configurations -- no ``transport=`` named anywhere --
and records, per fixture: the outcome, the exception text, every warning in
emission order, and (when it returns) the peak and the pitch.  Run it on the
base tree and on the branch tree and diff the two JSONs.

    python probe_newraise.py <tree> <out.json>
"""
from __future__ import annotations

import json
import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                            # noqa: E402

import vlib3                                                  # noqa: E402

vlib3.bind(_TREE)

import lumenairy.propagators.carrier as CA                    # noqa: E402

WL = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')
OUT = {}


def sq(n, dx, w):
    x = (np.arange(n) - n / 2.0) * dx
    return np.exp(-((x[:, None] / w) ** 2
                    + (x[None, :] / w) ** 2)).astype(np.complex128)


def presc(name, r1, r2, t, ap=16e-3):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [t],
            'surfaces': [
                {'radius': r1, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': r2, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


PCX = presc('pcx', 52e-3, np.inf, 5e-3)
BCX = presc('bcx', 75e-3, -75e-3, 4e-3)
BIG = presc('big', 120e-3, -120e-3, 6e-3, ap=25.4e-3)


def run(tag, fn, *a, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        try:
            res = fn(*a, **kw)
            f = np.asarray(res.field)
            rec = {'outcome': 'ok',
                   'peak': float(np.nanmax(np.abs(f)) ** 2),
                   'power': float(np.nansum(np.abs(f) ** 2)),
                   'dx': repr(res.dx)}
        except BaseException as exc:                          # noqa: BLE001
            rec = {'outcome': 'raised', 'exc': type(exc).__name__,
                   'msg': str(exc)[:260]}
    rec['warnings'] = [(w.category.__name__, str(w.message)[:150])
                       for w in caught]
    rec['n_runtimewarn'] = sum(1 for w in caught
                               if w.category is RuntimeWarning)
    OUT[tag] = rec


def chain(**kw):
    return CA.propagate_traced_carrier_chain(**kw)


FIXTURES = []


def add(tag, **kw):
    FIXTURES.append((tag, kw))


# --- a SINGLE group, a wide grid, a comfortable aperture ------------------
for N, dx, w in ((256, 40e-6, 2.0e-3), (512, 20e-6, 2.0e-3),
                 (256, 20e-6, 1.0e-3), (512, 40e-6, 4.0e-3)):
    add('F1-single-N%d-dx%gum-w%gmm' % (N, dx * 1e6, w * 1e3),
        E_in=sq(N, dx, w), groups=[{'prescription': BIG,
                                    'gap_before': 20e-3}],
        wavelength=WL, dx=dx, r_in=np.inf, ray_subsample=16, n_workers=1,
        traced_kwargs=TKW, final_leg='paraxial', final_distance=10e-3,
        focus_readout=dict(dx_out=0.5e-6, N_out=64))

# --- two groups, collimated in, ordinary gaps -----------------------------
for N, dx, w in ((256, 40e-6, 2.0e-3), (512, 20e-6, 2.0e-3)):
    add('F2-two-N%d-dx%gum-w%gmm' % (N, dx * 1e6, w * 1e3),
        E_in=sq(N, dx, w),
        groups=[{'prescription': BIG, 'gap_before': 20e-3},
                {'prescription': BCX, 'gap_before': 30e-3}],
        wavelength=WL, dx=dx, r_in=np.inf, ray_subsample=16, n_workers=1,
        traced_kwargs=TKW, final_leg='paraxial', final_distance=10e-3,
        focus_readout=dict(dx_out=0.5e-6, N_out=64))

# --- the stop-plane keys the Migration-Guide says keep working -------------
add('F3-standoff-2mm', E_in=sq(256, 40e-6, 2.0e-3),
    groups=[{'prescription': BIG, 'gap_before': 20e-3}],
    wavelength=WL, dx=40e-6, r_in=np.inf, ray_subsample=16, n_workers=1,
    traced_kwargs=TKW, final_leg='paraxial', final_distance=10e-3,
    focus_readout=dict(dx_out=0.5e-6, N_out=64, standoff=2e-3))
add('F3-containment-warn', E_in=sq(256, 40e-6, 2.0e-3),
    groups=[{'prescription': BIG, 'gap_before': 20e-3}],
    wavelength=WL, dx=40e-6, r_in=np.inf, ray_subsample=16, n_workers=1,
    traced_kwargs=TKW, final_leg='paraxial', final_distance=10e-3,
    focus_readout=dict(dx_out=0.5e-6, N_out=64,
                       on_focus_containment='warn'))
add('F3-zero-distance', E_in=sq(256, 40e-6, 2.0e-3),
    groups=[{'prescription': BIG, 'gap_before': 20e-3}],
    wavelength=WL, dx=40e-6, r_in=np.inf, ray_subsample=16, n_workers=1,
    traced_kwargs=TKW, final_leg='paraxial', final_distance=0.0,
    focus_readout=dict(dx_out=0.5e-6, N_out=64))

# --- final_leg='exact' -- claimed unmoved ---------------------------------
add('F4-final-leg-exact-bare', E_in=sq(256, 40e-6, 2.0e-3),
    groups=[{'prescription': BIG, 'gap_before': 20e-3},
            {'prescription': BCX, 'gap_before': 30e-3}],
    wavelength=WL, dx=40e-6, r_in=np.inf, ray_subsample=16, n_workers=1,
    traced_kwargs=TKW, final_leg='exact', final_distance=10e-3)
add('F4-final-leg-exact-readout', E_in=sq(256, 40e-6, 2.0e-3),
    groups=[{'prescription': BIG, 'gap_before': 20e-3},
            {'prescription': BCX, 'gap_before': 30e-3}],
    wavelength=WL, dx=40e-6, r_in=np.inf, ray_subsample=16, n_workers=1,
    traced_kwargs=TKW, final_leg='exact', final_distance=10e-3,
    focus_readout=dict(dx_out=0.5e-6, N_out=64))
add('F4-single-final-leg-exact-bare', E_in=sq(256, 40e-6, 2.0e-3),
    groups=[{'prescription': BIG, 'gap_before': 20e-3}],
    wavelength=WL, dx=40e-6, r_in=np.inf, ray_subsample=16, n_workers=1,
    traced_kwargs=TKW, final_leg='exact', final_distance=10e-3)

# --- a converging launch, the shape the wayback probe used ----------------
add('F5-converging-two', E_in=sq(256, 55e-6, 4.0e-3),
    groups=[{'prescription': presc('pcx', 52e-3, np.inf, 5e-3),
             'gap_before': 18e-3},
            {'prescription': BCX, 'gap_before': 12e-3}],
    wavelength=WL, dx=55e-6, r_in=-70e-3, ray_subsample=16, n_workers=1,
    traced_kwargs=TKW, final_leg='paraxial', final_distance=9e-3,
    focus_readout=dict(dx_out=0.4e-6, N_out=64))


def main():
    for tag, kw in FIXTURES:
        run(tag, chain, **kw)
    OUT['_meta'] = {'tree': _TREE, 'build': vlib3.build_tag()}
    with open(sys.argv[2], 'w', encoding='utf-8') as fh:
        json.dump(OUT, fh, indent=1, sort_keys=True, default=str)
    sys.stdout.write('[probe_newraise] %d fixtures -> %s\n'
                     % (len(FIXTURES), sys.argv[2]))


if __name__ == '__main__':
    main()
