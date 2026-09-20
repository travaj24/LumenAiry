"""VERIFY-WP-C3 -- HOW WIDE is the flip's blast radius on ordinary chains?

Defect D5 (an ordinary chain that returned at 49ddf4bd raising on the flipped
default) was found on one fixture family.  A ship recommendation needs its
width, not one counterexample.  This probe sweeps a deliberately ORDINARY
space -- one and two groups, three prescriptions, collimated and converging
launches, three grids, three beam widths, three final distances, with and
without a focus readout -- and classifies every cell as

    IDENTICAL / MOVED / OK->RAISED / RAISED->OK / BOTH-RAISED

between base 49ddf4bd (default 'sziklas') and this branch (default
'collins'), with NO `transport=` named on either side.  Nothing here uses a
stop-plane key, a tilt, an explicit `gap_kernel`, or any knob at all.

Run with cwd = the tree root, PYTHONPATH = the tree root, VC3_TREE = the tree
root, VC3_OUT = the output directory.
"""
import hashlib
import itertools
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)

LAM = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def singlet(r1, r2, t, ap):
    return {'name': 'p', 'aperture_diameter': ap, 'thicknesses': [t],
            'surfaces': [
                {'radius': r1, 'glass_before': 'air', 'glass_after': 'N-BK7',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': r2, 'glass_before': 'N-BK7', 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


PRESC = {
    'f60': singlet(60e-3, -60e-3, 3e-3, 14e-3),
    'f120': singlet(120e-3, -120e-3, 6e-3, 25.4e-3),
    'f300': singlet(300e-3, -300e-3, 4e-3, 25.4e-3),
}


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)


def cells():
    for pk, ngrp, n, w_mm, r_in, fd_mm, ro in itertools.product(
            ('f60', 'f120', 'f300'), (1, 2), (256, 512),
            (1.5, 3.0), (np.inf, 60e-3), (5.0, 15.0), (False, True)):
        yield dict(presc=pk, groups=ngrp, N=n, w_mm=w_mm,
                   r_in=('inf' if not np.isfinite(r_in) else r_in),
                   fd_mm=fd_mm, readout=ro)


def run(c):
    n = c['N']
    dx = 10.24e-3 / n
    p = PRESC[c['presc']]
    gs = [{'prescription': p, 'gap_before': 20e-3}]
    if c['groups'] == 2:
        gs.append({'prescription': p, 'gap_before': 15e-3})
    kw = dict(r_in=(np.inf if c['r_in'] == 'inf' else c['r_in']),
              ray_subsample=16, n_workers=1, traced_kwargs=TKW,
              final_leg='paraxial', final_distance=c['fd_mm'] * 1e-3)
    if c['readout']:
        kw['focus_readout'] = dict(dx_out=0.5e-6, N_out=64)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        try:
            r = C.propagate_traced_carrier_chain(
                gauss(n, dx, c['w_mm'] * 1e-3), gs, LAM, dx, **kw)
        except BaseException as exc:                    # noqa: BLE001
            return dict(outcome='raised', exc=type(exc).__name__,
                        msg=str(exc)[:160])
    h = hashlib.sha256(np.ascontiguousarray(
        r.field, dtype=np.complex128).tobytes()).hexdigest()
    kelly = sum(1 for x in wl if 'under-sampled' in str(x.message))
    return dict(outcome='returned', sha=h, kelly=kelly,
                peak=float(np.max(np.abs(r.field) ** 2)))


def main():
    import inspect
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'default_transport': inspect.signature(
               C.propagate_traced_carrier_chain
           ).parameters['transport'].default,
           'cells': {}}
    for c in cells():
        key = (f"{c['presc']}|g{c['groups']}|N{c['N']}|w{c['w_mm']}"
               f"|r{c['r_in']}|fd{c['fd_mm']}|ro{int(c['readout'])}")
        out['cells'][key] = run(c)
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.environ['VC3_OUT'], f'blastwidth_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    nr = sum(1 for v in out['cells'].values() if v['outcome'] == 'raised')
    nk = sum(v.get('kelly', 0) for v in out['cells'].values())
    print(f"DEFAULT={out['default_transport']} cells={len(out['cells'])} "
          f"raised={nr} kelly_warnings={nk}")
    print('WROTE', p)


if __name__ == '__main__':
    main()
