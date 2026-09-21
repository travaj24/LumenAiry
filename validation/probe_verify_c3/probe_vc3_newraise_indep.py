"""VERIFY-WP-C3 -- an INDEPENDENT reproduction of the one claim that decides
the ship recommendation:

    CHANGELOG.md, Migration paragraph: "No public call that worked on 5.48.1
    raises on 5.49.0, with ONE exception, and it is a JAX one."

A sibling measurement reported 23 of 103 archive keys going ok -> RuntimeError
on the flipped DEFAULT.  This probe is written from scratch, on a fixture
chosen for being ordinary (a collimated launch into a two-group relay, no
stop-plane keys, no exotic kwargs), and records base-vs-branch outcome per
configuration.  Run with cwd = the tree root, PYTHONPATH = the tree root,
VC3_TREE = the tree root, VC3_OUT = the output directory.
"""
import json
import os
import sys
import traceback
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


def singlet(r1, r2, t, glass, ap):
    return {'name': 'p', 'aperture_diameter': ap, 'thicknesses': [t],
            'surfaces': [
                {'radius': r1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': r2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)


def main():
    import inspect
    out = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'default_transport': inspect.signature(
               C.propagate_traced_carrier_chain
           ).parameters['transport'].default,
           'rows': []}
    # an ORDINARY two-group relay, COLLIMATED in, no stop-plane keys.
    for n, dx, w in ((512, 20e-6, 2.0e-3), (256, 20e-6, 2.0e-3),
                     (512, 20e-6, 3.0e-3), (1024, 10e-6, 2.0e-3)):
        for fd in (5e-3, 8e-3, 20e-3):
            p = singlet(120e-3, -120e-3, 6e-3, 'N-BK7', 25.4e-3)
            groups = [{'prescription': p, 'gap_before': 20e-3},
                      {'prescription': p, 'gap_before': 15e-3}]
            row = dict(N=n, dx_um=dx * 1e6, w_mm=w * 1e3, final_distance=fd)
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                try:
                    res = C.propagate_traced_carrier_chain(
                        gauss(n, dx, w), groups, LAM, dx, r_in=np.inf,
                        ray_subsample=16, n_workers=1, traced_kwargs=TKW,
                        final_leg='paraxial', final_distance=fd,
                        focus_readout=dict(dx_out=0.5e-6, N_out=64))
                    row['outcome'] = 'returned'
                    row['peak'] = float(np.max(np.abs(res.field) ** 2))
                    st = res.stages[-1]
                    row['route'] = st.get('readout_route')
                    row['route_k1'] = st.get('readout_route_k1')
                    row['reason'] = st.get('readout_route_reason')
                except BaseException as exc:            # noqa: BLE001
                    row['outcome'] = 'raised'
                    row['exc'] = type(exc).__name__
                    row['msg'] = str(exc)[:300]
                    row['where'] = traceback.extract_tb(
                        exc.__traceback__)[-1].name
                row['carrier_warnings'] = sorted(
                    {str(x.message)[:60] for x in wl
                     if 'carrier' in str(x.filename)})
            out['rows'].append(row)
            print(f"N={n:5d} dx={dx*1e6:5.1f}um w={w*1e3:4.1f}mm fd={fd*1e3:5.1f}mm"
                  f"  -> {row['outcome']:8s}"
                  f"  {row.get('exc', '')}{row.get('peak', '')}"
                  f"  route={row.get('route')}"
                  f" k1={row.get('route_k1')}")
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.environ['VC3_OUT'], f'newraise_indep_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('DEFAULT =', out['default_transport'], '| WROTE', p)


if __name__ == '__main__':
    main()
