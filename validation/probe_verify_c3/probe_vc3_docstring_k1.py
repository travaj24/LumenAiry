"""VERIFY-WP-C3 -- settle the two mutually inconsistent K1 readings that
``_collins_readout_k1``'s own docstring carries for ONE fixture.

carrier.py:2786 says "exit pitch 76.5 um, exit support 5.76 mm, K1 = 56.0";
carrier.py:2795 (same docstring, three sentences later) says "82.4 / 21.4 /
10.9 on N = 256 / 1024 / 2048"; carrier.py:1242 and 10364 say 82.36.  This
probe MEASURES the quantity on WP-B4's own ``_chain_fixture`` and prints the
exit pitch, the exit support box and K1 at three grids, on whichever tree the
caller pins.  Run: <tree-root> as cwd, PYTHONPATH=<tree-root>.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

TREE = os.path.abspath(os.environ.get('VC3_TREE', os.getcwd()))
assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)


def _gauss_env(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def _singlet(r1, r2, t, mat, semi, tag):
    from tests.unit.test_audit2609_b4_collins_transport import _singlet as s
    return s(r1, r2, t, mat, semi, tag)


def main():
    from tests.unit.test_audit2609_b4_collins_transport import (
        _chain_fixture, _singlet as mk)
    out = {'lumenairy': lumenairy.__file__, 'version': lumenairy.__version__,
           'python': sys.version.split()[0], 'numpy': np.__version__,
           'rows': []}
    from tests.unit.test_audit2609_b4_collins_transport import _CHAIN_TKW
    LAM = 1.31e-6
    env0, dx0, r_in, groups = _chain_fixture()
    n0 = env0.shape[0]
    for n in (256, 512, 1024, 2048):
        dx = dx0 * n0 / n
        env = _gauss_env(n, dx, 4.5e-3)
        # stop the chain AT its exit plane -- the plane the readout runs on
        res = C.propagate_traced_carrier_chain(
            env, groups, LAM, dx, r_in=r_in, ray_subsample=16, n_workers=1,
            traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
            final_distance=0.0, transport='sziklas')
        e, R, dxe = res.field, res.R, res.dx
        dxe = float(dxe[0]) if isinstance(dxe, tuple) else float(dxe)
        Rx, Ry, _ = C._parse_carrier(R, 'probe')
        rx, ry, thx, thy = C._collins_input_box(
            e, dxe, dxe, LAM, C._COLLINS_TAIL_FRAC)
        z = 8e-3
        Ax, B, _, _ = C._collins_envelope_abcd(Rx, z, np.inf)
        k1_hand = 2.0 * dxe * (abs(Ax) * rx / abs(B) + thx) / LAM
        k1_lib = C._collins_readout_k1(e, R, z, LAM, dxe, dxe)
        out['rows'].append(dict(
            N=n, exit_dx_um=dxe * 1e6, exit_R_m=Rx,
            exit_support_r_mm=rx * 1e3, exit_support_2r_mm=2 * rx * 1e3,
            exit_theta_mrad=thx * 1e3, A=Ax,
            k1_hand=k1_hand, k1_lib=k1_lib,
            k1_agree_ulp=abs(k1_hand - k1_lib) / max(np.spacing(k1_lib), 1e-300)))
    print(json.dumps(out, indent=1))
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'docstring_k1_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print('WROTE', p)


if __name__ == '__main__':
    main()
