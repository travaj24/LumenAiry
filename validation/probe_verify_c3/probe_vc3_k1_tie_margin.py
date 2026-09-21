"""VERIFY-WP-C3 -- HOW FAR is the route decision from flipping, measured on
the quantity it is actually decided on?

``_collins_readout_k1`` is a STAIRCASE (see probe_vc3_k1_quantisation.py).  So
"K1 = 0.99958, i.e. 4.2e-4 below the bar" is NOT the margin: K1 cannot take a
value in (1 - 2/N, 1) at all unless the SPACE term lands there, and one bin of
the ANGLE containment radius moves K1 by 2/N = 1.95e-3 at N = 1024 -- 4.7x the
quoted margin.

The real margin is the TIE MARGIN of the ``searchsorted`` inside
``_collins_containment_radius``: how far ``(1 - 1e-6) * tot`` sits from the
cumulative power ``c[i-1]`` that decides the index, in units of the ULP of
``tot``.  A reduction whose last bits move (a different FFT, a different
summation order, a different BLAS) flips the index iff it moves the cumsum by
more than that gap.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

TREE = os.path.abspath(os.environ.get('VC3_TREE', os.getcwd()))
assert os.path.abspath(lumenairy.__file__).startswith(TREE)


def tie(P, coord, frac):
    """The searchsorted decision, and how close it is to a tie."""
    P = np.asarray(P, dtype=np.float64)
    tot = float(P.sum())
    d = np.abs(np.asarray(coord, dtype=np.float64))
    order = np.argsort(d, kind='stable')
    c = np.cumsum(P[order])
    target = (1.0 - float(frac)) * tot
    i = int(np.searchsorted(c, target, side='left'))
    lo = float(c[i - 1]) if i > 0 else 0.0
    hi = float(c[min(i, c.size - 1)])
    ulp = float(np.spacing(tot))
    return dict(index=i, n=int(c.size), target=target, c_lo=lo, c_hi=hi,
                gap_below_ulps=(target - lo) / ulp,
                gap_above_ulps=(hi - target) / ulp,
                gap_below_rel=(target - lo) / tot,
                gap_above_rel=(hi - target) / tot,
                radius=float(d[order[min(i, d.size - 1)]]))


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def main():
    from tests.unit.test_audit2609_b4_collins_transport import (
        _chain_fixture, _CHAIN_TKW)
    LAM = 1.31e-6
    out = {'lumenairy': lumenairy.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__, 'rows': []}
    env0, dx0, r_in, groups = _chain_fixture()
    n0 = env0.shape[0]
    for n in (256, 512, 1024):
        dx = dx0 * n0 / n
        env = gauss(n, dx, 4.5e-3)
        res = C.propagate_traced_carrier_chain(
            env, groups, LAM, dx, r_in=r_in, ray_subsample=16, n_workers=1,
            traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
            final_distance=0.0, transport='sziklas')
        e, R, dxe = res.field, res.R, res.dx
        dxe = float(dxe[0]) if isinstance(dxe, tuple) else float(dxe)
        spec = np.fft.fft2(np.ascontiguousarray(e, dtype=np.complex128))
        Sx, _Sy = C._collins_power_marginals(spec)
        Px, _Py = C._collins_power_marginals(e)
        fx = np.fft.fftfreq(n, d=dxe)
        x = (np.arange(n, dtype=np.float64) - n / 2) * dxe
        ta = tie(Sx, fx, C._COLLINS_TAIL_FRAC)
        ts = tie(Px, x, C._COLLINS_TAIL_FRAC)
        k1 = C._collins_readout_k1(e, R, 8e-3, LAM, dxe, dxe)
        out['rows'].append(dict(
            N=n, exit_dx_um=dxe * 1e6, k1=k1,
            angle_bin_step_in_k1=2.0 / n,
            angle_term=2.0 * dxe * ta['radius'] * LAM / LAM / LAM * 0 +
            2.0 * dxe * (ta['radius'] * LAM) / LAM,
            angle_tie=ta, space_tie=ts,
            angle_index_over_nyquist=ta['index'],
            k1_if_angle_index_moved_one_bin_up=k1 + 2.0 / n,
            k1_if_angle_index_moved_one_bin_down=k1 - 2.0 / n))
    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'k1_tie_margin_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print('WROTE', p)


if __name__ == '__main__':
    main()
