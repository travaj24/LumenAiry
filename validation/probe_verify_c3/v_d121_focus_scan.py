"""VERIFY-WP-C3 CLAIM 9e -- is "the fixed MSoP plane, not best focus" a
plausible explanation for 6.61 um against the shipped acceptance's 3.450 um?

    python v_d121_focus_scan.py <tree> <out.json> <N>

Same geometry, same launch, same readout lattice, ONE variable: the trailing
leg.  If the fixed plane is simply off best focus, a short through-focus scan
must show the FWHM collapsing toward the acceptance's number.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import vlib  # noqa: E402
import numpy as np  # noqa: E402

import lumenairy as la  # noqa: E402
import lumenairy.propagators.carrier as CA  # noqa: E402

vlib.anchor(_TREE)
sys.path.insert(0, os.path.join(_TREE, 'validation',
                                'repro_traced_carrier_121'))
os.environ['LUMENAIRY_ROOT'] = _TREE
os.environ.setdefault(
    'D121_ROOT',
    'D:' + os.sep + os.path.join('Metacept', 'Neurophos',
                                 'Python_Test_Scripts', 'Free_Space_Optics'))

LAM = 1.31e-6
W0 = 4e-6
TRAILING = 7.7058e-3
DXO = 0.25e-6
NOUT = 256


def _metrics(field, dx_out):
    I = np.abs(np.asarray(field)) ** 2
    n = I.shape[-1]
    ax = (np.arange(n) - n / 2.0) * dx_out
    pk = float(I.max())
    iy, ix = np.unravel_index(int(np.argmax(I)), I.shape)
    row = I[iy]
    half = pk / 2.0
    xs = []
    for d in (-1, 1):
        j = ix
        while 0 < j < n - 1 and row[j] > half:
            j += d
        a, b = row[j], row[j - d]
        t = 0.0 if b == a else (half - a) / (b - a)
        xs.append(ax[j] + t * (ax[j - d] - ax[j]))
    r2 = (ax - ax[ix])[None, :] ** 2 + (ax - ax[iy])[:, None] ** 2
    tot = float(I.sum())
    ee = {'ee%d' % int(r * 1e6): float(I[r2 <= r * r].sum() / tot * 100.0)
          for r in (3e-6, 6e-6, 12e-6)}
    return dict(peak=pk, fwhm_um=abs(xs[1] - xs[0]) * 1e6, **ee)


def main():
    out = sys.argv[2]
    N = int(sys.argv[3])
    import _d121_common as D
    assert os.path.abspath(la.__file__).lower().startswith(
        _TREE.lower() + os.sep), la.__file__
    pre, post, gap_to_doe, _period = D.geometry()
    groups = list(pre)
    if post:
        post = [dict(post[0], gap_before=post[0]['gap_before'] + gap_to_doe)] \
            + list(post[1:])
        groups += post
    zR = np.pi * W0 * W0 / LAM
    z1 = 2e-3
    w_z1 = W0 * np.sqrt(1.0 + (z1 / zR) ** 2)
    R1 = z1 * (1.0 + (zR / z1) ** 2)
    dx0 = 1.0e-6 * 2048.0 / N
    x = (np.arange(N) - N // 2) * dx0
    env = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                 / (w_z1 * w_z1)).astype(np.complex128)
    tkw = dict(on_undersample='silent', on_noncollimated='silent')
    rows = []
    for dz_um in (-300.0, -150.0, -60.0, 0.0, 60.0, 150.0, 300.0):
        z = TRAILING + dz_um * 1e-6
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = CA.propagate_traced_carrier_chain(
                env, groups, LAM, dx0, r_in=R1, ray_subsample=4, n_workers=1,
                traced_kwargs=tkw, final_leg='paraxial', final_distance=z,
                focus_readout=dict(dx_out=DXO, N_out=NOUT),
                transport='collins')
        st = res.stages[-1]
        r = dict(dz_um=dz_um, final_distance=z,
                 route=st.get('readout_route'),
                 k1=st.get('readout_route_k1'),
                 n_kelly=len([q for q in w
                              if 'under-sampled' in str(q.message)]))
        r.update(_metrics(res.field, DXO))
        rows.append(r)
        print('[scan] dz=%+8.1f um  FWHM %8.4f  EE3 %7.3f  EE6 %7.3f  '
              'peak %.4g  route %s  K1 %r'
              % (dz_um, r['fwhm_um'], r['ee3'], r['ee6'], r['peak'],
                 r['route'], r['k1']))
    vlib.write_json({'build': vlib.build_tag(), 'tree': _TREE, 'N': N,
                     'rows': rows}, out)


if __name__ == '__main__':
    main()
