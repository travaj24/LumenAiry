"""VERIFY-WP-C3 ROUND 2 -- the synthetic stand-in for the p8 capstone's
near-focus leg, sized so a decision test can run it in under a second.

Same shape as the composed chain's final leg: a flat-resolving Collins leg at
A ~ 0.0067 whose chirp-Z IS representable (K1 = K3 << 1) and whose output
pitch is therefore set by the FLOOR ``2 r_out / N``, not by the co-moving
contraction.  Prints the three readings the decision rests on.
"""
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(os.environ['VC3_TREE'])
sys.path.insert(0, TREE)
import lumenairy as la                                        # noqa: E402

assert os.path.abspath(la.__file__).startswith(TREE)

WL, N, DX, W, A = 1.31e-6, 256, 1e-6, 90e-6, 0.0067
R = -40e-3
Z = -R * (1.0 - A)


def ee(I, dx, win, q=0.8):
    n = I.shape[0]
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    jp, ip = np.unravel_index(np.argmax(I), I.shape)
    rr = np.sqrt((X - x[ip]) ** 2 + (Y - x[jp]) ** 2)
    m = rr <= win
    Iw, Rw, Pw = I[m], rr[m], I[m].sum()
    rb = np.linspace(0, win, 600)
    cum = np.array([Iw[Rw <= t].sum() for t in rb]) / Pw
    return float(np.interp(q, cum, rb))


x = (np.arange(N) - N / 2) * DX
X, Y = np.meshgrid(x, x, indexing='xy')
env = np.exp(-(X ** 2 + Y ** 2) / W ** 2).astype(np.complex128)

with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    cs = la.propagate_carrier_referenced(env, R, Z, WL, DX,
                                         transport='sziklas')
    cc = la.propagate_carrier_referenced(env, R, Z, WL, DX,
                                         transport='collins')
dxs = float(cs.dx if not isinstance(cs.dx, tuple) else cs.dx[0])
dxc = float(cc.dx if not isinstance(cc.dx, tuple) else cc.dx[0])
cf = la.propagate_carrier_referenced(env, R, Z, WL, DX, transport='collins',
                                     dx_out=dxs, on_collins_sampling='ignore')
dxf = float(cf.dx if not isinstance(cf.dx, tuple) else cf.dx[0])

win = min(0.45 * N * dxs, 0.45 * N * dxc)
es = ee(np.abs(np.asarray(cs.env)) ** 2, dxs, win)
ec = ee(np.abs(np.asarray(cc.env)) ** 2, dxc, win)
ef = ee(np.abs(np.asarray(cf.env)) ** 2, dxf, win)

out = {'tree': TREE, 'lumenairy': la.__file__,
       'python': sys.version.split()[0], 'numpy': np.__version__,
       'fixture': dict(N=N, dx=DX, w=W, R=R, z=Z, A=A, wavelength=WL),
       'R_sziklas': repr(cs.R), 'R_collins': repr(cc.R),
       'R_collins_named': repr(cf.R),
       'dx_sziklas_um': dxs * 1e6, 'dx_collins_um': dxc * 1e6,
       'dx_collins_named_um': dxf * 1e6,
       'win_um': win * 1e6,
       'EE80_sziklas_um': es * 1e6, 'EE80_collins_um': ec * 1e6,
       'EE80_collins_named_um': ef * 1e6,
       'samples_in_EE80_collins': ec / dxc,
       'samples_in_EE80_sziklas': es / dxs,
       'err_on_returned_lattice': abs(ec / es - 1.0),
       'err_on_common_lattice': abs(ef / es - 1.0),
       'warnings': [str(w.message)[:110] for w in rec]}
out['separation'] = (out['err_on_returned_lattice']
                     / out['err_on_common_lattice'])
tag = os.environ.get('VC3_TAG', 'x')
p = os.path.join(os.environ['VC3_OUT'], f'vr2_p8_decision_{tag}.json')
with open(p, 'w', encoding='utf-8') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
