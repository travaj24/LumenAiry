"""VERIFY-WP-C3 CLAIM 2 -- the readout resolution on ``_collins_readout_k1 <= 1``.

    python probe_claim2.py <tree> <out.json>

(a) is K1 = 1 a CLIFF or a SLOPE?  A fixture where K1 is DIALLED continuously:
    a COLLIMATED Gaussian (``R = inf``, so ``A = 1`` and ``C = 0``), read out
    one step at a distance ``z``.  Then
        K1 = 2 dx (r/z + theta)/lambda,
    which is monotone in ``1/z`` and nothing else in the call moves with it --
    K2 = 2 dx_out theta/lambda is 0.0104 at every rung (``C = 0``, ``D = 1``)
    and K3 = (N_out dx_out)/(lambda z/dx) stays under 1 at every rung.  So the
    rung IS K1.  Truth: the analytic Gaussian of :mod:`vclib`, validated
    separately.

(b) does ``readout_route_reason`` name the branch actually taken?  Checked by
    BIT IDENTITY against ``transport='sziklas'`` rather than by reading the
    key: route 'sziklas' MUST be bit-identical, route 'collins' MUST NOT be.

(d) ``z == 0`` returns ``inf`` and therefore routes to Sziklas; and
    ``final_distance = 0`` with a readout works on the default.
"""
from __future__ import annotations

import hashlib
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                     # noqa: E402
import vclib as V                                      # noqa: E402

V.anchor(_TREE)

import lumenairy.propagators.carrier as CA             # noqa: E402

out = {}

# ===========================================================================
# (a)  THE K1 DIAL
# ===========================================================================
LAM = 1.064e-6
W = 0.30e-3
N = 512
DX = 6.0 * W / N
DXO, NOUT = 2.0e-6, 96

x = V.axis(N, DX)
env = V.gauss_env(x, x, W)
r_x, r_y, th_x, th_y = CA._collins_input_box(env, DX, DX, LAM,
                                             CA._COLLINS_TAIL_FRAC)
# K1(z) = 2 dx (r/z + theta)/lambda  ->  z(K1)
def z_for(k1):
    return 2.0 * DX * r_x / (LAM * k1 - 2.0 * DX * th_x)


RUNGS = [0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0,
         1.01, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
xo = V.axis(NOUT, DXO)
rows = []
for k1 in RUNGS:
    z = z_for(k1)
    got_k1 = CA._collins_readout_k1(env, np.inf, z, LAM, DX, DX)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        try:
            E = np.asarray(CA._collins_focus_readout(
                env, np.inf, z, LAM, DX, DX, dx_out=DXO, N_out=NOUT,
                on_collins_sampling='warn', on_replica='error'))
            exc = None
        except Exception as e:                          # noqa: BLE001
            E, exc = None, f'{type(e).__name__}: {e}'
    ref = V.gauss_field_at(xo, xo, W, np.inf, LAM, z)
    row = {'k1_target': k1, 'z': z, 'k1_measured': got_k1,
           'period_um': LAM * z / DX * 1e6,
           'window_um': NOUT * DXO * 1e6,
           'k3': NOUT * DXO / (LAM * z / DX),
           'k2': 2.0 * DXO * th_x / LAM,
           'n_kelly': sum('under-sampled' in str(w.message) for w in rec),
           'raised': exc}
    if E is not None:
        c = NOUT // 2
        row.update({
            'rel_l2': V.rel_l2(E, ref),
            'rel_l2_abs': V.rel_l2(np.abs(E), np.abs(ref)),
            'on_axis_got': float(abs(E[c, c]) ** 2),
            'on_axis_ref': float(abs(ref[c, c]) ** 2),
            'on_axis_rel_err': float(abs(abs(E[c, c]) ** 2
                                         - abs(ref[c, c]) ** 2)
                                     / abs(ref[c, c]) ** 2),
            'centre_ratio_abs': float(abs(E[c, c] / ref[c, c])),
            'centre_ratio_arg': float(np.angle(E[c, c] / ref[c, c])),
        })
    rows.append(row)
out['a_k1_dial'] = {
    'fixture': {'lam': LAM, 'w': W, 'N': N, 'dx': DX, 'dx_out': DXO,
                'N_out': NOUT, 'R': 'inf',
                'r_support_mm': r_x * 1e3, 'theta_mrad': th_x * 1e3},
    'rows': rows}

# ===========================================================================
# (b)  does the reason name the branch that ran?
# ===========================================================================
WL2 = 1.31e-6
TKW = dict(on_undersample='silent', on_noncollimated='silent')


def singlet(R1, R2, d, glass, ap, name):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def fixture(n=256):
    dx = 60e-6 * 256 / n
    g = V.axis(n, dx)
    e = np.exp(-((g[None, :] ** 2 + g[:, None] ** 2) / 4.5e-3 ** 2)
               ).astype(np.complex128)
    p = singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 14e-3, 'p')
    return e, dx, 60e-3, [{'prescription': p, 'gap_before': 20e-3},
                          {'prescription': p, 'gap_before': 10e-3}]


def run(n, fd, fr, transport=None, **kw):
    e, dx, ri, gr = fixture(n)
    kws = dict(final_leg='paraxial', traced_kwargs=TKW, ray_subsample=16,
               n_workers=1, final_distance=fd)
    if fr is not None:
        kws['focus_readout'] = dict(fr)
    if transport is not None:
        kws['transport'] = transport
    kws.update(kw)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        r = CA.propagate_traced_carrier_chain(e, gr, WL2, dx, r_in=ri, **kws)
    return r, [str(w.message) for w in rec]


# find a chain whose readout IS representable (K1 <= 1): sweep final_distance
scan = []
for n in (512, 1024):
    for fd in (8e-3, 20e-3, 30e-3, 40e-3, 44e-3, 46e-3, 46.2e-3, 50e-3,
               60e-3, 80e-3):
        try:
            r, _w = run(n, fd, dict(dx_out=0.5e-6, N_out=64, on_replica='ignore'))
            st = r.stages[-1]
            scan.append({'N': n, 'fd': fd,
                         'k1': st.get('readout_route_k1'),
                         'route': st.get('readout_route'),
                         'reason': st.get('readout_route_reason')})
        except Exception as e:                          # noqa: BLE001
            scan.append({'N': n, 'fd': fd, 'raised': f'{type(e).__name__}: {e}'})
out['b_scan'] = scan

_repr = [s for s in scan if s.get('route') == 'collins']
cases = [
    ('k1_route_sziklas', dict(n=256, fd=8e-3,
                              fr=dict(dx_out=0.5e-6, N_out=64))),
    ('stop_plane_standoff', dict(n=256, fd=8e-3,
                                 fr=dict(dx_out=0.5e-6, N_out=64,
                                         standoff=2e-3))),
    ('stop_plane_containment', dict(
        n=256, fd=8e-3, fr=dict(dx_out=0.5e-6, N_out=64,
                                on_focus_containment='warn'))),
]
if _repr:
    b = _repr[0]
    cases.append(('representable_route_collins',
                  dict(n=b['N'], fd=b['fd'],
                       fr=dict(dx_out=0.5e-6, N_out=64,
                               on_replica='ignore'))))
    if len(_repr) > 1:
        b2 = _repr[-1]
        cases.append(('representable_route_collins_2',
                      dict(n=b2['N'], fd=b2['fd'],
                           fr=dict(dx_out=0.5e-6, N_out=64,
                                   on_replica='ignore'))))

brows = []
for tag, kw in cases:
    try:
        a, _wa = run(kw['n'], kw['fd'], kw['fr'])                 # default
        bb, _wb = run(kw['n'], kw['fd'], kw['fr'], transport='sziklas')
    except Exception as e:                              # noqa: BLE001
        brows.append({'case': tag, 'raised': f'{type(e).__name__}: {e}'})
        continue
    st = a.stages[-1]
    same = bool(np.array_equal(np.asarray(a.field), np.asarray(bb.field)))
    route = st.get('readout_route')
    brows.append({
        'case': tag, 'N': kw['n'], 'final_distance': kw['fd'],
        'readout_route': route,
        'readout_route_k1': st.get('readout_route_k1'),
        'readout_route_reason': st.get('readout_route_reason'),
        'bit_identical_to_sziklas': same,
        'rel_l2_vs_sziklas': V.rel_l2(np.asarray(a.field),
                                      np.asarray(bb.field)),
        'CONSISTENT': bool((route == 'sziklas') == same),
        'keys_on_sziklas_side': [k for k in
                                 ('readout_route', 'readout_route_k1',
                                  'readout_route_reason')
                                 if k in bb.stages[-1]],
    })
out['b_route_vs_bits'] = brows

# ===========================================================================
# (d)  z == 0
# ===========================================================================
e, dx, ri, gr = fixture(256)
d = {'readout_k1_at_z0': CA._collins_readout_k1(e, 60e-3, 0.0, WL2, dx, dx)}
try:
    CA._collins_focus_readout(e, 60e-3, 0.0, WL2, dx, dx,
                              dx_out=0.5e-6, N_out=64)
    d['one_step_at_z0'] = 'RETURNED (expected a refusal)'
except Exception as ex:                                 # noqa: BLE001
    d['one_step_at_z0'] = f'{type(ex).__name__}: {str(ex)[:120]}'
for tr in (None, 'sziklas'):
    try:
        r, _w = run(256, 0.0, dict(dx_out=0.5e-6, N_out=64), transport=tr)
        st = r.stages[-1]
        d[f'chain_fd0_{tr or "default"}'] = {
            'ok': True, 'route': st.get('readout_route'),
            'k1': st.get('readout_route_k1'),
            'reason': st.get('readout_route_reason'),
            'peak': float((np.abs(np.asarray(r.field)) ** 2).max())}
    except Exception as ex:                             # noqa: BLE001
        d[f'chain_fd0_{tr or "default"}'] = {
            'ok': False, 'raised': f'{type(ex).__name__}: {str(ex)[:160]}'}
try:
    a, _ = run(256, 0.0, dict(dx_out=0.5e-6, N_out=64))
    b2, _ = run(256, 0.0, dict(dx_out=0.5e-6, N_out=64), transport='sziklas')
    d['fd0_bit_identical'] = bool(np.array_equal(np.asarray(a.field),
                                                 np.asarray(b2.field)))
except Exception as ex:                                 # noqa: BLE001
    d['fd0_bit_identical'] = f'{type(ex).__name__}: {ex}'
out['d_zero'] = d

V.write_json(sys.argv[2], out)
print('--- a: the K1 dial ---')
for r in out['a_k1_dial']['rows']:
    print(f"  K1={r['k1_target']:<5} meas={r.get('k1_measured'):9.5f} "
          f"z={r['z']*1e3:8.4f}mm relL2={r.get('rel_l2')} "
          f"onax_err={r.get('on_axis_rel_err')} kelly={r['n_kelly']} "
          f"raised={r['raised']}")
print('--- b: route vs bits ---')
for r in out['b_route_vs_bits']:
    print(' ', r)
print('--- b scan (representable hunt) ---')
for r in out['b_scan']:
    print(' ', r)
print('--- d ---', out['d_zero'])
