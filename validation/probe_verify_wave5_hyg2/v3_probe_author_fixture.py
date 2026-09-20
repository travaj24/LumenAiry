"""v3 -- INDEPENDENT re-measurement of H2-3 on the AUTHOR's fixture.

Items (1) bookkeeping, (2) the 3x2x9 table, (5) the k4 gate table.
Nothing here is read from the author's test file; the fixture is re-derived
from ``w0``, ``theta`` and ``f`` and the oracle is written from the q form.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vlib  # noqa: E402

TREE = os.path.join(os.getcwd(), 'lumenairy')
L = vlib.anchor(TREE)

from lumenairy.propagators.carrier import (  # noqa: E402
    _collins_transport, carrier_referenced_reconstruct,
    propagate_carrier_referenced)
import lumenairy.propagators.carrier as C  # noqa: E402

W0 = 15.915e-6
THETA = 20.0e-3
LAM = float(np.pi * W0 * THETA)
ZR = float(np.pi * W0 ** 2 / LAM)
F = 20.0e-3
K = 2.0 * np.pi / LAM
N = 512
DX = 8e-6
EPS = float(np.finfo(np.float64).eps)


def w_of_q(q):
    return float(np.sqrt(LAM / (np.pi * np.imag(1.0 / q))))


def R_of_q(q):
    re = float(np.real(1.0 / q))
    return float('inf') if re == 0.0 else 1.0 / re


def axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def q_field(xo, yo, q_in, z):
    q2 = q_in + z
    xx, yy = np.meshgrid(xo, yo, indexing='xy')
    r2 = xx ** 2 + yy ** 2
    return (np.exp(1j * K * z) / (1.0 + z / q_in)
            * np.exp(1j * K * r2 / (2.0 * q2)))


def pitch_for(w, R, n=N):
    p_win = 6.0 * w / float(n)
    p_amp = w / 8.0
    p_curv = (LAM * abs(R) / (4.0 * w)) if np.isfinite(R) else p_amp
    return float(min(p_win, p_amp, p_curv))


def floor_bar(z, n_out, dx_out, w_out, decades=10.0):
    piston = EPS * K * abs(float(z))
    edge = float(n_out) * float(dx_out) / 2.0
    trunc = float(np.exp(-(edge / w_out) ** 2)) if w_out > 0 else 1.0
    return decades * (piston + trunc), piston, trunc


def rel(got, ref):
    g = np.asarray(got)
    r = np.asarray(ref)
    return float(np.linalg.norm(g - r) / np.linalg.norm(r))


out = {'build': vlib.build_tag(), 'lumenairy': L.__version__,
       'numpy': np.__version__, 'python': sys.version.split()[0]}

q_in = complex(-F, -ZR)
R_IN = R_of_q(q_in)
W_IN = w_of_q(q_in)
x = axis(N, DX)
ENV = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2) / W_IN ** 2).astype(
    np.complex128)

out['fixture'] = {
    'lam': LAM, 'zR': ZR, 'f': F, 'k': K, 'N': N, 'dx': DX,
    'R_in': R_IN, 'w_in': W_IN, 'theta_beam': THETA,
    'theta_env_analytic': float(LAM / (np.pi * W_IN)),
    'R_in_closed_form': float(-(F ** 2 + ZR ** 2) / F),
    'w_in_closed_form': float(np.sqrt(LAM * (F ** 2 + ZR ** 2)
                                      / (np.pi * ZR))),
}

# ---------------------------------------------------------------- item (1)
book = {}

q_a = complex(0.0, -ZR)
w_a = w_of_q(q_a)
z_c = 1.5 * ZR
w_out_c = w_of_q(q_a + z_c)
dx_c = 10.0 * w_out_c / N
xc = axis(N, dx_c)
env_c = np.exp(-(xc[None, :] ** 2 + xc[:, None] ** 2) / w_a ** 2).astype(
    np.complex128)
ref_c = q_field(xc, xc, q_a, z_c)
bar_c, pist_c, trunc_c = floor_bar(z_c, N, dx_c, w_out_c)
for kern in ('fresnel', 'auto', 'exact'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = propagate_carrier_referenced(env_c, float('inf'), z_c, LAM, dx_c,
                                         gap_kernel=kern)
    fld = carrier_referenced_reconstruct(g.env, g.R, LAM, g.dx)
    book['collimated_%s_reconstruct' % kern] = rel(fld, ref_c)
book['collimated_bar'] = bar_c
book['collimated_bar_parts'] = {'piston': pist_c, 'trunc': trunc_c,
                                'dx_out': float(dx_c), 'w_out': w_out_c,
                                'z': z_c, 'w_a': w_a}
book['collimated_quartic_beam'] = float(
    K * abs(z_c) * (LAM / (np.pi * w_a)) ** 4 / 8.0)

d = 5e-3
z = F - d
q_out = q_in + z
w_out = w_of_q(q_out)
dx_out = pitch_for(w_out, R_of_q(q_out))
bar5, pist5, trunc5 = floor_bar(z, N, dx_out, w_out)
common = dict(wavelength=LAM, dx=DX, transport='collins',
              gap_kernel='fresnel', dx_out=dx_out,
              on_collins_sampling='ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    a = propagate_carrier_referenced(ENV, R_IN, z,
                                     **dict(common, carrier_out=float('inf')))
    b = propagate_carrier_referenced(ENV, R_IN, z, **common)
xo = axis(N, float(a.dx))
ref5 = q_field(xo, xo, q_in, z)
b_fld = carrier_referenced_reconstruct(b.env, b.R, LAM, b.dx)
book['converging_5mm_carrier_out_inf'] = rel(a.env, ref5)
book['converging_5mm_reconstruct'] = rel(b_fld, ref5)
book['converging_5mm_two_spellings_agree_to'] = abs(
    book['converging_5mm_carrier_out_inf']
    - book['converging_5mm_reconstruct'])
book['converging_5mm_a_R_is_inf'] = bool(np.isinf(float(a.R)))
book['converging_5mm_b_R'] = float(b.R)
book['converging_5mm_dx_out'] = float(a.dx)
book['converging_5mm_bar'] = bar5
book['converging_5mm_bar_parts'] = {'piston': pist5, 'trunc': trunc5,
                                    'w_out': w_out, 'z': z}
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sz = propagate_carrier_referenced(ENV, R_IN, z, LAM, DX,
                                      gap_kernel='fresnel')
sz_fld = carrier_referenced_reconstruct(sz.env, sz.R, LAM, sz.dx)
xs = axis(N, float(sz.dx))
book['converging_5mm_sziklas_reconstruct'] = rel(
    sz_fld, q_field(xs, xs, q_in, z))
bar_sz, pist_sz, trunc_sz = floor_bar(z, N, float(sz.dx), w_out)
book['converging_5mm_sziklas_bar'] = bar_sz
book['converging_5mm_sziklas_bar_parts'] = {'piston': pist_sz,
                                            'trunc': trunc_sz,
                                            'dx_out': float(sz.dx)}
book['converging_5mm_double_carrier'] = rel(
    carrier_referenced_reconstruct(a.env, R_IN + z, LAM, float(a.dx)), ref5)
out['bookkeeping'] = book

# ---------------------------------------------------------------- item (2)
LADDER = (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3)
table = {}
for dd in LADDER:
    zz = F - dd
    qq = q_in + zz
    ww = w_of_q(qq)
    dxo = pitch_for(ww, R_of_q(qq))
    key = '%g' % dd
    row = {'z': zz, 'w_out': ww, 'dx_out_collins': dxo}
    for kern in ('auto', 'fresnel', 'exact'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                g = propagate_carrier_referenced(
                    ENV, R_IN, zz, wavelength=LAM, dx=DX, transport='collins',
                    gap_kernel=kern, dx_out=dxo, carrier_out=float('inf'),
                    on_collins_sampling='ignore')
            xg = axis(N, float(g.dx))
            row['collins_' + kern] = rel(g.env, q_field(xg, xg, q_in, zz))
        except Exception as exc:                                # noqa: BLE001
            row['collins_' + kern] = 'RAISED %s: %s' % (
                type(exc).__name__, str(exc)[:120])
    row['collins_bar'] = floor_bar(zz, N, dxo, ww)[0]
    for kern in ('auto', 'fresnel', 'exact'):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                g = propagate_carrier_referenced(ENV, R_IN, zz, LAM, DX,
                                                 gap_kernel=kern)
            fl = carrier_referenced_reconstruct(g.env, g.R, LAM, g.dx)
            xg = axis(N, float(g.dx))
            row['sziklas_' + kern] = rel(fl, q_field(xg, xg, q_in, zz))
            row['sziklas_dx_out'] = float(g.dx)
            row['sziklas_R_out'] = float(g.R)
        except Exception as exc:                                # noqa: BLE001
            row['sziklas_' + kern] = 'RAISED %s: %s' % (
                type(exc).__name__, str(exc)[:120])
    m = abs((R_IN + zz) / R_IN)
    row['m'] = m
    row['m_dx'] = m * DX
    row['sziklas_pitch_over_m_dx'] = (float(row.get('sziklas_dx_out', np.nan))
                                      / (m * DX))
    A_env = 1.0 + zz / R_IN
    row['beam_quartic'] = float(K * abs(zz / A_env) * THETA ** 4 / 8.0)
    row['near_focus_needs_bridge'] = bool(C._near_focus_needs_bridge(
        ENV, R_IN, R_IN + zz, LAM, DX, DX))
    table[key] = row
out['table'] = table

# which sziklas branch actually runs
calls = {'focus_crossing': 0, 'step_fast': 0}
_fc, _sf = C._propagate_carrier_focus_crossing, C._carrier_step_fast


def _wrap_fc(*a, **kw):
    calls['focus_crossing'] += 1
    return _fc(*a, **kw)


def _wrap_sf(*a, **kw):
    calls['step_fast'] += 1
    return _sf(*a, **kw)


C._propagate_carrier_focus_crossing = _wrap_fc
C._carrier_step_fast = _wrap_sf
branch = {}
for dd in LADDER:
    calls['focus_crossing'] = calls['step_fast'] = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        g = propagate_carrier_referenced(ENV, R_IN, F - dd, LAM, DX,
                                         gap_kernel='fresnel')
    branch['%g' % dd] = {'focus_crossing': calls['focus_crossing'],
                         'step_fast': calls['step_fast'],
                         'dx_out': float(g.dx), 'R_out': float(g.R)}
C._propagate_carrier_focus_crossing = _fc
C._carrier_step_fast = _sf
out['sziklas_branch_taken'] = branch

# ---------------------------------------------------------------- item (5)
gate = {}
for dd in LADDER:
    zz = F - dd
    qq = q_in + zz
    ww = w_of_q(qq)
    dxo = pitch_for(ww, R_of_q(qq))
    st = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _collins_transport(ENV, R_IN, zz, LAM, DX, DX, dx_out=dxo, dy_out=dxo,
                           N_out_x=N, N_out_y=N, R_ref=float('inf'),
                           gap_kernel='auto', on_collins_sampling='ignore',
                           stats_out=st)
    A, B = float(st['abcd'][0]), float(st['abcd'][1])
    gate['%g' % dd] = {
        'A': A, 'B': B, 'z_eff': abs(B / A), 'k4': float(st['k4']),
        'kernel': st['kernel'], 'K1': max(st['k1']), 'K2': max(st['k2']),
        'K3': max(st['k3']),
        'theta_x': float(st['theta_x']), 'theta_y': float(st['theta_y']),
    }
out['gate'] = gate

vlib.write_json(out, sys.argv[1])
