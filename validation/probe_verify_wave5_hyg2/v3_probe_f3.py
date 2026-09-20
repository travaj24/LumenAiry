"""v3 -- item (4): VERIFY-B4 F3's OWN fixture, rebuilt, and the reconciliation.

F3's fixture, read off ``tests/unit/test_audit2609_b4_collins_transport.py``
lines 127-141 and 1338-1372 (which is the pin VERIFY-B4 F3 says it wrote):
``lambda = 1.064 um``, ``N = 1024``, ``dx = 4 um``, ``w = 0.30 mm``,
``R = -40 mm``, the leg ``z = -R + dz`` (dz PAST the geometric focus), the
readout lattice ``dx_out = w0/8`` with ``w0 = lambda|R|/(pi w)`` and
``N_out = 128``, compared ABSOLUTELY against the analytic Gaussian.

Also measures the AUTHOR's fixture walked to ITS carrier's geometric focus,
which its published ladder never reaches.
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

from lumenairy.propagators.carrier import _collins_transport  # noqa: E402

EPS = float(np.finfo(np.float64).eps)


def grid(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def abcd_gauss(xo, w_in, r_beam, z, lam):
    k = 2.0 * np.pi / lam
    q = 1.0 / (1.0 / r_beam + 1j * lam / (np.pi * w_in ** 2))
    q2 = q + z
    r2 = xo[None, :] ** 2 + xo[:, None] ** 2
    wz = float(np.sqrt(lam / (np.pi * (1.0 / q2).imag)))
    return (np.exp(1j * k * z) / (1.0 + z / q)
            * np.exp(1j * k * r2 / (2.0 * q2))), wz


def rel(E, T):
    E, T = np.asarray(E), np.asarray(T)
    return float(np.linalg.norm(E - T) / np.linalg.norm(T))


def piston_free(E, T):
    E, T = np.asarray(E), np.asarray(T)
    ov = np.vdot(T, E)
    return rel(E / (ov / abs(ov)) if abs(ov) > 0 else E, T)


def numpy_only(w_env, z_eff, lam, n, dx_in):
    """The departure with numpy alone: ||S(e^{i phi} - 1)||/||S||."""
    k = 2.0 * np.pi / lam
    g = grid(n, dx_in)
    env = np.exp(-(g[None, :] ** 2 + g[:, None] ** 2) / w_env ** 2)
    S = np.fft.fft2(env)
    qx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx_in)
    q2 = qx[None, :] ** 2 + qx[:, None] ** 2
    phi = z_eff * (np.sqrt(np.maximum(k * k - q2, 0.0)) - k + q2 / (2.0 * k))
    th0 = lam / (np.pi * w_env)
    return (float(np.linalg.norm(S * (np.exp(1j * phi) - 1.0))
                  / np.linalg.norm(S)),
            float(np.sqrt(1.5) * k * abs(z_eff) * th0 ** 4 / 8.0), float(th0))


out = {'build': vlib.build_tag(), 'numpy': np.__version__,
       'python': sys.version.split()[0]}

# ======================================================================
# A.  VERIFY-B4 F3's fixture, rebuilt
# ======================================================================
WL, N, DX, W, R = 1.064e-6, 1024, 4.0e-6, 0.30e-3, -40.0e-3
K = 2.0 * np.pi / WL
ENV = np.exp(-((grid(N, DX)[None, :] ** 2 + grid(N, DX)[:, None] ** 2)
               / W ** 2)).astype(np.complex128)
W0 = WL * abs(R) / (np.pi * W)
DXO, NOUT = W0 / 8.0, 128
out['f3_fixture'] = {
    'lam': WL, 'N': N, 'dx': DX, 'w': W, 'R': R, 'k': K, 'w0_focal': W0,
    'dx_out': DXO, 'N_out': NOUT, 'window_out': NOUT * DXO,
    'window_out_over_w0': NOUT * DXO / W0,
    'theta_beam': W / abs(R), 'theta_env_analytic': WL / (np.pi * W),
    'grid_radii': (N * DX / 2.0) / W,
    'edge_truncation': float(np.exp(-((N * DX / 2.0) / W) ** 2)),
}


def f3_row(dz, sign=+1, n=N, dx=DX, dxo=DXO, nout=NOUT, env=None):
    """``sign=+1`` is F3's own spelling ``z = -R + dz`` (PAST the focus)."""
    z = -R + sign * dz
    env = ENV if env is None else env
    row = {'dz': dz, 'sign': sign, 'z': z, 'N': n, 'dx': dx, 'dx_out': dxo,
           'N_out': nout}
    T, wz = abcd_gauss(grid(nout, dxo), W, R, z, WL)
    row['w_at_plane'] = wz
    for kern in ('fresnel', 'auto', 'exact'):
        st = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                E = np.asarray(_collins_transport(
                    env, R, z, WL, dx, dx, dx_out=dxo, dy_out=dxo,
                    N_out_x=nout, N_out_y=nout, R_ref=float('inf'),
                    gap_kernel=kern, on_collins_sampling='ignore',
                    stats_out=st))
            row[kern] = rel(E, T)
            row[kern + '_piston_free'] = piston_free(E, T)
            if kern == 'auto':
                A, B = float(st['abcd'][0]), float(st['abcd'][1])
                row.update({'A': A, 'B': B, 'z_eff': B / A,
                            'k4': float(st['k4']), 'kernel': st['kernel'],
                            'K1': max(st['k1']), 'K2': max(st['k2']),
                            'K3': max(st['k3']),
                            'period': max(st['period']),
                            'theta_measured': float(max(st['theta_x'],
                                                        st['theta_y']))})
        except Exception as exc:                                # noqa: BLE001
            row[kern] = 'RAISED %s: %s' % (type(exc).__name__, str(exc)[:160])
    if 'z_eff' in row:
        pred, closed, th0 = numpy_only(W, row['z_eff'], WL, n, dx)
        row['numpy_only_pred'] = pred
        row['law_sqrt1p5'] = closed
        row['theta_analytic'] = th0
        row['beam_quartic'] = float(K * abs(row['z_eff'])
                                    * (W / abs(R)) ** 4 / 8.0)
        row['measured_angle_quartic'] = float(
            K * abs(row['z_eff']) * row['theta_measured'] ** 4 / 8.0)
    return row


rows = []
for dz in (1e-6, 1e-5, 1e-4, 1e-3, 5e-3):
    rows.append(f3_row(dz, +1))
for dz in (1e-6, 1e-5, 1e-4):
    rows.append(f3_row(dz, -1))
out['f3_rows'] = rows

# N-independence on the same physical extent (F3's own control).
ninv = []
for n in (512, 1024, 2048):
    dx = (N * DX) / n
    env = np.exp(-((grid(n, dx)[None, :] ** 2 + grid(n, dx)[:, None] ** 2)
                   / W ** 2)).astype(np.complex128)
    ninv.append(f3_row(1e-6, +1, n=n, dx=dx, env=env))
out['f3_N_invariance'] = ninv

# Window / lattice variations -- is F3's reading a clipped or wrapped window?
wins = []
for nout, dxo in ((128, DXO), (512, DXO), (128, 4.0 * DXO),
                  (512, 6.0 * W0 / 512)):
    wins.append(f3_row(1e-6, +1, dxo=dxo, nout=nout))
out['f3_window_variations'] = wins

# ======================================================================
# B.  The AUTHOR's fixture, walked to ITS carrier's geometric focus
# ======================================================================
AW0, ATH, AF, AN, ADX = 15.915e-6, 20.0e-3, 20.0e-3, 512, 8e-6
ALAM = float(np.pi * AW0 * ATH)
AZR = float(np.pi * AW0 ** 2 / ALAM)
AK = 2.0 * np.pi / ALAM
q_in = complex(-AF, -AZR)
AR = 1.0 / float(np.real(1.0 / q_in))
AW = float(np.sqrt(ALAM / (np.pi * np.imag(1.0 / q_in))))
ag = grid(AN, ADX)
AENV = np.exp(-((ag[None, :] ** 2 + ag[:, None] ** 2) / AW ** 2)).astype(
    np.complex128)
out['author_fixture'] = {
    'lam': ALAM, 'R_in': AR, 'w_in': AW, 'f_waist': AF,
    'carrier_focus_at': abs(AR),
    'waist_is_this_far_short_of_the_carrier_focus': abs(AR) - AF,
    'theta_env_analytic': ALAM / (np.pi * AW)}


def author_row(z, label):
    q_out = q_in + z
    w_out = float(np.sqrt(ALAM / (np.pi * np.imag(1.0 / q_out))))
    re = float(np.real(1.0 / q_out))
    R_out = float('inf') if re == 0.0 else 1.0 / re
    dxo = float(min(6.0 * w_out / AN, w_out / 8.0,
                    (ALAM * abs(R_out) / (4.0 * w_out))
                    if np.isfinite(R_out) else w_out / 8.0))
    xo = grid(AN, dxo)
    T, _ = abcd_gauss(xo, AW, AR, z, ALAM)
    row = {'label': label, 'z': z, 'dist_to_carrier_focus': abs(AR) - z,
           'dist_to_waist': AF - z, 'w_out': w_out, 'dx_out': dxo}
    for kern in ('fresnel', 'auto', 'exact'):
        st = {}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                E = np.asarray(_collins_transport(
                    AENV, AR, z, ALAM, ADX, ADX, dx_out=dxo, dy_out=dxo,
                    N_out_x=AN, N_out_y=AN, R_ref=float('inf'),
                    gap_kernel=kern, on_collins_sampling='ignore',
                    stats_out=st))
            row[kern] = rel(E, T)
            if kern == 'auto':
                A, B = float(st['abcd'][0]), float(st['abcd'][1])
                row.update({'A': A, 'B': B,
                            'z_eff': (B / A) if A != 0 else float('inf'),
                            'k4': float(st['k4']), 'kernel': st['kernel'],
                            'K1': max(st['k1']), 'K3': max(st['k3']),
                            'theta_measured': float(max(st['theta_x'],
                                                        st['theta_y']))})
        except Exception as exc:                                # noqa: BLE001
            row[kern] = 'RAISED %s: %s' % (type(exc).__name__, str(exc)[:160])
    if np.isfinite(row.get('z_eff', np.inf)):
        pred, closed, th0 = numpy_only(AW, row['z_eff'], ALAM, AN, ADX)
        row['numpy_only_pred'] = pred
        row['law_sqrt1p5'] = closed
    return row


arows = [author_row(AF - 1e-6, 'the published rung: 1 um short of the WAIST')]
for e in (1e-4, 1e-5, 1e-6, 1e-7):
    arows.append(author_row(abs(AR) - e,
                            '%.0e m short of the CARRIER focus' % e))
arows.append(author_row(abs(AR), 'exactly ON the carrier focus (A = 0)'))
out['author_walked_to_carrier_focus'] = arows

vlib.write_json(out, sys.argv[1])
