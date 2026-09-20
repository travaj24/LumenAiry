"""v3 -- item (3): the derived law, on MY OWN fixture and on the author's.

Usage: v3_probe_law.py <out.json> <author|mine>

Two one-variable sweeps (z_eff at fixed envelope; envelope angle at fixed
geometry), the fitted constant C, and -- the part the author does not state --
an INDEPENDENT closed-form prediction of C from the RMS-over-the-aperture
reading of the perturbation phase, plus a numpy-only oracle for the whole
departure that never calls the transport.
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
    _collins_transport, propagate_carrier_referenced)

WHICH = sys.argv[2] if len(sys.argv) > 2 else 'mine'

if WHICH == 'author':
    # w0, theta, f -- the three numbers the author's fixture is derived from.
    W0, THETA, F, N, DX = 15.915e-6, 20.0e-3, 20.0e-3, 512, 8e-6
    LAM = float(np.pi * W0 * THETA)
    D_LADDER = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3)
    D_ANGLE = 1e-4
else:
    # MINE.  All three of f, w0 and lambda changed, and the geometry kept
    # sane: NA = w_in/|R_in| = 0.0224 (paraxial to 2.5e-04 in theta^2/2),
    # the grid holds 5.2 input radii (truncation 1.5e-12 of peak), and the
    # ladder spans z_eff by 206x because zR^2/f = 27.5 um sets the closest
    # approach to A = 0 that d = 1 um reaches.
    LAM, W0, F, N, DX = 1.55e-6, 22.0e-6, 35.0e-3, 512, 16e-6
    THETA = float(LAM / (np.pi * W0))
    D_LADDER = (1e-6, 1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 5e-3)
    D_ANGLE = 1e-4

ZR = float(np.pi * W0 ** 2 / LAM)
K = 2.0 * np.pi / LAM
EPS = float(np.finfo(np.float64).eps)


def w_of_q(q):
    return float(np.sqrt(LAM / (np.pi * np.imag(1.0 / q))))


def R_of_q(q):
    re = float(np.real(1.0 / q))
    return float('inf') if re == 0.0 else 1.0 / re


def axis(n, d):
    return (np.arange(int(n), dtype=np.float64) - int(n) / 2) * float(d)


def pitch_for(w, R, n=N):
    p_win = 6.0 * w / float(n)
    p_amp = w / 8.0
    p_curv = (LAM * abs(R) / (4.0 * w)) if np.isfinite(R) else p_amp
    return float(min(p_win, p_amp, p_curv))


def rel(got, ref):
    g, r = np.asarray(got), np.asarray(ref)
    return float(np.linalg.norm(g - r) / np.linalg.norm(r))


def fit(xs, ys):
    p = np.polyfit(np.log(xs), np.log(ys), 1)
    resid = np.log(ys) - np.polyval(p, np.log(xs))
    return float(p[0]), float(np.max(np.abs(np.exp(resid) - 1.0)))


q_in = complex(-F, -ZR)
R_IN = R_of_q(q_in)
W_IN = w_of_q(q_in)
x = axis(N, DX)
r2 = x[None, :] ** 2 + x[:, None] ** 2
ENV = np.exp(-r2 / W_IN ** 2).astype(np.complex128)

out = {'build': vlib.build_tag(), 'which': WHICH,
       'numpy': np.__version__, 'python': sys.version.split()[0],
       'fixture': {'lam': LAM, 'w0': W0, 'f': F, 'theta_beam': THETA,
                   'zR': ZR, 'N': N, 'dx': DX, 'R_in': R_IN, 'w_in': W_IN,
                   'k': K, 'NA_in': W_IN / abs(R_IN),
                   'theta_env_analytic': float(LAM / (np.pi * W_IN)),
                   'grid_radii': (N * DX / 2.0) / W_IN,
                   'edge_truncation': float(np.exp(-((N * DX / 2.0)
                                                     / W_IN) ** 2))}}


def departure(env, R_in, z, dx_out, n=N, dx_in=DX):
    """rel L2 between the 'exact' and 'fresnel' arms on the SAME lattice,
    plus the leg's measured (z_eff, k4, theta)."""
    common = dict(wavelength=LAM, dx=dx_in, transport='collins',
                  dx_out=dx_out, carrier_out=float('inf'),
                  on_collins_sampling='ignore')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        a = propagate_carrier_referenced(env, R_in, z,
                                         **dict(common, gap_kernel='exact'))
        b = propagate_carrier_referenced(env, R_in, z,
                                         **dict(common, gap_kernel='fresnel'))
        st = {}
        _collins_transport(env, R_in, z, LAM, dx_in, dx_in, dx_out=dx_out,
                           dy_out=dx_out, N_out_x=n, N_out_y=n,
                           R_ref=float('inf'), gap_kernel='auto',
                           on_collins_sampling='ignore', stats_out=st)
    A, B = float(st['abcd'][0]), float(st['abcd'][1])
    return (rel(a.env, b.env), abs(B / A), float(st['k4']), st['kernel'],
            float(max(st['theta_x'], st['theta_y'])))


def numpy_only_prediction(w_env, z_eff, n=N, dx_in=DX):
    """The SAME departure, computed with numpy alone -- no library call.

    The exact/Fresnel kernel ratio is the diagonal phase
    ``phi(q) = z_eff (sqrt(k^2-q^2) - k + q^2/2k)``; the perturbed envelope is
    ``F^-1[S e^{i phi}]``, so the relative L2 on the input grid is
    ``||S (e^{i phi} - 1)|| / ||S||`` by Parseval, and the transport (an
    isometry up to the ABCD scale factor) carries it to the output lattice.
    Also returns the leading-order RMS reading ``sqrt(<phi^2>)`` and the
    closed form ``sqrt(3/2) k |z_eff| theta0^4 / 8``.
    """
    g = axis(n, dx_in)
    env = np.exp(-(g[None, :] ** 2 + g[:, None] ** 2) / w_env ** 2)
    S = np.fft.fft2(env)
    qx = 2.0 * np.pi * np.fft.fftfreq(n, d=dx_in)
    q2 = qx[None, :] ** 2 + qx[:, None] ** 2
    rad = np.maximum(K * K - q2, 0.0)
    phi = z_eff * (np.sqrt(rad) - K + q2 / (2.0 * K))
    w = np.abs(S) ** 2
    exact = float(np.linalg.norm(S * (np.exp(1j * phi) - 1.0))
                  / np.linalg.norm(S))
    rms = float(np.sqrt((w * phi ** 2).sum() / w.sum()))
    th0 = LAM / (np.pi * w_env)
    closed = float(np.sqrt(1.5) * K * abs(z_eff) * th0 ** 4 / 8.0)
    return exact, rms, closed, float(th0)


# ---- sweep 1: z_eff, at fixed envelope --------------------------------
lad = []
for d in D_LADDER:
    z = F - d
    qq = q_in + z
    ww = w_of_q(qq)
    dxo = pitch_for(ww, R_of_q(qq))
    dep, z_eff, k4, kern, th = departure(ENV, R_IN, z, dxo)
    pred, rms, closed, th0 = numpy_only_prediction(W_IN, z_eff)
    lad.append({'d': d, 'z': z, 'w_out': ww, 'dx_out': dxo, 'dep': dep,
                'z_eff': z_eff, 'k4': k4, 'kernel': kern,
                'theta_measured': th, 'theta_analytic': th0,
                'numpy_only_pred': pred, 'numpy_only_rms': rms,
                'closed_form_sqrt1p5': closed,
                'C_measured': dep / (K * z_eff * th0 ** 4 / 8.0)})
zs = [r['z_eff'] for r in lad]
ds = [r['dep'] for r in lad]
slope_z, worst_z = fit(zs, ds)
out['ladder_z_eff'] = {'rows': lad, 'slope': slope_z, 'worst_dev': worst_z,
                       'span': max(zs) / min(zs), 'points': len(zs)}

# ---- sweep 2: envelope angle, at fixed geometry -----------------------
ang = []
z = F - D_ANGLE
for scale in (0.25, 0.35, 0.5, 0.7, 1.0, 1.4):
    w_env = scale * W_IN
    q_e = 1.0 / (1.0 / R_IN + 1j * LAM / (np.pi * w_env ** 2))
    w_out_e = w_of_q(q_e + z)
    dxo = pitch_for(w_out_e, R_of_q(q_e + z))      # ITS OWN beam, not the
    env = np.exp(-r2 / w_env ** 2).astype(np.complex128)   # fixture's
    dep, z_eff, k4, kern, th = departure(env, R_IN, z, dxo)
    pred, rms, closed, th0 = numpy_only_prediction(w_env, z_eff)
    ang.append({'scale': scale, 'w_env': w_env, 'dx_out': dxo,
                'theta_analytic': th0, 'theta_measured': th, 'dep': dep,
                'z_eff': z_eff, 'k4': k4, 'kernel': kern,
                'numpy_only_pred': pred, 'numpy_only_rms': rms,
                'closed_form_sqrt1p5': closed,
                'C_measured': dep / (K * z_eff * th0 ** 4 / 8.0)})
th_a = [r['theta_analytic'] for r in ang]
th_m = [r['theta_measured'] for r in ang]
da = [r['dep'] for r in ang]
sl_a, wo_a = fit(th_a, da)
sl_m, wo_m = fit(th_m, da)
out['ladder_theta'] = {'rows': ang, 'slope_analytic': sl_a,
                       'worst_dev_analytic': wo_a, 'slope_measured': sl_m,
                       'worst_dev_measured': wo_m,
                       'span': max(th_a) / min(th_a), 'points': len(th_a)}

# ---- the constant -----------------------------------------------------
Cs = [r['C_measured'] for r in lad] + [r['C_measured'] for r in ang]
out['C'] = {'ladder': [r['C_measured'] for r in lad],
            'angle': [r['C_measured'] for r in ang],
            'mean': float(np.mean(Cs)), 'min': float(np.min(Cs)),
            'max': float(np.max(Cs)),
            'spread_rel': float((np.max(Cs) - np.min(Cs)) / np.mean(Cs)),
            'sqrt_3_over_2': float(np.sqrt(1.5)),
            'author_C': 1.2248}

# ---- the COLLIMATED leg, a different transport and no carrier ---------
q_a = complex(0.0, -ZR)
w_a = w_of_q(q_a)
z2 = 1.5 * ZR
dx2 = 10.0 * w_of_q(q_a + z2) / N
x2 = axis(N, dx2)
env2 = np.exp(-(x2[None, :] ** 2 + x2[:, None] ** 2) / w_a ** 2).astype(
    np.complex128)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    e2 = propagate_carrier_referenced(env2, float('inf'), z2, LAM, dx2,
                                      gap_kernel='exact')
    f2 = propagate_carrier_referenced(env2, float('inf'), z2, LAM, dx2,
                                      gap_kernel='fresnel')
th_c = float(LAM / (np.pi * w_a))
dep_c = rel(e2.env, f2.env)
pc, rc, cc, _ = numpy_only_prediction(w_a, z2, n=N, dx_in=dx2)
out['collimated'] = {'z': z2, 'w': w_a, 'theta': th_c, 'dep': dep_c,
                     'C': dep_c / (K * abs(z2) * th_c ** 4 / 8.0),
                     'numpy_only_pred': pc, 'numpy_only_rms': rc,
                     'closed_form_sqrt1p5': cc, 'dx': dx2}

# ---- the RMS-vs-peak geometry factor, from the weight alone -----------
# <theta^8>/<theta^0> over the 2-D spectral intensity exp(-2 (q/q0)^2) is
# 3/2 theta0^8, so RMS(theta^4) = sqrt(3/2) theta0^4.  Checked by quadrature
# here, and the 1-D value is reported beside it because they differ -- which
# is what makes the constant a GEOMETRY factor and not a universal number.
u = np.linspace(0.0, 40.0, 2000001)
w2 = np.exp(-2.0 * u ** 2)
m8 = float(np.trapezoid(u ** 8 * w2 * u, u) / np.trapezoid(w2 * u, u))
m8_1d = float(np.trapezoid(u ** 8 * w2, u) / np.trapezoid(w2, u))
out['rms_factor'] = {'2d_circular_mean_u8': m8, '2d_C': float(np.sqrt(m8)),
                     '1d_mean_u8': m8_1d, '1d_C': float(np.sqrt(m8_1d)),
                     'closed_2d': 1.5, 'closed_1d': 105.0 / 256.0}

vlib.write_json(out, sys.argv[1])
