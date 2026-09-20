"""CLAIM 5 (c) ADVERSARIAL -- a leg the Collins transport resolves FLAT and
whose chirp-Z is NOT representable (K1 = K3 = 1.77), on a geometry the
Sziklas transport can evaluate perfectly well (m = +0.1).

The branch's own comment says the flat legs are "precisely the legs the
Sziklas transport could never evaluate".  This probe tests that sentence.

    python <this> <tree_root> <label> <out.json>
"""
import json
import sys
import warnings

import numpy as np

ROOT = sys.argv[1].replace('\\', '/').rstrip('/')
LABEL = sys.argv[2]
OUT = sys.argv[3]

import lumenairy                                            # noqa: E402
import lumenairy.propagators.carrier as C                   # noqa: E402

print('lumenairy.__file__ =', lumenairy.__file__)
assert lumenairy.__file__.replace('\\', '/').lower().startswith(ROOT.lower())

WL, N, DX = 1.064e-6, 1024, 4.0e-6
W = 0.30e-3       # 6.83 grid half-widths: truncation exp(-46.6), so the
                  # analytic Gaussian below really is the oracle
F0 = 1.0e5                       # 0.4 cycles / sample -> theta = 0.1064 rad
TH = WL * F0
R = -2e-3 / 0.89                 # A = 1 + z/R = 0.11 at z = 2 mm
Z = 2e-3
A = 1.0 + Z / R
K = 2.0 * np.pi / WL

x = (np.arange(N) - N / 2) * DX
X, Y = np.meshgrid(x, x)
env = (np.exp(-(X ** 2 + Y ** 2) / (W * W))
       * np.exp(2j * np.pi * F0 * X)).astype(np.complex128)

out = {'label': LABEL, 'lumenairy_file': lumenairy.__file__,
       'A': A, 'R': R, 'z': Z, 'theta_rad': TH,
       'theta_over_nyquist': TH / (WL / (2 * DX)),
       'm_sziklas': (R + Z) / R}

# --------------------------------------------------------------- oracle ----
# paraxial (Fresnel) ABCD of a TILTED Gaussian: the untilted Gaussian
# propagated by z, displaced by B*theta, times exp(i k theta x) and a piston.
q_in = 1.0 / complex(1.0 / R, WL / (np.pi * W * W))
q_out = q_in + Z
inv = 1.0 / q_out
w_out = float(np.sqrt(WL / (np.pi * inv.imag)))
R_out_an = float(1.0 / inv.real)
x_c_an = Z * TH                                # B*theta, A*x0 = 0
out['analytic'] = {'w_out_um': w_out * 1e6, 'R_out': R_out_an,
                   'centroid_x_um': x_c_an * 1e6,
                   'sigma_x_um': w_out / 2.0 * 1e6,
                   'sigma_y_um': w_out / 2.0 * 1e6}


def moments(E, dxa, dya):
    I = np.abs(np.asarray(E)) ** 2
    Ny, Nx = I.shape
    xx = (np.arange(Nx) - Nx / 2) * dxa
    yy = (np.arange(Ny) - Ny / 2) * dya
    tot = I.sum()
    cx = (I * xx[None, :]).sum() / tot
    cy = (I * yy[:, None]).sum() / tot
    sx = np.sqrt((I * (xx[None, :] - cx) ** 2).sum() / tot)
    sy = np.sqrt((I * (yy[:, None] - cy) ** 2).sum() / tot)
    return (float(cx), float(cy), float(sx), float(sy),
            float(tot * dxa * dya))


def split_d(d):
    return (float(d), float(d)) if not isinstance(d, tuple) else (
        float(d[0]), float(d[1]))


p_in = float((np.abs(env) ** 2).sum()) * DX * DX
out['power_in'] = p_in

for gk in ('fresnel', 'auto'):
    diag = {}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        cr = C._collins_carrier_leg(env, R, Z, WL, DX, DX, gap_kernel=gk,
                                    on_collins_sampling='warn', diag=diag)
    dxo, dyo = split_d(cr.dx)
    cx, cy, sx, sy, pw = moments(cr.env, dxo, dyo)
    row = {'gap_kernel': gk,
           'form': diag.get('collins_form'),
           'k1': max(diag.get('collins_k1') or (0, 0)),
           'k3': max(diag.get('collins_k3') or (0, 0)),
           'flat': diag.get('collins_flat_reference'),
           'dx_out_um': dxo * 1e6, 'R_out': repr(cr.R),
           'window_half_um': N * dxo / 2 * 1e6,
           'centroid_x_um': cx * 1e6, 'centroid_y_um': cy * 1e6,
           'sigma_x_um': sx * 1e6, 'sigma_y_um': sy * 1e6,
           'power': pw, 'power_ratio': pw / p_in,
           'warnings': [str(w.message)[:200] for w in rec]}
    out['collins_' + gk] = row
    print('collins', gk, row['form'], 'K1=%.4f' % row['k1'],
          'flat=%s' % row['flat'], 'cx=%.2f um sx=%.2f um'
          % (row['centroid_x_um'], row['sigma_x_um']))

    try:
        crs = C.propagate_carrier_referenced(env, R, Z, WL, DX,
                                             gap_kernel=gk,
                                             transport='sziklas')
        dxs, dys = split_d(crs.dx)
        cxs, cys, sxs, sys2, pws = moments(crs.env, dxs, dys)
        outs = {'dx_out_um': dxs * 1e6, 'R_out': repr(crs.R),
                'window_half_um': np.shape(crs.env)[-1] * dxs / 2 * 1e6,
                'centroid_x_um': cxs * 1e6, 'centroid_y_um': cys * 1e6,
                'sigma_x_um': sxs * 1e6, 'sigma_y_um': sys2 * 1e6,
                'power': pws, 'power_ratio': pws / p_in}
    except Exception as exc:                                 # noqa: BLE001
        outs = {'raised': type(exc).__name__, 'msg': str(exc)[:300]}
    out['sziklas_' + gk] = outs
    print('sziklas', gk, outs if 'raised' in outs
          else 'cx=%.2f um sx=%.2f um' % (outs['centroid_x_um'],
                                          outs['sigma_x_um']))


# --- control: the SAME geometry with no tilt (theta small -> not flat) -----
env0 = np.exp(-(X ** 2 + Y ** 2) / (W * W)).astype(np.complex128)
q0 = 1.0 / complex(1.0 / R, WL / (np.pi * W * W))
i0 = 1.0 / (q0 + Z)
out['control_analytic'] = {'w_out_um': float(np.sqrt(WL / (np.pi * i0.imag)))
                           * 1e6,
                           'sigma_x_um': float(np.sqrt(WL / (np.pi * i0.imag)))
                           / 2.0 * 1e6}
d0 = {}
with warnings.catch_warnings(record=True):
    warnings.simplefilter('always')
    c0 = C._collins_carrier_leg(env0, R, Z, WL, DX, DX, gap_kernel='fresnel',
                                on_collins_sampling='ignore', diag=d0)
dx0, dy0 = split_d(c0.dx)
cx0, cy0, sx0, sy0, pw0 = moments(c0.env, dx0, dy0)
p0 = float((np.abs(env0) ** 2).sum()) * DX * DX
out['control_collins'] = {'form': d0.get('collins_form'),
                          'k1': max(d0.get('collins_k1') or (0, 0)),
                          'flat': d0.get('collins_flat_reference'),
                          'sigma_x_um': sx0 * 1e6, 'power_ratio': pw0 / p0,
                          'dx_out_um': dx0 * 1e6}
cs0 = C.propagate_carrier_referenced(env0, R, Z, WL, DX,
                                     gap_kernel='fresnel',
                                     transport='sziklas')
dxs0, dys0 = split_d(cs0.dx)
cxs0, cys0, sxs0, sys0, pws0 = moments(cs0.env, dxs0, dys0)
out['control_sziklas'] = {'sigma_x_um': sxs0 * 1e6,
                          'power_ratio': pws0 / p0, 'dx_out_um': dxs0 * 1e6}
print('control collins sx=%.4f  sziklas sx=%.4f  analytic sx=%.4f'
      % (sx0 * 1e6, sxs0 * 1e6, out['control_analytic']['sigma_x_um']))

with open(OUT, 'w', encoding='cp1252') as fh:
    json.dump(out, fh, indent=1, default=repr)
print('WROTE', OUT)
