"""CLAIM 5 (a, c, d, e) -- the three fallback geometries, on BOTH trees.

Run with cwd == the tree root and PYTHONPATH == the tree root.
    python <this> <tree_root> <label> <out.json>
"""
import inspect
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
assert lumenairy.__file__.replace('\\', '/').lower().startswith(ROOT.lower()), \
    'WRONG TREE: ' + lumenairy.__file__ + ' not under ' + ROOT
print('carrier.__file__   =', C.__file__)

_sig = inspect.signature(C.propagate_carrier_referenced)
TRANSPORT_DEFAULT = _sig.parameters['transport'].default
print('transport default  =', TRANSPORT_DEFAULT)

res = {'label': LABEL, 'lumenairy_file': lumenairy.__file__,
       'transport_default': TRANSPORT_DEFAULT,
       'python': sys.version.split()[0], 'numpy': np.__version__}


# --------------------------------------------------------------------------
# my own analytic Gaussian (exp(-i w t) / exp(+i k z); 1/q = 1/R + i l/(pi w^2))
# --------------------------------------------------------------------------
def q_of(w, R, wl):
    inv_R = 0.0 if R is None else 1.0 / R
    return 1.0 / complex(inv_R, wl / (np.pi * w * w))


def prop_q(q, z):
    return q + z


def wR_of_q(q, wl):
    inv = 1.0 / q
    w = np.sqrt(wl / (np.pi * inv.imag))
    R = (1.0 / inv.real) if inv.real != 0.0 else np.inf
    return float(w), float(R)


def analytic_windowed_r2m(w, a):
    """exact windowed r2m of I=exp(-2r^2/w^2) over r<=a (closed form + check)."""
    t = 2.0 * a * a / (w * w)
    num = (w ** 4 / 8.0) * (1.0 - (1.0 + t) * np.exp(-t))
    den = (w * w / 4.0) * (1.0 - np.exp(-t))
    closed = float(np.sqrt(num / den))
    rr = np.linspace(0.0, a, 200001)
    I = np.exp(-2.0 * rr ** 2 / (w * w))
    quad = float(np.sqrt(np.trapezoid(I * rr ** 3, rr)
                         / np.trapezoid(I * rr, rr)))
    assert abs(closed - quad) / closed < 1e-7, (closed, quad)
    return closed


def grid_windowed_r2m(I, dx, a):
    N = I.shape[0]
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    Iw = np.where(r <= a, I, 0.0)
    tot = Iw.sum()
    return float(np.sqrt((Iw * r ** 2).sum() / tot)), float(tot)


def gauss_env(N, dx, w):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)


def fin(a):
    a = np.asarray(a)
    ok = np.isfinite(a)
    return {'n_finite': int(ok.sum()), 'n': int(a.size),
            'all_finite': bool(np.all(ok)),
            'max_abs': (float(np.abs(a[ok]).max()) if ok.any() else None)}


def jsonable(o):
    if isinstance(o, bool):
        return o
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return ('inf' if v > 0 else '-inf') if np.isinf(v) else (
            'nan' if np.isnan(v) else v)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (tuple, list)):
        return [jsonable(v) for v in o]
    if isinstance(o, dict):
        return dict((str(k), jsonable(v)) for k, v in o.items())
    if o is None or isinstance(o, str):
        return o
    return repr(o)


def axis_second_moments(E, dxa, dya):
    I = np.abs(np.asarray(E)) ** 2
    Ny, Nx = I.shape
    x = (np.arange(Nx) - Nx / 2) * dxa
    y = (np.arange(Ny) - Ny / 2) * dya
    tot = I.sum()
    sx = np.sqrt((I * (x ** 2)[None, :]).sum() / tot)
    sy = np.sqrt((I * (y ** 2)[:, None]).sum() / tot)
    return float(sx), float(sy)


def split_d(d):
    return (float(d), float(d)) if not isinstance(d, tuple) else (
        float(d[0]), float(d[1]))


# ==========================================================================
# B4 fixture
# ==========================================================================
_WL, _N, _DX, _W = 1.064e-6, 1024, 4.0e-6, 0.30e-3
env_b4 = gauss_env(_N, _DX, _W)

# --- (i) COLLIMATED, R = inf, z = 5 mm ------------------------------------
diag = {}
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    cr = C._collins_carrier_leg(env_b4, np.inf, 5e-3, _WL, _DX, _DX,
                                on_collins_sampling='ignore', diag=diag)
e = np.asarray(cr.env)
res['i_collimated'] = {
    'diag': jsonable(diag), 'env': fin(e),
    'dx_out': repr(cr.dx), 'R_out': repr(cr.R),
    'warnings': [str(w.message)[:160] for w in rec],
}
cr_s = C.propagate_carrier_referenced(env_b4, np.inf, 5e-3, _WL, _DX,
                                      transport='sziklas')
es = np.asarray(cr_s.env)
res['i_collimated']['sziklas_env'] = fin(es)
res['i_collimated']['sziklas_dx'] = repr(cr_s.dx)
if e.shape == es.shape:
    res['i_collimated']['bit_identical_to_sziklas'] = bool(
        np.array_equal(e, es))
q0 = q_of(_W, None, _WL)
w5, R5 = wR_of_q(prop_q(q0, 5e-3), _WL)
res['i_collimated']['analytic_w_out_um'] = w5 * 1e6
a_win = 3.0 * w5
res['i_collimated']['r2m_analytic_um'] = analytic_windowed_r2m(w5, a_win) * 1e6
if np.all(np.isfinite(e)) and np.all(np.isfinite(np.atleast_1d(
        np.asarray(split_d(cr.dx))))):
    r2m_g, _ = grid_windowed_r2m(np.abs(e) ** 2, split_d(cr.dx)[0], a_win)
    res['i_collimated']['r2m_um'] = r2m_g * 1e6
r2m_s, _ = grid_windowed_r2m(np.abs(es) ** 2, split_d(cr_s.dx)[0], a_win)
res['i_collimated']['r2m_sziklas_um'] = r2m_s * 1e6

# --- (ii) ASTIGMATIC, R = (-40, -55) mm, z = 5 mm --------------------------
Rx, Ry, z2 = -40e-3, -55e-3, 5e-3
diag = {}
with warnings.catch_warnings(record=True) as rec:
    warnings.simplefilter('always')
    cr = C._collins_carrier_leg(env_b4, (Rx, Ry), z2, _WL, _DX, _DX,
                                on_collins_sampling='ignore', diag=diag)
e = np.asarray(cr.env)
mx, my = (Rx + z2) / Rx, (Ry + z2) / Ry
res['ii_astigmatic'] = {
    'diag': jsonable(diag), 'env': fin(e),
    'dx_out': jsonable(cr.dx), 'R_out': jsonable(cr.R),
    'promised_R': [Rx + z2, Ry + z2],
    'promised_dx_um': [mx * _DX * 1e6, my * _DX * 1e6],
    'm': [mx, my],
    'warnings': [str(w.message)[:200] for w in rec],
}
_bx = C._collins_input_box(env_b4, _DX, _DX, _WL, C._COLLINS_TAIL_FRAC)
res['ii_astigmatic']['input_box'] = jsonable(_bx)
period = _WL * abs(z2) / _DX
res['ii_astigmatic']['chirpz_period_m'] = period
_dxo, _dyo = split_d(cr.dx)
res['ii_astigmatic']['window_over_period_x'] = _N * _dxo / period
res['ii_astigmatic']['window_over_period_y'] = _N * _dyo / period
cr_s = C.propagate_carrier_referenced(env_b4, (Rx, Ry), z2, _WL, _DX,
                                      transport='sziklas')
es = np.asarray(cr_s.env)
res['ii_astigmatic']['sziklas'] = {'dx': jsonable(cr_s.dx),
                                   'R': jsonable(cr_s.R), 'env': fin(es)}
res['ii_astigmatic']['equals_sziklas'] = bool(
    e.shape == es.shape and np.array_equal(e, es))
wx_o, Rx_o = wR_of_q(prop_q(q_of(_W, Rx, _WL), z2), _WL)
wy_o, Ry_o = wR_of_q(prop_q(q_of(_W, Ry, _WL), z2), _WL)
res['ii_astigmatic']['analytic'] = {'w_x_um': wx_o * 1e6, 'w_y_um': wy_o * 1e6,
                                    'R_x': Rx_o, 'R_y': Ry_o}
sx, sy = axis_second_moments(e, _dxo, _dyo)
res['ii_astigmatic']['sigma_x_um'] = sx * 1e6
res['ii_astigmatic']['sigma_y_um'] = sy * 1e6
res['ii_astigmatic']['sigma_x_analytic_um'] = wx_o / 2.0 * 1e6
res['ii_astigmatic']['sigma_y_analytic_um'] = wy_o / 2.0 * 1e6
_dxs, _dys = split_d(cr_s.dx)
sxs, sys2 = axis_second_moments(es, _dxs, _dys)
res['ii_astigmatic']['sigma_x_sziklas_um'] = sxs * 1e6
res['ii_astigmatic']['sigma_y_sziklas_um'] = sys2 * 1e6

# ==========================================================================
# (iii) PAST THE FOCUS -- test_carrier_referenced.py's own oracle
# ==========================================================================
WL3 = 1.31e-6
W0 = 4e-6
ZR = np.pi * W0 ** 2 / WL3
ZF = 30e-3
RIN = -(ZF + ZR ** 2 / ZF)
WIN = W0 * np.sqrt(1 + (ZF / ZR) ** 2)
N3 = 2048
DX3 = (8.0 * WIN) / N3
env3 = gauss_env(N3, DX3, WIN)
res['iii_setup'] = {'zR_um': ZR * 1e6, 'R_in': RIN, 'w_in_m': WIN,
                    'dx_um': DX3 * 1e6, 'N': N3}
res['iii'] = {}
for z_mm in (45.0, 60.0):
    z = z_mm * 1e-3
    A = 1.0 + z / RIN
    dz = z - ZF
    w_true = W0 * np.sqrt(1 + (dz / ZR) ** 2)
    w_q, R_q = wR_of_q(prop_q(q_of(WIN, RIN, WL3), z), WL3)
    win = 3.0 * w_true
    r2m_a = analytic_windowed_r2m(w_true, win)
    diag = {}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        cr = C._collins_carrier_leg(env3, RIN, z, WL3, DX3, DX3,
                                    on_collins_sampling='ignore', diag=diag)
    e = np.asarray(cr.env)
    dxo = split_d(cr.dx)[0]
    r2m_c, pw_c = grid_windowed_r2m(np.abs(e) ** 2, dxo, win)
    crs = C.propagate_carrier_referenced(env3, RIN, z, WL3, DX3,
                                         transport='sziklas')
    esz = np.asarray(crs.env)
    dxs = split_d(crs.dx)[0]
    r2m_s, pw_s = grid_windowed_r2m(np.abs(esz) ** 2, dxs, win)
    res['iii']['z%dmm' % int(z_mm)] = {
        'A': A, 'diag': jsonable(diag),
        'w_true_um': w_true * 1e6, 'w_q_um': w_q * 1e6, 'R_q': R_q,
        'window_um': win * 1e6,
        'r2m_analytic_um': r2m_a * 1e6,
        'r2m_leg_um': r2m_c * 1e6, 'ratio_leg_over_analytic': r2m_c / r2m_a,
        'dx_out_um': dxo * 1e6, 'R_out': jsonable(cr.R),
        'r2m_sziklas_um': r2m_s * 1e6,
        'ratio_sziklas_over_analytic': r2m_s / r2m_a,
        'dx_sziklas_um': dxs * 1e6, 'R_sziklas': jsonable(crs.R),
        'leg_equals_sziklas': bool(e.shape == esz.shape
                                   and np.array_equal(e, esz)),
        'warnings': [str(w.message)[:200] for w in rec],
    }

# ==========================================================================
# (c) A == 0 exactly, and a resolved FLAT reference -- no fallback
# ==========================================================================
res['c_boundary'] = {}
for Rv, zv, nm in ((-40e-3, 40e-3, 'A_eq_0'),):
    diag = {}
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            cr = C._collins_carrier_leg(env_b4, Rv, zv, _WL, _DX, _DX,
                                        on_collins_sampling='ignore',
                                        diag=diag)
        e = np.asarray(cr.env)
        dxo = split_d(cr.dx)[0]
        w_o, R_o = wR_of_q(prop_q(q_of(_W, Rv, _WL), zv), _WL)
        r2m_g, _ = grid_windowed_r2m(np.abs(e) ** 2, dxo, 3.0 * w_o)
        d = {'A': 1.0 + zv / Rv, 'diag': jsonable(diag), 'env': fin(e),
             'dx_out_um': dxo * 1e6, 'R_out': jsonable(cr.R),
             'analytic_w_out_um': w_o * 1e6, 'analytic_R_out': R_o,
             'r2m_um': r2m_g * 1e6,
             'r2m_analytic_um': analytic_windowed_r2m(w_o, 3.0 * w_o) * 1e6,
             'warnings': [str(w.message)[:200] for w in rec]}
    except Exception as exc:                                 # noqa: BLE001
        d = {'A': 1.0 + zv / Rv, 'raised': type(exc).__name__,
             'msg': str(exc)[:300], 'diag': jsonable(diag)}
    try:
        crs = C.propagate_carrier_referenced(env_b4, Rv, zv, _WL, _DX,
                                             transport='sziklas')
        d['sziklas'] = {'dx': jsonable(crs.dx), 'R': jsonable(crs.R),
                        'env': fin(np.asarray(crs.env))}
    except Exception as exc:                                 # noqa: BLE001
        d['sziklas'] = {'raised': type(exc).__name__, 'msg': str(exc)[:300]}
    res['c_boundary'][nm] = d

for zv in (39.9e-3, 39.99e-3):
    Rv = -40e-3
    diag = {}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        cr = C._collins_carrier_leg(env_b4, Rv, zv, _WL, _DX, _DX,
                                    on_collins_sampling='ignore', diag=diag)
    e = np.asarray(cr.env)
    dxo = split_d(cr.dx)[0]
    w_o, R_o = wR_of_q(prop_q(q_of(_W, Rv, _WL), zv), _WL)
    r2m_g, _ = grid_windowed_r2m(np.abs(e) ** 2, dxo, 3.0 * w_o)
    d = {'A': 1.0 + zv / Rv, 'diag': jsonable(diag), 'env': fin(e),
         'dx_out_um': dxo * 1e6, 'R_out': jsonable(cr.R),
         'analytic_w_out_um': w_o * 1e6, 'analytic_R_out': R_o,
         'r2m_um': r2m_g * 1e6,
         'r2m_analytic_um': analytic_windowed_r2m(w_o, 3.0 * w_o) * 1e6,
         'warnings': [str(w.message)[:200] for w in rec]}
    try:
        crs = C.propagate_carrier_referenced(env_b4, Rv, zv, _WL, _DX,
                                             transport='sziklas')
        esz = np.asarray(crs.env)
        dxs = split_d(crs.dx)[0]
        r2ms, _ = grid_windowed_r2m(np.abs(esz) ** 2, dxs, 3.0 * w_o)
        d['sziklas'] = {'dx_um': dxs * 1e6, 'R': jsonable(crs.R),
                        'env': fin(esz), 'r2m_um': r2ms * 1e6}
    except Exception as exc:                                 # noqa: BLE001
        d['sziklas'] = {'raised': type(exc).__name__, 'msg': str(exc)[:300]}
    res['c_boundary']['flat_z%gmm' % (zv * 1e3)] = d

with open(OUT, 'w', encoding='cp1252') as fh:
    json.dump(jsonable(res), fh, indent=1)
print('WROTE', OUT)
