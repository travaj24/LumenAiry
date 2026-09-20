"""CLAIM 6 (a, b) -- the pinned standoff leg of ``carrier_referenced_focus_
readout``, reproduced and ADJUDICATED against an independent dense-Fresnel
oracle.

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
import inspect                                              # noqa: E402
TD = inspect.signature(C.propagate_carrier_referenced).parameters[
    'transport'].default
print('transport default  =', TD)

N, DX, WL = 128, 4e-6, 633e-9
W, R, Z = 120e-6, -0.03, 0.03
STANDOFF = 1e-3
DXO, NOUT = 2e-7, 32
K = 2.0 * np.pi / WL

x = (np.arange(N) - N / 2) * DX
X, Y = np.meshgrid(x, x)
env = np.exp(-(X ** 2 + Y ** 2) / (W ** 2)).astype(np.complex128)

out = {'label': LABEL, 'lumenairy_file': lumenairy.__file__,
       'transport_default': TD,
       'fixture': {'N': N, 'dx': DX, 'wl': WL, 'w': W, 'R': R, 'z': Z,
                   'standoff': STANDOFF, 'dx_out': DXO, 'N_out': NOUT}}

# ---------------------------------------------------------------------------
# the two stop-plane legs, measured directly
# ---------------------------------------------------------------------------
z_stop = Z - np.copysign(STANDOFF, Z)
out['z_stop'] = z_stop
legs = {}
for tr in ('sziklas', 'collins'):
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            cr = C.propagate_carrier_referenced(env, R, z_stop, WL, DX,
                                                transport=tr)
        es = np.asarray(cr.env)
        dxs = float(cr.dx if not isinstance(cr.dx, tuple) else cr.dx[0])
        cen = C._envelope_amp_centroid(es, dxs, dxs)
        w_stop = C._envelope_amp_radius(es, dxs, dxs, centre=cen)
        half = (0.5 * min(es.shape[-1], es.shape[-2]) * dxs
                - max(abs(cen[0]), abs(cen[1])))
        legs[tr] = {'dx_stop_um': dxs * 1e6, 'R_stop': repr(cr.R),
                    'centroid_um': [c * 1e6 for c in cen],
                    'w_stop_um': w_stop * 1e6,
                    'half_stop_um': half * 1e6,
                    'containment': half / w_stop,
                    'shape': list(es.shape),
                    'warnings': [str(w.message)[:160] for w in rec]}
    except Exception as exc:                                 # noqa: BLE001
        legs[tr] = {'raised': type(exc).__name__, 'msg': str(exc)[:300]}
    print(tr, 'stop leg:', legs[tr].get('half_stop_um'),
          legs[tr].get('w_stop_um'), legs[tr].get('containment'))
out['stop_legs'] = legs

# ---------------------------------------------------------------------------
# the readout itself, under BOTH transports for its internal leg
# ---------------------------------------------------------------------------
_real_pcr = C.propagate_carrier_referenced


def _as(tr):
    def _f(*a, **k):
        k['transport'] = tr
        return _real_pcr(*a, **k)
    return _f


def run_readout(tr, **extra):
    kw = dict(dx_out=DXO, N_out=NOUT, standoff=STANDOFF, on_replica='ignore')
    kw.update(extra)
    C.propagate_carrier_referenced = _as(tr)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            f = C.carrier_referenced_focus_readout(env, R, Z, WL, DX, **kw)
        return (np.asarray(f), [str(w.message)[:200] for w in rec], None)
    except Exception as exc:                                 # noqa: BLE001
        return (None, [], type(exc).__name__ + ': ' + str(exc)[:400])
    finally:
        C.propagate_carrier_referenced = _real_pcr


fields = {}
for tr in ('sziklas', 'collins'):
    f, w, err = run_readout(tr)
    fields[tr] = f
    out['readout_' + tr] = {'raised': err, 'warnings': w,
                            'peak': (None if f is None
                                     else float(np.abs(f).max() ** 2))}
    print('readout via', tr, '->', 'RAISED ' + err[:90] if err else 'returned')
# the Sziklas answer with the guard downgraded, so the two are comparable
f_sz_w, w_sz_w, e_sz_w = run_readout('sziklas', on_focus_containment='warn')
fields['sziklas_warn'] = f_sz_w
out['readout_sziklas_warn'] = {'raised': e_sz_w, 'warnings': w_sz_w,
                               'peak': (None if f_sz_w is None else
                                        float(np.abs(f_sz_w).max() ** 2))}
f_co_w, w_co_w, e_co_w = run_readout('collins', on_focus_containment='warn')
fields['collins_warn'] = f_co_w
out['readout_collins_warn'] = {'raised': e_co_w, 'warnings': w_co_w,
                               'peak': (None if f_co_w is None else
                                        float(np.abs(f_co_w).max() ** 2))}

# ---------------------------------------------------------------------------
# INDEPENDENT ORACLE: dense separable Fresnel of the reconstructed FULL field
# (the input grid samples that full field at 74 um fringe pitch on a 4 um
# lattice, so the sampled sum and a 64x-oversampled continuous quadrature of
# the analytic input must agree -- both are computed and compared)
# ---------------------------------------------------------------------------
xo = (np.arange(NOUT) - NOUT / 2) * DXO


def fresnel_1d(xs, Es, xp_out, z, wl):
    k = 2.0 * np.pi / wl
    ph = np.exp(1j * k * xs ** 2 / (2.0 * z))
    kern = np.exp(-1j * k * np.outer(xp_out, xs) / z)
    w = np.gradient(xs)
    return kern @ (Es * ph * w)


# (a) the SAMPLED input, summed
E1 = np.exp(-x ** 2 / W ** 2) * np.exp(1j * K * x ** 2 / (2.0 * R))
f_s = fresnel_1d(x, E1, xo, Z, WL)
# (b) a 64x oversampled continuous quadrature of the same analytic function
xd = np.linspace(x[0], x[-1] + DX, N * 64, endpoint=False)
E1d = np.exp(-xd ** 2 / W ** 2) * np.exp(1j * K * xd ** 2 / (2.0 * R))
f_d = fresnel_1d(xd, E1d, xo, Z, WL)
pre = np.exp(1j * K * Z) / (1j * WL * Z)
outer_s = pre * np.outer(f_s, f_s) * np.exp(
    1j * K * (xo[:, None] ** 2 + xo[None, :] ** 2) / (2.0 * Z))
outer_d = pre * np.outer(f_d, f_d) * np.exp(
    1j * K * (xo[:, None] ** 2 + xo[None, :] ** 2) / (2.0 * Z))
rel_sd = float(np.linalg.norm(outer_s - outer_d) / np.linalg.norm(outer_d))
out['oracle_sampled_vs_dense_relL2'] = rel_sd
print('oracle self-consistency (sampled sum vs 64x dense quadrature):', rel_sd)
# a 256x run, to show the dense quadrature is converged
xd2 = np.linspace(x[0], x[-1] + DX, N * 256, endpoint=False)
E1d2 = np.exp(-xd2 ** 2 / W ** 2) * np.exp(1j * K * xd2 ** 2 / (2.0 * R))
f_d2 = fresnel_1d(xd2, E1d2, xo, Z, WL)
outer_d2 = pre * np.outer(f_d2, f_d2) * np.exp(
    1j * K * (xo[:, None] ** 2 + xo[None, :] ** 2) / (2.0 * Z))
out['oracle_64x_vs_256x_relL2'] = float(
    np.linalg.norm(outer_d - outer_d2) / np.linalg.norm(outer_d2))
print('oracle convergence 64x vs 256x:', out['oracle_64x_vs_256x_relL2'])

ORACLE = outer_d2


def compare(f, T):
    if f is None:
        return None
    f = np.asarray(f)
    if f.shape != T.shape:
        return {'shape_mismatch': [list(f.shape), list(T.shape)]}
    ov = np.vdot(T, f)
    ph = ov / abs(ov) if abs(ov) > 0 else 1.0
    sc = ov / np.vdot(T, T)
    return {'relL2_absolute': float(np.linalg.norm(f - T)
                                    / np.linalg.norm(T)),
            'relL2_phase_free': float(np.linalg.norm(f / ph - T)
                                      / np.linalg.norm(T)),
            'relL2_scale_free': float(np.linalg.norm(f / sc - T)
                                      / np.linalg.norm(T)),
            'best_scale_abs': float(abs(sc)),
            'peak_ratio': float(np.abs(f).max() ** 2
                                / np.abs(T).max() ** 2),
            'intensity_relL2': float(
                np.linalg.norm(np.abs(f) ** 2 - np.abs(T) ** 2)
                / np.linalg.norm(np.abs(T) ** 2))}


out['oracle_peak'] = float(np.abs(ORACLE).max() ** 2)
for nm, f in fields.items():
    out['vs_oracle_' + nm] = compare(f, ORACLE)
    print('vs oracle', nm, out['vs_oracle_' + nm])

# the analytic Gaussian at the readout plane, as a cross-check of the oracle
q = 1.0 / complex(1.0 / R, WL / (np.pi * W * W))
inv = 1.0 / (q + Z)
w_out = float(np.sqrt(WL / (np.pi * inv.imag)))
out['analytic_w_out_um'] = w_out * 1e6
out['analytic_R_out'] = (float(1.0 / inv.real) if inv.real else None)

with open(OUT, 'w', encoding='cp1252') as fh:
    json.dump(out, fh, indent=1, default=repr)
print('WROTE', OUT)
