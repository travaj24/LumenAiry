"""CLAIM 6 (b, c) -- is pinning 'sziklas' at the readout's internal standoff
leg the right general call?  A sweep of geometries x standoffs, each route
graded against an independent dense separable Fresnel oracle.

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

# NO monkeypatch: the TREE decides which transport the readout's internal
# standoff leg takes (vc3_head pins 'sziklas'; vc3_mutC_standoff rides the
# default, which is 'collins' on the branch).  An earlier version of this
# probe forced transport= on EVERY internal call and thereby overrode the
# FALLBACK's own transport='sziklas' -- which recursed, and was an artifact
# of the patch, not of the library.
import ast                                                  # noqa: E402
_src = open(C.__file__, encoding='cp1252').read()
_pinned = "transport='sziklas')" in _src.split(
    'def carrier_referenced_focus_readout')[1].split(
    '_check_focus_containment')[0]
ROUTE = 'sziklas-pinned' if _pinned else 'rides-default'
print('standoff leg route =', ROUTE)


def fresnel_1d(xs, Es, xp_out, z, wl):
    k = 2.0 * np.pi / wl
    ph = np.exp(1j * k * xs ** 2 / (2.0 * z))
    kern = np.exp(-1j * k * np.outer(xp_out, xs) / z)
    return kern @ (Es * ph * np.gradient(xs))


def oracle(N, dx, wl, w, R, z, dxo, nout, over=256):
    k = 2.0 * np.pi / wl
    x = (np.arange(N) - N / 2) * dx
    xd = np.linspace(x[0], x[-1] + dx, N * over, endpoint=False)
    E = np.exp(-xd ** 2 / w ** 2) * np.exp(1j * k * xd ** 2 / (2.0 * R))
    xo = (np.arange(nout) - nout / 2) * dxo
    f = fresnel_1d(xd, E, xo, z, wl)
    pre = np.exp(1j * k * z) / (1j * wl * z)
    return pre * np.outer(f, f) * np.exp(
        1j * k * (xo[:, None] ** 2 + xo[None, :] ** 2) / (2.0 * z))


def grade(f, T):
    f = np.asarray(f)
    ov = np.vdot(T, f)
    sc = ov / np.vdot(T, T)
    return {'relL2': float(np.linalg.norm(f - T) / np.linalg.norm(T)),
            'relL2_scale_free': float(np.linalg.norm(f / sc - T)
                                      / np.linalg.norm(T)),
            'amp_scale': float(abs(sc)),
            'peak_ratio': float(np.abs(f).max() ** 2 / np.abs(T).max() ** 2)}


GEOM = [
    ('G1-test-fixture', 128, 4e-6, 633e-9, 120e-6, -0.03, 0.03, 2e-7, 32),
    ('G2-tight', 128, 4e-6, 633e-9, 120e-6, -0.01, 0.01, 2e-7, 32),
    ('G3-N256', 256, 4e-6, 633e-9, 120e-6, -0.03, 0.03, 2e-7, 32),
    ('G4-coarse', 128, 8e-6, 633e-9, 240e-6, -0.03, 0.03, 4e-7, 32),
    ('G5-1p31um', 128, 4e-6, 1.31e-6, 120e-6, -0.03, 0.03, 4e-7, 32),
]
STANDOFFS = [None, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]

rows = []
for (nm, N, dx, wl, w, R, z, dxo, nout) in GEOM:
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / (w ** 2)).astype(np.complex128)
    T = oracle(N, dx, wl, w, R, z, dxo, nout)
    T64 = oracle(N, dx, wl, w, R, z, dxo, nout, over=64)
    conv = float(np.linalg.norm(T - T64) / np.linalg.norm(T))
    for so in STANDOFFS:
        if True:
            tr = ROUTE
            kw = dict(dx_out=dxo, N_out=nout, on_replica='ignore')
            if so is not None:
                kw['standoff'] = so
            row = {'geom': nm, 'standoff': so, 'transport': tr,
                   'oracle_conv': conv}
            for guard in ('error', 'warn'):
                kw2 = dict(kw, on_focus_containment=guard)
                try:
                    with warnings.catch_warnings(record=True) as rec:
                        warnings.simplefilter('always')
                        f = C.carrier_referenced_focus_readout(
                            env, R, z, wl, dx, **kw2)
                    g = grade(np.asarray(f), T)
                    g['raised'] = None
                    g['n_warn'] = len(rec)
                except Exception as exc:                     # noqa: BLE001
                    g = {'raised': type(exc).__name__ + ': '
                         + str(exc)[:110]}
                row[guard] = g
            rows.append(row)
            print('%-16s so=%-7s %-14s error:%-10s warn-relL2=%s'
                  % (nm, so, tr,
                     ('RAISE' if row['error'].get('raised') else 'ok'),
                     ('%.3e' % row['warn']['relL2']
                      if row['warn'].get('raised') is None
                      else row['warn']['raised'][:40])))

with open(OUT, 'w', encoding='cp1252') as fh:
    json.dump({'label': LABEL, 'lumenairy_file': lumenairy.__file__,
               'route': ROUTE, 'rows': rows}, fh, indent=1, default=repr)
print('WROTE', OUT)
