"""CLAIM 5 (b) -- can the Collins leg's Sziklas fallback recurse?

Drives every geometry through the INSTRUMENTED tree (a depth counter wrapped
round ``_collins_carrier_leg`` and ``propagate_carrier_referenced``) and
asserts the leg is never re-entered.

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
assert hasattr(C, '_C3_MAXDEPTH'), 'this tree is NOT instrumented'


def gauss(N, dx, w):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)


def tilted(N, dx, w, f0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return (np.exp(-(X ** 2 + Y ** 2) / (w * w))
            * np.exp(2j * np.pi * f0 * X)).astype(np.complex128)


_WL, _N, _DX, _W = 1.064e-6, 1024, 4.0e-6, 0.30e-3
env = gauss(_N, _DX, _W)
env_small = gauss(64, 8e-6, 60e-6)
env_hi = tilted(_N, _DX, 1.8e-3, 1.0e5)

WL3 = 1.31e-6
W0, ZF = 4e-6, 30e-3
ZR = np.pi * W0 ** 2 / WL3
RIN = -(ZF + ZR ** 2 / ZF)
WIN = W0 * np.sqrt(1 + (ZF / ZR) ** 2)
N3 = 2048
DX3 = (8.0 * WIN) / N3
env3 = gauss(N3, DX3, WIN)

CASES = []
for gk in ('auto', 'fresnel', 'exact'):
    CASES += [
        ('collimated-%s' % gk, env, np.inf, 5e-3, _WL, _DX, dict(gap_kernel=gk)),
        ('converging-%s' % gk, env, -40e-3, 5e-3, _WL, _DX, dict(gap_kernel=gk)),
        ('back-prop-%s' % gk, env, -40e-3, -3e-3, _WL, _DX, dict(gap_kernel=gk)),
        ('diverging-%s' % gk, env, 80e-3, 12e-3, _WL, _DX, dict(gap_kernel=gk)),
        ('A=0-%s' % gk, env, -40e-3, 40e-3, _WL, _DX, dict(gap_kernel=gk)),
        ('near-focus-%s' % gk, env, -40e-3, 39.99e-3, _WL, _DX,
         dict(gap_kernel=gk)),
        ('past-focus-%s' % gk, env3, RIN, 45e-3, WL3, DX3, dict(gap_kernel=gk)),
        ('far-past-focus-%s' % gk, env3, RIN, 60e-3, WL3, DX3,
         dict(gap_kernel=gk)),
        ('zero-length-%s' % gk, env, -40e-3, 0.0, _WL, _DX,
         dict(gap_kernel=gk)),
        ('small-collimated-%s' % gk, env_small, np.inf, 5e-3, 633e-9, 8e-6,
         dict(gap_kernel=gk)),
        ('hi-angle-flat-%s' % gk, env_hi, -2.2222222222222222e-3, 2e-3, _WL,
         _DX, dict(gap_kernel=gk)),
    ]
# astigmatic: 'exact' is refused on that carrier, so only the two kernels
for gk in ('auto', 'fresnel'):
    CASES += [
        ('astigmatic-%s' % gk, env, (-40e-3, -55e-3), 5e-3, _WL, _DX,
         dict(gap_kernel=gk)),
        ('astig-past-focus-%s' % gk, env, (-40e-3, -55e-3), 60e-3, _WL, _DX,
         dict(gap_kernel=gk)),
    ]
CASES += [
    ('tilted', env, -40e-3, 5e-3, _WL, _DX,
     dict(tilt=(0.02, -0.01), gap_kernel='exact')),
    ('tilted-past-focus', env3, RIN, 45e-3, WL3, DX3,
     dict(tilt=(0.01, 0.0), gap_kernel='exact')),
    ('caller-dx_out', env, -40e-3, 5e-3, _WL, _DX, dict(dx_out=1e-6)),
    ('caller-carrier_out', env, -40e-3, 5e-3, _WL, _DX,
     dict(carrier_out=np.inf)),
    ('caller-dx_out+carrier_out', env, -40e-3, 5e-3, _WL, _DX,
     dict(dx_out=1e-6, carrier_out=-1.0)),
]

rows = []
for name, E, R, z, wl, dx, kw in CASES:
    C._c3_reset()
    diag = {}
    rec_kw = dict(kw)
    dxo = rec_kw.pop('dx_out', None)
    cout = rec_kw.pop('carrier_out', None)
    err = None
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            C._collins_carrier_leg(E, R, z, wl, dx, dx,
                                   on_collins_sampling='ignore', diag=diag,
                                   dx_out=dxo, carrier_out=cout, **rec_kw)
    except RecursionError as exc:                            # noqa: BLE001
        err = 'RecursionError: ' + str(exc)[:80]
    except Exception as exc:                                 # noqa: BLE001
        err = type(exc).__name__ + ': ' + str(exc)[:120]
    rows.append({'case': name, 'leg_calls': C._C3_CALLS,
                 'leg_max_depth': C._C3_MAXDEPTH,
                 'pcr_max_depth': C._C3_PCR_MAXDEPTH,
                 'form': diag.get('collins_form'),
                 'k1': (None if diag.get('collins_k1') is None
                        else max(diag['collins_k1'])),
                 'k3': (None if diag.get('collins_k3') is None
                        else max(diag['collins_k3'])),
                 'flat': diag.get('collins_flat_reference'),
                 'error': err})
    print('%-32s calls=%d depth=%d pcr=%d form=%-7s err=%s'
          % (name, C._C3_CALLS, C._C3_MAXDEPTH, C._C3_PCR_MAXDEPTH,
             diag.get('collins_form'), err))

# the PUBLIC entry, default transport, same geometries
pub = []
for name, E, R, z, wl, dx, kw in CASES:
    if 'dx_out' in kw or 'carrier_out' in kw:
        continue
    C._c3_reset()
    err = None
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            C.propagate_carrier_referenced(E, R, z, wl, dx,
                                           on_collins_sampling='ignore', **kw)
    except RecursionError as exc:                            # noqa: BLE001
        err = 'RecursionError'
    except Exception as exc:                                 # noqa: BLE001
        err = type(exc).__name__ + ': ' + str(exc)[:120]
    pub.append({'case': name, 'leg_calls': C._C3_CALLS,
                'leg_max_depth': C._C3_MAXDEPTH,
                'pcr_max_depth': C._C3_PCR_MAXDEPTH, 'error': err})

worst_leg = max(r['leg_max_depth'] for r in rows + pub)
worst_pcr = max(r['pcr_max_depth'] for r in rows + pub)
out = {'label': LABEL, 'lumenairy_file': lumenairy.__file__,
       'legs': rows, 'public': pub,
       'worst_leg_depth': worst_leg, 'worst_pcr_depth': worst_pcr}
with open(OUT, 'w', encoding='cp1252') as fh:
    json.dump(out, fh, indent=1, default=repr)
print('WORST leg depth', worst_leg, ' WORST pcr depth', worst_pcr)
print('WROTE', OUT)
