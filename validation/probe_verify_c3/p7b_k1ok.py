"""CLAIM 7 (d, e, f) -- a chain whose readout K1 <= 1, so the ONE-STEP Collins
readout is the route the resolution would take, and the stop-plane key has to
WIN over it.  Also: every other focus_readout key on that same chain.

    python <this> <tree_root> <label> <out.json>
"""
import hashlib
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
HAS_K1_COUNTER = hasattr(C, '_C3_K1_CALLS')
print('k1 counter present =', HAS_K1_COUNTER)

WL = 1.064e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent')
N, DX, W, F = 512, 8e-6, 0.30e-3, 300e-3


def _surf(r, gb, ga):
    return {'radius': r, 'glass_before': gb, 'glass_after': ga, 'conic': 0.0,
            'radius_y': None, 'conic_y': None, 'aspheric_coeffs': None,
            'aspheric_coeffs_y': None}


PRESC = {'name': 'p', 'aperture_diameter': 6e-3, 'thicknesses': [3e-3],
         'surfaces': [_surf(2 * F, 'air', 'N-BK7'),
                      _surf(-2 * F, 'N-BK7', 'air')]}
x = (np.arange(N) - N // 2) * DX
XX, YY = np.meshgrid(x, x, indexing='ij')
ENV = np.exp(-(XX ** 2 + YY ** 2) / (W * W)).astype(np.complex128)
GROUPS = [{'prescription': PRESC, 'gap_before': 10e-3}]
KW = dict(r_in=np.inf, ray_subsample=16, n_workers=1, traced_kwargs=_TKW,
          final_leg='paraxial')


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(
        a, dtype=np.complex128).tobytes()).hexdigest()


def run(transport, fr, fd):
    if HAS_K1_COUNTER:
        C._c3_reset_k1()
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            r = C.propagate_traced_carrier_chain(
                ENV, GROUPS, WL, DX, final_distance=fd,
                focus_readout=dict(fr), transport=transport, **KW)
        return {'ok': True, 'field': np.asarray(r.field), 'stages': r.stages,
                'dx': r.dx,
                'k1_calls': (C._C3_K1_CALLS if HAS_K1_COUNTER else None),
                'warnings': [str(w.message)[:140] for w in rec]}
    except Exception as exc:                                 # noqa: BLE001
        return {'ok': False, 'exc': type(exc).__name__, 'msg': str(exc)[:400],
                'k1_calls': (C._C3_K1_CALLS if HAS_K1_COUNTER else None)}


FR0 = dict(dx_out=0.5e-6, N_out=64)
out = {'label': LABEL, 'lumenairy_file': lumenairy.__file__,
       'k1_counter': HAS_K1_COUNTER}

# --- find the final distances whose K1 is on each side of 1 ---------------
scan = {}
for fd_mm in (8.0, 20.0, 30.0, 40.0, 50.0, 60.0):
    r = run('collins', FR0, fd_mm * 1e-3)
    if r['ok']:
        st = r['stages'][-1]
        scan['%g' % fd_mm] = {'k1': st.get('readout_route_k1'),
                              'route': st.get('readout_route'),
                              'reason': st.get('readout_route_reason'),
                              'k1_calls': r['k1_calls']}
    else:
        scan['%g' % fd_mm] = {'raised': r['exc'], 'msg': r['msg'][:140]}
    print('fd=%-5g %s' % (fd_mm, scan['%g' % fd_mm]))
out['k1_scan'] = scan
FD_OK = None
for kk, vv in scan.items():
    if vv.get('route') == 'collins':
        FD_OK = float(kk) * 1e-3
        break
out['fd_collins_route'] = FD_OK
print('FD with the collins route =', FD_OK)
if FD_OK is None:
    with open(OUT, 'w', encoding='cp1252') as fh:
        json.dump(out, fh, indent=1, default=repr)
    sys.exit('no final_distance on this fixture reaches K1 <= 1')

# --- 7d: the stop-plane key must WIN over K1 <= 1, and K1 must not be run --
base_c = run('collins', FR0, FD_OK)
base_s = run('sziklas', FR0, FD_OK)
out['baseline'] = {
    'collins_route': base_c['stages'][-1].get('readout_route'),
    'collins_reason': base_c['stages'][-1].get('readout_route_reason'),
    'collins_k1': base_c['stages'][-1].get('readout_route_k1'),
    'collins_k1_calls': base_c['k1_calls'],
    'sha_collins': sha(base_c['field']), 'sha_sziklas': sha(base_s['field']),
    'differ': sha(base_c['field']) != sha(base_s['field']),
    'peak_collins': float(np.abs(base_c['field']).max() ** 2),
    'peak_sziklas': float(np.abs(base_s['field']).max() ** 2)}
print('baseline:', out['baseline'])

for nm, extra in (('standoff', {'standoff': 2e-3}),
                  ('on_focus_containment', {'on_focus_containment': 'warn'}),
                  ('both', {'standoff': 2e-3,
                            'on_focus_containment': 'warn'})):
    rc = run('collins', dict(FR0, **extra), FD_OK)
    rs = run('sziklas', dict(FR0, **extra), FD_OK)
    d = {'raised': None if rc['ok'] else rc['exc']}
    if rc['ok']:
        st = rc['stages'][-1]
        d.update(route=st.get('readout_route'),
                 reason=st.get('readout_route_reason'),
                 route_k1=st.get('readout_route_k1'),
                 k1_calls=rc['k1_calls'],
                 sha=sha(rc['field']))
    if rc['ok'] and rs['ok']:
        d['sha_sziklas'] = sha(rs['field'])
        d['field_bit_identical_to_sziklas'] = (
            sha(rc['field']) == sha(rs['field']))
        d['equals_collins_baseline'] = sha(rc['field']) == out[
            'baseline']['sha_collins']
        ek, dk = [], []
        for sa, sb in zip(rc['stages'], rs['stages']):
            for k in set(sa) | set(sb):
                if k not in sa or k not in sb:
                    ek.append((k, 'collins' if k in sa else 'sziklas'))
                else:
                    va, vb = sa[k], sb[k]
                    same = (np.array_equal(va, vb)
                            if isinstance(va, np.ndarray) else va == vb)
                    if not same:
                        dk.append((k, repr(va)[:60], repr(vb)[:60]))
        d['stage_keys_only_in_one'] = sorted(set(ek))
        d['stage_values_that_differ'] = dk
    out['stopkey_' + nm] = d
    print(nm, '->', {k: v for k, v in d.items()
                     if k not in ('sha', 'sha_sziklas')})

# --- 7f: every other focus_readout key on the COLLINS route ---------------
PROBE = {
    'bandlimit': False,
    'centre_out': (1e-6, 0.0),
    'window_factor': 4.0,
    'n_fine_cap': 4096,
    'on_n_fine_cap': 'warn',
    'max_fine_launch_points': 100000,
    'ram_budget': 8.0,
    'dx_fine': 1e-7,
    'N_fine': 512,
    'on_readout_window': 'warn',
    'readout_window_tol': 0.5,
    'on_replica': 'ignore',
    'replica_fill': 'zero',
}
rows = {}
for k, v in PROBE.items():
    row = {}
    for tr, base in (('sziklas', base_s), ('collins', base_c)):
        r = run(tr, dict(FR0, **{k: v}), FD_OK)
        if not r['ok']:
            row[tr] = {'raised': r['exc'], 'msg': r['msg'][:200]}
        else:
            row[tr] = {'raised': None,
                       'identical_to_no_key': sha(r['field']) == sha(
                           base['field']),
                       'route': r['stages'][-1].get('readout_route'),
                       'reason': r['stages'][-1].get('readout_route_reason'),
                       'n_warn': len(r['warnings'])}
    rows[k] = row
    print('%-24s sziklas: %-52s collins: %s'
          % (k, str(row['sziklas'])[:52], str(row['collins'])[:110]))
out['other_keys_on_k1ok_chain'] = rows

with open(OUT, 'w', encoding='cp1252') as fh:
    json.dump(out, fh, indent=1, default=repr)
print('WROTE', OUT)
