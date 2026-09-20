"""CLAIM 7 -- the two stop-plane focus_readout keys SELECT the Sziklas readout
on transport='collins' (and what the other focus_readout keys do).

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
TD = inspect.signature(C.propagate_traced_carrier_chain).parameters[
    'transport'].default
print('chain transport default =', TD)

WL = 1.31e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent')


def _singlet():
    return {'name': 'p', 'aperture_diameter': 14e-3, 'thicknesses': [3e-3],
            'surfaces': [
                {'radius': 60e-3, 'glass_before': 'air',
                 'glass_after': 'N-BK7', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None},
                {'radius': -60e-3, 'glass_before': 'N-BK7',
                 'glass_after': 'air', 'conic': 0.0, 'radius_y': None,
                 'conic_y': None, 'aspheric_coeffs': None,
                 'aspheric_coeffs_y': None}]}


def fixture(n=256, dx=60e-6, w=4.5e-3):
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(np.complex128)
    p = _singlet()
    return env, dx, 60e-3, [{'prescription': p, 'gap_before': 20e-3},
                            {'prescription': p, 'gap_before': 10e-3}]


ENV, DX, RIN, GROUPS = fixture()
BASE_KW = dict(r_in=RIN, ray_subsample=16, n_workers=1, traced_kwargs=_TKW,
               final_leg='paraxial')


def run(transport, fr, fd=8e-3, **extra):
    kw = dict(BASE_KW, final_distance=fd, focus_readout=dict(fr))
    kw.update(extra)
    try:
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            r = C.propagate_traced_carrier_chain(ENV, GROUPS, WL, DX,
                                                 transport=transport, **kw)
        return {'ok': True, 'field': np.asarray(r.field),
                'stages': r.stages, 'dx_out': r.dx,
                'warnings': [str(w.message)[:140] for w in rec]}
    except Exception as exc:                                 # noqa: BLE001
        return {'ok': False, 'exc': type(exc).__name__,
                'msg': str(exc)[:400]}


FR0 = dict(dx_out=0.5e-6, N_out=64)
out = {'label': LABEL, 'lumenairy_file': lumenairy.__file__,
       'chain_transport_default': TD}

# ------------------------------------------------------------------ 7a ----
KEYSETS = {
    'none': {},
    'standoff': {'standoff': 1e-3},
    'on_focus_containment': {'on_focus_containment': 'warn'},
    'both': {'standoff': 1e-3, 'on_focus_containment': 'warn'},
}
res = {}
for nm, extra in KEYSETS.items():
    for tr in ('sziklas', 'collins'):
        r = run(tr, dict(FR0, **extra))
        res[(nm, tr)] = r
        st = r['stages'][-1] if r['ok'] else None
        print('%-22s %-8s %s' % (
            nm, tr,
            ('OK route=%s reason=%s k1=%s' % (
                st.get('readout_route'), st.get('readout_route_reason'),
                st.get('readout_route_k1')))
            if r['ok'] else ('RAISED %s: %s' % (r['exc'], r['msg'][:90]))))
        out['fd8_%s_%s' % (nm, tr)] = (
            {'raised': r['exc'], 'msg': r['msg']} if not r['ok'] else
            {'raised': None,
             'route': st.get('readout_route'),
             'reason': st.get('readout_route_reason'),
             'route_k1': st.get('readout_route_k1'),
             'peak': float(np.abs(r['field']).max() ** 2),
             'n_stages': len(r['stages']),
             'warnings': r['warnings']})

# ------------------------------------------------------------------ 7b/c ---
def cmp_two(a, b):
    if not (a['ok'] and b['ok']):
        return {'one_raised': [a.get('exc'), b.get('exc')]}
    fa, fb = a['field'], b['field']
    d = {'field_bit_identical': bool(fa.shape == fb.shape
                                     and np.array_equal(fa, fb)),
         'field_relL2': (float(np.linalg.norm(fa - fb) / np.linalg.norm(fb))
                         if fa.shape == fb.shape else None),
         'dx_out_equal': a['dx_out'] == b['dx_out'],
         'n_stages_equal': len(a['stages']) == len(b['stages'])}
    extra_keys, diff_keys = [], []
    for sa, sb in zip(a['stages'], b['stages']):
        for k in set(sa) | set(sb):
            if k not in sa or k not in sb:
                extra_keys.append((k, 'a' if k in sa else 'b'))
            else:
                va, vb = sa[k], sb[k]
                same = (np.array_equal(va, vb)
                        if isinstance(va, np.ndarray)
                        else (va == vb or (isinstance(va, float)
                                           and isinstance(vb, float)
                                           and np.isnan(va) and np.isnan(vb))))
                if not same:
                    diff_keys.append((k, repr(va)[:60], repr(vb)[:60]))
    d['stage_keys_only_in_one'] = sorted(set(extra_keys))
    d['stage_values_that_differ'] = diff_keys
    d['stages_equal_after_removing_route_keys'] = (
        not diff_keys and all(k.startswith('readout_route')
                              for k, _s in set(extra_keys)))
    return d


for nm in ('standoff', 'on_focus_containment', 'both'):
    if res[(nm, 'sziklas')]['ok'] and res[(nm, 'collins')]['ok']:
        out['two_sided_' + nm] = cmp_two(res[(nm, 'collins')],
                                         res[(nm, 'sziklas')])
        print('two-sided', nm, out['two_sided_' + nm].get(
            'field_bit_identical'), out['two_sided_' + nm].get(
            'stage_keys_only_in_one'))

# ------------------------------------------------------------------ 7d ----
# find a final_distance whose readout K1 <= 1, so the stop-plane key has to
# WIN over a K1 test that would otherwise take the Collins route
scan = {}
for fd_mm in (8.0, 20.0, 25.0, 28.0, 29.0, 29.5, 30.0, 30.5, 31.0, 35.0):
    r = run('collins', dict(FR0), fd=fd_mm * 1e-3)
    if r['ok']:
        st = r['stages'][-1]
        scan['%g' % fd_mm] = {'k1': st.get('readout_route_k1'),
                              'route': st.get('readout_route'),
                              'reason': st.get('readout_route_reason')}
    else:
        scan['%g' % fd_mm] = {'raised': r['exc'], 'msg': r['msg'][:120]}
    print('fd=%-6g %s' % (fd_mm, scan['%g' % fd_mm]))
out['k1_scan'] = scan
FD_K1_OK = None
for kk, vv in scan.items():
    if vv.get('route') == 'collins':
        FD_K1_OK = float(kk) * 1e-3
        break
out['fd_with_collins_route'] = FD_K1_OK
if FD_K1_OK is not None:
    for nm, extra in KEYSETS.items():
        r = run('collins', dict(FR0, **extra), fd=FD_K1_OK)
        st = r['stages'][-1] if r['ok'] else None
        out['k1ok_%s_collins' % nm] = (
            {'raised': r['exc'], 'msg': r['msg']} if not r['ok'] else
            {'raised': None, 'route': st.get('readout_route'),
             'reason': st.get('readout_route_reason'),
             'route_k1': st.get('readout_route_k1'),
             'peak': float(np.abs(r['field']).max() ** 2)})
        rs = run('sziklas', dict(FR0, **extra), fd=FD_K1_OK)
        if r['ok'] and rs['ok']:
            out['k1ok_two_sided_' + nm] = cmp_two(r, rs)
        print('K1<=1 fd=%g %-22s -> %s' % (FD_K1_OK, nm,
                                           out['k1ok_%s_collins' % nm]))

# ------------------------------------------------------------------ 7f ----
# every other focus_readout key, on both transports, at BOTH final distances
PROBE_KEYS = {
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
    'bogus_key_typo': 1.0,
}
keyrows = {}
for fd_tag, fd in (('fd8mm', 8e-3),) + ((('fdK1ok', FD_K1_OK),)
                                        if FD_K1_OK else ()):
    base = {}
    for tr in ('sziklas', 'collins'):
        base[tr] = run(tr, dict(FR0), fd=fd)
    for k, v in PROBE_KEYS.items():
        row = {}
        for tr in ('sziklas', 'collins'):
            r = run(tr, dict(FR0, **{k: v}), fd=fd)
            b = base[tr]
            if not r['ok']:
                row[tr] = {'raised': r['exc'], 'msg': r['msg'][:200]}
            elif not b['ok']:
                row[tr] = {'baseline_raised': b.get('exc')}
            else:
                same = (r['field'].shape == b['field'].shape
                        and np.array_equal(r['field'], b['field']))
                row[tr] = {'raised': None, 'identical_to_no_key': bool(same),
                           'route': r['stages'][-1].get('readout_route'),
                           'n_warn': len(r['warnings'])}
        keyrows['%s::%s' % (fd_tag, k)] = row
        print('%-8s %-24s sziklas=%-46s collins=%s'
              % (fd_tag, k, str(row['sziklas'])[:46], str(row['collins'])[:80]))
out['other_keys'] = keyrows

with open(OUT, 'w', encoding='cp1252') as fh:
    json.dump(out, fh, indent=1, default=repr)
print('WROTE', OUT)
