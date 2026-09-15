"""Task F -- ARCHIVE-TO-ARCHIVE bit identity of the traced-uniform and
multibranch fields between the audit base ``96cb2096`` and the WP-B7c head.

``git archive`` extracts each commit's ``lumenairy/`` into its own scratch
tree; a CHILD process is then run with ``cwd`` AND ``PYTHONPATH`` set to that
tree, and ``lumenairy.__file__`` is ASSERTED to live under it before a single
field is built (the pip -e install on this interpreter points at a different
repository on a different branch, so the assertion is not a formality).  The
digest is SHA-256 over ``numpy.ndarray.tobytes()`` of the returned complex
array, so it is exact.

Thirty fixtures over the four VERIFY optics and their alternate grids: a
healthy fold plane, a fallback plane, a blow-up plane and the exit vertex for
each; the branch sum alone at the same planes; the two grids; a
``caustic_band='plain'`` reconstruction; a complex64 input; the two planes in
the ratio band WP-B7c reports as empty; and the two planes the head refuses.

Usage: python bitid.py <scratch_dir> <base_sha> <head_sha> <out_prefix>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

WORKER = r'''
import hashlib, json, os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX
import lumenairy
_tree = os.path.dirname(os.path.abspath(__file__))
assert os.path.abspath(lumenairy.__file__).startswith(_tree), (
    'wrong tree: ' + lumenairy.__file__ + ' not under ' + _tree)
from lumenairy.elements._lens_traced_multibranch import (
    apply_real_lens_traced_multibranch)
from lumenairy.elements._lens_traced_uniform import (
    apply_real_lens_traced_uniform)

CASES = json.load(open(sys.argv[1]))
out = {'lumenairy_file': lumenairy.__file__,
       'lumenairy_version': lumenairy.__version__,
       'python': sys.version.split()[0], 'numpy': np.__version__,
       'cases': {}}
for c in CASES:
    fx = FX.FIXTURES[c['fixture']]
    dtype = np.complex64 if c.get('dtype') == 'complex64' else np.complex128
    E = FX.gauss(fx['N'], fx['dx'], fx['w0'], dtype=dtype)
    kw = dict(prescription=fx['prescription'], wavelength=fx['wavelength'],
              dx=fx['dx'], output_plane_distance=c['z'])
    for k in ('caustic_band', 'ray_subsample', 'min_area_ratio'):
        if k in c:
            kw[k] = c[k]
    fn = (apply_real_lens_traced_multibranch if c['member'] == 'multibranch'
          else apply_real_lens_traced_uniform)
    rec = {'member': c['member'], 'fixture': c['fixture'], 'z': c['z']}
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            Eo = np.asarray(fn(E, **kw))
        rec['digest'] = hashlib.sha256(Eo.tobytes()).hexdigest()
        rec['dtype'] = str(Eo.dtype)
        rec['shape'] = list(Eo.shape)
        rec['power'] = float(np.sum(np.abs(Eo) ** 2)) * fx['dx'] ** 2
        rec['n_warnings'] = len(w)
        rec['error'] = None
    except Exception as e:
        rec['digest'] = None
        rec['error'] = type(e).__name__ + ': ' + str(e)[:200]
    out['cases'][c['id']] = rec
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print('WORKER_DONE', lumenairy.__file__)
'''


def cases():
    out = []

    def add(i, fixture, member, z, **kw):
        out.append(dict(id=i, fixture=fixture, member=member, z=z, **kw))

    plan = {
        'P': dict(fold=888e-6, fallback=1000e-6, blow=1016e-6,
                  gap=1015.20e-6, refused=1015.34e-6),
        'Q': dict(fold=5680e-6, fallback=5400e-6, blow=5466e-6,
                  gap=5420e-6, refused=5460e-6),
        'M': dict(fold=2242.17e-6, fallback=2330e-6, blow=2343e-6),
        'S': dict(fold=3214.78e-6, fallback=3253.91e-6, blow=3274.02e-6),
    }
    for nm, p in plan.items():
        add(f'{nm}_uni_fold', nm, 'uniform', p['fold'])
        add(f'{nm}_uni_fallback', nm, 'uniform', p['fallback'])
        add(f'{nm}_uni_vertex', nm, 'uniform', 0.0)
        add(f'{nm}_mb_fold', nm, 'multibranch', p['fold'])
        add(f'{nm}_mb_blow', nm, 'multibranch', p['blow'])
    for nm in ('P', 'Q', 'M', 'S'):
        alt = nm + '_alt'
        add(f'{alt}_uni_fold', alt, 'uniform', plan[nm]['fold'])
    add('P_mb_plain', 'P', 'multibranch', 888e-6, caustic_band='plain')
    add('S_uni_c64', 'S', 'uniform', 3214.78e-6, dtype='complex64')
    add('P_uni_gap', 'P', 'uniform', plan['P']['gap'])
    add('Q_uni_gap', 'Q', 'uniform', plan['Q']['gap'])
    add('P_uni_refused', 'P', 'uniform', plan['P']['refused'])
    add('Q_uni_refused', 'Q', 'uniform', plan['Q']['refused'])
    add('P_uni_blow', 'P', 'uniform', plan['P']['blow'])
    add('M_mb_sub4', 'M', 'multibranch', 2242.17e-6, ray_subsample=4)
    return out


def build_tree(scratch, sha):
    tree = os.path.join(scratch, 'tree_' + sha[:8])
    if os.path.isdir(tree):
        shutil.rmtree(tree)
    os.makedirs(tree)
    tar = os.path.join(scratch, sha[:8] + '.tar')
    subprocess.run(['git', '-C', _REPO, 'archive', sha, 'lumenairy',
                    '-o', tar], check=True)
    subprocess.run(['tar', '-xf', tar, '-C', tree], check=True)
    os.remove(tar)
    shutil.copy(os.path.join(_HERE, 'fixtures.py'), tree)
    with open(os.path.join(tree, '_bitid_worker.py'), 'w') as fh:
        fh.write(WORKER)
    return tree


def run(tree, cases_path, out_path):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=tree)
    r = subprocess.run([sys.executable, '_bitid_worker.py', cases_path,
                        out_path], cwd=tree, env=env,
                       capture_output=True, text=True)
    print(r.stdout[-400:], r.stderr[-800:])
    r.check_returncode()
    return json.load(open(out_path))


def main(argv):
    scratch, base, head, prefix = argv[1], argv[2], argv[3], argv[4]
    os.makedirs(scratch, exist_ok=True)
    cs = cases()
    cases_path = os.path.join(scratch, 'cases.json')
    json.dump(cs, open(cases_path, 'w'), indent=1)
    res = {}
    for tag, sha in (('base', base), ('head', head)):
        tree = build_tree(scratch, sha)
        res[tag] = run(tree, cases_path,
                       os.path.join(scratch, f'{tag}.json'))
        res[tag]['sha'] = sha
    same, moved = [], []
    for c in cs:
        i = c['id']
        a = res['base']['cases'][i]
        b = res['head']['cases'][i]
        if a['digest'] is not None and a['digest'] == b['digest']:
            same.append(i)
        else:
            moved.append({'id': i, 'base_digest': a['digest'],
                          'head_digest': b['digest'],
                          'base_error': a['error'], 'head_error': b['error'],
                          'base_power': a.get('power'),
                          'head_power': b.get('power'),
                          'base_warnings': a.get('n_warnings'),
                          'head_warnings': b.get('n_warnings')})
    summary = {'n_cases': len(cs), 'n_identical': len(same),
               'n_moved': len(moved), 'identical': same, 'moved': moved,
               'base': {k: res['base'][k] for k in
                        ('sha', 'lumenairy_file', 'lumenairy_version',
                         'python', 'numpy')},
               'head': {k: res['head'][k] for k in
                        ('sha', 'lumenairy_file', 'lumenairy_version',
                         'python', 'numpy')}}
    with open(prefix + '.json', 'w') as fh:
        json.dump({'summary': summary, 'base': res['base'],
                   'head': res['head']}, fh, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main(sys.argv)
