"""Bit identity, tree to tree, on the VERIFIER'S OWN fixture matrix.

Claim (3) is "26 identical / 0 moved / 4 newly refused of 30" against round 1
(``250421ed``).  Re-measured here on a matrix the builder did not choose, in a
CHILD process per tree with ``cwd`` and ``PYTHONPATH`` set to it, with
``lumenairy.__file__`` ASSERTED under it, and with ``LUMENAIRY_MEM_BUDGET_MB``
PINNED -- unpinned digests are chunking-dependent (VERIFY-WP-B12 D-4), so an
unpinned comparison can read "moved" for a reason that has nothing to do with
this change.

Usage:  python vbitid.py <out.json> <tree_a> <tree_b>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

CHILD = r'''
import hashlib, json, os, sys, warnings
import numpy as np
sys.path.insert(0, os.environ['PROBE_DIR'])
import lumenairy
assert os.path.abspath(lumenairy.__file__).startswith(
    os.path.abspath(os.environ['TREE'])), (lumenairy.__file__, os.environ['TREE'])
import vfixtures as FX
from lumenairy.elements import _lens_traced_uniform as U
from lumenairy.elements import _lens_traced_multibranch as MBm

cases = json.loads(sys.argv[1])
out = {'lumenairy': lumenairy.__file__, 'version': lumenairy.__version__,
       'python': sys.version.split()[0], 'numpy': np.__version__,
       'mem_budget': os.environ.get('LUMENAIRY_MEM_BUDGET_MB'), 'rows': {}}
for c in cases:
    fx = (FX.builder(c['fx']) if c.get('builder') else FX.FIXTURES[c['fx']])
    N = c.get('N', fx['N']); dx = c.get('dx', fx['dx'])
    dt = np.complex64 if c.get('c64') else np.complex128
    E = FX.gauss(N, dx, fx['w0'], dtype=dt)
    kw = dict(prescription=fx['prescription'], wavelength=fx['wavelength'],
              dx=dx, output_plane_distance=c['z'] * 1e-6,
              ray_subsample=c.get('sub', 2),
              caustic_band=c.get('band', 'ludwig'))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            if c['route'] == 'mb':
                R = MBm.apply_real_lens_traced_multibranch(E, **kw)
            else:
                R = U.apply_real_lens_traced_uniform(E, **kw)
        A = np.asarray(R)
        out['rows'][c['id']] = {
            'sha256': hashlib.sha256(A.tobytes()).hexdigest(),
            'dtype': str(A.dtype), 'shape': list(A.shape),
            'power': float(np.sum(np.abs(A) ** 2)) * dx * dx}
    except Exception as exc:
        out['rows'][c['id']] = {'error': type(exc).__name__,
                                'msg': str(exc)[:160]}
print('@@JSON@@' + json.dumps(out))
'''


def build_cases():
    cs = []
    # (fixture, healthy fold z, blown-up z, fallback z, gap-ladder z)
    spec = [('W', 4870, 5000, 5210, 4920),
            ('X', 960, 1020, 1200, 990),
            ('Y', 1970, 2100, 2220, 2020),
            ('Z', 3430, 3490, 3550, 3510),
            ('C', 16000, 23200, 24000, 22800)]
    for nm, z_ok, z_bad, z_fb, z_edge in spec:
        cs.append(dict(id=f'{nm}_uni_ok', fx=nm, z=z_ok, route='uni'))
        cs.append(dict(id=f'{nm}_uni_bad', fx=nm, z=z_bad, route='uni'))
        cs.append(dict(id=f'{nm}_uni_fb', fx=nm, z=z_fb, route='uni'))
        cs.append(dict(id=f'{nm}_uni_edge', fx=nm, z=z_edge, route='uni'))
        cs.append(dict(id=f'{nm}_mb_ok', fx=nm, z=z_ok, route='mb'))
        cs.append(dict(id=f'{nm}_uni_vertex', fx=nm, z=0.0, route='uni'))
        cs.append(dict(id=f'{nm}_uni_plain', fx=nm, z=z_ok, route='uni',
                       band='plain'))
        cs.append(dict(id=f'{nm}_uni_sub4', fx=nm, z=z_ok, route='uni',
                       sub=4))
        cs.append(dict(id=f'{nm}_uni_c64', fx=nm, z=z_ok, route='uni',
                       c64=True))
        alt = nm + '_alt'
        cs.append(dict(id=f'{nm}_uni_alt', fx=alt, z=z_ok, route='uni'))
    # the builder's own two, including D1's plane and the round-1 refusals
    for z in (1758, 1761, 1768):
        cs.append(dict(id=f'BV_uni_{z}', fx='V', builder=True, z=z,
                       route='uni'))
    for z in (1063, 1073, 1076, 1080):
        cs.append(dict(id=f'BF_uni_{z}', fx='F_alt', builder=True, z=z,
                       route='uni'))
    return cs


def run(tree, cases, probe_dir, budget='2048'):
    env = dict(os.environ)
    env.update(PYTHONPATH=tree, TREE=tree, PROBE_DIR=probe_dir,
               LUMENAIRY_MEM_BUDGET_MB=budget, OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    p = subprocess.run([sys.executable, '-c', CHILD, json.dumps(cases)],
                       cwd=tree, env=env, capture_output=True, text=True,
                       timeout=7200)
    for ln in p.stdout.splitlines():
        if ln.startswith('@@JSON@@'):
            return json.loads(ln[len('@@JSON@@'):])
    raise SystemExit(f'child on {tree} produced no JSON:\n'
                     f'{p.stdout[-2000:]}\n{p.stderr[-3000:]}')


def main():
    out_path, tree_a, tree_b = sys.argv[1], sys.argv[2], sys.argv[3]
    probe_dir = os.path.dirname(os.path.abspath(__file__))
    cases = build_cases()
    print(f'{len(cases)} cases', flush=True)
    ra = run(tree_a, cases, probe_dir)
    print('A', ra['lumenairy'], ra['version'], flush=True)
    rb = run(tree_b, cases, probe_dir)
    print('B', rb['lumenairy'], rb['version'], flush=True)
    ident = moved = newref = unref = botherr = 0
    rows = []
    for c in cases:
        a = ra['rows'][c['id']]
        b = rb['rows'][c['id']]
        if 'error' in a and 'error' in b:
            verdict = 'both_refused'
            botherr += 1
        elif 'error' in b and 'error' not in a:
            verdict = 'NEWLY_REFUSED'
            newref += 1
        elif 'error' in a and 'error' not in b:
            verdict = 'NEWLY_RETURNED'
            unref += 1
        elif a['sha256'] == b['sha256']:
            verdict = 'identical'
            ident += 1
        else:
            verdict = 'MOVED'
            moved += 1
        rows.append(dict(id=c['id'], verdict=verdict, a=a, b=b))
    summary = dict(n=len(cases), identical=ident, moved=moved,
                   newly_refused=newref, newly_returned=unref,
                   both_refused=botherr, tree_a=ra, tree_b=rb)
    summary['tree_a'].pop('rows', None)
    summary['tree_b'].pop('rows', None)
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('tree_a', 'tree_b')}))
    for r in rows:
        if r['verdict'] not in ('identical',):
            print(' ', r['id'], r['verdict'],
                  r['b'].get('error', ''), str(r['b'].get('msg', ''))[:80])


if __name__ == '__main__':
    main()
