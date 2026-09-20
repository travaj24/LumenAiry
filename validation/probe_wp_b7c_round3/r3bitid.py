"""Bit identity, ARCHIVE to ARCHIVE, over the round-3 fixture matrix.

R3-3 changes where the half-pitch render's pixel CENTRES sit.  That render is
the arbiter's alone -- it never reaches the caller and the returned field is
built from the coarse render -- so every returned field must be bit-identical
across the change, and the only legitimate movement is a plane whose READING
crossed the bar.  This measures that, rather than asserting it.

Both trees are ``git archive`` exports (no ``.git``, no working-tree state,
no stale ``__pycache__``), a CHILD process per tree with ``cwd`` and
``PYTHONPATH`` set to it, ``lumenairy.__file__`` ASSERTED under it, and
``LUMENAIRY_MEM_BUDGET_MB`` PINNED -- VERIFY-WP-B12 D-4: unpinned digests are
chunking-dependent, so an unpinned comparison can read "moved" for a reason
that has nothing to do with the change under test.

The fixture module comes from the CURRENT worktree in both runs (it is a
probe, not library code), so the two children differ only in ``lumenairy/``.

Usage:  python r3bitid.py <out.json> <tree_a> <tree_b>
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
    os.path.abspath(os.environ['TREE'])), (lumenairy.__file__,
                                           os.environ['TREE'])
import r3fixtures as FX
from lumenairy.elements import _lens_traced_uniform as U
from lumenairy.elements import _lens_traced_multibranch as MBm

cases = json.loads(sys.argv[1])
out = {'lumenairy': lumenairy.__file__, 'version': lumenairy.__version__,
       'python': sys.version.split()[0], 'numpy': np.__version__,
       'mem_budget': os.environ.get('LUMENAIRY_MEM_BUDGET_MB'),
       'bar': float(U._MB_PIXEL_CONTINUITY_MAX), 'rows': {}}
for c in cases:
    fx = FX.FIXTURES[c['fx']]
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
                                'msg': str(exc)[:200]}
print('@@JSON@@' + json.dumps(out))
'''

#: (fixture, healthy fold z, blown-up z, fallback z, gap-edge z) in um,
#: read off this round's own band scans.
SPEC = [
    ('V', 1679, 1751, 1870, 1727),
    ('S', 3231, 3272, 3293, 3258),
    ('M', 2290, 2379, 2432, 2361),
    ('Q', 5669, 5988, 6148, 5935),
    ('F_alt', 1070, 1126, 1182, 1104),
    ('A', 2805, 2978, 3108, 2935),
    ('K', 2060, 2179, 2250, 2155),
    ('G', 979, 1108, 1180, 1082),
    ('P', 914, 1092, 1184, 1056),
    ('W', 4767, 5011, 5158, 4962),
    ('X', 922, 1145, 1270, 1100),
    ('Y', 1946, 2098, 2190, 2068),
    ('Z', 3466, 3488, 3526, 3480),
    ('C', 16000, 23200, 24000, 22800),
    ('AS', 3059, 4256, 4598, 4085),
    ('MC', 2689, 3356, 3578, 3244),
    ('HN', 1961, 2271, 2375, 2219),
]


def build_cases():
    cs = []
    for nm, z_ok, z_bad, z_fb, z_edge in SPEC:
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
    for nm, z in (('W_alt', 4767), ('X_alt', 922), ('V_alt', 1679),
                  ('AS_alt', 3059)):
        cs.append(dict(id=f'{nm}_uni_ok', fx=nm, z=z, route='uni'))
    return cs


def run(tree, cases, probe_dir, budget='2048'):
    env = dict(os.environ)
    env.update(PYTHONPATH=tree, TREE=tree, PROBE_DIR=probe_dir,
               LUMENAIRY_MEM_BUDGET_MB=budget, OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    p = subprocess.run([sys.executable, '-c', CHILD, json.dumps(cases)],
                       cwd=tree, env=env, capture_output=True, text=True,
                       timeout=14400)
    for ln in p.stdout.splitlines():
        if ln.startswith('@@JSON@@'):
            return json.loads(ln[len('@@JSON@@'):])
    raise SystemExit(f'child on {tree} produced no JSON:\n'
                     f'{p.stdout[-2000:]}\n{p.stderr[-4000:]}')


def main():
    out_path, tree_a, tree_b = sys.argv[1], sys.argv[2], sys.argv[3]
    probe_dir = os.path.dirname(os.path.abspath(__file__))
    cases = build_cases()
    print(f'{len(cases)} cases', flush=True)
    ra = run(tree_a, cases, probe_dir)
    print('A', ra['lumenairy'], ra['version'], 'bar', ra['bar'], flush=True)
    rb = run(tree_b, cases, probe_dir)
    print('B', rb['lumenairy'], rb['version'], 'bar', rb['bar'], flush=True)
    ident = moved = newref = unref = botherr = 0
    rows = []
    for c in cases:
        a, b = ra['rows'][c['id']], rb['rows'][c['id']]
        if 'error' in a and 'error' in b:
            verdict, botherr = 'both_refused', botherr + 1
        elif 'error' in b and 'error' not in a:
            verdict, newref = 'NEWLY_REFUSED', newref + 1
        elif 'error' in a and 'error' not in b:
            verdict, unref = 'NEWLY_RETURNED', unref + 1
        elif a['sha256'] == b['sha256']:
            verdict, ident = 'identical', ident + 1
        else:
            verdict, moved = 'MOVED', moved + 1
        rows.append(dict(id=c['id'], verdict=verdict, a=a, b=b))
    summary = dict(n=len(cases), identical=ident, moved=moved,
                   newly_refused=newref, newly_returned=unref,
                   both_refused=botherr,
                   tree_a={k: v for k, v in ra.items() if k != 'rows'},
                   tree_b={k: v for k, v in rb.items() if k != 'rows'})
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(dict(summary=summary, rows=rows), f, indent=1)
    print(json.dumps({k: v for k, v in summary.items()
                      if k not in ('tree_a', 'tree_b')}))
    for r in rows:
        if r['verdict'] != 'identical':
            print(' ', r['id'], r['verdict'], r['b'].get('error', ''),
                  str(r['b'].get('msg', ''))[:90])


if __name__ == '__main__':
    main()
