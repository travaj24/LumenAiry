"""VERIFY-WP-B7c round 3 -- bit identity, ARCHIVE to ARCHIVE (claim 7).

Round 3 claims 157 cases over seventeen optics with 116 identical, 0 moved,
0 newly refused, 0 newly returned, 41 refused on both.  This is my own matrix
on my own optics, run the same way and for the same reason:

* both trees are ``git archive`` EXPORTS (no ``.git``, no working-tree state,
  no stale ``__pycache__``), so nothing a worktree carries can leak between
  them -- and a worktree's ``.git`` is a Windows gitdir pointer WSL cannot
  resolve, which is a second reason not to run this inside one;
* a CHILD PROCESS per tree with ``cwd`` and ``PYTHONPATH`` set to it, and
  ``lumenairy.__file__`` ASSERTED to live under it, so a stale ``sys.path``
  entry fails loudly instead of silently scoring one tree twice;
* ``LUMENAIRY_MEM_BUDGET_MB`` PINNED, because the library chunks by the
  budget and an unpinned digest is chunking-dependent (VERIFY-WP-B12 D-4);
* SHA-256 over ``ndarray.tobytes()`` plus the dtype and shape, and the
  DECISION (returned / refused / the exception's class) recorded beside it, so
  "0 moved" and "0 newly refused" are separate statements.

Usage (driver):   python v3bitid.py <PRE tree> <POST tree> <out.json>
Usage (worker):   python v3bitid.py --worker <case-json>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
MEM_BUDGET_MB = '2048'


def cases():
    """The matrix: for every optic a fold plane, a blow-up plane, a fallback,
    the exit vertex, the branch sum alone, ``caustic_band='plain'``,
    ``ray_subsample=4`` and a complex64 input."""
    sys.path.insert(0, _HERE)
    import v3fixtures as FX
    import v3geom as G
    import v3oracle as OR
    out = []
    for nm in ('VA', 'VA_alt', 'VX', 'VX_alt', 'VC', 'VC_alt', 'VD',
               'V', 'V_alt', 'HN', 'W', 'W_alt', 'Q'):
        fx = FX.FIXTURES[nm]
        presc, wl = fx['prescription'], fx['wavelength']
        _na, fp, fm, _h, _f = G.na_and_foci(presc, wl)
        zs_geom = np.linspace(0.55 * min(fp, fm), 1.45 * max(fp, fm), 120)
        zt = [z for z, rc, _ in G.fold_window(presc, wl, zs_geom) if rc]
        lo, hi = (min(zt), max(zt)) if zt else (0.9 * fm, 1.05 * fp)
        # derived planes: inside the fold band, at its far edge, past it,
        # short of it, and the exit vertex
        plan = [('fold', lo + 0.35 * (hi - lo), {}),
                ('foldedge', lo + 0.92 * (hi - lo), {}),
                ('blowup', hi + 0.05 * (hi - lo), {}),
                ('short', 0.55 * lo, {}),
                ('vertex', 0.0, {}),
                ('plain', lo + 0.35 * (hi - lo), {'caustic_band': 'plain'}),
                ('sub4', lo + 0.35 * (hi - lo), {'ray_subsample': 4}),
                ('c64', lo + 0.35 * (hi - lo), {'dtype': 'complex64'}),
                ('branchsum', lo + 0.35 * (hi - lo), {'fn': 'multibranch'})]
        for tag, z, kw in plan:
            out.append(dict(name=f'{nm}:{tag}', fixture=nm, z=float(z), **kw))
    del OR
    return out


def _run_case(case):
    sys.path.insert(0, _HERE)
    import warnings

    import v3fixtures as FX

    from lumenairy.elements._lens_traced_multibranch import (
        apply_real_lens_traced_multibranch,
    )
    from lumenairy.elements._lens_traced_uniform import (
        apply_real_lens_traced_uniform,
    )
    fx = FX.FIXTURES[case['fixture']]
    dtype = np.complex64 if case.get('dtype') == 'complex64' else np.complex128
    E_in = FX.input_field(fx, dtype=dtype)
    fn = (apply_real_lens_traced_multibranch if case.get('fn') == 'multibranch'
          else apply_real_lens_traced_uniform)
    kw = dict(prescription=fx['prescription'], wavelength=fx['wavelength'],
              dx=fx['dx'], output_plane_distance=float(case['z']),
              ray_subsample=int(case.get('ray_subsample', 2)),
              return_diagnostics=True)
    if 'caustic_band' in case:
        kw['caustic_band'] = case['caustic_band']
    if fn is apply_real_lens_traced_uniform:
        kw['n_fan'] = 4000
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            E, d = fn(E_in, **kw)
    except Exception as exc:                                # noqa: BLE001
        return dict(name=case['name'], decision='REFUSED',
                    exc=type(exc).__name__, msg=str(exc)[:110], digest=None)
    a = np.ascontiguousarray(np.asarray(E))
    return dict(name=case['name'], decision='returned', exc=None,
                digest=hashlib.sha256(a.tobytes()).hexdigest(),
                dtype=str(a.dtype), shape=list(a.shape),
                reading=(None if d.get('pixel_continuity') is None
                         else float(d['pixel_continuity'])),
                reason=d.get('reason'), fell_back=bool(d.get('fell_back')))


def worker(path):
    import lumenairy
    root = os.path.abspath(os.environ['V3_TREE'])
    got = os.path.abspath(lumenairy.__file__)
    if not got.lower().startswith(root.lower()):
        raise SystemExit(f'lumenairy is {got}, not under {root}')
    with open(path, encoding='cp1252') as f:
        cs = json.load(f)
    out = [_run_case(c) for c in cs]
    print('V3BITID ' + json.dumps(dict(lumenairy=got,
                                       python=sys.version.split()[0],
                                       numpy=np.__version__, rows=out)))


def _child(tree, case_path, py=None):
    env = dict(os.environ)
    env['PYTHONPATH'] = tree + os.pathsep + os.path.join(
        tree, 'validation', 'probe_verify_b7c_round3')
    env['V3_TREE'] = tree
    env['LUMENAIRY_MEM_BUDGET_MB'] = MEM_BUDGET_MB
    for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[v] = '1'
    script = os.path.join(tree, 'validation', 'probe_verify_b7c_round3',
                          'v3bitid.py')
    p = subprocess.run([py or sys.executable, script, '--worker', case_path],
                       cwd=tree, env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith('V3BITID '):
            return json.loads(line[len('V3BITID '):])
    raise SystemExit(f'worker failed in {tree}:\n{p.stdout[-3000:]}\n'
                     f'{p.stderr[-3000:]}')


def main():
    if sys.argv[1] == '--worker':
        return worker(sys.argv[2])
    pre, post, out = sys.argv[1], sys.argv[2], sys.argv[3]
    cs = cases()
    case_path = os.path.join(os.path.dirname(os.path.abspath(out)),
                             '_v3bitid_cases.json')
    with open(case_path, 'w', encoding='cp1252') as f:
        json.dump(cs, f)
    print(f'{len(cs)} cases; PRE {pre}  POST {post}', flush=True)
    A = _child(pre, case_path)
    B = _child(post, case_path)
    print('PRE  lumenairy', A['lumenairy'])
    print('POST lumenairy', B['lumenairy'])
    ia = {r['name']: r for r in A['rows']}
    ib = {r['name']: r for r in B['rows']}
    rows, tally = [], dict(identical=0, moved=0, newly_refused=0,
                           newly_returned=0, refused_both=0)
    for nm in [c['name'] for c in cs]:
        a, b = ia[nm], ib[nm]
        if a['decision'] == 'REFUSED' and b['decision'] == 'REFUSED':
            verdict = 'refused_both'
        elif a['decision'] == 'REFUSED':
            verdict = 'newly_returned'
        elif b['decision'] == 'REFUSED':
            verdict = 'newly_refused'
        elif a['digest'] == b['digest']:
            verdict = 'identical'
        else:
            verdict = 'moved'
        tally[verdict] += 1
        rows.append(dict(name=nm, verdict=verdict, pre=a, post=b))
        print(f'{nm:22s} {verdict}', flush=True)
    rep = dict(pre_tree=pre, post_tree=post, pre=A['lumenairy'],
               post=B['lumenairy'], mem_budget_mb=MEM_BUDGET_MB,
               n_cases=len(cs), tally=tally, rows=rows)
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(tally))
    print('wrote', out)


if __name__ == '__main__':
    main()
