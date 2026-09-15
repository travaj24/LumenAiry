"""Claim (7): the arbiter costs ONE extra rasterisation and never a second
trace -- structurally, and on the clock.

Structural (on HEAD only): count the calls to every module-level entry the
branch sum and the completion make per call -- the launch trace, the KMAH free
leg and the meridional fold trace -- with the arbiter ON and OFF, and check
that the half-pitch RENDER is present in the diagnostics at exactly
``(2 N, 2 N)`` (which is what "one extra rasterisation" means).

Clock: round-1 tree against HEAD, best of ``reps``, same box.

Usage:  python vcost.py <out.json> <ref_tree> <head_tree>
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

STRUCT = r'''
import json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.environ['PROBE_DIR'])
import lumenairy
assert os.path.abspath(lumenairy.__file__).startswith(
    os.path.abspath(os.environ['TREE']))
import vfixtures as FX
from lumenairy.elements import _lens_traced_multibranch as MB
from lumenairy.elements import _lens_traced_uniform as U

COUNT = {}
def wrap(mod, name):
    fn = getattr(mod, name)
    COUNT[name] = 0
    def w(*a, **k):
        COUNT[name] += 1
        return fn(*a, **k)
    setattr(mod, name, w)
    return fn

res = {}
for name in ('_trace_launch_grid', '_kmah_free_leg'):
    if hasattr(MB, name):
        wrap(MB, name)
for name in ('_trace_meridional_fold',):
    if hasattr(U, name):
        wrap(U, name)

fx = FX.FIXTURES[os.environ['FIXTURE']]
z = float(os.environ['Z_UM']) * 1e-6
E = FX.input_field(fx)
kw = dict(prescription=fx['prescription'], wavelength=fx['wavelength'],
          dx=fx['dx'], output_plane_distance=z, return_diagnostics=True)

for tag, arb in (('off', False), ('on', True)):
    for k in COUNT: COUNT[k] = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        Emb, d = MB._multibranch_render(E, pixel_halving_arbiter=arb, **kw)
    half = d.get('pixel_halved_field')
    res['mb_' + tag] = dict(counts=dict(COUNT),
                            half_shape=(None if half is None
                                        else list(np.asarray(half).shape)),
                            coarse_shape=list(np.asarray(Emb).shape),
                            continuity=d.get('pixel_continuity'),
                            decision=d.get('pixel_continuity_decision'))

for k in COUNT: COUNT[k] = 0
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    try:
        Eu, ud = U.apply_real_lens_traced_uniform(E, **kw)
        res['uni'] = dict(counts=dict(COUNT),
                          continuity=ud.get('pixel_continuity'),
                          of=ud.get('pixel_continuity_of'),
                          leaked_half=('pixel_halved_field' in ud))
    except RuntimeError as exc:
        res['uni'] = dict(counts=dict(COUNT), refused=str(exc)[:120])

print('@@JSON@@' + json.dumps(res))
'''

TIME = r'''
import json, os, sys, time, warnings
import numpy as np
sys.path.insert(0, os.environ['PROBE_DIR'])
import lumenairy
assert os.path.abspath(lumenairy.__file__).startswith(
    os.path.abspath(os.environ['TREE']))
import vfixtures as FX
from lumenairy.elements import _lens_traced_multibranch as MB
from lumenairy.elements import _lens_traced_uniform as U

cases = json.loads(sys.argv[1])
reps = int(os.environ.get('REPS', '3'))
out = {'lumenairy': lumenairy.__file__, 'version': lumenairy.__version__,
       'rows': {}}
for c in cases:
    fx = FX.FIXTURES[c['fx']]
    E = FX.input_field(fx)
    kw = dict(prescription=fx['prescription'], wavelength=fx['wavelength'],
              dx=fx['dx'], output_plane_distance=c['z'] * 1e-6)
    for route, fn in (('uni', U.apply_real_lens_traced_uniform),
                      ('mb', MB.apply_real_lens_traced_multibranch)):
        best = None
        err = None
        for _ in range(reps):
            t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fn(E, **kw)
            except Exception as exc:
                err = type(exc).__name__
                break
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
        out['rows'][f"{c['id']}_{route}"] = dict(best_s=best, error=err)
print('@@JSON@@' + json.dumps(out))
'''


def child(code, tree, probe_dir, args=(), env_extra=None):
    env = dict(os.environ)
    env.update(PYTHONPATH=tree, TREE=tree, PROBE_DIR=probe_dir,
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', LUMENAIRY_MEM_BUDGET_MB='2048')
    env.update(env_extra or {})
    p = subprocess.run([sys.executable, '-c', code, *args], cwd=tree, env=env,
                       capture_output=True, text=True, timeout=7200)
    for ln in p.stdout.splitlines():
        if ln.startswith('@@JSON@@'):
            return json.loads(ln[len('@@JSON@@'):])
    raise SystemExit(f'no JSON from {tree}:\n{p.stdout[-1500:]}\n'
                     f'{p.stderr[-3000:]}')


CASES = [dict(id='W_fold', fx='W', z=4870), dict(id='W_vertex', fx='W', z=0),
         dict(id='X_fold', fx='X', z=960), dict(id='Y_fold', fx='Y', z=1970),
         dict(id='Z_fallback', fx='Z', z=3430),
         dict(id='C_far', fx='C', z=16000)]


def main():
    out_path, ref, head = sys.argv[1], sys.argv[2], sys.argv[3]
    probe_dir = os.path.dirname(os.path.abspath(__file__))
    struct = {}
    for c in CASES[:4]:
        struct[c['id']] = child(STRUCT, head, probe_dir,
                                env_extra=dict(FIXTURE=c['fx'],
                                               Z_UM=str(c['z'])))
        print(c['id'], json.dumps(struct[c['id']])[:300], flush=True)
    t_ref = child(TIME, ref, probe_dir, args=(json.dumps(CASES),))
    t_head = child(TIME, head, probe_dir, args=(json.dumps(CASES),))
    ratios = {}
    for k, v in t_head['rows'].items():
        a = t_ref['rows'].get(k, {}).get('best_s')
        b = v.get('best_s')
        ratios[k] = dict(ref_s=a, head_s=b,
                         ratio=(b / a if (a and b) else None),
                         head_error=v.get('error'),
                         ref_error=t_ref['rows'].get(k, {}).get('error'))
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(dict(structural=struct, ref=t_ref['lumenairy'],
                       head=t_head['lumenairy'], timing=ratios), f, indent=1)
    print(json.dumps(ratios, indent=1))


if __name__ == '__main__':
    main()
