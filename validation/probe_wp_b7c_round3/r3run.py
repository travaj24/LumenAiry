"""Run the round-3 scans, N at a time, one child process per fixture.

Each child is an independent ``r3scan.py`` invocation with the thread caps and
``PYTHONPATH`` on its own environment, so a crash or a memory spike takes one
fixture and not the population.  The pool width is a parameter because the
angular-spectrum oracle of the fast optics allocates a few GB per plane.

Usage:
    python r3run.py <tag> <pool> [--oracle none] [--only A,B] [--alt]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import r3fixtures as FX
import r3ladder as LAD

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.abspath(os.path.join(HERE, '..', '..'))

#: the second grid is run on these four, chosen to span the axis: the slowest
#: (``W``), the fastest (``X``), the campaign's own reference (``V``) and one
#: of this round's new optics (``AS``).
ALT_OF = ('W_alt', 'X_alt', 'V_alt', 'AS_alt')


def child(fixture, zs, out, oracle, log):
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', PYTHONPATH=TREE)
    cmd = [sys.executable, os.path.join(HERE, 'r3scan.py'), fixture,
           LAD.spec(zs), out, '--oracle', oracle]
    f = open(log, 'w', encoding='cp1252')
    return subprocess.Popen(cmd, cwd=HERE, env=env, stdout=f, stderr=f), f


def main():
    tag = sys.argv[1]
    pool = int(sys.argv[2])
    args = sys.argv[3:]
    oracle = 'none' if '--oracle' in args and 'none' in args else 'asm'
    only = None
    if '--only' in args:
        only = set(args[args.index('--only') + 1].split(','))
    want_alt = '--alt' in args
    G = LAD.geom()
    jobs = []
    names = list(FX.OPTICS) + (list(ALT_OF) if want_alt else [])
    for nm in names:
        if only and nm not in only:
            continue
        g = G.get(nm) or G.get(nm[:-4] if nm.endswith('_alt') else nm)
        if g is None or 'error' in g:
            print('skip', nm, 'no geometry')
            continue
        zs = sorted(set([round(z, 6) for z in
                         LAD.band_zs(g) + LAD.tail_zs(g)]))
        zs = [z for z in zs if z >= 0.0]
        jobs.append((nm, zs))
    print(json.dumps({nm: len(zs) for nm, zs in jobs}, indent=1))
    print('total planes', sum(len(z) for _, z in jobs), flush=True)
    running = []
    t0 = time.time()
    queue = list(jobs)
    done = []
    while queue or running:
        while queue and len(running) < pool:
            nm, zs = queue.pop(0)
            out = os.path.join(HERE, f'band_{nm}_{tag}_win.json')
            log = os.path.join(HERE, f'log_band_{nm}_{tag}_win.txt')
            p, f = child(nm, zs, out, oracle, log)
            running.append((nm, p, f, time.time(), out))
            print('start', nm, len(zs), 'planes', flush=True)
        time.sleep(3.0)
        for rec in list(running):
            nm, p, f, ts, out = rec
            if p.poll() is not None:
                f.close()
                running.remove(rec)
                done.append(dict(fixture=nm, rc=p.returncode,
                                 seconds=round(time.time() - ts, 1), out=out))
                print('done', nm, 'rc', p.returncode,
                      '%.1fs' % (time.time() - ts), flush=True)
    print('ALL DONE %.1fs' % (time.time() - t0))
    with open(os.path.join(HERE, f'runsummary_{tag}_win.json'), 'w',
              encoding='cp1252') as f:
        json.dump(done, f, indent=1)


if __name__ == '__main__':
    main()
