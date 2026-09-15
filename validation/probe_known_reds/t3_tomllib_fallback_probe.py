"""Probe -- does tests/unit/test_audit2609_a15a_packaging.py import and
collect on a Python 3.10 interpreter?

CI run 34914295323 answered "no": the unconditional ``import tomllib`` was a
collection ERROR on all five 3.10 shards, and one collection error aborts the
whole shard (``Interrupted: 1 error during collection``), so the 3.10 lane ran
zero of its ~2,900 selected tests.

No 3.10 interpreter exists on this box, so the arms below SIMULATE one with
``t3_block_tomllib.py`` (see that file for the mechanism and why it is
faithful).  Three arms:

  native        -- nothing blocked (the interpreter's own stdlib tomllib).
  sim310_tomli  -- tomllib blocked, real ``tomli`` 2.4.1 on sys.path.  This is
                   the CI 3.10 leg; ALL ELEVEN gates must run.
  sim310_none   -- tomllib blocked and no tomli.  The module must still
                   IMPORT and COLLECT; the four pyproject-reading gates skip
                   with a stated reason, the other seven still run.

Writes ``t3_tomllib_fallback_probe_<ARM>.json`` beside this file.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..'))
TARGET = 'tests/unit/test_audit2609_a15a_packaging.py'

# Which module actually supplies ``load`` under the arm, printed by a plain
# (non-pytest) child so the arm's premise is measured, not assumed.
_WHICH = (
    "import sys\n"
    "for n in %r:\n"
    "    sys.modules[n] = None\n"
    "err = None\n"
    "try:\n"
    "    import tomllib\n"
    "except ModuleNotFoundError as e:\n"
    "    err = str(e)\n"
    "    try:\n"
    "        import tomli as tomllib\n"
    "    except ModuleNotFoundError as e2:\n"
    "        tomllib = None\n"
    "        err = err + ' | ' + str(e2)\n"
    "import json\n"
    "print(json.dumps({'resolved': None if tomllib is None else tomllib.__name__,\n"
    "                  'file': None if tomllib is None else tomllib.__file__,\n"
    "                  'first_error': err,\n"
    "                  'python': sys.version.split()[0]}))\n"
)


def _env(arm, tomli_target):
    env = dict(os.environ)
    env['OMP_NUM_THREADS'] = '1'
    env['OPENBLAS_NUM_THREADS'] = '1'
    env['MKL_NUM_THREADS'] = '1'
    parts = [REPO, HERE]
    if arm == 'sim310_tomli':
        parts.insert(0, tomli_target)
    env['PYTHONPATH'] = os.pathsep.join(parts)
    if arm == 'native':
        env.pop('T3_BLOCK', None)
    elif arm == 'sim310_tomli':
        env['T3_BLOCK'] = 'tomllib'
    else:
        env['T3_BLOCK'] = 'tomllib,tomli'
    return env


def run_arm(arm, tomli_target):
    env = _env(arm, tomli_target)
    blocked = [] if arm == 'native' else env['T3_BLOCK'].split(',')
    which = subprocess.run(
        [sys.executable, '-c', _WHICH % (blocked,)],
        cwd=REPO, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        resolved = json.loads(which.stdout.strip().splitlines()[-1])
    except Exception:
        resolved = {'raw_stdout': which.stdout, 'raw_stderr': which.stderr}

    cmd = [sys.executable, '-m', 'pytest', TARGET,
           '--capture=sys', '-q', '-p', 'no:cacheprovider', '-rs']
    if arm != 'native':
        cmd += ['-p', 't3_block_tomllib']
    proc = subprocess.run(cmd, cwd=REPO, env=env, stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True)
    out = proc.stdout
    tail = out.strip().splitlines()[-14:]
    rec = {
        'arm': arm,
        'python': sys.version.split()[0],
        'blocked': blocked,
        'pytest_argv': cmd,
        'returncode': proc.returncode,
        'toml_parser_resolution': resolved,
        'collection_error': ('error during collection' in out
                             or 'ModuleNotFoundError' in out),
        'tail': tail,
    }
    path = os.path.join(HERE, 't3_tomllib_fallback_probe_%s.json' % arm)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=2)
    print('=== arm %s -> rc=%d ===' % (arm, proc.returncode))
    print('    parser:', resolved)
    for ln in tail:
        print('   ', ln)
    print('    json:', path)
    return rec


if __name__ == '__main__':
    tomli_target = sys.argv[1] if len(sys.argv) > 1 else ''
    for a in ('native', 'sim310_tomli', 'sim310_none'):
        run_arm(a, tomli_target)
