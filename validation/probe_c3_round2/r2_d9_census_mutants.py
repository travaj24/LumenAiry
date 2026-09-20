"""WP-C3 round 2 -- D9: does the internal call-site census fire for all six
un-named fourth call sites?

VERIFY-WP-C3 D9 measured the shipped census against six placements and found
it fired for two.  This driver re-applies all six to a COPY of the tree (never
to the worktree) and records, per placement, whether
``test_every_internal_transport_call_site_names_its_transport`` goes red.

    python r2_d9_census_mutants.py <mutant_tree> <out.json>
"""
import json
import os
import re
import shutil
import subprocess
import sys

TREE = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
CA = os.path.join(TREE, 'lumenairy', 'propagators', 'carrier.py')
MFT = os.path.join(TREE, 'lumenairy', 'propagators', 'mft.py')
TESTID = ('tests/unit/test_c3_collins_default.py::'
          'test_every_internal_transport_call_site_names_its_transport')

# the anchor each placement is inserted after / into
ANCHOR_MOD = "def _collins_carrier_leg("
ANCHOR_FN = "    env_a = xp.asarray(env)\n    if z == 0:"

PLACEMENTS = {
    # A -- a new module-level helper, plain call
    'A_module_level_helper': (CA, ANCHOR_MOD,
                              "def _vc3_mutA(env, R, z, wl, dx):\n"
                              "    return propagate_carrier_referenced("
                              "env, R, z, wl, dx)\n\n\n" + ANCHOR_MOD),
    # B -- a call inside an existing function
    'B_inside_existing_fn': (CA, ANCHOR_FN,
                             "    if False:\n"
                             "        propagate_carrier_referenced("
                             "env, R, z, wavelength, dx)\n" + ANCHOR_FN),
    # C -- a splat call in a function that is NOT an allow-listed forwarder
    'C_splat_in_non_forwarder': (CA, ANCHOR_MOD,
                                 "def _vc3_mutC(env, R, z, wl, dx, **kw):\n"
                                 "    return propagate_carrier_referenced("
                                 "env, R, z, wl, dx, **kw)\n\n\n"
                                 + ANCHOR_MOD),
    # C2 -- the same, splatting an empty dict literal
    'C2_splat_dict_literal': (CA, ANCHOR_MOD,
                              "def _vc3_mutC2(env, R, z, wl, dx):\n"
                              "    return propagate_carrier_referenced("
                              "env, R, z, wl, dx, **{})\n\n\n" + ANCHOR_MOD),
    # D -- a module-level alias, then a call through it
    'D_module_level_alias': (CA, ANCHOR_MOD,
                             "_vc3_alias = propagate_carrier_referenced\n\n\n"
                             "def _vc3_mutD(env, R, z, wl, dx):\n"
                             "    return _vc3_alias(env, R, z, wl, dx)\n\n\n"
                             + ANCHOR_MOD),
    # E -- a dynamic lookup
    'E_globals_lookup': (CA, ANCHOR_MOD,
                         "def _vc3_mutE(env, R, z, wl, dx):\n"
                         "    return globals()["
                         "'propagate_carrier_referenced'](env, R, z, wl, dx)\n"
                         "\n\n" + ANCHOR_MOD),
    # F -- the same call in ANOTHER shipped module
    'F_other_module': (MFT, None, None),
}


def run():
    src_ca = open(CA, encoding='cp1252').read()
    src_mft = open(MFT, encoding='cp1252').read()
    rows = {}
    for name, (path, anchor, repl) in PLACEMENTS.items():
        orig = src_ca if path == CA else src_mft
        if name == 'F_other_module':
            # APPENDED, not prepended: a module-level import ahead of
            # ``from __future__ import annotations`` is a SyntaxError, which
            # would make the placement untestable rather than uncaught.
            mutated = orig + (
                "\n\ndef _vc3_mutF(env, R, z, wl, dx):\n"
                "    from .carrier import propagate_carrier_referenced\n"
                "    return propagate_carrier_referenced(env, R, z, wl, dx)\n")
        else:
            assert orig.count(anchor) >= 1, name
            mutated = orig.replace(anchor, repl, 1)
        open(path, 'w', encoding='cp1252').write(mutated)
        try:
            pr = subprocess.run(
                [sys.executable, '-m', 'pytest', TESTID, '-q',
                 '--capture=sys', '-p', 'no:randomly'],
                cwd=TREE, capture_output=True, text=True,
                env=dict(os.environ, OMP_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            tail = (pr.stdout or '')[-2500:]
            m = re.search(r'(\d+ (?:passed|failed)[^\n]*)', tail)
            rows[name] = {'caught': ' failed' in tail or 'error' in tail,
                          'tail': (m.group(1) if m else tail[-200:]),
                          'why': next((ln.strip() for ln in tail.splitlines()
                                       if 'carrier.py:' in ln
                                       or 'mft.py:' in ln), '')}
        finally:
            open(path, 'w', encoding='cp1252').write(orig)
        print('%-26s caught=%-5s %s' % (name, rows[name]['caught'],
                                        rows[name]['tail']))
    n = sum(1 for v in rows.values() if v['caught'])
    print('CAUGHT %d of %d' % (n, len(rows)))
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump({'tree': TREE, 'caught': n, 'n': len(rows), 'rows': rows},
                  fh, indent=1)
    print('WROTE', OUT)


if __name__ == '__main__':
    assert 'lum_c3b' not in TREE.replace('\\', '/'), 'never mutate the worktree'
    assert os.path.isdir(os.path.join(TREE, 'tests')), TREE
    shutil.which('python')
    run()
