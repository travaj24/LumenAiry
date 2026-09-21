"""WP-C3 round 2 -- D10(b, c): does the host-demotion census catch the
thirteen injected demotions VERIFY-WP-C3 found it missing nine of?

Each injection goes into ``_collins_carrier_leg``'s body (or, for the
new-helper case, beside it and on its call graph) in a COPY of the tree, and
``test_no_collins_helper_demotes_the_field_to_host_numpy`` is run against it.

    python r2_d10_demotion_mutants.py <mutant_tree> <out.json>
"""
import json
import os
import re
import subprocess
import sys

TREE = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
CA = os.path.join(TREE, 'lumenairy', 'propagators', 'carrier.py')
#: BOTH censuses, because they divide the work: the demotion matcher owns the
#: shapes, the xp census owns the classification of a NEW helper that receives
#: a field (VERIFY-WP-C3 D10(c)'s ``_carrier_brand_new``).
TESTIDS = ['tests/unit/test_c3_collins_default.py::'
           'test_no_collins_helper_demotes_the_field_to_host_numpy',
           'tests/unit/test_c3_collins_default.py::'
           'test_every_helper_on_the_collins_path_is_xp_parametrised']

ANCHOR = "    env_a = xp.asarray(env)\n    if z == 0:"
HELPER_ANCHOR = "def _collins_carrier_leg("

#: (label, where, injected source).  ``body`` goes inside the leg; ``module``
#: goes beside it with a call added to the leg so the census reaches it.
INJ = [
    ('I01_np_asarray_env', 'body',
     "    _v1 = np.asarray(env)\n"),
    ('I02_np_ascontiguousarray_env', 'body',
     "    _v2 = np.ascontiguousarray(env)\n"),
    ('I03_np_asarray_env_a', 'body',
     "    _v3 = np.asarray(env)\n"),
    ('I04_np_array_env', 'body',
     "    _v4 = np.array(env)\n"),
    ('I05_numpy_alias_assignment', 'body',
     "    _vnp = np\n    _v5 = _vnp.asarray(env)\n"),
    ('I06_full_module_name', 'body',
     "    import numpy\n    _v6 = numpy.asarray(env)\n"),
    ('I07_renamed_local', 'body',
     "    _vtmp = env\n    _v7 = np.asarray(_vtmp)\n"),
    ('I08_numpy_constructor_in_expr', 'body',
     "    _v8 = env * np.arange(Nx if False else 4)\n"),
    ('I09_np_ufunc_on_field', 'body',
     "    _v9 = np.exp(env)\n"),
    ('I10_float_of_field', 'body',
     "    _v10 = float(env[0, 0].real)\n"),
    ('I11_item_of_field', 'body',
     "    _v11 = env.ravel()[0].item()\n"),
    ('I12_math_on_field', 'body',
     "    import math\n    _v12 = math.sqrt(abs(env[0, 0]))\n"),
    ('I13_astype_result_type', 'body',
     "    _v13 = env.astype(np.result_type(np.complex128))\n"),
    ('I14_new_unclassified_helper', 'module', None),
]


def main():
    orig = open(CA, encoding='cp1252').read()
    assert orig.count(ANCHOR) == 1
    rows = {}
    for label, where, inj in INJ:
        if where == 'body':
            mutated = orig.replace(
                ANCHOR, "    env_a = xp.asarray(env)\n" + inj
                + "    if z == 0:", 1)
        else:
            mutated = orig.replace(
                HELPER_ANCHOR,
                "def _carrier_brand_new(env):\n"
                "    return np.asarray(env)\n\n\n" + HELPER_ANCHOR, 1)
            mutated = mutated.replace(
                ANCHOR, "    env_a = xp.asarray(env)\n"
                        "    _carrier_brand_new(env)\n    if z == 0:", 1)
        open(CA, 'w', encoding='cp1252').write(mutated)
        try:
            pr = subprocess.run(
                [sys.executable, '-m', 'pytest', *TESTIDS, '-q',
                 '--capture=sys', '-p', 'no:randomly'],
                cwd=TREE, capture_output=True, text=True,
                env=dict(os.environ, OMP_NUM_THREADS='1',
                         OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
            tail = (pr.stdout or '')[-2500:]
            m = re.search(r'(\d+ (?:passed|failed)[^\n]*)', tail)
            rows[label] = {
                'caught': ' failed' in tail or 'error' in tail,
                'tail': m.group(1) if m else tail[-160:],
                'why': next((ln.strip() for ln in tail.splitlines()
                             if '_collins_carrier_leg:' in ln
                             or '_carrier_brand_new' in ln), '')}
        finally:
            open(CA, 'w', encoding='cp1252').write(orig)
        print('%-32s caught=%-5s %s' % (label, rows[label]['caught'],
                                        rows[label]['tail']))
    n = sum(1 for v in rows.values() if v['caught'])
    print('CAUGHT %d of %d' % (n, len(rows)))
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump({'tree': TREE, 'caught': n, 'n': len(rows), 'rows': rows},
                  fh, indent=1)
    print('WROTE', OUT)


if __name__ == '__main__':
    assert 'lum_c3b' not in TREE.replace('\\', '/'), 'never mutate the worktree'
    main()
