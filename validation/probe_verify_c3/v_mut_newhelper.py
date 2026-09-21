"""VERIFY-WP-C3 CLAIM 8d -- "a new helper added to the path with neither
property fails the census as unclassified".  Is that true for a helper whose
name does not begin with ``_collins``?

    python v_mut_newhelper.py <mutation-tree> <out.json>
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys

TREE = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
CARRIER = os.path.join(TREE, 'lumenairy', 'propagators', 'carrier.py')
PRISTINE = CARRIER + '.pristine'
ANCHOR = '    env_a = xp.asarray(env)\n'
TESTS = 'tests/unit/test_c3_collins_default.py'
XP = 'test_every_helper_on_the_collins_path_is_xp_parametrised'
DEM = 'test_no_collins_helper_demotes_the_field_to_host_numpy'

CASES = [
    ('A_new_helper_named__collins_x',
     '_collins_brand_new',
     'a NEW helper on the path, named _collins*, host-only, demoting'),
    ('B_new_helper_named_without_the_collins_prefix',
     '_carrier_brand_new',
     'the SAME helper, named without the _collins prefix'),
]


def run(kexpr):
    env = dict(os.environ)
    env.update(PYTHONPATH=TREE, OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    p = subprocess.run([sys.executable, '-m', 'pytest', TESTS, '-k', kexpr,
                        '-q', '--no-header', '-p', 'no:cacheprovider'],
                       cwd=TREE, env=env, capture_output=True, text=True,
                       timeout=900)
    return p.returncode, (p.stdout + p.stderr)[-700:]


def main():
    if not os.path.exists(PRISTINE):
        shutil.copy2(CARRIER, PRISTINE)
    rows = []
    for tag, name, why in CASES:
        src = open(PRISTINE, 'r', encoding='cp1252').read()
        newdef = ('\ndef %s(env):\n'
                  '    """A new helper with NEITHER property: no xp, and it '
                  'host-normalises the field."""\n'
                  '    return np.asarray(env)\n\n' % name)
        # define it just before _collins_transport, and CALL it from there
        marker = 'def _collins_transport('
        assert src.count(marker) == 1
        src = src.replace(marker, newdef.lstrip('\n') + '\n' + marker, 1)
        src = src.replace(ANCHOR, ANCHOR + '    %s(env)\n' % name, 1)
        open(CARRIER, 'w', encoding='cp1252').write(src)
        rc_x, out_x = run(XP)
        rc_d, out_d = run(DEM)
        rows.append({'case': tag, 'helper': name, 'why': why,
                     'xp_census_fired': rc_x != 0, 'xp_tail': out_x[-300:],
                     'demotion_census_fired': rc_d != 0,
                     'demotion_tail': out_d[-300:]})
        print('[new] %-48s xp-census %-5s  demotion-census %-5s'
              % (tag, 'FIRED' if rc_x else 'MISS',
                 'FIRED' if rc_d else 'MISS'))
    shutil.copy2(PRISTINE, CARRIER)
    with open(OUT, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump({'tree': TREE, 'rows': rows}, fh, indent=1, default=str)
    print('[wrote] %s' % OUT)


if __name__ == '__main__':
    main()
