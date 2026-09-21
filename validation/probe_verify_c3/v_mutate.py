"""VERIFY-WP-C3 CLAIM 8e -- can the host-demotion census SEE a demotion?

    python v_mutate.py <mutation-tree> <out.json>

For each injection the pristine ``carrier.py`` is restored, ONE mutation is
written into a helper the census names, and the SHIPPED census tests are run
against the mutated tree.  A mutation the census does not fail on is a MISS.
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

ANCHOR_LEG = '    env_a = xp.asarray(env)\n'
ANCHOR_TRANSPORT = None   # resolved below

TESTS = 'tests/unit/test_c3_collins_default.py'
CENSUS_DEMOTION = 'test_no_collins_helper_demotes_the_field_to_host_numpy'
CENSUS_XP = 'test_every_helper_on_the_collins_path_is_xp_parametrised'

MUTATIONS = [
    ('M1_float_on_device_array',
     '    _vdbg = float(env[0, 0].real)\n',
     'float(x) on a device array -- forces a host sync / concretisation'),
    ('M2_np_asarray_control',
     '    _vdbg = np.asarray(env)\n',
     'THE CONTROL: the exact spelling V-D3 fixed'),
    ('M3_item_on_device_array',
     '    _vdbg = env.ravel()[0].item()\n',
     'x.item() -- host pull of one element'),
    ('M4_math_call_on_element',
     '    import math as _vmath\n'
     '    _vdbg = _vmath.sqrt(abs(env[0, 0]))\n',
     'a math. call on a device-array element'),
    ('M5_numpy_constant_times_device',
     '    env_a = env_a * np.arange(env_a.shape[-1])\n',
     'a NumPy CONSTANT array combined with the device array'),
    ('M6_np_exp_on_device',
     '    _vdbg = np.exp(env)\n',
     'np.exp applied to the device array'),
    ('M7_alias_np',
     '    _vnp = np\n'
     '    _vdbg = _vnp.asarray(env)\n',
     'the demotion behind an alias'),
    ('M8_nested_function',
     '    def _vinner():\n'
     '        return np.asarray(env)\n'
     '    _vdbg = _vinner\n',
     'the demotion inside a nested function'),
    ('M8b_comprehension',
     '    _vdbg = [np.asarray(env) for _ in range(1)]\n',
     'the demotion inside a comprehension'),
    ('M9_full_module_name',
     '    import numpy\n'
     '    _vdbg = numpy.asarray(env)\n',
     'np.asarray reached through the full module name'),
    ('M10_astype_via_result_type',
     '    env_a = env_a.astype(np.result_type(env, np.complex128))\n',
     'env.astype(...) via np.result_type on the device array'),
    ('M11_renamed_local',
     '    _vtmp = env\n'
     '    _vdbg = np.asarray(_vtmp)\n',
     'the same np.asarray, on a local the field was renamed into'),
    ('M12_ascontiguous_on_spectrum',
     '    _vdbg = np.ascontiguousarray(env)\n',
     'the OTHER normaliser the census lists, same spelling'),
]


def run_pytest(kexpr):
    env = dict(os.environ)
    env.update(PYTHONPATH=TREE, OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    p = subprocess.run(
        [sys.executable, '-m', 'pytest', TESTS, '-k', kexpr, '-q',
         '--no-header', '-p', 'no:cacheprovider', '-x'],
        cwd=TREE, env=env, capture_output=True, text=True, timeout=900)
    return p.returncode, (p.stdout + p.stderr)[-1600:]


def inject(line):
    # The anchor appears twice -- in _collins_transport (line 2340) and in
    # _collins_carrier_leg (line 2637).  BOTH are census sites; inject into
    # the first, _collins_transport, so the census has every chance.
    src = open(PRISTINE, 'r', encoding='cp1252').read()
    assert src.count(ANCHOR_LEG) == 2, src.count(ANCHOR_LEG)
    open(CARRIER, 'w', encoding='cp1252').write(
        src.replace(ANCHOR_LEG, ANCHOR_LEG + line, 1))


def main():
    if not os.path.exists(PRISTINE):
        shutil.copy2(CARRIER, PRISTINE)
    shutil.copy2(PRISTINE, CARRIER)
    rows = []
    rc, out = run_pytest('%s or %s' % (CENSUS_DEMOTION, CENSUS_XP))
    rows.append({'mutation': 'BASELINE (pristine)', 'why': '--',
                 'returncode': rc, 'census_fired': rc != 0, 'tail': out[-500:]})
    print('[mut] BASELINE rc=%d  %s' % (rc, out.strip().splitlines()[-1]))
    for name, line, why in MUTATIONS:
        inject(line)
        rc_d, out_d = run_pytest(CENSUS_DEMOTION)
        rc_x, out_x = run_pytest(CENSUS_XP)
        row = {'mutation': name, 'why': why, 'injected': line,
               'demotion_census_rc': rc_d, 'demotion_census_fired': rc_d != 0,
               'xp_census_rc': rc_x, 'xp_census_fired': rc_x != 0,
               'tail_demotion': out_d[-400:]}
        rows.append(row)
        print('[mut] %-32s demotion-census %s   xp-census %s   (%s)'
              % (name, 'FIRED' if rc_d else 'MISS',
                 'FIRED' if rc_x else 'miss', why))
    shutil.copy2(PRISTINE, CARRIER)
    with open(OUT, 'w', encoding='cp1252', errors='replace') as fh:
        json.dump({'tree': TREE, 'rows': rows}, fh, indent=1, default=str)
    print('[wrote] %s' % OUT)
    misses = [r['mutation'] for r in rows[1:]
              if not r['demotion_census_fired']]
    print('MISSES (%d/%d): %s' % (len(misses), len(rows) - 1, misses))


if __name__ == '__main__':
    main()
