"""WP-C1 ROUND 3 -- R8: a generated script EXECUTED, archive-to-archive.

VERIFY_WP-C1_ROUND2.md R8 reads the emitted TEXT of ``generate_simulation_script``'s
two styles and records, under "what could not be measured", that "neither
script was run against both trees, so the size of the move in a generated
script's own output is inferred from ``apply_aperture``'s, not measured".

This probe measures it.  For one STOP prescription it generates the script in
BOTH styles, in three arms --

  * from the PRE tree (``git archive 49ddf4bd``, the pre-WP-C1 parent), where
    ``apply_aperture``'s default rim was ``'hard'``: the pre-5.49 answer;
  * from the tree under test, untouched: today's default (grey);
  * from the tree under test, with that style's own MIGRATION RECIPE applied
    (``edge="hard"`` appended to the ``la.apply_aperture(...)`` call for
    ``style='unrolled'``; ``'edge': 'hard'`` added to the emitted element dict
    for ``style='system'``)

-- writes each to a temporary file and RUNS it in a subprocess with
``stdin=subprocess.DEVNULL``, against the tree that generated it, then digests
the exit-plane field.  The claim the Migration table makes is
``way_back == pre`` and ``default != pre``, for each style.

Each arm runs from inside its own tree: ``PYTHONPATH`` names it and the
subprocess prints the ``lumenairy.__file__`` it actually imported.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_r8_codegen_execute.py <out.json> <pre_tree>
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

import lumenairy  # noqa: F401

NL = chr(10)

RX = {
    'elements': [
        {'surf_num': 1, 'element_type': 'surface', 'radius': np.inf,
         'glass_after': 'air', 'is_stop': True, 'semi_diameter': 0.9e-3},
        {'surf_num': 2, 'element_type': 'surface', 'radius': 0.032,
         'glass_after': 'N-SF11', 'semi_diameter': 1.6e-3},
        {'surf_num': 3, 'element_type': 'surface', 'radius': -0.075,
         'glass_after': 'air', 'semi_diameter': 1.6e-3},
    ],
    'all_thicknesses': [1.5e-3, 2.5e-3, 18e-3],
    'aperture_diameter': 1.8e-3,
}

UNROLLED_CALL = ('la.apply_aperture(E, dx, shape="circular", '
                 'params={"diameter": 1.79999999999999995e-03})')
SYSTEM_ELEM = ("{'type': 'aperture', 'shape': 'circular', "
               "'params': {'diameter': 1.79999999999999995e-03}}")

GEN = '''
import json, sys
sys.path.insert(0, TREE)
import lumenairy
from lumenairy.io.codegen import generate_simulation_script
rx = json.loads(RXJSON)
rx["elements"][0]["radius"] = float("inf")
src = generate_simulation_script(rx, wavelength=633e-9, N=128, dx=40e-6,
                                 style=STYLE)
sys.stdout.write("LUMFILE " + lumenairy.__file__ + chr(10))
with open(OUT, "w", encoding="utf-8") as fh:
    fh.write(src)
'''

DRIVER_UNROLLED = (NL + 'import hashlib as _h, lumenairy as _la' + NL
                   + "print('LUMFILE ' + _la.__file__)" + NL
                   + '_E, _planes = run_simulation(verbose=False)' + NL
                   + "print('DIGEST ' + _h.blake2b(_E.tobytes(), "
                     "digest_size=8).hexdigest())" + NL)
DRIVER_SYSTEM = (NL + 'import hashlib as _h, lumenairy as _la' + NL
                 + "print('LUMFILE ' + _la.__file__)" + NL
                 + "print('DIGEST ' + _h.blake2b(E_out.tobytes(), "
                   "digest_size=8).hexdigest())" + NL)


def _env(tree):
    env = dict(os.environ)
    env['PYTHONPATH'] = tree
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[var] = '1'
    return env


def _run(path, tree, tmp):
    r = subprocess.run([sys.executable, path], stdin=subprocess.DEVNULL,
                       capture_output=True, text=True, env=_env(tree),
                       cwd=tmp, timeout=900)
    if r.returncode != 0:
        raise RuntimeError('{0} rc={1}{2}{3}{2}{4}'.format(
            path, r.returncode, NL, r.stdout[-1500:], r.stderr[-1500:]))
    return r.stdout


def _generate(tree, style, tmp, tag):
    """Generate the script from INSIDE ``tree`` (its own codegen, its own
    version stamp), in a subprocess so nothing of this process leaks in."""
    out = os.path.join(tmp, 'src_{0}.py'.format(tag))
    rx = dict(RX)
    rx = json.loads(json.dumps(rx, default=lambda o: 'inf'))
    gen = os.path.join(tmp, 'gen_{0}.py'.format(tag))
    body = ('TREE = {0!r}' + NL + 'RXJSON = {1!r}' + NL + 'STYLE = {2!r}' + NL
            + 'OUT = {3!r}' + NL + GEN).format(
        tree, json.dumps(rx), style, out)
    with open(gen, 'w', encoding='utf-8') as fh:
        fh.write(body)
    stdout = _run(gen, tree, tmp)
    with open(out, encoding='utf-8') as fh:
        return fh.read(), [ln.split(' ', 1)[1] for ln in stdout.splitlines()
                           if ln.startswith('LUMFILE ')][0]


def _digest(tree, style, src, way_back, tmp, tag):
    if style == 'unrolled':
        assert src.count(UNROLLED_CALL) == 1, 'unrolled anchor: ' + tag
        if way_back:
            src = src.replace(UNROLLED_CALL,
                              UNROLLED_CALL[:-1] + ', edge="hard")')
        head = "if __name__ == '__main__':"
        assert head in src, 'unrolled driver: ' + tag
        src = src[:src.index(head)] + DRIVER_UNROLLED
    else:
        assert src.count(SYSTEM_ELEM) == 1, 'system anchor: ' + tag
        if way_back:
            src = src.replace(SYSTEM_ELEM,
                              SYSTEM_ELEM[:-1] + ", 'edge': 'hard'}")
        src += DRIVER_SYSTEM
    path = os.path.join(tmp, 'run_{0}.py'.format(tag))
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(src)
    stdout = _run(path, tree, tmp)
    lines = stdout.splitlines()
    lum = [ln.split(' ', 1)[1] for ln in lines if ln.startswith('LUMFILE ')][0]
    dig = [ln.split()[1] for ln in lines if ln.startswith('DIGEST ')]
    assert len(dig) == 1, stdout[-1500:]
    return dig[0], lum


def main(out_path, pre_tree):
    # the tree under test = the parent of the imported ``lumenairy`` package
    here = os.path.dirname(os.path.dirname(os.path.abspath(
        lumenairy.__file__)))
    tmp = tempfile.mkdtemp()
    rows = {}
    try:
        for style in ('unrolled', 'system'):
            src_pre, lum_pre = _generate(pre_tree, style, tmp,
                                         'pre_' + style)
            src_now, lum_now = _generate(here, style, tmp, 'now_' + style)
            d_pre, l_pre = _digest(pre_tree, style, src_pre, False, tmp,
                                   'pre_' + style)
            d_def, l_def = _digest(here, style, src_now, False, tmp,
                                   'def_' + style)
            d_way, l_way = _digest(here, style, src_now, True, tmp,
                                   'way_' + style)
            rows[style] = {
                'pre_49ddf4bd': d_pre, 'default_here': d_def,
                'way_back_here': d_way,
                'way_back_equals_pre': d_way == d_pre,
                'default_differs_from_pre': d_def != d_pre,
                'pre_emits_edge_key': ("'edge'" in src_pre
                                       or 'edge=' in src_pre),
                'lumenairy_pre': l_pre, 'lumenairy_here': l_def,
                'generator_pre': lum_pre, 'generator_here': lum_now,
            }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    out = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'pre_tree': pre_tree,
        'styles': rows,
        'both_styles_agree_way_back': (rows['unrolled']['way_back_here']
                                       == rows['system']['way_back_here']),
        'both_styles_agree_default': (rows['unrolled']['default_here']
                                      == rows['system']['default_here']),
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)
    print('lumenairy:', lumenairy.__file__)
    for style, r in rows.items():
        print('  {0:9s} pre={1} default={2} way_back={3}  '
              'way_back==pre: {4}  default!=pre: {5}'.format(
                  style, r['pre_49ddf4bd'], r['default_here'],
                  r['way_back_here'], r['way_back_equals_pre'],
                  r['default_differs_from_pre']))
        print('            pre tree imported: {0}'.format(r['lumenairy_pre']))
    print('  both styles agree, way back:', out['both_styles_agree_way_back'])
    print('  both styles agree, default :', out['both_styles_agree_default'])


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
