"""R4 -- VERIFY-WP-B12b D-3: WP-B12b section 7.4 publishes a cross-build pair
(8.318e-17 / 8.298e-17 rad) that the SHIPPED JSONs do not contain.

This re-runs the builder's own ``probe_c_decompose.py`` unchanged on this
tree, then reads the value out of the JSON it rewrote and reports it
alongside the committed one from the OTHER build.  The proof that the
artefact is bit-reproducible is that ``probe_c_decompose.py`` rewrites its
own JSON and ``git status`` stays clean, which the runner checks; this probe
records the NUMBERS so the report's correction cites a measurement.

Author: Andrew Traverso
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import r2_common as C

_GBDPROJ = Path(__file__).resolve().parent.parent / 'probe_gbd_projection'


def _read(tag):
    p = _GBDPROJ / f'probe_c_decompose_{tag}.json'
    if not p.exists():
        return None
    with open(p, encoding='cp1252') as fh:
        return json.load(fh)


def _find(obj, key, out=None):
    """Every value stored under ``key`` anywhere in the document."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key:
                out.append(v)
            _find(v, key, out)
    elif isinstance(obj, list):
        for v in obj:
            _find(v, key, out)
    return out


def main():
    C.assert_tree()
    arm, tokens = C.detect_arm()
    tag = C.build_tag()
    out = dict(env=C.env_block(), arm=arm, arm_tokens=tokens,
               published_in_report=dict(win32_314='8.318e-17',
                                        linux_312='8.298e-17'))

    env = dict(os.environ)
    env.setdefault('B12B_TREE', os.environ.get('R2_TREE', ''))
    r = subprocess.run([sys.executable, str(_GBDPROJ / 'probe_c_decompose.py')],
                       env=env, cwd=str(_GBDPROJ), capture_output=True,
                       text=True)
    out['rerun_returncode'] = r.returncode
    out['rerun_tail'] = r.stdout[-1500:] + r.stderr[-800:]
    print(r.stdout[-1500:])
    if r.returncode != 0:
        print(r.stderr[-2000:])

    for t in ('win32_314', 'linux_312'):
        doc = _read(t)
        if doc is None:
            out[f'json_{t}'] = None
            continue
        vals = _find(doc, 'max_relative_dphase_rad')
        out[f'json_{t}'] = dict(max_relative_dphase_rad=[repr(v) for v in vals])
        print(f'  {t}: max_relative_dphase_rad = {[repr(v) for v in vals]}')
    out['this_build'] = tag
    C.dump(out, 'probe_r4_probec_' + arm)


if __name__ == '__main__':
    main()
