"""Oracle-FREE plane scan -- locate the fold window of a fixture cheaply.

Reports, per plane: the branch sum's bracketed gain reading, the pixel-halving
continuity ratio and its decision, ``n_branch_max``, and the uniform
completion's route (``reason`` / ``fell_back`` / which arm refused).  No
oracle, so this is seconds per plane and is used only to CHOOSE the planes the
oracle-scored probe then runs on.

Usage::

    python scan.py <FIXTURE> [--out out.json] --range <lo_um> <hi_um> <n>
    python scan.py <FIXTURE> [--out out.json] <z_um> [<z_um> ...]
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import os
import re
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fixtures as FX                                     # noqa: E402

import lumenairy                                          # noqa: E402
from lumenairy.elements._lens_traced_multibranch import (  # noqa: E402
    _multibranch_render)
from lumenairy.elements._lens_traced_uniform import (      # noqa: E402
    apply_real_lens_traced_uniform)


def row(fx, E, z):
    with warnings.catch_warnings(record=True):
        warnings.simplefilter('always')
        _E_mb, md = _multibranch_render(
            E, prescription=fx['prescription'], wavelength=fx['wavelength'],
            dx=fx['dx'], output_plane_distance=z, return_diagnostics=True,
            pixel_halving_arbiter=True)
    vals = [float(v) for v in (md.get('power_ratio'),
                               md.get('power_ratio_triangles'))
            if v is not None and np.isfinite(v)]
    r = {'z_um': z * 1e6, 'bracket': (min(vals) if vals else None),
         'continuity': md.get('pixel_continuity'),
         'continuity_decision': md.get('pixel_continuity_decision'),
         'n_branch_max': md.get('n_branch_max'),
         'n_triangles_degenerate': md.get('n_triangles_degenerate')}
    try:
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            _E, ud = apply_real_lens_traced_uniform(
                E, prescription=fx['prescription'],
                wavelength=fx['wavelength'], dx=fx['dx'],
                output_plane_distance=z, return_diagnostics=True)
        r['reason'] = ud.get('reason')
        r['fell_back'] = ud.get('fell_back')
        r['zeta_x'] = ud.get('zeta_extrapolation')
        r['refused'] = None
        r['continuity_returned'] = ud.get('pixel_continuity')
        r['continuity_of'] = ud.get('pixel_continuity_of')
        r['continuity_decision'] = ud.get('pixel_continuity_decision')
    except RuntimeError as e:
        r['reason'] = None
        r['refused'] = ('pixel_continuity' if 'NOT CONVERGED' in str(e)
                        else 'power_ratio')
        # the refusal NAMES the reading it refused on, so the probe can read
        # it back at planes the shipped build does not return a field for
        m = re.search(r'continuity ratio of ([0-9.eE+-]+)', str(e))
        if m:
            r['continuity_returned'] = float(m.group(1))
            r['continuity_of'] = 'from the refusal message'
        r['message'] = str(e)[:600]
    return r


def main(argv):
    name = argv[1]
    rest = list(argv[2:])
    out = None
    if rest and rest[0] == '--out':
        rest.pop(0)
        out = rest.pop(0)
    if rest and rest[0] == '--range':
        zs = np.linspace(float(rest[1]), float(rest[2]), int(rest[3])) * 1e-6
    else:
        zs = np.array([float(v) for v in rest]) * 1e-6
    fx = FX.FIXTURES[name]
    E = FX.input_field(fx)
    print(f"lumenairy {lumenairy.__file__}  {name}: {fx['note']}", flush=True)
    rows = []
    for z in zs:
        r = row(fx, E, float(z))
        rows.append(r)
        print(f"z={r['z_um']:10.3f} br={r['bracket']!s:>11.11} "
              f"Cmb={r['continuity']!s:>9.9} "
              f"Cret={r.get('continuity_returned')!s:>9.9} "
              f"nbr={r['n_branch_max']!s:>5} "
              f"deg={r['n_triangles_degenerate']!s:>6} "
              f"zx={r.get('zeta_x')!s:>9.9} "
              f"{r.get('reason') or ('REFUSED:' + str(r['refused']))}",
              flush=True)
    if out:
        with open(out, 'w') as fh:
            json.dump({'fixture': name, 'note': fx['note'],
                       'lumenairy_file': lumenairy.__file__,
                       'N': fx['N'], 'dx': fx['dx'], 'rows': rows}, fh,
                      indent=1, default=str)


if __name__ == '__main__':
    main(sys.argv)
