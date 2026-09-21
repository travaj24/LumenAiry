"""WP-C3 round 2 -- which quadrature does the p8 capstone's chain take?

``test_niche_p8_capstone.py::test_stepB_composed_doublet_relay_matches_debye``
passes on 49ddf4bd and on the pre-round-2 branch and fails on the round-2
tree (EE80 ratio 1.1124 against a 1.0 +/- 0.06 bar).  This records, per
carrier leg, the form the leg resolved, its Kelly conditions, whether the
output reference went flat, and the EE metrics the test grades -- so the
question "which tree is right" is asked of the numbers rather than of the
pass/fail.

    python r2_p8_capstone_diag.py <tree> <out.json>
"""
import json
import os
import sys
import warnings

import numpy as np

TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, 'tests', 'unit'))
OUT = sys.argv[2]

import lumenairy                                              # noqa: E402
import lumenairy.propagators.carrier as C                     # noqa: E402

assert os.path.abspath(lumenairy.__file__).startswith(TREE), (
    lumenairy.__file__, TREE)
print('[anchor]', lumenairy.__file__)

import test_niche_p8_capstone as P                            # noqa: E402

# ``tests/conftest.py`` installs a module's ``MODULE_GLASSES`` through a
# fixture; outside pytest the registry has to be primed by hand.
from lumenairy import glass as _G                             # noqa: E402
for _n, _f in (getattr(P, 'MODULE_GLASSES', None) or {}).items():
    _G.GLASS_REGISTRY[_n] = _f

LEGS = []
_real_leg = C._collins_carrier_leg


def _spy(env, R, z, wavelength, dx, dy, **kw):
    d = dict(kw.pop('diag', None) or {})
    out = _real_leg(env, R, z, wavelength, dx, dy, diag=d, **kw)
    rx, _ry, _ = C._parse_carrier(R, 'spy')
    a, b, _c, _dd = C._collins_envelope_abcd(rx, z, np.inf)
    LEGS.append({
        'z_mm': float(z) * 1e3, 'R_in_mm': float(rx) * 1e3,
        'A': float(a), 'dx_in_um': float(dx) * 1e6,
        'form': d.get('collins_form'),
        'k1': None if d.get('collins_k1') is None else max(d['collins_k1']),
        'k3': None if d.get('collins_k3') is None else max(d['collins_k3']),
        'flat': d.get('collins_flat_reference'),
        'dx_floor_hit': d.get('collins_dx_floor_hit'),
        'dx_out_um': float(out.dx if not isinstance(out.dx, tuple)
                           else out.dx[0]) * 1e6,
        'R_out': repr(out.R),
    })
    return out


def main():
    import inspect
    C._collins_carrier_leg = _spy
    rec = {'tree': TREE, 'lumenairy': lumenairy.__file__,
           'default_transport': inspect.signature(
               C.propagate_carrier_referenced).parameters['transport'].default}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            (r2m, e50, e80), method, _rest = P._run_composed_chain(P._WL)
    finally:
        C._collins_carrier_leg = _real_leg
    rec.update(legs=LEGS, method=method, r2m_um=r2m * 1e6,
               e50_um=e50 * 1e6, e80_um=e80 * 1e6)
    with open(OUT, 'w', encoding='utf-8') as fh:
        json.dump(rec, fh, indent=1)
    print(f"default={rec['default_transport']} method={method}")
    for lg in LEGS:
        print('  leg z=%.4f mm A=%+.6f form=%-7s K1=%s K3=%s flat=%s '
              'dx_out=%.4f um R_out=%s'
              % (lg['z_mm'], lg['A'], lg['form'],
                 None if lg['k1'] is None else round(lg['k1'], 4),
                 None if lg['k3'] is None else round(lg['k3'], 4),
                 lg['flat'], lg['dx_out_um'], lg['R_out']))
    print('  r2m=%.6f um  EE50=%.6f um  EE80=%.6f um'
          % (rec['r2m_um'], rec['e50_um'], rec['e80_um']))
    print('WROTE', OUT)


if __name__ == '__main__':
    main()
