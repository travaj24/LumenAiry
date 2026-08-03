# C11 side-quest -- ADJUDICATE the two v5.32.1 CI reconciliations against the
# library states that could have caused them.
#
# PR #25 (commit 5af1edf, the C9 + C10 tree) failed on Linux/OpenBLAS py3.10-13
# in two places that both PREDATE C9/C10 as calibrations:
#
#   D3   ``test_the_guarded_input_really_is_the_wrong_answer`` -- the chain's
#        LINEARITY violation on a near-collinear +-0.5 mrad pair, pinned
#        ``< 0.05`` against a documented 0.002.
#   RD   the ``amplitude_model='ray_density'`` energy self-check band
#        ``[1 - (BASE + PER_SUB (rs-1)), 1 + GAIN_TOL]``, read 0.8757 against
#        a 0.8900 floor on two fixtures.
#
# Both are candidates for "C9/C10 legitimately moved the number".  Rather than
# assume, this sweeps the two knobs that moved -- ``SPHERE_PARAB_CONVERSION_EXACT``
# (C9) and ``_REMAP_RESID_EIKONAL_DEGREE`` (C10) -- plus the C6 launch that
# both sit on, and reports the quantity each test measures in every state.  A
# number that moves with the knob is attributable; one that does not is not.
#
# usage:  WHAT=d3   python c11_ci_recon.py
#         WHAT=rd   python c11_ci_recon.py
import os
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '..', '..'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'tests', 'unit'))

import lumenairy.elements._lens_traced as LT                   # noqa: E402
import lumenairy.propagators.carrier as CM                     # noqa: E402

STATES = [
    ('shipped  C9=on  deg=6  C6=on', dict()),
    ('C10 off  C9=on  deg=4  C6=on', dict(deg=4)),
    ('C9  off  C9=off deg=6  C6=on', dict(c9=False)),
    ('both off C9=off deg=4  C6=on', dict(c9=False, deg=4)),
    ('C6  off  C9=on  deg=-  C6=off', dict(c6=False)),
]


class State(object):
    def __init__(self, c9=None, deg=None, c6=None):
        self.c9, self.deg, self.c6 = c9, deg, c6

    def __enter__(self):
        self.old = (CM.SPHERE_PARAB_CONVERSION_EXACT,
                    LT._REMAP_RESID_EIKONAL_DEGREE,
                    LT.REMAP_STATIONARY_PHASE_LAUNCH)
        if self.c9 is not None:
            CM.SPHERE_PARAB_CONVERSION_EXACT = bool(self.c9)
        if self.deg is not None:
            LT._REMAP_RESID_EIKONAL_DEGREE = int(self.deg)
        if self.c6 is not None:
            LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(self.c6)
        return self

    def __exit__(self, *e):
        (CM.SPHERE_PARAB_CONVERSION_EXACT,
         LT._REMAP_RESID_EIKONAL_DEGREE,
         LT.REMAP_STATIONARY_PHASE_LAUNCH) = self.old
        return False


def d3():
    import test_niche_d3_guards as D
    print("D3 -- relative L2 LINEARITY violation of the chain, "
          "chain(sum E_k) vs sum chain(E_k).")
    print("     The pin is  good < 0.05  and  bad > 20 good.\n")
    print(f"{'state':32} {'bad (23 mrad)':>14} {'good (0.5 mrad)':>16} "
          f"{'ratio':>8}")
    print('-' * 74)
    for name, kw in STATES:
        with State(**kw):
            bad = D._linearity_error(0.023)
            good = D._linearity_error(0.0005)
        print(f"{name:32} {bad:14.6f} {good:16.6f} "
              f"{bad / max(good, 1e-300):8.1f}", flush=True)
    return 0


def rd():
    import test_niche_p2_design_battery as P
    print("RD -- ray_density energy self-check ratio P_out/P_ap per fixture.")
    print(f"     band floor = 1 - (BASE {LT._RD_ENERGY_DEFICIT_BASE} + "
          f"PER_SUB {LT._RD_ENERGY_DEFICIT_PER_SUB} (rs-1))\n")
    names = [n for n in dir(P) if n.startswith('_DESIGNS')]
    print("   design table symbols:", names, flush=True)
    return 0


if __name__ == '__main__':
    sys.exit({'d3': d3, 'rd': rd}[os.environ.get('WHAT', 'd3')]())
