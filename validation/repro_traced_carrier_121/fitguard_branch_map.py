# WHICH GROUPS DOES ``REMAP_STATIONARY_PHASE_FIT_GUARD`` ACTUALLY TOUCH?
#
# The record says the guard "acts only on the concentric (on-axis) fit branch"
# and is "byte-identical on every tilted order".  The chain measurement
# (energy_stage_audit_121.py) contradicts the second half: at orders (-1,0),
# (-2,0) and (-3,0) every stage of the chain moves.  That is not enough to
# convict it, because a chain cascades -- if group 0 moves, groups 1-5 move
# whether or not the guard touched them.
#
# This isolates it.  One shipped chain run per order is captured, and then EACH
# group's element call is REPLAYED on its own captured input with the guard off
# and on.  A group where the two replays are byte-identical is a group the
# guard does not touch, cascade or no cascade.  The beam decentre in units of
# the input beam radius is reported alongside, since that is the quantity the
# ``_DECENTRE_GATE_W_FRAC`` gate tests to choose the branch.
#
# LOCAL-ONLY; the library is read only and the flags are module attributes set
# inside try/finally.
#
# usage:  ORDERS='0,0 -1,0 -2,0 -3,0 -4,0 -4,-2' python fitguard_branch_map.py
import hashlib
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import energy_stage_audit_121 as SA                               # noqa: E402,I001
import lumenairy.elements._lens_traced as LT                      # noqa: E402
from lumenairy.elements._lens_traced import _input_beam_amp_radius  # noqa: E402,E501


def replay(E_in, kw, guard):
    old = LT.REMAP_STATIONARY_PHASE_FIT_GUARD
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            out = np.asarray(LT.apply_real_lens_traced(E_in, **kw))
    finally:
        LT.REMAP_STATIONARY_PHASE_FIT_GUARD = old
    halo = sum(1 for w in wl if 'HALO self-check FAILED' in str(w.message))
    return out, halo


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get(
                  'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()]
    rs = int(os.environ.get('RS', '4'))
    print("   lib %s  sha256 %s" % (
        os.path.basename(LT.__file__),
        hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]))
    print(f"   ray_subsample={rs}.  Each group's element call is replayed on "
          f"its OWN captured input, guard off vs on.")
    print(f"   gate: a disc is treated as CONCENTRIC (the branch the guard "
          f"acts on) when decentre <= {LT._DECENTRE_GATE_W_FRAC:g} w.")
    print()
    hdr = (f"{'order':>9} {'grp':>4} {'dec/w':>8} {'branch':>11} "
           f"{'guard moves it':>15} {'max|dE|':>10} {'rel':>10} "
           f"{'HALO off/on':>12}")
    print(hdr)
    print('-' * len(hdr))
    for order in orders:
        calls, _res, _cw, _s = SA.run(order, 'ship', rs=rs)
        for k, rec in enumerate(calls):
            kw = dict(rec['kw'])
            E_in = rec['E_in']
            car = kw.get('carrier')
            x0 = float(getattr(car, 'x0', 0.0) or 0.0)
            y0 = float(getattr(car, 'y0', 0.0) or 0.0)
            dxk = float(kw['dx'])
            dec = float(np.hypot(x0, y0))
            w = float(_input_beam_amp_radius(
                E_in, dxk, dxk, centre=((x0, y0) if dec > 0 else None)))
            a, ha = replay(E_in, kw, False)
            b, hb = replay(E_in, kw, True)
            eq = bool(np.array_equal(a, b))
            mx = float(np.abs(a - b).max())
            pk = float(np.abs(a).max())
            ratio = dec / w if w > 0 else float('nan')
            branch = ('CONCENTRIC' if ratio <= LT._DECENTRE_GATE_W_FRAC
                      else 'off-centre')
            print(f"{str(order):>9} {k:>4} {ratio:>8.4f} {branch:>11} "
                  f"{('no' if eq else 'YES'):>15} {mx:>10.3e} "
                  f"{(mx / pk if pk > 0 else float('nan')):>10.3e} "
                  f"{ha:>5d}/{hb:<6d}", flush=True)
        print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
