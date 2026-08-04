# C12 -- GROUND TRUTH for the arbiter's per-group inputs on design 121.
#
# ``c11_discrim_121.py`` builds the two candidates SCRIPT-side, from two
# separately-forced traces.  That is not what the library does: the in-library
# arbiter builds both candidates from ONE trace, and its concentric disc is
# additionally intersected with the R7 carrier disc via ``_fit_r_max_conc``.
# So the script-side table can differ from the shipped decision.
#
# This captures the arbiter's OWN arguments, in process, by wrapping
# ``_decentred_fit_score`` -- every (xs, opl, weight, disc, weights, order) it
# is handed, and every score it returns -- with the flag ON.  Alternative
# weightings are then re-scored on EXACTLY those inputs, so a floor sweep is a
# statement about the shipped decision rather than about a re-derivation of it.
#
# The captures are written to ``_c12_arb_<order>.npz`` so the sweeps below can
# be replayed without re-running the chain.
#
# usage:  ORDERS='-1,0 -2,0 -3,0 -4,0 -4,-2' python c12_arb_trace_121.py
import hashlib
import os
import sys
import time

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import energy_stage_audit_121 as SA                             # noqa: E402
import lumenairy.elements._lens_traced as LT                    # noqa: E402

WL = SA.LAM


class ScoreSpy(object):
    """Record every ``_decentred_fit_score`` call the arbiter makes."""

    def __enter__(self):
        self.seen = []
        self._orig = LT._decentred_fit_score
        seen = self.seen

        def spy(xs_in, opl_grid, weight, disc, weights, order):
            v = self._orig(xs_in, opl_grid, weight, disc, weights, order)
            seen.append({'xs': np.asarray(xs_in).copy(),
                         'opl': np.asarray(opl_grid).copy(),
                         'weight': np.asarray(weight).copy(),
                         'disc': np.asarray(disc).copy(),
                         'weights': (None if weights is None
                                     else np.asarray(weights).copy()),
                         'order': int(order), 'score': float(v)})
            return v

        LT._decentred_fit_score = spy
        return self

    def __exit__(self, *e):
        LT._decentred_fit_score = self._orig
        return False


def main():
    orders = [tuple(int(v) for v in o.split(','))
              for o in os.environ.get(
                  'ORDERS', '-1,0 -2,0 -3,0 -4,0 -4,-2').split()]
    rs = int(os.environ.get('RS', '4'))
    save = os.environ.get('SAVE', '1') not in ('0', '')
    print(f"   lib {os.path.basename(LT.__file__)}  sha256 "
          f"{hashlib.sha256(open(LT.__file__, 'rb').read()).hexdigest()[:16]}",
          flush=True)
    old = LT.DECENTRED_FIT_ARBITER
    LT.DECENTRED_FIT_ARBITER = True
    try:
        for order in orders:
            t0 = time.time()
            with ScoreSpy() as sp:
                SA.run(order, 'ship', rs=rs)
            n = len(sp.seen)
            print(f"\n{str(order):>9}  {n} score calls "
                  f"({n // 2} arbitrated groups)  [{time.time()-t0:.0f}s]",
                  flush=True)
            print(f"{'pair':>5} {'ord_off':>8} {'ord_conc':>9} "
                  f"{'S_off':>12} {'S_conc':>12} {'ratio c/o':>10} "
                  f"{'PICK':>6}", flush=True)
            for k in range(0, n - 1, 2):
                o_, c_ = sp.seen[k], sp.seen[k + 1]
                rt = c_['score'] / max(o_['score'], 1e-300)
                print(f"{k // 2:>5} {o_['order']:>8} {c_['order']:>9} "
                      f"{o_['score']:>12.4e} {c_['score']:>12.4e} "
                      f"{rt:>10.4f} "
                      f"{'conc' if c_['score'] <= o_['score'] else 'off':>6}",
                      flush=True)
            if save:
                tag = f"{order[0]}_{order[1]}".replace('-', 'm')
                fn = f"_c12_arb_{tag}.npz"
                d = {}
                for k, r in enumerate(sp.seen):
                    d[f"xs{k}"] = r['xs']
                    d[f"opl{k}"] = r['opl']
                    d[f"wgt{k}"] = r['weight']
                    d[f"disc{k}"] = r['disc']
                    d[f"wts{k}"] = (np.zeros(0) if r['weights'] is None
                                    else r['weights'])
                    d[f"hasw{k}"] = np.array(r['weights'] is not None)
                    d[f"ord{k}"] = np.array(r['order'])
                    d[f"sc{k}"] = np.array(r['score'])
                d['n'] = np.array(n)
                np.savez_compressed(fn, **d)
                print(f"   -> {fn}  ({os.path.getsize(fn) / 1e6:.1f} MB)",
                      flush=True)
    finally:
        LT.DECENTRED_FIT_ARBITER = old
    return 0


if __name__ == '__main__':
    sys.exit(main())
