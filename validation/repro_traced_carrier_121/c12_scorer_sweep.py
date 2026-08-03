# C12 -- re-score the CAPTURED arbiter inputs under alternative weightings.
#
# ``c12_arb_trace_121.py`` records every ``(xs, opl, weight, disc, weights,
# order)`` the shipped arbiter is handed, in process, with the flag ON.  This
# replays those exact inputs under a family of weights and reports how each
# group's verdict moves -- so a scorer change is priced against the shipped
# decision without re-running a single ray trace.
#
# The weight families, all built from the SAME beam intensity
# ``g = exp(-2 |r - c|^2 / w^2)`` the shipped scorer uses:
#
#   c11        g                        (shipped: core only, ~0 beyond 2 w)
#   floor:F    max(g, F)                (skirt at a fixed floor, unbounded)
#   supp:F     max(g, F) where g >= F   (skirt at a floor, ILLUMINATED
#                                        SUPPORT only -- 0 where no light is)
#   amp        sqrt(g)                  (amplitude weighting)
#   ampsupp:F  sqrt(g) where g >= F
#
# usage:  python c12_scorer_sweep.py
import os
import sys

import numpy as np

os.environ.setdefault('LUMEN_PIN', '0')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import lumenairy.elements._lens_traced as LT                    # noqa: E402

TAGS = os.environ.get('TAGS', 'm1_0 m2_0 m3_0 m4_0 m4_m2').split()


def load(tag):
    fn = f"_c12_arb_{tag}.npz"
    if not os.path.exists(fn):
        return None
    z = np.load(fn)
    n = int(z['n'])
    out = []
    for k in range(n):
        out.append({'xs': z[f"xs{k}"], 'opl': z[f"opl{k}"],
                    'wgt': z[f"wgt{k}"], 'disc': z[f"disc{k}"],
                    'weights': (z[f"wts{k}"] if bool(z[f"hasw{k}"]) else None),
                    'order': int(z[f"ord{k}"]), 'score': float(z[f"sc{k}"])})
    return out


def families(g):
    """label -> weight array, all derived from the shipped beam intensity."""
    out = [('c11', g)]
    for F in (1e-8, 1e-6, 1e-4, 1e-2):
        out.append((f"fl{F:.0e}", np.maximum(g, F)))
    for F in (1e-8, 1e-6, 1e-4, 1e-3, 1e-2):
        out.append((f"su{F:.0e}", np.where(g >= F, np.maximum(g, F), 0.0)))
    out.append(('amp', np.sqrt(g)))
    for F in (1e-8, 1e-6, 1e-4):
        out.append((f"as{F:.0e}", np.where(g >= F, np.sqrt(g), 0.0)))
    return out


def main():
    rows = {}
    labels = None
    for tag in TAGS:
        rec = load(tag)
        if rec is None:
            print(f"   (no capture for {tag})")
            continue
        for k in range(0, len(rec) - 1, 2):
            o_, c_ = rec[k], rec[k + 1]
            g = o_['wgt']
            fam = families(g)
            if labels is None:
                labels = [f for f, _ in fam]
            cells = []
            for _lab, wgt in fam:
                s_o = LT._decentred_fit_score(o_['xs'], o_['opl'], wgt,
                                              o_['disc'], o_['weights'],
                                              o_['order'])
                s_c = LT._decentred_fit_score(c_['xs'], c_['opl'], wgt,
                                              c_['disc'], c_['weights'],
                                              c_['order'])
                cells.append(s_c / max(s_o, 1e-300))
            rows[(tag, k // 2)] = cells
    if labels is None:
        print("no captures found -- run c12_arb_trace_121.py first")
        return 1
    hdr = f"{'order':>7} {'pair':>5}" + ''.join(f"{lb:>10}" for lb in labels)
    print(hdr)
    print('-' * len(hdr))
    for (tag, k), cells in rows.items():
        print(f"{tag:>7} {k:>5}"
              + ''.join(f"{v:10.3f}" for v in cells))
    print("\nvalue = concentric/off-centre score ratio.  < 1 picks CONCENTRIC.")
    print("fl = floored (unbounded), su = floored on the ILLUMINATED SUPPORT,")
    print("amp/as = amplitude weighting.  'c11' is the shipped scorer.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
