# D121 RESIDUAL CLOSURE -- score the SAVED readout intensities every way, so
# the metric convention can be separated from the field.
#
# Pure post-processing of ``rc_readout_121.py``'s npz: no propagation, no ray
# trace, no library call.  Nothing here can change what the field is; it can
# only change what the number means.
#
# Reports, per order:
#   1. EE3/EE6 under HARD (shipped) and AREA-EXACT masks, each about the arm's
#      OWN centroid, about the sub-pixel PEAK, and about a COMMON centre;
#   2. the pure-quantisation term  EE_hard - EE_area  per arm;
#   3. the mask-centre sensitivity: EE3 swept over one full readout pixel,
#      hard and area-exact, so the sawtooth and the smooth part separate;
#   4. the cumulative radial energy difference chain - oracle, which says at
#      WHICH radius the chain's missing energy actually sits.
#
# usage:  TAG=0.4um_61 python rc_score_121.py
import os
import sys

import numpy as np

import rc_readout_121 as RD

_HERE = os.path.dirname(os.path.abspath(__file__))


def cum_radial(I, ax, cx, cy, edges):
    """Area-exact cumulative energy fraction at each radius in ``edges``."""
    t = float(I.sum())
    return np.array([float((I * RD.area_frac(ax, cx, cy, r)).sum()) / t
                     for r in edges])


def main():
    tag = os.environ.get('TAG', '0.4um_61')
    orders = os.environ.get(
        'ORDERS', '0,0 -1,0 -2,0 -3,0 -4,0 -4,-2').split()
    d = np.load(os.path.join(_HERE, f"_rc_readout_{tag}.npz"))
    print(f"=== scoring _rc_readout_{tag}.npz ===\n")

    # --- 0. estimator self-check: the area mask must be converged in ss -----
    o0 = orders[0]
    I0, ax0 = d[f"I_{o0}_oracle"], d[f"ax_{o0}_oracle"]
    c0 = d[f"c_{o0}_oracle"]
    v = [RD.ee(I0, ax0, c0[0], c0[1], 3e-6, 'area', ss) for ss in (8, 16, 32)]
    print(f"AREA-MASK CONVERGENCE (ss = 8/16/32): "
          f"{v[0] * 100:.6f} / {v[1] * 100:.6f} / {v[2] * 100:.6f}  "
          f"-> spread {(max(v) - min(v)) * 100:.2e} points\n")

    # --- 1/2. the conventions ----------------------------------------------
    hdr = (f"{'order':>8} {'arm':>7} {'EE3 hard':>9} {'EE3 area':>9} "
           f"{'h - a':>7} {'EE3 pk-a':>9} {'EE6 hard':>9} {'EE6 area':>9} "
           f"{'h - a':>7}")
    print("1/2. MASK CONVENTION  (hard = shipped binary pixel mask; "
          "area = same circle, pixel-area weighted)")
    print(hdr)
    print('-' * len(hdr))
    res = {}
    for o in orders:
        for arm in ('oracle', 'chain'):
            I, ax = d[f"I_{o}_{arm}"], d[f"ax_{o}_{arm}"]
            cx, cy, px, py = d[f"c_{o}_{arm}"]
            e3h = RD.ee(I, ax, cx, cy, 3e-6, 'hard')
            e3a = RD.ee(I, ax, cx, cy, 3e-6, 'area')
            e3p = RD.ee(I, ax, px, py, 3e-6, 'area')
            e6h = RD.ee(I, ax, cx, cy, 6e-6, 'hard')
            e6a = RD.ee(I, ax, cx, cy, 6e-6, 'area')
            res[(o, arm)] = dict(e3h=e3h, e3a=e3a, e3p=e3p, e6h=e6h, e6a=e6a)
            print(f"{o:>8} {arm:>7} {e3h * 100:9.4f} {e3a * 100:9.4f} "
                  f"{(e3h - e3a) * 100:+7.4f} {e3p * 100:9.4f} "
                  f"{e6h * 100:9.4f} {e6a * 100:9.4f} "
                  f"{(e6h - e6a) * 100:+7.4f}")
    print()
    print("   RESIDUAL (oracle - chain) UNDER EACH CONVENTION")
    h2 = (f"{'order':>8} {'hard@own':>9} {'area@own':>9} {'area@peak':>10} "
          f"{'area@common':>12} {'EE6 hard':>9} {'EE6 area':>9}")
    print(h2)
    print('-' * len(h2))
    for o in orders:
        a, c = res[(o, 'oracle')], res[(o, 'chain')]
        cxo, cyo = d[f"c_{o}_oracle"][:2]
        com = [RD.ee(d[f"I_{o}_{k}"], d[f"ax_{o}_{k}"], cxo, cyo, 3e-6, 'area')
               for k in ('oracle', 'chain')]
        print(f"{o:>8} {(a['e3h'] - c['e3h']) * 100:9.4f} "
              f"{(a['e3a'] - c['e3a']) * 100:9.4f} "
              f"{(a['e3p'] - c['e3p']) * 100:10.4f} "
              f"{(com[0] - com[1]) * 100:12.4f} "
              f"{(a['e6h'] - c['e6h']) * 100:9.4f} "
              f"{(a['e6a'] - c['e6a']) * 100:9.4f}")

    # --- 3. mask-centre sensitivity ----------------------------------------
    print("\n3. MASK-CENTRE SENSITIVITY -- EE3 swept over one readout pixel "
          "about each arm's own centroid")
    print("   (hard span = what the shipped metric can do on its own; "
          "area span = the genuine off-centre cost)")
    dxo = float(d[f"ax_{orders[0]}_oracle"][1]
                - d[f"ax_{orders[0]}_oracle"][0])
    off = (np.arange(9) / 8.0 - 0.5) * dxo
    h3 = (f"{'order':>8} {'arm':>7} {'hard min':>9} {'hard max':>9} "
          f"{'span':>7} {'area min':>9} {'area max':>9} {'span':>7}")
    print(h3)
    print('-' * len(h3))
    for o in orders:
        for arm in ('oracle', 'chain'):
            I, ax = d[f"I_{o}_{arm}"], d[f"ax_{o}_{arm}"]
            cx, cy = d[f"c_{o}_{arm}"][:2]
            vh, va = [], []
            for ddx in off:
                for ddy in off:
                    vh.append(RD.ee(I, ax, cx + ddx, cy + ddy, 3e-6, 'hard'))
                    va.append(RD.ee(I, ax, cx + ddx, cy + ddy, 3e-6, 'area'))
            vh, va = np.array(vh) * 100, np.array(va) * 100
            print(f"{o:>8} {arm:>7} {vh.min():9.4f} {vh.max():9.4f} "
                  f"{np.ptp(vh):7.4f} {va.min():9.4f} {va.max():9.4f} "
                  f"{np.ptp(va):7.4f}")

    # --- 4. where the missing energy is ------------------------------------
    print("\n4. CUMULATIVE ENERGY (area-exact, about each arm's own centroid)"
          " -- oracle - chain, in points, vs radius")
    edges = np.array([1., 1.5, 2., 2.5, 3., 3.5, 4., 5., 6., 8., 10., 12.]) \
        * 1e-6
    print(f"{'order':>8} " + ''.join(f"{e * 1e6:8.1f}" for e in edges))
    print('-' * (9 + 8 * len(edges)))
    for o in orders:
        co = d[f"c_{o}_oracle"]
        cc = d[f"c_{o}_chain"]
        a = cum_radial(d[f"I_{o}_oracle"], d[f"ax_{o}_oracle"], co[0], co[1],
                       edges)
        c = cum_radial(d[f"I_{o}_chain"], d[f"ax_{o}_chain"], cc[0], cc[1],
                       edges)
        print(f"{o:>8} " + ''.join(f"{v:8.4f}" for v in (a - c) * 100))
    print("\n   (the LAST column is 0 by construction: both arms are "
          "normalised by their own tile sum)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
