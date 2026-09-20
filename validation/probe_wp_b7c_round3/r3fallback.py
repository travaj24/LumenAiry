"""R3-4 / E7 -- does ANYTHING order the fallback route's fidelity?

On the fallback route the field returned is the bright-side-only branch sum:
the uniform completion has declined, so the dark Airy tail the fold would
have supplied is simply absent.  The round-2 verification found the arbiter
silent there -- 62 returned planes below fidelity 0.95 and one at **0.4639**
with ``pixel_continuity`` 1.00201, the launched-power bracket 1.005, both
decisions ``'ok'`` and no warning from either arm.

The brief asks for a THIRD reading that orders that population if one exists.
This probe answers it from the population rather than from an opinion.  Every
candidate is scored two ways:

* SPEARMAN rank correlation against the oracle fidelity (does it order the
  population at all?);
* the best SEPARATION any threshold on it achieves between "the field is
  right" and "the field is wrong" at each fidelity criterion, with the
  two-sided margin that threshold would have.  A candidate with a high
  correlation but no gap is not a bar.

Candidates already in the diagnostics: the two power ratios, the bracketed
one, the branch-sum continuity reading, the branch census, the fallback
reason and the fraction of the returned energy in the outer tenth of the
window.  Two more are computed here from the ray trace alone, because they
are the two quantities the failure mode actually is:

``go_over_airy``     the GEOMETRICAL spot radius at this plane over the
                     diffraction width ``lambda z / D``.  The fallback field
                     is a geometrical-optics field; when the geometrical
                     structure is finer than the diffraction width, no
                     geometrical field can be right at any pitch, whatever
                     the quadrature has done.
``go_blur_um``       the geometrical spot radius on its own, and ``airy_um``
                     the diffraction width on its own, so that a correlation
                     with their RATIO can be told from a correlation with
                     either alone.

The population is the one E7 is about: the fallback planes the SHIPPED
library RETURNS.  A scan with the refusal bars lifted also contains the
planes the library refuses -- 304 of the 623 here -- and including them would
credit a candidate with ordering cases no caller ever sees.  Both populations
are reported; the analysis is on the returned one.

Usage:
    python r3fallback.py <out.json> <joined.json>
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import r3fixtures as FX
import r3oracle as OR

CANDIDATES = ('pixel_continuity', 'multibranch_pixel_continuity',
              'power_ratio', 'power_ratio_triangles',
              'multibranch_power_ratio_bracketed', 'edge_fraction',
              'n_branch_max', 'go_over_airy', 'go_blur_um',
              'airy_um')


def spearman(xs, ys):
    n = len(xs)
    if n < 4:
        return None

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = 0.5 * (i + j) + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    syy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if sxx == 0.0 or syy == 0.0:
        return None
    return sxy / (sxx * syy)


def best_threshold(vals, good):
    """Best single threshold on ``vals`` separating good from bad, in both
    directions, with the gap it leaves.

    Returns the direction, the threshold's own gap (the ratio of the nearest
    value on each side, which is the margin a bar there would have), and the
    confusion it achieves.
    """
    best = None
    for sign in (+1, -1):
        v = [sign * x for x in vals]
        order = sorted(range(len(v)), key=lambda i: v[i])
        # sweep the threshold between every adjacent pair
        for i in range(len(order) - 1):
            lo, hi = v[order[i]], v[order[i + 1]]
            if lo == hi:
                continue
            t = 0.5 * (lo + hi)
            tp = sum(1 for k, x in enumerate(v) if x <= t and good[k])
            fp = sum(1 for k, x in enumerate(v) if x <= t and not good[k])
            fn = sum(1 for k, x in enumerate(v) if x > t and good[k])
            tn = sum(1 for k, x in enumerate(v) if x > t and not good[k])
            ng, nb = tp + fn, fp + tn
            if ng == 0 or nb == 0:
                continue
            bal = 0.5 * (tp / ng + tn / nb)
            rec = dict(direction=('accept_below' if sign > 0
                                  else 'accept_above'),
                       threshold=sign * t, balanced_accuracy=bal,
                       gap=abs(hi / lo) if lo != 0 else None,
                       true_good=tp, false_refuse=fn, missed_bad=fp,
                       true_bad=tn)
            if best is None or bal > best['balanced_accuracy']:
                best = rec
    return best


def geometry_candidates(row):
    fx = FX.FIXTURES[row['fixture']]
    wl = fx['wavelength']
    z = row['z_um'] * 1e-6
    D = float(fx['prescription']['aperture_diameter'])
    h, y, opl, amp, yl, P_in = OR.exit_field(fx['prescription'], wl, fx['w0'],
                                             z, n_fan=1200)
    if yl.size == 0:
        return {}
    blur = float(np.max(np.abs(yl)))
    airy = wl * max(z, 1e-12) / D
    return dict(go_blur_um=blur * 1e6, airy_um=airy * 1e6,
                go_over_airy=blur / max(airy, 1e-30))


def main():
    out_path, joined = sys.argv[1], sys.argv[2]
    with open(joined, encoding='cp1252') as f:
        rows = json.load(f)['rows']
    fb_all = [r for r in rows if r.get('fell_back')
              and r.get('fidelity') is not None]
    #: E7's population: what the SHIPPED bars actually return
    fb = [r for r in fb_all if r.get('shipped_decision') != 'REFUSED'
          and not r.get('power_arm_refuses')]
    print('fallback planes: %d scored, %d returned by the shipped bars'
          % (len(fb_all), len(fb)), flush=True)
    for r in fb:
        try:
            r.update(geometry_candidates(r))
        except Exception as exc:                            # noqa: BLE001
            r['geom_error'] = f'{type(exc).__name__}: {exc}'
    res = dict(n=len(fb), n_scored=len(fb_all),
               optics=sorted({r['fixture'] for r in fb}),
               fidelity_range=[min(r['fidelity'] for r in fb),
                               max(r['fidelity'] for r in fb)] if fb else None,
               reasons={})
    for r in fb:
        k = str(r.get('reason'))
        d = res['reasons'].setdefault(k, dict(n=0, fid=[]))
        d['n'] += 1
        d['fid'].append(r['fidelity'])
    for k, d in res['reasons'].items():
        d['fid_min'], d['fid_max'] = min(d['fid']), max(d['fid'])
        d['fid_median'] = sorted(d['fid'])[len(d['fid']) // 2]
        d.pop('fid')
    # two DERIVED candidates that are not scalars: the loss arm's own
    # decision, and the fallback reason.  Reported as their confusion rather
    # than as a correlation, because a boolean has no rank.
    res['derived'] = {}
    for crit in (0.95, 0.883):
        wrong = [r for r in fb if r['fidelity'] < crit]
        for nm, pred in (
                ('continuity_loss_arm', lambda r: (
                    r.get('pixel_continuity_decision')
                    == 'not_converged_loss'
                    or (r['pixel_continuity'] is not None
                        and r['pixel_continuity'] < 1.0 / 1.06))),
                ('power_bracket_below_0.90', lambda r: (
                    (r.get('multibranch_power_ratio_bracketed') or 1.0)
                    < 0.90)),
                ('reason_is_not_no_fold',
                 lambda r: r.get('reason') != 'no_fold'),
                ('either_loss_arm_or_reason', lambda r: (
                    (r['pixel_continuity'] is not None
                     and r['pixel_continuity'] < 1.0 / 1.06)
                    or r.get('reason') != 'no_fold'))):
            flagged = [r for r in fb if pred(r)]
            kept = [r for r in fb if not pred(r)]
            res['derived'][f'{nm}@{crit}'] = dict(
                n_flagged=len(flagged),
                flagged_wrong=sum(1 for r in flagged
                                  if r['fidelity'] < crit),
                flagged_right=sum(1 for r in flagged
                                  if r['fidelity'] >= crit),
                kept=len(kept),
                kept_wrong=sum(1 for r in kept if r['fidelity'] < crit),
                n_wrong=len(wrong))
    res['candidates'] = {}
    for c in CANDIDATES:
        pop = [r for r in fb if isinstance(r.get(c), (int, float))
               and r.get(c) is not None]
        if len(pop) < 6:
            res['candidates'][c] = dict(n=len(pop), note='too few')
            continue
        vals = [float(r[c]) for r in pop]
        fids = [float(r['fidelity']) for r in pop]
        d = dict(n=len(pop), spearman_vs_fidelity=spearman(vals, fids),
                 range=[min(vals), max(vals)])
        for crit in (0.95, 0.883):
            good = [f >= crit for f in fids]
            if 0 < sum(good) < len(good):
                d[f'best_threshold_at_{crit}'] = best_threshold(vals, good)
        res['candidates'][c] = d
    # the single worst returned plane, named
    worst = sorted(fb, key=lambda r: r['fidelity'])[:8]
    res['worst_returned'] = [
        dict(fixture=r['fixture'], z=r['z_um'], fid=r['fidelity'],
             C=r.get('pixel_continuity'),
             bracket=r.get('multibranch_power_ratio_bracketed'),
             decision=r.get('pixel_continuity_decision'),
             power_decision=r.get('power_ratio_decision'),
             reason=r.get('reason'), warnings=r.get('warnings'),
             go_over_airy=r.get('go_over_airy'),
             bright_fraction=r.get('bright_fraction'))
        for r in worst]
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(dict(summary=res, rows=fb), f, indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
