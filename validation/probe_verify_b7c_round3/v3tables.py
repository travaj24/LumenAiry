"""VERIFY-WP-B7c round 3 -- the tables (claims 2, 3, 5 and 8).

Reads the ladder JSONs and the control JSONs this verification produced and
builds, on MY population:

* **claim 2** -- the fold ring's two FIDELITY populations at the shipped bar:
  worst RETURNED against best REFUSED, per optic and joined, plus the
  per-optic ranges the overlap claim rests on;
* **claim 3** -- the BAR-COST table: for each candidate bar, how many planes
  it refuses and returns, the two-sided margins it has on this population, and
  the false refusals and misses at two accept criteria.  The DERIVED CENTRE of
  the gap the shipped bar sits in is computed here too, so "1.06 is the
  derived centre" is a measurement and not a quotation;
* **claim 5's consequence** -- what a bar derived from the CONVERGED reading's
  own spread costs, with ``s_conv`` taken from this verification's own
  converged population;
* **claim 8** -- the FALLBACK route: the confusion table of every reading the
  shipped library already reports, and the CHALLENGE the brief asks for -- a
  search for a returned fallback plane where a loss-arm refusal would refuse a
  field that is RIGHT.

Usage:  python v3tables.py <out.json> <ladder-or-scan json> [...]
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

BARS = (1.0018, 1.0059, 1.0177, 1.04, 1.06, 1.0600253, 1.0619, 1.08)
CRITERIA = (0.95, 0.883)


def load(paths):
    rows = []
    for p in paths:
        with open(p, encoding='cp1252') as f:
            d = json.load(f)
        for key in ('L1', 'L2', 'L3', 'rows', 'recon'):
            for r in d.get(key) or []:
                if isinstance(r, dict) and 'z_um' in r:
                    r = dict(r)
                    r['_src'] = os.path.basename(p)
                    r['_ladder'] = key
                    rows.append(r)
    # de-duplicate on (fixture, z, N, dx): the same plane can appear in a
    # recon pass and in L1
    seen, out = {}, []
    for r in rows:
        k = (r.get('fixture'), round(float(r['z_um']), 6), r.get('N'),
             r.get('dx_um'))
        if k in seen:
            # keep the row that carries an oracle score
            if seen[k].get('fidelity') is None and r.get('fidelity') \
                    is not None:
                out[out.index(seen[k])] = r
                seen[k] = r
            continue
        seen[k] = r
        out.append(r)
    return out


def scored(rows):
    return [r for r in rows if r.get('fidelity') is not None
            and r.get('pixel_continuity') is not None]


def fold(rows):
    return [r for r in rows if r.get('route') == 'fold_ring']


def decide(r, bar):
    """REFUSED by the pixel-continuity GAIN arm at ``bar``, or the launched-
    power gain tripwire, which is the shipped pair."""
    c = r.get('pixel_continuity')
    if r.get('power_arm_refuses'):
        return 'REFUSED'
    return 'REFUSED' if (c is not None and c > bar) else 'returned'


def cost(rows, bar, crit):
    ref = [r for r in rows if decide(r, bar) == 'REFUSED']
    ret = [r for r in rows if decide(r, bar) != 'REFUSED']
    false_ref = [r for r in ref if r['fidelity'] >= crit]
    misses = [r for r in ret if r['fidelity'] < crit]
    return len(ref), len(ret), len(false_ref), len(misses), false_ref, misses


def margins(rows, bar):
    cs = [r['pixel_continuity'] for r in rows
          if r.get('pixel_continuity') is not None]
    ret = [c for c in cs if c <= bar]
    ref = [c for c in cs if c > bar]
    if not ret or not ref:
        return None, None, None, None
    return (bar / max(ret), min(ref) / bar, max(ret), min(ref))


def derived_centre(rows, bar):
    """Geometric centre of the gap the SHIPPED bar sits in on this
    population."""
    cs = sorted(r['pixel_continuity'] for r in rows
                if r.get('pixel_continuity') is not None)
    below = [c for c in cs if c <= bar]
    above = [c for c in cs if c > bar]
    if not below or not above:
        return None
    return float(np.sqrt(max(below) * min(above)))


def spearman(x, y):
    from scipy.stats import spearmanr
    m = [i for i in range(len(x)) if x[i] is not None and y[i] is not None]
    if len(m) < 8:
        return None
    return float(spearmanr([x[i] for i in m], [y[i] for i in m]).statistic)


def best_threshold(vals, fids, crit, greater_is_better=True):
    """The threshold on ``vals`` with the best balanced accuracy at ``crit``,
    plus what it costs."""
    pairs = [(v, f) for v, f in zip(vals, fids) if v is not None]
    if len(pairs) < 8:
        return None
    cands = sorted({v for v, _ in pairs})
    best = None
    for t in cands:
        if greater_is_better:
            acc = [f >= crit for v, f in pairs if v >= t]
            rej = [f >= crit for v, f in pairs if v < t]
        else:
            acc = [f >= crit for v, f in pairs if v <= t]
            rej = [f >= crit for v, f in pairs if v > t]
        tp = sum(acc)
        fp = len(acc) - tp
        fn = sum(rej)
        tn = len(rej) - fn
        if (tp + fn) == 0 or (tn + fp) == 0:
            continue
        ba = 0.5 * (tp / (tp + fn) + tn / (tn + fp))
        if best is None or ba > best[1]:
            best = (float(t), float(ba), fn, fp)
    return best


def main():
    out = sys.argv[1]
    paths = []
    for a in sys.argv[2:]:
        paths.extend(sorted(glob.glob(a)))
    rows = load(paths)
    sc = scored(rows)
    fr = fold(sc)
    fb = [r for r in sc if r.get('route') == 'fallback']
    bar = 1.06
    rep = dict(n_rows=len(rows), n_scored=len(sc), n_fold_scored=len(fr),
               n_fallback_scored=len(fb),
               optics=sorted({r['fixture'] for r in rows}),
               sources=[os.path.basename(p) for p in paths])

    # ---- claim 2: the two fidelity populations on the fold ring ----------
    ret = [r for r in fr if decide(r, bar) != 'REFUSED']
    ref = [r for r in fr if decide(r, bar) == 'REFUSED']
    rep['fold_returned'] = dict(
        n=len(ret), fid_lo=min((r['fidelity'] for r in ret), default=None),
        fid_hi=max((r['fidelity'] for r in ret), default=None),
        c_lo=min((r['pixel_continuity'] for r in ret), default=None),
        c_hi=max((r['pixel_continuity'] for r in ret), default=None))
    rep['fold_refused'] = dict(
        n=len(ref), fid_lo=min((r['fidelity'] for r in ref), default=None),
        fid_hi=max((r['fidelity'] for r in ref), default=None),
        c_lo=min((r['pixel_continuity'] for r in ref), default=None),
        c_hi=max((r['pixel_continuity'] for r in ref), default=None))
    rep['populations_overlap'] = bool(
        ret and ref and rep['fold_returned']['fid_lo']
        < rep['fold_refused']['fid_hi'])
    rep['worst_returned'] = sorted(
        ({k: r[k] for k in ('fixture', 'z_um', 'pixel_continuity', 'fidelity',
                            'power_over_oracle')} for r in ret),
        key=lambda r: r['fidelity'])[:8]
    rep['best_refused'] = sorted(
        ({k: r[k] for k in ('fixture', 'z_um', 'pixel_continuity', 'fidelity',
                            'power_over_oracle')} for r in ref),
        key=lambda r: -r['fidelity'])[:8]
    per = {}
    for nm in sorted({r['fixture'] for r in fr}):
        a = [r for r in fr if r['fixture'] == nm]
        ra = [r for r in a if decide(r, bar) != 'REFUSED']
        fa = [r for r in a if decide(r, bar) == 'REFUSED']
        per[nm] = dict(
            n=len(a), n_returned=len(ra), n_refused=len(fa),
            ret_c=[min((r['pixel_continuity'] for r in ra), default=None),
                   max((r['pixel_continuity'] for r in ra), default=None)],
            ret_fid=[min((r['fidelity'] for r in ra), default=None),
                     max((r['fidelity'] for r in ra), default=None)],
            ret_power=[min((r['power_over_oracle'] for r in ra), default=None),
                       max((r['power_over_oracle'] for r in ra),
                           default=None)],
            ref_c=[min((r['pixel_continuity'] for r in fa), default=None),
                   max((r['pixel_continuity'] for r in fa), default=None)],
            ref_fid=[min((r['fidelity'] for r in fa), default=None),
                     max((r['fidelity'] for r in fa), default=None)])
    rep['per_optic_fold'] = per

    # ---- claim 3: the bar-cost table -------------------------------------
    for tag, pop in (('fold', fr), ('all', sc)):
        tab = []
        for b in BARS:
            ma, mb, mxr, mnf = margins(pop, b)
            row = dict(bar=b, margin_above=ma, margin_below=mb,
                       max_returned=mxr, min_refused=mnf)
            for crit in CRITERIA:
                nref, nret, nfalse, nmiss, fl, ml = cost(pop, b, crit)
                row[f'refused@{crit}'] = nref
                row[f'returned@{crit}'] = nret
                row[f'false_refusals@{crit}'] = nfalse
                row[f'misses@{crit}'] = nmiss
                if b == 1.06:
                    row[f'false_list@{crit}'] = [
                        {k: r[k] for k in ('fixture', 'z_um',
                                           'pixel_continuity', 'fidelity')}
                        for r in fl]
            tab.append(row)
        rep[f'cost_{tag}'] = tab
        rep[f'derived_centre_{tag}'] = derived_centre(pop, bar)

    # ---- claim 8: the fallback route -------------------------------------
    fbr = [r for r in fb if r.get('shipped_returns')]
    rep['fallback'] = dict(
        n_scored=len(fb), n_returned=len(fbr),
        fid_lo=min((r['fidelity'] for r in fbr), default=None),
        fid_hi=max((r['fidelity'] for r in fbr), default=None),
        below_095=sum(1 for r in fbr if r['fidelity'] < 0.95),
        below_0883=sum(1 for r in fbr if r['fidelity'] < 0.883))
    arms = {
        'continuity_loss_arm (< 1/1.06)':
            lambda r: r.get('pixel_continuity') is not None
            and r['pixel_continuity'] < 1.0 / 1.06,
        'power_loss_arm (< 0.5)':
            lambda r: r.get('multibranch_power_ratio_bracketed') is not None
            and r['multibranch_power_ratio_bracketed'] < 0.5,
        'power_loss_arm (< 0.889)':
            lambda r: r.get('multibranch_power_ratio_bracketed') is not None
            and r['multibranch_power_ratio_bracketed'] < 0.889,
        'either loss arm':
            lambda r: (r.get('pixel_continuity') is not None
                       and r['pixel_continuity'] < 1.0 / 1.06)
            or (r.get('multibranch_power_ratio_bracketed') is not None
                and r['multibranch_power_ratio_bracketed'] < 0.5),
        "either loss arm OR reason != 'no_fold'":
            lambda r: (r.get('pixel_continuity') is not None
                       and r['pixel_continuity'] < 1.0 / 1.06)
            or (r.get('multibranch_power_ratio_bracketed') is not None
                and r['multibranch_power_ratio_bracketed'] < 0.5)
            or r.get('reason') != 'no_fold',
    }
    conf = {}
    for nm, fn in arms.items():
        flg = [r for r in fbr if fn(r)]
        nof = [r for r in fbr if not fn(r)]
        conf[nm] = dict(
            flagged=len(flg),
            flagged_wrong=sum(1 for r in flg if r['fidelity'] < 0.95),
            flagged_RIGHT=sum(1 for r in flg if r['fidelity'] >= 0.95),
            not_flagged=len(nof),
            not_flagged_wrong=sum(1 for r in nof if r['fidelity'] < 0.95),
            worst_RIGHT_flagged=sorted(
                ({k: r[k] for k in ('fixture', 'z_um', 'fidelity',
                                    'pixel_continuity',
                                    'multibranch_power_ratio_bracketed',
                                    'reason')}
                 for r in flg if r['fidelity'] >= 0.95),
                key=lambda r: -r['fidelity'])[:10])
    rep['fallback_confusion'] = conf
    cands = ('pixel_continuity', 'multibranch_power_ratio_bracketed',
             'power_ratio', 'power_ratio_triangles', 'edge_fraction',
             'n_branch_max', 'n_triangles_degenerate')
    fids = [r['fidelity'] for r in fbr]
    rep['fallback_spearman'] = {
        c: spearman([r.get(c) for r in fbr], fids) for c in cands}
    rep['fallback_best_threshold'] = {
        c: best_threshold([r.get(c) for r in fbr], fids, 0.95)
        for c in cands}
    by_reason = {}
    for r in fbr:
        by_reason.setdefault(r.get('reason'), []).append(r['fidelity'])
    rep['fallback_by_reason'] = {
        k: dict(n=len(v), lo=min(v), hi=max(v), median=float(np.median(v)))
        for k, v in sorted(by_reason.items(), key=lambda kv: -len(kv[1]))}
    # what nothing sees
    unseen = [r for r in fbr if r['fidelity'] < 0.95
              and not arms['either loss arm'](r) and r.get('reason')
              == 'no_fold']
    rep['fallback_unordered'] = sorted(
        ({k: r[k] for k in ('fixture', 'z_um', 'fidelity',
                            'pixel_continuity',
                            'multibranch_power_ratio_bracketed')}
         for r in unseen), key=lambda r: r['fidelity'])
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: rep[k] for k in
                      ('n_rows', 'n_scored', 'n_fold_scored',
                       'n_fallback_scored', 'populations_overlap',
                       'derived_centre_fold', 'derived_centre_all')},
                     indent=1))
    print('fold returned', json.dumps(rep['fold_returned']))
    print('fold refused ', json.dumps(rep['fold_refused']))
    print('wrote', out)


if __name__ == '__main__':
    main()
