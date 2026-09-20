"""The round-3 population, the gap, the two-sided margins and the BAR-COST
table.

Reads the scan JSONs (or their line logs), de-duplicates planes, and answers
four questions the round-2 report and its verification left open:

1. what are the populations and the gap on a population that is bigger than
   either round's, on optics neither derived the bar on (R3-1 / E1);
2. what are the bar's TWO-SIDED margins -- the distance from the bar down to
   the largest reading it returns and up to the smallest reading it refuses;
3. what does the shipped bar COST, in false refusals and misses, BY FIDELITY
   BAND rather than at one chosen accept criterion (which is a maintainer
   decision, not a measurement);
4. what would 1.04, 1.08 and a bar derived from the CONVERGED reading's own
   spread cost instead.

The reading column may come from a different tree than the fidelity column
(the R3-3 pixel-centre alignment moves the reading but not the returned
field).  When ``--readings`` is given, the two are joined per plane and the
invariant is CHECKED rather than assumed: the returned power must agree to
the bit, otherwise the plane is dropped and named.

Usage:
    python r3join.py <out.json> [--readings <glob-or-file> ...] -- <scan> ...
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

#: the fidelity bands every cost is reported in.  The two round-2 accept
#: criteria (0.883 and 0.95) are band EDGES so the earlier numbers can be
#: read straight off, but no band is privileged as "the" criterion.
BANDS = ((0.99, 1.01, 'fid >= 0.99'),
         (0.95, 0.99, '0.95 <= fid < 0.99'),
         (0.883, 0.95, '0.883 <= fid < 0.95'),
         (0.75, 0.883, '0.75 <= fid < 0.883'),
         (-1.0, 0.75, 'fid < 0.75'))


def rows_of(path):
    if path.endswith('.json'):
        with open(path, encoding='cp1252') as f:
            d = json.load(f)
        return d['rows'] if isinstance(d, dict) else d
    out = []
    with open(path, encoding='cp1252', errors='replace') as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith('{'):
                try:
                    out.append(json.loads(ln))
                except Exception:                           # noqa: BLE001
                    pass
    return out


def key_of(r):
    return (r.get('fixture'), round(float(r.get('z_um', -1)), 6),
            r.get('N'), round(float(r.get('dx_um', 0)), 9))


def collect(paths):
    seen = {}
    for p in paths:
        for q in sorted(glob.glob(p)) or [p]:
            if not os.path.exists(q):
                continue
            for r in rows_of(q):
                seen.setdefault(key_of(r), r)
    return seen


def band_of(fid):
    for lo, hi, name in BANDS:
        if lo <= fid < hi:
            return name
    return BANDS[-1][2]


def cost(pop, bar):
    """Confusion of a candidate bar, by fidelity band.

    A plane is REFUSED when its reading is above ``bar`` (the loss arm does
    not refuse, by design, so only the gain arm partitions).
    """
    ref = [r for r in pop if r['pixel_continuity'] > bar]
    ret = [r for r in pop if r['pixel_continuity'] <= bar]
    d = dict(bar=bar, n=len(pop), n_refused=len(ref), n_returned=len(ret))
    d['refused_by_band'] = {name: sum(1 for r in ref
                                      if band_of(r['fidelity']) == name)
                            for _, _, name in BANDS}
    d['returned_by_band'] = {name: sum(1 for r in ret
                                       if band_of(r['fidelity']) == name)
                             for _, _, name in BANDS}
    if ret:
        d['largest_returned'] = max(r['pixel_continuity'] for r in ret)
        d['margin_above'] = bar / d['largest_returned']
        d['worst_returned_fidelity'] = min(r['fidelity'] for r in ret)
    if ref:
        d['smallest_refused'] = min(r['pixel_continuity'] for r in ref)
        d['margin_below'] = d['smallest_refused'] / bar
        d['best_refused_fidelity'] = max(r['fidelity'] for r in ref)
    if ret and ref:
        d['gap'] = d['smallest_refused'] / d['largest_returned']
        d['gap_centre'] = math.sqrt(d['smallest_refused']
                                    * d['largest_returned'])
        d['two_sided_margin'] = min(d['margin_above'], d['margin_below'])
        d['fidelity_populations_overlap'] = bool(
            d['worst_returned_fidelity'] <= d['best_refused_fidelity'])
    # the named extremes, so every number above can be traced to a plane
    d['smallest_refused_rows'] = [
        dict(fixture=r['fixture'], z=r['z_um'], C=r['pixel_continuity'],
             fid=r['fidelity'], pw=r.get('power_over_oracle'),
             reason=r.get('reason'), fb=r.get('fell_back'))
        for r in sorted(ref, key=lambda r: r['pixel_continuity'])[:8]]
    d['largest_returned_rows'] = [
        dict(fixture=r['fixture'], z=r['z_um'], C=r['pixel_continuity'],
             fid=r['fidelity'], pw=r.get('power_over_oracle'),
             reason=r.get('reason'), fb=r.get('fell_back'))
        for r in sorted(ref + ret,
                        key=lambda r: -r['pixel_continuity'])[:1]
        ] + [
        dict(fixture=r['fixture'], z=r['z_um'], C=r['pixel_continuity'],
             fid=r['fidelity'], pw=r.get('power_over_oracle'),
             reason=r.get('reason'), fb=r.get('fell_back'))
        for r in sorted(ret, key=lambda r: -r['pixel_continuity'])[:8]]
    return d


def converged_spread(pop, tol=None):
    """The CONVERGED population's own reading distribution.

    "A converged quadrature reads 1" is the claim the bar is a tolerance on.
    The converged planes are taken to be those whose oracle fidelity is at
    least 0.99 AND whose returned power is within 2 % of the oracle's -- a
    FIELD criterion, so the reading is not used to select the planes it is
    then measured on.
    """
    conv = [r for r in pop
            if r['fidelity'] >= 0.99
            and abs((r.get('power_over_oracle') or 9.0) - 1.0) <= 0.02]
    cs = sorted(r['pixel_continuity'] for r in conv)
    if not cs:
        return dict(n=0)
    n = len(cs)

    def q(p):
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return cs[i]
    dev = max(abs(c - 1.0) for c in cs)
    return dict(n=n, min=cs[0], max=cs[-1], median=q(0.5), p01=q(0.01),
                p99=q(0.99), max_abs_deviation_from_1=dev,
                mean=sum(cs) / n,
                rms_deviation=math.sqrt(sum((c - 1.0) ** 2 for c in cs) / n))


def main():
    out_path = sys.argv[1]
    args = sys.argv[2:]
    rpaths, spaths = [], []
    cur = spaths
    for a in args:
        if a == '--readings':
            cur = rpaths
            continue
        if a == '--':
            cur = spaths
            continue
        cur.append(a)
    scored = collect(spaths)
    readings = collect(rpaths) if rpaths else {}
    joined, dropped = [], []
    for k, r in scored.items():
        if r.get('fidelity') is None or 'error' in r:
            continue
        row = dict(r)
        if readings:
            q = readings.get(k)
            if q is None or q.get('pixel_continuity') is None:
                dropped.append(dict(key=list(k), why='no post-fix reading'))
                continue
            if (q.get('returned_power') is None
                    or r.get('returned_power') is None
                    or q['returned_power'] != r['returned_power']):
                dropped.append(dict(key=list(k), why='returned power moved',
                                    a=r.get('returned_power'),
                                    b=q.get('returned_power')))
                continue
            row['pixel_continuity_pre'] = r['pixel_continuity']
            row['pixel_continuity'] = q['pixel_continuity']
            row['pixel_continuity_of'] = q.get('pixel_continuity_of')
            row['multibranch_pixel_continuity'] = q.get(
                'multibranch_pixel_continuity')
        if row.get('pixel_continuity') is None:
            continue
        joined.append(row)
    fold = [r for r in joined
            if r.get('reason') == 'fold_ring' and not r.get('fell_back')]
    fb = [r for r in joined if r.get('fell_back')]
    cusp = [r for r in joined if r.get('reason') == 'cusp_ring']
    res = dict(n_joined=len(joined), n_dropped=len(dropped),
               dropped=dropped[:40],
               n_fold_ring=len(fold), n_fallback=len(fb), n_cusp=len(cusp),
               optics=sorted({r['fixture'] for r in joined}))
    res['converged_spread'] = {
        'all_planes': converged_spread(joined),
        'fold_ring': converged_spread(fold)}
    spread = res['converged_spread']['all_planes'].get(
        'max_abs_deviation_from_1')
    bars = [1.04, 1.06, 1.08]
    if spread:
        for k in (3, 10, 30):
            bars.append(round(1.0 + k * spread, 6))
    res['bars'] = sorted(set(bars))
    res['cost'] = {}
    for label, pop in (('fold_ring', fold), ('all_planes', joined),
                       ('fallback', fb)):
        res['cost'][label] = {('%.6g' % b): cost(pop, b)
                              for b in res['bars'] if pop}
    if fold and joined:
        res['derived_centre'] = {
            'fold_ring': res['cost']['fold_ring']['1.06'].get('gap_centre'),
            'all_planes': res['cost']['all_planes']['1.06'].get('gap_centre')}
    with open(out_path, 'w', encoding='cp1252') as f:
        json.dump(dict(summary=res, rows=joined), f, indent=1)
    slim = dict(res)
    slim.pop('dropped', None)
    print(json.dumps(slim, indent=1, default=float))


if __name__ == '__main__':
    main()
