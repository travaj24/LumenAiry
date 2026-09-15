"""Join every scan into the populations, the gap and the confusion matrix.

Usage:  python vjoin.py <out.json> <log-or-json> [...]
"""
from __future__ import annotations

import json
import sys

import vsum


def main():
    out = sys.argv[1]
    rows = []
    for p in sys.argv[2:]:
        for r in vsum.rows(p):
            if 'error' in r or r.get('fidelity') is None:
                if 'error' in r:
                    rows.append(r)
                continue
            rows.append(r)
    # dedupe: a plane may appear in more than one ladder
    seen = {}
    for r in rows:
        k = (r.get('fixture'), round(float(r.get('z_um', -1)), 4),
             r.get('N'), r.get('dx_um'))
        seen.setdefault(k, r)
    rows = list(seen.values())
    scored = [r for r in rows if r.get('fidelity') is not None
              and r.get('pixel_continuity') is not None]
    fold = [r for r in scored if r.get('reason') == 'fold_ring'
            and not r.get('fell_back')]
    res = {}
    for label, pop in (('fold_ring', fold), ('all_planes', scored)):
        ret = [r for r in pop if r['shipped_decision'] != 'REFUSED']
        ref = [r for r in pop if r['shipped_decision'] == 'REFUSED']
        d = dict(n=len(pop), n_returned=len(ret), n_refused=len(ref))
        if ret:
            d['returned_continuity'] = [min(r['pixel_continuity']
                                            for r in ret),
                                        max(r['pixel_continuity']
                                            for r in ret)]
            d['returned_fidelity'] = [min(r['fidelity'] for r in ret),
                                      max(r['fidelity'] for r in ret)]
        if ref:
            d['refused_continuity'] = [min(r['pixel_continuity']
                                           for r in ref),
                                       max(r['pixel_continuity']
                                           for r in ref)]
            d['refused_fidelity'] = [min(r['fidelity'] for r in ref),
                                     max(r['fidelity'] for r in ref)]
        if ret and ref:
            lo = max(r['pixel_continuity'] for r in ret)
            hi = min(r['pixel_continuity'] for r in ref)
            d['gap'] = dict(largest_returned=lo, smallest_refused=hi,
                            ratio=hi / lo if lo > 0 else None,
                            populations_overlap=bool(hi <= lo))
            d['fidelity_overlap'] = bool(
                ret and ref and min(r['fidelity'] for r in ret)
                <= max(r['fidelity'] for r in ref))
        for bar in (0.883, 0.95):
            fr = [dict(fixture=r['fixture'], z=r['z_um'],
                       C=r['pixel_continuity'], fid=r['fidelity'],
                       pw=r.get('power_over_oracle'), reason=r.get('reason'),
                       fb=r.get('fell_back'))
                  for r in ref if r['fidelity'] >= bar]
            miss = [dict(fixture=r['fixture'], z=r['z_um'],
                         C=r['pixel_continuity'], fid=r['fidelity'],
                         reason=r.get('reason'), fb=r.get('fell_back'))
                    for r in ret if r['fidelity'] < bar]
            d[f'accept_bar_{bar}'] = dict(
                false_refusals=len(fr), misses=len(miss),
                refused_broken=len(ref) - len(fr),
                returned_accepted=len(ret) - len(miss),
                false_refusal_rows=fr[:20], miss_rows=miss[:40])
        res[label] = d
    res['n_rows'] = len(rows)
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(dict(summary=res, rows=rows), f, indent=1)
    print(json.dumps(res, indent=1, default=float))


if __name__ == '__main__':
    main()
