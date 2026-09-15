"""Join the HEAD readings with the base tree's oracle scores and DERIVE the
pixel-halving arbiter's bar.

The continuity reading and the shipped decision can only be read on HEAD (at
a refused plane the reading is recovered from the refusal message, which names
it); the classification -- "does the oracle accept the field this plane
RETURNS" -- can only be read on a tree that returns a field at every plane,
i.e. the audit base 96cb2096.  The two are joined on ``(fixture, z)``.

Usage::

    python summarise.py <out.json> <head_*.json> <band_*_pre_*.json> ...
"""
# ruff: noqa: E402, I001
from __future__ import annotations

import json
import math
import sys


def _key(fixture, z_um):
    return (fixture, round(float(z_um), 3))


def load(paths):
    head, pre = {}, {}
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        fixture = d['fixture']
        for r in d['rows']:
            k = _key(fixture, r['z_um'])
            if 'bracket' in r:
                head[k] = dict(r, fixture=fixture, N=d.get('N'),
                               dx=d.get('dx'))
            if r.get('fid_uni') is not None:
                pre[k] = dict(r, fixture=fixture)
    return head, pre


def join(head, pre):
    rows = []
    for k in sorted(set(head) | set(pre)):
        h = head.get(k, {})
        p = pre.get(k, {})
        reason = h.get('reason') or p.get('uni_reason')
        rows.append({
            'fixture': k[0], 'z_um': k[1],
            'bracket': h.get('bracket'),
            'continuity_mb': h.get('continuity'),
            # at a plane refused on the LAUNCHED-POWER arm no field is
            # returned, so "the continuity of the returned field" does not
            # exist; the branch sum's own reading stands in, and the row is
            # marked so the two are never confused
            'continuity': (h.get('continuity_returned')
                           if h.get('continuity_returned') is not None
                           else h.get('continuity')),
            'continuity_is_branch_sum': (h.get('continuity_returned')
                                         is None),
            'continuity_decision': h.get('continuity_decision'),
            'n_branch_max': h.get('n_branch_max'),
            'reason': reason,
            'fold_ring': (reason == 'fold_ring'),
            'refused': h.get('refused'),
            'fid_uni': p.get('fid_uni'),
            'pow_uni': p.get('pow_uni_over_oracle'),
            'fid_wave': p.get('fid_wave'),
            'zeta_x': h.get('zeta_x'),
            'oracle_phi': p.get('oracle_phi'),
        })
    return rows


def _usable(rows, fold_only):
    out = []
    for r in rows:
        if r['continuity'] is None or r['fid_uni'] is None:
            continue
        if fold_only and not r['fold_ring']:
            continue
        out.append(r)
    return out


def separation(rows, bar, fold_only):
    """The criterion-free statement: what the guard's own decision does to the
    FIDELITY populations.  No accept bar is chosen; the split is the guard's."""
    use = _usable(rows, fold_only)
    ret = [r for r in use if r['continuity'] <= bar and not r['refused']]
    ref = [r for r in use if r['continuity'] > bar or r['refused']]
    # planes refused on the OTHER arm are not this arm's decision; separate
    ref_cont = [r for r in ref if r['continuity'] > bar]
    ref_pow = [r for r in ref if r['continuity'] <= bar]
    def _mm(rs, key):
        vals = [r[key] for r in rs if r[key] is not None]
        return (min(vals), max(vals)) if vals else (None, None)
    return {
        'bar': bar, 'fold_only': fold_only,
        'n_returned': len(ret), 'n_refused_continuity': len(ref_cont),
        'n_refused_power_only': len(ref_pow),
        'returned_continuity': _mm(ret, 'continuity'),
        'returned_fidelity': _mm(ret, 'fid_uni'),
        'refused_continuity': _mm(ref_cont, 'continuity'),
        'refused_fidelity': _mm(ref_cont, 'fid_uni'),
        'worst_returned': sorted(
            [f"{r['fixture']} z={r['z_um']} C={r['continuity']:.4f} "
             f"fid={r['fid_uni']:.4f} pow={r['pow_uni']:.4f} {r['reason']}"
             for r in ret if r['fid_uni'] is not None],
            key=lambda s: float(s.split('fid=')[1][:6]))[:6],
        'best_refused': sorted(
            [f"{r['fixture']} z={r['z_um']} C={r['continuity']:.4f} "
             f"fid={r['fid_uni']:.4f} pow={r['pow_uni']:.4f} {r['reason']}"
             for r in ref_cont if r['fid_uni'] is not None],
            key=lambda s: -float(s.split('fid=')[1][:6]))[:6],
    }


def derive(rows, fid_bar, fold_only):
    """Largest ACCEPTED and smallest BROKEN continuity reading at a given
    accept criterion, and the geometric-centre bar between them."""
    use = _usable(rows, fold_only)
    acc = [r for r in use if r['fid_uni'] >= fid_bar]
    bro = [r for r in use if r['fid_uni'] < fid_bar]
    acc_hi = max(acc, key=lambda r: r['continuity']) if acc else None
    acc_lo = min(acc, key=lambda r: r['continuity']) if acc else None
    gain = [r for r in bro if r['continuity'] > 1.0]
    bro_min = min(gain, key=lambda r: r['continuity']) if gain else None
    out = {
        'fid_bar': fid_bar, 'fold_only': fold_only,
        'n_accepted': len(acc), 'n_broken': len(bro),
        'accepted_min': (acc_lo['continuity'] if acc_lo else None),
        'accepted_min_at': (f"{acc_lo['fixture']} z={acc_lo['z_um']}"
                            if acc_lo else None),
        'accepted_max': (acc_hi['continuity'] if acc_hi else None),
        'accepted_max_at': (f"{acc_hi['fixture']} z={acc_hi['z_um']} "
                            f"fid={acc_hi['fid_uni']:.4f}" if acc_hi else None),
        'broken_min_gain': (bro_min['continuity'] if bro_min else None),
        'broken_min_gain_at': (f"{bro_min['fixture']} z={bro_min['z_um']} "
                               f"fid={bro_min['fid_uni']:.4f}"
                               if bro_min else None),
    }
    if acc_hi and bro_min and bro_min['continuity'] > acc_hi['continuity']:
        g = bro_min['continuity'] / acc_hi['continuity']
        c = math.sqrt(bro_min['continuity'] * acc_hi['continuity'])
        out.update(gap=g, bar_geometric_centre=c,
                   margin_above_accepted=c / acc_hi['continuity'],
                   margin_below_broken=bro_min['continuity'] / c)
    return out


def confusion(rows, bar, fid_bar, fold_only):
    use = _usable(rows, fold_only)
    tp = fp = tn = fn = 0
    fps, fns = [], []
    for r in use:
        broken = r['fid_uni'] < fid_bar
        refused = r['continuity'] > bar
        if broken and refused:
            tp += 1
        elif broken:
            fn += 1
            fns.append(r)
        elif refused:
            fp += 1
            fps.append(r)
        else:
            tn += 1

    def _fmt(rs):
        return [f"{r['fixture']} z={r['z_um']} C={r['continuity']:.4f} "
                f"fid={r['fid_uni']:.4f} pow={r['pow_uni']:.4f} "
                f"{r['reason']}" for r in rs]
    return {'bar': bar, 'fid_bar': fid_bar, 'fold_only': fold_only,
            'n': len(use),
            'refused_broken': tp, 'refused_accepted_FALSE': fp,
            'returned_broken_MISS': fn, 'returned_accepted': tn,
            'false_refusals': _fmt(fps), 'misses': _fmt(fns)}


def main(argv):
    out = argv[1]
    head, pre = load(argv[2:])
    rows = join(head, pre)
    bar = 1.06
    res = {'n_rows': len(rows), 'bar': bar, 'rows': rows,
           'separations': [], 'derivations': [], 'confusions': []}
    print(f"{'fx':7s} {'z_um':>9s} {'bracket':>11s} {'C_mb':>9s} {'C_ret':>9s} "
          f"{'fid_uni':>8s} {'pow_uni':>8s} {'reason':<24s} decision")
    for r in rows:
        print(f"{r['fixture']:7s} {r['z_um']:9.2f} {r['bracket']!s:>11.11} "
              f"{r['continuity_mb']!s:>9.9} {r['continuity']!s:>9.9} "
              f"{r['fid_uni']!s:>8.8} {r['pow_uni']!s:>8.8} "
              f"{str(r['reason']):<24.24} "
              f"{('REFUSED:' + str(r['refused'])) if r['refused'] else 'return'}")
    for fold_only in (True, False):
        sep = separation(rows, bar, fold_only)
        res['separations'].append(sep)
        tag = 'FOLD-RING' if fold_only else 'ALL planes'
        print(f"\n== {tag}: the guard's own split, no accept bar chosen ==")
        print(f"   returned  n={sep['n_returned']:3d}  C in "
              f"{sep['returned_continuity']}  fidelity "
              f"{sep['returned_fidelity']}")
        print(f"   refused   n={sep['n_refused_continuity']:3d}  C in "
              f"{sep['refused_continuity']}  fidelity "
              f"{sep['refused_fidelity']}")
        print(f"   (also refused on the launched-power arm alone: "
              f"{sep['n_refused_power_only']})")
        for line in sep['worst_returned']:
            print(f"     worst returned: {line}")
        for line in sep['best_refused']:
            print(f"     best refused  : {line}")
        for fid_bar in (0.883, 0.95):
            d = derive(rows, fid_bar, fold_only)
            res['derivations'].append(d)
            print(f"\n   -- accept bar fid>={fid_bar}: n="
                  f"{d['n_accepted']}+{d['n_broken']}  accepted "
                  f"{d['accepted_min']!s:.7}..{d['accepted_max']!s:.7} "
                  f"({d['accepted_max_at']})  smallest broken gain "
                  f"{d['broken_min_gain']!s:.7} ({d['broken_min_gain_at']})  "
                  f"gap {d.get('gap')!s:.6} centre "
                  f"{d.get('bar_geometric_centre')!s:.6}")
            c = confusion(rows, bar, fid_bar, fold_only)
            res['confusions'].append(c)
            print(f"      at bar={bar}: refused-broken {c['refused_broken']}, "
                  f"FALSE refusals {c['refused_accepted_FALSE']}, "
                  f"misses {c['returned_broken_MISS']}, "
                  f"returned-accepted {c['returned_accepted']}")
            for line in c['false_refusals']:
                print(f"         FALSE: {line}")
    with open(out, 'w') as fh:
        json.dump(res, fh, indent=1, default=str)
    return res


if __name__ == '__main__':
    main(sys.argv)
