"""Join the per-fixture band ladders of both builds and re-derive the band.

The classification -- "the oracle accepts this completed field" -- can only be
read on a build that still RETURNS it, so the base tree's ladder supplies
``fid_uni`` / ``pow_uni_over_oracle`` for the planes the head refuses, and the
head's ladder supplies the DECISION.  The two are joined on ``z_um``.

Prints: the accepted / broken populations of the bracketed multibranch ratio,
the closest accepted reading from below and the mildest broken one from above,
the resulting margins, and the member ranking per plane.

Usage: python summarise.py <head_glob_tag> <base_glob_tag> [accept_fidelity]
"""
from __future__ import annotations

import glob
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def load(tag):
    out = {}
    for p in sorted(glob.glob(os.path.join(_HERE, f'band_*_{tag}.json'))):
        d = json.load(open(p))
        out[d['fixture']] = d
    return out


def main(argv):
    head = load(argv[1])
    base = load(argv[2])
    acc = float(argv[3]) if len(argv) > 3 else 0.883
    rows = []
    for fixname, hd in head.items():
        bd = base.get(fixname)
        bz = {round(r['z_um'], 4): r for r in bd['rows']} if bd else {}
        for r in hd['rows']:
            b = bz.get(round(r['z_um'], 4), {})
            fid = r['fid_uni'] if r['fid_uni'] is not None else b.get('fid_uni')
            pw = (r['pow_uni_over_oracle']
                  if r['pow_uni_over_oracle'] is not None
                  else b.get('pow_uni_over_oracle'))
            rows.append({
                'fixture': fixname, 'z_um': r['z_um'],
                'bracket': r['mb_bracket'], 'n_branch_max': r['n_branch_max'],
                'reason': r.get('uni_reason') or (
                    'REFUSED' if r['uni_error'] else b.get('uni_reason')),
                'refused_by_head': r['uni_error'] is not None,
                'fid_uni': fid, 'pow_uni_over_oracle': pw,
                'fid_wave': r['fid_wave'],
                'pow_wave_over_oracle': r['pow_wave_over_oracle'],
                'fid_mb': r['fid_mb'], 'fid_screen': r.get('fid_screen'),
                'zeta_x': r.get('uni_zeta_extrapolation')
                or b.get('uni_zeta_extrapolation'),
                'oracle_closure': r['oracle_energy_closure'],
            })
    rows = [r for r in rows if r['fid_uni'] is not None
            and r['bracket'] is not None]
    accepted = [r for r in rows if r['fid_uni'] >= acc]
    broken = [r for r in rows if r['fid_uni'] < acc]
    print(f'{len(rows)} scored planes; accept bar fidelity >= {acc}')
    print(f'accepted n={len(accepted)} broken n={len(broken)}')
    if accepted:
        a = max(accepted, key=lambda r: r['bracket'])
        print(f"largest ACCEPTED bracket {a['bracket']:.4g} "
              f"({a['fixture']} z={a['z_um']:.2f} fid={a['fid_uni']:.4f} "
              f"reason={a['reason']})")
    if broken:
        b = min(broken, key=lambda r: r['bracket'])
        print(f"smallest BROKEN bracket  {b['bracket']:.4g} "
              f"({b['fixture']} z={b['z_um']:.2f} fid={b['fid_uni']:.4f} "
              f"reason={b['reason']})")
    if accepted and broken:
        print(f"gap = {b['bracket'] / a['bracket']:.3f}x  "
              f"margin_below_bar = {2.0 / a['bracket']:.3f}x  "
              f"margin_above_bar = {b['bracket'] / 2.0:.3f}x")
    fp = [r for r in rows if r['refused_by_head'] and r['fid_uni'] >= acc]
    fn = [r for r in rows if not r['refused_by_head'] and r['fid_uni'] < acc]
    print(f'FALSE REFUSALS (refused, oracle-accepted): {len(fp)}')
    for r in fp:
        print('   ', r['fixture'], round(r['z_um'], 2),
              round(r['bracket'], 4), round(r['fid_uni'], 4), r['reason'])
    print(f'MISSED (returned, oracle-broken): {len(fn)}')
    for r in fn:
        print('   ', r['fixture'], round(r['z_um'], 2),
              round(r['bracket'], 4), round(r['fid_uni'], 4), r['reason'])
    print()
    hdr = ('fixture     z_um  bracket   nbr  reason            zeta_x'
           '    fid_uni  fid_wave   fid_mb  pow_uni  pow_wave  refused')
    print(hdr)
    for r in sorted(rows, key=lambda r: (r['fixture'], r['z_um'])):
        zx = r['zeta_x']
        print(f"{r['fixture']:<6} {r['z_um']:9.2f} {r['bracket']:9.4g} "
              f"{r['n_branch_max']:>4} {str(r['reason'])[:18]:<18} "
              f"{('%.4g' % zx) if zx else '-':>9} "
              f"{r['fid_uni']:8.4f} {r['fid_wave']:8.4f} {r['fid_mb']:8.4f} "
              f"{r['pow_uni_over_oracle']:8.4g} "
              f"{r['pow_wave_over_oracle']:8.4f} "
              f"{'Y' if r['refused_by_head'] else '.'}")
    json.dump(rows, open(os.path.join(_HERE, f'joined_{argv[1]}.json'), 'w'),
              indent=1)


if __name__ == '__main__':
    main(sys.argv)
