"""Summarise the probe4 ladders: the zeta envelope, the member ranking and the
multibranch power-ratio band, per fixture and per build.

Usage:  python summarise.py <ladder.json> [<ladder.json> ...]
"""
from __future__ import annotations

import json
import sys


def num(v, default=float('nan')):
    if v is None:
        return default
    if isinstance(v, str):
        return default
    return float(v)


def main():
    accepted, broken = [], []
    for path in sys.argv[1:]:
        d = json.load(open(path, encoding='cp1252'))
        h = d['header']
        print(f"\n=== {h['fixture']}  {h['note']}")
        print(f"    {h['lumenairy_file']}  py{h['python']} np{h['numpy']}  "
              f"N={h['N']} dx={h['dx'] * 1e6:.2f} um  index control "
              f"{h.get('index_control_delta')}")
        print(f"{'z_um':>9} {'zeta_x':>10} {'mb_P':>10} {'uni_fid':>8} "
              f"{'uni_P/or':>9} {'mb_fid':>8} {'rd_fid':>8} {'wv_fid':>8} "
              f"{'best':>12}")
        for r in d['rows']:
            ud = r.get('uniform_diag', {})
            m = r['members']
            zx = num(ud.get('zeta_extrapolation'))
            mbp = num(ud.get('power_ratio'))
            fid = {k: num(m.get(k, {}).get('fidelity'))
                   for k in ('uniform', 'multibranch', 'ray_density', 'wave')}
            up = num(m.get('uniform', {}).get('P_over_oracle'))
            best = max(fid, key=lambda k: (fid[k] if fid[k] == fid[k] else -1))
            print(f"{r['z_um']:9.2f} {zx:10.4g} {mbp:10.4g} "
                  f"{fid['uniform']:8.4f} {up:9.4g} {fid['multibranch']:8.4f} "
                  f"{fid['ray_density']:8.4f} {fid['wave']:8.4f} {best:>12}")
            row = (h['fixture'], r['z_um'], zx, mbp, fid['uniform'], up)
            if fid['uniform'] > 0.5:
                accepted.append(row)
            else:
                broken.append(row)
    print('\n=== multibranch power-ratio band across every ladder ===')
    if accepted:
        lo = min(r[3] for r in accepted)
        hi = max(r[3] for r in accepted)
        print(f'  ACCEPTED (uniform fidelity > 0.5): {len(accepted)} planes, '
              f'mb power_ratio {lo:.4g} .. {hi:.4g}')
    if broken:
        bl = min(r[3] for r in broken)
        print(f'  BROKEN   (uniform fidelity <= 0.5): {len(broken)} planes, '
              f'smallest mb power_ratio {bl:.4g}')
        if accepted:
            print(f'  gap: {hi:.4g} -> {bl:.4g}  ({bl / hi:.3g}x), '
                  f'geometric mean {(hi * bl) ** 0.5:.4g}')
    print('\n=== zeta envelope: uniform power / oracle vs zeta_extrapolation ==')
    rows = sorted((r for r in accepted if r[2] == r[2]), key=lambda r: r[2])
    for fx, z, zx, mbp, f, up in rows:
        print(f'  {fx} z={z:9.2f}  zeta_x={zx:10.4g}  P/P_oracle={up:8.4f}  '
              f'({100 * (up - 1):+7.2f} %)')


if __name__ == '__main__':
    main()
