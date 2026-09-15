"""Compact table of a scan log / json."""
import glob
import json
import sys


def rows(path):
    if path.endswith('.json'):
        d = json.load(open(path, encoding='cp1252'))
        return d['rows'] if isinstance(d, dict) else d
    out = []
    for ln in open(path, encoding='cp1252', errors='replace'):
        ln = ln.strip()
        if ln.startswith('{'):
            try:
                out.append(json.loads(ln))
            except Exception:                              # noqa: BLE001
                pass
    return out


def fmt(v, n='%.4g'):
    return 'None' if v is None else (n % v if isinstance(v, float) else str(v))


def main():
    for path in sys.argv[1:]:
        for p in sorted(glob.glob(path)):
            print('##', p)
            print('%9s %10s %9s %8s %8s %7s %16s %6s %9s %9s' % (
                'z_um', 'C_ret', 'C_mb', 'bracket', 'fid', 'pw/or', 'reason',
                'fb', 'nbr', 'dec'))
            for r in rows(p):
                if 'error' in r:
                    print('%9s  %s' % (fmt(r.get('z_um'), '%.1f'),
                                       r['error'][:70]))
                    continue
                print('%9s %10s %9s %8s %8s %7s %16s %6s %9s %9s' % (
                    fmt(r.get('z_um'), '%.2f'),
                    fmt(r.get('pixel_continuity'), '%.5g'),
                    fmt(r.get('multibranch_pixel_continuity'), '%.5g'),
                    fmt(r.get('multibranch_power_ratio_bracketed'), '%.4g'),
                    fmt(r.get('fidelity'), '%.4f'),
                    fmt(r.get('power_over_oracle'), '%.4g'),
                    str(r.get('reason'))[:16], str(r.get('fell_back')),
                    fmt(r.get('n_branch_max'), '%.0f'),
                    str(r.get('shipped_decision'))))


if __name__ == '__main__':
    main()
