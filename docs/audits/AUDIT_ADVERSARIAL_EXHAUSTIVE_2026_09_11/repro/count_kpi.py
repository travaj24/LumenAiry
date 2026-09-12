"""Count severity cells and orchestrator-verified marks across the curated consolidated sections."""
import os, re, glob
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
sev = {'P0': 0, 'P1': 0, 'P2': 0, 'P3': 0}
ver = 0
rows = []
for p in sorted(glob.glob(os.path.join(R, 'CONSOLIDATED_PART_*.md'))):
    for line in open(p, encoding='utf-8'):
        if not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 3:
            continue
        m = re.match(r'^\**\s*(P[0-3])\b', cells[1])
        if m and re.match(r'^[A-Z]{1,2}\d+$', cells[0]):
            sev[m.group(1)] += 1
            if '✔' in cells[1]:
                ver += 1
            rows.append((os.path.basename(p), cells[0], m.group(1), '✔' in cells[1]))
for r in rows:
    print(r)
print('KPI: P0=%d P1=%d P2=%d P3=%d verified=%d' % (sev['P0'], sev['P1'], sev['P2'], sev['P3'], ver))
print('rows', len(rows))
