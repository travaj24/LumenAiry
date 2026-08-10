# Rebuild the adjudication CSV from the per-shard JSON dumps.
#
# The JSONs are the source of truth (adjudicate_nfc_8192.py writes one per
# shard, and it writes it AFTER the rows are built).  The CSV is the resumable
# human-readable log.  This exists because the first CSV writer joined on ','
# by hand while two columns ('order' = '(-4,-2)', 'keep' = '-4,-2;-3,-2')
# carry commas -- which shifts every column to their right without failing.
#
# Usage: python adj_rebuild_csv.py <json_dir> <out_csv>
import csv
import glob
import json
import os
import sys

d = sys.argv[1]
out = sys.argv[2]
rows = []
for p in sorted(glob.glob(os.path.join(d, '*.json'))):
    try:
        j = json.load(open(p, encoding='cp1252', errors='replace'))
    except Exception as exc:                      # noqa: BLE001
        print('SKIP %s (%s)' % (os.path.basename(p), exc))
        continue
    for r in j.get('rows', []):
        r = dict(r)
        r['_src'] = os.path.basename(p)
        rows.append(r)
if not rows:
    print('no rows found in %s' % d)
    raise SystemExit(1)
# UNION of keys, not rows[0]'s: the harness gained columns mid-campaign
# (the exact leg's measured-NA diagnostics), so early shards are short.
cols = []
for r in rows:
    for k in r:
        if k not in cols:
            cols.append(k)
rows.sort(key=lambda r: (r['arm'], int(r['my']), int(r['mx'])))
with open(out, 'w', encoding='cp1252', errors='replace', newline='') as fh:
    w = csv.writer(fh, quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    w.writerow(cols)
    for r in rows:
        w.writerow(['' if r.get(c) is None else r.get(c, '') for c in cols])
print('wrote %d rows (%d cols) to %s' % (len(rows), len(cols), out))
na = sorted({(r['arm'], r['order']) for r in rows})
print('arms/orders: %d unique' % len(na))
