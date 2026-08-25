import json, os, numpy as np
recs = json.load(open('fits/adjudication.json'))
cache = {}
print(f"{'tag':22s} {'M':>4s} {'rhs':>3s} {'rcond':>10s} {'r*/|b|':>10s} "
      f"{'excess_ne':>11s} {'excess_qr':>11s} {'excess_ship':>11s} "
      f"{'err_ne':>9s} {'err_ship':>9s}")
for r in recs:
    tag = r['tag']
    if tag not in cache:
        d = np.load(os.path.join('fits', tag + '.npz'))
        cache[tag] = np.asarray(d['b'], dtype=float)
    b = cache[tag]
    bc = b[:, r['rhs']] if b.ndim == 2 else b
    nb = float(np.linalg.norm(bc))
    r['r_over_b'] = r['r_star'] / nb if nb else float('nan')
    print(f"{tag:22s} {r['M']:4d} {r['rhs']:3d} {r['rcond']:10.3e} "
          f"{r['r_over_b']:10.2e} {r['excess_ne']:+11.3e} "
          f"{r['excess_qr']:+11.3e} {r['excess_ship']:+11.3e} "
          f"{r['err_ne']:9.2e} {r['err_ship']:9.2e}")
json.dump(recs, open('fits/adjudication.json','w'), indent=1)
