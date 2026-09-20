"""VERIFY-WP-B7c round 3 -- the z LADDER and its refinements (claim 1).

WP-B7c round 3's central structural claim is that the bar's margin is a
reading of the z ladder and not a property of the quantity: refining the same
optics' ladders keeps finding readings closer to 1.06, so the "gap" the bar
sits in shrinks without a floor.  This probe reproduces that experiment on MY
optics, from MY geometry.

The three ladders are DERIVED, never copied:

* **L1** -- a RECONNAISSANCE pass (readings only) over the GEOMETRIC fold
  window (the z range in which the meridional landing map has an interior
  turning point, measured by :mod:`v3geom` with no reading of the library in
  it) locates the z band in which this build actually takes the completion
  route; ``L1`` is then 34 planes over that band widened by 8 % of its width
  on each side, plus a 12-plane TAIL over the rest of the geometric window.
  Every L1 plane is oracle-scored.
* **L2** -- every ADJACENT PAIR of L1 planes that STRADDLES the shipped bar
  (one returned, one refused) is re-scanned with 21 planes between them, i.e.
  ~20x the step.
* **L3** -- the same rule applied to L2's own straddling pairs, another ~10x.

The "gap" reported per ladder is exactly the quantity round 2 published:
``min(refused reading) / max(returned reading)`` over the planes of that
ladder, and the two-sided margins are ``bar / max(returned)`` and
``min(refused) / bar``.

Usage:
    python v3ladder.py <optic> <outprefix> [--oracle asm|none] [--refine N]
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import v3fixtures as FX
import v3geom as G
import v3scan as S


def recon_band(fx, n=40):
    """The L1 band, derived from the GEOMETRY and then confirmed.

    The seed is the ``z`` window in which the meridional LANDING map has an
    interior turning point -- a ray-optics statement about the prescription,
    measured by :mod:`v3geom`, with no reading of the library in it.  A
    reconnaissance pass (readings only) then reports where on that window
    THIS build actually takes the completion route, which is printed so the
    band can be read back, and the band is widened by 12 % of its own width
    on each side so L1 straddles both edges of the route.
    """
    presc, wl = fx['prescription'], fx['wavelength']
    _na, fp, fm, _hs, _fl = G.na_and_foci(presc, wl)
    zs_geom = np.linspace(0.55 * min(fp, fm), 1.45 * max(fp, fm), 120)
    fw = G.fold_window(presc, wl, zs_geom)
    zt = [z for z, rc, _ in fw if rc]
    if zt:
        lo, hi = min(zt), max(zt)
    else:
        lo, hi = 0.70 * min(fp, fm), 1.12 * max(fp, fm)
    w = hi - lo
    lo, hi = lo - 0.12 * w, hi + 0.12 * w
    rows = S.scan(fx, np.linspace(lo, hi, n), oracle='none')
    zf = [r['z_um'] * 1e-6 for r in rows if r.get('route') == 'fold_ring']
    print(f'--- geometric fold window {lo * 1e6:.1f} .. {hi * 1e6:.1f} um; '
          f'completion route on {len(zf)} of {n} recon planes'
          + ('' if not zf else
             f' ({min(zf) * 1e6:.1f} .. {max(zf) * 1e6:.1f} um)'), flush=True)
    if zf:
        a, b = min(zf), max(zf)
        w = max(b - a, 0.01 * b)
        band = (a - 0.08 * w, b + 0.08 * w)
    else:
        band = (lo, hi)
    return band, (lo, hi), rows


def straddling_pairs(rows, bar):
    """Adjacent (z, z) pairs on which the shipped decision flips."""
    good = [r for r in rows if r.get('pixel_continuity') is not None]
    good.sort(key=lambda r: r['z_um'])
    out = []
    for a, b in zip(good[:-1], good[1:]):
        ra = a['pixel_continuity'] > bar
        rb = b['pixel_continuity'] > bar
        if ra != rb:
            out.append((a['z_um'], b['z_um']))
    return out


def gap(rows, bar):
    """``min(refused) / max(returned)`` and the two one-sided margins."""
    cs = [r['pixel_continuity'] for r in rows
          if r.get('pixel_continuity') is not None]
    ret = [c for c in cs if c <= bar]
    ref = [c for c in cs if c > bar]
    if not ret or not ref:
        return dict(n=len(cs), n_returned=len(ret), n_refused=len(ref),
                    gap=None, margin_above=None, margin_below=None)
    return dict(n=len(cs), n_returned=len(ret), n_refused=len(ref),
                gap=min(ref) / max(ret), margin_above=bar / max(ret),
                margin_below=min(ref) / bar,
                max_returned=max(ret), min_refused=min(ref))


def fold_rows(rows):
    return [r for r in rows if r.get('route') == 'fold_ring']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('fixture')
    ap.add_argument('prefix')
    ap.add_argument('--oracle', default='asm')
    ap.add_argument('--refine', type=int, default=3)
    ap.add_argument('--n1', type=int, default=34)
    ap.add_argument('--ntail', type=int, default=12)
    ap.add_argument('--n2', type=int, default=21)
    ap.add_argument('--n3', type=int, default=11)
    a = ap.parse_args()
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    fx = FX.FIXTURES[a.fixture]
    bar = S.bars()['cmax']
    print(f'--- {fx["name"]}: reconnaissance', flush=True)
    (blo, bhi), (glo, ghi), recon = recon_band(fx)
    print(f'--- completion band {blo * 1e6:.1f} .. {bhi * 1e6:.1f} um',
          flush=True)

    # L1 = the completion BAND plus a TAIL over the rest of the geometric
    # window, which is the shape round 3's own coarse ladder has.
    z_band = np.linspace(blo, bhi, a.n1)
    z_tail = np.concatenate([
        np.linspace(glo, blo, a.ntail // 2 + 1)[:-1],
        np.linspace(bhi, ghi, a.ntail - a.ntail // 2 + 1)[1:]])
    z1 = np.sort(np.concatenate([z_band, z_tail]))
    print(f'--- L1: {z1.size} planes ({a.n1} in band, step '
          f'{(z_band[1] - z_band[0]) * 1e6:.3f} um; {z_tail.size} tail)',
          flush=True)
    L1 = S.scan(fx, z1, oracle=a.oracle, refine=a.refine)

    L2 = []
    for lo, hi in straddling_pairs(fold_rows(L1), bar):
        zs = np.linspace(lo * 1e-6, hi * 1e-6, a.n2 + 2)[1:-1]
        print(f'--- L2 bracket {lo:.3f}..{hi:.3f} um, step '
              f'{(zs[1] - zs[0]) * 1e6:.4f} um', flush=True)
        L2.extend(S.scan(fx, zs, oracle='none'))
    L3 = []
    for lo, hi in straddling_pairs(fold_rows(L1 + L2), bar):
        zs = np.linspace(lo * 1e-6, hi * 1e-6, a.n3 + 2)[1:-1]
        print(f'--- L3 bracket {lo:.4f}..{hi:.4f} um, step '
              f'{(zs[1] - zs[0]) * 1e6:.5f} um', flush=True)
        L3.extend(S.scan(fx, zs, oracle='none'))

    rep = dict(lumenairy=lumenairy.__file__, python=sys.version.split()[0],
               numpy=np.__version__, fixture=fx['name'], note=fx['note'],
               bar=bar, band_um=[blo * 1e6, bhi * 1e6],
               geom_window_um=[glo * 1e6, ghi * 1e6],
               oracle=a.oracle,
               refine=a.refine, recon=recon, L1=L1, L2=L2, L3=L3)
    for tag, rows in (('L1', L1), ('L1+L2', L1 + L2), ('L1+L2+L3',
                                                       L1 + L2 + L3)):
        rep[f'gap_fold_{tag}'] = gap(fold_rows(rows), bar)
        rep[f'gap_all_{tag}'] = gap(rows, bar)
        print(tag, 'fold', json.dumps(rep[f'gap_fold_{tag}']), flush=True)
    out = f'{a.prefix}.json'
    with open(out, 'w', encoding='cp1252') as f:
        json.dump(rep, f, indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
