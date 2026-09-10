"""V13 -- the two-row-halo SIGN SCAN, exhaustively.

The one piece of the caustic census that is NOT pointwise is the
adjacent-pixel ``det J`` sign scan: the whole-grid closure tests every
horizontal and vertical neighbour pair over the full grid; the band loop
tests them per band and carries ONE row of halo (``_prev_sd`` /
``_prev_fin``) across the band boundary.

This re-implements both algorithms EXACTLY as the library writes them and
runs them against each other on randomised sign / finite-mask fields that
are engineered to put sign changes ON band boundaries -- the only place the
band decomposition can lose one.  A single missed sign change is a
silent-wrong (the fold-caustic warning would not fire).

Usage:  python v13_sign_scan.py <out.json>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def whole_grid(det_j, fin):
    """The closure's scan (_lens_traced.py, _ray_density_amp_grid)."""
    sd = np.sign(det_j)
    mh = fin[:, 1:] & fin[:, :-1]
    mv = fin[1:, :] & fin[:-1, :]
    return (bool(np.any((sd[:, 1:] * sd[:, :-1] < 0.0) & mh))
            or bool(np.any((sd[1:, :] * sd[:-1, :] < 0.0) & mv)))


def banded(det_j, fin, cr):
    """The v5.44 band loop's scan, transcribed."""
    N = det_j.shape[0]
    sign_change = False
    prev_sd = None
    prev_fin = None
    for r0 in range(0, N, cr):
        r1 = min(N, r0 + cr)
        sd_b = np.sign(det_j[r0:r1])
        fin_b = fin[r0:r1]
        mh = fin_b[:, 1:] & fin_b[:, :-1]
        if bool(np.any((sd_b[:, 1:] * sd_b[:, :-1] < 0.0) & mh)):
            sign_change = True
        if r1 - r0 > 1:
            mv = fin_b[1:, :] & fin_b[:-1, :]
            if bool(np.any((sd_b[1:, :] * sd_b[:-1, :] < 0.0) & mv)):
                sign_change = True
        if prev_sd is not None and bool(np.any(
                (prev_sd * sd_b[0] < 0.0) & (prev_fin & fin_b[0]))):
            sign_change = True
        prev_sd = sd_b[-1].copy()
        prev_fin = fin_b[-1].copy()
    return sign_change


def main():
    rng = np.random.default_rng(11)
    N = 64
    rows = [1, 2, 3, 5, 7, 8, 16, 31, 32, 63, 64, 65]
    cases = []
    # (a) fully random
    for _ in range(300):
        det = rng.standard_normal((N, N))
        fin = rng.random((N, N)) > 0.15
        cases.append((det, fin))
    # (b) engineered: ONE sign flip, at a chosen row boundary, everything
    #     else positive -- the exact state a lost halo row would miss
    for r in range(1, N):
        det = np.ones((N, N))
        det[r:, :] = -1.0
        fin = np.ones((N, N), dtype=bool)
        cases.append((det, fin))
    # (c) engineered: ONE sign flip at a row boundary with the pair's
    #     finite mask on/off, and a single flipped COLUMN
    for r in range(1, N):
        det = np.ones((N, N))
        det[r, 7] = -1.0
        fin = np.ones((N, N), dtype=bool)
        fin[r - 1, 7] = False            # kills the vertical pair above
        fin[r + 1 if r + 1 < N else r, 7] = False
        fin[r, 6] = fin[r, 8] = False    # kills both horizontal pairs
        cases.append((det, fin))
    # (d) sparse finite mask: only two adjacent finite pixels, straddling
    #     every possible band boundary
    for r in range(1, N):
        det = np.zeros((N, N))
        det[r - 1, 3] = 1.0
        det[r, 3] = -1.0
        fin = np.zeros((N, N), dtype=bool)
        fin[r - 1, 3] = fin[r, 3] = True
        cases.append((det, fin))
    bad = []
    n = 0
    for i, (det, fin) in enumerate(cases):
        w = whole_grid(det, fin)
        for cr in rows:
            b = banded(det, fin, cr)
            n += 1
            if b != w:
                bad.append({'case': i, 'rows': cr, 'whole': w, 'band': b})
    out = {'N': N, 'n_cases': len(cases), 'n_comparisons': n,
           'band_heights': rows, 'mismatches': bad,
           'agree': len(bad) == 0}
    print('sign scan: %d cases x %d band heights = %d comparisons, '
          'mismatches = %d' % (len(cases), len(rows), n, len(bad)), flush=True)
    for b in bad[:20]:
        print('   MISMATCH', b, flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
