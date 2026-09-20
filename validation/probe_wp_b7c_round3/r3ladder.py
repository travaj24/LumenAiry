"""The plane ladders, DERIVED from this round's own geometry probe.

No z list in round 3 is copied from an earlier report.  ``geom_win.json``
carries, per optic, the paraxial and marginal foci and the z window in which
the meridional landing map has an interior turning point; the ladders below
are functions of those numbers alone, so re-running ``r3geom.py`` on a
changed prescription re-derives them.

Three ladders per optic:

``band``  the FOLD RING, sampled across the whole window the geometry probe
          found, plus a margin either side so the onset and the exit of the
          two-branch band are both crossed.  This is the population the bar
          is derived on.
``tail``  the FAR TAIL: the exit vertex, three planes well short of the fold,
          the paraxial focus itself and three beyond it, out to 2x the
          paraxial focal distance.  These are the planes that fall back, and
          they are what makes the "all planes" population different from the
          fold-ring one.
``fine``  built in a second pass by ``r3fine.py`` around whatever the band
          ladder found to be the largest RETURNED and the smallest REFUSED
          reading on that optic -- the two numbers the bar's margins are, so
          they are measured at a step 30x finer than the band ladder rather
          than inherited from it.
"""
from __future__ import annotations

import json
import os

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))


def geom(path=None):
    with open(path or os.path.join(_HERE, 'geom_win.json'),
              encoding='cp1252') as f:
        return json.load(f)


def band_zs(g, n=14):
    """The fold window, with 4 % of its own width of margin either side."""
    r = g.get('fold_z_range')
    if not r:
        return []
    lo, hi = float(r[0]), float(r[1])
    w = max(hi - lo, 1e-6 * max(hi, 1.0))
    return list(np.linspace(lo - 0.04 * w, hi + 0.04 * w, int(n)) * 1e6)


def tail_zs(g, n_pre=3, n_post=4):
    """Everything that is NOT the fold ring: the exit vertex, short of the
    fold, and past the paraxial focus."""
    fp = float(g['f_paraxial'])
    r = g.get('fold_z_range')
    lo = float(r[0]) if r else 0.55 * fp
    out = [0.0]
    out += list(np.linspace(0.18 * fp, 0.90 * lo, n_pre))
    out += [fp]
    out += list(np.linspace(1.06 * fp, 2.0 * fp, n_post))
    return [z * 1e6 for z in out]


def spec(zs):
    return ','.join('%.6f' % z for z in zs)
