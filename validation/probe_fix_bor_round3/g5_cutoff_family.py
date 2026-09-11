"""GAP 5 -- the near-cutoff ladder's closure over the RADIAL CUTOFF INDEX, and
what the worst rung actually is.

``tests/unit/test_fix_bor_multilayer_guards._gamma_of(m, idx=2)`` fixes the
radial cutoff index by an undocumented default that every caller takes, so
``_CUTOFF_LADDER_BAR = 1e-5`` is quoted over ``m`` and never over ``idx``.  The
round-2 verification measured ``1.787437e-04`` at ``(m=1, idx=1)`` -- 17.9x
above the bar.

THIS PROBE sweeps ``m`` in 0..3 x ``idx`` in 1..3 on the SHIPPED fixture, and
at the WORST rung of the worst combination takes the two censuses that decide
whether that number is the band's noise side or a residual defect:

  * the CHANNEL COUNT over the ladder (an integer; the invariant the
    orientation band exists to protect), and
  * the BACKWARD-FLUX census of every layer's spectrum at that rung -- how many
    modes sit INSIDE the classifier band, how many of those were FLIPPED by the
    sign of their flux, and how many have a flux too weak to normalize by
    (``_orient.flux_is_strong``), which is the population whose orientation is
    decided by noise and which the 5.45.1 band fix is about.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _g3  # noqa: E402

from lumenairy.elements.bor import _orient as orient  # noqa: E402
from lumenairy.elements.bor.bor_solve import build_layer  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402

RBIG, NFD, NREF = 24.0, 120, 1.41
EPS = NREF ** 2
MS = (0, 1, 2, 3)
IDXS = (1, 2, 3)


def _fd_modes(m, k0):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return build_layer(m, RBIG, NFD,
                           lambda r: np.full(np.shape(r), complex(EPS)), k0)


def gamma_of(m, idx):
    L = _fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    g = np.sort(g[g > 1e-6])
    return float(g[idx]) if idx < g.size else None


def rungs():
    floor = orient._BOR_CHANNEL_REAL_FLOOR * 10.0
    out = []
    for e_ in range(8, 40):
        dl = 10.0 ** (-e_ / 2.0)
        if NREF * np.sqrt(dl) < floor:
            break
        out.append(dl)
    return out


def cutoff_stack(m, k0):
    s = BORStack(RBIG, m, n_substrate=NREF, n_superstrate=NREF, N=NFD,
                 basis="fd")
    s.add_layer(0.4, eps=EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=EPS)
    s.set_source(k0=float(k0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s, s.solve()


def flux_census(m, k0):
    """The backward-flux census of every layer's spectrum, recorded from the
    ORIENTATION KERNEL ITSELF.

    ``zcascade.forward_orient`` is the one site that decides a mode's
    orientation, and it receives exactly ``(q, flux, k0)``.  Wrapping it for
    the duration of one solve reads the classifier's own inputs rather than a
    re-implementation of them.
    """
    from lumenairy.elements.bor import _orient as orient_mod
    from lumenairy.elements.bor import zcascade as zc
    seen = []
    real = zc.forward_orient

    def rec(q, flux, kk, **kw):
        q = np.asarray(q)
        flux = np.real(np.asarray(flux))
        scale = float(orient_mod.orient_band_scale(q, kk, xp=np))
        inband = np.abs(np.imag(q)) <= orient_mod._BOR_CUT_BAND_REL * scale
        fb = np.abs(flux[inband]) if np.any(inband) else np.array([])
        seen.append(dict(
            n_modes=int(q.size),
            band_scale=scale,
            n_in_band=int(np.count_nonzero(inband)),
            n_backward_in_band=int(np.count_nonzero(inband & (flux < 0.0))),
            n_backward_all=int(np.count_nonzero(flux < 0.0)),
            min_abs_flux_in_band=(float(np.min(fb)) if fb.size else None),
            max_abs_flux_in_band=(float(np.max(fb)) if fb.size else None),
            n_near_zero_flux_in_band=int(np.count_nonzero(
                fb < 1e-10 * (np.max(fb) if fb.size else 1.0)))
            if fb.size else 0,
        ))
        return real(q, flux, kk, **kw)

    zc.forward_orient = rec
    try:
        _st, res = cutoff_stack(m, k0)
    finally:
        zc.forward_orient = real
    return seen, res


def ladder(m, idx):
    g = gamma_of(m, idx)
    if g is None:
        return None
    counts, worst, worst_dl, worst_k0 = set(), 0.0, None, None
    per = []
    for dl in rungs():
        k0 = g / (NREF * np.sqrt(1.0 - dl))
        _st, res = cutoff_stack(m, k0)
        n = int(np.size(res["R"]))
        counts.add(n)
        en = np.asarray(res["energy"])
        c = float(np.max(np.abs(en - 1.0))) if en.size else 0.0
        per.append(dict(dl=dl, k0=k0, n_channels=n, closure=c))
        if c > worst:
            worst, worst_dl, worst_k0 = c, dl, k0
    return dict(m=m, idx=idx, gamma=g, counts=sorted(counts), worst=worst,
                worst_dl=worst_dl, worst_k0=worst_k0, rungs=per)


def main():
    a = _g3.arm()
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"])
    t0 = time.time()
    grid = []
    for m in MS:
        for idx in IDXS:
            r = ladder(m, idx)
            if r is not None:
                grid.append(r)
                print("  m=%d idx=%d counts=%s worst=%.6e @ dl=%.3g"
                      % (m, idx, r["counts"], r["worst"], r["worst_dl"]))
    env = max(g["worst"] for g in grid)
    argmax = [g for g in grid if g["worst"] == env][0]
    swept = [g for g in grid if g["idx"] == 2]
    env_swept = max(g["worst"] for g in swept)
    counts_move = [(g["m"], g["idx"], g["counts"]) for g in grid
                   if len(g["counts"]) != 1]
    # the two censuses at the WORST rung of the WORST combination
    fc, res = flux_census(argmax["m"], argmax["worst_k0"])
    en = np.asarray(res["energy"])
    summary = dict(
        n_combinations=len(grid),
        n_rungs=len(rungs()),
        envelope_all=env,
        envelope_swept_index=env_swept,
        argmax=dict(m=argmax["m"], idx=argmax["idx"], dl=argmax["worst_dl"],
                    k0=argmax["worst_k0"]),
        above_1e5=[(g["m"], g["idx"], g["worst"]) for g in grid
                   if g["worst"] > 1e-5],
        n_above_1e5=sum(1 for g in grid if g["worst"] > 1e-5),
        counts_that_move=counts_move,
        worst_rung_channels=int(np.size(res["R"])),
        worst_rung_counts=argmax["counts"],
        worst_rung_energy=[float(np.min(en)), float(np.max(en))] if en.size
        else None,
        worst_rung_flux_census=fc,
        seconds=time.time() - t0,
    )
    for k in sorted(summary):
        print(" ", k, summary[k])
    _g3.dump("g5_cutoff_family", dict(grid=grid, summary=summary), a)


if __name__ == "__main__":
    main()
