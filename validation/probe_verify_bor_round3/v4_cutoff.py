"""GAP 5, INDEPENDENT -- the near-cutoff closure residual, re-derived on this
verification's own fixture.

THE FIXTURE IS NOT THE GATE'S.  ``test_fix_bor_multilayer_guards`` measures
``Rbig`` = 24, ``N`` = 120, ``n`` = 1.41, a ``(3.0, 0.5, 2.45, 1.41)`` ring
layer between two 0.4-thick slabs.  This probe measures ``Rbig`` = 19,
``N`` = 96, ``n`` = 1.63, a ``(2.7, 0.45, 2.10, 1.63)`` ring layer between two
0.35-thick slabs.  Same MECHANISM, different arithmetic, so a knee that
reproduces is corroboration.

Parts:

``family``     ``m`` 0..3 x ``idx`` 1..3 x the floor-derived ladder: the
               channel count, the closure, and the split of the closure into
               the two populations the round ships (``qn_marginal`` at or above
               100x ``_BOR_CHANNEL_REAL_FLOOR``, and below it).  Also scans the
               knee itself rather than assuming 100.
``mechanism``  at the worst rung: is the residual on the MARGINAL CHANNEL's own
               ``R + T`` row, or is it an orientation error?  Census of
               backward-flux modes INSIDE the classifier band, per half-space,
               and of the in-band ``|flux|`` spread.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

from lumenairy.elements.bor import _orient as _or  # noqa: E402
from lumenairy.elements.bor.bor_stack import BORStack  # noqa: E402

RBIG = 19.0
NFD = 96
NREF = 1.63
EPS = NREF ** 2
RINGS = (2.7, 0.45, 2.10, 1.63)
T_SLAB, T_RING = 0.35, 0.6

#: the round's two shipped scope constants, read from the test module so this
#: probe measures what SHIPS rather than a copy of it.
FLOOR = float(_or._BOR_CHANNEL_REAL_FLOOR)


def _fd_modes(m, k0, eps=EPS, N=NFD):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, RBIG, N,
                       lambda r: np.full_like(r, eps, dtype=complex),
                       float(k0), staggered=True)


def gamma_of(m, idx):
    """The cutoff wavenumber of the ``idx``-th radial order at azimuthal order
    ``m``, over the orders with ``gamma > 0``, counted from the axis out."""
    L = _fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    g = np.sort(g[g > 1e-6])
    return float(g[idx]) if g.size > idx else None


def cutoff_stack(m, k0):
    s = BORStack(RBIG, m, n_substrate=NREF, n_superstrate=NREF, N=NFD,
                 basis="fd")
    s.add_layer(T_SLAB, eps=EPS)
    s.add_layer(T_RING, rings=RINGS)
    s.add_layer(T_SLAB, eps=EPS)
    s.set_source(k0=float(k0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return s.solve()


def rungs(mult=10.0):
    """The ladder, bounded below by the R/T channel gate's own floor."""
    out = []
    for e_ in range(8, 40):
        dl = 10.0 ** (-e_ / 2.0)
        if NREF * np.sqrt(dl) < FLOOR * mult:
            break
        out.append(dl)
    return out


def part_family():
    t0 = time.time()
    rows = []
    ladder = rungs()
    print("  ladder: %d rungs, qn from %.4g down to %.4g"
          % (len(ladder), NREF * np.sqrt(ladder[0]),
             NREF * np.sqrt(ladder[-1])), flush=True)
    for m in (0, 1, 2, 3):
        for idx in (1, 2, 3):
            g = gamma_of(m, idx)
            if g is None:
                print("  m=%d idx=%d: no cutoff at that index" % (m, idx))
                continue
            counts = set()
            for dl in ladder:
                k0 = g / (NREF * np.sqrt(1.0 - dl))
                res = cutoff_stack(m, k0)
                en = np.asarray(res["energy"], float)
                qn_marg = NREF * np.sqrt(dl)
                closure = (float(np.max(np.abs(en - 1.0))) if en.size
                           else None)
                counts.add(int(np.size(res["R"])))
                rows.append(dict(m=m, idx=idx, delta=dl, k0=float(k0),
                                 qn_marginal=float(qn_marg),
                                 qn_over_floor=float(qn_marg / FLOOR),
                                 n_channels=int(np.size(res["R"])),
                                 closure=closure))
            print("  m=%d idx=%d counts=%s worst=%.6e"
                  % (m, idx, sorted(counts),
                     max(r["closure"] for r in rows
                         if r["m"] == m and r["idx"] == idx
                         and r["closure"] is not None)), flush=True)
    #: the two populations at the SHIPPED knee, and the knee scanned
    def split(mult):
        hi = [r for r in rows if r["closure"] is not None
              and r["qn_over_floor"] >= mult]
        lo = [r for r in rows if r["closure"] is not None
              and r["qn_over_floor"] < mult]
        return (max([r["closure"] for r in hi] or [0.0]),
                max([r["closure"] for r in lo] or [0.0]), len(hi), len(lo))

    shallow, deep, nhi, nlo = split(100.0)
    knee_scan = {}
    for mult in (1.0, 3.0, 10.0, 20.0, 32.0, 50.0, 79.0, 100.0, 150.0, 300.0,
                 1000.0):
        s, d, a, b = split(mult)
        knee_scan["%g" % mult] = dict(shallow_envelope=s, deep_envelope=d,
                                      n_shallow=a, n_deep=b)
    per_ladder = {}
    for m in (0, 1, 2, 3):
        for idx in (1, 2, 3):
            sub = [r for r in rows if r["m"] == m and r["idx"] == idx]
            if not sub:
                continue
            cs = {r["n_channels"] for r in sub}
            per_ladder["m%d_idx%d" % (m, idx)] = dict(
                counts=sorted(cs), count_is_one_number=(len(cs) == 1),
                count_is_idx_plus_one=(cs == {idx + 1}),
                worst=max(r["closure"] for r in sub
                          if r["closure"] is not None),
                worst_shallow=max([r["closure"] for r in sub
                                   if r["closure"] is not None
                                   and r["qn_over_floor"] >= 100.0] or [0.0]),
                worst_deep=max([r["closure"] for r in sub
                                if r["closure"] is not None
                                and r["qn_over_floor"] < 100.0] or [0.0]))
    #: the first rung anywhere in the grid over 1e-6, by qn/floor
    over = sorted((r["qn_over_floor"], r["closure"], r["m"], r["idx"])
                  for r in rows
                  if r["closure"] is not None and r["closure"] > 1e-6)
    summary = dict(
        rows=len(rows), ladders=len(per_ladder), rungs=len(ladder),
        shallow_envelope=shallow, deep_envelope=deep,
        n_shallow=nhi, n_deep=nlo,
        decades_between_populations=(np.log10(deep / shallow)
                                     if shallow > 0 else None),
        highest_qn_over_floor_exceeding_1e_6=(over[-1] if over else None),
        lowest_qn_over_floor_exceeding_1e_6=(over[0] if over else None),
        knee_scan=knee_scan, per_ladder=per_ladder,
        count_one_number_everywhere=all(v["count_is_one_number"]
                                        for v in per_ladder.values()),
        count_idx_plus_one_everywhere=all(v["count_is_idx_plus_one"]
                                          for v in per_ladder.values()),
        bar_margin_shallow=(1.0e-5 / shallow if shallow > 0 else None),
        bar_margin_deep=(2.0e-3 / deep if deep > 0 else None),
        seconds=time.time() - t0)
    for k in sorted(summary):
        if k not in ("knee_scan", "per_ladder"):
            print(" ", k, summary[k])
    _vb3.dump("v4_family", dict(rows=rows, summary=summary))


def part_mechanism():
    """At the worst rung of each ladder: WHERE does the closure defect sit?

    Three measurements, all read off the same solve:

    1. the per-channel ``R + T`` row, and whether the worst row is the MARGINAL
       channel (the one the ladder drives to cutoff -- the smallest ``Re qn``);
    2. how many modes INSIDE the classifier band carry BACKWARD flux in either
       half-space -- the 5.45.1 defect's own signature;
    3. the in-band ``|flux|`` spread, which is the flux-normalisation mechanism
       the round names.
    """
    t0 = time.time()
    out = []
    ladder = rungs()
    deep = [dl for dl in ladder if NREF * np.sqrt(dl) < 100.0 * FLOOR]
    for m in (0, 1, 2, 3):
        for idx in (1, 2, 3):
            g = gamma_of(m, idx)
            if g is None:
                continue
            for dl in deep[-2:]:
                k0 = g / (NREF * np.sqrt(1.0 - dl))
                res = cutoff_stack(m, k0)
                en = np.asarray(res["energy"], float)
                if not en.size:
                    continue
                qi = np.asarray(res.get("q", []), complex) / k0
                jworst = int(np.argmax(np.abs(en - 1.0)))
                jmarg = (int(np.argmin(np.real(qi))) if qi.size else -1)
                band = _band_census(m, k0)
                out.append(dict(
                    m=m, idx=idx, delta=dl,
                    qn_marginal=float(NREF * np.sqrt(dl)),
                    closure=float(np.max(np.abs(en - 1.0))),
                    n_channels=int(en.size),
                    worst_row=jworst, marginal_row=jmarg,
                    worst_is_marginal=bool(jworst == jmarg),
                    energy_rows=[float(x) for x in en],
                    qn_real=[float(x) for x in np.real(qi)],
                    **band))
                print("  m=%d idx=%d dl=%.3g closure=%.4e worst_row=%d "
                      "marginal_row=%d same=%s in_band_backward=%s"
                      % (m, idx, dl, out[-1]["closure"], jworst, jmarg,
                         out[-1]["worst_is_marginal"],
                         out[-1]["in_band_backward"]), flush=True)
    n = len(out)
    same = sum(r["worst_is_marginal"] for r in out)
    bw = sum(r["in_band_backward"] for r in out)
    summary = dict(
        rows=n,
        worst_row_is_the_marginal_channel=same,
        worst_row_is_the_marginal_channel_frac=(same / n if n else None),
        rows_with_any_in_band_backward_mode=bw,
        total_in_band_backward_modes=sum(r["in_band_backward"] for r in out),
        total_backward_modes=sum(r["backward_total"] for r in out),
        weakest_in_band_flux=min([r["in_band_flux_min"] for r in out
                                  if r["in_band_flux_min"] is not None]
                                 or [None]),
        strongest_in_band_flux=max([r["in_band_flux_max"] for r in out
                                    if r["in_band_flux_max"] is not None]
                                   or [None]),
        seconds=time.time() - t0)
    for k in sorted(summary):
        print(" ", k, summary[k])
    _vb3.dump("v4_mechanism", dict(rows=out, summary=summary))


def _band_census(m, k0):
    """Backward-flux modes INSIDE the classifier band, in the half-space the
    cutoff ladder drives, and the in-band ``|flux|`` spread."""
    L = _fd_modes(m, k0)
    q = np.asarray(L["q"])
    W, V = L["W"], L["V"]
    wq_f = np.real(np.asarray(L["wq_face"]))
    wq_n = np.real(np.asarray(L["wq_node"]))
    N = len(wq_f)
    flux = np.real(np.sum(W[:N] * np.conj(V[N:]) * wq_f[:, None], axis=0)
                   - np.sum(W[N:] * np.conj(V[:N]) * wq_n[:, None], axis=0))
    scale = _or.orient_band_scale(q, k0, xp=np)
    inband = np.abs(np.imag(q)) <= _or._BOR_CUT_BAND_REL * scale
    qo = _or.forward_orient(q, flux, k0, xp=np)
    fo = np.where(qo == q, flux, -flux)
    back = fo < 0.0
    ib = np.abs(fo[inband])
    return dict(n_modes=int(q.size), n_in_band=int(np.count_nonzero(inband)),
                in_band_backward=int(np.count_nonzero(back & inband)),
                backward_total=int(np.count_nonzero(back)),
                in_band_flux_min=(float(np.min(ib)) if ib.size else None),
                in_band_flux_max=(float(np.max(ib)) if ib.size else None))


PARTS = {"family": part_family, "mechanism": part_mechanism}

if __name__ == "__main__":
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    for name in (sys.argv[1:] or list(PARTS)):
        print("== PART", name, flush=True)
        PARTS[name]()
