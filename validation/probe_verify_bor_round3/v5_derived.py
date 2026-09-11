"""GAP 2, the DERIVED lossy-ceiling term -- is it TIGHT, or is it vacuous?

``_index_ceiling_slack`` widens the index-ceiling slack on an ABSORBING
half-space by ``_BOR_CHANNEL_IMAG_BAR**2 / (2 n)`` = 1.25e-09 / n, derived from
``Re(q^2) <= max(Re eps) k0^2`` (the numerical range of ``eps k0^2 + D`` with
``D`` real symmetric negative semi-definite) plus the channel gate's own
``|Im qn| < _BOR_CHANNEL_IMAG_BAR``.  The fix round records that no row of its
640-solve census comes within 5.1 decades of it, and lists "whether the derived
term is TIGHT" under what it could not measure.

THREE PARTS, in increasing strength:

``exact``      the CLOSED FORM for a UNIFORM passive half-space.  There
               ``q^2 = eps k0^2 - gamma^2`` with ``gamma`` REAL (the transverse
               operator does not see ``eps``), so the excess
               ``Re sqrt(eps - (gamma/k0)^2) - Re sqrt(eps)`` can be evaluated
               to machine precision over a dense grid of ``eps``, loss and
               ``gamma``.  If its maximum is 0 at ``gamma = 0`` and negative
               everywhere else, the derived term is not needed AT ALL on a
               uniform half-space, whatever the loss -- which is a statement
               about the WHOLE population the census samples.
``synthetic``  the DECISION boundary, driven through the shipped
               ``_check_nodal_passivity`` with a channel array placed by hand
               at ``n_max + delta``.  This is the only way to reach the window
               ``(5e-10, 5e-10 + 1.25e-09/n]`` at all, and it says whether the
               term is USED where the derivation puts it.
``hunt``       the PHYSICAL construction the fix round did not make: a
               nearly-lossless ABSORBING half-space whose returned channel is
               driven as close to its own ceiling as the discretisation
               allows, by bisecting on ``Rbig``.  Reports the closest approach
               from below and whether anything lands inside the window.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vb3  # noqa: E402

import lumenairy.elements.bor.bor_solve as bs  # noqa: E402
from lumenairy.elements.bor._orient import (  # noqa: E402
    _BOR_CHANNEL_IMAG_BAR,
)

K0 = _vb3.K0


def part_exact():
    """The closed form, in float128-free exact-as-numpy arithmetic."""
    worst = -np.inf
    worst_at = None
    rows = 0
    viol = []
    for e in (1.0, 1.5, 2.25, 4.0, 6.0, 12.0, 30.0):
        for a in (0.0, 1e-14, 1e-12, 3e-12, 1e-9, 1e-7, 1e-5, 1e-4, 1e-3,
                  1e-2, 1e-1, 0.5, 1.0):
            eps = complex(e, a * e)
            n_max = float(np.real(np.sqrt(eps)))
            # gamma/k0 from 0 up to the point where the mode is evanescent
            for g in np.concatenate((np.zeros(1),
                                     np.logspace(-16, np.log10(max(e, 1e-3)),
                                                 400))):
                qn = np.sqrt(eps - g ** 2)
                if np.real(qn) < 0:
                    qn = -qn
                if abs(np.imag(qn)) >= _BOR_CHANNEL_IMAG_BAR:
                    continue                     # the channel gate drops it
                if np.real(qn) <= 1e-6:
                    continue                     # the real floor drops it
                exc = float(np.real(qn)) - n_max
                rows += 1
                if exc > worst:
                    worst, worst_at = exc, (e, a, float(g))
                if exc > 0.0:
                    viol.append((e, a, float(g), exc))
    summary = dict(
        rows=rows,
        worst_excess_over_the_exact_uniform_spectrum=worst,
        worst_at_eps_a_gamma=worst_at,
        rows_with_a_positive_excess=len(viol),
        examples=viol[:10],
        derived_term_at_n_1=float(_BOR_CHANNEL_IMAG_BAR ** 2 / 2.0),
        base_slack=float(bs._BOR_INDEX_CEILING_SLACK))
    for k in sorted(summary):
        if k != "examples":
            print(" ", k, summary[k])
    _vb3.dump("v5_exact", dict(summary=summary))


def _layer(eps, im_rel, m=1, N=120, rbl=2.0, k0=K0):
    Rbig = float(rbl) * 2.0 * np.pi / float(k0)
    prof = _vb3._lossy(_vb3.uni(eps), im_rel, im_rel != 0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bs.build_layer(m, Rbig, N, prof, k0, basis="nodal")


def _decide(L, n_max, delta):
    """Feed ``_check_nodal_passivity`` ONE channel placed by hand at
    ``n_max + delta`` and report whether it refuses."""
    qn = np.array([n_max + delta], dtype=complex)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bs._check_nodal_passivity([L, L], np.array([1.0]),
                                      channels=((L, qn), (L, qn)))
        return False
    except bs.BORNodalPassivityError:
        return True


def part_synthetic():
    """WHERE the decision boundary actually sits, lossless vs absorbing."""
    rows = []
    for eps, im_rel, label in ((2.25, 0.0, "lossless"),
                              (2.25, 3e-12, "absorbing_3e-12"),
                              (2.25, 1e-6, "absorbing_1e-6"),
                              (1.0, 3e-12, "absorbing_n1"),
                              (36.0, 3e-12, "absorbing_n6")):
        L = _layer(eps, im_rel)
        n_max = float(np.real(np.sqrt(complex(L["eps_ceiling"]))))
        slack = bs._index_ceiling_slack(L, n_max)
        derived = (0.0 if bs._layer_is_lossless(L)
                   else _BOR_CHANNEL_IMAG_BAR ** 2 / (2.0 * n_max))
        # bisect the decision boundary in delta
        lo, hi = 1e-14, 1e-3
        assert not _decide(L, n_max, lo)
        assert _decide(L, n_max, hi)
        for _ in range(200):
            mid = np.sqrt(lo * hi)
            if _decide(L, n_max, mid):
                hi = mid
            else:
                lo = mid
            if hi / lo < 1.0 + 1e-12:
                break
        rows.append(dict(label=label, eps=eps, im_rel=im_rel, n_max=n_max,
                         lossless=bool(bs._layer_is_lossless(L)),
                         slack_reported=float(slack),
                         derived_term=float(derived),
                         base_slack=float(bs._BOR_INDEX_CEILING_SLACK),
                         boundary_lo=float(lo), boundary_hi=float(hi),
                         boundary_matches_slack=bool(
                             lo <= slack <= hi
                             or abs(hi / max(slack, 1e-300) - 1.0) < 1e-6),
                         window_lo=float(bs._BOR_INDEX_CEILING_SLACK),
                         window_hi=float(bs._BOR_INDEX_CEILING_SLACK
                                         + derived),
                         refused_at_window_lo_plus=bool(
                             _decide(L, n_max,
                                     bs._BOR_INDEX_CEILING_SLACK * 1.0001)),
                         refused_just_inside_window=bool(
                             _decide(L, n_max,
                                     bs._BOR_INDEX_CEILING_SLACK
                                     + 0.5 * derived)) if derived else None))
        print("  %-18s n=%.6g lossless=%-5s slack=%.6e boundary=[%.9e, "
              "%.9e] derived=%.6e" % (label, n_max, rows[-1]["lossless"],
                                      slack, lo, hi, derived), flush=True)
    summary = dict(
        rows=rows,
        boundary_is_the_slack_everywhere=all(r["boundary_matches_slack"]
                                             for r in rows),
        lossy_boundary_over_lossless=(
            rows[1]["boundary_hi"] / rows[0]["boundary_hi"]),
        window_width_is_the_derived_term=[r["derived_term"] for r in rows])
    for k in sorted(summary):
        if k != "rows":
            print(" ", k, summary[k])
    _vb3.dump("v5_synthetic", dict(summary=summary))


def part_hunt():
    """The PHYSICAL construction: drive a nearly-lossless ABSORBING
    half-space's worst returned channel as close to its own ceiling as the
    discretisation allows, by bisecting on ``Rbig``."""
    t0 = time.time()

    def exc_of(rbl, m, N, eps, im_rel):
        Rbig = float(rbl) * 2.0 * np.pi / K0
        prof = _vb3._lossy(_vb3.uni(eps), im_rel, im_rel != 0.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            L = bs.build_layer(m, Rbig, N, prof, K0, basis="nodal")
        keep = np.where(bs._physical_propagating(L, K0))[0]
        qn = np.asarray(L["q"])[keep] / K0
        e = bs._channel_index_excess(L, qn)
        n_max = float(np.real(np.sqrt(complex(L["eps_ceiling"]))))
        return (e, n_max, int(keep.size))

    rows = []
    best_below, best_row = -np.inf, None
    #: coarse scan first -- the excess is a discrete function of the spectrum,
    #: so a bracket has to be found before it can be bisected
    for m in (0, 1, 2, 3):
        for N in (80, 120, 200):
            for im_rel in (3e-12, 1e-9):
                prev = None
                for rbl in np.linspace(0.4, 6.0, 29):
                    e, n_max, nch = exc_of(rbl, m, N, 2.25, im_rel)
                    if e is None:
                        prev = None
                        continue
                    rows.append(dict(m=m, N=N, im_rel=im_rel, rbl=float(rbl),
                                     excess=e, n_max=n_max, n_channels=nch))
                    if e <= 0.0 and e > best_below:
                        best_below, best_row = e, rows[-1]
                    # a sign change between adjacent rbl is a bracket
                    if (prev is not None and prev[1] is not None
                            and (prev[1] <= 0.0) != (e <= 0.0)):
                        lo_r, hi_r = prev[0], float(rbl)
                        for _ in range(40):
                            mid = 0.5 * (lo_r + hi_r)
                            em, nm, _n = exc_of(mid, m, N, 2.25, im_rel)
                            if em is None:
                                break
                            if em <= 0.0:
                                lo_r = mid
                            else:
                                hi_r = mid
                            if em <= 0.0 and em > best_below:
                                best_below, best_row = em, dict(
                                    m=m, N=N, im_rel=im_rel, rbl=float(mid),
                                    excess=em, n_max=nm, n_channels=_n,
                                    via="bisection")
                            if hi_r - lo_r < 1e-15 * max(1.0, hi_r):
                                break
                    prev = (float(rbl), e)
    pos = [r["excess"] for r in rows if r["excess"] is not None
           and r["excess"] > 0.0]
    derived_at = (_BOR_CHANNEL_IMAG_BAR ** 2
                  / (2.0 * (best_row["n_max"] if best_row else 1.5)))
    window = (bs._BOR_INDEX_CEILING_SLACK,
              bs._BOR_INDEX_CEILING_SLACK + derived_at)
    inside = [r for r in rows if r["excess"] is not None
              and window[0] < r["excess"] <= window[1]]
    summary = dict(
        rows=len(rows),
        closest_approach_from_below=float(best_below),
        closest_row=best_row,
        decades_from_zero=(float(np.log10(abs(best_below)))
                           if best_below < 0 else None),
        mildest_positive=(min(pos) if pos else None),
        n_positive=len(pos),
        derived_term_at_that_n=float(derived_at),
        window=[float(window[0]), float(window[1])],
        rows_inside_the_window=len(inside),
        decades_from_the_window=(float(np.log10(window[1] / abs(best_below)))
                                 if best_below < 0 else None),
        seconds=time.time() - t0)
    for k in sorted(summary):
        if k != "closest_row":
            print(" ", k, summary[k])
    print("  closest row:", best_row)
    _vb3.dump("v5_hunt", dict(rows=rows, summary=summary))


PARTS = {"exact": part_exact, "synthetic": part_synthetic, "hunt": part_hunt}

if __name__ == "__main__":
    a = _vb3.arm()
    _vb3.require_tree(a)
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"], flush=True)
    for name in (sys.argv[1:] or list(PARTS)):
        print("== PART", name, flush=True)
        PARTS[name]()
