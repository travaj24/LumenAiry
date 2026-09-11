"""V3 -- the two-sided band, censused on fixtures built here, and the
misclassification hunt.

THE QUANTITY.  ``_sqrt_decay`` acts on exactly one population: the modes whose
PRINCIPAL root ``r = sqrt(lam^2)`` has ``Im(r) < 0``.  For those it asks whether
``|Re(r)| / max(max|r|, 1) <= _CUT_BAND_REL`` and, if so, replaces ``r`` by
``conj(r)``.  So the band is sound iff that ratio separates

  NOISE  -- a LOSSLESS layer's PROPAGATING mode, whose ``lam^2`` is exactly real
            negative and whose tiny real part is the eigensolver's backward
            error (the sign of ``Im(lam^2)`` carries no physics), and
  PHYSICS -- a mode whose negative imaginary part is a real decay rate or a
            real propagation direction.

THE HUNT (both directions, per the brief):
  * a genuinely LOSSY or EVANESCENT mode whose ratio falls UNDER 1e-8 -- the
    band would then conjugate a physical root;
  * a LOSSLESS PROPAGATING mode whose ratio EXCEEDS 1e-8 -- the band would then
    miss it and leave the build-dependent root in place.

Sixty-one fixtures: lossless anisotropic and scalar 2-D at several truncations
and twists; 1-D TE/TM; oblique and conical; a LOSS LADDER from Im(eps) = 1e-2
down to 1e-14 (six decades deeper than the fix's own ladder); metals with large
negative real permittivity; near-Wood / near-cutoff mounts where ``lam^2 ~ 0``;
high-contrast gratings carrying many orders; and deliberately near-DEGENERATE
mounts, which are the only structural route to a large backward error on a real
eigenvalue (a defective pair perturbs as ``sqrt(eps_mach * ||M||)``, i.e. ~1e-8,
not ``eps_mach * ||M||``).

Also recorded: the ``lam^2 -> 0`` (exact Wood) handling and the ``|X| <= 1``
contraction guarantee over every fixture's returned roots.

Usage: python v3_band.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

BAND = 1e-8
P = V._P
WL = V._WL
DEPTH = V._DEPTH


def _j2(cell, n_sub, n_sup=1.0, theta=0.0, phi=0.0, nord=5, sym="auto"):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    return rcwa_jones_2d(P, P, cell, n_sub, n_sup, DEPTH, WL, theta=theta,
                         phi=phi, n_orders_x=nord, n_orders_y=nord, symmetry=sym)


def _e2(cell, n_sub, n_sup=1.0, theta=0.0, nord=5, pol="te"):
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    return rcwa_efficiency_2d(P, P, cell, n_sub, n_sup, DEPTH, WL,
                              theta=theta, polarization=pol,
                              n_orders_x=nord, n_orders_y=nord)


def _e1(nr, ng, ns, duty=0.5, pol="te", nord=15, theta=0.0, nsup=1.0,
        period=P, wl=WL, depth=DEPTH):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    return rcwa_efficiency_1d(period, nr, ng, ns, nsup, depth, duty, wl,
                              theta=theta, polarization=pol, n_orders=nord)


def _j1(er, eg, ns, duty=0.5, nord=15, theta=0.0, nsup=1.0):
    from lumenairy.elements.rcwa import rcwa_jones_1d
    return rcwa_jones_1d(P, er, eg, ns, nsup, DEPTH, duty, WL, theta=theta,
                         n_orders=nord)


def _layer_lam2_min(period):
    """Smallest |lam^2| of the TE layer eigenproblem at this period."""
    eig = V.EigSpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with eig:
            _e1(2.1, 1.5, 1.5, duty=0.5, pol="te", nord=11, period=period)
    return min(float(np.min(np.abs(w))) for w in eig.seen)


def _find_layer_cutoff_period():
    """Golden-section-free coarse+fine scan for the period at which a LAYER
    mode sits at cutoff (``lam^2 -> 0``).  Returns ``(period, |lam^2|)``."""
    lo, hi = 0.20e-6, 1.60e-6
    grid = np.linspace(lo, hi, 281)
    vals = [_layer_lam2_min(p) for p in grid]
    k = int(np.argmin(vals))
    a0 = grid[max(k - 1, 0)]
    b0 = grid[min(k + 1, len(grid) - 1)]
    for _ in range(60):                       # bisect on the minimiser
        m1 = a0 + (b0 - a0) / 3.0
        m2 = b0 - (b0 - a0) / 3.0
        if _layer_lam2_min(m1) < _layer_lam2_min(m2):
            b0 = m2
        else:
            a0 = m1
    p = 0.5 * (a0 + b0)
    return float(p), _layer_lam2_min(p)


# ------------------------------------------------------------------- fixtures
def fixtures():
    """(name, class, callable).  ``class`` says what the layer MATERIALS are,
    which is what decides whether a small ratio is noise or physics."""
    f = []
    a = f.append

    # -- lossless anisotropic 2-D: twist x truncation x substrate ------------
    for tw in (0.0, 0.3, 0.7, 1.1):
        for nord in (3, 4, 5):
            for ns in (1.5, 1.63):
                a(("aniso_tw%.1f_no%d_ns%.2f" % (tw, nord, ns), "lossless",
                   (lambda tw=tw, nord=nord, ns=ns:
                    _j2(V.uniaxial_cell(twist=tw), ns, nord=nord))))
    # -- lossless anisotropic, oblique and conical ---------------------------
    for th, ph in ((0.3, 0.0), (0.2, 0.7), (0.55, 1.2), (0.9, 0.0)):
        a(("aniso_obl_t%.2f_p%.2f" % (th, ph), "lossless",
           (lambda th=th, ph=ph:
            _j2(V.uniaxial_cell(), 1.5, theta=th, phi=ph, nord=4))))
    # -- lossless scalar 2-D --------------------------------------------------
    for blk in (4.0, 12.0, 2.25 + 1e-6):
        a(("scalar2d_blk%.6g" % blk, "lossless",
           (lambda blk=blk: _e2(V.scalar_cell(blk=blk), 1.5))))
    # -- lossless 1-D TE / TM -------------------------------------------------
    for pol in ("te", "tm"):
        for duty in (0.02, 0.5, 0.85):
            a(("oned_%s_duty%.2f" % (pol, duty), "lossless",
               (lambda pol=pol, duty=duty: _e1(2.1, 1.5, 1.5, duty=duty,
                                               pol=pol))))
    # -- high contrast, many orders (the deepest evanescent spectrum) --------
    for nord in (21, 31):
        a(("oned_hicontrast_n%d" % nord, "lossless",
           (lambda nord=nord: _e1(3.5, 1.0, 1.45, duty=0.4, pol="te", nord=nord))))
    a(("aniso_n8_hitrunc", "lossless",
       (lambda: _j2(V.uniaxial_cell(), 1.5, nord=8))))

    # -- LOSS LADDER, anisotropic 2-D (the population the band must not touch)
    for im in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12, 1e-14):
        a(("lossladder_aniso_im%.0e" % im, "lossy",
           (lambda im=im: _j2(V.uniaxial_cell(eps_im=im), 1.5))))
    # -- LOSS LADDER, 1-D ridge only -----------------------------------------
    for im in (1e-2, 1e-4, 1e-8, 1e-12, 1e-14):
        a(("lossladder_oned_im%.0e" % im, "lossy",
           (lambda im=im: _j1(np.array(4.41 + 1j * im),
                              np.array(2.25 + 0j), 1.5))))
    # -- metals: large NEGATIVE real permittivity ----------------------------
    for er, ei in ((-20.0, 1.5), (-5.0, 0.2), (-40.0, 0.5), (-100.0, 5.0)):
        a(("metal_%g%+gj" % (er, ei), "lossy",
           (lambda er=er, ei=ei: _j1(np.array(er + 1j * ei),
                                     np.array(2.25 + 0j), 1.5, duty=0.4))))

    # -- near-cutoff / near-Wood: an order at kz ~ 0 -------------------------
    # A Rayleigh (Wood) anomaly of the SUBSTRATE sits where
    #   n_sub = |sin(theta) + m wl / period|.
    # Walk the period so the m = -1 order grazes the substrate, i.e.
    #   wl / period -> n_sub  ->  period -> wl / n_sub.
    wood = WL / 1.5
    for d in (0.0, 1e-12, 1e-9, 1e-6, 1e-3):
        a(("wood_sub_d%.0e" % d, "lossless",
           (lambda d=d: _e1(2.1, 1.5, 1.5, duty=0.5, pol="te", nord=11,
                            period=wood * (1.0 + d)))))
    # ... and of the SUPERSTRATE (n = 1), where lam^2 of a region mode is 0
    for d in (0.0, 1e-9, 1e-3):
        a(("wood_sup_d%.0e" % d, "lossless",
           (lambda d=d: _e1(2.1, 1.5, 1.5, duty=0.5, pol="te", nord=11,
                            period=WL * (1.0 + d)))))

    # -- NEAR-CUTOFF LAYER MODE: engineered, not hoped for.  The layer's own
    # lam^2 -> 0 when one of its modal effective indices meets |kx| of some
    # order.  Scan the period, take the minimiser, then walk relative detunes
    # away from it -- so the fixture carries a mode at the cut whatever the
    # build's arithmetic does (TESTING_STANDARDS rule 3).
    p_star, lam2_star = _find_layer_cutoff_period()
    for d in (0.0, 1e-12, 1e-8, 1e-4):
        a(("layercut_d%.0e" % d, "lossless",
           (lambda d=d: _e1(2.1, 1.5, 1.5, duty=0.5, pol="te", nord=11,
                            period=p_star * (1.0 + d)))))
    for d in (0.0, 1e-8):
        a(("layercut_tm_d%.0e" % d, "lossless",
           (lambda d=d: _e1(2.1, 1.5, 1.5, duty=0.5, pol="tm", nord=11,
                            period=p_star * (1.0 + d)))))

    # -- near-DEGENERATE mounts: a square cell whose (+-1,0)/(0,+-1) modes are
    # exactly four-fold degenerate, walked off the degeneracy by a relative d.
    for d in (0.0, 1e-12, 1e-8, 1e-4):
        a(("degen_square_d%.0e" % d, "lossless",
           (lambda d=d: _j2(V.uniaxial_cell(twist=0.0, ne=1.5 * (1.0 + d)),
                            1.5, nord=5))))
    return f


# ----------------------------------------------------------------- the census
def census(name, cls, call):
    eig = V.EigSpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with eig:
            res = call()
    from lumenairy.elements.rcwa import _core as rc
    ratios = []
    lam2_min_abs = np.inf
    worst_growth = 0.0        # max Re(-lam) over returned roots -> |X| > 1
    n_exact_zero = 0
    for lam2 in eig.seen:
        r = V.principal_root(lam2)
        ratio, scale = V.band_ratio(r)
        sel = r.imag < 0
        if sel.any():
            ratios.extend(ratio[sel].tolist())
        lam2_min_abs = min(lam2_min_abs, float(np.min(np.abs(lam2))))
        n_exact_zero += int(np.sum(lam2 == 0))
        lam = np.asarray(rc._sqrt_decay(lam2))
        worst_growth = max(worst_growth, float(np.max(-lam.real)))
    ratios = np.asarray(ratios) if ratios else np.zeros(0)
    lo = ratios[ratios <= BAND]
    hi = ratios[ratios > BAND]
    return dict(
        name=name, cls=cls,
        n_eig_arrays=len(eig.seen),
        n_negimag=int(ratios.size),
        n_below_band=int(lo.size), n_above_band=int(hi.size),
        max_below=float(lo.max()) if lo.size else None,
        min_above=float(hi.min()) if hi.size else None,
        max_ratio=float(ratios.max()) if ratios.size else None,
        lam2_min_abs=float(lam2_min_abs),
        n_lam2_exact_zero=n_exact_zero,
        max_negative_Re_lam=worst_growth,
        res_sum=float(np.sum([np.sum(np.abs(np.asarray(x)))
                              for x in res[1:3]])),
    )


def main():
    V.require_local_tree()
    out = sys.argv[1]
    V.claim_output(out)
    rows = []
    for name, cls, call in fixtures():
        try:
            rows.append(census(name, cls, call))
        except Exception as exc:
            rows.append(dict(name=name, cls=cls, error=repr(exc)))
        r = rows[-1]
        print("%-28s %-9s neg=%-5s below=%-4s max_below=%-11s above=%-5s "
              "min_above=%-11s |X|>1:%s" % (
                  name, cls, r.get("n_negimag", "-"), r.get("n_below_band", "-"),
                  ("%.3e" % r["max_below"]) if r.get("max_below") is not None
                  else "-",
                  r.get("n_above_band", "-"),
                  ("%.3e" % r["min_above"]) if r.get("min_above") is not None
                  else "-",
                  ("%.1e" % r["max_negative_Re_lam"])
                  if r.get("max_negative_Re_lam") is not None else "-"))

    def agg(pred):
        below, above = [], []
        for r in rows:
            if "error" in r or not pred(r):
                continue
            if r.get("max_below") is not None:
                below.append(r["max_below"])
            if r.get("min_above") is not None:
                above.append(r["min_above"])
        return dict(n_fixtures=sum(1 for r in rows
                                   if "error" not in r and pred(r)),
                    max_below_band=max(below) if below else None,
                    min_above_band=min(above) if above else None,
                    n_below=sum(r.get("n_below_band", 0) for r in rows
                                if "error" not in r and pred(r)),
                    n_above=sum(r.get("n_above_band", 0) for r in rows
                                if "error" not in r and pred(r)))

    summary = dict(
        all=agg(lambda r: True),
        lossless=agg(lambda r: r["cls"] == "lossless"),
        lossy=agg(lambda r: r["cls"] == "lossy"),
        max_negative_Re_lam=max(r.get("max_negative_Re_lam", 0.0)
                                for r in rows if "error" not in r),
        n_lam2_exact_zero=sum(r.get("n_lam2_exact_zero", 0) for r in rows
                              if "error" not in r),
        min_abs_lam2=min(r.get("lam2_min_abs", np.inf) for r in rows
                         if "error" not in r),
    )
    print("\nSUMMARY")
    for k in ("all", "lossless", "lossy"):
        s = summary[k]
        print("  %-9s fixtures=%-3s below=%-6s max_below=%-11s above=%-6s "
              "min_above=%s" % (
                  k, s["n_fixtures"], s["n_below"],
                  ("%.4e" % s["max_below_band"])
                  if s["max_below_band"] is not None else "-",
                  s["n_above"],
                  ("%.4e" % s["min_above_band"])
                  if s["min_above_band"] is not None else "-"))
    print("  max Re(-lam) over every returned root (|X|>1 iff > 0): %.3e"
          % summary["max_negative_Re_lam"])
    print("  min |lam^2| seen: %.3e ; exact zeros: %d"
          % (summary["min_abs_lam2"], summary["n_lam2_exact_zero"]))
    V.dump(out, dict(rows=rows, summary=summary, band=BAND))


if __name__ == "__main__":
    main()
