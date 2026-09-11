"""V4 -- the MISCLASSIFICATION hunt, in both directions, and the ``lam^2 -> 0``
corner.

The band the fix installed is

    on_cut = |Re(r)| <= _CUT_BAND_REL * max(max|r|, 1),   _CUT_BAND_REL = 1e-8

applied only to modes with ``Im(r) < 0``.  The audit reports the two populations
as separated by fourteen decades with the bar 7.6 decades above the noise side.
This probe attacks that from both ends, on states it ENGINEERS rather than hopes
a fixture will produce.

HUNT A -- a LOSSLESS PROPAGATING mode ABOVE the bar (the band would MISS it and
leave the build-dependent root in place).  For ``lam^2 = -s + i eta`` the
principal root's real part is ``eta / (2 sqrt(s))``, so the ratio grows without
limit as a layer mode approaches CUTOFF (``s -> 0``) at fixed backward error
``eta ~ eps_mach ||M||``.  The period is bisected to walk ``min |lam^2|`` down
through the decades and the ratio is read at each rung.

HUNT B -- a genuinely LOSSY mode BELOW the bar (the band would CONJUGATE a
physical root).  A fine ``Im(eps)`` ladder from 1e-2 to 1e-18, recording for
each rung how many modes are in the acted-on population at all (``Im(r) < 0``)
and the smallest ratio among them.

HUNT C -- ``lam^2`` exactly zero and denormal-small, straight through
``_sqrt_decay``; and the ``|X| = |exp(-lam k0 L)| <= 1`` contraction guarantee,
which is equivalent to ``Re(lam) >= 0`` on every returned root.

Usage: python v4_hunt.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v_fixtures as V  # noqa: E402

BAND = 1e-8


def _e1(period, pol="tm", nord=11, duty=0.5, nr=2.1, ng=1.5, ns=1.5):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    return rcwa_efficiency_1d(period, nr, ng, ns, 1.0, V._DEPTH, duty, V._WL,
                              polarization=pol, n_orders=nord)


def _layer_spectrum(period, pol="tm", nord=11):
    """The layer eigenvalues the solve itself produced, plus the closure.

    A mount driven onto a layer cutoff can trip the library's own energy
    tripwire (``_EnergyError``); the eigenvalues are produced BEFORE that
    raise, so they are still read out and the closure is recorded as the
    refusal's own super-unity.  Refusing to look would hide precisely the
    regime the hunt is about.
    """
    eig = V.EigSpy()
    clo = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with eig:
            try:
                res = _e1(period, pol=pol, nord=nord)
                clo = V.closure_defect_eff(res)
            except Exception as exc:
                clo = float("nan")
                _LAST_ERROR[0] = repr(exc)[:120]
    if not eig.seen:
        return np.zeros(0, complex), clo
    lam2 = np.concatenate([np.asarray(w) for w in eig.seen])
    return lam2, clo


_LAST_ERROR = [None]


def _score(lam2):
    """Score one layer spectrum.

    ``ratio_prop`` is the LARGEST band ratio among modes that are BOTH in the
    acted-on population (``Im(r) < 0``) AND classified PROPAGATING for a
    lossless layer (``Re(lam^2) < 0`` with ``|Im(lam^2)|`` a rounding-level
    fraction of ``|Re(lam^2)|``).  That is exactly the population the band must
    reach: if this exceeds ``_CUT_BAND_REL`` the band MISSES a mode whose root
    is then decided by the eigensolver's last bit.

    ``ratio_min_lam2`` is the ratio of the single mode nearest CUTOFF, whether
    or not it is in the population.
    """
    if lam2.size == 0:
        return dict(min_abs_lam2=float("nan"))
    r = V.principal_root(lam2)
    ratio, scale = V.band_ratio(r)
    pop = r.imag < 0
    prop = (lam2.real < 0) & (np.abs(lam2.imag) <= 1e-6 * np.abs(lam2.real))
    k0 = int(np.argmin(np.abs(lam2)))
    sel = pop & prop
    return dict(
        min_abs_lam2=float(np.min(np.abs(lam2))),
        scale=float(scale),
        n_population=int(pop.sum()),
        n_propagating_population=int(sel.sum()),
        ratio_prop=(float(np.max(ratio[sel])) if sel.any() else None),
        ratio_min_lam2=float(ratio[k0]),
        min_lam2_is_propagating=bool(prop[k0]),
        min_lam2_in_population=bool(pop[k0]),
    )


# ------------------------------------------------------------------- HUNT A
def hunt_a(pol="tm", nord=11):
    """Bisect the period toward a LAYER mode at cutoff and read the ratio."""
    rows = []
    lo, hi = 0.20e-6, 1.60e-6
    grid = np.linspace(lo, hi, 401)
    vals = []
    for p in grid:
        lam2, _ = _layer_spectrum(p, pol=pol, nord=nord)
        vals.append(float(np.min(np.abs(lam2))) if lam2.size else float("inf"))
    k = int(np.argmin(vals))
    a0 = float(grid[max(k - 1, 0)])
    b0 = float(grid[min(k + 1, len(grid) - 1)])

    def m(p):
        lam2, _ = _layer_spectrum(p, pol=pol, nord=nord)
        return float(np.min(np.abs(lam2))) if lam2.size else float("inf")

    seen_decades = set()
    for _ in range(120):
        m1 = a0 + (b0 - a0) / 3.0
        m2 = b0 - (b0 - a0) / 3.0
        v1, v2 = m(m1), m(m2)
        p_here, v_here = (m1, v1) if v1 < v2 else (m2, v2)
        dec = (int(np.floor(np.log10(v_here)))
               if np.isfinite(v_here) and v_here > 0 else -400)
        if dec not in seen_decades:
            seen_decades.add(dec)
            lam2, clo = _layer_spectrum(p_here, pol=pol, nord=nord)
            with V.PreSqrtDecay():
                lam2_pre, clo_pre = _layer_spectrum(p_here, pol=pol,
                                                    nord=nord)
            row = _score(lam2)
            row.update(period=p_here, pol=pol, closure=clo,
                       closure_pre_arm=clo_pre)
            rp = row.get("ratio_prop")
            row["propagating_mode_above_band"] = (rp is not None and rp > BAND)
            rows.append(row)
        if v1 < v2:
            b0 = m2
        else:
            a0 = m1
        if b0 - a0 < 1e-22:
            break
    rows.sort(key=lambda r: r["min_abs_lam2"], reverse=True)
    return rows


# ------------------------------------------------------------------- HUNT B
def hunt_b():
    rows = []
    ladder = [10.0 ** (-k) for k in range(2, 19)]
    for im in ladder:
        for tag, call in (
            ("aniso2d", lambda im=im: _aniso(im)),
            ("oned_tm", lambda im=im: _oned_lossy(im)),
        ):
            eig = V.EigSpy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with eig:
                    call()
            lam2 = np.concatenate([np.asarray(w) for w in eig.seen])
            r = V.principal_root(lam2)
            ratio, scale = V.band_ratio(r)
            sel = r.imag < 0
            n_pop = int(sel.sum())
            below = ratio[sel & (ratio <= BAND)] if n_pop else np.zeros(0)
            rows.append(dict(kind=tag, eps_imag=im, n_population=n_pop,
                             n_below_band=int(below.size),
                             min_ratio_in_population=(float(np.min(ratio[sel]))
                                                      if n_pop else None),
                             max_ratio_below=(float(below.max())
                                              if below.size else None),
                             scale=scale))
    return rows


def _aniso(im):
    from lumenairy.elements.rcwa import rcwa_jones_2d
    return rcwa_jones_2d(V._P, V._P, V.uniaxial_cell(eps_im=im), 1.5, 1.0,
                         V._DEPTH, V._WL, n_orders_x=5, n_orders_y=5)


def _oned_lossy(im):
    from lumenairy.elements.rcwa import rcwa_efficiency_1d
    return rcwa_efficiency_1d(V._P, np.sqrt(4.41 + 1j * im), 1.5, 1.5, 1.0,
                              V._DEPTH, 0.5, V._WL, polarization="tm",
                              n_orders=11)


# ------------------------------------------------------------------- HUNT C
def hunt_c():
    from lumenairy.elements.rcwa import _core as rc
    probes = {
        "exact_zero": 0.0 + 0.0j,
        "neg_zero_imag": complex(0.0, -0.0),
        "tiny_neg_real": -1e-300 + 0j,
        "tiny_pos_real": 1e-300 + 0j,
        "denormal": 5e-324 + 0j,
        "on_cut_plus": complex(-2.25, 0.0),
        "on_cut_minus": complex(-2.25, -0.0),
        "on_cut_noise": complex(-2.25, -2.911e-15),
        "evanescent_noise": complex(+2.25, -2.911e-15),
        "lossy_real": complex(-2.25, -0.5),
        "nan": complex(float("nan"), 0.0),
    }
    out = {}
    for k, v in probes.items():
        lam = np.asarray(rc._sqrt_decay(np.array([v], dtype=complex)))[0]
        out[k] = dict(lam2=[v.real, v.imag], lam=[float(lam.real),
                                                  float(lam.imag)],
                      re_nonneg=bool(lam.real >= 0.0)
                      or bool(np.isnan(lam.real)))
    # the guarantee, over a wide random spectrum plus every cut case
    rng = np.random.default_rng(20260911)
    big = (rng.normal(size=4000) + 1j * rng.normal(size=4000)) * 10.0
    big = np.concatenate([big, np.array(list(probes.values()))[:-1]])
    lam = np.asarray(rc._sqrt_decay(big))
    out["_sweep"] = dict(
        n=int(lam.size),
        min_Re_lam=float(np.min(lam.real)),
        n_negative_Re=int(np.sum(lam.real < 0.0)),
        max_root_residual=float(np.max(np.abs(lam ** 2 - big))
                                / np.max(np.abs(big))),
    )
    return out


def main():
    V.require_local_tree()
    out = sys.argv[1]
    V.claim_output(out)
    print("HUNT A -- near-cutoff layer mode (lossless, PROPAGATING?)")
    a_tm = hunt_a(pol="tm")
    a_te = hunt_a(pol="te")
    for tag, rows in (("tm", a_tm), ("te", a_te)):
        for r in rows:
            print("  %-3s |lam2|min=%.3e ratio_prop=%-11s nprop=%-3s "
                  "ratio@cut=%-11.4e cutIsProp=%-5s above=%-5s closure=%+.2e"
                  % (tag, r["min_abs_lam2"],
                     ("%.4e" % r["ratio_prop"])
                     if r.get("ratio_prop") is not None else "-",
                     r.get("n_propagating_population"),
                     r["ratio_min_lam2"], r["min_lam2_is_propagating"],
                     r["propagating_mode_above_band"], r["closure"]),
                  " pre=%+.2e" % r["closure_pre_arm"])
    print("\nHUNT B -- loss ladder")
    b = hunt_b()
    for r in b:
        print("  %-8s Im(eps)=%.0e pop=%-4s below=%-3s min_ratio=%s"
              % (r["kind"], r["eps_imag"], r["n_population"],
                 r["n_below_band"],
                 ("%.4e" % r["min_ratio_in_population"])
                 if r["min_ratio_in_population"] is not None else "-"))
    print("\nHUNT C -- lam^2 corners")
    c = hunt_c()
    for k, v in c.items():
        print("  %-18s %s" % (k, v))
    V.dump(out, dict(hunt_a_tm=a_tm, hunt_a_te=a_te, hunt_b=b, hunt_c=c,
                     band=BAND))
    cand = [r for r in a_tm + a_te if r.get("ratio_prop") is not None]
    if cand:
        w = max(cand, key=lambda r: r["ratio_prop"])
        print("")
        print("CLOSEST APPROACH FROM THE NOISE SIDE (a LOSSLESS "
              "PROPAGATING mode inside the acted-on population): "
              "ratio %.4e at |lam^2| = %.3e -- bar %.0e -> %s"
              % (w["ratio_prop"], w["min_abs_lam2"], BAND,
                 "MISSED BY THE BAND" if w["ratio_prop"] > BAND
                 else "inside the band"))
    wc = max(a_tm + a_te, key=lambda r: r["ratio_min_lam2"])
    print("CLOSEST APPROACH BY THE MODE AT CUTOFF: ratio %.4e at "
          "|lam^2| = %.3e (propagating=%s) -- bar %.0e"
          % (wc["ratio_min_lam2"], wc["min_abs_lam2"],
             wc["min_lam2_is_propagating"], BAND))


if __name__ == "__main__":
    main()
