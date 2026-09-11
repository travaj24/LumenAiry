"""ROUND 2 restatements -- the orientation band's four sides, and whether the
``k0`` floor ever binds.

TWO NUMBERS THE 5.45.1 BUILD STATED FROM THE WRONG STATISTIC OR WITHOUT A
POPULATION:

  * THE SIGNAL SIDE.  The build quotes 0.98 decades at ``Im(n) = 1e-6``.  That
    is a MAXIMUM over the lossy population; the quantity that decides is the
    MINIMUM -- the closest approach to the band from above -- because the band
    must not reach ANY genuinely lossy mode.  The verification measured
    2.382e-08, i.e. 0.38 decades.  Re-measured here, with the population
    reported so the statistic can be checked rather than trusted.

  * THE ``k0`` FLOOR.  ``orient_band_scale`` returns ``max(max|q|, k0)``.  The
    floor is right in principle -- a literal 1.0 would make the band
    unit-system-dependent, which is audit P2-06 and is defect D13 in the EME
    peer -- but whether it ever BINDS is a measurement nobody made.  This probe
    counts the layers where ``max|q| < k0``, over the whole census.

Run:  python validation/probe_fix_bor_round2/r8_band_sides.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import banner, dump  # noqa: E402

RBIG = 24.0
NFD = 120
NREF = 1.41


def fd_modes(m, k0, eps=NREF ** 2, N=NFD, Rbig=RBIG):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, Rbig, N,
                       lambda r: np.full_like(r, eps, dtype=complex),
                       float(k0), staggered=True)


def sigma(L, k0):
    from lumenairy.elements.bor._orient import orient_band_scale
    q = np.asarray(L["q"])
    scale = float(np.real(orient_band_scale(q, float(k0), xp=np)))
    return q, np.abs(q.imag) / scale, scale


def gamma_of(m, idx=2):
    L = fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * NREF ** 2 - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def main():
    rec = banner("r8_band_sides")
    from lumenairy.elements.bor._orient import _BOR_CUT_BAND_REL as BAND
    print("  _BOR_CUT_BAND_REL = %.0e" % (BAND,))
    out = dict(band=BAND)
    floor_rows = []

    def note(label, L, k0):
        q = np.asarray(L["q"])
        mx = float(np.max(np.abs(q))) if q.size else 0.0
        floor_rows.append(dict(label=label, max_abs_q=mx, k0=float(k0),
                               ratio=mx / float(k0) if k0 else float("inf"),
                               floor_binds=bool(mx < float(k0))))

    # --- NOISE, ordinary lossless geometry: the MAXIMUM over physically
    #     propagating modes is what the band must REACH.
    worst_ord, n_ord, n_modes = 0.0, 0, 0
    for eps in (1.41 ** 2, 1.50 ** 2, 2.00 ** 2):
        for m in (0, 1, 2):
            for k0 in (0.8, 2.0, 3.5):
                for Rbig in (RBIG, RBIG * 1e-6):     # two unit systems
                    kk = k0 if Rbig == RBIG else k0 * 1e6
                    L = fd_modes(m, kk, eps=eps, Rbig=Rbig)
                    q, sg, _sc = sigma(L, kk)
                    note("ordinary eps=%g m=%d k0=%g Rbig=%g"
                         % (eps, m, kk, Rbig), L, kk)
                    phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
                    if phys.any():
                        worst_ord = max(worst_ord, float(np.max(sg[phys])))
                        n_ord += 1
                        n_modes += int(phys.sum())
    print("  NOISE ordinary   : %d layers / %d modes, worst sigma %.6e "
          "(%.2f decades of room)"
          % (n_ord, n_modes, worst_ord, np.log10(BAND / worst_ord)))
    out["noise_ordinary"] = dict(layers=n_ord, modes=n_modes, worst=worst_ord,
                                 decades=float(np.log10(BAND / worst_ord)))

    # --- NOISE at a deep cutoff: the binding side.
    worst_cut, n_cut, n_cmodes = 0.0, 0, 0
    for m in (0, 1, 2):
        g = gamma_of(m)
        for e_ in range(4, 27, 2):
            dl = 10.0 ** (-e_)
            k0 = g / (NREF * np.sqrt(1.0 - dl))
            L = fd_modes(m, k0)
            q, sg, _sc = sigma(L, k0)
            note("cutoff m=%d e=%d" % (m, e_), L, k0)
            phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
            if phys.any():
                worst_cut = max(worst_cut, float(np.max(sg[phys])))
                n_cut += 1
                n_cmodes += int(phys.sum())
    print("  NOISE deep cutoff: %d layers / %d modes, worst sigma %.6e "
          "(%.2f decades of room)"
          % (n_cut, n_cmodes, worst_cut, np.log10(BAND / worst_cut)))
    out["noise_cutoff"] = dict(layers=n_cut, modes=n_cmodes, worst=worst_cut,
                               decades=float(np.log10(BAND / worst_cut)))

    # --- SIGNAL: the MINIMUM is the statistic, over a ladder in Im(n).
    out["signal"] = {}
    for imn in (1e-3, 1e-5, 1e-6, 1e-7, 1e-8):
        smallest, largest, n = np.inf, 0.0, 0
        for m in (0, 1, 2):
            for k0 in (0.8, 2.0, 3.5):
                L = fd_modes(m, k0, eps=(NREF + 1j * imn) ** 2)
                q, sg, _sc = sigma(L, k0)
                note("lossy imn=%g m=%d k0=%g" % (imn, m, k0), L, k0)
                phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
                if phys.any():
                    smallest = min(smallest, float(np.min(sg[phys])))
                    largest = max(largest, float(np.max(sg[phys])))
                    n += int(phys.sum())
        dec = float(np.log10(smallest / BAND))
        print("  SIGNAL Im(n)=%.0e : %d modes, MIN sigma %.6e (%.2f decades "
              "above the band), max %.6e -- the build quoted the MAX"
              % (imn, n, smallest, dec, largest))
        out["signal"]["%.0e" % imn] = dict(modes=n, minimum=smallest,
                                           maximum=largest, decades=dec)

    # --- the k0 floor: does it ever bind?
    binds = [r for r in floor_rows if r["floor_binds"]]
    closest = min(floor_rows, key=lambda r: r["ratio"])
    print("  k0 FLOOR: binds on %d of %d measured layers; closest approach "
          "max|q| / k0 = %.4g (%s)"
          % (len(binds), len(floor_rows), closest["ratio"], closest["label"]))
    out["k0_floor"] = dict(layers=len(floor_rows), binds=len(binds),
                           closest_ratio=closest["ratio"],
                           closest_label=closest["label"])
    dump("r8_band_sides", dict(summary=out, floor_rows=floor_rows), rec)


if __name__ == "__main__":
    main()
