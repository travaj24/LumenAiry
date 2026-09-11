"""ROUND 2, D9 -- the near-cutoff ladder as a FAMILY, not as one sample.

``test_near_cutoff_channel_count_is_stable_over_the_ladder`` claims a property
of "the whole near-cutoff ladder" and asserts it on ``m = 0`` only, with a
``1e-6`` bar on the worst lossless closure.  The 5.45.1 build's OWN report
records the same ladder reaching **1.2716e-06** on Windows / PRESCOTT -> Katmai
/ 1 thread -- above the bar.  The gate escapes only because it hardcodes
``m = 0``.

This probe measures the ladder's closure envelope over ``m`` in {0, 1, 2} and
over the rung depth, on whatever arm it is run on, so the bar can be re-derived
from the measured envelope over the FAMILY rather than from one sample.

It also records the two bounds that belong in the gate and are absent from it:

  * ``_BOR_CHANNEL_REAL_FLOOR`` (1e-6 in ``qn``) bounds the ladder FROM BELOW.
    ``qn = n sqrt(delta)``, so once ``n sqrt(10^-e/2) < 1e-6`` the cutoff order
    is legitimately dropped by the channel gate and the count legitimately
    falls.  That is not a band defect, and a ladder that runs past it is
    measuring the channel gate instead.
  * the closure grows monotonically with rung depth, so a bar taken from a
    ladder that stops at ``e = 20`` is a statement about where the ladder
    stops.

Run:  python validation/probe_fix_bor_round2/r6_cutoff_family.py [--deep]
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
EPS = NREF ** 2


def fd_modes(m, k0, eps=EPS, N=NFD):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, RBIG, N,
                       lambda r: np.full_like(r, eps, dtype=complex),
                       float(k0), staggered=True)


def gamma_of(m, idx=2):
    L = fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def cutoff_stack(m, k0):
    from lumenairy.elements.bor.bor_stack import BORStack
    s = BORStack(RBIG, m, n_substrate=NREF, n_superstrate=NREF, N=NFD,
                 basis="fd")
    s.add_layer(0.4, eps=EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=EPS)
    s.set_source(k0=float(k0))
    return s.solve()


def ladder(m, e_lo=8, e_hi=21):
    from lumenairy.elements.bor._orient import _BOR_CHANNEL_REAL_FLOOR as FLOOR
    g = gamma_of(m)
    rungs = []
    counts = set()
    worst = 0.0
    for e_ in range(e_lo, e_hi):
        dl = 10.0 ** (-e_ / 2.0)
        qn = NREF * np.sqrt(dl)
        k0 = g / (NREF * np.sqrt(1.0 - dl))
        res = cutoff_stack(m, k0)
        en = np.asarray(res["energy"])
        c = float(np.max(np.abs(en - 1.0))) if en.size else float("nan")
        n = int(np.size(res["R"]))
        counts.add(n)
        if np.isfinite(c):
            worst = max(worst, c)
        rungs.append(dict(e=e_, delta=dl, qn=float(qn), n_channels=n,
                          closure=c,
                          above_channel_floor=bool(qn > FLOOR)))
    return dict(m=m, counts=sorted(counts), worst=worst, rungs=rungs,
                channel_real_floor=float(FLOOR))


def main():
    rec = banner("r6_cutoff_family")
    deep = "--deep" in sys.argv
    e_hi = 26 if deep else 21
    out = {}
    for m in (0, 1, 2):
        d = ladder(m, 8, e_hi)
        out["m%d" % m] = d
        print("  m=%d  counts=%s  worst closure=%.6e" % (m, d["counts"],
                                                         d["worst"]))
        for r in d["rungs"]:
            print("     e=%2d  qn=%.4e  n=%2d  closure=%.6e  %s"
                  % (r["e"], r["qn"], r["n_channels"], r["closure"],
                     "" if r["above_channel_floor"]
                     else "<- BELOW the channel gate's real floor"))
    fam = [out["m%d" % m] for m in (0, 1, 2)]
    # the family statistic, restricted to the rungs the channel gate admits
    env = 0.0
    env_all = 0.0
    counts_ok = True
    for d in fam:
        good = [r for r in d["rungs"] if r["above_channel_floor"]]
        if good:
            env = max(env, max(r["closure"] for r in good
                               if np.isfinite(r["closure"])))
            cs = {r["n_channels"] for r in good}
            if len(cs) != 1:
                counts_ok = False
        env_all = max(env_all, d["worst"])
    print("\n--- summary ---")
    print("  FAMILY envelope over m in {0,1,2}, rungs ABOVE the channel gate's")
    print("    own qn floor: worst closure = %.6e" % (env,))
    print("  same over EVERY rung measured (including below the floor): %.6e"
          % (env_all,))
    print("  channel count is ONE number per m on the admitted rungs: %s"
          % (counts_ok,))
    dump("r6_cutoff_family", dict(family=out, envelope_admitted=env,
                                  envelope_all=env_all,
                                  counts_single_per_m=counts_ok), rec)


if __name__ == "__main__":
    main()
