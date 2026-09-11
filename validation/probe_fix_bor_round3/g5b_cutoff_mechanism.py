"""GAP 5, the MECHANISM -- is the 1.79e-04 near-cutoff closure the band's noise
side or a residual defect, and what separates it from the PRE-fix defect?

THE PROBLEM WITH SIMPLY WIDENING THE BAR.  The PRE-fix per-mode band read
``|R + T - 1| = 1.2167e-04`` at ``qn ~ 2.5e-03`` -- a SHALLOW rung, where this
tree reads 2.3e-11.  The round-2 verification's 1.787437e-04 is at the
DEEPEST rung the ladder is entitled to, ``qn = 1.41e-05``.  A scalar bar
between them does not exist: the residual is LARGER than the defect it must
refuse.  So the two populations are separated by the rung's own distance from
cutoff, not by a scalar, and this probe measures that axis.

WHAT IS MEASURED, per ``(m, idx)`` and per rung:

  * ``qn_marginal`` = ``n sqrt(delta)`` -- the axial index of the channel the
    ladder is driving to cutoff, in units of the R/T channel gate's own floor
    ``_BOR_CHANNEL_REAL_FLOOR``;
  * the closure, and the per-incident-mode decomposition of it, so the excess
    can be attributed to a channel rather than to "the solve";
  * the SHALLOW / DEEP envelopes split at ``qn = 100 x`` the channel floor,
    which is where the family's closure leaves its round-off floor;
  * the backward-flux census at the worst rung (the orientation claim itself).
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _g3  # noqa: E402
import g5_cutoff_family as g5  # noqa: E402

from lumenairy.elements.bor import _orient as orient  # noqa: E402

#: where the family's closure leaves its round-off floor, in units of the R/T
#: channel gate's own floor ``_BOR_CHANNEL_REAL_FLOOR`` (1e-6).  MEASURED, not
#: chosen: see the SHALLOW / DEEP envelopes in the summary.
DEEP_MULT = 100.0
FLOOR = orient._BOR_CHANNEL_REAL_FLOOR


def main():
    a = _g3.arm()
    print("ARM", a["build"], a["loaded_kernel"], "t%s" % a["threads"],
          a["lumenairy_file"])
    t0 = time.time()
    rows = []
    for m in g5.MS:
        for idx in g5.IDXS:
            g = g5.gamma_of(m, idx)
            if g is None:
                continue
            for dl in g5.rungs():
                k0 = g / (g5.NREF * np.sqrt(1.0 - dl))
                _st, res = g5.cutoff_stack(m, k0)
                en = np.asarray(res["energy"], float)
                R = np.asarray(res["R"], float)
                T = np.asarray(res["T"], float)
                # BORStack.solve returns the incident propagating orders'
                # axial wavenumbers under ``q``; bor_solve.solve calls the
                # same array ``q_inc``.  Read whichever this container has.
                _q = res.get("q_inc", res.get("q"))
                qn = (np.real(np.asarray(_q)) / k0
                      if _q is not None else None)
                closure = float(np.max(np.abs(en - 1.0))) if en.size else 0.0
                jworst = int(np.argmax(np.abs(en - 1.0))) if en.size else -1
                rows.append(dict(
                    m=m, idx=idx, dl=dl, k0=k0,
                    qn_marginal=g5.NREF * float(np.sqrt(dl)),
                    qn_over_floor=g5.NREF * float(np.sqrt(dl)) / FLOOR,
                    n_channels=int(en.size),
                    closure=closure,
                    worst_channel=jworst,
                    qn_channels=(sorted(np.real(qn).tolist())
                                 if qn is not None else None),
                    energy=en.tolist(), R=R.tolist(), T=T.tolist()))
                print("  m=%d idx=%d dl=%-9.3g qn=%.4e n=%d closure=%.4e "
                      "worst_ch=%d" % (m, idx, dl, rows[-1]["qn_marginal"],
                                       en.size, closure, jworst))

    deep = [r for r in rows if r["qn_over_floor"] < DEEP_MULT]
    shal = [r for r in rows if r["qn_over_floor"] >= DEEP_MULT]

    def env(pop):
        return (max((r["closure"] for r in pop), default=None), len(pop))

    env_deep, n_deep = env(deep)
    env_shal, n_shal = env(shal)
    # is the excess carried by the MARGINAL channel?  ``worst_channel`` indexes
    # the incident mode; the marginal one is the smallest ``qn``.
    def marginal_is_worst(r):
        q = r["qn_channels"]
        if not q or r["worst_channel"] < 0:
            return None
        return bool(r["worst_channel"] == int(np.argmin(q)))
    att = [marginal_is_worst(r) for r in rows if r["closure"] > 1e-7]
    att = [x for x in att if x is not None]
    worst_row = max(rows, key=lambda r: r["closure"])
    fc, _res = g5.flux_census(worst_row["m"], worst_row["k0"])
    summary = dict(
        n_rows=len(rows),
        deep_mult=DEEP_MULT,
        channel_floor=FLOOR,
        envelope_deep=env_deep, n_deep=n_deep,
        envelope_shallow=env_shal, n_shallow=n_shal,
        decades_between=(np.log10(env_deep / env_shal)
                         if env_deep and env_shal else None),
        worst=dict(m=worst_row["m"], idx=worst_row["idx"], dl=worst_row["dl"],
                   qn=worst_row["qn_marginal"], closure=worst_row["closure"]),
        worst_flux_census=fc,
        marginal_carries_excess=dict(n=len(att), yes=int(sum(att))),
        counts_move=[(r["m"], r["idx"]) for r in rows
                     if r["n_channels"] != r["idx"] + 1],
        seconds=time.time() - t0,
    )
    for k in sorted(summary):
        print(" ", k, summary[k])
    _g3.dump("g5b_cutoff_mechanism", dict(rows=rows, summary=summary), a)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
