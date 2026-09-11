"""ROUND 2, D9 -- the near-cutoff ladder's envelope over the FAMILY and the
ARMS, and the bound that says where the ladder stops.

Post-processes every ``r6_cutoff_family_*.json`` arm and reports, per rung
depth cut:

  * the channel COUNT per ``m`` (must be ONE number per ``m``, on every arm);
  * the worst lossless closure over the family, which is what the gate's bar
    must clear.

THE LADDER'S OWN LOWER BOUND, which the shipped gate does not state.  The R/T
channel gate keeps a channel only while ``Re qn > _BOR_CHANNEL_REAL_FLOOR``
(1e-6).  The ladder places its cutoff order at ``qn = n sqrt(delta)``, so once
``qn`` approaches that floor the channel is LEGITIMATELY dropped and the count
legitimately falls -- that is the channel gate working, not the orientation
band failing.  A ladder that runs past it is measuring the wrong thing.  The
cuts below are stated as multiples of the floor so the bound is derived from
the library's own constant.

Run:  python validation/probe_fix_bor_round2/r9_cutoff_envelope.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    paths = sys.argv[1:] or sorted(
        glob.glob(os.path.join(HERE, "r6_cutoff_family_*.json")))
    if not paths:
        print("no r6 JSON found")
        return 1
    docs = []
    for p in paths:
        with open(p, encoding="utf-8") as fh:
            docs.append((os.path.basename(p), json.load(fh)))
    floor = docs[0][1]["family"]["m0"]["channel_real_floor"]
    print("channel gate's own floor  _BOR_CHANNEL_REAL_FLOOR = %.0e (in qn)"
          % (floor,))
    out = {"channel_real_floor": floor, "cuts": {}}
    for mult in (1.0, 3.0, 10.0, 30.0):
        cut = mult * floor
        print("\n=== rungs with qn >= %.3gx the floor (qn >= %.4g) ==="
              % (mult, cut))
        env = 0.0
        counts_ok = True
        rows = []
        for name, doc in docs:
            arm = doc["arm"]
            worst = 0.0
            cs = {}
            nr = 0
            for mk in ("m0", "m1", "m2"):
                d = doc["family"][mk]
                good = [r for r in d["rungs"] if r["qn"] >= cut]
                nr += len(good)
                cs[mk] = sorted({r["n_channels"] for r in good})
                if len(cs[mk]) != 1:
                    counts_ok = False
                for r in good:
                    if np.isfinite(r["closure"]):
                        worst = max(worst, r["closure"])
            env = max(env, worst)
            rows.append(dict(arm="%s/%s/t%s" % (
                "wsl" if arm["platform"].startswith("linux") else "win",
                arm.get("kernel"), arm.get("threads")),
                counts=cs, worst=worst, rungs=nr))
            print("  %-22s rungs=%2d  counts m0/m1/m2 = %s/%s/%s  worst=%.6e"
                  % (rows[-1]["arm"], nr, cs["m0"], cs["m1"], cs["m2"], worst))
        print("  ENVELOPE over %d arm(s): %.6e   counts single per m: %s"
              % (len(docs), env, counts_ok))
        if env > 0:
            for bar in (1e-6, 3e-6, 1e-5):
                print("     bar %.0e -> margin %.3gx (%.3f decades)"
                      % (bar, bar / env, np.log10(bar / env)))
        out["cuts"]["%.3gx" % mult] = dict(cut=cut, envelope=env,
                                           counts_single_per_m=counts_ok,
                                           rows=rows)
    with open(os.path.join(HERE, "r9_cutoff_envelope.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=str)
    print("\n[probe] wrote %s"
          % (os.path.join(HERE, "r9_cutoff_envelope.json"),))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
