"""Q5d -- O2, the interface-conditioning and near-cut-off half of the
mechanism, at the (mount, n_orders) pairs q5c found.

q5c settles the MODE-LEVEL half: which pairs blow up, and what the
forward/backward split does.  This one measures, at the same pairs:

  * ``cond`` of the mode matrix each generalized interface inverts and of the
    ``T22`` block it inverts explicitly;
  * the largest propagation factor the star product actually forms;
  * every retained order's ``|alpha|`` against the SUPERSTRATE, SUBSTRATE and
    every CELL-VALUE cut-off, so a near-degenerate layer <-> region match can
    be read off rather than assumed;
  * the same fixture on a VERTICAL layer and on the PURE engine.
"""
from __future__ import annotations

import math
import time
import warnings

import _lib as L
import numpy as np
from q5_o2_blowup import Probe
from q5c_mechanism import D1, DF, FEPS, MOUNTS, NSUB, NSUP, PX, PY, TSL, WL, cell

PAIRS = [("oblique25", 3), ("oblique25", 5), ("oblique25", 7),
         ("oblique25", 9), ("conical25_40", 5), ("oblique40", 3),
         ("normal", 5)]


def solve(mount, M, kind, probe=None):
    from lumenairy.elements.pmm import stack2d as S2
    st = S2.PMM2DStackHybrid(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                             n_orders=M)
    sl = (TSL, 0.0) if kind.startswith("slant") else None
    st.add_layer(D1, eps_cell=cell(), slant=sl)
    if kind.endswith("over_film"):
        st.add_layer(DF, eps=FEPS)
    th, ph = MOUNTS[mount]
    st.set_source(WL, theta=th, phi=ph)
    if probe is not None:
        probe.reset()
        probe.install(S2)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _o, R, T, _J = st.solve()
        rec = dict(outcome="SOLVED",
                   RT=float(np.max(np.asarray(R).sum(axis=1)
                                   + np.asarray(T).sum(axis=1))),
                   warned=[str(x.message)[:80] for x in w])
    except Exception as e:                              # noqa: BLE001
        rec = dict(outcome="RAISE", exc=type(e).__name__, msg=str(e)[:160])
    finally:
        if probe is not None:
            probe.restore(S2)
    if probe is not None:
        rec["probe"] = probe.summary()
    return rec


def cutoff_table(mount, M):
    th, ph = MOUNTS[mount]
    a0x = math.sin(th) * math.cos(ph)
    a0y = math.sin(th) * math.sin(ph)
    vals = sorted({float(v.real) for v in cell().ravel()})
    cuts = {"superstrate(1.00)": 1.0, "substrate(2.25)": float(NSUB)}
    for v in vals:
        cuts["cell eps=%.2f" % v] = math.sqrt(v)
    al = []
    for m in range(-M, M + 1):
        for n in range(-M, M + 1):
            al.append((m, n, math.hypot(a0x + m * WL / PX,
                                        a0y + n * WL / PY)))
    out = {}
    for tag, c in cuts.items():
        near = min(al, key=lambda r: abs(r[2] - c))
        out[tag] = dict(cutoff=round(c, 6), m=near[0], n=near[1],
                        alpha=round(near[2], 6),
                        gap=round(near[2] - c, 6),
                        rel_gap=round((near[2] - c) / c, 6))
    return out


def main():
    t0 = time.time()
    res = {}
    pr = Probe()
    for mount, M in PAIRS:
        key = "%s|M%d" % (mount, M)
        row = {}
        for kind in ("slant_over_film", "slant_only", "vertical_over_film"):
            row[kind] = solve(mount, M, kind, probe=pr)
        row["cutoffs"] = cutoff_table(mount, M)
        try:
            from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
            pu = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                                n_modes=5, n_orders=min(M, 5))
            pu.add_layer(D1, eps_cell=cell(), slant=(TSL, 0.0))
            pu.add_layer(DF, eps=FEPS)
            th, ph = MOUNTS[mount]
            pu.set_source(WL, theta=th, phi=ph)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                po = pu.solve()
            row["pure"] = dict(
                outcome="SOLVED",
                RT=float(np.max(np.asarray(po[1]).sum(axis=1)
                                + np.asarray(po[2]).sum(axis=1))),
                warned=[str(x.message)[:80] for x in w])
        except Exception as e:                          # noqa: BLE001
            row["pure"] = dict(outcome="RAISE", exc=type(e).__name__,
                               msg=str(e)[:200])
        res[key] = row
        print("==", key)
        for k in ("slant_over_film", "slant_only", "vertical_over_film"):
            v = row[k]
            p = v.get("probe", {})
            print("   %-20s RT=%-12.5g cond(Mb)=%-12.5g cond(T22)=%-12.5g "
                  "maxprop=%-8.4g warned=%s"
                  % (k, v.get("RT") or float("nan"),
                     p.get("max_cond_Mb") or float("nan"),
                     p.get("max_cond_T22") or float("nan"),
                     p.get("max_prop_factor") or float("nan"),
                     bool(v.get("warned"))))
        print("   pure:", row["pure"].get("RT", row["pure"].get("exc")))
        for tag, c in row["cutoffs"].items():
            if abs(c["rel_gap"]) < 0.05:
                print("   NEAR CUT-OFF %-20s %s" % (tag, c))
    res["_seconds"] = round(time.time() - t0, 1)
    L.dump("q5d_conditioning", res)


if __name__ == "__main__":
    main()
