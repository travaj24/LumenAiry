"""CLASS C on BOR SEM -- WHICH SCALE GOVERNS THE BAR?

The 2-D staggered PMM's contract is a PURE PERIOD FRACTION
(``_STAG_MIN_SEG_FRAC = 1e-3`` of the period, with a 1e-3 .. 3e-2 degradation
band).  The cylindrical peer has THREE candidate scales and the native
fixtures cannot separate them, because they happen to share
``Rbig / lambda_core`` to within 2.5x:

  S1  a fraction of ``Rbig``           (the domain radius -- the period's peer)
  S2  a fraction of the LOCAL WAVELENGTH ``lambda_loc = 2 pi / (k0 n)``
  S3  a fraction of the layer's OWN ORDINARY ELEMENT WIDTH -- which the DPW = 8
      cap sets to ``degree * lambda_loc / 8``, so S3 is an S2 in disguise
      EXCEPT that it also carries the degree.  This is the cylindrical peer of
      the 1-D guard's ``_SLIVER_OWN_SCALE_RATIO`` ("at least 100x finer than
      the finest wall spacing any single layer asked for").

THE DESIGN THAT SEPARATES THEM.  Sweep ``k0`` at FIXED ``Rbig`` (so
``lambda`` moves 16x while ``Rbig`` does not) and sweep ``Rbig`` at FIXED
``k0`` (so ``Rbig`` moves 4x while ``lambda`` does not).  Whichever scale the
ONSET of the damage holds still in is the one that governs.

The fixture is the WITHIN-LAYER liner of c2, whose eps equals the host it
replaces, so THE DEVICE DOES NOT DEPEND ON THE WIDTH and every deviation from
the width-free reference is numerical damage with no physics to subtract.

Reported per rung, all KERNEL-INDEPENDENT by construction except the last two:
``w / Rbig``, ``w / lambda_loc``, ``w / h_ordinary``, the spurious
``|q|max`` against the physical ceiling ``n_max k0``, and -- for contrast --
the closure and the q-matched per-order error.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, pin_tree  # noqa: E402
from c4_sliver_attrib import q_matched  # noqa: E402

print("TREE", pin_tree())

E_HI = 2.45 ** 2
E_LO = 1.41 ** 2
M = 1
DEGREE = 12


def build(Rbig, k0, wall, width, *, degree=DEGREE, N=200):
    from lumenairy import BORStack
    s = BORStack(Rbig, M, n_substrate=1.41, n_superstrate=1.41, N=N,
                 basis="sem", degree=degree)
    s.add_layer(0.4, eps=E_LO)
    segs = ([(wall, E_HI), (Rbig, E_LO)] if width <= 0.0 else
            [(wall - width, E_HI), (wall, E_HI), (Rbig, E_LO)])
    s.add_layer(0.5, segments=segs)
    s.add_layer(0.4, eps=E_LO)
    s.set_source(k0=k0)
    return s


def one(Rbig, k0, wall, width, *, degree=DEGREE):
    s = build(Rbig, k0, wall, width, degree=degree)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = s.solve()
    d = s._last
    qmax, wmin, hs = 0.0, np.inf, []
    for _t, L in ([("s", d["sup"]), ("b", d["sub"])]
                  + [("m", LL) for _tt, LL in d["mids"]]):
        qmax = max(qmax, float(np.max(np.abs(np.asarray(L["q"])))))
        mesh = L.get("mesh")
        if mesh is not None:
            dd = np.diff(mesh.b)
            wmin = min(wmin, float(dd.min()))
            hs.append(float(np.median(dd)))
    lam_hi = 2 * np.pi / (k0 * 2.45)
    return dict(Rbig=Rbig, k0=k0, wall=wall, width=float(width),
                degree=degree,
                q=np.asarray(res["q"], float).tolist(),
                R=np.asarray(res["R"], float).tolist(),
                n_orders=int(np.size(res["R"])), closure=closure(res),
                superunity=float(np.max(np.asarray(res["energy"])) - 1.0)
                if np.size(res["R"]) else None,
                qmax=qmax, qmax_over_ceiling=qmax / (2.45 * k0),
                min_element=None if not np.isfinite(wmin) else wmin,
                h_ordinary=float(np.median(hs)) if hs else None,
                lam_hi=float(lam_hi),
                Rbig_over_lam=float(Rbig / lam_hi),
                n_warn=len(w))


def ladder(Rbig, k0, *, fracs):
    wall = 0.25 * Rbig
    ref = one(Rbig, k0, wall, 0.0)
    rows = []
    for f in fracs:
        w = f * Rbig
        if w >= 0.5 * wall:
            continue
        r = one(Rbig, k0, wall, w)
        dR, resid, nm = q_matched(ref, r, "R")
        r.update(frac_Rbig=f, frac_lam=float(w / r["lam_hi"]),
                 frac_h=float(w / r["h_ordinary"]) if r["h_ordinary"] else None,
                 dR=dR, q_resid=resid, n_matched=nm)
        rows.append(r)
    return ref, rows


def onset(rows, *, bar=1e-3):
    """The widest rung whose q-matched error exceeds ``bar`` -- the ONSET of
    the damage, reported in all three candidate scales."""
    bad = [r for r in rows if r["dR"] is not None and r["dR"] > bar]
    if not bad:
        return None
    r = max(bad, key=lambda x: x["width"])
    return dict(width=r["width"], frac_Rbig=r["frac_Rbig"],
                frac_lam=r["frac_lam"], frac_h=r["frac_h"],
                dR=r["dR"], closure=r["closure"],
                qmax_over_ceiling=r["qmax_over_ceiling"])


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    krn = os.environ.get("OPENBLAS_CORETYPE", "(default)")
    fracs = [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7]
    # (Rbig, k0): the k0 sweep at fixed Rbig separates lambda from Rbig;
    # the Rbig sweep at fixed k0 separates Rbig from lambda.
    cases = [(24.0, 0.5), (24.0, 2.0), (24.0, 8.0), (12.0, 2.0), (48.0, 2.0)]
    payload = dict(threads=thr, kernel=krn, cases=[])
    for Rbig, k0 in cases:
        ref, rows = ladder(Rbig, k0, fracs=fracs)
        payload["cases"].append(dict(Rbig=Rbig, k0=k0, ref=ref, rows=rows,
                                     onset=onset(rows)))
        print(f"\n== Rbig={Rbig}  k0={k0}  lam_core={ref['lam_hi']:.4f}  "
              f"Rbig/lam={ref['Rbig_over_lam']:.2f}  h_ord="
              f"{ref['h_ordinary']:.4f}  ref closure={ref['closure']:.3e} ==")
        print("  w/Rbig   w/lam       w/h_ord     |q|/ceil    closure     "
              "R+T-1        dR (q-matched)  warn")
        for r in rows:
            print(f"  {r['frac_Rbig']:.0e}  {r['frac_lam']:.4e}  "
                  f"{r['frac_h']:.4e}  {r['qmax_over_ceiling']:.3e}  "
                  f"{r['closure']:.3e}  "
                  f"{(r['superunity'] if r['superunity'] is not None else float('nan')):+.3e}  "
                  f"{r['dR']:.6e}    {r['n_warn']}")
    dump(f"c5_which_scale_{tag}_{krn}_t{thr}.json", payload)

    print("\n== WHICH SCALE HOLDS STILL?  (onset = widest rung with "
          "q-matched dR > 1e-3) ==")
    print("  Rbig    k0     Rbig/lam   onset w      w/Rbig     w/lam      "
          "w/h_ord     |q|/ceil")
    for c in payload["cases"]:
        o = c["onset"]
        if o is None:
            print(f"  {c['Rbig']:<6} {c['k0']:<6} "
                  f"{c['ref']['Rbig_over_lam']:<10.2f} (no onset on this "
                  f"ladder)")
            continue
        print(f"  {c['Rbig']:<6} {c['k0']:<6} "
              f"{c['ref']['Rbig_over_lam']:<10.2f} {o['width']:.4e}  "
              f"{o['frac_Rbig']:.3e}  {o['frac_lam']:.3e}  "
              f"{o['frac_h']:.3e}  {o['qmax_over_ceiling']:.3e}")
    ok = [c["onset"] for c in payload["cases"] if c["onset"]]
    if len(ok) >= 2:
        for nm, key in (("w/Rbig", "frac_Rbig"), ("w/lambda", "frac_lam"),
                        ("w/h_ordinary", "frac_h"),
                        ("|q|max/ceiling", "qmax_over_ceiling")):
            v = [o[key] for o in ok]
            print(f"  spread of the onset in {nm:<15}: "
                  f"{min(v):.3e} .. {max(v):.3e}   "
                  f"({max(v) / min(v):.1f}x)")


if __name__ == "__main__":
    main()
