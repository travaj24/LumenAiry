"""CLASS A on BOR -- the forward-orientation (branch-cut peer) census.

WHAT IS BEING TESTED.  The Cartesian defect was a ROOT-SELECTION rule
(``sqrt`` + an exact-zero pin on ``Re(r)``) that could hand a propagating
lossless layer mode the INCOMING root.  The BOR engines do NOT select a root
by a sign test on ``Im(lam^2)``: all five sites classify with the RELATIVE
band ``|Im q| < 1e-9 max(|Re q|, 1e-300)`` and then orient a PROPAGATING mode
by the sign of its own ``r dr`` z-Poynting flux and an EVANESCENT one by
``Im q > 0``.  Three things can still go wrong; each is measured here:

  (1) FLUX DECIDED ON NOISE -- a mode whose |flux| is at rounding level has
      its sign chosen by rounding.  Measured as the population that falls
      into the flux normalizer's own field-norm FALLBACK branch
      (``|P| <= 1e-10 fnrm``) while classified propagating.
  (2) A MODE ORIENTED BACKWARD -- a propagating mode shipped with negative
      z-flux, or an evanescent mode shipped with ``Im q < 0`` (a GROWING
      forward propagator).
  (3) THE CLASSIFIER'S TWO-SIDED MARGIN -- the largest ``|Im q| / |Re q|``
      among modes CALLED propagating (the NOISE side, which the band must
      reach) against the smallest among modes CALLED evanescent (the SIGNAL
      side, which it must not), plus an ``Im(n)`` ladder from 1e-1 to 1e-13
      that locates the crossing exactly.

Plus the RCWA killer fixture ported to cylindrical coordinates: a uniform
layer whose eps EXACTLY EQUALS the region's, and a 1e-6 detune, where the
Cartesian mode match went singular at ``cond(a + b) = 1.97e15``.
"""
from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (  # noqa: E402
    closure,
    dump,
    make_stack,
    mode_table,
    orientation_audit,
    pin_tree,
    solve_quiet,
)

print("TREE", pin_tree())

K0 = 2.0
LAM = 3.0
RBIG = 8 * LAM


def fixtures():
    out = []
    media = [("lossless_141", 1.41), ("lossless_150", 1.50),
             ("lossless_200", 2.00),
             ("lossy_150", 1.50 + 0.05j), ("lossy_200", 2.00 + 0.20j)]
    for (mname, n), m, basis in itertools.product(media, (0, 1, 2),
                                                  ("fd", "sem")):
        eps = complex(n) ** 2
        out.append(dict(
            id=f"coincide_{mname}_m{m}_{basis}", m=m, basis=basis, k0=K0,
            Rbig=RBIG, n_sup=n, n_sub=n, kind="coincide",
            layers=[dict(t=0.4, eps=eps)]))
        out.append(dict(
            id=f"spacers_{mname}_m{m}_{basis}", m=m, basis=basis, k0=K0,
            Rbig=RBIG, n_sup=n, n_sub=n, kind="spacer_grating",
            layers=[dict(t=0.4, eps=eps),
                    dict(t=0.5, rings=[LAM, 0.5, 2.45, 1.41]),
                    dict(t=0.4, eps=eps)]))
        out.append(dict(
            id=f"detune_{mname}_m{m}_{basis}", m=m, basis=basis, k0=K0,
            Rbig=RBIG, n_sup=n, n_sub=n, kind="detune",
            layers=[dict(t=0.4, eps=eps * (1 + 1e-6)),
                    dict(t=0.5, rings=[LAM, 0.5, 2.45, 1.41]),
                    dict(t=0.4, eps=eps * (1 + 1e-6))]))
    for m, basis in itertools.product((0, 1, 2), ("fd", "sem")):
        out.append(dict(
            id=f"thinring_m{m}_{basis}", m=m, basis=basis, k0=K0, Rbig=RBIG,
            n_sup=1.41, n_sub=1.41, kind="thin_ring",
            layers=[dict(t=0.4, eps=1.41 ** 2),
                    dict(t=0.5, segments=[[2.0, 6.0], [2.05, 2.25],
                                          [RBIG, 1.41 ** 2]]),
                    dict(t=0.4, eps=1.41 ** 2)]))
    for kk, m in itertools.product((0.45, 0.4501, 0.4502), (0, 1)):
        out.append(dict(
            id=f"cutoff_k{kk}_m{m}_fd", m=m, basis="fd", k0=kk, Rbig=RBIG,
            n_sup=1.41, n_sub=1.41, kind="near_cutoff", N=120,
            layers=[dict(t=0.4, eps=1.41 ** 2),
                    dict(t=0.5, rings=[LAM, 0.5, 2.45, 1.41])]))
    for m, basis in itertools.product((0, 1), ("fd", "sem")):
        out.append(dict(
            id=f"contrast_m{m}_{basis}", m=m, basis=basis, k0=K0, Rbig=RBIG,
            n_sup=1.41, n_sub=1.80, kind="contrast",
            layers=[dict(t=0.5, rings=[LAM, 0.5, 2.45, 1.41])]))
    return out


def cond_apb(La, Lb):
    """cond(a + b) of the pointwise mode match -- the quantity the Cartesian
    branch-cut defect drove to 1.97e15."""
    try:
        a = np.linalg.solve(Lb["W"], La["W"])
        b = np.linalg.solve(Lb["V"], La["V"])
        return float(np.linalg.cond(a + b))
    except Exception as exc:                       # noqa: BLE001
        return float("inf") if "ingular" in str(exc) else float("nan")


def audit_layer(L, sem):
    aud = orientation_audit(L, sem=sem)
    q, flux, rel, rho = mode_table(L, sem=sem)
    # BOTH-SIGNAL disagreement, split by which rule actually GOVERNED:
    #   prop side -- flux governed, Im was resolvable (>1e-13) and opposite
    #   evan side -- Im governed, flux was signal (>1e-10 rel) and opposite
    prop = rho < 1e-9
    agree = (flux >= 0.0) == (q.imag >= 0.0)
    p = prop & (rel > 1e-10) & (rho > 1e-13) & (~agree)
    e = (~prop) & (rel > 1e-10) & (rho > 1e-13) & (~agree)
    aud["n_disagree_prop_governed"] = int(p.sum())
    aud["n_disagree_evan_governed"] = int(e.sum())
    aud["worst_disagree_prop_rho"] = float(rho[p].max()) if p.any() else None
    aud["worst_disagree_evan_relflux"] = (float(rel[e].max()) if e.any()
                                          else None)
    return aud


def run_fixture(spec):
    s = make_stack(spec)
    res, warns = solve_quiet(s)
    d = s._last
    sem = spec["basis"] == "sem"
    rows = [dict(layer=t, **audit_layer(L, sem))
            for t, L in (("sup", d["sup"]), ("sub", d["sub"]))]
    rows += [dict(layer=f"mid{i}", **audit_layer(L, sem))
             for i, (_thk, L) in enumerate(d["mids"])]
    ca = None
    if d["mids"] and not sem:
        ca = cond_apb(d["sup"], d["mids"][0][1])
    return dict(id=spec["id"], kind=spec["kind"], m=spec["m"],
                basis=spec["basis"], k0=spec["k0"],
                n_sup=str(spec["n_sup"]), n_sub=str(spec["n_sub"]),
                n_orders=int(np.size(res["R"])),
                R=float(np.sum(res["R"])), T=float(np.sum(res["T"])),
                closure=closure(res), cond_apb_sup_mid0=ca,
                warnings=warns[:3], layers=rows)


def imag_ladder():
    """Locate the exact Im(n) at which a lossy mode crosses the 1e-9 bar."""
    from lumenairy.elements.bor.zcascade import layer_modes
    rows = []
    for m in (0, 1, 2):
        for ex in range(0, 14):
            ni = 0.0 if ex == 0 else 10.0 ** (-ex)
            eps = complex(1.41 + 1j * ni) ** 2
            L = layer_modes(m, RBIG, 120, lambda r, e=eps:
                            np.full_like(r, e, dtype=complex), K0,
                            staggered=True)
            q = np.asarray(L["q"])
            rho = np.abs(q.imag) / np.maximum(np.abs(q.real), 1e-300)
            prop = rho < 1e-9
            phys = np.abs(q.real) > np.abs(q.imag)
            rows.append(dict(
                m=m, imag_n=ni, n_modes=int(q.size),
                n_classified_prop=int(prop.sum()),
                n_phys_prop=int(phys.sum()),
                phys_prop_rho_min=float(rho[phys].min()) if phys.any() else None,
                n_lossy_called_prop=int((prop & phys).sum()) if ni > 0 else 0,
            ))
    return rows


def main():
    fx = fixtures()
    print(f"{len(fx)} fixtures")
    rows = []
    for spec in fx:
        try:
            rows.append(run_fixture(spec))
        except Exception as exc:                   # noqa: BLE001
            rows.append(dict(id=spec["id"],
                             error=f"{type(exc).__name__}: {exc}"[:300]))
        print("  ", rows[-1].get("id"), rows[-1].get("closure"),
              rows[-1].get("error", ""))
    lad = imag_ladder()
    tag = os.environ.get("PROBE_TAG", "win")
    dump(f"a1_orientation_{tag}.json", dict(fixtures=rows, imag_ladder=lad))

    lay = [L for r in rows for L in r.get("layers", [])]
    print("\n== SUMMARY ==")
    print("fixtures:", len(rows), " errored:",
          sum(1 for r in rows if "error" in r), " layers:", len(lay))
    for k in ("n_backward_prop", "n_backward_evan", "n_noise_oriented",
              "n_disagree_prop_governed", "n_disagree_evan_governed"):
        print(f"  {k:<28} {sum(L[k] for L in lay)}")
    pm = [L["prop_ratio_max"] for L in lay if L["prop_ratio_max"] is not None]
    em = [L["evan_ratio_min"] for L in lay if L["evan_ratio_min"] is not None]
    print(f"  classifier NOISE side worst  {max(pm):.4e}  "
          f"({np.log10(1e-9 / max(pm)):.2f} decades below the 1e-9 bar)")
    print(f"  classifier SIGNAL side worst {min(em):.4e}  "
          f"({np.log10(min(em) / 1e-9):.2f} decades above)")
    cl = [r["closure"] for r in rows
          if r.get("closure") is not None and np.isfinite(r["closure"])]
    print("  closure max:", max(cl) if cl else None)
    ca = [r["cond_apb_sup_mid0"] for r in rows
          if r.get("cond_apb_sup_mid0") is not None]
    print(f"  cond(a+b) over coincident/detuned FD interfaces: "
          f"{min(ca):.4f} .. {max(ca):.4f}")


if __name__ == "__main__":
    main()
