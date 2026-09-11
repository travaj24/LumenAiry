"""TASK D, second pass -- can the SPECTRAL conjunct ``|q|max / (n_max k0) >
1e4`` be made to MISS a manufactured sliver that is provably damaging?

THE STRUCTURE OF THE QUANTITY.  A manufactured element of width ``w`` gives the
spectral-element operator a ``1 / w^2`` stiffness, so the layer's spectrum
acquires ``|q|max ~ C / w`` with ``C`` a dimensionless constant of the basis
(measurable: the build's own ladder gives ``|q|max * w`` between 15 and 48 over
degrees 6 .. 12).  The screened quantity is therefore

    q_excess = |q|max / (n_max k0) ~ C / (w n_max k0).

At the refusal's geometric edge ``w = 1e-6 Rbig`` this is

    q_excess ~ C * 1e6 / (Rbig n_max k0) = C * 1e6 / (2 pi n_max (Rbig/lambda)),

i.e. it falls as the cell gets OPTICALLY LARGER.  The build measured the
LOW-``k0`` direction (which RAISES the ratio and is the false-positive side) and
fixed the bar from it.  Nothing in the build measures the other direction, and
on this algebra an optically large cell drives the very same relative geometric
defect BELOW the bar.

THE TEST.  Hold the geometric defect fixed in RELATIVE terms
(``delta / Rbig = 1e-6`` and ``1e-7``, i.e. exactly the rungs the contract
refuses at ``Rbig/lambda = 7.6``) and sweep the optical size.  Record the
verdict the shipped contract reaches, the measured ``q_excess``, and the
answer's distance from the exact ``delta -> 0`` reference (channel-SORTED, so a
permutation of the R array cannot be mistaken for a moved answer).

A rung that is DAMAGING and merely warned is a MISS.

Usage:  python v7_qbar.py <pre|post> [outdir]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import _vh  # noqa: E402


def _guard(state):
    from lumenairy.elements.bor import _sem_contract as C
    prev = C.BOR_SEM_MESH_GUARD
    C.BOR_SEM_MESH_GUARD = state
    return prev


def _solve(Rbig, k0, delta, eps_ring, degree, arm, m=1, ring_w=2.0):
    from lumenairy.elements.bor._sem_contract import verdict
    from lumenairy.elements.bor.bor_stack import BORStack
    prev = _guard(arm)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            s = BORStack(Rbig=Rbig, m=m, N=120, n_superstrate=1.0,
                         n_substrate=1.0, basis="sem", degree=degree)
            r0 = 0.375 * Rbig
            s.add_layer(0.5, segments=[(r0, eps_ring),
                                       (r0 + ring_w, 1.0), (Rbig, 1.0)])
            s.add_layer(0.5, segments=[(r0 + delta, 0.5 * eps_ring),
                                       (Rbig, 1.0)])
            s.set_source(k0=k0)
            try:
                res = s.solve()
                R = np.sort(np.asarray(res["R"]))[::-1]
                T = np.sort(np.asarray(res["T"]))[::-1]
                out = dict(raised=None, n=int(R.size), R=R.tolist(),
                           T=T.tolist(),
                           closure=float(np.max(np.abs(
                               np.asarray(res["R"]) + np.asarray(res["T"])
                               - 1.0))))
            except BaseException as e:              # noqa: BLE001
                out = dict(raised=type(e).__name__, msg=str(e)[:200])
            out["n_warn"] = len(w)
            rep = getattr(s, "_sem_mesh_report", None)
            if rep:
                out["q_excess"] = max(x["q_excess"] for x in rep)
                out["w_union_frac"] = min(x["w_min_union_frac"] for x in rep)
                out["verdicts"] = sorted({verdict(x) for x in rep})
                out["n_elements"] = max(x["n_elements"] for x in rep)
            return out
    finally:
        _guard(prev)


def sweep():
    rows = []
    # Rbig/lambda from 7.6 (the build's own ladder) up.  eps_ring raises
    # n_max, which is the other half of the denominator.
    for Rbig, k0, eps_ring in ((24.0, 2.0, 4.0),      # the build's ladder
                               (24.0, 2.0, 16.0),
                               (24.0, 6.0, 16.0),
                               (24.0, 12.0, 16.0),
                               (24.0, 20.0, 16.0),
                               (24.0, 32.0, 16.0),
                               (24.0, 32.0, 36.0),
                               (24.0, 48.0, 36.0)):
        for dfrac in (1e-6, 1e-7):
            for degree in (6, 8):
                ref = _solve(Rbig, k0, 0.0, eps_ring, degree, False)
                raw = _solve(Rbig, k0, dfrac * Rbig, eps_ring, degree, False)
                arm = _solve(Rbig, k0, dfrac * Rbig, eps_ring, degree, True)
                rec = dict(Rbig=Rbig, k0=k0, eps_ring=eps_ring,
                           Rbig_over_lambda=Rbig * k0 / (2 * np.pi),
                           n_max=float(np.sqrt(eps_ring)),
                           delta_frac=dfrac, degree=degree,
                           q_excess=raw.get("q_excess"),
                           w_union_frac=raw.get("w_union_frac"),
                           verdicts=raw.get("verdicts"),
                           n_elements=raw.get("n_elements"),
                           armed_raised=arm.get("raised"),
                           armed_nwarn=arm.get("n_warn"),
                           closure=raw.get("closure"),
                           ref_closure=ref.get("closure"),
                           n=raw.get("n"), n_ref=ref.get("n"))
                if raw.get("R") and ref.get("R") and \
                        len(raw["R"]) == len(ref["R"]):
                    a = np.asarray(raw["R"])
                    b = np.asarray(ref["R"])
                    rec["dR_sorted_max"] = float(np.max(np.abs(a - b)))
                    rec["dR_sorted_rel"] = float(
                        np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))
                elif raw.get("R") and ref.get("R"):
                    rec["channel_count_moved"] = True
                rows.append(rec)
                print("  R/l=%-6.1f n_max=%-4.1f d=%-7.0e deg%-3d qexc=%-11.4g "
                      "%-20s armed=%-20s dRs=%s  nel=%s"
                      % (rec["Rbig_over_lambda"], rec["n_max"], dfrac, degree,
                         rec["q_excess"] or float("nan"),
                         str(rec["verdicts"]),
                         rec["armed_raised"] or ("warn%d" % rec["armed_nwarn"]),
                         ("%.4g" % rec["dR_sorted_max"])
                         if rec.get("dR_sorted_max") is not None else "chmoved",
                         rec.get("n_elements")), flush=True)
    return rows


def main():
    build = sys.argv[1]
    _vh.require_tree(build)
    a = _vh.arm()
    print("ARM", a, flush=True)
    with _vh.timed("qbar sweep"):
        rows = sweep()
    o = sys.argv[2] if len(sys.argv) > 2 else "."
    _vh.dump("%s/v7_qbar_%s_%s_%s_t%s.json"
             % (o, build, a["platform"], a["loaded_kernel"],
                a["blas_threads"]), dict(arm=a, build=build, sweep=rows))


if __name__ == "__main__":
    main()
