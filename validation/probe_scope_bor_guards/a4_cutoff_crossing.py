"""CLASS A on BOR -- THE BINDING POPULATION: a near-cutoff order crossing the
propagating classifier, and what decides its orientation once it has.

THE STRUCTURE.  The BOR eigensolve returns ``q^2`` with backward error
``|Im q^2| ~ eps_mach ||K||``; ``q = sqrt(q^2)`` then carries
``Im q ~ Im(q^2) / (2 |q|)``, so the classifier's discriminating ratio

    rho = |Im q| / |Re q| ~ |Im q^2| / (2 (Re q)^2)

grows as ``1 / qn^2`` as an order approaches its own cutoff.  At FIXED
backward error NO band scaled by the mode's own ``|Re q|`` survives that --
the same structural statement ``rcwa/_core._CUT_BAND_REL`` makes for the
Cartesian side, and the reason that constant is scaled by the SPECTRUM's
largest root instead.

THE QUESTION THIS PROBE ANSWERS.  Once ``rho`` crosses 1e-9 the order is
called EVANESCENT and is oriented by ``Im q > 0`` -- and ``Im q`` IS the
backward error.  Does the sign of that backward error agree with the sign of
the mode's own z-flux?  Where it AGREES the misclassification is harmless.
Where it DISAGREES the mode is shipped FORWARD while carrying BACKWARD power
flux, decided by a quantity no two BLAS builds agree on.

Per rung: the order's ``qn``, ``rho``, the class it landed in, the flux
verdict, the ``Im q`` verdict, whether they agree, and the full stack's
R / T / closure so any consequence is visible in an OBSERVABLE.  Re-run under
different ``OPENBLAS_NUM_THREADS`` and on both builds to read the spread.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, mode_table, pin_tree  # noqa: E402

print("TREE", pin_tree())
RBIG = 24.0
NFD = 120
EPS = 1.41 ** 2
NREF = 1.41


def fd_modes(m, k0, N=NFD):
    from lumenairy.elements.bor.zcascade import layer_modes
    return layer_modes(m, RBIG, N,
                       lambda r: np.full_like(r, EPS, dtype=complex),
                       float(k0), staggered=True)


def gamma_of(m, idx=2):
    L = fd_modes(m, 2.0)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def crossing(m, gamma, deltas):
    rows = []
    for dl in deltas:
        k0 = gamma / (NREF * np.sqrt(1.0 - dl))
        q, flux, rel, rho = mode_table(fd_modes(m, k0), sem=False)
        qn = q / k0
        # THE ORDER: physically propagating (|Re q| dominates) and closest to
        # cutoff.  This selection does NOT use the 1e-9 classifier, so the
        # order stays identifiable AFTER it has crossed.
        phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
        if not phys.any():
            continue
        j = np.where(phys)[0][int(np.argmin(np.abs(qn.real)[phys]))]
        called_prop = bool(rho[j] < 1e-9)
        rows.append(dict(
            delta=float(dl), k0=float(k0), m=m,
            qn=float(abs(qn[j].real)), rho=float(rho[j]),
            called_propagating=called_prop,
            oriented_by="flux" if called_prop else "Im(q)",
            flux_fwd=bool(flux[j] >= 0.0), im_fwd=bool(q[j].imag > 0.0),
            rules_agree=bool((flux[j] >= 0.0) == (q[j].imag > 0.0)),
            relflux=float(rel[j]),
            q=[float(q[j].real), float(q[j].imag)],
            kept_by_solve_gate=bool(abs(qn[j].imag) < 5e-5
                                    and qn[j].real > 1e-6)))
    return rows


def stack_consequence(m, gamma, deltas):
    from lumenairy import BORStack
    out = []
    for dl in deltas:
        k0 = gamma / (NREF * np.sqrt(1.0 - dl))
        s = BORStack(RBIG, m, n_substrate=NREF, n_superstrate=NREF, N=NFD,
                     basis="fd")
        s.add_layer(0.4, eps=EPS)                    # COINCIDENT spacer
        s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
        s.add_layer(0.4, eps=EPS)
        s.set_source(k0=float(k0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = s.solve()
        qi = np.asarray(res["q"])
        jj = int(np.argmin(qi)) if qi.size else -1
        out.append(dict(
            delta=float(dl), k0=float(k0), m=m,
            n_orders=int(np.size(res["R"])), closure=closure(res),
            sumR=float(np.sum(res["R"])), sumT=float(np.sum(res["T"])),
            cutoff_order_qn=float(qi[jj] / k0) if jj >= 0 else None,
            cutoff_order_R=float(res["R"][jj]) if jj >= 0 else None,
            cutoff_order_T=float(res["T"][jj]) if jj >= 0 else None,
            energy=[float(x) for x in np.asarray(res["energy"])[:6]]))
    return out


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    deltas = sorted({10.0 ** (-e / 2.0) for e in range(8, 21)}, reverse=True)
    payload = dict(threads=thr, crossings=[], stacks=[])
    for m in (0, 1, 2):
        g = gamma_of(m)
        rows = crossing(m, g, deltas)
        payload["crossings"].append(dict(m=m, gamma=g, rows=rows))
        payload["stacks"].append(dict(m=m, gamma=g,
                                      rows=stack_consequence(m, g, deltas)))
        print(f"\n== FD m={m} gamma={g:.8f} ==")
        print("  delta       qn          rho             class  orient_by  "
              "flux_fwd im_fwd agree  relflux")
        for r in rows:
            print(f"  {r['delta']:.2e}  {r['qn']:.4e}  {r['rho']:.4e}      "
                  f"{'PROP' if r['called_propagating'] else 'EVAN'}   "
                  f"{r['oriented_by']:>6}   {str(r['flux_fwd']):>5}  "
                  f"{str(r['im_fwd']):>5}  {str(r['rules_agree']):>5}  "
                  f"{r['relflux']:.3e}")
    dump(f"a4_crossing_{tag}_t{thr}.json", payload)

    allr = [r for c in payload["crossings"] for r in c["rows"]]
    crossed = [r for r in allr if not r["called_propagating"]]
    dis = [r for r in crossed if not r["rules_agree"]]
    print("\n== HEADLINE ==")
    print("  rungs:", len(allr))
    print("  rungs where a PHYSICALLY PROPAGATING order was called EVANESCENT:",
          len(crossed))
    if crossed:
        print(f"    first crossing at qn = {max(r['qn'] for r in crossed):.4e}"
              f", rho = {min(r['rho'] for r in crossed):.4e}")
        print("    of those, still counted as an R/T CHANNEL by solve():",
              sum(1 for r in crossed if r["kept_by_solve_gate"]))
    print("  rungs where the shipped FORWARD mode carries BACKWARD flux:",
          len(dis), [(r["m"], round(r["qn"], 8), r["rho"]) for r in dis][:6])
    for st in payload["stacks"]:
        cl = [r["closure"] for r in st["rows"] if r["closure"] == r["closure"]]
        print(f"  m={st['m']} stack closure max {max(cl) if cl else None:.4e}")


if __name__ == "__main__":
    main()
