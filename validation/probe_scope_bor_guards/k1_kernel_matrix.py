"""THE KERNEL MATRIX -- every BOR guard DECISION, under one BLAS kernel.

WHY.  The 5.45.0 CI (AMD EPYC / OpenBLAS ZEN) disagreed with both local builds
on guard DECISIONS that the local builds agreed on -- the 1-D sliver guard's
super-unity trigger and its attribution came out kernel-dependent.  A guard
whose verdict moves with the BLAS kernel is not a guard, so every
build-dependence claim in this scoping and every candidate bar has to be read
per kernel, and every candidate bar has to key on a quantity the kernel cannot
move.

RUN ONE KERNEL PER INVOCATION: ``OPENBLAS_CORETYPE=<kernel>`` on the command
line, with the thread counts pinned.  The ARCHITECTURE ACTUALLY OBTAINED is
read back from ``threadpoolctl`` and recorded in the output filename and
payload -- a requested kernel that silently falls back would otherwise be
recorded as a kernel that was never run.

WHAT IS RECORDED -- DECISIONS, not just numbers:

  A  per near-cutoff rung: which CLASS the classifier put the order in, which
     rule therefore governed, the flux verdict, the Im(q) verdict, whether the
     shipped forward mode carries backward flux, and the stack's CHANNEL COUNT
     and closure.  Also the same for the candidate spectrum-scaled band.
  B  the equilibrated rcond / residual at ``inv(a + b)`` for the healthy
     (staggered) and broken (legacy nodal) populations, and the nodal
     ``max(R + T)``.
  C  per sliver rung: the GEOMETRY (narrowest manufactured element),
     the spurious ``|q|max`` against the physical index ceiling, the
     interface ``rcond``, the closure and super-unity, and the q-matched
     per-order R error -- so a candidate bar keyed on GEOMETRY or on
     ``|q|max``/ceiling can be compared against one keyed on the ENERGY
     VIOLATION for kernel stability.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import closure, dump, inv_census, mode_table, pin_tree  # noqa: E402
from a5_arbiter_band import patched  # noqa: E402
from c4_sliver_attrib import q_matched, solve_one  # noqa: E402

print("TREE", pin_tree())
RBIG = 24.0
NFD = 120
EPS = 1.41 ** 2
NREF = 1.41


def arch():
    try:
        import threadpoolctl
        info = threadpoolctl.threadpool_info()
        for d in info:
            if d.get("internal_api") == "openblas":
                return str(d.get("architecture"))
        return "|".join(str(d.get("internal_api")) for d in info) or "none"
    except Exception as exc:                       # noqa: BLE001
        return f"unknown({type(exc).__name__})"


# --------------------------------------------------------------------------- #
def gamma_of(m, idx=2):
    from lumenairy.elements.bor.zcascade import layer_modes
    L = layer_modes(m, RBIG, NFD,
                    lambda r: np.full_like(r, EPS, dtype=complex), 2.0,
                    staggered=True)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def class_a(deltas):
    from lumenairy import BORStack
    from lumenairy.elements.bor.zcascade import layer_modes
    rows = []
    for m in (0, 1, 2):
        g = gamma_of(m)
        for dl in deltas:
            k0 = g / (NREF * np.sqrt(1.0 - dl))
            L = layer_modes(m, RBIG, NFD,
                            lambda r: np.full_like(r, EPS, dtype=complex),
                            float(k0), staggered=True)
            q, flux, rel, rho = mode_table(L, sem=False)
            qn = q / k0
            phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
            if not phys.any():
                continue
            j = np.where(phys)[0][int(np.argmin(np.abs(qn.real)[phys]))]
            called = bool(rho[j] < 1e-9)
            # the CANDIDATE band: spectrum-scaled, the _sqrt_decay shape
            qtop = float(np.max(np.abs(q)))
            cand = bool(abs(q[j].imag) <= 1e-8 * max(qtop, k0))
            row = dict(m=m, delta=float(dl), k0=float(k0),
                       qn=float(abs(qn[j].real)), rho=float(rho[j]),
                       shipped_called_prop=called, cand_called_prop=cand,
                       flux_fwd=bool(flux[j] >= 0.0),
                       im_fwd=bool(q[j].imag > 0.0),
                       backward_flux_shipped=bool(
                           (not called) and flux[j] < 0.0))
            for tag, rule in (("shipped", "shipped"), ("cand", "candidate")):
                with patched(rule):
                    s = BORStack(RBIG, m, n_substrate=NREF,
                                 n_superstrate=NREF, N=NFD, basis="fd")
                    s.add_layer(0.4, eps=EPS)
                    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
                    s.add_layer(0.4, eps=EPS)
                    s.set_source(k0=float(k0))
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        res = s.solve()
                row[f"{tag}_n_orders"] = int(np.size(res["R"]))
                row[f"{tag}_closure"] = closure(res)
                row[f"{tag}_sumR"] = float(np.sum(res["R"]))
            rows.append(row)
    return rows


def class_b():
    from lumenairy.elements.bor.bor_solve import build_layer, solve
    k0 = 2.0
    lam = 2 * np.pi / k0
    rows = []

    def uni(r):
        return np.full_like(np.asarray(r, float), EPS, dtype=complex)

    for rl in (2.0, 8.0, 14.0):
        R = rl * lam

        def ring(r, R=R):
            return np.where((np.asarray(r) % (R / 8.0)) < 0.5 * (R / 8.0),
                            2.45 ** 2, EPS).astype(complex)
        for basis in ("staggered", "nodal"):
            recs = []
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    Ls = [build_layer(1, R, 140, uni, k0, basis=basis),
                          build_layer(1, R, 140, uni, k0, basis=basis,
                                      thickness=0.4),
                          build_layer(1, R, 140, ring, k0, basis=basis,
                                      thickness=0.5),
                          build_layer(1, R, 140, uni, k0, basis=basis,
                                      thickness=0.4),
                          build_layer(1, R, 140, uni, k0, basis=basis)]
                    with inv_census(recs):
                        res = solve(Ls, k0)
                e = np.asarray(res["energy"], float)
                apb = [x for x in recs
                       if "interface_smatrix" in x["site"] and x["kind"] == "inv"]
                rc = [x["rcond"] for x in apb
                      if x["rcond"] is not None and np.isfinite(x["rcond"])]
                rs = [x["resid"] for x in apb
                      if x["resid"] is not None and np.isfinite(x["resid"])]
                rows.append(dict(rbig_lambda=rl, basis=basis,
                                 n_orders=int(e.size),
                                 max_energy=float(np.max(e)) if e.size else None,
                                 apb_rcond_min=float(min(rc)) if rc else None,
                                 apb_resid_max=float(max(rs)) if rs else None,
                                 warned=bool(w), error=None))
            except Exception as exc:               # noqa: BLE001
                rows.append(dict(rbig_lambda=rl, basis=basis, n_orders=0,
                                 max_energy=None, apb_rcond_min=None,
                                 apb_resid_max=None, warned=False,
                                 error=f"{type(exc).__name__}: {exc}"[:150]))
    return rows


def class_c(fracs):
    rows = []
    for degree in (8, 12):
        for separated in (False, True):
            ref = solve_one(0.0, basis="sem", degree=degree,
                            separated=separated)
            for f in fracs:
                r = solve_one(f * RBIG, basis="sem", degree=degree,
                              separated=separated)
                dR, resid, nmatch = q_matched(ref, r, "R")
                rows.append(dict(
                    degree=degree, separated=separated, frac=f,
                    delta=float(f * RBIG),
                    min_element=r["min_element"],
                    min_elem_frac_Rbig=(r["min_element"] / RBIG
                                        if r["min_element"] else None),
                    qmax_over_ceiling=r["qmax_over_ceiling"],
                    closure=r["closure"], superunity=r["superunity"],
                    n_orders=r["n_orders"], dR_matched=dR, q_resid=resid,
                    n_matched=nmatch, n_warn=len(r["warnings"])))
    return rows


def main():
    a = arch()
    tag = os.environ.get("PROBE_TAG", "win")
    req = os.environ.get("OPENBLAS_CORETYPE", "(default)")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    print(f"REQUESTED kernel {req!r} -> OBTAINED architecture {a!r}   "
          f"threads={thr}")
    deltas = [10.0 ** (-e / 2.0) for e in (8, 10, 11, 12, 14, 16, 18, 20)]
    fracs = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
    payload = dict(requested_kernel=req, architecture=a, threads=thr,
                   A=class_a(deltas), B=class_b(), C=class_c(fracs))
    dump(f"k1_kernel_{tag}_{a}_{req}_t{thr}.json", payload)

    print("\n== A: near-cutoff orientation DECISIONS ==")
    print("  m  delta      qn         rho         shipped  cand   flux  im   "
          "backwd  ship n/clo            cand n/clo")
    for r in payload["A"]:
        print(f"  {r['m']}  {r['delta']:.2e}  {r['qn']:.3e}  {r['rho']:.3e}  "
              f"{'PROP' if r['shipped_called_prop'] else 'EVAN':>5}  "
              f"{'PROP' if r['cand_called_prop'] else 'EVAN':>5}  "
              f"{str(r['flux_fwd'])[0]}     {str(r['im_fwd'])[0]}    "
              f"{str(r['backward_flux_shipped'])[0]}      "
              f"{r['shipped_n_orders']}/{r['shipped_closure']:.3e}      "
              f"{r['cand_n_orders']}/{r['cand_closure']:.3e}")
    print("\n== B: inv(a+b) populations ==")
    for r in payload["B"]:
        print(f"  Rbig={r['rbig_lambda']:5.1f}lam {r['basis']:<10} "
              f"n_ord={r['n_orders']:4d} max(R+T)={r['max_energy']} "
              f"rcond={r['apb_rcond_min']} resid={r['apb_resid_max']} "
              f"warned={r['warned']}")
    print("\n== C: sliver ladder DECISION quantities ==")
    print("  deg sep  frac     min/Rbig    |q|/ceil    closure     R+T-1"
          "        dR(q-matched)")
    for r in payload["C"]:
        print(f"  {r['degree']:>3} {str(r['separated'])[0]}   "
              f"{r['frac']:.0e}  {r['min_elem_frac_Rbig']:.3e}  "
              f"{r['qmax_over_ceiling']:.3e}  {r['closure']:.3e}  "
              f"{(r['superunity'] if r['superunity'] is not None else float('nan')):+.3e}  "
              f"{r['dR_matched']:.6e}")


if __name__ == "__main__":
    main()
