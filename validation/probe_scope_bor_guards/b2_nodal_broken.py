"""CLASS B on BOR -- THE BROKEN POPULATION, and the two-sided gap it gives.

b1 censused 2031 explicit inverses of the PRODUCTION (staggered FD and SEM)
cascade over 132 fixtures built to contain every candidate failure family
(coincident uniform layers, a 1e-6 detune, exactly degenerate twin layers,
very thin layers, near-cutoff, high m, many rings, SEM degrees 6-16) and found
ONE population: equilibrated ``rcond`` from 1.99e-07 to 1.0, residual at most
3.41e-13.  A conjunction guard armed on that population would be dormant, and
a bar cannot be derived from one population.

THE OTHER POPULATION IS ALREADY IN THE TREE, AND IT IS STILL REACHABLE.
``docs/audits/AUDIT_BOR_PROPAGATING_CUTOFF_ENERGY_2026_07_13.md`` (follow-up
finding) diagnosed the legacy NODAL FD cascade: its divergence-violating
spurious-mode sea carries ZERO z-flux, so those modes' forward/backward
orientation is decided by the SIGN OF NOISE; adjacent layers sharing most of
their cross-section then carry near-identical spurious modes oriented
OPPOSITELY, which is exactly a null vector of the interface transmission block
``a + b`` -- measured ``cond(a + b) = 2.6e15`` with ``cond(W), cond(V) ~ 1e3``,
and cascade energy ``R + T`` up to ~1e29..1e32.

That was remediated by CHANGING THE DEFAULT (``build_layer`` now defaults to
the staggered basis) and adding a ``UserWarning`` past ~4 vacuum wavelengths.
``basis='nodal'`` is retained as a documented escape hatch, and ON IT THE
WRONG ANSWER IS STILL RETURNED -- 1e29 under a warning, not a refusal.  So
this probe measures that population at the SAME instrument b1 used, to see
whether the two separate two-sidedly at ``zcascade.interface_smatrix``'s
``inv(a + b)``: the bar for a ported ``_guarded_inverse``.
"""
from __future__ import annotations

import collections
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import dump, inv_census, pin_tree  # noqa: E402

print("TREE", pin_tree())

K0 = 2.0
LAM_VAC = 2 * np.pi / K0
E_LO = 1.41 ** 2
E_HI = 2.45 ** 2


def ring_profile(period, duty, e_hi, e_lo):
    def prof(r):
        return np.where((np.asarray(r) % period) < duty * period,
                        e_hi, e_lo).astype(complex)
    return prof


def run(rbig_lambda, basis, *, N=140, m=1):
    from lumenairy.elements.bor.bor_solve import build_layer, solve
    Rbig = rbig_lambda * LAM_VAC
    recs = []
    uni = (lambda r: np.full_like(np.asarray(r, float), E_LO, dtype=complex))
    ring = ring_profile(Rbig / 8.0, 0.5, E_HI, E_LO)
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            layers = [
                build_layer(m, Rbig, N, uni, K0, basis=basis),
                build_layer(m, Rbig, N, uni, K0, basis=basis, thickness=0.4),
                build_layer(m, Rbig, N, ring, K0, basis=basis, thickness=0.5),
                build_layer(m, Rbig, N, uni, K0, basis=basis, thickness=0.4),
                build_layer(m, Rbig, N, uni, K0, basis=basis),
            ]
            with inv_census(recs):
                res = solve(layers, K0)
        e = np.asarray(res["energy"], float)
        out = dict(rbig_lambda=float(rbig_lambda), basis=basis, m=m, N=N,
                   n_orders=int(e.size),
                   closure=float(np.max(np.abs(e - 1.0))) if e.size else None,
                   max_energy=float(np.max(e)) if e.size else None,
                   sumR=float(np.sum(res["R"])),
                   warned=[str(x.message)[:100] for x in w][:1],
                   error=None)
    except Exception as exc:                       # noqa: BLE001
        out = dict(rbig_lambda=float(rbig_lambda), basis=basis, m=m, N=N,
                   n_orders=0, closure=None, max_energy=None, sumR=None,
                   warned=[], error=f"{type(exc).__name__}: {exc}"[:200])
    out["inverses"] = recs
    return out


def main():
    tag = os.environ.get("PROBE_TAG", "win")
    thr = os.environ.get("OPENBLAS_NUM_THREADS", "?")
    rows = []
    for rl in (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0):
        for basis in ("staggered", "nodal"):
            r = run(rl, basis)
            rows.append(r)
            apb = [x for x in r["inverses"]
                   if "interface_smatrix" in x["site"] and x["kind"] == "inv"]
            rc = [x["rcond"] for x in apb
                  if x["rcond"] is not None and np.isfinite(x["rcond"])]
            rs = [x["resid"] for x in apb
                  if x["resid"] is not None and np.isfinite(x["resid"])]
            r["apb_rcond_min"] = float(min(rc)) if rc else None
            r["apb_resid_max"] = float(max(rs)) if rs else None
            print(f"  Rbig={rl:5.1f} lam  {basis:<10} n_ord={r['n_orders']:4d}"
                  f"  max(R+T)={r['max_energy']}"
                  f"  cond(a+b) rcond_min="
                  f"{r['apb_rcond_min'] if r['apb_rcond_min'] is not None else float('nan'):.3e}"
                  f"  resid_max="
                  f"{r['apb_resid_max'] if r['apb_resid_max'] is not None else float('nan'):.3e}"
                  f"  warn={len(r['warned'])}")
    # strip the per-call records down before dumping (keep the a+b rows)
    slim = []
    for r in rows:
        rr = dict(r)
        rr["inverses"] = [x for x in r["inverses"] if x["kind"] == "inv"]
        slim.append(rr)
    dump(f"b2_nodal_broken_{tag}_t{thr}.json", dict(threads=thr, rows=slim))

    print("\n== TWO-SIDED GAP AT zcascade.interface_smatrix inv(a + b) ==")
    pop = collections.defaultdict(list)
    for r in rows:
        for x in r["inverses"]:
            if "interface_smatrix" in x["site"] and x["kind"] == "inv":
                healthy = (r["basis"] == "staggered")
                pop["healthy" if healthy else "broken"].append(
                    (x["rcond"], x["resid"], r["rbig_lambda"],
                     r["max_energy"]))
    for k in ("healthy", "broken"):
        v = [x for x in pop[k] if x[0] is not None and np.isfinite(x[0])]
        if not v:
            continue
        rc = [x[0] for x in v]
        rs = [x[1] for x in v if x[1] is not None and np.isfinite(x[1])]
        print(f"  {k:<8} n={len(v):4d}  rcond {min(rc):.3e} .. {max(rc):.3e}"
              f"   resid {min(rs):.3e} .. {max(rs):.3e}")
    hv = [x[0] for x in pop["healthy"] if x[0] and np.isfinite(x[0])]
    bv = [x[0] for x in pop["broken"] if x[0] and np.isfinite(x[0])]
    if hv and bv:
        print(f"\n  healthy floor  {min(hv):.4e}")
        print(f"  broken ceiling {max(bv):.4e}")
        print(f"  broken worst   {min(bv):.4e}")
        print(f"  a bar anywhere in ({min(bv):.3e}, {min(hv):.3e}) separates "
              f"the worst broken row from every healthy row: "
              f"{np.log10(min(hv) / min(bv)):.2f} decades")
    hr = [x[1] for x in pop["healthy"] if x[1] and np.isfinite(x[1])]
    br = [x[1] for x in pop["broken"] if x[1] and np.isfinite(x[1])]
    if hr and br:
        print(f"  RESIDUAL: healthy max {max(hr):.4e}   broken max "
              f"{max(br):.4e}   (Cartesian _INV_RESID_REFUSE = 1e-8)")
    print("\n== THE WRONG ANSWER THE GUARD WOULD HAVE REFUSED ==")
    for r in rows:
        if r["basis"] == "nodal" and r["max_energy"] and r["max_energy"] > 1.1:
            print(f"  Rbig={r['rbig_lambda']:.1f} lam: max(R+T)="
                  f"{r['max_energy']:.6e}  rcond(a+b)_min="
                  f"{r['apb_rcond_min']:.3e}  resid={r['apb_resid_max']:.3e}"
                  f"  warned={bool(r['warned'])}")


if __name__ == "__main__":
    main()
