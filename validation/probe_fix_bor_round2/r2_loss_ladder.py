"""ROUND 2, D1 -- the loss ladder that walks through the shipped screen.

``bor_solve._stack_is_provably_passive`` (5.45.1 as built) disarms the WHOLE
nodal passivity screen unless every layer's ``max|Im eps| / max|Re eps|`` is
<= 1e-12, on the reasoning "on a lossy stack there is no theorem to violate".
That reasoning covers the BELOW-unity direction only.  ``R + T <= 1`` is a
theorem on EVERY passive stack (``R + T + A = 1`` with ``A >= 0``), so the
super-unity half of the screen is valid for absorbing media too.

Two ladders, both run with the guard ARMED so the row records the DECISION:

  A. the refused 1.02882 stack with a relative loss ``Im(eps)/Re(eps)`` from
     1e-14 to 1e-1 on the ring's high region.  Super-unity persists across the
     whole ladder; the screen must therefore keep refusing across the whole
     ladder.
  B. HEALTHY lossy stacks -- uniform half-spaces on a small cell, the family
     the 5.45.1 census calls accurate -- over the same loss ladder.  These must
     NEVER be refused: their ``R + T`` is legitimately below unity.

GAIN (``Im eps < 0``) is not passive and stays OUTSIDE the screen: measured as
a third ladder, which must return on every rung.

Run:  python validation/probe_fix_bor_round2/r2_loss_ladder.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import banner, dump, nodal_stack, solve_disarmed, uniform_stack  # noqa: E402

LADDER = [0.0, 1e-14, 1e-13, 1e-12, 3e-12, 1e-11, 1e-9, 1e-8, 1e-6, 1e-4,
          1e-3, 1e-2, 1e-1]


def _armed(layers, k0):
    """Run with the guard ARMED and report the DECISION, not the number."""
    from lumenairy.elements.bor.bor_solve import BORNodalPassivityError, solve
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            res = solve(layers, k0)
        except BORNodalPassivityError as exc:
            return dict(decision="refused", message=str(exc)[:160])
    R, T = np.asarray(res["R"]), np.asarray(res["T"])
    E = R + T
    n_warn = len([x for x in w if "passiv" in str(x.message).lower()
                  or "R + T" in str(x.message)])
    return dict(decision="warned" if n_warn else "returned",
                n_warnings=n_warn,
                excess=float(np.max(E)) - 1.0 if E.size else None,
                deficit=float(np.min(E)) - 1.0 if E.size else None,
                n_channels=int(R.size))


def ladder_broken(rec):
    """A. the refused stack, plus its staggered twin as the physical truth."""
    k0, out = 2.0, []
    for rel in LADDER:
        e_hi = complex(6.0, 6.0 * rel)
        row = dict(rel_im=rel)
        row.update(_armed(nodal_stack(e_hi), k0))
        st = solve_disarmed(nodal_stack(e_hi, basis="staggered"), k0)
        Es = np.asarray(st["R"]) + np.asarray(st["T"])
        row["staggered_max"] = float(np.max(Es))
        row["staggered_min"] = float(np.min(Es))
        # the number the NODAL cascade actually returns, guard off
        nd = solve_disarmed(nodal_stack(e_hi), k0)
        En = np.asarray(nd["R"]) + np.asarray(nd["T"])
        row["nodal_excess_disarmed"] = float(np.max(En)) - 1.0
        out.append(row)
        print("  A rel=%9.2e  %-9s  nodal_exc=%+.6e  stag=[%.12f, %.12f]"
              % (rel, row["decision"], row["nodal_excess_disarmed"],
                 row["staggered_min"], row["staggered_max"]))
    return out


def ladder_healthy(rec):
    """B. healthy lossy stacks: uniform half-spaces on a small cell."""
    k0, out = 2.0, []
    for m in (0, 1, 2):
        for rel in LADDER:
            Rbig = 1.0 * 2.0 * np.pi / k0
            e_mid = complex(4.0, 4.0 * rel)
            row = dict(m=m, rel_im=rel)
            row.update(_armed(uniform_stack(Rbig, 200, k0, m, e_mid=e_mid), k0))
            out.append(row)
            print("  B m=%d rel=%9.2e  %-9s  exc=%s def=%s"
                  % (m, rel, row["decision"],
                     ("%+.3e" % row["excess"]) if row.get("excess") is not None
                     else "-",
                     ("%+.3e" % row["deficit"]) if row.get("deficit") is not None
                     else "-"))
    return out


def ladder_gain(rec):
    """C. GAIN is not passive: the screen must stay outside it on every rung."""
    k0, out = 2.0, []
    for rel in (1e-12, 3e-12, 1e-8, 1e-4, 1e-2):
        e_hi = complex(6.0, -6.0 * rel)
        row = dict(rel_im=-rel)
        row.update(_armed(nodal_stack(e_hi), k0))
        out.append(row)
        print("  C rel=%9.2e  %-9s  exc=%s"
              % (-rel, row["decision"],
                 ("%+.3e" % row["excess"]) if row.get("excess") is not None
                 else "-"))
    return out


def main():
    rec = banner("r2_loss_ladder")
    fast = "--fast" in sys.argv          # A and C only: the DECISION ladders
    print("-- A: the refused 1.02882 stack under a loss ladder --")
    a = ladder_broken(rec)
    if fast:
        b = []
        print("-- B: SKIPPED (--fast) --")
    else:
        print("-- B: healthy lossy stacks (must never be refused) --")
        b = ladder_healthy(rec)
    print("-- C: GAIN (not passive; must stay outside the screen) --")
    c = ladder_gain(rec)
    print("\n--- summary ---")
    print("  A refused on %d of %d rungs" % (
        sum(1 for r in a if r["decision"] == "refused"), len(a)))
    print("  A super-unity persists: nodal excess %.6e .. %.6e" % (
        min(r["nodal_excess_disarmed"] for r in a),
        max(r["nodal_excess_disarmed"] for r in a)))
    print("  B refused on %d of %d rungs (must be 0)" % (
        sum(1 for r in b if r["decision"] == "refused"), len(b)))
    print("  B warned  on %d of %d rungs" % (
        sum(1 for r in b if r["decision"] == "warned"), len(b)))
    print("  C refused on %d of %d rungs" % (
        sum(1 for r in c if r["decision"] == "refused"), len(c)))
    dump("r2_loss_ladder_fast" if fast else "r2_loss_ladder",
         dict(broken=a, healthy=b, gain=c, fast=fast), rec)


if __name__ == "__main__":
    main()
