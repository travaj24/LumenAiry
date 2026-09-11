"""ROUND 2 restatement -- what ORDINARY geometry the SEM warn edge actually
reaches, with the graded hp family the 5.45.1 census did not sweep.

``_BOR_SLIVER_BAND_FRAC = 1e-4`` was fixed from a census whose binding ordinary
geometry was a taper staircase at 256 slices (``w_min_union_frac`` = 9.766e-04,
quoted as a 9.77x margin).  The verification found a family that census did not
contain: hp refinement with GRADING on.  ``BORStack(elements_per_segment=k,
grade=True)`` splits every segment interval into ``k`` Chebyshev-Lobatto graded
sub-elements, which puts a NARROW sub-cell at each end of every interval -- and
those ends are walls of DIFFERENT layers, so the ``+-1`` enrichment window
attributes them to the union and they are CROSS-LAYER cells on geometry nobody
would call pathological.

This probe measures ``w_min_union_frac`` and ``q_excess`` over that family on an
ordinary two-layer ring stack, at ``elements_per_segment`` 1..32 with grading on
and off, so the warn edge's ORDINARY margin can be restated from a census that
contains it.

It also measures what the ``_BOR_Q_EXCESS`` conjunct can and cannot do on that
family: the refusal is ``w_min_union_frac < 1e-6 AND q_excess > 1e4``, so any
family whose ``w_min_union_frac`` never reaches 1e-6 cannot be refused at ANY
``q_excess`` -- which is the verification's §5.3 point, restated as a
measurement rather than as an argument.

Run:  python validation/probe_fix_bor_round2/r7_sem_hp_census.py
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import banner, dump  # noqa: E402


def _stack(eps_seg, grade, degree=8, Rbig=24.0, k0=2.0, m=1, N=120,
           n_rings=2):
    from lumenairy.elements.bor.bor_stack import BORStack
    s = BORStack(Rbig=Rbig, m=m, N=N, n_superstrate=1.0, n_substrate=1.0,
                 basis="sem", degree=degree,
                 elements_per_segment=eps_seg, grade=grade)
    # an ORDINARY two-layer ring pair: real features, no coincidences, no
    # sliver -- the kind of geometry the contract must never speak about
    s.add_layer(0.5, segments=[(6.0, 4.0), (12.0, 1.0), (Rbig, 2.0)])
    s.add_layer(0.5, segments=[(7.0, 2.0), (13.0, 1.0), (Rbig, 4.0)])
    s.set_source(k0=k0)
    return s


def row(eps_seg, grade, degree):
    """The stack's OWN mesh report, read back from ``_sem_mesh_report`` -- the
    same records the guard reads, so the census is a statement about the
    shipped behaviour and not about a re-implementation."""
    from lumenairy.elements.bor import _sem_contract as C
    prev = C.BOR_SEM_MESH_GUARD
    C.BOR_SEM_MESH_GUARD = False
    try:
        st = _stack(eps_seg, grade, degree=degree)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    except Exception as exc:                       # pragma: no cover - probe
        return dict(eps_seg=eps_seg, grade=grade, degree=degree,
                    error="%s: %s" % (type(exc).__name__, exc))
    finally:
        C.BOR_SEM_MESH_GUARD = prev
    msgs = [r for r in recs if C.verdict(r) in ("warn_manufactured",
                                                "warn_own")]
    if not recs:
        return dict(eps_seg=eps_seg, grade=grade, degree=degree,
                    error="no mesh records")
    fu = min(r["w_min_union_frac"] for r in recs)
    fo = min(r.get("w_min_own_frac", float("inf")) for r in recs)
    fa = min(r["w_min_frac"] for r in recs)
    qx = max(r["q_excess"] for r in recs if np.isfinite(r["q_excess"]))
    verds = sorted({C.verdict(r) for r in recs})
    return dict(eps_seg=eps_seg, grade=grade, degree=degree,
                w_min_union_frac=float(fu), w_min_own_frac=float(fo),
                w_min_frac=float(fa), q_excess=float(qx),
                verdicts=verds, n_warnings=len(msgs), n_layers=len(recs))


def main():
    rec = banner("r7_sem_hp_census")
    from lumenairy.elements.bor import _sem_contract as C
    sys.stdout.flush()
    print("  band edge  _BOR_SLIVER_BAND_FRAC = %.0e" % (C._BOR_SLIVER_BAND_FRAC,))
    print("  refusal    _BOR_MIN_ELEM_FRAC    = %.0e" % (C._BOR_MIN_ELEM_FRAC,))
    print("  spectral   _BOR_Q_EXCESS         = %.0e" % (C._BOR_Q_EXCESS,))
    rows = []
    # degree is swept where it is affordable; the geometric quantity the warn
    # edge reads (``w_min_union_frac``) is degree-INDEPENDENT, and the deep
    # hp arms at degree 12 x eps_seg 32 build a mesh whose dense eig runs for
    # minutes -- which is the run the verification recorded as timed out.
    plan = [(6, g, e) for g in (False, True)
            for e in (1, 2, 4, 8, 16, 32)]
    plan += [(8, g, e) for g in (False, True) for e in (1, 2, 4, 8, 16)]
    plan += [(12, g, e) for g in (False, True) for e in (1, 2, 4, 8)]
    for (degree, grade, eps_seg) in plan:
        if True:
            if True:
                r = row(eps_seg, grade, degree)
                rows.append(r)
                sys.stdout.flush()
                if "error" in r:
                    print("  deg=%-2d grade=%-5s eps=%2d  ERROR %s"
                          % (degree, grade, eps_seg, r["error"][:60]))
                    continue
                marg = (C._BOR_SLIVER_BAND_FRAC / r["w_min_union_frac"]
                        if np.isfinite(r["w_min_union_frac"]) else 0.0)
                print("  deg=%-2d grade=%-5s eps=%2d  union=%10.4e "
                      "(edge/union=%7.3g)  own=%10.4e  q_excess=%9.4g  %s "
                      "warns=%d"
                      % (degree, grade, eps_seg, r["w_min_union_frac"],
                         1.0 / marg if marg else float("inf"),
                         r["w_min_own_frac"], r["q_excess"],
                         "/".join(r["verdicts"]), r["n_warnings"]))
    ok = [r for r in rows if "error" not in r]
    fin = [r for r in ok if np.isfinite(r["w_min_union_frac"])]
    print("\n--- summary ---")
    print("  rows %d (%d with a CROSS-LAYER cell at all)" % (len(ok), len(fin)))
    if fin:
        worst = min(fin, key=lambda r: r["w_min_union_frac"])
        margin = C._BOR_SLIVER_BAND_FRAC / worst["w_min_union_frac"]
        print("  narrowest ORDINARY cross-layer cell: %.6e of Rbig "
              "(deg=%d grade=%s eps_seg=%d)"
              % (worst["w_min_union_frac"], worst["degree"], worst["grade"],
                 worst["eps_seg"]))
        print("  margin to the warn edge %.0e : %.4gx (%.3f decades)"
              % (C._BOR_SLIVER_BAND_FRAC, 1.0 / margin
                 if margin else float("inf"),
                 np.log10(1.0 / margin) if margin else float("nan")))
        print("  distance to the REFUSAL's geometric conjunct %.0e : %.4gx"
              % (C._BOR_MIN_ELEM_FRAC,
                 worst["w_min_union_frac"] / C._BOR_MIN_ELEM_FRAC))
    warned = [r for r in ok if r["n_warnings"]]
    print("  rows that WARNED: %d" % (len(warned),))
    for r in warned:
        print("    deg=%d grade=%s eps_seg=%d -> %s (%d messages)"
              % (r["degree"], r["grade"], r["eps_seg"],
                 "/".join(r["verdicts"]), r["n_warnings"]))
    refused = [r for r in ok if "refuse" in r["verdicts"]]
    print("  rows that would be REFUSED: %d (must be 0 on ordinary geometry)"
          % (len(refused),))
    print("  worst ORDINARY q_excess over this family: %.6g"
          % (max(r["q_excess"] for r in ok),))
    dump("r7_sem_hp_census", dict(rows=rows), rec)


if __name__ == "__main__":
    main()
