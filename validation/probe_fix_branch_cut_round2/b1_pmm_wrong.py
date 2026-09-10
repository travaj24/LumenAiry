"""B1 -- CAN THE PMM COPIES OF ``_sqrt_decay`` PRODUCE A WRONG ANSWER?

The round-1 verification (D1) established that five PMM modules keep their own
``_sqrt_decay`` with the EXACT-ZERO pin ``r.real == 0``, that the ``pmm/twod.py``
copy is handed LAYER eigenvalues, and that 5-9 propagating modes per
``pmm_efficiency_2d_cell`` solve come back on the INCOMING root as a result.
What it could NOT establish is whether that mis-rooting can make a PMM answer
WRONG, as opposed to merely build-dependent: it tried nine fixtures and the
largest motion from correcting the root was 8.660e-15.

This probe asks that question with the two partners the round-1 work identified
as the ones that turn a mis-rooted mode into a SINGULAR interface:

  (i)  a REGION partner -- the patterned layer's host permittivity equals
       ``n_substrate**2`` exactly, so the layer carries the substrate's own
       modes;
  (ii) a UNIFORM SPACER partner -- a uniform layer of the same permittivity
       sitting next to the patterned one.  Section 6 of the verification showed
       a uniform layer is a partner exactly as a region is (its modes are built
       in EXACT arithmetic by the same helper), and that the substrate index is
       then irrelevant;
  (iii) both at once.

and with the fixture shape that removes the confound the verification named:
the hybrid PMM has a FOURIER TRUNCATION FLOOR, so a strongly modulated cell's
own truncation error (~1e-3 at degree 5-7) masks anything smaller.  A WEAKLY
MODULATED cell -- pillar permittivity a relative 1e-6 from the host -- is
converged to the arithmetic floor at any truncation, so its lossless closure
defect is a clean instrument.  That is the same shape that took the RCWA scalar
2-D path's ``cond(a+b)`` to 2.5e+08 in the verification (its fixture ``c5``).

ARMS.  The NumPy copies are A/B-ed WITHIN ONE INTERPRETER (``PreSqrtDecayPMM``)
so a thread ladder can be run on both arms at one thread setting, and the
as-installed numbers are recorded too so the cross-tree (pre-edit / post-edit)
comparison covers the JAX twins, whose ``_sqrt_decay`` is a nested closure that
cannot be monkeypatched.

ORACLE.  ``sum R + sum T - 1`` (or ``- 2``) on a provably lossless cell, which
needs no reference solve; plus, where the geometry allows, an INDEPENDENT
REFERENCE from the RCWA path on the same pixel cell (round 1 fixed it, so on
this branch it is a different method with a correct root).

Usage:  OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b1_pmm_wrong.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
NS = 1.5                 # NS**2 = 2.25 -- the coincidence value
NS_OFF = 1.63            # nothing coincides
HOST = 2.25
WEAK = HOST * (1.0 + 1e-6)      # weakly modulated pillar
STRONG = 6.0


def _cell(pillar, host=HOST, S=32):
    return F.pillar_cell(S=S, host=host, pillar=pillar)


# --------------------------------------------------------------- the surfaces
def surfaces():
    """(name, kind, fn, class) for every surface measured.

    ``class`` records the coincidence partner, which is what the whole probe is
    about: ``region`` / ``spacer`` / ``both`` / ``none``.
    """
    from lumenairy.elements.pmm import (
        PMM2DStackHybrid,
        PMM2DStackPure,
        pmm_efficiency_2d_cell,
        pmm_efficiency_2d_staggered,
        pmm_jones_2d,
        pmm_jones_2d_staggered,
    )

    out = []

    def add(name, kind, fn, cls):
        out.append((name, kind, fn, cls))

    # ---- single-layer hybrid: the copy that the verification proved is live
    for tag, pillar in (("weak", WEAK), ("strong", STRONG)):
        for scls, nsub in (("region", NS), ("none", NS_OFF)):
            for sym in (False, True):
                add("cell_%s_%s_sym%s" % (tag, scls, sym), "eff",
                    (lambda p=pillar, n=nsub, s=sym: pmm_efficiency_2d_cell(
                        PX, PX, _cell(p), n, 1.0, D, WL, degree=7,
                        n_orders=5, symmetry=s)), scls)
        # oblique skips the fold entirely
        add("cell_%s_region_oblique" % tag, "eff",
            (lambda p=pillar: pmm_efficiency_2d_cell(
                PX, PX, _cell(p), NS, 1.0, D, WL, degree=7, n_orders=4,
                theta=0.3, symmetry=False)), "region")

    # ---- hybrid tensor entry (its LAYER goes through the FIXED rcwa function;
    #      only its REGION modes use the PMM copy -- recorded as the control)
    add("jones2d_region", "jones",
        (lambda: pmm_jones_2d(PX, PX, F.tensor_cell(), NS, 1.0, D, WL,
                              degree=7, n_orders=3)), "region")

    # ---- the hybrid STACK, with symmetry OFF so the layer eig runs in the
    #      PMM copy rather than in rcwa's even-sector cascade
    def hyb(spacer, nsub, pillar, sym=False, theta=0.0):
        st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                              degree=7, n_orders=4, symmetry=sym)
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        st.add_layer(D, eps_cell=_cell(pillar))
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        return st.set_source(WL, theta=theta).solve()

    for tag, pillar in (("weak", WEAK), ("strong", STRONG)):
        add("hyb_%s_region" % tag, "jones",
            lambda p=pillar: hyb(False, NS, p), "region")
        add("hyb_%s_spacer" % tag, "jones",
            lambda p=pillar: hyb(True, NS_OFF, p), "spacer")
        add("hyb_%s_both" % tag, "jones",
            lambda p=pillar: hyb(True, NS, p), "both")
        add("hyb_%s_none" % tag, "jones",
            lambda p=pillar: hyb(False, NS_OFF, p), "none")
    add("hyb_weak_both_sym", "jones",
        lambda: hyb(True, NS, WEAK, sym=True), "both")
    add("hyb_weak_both_oblique", "jones",
        lambda: hyb(True, NS, WEAK, theta=0.25), "both")

    # ---- PURE STAGGERED: the user's main 2-D engine.  Its forward branch is
    #      chosen by _forward_branch_flip (already a relative band) and its OOP
    #      path by _select_forward_flux, so these are CONTROLS: they must be
    #      bit-identical between the arms.
    def stagcell(pillar):
        return F.stag_cell(host=HOST, pillar=pillar)

    for tag, pillar in (("weak", HOST * (1.0 + 1e-6)), ("strong", STRONG)):
        add("stag_%s_region" % tag, "eff",
            (lambda p=pillar: pmm_efficiency_2d_staggered(
                PX, PX, stagcell(p), NS, 1.0, D, WL, degree=6, n_orders=4)),
            "region")
        add("stag_%s_none" % tag, "eff",
            (lambda p=pillar: pmm_efficiency_2d_staggered(
                PX, PX, stagcell(p), NS_OFF, 1.0, D, WL, degree=6,
                n_orders=4)), "none")
    add("stagjones_tensor_region", "jones",
        (lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(), NS, 1.0,
                                        D, WL, degree=6, n_orders=3)),
        "region")
    add("stagjones_oop_region", "jones",
        (lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(oop=0.3),
                                        NS, 1.0, D, WL, degree=6, n_orders=3)),
        "region")

    def pure(spacer, nsub, pillar):
        st = PMM2DStackPure(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                            degree=6, n_orders=4)
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        st.add_layer(D, eps_cell=stagcell(pillar))
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        return st.set_source(WL, theta=0.0).solve()

    add("pure_weak_both", "jones",
        lambda: pure(True, NS, HOST * (1.0 + 1e-6)), "both")
    add("pure_weak_region", "jones",
        lambda: pure(False, NS, HOST * (1.0 + 1e-6)), "region")
    add("pure_strong_both", "jones", lambda: pure(True, NS, STRONG), "both")
    add("pure_strong_none", "jones",
        lambda: pure(False, NS_OFF, STRONG), "none")
    return out


# ------------------------------------------------------------------- running
def run_one(fn, kind, spy_module="lumenairy.elements.pmm.twod"):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            with F.PMMEigSpy(spy_module) as spy:
                res = fn()
            summ = spy.summary()
        except Exception as exc:
            return dict(raised=type(exc).__name__, message=repr(exc)[:200])
    cl = F.closure_jones(res) if kind == "jones" else F.closure_eff(res)
    return dict(raised=None, closure=cl, rt=F.rt_vec(res).tolist(),
                spy=summ,
                warnings=sorted({type(x.message).__name__ for x in w}))


def reference(pillar, nsub, n_orders=9):
    """INDEPENDENT reference for the scalar hybrid surfaces: the RCWA path on
    the SAME pixel cell.  Round 1 pinned its root, so on this branch it is a
    different method carrying a correct one.  The comparison quantity is
    ``sum(R)`` (the sensitive one on these near-null cells)."""
    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R, T = rcwa_efficiency_2d(PX, PX, _cell(pillar), nsub, 1.0, D, WL,
                                      n_orders_x=n_orders, n_orders_y=n_orders,
                                      polarization="te")
    return dict(sumR=float(np.sum(R)), sumT=float(np.sum(T)),
                closure=float(np.sum(R) + np.sum(T) - 1.0))


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b1.json"
    rows = {}
    for name, kind, fn, cls in surfaces():
        post = run_one(fn, kind)
        with F.PreSqrtDecayPMM():
            pre = run_one(fn, kind)
        mv = (F.motion(pre["rt"], post["rt"])
              if pre.get("rt") is not None and post.get("rt") is not None
              else None)
        rows[name] = dict(cls=cls, kind=kind, installed=post, numpy_pre=pre,
                          numpy_ab_motion=mv)
        print("%-30s %-7s inst.closure %-12s pre.closure %-12s motion %-11s "
              "oncut %-4s incoming %-4s %s"
              % (name, cls,
                 "%+.4e" % post["closure"] if post["raised"] is None
                 else post["raised"],
                 "%+.4e" % pre["closure"] if pre["raised"] is None
                 else pre["raised"],
                 "%.3e" % mv if mv is not None else "-",
                 post.get("spy", {}).get("on_cut"),
                 post.get("spy", {}).get("incoming_after_exact_pin"),
                 ",".join(post.get("warnings", []) or [])))

    refs = {}
    for tag, pillar in (("weak", WEAK), ("strong", STRONG)):
        for scls, nsub in (("region", NS), ("none", NS_OFF)):
            try:
                refs["%s_%s" % (tag, scls)] = reference(pillar, nsub)
            except Exception as exc:                      # pragma: no cover
                refs["%s_%s" % (tag, scls)] = dict(error=repr(exc)[:200])
    print("RCWA reference (n_orders 9x9, TE):")
    for k, v in sorted(refs.items()):
        print("   %-16s %s" % (k, v))

    F.dump(out, dict(rows=rows, rcwa_reference=refs,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()
