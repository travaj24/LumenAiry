"""ROUND 4, P4 -- the DECISION TABLE, one JSON per (build, BLAS kernel) arm.

This is the probe the cross-arm pin reads.  For a FIXED, deterministic row set
spanning every population the guard has to be right on, it records

  * what the library DECIDED -- ``refused`` / ``wall`` / ``truncation`` /
    ``clean`` -- end to end, through :meth:`PMMStack.solve`;
  * what the exact ``delta -> 0`` reference says about the answer that was
    returned, scored on BOTH polarizations (the statistic the arbiter's move
    uses) and on the campaign's pol-1 convention;
  * the arbiter's own quantities (``d0``, ``d12``, ``w_wide``), so the bars'
    margins can be re-derived per arm from the same file; and
  * the energy reading ``max R+T``, which is what rounds 1-3 triggered and
    attributed on and what the release CI matrix showed to be arm-dependent.

Run it once per arm:

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    OPENBLAS_CORETYPE=<kernel> python -u p4_decisions.py

and the file lands as ``p4_decisions_<build>_<requested-coretype>.json``.
``tests/unit/test_fix_pmmstack_sliver_round4.py`` reads every one of them and
asserts (a) no arm returns a WRONG answer and (b) the decision column is the
same on all of them.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import g_fixtures as G  # noqa: E402
import p3_bars as P  # noqa: E402
import v_fixtures as V  # noqa: E402

from lumenairy.elements.pmm import stack as ps  # noqa: E402

G.assert_tree()

#: The phrases the guard's three non-refusing outcomes are recognised by.
#: They are the message text, so a rename here is a deliberate contract change.
_WALL = "SENSITIVE TO WHERE THAT WALL IS PUT"
_TRUNC = "is NOT what moved this answer"
_WITHIN = "WITHIN-LAYER feature"
_NOT_PASSIVE = "WARNING (the answer is RETURNED, see below)"


def decide(build, delta, ref):
    """One row of the table: the library's decision, and the truth."""
    base = G.unguarded(build(delta))
    e1 = G.shared_move(base, ref, pol=1)
    eb = G.shared_move(base, ref)
    rec = dict(delta=float(delta), worst=base["worst"],
               err_pol1=e1, err_both=eb,
               eod_pol1=e1 / delta, eod_both=eb / delta,
               kind=G.classify(eb, delta), kind_pol1=G.classify(e1, delta))
    st = build(delta)
    hit = ps._cross_layer_sliver([L[1] for L in st._layers],
                                 float(st.min_feature) / float(st.period))
    rec["screened"] = hit is not None
    rec["passive"] = bool(ps._stack_provably_passive(st))
    if hit is not None:
        rec["w_wide"] = float(hit[3])
        rec["own_over_w"] = float(hit[4] / hit[0])
    # The arbiter's verdict and evidence are captured FROM the guarded solve
    # rather than by a second call: the arbitration is three solves, and this
    # probe runs on eight arms.
    seen = []
    real = ps._sliver_arbiter

    def _spy(*a, **k):
        got = real(*a, **k)
        seen.append(got)
        return got

    ps._sliver_arbiter = _spy
    try:
        refused, msg, out, warns = G.guarded(build(delta))
    finally:
        ps._sliver_arbiter = real
    # The arbiter's three probe solves each re-enter ``_warn_stack_energy``
    # and therefore call the arbiter again -- on a clone marked
    # ``_sliver_probe``, which screens to ``None`` immediately.  Those inner
    # calls RETURN FIRST, so the OUTER verdict is the LAST recorded, not the
    # first.
    got = seen[-1] if seen else None
    if got is None:
        rec["verdict"] = None
    else:
        rec["verdict"] = got[0]
        if got[1] is not None:
            ev = got[1]
            rec.update(d0=ev["move"], d12=ev["d12"],
                       d0_over_w=ev["d0_over_w"],
                       d0_over_d12=ev["d0_over_d12"],
                       su_snapped=ev["snapped_super_unity"],
                       drop=ev["drop"])
    rec["n_arbitrations"] = len(seen)
    # THE DECISION COLUMN.  Deliberately NOT the presence of the plain
    # super-unity warning: that warning fires on ``max R+T``, which is the
    # arm-dependent reading round 4 took out of the guard, so folding it in
    # would put the arm dependence straight back into the column the cross-arm
    # pin reads.  What is recorded is what the GUARD did.
    if refused:
        rec["decision"] = "refused"
    elif any(_WALL in w for w in warns):
        rec["decision"] = "wall"
    elif any(_NOT_PASSIVE in w for w in warns):
        rec["decision"] = "warn_not_passive"
    else:
        rec["decision"] = "returned"
    rec["truncation_note"] = any(_TRUNC in w for w in warns)
    rec["energy_warn"] = any("energy not conserved" in w for w in warns)
    rec["within_layer_warn"] = any(_WITHIN in w for w in warns)
    rec["n_warn"] = len(warns)
    if out is not None:
        # the answer the CALLER got, scored against the same reference
        rec["returned_err_both"] = G.shared_move(out, ref)
        rec["returned_eod_both"] = rec["returned_err_both"] / delta
        rec["returned_kind"] = G.classify(rec["returned_err_both"], delta)
    return rec


def _mk(fn, **kw):
    return lambda d: fn(d, **kw)


def cases():
    """``{name: (builder, [deltas])}`` -- the decision set.

    Deterministic and modest (every row costs one reference solve, one
    unguarded solve and, where the screen fires, the arbiter's three), because
    it is run on eight arms."""
    out = {}
    for name, cfg in G.FIXTURES.items():
        for deg in (12, 14, 20):
            out["ladder:%s:d%d" % (name, deg)] = (
                (lambda d, _g=deg, _c=cfg: G.wbuild(d, _g, **_c)),
                [1e-3, 1e-4, 3e-5, 1e-5, 3e-6])
    for nsub, tag in ((complex(1.45, 0.08), "lo"), (complex(3.4, 1.7), "hi")):
        for deg in (6, 10):
            out["census:%s:d%d" % (tag, deg)] = (
                lambda d, _s=nsub, _d=deg: G.cbuild(d, _d, _s, 2.4, 1.22, 4,
                                                    10.5),
                [3e-3, 1e-3, 3e-4])
    for name in P.D5:
        fn, kw = P.D5[name]
        out["d5:" + name] = (_mk(fn, **kw), [1e-4, 1e-5, 1e-6])
    out["v4"] = (_mk(P._v4, degree=4), [1.6622e-05, 1.2690e-05, 7.3955e-06])
    out["steep:gmr2:d16"] = (lambda d: P._gmr2(d, 1.749000e-6, 16),
                             [1e-3, 1e-4, 1e-5])
    cfg = dict(G.FIXTURES["O11"])
    for nl in (6, 16):
        out["steep:taper%d" % nl] = (
            (lambda d, _n=nl, _c=cfg: G.wbuild(d, 8, nl=_n, **_c)),
            [3e-3, 3e-4, 3e-5])
    out["tensor:in_plane"] = (V.vtensor, [3e-4, 3e-5, 3e-6])
    out["tensor:oop"] = (V.voop, [3e-4, 3e-5, 3e-6])
    return out


def main():
    """Run the decision table.  Optional argv: case-name PREFIXES to keep.

    The full 123-row table is what every pinned arm runs.  The UNPINNED arm
    needs the filter: with all three thread variables removed, OpenBLAS takes
    all 24 hardware threads of this box and these small spectral-element
    eigenproblems spend their time in thread launch -- measured >30x slower
    than the same run pinned to one thread, so the full table does not
    finish.  CI's unpinned lane is a FOUR-core runner, i.e. the ``t4`` arm,
    which is measured in full; the reduced unpinned arm exists to show that
    REMOVING the variables entirely decides the same way, and it is run on
    the ladder groups because those are the rows the collapse band lives on.
    """
    rows = []
    todo = cases()
    keep = [a for a in sys.argv[1:] if not a.startswith("-")]
    if keep:
        todo = {k: v for k, v in todo.items()
                if any(k.startswith(pre) for pre in keep)}
        if not todo:
            raise SystemExit("no case matches %r" % (keep,))
    for name in sorted(todo):
        build, deltas = todo[name]
        ref = G.unguarded(build(0.0))
        for d in deltas:
            r = decide(build, d, ref)
            r["case"] = name
            rows.append(r)
        print(name, "done", flush=True)
    G.dump(dict(rows=rows, n=len(rows), subset=(keep or None)),
           "p4_decisions")


if __name__ == "__main__":
    main()
