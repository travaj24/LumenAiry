"""TASK C, second pass -- the two HOLES the nodal passivity screen leaves, and
whether the mildest REFUSED rows are refused correctly.

THE PHYSICS THE SCREEN RESTS ON.  For a stack of PASSIVE media (Im eps >= 0 in
the ``exp(-i omega t)`` convention this library uses) energy conservation reads

    R + T + A = 1,    A >= 0 the absorbed fraction,

so ``R + T <= 1`` on ANY passive stack, lossy or not.  Only a GAIN medium
(Im eps < 0) can break it.  The shipped screen instead disarms on ANY loss
(``_stack_is_provably_passive`` demands every layer's
``max|Im eps| / max|Re eps| <= 1e-12``), on the stated reasoning that "on a
lossy stack there is no theorem to violate".  That reasoning covers the
BELOW-unity direction, which is indeed legitimate under absorption; it does not
cover the ABOVE-unity direction, which is not.

HOLE 1 -- A NEGLIGIBLE LOSS DISARMS THE GUARD.  A ladder in the layer's
relative imaginary permittivity, with LOSSLESS half-spaces so R and T are
unambiguous.

HOLE 2 -- THE SCREEN IS ONE-SIDED.  On a stack the solver has already PROVEN
lossless, ``R + T = 1`` is an EQUALITY (a PEC-walled cell has no other exit),
so a deficit is as non-physical as an excess.  The screen tests only the
excess.

AND THE OTHER DIRECTION (bidirectional): the mildest rung the bar REFUSES is
1.0176e-02 on my population.  Is that answer actually wrong, or is the refusal
a false positive?  Same method as the flagged fixture: per-channel against the
converged staggered twin.

Usage:  python v5_nodal_holes.py <pre|post> [outdir]
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
import _vh  # noqa: E402


def _uni(val):
    return lambda r: np.full_like(r, val, dtype=complex)


def _ring(period, e_lo, e_hi, duty=0.5):
    def f(r):
        e = np.full_like(r, e_lo, dtype=complex)
        e[(r % period) < duty * period] = e_hi
        return e
    return f


def _build(basis, m, Rbig, N, k0, prof, thick, e_out):
    from lumenairy.elements.bor.bor_solve import build_layer
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return [build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis),
                build_layer(m, Rbig, N, prof, k0, thickness=thick,
                            basis=basis),
                build_layer(m, Rbig, N, _uni(e_out), k0, basis=basis)]


def _run(layers, k0, disarm=False):
    import lumenairy.elements.bor.bor_solve as bs
    had = hasattr(bs, "BOR_NODAL_PASSIVITY_GUARD")
    prev = getattr(bs, "BOR_NODAL_PASSIVITY_GUARD", None)
    if had and disarm:
        bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                r = bs.solve(layers, k0)
                R, T = np.asarray(r["R"]), np.asarray(r["T"])
                E = R + T
                return dict(raised=None, n_inc=int(R.size),
                            max_energy=float(np.max(E)) if E.size else None,
                            min_energy=float(np.min(E)) if E.size else None,
                            R=R.tolist(), T=T.tolist(),
                            qn=[complex(z).real
                                for z in np.asarray(r["q_inc"]) / k0],
                            n_warn=len(w),
                            warn=[str(x.message)[:150] for x in w])
            except BaseException as e:              # noqa: BLE001
                return dict(raised=type(e).__name__, msg=str(e)[:250],
                            n_warn=len(w))
    finally:
        if had:
            bs.BOR_NODAL_PASSIVITY_GUARD = prev


def hole1_lossy_disarm():
    """A LOSSY LAYER between LOSSLESS half-spaces.  ``R + T <= 1`` is still a
    theorem (absorption only removes energy), but any loss above 1e-12
    relative disarms the shipped screen."""
    rows = []
    k0, Rbig, N, m = 2.0, 4.0, 200, 1
    for rel in (0.0, 1e-14, 1e-13, 1e-12, 3e-12, 1e-11, 1e-10, 1e-9,
                1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        e_hi = complex(6.0, 6.0 * rel)
        prof = _ring(0.8, 2.0 + 0j, e_hi)
        rec = dict(rel_im=rel)
        rec["nodal_armed"] = _run(_build("nodal", m, Rbig, N, k0, prof, 0.5,
                                         2.0 + 0j), k0)
        rec["nodal_disarmed"] = _run(_build("nodal", m, Rbig, N, k0, prof,
                                            0.5, 2.0 + 0j), k0, disarm=True)
        rec["staggered"] = _run(_build("staggered", m, Rbig, N, k0, prof,
                                       0.5, 2.0 + 0j), k0)
        a, s = rec["nodal_armed"], rec["staggered"]
        print("  rel_im=%-8g nodal armed=%-26s maxE=%s   stag maxE=%s"
              % (rel, a.get("raised") or ("warn%d" % a["n_warn"]),
                 ("%.6g" % a["max_energy"]) if a.get("max_energy")
                 is not None else "-",
                 ("%.12g" % s["max_energy"]) if s.get("max_energy")
                 is not None else "-"), flush=True)
        rows.append(rec)
    return rows


def hole2_one_sided():
    """The DEFICIT direction on a provably lossless stack.  In a PEC-walled
    cell the only exits are the half-spaces' propagating channels, so
    ``R + T = 1`` is an equality; a deficit means channels were dropped."""
    rows = []
    k0 = 2.0
    for m in (0, 1, 2):
        for rl in (0.5, 0.75, 1.0, 1.5):
            Rbig = rl * 2.0 * np.pi / k0
            for N in (120, 200):
                for fam, prof in (("ring", _ring(0.8, 2.0, 6.0)),
                                  ("uniform", _uni(4.0))):
                    r = _run(_build("nodal", m, Rbig, N, k0, prof, 0.5, 2.0),
                             k0, disarm=True)
                    a = _run(_build("nodal", m, Rbig, N, k0, prof, 0.5, 2.0),
                             k0)
                    s = _run(_build("staggered", m, Rbig, N, k0, prof, 0.5,
                                    2.0), k0)
                    rows.append(dict(
                        m=m, Rbig_over_lambda=rl, N=N, family=fam,
                        nodal_min_energy=r.get("min_energy"),
                        nodal_max_energy=r.get("max_energy"),
                        nodal_n_inc=r.get("n_inc"),
                        armed_raised=a.get("raised"),
                        armed_nwarn=a.get("n_warn"),
                        stag_min_energy=s.get("min_energy"),
                        stag_max_energy=s.get("max_energy"),
                        stag_n_inc=s.get("n_inc")))
                    print("  m%d R/l=%-5g N=%-4d %-8s nodal E in [%s, %s] "
                          "n=%s  armed=%s   stag n=%s"
                          % (m, rl, N, fam,
                             ("%.6g" % r["min_energy"]) if r.get("min_energy")
                             is not None else "-",
                             ("%.6g" % r["max_energy"]) if r.get("max_energy")
                             is not None else "-",
                             r.get("n_inc"),
                             a.get("raised") or ("warn%d" % a["n_warn"]),
                             s.get("n_inc")), flush=True)
    return rows


def _match(qa, qb, rtol=3e-2):
    pairs, used = [], set()
    for i, va in enumerate(qa):
        d = np.abs(np.asarray(qb) - va) / max(abs(va), 1e-300)
        for j in np.argsort(d):
            if int(j) in used:
                continue
            if d[j] <= rtol:
                pairs.append((i, int(j), float(d[j])))
                used.add(int(j))
            break
    return pairs


def mildest_refused():
    """Is the mildest REFUSED rung actually wrong?  The four rows my
    population refuses with ``R + T - 1`` between 1.0e-02 and 1.7e-02, scored
    per-channel against their converged staggered twins."""
    rows = []
    k0 = 2.0
    for m, N, rl in ((2, 120, 8.0), (2, 200, 14.0), (2, 120, 14.0),
                     (2, 200, 8.0), (1, 200, 4.0), (1, 120, 4.0)):
        Rbig = rl * 2.0 * np.pi / k0
        nod = _run(_build("nodal", m, Rbig, N, k0, _uni(4.0), 0.4, 2.0), k0,
                   disarm=True)
        arm = _run(_build("nodal", m, Rbig, N, k0, _uni(4.0), 0.4, 2.0), k0)
        sta = _run(_build("staggered", m, Rbig, N, k0, _uni(4.0), 0.4, 2.0),
                   k0)
        rec = dict(m=m, N=N, Rbig_over_lambda=rl,
                   nodal_excess=(nod["max_energy"] - 1.0)
                   if nod.get("max_energy") is not None else None,
                   armed_raised=arm.get("raised"),
                   armed_nwarn=arm.get("n_warn"),
                   nodal_n=nod.get("n_inc"), stag_n=sta.get("n_inc"),
                   stag_excess=(sta["max_energy"] - 1.0)
                   if sta.get("max_energy") is not None else None)
        if nod.get("qn") and sta.get("qn"):
            pr = _match(nod["qn"], sta["qn"])
            dR = [abs(nod["R"][i] - sta["R"][j]) for i, j, _ in pr]
            relR = [abs(nod["R"][i] - sta["R"][j])
                    / max(sta["R"][j], 1e-12) for i, j, _ in pr]
            rec.update(matched=len(pr), unmatched_nodal=nod["n_inc"] - len(pr),
                       worst_dR=max(dR) if dR else None,
                       worst_rel_dR=max(relR) if relR else None,
                       median_rel_dR=float(np.median(relR)) if relR else None)
        rows.append(rec)
        print("  m%d N=%d R/l=%-5g excess=%-11.5g armed=%-26s nodal_n=%s "
              "stag_n=%s matched=%s worst_relR=%s"
              % (m, N, rl, rec["nodal_excess"] or float("nan"),
                 rec["armed_raised"] or ("warn%d" % rec["armed_nwarn"]),
                 rec["nodal_n"], rec["stag_n"], rec.get("matched"),
                 ("%.3g" % rec["worst_rel_dR"]) if rec.get("worst_rel_dR")
                 else None), flush=True)
    return rows


def warned_rows_accuracy():
    """And the rows the bar merely WARNS about (9.3e-05, 6.7e-04 on my
    population): is the answer there right?  If it is not, the bar is too
    high; if it is, the bar has 0.17 decades of honest room below it."""
    rows = []
    k0 = 2.0
    for m, N, rl in ((1, 200, 4.0), (1, 120, 4.0), (1, 200, 2.0),
                     (2, 120, 4.0)):
        Rbig = rl * 2.0 * np.pi / k0
        nod = _run(_build("nodal", m, Rbig, N, k0, _uni(4.0), 0.4, 2.0), k0,
                   disarm=True)
        sta = _run(_build("staggered", m, Rbig, N, k0, _uni(4.0), 0.4, 2.0),
                   k0)
        rec = dict(m=m, N=N, Rbig_over_lambda=rl,
                   nodal_excess=(nod["max_energy"] - 1.0)
                   if nod.get("max_energy") is not None else None,
                   nodal_n=nod.get("n_inc"), stag_n=sta.get("n_inc"))
        if nod.get("qn") and sta.get("qn"):
            pr = _match(nod["qn"], sta["qn"])
            relR = [abs(nod["R"][i] - sta["R"][j]) / max(sta["R"][j], 1e-12)
                    for i, j, _ in pr]
            rec.update(matched=len(pr),
                       worst_rel_dR=max(relR) if relR else None,
                       median_rel_dR=float(np.median(relR)) if relR else None)
        rows.append(rec)
        print("  WARN-ROW m%d N=%d R/l=%g excess=%.5g nodal_n=%s stag_n=%s "
              "matched=%s worst_relR=%s median_relR=%s"
              % (m, N, rl, rec["nodal_excess"] or float("nan"),
                 rec["nodal_n"], rec["stag_n"], rec.get("matched"),
                 ("%.3g" % rec["worst_rel_dR"]) if rec.get("worst_rel_dR")
                 else None,
                 ("%.3g" % rec["median_rel_dR"]) if rec.get("median_rel_dR")
                 else None), flush=True)
    return rows


def main():
    build = sys.argv[1]
    _vh.require_tree(build)
    a = _vh.arm()
    print("ARM", a, flush=True)
    res = dict(arm=a, build=build)
    with _vh.timed("hole1"):
        res["hole1_lossy_disarm"] = hole1_lossy_disarm()
    with _vh.timed("hole2"):
        res["hole2_one_sided"] = hole2_one_sided()
    with _vh.timed("mildest"):
        res["mildest_refused"] = mildest_refused()
    with _vh.timed("warned"):
        res["warned_rows"] = warned_rows_accuracy()
    o = sys.argv[2] if len(sys.argv) > 2 else "."
    _vh.dump("%s/v5_nodal_holes_%s_%s_%s_t%s.json"
             % (o, build, a["platform"], a["loaded_kernel"],
                a["blas_threads"]), res)


if __name__ == "__main__":
    main()
