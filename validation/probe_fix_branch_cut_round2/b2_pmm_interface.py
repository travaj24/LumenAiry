"""B2 -- the SHARP instrument for the PMM wrong-answer question.

Round 1 localised the RCWA defect not by an accuracy reading but by the
CONDITIONING of the interface mode-match: joining media ``a -> b`` forms
``a = Wb^-1 Wa``, ``b = Vb^-1 Va`` and inverts ``a + b`` EXPLICITLY, because
``S12 = 2 (a+b)^-1``.  ``a + b`` is singular exactly when a FORWARD mode of
``a`` reproduces a BACKWARD mode of ``b``, and a mis-rooted propagating mode is
precisely such a reproduction.  On the RCWA anisotropic coincidence cell
``cond(a+b)`` read 1.97e+15 at the layer -> substrate interface and 5.63e+03
after the fix -- eleven decades.

So: "can a PMM answer be WRONG?" is answered by asking whether the PMM's own
``a + b`` is ever SINGULAR at a coincidence.  A closure reading cannot answer
it (the hybrid's Fourier floor is ~1e-3 at a realistic contrast and would mask
anything smaller), but the conditioning can: it is a property of the operator,
not of the truncation error.

Both arms in ONE interpreter.  Every PMM binding of ``_interface_smatrix`` is
patched, so the single-cell path, the even-sector fold and the stack cascade
are all seen; patching only ``_core`` would silently see nothing on the 2-D
hybrid path (the round-1 verification records exactly that trap).

Usage:  OPENBLAS_NUM_THREADS=<n> PYTHONPATH=. python b2_pmm_interface.py <out.json>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b_fixtures as F  # noqa: E402

WL, PX, D = F.WL, F.PX, F.DEPTH
NS, NS_OFF, HOST = 1.5, 1.63, 2.25
WEAK = HOST * (1.0 + 1e-6)
STRONG = 6.0


class PMMInterfaceSpy:
    """``cond(a+b)``, the tiny-singular-value count and the null direction's
    inverse participation ratio at every PMM interface mode-match."""

    _MODULES = ("lumenairy.elements.pmm._core",
                "lumenairy.elements.pmm.twod",
                "lumenairy.elements.pmm.stack2d",
                "lumenairy.elements.pmm.stack2d_pure",
                "lumenairy.elements.pmm.twod_staggered",
                "lumenairy.elements.pmm.conical")

    def __init__(self):
        self.rows = []
        self._saved = []

    def __enter__(self):
        import importlib

        from lumenairy.elements.rcwa import _core as rc
        orig = rc._interface_smatrix
        rows = self.rows

        def wrapped(Wa, Va, Wb, Vb):
            try:
                a = np.linalg.solve(np.asarray(Wb), np.asarray(Wa))
                b = np.linalg.solve(np.asarray(Vb), np.asarray(Va))
                apb = a + b
                _U, s, Vh = np.linalg.svd(apb)
                smax, smin = float(s[0]), float(s[-1])
                w = np.abs(np.conj(Vh[-1])) ** 2
                w = w / np.sum(w)
                rows.append(dict(
                    n=int(apb.shape[0]),
                    cond=(smax / smin if smin > 0 else float("inf")),
                    n_tiny=int(np.sum(s < 1e-10 * smax)),
                    ipr=float(1.0 / np.sum(w ** 2)),
                    top_weight=float(np.max(w))))
            except Exception as exc:                      # pragma: no cover
                rows.append(dict(error=repr(exc)[:120]))
            return orig(Wa, Va, Wb, Vb)

        for name in self._MODULES:
            try:
                mod = importlib.import_module(name)
            except Exception:                             # pragma: no cover
                continue
            if hasattr(mod, "_interface_smatrix"):
                self._saved.append((mod, mod._interface_smatrix))
                mod._interface_smatrix = wrapped
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._interface_smatrix = fn
        return False

    def summary(self):
        ok = [r for r in self.rows if "cond" in r]
        if not ok:
            return dict(n_interfaces=0)
        return dict(n_interfaces=len(ok),
                    max_cond=max(r["cond"] for r in ok),
                    n_tiny_total=int(sum(r["n_tiny"] for r in ok)),
                    min_ipr=min(r["ipr"] for r in ok),
                    max_top_weight=max(r["top_weight"] for r in ok))


def _cell(pillar):
    return F.pillar_cell(S=32, host=HOST, pillar=pillar)


def fixtures():
    from lumenairy.elements.pmm import (
        PMM2DStackHybrid,
        PMM2DStackPure,
        pmm_efficiency_2d_cell,
        pmm_efficiency_2d_staggered,
        pmm_jones_2d,
        pmm_jones_2d_staggered,
    )
    out = []

    def add(name, cls, kind, fn):
        out.append((name, cls, kind, fn))

    # the hybrid single-cell path at four truncations: the coincidence's
    # signature on the RCWA side was truncation-INDEPENDENT, so a ladder is the
    # discriminator between "conditioning" and "truncation".
    for M in (3, 5, 7, 9):
        for tag, pillar in (("weak", WEAK), ("strong", STRONG)):
            for cls, nsub in (("region", NS), ("none", NS_OFF)):
                add("cell_%s_%s_M%d" % (tag, cls, M), cls, "eff",
                    (lambda p=pillar, n=nsub, m=M: pmm_efficiency_2d_cell(
                        PX, PX, _cell(p), n, 1.0, D, WL, degree=7,
                        n_orders=m, symmetry=False)))
    # the layer background EXACTLY equal to the SUPERSTRATE instead
    add("cell_weak_superstrate", "region-sup", "eff",
        lambda: pmm_efficiency_2d_cell(PX, PX, _cell(WEAK), 1.9, HOST ** 0.5,
                                       D, WL, degree=7, n_orders=5,
                                       symmetry=False))
    # tensor hybrid (layer through the FIXED rcwa function; region modes here)
    add("jones2d_region", "region", "jones",
        lambda: pmm_jones_2d(PX, PX, F.tensor_cell(), NS, 1.0, D, WL,
                             degree=7, n_orders=3))

    def hyb(spacer, nsub, pillar, M=4, sym=False, theta=0.0):
        st = PMM2DStackHybrid(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                              degree=7, n_orders=M, symmetry=sym)
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        st.add_layer(D, eps_cell=_cell(pillar))
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        return st.set_source(WL, theta=theta).solve()

    for M in (3, 5):
        for tag, pillar in (("weak", WEAK), ("strong", STRONG)):
            add("hyb_%s_region_M%d" % (tag, M), "region", "jones",
                lambda p=pillar, m=M: hyb(False, NS, p, m))
            add("hyb_%s_spacer_M%d" % (tag, M), "spacer", "jones",
                lambda p=pillar, m=M: hyb(True, NS_OFF, p, m))
            add("hyb_%s_both_M%d" % (tag, M), "both", "jones",
                lambda p=pillar, m=M: hyb(True, NS, p, m))
            add("hyb_%s_none_M%d" % (tag, M), "none", "jones",
                lambda p=pillar, m=M: hyb(False, NS_OFF, p, m))
    add("hyb_weak_both_oblique", "both", "jones",
        lambda: hyb(True, NS, WEAK, 4, theta=0.25))

    # PURE STAGGERED controls (no _sqrt_decay call at all)
    add("stag_weak_region", "region", "eff",
        lambda: pmm_efficiency_2d_staggered(
            PX, PX, F.stag_cell(host=HOST, pillar=WEAK), NS, 1.0, D, WL,
            degree=6, n_orders=4))
    add("stagjones_tensor_region", "region", "jones",
        lambda: pmm_jones_2d_staggered(PX, PX, F.stag_tensor_cell(), NS, 1.0,
                                       D, WL, degree=6, n_orders=3))

    def pure(spacer, nsub, pillar):
        st = PMM2DStackPure(PX, PX, n_substrate=nsub, n_superstrate=1.0,
                            degree=6, n_orders=4)
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        st.add_layer(D, eps_cell=F.stag_cell(host=HOST, pillar=pillar))
        if spacer:
            st.add_layer(0.1e-6, eps=HOST)
        return st.set_source(WL, theta=0.0).solve()

    add("pure_weak_both", "both", "jones", lambda: pure(True, NS, WEAK))
    add("pure_strong_none", "none", "jones", lambda: pure(False, NS_OFF,
                                                          STRONG))
    return out


def run(fn, kind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            with PMMInterfaceSpy() as ifc, F.PMMEigSpy() as eig:
                res = fn()
            return dict(raised=None,
                        closure=(F.closure_jones(res) if kind == "jones"
                                 else F.closure_eff(res)),
                        interfaces=ifc.summary(), eig=eig.summary())
        except Exception as exc:
            return dict(raised=type(exc).__name__, message=repr(exc)[:160])


def main():
    F.require_local_tree()
    out = sys.argv[1] if len(sys.argv) > 1 else "b2.json"
    rows = {}
    print("%-28s %-10s %-11s %-11s %-6s %-6s %s"
          % ("fixture", "class", "closure", "maxCond", "tiny", "incom", "ifc"))
    for name, cls, kind, fn in fixtures():
        post = run(fn, kind)
        with F.PreSqrtDecayPMM():
            pre = run(fn, kind)
        rows[name] = dict(cls=cls, installed=post, numpy_pre=pre)
        for arm, r in (("inst", post), ("pre", pre)):
            if r["raised"] is not None:
                print("%-28s %-10s %-4s RAISED %s" % (name, cls, arm,
                                                      r["raised"]))
                continue
            i = r["interfaces"]
            print("%-28s %-10s %-4s %+.3e %.3e %-6s %-6s %s"
                  % (name, cls, arm, r["closure"],
                     i.get("max_cond", float("nan")), i.get("n_tiny_total"),
                     r["eig"].get("incoming_after_exact_pin"),
                     i.get("n_interfaces")))
    F.dump(out, dict(rows=rows,
                     openblas_num_threads=os.environ.get(
                         "OPENBLAS_NUM_THREADS", "unpinned")))


if __name__ == "__main__":
    main()
