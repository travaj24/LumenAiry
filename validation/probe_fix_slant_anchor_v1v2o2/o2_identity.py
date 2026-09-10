"""O2 -- BIT-IDENTITY of every HEALTHY generalized-cascade fixture, and the
conditioning of the two paths the refusal does NOT cover.

Part 1: the same fixtures ``o2_census.py`` scores, hashed on
``(orders, R, T, jones)``.  A screen that only ADDS a raise must leave every
row it does not refuse byte-for-byte where it was.

Part 2: the two neighbouring paths the brief asks about.

  * the PURE engine's own generalized route.  It calls the SAME
    ``_interface_smatrix_general`` (``stack2d_pure.py:1423-1426``) after
    ``_modes_as_general`` packs its six-tuple, so it is covered by the armed
    guard rather than needing a second one -- and its measured ``T22``
    readings say it is genuinely fine, not merely unmeasured.
  * the MORTAR interfaces (``_interface_smatrix_general_mortar`` and
    ``_interface_smatrix_general_mortar_2d``).  These do NOT take an explicit
    inverse at all -- both end in ``np.linalg.solve(A, B)`` -- so the
    ``_guarded_inverse`` shape does not exist there.  What IS measured here is
    ``cond(A)`` of that solve on the same blow-up fixture class, so the claim
    "different shape, and not exposed" is a measurement rather than a reading
    of the source.
"""
from __future__ import annotations

import importlib
import sys
import time
import warnings

import _lib as L
import numpy as np
import o2_census as C

_MORTAR = ("_interface_smatrix_general_mortar",
           "_interface_smatrix_general_mortar_2d")


def solve_hashed(name):
    """Re-run one census fixture, hashing its returns instead of its R+T."""
    src = C.build_cases()[name]
    got = {}
    C.LAST.pop("res", None)
    rec = C.scored(src)
    res = C.LAST.get("res")
    if res is not None:
        got["orders"] = L.sha(res[0])
        got["R"] = L.sha(res[1])
        got["T"] = L.sha(res[2])
        got["J"] = L.sha(res[3])
    got["outcome"] = rec["outcome"]
    if rec["outcome"] == "RAISE":
        got["exc"] = rec["exc"]
        got["msg"] = rec["msg"][:160]
    else:
        got["RT"] = rec["RT"]
    return got


def mortar_conditioning():
    """``cond(A)`` at every mortar interface reached by the blow-up fixture
    class, via a direct patch of the two functions' ``np.linalg.solve``."""
    import lumenairy.elements.pmm._core as pc
    rows = []
    real_solve = np.linalg.solve
    depth = {"in": 0}

    def spy(A, B):
        if depth["in"]:
            An = np.asarray(A)
            if An.ndim == 2 and An.shape[0] == An.shape[1]:
                rows.append(dict(n=int(An.shape[0]),
                                 cond=float(np.linalg.cond(An))))
        return real_solve(A, B)

    saved = {}
    for nm in _MORTAR:
        saved[nm] = getattr(pc, nm)

    def wrap(fn):
        def inner(*a, **kw):
            depth["in"] += 1
            try:
                return fn(*a, **kw)
            finally:
                depth["in"] -= 1
        return inner

    for nm, fn in saved.items():
        setattr(pc, nm, wrap(fn))
        for mod in ("lumenairy.elements.pmm.stack",
                    "lumenairy.elements.pmm.stack2d_pure"):
            m = importlib.import_module(mod)
            if hasattr(m, nm):
                setattr(m, nm, getattr(pc, nm))
    np.linalg.solve = spy
    out = {}
    try:
        for name, fn in _mortar_cases().items():
            rows.clear()
            try:
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    rt = fn()
                rec = dict(outcome="SOLVED", RT=rt,
                           warned=sorted({str(x.message)[:60] for x in w}))
            except Exception as e:                        # noqa: BLE001
                rec = dict(outcome="RAISE", exc=type(e).__name__,
                           msg=str(e)[:200])
            rec["mortar"] = dict(
                n=len(rows),
                max_cond=(max(r["cond"] for r in rows) if rows else None),
                sizes=sorted({r["n"] for r in rows}))
            out[name] = rec
    finally:
        np.linalg.solve = real_solve
        for nm, fn in saved.items():
            setattr(pc, nm, fn)
            for mod in ("lumenairy.elements.pmm.stack",
                        "lumenairy.elements.pmm.stack2d_pure"):
                m = importlib.import_module(mod)
                if hasattr(m, nm):
                    setattr(m, nm, fn)
    return out


def _mortar_cases():
    """Fixtures that route through a MORTAR interface: non-conforming
    per-layer grids, 1-D and 2-D, on the blow-up class and beside it."""
    import math

    def oned_perlayer(shear, M, degree):
        def go():
            from lumenairy.elements.pmm.stack import PMMStack
            st = PMMStack(0.80e-6, n_superstrate=1.0, n_substrate=1.6,
                          degree=degree, n_orders=M,
                          factorization="convection",
                          layer_grids="per-layer")
            st.add_sheared_grating(0.40e-6, eps_ridge=4.2, eps_groove=1.45,
                                   duty=0.45, shear=shear)
            st.add_layer(0.10e-6, segments=[(0.31, 2.9), (0.69, 1.6)])
            st.set_source(0.55e-6, theta=math.radians(25.0))
            return C._rt(st.solve())
        return go

    def oop_grid(n, ground):
        c = np.full((n, n), ground, dtype=complex)
        c[:, :n // 2] = ground + 2.0
        return C._oop_tensor(c)

    def pure_mortar(M, n_modes, ground):
        """The ONLY route to ``_interface_smatrix_general_mortar_2d``: an
        OUT-OF-PLANE pure stack on ``layer_grids='per-layer'`` whose two layers
        carry DIFFERENT grids (an identical-grid pair takes the square
        ``_interface_smatrix_general`` bypass instead)."""
        def go():
            from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
            st = PMM2DStackPure(C.PX, C.PY, n_superstrate=C.NSUP,
                                n_substrate=C.NSUB, n_modes=n_modes,
                                n_orders=M, layer_grids="per-layer")
            st.add_layer(C.D1, eps_cell=oop_grid(6, ground))
            st.add_layer(C.DF, eps_cell=oop_grid(4, ground + 1.1))
            st.set_source(C.WL, theta=math.radians(25.0), phi=0.0)
            return C._rt(st.solve())
        return go

    cases = {}
    for sh in (0.30, 0.60):
        for M, dg in ((7, 12), (11, 12), (7, 16)):
            cases["oned_perlayer_sh%.2f_M%d_dg%d" % (sh, M, dg)] = \
                oned_perlayer(sh, M, dg)
    for M in (3, 5):
        for g in (1.0, 2.1):
            cases["pure_mortar2d_M%d_g%.1f" % (M, g)] = pure_mortar(M, 4, g)
    cases["pure_mortar2d_M3_nm6_g1.0"] = pure_mortar(3, 6, 1.0)
    return cases


def main():
    t0 = time.time()
    res = {}
    names = sorted(C.build_cases())
    for nm in names:
        res[nm] = solve_hashed(nm)
        print("%-34s %-8s %s" % (nm, res[nm]["outcome"],
                                 res[nm].get("R", res[nm].get("exc"))))
    res["_mortar"] = mortar_conditioning()
    print("\n== MORTAR interfaces (np.linalg.solve, no explicit inverse) ==")
    for k, v in res["_mortar"].items():
        print("%-34s %-8s RT=%-12.6g n=%-3s cond=%s"
              % (k, v["outcome"], v.get("RT", float("nan")),
                 v["mortar"]["n"], v["mortar"]["max_cond"]))
    res["_seconds"] = round(time.time() - t0, 1)
    L.dump("o2_identity", res,
           suffix=(sys.argv[1] if len(sys.argv) > 1 else ""))


if __name__ == "__main__":
    main()
