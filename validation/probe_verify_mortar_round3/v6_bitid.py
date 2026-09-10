"""V6 -- BIT-IDENTITY of everything the round-3 diff was NOT supposed to move,
on an INDEPENDENT battery, against the PRE-fix worktree.

The battery is built here, not borrowed: 34 fixtures covering the shared-grid
path (scalar / tensor / OUT-OF-PLANE / magnetic / slanted / ``retain_internal``
/ per-order amplitudes / ``jones=False`` / a second modal count), the per-layer
path (conforming, non-conforming, nested, per-layer ``n_modes``, a uniform
spacer, a uniform-lattice neighbour, both taper builders at two rules), the
1-D ``PMMStack``, and the MIXED in-plane / out-of-plane stacks round 2
refused -- which are the ONLY entries permitted to change, and only from a
refusal to an answer.

Every returned array is hashed as ``sha256(dtype | shape | tobytes)``, and the
WARNING SET is recorded beside it, so a fixture that silently gained or lost a
warning is a mismatch even when its numbers are identical.

Run:
  ``python v6_bitid.py post C:/tmp/lum_vmortar3``
  ``python v6_bitid.py pre  C:/tmp/lum_vmortar3_pre``
  ``python v6_bitid.py compare``
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys
import warnings

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def _use_root(root):
    root = pathlib.Path(root).resolve()
    sys.path.insert(0, str(root))
    import lumenairy
    got = pathlib.Path(lumenairy.__file__).resolve()
    if root not in got.parents:
        raise SystemExit(f"REFUSING: lumenairy resolved to {got}, not {root}")
    return lumenairy


def _h(a):
    a = np.ascontiguousarray(np.asarray(a))
    d = hashlib.sha256()
    d.update(str(a.dtype).encode())
    d.update(str(a.shape).encode())
    d.update(a.tobytes())
    return d.hexdigest()


def battery(lum):
    from lumenairy.elements.pmm import PMM2DStackPure, PMMStack
    from lumenairy.elements.pmm.twod_staggered import _C

    P, WL, TH, PH = 0.87e-6, 0.73e-6, 0.17, 1.1
    EH, EB = 1.96, 8.41
    WA = (0.1873, 0.5412)
    WB = (0.3106, 0.8039)
    WU = (0.1873, 0.3106, 0.5412, 0.8039)
    WN = (0.25, 0.375, 0.5, 0.75)
    EOOP = np.array([[3.10, 0.0, 0.62], [0.0, 2.70, 0.0],
                     [0.55, 0.0, 2.45]], dtype=_C)
    MU = np.array([[1.35, 0.0, 0.0], [0.0, 1.20, 0.0],
                   [0.0, 0.0, 1.10]], dtype=_C)

    def sc(w):
        return [x * P for x in w]

    def mids(w):
        b = (0.0,) + tuple(w) + (1.0,)
        return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]

    def scal(w, lo, hi, e=EB):
        m = mids(w)
        c = np.full((len(m), len(m)), _C(EH))
        for i in range(len(m)):
            for j in range(len(m)):
                if lo < m[i] < hi and lo < m[j] < hi:
                    c[i, j] = _C(e)
        return c

    def tens(w, lo, hi, e=None):
        m = mids(w)
        n = len(m)
        c = np.empty((n, n, 3, 3), dtype=_C)
        ee = EOOP if e is None else np.asarray(e, dtype=_C)
        for i in range(n):
            for j in range(n):
                c[i, j] = ee if (lo < m[i] < hi and lo < m[j] < hi) \
                    else np.eye(3) * EH
        return c

    def mucell(w, lo, hi):
        m = mids(w)
        n = len(m)
        c = np.empty((n, n, 3, 3), dtype=_C)
        for i in range(n):
            for j in range(n):
                c[i, j] = MU if (lo < m[i] < hi and lo < m[j] < hi) \
                    else np.eye(3)
        return c

    def new(M=4, no=2, grids="shared", **kw):
        return PMM2DStackPure(P, n_modes=M, n_orders=no, n_substrate=1.45,
                              layer_grids=grids, **kw)

    F = {}

    # ---- SHARED grid path ------------------------------------------------
    def f_shared_scalar():
        st = new()
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["shared_scalar"] = f_shared_scalar

    def f_shared_scalar_jones():
        st = new()
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve()
    F["shared_scalar_jones"] = f_shared_scalar_jones

    def f_shared_scalar_M6():
        st = new(M=6, no=3)
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["shared_scalar_M6_N3"] = f_shared_scalar_M6

    def f_shared_tensor():
        e = np.array([[2.9, 0.35, 0.0], [0.32, 3.3, 0.0], [0.0, 0.0, 2.6]],
                     dtype=_C)
        st = new()
        st.add_layer(0.118e-6, eps=e)
        st.add_layer(0.094e-6, eps_cell=scal(WA, *WA))
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["shared_tensor_uniform"] = f_shared_tensor

    def f_shared_oop():
        st = new()
        st.add_layer(0.118e-6, eps_cell=tens(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["shared_oop"] = f_shared_oop

    def f_shared_oop_normal():
        st = new()
        st.add_layer(0.118e-6, eps_cell=tens(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=0.0, phi=0.0)
        return st.solve(jones=False)
    F["shared_oop_normal_incidence"] = f_shared_oop_normal

    def f_shared_magnetic():
        st = new()
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), mu_cell=mucell(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["shared_magnetic"] = f_shared_magnetic

    def f_shared_slant():
        st = new()
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["shared_slant"] = f_shared_slant

    def f_shared_retain_internal():
        st = new()
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        out = st.solve(jones=False, retain_internal=True)
        return out + (st.layer_absorption(),)
    F["shared_retain_internal"] = f_shared_retain_internal

    def f_shared_amplitudes():
        st = new()
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        st.solve(jones=False)
        a = st.per_order_amplitudes()
        # a DICT: hash each value, ordered by key.  ``np.asarray`` of a dict
        # is a 0-d OBJECT array whose ``tobytes`` is a POINTER, so it hashes
        # differently in every process -- a HARNESS bug that reads exactly
        # like a library regression (it did: on this file's first run
        # shared_per_order_amplitudes was the only HASH DIFF of 37 fixtures).
        return tuple(np.asarray(a[k]) for k in sorted(a))
    F["shared_per_order_amplitudes"] = f_shared_amplitudes

    def f_shared_lossy():
        st = new()
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA, e=complex(-18.0, 1.2)))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["shared_lossy_metal"] = f_shared_lossy

    # ---- PER-LAYER, IN-PLANE only (the two in-plane mortar sites) --------
    def _perlayer(wa, wb, **kw):
        def f():
            st = new(grids="per-layer", **kw)
            st.add_layer(0.118e-6, eps_cell=scal(wa, wa[0], wa[-1]),
                         x_walls=sc(wa), y_walls=sc(wa))
            st.add_layer(0.094e-6, eps_cell=scal(wb, wb[0], wb[-1], e=5.3),
                         x_walls=sc(wb), y_walls=sc(wb))
            st.set_source(WL, theta=TH, phi=PH)
            return st.solve(jones=False)
        return f
    F["perlayer_conforming"] = _perlayer(WA, WA)
    F["perlayer_non_conforming"] = _perlayer(WA, WB)
    F["perlayer_non_conforming_M6"] = _perlayer(WA, WB, M=6)
    F["perlayer_nested"] = _perlayer(WN, WA)
    F["perlayer_union"] = _perlayer(WU, WU)

    def f_perlayer_permodes():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), x_walls=sc(WA),
                     y_walls=sc(WA), n_modes=5)
        st.add_layer(0.094e-6, eps_cell=scal(WB, *WB, e=5.3), x_walls=sc(WB),
                     y_walls=sc(WB), n_modes=4)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_per_layer_n_modes"] = f_perlayer_permodes

    def f_perlayer_uniform_spacer():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), x_walls=sc(WA),
                     y_walls=sc(WA))
        st.add_layer(0.094e-6, eps=2.56)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_uniform_spacer"] = f_perlayer_uniform_spacer

    def f_perlayer_uniform_lattice():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), x_walls=sc(WA),
                     y_walls=sc(WA))
        st.add_layer(0.094e-6, eps=2.56, grid=5)
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_uniform_lattice_N5"] = f_perlayer_uniform_lattice

    def f_perlayer_magnetic():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), mu_cell=mucell(WA, *WA),
                     x_walls=sc(WA), y_walls=sc(WA))
        st.add_layer(0.094e-6, eps_cell=scal(WB, *WB, e=5.3), x_walls=sc(WB),
                     y_walls=sc(WB))
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_magnetic"] = f_perlayer_magnetic

    def f_perlayer_slant_both():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), x_walls=sc(WA),
                     y_walls=sc(WA), slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps_cell=scal(WB, *WB, e=5.3), x_walls=sc(WB),
                     y_walls=sc(WB), slant=(0.11, 0.05))
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_slant_both"] = f_perlayer_slant_both

    def f_perlayer_slant_conforming():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=scal(WA, *WA), x_walls=sc(WA),
                     y_walls=sc(WA), slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps_cell=scal(WA, *WA, e=5.3), x_walls=sc(WA),
                     y_walls=sc(WA), slant=(0.11, 0.05))
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_slant_conforming"] = f_perlayer_slant_conforming

    def f_perlayer_oop_both():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=tens(WA, *WA), x_walls=sc(WA),
                     y_walls=sc(WA))
        st.add_layer(0.094e-6, eps_cell=tens(WB, *WB, e=0.7 * EOOP),
                     x_walls=sc(WB), y_walls=sc(WB))
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_oop_both"] = f_perlayer_oop_both

    def f_perlayer_oop_conforming():
        st = new(grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=tens(WA, *WA), x_walls=sc(WA),
                     y_walls=sc(WA))
        st.add_layer(0.094e-6, eps_cell=tens(WA, *WA, e=0.7 * EOOP),
                     x_walls=sc(WA), y_walls=sc(WA))
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["perlayer_oop_conforming"] = f_perlayer_oop_conforming

    # ---- TAPERS ----------------------------------------------------------
    for ns, rule in ((4, "midpoint"), (8, "midpoint"), (8, "bottom"),
                     (16, "midpoint")):
        def f_taper(ns=ns, rule=rule):
            st = new(grids="per-layer")
            st.add_tapered_pillar(
                0.3e-6, eps_pillar=6.0, eps_host=2.0,
                x_bounds_bottom=(0.25 * P, 0.75 * P),
                y_bounds_bottom=(0.25 * P, 0.75 * P),
                x_bounds_top=(0.32 * P, 0.68 * P),
                y_bounds_top=(0.32 * P, 0.68 * P),
                n_slices=ns, rule=rule)
            st.set_source(WL, theta=TH, phi=PH)
            return st.solve(jones=False)
        F[f"taper_{rule}_{ns}"] = f_taper

    def f_tapered_pillars():
        st = new(grids="per-layer")
        st.add_tapered_pillars(
            0.3e-6, eps_host=2.0, n_slices=8,
            pillars=[((0.28 * P, 0.28 * P), (0.14 * P, 0.14 * P),
                      (0.20 * P, 0.20 * P), 6.0),
                     ((0.72 * P, 0.72 * P), (0.14 * P, 0.14 * P),
                      (0.20 * P, 0.20 * P), 6.0)])
        st.set_source(WL, theta=TH, phi=PH)
        return st.solve(jones=False)
    F["tapered_pillars_8"] = f_tapered_pillars

    # ---- 1-D PMMStack ----------------------------------------------------
    for deg in (8, 12):
        def f_1d(deg=deg):
            st = PMMStack(1.07, degree=deg, far_field_orders=5)
            # segments are FRACTIONS of the period and must sum to 1
            st.add_layer(0.17, segments=[(0.22, 1.69), (0.31, 9.0),
                                         (0.47, 1.69)])
            st.add_layer(0.17, segments=[(1.0, 1.69)])
            st.add_layer(0.17, segments=[(0.39, 1.69), (0.40, 9.0),
                                         (0.21, 1.69)])
            st.set_source(0.79, theta=0.31)
            return tuple(np.asarray(x) for x in st.solve(stabilize=None)[:3])
        F[f"pmmstack_1d_deg{deg}"] = f_1d

    # ---- the MIXED stacks round 2 REFUSED (permitted to change) ----------
    def _mixed(kind, M):
        def f():
            st = new(M=M, grids="per-layer")
            st.add_layer(0.118e-6, eps_cell=tens(WA, *WA), x_walls=sc(WA),
                         y_walls=sc(WA))
            if kind == "spacer":
                st.add_layer(0.094e-6, eps=2.56)
            else:
                st.add_layer(0.094e-6, eps_cell=scal(WB, *WB, e=5.3),
                             x_walls=sc(WB), y_walls=sc(WB))
            st.set_source(WL, theta=TH, phi=PH)
            return st.solve(jones=False)
        return f
    for kind in ("spacer", "pattern"):
        for M in (4, 5, 6):
            F[f"MIXED_{kind}_M{M}"] = _mixed(kind, M)

    return F


def run(tag, root):
    lum = _use_root(root)
    F = battery(lum)
    rows = {}
    for name, fn in F.items():
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter("always")
            try:
                out = fn()
                hashes = [_h(a) for a in
                          (out if isinstance(out, tuple) else (out,))]
                err = None
            except Exception as e:                          # noqa: BLE001
                hashes, err = [], f"{type(e).__name__}: {str(e)[:160]}"
        rows[name] = {
            "hashes": hashes, "error": err,
            "warnings": sorted(f"{w.category.__name__}:{str(w.message)[:90]}"
                               for w in ws)}
        print(f"  {name:34s} {len(hashes):2d} hashes  "
              f"{len(rows[name]['warnings'])} warn  "
              f"{'ERR ' + err[:60] if err else ''}", flush=True)
    out = {"tag": tag, "root": str(pathlib.Path(root).resolve()),
           "lumenairy": lum.__file__, "version": lum.__version__,
           "python": platform.python_version(), "numpy": np.__version__,
           "n_fixtures": len(rows),
           "n_hashes": sum(len(r["hashes"]) for r in rows.values()),
           "rows": rows}
    (HERE / f"v6_bitid_{tag}.json").write_text(json.dumps(out, indent=1),
                                               encoding="cp1252")
    print(f"wrote v6_bitid_{tag}.json  ({len(rows)} fixtures, "
          f"{out['n_hashes']} hashes)")


def compare(a="post", b="pre"):
    A = json.loads((HERE / f"v6_bitid_{a}.json").read_text(encoding="cp1252"))
    B = json.loads((HERE / f"v6_bitid_{b}.json").read_text(encoding="cp1252"))
    same = diff = 0
    report = []
    for name in sorted(set(A["rows"]) | set(B["rows"])):
        ra, rb = A["rows"].get(name), B["rows"].get(name)
        if ra is None or rb is None:
            report.append((name, "MISSING", "", ""))
            diff += 1
            continue
        hs = "identical" if ra["hashes"] == rb["hashes"] else "HASH DIFF"
        ws = "identical" if ra["warnings"] == rb["warnings"] \
            else "WARNING DIFF"
        if ra["error"] != rb["error"]:
            hs = f"ERROR CHANGED: pre={rb['error']} post={ra['error']}"
        if hs == "identical" and ws == "identical":
            same += 1
        else:
            diff += 1
            report.append((name, hs, ws,
                           f"pre_warn={rb['warnings']} "
                           f"post_warn={ra['warnings']}"))
    print(f"{a} vs {b}: {same} identical, {diff} differing "
          f"(of {len(A['rows'])} fixtures, {A['n_hashes']} hashes)")
    for r in report:
        print("  ", r[0], "|", r[1], "|", r[2])
        if r[3]:
            print("      ", r[3][:300])
    (HERE / "v6_bitid_compare.json").write_text(
        json.dumps({"a": a, "b": b, "identical": same, "differing": diff,
                    "n_fixtures": len(A["rows"]),
                    "n_hashes_a": A["n_hashes"], "n_hashes_b": B["n_hashes"],
                    "report": report}, indent=1), encoding="cp1252")


def selfcheck(root):
    """The harness's OWN determinism: two independent draws in the SAME tree
    must hash identically.  A fixture that fails this cannot detect a library
    change -- it reports one whatever happens."""
    lum = _use_root(root)
    F = battery(lum)
    bad = []
    for name, fn in F.items():
        h = []
        for _ in range(2):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                try:
                    out = fn()
                    h.append([_h(a) for a in
                              (out if isinstance(out, tuple) else (out,))])
                except Exception as e:                      # noqa: BLE001
                    h.append([type(e).__name__])
        if h[0] != h[1]:
            bad.append(name)
    print(f"selfcheck: {len(F) - len(bad)}/{len(F)} fixtures reproduce their "
          f"own hashes; NON-REPRODUCIBLE: {bad}")
    (HERE / "v6_bitid_selfcheck.json").write_text(
        json.dumps({"root": str(root), "n_fixtures": len(F),
                    "non_reproducible": bad}, indent=1), encoding="cp1252")


if __name__ == "__main__":
    if sys.argv[1] == "selfcheck":
        selfcheck(sys.argv[2])
    elif sys.argv[1] == "compare":
        compare(*(sys.argv[2:4] or ["post", "pre"]))
    else:
        run(sys.argv[1], sys.argv[2])
