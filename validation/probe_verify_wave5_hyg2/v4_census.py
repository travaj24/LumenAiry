"""V4/A2 + A3 -- my own key census over the lens facade's attribute protocol
and the lens FIXTURES, plus the refusal's own contract.

    python v4_census.py <tree> <out.json>

My key set, not the author's 131.  Every write / delete attempt is bracketed
by a save-and-restore of BOTH modules' ``__dict__`` entries (direct dict
mutation, which bypasses ``__setattr__`` on either tree), because on the BASE
tree ``del lenses.CUPY_AVAILABLE`` removes the re-export outright and every
later read of it raises for the rest of the process.

Key families
------------
``A-read/write/rw/del/rd/restored-<name>``  the eight leaf-owned names x 6
``A-live-read/write-<name>``                the eight live forwards x 2
``A-dir*`` / ``A-type*`` / ``A-star*``      surface keys (A3)
``A-msg-<name>``                            refusal message shape (A3)
``A-leafwrite-<name>``                      the REDIRECTED statement (A3)
``F-*``                                     lens fixtures, numeric
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
_OUT = os.path.abspath(sys.argv[2])
sys.path.insert(0, _TREE)

import numpy as np  # noqa: E402

import vlib  # noqa: E402

vlib.anchor(_TREE)

from lumenairy.elements import _lens_kernels, lenses  # noqa: E402

LEAF_OWNED = ("CUPY_AVAILABLE", "_is_cupy_array", "_ensure_cupy_loaded",
              "_load_numba", "_get_aspheric_sag_accum_numba",
              "_ensure_numexpr_loaded", "_collect_semi_diameters",
              "_warn_if_aperture_exceeds_grid")

LIVE_FORWARD = ("cp", "_ne", "NUMEXPR_AVAILABLE", "_NUMBA_AVAILABLE",
                "_numba", "_njit", "_prange", "_NUMBA_KERNELS")

_MISS = object()


class _Sub:
    """A distinguishable substitute; its repr is stable across runs."""

    def __repr__(self):
        return "<v4-SUBSTITUTE>"


def _snap(mod, name):
    d = vars(mod)
    return (name in d, d.get(name, _MISS))


def _put(mod, name, snap):
    had, obj = snap
    d = vars(mod)
    if had:
        d[name] = obj
    else:
        d.pop(name, None)


def _attempt(fn, *a):
    try:
        return ("ok", repr(fn(*a))[:200])
    except BaseException as exc:                  # noqa: BLE001 -- recorded
        return ("raised", type(exc).__name__, str(exc))


def _star_names():
    ns = {}
    exec(compile("from lumenairy.elements.lenses import *",
                 "<v4-star>", "exec"), ns)
    return sorted(k for k in ns if k != "__builtins__")


def section_protocol(p, meas):
    sub = _Sub()
    for name in LEAF_OWNED:
        leaf_obj = getattr(_lens_kernels, name)
        s_shell = _snap(lenses, name)
        s_leaf = _snap(_lens_kernels, name)

        p.add(f"A-read-{name}", (
            hasattr(lenses, name),
            getattr(lenses, name, _MISS) is leaf_obj,
            name in vars(lenses),
            name in dir(lenses),
        ))

        # ---- WRITE
        res_w = _attempt(setattr, lenses, name, sub)
        p.add(f"A-write-{name}", res_w)
        p.add(f"A-rw-{name}", (                     # read-after-write
            getattr(lenses, name, _MISS) is leaf_obj,
            repr(getattr(lenses, name, _MISS))[:60],
            name in vars(lenses),
            getattr(_lens_kernels, name, _MISS) is leaf_obj,
        ))
        meas.setdefault("write_msgs", {})[name] = res_w
        _put(lenses, name, s_shell)
        _put(_lens_kernels, name, s_leaf)

        # ---- DELETE
        res_d = _attempt(delattr, lenses, name)
        p.add(f"A-del-{name}", res_d)
        p.add(f"A-rd-{name}", (                     # read-after-delete
            hasattr(lenses, name),
            getattr(lenses, name, _MISS) is leaf_obj,
            name in vars(lenses),
            getattr(_lens_kernels, name, _MISS) is leaf_obj,
        ))
        meas.setdefault("del_msgs", {})[name] = res_d
        _put(lenses, name, s_shell)
        _put(_lens_kernels, name, s_leaf)

        p.add(f"A-restored-{name}", (
            hasattr(lenses, name),
            getattr(lenses, name, _MISS) is leaf_obj,
            name in vars(lenses),
            name in dir(lenses),
        ))

        # ---- A3: the REDIRECTED statement against the leaf
        res_leaf = _attempt(setattr, _lens_kernels, name, sub)
        redirected = (res_leaf,
                      getattr(_lens_kernels, name, _MISS) is sub,
                      getattr(lenses, name, _MISS) is leaf_obj)
        p.add(f"A-leafwrite-{name}", redirected)
        meas.setdefault("leafwrite", {})[name] = [res_leaf, redirected[1],
                                                  redirected[2]]
        _put(_lens_kernels, name, s_leaf)
        _put(lenses, name, s_shell)

    # the eight LIVE forwards still forward
    for name in LIVE_FORWARD:
        s_leaf = _snap(_lens_kernels, name)
        s_shell = _snap(lenses, name)
        p.add(f"A-live-read-{name}", (
            hasattr(lenses, name),
            getattr(lenses, name, _MISS) is getattr(_lens_kernels, name, _MISS),
            name in vars(lenses),
            name in dir(lenses),
        ))
        res = _attempt(setattr, lenses, name, sub)
        p.add(f"A-live-write-{name}", (
            res,
            getattr(_lens_kernels, name, _MISS) is sub,
            getattr(lenses, name, _MISS) is sub,
            name in vars(lenses),
        ))
        _put(_lens_kernels, name, s_leaf)
        _put(lenses, name, s_shell)


def section_surface(p, meas):
    d = sorted(dir(lenses))
    p.add("A-dir-list", d)
    p.add("A-dir-len", len(d))
    star = _star_names()
    p.add("A-star-list", star)
    p.add("A-star-len", len(star))
    p.add("A-type-name", type(lenses).__name__)
    p.add("A-type-mro", [c.__name__ for c in type(lenses).__mro__])
    p.add("A-vars-keys", sorted(vars(lenses)))
    p.add("A-eight-in-vars", [n in vars(lenses) for n in LEAF_OWNED])
    p.add("A-eight-is-leaf", [getattr(lenses, n, _MISS)
                              is getattr(_lens_kernels, n, _MISS)
                              for n in LEAF_OWNED])
    # falsification arm: any OTHER name still assigns / reads / deletes
    other = "_v4_unrelated_name"
    r1 = _attempt(setattr, lenses, other, 7)
    r2 = _attempt(getattr, lenses, other)
    r3 = _attempt(delattr, lenses, other)
    p.add("A-other-name-roundtrip", (r1, r2, r3, other in vars(lenses)))
    meas["dir_list"] = d
    meas["star_list"] = star
    meas["type_name"] = type(lenses).__name__
    meas["type_mro"] = [c.__name__ for c in type(lenses).__mro__]
    meas["vars_keys"] = sorted(vars(lenses))


def _presc():
    return {
        "aperture_diameter": 0.050,
        "elements": [
            {"surf_num": 1, "semi_diameter": 0.012, "comment": "front"},
            {"surf_num": 2, "semi_diameter": 0.030, "comment": "rear"},
            {"surf_num": 3, "semi_diameter": float("nan")},
        ],
        "surfaces": [
            {"semi_diameter": 0.004},
            {"semi_diameter": 0.021},
        ],
    }


def section_fixtures(p):
    import warnings
    h = np.linspace(0.0, 0.02, 97) ** 2
    cases = [
        (0.05, 0.0, None), (-0.05, 0.0, None), (0.05, -1.0, None),
        (0.05, -1.5, {4: 1e-3, 6: -2e-5}), (0.05, 0.5, {4: -3e-4}),
        (np.inf, 0.0, None), (0.012, -0.7, {4: 5e-2, 6: 1e-3, 8: -2e-4}),
        (0.30, 2.0, {10: 7e-7}),
    ]
    for i, (R, k, a) in enumerate(cases):
        p.call(f"F-sag-general-{i}", lenses.surface_sag_general, h, R, k, a)
    X, Y = np.meshgrid(np.linspace(-0.01, 0.01, 41),
                       np.linspace(-0.008, 0.008, 37))
    bic = [
        dict(R_x=0.04, R_y=0.06, conic_x=-0.3, conic_y=0.2),
        dict(R_x=0.04, R_y=np.inf),
        dict(R_x=0.04, R_y=None, conic_x=-1.0,
             aspheric_coeffs={4: 1e-3}),
    ]
    for i, kw in enumerate(bic):
        p.call(f"F-sag-biconic-{i}", lenses.surface_sag_biconic, X, Y, **kw)

    pres = _presc()
    for i, (N, dx, sf) in enumerate([(512, 5e-5, 1.0), (2048, 5e-5, 1.0),
                                     (2048, 5e-5, 0.5), (8192, 2e-5, 2.0)]):
        p.call(f"F-gridcheck-{i}", lenses.check_grid_vs_apertures,
               pres, N, dx, safety_factor=sf)
    for i, (wl, sw) in enumerate([(1.55e-6, None), (1.064e-6, 2e-3)]):
        p.call(f"F-recommend-{i}", lenses.recommend_grid_for_prescription,
               pres, wl, source_waist=sw)
    for i, (N, dx) in enumerate([(256, 5e-5), (16384, 5e-5)]):
        def _w(N=N, dx=dx):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                lenses._warn_if_aperture_exceeds_grid(pres, N, dx,
                                                      source="v4")
            return [(w.category.__name__, str(w.message)) for w in caught]
        p.call(f"F-warn-{i}", _w)
    p.call("F-collect-0", lenses._collect_semi_diameters, pres)
    p.call("F-collect-1", lenses._collect_semi_diameters,
           {"surfaces": [{"semi_diameter": 1.0}]})
    for i, (nv, mo) in enumerate([(1, 0), (2, 3), (3, 2), (2, 5), (4, 2),
                                  (3, 4)]):
        p.call(f"F-multi-{i}", lenses._multi_indices_total_degree, nv, mo)
    rng = np.random.default_rng(20260919)
    v = rng.normal(size=257) * 3.0 + 1.5
    for i, pad in enumerate([0.0, 0.05, 0.25]):
        p.call(f"F-fitnorm-{i}", lenses._fit_normaliser, v, pad)
        p.call(f"F-fitnorm-const-{i}", lenses._fit_normaliser,
               np.full(11, 2.5), pad)


def main():
    p = vlib.Probe()
    meas = {"build": vlib.build_tag(), "tree": _TREE, "python": sys.version}
    section_protocol(p, meas)
    section_surface(p, meas)
    section_fixtures(p)
    p.write(_OUT)
    vlib.write_json(meas, _OUT.replace(".json", "_meas.json"))


if __name__ == "__main__":
    main()
