"""VERIFY round 2, task 2 -- MY OWN bit-identity fixtures across 24651c8 -> HEAD.

Run TWICE, once per arm, with PYTHONPATH pointing at that arm's tree:

    V1_TAG=with PYTHONPATH=/c/tmp/lum_vmortar2  python .../v1_bitid.py
    V1_TAG=pre  PYTHONPATH=/c/tmp/lum_prem2 V1_EXPECT_ROOT=/c/tmp/lum_prem2 \
                                            python .../v1_bitid.py

then ``python v1_compare.py with pre``.  Hashes are sha256 over
dtype|shape|tobytes of every returned array, so a single last-bit move shows.
Warning categories+messages are recorded alongside, because "no bit moved" is
only half the claim -- the WARNING SET must not move either.
"""
import hashlib
import json
import os
import sys
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack import PMMStack
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

TAG = os.environ.get("V1_TAG", "with")
EXPECT = os.environ.get("V1_EXPECT_ROOT")
if EXPECT:
    assert os.path.abspath(lumenairy.__file__).lower().startswith(
        os.path.abspath(EXPECT).lower()), (
        f"WRONG TREE: {lumenairy.__file__} not under {EXPECT}")
print(f"[{TAG}] lumenairy {lumenairy.__file__} v{lumenairy.__version__}",
      flush=True)

OUT = {}


def h(a):
    a = np.ascontiguousarray(np.asarray(a))
    m = hashlib.sha256()
    m.update(str(a.dtype).encode())
    m.update(str(a.shape).encode())
    m.update(a.tobytes())
    return m.hexdigest()


def rec(name, arrays, warns):
    OUT[name] = {
        "hashes": {k: h(v) for k, v in arrays.items()},
        "warnings": sorted(f"{w.category.__name__}: {str(w.message)[:120]}"
                           for w in warns),
    }
    print(f"  [{TAG}] {name}: {len(arrays)} hashes, {len(warns)} warnings",
          flush=True)


def run(name, fn):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        arrays = fn()
    rec(name, arrays, ws)


# ------------------------------------------------------------------ helpers
WL = 0.62e-6
PX = 0.9e-6


def cell(nx, ny, e_host, e_incl, box):
    c = np.full((nx, ny), complex(e_host))
    (i0, i1), (j0, j1) = box
    c[i0:i1, j0:j1] = complex(e_incl)
    return c


def st(**kw):
    kw.setdefault("period_x", PX)
    kw.setdefault("n_substrate", 1.5)
    kw.setdefault("n_modes", 5)
    kw.setdefault("n_orders", 3)
    return PMM2DStackPure(**kw)


def out(res):
    o, R, T = res[0], res[1], res[2]
    d = {"orders": np.asarray(o), "R": np.asarray(R), "T": np.asarray(T)}
    if len(res) > 3:
        d["jones"] = np.asarray(res[3])
    return d


# =========================================================== SHARED-GRID arms
def f_shared_scalar():
    s = st()
    s.add_layer(0.15e-6, eps=2.1)
    s.add_layer(0.22e-6, eps_cell=cell(3, 3, 2.25, 6.0, ((1, 2), (1, 2))))
    s.add_layer(0.10e-6, eps=1.8)
    return out(s.set_source(WL, theta=0.21, phi=0.4).solve())


def f_shared_scalar_normal():
    s = st(n_modes=6)
    s.add_layer(0.2e-6, eps_cell=cell(3, 3, 2.25, 9.0, ((0, 2), (1, 3))))
    return out(s.set_source(WL).solve())


def f_shared_tensor_inplane():
    e = np.array([[4.0 + 0.02j, 0.6, 0.0],
                  [0.55, 3.1, 0.0],
                  [0.0, 0.0, 3.6]], dtype=complex)
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[...] = np.eye(3) * 2.25
    c[0, 0] = e
    s = st()
    s.add_layer(0.18e-6, eps_cell=c)
    s.add_layer(0.09e-6, eps=2.0)
    return out(s.set_source(WL, theta=0.15, phi=0.7).solve())


def f_shared_tensor_oop():
    e = np.array([[4.0, 0.0, 0.9],
                  [0.0, 3.4, 0.0],
                  [0.85, 0.0, 3.2]], dtype=complex)
    c = np.empty((2, 2, 3, 3), dtype=complex)
    c[...] = np.eye(3) * 2.25
    c[1, 0] = e
    s = st(n_modes=5)
    s.add_layer(0.16e-6, eps_cell=c)
    return out(s.set_source(WL, theta=0.12, phi=0.25).solve())


def f_shared_magnetic():
    s = st()
    s.add_layer(0.17e-6,
                eps_cell=cell(2, 2, 2.25, 5.0, ((0, 1), (0, 1))),
                mu_cell=cell(2, 2, 1.0, 1.7, ((0, 1), (0, 1))))
    return out(s.set_source(WL, theta=0.1).solve())


def f_shared_slant():
    s = st(n_modes=5)
    s.add_layer(0.14e-6, eps_cell=cell(2, 2, 2.25, 5.5, ((0, 1), (0, 2))),
                slant=(0.09, 0.0))
    return out(s.set_source(WL, theta=0.08).solve())


def f_shared_retain_internal():
    s = st()
    s.add_layer(0.13e-6, eps_cell=cell(3, 3, 2.25, 6.0 + 0.3j,
                                       ((1, 2), (0, 2))))
    s.add_layer(0.11e-6, eps=2.4)
    res = s.set_source(WL, theta=0.19).solve(retain_internal=True)
    d = out(res)
    ab = s.layer_absorption()
    d["absorption"] = np.asarray(ab, dtype=float)
    return d


def f_shared_modal_amps():
    s = st()
    s.add_layer(0.2e-6, eps_cell=cell(3, 3, 2.25, 7.0, ((0, 2), (1, 2))))
    res = s.set_source(WL, theta=0.23, phi=0.33).solve()
    d = out(res)
    for port in ("reflection", "transmission"):
        amp = s.per_order_amplitudes(port=port)
        # a DICT of arrays: hash each entry, never the dict (np.asarray of a
        # dict is a 0-d object array whose tobytes is a POINTER -- a trap this
        # probe fell into on its first pass)
        for k in sorted(amp):
            v = amp[k]
            if isinstance(v, (int, float, complex, np.number)):
                v = np.array([v])
            d[f"amp_{port}_{k}"] = np.asarray(v)
    return d


def f_shared_jones_false():
    s = st()
    s.add_layer(0.2e-6, eps_cell=cell(2, 2, 2.25, 4.0, ((0, 1), (0, 1))))
    return out(s.set_source(WL, theta=0.05).solve(jones=False))


def f_shared_M6_N4():
    s = st(n_modes=6, n_orders=4)
    s.add_layer(0.19e-6, eps_cell=cell(4, 4, 2.25, 8.0, ((1, 3), (1, 3))))
    return out(s.set_source(WL, theta=0.3, phi=1.1).solve())


# ======================================== PER-LAYER, ORDINARY NON-UNIFORM
def f_pl_conforming():
    s = st(layer_grids="per-layer")
    xw = [0.2371 * PX, 0.6183 * PX]
    s.add_layer(0.15e-6, eps_cell=cell(3, 3, 2.25, 6.0, ((1, 2), (1, 2))),
                x_walls=xw, y_walls=xw)
    s.add_layer(0.12e-6, eps_cell=cell(3, 3, 2.25, 4.0, ((1, 2), (0, 2))),
                x_walls=xw, y_walls=xw)
    return out(s.set_source(WL, theta=0.17, phi=0.5).solve())


def f_pl_nonconforming():
    s = st(layer_grids="per-layer")
    s.add_layer(0.15e-6, eps_cell=cell(3, 3, 2.25, 6.0, ((1, 2), (1, 2))),
                x_walls=[0.2371 * PX, 0.6183 * PX],
                y_walls=[0.31 * PX, 0.72 * PX])
    s.add_layer(0.12e-6, eps_cell=cell(3, 3, 2.25, 4.0, ((1, 2), (0, 2))),
                x_walls=[0.3117 * PX, 0.7402 * PX],
                y_walls=[0.24 * PX, 0.66 * PX])
    return out(s.set_source(WL, theta=0.17, phi=0.5).solve())


def f_pl_nested():
    s = st(layer_grids="per-layer", n_modes=5)
    s.add_layer(0.14e-6,
                eps_cell=cell(5, 5, 2.25, 7.0, ((2, 3), (2, 3))),
                x_walls=[0.125 * PX, 0.25 * PX, 0.75 * PX, 0.875 * PX],
                y_walls=[0.125 * PX, 0.25 * PX, 0.75 * PX, 0.875 * PX])
    s.add_layer(0.10e-6, eps=2.4)
    return out(s.set_source(WL, theta=0.11).solve())


def f_pl_mixed_uniform_patterned():
    s = st(layer_grids="per-layer")
    s.add_layer(0.10e-6, eps=2.1)
    s.add_layer(0.16e-6, eps_cell=cell(3, 3, 2.25, 6.0, ((1, 2), (1, 2))))
    s.add_layer(0.13e-6, eps_cell=cell(2, 2, 2.25, 5.0, ((0, 1), (0, 1))),
                x_walls=[0.44 * PX], y_walls=[0.58 * PX])
    return out(s.set_source(WL, theta=0.2, phi=0.9).solve())


def f_pl_per_layer_modes():
    s = st(layer_grids="per-layer")
    s.add_layer(0.15e-6, eps_cell=cell(3, 3, 2.25, 6.0, ((1, 2), (1, 2))),
                x_walls=[0.21 * PX, 0.55 * PX],
                y_walls=[0.33 * PX, 0.78 * PX], n_modes=6)
    s.add_layer(0.12e-6, eps_cell=cell(3, 3, 2.25, 4.0, ((1, 2), (0, 2))),
                x_walls=[0.3117 * PX, 0.7402 * PX],
                y_walls=[0.29 * PX, 0.61 * PX], n_modes=4)
    return out(s.set_source(WL, theta=0.13, phi=0.2).solve())


def f_pl_taper():
    s = st(layer_grids="per-layer", n_modes=4)
    s.add_tapered_pillar(0.20e-6, eps_pillar=6.0, eps_host=2.25,
                         x_bounds_bottom=(0.22 * PX, 0.70 * PX),
                         y_bounds_bottom=(0.25 * PX, 0.68 * PX),
                         x_bounds_top=(0.33 * PX, 0.59 * PX),
                         y_bounds_top=(0.36 * PX, 0.57 * PX),
                         n_slices=5)
    return out(s.set_source(WL, theta=0.1).solve())


def f_pl_tapers():
    s = st(layer_grids="per-layer", n_modes=4)
    s.add_tapered_pillars(
        0.18e-6,
        pillars=[((0.5 * PX, 0.5 * PX), (0.24 * PX, 0.22 * PX),
                  (0.38 * PX, 0.36 * PX), 7.0)],
        eps_host=2.25, n_slices=6)
    return out(s.set_source(WL, theta=0.07, phi=0.4).solve())


def f_pl_taper_bottom_rule():
    s = st(layer_grids="per-layer", n_modes=4)
    s.add_tapered_pillar(0.15e-6, eps_pillar=5.0, eps_host=2.25,
                         x_bounds_bottom=(0.20 * PX, 0.72 * PX),
                         y_bounds_bottom=(0.24 * PX, 0.70 * PX),
                         x_bounds_top=(0.30 * PX, 0.62 * PX),
                         y_bounds_top=(0.34 * PX, 0.60 * PX),
                         n_slices=4, rule="bottom")
    return out(s.set_source(WL, theta=0.06).solve())


def f_pl_tensor_oop():
    e = np.array([[4.0, 0.0, 0.8],
                  [0.0, 3.4, 0.0],
                  [0.75, 0.0, 3.2]], dtype=complex)
    c = np.empty((3, 3, 3, 3), dtype=complex)
    c[...] = np.eye(3) * 2.25
    c[1, 1] = e
    s = st(layer_grids="per-layer", n_modes=4)
    s.add_layer(0.14e-6, eps_cell=c, x_walls=[0.27 * PX, 0.63 * PX],
                y_walls=[0.31 * PX, 0.69 * PX])
    s.add_layer(0.10e-6, eps=2.2)
    return out(s.set_source(WL, theta=0.09, phi=0.3).solve())


def f_pl_slant():
    s = st(layer_grids="per-layer", n_modes=4)
    s.add_layer(0.13e-6, eps_cell=cell(3, 3, 2.25, 5.0, ((1, 2), (1, 2))),
                x_walls=[0.26 * PX, 0.64 * PX],
                y_walls=[0.30 * PX, 0.70 * PX], slant=(0.07, 0.03))
    s.add_layer(0.10e-6, eps=2.0)
    return out(s.set_source(WL, theta=0.05).solve())


def f_pl_magnetic():
    s = st(layer_grids="per-layer", n_modes=4)
    s.add_layer(0.15e-6,
                eps_cell=cell(3, 3, 2.25, 5.0, ((1, 2), (1, 2))),
                mu_cell=cell(3, 3, 1.0, 1.6, ((1, 2), (1, 2))),
                x_walls=[0.29 * PX, 0.67 * PX],
                y_walls=[0.29 * PX, 0.67 * PX])
    s.add_layer(0.09e-6, eps=2.0)
    return out(s.set_source(WL, theta=0.08).solve())


def f_pl_retain_internal():
    s = st(layer_grids="per-layer", n_modes=4)
    s.add_layer(0.13e-6,
                eps_cell=cell(3, 3, 2.25, 6.0 + 0.4j, ((1, 2), (1, 2))),
                x_walls=[0.24 * PX, 0.61 * PX],
                y_walls=[0.28 * PX, 0.66 * PX])
    s.add_layer(0.11e-6, eps=2.4)
    res = s.set_source(WL, theta=0.15).solve(retain_internal=True)
    d = out(res)
    d["absorption"] = np.asarray(s.layer_absorption(), dtype=float)
    return d


def f_pl_uniform_array_spelling():
    """The SAME uniform lattice spelled as an explicit ARRAY -- this is the
    arm the D3 formula CAN move (the integer spelling does not reach it)."""
    s = st(layer_grids="per-layer", n_modes=5)
    s.add_layer(0.16e-6, eps_cell=cell(3, 3, 2.25, 6.0, ((1, 2), (1, 2))),
                x_walls=[PX / 3.0, 2.0 * PX / 3.0],
                y_walls=[PX / 3.0, 2.0 * PX / 3.0])
    s.add_layer(0.10e-6, eps=2.1)
    return out(s.set_source(WL, theta=0.21, phi=0.6).solve())


def f_pl_integer_spelling():
    """The same device on the INTEGER lattice (no x_walls) -- exempt."""
    s = st(layer_grids="per-layer", n_modes=5)
    s.add_layer(0.16e-6, eps_cell=cell(3, 3, 2.25, 6.0, ((1, 2), (1, 2))))
    s.add_layer(0.10e-6, eps=2.1)
    return out(s.set_source(WL, theta=0.21, phi=0.6).solve())


# ================================================================= 1-D arms
def f_1d_shared():
    s = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    s.add_layer(0.2e-6, eps=2.1)
    s.add_layer(0.3e-6, segments=[(0.42, 6.0), (0.58, 2.25)])
    return out(s.set_source(0.6e-6, angle=0.22).solve())


def f_1d_perlayer():
    s = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9,
                 layer_grids="per-layer")
    s.add_layer(0.3e-6, segments=[(0.42, 6.0), (0.58, 2.25)])
    s.add_layer(0.25e-6, segments=[(0.31, 4.0), (0.69, 2.25)])
    return out(s.set_source(0.6e-6, angle=0.18).solve())


def f_1d_conical():
    s = PMMStack(0.8e-6, n_substrate=1.5, degree=10, far_field_orders=9)
    s.add_layer(0.3e-6, segments=[(0.42, 6.0), (0.58, 2.25)])
    return out(s.set_source(0.6e-6, theta=0.25, phi=0.6).solve())


def f_1d_slanted():
    s = PMMStack(0.8e-6, n_substrate=1.5, degree=9, far_field_orders=7)
    s.add_layer(0.25e-6, segments=[(0.4, 5.0), (0.6, 2.25)],
                slant_angle=0.15)
    return out(s.set_source(0.6e-6, angle=0.1).solve())


# ======================================== the PROJECTOR, directly
def _proj_hashes(prefix, mk):
    from lumenairy.elements.pmm.twod_staggered import _stag_fourier_projection
    d = {}
    for (per, N, M, a0, mmax) in [(1.2, 3, 5, 0.0, 4), (0.9, 4, 4, 2.08, 5),
                                  (1.4, 6, 6, -3.1, 7), (0.7, 2, 8, 1.7, 6),
                                  (1.0, 12, 3, 0.4, 8), (1.0, 1, 7, 5.5, 2),
                                  (1.1, 5, 6, -1.3, 6), (0.85, 9, 4, 0.9, 5)]:
        b = mk(per, N, M)
        orders = np.arange(-mmax, mmax + 1)
        # _stag_fourier_projection returns a CLOSURE; the arrays are what it
        # produces on the basis's two global stencil sets.
        asm = _stag_fourier_projection(b, orders, a0)
        for tag, gs in (("B", b.B), ("Bt", b.Btilde)):
            d[f"{prefix}_{per}_{N}_{M}_{mmax}_{tag}"] = np.asarray(asm(gs))
    return d


def f_projector_integer():
    from lumenairy.elements.pmm.twod_staggered import Basis1D
    return _proj_hashes("int", lambda per, N, M: Basis1D(per, N, M))


def f_projector_uniform_array():
    from lumenairy.elements.pmm.twod_staggered import Basis1D
    return _proj_hashes(
        "arr", lambda per, N, M: Basis1D(per, np.linspace(0.0, per, N + 1), M))


FIXTURES = [
    ("shared_scalar", f_shared_scalar),
    ("shared_scalar_normal", f_shared_scalar_normal),
    ("shared_tensor_inplane", f_shared_tensor_inplane),
    ("shared_tensor_oop", f_shared_tensor_oop),
    ("shared_magnetic", f_shared_magnetic),
    ("shared_slant", f_shared_slant),
    ("shared_retain_internal", f_shared_retain_internal),
    ("shared_modal_amps", f_shared_modal_amps),
    ("shared_jones_false", f_shared_jones_false),
    ("shared_M6_N4", f_shared_M6_N4),
    ("pl_conforming", f_pl_conforming),
    ("pl_nonconforming", f_pl_nonconforming),
    ("pl_nested", f_pl_nested),
    ("pl_mixed_uniform_patterned", f_pl_mixed_uniform_patterned),
    ("pl_per_layer_modes", f_pl_per_layer_modes),
    ("pl_taper", f_pl_taper),
    ("pl_tapers", f_pl_tapers),
    ("pl_taper_bottom_rule", f_pl_taper_bottom_rule),
    ("pl_tensor_oop", f_pl_tensor_oop),
    ("pl_slant", f_pl_slant),
    ("pl_magnetic", f_pl_magnetic),
    ("pl_retain_internal", f_pl_retain_internal),
    ("pl_uniform_array_spelling", f_pl_uniform_array_spelling),
    ("pl_integer_spelling", f_pl_integer_spelling),
    ("d1_shared", f_1d_shared),
    ("d1_perlayer", f_1d_perlayer),
    ("d1_conical", f_1d_conical),
    ("d1_slanted", f_1d_slanted),
    ("projector_integer", f_projector_integer),
    ("projector_uniform_array", f_projector_uniform_array),
]

if __name__ == "__main__":
    only = sys.argv[1:]
    for name, fn in FIXTURES:
        if only and name not in only:
            continue
        try:
            run(name, fn)
        except Exception as exc:                        # noqa: BLE001
            OUT[name] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"  [{TAG}] {name}: ERROR {type(exc).__name__}: {exc}",
                  flush=True)
    here = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(here, f"v1_bitid_{TAG}.json")
    with open(p, "w") as fh:
        json.dump({"tag": TAG, "lumenairy": lumenairy.__file__,
                   "version": lumenairy.__version__,
                   "numpy": np.__version__,
                   "python": sys.version.split()[0],
                   "fixtures": OUT}, fh, indent=1)
    nh = sum(len(v.get("hashes", {})) for v in OUT.values())
    ne = sum(1 for v in OUT.values() if "error" in v)
    print(f"[{TAG}] {len(OUT)} fixtures, {nh} hashes, {ne} errors -> {p}")
