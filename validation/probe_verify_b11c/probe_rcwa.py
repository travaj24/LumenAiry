"""VERIFY-B11c bit identity, arm 1: everything that runs under the BLAS-thread
cap -- the RCWA 1-D / 2-D / anisotropic engines, the stack sweeps (serial and
threaded), and the PMM 1-D / 2-D consumers of ``_blas_limit`` /
``_blas_threads_quiet`` -- plus the cap's own public surface.

Fixtures are this verification's own: a different period, a different material
set, different truncations, different incidence angles and a different sweep
grid from ``validation/probe_wp_b11c/probe_item1_rcwa.py``, so an accidental
agreement on one grating cannot carry the claim.

argv: <tree-root> <output-json>.  Run by ``run_bitid.py``, never under pytest.
"""
# ruff: noqa: E402, I001 -- the tree is bound before the library is imported.
from __future__ import annotations

import os
import sys
import warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
TREE = os.path.abspath(sys.argv[1])
OUT = sys.argv[2]
sys.path.insert(0, TREE)

import vlib

la = vlib.anchor(TREE)

import numpy as np

from lumenairy.elements import rcwa as _rcwa_pkg
from lumenairy.elements.rcwa import _core as _core

P = vlib.Probe()
WL = 0.78e-6
LAM = 0.62e-6                      # period, different from the package probe


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------------------------------------------------
# A. The cap's public surface (identity, ordering, the knob, the floor)
# ---------------------------------------------------------------------------
_SEVEN = ("_BLAS_STATE", "_get_blas_threads", "set_blas_threads",
          "rcwa_blas_threads", "_blas_threads_quiet", "_blas_limit",
          "_with_blas_limit")

P.add("A01_core_all_ordered", list(_core.__all__))
P.add("A02_core_all_set", sorted(set(_core.__all__)))
P.add("A03_all_resolve", sorted(n for n in _core.__all__ if hasattr(_core, n)))
P.add("A04_all_missing", sorted(n for n in _core.__all__
                                if not hasattr(_core, n)))
P.add("A05_identity_triple", [
    (n,
     getattr(_core, n) is getattr(_rcwa_pkg, n, None),
     getattr(_core, n) is getattr(la, n, None))
    for n in _SEVEN])


def _set_then_read(v):
    la.set_blas_threads(v)
    return _core._get_blas_threads()


P.add("A06_knob_roundtrip", [_set_then_read(v)
                             for v in (None, 1, 4, 0, -3, 7, None)])
P.add("A07_floor", [_set_then_read(v) for v in (0, -3, 2, 9)])
la.set_blas_threads(None)
with la.rcwa_blas_threads(3):
    _inside = _core._get_blas_threads()
P.add("A08_scoped", (_inside, _core._get_blas_threads()))
with _core._blas_threads_quiet(5):
    _inside_q = _core._get_blas_threads()
P.add("A09_quiet", (_inside_q, _core._get_blas_threads()))
_deco = la.rcwa_efficiency_1d
P.add("A10_wraps_meta", (_deco.__name__, _deco.__doc__ is not None,
                         _deco.__module__, hasattr(_deco, "__wrapped__"),
                         str(__import__("inspect").signature(_deco))))
P.call("A11_set_bad", la.set_blas_threads, "seven")
la.set_blas_threads(None)


# ---------------------------------------------------------------------------
# B. RCWA 1-D, every formulation / polarization / incidence the cap touches
# ---------------------------------------------------------------------------
E1 = la.rcwa_efficiency_1d
P.call("B01_1d_te", E1, LAM, 2.05, 1.0, 1.46, 1.0, 0.31e-6, 0.42, WL,
       polarization="te", n_orders=17)
P.call("B02_1d_tm", E1, LAM, 2.05, 1.0, 1.46, 1.0, 0.31e-6, 0.42, WL,
       polarization="tm", n_orders=17)
P.call("B03_1d_te_oblique", E1, LAM, 2.05, 1.0, 1.46, 1.0, 0.31e-6, 0.42, WL,
       polarization="te", angle=0.273, n_orders=17)
P.call("B04_1d_tm_metal_li", E1, LAM, 0.27 + 3.41j, 1.0, 1.46, 1.0,
       0.11e-6, 0.55, WL, polarization="tm", formulation="li", n_orders=21)
P.call("B05_1d_tm_metal_laurent", E1, LAM, 0.27 + 3.41j, 1.0, 1.46, 1.0,
       0.11e-6, 0.55, WL, polarization="tm", formulation="laurent",
       n_orders=21)
P.call("B06_1d_stabilize", E1, LAM, 2.05, 1.0, 1.46, 1.0, 1.9e-6, 0.42, WL,
       polarization="tm", n_orders=15, stabilize=True)
P.call("B07_1d_deep_te", E1, LAM, 2.05, 1.0, 1.46, 1.0, 1.9e-6, 0.42, WL,
       polarization="te", n_orders=15)
_EPSD = np.diag([4.2 + 0.0j, 2.6 + 0.0j, 3.1 + 0.0j])
P.call("B08_jones_1d", la.rcwa_jones_1d, LAM, _EPSD,
       np.eye(3, dtype=complex), 1.46, 1.0,
       0.31e-6, 0.42, WL, n_orders=13, angle=0.19)
P.call("B09_jones_1d_t", la.rcwa_jones_1d, LAM, _EPSD,
       np.eye(3, dtype=complex), 1.46, 1.0,
       0.31e-6, 0.42, WL, n_orders=13, angle=0.19,
       return_jones_transmission=True)
P.call("B10_1d_vs_wl", la.rcwa_efficiency_vs_wavelength, LAM, 2.05, 1.0, 1.46,
       1.0, 0.31e-6, 0.42, np.linspace(0.76e-6, 0.80e-6, 4), order=0,
       n_orders=11)
P.call("B11_extrapolate", la.rcwa_extrapolate,
       [0.311, 0.3155, 0.31702, 0.317501], n_orders=[7, 11, 15, 19])


# ---------------------------------------------------------------------------
# C. RCWA 2-D, crossed / conical / shapes / full anisotropic tensor
# ---------------------------------------------------------------------------
def _cell(n=14, hi=6.1, lo=1.0):
    c = np.full((n, n), lo, dtype=complex)
    c[3:11, 4:10] = hi
    c[0:2, 0:2] = 2.3
    return c


def _tensor_cell(n=10):
    t = np.zeros((n, n, 3, 3), dtype=complex)
    for i in range(n):
        for j in range(n):
            inside = (3 <= i < 8) and (2 <= j < 7)
            t[i, j] = np.diag([5.2, 3.4, 4.1] if inside else [1.0, 1.0, 1.0])
            if inside:
                t[i, j, 0, 1] = 0.35
                t[i, j, 1, 0] = 0.35
    return t


P.call("C01_2d_normal", la.rcwa_efficiency_2d, LAM, 0.58e-6, _cell(), 1.46,
       1.0, 0.24e-6, WL, n_orders_x=3, n_orders_y=3, polarization="te")
P.call("C02_2d_conical", la.rcwa_efficiency_2d, LAM, 0.58e-6, _cell(), 1.46,
       1.0, 0.24e-6, WL, theta=0.31, phi=0.65, n_orders_x=3, n_orders_y=3,
       polarization="tm")
P.call("C03_2d_circular_trunc", la.rcwa_efficiency_2d, LAM, 0.58e-6, _cell(),
       1.46, 1.0, 0.24e-6, WL, n_orders_x=3, n_orders_y=3,
       truncation="circular", polarization="te")
P.call("C04_2d_shapes", la.rcwa_efficiency_2d_shapes, LAM, 0.58e-6, 1.0 + 0j,
       [{"shape": "ellipse", "center": (0.05e-6, -0.03e-6),
         "semi_axes": (0.13e-6, 0.09e-6), "eps": 6.1 + 0j},
        {"shape": "rectangle", "center": (-0.2e-6, 0.18e-6),
         "size": (0.09e-6, 0.07e-6), "eps": 3.4 + 0.05j}],
       1.46, 1.0, 0.24e-6, WL, n_orders_x=3, n_orders_y=3)
P.call("C05_jones_2d_tensor", la.rcwa_jones_2d, LAM, 0.58e-6, _tensor_cell(),
       1.46, 1.0, 0.24e-6, WL, n_orders_x=2, n_orders_y=2)
P.call("C06_jones_2d_conical", la.rcwa_jones_2d, LAM, 0.58e-6, _tensor_cell(),
       1.46, 1.0, 0.24e-6, WL, theta=0.23, phi=0.42, n_orders_x=2,
       n_orders_y=2)


def _stack():
    st = la.RCWAStack(period=LAM, period_y=0.58e-6, n_superstrate=1.0,
                      n_substrate=1.46, n_orders=2, n_orders_y=2)
    st.add_layer(0.06e-6, eps=2.9)
    st.add_layer(0.18e-6, eps_cell=_cell(12))
    st.add_layer(0.05e-6, eps=1.7 + 0.02j)
    st.set_source(WL, theta=0.19, phi=0.73)
    return st


_res = _quiet(_stack().solve)
P.add("C07_stack_eff", _res.efficiencies())
P.add("C08_stack_jones", (_res.jones_reflection(), _res.jones_transmission()))
P.add("C09_stack_abs", _res.absorptance())

_WLS = np.linspace(0.77e-6, 0.79e-6, 3)
_serial = _quiet(_stack().solve_vs_wavelength, _WLS, max_workers=1)
_thread = _quiet(_stack().solve_vs_wavelength, _WLS, max_workers=3)
P.add("C10_sweep_serial", _serial)
P.add("C11_sweep_threaded", _thread)
P.add("C12_sweep_equal", vlib.digest(_serial) == vlib.digest(_thread))


# ---------------------------------------------------------------------------
# D. The same arithmetic INSIDE and OUTSIDE an explicit cap -- the only place a
#    BLAS-thread change can reach a number.
# ---------------------------------------------------------------------------
def _ref():
    return E1(LAM, 2.05, 1.0, 1.46, 1.0, 0.31e-6, 0.42, WL,
              polarization="tm", angle=0.23, n_orders=19)


P.add("D01_uncapped", _ref())
with la.rcwa_blas_threads(1):
    P.add("D02_cap1", _ref())
    P.add("D02b_cap1_stack", _quiet(_stack().solve).efficiencies())
with la.rcwa_blas_threads(4):
    P.add("D03_cap4", _ref())
with _core._blas_threads_quiet(2):
    P.add("D04_quiet2", _ref())
with la.rcwa_blas_threads(None):
    P.add("D05_capNone", _ref())
P.add("D06_uncapped_again", _ref())
la.set_blas_threads(2)
P.add("D07_global_cap2", _ref())
P.add("D07b_global_cap2_2d", _quiet(
    la.rcwa_efficiency_2d, LAM, 0.58e-6, _cell(), 1.46, 1.0, 0.24e-6, WL,
    n_orders_x=3, n_orders_y=3, polarization="te"))
la.set_blas_threads(None)
P.add("D08_global_none", _ref())
P.add("D09_limit_callable", callable(_core._blas_limit))


# ---------------------------------------------------------------------------
# E. The PMM consumers of the cap
# ---------------------------------------------------------------------------
P.call("E01_pmm1d_te", la.pmm_efficiency_1d, LAM, 2.05, 1.0, 1.46, 1.0,
       0.31e-6, 0.42, WL, polarization="te", degree=9, far_field_orders=7)
P.call("E02_pmm1d_tm", la.pmm_efficiency_1d, LAM, 2.05, 1.0, 1.46, 1.0,
       0.31e-6, 0.42, WL, polarization="tm", degree=9, far_field_orders=7)
P.call("E03_pmm1d_oblique", la.pmm_efficiency_1d, LAM, 2.05, 1.0, 1.46, 1.0,
       0.31e-6, 0.42, WL, polarization="tm", angle=0.35, degree=9,
       far_field_orders=7)
P.call("E04_pmm_jones_1d", la.pmm_jones_1d, LAM, _EPSD,
       np.eye(3, dtype=complex), 1.46, 1.0, 0.31e-6, 0.42, WL, degree=9,
       far_field_orders=7, angle=0.14)


def _pmm_stack():
    s = la.PMMStack(LAM, n_substrate=1.46, degree=7, far_field_orders=5)
    s.add_layer(0.16e-6, segments=[(0.42, 4.2 + 0j), (0.58, 1.0 + 0j)])
    s.add_layer(0.07e-6, eps=2.9 + 0j)
    return s


_pw = np.linspace(0.77e-6, 0.79e-6, 3)
_ps = _pmm_stack()
_quiet(_ps.set_source, float(_pw[0]), theta=0.0)
P.add("E05_pmm_sweep_serial", _quiet(_ps.solve_vs_wavelength, _pw, theta=0.0,
                                     max_workers=1))
P.add("E06_pmm_sweep_threaded", _quiet(_pmm_stack().solve_vs_wavelength, _pw,
                                       theta=0.0, max_workers=3))


def _hybrid():
    hy = la.PMM2DStackHybrid(LAM, 0.58e-6, n_substrate=1.46, n_superstrate=1.0,
                             n_orders=1, degree=5)
    hy.add_layer(0.09e-6, eps_cell=_cell(10))
    return hy


P.add("E07_pmm2d_hybrid_sweep", _quiet(_hybrid().solve_vs_wavelength, _pw,
                                       theta=0.0, max_workers=2))
P.add("E08_pmm2d_hybrid_jones", _quiet(_hybrid().solve_vs_wavelength, _pw,
                                       theta=0.0, jones=True, max_workers=1))
P.call("E09_pmm2d_cell", la.pmm_efficiency_2d_cell, LAM, 0.58e-6, _cell(10),
       1.46, 1.0, 0.09e-6, WL, degree=5, n_orders=1)
with la.rcwa_blas_threads(1):
    P.add("E10_pmm2d_cell_capped", _quiet(
        la.pmm_efficiency_2d_cell, LAM, 0.58e-6, _cell(10), 1.46, 1.0,
        0.09e-6, WL, degree=5, n_orders=1))

P.write(OUT)
