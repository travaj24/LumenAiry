"""WP-B11c item 1 gate: the RCWA / PMM answers the ``_blas`` split must not move.

Covers the three surfaces the split touches:

* the seven re-exported names themselves -- their identity through ``_core``,
  the package facade and ``lumenairy``, the knob round trip, and the
  once-per-process inert-cap warning with its exact text;
* the RCWA engines that run UNDER the cap decorator (1-D TE/TM/metal, 1-D
  Jones, 2-D crossed, 2-D shapes, the JAX-free stack, a threaded and a serial
  ``solve_vs_wavelength``, and the same solve inside an explicit
  ``rcwa_blas_threads`` scope, which is the only path whose arithmetic the cap
  could reach);
* the PMM consumers of ``_blas_threads_quiet`` / ``_blas_limit``
  (``pmm_efficiency_1d``, ``pmm_jones_1d``, ``PMMStack.solve_vs_wavelength``,
  the 2-D hybrid stack sweep).

Run through ``bi.py``; never under pytest.
"""
# ruff: noqa: E402, I001 -- a probe BINDS its tree before it imports the
# library, so the imports cannot be hoisted above ``bind()``.
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, ".")
from probelib import bind, caught, emit, h            # noqa: E402

bind(sys.argv[1])

import numpy as np                                     # noqa: E402

import lumenairy as L                                  # noqa: E402
from lumenairy.elements import rcwa as RC              # noqa: E402
from lumenairy.elements import pmm as PM               # noqa: E402
from lumenairy.elements.rcwa import _core              # noqa: E402

R = {}
NM = 1e-9

# --------------------------------------------------------------------------
# A. The re-exported surface itself
# --------------------------------------------------------------------------
NAMES = ("_BLAS_STATE", "_get_blas_threads", "set_blas_threads",
         "rcwa_blas_threads", "_blas_threads_quiet", "_blas_limit",
         "_with_blas_limit")
R["A1_all_list"] = h(list(_core.__all__))
R["A2_all_set"] = h(sorted(set(_core.__all__)))
R["A3_core_has_all"] = h([n for n in _core.__all__ if not hasattr(_core, n)])
R["A4_facade_identity"] = h([getattr(RC, n) is getattr(_core, n)
                             for n in NAMES])
R["A5_toplevel_names"] = h([hasattr(L, n) for n in
                            ("set_blas_threads", "rcwa_blas_threads")])
R["A6_knob_roundtrip"] = h([
    _core._get_blas_threads(),
    [(_core._get_blas_threads()) for _ in [L.override(blas_threads=3).__enter__()]],
])
_core.set_blas_threads(None)
R["A7_cap_roundtrip"] = h(caught(lambda: [
    _core._get_blas_threads(),
    [_core.set_blas_threads(2), _core._get_blas_threads()][1],
    [_core.set_blas_threads(None), _core._get_blas_threads()][1],
]))


def _scoped():
    with _core.rcwa_blas_threads(4):
        inner = _core._get_blas_threads()
    return inner, _core._get_blas_threads()


R["A8_scoped"] = h(caught(_scoped))


def _quiet():
    with _core._blas_threads_quiet(1):
        inner = _core._get_blas_threads()
    return inner, _core._get_blas_threads()


R["A9_quiet_scope"] = h(caught(_quiet))
R["A10_with_blas_limit_wraps"] = h(caught(
    lambda: (lambda f: (f.__name__, f.__doc__))(
        _core._with_blas_limit(lambda a, b=2: a + b))))
R["A11_limit_noop_type"] = h(str(type(_core._blas_limit()).__name__))
R["A12_decorated_answer"] = h(caught(
    _core._with_blas_limit(lambda a, b: a * b), 6, 7))
# The measured floor at n <= 0 and the int coercion (max(1, int(n))).
R["A13_cap_floor"] = h(caught(lambda: [
    [_core.set_blas_threads(v), _core._get_blas_threads()][1]
    for v in (0, -3, 2.9, 7)]))
_core.set_blas_threads(None)

# --------------------------------------------------------------------------
# B. RCWA engines under the decorator
# --------------------------------------------------------------------------
def _eps_cell(n=16, hi=6.25):
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    return np.where((np.abs(xx - 0.5) < 0.23) & (np.abs(yy - 0.5) < 0.31),
                    hi + 0j, 1.0 + 0j)


R["B1_1d_te"] = h(RC.rcwa_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                        0.2e-6, 0.45, 0.55e-6, n_orders=9))
R["B2_1d_tm"] = h(RC.rcwa_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                        0.2e-6, 0.45, 0.55e-6, n_orders=9,
                                        polarization="tm"))
R["B3_1d_tm_metal"] = h(RC.rcwa_efficiency_1d(
    0.4e-6, 0.2 + 3.4j, 1.0, 1.5, 1.0, 0.12e-6, 0.5, 0.633e-6,
    n_orders=11, polarization="tm", formulation="li"))
R["B4_1d_oblique"] = h(RC.rcwa_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                             0.2e-6, 0.45, 0.55e-6,
                                             n_orders=9, theta=0.31,
                                             polarization="tm"))
R["B5_1d_laurent"] = h(RC.rcwa_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                             0.2e-6, 0.45, 0.55e-6,
                                             n_orders=9, polarization="tm",
                                             formulation="laurent"))
R["B6_jones_1d"] = h(RC.rcwa_jones_1d(0.5e-6, 6.25 + 0j, 1.0 + 0j, 1.5, 1.0,
                                      0.2e-6, 0.45, 0.55e-6, n_orders=9))
R["B6b_jones_1d_segments"] = h(RC.rcwa_jones_1d_segments(
    0.5e-6, RC.binary_grating_segments(0.45, 6.25 + 0j, 1.0 + 0j),
    1.5, 1.0, 0.2e-6, 0.55e-6, n_orders=9))
R["B7_2d"] = h(RC.rcwa_efficiency_2d(0.5e-6, 0.5e-6, _eps_cell(), 1.5, 1.0,
                                     0.15e-6, 0.55e-6, n_orders_x=3,
                                     n_orders_y=3))
R["B8_2d_oblique"] = h(RC.rcwa_efficiency_2d(0.5e-6, 0.5e-6, _eps_cell(),
                                             1.5, 1.0, 0.15e-6, 0.55e-6,
                                             n_orders_x=3, n_orders_y=3,
                                             theta=0.2, phi=0.4))
R["B9_2d_shapes"] = h(RC.rcwa_efficiency_2d_shapes(
    0.5e-6, 0.5e-6, 1.0 + 0j,
    [{"shape": "disk", "center": (0.25e-6, 0.25e-6), "radius": 0.11e-6,
      "eps": 6.25 + 0j}],
    1.5, 1.0, 0.15e-6, 0.55e-6, n_orders_x=3, n_orders_y=3))
def _eps_tensor_cell(n=12):
    c = _eps_cell(n)
    t = np.zeros(c.shape + (3, 3), dtype=complex)
    t[..., 0, 0] = c
    t[..., 1, 1] = c * 1.04
    t[..., 2, 2] = c
    return t


R["B10_jones_2d"] = h(RC.rcwa_jones_2d(0.5e-6, 0.5e-6, _eps_tensor_cell(),
                                       1.5, 1.0, 0.15e-6, 0.55e-6,
                                       n_orders_x=2, n_orders_y=2))
R["B11_vs_wavelength"] = h(RC.rcwa_efficiency_vs_wavelength(
    0.5e-6, 2.5, 1.0, 1.5, 1.0, 0.2e-6, 0.45,
    np.linspace(0.54e-6, 0.56e-6, 4), n_orders=7))


def _stack(nx=2, ny=2):
    st = RC.RCWAStack(period=0.5e-6, period_y=0.5e-6, n_superstrate=1.0,
                      n_substrate=1.5, n_orders=nx, n_orders_y=ny)
    st.add_layer(0.08e-6, eps=2.25)
    st.add_layer(0.12e-6, eps_cell=_eps_cell(12))
    st.set_source(0.55e-6, theta=0.1, phi=0.2)
    return st


res = _stack().solve()
R["B12_stack_solve"] = h(res.efficiencies())
R["B12b_stack_jones"] = h((res.jones_reflection(),
                           res.jones_transmission()))
R["B12c_stack_abs"] = h(res.absorptance())
wls = np.linspace(0.54e-6, 0.56e-6, 4)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    o1, R1, T1, J1 = _stack().solve_vs_wavelength(wls, max_workers=1)
    o4, R4, T4, J4 = _stack().solve_vs_wavelength(wls, max_workers=4)
R["B13_sweep_serial"] = h((o1, R1, T1, J1))
R["B14_sweep_threaded"] = h((o4, R4, T4, J4))
R["B15_sweep_agrees"] = h([bool(np.array_equal(R1, R4)),
                           bool(np.array_equal(T1, T4))])
# The only arithmetic the cap could reach: a solve inside an explicit scope.
with _core.rcwa_blas_threads(1):
    R["B16_capped_1d"] = h(RC.rcwa_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                                 0.2e-6, 0.45, 0.55e-6,
                                                 n_orders=9))
    stc = _stack()
    R["B17_capped_stack"] = h(stc.solve().efficiencies())
with _core._blas_threads_quiet(2):
    R["B18_quiet_1d"] = h(RC.rcwa_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                                0.2e-6, 0.45, 0.55e-6,
                                                n_orders=9))
R["B19_uncapped_again"] = h([
    _core._get_blas_threads(),
    h(RC.rcwa_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0, 0.2e-6, 0.45,
                            0.55e-6, n_orders=9)),
])
R["B20_extrapolate"] = h(caught(
    RC.rcwa_extrapolate, [0.10, 0.11, 0.111], n_orders=[5, 7, 9]))

# --------------------------------------------------------------------------
# C. PMM consumers of the cap
# --------------------------------------------------------------------------
segs = [(0.45, 6.25 + 0j), (0.55, 1.0 + 0j)]
R["C1_pmm_1d_te"] = h(PM.pmm_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                           0.2e-6, 0.45, 0.55e-6,
                                           degree=10, far_field_orders=7))
R["C2_pmm_1d_tm"] = h(PM.pmm_efficiency_1d(0.5e-6, 2.5, 1.0, 1.5, 1.0,
                                           0.2e-6, 0.45, 0.55e-6,
                                           degree=10, far_field_orders=7,
                                           polarization="tm"))
_eps_r = np.diag([6.25 + 0j, 6.50 + 0j, 6.25 + 0j])
_eps_g = np.eye(3, dtype=complex)
R["C3_pmm_jones_1d"] = h(PM.pmm_jones_1d(0.5e-6, _eps_r, _eps_g, 1.5,
                                         1.0, 0.2e-6, 0.45, 0.55e-6,
                                         degree=10, far_field_orders=7))


def _pmm_stack():
    s = PM.PMMStack(0.5e-6, n_substrate=1.5, degree=8, far_field_orders=5)
    s.add_layer(0.2e-6, segments=segs)
    s.add_layer(0.1e-6, eps=2.25 + 0j)
    return s


pw = np.linspace(0.54e-6, 0.56e-6, 3)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    ps = _pmm_stack()
    ps.set_source(float(pw[0]), theta=0.0)
    R["C4_pmm_sweep_serial"] = h(ps.solve_vs_wavelength(pw, theta=0.0,
                                                        max_workers=1))
    ps2 = _pmm_stack()
    ps2.set_source(float(pw[0]), theta=0.0)
    R["C5_pmm_sweep_threaded"] = h(ps2.solve_vs_wavelength(pw, theta=0.0,
                                                           max_workers=4))
    hy = PM.PMM2DStackHybrid(0.5e-6, 0.5e-6, n_substrate=1.5,
                             n_superstrate=1.0, n_orders=1, degree=5)
    hy.add_layer(0.1e-6, eps_cell=_eps_cell(12))
    R["C6_pmm2d_sweep"] = h(hy.solve_vs_wavelength(pw, theta=0.0,
                                                   max_workers=2))
R["C7_pmm_2d_cell"] = h(PM.pmm_efficiency_2d_cell(0.5e-6, 0.5e-6,
                                                  _eps_cell(12), 1.5, 1.0,
                                                  0.1e-6, 0.55e-6, degree=5,
                                                  n_orders=1))

emit(R)
