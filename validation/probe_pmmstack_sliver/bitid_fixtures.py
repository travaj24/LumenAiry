"""BIT-IDENTITY fixture set for the O-11 sliver guard.

Eighteen ``PMMStack`` solves drawn from the shipped test files
(``test_v5_11_0_pmm_stack``, ``test_v5_13_0_pmm_tapered``,
``test_pmm_per_layer_grids``, ``test_pmm_m2_window_contract``,
``test_v5_20_0_pmm_stack_conical``, ``test_v5_14_3_pmm_internal_field``,
``test_v5_21_pmm_threaded_sweep``, ``test_niche_audit_w9_pmm_taper``), covering
the shared union grid, the per-layer window grids at halfwidth 1 and 2, the
tapered staircase with the wall-snap DORMANT and ACTIVE, conical, slant,
out-of-plane tensors, a lossy layer, the wavelength sweep, ``prepare()``,
``stabilize='slices'``, ``retain_internal`` + ``internal_field`` and
``layer_absorption``.

Public API only, so the same file runs against the worktree and against the
read-only main clone.  Every fixture returns a tuple of arrays; ``p10_bitid.py``
hashes their raw bytes.
"""
import numpy as np

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import uniaxial_tensor

_C = np.complex128
GR = np.eye(3, dtype=_C)
WL = 0.55e-6
P = 0.8e-6
LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.4)


def _t(*arrays):
    return tuple(np.ascontiguousarray(np.asarray(a)) for a in arrays)


def f01_one_layer_normal():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=20)
    st.add_layer(0.5e-6, segments=[(0.5, LC), (0.5, GR)])
    o, R, T, J = st.set_source(WL, angle=0.0).solve()
    return _t(o, R, T, J)


def f02_one_layer_oblique():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=20)
    st.add_layer(0.5e-6, segments=[(0.5, LC), (0.5, GR)])
    o, R, T, J = st.set_source(WL, angle=np.radians(25.0)).solve()
    return _t(o, R, T, J)


def f03_two_layer_tensor_oblique():
    lcB = uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.3)
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=14)
    st.add_layer(0.2e-6, segments=[(0.25, GR), (0.5, 2.5 * GR), (0.25, GR)])
    st.add_layer(0.3e-6, segments=[(0.1, 2.0 * GR), (0.4, lcB),
                                   (0.5, 2.0 * GR)])
    o, R, T, J = st.set_source(WL, angle=np.radians(15.0)).solve()
    return _t(o, R, T, J)


def f04_bragg_abab():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    for _ in range(3):
        st.add_layer(0.12e-6, segments=[(0.5, 6.0 * GR), (0.5, GR)])
        st.add_layer(0.12e-6, segments=[(0.5, 2.1 * GR), (0.5, GR)])
    o, R, T, J = st.set_source(WL, angle=np.radians(10.0)).solve()
    return _t(o, R, T, J)


def f05_taper_5_slices():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_tapered_grating(0.3e-6, eps_ridge=6.0 + 0j, eps_groove=1.0 + 0j,
                           duty_bottom=0.6, duty_top=0.4, n_slices=5)
    o, R, T, J = st.set_source(WL, angle=0.0).solve()
    return _t(o, R, T, J)


def f06_taper_8_slices_oblique():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_tapered_grating(0.3e-6, eps_ridge=6.0 + 0j, eps_groove=1.0 + 0j,
                           duty_bottom=0.62, duty_top=0.38, n_slices=8)
    o, R, T, J = st.set_source(WL, angle=np.radians(12.0)).solve()
    return _t(o, R, T, J)


def f07_taper_snap_active():
    """The SAME taper with ``min_feature`` raised until the snap fires -- the
    audit-2026-07-28 accuracy lever, and the one path that already warns."""
    import warnings
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10,
                  min_feature=2.0e-8)
    st.add_tapered_grating(0.3e-6, eps_ridge=6.0 + 0j, eps_groove=1.0 + 0j,
                           duty_bottom=0.62, duty_top=0.38, n_slices=8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.set_source(WL, angle=np.radians(12.0)).solve()
    return _t(o, R, T, J)


def f08_per_layer_halfwidth1():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10,
                  layer_grids="per-layer")
    st.add_tapered_grating(0.3e-6, eps_ridge=6.0 + 0j, eps_groove=1.0 + 0j,
                           duty_bottom=0.62, duty_top=0.38, n_slices=6)
    o, R, T, J = st.set_source(WL, angle=np.radians(12.0)).solve()
    return _t(o, R, T, J)


def f09_per_layer_halfwidth2():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10,
                  layer_grids="per-layer", window_halfwidth=2)
    st.add_tapered_grating(0.3e-6, eps_ridge=6.0 + 0j, eps_groove=1.0 + 0j,
                           duty_bottom=0.62, duty_top=0.38, n_slices=6)
    o, R, T, J = st.set_source(WL, angle=np.radians(12.0)).solve()
    return _t(o, R, T, J)


def f10_conical():
    st = PMMStack(0.6e-6, n_substrate=1.5, n_superstrate=1.0, degree=9)
    st.add_layer(0.30e-6, segments=[(0.5, 4.0), (0.5, 1.0)])
    o, R, T, J = st.set_source(WL, theta=np.deg2rad(30.0),
                               phi=np.deg2rad(40.0)).solve()
    return _t(o, R, T, J)


def f11_slanted():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.25e-6, segments=[(0.5, 6.0 * GR), (0.5, GR)],
                 slant_angle=np.deg2rad(8.0))
    o, R, T, J = st.set_source(WL, angle=np.radians(8.0)).solve()
    return _t(o, R, T, J)


def f12_out_of_plane_tensor():
    M = np.array([[4.0, 0.0, 0.6], [0.0, 3.6, 0.0], [0.6, 0.0, 3.2]],
                 dtype=_C)
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.2e-6, segments=[(0.5, M), (0.5, GR)])
    st.add_layer(0.1e-6, segments=[(0.3, 2.1 * GR), (0.7, GR)])
    o, R, T, J = st.set_source(WL, angle=np.radians(9.0)).solve()
    return _t(o, R, T, J)


def f13_lossy_layer():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_layer(0.15e-6, segments=[(0.4, (2.0 + 0.35j) ** 2), (0.6, 1.0)])
    st.add_layer(0.20e-6, segments=[(0.55, 6.0), (0.45, 1.0)])
    o, R, T, J = st.set_source(WL, angle=np.radians(14.0)).solve()
    return _t(o, R, T, J)


def f14_solve_vs_wavelength():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.2e-6, segments=[(0.5, 6.0), (0.5, 1.0)])
    st.add_layer(0.1e-6, segments=[(0.35, 2.1), (0.65, 1.0)])
    o, R, T, J = st.solve_vs_wavelength(
        [0.50e-6, 0.55e-6, 0.60e-6], angle=np.radians(7.0), jones=True,
        max_workers=1)
    return _t(o, R, T, J)


def f15_prepare():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=10)
    st.add_layer(0.2e-6, segments=[(0.5, 6.0), (0.5, 1.0)])
    st.add_layer(0.1e-6, segments=[(0.35, 2.1), (0.65, 1.0)])
    pr = st.prepare()
    o, R, T, J = pr.solve(wavelength=WL, angle=np.radians(7.0))
    return _t(o, R, T, J)


def f16_stabilize_slices():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_tapered_grating(0.3e-6, eps_ridge=6.0 + 0j, eps_groove=1.0 + 0j,
                           duty_bottom=0.6, duty_top=0.4, n_slices=4)
    o, R, T, J = st.set_source(WL, angle=0.0).solve(stabilize="slices")
    return _t(o, R, T, J)


def f17_internal_field():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_layer(0.2e-6, segments=[(0.5, 6.0), (0.5, 1.0)])
    st.add_layer(0.1e-6, segments=[(0.35, 2.1), (0.65, 1.0)])
    st.set_source(WL, angle=np.radians(11.0)).solve(retain_internal=True)
    fld = st.internal_field(0.15e-6, component="all", pol=0)
    return _t(*[np.asarray(v) for _k, v in sorted(fld.items())]
              if isinstance(fld, dict) else [np.asarray(fld)])


def f18_layer_absorption():
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=12)
    st.add_layer(0.15e-6, segments=[(0.4, (2.0 + 0.35j) ** 2), (0.6, 1.0)])
    st.add_layer(0.20e-6, segments=[(0.55, 6.0), (0.45, 1.0)])
    st.set_source(WL, angle=np.radians(14.0)).solve(retain_internal=True)
    return _t(np.asarray(st.layer_absorption()))


FIXTURES = {k: v for k, v in sorted(globals().items())
            if k.startswith("f") and k[1:3].isdigit() and callable(v)}
