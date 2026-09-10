"""VERIFY task 1 -- my OWN PMMStack fixture set for the bit-identity arms.

Independent of ``validation/probe_pmmstack_sliver/bitid_fixtures.py``: different
periods, wavelengths, angles, indices, degrees, segment counts and tensors, so an
agreement here is not an agreement about one author's parameter choices.

Every fixture is a callable that returns a dict of named numpy arrays.  The
caller hashes the raw bytes.  Nothing here imports the probe package.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm._core import _pmm_union_grid


def _q(o, R, T, J):
    return dict(orders=np.asarray(o), R=np.asarray(R), T=np.asarray(T),
                jones=np.asarray(J))


# --------------------------------------------------------------------------
# v01 -- one layer, normal incidence, a period and wavelength of my own
# --------------------------------------------------------------------------
def v01_single_layer_normal():
    st = PMMStack(0.93e-6, n_superstrate=1.0, n_substrate=1.46, degree=13)
    st.add_layer(0.21e-6, segments=[(0.41, 4.41), (0.59, 1.0)])
    st.set_source(0.633e-6, theta=0.0)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v02 -- one layer, a big oblique angle
# --------------------------------------------------------------------------
def v02_single_layer_oblique_40deg():
    st = PMMStack(0.93e-6, n_superstrate=1.0, n_substrate=1.46, degree=13)
    st.add_layer(0.21e-6, segments=[(0.41, 4.41), (0.59, 1.0)])
    st.set_source(0.633e-6, theta=40.0 * np.pi / 180.0)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v03 -- five-layer stack, three segments each, shared union grid
# --------------------------------------------------------------------------
def v03_five_layer_shared():
    st = PMMStack(1.55e-6, n_superstrate=1.5, n_substrate=3.48, degree=11)
    for i in range(5):
        a = 0.19 + 0.04 * i
        st.add_layer(0.11e-6, segments=[(a, 2.1), (0.44, 11.9), (1.0 - a - 0.44, 2.1)])
    st.set_source(1.31e-6, theta=0.11)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v04 -- the SAME five-layer stack on per-layer window grids, halfwidth 1
# --------------------------------------------------------------------------
def v04_five_layer_perlayer_hw1():
    st = PMMStack(1.55e-6, n_superstrate=1.5, n_substrate=3.48, degree=11,
                  layer_grids="per-layer", window_halfwidth=1)
    for i in range(5):
        a = 0.19 + 0.04 * i
        st.add_layer(0.11e-6, segments=[(a, 2.1), (0.44, 11.9), (1.0 - a - 0.44, 2.1)])
    st.set_source(1.31e-6, theta=0.11)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v05 -- same, halfwidth 2
# --------------------------------------------------------------------------
def v05_five_layer_perlayer_hw2():
    st = PMMStack(1.55e-6, n_superstrate=1.5, n_substrate=3.48, degree=11,
                  layer_grids="per-layer", window_halfwidth=2)
    for i in range(5):
        a = 0.19 + 0.04 * i
        st.add_layer(0.11e-6, segments=[(a, 2.1), (0.44, 11.9), (1.0 - a - 0.44, 2.1)])
    st.set_source(1.31e-6, theta=0.11)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v06 -- CONICAL (phi != 0)
# --------------------------------------------------------------------------
def v06_conical():
    st = PMMStack(0.80e-6, n_superstrate=1.0, n_substrate=1.52, degree=12)
    st.add_layer(0.18e-6, segments=[(0.37, 5.76), (0.63, 1.0)])
    st.add_layer(0.09e-6, segments=[(0.44, 2.25), (0.56, 1.0)])
    st.set_source(0.55e-6, theta=0.27, phi=0.62)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v07 -- SLANTED side-walls (the covariant generator + general cascade)
# --------------------------------------------------------------------------
def v07_slanted():
    st = PMMStack(1.10e-6, n_superstrate=1.0, n_substrate=1.45, degree=10)
    st.add_layer(0.24e-6, segments=[(0.33, 6.25), (0.67, 1.0)],
                 slant_angle=0.17)
    st.add_layer(0.16e-6, segments=[(0.50, 2.25), (0.50, 1.0)])
    st.set_source(0.94e-6, theta=0.09)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v08 -- OUT-OF-PLANE tensor (eps_xz / eps_zx populated)
# --------------------------------------------------------------------------
def v08_out_of_plane_tensor():
    e = np.array([[2.9, 0.0, 0.45],
                  [0.0, 2.4, 0.0],
                  [0.45, 0.0, 3.3]], dtype=complex)
    st = PMMStack(1.02e-6, n_superstrate=1.0, n_substrate=1.5, degree=10)
    st.add_layer(0.19e-6, segments=[(0.42, e), (0.58, 1.0)])
    st.add_layer(0.13e-6, segments=[(0.30, 4.0), (0.70, 1.0)])
    st.set_source(0.86e-6, theta=0.20)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v09 -- IN-PLANE anisotropic tensor with off-diagonal xy
# --------------------------------------------------------------------------
def v09_in_plane_tensor_xy():
    e = np.array([[2.56, 0.31, 0.0],
                  [0.31, 2.10, 0.0],
                  [0.0, 0.0, 2.34]], dtype=complex)
    st = PMMStack(0.76e-6, n_superstrate=1.0, n_substrate=1.44, degree=12)
    st.add_layer(0.17e-6, segments=[(0.48, e), (0.52, 1.0)])
    st.set_source(0.70e-6, theta=0.35)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v10 -- LOSSY passive layer
# --------------------------------------------------------------------------
def v10_lossy_layer():
    st = PMMStack(0.64e-6, n_superstrate=1.0, n_substrate=1.5, degree=12)
    st.add_layer(0.05e-6, segments=[(0.35, (0.27 + 3.41j) ** 2), (0.65, 1.0)])
    st.add_layer(0.22e-6, segments=[(0.35, 4.0), (0.65, 1.0)])
    st.set_source(0.532e-6, theta=0.13)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v11 -- an ABSORBING SUPERSTRATE (the _lossy_incidence exemption path)
# --------------------------------------------------------------------------
def v11_absorbing_superstrate():
    st = PMMStack(0.64e-6, n_superstrate=1.4 + 0.004j, n_substrate=1.5,
                  degree=10)
    st.add_layer(0.20e-6, segments=[(0.45, 4.0), (0.55, 1.0)])
    st.set_source(0.532e-6, theta=0.10)
    return _q(*st.solve())


# --------------------------------------------------------------------------
# v12 -- WAVELENGTH SWEEP (solve_vs_wavelength, single worker)
# --------------------------------------------------------------------------
def v12_solve_vs_wavelength():
    st = PMMStack(0.93e-6, n_superstrate=1.0, n_substrate=1.46, degree=11)
    st.add_layer(0.21e-6, segments=[(0.41, 4.41), (0.59, 1.0)])
    wl = np.linspace(0.60e-6, 0.68e-6, 5)
    o, R, T, J = st.solve_vs_wavelength(wl, theta=0.15, max_workers=1,
                                        jones=True)
    return _q(o, R, T, J)


# --------------------------------------------------------------------------
# v13 -- prepare() + keyed material sweep (two points)
# --------------------------------------------------------------------------
def v13_prepare_material_sweep():
    st = PMMStack(1.20e-6, n_superstrate=1.0, n_substrate=1.5, degree=11)
    st.add_layer(0.30e-6, segments=[(0.45, "LC"), (0.55, 1.0)])
    st.add_layer(0.12e-6, segments=[(0.45, 4.0), (0.55, 1.0)])
    prep = st.prepare()
    out = {}
    for tag, val in (("a", 2.56), ("b", 2.89)):
        o, R, T, J = prep.solve(wavelength=1.05e-6, materials={"LC": val},
                                theta=0.19)
        out[f"orders_{tag}"] = np.asarray(o)
        out[f"R_{tag}"] = np.asarray(R)
        out[f"T_{tag}"] = np.asarray(T)
        out[f"jones_{tag}"] = np.asarray(J)
    return out


# --------------------------------------------------------------------------
# v14 -- stabilize='slices'
# --------------------------------------------------------------------------
def v14_stabilize_slices():
    st = PMMStack(1.20e-6, n_superstrate=1.0, n_substrate=1.5, degree=12)
    for i in range(4):
        st.add_layer(0.08e-6, segments=[(0.30 + 0.01 * i, 6.25),
                                        (0.70 - 0.01 * i, 1.0)])
    st.set_source(1.00e-6, theta=0.12)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _q(*st.solve(stabilize="slices"))


# --------------------------------------------------------------------------
# v15 -- retain_internal + internal_field on a nodal AND a resampled grid
# --------------------------------------------------------------------------
def v15_internal_field():
    st = PMMStack(0.90e-6, n_superstrate=1.0, n_substrate=1.5, degree=11)
    st.add_layer(0.15e-6, segments=[(0.40, 4.0), (0.60, 1.0)])
    st.add_layer(0.10e-6, segments=[(0.55, 2.25), (0.45, 1.0)])
    st.set_source(0.78e-6, theta=0.22)
    o, R, T, J = st.solve(retain_internal=True)
    out = _q(o, R, T, J)
    f = st.internal_field(np.array([0.03e-6, 0.11e-6, 0.20e-6]))
    for k in sorted(f):
        out[f"if_{k}"] = np.asarray(f[k])
    g = st.internal_field(0.11e-6, nx=17, component="E")
    for k in sorted(g):
        out[f"ifx_{k}"] = np.asarray(g[k])
    return out


# --------------------------------------------------------------------------
# v16 -- layer_absorption (plain and by_material)
# --------------------------------------------------------------------------
def v16_layer_absorption():
    st = PMMStack(0.64e-6, n_superstrate=1.0, n_substrate=1.5, degree=11)
    st.add_layer(0.04e-6, segments=[(0.35, (0.27 + 3.41j) ** 2), (0.65, 1.0)])
    st.add_layer(0.18e-6, segments=[(0.35, 4.0 + 0.05j), (0.65, 1.0)])
    st.set_source(0.532e-6, theta=0.13)
    o, R, T, J = st.solve(retain_internal=True)
    out = _q(o, R, T, J)
    out["absorption"] = np.asarray(st.layer_absorption())
    A, by = st.layer_absorption(by_material=True)
    out["absorption2"] = np.asarray(A)
    for i, k in enumerate(sorted(by, key=repr)):
        out[f"abs_mat_{i}"] = np.asarray(by[k])
    return out


# --------------------------------------------------------------------------
# v17 -- a TAPER whose walls collide, min_feature SNAPPING actively
#        (the snap warns; the numbers must still be identical)
# --------------------------------------------------------------------------
def v17_taper_snap_active():
    P = 1.0e-6
    st = PMMStack(P, n_superstrate=1.0, n_substrate=1.5, degree=10,
                  min_feature=6e-9)
    for i in range(6):
        a = 0.325 - 0.0009 * i
        b = 0.675 + 0.0009 * i
        st.add_layer(0.05e-6, segments=[(a, 1.0), (b - a, 4.0), (1.0 - b, 1.0)])
    st.set_source(0.85e-6, theta=0.08)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _q(*st.solve())


# --------------------------------------------------------------------------
# v18 -- the SAME taper with the snap DORMANT (min_feature far below the
#        collisions): a healthy, non-refused solve on a colliding grid
# --------------------------------------------------------------------------
def v18_taper_snap_dormant():
    P = 1.0e-6
    st = PMMStack(P, n_superstrate=1.0, n_substrate=1.5, degree=10,
                  min_feature=1e-14)
    for i in range(6):
        a = 0.325 - 0.0009 * i
        b = 0.675 + 0.0009 * i
        st.add_layer(0.05e-6, segments=[(a, 1.0), (b - a, 4.0), (1.0 - b, 1.0)])
    st.set_source(0.85e-6, theta=0.08)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _q(*st.solve())


# --------------------------------------------------------------------------
# v19 -- per_order_amplitudes (the retained modal payload)
# --------------------------------------------------------------------------
def v19_per_order_amplitudes():
    st = PMMStack(0.93e-6, n_superstrate=1.0, n_substrate=1.46, degree=12)
    st.add_layer(0.21e-6, segments=[(0.41, 4.41), (0.59, 1.0)])
    st.add_layer(0.07e-6, segments=[(0.60, 2.25), (0.40, 1.0)])
    st.set_source(0.633e-6, theta=0.24)
    out = _q(*st.solve())
    for port in ("reflection", "transmission"):
        a = st.per_order_amplitudes(port=port)
        if isinstance(a, tuple):
            for i, x in enumerate(a):
                out[f"amp_{port}_{i}"] = np.asarray(x)
        elif isinstance(a, dict):
            for k in sorted(a):
                out[f"amp_{port}_{k}"] = np.asarray(a[k])
        else:
            out[f"amp_{port}"] = np.asarray(a)
    return out


# --------------------------------------------------------------------------
# v20 -- a many-order, high-degree Bragg ABAB (the eig / cascade memo path)
# --------------------------------------------------------------------------
def v20_bragg_abab():
    st = PMMStack(0.70e-6, n_superstrate=1.0, n_substrate=1.5, degree=14,
                  far_field_orders=31)
    for i in range(8):
        e = 4.0 if i % 2 == 0 else 2.25
        st.add_layer(0.06e-6, segments=[(0.5, e), (0.5, 1.0)])
    st.set_source(0.60e-6, theta=0.30)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _q(*st.solve())


# --------------------------------------------------------------------------
# v21 -- _pmm_union_grid's 2-TUPLE return, snap dormant and snap active
# --------------------------------------------------------------------------
def _segs(walls, eps):
    out, prev = [], 0.0
    for w, e in zip(list(walls) + [1.0], list(eps)):
        out.append((w - prev, e))
        prev = w
    return [s for s in out if s[0] > 0.0]


def v21_union_grid_two_tuple():
    cases = {
        "dormant": ([_segs([0.30, 0.70], [1.0, 4.0, 1.0]),
                     _segs([0.3001, 0.7001], [1.0, 4.0, 1.0])], 1e-9),
        "snapped": ([_segs([0.30, 0.70], [1.0, 4.0, 1.0]),
                     _segs([0.3001, 0.7001], [1.0, 4.0, 1.0])], 1e-3),
        "nonconf": ([_segs([0.30, 0.50], [1.0, 4.0, 1.0]),
                     _segs([0.35, 0.55], [1.0, 4.0, 1.0])], 1e-5),
        "liner": ([_segs([0.30, 0.3001, 0.70], [1.0, 4.0, 2.0, 1.0]),
                   _segs([0.30, 0.70], [1.0, 4.0, 1.0])], 1e-9),
        "single": ([_segs([0.25, 0.75], [1.0, 4.0, 1.0])], None),
        "taper6": ([_segs([0.325 - 0.0009 * i, 0.675 + 0.0009 * i],
                          [1.0, 4.0, 1.0]) for i in range(6)], 6e-3),
    }
    out = {}
    for tag, (segs, mf) in cases.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ret = _pmm_union_grid(segs, mf)
        assert isinstance(ret, tuple) and len(ret) == 2, (tag, len(ret))
        uw, leps = ret
        out[f"uw_{tag}"] = np.asarray(uw, dtype=float)
        out[f"eps_{tag}"] = np.asarray(leps, dtype=complex)
    return out


FIXTURES = [
    v01_single_layer_normal,
    v02_single_layer_oblique_40deg,
    v03_five_layer_shared,
    v04_five_layer_perlayer_hw1,
    v05_five_layer_perlayer_hw2,
    v06_conical,
    v07_slanted,
    v08_out_of_plane_tensor,
    v09_in_plane_tensor_xy,
    v10_lossy_layer,
    v11_absorbing_superstrate,
    v12_solve_vs_wavelength,
    v13_prepare_material_sweep,
    v14_stabilize_slices,
    v15_internal_field,
    v16_layer_absorption,
    v17_taper_snap_active,
    v18_taper_snap_dormant,
    v19_per_order_amplitudes,
    v20_bragg_abab,
    v21_union_grid_two_tuple,
]
