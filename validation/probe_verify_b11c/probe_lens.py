"""VERIFY-B11c bit identity, arm 2: the lens family across every entry point
the ``lenses`` -> ``_lens_kernels`` move can reach -- the sag builders and their
optional-backend plumbing, the facade's live forward (read, WRITE, delete and
``dir``), the two fitting helpers, and the analytic / real / traced / Maslov /
GBD / FGA / thin / DOE consumers.

Fixtures are this verification's own: a different grid, a different wavelength,
different glasses, a different singlet and a THREE-surface cemented doublet
that the package probe never builds, plus the arms it recorded as unreachable
(a REAL CuPy device, which this box has).

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

from lumenairy.elements import lenses as LE
from lumenairy.elements import lenses_maslov as LM

P = vlib.Probe()


def _quiet(fn, *a, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **kw)


# ---------------------------------------------------------------------------
# A. surface_sag_general -- every branch of the conic + aspheric kernel
# ---------------------------------------------------------------------------
_h = np.linspace(0.0, 1.4e-2, 257) ** 2            # h_sq, includes 0
_H2 = (np.linspace(-9e-3, 9e-3, 96)[:, None] ** 2
       + np.linspace(-9e-3, 9e-3, 96)[None, :] ** 2)
_ASPH = {4: 3.1e2, 6: -8.4e6, 8: 1.7e11, 10: -4.2e15}

SG = LE.surface_sag_general
P.call("A01_sag_sphere", SG, _h, 3.7e-2)
P.call("A02_sag_conic_prolate", SG, _h, 3.7e-2, -0.62)
P.call("A03_sag_parabola", SG, _h, 3.7e-2, -1.0)
P.call("A04_sag_hyperbolic", SG, _h, 3.7e-2, -2.35)
P.call("A05_sag_oblate", SG, _h, 3.7e-2, 0.9)
P.call("A06_sag_negative_R", SG, _h, -2.15e-2, -0.4)
P.call("A07_sag_flat_inf", SG, _h, np.inf)
P.call("A08_sag_flat_none", SG, _h, None)
P.call("A09_sag_R_zero", SG, _h, 0.0)
P.call("A10_sag_R_nan", SG, _h, float("nan"))
P.call("A11_sag_beyond_domain", SG, np.linspace(0.0, 9e-2, 64) ** 2,
       3.7e-2, -2.35)
P.call("A12_sag_aspheric_1d", SG, _h, 3.7e-2, -0.62, _ASPH)
P.call("A13_sag_aspheric_2d", SG, _H2, 3.7e-2, -0.62, _ASPH)
P.call("A14_sag_aspheric_flat", SG, _H2, np.inf, 0.0, {4: 5.0e2})
P.call("A15_sag_f32", SG, _h.astype(np.float32), 3.7e-2, -0.62, _ASPH)
P.call("A16_sag_fortran_order", SG, np.asfortranarray(_H2), 3.7e-2, -0.62,
       _ASPH)
P.call("A17_sag_odd_power", SG, _h, 3.7e-2, 0.0, {5: 1.0e6})
P.call("A18_sag_private_alias", LE._surface_sag_general, _H2, 3.7e-2, -0.62,
       _ASPH)

SB = LE.surface_sag_biconic
_X = np.linspace(-8e-3, 8e-3, 81)[:, None] * np.ones((1, 81))
_Y = np.ones((81, 1)) * np.linspace(-8e-3, 8e-3, 81)[None, :]
P.call("A19_biconic_symmetric", SB, _X, _Y, 4.1e-2)
P.call("A20_biconic_asymmetric", SB, _X, _Y, 4.1e-2, -2.9e-2)
P.call("A21_biconic_conics", SB, _X, _Y, 4.1e-2, -2.9e-2, -0.5, 0.35)
P.call("A22_biconic_aspheric", SB, _X, _Y, 4.1e-2, -2.9e-2, -0.5, 0.35,
       {4: 2.2e2, 6: -5.1e6})
P.call("A23_biconic_per_axis", SB, _X, _Y, 4.1e-2, -2.9e-2, -0.5, 0.35,
       {4: 2.2e2}, {4: -1.4e2, 6: 3.3e6})
P.call("A24_biconic_flat_y", SB, _X, _Y, 4.1e-2, np.inf)
P.call("A25_biconic_f32", SB, _X.astype(np.float32), _Y.astype(np.float32),
       4.1e-2, -2.9e-2)
P.call("A26_xy_polynomial", la.surface_sag_xy_polynomial, _X, _Y,
       4.1e-2, -0.5, {(2, 0): 1.1e2, (0, 2): -0.7e2, (2, 2): 4.0e5},
       8e-3, 8e-3)


# ---------------------------------------------------------------------------
# B. The gates as LIVE state, exercised THROUGH the facade (read + write +
#    delete + dir), which is exactly the monkeypatch shape.
# ---------------------------------------------------------------------------
P.add("B01_gate_values", (bool(LE._NUMBA_AVAILABLE),
                          bool(LE.NUMEXPR_AVAILABLE),
                          bool(LE.CUPY_AVAILABLE)))
_fast = SG(_H2, 3.7e-2, -0.62, _ASPH)
P.add("B02_numba_fastpath", _fast)
_saved = LE._NUMBA_AVAILABLE
LE._NUMBA_AVAILABLE = False                      # the write, through the facade
P.add("B03_gate_after_write", bool(LE._NUMBA_AVAILABLE))
P.call("B04_load_numba_false", LE._load_numba)
_slow = SG(_H2, 3.7e-2, -0.62, _ASPH)
P.add("B05_numpy_arm", _slow)
P.add("B06_arms_bit_equal", vlib.digest(_fast) == vlib.digest(_slow))
LE._NUMBA_AVAILABLE = _saved
P.add("B07_gate_restored", bool(LE._NUMBA_AVAILABLE))
P.call("B08_load_numba_true", LE._load_numba)
P.add("B09_numba_kernels_keys", sorted(LE._NUMBA_KERNELS))

P.call("B10_ensure_numexpr", LE._ensure_numexpr_loaded)
_nesave = LE.NUMEXPR_AVAILABLE
LE.NUMEXPR_AVAILABLE = False
P.add("B11_numexpr_gate_off", bool(LE.NUMEXPR_AVAILABLE))
P.call("B12_ensure_numexpr_off", LE._ensure_numexpr_loaded)
LE.NUMEXPR_AVAILABLE = _nesave
P.add("B13_numexpr_restored", bool(LE.NUMEXPR_AVAILABLE))


def _missing():
    return LE.no_such_name_anywhere


P.call("B14_attribute_error", _missing)
_FORWARD = ("cp", "_ne", "NUMEXPR_AVAILABLE", "_NUMBA_AVAILABLE", "_numba",
            "_njit", "_prange", "_NUMBA_KERNELS")
_MOVED = ("surface_sag_general", "surface_sag_biconic", "_surface_sag_general",
          "_fit_normaliser", "_multi_indices_total_degree",
          "_ensure_numexpr_loaded", "_ensure_cupy_loaded", "_is_cupy_array",
          "_load_numba", "_get_aspheric_sag_accum_numba", "CUPY_AVAILABLE",
          "check_grid_vs_apertures", "recommend_grid_for_prescription",
          "_collect_semi_diameters", "_warn_if_aperture_exceeds_grid")
P.add("B15_dir_lists_forwarded", sorted(n for n in dir(LE) if n in _FORWARD))
P.add("B16_dir_lists_moved", sorted(n for n in dir(LE) if n in _MOVED))
P.add("B17_hasattr_matrix", [(n, hasattr(LE, n))
                             for n in _FORWARD + _MOVED])
P.add("B18_toplevel_resolve", sorted(
    n for n in ("surface_sag_general", "surface_sag_biconic",
                "surface_sag_xy_polynomial", "apply_real_lens")
    if hasattr(la, n)))
# delete + restore, through the facade
_cp_before = LE.cp
del LE.cp
P.add("B19_after_delete", (hasattr(LE, "cp"),))
LE.cp = _cp_before
P.add("B20_after_restore", LE.cp is _cp_before)


# ---------------------------------------------------------------------------
# C. The two fitting helpers the item-3 back-edge carried
# ---------------------------------------------------------------------------
MI = LE._multi_indices_total_degree
P.add("C01_multi_indices", [(nv, od, [tuple(t) for t in MI(nv, od)])
                            for nv in (1, 2, 3, 4) for od in (0, 1, 2, 3, 5)])
FN = LE._fit_normaliser
_rng = np.random.default_rng(90211)
_cases = [np.zeros(9), np.full(7, 2.5), np.array([-1e-9, 1e-9]),
          _rng.standard_normal(40) * 3.3e-3,
          np.linspace(-4.0, 9.0, 33), np.array([0.0]),
          np.array([np.nan, 1.0, 2.0])]
for _i, _c in enumerate(_cases):
    for _pad in (0.0, 0.05, 1.0):
        P.call(f"C02_fit_norm_{_i}_{_pad}", FN, _c, _pad)
P.add("C03_maslov_spelling_identity", [
    (n, getattr(LM, n, None) is getattr(LE, n, None))
    for n in ("_ensure_numexpr_loaded", "_fit_normaliser",
              "_multi_indices_total_degree")])
P.add("C04_numexpr_value_agrees",
      bool(LM.NUMEXPR_AVAILABLE) == bool(LE.NUMEXPR_AVAILABLE))


# ---------------------------------------------------------------------------
# D. The CuPy arm, ON A REAL DEVICE where the box has one (the WP-B11c report
#    records this as unmeasurable; it is measurable here).
# ---------------------------------------------------------------------------
P.add("D01_cupy_available", bool(LE.CUPY_AVAILABLE))
if LE.CUPY_AVAILABLE:
    P.call("D02_ensure_cupy", LE._ensure_cupy_loaded)
    _cp = LE.cp
    P.add("D03_cp_name", None if _cp is None else _cp.__name__)
    P.add("D04_is_cupy_np", LE._is_cupy_array(_H2))
    if _cp is not None:
        try:
            _g = _cp.asarray(_H2)
            P.add("D05_is_cupy_dev", bool(LE._is_cupy_array(_g)))
            P.add("D06_sag_on_device",
                  _cp.asnumpy(SG(_g, 3.7e-2, -0.62, _ASPH)))
            P.add("D07_sag_biconic_device",
                  _cp.asnumpy(SB(_cp.asarray(_X), _cp.asarray(_Y),
                                 4.1e-2, -2.9e-2, -0.5, 0.35, {4: 2.2e2})))
            _dev_err = None
        except Exception as exc:                 # noqa: BLE001 -- recorded
            _dev_err = type(exc).__name__
        P.add("D08_device_error", _dev_err)
else:
    P.add("D02_ensure_cupy", "cupy-absent")


# ---------------------------------------------------------------------------
# E. The consumers -- analytic, real, traced, Maslov, GBD, FGA, thin, DOE
# ---------------------------------------------------------------------------
N, DX, WL = 160, 6.5e-6, 1.064e-6
_ax = (np.arange(N) - N / 2 + 0.5) * DX
_XX, _YY = np.meshgrid(_ax, _ax, indexing="xy")
E0 = np.exp(-(_XX ** 2 + _YY ** 2) / (2 * (1.05e-3) ** 2)).astype(complex)
E0 = E0 * np.exp(1j * (417.0 * _XX - 233.0 * _YY))

P.call("E01_thin", la.apply_thin_lens, E0, f=37e-3, wavelength=WL, dx=DX)
P.call("E02_thin_decentred", la.apply_thin_lens, E0, f=37e-3, wavelength=WL,
       dx=DX, xc=1.7e-4, yc=-2.9e-4)
P.call("E03_spherical", la.apply_spherical_lens, E0, R1=41e-3, R2=-27e-3,
       d=3.2e-3, n_lens=1.5062, wavelength=WL, dx=DX)
P.call("E04_aspheric", la.apply_aspheric_lens, E0, R1=41e-3, R2=-27e-3,
       d=3.2e-3, n_lens=1.5062, k1=-0.83, k2=0.21,
       A1={4: 1.9e2, 6: -4.4e6}, A2={4: -6.1e1}, wavelength=WL, dx=DX)
P.call("E05_cylindrical_y", la.apply_cylindrical_lens, E0, f=37e-3,
       wavelength=WL, dx=DX, axis="y")

_singlet = la.make_singlet(R1=41e-3, R2=-27e-3, d=3.2e-3, glass="N-SF11",
                           aperture=9e-3)
_planoc = la.make_singlet(R1=52e-3, R2=np.inf, d=2.6e-3, glass="N-BK7",
                          aperture=9e-3)
_doublet = {
    "surfaces": [
        {"radius": 33e-3, "conic": -0.21, "glass_before": "air",
         "glass_after": "N-BK7"},
        {"radius": -21e-3, "conic": 0.0, "glass_before": "N-BK7",
         "glass_after": "N-SF11"},
        {"radius": -74e-3, "conic": 0.0, "glass_before": "N-SF11",
         "glass_after": "air"},
    ],
    "thicknesses": [3.1e-3, 1.9e-3],
    "aperture_diameter": 8.4e-3,
}
_asph_presc = {
    "surfaces": [
        {"radius": 29e-3, "conic": -0.77,
         "aspheric_coeffs": {4: 1.4e2, 6: -3.9e6},
         "glass_before": "air", "glass_after": "N-BK7"},
        {"radius": np.inf, "conic": 0.0, "glass_before": "N-BK7",
         "glass_after": "air"},
    ],
    "thicknesses": [2.8e-3],
    "aperture_diameter": 8.0e-3,
}

P.call("E06_real_singlet", _quiet, la.apply_real_lens, E0,
       prescription=_singlet, wavelength=WL, dx=DX)
P.call("E07_real_doublet", _quiet, la.apply_real_lens, E0,
       prescription=_doublet, wavelength=WL, dx=DX)
P.call("E08_real_slant_fresnel", _quiet, la.apply_real_lens, E0,
       prescription=_doublet, wavelength=WL, dx=DX, slant_correction=True,
       fresnel=True)
P.call("E09_real_aspheric", _quiet, la.apply_real_lens, E0,
       prescription=_asph_presc, wavelength=WL, dx=DX)
P.call("E10_real_c64", _quiet, la.apply_real_lens, E0.astype(np.complex64),
       prescription=_singlet, wavelength=WL, dx=DX)
P.call("E11_real_absorption", _quiet, la.apply_real_lens, E0,
       prescription=_singlet, wavelength=WL, dx=DX, absorption=True)
P.call("E12_real_seidel", _quiet, la.apply_real_lens, E0,
       prescription=_planoc, wavelength=WL, dx=DX, seidel_correction=True)
P.call("E13_traced", _quiet, la.apply_real_lens_traced, E0,
       prescription=_doublet, wavelength=WL, dx=DX, n_workers=1)
P.call("E14_traced_multibranch", _quiet, la.apply_real_lens_traced_multibranch,
       E0, prescription=_doublet, wavelength=WL, dx=DX)
P.call("E15_maslov_singlet", _quiet, la.apply_real_lens_maslov, E0,
       prescription=_singlet, wavelength=WL, dx=DX)
P.call("E16_maslov_doublet", _quiet, la.apply_real_lens_maslov, E0,
       prescription=_doublet, wavelength=WL, dx=DX, poly_order=5)
P.call("E17_maslov_asph", _quiet, la.apply_real_lens_maslov, E0,
       prescription=_asph_presc, wavelength=WL, dx=DX)
P.call("E18_gbd", _quiet, la.apply_real_lens_gbd, E0,
       prescription=_singlet, wavelength=WL, dx=DX)
P.call("E19_fga", _quiet, la.apply_real_lens_fga, E0,
       prescription=_singlet, wavelength=WL, dx=DX)

_Evec = np.stack([(0.73 * E0), (0.41 * E0 * np.exp(0.7j))], axis=0)
P.call("E20_maslov_vector", _quiet, la.apply_real_lens_maslov_vector, _Evec,
       prescription=_doublet, wavelength=WL, dx=DX,
       integration_method="quadrature", n_v2=40, poly_order=4)
for _mode in ("none", "power", "peak"):
    P.call(f"E21_maslov_vector_{_mode}", _quiet,
           la.apply_real_lens_maslov_vector, _Evec, prescription=_doublet,
           wavelength=WL, dx=DX, integration_method="quadrature", n_v2=40,
           poly_order=4, normalize_output=_mode)


# ---------------------------------------------------------------------------
# F. Grid-versus-aperture bookkeeping, Surface sag, and the DOE leg
# ---------------------------------------------------------------------------
P.call("F01_check_grid_small", LE.check_grid_vs_apertures, _doublet, DX, N)
P.call("F02_check_grid_big", LE.check_grid_vs_apertures, _doublet, DX, 4096)
P.call("F03_recommend", LE.recommend_grid_for_prescription, _doublet, DX)
P.call("F04_semi_diameters", LE._collect_semi_diameters, _doublet)
P.call("F05_warn_aperture", LE._warn_if_aperture_exceeds_grid, _doublet, N,
       DX, source="verify-b11c")
P.call("F05b_warn_aperture_small", LE._warn_if_aperture_exceeds_grid,
       _doublet, 48, DX, source="verify-b11c", safety_factor=1.4)

_surfs = la.surfaces_from_prescription(_doublet)
P.add("F06_surface_sag_rot", [
    _quiet(s.sag, _X, _Y) if hasattr(s, "sag") else "no-sag" for s in _surfs])

_bic = la.Surface(radius=33e-3, radius_y=-21e-3, conic=-0.2, thickness=2e-3,
                  n_before=1.0, n_after=1.5168) \
    if "radius_y" in str(la.Surface.__init__.__doc__ or "") else None
P.add("F07_biconic_surface_built", _bic is not None)

P.call("F08_diffractive_lens", la.create_diffractive_lens, N, DX, 37e-3, WL)
_rb = la.RayBundle(x=np.linspace(-2e-3, 2e-3, 9),
                   y=np.linspace(1e-3, -1e-3, 9),
                   z=np.zeros(9), L=np.full(9, 0.03), M=np.full(9, -0.02),
                   N=np.sqrt(1 - 0.03 ** 2 - 0.02 ** 2) * np.ones(9),
                   wavelength=WL, opd=np.zeros(9),
                   alive=np.ones(9, dtype=bool))
P.call("F09_doe_phase_traced", _quiet, la.apply_doe_phase_traced, _rb, 1, 0,
       period_x=11e-6, period_y=13e-6, wavelength=WL)
P.call("F10_doe_order2", _quiet, la.apply_doe_phase_traced, _rb, 2, -1,
       period_x=11e-6, period_y=13e-6, wavelength=WL, n_medium=1.5168)

P.write(OUT)
