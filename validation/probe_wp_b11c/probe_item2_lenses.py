"""WP-B11c items 2 + 3 gate: the lens / Maslov answers the leaf move must not move.

Three surfaces:

* the moved definitions themselves -- both sag builders over a conic +
  aspheric + biconic ladder including the numba fast path and the pure-NumPy
  arm reached by flipping the gate, ``_multi_indices_total_degree`` over a
  (n_vars, order) grid, ``_fit_normaliser`` on degenerate and ordinary data,
  and every refusal message they raise;
* the LIVE forward -- the numba gate written through ``lenses``, read back,
  used by the kernel (the pure-NumPy sag arm) and restored, plus the two
  spellings ``lenses_maslov`` uses;
* the consumers -- ``apply_real_lens`` (plain, slant + fresnel, aspheric,
  ``apply_thin_lens`` / ``apply_spherical_lens`` / ``apply_aspheric_lens``,
  ``apply_real_lens_maslov``, the asymptotic propagator, the traced lens, the
  freeform sag entry points and ``Surface`` sag, each on a fixture whose field
  is deterministic.

Run through ``bi.py``; never under pytest.
"""
# ruff: noqa: E402, I001 -- a probe BINDS its tree before it imports the
# library, so the imports cannot be hoisted above ``bind()``.
from __future__ import annotations

import sys
import warnings

sys.path.insert(0, ".")
from probelib import bind, caught, emit, h

bind(sys.argv[1])

import numpy as np

import lumenairy as L
from lumenairy.elements import lenses as LE
from lumenairy.elements import lenses_maslov as LM

# Everything is addressed through the STABLE paths -- ``lumenairy``,
# ``lenses`` and ``lenses_maslov`` -- because the probe has to run unchanged on
# BOTH trees, and on the parent tree the leaf does not carry these names at
# all.  That is also the honest question to ask of a re-export: a caller's
# spelling did not change, so its answer must not either.  The structural
# claims (which module a name is DEFINED in, and that the facade hands out the
# same object) are DECISIONS and live in
# ``tests/unit/test_audit2609_b11c_structure.py``, not in a hash.

R = {}

# --------------------------------------------------------------------------
# A. The moved definitions
# --------------------------------------------------------------------------
h_sq = np.linspace(0.0, (6e-3) ** 2, 257)
ASPH = {4: 1.0e6, 6: -2.0e9, 8: 3.5e12}

R["A1_sag_conic"] = h(LE.surface_sag_general(h_sq, 25e-3, -0.7, None))
R["A2_sag_aspheric"] = h(LE.surface_sag_general(h_sq, 25e-3, -0.7, ASPH))
R["A3_sag_flat"] = h(LE.surface_sag_general(h_sq, np.inf, 0.0, ASPH))
R["A4_sag_negative_R"] = h(LE.surface_sag_general(h_sq, -18e-3, 0.0, ASPH))
R["A5_sag_hyperbolic"] = h(LE.surface_sag_general(h_sq, 12e-3, -2.5, None))
R["A6_sag_beyond_aperture"] = h(caught(
    LE.surface_sag_general, np.linspace(0.0, (40e-3) ** 2, 65), 12e-3, 0.0,
    None))
R["A7_sag_float32"] = h(LE.surface_sag_general(
    h_sq.astype(np.float32), 25e-3, -0.7, ASPH))

# The numba fast path and the pure-NumPy arm must agree bit for bit; BOTH are
# hashed, so a move that silently stopped reaching one of them is visible.
R["A8_sag_numba_arm"] = h(LE.surface_sag_general(h_sq, 25e-3, -0.7, ASPH))
_saved_gate, _saved_numba = LE._NUMBA_AVAILABLE, LE._numba
LE._NUMBA_AVAILABLE = False
LE._numba = None
R["A9_sag_numpy_arm"] = h(LE.surface_sag_general(h_sq, 25e-3, -0.7, ASPH))
R["A10_load_numba_gate_off"] = h(LE._load_numba())
R["A11_gate_read_back"] = h(bool(LE._NUMBA_AVAILABLE))
LE._NUMBA_AVAILABLE, LE._numba = _saved_gate, _saved_numba
R["A12_gate_restored"] = h(bool(LE._NUMBA_AVAILABLE))

x = np.linspace(-6e-3, 6e-3, 65)
y = np.linspace(-4e-3, 4e-3, 65)
X, Y = np.meshgrid(x, y, indexing="xy")
R["B1_biconic_sym"] = h(LE.surface_sag_biconic(X, Y, 25e-3))
R["B2_biconic_asym"] = h(LE.surface_sag_biconic(X, Y, 25e-3, -40e-3, -0.7, 0.3))
R["B3_biconic_aspheric"] = h(LE.surface_sag_biconic(
    X, Y, 25e-3, -40e-3, -0.7, 0.3, aspheric_coeffs=ASPH))
R["B4_biconic_per_axis"] = h(LE.surface_sag_biconic(
    X, Y, 25e-3, -40e-3, -0.7, 0.3, aspheric_coeffs=ASPH,
    aspheric_coeffs_y={4: -2.0e6}))
R["B5_biconic_flat_y"] = h(LE.surface_sag_biconic(X, Y, 25e-3, np.inf))

# --------------------------------------------------------------------------
# D. The forward itself
# --------------------------------------------------------------------------
R["D2_gate_values"] = h([bool(LE.NUMEXPR_AVAILABLE),
                         bool(LE._NUMBA_AVAILABLE),
                         bool(LE.CUPY_AVAILABLE)])
R["D3_numexpr_loader"] = h(caught(LE._ensure_numexpr_loaded))
R["D4_missing_attr"] = h(caught(getattr, LE, "_no_such_name_at_all"))
R["D5_public_names_present"] = h(sorted(
    n for n in ("surface_sag_general", "surface_sag_biconic",
                "check_grid_vs_apertures", "recommend_grid_for_prescription",
                "NUMEXPR_AVAILABLE", "CUPY_AVAILABLE", "apply_real_lens",
                "apply_real_lens_maslov", "apply_thin_lens")
    if hasattr(LE, n)))
R["D6_toplevel"] = h(sorted(
    n for n in ("NUMEXPR_AVAILABLE", "surface_sag_general", "apply_real_lens")
    if hasattr(L, n)))

# --------------------------------------------------------------------------
# E. The consumers
# --------------------------------------------------------------------------
N, DX, WL = 128, 8e-6, 633e-9
xx = (np.arange(N) - N / 2 + 0.5) * DX
XX, YY = np.meshgrid(xx, xx, indexing="xy")
E0 = np.exp(-(XX ** 2 + YY ** 2) / (2 * (1.2e-3) ** 2)).astype(np.complex128)
E0 *= np.exp(1j * 300.0 * XX)

R["E1_thin"] = h(L.apply_thin_lens(E0, f=50e-3, wavelength=WL, dx=DX))
R["E2_spherical"] = h(L.apply_spherical_lens(E0, R1=50e-3, R2=-50e-3, d=4e-3,
                                             n_lens=1.5168, wavelength=WL,
                                             dx=DX))
R["E3_aspheric"] = h(L.apply_aspheric_lens(E0, R1=25e-3, R2=-40e-3, d=4e-3,
                                           n_lens=1.5168, k1=-0.7, k2=0.3,
                                           A1=ASPH, A2={4: -2.0e6},
                                           wavelength=WL, dx=DX))
R["E4_cylindrical"] = h(L.apply_cylindrical_lens(E0, f=50e-3, wavelength=WL,
                                                 dx=DX))

presc = L.make_singlet(R1=50e-3, R2=-50e-3, d=4e-3, glass="N-BK7",
                       aperture=10e-3)
presc_a = L.make_singlet(R1=50e-3, R2=np.inf, d=4e-3, glass="N-BK7",
                         aperture=10e-3)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    R["E5_real_lens"] = h(L.apply_real_lens(E0, prescription=presc,
                                            wavelength=WL, dx=DX))
    R["E6_real_lens_slant"] = h(L.apply_real_lens(
        E0, prescription=presc, wavelength=WL, dx=DX,
        slant_correction=True, fresnel=True))
    R["E7_real_lens_asph"] = h(L.apply_real_lens(
        E0, prescription=presc_a, wavelength=WL, dx=DX))
    R["E8_real_lens_f32"] = h(L.apply_real_lens(
        E0.astype(np.complex64), prescription=presc, wavelength=WL, dx=DX))
    R["E9_maslov"] = h(L.apply_real_lens_maslov(E0, prescription=presc,
                                                wavelength=WL, dx=DX))
    R["E10_maslov_asph"] = h(L.apply_real_lens_maslov(
        E0, prescription=presc_a, wavelength=WL, dx=DX))
    R["E11_traced"] = h(L.apply_real_lens_traced(E0, prescription=presc,
                                                 wavelength=WL, dx=DX,
                                                 n_workers=1))
    R["E12_gbd"] = h(caught(L.apply_real_lens_gbd, E0, prescription=presc,
                            wavelength=WL, dx=DX))

# Freeform + Surface sag read the same two builders through other doors.
from lumenairy.elements.freeform import (                        # noqa: E402
    surface_sag_xy_polynomial,
)
from lumenairy.raytrace.surface import Surface, _surface_sag_xy  # noqa: E402

R["F1_freeform_xy"] = h(surface_sag_xy_polynomial(
    X, Y, R=25e-3, conic=-0.7,
    xy_coeffs={(2, 0): 1.0e-3, (0, 2): -0.5e-3, (4, 0): 2.0e3},
    norm_x=6e-3, norm_y=4e-3))
srf = Surface(radius=25e-3, conic=-0.7, aspheric_coeffs=ASPH)
R["F2_surface_sag"] = h(_surface_sag_xy(X, Y, srf))
srf_b = Surface(radius=25e-3, radius_y=-40e-3, conic=-0.7, conic_y=0.3)
R["F3_surface_sag_biconic"] = h(_surface_sag_xy(X, Y, srf_b))

# The grid-vs-aperture bookkeeping, warnings and all.
R["G1_check_grid"] = h(caught(LE.check_grid_vs_apertures, presc, 64, DX))
R["G2_recommend"] = h(caught(LE.recommend_grid_for_prescription, presc, DX))
R["G3_semi_diameters"] = h(LE._collect_semi_diameters(presc))

emit(R)
