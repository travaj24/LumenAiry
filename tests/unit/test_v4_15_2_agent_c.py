"""v4.15.2 Agent-C tests -- P1-NEW-E + P1-NEW-F defensive guards.

The v4.15.1 audit (``docs/audits/AUDIT_V4_15_1_2026_05_18.md``) found
that v4.15.1 ships two new types -- :class:`PartialCoherenceMCF` (from
``sources/core.py``) and 3-D ensemble arrays of shape
``(n_realizations, Ny, Nx)`` from ``create_*_schell_source()`` -- that
the existing propagator chain silently accepts and then crashes with
either a cryptic ``AttributeError`` ("no attribute 'copy'") or
performs the FFT over the wrong axis on the 3-D ensemble.

v4.15.2 (P1-NEW-E + P1-NEW-F) adds defensive guards at the top of every
propagator entry point so the user gets a clear, actionable error
instead of a downstream surprise.  The guards are the FIRST executable
statements in each entry point (before any input validation or backend
dispatch) so the error reaches the user as fast as possible.

Entry points covered:

* :func:`propagate_through_system` (``lumenairy.system``)
* :func:`propagate` (``lumenairy.propagators.dispatch``)
* :func:`angular_spectrum_propagate`,
  :func:`fresnel_propagate`,
  :func:`fraunhofer_propagate`,
  :func:`rayleigh_sommerfeld_propagate`,
  :func:`scalable_angular_spectrum_propagate`
  (``lumenairy.propagators.propagation``)
* :func:`apply_thin_lens`, :func:`apply_cylindrical_lens`
  (``lumenairy.elements._lens_thin``, re-exported via
  ``lumenairy.elements.lenses``)
* :func:`apply_real_lens`
  (``lumenairy.elements._lens_real``, re-exported via
  ``lumenairy.elements.lenses``)

MCF-aware downstream propagators are scheduled for v4.16+.  Until then,
``PartialCoherenceMCF`` and 3-D ensemble inputs raise ``TypeError`` /
``ValueError`` respectively with the canonical iterate-over-ensemble
workaround in the message body.

Author: Andrew Traverso -- v4.15.2 / Agent C
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.lenses import (
    apply_cylindrical_lens,
    apply_real_lens,
    apply_thin_lens,
)
from lumenairy.propagators.dispatch import propagate
from lumenairy.propagators.propagation import (
    angular_spectrum_propagate,
    fraunhofer_propagate,
    fresnel_propagate,
    rayleigh_sommerfeld_propagate,
    scalable_angular_spectrum_propagate,
)
from lumenairy.sources.core import (
    PartialCoherenceMCF,
    create_gaussian_schell_source,
)
from lumenairy.system import propagate_through_system

# v4.15.4 (AUDIT_V4_15_3 P3-NEW-F2-4): the helper fixtures here
# legitimately invoke ``create_gaussian_schell_source(...)`` with the
# legacy positional default to obtain a 3-D ensemble for negative-path
# tests.  Pin DeprecationWarning back to ``default`` for this module so
# the suite stays clean under strict ``-W error::DeprecationWarning``.
pytestmark = pytest.mark.filterwarnings(
    'default::DeprecationWarning',
)


# ============================================================================
# Helpers
# ============================================================================

WAVELENGTH = 633e-9
DX = 2e-6
N = 32
W0 = 40e-6
SIGMA_G = 10e-6
N_REALIZATIONS = 4


def _make_mcf() -> PartialCoherenceMCF:
    """Build a small ``PartialCoherenceMCF`` for guard testing."""
    return create_gaussian_schell_source(
        N=N, dx=DX, wavelength=WAVELENGTH,
        w0=W0, sigma_g=SIGMA_G,
        n_realizations=N_REALIZATIONS,
        seed=0, return_kind='mcf',
    )


def _make_ensemble() -> np.ndarray:
    """Build a small 3-D ensemble from ``create_gaussian_schell_source``."""
    ens, _dx, _dy, _wl = create_gaussian_schell_source(
        N=N, dx=DX, wavelength=WAVELENGTH,
        w0=W0, sigma_g=SIGMA_G,
        n_realizations=N_REALIZATIONS,
        seed=0,
    )
    return ens


def _make_2d_field() -> np.ndarray:
    """Plain 2-D complex field for regression / control tests."""
    rng = np.random.default_rng(0)
    return (rng.standard_normal((N, N))
            + 1j * rng.standard_normal((N, N))).astype(np.complex128)


def _minimal_singlet_prescription():
    """Two-surface plano-convex N-BK7 singlet -- the minimal valid
    ``prescription`` for ``apply_real_lens``."""
    return {
        'surfaces': [
            {
                'radius': 50e-3,
                'conic': 0.0,
                'glass_before': 'air',
                'glass_after': 'N-BK7',
            },
            {
                'radius': float('inf'),
                'conic': 0.0,
                'glass_before': 'N-BK7',
                'glass_after': 'air',
            },
        ],
        'thicknesses': [2e-3],
    }


# ============================================================================
# P1-NEW-E -- PartialCoherenceMCF rejection at every propagator entry
# ============================================================================

class TestPartialCoherenceMCFRejection:
    """Every propagator entry point must raise ``TypeError`` -- with a
    clear v4.16+ scope message -- when handed a
    :class:`PartialCoherenceMCF` instance.  Pre-fix the chain would
    crash with ``AttributeError: 'PartialCoherenceMCF' object has no
    attribute 'copy'`` or similar.
    """

    def test_propagate_through_system_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        elements = [{'type': 'propagate', 'z': 1e-3}]
        with pytest.raises(TypeError) as excinfo:
            propagate_through_system(
                mcf, elements, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg
        assert 'propagate_through_system' in msg

    def test_propagate_dispatcher_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            propagate(mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX,
                       method='asm')
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_angular_spectrum_propagate_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            angular_spectrum_propagate(mcf, z=1e-3,
                                        wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_fresnel_propagate_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            fresnel_propagate(mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_fraunhofer_propagate_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            fraunhofer_propagate(mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_rayleigh_sommerfeld_propagate_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            rayleigh_sommerfeld_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_scalable_angular_spectrum_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            scalable_angular_spectrum_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_apply_thin_lens_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_thin_lens(mcf, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_apply_cylindrical_lens_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_cylindrical_lens(
                mcf, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg

    def test_apply_real_lens_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        rx = _minimal_singlet_prescription()
        with pytest.raises(TypeError) as excinfo:
            apply_real_lens(
                mcf, prescription=rx, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        assert 'v4.16' in msg


# ============================================================================
# P1-NEW-F -- 3-D ensemble rejection at every propagator entry
# ============================================================================

class TestThreeDEnsembleRejection:
    """Every propagator entry point must raise ``ValueError`` -- with
    the canonical iterate-over-ensemble suggestion -- when handed a 3-D
    ensemble of shape ``(n_realizations, Ny, Nx)``.  Pre-fix the FFT
    would run over the unintended axis on most propagators silently
    producing wrong-but-plausible output.
    """

    def test_propagate_through_system_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        elements = [{'type': 'propagate', 'z': 1e-3}]
        with pytest.raises(ValueError) as excinfo:
            propagate_through_system(
                ens, elements, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert '3-D' in msg or 'shape' in msg.lower()
        assert 'ensemble.shape[0]' in msg
        assert 'propagate_through_system' in msg

    def test_propagate_dispatcher_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            propagate(ens, z=1e-3, wavelength=WAVELENGTH, dx=DX,
                       method='asm')
        msg = str(excinfo.value)
        assert 'ensemble.shape[0]' in msg

    def test_angular_spectrum_propagate_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            angular_spectrum_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'angular_spectrum_propagate' in msg
        assert 'ensemble.shape[0]' in msg

    def test_fresnel_propagate_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            fresnel_propagate(ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'fresnel_propagate' in msg
        assert 'ensemble.shape[0]' in msg

    def test_fraunhofer_propagate_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            fraunhofer_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'fraunhofer_propagate' in msg
        assert 'ensemble.shape[0]' in msg

    def test_rayleigh_sommerfeld_propagate_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            rayleigh_sommerfeld_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'rayleigh_sommerfeld_propagate' in msg
        assert 'ensemble.shape[0]' in msg

    def test_scalable_angular_spectrum_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            scalable_angular_spectrum_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'scalable_angular_spectrum_propagate' in msg
        assert 'ensemble.shape[0]' in msg

    def test_apply_thin_lens_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_thin_lens(ens, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_thin_lens' in msg
        assert 'ensemble.shape[0]' in msg

    def test_apply_cylindrical_lens_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_cylindrical_lens(
                ens, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_cylindrical_lens' in msg
        assert 'ensemble.shape[0]' in msg

    def test_apply_real_lens_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        rx = _minimal_singlet_prescription()
        with pytest.raises(ValueError) as excinfo:
            apply_real_lens(
                ens, prescription=rx, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_real_lens' in msg
        assert 'ensemble.shape[0]' in msg


# ============================================================================
# Regression: plain 2-D inputs still work end-to-end
# ============================================================================

class TestTwoDPassthrough:
    """The guards must not affect the legitimate 2-D-ndarray path.
    Smoke-test every entry point with a plain ``np.complex128`` 2-D
    array and assert it runs to completion."""

    def test_2d_input_still_works_propagate_through_system(self):
        E = _make_2d_field()
        elements = [{'type': 'propagate', 'z': 1e-3}]
        E_out, _ = propagate_through_system(
            E, elements, wavelength=WAVELENGTH, dx=DX)
        assert E_out.shape == E.shape
        assert E_out.dtype == E.dtype

    def test_2d_input_still_works_dispatch(self):
        E = _make_2d_field()
        E_out = propagate(E, z=1e-3, wavelength=WAVELENGTH, dx=DX,
                           method='asm')
        # ASM returns a bare ndarray.
        assert isinstance(E_out, np.ndarray)
        assert E_out.shape == E.shape

    def test_2d_input_still_works_angular_spectrum(self):
        E = _make_2d_field()
        E_out = angular_spectrum_propagate(
            E, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        assert E_out.shape == E.shape

    def test_2d_input_still_works_fresnel(self):
        E = _make_2d_field()
        out = fresnel_propagate(E, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        E_out, dx_out, dy_out = out
        assert E_out.shape == E.shape
        assert dx_out > 0
        assert dy_out > 0

    def test_2d_input_still_works_fraunhofer(self):
        E = _make_2d_field()
        out = fraunhofer_propagate(E, z=1.0, wavelength=WAVELENGTH, dx=DX)
        E_out, dx_out, dy_out = out
        assert E_out.shape == E.shape

    def test_2d_input_still_works_rayleigh_sommerfeld(self):
        E = _make_2d_field()
        E_out = rayleigh_sommerfeld_propagate(
            E, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        assert E_out.shape == E.shape

    def test_2d_input_still_works_apply_thin_lens(self):
        E = _make_2d_field()
        E_out = apply_thin_lens(E, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        assert E_out.shape == E.shape
        assert E_out.dtype == E.dtype

    def test_2d_input_still_works_apply_cylindrical_lens(self):
        E = _make_2d_field()
        E_out = apply_cylindrical_lens(
            E, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        assert E_out.shape == E.shape
        assert E_out.dtype == E.dtype

    def test_2d_input_still_works_apply_real_lens(self):
        E = _make_2d_field()
        rx = _minimal_singlet_prescription()
        E_out = apply_real_lens(
            E, prescription=rx, wavelength=WAVELENGTH, dx=DX)
        assert E_out.shape == E.shape


# ============================================================================
# Error-message contract
# ============================================================================

class TestErrorMessageContract:
    """The raised error messages must carry the information users need
    to know this is a scheduled-for-future capability AND show the
    canonical iterate-over-ensemble workaround."""

    def test_error_messages_mention_v4_16_for_mcf(self):
        """Every TypeError raised on a PartialCoherenceMCF input must
        mention v4.16 so users know this is a planned future feature
        and not a permanent limitation."""
        mcf = _make_mcf()
        entry_points = [
            lambda: angular_spectrum_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: fresnel_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: fraunhofer_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: rayleigh_sommerfeld_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: scalable_angular_spectrum_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: apply_thin_lens(
                mcf, f=10e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: apply_cylindrical_lens(
                mcf, f=10e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: apply_real_lens(
                mcf, prescription=_minimal_singlet_prescription(),
                wavelength=WAVELENGTH, dx=DX),
            lambda: propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX,
                method='asm'),
            lambda: propagate_through_system(
                mcf, [{'type': 'propagate', 'z': 1e-3}],
                wavelength=WAVELENGTH, dx=DX),
        ]
        for call in entry_points:
            with pytest.raises(TypeError) as excinfo:
                call()
            msg = str(excinfo.value)
            assert 'v4.16' in msg, (
                f"TypeError text is missing the v4.16 scope marker: "
                f"{msg!r}")

    def test_error_messages_show_iteration_pattern_for_ensemble(self):
        """Every ValueError raised on a 3-D ensemble input must show
        the ``for k in range(ensemble.shape[0])`` iteration pattern so
        users can transition to the supported call style without
        consulting the docs."""
        ens = _make_ensemble()
        entry_points = [
            lambda: angular_spectrum_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: fresnel_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: fraunhofer_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: rayleigh_sommerfeld_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: scalable_angular_spectrum_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: apply_thin_lens(
                ens, f=10e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: apply_cylindrical_lens(
                ens, f=10e-3, wavelength=WAVELENGTH, dx=DX),
            lambda: apply_real_lens(
                ens, prescription=_minimal_singlet_prescription(),
                wavelength=WAVELENGTH, dx=DX),
            lambda: propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX,
                method='asm'),
            lambda: propagate_through_system(
                ens, [{'type': 'propagate', 'z': 1e-3}],
                wavelength=WAVELENGTH, dx=DX),
        ]
        for call in entry_points:
            with pytest.raises(ValueError) as excinfo:
                call()
            msg = str(excinfo.value)
            assert 'for k in range(ensemble.shape[0])' in msg, (
                f"ValueError text is missing the canonical iteration "
                f"pattern: {msg!r}")

    def test_error_messages_show_iteration_pattern_works(self):
        """Functional cross-check: the iterate-over-ensemble pattern
        suggested in the error message actually works end-to-end on a
        Schell ensemble.  This pins the documented workaround as a
        first-class supported usage."""
        ens = _make_ensemble()
        out_stack = np.empty_like(ens)
        for k in range(ens.shape[0]):
            out_stack[k] = angular_spectrum_propagate(
                ens[k], z=1e-3, wavelength=WAVELENGTH, dx=DX)
        # Partial intensity is the ensemble-averaged intensity.
        I_partial = np.mean(np.abs(out_stack) ** 2, axis=0)
        assert I_partial.shape == ens.shape[1:]
        assert np.all(I_partial >= 0)
        assert np.isfinite(I_partial).all()
