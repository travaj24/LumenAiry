"""v4.15.3 Agent-A tests -- P0-NEW-F2-1 sibling-guard closure.

The v4.15.2 audit (``docs/audits/AUDIT_V4_15_2_2026_05_18.md``)
flagged 9 additional public entry points in
``lumenairy.propagators.propagation``, ``lumenairy.elements._lens_thin``,
``lumenairy.elements._lens_traced``, and
``lumenairy.elements.lenses_maslov`` that operate on 2-D coherent
scalar fields but were missed by the v4.15.2 P1-NEW-E/F closure --
the textbook "fix N, miss N+1" meta-pattern that has now recurred in
every audit round v4.13.0 -> v4.15.2.

v4.15.3 (Agent A) closes the 9 missed siblings by routing every guard
through the new ``lumenairy._validation._check_2d_scalar_field``
shared helper, AND structurally pins the invariant via the
``test_v4_15_3_dispatcher_pin_2d_scalar_field`` meta-pin so the 20th
entry point can't be added unguarded.

This test file covers:

* **9 sibling rejection tests** -- one per newly-guarded entry point,
  asserting ``TypeError`` on ``PartialCoherenceMCF`` AND
  ``ValueError`` on a 3-D ensemble.
* **1 sanity regression** -- the 10 v4.15.2-guarded sites still
  reject the same way after migrating onto the shared helper.

Author: Andrew Traverso -- v4.15.3 / Agent A
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements._lens_thin import (
    apply_aspheric_lens,
    apply_axicon,
    apply_grin_lens,
    apply_spherical_lens,
)
from lumenairy.elements._lens_traced import apply_real_lens_traced
from lumenairy.elements.lenses import (
    apply_cylindrical_lens,
    apply_real_lens,
    apply_thin_lens,
)
from lumenairy.elements.lenses_maslov import apply_real_lens_maslov
from lumenairy.propagators.dispatch import propagate
from lumenairy.propagators.propagation import (
    angular_spectrum_propagate,
    angular_spectrum_propagate_mft,
    angular_spectrum_propagate_tilted,
    fraunhofer_propagate,
    fraunhofer_propagate_mft,
    fresnel_propagate,
    fresnel_propagate_mft,
    rayleigh_sommerfeld_propagate,
    scalable_angular_spectrum_propagate,
)
from lumenairy.propagators.system import propagate_through_system
from lumenairy.sources.core import (
    PartialCoherenceMCF,
    create_gaussian_schell_source,
)

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


def _minimal_singlet_prescription():
    """Two-surface plano-convex N-BK7 singlet -- the minimal valid
    ``prescription`` for ``apply_real_lens`` / traced / maslov."""
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
# 9 NEW SIBLING GUARDS -- one MCF-rejection + one 3-D-ensemble-rejection
# test per newly-guarded entry point.
# ============================================================================

class TestNewSibling_angular_spectrum_propagate_tilted:
    """``angular_spectrum_propagate_tilted`` is one of the 9 siblings
    the v4.15.2 closure missed.  v4.15.3 routes its guard through
    ``_check_2d_scalar_field``.
    """

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            angular_spectrum_propagate_tilted(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'angular_spectrum_propagate_tilted' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            angular_spectrum_propagate_tilted(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'angular_spectrum_propagate_tilted' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_angular_spectrum_propagate_mft:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            angular_spectrum_propagate_mft(
                mcf, z=1e-3, wavelength=WAVELENGTH,
                dx_in=DX, dx_out=DX, N_out=N)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'angular_spectrum_propagate_mft' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            angular_spectrum_propagate_mft(
                ens, z=1e-3, wavelength=WAVELENGTH,
                dx_in=DX, dx_out=DX, N_out=N)
        msg = str(excinfo.value)
        assert 'angular_spectrum_propagate_mft' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_fresnel_propagate_mft:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            fresnel_propagate_mft(
                mcf, z=1e-3, wavelength=WAVELENGTH,
                dx_in=DX, dx_out=DX, N_out=N)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'fresnel_propagate_mft' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            fresnel_propagate_mft(
                ens, z=1e-3, wavelength=WAVELENGTH,
                dx_in=DX, dx_out=DX, N_out=N)
        msg = str(excinfo.value)
        assert 'fresnel_propagate_mft' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_fraunhofer_propagate_mft:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            fraunhofer_propagate_mft(
                mcf, z=1.0, wavelength=WAVELENGTH,
                dx_in=DX, dx_out=DX, N_out=N)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'fraunhofer_propagate_mft' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            fraunhofer_propagate_mft(
                ens, z=1.0, wavelength=WAVELENGTH,
                dx_in=DX, dx_out=DX, N_out=N)
        msg = str(excinfo.value)
        assert 'fraunhofer_propagate_mft' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_apply_spherical_lens:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_spherical_lens(
                mcf, R1=50e-3, R2=float('inf'),
                d=2e-3, n_lens=1.5,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'apply_spherical_lens' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_spherical_lens(
                ens, R1=50e-3, R2=float('inf'),
                d=2e-3, n_lens=1.5,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_spherical_lens' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_apply_aspheric_lens:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_aspheric_lens(
                mcf, R1=50e-3, R2=float('inf'),
                d=2e-3, n_lens=1.5,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'apply_aspheric_lens' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_aspheric_lens(
                ens, R1=50e-3, R2=float('inf'),
                d=2e-3, n_lens=1.5,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_aspheric_lens' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_apply_grin_lens:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_grin_lens(
                mcf, n0=1.5, g=300.0, d=5e-3,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'apply_grin_lens' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_grin_lens(
                ens, n0=1.5, g=300.0, d=5e-3,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_grin_lens' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_apply_axicon:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_axicon(
                mcf, alpha=0.01, n_axicon=1.5,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'apply_axicon' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_axicon(
                ens, alpha=0.01, n_axicon=1.5,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_axicon' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_apply_real_lens_traced:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        rx = _minimal_singlet_prescription()
        with pytest.raises(TypeError) as excinfo:
            apply_real_lens_traced(
                mcf, prescription=rx,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'apply_real_lens_traced' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        rx = _minimal_singlet_prescription()
        with pytest.raises(ValueError) as excinfo:
            apply_real_lens_traced(
                ens, prescription=rx,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_real_lens_traced' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


class TestNewSibling_apply_real_lens_maslov:

    def test_rejects_partial_coherence_mcf(self):
        mcf = _make_mcf()
        rx = _minimal_singlet_prescription()
        with pytest.raises(TypeError) as excinfo:
            apply_real_lens_maslov(
                mcf, prescription=rx,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'PartialCoherenceMCF' in msg
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the
        # rejection message no longer cites the stale "v4.16+"
        # scope marker; it now points at the propagate_ensemble
        # helper.  Pin against the new canonical hint.
        assert 'propagate_ensemble' in msg
        assert 'apply_real_lens_maslov' in msg

    def test_rejects_3d_ensemble(self):
        ens = _make_ensemble()
        rx = _minimal_singlet_prescription()
        with pytest.raises(ValueError) as excinfo:
            apply_real_lens_maslov(
                ens, prescription=rx,
                wavelength=WAVELENGTH, dx=DX)
        msg = str(excinfo.value)
        assert 'apply_real_lens_maslov' in msg
        assert 'for k in range(ensemble.shape[0])' in msg


# ============================================================================
# Migration sanity: the 10 v4.15.2-guarded sites still reject identically
# after migrating onto the shared helper.  Pre-fix the 10 sites had
# inline duplicated boilerplate; the v4.15.3 migration consolidates onto
# ``_check_2d_scalar_field``.  The error-message contract that
# ``test_v4_15_2_agent_c.py`` pins MUST still hold.
# ============================================================================

class TestMigrationSanityV4_15_2GuardedSitesIntact:
    """Smoke test on every v4.15.2-guarded site: the migrated helper
    delivers the same TypeError/ValueError + key message tokens.
    """

    def test_propagate_through_system_still_rejects_mcf_and_ensemble(self):
        mcf = _make_mcf()
        elements = [{'type': 'propagate', 'z': 1e-3}]
        with pytest.raises(TypeError) as excinfo:
            propagate_through_system(
                mcf, elements, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            propagate_through_system(
                ens, elements, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_propagate_dispatch_still_rejects_mcf_and_ensemble(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            propagate(mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX,
                      method='asm')
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            propagate(ens, z=1e-3, wavelength=WAVELENGTH, dx=DX,
                      method='asm')
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_angular_spectrum_propagate_still_rejects(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            angular_spectrum_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            angular_spectrum_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_fresnel_propagate_still_rejects(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            fresnel_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            fresnel_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_fraunhofer_propagate_still_rejects(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            fraunhofer_propagate(
                mcf, z=1.0, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            fraunhofer_propagate(
                ens, z=1.0, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_rayleigh_sommerfeld_propagate_still_rejects(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            rayleigh_sommerfeld_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            rayleigh_sommerfeld_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_scalable_angular_spectrum_propagate_still_rejects(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            scalable_angular_spectrum_propagate(
                mcf, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            scalable_angular_spectrum_propagate(
                ens, z=1e-3, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_apply_thin_lens_still_rejects(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_thin_lens(mcf, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_thin_lens(ens, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_apply_cylindrical_lens_still_rejects(self):
        mcf = _make_mcf()
        with pytest.raises(TypeError) as excinfo:
            apply_cylindrical_lens(
                mcf, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_cylindrical_lens(
                ens, f=10e-3, wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)

    def test_apply_real_lens_still_rejects(self):
        mcf = _make_mcf()
        rx = _minimal_singlet_prescription()
        with pytest.raises(TypeError) as excinfo:
            apply_real_lens(
                mcf, prescription=rx,
                wavelength=WAVELENGTH, dx=DX)
        # v4.16.1 (audit item 5b): stale "v4.16+" scope marker
        # retired; rejection now points at propagate_ensemble.
        assert 'propagate_ensemble' in str(excinfo.value)
        ens = _make_ensemble()
        with pytest.raises(ValueError) as excinfo:
            apply_real_lens(
                ens, prescription=rx,
                wavelength=WAVELENGTH, dx=DX)
        assert 'ensemble.shape[0]' in str(excinfo.value)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
