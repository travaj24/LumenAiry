"""v4.15.1 Agent-B tests -- P0-NEW-2 Schell-family partial-coherence redesign.

The v4.15.0 release of ``create_gaussian_schell_source``,
``create_schell_model_source``, and ``create_annular_incoherent_source``
shipped a contract bug (3-way confirmed in
``docs/audits/AUDIT_V4_15_0_2026_05_18.md``, P0-NEW-2): all three
factories computed an ensemble of random-phase realisations and then
COLLAPSED it to a single complex field
``E = sqrt(<I>) * exp(i * angle(<E>))`` before return.  The returned
single field has fully-coherent MCF ``J(r1, r2) = E*(r1) * E(r2)``,
NOT the documented Schell-model factorisation
``sqrt(I(r1) I(r2)) * mu(r1 - r2)``.  Plus the random-phase RMS
normalisation forced ``sigma_phi == 1`` regardless of ``sigma_g`` so
neither the ``sigma_g -> 0`` nor the ``sigma_g -> infinity`` limit
was recovered.

v4.15.1 (P0-NEW-2) redesigns the 3 factories:

* Default ``return_kind='ensemble'``: the factory returns the raw
  ``(n_realizations, Ny, Nx)`` complex ensemble plus ``dx``, ``dy``,
  ``wavelength``.  The caller iterates over realisations and averages
  intensities downstream.
* Opt-in ``return_kind='mcf'``: the factory returns a
  :class:`PartialCoherenceMCF` instance.  For small N (<= 64) the
  dense ``J_full`` matrix is stored; for larger N the coherent-mode
  decomposition (Wolf 1982 JOSA 72 343) keeps the leading-K modes
  selected by a 99% cumulative-eigenvalue threshold.

The random-phase recipe is now a band-limited complex Gaussian noise
field with the correct two-point correlation structure: complex white
noise -> Fourier-space Gaussian filter ``exp(-|k|^2 sigma_g^2 / 4)``
-> IFFT -> unit-mean-intensity renormalisation.  The result has
``<phi(r1) conj(phi(r2))> = exp(-|r1-r2|^2 / (2 sigma_g^2))`` --
exactly the Gaussian Schell kernel.

Reference: Goodman, _Statistical Optics_, Sec 5.5;
Mandel & Wolf, _Optical Coherence and Quantum Optics_, Section 4.7.

Author: Andrew Traverso -- v4.15.1 / Agent B
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la
from lumenairy.sources.core import (
    PartialCoherenceMCF,
    Source,
    create_annular_incoherent_source,
    create_gaussian_schell_source,
    create_schell_model_source,
)


# ============================================================================
# Ensemble contract: shape, dtype, return-value layout
# ============================================================================

class TestEnsembleContract:
    """All three Schell-family factories must return the raw
    ``(n_realizations, Ny, Nx)`` complex ensemble by default."""

    def test_gaussian_schell_returns_ensemble_shape(self):
        ens, dx, dy, wl = create_gaussian_schell_source(
            N=32, dx=2e-6, wavelength=633e-9,
            w0=40e-6, sigma_g=10e-6,
            n_realizations=8, seed=0)
        assert isinstance(ens, np.ndarray)
        assert ens.shape == (8, 32, 32)
        assert np.iscomplexobj(ens)
        assert dx == pytest.approx(2e-6)
        assert dy == pytest.approx(2e-6)
        assert wl == pytest.approx(633e-9)

    def test_schell_model_returns_ensemble_shape(self):
        N = 32
        xs = np.linspace(-1, 1, N)
        I_target = np.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / 0.25)
        ens, dx, dy, wl = create_schell_model_source(
            N=N, dx=2e-6, wavelength=633e-9,
            intensity_profile=I_target,
            coherence_length=10e-6,
            n_realizations=8, seed=0)
        assert isinstance(ens, np.ndarray)
        assert ens.shape == (8, N, N)
        assert np.iscomplexobj(ens)
        assert dx == pytest.approx(2e-6)
        assert dy == pytest.approx(2e-6)
        assert wl == pytest.approx(633e-9)

    def test_annular_incoherent_returns_ensemble_shape(self):
        ens, dx, dy, wl = create_annular_incoherent_source(
            N=32, dx=2e-6, wavelength=633e-9,
            inner_radius=20e-6, outer_radius=50e-6,
            n_realizations=8, seed=0)
        assert isinstance(ens, np.ndarray)
        assert ens.shape == (8, 32, 32)
        assert np.iscomplexobj(ens)
        assert dx == pytest.approx(2e-6)
        assert dy == pytest.approx(2e-6)
        assert wl == pytest.approx(633e-9)

    def test_ensemble_realisations_are_independent(self):
        """Each realisation in the ensemble must differ from the
        others (no accidental aliasing / cached single-field bug)."""
        ens, _, _, _ = create_gaussian_schell_source(
            N=16, dx=2e-6, wavelength=633e-9,
            w0=20e-6, sigma_g=8e-6,
            n_realizations=4, seed=0)
        for i in range(ens.shape[0]):
            for j in range(i + 1, ens.shape[0]):
                # Phase fields must not coincide; intensity overlap is
                # OK (both have the same envelope) but the complex field
                # must differ noticeably across realisations.
                diff = float(np.mean(np.abs(ens[i] - ens[j])))
                same = float(np.mean(np.abs(ens[i])))
                assert diff > 0.05 * same, (
                    f"Realisation {i} and {j} are nearly identical; "
                    f"the ensemble is collapsed.  diff/same={diff/same}.")


# ============================================================================
# Schell-theorem physics: Gaussian-Schell coherence kernel + limits
# ============================================================================

def _mcf_from_ensemble_dense(ens: np.ndarray) -> np.ndarray:
    """Build the dense MCF directly from the ensemble for ground-truth
    comparison in tests.  Independent of the production code path so
    we cross-check the production ``PartialCoherenceMCF`` API."""
    nr, Ny, Nx = ens.shape
    E_mat = ens.reshape(nr, Ny * Nx)
    J = (E_mat.conj().T @ E_mat) / float(nr)
    return J  # (Ny*Nx, Ny*Nx) complex


class TestSchellPhysics:
    """Validate that the new factories deliver the documented Schell-
    model coherence kernel + correct limits.

    These tests are the audit-flagged P0-NEW-2 ground truth.
    """

    def test_gaussian_schell_coherence_kernel_matches_theory(self):
        """The two-point coherence ``|mu(r1, r2)|`` between the centre
        pixel and a horizontal neighbour at separation ``Delta`` must
        match the target Gaussian kernel
        ``exp(-Delta^2 / (2 sigma_g^2))`` to within ~15% (a moderate
        finite-sample tolerance on a 64-realisation MCF estimate)."""
        N = 32
        dx = 2e-6
        sigma_g = 20e-6  # 10 pixels
        w0 = 400e-6  # essentially flat across the grid
        n_real = 64
        ens, _, _, _ = create_gaussian_schell_source(
            N=N, dx=dx, wavelength=633e-9,
            w0=w0, sigma_g=sigma_g,
            n_realizations=n_real, seed=42)
        J = _mcf_from_ensemble_dense(ens)
        cy, cx = N // 2, N // 2
        flat_c = cy * N + cx
        # Sample several separations along the +x axis.
        for k in (1, 3, 5, 8):
            flat_k = cy * N + (cx + k)
            I1 = np.abs(J[flat_c, flat_c])
            I2 = np.abs(J[flat_k, flat_k])
            mu = np.abs(J[flat_c, flat_k]) / np.sqrt(I1 * I2)
            expected = np.exp(-(k * dx) ** 2 / (2.0 * sigma_g ** 2))
            assert abs(mu - expected) < 0.15, (
                f"Schell kernel mismatch at Delta = {k}*dx = {k*dx*1e6:.1f}um: "
                f"measured |mu| = {mu:.3f}, expected {expected:.3f} "
                f"(sigma_g = {sigma_g*1e6:.1f}um).")

    def test_gaussian_schell_incoherent_limit(self):
        """``sigma_g -> 0`` (much smaller than the grid pitch) drives
        the source to the incoherent limit: the off-diagonal MCF
        decays to zero.

        Statistical bound: with ``n_realizations`` independent random-
        phase draws, the off-diagonal coherence ``|mu_{ij}|`` has
        sample-mean fluctuation ~``1 / sqrt(n_realizations)`` even in
        the true-zero limit.  Use 2000 realisations on a small grid and
        a generous tolerance.
        """
        N = 8
        dx = 2e-6
        ens, _, _, _ = create_gaussian_schell_source(
            N=N, dx=dx, wavelength=633e-9,
            w0=40e-6, sigma_g=dx / 10.0,
            n_realizations=2000, seed=0)
        J = _mcf_from_ensemble_dense(ens)
        diag = np.real(np.diag(J))
        # Use the centre pixel as the reference (avoids the support
        # boundary where I -> 0 and mu blows up numerically).
        cy, cx = N // 2, N // 2
        flat_c = cy * N + cx
        I_c = diag[flat_c]
        # Average |mu| over all other on-support pixels.
        peak = float(diag.max())
        on_support = diag > 0.05 * peak
        on_support[flat_c] = False  # exclude the centre itself
        idx = np.where(on_support)[0]
        mu_vals = (np.abs(J[flat_c, idx])
                   / np.sqrt(I_c * np.real(diag[idx])))
        # Statistical bound: mean off-diagonal |mu| ~ 1/sqrt(n_real).
        # 1/sqrt(2000) ~ 0.022, set tolerance at 0.10 (5x bound).
        assert float(np.mean(mu_vals)) < 0.10, (
            f"Incoherent limit failed: mean off-diagonal |mu| = "
            f"{float(np.mean(mu_vals)):.3f}, expected < 0.10.")

    def test_gaussian_schell_coherent_limit(self):
        """``sigma_g >> grid extent`` drives the source to the
        coherent limit: the MCF is effectively rank 1 (one dominant
        coherent mode).  The largest eigenvalue must account for >=
        99% of the total energy."""
        N = 32
        dx = 2e-6
        sigma_g = 100.0 * (N * dx)  # 100x the grid extent
        ens, _, _, _ = create_gaussian_schell_source(
            N=N, dx=dx, wavelength=633e-9,
            w0=40e-6, sigma_g=sigma_g,
            n_realizations=16, seed=0)
        # Use the production MCF builder; this also exercises the
        # full-storage path on N=32.
        mcf = PartialCoherenceMCF.from_ensemble(
            ens, dx=dx, dy=dx, wavelength=633e-9,
            max_full_N=64)
        modes, eigvals = mcf.coherent_modes()
        assert eigvals[0] > 0.0
        total = float(np.sum(eigvals))
        ratio = float(eigvals[0]) / total
        assert ratio > 0.99, (
            f"Coherent limit failed: leading eigenvalue / total = "
            f"{ratio:.4f}, expected > 0.99.  sigma_g = {sigma_g*1e6:.1f}um, "
            f"grid extent = {N*dx*1e6:.1f}um.")


# ============================================================================
# MCF return mode
# ============================================================================

class TestMCFReturnMode:
    """``return_kind='mcf'`` produces a :class:`PartialCoherenceMCF`."""

    def test_return_kind_mcf_returns_mcf_object(self):
        result = create_gaussian_schell_source(
            N=16, dx=2e-6, wavelength=633e-9,
            w0=20e-6, sigma_g=8e-6,
            n_realizations=8, seed=0,
            return_kind='mcf')
        assert isinstance(result, PartialCoherenceMCF)
        assert result.shape == (16, 16)
        assert result.dx == pytest.approx(2e-6)
        assert result.dy == pytest.approx(2e-6)
        assert result.wavelength == pytest.approx(633e-9)
        assert result.n_realizations == 8

    def test_mcf_intensity_matches_ensemble_mean(self):
        """``mcf.intensity()`` must match ``mean(|E|^2, axis=0)`` to
        within numerical round-off when both are computed from the
        same ensemble."""
        kw = dict(N=16, dx=2e-6, wavelength=633e-9,
                  w0=20e-6, sigma_g=8e-6,
                  n_realizations=8, seed=0)
        ens, _, _, _ = create_gaussian_schell_source(**kw)
        # Pass the same ensemble through the dense-builder, matching
        # the seed/n_realizations exactly.
        mcf = PartialCoherenceMCF.from_ensemble(
            ens, dx=kw['dx'], dy=kw['dx'], wavelength=kw['wavelength'])
        I_ensemble = np.mean(np.abs(ens) ** 2, axis=0)
        I_mcf = mcf.intensity()
        assert np.allclose(I_mcf, I_ensemble, atol=1e-6, rtol=1e-6)

    def test_mcf_coherent_modes_eigenvalues_descending(self):
        """``coherent_modes()`` returns eigenvalues in descending
        non-negative order, summing to the total source power."""
        ens, _, _, _ = create_gaussian_schell_source(
            N=16, dx=2e-6, wavelength=633e-9,
            w0=20e-6, sigma_g=8e-6,
            n_realizations=16, seed=0)
        mcf = PartialCoherenceMCF.from_ensemble(
            ens, dx=2e-6, dy=2e-6, wavelength=633e-9)
        modes, eigvals = mcf.coherent_modes()
        # Descending order.
        diffs = np.diff(eigvals)
        assert np.all(diffs <= 1e-12), (
            f"Eigenvalues not in descending order; diffs = {diffs}.")
        # All non-negative.
        assert np.all(eigvals >= -1e-12), (
            f"Negative eigenvalue: min = {eigvals.min()}.")
        # Sum to the total source power = trace(J) = sum_pixel I(r).
        I_total = float(np.sum(mcf.intensity()))
        assert abs(float(np.sum(eigvals)) - I_total) < 1e-6 * max(I_total, 1.0)

    def test_mcf_small_N_full_storage(self):
        """N=16 < default ``max_full_N=64`` -> full MCF stored,
        Hermitian, positive semi-definite."""
        result = create_gaussian_schell_source(
            N=16, dx=2e-6, wavelength=633e-9,
            w0=20e-6, sigma_g=8e-6,
            n_realizations=8, seed=0,
            return_kind='mcf')
        assert result.J_full is not None
        assert result.modes is None
        assert result.eigenvalues is None
        J = result.J_full
        assert J.shape == (16 * 16, 16 * 16)
        # Hermitian: J == J.conj().T.
        herm_err = float(np.max(np.abs(J - J.conj().T)))
        # Floating-point round-off from the (E^H @ E) build only.
        assert herm_err < 1e-10, (
            f"MCF not Hermitian: max(|J - J^H|) = {herm_err}.")
        # PSD: all eigvals >= 0 (allow tiny negative from round-off).
        w = np.linalg.eigvalsh(J)
        assert float(w.min()) >= -1e-10, (
            f"MCF not PSD: min eigenvalue = {w.min()}.")

    def test_mcf_large_N_modal_storage(self):
        """N=128 too large for full ``128^2 x 128^2`` storage at the
        default ``max_full_N=64`` (==> 64**2 = 4096 == 16384? no:
        128*128 = 16384 > 64**2 = 4096).  Modal storage kicks in:
        ``modes`` + ``eigenvalues`` populated, ``J_full`` is None."""
        result = create_gaussian_schell_source(
            N=128, dx=2e-6, wavelength=633e-9,
            w0=200e-6, sigma_g=40e-6,
            n_realizations=16, seed=0,
            return_kind='mcf')
        assert result.J_full is None
        assert result.modes is not None
        assert result.eigenvalues is not None
        # Modes are (K, Ny, Nx).
        K, Ny, Nx = result.modes.shape
        assert (Ny, Nx) == (128, 128)
        # K <= min(n_realizations, N_pix).
        assert 1 <= K <= 16
        # eigenvalues has shape (K,).
        assert result.eigenvalues.shape == (K,)


# ============================================================================
# Return-kind validation
# ============================================================================

class TestReturnKindValidation:
    """``return_kind`` must be 'ensemble' or 'mcf'."""

    def test_invalid_return_kind_raises_gaussian_schell(self):
        with pytest.raises(ValueError) as info:
            create_gaussian_schell_source(
                N=16, dx=2e-6, wavelength=633e-9,
                w0=20e-6, sigma_g=8e-6,
                n_realizations=4, seed=0,
                return_kind='bogus')
        assert 'return_kind' in str(info.value)

    def test_invalid_return_kind_raises_schell_model(self):
        with pytest.raises(ValueError):
            create_schell_model_source(
                N=16, dx=2e-6, wavelength=633e-9,
                intensity_profile=np.ones((16, 16)),
                coherence_length=8e-6,
                n_realizations=4, seed=0,
                return_kind='wrong')

    def test_invalid_return_kind_raises_annular_incoherent(self):
        with pytest.raises(ValueError):
            create_annular_incoherent_source(
                N=16, dx=2e-6, wavelength=633e-9,
                inner_radius=10e-6, outer_radius=30e-6,
                n_realizations=4, seed=0,
                return_kind=None)


# ============================================================================
# Source classmethod migration
# ============================================================================

class TestSourceClassmethodMigration:
    """``Source.gaussian_schell`` / ``Source.schell_model`` return the
    ensemble tuple (v4.15.2+) or the bare ``PartialCoherenceMCF`` --
    NOT a :class:`Source`-wrapped 3-D ensemble.

    v4.15.2 (AUDIT_V4_15_1 P2 closure): the v4.15.1-era contract that
    wrapped a 3-D ensemble inside a :class:`Source` whose ``E`` was
    3-D broke the Source contract (every other ``Source.*``
    classmethod produces a 2-D field).  v4.15.2 aligns the classmethod
    return type with the top-level factory --
    :meth:`Source.gaussian_schell` now returns the same 4-tuple
    ``(ensemble, dx, dy, wavelength)`` as
    :func:`create_gaussian_schell_source`.
    """

    def test_source_gaussian_schell_returns_ensemble_tuple(self):
        """v4.15.2 return contract: 4-tuple matching the top-level
        factory.  No :class:`Source` wrapping of the 3-D ensemble."""
        result = Source.gaussian_schell(
            N=16, dx=2e-6, wavelength=633e-9,
            w0=20e-6, sigma_g=8e-6,
            n_realizations=4, seed=0)
        assert not isinstance(result, Source), (
            "v4.15.2: Source.gaussian_schell must NOT wrap a 3-D "
            "ensemble in a Source (every other Source.* produces a "
            "2-D field; partial-coherence is a different abstraction).")
        # Must be the (ensemble, dx, dy, wavelength) 4-tuple.
        assert isinstance(result, tuple)
        assert len(result) == 4
        ens, _dx, _dy, _wl = result
        assert isinstance(ens, np.ndarray)
        assert ens.shape == (4, 16, 16)
        assert np.iscomplexobj(ens)
        assert _dx == pytest.approx(2e-6)
        assert _wl == pytest.approx(633e-9)

    def test_source_gaussian_schell_with_return_kind_mcf(self):
        """``return_kind='mcf'`` returns the bare ``PartialCoherenceMCF``,
        NOT a Source.  The MCF object is consumable for inspection
        only -- propagator integration is deferred to v4.16+."""
        result = Source.gaussian_schell(
            N=16, dx=2e-6, wavelength=633e-9,
            w0=20e-6, sigma_g=8e-6,
            n_realizations=4, seed=0,
            return_kind='mcf')
        assert isinstance(result, PartialCoherenceMCF)
        assert result.shape == (16, 16)

    def test_source_schell_model_returns_ensemble_tuple(self):
        """``Source.schell_model`` follows the same v4.15.2 contract
        as ``Source.gaussian_schell`` -- ensemble tuple, no Source
        wrapping."""
        N = 16
        xs = np.linspace(-1, 1, N)
        I = np.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / 0.25)
        result = Source.schell_model(
            N=N, dx=2e-6, wavelength=633e-9,
            intensity_profile=I,
            coherence_length=8e-6,
            n_realizations=4, seed=0)
        assert not isinstance(result, Source)
        assert isinstance(result, tuple)
        assert len(result) == 4
        ens, _, _, _ = result
        assert isinstance(ens, np.ndarray)
        assert ens.shape == (4, N, N)


# ============================================================================
# Schell-model generic factory: intensity-profile honour + kernel
# ============================================================================

class TestSchellModelIntensityKernel:
    """``create_schell_model_source`` must honour BOTH the user-
    supplied intensity profile AND the Gaussian coherence kernel."""

    def test_schell_model_ensemble_mean_intensity_follows_profile(self):
        """The ensemble-mean intensity ``<|E|^2>`` must follow the
        user-supplied intensity profile up to a normalisation."""
        N = 32
        xs = np.linspace(-1, 1, N)
        I_target = np.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / 0.25)
        ens, _, _, _ = create_schell_model_source(
            N=N, dx=10e-6, wavelength=633e-9,
            intensity_profile=I_target,
            coherence_length=20e-6,
            n_realizations=32, seed=0)
        I_out = np.mean(np.abs(ens) ** 2, axis=0)
        corr = float(np.corrcoef(I_out.ravel(), I_target.ravel())[0, 1])
        assert corr > 0.9, (
            f"Schell-model ensemble mean intensity must correlate with "
            f"the supplied profile; got corr = {corr:.3f}.")


# ============================================================================
# Annular incoherent: support + incoherence at the source plane
# ============================================================================

class TestAnnularIncoherentSupport:
    """Annular incoherent source -- per-realisation support inside
    the annulus + ensemble-averaged intensity diagonal-MCF
    (incoherent-source character)."""

    def test_annular_inco_support_inside_annulus(self):
        N = 32
        dx = 5e-6
        inner = 30e-6
        outer = 60e-6
        ens, _, _, _ = create_annular_incoherent_source(
            N=N, dx=dx, wavelength=633e-9,
            inner_radius=inner, outer_radius=outer,
            n_realizations=4, seed=0)
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        r = np.sqrt(X * X + Y * Y)
        # Every realisation vanishes outside the annulus.
        for k in range(ens.shape[0]):
            inside_mask = r < inner * 0.9
            outside_mask = r > outer * 1.1
            peak = float(np.abs(ens[k]).max())
            assert float(np.abs(ens[k][inside_mask]).max()) < peak * 0.05, (
                f"Annular realisation {k} leaks inside inner_radius.")
            assert float(np.abs(ens[k][outside_mask]).max()) < peak * 0.05, (
                f"Annular realisation {k} leaks outside outer_radius.")


# ============================================================================
# v4.15.2 (AUDIT_V4_15_1 P2 closure): coherence_at Hermiticity unit test
# ============================================================================

class TestMCFCoherenceAtHermiticity:
    """``PartialCoherenceMCF.coherence_at(r1, r2)`` must satisfy the
    Hermiticity property

        ``mu(r1, r2) == conj(mu(r2, r1))``

    for every grid-point pair (Mandel & Wolf, Optical Coherence and
    Quantum Optics Section 4.2; Goodman Statistical Optics Section
    5.2).  v4.15.1 shipped ``coherence_at`` without a direct
    Hermiticity unit test (the audit's V4 finding noted "property
    holds by construction; uncovered").  This pin covers the
    property on a small grid where every pair can be checked
    exhaustively.
    """

    def test_mcf_coherence_at_hermitian_dense_branch(self):
        """N=8 grid; dense ``J_full`` storage path.  Iterate over
        several ``(r1, r2)`` pairs and verify
        ``mu(r1, r2) == conj(mu(r2, r1))`` to machine precision."""
        ens, _, _, _ = create_gaussian_schell_source(
            N=8, dx=2e-6, wavelength=633e-9,
            w0=10e-6, sigma_g=4e-6,
            n_realizations=12, seed=0)
        mcf = PartialCoherenceMCF.from_ensemble(
            ens, dx=2e-6, dy=2e-6, wavelength=633e-9)
        # The 8x8 grid is small enough to default to dense J_full.
        assert mcf.J_full is not None, (
            "Expected dense J_full storage at N=8 < max_full_N.")
        # Spot-check 6 grid-pairs spanning the dynamic range:
        # near-diagonal (correlated), far-diagonal (low |mu|), and
        # a couple of mid-distance points.
        pairs = [
            ((0, 0), (0, 0)),       # diagonal -- mu == 1 by definition
            ((0, 0), (1, 0)),       # adjacent -- near-1 |mu|
            ((0, 0), (2, 2)),       # close
            ((0, 0), (7, 7)),       # corner-to-corner (low |mu|)
            ((3, 4), (5, 2)),       # arbitrary interior pair
            ((1, 6), (6, 1)),       # off-diagonal symmetric pair
        ]
        for (ix1, iy1), (ix2, iy2) in pairs:
            mu_12 = mcf.coherence_at(ix1, iy1, ix2, iy2)
            mu_21 = mcf.coherence_at(ix2, iy2, ix1, iy1)
            err = abs(mu_12 - np.conj(mu_21))
            assert err < 1e-10, (
                f"Hermiticity violated for pair "
                f"r1=({ix1},{iy1}) r2=({ix2},{iy2}): "
                f"mu_12={mu_12}, mu_21={mu_21}, "
                f"|mu_12 - conj(mu_21)|={err} > 1e-10.")
        # Also check the diagonal is exactly 1+0j (or 0 if I=0).
        for ix in range(8):
            for iy in range(8):
                mu_dd = mcf.coherence_at(ix, iy, ix, iy)
                if abs(mu_dd) > 0.0:
                    assert abs(mu_dd - 1.0) < 1e-10, (
                        f"mu(r,r) should be 1+0j; got {mu_dd} at "
                        f"(ix={ix}, iy={iy}).")

    def test_mcf_coherence_at_hermitian_modal_branch(self):
        """N=80 forces modal storage (``J_full`` is None, eigenvectors
        + eigenvalues populated).  Same Hermiticity property -- the
        modal-branch ``coherence_at`` builds ``J(r1, r2)`` from
        ``sum_k lambda_k phi_k(r1) phi_k(r2)^*`` which is Hermitian by
        construction; the pin asserts the construction is honoured."""
        result = create_gaussian_schell_source(
            N=80, dx=2e-6, wavelength=633e-9,
            w0=80e-6, sigma_g=30e-6,
            n_realizations=8, seed=0,
            return_kind='mcf')
        # max_full_N defaults to 64 so N=80 should be modal.
        assert result.J_full is None, (
            "Expected modal storage at N=80 > max_full_N=64.")
        assert result.modes is not None
        # A handful of pairs in the modal branch.
        pairs = [
            ((0, 0), (10, 10)),
            ((20, 5), (40, 30)),
            ((39, 39), (40, 40)),
            ((10, 70), (70, 10)),
        ]
        for (ix1, iy1), (ix2, iy2) in pairs:
            mu_12 = result.coherence_at(ix1, iy1, ix2, iy2)
            mu_21 = result.coherence_at(ix2, iy2, ix1, iy1)
            err = abs(mu_12 - np.conj(mu_21))
            assert err < 1e-10, (
                f"Modal-branch Hermiticity violated for pair "
                f"r1=({ix1},{iy1}) r2=({ix2},{iy2}): err={err}.")
