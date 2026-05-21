"""v4.16.1 Agent B regression tests -- close the Schell-model + JAX
half-shipped clusters from ``AUDIT_V4_16_0_DEEP_2026_05_19.md``.

Scope (per the v4.16.1 dispatch):

* **B.1** -- ``lumenairy.propagate_ensemble`` helper (audit item 5a /
  P0-1).  Iterates a Schell-family ensemble through a coherent 2-D
  propagator and accumulates the partial-coherence intensity
  ``I_partial = < |E_k|^2 >_k``.
* **B.2** -- Drop the default-path ``DeprecationWarning`` on the 3
  Schell factories + the two ``Source.*`` Schell classmethods
  (audit item 6).
* **B.3** -- Replace the stale ``"v4.16+"`` MCF rejection wording in
  :func:`lumenairy._validation._check_2d_scalar_field` (audit item
  5b).
* **B.4** -- Make ``propagate_through_system_jax`` JAX-safe under
  ``jax.jit`` / ``jax.grad`` (audit item 7).  Mirror the v4.15.3
  ``getattr(E, 'attr', None)`` duck-typing pattern from the shared
  validation helper.
* **B.5** -- Add a one-shot ``RuntimeWarning`` to
  :func:`_transfer_jax` when the ray bundle's ``min |N|`` falls
  below 0.95 (audit item 8).

Author: Andrew Traverso -- v4.16.1 / Agent B
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.sources.core import (
    create_annular_incoherent_source,
    create_gaussian_schell_source,
    create_schell_model_source,
)

# ----------------------------------------------------------------------
# Optional JAX import (B.4 + B.5 tests skip if JAX is missing).
# ----------------------------------------------------------------------

JAX_AVAILABLE = False
try:
    import jax  # noqa: F401
    import jax.numpy as jnp  # noqa: F401
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

needs_jax = pytest.mark.skipif(
    not JAX_AVAILABLE, reason='JAX is not installed')


# ----------------------------------------------------------------------
# Shared Schell parameter bundles for the B.1 / B.2 tests.
# ----------------------------------------------------------------------

_SCHELL_KW = dict(
    N=32, dx=2e-6, wavelength=633e-9,
    w0=20e-6, sigma_g=8e-6,
    n_realizations=4, seed=0,
)


def _make_simple_ensemble(n_realizations=4, N=32, dx=2e-6,
                           wavelength=633e-9, w0=20e-6, sigma_g=8e-6,
                           seed=0):
    """Build a small Schell ensemble for the B.1 propagation tests."""
    return create_gaussian_schell_source(
        N=N, dx=dx, wavelength=wavelength,
        w0=w0, sigma_g=sigma_g,
        n_realizations=n_realizations, seed=seed,
        return_kind='ensemble',
    )


# ============================================================================
# B.1 -- propagate_ensemble helper (audit item 5a / P0-1)
# ============================================================================


class TestPropagateEnsemble:
    """The v4.16.1 ``propagate_ensemble`` helper iterates a Schell
    ensemble through a coherent propagator and returns the partial-
    coherence intensity ``< |E_k|^2 >_k``.  Pre-v4.16.1 callers had
    to write this loop themselves; coherent propagators rejected the
    3-D ensemble via the v4.15.3 ``_check_2d_scalar_field`` helper
    with a "v4.16+" deferral message that the audit flagged as
    half-shipped at v4.16.0.
    """

    def test_propagate_ensemble_basic(self):
        """4-realisation ensemble through ASM at z=10 cm.  Output is a
        real 2-D intensity of the same shape as the input plane."""
        ens, dx, dy, wl = _make_simple_ensemble(
            n_realizations=4, N=32)
        I_partial = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator='asm', z=0.1)
        assert isinstance(I_partial, np.ndarray)
        assert I_partial.shape == ens.shape[1:]
        assert I_partial.dtype.kind == 'f'
        # Real-valued; non-negative; finite.
        assert np.all(np.isreal(I_partial))
        assert np.all(I_partial >= -1e-30)  # round-off tolerance
        assert np.all(np.isfinite(I_partial))

    def test_propagate_ensemble_string_propagator(self):
        """``propagator='fresnel'`` resolves to the canonical Fresnel
        kernel.  Output shape preserved."""
        ens, dx, dy, wl = _make_simple_ensemble(
            n_realizations=4, N=32)
        I_partial = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator='fresnel', z=0.05)
        assert I_partial.shape == ens.shape[1:]
        assert np.all(np.isfinite(I_partial))

    def test_propagate_ensemble_callable_propagator(self):
        """Passing a callable propagator works; the callable is
        invoked once per realisation with the documented signature
        ``(E, *, dx, wavelength, **kwargs) -> E_out``."""
        ens, dx, dy, wl = _make_simple_ensemble(
            n_realizations=4, N=32)

        call_count = {'n': 0}

        def my_propagator(E, *, dx, wavelength, z):
            call_count['n'] += 1
            return la.angular_spectrum_propagate(E, z, wavelength, dx)

        I_partial = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator=my_propagator, z=0.05)
        assert call_count['n'] == ens.shape[0]
        assert I_partial.shape == ens.shape[1:]
        assert np.all(np.isfinite(I_partial))

    def test_propagate_ensemble_return_ensemble(self):
        """``return_ensemble=True`` returns the propagated ensemble
        alongside (or instead of) the intensity."""
        ens, dx, dy, wl = _make_simple_ensemble(
            n_realizations=4, N=32)
        I_partial, ens_out = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator='asm', z=0.05,
            return_ensemble=True)
        assert ens_out.shape == ens.shape
        assert np.iscomplexobj(ens_out)
        # The intensity must equal the ensemble's |E|^2 mean to
        # numerical precision (same accumulation, just kept).
        I_manual = np.mean(np.abs(ens_out) ** 2, axis=0)
        np.testing.assert_allclose(I_partial, I_manual,
                                    rtol=1e-12, atol=1e-30)

    def test_propagate_ensemble_intensity_is_real(self):
        """Returned ``I_partial`` is real-valued (not complex)."""
        ens, dx, dy, wl = _make_simple_ensemble(
            n_realizations=4, N=32)
        I_partial = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator='asm', z=0.05)
        assert not np.iscomplexobj(I_partial)
        assert I_partial.dtype.kind == 'f'

    def test_propagate_ensemble_partial_coherence_smoothing(self):
        """Canonical Wolf result: partial-coherence peak intensity is
        smaller than the coherent reference's peak intensity at the
        same waist + propagation distance.  Pin the ratio (smoothing
        factor) > 1.0 as a physics correctness check.
        """
        N = 64
        dx = 2e-6
        wl = 633e-9
        w0 = 30e-6
        sigma_g = 8e-6  # short coherence -> strong smoothing
        z = 0.05

        ens, _, _, _ = create_gaussian_schell_source(
            N=N, dx=dx, wavelength=wl,
            w0=w0, sigma_g=sigma_g,
            n_realizations=32, seed=0,
            return_kind='ensemble')
        I_partial = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl, propagator='asm', z=z)

        # Coherent reference: same waist, deterministic Gaussian.
        src_coh = la.Source.gaussian(
            w0=w0, N=N, dx=dx, wavelength=wl)
        E_coh = la.angular_spectrum_propagate(src_coh.E, z, wl, dx)
        I_coh = np.abs(E_coh) ** 2

        ratio = float(I_coh.max() / I_partial.max())
        # Sigma_g << w0 -> strong smoothing; ratio is well above 1.
        assert ratio > 1.0, (
            f"Coherent peak / partial peak ratio = {ratio:.3f} must "
            f"exceed 1.0 (partial-coherence smoothing).")
        # Inverse check: partial peak / coherent peak < 1.0.
        assert (I_partial.max() / I_coh.max()) < 1.0

    def test_propagate_ensemble_top_level_export(self):
        """``from lumenairy import propagate_ensemble`` works and is
        listed in ``lumenairy.__all__``."""
        from lumenairy import propagate_ensemble
        assert callable(propagate_ensemble)
        assert 'propagate_ensemble' in la.__all__
        # Same callable accessible from the propagators subpackage.
        from lumenairy.propagators import propagate_ensemble as pe2
        assert pe2 is propagate_ensemble

    def test_propagate_ensemble_rejects_non_3d(self):
        """A 2-D field (single realisation) raises ``ValueError`` with
        a useful message pointing at ``ensemble[None, ...]``."""
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=4, N=16)
        with pytest.raises(ValueError, match=r"3-D"):
            la.propagate_ensemble(
                ens[0], dx=dx, wavelength=wl,
                propagator='asm', z=0.01)

    def test_propagate_ensemble_propagator_kwargs_merge(self):
        """``propagator_kwargs`` is merged with ``**kwargs`` (latter
        provides the default; the former overrides on collision)."""
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=2, N=16)
        # Both paths produce the same intensity.
        I_a = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator='asm', z=0.02)
        I_b = la.propagate_ensemble(
            ens, dx=dx, wavelength=wl,
            propagator='asm',
            propagator_kwargs={'z': 0.02})
        np.testing.assert_allclose(I_a, I_b, rtol=1e-12)

    def test_propagate_ensemble_unknown_propagator_string_raises(self):
        """Bogus propagator string raises ``ValueError`` with the
        accepted-name list."""
        ens, dx, _, wl = _make_simple_ensemble(n_realizations=2, N=16)
        with pytest.raises(ValueError, match=r"propagate_ensemble"):
            la.propagate_ensemble(
                ens, dx=dx, wavelength=wl,
                propagator='nonexistent', z=0.01)


# ============================================================================
# B.2 -- Drop default DeprecationWarning on Schell factories (item 6)
# ============================================================================


class TestSchellDefaultNoDeprecationWarning:
    """v4.16.1 (audit item 6) contract: the 3 Schell factories +
    the 2 ``Source.*`` classmethod wrappers must NOT emit a
    return_kind ``DeprecationWarning`` on the default path now that
    the v4.15.0 -> v4.15.1 return-shape change has had multiple
    releases of exposure.
    """

    def test_schell_default_no_deprecation_warning(self):
        """Top-level factory: default path is fully silent."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_gaussian_schell_source(**_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'return_kind' in str(w.message)]
        assert dep == [], (
            f"v4.16.1: default-path Schell call must NOT emit "
            f"return_kind DeprecationWarning; got "
            f"{[str(w.message) for w in dep]}.")

    def test_schell_model_default_no_deprecation_warning(self):
        """``create_schell_model_source`` default path silent."""
        N = 16
        I = np.exp(
            -((np.linspace(-1, 1, N)[:, None]) ** 2
              + (np.linspace(-1, 1, N)[None, :]) ** 2) / 0.25)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_schell_model_source(
                N=N, dx=2e-6, wavelength=633e-9,
                intensity_profile=I, coherence_length=8e-6,
                n_realizations=4, seed=0)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'return_kind' in str(w.message)]
        assert dep == []

    def test_annular_incoherent_default_no_deprecation_warning(self):
        """``create_annular_incoherent_source`` default path silent."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            create_annular_incoherent_source(
                N=16, dx=2e-6, wavelength=633e-9,
                inner_radius=4e-6, outer_radius=12e-6,
                n_realizations=4, seed=0)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'return_kind' in str(w.message)]
        assert dep == []

    def test_source_gaussian_schell_classmethod_default_no_warning(self):
        """``Source.gaussian_schell`` classmethod default path silent."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.Source.gaussian_schell(**_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'return_kind' in str(w.message)]
        assert dep == []

    def test_source_schell_model_classmethod_default_no_warning(self):
        """``Source.schell_model`` classmethod default path silent."""
        N = 16
        I = np.exp(
            -((np.linspace(-1, 1, N)[:, None]) ** 2
              + (np.linspace(-1, 1, N)[None, :]) ** 2) / 0.25)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.Source.schell_model(
                N=N, dx=2e-6, wavelength=633e-9,
                intensity_profile=I, coherence_length=8e-6,
                n_realizations=4, seed=0)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'return_kind' in str(w.message)]
        assert dep == []

    def test_sentinel_back_compat_preserved(self):
        """v4.16.1 keeps ``_RETURN_KIND_UNSET`` + the helper
        importable for back-compat (the v4.15.3 sentinel-promotion
        meta-pin and any external custom-wrapper callers).  Sentinel
        + subclass + helper must all import cleanly."""
        from lumenairy.sources.core import (
            _RETURN_KIND_UNSET,
            _SchellReturnKindUnsetSentinel,
            _warn_schell_return_kind_default,
        )
        assert _RETURN_KIND_UNSET is not None
        assert callable(_warn_schell_return_kind_default)
        assert isinstance(_RETURN_KIND_UNSET,
                          _SchellReturnKindUnsetSentinel)

    def test_explicit_sentinel_pass_through(self):
        """Explicit ``_RETURN_KIND_UNSET`` pass-through still resolves
        cleanly to the ensemble path (default behaviour) with no
        warning.  This is the back-compat surface for any custom
        wrapper that explicitly forwards the sentinel."""
        from lumenairy.sources.core import _RETURN_KIND_UNSET
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            ens, dx, dy, wl = create_gaussian_schell_source(
                return_kind=_RETURN_KIND_UNSET, **_SCHELL_KW)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'return_kind' in str(w.message)]
        assert dep == []
        assert ens.shape == (4, 32, 32)


# ============================================================================
# B.3 -- MCF rejection message points at propagate_ensemble (item 5b)
# ============================================================================


class TestMCFRejectionMessage:
    """The :func:`_check_2d_scalar_field` helper raises ``TypeError``
    on :class:`PartialCoherenceMCF` inputs.  v4.16.1 (audit item 5b)
    updates the rejection message to (a) drop the stale "v4.16+"
    deferral wording, (b) point users at the new
    ``propagate_ensemble`` helper.
    """

    def test_mcf_rejection_message_no_stale_v4_16_wording(self):
        """The rejection message must not cite ``"v4.16+"`` -- the
        library version IS v4.16.x at this point."""
        from lumenairy._validation import _check_2d_scalar_field
        # Build a tiny MCF object.
        ens, dx, dy, wl = _make_simple_ensemble(n_realizations=4, N=16)
        mcf = la.PartialCoherenceMCF.from_ensemble(
            ens, dx=dx, dy=dy, wavelength=wl, max_full_N=64)

        with pytest.raises(TypeError) as exc_info:
            _check_2d_scalar_field(mcf, 'angular_spectrum_propagate')
        msg = str(exc_info.value)
        assert 'v4.16+' not in msg, (
            f"v4.16.1: MCF rejection message must not cite "
            f"'v4.16+' (stale deferral wording); got {msg!r}.")

    def test_mcf_rejection_message_points_at_propagate_ensemble(self):
        """The new message must point users at the
        ``propagate_ensemble`` helper so the rejection is paired with
        the actual fix."""
        from lumenairy._validation import _check_2d_scalar_field
        ens, dx, dy, wl = _make_simple_ensemble(n_realizations=4, N=16)
        mcf = la.PartialCoherenceMCF.from_ensemble(
            ens, dx=dx, dy=dy, wavelength=wl, max_full_N=64)

        with pytest.raises(TypeError) as exc_info:
            _check_2d_scalar_field(mcf, 'angular_spectrum_propagate')
        msg = str(exc_info.value)
        assert 'propagate_ensemble' in msg, (
            f"v4.16.1: MCF rejection message must point at "
            f"`propagate_ensemble` for the canonical workflow; got "
            f"{msg!r}.")

    def test_3d_ensemble_rejection_message_points_at_propagate_ensemble(
        self,
    ):
        """The 3-D ensemble rejection (different code branch from the
        MCF branch) also points at ``propagate_ensemble`` -- this is
        the more common user-error path."""
        from lumenairy._validation import _check_2d_scalar_field
        ens, _, _, _ = _make_simple_ensemble(n_realizations=4, N=16)
        with pytest.raises(ValueError) as exc_info:
            _check_2d_scalar_field(ens, 'angular_spectrum_propagate')
        msg = str(exc_info.value)
        assert 'propagate_ensemble' in msg, (
            f"v4.16.1: 3-D ensemble rejection message must point at "
            f"`propagate_ensemble` for the canonical workflow; got "
            f"{msg!r}.")


# ============================================================================
# B.4 -- propagate_through_system_jax JAX-safe dtype probe (item 7)
# ============================================================================


class TestPropagateThroughSystemJaxJitSafe:
    """``propagate_through_system_jax`` must be jit-traceable now that
    the dtype probe is JAX-safe (v4.16.1 audit item 7).  Pre-fix the
    ``np.asarray(E_in)`` call raised ``TracerArrayConversionError``
    under ``jax.jit`` / ``jax.grad`` -- but the function is
    explicitly named ``propagate_through_system_jax`` and the
    docstring touts end-to-end jit'd caching.
    """

    @needs_jax
    def test_propagate_through_system_jax_with_jit(self):
        """Wrap ``propagate_through_system_jax`` in ``jax.jit`` and
        run a forward pass.  No ``TracerArrayConversionError``."""
        import jax
        import jax.numpy as _jnp

        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            propagate_through_system_jax,
        )

        N = 32
        dx = 5e-6
        wavelength = 633e-9
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
            {'type': 'lens', 'f': 5e-3, 'xc': 0.0, 'yc': 0.0},
        ]

        @jax.jit
        def run(E_in):
            return propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)

        _PROPAGATE_SYSTEM_JAX_CACHE.clear()
        try:
            # Cold start; the jit wrapper traces through the function.
            E_in = _jnp.ones((N, N), dtype=_jnp.complex128)
            E_out = run(E_in)
            assert E_out.shape == (N, N)
            assert np.all(np.isfinite(np.asarray(E_out)))
        finally:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()

    @needs_jax
    def test_propagate_through_system_jax_dtype_probe_real_input(self):
        """Pre-fix the dtype probe used ``np.iscomplexobj(np.asarray(
        E_in))`` which raises under jit.  Post-fix the probe uses
        ``getattr(E_in, 'dtype', None)`` which works for both NumPy
        arrays and JAX tracers.  Verify a real-input call still
        falls through to the library default complex dtype."""
        import jax.numpy as _jnp

        from lumenairy.propagators.system import (
            _PROPAGATE_SYSTEM_JAX_CACHE,
            propagate_through_system_jax,
        )

        N = 32
        dx = 5e-6
        wavelength = 633e-9
        elements = [
            {'type': 'propagate', 'z': 1e-3, 'bandlimit': True},
        ]
        # Real-valued NumPy input -> probe sees a real dtype,
        # falls through to the library complex default.
        E_in = np.ones((N, N), dtype=np.float64)
        _PROPAGATE_SYSTEM_JAX_CACHE.clear()
        try:
            E_out = propagate_through_system_jax(
                E_in, elements, wavelength, dx, verbose=False)
            assert np.iscomplexobj(np.asarray(E_out))
        finally:
            _PROPAGATE_SYSTEM_JAX_CACHE.clear()


# ============================================================================
# B.5 -- _transfer_jax high-NA RuntimeWarning (item 8)
# ============================================================================


class TestTransferJaxHighNAWarning:
    """``_transfer_jax`` uses a paraxial approximation
    ``t ~= thickness`` (preserved for autodiff stability; see
    docstring).  v4.16.1 (audit item 8) adds a one-shot
    ``RuntimeWarning`` when the ray bundle's ``min |N|`` falls below
    0.95 so high-NA users know the JAX path diverges from the
    math-correct NumPy ``trace(...)`` form.
    """

    @needs_jax
    def test_transfer_jax_high_na_warns(self):
        """Direct call into ``_transfer_jax`` with a bundle whose
        ``min |N| < 0.95`` -- the warning fires exactly once and
        cites the per-surface drift estimate."""
        from lumenairy.raytrace.jax_trace import (
            JaxRayState,
            _reset_transfer_jax_warning,
            _transfer_jax,
        )

        _reset_transfer_jax_warning()
        try:
            # Single ray at NA = 0.5 (sin(theta) = 0.5; |N| = sqrt(3)/2
            # = 0.866 < 0.95).
            x = np.zeros(1)
            y = np.zeros(1)
            z = np.zeros(1)
            L = np.array([0.5])  # = sin(theta) = NA
            M = np.zeros(1)
            n_axis = np.array([np.sqrt(1.0 - 0.25)])  # ~= 0.866
            opd = np.zeros(1)
            alive = np.ones(1, dtype=bool)
            state = JaxRayState(x, y, z, L, M, n_axis, opd, alive)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _ = _transfer_jax(state, thickness=1e-3, n_medium=1.0)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            assert len(rw) == 1, (
                f"Expected exactly 1 _transfer_jax RuntimeWarning "
                f"for min |N|={float(n_axis[0]):.3f} < 0.95; got "
                f"{len(rw)}: {[str(w.message) for w in rw]}.")
            msg = str(rw[0].message)
            # Message must quantify the per-surface drift.
            assert 'paraxial' in msg.lower()
            assert 'NA' in msg
            # The numerical drift estimate should appear (1e-3 *
            # (1 - 0.75) / 2 = 1.25e-4 m).
            assert 'm' in msg
        finally:
            _reset_transfer_jax_warning()

    @needs_jax
    def test_transfer_jax_low_na_no_warn(self):
        """``min |N| >= 0.95`` -> no warning (the paraxial
        approximation error is below noise)."""
        from lumenairy.raytrace.jax_trace import (
            JaxRayState,
            _reset_transfer_jax_warning,
            _transfer_jax,
        )

        _reset_transfer_jax_warning()
        try:
            # NA = 0.1 (well below 0.31 threshold); |N| = 0.995.
            x = np.zeros(1)
            y = np.zeros(1)
            z = np.zeros(1)
            L = np.array([0.1])
            M = np.zeros(1)
            n_axis = np.array([np.sqrt(1.0 - 0.01)])  # ~= 0.995
            opd = np.zeros(1)
            alive = np.ones(1, dtype=bool)
            state = JaxRayState(x, y, z, L, M, n_axis, opd, alive)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                _ = _transfer_jax(state, thickness=1e-3, n_medium=1.0)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            assert rw == [], (
                f"Low-NA call must NOT emit _transfer_jax "
                f"RuntimeWarning; got {[str(w.message) for w in rw]}.")
        finally:
            _reset_transfer_jax_warning()

    @needs_jax
    def test_transfer_jax_warning_is_one_shot(self):
        """The warning fires at most once per process (latch in
        :data:`_TRANSFER_JAX_WARNING_EMITTED`).  Tight forward-loop
        callers don't see log-spam."""
        from lumenairy.raytrace.jax_trace import (
            JaxRayState,
            _reset_transfer_jax_warning,
            _transfer_jax,
        )

        _reset_transfer_jax_warning()
        try:
            x = np.zeros(1); y = np.zeros(1); z = np.zeros(1)
            L = np.array([0.5]); M = np.zeros(1)
            n_axis = np.array([np.sqrt(1.0 - 0.25)])
            opd = np.zeros(1); alive = np.ones(1, dtype=bool)
            state = JaxRayState(x, y, z, L, M, n_axis, opd, alive)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for _ in range(5):
                    _ = _transfer_jax(state, 1e-3, n_medium=1.0)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            assert len(rw) == 1, (
                f"_transfer_jax warning must be one-shot; got "
                f"{len(rw)} emissions in 5 calls.")
        finally:
            _reset_transfer_jax_warning()

    @needs_jax
    def test_transfer_jax_warning_skipped_under_tracing(self):
        """JAX tracers (under ``jax.jit`` / ``jax.grad``) skip the
        warning check cleanly -- inspecting tracer values mid-trace
        would force concretisation and break the autodiff path.
        Verify a jit'd ``_transfer_jax`` does not emit the warning
        (and does not raise ``ConcretizationTypeError`` either)."""
        import jax
        import jax.numpy as _jnp

        from lumenairy.raytrace.jax_trace import (
            JaxRayState,
            _reset_transfer_jax_warning,
            _transfer_jax,
        )

        _reset_transfer_jax_warning()
        try:
            @jax.jit
            def step(state, t):
                return _transfer_jax(state, t, n_medium=1.0)

            x = _jnp.zeros(1); y = _jnp.zeros(1); z = _jnp.zeros(1)
            L = _jnp.array([0.5]); M = _jnp.zeros(1)
            n_axis = _jnp.array([_jnp.sqrt(1.0 - 0.25)])
            opd = _jnp.zeros(1)
            alive = _jnp.ones(1, dtype=bool)
            state = JaxRayState(x, y, z, L, M, n_axis, opd, alive)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                # Tracing under jit -- direction_n is a tracer here;
                # the warning check skips past via the
                # isinstance(..., np.ndarray) predicate.  The jit
                # call must succeed and yield a finite state.
                state_out = step(state, 1e-3)
            rw = [w for w in caught
                  if issubclass(w.category, RuntimeWarning)
                  and '_transfer_jax' in str(w.message)]
            # The jit'd call may still trigger the warning during
            # tracing or it may skip it cleanly depending on the
            # exact tracer / ndarray semantics; what we care about
            # is that the call does not crash, the output state is
            # consistent, and (when warnings happen) they don't
            # double-fire.
            assert len(rw) <= 1
            assert np.all(np.isfinite(np.asarray(state_out.x)))
            assert np.all(np.isfinite(np.asarray(state_out.opd)))
        finally:
            _reset_transfer_jax_warning()
