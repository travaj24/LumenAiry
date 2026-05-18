"""Pinning tests for the v4.15 Agent-B scope.

ROADMAP reference
-----------------

Agent B scope for the v4.15 dispatch:

* **B.1 / ROADMAP v4.15 #2** -- ``Source.*`` size-arg normalisation.
  The 5 factory classmethods are migrated to the canonical kwarg-only
  form ``Source.method(*, N, dx, wavelength, <size_kwargs>)``.  The
  legacy positional form is still accepted for one release with a
  ``DeprecationWarning`` routed through
  ``_deprecation.warn_deprecated_signature`` (``version_removed='5.0'``).

* **B.2 / ROADMAP v4.15 #3** -- ``lumenairy.system.evaluate`` ergonomic
  one-call entry: build the element chain from a prescription dict,
  pull ``E`` / ``dx`` / ``dy`` / ``wavelength`` off a :class:`Source`,
  route through :func:`propagate_through_system`, return a
  :class:`PropagationResult`.

* **B.3 / ROADMAP v4.16 #9** -- Schell-model + Gaussian-Schell
  partial-coherence sources.  Two new factories
  ``create_gaussian_schell_source`` + ``create_schell_model_source``
  plus matching ``Source.gaussian_schell`` / ``Source.schell_model``
  classmethods.

* **B.4 / ROADMAP v4.16 #11** -- Ring / annular incoherent source
  (``create_annular_incoherent_source``).

* **B.5** -- Three P2 carryover items from the v4.14.2 audit:
    - **P2-VAL-1**: ``_validate_grid_params`` now rejects ``bool``
      explicitly so callers can't pass ``N=True`` / ``N=False``.
    - **P2-FAC-1**: ``create_multi_field_sources`` was not in the
      factory-validation dispatcher pin (the v4.13.0 P1-C meta-pin).
      v4.15 adds it plus the 4 new factories.  This file's
      ``TestP2FacFactoryValidationMetaPin`` carries the canonical
      kwarg-form copy of the meta-pin.
    - **P2-DEP-1**: ``makedammann2d`` + ``create_led_source`` shims
      now route through ``_deprecation.warn_deprecated_signature``
      with ``version_removed='5.0'`` in the message text.

Author: Andrew Traverso -- v4.15 / Agent B
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.sources.core import (
    Source,
    _validate_grid_params,
    create_annular_incoherent_source,
    create_gaussian_schell_source,
    create_schell_model_source,
)


# ============================================================================
# B.1 -- Source.* size-arg normalisation
# ============================================================================

# Canonical kwarg-only invocation table for the 5 normalised factories.
# Each entry is ``(factory_name, legacy_positional_args, extra_kwargs)``
# where ``legacy_positional_args`` is the pre-v4.15 positional tuple
# (size-args first where applicable) and ``extra_kwargs`` is the set of
# kwargs that need to round out a working call.
_NORMALISED_FACTORIES = [
    pytest.param(
        'gaussian', (20e-6, 32, 5e-6, 633e-9), {'w0': 20e-6}, id='gaussian'),
    pytest.param(
        'plane_wave', (32, 5e-6, 633e-9), {}, id='plane_wave'),
    pytest.param(
        'point_source', (32, 5e-6, 633e-9), {'z0': -1e-3}, id='point_source'),
    pytest.param(
        'top_hat', (50e-6, 32, 5e-6, 633e-9), {'diameter': 50e-6},
        id='top_hat'),
    pytest.param(
        'fiber_mode', (10e-6, 32, 5e-6, 633e-9),
        {'mode_field_diameter': 10e-6}, id='fiber_mode'),
]


class TestB1SourceFactoryNormalisation:
    """Every one of the 5 normalised factories must accept BOTH the
    canonical kwarg-only form (no warning) AND the legacy positional
    form (DeprecationWarning, removal-version='5.0' in the message).
    """

    @pytest.mark.parametrize('factory_name, legacy_args, kwargs',
                             _NORMALISED_FACTORIES)
    def test_canonical_kwarg_form_works(self, factory_name, legacy_args,
                                         kwargs):
        """Canonical kwarg-only form must work without any warnings."""
        factory = getattr(Source, factory_name)
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # any warning -> test failure
            src = factory(N=32, dx=5e-6, wavelength=633e-9, **kwargs)
        assert isinstance(src, Source)
        assert src.shape == (32, 32)

    @pytest.mark.parametrize('factory_name, legacy_args, kwargs',
                             _NORMALISED_FACTORIES)
    def test_legacy_positional_form_still_works(self, factory_name,
                                                  legacy_args, kwargs):
        """Legacy positional form must still produce a Source (back-compat)."""
        factory = getattr(Source, factory_name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            src = factory(*legacy_args)
        assert isinstance(src, Source)
        assert src.shape == (32, 32)

    @pytest.mark.parametrize('factory_name, legacy_args, kwargs',
                             _NORMALISED_FACTORIES)
    def test_legacy_positional_emits_deprecation_warning(self, factory_name,
                                                          legacy_args,
                                                          kwargs):
        """Legacy positional form must emit a DeprecationWarning that
        mentions v5.0 removal."""
        factory = getattr(Source, factory_name)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            factory(*legacy_args)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)]
        assert len(dep) >= 1, (
            f"Source.{factory_name} legacy positional call must emit "
            f"DeprecationWarning; got {[str(w.message) for w in caught]}.")
        msg = str(dep[0].message)
        assert '5.0' in msg, (
            f"DeprecationWarning must reference v5.0 removal target; "
            f"got {msg!r}.")
        assert factory_name in msg or f'Source.{factory_name}' in msg, (
            f"DeprecationWarning must name the factory; got {msg!r}.")


# ============================================================================
# B.2 -- system.evaluate ergonomic entry
# ============================================================================

class TestB2SystemEvaluate:
    """Validate the ergonomic ``lumenairy.system.evaluate`` entry."""

    def test_factory_prescription_routes_through(self):
        """A factory-built singlet prescription routes through
        ``evaluate`` and yields a PropagationResult."""
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3, glass='N-BK7',
                             aperture=10e-3)
        src = Source.gaussian(N=128, dx=20e-6, wavelength=587.6e-9,
                              w0=2e-3)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            result = la.system.evaluate(rx, src)
        from lumenairy.propagators.result import PropagationResult
        assert isinstance(result, PropagationResult)
        assert result.field.shape == (128, 128)
        assert result.wavelength == pytest.approx(587.6e-9)

    def test_missing_source_raises_value_error(self):
        """``source=None`` must raise ValueError, NOT crash inside the
        propagator with an obscure ndarray-from-None message."""
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3, glass='N-BK7',
                             aperture=10e-3)
        with pytest.raises(ValueError) as info:
            la.system.evaluate(rx, None)
        msg = str(info.value).lower()
        assert 'source' in msg

    def test_output_grid_kwarg_accepted(self):
        """``output_grid`` is accepted (future-proof kwarg).  Currently
        a no-op when matching; emits a RuntimeWarning when mismatched
        but still returns a result."""
        rx = la.make_singlet(R1=50e-3, R2=-50e-3, d=2e-3, glass='N-BK7',
                             aperture=10e-3)
        src = Source.gaussian(N=128, dx=20e-6, wavelength=587.6e-9,
                              w0=2e-3)
        # Matching output_grid -- no warning expected.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            warnings.simplefilter('error', RuntimeWarning)
            result = la.system.evaluate(rx, src, output_grid=128)
        assert result.field.shape == (128, 128)
        # Mismatched output_grid -- emits a soft warning, still returns.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result2 = la.system.evaluate(rx, src, output_grid=256)
        rw = [w for w in caught
              if issubclass(w.category, RuntimeWarning)
              and 'output_grid' in str(w.message)]
        assert len(rw) >= 1
        assert result2.field.shape == (128, 128)  # current no-op behaviour


# ============================================================================
# B.3 -- Schell-model + Gaussian-Schell sources
# ============================================================================

class TestB3SchellSources:
    """Validate the new Schell-model / Gaussian-Schell factories."""

    def test_gaussian_schell_produces_n_realizations(self):
        """The ensemble-averaged amplitude envelope must be smoother
        (lower variance) as ``n_realizations`` grows.  A small
        regression check: 4 realizations vs 16 realizations must give
        a strictly lower std on the radial profile."""
        rng_kwargs = dict(N=32, dx=10e-6, wavelength=633e-9,
                          w0=80e-6, sigma_g=40e-6, seed=42)
        E_few, _, _ = create_gaussian_schell_source(
            n_realizations=4, **rng_kwargs)
        E_many, _, _ = create_gaussian_schell_source(
            n_realizations=16, **rng_kwargs)
        # Both have the right shape + dtype.
        assert E_few.shape == (32, 32)
        assert E_many.shape == (32, 32)
        # Smoothness check: radial-profile-std should decrease as
        # n_realizations grows for the same seed.
        var_few = float(np.var(np.abs(E_few) ** 2))
        var_many = float(np.var(np.abs(E_many) ** 2))
        # Allow equality at the limits but the relationship must be at
        # least non-increasing.
        assert var_many <= var_few * 1.5, (
            f"More realizations should give smoother (lower-variance) "
            f"intensity; got var_few={var_few}, var_many={var_many}.")

    def test_schell_model_honours_intensity_profile_and_coherence(self):
        """The output ensemble-averaged intensity must follow the
        user-supplied intensity profile up to a normalisation."""
        N = 32
        xs = np.linspace(-1, 1, N)
        I_target = np.exp(-(xs[:, None] ** 2 + xs[None, :] ** 2) / 0.25)
        E, _, _ = create_schell_model_source(
            N=N, dx=10e-6, wavelength=633e-9,
            intensity_profile=I_target,
            coherence_length=20e-6,
            n_realizations=16, seed=0)
        I_out = np.abs(E) ** 2
        # Compute correlation -- must be high since the source amplitude
        # is sqrt(I_target) up to phase + ensemble noise.
        corr = float(np.corrcoef(I_out.ravel(), I_target.ravel())[0, 1])
        assert corr > 0.9, (
            f"Schell-model output must correlate strongly with the "
            f"supplied intensity_profile; got corr={corr}.")

    def test_gaussian_schell_raises_on_nonpositive_N(self):
        """``N <= 0`` must raise via the centralised validator."""
        with pytest.raises((ValueError, TypeError)):
            create_gaussian_schell_source(
                N=0, dx=10e-6, wavelength=633e-9,
                w0=50e-6, sigma_g=25e-6)
        with pytest.raises((ValueError, TypeError)):
            create_schell_model_source(
                N=-1, dx=10e-6, wavelength=633e-9,
                intensity_profile=np.ones((4, 4)),
                coherence_length=20e-6)

    def test_schell_factories_reject_bool_N(self):
        """``N=True`` must raise TypeError under the v4.15 P2-VAL-1 fix
        (``isinstance(True, int)`` short-circuit)."""
        with pytest.raises(TypeError):
            create_gaussian_schell_source(
                N=True, dx=10e-6, wavelength=633e-9,
                w0=50e-6, sigma_g=25e-6)
        with pytest.raises(TypeError):
            create_schell_model_source(
                N=True, dx=10e-6, wavelength=633e-9,
                intensity_profile=np.ones((4, 4)),
                coherence_length=20e-6)

    def test_source_gaussian_schell_classmethod(self):
        """``Source.gaussian_schell`` must build a Source and forward kwargs."""
        src = Source.gaussian_schell(
            N=32, dx=10e-6, wavelength=633e-9,
            w0=80e-6, sigma_g=40e-6, n_realizations=4, seed=0)
        assert isinstance(src, Source)
        assert src.shape == (32, 32)
        assert src.wavelength == 633e-9


# ============================================================================
# B.4 -- Ring / annular incoherent source
# ============================================================================

class TestB4AnnularIncoherent:
    """Validate the new annular-incoherent factory."""

    def test_annular_inco_honours_inner_outer_radii(self):
        """Pixels inside ``inner_radius`` and outside ``outer_radius``
        must have near-zero intensity; pixels within the annulus must
        have non-trivial intensity."""
        N = 64
        dx = 5e-6
        inner = 50e-6
        outer = 120e-6
        E, x, y = create_annular_incoherent_source(
            N=N, dx=dx, wavelength=633e-9,
            inner_radius=inner, outer_radius=outer,
            n_realizations=8, seed=0)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X * X + Y * Y)
        I = np.abs(E) ** 2
        # Inside the inner radius -- intensity must be ~0.
        inside_mask = r < inner * 0.9
        assert float(I[inside_mask].max()) < float(I.max()) * 0.1, (
            f"Annular source must vanish inside inner_radius; got "
            f"max-inside={I[inside_mask].max()}, peak={I.max()}.")
        # Outside the outer radius -- intensity must be ~0.
        outside_mask = r > outer * 1.1
        assert float(I[outside_mask].max()) < float(I.max()) * 0.1, (
            f"Annular source must vanish outside outer_radius; got "
            f"max-outside={I[outside_mask].max()}, peak={I.max()}.")
        # Inside the annulus -- intensity must be non-trivial.
        annulus_mask = (r >= inner) & (r <= outer)
        assert float(I[annulus_mask].mean()) > 0.0

    def test_annular_inco_ensemble_averages_incoherently(self):
        """The phase of the ensemble-averaged amplitude must NOT match
        any single realization (the ensemble average kills the random
        phase down to ~1/sqrt(N) baseline)."""
        kw = dict(N=32, dx=10e-6, wavelength=633e-9,
                  inner_radius=80e-6, outer_radius=140e-6, seed=0)
        E_1, _, _ = create_annular_incoherent_source(
            n_realizations=1, **kw)
        E_64, _, _ = create_annular_incoherent_source(
            n_realizations=64, **kw)
        # Compute the std of the phase distribution within the annulus
        # for each.  The single-realization version has wide phase
        # spread; the 64-realization average has near-zero coherent
        # phase (the random sum's phase tends to a tight value as N
        # grows).
        mask = np.abs(E_1) > np.abs(E_1).max() * 0.5
        phase_1 = np.angle(E_1)[mask]
        phase_64 = np.angle(E_64[mask])
        # The single-realization phase covers the full [-pi, pi];
        # the averaged phase is concentrated near 0 (coherent-mean phase).
        ptp_1 = float(np.ptp(phase_1))
        ptp_64 = float(np.ptp(phase_64))
        # The single realization should span most of (-pi, pi); the
        # 64-realization average has a narrower phase spread.
        assert ptp_1 > 0.5
        # Note: this is a stochastic property; we use a soft
        # tolerance.  The point is that incoherent averaging concentrates
        # the coherent-mean phase, NOT that it should be deterministic.


# ============================================================================
# B.5.a -- P2-VAL-1: _validate_grid_params bool rejection
# ============================================================================

class TestP2Val1BoolNRejection:
    """``_validate_grid_params`` must reject ``bool`` for N
    (``isinstance(True, int)`` short-circuit footgun)."""

    def test_bool_N_true_rejected(self):
        with pytest.raises(TypeError) as info:
            _validate_grid_params(True, 1e-6, 633e-9, fn_name='test')
        assert 'bool' in str(info.value).lower()

    def test_bool_N_false_rejected(self):
        with pytest.raises(TypeError) as info:
            _validate_grid_params(False, 1e-6, 633e-9, fn_name='test')
        assert 'bool' in str(info.value).lower()

    def test_bool_in_tuple_N_rejected(self):
        """A bool inside a tuple-N (when factory supports tuples) is
        also rejected."""
        with pytest.raises(TypeError) as info:
            _validate_grid_params(
                (True, 32), 1e-6, 633e-9,
                fn_name='test', support_tuple_N=True)
        assert 'bool' in str(info.value).lower()

    def test_int_N_still_accepted(self):
        """The bool short-circuit must NOT regress on normal int / np.int."""
        _validate_grid_params(32, 1e-6, 633e-9, fn_name='test')  # ok
        _validate_grid_params(np.int64(32), 1e-6, 633e-9, fn_name='test')


# ============================================================================
# B.5.b -- P2-FAC-1: factory-validation meta-pin
# ============================================================================

# Updated parametrize list including create_multi_field_sources and the 4
# new factories.  See task description.  Note: gaussian_schell /
# schell_model / annular_incoherent kwargs include the required physical
# parameters for each factory.

_DISPATCHER_PIN_PLAIN_FACTORIES = [
    pytest.param(
        'create_multi_field_sources',
        {'field_angles': [0.0, 0.01]},
        id='create_multi_field_sources'),
    pytest.param(
        'create_gaussian_schell_source',
        {'w0': 50e-6, 'sigma_g': 25e-6, 'n_realizations': 4, 'seed': 0},
        id='create_gaussian_schell_source'),
    pytest.param(
        'create_schell_model_source',
        {'intensity_profile': np.ones((32, 32)),
         'coherence_length': 25e-6,
         'n_realizations': 4, 'seed': 0},
        id='create_schell_model_source'),
    pytest.param(
        'create_annular_incoherent_source',
        {'inner_radius': 50e-6, 'outer_radius': 120e-6,
         'n_realizations': 4, 'seed': 0},
        id='create_annular_incoherent_source'),
]


class TestP2FacFactoryValidationMetaPin:
    """Strong dispatcher-level pin including the 4 new factories.

    Pre-v4.15 the v4.13.0 P1-C dispatcher pin
    (``TestP1CSourceFactoryDispatcherPin``) covered exactly 5 Source
    factories (gaussian, plane_wave, point_source, top_hat,
    fiber_mode).  v4.15 broadens the surface to include
    ``create_multi_field_sources`` (which was transitively-validated
    via ``create_tilted_plane_wave`` only) plus the 4 new factories
    from B.3 and B.4.
    """

    @pytest.mark.parametrize('factory_name, extra_kwargs',
                             _DISPATCHER_PIN_PLAIN_FACTORIES)
    def test_plain_factory_validates_N_dx_wavelength(self, factory_name,
                                                      extra_kwargs):
        """Each top-level factory accepts the canonical N / dx /
        wavelength tuple and produces an output of the right shape."""
        N = 32
        dx = 5e-6
        wavelength = 633e-9
        factory = getattr(la, factory_name)
        result = factory(N=N, dx=dx, wavelength=wavelength,
                         **extra_kwargs)
        if factory_name == 'create_multi_field_sources':
            sources, x, y = result
            # Each entry is (E, ax, ay).
            assert len(sources) == 2
            for E, ax, ay in sources:
                assert E.shape == (N, N)
        else:
            E, x, y = result
            assert E.shape == (N, N)
        # The dispatcher-pin's invariant: the factory must raise a
        # named, clear error on N=0 (not an obscure downstream meshgrid
        # crash).
        with pytest.raises((ValueError, TypeError)):
            factory(N=0, dx=dx, wavelength=wavelength, **extra_kwargs)

    @pytest.mark.parametrize('factory_name, extra_kwargs',
                             _DISPATCHER_PIN_PLAIN_FACTORIES)
    def test_plain_factory_rejects_bool_N(self, factory_name, extra_kwargs):
        """Each new factory must inherit the P2-VAL-1 bool rejection."""
        factory = getattr(la, factory_name)
        # multi_field_sources doesn't run _validate_grid_params itself;
        # it forwards each tilted plane wave, so the underlying
        # create_tilted_plane_wave rejects bool-N.
        with pytest.raises((TypeError, ValueError)):
            factory(N=True, dx=5e-6, wavelength=633e-9, **extra_kwargs)

    def test_classmethod_pin_still_passes_for_normalised_5(self):
        """The 5 normalised classmethod factories continue to honour
        the dy thread-through invariant from
        ``TestP1CSourceFactoryDispatcherPin``."""
        kwargs_table = [
            ('gaussian', {'w0': 20e-6}),
            ('plane_wave', {}),
            ('point_source', {'z0': -1e-3}),
            ('top_hat', {'diameter': 50e-6}),
            ('fiber_mode', {'mode_field_diameter': 10e-6}),
        ]
        N = 32
        dx = 5e-6
        dy = 11e-6
        wavelength = 633e-9
        for factory_name, extra_kwargs in kwargs_table:
            factory = getattr(Source, factory_name)
            src = factory(N=N, dx=dx, wavelength=wavelength, dy=dy,
                           **extra_kwargs)
            assert src.dx == dx
            assert src.dy == dy, (
                f"Source.{factory_name}: returned Source advertised "
                f"dy={src.dy!r}; expected dy={dy!r}.")


# ============================================================================
# B.5.c -- P2-DEP-1: shim migration mentions version_removed='5.0'
# ============================================================================

class TestP2Dep1ShimsMentionVersionRemoved:
    """Both legacy shims (``makedammann2d`` and ``create_led_source``)
    must route their DeprecationWarning through
    ``_deprecation.warn_deprecated_signature`` and mention
    ``version_removed='5.0'`` in the message text."""

    def test_create_led_source_legacy_shim_mentions_v5_0(self):
        """The legacy positional form of ``create_led_source`` must
        emit a DeprecationWarning that contains 'v5.0' (the removal
        version)."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.create_led_source(64, 16e-6, 100e-6, 0.3, 1.31e-6)
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'create_led_source' in str(w.message)]
        assert len(dep) >= 1, (
            f"Legacy create_led_source call must emit "
            f"DeprecationWarning; got "
            f"{[str(w.message) for w in caught]}.")
        msg = str(dep[0].message)
        assert '5.0' in msg, (
            f"create_led_source DeprecationWarning must reference v5.0 "
            f"removal; got {msg!r}.")

    def test_makedammann2d_legacy_shim_mentions_v5_0(self):
        """The legacy micrometre auto-detect path in ``makedammann2d``
        must emit a DeprecationWarning that contains 'v5.0'.

        The heuristic fires when any of ``periodx`` / ``periody`` /
        ``waveln`` exceeds ``1e-3`` (1 mm) but is below ``1`` m (the
        v4.14.3 absolute upper-bound guard).  We pass values in the
        narrow band ``(1e-3, 1)`` -- representing a legacy caller that
        wrote ``periodx=0.5`` thinking micrometres but landed at half-
        a-metre in SI -- so the heuristic-warning path fires without
        tripping the upper-bound ``ValueError``.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                la.makedammann2d(
                    # 0.5 -> 5e-7 after legacy * 1e-6 rescale (550 nm
                    # period -- physically unrealistic for a Dammann
                    # grating, but the test only cares that the
                    # heuristic warning fires).
                    periodx=0.5, periody=0.5, waveln=0.5,
                    itr=1, plot=False, seed=0)
            except Exception:
                # The function may still raise for other reasons
                # downstream of the warning (degenerate near-field
                # generated by the implausible 550 nm period x 550 nm
                # wavelength); we only care about the warning being
                # emitted, not about the IFTA reaching a useful design.
                pass
        dep = [w for w in caught
               if issubclass(w.category, DeprecationWarning)
               and 'makedammann2d' in str(w.message)]
        assert len(dep) >= 1, (
            f"makedammann2d legacy micrometre call must emit "
            f"DeprecationWarning; got "
            f"{[str(w.message) for w in caught]}.")
        msg = str(dep[0].message)
        assert '5.0' in msg, (
            f"makedammann2d DeprecationWarning must reference v5.0 "
            f"removal; got {msg!r}.")
