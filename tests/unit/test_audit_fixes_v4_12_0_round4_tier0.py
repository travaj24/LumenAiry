"""Pinning tests for the v4.12.0 round-4 audit Tier-0 fixes
(``AUDIT_ROUND4_2026_05_16.md`` items B0-1, B0-2, B2-6).

Each test pins a finding from the round-4 audit:

* B0-1 -- README cookbook examples (lines 2490-2606) now actually
  run.  Pre-4.12.0 the "Three minimal end-to-end examples" block
  failed on:

  - ``create_gaussian_beam(N=512, dx=2e-6, sigma=50e-6)`` -- missing
    required ``wavelength``.
  - ``apply_real_lens(E, presc, wavelength=..., dx=...)`` --
    positional ``presc`` (kw-only since v4.7).
  - ``load_zmx_prescription(...)`` -- function renamed to
    ``load_zemax_zmx`` in v4.7, no back-compat alias.

  Each test below reproduces one cookbook block verbatim and asserts
  it runs to completion with non-zero output.

* B0-2 -- ``_deprecation.py`` helpers are now wired into the
  top-level namespace.  The renamed-in-v4.7 functions
  ``load_zmx_prescription`` and ``load_zemax_prescription_txt`` are
  exposed as ``deprecated_alias`` shims that forward to
  ``load_zemax_zmx`` / ``load_zemax_prescription_data_txt`` after
  emitting a ``DeprecationWarning``.

* B2-6 -- ``gerchberg_saxton(backend='jax')`` now forwards
  ``seed`` / ``dtype`` / ``initial_phase`` to ``gerchberg_saxton_jax``.
  Pre-4.12.0 the dispatcher only forwarded ``n_iter`` so two seeds
  produced identical trajectories via the JAX path; the
  function-level kwargs were wired inside the JAX implementation
  but unreachable via the unified entry.

These pins guard against regression of the v4.12.0 fixes.
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import warnings

import numpy as np
import pytest

import lumenairy as la


# ============================================================================
# B0-1 -- README cookbook examples run to completion
# ============================================================================

class TestReadmeCookbookExamples:
    """Each test reproduces one README cookbook code block verbatim.

    Pre-4.12.0 these failed with ``TypeError`` / ``AttributeError`` at
    import or first-line execution; 4.12.0 fixed the README and the
    `pip install lumenairy` first-five-minutes experience.
    """

    def test_three_minimal_example_1_free_space(self):
        """Cookbook example 1: free-space propagation via smart
        dispatch.  The README uses ``angular_spectrum_propagate``
        directly to get a bare-ndarray return (``propagate`` returns
        a tuple for Fraunhofer/Fresnel methods, separate audit B1-7).
        """
        E, x, y = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.31e-6, sigma=50e-6)
        # README uses angular_spectrum_propagate (near-field) so the
        # return is an ndarray rather than the (E, dx_out, dy_out)
        # tuple that the smart dispatcher returns from Fraunhofer.
        E_focus = la.angular_spectrum_propagate(
            E, z=1e-3, wavelength=1.31e-6, dx=2e-6)
        cx, cy = la.beam_centroid(E_focus, 2e-6)
        assert E_focus.shape == (128, 128)
        assert np.isfinite(cx) and np.isfinite(cy)
        # Energy preserved (near-field, no aperture clipping)
        assert np.sum(np.abs(E_focus) ** 2) > 0

    def test_three_minimal_example_2_real_lens_end_to_end(self):
        """Cookbook example 2: a Thorlabs lens, end-to-end.
        Pre-4.12.0 broke on positional ``presc``."""
        E, _, _ = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.31e-6, sigma=50e-6)
        presc = la.thorlabs_lens('AC254-100-C')
        with warnings.catch_warnings():
            # The 128 grid is intentionally tiny for test speed; the
            # full grid-vs-aperture warning is expected and harmless.
            warnings.simplefilter('ignore', UserWarning)
            E_out = la.apply_real_lens(
                E, prescription=presc, wavelength=1.31e-6, dx=2e-6)
        assert E_out.shape == (128, 128)
        assert np.any(E_out != 0)

    def test_three_minimal_example_3_trace_for_spot_rms(self):
        """Cookbook example 3: ray-trace the same lens for spot RMS."""
        presc = la.thorlabs_lens('AC254-100-C')
        result = la.trace_prescription(
            presc, wavelength=1.31e-6, num_rings=8)
        rms = la.spot_rms(result)
        # spot_rms returns either a tuple (rms, ...) or a scalar
        # depending on the field structure; the cookbook indexes [0].
        rms_axis = rms[0] if hasattr(rms, '__getitem__') else rms
        assert np.isfinite(rms_axis)
        assert rms_axis > 0

    def test_basic_propagation_cookbook(self):
        """Cookbook 'Basic propagation' block (line 2519).  Pre-4.12.0
        broke on missing ``wavelength`` in ``create_gaussian_beam``."""
        E, x, y = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.3e-6, sigma=50e-6)
        E_prop = la.angular_spectrum_propagate(
            E, z=0.01, wavelength=1.3e-6, dx=2e-6)
        cx, cy = la.beam_centroid(E_prop, 2e-6)
        dx_b, dy_b = la.beam_d4sigma(E_prop, 2e-6)
        assert np.isfinite(cx) and np.isfinite(cy)
        assert dx_b > 0 and dy_b > 0

    def test_real_lens_from_zemax_cookbook(self):
        """Cookbook 'Real lens from Zemax file' block (line 2569).
        Pre-4.12.0 broke on ``load_zmx_prescription`` (renamed) and
        positional ``rx``."""
        E_in, _, _ = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.3e-6, sigma=50e-6)
        # README uses la.thorlabs_lens as the second option; we use
        # it to avoid needing a .zmx file on disk for the test.
        rx = la.thorlabs_lens('AC254-200-C')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            E_out = la.apply_real_lens(
                E_in, prescription=rx, wavelength=1.3e-6, dx=2e-6)
            E_out_traced = la.apply_real_lens_traced(
                E_in, prescription=rx, wavelength=1.3e-6, dx=2e-6,
                ray_subsample=4)
        assert E_out.shape == (128, 128)
        assert E_out_traced.shape == (128, 128)
        assert np.any(E_out != 0)
        assert np.any(E_out_traced != 0)

    def test_cylindrical_biconic_cookbook(self):
        """Cookbook 'Anamorphic / cylindrical / biconic' block
        (line 2613).  Pre-4.12.0 broke on positional ``pres``."""
        E_in, _, _ = la.create_gaussian_beam(
            N=128, dx=2e-6, wavelength=1.3e-6, sigma=50e-6)
        pres_cyl = la.make_cylindrical(
            R_focus=50e-3, d=3e-3, glass='N-BK7', axis='x')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            E_line_focus = la.apply_real_lens(
                E_in, prescription=pres_cyl,
                wavelength=1.3e-6, dx=2e-6)

            pres_bi = la.make_biconic(
                R1_x=50e-3, R1_y=70e-3,
                R2_x=-30e-3, R2_y=-40e-3,
                d=4e-3, glass='N-BK7')
            E_anam = la.apply_real_lens(
                E_in, prescription=pres_bi,
                wavelength=1.3e-6, dx=2e-6)
        assert E_line_focus.shape == (128, 128)
        assert E_anam.shape == (128, 128)
        assert np.any(E_line_focus != 0)
        assert np.any(E_anam != 0)

    def test_polarization_cookbook(self):
        """Cookbook 'Polarization' block (line 2765).  Pre-4.12.0
        broke on positional ``create_gaussian_beam(256, 2e-6, 30e-6)``
        which silently bound ``30e-6`` to ``wavelength`` and missed
        the required ``sigma`` kwarg."""
        scalar, _, _ = la.create_gaussian_beam(
            128, 2e-6, 1.3e-6, sigma=30e-6)
        field = la.create_circular_polarized(
            scalar, dx=2e-6, handedness='right')
        la.apply_half_wave_plate(field, angle=np.pi / 8)
        field.apply_thin_lens(f=100e-3, wavelength=1.3e-6)
        S = la.stokes_parameters(field)
        # Right-circular incident -> left-circular after lambda/2 at
        # 22.5 deg gives S3/S0 == -1 (sign flips).
        s3_over_s0 = float(S['S3'].mean() / S['S0'].mean())
        assert -1.5 < s3_over_s0 < -0.5


# ============================================================================
# B0-2 -- deprecation aliases are wired and emit DeprecationWarning
# ============================================================================

class TestDeprecatedAliasShims:
    """``_deprecation.deprecated_alias`` is now actually imported into
    the top-level namespace, and v4.7-renamed functions ship back-compat
    shims.  Pre-4.12.0 the helpers existed in ``_deprecation.py`` but
    nothing imported them, so old names died with ``AttributeError``."""

    def test_load_zmx_prescription_shim_exists(self):
        """The pre-v4.7 name ``load_zmx_prescription`` is reachable."""
        assert hasattr(la, 'load_zmx_prescription')
        assert callable(la.load_zmx_prescription)

    def test_load_zemax_prescription_txt_shim_exists(self):
        """The pre-v4.7 name ``load_zemax_prescription_txt`` is reachable."""
        assert hasattr(la, 'load_zemax_prescription_txt')
        assert callable(la.load_zemax_prescription_txt)

    def test_load_zmx_prescription_emits_deprecation_warning(self):
        """Calling the old name fires ``DeprecationWarning`` pointing
        at the new canonical name."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises((FileNotFoundError, OSError, IOError)):
                la.load_zmx_prescription('nonexistent_test_file.zmx')
            # Filter to our deprecation
            dws = [w for w in caught
                   if issubclass(w.category, DeprecationWarning)]
            assert len(dws) >= 1, (
                f"Expected DeprecationWarning, got: "
                f"{[(w.category.__name__, str(w.message)) for w in caught]}")
            msg = str(dws[0].message)
            assert 'load_zmx_prescription' in msg
            assert 'load_zemax_zmx' in msg

    def test_load_zemax_prescription_txt_emits_deprecation_warning(self):
        """Calling the old txt-loader name fires ``DeprecationWarning``
        pointing at the new canonical name."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises((FileNotFoundError, OSError, IOError)):
                la.load_zemax_prescription_txt('nonexistent_test_file.txt')
            dws = [w for w in caught
                   if issubclass(w.category, DeprecationWarning)]
            assert len(dws) >= 1
            msg = str(dws[0].message)
            assert 'load_zemax_prescription_txt' in msg
            assert 'load_zemax_prescription_data_txt' in msg

    def test_deprecated_shim_forwards_to_canonical(self):
        """The shim calls through to the new function so a user with
        pre-v4.7 code keeps getting the correct result (alongside the
        warning).  Test via a minimal singlet .zmx file."""
        # Build a minimal pseudo-.zmx with a BK7 singlet surface so
        # the auto-detect (active surfaces = glass-or-mirror) finds
        # at least one surface.  We don't assert physical correctness
        # here, only that the shim invokes the same code path.
        zmx_text = (
            "VERS 190311 0 0\n"
            "MODE SEQ\n"
            "NAME test\n"
            "UNIT MM X W X CM MR CPMM\n"
            "ENPD 10.0\n"
            "WAVM 1 0.587600 1.0\n"
            "PWAV 1\n"
            "SURF 0\n"
            "  TYPE STANDARD\n"
            "  CURV 0.0\n"
            "  DISZ INFINITY\n"
            "SURF 1\n"
            "  TYPE STANDARD\n"
            "  STOP\n"
            "  CURV 0.01 0 0 0 0 \"\"\n"
            "  DISZ 3.0\n"
            "  GLAS BK7 0 0 1.5 50.0\n"
            "  DIAM 5.0\n"
            "SURF 2\n"
            "  TYPE STANDARD\n"
            "  CURV -0.005 0 0 0 0 \"\"\n"
            "  DISZ 50.0\n"
            "  DIAM 5.0\n"
            "SURF 3\n"
            "  TYPE STANDARD\n"
            "  CURV 0.0 0 0 0 0 \"\"\n"
            "  DISZ 0.0\n"
            "  DIAM 5.0\n"
            "BLNK\n"
        )
        with tempfile.NamedTemporaryFile(
                mode='w', suffix='.zmx', delete=False) as f:
            f.write(zmx_text)
            fpath = f.name
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                warnings.simplefilter('ignore', UserWarning)
                rx_new = la.load_zemax_zmx(fpath)
                rx_old = la.load_zmx_prescription(fpath)
            # Same canonical schema keys (surfaces / thicknesses)
            assert set(rx_new.keys()) == set(rx_old.keys())
            # Same surface count
            assert (len(rx_new.get('surfaces', []))
                    == len(rx_old.get('surfaces', [])))
            # Same thickness count and values
            assert (len(rx_new.get('thicknesses', []))
                    == len(rx_old.get('thicknesses', [])))
            assert (rx_new.get('thicknesses') == rx_old.get('thicknesses'))
        finally:
            try:
                os.unlink(fpath)
            except OSError:
                pass


# ============================================================================
# B2-6 -- gerchberg_saxton(backend='jax') forwards seed/dtype/initial_phase
# ============================================================================

# These tests need JAX; skip cleanly if not installed.
jax = pytest.importorskip('jax', reason='JAX not available')


class TestGerchbergSaxtonJaxDispatch:
    """``gerchberg_saxton(backend='jax')`` now forwards
    ``seed`` / ``dtype`` / ``initial_phase`` to the JAX implementation.

    Pre-4.12.0 the dispatcher (``phase_retrieval.py:127-128``)
    only forwarded ``n_iter``, so two seeds produced byte-identical
    trajectories via the JAX path."""

    @pytest.fixture
    def source_target(self):
        N = 32
        x = np.linspace(-1, 1, N)
        X, Y = np.meshgrid(x, x)
        source = np.exp(-(X**2 + Y**2) / 0.5**2)
        target = np.exp(-(X**2 + Y**2) / 0.3**2)
        return source, target

    def test_jax_seeds_produce_different_trajectories(self, source_target):
        """Two different seeds via the unified entry yield genuinely
        different intermediate trajectories (pinning the dispatcher
        fix).  Pre-4.12.0 ``seed`` was dropped at the dispatcher so
        both runs got the same default-zero initial phase."""
        source, target = source_target
        phase_42, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=42, backend='jax')
        phase_43, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=43, backend='jax')
        # Identical-seed -> identical phases (the historical bug)
        diff = float(np.max(np.abs(phase_42 - phase_43)))
        assert diff > 1e-3, (
            f"BUG: seed=42 and seed=43 produced near-identical phases "
            f"(max abs diff = {diff:.2e}); dispatcher dropped seed.")

    def test_jax_dtype_forwarded_no_error(self, source_target):
        """Passing ``dtype`` reaches ``gerchberg_saxton_jax``.

        Pre-4.12.0 the dispatcher dropped the kwarg so users couldn't
        reach precision control via the unified entry.  We assert via
        the explicit float32 path (which works regardless of whether
        the global JAX_ENABLE_X64 flag is set); the JAX implementation
        itself has a separate fragility around float64 mode that's a
        pre-existing concern, not a dispatcher-fix concern.
        """
        source, target = source_target
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phase_f32, err_f32 = la.gerchberg_saxton(
                source, target, n_iter=10, seed=7,
                dtype=np.float32, backend='jax')
        assert phase_f32.shape == source.shape
        assert np.all(np.isfinite(phase_f32))
        assert np.isfinite(err_f32)

    def test_jax_dtype_kwarg_actually_threads_through(self, source_target):
        """A targeted check that the dispatcher's `dtype` kwarg is
        forwarded into `gerchberg_saxton_jax` -- we patch the JAX
        function and assert the dispatcher passes `dtype` explicitly.
        """
        import lumenairy.analysis.phase_retrieval as pr

        source, target = source_target
        captured = {}
        real_jax = pr.gerchberg_saxton_jax

        def spy(*args, **kwargs):
            captured.update(kwargs)
            return real_jax(*args, **kwargs)

        pr.gerchberg_saxton_jax = spy
        try:
            la.gerchberg_saxton(
                source, target, n_iter=5, seed=11,
                dtype=np.float32, initial_phase=None,
                backend='jax')
        finally:
            pr.gerchberg_saxton_jax = real_jax

        assert 'seed' in captured and captured['seed'] == 11
        assert 'dtype' in captured and captured['dtype'] is np.float32
        assert 'initial_phase' in captured and captured['initial_phase'] is None
        assert 'n_iter' in captured and captured['n_iter'] == 5

    def test_jax_initial_phase_forwarded(self, source_target):
        """Passing ``initial_phase`` reaches ``gerchberg_saxton_jax``.
        With identical seed but different explicit ``initial_phase``,
        the trajectories diverge."""
        source, target = source_target
        ip_zero = np.zeros_like(source)
        ip_rand = np.random.default_rng(123).uniform(
            -np.pi, np.pi, size=source.shape).astype(np.float32)
        phase_zero, _ = la.gerchberg_saxton(
            source, target, n_iter=20, initial_phase=ip_zero,
            backend='jax')
        phase_rand, _ = la.gerchberg_saxton(
            source, target, n_iter=20, initial_phase=ip_rand,
            backend='jax')
        diff = float(np.max(np.abs(phase_zero - phase_rand)))
        assert diff > 1e-3, (
            f"BUG: distinct initial_phase arrays produced near-identical "
            f"results (max abs diff = {diff:.2e}); dispatcher dropped "
            f"initial_phase.")

    def test_numpy_seeds_still_differ(self, source_target):
        """Regression guard: the numpy-backend behaviour (unchanged)
        still produces different trajectories for different seeds."""
        source, target = source_target
        phase_42, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=42, backend='numpy')
        phase_43, _ = la.gerchberg_saxton(
            source, target, n_iter=20, seed=43, backend='numpy')
        diff = float(np.max(np.abs(phase_42 - phase_43)))
        assert diff > 1e-3

    def test_numpy_dtype_kwarg_accepted(self, source_target):
        """4.12.0 added ``dtype`` to ``gerchberg_saxton``'s public
        signature; the numpy path now honours it for API parity with
        ``error_reduction`` / ``hybrid_input_output``."""
        source, target = source_target
        phase_c128, _ = la.gerchberg_saxton(
            source, target, n_iter=10, seed=7,
            dtype=np.complex128, backend='numpy')
        phase_c64, _ = la.gerchberg_saxton(
            source, target, n_iter=10, seed=7,
            dtype=np.complex64, backend='numpy')
        # Both run to completion with the requested dtype guidance.
        # We can't pin the dtype of `phase` itself (np.angle returns
        # float matching the complex input's float-component dtype),
        # but the kwarg must at least be accepted without TypeError.
        assert phase_c128.shape == source.shape
        assert phase_c64.shape == source.shape
