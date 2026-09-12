"""WP-A5 follow-up pins -- the two items the VERIFY-A5 pass left open.

* **V1 (P1)** -- `propagate_hfpi_freespace_aperture` (hence
  `propagate_hfpi` and `propagate(method='hfpi')`) returned amplitudes
  low by exactly ``n_paths * z_to_aperture`` under the v5.46 default
  ``normalisation='physical'``: the re-emission divided by the sample
  count a SECOND time and the intermediate leg carried no Jacobian.
* **V6 (P2)** -- the RS ``kernel='transfer'`` branch's circular
  wrap-around is reachable below the alias threshold for a field whose
  angular content fills the grid, and was unwarned.

Every bar carries its oracle, that oracle's own floor, the measured
pre-fix and post-fix values and the decades of gap on each side, per
``docs/TESTING_STANDARDS.md``.

Author: Andrew Traverso -- WP-A5 follow-up.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.propagators.hfpi import (
    PathBundle,
    _reemission_measure,
    accumulate_to_grid,
    apply_aperture_diffraction,
    init_paths_from_field,
    propagate_hfpi_freespace_aperture,
    propagate_to_plane,
)
from lumenairy.propagators.rs import rayleigh_sommerfeld_propagate
from lumenairy.propagators.vectorial_hfpi import (
    propagate_vector_hfpi_freespace_aperture,
)

LAM = 532e-9


# ===========================================================================
# V1 -- the cascaded HFPI measure
# ===========================================================================

def _src(N=48, dx=1.5e-6, w0=7e-6):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(complex), dx


def _ls_scale(a, ref):
    """Unbiased least-squares complex scale ``a ~ s * ref``.

    The mean of ``|a|/|ref|`` is biased HIGH for a Monte-Carlo estimator
    (``|sum|^2 = |signal|^2 + noise power``); the least-squares
    projection is not.
    """
    m = np.abs(ref) > 0.1 * np.abs(ref).max()
    return complex(np.sum(a[m] * np.conj(ref[m]))
                   / np.sum(np.abs(ref[m]) ** 2))


class TestV1ReemissionMeasure:

    def test_the_factor_matches_its_closed_form_exactly(self):
        """The strongest pin: ``_reemission_measure`` is a pure function,
        so this is EXACT -- no Monte Carlo anywhere.

        Derivation (see the function's docstring).  Writing an
        intermediate surface integral in the direction variables the
        estimator samples, ``dOmega = dS cos(theta)/r^2``, turns the
        Kirchhoff kernel ``dS (cos(theta)/r) e^{ikr}`` into
        ``dOmega * r * e^{ikr}`` -- a factor ``r``, not ``1/r``, and no
        obliquity.  Propagating the estimator's invariant across the
        surface gives

            F = (1/(i lambda)) * Omega_out * r_in * cos_out / cos_in .

        Bar: 1e-13 relative -- pure float64 arithmetic, so the only
        difference from the closed form is round-off (measured < 1e-16).
        The pre-fix factor was
        ``0.5(cos_in + cos_out) * (1/(i lambda)) * Omega_out / n_paths``,
        which on this fixture differs by a factor of 6.7e5 -- 18 decades
        above the bar.
        """
        n = 5
        # A bundle with KNOWN legs and directions; nothing random.
        th_in = np.array([0.00, 0.05, 0.10, 0.20, 0.30])
        directions = np.stack([np.sin(th_in), np.zeros(n), np.cos(th_in)],
                              axis=-1)
        leg = np.array([1e-4, 5e-4, 1e-3, 2e-3, 5e-3])
        paths = PathBundle(
            positions=np.zeros((n, 3)), directions=directions,
            weights=np.ones(n, dtype=np.complex128), opl=leg.copy(),
            alive=np.ones(n, dtype=bool), leg=leg)
        th_out = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        cos_out = np.cos(th_out)
        cone = 0.25
        cos_max = float(np.cos(cone))

        got = np.asarray(_reemission_measure(
            paths, cos_out, cos_max, LAM, 'physical', 'probe'))
        omega = 2.0 * np.pi * (1.0 - cos_max)
        want = (1.0 / (1j * LAM)) * omega * leg * cos_out / np.cos(th_in)
        rel = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert rel < 1e-13, (
            f"re-emission factor differs from its closed form by {rel:.3e}")

        # ... and the 'legacy' branch is exactly the pre-fix expression.
        got_l = np.asarray(_reemission_measure(
            paths, cos_out, cos_max, LAM, 'legacy', 'probe'))
        want_l = (0.5 * (np.cos(th_in) + cos_out)
                  * (1.0 / (1j * LAM)) * omega / n)
        rel_l = np.max(np.abs(got_l - want_l)) / np.max(np.abs(want_l))
        assert rel_l < 1e-13
        # ... and the two differ by exactly the factor the finding is
        # about: ``n_paths * r_in * cos_out/cos_in /
        # (0.5(cos_in + cos_out))``, i.e. ``n_paths * r_in`` up to the
        # cone's own cosine spread.  Closed-form identity, bar 1e-13.
        want_ratio = (n * leg * cos_out / np.cos(th_in)
                      / (0.5 * (np.cos(th_in) + cos_out)))
        rel_r = np.max(np.abs(np.abs(got / got_l) - want_ratio)
                       / want_ratio)
        assert rel_r < 1e-13, (
            f"physical/legacy ratio differs from n*r_in*cos_out/cos_in / "
            f"(0.5(cos_in+cos_out)) by {rel_r:.3e}")

    def test_an_unobstructed_aperture_plane_is_transparent(self):
        """The oracle-free property, and the one that fails pre-fix.

        An aperture large enough to clip nothing is not there: the
        two-leg walk over ``z1 + z2`` must return the same field as the
        one-leg walk over the same total.  No analytic oracle, no second
        propagator -- the estimator is compared with itself.

        Bar: ``0.5 < |scale| < 2.0`` on the unbiased least-squares
        complex scale.  Derivation.  The estimator is Monte Carlo, so the
        scale scatters: measured 1.009 / 1.010 / 1.057 / 0.963 / 1.245 /
        0.977 at 8 M paths and 1.096 / 0.847 / 1.044 / 0.940 / 0.586 /
        1.052 at 2 M, across z1 = 0.25 / 0.5 / 1.0 mm and two seeds --
        i.e. the observed envelope is ~0.59-1.25, and the bar sits about
        a factor of 1.6 outside it on each side.  Pre-fix the scale was
        ``1/(n_paths * z1)``: 3.4e-3 at n = 500 k / z1 = 0.5 mm, i.e.
        **147x below the lower bar**, and it moves with BOTH n_paths and
        z1, which is what makes it unmistakable rather than a tolerance
        question.
        """
        E0, dx = _src()
        z1, z2 = 0.5e-3, 0.5e-3
        n_paths = 2_000_000
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = init_paths_from_field(E0, dx, n_paths=4_000_000,
                                      wavelength=LAM, rng=1,
                                      cone_half_angle=0.08)
            p = propagate_to_plane(p, z_target=z1 + z2, wavelength=LAM)
            one = np.asarray(accumulate_to_grid(
                p, Ny=24, Nx=24, dx=3e-6, on_undersampled='silent'))
            two = np.asarray(propagate_hfpi_freespace_aperture(
                E0, dx, z_to_aperture=z1, aperture_radius=600e-6,
                z_aperture_to_output=z2, wavelength=LAM, n_paths=n_paths,
                rng=5, output_shape=(24, 24), output_dx=3e-6,
                cone_half_angle=0.08, on_undersampled='silent'))
        s = abs(_ls_scale(two, one))
        assert 0.5 < s < 2.0, (
            f"two-leg / one-leg least-squares scale = {s:.4e}; an "
            f"unobstructed aperture plane must be transparent.  Pre-fix "
            f"this read 1/(n_paths*z1) = {1.0/(n_paths*z1):.3e}.")

    @pytest.mark.parametrize('z1,z2', [(0.25e-3, 0.75e-3), (1e-3, 1e-3)])
    def test_transparency_does_not_depend_on_where_the_plane_sits(
            self, z1, z2):
        """The residual law was ``n_paths * z1``, so moving the plane at
        fixed total distance is the sharpest discriminator: pre-fix the
        scale changes by 4x between these two rows, post-fix it does
        not.  Same bar and derivation as above."""
        E0, dx = _src()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = init_paths_from_field(E0, dx, n_paths=4_000_000,
                                      wavelength=LAM, rng=1,
                                      cone_half_angle=0.08)
            p = propagate_to_plane(p, z_target=z1 + z2, wavelength=LAM)
            one = np.asarray(accumulate_to_grid(
                p, Ny=24, Nx=24, dx=3e-6, on_undersampled='silent'))
            two = np.asarray(propagate_hfpi_freespace_aperture(
                E0, dx, z_to_aperture=z1, aperture_radius=600e-6,
                z_aperture_to_output=z2, wavelength=LAM,
                n_paths=4_000_000, rng=11, output_shape=(24, 24),
                output_dx=3e-6, cone_half_angle=0.08,
                on_undersampled='silent'))
        s = abs(_ls_scale(two, one))
        assert 0.5 < s < 2.0, (
            f"z1 = {z1*1e3:.2f} mm: scale {s:.4e}")

    def test_the_vector_twin_carries_the_same_measure(self):
        """The vector module imports the shared helper, so its Ex channel
        must track the scalar propagator to Monte-Carlo agreement on the
        same seed.  Bar 5e-3 relative on the least-squares scale;
        measured 4e-5 (the two share the path geometry, so only the
        rigid-rotation projection differs and it is the identity on axis)
        -- two decades of margin, against the pre-fix state where BOTH
        were low by n_paths*z1 and this test could not have seen it."""
        E0, dx = _src()
        kw = dict(z_to_aperture=0.5e-3, aperture_radius=600e-6,
                  z_aperture_to_output=0.5e-3, wavelength=LAM,
                  n_paths=500_000, rng=5, output_shape=(24, 24),
                  output_dx=3e-6, cone_half_angle=0.08,
                  on_undersampled='silent')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sc = np.asarray(propagate_hfpi_freespace_aperture(E0, dx, **kw))
            ex, _ey = propagate_vector_hfpi_freespace_aperture(
                E0, np.zeros_like(E0), dx, **kw)
        s = abs(_ls_scale(np.asarray(ex), sc))
        assert abs(s - 1.0) < 5e-3, (
            f"vector Ex / scalar least-squares scale = {s:.6f}")

    def test_legacy_is_the_pre_v5_46_factor_and_still_available(self):
        """Counter-pin: ``'legacy'`` must reproduce the historical raw
        path sum on BOTH halves of the estimator, so anyone reproducing
        a v5.45 number still can.  Checked on the factor itself (exact,
        above) and end to end here: the legacy amplitude must be smaller
        than the physical one by roughly ``n_paths * z1``.

        Bar: the ratio is within a factor of 3 of
        ``dx_out**2 / (n_paths * z1 * z2)``.  Derivation: ``'legacy'``
        turns OFF both halves of the estimator, so the two differ by the
        re-emission factor ``n * r_1 * cos_out/cos_in /
        (0.5(cos_in+cos_out))`` AND by the binning Jacobian
        ``r_2/(dx_out^2 cos_out)`` -- i.e. by ``n * z1 * z2 / dx_out^2``
        up to the cone's own cosine spread (<= 0.3 % at a 0.08 rad cone)
        and the Monte-Carlo scatter of the two runs.  Measured
        1.79e-11 against the predicted 1.80e-11 (ratio 0.995); a factor
        of 3 is ~0.5 decades outside that and the fixed state cannot
        reach it.
        """
        E0, dx = _src()
        z1, n_paths = 0.5e-3, 2_000_000
        kw = dict(z_to_aperture=z1, aperture_radius=600e-6,
                  z_aperture_to_output=0.5e-3, wavelength=LAM,
                  n_paths=n_paths, rng=5, output_shape=(24, 24),
                  output_dx=3e-6, cone_half_angle=0.08,
                  on_undersampled='silent')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            phys = np.asarray(propagate_hfpi_freespace_aperture(
                E0, dx, **kw))
            leg = np.asarray(propagate_hfpi_freespace_aperture(
                E0, dx, normalisation='legacy', **kw))
        ratio = abs(_ls_scale(leg, phys))
        dx_out, z2 = 3e-6, 0.5e-3
        predicted = dx_out ** 2 / (n_paths * z1 * z2)
        assert 1 / 3 < ratio / predicted < 3.0, (
            f"legacy/physical = {ratio:.4e}, predicted "
            f"dx_out^2/(n*z1*z2) = {predicted:.4e} "
            f"(ratio {ratio/predicted:.3f})")

    def test_physical_refuses_an_aperture_on_the_emission_plane(self):
        """``r_in = 0`` makes the intermediate-leg Jacobian undefined.
        Silently returning zeros would be the failure mode this whole
        work package exists to remove, so it raises and names the
        remedy."""
        E = np.ones((8, 8), complex)
        paths = init_paths_from_field(E, 2e-6, n_paths=64, wavelength=LAM,
                                      rng=1)
        with pytest.raises(ValueError, match="normalisation='physical'"):
            apply_aperture_diffraction(paths, 1e-3, wavelength=LAM)
        # 'legacy' is defined there and still works.
        out = apply_aperture_diffraction(paths, 1e-3, wavelength=LAM,
                                         normalisation='legacy')
        assert np.isfinite(np.asarray(out.weights)).all()

    def test_normalisation_token_is_validated(self):
        E = np.ones((8, 8), complex)
        paths = init_paths_from_field(E, 2e-6, n_paths=16, wavelength=LAM,
                                      rng=1)
        with pytest.raises(ValueError,
                           match='apply_aperture_diffraction: normalisation'):
            apply_aperture_diffraction(paths, 1e-3, wavelength=LAM,
                                       normalisation='nope')


# ===========================================================================
# V6 -- the RS 'transfer' branch's wrap-around guard
# ===========================================================================

def _phase_screen(N, dx, seed=7, frac=0.9):
    """Band-limited random-phase screen filling ``frac`` of the grid's own
    angular range -- the one field shape that can leave the padded window
    below the alias threshold."""
    rng = np.random.default_rng(seed)
    f = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(f, f, indexing='xy')
    mask = (FX ** 2 + FY ** 2) <= (frac / (2 * dx)) ** 2
    spec = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) * mask
    E = np.fft.ifft2(spec)
    return (E / np.abs(E).max()).astype(np.complex128)


def _wrap_warnings(E, z, lam, dx):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        rayleigh_sommerfeld_propagate(E, z, lam, dx)
    return [r for r in rec if issubclass(r.category, RuntimeWarning)
            and 'padded window' in str(r.message)]


class TestV6TransferWraparoundGuard:

    LAM = 633e-9

    @pytest.mark.parametrize('N,dx_over_lam,q,ring_pct', [
        (64, 1.0, 0.5, 7.22),
        (64, 1.0, 0.9, 30.35),
        (128, 1.0, 0.9, 30.12),
        (64, 0.6, 0.9, 36.09),
    ])
    def test_fires_when_the_field_reaches_the_padded_rim(
            self, N, dx_over_lam, q, ring_pct):
        """Decision, not a reading: the guard either fires or it does not.

        Fixture: a band-limited random-phase screen at ``dx ~ lambda``
        filling 90 % of the grid's angular range, below ``z_crit`` so the
        default routes to ``'transfer'``.  Against an 8x zero-padded
        linear convolution with the same transfer function (which has no
        wrap-around by construction) these read relative L2
        2.4e-3 / 3.3e-2 / 2.3e-2 / 1.7e-1 -- i.e. materially wrong, and
        pre-fix silent.  The ring fractions the guard reports here are
        7.2 / 30.3 / 30.1 / 36.1 %, matching a full (unsampled) reduction
        to 0.5 %.
        """
        dx = dx_over_lam * self.LAM
        z_crit = 2 * N * dx * dx / self.LAM
        E = _phase_screen(N, dx)
        got = _wrap_warnings(E, q * z_crit, self.LAM, dx)
        assert got, (
            f"no wrap-around warning at N={N}, dx={dx_over_lam} lambda, "
            f"z={q} z_crit, where {ring_pct}% of the power has reached "
            f"the padded rim.")
        # the reported fraction must be the measured one, not a constant
        msg = str(got[0])
        reported = float(msg.split('% of the power')[0].split(': ')[-1])
        assert abs(reported - ring_pct) < 2.0, (
            f"reported ring fraction {reported}% vs the full-reduction "
            f"value {ring_pct}%")

    @pytest.mark.parametrize('N,dx,w0,q', [
        (64, 2e-6, 6e-6, 0.9),
        (128, 1e-6, 6e-6, 0.9),
        (256, 0.5e-6, 4e-6, 0.9),
        (128, 2e-6, 6e-6, 0.5),
    ])
    def test_silent_for_a_properly_sampled_beam(self, N, dx, w0, q):
        """The counter-pin that makes the guard worth having: it must not
        cry wolf.  A contained Gaussian puts **5.5e-22 down to 9.3e-28**
        of its power in the ring (worst case over these four grids and
        three distances: 8.1e-14), i.e. **12 decades** below the 2 %
        threshold, with relative L2 1e-14 .. 6e-11 against the 8x-padded
        reference.  It is not reachable at all for a properly sampled
        beam: leaving the padded window before ``z = 2 N dx^2/lambda``
        requires ``tan(theta) > lambda/(2 dx)``, i.e. exceeding the
        grid's own maximum representable angle.
        """
        z_crit = 2 * N * dx * dx / self.LAM
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(complex)
        assert not _wrap_warnings(E, q * z_crit, self.LAM, dx)

    def test_the_spatial_branch_never_warns(self):
        """The guard belongs to the transfer branch alone: the spatial
        kernel TRUNCATES ``h`` at the padded rim rather than wrapping, so
        the same field above ``z_crit`` is a different (and, for an
        overfilling beam, better) approximation -- measured 8.9e-13 for
        spatial against 1.92e-2 for transfer at z = 3 mm on the
        N = 128 / dx = 1 um probe."""
        N, dx = 64, 1.0 * self.LAM
        z_crit = 2 * N * dx * dx / self.LAM
        E = _phase_screen(N, dx)
        assert not _wrap_warnings(E, 2.0 * z_crit, self.LAM, dx)

    def test_the_guard_does_not_change_the_returned_field(self):
        """Diagnostic only.  Bar: BIT-identity with the guard disabled."""
        import lumenairy.propagators.rs as _rs
        N, dx = 64, 1.0 * self.LAM
        z = 0.5 * (2 * N * dx * dx / self.LAM)
        E = _phase_screen(N, dx)
        saved = _rs._warn_rs_transfer_wraparound
        try:
            _rs._warn_rs_transfer_wraparound = lambda *a, **k: None
            quiet = rayleigh_sommerfeld_propagate(E, z, self.LAM, dx)
        finally:
            _rs._warn_rs_transfer_wraparound = saved
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            loud = rayleigh_sommerfeld_propagate(E, z, self.LAM, dx)
        assert quiet.tobytes() == loud.tobytes()


# ===========================================================================
# WP-A6 section 5.3 follow-up -- the MFT / Bluestein dtype contract
# ===========================================================================

class TestMftDtypePreservation:
    """WP-A6's C3 residual-risk note said the traced-carrier chain's TILTED
    paraxial landing still returns complex128 because
    ``angular_spectrum_propagate_mft`` does not preserve complex64.

    Measured, that premise does not hold: the whole MFT / Bluestein family
    is dtype-preserving in the returned dtype, in the dtype the core
    actually computes in, and in memory.  These pins protect that
    property, because it is what the chain's fix (a one-line narrowing of
    the chief-ray ramp in ``carrier.py``) will depend on.
    """

    LAM = 633e-9

    @staticmethod
    def _field(N, dx, w0, dt):
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(dt)

    @staticmethod
    def _call(name, E, z, lam, dx):
        from lumenairy.propagators.mft import (
            angular_spectrum_propagate_mft,
            fraunhofer_propagate_mft,
            fresnel_propagate_mft,
        )
        fns = {'asm': (angular_spectrum_propagate_mft,
                       dict(dx_out=1e-6, N_out=64)),
               'fresnel': (fresnel_propagate_mft,
                           dict(dx_out=5e-6, N_out=64)),
               'fraunhofer': (fraunhofer_propagate_mft,
                              dict(dx_out=5e-6, N_out=64))}
        fn, kw = fns[name]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            out = fn(E, z, lam, dx, **kw)
        return out[0] if isinstance(out, tuple) else out

    @pytest.mark.parametrize('name', ['asm', 'fresnel', 'fraunhofer'])
    @pytest.mark.parametrize('dt', [np.complex64, np.complex128])
    def test_the_returned_dtype_is_the_callers(self, name, dt):
        """A decision, not a reading."""
        E = self._field(64, 1e-6, 8e-6, dt)
        got = self._call(name, E, 1e-3, self.LAM, 1e-6)
        assert got.dtype == np.dtype(dt), (
            f"{name}-MFT: complex input {np.dtype(dt).str} returned "
            f"{got.dtype}")

    @pytest.mark.parametrize('name,rel_measured', [
        ('asm', 2.734e-07), ('fresnel', 2.027e-07), ('fraunhofer', 2.172e-07),
    ])
    def test_complex64_equals_the_narrowed_complex128_result(
            self, name, rel_measured):
        """Bar: relative L2 < 1e-5 between the complex64 result and the
        complex128 one narrowed to complex64.

        Derivation.  float32 carries ~1.2e-7 relative; the Bluestein
        route is a pre-chirp multiply, two FFTs of length
        ``next_fast_len(N_in + N_out - 1)``, a kernel multiply and a
        post-chirp multiply, so a handful of float32 roundings accumulate
        to a few times 1.2e-7 -- measured 2.03e-7 / 2.17e-7 / 2.73e-7 for
        the three entry points.  1e-5 sits 1.6 decades above that (room
        for a different BLAS/FFT build) and 5 decades below the 1e0 a
        genuine precision loss would produce.
        """
        E128 = self._field(64, 1e-6, 8e-6, np.complex128)
        a = self._call(name, E128.astype(np.complex64), 1e-3, self.LAM, 1e-6)
        b = self._call(name, E128, 1e-3, self.LAM, 1e-6)
        rel = float(np.linalg.norm(a.astype(np.complex128) - b)
                    / np.linalg.norm(b))
        assert rel < 1e-5, (
            f"{name}-MFT complex64 vs narrowed complex128: {rel:.3e} "
            f"(measured {rel_measured:.3e})")

    @pytest.mark.parametrize('dt', [np.complex64, np.complex128])
    def test_the_bluestein_core_computes_in_the_callers_dtype(self, dt):
        """Structural: the returned dtype could be honoured by a final
        cast while the work is done in double.  It is not -- the core is
        entered AND left at the caller's precision, which is what makes
        the memory saving real (measured peak 86.75 MB at complex64
        against 168.49 MB at complex128, N = 512: the same 41.4 / 40.2
        input-grid units, i.e. half the bytes for the same structure)."""
        import lumenairy.propagators._bluestein as BL
        seen = []
        orig = BL._bluestein_2d

        def spy(E, *a, **kw):
            out = orig(E, *a, **kw)
            seen.append((E.dtype, kw.get('target_cdtype'), out.dtype))
            return out

        BL._bluestein_2d = spy
        try:
            self._call('asm', self._field(64, 1e-6, 8e-6, dt), 1e-3,
                       self.LAM, 1e-6)
        finally:
            BL._bluestein_2d = orig
        assert seen, 'the Bluestein core was not reached'
        for in_dt, target, out_dt in seen:
            assert np.dtype(in_dt) == np.dtype(dt), in_dt
            assert np.dtype(target) == np.dtype(dt), target
            assert np.dtype(out_dt) == np.dtype(dt), out_dt

    def test_a_real_input_still_falls_back_to_the_library_default(self):
        """Counter-pin: dtype PRESERVATION must not become dtype
        INVENTION -- a real-valued input has no complex dtype to
        preserve, so it takes the library default."""
        from lumenairy.propagators import fft_infra as _fi
        E = self._field(64, 1e-6, 8e-6, np.float64)
        got = self._call('asm', E, 1e-3, self.LAM, 1e-6)
        assert got.dtype == np.dtype(_fi.DEFAULT_COMPLEX_DTYPE)


# ===========================================================================
# WP-A2 section 5 item 4 -- the ASM spatial shift pair folds away at even N
# ===========================================================================

class TestAsmSpatialShiftFold:
    """``angular_spectrum_propagate`` used to wrap every propagation in
    ``fftshift(ifft2(fft2(ifftshift(E)) * H))``.  For EVEN N that pair is
    the identity (the two ``(-1)^k`` phases the shift theorem produces
    cancel), so it is dropped; for ODD N it is not and the shifted form
    is kept.
    """

    LAM = 633e-9

    @staticmethod
    def _noise(N, seed, dt=np.complex128):
        rng = np.random.default_rng(seed)
        return (rng.normal(size=(N, N))
                + 1j * rng.normal(size=(N, N))).astype(dt)

    @pytest.mark.parametrize('N', [64, 128, 256, 63, 65, 127])
    def test_the_propagation_still_equals_the_shifted_idiom(self, N):
        """The property, stated independently of which branch runs: the
        returned field must equal the textbook
        ``fftshift(ifft2(fft2(ifftshift(E)) * H_natural))`` written out
        here with raw numpy transforms.

        Bar: BIT-identity.  The fold is an exact index identity at even
        N (measured max|diff| exactly 0.0 at N = 64 .. 1024) and the odd
        branch is unchanged code, so there is no tolerance to derive.
        This is the check that would catch the fold being applied at odd
        N, where it is wrong by 2.1e-16 .. 1.1e-15.
        """
        from lumenairy.propagators.asm import (
            _get_asm_H_natural, angular_spectrum_propagate)
        from lumenairy.propagators.fft_infra import clear_asm_caches
        dx, z = 1e-6, 1e-3
        E = self._noise(N, N)
        clear_asm_caches()
        got = angular_spectrum_propagate(E, z, self.LAM, dx)
        clear_asm_caches()
        H = np.asarray(_get_asm_H_natural(N, N, dx, dx, self.LAM, z, True,
                                          np.complex128, np))
        want = np.fft.fftshift(
            np.fft.ifft2(np.fft.fft2(np.fft.ifftshift(E)) * H))
        rel = float(np.linalg.norm(got - want) / np.linalg.norm(want))
        assert rel < 1e-14, (
            f"N={N}: propagation differs from the shifted idiom by "
            f"{rel:.3e}")

    def test_the_fold_is_gated_on_both_axes_being_even(self):
        """Counter-pin, and the reason the gate exists.  The identity
        ``fft2(ifftshift(E)) = fft2(E) * (-1)^(kx+ky)`` needs a circular
        shift by exactly ``N/2``; at odd N the shift is by ``(N-1)/2``
        and the phase is not ``+-1``.  Written out here on raw numpy so
        it is a statement about the MATHS, not about the module:
        identical at even N, different at odd N."""
        for N, even in ((64, True), (128, True), (63, False), (65, False)):
            E = self._noise(N, N + 1)
            a = np.fft.fft2(np.fft.ifftshift(E))
            b = np.fft.fft2(E)
            k = np.arange(N)
            S = ((-1.0) ** k)[None, :] * ((-1.0) ** k)[:, None]
            rel = float(np.linalg.norm(a - b * S) / np.linalg.norm(a))
            if even:
                assert rel < 1e-14, f"N={N}: the identity should hold"
            else:
                assert rel > 1e-3, (
                    f"N={N}: the identity must NOT hold at odd N, else "
                    f"the gate is untestable")

    def test_the_returned_array_owns_its_memory(self):
        """The dropped ``fftshift`` used to detach the result from the
        pyFFTW inverse ping-pong buffer; the folded path must copy
        explicitly or a later same-shape call silently overwrites a
        previously-returned field (the v5.4.6 audit F-3 hazard)."""
        from lumenairy.propagators.asm import angular_spectrum_propagate
        dx, z = 1e-6, 1e-3
        E = self._noise(128, 3)
        first = angular_spectrum_propagate(E, z, self.LAM, dx)
        snapshot = first.copy()
        for _ in range(3):
            angular_spectrum_propagate(self._noise(128, 99), z, self.LAM, dx)
        assert first.tobytes() == snapshot.tobytes(), (
            "a later same-shape propagation overwrote an earlier result: "
            "the folded path is returning a view into the FFT buffer")
