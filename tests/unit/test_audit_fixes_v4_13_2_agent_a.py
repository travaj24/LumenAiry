"""Pinning tests for the v4.13.2 audit (Agent A scope).

Covers six audit items closed by Agent A in the v4.13.1 -> v4.13.2 patch
pass.  Each test pins exactly one finding so a regression points
straight at the relevant fix.

* **P1-NEW-A** :func:`propagate_vector_hfpi_freespace_aperture` shared
  the caller ``rng`` between source-plane init and aperture re-emission
  (same bug fixed in scalar HFPI in v4.11.2).  v4.13.2 calls
  ``_spawn_rng(rng, 0)`` / ``_spawn_rng(rng, 1)`` for the two sites so
  the draws are statistically independent.
* **P1-NEW-H** Scalar :func:`hfpi.propagate_to_plane` and vector
  :func:`vectorial_hfpi.propagate_vector_to_plane` produced inf/NaN
  positions for grazing rays (``|N_z| <= 1e-30`` and
  ``z_target != z_curr``).  ``new_alive`` correctly tagged the rays
  dead, but ``paths.positions`` still carried +/-inf so any downstream
  consumer that read positions without masking by ``alive`` was
  poisoned.  v4.13.2 zeros the step on dead rays so the position
  update is a no-op.
* **P1-NEW-B** :func:`propagate_subaperture_asymptotic` always passed
  ``source_amplitudes={(0, 0): 1.0 + 0.0j}`` to the underlying modal
  asymptotic propagator -- ``E_in`` was silently discarded beyond a
  waist estimate.  v4.13.2 projects ``E_in`` onto the LG basis
  per-patch (centred at each patch's source point), mirroring the
  v4.11.2 fix in :func:`hf.propagate_huygens_fresnel_through_prescription`.
* **P1-NEW-C** :func:`analysis.field.petzval_radius` skipped surfaces
  with ``n1 == n2`` -- which silently drops every mirror in a
  prescription (the loader returns mirrors with
  ``glass_after == glass_before``).  v4.13.2 applies the Welford
  ``n2 = -n1`` convention with mirror-parity tracking, matching the
  canonical :func:`raytrace.seidel_analysis.seidel_coefficients`.
* **P1-NEW-D** :func:`_build_jax_prescription` refused
  ``is_mirror=True`` but did not check the
  ``glass_after='MIRROR'`` marker convention.  v4.13.2 adds the
  case-insensitive ``glass_after`` check, matching the v4.13.1 P1-A
  ``apply_real_lens`` fix.
* **C-P1-1** :func:`analysis.core.strehl_ratio` and
  :func:`analysis.core.polychromatic_psf` used ``dx ** 2`` for the
  per-pixel area in the total-power normalisation; v4.13.2 adds a
  ``dy`` kwarg (defaulting to ``dx`` for backward compatibility) and
  uses ``dx * dy``.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as lm
from lumenairy.propagators.hfpi import (
    PathBundle,
    propagate_to_plane,
    propagate_hfpi_freespace_aperture,
)
from lumenairy.propagators.vectorial_hfpi import (
    VectorPathBundle,
    propagate_vector_to_plane,
    propagate_vector_hfpi_freespace_aperture,
)
from lumenairy.propagators.subaperture import (
    propagate_subaperture_asymptotic,
)
from lumenairy.raytrace.core import Surface
from lumenairy.analysis.field import petzval_radius
from lumenairy.analysis.core import strehl_ratio, polychromatic_psf


# ============================================================================
# A.1 -- P1-NEW-A: vectorial HFPI RNG sibling-gap
# ============================================================================

class TestVectorialHfpiRngSpawn:
    """Pre-4.13.2 :func:`propagate_vector_hfpi_freespace_aperture` passed
    the same int ``rng`` to BOTH ``init_vector_paths_from_field`` AND
    ``apply_vector_aperture_diffraction``.  Because
    :class:`RandomState(rng=int)` rebuilds ``default_rng(int)`` on every
    construction, both sites drew identical uniform samples -- perfectly
    correlated init / re-emission.  v4.13.2 derives a distinct child
    seed via ``_spawn_rng`` for each site.
    """

    def test_p1_new_a_vector_hfpi_init_and_aperture_decorrelated(self):
        """Comparing the v4.13.2 result to a synthetic 'always-equal-seed'
        baseline picks up the correlation difference.  We capture the
        per-site RNG draws via :func:`_spawn_rng` and assert that the
        two child seeds are different (the kernel of the fix).
        """
        from lumenairy.propagators.hfpi import _spawn_rng

        s0 = _spawn_rng(42, 0)
        s1 = _spawn_rng(42, 1)
        assert s0 != s1, (
            f"_spawn_rng(42, 0) = {s0!r} == _spawn_rng(42, 1) = {s1!r}; "
            f"vector HFPI source-init and aperture re-emission would "
            f"redraw identical uniform samples (the P1-NEW-A bug)."
        )

    def test_p1_new_a_vector_hfpi_runs_and_produces_finite_field(self):
        """End-to-end smoke test: the vectorial HFPI pipeline must run
        and produce a finite (Ex, Ey) field.  Without the fix, the same
        integer rng for init and aperture still 'works' numerically;
        the pinning test is the spawn-distinctness test above and this
        smoke test guards the call-site plumbing.
        """
        N = 32
        dx = 5e-6
        Ex_in = np.ones((N, N), dtype=np.complex128)
        Ey_in = np.zeros((N, N), dtype=np.complex128)
        wavelength = 1.55e-6
        Ex_out, Ey_out = propagate_vector_hfpi_freespace_aperture(
            Ex_in, Ey_in, dx,
            z_to_aperture=1e-2,
            aperture_radius=20e-6,
            z_aperture_to_output=1e-2,
            wavelength=wavelength,
            n_paths=512,
            rng=7,
        )
        assert np.all(np.isfinite(Ex_out)), (
            "Vector HFPI Ex_out has non-finite entries -- pipeline broken."
        )
        assert np.all(np.isfinite(Ey_out)), (
            "Vector HFPI Ey_out has non-finite entries -- pipeline broken."
        )


# ============================================================================
# A.2 -- P1-NEW-H: propagate_to_plane inf/NaN positions on grazing rays
# ============================================================================

class TestPropagateToPlaneGrazingRays:
    """Pre-4.13.2 a grazing ray (``|N_z| <= 1e-30`` and
    ``z_target != z_curr``) produced ``t ~ 1e30`` and the position
    update added inf/NaN to ``paths.positions`` even though ``alive``
    correctly flagged the ray dead.  Any downstream consumer reading
    positions without re-masking would be poisoned.  v4.13.2 zeros
    ``t`` on dead rays so the position update is a no-op.
    """

    def test_p1_new_h_scalar_hfpi_grazing_ray_finite_position(self):
        """A grazing ray (N_z=0) at a fresh position must end up with
        a finite (not inf/NaN) position after :func:`propagate_to_plane`
        even though ``alive`` is False.
        """
        # One alive grazing ray at the origin pointing +x.
        pos = np.array([[0.0, 0.0, 0.0]])
        dir_ = np.array([[1.0, 0.0, 0.0]])  # N_z = 0
        weights = np.array([1.0 + 0.0j])
        opl = np.array([0.0])
        alive = np.array([True])
        paths = PathBundle(
            positions=pos, directions=dir_,
            weights=weights, opl=opl, alive=alive)

        out = propagate_to_plane(paths, z_target=1.0, wavelength=1.55e-6)
        # The ray is killed by the alive mask...
        assert bool(out.alive[0]) is False, (
            "Grazing ray (N_z=0, z_target != z) must be flagged dead."
        )
        # ...but its position must be finite (the P1-NEW-H bug
        # produced +/-inf here).
        assert np.all(np.isfinite(out.positions[0])), (
            f"Grazing ray position has non-finite entries: "
            f"{out.positions[0]!r}.  Pre-4.13.2 t ~ 1e30 poisoned "
            f"new_positions with inf even though alive=False."
        )

    def test_p1_new_h_vector_hfpi_grazing_ray_finite_position(self):
        """Same test for the vectorial twin."""
        pos = np.array([[0.0, 0.0, 0.0]])
        dir_ = np.array([[1.0, 0.0, 0.0]])  # N_z = 0
        Ex = np.array([1.0 + 0.0j])
        Ey = np.array([0.0 + 0.0j])
        opl = np.array([0.0])
        alive = np.array([True])
        paths = VectorPathBundle(
            positions=pos, directions=dir_,
            Ex=Ex, Ey=Ey, opl=opl, alive=alive)

        out = propagate_vector_to_plane(paths, z_target=1.0,
                                          wavelength=1.55e-6)
        assert bool(out.alive[0]) is False, (
            "Vector grazing ray must be flagged dead."
        )
        assert np.all(np.isfinite(out.positions[0])), (
            f"Vector grazing ray position has non-finite entries: "
            f"{out.positions[0]!r}.  Pre-4.13.2 the same bug poisoned "
            f"the vectorial twin."
        )


# ============================================================================
# A.3 -- P1-NEW-B: subaperture decompose_lg sibling-gap
# ============================================================================

class TestSubapertureDecomposeLg:
    """Pre-4.13.2 :func:`propagate_subaperture_asymptotic` always set
    ``source_amplitudes={(0, 0): 1.0+0.0j}`` regardless of ``E_in``,
    so a structured input field (off-axis Gaussian, vortex, Airy) was
    silently replaced by a unit fundamental Gaussian.  v4.13.2
    projects ``E_in`` onto the LG basis per-patch (centred at each
    patch's source point) and forwards the actual amplitudes.
    """

    def _make_singlet(self):
        rx = lm.make_singlet(
            R1=20e-3, R2=-20e-3, d=2e-3,
            glass='N-BK7', aperture=6e-3,
        )
        rx['object_distance'] = 0.1
        return rx

    def test_p1_new_b_subaperture_distinguishes_structured_input(self):
        """A non-trivial off-axis input field must give a different
        output than a unit on-axis Gaussian.  Pre-4.13.2 both would
        produce identical (Gaussian-image) output because E_in was
        silently discarded.
        """
        rx = self._make_singlet()
        wavelength = 1.55e-6
        N = 32
        dx = 2e-6

        # Build two distinct structured inputs to compare.
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y, indexing='xy')

        # Input 1: on-axis Gaussian (single LG_{0,0} mode).
        w0 = 30e-6
        E_a = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)

        # Input 2: same Gaussian with a vortex (l=1) phase ramp -- a
        # very different LG decomposition (concentrates on (p=0, l=1)).
        phi = np.arctan2(Y, X)
        E_b = (E_a * np.exp(1j * phi)).astype(np.complex128)

        # Run subaperture asymptotic for each.  Small box / coarse fits
        # to keep the test fast.
        F_a = propagate_subaperture_asymptotic(
            E_a, dx, rx, wavelength=wavelength,
            source_box_half=20e-6, pupil_box_half=2e-3,
            n_field=4, n_pupil=4, poly_order=3,
            n_patches=(2, 2),
        )
        F_b = propagate_subaperture_asymptotic(
            E_b, dx, rx, wavelength=wavelength,
            source_box_half=20e-6, pupil_box_half=2e-3,
            n_field=4, n_pupil=4, poly_order=3,
            n_patches=(2, 2),
        )

        # Pre-4.13.2: F_a == F_b (both replaced by unit LG_{0,0}).
        # v4.13.2: F_a != F_b because decompose_lg picks up the vortex
        # phase.  We compare via a normalised L2 distance.
        denom = max(float(np.linalg.norm(F_a)), 1e-30)
        diff = float(np.linalg.norm(F_a - F_b)) / denom
        assert diff > 1e-4, (
            f"propagate_subaperture_asymptotic returned essentially "
            f"identical fields for an on-axis Gaussian and a vortex "
            f"variant (diff={diff:.3e}).  This is the P1-NEW-B bug -- "
            f"E_in is being silently discarded and replaced with a "
            f"unit LG_{{0,0}} mode."
        )


# ============================================================================
# A.4 -- P1-NEW-C: petzval_radius Welford-mirror sibling-gap
# ============================================================================

class TestPetzvalRadiusMirror:
    """Pre-4.13.2 :func:`petzval_radius` skipped surfaces where
    ``n1 == n2``.  But ``surfaces_from_prescription`` returns mirrors
    with ``glass_before == glass_after`` (so ``n1 == n2``), silently
    dropping every mirror's Petzval contribution.  v4.13.2 uses the
    Welford ``n2 = -n1`` convention with mirror-parity tracking,
    matching :func:`raytrace.seidel_analysis.seidel_coefficients`.
    """

    def test_p1_new_c_single_mirror_matches_mahajan(self):
        """For a single concave mirror in air at radius ``R = -0.5 m``
        (Welford convention: R<0 means centre on the incoming side),
        the Petzval radius is

            R_p = -1 / sum,   sum = (n2 - n1) / (n1 n2 R)
                = -1 / ((-2) / ((+1)(-1)(-0.5)))
                = -1 / ((-2) / (+0.5))
                = -1 / (-4)
                = +0.25 m

        Pre-4.13.2 the mirror was skipped entirely and the result was
        +inf (a 'Petzval-flat' system).  v4.13.2 returns +0.25 m.
        """
        R_mirror = -0.5
        surfaces = [
            Surface(radius=R_mirror, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True, semi_diameter=50e-3),
        ]
        wavelength = 0.55e-6
        R_p = petzval_radius(surfaces, wavelength)
        # n1 = +1, n2 = -1 (Welford), R = -0.5.
        # (n2 - n1) / (n1 n2 R) = (-2) / ((+1)(-1)(-0.5)) = -2 / 0.5 = -4
        # R_p = -1 / (-4) = +0.25
        expected = 0.25
        assert np.isfinite(R_p), (
            f"Single-mirror Petzval radius must be finite; got {R_p!r}.  "
            f"Pre-4.13.2 the n1==n2 skip dropped the mirror's "
            f"contribution and the function returned +inf."
        )
        assert abs(R_p - expected) < 1e-9, (
            f"Single-mirror Petzval radius = {R_p!r}; expected "
            f"{expected} (Welford n2=-n1).  Pre-4.13.2: +inf."
        )

    def test_p1_new_c_two_mirror_cassegrain_petzval_finite(self):
        """A two-mirror Cassegrain-like geometry must yield a finite
        Petzval radius.  Pre-4.13.2 both mirrors were silently dropped
        and the result was +inf.  We don't need the exact analytic
        answer here -- the regression is "is the contribution
        included?" -- but we do compute it to pin a specific value.

        Primary:   R1 = -1.0, mirror_parity 0->1
                   contribution = (n2 - n1)/(n1 n2 R1)
                                = (-1 - 1) / ((1)(-1)(-1.0)) = -2
        Secondary: R2 = -0.3, mirror_parity 1 (n1 = -1) -> 0 (n2 = +1)
                   contribution = (n2 - n1)/(n1 n2 R2)
                                = (1 - (-1)) / ((-1)(1)(-0.3)) = 2/0.3
                                = +6.666...
        sum = -2 + 6.6667 = 4.6667
        R_p = -1 / 4.6667 = -0.214286
        """
        surfaces = [
            Surface(radius=-1.0, thickness=0.4,
                    glass_before='air', glass_after='air',
                    is_mirror=True, is_stop=True,
                    semi_diameter=50e-3),
            Surface(radius=-0.3, thickness=0.0,
                    glass_before='air', glass_after='air',
                    is_mirror=True, semi_diameter=15e-3),
        ]
        wavelength = 0.55e-6
        R_p = petzval_radius(surfaces, wavelength)
        assert np.isfinite(R_p), (
            f"Cassegrain Petzval radius must be finite; got {R_p!r}.  "
            f"Pre-4.13.2 both mirrors were dropped by the n1==n2 "
            f"skip and the result was +inf -- 100% wrong."
        )
        sum_inv = -2.0 + 2.0 / 0.3
        expected = -1.0 / sum_inv
        assert abs(R_p - expected) < 1e-6, (
            f"Cassegrain Petzval radius = {R_p!r}; expected "
            f"{expected} (Welford parity-tracked, both contributions "
            f"included)."
        )


# ============================================================================
# A.5 -- P1-NEW-D: _build_jax_prescription glass_after='MIRROR' check
# ============================================================================

class TestJaxPrescriptionMirrorMarker:
    """Pre-4.13.2 :func:`_build_jax_prescription` rejected mirrors
    signalled by ``is_mirror=True`` but did NOT check the
    ``glass_after='MIRROR'`` marker convention used by the .zmx
    loader.  A hand-built Welford-style prescription with that marker
    slipped through and was silently traced as a refractive air->air
    surface.  v4.13.2 adds the case-insensitive ``glass_after`` check.
    """

    def test_p1_new_d_rejects_glass_after_MIRROR_marker(self):
        """A prescription with ``glass_after='MIRROR'`` must trigger
        the same NotImplementedError as ``is_mirror=True``.
        """
        try:
            import jax  # noqa: F401
        except ImportError:
            pytest.skip("JAX not installed -- skipping JAX-trace test.")

        from lumenairy.raytrace.jax_trace import _build_jax_prescription

        rx = {
            'surfaces': [
                {'radius': -1.0, 'conic': -1.0,
                 'glass_before': 'air', 'glass_after': 'MIRROR',
                 'semi_diameter': 50e-3},
                {'radius': float('inf'), 'conic': 0.0,
                 'glass_before': 'air', 'glass_after': 'air',
                 'semi_diameter': 50e-3},
            ],
            'thicknesses': [0.4],
            'aperture_diameter': 0.1,
        }
        wavelength = 587.56e-9
        with pytest.raises(NotImplementedError, match=r'is_mirror'):
            _build_jax_prescription(rx, wavelength)

    def test_p1_new_d_rejects_lowercase_mirror_marker(self):
        """The check must be case-insensitive (the .zmx loader uppercases
        on parse, but a hand-built prescription could be lowercase).
        """
        try:
            import jax  # noqa: F401
        except ImportError:
            pytest.skip("JAX not installed -- skipping JAX-trace test.")

        from lumenairy.raytrace.jax_trace import _build_jax_prescription

        rx = {
            'surfaces': [
                {'radius': -1.0, 'glass_before': 'air',
                 'glass_after': 'mirror', 'semi_diameter': 50e-3},
            ],
            'thicknesses': [],
            'aperture_diameter': 0.1,
        }
        wavelength = 587.56e-9
        with pytest.raises(NotImplementedError, match=r'is_mirror'):
            _build_jax_prescription(rx, wavelength)


# ============================================================================
# A.6 -- C-P1-1: strehl_ratio + polychromatic_psf use dx*dy
# ============================================================================

class TestStrehlPolyDxDy:
    """Pre-4.13.2 :func:`strehl_ratio` and :func:`polychromatic_psf`
    used ``dx ** 2`` for the per-pixel area normalisation.  Although
    the Strehl ratio is dimensionless and the pixel area cancels in
    the ratio, an anamorphic / non-square grid mis-scaled the
    intermediate total-power numbers.  v4.13.2 adds a ``dy`` kwarg
    (default ``dy=dx`` for back-compat) and uses ``dx * dy``.
    """

    def test_p1_c1_strehl_back_compat_with_dy_default(self):
        """Omitting ``dy`` reproduces the v4.13.1 result.  Numerics
        differ from ``dy=dx`` at roundoff level (``dx ** 2`` vs
        ``dx * dy`` use slightly different IEEE rounding even with
        ``dy == dx``); v4.13.2 preserves the historic ``dx ** 2``
        form when ``dy`` is omitted so callers that did not pass a
        ``dy`` see bit-identical behaviour to v4.13.1.
        """
        rng = np.random.default_rng(7)
        E = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        E_ref = rng.standard_normal((16, 16)) + 1j * rng.standard_normal((16, 16))
        dx = 1e-6
        s_default = strehl_ratio(E.copy(), E_ref.copy(), dx)
        s_explicit = strehl_ratio(E.copy(), E_ref.copy(), dx, dy=dx)
        # Both must yield finite, near-equal Strehl ratios; the
        # tolerance is set above 0 because ``dx ** 2`` and ``dx * dy``
        # are not bit-identical even when ``dy == dx`` (the rounding
        # pathway differs).
        assert np.isfinite(s_default) and np.isfinite(s_explicit), (
            f"strehl_ratio returned non-finite values: "
            f"default={s_default!r}, explicit={s_explicit!r}."
        )
        assert abs(s_default - s_explicit) < 1e-10, (
            f"strehl_ratio(dy=None) and strehl_ratio(dy=dx) disagree "
            f"by more than roundoff; got {s_default!r} vs "
            f"{s_explicit!r}."
        )

    def test_p1_c1_strehl_identity_under_anamorphic_grid(self):
        """Strehl is dimensionless.  Doubling ``dy`` must not change
        the ratio (the dx*dy factors cancel exactly).  This pins the
        anamorphic-correct behaviour.
        """
        rng = np.random.default_rng(11)
        E = (rng.standard_normal((24, 24))
             + 1j * rng.standard_normal((24, 24)))
        E_ref = (rng.standard_normal((24, 24))
                 + 1j * rng.standard_normal((24, 24)))
        dx = 1e-6
        s_square = strehl_ratio(E.copy(), E_ref.copy(), dx, dy=dx)
        s_anamorphic = strehl_ratio(E.copy(), E_ref.copy(), dx, dy=2 * dx)
        # The ratio is dimensionless -- max/sum scales independently of
        # the area factor (it's a true ratio of equally-shaped
        # quantities) so the result must be exactly equal.
        assert abs(s_square - s_anamorphic) < 1e-12, (
            f"strehl_ratio differs between dy=dx ({s_square!r}) and "
            f"dy=2*dx ({s_anamorphic!r}).  Anamorphic invariance "
            f"broken; the dx*dy factor is not being applied correctly."
        )

    def test_p1_c1_polychromatic_psf_dy_kwarg_accepted(self):
        """``polychromatic_psf`` accepts the new ``dy`` kwarg without
        error and defaults to ``dy=dx``.  This is a smoke test for
        the signature change.
        """
        rx = lm.make_singlet(
            R1=50e-3, R2=float('inf'), d=2e-3,
            glass='N-BK7', aperture=10e-3,
        )
        wavelengths = [1.30e-6, 1.55e-6]
        weights = [0.5, 0.5]
        # Tiny grid + 2 wavelengths for speed.
        N = 32
        dx = 5e-6
        psf_a, dx_a, info_a = polychromatic_psf(
            rx, wavelengths, weights, N, dx)
        psf_b, dx_b, info_b = polychromatic_psf(
            rx, wavelengths, weights, N, dx, dy=dx)
        assert np.allclose(psf_a, psf_b, atol=1e-15), (
            "polychromatic_psf(dy=None) != polychromatic_psf(dy=dx); "
            "back-compat broken."
        )
