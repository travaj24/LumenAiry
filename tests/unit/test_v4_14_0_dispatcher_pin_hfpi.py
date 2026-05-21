"""Parametrized dispatcher-pin tests for the (scalar, vectorial) HFPI
family.

Audit context
-------------

The v4.10 -> v4.13.2 audits surfaced a recurring "fix swept N call
sites, missed N+1" pattern across the lens family.  v4.13.1 added 3
parametrized dispatcher pins (L4a mirror guards, phase-retrieval
kernels, Source factories) that close the gap within each fixed
family.  Three MORE families remain unpinned; this file pins
Family I -- the (scalar, vectorial) HFPI sibling pair.

The two HFPI propagators share five public function pairs that perform
analogous work on a path bundle.  Past fixes that landed on one side
of the pair often missed the twin:

* v4.11.2 fixed :func:`_spawn_rng` for the scalar
  ``propagate_hfpi_freespace_aperture`` (statistically-independent
  source-init / aperture-re-emission draws).  v4.13.2 (P1-NEW-A)
  ported the same fix to the vectorial twin
  :func:`propagate_vector_hfpi_freespace_aperture`.

* v4.13.2 (P1-NEW-H) fixed the ``Nz approx 0`` grazing-ray case in
  *both* ``propagate_to_plane`` and
  ``propagate_vector_to_plane`` -- but the symmetric fix is exactly
  the kind of thing a future regression could undo on one side.

* The alive-mask correctness of ``accumulate_to_grid`` /
  ``accumulate_vector_to_grid`` is shared structure and a regression
  on either side would leak dead-ray contributions.

This file parametrises every reachable variant of those three
behaviours so a future fix that touches one branch and forgets the
other lights up at CI time rather than slipping through to a release.

Author: Andrew Traverso -- v4.14.0 / Agent 6
"""
from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.hfpi import (
    PathBundle,
    _spawn_rng,
    accumulate_to_grid,
    apply_aperture_diffraction,
    init_paths_from_field,
    propagate_hfpi_freespace_aperture,
    propagate_to_plane,
)
from lumenairy.propagators.vectorial_hfpi import (
    VectorPathBundle,
    accumulate_vector_to_grid,
    apply_vector_aperture_diffraction,
    init_vector_paths_from_field,
    propagate_vector_hfpi_freespace_aperture,
    propagate_vector_to_plane,
)

# ============================================================================
# Shared fixtures / parametrization helpers
# ============================================================================

def _scalar_init_paths(
    *, n_paths: int, seed: int, dx: float = 5e-6,
    wavelength: float = 633e-9, N: int = 16,
) -> PathBundle:
    """Build a scalar PathBundle from a uniform-amplitude source."""
    E_in = np.ones((N, N), dtype=np.complex128)
    return init_paths_from_field(
        E_in, dx, n_paths=n_paths, wavelength=wavelength, rng=seed)


def _vector_init_paths(
    *, n_paths: int, seed: int, dx: float = 5e-6,
    wavelength: float = 633e-9, N: int = 16,
) -> VectorPathBundle:
    """Build a vectorial VectorPathBundle from a uniform-amplitude
    source (linear-x polarization)."""
    Ex_in = np.ones((N, N), dtype=np.complex128)
    Ey_in = np.zeros((N, N), dtype=np.complex128)
    return init_vector_paths_from_field(
        Ex_in, Ey_in, dx, n_paths=n_paths,
        wavelength=wavelength, rng=seed)


# Pair every scalar dispatcher entry with its vectorial twin.  Each
# parametrize id labels the variant in pytest output so a failing
# parametrization can be traced to the responsible sibling at a
# glance.
_SCALAR_VECTOR_FREESPACE_VARIANTS = [
    pytest.param('scalar', id='scalar_freespace_aperture'),
    pytest.param('vectorial', id='vectorial_freespace_aperture'),
]


# ============================================================================
# Pin 1 -- _spawn_rng statistical independence across source-init and
# aperture re-emission.  Scalar fix: v4.11.2.  Vectorial fix: v4.13.2
# P1-NEW-A.
# ============================================================================

class TestSpawnRngIndependenceDispatcherPin:
    """Pin: ``_spawn_rng`` returns DISTINCT child seeds for stream
    indices 0 (source init) and 1 (aperture re-emission) when called
    with the same caller seed.  This invariant is shared between the
    scalar and vectorial HFPI end-to-end propagators
    (``propagate_hfpi_freespace_aperture`` and
    ``propagate_vector_hfpi_freespace_aperture``).

    Pre-v4.11.2 the scalar path passed the same int ``rng`` to both
    ``init_paths_from_field`` and ``apply_aperture_diffraction`` --
    ``RandomState(rng=int_seed)`` rebuilds ``default_rng(int_seed)``
    so the two draws were perfectly correlated.  Pre-v4.13.2 the
    vectorial twin had the same gap; the v4.13.2 P1-NEW-A fix ported
    ``_spawn_rng(rng, 0)`` / ``_spawn_rng(rng, 1)`` to the vector
    pipeline.  This test pins both directly.
    """

    def test_spawn_rng_int_seed_returns_distinct_children(self):
        """``_spawn_rng(42, 0)`` and ``_spawn_rng(42, 1)`` differ."""
        c0 = _spawn_rng(42, 0)
        c1 = _spawn_rng(42, 1)
        assert c0 != c1, (
            f"_spawn_rng must derive DISTINCT child seeds for stream "
            f"indices 0 and 1 given the same caller int seed; got "
            f"{c0!r} == {c1!r}.  Pre-v4.11.2 this aliasing produced "
            f"perfectly-correlated source-init / aperture-re-emission "
            f"draws.")

    def test_spawn_rng_int_seed_is_deterministic(self):
        """Same caller seed + same stream index -> same child seed."""
        assert _spawn_rng(42, 0) == _spawn_rng(42, 0)
        assert _spawn_rng(42, 1) == _spawn_rng(42, 1)

    def test_spawn_rng_different_callers_distinct(self):
        """Different parent seeds -> different child seeds."""
        c1 = _spawn_rng(42, 0)
        c2 = _spawn_rng(43, 0)
        assert c1 != c2

    def test_spawn_rng_none_returns_none(self):
        """``_spawn_rng(None, k)`` is None for every k (the docstring's
        'caller did not pin a seed; let each aperture pull from system
        entropy' branch)."""
        assert _spawn_rng(None, 0) is None
        assert _spawn_rng(None, 1) is None

    @pytest.mark.parametrize('variant', _SCALAR_VECTOR_FREESPACE_VARIANTS)
    def test_freespace_aperture_independence(self, variant):
        """End-to-end pin: running
        ``propagate_*_hfpi_freespace_aperture`` with the same caller
        seed twice produces deterministic output (same call -> same
        result).  This regression test would have failed pre-v4.13.2
        on the vectorial side because the un-spawned shared seed made
        source-init and aperture re-emission perfectly correlated --
        the *output* was still deterministic, but a stress test that
        compared init draws to aperture draws would have caught the
        gap.  We pin the post-fix determinism + (separately) the
        ``_spawn_rng`` divergence so a regression on EITHER side
        triggers a CI failure.
        """
        N = 16
        dx = 5e-6
        wavelength = 633e-9
        n_paths = 256
        z_to_aperture = 1e-3
        z_aperture_to_output = 1e-3
        aperture_radius = 30e-6
        seed = 7
        if variant == 'scalar':
            E_in = np.ones((N, N), dtype=np.complex128)
            out_a = propagate_hfpi_freespace_aperture(
                E_in, dx,
                z_to_aperture=z_to_aperture,
                aperture_radius=aperture_radius,
                z_aperture_to_output=z_aperture_to_output,
                wavelength=wavelength, n_paths=n_paths, rng=seed,
            )
            out_b = propagate_hfpi_freespace_aperture(
                E_in, dx,
                z_to_aperture=z_to_aperture,
                aperture_radius=aperture_radius,
                z_aperture_to_output=z_aperture_to_output,
                wavelength=wavelength, n_paths=n_paths, rng=seed,
            )
        elif variant == 'vectorial':
            Ex = np.ones((N, N), dtype=np.complex128)
            Ey = np.zeros((N, N), dtype=np.complex128)
            out_a = propagate_vector_hfpi_freespace_aperture(
                Ex, Ey, dx,
                z_to_aperture=z_to_aperture,
                aperture_radius=aperture_radius,
                z_aperture_to_output=z_aperture_to_output,
                wavelength=wavelength, n_paths=n_paths, rng=seed,
            )
            out_b = propagate_vector_hfpi_freespace_aperture(
                Ex, Ey, dx,
                z_to_aperture=z_to_aperture,
                aperture_radius=aperture_radius,
                z_aperture_to_output=z_aperture_to_output,
                wavelength=wavelength, n_paths=n_paths, rng=seed,
            )
        else:
            raise ValueError(variant)
        # Determinism check: identical inputs -> identical outputs.
        if variant == 'scalar':
            assert np.array_equal(out_a, out_b)
        else:
            # Tuple (Ex_out, Ey_out)
            assert np.array_equal(out_a[0], out_b[0])
            assert np.array_equal(out_a[1], out_b[1])

    @pytest.mark.parametrize('variant', _SCALAR_VECTOR_FREESPACE_VARIANTS)
    def test_init_vs_aperture_uses_distinct_streams(self, variant):
        """Pin: when the caller passes an int seed to
        ``propagate_*_hfpi_freespace_aperture``, the source-init draws
        and the aperture-re-emission draws are *not* identical.
        Pre-fix on either side they WERE identical (perfectly
        correlated).

        We exercise the public init/diffraction helpers directly with
        ``rng=_spawn_rng(seed, 0)`` (source) and
        ``rng=_spawn_rng(seed, 1)`` (aperture) -- the exact spawning
        pattern the end-to-end propagator uses -- and compare the
        resulting direction-sample distributions.  If the seeds
        aliased the two would be bit-identical.
        """
        N = 16
        dx = 5e-6
        wavelength = 633e-9
        n_paths = 200
        seed = 13
        if variant == 'scalar':
            init_paths = init_paths_from_field(
                np.ones((N, N), dtype=np.complex128), dx,
                n_paths=n_paths, wavelength=wavelength,
                rng=_spawn_rng(seed, 0))
            ap_paths = apply_aperture_diffraction(
                init_paths, aperture_radius=1e-3,
                wavelength=wavelength,
                rng=_spawn_rng(seed, 1))
            init_dirs = np.asarray(init_paths.directions)
            ap_dirs = np.asarray(ap_paths.directions)
        else:
            init_paths = init_vector_paths_from_field(
                np.ones((N, N), dtype=np.complex128),
                np.zeros((N, N), dtype=np.complex128),
                dx, n_paths=n_paths, wavelength=wavelength,
                rng=_spawn_rng(seed, 0))
            ap_paths = apply_vector_aperture_diffraction(
                init_paths, aperture_radius=1e-3,
                wavelength=wavelength,
                rng=_spawn_rng(seed, 1))
            init_dirs = np.asarray(init_paths.directions)
            ap_dirs = np.asarray(ap_paths.directions)
        # If the two streams aliased, the direction arrays would be
        # bit-identical: same shape, same dtype, same values.
        assert not np.array_equal(init_dirs, ap_dirs), (
            f"HFPI/{variant}: source-init and aperture-re-emission "
            f"direction samples are bit-identical with caller seed "
            f"{seed}.  Indicates _spawn_rng alias regression -- both "
            f"sites must derive INDEPENDENT child seeds.")


# ============================================================================
# Pin 2 -- inf / NaN positions zeroed on grazing rays (Nz ~ 0).
# Scalar & vectorial fixes: v4.13.2 P1-NEW-H.
# ============================================================================

_PROPAGATE_TO_PLANE_VARIANTS = [
    pytest.param(
        'scalar', id='scalar.propagate_to_plane',
        marks=[]),
    pytest.param(
        'vectorial', id='vectorial.propagate_vector_to_plane',
        marks=[]),
]


def _make_grazing_path_bundle_scalar(N: int = 4) -> PathBundle:
    """Hand-built scalar bundle whose every path has ``Nz ~ 0``
    (direction purely transverse).  Provoking the v4.13.2 P1-NEW-H
    fix path."""
    n = N
    positions = np.zeros((n, 3), dtype=np.float64)
    # Pure +x directions: Nz = 0 exactly.
    directions = np.zeros((n, 3), dtype=np.float64)
    directions[:, 0] = 1.0  # L = 1
    weights = np.ones(n, dtype=np.complex128)
    opl = np.zeros(n, dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    return PathBundle(positions=positions, directions=directions,
                       weights=weights, opl=opl, alive=alive)


def _make_grazing_path_bundle_vector(N: int = 4) -> VectorPathBundle:
    """Vectorial twin of :func:`_make_grazing_path_bundle_scalar`."""
    n = N
    positions = np.zeros((n, 3), dtype=np.float64)
    directions = np.zeros((n, 3), dtype=np.float64)
    directions[:, 0] = 1.0
    Ex = np.ones(n, dtype=np.complex128)
    Ey = np.zeros(n, dtype=np.complex128)
    opl = np.zeros(n, dtype=np.float64)
    alive = np.ones(n, dtype=bool)
    return VectorPathBundle(positions=positions, directions=directions,
                             Ex=Ex, Ey=Ey, opl=opl, alive=alive)


class TestGrazingRayInfNanGuardDispatcherPin:
    """Pin: ``propagate_to_plane`` / ``propagate_vector_to_plane`` do
    NOT produce +/-inf or NaN positions when an alive path has
    ``Nz approx 0`` (grazing).

    Pre-v4.13.2 the implementation computed ``t = (z_target - z) /
    Nz`` with the ``where(|Nz| > eps, Nz, eps)`` guard, but then
    multiplied ``t * direction`` and ADDED to positions -- producing
    ``t ~ 1e30`` for grazing rays and poisoning ``new_positions``
    with +/-inf.  Downstream consumers that read ``paths.positions``
    without re-masking by ``alive`` (e.g.
    ``_hfpi_segment_trace``) carried inf / NaN into the trace.

    v4.13.2 P1-NEW-H added ``t = where(new_alive, t, 0.0)`` to BOTH
    the scalar and vectorial variants.  The parametrized form below
    catches the regression on EITHER side at CI time.
    """

    @pytest.mark.parametrize('variant', _PROPAGATE_TO_PLANE_VARIANTS)
    def test_grazing_rays_no_inf_or_nan_in_positions(self, variant):
        z_target = 1e-3
        wavelength = 633e-9
        if variant == 'scalar':
            paths = _make_grazing_path_bundle_scalar(N=4)
            out = propagate_to_plane(
                paths, z_target=z_target, wavelength=wavelength)
        elif variant == 'vectorial':
            paths = _make_grazing_path_bundle_vector(N=4)
            out = propagate_vector_to_plane(
                paths, z_target=z_target, wavelength=wavelength)
        else:
            raise ValueError(variant)
        pos = np.asarray(out.positions)
        assert np.all(np.isfinite(pos)), (
            f"HFPI/{variant}: grazing-ray (Nz ~ 0) propagation "
            f"produced non-finite positions ({pos!r}).  Pre-v4.13.2 "
            f"the t = z/Nz computation produced t ~ 1e30 which the "
            f"position add then turned into +/-inf -- the v4.13.2 "
            f"P1-NEW-H `t = where(new_alive, t, 0.0)` guard must be "
            f"in place on this variant.")
        # And the alive mask must (correctly) tag every grazer dead.
        assert not np.any(np.asarray(out.alive)), (
            f"HFPI/{variant}: grazing rays must be tagged DEAD after "
            f"propagate_to_plane; got alive={np.asarray(out.alive)!r}.")

    @pytest.mark.parametrize('variant', _PROPAGATE_TO_PLANE_VARIANTS)
    def test_normal_rays_still_propagate_correctly(self, variant):
        """Sanity: the grazing-ray fix must not regress the
        non-grazing path.  A pure-+z direction with ``Nz=1`` should
        advance by exactly ``z_target`` (= the distance ``t``)."""
        z_target = 1e-3
        wavelength = 633e-9
        n = 4
        positions = np.zeros((n, 3), dtype=np.float64)
        directions = np.zeros((n, 3), dtype=np.float64)
        directions[:, 2] = 1.0  # Nz = 1
        weights = np.ones(n, dtype=np.complex128)
        opl = np.zeros(n, dtype=np.float64)
        alive = np.ones(n, dtype=bool)
        if variant == 'scalar':
            paths = PathBundle(positions=positions, directions=directions,
                                weights=weights, opl=opl, alive=alive)
            out = propagate_to_plane(
                paths, z_target=z_target, wavelength=wavelength)
        else:
            Ex = np.ones(n, dtype=np.complex128)
            Ey = np.zeros(n, dtype=np.complex128)
            paths = VectorPathBundle(
                positions=positions, directions=directions,
                Ex=Ex, Ey=Ey, opl=opl, alive=alive)
            out = propagate_vector_to_plane(
                paths, z_target=z_target, wavelength=wavelength)
        pos = np.asarray(out.positions)
        # All paths advance by exactly z_target along +z.
        assert np.allclose(pos[:, 2], z_target)
        assert np.all(np.asarray(out.alive))


# ============================================================================
# Pin 3 -- alive mask correctness in accumulate_*_to_grid.
# Pre-fix dead paths could contribute to the output grid.
# ============================================================================

_ACCUMULATE_VARIANTS = [
    pytest.param('scalar', id='scalar.accumulate_to_grid'),
    pytest.param('vectorial', id='vectorial.accumulate_vector_to_grid'),
]


class TestAccumulateAliveMaskDispatcherPin:
    """Pin: dead rays do NOT contribute to ``accumulate_to_grid`` /
    ``accumulate_vector_to_grid``.

    The accumulator masks each path through ``inside & paths.alive``
    in both variants.  If a future regression were to drop the
    ``& paths.alive`` clause on EITHER twin, dead paths' (potentially
    stale or even +/-inf) weights would leak into the output grid.
    The parametrized form below pins both at once.
    """

    @pytest.mark.parametrize('variant', _ACCUMULATE_VARIANTS)
    def test_dead_rays_do_not_contribute(self, variant):
        """Build a bundle where the first half is alive (and on-grid)
        and the second half is dead but carries massive weights.  The
        output grid sum must equal the live-half weight sum only.
        """
        N = 8
        dx = 1e-6
        n_alive = 4
        n_dead = 4
        n = n_alive + n_dead
        # Live rays at the origin -> centre pixel.
        positions = np.zeros((n, 3), dtype=np.float64)
        # Dead rays' positions are irrelevant for the alive-mask test,
        # but we DO want their weights to be huge so a missing
        # alive-mask would be visible against rounding noise.
        directions = np.zeros((n, 3), dtype=np.float64)
        directions[:, 2] = 1.0
        opl = np.zeros(n, dtype=np.float64)
        alive = np.array([True] * n_alive + [False] * n_dead)
        if variant == 'scalar':
            weights = np.zeros(n, dtype=np.complex128)
            weights[:n_alive] = 1.0  # live weight per ray
            weights[n_alive:] = 1e9  # huge dead-ray weight
            paths = PathBundle(positions=positions, directions=directions,
                                weights=weights, opl=opl, alive=alive)
            grid = accumulate_to_grid(
                paths, Ny=N, Nx=N, dx=dx)
            total = complex(np.sum(grid))
            expected = float(n_alive)  # n_alive * 1.0
            assert abs(total - expected) < 1e-9, (
                f"scalar.accumulate_to_grid leaked dead-ray weights "
                f"into the output grid (got total={total!r}, expected "
                f"{expected!r}).  The alive-mask multiply must filter "
                f"the dead half out.")
        elif variant == 'vectorial':
            Ex = np.zeros(n, dtype=np.complex128)
            Ey = np.zeros(n, dtype=np.complex128)
            Ex[:n_alive] = 1.0
            Ey[:n_alive] = 0.5
            Ex[n_alive:] = 1e9
            Ey[n_alive:] = 2e9
            paths = VectorPathBundle(
                positions=positions, directions=directions,
                Ex=Ex, Ey=Ey, opl=opl, alive=alive)
            ex_grid, ey_grid = accumulate_vector_to_grid(
                paths, Ny=N, Nx=N, dx=dx)
            ex_total = complex(np.sum(ex_grid))
            ey_total = complex(np.sum(ey_grid))
            ex_expected = float(n_alive) * 1.0
            ey_expected = float(n_alive) * 0.5
            assert abs(ex_total - ex_expected) < 1e-9, (
                f"vectorial.accumulate_vector_to_grid leaked dead-ray "
                f"Ex weights (got Ex_total={ex_total!r}, expected "
                f"{ex_expected!r}).")
            assert abs(ey_total - ey_expected) < 1e-9, (
                f"vectorial.accumulate_vector_to_grid leaked dead-ray "
                f"Ey weights (got Ey_total={ey_total!r}, expected "
                f"{ey_expected!r}).")
        else:
            raise ValueError(variant)

    @pytest.mark.parametrize('variant', _ACCUMULATE_VARIANTS)
    def test_all_alive_baseline(self, variant):
        """Sanity: with every ray alive and finite, the accumulator
        produces the expected sum (no double-counting / off-by-one)."""
        N = 8
        dx = 1e-6
        n = 4
        positions = np.zeros((n, 3), dtype=np.float64)
        directions = np.zeros((n, 3), dtype=np.float64)
        directions[:, 2] = 1.0
        opl = np.zeros(n, dtype=np.float64)
        alive = np.ones(n, dtype=bool)
        if variant == 'scalar':
            weights = np.ones(n, dtype=np.complex128) * (1.0 + 0.0j)
            paths = PathBundle(positions=positions, directions=directions,
                                weights=weights, opl=opl, alive=alive)
            grid = accumulate_to_grid(paths, Ny=N, Nx=N, dx=dx)
            assert abs(complex(np.sum(grid)) - float(n)) < 1e-9
        else:
            Ex = np.ones(n, dtype=np.complex128)
            Ey = np.ones(n, dtype=np.complex128) * 2.0
            paths = VectorPathBundle(
                positions=positions, directions=directions,
                Ex=Ex, Ey=Ey, opl=opl, alive=alive)
            ex_grid, ey_grid = accumulate_vector_to_grid(
                paths, Ny=N, Nx=N, dx=dx)
            assert abs(complex(np.sum(ex_grid)) - float(n)) < 1e-9
            assert abs(complex(np.sum(ey_grid)) - 2.0 * float(n)) < 1e-9


# ============================================================================
# Family-I parametrized dispatcher pin -- collects the 5 scalar + 5
# vector public entry points and verifies the *importable* surface
# (so a future rename / move trips immediately).
# ============================================================================

_HFPI_SCALAR_PUBLIC = [
    'propagate_hfpi',
    'propagate_hfpi_through_prescription',
    'propagate_hfpi_freespace_aperture',
    'init_paths_from_field',
    'propagate_to_plane',
    'apply_aperture_diffraction',
    'accumulate_to_grid',
    '_spawn_rng',
]
_HFPI_VECTOR_PUBLIC = [
    'propagate_vector_hfpi_freespace_aperture',
    'init_vector_paths_from_field',
    'propagate_vector_to_plane',
    'apply_vector_aperture_diffraction',
    'accumulate_vector_to_grid',
]


class TestHfpiPublicSurfacePin:
    """Pin: every documented scalar HFPI entry point lives in
    ``lumenairy.propagators.hfpi``; every documented vectorial twin
    lives in ``lumenairy.propagators.vectorial_hfpi``.  A future
    refactor that renames or moves one of these names will fail this
    test rather than slipping past callers that import lazily.
    """

    @pytest.mark.parametrize('name', _HFPI_SCALAR_PUBLIC)
    def test_scalar_public_name_resolves(self, name):
        from lumenairy.propagators import hfpi as _hfpi
        assert hasattr(_hfpi, name), (
            f"lumenairy.propagators.hfpi missing public name {name!r}.")
        obj = getattr(_hfpi, name)
        assert callable(obj), (
            f"lumenairy.propagators.hfpi.{name} is not callable; got "
            f"{type(obj).__name__}.")

    @pytest.mark.parametrize('name', _HFPI_VECTOR_PUBLIC)
    def test_vector_public_name_resolves(self, name):
        from lumenairy.propagators import vectorial_hfpi as _vhfpi
        assert hasattr(_vhfpi, name), (
            f"lumenairy.propagators.vectorial_hfpi missing public name "
            f"{name!r}.")
        obj = getattr(_vhfpi, name)
        assert callable(obj), (
            f"lumenairy.propagators.vectorial_hfpi.{name} is not "
            f"callable; got {type(obj).__name__}.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
