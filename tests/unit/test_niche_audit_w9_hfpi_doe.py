"""W9 follow-up -- HFPI's ``surface_diffraction`` diagnosis (audit W9-14).

The W9 dispatcher audit reported that HFPI "missed the analytic thin-grating
order-1 deflection by 85-97%".  That report was WRONG, and this file records
both halves of why:

* the DOE kick itself is EXACT -- ``raytrace.trace(surface_diffraction=...)``
  puts the exit direction cosine at ``m*lambda/Lambda`` to machine precision;
* the HFPI-level miss was a starved Monte-Carlo estimator, returning its own
  sampling envelope with no diagnostic, which is the real (and now fixed)
  defect.

Everything here is analytic-oracle based (no reference data files) and every
grating sits far from any cutoff (``s = m*lambda/Lambda <= 0.064``, i.e. two
orders of magnitude below the ``s -> 1`` evanescent edge).
"""
import warnings

import numpy as np
import pytest

from lumenairy.propagators.hfpi import (
    PathBundle,
    accumulate_to_grid,
    propagate_hfpi_freespace_aperture,
)
from lumenairy.raytrace import RayBundle, surfaces_from_prescription, trace

# NOTE: the guard's threshold constant is imported INSIDE the one test that
# pins it, not at module scope, so that at a pre-fix baseline the ray-level
# regression fences below still collect and PASS (they always did -- the kick
# was never the defect) while only the guard pins fail.

WL = 633e-9
T = 2e-3


def _air_plate(t=T, ap=1e-3):
    """Two flat AIR-to-AIR surfaces ``t`` apart: a pure carrier for a grating
    kick, so the analytic exit offset is exactly ``t*tan(asin(m*lam/Lambda))``
    with no glass refraction to model."""
    flat = dict(radius=np.inf, conic=0.0, aspheric_coeffs=None,
                radius_y=None, conic_y=None, aspheric_coeffs_y=None,
                glass_before='air', glass_after='air')
    return {'name': 'grating', 'aperture_diameter': ap, 'thicknesses': [t],
            'surfaces': [dict(flat), dict(flat)]}


# ===========================================================================
# 1. The DOE kick is EXACT at the ray level -- hypotheses (a) and (c) refuted.
# ===========================================================================
#
# (a) "wrong magnitude / units (radians vs sin, period in the wrong units)" and
# (c) "the kwarg plumbing drops or half-applies the spec" would both show up
# here.  Neither does.  This is also the code path
# ``fit_canonical_polynomials(surface_diffraction=...)`` uses, so it covers the
# sibling consumer too.

_GRATINGS = [
    (80e-6, 1),
    (40e-6, 1),
    (20e-6, 1),
    (20e-6, 2),
    (10e-6, 1),
]


@pytest.mark.parametrize('period,order', _GRATINGS)
def test_ray_level_doe_kick_is_exact(period, order):
    """MEASURED: relative error 0.0 - 2.2e-16 on the exit direction cosine.

    Budget 1e-12 -- four orders of magnitude above the measured FP-floor error,
    which is the right bar for an exact closed-form kick (it is a single add on
    a direction cosine, not an iterative solve).
    """
    surfs = surfaces_from_prescription(_air_plate())
    rb = RayBundle(x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
                   L=np.array([0.0]), M=np.array([0.0]), N=np.array([1.0]),
                   wavelength=WL, opd=np.array([0.0]),
                   alive=np.array([True]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = trace(rb, list(surfs), WL, output_filter='last',
                    surface_diffraction={0: (order, 0, period, period)})
    out = res.image_rays
    want_L = order * WL / period
    assert abs(float(out.L[0]) - want_L) / want_L < 1e-12, (
        f'direction cosine: want {want_L!r}, got {float(out.L[0])!r}')
    want_x = T * np.tan(np.arcsin(want_L))
    assert abs(float(out.x[0]) - want_x) / want_x < 1e-9, (
        f'exit offset: want {want_x!r}, got {float(out.x[0])!r}')


def test_the_kick_depends_only_on_m_over_Lambda():
    """Physics fence: ``(Lambda=20 um, m=2)`` and ``(Lambda=10 um, m=1)`` are
    the same grating equation and must give the identical ray."""
    surfs = surfaces_from_prescription(_air_plate())
    outs = []
    for period, order in ((20e-6, 2), (10e-6, 1)):
        rb = RayBundle(x=np.array([0.0]), y=np.array([0.0]),
                       z=np.array([0.0]), L=np.array([0.0]),
                       M=np.array([0.0]), N=np.array([1.0]), wavelength=WL,
                       opd=np.array([0.0]), alive=np.array([True]))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = trace(rb, list(surfs), WL, output_filter='last',
                      surface_diffraction={0: (order, 0, period, period)})
        outs.append((float(r.image_rays.L[0]), float(r.image_rays.x[0])))
    assert outs[0] == pytest.approx(outs[1], rel=1e-12)


def test_the_y_order_is_applied_too_not_only_m_x():
    """Hypothesis (c) sub-case: "only m_x used".  A pure y-order must deflect
    in y and leave x alone."""
    surfs = surfaces_from_prescription(_air_plate())
    rb = RayBundle(x=np.array([0.0]), y=np.array([0.0]), z=np.array([0.0]),
                   L=np.array([0.0]), M=np.array([0.0]), N=np.array([1.0]),
                   wavelength=WL, opd=np.array([0.0]),
                   alive=np.array([True]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = trace(rb, list(surfs), WL, output_filter='last',
                  surface_diffraction={0: (0, 1, 20e-6, 20e-6)})
    want = WL / 20e-6
    assert abs(float(r.image_rays.M[0]) - want) / want < 1e-12
    assert abs(float(r.image_rays.L[0])) < 1e-15


# ===========================================================================
# 2. The sampling-adequacy guard (the real defect).
# ===========================================================================
#
# MEASURED at 30ac116, thin air plate, 128^2 output, DEFAULT cone (~90 deg, a
# full forward hemisphere) while the grid subtends 7.3 deg:
#
#   n_paths   occupancy (pixels ever hit)   seed-to-seed intensity-shape fidelity
#    20000    0.0061                        0.0000
#    80000    0.0209                        0.0054
#   320000    0.0792                        0.0210
#
# i.e. 93-99% of output pixels are EXACTLY ZERO and two seeds of the same
# physics do not agree at all.  Nothing warned.  The docstring's guarantee that
# "fringe positions and interference contrast (phase structure) are correct"
# does not hold in that regime, and it is the regime a caller lands in by
# default.

_STARVED = dict(z_to_aperture=50e-6, aperture_radius=20e-6,
                z_aperture_to_output=150e-6, wavelength=0.5e-6,
                n_paths=20000, rng=42)


def test_a_starved_run_warns():
    E = np.ones((32, 32), dtype=np.complex128)
    with pytest.warns(RuntimeWarning, match='UNDER-SAMPLED'):
        propagate_hfpi_freespace_aperture(E, 1e-6, **_STARVED)


def test_the_warning_names_the_numbers_and_both_levers():
    """A diagnostic that does not say what to change is not a diagnostic."""
    E = np.ones((32, 32), dtype=np.complex128)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        propagate_hfpi_freespace_aperture(E, 1e-6, **_STARVED)
    msgs = [str(r.message) for r in rec
            if issubclass(r.category, RuntimeWarning)]
    assert msgs, 'no RuntimeWarning raised'
    m = msgs[0]
    assert 'landed' in m and 'per output pixel' in m
    assert 'n_paths' in m and 'cone_half_angle' in m


@pytest.mark.parametrize('policy', ['silent', 'error'])
def test_the_policy_knob(policy):
    E = np.ones((32, 32), dtype=np.complex128)
    kw = dict(_STARVED, on_undersampled=policy)
    if policy == 'silent':
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            propagate_hfpi_freespace_aperture(E, 1e-6, **kw)
        assert not [r for r in rec
                    if issubclass(r.category, RuntimeWarning)
                    and 'UNDER-SAMPLED' in str(r.message)]
    else:
        with pytest.raises(ValueError, match='UNDER-SAMPLED'):
            propagate_hfpi_freespace_aperture(E, 1e-6, **kw)


def test_the_policy_is_validated():
    E = np.ones((8, 8), dtype=np.complex128)
    with pytest.raises(ValueError, match='on_undersampled'):
        propagate_hfpi_freespace_aperture(
            E, 1e-6, **dict(_STARVED, n_paths=100, on_undersampled='shout'))


def _bundle(n, spread):
    """``n`` synthetic paths spread over ``+-spread`` metres in x."""
    x = np.linspace(-spread, spread, n)
    pos = np.stack([x, np.zeros(n), np.zeros(n)], axis=-1)
    dirs = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    return PathBundle(positions=pos, directions=dirs,
                      weights=np.ones(n, dtype=np.complex128),
                      opl=np.zeros(n), alive=np.ones(n, dtype=bool))


def test_the_threshold_is_landed_paths_per_pixel_not_paths_issued():
    """The guard must count what LANDED, not what was issued -- the whole
    failure mode is paths thrown where they cannot land."""
    N, dx = 8, 1e-6
    n_px = N * N
    # every path lands, comfortably above the bar
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        accumulate_to_grid(_bundle(4 * n_px, 3e-6), Ny=N, Nx=N, dx=dx)
    assert not [r for r in rec if 'UNDER-SAMPLED' in str(r.message)]
    # same path COUNT, but thrown far outside the grid -> almost none land
    with pytest.warns(RuntimeWarning, match='UNDER-SAMPLED'):
        accumulate_to_grid(_bundle(4 * n_px, 1.0), Ny=N, Nx=N, dx=dx)


def test_a_well_sampled_run_is_silent():
    """Fence: the guard must not cry wolf on an adequately-sampled call."""
    N, dx = 8, 1e-6
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        accumulate_to_grid(_bundle(64 * N * N, 3e-6), Ny=N, Nx=N, dx=dx)
    assert not [r for r in rec if 'UNDER-SAMPLED' in str(r.message)]


def test_the_threshold_constant_is_one_path_per_pixel():
    """Pinned so the bar cannot drift silently.  One per pixel is the
    UNAMBIGUOUS floor (below it most pixels are never touched), deliberately
    not the point where the answer becomes trustworthy -- measured, the same
    probe still only reached seed-to-seed fidelity 0.44 at ~12 per pixel."""
    from lumenairy.propagators.hfpi import (
        _MIN_LANDED_PATHS_PER_OUTPUT_PIXEL,
    )
    assert _MIN_LANDED_PATHS_PER_OUTPUT_PIXEL == 1.0


# ===========================================================================
# 3. The HFPI-level deflection is CONVERGENCE, not physics.
# ===========================================================================
#
# MEASURED at fixed geometry (source 16x16 @ 5 um, output 32x32 @ 10 um,
# cone_half_angle=0.10, Lambda=20 um, m=1, analytic 63.33 um), varying ONLY
# n_paths -- seed-averaged intensity centroid minus the no-grating control:
#
#   n_paths     measured    ratio to analytic
#    100000     25.40 um    0.401
#    400000     33.53 um    0.530
#   1600000     47.02 um    0.742
#   6400000     54.51 um    0.861
#
# A units or plumbing error would be n_paths-FLAT.  A monotone climb toward 1
# is an unconverged estimator, which is what it is.  The full sweep is too slow
# for CI; the pin below reproduces the two cheapest points and asserts only the
# ORDERING plus the direction, which is the load-bearing claim.


@pytest.mark.slow
def test_the_hfpi_deflection_improves_with_n_paths():
    """CI-cheap end of the convergence sweep: more paths must move the measured
    deflection TOWARD the analytic value, never away."""
    n_in, dx_in, n_out, dx_out, cone = 16, 5e-6, 32, 10e-6, 0.10
    period, order = 20e-6, 1
    want = T * np.tan(np.arcsin(order * WL / period))
    x = (np.arange(n_in) - n_in / 2 + 0.5) * dx_in
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X ** 2 + Y ** 2) / (20e-6) ** 2).astype(np.complex128)
    xo = (np.arange(n_out) - n_out // 2) * dx_out
    from lumenairy.propagators.hfpi import propagate_hfpi_through_prescription

    def centroid(sd, n_paths, seeds=3):
        acc = np.zeros((n_out, n_out))
        for s in range(seeds):
            kw = {'surface_diffraction': sd} if sd else {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                o = propagate_hfpi_through_prescription(
                    E, dx_in, _air_plate(), wavelength=WL, n_paths=n_paths,
                    rng=5000 + s, diffracting_surfaces=[],
                    cone_half_angle=cone, output_shape=(n_out, n_out),
                    output_dx=dx_out, on_undersampled='silent', **kw)
            acc += np.abs(np.asarray(o)) ** 2
        p = acc.sum(axis=0)
        return float((p * xo).sum() / p.sum())

    sd = {0: (order, 0, period, period)}
    ratios = []
    for n_paths in (100000, 400000):
        ratios.append((centroid(sd, n_paths) - centroid(None, n_paths)) / want)
    assert 0.0 < ratios[0] < 1.0, ratios
    assert ratios[1] > ratios[0], (
        f'more paths must move the deflection toward the analytic value; '
        f'got {ratios!r}')
