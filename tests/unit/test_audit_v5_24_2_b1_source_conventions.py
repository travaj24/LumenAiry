"""Audit v5.24.2 B1 / S3-16 -- source-factory convention normalization.

Convention pins (POST-v5.30 REMOVAL) for the three convention fixes in
``lumenairy/sources/core.py``.  This header used to describe the v5.25
deprecation-warning phase; v5.30 hard-removed both deprecated spellings, and
the assertions below were updated to that -- they now assert ``TypeError``,
not a ``DeprecationWarning``.  (The in-file comment block further down
already recorded the supersession; only this header was stale.)

1. ``rng=`` on the Schell-family factories (CONVENTIONS.md section 3),
   which REPLACED ``seed=``.  ``seed=`` was REMOVED in v5.30, so the old
   spelling must now raise ``TypeError``; ``rng=<int>`` stays deterministic
   and reproduces its stream bit-for-bit.
2. Explicit ``normalize=`` on ``create_top_hat_beam`` /
   ``create_annular_beam`` / ``create_bessel_beam`` whose DEFAULT
   reproduces the historical hard-coded convention exactly (top-hat /
   annular = unit power; Bessel = raw).
3. ``w0=`` (the 1/e^2 intensity radius, canonical) on
   ``create_gaussian_beam``, which REPLACED ``sigma=`` (field std-dev,
   ``w0 = sigma * sqrt(2)``); ``sigma=`` was REMOVED in v5.30.

Oracles are independent (analytic waist / power definitions), not the
code's own output.

Author: audit remediation -- v5.25 / B1
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.sources.core import (
    create_annular_beam,
    create_annular_incoherent_source,
    create_bessel_beam,
    create_gaussian_beam,
    create_gaussian_schell_source,
    create_schell_model_source,
    create_top_hat_beam,
)

# ---------------------------------------------------------------------------
# 1. rng / seed migration on the Schell family
# ---------------------------------------------------------------------------

_SCHELL_COMMON = dict(N=32, dx=5e-6, wavelength=633e-9)


def _gaussian_schell(**kw):
    return create_gaussian_schell_source(
        w0=40e-6, sigma_g=20e-6, n_realizations=4, **_SCHELL_COMMON, **kw)


def _schell_model(**kw):
    return create_schell_model_source(
        intensity_profile=np.ones((32, 32)), coherence_length=15e-6,
        n_realizations=4, **_SCHELL_COMMON, **kw)


def _annular_inco(**kw):
    return create_annular_incoherent_source(
        inner_radius=40e-6, outer_radius=80e-6, n_realizations=4,
        **_SCHELL_COMMON, **kw)


_SCHELL_FACTORIES = [
    pytest.param(_gaussian_schell, 'create_gaussian_schell_source',
                 id='gaussian_schell'),
    pytest.param(_schell_model, 'create_schell_model_source',
                 id='schell_model'),
    pytest.param(_annular_inco, 'create_annular_incoherent_source',
                 id='annular_incoherent'),
]


# v5.30 (W5 shim-removal wave) SUPERSESSION of three v5.25 pins:
#
#   * ``test_rng_int_matches_deprecated_seed_bit_for_bit`` -> the
#     bit-for-bit equivalence of ``seed=<int>`` and ``rng=<int>`` was the
#     evidence that the rename is a lossless migration.  That evidence is
#     what licensed the removal; there is no longer a second spelling to
#     compare against.  Superseded by
#     ``test_deprecated_seed_kwarg_is_removed`` (old form raises) plus the
#     bit-identity capture in
#     ``tests/unit/test_niche_audit_w5_shim_removals.py`` (modern form
#     unchanged).
#   * ``test_seed_still_works_and_warns`` -> inverted below: the kwarg is
#     gone, so it must RAISE, not warn.
#   * ``test_rng_and_seed_are_mutually_exclusive`` -> vacuous once ``seed``
#     is not a parameter (the TypeError it asserted now fires for a
#     different reason -- unexpected keyword -- which the supersession
#     below pins explicitly).


@pytest.mark.parametrize('factory, fn_name', _SCHELL_FACTORIES)
def test_deprecated_seed_kwarg_is_removed(factory, fn_name):
    """``seed=`` was deprecated in v5.25 (stated horizon v5.27, shipped
    unremoved through v5.29) and is REMOVED in v5.30.

    Contract: a plain ``TypeError`` from the signature, naming the kwarg.
    Precedent for the bare signature removal on a kwarg RENAME (as
    opposed to a value-intercepting shim):
    ``analysis/detector.py``'s v5.0 ``cosmic_ray_rate`` retirement and
    ``optimize/multiconfig.py``'s v5.0 ``wavelength``-default removal.
    """
    with pytest.raises(TypeError, match='seed'):
        factory(seed=7)


@pytest.mark.parametrize('factory, fn_name', _SCHELL_FACTORIES)
def test_rng_int_path_is_the_only_seeded_spelling(factory, fn_name):
    """The surviving ``rng=<int>`` path is deterministic and silent --
    the property the removed ``seed=`` spelling existed to provide."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        a, *_ = factory(rng=7)
        b, *_ = factory(rng=7)
    assert np.array_equal(a, b), (
        f"{fn_name}: rng=7 must be reproducible bit-for-bit")


@pytest.mark.parametrize('factory, fn_name', _SCHELL_FACTORIES)
def test_canonical_rng_form_emits_no_warning(factory, fn_name):
    """The canonical ``rng=`` form must be warning-clean."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        factory(rng=5)  # must not raise a (deprecation) warning-as-error


@pytest.mark.parametrize('factory, fn_name', _SCHELL_FACTORIES)
def test_rng_accepts_generator_and_none(factory, fn_name):
    g = np.random.default_rng(11)
    ens_g, *_ = factory(rng=g)
    assert ens_g.shape[0] == 4
    ens_none, *_ = factory(rng=None)  # fresh generator, just must run
    assert ens_none.shape[0] == 4


def test_rng_accepts_numpy_backed_randomstate():
    """A NumPy-backed lumenairy ``RandomState`` is accepted and, seeded
    identically, reproduces the ``rng=<int>`` stream."""
    ens_rs, *_ = _gaussian_schell(rng=la.RandomState(9))
    ens_int, *_ = _gaussian_schell(rng=9)
    assert np.array_equal(ens_rs, ens_int), (
        "a NumPy RandomState(9) must match rng=9 bit-for-bit")


def test_rng_rejects_unknown_type():
    with pytest.raises(TypeError):
        _gaussian_schell(rng='not-a-generator')


def test_source_classmethods_accept_rng_and_reject_seed():
    """SUPERSEDES ``test_source_classmethods_accept_rng_and_deprecate_seed``
    (v5.25): the classmethods forward ``rng`` and no longer accept
    ``seed`` at all (v5.30 removal).

    Note the classmethods take ``**factory_kwargs``, so a stray ``seed=``
    is rejected one frame in, by the top-level factory -- the message
    still names the offending keyword, which is what a migrating caller
    needs."""
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        ens_rng, *_ = la.Source.gaussian_schell(
            N=32, dx=5e-6, wavelength=633e-9, w0=40e-6, sigma_g=20e-6,
            n_realizations=4, rng=2)
    assert ens_rng.shape[0] == 4
    with pytest.raises(TypeError, match='seed'):
        la.Source.gaussian_schell(
            N=32, dx=5e-6, wavelength=633e-9, w0=40e-6, sigma_g=20e-6,
            n_realizations=4, seed=2)


# ---------------------------------------------------------------------------
# 2. create_gaussian_beam: w0 (canonical) vs deprecated sigma
# ---------------------------------------------------------------------------

def test_w0_is_the_1_over_e2_intensity_radius():
    """Independent physical oracle: with ``w0`` the beam waist, the
    intensity at r == w0 is exp(-2) of the on-axis peak."""
    w0 = 10e-6
    E, x, y = create_gaussian_beam(256, 0.2e-6, 633e-9, w0=w0,
                                   normalize='none')
    cy = E.shape[0] // 2
    I = np.abs(E) ** 2
    i0 = int(np.argmin(np.abs(x)))
    iw = int(np.argmin(np.abs(x - w0)))
    ratio = I[cy, iw] / I[cy, i0]
    assert np.isclose(ratio, np.exp(-2.0), rtol=2e-3), (
        f"I(w0)/I(0) = {ratio:.6f} should equal exp(-2) = {np.exp(-2):.6f}")


def test_w0_is_the_sqrt2_scaled_field_std_dev():
    """SUPERSEDES ``test_w0_equals_sigma_over_sqrt2_bit_for_bit``
    (v5.25), which compared the ``w0`` and (now removed) ``sigma``
    spellings.  The physical relation it was really pinning --
    ``w0 = sigma * sqrt(2)``, i.e. the field kernel is
    ``exp(-r^2 / w0^2)`` -- is asserted directly against an independent
    closed-form oracle instead, so it survives the removal.

    The oracle is the ``w0``-form kernel written out longhand; it agrees
    with the library's ``exp(-r^2/(2 sigma^2))``, ``sigma = w0/sqrt(2)``
    arithmetic to round-off (the two groupings are algebraically but not
    bit-wise identical), so this is a tight-tolerance -- not bit-exact --
    physics pin.  The bit-exactness claim lives in
    ``tests/unit/test_niche_audit_w5_shim_removals.py``, which compares the
    modern path against a captured pre-removal baseline."""
    w0 = 12e-6
    E, x, y = create_gaussian_beam(64, 1e-6, 633e-9, w0=w0,
                                   normalize='none')
    X, Y = np.meshgrid(x, y)
    ref = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(E.dtype)
    assert np.allclose(E, ref, rtol=0, atol=8 * np.finfo(np.float64).eps), (
        f'create_gaussian_beam(w0=w) must equal exp(-r^2/w^2); '
        f'max|delta|={np.max(np.abs(E - ref)):.3e}')
    # ... and exactly reproduce the internal sigma grouping.
    sigma = w0 / np.sqrt(2.0)
    ref_internal = np.exp(
        -(X ** 2 + Y ** 2) / (2 * sigma ** 2)).astype(E.dtype)
    assert np.array_equal(E, ref_internal), (
        'w0 must resolve to sigma = w0/sqrt(2) bit-for-bit')


def test_deprecated_sigma_kwarg_is_removed():
    """SUPERSEDES ``test_sigma_still_works_and_warns`` (v5.25).

    ``sigma=`` was deprecated in v5.25 (stated horizon v5.27, shipped
    unremoved through v5.29) and is REMOVED in v5.30.  Contract: a plain
    ``TypeError`` from the signature.  Migration: ``sigma=s`` ->
    ``w0=s*sqrt(2)``.
    """
    with pytest.raises(TypeError, match='sigma'):
        create_gaussian_beam(32, 1e-6, 633e-9, sigma=5e-6)


def test_missing_width_arg_names_the_sigma_migration():
    """The one place the removed name must still be spoken: the
    missing-argument error tells a ``sigma=`` caller what to do."""
    with pytest.raises(TypeError) as info:
        create_gaussian_beam(32, 1e-6, 633e-9)
    msg = str(info.value)
    assert 'w0' in msg and 'sigma' in msg and 'sqrt(2)' in msg, msg


def test_w0_form_is_warning_clean():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        create_gaussian_beam(32, 1e-6, 633e-9, w0=7e-6)


def test_sigma_and_w0_mutual_exclusion_pin_is_SUPERSEDED():
    """SUPERSEDES ``test_sigma_and_w0_mutually_exclusive`` (v5.25).

    With ``sigma`` gone there is nothing to be mutually exclusive WITH,
    so the ``ValueError`` the old pin asserted is unreachable; the
    over-specified call is now rejected earlier, by the signature.
    """
    with pytest.raises(TypeError, match='sigma'):
        create_gaussian_beam(32, 1e-6, 633e-9, sigma=5e-6, w0=7e-6)


def test_neither_width_arg_raises():
    with pytest.raises(TypeError):
        create_gaussian_beam(32, 1e-6, 633e-9)


def test_source_gaussian_classmethod_stays_warning_clean():
    """Source.gaussian (canonical w0) must not trip the sigma
    deprecation internally."""
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        src = la.Source.gaussian(N=32, dx=5e-6, wavelength=633e-9, w0=20e-6)
    assert src.shape == (32, 32)


# ---------------------------------------------------------------------------
# 3. explicit normalize= on the aperture factories (default preserved)
# ---------------------------------------------------------------------------

def test_top_hat_default_is_unit_power():
    """Independent oracle: the default (native 'power') field integrates
    to unit power."""
    E, _, _ = create_top_hat_beam(64, 1e-6, 633e-9, diameter=30e-6)
    power = float(np.sum(np.abs(E) ** 2) * 1e-6 * 1e-6)
    assert np.isclose(power, 1.0, rtol=1e-12), (
        f"top-hat default must be unit power; got {power}")
    # explicit 'power' must be byte-identical to the default
    Ep, _, _ = create_top_hat_beam(64, 1e-6, 633e-9, diameter=30e-6,
                                   normalize='power')
    assert np.array_equal(E, Ep)


def test_top_hat_none_is_raw_indicator():
    E, _, _ = create_top_hat_beam(64, 1e-6, 633e-9, diameter=30e-6,
                                  normalize='none')
    vals = set(np.unique(np.abs(E)))
    assert vals <= {0.0, 1.0}, "normalize='none' must be a raw 0/1 indicator"
    assert float(np.abs(E).max()) == 1.0


def test_top_hat_peak_is_unit_peak():
    E, _, _ = create_top_hat_beam(64, 1e-6, 633e-9, diameter=30e-6,
                                  normalize='peak')
    assert np.isclose(float(np.abs(E).max()), 1.0)


def test_annular_default_is_unit_power():
    E, _, _ = create_annular_beam(64, 1e-6, 633e-9, outer_diameter=40e-6,
                                  inner_diameter=20e-6)
    power = float(np.sum(np.abs(E) ** 2) * 1e-6 * 1e-6)
    assert np.isclose(power, 1.0, rtol=1e-12)
    Ep, _, _ = create_annular_beam(64, 1e-6, 633e-9, outer_diameter=40e-6,
                                   inner_diameter=20e-6, normalize='power')
    assert np.array_equal(E, Ep)


def test_bessel_default_is_raw():
    """Native Bessel convention is 'none': the raw J_0 field with
    on-axis peak J_0(0) == 1 (NOT window-power-normalised)."""
    E, x, y = create_bessel_beam(128, 1e-6, 633e-9, 0.05)
    cy, cx = E.shape[0] // 2, E.shape[1] // 2
    assert np.isclose(float(np.abs(E[cy, cx])), 1.0, atol=1e-9), (
        "raw Bessel must have unit on-axis amplitude J_0(0)=1")
    power = float(np.sum(np.abs(E) ** 2) * 1e-6 * 1e-6)
    assert not np.isclose(power, 1.0), (
        "the default (raw) Bessel field must NOT be power-normalised")
    # explicit 'none' is byte-identical to the default
    En, _, _ = create_bessel_beam(128, 1e-6, 633e-9, 0.05, normalize='none')
    assert np.array_equal(E, En)


def test_bessel_power_normalises_over_window():
    E, _, _ = create_bessel_beam(128, 1e-6, 633e-9, 0.05, normalize='power')
    power = float(np.sum(np.abs(E) ** 2) * 1e-6 * 1e-6)
    assert np.isclose(power, 1.0, rtol=1e-12)


@pytest.mark.parametrize('factory, kwargs', [
    (create_top_hat_beam, dict(diameter=30e-6)),
    (create_annular_beam, dict(outer_diameter=40e-6, inner_diameter=20e-6)),
])
def test_aperture_invalid_normalize_raises(factory, kwargs):
    with pytest.raises(ValueError):
        factory(64, 1e-6, 633e-9, normalize='bogus', **kwargs)


def test_bessel_invalid_normalize_raises():
    with pytest.raises(ValueError):
        create_bessel_beam(64, 1e-6, 633e-9, 0.05, normalize='bogus')


def test_top_hat_normalize_preserves_complex64_dtype():
    """The in-place normalization must not upcast complex64 -> complex128
    (regression guard for the shared normalizer)."""
    E, _, _ = create_top_hat_beam(64, 1e-6, 633e-9, diameter=30e-6,
                                  dtype=np.complex64)
    assert E.dtype == np.complex64
