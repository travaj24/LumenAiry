"""Audit v5.24.2 B1 / S3-16 -- source-factory convention normalization.

Deprecation-cycle (rule 4) pins for the three convention fixes in
``lumenairy/sources/core.py``:

1. ``rng=`` alongside deprecated ``seed=`` on the Schell-family
   factories (CONVENTIONS.md section 3).  ``seed=<int>`` must still work,
   emit a ``DeprecationWarning`` naming ``rng`` + the v5.27 removal, and
   reproduce the ``rng=<int>`` stream bit-for-bit.
2. Explicit ``normalize=`` on ``create_top_hat_beam`` /
   ``create_annular_beam`` / ``create_bessel_beam`` whose DEFAULT
   reproduces the historical hard-coded convention exactly (top-hat /
   annular = unit power; Bessel = raw).
3. ``w0=`` (the 1/e^2 intensity radius, canonical) alongside the
   deprecated ``sigma=`` (field std-dev) on ``create_gaussian_beam``,
   with ``w0 = sigma * sqrt(2)``.

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


@pytest.mark.parametrize('factory, fn_name', _SCHELL_FACTORIES)
def test_rng_int_matches_deprecated_seed_bit_for_bit(factory, fn_name):
    """The new ``rng=<int>`` path reproduces the legacy
    ``seed=<int>`` stream bit-for-bit (byte-identical migration)."""
    ens_rng, *_ = factory(rng=7)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        ens_seed, *_ = factory(seed=7)
    assert np.array_equal(ens_rng, ens_seed), (
        f"{fn_name}: rng=7 and seed=7 must give a bit-identical ensemble")


@pytest.mark.parametrize('factory, fn_name', _SCHELL_FACTORIES)
def test_seed_still_works_and_warns(factory, fn_name):
    """Legacy ``seed=`` must still work AND emit a DeprecationWarning
    naming ``rng`` and the v5.27 removal (rule-4 old-form contract)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ens, *_ = factory(seed=3)
    assert ens.shape[0] == 4
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(dep) >= 1, f"{fn_name}: seed= must emit a DeprecationWarning"
    msg = str(dep[0].message)
    assert 'seed' in msg and 'rng' in msg, (
        f"{fn_name}: warning must name both 'seed' and 'rng'; got {msg!r}")
    assert '5.27' in msg, (
        f"{fn_name}: warning must state the v5.27 removal; got {msg!r}")


@pytest.mark.parametrize('factory, fn_name', _SCHELL_FACTORIES)
def test_rng_and_seed_are_mutually_exclusive(factory, fn_name):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises(TypeError):
            factory(rng=1, seed=2)


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


def test_source_classmethods_accept_rng_and_deprecate_seed():
    """The Source.* Schell classmethods forward the rng/seed contract."""
    ens_rng, *_ = la.Source.gaussian_schell(
        N=32, dx=5e-6, wavelength=633e-9, w0=40e-6, sigma_g=20e-6,
        n_realizations=4, rng=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        ens_seed, *_ = la.Source.gaussian_schell(
            N=32, dx=5e-6, wavelength=633e-9, w0=40e-6, sigma_g=20e-6,
            n_realizations=4, seed=2)
    assert np.array_equal(ens_rng, ens_seed)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught)


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


def test_w0_equals_sigma_over_sqrt2_bit_for_bit():
    """``w0=w`` reproduces ``sigma=w/sqrt(2)`` bit-for-bit."""
    w0 = 12e-6
    E_w0, _, _ = create_gaussian_beam(64, 1e-6, 633e-9, w0=w0,
                                      normalize='none')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        E_sg, _, _ = create_gaussian_beam(64, 1e-6, 633e-9,
                                          sigma=w0 / np.sqrt(2),
                                          normalize='none')
    assert np.array_equal(E_w0, E_sg)


def test_sigma_still_works_and_warns():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        E, _, _ = create_gaussian_beam(32, 1e-6, 633e-9, sigma=5e-6)
    assert E.shape == (32, 32)
    dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep, "sigma= must emit a DeprecationWarning"
    msg = str(dep[0].message)
    assert 'sigma' in msg and 'w0' in msg and '5.27' in msg, (
        f"warning must name sigma, w0, and v5.27; got {msg!r}")


def test_w0_form_is_warning_clean():
    with warnings.catch_warnings():
        warnings.simplefilter('error', DeprecationWarning)
        create_gaussian_beam(32, 1e-6, 633e-9, w0=7e-6)


def test_sigma_and_w0_mutually_exclusive():
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', DeprecationWarning)
        with pytest.raises(ValueError):
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
