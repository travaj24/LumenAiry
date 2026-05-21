"""v4.16.0 Agent D: Sellmeier validity ranges (ROADMAP #14).

Pins
----

``GLASS_VALIDITY[name]`` carries a ``(lambda_min, lambda_max)`` tuple
in METRES for every glass that has a documented Sellmeier (or
polynomial) fit range.  When ``get_glass_index(name, wavelength)``
is called with a wavelength outside this range, a ``UserWarning`` is
emitted -- but the value is still returned (extrapolation is
sometimes acceptable for design-space exploration).

Pre-v4.16:
    ``get_glass_index('N-BK7', 200e-9)`` silently returned an
    extrapolated number that has no physical meaning (N-BK7 has a
    documented UV cutoff at ~310 nm).

v4.16.0:
    Same call returns the extrapolated number AND emits a
    ``UserWarning`` so the caller knows the value is outside the
    fit's documented validity.

Author: Andrew Traverso -- v4.16.0 / Agent D
"""
from __future__ import annotations

import warnings

import pytest

import lumenairy as la
from lumenairy.glass import (
    _DEFAULT_VALIDITY,
    GLASS_VALIDITY,
    _validity_warned,
)

# ===========================================================================
# Out-of-range warning behaviour (the headline)
# ===========================================================================


@pytest.fixture(autouse=True)
def _reset_validity_warned():
    """Reset the warn-once memo between tests so each test starts
    from a clean slate (otherwise the second test sees the first
    test's warned-state and skips its own warning).
    """
    _validity_warned.clear()
    yield
    _validity_warned.clear()


def test_validity_warning_emitted_outside_range():
    """``get_glass_index('N-BK7', 200e-9)`` emits a UserWarning.

    The Schott N-BK7 catalogue range is 0.30-2.5 um; 200 nm is well
    below the UV cutoff at 310 nm.  Pre-v4.16 the call returned an
    extrapolated number silently; v4.16 warns.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        n = la.get_glass_index('N-BK7', 200e-9)

    # Warning must have fired with the documented message format.
    matched = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'N-BK7' in str(w.message)
        and 'validity' in str(w.message)
    ]
    assert len(matched) >= 1, (
        f"Expected UserWarning with 'N-BK7' and 'validity' substrings; "
        f"got {[(w.category.__name__, str(w.message)) for w in caught]}")

    # But the call still returned a finite float (extrapolation
    # accepted; warning is advisory, not a raise).
    assert isinstance(n, float)
    assert n > 1.0  # any glass index > 1 (vacuum)


def test_validity_no_warning_in_range():
    """``get_glass_index('N-BK7', 587.56e-9)`` emits NO warning.

    The d-line (587.56 nm) is well within the Schott N-BK7 catalogue
    range of 0.30-2.5 um.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        n = la.get_glass_index('N-BK7', 587.56e-9)

    # No UserWarning containing 'validity' should be in the caught
    # list.  Other warnings (e.g. RuntimeWarning) are allowed -- the
    # filter is narrowed to the validity-specific message.
    validity_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'validity' in str(w.message)
    ]
    assert len(validity_warnings) == 0, (
        f"In-range query should NOT emit a validity warning; got "
        f"{[str(w.message) for w in validity_warnings]}")
    assert n == pytest.approx(1.5168, abs=5e-4)


def test_validity_warning_is_one_shot_per_pair():
    """Repeated out-of-range calls with the SAME (glass, wavelength)
    fire the warning only once -- otherwise log churn in parameter
    sweeps would drown the user in noise.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        # Call 5 times with the same args.
        for _ in range(5):
            la.get_glass_index('N-BK7', 200e-9)

    validity_warnings = [
        w for w in caught
        if issubclass(w.category, UserWarning)
        and 'validity' in str(w.message)
    ]
    assert len(validity_warnings) == 1, (
        f"Out-of-range warning must be one-shot per (glass, wavelength) "
        f"pair; got {len(validity_warnings)} warnings.")


# ===========================================================================
# Validity-range table structure
# ===========================================================================


def test_GLASS_VALIDITY_is_dict_of_pairs():
    """Sanity: every entry is a 2-tuple of floats with lmin <= lmax."""
    for name, val in GLASS_VALIDITY.items():
        assert isinstance(val, tuple), (
            f"GLASS_VALIDITY[{name!r}] must be a tuple; got "
            f"{type(val).__name__}")
        assert len(val) == 2, (
            f"GLASS_VALIDITY[{name!r}] must be length 2; got "
            f"{len(val)}")
        lmin, lmax = val
        assert lmin <= lmax, (
            f"GLASS_VALIDITY[{name!r}] lmin > lmax: "
            f"({lmin}, {lmax})")
        assert lmin >= 0.0, (
            f"GLASS_VALIDITY[{name!r}] lmin < 0: {lmin}")


def test_GLASS_VALIDITY_includes_all_schott_n_series():
    """Every Schott N-* glass in GLASS_REGISTRY must have an explicit
    validity entry -- per dispatch sheet, "Most crown glasses:
    0.31-2.5 um (UV cutoff)" is too coarse to skip the
    table-population step."""
    from lumenairy.glass import GLASS_REGISTRY
    schott_n_glasses = [n for n in GLASS_REGISTRY if n.startswith('N-')]
    for name in schott_n_glasses:
        assert name in GLASS_VALIDITY, (
            f"Schott N-series glass {name!r} missing from "
            f"GLASS_VALIDITY -- per dispatch sheet, every existing "
            f"glass must have an explicit validity range.")


def test_GLASS_VALIDITY_default_is_unrestricted():
    """The documented default is ``(0.0, np.inf)`` so glasses without
    an entry never trigger a warning (legacy back-compat)."""
    assert _DEFAULT_VALIDITY[0] == 0.0
    assert _DEFAULT_VALIDITY[1] == float('inf')


def test_glass_without_validity_entry_emits_no_warning():
    """A glass with no GLASS_VALIDITY entry must not trigger the
    warning at ANY wavelength -- the default is unrestricted.

    Register a synthetic glass for this test; teardown removes it
    so other tests are unaffected.
    """
    from lumenairy.glass import GLASS_REGISTRY
    synth_name = '__TEST_NO_VALIDITY__'
    GLASS_REGISTRY[synth_name] = lambda wl: 1.5
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            la.get_glass_index(synth_name, 100e-9)  # absurd UV
            la.get_glass_index(synth_name, 10e-3)  # absurd IR

        validity_warnings = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and 'validity' in str(w.message)
        ]
        assert len(validity_warnings) == 0, (
            f"Glass without GLASS_VALIDITY entry must not warn; got "
            f"{[str(w.message) for w in validity_warnings]}")
    finally:
        del GLASS_REGISTRY[synth_name]


# ===========================================================================
# Coverage: how many glasses carry an explicit validity?
# ===========================================================================


def test_validity_coverage_count():
    """v4.16.0 ships explicit validity entries for at least 60
    glasses (current population is ~77).  Pin the lower bound so a
    future PR that accidentally drops an entry trips this test."""
    assert len(GLASS_VALIDITY) >= 60, (
        f"GLASS_VALIDITY size {len(GLASS_VALIDITY)} < 60.  "
        f"v4.16.0 (ROADMAP #14) requires explicit ranges for the "
        f"Schott / Ohara / generic / CDGM / Hikari / Sumita catalogue.")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
