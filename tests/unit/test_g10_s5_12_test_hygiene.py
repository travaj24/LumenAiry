"""G10 / S5-12 (AUDIT_V5_24_2) -- test-suite hygiene config.

S5-12 flagged that the suite had NO ``filterwarnings`` config (a warning
mid-test could pass silently) and NO ``xfail_strict`` (a dead xfail that
starts passing silently reports XPASS instead of failing).

These tests read the LIVE, effective pytest configuration through
``pytestconfig.getini`` -- an independent oracle from the raw
``pyproject.toml`` text -- so they pin the policy regardless of how the
config file is formatted.  They fail before the config edit
(``xfail_strict`` defaults to False; ``filterwarnings`` is empty) and
pass after.
"""
from __future__ import annotations


def test_xfail_strict_is_enabled(pytestconfig):
    """A test marked xfail that starts passing must FAIL, not XPASS."""
    assert pytestconfig.getini('xfail_strict') is True, (
        'S5-12: xfail_strict must be enabled so dead xfail scaffolding '
        'surfaces as a failure instead of a silent XPASS.')


def test_filterwarnings_is_configured(pytestconfig):
    """An explicit warnings policy must exist (was entirely absent)."""
    fw = pytestconfig.getini('filterwarnings')
    assert fw, 'S5-12: no filterwarnings policy configured.'
    # The always-a-bug ``return instead of assert`` footgun is promoted
    # to an error so a returning test cannot silently "pass".
    assert any('PytestReturnNotNoneWarning' in entry for entry in fw), (
        'S5-12: PytestReturnNotNoneWarning is not promoted to an error.')


def test_return_not_none_filter_targets_a_real_category():
    """Guard against a typo'd category name that would silently no-op."""
    import pytest as _pytest
    assert hasattr(_pytest, 'PytestReturnNotNoneWarning'), (
        'pytest.PytestReturnNotNoneWarning must exist for the '
        'filterwarnings error rule to have teeth.')
