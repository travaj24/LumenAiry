"""C2 / S4-19 (AUDIT_V5_24_2 deferred roadmap) -- phase-mask expression
masks are evaluated by a restricted allowlist AST interpreter instead of
the built-in ``eval()`` with the whole ``np`` module exposed.

Two things are verified:

  * **Feature preserved** -- documented expressions still evaluate to the
    SAME phase, checked against an independent hand-analytic oracle (not
    the code's own evaluator output).
  * **Attack surface closed** -- hostile expressions (``__import__``,
    ``np.__loader__`` attribute escapes, ``lambda``, ``exec``,
    comprehensions, calls to non-whitelisted numpy callables) raise a
    clear ``ValueError``.

Plus the ``_safe_name`` sanitised-name collision warning (S4-19, :74).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy import user_library
from lumenairy.user_library import (
    _safe_eval_expression,
    _safe_name,
    load_phase_mask,
    save_phase_mask,
)


@pytest.fixture()
def isolated_library(tmp_path):
    """Point the user library at a temp dir and restore afterwards."""
    prior = user_library._library_path
    user_library.set_library_path(str(tmp_path))
    try:
        yield tmp_path
    finally:
        user_library._library_path = prior


# ------------------------------------------------------------------ #
# Hand-analytic grid oracle (independent of the code's evaluator).   #
# ------------------------------------------------------------------ #
def _grid(N, dx):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X ** 2 + Y ** 2)
    THETA = np.arctan2(Y, X)
    return X, Y, R, THETA


N = 16
DX = 2e-6
WL = 1.31e-6
K = 2 * np.pi / WL


# ------------------------------------------------------------------ #
# Documented expressions still evaluate identically (feature kept).  #
# ------------------------------------------------------------------ #
def test_expression_masks_match_analytic_oracle(isolated_library):
    X, Y, R, THETA = _grid(N, DX)
    cases = {
        # (expression string) : (independent analytic phase array)
        'atan2(Y, X) * 3': np.arctan2(Y, X) * 3.0,
        'k * R**2 / 2': K * R ** 2 / 2.0,
        'np.sin(2*pi*R/1e-3)': np.sin(2 * np.pi * R / 1e-3),
        'mod(THETA, 2*pi)': np.mod(THETA, 2 * np.pi),
        '2*pi * floor(R / 1e-3)': 2 * np.pi * np.floor(R / 1e-3),
    }
    for i, (expr, phase) in enumerate(cases.items()):
        name = f'c2_expr_{i}'
        save_phase_mask(name, expression=expr, wavelength=WL)
        got = load_phase_mask(name, N=N, dx=DX, wavelength=WL)
        expected = np.exp(1j * phase)
        np.testing.assert_allclose(
            got, expected, rtol=0, atol=1e-12,
            err_msg=f"expression {expr!r} did not match analytic oracle")


def test_comparison_and_bitwise_mask(isolated_library):
    """A numpy element-wise mask (comparison + bitwise-and) is supported
    and matches the hand oracle."""
    X, Y, R, THETA = _grid(N, DX)
    expr = '((R < 5*dx) & (X > 0)) * pi'.replace('dx', repr(DX))
    save_phase_mask('c2_mask', expression=expr, wavelength=WL)
    got = load_phase_mask('c2_mask', N=N, dx=DX, wavelength=WL)
    expected = np.exp(1j * (((R < 5 * DX) & (X > 0)) * np.pi))
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)


# ------------------------------------------------------------------ #
# Hostile expressions raise a clear ValueError (attack surface).     #
# ------------------------------------------------------------------ #
HOSTILE = [
    '__import__("os").system("echo pwned")',   # import escape
    'np.__loader__',                            # dunder attribute escape
    'np.load("x.npy")',                         # non-whitelisted np call
    '(lambda: 1)()',                            # lambda
    'exec("x = 1")',                            # exec
    'open("/etc/passwd")',                      # builtin open
    '[i for i in range(3)]',                    # comprehension
    'X.__class__.__mro__',                      # introspection escape
    'getattr(np, "load")',                      # getattr escape
]


@pytest.mark.parametrize('expr', HOSTILE)
def test_hostile_expression_rejected_direct(expr):
    ns = {'X': np.ones((4, 4)), 'np': np, 'pi': np.pi}
    with pytest.raises(ValueError):
        _safe_eval_expression(expr, ns)


@pytest.mark.parametrize('expr', HOSTILE)
def test_hostile_expression_rejected_via_load(isolated_library, expr):
    save_phase_mask('c2_hostile', expression=expr, wavelength=WL)
    with pytest.raises(ValueError):
        load_phase_mask('c2_hostile', N=N, dx=DX, wavelength=WL)


def test_no_arbitrary_side_effect(isolated_library, tmp_path):
    """A concrete proof the sandbox blocks code execution: an expression
    that would create a marker file must NOT create it (it raises)."""
    marker = tmp_path / 'PWNED'
    expr = (f'__import__("pathlib").Path({str(marker)!r}).write_text("x")')
    save_phase_mask('c2_side', expression=expr, wavelength=WL)
    with pytest.raises(ValueError):
        load_phase_mask('c2_side', N=N, dx=DX, wavelength=WL)
    assert not marker.exists(), 'sandbox allowed a filesystem side effect'


# ------------------------------------------------------------------ #
# Sanitised-name collision warning (S4-19, user_library.py:74).      #
# ------------------------------------------------------------------ #
def test_safe_name_warns_on_sanitisation():
    with pytest.warns(UserWarning, match='sanitised'):
        out = _safe_name('my mask/v2')
    assert out == 'my_mask_v2'


def test_safe_name_no_warning_when_clean():
    with warnings.catch_warnings():
        warnings.simplefilter('error')       # any warning -> failure
        assert _safe_name('clean_name_v2') == 'clean_name_v2'


def test_distinct_names_collide_to_same_file(isolated_library):
    """Two distinct names that sanitise to the same filename warn AND
    clobber -- the concrete hazard the warning surfaces."""
    with pytest.warns(UserWarning, match='sanitised'):
        save_phase_mask('spiral plate', expression='THETA', wavelength=WL)
    with pytest.warns(UserWarning, match='sanitised'):
        save_phase_mask('spiral/plate', expression='2*THETA', wavelength=WL)
    # Both mapped to spiral_plate.json; the second overwrote the first.
    lib = user_library.get_library_path() / 'phase_masks'
    assert (lib / 'spiral_plate.json').exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
