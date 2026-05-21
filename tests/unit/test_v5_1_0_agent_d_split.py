"""Regression tests for v5.1.0 Agent D split of
``lumenairy/propagators/asymptotic.py`` into five submodules
(``asymptotic_modes`` / ``asymptotic_canonical_fit`` /
``asymptotic_aberration_tensor`` / ``asymptotic_maslov`` /
``asymptotic_jax_twin``).

Invariants pinned here
----------------------

1. Every previously-public name remains importable from
   ``lumenairy.propagators.asymptotic`` (the shell).
2. Module-level cache + lock pairs (``_LG_MODE_STACK_CACHE`` /
   ``_LG_MODE_STACK_LOCK``, ``_HG_MODE_STACK_CACHE`` /
   ``_HG_MODE_STACK_LOCK``, ``_JAX_IFT_SOLVER_CACHE`` /
   ``_JAX_IFT_SOLVER_CACHE_LOCK``) remain present at the shell module
   path so the existing v4.14.x dispatcher pins keep tracking them.
3. The submodule re-exports are identity-equal to the names at their
   defining submodule -- patching the shell binding while the function
   reads from its own module's globals would silently de-sync; pin
   identity to catch a future regression.
4. The ``propagate_modal_asymptotic`` body lives in the shell so it
   resolves ``_solve_envelope_stationary_batch`` (and friends) against
   the shell's globals.  This preserves the v4.14.1 / v4.15
   monkey-patch contract.
5. Top-level ``lumenairy.la.<name>`` access still works for the names
   the parent package re-exports.

Author: Agent D / v5.1.0 file-split.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Smoke -- all previously-public names import from the shell
# ---------------------------------------------------------------------------

PUBLIC_NAMES = [
    # Containers
    'CanonicalPolyFit',
    'HFPolyFit',
    'AberrationTensorResult',
    'JaxAberrationTensorResult',
    # Modes
    'lg_polynomial',
    'hg_polynomial',
    'evaluate_lg_mode',
    'evaluate_hg_mode',
    'decompose_lg',
    'decompose_hg',
    'lg_seidel_label',
    'clear_lg_polynomial_cache',
    'clear_lg_mode_stack_cache',
    # Moments
    'gaussian_moment_2d',
    'gaussian_moment_table_2d',
    # Canonical / HF
    'fit_canonical_polynomials',
    'fit_hf_polynomials',
    'solve_envelope_stationary',
    'propagate_hf_chebyshev_quadrature',
    # Aberration tensor + propagator
    'aberration_tensor',
    'propagate_modal_asymptotic',
    # JAX paths
    'aberration_tensor_lg00_jax',
    'propagate_modal_asymptotic_lg00_jax',
    'solve_envelope_stationary_jax_ift',
    'fit_canonical_polynomials_jax',
]


def test_shell_exposes_all_public_names():
    """Every name that pre-v5.1.0 lived in
    ``lumenairy.propagators.asymptotic`` remains importable from the
    shell after the file-split.
    """
    mod = importlib.import_module('lumenairy.propagators.asymptotic')
    missing = [name for name in PUBLIC_NAMES if not hasattr(mod, name)]
    assert not missing, (
        f'Shell module ``lumenairy.propagators.asymptotic`` is missing '
        f'the following names after the v5.1.0 split: {missing}'
    )


def test_shell_all_lists_all_public_names():
    """``__all__`` on the shell still advertises the full public surface."""
    mod = importlib.import_module('lumenairy.propagators.asymptotic')
    all_list = set(getattr(mod, '__all__', ()))
    missing = [name for name in PUBLIC_NAMES if name not in all_list]
    assert not missing, (
        f'Shell ``__all__`` does not list the following pre-split public '
        f'names: {missing}'
    )


# ---------------------------------------------------------------------------
# Submodule layout -- each split file holds its declared contents
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('submod, name', [
    # asymptotic_modes
    ('asymptotic_modes', 'lg_polynomial'),
    ('asymptotic_modes', 'hg_polynomial'),
    ('asymptotic_modes', 'decompose_lg'),
    ('asymptotic_modes', 'decompose_hg'),
    ('asymptotic_modes', 'gaussian_moment_2d'),
    ('asymptotic_modes', 'gaussian_moment_table_2d'),
    ('asymptotic_modes', 'lg_seidel_label'),
    ('asymptotic_modes', 'clear_lg_polynomial_cache'),
    ('asymptotic_modes', 'clear_lg_mode_stack_cache'),
    ('asymptotic_modes', '_LG_MODE_STACK_CACHE'),
    ('asymptotic_modes', '_HG_MODE_STACK_CACHE'),
    ('asymptotic_modes', '_LG_MODE_STACK_LOCK'),
    ('asymptotic_modes', '_HG_MODE_STACK_LOCK'),
    # asymptotic_canonical_fit
    ('asymptotic_canonical_fit', 'CanonicalPolyFit'),
    ('asymptotic_canonical_fit', 'HFPolyFit'),
    ('asymptotic_canonical_fit', 'fit_canonical_polynomials'),
    ('asymptotic_canonical_fit', 'fit_hf_polynomials'),
    ('asymptotic_canonical_fit', 'solve_envelope_stationary'),
    ('asymptotic_canonical_fit', 'propagate_hf_chebyshev_quadrature'),
    # asymptotic_aberration_tensor
    ('asymptotic_aberration_tensor', 'AberrationTensorResult'),
    ('asymptotic_aberration_tensor', 'aberration_tensor'),
    ('asymptotic_aberration_tensor', '_compute_M_b'),
    ('asymptotic_aberration_tensor', '_phi_v2_hessian'),
    # asymptotic_maslov
    ('asymptotic_maslov', '_maslov_branch_corrected_sqrt'),
    ('asymptotic_maslov', '_solve_envelope_stationary_batch'),
    ('asymptotic_maslov', '_compute_M_b_batch'),
    ('asymptotic_maslov', '_phi_v2_hessian_batch'),
    ('asymptotic_maslov', '_gaussian_moment_table_2d_batch'),
    ('asymptotic_maslov', '_batched_polynomial_substitute_linear_2d'),
    ('asymptotic_maslov', '_batched_polynomial_under_affine_shift'),
    ('asymptotic_maslov', '_poly_dict_to_array'),
    # asymptotic_jax_twin
    ('asymptotic_jax_twin', 'aberration_tensor_lg00_jax'),
    ('asymptotic_jax_twin', 'propagate_modal_asymptotic_lg00_jax'),
    ('asymptotic_jax_twin', 'JaxAberrationTensorResult'),
    ('asymptotic_jax_twin', 'solve_envelope_stationary_jax_ift'),
    ('asymptotic_jax_twin', 'fit_canonical_polynomials_jax'),
    ('asymptotic_jax_twin', '_JAX_IFT_SOLVER_CACHE'),
    ('asymptotic_jax_twin', '_JAX_IFT_SOLVER_CACHE_LOCK'),
])
def test_submodule_owns_name(submod, name):
    mod = importlib.import_module(f'lumenairy.propagators.{submod}')
    assert hasattr(mod, name), (
        f'Submodule ``lumenairy.propagators.{submod}`` is missing '
        f'declared name {name!r}.'
    )


# ---------------------------------------------------------------------------
# Re-export identity -- shell and submodule bindings reference the
# SAME object so monkey-patches on the shell propagate to submodule
# call sites that look up via the shell module namespace.
# ---------------------------------------------------------------------------

REEXPORT_IDENTITY_CASES = [
    ('asymptotic_modes', 'lg_polynomial'),
    ('asymptotic_modes', 'gaussian_moment_table_2d'),
    ('asymptotic_modes', '_LG_MODE_STACK_CACHE'),
    ('asymptotic_modes', '_LG_MODE_STACK_LOCK'),
    ('asymptotic_modes', '_HG_MODE_STACK_CACHE'),
    ('asymptotic_modes', '_HG_MODE_STACK_LOCK'),
    ('asymptotic_canonical_fit', 'CanonicalPolyFit'),
    ('asymptotic_canonical_fit', 'HFPolyFit'),
    ('asymptotic_canonical_fit', 'solve_envelope_stationary'),
    ('asymptotic_canonical_fit', 'fit_canonical_polynomials'),
    ('asymptotic_aberration_tensor', 'aberration_tensor'),
    ('asymptotic_aberration_tensor', 'AberrationTensorResult'),
    ('asymptotic_aberration_tensor', '_compute_M_b'),
    ('asymptotic_maslov', '_solve_envelope_stationary_batch'),
    ('asymptotic_maslov', '_compute_M_b_batch'),
    ('asymptotic_maslov', '_maslov_branch_corrected_sqrt'),
    ('asymptotic_jax_twin', '_JAX_IFT_SOLVER_CACHE_LOCK'),
]


@pytest.mark.parametrize('submod, name', REEXPORT_IDENTITY_CASES)
def test_shell_reexport_is_identity(submod, name):
    shell = importlib.import_module('lumenairy.propagators.asymptotic')
    inner = importlib.import_module(f'lumenairy.propagators.{submod}')
    s_val = getattr(shell, name)
    i_val = getattr(inner, name)
    assert s_val is i_val, (
        f'Shell ``asymptotic.{name}`` is not identity-equal to the '
        f'binding in ``{submod}``.  Re-export must be a transparent '
        f'view of the canonical submodule object so monkey-patches and '
        f'cache-lock pins keep tracking the same object across the '
        f'shell <-> submodule boundary.'
    )


# ---------------------------------------------------------------------------
# Top-level ``lumenairy`` re-exports survive the split
# ---------------------------------------------------------------------------

LA_TOPLEVEL_NAMES = [
    'CanonicalPolyFit',
    'HFPolyFit',
    'AberrationTensorResult',
    'fit_canonical_polynomials',
    'fit_hf_polynomials',
    'aberration_tensor',
    'propagate_modal_asymptotic',
    'propagate_hf_chebyshev_quadrature',
    'solve_envelope_stationary',
    'lg_polynomial',
    'hg_polynomial',
    'evaluate_lg_mode',
    'evaluate_hg_mode',
    'decompose_lg',
    'decompose_hg',
    'lg_seidel_label',
    'gaussian_moment_2d',
    'gaussian_moment_table_2d',
    'clear_lg_polynomial_cache',
    'clear_lg_mode_stack_cache',
]


@pytest.mark.parametrize('name', LA_TOPLEVEL_NAMES)
def test_lumenairy_toplevel_still_exports(name):
    import lumenairy as la
    assert hasattr(la, name), (
        f'Top-level ``lumenairy.{name}`` is missing after the v5.1.0 '
        f'Agent D split.  ``lumenairy/__init__.py`` imports must keep '
        f'tracking the shell re-exports.'
    )


# ---------------------------------------------------------------------------
# propagate_modal_asymptotic body lives in the shell for monkey-patching
# ---------------------------------------------------------------------------

def test_propagate_modal_asymptotic_defined_in_shell():
    """The propagator body must live in ``asymptotic.py`` (the shell)
    so its globals match the shell module's globals; pre-v5.1.0
    tests monkey-patch ``_asy._solve_envelope_stationary_batch`` and
    expect the propagator to honour the patch.

    If a future refactor moves the body to a submodule, the patch
    contract silently breaks.  Pin the home explicitly here.
    """
    from lumenairy.propagators import asymptotic as shell
    f = shell.propagate_modal_asymptotic
    assert f.__module__ == 'lumenairy.propagators.asymptotic', (
        f'``propagate_modal_asymptotic`` is defined in '
        f'{f.__module__!r} -- must stay in '
        f'``lumenairy.propagators.asymptotic`` so its ``__globals__`` '
        f'are the shell module\'s globals (monkey-patch contract).'
    )


def test_monkeypatch_solve_envelope_stationary_batch_takes_effect(monkeypatch):
    """``propagate_modal_asymptotic`` resolves
    ``_solve_envelope_stationary_batch`` against the shell module's
    globals.  Replacing the shell binding must therefore replace the
    function actually invoked by the propagator body -- this is the
    same invariant pinned by
    ``tests/unit/test_audit_fixes_v4_14_1_agent_a.py``'s spy.

    The test builds a small fit, replaces the shell-level batched
    solver with a counting spy, and asserts the spy fired at least
    once on a single ``propagate_modal_asymptotic`` call.
    """
    import lumenairy as lm
    from lumenairy.propagators import asymptotic as shell

    rx = lm.make_singlet(
        R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3,
    )
    rx['object_distance'] = 0.1
    fit = shell.fit_canonical_polynomials(
        rx, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=4, n_pupil=4, poly_order=3,
    )

    call_count = {'n': 0}
    original = shell._solve_envelope_stationary_batch

    def spy(*args, **kwargs):
        call_count['n'] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(shell, '_solve_envelope_stationary_batch', spy)

    s2x_axis = np.linspace(-5e-6, 5e-6, 3)
    s2y_axis = np.linspace(-5e-6, 5e-6, 3)
    S2X, S2Y = np.meshgrid(s2x_axis, s2y_axis, indexing='xy')
    _ = shell.propagate_modal_asymptotic(
        fit,
        source_point=(0.0, 0.0),
        w_s=20e-6, w_p=0.02,
        v2_centre=(0.0, 0.0),
        s2_grid_x=S2X, s2_grid_y=S2Y,
        maslov_tracking='row_reset',
    )
    assert call_count['n'] >= 1, (
        'Monkey-patching the shell\'s '
        '``_solve_envelope_stationary_batch`` did not take effect '
        'during ``propagate_modal_asymptotic`` -- the v5.1.0 split '
        'has broken the v4.14.1 / v4.15 monkey-patch contract.'
    )


# ---------------------------------------------------------------------------
# Smoke: end-to-end numeric sanity for the four primary public paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def _singlet_fit():
    import lumenairy as lm
    from lumenairy.propagators import asymptotic as shell

    rx = lm.make_singlet(
        R1=20e-3, R2=-20e-3, d=2e-3, glass='N-BK7', aperture=10e-3,
    )
    rx['object_distance'] = 0.1
    return shell.fit_canonical_polynomials(
        rx, wavelength=1.31e-6,
        source_box_half=20e-6, pupil_box_half=0.02,
        n_field=4, n_pupil=4, poly_order=3,
    )


def test_lg_polynomial_basic(_singlet_fit):
    """``lg_polynomial(0, 0, w)`` returns the LG_00 Gaussian
    normalisation:  the only non-zero coefficient is at ``(0, 0)`` and
    equals ``N = sqrt(2 / (pi w^2))``.
    """
    import math
    from lumenairy.propagators.asymptotic import lg_polynomial
    w = 50e-6
    poly = lg_polynomial(0, 0, w)
    assert (0, 0) in poly
    N = math.sqrt(2.0 / (math.pi * w * w))
    assert abs(poly[(0, 0)] - N) / N < 1e-12


def test_gaussian_moment_2d_basic():
    """``gaussian_moment_2d(0, 0, sigma) == 1`` (normalisation
    convention)."""
    from lumenairy.propagators.asymptotic import gaussian_moment_2d
    sigma = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
    v = gaussian_moment_2d(0, 0, sigma)
    assert abs(v - 1.0) < 1e-12


def test_solve_envelope_stationary_smoke(_singlet_fit):
    from lumenairy.propagators.asymptotic import solve_envelope_stationary
    v_star, n_iter, resid = solve_envelope_stationary(
        _singlet_fit, (0.0, 0.0), (0.0, 0.0),
        w_s=20e-6, w_p=0.02,
    )
    assert isinstance(v_star, tuple)
    assert len(v_star) == 2
    assert np.isfinite(resid)


def test_propagate_modal_asymptotic_smoke(_singlet_fit):
    from lumenairy.propagators.asymptotic import propagate_modal_asymptotic
    s2x_axis = np.linspace(-5e-6, 5e-6, 5)
    s2y_axis = np.linspace(-5e-6, 5e-6, 5)
    S2X, S2Y = np.meshgrid(s2x_axis, s2y_axis, indexing='xy')
    field = propagate_modal_asymptotic(
        _singlet_fit,
        source_point=(0.0, 0.0),
        w_s=20e-6, w_p=0.02,
        v2_centre=(0.0, 0.0),
        s2_grid_x=S2X, s2_grid_y=S2Y,
    )
    assert field.shape == S2X.shape
    assert np.iscomplexobj(field)
    assert np.any(np.abs(field) > 0.0), (
        'Modal asymptotic propagator returned all-zero field on the '
        'singlet smoke check -- physically expected at least the on-'
        'axis chief-ray to evaluate non-zero.'
    )


def test_aberration_tensor_smoke(_singlet_fit):
    from lumenairy.propagators.asymptotic import aberration_tensor
    res = aberration_tensor(
        _singlet_fit, (0.0, 0.0),
        source_point=(0.0, 0.0),
        w_s=20e-6, w_p=0.02,
    )
    assert res.L.shape == (len(res.output_modes), len(res.source_modes))
    assert np.all(np.isfinite(res.L.real) & np.isfinite(res.L.imag))


# ---------------------------------------------------------------------------
# Cache + lock identity preservation
# ---------------------------------------------------------------------------

def test_cache_lock_pairs_present_on_shell():
    """The dispatcher pins in
    ``tests/unit/test_v4_14_2_dispatcher_pin_cache_locks.py`` walk
    every submodule under ``lumenairy`` discovering
    ``_<NAME>_CACHE`` globals and demanding companion
    ``_<NAME>_LOCK``s.  Verify the four pre-v5.1.0 caches survive at
    their new submodule homes AND remain visible on the shell.
    """
    from lumenairy.propagators import asymptotic as shell
    from lumenairy.propagators import asymptotic_modes as modes
    from lumenairy.propagators import asymptotic_jax_twin as jt
    import threading

    # Mode-stack caches/locks in asymptotic_modes (re-exported by shell)
    for cache_name, lock_name in [
        ('_LG_MODE_STACK_CACHE', '_LG_MODE_STACK_LOCK'),
        ('_HG_MODE_STACK_CACHE', '_HG_MODE_STACK_LOCK'),
    ]:
        assert hasattr(modes, cache_name), \
            f'asymptotic_modes lost {cache_name}'
        assert hasattr(modes, lock_name), \
            f'asymptotic_modes lost {lock_name}'
        assert hasattr(shell, cache_name), \
            f'shell lost {cache_name} re-export'
        assert hasattr(shell, lock_name), \
            f'shell lost {lock_name} re-export'
        lock_obj = getattr(modes, lock_name)
        assert isinstance(lock_obj, type(threading.Lock())), (
            f'{lock_name} on asymptotic_modes is not a threading.Lock')

    # JAX-IFT solver cache + lock in asymptotic_jax_twin
    assert hasattr(jt, '_JAX_IFT_SOLVER_CACHE')
    assert hasattr(jt, '_JAX_IFT_SOLVER_CACHE_LOCK')
    assert hasattr(shell, '_JAX_IFT_SOLVER_CACHE')
    assert hasattr(shell, '_JAX_IFT_SOLVER_CACHE_LOCK')
    assert isinstance(jt._JAX_IFT_SOLVER_CACHE_LOCK, type(threading.Lock()))
