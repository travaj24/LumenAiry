"""v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
smoke + behavioural pins for the new opt-in telemetry logging surface.

The audit cited three long-running paths that lacked per-iteration
telemetry:

* :func:`lumenairy.apply_real_lens_traced`  (per-Newton-iteration)
* :func:`lumenairy.design_optimize`         (per-scipy-iteration)
* :func:`lumenairy.monte_carlo_tolerancing` (per-trial)

v5.3.2 adds module-level ``logger = lumenairy._logging.get_logger(__name__)``
on the three consumer modules and emits ``logger.info(...)`` at entry +
per-iteration sites.  This file pins:

1. The new ``lumenairy._logging`` module imports and ``get_logger``
   returns a ``logging.Logger`` whose name is a descendant of
   ``'lumenairy'``.

2. **Default-quiet contract**: with no handler attached, the three
   target functions produce ZERO captured log output.  The
   ``NullHandler`` on the ``'lumenairy'`` root logger is responsible.

3. **Telemetry-active contract**: with ``caplog.at_level(logging.INFO,
   logger='lumenairy')`` set, each target function emits at least one
   INFO record bearing its recognisable per-iteration marker (``newton
   iter`` / ``iter`` / ``trial``).

4. **Behavioural preservation**: ``design_optimize`` returns the same
   merit / x-vector value with and without a handler attached.
   Telemetry must be purely observational; no side-effect on the
   physics.

Author: Andrew Traverso -- v5.3.2 ROADMAP logging adoption sweep.
"""
# v5.3.2 (ROADMAP logging adoption sweep -- per-iteration telemetry):
# Module-level smoke tests for the new opt-in telemetry surface added
# in lumenairy/_logging.py + three consumer modules.
from __future__ import annotations

import logging
from typing import Any, List, Tuple

import numpy as np
import pytest

# ----------------------------------------------------------------------
# Section 1 -- _logging module smoke
# ----------------------------------------------------------------------


def test_logging_module_importable():
    """``lumenairy._logging`` is importable and ``get_logger`` returns
    a ``logging.Logger`` instance whose name descends from the
    ``'lumenairy'`` root logger.
    """
    from lumenairy import _logging

    log = _logging.get_logger('lumenairy.foo')
    assert isinstance(log, logging.Logger), (
        f"get_logger must return a logging.Logger; got "
        f"{type(log).__name__}")
    assert log.name == 'lumenairy.foo', (
        f"get_logger should return the logger named 'lumenairy.foo'; "
        f"got {log.name!r}.")

    # The root logger must have a NullHandler attached so default
    # library use produces zero log output.  At least one handler must
    # be a NullHandler -- the contract is "non-empty handler list whose
    # net effect is silence" rather than "exactly one NullHandler",
    # because pytest's logging plugin may attach its own capture
    # handlers during a test run.
    root = logging.getLogger('lumenairy')
    has_null = any(
        isinstance(h, logging.NullHandler) for h in root.handlers)
    assert has_null, (
        f"lumenairy root logger must have a NullHandler so the "
        f"default-quiet contract holds; handlers={root.handlers!r}.")


# ----------------------------------------------------------------------
# Section 2 -- design_optimize telemetry pins
# ----------------------------------------------------------------------


def _build_small_quadratic_problem():
    """Minimal 2-parameter quadratic optimisation problem.  Used by
    multiple tests below to exercise design_optimize without the
    wave-leg cost.

    The initial point is deliberately offset from the optimum
    (template params = [1.0, 1.0], target = [0, 0]) so scipy's
    L-BFGS-B actually runs a non-trivial number of iterations
    before convergence -- a quadratic centered on the initial
    point would converge in zero iterations and never fire the
    per-iter callback we want to pin.
    """
    from lumenairy.optimize import (
        CallableMerit,
        DesignParameterization,
    )

    template = {
        'params': [1.0, 1.0],
        'surfaces': [{'radius': float('inf'), 'aperture': 5e-3,
                      'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    free_vars: List[Tuple[Any, ...]] = [('params', 0), ('params', 1)]
    bounds = [(-5.0, 5.0)] * 2
    param = DesignParameterization(
        template=template, free_vars=free_vars, bounds=bounds)

    def _quadratic_merit(ctx) -> float:
        x = np.asarray(ctx.x, dtype=np.float64)
        return float(np.sum(x * x))

    merit = CallableMerit(_quadratic_merit, weight=1.0, name='sum_sq')
    return param, [merit]


def test_design_optimize_default_quiet(caplog):
    """Default-quiet pin: design_optimize on the small quadratic
    problem produces ZERO captured log records on the ``'lumenairy'``
    logger when no handler is attached (``caplog`` not at-level-set).

    NB: ``caplog`` captures via pytest's own handler at WARNING by
    default; INFO records propagate to it only when ``at_level`` /
    ``set_level`` raises the threshold.  Without that, the NullHandler
    on the lumenairy root absorbs the INFO records.
    """
    from lumenairy.optimize import design_optimize

    param, merits = _build_small_quadratic_problem()
    # caplog is intentionally NOT raised to INFO here -- this exercises
    # the default-quiet path.
    _ = design_optimize(
        param, merits, wavelength=1.31e-6,
        method='L-BFGS-B', max_iter=2, verbose=False)
    # No INFO records on the lumenairy logger should have been captured.
    lumenairy_records = [r for r in caplog.records
                         if r.name.startswith('lumenairy')]
    assert len(lumenairy_records) == 0, (
        f"default-quiet contract violated: design_optimize captured "
        f"{len(lumenairy_records)} lumenairy log records without an "
        f"explicit handler attachment.  First: "
        f"{lumenairy_records[0].getMessage() if lumenairy_records else '?'}")


def test_design_optimize_telemetry_active(caplog):
    """Telemetry-active pin: with ``caplog.at_level(logging.INFO,
    logger='lumenairy')``, design_optimize emits at least one INFO
    record carrying the recognisable per-iteration marker
    ``'design_optimize: iter '``.
    """
    from lumenairy.optimize import design_optimize

    param, merits = _build_small_quadratic_problem()
    with caplog.at_level(logging.INFO, logger='lumenairy'):
        _ = design_optimize(
            param, merits, wavelength=1.31e-6,
            method='L-BFGS-B', max_iter=3, verbose=False)
    iter_records = [r for r in caplog.records
                    if r.name.startswith('lumenairy')
                    and 'design_optimize: iter ' in r.getMessage()]
    assert len(iter_records) >= 1, (
        f"telemetry-active contract violated: design_optimize emitted "
        f"no 'design_optimize: iter' INFO records.  Captured "
        f"lumenairy messages: "
        f"{[r.getMessage() for r in caplog.records if r.name.startswith('lumenairy')]}")


def test_design_optimize_telemetry_observational_only(caplog):
    """Behavioural preservation pin: design_optimize returns the same
    optimised x-vector and merit with and without a handler attached.
    Telemetry must be purely observational.
    """
    from lumenairy.optimize import design_optimize

    # Run 1: default (no handler).
    param_a, merits_a = _build_small_quadratic_problem()
    result_a = design_optimize(
        param_a, merits_a, wavelength=1.31e-6,
        method='L-BFGS-B', max_iter=5, verbose=False)

    # Run 2: with INFO-level logging on the lumenairy logger.
    param_b, merits_b = _build_small_quadratic_problem()
    with caplog.at_level(logging.INFO, logger='lumenairy'):
        result_b = design_optimize(
            param_b, merits_b, wavelength=1.31e-6,
            method='L-BFGS-B', max_iter=5, verbose=False)

    # The optimised x-vector and final merit must agree bit-for-bit
    # (or within negligible numerical noise from any extra work).
    np.testing.assert_array_equal(
        result_a.x, result_b.x,
        err_msg=("telemetry must be observational only: x-vector "
                 "differs between handler-off and handler-on runs."))
    assert float(result_a.merit) == float(result_b.merit), (
        f"telemetry must be observational only: final merit "
        f"differs between handler-off ({result_a.merit}) and "
        f"handler-on ({result_b.merit}) runs.")


# ----------------------------------------------------------------------
# Section 3 -- apply_real_lens_traced telemetry pins
# ----------------------------------------------------------------------


def _build_small_traced_inputs():
    """Build a tiny singlet + N=128 plane-wave input so
    apply_real_lens_traced converges in a couple of seconds on CPU.

    The grid is sized so ``ray_subsample=1`` (Newton at every pixel)
    has enough coarse-grid coverage to satisfy the default
    ``min_coarse_samples_per_aperture=32`` guard while still keeping
    the per-test cost low.
    """
    import lumenairy as la

    wavelength = 1.31e-6
    aperture = 0.5e-3
    prescription = la.make_singlet(
        R1=5e-3, R2=float('inf'), d=1e-3,
        glass='N-BK7', aperture=aperture,
    )
    N = 128
    # Semi-aperture covers ~48 px so the coarse grid satisfies the
    # default min_coarse_samples_per_aperture=32 check at
    # ray_subsample=1.
    dx = aperture / 96
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    semi_ap = aperture / 2
    E_in = np.where(np.sqrt(X * X + Y * Y) <= semi_ap,
                    1.0, 0.0).astype(np.complex128)
    return dict(E_in=E_in, prescription=prescription,
                wavelength=wavelength, dx=dx)


def test_apply_real_lens_traced_default_quiet(caplog):
    """Default-quiet pin: apply_real_lens_traced produces ZERO
    captured log records on the lumenairy logger when no handler is
    attached.
    """
    import lumenairy as la

    kwargs = _build_small_traced_inputs()
    _ = la.apply_real_lens_traced(
        ray_subsample=1, n_workers=1, **kwargs)
    lumenairy_records = [r for r in caplog.records
                         if r.name.startswith('lumenairy')]
    assert len(lumenairy_records) == 0, (
        f"default-quiet contract violated: apply_real_lens_traced "
        f"captured {len(lumenairy_records)} lumenairy log records "
        f"without an explicit handler attachment.")


def test_apply_real_lens_traced_telemetry_active(caplog):
    """Telemetry-active pin: with ``caplog.at_level(logging.INFO,
    logger='lumenairy')``, apply_real_lens_traced emits at least one
    INFO record carrying ``'apply_real_lens_traced: newton iter '`` or
    the entry marker ``'apply_real_lens_traced: entry'``.
    """
    import lumenairy as la

    kwargs = _build_small_traced_inputs()
    with caplog.at_level(logging.INFO, logger='lumenairy'):
        _ = la.apply_real_lens_traced(
            ray_subsample=1, n_workers=1, **kwargs)
    traced_records = [r for r in caplog.records
                      if r.name.startswith('lumenairy')
                      and 'apply_real_lens_traced' in r.getMessage()]
    assert len(traced_records) >= 1, (
        "telemetry-active contract violated: apply_real_lens_traced "
        "emitted no 'apply_real_lens_traced' INFO records.")
    # At least one record should be a per-iteration marker.
    newton_records = [r for r in traced_records
                      if 'newton iter ' in r.getMessage()]
    assert len(newton_records) >= 1, (
        f"telemetry-active contract violated: apply_real_lens_traced "
        f"emitted no per-Newton-iteration markers.  Captured "
        f"messages: {[r.getMessage() for r in traced_records]}")


# ----------------------------------------------------------------------
# Section 4 -- monte_carlo_tolerancing telemetry pins
# ----------------------------------------------------------------------


def _build_small_mc_inputs():
    """Build a small singlet + small pupil + tiny perturbation spec so
    the Monte-Carlo trial loop runs in a couple of seconds.
    """
    import lumenairy as la

    wavelength = 587.56e-9
    aperture = 12e-3
    prescription = la.make_singlet(
        R1=51.5e-3, R2=float('inf'), d=4e-3,
        glass='N-BK7', aperture=aperture,
    )
    surfs = la.surfaces_from_prescription(prescription)
    _, _efl, bfl, _ = la.system_abcd(surfs, wavelength=wavelength)
    focal_length = bfl

    N = 64
    dx = aperture / 32
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    semi_ap = aperture / 2
    E_source = np.where(np.sqrt(X * X + Y * Y) <= semi_ap,
                        1.0, 0.0).astype(np.complex128)

    perturbation_spec = [
        {'surface_index': 0, 'decenter_std': 20e-6,
         'tilt_std': 0.2e-3, 'form_error_rms': 0.0},
    ]
    return dict(
        prescription=prescription,
        wavelength=wavelength,
        N=N,
        dx=dx,
        E_source=E_source,
        perturbation_spec=perturbation_spec,
        focal_length=focal_length,
        aperture=aperture,
        z_scan_n=5,
        verbose=False,
    )


def test_monte_carlo_tolerancing_default_quiet(caplog):
    """Default-quiet pin: monte_carlo_tolerancing produces ZERO
    captured log records on the lumenairy logger when no handler is
    attached.
    """
    import lumenairy as la

    kwargs = _build_small_mc_inputs()
    _ = la.monte_carlo_tolerancing(n_trials=3, seed=0, **kwargs)
    lumenairy_records = [r for r in caplog.records
                         if r.name.startswith('lumenairy')]
    assert len(lumenairy_records) == 0, (
        f"default-quiet contract violated: monte_carlo_tolerancing "
        f"captured {len(lumenairy_records)} lumenairy log records "
        f"without an explicit handler attachment.")


def test_monte_carlo_tolerancing_telemetry_active(caplog):
    """Telemetry-active pin: with ``caplog.at_level(logging.INFO,
    logger='lumenairy')``, monte_carlo_tolerancing emits per-trial
    INFO records carrying ``'monte_carlo_tolerancing: trial '``.
    """
    import lumenairy as la

    kwargs = _build_small_mc_inputs()
    with caplog.at_level(logging.INFO, logger='lumenairy'):
        _ = la.monte_carlo_tolerancing(n_trials=3, seed=0, **kwargs)
    trial_records = [r for r in caplog.records
                     if r.name.startswith('lumenairy')
                     and 'monte_carlo_tolerancing: trial ' in r.getMessage()]
    assert len(trial_records) >= 1, (
        "telemetry-active contract violated: monte_carlo_tolerancing "
        "emitted no 'monte_carlo_tolerancing: trial' INFO records.")
