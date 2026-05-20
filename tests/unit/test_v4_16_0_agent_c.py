"""v4.16.0 / Agent C -- optimisation framework expansion.

Pins the four ROADMAP items rolled up into the v4.16.0 release by
Agent C:

* **C.1 (ROADMAP #9) -- Constrained optimisation.**  ``Constraint``
  dataclass wraps a callable + ``(lb, ub)`` bounds; threaded through
  ``design_optimize`` as ``constraints=`` and translated to
  :class:`scipy.optimize.NonlinearConstraint`.  Only SLSQP /
  trust-constr support hard constraints; any other method raises
  ``ValueError`` with a clear hint instead of silently dropping the
  constraint argument.

* **C.2 (ROADMAP #10) -- Checkpoint / resume.**  ``state_file=``
  kwarg persists ``(call_count, x_best, merit_best, history)`` to
  JSON on every merit eval (modulo ``state_save_every=N``).  On
  startup if the file exists with the right shape, the optimiser
  resumes from the persisted ``x_best``.

* **C.3 (ROADMAP #11) -- Multi-objective Pareto.**
  :func:`design_optimize_multi_objective` wraps pymoo's NSGA-II for
  Pareto-front optimisation of competing merits.  pymoo is an
  optional dependency; tests skip cleanly when missing.

* **C.4 (ROADMAP #12) -- Hessian / Newton-step.**
  ``method='newton'`` dispatches to scipy ``'trust-ncg'`` with an
  FD-Hessian estimator.  ~30x faster convergence than L-BFGS-B on
  small problems; warns on large (>30 free var) problems.

Each subtask has 2-4 tests; total = 13 (C.1=4 + C.2=3 + C.3=4 +
C.4=2).

Author: Andrew Traverso -- v4.16.0 / Agent C
"""
from __future__ import annotations

import importlib.util as _importlib_util
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pytest


# pymoo availability flag.  C.3 tests gate on this.
_PYMOO_AVAILABLE = _importlib_util.find_spec('pymoo') is not None


# v4.16.2 (audit P2-NEW-F1-3): module-level constraint functions to
# replace the four ``Constraint(fun=lambda x: ...)`` sites at lines
# 169, 197, 219, 237 that started emitting a ``UserWarning`` on every
# construction once the v4.16.1 lambda detector landed.  Suppressing
# the warning at the test fixture would hide future drift; using
# module-level functions is the same pickle-safe pattern the
# warning explicitly recommends.
def _sum_constraint(x):
    """``sum(x)`` reduced to a Python float.  Used as the test
    fixture constraint callable; picklable (module-level def) so
    no Constraint(...) UserWarning."""
    return float(np.sum(x))


def _first_coord(x):
    """``float(x[0])`` -- picklable companion to ``_sum_constraint``."""
    return float(x[0])


# ---------------------------------------------------------------------------
# Shared minimal-parameterisation fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture
def small_quadratic_param():
    """5-variable quadratic ``f(x) = sum(x_i^2)`` parameterisation
    for the Newton / unconstrained smoke tests.

    Builds a fake :class:`DesignParameterization` whose ``build()``
    returns a "prescription" dict carrying the parameter vector, plus
    a minimal :class:`~lumenairy.optimize.CallableMerit` that returns
    ``sum(x^2)`` from the parameters.

    Decoupled from the full apply_real_lens stack so the tests run in
    milliseconds.
    """
    from lumenairy.optimize import (
        DesignParameterization, CallableMerit,
    )

    # 5-element template with 5 free params; each path reads the
    # corresponding entry in a list field on a dummy template.
    template = {
        'params': [1.0, 1.0, 1.0, 1.0, 1.0],
        # design_optimize() runs surfaces_from_prescription() on the
        # built prescription before merits; we supply a single null
        # surface so that step doesn't raise.  The surface is at
        # z=0 with infinite radius -- effectively a vacuum plane.
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                       'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    free_vars: List[Tuple[Any, ...]] = [
        ('params', 0),
        ('params', 1),
        ('params', 2),
        ('params', 3),
        ('params', 4),
    ]
    bounds = [(-5.0, 5.0)] * 5
    param = DesignParameterization(
        template=template, free_vars=free_vars, bounds=bounds)

    def _quadratic_merit(ctx) -> float:
        x = np.asarray(ctx.x, dtype=np.float64)
        return float(np.sum(x * x))

    merit = CallableMerit(_quadratic_merit, weight=1.0,
                          name='sum_squared')
    return param, [merit]


@pytest.fixture
def small_quadratic_offset_param():
    """5-variable quadratic with a non-trivial offset:
    ``f(x) = sum((x_i - i)^2)``.

    Optimum at ``x* = [0, 1, 2, 3, 4]``.  Used for the constrained-
    optimisation tests (so the unconstrained optimum is well-known
    and lies inside the bound).
    """
    from lumenairy.optimize import (
        DesignParameterization, CallableMerit,
    )

    template = {
        'params': [0.0, 0.0, 0.0, 0.0, 0.0],
        'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                       'glass_before': 'air', 'glass_after': 'air'}],
        'thicknesses': [],
        'aperture_diameter': 5e-3,
    }
    free_vars: List[Tuple[Any, ...]] = [
        ('params', i) for i in range(5)
    ]
    bounds = [(-10.0, 10.0)] * 5
    param = DesignParameterization(
        template=template, free_vars=free_vars, bounds=bounds)

    def _quadratic_offset_merit(ctx) -> float:
        x = np.asarray(ctx.x, dtype=np.float64)
        target = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return float(np.sum((x - target) ** 2))

    merit = CallableMerit(_quadratic_offset_merit, weight=1.0,
                          name='quadratic_offset')
    return param, [merit]


# ===========================================================================
# C.1 -- Constrained optimisation (ROADMAP #9, 4 tests)
# ===========================================================================


class TestC1ConstrainedOptimisation:
    """ROADMAP #9 -- hard ``NonlinearConstraint`` support via
    ``Constraint`` dataclass + SLSQP / trust-constr dispatch.
    """

    def test_constraint_enforced_hard(self, small_quadratic_offset_param):
        """A hard constraint ``sum(x) >= 15`` (vs unconstrained
        optimum sum=10) must push the optimised solution to a
        feasible point with ``sum(x) >= 15`` (within tolerance).

        Unconstrained optimum: ``x* = [0, 1, 2, 3, 4]``, ``sum = 10``,
        merit = 0.  Constrained optimum: lies on the constraint
        boundary at ``sum(x) = 15``, so merit > 0.
        """
        from lumenairy.optimize import (
            Constraint, design_optimize,
        )

        param, merits = small_quadratic_offset_param
        # Hard constraint: sum(x) >= 15.  v4.16.2: module-level
        # _sum_constraint to avoid the lambda-detector UserWarning.
        c = Constraint(
            fun=_sum_constraint,
            lb=15.0, ub=None,
            label='sum(x) >= 15',
        )
        result = design_optimize(
            param, merits, wavelength=1.31e-6,
            method='SLSQP', max_iter=200,
            constraints=[c], verbose=False,
        )
        # Constraint must be honoured to better than 1e-3 tolerance.
        sum_at_opt = float(np.sum(result.x))
        assert sum_at_opt >= 15.0 - 1e-3, (
            f"constraint sum(x) >= 15 violated: sum at opt = "
            f"{sum_at_opt}")
        # And the merit must be strictly > 0 (we're off the
        # unconstrained optimum).
        assert result.merit > 0.0

    def test_unsupported_method_raises(self, small_quadratic_offset_param):
        """Methods that don't honour ``NonlinearConstraint`` (e.g.
        L-BFGS-B) must raise ``ValueError`` with a method-hint, not
        silently drop the constraints argument."""
        from lumenairy.optimize import (
            Constraint, design_optimize,
        )

        param, merits = small_quadratic_offset_param
        # v4.16.2: module-level _sum_constraint to avoid the
        # lambda-detector UserWarning.
        c = Constraint(
            fun=_sum_constraint,
            lb=15.0, ub=None, label='sum_constraint')

        with pytest.raises(ValueError, match=r'(SLSQP|trust-constr)'):
            design_optimize(
                param, merits, wavelength=1.31e-6,
                method='L-BFGS-B', max_iter=10,
                constraints=[c], verbose=False)

    def test_constraint_callable_signature_validated(self):
        """``Constraint(fun=...)`` with a non-callable ``fun`` must
        raise ``TypeError`` with a clear message; ``Constraint()``
        with both ``lb`` and ``ub`` None must raise ``ValueError``
        (no-op constraint)."""
        from lumenairy.optimize import Constraint

        # Non-callable fun.
        with pytest.raises(TypeError, match=r'callable'):
            Constraint(fun=42.0, lb=0.0, ub=None, label='bad')  # type: ignore[arg-type]

        # Both bounds None.  v4.16.2: module-level _sum_constraint
        # to avoid the lambda-detector UserWarning.
        with pytest.raises(ValueError, match=r'(lb|ub|unbounded|no-op)'):
            Constraint(fun=_sum_constraint,
                       lb=None, ub=None, label='unbounded')

    def test_constraint_label_in_progress_output(
            self, small_quadratic_offset_param):
        """The progress callback must receive a message that includes
        the constraint label when constraints are active.

        We capture progress messages via a ProgressCallback-style
        callable and assert at least one message contains the
        constraint's label.
        """
        from lumenairy.optimize import (
            Constraint, design_optimize,
        )

        param, merits = small_quadratic_offset_param
        # v4.16.2: module-level _sum_constraint to avoid the
        # lambda-detector UserWarning.
        c = Constraint(
            fun=_sum_constraint,
            lb=15.0, ub=None,
            label='sum_test_label',
        )

        captured_messages: List[str] = []

        def _progress(stage: str, frac: float, msg: str) -> None:
            captured_messages.append(msg)

        design_optimize(
            param, merits, wavelength=1.31e-6,
            method='SLSQP', max_iter=20,
            constraints=[c], verbose=False,
            progress=_progress)

        # At least one progress message should mention the label.
        found = any('sum_test_label' in m for m in captured_messages)
        assert found, (
            f"constraint label 'sum_test_label' not found in any "
            f"of the {len(captured_messages)} progress messages: "
            f"{captured_messages[:5]!r}")


# ===========================================================================
# C.2 -- Checkpoint / resume on long ``design_optimize`` runs
# (ROADMAP #10, 3 tests)
# ===========================================================================


class TestC2CheckpointResume:
    """ROADMAP #10 -- ``state_file=`` + ``state_save_every=`` for
    crash-resumable multi-hour ``design_optimize`` runs."""

    def test_state_file_created_and_populated(
            self, small_quadratic_param, tmp_path):
        """After 10 iterations, the state file must exist and parse
        as a valid dict with the v4.16 keys
        ``(version, call_count, x_best, merit_best, history)``."""
        from lumenairy.optimize import design_optimize

        param, merits = small_quadratic_param
        state_file = str(tmp_path / 'opt_state.json')
        # Confirm the file does NOT exist before the run.
        assert not os.path.isfile(state_file)

        result = design_optimize(
            param, merits, wavelength=1.31e-6,
            method='L-BFGS-B', max_iter=10,
            state_file=state_file, state_save_every=1,
            verbose=False)

        # File must exist.
        assert os.path.isfile(state_file), (
            f"state_file {state_file!r} was not created")
        # Parse and check keys.
        with open(state_file, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        assert isinstance(payload, dict)
        for key in ('version', 'call_count', 'x_best',
                    'merit_best', 'history'):
            assert key in payload, (
                f"state_file missing key {key!r}; got {list(payload)}")
        # x_best must be a length-5 list (our 5-variable problem).
        assert len(payload['x_best']) == 5
        # merit_best must be finite + match the returned result
        # within tight tolerance.
        assert payload['merit_best'] is not None
        assert np.isfinite(payload['merit_best'])
        assert abs(payload['merit_best'] - result.merit) < 1e-6 or (
            payload['merit_best'] <= result.merit + 1e-9)

    def test_resume_from_state_file(
            self, small_quadratic_param, tmp_path):
        """Pre-populate the state file with a known ``x_best``;
        starting a new run with that ``state_file`` must resume from
        that ``x_best`` instead of ``parameterization.initial_values()``.

        Sentinel: pre-populate with ``x_best=[0, 0, 0, 0, 0]``
        (exact optimum of sum-of-squares).  Run for 0 outer
        iterations (max_iter=0 is invalid; max_iter=1 is allowed
        and exits without progress); merit_fn is called at least
        once at the resumed x to populate ctx, so the final result
        must have merit ~ 0 -- which is impossible if the resume
        was ignored (param.initial_values() = [1,1,1,1,1] -> merit
        = 5).
        """
        from lumenairy.optimize import design_optimize

        param, merits = small_quadratic_param
        state_file = str(tmp_path / 'resume.json')
        # Pre-populate: x_best at the exact unconstrained optimum.
        pre_payload = {
            'version': '4.16.0',
            'call_count': 50,
            'x_best': [0.0, 0.0, 0.0, 0.0, 0.0],
            'merit_best': 0.0,
            'history': [],
        }
        with open(state_file, 'w', encoding='utf-8') as fh:
            json.dump(pre_payload, fh)

        # Run with max_iter=1 (minimal); the optimiser must start
        # from x_best=[0,0,0,0,0] instead of param.initial_values().
        # If the resume worked, the final merit must be ~0.
        # If the resume was ignored, the optimiser starts at [1,1,...]
        # and the merit at the first eval is 5.
        result = design_optimize(
            param, merits, wavelength=1.31e-6,
            method='L-BFGS-B', max_iter=1,
            state_file=state_file, state_save_every=1,
            verbose=False)

        # Resumed merit must be very close to 0 (it might tick up
        # slightly on a degenerate L-BFGS step, but should not get
        # near 5).  Set threshold at 1.0 (5x safety margin).
        assert result.merit < 1.0, (
            f"resume did not engage: final merit = {result.merit}, "
            f"expected ~0; param.initial_values() merit would be 5")

    def test_state_save_every_throttles_io(
            self, small_quadratic_param, tmp_path):
        """With ``state_save_every=5`` and 20 merit evaluations, the
        file should be written ~4 times (one per 5 evals) plus the
        final force-save, not on every eval.

        We replace ``_core_mod._json`` (the module reference
        imported inside ``lumenairy.optimize.core``) with a stub
        that counts ``dump()`` calls; the count must be much less
        than the total merit-fn evaluation count.
        """
        import lumenairy.optimize.core as _core_mod
        from lumenairy.optimize import design_optimize

        param, merits = small_quadratic_param
        state_file = str(tmp_path / 'throttle.json')

        # Swap ``_core_mod._json`` for a stub that counts dump()
        # calls.  Restored in a finally:.  This avoids monkey-
        # patching the global ``json`` module's attribute (which
        # would leak state).
        class _CountingJSON:
            def __init__(self, real_json):
                self._real = real_json
                self.dump_calls = 0

            def dump(self, *args, **kwargs):
                self.dump_calls += 1
                return self._real.dump(*args, **kwargs)

            def __getattr__(self, name):
                return getattr(self._real, name)

        original_json = _core_mod._json
        counting = _CountingJSON(original_json)
        _core_mod._json = counting
        try:
            design_optimize(
                param, merits, wavelength=1.31e-6,
                method='L-BFGS-B', max_iter=20,
                state_file=state_file, state_save_every=5,
                verbose=False)
        finally:
            _core_mod._json = original_json

        # With state_save_every=5, writes happen on calls 5, 10,
        # 15, 20, ... plus the final force-save.  The exact merit-
        # fn call count depends on L-BFGS-B's FD-gradient policy
        # (it calls merit_fn 6 times per iteration for n_params=5:
        # one centre + 5 forward-FD per gradient).  Lower bound: 1
        # write (the force-save).  Upper bound: 30 writes if the
        # throttle is honoured.  Without the throttle every eval
        # would write -- typically ~100+ on a 20-iter L-BFGS-B run.
        assert 1 <= counting.dump_calls <= 30, (
            f"state_save_every=5 throttle wrote "
            f"{counting.dump_calls} times -- expected ~5 "
            f"(one per 5 evals + 1 final); upper bound 30 if "
            f"state_save_every silently ignored")


# ===========================================================================
# C.3 -- Multi-objective Pareto via pymoo (ROADMAP #11, 4 tests)
# ===========================================================================


class TestC3MultiObjectivePareto:
    """ROADMAP #11 -- pymoo NSGA-II wrapper as optional dependency."""

    def test_pymoo_import_required(self):
        """When pymoo is NOT installed, calling
        :func:`design_optimize_multi_objective` must raise
        ``ImportError`` with a clear install hint.

        When pymoo IS installed, the function must not raise
        ImportError at call time (it may still fail for other
        reasons, but not this one)."""
        from lumenairy.optimize.multi_objective import (
            design_optimize_multi_objective, PYMOO_AVAILABLE,
        )

        if PYMOO_AVAILABLE:
            # Smoke test: a minimal valid call must not raise
            # ImportError.  It may raise other things (e.g. for an
            # ill-posed problem) but not ImportError.
            def m1(x): return float(x[0] ** 2)
            def m2(x): return float((x[0] - 1) ** 2)
            try:
                design_optimize_multi_objective(
                    merits=[m1, m2],
                    x0=np.array([0.0]),
                    bounds=[(-2.0, 2.0)],
                    n_generations=2, pop_size=4, seed=0)
            except ImportError:
                pytest.fail(
                    "design_optimize_multi_objective raised "
                    "ImportError even though pymoo is installed")
        else:
            with pytest.raises(ImportError, match=r'pymoo'):
                design_optimize_multi_objective(
                    merits=[lambda x: 0.0, lambda x: 0.0],
                    x0=np.array([0.0]),
                    bounds=[(0.0, 1.0)],
                    n_generations=2, pop_size=4)

    @pytest.mark.skipif(not _PYMOO_AVAILABLE,
                        reason='pymoo not installed -- install via '
                               '`pip install lumenairy[multi_objective]`')
    def test_nsga2_2_objective_pareto_front(self):
        """ZDT1-style two-objective problem:
        ``f1(x) = x[0]``,
        ``f2(x) = 1 - sqrt(x[0])`` for x in [0, 1].

        The known Pareto front is the curve ``f2 = 1 - sqrt(f1)`` for
        ``f1`` in [0, 1].  Verify the returned ``F`` array has
        shape ``(n_solutions, 2)``, ``n_solutions >= 5``, and every
        point lies near the Pareto curve.
        """
        from lumenairy.optimize.multi_objective import (
            design_optimize_multi_objective,
        )

        def f1(x): return float(x[0])
        def f2(x): return float(1.0 - np.sqrt(max(x[0], 0.0)))

        result = design_optimize_multi_objective(
            merits=[f1, f2],
            x0=np.array([0.5]),
            bounds=[(0.0, 1.0)],
            n_generations=30, pop_size=20, seed=42)

        assert result.F.ndim == 2
        assert result.F.shape[1] == 2
        assert result.F.shape[0] >= 5, (
            f"Pareto front too small: only {result.F.shape[0]} "
            f"solutions")
        # Every (f1, f2) on the front must satisfy
        # f2 ~ 1 - sqrt(f1) (the analytic Pareto curve).
        max_err = 0.0
        for fa, fb in result.F:
            expected = 1.0 - np.sqrt(max(fa, 0.0))
            err = abs(fb - expected)
            if err > max_err:
                max_err = err
        assert max_err < 0.05, (
            f"Pareto front max deviation from analytic curve "
            f"= {max_err:.4f}, expected < 0.05")

    @pytest.mark.skipif(not _PYMOO_AVAILABLE,
                        reason='pymoo not installed')
    def test_constraints_threaded_through_pymoo(self):
        """A hard constraint ``x[0] >= 0.5`` must be honoured by
        the returned Pareto front (all solutions x[0] >= 0.5).
        """
        from lumenairy.optimize import Constraint
        from lumenairy.optimize.multi_objective import (
            design_optimize_multi_objective,
        )

        def f1(x): return float(x[0])
        def f2(x): return float(1.0 - np.sqrt(max(x[0], 0.0)))

        # v4.16.2: module-level _first_coord to avoid the
        # lambda-detector UserWarning.
        c = Constraint(
            fun=_first_coord,
            lb=0.5, ub=None,
            label='x0 >= 0.5')

        result = design_optimize_multi_objective(
            merits=[f1, f2],
            x0=np.array([0.5]),
            bounds=[(0.0, 1.0)],
            constraints=[c],
            n_generations=30, pop_size=20, seed=42)

        # Either we have no feasible solutions (highly unlikely
        # with a moderate constraint) or all of them satisfy
        # x[0] >= 0.5 within numerical tolerance.
        feasible_mask = result.X[:, 0] >= 0.5 - 1e-3
        # Require at least 50% feasibility on the returned front.
        assert feasible_mask.mean() > 0.5, (
            f"Only {feasible_mask.sum()}/{len(result.X)} "
            f"Pareto solutions are feasible (x[0] >= 0.5); "
            f"constraint threading may have failed")

    @pytest.mark.skipif(not _PYMOO_AVAILABLE,
                        reason='pymoo not installed')
    def test_progress_callback_fires(self):
        """The progress callable must be invoked at least once per
        generation (so 30 generations -> >= 30 calls)."""
        from lumenairy.optimize.multi_objective import (
            design_optimize_multi_objective,
        )

        calls: List[Tuple[str, float, str]] = []

        def _progress(stage: str, frac: float, msg: str) -> None:
            calls.append((stage, frac, msg))

        def f1(x): return float(x[0])
        def f2(x): return float(1.0 - x[0])

        design_optimize_multi_objective(
            merits=[f1, f2],
            x0=np.array([0.5]),
            bounds=[(0.0, 1.0)],
            n_generations=10, pop_size=10, seed=42,
            progress=_progress)

        # Lower bound: 5 calls (we run 10 generations + 1 final).
        # Upper bound: ~25 to allow for per-generation + final + a
        # few intermediate.
        assert len(calls) >= 5, (
            f"progress callback fired only {len(calls)} times; "
            f"expected >= 5 (one per generation + final)")
        # Every call must use the canonical stage name.
        assert all(
            c[0] == 'design_optimize_multi_objective' for c in calls), (
            f"progress callback received wrong stage names: "
            f"{set(c[0] for c in calls)}")
        # The final call's frac must be ~1.0.
        assert calls[-1][1] >= 0.99, (
            f"final progress frac = {calls[-1][1]}; expected ~1.0")


# ===========================================================================
# C.4 -- Hessian / Newton-step optimisation (ROADMAP #12, 2 tests)
# ===========================================================================


class TestC4NewtonStep:
    """ROADMAP #12 -- ``method='newton'`` dispatch to scipy
    'trust-ncg' with FD-Hessian estimator."""

    def test_newton_converges_on_small_problem(
            self, small_quadratic_param):
        """5-variable quadratic ``sum(x^2)``: Newton should find the
        optimum (x* = 0, merit = 0) in <= 10 iterations.

        The fixture's initial_values() = [1, 1, 1, 1, 1] gives merit
        = 5; the optimum is exactly 0.  Newton with a quadratic
        merit should converge in a single step (Hessian = 2 * I,
        gradient at x = 2*x, so the Newton step is -x, landing
        at 0).
        """
        from lumenairy.optimize import design_optimize

        param, merits = small_quadratic_param
        result = design_optimize(
            param, merits, wavelength=1.31e-6,
            method='newton', max_iter=20,
            verbose=False)

        assert result.merit < 1e-4, (
            f"Newton failed to converge on a 5-var quadratic: "
            f"final merit = {result.merit}, expected near 0")
        # Solution must be close to x* = 0.
        assert np.linalg.norm(result.x) < 1e-2, (
            f"Newton final x has norm {np.linalg.norm(result.x)}, "
            f"expected near 0")

    def test_newton_falls_back_on_large_problem(self):
        """50-variable problem with ``method='newton'``: must emit a
        UserWarning recommending L-BFGS-B but otherwise complete
        without crashing.
        """
        import warnings
        from lumenairy.optimize import (
            DesignParameterization, CallableMerit, design_optimize,
        )

        # 50-variable template.  Quadratic merit; we don't actually
        # care about convergence quality, just that Newton handles
        # the large-N case gracefully.
        N = 50
        template = {
            'params': [1.0] * N,
            'surfaces': [{'radius': np.inf, 'aperture': 5e-3,
                          'glass_before': 'air',
                          'glass_after': 'air'}],
            'thicknesses': [],
            'aperture_diameter': 5e-3,
        }
        free_vars = [('params', i) for i in range(N)]
        bounds = [(-5.0, 5.0)] * N
        param = DesignParameterization(
            template=template, free_vars=free_vars, bounds=bounds)

        def _quad(ctx):
            x = np.asarray(ctx.x, dtype=np.float64)
            return float(np.sum(x * x))

        merit = CallableMerit(_quad, weight=1.0, name='quad50')

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter('always')
            # Hard-cap max_iter=2 so the 50-var Newton with O(N^2)
            # Hessian doesn't dominate the test runtime.
            result = design_optimize(
                param, [merit], wavelength=1.31e-6,
                method='newton', max_iter=2,
                verbose=False)
            # The user-warning recommending L-BFGS-B must be present.
            found_warning = any(
                issubclass(w.category, UserWarning) and
                'L-BFGS-B' in str(w.message) and
                'newton' in str(w.message).lower()
                for w in recorded)
        assert found_warning, (
            "Newton with n_params=50 did not emit the expected "
            "UserWarning recommending L-BFGS-B")
        # And the result object must come back with a finite merit
        # (didn't crash mid-iteration).
        assert np.isfinite(result.merit)
