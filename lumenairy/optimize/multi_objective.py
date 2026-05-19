"""Multi-objective Pareto optimisation via pymoo NSGA-II (v4.16 #11).

This module wraps `pymoo <https://pymoo.org>`_'s NSGA-II algorithm for
multi-objective optimisation of optical-design merits.

When the merit landscape has multiple competing objectives with no
``a priori`` weight balance (e.g. "minimise spot size AND match focal
length AND keep weight below X"), collapsing them into a scalar
:class:`~lumenairy.optimize.CompositeMerit` requires picking weights
that bias the search.  Pareto-front optimisation instead returns the
**set of non-dominated solutions**: every member is at least as good
as every other in some objective and strictly better in at least one.
The user then picks the trade-off they prefer.

pymoo is an **optional dependency**.  Install via:

.. code-block:: bash

    pip install lumenairy[multi_objective]   # or
    pip install pymoo>=0.6

The module imports unconditionally (so callers can write
``from lumenairy.optimize.multi_objective import ...`` regardless of
whether pymoo is installed); the actual pymoo import is deferred to
:func:`design_optimize_multi_objective` and raises a clear
``ImportError`` with the install hint when pymoo is missing.

Example
-------

.. code-block:: python

    import lumenairy as la
    from lumenairy.optimize.multi_objective import (
        design_optimize_multi_objective,
    )

    # Build a parameterization.
    template = la.thorlabs_lens('AC254-100-C')
    template['aperture_diameter'] = 10e-3
    param = la.DesignParameterization(template,
        free_vars=[('surfaces', 0, 'radius'),
                   ('surfaces', 1, 'radius')],
        bounds=[(50e-3, 80e-3), (-60e-3, -30e-3)])

    # Define two competing merits.
    def spot_size(x):
        # ... evaluate spot size at the current x ...
        return spot_rms

    def focal_length_err(x):
        # ... evaluate |EFL - target| at the current x ...
        return abs(efl - 100e-3)

    result = design_optimize_multi_objective(
        merits=[spot_size, focal_length_err],
        x0=param.initial_values(),
        bounds=param.bounds,
        n_generations=80, pop_size=40, seed=42)

    # result.X     -- (n_solutions, n_params) Pareto-front parameters
    # result.F     -- (n_solutions, n_merits) Pareto-front merit values
    # result.pymoo_result -- raw pymoo Result for advanced inspection
"""
from __future__ import annotations

import importlib.util as _importlib_util
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np


# pymoo availability flag.  Cheap module-level check so callers can
# ``if PYMOO_AVAILABLE: ...`` without paying the import cost.
PYMOO_AVAILABLE = _importlib_util.find_spec('pymoo') is not None


_PYMOO_INSTALL_HINT = (
    "design_optimize_multi_objective requires pymoo (optional "
    "dependency).  Install via 'pip install "
    "lumenairy[multi_objective]' or 'pip install pymoo>=0.6'."
)


@dataclass
class ParetoResult:
    """Result of a Pareto-front optimisation.

    Attributes
    ----------
    X : ndarray, shape (n_solutions, n_params)
        Parameter vectors of the Pareto-optimal solutions.
    F : ndarray, shape (n_solutions, n_merits)
        Merit values at each Pareto-optimal solution.  Each row gives
        the merit values for the corresponding row in ``X``.
    n_evaluations : int
        Total number of merit-function evaluations during the search.
    n_generations : int
        Number of NSGA-II generations actually run.
    pymoo_result : Any
        Raw pymoo ``Result`` object for advanced inspection
        (algorithm-internal diagnostics, full population history,
        etc.).  ``None`` when pymoo wasn't actually used (defensive).
    constraint_violations : ndarray, optional
        Per-solution maximum constraint violation (``CV``).  ``None``
        when no constraints were supplied.  Solutions with ``CV > 0``
        violate at least one constraint; pymoo's default
        feasibility-first ranking pushes them to the back of the
        Pareto set.
    """

    X: np.ndarray
    F: np.ndarray
    n_evaluations: int = 0
    n_generations: int = 0
    pymoo_result: Any = None
    constraint_violations: Optional[np.ndarray] = None


def _check_pymoo() -> None:
    """Raise a clear ImportError if pymoo isn't installed."""
    if not PYMOO_AVAILABLE:
        raise ImportError(_PYMOO_INSTALL_HINT)


def design_optimize_multi_objective(
    merits: Sequence[Callable],
    x0: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
    *,
    n_generations: int = 100,
    pop_size: int = 50,
    seed: Optional[int] = None,
    constraints: Optional[Sequence[Any]] = None,
    progress: Optional[Callable] = None,
    verbose: bool = False,
) -> ParetoResult:
    """Find the Pareto front of a multi-objective design problem.

    Wraps pymoo's NSGA-II algorithm
    (:class:`pymoo.algorithms.moo.nsga2.NSGA2`) over a list of merit
    callables, returning the non-dominated set in a :class:`ParetoResult`.

    Parameters
    ----------
    merits : sequence of callable
        Each ``m(x) -> float`` maps the current parameter vector to a
        scalar merit value to MINIMISE.  Use the negation pattern
        (``lambda x: -gain(x)``) for maximisation objectives.  Must
        contain at least 2 entries (single-objective optimisation
        should use :func:`~lumenairy.optimize.design_optimize`
        instead).
    x0 : array-like
        Initial parameter vector.  pymoo doesn't actually use a single
        starting point (it samples a population) but ``x0`` is used to
        infer ``n_params`` and as a sanity check against ``bounds``.
    bounds : sequence of (float, float)
        ``(lower, upper)`` bound per parameter.  Required by NSGA-II
        (pymoo uses the bounds to seed the initial population and to
        clip mutation steps).  Length must match ``len(x0)``.
    n_generations : int, default 100
        NSGA-II generation count.  Doubling this roughly doubles the
        Pareto-front density on convex problems; non-convex problems
        may need 200-500 generations to converge.
    pop_size : int, default 50
        NSGA-II population size.  pymoo recommends ``pop_size``
        >= ``4 * n_params`` for a stable Pareto front; the default
        50 covers up to ~12 free variables.
    seed : int, optional
        Random seed for reproducibility.  pymoo's NSGA-II is
        stochastic; setting ``seed`` makes the search deterministic.
    constraints : sequence of Constraint, optional
        Hard constraints applied via pymoo's ``ElementwiseProblem``
        ``_evaluate`` ``G`` (less-than-zero) array.  Each
        :class:`~lumenairy.optimize.Constraint` is translated to a
        pymoo constraint of the form ``g_lb(x) = lb - f(x) <= 0`` and
        / or ``g_ub(x) = f(x) - ub <= 0``.  Solutions violating any
        constraint are penalised by pymoo's feasibility-first
        ranking.
    progress : callable, optional
        ``progress(stage, frac, msg)`` invoked per generation, with
        ``stage='design_optimize_multi_objective'`` and
        ``frac=gen / n_generations``.  Matches the
        :class:`~lumenairy.progress.ProgressCallback` protocol.
    verbose : bool, default False
        Forwarded to pymoo's ``minimize(verbose=...)``.

    Returns
    -------
    ParetoResult

    Raises
    ------
    ImportError
        If pymoo isn't installed.  Install via
        ``pip install lumenairy[multi_objective]``.
    ValueError
        If ``len(merits) < 2`` (use single-objective
        ``design_optimize`` for one merit), if ``len(bounds) !=
        len(x0)``, or if any bound interval has ``lb >= ub``.

    Notes
    -----
    pymoo's NSGA-II uses simulated binary crossover + polynomial
    mutation by default.  These defaults work well for smooth real-
    valued bounded problems (which covers most optical-design merit
    landscapes); for discontinuous merits (Seidel coefficients near
    a degenerate prescription) consider increasing ``pop_size`` and
    ``n_generations``.

    The pymoo result's ``F`` array has shape ``(n_solutions,
    n_merits)`` where ``n_solutions <= pop_size`` (sometimes fewer if
    pymoo culls dominated solutions during the final ranking).
    """
    _check_pymoo()

    if len(merits) < 2:
        raise ValueError(
            f"design_optimize_multi_objective requires >=2 merit "
            f"callables (got {len(merits)}).  Use "
            f"lumenairy.optimize.design_optimize for single-objective "
            f"optimisation.")

    x0_arr = np.asarray(x0, dtype=np.float64)
    n_params = x0_arr.size
    if len(bounds) != n_params:
        raise ValueError(
            f"design_optimize_multi_objective: bounds length "
            f"({len(bounds)}) must match x0 length ({n_params}).")
    lb = np.empty(n_params, dtype=np.float64)
    ub = np.empty(n_params, dtype=np.float64)
    for i, b in enumerate(bounds):
        if b is None or len(b) != 2:
            raise ValueError(
                f"design_optimize_multi_objective: bounds[{i}] must "
                f"be a (lb, ub) pair, got {b!r}.  NSGA-II requires "
                f"finite bounds on every variable.")
        lo, hi = float(b[0]), float(b[1])
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise ValueError(
                f"design_optimize_multi_objective: bounds[{i}] = "
                f"({lo}, {hi}) must be finite.")
        if lo >= hi:
            raise ValueError(
                f"design_optimize_multi_objective: bounds[{i}] = "
                f"({lo}, {hi}) must have lb < ub.")
        lb[i] = lo
        ub[i] = hi

    # Translate Constraint instances to (g_lb, g_ub) pairs.
    cons = list(constraints or ())
    # Each Constraint contributes 0, 1, or 2 pymoo constraints (one
    # for lb, one for ub).  Build a flat list of g(x)<=0 callables.
    g_callables: List[Callable] = []
    g_labels: List[str] = []
    for ci, c in enumerate(cons):
        if not callable(getattr(c, 'fun', None)):
            raise TypeError(
                f"design_optimize_multi_objective: constraints[{ci}] "
                f"must be a Constraint-like with a callable .fun "
                f"attribute, got {type(c).__name__}.")
        c_lb = getattr(c, 'lb', None)
        c_ub = getattr(c, 'ub', None)
        c_fun = c.fun
        c_lbl = getattr(c, 'label', '') or f'constraint[{ci}]'
        if c_lb is not None:
            _lb_val = float(c_lb)
            g_callables.append(lambda xv, _f=c_fun, _v=_lb_val: _v - float(_f(xv)))
            g_labels.append(f'{c_lbl} (lb)')
        if c_ub is not None:
            _ub_val = float(c_ub)
            g_callables.append(lambda xv, _f=c_fun, _v=_ub_val: float(_f(xv)) - _v)
            g_labels.append(f'{c_lbl} (ub)')
    n_constr = len(g_callables)

    # Build the pymoo ElementwiseProblem subclass.
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.optimize import minimize as pymoo_minimize

    eval_counter = [0]

    class _LumenAiryMOOProblem(ElementwiseProblem):
        def __init__(_self):
            super().__init__(
                n_var=int(n_params),
                n_obj=int(len(merits)),
                n_ieq_constr=int(n_constr),
                xl=lb,
                xu=ub,
            )

        def _evaluate(_self, x, out, *args, **kwargs):
            # pymoo passes ``x`` as a 1-D ndarray of shape (n_var,).
            eval_counter[0] += 1
            F_row = np.empty(len(merits), dtype=np.float64)
            for j, m in enumerate(merits):
                try:
                    F_row[j] = float(m(x))
                except (TypeError, ValueError, RuntimeError,
                        ZeroDivisionError, OverflowError) as _exc:
                    # Penalise degenerate evaluations with a large but
                    # finite merit so pymoo's ranking dominates them
                    # naturally.
                    F_row[j] = 1e18
            out['F'] = F_row
            if n_constr > 0:
                G_row = np.empty(n_constr, dtype=np.float64)
                for k, g in enumerate(g_callables):
                    try:
                        G_row[k] = float(g(x))
                    except (TypeError, ValueError, RuntimeError,
                            ZeroDivisionError, OverflowError):
                        G_row[k] = 1e18
                out['G'] = G_row

    # Per-generation progress callback (pymoo passes the algorithm
    # instance; we extract ``n_gen`` from it).
    if progress is not None:
        from ..progress import call_progress as _call_progress

        class _ProgressCallback:
            def __init__(_self):
                _self._n_called = 0

            def __call__(_self, algorithm):
                _self._n_called += 1
                gen = int(getattr(algorithm, 'n_gen', _self._n_called))
                frac = float(gen) / float(max(n_generations, 1))
                frac = max(0.0, min(0.99, frac))
                _call_progress(
                    progress,
                    'design_optimize_multi_objective',
                    frac,
                    f'gen {gen}/{n_generations}: '
                    f'{eval_counter[0]} evals')
        callback = _ProgressCallback()
    else:
        callback = None

    problem = _LumenAiryMOOProblem()
    algorithm = NSGA2(pop_size=int(pop_size))

    pymoo_kwargs: dict = {
        'seed': seed,
        'verbose': bool(verbose),
        'save_history': False,
    }
    if callback is not None:
        pymoo_kwargs['callback'] = callback

    pymoo_res = pymoo_minimize(
        problem, algorithm,
        ('n_gen', int(n_generations)),
        **pymoo_kwargs,
    )

    # pymoo's Result.X / Result.F can be 1-D when the Pareto set
    # collapses to a single solution.  Normalise to 2-D.
    X = np.asarray(pymoo_res.X, dtype=np.float64)
    F = np.asarray(pymoo_res.F, dtype=np.float64)
    if X.ndim == 1:
        X = X[None, :]
    if F.ndim == 1:
        F = F[None, :]

    cv = getattr(pymoo_res, 'CV', None)
    if cv is not None:
        cv = np.asarray(cv, dtype=np.float64)
        if cv.ndim == 0:
            cv = cv[None]
        elif cv.ndim == 2 and cv.shape[1] == 1:
            cv = cv[:, 0]

    if progress is not None:
        from ..progress import call_progress as _call_progress
        _call_progress(
            progress, 'design_optimize_multi_objective', 1.0,
            f'done: {X.shape[0]} Pareto solutions, '
            f'{eval_counter[0]} evals')

    return ParetoResult(
        X=X,
        F=F,
        n_evaluations=int(eval_counter[0]),
        n_generations=int(n_generations),
        pymoo_result=pymoo_res,
        constraint_violations=cv,
    )


__all__ = [
    'ParetoResult',
    'design_optimize_multi_objective',
    'PYMOO_AVAILABLE',
]
