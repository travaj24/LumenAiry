"""
lumenairy.optimize.parameterizations -- map flat parameter vector to lens prescription.

v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
Hosts :class:`DesignParameterization` (single-prescription) and
:class:`MultiPrescriptionParameterization` (joint optimisation across
multiple lenses), plus the private ``_read_path`` / ``_write_path``
helpers they share.  ``optimize/core.py`` re-exports every name so all
historical import paths continue to work bit-for-bit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DesignParameterization:
    """Map a flat parameter vector to a lens prescription dict.

    Attributes
    ----------
    template : dict
        Base prescription.  Deep-copied on each ``build()`` call;
        the copy has the free variables replaced by the current
        parameter values.
    free_vars : list of tuple
        Each entry is a "path" into the prescription dict.  Supported
        forms:
          - ``('surfaces', i, key)`` -- surface-dict field like
            ``radius``, ``conic``, ``radius_y`` etc.
          - ``('thicknesses', i)`` -- ``thicknesses[i]``
          - ``('aperture_diameter',)`` -- top-level field
    bounds : list of tuple or None
        (lower, upper) for each parameter.  Used by bounded
        scipy solvers.  Pass ``None`` to disable bounds for a given
        parameter.
    """

    template: Dict[str, Any]
    free_vars: List[Tuple[Any, ...]]
    bounds: Optional[List[Optional[Tuple[float, float]]]] = None

    def __post_init__(self) -> None:
        if self.bounds is not None:
            if len(self.bounds) != len(self.free_vars):
                raise ValueError(
                    f"bounds length {len(self.bounds)} != free_vars "
                    f"length {len(self.free_vars)}")

    @property
    def n_params(self) -> int:
        return len(self.free_vars)

    def initial_values(self) -> np.ndarray:
        """Read the free-var values from the template as the starting x0."""
        x0 = np.empty(self.n_params, dtype=np.float64)
        for i, path in enumerate(self.free_vars):
            x0[i] = _read_path(self.template, path)
        return x0

    def build(self, x: np.ndarray) -> Dict[str, Any]:
        """Return a deep copy of the template with free vars set to x."""
        pres = copy.deepcopy(self.template)
        for i, path in enumerate(self.free_vars):
            _write_path(pres, path, float(x[i]))
        return pres


def _read_path(pres, path):
    """Read a value from a prescription dict along a tuple path."""
    cur = pres
    for p in path:
        if isinstance(cur, dict):
            cur = cur[p]
        else:  # list or tuple
            cur = cur[p]
    return float(cur)


def _write_path(pres, path, value):
    """Write a value into a prescription dict along a tuple path."""
    cur = pres
    for p in path[:-1]:
        if isinstance(cur, dict):
            cur = cur[p]
        else:
            cur = cur[p]
    last = path[-1]
    if isinstance(cur, dict):
        cur[last] = value
    else:
        cur[last] = value


@dataclass
class MultiPrescriptionParameterization:
    """Optimize multiple lens prescriptions jointly.

    Like :class:`DesignParameterization` but holds a LIST of template
    prescriptions and a free-var list whose entries are
    ``(prescription_index, *inner_path)`` tuples pointing into a
    specific template.  ``build()`` returns a list of concrete
    prescriptions of the same length; the context stores them as
    ``ctx.prescriptions``.

    Use this when an architecture has more than one lens whose
    parameters vary independently -- e.g. a 4f imaging system's two
    achromats, a Keplerian telescope's objective + eyepiece, or a
    zoom stage.

    Example
    -------
    >>> import lumenairy as la
    >>> obj = la.thorlabs_lens('AC254-200-C')
    >>> eye = la.thorlabs_lens('AC254-050-C')
    >>> param = la.MultiPrescriptionParameterization(
    ...     templates=[obj, eye],
    ...     free_vars=[
    ...         (0, 'surfaces', 0, 'radius'),  # obj R1
    ...         (0, 'surfaces', 2, 'radius'),  # obj R3
    ...         (1, 'surfaces', 0, 'radius'),  # eye R1
    ...         (1, 'thicknesses', 1),          # eye inner thickness
    ...     ],
    ...     bounds=[(100e-3, 300e-3), (-300e-3, -50e-3),
    ...             (20e-3, 100e-3), (1e-3, 5e-3)])
    >>> # The merit must know which prescription slot maps to which lens:
    >>> merit = la.MatchIdealSystemMerit(
    ...     ideal_elements=[
    ...         {'type': 'lens', 'f': 200e-3},
    ...         {'type': 'propagate', 'z': 250e-3},
    ...         {'type': 'lens', 'f': 50e-3},
    ...         {'type': 'propagate', 'z': 50e-3},
    ...     ],
    ...     real_elements=[
    ...         {'type': '_prescription_', 'index': 0},
    ...         {'type': 'propagate', 'z': 250e-3},
    ...         {'type': '_prescription_', 'index': 1},
    ...         {'type': 'propagate', 'z': 50e-3},
    ...     ],
    ...     match='field_overlap')

    Attributes
    ----------
    templates : list of dict
        Base prescriptions, one per lens.
    free_vars : list of tuple
        Each tuple starts with an int (the prescription index in
        ``templates``) followed by the same path format accepted by
        :class:`DesignParameterization`.
    bounds : list of (lo, hi) or None
    """

    templates: List[Dict[str, Any]]
    free_vars: List[Tuple[Any, ...]]
    bounds: Optional[List[Optional[Tuple[float, float]]]] = None

    def __post_init__(self) -> None:
        for fv in self.free_vars:
            if not fv or not isinstance(fv[0], (int, np.integer)):
                raise ValueError(
                    f"MultiPrescriptionParameterization free_vars entries "
                    f"must start with an int prescription index; got {fv!r}")
            if not (0 <= int(fv[0]) < len(self.templates)):
                raise ValueError(
                    f"free_var {fv!r} refers to template index "
                    f"{fv[0]}, but only {len(self.templates)} templates "
                    f"were provided")
        # v4.14 (audit P3 #19): duplicate (prescription_index, *path)
        # entries silently get separate ``x[i]`` slots that all write
        # to the same prescription field; the optimiser's gradient is
        # split arbitrarily across the duplicates and the design is
        # effectively over-parameterised.  Catch this at construction
        # time so the user gets a clear error rather than a quietly
        # wrong optimisation result.
        seen_keys: Dict[Tuple[Any, ...], int] = {}
        duplicates: List[Tuple[int, int, Tuple[Any, ...]]] = []
        for i, fv in enumerate(self.free_vars):
            # Normalise: cast prescription-index to plain int so
            # numpy int / Python int compare equal in the dict key.
            key = (int(fv[0]),) + tuple(fv[1:])
            if key in seen_keys:
                duplicates.append((seen_keys[key], i, key))
            else:
                seen_keys[key] = i
        if duplicates:
            dup_lines = '\n  '.join(
                f"slots x[{a}] and x[{b}] both target {k!r}"
                for a, b, k in duplicates)
            raise ValueError(
                f"MultiPrescriptionParameterization: duplicate "
                f"(prescription_index, *path) entries in free_vars -- "
                f"each prescription field can be a free variable at "
                f"most once:\n  {dup_lines}")
        if self.bounds is not None:
            if len(self.bounds) != len(self.free_vars):
                raise ValueError(
                    f"bounds length {len(self.bounds)} != free_vars "
                    f"length {len(self.free_vars)}")

    @property
    def n_params(self) -> int:
        return len(self.free_vars)

    @property
    def n_prescriptions(self) -> int:
        return len(self.templates)

    def initial_values(self) -> np.ndarray:
        x0 = np.empty(self.n_params, dtype=np.float64)
        for i, fv in enumerate(self.free_vars):
            pres_idx = int(fv[0])
            inner_path = fv[1:]
            x0[i] = _read_path(self.templates[pres_idx], inner_path)
        return x0

    def build(self, x: np.ndarray) -> List[Dict[str, Any]]:
        """Return a list of deep-copied prescriptions with free vars
        set to ``x``."""
        prescriptions = [copy.deepcopy(t) for t in self.templates]
        for i, fv in enumerate(self.free_vars):
            pres_idx = int(fv[0])
            inner_path = fv[1:]
            _write_path(prescriptions[pres_idx], inner_path, float(x[i]))
        return prescriptions
