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

# v5.2 (AUDIT_V4_13_1 P1-1 closure): per-variable-type scale floors used
# by the driver's finite-difference Hessian estimator.  Without an
# explicit floor a parameter near zero (e.g. a radius set to 0 for a
# planar surface) collapses ``x_scale[i]`` to 0 and the optimiser's
# step-size logic divides through to give a sub-eps relative
# perturbation -- the FD gradient for that variable becomes numeric
# noise.  Values are absolute (SI metres for length-like vars,
# dimensionless for shape-like vars).
_DEFAULT_SCALE_FLOORS = {
    'radius':   1e-6,   # m  (1 um -- conservative for sub-mm lenses)
    'radius_y': 1e-6,   # m  (anamorphic Ry mirror of ``radius``)
    'thickness': 1e-6,  # m  (1 um -- sub-mm spacings stay perturbable)
    'conic': 1e-3,      # dimensionless
    # Aspheric / Forbes / freeform coefficient floors.  These are
    # nominally dimensionless but each ``alpha_n`` carries an implicit
    # ``r^(2n)`` factor in the sag formula; the floor below tracks the
    # coefficient magnitude in its native units, leaving the radial
    # rescale to the merit's natural sensitivity.
    'aspheric': 1e-3,   # dimensionless per coeff
    'alpha':    1e-3,   # dimensionless per coeff (Forbes Q-poly)
    # Catch-all for unrecognised paths: 1 um was the historical hard-
    # coded driver default (driver.py:756), matched here for back-
    # compat on prescription fields we don't yet classify.
    '_default': 1e-6,
}


def _classify_path_to_floor(path: Tuple[Any, ...]) -> float:
    """v5.2 (AUDIT_V4_13_1 P1-1 closure): map a free-var path tuple to
    the appropriate per-variable scale floor.

    The path forms accepted are the same as those accepted by
    :func:`_read_path` / :func:`_write_path`:

    * ``('surfaces', i, key)`` -- per-surface field.  ``key`` is
      classified by name (``radius`` -> ``1e-6``, ``conic`` -> ``1e-3``,
      etc.).
    * ``('thicknesses', i)`` -- always the thickness floor (1 um).
    * ``('aperture_diameter',)`` -- classified as a length (uses the
      thickness floor).
    * Anything else -- the catch-all 1 um default.

    Conic-like / aspheric-like detection uses substring matching so
    Forbes Q-polys (``alpha0``, ``alpha1``, ...) and aspheric series
    (``A4``, ``A6``, ``a_4``, ...) both land on the dimensionless
    floor.
    """
    if not path:
        return _DEFAULT_SCALE_FLOORS['_default']
    head = path[0]
    if head == 'thicknesses':
        return _DEFAULT_SCALE_FLOORS['thickness']
    if head == 'aperture_diameter':
        return _DEFAULT_SCALE_FLOORS['thickness']
    if head == 'surfaces':
        # ('surfaces', i, key, ...)
        if len(path) < 3:
            return _DEFAULT_SCALE_FLOORS['_default']
        key = path[2]
        if not isinstance(key, str):
            return _DEFAULT_SCALE_FLOORS['_default']
        key_lc = key.lower()
        # Radius family first -- ``radius``, ``radius_y``.
        if key_lc.startswith('radius'):
            return _DEFAULT_SCALE_FLOORS['radius']
        # Conic constant.
        if key_lc == 'conic' or key_lc == 'k':
            return _DEFAULT_SCALE_FLOORS['conic']
        # Aspheric / Forbes-Q polynomial coefficients.  Substring
        # match catches ``alpha0`` / ``alpha_n`` / ``A4`` / ``a_8`` /
        # ``aspheric_coeffs[3]`` etc.
        if ('alpha' in key_lc or key_lc.startswith('a')
                or 'aspheric' in key_lc or 'forbes' in key_lc):
            return _DEFAULT_SCALE_FLOORS['aspheric']
        # Per-surface thickness override (rare; in some prescriptions
        # thickness is on the surface dict, not the top-level
        # ``thicknesses`` list).
        if 'thickness' in key_lc:
            return _DEFAULT_SCALE_FLOORS['thickness']
    return _DEFAULT_SCALE_FLOORS['_default']


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
    # v5.2 (AUDIT_V4_13_1 P1-1 closure): per-parameter absolute scale
    # floor used by the optimiser's finite-difference Hessian estimator
    # (driver.py:754 reads this via ``getattr(parameterization,
    # 'scale_floor', None)``).  ``None`` (default) auto-fills from
    # :func:`_classify_path_to_floor` -- radii / thicknesses get 1 um,
    # conics / aspherics get 1e-3.  Pass an explicit array of length
    # ``len(free_vars)`` to override per-slot.
    scale_floor: Optional[Any] = None

    def __post_init__(self) -> None:
        if self.bounds is not None:
            if len(self.bounds) != len(self.free_vars):
                raise ValueError(
                    f"bounds length {len(self.bounds)} != free_vars "
                    f"length {len(self.free_vars)}")
        # v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve ``scale_floor`` to a
        # length-``n_params`` ndarray.  When ``None``, infer from the
        # path classification table; when a scalar or array, broadcast
        # to the parameter count.
        self.scale_floor = self._resolve_scale_floor(self.scale_floor)

    def _resolve_scale_floor(self, value: Any) -> np.ndarray:
        """v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve a user-supplied
        ``scale_floor`` (``None`` / scalar / sequence) to an ndarray of
        length ``n_params``."""
        n = len(self.free_vars)
        if value is None:
            return np.array(
                [_classify_path_to_floor(p) for p in self.free_vars],
                dtype=np.float64,
            )
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(
                f"DesignParameterization.scale_floor length "
                f"{arr.shape} does not match free_vars length {n}.")
        return arr

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


@dataclass
class RawParameterization:
    """A flat parameter vector with **no lens-prescription template** -- the
    parameterization for wave-only / rigorous-element design (RCWA, thin-film
    coatings, metasurfaces) where the merit reads the raw parameters
    (``ctx.x``) or drives a custom wave propagator, rather than a ray-traced
    lens.

    Unlike :class:`DesignParameterization` (which resolves ``free_vars`` paths
    against a real surfaces/thicknesses dict), this requires only the starting
    vector ``x0`` and optional ``bounds`` / ``scale_floor``.  ``build(x)``
    returns a minimal prescription ``{'_raw_params': x, 'aperture_diameter':
    None}`` -- no ``'surfaces'`` -- so the wave leg's
    ``pres.get('aperture_diameter')`` falls back to its grid default and the
    merit can read ``ctx.x``.

    Use with merits that set ``needs_ray=False`` (the geometric ray leg has no
    sensible meaning without a prescription, and ``design_optimize`` raises a
    clear error if a ``needs_ray=True`` merit is paired with a template-free
    build).

    Attributes
    ----------
    x0 : array-like of float
        Starting parameter vector (also defines ``n_params``).
    bounds : list of (lower, upper) or None
        Per-parameter bounds for bounded scipy solvers.
    scale_floor : None | scalar | array
        Per-parameter absolute FD step floor (default ``1e-6`` everywhere).
    """

    x0: Any
    bounds: Optional[List[Optional[Tuple[float, float]]]] = None
    scale_floor: Optional[Any] = None

    def __post_init__(self) -> None:
        self.x0 = np.asarray(self.x0, dtype=np.float64).ravel()
        if self.bounds is not None and len(self.bounds) != self.x0.size:
            raise ValueError(
                f"RawParameterization: bounds length {len(self.bounds)} != "
                f"x0 length {self.x0.size}")
        self.scale_floor = self._resolve_scale_floor(self.scale_floor)

    def _resolve_scale_floor(self, value: Any) -> np.ndarray:
        n = self.x0.size
        if value is None:
            return np.full(n, _DEFAULT_SCALE_FLOORS['_default'],
                           dtype=np.float64)
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(
                f"RawParameterization.scale_floor length {arr.shape} does "
                f"not match x0 length {n}.")
        return arr

    @property
    def n_params(self) -> int:
        return int(self.x0.size)

    def initial_values(self) -> np.ndarray:
        return self.x0.copy()

    def build(self, x: np.ndarray) -> Dict[str, Any]:
        return {'_raw_params': np.asarray(x, dtype=np.float64),
                'aperture_diameter': None}


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
    # v5.2 (AUDIT_V4_13_1 P1-1 closure): per-parameter absolute scale
    # floor.  ``None`` (default) auto-fills from
    # :func:`_classify_path_to_floor` applied to the inner-path tuple
    # ``fv[1:]`` (the ``fv[0]`` prescription index is discarded for
    # classification).  Pass an explicit array of length
    # ``len(free_vars)`` to override per-slot.
    scale_floor: Optional[Any] = None

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
        # v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve ``scale_floor`` to
        # a length-``n_params`` ndarray using the inner path (the
        # ``fv[0]`` prescription index has no bearing on the variable
        # type).
        self.scale_floor = self._resolve_scale_floor(self.scale_floor)

    def _resolve_scale_floor(self, value: Any) -> np.ndarray:
        """v5.2 (AUDIT_V4_13_1 P1-1 closure): resolve a user-supplied
        ``scale_floor`` (``None`` / scalar / sequence) to an ndarray of
        length ``n_params``.  Inner paths (``fv[1:]``) are passed to
        :func:`_classify_path_to_floor`."""
        n = len(self.free_vars)
        if value is None:
            return np.array(
                [_classify_path_to_floor(tuple(fv[1:]))
                 for fv in self.free_vars],
                dtype=np.float64,
            )
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(n, float(arr), dtype=np.float64)
        if arr.shape != (n,):
            raise ValueError(
                f"MultiPrescriptionParameterization.scale_floor length "
                f"{arr.shape} does not match free_vars length {n}.")
        return arr

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
