"""
Hybrid wave/ray optical-design optimization.

Combines the fast, differentiable-in-parameters paraxial ray trace
(``raytrace`` module) with the full wave-optics propagation
(``apply_real_lens`` / ``apply_real_lens_traced``) to optimize lens
prescriptions against a user-specified merit function.

Architecture
------------
A lens design is specified by a *parameter vector* mapped onto a
*prescription template*.  :class:`DesignParameterization` handles the
mapping:

    free params     ->   prescription dict (for apply_real_lens etc.)

Each iteration the optimizer:

    1. Builds the current prescription from the parameter vector.
    2. Evaluates fast geometric figures (focal length, Seidel
       coefficients, ray fans) via the ray tracer.
    3. Optionally evaluates wave figures (Strehl ratio at best
       focus, RMS wavefront error via Zernike decomposition, spot
       size in a through-focus scan) via the wave-optics path.
    4. Combines these into a scalar merit via a sum of
       :class:`MeritTerm` objects, each weighted.

``scipy.optimize.minimize`` (or ``scipy.optimize.least_squares`` for
Gauss-Newton / Levenberg-Marquardt) drives the parameter updates.
Finite-difference gradients are used by default; users can supply
an analytic Jacobian where available.

Typical usage
-------------

.. code-block:: python

    import lumenairy as la
    from lumenairy.optimize import (
        DesignParameterization, design_optimize,
        FocalLengthMerit, StrehlMerit, RMSWavefrontMerit,
    )

    # Start from a Thorlabs AC254-100-C achromat, free up R1/R2/R3/d1.
    template = la.thorlabs_lens('AC254-100-C')
    template['aperture_diameter'] = 10e-3

    param = DesignParameterization(template,
        free_vars=[
            ('surfaces', 0, 'radius'),
            ('surfaces', 1, 'radius'),
            ('surfaces', 2, 'radius'),
            ('thicknesses', 0),
        ],
        bounds=[
            (50e-3, 80e-3),
            (-60e-3, -30e-3),
            (-250e-3, -150e-3),
            (4e-3, 8e-3),
        ])

    merit = [
        FocalLengthMerit(target=100e-3, weight=1.0),
        StrehlMerit(min_strehl=0.95, weight=10.0),
        RMSWavefrontMerit(max_rms_waves=0.05, weight=50.0),
    ]

    result = design_optimize(param, merit,
                             wavelength=1.31e-6,
                             N=512, dx=20e-6,
                             method='L-BFGS-B', verbose=True)

    print('Optimized prescription:', result.prescription)
    print('Merit:', result.merit, '  Strehl:', result.strehl_best)
"""

from __future__ import annotations

import copy
import threading
import time
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from ..elements.lenses import apply_real_lens, apply_real_lens_traced
from ..raytrace import (
    surfaces_from_prescription, system_abcd, trace,
    seidel_coefficients,
)
from ..analysis import wave_opd_2d, zernike_decompose
from ..analysis.through_focus import (
    through_focus_scan, find_best_focus, diffraction_limited_peak,
)
# 4.10.2: pull get_default_complex_dtype to module scope so merit-leg
# wave-propagation source fields (apply_real_lens inputs) can be
# allocated at the runtime-selected precision -- preserves the
# precision='single' choice through merit evaluations.
from ..propagators.propagation import get_default_complex_dtype
# v4.14.1 (P2-3): the v4.14.0 monkey-patch of
# ``_propagation_module.clear_asm_caches`` has been retired in favour
# of a lazy reverse-direction import inside
# ``propagation.clear_asm_caches()`` itself.  See
# :func:`_clear_wrapper_merit_cache` and the comment in propagation.py.


# =========================================================================
# Wave-propagator registry
# =========================================================================
#
# ``WAVE_PROPAGATOR_REGISTRY`` maps a string name to a callable with
# the signature
#
#     fn(E0, pres, *, wavelength, dx, N, wp_kwargs, opts) -> E_exit
#
# where ``opts`` is a dict of design_optimize-internal flags (e.g.
# ``wave_traced`` / ``ray_subsample``) so the registered function can
# honour them.  ``wp_kwargs`` is a mutable dict the function may
# ``.pop`` from to extract its own options.  Returns the post-lens
# field on the (Ny, Nx, dx) grid expected by the wave-leg merits.
#
# Users can register custom propagators with
# :func:`register_wave_propagator(name, fn)` and immediately use them
# via ``design_optimize(wave_propagator=name)``.  The default registry
# below is populated with the same five propagators that the legacy
# if/elif chain handled (real_lens, gbd, hf, hfpi, asymptotic) -- the
# refactor preserves behaviour byte-for-byte.

WAVE_PROPAGATOR_REGISTRY: Dict[str, Callable] = {}


def register_wave_propagator(name: str, fn: Callable) -> None:
    """Register a custom wave-leg propagator under ``name``.

    The function is called by :func:`design_optimize` when its
    ``wave_propagator=name``.  Signature:

    ``fn(E0, pres, *, wavelength, dx, N, wp_kwargs, opts) -> E_exit``

    where ``wp_kwargs`` is a mutable dict the function may pop from,
    and ``opts`` is a dict carrying ``design_optimize``-internal
    flags (``wave_traced``, ``ray_subsample``, ...).
    """
    WAVE_PROPAGATOR_REGISTRY[name] = fn


def unregister_wave_propagator(name: str) -> None:
    """Remove a wave propagator from the registry."""
    WAVE_PROPAGATOR_REGISTRY.pop(name, None)


def _wave_real_lens(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    if opts.get('wave_traced', False):
        return apply_real_lens_traced(
            E0, prescription=pres, wavelength=wavelength, dx=dx,
            ray_subsample=opts.get('ray_subsample', 4), n_workers=1,
            **wp_kwargs)
    return apply_real_lens(E0, prescription=pres, wavelength=wavelength, dx=dx, **wp_kwargs)


def _wave_gbd(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.gbd import propagate_gbd_through_prescription
    return propagate_gbd_through_prescription(
        E0, dx, pres, wavelength=wavelength, **wp_kwargs)


def _wave_hf(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.hf import propagate_huygens_fresnel_through_prescription
    return propagate_huygens_fresnel_through_prescription(
        E0, dx, pres, wavelength=wavelength, **wp_kwargs)


def _wave_hfpi(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.hfpi import propagate_hfpi_through_prescription
    return propagate_hfpi_through_prescription(
        E0, dx, pres, wavelength=wavelength, **wp_kwargs)


def _wave_asymptotic(E0, pres, *, wavelength, dx, N, wp_kwargs, opts):
    from ..propagators.asymptotic import (
        fit_canonical_polynomials, propagate_modal_asymptotic,
    )
    # Fit the prescription (or reuse a pre-built fit if supplied).
    fit = wp_kwargs.pop('fit', None)
    fit_kw = wp_kwargs.pop('fit_kwargs', {}) or {}
    if fit is None:
        fit = fit_canonical_polynomials(
            pres, wavelength=wavelength, **fit_kw)
    # Sample the asymptotic propagator on the (Ny, Nx, dx) wave grid
    # so downstream merits see a usable field.  Centre at the chief
    # image of the on-axis source.
    # v4.12.1 (B1-10): pixel-centred `(arange(N) - N/2)*dx`, matches the
    # library-wide convention (ASM, Fresnel, RS, sources).  Merit
    # functions that compare wave-leg fields across propagator
    # families need a single shared grid convention; the previous
    # `+0.5` produced a half-pixel offset between the asymptotic leg
    # and the ASM / GBD / HF legs.
    _ax = (np.arange(N) - N / 2) * dx
    _x_grid = _ax + fit.s2x_centre
    _y_grid = _ax + fit.s2y_centre
    _X, _Y = np.meshgrid(_x_grid, _y_grid, indexing='xy')
    _wp = wp_kwargs.pop('w_p', 0.05)
    _ws = wp_kwargs.pop('w_s', 50e-6)
    return propagate_modal_asymptotic(
        fit,
        s2_grid_x=_X, s2_grid_y=_Y,
        w_s=_ws, w_p=_wp,
        v2_centre=(fit.v2x_centre, fit.v2y_centre),
        **wp_kwargs,
    )


# Populate the default registry.  Users can override these by
# re-registering, but all five preserve exact legacy behaviour.
register_wave_propagator('real_lens', _wave_real_lens)
register_wave_propagator('gbd', _wave_gbd)
register_wave_propagator('hf', _wave_hf)
register_wave_propagator('hfpi', _wave_hfpi)
register_wave_propagator('asymptotic', _wave_asymptotic)


# =========================================================================
# Parameterization
# =========================================================================

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


# =========================================================================
# Merit terms
# =========================================================================

# Sentinel used by EvaluationContext when ABCD extraction failed.  Merit
# terms that consume ``ctx.efl`` / ``ctx.bfl`` should route through
# :func:`ctx_is_valid` rather than blindly plugging the sentinel into
# their formulas (a naive ``(1e9 - target)^2`` is astronomical and drags
# the optimizer away from good regions).
_INVALID_FL_SENTINEL = 1e9


def ctx_is_valid(ctx: Any, field: str) -> bool:
    """Return True if ``ctx.<field>`` holds a usable physical value.

    Guards against the sentinels set when the ray-leg failed (``1e9``
    for focal lengths) and against NaN/Inf from downstream computations.
    """
    try:
        v = getattr(ctx, field)
    except AttributeError:
        return False
    import numpy as _np
    if v is None:
        return False
    if not _np.isfinite(v):
        return False
    if abs(v) >= _INVALID_FL_SENTINEL * 0.5:
        return False
    return True


class MeritTerm:
    """Base class for a single term in the merit function.

    Each merit term takes the full ``EvaluationContext`` (ray-trace
    results, wave field, etc.) and returns a scalar contribution
    (already weighted).  Concrete subclasses override
    :meth:`evaluate`.

    Attributes
    ----------
    weight : float
        Multiplier applied to the raw term value.  Squared residuals
        in a least-squares sense, or additive penalty in a general
        minimize sense.
    needs_wave : bool, default False
        If True, the optimizer will run the wave-optics pipeline
        (``apply_real_lens_traced`` + through-focus) for each
        evaluation.  Set False for pure-geometric terms -- the
        optimizer will skip the expensive wave leg if NO merit
        terms need it.
    """

    weight: float = 1.0
    needs_wave: bool = False
    name: str = 'MeritTerm'

    def evaluate(self, ctx: Any) -> float:
        raise NotImplementedError


class FocalLengthMerit(MeritTerm):
    """Penalise deviation from target focal length.

    ``contribution = weight * (efl - target)^2 / target^2``

    When ``target == 0`` (afocal / collimator), the normalised-error
    formula is ill-defined; the merit falls back to a penalty on the
    *optical power* ``1/efl`` (in m^-1, equivalently dioptres).  This
    drives the merit to zero as the system becomes truly afocal
    (``efl -> infinity``) and grows without bound as ``efl -> 0``
    (point-image collapse), which is the right gradient direction for
    a collimator design.

    Use ``target > 0`` for a finite-EFL target; use ``target == 0``
    for an afocal / collimator target.  There is no use case for
    ``target < 0`` (a virtual-image lens) -- pass the positive EFL
    instead and rely on ABCD sign conventions in the prescription.
    """

    needs_wave = False
    name = 'FocalLength'

    def __init__(self, target: float, weight: float = 1.0) -> None:
        self.target = float(target)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        efl = getattr(ctx, 'efl', float('nan'))
        if not ctx_is_valid(ctx, 'efl'):
            return self.weight  # graceful large-but-finite penalty
        if self.target == 0.0:
            # Afocal / collimator: minimise 1/|efl| so merit -> 0 as
            # efl -> infinity.  Clamp efl below to keep the merit
            # finite when the optimiser walks through near-zero EFL
            # during the search.
            efl_clamped = max(abs(efl), 1e-12)
            power = 1.0 / efl_clamped  # dioptres
            return self.weight * power * power
        err = (efl - self.target) / self.target
        return self.weight * err * err


class BackFocalLengthMerit(MeritTerm):
    """Penalise deviation from target back focal length.

    Same zero-target behaviour as :class:`FocalLengthMerit`: BFL ->
    infinity drives the merit toward zero, BFL -> 0 explodes it.
    """

    needs_wave = False
    name = 'BackFocalLength'

    def __init__(self, target: float, weight: float = 1.0) -> None:
        self.target = float(target)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        bfl = getattr(ctx, 'bfl', float('nan'))
        if not ctx_is_valid(ctx, 'bfl'):
            return self.weight
        if self.target == 0.0:
            bfl_clamped = max(abs(bfl), 1e-12)
            power = 1.0 / bfl_clamped
            return self.weight * power * power
        err = (bfl - self.target) / self.target
        return self.weight * err * err


class SphericalSeidelMerit(MeritTerm):
    """Minimise Seidel spherical aberration coefficient S_I.

    Fast, geometric-only term.
    """

    needs_wave = False
    name = 'SphericalSeidel'

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        # 4.10.2: guard against the NaN-filled sentinel that
        # aberration_summary returns on failed Seidel computation
        # (4.10.1).  Pre-4.10.2 read ctx.seidel[0] blindly and
        # returned NaN, which scipy.optimize then refused as an
        # invalid objective.  Treat invalid Seidel as a moderate
        # penalty so the optimiser sees a finite value and can
        # step away from the bad region.
        s = ctx.seidel
        if s is None or len(s) == 0 or not np.isfinite(s[0]):
            return self.weight
        return self.weight * float(s[0]) ** 2


class StrehlMerit(MeritTerm):
    """Penalise Strehl ratio below ``min_strehl``.

    ``contribution = weight * max(0, min_strehl - best_strehl)^2``
    """

    needs_wave = True
    name = 'Strehl'

    def __init__(self, min_strehl: float = 0.8, weight: float = 1.0) -> None:
        self.min_strehl = float(min_strehl)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        deficit = max(0.0, self.min_strehl - ctx.strehl_best)
        return self.weight * deficit * deficit


class RMSWavefrontMerit(MeritTerm):
    """Penalise RMS wavefront error above a target (waves).

    Uses Zernike decomposition to exclude the first
    ``exclude_low_order`` modes (default 4: piston + 2 tilts +
    defocus), matching the optics-design convention of reporting
    'image-quality' RMS after best-focus.  Set ``exclude_low_order=3``
    to keep defocus in the RMS (penalises focus shift as well as
    high-order aberrations).
    """

    needs_wave = True
    name = 'RMSWavefront'

    def __init__(self, max_rms_waves: float = 0.07,
                 n_modes: int = 21,
                 exclude_low_order: int = 4,
                 weight: float = 1.0) -> None:
        self.max_rms_waves = float(max_rms_waves)
        self.n_modes = int(n_modes)
        self.exclude_low_order = int(exclude_low_order)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        rms_waves = ctx.rms_wavefront_waves(
            n_modes=self.n_modes,
            exclude_low_order=self.exclude_low_order)
        excess = max(0.0, rms_waves - self.max_rms_waves)
        return self.weight * excess * excess


class SpotSizeMerit(MeritTerm):
    """Penalise RMS spot radius at best focus above a target."""

    needs_wave = True
    name = 'SpotSize'

    def __init__(self, max_rms_radius: float, weight: float = 1.0) -> None:
        self.max_rms_radius = float(max_rms_radius)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        r = ctx.rms_radius_best
        excess = max(0.0, r - self.max_rms_radius)
        return self.weight * excess * excess


class MatchIdealThinLensMerit(MeritTerm):
    """Penalise deviation of the actual exit-pupil OPD from an
    idealised thin-lens wavefront with the same target focal length.

    This is the merit term you want when you're asking "make this real
    lens behave as much like an ideal thin lens of focal length f as
    possible".  At each evaluation:

    1. The actual exit-pupil OPD is extracted from the wave-optics
       output (using `wave_opd_2d`, with reference-sphere subtraction
       for numerical stability).
    2. An ideal thin-lens OPD ``OPD_ideal(r) = -r^2 / (2*f_target)`` is
       computed on the same grid.
    3. Their difference -- the *aberration wavefront* -- is masked to
       the pupil and its RMS computed.
    4. ``contribution = weight * RMS_diff^2 / wavelength^2`` (in
       wavelength-squared units, so weights of 1.0 produce
       ``waves^2`` of penalty).

    The result of optimizing against this merit is a wavefront whose
    departure from a perfect spherical converging wave is minimised
    -- which is the formal definition of "diffraction-limited" up to
    a tolerable RMS wavefront error.

    Parameters
    ----------
    target_focal_length : float
        Focal length [m] of the ideal thin lens to match.
    weight : float, default 1.0
    exclude_low_order : int, default 1
        Number of Zernike modes to exclude from the RMS (default 1 =
        piston only).  Set to 3 to also exclude tilts (so a lateral
        decenter doesn't dominate the merit), or 4 to also exclude
        defocus (so the merit becomes "match the ideal except for an
        arbitrary focus shift").
    n_modes : int, default 21
        How many Zernike modes the OPD is decomposed into to compute
        the high-order RMS.
    """

    needs_wave = True
    name = 'MatchIdealThinLens'

    def __init__(self, target_focal_length: float, weight: float = 1.0,
                 exclude_low_order: int = 1, n_modes: int = 21) -> None:
        self.target_focal_length = float(target_focal_length)
        self.weight = float(weight)
        self.exclude_low_order = int(exclude_low_order)
        self.n_modes = int(n_modes)

    def evaluate(self, ctx: Any) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        # Compute the ideal thin-lens OPD on the pupil grid
        Ny, Nx = ctx.opd_map.shape
        x = (np.arange(Nx) - Nx / 2) * ctx.dx
        y = (np.arange(Ny) - Ny / 2) * ctx.dx
        X, Y = np.meshgrid(x, y)
        opd_ideal = -(X ** 2 + Y ** 2) / (2.0 * self.target_focal_length)
        diff = ctx.opd_map - opd_ideal
        # Decompose into Zernikes; high-order RMS = aberration RMS
        # (excluding the requested low-order modes which represent
        # piston / tilt / defocus -- usually NOT what you want to
        # penalise unless you've fixed alignment).
        from ..analysis import zernike_decompose
        finite = np.isfinite(diff)
        # Replace NaN with 0 in the masked region; decompose handles it
        diff_clean = np.where(finite, diff, 0.0)
        try:
            coeffs, _ = zernike_decompose(
                diff_clean, ctx.dx, ap, n_modes=self.n_modes)
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                ZeroDivisionError):
            # zernike_decompose can fail on under-sampled pupils
            # (ValueError) or singular bases (LinAlgError); the
            # convention is to return a 0 merit so the optimizer
            # doesn't get derailed.
            return 0.0
        higher = coeffs[self.exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        rms_waves = rms_m / ctx.wavelength
        return self.weight * rms_waves * rms_waves


# =========================================================================
# Full-system "match this ideal thin-lens architecture" merit
# =========================================================================

class MatchIdealSystemMerit(MeritTerm):
    """Match the real system's output field to that of an idealised
    thin-lens reference system.

    Unlike :class:`MatchIdealThinLensMerit` -- which operates on the
    exit-pupil OPD of a single lens and compares it to a bare
    converging sphere -- this merit propagates a reference source
    through BOTH an ideal thin-lens element list AND the real
    prescription (wrapped in an optional pre/post-propagation envelope)
    and then compares the resulting **complex output fields**.

    Use cases:

    * Replace a paraxial thin lens with a singlet / doublet / aspheric
      while preserving the output radiation pattern + relative phase.
    * Replace a Keplerian / Galilean telescope's thin-lens pair with
      real achromats.
    * Replace a 4f imaging system's two thin lenses with two real
      lenses, jointly optimised.
    * Any architecture expressible as
      ``propagate_through_system(source, ideal_elements)``.

    The merit is designed for the common situation where the real
    system is **slightly longer** than the ideal (because real lenses
    have nonzero thickness) but otherwise functionally equivalent.
    Approximately preserving aperture and inter-element distances is
    the user's responsibility; the merit will drive whatever free
    variables it can to make the output fields match.

    Parameters
    ----------
    ideal_elements : list of dict
        Element list for the ideal reference system, in the same
        format as :func:`propagate_through_system`.  Typically a mix
        of ``{'type': 'lens', 'f': ...}`` and
        ``{'type': 'propagate', 'z': ...}`` elements, optionally with
        apertures / mirrors / masks.
    real_elements : list of dict, optional
        Element list for the real system.  Dicts with
        ``type='_prescription_'`` are replaced at evaluation time
        with the current ``ctx.prescription`` wrapped as a
        ``'real_lens'`` element (or ``'real_lens_traced'`` if
        ``use_traced_lens=True``).  Default: a single-lens drop-in,
        ``[{'type': '_prescription_'}]``, which is correct when the
        ideal is a single thin lens + propagate pair and the real
        prescription replaces that thin lens.
    source_fn : callable or None
        Factory returning the complex input field at the first plane.
        Signature ``source_fn(N, dx, wavelength) -> ndarray``.  The
        merit uses ``ctx.N``, ``ctx.dx``, ``ctx.wavelength``.  Default
        (``None``) is a uniform plane wave of unit amplitude inside
        the prescription's aperture (if specified) or the full grid
        (otherwise).
    match : ``'field_overlap'`` | ``'field_mse'`` | ``'intensity_mse'`` | ``'intensity_overlap'``
        Similarity metric:

        * ``'field_overlap'`` (**default**, recommended): coupling
          efficiency
          ``|<E_ideal | E_real>|^2 / (||E_ideal||^2 ||E_real||^2)``.
          Bounded in [0, 1], invariant to a global phase and to an
          overall amplitude scaling.  Merit = ``weight * (1 - overlap)``;
          drops to zero when the real field differs from the ideal only
          by a global phase factor and overall amplitude.  **This is
          the right choice for "match the radiation pattern + relative
          phase"** as requested -- global phase is not a physical
          observable.
        * ``'field_mse'``: power-normalised + phase-aligned MSE of the
          field difference.  Merit is roughly the squared "fraction"
          of energy in the difference.  More sensitive to absolute
          phase shape than ``field_overlap``.
        * ``'intensity_mse'``: MSE of ``|E|^2``, phase-blind.  Use
          this when only the radiation pattern (not phase) matters
          (e.g. matching a target irradiance profile).
        * ``'intensity_overlap'``: correlation of ``|E|^2`` patterns,
          phase-blind.
    aperture_mask : ndarray or None
        Optional boolean / real mask applied to BOTH output fields
        before the comparison.  Use it to restrict the match to a
        region of interest (e.g. the intended image area) and avoid
        letting low-intensity grid edges dominate.
    use_traced_lens : bool, default False
        If True, propagate the real prescription via
        ``apply_real_lens_traced`` (sub-nm OPD agreement with the
        ray trace, 10-30x slower) rather than ``apply_real_lens``.
    ray_subsample : int, default 4
        Passed to ``apply_real_lens_traced`` when used.
    focus_search : bool, default False
        If True, scan a small range of axial offsets on the real
        system's output plane and report the BEST (lowest-penalty)
        match.  Decouples "correct focal plane" from "aberration
        quality" so a small BFL shift caused by real-lens thickness
        doesn't dominate the penalty.  Not valid for
        ``match='intensity_mse'`` (no unique optimum under
        translation); enable it with any of the other three metrics.
    focus_search_range : tuple (z_lo, z_hi) or None
        Axial-offset bracket for the focus search, relative to the
        nominal output plane [m].  Default (None): +/- f/20 computed
        from ``ctx.bfl`` or ``ctx.efl``, falling back to +/- 5 mm.
    focus_search_n : int, default 9
        Number of samples in the z-offset scan.
    wavelengths : list of float, optional
        If given, evaluate the merit at each wavelength and average
        the results.  Drives the glass-index dispersion through
        ``apply_real_lens`` + ``propagate_through_system``.  Useful
        for broadband / chromatic-matching optimisation without
        needing a separate ``MultiWavelengthMerit`` wrapper.
    field_angles : list of (theta_x, theta_y) tuples, optional
        Off-axis tilts (radians) to evaluate.  Each field angle adds
        a carrier phase to the source so the merit penalises the
        real lens's output for multiple input beam directions
        simultaneously.  Combines Cartesian-product-wise with
        ``wavelengths``.
    weight : float

    Notes
    -----
    * The ideal system is propagated with ``propagate_through_system``;
      each thin lens applies the paraxial phase screen
      ``exp(-i k r^2 / 2f)``.  This is exact in the small-angle limit
      and close to correct for f/2 or slower systems.  For systems
      with non-paraxial focusing, consider using
      ``{'type': 'lens', 'f': ..., 'lens_model': 'nonparaxial'}`` in
      ``ideal_elements``.
    * ``field_overlap`` is the physical "coupling efficiency" metric
      used in fiber / optical-mode matching.  It's bounded and
      dimensionless, which makes it numerically well-behaved for the
      optimizer and meaningful as an absolute number (0.99 = nearly
      perfect; 0.50 = significant mismatch).
    * Multi-lens architectures where two or more prescriptions are
      independently varied require multiple
      :class:`DesignParameterization` -- not yet supported by the
      single-template design.  Open a PR if you hit this.
    """

    needs_wave = True
    name = 'MatchIdealSystem'

    def __init__(self, ideal_elements: Sequence[Dict[str, Any]],
                 real_elements: Optional[Sequence[Dict[str, Any]]] = None,
                 source_fn: Optional[Callable] = None,
                 match: str = 'field_overlap',
                 aperture_mask: Optional[np.ndarray] = None,
                 use_traced_lens: bool = False,
                 ray_subsample: int = 4,
                 focus_search: bool = False,
                 focus_search_range: Optional[Tuple[float, float]] = None,
                 focus_search_n: int = 9,
                 wavelengths: Optional[Sequence[float]] = None,
                 field_angles: Optional[Sequence[float]] = None,
                 weight: float = 1.0) -> None:
        self.ideal_elements = list(ideal_elements)
        self.real_elements = (list(real_elements)
                              if real_elements is not None
                              else [{'type': '_prescription_'}])
        self.source_fn = source_fn
        self.match = str(match)
        self.aperture_mask = aperture_mask
        self.use_traced_lens = bool(use_traced_lens)
        self.ray_subsample = int(ray_subsample)
        self.weight = float(weight)
        self.focus_search = bool(focus_search)
        self.focus_search_range = focus_search_range
        self.focus_search_n = int(focus_search_n)
        # ``wavelengths`` and ``field_angles`` drive built-in sweeps
        # (averaged penalty).  Both default to None = single
        # wavelength / on-axis.
        self.wavelengths = (list(wavelengths)
                            if wavelengths is not None else None)
        self.field_angles = (list(field_angles)
                             if field_angles is not None else None)
        valid = ('field_overlap', 'field_mse',
                 'intensity_mse', 'intensity_overlap')
        if self.match not in valid:
            raise ValueError(
                f"match must be one of {valid}; got {self.match!r}")
        if self.focus_search and self.match not in (
                'field_overlap', 'field_mse', 'intensity_overlap'):
            raise ValueError(
                f"focus_search requires match in "
                f"('field_overlap', 'field_mse', 'intensity_overlap'); "
                f"got {self.match!r}.  intensity_mse doesn't have a "
                f"unique optimum under axial translation.")

    # -- Helpers -----------------------------------------------------

    def _make_source(self, ctx, wavelength, field_angle=(0.0, 0.0)):
        """Build the reference input field at the first plane.

        Parameters
        ----------
        ctx : EvaluationContext
            Provides ``N``, ``dx``, ``prescription`` (for default
            aperture clipping).
        wavelength : float
            Wavelength [m] used both by ``source_fn`` (if supplied)
            and for the field-angle carrier phase.
        field_angle : (float, float)
            Off-axis tilt ``(theta_x, theta_y)`` in radians.  A linear
            phase ``exp(i * k_x X + i * k_y Y)`` is applied on top of
            whatever source the factory produced, with
            ``k_x = 2 pi sin(theta_x) / wavelength``.
        """
        cdtype = get_default_complex_dtype()  # 4.10.2: honour precision knob
        if self.source_fn is not None:
            E = self.source_fn(ctx.N, ctx.dx, wavelength)
            E = np.asarray(E, dtype=cdtype)
        else:
            # v4.14.2 (P1-NEW-1): three branches matching the canonical
            # _ZERO_APERTURE_MASK semantics shared by
            # ``MultiWavelengthMerit.evaluate``,
            # ``MultiFieldMerit.evaluate``, and
            # ``ToleranceAwareMerit.evaluate``:
            #   * ``ap`` finite and > 0   -> circular boolean mask.
            #   * ``ap`` finite and <= 0  -> deliberate-zero aperture;
            #     block all light by zeroing E entirely.  Pre-v4.14.2
            #     the ``ap > 0`` check silently fell through to the
            #     ``else`` branch and produced a full-grid plane wave,
            #     which apply_real_lens would then propagate as a
            #     bright on-axis "source" -- the exact bug v4.14.1
            #     fixed in the wrapper-merit cache but missed at this
            #     pre-existing site.
            #   * ``ap`` is None or non-finite -> no aperture specified;
            #     full-grid plane wave (unchanged behaviour).
            ap = (ctx.prescription.get('aperture_diameter')
                  if ctx.prescription else None)
            if ap is not None and np.isfinite(ap) and ap > 0:
                E = np.ones((ctx.N, ctx.N), dtype=cdtype)
                x = (np.arange(ctx.N) - ctx.N / 2) * ctx.dx
                X, Y = np.meshgrid(x, x)
                mask = (X * X + Y * Y) <= (ap / 2.0) ** 2
                # v4.14.2 (P1-NEW-4): dtype-aware zero so a complex64
                # cdtype is not silently upcast to complex128 by the
                # 0.0+0.0j literal.  Mirrors the v4.13.2 sweep at
                # apply_aperture / apply_mirror / _lens_thin /
                # _lens_real.
                E = np.where(mask, E, np.zeros((), dtype=cdtype))
            elif ap is not None and np.isfinite(ap) and ap <= 0:
                E = np.zeros((ctx.N, ctx.N), dtype=cdtype)
            else:
                E = np.ones((ctx.N, ctx.N), dtype=cdtype)

        # Field-angle tilt: apply a linear phase ramp for off-axis
        # illumination.  Identity for on-axis (0, 0).
        tx, ty = field_angle
        if tx or ty:
            x = (np.arange(ctx.N) - ctx.N / 2) * ctx.dx
            X, Y = np.meshgrid(x, x)
            k0 = 2.0 * np.pi / wavelength
            E = E * np.exp(1j * (k0 * np.sin(tx) * X
                                   + k0 * np.sin(ty) * Y))
        return E

    def _build_real_elements(self, ctx):
        """Expand ``_prescription_`` sentinels into real-lens elements.

        The sentinel supports an optional ``'index'`` key that selects
        from ``ctx.prescriptions`` (populated when using
        :class:`MultiPrescriptionParameterization`).  With no index
        supplied, falls back to ``ctx.prescription`` -- the
        single-prescription (backward-compatible) case.
        """
        lens_type = ('real_lens_traced' if self.use_traced_lens
                     else 'real_lens')
        extras = {}
        if self.use_traced_lens:
            extras['ray_subsample'] = self.ray_subsample

        prescriptions = ctx.prescriptions
        if prescriptions is None:
            prescriptions = [ctx.prescription]

        expanded = []
        for elem in self.real_elements:
            if elem.get('type') == '_prescription_':
                idx = int(elem.get('index', 0))
                if not (0 <= idx < len(prescriptions)):
                    raise IndexError(
                        f"'_prescription_' placeholder index {idx} out "
                        f"of range (ctx has {len(prescriptions)} "
                        f"prescriptions)")
                expanded.append({
                    'type': lens_type,
                    'prescription': prescriptions[idx],
                    **extras,
                    # Preserve any user-specified bandlimit / per-
                    # element overrides passed through the sentinel,
                    # except for the meta keys we've already consumed.
                    **{k: v for k, v in elem.items()
                       if k not in ('type', 'index')},
                })
            else:
                expanded.append(dict(elem))
        return expanded

    def _propagate(self, elements, E_in, ctx, wavelength):
        from ..system import propagate_through_system
        E_out, _ = propagate_through_system(
            E_in, elements, wavelength, ctx.dx)
        return E_out

    # -- Main evaluate ----------------------------------------------

    def evaluate(self, ctx: Any) -> float:
        if ctx.prescription is None:
            return self.weight

        wavelengths = self.wavelengths or [ctx.wavelength]
        field_angles = self.field_angles or [(0.0, 0.0)]

        penalties = []
        for wl in wavelengths:
            for fa in field_angles:
                try:
                    p = self._evaluate_one(ctx, wavelength=float(wl),
                                            field_angle=tuple(fa))
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # Per-(wavelength, field) evaluation can fail at
                    # extreme corners; substitute the worst-case
                    # weight so the optimizer steers away rather than
                    # crashing.
                    p = self.weight
                penalties.append(p)
        # Arithmetic mean across all (wavelength, field) combinations.
        return float(np.mean(penalties))

    def _evaluate_one(self, ctx, wavelength, field_angle):
        """Compute the merit for a single wavelength + field-angle pair."""
        E_in = self._make_source(ctx, wavelength, field_angle)
        E_ideal = self._propagate(self.ideal_elements, E_in, ctx, wavelength)
        real_elems = self._build_real_elements(ctx)
        E_real = self._propagate(real_elems, E_in, ctx, wavelength)

        if E_ideal.shape != E_real.shape:
            return self.weight

        mask = self.aperture_mask
        if mask is not None:
            E_ideal = E_ideal * mask
            E_real = E_real * mask

        # Optional axial focus search: find the z-offset where the
        # real field best matches the ideal's radiation pattern.  This
        # decouples "correct focal plane" from "aberration quality" so
        # a small BFL shift introduced by lens thickness doesn't
        # dominate the penalty.
        if self.focus_search:
            return self._focus_search_penalty(
                E_ideal, E_real, ctx, wavelength)

        return self._compute_penalty(E_ideal, E_real)

    def _focus_search_penalty(self, E_ideal, E_real, ctx, wavelength):
        """Propagate E_real through a small range of z offsets, pick
        the one that minimises the penalty (i.e. maximises overlap),
        and return that value.  Uses ASM (fast, exact, preserves dx).
        """
        from ..propagators.propagation import angular_spectrum_propagate
        # Default range: +-f/20 where f ~= ctx.efl or ctx.bfl; fall
        # back to +-5 mm if neither is available.
        if self.focus_search_range is not None:
            z_lo, z_hi = self.focus_search_range
        else:
            ref = ctx.bfl if (ctx.bfl and np.isfinite(ctx.bfl)
                                and abs(ctx.bfl) < 10) else ctx.efl
            if ref and np.isfinite(ref) and abs(ref) < 10:
                half = max(abs(ref) / 20.0, 1e-4)
            else:
                half = 5e-3
            z_lo, z_hi = -half, +half
        zs = np.linspace(z_lo, z_hi, max(3, self.focus_search_n))
        best = self.weight  # worst-case sentinel
        for dz in zs:
            E_shifted = (E_real if dz == 0.0
                         else angular_spectrum_propagate(
                             E_real, float(dz), wavelength, ctx.dx,
                             bandlimit=True))
            p = self._compute_penalty(E_ideal, E_shifted)
            if p < best:
                best = p
        return best

    def _compute_penalty(self, E_ideal, E_real):
        if self.match == 'field_overlap':
            return self._field_overlap_penalty(E_ideal, E_real)
        if self.match == 'field_mse':
            return self._field_mse_penalty(E_ideal, E_real)
        if self.match == 'intensity_mse':
            return self._intensity_mse_penalty(E_ideal, E_real)
        # intensity_overlap
        return self._intensity_overlap_penalty(E_ideal, E_real)

    # -- Metric kernels ---------------------------------------------

    @staticmethod
    def _field_overlap_penalty_raw(E_ideal, E_real):
        """Return (1 - coupling_efficiency).  Returns 1.0 if either
        field is zero (worst case)."""
        num = abs(np.vdot(E_ideal.ravel(), E_real.ravel())) ** 2
        p_i = float(np.sum(np.abs(E_ideal) ** 2))
        p_r = float(np.sum(np.abs(E_real) ** 2))
        den = p_i * p_r
        if den < 1e-60:
            return 1.0
        overlap = float(num / den)
        return 1.0 - overlap

    def _field_overlap_penalty(self, E_ideal, E_real):
        return self.weight * self._field_overlap_penalty_raw(E_ideal, E_real)

    def _field_mse_penalty(self, E_ideal, E_real):
        """Power-normalised, global-phase-aligned, squared L2 of the
        field residual.  Roughly the "fraction of energy in the
        difference" when amplitude-normalised."""
        p_i = float(np.sum(np.abs(E_ideal) ** 2))
        p_r = float(np.sum(np.abs(E_real) ** 2))
        if p_i < 1e-30 or p_r < 1e-30:
            return self.weight
        scale = np.sqrt(p_i / p_r)
        inner = np.vdot(E_ideal.ravel(), E_real.ravel())
        phase_align = (np.conj(inner) / abs(inner)) if abs(inner) > 1e-30 else 1.0 + 0.0j
        E_real_aligned = E_real * scale * phase_align
        mse = float(np.sum(np.abs(E_ideal - E_real_aligned) ** 2) / p_i)
        return self.weight * mse

    def _intensity_mse_penalty(self, E_ideal, E_real):
        """Phase-blind: compares |E|^2 patterns, normalised to equal
        total power."""
        I_i = np.abs(E_ideal) ** 2
        I_r = np.abs(E_real) ** 2
        p_i = float(np.sum(I_i))
        p_r = float(np.sum(I_r))
        if p_i < 1e-30 or p_r < 1e-30:
            return self.weight
        I_r_norm = I_r * (p_i / p_r)
        return self.weight * float(np.sum((I_i - I_r_norm) ** 2) / (p_i ** 2))

    def _intensity_overlap_penalty(self, E_ideal, E_real):
        I_i = np.abs(E_ideal) ** 2
        I_r = np.abs(E_real) ** 2
        num = float(np.sum(I_i * I_r))
        den = np.sqrt(float(np.sum(I_i ** 2)) * float(np.sum(I_r ** 2)))
        if den < 1e-30:
            return self.weight
        return self.weight * (1.0 - num / den)

    # -- Convenience -------------------------------------------------

    @classmethod
    def single_lens(cls, focal_length, post_distance=None, **kwargs):
        """Shortcut for the single-lens drop-in replacement case.

        Generates ``ideal_elements`` = [thin_lens(f), propagate(z=f
        or post_distance)].  Equivalent real_elements is the default
        single-``_prescription_`` drop-in.

        Parameters
        ----------
        focal_length : float
            Ideal thin-lens focal length [m].
        post_distance : float, optional
            Propagation distance from the lens to the output plane
            [m].  Default: ``focal_length`` (i.e. evaluate at the
            paraxial focus).
        kwargs : forwarded to :meth:`__init__`.
        """
        post = float(focal_length) if post_distance is None else float(post_distance)
        ideal = [
            {'type': 'lens', 'f': float(focal_length)},
            {'type': 'propagate', 'z': post},
        ]
        real_elems = kwargs.pop('real_elements', None)
        if real_elems is None:
            real_elems = [
                {'type': '_prescription_'},
                {'type': 'propagate', 'z': post},
            ]
        return cls(ideal_elements=ideal,
                   real_elements=real_elems,
                   **kwargs)


class MatchTargetOPDMerit(MeritTerm):
    """Penalise deviation of the actual exit-pupil OPD from a
    user-supplied target OPD map (or callable returning one).

    Use this when you have a desired wavefront -- not necessarily a
    perfect sphere -- that the lens should produce at its exit pupil.
    Examples: matching a measured wavefront, copying an existing
    well-corrected design, or shaping a beam with a target phase
    profile.

    Parameters
    ----------
    target_opd : ndarray or callable
        - If ndarray (shape (Ny, Nx) matching the simulation grid):
          used directly.
        - If callable: called as ``target_opd(X, Y, prescription)``
          and expected to return an ndarray of OPD [m] over the
          pupil grid.  Useful when the target depends on the current
          prescription (e.g. "match a target with the same EFL").
    weight : float, default 1.0
    exclude_low_order : int, default 1
        Number of Zernike modes to remove from the residual before
        computing RMS.
    n_modes : int, default 21
    """

    needs_wave = True
    name = 'MatchTargetOPD'

    def __init__(self, target_opd: Union[np.ndarray, Callable],
                 weight: float = 1.0,
                 exclude_low_order: int = 1,
                 n_modes: int = 21) -> None:
        self.target_opd = target_opd
        self.weight = float(weight)
        self.exclude_low_order = int(exclude_low_order)
        self.n_modes = int(n_modes)

    def evaluate(self, ctx: Any) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        Ny, Nx = ctx.opd_map.shape
        x = (np.arange(Nx) - Nx / 2) * ctx.dx
        y = (np.arange(Ny) - Ny / 2) * ctx.dx
        X, Y = np.meshgrid(x, y)
        if callable(self.target_opd):
            target = np.asarray(self.target_opd(X, Y, ctx.prescription))
        else:
            target = np.asarray(self.target_opd)
        if target.shape != ctx.opd_map.shape:
            raise ValueError(
                f'target_opd shape {target.shape} does not match '
                f'opd_map shape {ctx.opd_map.shape}')
        diff = ctx.opd_map - target
        from ..analysis import zernike_decompose
        finite = np.isfinite(diff)
        diff_clean = np.where(finite, diff, 0.0)
        try:
            coeffs, _ = zernike_decompose(
                diff_clean, ctx.dx, ap, n_modes=self.n_modes)
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                ZeroDivisionError):
            return 0.0
        higher = coeffs[self.exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        rms_waves = rms_m / ctx.wavelength
        return self.weight * rms_waves * rms_waves


class ZernikeCoefficientMerit(MeritTerm):
    """Penalise (or target) specific Zernike-mode coefficients of the
    actual exit-pupil OPD.

    Lets you express design intents like:
    - "Eliminate spherical aberration" -- target mode 12 (Z_4^0) = 0
    - "Allow some defocus but no tilt or coma" -- target tilts (1,2),
      vertical/horizontal coma (7,8), and trefoils (6,9) all = 0
    - "Match a measured aberration profile mode-by-mode"

    Parameters
    ----------
    targets : dict of {int: float}
        Map from OSA Zernike index to target coefficient [m].  Modes
        not in this dict are unconstrained.
    weight : float, default 1.0
    n_modes : int, default 21
        Number of modes to fit (must exceed max key in ``targets``).
    """

    needs_wave = True
    name = 'ZernikeCoefficient'

    def __init__(self, targets: Dict[int, float],
                 weight: float = 1.0, n_modes: int = 21) -> None:
        self.targets = {int(j): float(v) for j, v in targets.items()}
        self.weight = float(weight)
        self.n_modes = max(int(n_modes), max(self.targets) + 1
                           if self.targets else int(n_modes))

    def evaluate(self, ctx: Any) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        from ..analysis import zernike_decompose
        finite = np.isfinite(ctx.opd_map)
        opd_clean = np.where(finite, ctx.opd_map, 0.0)
        try:
            coeffs, _ = zernike_decompose(
                opd_clean, ctx.dx, ap, n_modes=self.n_modes)
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                ZeroDivisionError):
            return 0.0
        total = 0.0
        for j, target in self.targets.items():
            err_waves = (coeffs[j] - target) / ctx.wavelength
            total = total + err_waves * err_waves
        return self.weight * total


class LGAberrationMerit(MeritTerm):
    """Penalise specified Laguerre-Gaussian aberration-tensor channels
    via the closed-form modal asymptotic propagator (
    Section 7).

    Each entry ``L_{(p, ell), n}(s_2^img)`` of the LG aberration tensor
    is the projection of the system's leading-order asymptotic
    image-plane field onto a named classical aberration channel:
    ``(0, 0)`` is piston/Strehl, ``(1, 0)`` is defocus, ``(2, 0)`` is
    primary spherical, ``(0, +-1)`` is tilt, ``(1, +-1)`` is coma,
    ``(0, +-2)`` is astigmatism, ``(0, +-3)`` is trefoil.  Driving a
    given ``|L_{(p, ell), 0}|^2`` to zero suppresses that aberration
    in the merit-function loop without invoking the wave leg.

    The merit is computed from a Chebyshev tensor-product fit of the
    prescription's canonical map ``Phi(s2, v2), s1(s2, v2)``
    -- a single fit drives all targeted
    aberration channels at all chosen field points.

    Parameters
    ----------
    targets : dict
        Map from output LG index ``(p, ell)`` to a float weight.  Each
        listed channel contributes ``|L_{(p, ell), 0}(s_2^img)|^2``
        times the entry weight.  Channels not listed are unconstrained.
        Common targets:

            {(2, 0): 1.0, (1, 1): 1.0, (1, -1): 1.0, (0, 2): 1.0, (0, -2): 1.0}

        suppresses primary spherical, both coma orientations, and
        both astigmatism orientations.
    field_points : list of (float, float), optional
        Source-plane points [m] at which to evaluate the tensor.
        Default: a single on-axis point ``[(0.0, 0.0)]``.  Each field
        point's contribution is summed.
    image_points : list of (float, float), optional
        Image-plane evaluation points (one per field point).  If None,
        defaults to the chief-ray landing of each source point (which
        the merit evaluator finds via Newton).
    w_s, w_p : float
        Source-plane and pupil-plane Gaussian waists [m and direction
        cosine].  Defaults: ``w_s = 50e-6`` (50 um), ``w_p = 0.05``
        (50 mrad).
    w_o : float, optional
        Output Gaussian waist [m].  Default: derived per-pixel from
        the local complex beam matrix.
    fit_kwargs : dict, optional
        Additional keyword arguments passed to
        ``fit_canonical_polynomials``:  ``poly_order``,
        ``source_box_half``, ``pupil_box_half``, ``n_field``,
        ``n_pupil``, ``extract_linear_phase``, ``object_distance``.
    weight : float, default 1.0
    name : str, optional

    See Also
    --------
    lumenairy.asymptotic.aberration_tensor :  raw tensor evaluator.
    SphericalSeidelMerit :  Seidel-coefficient-based primary-spherical
        merit (uses paraxial coefficients; LGAberrationMerit's full
        non-paraxial generalisation is preferred for high-NA work).
    """

    needs_wave = False
    name = 'LGAberration'

    def __init__(self, targets: Dict[Any, float],
                 field_points: Optional[Sequence[Any]] = None,
                 image_points: Optional[Sequence[Any]] = None,
                 w_s: float = 50e-6, w_p: float = 0.05,
                 w_o: Optional[float] = None,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 weight: float = 1.0,
                 name: Optional[str] = None) -> None:
        if not targets:
            raise ValueError("LGAberrationMerit: targets dict is empty")
        self.targets = {tuple(k): float(v) for k, v in targets.items()}
        if field_points is None:
            field_points = [(0.0, 0.0)]
        self.field_points = [tuple(p) for p in field_points]
        if image_points is None:
            self.image_points = None
        else:
            ips = list(image_points)
            if len(ips) != len(self.field_points):
                raise ValueError(
                    f"LGAberrationMerit: image_points length "
                    f"{len(ips)} must match field_points length "
                    f"{len(self.field_points)}")
            self.image_points = [tuple(p) for p in ips]
        self.w_s = float(w_s)
        self.w_p = float(w_p)
        self.w_o = None if w_o is None else float(w_o)
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}
        self.weight = float(weight)
        if name is not None:
            self.name = str(name)

    def evaluate(self, ctx: Any) -> float:
        # Lazy import to avoid bootstrap cycles.
        from ..propagators.asymptotic import (fit_canonical_polynomials,
                                  aberration_tensor)
        # Canonical-fit cache: when a CompositeMerit contains several
        # LGAberrationMerit terms with the same fit_kwargs (typical:
        # one term per emitter class -- centre / edge / corner -- all
        # using the same source/pupil sampling box), build the fit
        # once per merit eval and reuse across terms.  Cache lives
        # only for the lifetime of one ctx; the next merit_fn(x) call
        # gets a fresh context.  Audit perf #2 (3.5.5).
        try:
            cache = ctx._canonical_fit_cache
        except AttributeError:
            cache = None
        # Hash key: wavelength + repr of sorted fit_kwargs items.
        # repr() handles nested dicts (e.g. surface_diffraction).
        # Different field_points with the SAME fit_kwargs share the
        # fit -- the fit is purely a property of (prescription,
        # wavelength, fit_kwargs); field_points are evaluation points
        # within the fit's domain.
        cache_key = None
        if cache is not None:
            try:
                cache_key = (float(ctx.wavelength),
                             repr(sorted(self.fit_kwargs.items(),
                                         key=lambda kv: kv[0])))
            except (TypeError, ValueError, AttributeError):
                # Unhashable / unsortable fit_kwargs (rare -- user
                # passed a non-stringifiable value).  Skip caching.
                cache_key = None
        if cache_key is not None and cache_key in cache:
            fit = cache[cache_key]
            if fit is None:
                # Previous attempt at this fit failed; short-circuit
                # to the same penalty.
                return 1e20
        else:
            try:
                fit = fit_canonical_polynomials(
                    ctx.prescription,
                    wavelength=ctx.wavelength,
                    **self.fit_kwargs,
                )
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # If the fit can't be built (e.g., aperture clipping
                # kills too many rays for the current prescription),
                # assign a large penalty so the optimiser steers away.
                if cache_key is not None and cache is not None:
                    cache[cache_key] = None
                return 1e20
            if cache_key is not None and cache is not None:
                cache[cache_key] = fit

        # The output modes are exactly the target keys, plus (0, 0)
        # for piston (always useful diagnostically).
        target_keys = list(self.targets.keys())
        output_modes = list(set([(0, 0)] + target_keys))

        total = 0.0
        for ifp, src in enumerate(self.field_points):
            if self.image_points is None:
                # Use the nominal chief-ray landing computed from the
                # paraxial back-map at v2_centre = 0.  fit's s2_centre/
                # halfrange box is centered on the actual landing
                # distribution, so s2_centre is a good chief estimate.
                s2_img = (fit.s2x_centre, fit.s2y_centre)
            else:
                s2_img = self.image_points[ifp]
            try:
                tensor = aberration_tensor(
                    fit,
                    s2_image=s2_img,
                    source_point=src,
                    source_modes=[(0, 0)],
                    pupil_modes=[(0, 0)],
                    output_modes=output_modes,
                    w_s=self.w_s, w_p=self.w_p, w_o=self.w_o,
                )
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # Aberration tensor evaluation can diverge on
                # singular pupil sums or out-of-domain field points;
                # max-penalty so the optimizer steers away.
                return 1e20
            # Index of each target in output_modes
            idx_map = {m: i for i, m in enumerate(output_modes)}
            for (p, ell), wgt in self.targets.items():
                try:
                    i = idx_map[(p, ell)]
                except KeyError:
                    continue
                val = complex(tensor.L[i, 0])
                total = total + wgt * (val.real * val.real
                                       + val.imag * val.imag)

        return self.weight * total


def make_lg_aberration_merit_jax(prescription: Dict[str, Any],
                                 wavelength: float,
                                 targets: Dict[Any, float],
                                 build_args: Callable,
                                 *,
                                 field_points: Optional[Sequence[Any]] = None,
                                 image_points: Optional[Sequence[Any]] = None,
                                 w_s: float = 50e-6,
                                 w_p: float = 0.05,
                                 poly_order: int = 4,
                                 n_field: int = 8, n_pupil: int = 8,
                                 weight: float = 1.0,
                                 name: str = 'LGAberrationJax') -> "JaxMeritTerm":
    """Build a JAX-grad-compatible LG-aberration merit term.

    Wraps :func:`fit_canonical_polynomials_jax` +
    :func:`aberration_tensor_lg00_jax` into a :class:`JaxMeritTerm`
    so :func:`design_optimize` picks up its analytic gradient via
    the existing ``jac='auto'`` routing.  The merit returns the
    weighted sum of ``|L_{(p, ell), (0,0)}|^2`` across all targets
    and field points, just like :class:`LGAberrationMerit`.

    *Currently supported as differentiable inputs* (anything you map
    into via ``build_args``):

    * ``wavelength`` (chromatic optimisation)
    * ``source_box_half``, ``pupil_box_half`` (fit-domain knobs;
      useful for fit-quality optimisation but rarely a design free
      var)
    * ``object_distance`` (source-plane positioning)
    * ``w_s``, ``w_p`` (source / pupil Gaussian waists)

    *NOT currently supported as differentiable inputs:* radii,
    thicknesses, conics, aspheric coefficients.  These are read by
    :func:`trace_jax` as static Python floats; making the trace
    differentiable in prescription parameters is on the roadmap (a
    "trace_jax with JAX-array surfaces" extension) but not in this
    release.  Until then, design-parameter optimisation uses the
    existing FD path, which is still adequate for systems with
    O(30-50) free vars.

    Parameters
    ----------
    prescription : dict
        Static lumenairy prescription (radii, conics, thicknesses,
        glass).  Treated as a closure constant by JAX.
    wavelength : float
        Wavelength [m] -- can be made differentiable via
        ``build_args``.
    targets : dict[(int, int), float]
        LG channels to penalise; same format as
        :class:`LGAberrationMerit`.
    build_args : callable
        ``build_args(x: ndarray) -> tuple`` mapping the parameter
        vector to differentiable scalars passed positionally to
        the inner JAX function.  The inner function's signature is
        ``fn(wavelength, source_box_half, pupil_box_half,
        object_distance, w_s, w_p)`` -- match the order of returned
        scalars to those positions, or pass ``None`` placeholders
        for static values.

    Other parameters mirror :class:`LGAberrationMerit`.

    Returns
    -------
    merit : JaxMeritTerm
        Plug into :class:`CompositeMerit` like any other merit
        term.  ``design_optimize(jac='auto', ...)`` will use
        ``merit.gradient_at_x`` automatically.

    Examples
    --------
    Optimise the source-Gaussian waist ``w_s`` to minimise primary
    spherical aberration::

        def build_args(x):
            return (None, None, None, None, x[0], None)

        merit = make_lg_aberration_merit_jax(
            prescription, wavelength=1.31e-6,
            targets={(2, 0): 1.0},
            build_args=build_args,
            field_points=[(0.0, 0.0)],
        )
        # x[0] = w_s, optimise via design_optimize.
    """
    from ..backend import JAX_AVAILABLE
    if not JAX_AVAILABLE:
        raise ImportError(
            "make_lg_aberration_merit_jax requires JAX.  "
            "Install with `pip install jax`.")
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp
    from ..propagators.asymptotic import (
        fit_canonical_polynomials_jax,
        aberration_tensor_lg00_jax,
    )
    if field_points is None:
        field_points = [(0.0, 0.0)]

    targets_kv = [(tuple(k), float(v)) for k, v in targets.items()]
    # v4.13.2 (C-P0-1): aberration_tensor_lg00_jax only computes the
    # (0, 0) -> (0, 0) -> (0, 0) coefficient.  General (p, ell)
    # targets need a full aberration_tensor_jax that does not yet
    # exist.  Reject non-(0, 0) targets at construction time with a
    # clear migration message so the silent ``pass``-only loop bug
    # (the merit returning the same piston sum regardless of the
    # ``targets`` dict) cannot recur.
    non_piston = [k for k, _ in targets_kv if k != (0, 0)]
    if non_piston:
        raise NotImplementedError(
            f"make_lg_aberration_merit_jax: targets other than "
            f"(0, 0) are not supported in the JAX path "
            f"(got {non_piston}).  aberration_tensor_lg00_jax "
            f"computes the (0, 0) Strehl amplitude only.  Use the "
            f"NumPy sibling :class:`LGAberrationMerit` for general "
            f"(p, ell) targets, or restrict ``targets`` to "
            f"``{{(0, 0): wgt}}``.")
    # Piston weight: default to 1.0 if (0, 0) is absent so the merit
    # still has well-defined semantics (matches the LGAberrationMerit
    # behaviour of always including piston in output_modes).
    piston_weight = 1.0
    for k, w in targets_kv:
        if k == (0, 0):
            piston_weight = float(w)
            break
    nominal_w_s = float(w_s)
    nominal_w_p = float(w_p)
    nominal_lambda = float(wavelength)
    nominal_obj_d = float(prescription.get('object_distance', 0.0) or 0.0)
    # Use these defaults for any None placeholders in build_args.
    nominal = (nominal_lambda, 50e-6, 0.05, nominal_obj_d,
                nominal_w_s, nominal_w_p)

    # Import the IFT-based envelope solver so v_star is JAX-traceable
    # (required positional argument of aberration_tensor_lg00_jax).
    from ..propagators.asymptotic import (
        solve_envelope_stationary_jax_ift,
    )

    def fn(*args):
        # Resolve None placeholders to nominals.
        wl, sbh, pbh, obj_d, w_s_local, w_p_local = (
            (a if a is not None else default)
            for a, default in zip(args, nominal))
        fit = fit_canonical_polynomials_jax(
            prescription, float(wl) if not hasattr(wl, 'shape') else wl,
            source_box_half=sbh, pupil_box_half=pbh,
            n_field=n_field, n_pupil=n_pupil,
            poly_order=poly_order,
            object_distance=float(obj_d) if not hasattr(obj_d, 'shape') else obj_d,
        )
        total = jnp.array(0.0)
        for ifp, src in enumerate(field_points):
            if image_points is not None:
                s2_img = image_points[ifp]
            else:
                s2_img = (fit.s2x_centre, fit.s2y_centre)
            # v_star is required positionally; compute via the
            # IFT-based solver so gradients still flow back through
            # build_args inputs (w_s, w_p, s2, source_point).
            v_star = solve_envelope_stationary_jax_ift(
                fit, s2_img, tuple(src),
                w_s=w_s_local, w_p=w_p_local,
                v2_centre=(fit.v2x_centre, fit.v2y_centre))
            res = aberration_tensor_lg00_jax(
                fit, s2_img, v_star,
                source_point=tuple(src),
                w_s=w_s_local, w_p=w_p_local,
                v2_centre=(fit.v2x_centre, fit.v2y_centre))
            # res is a complex scalar (the L_{(0,0),(0,0)} element);
            # weight by the user-supplied piston weight so changing
            # ``targets={(0, 0): w}`` actually scales the merit
            # linearly in w (the pre-fix code ignored ``wgt`` and
            # always returned the same |L|^2 sum).
            total = total + piston_weight * (jnp.abs(res) ** 2)
        return total

    return JaxMeritTerm(
        fn=fn, weight=weight, name=name,
        real_part=True,         # fn already returns real
        build_args=build_args,
    )


class CompositeMerit(MeritTerm):
    """Combine multiple sub-merits into one weighted sum.

    Useful for composing a complex objective from simpler pieces,
    or for grouping merits that share an expensive intermediate
    (e.g., the Zernike decomposition of the exit-pupil OPD).
    """

    name = 'Composite'

    def __init__(self, sub_merits: Sequence[MeritTerm],
                 weight: float = 1.0) -> None:
        self.sub_merits = list(sub_merits)
        self.weight = float(weight)
        self.needs_wave = any(m.needs_wave for m in self.sub_merits)

    def evaluate(self, ctx: Any) -> float:
        s = 0.0
        for m in self.sub_merits:
            s = s + m.evaluate(ctx)
        return self.weight * s


class CallableMerit(MeritTerm):
    """Generic merit term that delegates to a user-supplied callable.

    Use for one-off custom objectives that don't fit the prebuilt
    classes:

        def my_merit(ctx):
            # ctx.efl, ctx.bfl, ctx.seidel, ctx.E_exit, ctx.opd_map,
            # ctx.strehl_best, ctx.rms_radius_best, ctx.prescription, ...
            return some_scalar

        merit = CallableMerit(my_merit, weight=1.0, needs_wave=True)
    """

    name = 'Callable'

    def __init__(self, fn: Callable, weight: float = 1.0,
                 needs_wave: bool = False,
                 name: Optional[str] = None) -> None:
        self.fn = fn
        self.weight = float(weight)
        self.needs_wave = bool(needs_wave)
        if name is not None:
            self.name = name

    def evaluate(self, ctx: Any) -> float:
        return self.weight * float(self.fn(ctx))


class JaxMeritTerm(MeritTerm):
    """Differentiable merit term backed by a JAX-traceable callable.

    Has two operating modes:

    1. **Forward-only** (the default).  ``fn(ctx) -> JAX scalar``
       runs once per merit evaluation; the gradient w.r.t. design
       parameters comes from SciPy's finite-difference Jacobian.
       Useful for quick experimentation.

    2. **Differentiable in x** -- pass ``build_args=callable(x)``
       returning a tuple of JAX-traceable inputs to ``fn``.  In this
       mode :meth:`gradient_at_x` returns ``d(weight * |fn|) / dx``
       via :func:`jax.grad`, and :func:`design_optimize` will pick
       these gradients up automatically when ``jac='auto'`` (default
       there).  This closes the loop on JAX-differentiable merits:
       analytic gradients flow into SciPy without finite differences.

    Parameters
    ----------
    fn : callable
        Either ``fn(ctx) -> JAX scalar`` (mode 1) or
        ``fn(*args) -> JAX scalar`` (mode 2; args provided by
        ``build_args``).
    weight : float
    needs_wave : bool, default False
        Set True if the inner JAX function reads ``ctx.E_exit``.
        Forces the wave leg to run during :meth:`evaluate`.
    name : str, optional
    real_part : bool, default False
        If True, return ``real(fn(...))`` instead of ``|fn(...)|``.
        Useful when the JAX evaluator already returns a real scalar
        (e.g. RMS spot size).
    build_args : callable, optional
        ``build_args(x: ndarray) -> tuple of JAX scalars`` mapping a
        parameter vector to the positional arguments of ``fn``.
        Required for analytic gradient propagation through
        :func:`design_optimize`.

    Notes
    -----
    The forward-mode ``evaluate(ctx)`` works the same in both modes.
    When ``build_args`` is provided and ``ctx.x`` exists,
    ``evaluate`` calls ``fn(*build_args(ctx.x))`` (matching the
    gradient path).  Without ``ctx.x``, it falls back to the legacy
    ``fn(ctx)`` form -- letting the same JaxMeritTerm be used both
    inside ``design_optimize`` and in standalone evaluations.
    """

    name = 'JaxMerit'

    def __init__(self, fn: Callable, weight: float = 1.0,
                 needs_wave: bool = False,
                 name: Optional[str] = None,
                 real_part: bool = False,
                 build_args: Optional[Callable] = None) -> None:
        self.fn = fn
        self.weight = float(weight)
        self.needs_wave = bool(needs_wave)
        self.real_part = bool(real_part)
        self.build_args = build_args
        if name is not None:
            self.name = name

    def supports_jax_grad(self) -> bool:
        """True if this merit can produce an analytic gradient via
        :func:`jax.grad` on the parameter vector x."""
        return self.build_args is not None

    def _reduce(self, val):
        """Reduce a complex / array JAX value to a real scalar
        (matching :meth:`evaluate`'s output)."""
        import jax.numpy as jnp
        if self.real_part:
            return jnp.real(val)
        return jnp.abs(val)

    def evaluate(self, ctx: Any) -> float:
        if self.build_args is not None and getattr(ctx, 'x', None) is not None:
            val = self.fn(*self.build_args(ctx.x))
        else:
            val = self.fn(ctx)
        # Bridge JAX -> NumPy at the merit boundary.
        try:
            import numpy as _np
            arr = _np.asarray(val)
        except (TypeError, ValueError):
            # val is a non-array-like scalar (e.g. a Python float
            # returned from a non-JAX merit); fall through to the
            # scalar coercion path below.
            return self.weight * float(val)
        if self.real_part:
            return self.weight * float(arr.real)
        return self.weight * float(abs(arr))

    def gradient_at_x(self, x):
        """Analytic gradient of ``weight * reduction(fn(...))`` wrt x.

        Returns a NumPy array of shape ``(len(x),)``.  Requires
        ``build_args`` to have been supplied at construction.
        """
        if self.build_args is None:
            raise RuntimeError(
                "JaxMeritTerm.gradient_at_x requires build_args to be "
                "set; either pass build_args=... at construction or "
                "fall back to finite differences.")
        import jax
        import jax.numpy as jnp
        weight = self.weight

        def _scalar(x_jax):
            args = self.build_args(x_jax)
            val = self.fn(*args)
            return weight * self._reduce(val)

        # Use JAX's default float dtype -- requesting float64 raises
        # a UserWarning when jax_enable_x64 isn't set, which is the
        # common case for a fresh ``import jax``.
        x_jax = jnp.asarray(x)
        g = jax.grad(_scalar)(x_jax)
        return np.asarray(g)


class ChromaticFocalShiftMerit(MeritTerm):
    """Penalise focal-length variation across wavelengths.

    4.10.2: this term is now self-contained.  Pass the wavelengths
    explicitly at construction.  Pre-4.10.2 it depended on
    ``ctx.efls_per_wavelength`` being populated as a SIDE EFFECT of
    a prior ``MultiWavelengthMerit.evaluate()`` call earlier in the
    merit-term list -- if the ordering put this term first, the
    constraint silently disabled.

    Parameters
    ----------
    wavelengths : sequence of float, optional
        Wavelengths [m] to evaluate the EFL at.  When ``None`` falls
        back to the pre-4.10.2 behaviour of reading
        ``ctx.efls_per_wavelength`` (which requires a
        ``MultiWavelengthMerit`` to populate it earlier in the term
        list).
    """

    needs_wave = False
    name = 'ChromaticFocalShift'

    def __init__(self, weight: float = 1.0,
                 wavelengths: Optional[Sequence[float]] = None) -> None:
        self.weight = float(weight)
        self.wavelengths = ([float(w) for w in wavelengths]
                            if wavelengths is not None else None)

    def evaluate(self, ctx: Any) -> float:
        if self.wavelengths is not None:
            # 4.10.2: self-contained per-wavelength EFL evaluation.
            from ..raytrace import (surfaces_from_prescription,
                                     system_abcd)
            efls = []
            for wl in self.wavelengths:
                try:
                    surfs = surfaces_from_prescription(ctx.prescription)
                    _, efl, _, _ = system_abcd(surfs, wl)
                    if np.isfinite(efl):
                        efls.append(float(efl))
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # EFL unobtainable at this wavelength (degenerate
                    # ABCD / missing glass entry); drop from the
                    # spread accumulator.
                    pass
            if len(efls) < 2:
                return 0.0
            pv = max(efls) - min(efls)
            return self.weight * pv * pv
        # Legacy fallback: read context-attached EFLs.
        if (getattr(ctx, 'efls_per_wavelength', None) is None
                or len(ctx.efls_per_wavelength) < 2):
            return 0.0
        pv = (np.max(ctx.efls_per_wavelength)
              - np.min(ctx.efls_per_wavelength))
        return self.weight * pv * pv


# =========================================================================
# Wrapper-merit meshgrid cache (v4.14.0 perf)
# =========================================================================
#
# MultiWavelengthMerit, MultiFieldMerit, and ToleranceAwareMerit each
# rebuild np.indices / meshgrid arrays + aperture mask on every per-
# wavelength / per-field / per-trial leg.  For a 5 wavelengths x 5
# fields x 40 FD evals run at N=512 that is up to 1000 N x N meshgrid
# builds per optimisation step, none of which depend on the parameter
# vector being differenced.
#
# This module-level LRU cache memoises the wavelength/field/trial-
# invariant payload keyed on (Ny, Nx, dx, aperture_hash, dtype_str).
# Per-leg cost reduces to a single np.exp(1j * sin_a * cached_k0_Y) *
# cached_aperture_mask (MultiFieldMerit) or a single .copy() (the
# other two) plus the apply_real_lens call.
#
# Eval-count pin: the counter _WRAPPER_MERIT_MESHGRID_BUILDS records
# every actual meshgrid build (NOT cache hits) so tests can assert
# exactly one build per (N, dx, aperture) signature per optimisation
# run.

_WRAPPER_MERIT_CACHE: 'OrderedDict[tuple, dict]' = OrderedDict()
_WRAPPER_MERIT_CACHE_SIZE = 32
_WRAPPER_MERIT_MESHGRID_BUILDS = 0
# v4.14.1 (P2-1): guard concurrent get / move_to_end / __setitem__ /
# popitem(last=False) on _WRAPPER_MERIT_CACHE.  Follows the
# _ASM_CACHE_LOCK precedent in propagators/propagation.py.  Without
# this two threads racing through _get_wrapper_merit_cache could see a
# torn OrderedDict.
_WRAPPER_MERIT_CACHE_LOCK = threading.Lock()


# v4.14.1 (P1-NEW-1): sentinel meaning "aperture explicitly zero, block
# all light."  Distinguished from ``mask is None`` ("no aperture
# specified, use full grid").  Pre-v4.14.0 a scalar
# ``aperture_diameter=0`` produced an all-False boolean mask, which
# downstream apply_real_lens treated as "block all light"; v4.14.0
# collapsed that branch into ``mask=None``, flipping the semantics so
# ``aperture_diameter=0`` instead produced a grid-filling plane wave.
# Callers compare ``mask is _ZERO_APERTURE_MASK`` to detect the
# deliberate-zero case and zero their field accordingly.
#
# v4.15.1 (Agent E): now inherits from ``_deprecation._Sentinel`` to
# share the singleton-name registry + pickle-safe ``__reduce__``
# protocol.  Pre-v4.15.1 this class duplicated the singleton plumbing
# in 3 places (here, ``_AngleUnsetSentinel`` in ``polarization.py``,
# and ``_Sentinel`` in ``_deprecation.py``); none carried a
# ``__reduce__``, so pickling a sentinel produced a NEW instance on
# the receiving side and broke ``is``-identity checks in distributed
# merit evaluation / joblib caches.
from .._deprecation import _Sentinel as _Sentinel


class _ZeroApertureMaskSentinel(_Sentinel):
    """Singleton sentinel for aperture explicitly zero / blocked."""

    __slots__ = ()

    def __init__(self) -> None:
        # Use the existing repr-friendly name as the singleton key.
        super().__init__('_ZERO_APERTURE_MASK')


_ZERO_APERTURE_MASK = _ZeroApertureMaskSentinel()


# v4.15.2 (Agent E, AUDIT_V4_15_1 P2): three additional pre-existing
# sentinel patterns in this module are promoted to ``_Sentinel``
# subclasses for pickle-safety + ``is``-identity discoverability.  Pre-
# v4.15.2 these were bare scalar fallbacks (``1e9`` for invalid focal
# length, ``0.0`` for failed-scan Strehl, and a "fall-back-to-nominal"
# marker for perturbed-ABCD failures).  Scalar storage is preserved at
# the call sites for arithmetic compatibility; the dedicated sentinel
# classes here register in ``_SENTINEL_REGISTRY`` so downstream consumers
# can perform identity checks (``ctx.efl is _INVALID_FL_SENTINEL_OBJ``)
# without breaking the existing magnitude-based ``ctx_is_valid`` path.
# Each carries a ``.value`` attribute holding its canonical scalar
# fallback so call sites that want the numeric form can ``float(s)`` or
# ``s.value``.  All three inherit ``__bool__ -> False`` from the base
# ``_Sentinel`` (matching ``_ZeroApertureMaskSentinel`` semantics).
#
# Naming convention: ``_<Concept>Sentinel`` for the class +
# ``_<CONCEPT>_SENTINEL_OBJ`` for the singleton.  ``_OBJ`` suffix
# distinguishes the new class-instance singletons from the pre-existing
# ``_INVALID_FL_SENTINEL = 1e9`` scalar at module top (kept for
# arithmetic uses in ``ctx_is_valid``).


class _InvalidFocalLengthSentinel(_Sentinel):
    """Identity-checkable singleton for "ABCD extraction failed -- focal
    length collapsed to the ``1e9`` magnitude-flag fallback".

    Used at line 2271 (the wave-leg ABCD failure branch).  Pre-v4.15.2
    that branch wrote a bare scalar ``efl = bfl = 1e9``; the magnitude-
    check downstream (``ctx_is_valid`` at line 467) recovered the
    "invalid" semantics by comparing ``abs(v) >= _INVALID_FL_SENTINEL *
    0.5``.  v4.15.2 keeps the scalar write (arithmetic stability) and
    adds this singleton so a future caller wanting a strict identity
    check (``ctx.efl is _INVALID_FL_SENTINEL_OBJ``) can opt in without
    breaking the existing magnitude path.
    """
    __slots__ = ()

    value: float = 1e9

    def __init__(self) -> None:
        super().__init__('_INVALID_FL_SENTINEL_OBJ')

    def __float__(self) -> float:
        return float(self.value)


_INVALID_FL_SENTINEL_OBJ = _InvalidFocalLengthSentinel()


class _FailedScanStrehlSentinel(_Sentinel):
    """Identity-checkable singleton for "through-focus Strehl scan
    failed -- Strehl collapsed to the safe ``0.0`` fallback".

    Used at line 2530 (the through-focus-scan exception branch).  Pre-
    v4.15.2 the branch wrote ``sub_ctx.strehl_best = 0.0``; the
    optimizer treats ``0.0`` as "very bad design" so the merit-leg
    contribution sinks into the noise floor without dragging the
    parameter vector further than the dispatcher's adaptive-step
    safeguards allow.  v4.15.2 keeps the scalar write and adds this
    singleton for identity discoverability.
    """
    __slots__ = ()

    value: float = 0.0

    def __init__(self) -> None:
        super().__init__('_FAILED_SCAN_STREHL_SENTINEL_OBJ')

    def __float__(self) -> float:
        return float(self.value)


_FAILED_SCAN_STREHL_SENTINEL_OBJ = _FailedScanStrehlSentinel()


class _PerturbedABCDFallbackSentinel(_Sentinel):
    """Identity-checkable singleton for "perturbed ABCD extraction
    failed -- fall back to nominal ``(ctx.efl, ctx.bfl)``".

    Used at line 2772 (the tolerance-perturbation ABCD failure branch).
    Pre-v4.15.2 the branch wrote ``efl_p, bfl_p = ctx.efl, ctx.bfl``
    -- a "stable but probably wrong" fallback that under-estimates the
    perturbed Strehl drop.  v4.15.2 keeps the scalar fallback writes
    and adds this singleton so a downstream consumer (e.g. a future
    tolerance-confidence wrapper) can detect that the perturbation's
    ABCD propagation degenerated to the nominal channel.
    """
    __slots__ = ()

    # No single ``.value`` here -- the fallback is a tuple-pattern, not
    # a single scalar.  Consumers either use this for ``is``-identity
    # or query ``ctx.efl`` / ``ctx.bfl`` for the actual numeric values.

    def __init__(self) -> None:
        super().__init__('_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ')


_PERTURBED_ABCD_FALLBACK_SENTINEL_OBJ = _PerturbedABCDFallbackSentinel()


def _wrapper_merit_aperture_key(aperture: Any) -> tuple:
    """Build a hashable key fragment representing the aperture state.

    Three branches:
      - ``None``  -> ``('none',)``.
      - ndarray   -> ``('arr', shape, dtype, content_hash)``.
        ``hash(np.ascontiguousarray(a).tobytes())`` captures content
        cheaply (a single ~N^2 byte scan; for N=512^2 complex128 that
        is ~4 MB which hashes in <1 ms).
      - scalar    -> ``('scalar', float)`` covering the common case of
        a single aperture_diameter taken from ``prescription``.
    """
    if aperture is None:
        return ('none',)
    if isinstance(aperture, np.ndarray):
        arr = np.ascontiguousarray(aperture)
        return ('arr', arr.shape, str(arr.dtype),
                hash(arr.tobytes()))
    # Scalar aperture: a Python int/float/np.floating.  Forced to a
    # plain float so np.float64(1.0) and 1.0 share the same cache key.
    return ('scalar', float(aperture))


def _get_wrapper_merit_cache(
    N: int, dx: float, aperture: Any, dtype: Any,
) -> Dict[str, Any]:
    """Return the cached (Y, X, mask, k0_Y_factor) payload for these
    grid + aperture parameters.

    The payload is a dict with keys ``'X'``, ``'Y'``, ``'mask'``,
    ``'Y_factor'`` where:

    - ``X``, ``Y``: the meshgrid coordinate arrays (shape (N, N),
      dtype float64).
    - ``mask``: boolean aperture mask (or ``None`` when ``aperture``
      is ``None``-or-zero).
    - ``Y_factor``: the wavelength-independent ``2*pi * Y / 1`` such
      that the per-wavelength tilt phase is
      ``(Y_factor / wavelength) * sin(theta_y)``.  Cached so the
      MultiFieldMerit per-leg work is one np.exp + one multiply.
    - ``r_squared``: ``X*X + Y*Y`` (cached for callers that need to
      build their own custom aperture masks against the same grid).

    The cache is LRU-bounded at 32 entries (``_WRAPPER_MERIT_CACHE_SIZE``).
    A meshgrid build increments ``_WRAPPER_MERIT_MESHGRID_BUILDS``;
    cache hits do NOT increment.  Use this counter in tests to pin
    the invariance contract.
    """
    global _WRAPPER_MERIT_MESHGRID_BUILDS

    Ny = Nx = int(N)
    dx_f = float(dx)
    # dtype may arrive as numpy dtype object, dtype string, or
    # np.complex128 type.  ``str(np.dtype(x))`` normalises to a
    # canonical short name ('complex128', 'complex64', ...).
    try:
        _dtype_obj = np.dtype(dtype)
        dtype_str = str(_dtype_obj)
    except TypeError:
        _dtype_obj = np.complex128
        dtype_str = str(dtype)
    ap_key = _wrapper_merit_aperture_key(aperture)
    key = (Ny, Nx, dx_f, ap_key, dtype_str)

    with _WRAPPER_MERIT_CACHE_LOCK:
        entry = _WRAPPER_MERIT_CACHE.get(key)
        if entry is not None:
            # LRU bookkeeping: refresh recency.
            _WRAPPER_MERIT_CACHE.move_to_end(key)
            return entry

    # Cache miss: build the grid + mask + Y-factor once and store.
    # The build itself is pure-CPU numpy and re-entrant; only the
    # OrderedDict get/move_to_end/set/popitem operations need the lock.
    _WRAPPER_MERIT_MESHGRID_BUILDS += 1
    Y_idx, X_idx = np.indices((Ny, Nx))
    X = (X_idx - Nx / 2) * dx_f
    Y = (Y_idx - Ny / 2) * dx_f
    r_squared = X * X + Y * Y

    if isinstance(aperture, np.ndarray):
        # Custom user-supplied aperture array; assume boolean-coercible.
        mask = np.asarray(aperture, dtype=bool)
        if mask.shape != (Ny, Nx):
            raise ValueError(
                f"_get_wrapper_merit_cache: aperture array shape "
                f"{mask.shape} != grid shape ({Ny}, {Nx})")
    elif aperture is None:
        # "No aperture specified" -- callers treat mask=None as
        # "full grid, no clipping."
        mask = None
    else:
        # Scalar aperture_diameter.
        ap_diam = float(aperture)
        if ap_diam > 0:
            mask = r_squared <= (ap_diam / 2.0) ** 2
        else:
            # v4.14.1 (P1-NEW-1): aperture explicitly <= 0 means
            # "block all light."  Distinct from the ``None`` branch
            # above (which means "no aperture specified, use full
            # grid").  Pre-v4.14.0 a scalar 0 produced an all-False
            # boolean mask; v4.14.0 erroneously collapsed it to None
            # and the downstream callers then treated the deliberate
            # zero as "no aperture -> full plane wave," flipping the
            # semantics.  Use a sentinel so callers can detect this
            # case via ``is`` and zero their fields explicitly.
            mask = _ZERO_APERTURE_MASK

    # Wavelength-independent Y-tilt factor: 2*pi * Y.  Per-leg the
    # tilt phase is (Y_factor / wavelength) * sin(theta_y) plus the
    # analogous X term.  Materialised so the per-leg cost is a
    # single multiply.
    Y_factor = (2.0 * np.pi) * Y
    X_factor = (2.0 * np.pi) * X

    # Cached np.ones array for ToleranceAwareMerit's per-trial
    # source field.  Stored once per (N, dtype); per-trial just
    # .copy() this and feed apply_real_lens.  apply_real_lens
    # never writes its input, but downstream merit code paths may
    # so the .copy() at call site preserves correctness.  Uses the
    # ``_dtype_obj`` computed at the head of the function.
    E_ones = np.ones((Ny, Nx), dtype=_dtype_obj)

    entry = {
        'X': X,
        'Y': Y,
        'mask': mask,
        'Y_factor': Y_factor,
        'X_factor': X_factor,
        'r_squared': r_squared,
        'E_ones': E_ones,
    }
    with _WRAPPER_MERIT_CACHE_LOCK:
        _WRAPPER_MERIT_CACHE[key] = entry
        while len(_WRAPPER_MERIT_CACHE) > _WRAPPER_MERIT_CACHE_SIZE:
            _WRAPPER_MERIT_CACHE.popitem(last=False)
    return entry


def _clear_wrapper_merit_cache() -> None:
    """Drop the wrapper-merit meshgrid cache and reset the build counter.

    v4.14.1 (P2-3): now invoked from
    :func:`lumenairy.propagators.propagation.clear_asm_caches` via a
    lazy import inside that function (the v4.14.0 monkey-patch is
    gone).  The reverse-direction dependency keeps optimize/core
    free of propagation-layer side-effects at import time while still
    leaving both caches pristine on a single ``clear_asm_caches()``
    call.  Also callable directly from tests.
    """
    global _WRAPPER_MERIT_MESHGRID_BUILDS
    with _WRAPPER_MERIT_CACHE_LOCK:
        _WRAPPER_MERIT_CACHE.clear()
    _WRAPPER_MERIT_MESHGRID_BUILDS = 0


# =========================================================================
# Multi-wavelength support
# =========================================================================

class MultiWavelengthMerit(MeritTerm):
    """Evaluate a sub-merit at multiple wavelengths and average.

    Populates ``ctx.efls_per_wavelength`` with per-wavelength EFLs
    (computed geometrically, cheap).  The sub-merit is evaluated at
    each wavelength and the results are summed.

    .. warning::
        The off-wavelength wave-leg propagation in this merit's
        ``evaluate`` always calls :func:`apply_real_lens` directly,
        irrespective of the ``wave_propagator`` selected on the
        enclosing :func:`design_optimize` call.  For high-NA designs
        optimised with ``wave_propagator='gbd'`` (or any non-real-lens
        backend) the off-nominal-wavelength penalty therefore exercises
        a different physical model than the on-axis wave leg.  A
        runtime warning fires from :func:`design_optimize` when this
        mismatch is detected.  Threading the propagator through the
        sub-merit is a v4.14+ feature -- see audit P2 #14.

    Parameters
    ----------
    wavelengths : sequence of float
        Wavelengths [m] to evaluate at.
    sub_merit : MeritTerm
        Merit term to evaluate at each wavelength.  Its ``evaluate``
        receives a modified ``ctx`` with the corresponding wavelength.
    weight : float
    """

    name = 'MultiWavelength'

    def __init__(self, wavelengths: Sequence[float],
                 sub_merit: MeritTerm, weight: float = 1.0) -> None:
        self.wavelengths = [float(w) for w in wavelengths]
        self.sub_merit = sub_merit
        self.weight = float(weight)
        self.needs_wave = sub_merit.needs_wave

    def evaluate(self, ctx: Any) -> float:
        # 4.10: re-evaluate the wave leg at each wavelength.  Pre-4.10
        # only EFL/BFL changed per-wavelength while E_exit, opd_map,
        # strehl_best, rms_radius_best were copied unchanged from ctx,
        # so wrapping StrehlMerit / RMSWavefrontMerit / MatchTargetOPDMerit
        # in MultiWavelengthMerit just averaged the same single-wavelength
        # number N times -- the chromatic-aberration penalty was a
        # no-op.  Now: for each wavelength, propagate the same input
        # field through apply_real_lens at that wavelength, run a
        # quick through-focus scan for Strehl, build an OPD map, and
        # populate the sub-context with these per-wavelength wave
        # quantities before delegating to the sub-merit.
        from ..analysis.through_focus import (
            through_focus_scan, find_best_focus, diffraction_limited_peak)
        from ..analysis.core import wave_opd_2d
        from ..elements import apply_real_lens
        efls = []
        per_wl_strehl = []
        per_wl_rms = []
        total = 0.0
        for wl in self.wavelengths:
            surfs = surfaces_from_prescription(ctx.prescription)
            try:
                _, efl, bfl, _ = system_abcd(surfs, wl)
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, TypeError):
                # Degenerate ABCD at this wavelength; sentinel-large
                # EFL/BFL nudges the wave leg toward the fallback
                # branch downstream.
                efl = bfl = 1e9
            efls.append(float(efl))
            sub_E_exit = ctx.E_exit
            sub_opd = ctx.opd_map
            sub_strehl = ctx.strehl_best
            sub_rms = ctx.rms_radius_best
            sub_z = ctx.z_best
            if self.sub_merit.needs_wave and ctx.E_exit is not None and \
               np.isfinite(bfl) and abs(bfl) < 10:
                try:
                    # Build a plane-wave input on ctx's grid and push
                    # through the prescription at this wavelength.
                    # 4.11.1: keyword call (apply_real_lens is keyword-only
                    # after E_in since 4.7); honour precision knob via
                    # get_default_complex_dtype; explicit is-None check on
                    # aperture_diameter so a deliberate 0 isn't shadowed
                    # by the grid-arbitrary fallback.
                    # v4.14.0: reuse the cached aperture mask /
                    # coordinate grids (see _get_wrapper_merit_cache);
                    # rebuilding np.indices on every per-wavelength
                    # leg was the dominant cost for 5+ wavelengths *
                    # 2N+1 FD evals per outer iteration.
                    N_pix = ctx.N
                    dx_pix = ctx.dx
                    ap = ctx.prescription.get('aperture_diameter')
                    if ap is None:
                        ap = 0.4 * N_pix * dx_pix
                    _cdtype = get_default_complex_dtype()
                    _cache = _get_wrapper_merit_cache(
                        N_pix, dx_pix, ap, _cdtype)
                    mask = _cache['mask']
                    # Three branches:
                    #   mask is None             -> "no aperture
                    #     specified" (ap was None above, but the
                    #     0.4*N*dx fallback rules that out here).
                    #     Treat as full grid for completeness.
                    #   mask is _ZERO_APERTURE_MASK -> deliberate
                    #     aperture_diameter <= 0; block all light
                    #     (v4.14.1 P1-NEW-1 fix; v4.14.0 erroneously
                    #     mapped this to "full grid plane wave").
                    #   mask is an ndarray       -> circular boolean
                    #     mask for the requested aperture diameter.
                    if mask is _ZERO_APERTURE_MASK:
                        E_in_wl = np.zeros((N_pix, N_pix), dtype=_cdtype)
                    elif mask is None:
                        E_in_wl = np.ones((N_pix, N_pix), dtype=_cdtype)
                    else:
                        E_in_wl = mask.astype(_cdtype)
                    E_exit_wl = apply_real_lens(
                        E_in_wl,
                        prescription=ctx.prescription,
                        wavelength=wl,
                        dx=dx_pix)
                    sub_E_exit = E_exit_wl
                    half = max(abs(bfl) / 20.0, 1e-3)
                    z_vals = np.linspace(bfl - half, bfl + half, 7)
                    ideal_pk = diffraction_limited_peak(
                        E_exit_wl, wl, bfl, dx_pix)
                    scan = through_focus_scan(
                        E_exit_wl, dx_pix, wl, z_vals,
                        ideal_peak=ideal_pk, verbose=False)
                    z_b, sb = find_best_focus(scan, 'strehl')
                    sub_z = float(z_b)
                    sub_strehl = float(sb)
                    # 4.11.1: nanargmax so a single NaN per-z slice does
                    # not steal the argmax position.
                    if np.any(np.isfinite(scan.strehl)):
                        i_b = int(np.nanargmax(scan.strehl))
                        sub_rms = float(scan.rms_radius[i_b])
                    _, _, sub_opd = wave_opd_2d(
                        E_exit_wl, dx_pix, wl, aperture=ap,
                        focal_length=bfl, f_ref=bfl)
                except (TypeError, ValueError, RuntimeError) as exc:
                    # 4.11.1: was a bare ``except Exception: pass`` which
                    # silently swallowed call-signature mistakes (the
                    # 4.10 positional-call regression hid here for the
                    # entire v4.10 series).  Warn so the failure is
                    # visible without aborting the optimizer.
                    warnings.warn(
                        f"MultiWavelengthMerit: per-wavelength wave-leg "
                        f"propagation failed at wl={wl:.3e} m "
                        f"({type(exc).__name__}: {exc}); falling back "
                        f"to the parent context's wave-leg values.",
                        RuntimeWarning, stacklevel=2)
            per_wl_strehl.append(sub_strehl)
            per_wl_rms.append(sub_rms)
            # v4.13.2 (C-P1-2): thread ctx.x so JaxMeritTerm sub-
            # merits with build_args reach the analytic-gradient
            # path instead of legacy fn(ctx) -> FD.
            sub_ctx = EvaluationContext(
                prescription=ctx.prescription, wavelength=wl,
                N=ctx.N, dx=ctx.dx, efl=float(efl), bfl=float(bfl),
                seidel=ctx.seidel, E_exit=sub_E_exit,
                opd_map=sub_opd, strehl_best=sub_strehl,
                rms_radius_best=sub_rms, z_best=sub_z,
                x=getattr(ctx, 'x', None))
            total = total + self.sub_merit.evaluate(sub_ctx)
        ctx.efls_per_wavelength = np.array(efls)
        ctx.strehls_per_wavelength = np.array(per_wl_strehl)
        ctx.rms_per_wavelength = np.array(per_wl_rms)
        return self.weight * total


# =========================================================================
# Multi-field support (off-axis)
# =========================================================================

class MultiFieldMerit(MeritTerm):
    """Evaluate a sub-merit at multiple field angles.

    At each field angle a tilted plane wave is built, propagated
    through the lens, and the sub-merit is evaluated on the
    resulting wave field.

    .. warning::
        The off-field wave-leg propagation always calls
        :func:`apply_real_lens` directly, irrespective of the
        ``wave_propagator`` setting on the enclosing
        :func:`design_optimize` call.  See audit P2 #14 for context;
        a runtime warning fires from :func:`design_optimize` when the
        propagator mismatch is detected.

    Parameters
    ----------
    field_angles : sequence of float OR sequence of (theta_x, theta_y)
        Field angles in radians (half-angle from optical axis).
        ``0`` = on-axis.  A scalar entry is interpreted as a pure
        Y-axis tilt (preserved for back-compatibility); a
        ``(theta_x, theta_y)`` tuple is interpreted as an off-axis
        plane wave with independent X and Y tilts.  The scalar form
        emits a one-shot :class:`DeprecationWarning`.
    sub_merit : MeritTerm
        Wave-based merit term to evaluate at each field.
    weight : float
    """

    name = 'MultiField'
    # Class-level flag so the deprecation warning fires exactly once
    # per process (not once per instance, not once per evaluate()).
    _scalar_warning_issued = False

    def __init__(self, field_angles: Sequence[Any],
                 sub_merit: MeritTerm, weight: float = 1.0) -> None:
        # v4.13.2 (C-P0-2): accept EITHER scalars (back-compat:
        # Y-axis tilt) OR (theta_x, theta_y) tuples.  Detect the
        # form per-entry so a mixed list still works.
        normalised: List[Tuple[float, float]] = []
        had_scalar = False
        for a in field_angles:
            if isinstance(a, (tuple, list)) and len(a) == 2:
                tx, ty = a
                normalised.append((float(tx), float(ty)))
            else:
                had_scalar = True
                normalised.append((0.0, float(a)))
        if had_scalar and not MultiFieldMerit._scalar_warning_issued:
            warnings.warn(
                "MultiFieldMerit: scalar ``field_angles`` entries are "
                "interpreted as Y-axis tilt only.  Pass "
                "(theta_x, theta_y) tuples to control both axes; the "
                "scalar form will keep working but is deprecated.",
                DeprecationWarning, stacklevel=2)
            MultiFieldMerit._scalar_warning_issued = True
        self.field_angles = normalised
        self.sub_merit = sub_merit
        self.weight = float(weight)
        self.needs_wave = True

    def evaluate(self, ctx: Any) -> float:
        total = 0.0
        # v4.14.0: aperture mask + coordinate grids + the
        # wavelength-independent k0*Y / k0*X factors are invariant
        # across field angles and FD-eval perturbations.  Cache them
        # module-level keyed on (N, dx, aperture, dtype).  Per-leg
        # cost reduces to a single np.exp + np.where over the cached
        # mask + tilt phase; meshgrid_build_count drops from
        # n_fields * 2N_FD to 1 per optimisation run.
        Ny, Nx = ctx.N, ctx.N
        ap_diam = ctx.prescription.get('aperture_diameter')
        if ap_diam is None:
            ap_diam = 0.4 * Nx * ctx.dx
        _cdtype = get_default_complex_dtype()
        _cache = _get_wrapper_merit_cache(
            ctx.N, ctx.dx, float(ap_diam), _cdtype)
        # Wavelength-independent factors: per-field the tilt phase
        # is sin(theta_x) * (k0_X_factor / wavelength) +
        # sin(theta_y) * (k0_Y_factor / wavelength).  Pre-fold the
        # 1/wavelength into the wavelength-dependent multiplier
        # below.  Note ``ctx.wavelength`` IS invariant across
        # MultiFieldMerit's loop (the field sweep is the loop axis),
        # so we form k_X/k_Y just once.
        _wl = float(ctx.wavelength)
        k_X = _cache['X_factor'] / _wl
        k_Y = _cache['Y_factor'] / _wl
        aperture_mask = _cache['mask']
        for theta_x, theta_y in self.field_angles:
            # Build tilted plane wave clipped to the lens aperture so
            # the propagated intensity reflects the lens's actual
            # acceptance.  Pre-4.10 the unclipped grid-filling plane
            # wave fed every grid pixel through apply_real_lens, then
            # Strehl was computed against a "grid-filling" reference
            # which artificially lowered the value and biased the
            # optimizer toward apertures larger than designed.
            # v4.13.2 (C-P0-2): generic off-axis tilt with both X and
            # Y components.  Pre-fix the X term was silently dropped.
            tilt_phase = np.sin(theta_x) * k_X + np.sin(theta_y) * k_Y
            # 4.11.1: honour precision knob (was hard-coded complex128
            # which silently demoted precision='single' configs).
            # v4.14.1 (P1-NEW-1): three branches -- None means "no
            # aperture specified, full grid"; _ZERO_APERTURE_MASK
            # means "aperture explicitly zero, block all light";
            # ndarray means "circular boolean mask."  Pre-v4.14.0 the
            # zero-diameter case was an all-False ndarray (correctly
            # zeroing the field); v4.14.0 collapsed it into the None
            # branch (full-grid plane wave), flipping the semantics.
            if aperture_mask is _ZERO_APERTURE_MASK:
                E_tilted = np.zeros((Ny, Nx), dtype=_cdtype)
            elif aperture_mask is None:
                E_tilted = np.exp(1j * tilt_phase).astype(_cdtype)
            else:
                E_tilted = np.where(aperture_mask, np.exp(1j * tilt_phase),
                                     0.0).astype(_cdtype)
            E_exit = apply_real_lens(
                E_tilted, prescription=ctx.prescription, wavelength=ctx.wavelength, dx=ctx.dx)
            # Build sub-context.  v4.13.2 (C-P1-2): thread ctx.x so
            # JaxMeritTerm(build_args=...) sub-merits route through
            # the analytic-gradient path instead of falling back to
            # legacy fn(ctx) (which would silently degrade analytic
            # gradients to FD).
            sub_ctx = EvaluationContext(
                prescription=ctx.prescription,
                wavelength=ctx.wavelength, N=ctx.N, dx=ctx.dx,
                efl=ctx.efl, bfl=ctx.bfl, seidel=ctx.seidel,
                E_exit=E_exit, x=getattr(ctx, 'x', None))
            # Through-focus for this field
            if np.isfinite(ctx.bfl) and abs(ctx.bfl) < 10:
                half = max(abs(ctx.bfl) / 20.0, 1e-3)
                z_values = np.linspace(ctx.bfl - half, ctx.bfl + half, 21)
                try:
                    ideal = diffraction_limited_peak(
                        E_exit, ctx.wavelength, ctx.bfl, ctx.dx)
                    scan = through_focus_scan(
                        E_exit, ctx.dx, ctx.wavelength, z_values,
                        ideal_peak=ideal, verbose=False)
                    z_best, strehl_best = find_best_focus(scan, 'strehl')
                    sub_ctx.strehl_best = float(strehl_best)
                    # 4.11.1: nanargmax so a single NaN slice doesn't
                    # steal the argmax.
                    if np.any(np.isfinite(scan.strehl)):
                        i_best = int(np.nanargmax(scan.strehl))
                        sub_ctx.rms_radius_best = float(
                            scan.rms_radius[i_best])
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # Field-leg through-focus scan failed; zero
                    # Strehl is a safe sentinel (the optimizer treats
                    # it as a very-bad design).
                    sub_ctx.strehl_best = 0.0
            # OPD map if needed
            ap = ctx.prescription.get('aperture_diameter')
            if ap and hasattr(self.sub_merit, 'needs_wave') and self.sub_merit.needs_wave:
                try:
                    from ..analysis import wave_opd_2d
                    _, _, opd = wave_opd_2d(
                        E_exit, ctx.dx, ctx.wavelength,
                        aperture=ap, focal_length=ctx.bfl, f_ref=ctx.bfl)
                    sub_ctx.opd_map = opd
                except (ValueError, RuntimeError, ZeroDivisionError,
                        np.linalg.LinAlgError, IndexError, AttributeError,
                        TypeError):
                    # OPD-map extraction failed (aperture mismatch /
                    # singular least-squares fit); leave None so
                    # downstream Zernike merits return 0 contribution.
                    sub_ctx.opd_map = None
            total = total + self.sub_merit.evaluate(sub_ctx)
        return self.weight * total / max(len(self.field_angles), 1)


# =========================================================================
# Constraint-style merits
# =========================================================================

class MinThicknessMerit(MeritTerm):
    """Penalise any GLASS thickness below a minimum.

    ``contribution = weight * sum_glass_thicknesses max(0, min_t - t_i)^2``

    4.10: only glass thicknesses count; air gaps are skipped.  Pre-4.10
    iterated every entry in ``prescription['thicknesses']`` which
    included air gaps that legitimately need to be small (e.g. cemented
    interfaces, near-zero gap between two surfaces).

    Parameters
    ----------
    min_thickness : float
        Minimum acceptable GLASS thickness [m].  Use
        ``MinAirGapMerit`` (if added) for air-gap constraints.
    weight : float
    include_air : bool, optional
        Set True to restore the pre-4.10 behaviour and also penalise
        small air gaps.  Default False.
    """

    needs_wave = False
    name = 'MinThickness'

    def __init__(self, min_thickness: float = 1e-3,
                 weight: float = 1.0,
                 include_air: bool = False) -> None:
        self.min_thickness = float(min_thickness)
        self.weight = float(weight)
        self.include_air = bool(include_air)

    def evaluate(self, ctx: Any) -> float:
        thicknesses = ctx.prescription.get('thicknesses', [])
        surfaces = ctx.prescription.get('surfaces', [])
        total = 0.0
        for i, t in enumerate(thicknesses):
            if not self.include_air:
                # Determine if this thickness sits between two glass
                # interfaces: use the glass_after of the i-th surface
                # (which is the material the i-th thickness sits in).
                glass = 'air'
                if i < len(surfaces):
                    surf = surfaces[i]
                    glass = (surf.get('glass_after', 'air')
                              if isinstance(surf, dict)
                              else getattr(surf, 'glass_after', 'air'))
                if isinstance(glass, str) and glass.lower() in (
                        'air', 'vacuum', '', None):
                    continue
            deficit = max(0.0, self.min_thickness - float(t))
            total = total + deficit * deficit
        return self.weight * total


class MaxThicknessMerit(MeritTerm):
    """Penalise any glass thickness above a maximum."""

    needs_wave = False
    name = 'MaxThickness'

    def __init__(self, max_thickness: float = 20e-3,
                 weight: float = 1.0) -> None:
        self.max_thickness = float(max_thickness)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        thicknesses = ctx.prescription.get('thicknesses', [])
        total = 0.0
        for t in thicknesses:
            excess = max(0.0, float(t) - self.max_thickness)
            total = total + excess * excess
        return self.weight * total


class MinBackFocalLengthMerit(MeritTerm):
    """Penalise BFL below a minimum (e.g. to keep clearance for
    a sensor package)."""

    needs_wave = False
    name = 'MinBFL'

    def __init__(self, min_bfl: float = 5e-3,
                 weight: float = 1.0) -> None:
        self.min_bfl = float(min_bfl)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        # 4.10: ctx.bfl is set to the sentinel 1e9 when the ray leg
        # fails.  Pre-4.10 deficit = max(0, min_bfl - 1e9) = 0, so
        # invalid prescriptions silently scored as "satisfies clearance".
        # Penalise them with a large finite value instead.
        if not ctx_is_valid(ctx, 'bfl'):
            return self.weight * (self.min_bfl ** 2)
        deficit = max(0.0, self.min_bfl - ctx.bfl)
        return self.weight * deficit * deficit


class MaxFNumberMerit(MeritTerm):
    """Penalise an f/# above a maximum (force faster lens)."""

    needs_wave = False
    name = 'MaxFNumber'

    def __init__(self, max_f_number: float = 8.0,
                 weight: float = 1.0) -> None:
        self.max_f_number = float(max_f_number)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        # 4.10: guard against the ctx.efl = 1e9 sentinel.  Pre-4.10 a
        # failed ray leg produced fnum = 1e9 / aperture ≈ 1e12, squared
        # to ≈ 1e24 -- swamping every other merit term in the sum so the
        # optimizer "saw" only this penalty when the ray leg failed.
        if not ctx_is_valid(ctx, 'efl'):
            return self.weight
        ap = ctx.prescription.get('aperture_diameter', 1e-3)
        fnum = abs(ctx.efl) / ap if ap > 0 else 1e9
        excess = max(0.0, fnum - self.max_f_number)
        return self.weight * excess * excess


# =========================================================================
# Tolerance-aware merit
# =========================================================================

class ToleranceAwareMerit(MeritTerm):
    """Optimise the MEAN of a sub-merit across a set of random
    perturbations.

    Instead of optimising the *nominal* Strehl / wavefront, this
    optimises the *average* over a Monte-Carlo perturbation set.
    Produces designs that are robust to manufacturing tolerances
    rather than fragile at the nominal but excellent on paper.

    .. warning::
        The perturbed wave-leg propagation always calls
        :func:`apply_real_lens` directly, irrespective of the
        ``wave_propagator`` setting on the enclosing
        :func:`design_optimize` call.  See audit P2 #14; a runtime
        warning fires from :func:`design_optimize` when the
        propagator mismatch is detected.

    Parameters
    ----------
    sub_merit : MeritTerm
        The merit evaluated at each perturbation (typically
        ``StrehlMerit`` or ``RMSWavefrontMerit``).
    perturbation_spec : list of dict
        Same format as for :func:`monte_carlo_tolerancing`:
        ``[{'surface_index': i, 'decenter_std': ..., 'tilt_std': ...,
            'form_error_rms': ...}]``
    n_trials : int
        Number of random perturbation draws per evaluation.
    seed : int
        Base seed for reproducibility.
    weight : float
    """

    name = 'ToleranceAware'

    def __init__(self, sub_merit: MeritTerm,
                 perturbation_spec: Sequence[Dict[str, Any]],
                 n_trials: int = 5, seed: int = 42,
                 weight: float = 1.0) -> None:
        self.sub_merit = sub_merit
        self.perturbation_spec = list(perturbation_spec)
        self.n_trials = int(n_trials)
        self.seed = int(seed)
        self.weight = float(weight)
        self.needs_wave = sub_merit.needs_wave

    def evaluate(self, ctx: Any) -> float:
        from ..analysis.through_focus import apply_perturbations, Perturbation

        total = 0.0
        for t in range(self.n_trials):
            rng = np.random.default_rng(self.seed + t)
            perts = []
            for spec_idx, spec in enumerate(self.perturbation_spec):
                d_std = spec.get('decenter_std', 0.0)
                t_std = spec.get('tilt_std', 0.0)
                f_rms = spec.get('form_error_rms', 0.0)
                # Deterministic form-error seed: tying it directly to
                # the trial index + surface index means two runs with
                # the same ``self.seed`` produce identical form-error
                # realisations regardless of the global RNG state.
                # Mask to 31 bits to match the Perturbation API.
                fe_seed = ((self.seed + t) * 1_000_003
                           + spec['surface_index']
                           + spec_idx * 17) & 0x7FFFFFFF
                perts.append(Perturbation(
                    surface_index=spec['surface_index'],
                    decenter=(rng.normal(0, d_std) if d_std > 0 else 0.0,
                              rng.normal(0, d_std) if d_std > 0 else 0.0),
                    tilt=(rng.normal(0, t_std) if t_std > 0 else 0.0,
                          rng.normal(0, t_std) if t_std > 0 else 0.0),
                    form_error_rms=f_rms,
                    random_seed=fe_seed,
                    name=f'tol_trial_{t}_s{spec["surface_index"]}'))
            pres_pert = apply_perturbations(
                ctx.prescription, perts, N=ctx.N, dx=ctx.dx)

            # Per-trial ABCD: the perturbed prescription generally has a
            # different EFL/BFL from the nominal, and scanning around
            # the nominal BFL misses the actual best focus (giving an
            # artificially low Strehl that drags the optimizer away).
            try:
                surfs_p = surfaces_from_prescription(pres_pert)
                _, efl_p, bfl_p, _ = system_abcd(surfs_p, ctx.wavelength)
                efl_p = float(efl_p) if np.isfinite(efl_p) else ctx.efl
                bfl_p = float(bfl_p) if np.isfinite(bfl_p) else ctx.bfl
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, TypeError):
                # Perturbed ABCD failed -- fall back to nominal
                # focus, which will under-estimate the Strehl drop
                # but is a stable sentinel.
                efl_p, bfl_p = ctx.efl, ctx.bfl

            # Re-run wave propagation for this perturbation
            # 4.10.2: honour the runtime DEFAULT_COMPLEX_DTYPE so
            # precision='single' actually halves the memory / FFT cost
            # of merit-leg propagation.  Pre-4.10.2 the hard-coded
            # complex128 silently negated the precision='single' knob.
            # v4.14.0: route through the shared wrapper-merit cache
            # so the (Ny, Nx, dx) grid-build invariants are
            # established once per design_optimize run.  The cache
            # also memoises the np.ones source array against
            # mutation by apply_real_lens (which never writes its
            # input), so per-trial we just .copy() the cached
            # template.
            _cdtype = get_default_complex_dtype()
            _ap = ctx.prescription.get('aperture_diameter')
            # v4.14.2 (P1-NEW-1): the perturbed prescription preserves
            # ``aperture_diameter`` from the nominal, so a nominal-zero
            # aperture flows through to the per-trial wave-leg unchanged.
            # ``apply_perturbations`` itself does NOT call
            # ``validate_prescription``, so the validation-time rejection
            # of ``aperture_diameter <= 0`` cannot be relied upon to gate
            # this code path.  Honour the ``_ZERO_APERTURE_MASK`` sentinel
            # placed in ``_cache['mask']`` by ``_get_wrapper_merit_cache``
            # so a deliberate-zero aperture produces a zero E_in rather
            # than the cached full-ones template (which would otherwise
            # propagate a grid-filling plane wave through ``apply_real_lens``
            # and silently mis-score the perturbed trial).  Matches the
            # canonical branch at ``MultiWavelengthMerit.evaluate`` and
            # ``MultiFieldMerit.evaluate``.
            _cache = _get_wrapper_merit_cache(
                ctx.N, ctx.dx, _ap, _cdtype)
            if _cache['mask'] is _ZERO_APERTURE_MASK:
                E_in = np.zeros((ctx.N, ctx.N), dtype=_cdtype)
            else:
                E_in = _cache['E_ones'].copy()
            E_exit = apply_real_lens(
                E_in, prescription=pres_pert, wavelength=ctx.wavelength, dx=ctx.dx)
            # v4.13.2 (C-P1-2): thread ctx.x so JaxMeritTerm sub-
            # merits with build_args reach the analytic-gradient
            # path instead of legacy fn(ctx) -> FD.
            sub_ctx = EvaluationContext(
                prescription=pres_pert, wavelength=ctx.wavelength,
                N=ctx.N, dx=ctx.dx, efl=efl_p, bfl=bfl_p,
                x=getattr(ctx, 'x', None))
            # Through-focus scan around the PERTURBED BFL, not the
            # nominal BFL.
            if np.isfinite(bfl_p) and abs(bfl_p) < 10:
                half = max(abs(bfl_p) / 20.0, 1e-3)
                z_values = np.linspace(bfl_p - half, bfl_p + half, 11)
                try:
                    ideal = diffraction_limited_peak(
                        E_exit, ctx.wavelength, bfl_p, ctx.dx)
                    scan = through_focus_scan(
                        E_exit, ctx.dx, ctx.wavelength, z_values,
                        ideal_peak=ideal, verbose=False)
                    z_best, strehl_best = find_best_focus(scan, 'strehl')
                    sub_ctx.strehl_best = float(strehl_best)
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # Tolerancing trial through-focus failed; treat
                    # this perturbation as worst-case (Strehl=0).
                    sub_ctx.strehl_best = 0.0
            total = total + self.sub_merit.evaluate(sub_ctx)
        return self.weight * total / max(self.n_trials, 1)


# =========================================================================
# Evaluation context
# =========================================================================

@dataclass
class EvaluationContext:
    prescription: Dict[str, Any]
    wavelength: float
    N: int
    dx: float
    efl: float = 0.0
    bfl: float = 0.0
    seidel: np.ndarray = field(default_factory=lambda: np.zeros(5))
    E_exit: Optional[np.ndarray] = None  # wave leg output
    strehl_best: float = 0.0
    rms_radius_best: float = np.inf
    z_best: float = 0.0
    opd_map: Optional[np.ndarray] = None
    efls_per_wavelength: Optional[np.ndarray] = None
    # Populated when a MultiPrescriptionParameterization is used.
    # ``prescription`` stays == ``prescriptions[0]`` for backward
    # compatibility so single-prescription merit terms keep working.
    prescriptions: Optional[List[Dict[str, Any]]] = None
    # Current parameter vector (populated by design_optimize).  Lets
    # JaxMeritTerm route through its build_args(x) for analytic
    # gradient propagation.  Standalone evaluations may leave this
    # None; merits should fall back to ctx-based code paths in that
    # case.
    x: Optional[np.ndarray] = None
    # Per-eval cache for canonical-polynomial fits.  Used by
    # LGAberrationMerit so a CompositeMerit with multiple LG terms
    # (centre / edge / corner emitter classes) builds the fit ONCE
    # per merit eval instead of once per term.  Lives only for the
    # lifetime of a single merit_fn(x) call -- the next eval gets a
    # fresh context with an empty cache.  Audit perf #2 (3.5.5).
    _canonical_fit_cache: Dict[Any, Any] = field(default_factory=dict)

    def rms_wavefront_waves(self, n_modes: int = 21,
                             exclude_low_order: int = 3) -> float:
        """RMS wavefront error in waves, excluding the first
        ``exclude_low_order`` Zernike modes (default: piston, tilt X,
        tilt Y).  Computed from the current OPD map.
        """
        if self.opd_map is None:
            return np.inf
        ap = self.prescription.get('aperture_diameter')
        if ap is None:
            return np.inf
        coeffs, _ = zernike_decompose(
            self.opd_map, self.dx, ap, n_modes=n_modes)
        # rms of higher-order modes, in meters
        higher = coeffs[exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        return rms_m / self.wavelength  # waves


# =========================================================================
# Main entry point
# =========================================================================

@dataclass
class DesignResult:
    x: np.ndarray
    prescription: Dict[str, Any]
    merit: float
    converged: bool
    iterations: int
    time_sec: float
    context_final: EvaluationContext
    scipy_result: Any = None
    # Populated when a MultiPrescriptionParameterization was used.
    # Otherwise None (use ``prescription`` for the single-lens case).
    prescriptions: Optional[List[Dict[str, Any]]] = None


def _fd_grad_pure(
    f: Callable,
    x: np.ndarray,
    eps: float = 1e-7,
    scale_floor: Optional[np.ndarray] = None,
    f0: Optional[float] = None,
    scheme: str = 'central',
    validate_f0: bool = False,
) -> np.ndarray:
    """Finite-difference gradient of a scalar callable ``f(x)``.

    Pure-function helper extracted from :func:`design_optimize` so the
    FD-gradient core is testable without spinning up an optimization
    context.  Returns ``df/dx`` as an ndarray with the same shape as
    ``x``.

    Caller contract for ``f0``
    --------------------------
    When ``scheme='forward'`` and the caller supplies ``f0``, **the
    caller is responsible for the invariant ``f0 == f(x)`` at the
    same ``x`` passed in**.  A stale ``f0`` silently produces a
    wrong-direction gradient (the forward-difference quotient
    ``(f(x + h*e_i) - f0) / h`` reduces to junk plus the gradient if
    ``f0`` is off the true centre value).  This is by design -- the
    perf saving from skipping the centre evaluation is the whole
    reason ``f0`` exists.

    Set ``validate_f0=True`` to opt into a defensive ``f(x)`` re-
    evaluation that asserts the invariant; this halves the perf
    saving but catches stale-cache bugs immediately.  Off by default
    so the default path stays as cheap as the audit (P2 #11)
    expected.

    Parameters
    ----------
    f : callable
        Scalar function of the parameter vector.
    x : ndarray, shape (N,)
        Evaluation point.
    eps : float, default 1e-7
        Relative step parameter.  The actual step in coordinate ``i``
        is ``eps * max(|x[i]|, scale_floor[i])``.
    scale_floor : ndarray, shape (N,), optional
        Per-variable absolute step floor.  Defaults to 1 micron per
        variable (matches the legacy ``_fd_grad_for`` default for radii
        / thicknesses).
    f0 : float, optional
        Pre-computed ``f(x)``.  Only consulted when ``scheme='forward'``;
        ignored otherwise.  When supplied for the forward path, the
        central-point evaluation is skipped, saving one call to ``f``
        per gradient.  **The caller is responsible for ensuring
        ``f0 == f(x)``** (see the caller contract above); a stale
        value silently produces wrong gradients.
    scheme : {'central', 'forward'}, default 'central'
        Finite-difference scheme.  ``'central'`` evaluates ``f`` at
        ``x +/- h*e_i`` for each variable (2N evals, O(h^2) truncation
        error); this is the historical default and preserves bit-
        identical gradient values with pre-v4.13.0 behaviour.
        ``'forward'`` evaluates ``f`` at ``x`` and ``x + h*e_i`` (N+1
        evals, or N when ``f0`` is supplied) at the cost of O(h)
        truncation error.  Opt-in for perf-sensitive callers where
        the larger truncation is acceptable.
    validate_f0 : bool, default False
        Audit P2 #16 (v4.14): when ``True`` AND ``scheme='forward'``
        AND ``f0`` is supplied, re-evaluate ``f(x)`` once and raise
        ``ValueError`` if ``f0`` does not match within a tight
        tolerance.  Off by default because the validation costs one
        ``f`` call, which exactly cancels the saving from skipping
        the centre evaluation.  Useful for debugging stale-cache
        bugs in caller code.

    Returns
    -------
    g : ndarray, shape (N,)
        Finite-difference gradient.
    """
    if scheme not in ('central', 'forward'):
        raise ValueError(
            f"scheme must be 'central' or 'forward', got {scheme!r}")
    x = np.asarray(x, dtype=np.float64)
    N = x.size
    if scale_floor is None:
        scale_floor = np.full(N, 1e-6, dtype=np.float64)
    else:
        scale_floor = np.broadcast_to(
            np.asarray(scale_floor, dtype=np.float64), (N,))
    g = np.zeros(N, dtype=np.float64)
    if scheme == 'central':
        for i in range(N):
            step = eps * max(abs(x[i]), float(scale_floor[i]))
            xp_step = x.copy()
            xm_step = x.copy()
            xp_step[i] = x[i] + step
            xm_step[i] = x[i] - step
            fp = float(f(xp_step))
            fm = float(f(xm_step))
            g[i] = (fp - fm) / (2.0 * step)
    else:  # forward
        if f0 is None:
            f0 = float(f(x))
        elif validate_f0:
            # v4.14 (audit P2 #16): opt-in stale-cache check.  Tight
            # but not exact tol to allow for benign re-evaluation
            # noise in non-deterministic merit functions.
            f0_check = float(f(x))
            tol = 1e-9 * max(abs(f0), abs(f0_check), 1.0)
            if not np.isfinite(f0_check) or abs(f0 - f0_check) > tol:
                raise ValueError(
                    f"_fd_grad_pure: validate_f0=True caught a stale "
                    f"f0 cache: f0={f0!r} but f(x) at the given x "
                    f"is {f0_check!r}.  Forward-FD with a stale f0 "
                    f"produces wrong gradients; the caller is "
                    f"responsible for the f0 == f(x) invariant.")
        for i in range(N):
            step = eps * max(abs(x[i]), float(scale_floor[i]))
            xp_step = x.copy()
            xp_step[i] = x[i] + step
            fp = float(f(xp_step))
            g[i] = (fp - f0) / step
    return g


def design_optimize(parameterization: Any,
                    merit_terms: Sequence[MeritTerm],
                    wavelength: float,
                    N: int = 512,
                    dx: float = 20e-6,
                    E_in: Optional[np.ndarray] = None,
                    method: str = 'L-BFGS-B',
                    max_iter: int = 100,
                    wave_traced: bool = False,
                    wave_propagator: str = 'real_lens',
                    wave_propagator_kwargs: Optional[Dict[str, Any]] = None,
                    ray_subsample: int = 4,
                    z_scan_range: Optional[Tuple[float, float]] = None,
                    z_scan_n: int = 31,
                    jac: Any = 'auto',
                    precision: str = 'double',
                    plane_logger: Optional[Callable] = None,
                    verbose: bool = True,
                    progress: Optional[Callable] = None) -> DesignResult:
    """Optimize a lens prescription against a set of merit terms.

    Parameters
    ----------
    parameterization : DesignParameterization
        Template + free variables + bounds.
    merit_terms : sequence of MeritTerm
        Each contributes an (already-weighted) scalar term that is
        summed into the total merit.  ``SphericalSeidelMerit``,
        ``FocalLengthMerit`` etc. are pure-geometric and fast;
        ``StrehlMerit`` / ``RMSWavefrontMerit`` / ``SpotSizeMerit``
        require the wave leg (slower).
    wavelength : float
        Optimization wavelength [m].  For chromatic merits pass a
        list of wavelengths as ``ChromaticFocalShiftMerit``
        dependency (not yet wired; geometric-only chromatic shift).
    N, dx : int, float
        Wave-grid size and spacing.  Only used when any merit term
        has ``needs_wave = True``.
    E_in : ndarray, optional
        Input field for the wave leg.  Defaults to a unit plane wave.
    method : str
        scipy.optimize method.  ``'L-BFGS-B'`` (bounded quasi-Newton,
        default), ``'trust-constr'``, ``'SLSQP'``, or ``'Powell'``.
        For Gauss-Newton / LM treatment, pass ``'lm'`` and the
        optimizer will switch to ``least_squares``.
    max_iter : int
        Maximum outer iterations.
    wave_traced : bool, default False
        If True, use :func:`apply_real_lens_traced` for the wave
        leg (sub-nm OPD accuracy but slower).  Otherwise use
        :func:`apply_real_lens` (fast analytic model).  Ignored
        when ``wave_propagator`` is non-default.
    wave_propagator : str, default 'real_lens'
        Selects the wave-leg propagator.  Options:

          * ``'real_lens'`` -- ``apply_real_lens`` /
            ``apply_real_lens_traced`` (the default; fast, paraxial).
          * ``'gbd'`` -- ``propagate_gbd_through_prescription``
            (beamlet decomposition; better for high-NA / thick
            optics).
          * ``'hf'`` -- ``propagate_huygens_fresnel_through_prescription``
            (Van-Vleck-corrected Huygens-Fresnel; broadest validity).
          * ``'hfpi'`` -- ``propagate_hfpi_through_prescription``
            (Monte Carlo path integration; honours hard apertures
            and DOEs natively).
          * ``'asymptotic'`` -- ``propagate_modal_asymptotic`` after
            ``fit_canonical_polynomials`` (per-pixel saddle-point
            evaluator; ~10^3-10^4x faster than direct quadrature for
            paraxial systems).
    wave_propagator_kwargs : dict, optional
        Extra keyword arguments forwarded to the chosen wave
        propagator (e.g. ``n_paths`` for HFPI, ``M_super`` for GBD).
        For ``wave_propagator='asymptotic'`` the special keys
        ``fit`` and ``fit_kwargs`` are honoured (pre-built
        :class:`CanonicalPolyFit` and ``fit_canonical_polynomials``
        kwargs respectively).
    ray_subsample : int, default 4
        Passed to ``apply_real_lens_traced`` when used.
    z_scan_range : tuple, optional
        (``z_min``, ``z_max``) relative to the nominal back focal
        length, for the through-focus scan.  Default: ±f/20.
    z_scan_n : int, default 31
        Points in the through-focus scan.
    jac : 'auto' | 'fd' | callable, default 'auto'
        Jacobian strategy.

          * ``'auto'`` -- if any :class:`JaxMeritTerm` was constructed
            with ``build_args``, assemble an analytic Jacobian for
            those terms (via :func:`jax.grad`) and combine with FD for
            the remaining merit terms.  Falls back to ``'fd'`` when
            no JAX merits with ``build_args`` are present.
          * ``'fd'`` -- pure finite differences (SciPy default).
          * callable -- a user-supplied ``f(x) -> ndarray`` passed
            through to SciPy as ``jac=...``.
    plane_logger : callable, optional
        ``plane_logger(iteration, ctx)`` called after every merit
        evaluation.  Useful for streaming intermediate
        prescriptions / E_exit / OPD maps to a unified store; see
        :func:`lumenairy.io.storage.append_plane`.
    verbose : bool
    progress : callable, optional
        ``ProgressCallback`` (see :mod:`lumenairy.progress`).
        Fired with ``stage='design_optimize'`` at start (frac=0.0),
        on every merit-function evaluation (``'eval N: ...'``), on
        every scipy iteration where available (``'iter N: ...'``),
        and at completion (frac=1.0).  Monotonic: the bar never
        moves backwards even when eval and iter series leapfrog.
        ``'lm'`` / ``least_squares`` emits eval-only (no scipy iter
        callback exists for it).

    Returns
    -------
    DesignResult
    """
    import scipy.optimize as so
    from ..progress import call_progress
    from ..propagators.propagation import (
        get_default_complex_dtype, set_default_complex_dtype)

    # 3.5.6: ``precision`` knob.  'double' (default) preserves the
    # historical np.complex128 path; 'single' switches to np.complex64
    # for the duration of this design_optimize call (~2x FFT
    # throughput, ~2x memory headroom, ~80 dB cumulative dynamic-range
    # noise floor).  Restored to its prior value at the end via
    # try/finally.
    if precision not in ('double', 'single'):
        raise ValueError(
            f"design_optimize: precision must be 'double' or 'single', "
            f"got {precision!r}.")
    _orig_complex_dtype = get_default_complex_dtype()
    if precision == 'single':
        set_default_complex_dtype(np.complex64)
    # 4.10: register dtype restoration through a sentinel object so any
    # exception (scipy raise, KeyboardInterrupt, etc.) before the
    # success-path restore at the end still puts the global complex
    # dtype back.  Without this, an interrupted `precision='single'`
    # design_optimize() leaked complex64 to every subsequent call in
    # the process.
    #
    # v4.14 (audit P2 #10): the dominant restore path is now an
    # explicit ``try/finally`` around the optimization body so the
    # cleanup is deterministic under ``KeyboardInterrupt`` and any
    # exception (CPython refcount semantics in ``__del__`` are
    # implementation-defined and can be deferred under PyPy / when a
    # reference cycle survives garbage collection).  ``__del__`` is
    # retained as a defensive safety net only -- the ``_restored``
    # flag stops it firing twice on the normal path.
    class _RestoreDtype:
        def __init__(self, dtype):
            self.dtype = dtype
            self._restored = False
        def restore(self):
            """Explicitly restore the saved complex dtype.

            Idempotent: subsequent calls are no-ops.  Call from a
            ``finally:`` block to guarantee restoration on every exit
            path (normal return, exception, ``KeyboardInterrupt``).
            """
            if self._restored:
                return
            self._restored = True
            set_default_complex_dtype(self.dtype)
        def __del__(self):
            # Safety net only: the explicit ``restore()`` call in the
            # caller's ``finally:`` block is the primary path.
            # ``__del__`` runs at arbitrary points during interpreter
            # shutdown; broad-except is the standard pattern here
            # because the module-level globals may already be gone.
            if self._restored:
                return
            try:
                set_default_complex_dtype(self.dtype)
            except Exception:
                pass
    _dtype_restore_guard = _RestoreDtype(_orig_complex_dtype)

    # v4.14 (audit P2 #14): warn if the user selected a non-default
    # wave_propagator (e.g. 'gbd') AND any of the three Merit classes
    # that hard-code apply_real_lens for off-nominal legs is in use.
    # Threading the propagator through the sub-merit is a v4.14+
    # feature; this warning surfaces the silent inconsistency.
    if wave_propagator != 'real_lens':
        _SENSITIVE = (MultiWavelengthMerit, MultiFieldMerit, ToleranceAwareMerit)
        offenders = [type(m).__name__ for m in merit_terms
                     if isinstance(m, _SENSITIVE)]
        if offenders:
            import warnings as _warn
            _warn.warn(
                f"design_optimize: wave_propagator={wave_propagator!r} "
                f"selected but the following merit term(s) hard-code "
                f"apply_real_lens for off-nominal legs and will NOT use "
                f"that propagator: {offenders}.  The off-nominal "
                f"(wavelength / field / perturbation) wave-leg "
                f"penalties therefore exercise a different physical "
                f"model than the on-axis wave leg.  See audit P2 #14 "
                f"and the merit classes' docstrings; propagator "
                f"threading is a v4.14+ feature.",
                UserWarning, stacklevel=2)

    need_wave = any(m.needs_wave for m in merit_terms)
    n_params = parameterization.n_params
    x0 = parameterization.initial_values()
    bounds = parameterization.bounds

    call_count = [0]
    iter_count = [0]
    last_value = [float('inf')]
    last_efl = [0.0]
    last_frac = [0.0]  # monotonic guard: progress bar never moves backwards
    call_progress(progress, 'design_optimize', 0.0,
                  f'method={method}, {len(merit_terms)} merit term(s)')

    multi_mode = isinstance(parameterization, MultiPrescriptionParameterization)

    def _emit_progress(frac: float, msg: str) -> None:
        # Clamp to [last_frac, 0.99] so the bar is monotonic.  The
        # eval-based and iter-based progress series can leapfrog each
        # other; we always take the larger value.
        frac = max(0.0, min(float(frac), 0.99))
        if frac < last_frac[0]:
            frac = last_frac[0]
        last_frac[0] = frac
        call_progress(progress, 'design_optimize', frac, msg)

    def _emit_iter_progress():
        # Fired from scipy's per-iteration callback (accurate iteration
        # counter, unlike merit_fn which fires on every FD gradient eval).
        iter_count[0] += 1
        frac = iter_count[0] / max(max_iter, 1)
        _emit_progress(
            frac,
            f'iter {iter_count[0]}: merit={last_value[0]:.4g}  '
            f'efl={last_efl[0]*1e3:.3f}mm')

    def evaluate(x):
        built = parameterization.build(x)
        if multi_mode:
            prescriptions = list(built)
            # Use the first prescription as the "primary" for backward-
            # compatible single-prescription merits (ABCD, Seidel, etc.).
            pres = prescriptions[0]
        else:
            pres = built
            prescriptions = [pres]
        ctx = EvaluationContext(
            prescription=pres, wavelength=wavelength, N=N, dx=dx,
            prescriptions=prescriptions,
            x=np.asarray(x, dtype=np.float64).copy())
        # Ray-leg (always)
        surfs = surfaces_from_prescription(pres)
        try:
            _, efl, bfl, _ = system_abcd(surfs, wavelength)
        except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                np.linalg.LinAlgError, IndexError, TypeError):
            # Degenerate / mirror-only / short prescription -- the
            # downstream sentinel filter (np.isfinite check) caps
            # these at 1e9.
            efl = bfl = float('inf')
        try:
            seidel_raw = seidel_coefficients(surfs, wavelength)
            # seidel_coefficients returns (per-surface-dict, totals-dict)
            if (isinstance(seidel_raw, tuple) and len(seidel_raw) == 2
                    and isinstance(seidel_raw[0], dict)):
                per_surf = seidel_raw[0]
                # Sum each aberration coefficient over surfaces
                seidel = np.array([
                    np.sum(per_surf.get(f'S{k}', np.zeros(1)))
                    for k in range(1, 6)], dtype=np.float64)
            else:
                seidel = np.asarray(seidel_raw, dtype=np.float64)
        except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                np.linalg.LinAlgError, IndexError, AttributeError,
                TypeError):
            # Seidel sum unavailable -- zero-fill so the optimizer
            # sees no aberration contribution from this iteration.
            seidel = np.zeros(5)
        ctx.efl = float(efl) if np.isfinite(efl) else 1e9
        ctx.bfl = float(bfl) if np.isfinite(bfl) else 1e9
        ctx.seidel = np.asarray(seidel, dtype=np.float64).ravel()
        # Wave leg (only if any merit term needs it)
        if need_wave:
            if E_in is None:
                # 4.11.2: honour precision='single' via
                # get_default_complex_dtype() so the wave-leg input
                # field matches the rest of the precision pipeline.
                E0 = np.ones((N, N), dtype=get_default_complex_dtype())
            else:
                E0 = E_in
            wp_kwargs = dict(wave_propagator_kwargs or {})
            try:
                _wave_fn = WAVE_PROPAGATOR_REGISTRY[wave_propagator]
            except KeyError:
                raise ValueError(
                    f"design_optimize: unknown wave_propagator "
                    f"{wave_propagator!r}; expected one of "
                    f"{sorted(WAVE_PROPAGATOR_REGISTRY)} "
                    "(register custom propagators with "
                    "register_wave_propagator(name, fn)).")
            _opts = {
                'wave_traced': wave_traced,
                'ray_subsample': ray_subsample,
            }
            E_exit = _wave_fn(
                E0, pres,
                wavelength=wavelength, dx=dx, N=N,
                wp_kwargs=wp_kwargs, opts=_opts,
            )
            ctx.E_exit = E_exit
            # Through-focus scan
            if not np.isfinite(ctx.bfl) or abs(ctx.bfl) > 10:
                # 4.10: every other return path in evaluate() returns
                # (value, ctx) and every caller unpacks it that way.
                # Pre-4.10 this branch returned a bare scalar, so any
                # prescription that briefly drove BFL out of range
                # mid-optimisation raised "cannot unpack non-iterable
                # float object" and killed the run instead of being
                # penalised.
                return _sum_merits(ctx, merit_terms), ctx
            if z_scan_range is None:
                half = max(abs(ctx.bfl) / 20.0, 1e-3)
                z0, z1 = -half, +half
            else:
                z0, z1 = z_scan_range
            z_values = np.linspace(ctx.bfl + z0, ctx.bfl + z1, z_scan_n)
            ideal = diffraction_limited_peak(
                E_exit, wavelength, ctx.bfl, dx)
            scan = through_focus_scan(
                E_exit, dx, wavelength, z_values,
                ideal_peak=ideal, verbose=False)
            z_best, strehl_best = find_best_focus(scan, 'strehl')
            ctx.z_best = float(z_best)
            ctx.strehl_best = float(strehl_best)
            i_best = int(np.argmax(scan.strehl))
            ctx.rms_radius_best = float(scan.rms_radius[i_best])
            # Build OPD map for Zernike fit
            ap = pres.get('aperture_diameter') or (0.4 * N * dx)
            try:
                _, _, opd_map = wave_opd_2d(
                    E_exit, dx, wavelength, aperture=ap,
                    focal_length=ctx.bfl, f_ref=ctx.bfl)
                ctx.opd_map = opd_map
            except (ValueError, RuntimeError, ZeroDivisionError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # Unwrap / least-squares fit failed in wave_opd_2d;
                # leave opd_map None so Zernike merits contribute 0.
                ctx.opd_map = None

        return _sum_merits(ctx, merit_terms), ctx

    # ------------------------------------------------------------------
    # JaxMeritTerm gradient routing.  When jac='auto' and at least one
    # JaxMeritTerm has build_args set, we combine analytic JAX
    # gradients for those terms with finite-difference partials for
    # everything else.  Pure-FD ('fd' or no JAX merits) is the legacy
    # path; user-callable jac is forwarded as-is.
    # ------------------------------------------------------------------
    jax_grad_terms = [
        m for m in merit_terms
        if isinstance(m, JaxMeritTerm) and m.supports_jax_grad()
    ]
    other_terms = [m for m in merit_terms if m not in jax_grad_terms]

    def _fd_grad_for(terms, x, eps=1e-7, f0=None, scheme='central'):
        """Finite-difference gradient of sum(t.evaluate(ctx)) for the
        selected ``terms`` only.

        4.10: step sized by parameter magnitude with a per-variable
        floor pulled from ``parameterization.scale_floor`` (or the
        magnitude itself for an absent floor).  Pre-4.10 the floor
        ``max(|x|, 1.0)`` pinned the step at 1e-7 for radii /
        thicknesses (~mm = ~0.01) AND indices (~1.5) alike, so
        relative perturbations varied by 100x across variable types
        and the Hessian estimate L-BFGS-B builds was biased.

        Parameters
        ----------
        terms : sequence of MeritTerm
        x : array-like
            Current parameter vector.
        eps : float
            Relative step parameter.
        f0 : float, optional
            Pre-computed ``sum(t.evaluate(ctx)) for t in terms`` at the
            current ``x``.  Only consulted when ``scheme='forward'``.
        scheme : {'central', 'forward'}, default 'central'
            Forwarded to :func:`_fd_grad_pure`.  Central differences
            (the default) preserve bit-identical gradient values with
            pre-v4.13.0 behaviour at 2N evaluations per gradient.
            Forward differences are an opt-in perf option (N+1 evals,
            or N with ``f0``) at the cost of O(h) truncation.
        """
        if not terms:
            return np.zeros_like(x, dtype=np.float64)
        x = np.asarray(x, dtype=np.float64)
        # Per-variable scale floor: parameterization may expose it
        # explicitly; otherwise use a more conservative magnitude-
        # proportional floor that doesn't pin sub-mm radii / thicknesses
        # to 1e-7.
        scale_floor = getattr(parameterization, 'scale_floor', None)
        if scale_floor is None:
            scale_floor = np.full_like(x, 1e-6)  # 1 micron default for radii/thicknesses
        else:
            scale_floor = np.broadcast_to(
                np.asarray(scale_floor, dtype=np.float64), x.shape)

        def _f_terms(xv):
            _, ctx_ = evaluate(xv)
            return sum(t.evaluate(ctx_) for t in terms)

        return _fd_grad_pure(_f_terms, x, eps=eps,
                             scale_floor=scale_floor, f0=f0,
                             scheme=scheme)

    def _merit_jac_auto(x):
        # Analytic part: sum gradient_at_x for JAX terms.
        g = np.zeros_like(np.asarray(x, dtype=np.float64))
        for t in jax_grad_terms:
            g = g + t.gradient_at_x(x)
        # Finite-difference part: gradient of the remaining terms.
        #
        # v4.14 (audit P2 #11): switch to forward-FD with a cached
        # ``f0`` for the other-terms sum.  scipy already evaluates
        # ``merit_fn(x)`` before calling ``jac`` at the same ``x`` (the
        # FULL merit, including JAX terms); we can't reuse that
        # directly because ``_fd_grad_for`` operates on the
        # ``other_terms`` subset only.  But evaluating once at ``x``
        # to capture ``f0_other`` costs one evaluate() call and then
        # saves ``N-1`` evaluations per gradient versus central FD
        # (2N -> N+1 evals; net (2N) - (1+N) = N-1 saved per gradient).
        #
        # For large N (>=10 free vars), this halves the FD cost of
        # the gradient and is the dominant runtime saving in design
        # optimisation when ANY non-JAX merit term is present.  The
        # O(h) truncation error of forward FD is well-tolerated by
        # quasi-Newton line searches (L-BFGS-B etc.) at h~1e-7.
        if other_terms:
            _, ctx_f0 = evaluate(x)
            f0_other = float(sum(t.evaluate(ctx_f0) for t in other_terms))
            g = g + _fd_grad_for(other_terms, x, f0=f0_other,
                                 scheme='forward')
        return g

    use_analytic_jac = (jac == 'auto' and len(jax_grad_terms) > 0)
    user_jac = jac if (callable(jac)) else None
    final_jac = (
        user_jac if user_jac is not None
        else (_merit_jac_auto if use_analytic_jac else None)
    )

    def merit_fn(x):
        call_count[0] += 1
        value, ctx = evaluate(x)
        last_value[0] = float(value)
        last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
        if plane_logger is not None:
            try:
                plane_logger(call_count[0], ctx)
            except (TypeError, ValueError, RuntimeError, KeyError,
                    AttributeError, IndexError, OSError) as _exc:
                # logger errors must not derail optimization, but
                # warn once so silent telemetry gaps are visible.
                import warnings as _w
                _w.warn(
                    f"design_optimize: plane_logger callback failed "
                    f"({type(_exc).__name__}: {_exc}); continuing "
                    f"without telemetry for this iteration.",
                    RuntimeWarning, stacklevel=2)
        # Fallback eval-counter progress for methods without a per-
        # iteration callback hook (Powell, DE, dual_annealing, basin-
        # hopping).  For methods with a scipy callback we also emit
        # from there, which is more accurate iteration-wise; the
        # monotonic guard in _emit_progress prevents the bar from
        # going backwards when the two series leapfrog.
        frac = call_count[0] / max(max_iter * 5, 1)
        _emit_progress(
            frac,
            f'eval {call_count[0]}: merit={value:.4g}  '
            f'efl={ctx.efl*1e3:.3f}mm')
        if verbose and call_count[0] % 5 == 1:
            print(f'  iter {call_count[0]}: merit = {value:.6g}  '
                  f'efl = {ctx.efl*1e3:.3f} mm  '
                  f'strehl = {ctx.strehl_best:.4f}')
        return value

    # v4.14 (audit P2 #13): honour the progress cancellation protocol
    # (``progress.should_stop``).  scipy stops the optimiser when the
    # callback returns ``True`` for L-BFGS-B / SLSQP / trust-constr /
    # Nelder-Mead etc.; differential_evolution interprets a True
    # return the same way; basin-hopping accepts True via its
    # ``callback`` arg.  See :mod:`lumenairy.progress`.
    from ..progress import is_cancelled

    def _scipy_cb_minimize(xk, *args, **kwargs):
        # Callback signature varies by method (some pass xk only,
        # trust-constr passes (xk, state), SLSQP passes xk).  Accept
        # anything.
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    def _scipy_cb_de(xk, convergence):
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    def _scipy_cb_basin(xk, f, accept):
        last_value[0] = float(f)
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    # v4.13.2 (P1-NEW-L): dual_annealing's callback was an inline
    # lambda that did NOT poll ``is_cancelled(progress)`` -- a Qt
    # ``Stop`` press during a dual_annealing run was silently
    # ignored.  Promote to a named callback matching the pattern of
    # the other three scipy callbacks; returning True asks
    # dual_annealing to terminate the run.
    def _scipy_cb_da(x, f, context):
        last_value[0] = float(f)
        _emit_iter_progress()
        if is_cancelled(progress):
            return True
        return None

    # v4.14 (audit P2 #10): wrap the dispatch + final evaluation in
    # try/finally so the complex-dtype restore is deterministic even
    # under KeyboardInterrupt / scipy raise.  The ``_dtype_restore_guard``
    # ``__del__`` remains as a defensive safety net.
    t0 = time.time()
    try:
        if method == 'lm':
            # Gauss-Newton / Levenberg-Marquardt via least_squares.  No
            # per-iteration callback is available, so emit progress from
            # inside residuals() using the eval counter.
            def residuals(x):
                call_count[0] += 1
                value, ctx = evaluate(x)
                last_value[0] = float(value)
                last_efl[0] = float(ctx.efl) if np.isfinite(ctx.efl) else 0.0
                frac = call_count[0] / max(max_iter * 5, 1)
                _emit_progress(
                    frac,
                    f'eval {call_count[0]}: merit={value:.4g}  '
                    f'efl={ctx.efl*1e3:.3f}mm')
                # 4.10.2: LM residual = sqrt(m.evaluate(ctx)) is non-
                # differentiable at zero (sqrt'(0) = inf), which produces
                # inf/nan columns in the FD Jacobian near a converged
                # design.  Soft-floor with a tiny epsilon so the residual
                # is differentiable everywhere; the floor is well below
                # typical merit-term magnitudes so it doesn't affect the
                # converged solution.
                _LM_FLOOR = 1e-30
                return np.array(
                    [np.sqrt(max(m.evaluate(ctx), 0.0) + _LM_FLOOR)
                     for m in merit_terms],
                    dtype=np.float64)
            lb = np.array([b[0] if b else -np.inf for b in (bounds or [None] * n_params)])
            ub = np.array([b[1] if b else +np.inf for b in (bounds or [None] * n_params)])
            res = so.least_squares(
                residuals, x0, method='lm' if not (bounds is not None) else 'trf',
                bounds=(lb, ub) if bounds is not None else (-np.inf, np.inf),
                max_nfev=max_iter, verbose=1 if verbose else 0)
            x_opt = res.x
        elif method in ('differential_evolution', 'de', 'global'):
            # Differential evolution: stochastic global optimizer.
            # Requires bounds for all variables.
            if bounds is None:
                raise ValueError(
                    'differential_evolution requires bounds for all variables.')
            res = so.differential_evolution(
                merit_fn, bounds, maxiter=max_iter, seed=42,
                tol=1e-8, disp=verbose, polish=True,
                callback=_scipy_cb_de)
            x_opt = res.x
        elif method == 'basin_hopping':
            # Basin-hopping: global optimizer with local minimisation steps.
            minimizer_kwargs = {
                'method': 'L-BFGS-B',
                'bounds': bounds,
                'options': {'maxiter': 50},
            }
            res = so.basinhopping(
                merit_fn, x0, niter=max_iter,
                minimizer_kwargs=minimizer_kwargs, seed=42,
                disp=verbose, callback=_scipy_cb_basin)
            x_opt = res.x
        elif method == 'dual_annealing':
            if bounds is None:
                raise ValueError(
                    'dual_annealing requires bounds for all variables.')
            res = so.dual_annealing(
                merit_fn, bounds, maxiter=max_iter, seed=42,
                callback=_scipy_cb_da)
            x_opt = res.x
        else:
            res = so.minimize(
                merit_fn, x0, method=method,
                jac=final_jac,
                bounds=bounds if method in ('L-BFGS-B', 'SLSQP', 'trust-constr') else None,
                options={'maxiter': max_iter, 'disp': verbose},
                callback=_scipy_cb_minimize)
            x_opt = res.x

        # Final evaluation for the returned context
        final_value, final_ctx = evaluate(x_opt)
        dt = time.time() - t0
        iter_tag = (f'{iter_count[0]} iters, '
                    if iter_count[0] > 0 else '')
        call_progress(progress, 'design_optimize', 1.0,
                      f'converged: merit={final_value:.4g} '
                      f'({iter_tag}{call_count[0]} evals, {dt:.1f}s)')
    finally:
        # v4.14 (audit P2 #10): explicit deterministic restore.  Runs on
        # normal return AND on every exception path (scipy raise, user
        # KeyboardInterrupt, MemoryError from a huge FFT, etc.).
        _dtype_restore_guard.restore()

    return DesignResult(
        x=x_opt,
        prescription=final_ctx.prescription,
        merit=float(final_value),
        converged=getattr(res, 'success', True),
        iterations=call_count[0],
        time_sec=dt,
        context_final=final_ctx,
        scipy_result=res,
        prescriptions=(final_ctx.prescriptions if multi_mode else None))


def _sum_merits(ctx, merit_terms):
    total = 0.0
    for m in merit_terms:
        total = total + m.evaluate(ctx)
    return total
