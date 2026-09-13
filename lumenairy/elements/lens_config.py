"""
lumenairy.elements.lens_config -- configuration objects for the ``apply_real_lens`` family.
===========================================================================================

Three frozen dataclasses -- :class:`LensGeometry`, :class:`LensNumerics`,
:class:`LensResources` -- plus the :class:`LensConfig` triple that holds them,
so the seven real-lens entry points can be configured by OBJECT instead of by
28-48 keyword arguments each.

Why
---
The 2026-09-11 adversarial audit (TESTS-ARCH, "Proposed config-object design
for the lens family") measured 28 parameters on ``apply_real_lens``, 29 on
``_apply_real_lens_impl``, 48 on ``apply_real_lens_traced``, 29 on
``apply_real_lens_maslov``, 26 on ``apply_real_lens_gbd``, 21 on
``apply_real_lens_fga`` and 11 on ``apply_real_lens_traced_multibranch``, with
~20 names shared between siblings -- and 673 call sites in the corpus
exercising only 177 distinct combinations, 68 % of them passing zero or one
optional keyword.  Four of the five defects the audit's orchestrator seeded
were *interaction* defects (a flag combined with a geometry) that the suite had
no power against, because a 48-keyword signature is not a thing you can
parametrise over.

A config object fixes three specific things, in this order of importance:

1. **The settings become a value.**  ``LensNumerics(newton_fit='spline',
   newton_poly_order=8)`` can be built once, asserted on, stored in a table,
   diffed against another, and handed to every entry point that understands
   those names.  A ``dataclasses.fields()`` walk over it is a coverage
   generator; a 48-parameter call signature is not.
2. **Validation becomes testable in isolation.**  Range and enum checks that
   used to live several hundred lines inside a 5 700-line function run in
   ``__post_init__``, against no field and no prescription.
3. **A silently discarded setting becomes detectable.**  Every field declares
   which entry points accept it and under which keyword; a field that is set
   but does not apply RAISES rather than evaporating -- the exact failure mode
   the audit found in ``test_niche_audit_e_prepared_and_enums.py:738``.

What this module deliberately does NOT do
-----------------------------------------
* **Nothing is deprecated.**  Every existing keyword of every entry point
  still works, unchanged, with the same default.  The config objects are
  purely additive: an entry point called without ``geometry=`` /
  ``numerics=`` / ``resources=`` / ``config=`` runs byte-for-byte the code it
  ran before, and the four new parameters cost four ``is not None`` tests.
* **It does not own every keyword.**  38 of the family's ~110 distinct
  parameter names are here.  The rest are single-model tuning constants
  (``levin_tol``, ``w0_factor``, ``reexpand_threshold``, ...), private
  diagnostics sinks (``_exit_na_out``, ``_imap_out``, ...), or names whose
  DEFAULTS disagree between siblings (``normalize_output`` is ``'power'`` on
  Maslov and ``'none'`` on GBD/FGA) -- a config field cannot carry two
  defaults honestly.  ``docs/lens_configuration.md`` tables every one of them
  with the reason, and
  ``tests/unit/test_audit2609_a16_lens_config_round_trip.py`` fails if a
  parameter is in neither the config tables nor the documented exclusion list.
* **It does not re-validate what the entry point validates.**  Cross-field and
  prescription-dependent rules (``surface_model='displaced'`` vs ``fresnel``,
  ``caustic`` vs ``amplitude_model``, the undersample guard) stay exactly where
  they are; ``__post_init__`` checks only what a field can be judged on alone.

The precedence rule
-------------------
A config field and an explicit keyword for the same setting must AGREE or the
call raises (CONVENTIONS.md section 2 prefix).  Concretely, for each setting:

* the field is a REQUEST iff it differs from its dataclass default;
* the keyword is a REQUEST iff it differs from its signature default
  (the two defaults are identical by construction -- a test asserts it for
  every (entry point, field) pair, so the comparison is well defined);
* two requests that disagree -> ``ValueError``;
* exactly one request -> that value;
* neither -> the entry point's own default.

So a config whose fields are all at their defaults is indistinguishable from
passing no config at all, and there is no combination in which a caller's
explicit keyword is silently overwritten.

Author: Andrew Traverso
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np

__all__ = [
    'LensGeometry',
    'LensNumerics',
    'LensResources',
    'LensConfig',
]


# ---------------------------------------------------------------------------
# Value comparison
# ---------------------------------------------------------------------------

def _same(a: Any, b: Any) -> bool:
    """Best-effort "are these the same setting value?" test.

    Identity first (the common case, and the only test that is always
    meaningful -- ``None``, the sentinels, the singleton strings).  Then
    ``np.array_equal`` when either side is an array, because several of these
    settings legitimately carry one (``carrier`` may be a wavefront,
    ``beam_centre`` a 2-vector) and ``arr == arr`` is not a bool.  Then plain
    ``==``, reduced with ``np.all`` if it comes back non-scalar.

    ``TypeError`` / ``ValueError`` from the comparison count as "different",
    which can only ever cost a spurious conflict message naming both values --
    never a silent overwrite.  Those two are the exact set a comparison of two
    SETTING values produces: ``TypeError`` from an ``__eq__`` that refuses the
    other operand, ``ValueError`` from ``bool()`` of a non-scalar result or
    from ``np.array_equal`` on a ragged input.  Anything else is a broken
    ``__eq__`` and propagates rather than being swallowed.

    Deliberately the same shape as :func:`lumenairy._knobs._same`; the two are
    not shared because ``_knobs`` is a leaf with no NumPy import and this one
    needs the array arm.
    """
    if a is b:
        return True
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return bool(np.array_equal(a, b))
        r = a == b
        if isinstance(r, bool):
            return r
        return bool(np.all(r))
    except (TypeError, ValueError):
        return False


def _dataclass_eq(self: Any, other: Any) -> Any:
    """``__eq__`` for the config dataclasses, field by field through :func:`_same`.

    The generated ``__eq__`` compares the two field TUPLES, and a tuple
    comparison calls ``bool()`` on each element's ``==``.  Three of these
    fields are documented to accept an ndarray (``carrier`` a wavefront,
    ``beam_centre`` / ``origin`` a 2-vector), and ``bool(arr == arr)`` raises
    ``ValueError: the truth value of an array ... is ambiguous`` -- so the
    generated version turns the docstring's own round trip
    (``LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg``) into a crash for
    exactly the values :func:`_same` was written to handle.  Comparing through
    ``_same`` gives a bool for every legal field value.

    ``__hash__`` stays the generated field-tuple hash (declared beside each
    ``__eq__``), which keeps ``a == b`` implying ``hash(a) == hash(b)``: the
    two differ only on arrays, and an array-valued field makes the instance
    unhashable either way (``hash(ndarray)`` raises), so no equal pair is left
    with different hashes.
    """
    if other.__class__ is not self.__class__:
        return NotImplemented
    return all(_same(getattr(self, f.name), getattr(other, f.name))
               for f in fields(self))


def _dataclass_hash(self: Any) -> int:
    """The field-tuple hash ``@dataclass(frozen=True, eq=True)`` would have
    generated.  Spelled out because these classes declare ``eq=False`` to keep
    :func:`_dataclass_eq` (dataclasses only skips a class-body ``__eq__``, and
    relying on that is a subtlety a reader should not have to know)."""
    return hash(tuple(getattr(self, f.name) for f in fields(self)))


# ---------------------------------------------------------------------------
# Lazily-borrowed validators
# ---------------------------------------------------------------------------
# The enum vocabularies these dataclasses check against are OWNED by the
# modules that implement them; this module borrows the live tuples rather than
# restating their contents, so a model added there is legal here on the same
# commit.  The import is in-function (a ``sys.modules`` dict hit after the
# first call) because ``_lens_real`` imports THIS module at its own module
# scope: a module-level import back would be a cycle.

_VOCAB_CACHE: Dict[str, Any] = {}
# Companion lock: the fill below writes four keys, and a concurrent reader must
# see either none or all of them (the repository's cache/lock pin requires one
# per module-level cache).
_VOCAB_CACHE_LOCK = threading.Lock()


def _vocab(name: str) -> Any:
    """Return one of ``_lens_real``'s validation tuples, cached."""
    try:
        return _VOCAB_CACHE[name]
    except KeyError:
        from . import _lens_real as _lr
        with _VOCAB_CACHE_LOCK:
            for k in ('_VALID_SURFACE_MODELS', '_VALID_WAVE_PROPAGATORS',
                      '_VALID_REMAP_ORDERS', '_VALID_ACCUMULATOR_STORE'):
                _VOCAB_CACHE[k] = getattr(_lr, k)
        return _VOCAB_CACHE[name]


def clear_lens_config_vocabulary_cache() -> None:
    """Drop the borrowed enum vocabularies so the next validation re-reads them.

    The cache holds four short tuples of strings/ints -- tens of bytes, not a
    grid -- so this is not a memory measure.  It exists because the entries are
    BORROWED from ``_lens_real`` at first use: a test (or an
    ``importlib.reload``) that swaps one of those tuples would otherwise be
    validated against the pre-swap copy for the life of the process.  Dropping
    them is always safe -- :func:`_vocab` refills on the next call.
    """
    with _VOCAB_CACHE_LOCK:
        _VOCAB_CACHE.clear()


# Enrol with the central cache-clearer registry (late-binding lambda, mirroring
# ``analysis/beam_stats.py``) so ``clear_asm_caches`` /
# ``clear_all_registered_caches`` drain this one too.  ``_cache_registry`` is a
# stdlib-only leaf in the package root, so this does not give this module a
# dependency on anything that could import it back.
try:
    import sys as _sys

    from .._cache_registry import register_cache_clearer as _register_cache_clearer
    _this_mod = _sys.modules[__name__]
    _register_cache_clearer(
        'lens_config_vocabulary',
        lambda: getattr(_this_mod, 'clear_lens_config_vocabulary_cache')(),
    )
except ImportError:  # pragma: no cover - registry always present in-tree
    pass


def _require_choice(fn_name: str, name: str, value: Any,
                    choices: Iterable[Any]) -> None:
    if value not in choices:
        raise ValueError(
            f"{fn_name}: {name}={value!r} is not a valid choice.  "
            f"Choose from {list(choices)}.")


def _require_bool(fn_name: str, name: str, value: Any) -> None:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"{fn_name}: {name}={value!r} must be a bool, got "
            f"{type(value).__name__}.")


def _require_positive_int(fn_name: str, name: str, value: Any,
                          minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(
            f"{fn_name}: {name}={value!r} must be an int >= {minimum}, got "
            f"{type(value).__name__}.")
    if int(value) < minimum:
        raise ValueError(
            f"{fn_name}: {name}={value!r} must be >= {minimum}.")


def _require_finite(fn_name: str, name: str, value: Any, *,
                    positive: bool = False) -> None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(
            f"{fn_name}: {name}={value!r} must be a real number, got "
            f"{type(value).__name__}.") from None
    if not np.isfinite(v):
        raise ValueError(
            f"{fn_name}: {name}={value!r} must be finite.")
    if positive and v <= 0.0:
        raise ValueError(
            f"{fn_name}: {name}={value!r} must be > 0.")


# ---------------------------------------------------------------------------
# LensGeometry
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class LensGeometry:
    """WHAT optical problem is being solved -- planes, media, reference.

    Every field defaults to the value the corresponding keyword defaults to on
    every entry point that accepts it, so ``LensGeometry()`` is exactly "no
    request".  Units are SI (metres, radians) throughout.

    Attributes
    ----------
    dy : float or None, default None
        Sample pitch along y [m].  ``None`` -> the library default
        (``set_default_dy``), and failing that ``dx`` -- i.e. a square grid.
        Accepted by: ``apply_real_lens``, ``apply_real_lens_traced``,
        ``apply_real_lens_maslov``, ``apply_real_lens_gbd``,
        ``apply_real_lens_fga``.
    output_plane_distance : float, default 0.0
        Extra propagation past the prescription's exit vertex [m].  ``0.0``
        returns the field AT the exit-vertex plane.  Accepted by:
        ``apply_real_lens_traced``, ``apply_real_lens_maslov``,
        ``apply_real_lens_gbd``, ``apply_real_lens_fga``,
        ``apply_real_lens_traced_multibranch``.
    output_plane_n : float, default 1.0
        Refractive index of the medium the output plane sits in
        (dimensionless).  Accepted by: ``apply_real_lens_maslov``,
        ``apply_real_lens_gbd``, ``apply_real_lens_traced_multibranch``.
    conjugate : float or None, default None
        Object/image conjugate distance [m] for the ray-angle-aware
        ``surface_model='displaced'`` family.  ``None`` -> collimated.
        Accepted by: ``apply_real_lens``.
    surface_model : str, default 'thin'
        ``'thin'`` / ``'displaced'`` / ``'tangent_facet'`` /
        ``'tangent_facet_remap'`` -- which per-surface screen the analytic
        model imprints.  Validated against
        ``_lens_real._VALID_SURFACE_MODELS``.  Accepted by:
        ``apply_real_lens``.
    clip_aperture : bool, default True
        Clip the input at the circular entrance aperture before decomposition
        (the analytic / traced / thin models always clip; GBD makes it a
        choice).  Accepted by: ``apply_real_lens_gbd``.
    carrier : Any, default None
        The reference congruence the screen is built against -- ``None``
        (plane wave), a conjugate distance [m], ``'auto'``, a wavefront
        ndarray, or a ``TiltedCarrier``.  Not validated here: its legal domain
        depends on the field and the prescription, and the entry points
        already adjudicate it.  Accepted by: ``apply_real_lens``,
        ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    origin : tuple of float, default (0.0, 0.0)
        ``(x0, y0)`` origin of the ray-trace launch grid relative to the
        optical axis [m].  Accepted by: ``apply_real_lens_traced``.
    beam_centre : Any, default None
        Beam centre ``(x, y)`` [m] used to place the OPL fit domain.
        ``None`` -> measured from the field.  Not validated here (the entry
        point resolves several spellings).  Accepted by:
        ``apply_real_lens_traced``.
    roi : Any, default None
        Output region of interest -- ``None`` (the whole grid) or the entry
        point's slice/bounds spelling.  Not validated here (each entry point
        resolves it against its own output grid).  Accepted by:
        ``apply_real_lens_maslov``, ``apply_real_lens_gbd``.
    """

    dy: Optional[float] = None
    output_plane_distance: float = 0.0
    output_plane_n: float = 1.0
    conjugate: Any = None
    surface_model: str = 'thin'
    clip_aperture: bool = True
    carrier: Any = None
    origin: Tuple[float, float] = (0.0, 0.0)
    beam_centre: Any = None
    roi: Any = None

    # ``carrier`` / ``beam_centre`` / ``roi`` legitimately hold arrays; see
    # :func:`_dataclass_eq` for why the generated ``__eq__`` cannot.
    __eq__ = _dataclass_eq
    __hash__ = _dataclass_hash

    def __post_init__(self) -> None:
        fn = 'LensGeometry'
        if self.dy is not None:
            _require_finite(fn, 'dy', self.dy, positive=True)
        _require_finite(fn, 'output_plane_distance', self.output_plane_distance)
        _require_finite(fn, 'output_plane_n', self.output_plane_n,
                        positive=True)
        _require_choice(fn, 'surface_model', self.surface_model,
                        _vocab('_VALID_SURFACE_MODELS'))
        _require_bool(fn, 'clip_aperture', self.clip_aperture)
        try:
            ox, oy = self.origin
        except (TypeError, ValueError):
            raise ValueError(
                f"{fn}: origin={self.origin!r} must be a 2-tuple (x0, y0) in "
                f"metres.") from None
        _require_finite(fn, 'origin[0]', ox)
        _require_finite(fn, 'origin[1]', oy)


# ---------------------------------------------------------------------------
# LensNumerics
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class LensNumerics:
    """HOW the same problem is discretised and solved.

    Changing a field here should move the answer only by its own truncation
    error; changing a :class:`LensGeometry` field changes the answer because
    the problem changed.  That is the line the partition is drawn on.

    Attributes
    ----------
    bandlimit : bool, default True
        Apply the transfer function's band limit during in-glass propagation.
        Accepted by: ``apply_real_lens``, ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``.
    wave_propagator : str or None, default None
        ``'asm'`` / ``'sas'`` / ``'fresnel'`` / ``'rayleigh_sommerfeld'`` /
        ``'rs'``; ``None`` -> the library default
        (``set_default_wave_propagator``).  Validated against
        ``_lens_real._VALID_WAVE_PROPAGATORS``.  Accepted by:
        ``apply_real_lens``, ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``.
    ray_subsample : int, default 8
        Coarse ray-grid spacing in units of ``dx`` for the traced OPL.
        Accepted by: ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
        NOT mapped to ``apply_real_lens_traced_multibranch.ray_subsample``,
        which is the CAUSTIC launch spacing -- see ``caustic_ray_subsample``.
    output_subsample : int, default 1
        Evaluate the output field on every n-th pixel and interpolate.
        Accepted by: ``apply_real_lens_maslov``, ``apply_real_lens_gbd``.
    remap_order : int, default 3
        Spline order (1/3/5) of the tangent-facet transverse remap.  Validated
        against ``_lens_real._VALID_REMAP_ORDERS``.  Accepted by:
        ``apply_real_lens``.
    min_coarse_samples_per_aperture : int, default 32
        Undersample guard: the minimum number of coarse ray samples across the
        aperture before ``on_undersample`` fires.  ``0`` disables the guard.
        Accepted by: ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    fit_radius_beam_factor : float or None, default None
        Restrict the OPL fit domain to this multiple of the measured beam
        radius (dimensionless).  ``None`` -> the entry point's own default
        policy.  Accepted by: ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``.
    newton_fit : str, default 'auto'
        OPL fit backend for the per-pixel Newton inversion: ``'auto'``
        (-> polynomial), ``'polynomial'`` or ``'spline'``.  The membership
        check lives in the entry point (see the module docstring).  Accepted
        by: ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    newton_poly_order : int, default 6
        Total degree of the Chebyshev OPL fit.  Accepted by:
        ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    newton_max_iters : int or None, default None
        Cap on per-pixel Newton iterations; ``None`` -> the module default.
        Accepted by: ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    inversion_method : str, default 'newton'
        ``'newton'`` / ``'fit'`` / ``'backward_trace'``.  Membership checked by
        the entry point.  Accepted by: ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``.
    amplitude_model : str, default 'screen'
        ``'screen'`` (wave-optics envelope) or ``'ray_density'`` (geometric
        Jacobian).  Membership checked by the entry point.  Accepted by:
        ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    caustic : str or None, default None
        ``None`` / ``'single'`` / ``'multibranch'`` / ``'uniform'`` /
        ``'wave'`` -- the through-focus strategy.  Requires
        ``amplitude_model='ray_density'``, which the entry point enforces.
        Accepted by: ``apply_real_lens_traced``.
    caustic_band : str, default 'ludwig'
        ``'ludwig'`` (uniform-Airy replacement in the Kravtsov-Orlov band) or
        ``'plain'`` (raw branch sum).  Accepted by:
        ``apply_real_lens_traced``, ``apply_real_lens_traced_multibranch``.
    caustic_ray_subsample : int, default 2
        Launch-grid spacing in units of ``dx`` for the multi-branch caustic
        trace.  Accepted by: ``apply_real_lens_traced``
        (``caustic_ray_subsample``) and ``apply_real_lens_traced_multibranch``
        (spelled ``ray_subsample`` there -- same setting, same default 2).
    caustic_min_area_ratio : float, default 1e-6
        Skip mapped triangles below this fraction of their launch area (the
        degenerate caustic set), dimensionless.  Accepted by:
        ``apply_real_lens_traced`` (``caustic_min_area_ratio``) and
        ``apply_real_lens_traced_multibranch`` (``min_area_ratio``).
    inverse_map : bool or None, default None
        Per-call override of the inverse-characteristic evaluator gate;
        ``None`` -> follow ``_lens_imap.TRACED_INVERSE_MAP``.  Accepted by:
        ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    """

    bandlimit: bool = True
    wave_propagator: Optional[str] = None
    ray_subsample: int = 8
    output_subsample: int = 1
    remap_order: int = 3
    min_coarse_samples_per_aperture: int = 32
    fit_radius_beam_factor: Optional[float] = None
    newton_fit: str = 'auto'
    newton_poly_order: int = 6
    newton_max_iters: Optional[int] = None
    inversion_method: str = 'newton'
    amplitude_model: str = 'screen'
    caustic: Optional[str] = None
    caustic_band: str = 'ludwig'
    caustic_ray_subsample: int = 2
    caustic_min_area_ratio: float = 1e-6
    inverse_map: Optional[bool] = None

    # Value equality through ``_same`` -- see :func:`_dataclass_eq`.  Declared
    # on all three so the group compares uniformly (a LensConfig comparison
    # reaches every component).
    __eq__ = _dataclass_eq
    __hash__ = _dataclass_hash

    def __post_init__(self) -> None:
        fn = 'LensNumerics'
        _require_bool(fn, 'bandlimit', self.bandlimit)
        if self.wave_propagator is not None:
            _require_choice(fn, 'wave_propagator', self.wave_propagator,
                            _vocab('_VALID_WAVE_PROPAGATORS'))
        _require_positive_int(fn, 'ray_subsample', self.ray_subsample)
        _require_positive_int(fn, 'output_subsample', self.output_subsample)
        _require_choice(fn, 'remap_order', self.remap_order,
                        _vocab('_VALID_REMAP_ORDERS'))
        _require_positive_int(fn, 'min_coarse_samples_per_aperture',
                              self.min_coarse_samples_per_aperture, minimum=0)
        if self.fit_radius_beam_factor is not None:
            _require_finite(fn, 'fit_radius_beam_factor',
                            self.fit_radius_beam_factor, positive=True)
        _require_positive_int(fn, 'newton_poly_order', self.newton_poly_order)
        if self.newton_max_iters is not None:
            _require_positive_int(fn, 'newton_max_iters', self.newton_max_iters)
        _require_positive_int(fn, 'caustic_ray_subsample',
                              self.caustic_ray_subsample)
        _require_finite(fn, 'caustic_min_area_ratio',
                        self.caustic_min_area_ratio, positive=True)
        if self.caustic_min_area_ratio > 1.0:
            raise ValueError(
                f"{fn}: caustic_min_area_ratio="
                f"{self.caustic_min_area_ratio!r} is a FRACTION of the launch "
                f"area and must be <= 1; above 1 every triangle is degenerate "
                f"and the branch sum is empty.")
        if self.inverse_map is not None:
            _require_bool(fn, 'inverse_map', self.inverse_map)


# ---------------------------------------------------------------------------
# LensResources
# ---------------------------------------------------------------------------

@dataclass(frozen=True, eq=False)
class LensResources:
    """What MACHINE the call may use, and what it reports while it runs.

    Nothing here may change the returned field.  ``sag_dtype`` is the one
    deliberate exception and says so: it trades ~1e-7 relative surface
    departure for half the dtype-independent memory core, which is why it
    carries an accuracy warning rather than living in
    :class:`LensNumerics`.

    Attributes
    ----------
    use_gpu : bool, default False
        Run the phase-screen + in-glass propagation on CuPy.  Accepted by:
        ``apply_real_lens``, ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``, ``apply_real_lens_maslov``.
    amp_use_gpu : bool, default False
        Run only the amplitude leg on the GPU.  Accepted by:
        ``apply_real_lens_traced``, ``prepare_real_lens_traced``.
    n_workers : int or None, default None
        Worker processes for the ray trace; ``None`` -> the library's
        affinity-aware default.  Accepted by: ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``.
    parallel_amp : bool or None, default None
        Run the amp and amp(pw) legs concurrently; ``None`` -> the process
        knob ``lumenairy.set_lens_parallel_amp``.  Byte-identical either way;
        ``False`` roughly halves the lens-step peak working set.  Accepted by:
        ``apply_real_lens_traced``.
    parallel_amp_min_free_gb : float, default 48.0
        Free-RAM floor [GB] below which the concurrent amp legs are declined.
        Accepted by: ``apply_real_lens_traced``.
    sag_dtype : dtype or None, default None
        Geometry (sag / coordinate / OPD) dtype -- ``None`` or
        ``np.float64`` for the byte-identical default, ``np.float32`` to halve
        the dtype-independent memory core.  ACCURACY-RISKY: validate with
        ``lens_sag_float32_opd_error`` before trusting a float32 result.
        Accepted by: ``apply_real_lens``, ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``.
    sag_chunk_rows : int or None, default None
        Row-band height for the chunked sag/screen loop.  ``None`` -> auto,
        ``0`` -> force the whole-grid path, ``> 0`` -> that band size.
        Byte-identical to the whole-grid path.  Accepted by:
        ``apply_real_lens``, ``apply_real_lens_traced``,
        ``prepare_real_lens_traced``.
    accumulator_store : str, default 'ram'
        ``'ram'`` or ``'memmap'`` -- where the persistent full-grid
        accumulators of the angle-true screen paths live.  Validated against
        ``_lens_real._VALID_ACCUMULATOR_STORE``.  Accepted by:
        ``apply_real_lens``.
    scratch_dir : str or os.PathLike or None, default None
        Directory for ``accumulator_store='memmap'`` scratch files; ``None``
        -> the platform temp dir.  Accepted by: ``apply_real_lens``.
    progress : Any, default None
        Progress sink handed to ``lumenairy.progress.call_progress``
        (a callable, a tqdm-like object, or ``None``).  Not validated here --
        ``call_progress`` adjudicates it.  Accepted by: ``apply_real_lens``,
        ``apply_real_lens_traced``, ``prepare_real_lens_traced``,
        ``apply_real_lens_maslov``, ``apply_real_lens_gbd``.
    verbose : bool, default False
        Print per-stage timing / diagnostics to stdout.  Accepted by:
        ``apply_real_lens_maslov``, ``apply_real_lens_gbd``.
    """

    use_gpu: bool = False
    amp_use_gpu: bool = False
    n_workers: Optional[int] = None
    parallel_amp: Optional[bool] = None
    parallel_amp_min_free_gb: float = 48.0
    sag_dtype: Any = None
    sag_chunk_rows: Optional[int] = None
    accumulator_store: str = 'ram'
    scratch_dir: Optional[str] = None
    progress: Any = None
    verbose: bool = False

    # Value equality through ``_same`` -- see :func:`_dataclass_eq`.
    # ``progress`` may be any sink object, including one whose ``__eq__``
    # returns an array.
    __eq__ = _dataclass_eq
    __hash__ = _dataclass_hash

    def __post_init__(self) -> None:
        fn = 'LensResources'
        _require_bool(fn, 'use_gpu', self.use_gpu)
        _require_bool(fn, 'amp_use_gpu', self.amp_use_gpu)
        _require_bool(fn, 'verbose', self.verbose)
        if self.n_workers is not None:
            _require_positive_int(fn, 'n_workers', self.n_workers)
        if self.parallel_amp is not None:
            _require_bool(fn, 'parallel_amp', self.parallel_amp)
        _require_finite(fn, 'parallel_amp_min_free_gb',
                        self.parallel_amp_min_free_gb)
        if float(self.parallel_amp_min_free_gb) < 0.0:
            raise ValueError(
                f"{fn}: parallel_amp_min_free_gb="
                f"{self.parallel_amp_min_free_gb!r} must be >= 0 GB.")
        if self.sag_dtype is not None:
            # One rule for all three spellings of this setting: the same one
            # ``_lens_real.set_lens_sag_dtype`` (the process knob) and
            # ``_lens_real._resolve_sag_real`` (the per-call keyword) enforce.
            # Accepting ``np.float16`` here and quietly giving float64 would be
            # exactly the discarded-setting class this module exists to close.
            try:
                d = np.dtype(self.sag_dtype)
            except TypeError:
                raise ValueError(
                    f"{fn}: sag_dtype={self.sag_dtype!r} is not a dtype.  "
                    f"Pass None (float64, the default), np.float64 or "
                    f"np.float32.") from None
            if d not in (np.dtype(np.float32), np.dtype(np.float64)):
                raise ValueError(
                    f"{fn}: sag_dtype={self.sag_dtype!r} must be float32 or "
                    f"float64 (the geometry lineage is real).")
        if self.sag_chunk_rows is not None:
            _require_positive_int(fn, 'sag_chunk_rows', self.sag_chunk_rows,
                                  minimum=0)
        _require_choice(fn, 'accumulator_store', self.accumulator_store,
                        _vocab('_VALID_ACCUMULATOR_STORE'))
        if self.scratch_dir is not None and not isinstance(
                self.scratch_dir, (str, os.PathLike)):
            raise ValueError(
                f"{fn}: scratch_dir={self.scratch_dir!r} must be a path-like "
                f"or None, got {type(self.scratch_dir).__name__}.")


# ---------------------------------------------------------------------------
# The per-entry-point field tables
# ---------------------------------------------------------------------------
# ``entry point -> {config field name: the keyword that entry point spells it
# with}``.  The mapping is a RENAME table, not an alias table: a field appears
# for an entry point only where the setting genuinely exists there with the
# same meaning AND the same default, and where the spelling differs the
# difference is called out in the field's docstring.  The three renames:
#
#   caustic_ray_subsample  -> multibranch ``ray_subsample``   (both default 2)
#   caustic_min_area_ratio -> multibranch ``min_area_ratio``  (both 1e-6)
#
# ``tests/unit/test_audit2609_a16_lens_config_round_trip.py`` asserts, for
# every pair below, that the keyword exists on that entry point and that its
# signature default equals the dataclass default -- so this table cannot drift
# from the signatures without turning red.

_GEOMETRY_FOR: Dict[str, Dict[str, str]] = {
    'apply_real_lens': {
        'dy': 'dy', 'conjugate': 'conjugate',
        'surface_model': 'surface_model', 'carrier': 'carrier'},
    'apply_real_lens_traced': {
        'dy': 'dy', 'output_plane_distance': 'output_plane_distance',
        'carrier': 'carrier', 'origin': 'origin',
        'beam_centre': 'beam_centre'},
    'prepare_real_lens_traced': {
        'carrier': 'carrier'},
    'apply_real_lens_maslov': {
        'dy': 'dy', 'output_plane_distance': 'output_plane_distance',
        'output_plane_n': 'output_plane_n', 'roi': 'roi'},
    'apply_real_lens_gbd': {
        'dy': 'dy', 'output_plane_distance': 'output_plane_distance',
        'output_plane_n': 'output_plane_n',
        'clip_aperture': 'clip_aperture', 'roi': 'roi'},
    'apply_real_lens_fga': {
        'dy': 'dy', 'output_plane_distance': 'output_plane_distance'},
    'apply_real_lens_traced_multibranch': {
        'output_plane_distance': 'output_plane_distance',
        'output_plane_n': 'output_plane_n'},
}

_NUMERICS_FOR: Dict[str, Dict[str, str]] = {
    'apply_real_lens': {
        'bandlimit': 'bandlimit', 'wave_propagator': 'wave_propagator',
        'remap_order': 'remap_order'},
    'apply_real_lens_traced': {
        'bandlimit': 'bandlimit', 'wave_propagator': 'wave_propagator',
        'ray_subsample': 'ray_subsample',
        'min_coarse_samples_per_aperture': 'min_coarse_samples_per_aperture',
        'fit_radius_beam_factor': 'fit_radius_beam_factor',
        'newton_fit': 'newton_fit', 'newton_poly_order': 'newton_poly_order',
        'newton_max_iters': 'newton_max_iters',
        'inversion_method': 'inversion_method',
        'amplitude_model': 'amplitude_model', 'caustic': 'caustic',
        'caustic_band': 'caustic_band',
        'caustic_ray_subsample': 'caustic_ray_subsample',
        'caustic_min_area_ratio': 'caustic_min_area_ratio',
        'inverse_map': 'inverse_map'},
    'prepare_real_lens_traced': {
        'bandlimit': 'bandlimit', 'wave_propagator': 'wave_propagator',
        'ray_subsample': 'ray_subsample',
        'min_coarse_samples_per_aperture': 'min_coarse_samples_per_aperture',
        'fit_radius_beam_factor': 'fit_radius_beam_factor',
        'newton_fit': 'newton_fit', 'newton_poly_order': 'newton_poly_order',
        'newton_max_iters': 'newton_max_iters',
        'inversion_method': 'inversion_method',
        'amplitude_model': 'amplitude_model',
        'inverse_map': 'inverse_map'},
    'apply_real_lens_maslov': {
        'output_subsample': 'output_subsample'},
    'apply_real_lens_gbd': {
        'output_subsample': 'output_subsample'},
    'apply_real_lens_fga': {},
    'apply_real_lens_traced_multibranch': {
        'caustic_band': 'caustic_band',
        'caustic_ray_subsample': 'ray_subsample',
        'caustic_min_area_ratio': 'min_area_ratio'},
}

_RESOURCES_FOR: Dict[str, Dict[str, str]] = {
    'apply_real_lens': {
        'use_gpu': 'use_gpu', 'sag_dtype': 'sag_dtype',
        'sag_chunk_rows': 'sag_chunk_rows',
        'accumulator_store': 'accumulator_store',
        'scratch_dir': 'scratch_dir', 'progress': 'progress'},
    'apply_real_lens_traced': {
        'use_gpu': 'use_gpu', 'amp_use_gpu': 'amp_use_gpu',
        'n_workers': 'n_workers', 'parallel_amp': 'parallel_amp',
        'parallel_amp_min_free_gb': 'parallel_amp_min_free_gb',
        'sag_dtype': 'sag_dtype', 'sag_chunk_rows': 'sag_chunk_rows',
        'progress': 'progress'},
    'prepare_real_lens_traced': {
        'use_gpu': 'use_gpu', 'amp_use_gpu': 'amp_use_gpu',
        'n_workers': 'n_workers', 'sag_dtype': 'sag_dtype',
        'sag_chunk_rows': 'sag_chunk_rows', 'progress': 'progress'},
    'apply_real_lens_maslov': {
        'use_gpu': 'use_gpu', 'progress': 'progress', 'verbose': 'verbose'},
    'apply_real_lens_gbd': {
        'progress': 'progress', 'verbose': 'verbose'},
    'apply_real_lens_fga': {},
    'apply_real_lens_traced_multibranch': {},
}

#: ``(attribute on LensConfig, dataclass, per-entry-point table)``, in the
#: order the resolver applies them.
_GROUPS: Tuple[Tuple[str, type, Dict[str, Dict[str, str]]], ...] = (
    ('geometry', LensGeometry, _GEOMETRY_FOR),
    ('numerics', LensNumerics, _NUMERICS_FOR),
    ('resources', LensResources, _RESOURCES_FOR),
)

#: The entry points that accept config objects, in declaration order.
ENTRY_POINTS: Tuple[str, ...] = tuple(_GEOMETRY_FOR)

#: Parameters that are deliberately NOT config fields, with the reason.
#: ``entry point -> {keyword: reason}``.  The round-trip test asserts that
#: every keyword of every entry point is either in one of the three tables
#: above, in the four-name contract set, or here -- so a parameter cannot be
#: forgotten, only classified.
CONTRACT_PARAMETERS: Tuple[str, ...] = (
    'E_in', 'prescription', 'wavelength', 'dx', 'N',
    'geometry', 'numerics', 'resources', 'config',
)

_PRIVATE_SINK = (
    "private diagnostics sink: an underscore-prefixed MUTABLE out-parameter "
    "the call writes into.  A frozen config must not carry one -- sharing a "
    "config between two calls would make them share a sink.")
_MODEL_PRIVATE = (
    "tuning constant of this model only; it has no sibling with the same "
    "name and semantics, so hoisting it would put a one-engine knob in a "
    "family-wide object.")
_PHYSICS_FLAG = (
    "analytic-model physics option (it changes which terms the screen "
    "carries, not the geometry or the discretisation).  Left as a keyword in "
    "this pass: the audit's partition names three roles and this is a fourth; "
    "see docs/lens_configuration.md 'Deferred: LensPhysics'.")

KWARG_ONLY: Dict[str, Dict[str, str]] = {
    'apply_real_lens': {
        'fresnel': _PHYSICS_FLAG,
        'slant_correction': _PHYSICS_FLAG,
        'absorption': _PHYSICS_FLAG,
        'seidel_correction': _PHYSICS_FLAG,
        'seidel_poly_order':
            "governs seidel_correction, which is itself keyword-only here; a "
            "numerics field whose enabling flag is not a field would be "
            "half-configurable.",
        'surface_frame': _PHYSICS_FLAG,
        'displaced_mode':
            "legal only under surface_model='displaced' and validated against "
            "it by _check_displaced_support; carrying it as an independent "
            "field would let a config declare an illegal pair that only the "
            "call can adjudicate.",
        'displaced_obliquity':
            "same as displaced_mode -- a sub-mode of surface_model.",
        'screen_obliquity':
            "legal only with carrier=; adjudicated by "
            "_check_screen_obliquity_support against the prescription.",
        'on_screen_obliquity':
            "policy knob ('warn'/'error'/'silent') for a single call's "
            "diagnostics, not a setting of the optical problem.",
        'stream_transfer_function':
            "memory/streaming strategy of the analytic in-glass leg only; it "
            "has no sibling and no family-wide meaning.",
    },
    'apply_real_lens_traced': {
        'on_undersample': "per-call diagnostic policy knob.",
        'on_noncollimated': "per-call diagnostic policy knob.",
        'on_aperture_beam': "per-call diagnostic policy knob.",
        'on_fit_domain_basis': "per-call diagnostic policy knob.",
        'on_pool_memory': "per-call diagnostic policy knob.",
        'preserve_input_phase': _MODEL_PRIVATE,
        'remap_sampling': _MODEL_PRIVATE,
        'tilt_aware_rays': _MODEL_PRIVATE,
        'decentred_fit_poly_order': _MODEL_PRIVATE,
        'newton_amp_mask_rel': _MODEL_PRIVATE,
        'newton_mask_dilate_coarse_px': _MODEL_PRIVATE,
        'fast_analytic_phase': _MODEL_PRIVATE,
        'return_screen':
            "changes the RETURN TYPE (field -> (field, screen)); a setting "
            "that changes what a function returns belongs at the call, not in "
            "a shared config object.",
        '_exit_na_out': _PRIVATE_SINK,
        '_remap_launch_out': _PRIVATE_SINK,
        '_imap_out': _PRIVATE_SINK,
    },
    'prepare_real_lens_traced': {
        'on_undersample': "per-call diagnostic policy knob.",
        'on_noncollimated': "per-call diagnostic policy knob.",
    },
    'apply_real_lens_maslov': {
        'ray_field_samples': _MODEL_PRIVATE,
        'ray_pupil_samples': _MODEL_PRIVATE,
        'poly_order': _MODEL_PRIVATE,
        'n_v2': _MODEL_PRIVATE,
        'extract_linear_phase': _MODEL_PRIVATE,
        'chunk_v2':
            "work-chunk size, but the family spells it three different ways "
            "with three different defaults (maslov chunk_v2=64, gbd "
            "chunk_beamlets=2048, fga chunk=None), so one field cannot carry "
            "an honest default.  See docs/lens_configuration.md 'Naming "
            "mismatches found'.",
        'use_numexpr': _MODEL_PRIVATE,
        'integration_method': _MODEL_PRIVATE,
        'stationary_newton_iter': _MODEL_PRIVATE,
        'stationary_newton_tol': _MODEL_PRIVATE,
        'local_n_samples': _MODEL_PRIVATE,
        'local_window_sigma': _MODEL_PRIVATE,
        'levin_tol': _MODEL_PRIVATE,
        'collimated_input': _MODEL_PRIVATE,
        'input_na': _MODEL_PRIVATE,
        'fold_split': _MODEL_PRIVATE,
        'normalize_output':
            "DEFAULT CLASH: 'power' here, 'none' on gbd and fga.  A single "
            "field cannot default to both, and silently picking one would "
            "change a sibling's behaviour.  See docs/lens_configuration.md.",
    },
    'apply_real_lens_gbd': {
        'sample_step': _MODEL_PRIVATE,
        'beamlets_per_aperture': _MODEL_PRIVATE,
        'waist_factor': _MODEL_PRIVATE,
        'direction_sampling': _MODEL_PRIVATE,
        'reexpand': _MODEL_PRIVATE,
        'reexpand_carrier': _MODEL_PRIVATE,
        'reexpand_threshold': _MODEL_PRIVATE,
        'per_surface': _MODEL_PRIVATE,
        'jacobian': _MODEL_PRIVATE,
        'window': _MODEL_PRIVATE,
        'chunk_beamlets': "work-chunk size -- see maslov's chunk_v2 entry.",
        'mem_budget_mb':
            "DEFAULT CLASH: 512.0 here, None on fga.  See "
            "docs/lens_configuration.md.",
        'normalize_output':
            "DEFAULT CLASH -- see the maslov entry.",
        'diagnostics':
            "MUTABLE out-parameter (the call fills the dict the caller "
            "passes); a frozen config must not carry one.",
    },
    'apply_real_lens_fga': {
        'w0_factor': _MODEL_PRIVATE,
        'dq_step': _MODEL_PRIVATE,
        'p_max': _MODEL_PRIVATE,
        'n_p': _MODEL_PRIVATE,
        'nsig': _MODEL_PRIVATE,
        'mem_budget_mb': "DEFAULT CLASH -- see the gbd entry.",
        'chunk': "work-chunk size -- see maslov's chunk_v2 entry.",
        'prune_frac': _MODEL_PRIVATE,
        'coeff_frac': _MODEL_PRIVATE,
        'separable': _MODEL_PRIVATE,
        'coarse_stride': _MODEL_PRIVATE,
        'exact_jacobian': _MODEL_PRIVATE,
        'cache_trace': _MODEL_PRIVATE,
        'normalize_output': "DEFAULT CLASH -- see the maslov entry.",
        'momentum_sampling': _MODEL_PRIVATE,
    },
    'apply_real_lens_traced_multibranch': {
        'input_carrier':
            "NOT the same setting as LensGeometry.carrier despite the "
            "similar name: this one is a transverse carrier WAVEVECTOR "
            "(None | 'auto' | (kx, ky) in rad/m), while carrier= on the "
            "analytic and traced entry points is a reference CONGRUENCE "
            "(None | 'auto' | conjugate distance in m | wavefront ndarray | "
            "TiltedCarrier).  Mapping them onto one field would silently "
            "reinterpret metres as rad/m.  See docs/lens_configuration.md "
            "'Naming mismatches found'.",
        'return_diagnostics':
            "changes the RETURN TYPE -- see apply_real_lens_traced's "
            "return_screen.",
    },
}


# ---------------------------------------------------------------------------
# LensConfig
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LensConfig:
    """The :class:`LensGeometry` / :class:`LensNumerics` /
    :class:`LensResources` triple, so one object configures a whole call.

    ``config=LensConfig(...)`` is interchangeable with passing the three
    components separately; passing both is allowed and the explicit component
    REPLACES the config's::

        base = LensConfig(numerics=LensNumerics(newton_poly_order=8))
        E = apply_real_lens_traced(E_in, prescription=rx, wavelength=lam,
                                   dx=dx, config=base,
                                   resources=LensResources(n_workers=4))

    Examples
    --------
    >>> from lumenairy import LensConfig, LensGeometry, LensNumerics
    >>> cfg = LensConfig(geometry=LensGeometry(output_plane_distance=2.0e-3),
    ...                  numerics=LensNumerics(ray_subsample=2))
    >>> cfg.to_kwargs()
    {'output_plane_distance': 0.002, 'ray_subsample': 2}
    >>> LensConfig.from_kwargs(**cfg.to_kwargs()) == cfg
    True
    """

    geometry: LensGeometry = field(default_factory=LensGeometry)
    numerics: LensNumerics = field(default_factory=LensNumerics)
    resources: LensResources = field(default_factory=LensResources)

    def __post_init__(self) -> None:
        for attr, cls, _ in _GROUPS:
            got = getattr(self, attr)
            if not isinstance(got, cls):
                raise TypeError(
                    f"LensConfig: {attr}= must be a {cls.__name__}, got "
                    f"{type(got).__name__}.")

    # -- construction -----------------------------------------------------

    @classmethod
    def from_kwargs(cls, *, entry_point: Optional[str] = None,
                    **kwargs: Any) -> 'LensConfig':
        """Build a config from a FLAT keyword mapping.

        Parameters
        ----------
        entry_point : str, optional
            When given, ``kwargs`` are spelled the way that entry point spells
            them (so ``min_area_ratio`` is accepted for
            ``apply_real_lens_traced_multibranch``) and only that entry
            point's settings are accepted.  When omitted, the canonical field
            names are used and any config field is accepted.
        **kwargs
            ``field=value`` pairs.  A name that is not a config field raises,
            naming the field it is closest to owning -- a silently swallowed
            keyword here would defeat the whole point of the object.

        Returns
        -------
        LensConfig

        Raises
        ------
        ValueError
            On an unknown name, or on a name that exists but does not apply to
            ``entry_point``.

        Examples
        --------
        >>> cfg = LensConfig.from_kwargs(ray_subsample=2, use_gpu=True)
        >>> cfg.numerics.ray_subsample, cfg.resources.use_gpu
        (2, True)
        >>> LensConfig.from_kwargs(
        ...     entry_point='apply_real_lens_traced_multibranch',
        ...     min_area_ratio=1e-4).numerics.caustic_min_area_ratio
        0.0001
        """
        if entry_point is not None and entry_point not in _GEOMETRY_FOR:
            raise ValueError(
                f"LensConfig.from_kwargs: entry_point={entry_point!r} is not "
                f"a real-lens entry point.  Known: {list(ENTRY_POINTS)}.")
        buckets: Dict[str, Dict[str, Any]] = {a: {} for a, _, _ in _GROUPS}
        for name, value in kwargs.items():
            placed = False
            for attr, dcls, table in _GROUPS:
                if entry_point is None:
                    canonical = name if any(
                        f.name == name for f in fields(dcls)) else None
                else:
                    canonical = next(
                        (c for c, kw in table[entry_point].items()
                         if kw == name), None)
                if canonical is not None:
                    buckets[attr][canonical] = value
                    placed = True
                    break
            if not placed:
                raise ValueError(
                    f"LensConfig.from_kwargs: {name!r} is not a lens config "
                    f"field"
                    + (f" of {entry_point}" if entry_point else "")
                    + f".  Known fields: {sorted(cls.field_names())}."
                    + ("  Several of the family's keywords are deliberately "
                       "NOT config fields (private diagnostics sinks, "
                       "single-model tuning constants, and names whose "
                       "defaults disagree between siblings); see "
                       "docs/lens_configuration.md."))
        return cls(geometry=LensGeometry(**buckets['geometry']),
                   numerics=LensNumerics(**buckets['numerics']),
                   resources=LensResources(**buckets['resources']))

    # -- inspection -------------------------------------------------------

    @staticmethod
    def field_names() -> Tuple[str, ...]:
        """Every config field name across the three dataclasses, sorted."""
        out: List[str] = []
        for _, dcls, _ in _GROUPS:
            out.extend(f.name for f in fields(dcls))
        return tuple(sorted(out))

    def to_kwargs(self, *, entry_point: Optional[str] = None,
                  include_defaults: bool = False) -> Dict[str, Any]:
        """Flatten back to a keyword mapping.

        Parameters
        ----------
        entry_point : str, optional
            Restrict and RENAME to the keywords that entry point accepts, so
            the result can be splatted straight into it::

                kw = cfg.to_kwargs(entry_point='apply_real_lens_traced')
                E = apply_real_lens_traced(E_in, prescription=rx,
                                           wavelength=lam, dx=dx, **kw)

            Fields that entry point does not accept are dropped SILENTLY here
            -- unlike a config passed to the call itself, which raises.  That
            is the point of the method: it is the explicit "narrow this to
            what that function understands" step.
        include_defaults : bool, default False
            ``False`` (default) emits only the fields that differ from their
            dataclass default -- the REQUESTS, which is what makes
            ``from_kwargs(**to_kwargs())`` a round trip and what makes the
            result safe to splat next to other keywords.  ``True`` emits every
            applicable field.

        Returns
        -------
        dict
        """
        if entry_point is not None and entry_point not in _GEOMETRY_FOR:
            raise ValueError(
                f"LensConfig.to_kwargs: entry_point={entry_point!r} is not a "
                f"real-lens entry point.  Known: {list(ENTRY_POINTS)}.")
        out: Dict[str, Any] = {}
        for attr, dcls, table in _GROUPS:
            obj = getattr(self, attr)
            allowed = table[entry_point] if entry_point is not None else None
            for f in fields(dcls):
                value = getattr(obj, f.name)
                if not include_defaults and _same(value, f.default):
                    continue
                if allowed is None:
                    out[f.name] = value
                elif f.name in allowed:
                    out[allowed[f.name]] = value
        return out

    def narrowed_to(self, entry_point: str) -> 'LensConfig':
        """Return a copy with every field this entry point does not accept
        reset to its default.

        The escape hatch for sharing one config across engines: a
        :class:`LensNumerics` built for ``apply_real_lens_traced`` carries
        Newton settings that ``apply_real_lens`` has no parameter for, and
        handing it to ``apply_real_lens`` RAISES rather than dropping them
        silently.  ``cfg.narrowed_to('apply_real_lens')`` is the explicit "yes,
        drop them" statement.

        >>> cfg = LensConfig.from_kwargs(newton_poly_order=8, bandlimit=False)
        >>> cfg.narrowed_to('apply_real_lens').to_kwargs()
        {'bandlimit': False}
        """
        if entry_point not in _GEOMETRY_FOR:
            raise ValueError(
                f"LensConfig.narrowed_to: entry_point={entry_point!r} is not "
                f"a real-lens entry point.  Known: {list(ENTRY_POINTS)}.")
        parts: Dict[str, Any] = {}
        for attr, dcls, table in _GROUPS:
            obj = getattr(self, attr)
            keep = {f.name: getattr(obj, f.name) for f in fields(dcls)
                    if f.name in table[entry_point]}
            parts[attr] = dcls(**keep)
        return LensConfig(**parts)

    def requests(self) -> Dict[str, Any]:
        """``{field: value}`` for every field that differs from its default.

        Alias for ``to_kwargs()``; spelled out because "what did this config
        actually ask for?" is the question a diagnostic wants."""
        return self.to_kwargs()


# ---------------------------------------------------------------------------
# The resolver the entry points call
# ---------------------------------------------------------------------------

_SIGNATURE_INFO: Dict[int, Dict[str, Any]] = {}

#: The four parameters this module adds to every entry point.  They are
#: stripped from the forwarded keywords so re-entering the entry point with
#: the merged mapping terminates instead of recursing.
_CONFIG_PARAMETERS = ('geometry', 'numerics', 'resources', 'config')


def _signature_info(fn: Callable[..., Any]) -> Dict[str, Any]:
    """``{'params': (...), 'defaults': {...}}`` for one entry point, cached.

    ``params`` is every KEYWORD-ONLY parameter except the four config ones --
    i.e. exactly what has to be forwarded, including the three required ones
    (``prescription`` / ``wavelength`` / ``dx``) that have no default.
    ``defaults`` is the subset that has one, which is what the precedence rule
    compares against.  ``E_in`` is positional-or-keyword on every entry point
    and is passed positionally by the caller, so it is not here.

    Cached by ``id(fn)`` with the function held in the entry, so a reloaded
    module gets a fresh entry rather than a stale one.
    """
    key = id(fn)
    hit = _SIGNATURE_INFO.get(key)
    if hit is not None and hit['fn'] is fn:
        return hit
    import inspect
    params = []
    defaults: Dict[str, Any] = {}
    for name, p in inspect.signature(fn).parameters.items():
        if p.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        if name in _CONFIG_PARAMETERS:
            continue
        params.append(name)
        if p.default is not inspect.Parameter.empty:
            defaults[name] = p.default
    info = {'fn': fn, 'params': tuple(params), 'defaults': defaults}
    _SIGNATURE_INFO[key] = info
    return info


def _accepting(field_name: str) -> str:
    """Human list of the entry points that accept ``field_name``."""
    who = sorted({ep for _, _, table in _GROUPS
                  for ep, m in table.items() if field_name in m})
    return ', '.join(who) if who else '(none)'


def resolve_entry_point_kwargs(fn: Callable[..., Any],
                               caller_locals: Mapping[str, Any], *,
                               geometry: Optional[LensGeometry] = None,
                               numerics: Optional[LensNumerics] = None,
                               resources: Optional[LensResources] = None,
                               config: Optional[LensConfig] = None,
                               ) -> Dict[str, Any]:
    """Merge config objects into one entry point's keyword arguments.

    Called by each entry point as its second executable statement, ONLY when
    at least one of the four config parameters is not ``None`` -- so a call
    that passes none of them never reaches this function and runs exactly the
    code it ran before the config objects existed.

    Parameters
    ----------
    fn : callable
        The entry point itself.  Its ``__name__`` selects the field tables and
        its signature supplies the defaults the precedence rule compares
        against.
    caller_locals : mapping
        ``locals()`` taken at the top of ``fn``'s body, before any parameter
        has been rebound.  Only the names in ``fn``'s signature are read, so
        stray locals (the lazily imported guard, for instance) are ignored.
    geometry, numerics, resources, config
        What the caller passed.

    Returns
    -------
    dict
        Every keyword ``fn`` accepts, with the config's requests merged in --
        ready to splat into ``fn`` itself.  The four config parameters are NOT
        in the result, so re-entering ``fn`` with it terminates.

    Raises
    ------
    ValueError
        If a config field that is SET does not apply to this entry point
        (``narrowed_to`` is the explicit way to drop it), or if a set field
        and an explicit keyword for the same setting disagree.
    TypeError
        If any of the four is not of its declared type.
    """
    fn_name = getattr(fn, '__name__', str(fn))
    if fn_name not in _GEOMETRY_FOR:
        raise ValueError(
            f"resolve_entry_point_kwargs: {fn_name!r} has no config field "
            f"table.  Known: {list(ENTRY_POINTS)}.")

    if config is not None and not isinstance(config, LensConfig):
        raise TypeError(
            f"{fn_name}: config= must be a LensConfig, got "
            f"{type(config).__name__}.")
    # An explicitly passed component REPLACES the config's, whole.  Component
    # granularity is deliberate: there is then no way for ``config=`` and
    # ``numerics=`` to half-disagree about one field.
    chosen: Dict[str, Any] = {}
    for attr, dcls, _ in _GROUPS:
        explicit = {'geometry': geometry, 'numerics': numerics,
                    'resources': resources}[attr]
        if explicit is not None and not isinstance(explicit, dcls):
            raise TypeError(
                f"{fn_name}: {attr}= must be a {dcls.__name__}, got "
                f"{type(explicit).__name__}.")
        if explicit is not None:
            chosen[attr] = explicit
        elif config is not None:
            chosen[attr] = getattr(config, attr)
        else:
            chosen[attr] = None

    info = _signature_info(fn)
    defaults = info['defaults']
    out = {name: caller_locals[name] for name in info['params']}

    for attr, dcls, table in _GROUPS:
        obj = chosen[attr]
        if obj is None:
            continue
        accepted = table[fn_name]
        for f in fields(dcls):
            value = getattr(obj, f.name)
            if _same(value, f.default):
                continue                     # not a request; nothing to do
            kw = accepted.get(f.name)
            if kw is None:
                raise ValueError(
                    f"{fn_name}: {attr}.{f.name}={value!r} is not a setting "
                    f"{fn_name} accepts (it applies to: {_accepting(f.name)})."
                    f"  Passing it here would silently discard it.  Use "
                    f"config.narrowed_to({fn_name!r}) to drop the fields this "
                    f"entry point has no parameter for, or move the setting "
                    f"to the entry point that owns it.")
            given = out[kw]
            if not _same(given, defaults[kw]):
                # The caller ALSO passed the keyword explicitly.
                if not _same(given, value):
                    raise ValueError(
                        f"{fn_name}: {kw}={given!r} was passed explicitly but "
                        f"{attr}.{f.name}={value!r} asks for a different "
                        f"value for the same setting.  A keyword and a config "
                        f"field must agree; drop one of them (the keyword "
                        f"wins nothing here on purpose -- silently preferring "
                        f"either one is how a setting gets discarded).")
                continue                     # they agree; leave it alone
            out[kw] = value
    return out


def _wants_config(geometry: Optional[LensGeometry],
                  numerics: Optional[LensNumerics],
                  resources: Optional[LensResources],
                  config: Optional[LensConfig]) -> bool:
    """True iff any config object was passed.  Spelled once so the four-way
    test at the top of seven entry points cannot drift apart."""
    return (geometry is not None or numerics is not None
            or resources is not None or config is not None)
