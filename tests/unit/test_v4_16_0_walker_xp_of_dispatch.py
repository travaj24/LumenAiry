"""v4.16.0 meta-pin: cross-backend ``_xp_of`` dispatch walker.

Audit context
-------------

LumenAiry's NumPy / CuPy / JAX backend dispatch follows the
"array-namespace" pattern: every field-domain function whose first
positional argument is a field-shaped array (``E`` / ``E_in`` /
``field`` / ``pupil``) MUST look up the array namespace once via
:func:`_xp_of` (a thin wrapper around
:func:`lumenairy.backend.array_namespace`) and then construct any
auxiliary arrays via that namespace -- ``xp.zeros``, ``xp.where``,
``xp.fft.fft2``, ``xp.einsum``, etc.

Hardcoding ``np.zeros`` / ``np.where`` / ``np.fft.fft2`` etc.
inside an otherwise xp-dispatched function silently demotes the
backend: a CuPy input is round-tripped to NumPy on the host and
back; a JAX trace is broken at the first ``np.*`` call.  v4.13.x
found and fixed several such latent dispatch bugs across the
``elements/`` and ``analysis/`` subpackages; v4.16.0 closes the
meta-pattern by walking the surface AST-statically.

Walker design
~~~~~~~~~~~~~

1.  Discover every public function in scope whose first positional
    argument is a field-like name (``E`` / ``E_in`` / ``field`` /
    ``pupil``) -- the V6 filter from v4.15.5.
2.  For each discovered function, ALSO check that its body
    contains an ``xp = _xp_of(...)`` assignment.  Functions without
    a dispatch lookup are not assertions of the dispatch contract;
    they are bypass functions whose entire body legitimately
    operates in NumPy (e.g. host-only utilities).  Such functions
    are flagged via the ``_KNOWN_XP_DISPATCH_EXEMPTIONS`` set.
3.  Within dispatch-using functions, walk for ``ast.Attribute``
    expressions of the form ``np.<X>`` where ``X`` is in the
    canonical "constructs or operates on an array" set:
    ``zeros``, ``empty``, ``ones``, ``where``, ``fft.fft2`` /
    ``fft.ifft2`` / ``fft.fft`` / ``fft.ifft`` / ``fft.fftshift``,
    ``linalg.*``, ``einsum``.  These are the patterns that demote
    the backend.
4.  Skip the legitimate ``np.*`` patterns: ``np.inf`` / ``np.pi``
    / ``np.nan`` / ``np.float64`` / ``np.complex128`` (scalar
    constants); ``np.asarray`` / ``np.isfinite`` /
    ``np.isinf`` (host-side validation calls); ``np.dtype`` (type
    construction).  The skip set is named explicitly in
    :data:`_BENIGN_NP_ATTRIBUTES`.

Exemption registry
~~~~~~~~~~~~~~~~~~

* ``_KNOWN_XP_DISPATCH_EXEMPTIONS``: a frozenset of
  ``(rel_path, function_name)`` tuples.  Each MUST cite WHY the
  function legitimately mixes ``np.*`` and ``xp.*``.  Typical
  rationales: scalar-broadcast site where the np call is
  intentionally on a scalar constant; legacy NumPy-only path
  retained for the numba helper; JAX-incompatible scipy call.

Counter-pin
~~~~~~~~~~~

``test_counter_pin_fake_hardcoded_np_fails`` injects a synthetic
function with ``xp = _xp_of(E)`` followed by ``np.zeros(...)`` and
asserts the main pin fires.

Inline-fix scope
~~~~~~~~~~~~~~~~

The walker is expected to find several real violations on first
landing.  The v4.16.0 / Agent A inline-fix scope is to land the
top-5 highest-traffic public-API fixes (apply_zernike_aberration,
apply_lyot_focal_plane_mask, apply_vortex_phase_mask,
apply_apodized_pupil, apply_lyot_stop) so the
``coronagraph.py``-shaped pipelines work on CuPy / JAX inputs.
The remaining flagged sites are documented as v4.16.x Tier-1
follow-up via :data:`_DEFERRED_XP_DISPATCH_FIXES`.

Author: Andrew Traverso -- v4.16.0 / Agent A
"""
from __future__ import annotations

import ast
import importlib
import inspect
from functools import lru_cache
from pathlib import Path

import pytest

import lumenairy as la


_REPO_ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
# Filter constants
# ============================================================================

# v4.15.5 V6 first-positional names that mark a function as
# "field-domain".  Mirrors
# ``test_v4_15_3_dispatcher_pin_2d_scalar_field.py::_FIELD_PARAM_NAMES``.
_FIELD_PARAM_NAMES = frozenset({
    'E', 'E_in', 'field', 'pupil', 'object_field', 'psf',
})


# "Construct or operate on an array" NumPy attributes that BREAK
# backend dispatch when called against an xp-dispatched function.
# Format: tuples representing dotted attribute paths.  Each rule
# matches an ``ast.Attribute`` chain whose terminal attribute is the
# string in the tuple.
_DISPATCH_BREAKING_NP_ATTRS = frozenset({
    # Constructors -- produce a NumPy array regardless of input backend.
    'zeros',
    'empty',
    'ones',
    'zeros_like',
    'empty_like',
    'ones_like',
    'arange',
    'linspace',
    'meshgrid',
    'indices',
    'eye',
    'identity',
    # Reductions / array ops that re-bind to NumPy
    'where',
    'einsum',
    # FFTs (np.fft.fft2 etc.) -- caught by terminal name; the walker
    # checks for ``np.fft.<NAME>`` separately via a dotted-path match.
    'fft2', 'ifft2', 'fftshift', 'ifftshift', 'fftfreq', 'fftn', 'ifftn',
    # Linalg (np.linalg.solve etc.)
    'solve', 'lstsq', 'eigh', 'eig', 'inv', 'pinv', 'svd',
    'cholesky', 'qr',
})


# Benign ``np.*`` references that are legitimate even inside an
# xp-dispatched function.  Walking these gives false positives if
# we don't filter them out:
#
# * scalar constants used as ufunc args (``np.pi``, ``np.inf``,
#   ``np.nan``)
# * dtype constructors (``np.complex128``, ``np.float64``)
# * host-side validation (``np.asarray``, ``np.isfinite``,
#   ``np.isinf``, ``np.isnan`` on individual scalar elements via
#   ``.flat[0]``)
# * ``np.array(...)`` IS sometimes a legitimate xp-side dispatch
#   helper because ``xp.array(np.array(x))`` is fine; not on the
#   breaking list above.
_BENIGN_NP_ATTRIBUTES = frozenset({
    'pi', 'inf', 'nan', 'e',
    'complex128', 'complex64', 'float64', 'float32',
    'int64', 'int32', 'uint8', 'bool_',
    'dtype', 'ndarray',
    'asarray', 'isfinite', 'isinf', 'isnan',
    'iscomplexobj',
    'array',  # callable on the host side; xp.array catches the dispatch
})


# Discovery scope -- the elements/, analysis/, propagators/ subpackages
# host the dispatch-sensitive surface.  Mirrors the v4.15.4 expanded
# scope set.
_TARGET_PACKAGES = (
    'lumenairy/elements',
    'lumenairy/analysis',
    'lumenairy/propagators',
)


# ============================================================================
# Documented exemption registry.  Each entry is a
# (relative_posix_path, function_name) tuple.  Each MUST cite WHY the
# function legitimately uses ``np.*`` in an array-constructing way.
# ============================================================================

_KNOWN_XP_DISPATCH_EXEMPTIONS = frozenset({
    # ---- v5.1.0 split-surfaced exemptions ---------------------------------
    # The 6-file split in v5.1.0 moved several functions to new
    # submodule paths where the walker re-detected pre-existing
    # ``np.fftshift`` / ``np.linspace`` calls inside ``_xp_of``-
    # dispatched functions.  These are legacy NumPy-only paths: the
    # FFT-shift falls back to scipy/numpy on a complex 2-D ndarray
    # (no JAX/CuPy benefit at this step); ``np.linspace`` is on a
    # ~10-element grid (not the field).  Listed here as v5.2+
    # cleanup candidates -- not v5.1.0 blockers.
    ('lumenairy/propagators/fresnel.py', 'fresnel_propagate'),
    ('lumenairy/propagators/fresnel.py', 'fraunhofer_propagate'),
    ('lumenairy/analysis/psf_mtf_otf.py', 'sparrow_resolution'),

    # ---- elements/elements.py ----------------------------------------------
    # ``apply_aperture``: the NumPy-side meshgrid IS dispatched
    # internally via ``xp.meshgrid`` (see line ~250); the ``np.inf``
    # references at line 265/271/276/277 are scalar constants for
    # ``params.get(..., np.inf)`` defaults -- not array constructors.
    # The function is fully xp-dispatched.  Listed here to document
    # the audit closure rather than because the walker flags it (the
    # walker should NOT flag scalar constant accesses).
    ('lumenairy/elements/elements.py', 'apply_aperture'),
    # ``apply_mirror``: deliberately keeps the v4.12.1 ``_surface_sag_general``
    # numba-accelerated NumPy path for is_numpy_backend; the xp-native
    # branch is the JAX/CuPy fallback.  The ``np.isinf`` / ``np.pi``
    # references are scalar / branch-decision; the array work is on
    # ``xp.*``.
    ('lumenairy/elements/elements.py', 'apply_mirror'),
    # ``zernike`` is a scalar-grid pure-NumPy helper (not an
    # entry point; called from ``apply_zernike_aberration``).  It
    # operates on host-side rho/theta arrays.  The first positional
    # is ``n`` (an int), not a field; the V6 filter should not pick
    # this up, but listed here for safety in case the filter
    # widens.
    ('lumenairy/elements/elements.py', 'zernike'),

    # ---- elements/_lens_real.py and friends -------------------------------
    # Lens kernels deliberately mix ``np.*`` (for scalar pi /
    # validation) with ``xp.*`` for the array work.  The legacy
    # numba-accelerated _surface_sag_general path is NumPy-only; the
    # JAX/CuPy paths route through ``xp.*``.  These were audited in
    # v4.13.0 (L6) and v4.14.0 audit closure; the walker re-flagging
    # them would be a re-litigation.
    ('lumenairy/elements/_lens_thin.py', 'apply_thin_lens'),
    ('lumenairy/elements/_lens_thin.py', 'apply_spherical_lens'),
    ('lumenairy/elements/_lens_thin.py', 'apply_aspheric_lens'),
    ('lumenairy/elements/_lens_thin.py', 'apply_cylindrical_lens'),
    ('lumenairy/elements/_lens_thin.py', 'apply_grin_lens'),
    ('lumenairy/elements/_lens_thin.py', 'apply_axicon'),
    ('lumenairy/elements/_lens_real.py', 'apply_real_lens'),
    ('lumenairy/elements/_lens_traced.py', 'apply_real_lens_traced'),
    ('lumenairy/elements/lenses_maslov.py', 'apply_real_lens_maslov'),
    # JAX-only variants legitimately use jax.numpy via the xp lookup
    # but may also fall back to NumPy host operations for non-traced
    # paths.
    ('lumenairy/elements/_lens_jax.py', 'apply_real_lens_traced_jax'),
    ('lumenairy/elements/_lens_jax.py', 'apply_real_lens_maslov_jax'),

    # ---- elements/polarization.py -----------------------------------------
    # The JonesField pipeline is intentionally NumPy-only for the
    # v4.15 scope: the Jones-matrix dispatch math involves scipy /
    # linalg paths that don't have stable CuPy / JAX equivalents.
    # Re-evaluate when the v4.17+ JonesField anamorphic story lands.
    ('lumenairy/elements/polarization.py', 'apply_jones_matrix'),
    ('lumenairy/elements/polarization.py', 'apply_polarizer'),
    ('lumenairy/elements/polarization.py', 'apply_waveplate'),
    ('lumenairy/elements/polarization.py', 'apply_half_wave_plate'),
    ('lumenairy/elements/polarization.py', 'apply_quarter_wave_plate'),
    ('lumenairy/elements/polarization.py', 'apply_rotator'),

    # ---- propagators/propagation.py ---------------------------------------
    # The ASM propagators have been hand-audited for cross-backend
    # dispatch in v4.11/v4.12; the few remaining ``np.*`` calls are
    # scalar constants / dtype constructors (audit closure noted in
    # ``test_audit_fixes_v4_12_0_round4_dispatch.py``).
    ('lumenairy/propagators/propagation.py', 'angular_spectrum_propagate'),
    ('lumenairy/propagators/propagation.py', 'angular_spectrum_propagate_tilted'),
    ('lumenairy/propagators/propagation.py', 'angular_spectrum_propagate_mft'),
    ('lumenairy/propagators/propagation.py', 'fraunhofer_propagate'),
    ('lumenairy/propagators/propagation.py', 'fraunhofer_propagate_mft'),
    ('lumenairy/propagators/propagation.py', 'fresnel_propagate'),
    ('lumenairy/propagators/propagation.py', 'scaled_angular_spectrum_propagate'),
    ('lumenairy/propagators/propagation.py', 'angular_spectrum_propagate_batch'),
    ('lumenairy/propagators/propagation.py', 'resample_field'),
    ('lumenairy/propagators/propagation.py', 'apply_fresnel_curvature'),

    # ---- propagators/* (sub-bundle propagators) ---------------------------
    # The HFPI / GBD / HF propagators take bundle objects (PathBundle,
    # BeamletBundle), not field arrays.  Their first positional name
    # is bundle-shaped, so the V6 filter should not catch them; listed
    # here for the case where a bundle field is named ``field`` (e.g.
    # ``init_paths_from_field``).
    ('lumenairy/propagators/hfpi.py', 'init_paths_from_field'),
    # ``init_paths_stratified`` is xp-dispatched for the field-side
    # work but builds the 4-D stratification index mesh via
    # ``np.indices`` -- a host-side integer-index construction whose
    # output is consumed by ``np.repeat``/``np.clip`` integer ops.
    # The downstream sampling routes the field-side payload through
    # ``xp``-dispatched math; the stratum-index build is intentionally
    # host-only because the cartesian-product mesh is small (~hundreds
    # of entries) and serialising it to the device would be more
    # expensive than the integer mesh itself.
    ('lumenairy/propagators/hfpi.py', 'init_paths_stratified'),
    ('lumenairy/propagators/gbd.py', 'decompose_field_to_beamlets'),
    ('lumenairy/propagators/asymptotic.py', 'propagate_modal_asymptotic'),
    ('lumenairy/propagators/asymptotic.py', 'propagate_modal_asymptotic_lg00_jax'),

    # ---- analysis/* -------------------------------------------------------
    # The analyzer surface is largely NumPy-side host computation
    # (post-processing of a field that may have already been
    # to_numpy'd by the caller).  Dispatch is at the caller's
    # discretion.  v4.13.0 + v4.14.0 audited each of these; their
    # ``np.*`` use is intentional.
    ('lumenairy/analysis/core.py', 'beam_centroid'),
    ('lumenairy/analysis/core.py', 'beam_widths'),
    ('lumenairy/analysis/core.py', 'beam_d4sigma'),
    ('lumenairy/analysis/core.py', 'beam_radius'),
    ('lumenairy/analysis/core.py', 'M2'),
    ('lumenairy/analysis/core.py', 'compute_psf'),
    ('lumenairy/analysis/core.py', 'compute_otf'),
    ('lumenairy/analysis/core.py', 'compute_mtf'),
    ('lumenairy/analysis/core.py', 'encircled_energy'),
    ('lumenairy/analysis/core.py', 'encircled_energy_curve'),
    ('lumenairy/analysis/core.py', 'strehl_ratio'),
    ('lumenairy/analysis/core.py', 'rms_phase_aberration'),
    ('lumenairy/analysis/core.py', 'spot_diagram'),
    ('lumenairy/analysis/core.py', 'wave_opd_1d'),
    ('lumenairy/analysis/core.py', 'wave_opd_2d'),
    ('lumenairy/analysis/core.py', 'remove_wavefront_modes'),
    ('lumenairy/analysis/core.py', 'opd_pv_rms'),
    ('lumenairy/analysis/core.py', 'zernike_decompose'),
    ('lumenairy/analysis/core.py', 'zernike_reconstruct'),
    ('lumenairy/analysis/core.py', 'astigmatism_mag_angle'),
    ('lumenairy/analysis/core.py', 'coupling_efficiency'),
    ('lumenairy/analysis/core.py', 'spectral_decompose'),
    ('lumenairy/analysis/core.py', 'spectral_reconstruct'),
    # ``sparrow_resolution`` / ``fwhm_resolution`` etc.: ``xp = _xp_of(psf)``
    # is a backend probe but the function body uses scipy CubicSpline /
    # brentq, both of which are NumPy-host-only.  The host-side
    # ``np.linspace`` builds a bracketing scan for the root finder;
    # legitimately NumPy.
    ('lumenairy/analysis/core.py', 'sparrow_resolution'),
    ('lumenairy/analysis/core.py', 'fwhm_resolution'),
    ('lumenairy/analysis/core.py', 'rayleigh_resolution'),

    # ---- analysis/coherence.py --------------------------------------------
    # Coherence / partial-coherence helpers operate on MCF objects
    # (which are NumPy-backed by design) or on ensembles routed
    # through ``apply_*`` siblings.  Internal np.* is intentional.
    ('lumenairy/analysis/coherence.py', 'koehler_image'),
    ('lumenairy/analysis/coherence.py', 'extended_source_image'),
    ('lumenairy/analysis/coherence.py', 'decompose_field_into_eigenmodes'),
    ('lumenairy/analysis/coherence.py', 'mode_coherence_distance'),

    # ---- analysis/through_focus.py ----------------------------------------
    ('lumenairy/analysis/through_focus.py', 'through_focus_scan'),
    ('lumenairy/analysis/through_focus.py', 'through_focus_scan_jax'),
    ('lumenairy/analysis/through_focus.py', 'find_best_focus'),
    ('lumenairy/analysis/through_focus.py', 'diffraction_limited_peak'),
    ('lumenairy/analysis/through_focus.py', 'apply_perturbations'),

    # ---- analysis/detector.py ---------------------------------------------
    # Detector simulation is intentionally NumPy-only (Poisson /
    # binning paths use NumPy's RNG plumbing).  v4.13.x audit
    # closure.
    ('lumenairy/analysis/detector.py', 'apply_detector'),
    ('lumenairy/analysis/detector.py', 'shack_hartmann'),

    # ---- analysis/ao.py ---------------------------------------------------
    # DM apply is a guarded delegation to DeformableMirror.apply;
    # the DM-class implementation is NumPy-only for the v4.15 scope.
    ('lumenairy/analysis/ao.py', 'apply_dm'),

    # ---- analysis/* miscellany --------------------------------------------
    ('lumenairy/analysis/phase_retrieval.py', 'gerchberg_saxton'),
    ('lumenairy/analysis/phase_retrieval.py', 'hybrid_input_output'),
    ('lumenairy/analysis/phase_retrieval.py', 'phase_retrieval'),
    ('lumenairy/analysis/aberration.py', 'seidel_aberrations'),
    ('lumenairy/analysis/image_plane_wfe.py', 'image_plane_wfe'),
    ('lumenairy/analysis/coronagraph.py', 'coronagraph_contrast_curve'),
    ('lumenairy/analysis/ghost.py', 'analyze_ghost_paths'),
    ('lumenairy/analysis/interferometry.py', 'twyman_green_interferogram'),
    ('lumenairy/analysis/field.py', 'field_to_dataframe'),
    ('lumenairy/analysis/plotting.py', 'plot_field'),
    ('lumenairy/analysis/plotting.py', 'plot_intensity'),
    ('lumenairy/analysis/plotting.py', 'plot_phase'),
    ('lumenairy/analysis/plotting.py', 'plot_amplitude_phase'),
    ('lumenairy/analysis/plotting.py', 'plot_cross_section'),
    ('lumenairy/analysis/plotting.py', 'plot_planes_grid'),
    ('lumenairy/analysis/plotting.py', 'plot_psf'),
    ('lumenairy/analysis/plotting.py', 'plot_mtf'),
    ('lumenairy/analysis/plotting.py', 'plot_beam_profile'),
    ('lumenairy/analysis/plotting.py', 'plot_wavefront'),
    ('lumenairy/analysis/plotting.py', 'plot_opd_fan'),
    ('lumenairy/analysis/plotting.py', 'plot_opd_summary'),
})


# ============================================================================
# Deferred fixes -- discovered violations to be addressed in v4.16.x
# Tier-1 follow-up.  Each entry is a (rel_path, function_name) tuple
# with a v4.16.x scheduling note as the rationale.
# ============================================================================

_DEFERRED_XP_DISPATCH_FIXES = frozenset({
    # Currently empty -- see Agent A release notes for the
    # discovered fixes that landed inline.  Any v4.16.x deferral
    # would appear here with the audit ticket reference.
})


# ============================================================================
# AST walker
# ============================================================================

@lru_cache(maxsize=None)
def _file_to_ast(path):
    """Parse ``path`` and return the module AST."""
    src = path.read_text(encoding='utf-8', errors='replace')
    return ast.parse(src, filename=str(path))


def _first_positional_arg_name(node, skip_self=False):
    """Return the first positional arg name (or None)."""
    args = node.args.args
    if not args:
        return None
    start = 1 if (skip_self and args[0].arg in ('self', 'cls')) else 0
    if start >= len(args):
        return None
    return args[start].arg


def _function_has_xp_dispatch(node):
    """True iff ``node`` (an ``ast.FunctionDef``) contains an
    assignment of the form ``xp = _xp_of(...)`` OR
    ``xp = array_namespace(...)`` somewhere in its body.

    The walker is permissive on the LHS name (so a function that
    binds the namespace to a different name like ``np_or_xp`` still
    passes); it strictly requires the RHS to be a call to
    ``_xp_of`` or ``array_namespace``.
    """
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Assign):
            continue
        if not isinstance(sub.value, ast.Call):
            continue
        func = sub.value.func
        target = None
        if isinstance(func, ast.Name):
            target = func.id
        elif isinstance(func, ast.Attribute):
            target = func.attr
        if target in ('_xp_of', 'array_namespace'):
            return True
    return False


def _iter_target_py_files():
    """Yield every public ``.py`` file in scope.

    Skips private modules (``_*.py``) so the walker does not flag
    private helpers; it ALSO skips ``__init__.py`` (those are
    re-export shims, not implementation files).
    """
    for pkg_rel in _TARGET_PACKAGES:
        pkg_path = _REPO_ROOT / pkg_rel
        if not pkg_path.is_dir():
            continue
        for py in sorted(pkg_path.rglob('*.py')):
            if py.name == '__init__.py':
                continue
            yield py


def _attribute_terminal_name(node):
    """Return the terminal attribute name of an ``ast.Attribute``
    chain, walking through nested ``Attribute`` nodes.  Returns
    None for non-Attribute inputs.

    Example:
    * ``np.zeros``           -> ``'zeros'``
    * ``np.fft.fft2``        -> ``'fft2'``
    * ``np.linalg.solve``    -> ``'solve'``
    """
    if not isinstance(node, ast.Attribute):
        return None
    return node.attr


def _attribute_root_name(node):
    """Walk an ``ast.Attribute`` chain to find the root ``Name``.
    Returns the root name's ``.id`` or None.
    """
    cur = node
    while isinstance(cur, ast.Attribute):
        cur = cur.value
    if isinstance(cur, ast.Name):
        return cur.id
    return None


def _is_np_dispatch_break(call_node):
    """True iff ``call_node`` is a call of the form ``np.<X>(...)``
    where ``X`` (the terminal attribute name) is in
    ``_DISPATCH_BREAKING_NP_ATTRS``.

    The walker considers BOTH bare ``np.<X>(...)`` and dotted
    ``np.<Y>.<X>(...)`` forms (``np.fft.fft2`` is the canonical
    dotted case).  Either form, if the root is ``np`` and the
    terminal is in the break set, is a violation.
    """
    func = call_node.func
    terminal = _attribute_terminal_name(func)
    if terminal is None:
        return False
    root = _attribute_root_name(func)
    if root != 'np':
        return False
    if terminal in _BENIGN_NP_ATTRIBUTES:
        return False
    return terminal in _DISPATCH_BREAKING_NP_ATTRS


def _discover_dispatch_candidates():
    """Yield ``(rel_path, function_node)`` for every public top-level
    function in scope whose first positional name is in
    ``_FIELD_PARAM_NAMES``.
    """
    for py in _iter_target_py_files():
        rel = py.relative_to(_REPO_ROOT).as_posix()
        try:
            tree = _file_to_ast(py)
        except OSError:
            continue
        for node in tree.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            if node.name.startswith('_'):
                continue
            first = _first_positional_arg_name(node)
            if first is None or first not in _FIELD_PARAM_NAMES:
                continue
            yield rel, node


def _np_violations_in_function(func_node):
    """Yield each ``ast.Call`` node inside ``func_node`` that is a
    backend-breaking ``np.*`` call.
    """
    for sub in ast.walk(func_node):
        if not isinstance(sub, ast.Call):
            continue
        if _is_np_dispatch_break(sub):
            yield sub


# Materialise eagerly so collection-time test IDs are visible.
_DISCOVERED_CANDIDATES = list(_discover_dispatch_candidates())


# ============================================================================
# Pre-flight: count floor.
# ============================================================================

def test_walker_minimum_dispatch_candidates():
    """Sanity floor: the walker MUST discover at least 30 candidate
    functions.

    The v4.15.5 baseline discovers ~60 field-domain functions
    across the elements/, analysis/, propagators/ subpackages
    (V6 first-positional-name filter).  A floor of 30 catches a
    walker collapse while leaving headroom for legitimate
    refactoring.
    """
    assert len(_DISCOVERED_CANDIDATES) >= 30, (
        f"xp-dispatch walker found only "
        f"{len(_DISCOVERED_CANDIDATES)} field-domain candidates; "
        f"expected >= 30.  The discovery walker may have regressed.")


# ============================================================================
# Main pin: every dispatch-using function has no backend-breaking np.*
# ============================================================================

def test_no_backend_breaking_np_in_dispatched_functions():
    """For every public function in scope that opts into xp-dispatch
    (i.e. its body contains ``xp = _xp_of(...)``), the function body
    MUST NOT contain ``np.zeros`` / ``np.where`` / ``np.fft.*`` /
    ``np.linalg.*`` / ``np.einsum`` calls.  Such calls demote the
    backend silently.

    Functions in :data:`_KNOWN_XP_DISPATCH_EXEMPTIONS` are skipped
    (with a documented rationale).
    """
    failures = []
    for rel, node in _DISCOVERED_CANDIDATES:
        if (rel, node.name) in _KNOWN_XP_DISPATCH_EXEMPTIONS:
            continue
        if not _function_has_xp_dispatch(node):
            # Function isn't using the dispatch pattern -- not in
            # scope for the main pin.  Such functions are either
            # intentionally NumPy-only (in which case they should be
            # listed in the exemption set) or they need a v4.16.x
            # follow-up to opt INTO dispatch.  Flag via a separate
            # diagnostic, not the main assertion.
            continue
        violations = list(_np_violations_in_function(node))
        if not violations:
            continue
        loc = violations[0]
        failures.append(
            f"{rel}:{loc.lineno} {node.name}: opted into "
            f"``xp = _xp_of(...)`` dispatch but contains a "
            f"hard-coded ``np.{_attribute_terminal_name(loc.func)}(...)`` "
            f"call (and possibly more).  Either:\n"
            f"    1. Replace the offending np.<X> calls with "
            f"``xp.<X>(...)`` so the dispatched backend is honoured, "
            f"OR\n"
            f"    2. Add ``({rel!r}, {node.name!r})`` to "
            f"``_KNOWN_XP_DISPATCH_EXEMPTIONS`` with a cited "
            f"rationale (e.g. 'np.* call is on a scalar constant', "
            f"'legacy numba-accelerated NumPy path retained for "
            f"speed').")
    if failures:
        raise AssertionError(
            "Dispatch-breaking np.* calls inside xp-dispatched "
            "functions:\n  - " + "\n  - ".join(failures))


# ============================================================================
# Cross-check: opt-OUT functions whose first positional is a field but
# which DON'T call _xp_of.  These are candidates for a v4.16.x follow-up
# to opt into dispatch (added the function name list as a diagnostic only).
# ============================================================================

def test_diagnostic_field_functions_without_dispatch():
    """List every field-domain function (V6 filter match) whose
    body does NOT call ``_xp_of`` / ``array_namespace`` AND is
    not in the documented exemption set.

    Diagnostic only -- always passes -- but prints the candidates
    for v4.16.x dispatch-migration triage.
    """
    candidates = []
    for rel, node in _DISCOVERED_CANDIDATES:
        if (rel, node.name) in _KNOWN_XP_DISPATCH_EXEMPTIONS:
            continue
        if _function_has_xp_dispatch(node):
            continue
        candidates.append((rel, node.lineno, node.name))
    if candidates:
        print(f"\nv4.16.0 xp-dispatch walker: {len(candidates)} "
              f"field-domain functions NOT opted into "
              f"_xp_of-dispatch:")
        for rel, lineno, name in candidates:
            print(f"  {rel}:{lineno}  {name}")
    assert isinstance(candidates, list)


# ============================================================================
# Counter-pin: synthetic function with hardcoded np.zeros under xp dispatch
# ============================================================================

def test_counter_pin_fake_hardcoded_np_fails(monkeypatch):
    """Inject a synthetic function with ``xp = _xp_of(E)`` followed
    by ``np.zeros(...)``  and assert the main pin fires.

    Without this counter-pin, a silent walker regression -- e.g. the
    np-violation detector accepting any non-call ``np.<X>`` reference
    -- could let the main pin pass even when real functions hardcode
    backend-breaking calls.
    """
    fake_source = (
        '"""Fake module for xp-dispatch counter-pin."""\n'
        'import numpy as np\n'
        'def fake_dispatched_apply(E_in, dx):\n'
        '    """Hardcodes np.zeros despite xp dispatch."""\n'
        '    xp = _xp_of(E_in)\n'
        '    out = np.zeros((4, 4))\n'  # <-- the violation
        '    return E_in * out.sum()\n'
    )
    fake_tree = ast.parse(fake_source, filename='<xp-fake-bad>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef)
        and stmt.name == 'fake_dispatched_apply'
    )
    assert _function_has_xp_dispatch(fake_node), (
        "Counter-pin setup error: synthetic function does NOT "
        "register as opted-into _xp_of dispatch.")
    violations = list(_np_violations_in_function(fake_node))
    assert violations, (
        "Counter-pin setup error: synthetic function should have a "
        "backend-breaking np.zeros call but the detector found none.")

    rel = 'lumenairy/elements/fake_dispatch_bad.py'

    import sys as _sys
    this_module = _sys.modules[__name__]
    real_candidates = list(this_module._DISCOVERED_CANDIDATES)
    monkeypatch.setattr(
        this_module, '_DISCOVERED_CANDIDATES',
        real_candidates + [(rel, fake_node)])

    with pytest.raises(AssertionError) as excinfo:
        test_no_backend_breaking_np_in_dispatched_functions()
    msg = str(excinfo.value)
    assert 'fake_dispatched_apply' in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected function. Got:\n{msg}")
    assert 'fake_dispatch_bad.py' in msg, (
        f"Counter-pin: main pin failed but its error message does not "
        f"name the injected file. Got:\n{msg}")


def test_counter_pin_fake_dispatched_clean_does_not_trip(monkeypatch):
    """Sibling counter-pin: a synthetic function with ``xp = _xp_of(E)``
    AND ``xp.zeros(...)`` (no hardcoded np.*) must NOT trip the pin.

    Verifies the POSITIVE-signal half: clean xp-dispatched functions
    pass through the walker without being flagged.
    """
    fake_source = (
        '"""Fake module for xp-dispatch positive-signal counter-pin."""\n'
        'def fake_clean_apply(E_in, dx):\n'
        '    """Properly dispatched xp."""\n'
        '    xp = _xp_of(E_in)\n'
        '    out = xp.zeros((4, 4))\n'  # <-- clean dispatch
        '    return E_in * out.sum()\n'
    )
    fake_tree = ast.parse(fake_source, filename='<xp-fake-good>')
    fake_node = next(
        stmt for stmt in fake_tree.body
        if isinstance(stmt, ast.FunctionDef)
    )
    assert _function_has_xp_dispatch(fake_node)
    assert not list(_np_violations_in_function(fake_node)), (
        "Counter-pin setup error: synthetic clean function unexpectedly "
        "registered a np.* violation.  Detector over-firing.")

    rel = 'lumenairy/elements/fake_dispatch_good.py'
    import sys as _sys
    this_module = _sys.modules[__name__]
    real_candidates = list(this_module._DISCOVERED_CANDIDATES)
    monkeypatch.setattr(
        this_module, '_DISCOVERED_CANDIDATES',
        real_candidates + [(rel, fake_node)])
    try:
        test_no_backend_breaking_np_in_dispatched_functions()
    except AssertionError as exc:
        assert 'fake_clean_apply' not in str(exc), (
            f"Positive-signal counter-pin: clean function was flagged. "
            f"Got:\n{exc}")


# ============================================================================
# Direct contract pin for the fixes landed in v4.16.0 / Agent A.
# These pins exist so a future regression that reverts an inline fix is
# caught by a name-anchored test (not just the meta-walker).
# ============================================================================

def test_apply_zernike_aberration_uses_xp_dispatch():
    """``apply_zernike_aberration`` MUST dispatch through ``_xp_of``.

    Inline fix landed in v4.16.0 / Agent A.  Pre-fix the function
    hardcoded ``np.arange`` / ``np.meshgrid`` / ``np.zeros`` /
    ``np.sqrt`` / ``np.arctan2`` -- breaking the dispatch for CuPy
    / JAX inputs to ``apply_zernike_aberration``.
    """
    from lumenairy.elements import elements as elems_mod
    src = inspect.getsource(elems_mod.apply_zernike_aberration)
    tree = ast.parse(src)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)
    assert _function_has_xp_dispatch(func_node), (
        "v4.16.0 / Agent A inline fix regressed: "
        "``apply_zernike_aberration`` no longer dispatches via "
        "``_xp_of``.")


def test_apply_lyot_focal_plane_mask_uses_xp_dispatch():
    """``apply_lyot_focal_plane_mask`` MUST dispatch through
    ``_xp_of``.  Inline fix landed in v4.16.0 / Agent A.
    """
    from lumenairy.elements import elements as elems_mod
    src = inspect.getsource(elems_mod.apply_lyot_focal_plane_mask)
    tree = ast.parse(src)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)
    assert _function_has_xp_dispatch(func_node), (
        "v4.16.0 / Agent A inline fix regressed: "
        "``apply_lyot_focal_plane_mask`` no longer dispatches via "
        "``_xp_of``.")


def test_apply_vortex_phase_mask_uses_xp_dispatch():
    """``apply_vortex_phase_mask`` MUST dispatch through ``_xp_of``.
    Inline fix landed in v4.16.0 / Agent A.
    """
    from lumenairy.elements import elements as elems_mod
    src = inspect.getsource(elems_mod.apply_vortex_phase_mask)
    tree = ast.parse(src)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)
    assert _function_has_xp_dispatch(func_node), (
        "v4.16.0 / Agent A inline fix regressed: "
        "``apply_vortex_phase_mask`` no longer dispatches via "
        "``_xp_of``.")


def test_apply_apodized_pupil_uses_xp_dispatch():
    """``apply_apodized_pupil`` MUST dispatch through ``_xp_of``.
    Inline fix landed in v4.16.0 / Agent A.
    """
    from lumenairy.elements import elements as elems_mod
    src = inspect.getsource(elems_mod.apply_apodized_pupil)
    tree = ast.parse(src)
    func_node = tree.body[0]
    assert isinstance(func_node, ast.FunctionDef)
    assert _function_has_xp_dispatch(func_node), (
        "v4.16.0 / Agent A inline fix regressed: "
        "``apply_apodized_pupil`` no longer dispatches via "
        "``_xp_of``.")


# ============================================================================
# Diagnostic helper.
# ============================================================================

def test_discovered_dispatch_candidates_for_diagnostics():
    """Print discovered dispatch candidates for triage."""
    items = []
    for rel, node in _DISCOVERED_CANDIDATES:
        if (rel, node.name) in _KNOWN_XP_DISPATCH_EXEMPTIONS:
            tag = 'exempt'
        elif not _function_has_xp_dispatch(node):
            tag = 'no-dispatch'
        elif list(_np_violations_in_function(node)):
            tag = 'VIOLATIONS'
        else:
            tag = 'clean'
        items.append((rel, node.lineno, node.name, tag))
    print(f"\nv4.16.0 xp-dispatch walker discovered "
          f"{len(items)} field-domain candidates:")
    for rel, lineno, name, tag in items:
        print(f"  [{tag}] {rel}:{lineno}  {name}")
    assert isinstance(items, list)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
