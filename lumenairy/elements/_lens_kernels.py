"""Lens-family LEAF kernels: the grid-versus-aperture bookkeeping every entry
point in the family runs before it touches a field.

WHY THIS MODULE EXISTS.  ``elements/lenses.py`` is two things at once: a body of
shared leaf helpers, and the FACADE that re-exports the whole lens family
(``_lens_real``, ``_lens_traced``, ``lenses_maslov``, ...).  Because the facade
half imports those modules at module scope, any family module that reaches back
into ``lenses`` for a leaf helper closes a module-level import 2-cycle -- and a
2-cycle means the family's import order is load-bearing, so a new module-level
statement in the wrong half can turn a working import into a partially
initialised one.  Audit ``AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11`` finding H6 /
WP-A16 addendum 10 counted four such cycles and enumerated what each back-edge
carries.

This module is the leaf those back-edges can point at instead.  It imports
``numpy`` and ``warnings`` and nothing from ``lumenairy``, so it can never
participate in a cycle.  ``lenses`` re-exports every name here, so
``lumenairy.elements.lenses.check_grid_vs_apertures`` and every existing
``from .lenses import _warn_if_aperture_exceeds_grid`` keep resolving.

WHAT IS HERE.  The prescription-to-semi-diameter census, the public grid check,
the grid recommendation and the warning the entry points emit.  Nothing in it
is physics: it reads a prescription dict and compares lengths.

WHAT ARRIVED WITH WP-B11c, and why.  ``surface_sag_general`` and
``surface_sag_biconic`` -- the whole of the ``_lens_real -> lenses`` back-edge
-- now live here, which closes that module-level import 2-cycle.  They could
not travel alone: they read the optional-backend plumbing (the lazy CuPy alias
``cp``, the numba gate and its compiled-kernel cache) from module scope, so
that plumbing came with them.  ``lenses_maslov``'s four remaining names are the
one back-edge left; ``docs/lens_configuration.md`` section "Module layout"
carries it.

THE LEAF PROPERTY, RESTATED.  This module no longer imports NOTHING from
``lumenairy`` -- it imports ``lumenairy.backend._optional``, the one place the
library probes and lazily loads CuPy and numba (WP-A16; before it there were
five hand-copied pairs, and re-inlining a sixth to keep the old wording would
undo that).  The import cannot make a cycle: ``_optional`` is itself a
stdlib-only leaf, and the property this module actually needs is the stronger,
transitive one -- nothing reachable from here, at any depth, imports
``lumenairy.elements``.  ``test_audit2609_b11_hygiene.py`` walks that closure.

LIVE STATE, AND WHERE TO PATCH IT.  ``cp``, ``_ne``, ``_numba``, ``_njit`` and
``_prange`` are populated on FIRST USE, and ``_NUMBA_AVAILABLE`` /
``NUMEXPR_AVAILABLE`` are gates the test suite flips to reach the pure-NumPy
arm on a box that HAS the accelerator.  All of them are read at CALL time out
of THIS module's globals.  ``lenses`` therefore cannot re-export them by value
-- a snapshot taken at import time would bind ``None`` for ever and would
swallow the gate flip -- so it forwards them, in BOTH directions (see
``lenses._LIVE_FORWARD_NAMES``): ``lenses._NUMBA_AVAILABLE = False`` still
reaches the kernel below, and ``lenses.cp`` is still a live view.
"""
from __future__ import annotations

import importlib.util as _importlib_util
import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Optional CuPy / numba backends (lazy).  The availability probe, the first-use
# import and the isinstance test live in ONE place for the whole library
# (``backend/_optional.py``; audit 2026-09-11 TESTS-ARCH P2-9).  It is a
# stdlib-only leaf, so this edge cannot close a cycle -- see the module
# docstring.
from ..backend._optional import (
    CUPY_AVAILABLE,
    NUMBA_AVAILABLE as _OPTIONAL_NUMBA_AVAILABLE,
    ensure_cupy as _ensure_cupy,
    is_cupy_array as _optional_is_cupy_array,
    numba_handles as _optional_numba_handles,
)

#: Absolute path of the ``lumenairy`` package directory, with a trailing
#: separator so a sibling directory whose name merely starts with the same
#: characters cannot match.  This module is ``lumenairy/elements/_lens_kernels``,
#: so the package root is two levels up.
_PACKAGE_ROOT = os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))) + os.sep


def caller_stacklevel(*, _root: str = _PACKAGE_ROOT, _limit: int = 64) -> int:
    """The ``stacklevel`` that makes a ``warnings.warn`` in the CALLING frame
    name the nearest frame outside ``lumenairy``.

    Why this is computed rather than written down.  ``stacklevel`` counts
    frames, so a literal is only right for one call path: a helper that grew a
    wrapper, an entry point that re-enters itself, or a warning raised from a
    nested library call all move the user's frame without changing the source
    line.  Both happen here.  The configuration objects (WP-A16) made every
    entry point re-enter ITSELF once when a config is passed --
    ``return apply_real_lens(E_in, **resolve(...))`` -- so on a configured call
    every hard-coded level in the body was one frame short and named the
    library's own re-entry line; the split of ``apply_real_lens`` into a public
    wrapper and ``_apply_real_lens_impl`` did the same thing to the plain path
    one release earlier.

    Walking the stack is the same rule 3.12's ``skip_file_prefixes=`` applies,
    written out so it also holds on the 3.10 and 3.11 this package supports.
    It runs only on a warning path, which is never hot.

    Returns the depth of the first frame whose file is not under the package
    directory, counted the way ``warnings.warn`` counts (``1`` = the frame that
    calls ``warn``).  When the whole stack is inside the package -- a lumenairy
    script, or a warning raised during import -- it returns the outermost
    in-package depth, which is the closest thing to "the caller" that exists.

    ``_root`` and ``_limit`` are private test seams: the package directory to
    treat as library code, and the number of frames to walk before giving up.
    """
    frame = sys._getframe(1)
    depth = 0
    last_inside = 1
    while frame is not None and depth < _limit:
        depth += 1
        name = frame.f_code.co_filename
        if not os.path.abspath(name).startswith(_root):
            return depth
        last_inside = depth
        frame = frame.f_back
    return last_inside


def _collect_semi_diameters(prescription):
    """Return [(label, semi_diameter_m)] for every surface in the
    prescription that has a ``semi_diameter`` set.

    Looks at both ``prescription['elements']`` (Zemax-loaded path,
    where ``semi_diameter`` is the per-surface CLAP) and
    ``prescription['surfaces']`` (builder-style).  The top-level
    ``aperture_diameter`` (system-wide clear aperture) is also
    included if present.  Surfaces without a finite ``semi_diameter``
    are skipped.
    """
    out = []
    seen_labels = set()
    elements = prescription.get('elements')
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            sd = elem.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            surf_num = elem.get('surf_num', '?')
            comment = (elem.get('comment') or elem.get('name') or '').strip()
            label = f"surf {surf_num}"
            if comment:
                label = f"{label} '{comment}'"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    surfaces = prescription.get('surfaces')
    if isinstance(surfaces, list):
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, dict):
                continue
            sd = surf.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            label = f"surfaces[{i}]"
            if label not in seen_labels:
                out.append((label, float(sd)))
                seen_labels.add(label)
    ap = prescription.get('aperture_diameter')
    if ap is not None and np.isfinite(float(ap)):
        out.append(('system aperture_diameter', 0.5 * float(ap)))
    return out


def check_grid_vs_apertures(
    prescription: Dict[str, Any],
    N: int,
    dx: float,
    *,
    safety_factor: float = 1.0,
) -> List[Tuple[str, float, float, float]]:
    """Identify every prescription surface whose semi-aperture exceeds
    the simulation grid's half-extent.

    Use this as a pre-flight check before a full ASM-through-lens run.
    A surface whose ``semi_diameter`` is larger than ``safety_factor *
    N * dx / 2`` will silently truncate the field's outer rim during
    propagation, dropping any energy the real hardware would have
    transmitted past the grid edge.  This is a sim-infrastructure
    artifact, not a physical clipping by the lens itself, and will
    show up in centroid measurements as a uniform inward bias.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict (loaded or builder-built).
    N : int
        Simulation grid size (assumed square).
    dx : float
        Grid pitch [m].
    safety_factor : float, optional
        Margin between the grid semi and the largest surface
        semi-aperture.  Pass 0.95 to flag surfaces whose semi exceeds
        95% of the grid half-extent (recommended for clean Gaussian
        wing containment); pass 1.0 (default) to flag only surfaces
        that exceed the grid outright.

    Returns
    -------
    issues : list of (label, semi_aperture_m, grid_semi_m, gap_m)
        One tuple per offending surface.  Empty if every aperture fits.

    See Also
    --------
    apply_real_lens, apply_real_lens_traced, apply_real_lens_maslov
        These functions automatically run this check on entry and emit
        a UserWarning if any aperture exceeds the grid.
    """
    if N <= 0 or dx <= 0:
        raise ValueError(f"N and dx must be positive, got N={N}, dx={dx}")
    grid_semi = 0.5 * float(N) * float(dx)
    threshold = float(safety_factor) * grid_semi
    issues = []
    for label, sd in _collect_semi_diameters(prescription):
        if sd > threshold:
            issues.append((label, sd, grid_semi, sd - grid_semi))
    return issues


def recommend_grid_for_prescription(
    prescription: Dict[str, Any],
    wavelength: float,
    *,
    source_waist: Optional[float] = None,
    doe_orders_max: Optional[Tuple[float, float]] = None,
    doe_period: Optional[Union[float, Tuple[float, float]]] = None,
    doe_to_destination_distance: Optional[float] = None,
    margin_ratio: float = 1.05,
    samples_per_wavelength: float = 4.0,
    samples_per_source_waist: float = 6.0,
    round_n_to_power_of_two: bool = True,
) -> Dict[str, Any]:
    """Recommend simulation grid parameters ``(N, dx)`` for an ASM run
    through a sequential prescription.

    The recommendation answers two coupled questions before a long
    simulation run:

    * **How wide does the grid need to be?**  Determined by the largest
      ``semi_diameter`` in the prescription, optionally extended for
      DOE diffraction-order spread (corner-order chief rays at the
      destination plane sit at ``m * lambda * z_propagation / period``
      off-axis and must fit inside ``N * dx / 2``).

    * **How fine does the pixel pitch need to be?**  Bounded above by
      the wavelength's effective Nyquist (``lambda /
      samples_per_wavelength``) and, if a source size is supplied, by
      the source-waist resolution (``source_waist /
      samples_per_source_waist``).

    The two requirements together pin ``N`` and ``dx`` (with ``N``
    rounded up to the nearest power of two for FFT efficiency by
    default) such that ``N * dx / 2 >= margin_ratio * required_extent``
    and ``dx <= min(dx_constraints)``.

    Parameters
    ----------
    prescription : dict
        lumenairy prescription dict.  Must include either ``elements``
        with ``semi_diameter`` fields or a top-level ``aperture_diameter``;
        without one of those the function raises ``ValueError``.
    wavelength : float
        Vacuum wavelength [m].  Sets the absolute Nyquist bound on
        ``dx``.
    source_waist : float, optional
        Source 1/e Gaussian waist [m].  When supplied, ``dx`` is also
        bounded above by ``source_waist / samples_per_source_waist`` so
        the source itself is well-resolved on the grid.
    doe_orders_max : (int, int) or (float, float), optional
        Maximum diffraction-order index along (x, y) the simulation
        will need to retain (e.g. ``(5.5, 5.5)`` for a 12 x 12 Dammann
        splitter's diagonal corner).  Must be supplied together with
        ``doe_period`` and ``doe_to_destination_distance`` for the
        DOE-corner-order spread to be added to the required grid
        extent.
    doe_period : float or (float, float), optional
        Grating period [m].  Scalar = isotropic; tuple = (x, y).
    doe_to_destination_distance : float, optional
        Free-space propagation distance from the DOE plane to the
        destination plane that the recommendation is sizing for [m].
    margin_ratio : float, optional
        Multiplicative safety margin on the required grid semi
        (default 1.05 = 5%).
    samples_per_wavelength : float, optional
        Minimum samples per wavelength along each grid axis (default
        4.0).  Nyquist is 2.0; 4.0 leaves comfortable headroom for the
        ASM kernel's exp(i*k*z*sqrt(...)) phase ramp.
    samples_per_source_waist : float, optional
        Minimum samples across one source waist (default 6.0).  Below
        ~4 the Gaussian source is poorly resolved.
    round_n_to_power_of_two : bool, optional
        If True (default), round ``N`` up to the nearest power of two.
        FFT-based propagators run several times faster on power-of-two
        grids than on arbitrary sizes.

    Returns
    -------
    dict
        Recommendation with at least these keys:

          ``N`` -- recommended grid size (int)
          ``dx`` -- recommended pixel pitch [m]
          ``grid_semi_m`` -- ``N * dx / 2``
          ``required_extent_m`` -- before margin
          ``limiting_aperture_m`` -- the surface that set the grid extent
          ``limiting_aperture_label`` -- human-readable label of that surface
          ``doe_extra_extent_m`` -- additional extent demanded by the DOE
          ``dx_constraints`` -- dict of every dx upper bound considered
          ``dx_limiting_constraint`` -- name of the binding constraint
          ``samples_per_wavelength`` -- effective post-rounding value

    Raises
    ------
    ValueError
        If the prescription has no resolvable aperture, or if any of
        ``doe_*`` are partially supplied (must be all-or-none).

    See Also
    --------
    check_grid_vs_apertures
        The post-flight companion that flags whether a chosen ``(N,
        dx)`` actually contains every prescription aperture.
    """
    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0, got {wavelength}")
    if margin_ratio < 1.0:
        raise ValueError(
            f"margin_ratio must be >= 1.0 (got {margin_ratio}); use 1.0 "
            "for the bare-minimum grid that just contains every aperture")

    # DOE arguments are all-or-none
    doe_args = (doe_orders_max, doe_period, doe_to_destination_distance)
    if any(a is not None for a in doe_args) and not all(
            a is not None for a in doe_args):
        raise ValueError(
            "doe_orders_max, doe_period, and doe_to_destination_distance "
            "must all be supplied together (or all be None)")

    # 1) Find the largest aperture in the prescription
    max_semi = 0.0
    aperture_label = ''
    elements = prescription.get('elements')
    if isinstance(elements, list):
        for elem in elements:
            if not isinstance(elem, dict):
                continue
            sd = elem.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            sd = float(sd)
            if sd > max_semi:
                max_semi = sd
                lbl_parts = []
                if elem.get('comment'):
                    lbl_parts.append(str(elem['comment']))
                if elem.get('surf_num') is not None:
                    lbl_parts.append(f"surf {elem['surf_num']}")
                aperture_label = (
                    ' / '.join(lbl_parts) if lbl_parts
                    else 'unknown element')
    surfaces = prescription.get('surfaces')
    if isinstance(surfaces, list):
        for i, surf in enumerate(surfaces):
            if not isinstance(surf, dict):
                continue
            sd = surf.get('semi_diameter')
            if sd is None or not np.isfinite(float(sd)):
                continue
            sd = float(sd)
            if sd > max_semi:
                max_semi = sd
                aperture_label = f"surfaces[{i}]"
    sys_ap = prescription.get('aperture_diameter')
    if sys_ap is not None and np.isfinite(float(sys_ap)):
        if 0.5 * float(sys_ap) > max_semi:
            max_semi = 0.5 * float(sys_ap)
            aperture_label = 'system aperture_diameter'

    if max_semi <= 0.0:
        raise ValueError(
            "prescription has no resolvable aperture: pass a prescription "
            "whose elements / surfaces carry 'semi_diameter' or whose "
            "top-level 'aperture_diameter' is finite")

    # 2) DOE-order extent extension (if supplied)
    doe_extra = 0.0
    if doe_orders_max is not None:
        m_x, m_y = doe_orders_max
        if isinstance(doe_period, (tuple, list)):
            period_x, period_y = (float(doe_period[0]),
                                   float(doe_period[1]))
        else:
            period_x = period_y = float(doe_period)
        if period_x <= 0 or period_y <= 0:
            raise ValueError(f"doe_period must be > 0, got {doe_period}")
        z_doe = float(doe_to_destination_distance)
        # Diagonal corner order's chief-ray displacement at the
        # destination plane (paraxial direction-cosine approximation).
        theta_x = float(m_x) * wavelength / period_x
        theta_y = float(m_y) * wavelength / period_y
        doe_extra = abs(z_doe) * float(np.hypot(theta_x, theta_y))

    required_extent = max_semi + doe_extra
    required_grid_semi = margin_ratio * required_extent

    # 3) dx upper bounds
    dx_constraints = {
        'wavelength_nyquist': float(wavelength) / float(samples_per_wavelength),
    }
    if source_waist is not None and float(source_waist) > 0:
        dx_constraints['source_waist'] = (
            float(source_waist) / float(samples_per_source_waist))
    dx_max = min(dx_constraints.values())
    dx_limiting = min(dx_constraints, key=dx_constraints.get)

    # 4) Pick (N, dx)
    n_min = int(np.ceil(2.0 * required_grid_semi / dx_max))
    if round_n_to_power_of_two:
        n = int(2 ** int(np.ceil(np.log2(max(n_min, 1)))))
    else:
        n = int(n_min)
    # Recompute dx to land grid_semi exactly at required_grid_semi.
    dx = 2.0 * required_grid_semi / n

    return {
        'N': n,
        'dx': float(dx),
        'grid_semi_m': float(n * dx / 2.0),
        'required_extent_m': float(required_extent),
        'limiting_aperture_m': float(max_semi),
        'limiting_aperture_label': aperture_label,
        'doe_extra_extent_m': float(doe_extra),
        'dx_constraints': dx_constraints,
        'dx_limiting_constraint': dx_limiting,
        'samples_per_wavelength': float(wavelength) / float(dx),
        'margin_ratio': float(margin_ratio),
    }


def _warn_if_aperture_exceeds_grid(prescription, N, dx, *,
                                    source='apply_real_lens',
                                    safety_factor=1.0,
                                    stacklevel=None,
                                    N_y=None, dy=None):
    """Emit a UserWarning if any prescription aperture exceeds the
    simulation grid.  Called at the top of ``apply_real_lens``,
    ``apply_real_lens_traced``, and ``apply_real_lens_maslov``.

    ``N``/``dx`` describe the x axis.  ``N_y``/``dy`` describe the y axis on an
    ANAMORPHIC grid (non-square ``N``, or ``dy != dx``); both default to the x
    values, so a square grid behaves exactly as before.  The check is made
    against the SMALLER of the two semi-extents, because that is the axis that
    truncates first -- passing ``shape[0]`` (i.e. Ny) together with ``dx``, as
    the analytic model used to, describes a semi-extent that exists on neither
    axis.

    ``stacklevel=None`` (the default) computes the level with
    :func:`caller_stacklevel`, so the warning names the first frame outside
    ``lumenairy`` whatever chain of entry points, wrappers and configuration
    re-entries reached it.  An explicit integer is still honoured, for a caller
    that means a specific frame.

    Python's default warning filter dedups by ``(module, lineno)`` of the
    ATTRIBUTED frame, so repeated calls from the same user site only warn once.
    """
    _ny = N if N_y is None else N_y
    _dy = dx if dy is None else dy
    semi_x = 0.5 * N * dx
    semi_y = 0.5 * _ny * _dy
    if semi_y < semi_x:
        N, dx = _ny, _dy
    try:
        issues = check_grid_vs_apertures(
            prescription, N, dx, safety_factor=safety_factor)
    except (KeyError, ValueError, TypeError, AttributeError):
        # Aperture check is best-effort; a malformed prescription
        # shouldn't block the warning path entirely (the
        # propagator's own validators will raise downstream).
        return
    if not issues:
        return
    grid_semi = 0.5 * N * dx
    issues_sorted = sorted(issues, key=lambda r: -r[1])
    biggest_label, biggest_sd, _, biggest_gap = issues_sorted[0]
    body = ", ".join(
        f"{lab}={sd*1e3:.2f}mm"
        for lab, sd, _, _ in issues_sorted[:5]
    )
    if len(issues_sorted) > 5:
        body = body + f", ... (+{len(issues_sorted)-5} more)"
    msg = (
        f"{source}: {len(issues)} prescription aperture(s) exceed the "
        f"simulation grid (N={N}, dx={dx*1e6:.3f} um, "
        f"semi={grid_semi*1e3:.3f} mm). "
        f"Largest is {biggest_label} with semi_diameter="
        f"{biggest_sd*1e3:.3f} mm "
        f"({biggest_gap*1e3:+.3f} mm beyond the grid); the field will "
        f"be truncated at the grid edge during propagation, "
        f"silently dropping energy the real lens would have "
        f"transmitted. Consider increasing N or dx so "
        f"N*dx/2 >= max(semi_diameter). "
        f"Affected surfaces: {body}."
    )
    if stacklevel is None:
        # Measured from THIS frame, which is the frame that calls ``warn``, so
        # the two count the same thing and no offset is needed.
        stacklevel = caller_stacklevel()
    warnings.warn(msg, UserWarning, stacklevel=stacklevel)


# ===========================================================================
# WP-B11c: the payload of the back-edges that used to point at ``lenses``
# ===========================================================================
# Everything below arrived VERBATIM from ``elements/lenses.py``.  It is here
# rather than there for one reason: family modules read it at module scope and
# ``lenses`` imports those modules at module scope, so as long as it lived in
# the facade the family carried an import 2-cycle.  The facade re-exports every
# name, and forwards the live ones both ways.

cp = None  # this module's alias for the cupy module; see _ensure_cupy_loaded


def _ensure_cupy_loaded():
    """Load CuPy on first use; return True iff it is available.

    Keeps this module's ``cp`` alias populated because the GPU branches here
    -- and ``_lens_thin``'s PEP 562 ``cp`` forward, which reads
    ``_lenses_module.cp`` -- resolve the module-level name directly.
    """
    global cp
    if cp is None:
        cp = _ensure_cupy()
    return cp is not None


# Optional fused-expression backend ------------------------------------------
# numexpr evaluates array expressions in chunked, multi-threaded passes
# without materialising full N x N intermediates.  Used by apply_real_lens
# to fuse the ``E * exp(-1j*k0*opd)`` phase-screen multiply, which at
# N=32768 otherwise allocates three 17 GB complex128 temporaries.
NUMEXPR_AVAILABLE = _importlib_util.find_spec('numexpr') is not None
_ne = None  # populated by _ensure_numexpr_loaded() on first use


def _ensure_numexpr_loaded():
    global _ne
    if _ne is None and NUMEXPR_AVAILABLE:
        import numexpr as _n
        _ne = _n
    return _ne is not None

# Optional Numba JIT, LAZILY imported on first kernel use (audit P2-D: the eager
# ``import numba`` cost ~1.8 s of ``import lumenairy`` cold start).  Used by the
# fused polynomial-aspheric loop ``_aspheric_sag_accum_numba`` (3.2.14), which has
# a pure-NumPy fallback -- so numba is pulled in only when a caller actually hits
# that fast path AND numba is installed.  ``find_spec`` checks availability
# WITHOUT importing numba.
# The MODULE-LEVEL ``_NUMBA_AVAILABLE`` is load-bearing and stays a module
# attribute: it is read at CALL time and the test suite monkeypatches it to
# ``False`` to reach the pure-NumPy arm on a box where numba IS installed.  So
# the availability GATE is local while the import is shared
# (``backend/_optional.py``; audit 2026-09-11 TESTS-ARCH P2-9).
_NUMBA_AVAILABLE = _OPTIONAL_NUMBA_AVAILABLE
_numba = None                         # populated by _load_numba() on first use
_njit = None
_prange = None
_NUMBA_KERNELS: dict = {}             # kernel-name -> compiled fn (or None)


def _load_numba():
    """Import numba + njit/prange on first use; cache the handles.  Returns True
    iff numba is importable (False -> callers take the pure-NumPy fallback).

    Honours a monkeypatched module-level ``_NUMBA_AVAILABLE = False`` -- this
    library's spelling for "pretend the accelerator is absent" -- before
    consulting the shared loader."""
    global _numba, _njit, _prange
    if _numba is not None:
        return True
    if not _NUMBA_AVAILABLE:
        return False
    _numba, _njit, _prange = _optional_numba_handles()
    return _numba is not None


# The numexpr scaffold ABOVE (``NUMEXPR_AVAILABLE`` / ``_ne`` /
# ``_ensure_numexpr_loaded``) is NOT dead: ``lenses_maslov`` reads the flag and
# the loader through the ``lenses`` facade, which forwards both LIVE to this
# module, and ``elements/__init__`` + ``lumenairy/__init__`` re-export
# ``NUMEXPR_AVAILABLE`` publicly.


def _is_cupy_array(x):
    """
    Reliable CuPy array check.  ``hasattr(x, 'device')`` used to be a
    duck-type test for a CuPy device array but broke in NumPy 2.x
    (``ndarray`` now exposes ``.device`` via the Array API standard),
    causing every NumPy array to get routed into the CuPy branch.

    ``_lens_thin`` asks ``backend._optional.is_cupy_array`` directly rather
    than delegating here -- it takes the same answer from the same helper,
    without the module-level import back into this file that a delegation
    would need.  The extra short-circuit below is the only difference, and it
    is about call cost, not about the answer.
    """
    if not CUPY_AVAILABLE:
        # Local short-circuit, not a delegation: this is the hot per-call
        # dispatch for the whole thin/spherical/aspheric family and the
        # CuPy-absent answer must stay one global read.
        return False
    if not _optional_is_cupy_array(x):
        return False
    _ensure_cupy_loaded()   # a True answer implies ``cp`` is live -- bind it
    return True



# ---------------------------------------------------------------------------
# Helper: general conic + aspheric surface sag
# ---------------------------------------------------------------------------

def _get_aspheric_sag_accum_numba():
    """Compile (once, on first call) and return the fused aspheric-sag numba
    kernel, or ``None`` if numba is unavailable.  Lazy so ``import lumenairy``
    never pays the numba import / compile cost (audit P2-D)."""
    if "aspheric_sag" in _NUMBA_KERNELS:
        return _NUMBA_KERNELS["aspheric_sag"]
    if not _load_numba():
        _NUMBA_KERNELS["aspheric_sag"] = None
        return None

    @_njit(cache=True, parallel=True, fastmath=True)
    def _aspheric_sag_accum_numba(h_sq, sag, powers, coeffs):
        """In-place accumulate sum_i coeff_i * h_sq**(power_i // 2) onto
        ``sag``.  Single fused pass over h_sq, no temporary arrays.

        ``h_sq`` and ``sag`` must be contiguous float64 arrays of the
        same shape.  ``powers`` is int32, ``coeffs`` is float64; both
        1-D and same length.
        """
        flat_h = h_sq.ravel()
        flat_s = sag.ravel()
        n = flat_h.size
        n_terms = powers.size
        for i in _prange(n):
            v = flat_h[i]
            acc = 0.0
            for j in range(n_terms):
                p = powers[j] // 2
                # h_sq^p via repeated squaring keeps the inner loop
                # branch-free (Numba unrolls small fixed-power loops).
                hp = 1.0
                for _ in range(p):
                    hp *= v
                acc += coeffs[j] * hp
            flat_s[i] += acc

    _NUMBA_KERNELS["aspheric_sag"] = _aspheric_sag_accum_numba
    return _aspheric_sag_accum_numba


def surface_sag_general(
    h_sq: np.ndarray,
    R: float,
    conic: float = 0.0,
    aspheric_coeffs: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Compute surface sag for a general conic + even-aspheric surface.

    This function is used by both the lens and mirror modules, so it is
    exported at module level (no leading underscore).

    Parameters
    ----------
    h_sq : ndarray
        Squared radial distance from the optical axis, x**2 + y**2  [m**2].
    R : float
        Radius of curvature [m].  Use ``float('inf')`` or ``np.inf`` for a
        flat surface.
    conic : float, optional
        Conic constant (default 0 = sphere, -1 = paraboloid, < -1 =
        hyperboloid, -1 < k < 0 = prolate ellipsoid, > 0 = oblate ellipsoid).
    aspheric_coeffs : dict or None, optional
        Even polynomial aspheric coefficients ``{power: coeff}``, e.g.
        ``{4: A4, 6: A6, 8: A8, 10: A10}``.  Each term contributes
        ``coeff * h_sq**(power // 2)`` to the sag.

    Returns
    -------
    sag : ndarray
        Signed surface sag (positive when R > 0).
    """
    # Array-API polymorphic: detect cupy vs numpy from the input.
    # This keeps the helper usable from both the CPU and GPU paths of
    # apply_real_lens without duplicating code.  Arithmetic
    # broadcasting (np.where, np.sqrt on cupy arrays) silently
    # converts to host, so we dispatch explicitly.
    xp = cp if _is_cupy_array(h_sq) else np

    # ``R = 0`` and ``R = nan`` are not surfaces: the conic expression divides
    # by ``R**2`` and then by ``R``, so both returned an all-NaN sag behind
    # anonymous numpy RuntimeWarnings ("divide by zero", "invalid value") that
    # name neither this function nor the offending key -- and a NaN radius
    # propagates silently all the way into the phase screen, where it zeroes
    # the whole field.  ``R = inf`` and ``R = None`` are the two spellings of a
    # FLAT surface and are handled below; anything else non-finite or zero is a
    # malformed prescription and gets the CONVENTIONS SS2 message.
    if R is not None and not np.isinf(R) and not (float(R) != 0.0
                                                  and np.isfinite(R)):
        _what = 'nan' if not np.isfinite(R) else '0'
        raise ValueError(
            f"surface_sag_general: radius R = {_what} is not a surface (the "
            f"conic sag divides by R).  Use R = np.inf or R = None for a FLAT "
            f"surface; a finite non-zero R for a curved one.")

    if R is not None and not np.isinf(R):
        # Conic sag: h^2 / (R * (1 + sqrt(1 - (1+k)*h^2/R^2)))
        # 4.10: outside the conic domain (norm >= 0.9999) the surface
        # is not defined.  A silent 0 sag there produces an
        # which produced an apparently-flat ring at the surface edge
        # for hyperbolic / oblate conics extending past the geometric
        # rim.  Return NaN instead so downstream consumers either mask
        # those pixels (via an aperture mask) or see the failure.
        #
        # Written as one in-place chain through a single scratch grid.  The
        # expression-per-line form allocated a fresh full grid for each of
        # ``norm``, ``denom_arg``, the ``sqrt``, the ``1 + ...``, the ``R * ...``,
        # the division and the final ``where`` -- 5.13 float64 grids at a
        # tracemalloc peak, 4.5x the wall clock of the identical arithmetic
        # written with ``out=`` (248 -> 55 ms at N = 2048), and 22 % of a
        # default three-surface ``apply_real_lens`` call.  Every operation and
        # its ORDER is unchanged, so the result is bit-identical; only the
        # temporaries are gone.  The domain mask stays a separate bool grid
        # (1/8 of a float64 one) because ``norm < 0.9999`` has to be taken
        # BEFORE ``norm`` is overwritten -- and it is taken in that sense and
        # then inverted, rather than as ``>= 0.9999``, so a NaN ``h_sq`` lands
        # on the INVALID side exactly where ``xp.where`` put it.
        sag = xp.multiply(h_sq, (1 + conic))
        sag = xp.divide(sag, R**2, out=sag)
        invalid = sag < 0.9999
        xp.logical_not(invalid, out=invalid)
        xp.subtract(1, sag, out=sag)
        sag[invalid] = 0.01
        xp.sqrt(sag, out=sag)
        xp.add(sag, 1, out=sag)
        xp.multiply(sag, R, out=sag)
        xp.divide(h_sq, sag, out=sag)
        sag[invalid] = xp.nan
        del invalid
    else:
        sag = xp.zeros_like(h_sq)

    if aspheric_coeffs:
        # Reject ODD powers HERE, at the
        # wave-optics sag entry point.  Both branches below evaluate
        # ``h_sq ** (power // 2)`` (the numba kernel's ``powers[j] // 2`` and
        # the NumPy fallback's ``power // 2``), so an odd power silently floors
        # to the NEXT-LOWER EVEN one -- a different surface, returned with no
        # diagnostic.  Measured pre-guard at ``{5: 1e6}``, h = 10 mm, flat base:
        # sag 0.01 m (== the ``{4: 1e6}`` sag, BIT-identical) against the true
        # 1.0e-4 m -- 100x -- with dz/dh 4.0 vs the true 0.05 (80x).  The same
        # ``{5: ...}`` fed through ``apply_real_lens`` returned a field
        # bit-identical to the ``{4: ...}`` lens.  ``Surface`` and the JAX
        # prescription path already reject it via the SAME shared checker; this
        # is the wave-optics path, which never builds a ``Surface``.
        # Import is function-local: ``lumenairy.raytrace.__init__`` pulls in
        # ``raytrace.surface``, which imports THIS module, so a module-level
        # ``from ..raytrace._conic_core import ...`` cycles at import time.
        # (Same deferred-import pattern as ``.._validation`` in _lens_thin.py.)
        from ..raytrace._conic_core import check_even_aspheric_powers
        check_even_aspheric_powers(aspheric_coeffs.keys(),
                                   fn_label='surface_sag_general')
        # 3.2.14: fused single-pass numba kernel when available.
        # Skips the per-term temporary array allocation that the
        # legacy NumPy fallback required (5 aspheric coeffs at N=4096
        # is ~640 MB of transient memory in that path).  CuPy stays
        # on the legacy path because numba targets host arrays.
        _sag_kernel = (_get_aspheric_sag_accum_numba()
                       if xp is np and _NUMBA_AVAILABLE else None)
        if (_sag_kernel is not None
                and h_sq.dtype == np.float64
                and sag.dtype == np.float64):
            powers_arr = np.fromiter(
                (int(p) for p in aspheric_coeffs.keys()), dtype=np.int32)
            coeffs_arr = np.fromiter(
                (float(c) for c in aspheric_coeffs.values()),
                dtype=np.float64)
            # The kernel accumulates through ``sag.ravel()``, which is a VIEW
            # only when ``sag`` is C-contiguous; for an F-ordered or transposed
            # array ``ravel()`` copies, the kernel adds the whole polynomial
            # into that copy and the copy is discarded -- the aspheric term
            # vanished silently and completely (measured 9.41e-6 m = 100 % of
            # the term).  ``sag`` inherits its memory order from ``h_sq``, so
            # any caller passing a non-C-contiguous ``h_sq`` hit it; every
            # in-repo caller happens not to, which is why it survived.  Make
            # the buffer contiguous, accumulate, and copy back when it was not
            # the same object.
            _sag_c = np.ascontiguousarray(sag)
            _sag_kernel(
                np.ascontiguousarray(h_sq), _sag_c, powers_arr, coeffs_arr)
            if _sag_c is not sag:
                sag[...] = _sag_c
            del _sag_c
        else:
            for power, coeff in aspheric_coeffs.items():
                sag = sag + coeff * h_sq ** (power // 2)

    return sag


# Keep the private alias so internal callers (apply_real_lens) can use either
# name without changing semantics.
_surface_sag_general = surface_sag_general


def surface_sag_biconic(
    X: np.ndarray,
    Y: np.ndarray,
    R_x: float,
    R_y: Optional[float] = None,
    conic_x: float = 0.0,
    conic_y: Optional[float] = None,
    aspheric_coeffs: Optional[Dict[int, float]] = None,
    aspheric_coeffs_y: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """Biconic / cylindrical / toroidal surface sag.

    Generalises :func:`surface_sag_general` to surfaces that have
    different curvatures and conics along the x and y axes.  Covers:

    * **Biconic** (SEPARABLE per-axis form): independent R_x, R_y, K_x,
      K_y, where each axis contributes its own conic sag INDEPENDENTLY:

          z(x,y) = C_x*x² / (1 + sqrt(1 - (1+K_x)*C_x²*x²))
                 + C_y*y² / (1 + sqrt(1 - (1+K_y)*C_y²*y²))

      where C_x = 1/R_x, C_y = 1/R_y.

      RT-2 (AUDIT_RAYTRACE_CORE) -- deviation from Zemax "Biconic": this
      is the ``z = z_x(x) + z_y(y)`` SEPARABLE sum, NOT Zemax's biconic,
      which shares a SINGLE square root across both axes:

          z = (C_x*x² + C_y*y²)
              / (1 + sqrt(1 - (1+K_x)*C_x²*x² - (1+K_y)*C_y²*y²))

      The two agree paraxially (and exactly on either axis, y=0 or x=0)
      but diverge in the fourth-order cross-term, so an imported Zemax
      BICONIC surface is approximated at large aperture / off both axes.
      (``_surface_sag_derivatives_xy`` is exactly consistent with the
      SEPARABLE form used here, so the sag and its normals still agree
      internally.)
    * **Cylindrical**: pass ``R_y = inf`` (focusing in x only) or
      ``R_x = inf`` (focusing in y only).
    * **Toroidal** (approx., Zemax "Toroidal"): pass R_x (rotation-axis
      radius) and R_y (cross-section radius).
    * **Rotationally symmetric**: if ``R_y is None`` the function
      reduces to :func:`surface_sag_general` via h² = x² + y².

    Aspheric coefficients may be given per-axis for fully general
    anamorphic surfaces; ``aspheric_coeffs`` (x-axis) and
    ``aspheric_coeffs_y`` (y-axis) are separate dicts of
    ``{power: coeff}`` contributing ``coeff * h² ** (power // 2)``
    along each axis.

    Parameters
    ----------
    X, Y : ndarray
        Surface-local coordinates [m] (after any decenter/tilt).
        ``X`` and ``Y`` must have the same shape; meshgrid indexing is
        up to the caller.
    R_x : float
        Radius of curvature along x-axis [m] (``inf`` = flat in x).
    R_y : float, optional
        Radius of curvature along y-axis [m].  If ``None``, the surface
        is treated as rotationally symmetric with R = R_x (legacy).
    conic_x : float, default 0
        Conic constant along x.
    conic_y : float, optional
        Conic constant along y.  Defaults to ``conic_x`` if not given.
    aspheric_coeffs : dict or None
        Even-aspheric coefficients along x, ``{power: coeff}``.
    aspheric_coeffs_y : dict or None
        Even-aspheric coefficients along y.  If ``None`` and
        ``aspheric_coeffs`` is given, the x coefficients are reused for
        y (isotropic asphere).

    Returns
    -------
    sag : ndarray
        Signed surface sag, same shape as ``X`` / ``Y``.
    """
    # Detect array backend and use cupy ops when X/Y are device arrays.
    xp = cp if _is_cupy_array(X) else np
    X = xp.asarray(X)
    Y = xp.asarray(Y)

    if R_y is None:
        # Reduce to the rotationally-symmetric formula for backward
        # compatibility.  (Its own R-8 guard covers ``aspheric_coeffs``;
        # ``aspheric_coeffs_y`` is unread on this branch, as documented.)
        h_sq = X ** 2 + Y ** 2
        return surface_sag_general(h_sq, R_x, conic_x, aspheric_coeffs)

    if conic_y is None:
        conic_y = conic_x

    # Reject ODD powers on BOTH per-axis
    # coefficient dicts before ``_axis_sag`` floors them via
    # ``h_sq ** (power // 2)``.  Measured pre-guard at ``{3: 1e4}``, h = 10 mm,
    # flat base: sag 1.0 m -- BIT-identical to the ``{2: 1e4}`` sag -- against
    # the true 0.01 m, i.e. 100x, on the x AND the y coefficient set.  Same
    # shared checker (and message) as ``surface_sag_general`` /
    # ``raytrace.conic_sag``; see that site for the function-local-import note.
    from ..raytrace._conic_core import check_even_aspheric_powers
    if aspheric_coeffs:
        check_even_aspheric_powers(aspheric_coeffs.keys(),
                                   fn_label='surface_sag_biconic')
    if aspheric_coeffs_y:
        check_even_aspheric_powers(aspheric_coeffs_y.keys(),
                                   fn_label='surface_sag_biconic (aspheric_coeffs_y)')

    def _axis_sag(h_sq, R, K, asph):
        s = xp.zeros_like(h_sq)
        if R is not None and not np.isinf(R):
            norm = (1 + K) * h_sq / R ** 2
            valid = norm < 0.9999
            denom_arg = xp.where(valid, 1 - norm, 0.01)
            # Outside the conic domain
            # (norm >= 0.9999) the surface is not defined.  Return NaN
            # (not a silent 0.0 flat ring) so callers detect 'no real
            # surface', matching surface_sag_general.
            s = xp.where(
                valid,
                h_sq / (R * (1 + xp.sqrt(denom_arg))),
                xp.nan,
            )
        if asph:
            for power, coeff in asph.items():
                s = s + coeff * h_sq ** (power // 2)
        return s

    sag_x = _axis_sag(X ** 2, R_x, conic_x, aspheric_coeffs)
    sag_y = _axis_sag(Y ** 2, R_y, conic_y,
                      aspheric_coeffs_y if aspheric_coeffs_y is not None
                      else aspheric_coeffs)
    return sag_x + sag_y
