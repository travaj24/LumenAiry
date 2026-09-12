"""Shared input-validation helpers used at the boundary of every
public propagator and lens entry point.

The :func:`_check_2d_scalar_field` helper consolidates the
``PartialCoherenceMCF`` and ``ndim != 2`` defensive guards that
v4.15.1-v4.15.2 added inline at each entry point.  Centralising
the check eliminates ~24 lines of boilerplate per site (was
~240 LOC of copy/paste across the 10 v4.15.2 guarded sites
plus 9 new sibling sites added in v4.15.3) and prevents the
recurring 'fix N, miss N+1' audit-meta-pattern by making the
guard a single edit point.

Pair this with the
``tests/unit/test_v4_15_3_dispatcher_pin_2d_scalar_field.py``
meta-pin that walks ``propagators/`` and ``elements/`` for
``def apply_*`` / ``def *_propagate*`` and asserts
``_check_2d_scalar_field`` appears as the first executable
statement of each function body.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Module-level import -- v4.15.3 used a lazy import inside the function
# body, citing a (hypothetical) circular dependency with
# ``sources.core``.  ``sources.core`` does NOT actually import from
# ``_validation`` (it only imports from ``_deprecation``), so the
# circular-dep claim was inaccurate and the lazy import wasted ~1 us
# per propagator call (1-10 ms per merit eval in optimisation loops).
# v4.15.4 hoists the import to module scope.
#
# If a future refactor introduces a ``sources.core`` -> ``_validation``
# import path, switch to a ``TYPE_CHECKING`` import + duck-typed
# predicate (e.g. ``hasattr(E, 'mcf_matrix')`` or
# ``type(E).__name__ == 'PartialCoherenceMCF'``) instead of restoring
# the lazy import -- the duck-typed predicate is free of import-order
# constraints AND keeps the helper hot-loop cheap.
# Relative, matching the rest of the package -- an
# absolute ``from lumenairy...`` inside the package re-enters the top-level
# ``lumenairy/__init__`` by name and only works because it is already in
# sys.modules by the time this module is imported.
from .sources.core import PartialCoherenceMCF as _MCF

# The closed vocabulary for ``input_kind``.
#
# v4.15.5 introduced ``input_kind`` as a free-form string interpolated
# straight into the rejection message and documented three
# "conventional values" in prose.  Prose is not a contract: a typo
# (``'feild'``), a plural (``'pupils'``), or a value borrowed from a
# sibling knob (``'intensity'``) produced a *silently* misleading
# error message -- the guard still fired, but told the user to pass a
# 2-D complex "feild".  The A-9 rollout (2 -> 27 wired call sites)
# multiplies the number of places that mistake can be made by ~13x,
# so the vocabulary is now closed and checked.
#
# Membership is a frozenset lookup (~40 ns), which matters because
# this helper runs at the entry point of every propagator / lens call
# and is therefore on the merit-evaluation hot path (see the
# module-level import note above).  Adding a value here is the
# supported way to extend the vocabulary; passing an unlisted one is
# a library bug, not a user error, so it fails loudly.
_INPUT_KINDS = frozenset({'field', 'psf', 'pupil'})

# The numeric ``dtype.kind`` letters this guard accepts.
# The guard's own message has always said "expected 2-D complex <kind>" while
# it enforced only ``ndim == 2``; complex ('c') is the intended input and real
# ('f') / integer ('i', 'u') / bool ('b') are legitimately used as amplitude
# masks (every kernel promotes them to complex).  Everything else -- object
# ('O'), string ('U'/'S'), datetime, void, structured -- is not a field: an
# object-dtype array reaches the kernels and takes NumPy's Python-object slow
# path, producing a plausible-looking result from whatever ``__mul__`` the
# elements happen to define.  ``dtype.kind`` is a one-character attribute
# lookup (~60 ns), which matters because this helper runs at the entry point
# of every propagator / lens call; ``np.issubdtype`` costs ~20x that.
_FIELD_DTYPE_KINDS = frozenset('cfiub')


def _check_2d_scalar_field(
    E: Any,
    fn_name: str,
    *,
    input_kind: str = 'field',
) -> None:
    """Reject ``PartialCoherenceMCF`` and non-2-D inputs at the
    entry point of any coherent-field propagator or lens kernel.

    Parameters
    ----------
    E : array-like
        Candidate input field.  Expected to be a 2-D NumPy/CuPy
        complex array of shape ``(Ny, Nx)``.
    fn_name : str
        Name of the calling function -- used in error messages so
        the user knows which entry point rejected their input.
    input_kind : str, default ``'field'``
        Semantic name of the rejected argument -- used in the error
        message wording so callers passing a non-conforming
        ``pupil`` (``richards_wolf_focus``, ``debye_wolf_psf``) or
        ``psf`` (``compute_otf`` / ``compute_mtf``) see
        "expected 2-D complex pupil" / "psf" instead of the literal
        "field" string.  The value must be a member of the closed
        :data:`_INPUT_KINDS` vocabulary (``'field'`` / ``'psf'`` /
        ``'pupil'``) -- see the Raises section -- and is declared
        explicitly at every wired
        call site rather than defaulted.

    Raises
    ------
    TypeError
        If ``E`` is a :class:`lumenairy.PartialCoherenceMCF`
        instance.  MCF-aware downstream propagators are an
        architectural change tracked in ROADMAP v5.0+ (the prior
        "v4.16+" deferral was retired in v4.16.1 once the actual
        v4.16.0 release shipped without MCF-aware propagators).
        For partial-coherence propagation today, the error message
        points the user at the v4.16.1 ``propagate_ensemble`` helper
        which iterates a Schell-family ensemble through any coherent
        propagator and returns ``< |E_k|^2 >_k``.
    ValueError
        If ``input_kind`` is not a member of :data:`_INPUT_KINDS`.
        Checked BEFORE the input itself: an unlisted kind is a
        library bug (the string is supplied by lumenairy's own call
        sites, never by the end user), and reporting the user's
        array shape while the guard's own message wording is broken
        would bury the real defect.  The message names the offending
        value and the sorted allowed set (house rule for enum-valued
        knobs, cf. ``normalize`` in ``compute_psf`` and
        ``on_noncollimated`` in ``apply_real_lens_traced``).
    ValueError
        If ``E.ndim != 2``.  Common cause: the user passed a
        v4.15.1+ Schell-family ensemble (shape
        ``(n_realizations, Ny, Nx)``) directly to a coherent
        propagator.  The error message points at
        :func:`lumenairy.propagate_ensemble` (v4.16.1) for the
        canonical workflow.
    TypeError
        If ``E`` is an :class:`numpy.matrix`, or if its dtype kind is
        outside :data:`_FIELD_DTYPE_KINDS` (i.e. not complex / real /
        integer / boolean -- an object-dtype array being the usual
        offender).  A guard that promises "2-D complex" but enforces
        only ``ndim == 2`` lets both of these reach the kernels and
        produce a plausible finite result computed by the wrong
        arithmetic --
        :class:`numpy.matrix` because its ``*`` is a matrix product, so
        every elementwise mask / phase-screen multiply downstream became
        a matmul; object dtype because NumPy falls back to calling the
        elements' own ``__mul__``.  Real / integer / boolean input stays
        ACCEPTED: an amplitude mask is a legitimate field and every
        kernel promotes it to complex.
    """
    if input_kind not in _INPUT_KINDS:
        raise ValueError(
            f"{fn_name}: input_kind must be one of "
            f"{sorted(_INPUT_KINDS)}; got {input_kind!r}.  "
            f"``input_kind`` only selects the noun used in this "
            f"guard's rejection message ('expected 2-D complex "
            f"<kind>'); it does not change what is accepted.  If a "
            f"new entry point genuinely guards a different kind of "
            f"2-D array, add the value to "
            f"``lumenairy._validation._INPUT_KINDS`` rather than "
            f"passing an unlisted string."
        )

    if isinstance(E, _MCF):
        # The prior
        # rejection message cited "v4.16+ scope" -- but the library
        # version was v4.16.0 at the time of the audit, leaving the
        # message stale.  Replace with an honest pointer to the
        # ``propagate_ensemble`` helper (which actually solves the
        # caller's partial-coherence-propagation need) and note that
        # MCF-aware downstream propagators are a v5.0+ architectural
        # change rather than a near-term feature.
        raise TypeError(
            f"{fn_name}: PartialCoherenceMCF inputs are not yet "
            f"supported by this propagator.  MCF-aware downstream "
            f"propagators are an architectural change tracked in "
            f"ROADMAP v5.0+; for partial-coherence propagation today, "
            f"use ``lumenairy.propagate_ensemble(ensemble, ...)`` "
            f"instead.  Build the ensemble via "
            f"``create_gaussian_schell_source(..., "
            f"return_kind='ensemble')`` (or one of the sibling Schell "
            f"factories), then:\n"
            f"    I_partial = propagate_ensemble(\n"
            f"        ensemble, dx=dx, wavelength=wl,\n"
            f"        propagator={fn_name!r},  # or any callable\n"
            f"        # forwarded kwargs (e.g. z=...) here\n"
            f"    )\n"
            f"``propagate_ensemble`` returns ``< |E_k|^2 >_k`` -- the "
            f"canonical Wolf coherence-theory partial-coherence "
            f"intensity for a Schell-model source."
        )

    ndim = getattr(E, "ndim", None)
    if ndim is None or ndim != 2:
        # 0-D / 1-D / 3-D / scalar inputs all land here.  Tailor
        # the suggestion text by ndim so the message is useful.
        shape_str = getattr(E, "shape", "(unknown shape)")
        if ndim == 3:
            # The ensemble hint points at the
            # :func:`propagate_ensemble` helper rather than the bare
            # iterate-pattern.  The bare iterate still works (and is
            # left as a fallback below for users with custom
            # propagator pipelines), but ``propagate_ensemble`` is
            # the canonical entry point.
            hint = (
                "  If this is an ensemble of realizations from "
                "create_*_schell_source(), use the v4.16.1 "
                "``propagate_ensemble`` helper:\n"
                "    import lumenairy as la\n"
                f"    I_partial = la.propagate_ensemble(\n"
                f"        ensemble, dx=dx, wavelength=wl,\n"
                f"        propagator={fn_name!r},  # or any callable\n"
                f"    )\n"
                "  Or iterate manually for a custom propagator chain:\n"
                f"    for k in range(ensemble.shape[0]):\n"
                f"        E_out_k = {fn_name}(ensemble[k], ...)\n"
                "    I_partial = "
                "np.mean(np.abs(out_stack)**2, axis=0)"
            )
        elif ndim is None or ndim < 2:
            hint = (
                "  Expected a 2-D NumPy/CuPy complex array of shape "
                "(Ny, Nx).  A scalar / 1-D array suggests you may "
                "be passing the wrong object (e.g., a coordinate "
                "array, a scalar wavelength, or the wrong tuple "
                "element)."
            )
        else:
            hint = (
                "  Expected 2-D; higher-rank inputs are not "
                "supported."
            )
        raise ValueError(
            f"{fn_name}: expected 2-D complex {input_kind} of shape "
            f"(Ny, Nx); got {ndim}-D array of shape "
            f"{shape_str}.\n"
            f"{hint}"
        )

    # The two 2-D inputs that pass the ndim check and then compute
    # something plausible but wrong (docs/history/lumenairy._validation.md).
    if isinstance(E, np.matrix):
        raise TypeError(
            f"{fn_name}: np.matrix is not a valid {input_kind}.  Its ``*`` "
            f"is MATRIX multiplication and its ``**`` is matrix power, so "
            f"every elementwise step downstream -- multiplying by an "
            f"aperture mask, by a phase screen, by a transfer function -- "
            f"would silently compute a matrix product instead, returning a "
            f"finite array that is not the field you asked for.  Pass "
            f"``np.asarray(M)`` (a view; no copy) instead.  np.matrix is "
            f"itself deprecated in NumPy."
        )

    dtype = getattr(E, 'dtype', None)
    kind = getattr(dtype, 'kind', None)
    if kind is not None and kind not in _FIELD_DTYPE_KINDS:
        raise TypeError(
            f"{fn_name}: expected a 2-D complex (or real / integer / "
            f"boolean, for an amplitude mask) {input_kind}; got dtype "
            f"{dtype!r}.  An object-dtype array is the usual cause -- it "
            f"comes from ``np.array(list_of_ragged_rows)`` or from mixing "
            f"Python scalars with arrays -- and NumPy would run every "
            f"downstream kernel on the Python-object slow path, calling "
            f"whatever ``__mul__`` the elements define.  Build the field "
            f"with an explicit dtype, e.g. "
            f"``np.asarray(E, dtype=np.complex128)``."
        )
