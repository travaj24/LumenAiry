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
from lumenairy.sources.core import PartialCoherenceMCF as _MCF

# v5.31 (audit A-9): the closed vocabulary for ``input_kind``.
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
        "field" string.  v4.15.5 (P2-NEW-F1-3): parameterised after
        the v4.15.4 audit noted that the vector-diffraction sites
        took a pupil, not a field, but the error message hardcoded
        "field".  v5.31 (audit A-9): the value must be a member of
        the closed :data:`_INPUT_KINDS` vocabulary
        (``'field'`` / ``'psf'`` / ``'pupil'``) -- see the Raises
        section -- and is now declared explicitly at every wired
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
        # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b): the prior
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
            # v4.16.1 (audit AUDIT_V4_16_0_DEEP item 5b follow-up):
            # the ensemble hint now points at the
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
