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
        ``psf`` (``compute_otf`` / ``compute_mtf`` if Agent A wires
        them up) see "expected 2-D complex pupil" / "psf" instead
        of the literal "field" string.  Conventional values:
        ``'field'`` (default), ``'pupil'``, ``'psf'``.  v4.15.5
        (P2-NEW-F1-3): parameterised after the v4.15.4 audit noted
        that the vector-diffraction sites took a pupil, not a field,
        but the error message hardcoded "field".

    Raises
    ------
    TypeError
        If ``E`` is a :class:`lumenairy.PartialCoherenceMCF`
        instance.  MCF-aware downstream propagators are deferred
        to v4.16+; the error message points the user at the
        v4.16-roadmap entry and gives the ensemble-iteration
        workaround.
    ValueError
        If ``E.ndim != 2``.  Common cause: the user passed a
        v4.15.1+ Schell-family ensemble (shape
        ``(n_realizations, Ny, Nx)``) directly to a coherent
        propagator.  The error message shows the canonical
        iterate-and-average pattern.
    """
    if isinstance(E, _MCF):
        raise TypeError(
            f"{fn_name}: PartialCoherenceMCF inputs are not yet "
            f"supported by this propagator (v4.15+).  MCF-aware "
            f"downstream propagators are scheduled for v4.16+.  "
            f"To propagate a partially-coherent field via "
            f"ensemble averaging, iterate over the ensemble "
            f"realizations:\n"
            f"    for k in range(ensemble.shape[0]):\n"
            f"        E_out_k = {fn_name}(ensemble[k], ...)\n"
            f"    I_partial = np.mean(np.abs(out_stack)**2, axis=0)"
        )

    ndim = getattr(E, "ndim", None)
    if ndim is None or ndim != 2:
        # 0-D / 1-D / 3-D / scalar inputs all land here.  Tailor
        # the suggestion text by ndim so the message is useful.
        shape_str = getattr(E, "shape", "(unknown shape)")
        if ndim == 3:
            hint = (
                "  If this is an ensemble of realizations from "
                "create_*_schell_source(), iterate over the leading "
                "axis:\n"
                f"    for k in range(ensemble.shape[0]):\n"
                f"        E_out_k = {fn_name}(ensemble[k], ...)"
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
