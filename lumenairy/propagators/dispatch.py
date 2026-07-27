"""
lumenairy.propagators.dispatch -- top-level smart-method propagator.

Picks the most appropriate diffraction propagator for the given
input + prescription + output geometry.  The user calls one
function and the dispatcher figures out whether to use ASM,
Fresnel, Maslov, GBD, HFPI, or HF based on system properties.

Method selection logic (when ``method='auto'``)
-----------------------------------------------

1. **Prescription with diffractive surfaces** (DOEs / hard
   apertures inside): ``hfpi``.
2. **Prescription without diffractive surfaces**:  ``maslov``.
3. **No prescription, only z**:
   - Far-field (Fresnel number << 1):  ``fraunhofer``.
   - Otherwise:  ``asm``.

The user can always override by passing ``method='asm'`` /
``'fresnel'`` / ``'fraunhofer'`` / ``'rs'`` / ``'maslov'`` /
``'asymptotic'`` / ``'gbd'`` / ``'hfpi'`` / ``'hf'`` / ``'mhs'``.

Author: Andrew Traverso
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

from .._deprecation import _NO_DEFAULT

VALID_METHODS = (
    'auto', 'asm', 'sas', 'fresnel', 'fraunhofer', 'rs',
    'maslov', 'asymptotic', 'gbd', 'hfpi', 'hf', 'mhs',
)

# v5.30 (audit P5): the bare-grid kernels whose native return is the
# ``(E, dx_out, dy_out)`` triple at a kernel-chosen output pitch rather
# than a bare ndarray at the input pitch.  ``method='auto'`` can pick
# any of these, so a pre-v5.30 ``auto`` caller who read the return as an
# ndarray silently got a tuple -- and at a different sampling.  That is the
# instability recorded in
# ``docs/audits/AUDIT_ADVERSARIAL_CODEBASE_2026_07_25.md`` P5 and closed by
# the flip described in the block below: the DEFAULT return is now the
# shape-stable ``PropagationResult`` for every method.  This set stays as-is
# because the native shapes stay reachable -- it gates the shape-instability
# ``UserWarning``, which post-flip can only fire on the explicit
# ``return_result=False`` (legacy-contract) path.
_GRID_CHANGING_METHODS = ('sas', 'fresnel', 'fraunhofer')

# ---------------------------------------------------------------------------
# audit P5 / roadmap Part F1 -- the return-contract flip, EXECUTED (v5.30)
# ---------------------------------------------------------------------------
# The F1 decision (four costed options in
# ``docs/roadmap_deferred_2026_07_21.md``) landed as **option 4**: make the
# shape-stable :class:`~lumenairy.propagators.PropagationResult` the DEFAULT
# return while keeping the kernels' native shapes reachable behind an explicit
# ``return_result=False``.  v5.30 first shipped only the announcement (a
# registry-scheduled ``DeprecationWarning`` plus a falsy sentinel default);
# the owner then chose to EXECUTE the flip in the same release rather than
# ship a warning about a change nobody could yet see -- the same call the W5
# shim-removal wave made (see ``lumenairy/_deprecation.py``'s tombstones and
# the CHANGELOG's ``### Changed (BREAKING)`` section).
#
# As shipped:
#
#   * ``return_result`` UNSET -- a ``PropagationResult`` for **every** method.
#     ``.field`` / ``.dx`` / ``.dy`` are defined whichever kernel ran, so the
#     return shape no longer depends on ``z``.  That is the P5 finding closed.
#   * ``return_result=False`` -- the kernels' native shapes (bare ndarray OR
#     ``(E, dx_out, dy_out)``), bit-for-bit as before the flip.  A PERMANENT,
#     documented escape hatch: the migration path for ``E, dxo, dyo``
#     unpackers (``PropagationResult`` iteration stays 2-item, audit P16) and
#     for wrapper-free fast loops.  It is not deprecated and nothing is
#     scheduled against it.
#   * ``return_result=True`` -- unchanged.
#
# Retired WITH the flip (tombstone, v5.30): the transition
# ``DeprecationWarning`` (``_p5_transition_message``), its external-caller
# predicate (``_caller_is_internal``), ``_P5_DEPRECATED_SINCE`` /
# ``_P5_UNSTABLE_RETURN_TYPES``, and the ``API_TRANSITION_VERSION`` /
# ``resolve_removal_version`` imports that resolved its horizon.  A warning
# whose text is "the default WILL become a PropagationResult in vX" cannot
# outlive the version that makes it a PropagationResult; leaving it would
# advertise a future change that has already happened -- the exact
# registry-rot class the horizon mechanism exists to prevent.  Its purpose is
# served by the decision record that replaces it: this block, ``propagate``'s
# docstring, the roadmap's F1 EXECUTED entry, and the CHANGELOG.  Nothing
# remains to warn a *caller* about: the default is now the stable contract and
# the alternative is an explicit, supported argument.
#
# The ``_NO_DEFAULT`` sentinel STAYS as the parameter default (rather than
# becoming a literal ``True``) because the two are different statements:
# ``True`` says "this caller asked for the wrapper", the sentinel says "this
# caller did not choose, so the library's stable contract applies".  Keeping it
# means the distinction the transition measured stays available to any future
# contract decision, and ``inspect.signature(propagate)`` stays honest about
# which values are a *choice*.
#
# WARNING for future edits: the sentinel is FALSY.  ``if not return_result``
# would route it to the legacy contract -- i.e. silently un-flip this change.
# :func:`propagate` therefore resolves it ONCE, up front, into a local ``wrap``
# flag and routes on that; do the same in any new branch.


def propagate(
    E_in: np.ndarray,
    *,
    z: Optional[float] = None,
    wavelength: float,
    dx: float,
    prescription: Optional[Dict[str, Any]] = None,
    method: str = 'auto',
    accuracy: str = 'balanced',
    output_grid: Optional[tuple] = None,
    output_dx: Optional[float] = None,
    # v5.30 (audit P5 / roadmap F1, EXECUTED): the default is the "not passed"
    # sentinel, NOT a literal ``True``, so "the library's stable contract
    # applies" stays distinguishable from "this caller asked for the wrapper".
    # It resolves to the STABLE contract (a PropagationResult).  ``_NO_DEFAULT``
    # is FALSY, so it must never be routed on directly -- it is resolved once
    # into ``wrap`` below.  See the flip block at the top of this module.
    return_result: Any = _NO_DEFAULT,
    **method_kwargs: Any,
) -> Any:
    """Top-level smart-method propagator.

    Routes the call to the most appropriate underlying propagator
    based on the geometry of the request and the structure of the
    prescription (when provided).  See the module docstring for
    selection logic.

    .. warning::
       **Return contract, settled in v5.30 (audit P5, roadmap Part F1
       option 4 -- EXECUTED).**  The DEFAULT return is a
       :class:`~lumenairy.propagators.PropagationResult` for **every**
       method: ``.field`` / ``.dx`` / ``.dy`` are defined whichever kernel
       ran, and ``np.asarray(result)`` yields the field, so the return
       shape no longer depends on ``z``.  Pre-v5.30 the default was the
       chosen kernel's native shape -- a bare ``ndarray`` *or* an
       ``(E, dx_out, dy_out)`` triple, see the table below -- which is the
       instability P5 raised.

       * ``return_result=False`` -- the native shapes, bit-for-bit as
         before the flip.  A **permanent, supported escape hatch**, not a
         deprecated one: it is the migration for code that unpacks
         ``E, dxo, dyo`` and for fast loops that want no wrapper
         allocation.
       * ``return_result=True`` -- unchanged, and now the same contract the
         default hands back.

       ``PropagationResult`` iteration did **not** move to 3 items at the
       flip (audit P16): it stays ``(field, intermediates)``, which is what
       ``E, inter = propagate_through_system(..., return_result=True)``
       needs.  Re-arity-ing it would have traded one breakage for another,
       and option 4 does not require it -- ``return_result=False`` is the
       migration path for 3-tuple unpackers.

       The transition :class:`DeprecationWarning` that announced this flip
       while it was still scheduled is retired with the flip itself: the
       default it pointed callers away from no longer exists.

    .. versionchanged:: 5.30
       Default return became ``PropagationResult`` for every method
       (roadmap ``docs/roadmap_deferred_2026_07_21.md`` Part F1 option 4,
       audit P5).  ``return_result=False`` restores the pre-v5.30 shapes
       exactly; ``return_result=True`` is unaffected.

    ``method='auto'`` return contract (audit P5)
    -------------------------------------------
    **On the legacy path the native return type and the output sampling
    both depend on which kernel the selector picks**, i.e. on ``z`` /
    ``prescription`` / ``accuracy`` -- not on anything the caller wrote.
    That path is now reached only by asking for it
    (``return_result=False``); the default wraps every one of these in a
    ``PropagationResult`` whose ``.field`` / ``.dx`` / ``.dy`` mean the
    same thing for all of them.  With ``return_result=False`` the caller
    receives whatever the chosen kernel returns:

    * **asm** -- native return ``ndarray``; output pitch ``dx``
      (unchanged).  Chosen for free space when ``z`` is None / 0 /
      negative, or when ``N_F >= 0.1`` and ``Q <= 1``.
    * **sas** -- native return ``(E, dx_out, dy_out)``; output pitch
      ``lambda*z / (pad*N*dx)``.  Chosen for free space when
      ``N_F >= 0.1`` and ``Q > 1`` **and no output grid was requested**.
      v5.31 (audit W9-1): with ``output_grid`` / ``output_dx`` given, that
      band selects ``asm`` instead (SAS has no output-grid path, so
      selecting it raised a ``ValueError`` naming a kernel the caller
      never wrote); ``asm`` auto-promotes to the exact
      :func:`angular_spectrum_propagate_mft`.
    * **fraunhofer** -- native return ``(E, dx_out, dy_out)``; output
      pitch ``lambda*z / (N*dx)``.  Chosen for free space when
      ``N_F < 0.1``.
    * **maslov** / **gbd** / **hfpi** / **hf** -- native return
      ``ndarray``; output pitch ``dx`` (or ``output_dx`` when asked).
      Chosen when a ``prescription`` is supplied; see
      :func:`_auto_select_method` for the sub-branching.

    Measured at ``N=64``, ``dx = 2 um``, ``lambda = 633 nm``:
    ``z = 1e-4`` -> ``asm``, ndarray at 2.0000e-06 m; ``z = 1e-3`` ->
    ``sas``, 3-tuple at 2.4727e-06 m; ``z = 5`` -> ``fraunhofer``,
    3-tuple at 2.4727e-02 m.

    ``N_F = (N*dx/2)^2 / (lambda*|z|)`` is the aperture Fresnel number
    and ``Q = lambda*|z| / (N*dx^2)`` the grid Fresnel ratio; ``N`` is
    ``max(Ny, Nx)``.  ``fresnel`` is never auto-selected but shares the
    triple-return contract when named explicitly.

    Because a bare ndarray and a 3-tuple at a *different* pitch are not
    interchangeable, ``propagate`` still emits a :class:`UserWarning`
    whenever ``method='auto'`` resolves to a grid-changing kernel **and**
    the caller asked for the legacy contract (``return_result=False``) --
    naming the method, the output pitch it produced, and the stable
    alternative.  It cannot fire on the default path any more: that path
    hands back a :class:`~lumenairy.propagators.PropagationResult`, which
    exposes ``.field`` / ``.dx`` / ``.dy`` regardless of which kernel ran,
    so there is no shape instability left to report.

    .. note::
       The shapes in the table above are bit-for-bit what pre-v5.30
       releases returned by default; v5.30 changed *which of them you get
       without asking*, not what any of them contain.  The deferred
       four-option costing that chose this route (option 4) is recorded in
       ``docs/roadmap_deferred_2026_07_21.md`` Part F1 (audit P5), with the
       bit-identity evidence for both explicit modes.  ``return_result``
       is the only knob: there is no version-scheduled behaviour left here
       and no deprecation attached to either value.

    Parameters
    ----------
    accuracy : 'fast' | 'balanced' | 'accurate', default 'balanced'
        Hint for the ``method='auto'`` selector:

          * ``'fast'`` -- prefer the cheapest method that is
            asymptotically valid (e.g. ``'maslov'`` over GBD when
            both apply).
          * ``'balanced'`` -- the default; trades accuracy for
            speed on a case-by-case basis.
          * ``'accurate'`` -- prefer the highest-fidelity method
            for the geometry (e.g. ``'gbd'`` over ``'maslov'`` for
            aspherics, ``'hf'`` over ``'maslov'`` for general
            paraxial-violating systems).

        Has no effect when ``method`` is set to a specific string.
    output_grid, output_dx : tuple / float, optional
        Request an output grid that differs from the input pitch.

        - GBD / HFPI / HF forward these to their underlying
          (prescription-driven) propagators directly.
        - ASM / Fresnel / Fraunhofer auto-promote to their MFT
          variants (:func:`angular_spectrum_propagate_mft`,
          :func:`fresnel_propagate_mft`,
          :func:`fraunhofer_propagate_mft`) when ``output_grid`` or
          ``output_dx`` is given.
        - SAS / RS do not support arbitrary output-grid sampling and
          raise ``ValueError`` (pointing at the ASM-MFT entry point)
          if ``output_grid`` / ``output_dx`` is passed.  Since v5.31
          ``method='auto'`` no longer *selects* SAS when an output grid is
          requested (audit W9-1), so that raise is reachable only by
          naming ``method='sas'`` yourself.
        - Maslov / asymptotic / MHS do not thread the request to their
          kernels and raise ``ValueError`` naming the members that do
          (v5.31, audit W9-4).  Pre-v5.31 the request was silently
          dropped -- and with the ``output_dx`` shortcut the returned
          ``PropagationResult.dx`` reported the requested pitch while the
          field was still at the input pitch.
        - The pitch reported on the result honours **either** form: since
          v5.31 ``output_grid=(N_out, dx_out)`` sets
          ``PropagationResult.dx`` to ``dx_out`` (audit W9-5); pre-fix
          only the ``output_dx`` shortcut did, so an ``output_grid`` call
          came back labelled with the input pitch even though the field
          had genuinely been resampled.

        ``output_grid`` may be a ``(N_out, dx_out)`` tuple or a
        ``{'N': ..., 'dx': ...}`` dict.  ``output_dx`` is a shortcut
        when only the pitch needs to change (``N_out`` defaults to
        the input ``N``).  Pre-4.12 the ASM family silently dropped
        these kwargs and returned a bare-grid output at the input
        pitch -- a quiet wrong-physics path that audit round-4 B1-8
        flagged.
    return_result : bool, optional
        Selects the return contract.  **Unset (the default) is the stable
        contract: a ``PropagationResult`` for every method** (v5.30, audit
        P5 / roadmap F1 -- see the warning at the top of this docstring).

        When True, wrap the output in a
        :class:`lumenairy.propagators.PropagationResult` carrying
        the field plus ``dx``, ``wavelength``, ``method``, and a
        ``metadata`` dict.  When False, return the bare
        propagator output (typically a complex ndarray) -- preserving
        backward compatibility and zero-overhead fast loops.

        **``False`` is permanent and un-deprecated**, and is exactly how
        pre-v5.30 callers keep their shapes: ``propagate(...,
        return_result=False)`` is bit-for-bit the pre-flip default.  The
        sentinel default is kept (rather than a literal ``True``) so the
        library can still tell "did not choose -- give me the stable
        contract" apart from "this caller asked for the wrapper"; both
        resolve to the same return today.

        4.12: for tuple-returning kernels (Fresnel / Fraunhofer / SAS
        return ``(E, dx_out, dy_out)``) the wrapped result now reports
        the kernel's **output** dx, not the input dx.  Pre-4.12 audit
        round-4 B1-7: tuple unpacking silently failed, ``field`` was
        ``None``, and ``dx`` was the input pitch.

        .. warning::
           **The wrapper is not a drop-in for the un-wrapped tuple**
           (audit P16).  ``PropagationResult`` is iterable, but it yields
           exactly **two** items -- ``(field, intermediates)`` -- for
           back-compat with ``E, inter = propagate_through_system(...)``,
           whereas the un-wrapped Fresnel / Fraunhofer / SAS kernels
           yield **three** (``E, dx_out, dy_out``).  So::

               E, dxo, dyo = propagate(..., method='sas',
                                       return_result=False)             # OK
               E, dxo, dyo = propagate(..., method='sas')               # ValueError
               E, dxo, dyo = propagate(..., method='sas',
                                       return_result=True)              # ValueError

           Read the attributes (``.field``, ``.dx_out``, ``.dy_out``)
           instead of unpacking.  The 2-item iteration is pinned
           behaviour and did not change **at the F1 flip either**, which
           was decided explicitly in the same pass rather than left as a
           collision (audit P16).  ``return_result=False`` is the supported
           migration for ``E, dxo, dyo`` unpackers -- and since v5.30 they
           must pass it, because the bare 3-tuple is no longer the default.

    Warns
    -----
    UserWarning
        When ``method='auto'`` selects a grid-changing kernel
        (``sas`` / ``fraunhofer``) **and** the caller asked for the legacy
        contract with ``return_result=False`` -- see the return-contract
        table above (audit P5).  Explicit
        ``method='sas'`` / ``'fresnel'`` / ``'fraunhofer'`` calls are
        silent: the caller named that kernel and knows its contract.  The
        default and ``return_result=True`` are silent too -- both deliver
        the shape-stable wrapper, so the diagnostic has nothing to report
        (v5.30: pre-flip this also fired on the default path, which was
        then the legacy contract).
    """
    # v4.15.3 (P0-NEW-F2-1): defensive guard for PartialCoherenceMCF
    # and non-2-D inputs via the shared ``_check_2d_scalar_field``
    # helper.  v4.15.2 inlined the guard here; v4.15.3 routes through
    # the helper so future entry points can't be added unguarded.
    # Runs FIRST (before method-validation or auto-select) so the user
    # gets a clear, actionable error rather than a downstream
    # AttributeError or a silent wrong-axis FFT.
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'propagate', input_kind='field')

    if method not in VALID_METHODS:
        raise ValueError(
            f"propagate: method must be one of {VALID_METHODS}, "
            f"got {method!r}.")

    auto_selected = (method == 'auto')
    if auto_selected:
        # v5.31 (audit W9-1): the selector has to know that the caller asked
        # for a non-native output grid, otherwise it can pick a kernel that
        # cannot deliver one and the caller gets a ValueError naming a method
        # they never wrote.  Same principle as the 4.12 B1-6 z<0 guard.
        method = _auto_select_method(
            E_in, z=z, wavelength=wavelength, dx=dx,
            prescription=prescription, accuracy=accuracy,
            output_requested=(output_grid is not None or output_dx is not None))

    # v5.30 (audit P5 / roadmap F1 option 4, EXECUTED): resolve the return
    # contract ONCE, here, and route on ``wrap`` alone below.  ``_NO_DEFAULT``
    # ("caller did not choose") resolves to the STABLE contract -- that is the
    # flip.  Every other value keeps its truthiness, so an explicit
    # ``return_result=False`` (and any other falsy value a pre-flip caller
    # passed) still selects the kernels' native shapes, bit-for-bit as before.
    # Do NOT test ``return_result`` directly in new code: the sentinel is
    # falsy, so ``if not return_result`` would silently restore the pre-flip
    # default and un-do this change.
    wrap = True if return_result is _NO_DEFAULT else bool(return_result)

    out = _dispatch_to_method(
        method, E_in,
        z=z, wavelength=wavelength, dx=dx,
        prescription=prescription,
        output_grid=output_grid,
        output_dx=output_dx,
        **method_kwargs,
    )
    if not wrap:
        # The legacy contract, reached only by asking for it
        # (``return_result=False``) since the v5.30 flip.  v5.30 (audit P5):
        # the auto-selector can hand back a bare ndarray at the input pitch OR
        # an ``(E, dx_out, dy_out)`` triple at a kernel-chosen pitch, decided
        # purely by ``z`` -- and the caller has no way to know which without
        # re-running the selector.  This warning says so, out loud, exactly
        # when it bites: ``auto`` chose a grid-changing kernel and the caller
        # opted out of the shape-stable wrapper.  The reported pitch is read
        # off the kernel's own return, so no formula is duplicated here.
        # Post-flip this is the ONLY place either shape-warning survives: the
        # default and ``return_result=True`` both return the wrapper, whose
        # shape does not depend on z, so there is nothing to warn about.
        if auto_selected and method in _GRID_CHANGING_METHODS:
            warnings.warn(
                _auto_grid_change_message(method, out, dx),
                UserWarning, stacklevel=2)
        return out

    from .result import PropagationResult
    # v5.31 (audit W9-5): the requested output pitch can arrive EITHER as the
    # ``output_dx`` shortcut OR as the second element of the canonical
    # ``output_grid = (N_out, dx_out)`` tuple / ``{'N': ..., 'dx': ...}`` dict.
    # Pre-fix only the shortcut was read here, so an ``output_grid=(96, 80e-6)``
    # call -- which every honouring kernel really does resample to 80 um (the
    # field is bit-identical to additionally passing ``output_dx=80e-6``) --
    # came back labelled with the INPUT pitch.  MEASURED on a 64^2 / dx=40 um
    # probe: ``propagate(..., output_grid=(96, 80e-6))`` returned
    # ``field.shape == (96, 96)`` with ``result.dx == 4e-05`` for asm (via the
    # MFT promotion) and for gbd / hf / hfpi (which forward the request), i.e.
    # the wrapper's own sampling metadata was wrong by 2x on the DEFAULT
    # (post-P5-flip) contract, for every downstream coordinate / plot / store
    # consumer.  Kernels that report their own ``dx_out`` still win below.
    default_out_dx = _requested_output_dx(output_grid, output_dx)
    if default_out_dx is None:
        default_out_dx = dx
    # Best-effort: bare ndarray -> wrap directly; tuple / list / other
    # -> unpack the field and record the propagator-reported output
    # pitch when present.  4.12 fix (audit round-4 B1-7): kernels like
    # fresnel_propagate / fraunhofer_propagate / scalable_angular_spectrum_propagate
    # return ``(E, dx_out, dy_out)``; pre-4.12 the tuple path went
    # through _coerce_field which silently dropped to None and reported
    # the INPUT dx instead of the kernel's output dx.
    if isinstance(out, np.ndarray):
        return PropagationResult(
            field=out, dx=default_out_dx, wavelength=wavelength,
            z=z, method=method, metadata={},
        )
    # PropagationResult passthrough (some propagators may already wrap).
    if isinstance(out, PropagationResult):
        return out
    field_arr, dx_from_kernel, dy_from_kernel = _coerce_field(out)
    # v5.31 (audit W9-6): the whole point of the P5 flip is that ``.field`` is
    # defined "whichever kernel ran".  ``_coerce_field`` has a ``(None, None,
    # None)`` sentinel for returns it cannot read, and pre-fix that sentinel was
    # wrapped as-is -- so the flipped contract handed back a
    # ``PropagationResult(field=None)`` in complete silence.  MEASURED:
    # ``propagate(E, method='mhs', subdomains=[...], return_intermediate=True)``
    # -> ``PropagationResult`` with ``field is None``, no warning.  (MHS's
    # native shape there is a ``list`` of ``(HuygensSurface, ndarray)`` pairs,
    # and ``return_intermediate=True`` is MhsPipeline.run's OWN default -- the
    # dispatcher merely defaults it to False.)  A wrapper that cannot honour its
    # own contract must say so rather than emit a null field.
    if field_arr is None:
        raise ValueError(
            f"propagate(method={method!r}): the kernel returned a "
            f"{type(out).__name__} that this dispatcher cannot express as a "
            f"PropagationResult -- there is no single output field to put in "
            f"``.field`` (a per-plane/per-surface sequence, most likely).  "
            f"Since v5.30 the DEFAULT return is the shape-stable wrapper "
            f"(audit P5), and pre-v5.31 this path silently produced "
            f"``PropagationResult(field=None)``.  Pass return_result=False to "
            f"receive the kernel's native return unchanged (for method='mhs' "
            f"that is the ``[(surface, field), ...]`` history), or drop the "
            f"argument that makes the kernel return a sequence (e.g. "
            f"return_intermediate=True) to get a single output plane.")
    out_dx = dx_from_kernel if dx_from_kernel is not None else default_out_dx
    # v4.13.0 (audit L3): thread the kernel-reported ``dy_out`` onto
    # the wrapped result.  For square-grid kernels that only return
    # ``dx_out`` (or a bare ndarray) ``dy`` falls back to ``out_dx``,
    # preserving back-compat.  Pre-fix the y-pitch was silently
    # discarded for anamorphic Fresnel / Fraunhofer / SAS calls.
    out_dy = dy_from_kernel if dy_from_kernel is not None else out_dx
    return PropagationResult(
        field=field_arr,
        dx=out_dx, dy=out_dy, wavelength=wavelength,
        z=z, method=method,
        metadata={'native_return': out},
    )


def _auto_grid_change_message(method, out, dx_in):
    """Build the audit-P5 ``UserWarning`` text for an ``auto``-selected
    grid-changing kernel returned un-wrapped.

    Reads the delivered return SHAPE and the kernel-reported output pitch
    off ``out`` itself (rather than re-deriving ``dx_out`` from the
    kernel's formula) so the numbers quoted can never drift from what the
    caller actually holds.
    """
    _field, dx_out, dy_out = _coerce_field(out)
    ret_kind = ('a bare ndarray' if isinstance(out, np.ndarray)
                else f'a {len(out)}-tuple (E, dx_out, dy_out)'
                if isinstance(out, (tuple, list))
                else f'a {type(out).__name__}')
    if dx_out is None:
        pitch = 'a kernel-chosen output pitch'
    elif dy_out is not None and dy_out != dx_out:
        pitch = (f'output pitch dx_out={dx_out:.6e} m, dy_out={dy_out:.6e} m '
                 f'(input dx={float(dx_in):.6e} m)')
    else:
        pitch = (f'output pitch dx_out={dx_out:.6e} m '
                 f'(input dx={float(dx_in):.6e} m, '
                 f'ratio {dx_out / float(dx_in):.4g}x)')
    return (
        f"propagate(method='auto'): the selector chose {method!r}, which "
        f"returns {ret_kind} at {pitch} -- NOT a bare ndarray at the input "
        f"pitch as method='asm' would.  Which kernel runs (and therefore "
        f"both the return shape and the output sampling) depends on z, so a "
        f"caller that unpacks this return has no stable contract.  This call "
        f"opted into that legacy contract (return_result=False); drop the "
        f"argument -- since v5.30 the DEFAULT is the shape-stable "
        f"PropagationResult (.field / .dx / .dy on every method), and "
        f"return_result=True asks for it explicitly -- or name the method "
        f"explicitly to keep the native shape and silence this warning.  See "
        f"the method='auto' return-contract table in propagate()'s docstring "
        f"(audit P5)."
    )


def _coerce_field(x):
    """Coerce a non-ndarray propagator return into a (field, dx_out,
    dy_out) triple when possible.

    Returns ``(ndarray | None, dx_out | None, dy_out | None)``.

    * ``dx_out`` / ``dy_out`` are the propagator-reported output grid
      pitches if the kernel returns a ``(E, dx_out, ...)`` /
      ``(E, dx_out, dy_out)`` tuple, else ``None``.
    * v4.13.0 (audit L3): the triple-return is the closure for the
      anamorphic Fresnel info-loss bug -- pre-fix ``_coerce_field``
      ignored the third tuple element, silently discarding the y-axis
      pitch for any anamorphic Fresnel / Fraunhofer / SAS propagation.
    * 4.12 fix (audit round-4 B1-7): pre-4.12 the tuple-returning
      propagators (fresnel/fraunhofer/SAS) silently yielded
      ``field=None`` and ``dx=<input pitch>`` instead of the kernel's
      real output.

    The dispatcher records ``dx_out`` on :attr:`PropagationResult.dx`
    and ``dy_out`` on :attr:`PropagationResult.dy`; when the kernel
    returns only ``dx_out`` (or only the bare ndarray) the dispatcher
    falls back to ``dy = dx`` for back-compat.
    """
    # Tuple / list returned by fresnel_propagate, fraunhofer_propagate,
    # scalable_angular_spectrum_propagate -- shape ``(E, dx_out, dy_out)``
    # for the all-FFT methods; ``(E, dx_out)`` for the resample helper.
    if isinstance(x, (tuple, list)) and len(x) >= 1:
        first = x[0]
        if isinstance(first, np.ndarray):
            dx_out = None
            dy_out = None
            if len(x) >= 2:
                try:
                    dx_out = float(x[1])
                except (TypeError, ValueError):
                    dx_out = None
            if len(x) >= 3:
                try:
                    dy_out = float(x[2])
                except (TypeError, ValueError):
                    dy_out = None
            return first, dx_out, dy_out
        return None, None, None
    try:
        arr = np.asarray(x)
        if np.iscomplexobj(arr) or arr.dtype.kind == 'f':
            return arr, None, None
    except (TypeError, ValueError):
        # np.asarray rejects non-array-like inputs with TypeError;
        # ragged / inhomogeneous sequences raise ValueError.  Either
        # way the kernel return doesn't look like a field and we fall
        # through to the (None, None, None) sentinel below.
        pass
    return None, None, None


def _requested_output_dx(output_grid, output_dx):
    """The output pitch the CALLER asked for, or ``None`` if they asked for
    none (v5.31, audit W9-5).

    ``output_dx`` is the shortcut; ``output_grid`` is the canonical
    ``(N_out, dx_out)`` tuple / ``{'N': ..., 'dx': ...}`` dict whose second
    element carries the same quantity.  ``output_dx`` wins when both are given
    (the same precedence :func:`_dispatch_bare_grid_with_output` and
    :func:`_resolve_dispatcher_output_grid` already use, so the pitch reported
    on the result cannot disagree with the pitch handed to the kernel).

    Deliberately tolerant: a malformed ``output_grid`` returns ``None`` here
    rather than raising, because the authoritative validation lives in the two
    resolvers above -- this helper only labels the result.
    """
    if output_dx is not None:
        return float(output_dx)
    if output_grid is None:
        return None
    cand = None
    if isinstance(output_grid, dict):
        cand = output_grid.get('dx')
    elif isinstance(output_grid, (tuple, list)) and len(output_grid) >= 2:
        cand = output_grid[1]
    if cand is None:
        return None
    try:
        return float(cand)
    except (TypeError, ValueError):
        return None


# v5.31 (audit W9-4): the methods whose dispatcher branch does NOT thread
# ``output_grid`` / ``output_dx`` to its kernel.  ``maslov`` is the default
# ``method='auto'`` choice for any prescription without aspherics, so this was
# the most-travelled silent-drop path in the dispatcher.  MEASURED pre-fix on a
# 64^2 / dx=40 um singlet probe:
#
#   propagate(E, prescription=rx, output_dx=80e-6)
#     -> field BIT-IDENTICAL to the no-request call (still 40 um sampling)
#        but PropagationResult.dx reported 8e-05  <-- wrong metadata
#   propagate(E, prescription=rx, output_grid=(96, 80e-6))
#     -> shape (64, 64), dx 4e-05: the request vanished entirely, silently
#
# ``gbd`` / ``hf`` / ``hfpi`` all honour both forms (measured: shape 64->96 and
# dx 40->80 um), so the diagnostic names them.  Raising here is the 4.12 B1-8
# treatment already given to ``sas`` / ``rs``; the alternative (silently
# switching ``auto`` to ``gbd``) would trade a wrong answer for an unannounced
# 100x slowdown and a different physics model.
_NO_OUTPUT_GRID_METHODS = ('maslov', 'asymptotic', 'mhs')

_OUTPUT_GRID_CAPABLE_METHODS = ('gbd', 'hf', 'hfpi')


def _auto_select_method(E_in, *, z, wavelength, dx, prescription,
                          accuracy='balanced', output_requested=False):
    """Pick a method from the geometry + prescription structure.

    Selection logic
    ---------------

    With a prescription:
      1. If any surface carries a DOE / grating phase  ->  ``hfpi``
         (HFPI honours hard diffractive surfaces natively).
      2. If the prescription has any aspheric coefficients and
         ``accuracy in ('balanced', 'accurate')``               ->  ``gbd``
         (Gaussian Beamlet Decomposition is the right choice
         when the paraxial Maslov prediction breaks down at
         high-order asphere terms).
      3. If ``accuracy == 'accurate'`` and any surface has a
         finite ``semi_diameter`` or ``aperture_diameter``      ->  ``hf``
         (Van-Vleck-corrected Huygens-Fresnel handles hard-
         aperture diffraction better than Maslov for general
         systems).
      4. Otherwise                                              ->  ``maslov``
         (paraxial-corrected analytic propagator; fastest of the
         prescription methods).

    Without a prescription (free-space):
      - ``z`` is None or zero                                    ->  ``asm``.
      - Far-field (Fresnel number ``N_F < 0.1``)                ->  ``fraunhofer``.
      - Grid Fresnel ratio ``Q = z*lambda/(N*dx^2) > 1``         ->  ``sas``
        (scalable ASM rescales the output pitch so the spread
        beam fits without aliasing the ASM transfer function)
        -- **unless** ``output_requested``, see below.
      - Otherwise                                               ->  ``asm``.

    Parameters
    ----------
    output_requested : bool, default False
        True when the caller passed ``output_grid`` / ``output_dx`` to
        :func:`propagate`.  v5.31 (audit W9-1): ``sas`` has no output-grid
        path -- :func:`_dispatch_bare_grid_with_output` raises for it -- so
        selecting it for a caller who asked for one produced a ``ValueError``
        naming a kernel the caller never wrote, decided purely by ``z``.
        MEASURED pre-fix at N=64, dx=2 um, lambda=633 nm: ``output_dx=3e-6``
        succeeded at ``z=1e-4`` (asm) and ``z=5`` (fraunhofer) and raised
        ``"propagate(method='sas', ...): SAS does not support arbitrary
        output-grid sampling"`` at ``z=1e-3``.  With ``output_requested`` the
        ``Q > 1`` band selects ``asm`` instead, which auto-promotes to the
        EXACT :func:`angular_spectrum_propagate_mft` -- precisely the remedy
        that SAS error message recommends, applied automatically.  This is the
        4.12 B1-6 rule ("never route the user into a hard-raise from a kernel
        they did not pick by name") applied to the B1-8 feature.  ``fraunhofer``
        is left alone: it has an MFT variant.  Routing with no output-grid
        request is bit-for-bit unchanged.
    """
    if prescription is not None:
        events = prescription.get('events_json') or []
        has_doe = False
        if isinstance(events, list):
            for ev in events:
                if isinstance(ev, dict) and ev.get('type') == 'doe':
                    has_doe = True
                    break
        if has_doe:
            return 'hfpi'

        # Inspect surfaces for aspherics and hard apertures.
        surfs = prescription.get('surfaces') or []
        has_aspheric = False
        has_hard_aperture = False
        for s in surfs:
            if not isinstance(s, dict):
                continue
            asph = s.get('aspheric_coeffs')
            if asph:
                has_aspheric = True
            asph_y = s.get('aspheric_coeffs_y')
            if asph_y:
                has_aspheric = True
            sd = s.get('semi_diameter')
            if sd is not None:
                try:
                    if math.isfinite(float(sd)) and float(sd) > 0:
                        has_hard_aperture = True
                except (TypeError, ValueError):
                    pass
        # Top-level aperture stop counts as a hard aperture as well.
        if prescription.get('aperture_diameter') is not None:
            has_hard_aperture = True

        if has_aspheric and accuracy in ('balanced', 'accurate'):
            return 'gbd'
        if has_hard_aperture and accuracy == 'accurate':
            return 'hf'
        return 'maslov'

    if z is None or z == 0:
        return 'asm'

    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    N_max = max(Ny, Nx)
    a = 0.5 * dx * N_max
    abs_z = abs(z)
    if abs_z == 0:
        return 'asm'
    # 4.12 fix (audit round-4 B1-6): for z < 0 (back-propagation) the
    # forward-only kernels (Fresnel / Fraunhofer / SAS / RS) all raise
    # ValueError.  Restrict the regime check to the back-propagating
    # methods (ASM is the only auto-selectable option) so the dispatcher
    # never silently routes the user into a hard-raise from a kernel
    # they didn't pick by name.  Users who need MFT-style back-propagation
    # at custom output pitch should call angular_spectrum_propagate_mft
    # directly (the auto-selector here only routes between bare-z kernels).
    if z < 0:
        return 'asm'
    N_F = a * a / (wavelength * abs_z)
    if N_F < 0.1:
        return 'fraunhofer'
    # Grid Fresnel ratio Q = z*lambda/(N*dx**2).  When Q > 1 the plain
    # ASM transfer function aliases on the grid; scalable ASM rescales
    # the output pitch so the beam fits without aliasing.
    Q = wavelength * abs_z / (N_max * dx * dx)
    if Q > 1.0:
        # v5.31 (audit W9-1): SAS's output pitch is fixed by construction
        # (``lambda*z/(pad*N*dx)``) and it has no MFT analogue, so it cannot
        # serve a caller who asked for a specific output grid.  ASM can, exactly
        # (Bluestein), for any sign of z -- take it.  See ``output_requested``.
        if not output_requested:
            return 'sas'
        return 'asm'
    return 'asm'


_FORWARD_ONLY_METHODS = ('sas', 'fresnel', 'fraunhofer', 'rs')


def _resolve_dispatcher_output_grid(method, output_grid, output_dx, in_shape):
    """Resolve the dispatcher's ``output_grid = (N_out, dx_out)`` contract
    into the gbd / hfpi / hf sub-propagators' v5.2.0 ``output_shape`` +
    ``output_dx`` form.

    v5.2.3 (AUDIT_V4_13_1 P1-A residual closure).  v5.2.0 renamed the
    sub-propagators' ``output_grid`` -> ``output_shape`` (where the
    sub-propagators' ``output_grid`` had meant the ``(Ny, Nx)`` shape
    only, NOT the dispatcher's documented ``(N_out, dx_out)`` form).
    The dispatcher kept forwarding the legacy ``output_grid`` kwarg
    name AND tuple form, which triggered the v5.2.0
    ``DeprecationWarning`` shim AND mis-interpreted the dispatcher's
    canonical tuple as ``(Ny=N_out, Nx=dx_out)``.  v5.2.3 resolves
    the dispatcher's tuple here once, then forwards via the new
    kwargs.

    Returns ``(output_shape, dx_out)`` where ``output_shape`` is
    ``(N_out, N_out)`` (square) and ``dx_out`` is the resolved output
    pitch in meters.  Either may be ``None`` (caller didn't ask for
    a custom output grid).
    """
    if output_grid is None and output_dx is None:
        return None, None
    # Same parsing path as ``_dispatch_bare_grid_with_output`` but
    # returns shape + dx instead of calling the MFT variant.
    Ny, Nx = in_shape[-2], in_shape[-1]
    N_in = max(Ny, Nx)
    dx_out = None
    N_out = None
    if output_grid is not None:
        if isinstance(output_grid, dict):
            N_out = output_grid.get('N')
            dx_out = output_grid.get('dx')
        elif (isinstance(output_grid, (tuple, list))
              and len(output_grid) >= 2):
            N_out, dx_out = output_grid[0], output_grid[1]
        else:
            raise ValueError(
                f"propagate(method={method!r}, output_grid=...): "
                f"output_grid must be a (N_out, dx_out) tuple or "
                f"{{'N': ..., 'dx': ...}} dict, got "
                f"{type(output_grid).__name__}.")
    if output_dx is not None:
        dx_out = output_dx
    if N_out is None:
        N_out = N_in
    N_out = int(N_out)
    if dx_out is not None:
        dx_out = float(dx_out)
    output_shape = (N_out, N_out)
    return output_shape, dx_out


def _dispatch_bare_grid_with_output(method, E_in, *, z, wavelength, dx,
                                     output_grid, output_dx, **kwargs):
    """Route a bare-grid method (asm/fresnel/fraunhofer/sas/rs) to the
    correct MFT variant when the caller asks for an output-pitch /
    output-grid that differs from the natural FFT output.

    4.12 fix (audit round-4 B1-8).  Behaviour:
      - ``asm`` -> :func:`angular_spectrum_propagate_mft` (forward or
        back-prop -- ASM-MFT supports any sign of z).
      - ``fresnel`` -> :func:`fresnel_propagate_mft` (forward-only).
      - ``fraunhofer`` -> :func:`fraunhofer_propagate_mft` (forward-only).
      - ``sas`` / ``rs`` -> ValueError; no MFT analogue in 4.12.

    ``output_grid`` can be ``(N_out, dx_out)`` or a dict
    ``{'N': ..., 'dx': ...}``.  ``output_dx`` short-circuits and uses
    the input N for the MFT N_out.
    """
    if z is None:
        raise ValueError(
            f"propagate(method={method!r}, output_grid/output_dx=...): "
            f"z is required for an MFT-style output-grid call.")

    # Resolve N_out, dx_out from output_grid or output_dx.
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    N_in = max(Ny, Nx)
    dx_out = None
    N_out = None
    if output_grid is not None:
        if isinstance(output_grid, dict):
            N_out = output_grid.get('N')
            dx_out = output_grid.get('dx')
        elif isinstance(output_grid, (tuple, list)) and len(output_grid) >= 2:
            N_out, dx_out = output_grid[0], output_grid[1]
        else:
            raise ValueError(
                f"propagate(method={method!r}, output_grid=...): "
                f"output_grid must be a (N_out, dx_out) tuple or "
                f"{{'N': ..., 'dx': ...}} dict, got {type(output_grid).__name__}.")
    if output_dx is not None:
        dx_out = output_dx
    if dx_out is None:
        raise ValueError(
            f"propagate(method={method!r}, output_grid=...): could not "
            f"resolve an output dx from output_grid={output_grid!r} "
            f"or output_dx={output_dx!r}.")
    if N_out is None:
        N_out = N_in
    N_out = int(N_out)
    dx_out = float(dx_out)

    if method == 'asm':
        from .propagation import angular_spectrum_propagate_mft
        return angular_spectrum_propagate_mft(
            E_in, z, wavelength, dx, dx_out, N_out, **kwargs)
    if method == 'fresnel':
        from .propagation import fresnel_propagate_mft
        return fresnel_propagate_mft(
            E_in, z, wavelength, dx, dx_out, N_out, **kwargs)
    if method == 'fraunhofer':
        from .propagation import fraunhofer_propagate_mft
        return fraunhofer_propagate_mft(
            E_in, z, wavelength, dx, dx_out, N_out, **kwargs)
    if method == 'sas':
        raise ValueError(
            "propagate(method='sas', output_grid/output_dx=...): "
            "SAS does not support arbitrary output-grid sampling.  Its "
            "output pitch is fixed by `dx_out = lambda*z/(pad*N*dx)`.  "
            "Use method='asm' (auto-promotes to angular_spectrum_propagate_mft) "
            "for explicit output-pitch sampling, or method='fresnel' for "
            "the paraxial-MFT path.")
    if method == 'rs':
        raise ValueError(
            "propagate(method='rs', output_grid/output_dx=...): "
            "Rayleigh-Sommerfeld does not support arbitrary output-grid "
            "sampling in 4.12 (no MFT variant).  Use method='asm' "
            "(auto-promotes to angular_spectrum_propagate_mft) for "
            "output-pitch sampling.")
    raise NotImplementedError(
        f"_dispatch_bare_grid_with_output: method {method!r} not "
        f"covered.")


def _dispatch_to_method(method, E_in, *, z, wavelength, dx,
                        prescription, output_grid, output_dx,
                        **kwargs):
    """Call the chosen propagator with the appropriate signature.

    4.12 fix (audit round-4 B1-6): when the user explicitly picks a
    forward-only method (Fresnel / Fraunhofer / SAS / RS) with z < 0,
    raise a dispatcher-level ValueError naming :func:`propagate` rather
    than letting the kernel raise a confusing error that mentions the
    underlying function the user didn't call by name.  ASM is the only
    auto-supported back-propagation kernel here; users who need
    MFT-style back-prop at a custom output pitch should call
    :func:`angular_spectrum_propagate_mft` directly.

    4.12 fix (audit round-4 B1-8): when the caller passes
    ``output_grid`` / ``output_dx`` and the chosen method is not an MFT
    variant, raise a clear ValueError pointing at the right MFT entry
    point rather than silently dropping the user's request.  The ASM /
    Fresnel / Fraunhofer / SAS / RS kernels in this dispatcher accept
    only the natural FFT output grid; explicit output-pitch sampling
    needs the MFT family (angular_spectrum_propagate_mft,
    fresnel_propagate_mft, fraunhofer_propagate_mft).  GBD / HFPI / HF
    forward ``output_grid`` / ``output_dx`` directly to their
    underlying propagators.
    """
    if method in _FORWARD_ONLY_METHODS and z is not None and float(z) < 0:
        raise ValueError(
            f"propagate(method={method!r}): z must be > 0 (got z={z}).  "
            f"This method is a forward-only propagator.  Use "
            f"method='asm' (or call angular_spectrum_propagate_mft "
            f"directly for custom output-pitch sampling) for "
            f"back-propagation.")

    # Bare-grid methods (no prescription) do not honour output_grid /
    # output_dx -- they always produce the natural FFT output grid.
    # When the caller supplies an output-grid request, route them to
    # the MFT variant or raise a clear ValueError.  Free-space GBD /
    # HFPI / HF *do* take output_grid / output_dx and forward them
    # through their own dispatch below.
    # v5.31 (audit W9-4): ``maslov`` / ``asymptotic`` / ``mhs`` never thread
    # ``output_grid`` / ``output_dx`` to their kernels, and pre-fix the request
    # was dropped in silence -- with the ``output_dx`` shortcut the wrapper even
    # LABELLED the un-resampled field with the requested pitch.  Say so, out
    # loud, and name the members that do honour it (measured: gbd / hf / hfpi
    # all resample).  Same treatment ``sas`` / ``rs`` got in 4.12 (B1-8).
    if method in _NO_OUTPUT_GRID_METHODS and (output_grid is not None
                                              or output_dx is not None):
        raise ValueError(
            f"propagate(method={method!r}, output_grid/output_dx=...): this "
            f"method does not support an output grid that differs from the "
            f"input sampling -- its kernel is not given the request, so "
            f"pre-v5.31 it was silently dropped (and with the ``output_dx`` "
            f"shortcut the returned PropagationResult.dx reported the pitch "
            f"you asked for while the field was still at the input pitch).  "
            f"Use one of {list(_OUTPUT_GRID_CAPABLE_METHODS)} -- they forward "
            f"``output_grid`` / ``output_dx`` to their prescription "
            f"propagators -- or drop the argument and resample the result "
            f"yourself (lumenairy.resample_field).  NOTE: {method!r} is what "
            f"method='auto' selects for a prescription without aspheric "
            f"coefficients, so an ``auto`` call with an output-grid request "
            f"lands here; name method='gbd' (or 'hf' / 'hfpi') to keep the "
            f"request.")

    _BARE_GRID_METHODS = ('asm', 'sas', 'fresnel', 'fraunhofer', 'rs')
    if method in _BARE_GRID_METHODS and (output_grid is not None
                                          or output_dx is not None):
        return _dispatch_bare_grid_with_output(
            method, E_in, z=z, wavelength=wavelength, dx=dx,
            output_grid=output_grid, output_dx=output_dx, **kwargs,
        )

    if method == 'asm':
        from .propagation import angular_spectrum_propagate
        if z is None:
            # Dispatch nit: return a copy, not the input array itself -- a
            # caller mutating the "propagated" z=None output otherwise mutates
            # the source field in place (every other method returns a fresh
            # array).  ``.copy()`` is backend-generic (numpy / cupy / jax).
            return E_in.copy()
        return angular_spectrum_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'sas':
        from .propagation import scalable_angular_spectrum_propagate
        if z is None:
            raise ValueError("propagate(method='sas'): z is required.")
        return scalable_angular_spectrum_propagate(
            E_in, z, wavelength, dx, **kwargs)

    if method == 'fresnel':
        from .propagation import fresnel_propagate
        if z is None:
            raise ValueError("propagate(method='fresnel'): z is required.")
        return fresnel_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'fraunhofer':
        from .propagation import fraunhofer_propagate
        if z is None:
            raise ValueError("propagate(method='fraunhofer'): z is required.")
        return fraunhofer_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'rs':
        from .propagation import rayleigh_sommerfeld_propagate
        if z is None:
            raise ValueError("propagate(method='rs'): z is required.")
        return rayleigh_sommerfeld_propagate(E_in, z, wavelength, dx, **kwargs)

    if method == 'gbd':
        from .gbd import propagate_gbd_freespace, propagate_gbd_through_prescription
        # v5.2.3 (AUDIT_V4_13_1 P1-A residual closure): resolve the
        # dispatcher's ``output_grid = (N_out, dx_out)`` contract
        # into the sub-propagator's new ``output_shape`` + ``output_dx``
        # kwargs before forwarding, so the sub-propagator's
        # v5.2.0 ``output_grid``-as-(Ny, Nx) DeprecationWarning shim
        # doesn't fire on a dispatcher-canonical call.  v5.2.0 closed
        # the rename at the sub-propagator surface; v5.2.3 closes the
        # forwarding so the dispatcher contract is end-to-end clean.
        _shape, _dx_out = _resolve_dispatcher_output_grid(
            method, output_grid, output_dx, E_in.shape)
        if prescription is None:
            if z is None:
                raise ValueError(
                    "propagate(method='gbd') without prescription requires z.")
            return propagate_gbd_freespace(
                E_in, dx, z=z, wavelength=wavelength,
                output_shape=_shape, output_dx=_dx_out,
                **kwargs,
            )
        return propagate_gbd_through_prescription(
            E_in, dx, prescription,
            wavelength=wavelength,
            output_shape=_shape, output_dx=_dx_out,
            **kwargs,
        )

    if method == 'hfpi':
        from .hfpi import (
            propagate_hfpi_freespace_aperture,
            propagate_hfpi_through_prescription,
        )
        if prescription is None:
            if 'aperture_radius' not in kwargs:
                raise ValueError(
                    "propagate(method='hfpi') without prescription "
                    "needs at least an aperture geometry "
                    "(aperture_radius=...).")
            # v5.2.5 (AUDIT_V5_2_3 P2-F1-1): thread the resolved
            # ``output_grid``/``output_dx`` through the freespace
            # branch too.  v5.2.3 fixed the through-prescription path
            # but the freespace branch silently dropped them.
            _shape, _dx_out = _resolve_dispatcher_output_grid(
                method, output_grid, output_dx, E_in.shape)
            if _shape is not None or _dx_out is not None:
                kwargs.setdefault('output_shape', _shape)
                kwargs.setdefault('output_dx', _dx_out)
            return propagate_hfpi_freespace_aperture(
                E_in, dx,
                wavelength=wavelength,
                **kwargs,
            )
        # v5.2.3 (AUDIT_V4_13_1 P1-A residual closure): same dispatcher
        # forwarding fix as the ``gbd`` branch above.
        _shape, _dx_out = _resolve_dispatcher_output_grid(
            method, output_grid, output_dx, E_in.shape)
        return propagate_hfpi_through_prescription(
            E_in, dx, prescription,
            wavelength=wavelength,
            output_shape=_shape, output_dx=_dx_out,
            **kwargs,
        )

    if method == 'hf':
        from .hf import (
            propagate_huygens_fresnel_freespace,
            propagate_huygens_fresnel_through_prescription,
        )
        if prescription is None:
            if z is None:
                raise ValueError(
                    "propagate(method='hf') without prescription requires z.")
            # v5.2.5 (AUDIT_V5_2_3 P2-F1-1): thread the resolved
            # ``output_grid``/``output_dx`` through the freespace
            # branch too.  v5.2.3 fixed the through-prescription path
            # but the freespace branch silently dropped them.
            _shape, _dx_out = _resolve_dispatcher_output_grid(
                method, output_grid, output_dx, E_in.shape)
            if _shape is not None or _dx_out is not None:
                kwargs.setdefault('output_shape', _shape)
                kwargs.setdefault('output_dx', _dx_out)
            return propagate_huygens_fresnel_freespace(
                E_in, z, wavelength, dx, **kwargs,
            )
        # v5.2.3 (AUDIT_V4_13_1 P1-A residual closure): same dispatcher
        # forwarding fix as the ``gbd`` / ``hfpi`` branches above.
        _shape, _dx_out = _resolve_dispatcher_output_grid(
            method, output_grid, output_dx, E_in.shape)
        return propagate_huygens_fresnel_through_prescription(
            E_in, dx, prescription,
            wavelength=wavelength,
            output_shape=_shape, output_dx=_dx_out,
            **kwargs,
        )

    if method == 'maslov':
        from ..elements.lenses import apply_real_lens_maslov
        if prescription is None:
            raise ValueError(
                "propagate(method='maslov') requires a prescription.")
        return apply_real_lens_maslov(
            E_in, prescription=prescription, wavelength=wavelength, dx=dx,
            **kwargs,
        )

    if method == 'asymptotic':
        from .asymptotic import fit_canonical_polynomials, propagate_modal_asymptotic
        # Caller may pass a pre-built fit via kwargs['fit'] or supply
        # a prescription that the dispatcher will fit on the fly.
        fit = kwargs.pop('fit', None)
        if fit is None:
            if prescription is None:
                raise ValueError(
                    "propagate(method='asymptotic') requires either "
                    "fit=... or a prescription.")
            fit_kwargs = kwargs.pop('fit_kwargs', {}) or {}
            fit = fit_canonical_polynomials(
                prescription, wavelength=wavelength, **fit_kwargs)
        return propagate_modal_asymptotic(fit, **kwargs)

    if method == 'mhs':
        from .mhs import MhsPipeline
        # Accept either a fully-built pipeline OR a list of subdomains.
        pipeline = kwargs.pop('pipeline', None)
        subdomains = kwargs.pop('subdomains', None)
        if pipeline is None and subdomains is None:
            raise ValueError(
                "propagate(method='mhs') requires either pipeline=... "
                "or subdomains=... .")
        if pipeline is None:
            pipeline = MhsPipeline(subdomains)
        return_intermediate = kwargs.pop('return_intermediate', False)
        return pipeline.run(E_in, return_intermediate=return_intermediate,
                            **kwargs)

    raise NotImplementedError(f"Method {method!r} is not implemented.")


# ============================================================================
# ASM-family auto-selector (asm_propagate) and advisor (which_propagator)
# ============================================================================

ASM_FAMILY = ('asm', 'asm_tilted', 'asm_mft', 'sas', 'fresnel', 'fraunhofer')


def _select_asm_variant(
    E_in,
    z: float,
    wavelength: float,
    dx: float,
    *,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    output_dx: Optional[float] = None,
    aperture_radius: Optional[float] = None,
) -> str:
    """Choose the best ASM-family propagator for the given geometry.

    Decision order:

    1. Output grid pitch requested AND different from input -> ``asm_mft``
       (Bluestein output sampling).
    2. Significant tilt (> 1e-6 rad) -> ``asm_tilted``.
    3. ``z >> L^2 / (N * lambda)`` (small Fresnel number) -> ``sas``
       for a scalable output pitch, else ``fraunhofer`` if extreme.
    4. Intermediate Fresnel number -> plain ``asm`` (the band-limited
       transfer function handles both near- and intermediate-field).

    The regime threshold uses the full grid extent ``L = N * dx`` as the
    transverse scale.  ``aperture_radius`` is accepted for signature
    parity with :func:`which_propagator` (which uses it for the *reported*
    Fresnel number) but is **not** consulted by the branch logic here --
    the selector deliberately keys on the grid extent so the choice is
    reproducible from the field shape alone.

    .. versionchanged:: 5.31
       Two audit-W9 fixes, both bringing this selector in line with rules its
       twin :func:`_auto_select_method` has carried since 4.12:

       * **Back-propagation (W9-2).**  ``z < 0`` can no longer select ``sas`` /
         ``fraunhofer``.  Those kernels are forward-only and raised on the sign
         of ``z``, so :func:`asm_propagate` -- which runs whatever this returns
         -- crashed for any back-propagation past ``2 * L^2/(N*lambda)``.
         MEASURED pre-fix at N=64, dx=2 um, lambda=633 nm (threshold
         4.0442e-4 m): ``z=-1.2133e-3`` -> ``sas`` ->
         ``"scalable_angular_spectrum_propagate: z must be > 0"``;
         ``z=-1.2133e-2`` -> ``fraunhofer`` -> the analogous raise.  Every
         ASM-family member (``asm`` / ``asm_tilted`` / ``asm_mft``) accepts
         either sign, so the negative-``z`` case now stays inside that set --
         the 4.12 B1-6 guard, ported.
       * **Dropped tilt (W9-3).**  The ``asm_mft`` branch sits ABOVE the tilt
         branch and :func:`angular_spectrum_propagate_mft` has no ``tilt_x`` /
         ``tilt_y`` parameter, so a tilt passed alongside ``output_dx``
         vanished in complete silence: MEASURED bit-identical output
         (``max|difference| = 0.0``) for ``tilt_x=0.05`` versus ``tilt_x=0.0``
         at N=64, dx=2 um, z=5e-4, output_dx=3e-6.  The precedence is kept
         (there is no tilted-MFT kernel to route to) but the collision now
         emits a :class:`UserWarning` -- the same call v5.30 made for the
         sibling case, the legacy ``'propagate_tilted'`` element ignoring
         ``elem['method']``.
    """
    has_tilt = (abs(float(tilt_x)) > 1e-6) or (abs(float(tilt_y)) > 1e-6)
    if output_dx is not None and abs(float(output_dx) - float(dx)) > 0:
        if has_tilt:
            warnings.warn(
                f"which_propagator/asm_propagate: an output pitch "
                f"(output_dx={float(output_dx):.6e} m, input dx="
                f"{float(dx):.6e} m) was requested TOGETHER with a carrier "
                f"tilt (tilt_x={float(tilt_x):.6g}, tilt_y="
                f"{float(tilt_y):.6g} rad).  The output-grid branch wins and "
                f"routes to 'asm_mft', whose kernel "
                f"(angular_spectrum_propagate_mft) has no tilt parameter, so "
                f"THE TILT IS DROPPED -- the result is bit-identical to the "
                f"untilted call (measured).  There is no tilted-MFT kernel: "
                f"either drop output_dx and let the tilt route to "
                f"'asm_tilted', or apply the carrier yourself (multiply by "
                f"exp(i*k*(tilt_x*x + tilt_y*y)) before the MFT call) and "
                f"pass tilt_x=tilt_y=0 to silence this.",
                UserWarning, stacklevel=3)
        return 'asm_mft'
    if has_tilt:
        return 'asm_tilted'
    # v5.31 (audit W9-2): the two far-field branches below call forward-only
    # kernels.  Restrict the regime test to z >= 0 so a back-propagation never
    # lands on one; plain ASM handles either sign.  (``asm_mft`` / ``asm_tilted``
    # above are sign-agnostic and keep their precedence.)
    if float(z) < 0:
        return 'asm'
    # Compare propagation distance to L^2 / (N * lambda) -- the
    # SAS-regime threshold.  We need an aperture radius or the grid
    # extent L to make this judgement; if neither is supplied, fall
    # back to plain ASM.
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    N = max(Ny, Nx)
    L = N * dx
    threshold = (L * L) / (N * wavelength)
    if abs(float(z)) > 20.0 * threshold:
        # Far-field-ish; Fraunhofer is closed-form and cheaper.
        return 'fraunhofer'
    if abs(float(z)) > 2.0 * threshold:
        # The beam has spread far enough that the SAS rescaling
        # pays off.
        return 'sas'
    return 'asm'


def which_propagator(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    *,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    output_dx: Optional[float] = None,
    aperture_radius: Optional[float] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Advise which ASM-family propagator to use without running one.

    Returns a dict with the chosen method name and a brief reason.
    Useful for documenting a design choice or surfacing the decision
    in a notebook / GUI.

    Parameters
    ----------
    E_in : ndarray
        Input field (only the shape and dtype are consulted).
    z, wavelength, dx : float
        Propagation geometry [m].
    tilt_x, tilt_y : float, optional
        Mean-direction tilt [rad].  Non-zero values steer the
        choice toward ``asm_tilted``.
    output_dx : float, optional
        Requested output pitch [m].  When different from ``dx``,
        steers toward ``asm_mft``.
    aperture_radius : float, optional
        Source aperture [m] used in the Fresnel-number heuristic.
    verbose : bool
        Print the decision to stdout (useful in interactive use).

    Returns
    -------
    advice : dict
        ``{'method': str, 'reason': str, 'fresnel_number': float}``.
    """
    method = _select_asm_variant(
        E_in, z, wavelength, dx,
        tilt_x=tilt_x, tilt_y=tilt_y,
        output_dx=output_dx, aperture_radius=aperture_radius)

    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    N = max(Ny, Nx)
    L = N * dx
    threshold = (L * L) / (N * wavelength)
    a = aperture_radius if aperture_radius is not None else (L / 2.0)
    if abs(float(z)) > 0:
        fn = (a * a) / (wavelength * abs(float(z)))
    else:
        fn = float('inf')

    reasons = {
        'asm':       'near/intermediate field; band-limited ASM is exact.',
        'asm_tilted':'mean propagation direction is tilted; use carrier-shifted ASM.',
        'asm_mft':   'output grid pitch != input; Bluestein output sampling.',
        'sas':       (f'z = {z!r} >> L^2/(N*lambda) = {threshold:.3g}; '
                       'scalable ASM rescales the output grid.'),
        'fraunhofer':'extreme far field; closed-form Fraunhofer is cheapest.',
    }
    advice = {
        'method': method,
        'reason': reasons.get(method, ''),
        'fresnel_number': float(fn),
    }
    if verbose:
        print(f"which_propagator -> {method}: {advice['reason']}")
    return advice


def asm_propagate(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    *,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    output_dx: Optional[float] = None,
    output_N: Optional[int] = None,
    aperture_radius: Optional[float] = None,
    bandlimit: bool = True,
    verbose: bool = False,
    **method_kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
    """Auto-select and run the best ASM-family propagator.

    Calls :func:`which_propagator` to pick between ``asm`` /
    ``asm_tilted`` / ``asm_mft`` / ``sas`` / ``fraunhofer`` based on
    the geometry, then dispatches to the chosen function.

    Parameters
    ----------
    E_in, z, wavelength, dx, tilt_x, tilt_y, output_dx, aperture_radius :
        Forwarded to :func:`which_propagator`.
    output_N : int, optional
        Output grid size when ``output_dx`` is given (required for the
        MFT-style sampler).  Defaults to the input grid size.
    bandlimit : bool
        Passed through to ASM-family propagators that accept it.
    verbose : bool
        Print the chosen method.
    **method_kwargs : dict
        Forwarded to the underlying propagator.

    Returns
    -------
    The chosen propagator's native return value (most return a bare
    ``ndarray``; the MFT variants return a 3-tuple).
    """
    advice = which_propagator(
        E_in, z, wavelength, dx,
        tilt_x=tilt_x, tilt_y=tilt_y,
        output_dx=output_dx, aperture_radius=aperture_radius,
        verbose=verbose)
    method = advice['method']

    from .propagation import (
        angular_spectrum_propagate,
        angular_spectrum_propagate_mft,
        angular_spectrum_propagate_tilted,
        fraunhofer_propagate,
        scalable_angular_spectrum_propagate,
    )

    if method == 'asm':
        return angular_spectrum_propagate(
            E_in, z, wavelength, dx, bandlimit=bandlimit,
            **method_kwargs)
    if method == 'asm_tilted':
        return angular_spectrum_propagate_tilted(
            E_in, z, wavelength, dx, tilt_x=tilt_x, tilt_y=tilt_y,
            bandlimit=bandlimit, **method_kwargs)
    if method == 'asm_mft':
        Ny, Nx = E_in.shape[-2], E_in.shape[-1]
        N_out = int(output_N) if output_N is not None else max(Ny, Nx)
        return angular_spectrum_propagate_mft(
            E_in, z, wavelength, dx, output_dx, N_out,
            bandlimit=bandlimit, **method_kwargs)
    if method == 'sas':
        return scalable_angular_spectrum_propagate(
            E_in, z, wavelength, dx, **method_kwargs)
    if method == 'fraunhofer':
        return fraunhofer_propagate(
            E_in, z, wavelength, dx, **method_kwargs)
    raise NotImplementedError(
        f"asm_propagate: internal error -- method {method!r} not "
        f"dispatched.")


__all__ = [
    'propagate', 'VALID_METHODS',
    'asm_propagate', 'which_propagator', 'ASM_FAMILY',
]
