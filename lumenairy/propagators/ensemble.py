"""
lumenairy.propagators.ensemble -- partial-coherence ensemble propagator.

Helper for iterating a Schell-family ``(n_realizations, Ny, Nx)``
ensemble through a coherent 2-D propagator and accumulating the
partial-coherence intensity.

The canonical Wolf coherence-theory result for a Schell-model source
is that the partially-coherent intensity at the output plane is the
ensemble-averaged squared modulus of the propagated realisations::

    I_partial(r) = < |E_k(r)|^2 >_k

This helper closes the half-shipped Schell-model workflow that the
v4.15.x release cycle introduced (ensembles returned by
:func:`create_gaussian_schell_source` etc., but no public helper to
propagate the ensemble and recover ``I_partial``).  Authored for
v4.16.1 closing the deep audit ``AUDIT_V4_16_0_DEEP_2026_05_19.md``
Part 5 P0-1 finding.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from ..backend import array_namespace


# Canonical string-name -> propagator-callable mapping for the
# ``propagator='asm'`` / ``'fresnel'`` / ... convenience surface.  The
# imports are intentionally lazy (inside :func:`_resolve_propagator`) to
# keep this module's import cost cheap and avoid pulling the propagator
# graph into Python at ``lumenairy`` import time.
_STRING_PROPAGATORS = ('asm', 'fresnel', 'fraunhofer', 'rs', 'sas')


def _resolve_propagator(propagator: Union[str, Callable]) -> Callable:
    """Resolve a string-name propagator into the underlying callable.

    Strings map to the canonical 2-D coherent-field propagators in
    :mod:`lumenairy.propagators.propagation`.  Callables are returned
    unchanged; the helper does not verify their signature shape
    (the callable is invoked once per realisation, so a mismatched
    signature surfaces immediately at first call).
    """
    if callable(propagator):
        return propagator
    if not isinstance(propagator, str):
        raise TypeError(
            f"propagate_ensemble: propagator must be a string in "
            f"{_STRING_PROPAGATORS!r} or a callable with signature "
            f"``propagator(E, dx=dx, wavelength=wavelength, **kwargs) "
            f"-> E_out``; got type={type(propagator).__name__}.")
    key = propagator.lower()
    if key == 'asm':
        from .propagation import angular_spectrum_propagate
        return angular_spectrum_propagate
    if key == 'fresnel':
        from .propagation import fresnel_propagate
        return fresnel_propagate
    if key == 'fraunhofer':
        from .propagation import fraunhofer_propagate
        return fraunhofer_propagate
    if key == 'rs':
        from .propagation import rayleigh_sommerfeld_propagate
        return rayleigh_sommerfeld_propagate
    if key == 'sas':
        from .propagation import scalable_angular_spectrum_propagate
        return scalable_angular_spectrum_propagate
    raise ValueError(
        f"propagate_ensemble: unknown propagator name {propagator!r}; "
        f"expected one of {_STRING_PROPAGATORS!r} or a callable.")


def _coerce_field_from_propagator_return(out: Any) -> Any:
    """Unwrap the per-realisation propagator return into a bare 2-D
    field array.

    Some propagators return a bare array (ASM, RS); others return a
    ``(E, dx_out, dy_out)`` tuple (Fresnel, Fraunhofer, SAS).
    :class:`PropagationResult` wrappers expose ``.field``.  Anything
    else raises a clear :class:`TypeError`.

    v4.16.2 (audit P1-NEW-F1-2): preserve the backend of the returned
    field.  Pre-fix every branch coerced through ``np.asarray(...)``,
    silently transferring CuPy / JAX returns to host NumPy.  Now we
    return the underlying array as-is (NumPy, CuPy, or eager
    ``jax.Array``); the per-realisation accumulator runs through the
    matching ``xp.*`` namespace.
    """
    # Bare backend array (NumPy / CuPy / JAX) -- duck-typed via .ndim +
    # .shape + .dtype.
    if (hasattr(out, 'ndim') and hasattr(out, 'shape')
            and hasattr(out, 'dtype')):
        return out
    # PropagationResult passthrough (opt-in wrapped return).
    field_attr = getattr(out, 'field', None)
    if (field_attr is not None and hasattr(field_attr, 'ndim')
            and hasattr(field_attr, 'shape')
            and hasattr(field_attr, 'dtype')):
        return field_attr
    # Tuple / list ``(E, dx_out, ...)`` -- accept the leading array.
    if isinstance(out, (tuple, list)) and len(out) >= 1:
        first = out[0]
        if (hasattr(first, 'ndim') and hasattr(first, 'shape')
                and hasattr(first, 'dtype')):
            return first
    raise TypeError(
        f"propagate_ensemble: unsupported propagator return type "
        f"{type(out).__name__}.  Expected a 2-D array, a "
        f"``(E, dx_out, ...)`` tuple, or a PropagationResult.  Use "
        f"a callable propagator that returns one of these forms.")


def propagate_ensemble(
    ensemble: np.ndarray,
    *,
    dx: float,
    wavelength: float,
    propagator: Union[Callable[..., np.ndarray], str] = 'asm',
    propagator_kwargs: Optional[Dict[str, Any]] = None,
    return_intensity: bool = True,
    return_ensemble: bool = False,
    show_progress: bool = False,
    **kwargs: Any,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Propagate a partial-coherence ensemble through a coherent
    propagator.

    The standard partial-coherence workflow: v4.15.1's Schell-family
    factories return ``(n_realizations, Ny, Nx)`` ensembles; coherent
    propagators (:func:`angular_spectrum_propagate`,
    :func:`apply_real_lens`, etc.) expect 2-D inputs and reject 3-D
    ensembles via the v4.15.3 ``_check_2d_scalar_field`` helper.
    This helper iterates each realisation through the propagator and
    averages the intensities to recover the partial-coherence
    intensity::

        I_partial(r) = < |E_k(r)|^2 >_k

    This is the canonical Wolf coherence-theory result for a
    Schell-model source -- see Goodman, *Statistical Optics*, Sec 5.5
    or Mandel & Wolf, *Optical Coherence and Quantum Optics*, Sec 4.3.

    Parameters
    ----------
    ensemble : ndarray, shape ``(n_realizations, Ny, Nx)``, complex
        Schell-family ensemble from
        :func:`create_gaussian_schell_source` /
        :func:`create_schell_model_source` /
        :func:`create_annular_incoherent_source` (or any user-supplied
        equivalent).  Must have ``ndim == 3``; the leading axis is the
        ensemble index.
    dx : float
        Grid spacing [m].  Forwarded to the propagator.
    wavelength : float
        Vacuum wavelength [m].  Forwarded to the propagator.
    propagator : callable or {'asm', 'fresnel', 'fraunhofer', 'rs', 'sas'}
        Coherent 2-D-field propagator.  String form looks up the
        canonical propagator from :mod:`lumenairy.propagators.propagation`;
        callable signature must be
        ``propagator(E, dx=dx, wavelength=wavelength, **kwargs) -> E_out``.
        The callable's return may be a bare 2-D ndarray, a
        ``(E, dx_out, ...)`` tuple, or a :class:`PropagationResult`.
    propagator_kwargs : dict, optional
        Extra kwargs forwarded to the propagator.  Merged with the
        catch-all ``**kwargs`` (``propagator_kwargs`` takes precedence
        on key collisions).
    return_intensity : bool, default True
        If True, return ``I_partial`` (real-valued ndarray of shape
        ``(Ny, Nx)``).
    return_ensemble : bool, default False
        If True, also return the propagated ensemble of shape
        ``(n_realizations, Ny, Nx)``.  Storing every output realisation
        costs ``n_realizations * Ny * Nx * 16`` bytes (complex128); for
        large ensembles this can dominate memory.  Defaults to False
        so the intensity-only path stays memory-efficient.
    show_progress : bool, default False
        If True and the propagator accepts a ``progress`` kwarg, pass
        the library's default progress-bar callback (currently a
        no-op; reserved for future progress-tracking propagators).
    **kwargs : any
        Additional kwargs forwarded to the propagator (e.g.
        ``z=0.1`` for free-space ASM).  ``propagator_kwargs`` takes
        precedence on key collisions.

    Returns
    -------
    I_partial : ndarray, real, shape ``(Ny, Nx)``
        Partial-coherence intensity (default return).  Only returned
        when ``return_intensity=True``.
    ensemble_out : ndarray, complex, shape ``(n_realizations, Ny, Nx)``
        Propagated ensemble.  Only returned when
        ``return_ensemble=True``.  Returned alongside ``I_partial`` as
        a tuple ``(I_partial, ensemble_out)`` when both flags are set.
        When ``return_intensity=False`` and ``return_ensemble=True``,
        returned bare.

    Raises
    ------
    ValueError
        If ``ensemble.ndim != 3`` or both ``return_intensity`` and
        ``return_ensemble`` are False (no return).
    TypeError
        If ``propagator`` is neither a string nor a callable.

    Examples
    --------
    Propagate a Gaussian-Schell source through 10 cm of free space::

        >>> import numpy as np
        >>> import lumenairy as la
        >>> ens, dx, dy, wl = la.create_gaussian_schell_source(
        ...     N=128, dx=2e-6, wavelength=633e-9,
        ...     w0=20e-6, sigma_g=5e-6,
        ...     n_realizations=64, return_kind='ensemble')
        >>> I_partial = la.propagate_ensemble(
        ...     ens, dx=dx, wavelength=wl,
        ...     propagator='asm', z=0.1)
        >>> I_partial.shape
        (128, 128)
        >>> bool(np.all(np.isreal(I_partial)))
        True

    Pass a callable for a custom propagator::

        >>> def my_lens_chain(E, *, dx, wavelength, z=0.0):
        ...     E1 = la.apply_thin_lens(E, f=50e-3, wavelength=wavelength, dx=dx)
        ...     return la.angular_spectrum_propagate(
        ...         E1, z, wavelength, dx)
        >>> I_focused = la.propagate_ensemble(
        ...     ens, dx=dx, wavelength=wl,
        ...     propagator=my_lens_chain, z=50e-3)

    Notes
    -----
    * Memory-efficient by default: ``return_ensemble=False`` accumulates
      ``|E_out|^2`` into a single ``(Ny, Nx)`` real array, never
      materialising the full propagated ensemble.
    * The intensity normalisation matches the standard Wolf result --
      each realisation's ``|E_k|^2`` is summed and divided by
      ``n_realizations``.  For a Schell-model input the input ensemble
      satisfies ``< |E_k|^2 >_k = I_source(r)`` and the output
      satisfies ``< |E_out_k|^2 >_k = I_partial(r)``.
    * For tuple-returning propagators (Fresnel / Fraunhofer / SAS) only
      the leading complex field is consumed; the output-pitch elements
      of the tuple are ignored.  Callers needing the output pitch
      should call the propagator directly on one realisation to recover
      it.
    * Backend preservation (v4.16.2 / audit P1-NEW-F1-2): when
      ``ensemble`` is a CuPy or JAX array, the accumulator runs on the
      same backend.  Whether the per-realisation propagator preserves
      that backend depends on the propagator: the canonical
      ``angular_spectrum_propagate`` / ``fresnel_propagate`` accept and
      return the input's backend; user-supplied callables are
      responsible for their own backend handling.
    """
    # v4.16.2 (audit P3-NEW-F1-2): explicit kwargs-collision check.
    # Pre-fix a caller passing ``propagator_kwargs={'dx': ...}`` plus a
    # positional ``dx=...`` triggered the bare Python ``TypeError: got
    # multiple values for keyword argument 'dx'`` -- correct but
    # confusing.  Raise a domain-level ``ValueError`` with a clear
    # remediation pointer.
    if propagator_kwargs is not None:
        collisions = (set(propagator_kwargs)
                      & {'dx', 'wavelength'})
        if collisions:
            raise ValueError(
                f"propagate_ensemble: propagator_kwargs contains keys "
                f"that collide with positional arguments: "
                f"{sorted(collisions)}.  Pass dx / wavelength only via "
                f"the top-level kwargs, not propagator_kwargs.")

    # v4.16.2 (audit P1-NEW-F1-2): backend-preserving dispatch.
    # Pre-fix the duck-typed ``np.asarray(ensemble)`` fallback below
    # silently transferred CuPy ensembles to host NumPy and forced
    # concretisation of JAX arrays, defeating any GPU / autodiff
    # workflow the user built upstream.  Now we detect the backend via
    # ``array_namespace`` and run the accumulator on the same xp.
    # Inputs that aren't a NumPy / CuPy / JAX array (e.g. a Python
    # list of 2-D arrays) fall back to ``np.asarray``; the warning
    # below documents the coercion.
    if not (hasattr(ensemble, 'ndim') and hasattr(ensemble, 'shape')
            and hasattr(ensemble, 'dtype')):
        try:
            ensemble = np.asarray(ensemble)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                f"propagate_ensemble: ensemble must be a 3-D array; "
                f"got type={type(ensemble).__name__!r} which cannot be "
                f"coerced to an array ({exc})."
            ) from exc
    try:
        xp = array_namespace(ensemble)
    except TypeError:
        # Mixed-backend or unrecognised; fall back to NumPy and coerce.
        ensemble = np.asarray(ensemble)
        xp = np

    if ensemble.ndim != 3:
        raise ValueError(
            f"propagate_ensemble: ensemble must be a 3-D array of shape "
            f"(n_realizations, Ny, Nx); got ndim={ensemble.ndim}, "
            f"shape={ensemble.shape}.  If you passed a single 2-D "
            f"realisation, wrap it as ``ensemble[None, ...]`` or call "
            f"the propagator directly.")
    if not (return_intensity or return_ensemble):
        raise ValueError(
            "propagate_ensemble: at least one of `return_intensity` "
            "or `return_ensemble` must be True (otherwise the call "
            "produces no output).")
    # v4.16.2 (audit P3-NEW-F1-1): reject empty ensembles cleanly.
    # ``shape=(0, Ny, Nx)`` passes the ndim check above; pre-fix the
    # downstream ``I_acc / float(0)`` raised an opaque
    # ``ZeroDivisionError`` (or NaN-poisoned the result depending on
    # the dtype).  Surface the empty-ensemble case as a domain-level
    # ``ValueError`` at the entry point.
    if ensemble.shape[0] == 0:
        raise ValueError(
            "propagate_ensemble: empty ensemble (n_realizations=0).  "
            "The partial-coherence intensity ``< |E_k|^2 >_k`` is "
            "undefined without at least one realisation.")

    fn = _resolve_propagator(propagator)
    kw = dict(kwargs)
    if propagator_kwargs:
        kw.update(propagator_kwargs)
    if show_progress and 'progress' not in kw:
        # Reserved for future progress-tracking propagators -- safe to
        # set when accepted, harmless when ignored (most propagators do
        # not currently consume it).  Pass a noop to keep the kwarg
        # forwarding cheap and idempotent.
        kw['progress'] = None

    n_real = ensemble.shape[0]
    Ny, Nx = ensemble.shape[1], ensemble.shape[2]

    # Map the input's complex dtype to its real counterpart for the
    # accumulator.  Falls through to ``float64`` for non-complex or
    # exotic dtypes.  Using the matching real dtype preserves backend
    # parity (e.g. JAX's complex64 -> float32 default; CuPy's
    # complex128 -> float64).
    in_dtype = getattr(ensemble, 'dtype', None)
    try:
        real_dtype = np.dtype(in_dtype).type(0).real.dtype
    except (TypeError, ValueError):
        # v4.16.2 (pre-v5.0 prep): honour set_default_real_dtype for
        # the no-input-dtype fallback path.  Was hardcoded to
        # ``np.float64`` pre-v4.16.2.
        from .propagation import get_default_real_dtype
        real_dtype = get_default_real_dtype()

    # Accumulator buffer for the partial-coherence intensity.  Built
    # on the matching xp so GPU / JAX paths stay on the backend.
    I_acc: Any = None
    if return_intensity:
        I_acc = xp.zeros((Ny, Nx), dtype=real_dtype)

    ensemble_out: Any = None
    if return_ensemble:
        # Same xp + dtype as the input ensemble.  JAX is immutable so
        # we collect realisations in a list and stack at the end; the
        # NumPy / CuPy fast path uses an in-place ``empty`` buffer.
        if xp is np or (hasattr(xp, '__name__')
                        and 'cupy' in xp.__name__):
            ensemble_out = xp.empty(ensemble.shape, dtype=ensemble.dtype)
        else:
            ensemble_out = []  # JAX -> list-then-stack

    for k in range(n_real):
        out_k = fn(ensemble[k], dx=dx, wavelength=wavelength, **kw)
        E_out_k = _coerce_field_from_propagator_return(out_k)
        if E_out_k.shape != (Ny, Nx):
            # The propagator changed the output grid shape (e.g. SAS
            # with implicit rescale, or Fresnel-MFT to a different N).
            # Bail loudly rather than silently averaging mismatched
            # shapes -- partial-coherence averaging requires every
            # realisation to land on the same output grid.
            raise ValueError(
                f"propagate_ensemble: propagator returned shape "
                f"{E_out_k.shape} for realisation {k}, but the input "
                f"realisations were {(Ny, Nx)}.  "
                f"propagate_ensemble requires every realisation to "
                f"land on the same output grid (so |E|^2 can be "
                f"averaged).  Either choose a propagator that "
                f"preserves grid shape (e.g. 'asm') or call the "
                f"propagator manually on a single realisation to "
                f"discover the output shape, then build an "
                f"appropriate accumulator yourself.")
        if I_acc is not None:
            # |E|^2 squared modulus.  JAX is immutable so we re-bind
            # (cheap: the xp.abs+square op is fused inside XLA); the
            # NumPy / CuPy paths fold into a single in-place op.
            sq = xp.abs(E_out_k) ** 2
            if xp is np or (hasattr(xp, '__name__')
                            and 'cupy' in xp.__name__):
                I_acc += sq
            else:
                I_acc = I_acc + sq
        if ensemble_out is not None:
            if isinstance(ensemble_out, list):
                ensemble_out.append(E_out_k)
            else:
                ensemble_out[k] = E_out_k

    if isinstance(ensemble_out, list):
        ensemble_out = xp.stack(ensemble_out, axis=0)

    if I_acc is not None:
        # Ensemble average: < |E_k|^2 >_k = sum / n_realizations.
        # This is the canonical Wolf result for partial-coherence
        # propagation of a Schell-family source through a coherent
        # propagator.
        I_partial: Any = I_acc / float(n_real)

    if return_intensity and return_ensemble:
        return I_partial, ensemble_out  # type: ignore[return-value]
    if return_intensity:
        return I_partial  # type: ignore[return-value]
    # return_ensemble only:
    return ensemble_out  # type: ignore[return-value]


__all__ = ['propagate_ensemble']
