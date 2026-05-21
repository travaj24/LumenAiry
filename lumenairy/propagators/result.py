"""
lumenairy.propagators.result -- unified PropagationResult container.

A lightweight, opt-in result type that bundles a propagated field with
its sampling metadata and (when applicable) the per-stage history of a
multi-step propagation.

PropagationResult is **additive** -- every existing propagator continues
to return its native shape (bare ndarray, list of (surface, field), or
(E, intermediates) tuple) by default.  Pass ``return_result=True`` to
opt into the structured container instead.  This keeps fast-loop and
notebook code unaffected while letting downstream consumers (plotting,
storage, replay) take a single uniform input.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class PropagationResult:
    """Unified result container for any propagator output.

    Attributes
    ----------
    field : ndarray (complex)
        The exit-plane field on the (Ny, Nx) grid.
    dx : float
        Sample spacing of ``field`` along x [m].
    dy : float, optional
        Sample spacing of ``field`` along y [m].  Defaults to ``dx``
        for square-grid kernels (preserves back-compat for ASM, RS,
        and any other propagator that returns a square pitch).
        Anamorphic Fresnel / Fraunhofer / SAS kernels return a
        distinct ``dy_out``; v4.13.0 (audit L3) threads that value
        through :func:`_coerce_field` so the wrapped result no longer
        silently discards the y-axis pitch.
    wavelength : float
        Vacuum wavelength used for the propagation [m].
    z : float, optional
        Z-coordinate of the exit plane (when meaningful).
    method : str, optional
        Method label ('asm', 'gbd', 'mhs', 'system', ...).  Free-form
        -- callers may use sub-method names like ``'mhs:gbd'``.
    history : list, optional
        Per-stage history.  Each entry is a 3-tuple
        ``(label, field, dx)`` for plane-walking propagators (MHS,
        propagate_through_system).  ``None`` for single-stage
        propagators.
    metadata : dict
        Free-form key/value bag for propagator-specific outputs (fit
        objects, residuals, RNG seeds, etc.).
    intermediates : list, optional
        Legacy shorthand: same as ``[h[1] for h in history]`` when
        ``history`` is populated.  Auto-derived from ``history`` if
        not supplied; preserved separately so older code that
        unpacked ``(E, intermediates)`` from
        :func:`propagate_through_system` keeps working when the
        caller re-attaches via ``PropagationResult.field`` /
        ``.intermediates``.
    """
    field: np.ndarray
    dx: float
    wavelength: float
    z: Optional[float] = None
    method: Optional[str] = None
    history: Optional[List[Tuple[str, np.ndarray, float]]] = None
    metadata: Dict[str, Any] = _dc_field(default_factory=dict)
    intermediates: Optional[List[np.ndarray]] = None
    # v4.13.0 (audit L3): anamorphic pitch on the y-axis.  Defaults to
    # ``dx`` for square-grid kernels, preserving back-compat for ASM /
    # RS / sources / Fresnel-square callers.  Anamorphic Fresnel kernels
    # (Fresnel / Fraunhofer / SAS with dx_in != dy_in) return distinct
    # ``dx_out, dy_out`` triples and the dispatcher now forwards both
    # through ``_coerce_field``.
    dy: Optional[float] = None

    def __post_init__(self) -> None:
        # v4.13.0 (audit L3): square-grid default.
        if self.dy is None:
            self.dy = self.dx
        # Auto-derive intermediates from history if the caller didn't
        # supply both.  This lets PropagationResult be a drop-in for
        # callers that did ``E, intermediates = propagate_through_system(...)``
        # by reading ``.field`` and ``.intermediates``.
        if self.intermediates is None and self.history is not None:
            self.intermediates = [h[1] for h in self.history]

    @property
    def dx_out(self) -> float:
        """Alias for :attr:`dx`.  Matches the naming convention used
        by the tuple-returning kernels (``E_out, dx_out, dy_out``)
        that audit L3 brings into the wrapped-result contract."""
        return self.dx

    @property
    def dy_out(self) -> float:
        """Alias for :attr:`dy`.  Matches the naming convention used
        by the tuple-returning kernels."""
        return self.dy if self.dy is not None else self.dx

    # -- Convenience interop -----------------------------------------

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape of ``field`` -- ``(Ny, Nx)``."""
        return tuple(self.field.shape[-2:])

    def __iter__(self) -> Any:
        """Tuple-unpack as ``(field, intermediates)`` for backward
        compat with callers that did
        ``E, intermediates = propagate_through_system(...)``.
        """
        yield self.field
        yield self.intermediates if self.intermediates is not None else []

    def __array__(self, dtype: Optional[Any] = None) -> np.ndarray:
        """Allow ``np.asarray(result)`` to fall through to the field
        array, so PropagationResult slots in everywhere a bare
        ndarray is accepted."""
        return np.asarray(self.field, dtype=dtype) if dtype is not None \
            else np.asarray(self.field)

    def labels(self) -> List[str]:
        """Plane labels in order, or ``[]`` if no history."""
        if self.history is None:
            return []
        return [h[0] for h in self.history]

    def to_source(self) -> Any:
        """Wrap ``field`` in a :class:`lumenairy.sources.Source` for
        chained propagation.  Wavelength, dx, and (v4.13.0 audit L3)
        dy are inherited; the Source name is set from ``method`` when
        available."""
        from ..sources.core import Source
        return Source(
            E=self.field, dx=self.dx, dy=self.dy, wavelength=self.wavelength,
            name=f'PropagationResult({self.method})' if self.method
                 else 'PropagationResult',
        )

    def __repr__(self) -> str:
        method_part = (f', method={self.method!r}'
                        if self.method is not None else '')
        z_part = (f', z={self.z:.3g}m' if self.z is not None else '')
        hist_part = (f', history({len(self.history)})'
                      if self.history else '')
        return (f"PropagationResult(shape={self.shape}, "
                f"dx={self.dx:.3g}m, "
                f"wavelength={self.wavelength*1e9:.1f}nm"
                f"{method_part}{z_part}{hist_part})")


__all__ = ['PropagationResult']
