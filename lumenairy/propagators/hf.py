"""
lumenairy.propagators.hf -- Van-Vleck-corrected deterministic
Huygens-Fresnel propagator.

Implements the direct Huygens-Fresnel diffraction integral with the
**Van Vleck density correction** in the integrand:

    E_out(s2) = integral E_in(s1) sqrt(|det d2 Phi / d s1 d s2|)
                * exp(2 pi i Phi(s1, s2)) d^2 s1

The Van Vleck factor makes the bare HF integrand energy-conserving
on non-conjugate output planes and keeps it finite at the focus.

See ``REFERENCES.txt`` Sections A and B for the foundational
publications.

Author: Andrew Traverso
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array


# v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): output-grid kwarg semantics
# disambiguation; see :mod:`lumenairy.propagators.gbd` for full
# rationale.
def _resolve_output_shape(
    output_shape: Optional[Tuple[int, int]],
    output_grid: Optional[Any],
    *,
    fn_name: str,
    default_shape: Tuple[int, int],
) -> Tuple[int, int]:
    """Resolve the (Ny, Nx) output shape from the v5.2 ``output_shape``
    kwarg and the deprecated ``output_grid`` legacy kwarg."""
    if output_shape is not None and output_grid is not None:
        raise ValueError(
            f"{fn_name}: both ``output_shape`` and ``output_grid`` were "
            f"provided.  Pass only ``output_shape=(Ny, Nx)`` (v5.2+) or "
            f"the dispatcher's ``output_grid=(N_out, dx_out)`` form via "
            f"``propagate(method=...)``.")
    if output_shape is not None:
        return (int(output_shape[0]), int(output_shape[1]))
    if output_grid is not None:
        warnings.warn(
            f"{fn_name}: the ``output_grid`` kwarg now (v5.2+) means "
            f"the dispatcher's ``(N_out, dx_out)`` grid spec; on "
            f"sub-propagators it has been renamed to ``output_shape`` "
            f"for the ``(Ny, Nx)`` shape-only meaning.  Pass "
            f"``output_shape=(Ny, Nx)`` to silence this warning, or "
            f"call via ``propagate(method='hf', output_grid=(N_out, "
            f"dx_out), ...)`` if you actually want grid resampling.",
            DeprecationWarning, stacklevel=3,
        )
        return (int(output_grid[0]), int(output_grid[1]))
    return default_shape


def propagate_huygens_fresnel(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    **kwargs: Any,
) -> np.ndarray:
    """Canonical-order Huygens-Fresnel free-space propagation.

    Argument order ``(E_in, z, wavelength, dx)`` matches
    :func:`angular_spectrum_propagate`, :func:`propagate_gbd`, and
    :func:`propagate_hfpi`.  This is the recommended entry point for
    new code; the trio ``propagate_huygens_fresnel_freespace`` /
    ``_with_opl_callable`` / ``_through_prescription`` is retained
    for specialised use cases.

    Internally delegates to
    :func:`propagate_huygens_fresnel_freespace`.
    """
    return propagate_huygens_fresnel_freespace(
        E_in, z, wavelength, dx, **kwargs)


def propagate_huygens_fresnel_freespace(
    E_in: np.ndarray,
    z: float,
    wavelength: float,
    dx: float,
    *,
    dy: Optional[float] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    **kwargs: Any,
) -> np.ndarray:
    """Free-space Huygens-Fresnel propagation with the standard
    ``1 / (i lambda z)`` Van Vleck factor.

    Equivalent to :func:`lumenairy.propagation.rayleigh_sommerfeld_propagate`;
    re-exported here for API consistency with the other ``hf.*``
    entry points.

    v5.3 (AUDIT_V5_2_5 P1-1 closure): ``output_shape`` and
    ``output_dx`` kwargs are accepted and honored via a post-kernel
    ``resample_field`` step (matches the v5.2.3 MHS substantive-
    resampling pattern at ``mhs.py:573-611``).  The underlying
    ``rayleigh_sommerfeld_propagate`` kernel returns on the input
    grid; the resample step bridges to the caller-requested output
    grid.  v5.2.5 routed these kwargs from the dispatcher into this
    function but the v5.2.5 pass-through to
    ``rayleigh_sommerfeld_propagate`` raised ``TypeError`` because
    the RS kernel does not accept either kwarg.  v5.3 fixes the
    pass-through by handling the resample here instead of
    forwarding to the kernel.

    Return type
    -----------
    When neither ``output_shape`` nor ``output_dx`` is given (the
    common pass-through case), returns the bare ``ndarray`` -- same
    contract as the underlying RS kernel.

    When ``output_shape`` or ``output_dx`` IS given (the v5.3
    resample path), returns a ``(E_out, dx_out)`` 2-tuple matching
    the ``resample_field`` contract -- the call has changed the
    grid spacing and the caller needs to know the new pitch.
    """
    from .propagation import rayleigh_sommerfeld_propagate, resample_field
    E_native = rayleigh_sommerfeld_propagate(
        E_in, z, wavelength, dx, dy=dy, **kwargs,
    )
    if output_shape is None and output_dx is None:
        return E_native

    # Resample to the requested output grid (matches MHS pattern).
    target_dx = float(output_dx) if output_dx is not None else float(dx)
    if output_shape is None:
        # Same shape as input; only the pitch changed.
        N_out = E_native.shape[-1]
    else:
        if len(output_shape) != 2:
            raise ValueError(
                f"propagate_huygens_fresnel_freespace: output_shape "
                f"must be a (Ny, Nx) tuple of two ints; got "
                f"{output_shape!r}.")
        Ny, Nx = int(output_shape[0]), int(output_shape[1])
        if Ny != Nx:
            raise ValueError(
                f"propagate_huygens_fresnel_freespace: non-square "
                f"output_shape ({Ny}, {Nx}) not supported -- the "
                f"underlying resample_field assumes a square "
                f"target grid.  Either request a square shape or "
                f"call ``rayleigh_sommerfeld_propagate`` + your own "
                f"resampler directly.")
        N_out = Ny

    # v5.4 (audit P2): same-shape short-circuit -- mirrors mhs.py:583-587.
    # If input grid already matches the requested target grid (same N
    # and same dx within tight tolerance), skip ``resample_field``
    # entirely.  ``map_coordinates`` introduces a small power drift
    # at the edges even when the grids nominally match; the
    # short-circuit guarantees bit-for-bit native-kernel return.
    xp = array_namespace(E_native)
    N_in = int(E_native.shape[-1])
    if N_in == int(N_out) and np.isclose(
            float(dx), float(target_dx), rtol=1e-12):
        return E_native, target_dx

    E_resampled, dx_resampled = resample_field(
        E_native, dx, target_dx, N_out)

    # v5.4 (audit P2): Parseval renorm to restore total power -- mirrors
    # mhs.py:602-606.  Bicubic ``map_coordinates`` interpolation
    # introduces a small power drift (mainly at edges where the
    # resample crops or extends the support); rescale by
    # sqrt(p_in / p_out) so the resample is L2-energy preserving.
    p_in = float(xp.sum(xp.abs(E_native) ** 2)) * (float(dx) ** 2)
    p_out = float(xp.sum(xp.abs(E_resampled) ** 2)) * (float(dx_resampled) ** 2)
    if p_out > 0.0 and p_in > 0.0:
        scale = float(xp.sqrt(p_in / p_out))
        E_resampled = E_resampled * scale
    # Preserve dtype (resample_field promotes complex64 -> complex128
    # via map_coordinates' float64 output).
    if E_resampled.dtype != E_native.dtype:
        E_resampled = E_resampled.astype(E_native.dtype)
    return E_resampled, dx_resampled


def propagate_huygens_fresnel_with_opl_callable(
    E_in: np.ndarray,
    *,
    opl_fn: Callable,
    output_grid_x: np.ndarray,
    output_grid_y: np.ndarray,
    input_grid_dx: float,
    wavelength: float,
    apply_van_vleck: bool = True,
    finite_diff_step: float = 1e-9,
    chunk_output: int = 64,
) -> np.ndarray:
    """Evaluate the HF integral for a user-supplied OPL callable
    ``Phi(s1, s2)``.

    Computes::

        E(s2) = sum over input pixels of
                E_in(s1) * sqrt(|det d2 Phi / d s1 d s2|)
                * exp(2 pi i Phi(s1, s2)) * (d s1)^2

    where the cross-Hessian determinant is evaluated by central
    differences on the supplied callable.
    """
    xp = array_namespace(E_in)

    Ny_in, Nx_in = E_in.shape[-2], E_in.shape[-1]
    # v4.12.0 (B1-10): switch from cell-centred `(arange(N) - N/2 + 0.5)*dx`
    # to pixel-centred `(arange(N) - N/2)*dx`, matching the library-wide
    # convention (ASM, Fresnel, RS, sources, ``apply_fresnel_curvature``).
    # The OPL callable is evaluated on input-plane coordinates so they
    # must match the grid that ``E_in`` was sampled on by upstream
    # propagators / source builders.
    s1_x = (xp.arange(Nx_in, dtype=xp.float64) - Nx_in / 2) * input_grid_dx
    s1_y = (xp.arange(Ny_in, dtype=xp.float64) - Ny_in / 2) * input_grid_dx
    S1X, S1Y = xp.meshgrid(s1_x, s1_y, indexing='xy')

    Ny_out = output_grid_y.shape[0]
    Nx_out = output_grid_x.shape[0]
    # 4.10: force a complex dtype so a real-valued E_in (e.g. a pure
    # intensity mask) doesn't silently strip the imaginary part of the
    # HF kernel during the multiply.  Pre-4.10 produced a real-valued
    # "field" with the imaginary half summed into nothing.
    if xp.iscomplexobj(E_in):
        out_dtype = E_in.dtype
    elif E_in.dtype == xp.float64:
        out_dtype = xp.complex128
    else:
        out_dtype = xp.complex64
    out = xp.zeros((Ny_out, Nx_out), dtype=out_dtype)
    pixel_area = input_grid_dx * input_grid_dx
    h = float(finite_diff_step)

    n_out = Ny_out * Nx_out
    flat_x = xp.reshape(xp.broadcast_to(output_grid_x[None, :],
                                        (Ny_out, Nx_out)), (-1,))
    flat_y = xp.reshape(xp.broadcast_to(output_grid_y[:, None],
                                        (Ny_out, Nx_out)), (-1,))

    for start in range(0, n_out, chunk_output):
        end = min(start + chunk_output, n_out)
        for k in range(start, end):
            s2x = float(flat_x[k]) if hasattr(flat_x[k], '__float__') else flat_x[k]
            s2y = float(flat_y[k]) if hasattr(flat_y[k], '__float__') else flat_y[k]

            phi = opl_fn(S1X, S1Y, s2x, s2y)

            if apply_van_vleck:
                pxx = (
                    opl_fn(S1X + h, S1Y, s2x + h, s2y)
                    - opl_fn(S1X + h, S1Y, s2x - h, s2y)
                    - opl_fn(S1X - h, S1Y, s2x + h, s2y)
                    + opl_fn(S1X - h, S1Y, s2x - h, s2y)
                ) / (4 * h * h)
                pyy = (
                    opl_fn(S1X, S1Y + h, s2x, s2y + h)
                    - opl_fn(S1X, S1Y + h, s2x, s2y - h)
                    - opl_fn(S1X, S1Y - h, s2x, s2y + h)
                    + opl_fn(S1X, S1Y - h, s2x, s2y - h)
                ) / (4 * h * h)
                pxy = (
                    opl_fn(S1X + h, S1Y, s2x, s2y + h)
                    - opl_fn(S1X + h, S1Y, s2x, s2y - h)
                    - opl_fn(S1X - h, S1Y, s2x, s2y + h)
                    + opl_fn(S1X - h, S1Y, s2x, s2y - h)
                ) / (4 * h * h)
                pyx = (
                    opl_fn(S1X, S1Y + h, s2x + h, s2y)
                    - opl_fn(S1X, S1Y + h, s2x - h, s2y)
                    - opl_fn(S1X, S1Y - h, s2x + h, s2y)
                    + opl_fn(S1X, S1Y - h, s2x - h, s2y)
                ) / (4 * h * h)
                det = pxx * pyy - pxy * pyx
                density = xp.sqrt(xp.abs(det))
            else:
                density = 1.0

            # 4.10: cast to the complex output dtype, not E_in.dtype
            # (which may be real -- see comment above the out-array
            # allocation).  Pre-4.10 a real E_in stripped the imag
            # part of the kernel before the multiply.
            kernel = xp.exp(2j * float(np.pi) * phi).astype(out_dtype)
            integrand = E_in * density * kernel
            iy = k // Nx_out
            ix = k % Nx_out
            out_value = xp.sum(integrand) * pixel_area
            if is_jax_array(E_in):
                out = out.at[iy, ix].set(out_value)
            else:
                out[iy, ix] = out_value

    # 4.11.2: apply the Van Vleck-Morette asymptotic prefactor
    # (2π)^(-d/2)·i^(-d/2) for d=2, which is -i/(2π).  The 2π part is
    # absorbed in the Phi convention (phase = exp(2πi Phi)), leaving
    # the global ``-1j`` Maslov factor.  The sibling
    # :func:`propagate_hf_chebyshev_quadrature` already applies this
    # (v4.10 C-AS-2 fix); without it the OPL-callable variant is 90°
    # out of phase with the Fresnel kernel ``1/(iλz) = -i/(λz)``,
    # producing incoherent superposition when stacking with
    # ASM/Fresnel outputs.
    out = out * (-1j)
    return out


# ============================================================================
# Prescription-aware HF (Van-Vleck-corrected, via fit_canonical_polynomials)
# ============================================================================

def propagate_huygens_fresnel_through_prescription(
    E_in: np.ndarray,
    dx: float,
    prescription: Dict[str, Any],
    *,
    wavelength: float,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    method: str = 'asymptotic',
    source_lg_p_max: int = 3,
    source_lg_ell_max: int = 3,
    source_lg_amp_threshold: float = 1e-6,
) -> np.ndarray:
    """End-to-end Van-Vleck-corrected HF through a sequential
    prescription.

    Two evaluation modes:

    * ``method='asymptotic'`` (default) -- evaluates the HF
      integral in the leading-order saddle-point (Van Vleck)
      asymptotic limit by routing to
      :func:`lumenairy.propagators.asymptotic.propagate_modal_asymptotic`.
      Closed-form, fast (~milliseconds), accurate for most
      well-conditioned refractive systems and for output planes
      that are not inside a fold caustic.

    * ``method='direct'`` -- direct 2-D quadrature of the HF integral
      using the Chebyshev polynomial fit of ``Phi(s2, v2)`` and
      ``s1(s2, v2)`` from
      :func:`lumenairy.propagators.asymptotic.fit_canonical_polynomials`.
      Includes the Van Vleck density factor
      ``sqrt(|det d2 Phi / d s1 d s2|)``.  Slower (~seconds at
      moderate output-grid sizes) but does not assume the saddle-
      point approximation.

    Parameters
    ----------
    E_in : array (Ny, Nx) complex
        Source-plane field.
    dx : float
        Source-grid pitch (m).
    prescription : dict
    wavelength : float
    output_shape : (int, int), optional
        Output-grid (Ny, Nx) shape.  Defaults to ``E_in.shape``.  v5.2
        rename of the legacy ``output_grid`` kwarg (AUDIT_V4_13_1 Part 2
        P1-A); the dispatcher's :func:`propagate(output_grid=...)`
        contract carries the ``(N_out, dx_out)`` semantics instead.
    output_grid : (int, int), optional
        Deprecated v5.2 alias for ``output_shape``.  Emits a
        ``DeprecationWarning``.
    output_dx, output_centre : grid geometry
    source_box_half, pupil_box_half : float
        Half-widths of the source / pupil sampling boxes for the
        polynomial fit.
    n_field, n_pupil : int
        Per-axis Chebyshev-node grid sizes for the fit.
    poly_order : int
        Total-degree truncation of the Chebyshev fit.
    method : str
        ``'asymptotic'`` or ``'direct'``.

    Returns
    -------
    array (Ny, Nx) complex
        Output-plane complex field.

    .. versionchanged:: 5.2
        ``output_grid`` -> ``output_shape`` rename (AUDIT_V4_13_1 Part 2
        P1-A).
    """
    import numpy as _np

    from .asymptotic import (
        fit_canonical_polynomials,
        propagate_modal_asymptotic,
    )

    # v5.2 (AUDIT_V4_13_1 Part 2 P1-A closure): ``output_grid`` is now
    # the deprecated spelling of ``output_shape``; see module helper.
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_huygens_fresnel_through_prescription',
        default_shape=(E_in.shape[-2], E_in.shape[-1]),
    )
    if output_dx is None:
        output_dx = dx

    if method == 'asymptotic':
        # Build the canonical polynomial fit.
        fit = fit_canonical_polynomials(
            prescription, wavelength,
            source_box_half=source_box_half,
            pupil_box_half=pupil_box_half,
            n_field=n_field,
            n_pupil=n_pupil,
            poly_order=poly_order,
        )

        # Evaluate the modal asymptotic propagator on an output
        # grid.  4.11.2: build the source LG-mode amplitudes by
        # projecting ``E_in`` onto the LG basis (truncated at
        # ``source_lg_p_max`` / ``source_lg_ell_max``).  Pre-4.11.2
        # replaced ``E_in`` with a unit fundamental Gaussian and
        # produced a Gaussian output regardless of the input
        # field's structure -- a structured source (e.g. a vortex
        # beam, off-axis Gaussian, Airy pattern) was silently
        # discarded.
        from ..analysis.core import beam_d4sigma
        from .asymptotic import decompose_lg
        cx, cy = output_centre
        # v4.12.0 (B1-10): pixel-centred grid (drop the `+0.5`),
        # matches ASM/Fresnel/RS/sources so subsequent through-focus
        # scans and overlays stay coherent across propagator families.
        out_x = (_np.arange(Nx) - Nx / 2) * output_dx + cx
        out_y = (_np.arange(Ny) - Ny / 2) * output_dx + cy
        OX, OY = _np.meshgrid(out_x, out_y, indexing='xy')

        # Estimate source waist from input field.
        try:
            d4 = beam_d4sigma(E_in, dx=dx)
            w_s = float(d4) / 4.0
        except (TypeError, ValueError, RuntimeError, ZeroDivisionError):
            # beam_d4sigma can raise: TypeError on non-array E_in,
            # ValueError on empty / wrong-rank inputs, RuntimeError
            # when the moments diverge, ZeroDivisionError on a zero
            # total-power normalisation.  Fall back to the explicit
            # second-moment estimate (which is numerically stable for
            # any finite |E|^2 distribution).
            w_s = float(_np.sqrt(_np.sum(_np.abs(E_in) ** 2 *
                                          (_np.arange(Nx) - Nx / 2) ** 2 * dx ** 2)
                                  / max(_np.sum(_np.abs(E_in) ** 2), 1e-30)))
        if w_s <= 0 or not _np.isfinite(w_s):
            w_s = source_box_half / 2

        # Build input-plane coordinates for LG decomposition.
        # v4.12.0 (B1-10): pixel-centred grid (drop the `+0.5`).  The
        # input field ``E_in`` was sampled by upstream propagators /
        # source builders on the library-standard `(arange(N) - N/2)*dx`
        # grid; the LG decomposition must use the same coordinates so
        # the projected mode amplitudes correctly represent ``E_in``.
        Ny_in, Nx_in = E_in.shape[-2], E_in.shape[-1]
        in_x = (_np.arange(Nx_in) - Nx_in / 2) * dx
        in_y = (_np.arange(Ny_in) - Ny_in / 2) * dx
        IX, IY = _np.meshgrid(in_x, in_y, indexing='xy')
        # Decompose E_in onto LG modes at the source plane.
        try:
            E_in_np = _np.asarray(E_in)
            source_lg = decompose_lg(
                E_in_np, IX, IY, w_s,
                source_lg_p_max, source_lg_ell_max,
                cx=0.0, cy=0.0,
            )
        except (TypeError, ValueError, RuntimeError) as _exc:
            # decompose_lg fails on TypeError (non-array E_in or bad
            # dtypes), ValueError (shape mismatch with IX/IY or w_s<=0),
            # and RuntimeError (singular projection matrix on a
            # degenerate field).  Surface the failure so the silent
            # plane-wave fallback below doesn't hide a real upstream
            # bug -- the asymptotic propagator is essentially useless
            # without a valid LG decomposition.
            import warnings as _w
            _w.warn(
                f"propagate_hf: source LG decomposition failed "
                f"({type(_exc).__name__}: {_exc}); falling back to a "
                f"single (p=0, l=0) plane-wave mode.  This may "
                f"indicate a bug; please report at "
                f"https://github.com/travaj24/LumenAiry/issues",
                RuntimeWarning, stacklevel=2)
            source_lg = {(0, 0): 1.0 + 0.0j}
        # Drop amplitudes below threshold (sparsity speedup).
        max_amp = max((abs(a) for a in source_lg.values()), default=1.0)
        if max_amp > 0:
            source_lg = {k: v for k, v in source_lg.items()
                         if abs(v) >= source_lg_amp_threshold * max_amp}
        if not source_lg:
            source_lg = {(0, 0): 1.0 + 0.0j}
        # Pupil defaults to plane-wave (single-mode).
        pupil_lg = {(0, 0): 1.0}

        # Source point at origin (LG basis is centred there).
        source_point = (0.0, 0.0)
        w_p = pupil_box_half
        v2_centre = (0.0, 0.0)

        # propagate_modal_asymptotic expects per-pixel s2 grids.
        # OX, OY were built above on the output_grid_xy / output_dx
        # spec and are the right grids to sample on.
        return propagate_modal_asymptotic(
            fit,
            source_amplitudes=source_lg,
            pupil_amplitudes=pupil_lg,
            source_point=source_point,
            w_s=w_s,
            w_p=w_p,
            v2_centre=v2_centre,
            s2_grid_x=OX,
            s2_grid_y=OY,
        )

    if method == 'direct':
        from .asymptotic import (
            fit_hf_polynomials,
            propagate_hf_chebyshev_quadrature,
        )
        # Build the HF-form polynomial fit Phi(s1, s2).
        hf_fit = fit_hf_polynomials(
            prescription, wavelength,
            source_box_half=source_box_half,
            pupil_box_half=pupil_box_half,
            n_field=n_field,
            n_pupil=n_pupil,
            poly_order=poly_order,
        )

        # Build input and output grids.
        # v4.12.0 (B1-10): pixel-centred grid (drop the `+0.5`),
        # matches ASM/Fresnel/RS/sources.
        Ny_in, Nx_in = E_in.shape[-2], E_in.shape[-1]
        in_x = (_np.arange(Nx_in) - Nx_in / 2) * dx
        in_y = (_np.arange(Ny_in) - Ny_in / 2) * dx
        cx, cy = output_centre
        out_x = (_np.arange(Nx) - Nx / 2) * output_dx + cx
        out_y = (_np.arange(Ny) - Ny / 2) * output_dx + cy

        return propagate_hf_chebyshev_quadrature(
            hf_fit, E_in,
            input_grid_x=in_x, input_grid_y=in_y,
            output_grid_x=out_x, output_grid_y=out_y,
            apply_van_vleck=True,
        )

    raise ValueError(
        f"propagate_huygens_fresnel_through_prescription: method must be "
        f"'asymptotic' or 'direct', got {method!r}.")


__all__ = [
    'propagate_huygens_fresnel_freespace',
    'propagate_huygens_fresnel_with_opl_callable',
    'propagate_huygens_fresnel_through_prescription',
]
