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

from typing import Any, Dict, Callable, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array


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
    **kwargs: Any,
) -> np.ndarray:
    """Free-space Huygens-Fresnel propagation with the standard
    ``1 / (i lambda z)`` Van Vleck factor.

    Equivalent to :func:`lumenairy.propagation.rayleigh_sommerfeld_propagate`;
    re-exported here for API consistency with the other ``hf.*``
    entry points.
    """
    from .propagation import rayleigh_sommerfeld_propagate
    return rayleigh_sommerfeld_propagate(
        E_in, z, wavelength, dx, dy=dy, **kwargs,
    )


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
    output_grid, output_dx, output_centre : grid geometry
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
    """
    from .asymptotic import (
        fit_canonical_polynomials,
        propagate_modal_asymptotic,
    )
    import numpy as _np

    Ny, Nx = (E_in.shape[-2], E_in.shape[-1]) if output_grid is None else output_grid
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
        except Exception:
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
        except Exception:
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
            fit_hf_polynomials, propagate_hf_chebyshev_quadrature,
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
