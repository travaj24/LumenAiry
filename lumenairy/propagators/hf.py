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

from typing import Callable, Optional, Tuple

import numpy as np

from ..backend import array_namespace, is_jax_array


def propagate_huygens_fresnel_freespace(
    E_in,
    z: float,
    wavelength: float,
    dx: float,
    *,
    dy: Optional[float] = None,
    **kwargs,
):
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
    E_in,
    *,
    opl_fn: Callable,
    output_grid_x,
    output_grid_y,
    input_grid_dx: float,
    wavelength: float,
    apply_van_vleck: bool = True,
    finite_diff_step: float = 1e-9,
    chunk_output: int = 64,
):
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
    s1_x = (xp.arange(Nx_in, dtype=xp.float64) - Nx_in / 2 + 0.5) * input_grid_dx
    s1_y = (xp.arange(Ny_in, dtype=xp.float64) - Ny_in / 2 + 0.5) * input_grid_dx
    S1X, S1Y = xp.meshgrid(s1_x, s1_y, indexing='xy')

    Ny_out = output_grid_y.shape[0]
    Nx_out = output_grid_x.shape[0]
    out = xp.zeros((Ny_out, Nx_out), dtype=E_in.dtype)
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

            kernel = xp.exp(2j * float(np.pi) * phi).astype(E_in.dtype)
            integrand = E_in * density * kernel
            iy = k // Nx_out
            ix = k % Nx_out
            out_value = xp.sum(integrand) * pixel_area
            if is_jax_array(E_in):
                out = out.at[iy, ix].set(out_value)
            else:
                out[iy, ix] = out_value

    return out


# ============================================================================
# Prescription-aware HF (Van-Vleck-corrected, via fit_canonical_polynomials)
# ============================================================================

def propagate_huygens_fresnel_through_prescription(
    E_in,
    dx: float,
    prescription: dict,
    *,
    wavelength: float,
    output_grid=None,
    output_dx=None,
    output_centre=(0.0, 0.0),
    source_box_half: float = 50e-6,
    pupil_box_half: float = 0.05,
    n_field: int = 8,
    n_pupil: int = 8,
    poly_order: int = 6,
    method: str = 'asymptotic',
):
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
        # grid.  This requires a source LG-mode decomposition of
        # E_in; for the simplest case we approximate the source
        # as a single fundamental Gaussian whose waist matches
        # the input field's apparent waist.
        from ..analysis.analysis import beam_d4sigma
        cx, cy = output_centre
        out_x = (_np.arange(Nx) - Nx / 2 + 0.5) * output_dx + cx
        out_y = (_np.arange(Ny) - Ny / 2 + 0.5) * output_dx + cy
        OX, OY = _np.meshgrid(out_x, out_y, indexing='xy')
        output_grid_xy = _np.stack([OX, OY], axis=-1)

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

        # Single-mode source LG amplitude {(0, 0): 1.0}; pupil
        # amplitudes default to plane-wave (single-mode).
        source_lg = {(0, 0): 1.0}
        pupil_lg = {(0, 0): 1.0}

        # Source point at origin.
        source_point = (0.0, 0.0)
        w_p = pupil_box_half
        v2_centre = (0.0, 0.0)

        return propagate_modal_asymptotic(
            fit,
            source_lg_amps=source_lg,
            pupil_lg_amps=pupil_lg,
            source_point=source_point,
            w_s=w_s,
            w_p=w_p,
            v2_centre=v2_centre,
            output_grid=output_grid_xy,
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
        Ny_in, Nx_in = E_in.shape[-2], E_in.shape[-1]
        in_x = (_np.arange(Nx_in) - Nx_in / 2 + 0.5) * dx
        in_y = (_np.arange(Ny_in) - Ny_in / 2 + 0.5) * dx
        cx, cy = output_centre
        out_x = (_np.arange(Nx) - Nx / 2 + 0.5) * output_dx + cx
        out_y = (_np.arange(Ny) - Ny / 2 + 0.5) * output_dx + cy

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
