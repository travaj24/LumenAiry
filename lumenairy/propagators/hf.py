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


__all__ = [
    'propagate_huygens_fresnel_freespace',
    'propagate_huygens_fresnel_with_opl_callable',
]
