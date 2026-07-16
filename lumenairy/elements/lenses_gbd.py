"""GBD (Gaussian-beamlet-decomposition) real-lens propagator.

``apply_real_lens_gbd`` is the beamlet-based peer of :func:`apply_real_lens`
(analytic split-step), :func:`apply_real_lens_traced` (per-pixel ray-traced
OPL) and :func:`apply_real_lens_maslov` (phase-space integral).  It shares the
same contract -- ``(E_in, *, prescription, wavelength, dx, ...) -> field at the
lens EXIT plane on the input grid`` -- so it drops into the same
``LENS_MODEL`` dispatch as the other three.

Why a fourth model
------------------
The field is decomposed into a frame of overlapping Gaussian *beamlets*; each
beamlet is carried through the prescription by the **exact per-surface
differential ray transfer** (:func:`lumenairy.raytrace.ray_transfer_jacobian`),
promoting its complex curvature ``Q`` to a ``(2, 2)`` tensor that captures
astigmatism and higher-order aberration.  There is **no thin-screen collapse**
(the analytic model's ``sag * theta**2`` error) and **no single global
polynomial eikonal fit** (the Maslov model's high-order instability): aberration
is represented *piecewise*, one exact-Jacobian beamlet per patch of the pupil,
and refined by adding beamlets rather than raising an order.  This makes it the
natural high-NA / strongly-aberrated companion to the other three.

Auto behaviour (nothing to tune for the common case)
----------------------------------------------------
* ``sample_step=None`` -> the beamlet spacing is auto-sized to place
  ~``_GBD_BEAMLETS_PER_APERTURE`` beamlets across the entrance aperture.
* ``waist_factor=None`` -> tied to ``sample_step`` so the beamlets **overlap**
  (a proper frame).  Passing ``sample_step`` without ``waist_factor`` keeps
  them tied; the historical ``waist_factor=1`` sparse-frame footgun is avoided.
* ``window=5.0`` -> the fast windowed / FFT reconstruct is used by default
  (the dense ``O(beamlets * Ny * Nx)`` sum is only ever a fallback).
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..propagators.gbd import (
    BeamletBundle,
    apply_prescription_persurface_to_beamlets,
    decompose_field_to_beamlets,
    reconstruct_field_from_beamlets,
)

__all__ = ['apply_real_lens_gbd']


def _prune_zero_beamlets(bundle: BeamletBundle, rel_thresh: float = 1e-6
                         ) -> Tuple[BeamletBundle, int]:
    """Drop beamlets whose on-axis amplitude is negligible (e.g. the grid cells
    outside the illuminated aperture -- ``decompose`` places a beamlet on EVERY
    ``sample_step`` cell of the full grid, so a clipped/finite field leaves most
    of them at zero amplitude).  They contribute nothing to the coherent sum but
    dominate the reconstruct cost.  Returns the pruned bundle and the kept count.
    """
    amp = np.asarray(bundle.amplitude)
    mag = np.abs(amp)
    peak = float(mag.max()) if mag.size else 0.0
    if peak <= 0.0:
        return bundle, int(mag.size)
    keep = mag > rel_thresh * peak
    if bool(np.all(keep)):
        return bundle, int(mag.size)
    Q = np.asarray(bundle.Q)
    return (BeamletBundle(
        positions=np.asarray(bundle.positions)[keep],
        directions=np.asarray(bundle.directions)[keep],
        Q=Q[keep],
        amplitude=amp[keep],
        waist0=np.asarray(bundle.waist0)[keep]),
        int(keep.sum()))

def _fft_upsample(Ec: np.ndarray, ky: int, kx: int) -> np.ndarray:
    """Band-limited (Fourier zero-pad) upsample of a coarse field ``Ec`` by
    integer factors ``(ky, kx)``.  Exact when ``Ec`` resolves the field's full
    bandwidth -- true at the lens EXIT plane, where the beam is broad and smooth
    (no spot structure forms until the later ASM to focus), so a coarse
    reconstruct + upsample matches the full-grid reconstruct to ~1e-5 at k^2 less
    reconstruct work AND peak memory.  Mirrors the Maslov ``output_subsample``.
    """
    nyc, nxc = Ec.shape[-2], Ec.shape[-1]
    Ny, Nx = nyc * ky, nxc * kx
    F = np.fft.fftshift(np.fft.fft2(Ec))
    Fp = np.zeros((Ny, Nx), dtype=complex)
    oy, ox = (Ny - nyc) // 2, (Nx - nxc) // 2
    Fp[oy:oy + nyc, ox:ox + nxc] = F
    return np.fft.ifft2(np.fft.ifftshift(Fp)) * (ky * kx)


# Auto-density target: how many beamlets to place across the entrance aperture
# when ``sample_step`` is not given.  Calibrated so the GBD field matches the
# analytic model on a paraxial (aberration-free) lens to <1% and is converged
# (halving the spacing does not move the focus); see the module test.
_GBD_BEAMLETS_PER_APERTURE = 256


def _auto_sample_step(E_in: np.ndarray, dx: float,
                      prescription: Dict[str, Any],
                      beamlets_per_aperture: int = _GBD_BEAMLETS_PER_APERTURE) -> int:
    """Beamlet spacing (in pixels) that puts ~``beamlets_per_aperture``
    beamlets across the **illuminated beam** (not the full aperture stop).

    The stop can be much larger than the beam (the (0,0) DOE order underfills
    a 25 mm relay aperture): sizing the frame to the aperture would leave only
    a handful of beamlets across the actual beam and under-resolve it, so we
    size to the illuminated extent of ``E_in`` (capped by the aperture, since
    the beam cannot be wider than the stop).  Falls back to the grid extent for
    a uniform / full-grid field.
    """
    mag = np.abs(E_in)
    peak = float(mag.max()) if mag.size else 0.0
    if peak <= 0.0:
        return 1
    thr = 1e-3 * peak
    cols = np.where(mag.max(axis=0) > thr)[0]
    rows = np.where(mag.max(axis=1) > thr)[0]
    if len(cols) and len(rows):
        span = float(max(cols[-1] - cols[0] + 1, rows[-1] - rows[0] + 1))
    else:
        span = float(max(E_in.shape[-2:]))
    ap = prescription.get('aperture_diameter')
    if ap and ap > 0:
        span = min(span, float(ap) / dx)   # beam can't exceed the stop
    return max(1, int(round(span / max(1, beamlets_per_aperture))))


def _entrance_aperture_mask(E_in: np.ndarray, dx: float, dy: float,
                            prescription: Dict[str, Any]) -> Optional[np.ndarray]:
    """Circular entrance-aperture mask (matches where the analytic / traced /
    thin models clip), or ``None`` if the prescription carries no aperture."""
    ap = prescription.get('aperture_diameter')
    if not (ap and ap > 0):
        return None
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]
    x = (np.arange(Nx) - Nx / 2) * dx
    y = (np.arange(Ny) - Ny / 2) * dy
    X, Y = np.meshgrid(x, y)
    return (X * X + Y * Y) <= (0.5 * float(ap)) ** 2


def apply_real_lens_gbd(
    E_in: np.ndarray,
    *,
    prescription: Dict[str, Any],
    wavelength: float,
    dx: float,
    dy: Optional[float] = None,
    sample_step: Optional[int] = None,
    beamlets_per_aperture: int = _GBD_BEAMLETS_PER_APERTURE,
    waist_factor: Optional[float] = None,
    direction_sampling: bool = False,
    per_surface: bool = True,
    jacobian: str = 'auto',
    window: Optional[float] = 5.0,
    chunk_beamlets: int = 2048,
    mem_budget_mb: float = 512.0,
    output_subsample: int = 1,
    output_plane_distance: float = 0.0,
    output_plane_n: float = 1.0,
    clip_aperture: bool = True,
    roi: Optional[Any] = None,
    normalize_output: str = 'none',
    progress: Optional[Any] = None,
    verbose: bool = False,
) -> np.ndarray:
    """Propagate ``E_in`` through a thick-lens ``prescription`` via Gaussian
    beamlet decomposition and return the field at the lens exit plane
    (last surface vertex ``+ output_plane_distance``) on the input grid.

    Peer of :func:`apply_real_lens`, :func:`apply_real_lens_traced`,
    :func:`apply_real_lens_maslov`; see the module docstring for when to reach
    for it.  ``sample_step`` / ``waist_factor`` default to an auto overlapping
    frame; ``window=5.0`` selects the fast reconstruct.

    Parameters
    ----------
    output_plane_distance, output_plane_n :
        Extra propagation past the exit vertex.  Only ``output_plane_n == 1``
        (air) is supported for a non-zero distance; the sim continues its own
        downstream legs from the exit plane, so it passes ``0.0`` here.
    clip_aperture :
        Clip ``E_in`` at the circular entrance aperture before decomposition
        (matches where the analytic / traced / thin models clip).
    normalize_output :
        ``'none'`` (default, raw energy-conserving field) or ``'power'``
        (rescale so the output power equals the aperture-transmitted input
        power -- for like-for-like profile comparison with the other models).
    """
    from .._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_real_lens_gbd')
    if roi is not None:
        raise NotImplementedError(
            "apply_real_lens_gbd: roi windowing is not yet supported; "
            "reconstruct the full grid and crop, or use apply_real_lens.")
    if output_plane_distance != 0.0 and output_plane_n != 1.0:
        raise NotImplementedError(
            "apply_real_lens_gbd: output_plane_distance in a medium "
            "(output_plane_n != 1) is not supported; do the medium leg "
            "separately (the exit medium is air in the sim's usage).")

    E_in = np.asarray(E_in)
    if dy is None:
        dy = dx
    Ny, Nx = E_in.shape[-2], E_in.shape[-1]

    if progress is not None:
        try:
            progress("gbd: decompose", 0.0)
        except TypeError:
            pass

    # Clip at the entrance aperture (where the real lens stops the beam).
    mask = _entrance_aperture_mask(E_in, dx, dy, prescription) if clip_aperture else None
    E_dec = E_in * mask if mask is not None else E_in

    if sample_step is None:
        sample_step = _auto_sample_step(E_dec, dx, prescription,
                                        beamlets_per_aperture=beamlets_per_aperture)
    if waist_factor is None:
        # Overlapping frame: waist ~ beamlet spacing.  (waist_factor=1 with
        # sample_step>1 would leave gaps -- the sparse-frame footgun.)
        waist_factor = float(sample_step)

    if verbose:
        nb = ((Ny + sample_step - 1) // sample_step) * \
             ((Nx + sample_step - 1) // sample_step)
        print(f"  [gbd] sample_step={sample_step} waist_factor={waist_factor} "
              f"(~{nb} beamlets), window={window}, "
              f"per_surface={per_surface}", flush=True)

    bundle = decompose_field_to_beamlets(
        E_dec, dx, wavelength=wavelength, dy=dy,
        waist_factor=float(waist_factor), sample_step=int(sample_step),
        direction_sampling=direction_sampling)

    # decompose places a beamlet on EVERY sample_step cell of the full grid;
    # for a clipped/finite field most carry ~zero amplitude and only inflate the
    # reconstruct cost.  Drop them (contribute nothing to the coherent sum).
    bundle, n_kept = _prune_zero_beamlets(bundle)
    if verbose:
        print(f"  [gbd] {n_kept} illuminated beamlets after pruning", flush=True)

    if progress is not None:
        try:
            progress("gbd: per-surface evolve", 0.4)
        except TypeError:
            pass

    if per_surface:
        evolved = apply_prescription_persurface_to_beamlets(
            bundle, prescription, wavelength,
            z_image=float(output_plane_distance), jacobian=jacobian)
    else:
        # Paraxial whole-system ABCD (aberration-free reference); imported
        # lazily so the common per-surface path has no extra import cost.
        from ..propagators.gbd import propagate_gbd_through_prescription
        return propagate_gbd_through_prescription(
            E_dec, dx, prescription, wavelength=wavelength,
            output_shape=(Ny, Nx), output_dx=dx,
            per_surface=False, z_image=float(output_plane_distance))

    if progress is not None:
        try:
            progress("gbd: reconstruct", 0.6)
        except TypeError:
            pass

    # output_subsample: reconstruct on a coarser N/k grid then band-limited
    # FFT-upsample.  The reconstruct cost and its peak buffer both scale with
    # the output pixel count, so this is a ~k^2 speed AND memory win; it is exact
    # here because the lens-exit field is smooth (see _fft_upsample).  Only
    # engages when k>1 and evenly divides the grid.
    ss = max(1, int(output_subsample))
    while ss > 1 and (Ny % ss or Nx % ss):
        ss -= 1
    if ss > 1:
        E_c = reconstruct_field_from_beamlets(
            evolved, Ny=Ny // ss, Nx=Nx // ss, dx=dx * ss, dy=dy * ss,
            wavelength=wavelength, centre=(0.0, 0.0), window=window,
            chunk_beamlets=chunk_beamlets, mem_budget_mb=mem_budget_mb)
        E_out = _fft_upsample(np.asarray(E_c), ss, ss)
    else:
        E_out = np.asarray(reconstruct_field_from_beamlets(
            evolved, Ny=Ny, Nx=Nx, dx=dx, dy=dy, wavelength=wavelength,
            centre=(0.0, 0.0), window=window,
            chunk_beamlets=chunk_beamlets, mem_budget_mb=mem_budget_mb))

    if normalize_output == 'power':
        p_in = float(np.sum(np.abs(E_dec) ** 2))
        p_out = float(np.sum(np.abs(E_out) ** 2))
        if p_out > 0:
            E_out = E_out * np.sqrt(p_in / p_out)
    elif normalize_output not in ('none', None):
        raise ValueError(
            f"normalize_output={normalize_output!r}; expected 'none' or 'power'.")

    if progress is not None:
        try:
            progress("gbd: done", 1.0)
        except TypeError:
            pass
    return E_out
