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
    frame_completeness,
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

# H7 (2026-07-19): ``direction_sampling='auto'`` engages the Husimi (carrier-
# normal) launch when the input carries transverse angular content.  The
# threshold and the RMS-local-tilt detector now live in the beamlet layer
# (:mod:`lumenairy.propagators.gbd`) so ``apply_real_lens_gbd`` and
# ``propagate_gbd_through_prescription`` share ONE definition (N4, 2026-07-19);
# re-exported here for backward-compatible imports.
from ..propagators.gbd import (  # noqa: E402
    _GBD_AUTO_HUSIMI_THRESH,
    _input_angular_spread,
)


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


# P4 / N3 (2026-07-20): frame-completeness threshold below which the
# ``reexpand='auto'`` path re-decomposes the input with a carrier reference.  A
# well-conditioned frame (collimated / mildly-curved input) reconstructs its
# input to > 0.99; a strong reconvergence congruence (converging input through a
# negative element) drops to ~0.88-0.94 because the flat-waist beamlets cannot
# carry the input wavefront curvature.  0.98 sits above the best achievable naive
# frame for that class (so it triggers) and below the collimated baseline (so it
# does not fire there -- byte-identical output).
_GBD_REEXPAND_COMPLETENESS_THRESH = 0.98


def _carrier_referenced_bundle(
    E_dec: np.ndarray, dx: float, dy: float, wavelength: float,
    sample_step: int, waist_factor: float, carrier: Any = 'auto',
):
    """Carrier-referenced beamlet decomposition (P4 / N3 re-expansion).

    A flat-waist Gaussian beamlet frame cannot represent a strongly-curved input
    wavefront (a converging congruence reconverged by a negative element): the
    un-captured per-beamlet curvature makes the frame incomplete, so its coherent
    sum sheds ~6-12% of the power (measured at the DECOMPOSITION plane, before any
    propagation -- an unrecoverable loss baked into the frame).

    This re-decomposition removes the smooth carrier congruence ``W`` (the local
    wavefront the beamlets cannot otherwise carry), decomposes the resulting
    COMPACT, near-flat residual (which forms a near-complete frame), then seeds
    each beamlet's launch **direction** (``grad W`` -- the carrier normals, the
    H7 machinery), **curvature** (``Q += mean Hessian(W)``, so the beamlet rides
    the local congruence rather than a flat waist), and **piston**
    (``exp(i k0 W(centre))`` -- restoring the carrier phase the de-carriered
    amplitude dropped).  The frame then reproduces the curved input to a
    near-complete ``> 0.99`` and conserves power through the element evolution.

    ``carrier`` reuses the traced ``carrier`` vocabulary via
    :func:`lumenairy.elements._lens_traced._compute_carrier`: ``'auto'`` fits a
    low-order polynomial congruence from ``E_dec``; a ``float`` is a signed
    on-axis conjugate distance (m); an ndarray is an explicit wavefront ``W``.

    Returns ``(bundle, n_kept)`` or ``(None, 0)`` when the carrier cannot be
    built for this grid (anamorphic ``dy != dx`` -- ``_compute_carrier`` is
    isotropic-grid only; the caller then keeps the naive frame).  The seeded
    scalar (mean-curvature) ``Q`` matches an isotropic / mildly-astigmatic
    carrier exactly; a strongly astigmatic carrier is N7's separable-axis scope.
    """
    if abs(float(dy) - float(dx)) > 1e-12 * max(abs(float(dx)), 1.0):
        return None, 0                         # isotropic-grid carrier only
    from ._lens_traced import _compute_carrier
    N = E_dec.shape[-1]
    ax = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(ax, ax)
    try:
        W_full, grad_fn, w_fn = _compute_carrier(
            carrier, E_dec, wavelength, dx, X, Y)
    except (ValueError, TypeError, np.linalg.LinAlgError):
        return None, 0
    k0 = 2.0 * np.pi / wavelength
    E_flat = E_dec * np.exp(-1j * k0 * W_full)
    bundle = decompose_field_to_beamlets(
        E_flat, dx, wavelength=wavelength, dy=dy,
        waist_factor=float(waist_factor), sample_step=int(sample_step),
        direction_sampling=False)
    bundle, n_kept = _prune_zero_beamlets(bundle)
    pos = np.asarray(bundle.positions)
    xc, yc = pos[:, 0], pos[:, 1]
    # launch direction = carrier normal grad W (H7 carrier-normal launch)
    Lc, Mc = grad_fn(xc, yc)
    norm = np.sqrt(Lc * Lc + Mc * Mc + 1.0)
    directions = np.stack([Lc / norm, Mc / norm, 1.0 / norm], axis=-1)
    # seed beamlet curvature: Re(Q) += mean Hessian(W) (the local congruence the
    # flat waist otherwise omits).  Hessian from finite differences of W_full;
    # beamlet centres lie in the interior bright support so edge error is moot.
    gWy, gWx = np.gradient(W_full, dx, dx)
    Wxx = np.gradient(gWx, dx, axis=1)
    Wyy = np.gradient(gWy, dx, axis=0)
    fx = np.clip(xc / dx + N / 2.0, 0, N - 1).astype(np.int64)
    fy = np.clip(yc / dx + N / 2.0, 0, N - 1).astype(np.int64)
    curv = 0.5 * (Wxx[fy, fx] + Wyy[fy, fx])
    Q = np.asarray(bundle.Q).astype(np.complex128) + curv
    # restore the carrier piston the de-carriered amplitude dropped
    amp = np.asarray(bundle.amplitude) * np.exp(1j * k0 * w_fn(xc, yc))
    return (BeamletBundle(
        positions=bundle.positions, directions=directions, Q=Q,
        amplitude=amp.astype(Q.dtype), waist0=bundle.waist0), n_kept)


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
    direction_sampling: Any = 'auto',
    reexpand: str = 'off',
    reexpand_carrier: Any = 'auto',
    reexpand_threshold: float = _GBD_REEXPAND_COMPLETENESS_THRESH,
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
    diagnostics: Optional[Dict[str, Any]] = None,
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
    direction_sampling :
        Beamlet LAUNCH-direction policy (H7).  ``'auto'`` (default) launches
        each beamlet along the input field's local wavevector (Husimi /
        carrier-normal launch) when the input carries transverse angular
        content -- a strongly diverging / converging or tilted wavefront --
        and along the axis otherwise.  A flat-wavefront (collimated) field
        takes the axial path, byte-identical to the historical default; a
        diverging / converging source then focuses at its true (finite-
        conjugate) image rather than collapsing to the collimated focal plane
        and shedding frame energy.  ``True`` / ``False`` force the Husimi /
        axial launch explicitly.

        Envelope (G1, 2026-07-19): power conservation > 0.99 for a diverging
        input (any lens), a converging input through a POSITIVE lens, and a
        multi-element diverging chain (doublet).  A converging input
        RECONVERGED by a NEGATIVE lens to a *near* real focus is a
        frame-completeness EDGE: the launch is still correct (focuses at the
        ABCD image) but the beamlet frame sheds a few percent of power that a
        finer frame does NOT recover (it saturates at the maximum density, ~0.94
        for the G1 M5 biconcave at R_in=-35mm).  **P4 (2026-07-20): that ~0.94
        limit is now closable with ``reexpand='auto'``** (carrier-referenced
        re-decomposition -- see below), reaching > 0.99 power and windowed r2m
        within 0.3% of ``traced``.  A negative element also needs a finer
        ``beamlets_per_aperture`` than a positive one to tile its diverging exit.
        For absolute power without re-expansion pass ``normalize_output='power'``
        (restores the total but not the shed spatial structure).
    reexpand :
        Frame re-expansion policy for the strong-reconvergence edge (P4 / N3).
        ``'off'`` (default) -- the single input-plane frame, **byte-identical**
        to prior releases.  ``'auto'`` -- when the input carries wavefront
        curvature AND the naive frame's input-plane completeness (see
        :func:`lumenairy.propagators.gbd.frame_completeness`) falls below
        ``reexpand_threshold``, re-decompose the input with a CARRIER REFERENCE:
        remove the smooth congruence ``W`` (which a flat-waist beamlet cannot
        carry), decompose the compact near-flat residual (a near-complete frame),
        and seed each beamlet's launch direction (``grad W``), curvature
        (``Q += Hessian W``) and piston from the carrier.  This closes the
        frame-completeness loss that is otherwise baked into the input frame
        (unrecoverable downstream).  A collimated / well-conditioned input
        measures completeness > ``reexpand_threshold`` and is NOT re-expanded, so
        ``'auto'`` there returns output byte-identical to ``'off'`` (it only adds
        one input-plane completeness reconstruction).  Overhead when it fires is
        ~2x (measured 1.5-1.9x on the M5 reconvergence case).
    reexpand_carrier :
        Carrier congruence for ``reexpand='auto'`` (reuses the ``traced``
        ``carrier`` vocabulary): ``'auto'`` (default) fits a low-order polynomial
        congruence from the input; a ``float`` is a signed on-axis conjugate
        distance (m); an ndarray is an explicit wavefront ``W`` (m).  The seeded
        scalar (mean-curvature) beamlet ``Q`` matches an isotropic / mildly
        astigmatic carrier exactly; strongly astigmatic carriers are N7 scope.
    reexpand_threshold :
        Completeness below which ``reexpand='auto'`` re-decomposes (default 0.98).
    normalize_output :
        ``'none'`` (default, raw energy-conserving field) or ``'power'``
        (rescale so the output power equals the aperture-transmitted input
        power -- for like-for-like profile comparison with the other models).
    diagnostics :
        Optional dict; if provided it is populated in place (output unchanged)
        with ``'frame_completeness'`` (reconstructed output power / aperture-
        transmitted input power -- the P4 metric), ``'reexpanded'`` (bool),
        ``'n_beamlets'``, and, on the ``reexpand='auto'`` path,
        ``'frame_completeness_input'`` (the naive frame's input-plane
        completeness that drove the re-expansion decision) and, when it fired,
        ``'frame_completeness_reexpanded'``.  ``None`` (default) computes no
        metric (zero overhead).
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

    # H7: resolve the beamlet launch-direction policy.  A strongly-diverging /
    # converging input's wavefront curvature must ride the beamlet LAUNCH
    # DIRECTIONS (Husimi / carrier normals), not just their amplitude --
    # otherwise the axial base rays focus at the COLLIMATED plane and the frame
    # sheds energy (power collapses to ~0.2-0.5 on the dual-oracle singlet, EE
    # at the true image ~0.09).  'auto' engages Husimi only when the input
    # carries measurable transverse angular content; a flat-wavefront
    # (collimated) field measures ~0 and takes the byte-identical axial path.
    if reexpand not in ('off', 'auto'):
        raise ValueError(
            "apply_real_lens_gbd: reexpand must be 'off' or 'auto', "
            f"got {reexpand!r}.")
    # Input angular spread (RMS local tilt) -- gates both the Husimi launch and
    # the P4 re-expansion (a flat / collimated frame is already complete, so
    # re-expansion is skipped there -- byte-identical to 'off' + ~1x cost).
    _spread = _input_angular_spread(E_dec, dx, dy, wavelength)
    if direction_sampling == 'auto':
        use_direction_sampling = _spread > _GBD_AUTO_HUSIMI_THRESH
    elif direction_sampling in (True, False):
        use_direction_sampling = bool(direction_sampling)
    else:
        raise ValueError(
            "apply_real_lens_gbd: direction_sampling must be 'auto', True or "
            f"False, got {direction_sampling!r}.")

    if verbose:
        nb = ((Ny + sample_step - 1) // sample_step) * \
             ((Nx + sample_step - 1) // sample_step)
        print(f"  [gbd] sample_step={sample_step} waist_factor={waist_factor} "
              f"(~{nb} beamlets), window={window}, "
              f"per_surface={per_surface}, "
              f"direction_sampling={use_direction_sampling}"
              f"{' (auto)' if direction_sampling == 'auto' else ''}", flush=True)

    bundle = decompose_field_to_beamlets(
        E_dec, dx, wavelength=wavelength, dy=dy,
        waist_factor=float(waist_factor), sample_step=int(sample_step),
        direction_sampling=use_direction_sampling)

    # decompose places a beamlet on EVERY sample_step cell of the full grid;
    # for a clipped/finite field most carry ~zero amplitude and only inflate the
    # reconstruct cost.  Drop them (contribute nothing to the coherent sum).
    bundle, n_kept = _prune_zero_beamlets(bundle)
    if verbose:
        print(f"  [gbd] {n_kept} illuminated beamlets after pruning", flush=True)

    # P4 / N3 (2026-07-20): frame re-expansion for the strong-reconvergence edge.
    # A converging input reconverged by a negative element sheds ~6-12% of its
    # power at the INPUT decomposition (the flat-waist beamlets cannot carry the
    # input wavefront curvature; the loss is baked into the frame and is NOT
    # recoverable downstream).  When the naive frame's input-plane completeness
    # falls below the threshold, re-decompose with a carrier reference (remove
    # the smooth congruence, decompose the compact residual, seed direction /
    # curvature / piston from the carrier).  Only fires for a CURVED input
    # (a flat frame is already complete), so a collimated input is untouched and
    # byte-identical to reexpand='off'.
    reexpanded = False
    comp_input = None
    comp_reexp = None
    if (reexpand == 'auto' and per_surface
            and _spread > _GBD_AUTO_HUSIMI_THRESH):
        comp_input = frame_completeness(
            bundle, E_dec, dx, wavelength=wavelength, dy=dy, window=window,
            mem_budget_mb=mem_budget_mb)
        if comp_input < reexpand_threshold:
            rb, rk = _carrier_referenced_bundle(
                E_dec, dx, dy, wavelength, int(sample_step),
                float(waist_factor), carrier=reexpand_carrier)
            if rb is not None:
                comp_reexp = frame_completeness(
                    rb, E_dec, dx, wavelength=wavelength, dy=dy, window=window,
                    mem_budget_mb=mem_budget_mb)
                # accept only if the carrier reference genuinely improves the
                # frame (guards against a mis-fit carrier degrading it).
                if comp_reexp > comp_input:
                    bundle, n_kept = rb, rk
                    reexpanded = True
                    if verbose:
                        print(f"  [gbd] reexpand: carrier-referenced frame "
                              f"({comp_input:.4f} -> {comp_reexp:.4f} "
                              f"completeness, {n_kept} beamlets)", flush=True)

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
            direction_sampling=use_direction_sampling,
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

    # P4 frame-completeness metric (opt-in via ``diagnostics``): the raw
    # reconstructed output power over the aperture-transmitted input power --
    # "power captured by the beamlet frame".  Computed from the RAW field before
    # normalize_output='power' rescales it (which would force it to 1.0).
    if diagnostics is not None:
        p_in_ap = float(np.sum(np.abs(E_dec) ** 2))
        p_out_raw = float(np.sum(np.abs(E_out) ** 2))
        diagnostics['frame_completeness'] = (
            p_out_raw / p_in_ap if p_in_ap > 0 else 0.0)
        diagnostics['reexpanded'] = reexpanded
        diagnostics['n_beamlets'] = int(n_kept)
        if comp_input is not None:
            diagnostics['frame_completeness_input'] = comp_input
        if comp_reexp is not None:
            diagnostics['frame_completeness_reexpanded'] = comp_reexp

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
