"""
Interferogram simulation and analysis.

Convert OPD maps to fringe patterns (Twyman-Green, Fizeau, Mach-Zehnder
style), and extract OPD from measured fringes via phase-shifting.

Author: Andrew Traverso
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    'simulate_interferogram',
    'phase_shift_extract',
    'fringe_spacing',
]


def simulate_interferogram(
    opd_map: np.ndarray,
    wavelength: float,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    visibility: float = 1.0,
    background: float = 0.5,
    dx: Optional[float] = None,
) -> np.ndarray:
    """Generate a fringe pattern from an OPD map.

    Models a Twyman-Green or Fizeau interferometer where the test
    wavefront (encoded in ``opd_map``) interferes with a flat
    reference tilted by ``(tilt_x, tilt_y)`` fringes per meter.

    Parameters
    ----------
    opd_map : ndarray (2-D, real)
        Optical path difference [m].  ``NaN`` pixels are rendered as
        zero intensity (masked).
    wavelength : float
        Wavelength [m] for the fringe computation.
    tilt_x, tilt_y : float, default 0
        Reference-beam tilt in fringes per meter along x / y.  A tilt
        of ``1 / (N * dx)`` gives one fringe across the field.
    visibility : float, default 1
        Fringe visibility (0 = no fringes, 1 = max contrast).
    background : float, default 0.5
        Mean intensity level (DC offset).
    dx : float, optional
        Grid spacing [m].  Required if ``tilt_x`` or ``tilt_y`` are
        nonzero.

    Returns
    -------
    fringe : ndarray (2-D, real)
        Intensity fringe pattern, values in [0, 1].
    """
    opd = np.asarray(opd_map, dtype=np.float64)
    Ny, Nx = opd.shape
    phase = 2 * np.pi * opd / wavelength
    if (tilt_x != 0 or tilt_y != 0) and dx is not None:
        x = (np.arange(Nx) - Nx / 2) * dx
        y = (np.arange(Ny) - Ny / 2) * dx
        X, Y = np.meshgrid(x, y)
        phase = phase + 2 * np.pi * (tilt_x * X + tilt_y * Y)
    # 4.10: classic Michelson fringe is
    #   I = background * (1 + visibility * cos(phase))
    # which produces Michelson contrast V = (Imax - Imin) / (Imax + Imin)
    # = visibility (matching the kwarg semantics).  Pre-4.10 used
    # `background + 0.5 * visibility * cos(phase)`, which produced
    # contrast 0.5 even with visibility=1, breaking the docstring's
    # round-trip claim.
    fringe = background * (1.0 + visibility * np.cos(phase))
    fringe = np.where(np.isfinite(opd), fringe, 0.0)
    return fringe


def phase_shift_extract(
    frames: Sequence[np.ndarray],
    shifts: Optional[Sequence[float]] = None,
    convention: str = 'hardware',
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract phase (OPD) from a set of phase-shifted interferograms.

    Standard 4-step algorithm by default (shifts = 0, pi/2, pi, 3pi/2).

    Two sign conventions are in common use:

    * ``'hardware'`` (default): the reference arm is **advanced** by
      ``s``, so the intensity is ``I = a + b * cos(phi - s)``.  This is
      the convention used by Hariharan (1987), Schwider (1990), and
      essentially every real interferometer hardware-controller.
    * ``'library'``: the phase shift **adds** to the fringe phase, so
      ``I = a + b * cos(phi + s)``.  Matches this library's own
      :func:`simulate_interferogram` output.

    The two differ only in the sign of the returned phase.  If you
    feed frames produced by :func:`simulate_interferogram` into this
    function with ``convention='library'``, you round-trip exactly.
    If you feed frames from a real piezo-actuated interferometer and
    want sign-correct OPD, use ``convention='hardware'``.

    Parameters
    ----------
    frames : list of ndarray
        Phase-shifted intensity frames (at least 3).
    shifts : list of float, optional
        Phase shifts [rad] applied to the reference for each frame.
        Default: equally spaced over [0, 2*pi).
    convention : ``'hardware'`` or ``'library'``, default ``'hardware'``
        Which fringe-phase sign convention the input frames follow.
        See the discussion above.

    Returns
    -------
    phase : ndarray
        Extracted wrapped phase [rad], range (-pi, pi].
    modulation : ndarray
        Fringe modulation (visibility proxy).
    """
    n = len(frames)
    if shifts is None:
        shifts = [2 * np.pi * i / n for i in range(n)]
    frames = [np.asarray(f, dtype=np.float64) for f in frames]
    shifts = np.asarray(shifts, dtype=np.float64)
    # v5.4.6 (audit F-13): GENERAL least-squares extraction valid for
    # ARBITRARY (non-equispaced) shifts.  Model each frame as
    #   I_k = a + A*cos(s_k) + B*sin(s_k),  A = b*cos(phi), B = b*sin(phi)
    # and solve the linear LSQ for (a, A, B) per pixel via the design
    # matrix S = [1, cos(s), sin(s)].  The previous correlation estimator
    # atan2(sum(I*sin), sum(I*cos)) is the LSQ solution ONLY when the S
    # columns are orthogonal (equispaced full-period shifts); for that
    # case S^T S = diag(n, n/2, n/2) and this reduces to exactly the old
    # result (so equispaced callers are bit-preserved), but for arbitrary
    # shifts the old form was biased.
    if convention not in ('hardware', 'library'):
        raise ValueError(
            f"convention must be 'hardware' or 'library', got {convention!r}")
    S = np.stack([np.ones_like(shifts), np.cos(shifts), np.sin(shifts)], axis=1)
    pinv = np.linalg.pinv(S)  # (3, n)
    I_stack = np.stack(frames, axis=0)  # (n, Ny, Nx)
    coeffs = np.tensordot(pinv, I_stack, axes=([1], [0]))  # (3, ...) = (a, A, B)
    A = coeffs[1]
    B = coeffs[2]
    if convention == 'hardware':
        # I = a + b cos(phi - s) -> A = b cos(phi), B = b sin(phi)
        phase = np.arctan2(B, A)
    else:
        # I = a + b cos(phi + s) -> A = b cos(phi), B = -b sin(phi)
        phase = np.arctan2(-B, A)
    modulation = np.sqrt(A ** 2 + B ** 2)
    return phase, modulation


def fringe_spacing(wavelength: float, tilt_angle: float) -> float:
    """Compute fringe spacing [m] for a given reference-beam tilt.

    Parameters
    ----------
    wavelength : float [m]
    tilt_angle : float [rad]

    Returns
    -------
    spacing : float [m]
    """
    return wavelength / (2 * np.sin(tilt_angle / 2)) if tilt_angle != 0 else np.inf
