"""
Interferogram simulation and analysis.

Convert OPD maps to fringe patterns (Twyman-Green, Fizeau, Mach-Zehnder
style), and extract OPD from measured fringes via phase-shifting.

Author: Andrew Traverso
"""
from __future__ import annotations
import numpy as np


def simulate_interferogram(opd_map, wavelength, tilt_x=0.0, tilt_y=0.0,
                           visibility=1.0, background=0.5, dx=None):
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
    fringe = background + 0.5 * visibility * np.cos(phase)
    fringe = np.where(np.isfinite(opd), fringe, 0.0)
    return fringe


def phase_shift_extract(frames, shifts=None, convention='hardware'):
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
    # Least-squares extraction: phase = atan2(sum(I*sin), sum(I*cos))
    sin_sum = np.zeros_like(frames[0])
    cos_sum = np.zeros_like(frames[0])
    for f, s in zip(frames, shifts):
        sin_sum += f * np.sin(s)
        cos_sum += f * np.cos(s)
    if convention == 'hardware':
        # I = a + b * cos(phi - s)
        #   = a + b*(cos(phi)cos(s) + sin(phi)sin(s))
        # => sum(I*sin(s)) = +b*(n/2)*sin(phi);
        #    sum(I*cos(s)) = +b*(n/2)*cos(phi)
        # => phi = atan2(sin_sum, cos_sum)
        phase = np.arctan2(sin_sum, cos_sum)
    elif convention == 'library':
        # I = a + b * cos(phi + s)
        # => sum(I*sin(s)) = -b*(n/2)*sin(phi);
        # => phi = atan2(-sin_sum, cos_sum)
        phase = np.arctan2(-sin_sum, cos_sum)
    else:
        raise ValueError(
            f"convention must be 'hardware' or 'library', got {convention!r}")
    modulation = 2 * np.sqrt(sin_sum**2 + cos_sum**2) / n
    return phase, modulation


def fringe_spacing(wavelength, tilt_angle):
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
