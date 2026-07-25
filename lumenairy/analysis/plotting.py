"""
Standard plotting utilities for optical propagation simulations.

Provides publication-ready plots tailored to coherent optical fields:

- ``plot_intensity``            -- 2D intensity map (linear or log)
- ``plot_phase``                -- 2D phase map with optional masking
- ``plot_amplitude_phase``      -- side-by-side amplitude and phase
- ``plot_field``                -- general 2-panel (intensity + phase)
- ``plot_cross_section``        -- 1D cut (x or y) through a field
- ``plot_planes_grid``          -- grid of intensity maps for multiple planes
- ``plot_mtf``                  -- radial MTF profile with diffraction limit
- ``plot_psf``                  -- 2D PSF with log-scale option
- ``plot_stokes``               -- 4-panel Stokes parameter map
- ``plot_polarization_ellipses`` -- local polarization ellipses overlaid on intensity
- ``plot_beam_profile``         -- 1D intensity cross-section with D4sigma overlay

All functions return the matplotlib ``Figure`` so they can be further
customized or saved. They do not call ``plt.show()``.

Requires the optional ``matplotlib`` dependency::

    pip install matplotlib

Author: Andrew Traverso
"""

from __future__ import annotations

import importlib.util as _importlib_util
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_MPL_AVAILABLE = _importlib_util.find_spec('matplotlib') is not None
plt = None
Ellipse = Rectangle = None
LogNorm = Normalize = TwoSlopeNorm = None


__all__ = [
    # field / PSF / MTF
    'plot_intensity',
    'plot_phase',
    'plot_field',
    'plot_amplitude_phase',
    'plot_cross_section',
    'plot_planes_grid',
    'plot_psf',
    'plot_mtf',
    'plot_beam_profile',
    # wavefront map (v4.14.0)
    'plot_wavefront',
    # OPD fan + summary panels (v4.15.4)
    'plot_opd_fan',
    'plot_opd_summary',
    # polarization
    'plot_stokes',
    'plot_polarization_ellipses',
    # Jones pupil (vector / Richards-Wolf)
    'compute_jones_pupil',
    'plot_jones_pupil',
    # Lens-layout cross-section (4.7)
    'plot_lens_layout',
    # Glass map / Abbe diagram (4.7)
    'abbe_diagram',
    'plot_glass_map',
]


def _require_mpl():
    global plt, Ellipse, Rectangle, LogNorm, Normalize, TwoSlopeNorm
    if not _MPL_AVAILABLE:
        raise ImportError("matplotlib is required for plotting. "
                          "Install with: pip install matplotlib")
    if plt is None:
        import matplotlib.pyplot as _plt
        from matplotlib.colors import LogNorm as _LogNorm
        from matplotlib.colors import Normalize as _Normalize
        from matplotlib.colors import TwoSlopeNorm as _TwoSlopeNorm
        from matplotlib.patches import Ellipse as _Ellipse
        from matplotlib.patches import Rectangle as _Rectangle
        plt = _plt
        Ellipse, Rectangle = _Ellipse, _Rectangle
        LogNorm, Normalize, TwoSlopeNorm = _LogNorm, _Normalize, _TwoSlopeNorm
        # Update module globals so other functions in this file see them.
        globals().update(plt=plt, Ellipse=Ellipse, Rectangle=Rectangle,
                          LogNorm=LogNorm, Normalize=Normalize,
                          TwoSlopeNorm=TwoSlopeNorm)


def _auto_extent(N, dx, unit='auto'):
    """
    Pick plot extent and unit label from grid size and spacing.

    Returns (extent_tuple, unit_label, scale_factor).
    The scale_factor multiplies meters to reach the chosen display unit.
    """
    L = N * dx  # grid extent in meters
    if unit == 'auto':
        if L < 1e-4:
            unit, scale = 'um', 1e6
        elif L < 0.1:
            unit, scale = 'mm', 1e3
        else:
            unit, scale = 'm', 1.0
    else:
        scales = {'m': 1.0, 'mm': 1e3, 'um': 1e6, 'nm': 1e9}
        if unit not in scales:
            raise ValueError(f"Unknown unit {unit!r}")
        scale = scales[unit]

    half = (N / 2) * dx * scale
    extent = (-half, +half, -half, +half)
    return extent, unit, scale


# =============================================================================
# BASIC FIELD DISPLAYS
# =============================================================================

def plot_intensity(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
    log: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    unit: str = 'auto',
    cmap: str = 'inferno',
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (6, 5),
    colorbar: bool = True,
) -> Tuple[Any, Any]:
    """
    Plot the intensity of a complex field as a 2D image.

    Parameters
    ----------
    E : ndarray (complex, Ny×Nx)
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m]. Defaults to dx.
    log : bool, default False
        Use log-scale color map. Handles zero by clipping below
        ``max(I) * 1e-6``.
    vmin, vmax : float or None
        Color scale limits (absolute intensity values). If None, auto-scaled.
    unit : {'auto', 'm', 'mm', 'um', 'nm'}
        Display unit for axes. 'auto' picks based on grid size.
    cmap : str
        Matplotlib colormap name.
    title : str, optional
    ax : matplotlib Axes, optional
        Plot onto an existing axes. If None, a new figure is created.
    figsize : tuple
    colorbar : bool, default True

    Returns
    -------
    fig : matplotlib Figure
    ax  : matplotlib Axes
    """
    _require_mpl()
    if dy is None:
        dy = dx
    I = np.abs(E)**2
    Ny, Nx = I.shape

    extent, unit_label, _ = _auto_extent(Nx, dx, unit)
    extent_y, _, _ = _auto_extent(Ny, dy, unit)
    full_extent = (extent[0], extent[1], extent_y[2], extent_y[3])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if log:
        I_max = I.max() if I.max() > 0 else 1.0
        floor = I_max * 1e-6
        I_disp = np.maximum(I, floor)
        norm = LogNorm(vmin=vmin or floor, vmax=vmax or I_max)
        im = ax.imshow(I_disp, extent=full_extent, origin='lower',
                       cmap=cmap, norm=norm, aspect='equal')
    else:
        im = ax.imshow(I, extent=full_extent, origin='lower',
                       cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')

    ax.set_xlabel(f'x [{unit_label}]')
    ax.set_ylabel(f'y [{unit_label}]')
    if title:
        ax.set_title(title)
    if colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.85)
        cb.set_label('Intensity' + (' (log)' if log else ''))
    return fig, ax


def plot_phase(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
    mask_threshold: float = 0.01,
    unit: str = 'auto',
    cmap: str = 'twilight',
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (6, 5),
    colorbar: bool = True,
) -> Tuple[Any, Any]:
    """
    Plot the phase of a complex field.

    Parameters
    ----------
    E : ndarray (complex)
    dx, dy : float
    mask_threshold : float, default 0.01
        Intensity threshold (fraction of peak) below which phase is masked
        (shown as transparent / NaN). Set to 0 to show all phases.
    unit : str
    cmap : str, default 'twilight'
        Cyclic colormap recommended for phase.
    title : str, optional
    ax : matplotlib Axes, optional
    figsize : tuple
    colorbar : bool

    Returns
    -------
    fig, ax
    """
    _require_mpl()
    if dy is None:
        dy = dx
    I = np.abs(E)**2
    phi = np.angle(E)
    Ny, Nx = phi.shape

    if mask_threshold > 0 and I.max() > 0:
        phi = np.where(I > mask_threshold * I.max(), phi, np.nan)

    extent, unit_label, _ = _auto_extent(Nx, dx, unit)
    extent_y, _, _ = _auto_extent(Ny, dy, unit)
    full_extent = (extent[0], extent[1], extent_y[2], extent_y[3])

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    im = ax.imshow(phi, extent=full_extent, origin='lower',
                   cmap=cmap, vmin=-np.pi, vmax=np.pi, aspect='equal')
    ax.set_xlabel(f'x [{unit_label}]')
    ax.set_ylabel(f'y [{unit_label}]')
    if title:
        ax.set_title(title)
    if colorbar:
        cb = fig.colorbar(im, ax=ax, shrink=0.85,
                          ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cb.ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        cb.set_label('Phase [rad]')
    return fig, ax


def plot_field(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
    log_intensity: bool = False,
    mask_phase: float = 0.01,
    unit: str = 'auto',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[Any, Any]:
    """
    Plot intensity and phase of a field side-by-side.

    Parameters
    ----------
    E : ndarray (complex)
    dx, dy : float
    log_intensity : bool
    mask_phase : float
        Phase masking threshold (fraction of peak intensity).
    unit : str
    title : str, optional
        Overall figure title (suptitle).
    figsize : tuple

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple of 2 Axes (intensity, phase)
    """
    _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    plot_intensity(E, dx, dy, log=log_intensity, unit=unit,
                   title='Intensity', ax=axes[0])
    plot_phase(E, dx, dy, mask_threshold=mask_phase, unit=unit,
               title='Phase', ax=axes[1])
    if title:
        fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig, axes


def plot_amplitude_phase(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
    unit: str = 'auto',
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 5),
) -> Tuple[Any, Any]:
    """Alias for :func:`plot_field` (intensity + phase side-by-side)."""
    return plot_field(E, dx, dy, unit=unit, title=title, figsize=figsize)


# =============================================================================
# CROSS SECTIONS
# =============================================================================

def plot_cross_section(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
    axis: str = 'x',
    position: float = 0.0,
    unit: str = 'auto',
    log: bool = False,
    show_phase: bool = False,
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[Any, Any]:
    """
    Plot a 1D cross section of a field along x or y.

    Parameters
    ----------
    E : ndarray (complex, Ny×Nx)
    dx : float
    dy : float, optional
    axis : {'x', 'y'}
        Direction of the cut (the other axis is held fixed at ``position``).
    position : float
        Position along the perpendicular axis [m]. 0 = center.
    unit : str
    log : bool, default False
        Log-scale y-axis for the intensity plot.
    show_phase : bool, default False
        If True, overlay the phase on a twin axis.
    title : str, optional
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    _require_mpl()
    if dy is None:
        dy = dx

    I = np.abs(E)**2
    phi = np.angle(E)
    Ny, Nx = I.shape

    # Find row/col index nearest the requested position
    if axis == 'x':
        j = int(round(Ny / 2 + position / dy))
        j = np.clip(j, 0, Ny - 1)
        I_cut = I[j, :]
        phi_cut = phi[j, :]
        coords = (np.arange(Nx) - Nx / 2) * dx
    elif axis == 'y':
        i = int(round(Nx / 2 + position / dx))
        i = np.clip(i, 0, Nx - 1)
        I_cut = I[:, i]
        phi_cut = phi[:, i]
        coords = (np.arange(Ny) - Ny / 2) * dy
    else:
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    extent, unit_label, scale = _auto_extent(len(coords), dx if axis == 'x' else dy, unit)
    coords_disp = coords * scale

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if log:
        # 4.10: guard against all-zero / negative I_cut so semilogy
        # doesn't raise on the boundary case where the entire cross-
        # section is below the floor.
        peak_for_floor = float(I.max())
        if peak_for_floor <= 0:
            # Fall back to linear plot for a degenerate input.
            ax.plot(coords_disp, I_cut, 'r-', lw=1.2, label='Intensity')
        else:
            floor = max(peak_for_floor * 1e-6, 1e-20)
            I_safe = np.where(I_cut > floor, I_cut, floor)
            ax.semilogy(coords_disp, I_safe, 'r-', lw=1.2, label='Intensity')
            ax.set_ylim(bottom=floor)
    else:
        ax.plot(coords_disp, I_cut, 'r-', lw=1.2, label='Intensity')

    ax.set_xlabel(f'{axis} [{unit_label}]')
    ax.set_ylabel('Intensity', color='r')
    ax.tick_params(axis='y', labelcolor='r')
    ax.grid(alpha=0.3)

    if show_phase:
        ax2 = ax.twinx()
        # Mask phase outside the beam
        mask = I_cut > I_cut.max() * 0.01 if I_cut.max() > 0 else np.zeros_like(I_cut, dtype=bool)
        phi_disp = np.where(mask, phi_cut, np.nan)
        ax2.plot(coords_disp, phi_disp, 'b-', lw=1.0, label='Phase')
        ax2.set_ylabel('Phase [rad]', color='b')
        ax2.set_ylim([-np.pi, np.pi])
        ax2.tick_params(axis='y', labelcolor='b')

    if title:
        ax.set_title(title)
    return fig, ax


# =============================================================================
# MULTI-PLANE GRIDS
# =============================================================================

def plot_planes_grid(
    planes: Sequence[Dict[str, Any]],
    n_cols: int = 4,
    log: bool = False,
    unit: str = 'auto',
    cmap: str = 'inferno',
    auto_crop: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    suptitle: Optional[str] = None,
) -> Tuple[Any, Any]:
    """
    Plot a grid of intensity maps for multiple propagation planes.

    Parameters
    ----------
    planes : list of dict
        Each dict must have keys 'field' and 'dx'. Optional: 'dy', 'label',
        'z'. This is the same format produced by :func:`load_planes_h5`.
    n_cols : int, default 4
        Number of columns in the grid.
    log : bool, default False
    unit : str
    cmap : str
    auto_crop : bool, default True
        Crop each panel to the region containing significant intensity.
    figsize : tuple or None
    suptitle : str, optional

    Returns
    -------
    fig : matplotlib Figure
    axes : ndarray of Axes
    """
    _require_mpl()
    n = len(planes)
    n_rows = (n + n_cols - 1) // n_cols
    if figsize is None:
        figsize = (4 * n_cols, 4 * n_rows)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    for i, plane in enumerate(planes):
        r, c = divmod(i, n_cols)
        ax = axes[r, c]
        E = plane['field']
        dx = plane['dx']
        dy = plane.get('dy', dx)
        label = plane.get('label', f'Plane {i}')
        z = plane.get('z', None)

        I = np.abs(E)**2
        Ny, Nx = I.shape

        if auto_crop and I.max() > 0:
            thresh = I.max() * 1e-4
            rows_on = np.any(I > thresh, axis=1)
            cols_on = np.any(I > thresh, axis=0)
            if np.any(rows_on) and np.any(cols_on):
                r0, r1 = np.where(rows_on)[0][[0, -1]]
                c0, c1 = np.where(cols_on)[0][[0, -1]]
                # 4.10: pad each axis from its own extent so elongated
                # beams get a sensible margin on both sides; pre-4.10
                # used the row extent for both axes, leaving uneven
                # margins or cropping into columnar features.
                pad_r = max((r1 - r0) * 0.15, 20)
                pad_c = max((c1 - c0) * 0.15, 20)
                r0 = max(0, int(r0 - pad_r))
                r1 = min(Ny - 1, int(r1 + pad_r))
                c0 = max(0, int(c0 - pad_c))
                c1 = min(Nx - 1, int(c1 + pad_c))
                I = I[r0:r1 + 1, c0:c1 + 1]
                x = (np.arange(Nx) - Nx / 2) * dx
                y = (np.arange(Ny) - Ny / 2) * dy
                _, unit_label, scale = _auto_extent(Nx, dx, unit)
                ext = (x[c0] * scale, x[c1] * scale,
                       y[r0] * scale, y[r1] * scale)
            else:
                _, unit_label, scale = _auto_extent(Nx, dx, unit)
                ext = (-(Nx / 2) * dx * scale, (Nx / 2) * dx * scale,
                       -(Ny / 2) * dy * scale, (Ny / 2) * dy * scale)
        else:
            _, unit_label, scale = _auto_extent(Nx, dx, unit)
            ext = (-(Nx / 2) * dx * scale, (Nx / 2) * dx * scale,
                   -(Ny / 2) * dy * scale, (Ny / 2) * dy * scale)

        if log:
            I_max = I.max() if I.max() > 0 else 1.0
            norm = LogNorm(vmin=I_max * 1e-4, vmax=I_max)
            ax.imshow(np.maximum(I, I_max * 1e-4), extent=ext, origin='lower',
                      cmap=cmap, norm=norm, aspect='equal')
        else:
            ax.imshow(I, extent=ext, origin='lower', cmap=cmap, aspect='equal')

        title = label
        if z is not None:
            title += f'\nz = {z * 1e3:.1f} mm'
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f'x [{unit_label}]', fontsize=8)
        ax.set_ylabel(f'y [{unit_label}]', fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for i in range(n, n_rows * n_cols):
        r, c = divmod(i, n_cols)
        axes[r, c].set_visible(False)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig, axes


# =============================================================================
# PSF / MTF
# =============================================================================

def plot_psf(
    psf: np.ndarray,
    dx_psf: Optional[float] = None,
    log: bool = True,
    extent_um: Optional[float] = None,
    cmap: str = 'inferno',
    title: str = 'PSF',
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (6, 5),
) -> Tuple[Any, Any]:
    """
    Plot a 2D PSF.

    Parameters
    ----------
    psf : ndarray (real, N×N)
        Intensity PSF (real, non-negative), e.g. from ``compute_psf``
        (which defaults to ``normalize='power'``, NOT peak).  The input
        normalization is cosmetic here: the log display re-clips relative
        to ``psf.max()``, so power- / peak- / un-normalized inputs render
        the same.
    dx_psf : float, optional
        PSF-plane grid spacing [m]. Used for axis labels in micrometers.
    log : bool, default True
        Log-scale (clipped at peak*1e-6).
    extent_um : float, optional
        Half-extent to show in micrometers. If None, shows full array.
    cmap : str
    title : str
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    N = psf.shape[0]
    if dx_psf is not None:
        x_um = (np.arange(N) - N / 2) * dx_psf * 1e6
        extent = (x_um[0], x_um[-1], x_um[0], x_um[-1])
        ax.set_xlabel('x [µm]')
        ax.set_ylabel('y [µm]')
    else:
        extent = None
        ax.set_xlabel('pixel')
        ax.set_ylabel('pixel')

    if log:
        disp = np.maximum(psf, psf.max() * 1e-6)
        im = ax.imshow(disp, extent=extent, origin='lower', cmap=cmap,
                       norm=LogNorm(vmin=psf.max() * 1e-6, vmax=psf.max()))
    else:
        im = ax.imshow(psf, extent=extent, origin='lower', cmap=cmap)

    if extent_um is not None and dx_psf is not None:
        ax.set_xlim([-extent_um, extent_um])
        ax.set_ylim([-extent_um, extent_um])

    fig.colorbar(im, ax=ax, shrink=0.85)
    ax.set_title(title)
    return fig, ax


def plot_mtf(
    freq: np.ndarray,
    mtf_profile: np.ndarray,
    diffraction_limit: Optional[float] = None,
    title: str = 'MTF',
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[Any, Any]:
    """
    Plot a radial MTF profile.

    Parameters
    ----------
    freq : ndarray
        Spatial frequency in cycles/mm (from mtf_radial).
    mtf_profile : ndarray
        MTF values, normalized to 1 at DC.
    diffraction_limit : float, optional
        Cutoff frequency [cycles/mm] for the diffraction limit. If given,
        a theoretical MTF curve is overlaid for comparison.
    title : str
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    _require_mpl()
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(freq, mtf_profile, 'b-', lw=1.5, label='Measured')

    if diffraction_limit is not None:
        # Theoretical MTF for a circular aperture
        nu = freq / diffraction_limit
        nu_clip = np.clip(nu, 0, 1)
        theory = (2 / np.pi) * (np.arccos(nu_clip)
                                - nu_clip * np.sqrt(1 - nu_clip**2))
        theory[nu > 1] = 0.0
        ax.plot(freq, theory, 'k--', lw=1.0, label='Diffraction limit')
        ax.axvline(diffraction_limit, color='gray', ls=':',
                   label=f'Cutoff = {diffraction_limit:.0f} cyc/mm')

    ax.set_xlabel('Spatial frequency [cycles/mm]')
    ax.set_ylabel('MTF')
    ax.set_ylim([0, 1.05])
    ax.set_xlim([0, freq.max()])
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right')
    ax.set_title(title)
    return fig, ax


# =============================================================================
# POLARIZATION
# =============================================================================

def plot_stokes(
    jones_field: Any,
    dx: Optional[float] = None,
    unit: str = 'auto',
    figsize: Tuple[float, float] = (12, 10),
    suptitle: str = 'Stokes parameters',
) -> Tuple[Any, Any]:
    """
    Plot the four Stokes parameters (S0, S1, S2, S3) for a JonesField.

    Parameters
    ----------
    jones_field : JonesField
    dx : float, optional
        Grid spacing. If None, uses jones_field.dx.
    unit : str
    figsize : tuple
    suptitle : str

    Returns
    -------
    fig, axes
    """
    _require_mpl()
    from ..elements.polarization import stokes_parameters
    S = stokes_parameters(jones_field)
    if dx is None:
        dx = jones_field.dx

    Ny, Nx = S['S0'].shape
    extent, unit_label, _ = _auto_extent(Nx, dx, unit)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    names = ['S0', 'S1', 'S2', 'S3']
    titles = [
        'S0 (total intensity)',
        'S1 (H vs V)',
        'S2 (±45° linear)',
        'S3 (circular)',
    ]
    for i, (name, t) in enumerate(zip(names, titles)):
        r, c = divmod(i, 2)
        ax = axes[r, c]
        data = S[name]
        if name == 'S0':
            im = ax.imshow(data, extent=extent, origin='lower',
                           cmap='inferno', aspect='equal')
        else:
            vmax = max(abs(data.min()), abs(data.max()))
            vmax = vmax if vmax > 0 else 1.0
            im = ax.imshow(data, extent=extent, origin='lower',
                           cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           aspect='equal')
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(t)
        ax.set_xlabel(f'x [{unit_label}]')
        ax.set_ylabel(f'y [{unit_label}]')

    fig.suptitle(suptitle, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig, axes


def plot_polarization_ellipses(
    jones_field: Any,
    n_ellipses: int = 16,
    unit: str = 'auto',
    ellipse_scale: float = 0.8,
    intensity_alpha: float = 0.7,
    cmap: str = 'inferno',
    title: str = 'Polarization ellipses',
    figsize: Tuple[float, float] = (8, 8),
) -> Tuple[Any, Any]:
    """
    Plot the local polarization ellipse on top of the intensity image.

    Parameters
    ----------
    jones_field : JonesField
    n_ellipses : int, default 16
        Number of ellipses per side (a grid of n_ellipses × n_ellipses).
    unit : str
    ellipse_scale : float, default 0.8
        Scale factor for the ellipse size (fraction of grid-cell spacing).
    intensity_alpha : float, default 0.7
    cmap : str
    title : str
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    _require_mpl()
    from ..elements.polarization import polarization_ellipse

    fig, ax = plt.subplots(figsize=figsize)

    I = jones_field.intensity()
    Ny, Nx = I.shape
    dx = jones_field.dx
    dy = jones_field.dy

    extent, unit_label, scale = _auto_extent(Nx, dx, unit)
    ext_y, _, _ = _auto_extent(Ny, dy, unit)
    full_extent = (extent[0], extent[1], ext_y[2], ext_y[3])

    ax.imshow(I, extent=full_extent, origin='lower', cmap=cmap,
              alpha=intensity_alpha, aspect='equal')

    orientation, ellipticity = polarization_ellipse(jones_field)

    step_x = max(1, Nx // n_ellipses)
    step_y = max(1, Ny // n_ellipses)
    cell_size_x = step_x * dx * scale
    cell_size_y = step_y * dy * scale
    ellipse_size = min(cell_size_x, cell_size_y) * ellipse_scale

    x = (np.arange(Nx) - Nx / 2) * dx * scale
    y = (np.arange(Ny) - Ny / 2) * dy * scale

    I_thresh = I.max() * 0.01 if I.max() > 0 else 0.0

    for iy in range(step_y // 2, Ny, step_y):
        for ix in range(step_x // 2, Nx, step_x):
            if I[iy, ix] < I_thresh:
                continue
            psi = orientation[iy, ix]
            chi = ellipticity[iy, ix]
            # Major axis = ellipse_size/2, minor = (ellipse_size/2)*|tan(chi)|
            a = ellipse_size / 2
            b = a * abs(np.tan(chi))
            e = Ellipse(
                (x[ix], y[iy]),
                width=2 * a, height=2 * b,
                angle=np.degrees(psi),
                edgecolor='cyan', facecolor='none', lw=0.8,
            )
            ax.add_patch(e)

    ax.set_xlabel(f'x [{unit_label}]')
    ax.set_ylabel(f'y [{unit_label}]')
    ax.set_title(title)
    return fig, ax


# =============================================================================
# BEAM PROFILE
# =============================================================================

def plot_beam_profile(
    E: np.ndarray,
    dx: float,
    dy: Optional[float] = None,
    axis: str = 'x',
    show_d4sigma: bool = True,
    fit_gaussian: bool = False,
    unit: str = 'auto',
    title: Optional[str] = None,
    ax: Optional[Any] = None,
    figsize: Tuple[float, float] = (8, 5),
) -> Tuple[Any, Any]:
    """
    Plot a 1D beam intensity profile through the centroid.

    Parameters
    ----------
    E : ndarray (complex)
    dx, dy : float
    axis : {'x', 'y'}
    show_d4sigma : bool, default True
        Overlay D4σ markers.
    fit_gaussian : bool, default False
        Fit and overlay a Gaussian profile.
    unit : str
    title : str, optional
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    fig, ax
    """
    _require_mpl()
    from .core import beam_centroid, beam_d4sigma

    if dy is None:
        dy = dx

    I = np.abs(E)**2
    Ny, Nx = I.shape
    cx, cy = beam_centroid(E, dx, dy)

    # Cross-section through the centroid
    if axis == 'x':
        j = int(round(Ny / 2 + cy / dy))
        j = np.clip(j, 0, Ny - 1)
        I_cut = I[j, :]
        coords = (np.arange(Nx) - Nx / 2) * dx
    else:
        i = int(round(Nx / 2 + cx / dx))
        i = np.clip(i, 0, Nx - 1)
        I_cut = I[:, i]
        coords = (np.arange(Ny) - Ny / 2) * dy

    # 4.10: use dy for the y-axis-pitch when axis='y' so the axis
    # label / scaling matches the actual sample pitch on anamorphic
    # grids (dx != dy).
    axis_dx = dx if axis == 'x' else dy
    _, unit_label, scale = _auto_extent(len(coords), axis_dx, unit)
    coords_disp = coords * scale

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(coords_disp, I_cut, 'b-', lw=1.5, label='Intensity')

    if show_d4sigma:
        d4x, d4y = beam_d4sigma(E, dx, dy)
        d4 = d4x if axis == 'x' else d4y
        c = cx if axis == 'x' else cy
        ax.axvline((c - d4 / 2) * scale, color='r', ls='--',
                   label=f'D4σ = {d4 * scale:.1f} {unit_label}')
        ax.axvline((c + d4 / 2) * scale, color='r', ls='--')

    if fit_gaussian:
        peak = I_cut.max()
        if peak > 0:
            # Simple moment fit
            norm = I_cut.sum()
            if norm > 0:
                mean = (coords * I_cut).sum() / norm
                var = ((coords - mean)**2 * I_cut).sum() / norm
                sigma = np.sqrt(var)
                gauss = peak * np.exp(-(coords - mean)**2 / (2 * sigma**2))
                ax.plot(coords_disp, gauss, 'g--', lw=1.0,
                        label=f'Gaussian fit (σ={sigma * scale:.1f} {unit_label})')

    ax.set_xlabel(f'{axis} [{unit_label}]')
    ax.set_ylabel('Intensity')
    ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)
    if title:
        ax.set_title(title)
    return fig, ax


# =============================================================================
# WAVEFRONT MAP (v4.14.0)
# =============================================================================

def plot_wavefront(
    opd: np.ndarray,
    dx: float,
    *,
    dy: Optional[float] = None,
    aperture: Optional[np.ndarray] = None,
    units: str = 'waves',
    wavelength: Optional[float] = None,
    cmap: str = 'RdBu_r',
    show_stats: bool = True,
    ax: Optional[Any] = None,
    fig: Optional[Any] = None,
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Plot a 2-D wavefront / OPD map (Zemax-style).

    Renders an OPD array with a divergent colormap centred at zero, the
    grid axes labelled in physical units, and (optionally) a PV / RMS
    annotation overlaid on the plot.  Pixels outside the supplied
    ``aperture`` mask are rendered as ``NaN`` so they show as the
    figure's background colour (matching standard wavefront-map
    conventions in Zemax / Code V / Synopsys CODE V).

    Parameters
    ----------
    opd : ndarray, shape (Ny, Nx)
        Optical-path-difference map.  Units are interpreted via the
        ``units`` / ``wavelength`` kwargs.  ``NaN`` entries are passed
        through unchanged so callers that have pre-masked their OPD
        can still annotate stats correctly.
    dx : float
        Grid spacing in x [m].
    dy : float, optional
        Grid spacing in y [m].  Defaults to ``dx``.
    aperture : ndarray bool, optional
        Boolean (or 0/1) mask the same shape as ``opd``.  Pixels with
        ``aperture == False`` are set to ``NaN`` before plotting so
        they are visually suppressed.  If ``None``, the full grid is
        rendered.
    units : {'waves', 'nm', 'um', 'm'}, default 'waves'
        Units for the colour scale and the PV / RMS annotation.
        ``'waves'`` requires a ``wavelength`` keyword; the input OPD
        is divided by ``wavelength`` to convert metres to waves.  The
        other unit choices apply a fixed scale factor to the OPD
        before plotting (``'nm'`` x 1e9, ``'um'`` x 1e6).
    wavelength : float, optional
        Vacuum wavelength [m].  Required when ``units == 'waves'``.
    cmap : str, default ``'RdBu_r'``
        Matplotlib colormap.  Divergent colormaps are recommended so
        zero OPD shows as a neutral mid-tone.
    show_stats : bool, default True
        Overlay a "PV: ..., RMS: ..." annotation in the upper-left of
        the plot.  Stats are computed on the (masked) OPD in the
        chosen display units.
    ax : matplotlib Axes, optional
        Plot onto an existing axes.  A new figure is created if
        ``ax`` is ``None``.
    fig : matplotlib Figure, optional
        Existing figure handle.  Used when ``ax`` is also supplied
        from the same figure; otherwise derived from ``ax`` or a new
        figure.
    title : str, optional
        Plot title.

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Examples
    --------
    >>> import numpy as np
    >>> from lumenairy.analysis import plot_wavefront
    >>> import matplotlib
    >>> matplotlib.use('Agg')
    >>> N = 64
    >>> x = (np.arange(N) - N / 2) / (N / 2)
    >>> X, Y = np.meshgrid(x, x)
    >>> ap = (X**2 + Y**2) <= 1.0
    >>> opd = 0.5 * (X**2 + Y**2)               # defocus, in waves
    >>> fig, ax = plot_wavefront(
    ...     opd, dx=1e-6, aperture=ap,
    ...     units='waves', wavelength=632.8e-9)  # doctest: +SKIP
    """
    _require_mpl()
    if dy is None:
        dy = dx
    if opd.ndim != 2:
        raise ValueError(
            f"plot_wavefront: opd must be 2-D; got shape {opd.shape!r}.")

    # Resolve display units.
    if units == 'waves':
        if wavelength is None or not np.isfinite(wavelength) \
                or wavelength <= 0:
            raise ValueError(
                "plot_wavefront: units='waves' requires a positive "
                "'wavelength' keyword [m].")
        scale = 1.0 / float(wavelength)
        units_label = 'waves'
    elif units == 'nm':
        scale = 1e9
        units_label = 'nm'
    elif units == 'um':
        scale = 1e6
        units_label = 'um'
    elif units == 'm':
        scale = 1.0
        units_label = 'm'
    else:
        raise ValueError(
            f"plot_wavefront: units must be one of "
            f"{{'waves', 'nm', 'um', 'm'}}; got {units!r}.")

    opd_disp = np.asarray(opd, dtype=float) * scale

    # Aperture masking -- set pixels outside the aperture to NaN so
    # they're suppressed in the imshow.
    if aperture is not None:
        if aperture.shape != opd.shape:
            raise ValueError(
                f"plot_wavefront: aperture shape {aperture.shape!r} "
                f"!= opd shape {opd.shape!r}.")
        mask = np.asarray(aperture).astype(bool)
        opd_disp = np.where(mask, opd_disp, np.nan)

    # Figure / axes plumbing -- match plot_intensity / plot_phase.
    if ax is None:
        if fig is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            ax = fig.add_subplot(111)
    else:
        if fig is None:
            fig = ax.get_figure()

    Ny, Nx = opd_disp.shape
    extent, unit_label, _ = _auto_extent(Nx, dx, 'auto')
    extent_y, _, _ = _auto_extent(Ny, dy, 'auto')
    full_extent = (extent[0], extent[1], extent_y[2], extent_y[3])

    # Symmetric colour limits about zero so the divergent colormap
    # reads as "positive vs negative OPD".  Use TwoSlopeNorm to keep
    # the centre at exactly zero even when the data is asymmetric;
    # fall back to a symmetric vmin/vmax for all-NaN inputs.
    finite = opd_disp[np.isfinite(opd_disp)]
    if finite.size == 0:
        vlim = 1.0
    else:
        vlim = float(np.nanmax(np.abs(finite)))
        if vlim == 0.0 or not np.isfinite(vlim):
            vlim = 1.0

    im = ax.imshow(opd_disp, extent=full_extent, origin='lower',
                   cmap=cmap, vmin=-vlim, vmax=vlim, aspect='equal')

    ax.set_xlabel(f'x [{unit_label}]')
    ax.set_ylabel(f'y [{unit_label}]')
    if title:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label(f'OPD [{units_label}]')

    if show_stats:
        # Compute PV / RMS on the *displayed* (masked, scaled) array
        # so the annotation matches the colour-bar units.  Use NaN-
        # aware reductions so masked pixels don't leak in.
        if finite.size > 0:
            pv = float(np.nanmax(opd_disp) - np.nanmin(opd_disp))
            rms = float(np.sqrt(np.nanmean(
                (opd_disp - np.nanmean(opd_disp)) ** 2)))
            ax.text(
                0.02, 0.98,
                f'PV: {pv:.3g} {units_label}\n'
                f'RMS: {rms:.3g} {units_label}',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.75,
                          edgecolor='gray', lw=0.5),
            )

    return fig, ax


# =============================================================================
# OPD FAN + SUMMARY PANELS (v4.15.4)
# =============================================================================


def _resolve_opd_units(
    units: str,
    wavelength: Optional[float],
    *,
    fn_name: str = 'plot_opd_fan',
) -> Tuple[float, str]:
    """Convert ``units`` kwarg to a (scale, units_label) pair.

    Mirrors the unit-resolution block inside :func:`plot_wavefront` so
    the OPD-fan helpers display data on the same axes.  ``scale``
    multiplies the input OPD (assumed in metres) to land in the chosen
    display unit.
    """
    if units == 'waves':
        if wavelength is None or not np.isfinite(wavelength) \
                or wavelength <= 0:
            raise ValueError(
                f"{fn_name}: units='waves' requires a positive "
                "'wavelength' keyword [m].")
        return 1.0 / float(wavelength), 'waves'
    if units == 'nm':
        return 1e9, 'nm'
    if units == 'um':
        return 1e6, 'um'
    if units == 'm':
        return 1.0, 'm'
    raise ValueError(
        f"{fn_name}: units must be one of "
        f"{{'waves', 'nm', 'um', 'm'}}; got {units!r}.")


def _opd_pv_rms_1d(arr: np.ndarray) -> Tuple[float, float]:
    """Compute (PV, RMS) of a 1-D OPD fan over its finite (alive)
    samples.  Returns ``(nan, nan)`` if no finite values exist.

    PV = max - min over the finite mask; RMS is the unbiased
    deviation about the in-aperture mean (matches the convention in
    :func:`plot_wavefront`).
    """
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float('nan'), float('nan')
    pv = float(finite.max() - finite.min())
    rms = float(np.sqrt(np.mean((finite - finite.mean()) ** 2)))
    return pv, rms


def _render_opd_fan_panel(
    ax: Any,
    p: np.ndarray,
    opd: np.ndarray,
    units_label: str,
    title: str,
    show_stats: bool,
) -> None:
    """Render a single OPD-fan panel onto an existing matplotlib axis.

    Internal helper used by both :func:`plot_opd_fan` and
    :func:`plot_opd_summary` so the panels share a consistent look
    (solid line, zero reference, grid, optional PV / RMS overlay).
    """
    ax.plot(p, opd, color='C0', lw=1.5)
    ax.axhline(0.0, color='gray', lw=0.5, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.set_xlabel('Normalised pupil coordinate')
    ax.set_ylabel(f'OPD ({units_label})')
    ax.set_title(title)

    if show_stats:
        pv, rms = _opd_pv_rms_1d(opd)
        if np.isfinite(pv) and np.isfinite(rms):
            ax.text(
                0.02, 0.98,
                f'PV: {pv:.3g} {units_label}\n'
                f'RMS: {rms:.3g} {units_label}',
                transform=ax.transAxes,
                ha='left', va='top',
                fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.75,
                          edgecolor='gray', lw=0.5),
            )


def plot_opd_fan(
    py: np.ndarray,
    opd_y: np.ndarray,
    px: np.ndarray,
    opd_x: np.ndarray,
    *,
    wavelength: Optional[float] = None,
    units: str = 'waves',
    fan_units: str = 'm',
    show_stats: bool = True,
    title: Optional[str] = None,
    fig: Optional[Any] = None,
    axes: Optional[Tuple[Any, Any]] = None,
) -> Tuple[Any, Tuple[Any, Any]]:
    """Plot OPD ray-fan curves (tangential + sagittal).

    Renders the canonical two-panel OPD-vs-pupil diagram used in
    classical Fourier-optics references and in the cross-check
    figures at ``OPDPy_Lumenairy_Crosscheck/fig_variety_L*.png``.
    The left panel shows the tangential (y-fan, ``opd_y`` vs ``py``);
    the right panel shows the sagittal (x-fan, ``opd_x`` vs ``px``).

    Parameters
    ----------
    py : ndarray, shape (n_rays,)
        Normalised pupil y-coordinate of the tangential fan in
        ``[-1, 1]``.
    opd_y : ndarray, shape (n_rays,)
        Tangential-fan OPD samples.  Interpreted in the unit
        selected by ``fan_units`` (METRES by default, or WAVES if
        ``fan_units='waves'``).  ``NaN`` entries (rays outside the
        aperture, vignetted, or TIR'd) are passed through unchanged
        so the PV / RMS overlay reports stats over the in-aperture
        portion only.
    px : ndarray, shape (n_rays,)
        Normalised pupil x-coordinate of the sagittal fan in
        ``[-1, 1]``.
    opd_x : ndarray, shape (n_rays,)
        Sagittal-fan OPD samples.  Same unit handling as ``opd_y``.
    wavelength : float, optional
        Vacuum wavelength [m].  Required when ``units == 'waves'``
        or when ``fan_units == 'waves'`` and ``units != 'waves'``.
    units : {'waves', 'nm', 'um', 'm'}, default 'waves'
        DISPLAY unit for the y-axis and the PV / RMS overlay.
        Controls how the (already-converted-to-metres) OPD is
        re-scaled before plotting.  ``'waves'`` requires a positive
        ``wavelength`` and divides by it; ``'nm'`` / ``'um'`` apply
        a fixed scale factor; ``'m'`` is the identity.
    fan_units : {'m', 'waves'}, default ``'m'``
        INPUT unit of ``opd_y`` / ``opd_x``.  ``'m'`` (the v4.15.4
        default; preserved here for back-compatibility) interprets
        the arrays as metres.  ``'waves'`` interprets them as waves
        of the trace wavelength -- this matches what
        :func:`lumenairy.raytrace.opd_fan_data` returns natively, so
        the canonical pipeline

            py, opd_y, px, opd_x = la.opd_fan_data(...)
            plot_opd_fan(py, opd_y, px, opd_x,
                         units='waves', wavelength=wl,
                         fan_units='waves')

        renders without the silent double-conversion trap.  Common
        pitfall (v4.15.4 audit): mixing ``opd_fan_data`` (returns
        waves) with the default ``fan_units='m'`` divides by
        ``wavelength`` twice, producing values smaller by ~6e5 than
        intended.  v4.15.5 (P1-NEW-V2-1): introduce ``fan_units``
        so the trap is callable-out via a single kwarg.
    show_stats : bool, default True
        Overlay a "PV: ..., RMS: ..." annotation in the upper-left
        of each panel, computed over the finite (alive) samples.
        PV: max - min over finite (in-aperture) pixels.
        RMS: centered RMS about the in-aperture mean (matches
        :func:`plot_wavefront` convention so the 2-D heatmap and
        the 1-D fan RMS numbers reconcile).
    title : str, optional
        Optional figure-level suptitle (passed to ``fig.suptitle``).
        Per-panel titles are always "y-fan (tangential)" and
        "x-fan (sagittal)".
    fig : matplotlib Figure, optional
        Existing figure handle.  When supplied without ``axes``, two
        new subplots are added to it.  Otherwise a new ``(12, 4.5)``
        figure is created.
    axes : tuple of 2 Axes, optional
        Existing ``(ax_y, ax_x)`` axes to render into.  Both must
        belong to the same figure (``fig`` is inferred from
        ``axes[0]`` when not supplied).

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple ``(ax_y, ax_x)``
        ``ax_y`` is the tangential panel, ``ax_x`` is the sagittal
        panel.

    See Also
    --------
    lumenairy.raytrace.opd_fan_data : compute the input fans (returns
        OPD in WAVES; pass ``fan_units='waves'`` here).
    plot_opd_summary : 4-panel 2-D heatmap + radial RMS + 2 fans.
    plot_wavefront : 2-D OPD heatmap.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib
    >>> matplotlib.use('Agg')
    >>> from lumenairy.analysis import plot_opd_fan
    >>> n = 51
    >>> p = np.linspace(-1, 1, n)
    >>> opd_y = 0.5 * p ** 2  # defocus, in waves
    >>> opd_x = 0.5 * p ** 2
    >>> fig, (ax_y, ax_x) = plot_opd_fan(
    ...     p, opd_y, p, opd_x,
    ...     units='waves', wavelength=633e-9,
    ...     fan_units='waves')  # doctest: +SKIP
    """
    _require_mpl()

    if fan_units not in ('m', 'waves'):
        raise ValueError(
            f"plot_opd_fan: fan_units must be 'm' or 'waves'; "
            f"got {fan_units!r}.")

    py = np.asarray(py, dtype=float)
    opd_y_arr = np.asarray(opd_y, dtype=float)
    px = np.asarray(px, dtype=float)
    opd_x_arr = np.asarray(opd_x, dtype=float)

    if py.ndim != 1 or opd_y_arr.ndim != 1 \
            or px.ndim != 1 or opd_x_arr.ndim != 1:
        raise ValueError(
            "plot_opd_fan: py, opd_y, px, opd_x must all be 1-D arrays; "
            f"got shapes {py.shape!r}, {opd_y_arr.shape!r}, "
            f"{px.shape!r}, {opd_x_arr.shape!r}.")
    if py.shape != opd_y_arr.shape:
        raise ValueError(
            f"plot_opd_fan: py and opd_y must have the same shape; "
            f"got {py.shape!r} vs {opd_y_arr.shape!r}.")
    if px.shape != opd_x_arr.shape:
        raise ValueError(
            f"plot_opd_fan: px and opd_x must have the same shape; "
            f"got {px.shape!r} vs {opd_x_arr.shape!r}.")

    # v4.15.5 (P1-NEW-V2-1): pre-convert ``fan_units='waves'`` inputs
    # to metres so the existing metres-based ``_resolve_opd_units``
    # branch stays the single source of truth for display scaling.
    # ``fan_units='waves'`` requires a positive wavelength regardless
    # of the display ``units`` (we need it to go waves -> metres
    # first), so we validate it explicitly here.
    if fan_units == 'waves':
        if (wavelength is None or not np.isfinite(wavelength)
                or wavelength <= 0):
            raise ValueError(
                "plot_opd_fan: fan_units='waves' requires a positive "
                "'wavelength' keyword [m] to convert input waves to "
                "metres before the display-units conversion.")
        opd_y_arr = opd_y_arr * float(wavelength)
        opd_x_arr = opd_x_arr * float(wavelength)

    scale, units_label = _resolve_opd_units(units, wavelength,
                                            fn_name='plot_opd_fan')
    opd_y_disp = opd_y_arr * scale
    opd_x_disp = opd_x_arr * scale

    if axes is not None:
        ax_y, ax_x = axes
        if fig is None:
            fig = ax_y.get_figure()
    else:
        if fig is None:
            fig, (ax_y, ax_x) = plt.subplots(1, 2, figsize=(12, 4.5))
        else:
            ax_y = fig.add_subplot(1, 2, 1)
            ax_x = fig.add_subplot(1, 2, 2)

    _render_opd_fan_panel(ax_y, py, opd_y_disp, units_label,
                          'y-fan (tangential)', show_stats)
    _render_opd_fan_panel(ax_x, px, opd_x_disp, units_label,
                          'x-fan (sagittal)', show_stats)

    if title:
        fig.suptitle(title)

    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        # tight_layout can refuse on degenerate layouts (e.g. an
        # already-laid-out parent figure passed in by the caller).
        # The figure is still returned and renderable.
        pass

    return fig, (ax_y, ax_x)


def _radial_rms_profile(
    opd: np.ndarray,
    dx: float,
    dy: float,
    aperture: Optional[np.ndarray],
    n_bins: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute centered RMS(OPD) as a function of normalised pupil radius.

    Bins the (masked) 2-D OPD into ``n_bins`` annular rings between
    radius 0 and 1 (normalised by the maximum in-aperture radius).
    Returns ``(r_centres, rms_per_bin)`` with NaN-handling for empty
    or all-NaN bins.

    v4.15.5 (P1-NEW-V2-2): the per-bin reduction is centered about
    the SINGLE in-aperture mean of the entire OPD (the same baseline
    used by :func:`plot_wavefront`'s 2-D RMS annotation and by the
    1-D fan RMS in :func:`_opd_pv_rms_1d`).  Pre-v4.15.5 used the
    uncentered form ``sqrt(mean(opd**2))`` per bin, which made the
    radial curve piston-dominated and irreconcilable with the
    heatmap annotation on the same 4-panel figure.

    Using the GLOBAL in-aperture mean (rather than a per-annulus
    local mean) preserves the radial structure: a per-bin local mean
    would zero-out every annulus by construction, returning a flat
    line at machine epsilon for any axisymmetric OPD.

    Internal helper used by :func:`plot_opd_summary`.
    """
    Ny, Nx = opd.shape
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    # S11-6f NOT-CHANGED (AUDIT_SIBLING_PATTERN_SWEEP_2026_07_25 §1): this
    # radial anchor is the ARRAY centre ``(N-1)/2`` while the library's field
    # grids are centred on ``N/2`` (``x_j = (j - N/2)*dx``), a half-pixel
    # disagreement for even N.  Changing it shifts every radial bin on every
    # even-N input, so it is NOT bit-identical and is left for a deliberate
    # grid-convention pass; the two differ by dx/2, well inside the annulus
    # width this profile bins into.
    cy = (Ny - 1) / 2.0
    cx = (Nx - 1) / 2.0
    rx = (xx - cx) * dx
    ry = (yy - cy) * dy
    r = np.sqrt(rx ** 2 + ry ** 2)

    if aperture is not None:
        mask = np.asarray(aperture, dtype=bool)
    else:
        mask = np.isfinite(opd)

    if not np.any(mask):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return centres, np.full(n_bins, np.nan)

    r_max = float(r[mask].max())
    if r_max <= 0.0 or not np.isfinite(r_max):
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        centres = 0.5 * (edges[:-1] + edges[1:])
        return centres, np.full(n_bins, np.nan)

    r_norm = r / r_max
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])

    # v4.15.5: compute the in-aperture mean ONCE so every annulus is
    # centered about the same baseline (matches plot_wavefront RMS
    # and the 1-D fan RMS).
    in_aperture_values = opd[mask]
    in_aperture_values = in_aperture_values[np.isfinite(in_aperture_values)]
    if in_aperture_values.size == 0:
        return centres, np.full(n_bins, np.nan)
    in_aperture_mean = float(in_aperture_values.mean())

    out = np.full(n_bins, np.nan, dtype=float)
    for i in range(n_bins):
        bin_mask = mask & (r_norm >= edges[i]) & (r_norm < edges[i + 1])
        if i == n_bins - 1:
            # Include the right endpoint in the last bin.
            bin_mask = mask & (r_norm >= edges[i]) & (r_norm <= edges[i + 1])
        values = opd[bin_mask]
        values = values[np.isfinite(values)]
        if values.size > 0:
            centered = values - in_aperture_mean
            out[i] = float(np.sqrt(np.mean(centered ** 2)))
    return centres, out


def _auto_n_bins(n_in_aperture: int, ceiling: int = 32) -> int:
    """Pick a sensible ``n_bins`` for :func:`_radial_rms_profile`.

    v4.15.5 (P3-NEW-F1-5): on tiny pupil grids (e.g. 16x16) a fixed
    32-bin radial profile produces a visibly fragmented line because
    11 of 32 bins are empty.  Clamp to ``min(ceiling,
    int(sqrt(N_in_aperture)) // 2)`` so the resolution scales with
    the available pixel count.  Always returns ``>= 1``.
    """
    if n_in_aperture <= 0:
        return 1
    candidate = int(np.sqrt(float(n_in_aperture))) // 2
    return max(1, min(int(ceiling), candidate))


def plot_opd_summary(
    opd_2d: np.ndarray,
    dx: float,
    *,
    dy: Optional[float] = None,
    aperture: Optional[np.ndarray] = None,
    wavelength: Optional[float] = None,
    py: Optional[np.ndarray] = None,
    opd_y: Optional[np.ndarray] = None,
    px: Optional[np.ndarray] = None,
    opd_x: Optional[np.ndarray] = None,
    units: str = 'waves',
    cmap: str = 'RdBu_r',
    show_stats: bool = True,
    title: Optional[str] = None,
    fig: Optional[Any] = None,
    radial_rms_n_bins: Any = 'auto',
) -> Tuple[Any, Tuple[Tuple[Any, Any], Tuple[Any, Any]]]:
    """Plot a 4-panel OPD summary: heatmap + radial RMS + 2 fans.

    Lays out a 2x2 grid:

    +-----------------------+----------------------+
    | (0, 0) 2-D OPD map    | (0, 1) Radial RMS    |
    +-----------------------+----------------------+
    | (1, 0) Tangential fan | (1, 1) Sagittal fan  |
    +-----------------------+----------------------+

    The heatmap panel delegates to :func:`plot_wavefront`; the fan
    panels delegate to the same internal renderer used by
    :func:`plot_opd_fan`.  When ``(py, opd_y, px, opd_x)`` are all
    supplied (e.g., from :func:`lumenairy.raytrace.opd_fan_data`) the
    fans are taken from those arrays directly -- preferred because
    the ray-trace pipeline already references the chief ray.  If the
    fan arrays are not supplied, the function falls back to the
    central row and column of ``opd_2d`` so the figure still
    populates for callers that only have the 2-D OPD.

    Parameters
    ----------
    opd_2d : ndarray, shape (Ny, Nx)
        2-D OPD map in METRES (same convention as
        :func:`plot_wavefront`).

        .. note::

           The input is in METRES regardless of the ``units`` kwarg.
           ``units`` controls only the DISPLAY conversion -- it does
           NOT alter the interpretation of ``opd_2d``.  v4.15.4
           docstring said "interpreted in the units system" which
           was misleading; v4.15.5 (P2-NEW-V2-3) corrects the
           wording.
    dx : float
        Pupil grid spacing in x [m].
    dy : float, optional
        Pupil grid spacing in y [m].  Defaults to ``dx``.
    aperture : ndarray bool, optional
        Boolean mask the same shape as ``opd_2d``.  Used both for
        the heatmap (passed through to :func:`plot_wavefront`) and
        for the radial-RMS binning (rays outside the mask are
        ignored).
    wavelength : float, optional
        Vacuum wavelength [m].  Required when ``units == 'waves'``.
    py, opd_y, px, opd_x : ndarray, optional
        Pre-computed tangential / sagittal fans IN METRES.  When
        all four are supplied they are rendered directly.  When any
        of them is ``None`` the function extracts the central row +
        column of ``opd_2d`` as a fallback.  If you are working
        directly from :func:`lumenairy.raytrace.opd_fan_data` (which
        returns WAVES) and want sub-pixel-accurate chief-ray
        referencing, prefer :func:`plot_opd_fan` with
        ``fan_units='waves'`` over this 2-D fallback.
    units : {'waves', 'nm', 'um', 'm'}, default 'waves'
        DISPLAY units for all panels (passed through to
        :func:`plot_wavefront` and to the fan unit conversion).
        The 2-D OPD is always interpreted in metres internally.
    cmap : str, default ``'RdBu_r'``
        Colormap for the 2-D heatmap panel.
    show_stats : bool, default True
        Overlay PV / RMS annotations on the heatmap and on each fan
        panel.  RMS is the centered RMS about the in-aperture mean
        (matches :func:`plot_wavefront`); PV is max - min over
        finite (in-aperture) pixels.
    title : str, optional
        Figure-level suptitle.
    fig : matplotlib Figure, optional
        Existing figure handle.  When supplied, four new subplots are
        added in a 2x2 grid.  Otherwise a new ``(12, 10)`` figure is
        created.
    radial_rms_n_bins : int or ``'auto'``, default ``'auto'``
        Number of radial bins in the (0, 1) profile panel.
        ``'auto'`` (v4.15.5 default) clamps to
        ``min(32, int(sqrt(N_in_aperture)) // 2)`` so the resolution
        scales with the in-aperture pixel count; a fixed integer
        overrides the auto-clamp.  Coarser grids look smoother;
        finer grids preserve the historic v4.15.4 behaviour.

    Returns
    -------
    fig : matplotlib Figure
    axes : tuple of tuples
        ``((ax_hm, ax_rms), (ax_y, ax_x))`` -- the 2x2 axes grid.

    Notes
    -----
    Central-row / column fallback (when explicit fans are omitted):
    the fallback uses ``(Nx - 1) // 2`` and ``(Ny - 1) // 2`` so the
    extracted index lines up with the same geometric centre that
    :func:`_radial_rms_profile` uses for ``(Nx - 1) / 2.0`` /
    ``(Ny - 1) / 2.0``.  For odd N the result is the unique central
    sample; for even N it picks the lower of the two centre-adjacent
    indices.  For sub-pixel-accurate chief-ray referencing, prefer
    explicit ``(py, opd_y, px, opd_x)`` from
    :func:`lumenairy.raytrace.opd_fan_data`.

    See Also
    --------
    plot_wavefront : 2-D OPD heatmap (used internally for the (0, 0)
        panel).
    plot_opd_fan : 2-panel OPD fan plot (the rendering logic for the
        (1, 0) and (1, 1) panels).
    lumenairy.raytrace.opd_fan_data : compute the fan inputs.

    Examples
    --------
    >>> import numpy as np
    >>> import matplotlib
    >>> matplotlib.use('Agg')
    >>> from lumenairy.analysis import plot_opd_summary
    >>> N = 64
    >>> x = (np.arange(N) - N / 2) / (N / 2)
    >>> X, Y = np.meshgrid(x, x)
    >>> ap = (X ** 2 + Y ** 2) <= 1.0
    >>> opd = 0.5 * (X ** 2 + Y ** 2)  # defocus, waves
    >>> fig, axes = plot_opd_summary(
    ...     opd, dx=1e-6, aperture=ap,
    ...     units='waves', wavelength=633e-9)  # doctest: +SKIP
    """
    _require_mpl()

    if opd_2d.ndim != 2:
        raise ValueError(
            f"plot_opd_summary: opd_2d must be 2-D; got shape "
            f"{opd_2d.shape!r}.")
    if dy is None:
        dy = dx

    scale, units_label = _resolve_opd_units(units, wavelength,
                                            fn_name='plot_opd_summary')

    # Decide whether to use supplied fans or extract from the 2-D OPD.
    fan_sources = (py, opd_y, px, opd_x)
    use_supplied_fans = all(a is not None for a in fan_sources)

    if use_supplied_fans:
        py_arr = np.asarray(py, dtype=float)
        opd_y_arr = np.asarray(opd_y, dtype=float)
        px_arr = np.asarray(px, dtype=float)
        opd_x_arr = np.asarray(opd_x, dtype=float)
    else:
        # v4.15.5 (P2-NEW-V2-5): align the central-row / column index
        # to ``(N - 1) // 2`` so it matches the geometric centre used
        # by ``_radial_rms_profile`` (``(N - 1) / 2.0``) on both odd
        # and even N.  Pre-v4.15.5 used ``N // 2`` which sat one
        # half-pixel right of centre on even N (a 3% offset for
        # N = 32); the new convention picks the lower of the two
        # centre-adjacent samples on even N, sub-pixel error <= 0.5
        # pixel either way.
        Ny, Nx = opd_2d.shape
        col_centre = (Nx - 1) // 2
        row_centre = (Ny - 1) // 2
        py_arr = (np.arange(Ny) - (Ny - 1) / 2.0) / max((Ny - 1) / 2.0, 1.0)
        px_arr = (np.arange(Nx) - (Nx - 1) / 2.0) / max((Nx - 1) / 2.0, 1.0)
        opd_y_arr = np.asarray(opd_2d, dtype=float)[:, col_centre]
        opd_x_arr = np.asarray(opd_2d, dtype=float)[row_centre, :]
        if aperture is not None:
            ap_arr = np.asarray(aperture, dtype=bool)
            opd_y_arr = np.where(ap_arr[:, col_centre], opd_y_arr, np.nan)
            opd_x_arr = np.where(ap_arr[row_centre, :], opd_x_arr, np.nan)

    # Build figure with 2x2 grid.
    if fig is None:
        fig, axarr = plt.subplots(2, 2, figsize=(12, 10))
        ax_hm, ax_rms = axarr[0, 0], axarr[0, 1]
        ax_y, ax_x = axarr[1, 0], axarr[1, 1]
    else:
        ax_hm = fig.add_subplot(2, 2, 1)
        ax_rms = fig.add_subplot(2, 2, 2)
        ax_y = fig.add_subplot(2, 2, 3)
        ax_x = fig.add_subplot(2, 2, 4)

    # Panel (0, 0): 2-D heatmap -- delegate to plot_wavefront.
    plot_wavefront(
        opd_2d, dx=dx, dy=dy,
        aperture=aperture,
        units=units,
        wavelength=wavelength,
        cmap=cmap,
        show_stats=show_stats,
        ax=ax_hm,
        fig=fig,
        title='OPD map',
    )

    # Panel (0, 1): radial RMS profile.
    # v4.15.5 (P3-NEW-F1-5): resolve ``radial_rms_n_bins`` -- auto-
    # clamp on tiny grids to avoid the fragmented-line UX failure.
    if isinstance(radial_rms_n_bins, str) and radial_rms_n_bins == 'auto':
        if aperture is not None:
            n_in_ap = int(np.count_nonzero(np.asarray(aperture, dtype=bool)))
        else:
            n_in_ap = int(np.count_nonzero(np.isfinite(opd_2d)))
        n_bins_eff = _auto_n_bins(n_in_ap, ceiling=32)
    else:
        n_bins_eff = int(radial_rms_n_bins)
        if n_bins_eff < 1:
            raise ValueError(
                f"plot_opd_summary: radial_rms_n_bins must be 'auto' "
                f"or a positive integer; got {radial_rms_n_bins!r}.")
    r_centres, rms_radial = _radial_rms_profile(
        opd_2d, dx, dy, aperture, n_bins=n_bins_eff)
    rms_disp = rms_radial * scale  # convert metres -> display units
    ax_rms.plot(r_centres, rms_disp, color='C0', lw=1.5)
    ax_rms.set_xlabel('Normalised pupil radius')
    ax_rms.set_ylabel(f'RMS OPD ({units_label})')
    ax_rms.set_title('Radial RMS profile')
    ax_rms.grid(alpha=0.3)
    ax_rms.set_xlim(0.0, 1.0)

    # Panels (1, 0) and (1, 1): fan plots.
    opd_y_disp = opd_y_arr * scale
    opd_x_disp = opd_x_arr * scale

    _render_opd_fan_panel(ax_y, py_arr, opd_y_disp, units_label,
                          'y-fan (tangential)', show_stats)
    _render_opd_fan_panel(ax_x, px_arr, opd_x_disp, units_label,
                          'x-fan (sagittal)', show_stats)

    if title:
        fig.suptitle(title)

    try:
        fig.tight_layout()
    except (ValueError, RuntimeError):
        pass

    return fig, ((ax_hm, ax_rms), (ax_y, ax_x))


# =============================================================================
# JONES PUPIL (polarization ray-trace exit-pupil Jones matrix)
# =============================================================================


def compute_jones_pupil(
    apply_fn: Callable[..., Any],
    N: int,
    dx: float,
    wavelength: float,
    dy: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """Extract the full 2x2 Jones pupil from a polarization-capable system.

    Probes the system with two orthogonal input polarizations -- a pure
    x-polarized plane wave and a pure y-polarized plane wave -- and
    records each outgoing JonesField.  Stacked, they form the
    spatially-resolved 2x2 Jones matrix at the exit pupil:

        J[y, x] = [[Ex(x_input),  Ex(y_input)],
                   [Ey(x_input),  Ey(y_input)]]

    where ``x_input`` / ``y_input`` are the two probe polarizations.
    Rows are the output component; columns are the input polarization,
    matching the convention in standard polarization-ray-trace
    references.

    Parameters
    ----------
    apply_fn : callable
        Takes a :class:`polarization.JonesField` and applies the
        full optical system in place (or returns a new one).
        Typically ``lambda jf: jf.apply_real_lens(pres, wavelength)``.
    N : int
        Grid side length.
    dx : float
        Grid pitch [m].
    wavelength : float
        Vacuum wavelength [m].
    dy : float, optional
        Grid pitch in y [m].  Defaults to ``dx``.

    Returns
    -------
    J : ndarray complex, shape (N, N, 2, 2)
        Exit-pupil Jones matrix at every grid point.  Index order:
        ``J[row_out, col_in]``, i.e. ``J[..., 0, 0]`` is output-x from
        input-x, ``J[..., 0, 1]`` is output-x from input-y, etc.

    dx_out, dy_out : float
        Pitch of the returned array -- equal to ``dx`` / ``dy`` for
        a real-lens pipeline that preserves grid pitch.
    """
    from ..elements.polarization import JonesField

    if dy is None:
        dy = dx

    # Probe 1: x-polarized plane wave
    Ex1 = np.ones((N, N), dtype=np.complex128)
    Ey1 = np.zeros((N, N), dtype=np.complex128)
    jf1 = JonesField(Ex=Ex1, Ey=Ey1, dx=dx, dy=dy)
    out1 = apply_fn(jf1)
    if out1 is None:
        out1 = jf1

    # Probe 2: y-polarized plane wave
    Ex2 = np.zeros((N, N), dtype=np.complex128)
    Ey2 = np.ones((N, N), dtype=np.complex128)
    jf2 = JonesField(Ex=Ex2, Ey=Ey2, dx=dx, dy=dy)
    out2 = apply_fn(jf2)
    if out2 is None:
        out2 = jf2

    J = np.stack([
        np.stack([out1.Ex, out2.Ex], axis=-1),  # row 0: Ex(in=x), Ex(in=y)
        np.stack([out1.Ey, out2.Ey], axis=-1),  # row 1: Ey(in=x), Ey(in=y)
    ], axis=-2)  # shape (N, N, 2, 2)

    return J, dx, dy


def plot_jones_pupil(
    J: np.ndarray,
    dx: Optional[float] = None,
    dy: Optional[float] = None,
    unit: str = 'auto',
    show_amplitude: bool = True,
    show_phase: bool = True,
    mask_amplitude_threshold: float = 0.01,
    amp_scale: str = 'linear',
    phase_cmap: str = 'twilight',
    amp_cmap: str = 'inferno',
    figsize: Optional[Tuple[float, float]] = None,
    title: str = 'Jones pupil',
) -> Tuple[Any, Any]:
    """Plot the 2x2 Jones pupil as amplitude + phase spatial maps.

    Produces either a 2x2 grid (amplitude-only or phase-only) or a
    2x4 grid (both).  Each subplot shows the spatial variation of one
    Jones matrix element across the exit pupil, in the standard
    polarization-ray-trace convention:

    +---------------+---------------+
    | J_xx (out=x,  | J_xy (out=x,  |
    |   in=x)       |   in=y)       |
    +---------------+---------------+
    | J_yx (out=y,  | J_yy (out=y,  |
    |   in=x)       |   in=y)       |
    +---------------+---------------+

    Parameters
    ----------
    J : ndarray, shape (Ny, Nx, 2, 2) complex
        Jones pupil array.  Can be produced by :func:`compute_jones_pupil`
        or built manually from two JonesFields.
    dx, dy : float, optional
        Grid pitch [m].  If None, axes are shown in pixels.
    unit : str
        Axis-unit selector passed to :func:`_auto_extent`.
    show_amplitude : bool, default True
        Include the four amplitude subplots ``|J_ij|``.
    show_phase : bool, default True
        Include the four phase subplots ``arg(J_ij)``.
    mask_amplitude_threshold : float, default 0.01
        Fraction of max |J| below which phase is not drawn (those
        pixels are shown as neutral grey).  Prevents phase noise
        from dominating the colour scale outside the illuminated
        region.
    amp_scale : {'linear', 'log'}, default 'linear'
        Amplitude colour scaling.
    phase_cmap : str
        Matplotlib colormap for phase (``'twilight'`` / ``'hsv'`` /
        ``'twilight_shifted'`` recommended for cyclic data).
    amp_cmap : str
        Colormap for amplitude.
    figsize : tuple, optional
        Figure size.  Defaults adapt to ``show_amplitude`` /
        ``show_phase``.
    title : str
        Figure suptitle.

    Returns
    -------
    fig, axes
    """
    _require_mpl()

    if J.ndim != 4 or J.shape[-2:] != (2, 2):
        raise ValueError(
            f"plot_jones_pupil: expected J of shape (Ny, Nx, 2, 2), "
            f"got {J.shape}")

    Ny, Nx = J.shape[:2]

    if dx is not None:
        extent, unit_label, _ = _auto_extent(Nx, dx, unit)
        if dy is not None:
            ext_y, _, _ = _auto_extent(Ny, dy, unit)
        else:
            ext_y = extent
        full_extent = (extent[0], extent[1], ext_y[2], ext_y[3])
    else:
        full_extent = None
        unit_label = 'pixel'

    if not (show_amplitude or show_phase):
        raise ValueError(
            "plot_jones_pupil: at least one of show_amplitude / "
            "show_phase must be True.")

    labels = [['J_xx  (out=x, in=x)', 'J_xy  (out=x, in=y)'],
              ['J_yx  (out=y, in=x)', 'J_yy  (out=y, in=y)']]

    if show_amplitude and show_phase:
        ncols = 4
        fs = figsize or (16, 8)
    else:
        ncols = 2
        fs = figsize or (9, 8)
    fig, axes = plt.subplots(2, ncols, figsize=fs)
    if axes.ndim == 1:
        axes = axes[None, :]

    amp = np.abs(J)
    amp_max = float(amp.max()) if amp.max() > 0 else 1.0
    # Phase mask: only show phase where amplitude exceeds threshold.
    mask = amp > mask_amplitude_threshold * amp_max

    col_offset = 0
    if show_amplitude:
        vmax_amp = amp_max
        vmin_amp = (1e-6 * vmax_amp) if amp_scale == 'log' else 0.0
        for r in range(2):
            for c in range(2):
                ax = axes[r, col_offset + c]
                if amp_scale == 'log':
                    from matplotlib.colors import LogNorm
                    im = ax.imshow(amp[..., r, c] + 1e-30,
                                   extent=full_extent, origin='lower',
                                   cmap=amp_cmap,
                                   norm=LogNorm(vmin=vmin_amp,
                                                vmax=vmax_amp),
                                   aspect='equal')
                else:
                    im = ax.imshow(amp[..., r, c],
                                   extent=full_extent, origin='lower',
                                   cmap=amp_cmap,
                                   vmin=vmin_amp, vmax=vmax_amp,
                                   aspect='equal')
                fig.colorbar(im, ax=ax, shrink=0.75)
                ax.set_title(f'|{labels[r][c]}|')
                if full_extent is not None:
                    ax.set_xlabel(f'x [{unit_label}]')
                    ax.set_ylabel(f'y [{unit_label}]')
        col_offset = 2

    if show_phase:
        phase = np.angle(J)
        for r in range(2):
            for c in range(2):
                ax = axes[r, col_offset + c]
                ph = np.where(mask[..., r, c], phase[..., r, c], np.nan)
                im = ax.imshow(ph, extent=full_extent, origin='lower',
                               cmap=phase_cmap, vmin=-np.pi, vmax=np.pi,
                               aspect='equal')
                cbar = fig.colorbar(im, ax=ax, shrink=0.75,
                                    ticks=[-np.pi, -np.pi/2, 0,
                                           np.pi/2, np.pi])
                cbar.ax.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
                ax.set_title(f'arg({labels[r][c]})')
                if full_extent is not None:
                    ax.set_xlabel(f'x [{unit_label}]')
                    ax.set_ylabel(f'y [{unit_label}]')

    fig.suptitle(title, fontsize=13, fontweight='bold')
    fig.tight_layout()
    return fig, axes


# =========================================================================
# Lens-layout 2-D cross section (H.4 4.7)
# =========================================================================

def plot_lens_layout(
    prescription: Dict[str, Any],
    *,
    ax: Optional[Any] = None,
    wavelength: Optional[float] = None,
    show_rays: bool = True,
    n_field_angles: int = 3,
    max_field_deg: float = 0.0,
    rays_per_fan: int = 5,
    semi_diameter: Optional[float] = None,
    show_optical_axis: bool = True,
    show_image_plane: bool = True,
    title: Optional[str] = None,
) -> Tuple[Any, Any]:
    """Draw a 2-D cross-section of a lens prescription.

    Standalone script/notebook equivalent of the GUI's
    ``Layout2DView`` widget: each refracting surface is drawn as the
    sag curve over the clear aperture, glass regions are shaded, and
    (optionally) chief + marginal rays are traced through the system
    using :func:`lumenairy.raytrace.trace`.

    Parameters
    ----------
    prescription : dict
        Prescription dict (``surfaces``, ``thicknesses``,
        ``aperture_diameter`` ...) as accepted by
        :func:`apply_real_lens` and the raytrace.
    ax : matplotlib Axes, optional
        Where to draw.  A new figure is created if ``None``.
    wavelength : float, optional
        Wavelength [m] used for ray tracing.  Defaults to
        ``prescription.get('wavelength', 1.31e-6)``.
    show_rays : bool, default True
        Trace and overlay rays for ``n_field_angles`` field positions
        from on-axis to ``max_field_deg`` degrees.
    n_field_angles : int, default 3
        Number of field-angle ray fans to draw (on-axis + N-1
        off-axis).
    max_field_deg : float, default 0.0
        Maximum field half-angle [deg] for the off-axis fans.
        Ignored when ``n_field_angles == 1`` or ``show_rays`` is
        False.
    rays_per_fan : int, default 5
        Rays per field-angle fan, including the chief ray.
    semi_diameter : float, optional
        Override for the per-surface clear-aperture half-height [m].
    show_optical_axis : bool, default True
    show_image_plane : bool, default True
    title : str, optional

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes

    Notes
    -----
    Surface tilts / decenters are NOT rendered; folded designs draw
    as if all surfaces lie on the same z axis.  Use the GUI's
    Layout3DView for folded systems.
    """
    _require_mpl()
    from ..elements.lenses import surface_sag_general
    from ..raytrace import (
        find_paraxial_focus,
        make_fan,
        surfaces_from_prescription,
        trace,
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if wavelength is None:
        wavelength = prescription.get('wavelength', 1.31e-6)

    surfaces = surfaces_from_prescription(prescription)
    thicknesses = prescription.get('thicknesses', [])
    if semi_diameter is None:
        ap = prescription.get('aperture_diameter')
        if ap is not None:
            semi_diameter = float(ap) / 2.0
        else:
            sds = [float(s.semi_diameter) for s in surfaces
                   if np.isfinite(s.semi_diameter)]
            semi_diameter = max(sds) if sds else 5e-3

    z_vertex = [0.0]
    for t in thicknesses:
        z_vertex.append(z_vertex[-1] + float(t))

    h = np.linspace(-semi_diameter, semi_diameter, 81)
    surf_curves = []
    for i, s in enumerate(surfaces):
        sag = surface_sag_general(
            h ** 2, s.radius, s.conic, s.aspheric_coeffs)
        z_curve = z_vertex[i] + sag
        surf_curves.append((z_curve, h))
        ax.plot(z_curve, h, color='#1f3a73', lw=1.2)

    # Glass shading between adjacent surfaces.
    for i in range(len(surfaces) - 1):
        glass_in = (surfaces[i].glass_after or '').lower()
        if glass_in in ('', 'air', None):
            continue
        z_i, h_i = surf_curves[i]
        z_ip1, h_ip1 = surf_curves[i + 1]
        z_poly = np.concatenate([z_i, z_ip1[::-1]])
        h_poly = np.concatenate([h_i, h_ip1[::-1]])
        ax.fill(z_poly, h_poly, color='#a8c8e5', alpha=0.4,
                edgecolor='none')

    if show_optical_axis:
        ax.axhline(0.0, color='gray', lw=0.5, ls='--', alpha=0.6)

    image_z = None
    if show_image_plane or show_rays:
        try:
            image_z = float(find_paraxial_focus(surfaces, wavelength))
            if not np.isfinite(image_z):
                image_z = None
        except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                np.linalg.LinAlgError, IndexError, TypeError):
            image_z = None

    if show_image_plane and image_z is not None:
        z_img_world = z_vertex[-1] + image_z
        ax.axvline(z_img_world, color='black', lw=0.8, ls=':',
                   alpha=0.7)
        ax.text(z_img_world, 1.05 * semi_diameter, 'image',
                ha='center', va='bottom', fontsize=8, color='black')

    if show_rays and image_z is not None:
        for fa_idx in range(int(n_field_angles)):
            if n_field_angles > 1:
                fa = (fa_idx / (n_field_angles - 1)) * float(max_field_deg)
            else:
                fa = 0.0
            fa_rad = float(np.radians(fa))
            color = plt.cm.viridis(
                fa_idx / max(1, n_field_angles - 1)) \
                if n_field_angles > 1 else 'tab:orange'
            try:
                fan = make_fan(
                    semi_aperture=semi_diameter, n_rays=int(rays_per_fan),
                    field_angle=fa_rad, wavelength=wavelength)
                r = trace(fan, surfaces, wavelength)
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    IndexError, AttributeError, TypeError):
                # Layout-overlay ray trace failed for this field
                # angle (vignetted at every surface or invalid
                # input); skip the overlay and continue.
                continue
            for k in range(int(rays_per_fan)):
                if not r.image_rays.alive[k]:
                    continue
                z_world = []
                hs = []
                for i_s, rb in enumerate(r.ray_history):
                    sag_at_ray = surface_sag_general(
                        np.array([float(rb.x[k]) ** 2
                                   + float(rb.y[k]) ** 2]),
                        surfaces[i_s].radius, surfaces[i_s].conic,
                        surfaces[i_s].aspheric_coeffs)[0]
                    z_world.append(z_vertex[i_s] + float(sag_at_ray))
                    hs.append(float(rb.x[k]))
                ir = r.image_rays
                if ir.alive[k]:
                    z_img_world = z_vertex[-1] + image_z
                    Lf = float(ir.L[k])
                    Nf = float(ir.N[k]) if hasattr(ir, 'N') else 1.0
                    dz = z_img_world - z_world[-1]
                    h_image = hs[-1] + (Lf / max(abs(Nf), 1e-9)) * dz
                    z_world.append(z_img_world)
                    hs.append(h_image)
                ax.plot(z_world, hs, color=color, lw=0.8, alpha=0.7)

    ax.set_xlabel('z [m]')
    ax.set_ylabel('h [m]')
    if title is None:
        title = prescription.get('name', 'Lens layout')
    ax.set_title(title)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.2)

    return fig, ax


# =========================================================================
# Glass map / Abbe diagram (H.5 4.7)
# =========================================================================

def abbe_diagram(
    glasses: Optional[Sequence[str]] = None,
    *,
    wavelengths_nm: Tuple[float, float, float] = (587.6, 486.1, 656.3),
    ax: Optional[Any] = None,
    annotate: bool = True,
    title: str = 'Abbe diagram',
) -> Tuple[Any, Any, List[Tuple[str, float, float]]]:
    """Plot a refractive-index vs Abbe-number (n_d vs V_d) diagram.

    The classical Abbe diagram is a 2-D scatter of glass-catalogue
    entries with the d-line index ``n_d`` on the y-axis and the
    Abbe number ``V_d = (n_d - 1) / (n_F - n_C)`` on the x-axis (note:
    *higher* dispersion is to the *left*).  Used in lens design to
    pick complementary glass pairs for achromatic doublets.

    Parameters
    ----------
    glasses : list of str, optional
        Glass names from :data:`lumenairy.GLASS_REGISTRY`.  Defaults
        to every bundled glass with a successful index lookup at
        all three wavelengths.
    wavelengths_nm : tuple of 3 floats
        ``(d, F, C)`` lines in nm.  Defaults to the standard
        ``(587.6, 486.1, 656.3)``; the d line is sodium-D, F is
        hydrogen-F, C is hydrogen-C.
    ax : matplotlib Axes, optional
    annotate : bool, default True
        Label each point with the glass name.
    title : str

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    data : list of (name, n_d, V_d) tuples for the entries that
        were successfully evaluated.
    """
    _require_mpl()
    from ..glass import get_glass_index, list_glasses

    if glasses is None:
        glasses = list_glasses()

    lam_d, lam_F, lam_C = (w * 1e-9 for w in wavelengths_nm)

    data = []
    for name in glasses:
        try:
            n_d = float(get_glass_index(name, lam_d))
            n_F = float(get_glass_index(name, lam_F))
            n_C = float(get_glass_index(name, lam_C))
        except (KeyError, ValueError, TypeError):
            # Glass not in registry or wavelength out of Sellmeier
            # range -- silently drop from the Abbe diagram.
            continue
        if (n_F - n_C) == 0:
            continue
        V_d = (n_d - 1.0) / (n_F - n_C)
        if not (np.isfinite(n_d) and np.isfinite(V_d)):
            continue
        data.append((name, n_d, V_d))

    if not data:
        raise ValueError(
            'abbe_diagram: no glasses successfully evaluated; pass a '
            'subset of GLASS_REGISTRY keys via the `glasses=` arg.')

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6))
    else:
        fig = ax.figure

    Vs = [d[2] for d in data]
    ns = [d[1] for d in data]
    ax.scatter(Vs, ns, c='#1f3a73', s=42, alpha=0.75,
               edgecolors='black', linewidths=0.6)

    if annotate:
        for name, n_d, V_d in data:
            ax.annotate(name, xy=(V_d, n_d),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=7, alpha=0.85)

    ax.invert_xaxis()  # Abbe-diagram convention: high dispersion left
    ax.set_xlabel(r'Abbe number $V_d = (n_d - 1) / (n_F - n_C)$')
    ax.set_ylabel(r'$n_d$ (d-line refractive index)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return fig, ax, data


def plot_glass_map(
    glasses: Optional[Sequence[str]] = None,
    *,
    wavelengths_nm: Tuple[float, float, float] = (587.6, 486.1, 656.3),
    figsize: Tuple[float, float] = (11, 6),
    annotate: bool = True,
) -> Tuple[Any, Any]:
    """Two-panel figure: Abbe diagram (left) + ``n(lambda)`` dispersion
    curves (right) for the same glass set.

    A handy starting screen for lens-design exploration.
    """
    _require_mpl()
    from ..glass import get_glass_index, list_glasses

    if glasses is None:
        glasses = list_glasses()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _, _, data = abbe_diagram(
        glasses=glasses, wavelengths_nm=wavelengths_nm,
        ax=axes[0], annotate=annotate,
        title='Abbe diagram (n_d vs V_d)')

    # Right panel: n(lambda) curves over a visible+IR sweep
    wl_grid = np.linspace(0.4e-6, 1.8e-6, 80)
    cmap = plt.cm.viridis
    for i, (name, _, _) in enumerate(data):
        ns = []
        for wl in wl_grid:
            try:
                ns.append(float(get_glass_index(name, wl)))
            except (KeyError, ValueError, TypeError):
                # Glass not in registry or wavelength out of Sellmeier
                # range -- NaN entry will be hidden by the isfinite
                # filter on the plot.
                ns.append(np.nan)
        ns = np.array(ns)
        if np.isfinite(ns).any():
            color = cmap(i / max(1, len(data) - 1))
            axes[1].plot(wl_grid * 1e9, ns, color=color, lw=0.9,
                         label=name)
    axes[1].set_xlabel('Wavelength [nm]')
    axes[1].set_ylabel('Refractive index n')
    axes[1].set_title('Dispersion curves')
    axes[1].grid(True, alpha=0.3)
    if annotate and len(data) <= 20:
        axes[1].legend(fontsize=6, ncols=2, loc='upper right')

    fig.tight_layout()
    return fig, axes
