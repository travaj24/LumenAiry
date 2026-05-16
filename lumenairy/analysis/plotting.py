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

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

import importlib.util as _importlib_util
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
        from matplotlib.patches import (
            Ellipse as _Ellipse, Rectangle as _Rectangle)
        from matplotlib.colors import (
            LogNorm as _LogNorm, Normalize as _Normalize,
            TwoSlopeNorm as _TwoSlopeNorm)
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
        ax.semilogy(coords_disp, I_cut, 'r-', lw=1.2, label='Intensity')
        ax.set_ylim(bottom=max(I.max() * 1e-6, 1e-20))
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
                pad = max((r1 - r0) * 0.15, 20)
                r0 = max(0, int(r0 - pad))
                r1 = min(Ny - 1, int(r1 + pad))
                c0 = max(0, int(c0 - pad))
                c1 = min(Nx - 1, int(c1 + pad))
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
        Peak-normalized intensity PSF (from compute_psf).
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
    from ..elements.polarization import polarization_ellipse, stokes_parameters

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
    from ..raytrace import (
        surfaces_from_prescription, trace, make_fan,
        find_paraxial_focus,
    )
    from ..elements.lenses import surface_sag_general

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
        except Exception:
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
            except Exception:
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
        except Exception:
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
            except Exception:
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
