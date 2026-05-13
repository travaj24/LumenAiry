"""
lumenairy.analysis.ao -- adaptive-optics building blocks.

Minimum-viable closed-loop AO primitives that compose with the
existing Shack-Hartmann sensor (:func:`lumenairy.shack_hartmann`)
and turbulence-screen generator
(:func:`lumenairy.generate_turbulence_screen`) to enable end-to-end
AO simulation.

The three primitives:

* :class:`DeformableMirror` -- a grid of Gaussian-influence-function
  actuators that maps a command vector ``a`` to a 2-D phase screen
  on the pupil.  Apply via the bound ``apply(E)`` method or via
  :func:`apply_dm` for module-style usage.
* :func:`zernike_modal_basis` and :func:`slope_to_modal` -- least-
  squares slope-to-Zernike-coefficient reconstructor for
  Shack-Hartmann output.  Other modal bases work too (any basis
  whose gradient field can be sampled on the lenslet centres).
* :class:`LeakyIntegrator` -- closed-loop temporal filter
  ``c[t] = (1 - leak) * c[t-1] + gain * e[t]``.  Standard AO
  integrator used in single-conjugate AO systems.

Putting them together (typical AO loop):

.. code-block:: python

    import numpy as np
    import lumenairy as la

    # 1. System: turbulence screen + DM + WFS pre-conjugated to pupil.
    N, dx = 256, 1e-3
    pupil = la.apply_aperture(np.ones((N, N), dtype=complex), dx,
                              shape='circular',
                              params={'diameter': 0.2})
    screen = la.generate_turbulence_screen(N, dx, r0=0.05, seed=1)
    dm = la.DeformableMirror(n_actuators=11, pitch=0.02, dx=dx, N=N)

    # 2. Build the modal reconstructor.
    n_modes = 15
    basis = la.zernike_modal_basis(n_modes, n_lenslets=16, semi_aperture=0.1)
    integrator = la.LeakyIntegrator(gain=0.3, leak=0.01, n_modes=n_modes)

    # 3. Closed loop: each step subtracts the DM phase from the
    #    turbulence, runs the WFS, reconstructs modes, updates
    #    DM commands.
    for t in range(50):
        residual_phase = screen - dm.phase()
        E_pup = pupil * np.exp(1j * residual_phase)
        slopes = la.shack_hartmann(E_pup, dx, lenslet_pitch=0.0125, ...)
        modes = la.slope_to_modal(slopes, basis)
        dm_command = integrator.update(modes)
        dm.set_command(dm_command)

This is a single-conjugate (one DM, one WFS), zonal-DM / modal-
reconstructor architecture suitable for most ground-based AO
training and demonstration use.  For real AO design (multi-DM
laser-guide-star tomography, etc.) reach for HCIPy.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Deformable mirror
# =============================================================================

@dataclass
class DeformableMirror:
    """Gaussian-influence-function deformable mirror on a Cartesian
    actuator grid.

    The DM imparts a 2-D phase screen built from a weighted sum of
    Gaussian influence functions:

    .. math::
        \\phi(x, y)
        = \\sum_{ij} a_{ij}\\, \\exp\\!\\left(-\\frac{(x - x_{ij})^2 + (y - y_{ij})^2}{2 \\sigma_{IF}^2}\\right)

    where :math:`a_{ij}` is the command vector, :math:`(x_{ij}, y_{ij})`
    is the actuator-grid position, and :math:`\\sigma_{IF}` is the
    influence-function width derived from ``inter_actuator_coupling``.

    Attributes
    ----------
    n_actuators : int
        Number of actuators per axis (square grid ``n_actuators**2``
        total).
    pitch : float
        Actuator spacing [m] on the DM.
    dx : float
        Wavefront grid spacing [m].  Should match the pupil grid the
        DM is conjugated to.
    N : int
        Pupil grid size (``N x N``).
    inter_actuator_coupling : float, default 0.15
        Fractional crosstalk between neighbouring actuators
        (0 = perfectly localised; 1 = no localisation).  Sets the
        influence-function width via
        ``sigma_IF = pitch / sqrt(-2 * ln(coupling))``.
    command : ndarray of shape (n_actuators, n_actuators)
        Current command amplitudes [radians of OPD].  Initialised to
        zero.
    """
    n_actuators: int
    pitch: float
    dx: float
    N: int
    inter_actuator_coupling: float = 0.15
    command: np.ndarray = _dc_field(init=False)
    _IF_basis: Optional[np.ndarray] = _dc_field(default=None, init=False, repr=False)

    def __post_init__(self):
        self.command = np.zeros((self.n_actuators, self.n_actuators),
                                 dtype=np.float64)
        c = max(min(self.inter_actuator_coupling, 0.99), 1e-6)
        self._sigma_IF = self.pitch / np.sqrt(-2.0 * np.log(c))
        self._build_IF_basis()

    def _build_IF_basis(self):
        """Pre-compute the (n_act**2, N, N) influence-function stack."""
        N = self.N
        dx = self.dx
        n = self.n_actuators
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        # Actuators centred on the pupil grid.
        act_centres = (np.arange(n) - (n - 1) / 2) * self.pitch
        stack = np.empty((n, n, N, N), dtype=np.float64)
        s2 = self._sigma_IF ** 2
        for i, xi in enumerate(act_centres):
            for j, yj in enumerate(act_centres):
                d2 = (X - xi) ** 2 + (Y - yj) ** 2
                stack[j, i] = np.exp(-d2 / (2.0 * s2))
        self._IF_basis = stack

    def set_command(self, command):
        """Set actuator amplitudes from a (n_act, n_act) array or a
        flat (n_act**2,) vector."""
        c = np.asarray(command, dtype=np.float64)
        if c.size != self.n_actuators ** 2:
            raise ValueError(
                f"DeformableMirror.set_command: expected "
                f"{self.n_actuators ** 2} values, got {c.size}.")
        self.command = c.reshape(self.n_actuators, self.n_actuators)

    def phase(self) -> np.ndarray:
        """Current DM phase map on the wavefront grid [radians of OPD].

        Sum of per-actuator influence functions weighted by the
        command vector.
        """
        # (n_y_act, n_x_act, 1, 1) * (n_y_act, n_x_act, N, N) -> (N, N) via sum
        return np.einsum('ij,ijkl->kl', self.command, self._IF_basis)

    def apply(self, E_in, scale=1.0):
        """Apply the DM phase to a complex field.

        Parameters
        ----------
        E_in : ndarray, complex
            Input field on the pupil grid.
        scale : float, default 1.0
            Multiplicative scale on the DM command applied for this
            evaluation (does not modify ``self.command``).  Useful
            for stepping through a simulated DM excursion without
            re-allocating the IF basis.
        """
        phi = scale * self.phase()
        return E_in * np.exp(1j * phi)

    def reset(self):
        """Zero the command vector."""
        self.command = np.zeros_like(self.command)


def apply_dm(E_in, dm: DeformableMirror, scale: float = 1.0):
    """Module-level convenience: apply a ``DeformableMirror`` phase
    to ``E_in``.  Equivalent to ``dm.apply(E_in, scale)``."""
    return dm.apply(E_in, scale=scale)


# =============================================================================
# Modal reconstruction (slope-to-Zernike)
# =============================================================================

def zernike_modal_basis(n_modes: int, n_lenslets: int,
                        semi_aperture: float,
                        first_mode: int = 1) -> dict:
    """Build the slope-to-modal reconstruction matrix for a Zernike
    basis on a Shack-Hartmann lenslet grid.

    Returns a dict containing the ``(2 * N_lenslets, n_modes)``
    reconstruction matrix plus the lenslet-centre coordinates,
    so :func:`slope_to_modal` can dot-product the SH-WFS slopes
    against it.

    Parameters
    ----------
    n_modes : int
        Number of Zernike modes to reconstruct.  Starts at OSA index
        ``first_mode`` (default 1, i.e. tip; index 0 = piston is
        excluded because it's unobservable from slopes).
    n_lenslets : int
        Number of lenslets per axis on the SH-WFS (square grid).
    semi_aperture : float
        Pupil semi-aperture [m] -- the radius over which the Zernikes
        are normalised.  Should match the entrance-pupil semi-aperture
        used by :func:`lumenairy.shack_hartmann`.
    first_mode : int, default 1
        OSA index of the first Zernike to include.  Default 1 (skip
        piston).

    Returns
    -------
    basis : dict
        ``{'reconstructor': ndarray (n_modes, 2 N_lenslets),
            'mode_indices': ndarray (n_modes,),
            'lenslet_xy': (ndarray, ndarray)  -- lenslet centres [m],
            'semi_aperture': float}``

    Notes
    -----
    The Zernike-mode gradient at each lenslet is computed by finite
    differences (a 1e-4-radius numerical-gradient stencil).  For
    high accuracy at very high modes (n_modes > 50) replace with
    closed-form Zernike-derivative formulas.
    """
    from .analysis import (
        zernike_polynomial, zernike_index_to_nm,
    )
    if n_modes <= 0:
        raise ValueError("n_modes must be > 0")
    if n_lenslets <= 0:
        raise ValueError("n_lenslets must be > 0")
    if semi_aperture <= 0:
        raise ValueError("semi_aperture must be > 0")

    # Lenslet centres on a square grid clipped to the unit disk.
    p = (np.arange(n_lenslets) - (n_lenslets - 1) / 2) / ((n_lenslets - 1) / 2)
    X, Y = np.meshgrid(p, p)
    inside = (X ** 2 + Y ** 2) <= 1.0
    xlens = X[inside] * semi_aperture
    ylens = Y[inside] * semi_aperture
    n_lens = xlens.size

    # Zernike-mode gradient at each lenslet via central finite
    # differences in normalised coords (rho in [0, 1]).
    eps = 1e-4
    n_active = int(n_modes)
    mode_indices = np.arange(first_mode, first_mode + n_active)
    # Influence matrix M: (2 * n_lens, n_modes), columns are
    # [dWdx; dWdy] of each Zernike at the lenslet centres, in waves
    # of OPD per unit pupil-normalised displacement.
    M = np.zeros((2 * n_lens, n_active), dtype=np.float64)
    rho = np.sqrt(X ** 2 + Y ** 2)[inside]
    theta = np.arctan2(Y, X)[inside]
    for k, j in enumerate(mode_indices):
        n_idx, m_idx = zernike_index_to_nm(int(j))
        # Gradient via finite differences in (rho, theta) -> (x, y).
        rho_p = rho.copy()
        rho_m = rho.copy()
        rho_p_plus = np.clip(rho + eps, 0, 1)
        rho_p_minus = np.clip(rho - eps, 0, 1)
        # d/d(rho_x): chain through (rho, theta) <- (x, y).
        # Use forward differences along x and y in normalised coords:
        x_norm = X[inside]
        y_norm = Y[inside]
        rho_x_p = np.sqrt((x_norm + eps) ** 2 + y_norm ** 2)
        rho_x_m = np.sqrt((x_norm - eps) ** 2 + y_norm ** 2)
        the_x_p = np.arctan2(y_norm, x_norm + eps)
        the_x_m = np.arctan2(y_norm, x_norm - eps)
        rho_y_p = np.sqrt(x_norm ** 2 + (y_norm + eps) ** 2)
        rho_y_m = np.sqrt(x_norm ** 2 + (y_norm - eps) ** 2)
        the_y_p = np.arctan2(y_norm + eps, x_norm)
        the_y_m = np.arctan2(y_norm - eps, x_norm)
        Zx_p = zernike_polynomial(n_idx, m_idx, rho_x_p, the_x_p)
        Zx_m = zernike_polynomial(n_idx, m_idx, rho_x_m, the_x_m)
        Zy_p = zernike_polynomial(n_idx, m_idx, rho_y_p, the_y_p)
        Zy_m = zernike_polynomial(n_idx, m_idx, rho_y_m, the_y_m)
        dWdx = (Zx_p - Zx_m) / (2.0 * eps)
        dWdy = (Zy_p - Zy_m) / (2.0 * eps)
        M[:n_lens, k] = dWdx
        M[n_lens:, k] = dWdy

    # Reconstructor = pseudo-inverse of M.
    R = np.linalg.pinv(M)

    return {
        'reconstructor': R,
        'mode_indices': mode_indices,
        'lenslet_xy': (xlens, ylens),
        'semi_aperture': semi_aperture,
        'n_lenslets': n_lenslets,
        'influence_matrix': M,
    }


def slope_to_modal(slopes, basis: dict) -> np.ndarray:
    """Reconstruct modal coefficients from Shack-Hartmann slopes.

    Parameters
    ----------
    slopes : ndarray
        SH-WFS output.  Accepts either:

        * a ``(N_lens, 2)`` array with columns ``[dWdx, dWdy]``, or
        * a flat ``(2 * N_lens,)`` array ``[dWdx_all; dWdy_all]``.
    basis : dict
        Output of :func:`zernike_modal_basis` (or any analogous
        reconstructor with the same dict shape).

    Returns
    -------
    coefs : ndarray, shape ``(n_modes,)``
        Modal coefficients in the same units as the slopes' OPD
        (typically waves).
    """
    R = basis['reconstructor']
    slopes = np.asarray(slopes, dtype=np.float64)
    if slopes.ndim == 2 and slopes.shape[1] == 2:
        # (N_lens, 2) -> (2 N_lens,) by stacking columns
        flat = np.concatenate([slopes[:, 0], slopes[:, 1]])
    elif slopes.ndim == 1:
        flat = slopes
    else:
        raise ValueError(
            f"slopes must be (N, 2) or (2 N,); got shape {slopes.shape}.")
    return R @ flat


# =============================================================================
# Leaky-integrator control law
# =============================================================================

@dataclass
class LeakyIntegrator:
    """First-order leaky integrator for closed-loop AO control.

    The standard single-conjugate AO control law:

    .. math::
        c[t] = (1 - \\text{leak}) \\cdot c[t-1] + \\text{gain} \\cdot e[t]

    where ``c`` is the DM command and ``e`` is the modal-error
    estimate from the wavefront sensor.

    Attributes
    ----------
    gain : float
        Loop gain (0 < gain < 1 for stability with most plants).
        Typical values 0.3 - 0.6.
    leak : float, default 0.0
        Per-iteration "leak" applied to the previous command, in
        ``(0, 1)``.  Useful for un-locked control loops where the
        plant has integrator behaviour.  ``0`` = pure integrator;
        small positive leak prevents long-term command drift.
    n_modes : int
        Number of modal coefficients in the command vector.
    command : ndarray of shape (n_modes,)
        Current command vector (initialised to zero).
    """
    gain: float
    n_modes: int
    leak: float = 0.0
    command: np.ndarray = _dc_field(init=False)

    def __post_init__(self):
        if not 0.0 <= self.gain <= 2.0:
            raise ValueError(f"gain must be in [0, 2]; got {self.gain}")
        if not 0.0 <= self.leak <= 1.0:
            raise ValueError(f"leak must be in [0, 1]; got {self.leak}")
        self.command = np.zeros(self.n_modes, dtype=np.float64)

    def update(self, error: np.ndarray) -> np.ndarray:
        """Step the integrator and return the new command."""
        err = np.asarray(error, dtype=np.float64).ravel()
        if err.size != self.n_modes:
            raise ValueError(
                f"LeakyIntegrator.update: expected {self.n_modes} "
                f"error values, got {err.size}.")
        self.command = (1.0 - self.leak) * self.command + self.gain * err
        return self.command.copy()

    def reset(self):
        """Zero the command vector."""
        self.command = np.zeros_like(self.command)


__all__ = [
    'DeformableMirror',
    'apply_dm',
    'zernike_modal_basis',
    'slope_to_modal',
    'LeakyIntegrator',
]
