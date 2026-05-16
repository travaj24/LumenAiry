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
        # 4.10: shack_hartmann returns a 5-tuple (slopes_x, slopes_y,
        # wavefront, centroids_x, centroids_y).  slope_to_modal expects
        # an (N_lens, 2) array of slopes, so stack the first two.
        slopes_x, slopes_y, *_ = la.shack_hartmann(
            E_pup, dx, wavelength=550e-9,
            lenslet_pitch=0.0125, lenslet_focal=0.04)
        slopes = np.column_stack([slopes_x.ravel(), slopes_y.ravel()])
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
from typing import Any, Dict, Optional, Tuple

import numpy as np


# =============================================================================
# Deformable mirror
# =============================================================================

# Default ceiling for eager basis caching.  At 512 MB float64 (= 6.7e7
# values), a 32x32 DM on a 512x512 pupil grid (= 268 MB) caches eagerly;
# a 32x32 DM on a 1024x1024 grid (= 1 GB) goes lazy automatically.
_DEFAULT_CACHE_CEILING_BYTES = 512 * 1024 * 1024


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
    cache_basis : {'auto', True, False}, default 'auto'
        Controls the influence-function-stack caching strategy.

        - ``'auto'`` (default): pre-compute the ``(n_act, n_act, N, N)``
          basis if it fits under ``_DEFAULT_CACHE_CEILING_BYTES``
          (~512 MB); otherwise compute each actuator's influence
          on demand inside :meth:`phase` (no large allocation).
        - ``True``: always pre-compute.  Fast ``phase()`` but watch
          memory at large ``n_actuators * N`` (32 actuators x 1024 grid
          eagerly caches 8 GB float64).
        - ``False``: always compute on demand.  Slightly slower per
          :meth:`phase` call but uses ~``N**2`` floats of scratch
          memory regardless of ``n_actuators``.

    command : ndarray of shape (n_actuators, n_actuators)
        Current command amplitudes [radians of OPD].  Initialised to
        zero.
    """
    n_actuators: int
    pitch: float
    dx: float
    N: int
    inter_actuator_coupling: float = 0.15
    cache_basis: Any = 'auto'
    command: np.ndarray = _dc_field(init=False)
    _IF_basis: Optional[np.ndarray] = _dc_field(default=None, init=False, repr=False)
    _act_centres: Optional[np.ndarray] = _dc_field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.command = np.zeros((self.n_actuators, self.n_actuators),
                                 dtype=np.float64)
        c = max(min(self.inter_actuator_coupling, 0.99), 1e-6)
        self._sigma_IF = self.pitch / np.sqrt(-2.0 * np.log(c))
        # Actuator centres on the pupil grid (cheap, always kept).
        self._act_centres = (
            (np.arange(self.n_actuators) - (self.n_actuators - 1) / 2)
            * self.pitch
        )
        # Decide cache strategy.
        want_cache = self.cache_basis
        if want_cache == 'auto':
            bytes_needed = (self.n_actuators ** 2) * (self.N ** 2) * 8
            want_cache = bytes_needed <= _DEFAULT_CACHE_CEILING_BYTES
        self._cache_active = bool(want_cache)
        if self._cache_active:
            self._build_IF_basis()

    def _build_IF_basis(self) -> None:
        """Pre-compute the (n_act, n_act, N, N) influence-function stack.

        Only called when ``cache_basis`` resolves to True at init time.
        Memory: ``n_actuators**2 * N**2 * 8`` bytes (float64).
        """
        N = self.N
        dx = self.dx
        n = self.n_actuators
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        stack = np.empty((n, n, N, N), dtype=np.float64)
        s2 = self._sigma_IF ** 2
        for i, xi in enumerate(self._act_centres):
            for j, yj in enumerate(self._act_centres):
                d2 = (X - xi) ** 2 + (Y - yj) ** 2
                stack[j, i] = np.exp(-d2 / (2.0 * s2))
        self._IF_basis = stack

    def set_command(self, command: np.ndarray) -> None:
        """Set actuator amplitudes from a (n_act, n_act) array or a
        flat (n_act**2,) vector."""
        c = np.asarray(command, dtype=np.float64)
        if c.size != self.n_actuators ** 2:
            raise ValueError(
                f"DeformableMirror.set_command: expected "
                f"{self.n_actuators ** 2} values, got {c.size}.")
        self.command = c.reshape(self.n_actuators, self.n_actuators)

    def fit_phase(self, target_phase: np.ndarray) -> np.ndarray:
        """Project a target ``(N, N)`` phase map onto the actuator grid
        via least squares against the influence-function basis.

        Returns the fitted ``(n_actuators, n_actuators)`` command vector
        and also stores it via :meth:`set_command`.  Use this instead
        of reaching into the private ``_IF_basis`` attribute when
        closing a modal-to-zonal projection in your control loop.

        Always works regardless of ``cache_basis`` setting: streams
        per-actuator IF rows in to a normal-equations solver, so peak
        memory is ``n_actuators**2 * N**2 / chunk_size`` at most.
        """
        target = np.asarray(target_phase, dtype=np.float64).ravel()
        n2 = self.n_actuators ** 2
        # Build the design matrix A of shape (N**2, n_act**2).  For
        # small DMs (n_act <= 16 say) materialising A directly is
        # already < 1 GB; for larger DMs we stream rows and accumulate
        # the n2 x n2 normal matrix instead.
        bytes_design = (self.N ** 2) * n2 * 8
        if bytes_design <= _DEFAULT_CACHE_CEILING_BYTES // 4:
            # Materialise A and solve directly.
            A = np.empty((self.N ** 2, n2), dtype=np.float64)
            for k in range(n2):
                A[:, k] = self._influence_function_kth(k).ravel()
            coeffs, *_ = np.linalg.lstsq(A, target, rcond=None)
        else:
            # Streamed normal equations: AtA = sum_k a_k a_k^T,
            # Atb = sum_k a_k * <target, a_k> -- both n2 x n2 / n2.
            AtA = np.zeros((n2, n2), dtype=np.float64)
            Atb = np.zeros(n2, dtype=np.float64)
            cols = [self._influence_function_kth(k).ravel()
                    for k in range(n2)]
            for i in range(n2):
                ci = cols[i]
                Atb[i] = float(ci @ target)
                for j in range(i, n2):
                    AtA[i, j] = float(ci @ cols[j])
                    AtA[j, i] = AtA[i, j]
            coeffs = np.linalg.solve(AtA, Atb)
        self.set_command(coeffs)
        return self.command.copy()

    def _influence_function_kth(self, k: int) -> np.ndarray:
        """Return the k-th actuator's influence function as an (N, N)
        array.  ``k`` indexes the flat ``(n_act, n_act)`` grid in
        row-major (numpy) order."""
        n = self.n_actuators
        if self._IF_basis is not None:
            j, i = divmod(k, n)
            return self._IF_basis[j, i]
        # Compute on demand.
        j, i = divmod(k, n)
        xi = self._act_centres[i]
        yj = self._act_centres[j]
        dx = self.dx
        N = self.N
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        d2 = (X - xi) ** 2 + (Y - yj) ** 2
        return np.exp(-d2 / (2.0 * self._sigma_IF ** 2))

    def phase(self) -> np.ndarray:
        """Current DM phase map on the wavefront grid [radians of OPD].

        Sum of per-actuator influence functions weighted by the
        command vector.

        When ``cache_basis`` is active, this is one ``einsum`` call
        on the pre-computed basis.  Otherwise the contribution from
        each actuator is added on the fly into a single ``(N, N)``
        accumulator -- no large stack allocated.
        """
        if self._cache_active and self._IF_basis is not None:
            # (n_y_act, n_x_act) x (n_y_act, n_x_act, N, N) -> (N, N)
            return np.einsum('ij,ijkl->kl', self.command, self._IF_basis)
        # Lazy / on-demand path: stream per-actuator gaussians into the
        # accumulator.  Skips actuators with command 0.0 for free.
        out = np.zeros((self.N, self.N), dtype=np.float64)
        dx = self.dx
        N = self.N
        x = (np.arange(N) - N / 2) * dx
        y = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, y)
        s2 = self._sigma_IF ** 2
        cmd = self.command
        for j, yj in enumerate(self._act_centres):
            for i, xi in enumerate(self._act_centres):
                a = cmd[j, i]
                if a == 0.0:
                    continue
                d2 = (X - xi) ** 2 + (Y - yj) ** 2
                out += a * np.exp(-d2 / (2.0 * s2))
        return out

    def apply(self, E_in: np.ndarray, scale: float = 1.0) -> np.ndarray:
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

    def reset(self) -> None:
        """Zero the command vector."""
        self.command = np.zeros_like(self.command)


def apply_dm(
    E_in: np.ndarray,
    dm: DeformableMirror,
    scale: float = 1.0,
) -> np.ndarray:
    """Module-level convenience: apply a ``DeformableMirror`` phase
    to ``E_in``.  Equivalent to ``dm.apply(E_in, scale)``."""
    return dm.apply(E_in, scale=scale)


# =============================================================================
# Modal reconstruction (slope-to-Zernike)
# =============================================================================

def zernike_modal_basis(
    n_modes: int,
    n_lenslets: int,
    semi_aperture: float,
    first_mode: int = 1,
) -> Dict[str, Any]:
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
    from .core import (
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
        # 4.10: Use one-sided FD at rim lenslets (rho_x_p > 1 or rho_y_p > 1)
        # to avoid evaluating Zernike polynomials in the forbidden rho > 1
        # region, where zernike_polynomial returns 0 and the centred FD
        # picks up a spurious 0-vs-finite spike contaminating the
        # influence matrix.  Switch to backward differences when the
        # forward probe would escape the disk.
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
        # One-sided fallback at the rim.
        # 4.11.2: pre-4.11.2 only checked the +x and +y rims
        # (``rho_x_p > 1`` / ``rho_y_p > 1``) -- the SAME spurious
        # 0-vs-finite FD spike appeared on the -x and -y rims because
        # the backward probe (rho_x_m / rho_y_m) was unchecked.  Now
        # we detect all four quadrants and switch to the appropriate
        # one-sided FD:
        #   rim_x_pos (rho_x_p > 1) -> use (Z_c - Zx_m) / eps (backward)
        #   rim_x_neg (rho_x_m > 1) -> use (Zx_p - Z_c) / eps (forward)
        # and analogously for y.  For interior lenslets both probes
        # stay inside the disk and we use the standard centred FD.
        Z_centre_x = zernike_polynomial(n_idx, m_idx, rho, theta)
        rim_x_pos = rho_x_p > 1.0
        rim_x_neg = rho_x_m > 1.0
        rim_y_pos = rho_y_p > 1.0
        rim_y_neg = rho_y_m > 1.0
        dWdx = np.where(
            rim_x_pos,
            (Z_centre_x - Zx_m) / eps,
            np.where(
                rim_x_neg,
                (Zx_p - Z_centre_x) / eps,
                (Zx_p - Zx_m) / (2.0 * eps)))
        dWdy = np.where(
            rim_y_pos,
            (Z_centre_x - Zy_m) / eps,
            np.where(
                rim_y_neg,
                (Zy_p - Z_centre_x) / eps,
                (Zy_p - Zy_m) / (2.0 * eps)))
        # The Zernike gradient above is in normalised-pupil coords;
        # SH-WFS slopes are in physical units (dW/dx, dW/dy [m/m]).
        # Convert by 1/semi_aperture.
        if semi_aperture > 0:
            dWdx = dWdx / semi_aperture
            dWdy = dWdy / semi_aperture
        M[:n_lens, k] = dWdx
        M[n_lens:, k] = dWdy

    # 4.10: regularize pinv to avoid noise amplification when M is
    # ill-conditioned (low n_modes vs many lenslets, sparse-illumination
    # patterns, etc.).  rcond=1e-3 is a conservative truncation; users
    # who need different conditioning can post-process the reconstructor
    # via SVD-based modes.
    R = np.linalg.pinv(M, rcond=1e-3)

    return {
        'reconstructor': R,
        'mode_indices': mode_indices,
        'lenslet_xy': (xlens, ylens),
        'semi_aperture': semi_aperture,
        'n_lenslets': n_lenslets,
        'influence_matrix': M,
    }


def slope_to_modal(
    slopes: np.ndarray,
    basis: Dict[str, Any],
) -> np.ndarray:
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

    def __post_init__(self) -> None:
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

    def reset(self) -> None:
        """Zero the command vector."""
        self.command = np.zeros_like(self.command)


__all__ = [
    'DeformableMirror',
    'apply_dm',
    'zernike_modal_basis',
    'slope_to_modal',
    'LeakyIntegrator',
]
