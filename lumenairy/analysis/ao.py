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

from dataclasses import dataclass
from dataclasses import field as _dc_field
from typing import Any, Callable, Dict, Optional, Tuple, Union

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

        Always works regardless of ``cache_basis`` setting.  Small
        problems materialise the design matrix and solve via ``lstsq``
        directly; large problems (design matrix over ~128 MB) fall
        back to normal equations accumulated over horizontal bands of
        the pupil grid, so peak scratch memory is the band block
        (bounded at ~32 MB) plus the ``n_act**2 x n_act**2`` normal
        matrix -- the full ``N**2 x n_act**2`` design matrix is never
        materialised.  When the IF basis is already cached the normal
        equations are formed from views of the cache with a single
        BLAS gemm and no extra large allocation.  Caveat: normal
        equations square the condition number, so for pathologically
        degenerate actuator geometries the large-problem path is less
        accurate than the direct ``lstsq`` path (rank deficiency is
        handled via a least-squares solve of the normal system rather
        than a hard failure).
        """
        target = np.asarray(target_phase, dtype=np.float64).ravel()
        n2 = self.n_actuators ** 2
        # Build the design matrix A of shape (N**2, n_act**2).  For
        # small DMs (n_act <= 16 say) materialising A directly is
        # already < 1 GB; for larger DMs we stream row bands and
        # accumulate the n2 x n2 normal matrix instead.
        bytes_design = (self.N ** 2) * n2 * 8
        if bytes_design <= _DEFAULT_CACHE_CEILING_BYTES // 4:
            # Materialise A and solve directly.
            A = np.empty((self.N ** 2, n2), dtype=np.float64)
            for k in range(n2):
                A[:, k] = self._influence_function_kth(k).ravel()
            coeffs, *_ = np.linalg.lstsq(A, target, rcond=None)
        else:
            # Streamed normal equations (v5.17.1, audit P2-01): the
            # previous implementation materialised ALL n2 columns
            # (defeating the memory contract) and formed AtA via
            # n2*(n2+1)/2 Python-level np.dot calls (O(n_act^4) loop).
            # Now AtA = A^T A and Atb = A^T b are accumulated with
            # BLAS gemms without ever holding the full design matrix.
            if self._IF_basis is not None:
                # Cached basis: (n, n, N, N) reshapes to (n2, N**2) as
                # a VIEW (row k = actuator divmod(k, n), matching
                # _influence_function_kth) -- one gemm, no big alloc.
                A_rows = self._IF_basis.reshape(n2, -1)
                AtA = A_rows @ A_rows.T
                Atb = A_rows @ target
            else:
                # No cache: accumulate over horizontal bands of grid
                # rows.  Band block is (rows_band * N, n2); bound its
                # size to ~ceiling/16 (= 32 MB default) of scratch.
                N = self.N
                dx = self.dx
                s2 = self._sigma_IF ** 2
                x = (np.arange(N) - N / 2) * dx
                y = (np.arange(N) - N / 2) * dx
                bytes_per_row = N * n2 * 8
                rows_band = max(
                    1, (_DEFAULT_CACHE_CEILING_BYTES // 16)
                    // bytes_per_row)
                AtA = np.zeros((n2, n2), dtype=np.float64)
                Atb = np.zeros(n2, dtype=np.float64)
                target_2d = target.reshape(N, N)
                for r0 in range(0, N, rows_band):
                    r1 = min(r0 + rows_band, N)
                    Xb, Yb = np.meshgrid(x, y[r0:r1])
                    C = np.empty(((r1 - r0) * N, n2), dtype=np.float64)
                    for k in range(n2):
                        j, i = divmod(k, self.n_actuators)
                        d2 = ((Xb - self._act_centres[i]) ** 2
                              + (Yb - self._act_centres[j]) ** 2)
                        C[:, k] = np.exp(-d2 / (2.0 * s2)).ravel()
                    AtA += C.T @ C
                    Atb += C.T @ target_2d[r0:r1].ravel()
            # lstsq (not solve) so a rank-deficient normal matrix
            # yields the minimum-norm solution instead of blowing up.
            coeffs, *_ = np.linalg.lstsq(AtA, Atb, rcond=None)
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
        # v4.15.5 (P1-NEW-F2-2): defensive guard via the shared
        # ``_check_2d_scalar_field`` helper.  v4.15.4's walker
        # excluded class methods on the documented assumption that
        # "per-class wrapper methods delegate to the guarded scalar
        # functions"; this method is a direct ``E_in * np.exp(1j * phi)``
        # multiply with NO delegation, so without this guard an MCF /
        # 3-D ensemble silently broadcasts (e.g.
        # ``dm.apply(np.random.randn(4, 16, 16).astype(complex))``
        # returns a (4, 16, 16) array via NumPy broadcasting instead
        # of raising).  Paired with the v4.15.5 walker descent into
        # class bodies (P2-NEW-F2-4).
        from lumenairy._validation import _check_2d_scalar_field
        _check_2d_scalar_field(E_in, 'DeformableMirror.apply')
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
    # v4.15.4 (P2-NEW-3WAY-2): defensive guard via the shared
    # ``_check_2d_scalar_field`` helper.  v4.15.3 scoped the walker
    # to ``propagators/`` + ``elements/`` only; this ``analysis/``
    # sibling was missed.  Runs FIRST so PartialCoherenceMCF /
    # 3-D ensemble inputs get a clear v4.16-roadmap message rather
    # than a downstream ``AttributeError`` inside ``dm.apply``.
    from lumenairy._validation import _check_2d_scalar_field
    _check_2d_scalar_field(E_in, 'apply_dm')
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
        zernike_index_to_nm,
        zernike_polynomial,
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
        rho.copy()
        rho.copy()
        np.clip(rho + eps, 0, 1)
        np.clip(rho - eps, 0, 1)
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


# =============================================================================
# High-level closed-loop AO helper
# =============================================================================

# v5.2.3 (ROADMAP v5.2.x ao_closed_loop helper): canonical closed-loop AO
# helper.  At v5.2.0, ``examples/11_ao_closed_loop.py`` had to assemble the
# loop from ``DeformableMirror.fit_phase`` + cumulative-command
# accumulation by hand because no top-level helper existed.  This function
# ships that pattern as a one-line API and is the canonical idiom from
# v5.2.3 onwards.
def ao_closed_loop(
    phase_disturbance: np.ndarray,
    *,
    dm: 'DeformableMirror',
    n_iterations: int = 10,
    gain: float = 0.5,
    leak: float = 0.0,
    wavelength: float = 633e-9,
    dx: float,
    wfs: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    tol: Optional[float] = None,
    return_history: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
    """Run a closed-loop AO control sequence on a disturbance phase.

    Parameters
    ----------
    phase_disturbance : ndarray (Ny, Nx)
        Atmospheric (or other) phase disturbance to correct, in radians.
    dm : DeformableMirror
        The deformable mirror used for correction.  The DM is mutated
        in place -- on return, ``dm.command`` holds the accumulated
        closed-loop command.  Call ``dm.reset()`` first if you want a
        fresh loop.
    n_iterations : int, default 10
        Number of closed-loop iterations.  Treated as an upper bound
        if ``tol`` is set and the early-stop criterion fires first.
    gain : float, default 0.5
        Loop gain in ``[0, 2]``.  Standard leaky-integrator gain.
        Typical AO values are 0.3 -- 0.5; lower values reject
        wavefront-sensor noise better at the cost of convergence
        speed.  ``gain = 0`` is an open-loop fallback for noise-only
        WFS characterisation: NO new correction is computed from the
        WFS measurement at each step.  However, ``gain = 0`` alone
        does NOT freeze the DM command in general -- if ``leak > 0``
        the leak term ``(1 - leak) * cmd_k`` still decays the existing
        command toward zero each iteration.  Pure "command never
        updated" semantics require BOTH ``gain = 0`` AND ``leak = 0``;
        see the Notes section below.
    leak : float, default 0.0
        Leak factor in ``[0, 1]`` applied to the previous command at
        every step (v5.2.5: AUDIT_V5_2_3 V9 + P3-F1-2).  The control
        law is ``cmd_{k+1} = (1 - leak) * cmd_k + gain * fit(...)``;
        ``leak = 0`` (default) is a pure integrator and reproduces the
        v5.2.3 behaviour bit-for-bit.  Small positive leak prevents
        long-term command drift on un-locked loops.
    wavelength : float, default 633e-9
        Operating wavelength in meters.  Only used to convert
        ``history['rms_per_iter']`` to ``history['wfe_per_iter']``
        (waves of OPD).
    dx : float
        Grid spacing in meters.  Must match the grid the DM was built
        on (``dm.dx``).  Accepted as a separate keyword for explicit
        documentation of the spatial scale used by the disturbance
        phase.
    wfs : callable, optional
        Wavefront-sensor function with signature
        ``residual_phase -> measured_phase``.  If ``None`` (default),
        ideal phase sensing is assumed (``measured = residual``).
        Pass a ``functools.partial(...)`` wrapper around a realistic
        sensor (e.g. :func:`lumenairy.shack_hartmann`) for noisy or
        rate-limited measurements.  v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2):
        if ``wfs(residual)`` returns ``None`` the helper treats that
        iteration as a missed measurement and skips the command update
        (the DM command is unchanged on that step).
    tol : float, optional
        Residual-RMS early-stop threshold in radians (v5.2.5:
        AUDIT_V5_2_3 V9 + P3-F1-2).  If set and the residual RMS drops
        below ``tol`` after some iteration, the loop returns early.
        Default ``None`` runs the full ``n_iterations``.
    return_history : bool, default False
        If True, return a per-iteration diagnostic dict alongside the
        final residual.

    Returns
    -------
    residual_phase : ndarray, shape (Ny, Nx)
        The residual phase after the loop terminates (either
        ``n_iterations`` reached or ``tol`` met), in radians.  Equal to
        ``phase_disturbance - dm.phase()`` at the end of the loop.
    history : dict (only if ``return_history=True``)
        Diagnostic dict with keys:

        * ``'rms_per_iter'`` : ndarray of shape
          ``(iterations_run + 1,)``, float, RMS of the residual phase
          in radians at iteration 0 (initial) through the final
          iteration actually executed.  When ``tol`` is not set this
          length is ``n_iterations + 1``.
        * ``'wfe_per_iter'`` : ndarray of the same shape, the same RMS
          converted to meters of OPD via
          ``rms / (2 pi) * wavelength``.

    Notes
    -----
    The control law is the leaky integrator:

    .. math::
        \\mathrm{cmd}_{k+1} = (1 - \\mathrm{leak}) \\cdot \\mathrm{cmd}_k
        + g \\cdot \\mathrm{fit}(\\mathrm{wfs}(\\mathrm{residual}_k))

    where ``fit`` is the DM's least-squares projection
    (:meth:`DeformableMirror.fit_phase`) onto the influence-function
    basis.  Each iteration:

    1. Compute ``residual = phase_disturbance - dm.phase()``.
    2. Pass the residual through the user WFS (or pass through for
       ideal sensing).  If the WFS returns ``None`` this iteration is
       a no-op (no command update).
    3. Fit the measured phase to the DM's influence-function basis
       (on a scratch DM, so the user's ``dm.command`` accumulates).
    4. Update
       ``dm.command = (1 - leak) * dm.command + gain * delta_cmd``.

    The user-supplied ``wfs`` should return a phase map on the same
    grid as ``phase_disturbance`` (an ideal projection back to the
    pupil from whatever the sensor measures).  For real Shack-Hartmann
    integration, build a wrapper that runs the sensor and then
    reconstructs the wavefront via a modal basis.

    Joint ``gain`` / ``leak`` semantics (v5.3, AUDIT_V5_2_5 P3-1
    ao_closed_loop docstring fix): the two parameters control
    orthogonal pieces of the control law and the "command never
    updated" intuition is only correct in their joint zero limit.

    * ``gain = 0`` and ``leak = 0``: pure open-loop / frozen DM.  The
      command is left untouched; the residual at every iteration
      equals the input-state ``phase_disturbance - dm.phase()``.
    * ``gain = 0`` and ``leak > 0``: WFS measurements are ignored but
      the leak term still decays the existing DM command toward zero
      as ``cmd_{k+1} = (1 - leak) * cmd_k``.  After ``K`` iterations
      ``cmd_K = (1 - leak)**K * cmd_0``.  This behaviour is pinned by
      ``test_leak_nonzero_decays_command`` in
      ``tests/unit/test_v5_2_5_ao_closed_loop_residuals.py`` -- e.g.
      ``leak = 0.5`` with three iterations decays the command norm
      by ``(0.5)**3 = 0.125``.
    * ``gain > 0`` and ``leak = 0``: pure integrator (v5.2.3
      behaviour, reproduced bit-for-bit).
    * ``gain > 0`` and ``leak > 0``: full leaky integrator.

    Sign conventions: phases follow the library-wide convention table
    in ``CONVENTIONS.md`` Section 7 (OPD sign -- positive OPD means a
    phase advance / wavefront leads the reference sphere; time
    convention ``exp(-i omega t)`` with forward propagation
    ``exp(+i k z)``).

    Examples
    --------
    >>> import numpy as np
    >>> import lumenairy as la
    >>> N, dx = 128, 30e-3 / 128
    >>> dm = la.DeformableMirror(n_actuators=9, pitch=1e-3, dx=dx, N=N)
    >>> screen = la.generate_turbulence_screen(N, dx, r0=2e-3, seed=0)
    >>> residual, hist = la.ao_closed_loop(
    ...     screen, dm=dm, n_iterations=10, gain=0.5,
    ...     dx=dx, return_history=True)
    >>> hist['rms_per_iter'][-1] < hist['rms_per_iter'][0]
    True
    """
    if n_iterations < 0:
        raise ValueError(
            f"n_iterations must be >= 0; got {n_iterations}")
    # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): loosen gain to [0, 2] (matches
    # sibling LeakyIntegrator).  gain=0 is an open-loop fallback.
    if not (0.0 <= gain <= 2.0):
        raise ValueError(
            f"gain must be in [0, 2]; got {gain}")
    # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): leak factor validation.
    if not (0.0 <= leak <= 1.0):
        raise ValueError(
            f"leak must be in [0, 1]; got {leak}")
    if dx <= 0:
        raise ValueError(f"dx must be > 0; got {dx}")
    if wavelength <= 0:
        raise ValueError(f"wavelength must be > 0; got {wavelength}")
    # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): tol validation (None or
    # non-negative float).
    if tol is not None:
        tol = float(tol)
        if tol < 0.0:
            raise ValueError(f"tol must be >= 0 or None; got {tol}")
    # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): explicit wfs-callable check.
    # ``None`` -> ideal pass-through.  Anything else must be callable.
    if wfs is not None and not callable(wfs):
        raise TypeError(
            "ao_closed_loop: wfs must be None or callable; got "
            f"{type(wfs).__name__}")
    if not isinstance(dm, DeformableMirror):
        raise TypeError(
            "ao_closed_loop: dm must be a DeformableMirror instance, "
            f"got {type(dm).__name__}")

    phase = np.asarray(phase_disturbance, dtype=np.float64)
    if phase.ndim != 2:
        raise ValueError(
            f"phase_disturbance must be 2-D (Ny, Nx); got shape "
            f"{phase.shape}")

    # Track per-iteration residual RMS if requested.  Iteration 0 is
    # the initial disturbance minus the DM's current phase (so callers
    # who pass an already-flat DM see the pure-disturbance RMS at k=0).
    record_history = bool(return_history)
    if record_history:
        rms_per_iter = np.empty(n_iterations + 1, dtype=np.float64)

    # Initial residual (k = 0) before any control action this call.
    residual = phase - dm.phase()
    rms0 = float(np.sqrt(np.mean(residual * residual)))
    if record_history:
        rms_per_iter[0] = rms0

    # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): early-stop check on the
    # initial sample so callers who pass an already-converged state
    # don't waste iterations.
    if tol is not None and rms0 < tol:
        if record_history:
            rms_per_iter = rms_per_iter[:1]
            wfe_per_iter = rms_per_iter / (2.0 * np.pi) * wavelength
            return residual, {
                'rms_per_iter': rms_per_iter,
                'wfe_per_iter': wfe_per_iter,
            }
        return residual

    # Scratch DM mirrors the user DM's geometry so ``fit_phase`` can
    # solve for the incremental command without overwriting the
    # accumulator on ``dm``.  Mirrors the v5.2.0 example idiom.
    scratch = DeformableMirror(
        n_actuators=dm.n_actuators,
        pitch=dm.pitch,
        dx=dm.dx,
        N=dm.N,
        inter_actuator_coupling=dm.inter_actuator_coupling,
        cache_basis=dm.cache_basis,
    )

    iterations_run = 0
    for k in range(n_iterations):
        # 1. Wavefront-sensor measurement (ideal pass-through by default).
        measured = wfs(residual) if wfs is not None else residual
        # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): a WFS that returns None
        # signals "no measurement this frame".  Apply the leak (so the
        # command still relaxes if leak > 0) but skip the integrator
        # update.  When leak == 0 this is a true no-op on the command.
        if measured is None:
            dm.command = (1.0 - leak) * dm.command
        else:
            # 2. Project measured phase onto DM basis (incremental command).
            scratch.fit_phase(measured)
            delta_cmd = scratch.command
            # 3. Leaky-integrator update.  v5.2.5: leak=0 reproduces the
            # v5.2.3 pure-integrator update bit-for-bit.
            dm.command = (1.0 - leak) * dm.command + gain * delta_cmd
        # 4. Recompute residual after the update.
        residual = phase - dm.phase()
        rms_k = float(np.sqrt(np.mean(residual * residual)))
        if record_history:
            rms_per_iter[k + 1] = rms_k
        iterations_run += 1
        # v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2): early-stop on tol.
        if tol is not None and rms_k < tol:
            break

    if record_history:
        # Slice history to the iterations actually run (v5.2.5).
        rms_per_iter = rms_per_iter[:iterations_run + 1]
        # WFE in meters of OPD = (rad rms) / (2 pi) * wavelength.
        wfe_per_iter = rms_per_iter / (2.0 * np.pi) * wavelength
        history = {
            'rms_per_iter': rms_per_iter,
            'wfe_per_iter': wfe_per_iter,
        }
        return residual, history
    return residual


# =============================================================================
# Shack-Hartmann WFS factory (v5.4 -- AO closed-loop dock real-WFS wiring)
# =============================================================================

# v5.4 (AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A wfs adapter): the
# ``ao_closed_loop`` helper accepts an arbitrary callable
# ``wfs(residual_phase) -> measured_phase``, but until v5.4 the only
# canonical option was ``None`` (ideal phase sensing) or a hand-rolled
# closure -- the v5.3.2 GUI ``ao_dock`` ships a "WFS type" combo
# (``shack_hartmann`` / ``pyramid`` / ``curvature``) that had nothing
# to wire because no library-side adapter wrapped ``shack_hartmann``
# (the spot-displacement simulator) and ``slope_to_modal`` (the
# slope-to-Zernike reconstructor) into a single closure.  This
# factory closes that gap so the dock can call a real WFS path.
def make_shack_hartmann_wfs(
    *,
    subaperture_grid: int = 8,
    noise_sigma_pixels: float = 0.0,
    modal_basis: str = 'zernike',
    n_modes: int = 36,
    rng_seed: Optional[int] = None,
    dx_pupil: Optional[float] = None,
    wavelength: Optional[float] = None,
    lenslet_focal: float = 5e-3,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build a closure that simulates a Shack-Hartmann reconstruction
    of an input residual-phase map.

    The returned callable has the signature expected by the ``wfs=``
    kwarg of :func:`ao_closed_loop`:
    ``closure(residual: np.ndarray) -> measured: np.ndarray``.

    Internally the closure:

    1. Forms a unit-amplitude complex field
       ``E = exp(1j * residual)`` on the pupil grid.
    2. Runs :func:`lumenairy.shack_hartmann` with ``n_lenslets =
       subaperture_grid`` to obtain per-subaperture spot displacements
       (centroids) and slopes ``= centroid / lenslet_focal``.
    3. Adds optional Gaussian centroid noise scaled by
       ``noise_sigma_pixels * dx_pupil`` (i.e. interpreted as detector
       pixels equivalent to ``dx_pupil`` -- the library does not
       resolve sub-pixel SH detector geometry on this code path).
    4. Reconstructs modal coefficients via :func:`slope_to_modal`
       against a Zernike basis built for the same subaperture grid.
    5. Reprojects those coefficients onto the original pupil grid via
       :func:`lumenairy.zernike_basis_matrix` so the measured phase
       lives on the same ``(Ny, Nx)`` grid as the input residual.

    Parameters
    ----------
    subaperture_grid : int, default 8
        Number of subapertures per axis on the SH lenslet array
        (square grid).  Maps to ``shack_hartmann(n_lenslets=...)``
        and ``zernike_modal_basis(n_lenslets=...)``.
    noise_sigma_pixels : float, default 0.0
        Gaussian sigma of additive centroid noise in pixel units of
        the input phase grid.  Non-zero values inject fresh noise on
        every call; setting ``rng_seed`` makes the noise SEQUENCE
        reproducible (two closures built with the same seed produce
        the same per-call realisations), it does NOT freeze a single
        realisation across calls.
    modal_basis : {'zernike'}, default 'zernike'
        Modal-basis identifier.  ``'zernike'`` is the only supported
        option at v5.4 -- a placeholder for the v5.4-roadmap
        Karhunen-Loeve basis.
    n_modes : int, default 36
        Number of OSA-indexed Zernike modes to reconstruct (starting
        at OSA index 1 -- tip).  36 covers through 7th-order
        spherical aberration.
    rng_seed : int, optional
        Seed for the noise RNG.  The generator is constructed ONCE in
        the factory (v5.17, audit P3-01), so successive calls always
        draw fresh noise; a fixed seed reproduces the same SEQUENCE of
        realisations across factories (reproducible-run semantics),
        while ``None`` (default) seeds from the OS entropy pool.
    dx_pupil : float, optional
        Pupil-grid spacing [m].  If ``None``, derived from the input
        ``residual`` shape on the assumption that the residual fills a
        ``[-1, 1]`` normalised pupil (``dx = 2 / N``).  Pass an
        explicit value when the residual lives on a known physical
        grid.
    wavelength : float, optional
        Wavelength [m].  If ``None``, defaults to 633 nm (the library
        default).  Only affects ``shack_hartmann()``'s lenslet-focus
        FFT; the reconstructed phase is wavelength-agnostic.
    lenslet_focal : float, default 5e-3
        Lenslet focal length [m].  Sets the centroid-to-slope
        conversion (``slope = centroid / lenslet_focal``).

    Returns
    -------
    wfs : callable
        ``wfs(residual: np.ndarray) -> measured: np.ndarray`` closure
        suitable for passing as the ``wfs=`` kwarg of
        :func:`ao_closed_loop`.

    Notes
    -----
    RNG statefulness (v5.17, audit P3-01): with
    ``noise_sigma_pixels > 0`` the closure advances a factory-scoped
    RNG on every call, so repeated calls on the same residual return
    DIFFERENT reconstructions -- the canonical noisy-WFS semantics --
    even when ``rng_seed`` is set.  Seeding makes the noise sequence
    reproducible across closures built with the same seed, not frozen
    per frame.  (Pre-v5.17 a seeded closure re-drew the identical
    realisation every call, which biased closed-loop noise studies.)
    With ``noise_sigma_pixels == 0`` the closure is fully
    deterministic and side-effect free.

    The reconstructor truncates the input residual to the projection
    of the chosen Zernike basis on the in-disk subapertures, so very
    high-order disturbances will see the typical SH-modal aliasing.
    Increase ``n_modes`` and ``subaperture_grid`` together if you
    need finer reconstruction.
    """
    if subaperture_grid <= 0:
        raise ValueError(
            f"subaperture_grid must be > 0; got {subaperture_grid}")
    if n_modes <= 0:
        raise ValueError(f"n_modes must be > 0; got {n_modes}")
    if noise_sigma_pixels < 0:
        raise ValueError(
            f"noise_sigma_pixels must be >= 0; got {noise_sigma_pixels}")
    if modal_basis != 'zernike':
        # v5.4: zernike is the only canonical basis in the library;
        # 'karhunen_loeve' is a UI placeholder for the v5.4-roadmap
        # KL basis.  Reject anything else explicitly so callers know
        # the option is reserved rather than silently ignored.
        raise ValueError(
            f"modal_basis must be 'zernike'; got {modal_basis!r}.  "
            "Karhunen-Loeve / free-actuator bases are reserved for a "
            "future v5.4+ revision.")
    if lenslet_focal <= 0:
        raise ValueError(
            f"lenslet_focal must be > 0; got {lenslet_focal}")

    wl_default = 633e-9 if wavelength is None else float(wavelength)
    if wl_default <= 0:
        raise ValueError(f"wavelength must be > 0; got {wl_default}")

    # Capture parameters in locals for the closure.
    _subap = int(subaperture_grid)
    _n_modes = int(n_modes)
    _noise = float(noise_sigma_pixels)
    # v5.17 audit (P3-01): construct the noise RNG ONCE in the factory
    # so a fixed ``rng_seed`` yields a reproducible SEQUENCE of
    # per-frame noise realisations.  The previous per-call
    # ``default_rng(_rng_seed)`` re-seeded inside the closure, so every
    # AO frame received the IDENTICAL noise realisation and
    # ``ao_closed_loop``'s leaky integrator accumulated a constant bias
    # instead of averaging zero-mean noise.
    _rng = np.random.default_rng(rng_seed)
    _dx_user = dx_pupil
    _wl = wl_default
    _focal = float(lenslet_focal)

    # Deferred import: avoid a circular import on package init since
    # ``analysis.detector`` re-imports from this module's siblings.
    from .detector import shack_hartmann as _sh_call
    from .zernike import zernike_basis_matrix as _zbasis

    # v5.4 (calibration cache): the library SH simulator does NOT
    # produce slopes whose magnitude matches the linear theory
    # ``dW/dx = (dphi/dx) * lambda / (2*pi)`` for arbitrary phases --
    # it has its own centroid response curve set by the lenslet
    # diffraction PSF, finite sub-aperture size, FFT discretisation,
    # and a small reference/measurement-path bias from differing
    # FFT vs ASM propagation conventions.  Two calibration steps
    # capture this on first call and apply it on every subsequent
    # measurement:
    #
    #   (a) zero-phase reference centroids ``sx_ref / sy_ref`` --
    #       subtracted so a flat input phase maps to zero slope.
    #   (b) unit-tilt scale factor ``slope_scale`` -- ratio of the
    #       SH-measured slope to the linear-theory slope on a unit
    #       phase ramp.  Slopes are divided by this factor before
    #       being passed into ``slope_to_modal`` so the reconstructed
    #       Zernike coefficients have the correct OPD magnitude
    #       (within the linear regime of the SH PSF).
    #
    # Both are cached per ``(N, dx, lenslet_pitch)`` since each only
    # depends on the lenslet grid geometry.  Mirrors the calibration
    # any real SH-WFS hardware would need on a known reference beam.
    _calib_cache: Dict[str, Any] = {}

    def _wfs(residual: np.ndarray) -> np.ndarray:
        r = np.asarray(residual, dtype=np.float64)
        if r.ndim != 2:
            raise ValueError(
                f"residual must be 2-D (Ny, Nx); got shape {r.shape}")
        Ny, Nx = r.shape
        if Ny != Nx:
            raise ValueError(
                f"residual must be square (Ny == Nx); got shape {r.shape}")
        N = Ny
        # Pupil grid spacing: explicit dx_pupil if given, else assume
        # the residual fills a [-1, 1] normalised pupil so dx = 2 / N.
        dx = float(_dx_user) if _dx_user is not None else 2.0 / N
        semi_aperture = N * dx / 2.0
        # Choose a lenslet pitch that tiles the full grid: the SH call
        # uses int(round(pitch / dx)) so pitch must be >= 2 * dx.
        lenslet_pitch = (N * dx) / _subap
        if lenslet_pitch < 2.0 * dx:
            raise ValueError(
                f"subaperture_grid={_subap} too fine for N={N} grid; "
                f"need lenslet_pitch >= 2*dx ({2*dx:.3e}), got "
                f"{lenslet_pitch:.3e}.")

        # 1. Phase -> unit-amplitude complex field.
        E = np.exp(1j * r)

        # 2. SH-WFS: returns (slopes_x, slopes_y, wavefront,
        # centroids_x, centroids_y), each (n_lenslets, n_lenslets)
        # with NaN at out-of-bounds lenslets.
        slopes_x, slopes_y, _wf, cx, cy = _sh_call(
            E, dx, wavelength=_wl,
            lenslet_pitch=lenslet_pitch,
            lenslet_focal=_focal,
            n_lenslets=_subap,
        )

        # v5.4: subtract the zero-phase calibration and apply the
        # unit-phase-ramp linearity rescale.  Cached per grid geometry.
        # ``slope_scale`` is the SH-measured slope divided by the
        # linear-theory PHASE-gradient (rad/m of phi), so dividing
        # the SH slopes by it gives slopes directly in rad/m of phase
        # -- skipping the lambda conversion that would otherwise
        # amplify noise by ~1e7 at visible wavelengths.
        calib_key = f'{N}_{dx:.12e}_{lenslet_pitch:.12e}'
        if calib_key not in _calib_cache:
            # (a) Zero-phase reference centroids.
            E_ref = np.ones((N, N), dtype=complex)
            sx_ref, sy_ref, *_ = _sh_call(
                E_ref, dx, wavelength=_wl,
                lenslet_pitch=lenslet_pitch,
                lenslet_focal=_focal,
                n_lenslets=_subap,
            )
            # (b) Unit-phase-tilt calibration: inject a small phase
            # ramp ``phi = a_cal * x`` (well inside the SH linear
            # regime) and measure the SH-response factor relative to
            # the expected linear PHASE-gradient ``a_cal`` (rad/m).
            a_cal_rad_per_m = 0.05 / (N * dx)
            x_coord = (np.arange(N) - N / 2) * dx
            X_cal, _Y_cal = np.meshgrid(x_coord, x_coord)
            phase_cal = a_cal_rad_per_m * X_cal
            E_cal = np.exp(1j * phase_cal)
            sx_cal, sy_cal, *_ = _sh_call(
                E_cal, dx, wavelength=_wl,
                lenslet_pitch=lenslet_pitch,
                lenslet_focal=_focal,
                n_lenslets=_subap,
            )
            # Mean over finite lenslets gives the SH-measured tilt.
            finite = np.isfinite(sx_cal) & np.isfinite(sx_ref)
            if np.any(finite):
                measured_tilt = float(np.mean(sx_cal[finite] - sx_ref[finite]))
            else:
                measured_tilt = 0.0
            # slope_scale maps SH slope -> phase-gradient (rad/m).
            if abs(a_cal_rad_per_m) < 1e-30 or abs(measured_tilt) < 1e-30:
                slope_scale = 1.0
            else:
                slope_scale = measured_tilt / a_cal_rad_per_m
            _calib_cache[calib_key] = (sx_ref, sy_ref, slope_scale)
        sx_ref, sy_ref, slope_scale = _calib_cache[calib_key]
        # Zero-bias the slopes and rescale so the reconstructor sees
        # slopes in rad/m of phase (not m/m of OPD).  Only subtract on
        # lenslets where both measurement and reference are finite, so
        # NaN sentinels stay NaN (preserved through the in-disk mask).
        good = np.isfinite(slopes_x) & np.isfinite(sx_ref)
        slopes_x = np.where(good, (slopes_x - sx_ref) / slope_scale, slopes_x)
        slopes_y = np.where(good, (slopes_y - sy_ref) / slope_scale, slopes_y)

        # 3. Optional centroid noise: the factory-scoped RNG draws a
        # FRESH realisation on every call (canonical noisy-WFS
        # semantics); a fixed ``rng_seed`` makes the sequence of
        # realisations reproducible across factory instances.
        # ``noise_sigma_pixels`` is interpreted as SH-detector-pixel
        # sigma, with a detector pixel pitch equal to the input grid
        # spacing ``dx`` (the SH simulator samples the focal plane on
        # the same Cartesian grid as the input field).  The
        # corresponding slope-noise sigma is
        # ``sigma_pixels * dx / lenslet_focal`` (m of OPD per m of
        # pupil = radians of tilt).
        if _noise > 0.0:
            sigma_slope = _noise * dx / _focal
            rng = _rng
            slopes_x = slopes_x + rng.standard_normal(slopes_x.shape) * sigma_slope
            slopes_y = slopes_y + rng.standard_normal(slopes_y.shape) * sigma_slope

        # 4. Build the matching slope-to-modal reconstructor and pick
        # out the in-disk lenslets in the same row-major order
        # zernike_modal_basis uses internally.
        basis = zernike_modal_basis(
            n_modes=_n_modes,
            n_lenslets=_subap,
            semi_aperture=semi_aperture,
        )
        p = (np.arange(_subap) - (_subap - 1) / 2) / max((_subap - 1) / 2, 1e-12)
        XL, YL = np.meshgrid(p, p)
        inside = (XL ** 2 + YL ** 2) <= 1.0
        # Replace any NaN sentinels (OOB lenslets) with zero before
        # masking to the disk: NaN-poisoning the basis solve produces
        # all-NaN coefficients and a NaN reconstructed phase.
        sx_in = np.where(np.isfinite(slopes_x[inside]), slopes_x[inside], 0.0)
        sy_in = np.where(np.isfinite(slopes_y[inside]), slopes_y[inside], 0.0)
        slopes_flat = np.concatenate([sx_in, sy_in])
        coeffs = slope_to_modal(slopes_flat, basis)

        # 5. Reproject onto the full (N, N) pupil grid.
        # After the unit-phase-tilt rescale, ``coeffs`` are in RADIANS
        # of phase (not meters of OPD): the basis influence matrix is
        # in 1/m and the rescaled slopes are in rad/m, so the dot
        # product gives rad directly.
        #
        # ``zernike_modal_basis(first_mode=1)`` uses OSA indices
        # [1..n_modes] (skip piston).  ``zernike_basis_matrix(n_modes)``
        # uses OSA indices [0..n_modes-1] (include piston).  Build with
        # n_modes+1 and slice off the piston column so the two stay
        # aligned mode-for-mode.
        x = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(x, x)
        Zmat_full, pupil_mask = _zbasis(_n_modes + 1, X, Y, semi_aperture)
        Zmat = Zmat_full[:, 1:]  # drop piston to match first_mode=1
        measured = np.zeros((N, N), dtype=np.float64)
        # zernike_basis_matrix returns rows for in-pupil pixels only.
        measured[pupil_mask] = Zmat @ coeffs
        return measured

    return _wfs


__all__ = [
    'DeformableMirror',
    'apply_dm',
    'zernike_modal_basis',
    'slope_to_modal',
    'LeakyIntegrator',
    # v5.2.3 (ROADMAP v5.2.x ao_closed_loop helper):
    'ao_closed_loop',
    # v5.4 (AUDIT_V5_3_2_GUI_VS_LIBRARY_2026_05_24 P1-A wfs adapter):
    'make_shack_hartmann_wfs',
]
