"""RCWA 2-D crossed gratings (doubly periodic): efficiency + Jones,
normal-vector FFF, analytic-shape factorization."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from ...backend import (
    array_namespace,
    backend_name,
    to_numpy,
)
from ._core import (
    _C,
    Efficiency2D,
    _check_energy,
    _concrete,
    _EnergyError,
    _grazing_safe_wavelength,
    _homogeneous_eigenmodes,
    _interface_smatrix,
    _is_traced,
    _layer_eigenmodes,
    _layer_eigenmodes_tensor,
    _normalize_pol,
    _propagation_smatrix,
    _rcwa_xp,
    _redheffer_star,
    _require_inplane_tensor,
    _require_jax_x64,
    _require_propagating_incidence,
    _sqrt_forward,
    _stabilize_bumps,
    _symmetric_solve_rt,
    _validate_cell_sampling,
    _validate_geometry,
    _validate_shapes,
    _with_blas_limit,
)


def _normalize_2d_formulation(fn_name, formulation):
    """Normalise the 2-D Fourier-factorization selector.  ``'laurent'`` is the
    direct rule everywhere (default, bit-for-bit backward compatible);
    ``'li'`` (alias ``'fff'``) is the dual-Laurent z-rule: the ``E_z``
    elimination uses ``[[1/eps]]`` for fast TM / metal convergence;
    ``'fff_nv'`` is the normal-vector fast Fourier factorization (Schuster 2007)
    -- the in-plane inverse rule projected on the local wall-normal field, the
    correct factorization for 2-D metallic gratings."""
    f = str(formulation).lower()
    if f == "fff":
        f = "li"
    if f == "auto":
        # 1-D parity (audit P2): 1-D accepts 'auto'; map it to the 'li' inverse
        # rule here (the convergence-accelerating z-rule for TM / metals).  Note
        # this is a fixed upgrade, NOT the 1-D adaptive auto-detection -- the 2-D
        # default stays 'laurent' unless 'auto'/'li'/'fff_nv' is requested.
        f = "li"
    if f not in ("laurent", "li", "fff_nv"):
        raise ValueError(
            f"{fn_name}: formulation must be 'laurent', 'li'/'auto' (the inverse "
            f"rule), 'fff' (alias of 'li') or 'fff_nv', got {formulation!r}.")
    return f



# ===========================================================================
# 2-D crossed gratings (doubly periodic)
# ===========================================================================
#
# The 2-D path reuses the dimension-agnostic vectorial machinery above
# (_layer_eigenmodes, _homogeneous_eigenmodes, the gap-free interface /
# propagation / Redheffer assembly, the _sqrt_forward/_sqrt_decay branches,
# the loss-conjugation bridge and the longitudinal-field efficiency
# formula) verbatim; only the harmonic indexing (a 2-D reciprocal lattice)
# and the permittivity convolution (block-Toeplitz-of-block-Toeplitz)
# become two-dimensional.

def _harmonic_orders_2d(n_orders_x, n_orders_y, *, truncation="rectangular",
                        period_x=None, period_y=None):
    """Flat list of integer ``(m, n)`` diffraction-order pairs on the 2-D
    reciprocal lattice (``m`` slow in ``[-Mx..Mx]``, ``n`` fast in
    ``[-My..My]``).  Returns ``(orders, N)`` with ``orders`` an ``(N, 2)``
    int array.

    ``truncation='rectangular'`` (default) keeps the full Cartesian box,
    ``N = (2 Mx + 1)(2 My + 1)`` -- bit-for-bit the historical order set.

    ``truncation='circular'`` (Lalanne 1997) keeps only the orders inside the
    largest reciprocal-space circle inscribed in that box,
    ``(m / period_x)^2 + (n / period_y)^2 <= R^2`` with
    ``R = min(Mx / period_x, My / period_y)``.  This gives isotropic
    resolution and drops the wasted high-|G| corner orders, so a target
    accuracy is reached with fewer harmonics (and, since the eig is
    ``O(N^3)``, less work).  The kept set is period-dependent, so
    ``period_x`` / ``period_y`` are required.  The (0, 0) order is always
    retained.
    """
    Mx, My = int(n_orders_x), int(n_orders_y)
    m = np.repeat(np.arange(-Mx, Mx + 1), 2 * My + 1)
    n = np.tile(np.arange(-My, My + 1), 2 * Mx + 1)
    orders = np.stack([m, n], axis=1)
    if truncation == "rectangular":
        return orders, orders.shape[0]
    if truncation == "circular":
        if period_x is None or period_y is None:
            raise ValueError(
                "_harmonic_orders_2d: truncation='circular' needs period_x "
                "and period_y (the reciprocal-circle radius is "
                "period-dependent).")
        gx = m / float(period_x)
        gy = n / float(period_y)
        r2 = min(Mx / float(period_x), My / float(period_y)) ** 2
        keep = (gx ** 2 + gy ** 2) <= r2 * (1.0 + 1e-9)
        orders = orders[keep]
        return orders, orders.shape[0]
    raise ValueError(
        f"_harmonic_orders_2d: truncation must be 'rectangular' or 'circular', "
        f"got {truncation!r}.")



def _eps_convolution_2d(eps_cell, orders, n_orders_x, n_orders_y):
    """``N x N`` Laurent (direct-rule) permittivity convolution matrix from a
    one-cell sampling ``eps_cell`` (shape ``(Sx, Sy)``).

    Entry ``[p, p'] = c_{(m-m'), (n-n')}`` where ``c`` are the centred 2-D
    Fourier coefficients of ``eps``; built by vectorised fancy-indexing into
    the coefficient table (the block-Toeplitz-Toeplitz structure).
    """
    xp = array_namespace(eps_cell)
    Mx, My = int(n_orders_x), int(n_orders_y)
    eps_cell = xp.asarray(eps_cell).astype(_C)
    Sx, Sy = int(eps_cell.shape[0]), int(eps_cell.shape[1])
    full = xp.fft.fft2(eps_cell) / (Sx * Sy)  # full[k, l] = c_{k,l} (periodic)
    # Coefficient table over the difference range k in [-2Mx..2Mx], l in
    # [-2My..2My].  The index arrays are plain NumPy ints (lattice indices),
    # which index a NumPy / CuPy / JAX `full` identically via broadcasting.
    krange = np.arange(-2 * Mx, 2 * Mx + 1) % Sx
    lrange = np.arange(-2 * My, 2 * My + 1) % Sy
    table = full[krange[:, None], lrange[None, :]]  # (4Mx+1, 4My+1)
    dm = orders[:, 0][:, None] - orders[:, 0][None, :]   # (N, N)
    dn = orders[:, 1][:, None] - orders[:, 1][None, :]
    return table[dm + 2 * Mx, dn + 2 * My]



def _nv_field_2d(eps_cell, period_x, period_y, *, method="smoothed_gradient",
                 sigma_px=1.5, eps_reg_frac=1e-3):
    """Real normal-vector field ``(Nx, Ny)`` over the unit cell for the
    normal-vector FFF (Schuster 2007).  ``Nx, Ny`` have the same ``(Sx, Sy)``
    shape as ``eps_cell`` and point ACROSS every material boundary, unit-norm
    in the near-boundary band and tapered toward 0 in homogeneous regions.

    ``method='smoothed_gradient'`` (default, Goetz 2008): the normalised
    gradient of a Gaussian-smoothed material indicator ``|eps - eps_bg|``.
    Shape-agnostic -- works for numeric cells, disks, ellipses, multi-shape
    cells.  ``sigma_px`` is the smoothing width in pixels.

    ``method='xy_wedge'``: the closed-form axis-aligned field (Schuster
    Fig. 7c) -- the cell is split by the diagonals of a single centred
    rectangle and each wedge carries the wall normal it faces.  Exact for one
    axis-aligned rectangular feature; ``sigma_px`` is ignored.

    ``eps_cell`` MUST be the INTERNAL (loss-bridge-conjugated) sample so the
    indicator lives in the same namespace as the eps operators.  The field is
    built on the host (real NumPy) regardless of the backend; the (Sx, Sy)
    products feed ``_eps_convolution_2d`` which is backend-agnostic.
    """
    eps_np = to_numpy(eps_cell)
    Sx, Sy = int(eps_np.shape[0]), int(eps_np.shape[1])
    if method == "xy_wedge":
        ux = (np.arange(Sx) + 0.5) / Sx
        uy = (np.arange(Sy) + 0.5) / Sy
        xx, yy = np.meshgrid(ux, uy, indexing="ij")
        p = xx - 0.5
        q = yy - 0.5
        Nx = np.zeros((Sx, Sy))
        Ny = np.zeros((Sx, Sy))
        vert = np.abs(p) >= np.abs(q)            # left/right wedge -> normal +/- x
        Nx[vert] = np.sign(p[vert])
        Ny[~vert] = np.sign(q[~vert])
        bad = (Nx == 0) & (Ny == 0)              # exact centre row/col (measure 0)
        Nx[bad] = 1.0
        return Nx, Ny
    if method != "smoothed_gradient":
        raise ValueError(
            f"_nv_field_2d: method must be 'smoothed_gradient' or 'xy_wedge', "
            f"got {method!r}.")
    f = np.abs(eps_np - eps_np.flat[0]).astype(float)
    gx = 2.0 * np.pi * np.fft.fftfreq(Sx)        # per-pixel angular frequency
    gy = 2.0 * np.pi * np.fft.fftfreq(Sy)
    GX, GY = np.meshgrid(gx, gy, indexing="ij")
    phi_hat = np.fft.fft2(f) * np.exp(-0.5 * sigma_px ** 2 * (GX ** 2 + GY ** 2))
    dphidx = np.real(np.fft.ifft2(1j * GX * phi_hat))
    dphidy = np.real(np.fft.ifft2(1j * GY * phi_hat))
    mag = np.hypot(dphidx, dphidy)
    eps_reg = eps_reg_frac * (mag.max() + 1e-300)
    # Taper N -> 0 where the gradient is tiny (homogeneous interior); the
    # softened normalisation keeps |N| <= 1 everywhere.
    Nx = dphidx / np.sqrt(mag ** 2 + eps_reg ** 2)
    Ny = dphidy / np.sqrt(mag ** 2 + eps_reg ** 2)
    # Exact unit-norm renormalisation in the near-boundary band (large |grad|).
    band = mag > 0.1 * (mag.max() + 1e-300)
    nn = np.hypot(Nx, Ny)
    nn_safe = np.where(nn < 1e-12, 1.0, nn)
    Nx[band] = (Nx / nn_safe)[band]
    Ny[band] = (Ny / nn_safe)[band]
    norm_max = float(np.hypot(Nx, Ny).max())
    if not norm_max <= 1.0 + 1e-6:
        raise AssertionError(
            f"_nv_field_2d: normal-vector field exceeds unit norm "
            f"(max |N| = {norm_max:.6f}); the projector would not be a "
            f"valid orthogonal projection.")
    return Nx, Ny



def _nv_convolutions_2d(eps_cell, Nx, Ny, orders, n_orders_x, n_orders_y, xp):
    """Normal-vector FFF in-plane tensor operators (Schuster 2007).

    Returns ``(Cxx, Cxy, Cyx, Cyy, EZZ)`` for the full-tensor layer eigensolver
    ``_layer_eigenmodes_tensor``.  With the flux split (``N_z = 0``)::

        [Dx]   [ E - D.Nxx     -D.Nxy   ] [Sx]
        [Dy] = [ -D.Nxy      E - D.Nyy  ] [Sy]

    where ``E = [[eps]]`` (Laurent / direct rule), ``Einv = [[1/eps]]^{-1}``
    (Li inverse rule), ``D = E - Einv`` (the delta operator that the
    wall-normal projection ``N N^T`` switches to the inverse rule), and
    ``Nab = [[Na Nb]]`` (Laurent convolution of the real field products).

    ``EZZ = Einv = [[1/eps]]^{-1}`` (the DUAL-LAURENT / Li ``E_z`` elimination,
    load-bearing): the tensor eigensolver uses ``inv(EZZ) = [[1/eps]]`` for the
    ``P`` block, which is the SAME ``E_z`` rule the analytic-shape solver
    :func:`rcwa_efficiency_2d_shapes` uses and the rule that matches the
    staircase-free reference on a clean dielectric (the direct-rule
    ``EZZ = [[eps]]`` is biased low by ~6e-2 there).  Pairing the in-plane
    normal-vector projection with the dual-Laurent ``E_z`` is what makes the
    factorization reduce to the rigorous answer for both dielectrics and metals.

    ``eps_cell`` MUST be the INTERNAL (loss-bridge-conjugated) sample; ``Nx, Ny``
    are real host arrays from :func:`_nv_field_2d` on the same grid.
    """
    Mx, My = int(n_orders_x), int(n_orders_y)
    E = _eps_convolution_2d(eps_cell, orders, Mx, My)
    Einv = xp.linalg.inv(_eps_convolution_2d(1.0 / eps_cell, orders, Mx, My))
    Delta = E - Einv
    Nx = xp.asarray(np.asarray(Nx).astype(_C))
    Ny = xp.asarray(np.asarray(Ny).astype(_C))
    Nxx = _eps_convolution_2d(Nx * Nx, orders, Mx, My)
    Nyy = _eps_convolution_2d(Ny * Ny, orders, Mx, My)
    Nxy = _eps_convolution_2d(Nx * Ny, orders, Mx, My)
    Cxx = E - Delta @ Nxx
    Cyy = E - Delta @ Nyy
    Cxy = -(Delta @ Nxy)
    Cyx = Cxy                                    # N N^T is symmetric
    EZZ = Einv                                   # dual-Laurent E_z: inv(EZZ)=[[1/eps]]
    return Cxx, Cxy, Cyx, Cyy, EZZ



@_with_blas_limit
def rcwa_efficiency_2d(
    period_x: float,
    period_y: float,
    eps_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = "te",
    n_orders_x: int = 5,
    n_orders_y: int = 5,
    formulation: str = "laurent",
    truncation: str = "rectangular",
    stabilize: bool = False,
    symmetry: bool = False,
    use_gpu: bool = False,
) -> Efficiency2D:
    """Rigorous diffraction efficiencies of a 2-D (doubly periodic) crossed
    grating: a single patterned layer of permittivity ``eps_cell`` between a
    ``n_superstrate`` half-space and a ``n_substrate`` half-space.

    Backend-dispatched (NumPy / CuPy via ``use_gpu`` / differentiable JAX when
    ``eps_cell`` is a JAX array); see :func:`rcwa_efficiency_1d`.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods along x and y (metres).
    eps_cell : (Sx, Sy) array_like of complex
        Permittivity sampled over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  ``Sx``/``Sy`` must comfortably exceed
        ``4*n_orders_{x,y}`` to avoid Fourier aliasing.
    n_substrate, n_superstrate : complex
        Transmission and incidence half-space indices.
    depth : float
        Patterned-layer thickness (metres).
    wavelength : float
        Vacuum wavelength (metres).
    theta, phi : float, optional
        Polar angle from +z and azimuth from +x (radians) of the incident
        plane wave (conical mounting).  Defaults 0.
    polarization : {'te', 'tm'}, optional
        ``'te'`` (s) / ``'tm'`` (p) relative to the plane of incidence.
    n_orders_x, n_orders_y : int, optional
        Retained orders per side along each axis (default 5 -> 11x11 = 121
        harmonics).
    formulation : {'laurent', 'li', 'fff', 'fff_nv'}, optional
        Fourier factorization of the patterned layer.

        - ``'laurent'`` (default) -- the direct rule everywhere
          (``E_z`` elimination uses ``[[eps]]^{-1}``).  Correct and
          fast-converging for low-contrast dielectrics; kept as the default
          for bit-for-bit backward compatibility.
        - ``'li'`` / ``'fff'`` -- the dual-Laurent (Li 1996 inverse-rule)
          factorization: the ``E_z`` elimination uses ``[[1/eps]]`` (the
          Toeplitz of the Fourier coefficients of ``1/eps``) instead of
          ``[[eps]]^{-1}``.  This is the convergence-accelerating rule for
          **TM / metals / high contrast** -- the same formulation the
          analytic-shape solver :func:`rcwa_efficiency_2d_shapes` uses
          unconditionally, and the rule used by mature FMM codes
          (verified to converge toward grcwa / inkstone).  Recommended for
          metallic 2-D gratings; for a *pixelated* ``eps_cell`` the in-plane
          staircase still limits the rate, so prefer the analytic-shape
          solver when the geometry is describable by disks / rectangles.

          NOTE: the 2-D ``'li'`` ``E_z`` elimination uses the ``[[1/eps]]``
          pixel route, which is itself biased on a pixelated cell (a
          y-uniform pixelated stripe converges to a value offset from the
          rigorous 1-D-Li oracle); ``'fff_nv'`` below pairs the in-plane
          projection with the unbiased direct-rule (Laurent) ``E_z``.
        - ``'fff_nv'`` -- the **normal-vector fast Fourier factorization**
          (Schuster 2007): the in-plane permittivity operator is assembled as
          a full 2x2 tensor ``[[Cxx, Cxy], [Cyx, Cyy]]`` from the local
          wall-normal field ``N(x, y)``, so the inverse rule is applied along
          the (spatially varying) wall normal and the direct rule along the
          tangent -- the rigorous Li-1996 factorization generalised to
          arbitrarily oriented walls.  The ``E_z`` elimination uses the
          dual-Laurent ``[[1/eps]]`` rule (``inv(EZZ) = [[1/eps]]``, the same
          unbiased ``E_z`` rule the analytic-shape solver uses).  This is the
          convergence-accelerating factorization for **SEPARABLE / axis-aligned
          2-D metallic gratings**: there it reaches a target accuracy in fewer
          orders than ``'li'`` and matches the rigorous 1-D-Li oracle (absorptance
          to ~6e-5 on a metal stripe).  ``N`` is built automatically from a
          Gaussian-smoothed gradient of the material indicator.

          VALIDATION SCOPE (corrected, 2026-06-07 audit): the diagonal
          normal/tangential projection (the dominant win) is validated for
          AXIS-ALIGNED / SEPARABLE features -- it matches the analytic-shape
          reference on a clean dielectric to ~1e-3 and genuinely beats 2-D
          ``'li'`` / ``'laurent'`` on a metal stripe.  The off-diagonal cross term
          ``Cxy = -Delta @ [[Nx Ny]]`` (nonzero only on CURVED boundaries --
          disks/ellipses) is **NOT validated**: on a lossy metal disk it
          **mis-splits the absorptance by ~50%** -- a lossless-trap failure
          (``R+T+A`` still closes, but the per-channel split is wrong; converged
          oracles give ``A ~ 0.094-0.096``, ``fff_nv`` gives ``A ~ 0.046-0.077``).
          A ``UserWarning`` fires when the geometry is non-separable; **use**
          ``'li'`` **or** ``'laurent'`` **for curved / non-separable walls** until
          the cross-term factorization is corrected.  ``'fff_nv'`` is NumPy / CuPy
          only and is incompatible with the ``symmetry`` even-parity fast path
          (transparently skipped).

        (``'fff'`` is accepted as an alias of ``'li'``.)
    truncation : {'rectangular', 'circular'}, optional
        Order-set shape.  ``'rectangular'`` (default) keeps the full
        ``(2 Mx + 1)(2 My + 1)`` box; ``'circular'`` (Lalanne 1997) keeps
        only the orders inside the inscribed reciprocal circle, giving
        isotropic resolution and dropping the wasted corner orders -- the
        same target accuracy at fewer harmonics (and less ``O(N^3)`` eig
        work).  The returned ``orders`` length ``N`` shrinks accordingly.
    stabilize : bool, optional
        When ``True`` and the energy guard detects the measure-zero
        large-period / high-contrast instability, search the nearby
        truncations (``n_orders +/- {1, 2, ...}``, nearest first, BOTH
        directions -- the clean truncations bracket the bad ones and on some
        LAPACK builds sit below the request) and return the first
        energy-conserving solve (its order count may then differ from the
        request).  Default ``False`` raises (bit-for-bit backward compatible).
        NumPy / CuPy only.
    symmetry : bool, optional
        When ``True`` AND the cell is centro-symmetric AND incidence is normal
        (``theta == 0``), run the WHOLE single-layer solve in the even-parity
        subspace: the ``(0, 0)`` source is even and no operator couples the
        parities, so the odd half is never excited and the eig, interface, and
        Redheffer steps all shrink from ``2N`` to ``~N`` -- a ~2-4x end-to-end
        speed-up that grows with ``n_orders``.  An off-origin symmetry centre
        is handled by a diagonal recentering gauge.  Gated on the exact
        precondition; if it does not hold (oblique, non-centro-symmetric, or a
        uniform layer) the solver transparently falls back to the full ``2N``
        solve, so the result is always correct.  The even-adapted basis differs
        from the default by a mode-wise rescale/reorder, so efficiencies match
        the ``symmetry=False`` path to ~1e-12 but NOT bit-for-bit.  Default
        ``False``.  NumPy / CuPy only (no effect under JAX tracing).

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (N,) float ndarray
        Reflected / transmitted diffraction efficiency per order.
    """
    if stabilize and not (_is_traced(wavelength) or _is_traced(theta)):
        last = None
        for bump in _stabilize_bumps(min(int(n_orders_x), int(n_orders_y))):
            try:
                return rcwa_efficiency_2d(
                    period_x, period_y, eps_cell, n_substrate, n_superstrate,
                    depth, wavelength, theta=theta, phi=phi,
                    polarization=polarization,
                    n_orders_x=int(n_orders_x) + bump,
                    n_orders_y=int(n_orders_y) + bump, formulation=formulation,
                    truncation=truncation, stabilize=False, symmetry=symmetry,
                    use_gpu=use_gpu)
            except _EnergyError as e:
                last = e
        raise last
    _validate_geometry("rcwa_efficiency_2d",
                       **_concrete(period=period_x, period_y=period_y,
                                   depth=depth, wavelength=wavelength),
                       n_orders=n_orders_x, n_orders_y=n_orders_y)
    _validate_cell_sampling("rcwa_efficiency_2d", eps_cell,
                            n_orders_x, n_orders_y)
    polarization = _normalize_pol("rcwa_efficiency_2d", polarization)
    formulation = _normalize_2d_formulation("rcwa_efficiency_2d", formulation)

    xp = _rcwa_xp("rcwa_efficiency_2d", use_gpu, eps_cell)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _require_jax_x64("rcwa_efficiency_2d")
        if formulation == "fff_nv":
            raise NotImplementedError(
                "rcwa_efficiency_2d: formulation='fff_nv' (normal-vector FFF) "
                "has no JAX/differentiable path -- its normal-vector field is "
                "built on the host (real NumPy gradient).  Use 'laurent' or "
                "'li' for gradient-based design, or call on NumPy/CuPy.")

    # Loss-convention bridge (see rcwa_efficiency_1d): conjugate the PUBLIC
    # permittivity in the active namespace (so a JAX eps_cell stays
    # differentiable); the region scalars stay host complex.
    eps_cell = xp.conj(xp.asarray(eps_cell).astype(_C))
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y,
                                    truncation=truncation,
                                    period_x=period_x, period_y=period_y)
    nre = float(np.real(np.sqrt(eps_sup)))
    # NB: the RCWA ``kx0`` / ``ky0`` are DIMENSIONLESS (k0-normalised, same units
    # as the Kx/Ky order matrices) -- NOT the 1-D PMM convention where ``kx0`` is
    # the dimensional rad/m wavenumber (``* k0``).  See pmm/_core.py for that contrast.
    kx0 = nre * np.sin(theta) * np.cos(phi)        # concrete host floats
    ky0 = nre * np.sin(theta) * np.sin(phi)
    # Run the grazing / non-propagating guards whenever the GEOMETRY is
    # concrete (always on NumPy/CuPy; on JAX when angle/wavelength aren't
    # differentiated), so a region Rayleigh anomaly / non-propagating incidence
    # is caught instead of poisoning the gradient with NaN.  The (traced) cell
    # permittivities are added to the nudge only on the non-JAX path.
    geom_concrete = not (_is_traced(wavelength) or _is_traced(theta)
                         or _is_traced(phi))
    if not is_jax or geom_concrete:
        _require_propagating_incidence("rcwa_efficiency_2d", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)
        eps_reals = [eps_sup, eps_sub]
        if not is_jax:
            eps_reals += [float(xp.real(eps_cell).min()),
                          float(xp.real(eps_cell).max())]
        wl_eff = _grazing_safe_wavelength(
            float(wavelength), kx0, ky0, orders[:, 0], orders[:, 1], period_x,
            period_y, eps_reals)
    else:
        wl_eff = wavelength
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders[:, 0] * (wl_eff / period_x)   # host float arrays
    ky = ky0 + orders[:, 1] * (wl_eff / period_y)
    # Build the (constant) K matrices on the host then move to the backend so
    # they share eps_cell's namespace (mixing backends in one op would raise).
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.asarray(np.diag(ky.astype(_C)))
    kxv = xp.asarray(kx.astype(_C))
    kyv = xp.asarray(ky.astype(_C))

    EPS = _eps_convolution_2d(eps_cell, orders, n_orders_x, n_orders_y)
    # NOTE (audit P2): 2-D 'li' applies the inverse rule to E_z ONLY -- the in-plane
    # wall-normal operator stays Laurent ([[eps]]) here, UNLIKE 1-D 'li'.  Full in-
    # plane normal-vector factorization needs the NV/fff path (formulation='fff_nv'),
    # not a single global inverse rule (there is no global wall normal in 2-D).
    EPS_normal = EPS  # Laurent rule: wall-normal convolution == [[eps]]
    # Dual-Laurent (Li) z-rule: E_z elimination uses [[1/eps]] (Toeplitz of
    # the Fourier coefficients of 1/eps) rather than [[eps]]^{-1}.  This is
    # the convergence-accelerating factorization for TM / metals; gated so
    # the default 'laurent' path stays bit-for-bit unchanged.
    ez_inv = (_eps_convolution_2d(1.0 / eps_cell, orders, n_orders_x,
                                  n_orders_y)
              if formulation == "li" else None)
    # Normal-vector FFF (Schuster 2007): build the local wall-normal field from
    # the SAME internal (loss-bridge-conjugated) cell and assemble the full
    # in-plane tensor operator (the inverse rule projected on the normal, the
    # direct rule on the tangent).  Routed through the tensor eigensolver below.
    if formulation == "fff_nv":
        Nx_nv, Ny_nv = _nv_field_2d(eps_cell, period_x, period_y)
        # Non-separability gate (2026-06-07 audit P1-A).  The NV cross term
        # [[Nx*Ny]] is ~0 for axis-aligned / separable patterns (where fff_nv is
        # validated and beats 2-D li/laurent on metal stripes) but is exercised on
        # CURVED walls -- where the cross-term factorization mis-splits absorption
        # by ~50% (a lossless-trap failure: R+T+A closes but the channel split is
        # wrong).  Warn rather than silently return a wrong absorptance.
        _nv_cross = float(np.max(np.abs(np.asarray(Nx_nv) * np.asarray(Ny_nv))))
        if _nv_cross > 1e-2:
            import warnings
            warnings.warn(
                "rcwa_efficiency_2d(formulation='fff_nv'): the geometry is "
                "NON-SEPARABLE (curved / non-axis-aligned walls; NV cross term "
                f"max|Nx*Ny| = {_nv_cross:.3g}).  fff_nv's cross-term "
                "factorization is NOT validated there -- it can mis-split "
                "absorptance by ~50% (total R+T+A still closes).  Use "
                "formulation='li' or 'laurent' for curved / non-separable "
                "patterns; fff_nv is validated for separable axis-aligned "
                "features.", stacklevel=2)
        Cxx_nv, Cxy_nv, Cyx_nv, Cyy_nv, EZZ_nv = _nv_convolutions_2d(
            eps_cell, Nx_nv, Ny_nv, orders, n_orders_x, n_orders_y, xp)

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)

    # Incident unit plane wave on the (0, 0) order, TE/TM relative to the
    # plane of incidence (built from the in-plane azimuth direction).
    delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2)))
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:                       # normal incidence
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        ax, ay = kx0 / kt, ky0 / kt      # in-plane (rho) unit vector
        if polarization == "te":
            ex0, ey0 = -ay, ax           # s-pol: perpendicular, no z-component
            einc_sq = 1.0
        else:
            ex0, ey0 = ax, ay            # p-pol transverse part along rho
            einc_sq = 1.0 + (kt / kz_inc) ** 2
    cinc = xp.concatenate([ex0 * delta, ey0 * delta])

    # Opt-in even-parity-sector fast path: a centro-symmetric cell at normal
    # incidence excites only even modes, so the whole recursion runs in the
    # (N+1)-d even subspace (see the section header above _symmetric_solve_rt).
    # Returns None -> not applicable -> fall through to the full 2N solve.
    # GATED OFF for 'fff_nv': _symmetric_solve_rt is hard-wired to the scalar
    # (EPS, EPS_normal, ez_inv) core and cannot represent the full in-plane tensor.
    rt = None
    if symmetry and not is_jax and kt < 1e-12 and formulation != "fff_nv":
        rt = _symmetric_solve_rt(Vref, Vtrn, Kx, Ky, EPS, EPS_normal, ez_inv,
                                 orders, k0, depth, cinc, xp)
    if rt is not None:
        r, t = rt
    else:
        if formulation == "fff_nv":
            Wl, Vl, lam = _layer_eigenmodes_tensor(
                Kx, Ky, Cxx_nv, Cxy_nv, Cyx_nv, Cyy_nv, EZZ_nv)
        else:
            Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS_normal,
                                            ez_laurent_inv=ez_inv)
        S = _interface_smatrix(Wref, Vref, Wl, Vl)
        S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
        S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
        S11, S12, S21, S22 = S
        r = S11 @ cinc
        t = S21 @ cinc
    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R_eff = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                        + xp.abs(rz) ** 2) / einc_sq
    T_eff = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                        + xp.abs(tz) ** 2) / einc_sq
    R_eff = xp.where(xp.real(kz_ref) > 0, xp.real(R_eff), 0.0)
    T_eff = xp.where(xp.real(kz_trn) > 0, xp.real(T_eff), 0.0)
    if not is_jax:
        _check_energy("rcwa_efficiency_2d", R_eff, T_eff)
    # cross-suite return shape: unpacks as (orders, R, T); .dof = 2N eigenproblem dim
    return Efficiency2D(orders, R_eff, T_eff, 2 * len(orders))



@_with_blas_limit
def rcwa_jones_2d(
    period_x: float,
    period_y: float,
    eps_tensor_cell,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    n_orders_x: int = 5,
    n_orders_y: int = 5,
    use_gpu: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous 2-D (doubly periodic) anisotropic grating: a single layer
    whose permittivity is a full in-plane TENSOR FIELD (the z-decoupled LC
    subset; Li 2003, direct-rule factorization).  Returns diffraction
    efficiencies for both incident linear polarizations plus the 2x2
    zeroth-order Jones reflection matrix.

    Backend-dispatched (NumPy / CuPy via ``use_gpu`` / differentiable JAX when
    ``eps_tensor_cell`` is a JAX array); see :func:`rcwa_efficiency_1d`.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_tensor_cell : (Sx, Sy, 3, 3) array_like of complex
        Per-pixel permittivity tensor over one unit cell (PUBLIC convention
        ``Im(eps) > 0`` for loss).  Only the in-plane block ``[[xx, xy],
        [yx, yy]]`` and ``zz`` are used.  ``Sx``/``Sy`` must exceed
        ``4*n_orders_{x,y}``.
    n_substrate, n_superstrate, depth, wavelength, theta, phi,
    n_orders_x, n_orders_y
        As in :func:`rcwa_efficiency_2d`.

    Returns
    -------
    orders : (N, 2) int ndarray
        Diffraction-order pairs ``(m, n)``.
    R_eff, T_eff : (2, N) float ndarray
        Diffraction efficiencies per order; row 0 is the response to an
        incident ``E_x`` plane wave, row 1 to incident ``E_y``.
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis
        (columns = response to incident ``E_x`` / ``E_y``).

    Notes
    -----
    Uses the direct (Laurent) tensor factorization, which is exactly
    energy-conserving for a lossless tensor and reduces to
    :func:`rcwa_efficiency_2d` for a scalar cell; it converges fastest for
    smooth / dielectric anisotropic media (e.g. liquid-crystal cells).
    """
    _validate_geometry("rcwa_jones_2d",
                       **_concrete(period=period_x, period_y=period_y,
                                   depth=depth, wavelength=wavelength),
                       n_orders=n_orders_x, n_orders_y=n_orders_y)
    _validate_cell_sampling("rcwa_jones_2d", eps_tensor_cell,
                            n_orders_x, n_orders_y)
    _require_inplane_tensor("rcwa_jones_2d", eps_tensor_cell)

    xp = _rcwa_xp("rcwa_jones_2d", use_gpu, eps_tensor_cell)
    is_jax = backend_name(xp) == "jax"
    if is_jax:
        _require_jax_x64("rcwa_jones_2d")
    eps_t = xp.conj(xp.asarray(eps_tensor_cell).astype(_C))
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y)
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    geom_concrete = not (_is_traced(wavelength) or _is_traced(theta)
                         or _is_traced(phi))
    if not is_jax or geom_concrete:
        _require_propagating_incidence("rcwa_jones_2d", eps_sup,
                                       kx0 ** 2 + ky0 ** 2)
        eps_reals = [eps_sup, eps_sub]
        if not is_jax:
            dr = xp.real(eps_t[:, :, [0, 1, 2], [0, 1, 2]])
            eps_reals += [float(dr.min()), float(dr.max())]
        wl_eff = _grazing_safe_wavelength(
            float(wavelength), kx0, ky0, orders[:, 0], orders[:, 1], period_x,
            period_y, eps_reals)
    else:
        wl_eff = wavelength
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders[:, 0] * (wl_eff / period_x)
    ky = ky0 + orders[:, 1] * (wl_eff / period_y)
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.asarray(np.diag(ky.astype(_C)))
    kxv = xp.asarray(kx.astype(_C))
    kyv = xp.asarray(ky.astype(_C))

    # Direct-rule (Laurent) convolution of each tensor component.
    def _conv(comp):
        return _eps_convolution_2d(comp, orders, n_orders_x, n_orders_y)
    Cxx = _conv(eps_t[:, :, 0, 0])
    Cxy = _conv(eps_t[:, :, 0, 1])
    Cyx = _conv(eps_t[:, :, 1, 0])
    Cyy = _conv(eps_t[:, :, 1, 1])
    EZZ = _conv(eps_t[:, :, 2, 2])

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    Wl, Vl, lam = _layer_eigenmodes_tensor(Kx, Ky, Cxx, Cxy, Cyx, Cyy, EZZ)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    p0 = int(np.where((orders[:, 0] == 0) & (orders[:, 1] == 0))[0][0])
    delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2)))
    safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    R_rows, T_rows, j_cols = [], [], []
    for ex0, ey0 in ((1.0, 0.0), (0.0, 1.0)):
        # Unit tangential E along (ex0, ey0); the incident wave's longitudinal
        # Ez = -(kx0 ex + ky0 ey)/kz_inc inflates |E_inc|^2 (cf. the 1-D sec^2).
        long_inc = (kx0 * ex0 + ky0 * ey0)
        einc_sq = 1.0 + (long_inc / kz_inc) ** 2 if kz_inc != 0 else 1.0
        cinc = xp.concatenate([ex0 * delta, ey0 * delta])
        r = S11 @ cinc
        t = S21 @ cinc
        rx, ry = r[:N], r[N:]
        tx, ty = t[:N], t[N:]
        rz = -(kxv * rx + kyv * ry) / safe_r
        tz = -(kxv * tx + kyv * ty) / safe_t
        Re = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                         + xp.abs(rz) ** 2) / einc_sq
        Te = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                         + xp.abs(tz) ** 2) / einc_sq
        R_rows.append(xp.where(xp.real(kz_ref) > 0, xp.real(Re), 0.0))
        T_rows.append(xp.where(xp.real(kz_trn) > 0, xp.real(Te), 0.0))
        j_cols.append(xp.stack([xp.conj(rx[p0]), xp.conj(ry[p0])]))
    R_eff = xp.stack(R_rows)
    T_eff = xp.stack(T_rows)
    jones_reflection = xp.stack(j_cols, axis=1)
    if not is_jax:
        _check_energy("rcwa_jones_2d", R_eff, T_eff)
    return orders, R_eff, T_eff, jones_reflection



# ===========================================================================
# Analytic shape Fourier transforms + dual-Laurent 2-D factorization
# ===========================================================================
#
# For known shapes (rectangle, disk, ellipse) the permittivity Fourier
# coefficients are computed in CLOSED FORM (exact form factors) instead of
# by FFT-sampling a pixelated cell -- eliminating the aliasing / staircase
# error of pixelation, so the spectrum is exact and convergence is clean.
# Both [[eps]] and [[1/eps]] are built from the SAME analytic form factors,
# and the layer eigenproblem uses the dual-Laurent factorization (the
# in-plane Q block uses [[eps]]; the E_z elimination uses [[1/eps]]
# directly) -- the formulation used by mature analytic-FT FMM codes.

def _shape_form_factor(shape, gxv, gyv, period_x, period_y):
    """Analytic Fourier form factor ``F(G) = (1/A_cell) integral_shape
    exp(-i G.r) d^2r`` for one shape, at the (difference) reciprocal vectors
    ``(gxv, gyv)`` [1/m].  Closed form for ``rectangle`` / ``disk`` /
    ``ellipse``; the ``G = 0`` entry is the area fraction."""
    kind = shape["shape"]
    cx, cy = shape.get("center", (period_x / 2.0, period_y / 2.0))
    area = period_x * period_y
    phase = np.exp(-1j * (gxv * cx + gyv * cy))
    if kind == "rectangle":
        wx, wy = shape["size"]
        # np.sinc(z) = sin(pi z)/(pi z), so sinc(G w / (2 pi)) = sin(G w/2)/(G w/2).
        sx = np.sinc(gxv * wx / (2.0 * np.pi))
        sy = np.sinc(gyv * wy / (2.0 * np.pi))
        return (wx * wy / area) * sx * sy * phase
    if kind in ("disk", "ellipse"):
        from scipy.special import j1
        if kind == "disk":
            ax = ay = shape["radius"]
        else:
            ax, ay = shape["semi_axes"]
        q = np.sqrt((gxv * ax) ** 2 + (gyv * ay) ** 2)
        small = q < 1e-12
        qsafe = np.where(small, 1.0, q)
        bessel = np.where(small, 1.0, 2.0 * j1(qsafe) / qsafe)   # -> 1 as q -> 0
        return (np.pi * ax * ay / area) * bessel * phase
    raise ValueError(
        f"_shape_form_factor: unknown shape {kind!r} (expected 'rectangle', "
        f"'disk' or 'ellipse').")



def _analytic_convolutions_2d(eps_background, shapes, orders, n_orders_x,
                              n_orders_y, period_x, period_y):
    """Analytic ``[[eps]]`` and ``[[1/eps]]`` convolution matrices for a 2-D
    unit cell of background ``eps_background`` overlaid with ``shapes`` (each
    a dict ``{'shape', 'eps', geometry, ['center']}``).  Returns
    ``(EPS, EPS_inv_laurent)``."""
    Mx, My = int(n_orders_x), int(n_orders_y)
    ks = np.arange(-2 * Mx, 2 * Mx + 1)
    ls = np.arange(-2 * My, 2 * My + 1)
    KK, LL = np.meshgrid(ks, ls, indexing="ij")
    gxv = KK * (2.0 * np.pi / period_x)
    gyv = LL * (2.0 * np.pi / period_y)
    eps_bg = _C(eps_background)
    c_eps = np.zeros(KK.shape, dtype=_C)
    c_inv = np.zeros(KK.shape, dtype=_C)
    c_eps[2 * Mx, 2 * My] = eps_bg            # background (DC) term
    c_inv[2 * Mx, 2 * My] = 1.0 / eps_bg
    for sh in shapes:
        eps_s = _C(sh["eps"])
        F = _shape_form_factor(sh, gxv, gyv, period_x, period_y)
        c_eps = c_eps + (eps_s - eps_bg) * F
        c_inv = c_inv + (1.0 / eps_s - 1.0 / eps_bg) * F
    dm = orders[:, 0][:, None] - orders[:, 0][None, :]
    dn = orders[:, 1][:, None] - orders[:, 1][None, :]
    EPS = c_eps[dm + 2 * Mx, dn + 2 * My]
    EPS_inv = c_inv[dm + 2 * Mx, dn + 2 * My]
    return EPS, EPS_inv



@_with_blas_limit
def rcwa_efficiency_2d_shapes(
    period_x: float,
    period_y: float,
    eps_background: complex,
    shapes,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    theta: float = 0.0,
    phi: float = 0.0,
    polarization: str = "te",
    n_orders_x: int = 5,
    n_orders_y: int = 5,
    truncation: str = "rectangular",
    use_gpu: bool = False,
) -> Efficiency2D:
    """Rigorous 2-D crossed-grating efficiencies using **analytic** shape
    Fourier transforms and the dual-Laurent factorization.

    Backend-dispatched for NumPy / CuPy (``use_gpu``).  The analytic shape
    form factors are evaluated on the host (closed form) and the resulting
    convolution matrices are moved to the backend for the eigensolve; the
    JAX (differentiable) path is not offered for the analytic-shape solver
    (its Bessel form factors are host ``scipy.special``).

    The patterned layer is a background permittivity ``eps_background``
    overlaid with analytically-described ``shapes`` (no pixelation), so the
    permittivity spectrum is exact and convergence is clean.

    Parameters
    ----------
    period_x, period_y : float
        Lattice periods (metres).
    eps_background : complex
        Permittivity of the unpatterned background (PUBLIC convention
        ``Im(eps) > 0`` lossy).
    shapes : list of dict
        Each shape is ``{'shape': 'rectangle'|'disk'|'ellipse', 'eps':
        complex, ...geometry..., 'center': (cx, cy) [m]}``; geometry is
        ``'size': (wx, wy)`` for a rectangle, ``'radius': r`` for a disk,
        ``'semi_axes': (ax, ay)`` for an ellipse (all metres).  Shapes are
        painted in order over the background.
    n_substrate, n_superstrate, depth, wavelength, theta, phi, polarization,
    n_orders_x, n_orders_y
        As in :func:`rcwa_efficiency_2d`.

    Returns
    -------
    orders : (N, 2) int ndarray
    R_eff, T_eff : (N,) float ndarray
    """
    _validate_geometry("rcwa_efficiency_2d_shapes", period=period_x,
                       period_y=period_y, depth=depth, wavelength=wavelength,
                       n_orders=n_orders_x, n_orders_y=n_orders_y)
    _validate_shapes("rcwa_efficiency_2d_shapes", shapes, period_x, period_y)
    if abs(_C(eps_background)) < 1e-12:
        raise ValueError(
            f"rcwa_efficiency_2d_shapes: eps_background ~ 0 ({eps_background!r}); "
            f"a zero background permittivity blows up the averaged-eps / "
            f"inverse-rule convolution -- use a small non-zero eps.")
    polarization = _normalize_pol("rcwa_efficiency_2d_shapes", polarization)
    truncation = str(truncation)

    xp = _rcwa_xp("rcwa_efficiency_2d_shapes", use_gpu)
    if backend_name(xp) == "jax":
        raise NotImplementedError(
            "rcwa_efficiency_2d_shapes: the analytic-shape solver has no JAX "
            "(differentiable) path; use a JAX eps_cell with rcwa_efficiency_2d "
            "for gradient-based design, or call with use_gpu for CuPy.")
    # Loss-sign bridge: conjugate every public permittivity (host scalars).
    eps_bg = complex(np.conj(_C(eps_background)))
    shapes_c = [dict(s, eps=complex(np.conj(_C(s["eps"])))) for s in shapes]
    eps_sup = complex(np.conj(_C(n_superstrate) ** 2))
    eps_sub = complex(np.conj(_C(n_substrate) ** 2))

    orders, N = _harmonic_orders_2d(n_orders_x, n_orders_y,
                                    truncation=truncation,
                                    period_x=period_x, period_y=period_y)
    nre = float(np.real(np.sqrt(eps_sup)))
    kx0 = nre * np.sin(theta) * np.cos(phi)
    ky0 = nre * np.sin(theta) * np.sin(phi)
    _require_propagating_incidence("rcwa_efficiency_2d_shapes", eps_sup,
                                   kx0 ** 2 + ky0 ** 2)
    layer_eps = [eps_bg] + [s["eps"] for s in shapes_c]
    wl_eff = _grazing_safe_wavelength(
        wavelength, kx0, ky0, orders[:, 0], orders[:, 1], period_x, period_y,
        [eps_sup, eps_sub] + layer_eps)
    k0 = 2.0 * np.pi / wl_eff
    kx = kx0 + orders[:, 0] * (wl_eff / period_x)
    ky = ky0 + orders[:, 1] * (wl_eff / period_y)
    Kx = xp.asarray(np.diag(kx.astype(_C)))
    Ky = xp.asarray(np.diag(ky.astype(_C)))
    kxv = xp.asarray(kx.astype(_C))
    kyv = xp.asarray(ky.astype(_C))

    # Analytic (host) form factors -> move the convolution matrices to the
    # backend for the eigensolve.
    EPS_np, EPS_inv_np = _analytic_convolutions_2d(
        eps_bg, shapes_c, orders, n_orders_x, n_orders_y, period_x, period_y)
    EPS = xp.asarray(EPS_np)
    EPS_inv = xp.asarray(EPS_inv_np)

    Wref, Vref, kz_ref = _homogeneous_eigenmodes(Kx, Ky, eps_sup)
    Wtrn, Vtrn, kz_trn = _homogeneous_eigenmodes(Kx, Ky, eps_sub)
    # Dual-Laurent: in-plane uses [[eps]], E_z elimination uses [[1/eps]].
    Wl, Vl, lam = _layer_eigenmodes(Kx, Ky, EPS, EPS, ez_laurent_inv=EPS_inv)
    S = _interface_smatrix(Wref, Vref, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wtrn, Vtrn))
    S11, S12, S21, S22 = S

    delta = xp.asarray(((orders[:, 0] == 0) & (orders[:, 1] == 0)).astype(_C))
    kz_inc = float(np.real(_sqrt_forward(eps_sup - kx0 ** 2 - ky0 ** 2)))
    kt = float(np.hypot(kx0, ky0))
    if kt < 1e-12:
        ex0, ey0 = (0.0, 1.0) if polarization == "te" else (1.0, 0.0)
        einc_sq = 1.0
    else:
        ax, ay = kx0 / kt, ky0 / kt
        if polarization == "te":
            ex0, ey0 = -ay, ax
            einc_sq = 1.0
        else:
            ex0, ey0 = ax, ay
            einc_sq = 1.0 + (kt / kz_inc) ** 2
    cinc = xp.concatenate([ex0 * delta, ey0 * delta])
    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:N], r[N:]
    tx, ty = t[:N], t[N:]
    safe_r = xp.where(xp.abs(kz_ref) < 1e-12, 1.0, kz_ref)
    safe_t = xp.where(xp.abs(kz_trn) < 1e-12, 1.0, kz_trn)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R_eff = xp.real(kz_ref / kz_inc) * (xp.abs(rx) ** 2 + xp.abs(ry) ** 2
                                        + xp.abs(rz) ** 2) / einc_sq
    T_eff = xp.real(kz_trn / kz_inc) * (xp.abs(tx) ** 2 + xp.abs(ty) ** 2
                                        + xp.abs(tz) ** 2) / einc_sq
    R_eff = xp.where(xp.real(kz_ref) > 0, xp.real(R_eff), 0.0)
    T_eff = xp.where(xp.real(kz_trn) > 0, xp.real(T_eff), 0.0)
    _check_energy("rcwa_efficiency_2d_shapes", R_eff, T_eff)
    return Efficiency2D(orders, R_eff, T_eff, 2 * len(orders))


__all__ = [
    "_normalize_2d_formulation",
    "_harmonic_orders_2d",
    "_eps_convolution_2d",
    "_nv_field_2d",
    "_nv_convolutions_2d",
    "rcwa_efficiency_2d",
    "rcwa_jones_2d",
    "_shape_form_factor",
    "_analytic_convolutions_2d",
    "rcwa_efficiency_2d_shapes",
]
