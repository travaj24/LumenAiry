"""PMM 1-D gratings: the public efficiency + Jones entry points (vertical
and slanted, binary and multi-region), the unified ``pmm_1d`` dispatcher, the
differentiable JAX-1D twin, and the right-angle convergence-class predictors."""
from __future__ import annotations

import cmath
from typing import Tuple

import numpy as np

# Backend detection for the JAX (differentiable) dispatch in pmm_efficiency_1d.
# Mirrors rcwa's pattern: a JAX input routes to the self-contained jnp twin,
# while a NumPy input falls through to the original (byte-identical) code.
from ...backend import is_jax_array

# Reused for the slanted-grating solver: the slant breaks the +/-q field
# symmetry (like a full-3x3 tensor layer), so it needs the GENERALIZED
# (explicit forward/backward) S-matrix.  rcwa does NOT import pmm, so this
# top-level import introduces no cycle.
from ._core import (
    _C,
    _COV_MIN_SLANT_RAD,
    _pmm_efficiency_1d_jax,
    _pmm_jones_1d_jax,
    _pmm_jones_oblique_segments_solve,
    _pmm_jones_oblique_solve,
    _pmm_jones_slant_diag_solve,
    _pmm_jones_slant_segments_solve,
    _pmm_jones_slant_solve,
    _pmm_jones_solve,
    _pmm_jones_solve_segments,
    _pmm_slant_solve,
    _pmm_solve,
    _pmm_solve_segments,
    _promote_eps_tensor,
    _resolve_incidence,
    _resolve_order_count,
    _stabilize_jones,
    _stabilize_scalar,
)


def pmm_jones_1d(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic 1-D binary grating by the Polynomial Modal Method -- the full
    complex ``2x2`` Jones reflection (the spectral-element counterpart of
    :func:`~lumenairy.elements.rcwa.rcwa_jones_1d`).

    The ridge and groove are full ``(3, 3)`` permittivity tensors -- IN-PLANE OR
    OUT-OF-PLANE (an off-diagonal-z ``exz/eyz/ezx/ezy`` cell routes to the metric
    generator); the off-diagonal ``exy`` couples ``E_x`` and ``E_y`` in the
    spectral-element modal eigenproblem, so the response is a full Jones matrix (the
    phase relationship the scalar :func:`pmm_efficiency_1d` cannot carry).  Converges SPECTRALLY in the polynomial ``degree`` with no
    accuracy floor -- the PMM win on metals where the FMM needs many orders and
    the ASR stretch plateaus.

    Parameters
    ----------
    period : float
        Grating period (metres).
    eps_ridge, eps_groove : (3, 3) array_like of complex
        FULL ``(3, 3)`` permittivity tensors of the ridge / groove (PUBLIC
        convention ``Im(eps) > 0`` for loss).  Pass ``scalar * np.eye(3)`` for an
        isotropic region; build LC tensors with
        :func:`~lumenairy.elements.rcwa.uniaxial_tensor`.  OUT-OF-PLANE coupling
        (``eps_xz / eps_yz / eps_zx / eps_zy != 0``, e.g. a tilted-director LC or
        a magneto-optic medium) is supported via the native full-3x3 metric
        generator (the ``eps_zz``-Schur of ``E_z`` is folded pointwise, Li 1999
        Eq.12); per-order matches :func:`~lumenairy.elements.rcwa.rcwa_jones_1d`
        to ``<1e-3``.  In-plane tensors take the (byte-identical) in-plane path.
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space (isotropic) indices (PUBLIC ``n =
        n + i kappa``).
    depth, duty_cycle, wavelength : float
        As in :func:`pmm_efficiency_1d` (the ridge occupies ``duty_cycle`` of
        the period).
    angle : float, optional
        Incidence angle (radians) in the x-z plane (classical mount, ``ky=0``).
        Accepts ``theta`` as a cross-dimension alias (the 2-D / conical polar-
        angle spelling, also used by ``RCWAStack``); overrides ``angle`` when
        given.  Oblique is supported via the ``+i kx0`` Bloch shift; the coupled tensor
        modes' forward set is chosen by the z-Poynting flux.  Lossless / mild-
        loss anisotropic (the tunable-LC case) is robust across angle; very
        lossy metal-corner TM at steep angle can be resonance-limited.
    degree : int, optional
        Polynomial degree per spectral element -- the spectral convergence knob.
        Default 16.
    elements_per_region : int, optional
        Spectral elements per homogeneous subsection (ridge / groove).  Default
        1.  Raise with ``grade=True`` to resolve the wall-corner field.
    grade : bool, optional
        Cluster the elements toward the walls when
        ``elements_per_region > 1``.  Default ``True``.
    far_field_orders : int, optional
        Rayleigh order count for the once-only forward far-field projection
        (auto-grown to cover the propagating orders).  Default 21.  Accepts
        ``n_orders`` as a cross-suite alias (the RCWA / 2-D PMM spelling); when
        given it overrides ``far_field_orders``.
    stabilize : bool, optional
        Guard against the isolated-degree PMM resonances (a near-singular
        layer<->region mode-match injects spurious flux and inflates
        ``sum(R)+sum(T)``).  When ``True`` (default) the solver scans a short
        upward degree window and returns the lowest degree at/above the request
        whose BOTH incident-polarization totals are energy-passive.  Set
        ``False`` to solve at exactly ``degree``.

    Returns
    -------
    orders : (M,) int ndarray
        Retained Rayleigh-order indices (the far-field projection set).
    R_eff, T_eff : (2, M) float ndarray
        Reflected / transmitted diffraction efficiency per order; row 0 is the
        response to an incident ``E_x`` wave, row 1 to incident ``E_y`` (cross-
        polarization included).
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis (PUBLIC
        ``exp(-i w t)`` convention); columns are the responses to incident
        ``E_x`` / ``E_y``, rows are ``[E_x; E_y]`` reflected.  Matches
        :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` to the convergence
        tolerance.

    Notes
    -----
    NumPy / SciPy (dense generalized eig) by default.  **JAX-differentiable** for an
    IN-PLANE tensor (``exz = eyz = ezx = ezy = 0``) on a VERTICAL grating when any
    index / geometry / tensor argument is a JAX array: it routes to a self-contained
    ``jax.numpy`` twin (the 2n x 2n coupled ``[E_x; E_y]`` standard eig solved
    directly by the reused ``rcwa`` custom-VJP eig), returning ``jax.grad``-able
    ``R_eff`` / ``T_eff`` and the 2x2 ``jones`` w.r.t. the tensor entries (incl. the
    off-diagonal ``exy`` / ``eyx`` cross-pol coupling, real and imaginary), ``depth``,
    ``wavelength``, ``angle`` and the half-space indices.  The numpy path is
    **byte-identical** (the JAX branch fires only on JAX inputs).  ``d/d(angle)`` at
    EXACTLY normal incidence is a symmetry-protected zero where the degenerate TE/TM
    eigen-subspace leaves a tiny (~5e-5) artifact against the true zero -- harmless
    (the gradient there IS zero; no gauge fix is applied since it would corrupt the
    gauge-invariant ``|J|`` observables); differentiate at any oblique angle for a
    clean value.  ``stabilize=True``, ``elements_per_region>1``, a SLANTED wall, and
    an OUT-OF-PLANE tensor raise on the JAX path (NumPy-only).  Normal or oblique
    incidence, binary grating, full ``(3, 3)`` tensor (in-plane OR out-of-plane).  An
    out-of-plane tensor routes through the native full-3x3 metric generator (VERTICAL
    only).  Out-of-plane combined with a slanted wall is SUPPORTED (binary) as of
    2026-06-07: :func:`pmm_jones_1d_slanted` carries the slant as EXACT convection
    (``tan * d/dx``) instead of the static metric fold (whose ``ezz*tan^2`` re-
    injection capped the per-order accuracy at ~1e-2 for the full tensor), so
    out-of-plane + slant reaches the same ~1e-4 wall-normal per-order floor as the
    in-plane slant TM channel (validated vs an RCWA tensor z-staircase across slant
    15-60 deg, normal + oblique).  The multi-region (segments) out-of-plane + slant
    path remains guarded.
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d: degree must be >= 2.")
    # A TRACED (jit) duty_cycle has no concrete value to range-check.
    if not is_jax_array(duty_cycle):
        if not (0.0 < float(duty_cycle) < 1.0):
            raise ValueError(
                f"pmm_jones_1d: duty_cycle must be strictly in (0, 1), got {duty_cycle}.")

    # JAX (differentiable) dispatch -- mirror pmm_efficiency_1d / rcwa.  Routed
    # to the self-contained jnp Jones twin ONLY when a tensor / geometry / angle
    # input is a JAX array; NumPy inputs fall through to the original
    # (byte-identical) code.  The JAX surface is binary, IN-PLANE tensor
    # (exz/eyz/ezx/ezy == 0), VERTICAL wall, NORMAL or OBLIQUE incidence, a
    # single fixed degree.  This branch MUST precede the np.asarray(_C) coercion
    # below, which would sever the JAX trace.  is_jax_array on a (3,3) tensor
    # array detects a JAX tensor input.
    _jax_inputs = (eps_ridge, eps_groove, n_substrate, n_superstrate, depth,
                   wavelength, duty_cycle, angle)
    if any(is_jax_array(a) for a in _jax_inputs):
        if stabilize:
            raise ValueError(
                "pmm_jones_1d: the JAX (differentiable) path requires "
                "stabilize=False (the degree-scan returns a discrete degree via "
                "host control flow, which is non-differentiable); pass a single "
                "fixed degree where the numpy solve already conserves energy.")
        import jax.numpy as _jnp
        er_j = _jnp.asarray(eps_ridge, _jnp.complex128)
        eg_j = _jnp.asarray(eps_groove, _jnp.complex128)
        if er_j.shape[-2:] != (3, 3) or eg_j.shape[-2:] != (3, 3):
            raise ValueError(
                "pmm_jones_1d: eps_ridge / eps_groove must be (3, 3) "
                "permittivity tensors (use scalar * np.eye(3) for isotropic).")
        # OUT-OF-PLANE (exz/eyz/ezx/ezy != 0) and SLANT stay NumPy-only (the
        # native metric generator is not differentiable yet) -- a precise raise
        # so the user is never silently given a wrong path.  Use the CONCRETE
        # off-plane magnitude when available (a traced off-plane entry cannot be
        # checked, so an all-traced tensor is assumed in-plane and validated by
        # the numpy oracle in tests).
        def _off_mag(M):
            try:
                arr = np.asarray(M, dtype=_C)
                return float(np.max(np.abs(arr[[0, 1, 2, 2], [2, 2, 0, 1]])))
            except Exception:
                return 0.0
        scale = 1.0
        try:
            scale = max(float(np.max(np.abs(np.asarray(eps_ridge, dtype=_C)))),
                        float(np.max(np.abs(np.asarray(eps_groove, dtype=_C)))),
                        1.0)
        except Exception:
            scale = 1.0
        off = max(_off_mag(eps_ridge), _off_mag(eps_groove))
        if off > 1e-9 * scale:
            raise NotImplementedError(
                "pmm_jones_1d: the JAX (differentiable) path supports IN-PLANE "
                "tensors only (exz/eyz/ezx/ezy == 0); the out-of-plane native "
                "metric generator is NumPy-only.")
        return _pmm_jones_1d_jax(
            period, er_j, eg_j, n_substrate, n_superstrate, depth, duty_cycle,
            wavelength, angle=angle, degree=int(degree),
            elements_per_region=int(elements_per_region), grade=bool(grade),
            far_field_orders=int(far_field_orders))

    er = np.asarray(eps_ridge, dtype=_C)
    eg = np.asarray(eps_groove, dtype=_C)
    if er.shape[-2:] != (3, 3) or eg.shape[-2:] != (3, 3):
        raise ValueError(
            "pmm_jones_1d: eps_ridge / eps_groove must be (3, 3) permittivity "
            "tensors (use scalar * np.eye(3) for an isotropic region).")
    # FULL-3x3 OUT-OF-PLANE (exz/eyz/ezx/ezy != 0): route to the native full-3x3
    # metric generator (_pmm_jones_slant_solve at slant=0 -- the same operator the
    # slant/oblique paths use, now carrying the out-of-plane pointwise ezz-Schur).
    # The in-plane tensor keeps the (byte-identical) second-order _pmm_jones_solve.
    # Normal AND oblique are supported; per-order matches rcwa_jones_1d to <1e-3.
    scale = max(float(np.max(np.abs(er))), float(np.max(np.abs(eg))), 1.0)
    off = max(float(np.max(np.abs(er[[0, 1, 2, 2], [2, 2, 0, 1]]))),
              float(np.max(np.abs(eg[[0, 1, 2, 2], [2, 2, 0, 1]]))))
    if off > 1e-9 * scale:
        sargs = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
                 duty_cycle, wavelength, 0.0)
        skw = dict(n_ridge_el=int(elements_per_region),
                   n_groove_el=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_slant_solve(*sargs, degree=int(degree),
                                                   **skw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_slant_solve(*sargs, degree=d, **skw)[:4],
            int(degree), "pmm_jones_1d")

    args = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
            duty_cycle, wavelength)
    kw = dict(n_ridge_el=int(elements_per_region),
              n_groove_el=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_solve(*args, degree=int(degree), **kw)
        return o, R, T, J
    # Per-order + Jones convergence consensus (rejects the super-unity resonances
    # AND the under-resolved-but-energy-passive degrees; see _stabilize_jones).
    return _stabilize_jones(
        lambda d: _pmm_jones_solve(*args, degree=d, **kw)[:4], int(degree),
        "pmm_jones_1d")



def pmm_efficiency_1d(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    polarization: str = "te",
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 1-D binary grating by the
    Polynomial Modal Method (subsectional spectral element; Edee 2011).

    A NON-Fourier alternative to :func:`~lumenairy.elements.rcwa.rcwa_efficiency_1d`
    that converges SPECTRALLY in the polynomial ``degree`` (no Gibbs at the
    walls), with no accuracy floor and well-conditioned interfaces -- see the
    module docstring for why this beats the Fourier method and the ASR
    coordinate stretch for metals / high-contrast TM.

    Parameters
    ----------
    period, n_ridge, n_groove, n_substrate, n_superstrate, depth, duty_cycle,
    wavelength : as in :func:`~lumenairy.elements.rcwa.rcwa_efficiency_1d`
        (metres / PUBLIC ``n = n + i kappa``).  The ridge occupies the fraction
        ``duty_cycle`` of the period.
    angle : float, optional
        Incidence angle (radians).  Accepts ``theta`` as a cross-dimension alias
        (the 2-D / conical polar-angle spelling); overrides ``angle`` when given.
        Oblique is supported via the ``+i kx0``
        Bloch shift of the pseudo-periodic envelope (the convection term is
        antisymmetrized so the wall-varying ``1/eps`` weight is handled
        correctly for TM); the forward modes use a noise-robust branch.
        Dielectric is robust across angle; very lossy metal-corner TM at steep
        angle can be resonance-limited (``stabilize`` may raise -- use rcwa).
    polarization : {'te', 'tm'}, optional
        ``'te'`` (E along the grooves) or ``'tm'``.  Default ``'te'``.
    degree : int, optional
        Polynomial degree per spectral element -- the SPECTRAL convergence knob
        (raise it for accuracy, not the element count).  Default 16.
    elements_per_region : int, optional
        Spectral elements per homogeneous subsection (ridge / groove).  Default
        1.  Raise (e.g. 2-4) with ``grade=True`` to resolve the metal-corner
        field singularity -- the speed lever for TM (hp-refinement).
    grade : bool, optional
        When ``elements_per_region > 1``, cluster the elements toward the walls
        (Chebyshev-Lobatto) to resolve the corner singularity.  Default
        ``True`` (no effect for ``elements_per_region == 1``).
    far_field_orders : int, optional
        Rayleigh order count for the once-only forward far-field projection
        (auto-grown to cover the propagating orders; kept well below the nodal
        DOF).  Default 21.  Accepts ``n_orders`` as a cross-suite alias (the
        RCWA / 2-D PMM spelling); when given it overrides ``far_field_orders``.
    stabilize : bool, optional
        Guard against residual quasi-resonances (a near-singular
        layer<->region mode-match injecting spurious flux; the analogue of the
        FMM ``stabilize`` flag).  When ``True`` (default) the solver scans a
        short UPWARD degree window and returns the CONSENSUS result -- the
        value the converged degrees agree on, with the ENERGY-CLEANEST cluster
        member preferred on lossless structures (v5.14: two marginal degrees
        could previously corroborate each other ~1e-3 off).  The consensus is
        accurate to the scan's per-order tolerance (~3e-3 worst case), so it is
        NOT strictly monotone in the requested degree.  NB v5.14 also fixed
        the root cause of the formerly-DENSE normal-incidence resonances (a
        noise-sensitive legacy forward-mode branch); with that fix
        ``stabilize=False`` conserves energy at every probed degree, and the
        consensus is a safety net rather than a necessity.  Set ``False`` to
        solve at exactly ``degree``.

    Returns
    -------
    orders : (M,) int ndarray
        Retained Rayleigh-order indices (the far-field projection set).
    R_eff, T_eff : (M,) float ndarray
        Reflected / transmitted diffraction efficiency per order (real power
        fractions; evanescent orders 0).  For a lossless grating
        ``sum(R)+sum(T) == 1``; with loss the deficit is absorptance.

    Notes
    -----
    NumPy / SciPy (dense generalized eig) by default.  **JAX-differentiable** when
    any index / geometry argument is a JAX array (mirrors ``rcwa``): the call then
    routes to a self-contained ``jax.numpy`` twin and returns ``jax.grad``-able
    efficiencies w.r.t. ``eps`` (via ``n``, including COMPLEX/lossy eps -- both the
    real and imaginary parts, ``holomorphic=False``), ``depth``, ``wavelength``,
    ``angle`` and ``n_superstrate`` (the Bloch ``kx0`` convection is traced), AND
    ``duty_cycle`` (the moving grating wall -- a smooth fixed-topology shape
    gradient: the wall sits on an element boundary so the element Jacobians and the
    Rayleigh-projection phases carry it analytically, no remeshing), reusing the
    validated Lorentzian-broadened custom-VJP eig from ``rcwa``.  The numpy
    path is **byte-identical** (the JAX branch fires only on JAX inputs).  The
    differentiable surface is binary, NORMAL or OBLIQUE incidence, lossless or
    lossy eps, ``elements_per_region=1``, a single fixed ``degree`` with
    ``stabilize=False`` (a resonant degree gives the converged jnp answer but a
    non-physical numpy one -- pick a degree where the numpy solve already conserves
    energy); ``stabilize=True`` and ``elements_per_region>1`` raise on the JAX path
    (NumPy-only for now).  For a LOSSY cell, validate per-order / absorbed-fraction
    against an oracle, NOT energy (a lossless cell auto-balances power even with a
    wrong split).  Gradients are valid BETWEEN Rayleigh-order cutoffs (the
    propagating-order count is fixed per trace; set ``far_field_orders`` high enough
    to cover the working wavelength when differentiating ``wavelength``).  Requires
    ``jax_enable_x64`` (a warning fires otherwise -- complex64 is too ill-conditioned
    for the modal eig).  TM converges monotone-with-no-floor but only
    spectral-*ish* (the discontinuous TM partner is C0-averaged at the wall);
    ``elements_per_region>1, grade=True`` recovers the rate for metals.
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_1d: polarization must be 'te' or 'tm', got "
            f"{polarization!r}.")
    if int(degree) < 2:
        raise ValueError("pmm_efficiency_1d: degree must be >= 2.")
    # A TRACED (jit) duty_cycle has no concrete value to range-check; the JAX
    # path enforces 0 < duty < 1 on the concrete proxy / leading value instead.
    if not is_jax_array(duty_cycle):
        if not (0.0 < float(duty_cycle) < 1.0):
            raise ValueError(
                f"pmm_efficiency_1d: duty_cycle must be strictly in (0, 1), got "
                f"{duty_cycle}.")

    # JAX (differentiable) dispatch -- mirror rcwa's backend detection.  Routed
    # to the self-contained jnp twin ONLY when an index / geometry input is a
    # JAX array; NumPy inputs fall through to the original code verbatim (so the
    # NumPy path stays BYTE-IDENTICAL).  The JAX path covers the minimal
    # validated surface: binary, NORMAL incidence, fixed single degree.  This
    # branch MUST precede the _C(...) coercion below, which would sever the JAX
    # trace.
    if any(is_jax_array(a) for a in (n_ridge, n_groove, n_substrate,
                                     n_superstrate, depth, wavelength,
                                     duty_cycle, angle)):
        if stabilize:
            raise ValueError(
                "pmm_efficiency_1d: the JAX (differentiable) path requires "
                "stabilize=False (the degree-scan returns a discrete degree via "
                "host control flow, which is non-differentiable); pass a single "
                "fixed degree.")
        return _pmm_efficiency_1d_jax(
            period, n_ridge, n_groove, n_substrate, n_superstrate, depth,
            duty_cycle, wavelength, angle=angle, polarization=pol,
            degree=int(degree), elements_per_region=int(elements_per_region),
            grade=bool(grade), far_field_orders=int(far_field_orders))

    args = (period, _C(n_ridge), _C(n_groove), _C(n_substrate),
            _C(n_superstrate), depth, duty_cycle, wavelength)
    kw = dict(polarization=pol, n_ridge_el=int(elements_per_region),
              n_groove_el=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        orders, R, T, _ = _pmm_solve(*args, degree=int(degree), **kw)
        return orders, R, T
    # Robust degree selection by per-order CONVERGENCE CONSENSUS: collect the
    # PASSIVE solves (total within _PASSIVE_TOL of unity -- discards the
    # super-unity resonances) and lock onto the plateau the converged degrees
    # AGREE ON per-order (the total alone is conserved even when under-resolved);
    # return the requested degree if it is in the plateau, else the lowest
    # converged degree, else warn/raise.  See _stabilize_scalar.
    return _stabilize_scalar(
        lambda d: _pmm_solve(*args, degree=d, **kw)[:3], int(degree),
        "pmm_efficiency_1d")



def pmm_efficiency_1d_jax(
    period,
    n_ridge,
    n_groove,
    n_substrate,
    n_superstrate,
    depth,
    duty_cycle,
    wavelength,
    *,
    angle=0.0,
    theta=None,
    polarization="te",
    degree=16,
    elements_per_region=1,
    grade=True,
    far_field_orders=21,
    n_orders=None,
):
    """JAX (differentiable) twin of :func:`pmm_efficiency_1d`.

    Thin wrapper that promotes its inputs to ``jax.numpy`` arrays and forwards to
    the unified :func:`pmm_efficiency_1d`, which auto-dispatches to the
    differentiable JAX backend on JAX inputs.  Prefer calling
    ``pmm_efficiency_1d(...)`` with ``jax.numpy`` arguments directly; this explicit
    twin exists for discoverability and the cross-backend parity contract.  The JAX
    path uses ``stabilize=False`` (passed automatically) -- choose a single fixed
    ``degree`` where the NumPy solve already conserves energy.  Differentiable
    w.r.t. ``n_ridge`` / ``n_groove`` (via eps), ``depth``, ``wavelength``,
    ``angle`` and the half-space indices; first-order only (no eig Hessian).
    """
    from ...backend import JAX_AVAILABLE as _JAX_AVAILABLE
    if not _JAX_AVAILABLE:
        raise ImportError(
            "pmm_efficiency_1d_jax requires the optional 'jax' extra; install with "
            "`pip install lumenairy[jax]` (or `pip install jax`).  Use the NumPy "
            "pmm_efficiency_1d for non-differentiable evaluation.")
    import jax.numpy as jnp
    return pmm_efficiency_1d(
        period,
        jnp.asarray(n_ridge),
        jnp.asarray(n_groove),
        jnp.asarray(n_substrate),
        jnp.asarray(n_superstrate),
        jnp.asarray(depth),
        duty_cycle,
        jnp.asarray(wavelength),
        angle=jnp.asarray(angle),
        theta=theta,
        polarization=polarization,
        degree=degree,
        elements_per_region=elements_per_region,
        grade=grade,
        far_field_orders=far_field_orders,
        n_orders=n_orders,
        stabilize=False,
    )



def pmm_efficiency_1d_segments(
    period: float,
    segments,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    polarization: str = "te",
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scalar diffraction efficiencies of an ARBITRARY piecewise-constant 1-D
    grating by the PMM -- the multi-region / multi-level generalization of
    :func:`pmm_efficiency_1d` (the 2-region ridge/groove special case).

    The PMM's fast isotropic path: each region carries a scalar (possibly
    complex) refractive index, a region wall lands on every spectral-element
    boundary (so ``eps`` is exact per element -- no Gibbs), and the solve
    converges spectrally in ``degree`` with no accuracy floor.  For anisotropic
    (tensor) regions use :func:`pmm_jones_1d_segments`.

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_efficiency_1d`.
    segments : list of (width_fraction, n)
        Consecutive regions along ``x`` (the ridge side first), each a
        ``(width_fraction, refractive_index)`` pair; the fractions must sum to 1
        (within ``1e-6``).  Covers multi-level staircases (blazed-grating
        approximations) and arbitrary multi-region cells.
    angle, polarization, degree, elements_per_region, grade, far_field_orders,
    stabilize : as in :func:`pmm_efficiency_1d`.

    Returns
    -------
    orders, R_eff, T_eff : as in :func:`pmm_efficiency_1d`.
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_1d_segments: polarization must be 'te' or 'tm', "
            f"got {polarization!r}.")
    if int(degree) < 2:
        raise ValueError("pmm_efficiency_1d_segments: degree must be >= 2.")
    if len(segments) < 1:
        raise ValueError(
            "pmm_efficiency_1d_segments: need at least one segment.")
    widths = [float(w) for w, _ in segments]
    seg_n = [_C(n) for _, n in segments]
    sa = (period, widths, seg_n, _C(n_substrate), _C(n_superstrate), depth,
          wavelength)
    kw = dict(polarization=pol, n_el_per_region=int(elements_per_region),
              grade=bool(grade), far_field_orders=int(far_field_orders),
              angle=float(angle))

    if not stabilize:
        o, R, T, _ = _pmm_solve_segments(*sa, degree=int(degree), **kw)
        return o, R, T
    return _stabilize_scalar(
        lambda d: _pmm_solve_segments(*sa, degree=d, **kw)[:3], int(degree),
        "pmm_efficiency_1d_segments")



def pmm_jones_1d_segments(
    period: float,
    segments,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic 1-D grating with an ARBITRARY piecewise-constant profile by
    the PMM -- the multi-region / multi-level generalization of
    :func:`pmm_jones_1d` (the 2-segment ridge/groove special case) and the
    spectral-element counterpart of
    :func:`~lumenairy.elements.rcwa.rcwa_jones_1d_segments`.

    Each region carries its own (possibly anisotropic) IN-PLANE permittivity, so
    the response is a full complex ``2x2`` Jones reflection.  Covers multi-level
    staircases, interdigitated / N-region cells, and mixed isotropic / liquid-
    crystal regions (e.g. the grounded-tooth | LC | floating-tooth | LC device
    class).  Converges spectrally in ``degree`` with no accuracy floor.

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_jones_1d`.
    segments : list of (width_fraction, eps)
        Consecutive regions along ``x``; each ``eps`` is a scalar (isotropic
        region) or a ``(3, 3)`` IN-PLANE permittivity tensor.  Width fractions
        must sum to 1 (within ``1e-6``).  Accepts the output of the
        ``grating_segments`` / ``binary_grating_segments`` /
        ``interdigitated_grating_segments`` builders.
    angle, degree, elements_per_region, grade, far_field_orders, stabilize : as
        in :func:`pmm_jones_1d`.

    Returns
    -------
    orders, R_eff, T_eff, jones_reflection : as in :func:`pmm_jones_1d`.
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d_segments: degree must be >= 2.")
    if len(segments) < 1:
        raise ValueError("pmm_jones_1d_segments: need at least one segment.")
    widths = [float(w) for w, _ in segments]
    tensors = []
    for _w, eps in segments:
        M = np.asarray(eps, dtype=_C)
        if M.ndim == 0:                         # scalar -> isotropic tensor
            M = M * np.eye(3, dtype=_C)
        if M.shape[-2:] != (3, 3):
            raise ValueError(
                "pmm_jones_1d_segments: each segment eps must be a scalar or a "
                "(3, 3) permittivity tensor.")
        tensors.append(M)
    # FULL-3x3 OUT-OF-PLANE (exz/eyz/ezx/ezy != 0): route to the native full-3x3
    # metric generator (_pmm_jones_slant_segments_solve at slant=0 -- region-count-
    # agnostic, carrying the out-of-plane pointwise ezz-Schur).  The in-plane
    # tensor keeps the (byte-identical) second-order _pmm_jones_solve_segments.
    scale = max([float(np.max(np.abs(M))) for M in tensors] + [1.0])
    off = max(float(np.max(np.abs(M[[0, 1, 2, 2], [2, 2, 0, 1]])))
              for M in tensors)
    if off > 1e-9 * scale:
        oargs = (period, widths, tensors, _C(n_substrate), _C(n_superstrate),
                 depth, wavelength, 0.0)
        okw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_slant_segments_solve(
                *oargs, degree=int(degree), **okw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_slant_segments_solve(*oargs, degree=d, **okw)[:4],
            int(degree), "pmm_jones_1d_segments")
    sa = (period, widths, tensors, _C(n_substrate), _C(n_superstrate), depth,
          wavelength)
    kw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_solve_segments(*sa, degree=int(degree), **kw)
        return o, R, T, J
    return _stabilize_jones(
        lambda d: _pmm_jones_solve_segments(*sa, degree=d, **kw)[:4],
        int(degree), "pmm_jones_1d_segments")



def pmm_efficiency_1d_slanted(
    period: float,
    n_ridge: complex,
    n_groove: complex,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    slant_angle: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    polarization: str = "te",
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rigorous diffraction efficiencies of a 1-D SLANTED binary grating by the
    inclined-coordinate Polynomial Modal Method (Granet, Randriamihaja &
    Raniriharinosy, JOSA A 34:975, 2017).

    Extends :func:`pmm_efficiency_1d` to a grating whose straight side-walls are
    tilted by ``slant_angle`` from the vertical.  Where a Fourier method
    (:func:`~lumenairy.elements.rcwa.rcwa_efficiency_1d`) must STAIRCASE the
    slant into many laterally-shifted thin layers and converge in the slice
    count, PMM solves a SINGLE slanted layer exactly in the inclined coordinate
    ``u = x - tan(slant_angle) z`` -- no z-staircase -- reaching the converged
    efficiencies at a fraction of the cost (~30-70x fewer DOF*slices in the
    validated dielectric cases).

    Parameters
    ----------
    period ... wavelength : as in :func:`pmm_efficiency_1d` (metres / PUBLIC
        ``n = n + i kappa``).  The (projected) ridge occupies ``duty_cycle`` of
        the period; the planar layer interfaces are at ``z = 0`` and
        ``z = depth``.
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` = vertical (then this
        reduces bit-identically to :func:`pmm_efficiency_1d`).  Validated for
        ``0 <= slant_angle <= ~75 deg``; conditioning grows as
        ``sec^2(slant_angle)`` at steep tilt.
    angle : float, optional
        Incidence angle (radians).  COMBINED oblique incidence + nonzero slant IS
        supported: it is routed through the metric-generator Jones solver
        (:func:`pmm_jones_1d_slanted`) with an isotropic tensor and the requested
        scalar channel extracted (see Notes), per-order to ~2-3e-3 vs an RCWA
        staircase.  At normal incidence (or a vertical grating) the dedicated
        inclined-coordinate scalar solver is used.
    polarization, degree, elements_per_region, grade, far_field_orders,
    stabilize : as in :func:`pmm_efficiency_1d`.

    Returns
    -------
    orders, R_eff, T_eff : as in :func:`pmm_efficiency_1d`.

    Notes
    -----
    The slant injects a linear-in-``q`` convection term (the modal eigenproblem
    becomes quadratic, companion-linearized) and breaks the ``+/-q`` field
    symmetry, so the explicit forward/backward generalized S-matrix is used.
    The inclined coordinate is INTERNAL to the patterned layer; the homogeneous
    half-spaces and the Rayleigh far-field projection are in the lab ``(x, z)``
    frame.  TE matches a fine RCWA staircase to ~1e-5; TM self-converges and is
    the BETTER reference (RCWA-TM is Gibbs/slice-limited).

    SCOPE: binary (1 ridge + 1 groove), any slant, NORMAL or OBLIQUE incidence.
    The dedicated inclined-coordinate scalar eigenproblem is used at normal
    incidence (and for a vertical grating at any angle); for combined oblique
    incidence + nonzero slant -- where that scalar solver's ``kx0 <-> slant``
    convection cross-coupling is unresolved -- the call is delegated to the
    genuine Edee-Granet 2024 metric generator via :func:`pmm_jones_1d_slanted`
    (isotropic tensor ``n^2 I``, scalar channel extracted: TE = E along the
    grooves = Jones row 1, TM = row 0), which resolves the cross-coupling with the
    round-19 div-conforming ``E_z`` closure (per-order to ~2-3e-3 vs an RCWA
    staircase -- the wall-normal TM/p-pol inverse-rule limit plus the oblique-slant
    convection, not a coupling error; the in-plane wall-normal TM floor alone is much
    tighter, ~3e-5..1e-4 near degree 18-20, U-shaped).
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    pol = polarization.lower()
    if pol not in ("te", "tm"):
        raise ValueError(
            f"pmm_efficiency_1d_slanted: polarization must be 'te' or 'tm', "
            f"got {polarization!r}.")
    if int(degree) < 2:
        raise ValueError("pmm_efficiency_1d_slanted: degree must be >= 2.")
    if not (0.0 < float(duty_cycle) < 1.0):
        raise ValueError(
            f"pmm_efficiency_1d_slanted: duty_cycle must be strictly in (0, 1), got "
            f"{duty_cycle}.")
    if abs(float(angle)) > 1e-12 and abs(float(slant_angle)) > 1e-12:
        # COMBINED OBLIQUE + SLANT.  The scalar inclined-frame solver's own
        # Bloch<->slant convection cross-term is unresolved (energy conserves but
        # the per-order split is wrong), so route through the genuine Edee-Granet
        # 2024 metric generator (pmm_jones_1d_slanted), whose round-19 div-
        # conforming E_z closure DOES handle combined oblique + slant (per-order
        # to ~2-3e-3 vs an RCWA staircase, degree-cleanly).  An isotropic region
        # is the diagonal tensor n^2 * I; the Jones response is then uncoupled, so
        # the scalar channel is a single row: TE (E along the grooves, E_y) is
        # Jones row 1, TM (E_x) is row 0.
        eye3 = np.eye(3)
        o, R_j, T_j, _ = pmm_jones_1d_slanted(
            period, (_C(n_ridge) ** 2) * eye3, (_C(n_groove) ** 2) * eye3,
            _C(n_substrate), _C(n_superstrate), depth, duty_cycle, wavelength,
            float(slant_angle), angle=float(angle), degree=int(degree),
            elements_per_region=int(elements_per_region), grade=bool(grade),
            far_field_orders=int(far_field_orders), stabilize=bool(stabilize))
        row = 1 if pol == "te" else 0
        return o, R_j[row], T_j[row]

    args = (period, _C(n_ridge), _C(n_groove), _C(n_substrate),
            _C(n_superstrate), depth, duty_cycle, wavelength,
            float(slant_angle))
    kw = dict(angle=float(angle), polarization=pol,
              elements_per_region=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders))

    if not stabilize:
        orders, R, T, _ = _pmm_slant_solve(*args, degree=int(degree), **kw)
        return orders, R, T
    return _stabilize_scalar(
        lambda d: _pmm_slant_solve(*args, degree=d, **kw)[:3], int(degree),
        "pmm_efficiency_1d_slanted")



def pmm_jones_1d_slanted(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    duty_cycle: float,
    wavelength: float,
    slant_angle: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
    factorization: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SLANTED binary grating with full ``(3, 3)`` permittivity tensors (in-plane
    OR out-of-plane) -- the anisotropic-Jones counterpart of
    :func:`pmm_efficiency_1d_slanted`, by the Edee-Granet convection-metric
    spectral-element solver.

    Combines the tilted side-walls of :func:`pmm_efficiency_1d_slanted` with the
    coupled ``(E_x, E_y)`` Jones response of :func:`pmm_jones_1d`.  The tilt
    enters as an EXACT first-order convection (``tan * d/dx`` on the clean
    vertical metric generator -- this is what lets out-of-plane + slant reach the
    wall-normal floor; see :func:`_build_generator_metric`), and the modal
    eigenproblem is the TRUE first-order physical Maxwell
    generator ``-i k gamma psi = L psi`` (``psi = [E_x; E_y; iZ H_x; iZ H_y]``,
    LINEAR in ``gamma``): the magnetic field is read directly from the
    eigenvector, so the layer modes are flux-orthogonal by construction (the
    symplectic property a reshaped convection pencil lacks) and energy conserves
    to ~1e-13 across ``0-60`` deg for both real-symmetric and gyrotropic tensors.

    Parameters
    ----------
    period : float
        Grating period (metres).
    eps_ridge, eps_groove : (3, 3) array_like of complex
        IN-PLANE permittivity tensors of the ridge / groove (PUBLIC convention
        ``Im(eps) > 0`` for loss).  As in :func:`pmm_jones_1d`: pass
        ``scalar * np.eye(3)`` for an isotropic region; ``exy`` / ``eyx`` couple
        ``E_x`` and ``E_y`` (real-symmetric for a tilted LC director, anti-
        Hermitian ``+/- i g`` for a gyrotropic / magneto-optic medium).  FULL
        out-of-plane coupling (``eps_xz / eps_yz / eps_zx / eps_zy != 0``, e.g. a
        tilted-director LC) is SUPPORTED (binary grating): it routes through the
        coupled metric generator and reaches the same ~1e-4 wall-normal per-order
        floor as the in-plane slant TM channel (validated vs an RCWA tensor
        z-staircase, slant 15-60 deg, normal + oblique).
    n_substrate, n_superstrate : complex
        Transmission / incidence half-space (isotropic) indices (PUBLIC ``n =
        n + i kappa``).
    depth, duty_cycle, wavelength : float
        As in :func:`pmm_efficiency_1d_slanted` (the projected ridge occupies
        ``duty_cycle`` of the period; the planar interfaces are at ``z = 0`` and
        ``z = depth``).
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` = vertical (then this
        reduces to :func:`pmm_jones_1d`).  Validated for ``0 <= slant_angle <=
        ~60 deg`` (conditioning grows as ``sec^2`` at steep tilt).
    angle : float, optional
        Incidence angle (radians).  COMBINED oblique incidence + nonzero slant is
        SUPPORTED (the round-19 div-conforming ``E_z`` closure removes the
        Bloch-amplified Liu-2015 spurious null; ``kx0 = k0 Re(n_sup) sin(angle)``
        is wired through the generator, the half-spaces, and the lab Rayleigh far
        field): the metric generator conserves energy ~1e-13 and the per-order
        split matches an RCWA staircase to ~2-3e-3 at oblique+slant.  (The in-plane
        wall-normal TM/p-pol inverse-rule floor itself is far tighter: it converges
        to ~3e-5..1e-4 near degree 18-20, U-shaped, then mildly degrades past ~degree
        22 as the benign flux-null evanescent spurious sea grows with slant -- energy
        stays ~1e-13 throughout; TE is machine-clean ~1e-6.)  Combined oblique + slant
        always routes through the metric generator (the scalar diagonal cure's
        oblique+slant per-order is wrong).
    degree, elements_per_region, grade, far_field_orders, stabilize : as in
        :func:`pmm_jones_1d`.
    factorization : {'auto', 'convection', 'covariant'}, optional
        Slant treatment.  ``'auto'`` (default) picks the best path per cell:
        ``'covariant'`` for ANY slanted cell -- in-plane OR out-of-plane (the
        spectral win) -- and ``'convection'`` otherwise (vertical).
        ``'convection'`` carries the tilt as an exact first-order convection on
        the lab-Cartesian metric generator -- robust at all slants and tensors,
        but the TM/p-pol per-order accuracy converges ALGEBRAICALLY (~1e-4 at
        practical degree).  ``'covariant'`` routes through the Li-1999 oblique-
        coordinate covariant generator: the slanted wall becomes a coordinate
        surface, so the wall-normal discontinuity is handled algebraically and the
        TM channel converges SPECTRALLY (vertical-grade ~1e-7 by degree ~24) --
        the SAME physical answer as ``'convection'`` but ~100-2400x fewer degrees
        for matched accuracy.  ``'covariant'`` handles diagonal, coupled
        (``exy``/``eyx``), AND OUT-OF-PLANE (full 3x3 ``eps_xz/yz/zx/zy``)
        tensors, normal + oblique, lossless + lossy -- the out-of-plane coupling
        enters via the pointwise ezz-Schur composites (Li Eq.12) + the ``cos*Dop``
        single-derivative cross blocks (Li Eq.18/19), spectral at slant, so
        ``'auto'`` routes out-of-plane slanted cells to it too (~15x fewer degrees
        than convection).  Pass ``'convection'`` explicitly to force the fully-
        general
        algebraic path (e.g. for byte-stable cross-checks).

    Returns
    -------
    orders : (M,) int ndarray
        Retained Rayleigh-order indices.
    R_eff, T_eff : (2, M) float ndarray
        Reflected / transmitted diffraction efficiency per order; row 0 is the
        response to an incident ``E_x`` wave, row 1 to incident ``E_y`` (cross-
        polarization included).
    jones_reflection : (2, 2) complex ndarray
        Zeroth-order Jones reflection matrix in the lab ``(x, y)`` basis (PUBLIC
        ``exp(-i w t)`` convention); columns are the responses to incident
        ``E_x`` / ``E_y``, rows are ``[E_x; E_y]`` reflected.

    Notes
    -----
    NumPy / SciPy (dense eig); not JAX-differentiable.  SCOPE: BINARY grating
    (1 ridge + 1 groove), full ``(3, 3)`` tensor IN-PLANE OR OUT-OF-PLANE, normal
    OR oblique incidence at any slant.  The multi-region
    :func:`pmm_jones_1d_slanted_segments` companion carries full ``(3, 3)``
    tensors in-plane OR out-of-plane on every factorization.

    DIAGONAL CURE (round 16; Granet 2017 JOSA A 34:975 / Granet 2023; Liu 2015
    CiCP 18:467).  A DIAGONAL tensor (``exy = eyx = 0``) WITH ``exx == ezz`` in
    BOTH regions at NORMAL incidence (or any incidence on a VERTICAL grating) is
    solved through the DIV-CONFORMING scalar slant operator (``_sem_modes_slant``:
    the Li ``1/eps`` inverse rule sits INSIDE the z-stiffness, so it is free of the
    Liu-2015 spurious harmonic-mean static mode) -- TE via ``n = sqrt(eyy)``, TM
    via ``n = sqrt(exx)``, assembled into the diagonal Jones.  This is the FASTER
    (scalar ``n x n`` vs tensor ``4n x 4n``) and MORE-ACCURATE-per-degree path for
    that case.

    METRIC GENERATOR (round 11 + round-19 div-conforming closure).  COUPLED
    tensors (``exy / eyx != 0``), diagonal tensors with ``exx != ezz``, AND ALL
    combined oblique+slant cases use the convection-metric ``[E;H]`` first-order
    generator (``_build_generator_metric``).  As of round 19 its ``E_z``
    elimination is DIV-CONFORMING at ALL slants (``1/ezz`` placed BETWEEN the
    discrete z-derivatives, ``+ iS0 INT(1/ezz) B' B'``, Granet 2023 Eq.16-18 /
    Popov-Neviere App.B), so its TM-block spectrum bit-matches the scalar slant
    solver, is free of the Liu-2015 spurious harmonic-mean null, and per-order TM
    converges to the scalar oracle (~6.5e-4 deg32 / ~4.3e-4 deg40 at 45 deg).
    Energy conserves to ~1e-13 in all paths.  The combined oblique+slant far field
    (inclined-frame consistent, lab half-spaces) is degree-clean -- no stabilize
    crutch.
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d_slanted: degree must be >= 2.")
    if not (0.0 < float(duty_cycle) < 1.0):
        raise ValueError(
            f"pmm_jones_1d_slanted: duty_cycle must be strictly in (0, 1), got "
            f"{duty_cycle}.")
    er = np.asarray(eps_ridge, dtype=_C)
    eg = np.asarray(eps_groove, dtype=_C)
    if er.shape[-2:] != (3, 3) or eg.shape[-2:] != (3, 3):
        raise ValueError(
            "pmm_jones_1d_slanted: eps_ridge / eps_groove must be (3, 3) "
            "permittivity tensors (use scalar * np.eye(3) for an isotropic "
            "region).")
    # OUT-OF-PLANE (eps_xz/yz/zx/zy != 0): SUPPORTED (2026-06-07).  Since the
    # slant is carried as EXACT convection in the metric generator (see
    # _build_generator_metric), out-of-plane + slant now reaches the SAME
    # wall-normal ~1e-4 per-order floor as the in-plane slant TM channel
    # (validated vs an independent RCWA tensor z-staircase across slant 15-60 deg,
    # normal + oblique, symmetric / lossy / asymmetric tensors; energy conserves
    # to ~1e-13).  An out-of-plane tensor MUST route through the coupled metric
    # generator and NOT the scalar diagonal cure (which is the z-decoupled in-plane
    # subset and would silently drop the off-plane coupling), so `off` excludes it
    # from the diagonal cure below.  BINARY grating only -- the multi-region
    # (segments) out-of-plane + slant path stays guarded pending its own
    # per-order validation.
    scale = max(float(np.max(np.abs(er))), float(np.max(np.abs(eg))), 1.0)
    off = max(float(np.max(np.abs(er[[0, 1, 2, 2], [2, 2, 0, 1]]))),
              float(np.max(np.abs(eg[[0, 1, 2, 2], [2, 2, 0, 1]]))))

    # ---- COVARIANT OBLIQUE-COORDINATE path (SPECTRAL slant) ----------------
    # `factorization='covariant'` routes the slanted layer through the Li-1999
    # oblique-coordinate covariant generator instead of the convection generator.
    # The slanted wall becomes a coordinate surface, so the TM/p-pol channel
    # converges SPECTRALLY (vertical-grade ~1e-7 by degree ~24) instead of the
    # convection path's ALGEBRAIC ~1e-4 floor -- same physical answer, ~100-2400x
    # fewer degrees.  Handles diagonal AND coupled (exy/eyx) IN-PLANE tensors,
    # normal + oblique, lossless + lossy, IN-PLANE OR OUT-OF-PLANE.  The full-3x3
    # out-of-plane coupling (eps_xz/yz/zx/zy) enters the covariant generator via
    # the pointwise ezz-Schur composites (Li Eq.12) + the cos*Dop single-x-
    # derivative cross blocks (see _cov_blocks / _cov_generator_4n) -- SPECTRAL at
    # slant, same as the in-plane covariant.  The DEFAULT 'auto' picks covariant
    # for ANY slanted cell (the spectral win, now including out-of-plane) and
    # convection otherwise (vertical, where the vertical solver is already exact).
    if factorization not in ("auto", "convection", "covariant"):
        raise ValueError(
            "pmm_jones_1d_slanted: factorization must be 'auto', 'convection' "
            f"or 'covariant', got {factorization!r}.")
    if factorization == "auto":
        factorization = ("covariant"
                         if abs(float(slant_angle)) >= _COV_MIN_SLANT_RAD
                         else "convection")
    if (factorization == "covariant"
            and abs(float(slant_angle)) < _COV_MIN_SLANT_RAD):
        # The covariant oblique frame u = x - tan(phi) z DEGENERATES at ~zero slant:
        # the isotropic half-spaces' TE/TM modes become EXACTLY degenerate, the
        # interface mode-match goes near-singular, and the solve becomes BLAS-build-
        # dependent (it blew up on CI's OpenBLAS while passing on MKL).  The grating
        # is vertical there, so defer to the EXACT vertical Jones solver (handles
        # in-plane AND out-of-plane), which is well-conditioned and deterministic.
        # stabilize=True (NOT the slanted call's flag): the vertical Jones solver
        # has LAPACK-build-dependent resonances at a FIXED degree, so its robust
        # degree-scan is needed here -- and it matches the reduction tests'
        # pmm_jones_1d reference (which uses the default stabilize=True), making the
        # slant=0 result byte-deterministic across BLAS builds.
        return pmm_jones_1d(period, eps_ridge, eps_groove, n_substrate,
                            n_superstrate, depth, duty_cycle, wavelength,
                            angle=float(angle), degree=int(degree),
                            elements_per_region=int(elements_per_region),
                            grade=bool(grade),
                            far_field_orders=int(far_field_orders),
                            stabilize=True)
    if factorization == "covariant":
        cargs = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
                 duty_cycle, wavelength, float(slant_angle))
        ckw = dict(n_ridge_el=int(elements_per_region),
                   n_groove_el=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_oblique_solve(*cargs, degree=int(degree),
                                                     **ckw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_oblique_solve(*cargs, degree=d, **ckw)[:4],
            int(degree), "pmm_jones_1d_slanted")

    # ---- THE DIAGONAL CURE (round 16, Granet 2017/2023; Liu 2015) -----------
    # For a DIAGONAL in-plane tensor (exy = eyx = 0) WITH exx == ezz BOTH
    # regions, the TE / TM channels decouple and each maps onto the scalar slant
    # operator _sem_modes_slant -- which is DIV-CONFORMING (the Li 1/eps inverse
    # rule sits INSIDE the z-stiffness) and so SPURIOUS-MODE-FREE.  Route TE
    # through the scalar slant with n=sqrt(eyy) and TM with n=sqrt(exx), then
    # assemble the diagonal Jones.  This is the MORE-ACCURATE path: it sheds the
    # latent ~2e-4 per-order Liu-2015 spurious-harmonic-mean accuracy gap that
    # the pointwise metric-generator Ez-elimination (_build_metric_generator,
    # O_inv_ezz = iS0 @ [[1/ezz]]) carries (energy still conserves to ~1e-12).
    # Coupled tensors (exy/eyx != 0) AND diagonal tensors with exx != ezz fall
    # through to the metric generator (now div-conforming at all slants, so the
    # latent gap is gone there too).
    #
    # COMBINED OBLIQUE + SLANT is the ONE case the diagonal cure must NOT take:
    # the scalar slant operator's per-order split is wrong for oblique + nonzero
    # slant (energy still conserves -- a lossless cell auto-balances total power --
    # which is exactly why :func:`pmm_efficiency_1d_slanted` forbids that combo).
    # The round-19 div-conforming METRIC GENERATOR is the validated oblique+slant
    # path (per-order matches an RCWA staircase to ~2e-3, degree-clean), so route
    # combined oblique+slant through it even for a diagonal cell.  Oblique on a
    # VERTICAL grating (slant=0) and slant at NORMAL incidence are both still
    # handled correctly by the (more-accurate) scalar cure.
    inplane_off = max(abs(er[0, 1]), abs(er[1, 0]),
                      abs(eg[0, 1]), abs(eg[1, 0]))
    exx_eq_ezz = max(abs(er[0, 0] - er[2, 2]), abs(eg[0, 0] - eg[2, 2]))
    combined_oblique_slant = (abs(float(angle)) > 1e-12
                              and abs(float(slant_angle)) > 1e-12)
    diagonal_cure = (inplane_off <= 1e-9 * scale
                     and exx_eq_ezz <= 1e-9 * scale
                     and off <= 1e-9 * scale          # out-of-plane -> metric gen
                     and not combined_oblique_slant)
    if diagonal_cure:
        dargs = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
                 duty_cycle, wavelength, float(slant_angle))
        dkw = dict(elements_per_region=int(elements_per_region),
                   grade=bool(grade), far_field_orders=int(far_field_orders),
                   angle=float(angle))
        if not stabilize:
            return _pmm_jones_slant_diag_solve(*dargs, degree=int(degree),
                                               **dkw)
        return _stabilize_jones(
            lambda d: _pmm_jones_slant_diag_solve(*dargs, degree=d, **dkw),
            int(degree), "pmm_jones_1d_slanted")

    args = (period, er, eg, _C(n_substrate), _C(n_superstrate), depth,
            duty_cycle, wavelength, float(slant_angle))
    kw = dict(n_ridge_el=int(elements_per_region),
              n_groove_el=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_slant_solve(*args, degree=int(degree), **kw)
        return o, R, T, J
    return _stabilize_jones(
        lambda d: _pmm_jones_slant_solve(*args, degree=d, **kw)[:4],
        int(degree), "pmm_jones_1d_slanted")



def pmm_jones_1d_slanted_segments(
    period: float,
    segments,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    slant_angle: float,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
    factorization: str = "auto",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SLANTED multi-region grating with full ``(3, 3)`` permittivity tensors
    (IN-PLANE OR OUT-OF-PLANE) -- the multi-region generalization of
    :func:`pmm_jones_1d_slanted` and the slanted counterpart of
    :func:`pmm_jones_1d_segments`.  The default ``factorization='auto'`` picks the
    spectral covariant path for ANY slanted cell -- in-plane OR out-of-plane (the
    full-3x3 coupling enters via the Li Eq.12 ezz-Schur composites + cos*Dop cross
    blocks) -- and the convection path otherwise (vertical).
    ``factorization='covariant'`` carries out-of-plane too; pass ``'convection'``
    to force the fully-general algebraic path.

    Each region carries its own (possibly anisotropic) tensor, and the straight
    side-walls are tilted by ``slant_angle`` from the vertical.  Solved by the
    same div-conforming convection-metric ``[E;H]`` generator as the binary
    :func:`pmm_jones_1d_slanted` -- the metric generator and the lab-frame far
    field are region-count-agnostic, so the N-region cell uses the identical
    (validated) operator + half-space machinery on an N-segment nodal grid.
    Energy conserves to ~1e-13 across ``0-60`` deg for asymmetric multi-region
    cells (including coupled / gyrotropic regions); combined oblique + slant is
    supported (the metric generator handles it).

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_jones_1d`.
    segments : list of (width_fraction, eps)
        Consecutive regions along ``x``; each ``eps`` is a scalar (isotropic
        region) or a ``(3, 3)`` IN-PLANE permittivity tensor.  Width fractions
        must sum to 1 (within ``1e-6``).
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` reduces to
        :func:`pmm_jones_1d_segments`.  Validated ``0 <= slant_angle <= ~60 deg``.
    angle, degree, elements_per_region, grade, far_field_orders, stabilize,
    factorization : as in :func:`pmm_jones_1d_slanted`.  ``factorization=
        'covariant'`` gives SPECTRAL TM convergence for the multi-region cell too
        (in-plane OR out-of-plane); it pre-reverses the region order so the
        covariant far field lands in the user's input frame.

    Returns
    -------
    orders, R_eff, T_eff, jones_reflection : as in :func:`pmm_jones_1d_slanted`.

    Notes
    -----
    BINARY-cell reductions are bit-identical to :func:`pmm_jones_1d_slanted` and
    the ``slant=0`` limit reduces to :func:`pmm_jones_1d_segments` (to the
    div-conforming discretization difference).  Full ``(3, 3)`` tensor IN-PLANE
    OR OUT-OF-PLANE (out-of-plane + slant reaches the ~1e-4 wall-normal floor,
    validated vs a multi-region RCWA tensor z-staircase); NumPy/SciPy (not JAX).
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    if int(degree) < 2:
        raise ValueError("pmm_jones_1d_slanted_segments: degree must be >= 2.")
    if len(segments) < 1:
        raise ValueError(
            "pmm_jones_1d_slanted_segments: need at least one segment.")
    widths = [float(w) for w, _ in segments]
    tensors = []
    for _w, eps in segments:
        M = np.asarray(eps, dtype=_C)
        if M.ndim == 0:                         # scalar -> isotropic tensor
            M = M * np.eye(3, dtype=_C)
        if M.shape[-2:] != (3, 3):
            raise ValueError(
                "pmm_jones_1d_slanted_segments: each segment eps must be a "
                "scalar or a (3, 3) permittivity tensor.")
        tensors.append(M)
    # ---- COVARIANT OBLIQUE-COORDINATE path (SPECTRAL slant, opt-in) ---------
    # Multi-region generalization of the binary covariant path; spectral TM
    # convergence vs the convection path's algebraic floor (same answer).
    # IN-PLANE OR OUT-OF-PLANE (the full-3x3 coupling enters via the Li Eq.12
    # ezz-Schur composites + the cos*Dop single-derivative cross blocks).
    if factorization not in ("auto", "convection", "covariant"):
        raise ValueError(
            "pmm_jones_1d_slanted_segments: factorization must be 'auto', "
            f"'convection' or 'covariant', got {factorization!r}.")
    if factorization == "auto":
        factorization = ("covariant"
                         if abs(float(slant_angle)) >= _COV_MIN_SLANT_RAD
                         else "convection")
    if (factorization == "covariant"
            and abs(float(slant_angle)) < _COV_MIN_SLANT_RAD):
        # The covariant oblique frame degenerates at ~zero slant (the isotropic
        # half-spaces' TE/TM modes go exactly degenerate -> near-singular interface,
        # BLAS-build-dependent).  Defer to the EXACT vertical Jones segments solver,
        # which is well-conditioned and deterministic.
        # stabilize=True: the vertical solver's robust degree-scan handles its
        # fixed-degree LAPACK resonances and matches the reduction reference.
        return pmm_jones_1d_segments(
            period, segments, n_substrate, n_superstrate, depth, wavelength,
            angle=float(angle), degree=int(degree),
            elements_per_region=int(elements_per_region), grade=bool(grade),
            far_field_orders=int(far_field_orders), stabilize=True)
    if factorization == "covariant":
        ca = (period, widths, tensors, _C(n_substrate), _C(n_superstrate), depth,
              wavelength, float(slant_angle))
        ckw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
                   far_field_orders=int(far_field_orders), angle=float(angle))
        if not stabilize:
            o, R, T, J, _ = _pmm_jones_oblique_segments_solve(
                *ca, degree=int(degree), **ckw)
            return o, R, T, J
        return _stabilize_jones(
            lambda d: _pmm_jones_oblique_segments_solve(*ca, degree=d, **ckw)[:4],
            int(degree), "pmm_jones_1d_slanted_segments")
    # OUT-OF-PLANE (eps_xz/yz/zx/zy != 0): SUPPORTED (2026-06-07).  Multi-region
    # out-of-plane + slant rides the SAME exact-convection slant treatment as the
    # binary path (see _build_generator_metric), so it reaches the ~1e-4 wall-
    # normal per-order floor (validated vs a multi-region RCWA tensor z-staircase
    # oracle: 3-region dT ~3e-4 at slant 30; 2-segment == binary to ~1e-14; energy
    # conserves).  Segments route straight through the coupled metric generator
    # (no scalar diagonal cure here), so no out-of-plane dispatch guard is needed.
    sa = (period, widths, tensors, _C(n_substrate), _C(n_superstrate), depth,
          wavelength, float(slant_angle))
    kw = dict(n_el_per_region=int(elements_per_region), grade=bool(grade),
              far_field_orders=int(far_field_orders), angle=float(angle))

    if not stabilize:
        o, R, T, J, _ = _pmm_jones_slant_segments_solve(
            *sa, degree=int(degree), **kw)
        return o, R, T, J
    return _stabilize_jones(
        lambda d: _pmm_jones_slant_segments_solve(*sa, degree=d, **kw)[:4],
        int(degree), "pmm_jones_1d_slanted_segments")



def pmm_1d(
    period: float,
    n_substrate: complex,
    n_superstrate: complex,
    depth: float,
    wavelength: float,
    *,
    eps_ridge=None,
    eps_groove=None,
    duty_cycle: float = 0.5,
    segments=None,
    slant_angle: float = 0.0,
    angle: float = 0.0,
    theta: float | None = None,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unified 1-D anisotropic-Jones PMM dispatcher -- one entry point that
    auto-routes to the right solver by geometry.

    Covers the whole 1-D Jones family in a single call:

    ====================  =====================  =====================================
    geometry              ``slant_angle == 0``   ``slant_angle != 0``
    ====================  =====================  =====================================
    BINARY                :func:`pmm_jones_1d`    :func:`pmm_jones_1d_slanted`
    (``eps_ridge`` +      (vertical)              (slanted)
    ``eps_groove`` +
    ``duty_cycle``)
    MULTI-REGION          :func:`pmm_jones_1d_    :func:`pmm_jones_1d_slanted_segments`
    (``segments``)        segments`
    ====================  =====================  =====================================

    Each ``eps`` may be a scalar (promoted to an isotropic tensor) or a full
    ``(3, 3)`` permittivity tensor (IN-PLANE or, for the VERTICAL cases,
    OUT-OF-PLANE ``eps_xz/eyz/ezx/ezy != 0``).  Normal or oblique incidence.
    (Combined out-of-plane + slant is SUPPORTED by the binary and segments slanted
    solvers as of 2026-06-07 -- the slant is carried as exact convection, reaching
    the ~1e-4 wall-normal per-order floor.)

    Parameters
    ----------
    period, n_substrate, n_superstrate, depth, wavelength : as in
        :func:`pmm_jones_1d`.
    eps_ridge, eps_groove, duty_cycle : the BINARY grating spec (give these OR
        ``segments``, not both).
    segments : list of ``(width_fraction, eps)`` -- the MULTI-REGION spec.
    slant_angle : float
        Side-wall tilt from the vertical (radians); ``0`` = vertical.
    angle, degree, elements_per_region, grade, far_field_orders, stabilize : as in
        :func:`pmm_jones_1d`.

    Returns
    -------
    orders, R_eff, T_eff, jones_reflection : as in :func:`pmm_jones_1d`.
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    has_binary = eps_ridge is not None or eps_groove is not None
    if (segments is None) == (not has_binary):
        raise ValueError(
            "pmm_1d: give EITHER `segments` OR `eps_ridge` + `eps_groove` "
            "(+ `duty_cycle`), not both / neither.")
    common = dict(angle=float(angle), degree=int(degree),
                  elements_per_region=int(elements_per_region),
                  grade=bool(grade), far_field_orders=int(far_field_orders),
                  stabilize=bool(stabilize))
    slanted = abs(float(slant_angle)) > 1e-12
    if segments is not None:
        if slanted:
            return pmm_jones_1d_slanted_segments(
                period, segments, n_substrate, n_superstrate, depth, wavelength,
                float(slant_angle), **common)
        return pmm_jones_1d_segments(
            period, segments, n_substrate, n_superstrate, depth, wavelength,
            **common)
    if eps_ridge is None or eps_groove is None:
        raise ValueError(
            "pmm_1d: the binary spec needs BOTH `eps_ridge` and `eps_groove`.")
    # Honour the documented "scalar promoted to an isotropic tensor" contract
    # (the underlying Jones solvers require an explicit (3, 3) tensor).
    eps_ridge = _promote_eps_tensor(eps_ridge)
    eps_groove = _promote_eps_tensor(eps_groove)
    if slanted:
        return pmm_jones_1d_slanted(
            period, eps_ridge, eps_groove, n_substrate, n_superstrate, depth,
            float(duty_cycle), wavelength, float(slant_angle), **common)
    return pmm_jones_1d(
        period, eps_ridge, eps_groove, n_substrate, n_superstrate, depth,
        float(duty_cycle), wavelength, **common)



# ===========================================================================
# Convergence-class predictor for right-angle grating edges (Li-Granet 2011)
# ===========================================================================
def grating_convergence_class(eps_quadrants):
    """Classify modal-method convergence at a right-angle grating edge (Li &
    Granet, JOSA A 28:738, 2011).

    Predicts whether -- and how fast -- a modal method (FMM/RCWA, AMM, or PMM)
    converges at the right-angle corner where four regions of permittivity
    meet.  The governing field singularity is in the IN-PLANE electric field
    (the TM problem); TE (E parallel to the edge) is typically far better
    behaved.  This is a PURE O(1) diagnostic -- it does not solve the grating.

    Parameters
    ----------
    eps_quadrants : sequence of 4 complex
        The four permittivities meeting at the corner, IN ORDER AROUND THE
        VERTEX ``(eps1, eps2, eps3, eps4)``; ``eps1``/``eps3`` and
        ``eps2``/``eps4`` are the diagonal pairs.  PUBLIC convention
        ``Im(eps) > 0`` for loss.

    Returns
    -------
    dict
        ``type`` ('I' | 'II' | 'III' | 'degenerate'), ``tau`` (complex
        singularity exponent, Eq. 2), ``delta`` (Eq. 4), ``delta_prime``
        (Eq. 3), ``predicted_rate`` (``Re[tau]``, the algebraic decay exponent
        for Type I; else NaN), ``converges`` (bool), ``warning`` (str).

    Classification (lossless edges)
    -------------------------------
    * **Type I**  ``0 < Delta < 1`` (all-dielectric): REGULAR singularity;
      modal methods converge ALGEBRAICALLY at rate ``Re[tau]`` (slow as
      ``Re[tau] -> 0``; ``elements_per_region>1, grade=True`` recovers it).
    * **Type II** ``Delta < 0`` (LOSSLESS metal-dielectric): IRREGULAR
      singularity; NO method (FMM/AMM/PMM) converges.  Mitigate with metal
      loss, a rounded corner, or accept non-convergence.
    * **Type III** ``Delta > 1`` (requires a metal quadrant): no singularity,
      fast.  IMPOSSIBLE for an all-dielectric corner -- the squared numerator
      over a positive denominator forces ``Delta <= 1``.

    Loss caveat
    -----------
    A lossless metal corner is Type II; absorption (``Im(eps) > 0``) lifts the
    irregularity ASYMPTOTICALLY and is reported as (slowly) convergent.  This is
    NOT a hard switch -- a weakly lossy metal corner can still stall like
    Type II at practical truncation, so read ``converges=True`` for a
    near-lossless metal as "convergent in the limit," not "fast".

    References
    ----------
    L. Li & G. Granet, "Field singularities at lossless metal-dielectric
    right-angle edges and their ramifications to the numerical modeling of
    gratings," J. Opt. Soc. Am. A 28, 738-746 (2011).
    """
    e1, e2, e3, e4 = (_C(v) for v in eps_quadrants)
    loss_tol = 1e-9
    degen_tol = 1e-12

    has_loss = max(abs(e.imag) for e in (e1, e2, e3, e4)) > loss_tol

    f12, f23, f34, f41 = e1 + e2, e2 + e3, e3 + e4, e4 + e1
    num = (e1 * e3 - e2 * e4) ** 2
    den = f12 * f23 * f34 * f41
    scale = (abs(e1) + abs(e2) + abs(e3) + abs(e4)) ** 4 + 1e-30
    if abs(den) < degen_tol * scale or not np.isfinite(num / den):
        return dict(
            type="degenerate", tau=complex("nan"), delta=complex("nan"),
            delta_prime=complex("nan"), predicted_rate=float("nan"),
            converges=False,
            warning=("DEGENERATE EDGE: a (eps_i+eps_j) denominator factor "
                     "vanished (impedance-matched / resonant corner); "
                     "Delta_prime is ill-defined."))

    delta_prime = num / den
    delta = 1.0 - delta_prime
    tau = (2.0 / np.pi) * cmath.asin(cmath.sqrt(complex(delta_prime)))
    dr = delta.real

    if has_loss:
        # Loss lifts the lossless-metal irregularity (asymptotically).
        ctype = "I" if dr < 1.0 else "III"
        warning = ""
        if dr < 0.0:
            ctype = "I"
            warning = (
                "Lossy metal corner: the lossless Type-II irregularity is "
                "lifted by absorption, but a WEAKLY lossy metal can still "
                "stall at practical truncation; convergence may be SLOW (use "
                "mesh grading toward the wall).")
        return dict(type=ctype, tau=tau, delta=delta, delta_prime=delta_prime,
                    predicted_rate=float(abs(tau.real)), converges=True,
                    warning=warning)

    if dr < 0.0:
        return dict(
            type="II", tau=tau, delta=delta, delta_prime=delta_prime,
            predicted_rate=float("nan"), converges=False,
            warning=("TYPE II IRREGULAR SINGULARITY (lossless metal-dielectric "
                     "edge): NO modal method (FMM/AMM/PMM) converges. Add metal "
                     "loss, round the corner, or accept non-convergence -- do "
                     "NOT trust efficiencies from this edge."))
    if dr > 1.0:
        return dict(type="III", tau=tau, delta=delta, delta_prime=delta_prime,
                    predicted_rate=float("nan"), converges=True, warning="")

    rate = tau.real
    warning = ""
    if rate < 0.3:
        warning = (
            f"WEAK Type-I singularity (Re[tau]={rate:.4f} < 0.3): algebraic "
            "convergence is SLOW (~N^-tau); use elements_per_region>1, "
            "grade=True to recover the rate.")
    return dict(type="I", tau=tau, delta=delta, delta_prime=delta_prime,
                predicted_rate=float(rate), converges=True, warning=warning)



def classify_from_grating(eps_superstrate, eps_ridge, eps_groove,
                          eps_substrate):
    """Convenience wrapper of :func:`grating_convergence_class` for a 1-D binary
    grating: maps the four regions around the ridge/groove corner to the vertex
    ordering ``(sup, ridge, sub, groove)`` so the diagonal pairs are
    ``(sup, sub)`` and ``(ridge, groove)`` (Li & Granet Fig. 1)."""
    return grating_convergence_class(
        (eps_superstrate, eps_ridge, eps_substrate, eps_groove))


__all__ = [
    "pmm_jones_1d",
    "pmm_efficiency_1d",
    "pmm_efficiency_1d_jax",
    "pmm_efficiency_1d_segments",
    "pmm_jones_1d_segments",
    "pmm_efficiency_1d_slanted",
    "pmm_jones_1d_slanted",
    "pmm_jones_1d_slanted_segments",
    "pmm_1d",
    "pmm_efficiency_1d_vs_wavelength",
    "pmm_jones_1d_vs_wavelength",
    "pmm_graded_segments",
    "grating_convergence_class",
    "classify_from_grating",
]

def pmm_efficiency_1d_vs_wavelength(
    period: float,
    n_ridge,
    n_groove,
    n_substrate,
    n_superstrate,
    depth: float,
    duty_cycle: float,
    wavelengths,
    *,
    polarization: str = "te",
    angle: float = 0.0,
    theta: float | None = None,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
):
    """DISPERSIVE scalar spectral sweep of the binary PMM grating -- the PMM
    counterpart of :func:`rcwa_efficiency_vs_wavelength` (v5.14 generality
    audit: the PMM family previously had no dispersive-material sweep).

    Each of ``n_ridge``, ``n_groove``, ``n_substrate``, ``n_superstrate`` may
    be a FIXED value or a CALLABLE ``wl -> value`` (material dispersion via
    ``n(lambda)`` closures, e.g. from the bundled refractiveindex database).

    Returns
    -------
    wavelengths : ndarray
        The sweep grid (scalar in -> scalar out).
    R_total, T_total : (Nwl,) float ndarray
        Total reflected / transmitted efficiency (summed over orders) at each
        wavelength.
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wl.size == 0 or not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError(
            "pmm_efficiency_1d_vs_wavelength: every wavelength must be "
            "finite and > 0 [m] (got an empty or invalid sweep).")

    def _at(v, w):
        return v(w) if callable(v) else v

    Rt = np.empty(wl.size, dtype=float)
    Tt = np.empty(wl.size, dtype=float)
    for i, w in enumerate(wl):
        _o, R, T = pmm_efficiency_1d(
            period, _at(n_ridge, w), _at(n_groove, w), _at(n_substrate, w),
            _at(n_superstrate, w), depth, duty_cycle, float(w),
            polarization=polarization, angle=angle, degree=degree,
            elements_per_region=elements_per_region, grade=grade,
            far_field_orders=far_field_orders, stabilize=stabilize)
        Rt[i] = float(np.asarray(R).sum())
        Tt[i] = float(np.asarray(T).sum())
    if np.ndim(wavelengths):
        return wl, Rt, Tt
    return wl[0], Rt[0], Tt[0]


def pmm_jones_1d_vs_wavelength(
    period: float,
    eps_ridge,
    eps_groove,
    n_substrate,
    n_superstrate,
    depth: float,
    duty_cycle: float,
    wavelengths,
    *,
    angle: float = 0.0,
    theta: float | None = None,
    degree: int = 16,
    elements_per_region: int = 1,
    grade: bool = True,
    far_field_orders: int = 21,
    n_orders: int | None = None,
    stabilize: bool = True,
):
    """DISPERSIVE Jones spectral sweep of the 1-D anisotropic PMM grating --
    the PMM mirror of :func:`rcwa_jones_vs_wavelength` (same signature shape
    and returns).  Each of ``eps_ridge``, ``eps_groove``, ``n_substrate``,
    ``n_superstrate`` may be a fixed value or a ``wl -> value`` callable.

    Returns
    -------
    wavelengths : ndarray
        The sweep grid (scalar in -> scalar out).
    jones : (Nwl, 2, 2) complex ndarray
        Zeroth-order Jones reflection at each wavelength (PUBLIC convention).
    R_total, T_total : (Nwl, 2) float ndarray
        Total efficiencies per incident polarization (row order: incident
        ``E_x``, ``E_y``).
    """
    angle = _resolve_incidence(angle, theta)
    far_field_orders = _resolve_order_count(far_field_orders, n_orders)
    wl = np.atleast_1d(np.asarray(wavelengths, dtype=float))
    if wl.size == 0 or not np.all(np.isfinite(wl)) or np.any(wl <= 0.0):
        raise ValueError(
            "pmm_jones_1d_vs_wavelength: every wavelength must be finite and "
            "> 0 [m] (got an empty or invalid sweep).")

    def _at(v, w):
        return v(w) if callable(v) else v

    J = np.empty((wl.size, 2, 2), dtype=complex)
    Rt = np.empty((wl.size, 2), dtype=float)
    Tt = np.empty((wl.size, 2), dtype=float)
    for i, w in enumerate(wl):
        _o, R, T, jr = pmm_jones_1d(
            period, _at(eps_ridge, w), _at(eps_groove, w),
            _at(n_substrate, w), _at(n_superstrate, w), depth, duty_cycle,
            float(w), angle=angle, degree=degree,
            elements_per_region=elements_per_region, grade=grade,
            far_field_orders=far_field_orders, stabilize=stabilize)
        J[i] = np.asarray(jr)
        Rt[i] = np.asarray(R).sum(axis=1)
        Tt[i] = np.asarray(T).sum(axis=1)
    if np.ndim(wavelengths):
        return wl, J, Rt, Tt
    return wl[0], J[0], Rt[0], Tt[0]

def pmm_graded_segments(profile, n_segments=16):
    """Approximate a CONTINUOUS lateral permittivity profile by piecewise-
    constant segments for the PMM segment solvers (v5.14 roadmap item 4).

    ``profile`` is a callable ``u -> eps`` (or ``u -> (3, 3) tensor``) over the
    normalized period coordinate ``u in [0, 1)``; each of the ``n_segments``
    equal-width segments takes the MIDPOINT value (O(1/n^2) profile error).
    The audit measured a 16-segment approximation of a sinusoidal profile
    converging to the same-profile RCWA reference; double ``n_segments`` to
    quarter the staircase error.

    Returns the ``segments`` list for :func:`pmm_efficiency_1d_segments` /
    :func:`pmm_jones_1d_segments` / ``PMMStack.add_layer``.
    """
    n_segments = int(n_segments)
    if n_segments < 2:
        raise ValueError("pmm_graded_segments: n_segments must be >= 2")
    w = 1.0 / n_segments
    return [(w, profile((i + 0.5) * w)) for i in range(n_segments)]

