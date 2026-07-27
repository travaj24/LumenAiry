"""Berreman 4x4 anisotropic planar multilayer solver.

The fast, EXACT tool for laterally-UNIFORM stacks whose layers may be fully
anisotropic (birefringent / uniaxial / biaxial / gyrotropic) -- liquid-crystal
retarders, waveplates, multilayer dichroics, magneto-optic films.  This is the
anisotropic generalization of the scalar transfer-matrix coating model
(:func:`~lumenairy.elements.coatings.coating_reflectance`): a scalar isotropic
stack reduces to it EXACTLY, while a tensor stack carries the full ``2x2`` Jones
response that the scalar s/p model cannot.

It is NOT a competitor to RCWA / PMM -- those solve laterally PATTERNED
(grating / metasurface) structures, where a transverse Fourier / nodal basis and
a per-layer eigenproblem are unavoidable.  Berreman is the planar limit: a
``4x4`` state ``[Ex, Ey, Hx, Hy]`` per layer (Berreman 1972 / Yeh 1979), so the
whole solve is microseconds -- ideal for unpatterned retarder / coating design
and as the effective-medium screen that FEEDS the patterned solvers.

Method.  In a planar layer the tangential field state obeys ``dpsi/dz' =
Delta psi`` (``z' = k0 z``) with the ``4x4`` Berreman matrix ``Delta(eps, Kx,
Ky)``; ``Ez, Hz`` are eliminated from the normalized Maxwell curl equations.
Eigendecomposition gives 2 forward + 2 backward modes; the layers are composed
by the SAME numerically-stable generalized scattering-matrix cascade the PMM /
RCWA out-of-plane paths use (forward referenced to the layer top, backward to
its bottom, so every exponential decays -- no transfer-matrix overflow through
a thick / lossy layer).

Conventions are PUBLIC throughout (``exp(-i w t)``, ``Im(eps) > 0`` for loss):
the permittivity enters raw (no conjugation), the forward / backward mode split
is by the physical decay / energy direction so the flux-based ``R`` / ``T`` are
PHYSICAL (validated against the complex-angle scalar TMM on lossy stacks), and
the returned Jones matches :func:`~lumenairy.rcwa_jones_1d` to machine precision
on a uniform-tensor slab.  (The standalone single-slab Berreman ORACLE in the
test suite conjugates ``eps`` in and the Jones out -- an equivalent pair for the
Jones, but it never validated loss, where only the raw-``eps`` split gives the
physically-correct power split.)
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import List, Optional, Tuple, Union

import numpy as np

from .rcwa import (
    _interface_smatrix_general,
    _modes_to_M,
    _propagation_smatrix_general,
    _redheffer_star,
)
from .rcwa._core import (
    _forward_flux_kz,
    _homogeneous_eigenmodes,
    _layer_eigenmodes_tensor,
    _propagation_star_general,
    _sqrt_decay,
    _sqrt_forward,
)

_C = np.complex128

__all__ = ["berreman_jones_1d", "BerremanStack"]

# --- per-layer modal eig cache ------------------------------------------------
# The Berreman Delta -- and therefore the 4x4 layer eig -- depends ONLY on eps
# and Kx/Ky (angle), NOT on wavelength (the wavelength enters solely through the
# propagation phase k0*thickness).  So a fixed-angle wavelength sweep and a
# periodic ABAB... (DBR / Bragg) stack recompute the SAME eig over and over.  A
# module-level bounded LRU keyed on (eps bytes, Kx, Ky) returns byte-identical
# modes -- the PMMStack._eig_cache / BORStack._modal_cache precedent.  The cached
# mode arrays are treated read-only by every caller (only fresh matrices are
# built from them).
_MODE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_MODE_CACHE_SIZE = 256
_MODE_CACHE_LOCK = threading.Lock()

# The interface S-matrices are ALSO wavelength-independent (built from the
# wl-independent field-mode matrices M; only the propagation phase carries k0),
# so a fixed-angle sweep rebuilds the SAME _interface_smatrix_general every point.
# A second bounded LRU keyed on the two M matrices' bytes returns the byte-
# identical 4-tuple (read-only downstream) -- ~1.2-1.3x on top of the eig cache.
_IFACE_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_IFACE_CACHE_SIZE = 512
_IFACE_CACHE_LOCK = threading.Lock()


def _clear_berreman_mode_cache():
    """Registry hook: drop the per-layer modal eig + interface caches."""
    with _MODE_CACHE_LOCK:
        _MODE_CACHE.clear()
    with _IFACE_CACHE_LOCK:
        _IFACE_CACHE.clear()


try:
    from .._cache_registry import (
        register_cache_clearer as _register_cache_clearer,
    )
    _register_cache_clearer("berreman_layer_modes", _clear_berreman_mode_cache)
except ImportError:  # pragma: no cover - registry always present in-tree
    pass


# =========================================================================== #
# core operators
# =========================================================================== #

def _as_eps_tensor(eps):
    """Scalar or ``(3,3)`` permittivity -> ``(3,3)`` complex tensor (PUBLIC
    convention, ``Im > 0`` for loss).

    W6 F-2: a NaN/inf permittivity is rejected HERE, at the single conversion
    funnel both entry points share, so it never reaches ``np.linalg.eig`` as a
    bare ``LinAlgError`` (nor warns its way through ``inf * eye``)."""
    M = np.asarray(eps, dtype=_C)
    if not np.all(np.isfinite(M)):
        raise ValueError(
            f"berreman: layer permittivity is not finite ({eps!r}); a NaN/inf "
            f"eps would propagate silently through the modal eigensolve.")
    if M.ndim == 0:
        return M * np.eye(3, dtype=_C)
    if M.shape != (3, 3):
        raise ValueError(
            "berreman: each eps must be a scalar or a (3, 3) tensor, got "
            f"shape {M.shape}.")
    return M


def _berreman_delta(eps, Kx, Ky):
    """Berreman ``4x4`` ``Delta`` for state ``[Ex, Ey, Hx, Hy]``,
    ``dpsi/dz' = Delta psi`` (``z' = k0 z``), eigenvalues ``q = kz/k0``.  Full
    ``3x3`` eps (INTERNAL convention), ``mu = 1``; ``Ez, Hz`` eliminated.  The
    conical ``Kx*Ky`` terms at ``[2,0]/[2,3]/[3,2]`` are the audit-F1 fix
    (rotation-covariance-pinned)."""
    exx, exy, exz = eps[0, 0], eps[0, 1], eps[0, 2]
    eyx, eyy, eyz = eps[1, 0], eps[1, 1], eps[1, 2]
    ezx, ezy, ezz = eps[2, 0], eps[2, 1], eps[2, 2]
    d = 1.0 / ezz
    D = np.zeros((4, 4), dtype=_C)
    # FACTOR-i FIX (loose-ends audit 2026-07-14, lockstep with
    # rcwa._core._layer_eigenmodes_tensor / _berreman_jax._delta_jax /
    # tests _berreman4x4): the off-plane cross entries (E <- E block
    # [0:2, 0:2], H <- H block [2:4, 2:4]) carry relative -/+i factors in
    # the modal-u state the in-plane entries are written in; the legacy
    # real coefficients gave a wrong +/- symmetric extraordinary dispersion
    # for out-of-plane tensors at oblique incidence (exact-dispersion gate:
    # tests/unit/test_audit_oop_dispersion.py).  Note this native path only
    # exercises these entries for OOP-at-NORMAL (where they vanish) -- the
    # OOP-oblique route uses the rcwa generator -- but the copies stay in
    # lockstep.
    D[0, 0] = -1j * Kx * ezx * d
    D[0, 1] = -1j * Kx * ezy * d
    D[0, 2] = Kx * Ky * d
    D[0, 3] = 1.0 - Kx * Kx * d
    D[1, 0] = -1j * Ky * ezx * d
    D[1, 1] = -1j * Ky * ezy * d
    D[1, 2] = Ky * Ky * d - 1.0
    D[1, 3] = -Ky * Kx * d
    D[2, 0] = eyx - eyz * ezx * d + Kx * Ky
    D[2, 1] = eyy - eyz * ezy * d - Kx * Kx
    D[2, 2] = -1j * eyz * Ky * d
    D[2, 3] = 1j * eyz * Kx * d
    D[3, 0] = exz * ezx * d - exx + Ky * Ky
    D[3, 1] = exz * ezy * d - exy - Kx * Ky
    D[3, 2] = 1j * exz * Ky * d
    D[3, 3] = -1j * exz * Kx * d
    return D


def _split_fwd_bwd(gam):
    """Indices of the 2 forward (+z) and 2 backward modes.  ``gam = kz/k0`` in
    the ``exp(gam k0 z)`` convention, so forward is ``Re(gam) < 0`` (decays as
    ``z -> +inf``) or, for a propagating mode, ``Im(gam) > 0``.

    S1-13 (audit AUDIT_V5_24_2): the partition is a STABLE argsort on the
    per-mode forward flag -- byte-for-byte the rule the JAX twin
    ``_berreman_jax._layer_modes_jax`` uses -- so numpy and JAX agree in
    EVERY case, not just the physical one.  For a physical (non-bianiso-
    tropic) stack exactly two modes flag forward, and the stable sort
    keeps them in ascending index order; that reproduces the pre-fix
    physical partition exactly.  The two implementations previously forked
    ONLY in the degenerate (not-exactly-two-forward) fallback -- numpy
    ranked by decay (``argsort(Re gam)``) while JAX kept the flag-then-
    index order -- a divergence unreachable for the physical media tested
    but latent for degenerate bianisotropic inputs.  Aligning on the JAX
    rule removes the fork."""
    g = np.asarray(gam)
    tol = 1e-9 * max(1.0, float(np.max(np.abs(g))))
    is_fwd = np.where(g.real < -tol, True,
                      np.where(g.real > tol, False, g.imag > 0.0))
    order = np.argsort(np.where(is_fwd, 0, 1), kind='stable')
    return order[:2], order[2:]


def _layer_modes(eps_tensor, Kx, Ky):
    """Forward / backward modal blocks of one Berreman layer.

    Returns ``(Wf, Vf, lamf, Wb, Vb, lamb)`` with ``W = [Ex; Ey]`` rows
    (``2x2``), ``V = [Hx; Hy]`` rows (``2x2``), and ``lam = -gam`` so the forward
    set has ``Re(lam) >= 0`` (decay ``exp(-lam k0 z)``) -- the convention the
    generalized scattering-matrix cascade expects."""
    D = _berreman_delta(eps_tensor, Kx, Ky)
    gam, Psi = np.linalg.eig(D)
    fwd, bwd = _split_fwd_bwd(gam)
    Wf, Vf = Psi[:2, fwd], Psi[2:, fwd]
    Wb, Vb = Psi[:2, bwd], Psi[2:, bwd]
    lamf, lamb = -gam[fwd], -gam[bwd]
    return Wf, Vf, lamf, Wb, Vb, lamb


def _freeze(res):
    """Mark every ndarray in a cached tuple NON-WRITEABLE before it is shared.

    W6 F-5 (2026-07-26).  Both LRUs hand out the STORED arrays, and the
    docstrings assert "the arrays are read-only downstream" -- but nothing
    enforced it, so a caller that wrote into a returned block silently poisoned
    every later solve (measured pre-fix:
    ``_layer_modes_cached(eps, Kx, Ky)[0][0, 0] = 999`` made the very next call
    return ``Wf[0, 0] = 999``).  No in-tree caller mutates them, so this closes a
    hardening gap rather than a live defect -- the ``pmm/_core.py`` grid cache
    precedent, which freezes its cached arrays the same way."""
    for a in res:
        if isinstance(a, np.ndarray):
            a.flags.writeable = False
    return res


def _layer_modes_cached(eps_tensor, Kx, Ky):
    """:func:`_layer_modes` memoized on ``(eps, Kx, Ky)``.  The eig is
    wavelength-independent, so a fixed-angle sweep or a periodic stack reuses it
    byte-for-byte (the returned arrays are frozen NON-WRITEABLE -- W6 F-5)."""
    a = np.ascontiguousarray(np.asarray(eps_tensor, dtype=_C))
    key = (a.tobytes(), a.shape, complex(Kx), complex(Ky))
    with _MODE_CACHE_LOCK:
        hit = _MODE_CACHE.get(key)
        if hit is not None:
            _MODE_CACHE.move_to_end(key)            # LRU: refresh recency
            return hit
    res = _freeze(_layer_modes(eps_tensor, Kx, Ky))
    with _MODE_CACHE_LOCK:
        _MODE_CACHE[key] = res
        while len(_MODE_CACHE) > _MODE_CACHE_SIZE:
            _MODE_CACHE.popitem(last=False)
    return res


def _interface_smatrix_cached(Ma, Mb):
    """:func:`_interface_smatrix_general` memoized on the two field-mode matrices
    ``(Ma, Mb)``.  Both are wavelength-independent (built from the cached modes),
    so a fixed-angle sweep reuses the byte-identical 4-tuple (frozen
    NON-WRITEABLE -- W6 F-5).

    The key carries each matrix's SHAPE alongside its bytes (W6 F-5b, matching
    :func:`_layer_modes_cached`): the Berreman block is always ``4x4`` so a
    same-bytes / different-shape collision is unreachable here, but keying on raw
    bytes alone is the latent form of the bug."""
    A = np.ascontiguousarray(Ma)
    Bm = np.ascontiguousarray(Mb)
    key = (A.tobytes(), A.shape, Bm.tobytes(), Bm.shape)
    with _IFACE_CACHE_LOCK:
        hit = _IFACE_CACHE.get(key)
        if hit is not None:
            _IFACE_CACHE.move_to_end(key)              # LRU: refresh recency
            return hit
    res = _freeze(_interface_smatrix_general(Ma, Mb))
    with _IFACE_CACHE_LOCK:
        _IFACE_CACHE[key] = res
        while len(_IFACE_CACHE) > _IFACE_CACHE_SIZE:
            _IFACE_CACHE.popitem(last=False)
    return res


def _checked_angle(fn_name, angle, theta):
    """``angle`` / ``theta`` alias resolution PLUS the shared back-side-incidence
    range guard (W6 F-2, 2026-07-26).

    Routes to the SAME ``_resolve_incidence_checked`` the rcwa 1-D and PMM entry
    points use, so ``|angle| >= pi/2`` is rejected identically across the whole
    solver family.  The angle enters this solve only through
    ``Kx/Ky ~ sin(angle)``, so a back-side angle previously aliased BYTE-
    IDENTICALLY onto the supplementary front-side angle ``pi - angle`` and
    returned a plausible, energy-conserving answer for the WRONG geometry
    (measured pre-fix: ``angle = 2.0`` rad gave R = [0.01771, 0.19647], the
    ``pi - 2.0 = 1.1416`` rad answer, and ``angle = 10.0`` rad was accepted too;
    ``angle = pi/2`` and ``angle = nan`` raised a bare ``LinAlgError`` with no
    context).  A TRACED JAX angle skips the guard (the resolver's tracer
    carve-out); ``theta`` still wins when both are given."""
    from .rcwa.oned import _resolve_incidence_checked
    return _resolve_incidence_checked(fn_name, angle, theta)


def _check_inputs(fn_name, eps_sup, eps_sub, eps_layers, thicks, wavelength,
                  Kx, Ky):
    """Geometry + propagating-incidence guards for a Berreman solve (W6 F-2).

    Sibling-identical to the RCWA / PMM entry points: :func:`_validate_geometry`
    for the wavelength, the per-layer thickness rule
    ``BerremanStack.add_layer`` already enforced (the FUNCTIONAL entry silently
    accepted ``t = 0`` and ``t < 0``), a finiteness screen on every material
    value (audit P3's NaN-index class), and
    :func:`_require_propagating_incidence` for an evanescent / metallic / GAIN
    incidence half-space.  Measured pre-fix: a metallic superstrate
    ``n_sup = 0.15+3.5j`` at ``theta = 0.5`` returned ``T = [30.78, 30.73]`` -- a
    3000% energy violation -- silently, where ``rcwa_jones_1d`` raises
    ``the incidence half-space is non-propagating``.  ``eps_sup`` is PUBLIC here;
    the guard takes the INTERNAL gauge, so it is conjugated on the way in."""
    from .rcwa._core import _require_propagating_incidence, _validate_geometry
    _validate_geometry(fn_name, wavelength=wavelength)
    for i, t in enumerate(thicks):
        tv = float(np.real(t))
        if not np.isfinite(tv) or tv <= 0.0:
            raise ValueError(
                f"{fn_name}: layer {i} thickness must be > 0, got {t!r} "
                f"(BerremanStack.add_layer enforces the same rule).")
    for name, val in (("eps_superstrate", eps_sup), ("eps_substrate", eps_sub)):
        if not np.all(np.isfinite(np.asarray(val, dtype=_C))):
            raise ValueError(
                f"{fn_name}: {name} is not finite ({val!r}); a NaN/inf "
                f"half-space index would propagate silently through the solve.")
    for i, e in enumerate(eps_layers):
        ea = np.asarray(e, dtype=_C)
        if not np.all(np.isfinite(ea)):
            raise ValueError(
                f"{fn_name}: layer {i} permittivity is not finite ({e!r}).")
        if abs(ea[2, 2]) == 0.0:
            raise ValueError(
                f"{fn_name}: layer {i} has eps_zz = 0; the Berreman Delta "
                f"eliminates Ez / Hz through 1/eps_zz and is undefined there.")
    kt2 = float(np.real(Kx)) ** 2 + float(np.real(Ky)) ** 2
    if not np.isfinite(kt2):
        raise ValueError(
            f"{fn_name}: the transverse wavevector is not finite (Kx = {Kx!r}, "
            f"Ky = {Ky!r}); check angle / phi / n_superstrate.")
    _require_propagating_incidence(fn_name, np.conj(_C(eps_sup)), kt2)


def _is_passive(eps_layers, eps_sup, eps_sub):
    """True when EVERY medium in the problem is passive (loss-or-lossless).

    Passivity of a (possibly non-Hermitian, possibly gyrotropic) permittivity is
    the positive-semidefiniteness of its anti-Hermitian part
    ``(eps - eps^H) / (2i)`` -- which is ``Im(eps)`` for a scalar, ZERO for a
    Hermitian gyrotropic tensor (correctly lossless), and negative-definite for
    a gain medium.  Diagonal-``Im`` screening would misclassify a rotated lossy
    tensor, so use the eigenvalues."""
    for e in list(eps_layers) + [eps_sup, eps_sub]:
        M = np.asarray(e, dtype=_C)
        if M.ndim == 0:
            M = M * np.eye(3, dtype=_C)
        tol = 1e-9 * max(1.0, float(np.max(np.abs(M))))
        anti = (M - M.conj().T) / 2.0j
        if float(np.min(np.linalg.eigvalsh(anti))) < -tol:
            return False
    return True


def _check_energy_pols(fn_name, R, T, passive=True):
    """Passive-energy tripwire on the ``(2,)`` per-lab-polarization ``R`` / ``T``
    (W6 F-2).  Reuses the family-wide :func:`_check_energy` with the arrays
    reshaped to ``(2, 1)`` so ``n_states = 2`` (two incident polarizations) and a
    lossless stack's exact total of ``2.0`` sits inside the shared 5% band.

    Berreman was the one solver in the family with NO tripwire: measured pre-fix
    it returned ``R + T = 1.086`` per pol for a lossy superstrate, and
    ``T = [30.78, 30.73]`` for a metallic one, silently, where
    ``rcwa_jones_1d`` raises.

    ACTIVE MEDIA -- DECLINED WITH MEASUREMENT (W6).  ``rcwa_jones_1d`` DOES trip
    on a gain layer (measured: the ``(1.5-0.05j)^2`` / 0.5 um slab raises
    ``_EnergyError`` with ``sum R+T = 3.253``), and berreman returns 3.253
    instead.  That divergence is left in place ON PURPOSE: berreman's gain
    behaviour is contract-locked by
    ``tests/unit/test_audit_v5_24_2_g01_conventions.py::
    test_s1_6_berreman_public_path_raw_eps_absorbs_with_positive_imag``, which
    asserts a gain slab returns ``R + T > 1`` as the sign-convention oracle for
    the raw-eps public cascade -- and that value is CORRECT (it matches this
    library's scalar TMM to 1e-15 at 2 um and 5 um; it is the exact formal
    steady state of an above-threshold cavity, not a numerical artefact).  So
    the tripwire is applied only when :func:`_is_passive` holds; a gain
    SUPERSTRATE is still rejected outright by
    :func:`_require_propagating_incidence`, matching rcwa."""
    from .rcwa._core import _check_energy, _EnergyError
    if passive:
        _check_energy(fn_name, np.asarray(R)[:, None], np.asarray(T)[:, None])
    elif not np.isfinite(float(np.sum(R) + np.sum(T))):
        raise _EnergyError(
            f"{fn_name}: non-finite total efficiency (sum R+T = "
            f"{float(np.sum(R) + np.sum(T))}); a NaN/inf material value reached "
            f"the solve.")


def _flux(Etan, Hmodal):
    """Time-averaged z-Poynting flux from tangential ``E`` and the MODAL ``H``
    (the Berreman state's lower block).  The modal H carries an extra ``-i``
    vs the physical field (the same convention the PMM / RCWA internal-field
    machinery uses), so the physical pair is ``H_phys = -i H_modal`` and
    ``Sz = Re(Ex conj(Hy_phys) - Ey conj(Hx_phys))`` (the common 1/2 cancels in
    every R / T ratio)."""
    Ex, Ey = Etan[0], Etan[1]
    Hx, Hy = -1j * Hmodal[0], -1j * Hmodal[1]
    return float(np.real(Ex * np.conj(Hy) - Ey * np.conj(Hx)))


# =========================================================================== #
# the cascade (shared by the functional entry and the class)
# =========================================================================== #

def _solve_core(eps_layers, thicks, eps_sup, eps_sub, wavelength, Kx, Ky,
                retain=False):
    """S-matrix cascade for a planar Berreman stack (PUBLIC convention).

    ``eps_layers`` are RAW public ``(3,3)`` tensors and ``eps_sup/eps_sub`` raw
    public scalars (``exp(-i w t)`` / ``Im(eps) > 0`` for loss -- NO conjugation,
    the module-level convention).  Both callers (``berreman_jones_1d`` and
    ``BerremanStack.solve``) pass the public eps straight in (verified
    numerically).  CAUTION: the sibling ``_offplane_oblique_solve`` uses the
    OPPOSITE (conjugated-internal) gauge -- the two internal paths do NOT share a
    convention, so do not assume conjugation state when refactoring across them.
    Returns a dict with the half-space forward / backward mode blocks,
    ``S11/S21``, and (when ``retain``) the per-layer modes + partial cascades for
    internal-field / absorption reconstruction."""
    k0 = 2.0 * np.pi / wavelength

    Wf_s, Vf_s, lamf_s, Wb_s, Vb_s, lamb_s = _layer_modes_cached(
        eps_sup * np.eye(3, dtype=_C), Kx, Ky)
    Wf_b, Vf_b, lamf_b, Wb_b, Vb_b, lamb_b = _layer_modes_cached(
        eps_sub * np.eye(3, dtype=_C), Kx, Ky)
    M_sup = _modes_to_M(Wf_s, Vf_s, Wb_s, Vb_s)
    M_sub = _modes_to_M(Wf_b, Vf_b, Wb_b, Vb_b)

    modes = [_layer_modes_cached(e, Kx, Ky) for e in eps_layers]
    Ms = [_modes_to_M(Wf, Vf, Wb, Vb) for (Wf, Vf, _lf, Wb, Vb, _lb) in modes]
    nlay = len(modes)

    # full cascade sup -> [iface, prop]*layers -> sub
    if nlay == 0:
        S = _interface_smatrix_cached(M_sup, M_sub)
    else:
        S = _interface_smatrix_cached(M_sup, Ms[0])
        for i in range(nlay):
            lamf, lamb = modes[i][2], modes[i][5]
            S = _redheffer_star(
                S, _propagation_smatrix_general(lamf, lamb, k0 * thicks[i]))
            nxt = M_sub if i == nlay - 1 else Ms[i + 1]
            S = _redheffer_star(S, _interface_smatrix_cached(Ms[i], nxt))
    S11, _S12, S21, _S22 = S

    out = dict(M_sup=M_sup, M_sub=M_sub, S11=S11, S21=S21,
               Wf_s=Wf_s, Vf_s=Vf_s, Wb_s=Wb_s, Vb_s=Vb_s,
               Wf_b=Wf_b, Vf_b=Vf_b, k0=k0)
    if retain and nlay:
        # Three bracketing partial cascades for the in-structure field (the
        # general RCWAResult._internal_cpm recovery, valid for the asymmetric
        # forward/backward Berreman modes):
        #   S_above[i]     : sup -> TOP of layer i (before its propagation)
        #   S_below[i]     : TOP of layer i -> sub (its OWN propagation first)
        #   S_below_bot[i] : BOTTOM of layer i -> sub (propagation removed)
        # The backward amplitude references the layer BOTTOM, so every modal
        # exponential decays (no overflow through a deep / lossy layer).
        ifc = [_interface_smatrix_cached(M_sup, Ms[0])]
        for i in range(1, nlay):
            ifc.append(_interface_smatrix_cached(Ms[i - 1], Ms[i]))
        ifc.append(_interface_smatrix_cached(Ms[-1], M_sub))
        prop = [_propagation_smatrix_general(modes[i][2], modes[i][5],
                                             k0 * thicks[i])
                for i in range(nlay)]
        S_above = [None] * nlay
        S_above[0] = ifc[0]
        for i in range(1, nlay):
            S_above[i] = _redheffer_star(
                _redheffer_star(S_above[i - 1], prop[i - 1]), ifc[i])
        S_below = [None] * nlay
        S_below_bot = [None] * nlay
        for i in range(nlay - 1, -1, -1):
            S_below_bot[i] = (ifc[nlay] if i == nlay - 1
                              else _redheffer_star(ifc[i + 1], S_below[i + 1]))
            S_below[i] = _redheffer_star(prop[i], S_below_bot[i])
        out.update(modes=modes, thicks=thicks, S_above=S_above,
                   S_below=S_below, S_below_bot=S_below_bot,
                   eps_layers=eps_layers)
    return out


def _farfield(core):
    """Reflected / transmitted Jones (PUBLIC ``exp(-i w t)`` convention, ``2x2``
    columns = incident lab ``[Ex; Ey]``) and per-incident-polarization power
    ``R, T``.

    Everything needed is already in ``core``: the half-space mode blocks give the
    fields and the ``_flux`` Poynting projection carries the ``n cos(theta)`` /
    ``n / cos(theta)`` admittances, so the incident geometry never re-enters.
    W6 F-4 (2026-07-26): this used to take ``(core, eps_sup, eps_sub, Kx, Ky)``
    and reference NONE of the four extras (AST-verified) -- inert parameters that
    invited the reader to think the far field re-derives the flux weights.  The
    docstring also mislabelled the return gauge as INTERNAL; it is PUBLIC (this
    cascade runs on raw public eps end to end, and the returned Jones is pinned
    bit-identical to ``rcwa_jones_1d``'s public-convention Jones)."""
    Wf_s, Wb_s, Vf_s, Vb_s = (core["Wf_s"], core["Wb_s"],
                              core["Vf_s"], core["Vb_s"])
    Wf_b, Vf_b = core["Wf_b"], core["Vf_b"]
    S11, S21 = core["S11"], core["S21"]
    Jr = np.zeros((2, 2), dtype=_C)
    Jt = np.zeros((2, 2), dtype=_C)
    R = np.zeros(2)
    T = np.zeros(2)
    for col, Einc in enumerate((np.array([1.0, 0.0], _C),
                                np.array([0.0, 1.0], _C))):
        c_inc = np.linalg.solve(Wf_s, Einc)          # forward sup modal amps
        c_ref = S11 @ c_inc                           # backward sup
        c_trn = S21 @ c_inc                           # forward sub
        E_inc = Wf_s @ c_inc
        H_inc = Vf_s @ c_inc
        E_ref = Wb_s @ c_ref
        H_ref = Vb_s @ c_ref
        E_trn = Wf_b @ c_trn
        H_trn = Vf_b @ c_trn
        Jr[:, col] = E_ref
        Jt[:, col] = E_trn
        F_inc = _flux(E_inc, H_inc)
        R[col] = -_flux(E_ref, H_ref) / F_inc        # backward flux is -z
        T[col] = _flux(E_trn, H_trn) / F_inc
    return Jr, Jt, R, T


# =========================================================================== #
# out-of-plane-tensor + oblique incidence: generalized S-matrix cascade
# =========================================================================== #
#
# For an OUT-OF-PLANE tensor (``eps_xz/eps_yz != 0``) at OBLIQUE incidence this
# module routes to the SAME generalized (Li 2003) S-matrix that
# ``rcwa_jones_1d`` / ``RCWAStack`` use: a planar stack is a single-Fourier-order
# grating, so the RCWA mode machinery (``_homogeneous_eigenmodes`` for the
# isotropic half-spaces / scalar spacers, ``_layer_eigenmodes_tensor`` fed the
# ezz-Schur-condensed in-plane block + raw off-plane operators for tensor layers)
# reduces to it EXACTLY.  Validated to match ``rcwa_jones_1d`` / ``RCWAStack``
# to machine precision across iso / in-plane / OOP-normal / OOP-oblique /
# OOP-conical / lossy / multilayer.  (The two harness transfer-matrix oracles
# that previously flagged this as a "berreman bug" each carried their own
# transfer-direction error, single-layer and multilayer; the generalized S-matrix
# is the independently-cross-checked truth.)
#
# HISTORY / CORRECTION (W6 audit, 2026-07-26).  This block used to claim the
# NATIVE cascade above is "subtly wrong" for an out-of-plane tensor at oblique
# incidence -- that "the [W; -V] <-> -lam symmetry the native pairing implicitly
# relies on is BROKEN there, so the reflected amplitudes come out ~2% off".  That
# claim is REFUTED by measurement and is no longer true of this code: the native
# ``_solve_core`` / ``_farfield`` and this generalized cascade agree to 3.8e-15 on
# R, T, jones_r AND jones_t for rotx- / roty- / rotx-then-rotz-rotated biaxial
# slabs at theta in {0.2, 0.5, 0.9, 1.2} and phi in {0, 0.6, 2.4}, and to 1.0e-11
# worst over a 4000-config randomized sweep (1-3 fully-rotated biaxial layers,
# lossy layers, absorbing substrates, n_sup / n_sub in [1, 3.5], all angles and
# azimuths).  The pairing is restored by the 2026-07-14 FACTOR-i correction to
# :func:`_berreman_delta` -- which POST-DATES this router -- so the router is now
# a redundant (but independently cross-validated) second path, not an accuracy
# requirement.  It is KEPT because it is the path pinned against ``rcwa_jones_1d``
# and because the two-path agreement is itself a standing cross-family gate
# (``tests/unit/test_niche_audit_w6_berreman.py``).  Do NOT re-derive the "~2%"
# figure from this comment: measure it.


def _tensor_is_offplane(eps):
    """True if a ``(3, 3)`` permittivity has OUT-OF-PLANE coupling
    (``eps_xz / eps_yz / eps_zx / eps_zy`` above a tiny relative tolerance).
    Scalars / isotropic tensors return False."""
    M = np.asarray(eps, dtype=_C)
    if M.ndim != 2 or M.shape != (3, 3):
        return False
    off = max(abs(M[0, 2]), abs(M[1, 2]), abs(M[2, 0]), abs(M[2, 1]))
    diag = max(abs(M[0, 0]), abs(M[1, 1]), abs(M[2, 2]), 1.0)
    return bool(off > 1e-12 * diag)


def _offplane_oblique(eps_layers, Kx, Ky):
    """Route condition for the generalized S-matrix path: at least one
    out-of-plane tensor layer AND oblique incidence (``Kx`` or ``Ky != 0``).
    Everything else stays on the exact native Berreman cascade."""
    oblique = (abs(float(np.real(Kx))) > 1e-12
               or abs(float(np.real(Ky))) > 1e-12)
    return oblique and any(_tensor_is_offplane(e) for e in eps_layers)


def _offplane_condensed_M(eps_internal, Kx1, Ky1):
    """``(M, lam_f, lam_b)`` for one layer (INTERNAL-convention eps) via the RCWA
    mode functions.  Scalar / isotropic -> analytic homogeneous ``[W; -V]``
    (``lam, -lam``); tensor -> ezz-Schur-condensed in-plane block + raw off-plane
    (Li 2003), as the general 6-tuple (out-of-plane) or the symmetric 3-tuple
    (in-plane).  ``Kx1 / Ky1`` are ``1x1`` (single Fourier order)."""
    e = np.asarray(eps_internal, dtype=_C)
    if e.ndim == 0 or float(np.max(np.abs(e - e[0, 0] * np.eye(3)))) < \
            1e-12 * max(1.0, abs(e[0, 0] if e.ndim else e)):
        es = complex(e if e.ndim == 0 else e[0, 0])
        W, V, kz = _homogeneous_eigenmodes(Kx1, Ky1, es)
        kzc = kz.ravel()
        lam = _sqrt_decay(-np.concatenate([kzc, kzc]) ** 2)
        return _modes_to_M(W, V, W, -V), lam, -lam
    exx, exy, exz = e[0, 0], e[0, 1], e[0, 2]
    eyx, eyy, eyz = e[1, 0], e[1, 1], e[1, 2]
    ezx, ezy, ezz = e[2, 0], e[2, 1], e[2, 2]
    d = 1.0 / ezz
    o = lambda v: np.array([[v]], dtype=_C)     # noqa: E731 (1x1 Fourier op)
    Cxx, Cxy = o(exx - exz * ezx * d), o(exy - exz * ezy * d)
    Cyx, Cyy = o(eyx - eyz * ezx * d), o(eyy - eyz * ezy * d)
    if max(abs(exz), abs(eyz), abs(ezx), abs(ezy)) > 1e-12:
        Wf, Vf, lf, Wb, Vb, lb = _layer_eigenmodes_tensor(
            Kx1, Ky1, Cxx, Cxy, Cyx, Cyy, o(ezz),
            o(ezx), o(ezy), o(exz), o(eyz))
        return _modes_to_M(Wf, Vf, Wb, Vb), lf, lb
    W, V, lam = _layer_eigenmodes_tensor(Kx1, Ky1, Cxx, Cxy, Cyx, Cyy, o(ezz))
    return _modes_to_M(W, V, W, -V), lam, -lam


def _offplane_condensed_M_cached(eps_internal, Kx1, Ky1):
    """:func:`_offplane_condensed_M` memoized on ``(eps_internal, Kx1, Ky1)`` in
    the shared ``_MODE_CACHE`` (tagged-key namespace).

    S1-11 (audit AUDIT_V5_24_2): the generalized OOP-oblique path is eig-heavy
    (a dense ``_layer_eigenmodes_tensor`` per layer) and, exactly like the native
    Berreman cascade, its eig is WAVELENGTH-INDEPENDENT -- it depends only on
    ``eps`` and the normalized ``Kx/Ky`` (angle), never on ``k0`` (which enters
    solely through the propagation phase ``k0*thickness``).  So a fixed-angle
    wavelength sweep of a tilted-director stack recomputed every eig; caching
    returns the ``(M, lam_f, lam_b)`` triple byte-for-byte (arrays read-only
    downstream, mirroring ``_layer_modes_cached``).  The tagged key keeps this
    disjoint from ``_layer_modes_cached``'s native 6-tuple entries."""
    a = np.ascontiguousarray(np.asarray(eps_internal, dtype=_C))
    key = ("offplane_condensed_M", a.tobytes(), a.shape,
           complex(Kx1[0, 0]), complex(Ky1[0, 0]))
    with _MODE_CACHE_LOCK:
        hit = _MODE_CACHE.get(key)
        if hit is not None:
            _MODE_CACHE.move_to_end(key)            # LRU: refresh recency
            return hit
    res = _freeze(_offplane_condensed_M(eps_internal, Kx1, Ky1))
    with _MODE_CACHE_LOCK:
        _MODE_CACHE[key] = res
        while len(_MODE_CACHE) > _MODE_CACHE_SIZE:
            _MODE_CACHE.popitem(last=False)
    return res


def _offplane_oblique_solve(eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky,
                            retain=False):
    """``(R, T, Jr, Jt)`` for a stack with out-of-plane tensor layer(s) at oblique
    incidence, via the generalized (Li 2003) single-order S-matrix cascade.
    ``eps_*`` are PUBLIC (``Im > 0``); returns PUBLIC-convention Jones + per-lab-
    polarization power ``R, T``.  Validated to ``rcwa_jones_1d`` / ``RCWAStack``.

    ``retain=True`` (AUDIT_DYNAMETA_CONSUMER_API_GAPS C2) additionally returns
    a retained-internals ``core`` dict in the SAME shape ``_solve_core(retain=
    True)`` produces -- per-layer asymmetric modes ``(Wf, Vf, lamf, Wb, Vb,
    lamb)`` sliced from the generalized ``M`` blocks, plus the bracketing
    partial cascades built with the SAME generalized interface / propagation
    convention (backward propagator ``exp(+lamb k0 L)``, ``Re(lamb) <= 0``).
    The native internal observables (``_amplitudes`` / ``internal_field`` /
    ``layer_absorption``) are mode-shape agnostic S-matrix algebra + the
    full-tensor ``E_z`` recovery, so they serve this core unchanged -- the
    'asymmetric-mode recovery in the generalized convention' the old raise
    said was missing, with the flux taken from the tangential fields
    (Rayleigh-consistent by construction for the single planar order)."""
    kx0, ky0 = float(np.real(Kx)), float(np.real(Ky))
    o = lambda v: np.array([[v]], dtype=_C)     # noqa: E731 (1x1 Fourier op)
    Kx1, Ky1 = o(kx0), o(ky0)
    k0 = 2.0 * np.pi / wl
    Wr, Vr, _kzr = _homogeneous_eigenmodes(Kx1, Ky1, np.conj(_C(eps_sup)))
    Wt, Vt, _kzt = _homogeneous_eigenmodes(Kx1, Ky1, np.conj(_C(eps_sub)))
    M_prev = _modes_to_M(Wr, Vr, Wr, -Vr)
    S = None
    modes6, ifc, prop = [], [], []
    for eps, thk in zip(eps_layers, thicks):
        # S1-11: cached (wl-independent) eig + interface, so a fixed-angle
        # wavelength sweep reuses them byte-for-byte instead of recomputing.
        Ml, lf, lb = _offplane_condensed_M_cached(
            np.conj(np.asarray(eps, dtype=_C)), Kx1, Ky1)
        Si = _interface_smatrix_cached(M_prev, Ml)
        S = Si if S is None else _redheffer_star(S, Si)
        S = _propagation_star_general(S, lf, lb, k0 * float(thk))
        M_prev = Ml
        if retain:
            # slice the asymmetric mode blocks back out of M = [[Wf, Wb],
            # [Vf, Vb]] (an in-plane layer's condensed M carries Wb = W,
            # Vb = -V, lamb = -lam -- the degenerate symmetric case).
            n = Ml.shape[0] // 2
            modes6.append((Ml[:n, :n], Ml[n:, :n], lf,
                           Ml[:n, n:], Ml[n:, n:], lb))
            ifc.append(Si)
            prop.append(_propagation_smatrix_general(lf, lb, k0 * float(thk)))
    ifc_sub = _interface_smatrix_cached(M_prev, _modes_to_M(Wt, Vt, Wt, -Vt))
    S = _redheffer_star(S, ifc_sub)
    S11, _S12, S21, _S22 = S
    core = None
    if retain:
        ifc.append(ifc_sub)
        nlay = len(modes6)
        # bracketing partial cascades -- the _solve_core(retain=True) pattern:
        # S_above[i] = sup -> TOP of layer i (through the interface INTO it);
        # S_below_bot[i] = BOTTOM of layer i -> sub (no own propagation);
        # S_below[i] = prop(i) * S_below_bot[i].
        S_above = [None] * nlay
        S_above[0] = ifc[0]
        for i in range(1, nlay):
            S_above[i] = _redheffer_star(
                _redheffer_star(S_above[i - 1], prop[i - 1]), ifc[i])
        S_below = [None] * nlay
        S_below_bot = [None] * nlay
        for i in range(nlay - 1, -1, -1):
            S_below_bot[i] = (ifc[nlay] if i == nlay - 1
                              else _redheffer_star(ifc[i + 1], S_below[i + 1]))
            S_below[i] = _redheffer_star(prop[i], S_below_bot[i])
        # GAUGE: this cascade runs on the CONJUGATED permittivities (the
        # internal exp(+iwt) gauge), while the native core -- and therefore
        # the retained-internals consumers (internal_field's -i modal-H and
        # tensor E_z recovery on the PUBLIC eps) -- are public-gauge
        # end-to-end.  The internal problem is the complex conjugate of the
        # public one, so the conj of every retained operator IS the public
        # core (probe-verified: pre-conj fields came out as the exact conj
        # of the native path's).  The MODAL-H blocks additionally NEGATE:
        # the -i modal-H convention is not conj-invariant
        # (H_phys_pub = conj(-i H_int) = +i conj(H_int), so the public modal
        # H in the same -i convention is -conj(V_int); probe: without the
        # sign, H came out as exactly -H_native while E matched).  Absorption
        # is invariant (numerator and incident flux flip together).  Decay is
        # preserved (Re(lam) unchanged under conj).
        cj = np.conj
        modes6 = [(cj(Wf), -cj(Vf), cj(lf), cj(Wb), -cj(Vb), cj(lb))
                  for (Wf, Vf, lf, Wb, Vb, lb) in modes6]
        S_above = [tuple(cj(b) for b in s) for s in S_above]
        S_below = [tuple(cj(b) for b in s) for s in S_below]
        S_below_bot = [tuple(cj(b) for b in s) for s in S_below_bot]
        core = dict(modes=modes6, thicks=[float(t) for t in thicks],
                    S_above=S_above, S_below=S_below,
                    S_below_bot=S_below_bot, eps_layers=list(eps_layers),
                    k0=k0, Wf_s=cj(Wr), Vf_s=-cj(Vr))
    kz_inc = float(np.real(_sqrt_forward(_C(eps_sup) - kx0 ** 2 - ky0 ** 2)))
    # W6 F-1 (2026-07-26): ``_forward_flux_kz`` takes the INTERNAL-gauge eps and
    # UN-conjugates it itself (every rcwa call site passes ``eps_sup = conj(n**2)``);
    # this cascade holds the PUBLIC eps, so feeding it raw conjugated the value
    # TWICE and returned ``Re(kz) < 0`` for an absorbing half-space -- which the
    # ``Re(kz) > 0`` propagating mask below read as evanescent and SILENTLY
    # ZEROED.  Measured pre-fix: a tilted-director slab on
    # n_sub = 1.5+0.3j at theta = 0.3 returned T = 0.000000 where
    # ``rcwa_jones_1d`` (this path's own stated reference) gives T = 0.930316.
    # Conjugating here restores the rcwa call convention exactly; LOSSLESS eps is
    # real so conj is the identity and every lossless result is byte-unchanged.
    kzrf = _forward_flux_kz(np.conj(_C(eps_sup)),
                            np.array([kx0]), np.array([ky0]))[0]
    kztf = _forward_flux_kz(np.conj(_C(eps_sub)),
                            np.array([kx0]), np.array([ky0]))[0]
    Jr = np.zeros((2, 2), dtype=_C)
    Jt = np.zeros((2, 2), dtype=_C)
    R = np.zeros(2)
    T = np.zeros(2)
    for col, (ex0, ey0) in enumerate(((1.0, 0.0), (0.0, 1.0))):
        r = S11 @ np.array([ex0, ey0], dtype=_C)
        t = S21 @ np.array([ex0, ey0], dtype=_C)
        rx, ry, tx, ty = r[0], r[1], t[0], t[1]
        longi = kx0 * ex0 + ky0 * ey0
        einc_sq = 1.0 + (longi / kz_inc) ** 2 if kz_inc != 0 else 1.0
        # longitudinal (z) field from div(D)=0: BOTH kx and ky contribute at a
        # CONICAL mount (dropping ky under-counts the y-pol flux by ~1%).
        rz = -(kx0 * rx + ky0 * ry) / (kzrf if abs(kzrf) > 1e-12 else 1.0)
        tz = -(kx0 * tx + ky0 * ty) / (kztf if abs(kztf) > 1e-12 else 1.0)
        R[col] = (float(np.real(kzrf / kz_inc))
                  * (abs(rx) ** 2 + abs(ry) ** 2 + abs(rz) ** 2) / einc_sq
                  if np.real(kzrf) > 0 else 0.0)
        T[col] = (float(np.real(kztf / kz_inc))
                  * (abs(tx) ** 2 + abs(ty) ** 2 + abs(tz) ** 2) / einc_sq
                  if np.real(kztf) > 0 else 0.0)
        Jr[:, col] = [np.conj(rx), np.conj(ry)]
        Jt[:, col] = [np.conj(tx), np.conj(ty)]
    if retain:
        return R, T, Jr, Jt, core
    return R, T, Jr, Jt


# =========================================================================== #
# functional entry
# =========================================================================== #

def berreman_jones_1d(
    layers: List[Tuple[Union[complex, np.ndarray], float]],
    n_substrate: complex,
    n_superstrate: complex,
    wavelength: float,
    *,
    angle: float = 0.0,
    phi: float = 0.0,
    theta: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Full ``2x2`` Jones reflection / transmission of an ANISOTROPIC PLANAR
    multilayer by the Berreman ``4x4`` method.

    Parameters
    ----------
    layers : list of (eps, thickness)
        Ordered from the SUPERSTRATE side inward (first = outermost).  Each
        ``eps`` is a scalar (isotropic) or a ``(3, 3)`` tensor (PUBLIC
        ``Im > 0`` for loss); ``thickness`` in metres.
    n_substrate, n_superstrate : complex
        Half-space refractive indices (isotropic).
    wavelength : float
        Vacuum wavelength [m].
    angle / theta : float, optional
        Polar incidence angle [rad] (``theta`` is the cross-suite alias and
        wins when both are given).
    phi : float, optional
        Azimuth [rad] (conical mount; ``0`` = classical x-z plane).

    Returns
    -------
    R, T : (2,) float ndarray
        Power reflectance / transmittance for incident lab ``E_x`` (index 0)
        and ``E_y`` (index 1), flux-normalized (``R + T = 1`` for a lossless
        stack).
    jones_r, jones_t : (2, 2) complex ndarray
        Reflection / transmission Jones; columns = incident lab ``[Ex; Ey]``,
        rows = reflected / transmitted lab ``[Ex; Ey]``.  PUBLIC ``exp(-i w t)``.

    Notes
    -----
    CROSS-ENGINE SEAM (audit S5-4): this planar-stack function returns
    ``(R, T, jones_r, jones_t)`` -- so ``result[2]`` is the REFLECTION Jones
    MATRIX and ``result[3]`` is the TRANSMISSION Jones.  The grating engines
    :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` /
    :func:`~lumenairy.elements.pmm.pmm_jones_1d` return
    ``(orders, R_eff, T_eff, jones_reflection)`` instead, so THEIR ``result[2]``
    is the ``T_eff`` efficiency ARRAY and ``result[3]`` is the REFLECTION Jones
    -- ``result[3]`` therefore means TRANSMISSION here but REFLECTION there.
    Mind the position when writing engine-agnostic post-processing.

    Accuracy
    --------
    Exact (machine precision) for isotropic, in-plane-anisotropic and
    out-of-plane-tensor layers at any incidence, including OBLIQUE and CONICAL.
    Two internal paths: the native Berreman ``4x4`` S-matrix cascade, and -- for
    an OUT-OF-PLANE tensor (``eps_xz/eps_yz != 0``) at OBLIQUE incidence -- the
    generalized (Li 2003) single-Fourier-order S-matrix that
    :func:`~lumenairy.elements.rcwa.rcwa_jones_1d` / ``RCWAStack`` use, to which
    it is validated to machine precision (all regimes + lossy + multilayer).
    The route is automatic and the two paths agree to ~1e-15 on R, T and BOTH
    Jones matrices EVERYWHERE, including the out-of-plane-oblique regime the
    router exists for (W6 audit 2026-07-26; the older "the native cascade is ~2%
    off there" note is refuted -- see the comment above
    :func:`_tensor_is_offplane` for the measurement).

    Scope
    -----
    A propagating, non-amplifying INCIDENCE half-space is required: an
    evanescent / metallic / gain superstrate raises (the flux normalisation
    divides by ``kz_inc``), and for a PASSIVE stack an energy tripwire rejects a
    gross ``R + T >> 1`` -- the same guards ``rcwa_jones_1d`` / ``pmm_jones_1d``
    apply.  ``|angle| < pi/2`` (front-side illumination).  Unlike the grating
    engines, a GAIN LAYER is NOT rejected here: it returns the exact formal
    steady state (``R + T > 1``), a documented sign-convention contract of this
    module (audit S1-6) -- see :func:`_check_energy_pols`.  A LOSSY superstrate
    is accepted but the per-polarization ``R`` / ``T`` are only physically
    meaningful for a LOSSLESS incidence medium (each wave is normalised by its
    own ``|amp|^2`` z-flux while the incident/reflected cross-term carries net
    flux -- the family-wide caveat recorded on
    ``rcwa._core._forward_flux_kz``).  Note also that ``Kx``/``Ky`` use
    ``Re(n_superstrate) sin(angle)``, matching ``rcwa`` / ``pmm``; the scalar
    :func:`~lumenairy.elements.coatings.coating_reflectance` TMM instead carries
    a COMPLEX Snell invariant, so the two agree only for a lossless incidence
    medium (verified: 1.9e-15 across 48 lossless-superstrate configs).
    """
    # Differentiable (JAX) dispatch: any traced input routes to the jnp twin.
    from ..backend import is_jax_array
    if (is_jax_array(n_substrate) or is_jax_array(n_superstrate)
            or is_jax_array(wavelength) or is_jax_array(angle)
            or is_jax_array(phi) or is_jax_array(theta)
            or any(is_jax_array(e) or is_jax_array(t) for e, t in layers)):
        # The jnp twin routes a concretely-detected out-of-plane tensor to the
        # SAME generalized (out-of-plane-correct) S-matrix as the NumPy path, so
        # differentiable out-of-plane-oblique / conical is supported.  (A TRACED
        # (3, 3) tensor is routed there by SHAPE alone -- D12 -- so no stack
        # falls through to the native cascade for want of inspection; both
        # cascades are measured equal to ~1e-15 anyway, W6.)
        from ._berreman_jax import _berreman_jones_1d_jax
        return _berreman_jones_1d_jax(layers, n_substrate, n_superstrate,
                                      wavelength, angle=angle, phi=phi,
                                      theta=theta)
    angle = float(_checked_angle("berreman_jones_1d", angle, theta))
    for _nm, _n in (("n_superstrate", n_superstrate),
                    ("n_substrate", n_substrate)):
        if not np.isfinite(_C(_n)):            # W6 F-2, before the ** 2
            raise ValueError(
                f"berreman_jones_1d: {_nm} is not finite ({_n!r}); a NaN/inf "
                f"half-space index would propagate silently through the solve.")
    eps_layers = [_as_eps_tensor(e) for e, _t in layers]
    thicks = [float(t) for _e, t in layers]
    eps_sup = complex(_C(n_superstrate) ** 2)
    eps_sub = complex(_C(n_substrate) ** 2)
    nre = float(np.real(_C(n_superstrate)))
    Kx = nre * np.sin(angle) * np.cos(phi)
    Ky = nre * np.sin(angle) * np.sin(phi)
    _check_inputs("berreman_jones_1d", eps_sup, eps_sub, eps_layers, thicks,
                  wavelength, Kx, Ky)
    passive = _is_passive(eps_layers, eps_sup, eps_sub)
    if _offplane_oblique(eps_layers, Kx, Ky):
        # out-of-plane tensor at oblique incidence: the generalized S-matrix
        # cascade (matches rcwa_jones_1d / RCWAStack to machine precision).
        R, T, Jr, Jt = _offplane_oblique_solve(
            eps_layers, thicks, eps_sup, eps_sub, wavelength, Kx, Ky)
        _check_energy_pols("berreman_jones_1d", R, T, passive)
        return R, T, Jr, Jt
    core = _solve_core(eps_layers, thicks, eps_sup, eps_sub, wavelength,
                       Kx, Ky)
    Jr, Jt, R, T = _farfield(core)
    _check_energy_pols("berreman_jones_1d", R, T, passive)
    return R, T, Jr, Jt


# =========================================================================== #
# builder class (mirrors RCWAStack / PMMStack)
# =========================================================================== #

class BerremanStack:
    """Builder for an anisotropic planar multilayer solved by the Berreman
    ``4x4`` method -- the planar / unpatterned member of the solver family
    (cf. :class:`~lumenairy.elements.rcwa.RCWAStack`,
    :class:`~lumenairy.elements.pmm.PMMStack`).

    >>> st = BerremanStack(n_substrate=1.5, n_superstrate=1.0)
    >>> st.add_layer(140e-9, eps=lc_tensor)       # a planar LC retarder
    >>> st.add_layer(100e-9, eps=2.1)             # an isotropic cap
    >>> R, T, jones = st.set_source(0.633e-6, theta=0.2).solve()
    """

    def __init__(self, *, n_substrate=1.0, n_superstrate=1.0):
        from ..backend import is_jax_array
        # JAX half-space indices stay RAW so their gradient flows.
        self.n_sub = (n_substrate if is_jax_array(n_substrate)
                      else _C(n_substrate))
        self.n_sup = (n_superstrate if is_jax_array(n_superstrate)
                      else _C(n_superstrate))
        self._layers = []                     # (thickness, eps_tensor_public)
        self._src = None
        self._internal = None
        self._jones_t = None                  # transmission Jones of last solve

    def add_layer(self, thickness, *, eps):
        """Append a planar layer of thickness ``t`` [m] and permittivity
        ``eps`` (scalar isotropic or ``(3, 3)`` tensor, PUBLIC ``Im > 0``)."""
        from ..backend import is_jax_array
        # Keep traced thickness / eps raw (the differentiable solve path);
        # float()/_as_eps_tensor() would sever the trace.
        if is_jax_array(thickness):
            t_store = thickness
        else:
            t_store = float(thickness)
            if t_store <= 0:
                raise ValueError(
                    "BerremanStack.add_layer: thickness must be > 0.")
        eps_store = eps if is_jax_array(eps) else _as_eps_tensor(eps)
        self._layers.append((t_store, eps_store))
        return self

    def add_effective_grating(self, thickness, *, eps_ridge, eps_groove, fill,
                              period=None, order=0):
        """Append a sub-wavelength 1-D binary grating as its EFFECTIVE-MEDIUM
        (Rytov) uniaxial layer -- the EMT bridge that lets a whole design sweep
        run in microseconds before a rigorous RCWA / PMM cross-check.

        The grating (period along ``x``, ridge ``eps_ridge`` fill ``fill``,
        groove ``eps_groove``) homogenizes to ``diag(eps_perp, eps_par,
        eps_par)`` via :func:`~lumenairy.elements.emt.rytov_tensor`.  ``order=2``
        adds the second-order bulk-index correction and needs ``period`` (the
        wavelength comes from :meth:`set_source`, so call ``set_source`` first
        for ``order=2``).  EXACT only as ``period -> 0`` -- a screen, not a
        replacement for the rigorous solve."""
        from .emt import rytov_tensor
        wl = None
        if order == 2:
            if self._src is None:
                raise ValueError(
                    "BerremanStack.add_effective_grating: order=2 needs the "
                    "wavelength; call set_source(...) before adding the layer.")
            wl = self._src["wl"]
        eps_eff = rytov_tensor(eps_ridge, eps_groove, fill,
                               period=period, wavelength=wl, order=order)
        return self.add_layer(thickness, eps=eps_eff)

    def set_source(self, wavelength, *, angle=0.0, theta=None, phi=0.0):
        """Set the incident plane wave (vacuum ``wavelength`` [m], polar
        ``angle`` / ``theta`` [rad], azimuth ``phi`` [rad]).

        W6 F-2: the ``angle`` / ``theta`` alias now resolves through the shared
        CHECKED resolver, so a back-side ``|angle| >= pi/2`` is rejected HERE --
        the ``PMMStack.set_source`` precedent (audit S1-7), which found the
        setter bypassing the guard the functional entry had."""
        from ..backend import is_jax_array
        angle = _checked_angle("BerremanStack.set_source", angle, theta)
        self._src = dict(
            wl=wavelength if is_jax_array(wavelength) else float(wavelength),
            angle=angle if is_jax_array(angle) else float(angle),
            phi=phi if is_jax_array(phi) else float(phi))
        return self

    def _holds_traced(self):
        from ..backend import is_jax_array
        return (is_jax_array(self.n_sub) or is_jax_array(self.n_sup)
                or (self._src is not None
                    and (is_jax_array(self._src["wl"])
                         or is_jax_array(self._src["angle"])
                         or is_jax_array(self._src["phi"])))
                or any(is_jax_array(t) or is_jax_array(e)
                       for t, e in self._layers))

    def _geom(self):
        wl, ang, phi = self._src["wl"], self._src["angle"], self._src["phi"]
        eps_sup = complex(self.n_sup ** 2)
        eps_sub = complex(self.n_sub ** 2)
        nre = float(np.real(self.n_sup))
        Kx = nre * np.sin(ang) * np.cos(phi)
        Ky = nre * np.sin(ang) * np.sin(phi)
        eps_layers = [e for _t, e in self._layers]
        thicks = [t for t, _e in self._layers]
        return wl, eps_sup, eps_sub, Kx, Ky, eps_layers, thicks

    def solve(self, *, retain_internal=False):
        """Solve the stack.  Returns ``(R, T, jones)`` -- ``R``/``T`` are
        ``(2,)`` power per incident lab polarization (``E_x``, ``E_y``);
        ``jones`` is the ``2x2`` reflection Jones (PUBLIC convention)."""
        if self._src is None:
            raise ValueError("BerremanStack.solve: call set_source(...) first.")
        if not self._layers:
            raise ValueError("BerremanStack.solve: add at least one layer.")
        if self._holds_traced():
            # The differentiable twin now supports retain_internal too (the
            # internal-field / layer-absorption reconstruction is backend-generic
            # S-matrix algebra); it stores self._internal so the observables
            # dispatch to their jnp ports below.
            from ._berreman_jax import _berreman_stack_solve_jax
            return _berreman_stack_solve_jax(
                self, retain_internal=retain_internal)
        wl, eps_sup, eps_sub, Kx, Ky, eps_layers, thicks = self._geom()
        eps_layers = [_as_eps_tensor(e) for e in eps_layers]
        _check_inputs("BerremanStack.solve", eps_sup, eps_sub, eps_layers,
                      thicks, wl, Kx, Ky)
        passive = _is_passive(eps_layers, eps_sup, eps_sub)
        if _offplane_oblique(eps_layers, Kx, Ky):
            # out-of-plane tensor at oblique incidence -> generalized S-matrix.
            if retain_internal:
                # AUDIT_DYNAMETA_CONSUMER_API_GAPS C2: the generalized cascade
                # now retains the same internals shape the native core does
                # (asymmetric modes sliced from the M blocks + generalized-
                # convention partial cascades), so internal_field /
                # layer_absorption serve OOP-tensor-at-oblique stacks --
                # the flagship tilted-director regime.
                R, T, Jr, Jt, core = _offplane_oblique_solve(
                    eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky,
                    retain=True)
                cinc = np.zeros((2, 2), dtype=_C)
                for col, Einc in enumerate((np.array([1.0, 0.0], _C),
                                            np.array([0.0, 1.0], _C))):
                    cinc[:, col] = np.linalg.solve(core["Wf_s"], Einc)
                core["cinc"] = cinc
                core["Kx"], core["Ky"] = Kx, Ky
                core["R"], core["T"] = R, T
                self._internal = core
                self._jones_t = Jt                    # A1
                _check_energy_pols("BerremanStack.solve", R, T, passive)
                return R, T, Jr
            R, T, Jr, Jt = _offplane_oblique_solve(
                eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky)
            # AUDIT_DYNAMETA_CONSUMER_API_GAPS A1: retain the transmission
            # Jones the far-field close-out already computed (consumers
            # previously re-solved via the functional entry just for t).
            self._jones_t = Jt
            _check_energy_pols("BerremanStack.solve", R, T, passive)
            return R, T, Jr
        core = _solve_core(eps_layers, thicks, eps_sup, eps_sub, wl, Kx, Ky,
                           retain=retain_internal)
        Jr, Jt, R, T = _farfield(core)
        self._jones_t = Jt                    # A1: zero extra compute
        if retain_internal:
            # incident modal amplitudes per lab polarization, for the field /
            # absorption reconstruction.
            cinc = np.zeros((2, 2), dtype=_C)
            for col, Einc in enumerate((np.array([1.0, 0.0], _C),
                                        np.array([0.0, 1.0], _C))):
                cinc[:, col] = np.linalg.solve(core["Wf_s"], Einc)
            core["cinc"] = cinc
            core["Kx"], core["Ky"] = Kx, Ky
            core["R"], core["T"] = R, T
            self._internal = core
        _check_energy_pols("BerremanStack.solve", R, T, passive)
        return R, T, Jr

    def jones_transmission(self):
        """The ``2x2`` TRANSMISSION Jones of the last :meth:`solve` --
        columns keyed to incident lab ``E_x`` / ``E_y``, PUBLIC
        ``exp(-i w t)`` convention: the same contract as the functional
        :func:`berreman_jones_1d`'s ``jones_t`` return (bit-identical to it;
        the class solve computes it on every path and previously discarded
        it, forcing a second functional solve on consumers that need the
        transmitted phase alongside the internal observables --
        AUDIT_DYNAMETA_CONSUMER_API_GAPS A1)."""
        if self._jones_t is None:
            raise ValueError(
                "BerremanStack.jones_transmission: call solve() first.")
        return self._jones_t

    # -- internal observables ------------------------------------------- #

    def _amplitudes(self):
        """Per-layer ``(c_fwd_top, c_bwd_bot)`` modal amplitudes (each
        ``2x2``: cols = incident lab pol) from the retained partial cascades --
        the general (asymmetric-mode) RCWAResult._internal_cpm recovery:
        ``c+ = inv(I - Sa22 Sb11) Sa21 cinc`` at the layer TOP, then the
        backward amplitude at the BOTTOM from the forward field propagated
        down (a DECAYING ``Xf``)."""
        d = self._internal
        if d.get("_is_jax"):
            import jax.numpy as jnp

            from ._berreman_jax import _amplitudes_jax
            return _amplitudes_jax(d, jnp)
        cinc = d["cinc"]
        out = []
        k0 = d["k0"]
        for i, (Wf, Vf, lamf, Wb, Vb, lamb) in enumerate(d["modes"]):
            Sa = d["S_above"][i]
            Sb11 = d["S_below"][i][0]
            Bb11 = d["S_below_bot"][i][0]
            A22, A21 = Sa[3], Sa[2]
            denom = np.linalg.inv(np.eye(2, dtype=_C) - A22 @ Sb11)
            c_fwd = denom @ (A21 @ cinc)              # forward at layer TOP
            Xf = np.exp(-lamf * k0 * d["thicks"][i])
            c_bwd = Bb11 @ (Xf[:, None] * c_fwd)      # backward at layer BOTTOM
            out.append((c_fwd, c_bwd))
        return out

    def internal_field(self, z, *, component="all", incident=(1.0, 0.0)):
        """Reconstruct the E and/or H field INSIDE the stack -- the planar
        counterpart of :meth:`RCWAResult.internal_field` /
        :meth:`PMMStack.internal_field`.

        Requires ``solve(retain_internal=True)``.

        Parameters
        ----------
        z : float or array-like
            Depth(s) [m] from the stack TOP (the superstrate interface).
        component : {'E', 'H', 'all'}, optional
            Which field(s) to return (default all six components).
        incident : (complex, complex), optional
            PUBLIC incident Jones ``(E_x, E_y)`` (default x-polarized).

        Returns
        -------
        dict
            ``{'Ex','Ey','Ez','Hx','Hy','Hz'}`` (per ``component``) each a
            ``(nz,)`` complex array, plus ``'z'`` and ``'layer'``.  PUBLIC
            ``exp(-i w t)`` convention.
        """
        d = self._internal
        if d is None or "cinc" not in d:
            raise ValueError(
                "BerremanStack.internal_field: call solve(retain_internal="
                "True) first.")
        if d.get("_is_jax"):
            import jax.numpy as jnp

            from ._berreman_jax import _internal_field_jax
            return _internal_field_jax(d, z, component, incident, jnp)
        names = {"E": ("Ex", "Ey", "Ez"), "H": ("Hx", "Hy", "Hz"),
                 "all": ("Ex", "Ey", "Ez", "Hx", "Hy", "Hz")}
        if component not in names:
            raise ValueError(
                f"BerremanStack.internal_field: component must be 'E', 'H' or "
                f"'all', got {component!r}.")
        want = names[component]
        wx, wy = complex(incident[0]), complex(incident[1])
        amps = self._amplitudes()
        k0, Kx, Ky = d["k0"], d["Kx"], d["Ky"]
        thicks = d["thicks"]
        z_top = np.concatenate([[0.0], np.cumsum(thicks)])
        zs = np.atleast_1d(np.asarray(z, dtype=float))
        scalar_in = np.ndim(z) == 0
        out = {c: np.empty(zs.shape[0], dtype=_C) for c in want}
        layers_out = np.empty(zs.shape[0], dtype=int)
        cache = {}
        for zi, zz in enumerate(zs):
            i = int(np.clip(np.searchsorted(z_top, zz, side="right") - 1,
                            0, len(thicks) - 1))
            zloc = zz - z_top[i]
            layers_out[zi] = i
            Wf, Vf, lamf, Wb, Vb, lamb = d["modes"][i]
            if i not in cache:
                c_fwd, c_bwd = amps[i]
                cache[i] = (wx * c_fwd[:, 0] + wy * c_fwd[:, 1],
                            wx * c_bwd[:, 0] + wy * c_bwd[:, 1])
            cF, cB = cache[i]
            t_i = thicks[i]
            P = np.exp(-lamf * k0 * zloc)          # forward from layer TOP
            Q = np.exp(lamb * k0 * (t_i - zloc))   # backward from layer BOTTOM
            E = Wf @ (P * cF) + Wb @ (Q * cB)
            H = Vf @ (P * cF) + Vb @ (Q * cB)
            Ex, Ey = E[0], E[1]
            Hx, Hy = -1j * H[0], -1j * H[1]      # modal H -> -i eta0 scale
            eps_t = d["eps_layers"][i]            # raw (public) tensor
            # Ez from Dz = 0: eps_zx Ex + eps_zy Ey + eps_zz Ez = -(Kx Hy - Ky Hx)
            ezz = eps_t[2, 2]
            Ez = (-(Kx * Hy - Ky * Hx)
                  - eps_t[2, 0] * Ex - eps_t[2, 1] * Ey) / ezz
            Hz = Kx * Ey - Ky * Ex                # curl-E z-component
            full = dict(Ex=Ex, Ey=Ey, Ez=Ez, Hx=Hx, Hy=Hy, Hz=Hz)
            for c in want:
                out[c][zi] = full[c]
        res = {c: (out[c][0] if scalar_in else out[c]) for c in want}
        res["z"] = float(zs[0]) if scalar_in else zs
        res["layer"] = int(layers_out[0]) if scalar_in else layers_out
        return res

    def layer_absorption(self):
        """Per-layer absorbed power fraction ``(n_layers, 2)`` (cols = incident
        lab ``E_x`` / ``E_y``) from the internal z-Poynting flux difference
        across each layer.  Requires ``solve(retain_internal=True)``.  The
        cross-machinery invariant ``sum_i A_i ~= 1 - R - T`` is the honest
        closure check (internal amplitudes vs the half-space far field)."""
        d = self._internal
        if d is None or "cinc" not in d:
            raise ValueError(
                "BerremanStack.layer_absorption: call solve(retain_internal="
                "True) first.")
        if d.get("_is_jax"):
            import jax.numpy as jnp

            from ._berreman_jax import _layer_absorption_jax
            return _layer_absorption_jax(d, jnp)
        amps = self._amplitudes()
        k0 = d["k0"]
        nlay = len(d["modes"])

        def flux_at(i, zfrac):
            Wf, Vf, lamf, Wb, Vb, lamb = d["modes"][i]
            c_fwd, c_bwd = amps[i]
            t = d["thicks"][i]
            P = np.exp(-lamf * k0 * (zfrac * t))[:, None]
            Q = np.exp(lamb * k0 * ((1.0 - zfrac) * t))[:, None]
            E = Wf @ (P * c_fwd) + Wb @ (Q * c_bwd)
            H = Vf @ (P * c_fwd) + Vb @ (Q * c_bwd)
            return np.array([_flux(E[:, c], H[:, c]) for c in range(2)])

        F_top = np.array([flux_at(i, 0.0) for i in range(nlay)])
        F_bot = np.array([flux_at(i, 1.0) for i in range(nlay)])
        # F_inc per pol: the top-of-stack internal forward flux equals the
        # incident flux (the cascade is referenced to unit incident modal amp).
        F_inc = np.array([
            self._incident_flux(c) for c in range(2)])
        return (F_top - F_bot) / F_inc[None, :]

    def _incident_flux(self, col):
        d = self._internal
        cinc = d["cinc"][:, col]
        return _flux(d["Wf_s"] @ cinc, d["Vf_s"] @ cinc)
