"""Effective-medium (homogenization) bridge: sub-wavelength patterned layers
-> equivalent anisotropic permittivity tensors for the Berreman planar solver.

This is the FAST SCREEN that feeds the rigorous patterned solvers (RCWA / PMM):
when a grating / metasurface period is well below the wavelength it acts as a
HOMOGENEOUS anisotropic film, so a whole sweep can be run in microseconds with
:class:`~lumenairy.elements.berreman.BerremanStack` (then a shortlist validated
rigorously).  It is an APPROXIMATION -- exact only in the ``period -> 0`` limit,
and it does not capture resonances -- so it screens, it does not replace.

1-D lamellar gratings (the rigorous case).  A binary grating of ridge ``eps_a``
(fill ``f``) and groove ``eps_b`` along ``x`` homogenizes to a UNIAXIAL tensor
``diag(eps_perp, eps_par, eps_par)`` (Rytov 1956): ``E`` across the lamellae
(``x``) sees the series / harmonic mean ``eps_perp``, ``E`` along the lamellae
(``y``, ``z``) sees the parallel / arithmetic mean ``eps_par``.  This
zeroth-order tensor is the EXACT ``period -> 0`` limit (validated here:
the rigorous ``rcwa_jones_1d`` / ``pmm_jones_1d`` slab Jones converges
monotonically onto it as the period shrinks).  An optional second-order
``(period/lambda)^2`` correction (Lalanne & Lemercier-Lalanne 1996) sharpens
the BULK effective index at moderate period; note the homogenized SLAB still
carries an inherent ``O(period/lambda)`` interface error that the bulk
correction does not remove, so for slab reflection it is a refinement, not a
new convergence order -- screen, then validate rigorously.

2-D inclusion arrays (approximate scalar mixing).  Maxwell-Garnett (dilute) and
Bruggeman (symmetric) give an isotropic effective ``eps`` for a 2-D pillar /
hole array -- a coarse first pass; for an anisotropic 2-D cell use a rigorous
2-D solve.

All permittivities are PUBLIC (relative ``eps``, ``Im > 0`` for loss), matching
the Berreman / RCWA / PMM convention.
"""
from __future__ import annotations

import warnings
from typing import List, Optional, Tuple, Union

import numpy as np

from ..backend import array_namespace, is_jax_array

_C = np.complex128

# v5.17 audit (P3-24): validity-domain guard threshold.  The order=2
# Rytov correction is an asymptotic series term in (period/wavelength)^2;
# beyond ~0.5 it grows without bound and can push the effective tensor
# outside the constituent bounds (even negative for high contrast).
_RYTOV_ORDER2_RATIO_WARN = 0.5


def _warn_rytov_validity(func_name, period, wavelength, order, eps_list):
    """Warn when (period, wavelength) sits outside the homogenization
    regime the Rytov expansion is valid in (v5.17, audit P3-24)."""
    if period is None or wavelength is None:
        return
    p, wl = float(period), float(wavelength)
    if p <= 0 or wl <= 0:
        return
    ratio = p / wl
    if order == 2 and ratio > _RYTOV_ORDER2_RATIO_WARN:
        warnings.warn(
            f"{func_name}: period/wavelength = {ratio:.3g} exceeds "
            f"{_RYTOV_ORDER2_RATIO_WARN} -- the order=2 "
            f"(period/wavelength)^2 correction is an asymptotic term "
            f"valid only deep sub-wavelength and can drive the "
            f"effective tensor far outside the constituent bounds "
            f"here.  Use a rigorous grating solver (rcwa/pmm) at this "
            f"period.", UserWarning, stacklevel=3)
        return
    n_max = max(abs(np.sqrt(complex(e))) for e in eps_list)
    if p > wl / max(n_max, 1e-300):
        warnings.warn(
            f"{func_name}: period = {p:.3g} m exceeds "
            f"wavelength/max|n| = {wl / n_max:.3g} m -- non-zero "
            f"diffraction orders propagate inside the densest "
            f"constituent, so the homogenized (EMT) tensor is "
            f"resonance-blind here.  Use a rigorous grating solver "
            f"(rcwa/pmm) at this period.", UserWarning, stacklevel=3)

__all__ = [
    "rytov_tensor",
    "rytov_segments_tensor",
    "maxwell_garnett",
    "bruggeman",
]


# =========================================================================== #
# 1-D lamellar: Rytov uniaxial tensor
# =========================================================================== #

def rytov_tensor(
    eps_ridge: complex,
    eps_groove: complex,
    fill: float,
    *,
    period: Optional[float] = None,
    wavelength: Optional[float] = None,
    order: int = 0,
) -> np.ndarray:
    """Effective ``(3, 3)`` UNIAXIAL permittivity of a sub-wavelength binary
    1-D grating (period along ``x``, invariant along ``y``), by Rytov
    effective-medium theory.

    Parameters
    ----------
    eps_ridge, eps_groove : complex
        Ridge / groove relative permittivity (PUBLIC ``Im > 0``).  Pass
        ``n**2`` if you have an index.
    fill : float
        Ridge fraction (duty cycle) in ``(0, 1)``.
    period, wavelength : float, optional
        Grating period and vacuum wavelength [m] -- REQUIRED for ``order=2``
        (the correction scales as ``(period/wavelength)^2``); ignored for
        ``order=0`` (though if BOTH are supplied a validity check still
        runs).  v5.17 (audit P3-24): a ``UserWarning`` is emitted when
        ``order=2`` is used with ``period/wavelength > 0.5``, or when a
        supplied period exceeds ``wavelength/max|n|`` (diffraction
        orders propagate -- EMT is resonance-blind there).
    order : {0, 2}, optional
        ``0`` (default) = zeroth-order Rytov (the exact ``period -> 0``
        limit -- recommended for slab work); ``2`` = with the second-order
        ``(period/lambda)^2`` bulk-index correction (needs ``period`` and
        ``wavelength``), which sharpens the effective index at moderate
        period but does NOT improve the inherent ``O(period/lambda)`` slab
        interface error and can overshoot the slab Jones -- use for a
        dispersion estimate, not as a higher-order slab convergence.

    Returns
    -------
    (3, 3) complex ndarray
        ``diag(eps_perp, eps_par, eps_par)`` -- ``x`` (across the lamellae)
        carries the harmonic mean, ``y`` / ``z`` (along the lamellae) the
        arithmetic mean.  Feed directly to ``BerremanStack.add_layer(eps=...)``.
    """
    # Backend-generic: a traced (JAX) ridge/groove eps or fill flows through
    # differentiably (via array_namespace); a concrete call stays NumPy and
    # byte-identical to the pre-JAX path.
    xp = array_namespace(eps_ridge, eps_groove, fill)
    traced = (is_jax_array(eps_ridge) or is_jax_array(eps_groove)
              or is_jax_array(fill))
    cj = xp.complex128
    if traced:
        f = fill                                        # keep raw so grad flows
    else:
        f = float(fill)
        if not (0.0 < f < 1.0):
            raise ValueError(f"rytov_tensor: fill must be in (0, 1), got {f}.")
    if order not in (0, 2):
        raise ValueError(f"rytov_tensor: order must be 0 or 2, got {order}.")
    ea = xp.asarray(eps_ridge, dtype=cj)
    eb = xp.asarray(eps_groove, dtype=cj)
    if not traced:
        # v5.17 audit (P3-24): validity-domain diagnostic -- warn (do not
        # raise) outside the homogenization regime (concrete eps only).
        _warn_rytov_validity("rytov_tensor", period, wavelength, order,
                             (complex(eps_ridge), complex(eps_groove)))
    eps_par0 = f * ea + (1.0 - f) * eb                 # arithmetic (TE / par)
    eps_perp0 = 1.0 / (f / ea + (1.0 - f) / eb)        # harmonic (TM / perp)
    eps_par, eps_perp = eps_par0, eps_perp0
    if order == 2:
        if period is None or wavelength is None:
            raise ValueError(
                "rytov_tensor: order=2 needs period and wavelength.")
        u2 = (float(period) / float(wavelength)) ** 2   # period/wl stay concrete
        pref = (np.pi ** 2 / 3.0) * u2 * f ** 2 * (1.0 - f) ** 2
        # Lalanne & Lemercier-Lalanne 1996, eq. for the equivalent uniaxial
        # indices: the TE (parallel) correction is additive in eps; the TM
        # (perp) correction carries the eps_perp0^3 eps_par0 weight.
        eps_par = eps_par0 + pref * (ea - eb) ** 2
        eps_perp = (eps_perp0
                    + pref * (1.0 / ea - 1.0 / eb) ** 2
                    * eps_perp0 ** 3 * eps_par0)
    return xp.diag(xp.stack([eps_perp, eps_par, eps_par]))


def rytov_segments_tensor(
    segments: List[Tuple[float, Union[complex, np.ndarray]]],
    *,
    period: Optional[float] = None,
    wavelength: Optional[float] = None,
    order: int = 0,
) -> np.ndarray:
    """Zeroth-order Rytov uniaxial tensor of an ARBITRARY multi-region 1-D
    cell -- the N-segment generalization of :func:`rytov_tensor`.

    ``segments`` is ``[(width_fraction, eps), ...]`` (fractions sum to 1; each
    ``eps`` scalar).  ``eps_par = sum f_i eps_i`` (arithmetic),
    ``eps_perp = 1 / sum f_i / eps_i`` (harmonic).  ``order=2`` is only defined
    for the binary case (use :func:`rytov_tensor`); a multi-region ``order=2``
    raises."""
    if order not in (0,):
        raise ValueError(
            "rytov_segments_tensor: only order=0 is defined for multi-region "
            "cells; use rytov_tensor for the binary second-order correction.")
    w_list = [w for w, _e in segments]
    e_list = [e for _w, e in segments]
    xp = array_namespace(*w_list, *e_list)
    cj = xp.complex128
    traced = any(is_jax_array(v) for v in (*w_list, *e_list))
    ws = xp.stack([xp.asarray(w, dtype=cj) for w in w_list])
    es = xp.stack([xp.asarray(e, dtype=cj) for e in e_list])
    if not traced:
        wsum = float(np.real(np.asarray([complex(w) for w in w_list]).sum()))
        if abs(wsum - 1.0) > 1e-6:
            raise ValueError(
                f"rytov_segments_tensor: widths must sum to 1, got {wsum}.")
        # v5.17 audit (P3-24): validity-domain diagnostic (concrete eps only).
        _warn_rytov_validity("rytov_segments_tensor", period, wavelength,
                             order, [complex(e) for e in e_list])
    eps_par = xp.sum(ws * es)                           # arithmetic
    eps_perp = 1.0 / xp.sum(ws / es)                    # harmonic
    return xp.diag(xp.stack([eps_perp, eps_par, eps_par]))


# =========================================================================== #
# 2-D inclusion arrays: scalar mixing rules (approximate)
# =========================================================================== #

def maxwell_garnett(eps_host: complex, eps_inclusion: complex,
                    fill: float, *, geometry: str = "sphere") -> complex:
    """Maxwell-Garnett effective (scalar) permittivity of a DILUTE array of
    inclusions ``eps_inclusion`` (volume / area fraction ``fill``) in a host
    ``eps_host`` -- the coarse isotropic screen for a 2-D pillar / hole array.

    ``geometry='sphere'`` (3-D spheres, depolarization 1/3) or
    ``'cylinder'`` (2-D cylinders, field PERPENDICULAR to the axis,
    depolarization 1/2 -- the natural choice for vertical pillars at normal
    incidence).  Accurate only in the dilute, deeply sub-wavelength limit; it
    is asymmetric in host vs inclusion (swap them and the value changes)."""
    xp = array_namespace(eps_host, eps_inclusion, fill)
    cj = xp.complex128
    traced = (is_jax_array(eps_host) or is_jax_array(eps_inclusion)
              or is_jax_array(fill))
    eh = xp.asarray(eps_host, dtype=cj)
    ei = xp.asarray(eps_inclusion, dtype=cj)
    f = fill if traced else float(fill)
    if not traced and not (0.0 <= f <= 1.0):
        raise ValueError(f"maxwell_garnett: fill must be in [0, 1], got {f}.")
    g = {"sphere": 2.0, "cylinder": 1.0}.get(geometry)
    if g is None:
        raise ValueError(
            f"maxwell_garnett: geometry must be 'sphere' or 'cylinder', got "
            f"{geometry!r}.")
    # (eps - eh)/(eps + g eh) = f (ei - eh)/(ei + g eh)
    beta = (ei - eh) / (ei + g * eh)
    eff = eh * (1.0 + g * f * beta) / (1.0 - f * beta)
    return eff if traced else complex(eff)


def bruggeman(eps_a: complex, eps_b: complex, fill_a: float,
              *, geometry: str = "sphere") -> complex:
    """Bruggeman (symmetric, self-consistent) effective permittivity of a
    two-phase mixture -- ``fill_a`` of ``eps_a`` and ``1 - fill_a`` of
    ``eps_b``, both treated as grains in the effective medium (no host /
    inclusion asymmetry, so it spans the full fill range and a percolation
    threshold).  ``geometry='sphere'`` (1/3) or ``'cylinder'`` (1/2)."""
    traced = (is_jax_array(eps_a) or is_jax_array(eps_b)
              or is_jax_array(fill_a))
    g = {"sphere": 2.0, "cylinder": 1.0}.get(geometry)
    if g is None:
        raise ValueError(
            f"bruggeman: geometry must be 'sphere' or 'cylinder', got "
            f"{geometry!r}.")
    if traced:
        # Differentiable twin: solve the same quadratic in jnp and pick the
        # passive root via a nudge-continued reference (well-defined for lossy
        # AND lossless constituents), selecting the nearer EXACT root with
        # ``where`` -- a measure-zero branch predicate, so grad flows through
        # the chosen exact root (fully differentiable in ea/eb/f).
        xp = array_namespace(eps_a, eps_b, fill_a)
        cj = xp.complex128
        ea = xp.asarray(eps_a, dtype=cj)
        eb = xp.asarray(eps_b, dtype=cj)
        f = fill_a
        b = ((1.0 - f) - g * f) * ea + (f - g * (1.0 - f)) * eb
        disc = xp.sqrt(b * b + 4.0 * g * ea * eb)
        r1 = (-b + disc) / (2.0 * g)
        r2 = (-b - disc) / (2.0 * g)
        na = ea + 1j * 1e-8 * (1.0 + xp.abs(ea))
        nb = eb + 1j * 1e-8 * (1.0 + xp.abs(eb))
        bn = ((1.0 - f) - g * f) * na + (f - g * (1.0 - f)) * nb
        dn = xp.sqrt(bn * bn + 4.0 * g * na * nb)
        ref = (-bn + dn) / (2.0 * g)
        alt = (-bn - dn) / (2.0 * g)
        ref = xp.where(xp.imag(alt) > xp.imag(ref), alt, ref)
        return xp.where(xp.abs(r1 - ref) <= xp.abs(r2 - ref), r1, r2)
    # ---- concrete NumPy path (byte-identical to the pre-JAX behavior) -------
    ea, eb, f = _C(eps_a), _C(eps_b), float(fill_a)
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"bruggeman: fill_a must be in [0, 1], got {f}.")
    # f (ea - e)/(ea + g e) + (1-f)(eb - e)/(eb + g e) = 0 -> quadratic in e.
    # Solve the standard closed form, pick the Im>=0 root.
    b = ((1.0 - f) - g * f) * ea + (f - g * (1.0 - f)) * eb  # linear coeff
    # Closed form: g e^2 + b e - ea eb = 0  (from clearing denominators)
    disc = np.sqrt(b * b + 4.0 * g * ea * eb)
    r1 = (-b + disc) / (2.0 * g)
    r2 = (-b - disc) / (2.0 * g)
    # Physical root: the passive branch, Im(e) >= 0 (Im(eps) > 0 = loss).
    # When Im does not discriminate (lossless constituents -> both roots
    # real, e.g. metal/dielectric outside the resonance band), continue the
    # branch from an infinitesimal-loss nudge of the constituents.
    scale = abs(r1) + abs(r2) + 1.0
    if abs(r1.imag - r2.imag) <= 1e-9 * scale:
        na = ea + 1j * 1e-8 * (1.0 + abs(ea))
        nb = eb + 1j * 1e-8 * (1.0 + abs(eb))
        bn = ((1.0 - f) - g * f) * na + (f - g * (1.0 - f)) * nb
        dn = np.sqrt(bn * bn + 4.0 * g * na * nb)
        ref = (-bn + dn) / (2.0 * g)
        alt = (-bn - dn) / (2.0 * g)
        if alt.imag > ref.imag:
            ref = alt
        cand = r1 if abs(r1 - ref) <= abs(r2 - ref) else r2
    else:
        cand = r1 if r1.imag > r2.imag else r2
    return complex(cand)
