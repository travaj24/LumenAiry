"""
lumenairy.elements.pmm._jax_twod -- JAX twin of the 2-D hybrid PMM (Phase 7).
=============================================================================

Differentiable forward solve for :func:`pmm_efficiency_2d` (the PILLAR entry)
on JAX inputs: gradients flow through ``eps_pillar`` / ``eps_host`` (real and
imaginary parts), ``n_substrate`` / ``n_superstrate``, ``depth``,
``wavelength`` and ``theta`` / ``phi``.  The pillar BOUNDS, ``degree``,
``elements_per_strip`` and ``n_orders`` are STATIC (the spectral-element wall
layout cannot be traced -- the same scope split as the validated 1-D twin,
which traces material values on a fixed topology).

Why only the pillar entry: the cell path derives its walls by COMPARING pixel
values (`_cell_to_walls_tile`), which is data-dependent control flow -- a
traced ``eps_cell`` cannot define geometry.  For differentiable arbitrary
pixel cells use :func:`rcwa_efficiency_2d` / :func:`rcwa_jones_2d` (Fourier
sampling differentiates the pixel VALUES directly).

The key structural facts that make this twin cheap (Phase-7 pressure test):
the GLL mass matrices are DIAGONAL, so every eps operator reduces to a nodal
VECTOR that is LINEAR in the region permittivities (``eps_nodal = eps_h + (
eps_p - eps_h) * w`` with ``w`` the constant pillar partition-of-unity
weight), and the Fourier projectors ``Tp``/``Tpinv`` + the unit derivative
operators are geometry-only CONSTANTS precomputed in NumPy and frozen into the
trace.  Only the small projected eig (``2*Nf``, via the shared custom-VJP
``rcwa._jax_eig_stable``) and the S-matrix cascade are traced.

Caveats (mirroring the 1-D twin): ``stabilize`` is rejected (host-side degree
scan); the uniform-pillar analytic shortcut is NOT taken (a traced eps cannot
branch -- a uniform cell solves through the structured path); ``jnp.linalg.eig``
is CPU-only; x64 is required.  Measured gradient fidelity (AD vs central FD):
material eps ~3e-9 relative at BOTH a degenerate centered square and an
asymmetric pillar; depth/wavelength ~5e-10; theta at oblique ~1e-9; theta at
EXACTLY normal incidence on a CENTERED square returns the clean symmetry zero
(~4e-15 -- no degenerate-gauge artifact, unlike the 1-D twin's documented
~1e-3 normal-incidence wobble).
"""
from __future__ import annotations

import threading

import numpy as np

from ...backend import is_jax_array  # noqa: F401  (re-exported for the dispatch)

_C = np.complex128
_STATIC_CACHE = {}
_STATIC_CACHE_LOCK = threading.Lock()


def _clear_jax_twod_caches():
    """Registry hook: drop the geometry-constant cache."""
    with _STATIC_CACHE_LOCK:
        _STATIC_CACHE.clear()


# Enroll with the central cache registry (the v4.16.0 contract: every
# module-level cache is clearable via the global "clear all caches" path).
try:
    from ..._cache_registry import (
        register_cache_clearer as _register_cache_clearer,
    )
    _register_cache_clearer("pmm_jax_twod_static", _clear_jax_twod_caches)
except ImportError:  # pragma: no cover - registry always present in-tree
    pass


def _static_prep(period_x, period_y, x0, x1, y0, y1, degree, n_el, grade,
                 n_orders):
    """Geometry-only constants (NumPy): the Fourier-projected unit-derivative
    operators, the projected identity, the pillar partition-of-unity weight
    vector, and the order set.  Cached on the static key (lock-guarded for
    concurrent reader-writer safety)."""
    key = (period_x, period_y, x0, x1, y0, y1, degree, n_el, grade, n_orders)
    with _STATIC_CACHE_LOCK:
        hit = _STATIC_CACHE.get(key)
    if hit is not None:
        return hit
    from .twod import _build_axis, _projectors
    ax = _build_axis(period_x, [x0, x1], degree, n_el, grade)
    ay = _build_axis(period_y, [y0, y1], degree, n_el, grade)
    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    Tp, Tpinv = _projectors(ax, ay, ox, oy)
    # diagonal masses -> the pillar weight w = diag(kron(MtY[1], MtX[1]))/Mdiag
    mdx = np.diag(ax["M"])
    mdy = np.diag(ay["M"])
    Mdiag = np.kron(mdy, mdx)
    w = np.kron(np.diag(ay["Mtile"][1]), np.diag(ax["Mtile"][1])) / Mdiag
    # unit derivative operators (k0-free), projected
    Minv = np.diag(1.0 / Mdiag)
    Gx0 = -1j * (Minv @ np.kron(ay["M"], ax["D"]))
    Gy0 = -1j * (Minv @ np.kron(ay["D"], ax["M"]))
    out = dict(
        Tp=Tp, Tpinv=Tpinv, w=np.real(w),
        Gx0F=Tp @ Gx0 @ Tpinv, Gy0F=Tp @ Gy0 @ Tpinv,
        IprojF=Tp @ Tpinv,
        order_x=np.tile(ox, len(oy)), order_y=np.repeat(oy, len(ox)))
    with _STATIC_CACHE_LOCK:
        _STATIC_CACHE[key] = out
    return out


def _pmm_efficiency_2d_jax(period_x, period_y, eps_pillar, eps_host,
                           x_bounds, y_bounds, n_substrate, n_superstrate,
                           depth, wavelength, *, degree, elements_per_strip,
                           grade, polarization, theta, phi, n_orders,
                           formulation):
    import jax.numpy as jnp

    from ..rcwa import _jax_eig_stable, _require_jax_x64
    from ..rcwa._core import (
        _interface_smatrix,
        _propagation_smatrix,
        _redheffer_star,
    )
    _require_jax_x64("pmm_efficiency_2d")
    cj = jnp.complex128

    def _sqrt_decay(x):
        r = jnp.sqrt(x.astype(cj))
        on_cut = r.real == 0
        return jnp.where(on_cut & (r.imag < 0), -r, r)

    def _inv_lam(lam):
        safe = jnp.where(jnp.abs(lam) < 1e-12, 1e-12, lam)
        return 1.0 / safe

    def _kz_fwd(eps, kx, ky):
        val = jnp.sqrt((eps - kx ** 2 - ky ** 2).astype(cj))
        return jnp.where(val.imag < 0.0, -val, val)

    st = _static_prep(float(period_x), float(period_y),
                      float(x_bounds[0]), float(x_bounds[1]),
                      float(y_bounds[0]), float(y_bounds[1]),
                      int(degree), int(elements_per_strip), bool(grade),
                      int(n_orders))
    Tp = jnp.asarray(st["Tp"], cj)
    Tpinv = jnp.asarray(st["Tpinv"], cj)
    w = jnp.asarray(st["w"])
    Gx0F = jnp.asarray(st["Gx0F"], cj)
    Gy0F = jnp.asarray(st["Gy0F"], cj)
    IprojF = jnp.asarray(st["IprojF"], cj)
    order_x = jnp.asarray(st["order_x"])
    order_y = jnp.asarray(st["order_y"])
    Nf = st["order_x"].size

    # loss-convention bridge (PUBLIC Im>0 -> internal exp(+iwt))
    eps_p = jnp.conj(jnp.asarray(eps_pillar).astype(cj))
    eps_h = jnp.conj(jnp.asarray(eps_host).astype(cj))
    eps_sup = jnp.conj(jnp.asarray(n_superstrate).astype(cj) ** 2)
    eps_sub = jnp.conj(jnp.asarray(n_substrate).astype(cj) ** 2)

    k0 = 2.0 * jnp.pi / wavelength
    nre = jnp.real(jnp.sqrt(eps_sup))
    kx0 = nre * jnp.sin(theta) * jnp.cos(phi)
    ky0 = nre * jnp.sin(theta) * jnp.sin(phi)
    kxv = kx0 + order_x * (wavelength / period_x)
    kyv = ky0 + order_y * (wavelength / period_y)

    def _homog(eps):
        Kx = jnp.diag(kxv.astype(cj))
        Ky = jnp.diag(kyv.astype(cj))
        kz = _kz_fwd(eps, kxv, kyv)
        lam = _sqrt_decay(-jnp.concatenate([kz, kz]) ** 2)
        eps_I = eps * jnp.eye(Nf, dtype=cj)
        Q = jnp.block([[Kx @ Ky, eps_I - Kx @ Kx],
                       [Ky @ Ky - eps_I, -Ky @ Kx]])
        W = jnp.eye(2 * Nf, dtype=cj)
        V = Q @ jnp.diag(_inv_lam(lam))
        return W, V, lam

    Wsup, Vsup, _ls = _homog(eps_sup)
    Wsub, Vsub, _lb = _homog(eps_sub)

    # layer operators: nodal vectors LINEAR in the traced eps, sandwiched by
    # the constant projectors (Tp * v) @ Tpinv == Tp @ diag(v) @ Tpinv
    eps_nodal = eps_h + (eps_p - eps_h) * w
    inv_nodal = 1.0 / eps_h + (1.0 / eps_p - 1.0 / eps_h) * w
    EpsF = (Tp * eps_nodal[None, :]) @ Tpinv
    EinvF = (Tp * inv_nodal[None, :]) @ Tpinv
    EpnF = (Tp * (1.0 / inv_nodal)[None, :]) @ Tpinv
    GxF = Gx0F / k0 + kx0 * IprojF
    GyF = Gy0F / k0 + ky0 * IprojF

    I_F = jnp.eye(Nf, dtype=cj)
    EPS_normal = EpnF if formulation == "li" else EpsF
    Q = jnp.block([[GxF @ GyF, EpsF - GxF @ GxF],
                   [GyF @ GyF - EPS_normal, -GyF @ GxF]])
    EPS_inv = EinvF if formulation == "li" else jnp.linalg.inv(EpsF)
    P = jnp.block([[GxF @ EPS_inv @ GyF, I_F - GxF @ EPS_inv @ GxF],
                   [GyF @ EPS_inv @ GyF - I_F, -GyF @ EPS_inv @ GxF]])
    eig = _jax_eig_stable()
    lam2, Wl = eig(P @ Q)
    lam_l = _sqrt_decay(lam2)
    Vl = Q @ Wl @ jnp.diag(_inv_lam(lam_l))

    S = _interface_smatrix(Wsup, Vsup, Wl, Vl)
    S = _redheffer_star(S, _propagation_smatrix(lam_l, k0 * depth))
    S = _redheffer_star(S, _interface_smatrix(Wl, Vl, Wsub, Vsub))
    S11, _S12, S21, _S22 = S

    # incident (0,0) plane wave -- tracer-safe TE/TM basis (the kt -> 0 limit
    # blends to the lab axes exactly like the concrete path's branch)
    kt = jnp.hypot(jnp.real(kx0), jnp.real(ky0))
    safe_kt = jnp.where(kt < 1e-12, 1.0, kt)
    axu = jnp.where(kt < 1e-12, 0.0, jnp.real(kx0) / safe_kt)
    ayu = jnp.where(kt < 1e-12, 0.0, jnp.real(ky0) / safe_kt)
    kz_inc = jnp.real(_kz_fwd(jnp.conj(eps_sup), kx0, ky0))
    if polarization == "te":
        ex0 = jnp.where(kt < 1e-12, 0.0, -ayu)
        ey0 = jnp.where(kt < 1e-12, 1.0, axu)
        einc_sq = jnp.asarray(1.0)
    else:
        ex0 = jnp.where(kt < 1e-12, 1.0, axu)
        ey0 = jnp.where(kt < 1e-12, 0.0, ayu)
        einc_sq = jnp.where(kt < 1e-12, 1.0,
                            1.0 + (kt / jnp.where(kz_inc == 0, 1.0,
                                                  kz_inc)) ** 2)
    delta = ((st["order_x"] == 0) & (st["order_y"] == 0)).astype(_C)
    delta = jnp.asarray(delta)
    cinc = jnp.concatenate([ex0 * delta, ey0 * delta])
    r = S11 @ cinc
    t = S21 @ cinc
    rx, ry = r[:Nf], r[Nf:]
    tx, ty = t[:Nf], t[Nf:]

    kz_ref_f = _kz_fwd(jnp.conj(eps_sup), kxv, kyv)
    kz_trn_f = _kz_fwd(jnp.conj(eps_sub), kxv, kyv)
    safe_r = jnp.where(jnp.abs(kz_ref_f) < 1e-12, 1.0, kz_ref_f)
    safe_t = jnp.where(jnp.abs(kz_trn_f) < 1e-12, 1.0, kz_trn_f)
    rz = -(kxv * rx + kyv * ry) / safe_r
    tz = -(kxv * tx + kyv * ty) / safe_t
    R = jnp.real(kz_ref_f / kz_inc) * (jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2
                                       + jnp.abs(rz) ** 2) / einc_sq
    T = jnp.real(kz_trn_f / kz_inc) * (jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
                                       + jnp.abs(tz) ** 2) / einc_sq
    R = jnp.where(jnp.real(kz_ref_f) > 0, jnp.real(R), 0.0)
    T = jnp.where(jnp.real(kz_trn_f) > 0, jnp.real(T), 0.0)
    orders2d = np.stack([st["order_x"], st["order_y"]], axis=1)
    return orders2d, R, T
