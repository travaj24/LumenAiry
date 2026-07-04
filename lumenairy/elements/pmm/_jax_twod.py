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
is CPU-only; x64 is required.  Host-side guards that need concrete values
degrade gracefully (audit P2-11, mirroring ``_jax_stack2d``): the
propagating-incidence raise and the Wood-anomaly grazing-wavelength nudge run
only when ``wavelength`` / ``theta`` / ``phi`` / ``n_superstrate`` are
concrete; a traced source value skips them (gradients are valid only BETWEEN
order cutoffs).  Measured gradient fidelity (AD vs central FD):
material eps ~3e-9 relative at BOTH a degenerate centered square and an
asymmetric pillar; depth/wavelength ~5e-10; theta at oblique ~1e-9; theta at
EXACTLY normal incidence on a CENTERED square returns the clean symmetry zero
(~4e-15 -- no degenerate-gauge artifact, unlike the 1-D twin's documented
~1e-3 normal-incidence wobble).
"""
from __future__ import annotations

import threading
from collections import OrderedDict

import numpy as np

from ...backend import is_jax_array  # noqa: F401  (re-exported for the dispatch)

_C = np.complex128
# LRU-bounded (audit P3-28): the key includes the CONTINUOUS pillar bounds /
# cell-layout bytes, so an external geometry sweep or FD geometry
# optimization inserts one multi-MB projected-operator entry PER GEOMETRY
# (~3 MB at toy sizes, ~9 MB at production degree=5 / n_orders=7).  A single
# solve touches exactly ONE geometry entry, so 16 slots keep every realistic
# repeated-geometry hit while capping worst-case residency at ~150 MB.
# Follows the propagation.py / wrapper_merits.py OrderedDict-LRU convention
# (move_to_end on hit, popitem(last=False) evict, all under the cache lock).
_STATIC_CACHE: 'OrderedDict[tuple, dict]' = OrderedDict()
_STATIC_CACHE_SIZE = 16
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


def _concrete(v):
    """float(v) when concrete, else None (tracers have no concrete value)."""
    try:
        return float(np.real(np.asarray(v)))
    except Exception:
        return None


def _host_incidence_guard(fn_name, st, period_x, period_y, wavelength, theta,
                          phi, n_superstrate, n_substrate, eps_candidates):
    """Host-side propagating-incidence check + Wood-anomaly wavelength nudge,
    run ONLY when the source values are concrete -- verbatim the sibling
    multilayer twin (``_jax_stack2d._pmm_stack2d_solve_jax``) and the NumPy
    core (``twod._pmm2d_solve_core``).  Returns the (possibly nudged)
    wavelength to solve at.

    A TRACED ``wavelength`` / ``theta`` / ``phi`` / ``n_superstrate`` (a
    ``jax.grad`` / ``jax.jit`` Tracer has no concrete host value) SKIPS both
    guards: the wavelength passes through un-nudged (the documented
    Wood-anomaly caveat -- gradients are valid only BETWEEN order cutoffs) and
    grazing/evanescent incidence is not rejected.  ``eps_candidates`` holds
    the PUBLIC layer permittivities (scalars or arrays); traced entries are
    dropped from the nudge's constituent list (``Re()`` is what the nudge
    uses, and it is conjugation-invariant)."""
    wl_c = _concrete(wavelength)
    th_c = _concrete(theta)
    ph_c = _concrete(phi)
    nsup_c = _concrete(n_superstrate)
    if None in (wl_c, th_c, ph_c, nsup_c):
        return wavelength
    from .twod import (
        _grazing_safe_wavelength,
        _require_propagating_incidence,
    )
    kx0_c = nsup_c * np.sin(th_c) * np.cos(ph_c)
    ky0_c = nsup_c * np.sin(th_c) * np.sin(ph_c)
    eps_reals = []
    for v in (n_superstrate, n_substrate):
        try:
            eps_reals.append(complex(np.conj(complex(np.asarray(v)) ** 2)))
        except Exception:
            pass
    for e in eps_candidates:
        try:
            eps_reals += [complex(np.conj(complex(x)))
                          for x in np.asarray(e).ravel()]
        except Exception:
            pass
    _require_propagating_incidence(
        fn_name, complex(np.conj(complex(nsup_c) ** 2)),
        kx0_c ** 2 + ky0_c ** 2)
    return _grazing_safe_wavelength(wl_c, kx0_c, ky0_c, st["order_x"],
                                    st["order_y"], period_x, period_y,
                                    eps_reals)


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
            _STATIC_CACHE.move_to_end(key)   # LRU: refresh recency
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
        while len(_STATIC_CACHE) > _STATIC_CACHE_SIZE:
            _STATIC_CACHE.popitem(last=False)
    return out


def _static_prep_cell(period_x, period_y, layout, degree, n_el, grade,
                      n_orders):
    """Geometry constants for an arbitrary axis-aligned MULTI-REGION cell:
    the wall layout comes from a CONCRETE region-id grid (a traced eps_cell
    cannot define walls), and each region gets its partition-of-unity weight
    vector ``Wreg[r]`` so a traced region-value vector enters linearly as
    ``eps_nodal = eps_values @ Wreg``."""
    lay = np.ascontiguousarray(np.asarray(layout, dtype=np.int64))
    key = (period_x, period_y, lay.tobytes(), lay.shape, degree, n_el, grade,
           n_orders)
    with _STATIC_CACHE_LOCK:
        hit = _STATIC_CACHE.get(key)
        if hit is not None:
            _STATIC_CACHE.move_to_end(key)   # LRU: refresh recency
    if hit is not None:
        return hit
    from .twod import (
        _axis_elem_counts,
        _build_axis,
        _cell_to_walls_tile,
        _projectors,
        _validate_cell_orders,
    )
    x_walls, y_walls, tile = _cell_to_walls_tile(
        lay.astype(complex), period_x, period_y, "pmm_efficiency_2d_cell")
    el_x = _axis_elem_counts(period_x, x_walls, degree, n_el,
                             "pmm_efficiency_2d_cell", "x")
    el_y = _axis_elem_counts(period_y, y_walls, degree, n_el,
                             "pmm_efficiency_2d_cell", "y")
    _validate_cell_orders("pmm_efficiency_2d_cell", n_orders, degree, el_x,
                          el_y)
    ax = _build_axis(period_x, x_walls, degree, el_x, grade)
    ay = _build_axis(period_y, y_walls, degree, el_y, grade)
    ox = np.arange(-n_orders, n_orders + 1)
    oy = np.arange(-n_orders, n_orders + 1)
    Tp, Tpinv = _projectors(ax, ay, ox, oy)
    mdx = np.diag(ax["M"])
    mdy = np.diag(ay["M"])
    Mdiag = np.kron(mdy, mdx)
    # region ids per strip cell + per-region weight rows
    region_ids = np.real(tile).astype(np.int64)
    uniq = np.unique(lay)
    Wreg = np.zeros((uniq.size, Mdiag.size))
    nsx, nsy = region_ids.shape
    for sx in range(nsx):
        for sy in range(nsy):
            r = int(np.where(uniq == region_ids[sx, sy])[0][0])
            Wreg[r] += np.real(
                np.kron(np.diag(ay["Mtile"][sy]),
                        np.diag(ax["Mtile"][sx])) / Mdiag)
    # first-pixel gather indices per region (to read traced values)
    first_idx = []
    for u in uniq:
        i, j = np.argwhere(lay == u)[0]
        first_idx.append((int(i), int(j)))
    Minv = np.diag(1.0 / Mdiag)
    Gx0 = -1j * (Minv @ np.kron(ay["M"], ax["D"]))
    Gy0 = -1j * (Minv @ np.kron(ay["D"], ax["M"]))
    out = dict(
        Tp=Tp, Tpinv=Tpinv, Wreg=Wreg, first_idx=first_idx,
        Gx0F=Tp @ Gx0 @ Tpinv, Gy0F=Tp @ Gy0 @ Tpinv,
        IprojF=Tp @ Tpinv,
        order_x=np.tile(ox, len(oy)), order_y=np.repeat(oy, len(ox)))
    with _STATIC_CACHE_LOCK:
        _STATIC_CACHE[key] = out
        while len(_STATIC_CACHE) > _STATIC_CACHE_SIZE:
            _STATIC_CACHE.popitem(last=False)
    return out


def _pmm_efficiency_2d_cell_jax(period_x, period_y, eps_cell, region_layout,
                                n_substrate, n_superstrate, depth,
                                wavelength, *, degree, elements_per_strip,
                                grade, polarization, theta, phi, n_orders,
                                formulation):
    """JAX twin for the MULTI-REGION cell (v5.14 roadmap item 1): a CONCRETE
    ``region_layout`` (int grid, same shape as ``eps_cell``) defines the walls;
    the traced ``eps_cell`` provides the region VALUES (read at each region's
    first pixel -- pixels within a region must share the traced value, which
    cannot be checked under trace)."""
    import jax.numpy as jnp
    st = _static_prep_cell(float(period_x), float(period_y), region_layout,
                           int(degree), int(elements_per_strip), bool(grade),
                           int(n_orders))
    # concrete-only incidence + Wood-anomaly guards (audit P2-11; a traced
    # source value skips them -- see _host_incidence_guard)
    wavelength = _host_incidence_guard(
        "pmm_efficiency_2d_cell", st, float(period_x), float(period_y),
        wavelength, theta, phi, n_superstrate, n_substrate, (eps_cell,))
    cj = jnp.complex128
    eps_c = jnp.asarray(eps_cell).astype(cj)
    eps_regions = jnp.stack([eps_c[i, j] for (i, j) in st["first_idx"]])
    eps_regions = jnp.conj(eps_regions)          # loss bridge
    Wreg = jnp.asarray(st["Wreg"])
    eps_nodal = eps_regions @ Wreg
    inv_nodal = (1.0 / eps_regions) @ Wreg
    eps_sup = jnp.conj(jnp.asarray(n_superstrate).astype(cj) ** 2)
    eps_sub = jnp.conj(jnp.asarray(n_substrate).astype(cj) ** 2)
    return _scalar_jax_tail(jnp, st, eps_nodal, inv_nodal, eps_sup, eps_sub,
                            period_x, period_y, depth, wavelength,
                            polarization, theta, phi, formulation)


def _pmm_efficiency_2d_jax(period_x, period_y, eps_pillar, eps_host,
                           x_bounds, y_bounds, n_substrate, n_superstrate,
                           depth, wavelength, *, degree, elements_per_strip,
                           grade, polarization, theta, phi, n_orders,
                           formulation):
    import jax.numpy as jnp
    st = _static_prep(float(period_x), float(period_y),
                      float(x_bounds[0]), float(x_bounds[1]),
                      float(y_bounds[0]), float(y_bounds[1]),
                      int(degree), int(elements_per_strip), bool(grade),
                      int(n_orders))
    # concrete-only incidence + Wood-anomaly guards (audit P2-11; a traced
    # source value skips them -- see _host_incidence_guard)
    wavelength = _host_incidence_guard(
        "pmm_efficiency_2d", st, float(period_x), float(period_y),
        wavelength, theta, phi, n_superstrate, n_substrate,
        (eps_pillar, eps_host))
    cj = jnp.complex128
    w = jnp.asarray(st["w"])
    # loss-convention bridge (PUBLIC Im>0 -> internal exp(+iwt))
    eps_p = jnp.conj(jnp.asarray(eps_pillar).astype(cj))
    eps_h = jnp.conj(jnp.asarray(eps_host).astype(cj))
    eps_sup = jnp.conj(jnp.asarray(n_superstrate).astype(cj) ** 2)
    eps_sub = jnp.conj(jnp.asarray(n_substrate).astype(cj) ** 2)
    eps_nodal = eps_h + (eps_p - eps_h) * w
    inv_nodal = 1.0 / eps_h + (1.0 / eps_p - 1.0 / eps_h) * w
    return _scalar_jax_tail(jnp, st, eps_nodal, inv_nodal, eps_sup, eps_sub,
                            period_x, period_y, depth, wavelength,
                            polarization, theta, phi, formulation)


def _scalar_jax_tail(jnp, st, eps_nodal, inv_nodal, eps_sup, eps_sub,
                     period_x, period_y, depth, wavelength, polarization,
                     theta, phi, formulation):
    """Shared traced solve: projected operators from the nodal eps vectors,
    the 2Nf eig (custom-VJP), the cascade, and the flux tail."""
    from ..rcwa import _jax_eig_stable, _require_jax_x64
    from ..rcwa._core import (
        _interface_smatrix,
        _propagation_star,
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

    Tp = jnp.asarray(st["Tp"], cj)
    Tpinv = jnp.asarray(st["Tpinv"], cj)
    Gx0F = jnp.asarray(st["Gx0F"], cj)
    Gy0F = jnp.asarray(st["Gy0F"], cj)
    IprojF = jnp.asarray(st["IprojF"], cj)
    order_x = jnp.asarray(st["order_x"])
    order_y = jnp.asarray(st["order_y"])
    Nf = st["order_x"].size

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
    EpsF = (Tp * eps_nodal[None, :].astype(cj)) @ Tpinv
    EinvF = (Tp * inv_nodal[None, :].astype(cj)) @ Tpinv
    EpnF = (Tp * (1.0 / inv_nodal)[None, :].astype(cj)) @ Tpinv
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
    S = _propagation_star(S, lam_l, k0 * depth)
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
