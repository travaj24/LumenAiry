"""Differential ray transfer -- per-beamlet / per-ray ABCD Jacobians.

The reusable primitive underneath the *per-surface* forms of the phase-space
propagators.  Along each base ray of a bundle it returns the ray-transfer
Jacobian ``[[A, B], [C, D]]`` (2x2 blocks) of the real (aberrated) trace, by
central finite differences.  The complex-beam-parameter propagators consume it:

* Gaussian Beam Decomposition (``propagators.gbd``): evolves each beamlet's
  tensor ``Q`` via the generalized Collins relation
  ``Q_out = (C + D Q)(A + B Q)^{-1}`` and its amplitude via
  ``1/sqrt(det(A + B Q))`` -- capturing off-axis astigmatism / higher-order
  aberrations that a single whole-system paraxial ABCD cannot.
* (future) Maslov phase-space: the same differential transfer supplies the
  Hessian propagation.

Phase space is **unreduced** ``(x, y, ux, uy)`` with slopes ``ux = L/N``,
``uy = M/N`` (matching ``propagators.gbd.apply_abcd_to_beamlets``).  The input
state is referenced to the FIRST surface's vertex plane; the OUTPUT reference
plane is chosen by the ``reference`` keyword (see below).  The finite-
difference Jacobian bakes Snell's law and the glass indices in automatically
(no reduced ``n*u`` bookkeeping), so the on-axis 2x2 meridional block
reproduces ``raytrace.system_abcd_prescription`` to ~1e-8.

The output reference plane
--------------------------
``reference='surface'`` (the default) returns the state and the Jacobian ON
the last surface, i.e. at ``z = sag(rho)`` -- the same plane
:func:`lumenairy.raytrace.trace` leaves its rays on, and the plane
:attr:`lumenairy.raytrace.TraceResult.image_rays` reports.
``reference='exit_vertex'`` returns them on the last surface's VERTEX plane
``z = 0``, the plane
:meth:`lumenairy.raytrace.TraceResult.at_exit_vertex` transfers a ray bundle
to.  The two differ by the last surface's sag along the ray and agree exactly
only when that surface is flat; on an N-SF11 R = +/-1.6 mm biconvex the
difference at the rim is 0.66 um of height and 7.8 waves of optical path at
633 nm.  A consumer that adds an image-side free-space leg of length
``z_image`` measured from the vertex plane -- which is what a "back focal
distance" is -- must ask for ``reference='exit_vertex'``; one that composes the
transfer with a following surface wants the default.  The library's
``'exit_vertex'`` consumers are ``propagators.fga`` (``_fga_core``,
``_fga_coarse``'s two paths and ``_caustic_zone``) and
``propagators.gbd.apply_prescription_persurface_to_beamlets`` on its local-
frame branch (WP-B12b, 2026-09-15 -- it previously carried an in-line
conic-sag copy of this projection); that function's ``world_output_plane``
branch measures its own leg from the last-surface intersection and keeps the
default.
:func:`_project_to_exit_vertex_plane` is the single implementation of the
projection for this module's 4x4 state; the ray-bundle-shaped sibling is
:func:`lumenairy.raytrace.exit_vertex.exit_vertex_transfer`.

Validated in ``tests/unit/test_gbd_feature_complete.py`` and
``tests/unit/test_audit2609_b12_fga_reference_plane.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from ._conic_core import (
    check_even_aspheric_powers,
    reflect_mirror,
    refract_snell,
)
from .trace import _make_bundle, trace


@dataclass
class DifferentialTransfer:
    """Result of :func:`ray_transfer_jacobian`.

    Attributes
    ----------
    jacobian : ndarray
        ``(N_rays, 4, 4)`` composite input->output ray-transfer Jacobian, OR
        ``(N_surfaces, N_rays, 4, 4)`` per-surface local transfers when
        ``per_surface=True``.  Row/col order is ``(x, y, ux, uy)``; the 2x2
        blocks are ``A = J[..., 0:2, 0:2]`` (dx_out/dx_in),
        ``B = J[..., 0:2, 2:4]``, ``C = J[..., 2:4, 0:2]``,
        ``D = J[..., 2:4, 2:4]``.
    x, y, ux, uy : ndarray
        ``(N_rays,)`` base-ray state with unreduced slopes, on the plane the
        producing call's ``reference`` keyword named.  With
        ``reference='surface'`` (the default) that is the LAST SURFACE -- the
        intersection point, at ``z = sag(rho)``; with
        ``reference='exit_vertex'`` it is that surface's vertex plane
        ``z = 0``.  The two differ by the last surface's sag along the ray
        (measured 7.8 waves of OPL and 0.66 um of height at the rim of an
        N-SF11 R = 1.6 mm biconvex at 633 nm, 0.0 on a flat last surface).
        The projection between them is
        ``x - sag*ux``, ``y - sag*uy``,
        ``opd - n_exit*sag*sqrt(1 + ux^2 + uy^2)``, which is the same step
        :meth:`lumenairy.raytrace.TraceResult.at_exit_vertex` applies to a
        ray bundle.  Every consumer that adds an image-side free-space leg
        asks for ``'exit_vertex'``: the four ``propagators.fga`` sites and
        ``propagators.gbd.apply_prescription_persurface_to_beamlets``'s
        local-frame branch.
    opd : ndarray
        ``(N_rays,)`` base-ray accumulated optical path length [m] to the
        same plane as ``x``.
    alive : ndarray of bool
        ``(N_rays,)`` False for rays that vignetted / TIR'd / missed.
    """
    jacobian: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    opd: np.ndarray
    alive: np.ndarray


def _slopes_to_dirs(ux, uy):
    inv = 1.0 / np.sqrt(1.0 + ux * ux + uy * uy)
    return ux * inv, uy * inv, inv


def _validate_reference(reference, fn_name):
    if reference not in ('surface', 'exit_vertex'):
        raise ValueError(
            f"{fn_name}: reference must be 'surface' (the state and the "
            f"Jacobian on the last surface, z = sag(rho) -- the plane "
            f"raytrace.trace leaves its rays on) or 'exit_vertex' (that "
            f"surface's vertex plane z = 0, the plane "
            f"TraceResult.at_exit_vertex transfers to); got "
            f"{reference!r}.")
    return reference


def _last_surface_sag_vanishes(surface) -> bool:
    """True when ``surface``'s sag is identically zero, so the exit-vertex
    projection is the identity and can be skipped BIT-for-bit.

    Structural (a property of the surface), not data-dependent: a flat conic
    base with no aspheric departure, no biconic y-branch, no freeform and no
    field-frame decenter / tilt / sag callable.  ``1e15`` is the flat cutoff
    :func:`lumenairy.raytrace._conic_core.conic_sag` itself uses, so the two
    agree on what "flat" means.
    """
    from .surface import _field_frame_active
    if _field_frame_active(surface) or getattr(surface, 'freeform', None):
        return False
    if (getattr(surface, 'aspheric_coeffs', None)
            or getattr(surface, 'aspheric_coeffs_y', None)):
        return False
    for attr in ('radius', 'radius_y'):
        R = getattr(surface, attr, None)
        if R is None:
            continue
        R = float(R)
        if not (np.isinf(R) or abs(R) > 1e15):
            return False
    return True


def _exit_direction_sign(surfaces) -> float:
    """Sign of the outgoing ray's ``N`` relative to +z.

    Each MIRROR in the prescription reverses the propagation direction, and
    the unreduced slope state ``u = L/N`` does not record that sign (``u``
    flips with ``N``).  The signed vertex-plane transfer parameter
    ``t = -z/N = -sag * sign(N) * sqrt(1 + ux^2 + uy^2)`` needs it, so it is
    recovered from the surface list, where it is a deterministic property of
    the prescription rather than of the data.
    """
    s = 1.0
    for surf in surfaces:
        if getattr(surf, 'is_mirror', False):
            s = -s
    return s


def _project_to_exit_vertex_plane(transfer, surfaces, wavelength, n_exit,
                                  fn_name, reached_surface=None):
    """Return ``transfer`` re-referenced from the last surface to its VERTEX
    plane -- the single implementation of that projection for this module.

    The map is the straight-line transfer of each base ray along its own
    direction to ``z = 0``,
    ``t = -sag * sign(N) * sec``, ``sec = sqrt(1 + ux^2 + uy^2)``::

        x_v   = x   - sag * ux
        y_v   = y   - sag * uy
        opd_v = opd - n_exit * sign(N) * sag * sec

    -- the same arithmetic
    :func:`lumenairy.raytrace.exit_vertex.exit_vertex_transfer` applies to a
    ray bundle (there written as ``t = -z/N`` on the direction cosines).

    The 4x4 Jacobian is projected with it, EXACTLY: the transfer distance is
    itself a function of where the ray lands, so the composite derivative of
    the vertex-plane state with respect to the input state is ``P @ J`` with

    .. math::
        P = \\begin{pmatrix} I - u \\otimes \\nabla s & -s I \\\\
                            0 & I \\end{pmatrix}

    (``s`` the sag, ``\\nabla s`` its transverse gradient, ``u`` the output
    slope pair).  Dropping the ``u \\otimes \\nabla s`` and ``-s I`` blocks --
    projecting the state but leaving the Jacobian on the surface -- is a
    FIFTH-DECIMAL effect on the reconstructed field (measured 2026-09-14
    over twelve fixture-and-plane pairs: 1e-05 to 1.6e-05 of fidelity, ten
    of the twelve in the projection's favour), but it costs nothing (four
    row updates on an array already in cache) and it is the derivative of
    the map that was actually applied, so the full ``P`` is used.  It is
    pinned against an independent finite difference of the vertex-plane
    state in ``tests/unit/test_audit2609_b12_fga_reference_plane.py``.

    Sag and its gradient come from the package's shared surface kernels
    (:func:`lumenairy.raytrace.surface._surface_sag_xy` /
    ``_surface_sag_derivatives_xy`` on NumPy, which cover every surface type
    the finite-difference tracer supports; :func:`_conic_core.conic_sag` /
    ``conic_sag_derivs`` on JAX, whose rotationally-symmetric conic +
    even-aspheric domain is exactly the analytic backend's own).

    A surface whose sag is identically zero short-circuits: the input object
    is returned unchanged, so a FLAT-last-surface prescription is bit-for-bit
    what ``reference='surface'`` produces.

    DEAD RAYS ARE FROZEN, not projected (audit 2026-09-11 WAVE5-E, from
    VERIFY-WP-B12 open item O-1).  A vignetted / TIR'd / missed ray never
    reached the last surface, so there is no sag to walk back along and no
    exit-medium path to subtract: ``lumenairy.raytrace.exit_vertex``'s
    ``exit_vertex_transfer`` is explicit that such a ray keeps its position,
    direction and OPL "exactly".  This projection used to apply its arithmetic
    to every row, which moved a dead ray's ``opd`` by up to 1.87e-04 m on a fan
    clipped at the last surface -- unobservable today only because all four
    ``fga.py`` consumers zero the dead beamlets before the reconstruction, i.e.
    a divergence between the module's two vertex-plane operators that the next
    consumer would not know about.  Every field the map touches is now
    ``where(reached, projected, original)``: ``x``, ``y``, ``opd`` and the
    Jacobian rows.  ``ux`` / ``uy`` are a passthrough of the map (a transfer is
    not a refraction), so they are identical on both arms by construction and
    need no mask.  RAYS THAT REACHED THE SURFACE are bit-for-bit what they were
    before the mask existed.

    ``reached_surface`` is the mask of rays that REACHED the last surface,
    which is not always ``transfer.alive``.  The finite-difference backend's
    ``alive`` is ``base_alive & companion_alive`` -- it also drops a ray whose
    9-ray FD companion bundle vignettes even though the BASE ray landed
    (VERIFY-WP-B12 open item O-4).  Such a ray's Jacobian is meaningless but
    its state is not: it did reach the vertex plane, and
    ``TraceResult.at_exit_vertex()`` projects it.  Freezing it would put this
    operator back out of step with that one -- in the other direction -- so
    ``ray_transfer_jacobian`` passes the BASE ray's own alive here and the
    analytic backends, which trace one ray and have no companions, let it
    default to ``transfer.alive``.  Measured on the WP-B12 biconvex fan: 1 ray
    of 121 is companion-dead-but-base-alive, and on it the projected state
    agrees with ``at_exit_vertex`` to 4.3e-19 m.
    """
    last = surfaces[-1]
    if _last_surface_sag_vanishes(last):
        return transfer
    from .exit_vertex import resolve_exit_index
    n_out = float(resolve_exit_index(surfaces, wavelength, fn_name=fn_name,
                                     n_exit=n_exit))
    nz = _exit_direction_sign(surfaces)
    x, y = transfer.x, transfer.y
    ux, uy = transfer.ux, transfer.uy
    is_np = isinstance(x, np.ndarray)
    if is_np:
        from .surface import _surface_sag_derivatives_xy, _surface_sag_xy
        sag = np.asarray(_surface_sag_xy(x, y, last), dtype=np.float64)
        sx, sy = _surface_sag_derivatives_xy(x, y, last)
        sx = np.asarray(sx, dtype=np.float64)
        sy = np.asarray(sy, dtype=np.float64)
        xp = np
    else:                       # JAX tracer: the analytic backend's domain
        import jax.numpy as jnp

        from ._conic_core import conic_sag, conic_sag_derivs
        items = _adrt_aspheric_items(last)     # the module's shared (p, c) form
        R = float(getattr(last, 'radius', np.inf) or np.inf)
        kk = float(getattr(last, 'conic', 0.0) or 0.0)
        sag = conic_sag(x, y, R, kk, items, xp=jnp)
        sx, sy = conic_sag_derivs(x, y, R, kk, items, xp=jnp)
        xp = jnp
    sec = xp.sqrt(1.0 + ux * ux + uy * uy)

    def _apply_P(j):
        """``P @ j`` for one ``(N, 4, 4)`` block, written out as four row
        updates so the two identity rows stay exact."""
        j0, j1, j2, j3 = j[:, 0, :], j[:, 1, :], j[:, 2, :], j[:, 3, :]
        e = sag[:, None]
        row0 = (1.0 - sx * ux)[:, None] * j0 + (-sy * ux)[:, None] * j1 - e * j2
        row1 = (-sx * uy)[:, None] * j0 + (1.0 - sy * uy)[:, None] * j1 - e * j3
        return xp.stack([row0, row1, j2, j3], axis=-2)

    jac = transfer.jacobian
    if jac.ndim == 3:                       # composite input -> output
        jac_v = _apply_P(jac)
        jac_mask_shape = (slice(None), None, None)
    else:                                   # per-surface LOCAL transfers:
        #    only the last one ends on the last surface, so only it moves.
        jac_v = xp.concatenate(
            [jac[:-1], _apply_P(jac[-1])[None, ...]], axis=0)
        jac_mask_shape = (None, slice(None), None, None)
    if is_np:
        jac_v = np.nan_to_num(jac_v, nan=0.0, posinf=0.0, neginf=0.0)
    # Freeze the rays that never reached the surface -- see the docstring.
    # The mask is broadcast to each field's own rank rather than reshaped, so
    # the composite and per-surface Jacobian layouts take the SAME one.
    reached = xp.asarray(
        transfer.alive if reached_surface is None else reached_surface,
        dtype=bool)
    jac_v = xp.where(reached[jac_mask_shape], jac_v, jac)
    return DifferentialTransfer(
        jacobian=jac_v,
        x=xp.where(reached, x - sag * ux, x),
        y=xp.where(reached, y - sag * uy, y),
        ux=ux, uy=uy,
        opd=xp.where(reached,
                     transfer.opd - (n_out * nz) * sag * sec, transfer.opd),
        alive=transfer.alive)


def ray_transfer_jacobian(
    x: np.ndarray,
    y: np.ndarray,
    ux: np.ndarray,
    uy: np.ndarray,
    surfaces: List,
    wavelength: float,
    *,
    per_surface: bool = False,
    h_pos: float = 1e-6,
    h_slope: float = 5e-5,
    reference: str = 'surface',
    n_exit: Optional[float] = None,
) -> DifferentialTransfer:
    """Differential ray-transfer Jacobian along each base ray.

    Parameters
    ----------
    x, y : ndarray ``(N,)``
        Base-ray transverse positions at the input vertex [m].
    ux, uy : ndarray ``(N,)``
        Base-ray paraxial slopes ``L/N``, ``M/N`` at the input.
    surfaces : list of Surface
        As from :func:`raytrace.surfaces_from_prescription`.  Each surface's
        ``thickness`` is the axial distance to the next vertex.
    wavelength : float
        Vacuum wavelength [m].
    per_surface : bool, default False
        ``False`` -> composite input->output ``(N, 4, 4)`` Jacobian.
        ``True``  -> per-surface local transfers ``(N_surf, N, 4, 4)`` obtained
        from the cumulative Jacobians ``J_local_k = J_cum_k @ inv(J_cum_{k-1})``
        (needed for the per-surface complex-parameter amplitude accumulation,
        where each near-identity local factor keeps the ``sqrt(det)`` branch
        unambiguous).
    h_pos, h_slope : float
        Central-FD steps for the position and slope perturbations.  The result
        is insensitive across several decades (the trace is smooth); the
        defaults sit in the truncation-limited regime for mm-scale optics.
    reference : {'surface', 'exit_vertex'}, default 'surface'
        Output reference plane.  ``'surface'`` leaves the state and the
        Jacobian ON the last surface (``z = sag(rho)``), the plane
        :func:`lumenairy.raytrace.trace` itself stops on -- compose that with
        a following surface, or with anything that re-intersects.
        ``'exit_vertex'`` projects both onto the last surface's VERTEX plane
        (``z = 0``), which is what a caller that adds its own image-side
        free-space leg of length ``z_image`` needs: ``z_image`` is measured
        from the vertex plane, so starting the leg on the surface adds a
        spurious ``k * sag(rho)`` of phase (7.8 waves at the rim of an
        N-SF11 R = +/-1.6 mm biconvex at 633 nm; exactly zero when the last
        surface is flat, where the two options are bit-identical).  The
        projection is :func:`_project_to_exit_vertex_plane`.
    n_exit : float, optional
        Refractive index of the medium after the last surface, used by
        ``reference='exit_vertex'`` for the optical path of the projected
        segment.  Resolved from ``surfaces[-1].glass_after`` when omitted
        (:func:`lumenairy.raytrace.exit_vertex.resolve_exit_index`, which
        raises rather than guessing 1.0 for a name it cannot resolve).
        Ignored by ``reference='surface'``.

    Returns
    -------
    DifferentialTransfer
    """
    _validate_reference(reference, 'ray_transfer_jacobian')
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ux = np.asarray(ux, dtype=np.float64)
    uy = np.asarray(uy, dtype=np.float64)
    n = x.shape[0]
    steps = (h_pos, h_pos, h_slope, h_slope)

    # Build 9N rays: base + (+/-h) in each of x, y, ux, uy.  Group g runs
    # 0=base, then (dim, sign) pairs 1..8.
    base = (x, y, ux, uy)
    cols = [base]
    for dim in range(4):
        for sign in (+1.0, -1.0):
            pert = [a.copy() for a in base]
            pert[dim] = pert[dim] + sign * steps[dim]
            cols.append(tuple(pert))
    xx = np.concatenate([c[0] for c in cols])
    yy = np.concatenate([c[1] for c in cols])
    uxx = np.concatenate([c[2] for c in cols])
    uyy = np.concatenate([c[3] for c in cols])
    L, M, _ = _slopes_to_dirs(uxx, uyy)
    rb = _make_bundle(xx, yy, L, M, wavelength)

    of = 'all' if per_surface else 'last'
    res = trace(rb, surfaces, wavelength, output_filter=of)

    def _state(bundle):
        return (bundle.x, bundle.y, bundle.L / bundle.N, bundle.M / bundle.N,
                bundle.opd, bundle.alive)

    def _cum_jac(bundle):
        sx, sy, sux, suy, _, _ = _state(bundle)
        g = [(sx[i * n:(i + 1) * n], sy[i * n:(i + 1) * n],
              sux[i * n:(i + 1) * n], suy[i * n:(i + 1) * n])
             for i in range(9)]
        J = np.zeros((n, 4, 4))
        for dim in range(4):
            gp = g[1 + 2 * dim]
            gm = g[2 + 2 * dim]
            d = 2.0 * steps[dim]
            for outr in range(4):
                J[:, outr, dim] = (gp[outr] - gm[outr]) / d
        return J

    def _companion_alive(balive):
        """A base ray is only usable if ALL its 9 finite-difference companions
        survived: a +/-h companion that vignettes/TIRs at a rim yields NaN
        Jacobian rows while the base ray's own ``alive`` is True (D2).  AND the
        companion-aliveness into the returned mask so no NaN-Jacobian row
        reaches a downstream coherent sum -- the analytic backend already
        scrubs (nan_to_num); the FD backend must too."""
        return np.asarray(balive, bool).reshape(9, n).all(axis=0)

    if not per_surface:
        img = res.image_rays
        Jc = _cum_jac(img)
        bx, by, bux, buy, bopd, balive = _state(img)
        alive = np.asarray(balive[:n], bool) & _companion_alive(balive)
        Jc = np.nan_to_num(Jc, nan=0.0, posinf=0.0, neginf=0.0)
        out = DifferentialTransfer(
            jacobian=Jc, x=bx[:n], y=by[:n], ux=bux[:n], uy=buy[:n],
            opd=bopd[:n], alive=alive)
        if reference == 'exit_vertex':
            out = _project_to_exit_vertex_plane(
                out, surfaces, wavelength, n_exit,
                "ray_transfer_jacobian(reference='exit_vertex')",
                reached_surface=np.asarray(balive[:n], bool))
        return out

    # per-surface: cumulative J at each surface -> local transfers
    hist = res.ray_history
    cum = [np.broadcast_to(np.eye(4), (n, 4, 4)).copy()]  # J at input = I
    for hb in hist:
        cum.append(_cum_jac(hb))
    locals_ = np.stack([cum[k + 1] @ np.linalg.inv(cum[k])
                        for k in range(len(hist))], axis=0)  # (Nsurf,n,4,4)
    locals_ = np.nan_to_num(locals_, nan=0.0, posinf=0.0, neginf=0.0)
    final = hist[-1]
    bx, by, bux, buy, bopd, balive = _state(final)
    alive = np.asarray(balive[:n], bool) & _companion_alive(balive)
    out = DifferentialTransfer(
        jacobian=locals_, x=bx[:n], y=by[:n], ux=bux[:n], uy=buy[:n],
        opd=bopd[:n], alive=alive)
    if reference == 'exit_vertex':
        out = _project_to_exit_vertex_plane(
            out, surfaces, wavelength, n_exit,
            "ray_transfer_jacobian(reference='exit_vertex')",
            reached_surface=np.asarray(balive[:n], bool))
    return out


def ray_transfer_jacobian_jax(x, y, ux, uy, prescription, wavelength):
    """Differentiable per-ray composite ABCD Jacobian via ``jax`` autodiff.

    The JAX twin of :func:`ray_transfer_jacobian` (composite input->output):
    the 4x4 ``(x, y, ux, uy)`` ray-transfer Jacobian of the JAX-traceable
    :func:`raytrace.trace_jax`, computed by ``jax.jacfwd`` (EXACT, not finite
    differences) and vmapped over rays.  Because it is built on ``trace_jax``,
    the whole thing is ``jax.grad`` / ``jax.jit`` friendly and differentiable
    with respect to the ray state -- and, via
    :func:`raytrace.trace_jax_with_params`, with respect to prescription
    parameters (radii, thicknesses) -- enabling gradient-based lens design on
    the per-surface GBD.

    .. note::
       ``jax_trace._transfer_jax`` propagates ``x_new = x + L * thickness``
       using the direction cosine ``L`` rather than the slope ``u = L / N`` --
       paraxial *for that step in isolation*.  BUT it does **not** reset ``z``
       afterwards, so the next surface's intersection re-propagates the ray to
       the true vertex plane and *undoes* the under-count.  Empirically the
       **composed** output Jacobian therefore matches the NumPy FD
       :func:`ray_transfer_jacobian` (and the exact
       :func:`ray_transfer_jacobian_analytic`) to ~1e-8 even at high NA (e.g.
       slope ``u = 0.9`` through a powered lens), because every transfer is
       followed by an intersection.  (The only way to expose the isolated
       paraxial step would be a final transfer with no following surface, which
       this primitive never emits.)

    .. note::
       Like :func:`ray_transfer_jacobian`'s default, the Jacobian is referenced
       to the LAST SURFACE (``z = sag(rho)``), not to its vertex plane -- this
       primitive returns no base-ray state, so there is nothing here for a
       caller to mistake for a vertex-plane coordinate, and it carries no
       ``reference`` keyword.  A caller that needs the vertex-plane Jacobian
       uses :func:`ray_transfer_jacobian_analytic` (whose JAX branch takes the
       same ``reference='exit_vertex'``).

    Returns
    -------
    jax array ``(N, 4, 4)``
        Per-ray composite ray-transfer Jacobian (row/col order (x, y, ux, uy)).
    """
    import jax
    import jax.numpy as jnp

    from .jax_trace import make_jax_ray_state, trace_jax

    def _out_state(s4):
        xx, yy, uxx, uyy = s4[0], s4[1], s4[2], s4[3]
        inv = 1.0 / jnp.sqrt(1.0 + uxx * uxx + uyy * uyy)
        st = make_jax_ray_state(
            jnp.reshape(xx, (1,)), jnp.reshape(yy, (1,)),
            jnp.zeros((1,)), jnp.reshape(uxx * inv, (1,)),
            jnp.reshape(uyy * inv, (1,)), jnp.reshape(inv, (1,)))
        r = trace_jax(st, prescription, wavelength)
        return jnp.stack([r.x[0], r.y[0], r.L[0] / r.N[0], r.M[0] / r.N[0]])

    s4 = jnp.stack([jnp.asarray(x), jnp.asarray(y),
                    jnp.asarray(ux), jnp.asarray(uy)], axis=-1)
    return jax.vmap(jax.jacfwd(_out_state))(s4)


# ---------------------------------------------------------------------------
# Analytic differential ray transfer (forward-mode AD over the EXACT conic
# trace).  Forward-mode AD of the exact intersection + Snell map *is* the
# analytic differential ray tracing of Stone & Forbes (Volatier, JOSA A 34,
# 1146 (2017)); it produces the same 4x4 (x, y, ux, uy) Jacobian as the finite-
# difference :func:`ray_transfer_jacobian` but EXACTLY (no truncation), in pure
# NumPy, and -- on the JAX backend -- differentiably (jax.jacfwd / grad / jit).
# Conic surfaces (sphere / conic), refraction and reflection.
# See docs/ANALYTIC_DIFFERENTIAL_RAY_TRACING_LITERATURE.md.
# ---------------------------------------------------------------------------


class _AdrtDual:
    """Minimal forward-mode AD dual: value ``v`` ``(N,)`` and derivative ``d``
    ``(N, 4)`` w.r.t. the seed state ``(x, y, ux, uy)``."""
    __slots__ = ('v', 'd')
    # Defer to our __r*__ so ``ndarray * dual`` (sign selectors) doesn't get
    # absorbed into an object-array by NumPy.
    __array_ufunc__ = None

    def __init__(self, v, d):
        self.v = v
        self.d = d

    def __add__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(s.v + o.v, s.d + o.d)
    __radd__ = __add__

    def __sub__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(s.v - o.v, s.d - o.d)

    def __rsub__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(o.v - s.v, o.d - s.d)

    def __mul__(s, o):
        o = _as_dual(o, s)
        return _AdrtDual(s.v * o.v, s.v[:, None] * o.d + o.v[:, None] * s.d)
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = _as_dual(o, s)
        iv = 1.0 / o.v
        return _AdrtDual(s.v * iv,
                         (s.d * o.v[:, None] - s.v[:, None] * o.d)
                         * (iv * iv)[:, None])

    def __rtruediv__(s, o):
        return _as_dual(o, s).__truediv__(s)

    def __neg__(s):
        return _AdrtDual(-s.v, -s.d)


def _as_dual(o, ref):
    if isinstance(o, _AdrtDual):
        return o
    n = ref.v.shape[0]
    return _AdrtDual(np.broadcast_to(np.asarray(o, np.float64), (n,)).copy(),
                     np.zeros((n, 4)))


def _dual_sqrt(a):
    # clamp the radicand at 0 so a dead ray (missed surface / TIR: negative
    # discriminant) yields a FINITE (masked) value instead of a NaN that would
    # contaminate live neighbours; live rays (radicand > 0) are unaffected.
    vc = np.maximum(a.v, 0.0)
    v = np.sqrt(vc)
    return _AdrtDual(v, a.d / (2.0 * np.maximum(v, 1e-300))[:, None])


def _dual_where(m, a, b):
    a = _as_dual(a, b if isinstance(b, _AdrtDual) else a)
    b = _as_dual(b, a)
    return _AdrtDual(np.where(m, a.v, b.v), np.where(m[:, None], a.d, b.d))


_DUAL_OPS = {'sqrt': _dual_sqrt, 'val': lambda a: a.v,
             'pwhere': np.where, 'dwhere': _dual_where}


def _adrt_coordbreak(x, y, ux, uy, surf, wavelength, apply_transfer,
                     sqrt, val, compute_dead):
    """Zemax-style coordinate break (decenter + intrinsic X,Y,Z tilts) then the
    gap transfer -- no intersection.  A smooth frame rotation, so it is exactly
    differentiable; replicates ``raytrace.intersection._apply_coord_break``
    (decenter-then-tilt for ``coordbrk_order=0``, tilt-then-decenter for 1).
    The ray starts and ends at ``z = 0`` (the vertex planes), so no cross-step
    ``z`` state is needed -- the intermediate ``z`` from the tilt is consumed by
    the gap transfer.  Small tilts / decenters give differentiable alignment
    (tolerancing) sensitivity; large folds share the world-frame slope-space
    caveat (``u = L/N`` degenerates as ``N -> 0``)."""
    import math

    from ..glass import get_glass_index
    tx = math.radians(float(getattr(surf, 'tilt_x_deg', 0.0) or 0.0))
    ty = math.radians(float(getattr(surf, 'tilt_y_deg', 0.0) or 0.0))
    tz = math.radians(float(getattr(surf, 'tilt_z_deg', 0.0) or 0.0))
    dcx = float(getattr(surf, 'decenter_x_m', 0.0) or 0.0)
    dcy = float(getattr(surf, 'decenter_y_m', 0.0) or 0.0)
    order = int(getattr(surf, 'coordbrk_order', 0) or 0)

    sden = sqrt(1.0 + ux * ux + uy * uy)
    L = ux / sden
    M = uy / sden
    Nn = 1.0 / sden
    px, py, pz = x, y, 0.0 * ux            # z = 0 at the vertex plane

    def _decenter(px, py):
        return px - dcx, py - dcy

    def _tilts(px, py, pz, L, M, Nn):
        # W3-1 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): the ray-coordinate
        # transform is the TRANSPOSE of Zemax's local-to-world tilt
        # (``Rx_math(+theta)``, KB KA-01638), i.e. ``Rx_math(-theta)``.
        # Op-for-op replication of the corrected
        # ``intersection._apply_coord_break`` (see its _rot_x comment for
        # the derivation and the measured numbers).
        cx, sx = math.cos(tx), math.sin(tx)      # Rx_math(-tx)
        py, pz = cx * py + sx * pz, -sx * py + cx * pz
        M, Nn = cx * M + sx * Nn, -sx * M + cx * Nn
        cy, sy = math.cos(ty), math.sin(ty)      # Ry_math(-ty)
        px, pz = cy * px - sy * pz, sy * px + cy * pz
        L, Nn = cy * L - sy * Nn, sy * L + cy * Nn
        cz, sz = math.cos(tz), math.sin(tz)      # Rz_math(-tz)
        px, py = cz * px + sz * py, -sz * px + cz * py
        L, M = cz * L + sz * M, -sz * L + cz * M
        return px, py, pz, L, M, Nn

    if order == 1:
        px, py, pz, L, M, Nn = _tilts(px, py, pz, L, M, Nn)
        px, py = _decenter(px, py)
    else:
        px, py = _decenter(px, py)
        px, py, pz, L, M, Nn = _tilts(px, py, pz, L, M, Nn)

    opd = 0.0 * val(ux)
    if apply_transfer:
        t = float(getattr(surf, 'thickness', 0.0) or 0.0)
        n2 = float(get_glass_index(
            getattr(surf, 'glass_after', None) or 'air', wavelength))
        tau = (t - pz) / Nn
        px = px + L * tau
        py = py + M * tau
        # SIGNED transfer leg (RT-1), matching raytrace._transfer's n*t used by
        # the main-trace coord-break path (trace._apply_coord_break -> _transfer):
        # a negative gap tau must SUBTRACT its OPL, not add via abs (S3-9).
        opd = n2 * val(tau)
    ux_out = L / Nn
    uy_out = M / Nn
    dead = None
    if compute_dead:
        dead = np.zeros(val(ux).shape, dtype=bool)
    return px, py, ux_out, uy_out, opd, dead


_ADRT_ASPHERIC_NEWTON_STEPS = 6


def _adrt_aspheric_items(surf):
    """``((power, coeff), ...)`` for a rotationally-symmetric asphere.

    Empty when the surface carries no polynomial departure, so the caller
    can branch on truthiness.  Odd powers are rejected here with the same
    message the sag / normal twins use -- an odd power is
    sag/normal-inconsistent in every backend
    (:func:`_conic_core.check_even_aspheric_powers`).
    """
    asph = getattr(surf, 'aspheric_coeffs', None)
    if not asph:
        return ()
    check_even_aspheric_powers(asph.keys(), fn_label='_adrt_step')
    return tuple(sorted((int(p), float(c)) for p, c in asph.items()))


def _adrt_u_pow(u, m):
    """``u ** m`` for a non-negative integer ``m`` by squaring.

    ``_AdrtDual`` implements only ``+ - * /``, and the JAX backend must
    stay in the same elementary ops so the two agree; binary
    exponentiation keeps both to ``O(log m)`` multiplications.  ``m = 0``
    returns the Python float ``1.0``, which every consumer here adds or
    multiplies into a dual / array without promotion.
    """
    if m == 0:
        return 1.0
    result = None
    base = u
    while m:
        if m & 1:
            result = base if result is None else result * base
        m >>= 1
        if m:
            base = base * base
    return result


def _adrt_poly_sag(u, asph_items, O):
    """``(P(u), dP/du)`` of the even-power polynomial departure.

    ``u = x**2 + y**2``, so an even power ``p`` is ``u ** (p // 2)`` and
    the rotationally-symmetric chain rule is
    ``dP/dx = (dP/du) * 2x`` -- no ``sqrt(u)`` anywhere, which is what
    keeps the departure and its gradient smooth through the vertex.
    """
    P = 0.0
    dP = 0.0
    for power, coeff in asph_items:
        m = power // 2
        P = P + coeff * _adrt_u_pow(u, m)
        if m >= 1:
            dP = dP + (coeff * m) * _adrt_u_pow(u, m - 1)
    return P, dP


def _adrt_conic_sag(u, c, k, O):
    """``(S(u), dS/du)`` of the base conic, on ``u = x**2 + y**2``.

    ``S = c u / (1 + w)`` with ``w = sqrt(1 - (1+k) c^2 u)``; the radial
    derivative ``dS/dh = c h / w`` divided by ``2h`` gives ``dS/du =
    c / (2 w)``, which has no ``1/h`` and is therefore finite at the
    vertex.  A FLAT base (``c = 0``) gives ``w = 1``, ``S = 0``,
    ``dS/du = 0`` with no special case.
    """
    sqrt = O['sqrt']
    w = sqrt(1.0 - ((1.0 + k) * c * c) * u)
    return (c * u) / (1.0 + w), c / (2.0 * w)


def _adrt_aspheric_intersect(x, y, L, M, Nn, tau, c, k, asph_items, O):
    """Newton-refine the conic root onto ``conic + polynomial``, and
    return the intersection point with the surface normal there.

    The seed ``tau`` is the EXACT root of the base conic (the caller's
    Spencer & Murty ``e/q`` form), so only the polynomial departure is
    left to iterate on -- Newton on

        G(tau) = z(tau) - S(u(tau)) - P(u(tau)),
        dG/dtau = Nz - (dS/du + dP/du) * du/dtau,
        du/dtau = 2 (x(tau) L + y(tau) M)

    converges in two to three steps for a physical asphere.  The step
    count is FIXED (no data-dependent break) because this runs under
    forward-mode AD on both backends: a ``while`` on a dual value has no
    derivative, and a ``lax.while_loop`` would not be ``jacfwd``-able
    here.  Differentiating the iteration itself is what makes the
    returned Jacobian exact -- the tangent converges with the value.

    The normal comes from the implicit form ``F = z - S(u) - P(u)``:
    ``grad F = (-2x D, -2y D, 1)`` with ``D = dS/du + dP/du``, normalised.
    """
    for _ in range(_ADRT_ASPHERIC_NEWTON_STEPS):
        xi = x + tau * L
        yi = y + tau * M
        zi = tau * Nn
        u = xi * xi + yi * yi
        S, dSdu = _adrt_conic_sag(u, c, k, O)
        P, dPdu = _adrt_poly_sag(u, asph_items, O)
        D = dSdu + dPdu
        G = zi - S - P
        dG = Nn - D * (2.0 * (xi * L + yi * M))
        tau = tau - G / dG
    xi = x + tau * L
    yi = y + tau * M
    zi = tau * Nn
    u = xi * xi + yi * yi
    _S, dSdu = _adrt_conic_sag(u, c, k, O)
    _P, dPdu = _adrt_poly_sag(u, asph_items, O)
    D = dSdu + dPdu
    gx = (-2.0 * D) * xi
    gy = (-2.0 * D) * yi
    gn = O['sqrt'](gx * gx + gy * gy + 1.0)
    return tau, xi, yi, zi, gx / gn, gy / gn, 1.0 / gn


def _adrt_step(x, y, ux, uy, surf, wavelength, apply_transfer, O,
               compute_dead=True):
    """One surface: exact intersect (conic) + refract/reflect + optional
    transfer, OR a coordinate-break frame transform (``is_coordbrk``), on
    dual-or-jnp ``(x, y, ux, uy)``.  Returns the updated state plus the
    plain-array OPL increment and dead-ray mask."""
    from ..glass import get_glass_index
    sqrt, val = O['sqrt'], O['val']
    pwhere, dwhere = O['pwhere'], O['dwhere']
    if bool(getattr(surf, 'is_coordbrk', False)):
        return _adrt_coordbreak(x, y, ux, uy, surf, wavelength,
                                apply_transfer, sqrt, val, compute_dead)
    R = float(getattr(surf, 'radius', np.inf))
    c = 0.0 if not np.isfinite(R) else 1.0 / R
    k = float(getattr(surf, 'conic', 0.0) or 0.0)
    is_mir = bool(getattr(surf, 'is_mirror', False))
    n1 = float(get_glass_index(getattr(surf, 'glass_before', None) or 'air',
                               wavelength))
    n2 = float(get_glass_index(getattr(surf, 'glass_after', None) or 'air',
                               wavelength))
    sden = sqrt(1.0 + ux * ux + uy * uy)
    L = ux / sden
    M = uy / sden
    Nn = 1.0 / sden
    # intersect conic F = c(x^2+y^2) - 2z + (1+k) c z^2 = 0, ray from (x, y, 0)
    a = c * (L * L + M * M) + (1.0 + k) * c * (Nn * Nn)
    b = 2.0 * (c * (x * L + y * M) - Nn)
    e = c * (x * x + y * y)
    disc = b * b - 4.0 * a * e
    sq = sqrt(disc)
    sgn = pwhere(val(b) >= 0.0, 1.0, -1.0)
    q = -0.5 * (b + sgn * sq)
    # stable near-vertex root tau = e/q; flat surface (a == 0) -> tau = -e/b
    tau = dwhere(abs(val(a)) < 1e-14, (0.0 - e) / b, e / q)
    asph_items = _adrt_aspheric_items(surf)
    if asph_items:
        # Conic + POLYNOMIAL departure: the conic root above is the seed;
        # Newton refines it onto the full surface and the normal picks up
        # the polynomial gradient (:func:`_adrt_aspheric_intersect`).
        # ``disc`` from the base conic stays the miss test, exactly as
        # ``intersection._intersect_surface`` uses it (R4): exact for a
        # pure conic, conservative for a conic plus a departure.
        tau, xi, yi, zi, nx, ny, nz = _adrt_aspheric_intersect(
            x, y, L, M, Nn, tau, c, k, asph_items, O)
    else:
        xi = x + tau * L
        yi = y + tau * M
        zi = tau * Nn
        # surface normal grad F = (2c x, 2c y, -2 + 2(1+k)c z), oriented
        # against ray
        gx = (2.0 * c) * xi
        gy = (2.0 * c) * yi
        gz = -2.0 + (2.0 * (1.0 + k) * c) * zi
        gn = sqrt(gx * gx + gy * gy + gz * gz)
        nx = gx / gn
        ny = gy / gn
        nz = gz / gn
    # S3-10: vector Snell / reflection via the backend-agnostic shared
    # core (raytrace._conic_core).  The core orients the (un-oriented)
    # grad-F normal against the ray and applies the same law this site
    # used; ``eta_sq = eta * eta`` preserves this site's PRODUCT form
    # (NOT the scalar-power ``mu**2`` the NumPy / JAX sites use -- IEEE
    # gives ``x**2 != x*x`` for ~0.05% of ratios; see the shared-core
    # docstring), and the injected ADRT ``sqrt`` (``O['sqrt']`` --
    # ``_dual_sqrt`` on the dual backend, a clamping jnp sqrt on the JAX
    # backend) clamps the radicand so the default no-op TIR guard
    # reproduces the former ``root = sqrt(disc_r)`` exactly.  Pure /
    # dual-aware: no in-place writes.
    if is_mir:
        Lp, Mp, Np, nx, ny, nz, _cos_i = reflect_mirror(
            L, M, Nn, nx, ny, nz, where=pwhere, val=val)
        disc_r = None
    else:
        eta = n1 / n2
        Lp, Mp, Np, nx, ny, nz, _cos_i, disc_r, _tir = refract_snell(
            L, M, Nn, nx, ny, nz, eta, eta * eta,
            sqrt=sqrt, where=pwhere, val=val)
    if apply_transfer:
        t = float(getattr(surf, 'thickness', 0.0) or 0.0)
        tau2 = (t - zi) / Np
        x_out = xi + tau2 * Lp
        y_out = yi + tau2 * Mp
        # OPL: SIGNED intersection leg (a backtrack on a concave surface
        # subtracts over-counted OPL, matching raytrace._intersect_surface) plus
        # the SIGNED transfer leg (matching raytrace._transfer's RT-1 signed
        # n*t: a negative tau2 -- overlapping sag / post-mirror fold -- means the
        # ray already crossed the next vertex plane, so the over-counted OPL must
        # be SUBTRACTED, not added via abs; abs here diverged from the base-ray
        # OPL the FD/main trace produces whenever tau2 < 0, S3-9).
        opd = n1 * val(tau) + n2 * val(tau2)
    else:
        x_out = xi
        y_out = yi
        opd = n1 * val(tau)
    ux_out = Lp / Np
    uy_out = Mp / Np
    # dead: aperture vignette + TIR (NumPy path only; the JAX path returns
    # alive=True and would trip a tracer->ndarray conversion here).
    dead = None
    if compute_dead:
        dead = val(disc) < 0.0                    # missed the surface
        sd = float(getattr(surf, 'semi_diameter', np.inf))
        if np.isfinite(sd):
            dead = dead | (val(xi) ** 2 + val(yi) ** 2 > sd * sd)
        if disc_r is not None:
            dead = dead | (val(disc_r) < 0.0)     # TIR
    return x_out, y_out, ux_out, uy_out, opd, dead


def _adrt_numpy(x, y, ux, uy, surfaces, wavelength, per_surface):
    x = np.asarray(x, np.float64)
    n = x.shape[0]
    y = np.broadcast_to(np.asarray(y, np.float64), (n,)).copy()
    ux = np.broadcast_to(np.asarray(ux, np.float64), (n,)).copy()
    uy = np.broadcast_to(np.asarray(uy, np.float64), (n,)).copy()
    x = np.broadcast_to(x, (n,)).copy()
    eye = np.eye(4)

    def _seed(xv, yv, uxv, uyv):
        return (_AdrtDual(xv.copy(), np.tile(eye[0], (n, 1))),
                _AdrtDual(yv.copy(), np.tile(eye[1], (n, 1))),
                _AdrtDual(uxv.copy(), np.tile(eye[2], (n, 1))),
                _AdrtDual(uyv.copy(), np.tile(eye[3], (n, 1))))

    opd = np.zeros(n)
    alive = np.ones(n, dtype=bool)
    nsurf = len(surfaces)
    # dead rays (missed / TIR) can overflow the dual arithmetic; the result is
    # masked (alive) and nan_to_num'd below, so silence the expected noise.
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        if per_surface:
            locals_ = []
            cx, cy, cux, cuy = x, y, ux, uy
            for si, s in enumerate(surfaces):
                X, Y, UX, UY = _seed(cx, cy, cux, cuy)
                X, Y, UX, UY, dopd, dead = _adrt_step(
                    X, Y, UX, UY, s, wavelength, si < nsurf - 1, _DUAL_OPS)
                Jk = np.stack([X.d, Y.d, UX.d, UY.d], axis=1)   # (n, 4, 4)
                locals_.append(Jk)
                opd = opd + dopd
                alive = alive & ~dead
                cx, cy, cux, cuy = X.v, Y.v, UX.v, UY.v
            jac = np.stack(locals_, axis=0)                 # (nsurf, n, 4, 4)
            bx, by, bux, buy = cx, cy, cux, cuy
        else:
            X, Y, UX, UY = _seed(x, y, ux, uy)
            for si, s in enumerate(surfaces):
                X, Y, UX, UY, dopd, dead = _adrt_step(
                    X, Y, UX, UY, s, wavelength, si < nsurf - 1, _DUAL_OPS)
                opd = opd + dopd
                alive = alive & ~dead
            jac = np.stack([X.d, Y.d, UX.d, UY.d], axis=1)  # (n, 4, 4)
            bx, by, bux, buy = X.v, Y.v, UX.v, UY.v
    # A dead ray (missed / TIR) can leave non-finite entries (e.g. a grazing
    # 1/N'); zero them so they cannot contaminate array-wide ops downstream.
    # Live rays are already finite, so this is a no-op for them.
    jac = np.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
    bx, by, bux, buy, opd = (np.nan_to_num(v) for v in (bx, by, bux, buy, opd))
    return DifferentialTransfer(jacobian=jac, x=bx, y=by, ux=bux, uy=buy,
                                opd=opd, alive=alive)


# ---------------------------------------------------------------------------
# B4 (roadmap): numba-vectorized batched-dual acceleration of the composite
# analytic ray transfer.  The scalar-object ``_AdrtDual`` overloads above run
# one small NumPy op per elementary arithmetic step (K3: ``__mul__`` ~178 k
# calls dominates the adaptive-FGA ``exact_jacobian`` path).  This kernel does
# the identical forward-mode-AD ray transfer -- value ``v`` + the 4-vector
# tangent ``d`` w.r.t. the seed state ``(x, y, ux, uy)`` -- but per-ray in a
# single compiled loop, carrying each dual quantity as a homogeneous 5-tuple
# ``(v, d0, d1, d2, d3)`` through inlined AD primitives.  No Python-object
# allocation, no per-op NumPy dispatch, no ``(N, 4)`` temporaries.
#
# EXACTNESS: the primitives replicate the ``_AdrtDual`` arithmetic elementary
# op-for-op (product forms -- ``v*v`` not ``v**2``; the ``_dual_sqrt`` clamp at
# ``2*max(sqrt, 1e-300)``; the ``_dual_where`` flat-surface / TIR branches), and
# numba's default (``fastmath=False``) emits standard IEEE-754 double ops with
# no FMA contraction / reassociation, so the result matches the dual path to
# the last ULP (validated element-wise on a ray batch in
# tests/unit/test_niche_r4_fga_dual_vectorize.py).  It covers the composite
# (``per_surface=False``) all-conic (sphere / conic) refract/reflect path -- the
# FGA hot path.  Coordinate breaks, ``per_surface=True``, the JAX backend, and a
# missing numba all fall back to the ``_AdrtDual`` / JAX implementations.
#
# MEASURED ENVELOPE (dev box, warm best-of-3 -- stated so it is never
# overstated).  On the ISOLATED ray-transfer micro-batch this kernel replaces
# (49 momenta x 1600 rays, biconvex singlet) numba is ~12x the ``_AdrtDual``
# path (dual ~290 ms -> numba ~23 ms/loop).  END-TO-END adaptive FGA
# (``apply_real_lens_fga``, N=512, dx=24um, coarse_stride=1,
# exact_jacobian=True) is ~3.1x (dual ~26.5 s -> numba ~8.6 s): the ray
# transfer is ~71-77% of the dual FGA cost, so by Amdahl the whole-call win is
# bounded by the remaining FFT / momentum-quadrature work once the transfer
# itself is ~12x cheaper.  Byte-identical output either way (pure arithmetic
# replication, not an approximation) -- the value here is the ~12x on FGA's
# single dominant cost, which the ~3.1x whole-call figure reflects after
# dilution.
_ADRT_NUMBA_KERNEL = None       # lazily-compiled kernel (NOT a data cache)
_ADRT_NUMBA_STATE = None        # None=untried, False=unavailable, True=ready

# R-16 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): the njit dual primitives are
# CLOSURES inside :func:`_build_adrt_numba_kernel`, so there is no other way to
# exercise their arithmetic conventions directly against the ``_AdrtDual``
# twins -- and every public boundary (:func:`_adrt_numba`) scrubs non-finites
# through ``np.nan_to_num``, which hides exactly the NaN-convention class of
# divergence the audit found.  The builder publishes them here (populated only
# once the kernel is built) purely so the numba<->NumPy parity pin can compare
# them primitive-by-primitive.  NOT part of the public API; do not consume from
# library code.
_ADRT_NUMBA_PRIMS = None        # None until _build_adrt_numba_kernel() runs


def _adrt_surfaces_numba_eligible(surfaces):
    """True iff every surface is a plain rotationally-symmetric CONIC
    refract/reflect surface (no coordinate break, no polynomial asphere) --
    the class the numba forward-AD kernel handles.  Freeform / biconic /
    field-frame surfaces are rejected by ``ray_transfer_jacobian_analytic``
    before this is reached; coordinate breaks (a smooth frame transform
    handled only by the ``_AdrtDual`` ``_adrt_coordbreak``) and aspheric
    departures (the Newton refinement in ``_adrt_aspheric_intersect``, which
    the kernel's inlined conic primitives do not carry) are excluded here.
    The kernel would otherwise trace an asphere as its BASE CONIC -- right
    shape, wrong surface, silently."""
    for s in surfaces:
        if bool(getattr(s, 'is_coordbrk', False)):
            return False
        if getattr(s, 'aspheric_coeffs', None):
            return False
    return True


def _build_adrt_numba_kernel():
    """Compile (once) the per-ray forward-mode-AD composite ray-transfer numba
    kernel.  Returns the compiled callable, or raises on any numba failure (the
    caller catches and falls back to the dual path)."""
    import math

    from numba import njit

    # --- inlined dual primitives on 5-tuples (v, d0, d1, d2, d3) --------------
    @njit(inline='always')
    def _dadd(a, b):
        return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3],
                a[4] + b[4])

    @njit(inline='always')
    def _dsub(a, b):
        return (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3],
                a[4] - b[4])

    @njit(inline='always')
    def _dmul(a, b):
        # matches _AdrtDual.__mul__: v*v, v[:,None]*d + v[:,None]*d
        return (a[0] * b[0],
                a[0] * b[1] + b[0] * a[1],
                a[0] * b[2] + b[0] * a[2],
                a[0] * b[3] + b[0] * a[3],
                a[0] * b[4] + b[0] * a[4])

    @njit(inline='always')
    def _ddiv(a, b):
        # matches _AdrtDual.__truediv__: iv=1/b; (a.d*b.v - a.v*b.d)*(iv*iv)
        iv = 1.0 / b[0]
        iv2 = iv * iv
        return (a[0] * iv,
                (a[1] * b[0] - a[0] * b[1]) * iv2,
                (a[2] * b[0] - a[0] * b[2]) * iv2,
                (a[3] * b[0] - a[0] * b[3]) * iv2,
                (a[4] * b[0] - a[0] * b[4]) * iv2)

    @njit(inline='always')
    def _dscale(a, c):
        # dual * plain-scalar (c broadcast, zero tangent) -> c*d, c*v
        return (a[0] * c, a[1] * c, a[2] * c, a[3] * c, a[4] * c)

    @njit(inline='always')
    def _daddc(a, c):
        return (a[0] + c, a[1], a[2], a[3], a[4])

    @njit(inline='always')
    def _dneg(a):
        return (-a[0], -a[1], -a[2], -a[3], -a[4])

    @njit(inline='always')
    def _dsqrtq(a):
        # matches _dual_sqrt: vc=max(v,0); v=sqrt(vc); d/(2*max(v,1e-300))
        #
        # R-16 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): a NaN radicand
        # must PROPAGATE, exactly as the NumPy twin does.  ``nan > 0.0``
        # is False, so a ternary on ``v > 0.0`` clamps a NaN radicand to
        # ``vc = 0.0`` and returns a perfectly finite ``0.0`` value with
        # a huge-but-finite tangent -- while ``_dual_sqrt``'s
        # ``np.maximum(nan, 0.0)`` is ``nan`` (numpy's maximum
        # propagates NaN), giving ``nan`` value AND ``nan`` tangent
        # (``d / (2 * np.maximum(nan, 1e-300))``).  Measured: numpy nan
        # vs numba 0.0 (docs/history/lumenairy.raytrace.differential.md).
        # ray into a plausible on-axis one that the numba FGA kernel then
        # keeps alive, so the numba and NumPy dual backends disagreed
        # about which rays are faulted.  Finite radicands (including
        # exactly 0.0 and negatives) are bit-identical.
        if math.isnan(a[0]):
            nan = a[0]
            return (nan, a[1] / nan, a[2] / nan, a[3] / nan, a[4] / nan)
        vc = a[0] if a[0] > 0.0 else 0.0
        v = math.sqrt(vc)
        den = 2.0 * v if v > 1e-300 else 2.0e-300
        return (v, a[1] / den, a[2] / den, a[3] / den, a[4] / den)

    # SERIAL (parallel=False): each FGA momentum call carries a modest ray
    # batch (~1e3-1e4); the per-call parallel-region setup/teardown cost of
    # parallel=True dwarfs that little work and measured ~2x SLOWER.  The
    # compiled serial loop already removes all the Python-object / per-op NumPy
    # dispatch overhead that dominated the dual path.
    @njit(cache=True)
    def _kernel(x, y, ux, uy, radius, conic, ismir, n1a, n2a, thick, semidia,
                applyt, jac, ox, oy, oux, ouy, oopd, oalive):
        n = x.shape[0]
        nsurf = radius.shape[0]
        one = (1.0, 0.0, 0.0, 0.0, 0.0)
        for i in range(n):
            xq = (x[i], 1.0, 0.0, 0.0, 0.0)
            yq = (y[i], 0.0, 1.0, 0.0, 0.0)
            uxq = (ux[i], 0.0, 0.0, 1.0, 0.0)
            uyq = (uy[i], 0.0, 0.0, 0.0, 1.0)
            opd = 0.0
            alive = True
            for si in range(nsurf):
                Rv = radius[si]
                if math.isinf(Rv):
                    c = 0.0
                else:
                    c = 1.0 / Rv
                k = conic[si]
                n1 = n1a[si]
                n2 = n2a[si]
                # sden = sqrt(1 + ux*ux + uy*uy); L,M,Nn = ux/sden, uy/sden, 1/sden
                s2 = _dadd(_daddc(_dmul(uxq, uxq), 1.0), _dmul(uyq, uyq))
                sden = _dsqrtq(s2)
                L = _ddiv(uxq, sden)
                M = _ddiv(uyq, sden)
                Nn = _ddiv(one, sden)
                # conic intersection quadratic coeffs
                a = _dadd(_dscale(_dadd(_dmul(L, L), _dmul(M, M)), c),
                          _dscale(_dmul(Nn, Nn), (1.0 + k) * c))
                b = _dscale(_dsub(_dscale(_dadd(_dmul(xq, L), _dmul(yq, M)), c),
                                  Nn), 2.0)
                e = _dscale(_dadd(_dmul(xq, xq), _dmul(yq, yq)), c)
                disc = _dsub(_dmul(b, b), _dmul(_dscale(a, 4.0), e))
                sq = _dsqrtq(disc)
                sgn = 1.0 if b[0] >= 0.0 else -1.0
                q = _dscale(_dadd(b, _dscale(sq, sgn)), -0.5)
                aabs = a[0] if a[0] >= 0.0 else -a[0]
                if aabs < 1e-14:
                    tau = _ddiv(_dneg(e), b)
                else:
                    tau = _ddiv(e, q)
                xi = _dadd(xq, _dmul(tau, L))
                yi = _dadd(yq, _dmul(tau, M))
                zi = _dmul(tau, Nn)
                # surface normal grad F, normalized
                gx = _dscale(xi, 2.0 * c)
                gy = _dscale(yi, 2.0 * c)
                gz = _daddc(_dscale(zi, 2.0 * (1.0 + k) * c), -2.0)
                gn = _dsqrtq(_dadd(_dadd(_dmul(gx, gx), _dmul(gy, gy)),
                                   _dmul(gz, gz)))
                nx = _ddiv(gx, gn)
                ny = _ddiv(gy, gn)
                nz = _ddiv(gz, gn)
                # orient the normal against the incident ray
                dn = _dadd(_dadd(_dmul(L, nx), _dmul(M, ny)), _dmul(Nn, nz))
                fl = -1.0 if dn[0] > 0.0 else 1.0
                nx = _dscale(nx, fl)
                ny = _dscale(ny, fl)
                nz = _dscale(nz, fl)
                cos_i = _dneg(_dadd(_dadd(_dmul(L, nx), _dmul(M, ny)),
                                    _dmul(Nn, nz)))
                have_discr = False
                discr_v = 0.0
                if ismir[si]:
                    two_ci = _dscale(cos_i, 2.0)
                    Lp = _dadd(L, _dmul(two_ci, nx))
                    Mp = _dadd(M, _dmul(two_ci, ny))
                    Np = _dadd(Nn, _dmul(two_ci, nz))
                else:
                    eta = n1 / n2
                    eta_sq = eta * eta
                    disc_r = _dsub(one, _dscale(
                        _dsub(one, _dmul(cos_i, cos_i)), eta_sq))
                    root = _dsqrtq(disc_r)
                    coeff = _dsub(_dscale(cos_i, eta), root)
                    Lp = _dadd(_dscale(L, eta), _dmul(coeff, nx))
                    Mp = _dadd(_dscale(M, eta), _dmul(coeff, ny))
                    Np = _dadd(_dscale(Nn, eta), _dmul(coeff, nz))
                    have_discr = True
                    discr_v = disc_r[0]
                if applyt[si]:
                    t = thick[si]
                    t_minus_zi = (t - zi[0], -zi[1], -zi[2], -zi[3], -zi[4])
                    tau2 = _ddiv(t_minus_zi, Np)
                    xo = _dadd(xi, _dmul(tau2, Lp))
                    yo = _dadd(yi, _dmul(tau2, Mp))
                    opd = opd + n1 * tau[0] + n2 * tau2[0]
                else:
                    xo = xi
                    yo = yi
                    opd = opd + n1 * tau[0]
                uxo = _ddiv(Lp, Np)
                uyo = _ddiv(Mp, Np)
                # dead: missed surface + aperture vignette + TIR
                if disc[0] < 0.0:
                    alive = False
                sd = semidia[si]
                if math.isfinite(sd):
                    if xi[0] * xi[0] + yi[0] * yi[0] > sd * sd:
                        alive = False
                if have_discr and discr_v < 0.0:
                    alive = False
                xq = xo
                yq = yo
                uxq = uxo
                uyq = uyo
            ox[i] = xq[0]
            oy[i] = yq[0]
            oux[i] = uxq[0]
            ouy[i] = uyq[0]
            oopd[i] = opd
            oalive[i] = alive
            jac[i, 0, 0] = xq[1]
            jac[i, 0, 1] = xq[2]
            jac[i, 0, 2] = xq[3]
            jac[i, 0, 3] = xq[4]
            jac[i, 1, 0] = yq[1]
            jac[i, 1, 1] = yq[2]
            jac[i, 1, 2] = yq[3]
            jac[i, 1, 3] = yq[4]
            jac[i, 2, 0] = uxq[1]
            jac[i, 2, 1] = uxq[2]
            jac[i, 2, 2] = uxq[3]
            jac[i, 2, 3] = uxq[4]
            jac[i, 3, 0] = uyq[1]
            jac[i, 3, 1] = uyq[2]
            jac[i, 3, 2] = uyq[3]
            jac[i, 3, 3] = uyq[4]

    # R-16: publish the dual primitives for the numba<->NumPy parity pin
    # (see the ``_ADRT_NUMBA_PRIMS`` note at module level).
    global _ADRT_NUMBA_PRIMS
    _ADRT_NUMBA_PRIMS = {
        'dadd': _dadd, 'dsub': _dsub, 'dmul': _dmul, 'ddiv': _ddiv,
        'dscale': _dscale, 'daddc': _daddc, 'dneg': _dneg,
        'dsqrt': _dsqrtq,
    }
    return _kernel


def _adrt_numba_kernel():
    """Return the compiled numba kernel or ``None`` if numba is unavailable /
    the kernel failed to build (one attempt, memoized)."""
    global _ADRT_NUMBA_KERNEL, _ADRT_NUMBA_STATE
    if _ADRT_NUMBA_STATE is not None:
        return _ADRT_NUMBA_KERNEL if _ADRT_NUMBA_STATE else None
    try:
        import numba  # noqa: F401
    except ImportError:
        _ADRT_NUMBA_STATE = False
        return None
    try:
        _ADRT_NUMBA_KERNEL = _build_adrt_numba_kernel()
        _ADRT_NUMBA_STATE = True
    except Exception:                                     # pragma: no cover
        _ADRT_NUMBA_KERNEL = None
        _ADRT_NUMBA_STATE = False
    return _ADRT_NUMBA_KERNEL


def _adrt_numba(x, y, ux, uy, surfaces, wavelength):
    """numba forward-AD twin of :func:`_adrt_numpy` for the composite
    (``per_surface=False``) all-conic path.  Bit-for-(near)-bit identical to the
    ``_AdrtDual`` result (see the kernel docstring)."""
    from ..glass import get_glass_index

    kern = _adrt_numba_kernel()
    if kern is None:
        return None
    x = np.asarray(x, np.float64)
    n = x.shape[0]
    y = np.broadcast_to(np.asarray(y, np.float64), (n,)).copy()
    ux = np.broadcast_to(np.asarray(ux, np.float64), (n,)).copy()
    uy = np.broadcast_to(np.asarray(uy, np.float64), (n,)).copy()
    x = np.ascontiguousarray(np.broadcast_to(x, (n,)))
    y = np.ascontiguousarray(y)
    ux = np.ascontiguousarray(ux)
    uy = np.ascontiguousarray(uy)

    nsurf = len(surfaces)
    radius = np.empty(nsurf, np.float64)
    conic = np.empty(nsurf, np.float64)
    ismir = np.empty(nsurf, np.bool_)
    n1a = np.empty(nsurf, np.float64)
    n2a = np.empty(nsurf, np.float64)
    thick = np.empty(nsurf, np.float64)
    semidia = np.empty(nsurf, np.float64)
    applyt = np.empty(nsurf, np.bool_)
    for si, s in enumerate(surfaces):
        radius[si] = float(getattr(s, 'radius', np.inf))
        conic[si] = float(getattr(s, 'conic', 0.0) or 0.0)
        ismir[si] = bool(getattr(s, 'is_mirror', False))
        n1a[si] = float(get_glass_index(
            getattr(s, 'glass_before', None) or 'air', wavelength))
        n2a[si] = float(get_glass_index(
            getattr(s, 'glass_after', None) or 'air', wavelength))
        thick[si] = float(getattr(s, 'thickness', 0.0) or 0.0)
        semidia[si] = float(getattr(s, 'semi_diameter', np.inf))
        applyt[si] = si < nsurf - 1

    jac = np.zeros((n, 4, 4), np.float64)
    ox = np.empty(n, np.float64)
    oy = np.empty(n, np.float64)
    oux = np.empty(n, np.float64)
    ouy = np.empty(n, np.float64)
    oopd = np.empty(n, np.float64)
    oalive = np.empty(n, np.bool_)
    kern(x, y, ux, uy, radius, conic, ismir, n1a, n2a, thick, semidia, applyt,
         jac, ox, oy, oux, ouy, oopd, oalive)
    # A dead ray (missed / TIR) can leave non-finite entries; zero them so they
    # cannot contaminate array-wide ops downstream (matches _adrt_numpy).
    jac = np.nan_to_num(jac, nan=0.0, posinf=0.0, neginf=0.0)
    ox, oy, oux, ouy, oopd = (np.nan_to_num(v)
                              for v in (ox, oy, oux, ouy, oopd))
    return DifferentialTransfer(jacobian=jac, x=ox, y=oy, ux=oux, uy=ouy,
                                opd=oopd, alive=oalive)


def ray_transfer_jacobian_analytic(
    x, y, ux, uy, surfaces, wavelength, *, per_surface: bool = False,
    reference: str = 'surface', n_exit: Optional[float] = None,
):
    """Analytic (exact) differential ray-transfer Jacobian -- the closed-form /
    autodiff twin of the finite-difference :func:`ray_transfer_jacobian`.

    Forward-mode AD over the EXACT conic trace (intersection + vector Snell /
    reflection + vertex transfer), so the 4x4 ``(x, y, ux, uy)`` ray-transfer
    Jacobian is computed WITHOUT finite-difference truncation (the ``h -> 0``
    limit) and is correct at all NA.  On the JAX backend (``x`` a jax array) the
    trace is differentiated by ``jax.jacfwd`` and is itself ``jax.grad`` /
    ``jit`` friendly.

    Value vs the two existing primitives: it is exact where the FD
    :func:`ray_transfer_jacobian` carries ~1e-8 truncation, and pure NumPy (no
    JAX dependency, unlike :func:`ray_transfer_jacobian_jax`).  It is the
    closed-form realization of the differential ray tracing of Stone & Forbes
    (forward-mode AD == analytic differential ray tracing, Volatier 2017); see
    ``docs/ANALYTIC_DIFFERENTIAL_RAY_TRACING_LITERATURE.md``.  (Note: the
    *composed output* of :func:`ray_transfer_jacobian_jax` is empirically also
    exact at high NA -- its per-surface paraxial transfer under-count is undone
    by the next surface's intersection -- so this is not a high-NA correction of
    that path, just a NumPy-native, truncation-free one with a cleaner
    forward-AD structure.)

    Same signature / return (:class:`DifferentialTransfer`) and
    ``per_surface`` / ``reference`` / ``n_exit`` semantics as
    :func:`ray_transfer_jacobian`; agrees with it to the FD truncation floor
    (~1e-8) on BOTH reference planes -- the two backends share
    :func:`_project_to_exit_vertex_plane`, so ``reference='exit_vertex'``
    cannot drift between them.  On axis the 2x2 meridional block equals
    ``system_abcd_prescription`` (exactly for air-to-air prescriptions, where
    the unreduced slope ``u = L/N`` coincides with the reduced ``n*u`` momentum
    at the ``n = 1`` endpoints).  Refracting / reflecting surfaces must be conic
    (sphere / conic ``+`` thickness ``+`` glass ``+`` ``is_mirror``); **Zemax
    coordinate breaks** (``is_coordbrk`` -- decenter + X/Y/Z tilts, a smooth
    frame transform) ARE handled and differentiable, giving alignment /
    tolerancing sensitivity through a fold (a *large* tilt shares the slope-
    space caveat: ``u = L/N`` degenerates as the folded ``N -> 0``).
    EVEN-power aspheric-polynomial departures are handled: the conic root
    seeds a fixed 6-step Newton refinement onto ``conic + polynomial``, and
    the normal carries the polynomial gradient
    (:func:`_adrt_aspheric_intersect`).  Because the iteration itself is
    differentiated, the Jacobian is exact rather than
    converged-value-only -- cross-checked against the FD primitive on an
    A4 / A6 singlet.  An aspheric surface takes the ``_AdrtDual`` (or JAX)
    path: the numba kernel's inlined conic primitives carry no polynomial
    departure, so it is excluded there.  Freeforms and biconics are still
    not handled (use the FD primitive there).

    Returns
    -------
    DifferentialTransfer
    """
    _validate_reference(reference, 'ray_transfer_jacobian_analytic')
    from ..backend.array import is_jax_array
    for s in surfaces:
        # _adrt_step reads ``radius`` / ``conic`` / ``aspheric_coeffs`` --
        # all rotationally symmetric -- for refracting/reflecting surfaces
        # (coordinate breaks are handled separately), so a biconic
        # ``radius_y`` / ``conic_y`` / ``aspheric_coeffs_y`` or a freeform
        # must be rejected (else a biconic would be silently traced as if it
        # were rotationally symmetric, giving a wrong y-axis power).
        # N10a: a FIELD-FRAME decenter / tilt / freeform sag_callable breaks the
        # rotational symmetry the analytic conic ``_adrt_step`` assumes -- reject
        # so ``jacobian='auto'`` falls back to the finite-difference primitive
        # (which traces through the shared field-frame ``_surface_sag_xy`` and
        # therefore carries the decenter / tilt walk-off correctly).
        _ff = (getattr(s, 'field_sag_callable', None) is not None
               or (getattr(s, 'field_decenter', None) is not None
                   and tuple(float(v) for v in s.field_decenter) != (0.0, 0.0))
               or (getattr(s, 'field_tilt', None) is not None
                   and tuple(float(v) for v in s.field_tilt) != (0.0, 0.0)))
        if (getattr(s, 'freeform', None)
                or getattr(s, 'radius_y', None) is not None
                or getattr(s, 'conic_y', None) is not None
                or getattr(s, 'aspheric_coeffs_y', None) is not None
                or _ff):
            raise NotImplementedError(
                'ray_transfer_jacobian_analytic handles rotationally-symmetric '
                'conic + even-aspheric surfaces (plus coordinate breaks) only; '
                'freeforms, biconic (radius_y / conic_y / aspheric_coeffs_y) '
                'and field-frame decenter / tilt surfaces are not yet '
                'supported -- use ray_transfer_jacobian (FD) for those.')
    # NB (R2, AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11): ``_adrt_step``
    # starts every ray at ``opd = 0`` on the ``z = 0`` launch PLANE and
    # accumulates only the surface legs -- the same convention as
    # ``_make_bundle``'s default ``opd_seed='plane'``, which is what the
    # finite-difference twin :func:`ray_transfer_jacobian` launches with.
    # The two therefore stay OPL-identical (pinned to 1e-12 m by
    # ``tests/unit/test_analytic_ray_transfer.py``).  A caller that wants
    # the ENTRANCE-EIKONAL convention (OPL measured from the incident
    # wavefront -- the only one under which cross-ray OPL differences are
    # a wavefront error for a TILTED bundle) must add the same term to
    # BOTH primitives' ``opd``; see ``trace.seed_entrance_eikonal``.
    if is_jax_array(x) or is_jax_array(y) or is_jax_array(ux) \
            or is_jax_array(uy):
        return _finish_analytic(
            _adrt_jax(x, y, ux, uy, surfaces, wavelength, per_surface),
            surfaces, wavelength, reference, n_exit)
    # B4: the composite all-conic path (the adaptive-FGA exact_jacobian hot
    # spot) runs the numba forward-AD kernel when numba is available -- ULP-
    # identical to the ``_AdrtDual`` result, ~order-of-magnitude faster.  Any
    # miss (per_surface, a coordinate break, or numba unavailable / a build
    # failure) falls through to the pure-NumPy dual implementation.
    if not per_surface and _adrt_surfaces_numba_eligible(surfaces):
        # R-7 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): numba compiles with its
        # default ``error_model='python'``, so a DEGENERATE bundle (a slope so
        # large that N = 1/sqrt(1+u^2) underflows to 0, or a non-finite slope)
        # makes the kernel's ``1.0 / b[0]`` RAISE ZeroDivisionError -- while
        # the ``_AdrtDual`` sibling it is supposed to be bit-identical to runs
        # under ``np.errstate(divide='ignore', invalid='ignore')`` and returns
        # the documented masked/``nan_to_num``'d result (matching the FD
        # :func:`ray_transfer_jacobian`, which also copes).  Route that
        # numba-only failure into the SAME documented fallback the kernel
        # already uses for every other miss (see ``_adrt_numba`` -> ``None``):
        # recompute on the pure-NumPy dual path.  Non-degenerate bundles never
        # reach the except arm, so the ULP-parity of the fast path is intact.
        try:
            dt = _adrt_numba(x, y, ux, uy, surfaces, wavelength)
        except ZeroDivisionError:
            dt = None
        if dt is not None:
            return _finish_analytic(dt, surfaces, wavelength, reference,
                                    n_exit)
    return _finish_analytic(
        _adrt_numpy(x, y, ux, uy, surfaces, wavelength, per_surface),
        surfaces, wavelength, reference, n_exit)


def _finish_analytic(dt, surfaces, wavelength, reference, n_exit):
    """Re-reference an analytic backend's result onto the requested plane.

    One place, so the numba kernel, the ``_AdrtDual`` path and the JAX path
    cannot disagree about where their state and Jacobian live."""
    if reference == 'surface':
        return dt
    return _project_to_exit_vertex_plane(
        dt, surfaces, wavelength, n_exit,
        "ray_transfer_jacobian_analytic(reference='exit_vertex')")


def _adrt_jax(x, y, ux, uy, surfaces, wavelength, per_surface):
    import jax
    import jax.numpy as jnp
    if per_surface:
        raise NotImplementedError(
            'ray_transfer_jacobian_analytic: per_surface=True is NumPy-only; '
            'the JAX path returns the composite Jacobian (use jax.jacfwd on the '
            'per-surface steps if per-surface gradients are needed).')
    jnp_ops = {'sqrt': lambda z: jnp.sqrt(jnp.maximum(z, 0.0)),
               'val': lambda a: a, 'pwhere': jnp.where, 'dwhere': jnp.where}
    nsurf = len(surfaces)

    def _full(s4):
        """State + accumulated OPL.  ``opd`` rides as ``jacfwd`` AUX so
        one forward pass yields the Jacobian, the exit state AND the OPL.

        A separate ``jax.jacfwd(_state)`` plus a second ``vmap(_full)``
        walks the whole prescription TWICE.
        ``jax.jacfwd(..., has_aux=True)`` returns the primal outputs of
        the same forward pass alongside the Jacobian, so the second walk
        differentiates only the FIRST return value, so the exit state is
        returned as aux as well and the derivative target is the same
        ``jnp.stack([xx, yy, uxx, uyy])`` as before -- the Jacobian is
        bit-identical.
        """
        xx, yy, uxx, uyy = s4[0], s4[1], s4[2], s4[3]
        opd = jnp.zeros(())
        for si, s in enumerate(surfaces):
            xx, yy, uxx, uyy, dopd, _dead = _adrt_step(
                xx, yy, uxx, uyy, s, wavelength, si < nsurf - 1, jnp_ops,
                compute_dead=False)
            opd = opd + dopd
        state = jnp.stack([xx, yy, uxx, uyy])
        return state, (state, opd)

    s4 = jnp.stack([jnp.reshape(jnp.asarray(x), (-1,)),
                    jnp.reshape(jnp.asarray(y), (-1,)),
                    jnp.reshape(jnp.asarray(ux), (-1,)),
                    jnp.reshape(jnp.asarray(uy), (-1,))], axis=0)
    n = s4.shape[1]
    jac, (st, opd) = jax.vmap(jax.jacfwd(_full, has_aux=True),
                              in_axes=1, out_axes=(0, (0, 0)))(s4)
    return DifferentialTransfer(
        jacobian=jac, x=st[:, 0], y=st[:, 1], ux=st[:, 2], uy=st[:, 3],
        opd=opd, alive=jnp.ones((n,), dtype=bool))


__all__ = ['DifferentialTransfer', 'ray_transfer_jacobian',
           'ray_transfer_jacobian_analytic', 'ray_transfer_jacobian_jax']
