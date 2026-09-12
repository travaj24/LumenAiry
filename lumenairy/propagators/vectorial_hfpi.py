"""
lumenairy.propagators.vectorial_hfpi -- vectorial Huygens-Fresnel
Path Integration.

Extends the scalar HFPI (in :mod:`lumenairy.propagators.hfpi`) to
vector electromagnetic fields by carrying a full three-component
electric field ``(Ex, Ey, Ez)`` with every path.  At emission and at
every re-emission the field is carried onto the path's own transverse
plane by the RIGID ROTATION that takes the previous propagation
direction to the new one (Rodrigues; the aplanatic
``R_z(phi) R_y(theta) R_z(-phi)`` of the Richards-Wolf formulation when
the previous direction is ``+z``).  That rotation is norm-preserving, so
it adds no amplitude of its own -- the scalar Kirchhoff obliquity stays
the only amplitude factor -- and it produces the two things a vector
propagator exists for: DEPOLARISATION away from the axis, and a genuine
longitudinal ``Ez``.

What this module does NOT do
---------------------------
It is not a rigorous dyadic-Green / Stratton-Chu solver: the aperture
model is still a hard scalar mask with an isotropic re-emission cone,
there is no material response at the aperture edge, and nothing here is
validated against a rigorous vector diffraction code.  For a rigorous
high-NA vector FOCUS use
:func:`lumenairy.propagators.vector_diffraction.richards_wolf_focus`,
which evaluates the Debye-Wolf integral directly and carries ``Ez``
exactly.  For anything birefringent or polarisation-selective, apply the
element's Jones operator to the field and use the scalar propagator per
component; this module models no such element.

.. versionchanged:: 5.46
    Audit K17.  Before this release the module's header advertised "the
    m-theory dipole obliquity tensor for vector-correct secondary-source
    amplitudes" and listed high-NA imaging, cascaded polarizing elements
    and birefringent elements as cases that REQUIRE it.  No such tensor
    existed: the default path multiplied BOTH Jones components by the
    same scalar ``0.5*(cos theta_in + cos theta_out)``, i.e. by the
    identity as far as polarisation is concerned.  Measured -- 24x24
    source, dx = 2 um, lambda = 633 nm, two 200 um legs around a 40 um
    aperture, 60 000 paths, same seed -- the vector ``Ex`` output was
    bit-identical to a scalar :func:`~lumenairy.propagators.hfpi.
    propagate_hfpi` run on ``Ex_in`` (max|diff| = 1.9e-23), and a
    45-degree linear input showed ``|Ey/Ex - 1| <= 1.1e-16`` over the
    WHOLE output grid: zero depolarisation, anywhere, at twice the cost
    of the scalar propagator.  The opt-in ``vector_projection=True``
    path did rotate, but dropped the longitudinal component it created
    (measured 8.8 % of the incident ``|E|^2`` at a 0.8 rad cone), then
    multiplied by the scalar obliquity a second time, and never
    projected at emission at all.

Author: Andrew Traverso
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np

from ..backend import (
    RandomState,
    array_namespace,
    is_jax_array,
    to_numpy,
)
from .hfpi import (
    _bin_paths,
    _binning_jacobian,
    _check_landed,
    _complex_output_dtype,
    _resolve_output_shape,
    _spawn_rng,
)


def _rigid_rotate(Ex, Ey, Ez, s_from, s_to, xp):
    """Rotate a 3-component field rigidly from one unit direction to
    another (K17).

    The rotation is Rodrigues' formula about ``a = s_from x s_to``
    through the angle between them::

        E' = E cos(t) + (a_hat x E) sin(t) + a_hat (a_hat . E)(1 - cos(t))

    with ``cos(t) = s_from . s_to``.  It is the SHORTEST rotation taking
    ``s_from`` to ``s_to``, it is orthogonal (so ``|E'| = |E|`` exactly --
    it introduces no amplitude factor and cannot double-count the scalar
    obliquity), and for ``s_from = +z`` it reduces term by term to the
    aplanatic ``R_z(phi) R_y(theta) R_z(-phi)`` matrix the Richards-Wolf
    formulation uses::

        Ex' = Ex (cos t cos^2 phi + sin^2 phi) + Ey cos phi sin phi (cos t - 1)
        Ey' = Ex cos phi sin phi (cos t - 1) + Ey (cos t sin^2 phi + cos^2 phi)
        Ez' = -sin t (cos phi Ex + sin phi Ey)

    where ``(phi, t)`` is the RAY-DIRECTION azimuth / polar angle -- the
    convention in which ``Ez`` carries its textbook sign (Novotny & Hecht
    eq. 3.66; see the ``pupil`` note in
    :func:`~lumenairy.propagators.vector_diffraction.richards_wolf_focus`
    for the aperture-azimuth counterpart, audit K16).

    Anti-parallel or parallel ``s_from``/``s_to`` (``|a| = 0``) are
    handled by falling back to the identity, which is exact for the
    parallel case and the only sane choice for the anti-parallel one
    (a backward-going path is already dead in this pipeline).
    """
    ax = s_from[..., 1] * s_to[..., 2] - s_from[..., 2] * s_to[..., 1]
    ay = s_from[..., 2] * s_to[..., 0] - s_from[..., 0] * s_to[..., 2]
    az = s_from[..., 0] * s_to[..., 1] - s_from[..., 1] * s_to[..., 0]
    a_norm = xp.sqrt(ax * ax + ay * ay + az * az)
    degenerate = a_norm < 1e-15
    safe = xp.where(degenerate, 1.0, a_norm)
    ax = ax / safe
    ay = ay / safe
    az = az / safe
    ct = (s_from[..., 0] * s_to[..., 0]
          + s_from[..., 1] * s_to[..., 1]
          + s_from[..., 2] * s_to[..., 2])
    st = a_norm
    cdt = Ex.dtype
    ct_c = ct.astype(cdt)
    st_c = st.astype(cdt)
    ax_c, ay_c, az_c = ax.astype(cdt), ay.astype(cdt), az.astype(cdt)
    # a_hat x E
    cx = ay_c * Ez - az_c * Ey
    cy = az_c * Ex - ax_c * Ez
    cz = ax_c * Ey - ay_c * Ex
    # a_hat . E
    dot = ax_c * Ex + ay_c * Ey + az_c * Ez
    one_m = (1.0 - ct_c)
    rx = Ex * ct_c + cx * st_c + ax_c * dot * one_m
    ry = Ey * ct_c + cy * st_c + ay_c * dot * one_m
    rz = Ez * ct_c + cz * st_c + az_c * dot * one_m
    keep = degenerate
    return (xp.where(keep, Ex, rx),
            xp.where(keep, Ey, ry),
            xp.where(keep, Ez, rz))


@dataclass
class VectorPathBundle:
    """Vectorial counterpart to :class:`PathBundle` -- carries the full
    three-component electric field with every path.

    Attributes
    ----------
    positions : array (N, 3)
    directions : array (N, 3)
    Ex, Ey : array (N,) complex
        Transverse field components in the lab frame.
    opl : array (N,)
    alive : array (N,) bool
    Ez : array (N,) complex, optional
        Longitudinal field component in the lab frame (K17, v5.46).
        ``None`` means "no longitudinal component yet" and is treated as
        zeros; it is the default so that positional construction of the
        pre-v5.46 six-field bundle keeps working.
    """

    positions: object
    directions: object
    Ex: object
    Ey: object
    opl: object
    alive: object
    Ez: object = None
    # K13: geometric distance since the last emission / re-emission; see
    # :class:`~lumenairy.propagators.hfpi.PathBundle`.
    leg: object = None

    def ez_or_zeros(self):
        """``Ez``, or an all-zero array matching ``Ex`` when absent."""
        if self.Ez is not None:
            return self.Ez
        xp = array_namespace(self.Ex)
        return xp.zeros_like(self.Ex)

    def leg_or_opl(self):
        """Current-leg geometric length, falling back to ``opl``."""
        return self.opl if self.leg is None else self.leg

    def __len__(self) -> int:
        try:
            return int(self.positions.shape[0])
        except (AttributeError, TypeError, IndexError):
            return 0

    @property
    def n_alive(self) -> int:
        if self.alive is None:
            return len(self)
        return int(np.sum(to_numpy(self.alive)))


def init_vector_paths_from_field(
    Ex_in: np.ndarray,
    Ey_in: np.ndarray,
    dx: float,
    *,
    n_paths: int,
    wavelength: float,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    z_input_plane: float = 0.0,
    vector_projection: bool = True,
) -> VectorPathBundle:
    """Sample paths from a 2-component (Ex, Ey) Jones source field.

    Each path is initialised at a randomly chosen pixel with a
    forward-cone direction (uniform on the spherical cap), and
    inherits the source-pixel's Jones vector ``(Ex[i,j], Ey[i,j])``
    scaled by the obliquity factor ``cos(theta)`` and the pixel
    area ``dx**2``.

    Parameters
    ----------
    Ex_in, Ey_in : array (Ny, Nx) complex
        Jones vector components of the source field, transverse to +z.
    vector_projection : bool, default True
        Carry each path's field onto its OWN transverse plane by the
        rigid rotation taking ``+z`` to the emission direction
        (:func:`_rigid_rotate`).  This is what makes the bundle
        vectorial: it is where depolarisation and the longitudinal
        ``Ez`` come from.  ``False`` reproduces the pre-v5.46 behaviour
        (both Jones components scaled by the same scalar, ``Ez`` absent),
        which is bit-identical to two scalar HFPI runs.

        .. versionchanged:: 5.46
            Added, defaulting to ``True`` (audit K17).  There was NO
            projection at emission before v5.46, so a path leaving at
            angle ``theta`` carried a field with ``E . s_hat != 0`` --
            not a transverse field even before the first aperture.
    Same other parameters as :func:`init_paths_from_field`.

    Returns
    -------
    VectorPathBundle
    """
    xp = array_namespace(Ex_in, Ey_in)
    if Ex_in.shape != Ey_in.shape:
        raise ValueError("Ex_in and Ey_in must have matching shape.")
    Ny, Nx = Ex_in.shape[-2], Ex_in.shape[-1]
    # K19 (audit 2026-09-11): ``rng=None`` must draw fresh system
    # entropy -- see the scalar twin for the measurement.
    rs = RandomState(rng=rng)

    iy = rs.integers((n_paths,), low=0, high=Ny)
    ix = rs.integers((n_paths,), low=0, high=Nx)
    iy_xp = xp.asarray(iy)
    ix_xp = xp.asarray(ix)

    x_s = (ix_xp - Nx / 2) * dx
    y_s = (iy_xp - Ny / 2) * dx
    z_s = xp.full((n_paths,), float(z_input_plane), dtype=x_s.dtype)
    positions = xp.stack([x_s, y_s, z_s], axis=-1)

    cos_max = float(np.cos(cone_half_angle))
    u = rs.uniform((n_paths,))
    phi = rs.uniform((n_paths,), low=0.0, high=2 * float(np.pi))
    cos_theta = 1.0 - xp.asarray(u) * (1.0 - cos_max)
    sin_theta = xp.sqrt(xp.maximum(1.0 - cos_theta ** 2, 0.0))
    L = sin_theta * xp.cos(xp.asarray(phi))
    M = sin_theta * xp.sin(xp.asarray(phi))
    N = cos_theta
    directions = xp.stack([L, M, N], axis=-1)

    sx = Ex_in[iy_xp, ix_xp]
    sy = Ey_in[iy_xp, ix_xp]
    # 4.11.2: full HF Kirchhoff weighting per Jones component (mirrors
    # the scalar :func:`lumenairy.propagators.hfpi.init_paths_from_field`
    # 4.10 fix).  Pre-4.11.2 the vector source weights carried only the
    # cosθ·dx² obliquity factor -- the ``1/(iλ)`` prefactor and the
    # Monte Carlo solid-angle weight ``2π(1-cosθ_max)/N_paths`` were
    # absent, so absolute Jones amplitudes were unphysical by ~10^6 at
    # visible wavelengths.  Relative polarization structure was
    # unaffected (factor is global).
    # K18 (audit 2026-09-11): the SOURCE-AREA factor -- see the scalar
    # :func:`lumenairy.propagators.hfpi.init_paths_from_field`.
    solid_angle = (2.0 * float(np.pi) * (1.0 - cos_max)
                   * (float(Ny) * float(Nx)) / float(n_paths))
    inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
    kirchhoff = complex(inv_i_lambda) * solid_angle
    obl = cos_theta * (dx * dx)
    obl_complex = obl.astype(sx.dtype) if hasattr(obl, 'astype') else obl
    Ex_paths = sx * obl_complex * kirchhoff
    Ey_paths = sy * obl_complex * kirchhoff
    Ez_paths = xp.zeros_like(Ex_paths)

    if vector_projection:
        # K17: carry the field onto the path's own transverse plane.
        # Rigid (norm-preserving) rotation from +z to the emission
        # direction -- no amplitude factor, so the scalar obliquity above
        # is not double-counted.
        z_hat = xp.zeros_like(directions)
        z_hat = xp.stack([xp.zeros_like(cos_theta),
                          xp.zeros_like(cos_theta),
                          xp.ones_like(cos_theta)], axis=-1)
        Ex_paths, Ey_paths, Ez_paths = _rigid_rotate(
            Ex_paths, Ey_paths, Ez_paths, z_hat, directions, xp)

    opl = xp.zeros((n_paths,), dtype=xp.real(sx).dtype)
    alive = xp.ones((n_paths,), dtype=bool)

    return VectorPathBundle(
        positions=positions,
        directions=directions,
        Ex=Ex_paths,
        Ey=Ey_paths,
        opl=opl,
        alive=alive,
        Ez=Ez_paths,
        leg=xp.zeros_like(opl),
    )


def propagate_vector_to_plane(
    paths: VectorPathBundle,
    z_target: float,
    wavelength: float,
    *,
    n_medium: float = 1.0,
) -> VectorPathBundle:
    """Free-space advance of every alive vector-path to ``z_target``.

    The Jones vector picks up a global phase ``exp(i k OPL)``;
    polarization-rotation-by-propagation effects (which the
    full m-theory dipole formalism captures for general directions)
    are neglected for paraxial advances.  At each diffracting
    surface, see :func:`apply_vector_aperture_diffraction`.
    """
    xp = array_namespace(paths.positions)
    z_curr = paths.positions[..., 2]
    Nz = paths.directions[..., 2]
    eps = 1e-30
    t = (z_target - z_curr) / xp.where(xp.abs(Nz) > eps, Nz, eps)
    new_alive = paths.alive & (t >= 0) & (xp.abs(Nz) > eps)
    # 4.13.2 (P1-NEW-H): zero the step for grazing / dead rays so their
    # position update is a no-op.  Mirrors the scalar hfpi fix; see
    # :func:`lumenairy.propagators.hfpi.propagate_to_plane` for details.
    t = xp.where(new_alive, t, 0.0)
    new_positions = paths.positions + t[..., None] * paths.directions
    delta_opl = n_medium * xp.abs(t)
    new_opl = paths.opl + delta_opl
    k = 2 * float(np.pi) / wavelength
    phase = xp.exp(1j * k * delta_opl).astype(paths.Ex.dtype)
    _base_leg = (paths.leg if paths.leg is not None
                 else xp.zeros_like(paths.opl))
    return VectorPathBundle(
        positions=new_positions,
        directions=paths.directions,
        Ex=paths.Ex * phase,
        Ey=paths.Ey * phase,
        opl=new_opl,
        alive=new_alive,
        # K17: the longitudinal component rides the same global phase.
        Ez=(paths.Ez * phase) if paths.Ez is not None else None,
        # K13: geometric step is |t| (unit direction vectors).
        leg=_base_leg + xp.abs(t),
    )


def apply_vector_aperture_diffraction(
    paths: VectorPathBundle,
    aperture_radius: float,
    *,
    centre: Tuple[float, float] = (0.0, 0.0),
    shape: str = 'circular',
    wavelength: float,
    rng: Optional[Union[int, object]] = None,
    cone_half_angle: float = np.pi / 2 - 1e-6,
    vector_projection: bool = True,
) -> VectorPathBundle:
    """Vectorial counterpart of :func:`apply_aperture_diffraction`.

    Paths landing outside the aperture are killed.  Surviving paths
    re-emit secondary HF sources at their current position with
    fresh forward-cone directions; their Jones vector is multiplied
    by ``cos(theta_new)`` to account for the m-theory dipole
    obliquity, and the OPL accumulator is reset.

    Parameters
    ----------
    wavelength : float, keyword-only, REQUIRED
        Vacuum wavelength [m], strictly positive.

        .. versionchanged:: 5.46
            No longer defaults to ``0.0`` (audit K12) -- see the scalar
            :func:`lumenairy.propagators.hfpi.apply_aperture_diffraction`
            for the measurement (every weight wrong by 1/lambda in
            magnitude and -90 degrees in phase, silently).
    vector_projection : bool, default True
        Carry the incoming three-component field onto the NEW direction's
        transverse plane by the rigid rotation taking ``s_in`` to
        ``s_out`` (:func:`_rigid_rotate`).  ``False`` reproduces the
        pre-v5.46 default (both components scaled by the same scalar,
        no ``Ez``).

        .. versionchanged:: 5.46
            Default flipped ``False`` -> ``True``, and the operation
            changed (audit K17).  v5.4.6's opt-in used the ORTHOGONAL
            projection ``E - (E.rho_hat) rho_hat`` and then DISCARDED the
            longitudinal component it created -- measured 8.8 % of the
            incident ``|E|^2`` lost at a 0.8 rad cone, the two-component
            sum falling to 0.788x its unprojected value -- and then
            multiplied by the scalar obliquity a second time, double
            counting what the projection already is.  The rigid rotation
            is orthogonal, so it conserves ``|E|^2`` exactly and adds no
            amplitude factor; the scalar Kirchhoff obliquity below stays
            the only amplitude term.
    """
    if not (wavelength > 0) or not np.isfinite(wavelength):
        raise ValueError(
            f"apply_vector_aperture_diffraction: wavelength must be a "
            f"positive finite length in metres (got {wavelength!r}).  It "
            f"scales the 1/(i*lambda) Kirchhoff prefactor applied to every "
            f"re-emitted path.")
    xp = array_namespace(paths.positions)
    # K19: ``rng=None`` draws fresh system entropy.
    rs = RandomState(rng=rng)

    cx, cy = centre
    x = paths.positions[..., 0] - cx
    y = paths.positions[..., 1] - cy

    if shape == 'circular':
        in_aperture = x * x + y * y <= aperture_radius * aperture_radius
    elif shape == 'square':
        in_aperture = (xp.abs(x) <= aperture_radius) & (xp.abs(y) <= aperture_radius)
    else:
        raise ValueError(
            f"shape must be 'circular' or 'square', got {shape!r}.")

    survives = paths.alive & in_aperture
    n = int(paths.positions.shape[0])
    cos_max = float(np.cos(cone_half_angle))
    u = rs.uniform((n,))
    phi = rs.uniform((n,), low=0.0, high=2 * float(np.pi))
    cos_theta = 1.0 - xp.asarray(u) * (1.0 - cos_max)
    sin_theta = xp.sqrt(xp.maximum(1.0 - cos_theta ** 2, 0.0))
    L = sin_theta * xp.cos(xp.asarray(phi))
    M = sin_theta * xp.sin(xp.asarray(phi))
    Nz = cos_theta
    new_directions = xp.stack([L, M, Nz], axis=-1)

    # 4.11.2: include the Kirchhoff ``1/(iλ)·dΩ`` factor per
    # re-emission, matching the scalar
    # :func:`lumenairy.propagators.hfpi.apply_aperture_diffraction`
    # 4.11.2 fix.  Pre-4.11.2 cascaded vector apertures dropped this
    # global factor, so multi-aperture vector HFPI underweighted by
    # ~10^6 per extra aperture at visible wavelengths.  Relative
    # polarization / phase structure unaffected.
    cos_theta_in = paths.directions[..., 2]
    obliquity = 0.5 * (cos_theta_in + cos_theta)
    solid_angle = 2.0 * float(np.pi) * (1.0 - cos_max) / float(n)
    inv_i_lambda = (1.0 / (1j * wavelength)) if wavelength > 0 else 1.0
    kirchhoff = complex(inv_i_lambda) * solid_angle
    obl = obliquity.astype(paths.Ex.dtype)
    # K17: rigid (norm-preserving) rotation of the full three-component
    # field from the incoming to the outgoing direction.  Unlike the
    # v5.4.6 orthogonal projection it keeps the longitudinal component it
    # creates, and it introduces no amplitude factor -- so the scalar
    # Kirchhoff obliquity ``obl`` is applied exactly ONCE.
    Ez_in = paths.ez_or_zeros()
    if vector_projection:
        Ex_t, Ey_t, Ez_t = _rigid_rotate(
            paths.Ex, paths.Ey, Ez_in,
            paths.directions, new_directions, xp)
    else:
        Ex_t, Ey_t, Ez_t = paths.Ex, paths.Ey, Ez_in
    new_Ex = Ex_t * obl * kirchhoff
    new_Ey = Ey_t * obl * kirchhoff
    new_Ez = Ez_t * obl * kirchhoff
    new_opl = xp.zeros_like(paths.opl)
    return VectorPathBundle(
        positions=paths.positions,
        directions=new_directions,
        Ex=new_Ex,
        Ey=new_Ey,
        opl=new_opl,
        alive=survives,
        Ez=new_Ez,
        # K13: a re-emission starts a new leg.
        leg=xp.zeros_like(paths.opl),
    )


def accumulate_vector_to_grid(
    paths: VectorPathBundle,
    *,
    Ny: int,
    Nx: int,
    dx: float,
    centre: Tuple[float, float] = (0.0, 0.0),
    output_dtype: Optional[Any] = None,
    on_undersampled: str = 'warn',
    normalisation: str = 'physical',
    return_ez: bool = False,
):
    """Coherently bin a VectorPathBundle into separate output grids.

    Returns
    -------
    Ex_out, Ey_out : tuple of arrays (Ny, Nx) complex
        With ``return_ez=True``, ``(Ex_out, Ey_out, Ez_out)``.

    Parameters
    ----------
    on_undersampled : {'warn', 'silent', 'error'}, default 'warn'
        The v5.31 sampling-adequacy guard, via the shared
        :func:`~lumenairy.propagators.hfpi._check_landed`.

        .. versionadded:: 5.46
            Audit K23.  v4.13.1 forked this accumulator from the scalar
            one for index sharing, so the guard v5.31 added there never
            ran here: measured on identical geometry, 20 000 paths onto a
            64x64 grid gave 2 non-zero pixels (0.05 %) with the scalar
            path warning and the vector path silent, and the vector entry
            point accepted no ``on_undersampled`` kwarg at all.
    normalisation : {'physical', 'legacy'}, default 'physical'
        The K13 output-binning Jacobian, via the shared
        :func:`~lumenairy.propagators.hfpi._binning_jacobian`.  See the
        scalar :func:`~lumenairy.propagators.hfpi.accumulate_to_grid`.
    return_ez : bool, default False
        Also return the longitudinal component ``Ez_out`` (K17, v5.46).
        Kept opt-in so the historical 2-tuple return contract is
        unchanged.

    Notes
    -----
    v4.13.1 perf: shares the pixel index computation between the
    components; v5.46 moves that computation into the shared
    :func:`~lumenairy.propagators.hfpi._bin_paths` so scalar and vector
    cannot drift again (audit K24).
    """
    if normalisation not in ('physical', 'legacy'):
        raise ValueError(
            f"accumulate_vector_to_grid: normalisation must be 'physical' "
            f"or 'legacy'; got {normalisation!r}.")
    if output_dtype is None:
        output_dtype = paths.Ex.dtype

    xp = array_namespace(paths.positions)
    Ez_in = paths.ez_or_zeros()

    flat_idx, inside = _bin_paths(paths.positions, paths.alive,
                                  Ny, Nx, dx, centre)
    # K23: the guard, shared with the scalar accumulator.
    _check_landed(inside, Ny, Nx, on_undersampled,
                  'accumulate_vector_to_grid', positions=paths.positions)

    Ex_w, Ey_w, Ez_w = paths.Ex, paths.Ey, Ez_in
    if normalisation == 'physical':
        # K13: the output-binning Jacobian.  ``_binning_jacobian`` reads
        # ``opl`` and ``directions``, both present on a VectorPathBundle.
        jac = _binning_jacobian(paths, dx, inside).astype(Ex_w.dtype)
        Ex_w = Ex_w * jac
        Ey_w = Ey_w * jac
        Ez_w = Ez_w * jac

    # JAX path: keep the per-component scatter form so jax.jit / vmap
    # over the pipeline can trace through it unchanged.
    if is_jax_array(paths.positions):
        import jax.numpy as jnp

        def _scatter(vals):
            out = jnp.zeros(Ny * Nx, dtype=output_dtype)
            out = out.at[flat_idx].add(jnp.where(inside, vals, 0))
            return out.reshape(Ny, Nx)

        Ex_out, Ey_out, Ez_out = (_scatter(Ex_w), _scatter(Ey_w),
                                  _scatter(Ez_w))
        if return_ez:
            return Ex_out, Ey_out, Ez_out
        return Ex_out, Ey_out

    Ex_masked = xp.where(inside, Ex_w, 0)
    Ey_masked = xp.where(inside, Ey_w, 0)
    Ez_masked = xp.where(inside, Ez_w, 0)

    N_flat = Ny * Nx
    Ex_out_flat = xp.zeros(N_flat, dtype=output_dtype)
    Ey_out_flat = xp.zeros(N_flat, dtype=output_dtype)
    Ez_out_flat = xp.zeros(N_flat, dtype=output_dtype)
    if hasattr(xp, 'add') and hasattr(xp.add, 'at'):
        xp.add.at(Ex_out_flat, flat_idx, Ex_masked)
        xp.add.at(Ey_out_flat, flat_idx, Ey_masked)
        xp.add.at(Ez_out_flat, flat_idx, Ez_masked)
    else:
        # CuPy fallback: cupyx.scatter_add or NumPy round-trip.
        try:
            import cupyx
            cupyx.scatter_add(Ex_out_flat, flat_idx, Ex_masked)
            cupyx.scatter_add(Ey_out_flat, flat_idx, Ey_masked)
            cupyx.scatter_add(Ez_out_flat, flat_idx, Ez_masked)
        except (ImportError, AttributeError, TypeError, ValueError):
            # K23: route through ``to_numpy`` so a CuPy bundle does not
            # hit ``np.asarray`` on a device array.
            idx_h = to_numpy(flat_idx)
            ex_host = np.zeros(N_flat, dtype=output_dtype)
            ey_host = np.zeros(N_flat, dtype=output_dtype)
            ez_host = np.zeros(N_flat, dtype=output_dtype)
            np.add.at(ex_host, idx_h, to_numpy(Ex_masked))
            np.add.at(ey_host, idx_h, to_numpy(Ey_masked))
            np.add.at(ez_host, idx_h, to_numpy(Ez_masked))
            Ex_out_flat = xp.asarray(ex_host)
            Ey_out_flat = xp.asarray(ey_host)
            Ez_out_flat = xp.asarray(ez_host)
    Ex_out = Ex_out_flat.reshape(Ny, Nx)
    Ey_out = Ey_out_flat.reshape(Ny, Nx)
    Ez_out = Ez_out_flat.reshape(Ny, Nx)
    if return_ez:
        return Ex_out, Ey_out, Ez_out
    return Ex_out, Ey_out


def propagate_vector_hfpi_freespace_aperture(
    Ex_in: np.ndarray,
    Ey_in: np.ndarray,
    dx: float,
    *,
    z_to_aperture: float,
    aperture_radius: float,
    z_aperture_to_output: float,
    wavelength: float,
    n_paths: int,
    rng: Optional[Union[int, object]] = None,
    output_shape: Optional[Tuple[int, int]] = None,
    output_grid: Optional[Tuple[int, int]] = None,
    output_dx: Optional[float] = None,
    output_centre: Tuple[float, float] = (0.0, 0.0),
    aperture_shape: str = 'circular',
    aperture_centre: Tuple[float, float] = (0.0, 0.0),
    cone_half_angle: float = np.pi / 2 - 1e-6,
    on_undersampled: str = 'warn',
    normalisation: str = 'physical',
    vector_projection: bool = True,
    return_ez: bool = False,
):
    """End-to-end vectorial HFPI: vector source -> free-space hop ->
    aperture -> free-space hop -> vector output.

    Returns
    -------
    Ex_out, Ey_out : tuple of arrays (Ny, Nx) complex
        With ``return_ez=True``, ``(Ex_out, Ey_out, Ez_out)``.

    Parameters
    ----------
    cone_half_angle : float, default ~pi/2
        Half-angle of the emission / re-emission cone [rad].

        .. versionadded:: 5.46
            Audit K14/K23 -- the under-sampling guard's own recommended
            remedy was a ``TypeError`` on this entry point.
    on_undersampled : {'warn', 'silent', 'error'}, default 'warn'
        .. versionadded:: 5.46
            Audit K23 -- the vector path had no sampling guard at all.
    normalisation : {'physical', 'legacy'}, default 'physical'
        Audit K13; see
        :func:`~lumenairy.propagators.hfpi.accumulate_to_grid`.
    vector_projection : bool, default True
        Audit K17; see :func:`apply_vector_aperture_diffraction`.
    return_ez : bool, default False
        Also return the longitudinal component (K17).

    Notes
    -----
    ``output_shape=(Ny, Nx)`` is the v5.21.5 spelling of the output
    grid shape; the legacy ``output_grid`` kwarg keeps working but
    emits a ``DeprecationWarning`` (mirrors the scalar
    :func:`hfpi.propagate_hfpi_freespace_aperture` v5.2 rename).
    """
    # 4.13.2 (P1-NEW-A): spawn a distinct child RNG for the aperture
    # re-emission so source-plane init and aperture re-sample are
    # statistically independent.  Mirrors the scalar
    # :func:`hfpi.propagate_hfpi_freespace_aperture` 4.11.2 fix.
    # Pre-4.13.2 the same int ``rng`` was reused at both sites;
    # ``RandomState(rng=int)`` rebuilds default_rng(int) so init and
    # re-emission drew identical samples (perfectly correlated).
    rng_source = _spawn_rng(rng, 0)
    rng_aperture = _spawn_rng(rng, 1)
    paths = init_vector_paths_from_field(
        Ex_in, Ey_in, dx,
        n_paths=n_paths,
        wavelength=wavelength, rng=rng_source,
        cone_half_angle=cone_half_angle,
        z_input_plane=0.0,
        vector_projection=vector_projection,
    )
    paths = propagate_vector_to_plane(paths, z_target=z_to_aperture,
                                       wavelength=wavelength)
    paths = apply_vector_aperture_diffraction(
        paths, aperture_radius=aperture_radius,
        centre=aperture_centre, shape=aperture_shape,
        wavelength=wavelength, rng=rng_aperture,
        cone_half_angle=cone_half_angle,
        vector_projection=vector_projection,
    )
    paths = propagate_vector_to_plane(
        paths, z_target=z_to_aperture + z_aperture_to_output,
        wavelength=wavelength,
    )
    Ny, Nx = _resolve_output_shape(
        output_shape, output_grid,
        fn_name='propagate_vector_hfpi_freespace_aperture',
        default_shape=(Ex_in.shape[-2], Ex_in.shape[-1]))
    if output_dx is None:
        output_dx = dx
    return accumulate_vector_to_grid(
        paths, Ny=Ny, Nx=Nx, dx=output_dx, centre=output_centre,
        # VHFPI-1: promote to a COMPLEX accumulator (the v5.17 P2-32 scalar
        # fix, unmirrored here) -- a real-dtype (Ex_in, Ey_in) otherwise
        # allocates a real accumulator and np.add.at silently drops the
        # imaginary half of every path weight (~40% intensity loss).
        output_dtype=_complex_output_dtype(Ex_in.dtype),
        on_undersampled=on_undersampled,
        normalisation=normalisation,
        return_ez=return_ez,
    )


__all__ = [
    'VectorPathBundle',
    'init_vector_paths_from_field',
    'propagate_vector_to_plane',
    'apply_vector_aperture_diffraction',
    'accumulate_vector_to_grid',
    'propagate_vector_hfpi_freespace_aperture',
]
