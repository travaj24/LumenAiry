"""Image-plane reference-sphere wavefront error.

This module implements the textbook "chief-ray reference-sphere"
wavefront-error computation for a lens prescription, complementing
:func:`lumenairy.apply_real_lens_traced` which returns chief-relative
OPL evaluated on a flat plane at the LAST LENS SURFACE.

The two quantities differ by the geometric path from the lens-exit
ray state to the image-plane reference sphere centered on the chief
ray's image intersect.  This module performs the exact ray-sphere
intersection to convert.  See the cross-check methodology in
``OPDPy_Lumenairy_Crosscheck/CROSS_CHECK_METHODOLOGY.md`` (in the
companion validation repo) for derivations and accuracy benchmarks.

Public API
----------
* :class:`ImagePlaneWFE` -- dataclass result with per-ray pupil
  coords, OPD values, and aggregated PV / RMS / Strehl statistics.
* :func:`eval_image_plane_wfe` -- main entry point.
* :func:`remove_low_order_aberrations` -- subtract piston + tilt +
  defocus (optionally + 4th-order spherical) by least squares; used
  for cross-library comparison and for separating "design" from
  "fabrication" aberrations.

Convention
----------
* Positive RS-OPD at the marginal edge for an undercorrected singlet
  (matches rayoptics / Optiland / Zemax sign).
* Output units = waves of the input ``wavelength`` (so the result is
  dimensionless; multiply by ``wavelength`` to get metres).
* Chief is placed at OPD = 0 by re-zeroing the marginal values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..raytrace import (
    surfaces_from_prescription,
    trace,
    _make_bundle,
    find_paraxial_focus,
    first_order_data,
)


__all__ = [
    'ImagePlaneWFE',
    'eval_image_plane_wfe',
    'field_grid_wfe',
    'zemax_pupil_grid',
    'chebyshev_pupil_grid',
    'remove_low_order_aberrations',
]


@dataclass
class ImagePlaneWFE:
    """Per-ray image-plane reference-sphere wavefront error.

    Attributes
    ----------
    px, py : ndarray, shape (n_rays,)
        Normalised pupil coordinates in ``[-1, 1]``.
    opd_w : ndarray, shape (n_rays,)
        Reference-sphere OPD in WAVES of the input wavelength.
        Positive at the marginal edge for an undercorrected
        converging singlet (rayoptics / Optiland sign convention).
    alive : ndarray of bool, shape (n_rays,)
        ``True`` for rays that survived the trace.
    chief_idx : int
        Index of the chief ray (px ~ 0, py ~ 0) in the arrays.
    img_d_m : float
        Chief-ray image distance from the last lens vertex [m].
        For ``image_plane='best_rms'``/``'best_pv'`` this is the
        SHIFTED value, not the paraxial value.
    wavelength_m : float
        Wavelength used [m].
    aperture_m : float
        Entrance-pupil diameter used [m].
    image_plane : str
        ``'paraxial'`` / ``'best_rms'`` / ``'best_pv'`` -- which
        focus convention produced ``img_d_m`` (3.8.2+).
    sphere_tangent : str
        ``'vertex'`` / ``'exit_pupil'`` -- which point on the
        chief ray the reference sphere is tangent to (3.8.2+).
    r_sphere_m : float
        Signed reference-sphere radius actually used [m].  Equals
        ``img_d_m`` for ``sphere_tangent='vertex'``,
        ``img_d_m - xp_z`` for ``sphere_tangent='exit_pupil'``.
    img_d_m_paraxial : float
        The paraxial image distance (before any ``image_plane``
        shift), for diagnostic purposes.
    """
    px: np.ndarray
    py: np.ndarray
    opd_w: np.ndarray
    alive: np.ndarray
    chief_idx: int
    img_d_m: float
    wavelength_m: float
    aperture_m: float
    image_plane: str = 'paraxial'
    sphere_tangent: str = 'vertex'
    r_sphere_m: float = float('nan')
    img_d_m_paraxial: float = float('nan')

    @property
    def pv_waves(self) -> float:
        """Peak-to-valley OPD over alive rays, in waves."""
        v = self.opd_w[self.alive]
        v = v[np.isfinite(v)]
        return float(v.max() - v.min()) if v.size else float('nan')

    @property
    def rms_waves(self) -> float:
        """RMS OPD over alive rays, in waves.

        4.10: this is an UNWEIGHTED mean.  On a uniform-density pupil
        grid that's the correct numerical RMS, but on edge-clustered
        grids (Chebyshev / Gauss-Lobatto, exposed via
        ``chebyshev_pupil_grid`` for OPDPy cross-checks) the rim is
        over-counted and the Marechal-Strehl is biased.  For valid
        Strehl on a non-uniform grid, fit a Zernike basis and recover
        the rms-OPD from the Zernike coefficients
        (``ImagePlaneWFE.zernike_fit`` ... TBD), or supply the field
        on a uniform grid via ``zemax_pupil_grid``.
        """
        v = self.opd_w[self.alive]
        v = v[np.isfinite(v)]
        return float(np.sqrt(np.mean(v ** 2))) if v.size else float('nan')

    @property
    def strehl(self) -> float:
        """Marechal-approximation Strehl ratio ``exp(-(2*pi*RMS)^2)``.

        Valid only for small aberration (RMS << 1 wave).  For
        strongly aberrated systems use the diffraction PSF
        integral instead.
        """
        rms = self.rms_waves
        if not np.isfinite(rms):
            return float('nan')
        sigma = 2.0 * np.pi * rms
        return float(np.exp(-(sigma ** 2)))


def _ray_sphere_opd(opd_a_w, s2x, s2y, s2z, Ld, Md, Nd,
                      cz, R, wavelength_m, chief_idx, alive,
                      cx=0.0, cy=0.0):
    """Inner kernel: exact ray-sphere intersection for the reference
    sphere centered at ``(cx, cy, cz)`` with signed radius ``R``.

    Returns the per-ray reference-sphere OPD in waves, with
    rayoptics / Optiland sign convention and chief re-zeroed.

    For on-axis fields ``(cx, cy) = (0, 0)`` (default).  For off-axis
    fields the chief ray's transverse position at the image plane is
    passed via ``(cx, cy)``.
    """
    b = 2.0 * ((s2x - cx) * Ld + (s2y - cy) * Md + (s2z - cz) * Nd)
    c = ((s2x - cx) ** 2 + (s2y - cy) ** 2 + (s2z - cz) ** 2
         - R ** 2)
    disc = b ** 2 - 4.0 * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    # Smallest-|t| root: continuous through chief (t_chief=0 for
    # vertex-tangent; can be non-zero for exit-pupil-tangent or
    # diverging-system spheres).
    t = np.where(np.abs(t1) < np.abs(t2), t1, t2)
    rs_w = -(opd_a_w + t / wavelength_m)  # rayoptics sign
    if np.isfinite(rs_w[chief_idx]):
        rs_w = rs_w - rs_w[chief_idx]
    rs_w = np.where(alive, rs_w, np.nan)
    return rs_w


def eval_image_plane_wfe(
    prescription: dict,
    wavelength: float,
    field: Tuple[float, float] = (0.0, 0.0),
    n_pupil: int = 31,
    img_d_m: Optional[float] = None,
    image_plane: str = 'paraxial',
    sphere_tangent: str = 'vertex',
    field_max_m: Optional[float] = None,
    pupil_grid: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> ImagePlaneWFE:
    """Compute image-plane reference-sphere wavefront error.

    For each pupil-grid ray traced from the object plane through the
    lens, finds the exact intersection with the reference sphere
    centered on the chief ray's image-plane intersect, with radius
    determined by the ``sphere_tangent`` choice.  The chief is on
    the sphere by construction (zero OPD); marginal-ray OPDs
    capture the system's image-plane wavefront error.

    Parameters
    ----------
    prescription : dict
        Lumenairy lens-prescription dictionary (with
        ``object_distance``, ``aperture_diameter``, and a
        ``surfaces`` list -- see
        :func:`lumenairy.raytrace.surfaces_from_prescription`).
    wavelength : float
        Vacuum wavelength [m].
    field : tuple of float, default (0, 0)
        Normalised field coordinate ``(Hx, Hy)`` in ``[-1, 1]``.
        For off-axis fields ``(Hx, Hy) != (0, 0)`` you must also pass
        ``field_max_m`` (or set up the prescription with a
        ``field_max_m`` key) so the function knows the linear
        object-space scale corresponding to ``H = 1``.
    n_pupil : int, default 31
        Pupil samples per axis (a square ``n_pupil x n_pupil``
        Cartesian grid is generated and clipped to the unit disk).
    img_d_m : float, optional
        Override the paraxial image distance from the last lens
        surface [m].  Defaults to the value derived from the
        prescription's object distance + EFL + principal planes.
        Negative values are supported for diverging systems with
        virtual images.
    image_plane : {'paraxial', 'best_rms', 'best_pv'}, default 'paraxial'
        Which image-plane convention to use *(3.8.2+)*:

        * ``'paraxial'`` -- chief paraxial focus (Gauss imaging
          equation + principal planes).  Matches Zemax
          ``WavefrontMap`` default, rayoptics ``foc=0``, Optiland
          default.  Use this for cross-library validation.
        * ``'best_rms'`` -- shift the image plane to minimise the
          RMS of the WFE.  Closed-form: fit defocus to the
          paraxial-focus WFE, derive the shift, re-evaluate.
          This is the focus a lab tech finds by maximising
          intensity; published Strehl ratios assume it.
        * ``'best_pv'`` -- shift to minimise PV.  Uses a 1-D
          numerical search since closed-form doesn't exist for
          arbitrary aberration content.  Less common; useful for
          PV-defined tolerance specs.
    sphere_tangent : {'vertex', 'exit_pupil'}, default 'vertex'
        Where on the chief ray the reference sphere is tangent
        *(3.8.2+)*:

        * ``'vertex'`` -- tangent at the LAST LENS SURFACE vertex.
          Radius = ``img_d_m``.  Simplest convention; what
          ``conv_a_to_rs_opd`` and pre-3.8.2 versions of this
          function used.
        * ``'exit_pupil'`` -- tangent at the exit pupil.  Radius =
          ``img_d_m - xp_z`` where ``xp_z`` is the signed exit-
          pupil offset from the last surface (typically negative
          for stop-at-front singlets, putting XP inside the lens).
          Matches the convention rayoptics, Optiland, and Zemax use
          internally.
    field_max_m : float, optional
        Maximum half-field height in **object-space metres**, used
        to convert the normalised ``field`` coordinate to a physical
        source position.  Required when ``field != (0, 0)``.  If
        omitted and ``field == (0, 0)``, has no effect (on-axis
        case).  Falls back to ``prescription['field_max_m']`` if the
        prescription carries that key.
    pupil_grid : tuple ``(px, py)``, optional *(4.1+)*
        Custom pupil-grid coordinates as a pair of 1-D arrays of
        **normalised pupil coordinates** in ``[-1, 1]`` -- one ray
        is launched per ``(px[i], py[i])`` pair.  When given,
        ``n_pupil`` is ignored and the function bypasses its
        internal square-grid-clipped-to-disk generation.

        Use cases:

        * **Cross-library validation:** evaluate Lumenairy WFE at
          exactly the pupil sample points another library is using
          (rayoptics, Optiland, Zemax), eliminating the
          nearest-neighbour interpolation noise that otherwise
          dominates raw-diff floors at ~5-10% of WFE RMS.
        * **Chebyshev / Gauss-Lobatto quadrature** for high-order
          polynomial fits where the user wants nodes clustered at
          the pupil edge.
        * **Sparse / structured grids** for fast aberration screens
          (e.g. 8 rays on a ring for a quick spherical-vs-coma
          check).

        Coordinates outside the unit disk (``px**2 + py**2 > 1``)
        are silently dropped before the trace.

    Returns
    -------
    ImagePlaneWFE
        Per-ray pupil coordinates, OPD in waves, alive mask,
        aggregated PV / RMS / Strehl, and metadata recording
        which conventions produced this result.

    Raises
    ------
    ValueError
        For invalid or empty prescriptions, unrecognised
        ``image_plane`` / ``sphere_tangent`` choices, or non-zero
        ``field`` without a ``field_max_m`` value.

    Notes
    -----
    The conversion from lens-exit chief-relative OPL (Conv-A,
    what :func:`lumenairy.apply_real_lens_traced` consumes) to
    image-plane RS-OPD (Conv-B, the standard wavefront-error
    convention) is done by solving the ray-sphere quadratic

        t^2 + b*t + c = 0,
        b = 2*( s2x*L + s2y*M + (s2z - cz)*N ),
        c = s2x^2 + s2y^2 + (s2z - cz)^2 - R^2,

    for each ray, picking the root with smallest |t| so the
    formula remains continuous through ``t_chief = 0``.  The OPL
    adjustment is ``t * n_air`` (with ``n_air = 1``), added to
    the chief-relative lens-exit OPL, sign-flipped to match the
    rayoptics / Optiland convention, then re-zeroed on the chief
    value.

    For ``image_plane='best_rms'``, the closed-form shift comes
    from fitting ``c1 * r_norm^2`` (defocus, normalised pupil)
    to the paraxial-focus WFE:

        1/R'  =  1/R  +  2 * c1 [waves] * lambda / r_pupil^2

    where ``r_pupil`` is the entrance-pupil semi-aperture.  The
    sphere is then re-cast with radius ``R'`` and the ray-sphere
    intersection re-evaluated; the trace itself is not repeated.
    """
    if image_plane not in ('paraxial', 'best_rms', 'best_pv'):
        raise ValueError(
            f"eval_image_plane_wfe: image_plane must be one of "
            f"('paraxial','best_rms','best_pv'); got {image_plane!r}.")
    if sphere_tangent not in ('vertex', 'exit_pupil'):
        raise ValueError(
            f"eval_image_plane_wfe: sphere_tangent must be one of "
            f"('vertex','exit_pupil'); got {sphere_tangent!r}.")
    Hx = float(field[0]); Hy = float(field[1])
    if (Hx != 0.0 or Hy != 0.0):
        if field_max_m is None:
            field_max_m = prescription.get('field_max_m')
        if field_max_m is None or float(field_max_m) <= 0:
            raise ValueError(
                'eval_image_plane_wfe: non-zero field requires '
                'field_max_m (object-space half-field radius in m) '
                'either as a kwarg or in prescription["field_max_m"].')
        field_max_m = float(field_max_m)
    if not prescription.get('surfaces'):
        raise ValueError(
            'eval_image_plane_wfe: prescription has no "surfaces".')

    surfaces = surfaces_from_prescription(prescription)

    obj_d_m = float(prescription.get('object_distance', 0.0))
    if obj_d_m <= 0:
        raise ValueError(
            f'eval_image_plane_wfe: prescription object_distance must '
            f'be > 0 (got {obj_d_m:g} m).')

    # Need first-order data for both the paraxial image-distance
    # derivation AND the exit-pupil sphere tangent (3.8.2+).
    fod = first_order_data(surfaces, wavelength)

    if img_d_m is None:
        # Compute the paraxial image distance for the actual finite
        # object conjugate (not the BFL, which is the infinity-
        # conjugate special case).  Use the Gauss imaging equation
        # at the principal planes:
        #     1/v - 1/u = 1/f,  with u = -obj_d_m (object before lens)
        # then offset by the image-side principal-plane position.
        if not np.isfinite(fod.efl) or fod.efl == 0:
            raise ValueError(
                'eval_image_plane_wfe: lens has degenerate EFL; '
                'cannot derive paraxial image distance.  Pass '
                'img_d_m explicitly.')
        # Object distance from the OBJECT principal plane:
        #   u_pp = obj_d_m + pp_object_z  (pp_object_z is the H
        #   location relative to surf 0; obj_d_m is measured from
        #   surf 0 too, so we ADD when H is on +z of surf 0).
        u_pp = obj_d_m + fod.pp_object_z
        # Solve 1/v_pp = 1/efl - 1/u_pp for diverging or converging
        denom = (1.0 / fod.efl) - (1.0 / u_pp) if u_pp != 0 else np.inf
        if denom == 0:
            img_d_m = float('inf')  # object at front focal plane
        else:
            v_pp = 1.0 / denom
            # Image distance from LAST surface vertex:
            #   v_pp is measured from image PP (H'); H' is at
            #   pp_image_z relative to last surface.  So
            #   img_d_from_last = v_pp + pp_image_z.
            img_d_m = float(v_pp + fod.pp_image_z)

    img_d_m_paraxial = float(img_d_m)

    aperture_m = float(prescription.get('aperture_diameter', 0.0))
    if aperture_m <= 0:
        raise ValueError(
            f'eval_image_plane_wfe: aperture_diameter must be > 0 '
            f'(got {aperture_m:g} m).')
    semi = aperture_m / 2.0

    # Pupil-grid generation: either user-supplied (4.1+) or the
    # default square-clipped-to-disk used by all earlier versions.
    if pupil_grid is not None:
        px_in, py_in = pupil_grid
        px = np.asarray(px_in, dtype=np.float64).ravel()
        py = np.asarray(py_in, dtype=np.float64).ravel()
        if px.size != py.size:
            raise ValueError(
                f"pupil_grid: px and py must have the same length; "
                f"got {px.size} and {py.size}.")
        if px.size == 0:
            raise ValueError("pupil_grid: no sample points provided.")
        inside = (px ** 2 + py ** 2) <= 1.0 + 1e-12
        if not np.all(inside):
            px = px[inside]; py = py[inside]
        if px.size == 0:
            raise ValueError(
                "pupil_grid: all sample points fell outside the "
                "unit disk after clipping.")
    else:
        # Default: square grid clipped to the unit disk.
        p1 = np.linspace(-1.0, 1.0, n_pupil)
        PX, PY = np.meshgrid(p1, p1)
        px = PX.ravel(); py = PY.ravel()
        inside = (px ** 2 + py ** 2) <= 1.0
        px = px[inside]; py = py[inside]

    # Compute object-space source position.  On-axis (Hx=Hy=0)
    # this is (0, 0, -obj_d).  Off-axis the source is laterally
    # displaced by (Hx*field_max_m, Hy*field_max_m).
    if Hx == 0.0 and Hy == 0.0:
        src_x = 0.0
        src_y = 0.0
    else:
        src_x = Hx * field_max_m
        src_y = Hy * field_max_m

    # 4.10: aim rays at the entrance pupil (px*ep_radius, py*ep_radius,
    # ep_z) instead of the first surface vertex (px*semi, py*semi, 0).
    # For a stop-at-front system ep_z = 0 and ep_radius = semi, so the
    # two coincide.  For stop-in-the-middle the EP is the IMAGE of the
    # stop by the upstream sub-system, which sits at z = fod.ep_z away
    # from surface 0 with radius fod.ep_radius.  Pre-4.10 always aimed
    # at z=0 with the full aperture radius, so off-axis fields with a
    # mid-stop system landed at the wrong pupil position and reported
    # wrong WFE.
    ep_z = float(getattr(fod, 'ep_z', 0.0))
    ep_r = float(getattr(fod, 'ep_radius', semi))
    if not np.isfinite(ep_r) or ep_r <= 0:
        ep_r = semi
    aim_x = px * ep_r - src_x
    aim_y = py * ep_r - src_y
    aim_z = obj_d_m + ep_z  # ray length from -obj_d to ep_z
    norm_aim = np.sqrt(aim_x ** 2 + aim_y ** 2 + aim_z ** 2)
    L = aim_x / norm_aim
    M = aim_y / norm_aim

    bundle = _make_bundle(
        x=np.full_like(px, src_x), y=np.full_like(px, src_y),
        L=L, M=M, wavelength=wavelength)
    bundle.z = np.full(px.size, -obj_d_m)

    res = trace(bundle, surfaces, wavelength, output_filter='last')
    f = res.image_rays
    alive = np.asarray(f.alive, dtype=bool)
    opl = np.asarray(f.opd)

    # 4.10: identify chief = ALIVE ray closest to (0, 0) in pupil
    # coords.  Pre-4.10 could pick a dead vignetted on-axis ray (rare
    # but possible for systems where the chief is geometrically
    # blocked), which NaN-poisoned every downstream OPL calculation
    # via opl[chief] = NaN.  Fall back to the unconstrained nearest
    # if no rays survived (caller will see the all-NaN result anyway).
    pup_r2 = px ** 2 + py ** 2
    if alive.any():
        pup_r2_alive = np.where(alive, pup_r2, np.inf)
        chief = int(np.argmin(pup_r2_alive))
    else:
        chief = int(np.argmin(pup_r2))

    # Conv-A: chief-relative OPL at the last lens surface (in WAVES)
    opl_chief = opl[chief] if alive[chief] else float(
        np.median(opl[alive]) if alive.any() else 0.0)
    opd_a_w = (opl - opl_chief) / wavelength

    # Ray state at the lens exit -- shared input to every
    # reference-sphere evaluation below.
    s2x = np.asarray(f.x)
    s2y = np.asarray(f.y)
    s2z = np.asarray(f.z)
    Ld = np.asarray(f.L)
    Md = np.asarray(f.M)
    Nd = np.asarray(f.N)
    z_chief = s2z[chief]

    # 4.12.0 (B2-3): the reference-sphere radius is the chief ray's
    # PATH LENGTH from the tangent point (last-surface vertex chief
    # intersect or exit-pupil chief intersect) to the chief image,
    # not the axial distance.  For an off-axis chief with direction
    # cosines ``(L, M, N)`` and ``N = N_chief < 1`` the path length is
    # ``axial_distance / N_chief`` -- the same factor used in
    # ``_chief_image_xy`` to land the chief at the image plane.
    #
    # Pre-4.12.0 only the chief-image landing got the 1/N_chief factor
    # (the v4.11.2 fix); the sphere radius was left at the axial
    # ``img_d_m``, so for off-axis fields the sphere no longer passed
    # through the chief and the resulting quadratic shape error was
    # absorbed as phantom defocus by ``best_rms``.  On-axis (N=1) this
    # is a no-op.
    if alive[chief]:
        _N_chief = float(Nd[chief])
        if abs(_N_chief) < 1e-12:
            N_chief_for_R = 1.0
        else:
            N_chief_for_R = _N_chief
    else:
        N_chief_for_R = 1.0

    # Compute the reference-sphere radius from img_d_m + the
    # tangent-point choice.  Sphere centre is always at the chief
    # image intersect (cx, cy, z_chief + img_d_m).
    def _radius_for(d):
        """Signed sphere radius for chief image distance ``d``.

        Returns the chief ray's path length from the tangent point
        (last-surface vertex for ``vertex``, exit-pupil chief
        intersect for ``exit_pupil``) to the chief image at axial
        distance ``d`` past the last surface.  The ``1/N_chief``
        factor maps axial distance to ray-arc length and is
        consistent with ``_chief_image_xy``'s ``t_advance = d /
        N_chief`` -- without it the sphere doesn't pass through the
        chief for off-axis fields (B2-3 fix, 4.12.0+).
        """
        if sphere_tangent == 'vertex':
            return d / N_chief_for_R
        # exit_pupil: tangent at chief intersection with XP plane.
        # xp_z is the SIGNED offset from last vertex (negative when
        # XP is inside the lens).  Axial separation (image - XP) is
        # (d - xp_z); divide by N_chief for ray-arc length.
        return (d - fod.xp_z) / N_chief_for_R

    R = _radius_for(img_d_m)
    cz = z_chief + img_d_m

    def _chief_image_xy(d):
        """Transverse position of the chief ray at axial distance
        ``d`` past the last surface, by free-space propagation.

        Returns ``(nan, nan)`` for a grazing-incidence chief
        (``|N_chief| < 1e-12``) -- such rays effectively never
        reach the image plane.  Off-axis WFE for those cases is
        meaningless, so the caller will get NaN-propagated metrics
        and a visible failure rather than a silently-wrong result
        from a fallback unit vector.
        """
        if not alive[chief]:
            return float('nan'), float('nan')
        N_chief = float(Nd[chief])
        if abs(N_chief) < 1e-12:
            return float('nan'), float('nan')
        t_advance = d / N_chief  # geometric path length
        cx = float(s2x[chief] + Ld[chief] * t_advance)
        cy = float(s2y[chief] + Md[chief] * t_advance)
        return cx, cy

    cx, cy = _chief_image_xy(img_d_m)
    rs_w = _ray_sphere_opd(opd_a_w, s2x, s2y, s2z, Ld, Md, Nd,
                              cz, R, wavelength, chief, alive,
                              cx=cx, cy=cy)

    # 3.8.2: image-plane choice.  For 'best_rms' / 'best_pv' we
    # fit / search the longitudinal shift that minimises the
    # corresponding wavefront metric and re-evaluate.
    if image_plane != 'paraxial':
        if image_plane == 'best_rms':
            # Closed form: fit c0 + c1 * (px^2 + py^2) to the
            # paraxial WFE and convert c1 to a sphere-radius
            # shift via R' = 1 / (1/R + 2*c1*lambda / r_pup^2).
            r2 = px ** 2 + py ** 2
            mask = np.isfinite(rs_w)
            if mask.any():
                A = np.column_stack([np.ones(mask.sum()), r2[mask]])
                coefs, *_ = np.linalg.lstsq(A, rs_w[mask], rcond=None)
                c1 = float(coefs[1])  # waves of defocus
                r_pup = semi  # entrance-pupil semi-aperture [m]
                # 4.10: closed-form best-RMS uses the SPHERE radius
                # (which depends on sphere_tangent), not img_d_m
                # directly.  For 'exit_pupil' the sphere radius is
                # img_d_m - fod.xp_z; pre-4.10 always used 1/img_d_m
                # which is wrong for that branch.
                R_old = _radius_for(img_d_m)
                inv_R_new = (1.0 / R_old
                              + 2.0 * c1 * wavelength / r_pup ** 2)
                if abs(inv_R_new) > 1e-30:
                    R_new = 1.0 / inv_R_new
                    # Back-solve for the new chief image distance.
                    # 4.12.0 (B2-3): invert the `1/N_chief` factor
                    # applied in ``_radius_for`` so the new axial
                    # ``img_d_m`` stays self-consistent for off-axis
                    # fields.  On-axis (N_chief = 1) this is a no-op.
                    if sphere_tangent == 'exit_pupil':
                        img_d_m = (R_new * N_chief_for_R
                                   + float(getattr(fod, 'xp_z', 0.0)))
                    else:
                        img_d_m = R_new * N_chief_for_R
        elif image_plane == 'best_pv':
            # 1-D numerical search over the longitudinal shift dz
            # that minimises PV.  Uses scipy.optimize when available;
            # falls back to a coarse + bisect search.
            def _pv_at(d_test):
                R_t = _radius_for(d_test)
                cz_t = z_chief + d_test
                cx_t, cy_t = _chief_image_xy(d_test)
                w = _ray_sphere_opd(opd_a_w, s2x, s2y, s2z,
                                       Ld, Md, Nd, cz_t, R_t,
                                       wavelength, chief, alive,
                                       cx=cx_t, cy=cy_t)
                w = w[np.isfinite(w)]
                return float(w.max() - w.min()) if w.size else np.inf
            # Search range: paraxial +/- 10 * (Marechal wavelength
            # depth-of-focus) = +/- 10 * 4 * f/#^2 * lambda
            r_pup = semi
            fnum = abs(img_d_m) / (2.0 * r_pup) if r_pup > 0 else 8.0
            dof = 4.0 * fnum ** 2 * wavelength
            dz_max = 10.0 * dof
            try:
                from scipy.optimize import minimize_scalar as _mins
                res_opt = _mins(
                    lambda dz: _pv_at(img_d_m + dz),
                    bracket=(-dz_max, 0.0, dz_max),
                    method='brent',
                    options={'xtol': 1e-9})
                img_d_m = float(img_d_m + res_opt.x)
            except Exception:
                # 21-point coarse scan as fallback
                shifts = np.linspace(-dz_max, dz_max, 21)
                pvs = [_pv_at(img_d_m + dz) for dz in shifts]
                img_d_m = float(img_d_m + shifts[int(np.argmin(pvs))])
        # Re-evaluate with the shifted image distance
        R = _radius_for(img_d_m)
        cz = z_chief + img_d_m
        cx, cy = _chief_image_xy(img_d_m)
        rs_w = _ray_sphere_opd(opd_a_w, s2x, s2y, s2z, Ld, Md, Nd,
                                  cz, R, wavelength, chief, alive,
                                  cx=cx, cy=cy)

    return ImagePlaneWFE(
        px=px, py=py, opd_w=rs_w, alive=alive,
        chief_idx=chief, img_d_m=img_d_m,
        wavelength_m=wavelength, aperture_m=aperture_m,
        image_plane=image_plane,
        sphere_tangent=sphere_tangent,
        r_sphere_m=float(R),
        img_d_m_paraxial=img_d_m_paraxial,
    )


def zemax_pupil_grid(N: int = 512, clip_to_disk: bool = True
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Return the deterministic pupil-sample grid that Zemax
    OpticStudio's ``WavefrontMap`` analysis uses.

    Zemax `WavefrontMap` samples ``N x N`` rays on a Cartesian grid
    in normalised pupil coordinates:

    .. math::
        p_i = \\frac{i - (N - 1)/2}{(N - 1)/2}, \\quad i = 0, 1, \\ldots, N-1

    Spacing is ``2 / (N - 1)`` (both endpoints included).  ``N``
    is a power of two in Zemax (32, 64, 128, 256, 512, 1024, 2048);
    this function does not enforce that but will accept any
    positive integer for flexibility.  Points outside the unit
    disk are dropped by default (Zemax fills those with zero in
    the raw output, which is generally not what you want).

    Pair with :func:`eval_image_plane_wfe`'s ``pupil_grid`` kwarg
    to get a true per-ray Lumenairy-vs-Zemax comparison with zero
    KDTree-NN interpolation noise.

    Parameters
    ----------
    N : int, default 512
        Pupil samples per axis.  Use 32 / 64 / 128 / 256 / 512 /
        1024 to match Zemax's ``Sampling`` settings exactly.
    clip_to_disk : bool, default True
        Drop points outside the unit disk (``px**2 + py**2 > 1``).
        Set ``False`` to keep the full square grid (matches the
        raw shape Zemax returns before its mask is applied).

    Returns
    -------
    px, py : ndarrays of shape (n_rays,)
        Flat 1-D arrays of normalised pupil coordinates.  Length
        is ``N ** 2`` if ``clip_to_disk=False``, otherwise
        approximately ``π/4 · N ** 2``.

    Examples
    --------
    >>> import lumenairy as la
    >>> px, py = la.zemax_pupil_grid(N=512)
    >>> wfe = la.eval_image_plane_wfe(
    ...     prescription, wavelength=587.56e-9,
    ...     pupil_grid=(px, py))
    >>> # wfe.opd_w[i] is directly comparable to Zemax's
    >>> # WavefrontMap[i] at sampling 512x512, with zero
    >>> # interpolation noise.

    See also
    --------
    chebyshev_pupil_grid : edge-clustered (OPDPy-style) preset.
    eval_image_plane_wfe : consumes the ``(px, py)`` output via
        the ``pupil_grid`` kwarg.
    """
    if N <= 0:
        raise ValueError(f"zemax_pupil_grid: N must be > 0; got {N}.")
    ix = np.arange(N) - (N - 1) / 2.0
    norm = (N - 1) / 2.0
    p = ix / norm
    PX, PY = np.meshgrid(p, p)
    px = PX.ravel()
    py = PY.ravel()
    if clip_to_disk:
        inside = (px ** 2 + py ** 2) <= 1.0 + 1e-9
        px = px[inside]
        py = py[inside]
    return px, py


def chebyshev_pupil_grid(N: int = 31, clip_to_disk: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Return a Chebyshev-Gauss-Lobatto pupil-sample grid.

    Nodes cluster at the rim (``|p| -> 1``) where the wavefront
    error gradient is largest for typical aberrations (spherical,
    coma, …).  This is the grid OPDPy's ``OPDSystem.sample()``
    uses by default; passing it to
    :func:`eval_image_plane_wfe`'s ``pupil_grid`` kwarg matches
    the OPDPy-side comparison in the
    ``OPDPy_Lumenairy_Crosscheck`` repo exactly.

    Nodes are ``p_i = cos(pi * i / (N - 1))`` for
    ``i = 0, 1, ..., N - 1``.  ``N`` odd places one node exactly
    at the centre; ``N`` even straddles it.

    Parameters
    ----------
    N : int, default 31
        Number of Chebyshev nodes per axis.  Odd values are
        recommended so a centre node lands at ``p = 0``.
    clip_to_disk : bool, default True
        Drop points outside the unit disk after the tensor
        product.  Set ``False`` to keep the full square grid.

    Returns
    -------
    px, py : ndarrays of shape (n_rays,)

    See also
    --------
    zemax_pupil_grid : uniform-square preset.
    eval_image_plane_wfe : consumes the ``(px, py)`` output.
    """
    if N <= 0:
        raise ValueError(f"chebyshev_pupil_grid: N must be > 0; got {N}.")
    i = np.arange(N)
    p = np.cos(np.pi * i / max(N - 1, 1))
    PX, PY = np.meshgrid(p, p)
    px = PX.ravel()
    py = PY.ravel()
    if clip_to_disk:
        inside = (px ** 2 + py ** 2) <= 1.0 + 1e-9
        px = px[inside]
        py = py[inside]
    return px, py


def field_grid_wfe(
    prescription: dict,
    wavelength: float,
    field_max_m: float,
    n_field: int = 5,
    n_pupil: int = 21,
    image_plane: str = 'paraxial',
    sphere_tangent: str = 'vertex',
    img_d_m: Optional[float] = None,
) -> dict:
    """Evaluate image-plane WFE on a Cartesian grid of field points.

    Builds an ``n_field x n_field`` grid of normalised field
    coordinates ``Hx, Hy`` in ``[-1, 1]`` and runs
    :func:`eval_image_plane_wfe` at each.  Useful for the standard
    "how do PV/RMS/Strehl vary across the field?" plot.

    Parameters
    ----------
    prescription : dict
        Lumenairy lens prescription.
    wavelength : float
        Vacuum wavelength [m].
    field_max_m : float
        Object-space half-field height [m] corresponding to
        ``|H| = 1``.  Used by :func:`eval_image_plane_wfe`.
    n_field : int, default 5
        Field samples per axis.  Total points = ``n_field**2``;
        symmetric around the chief axis.
    n_pupil : int, default 21
        Pupil samples per axis at each field point.
    image_plane, sphere_tangent, img_d_m : as in
        :func:`eval_image_plane_wfe`.

    Returns
    -------
    result : dict with keys
        * ``'Hx'`` / ``'Hy'`` -- ``(n_field, n_field)`` arrays of
          normalised field coords.
        * ``'pv_waves'`` / ``'rms_waves'`` / ``'strehl'`` --
          ``(n_field, n_field)`` arrays of WFE statistics, NaN at
          fields where the trace failed.
        * ``'img_d_m'`` -- per-field image distance.
        * ``'wfe_per_field'`` -- list of per-field
          :class:`ImagePlaneWFE` objects (flattened in row-major
          ``(Hy, Hx)`` order).
        * ``'wavelength_m'``, ``'field_max_m'`` -- echo of inputs.

    Examples
    --------
    >>> import lumenairy as la
    >>> p = la.thorlabs_lens('AC254-100-C')
    >>> p['object_distance'] = 0.5
    >>> grid = la.field_grid_wfe(p, wavelength=1.31e-6,
    ...                            field_max_m=2e-3, n_field=5)
    >>> rms_corner = grid['rms_waves'][0, 0]   # full off-axis
    >>> rms_axis   = grid['rms_waves'][2, 2]   # on-axis (chief)

    Notes
    -----
    This is the workflow the existing on-axis `eval_image_plane_wfe`
    blocked before LumenAiry 4.0.  For per-field PSF / MTF, take
    each :class:`ImagePlaneWFE` in ``wfe_per_field``, reconstruct
    a pupil-phase map via the Zernike fit (``zernike_decompose``
    followed by ``apply_zernike_aberration``), and pass through
    :func:`compute_psf`.
    """
    h = np.linspace(-1.0, 1.0, int(n_field))
    Hx, Hy = np.meshgrid(h, h)
    pv = np.full(Hx.shape, np.nan, dtype=float)
    rms = np.full(Hx.shape, np.nan, dtype=float)
    strehl = np.full(Hx.shape, np.nan, dtype=float)
    img_d = np.full(Hx.shape, np.nan, dtype=float)
    wfes: list = []

    for iy in range(Hx.shape[0]):
        for ix in range(Hx.shape[1]):
            field = (float(Hx[iy, ix]), float(Hy[iy, ix]))
            try:
                wfe = eval_image_plane_wfe(
                    prescription, wavelength, field=field,
                    n_pupil=n_pupil, img_d_m=img_d_m,
                    image_plane=image_plane,
                    sphere_tangent=sphere_tangent,
                    field_max_m=field_max_m,
                )
                pv[iy, ix] = wfe.pv_waves
                rms[iy, ix] = wfe.rms_waves
                strehl[iy, ix] = wfe.strehl
                img_d[iy, ix] = wfe.img_d_m
            except Exception:
                wfe = None
            wfes.append(wfe)

    return {
        'Hx': Hx,
        'Hy': Hy,
        'pv_waves': pv,
        'rms_waves': rms,
        'strehl': strehl,
        'img_d_m': img_d,
        'wfe_per_field': wfes,
        'wavelength_m': float(wavelength),
        'field_max_m': float(field_max_m),
    }


def remove_low_order_aberrations(
    opd_w: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    include_r4: bool = True,
) -> np.ndarray:
    """Subtract best-fit low-order aberration content from an OPD field.

    By default fits

        W(px, py) = c0 + c1*r^2 + c2*px + c3*py + c4*r^4,
        r^2 = px^2 + py^2,

    by least squares and returns the residual.  Removes piston
    (c0), x/y tilt (c2, c3), defocus (c1), and 4th-order spherical
    aberration (c4) -- all the low-order shapes that the
    reference-sphere convention can swap freely between libraries.

    The remaining residual is the genuinely-higher-order
    aberration content (6th-order spherical, residual coma,
    astigmatism) where independent ray-trace implementations'
    actual numerical algorithms diverge -- the realistic
    apples-to-apples cross-library comparison metric.

    Parameters
    ----------
    opd_w : ndarray
        OPD values (any units; the fit is linear).
    px, py : ndarray
        Pupil coordinates, normalised to ``[-1, 1]``.  Same shape
        as ``opd_w``.
    include_r4 : bool, default True
        If True, include the ``r^4`` term in the fit (recommended
        for systems with non-negligible 4th-order spherical, e.g.
        fast singlets).  If False, fit only piston + tilt + defocus
        ("classic" low-order removal).

    Returns
    -------
    residual : ndarray
        Same shape as ``opd_w``.  NaN values are preserved.

    Examples
    --------
    >>> wfe = eval_image_plane_wfe(prescription, wavelength=587e-9)
    >>> resid = remove_low_order_aberrations(wfe.opd_w, wfe.px, wfe.py)
    >>> # resid is the higher-order aberration content only
    """
    opd_w = np.asarray(opd_w, dtype=float)
    px = np.asarray(px, dtype=float)
    py = np.asarray(py, dtype=float)
    mask = np.isfinite(opd_w) & np.isfinite(px) & np.isfinite(py)
    if not mask.any():
        return opd_w.copy()
    r2 = px ** 2 + py ** 2
    cols = [np.ones(mask.sum()), r2[mask], px[mask], py[mask]]
    if include_r4:
        cols.append(r2[mask] ** 2)
    A = np.column_stack(cols)
    coefs, *_ = np.linalg.lstsq(A, opd_w[mask], rcond=None)
    cols_all = [np.ones_like(opd_w), r2, px, py]
    if include_r4:
        cols_all.append(r2 ** 2)
    A_all = np.column_stack(cols_all)
    fit_all = A_all @ coefs
    return opd_w - fit_all
