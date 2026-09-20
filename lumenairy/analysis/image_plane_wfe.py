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

# Version history for this module: ``docs/history/lumenairy.analysis.image_plane_wfe.md``.

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..raytrace import (
    _make_bundle,
    first_order_data,
    surfaces_from_prescription,
    trace,
)
# R2's entrance-eikonal helper.  Imported from its home module because
# ``raytrace/__init__`` does not re-export it (unlike
# ``exit_vertex_transfer``); no local copy is kept here.
from ..raytrace.trace import _way_back_kwargs, seed_entrance_eikonal

# W4c: terminal-index factors from their SINGLE SOURCE in
# raytrace.seidel -- re-deriving them here is the drift R-1 and S11-1
# were caused by (see the module note in raytrace/seidel.py).
from ..raytrace.seidel import (
    _image_space_index,
    _mirror_parity_sign,
    _object_space_index,
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
        Measured as the axial offset from the chief ray's last-surface
        intersection; off-axis this differs from the vertex by the
        last-surface sag at the chief height.
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
        if v.size == 0:
            return float('nan')
        # v5.4.6 (audit F-11): remove piston (the mean) before the RMS.
        # A constant piston does not aberrate the wavefront and must not
        # enter the Marechal-Strehl RMS; including it biases the RMS high
        # and the Strehl low.  (This is the RMS about the mean = std.)
        return float(np.sqrt(np.mean((v - v.mean()) ** 2)))

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


def _warn_object_distance_precision(
    surfaces, obj_d_m: float, wavelength: float, fod,
) -> None:
    """Warn when a finite ``object_distance`` has lost the surface sag.

    Each surface intersection solves ``t^2 + b t + c = 0`` with
    ``b ~ 2 * obj_d_m`` and ``c = |P - C|^2 - R^2``.  The two roots are
    separated by ``sqrt(b^2 - 4c) ~ 2 |R|``, so the float64 rounding of
    ``b^2`` (``eps * b^2``) propagates into the root as

        ``dt ~ eps * (2 obj_d)^2 / (2 * 2 |R|) = eps * obj_d^2 / |R|``.

    Measured on the audit fixture (biconvex N-BK7, ``|R| = 50`` mm,
    587.6 nm) the marginal ray's surface-0 intersection error was
    ``0.024 / 5.3 / 590 um`` at ``obj_d = 1e4 / 1e5 / 1e6`` m against a
    bound of ``0.44 / 44 / 4400 um`` -- the estimator tracks the measured
    onset and over-states it by ~8x, so gating it at a tenth of a wave
    fires just as the reported PV starts to move (3.467 waves at 1e3 m,
    3.691 at 1e4 m, 47.3 at 1e5 m) and stays silent below.
    """
    radii = []
    for s in surfaces:
        r = float(getattr(s, 'radius', np.inf) or np.inf)
        if np.isfinite(r) and r != 0.0:
            radii.append(abs(r))
    if not radii:
        return                      # all-flat system: no quadratic solve
    r_min = min(radii)
    dt = np.finfo(np.float64).eps * obj_d_m ** 2 / r_min
    if dt <= 0.1 * wavelength:
        return
    efl = float(getattr(fod, 'efl', np.nan))
    import warnings as _w
    _w.warn(
        f'eval_image_plane_wfe: object_distance = {obj_d_m:.3g} m is far '
        f'enough that the object-side ray-surface intersection loses the '
        f'surface sag to float64 cancellation -- estimated intersection '
        f'error {dt*1e6:.3g} um (tightest surface radius {r_min*1e3:.3g} '
        f'mm, EFL {efl*1e3:.3g} mm), against a wavefront resolution of '
        f'{wavelength*1e6:.3g} um.  The reported PV / RMS are degraded '
        f'and the chief ray no longer lands on the axis.  For a distant '
        f'or infinite object set '
        f'prescription["object_distance"] = float("inf"), which launches '
        f'a collimated bundle at the entrance pupil and is exact.',
        RuntimeWarning, stacklevel=3)


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
    field_max_rad: Optional[float] = None,
    *,
    renormalize: Optional[str] = None,
    sphere_normal: Optional[str] = None,
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

        ``object_distance`` is the distance from surface 0 to the object
        [m], ``> 0``.  Set it to ``float('inf')`` (or ``None``) for an
        INFINITE CONJUGATE: the bundle is then launched collimated a few
        aperture-widths before surface 0, on a plane wavefront normal to
        the field direction, instead of from a point at a large finite
        distance.  The finite launch is not a usable stand-in for
        infinity: the object-side ray-sphere solve carries
        ``|P - C|^2 ~ object_distance^2`` against ``R^2``, which cancels
        catastrophically in float64 (see the ``object_distance`` note
        under Raises).
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
          Radius = ``img_d_m``.  Simplest convention; also what
          ``conv_a_to_rs_opd`` uses.
        * ``'exit_pupil'`` -- tangent at the exit pupil.  Radius =
          ``img_d_m - xp_z`` where ``xp_z`` is the signed exit-
          pupil offset from the last surface (typically negative
          for stop-at-front singlets, putting XP inside the lens).
          Matches the convention rayoptics, Optiland, and Zemax use
          internally.
    field_max_m : float, optional
        Maximum half-field height in **object-space metres**, used
        to convert the normalised ``field`` coordinate to a physical
        source position.  Required when ``field != (0, 0)`` at a FINITE
        object distance.  If omitted and ``field == (0, 0)``, has no
        effect (on-axis case).  Falls back to
        ``prescription['field_max_m']`` if the prescription carries that
        key.
    field_max_rad : float, optional
        Maximum half-field **angle in radians**, the infinite-conjugate
        counterpart of ``field_max_m``: at ``object_distance = inf`` a
        field point is a DIRECTION, not a height, so ``field`` is scaled
        by this instead.  Required when ``field != (0, 0)`` and the object
        is at infinity.  Falls back to
        ``prescription['field_max_rad']``.

        **Sign.**  ``field`` scales an OBJECT POSITION at both conjugates,
        so ``field = (0, +1)`` means the same physical field point whether
        the object is 1 m away or at infinity: an object ABOVE the axis,
        at ``y = Hy * field_max_m`` in the finite case and in the
        direction ``Hy * field_max_rad`` above the axis in the infinite
        one.  In both the chief ray therefore travels towards ``-y``, and
        the two agree in the limit -- measured on a biconvex N-BK7 singlet
        at 587.6 nm, ``object_distance = inf`` at ``field = (0, +1)``
        matches ``object_distance = 1e3 m`` at the same ``field = (0, +1)``
        with ``field_max_m = 1e3 * tan(theta)`` to 0.0025 / 0.0033 /
        0.0033 waves (<= 0.07 % of the map's span) at
        theta = 0.5 / 1 / 2 deg, where the opposite sign differs by
        0.854 / 1.706 / 3.404 waves (23 / 40 / 59 % of span).

        Note the relation to ``raytrace.ray_fan`` and
        :func:`~lumenairy.analysis.field.field_aberration_sweep`, whose
        ``field_angle`` is a ray DIRECTION angle (``M = +sin``): this
        ``field`` is its NEGATIVE, i.e.
        ``field_angle = -Hy * field_max_rad``.
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
    renormalize : ``None`` (default) | ``'exit'`` | ``'surface'``
        Forwarded verbatim to the internal :func:`trace` call -- WP-C2's
        way back, one keyword per flipped default.  ``None`` means
        "whatever the library's default is", so an unkeyworded call is
        unchanged and no call site pins today's default.
    sphere_normal : ``None`` (default) | ``'analytic'`` | ``'generic'``
        Forwarded verbatim to the same call.  Pass
        ``renormalize='surface'`` and ``sphere_normal='generic'`` together
        for the arithmetic this entry point produced before WP-C2 moved
        the two tracer defaults -- byte-identical, pinned archive to
        archive.

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
    Hx = float(field[0])
    Hy = float(field[1])
    if not prescription.get('surfaces'):
        raise ValueError(
            'eval_image_plane_wfe: prescription has no "surfaces".')

    surfaces = surfaces_from_prescription(prescription)

    _obj_raw = prescription.get('object_distance', 0.0)
    obj_d_m = float('inf') if _obj_raw is None else float(_obj_raw)
    if np.isnan(obj_d_m) or obj_d_m <= 0:
        raise ValueError(
            f"eval_image_plane_wfe: prescription object_distance must "
            f"be > 0 (got {obj_d_m:g} m), or float('inf') / None for an "
            f"infinite conjugate.")
    infinite_object = not np.isfinite(obj_d_m)

    if (Hx != 0.0 or Hy != 0.0):
        if infinite_object:
            if field_max_rad is None:
                field_max_rad = prescription.get('field_max_rad')
            if field_max_rad is None or float(field_max_rad) <= 0:
                raise ValueError(
                    'eval_image_plane_wfe: non-zero field at an infinite '
                    'conjugate requires field_max_rad (object-space '
                    'half-field ANGLE in radians) either as a kwarg or in '
                    'prescription["field_max_rad"] -- a field point at '
                    'infinity is a direction, not a height, so '
                    'field_max_m does not define it.')
            field_max_rad = float(field_max_rad)
        else:
            if field_max_m is None:
                field_max_m = prescription.get('field_max_m')
            if field_max_m is None or float(field_max_m) <= 0:
                raise ValueError(
                    'eval_image_plane_wfe: non-zero field requires '
                    'field_max_m (object-space half-field radius in m) '
                    'either as a kwarg or in prescription["field_max_m"].')
            field_max_m = float(field_max_m)

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
        # Solve the Gauss conjugate equation at the principal planes.
        #
        # W4c (finding 2).  The textbook air form ``1/u + 1/v = 1/f``
        # only holds for air-to-air conjugates.  ``fod.efl`` is the
        # REDUCED focal length ``1/Phi`` (W4b made that explicit) while
        # ``u_pp`` and ``v_pp`` are GEOMETRIC distances, so the general
        # form must carry the terminal indices:
        #
        #     n_obj / u_pp  +  n_img / v_pp  =  Phi  =  1 / efl
        #
        # Measured against an exact real-ray oracle (a ray from the axial
        # object point, its axis crossing read past the last surface), the
        # air form gets an N-BK7 IMAGE space wrong by -41.1 % --
        # ``img_d_m`` = +41.569590 mm against +70.557940 mm exact, the
        # misplaced reference sphere then reporting 114.8 waves PV -- and
        # an N-BK7 OBJECT space by -4.3 % (+34.023215 mm against
        # +35.549003 mm, 292.8 waves PV).  The index-threaded form below
        # matches that oracle to <= 5.3e-12 on both, and to 1.8e-12 on the
        # air control.  The two errors are NOT a common factor -- n_obj
        # and n_img enter differently -- which is why both must be
        # threaded.
        #
        # Air conjugates are bit-identical: both indices are exactly 1.0,
        # so ``n_obj / u_pp`` IS ``1.0 / u_pp`` and ``n_img / denom`` IS
        # ``1.0 / denom``.
        n_obj_g = _object_space_index(surfaces, wavelength)
        n_img_g = _image_space_index(surfaces, wavelength)
        denom = (1.0 / fod.efl) - (n_obj_g / u_pp) if u_pp != 0 else np.inf
        if denom == 0:
            img_d_m = float('inf')  # object at front focal plane
        else:
            v_pp = n_img_g / denom
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
            px = px[inside]
            py = py[inside]
        if px.size == 0:
            raise ValueError(
                "pupil_grid: all sample points fell outside the "
                "unit disk after clipping.")
    else:
        # Default: square grid clipped to the unit disk.
        p1 = np.linspace(-1.0, 1.0, n_pupil)
        PX, PY = np.meshgrid(p1, p1)
        px = PX.ravel()
        py = PY.ravel()
        inside = (px ** 2 + py ** 2) <= 1.0
        px = px[inside]
        py = py[inside]

    # Compute object-space source position.  On-axis (Hx=Hy=0)
    # this is (0, 0, -obj_d).  Off-axis the source is laterally
    # displaced by (Hx*field_max_m, Hy*field_max_m).
    if Hx == 0.0 and Hy == 0.0 or infinite_object:
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
    # from surface 0 with radius fod.ep_radius.  Aiming at z=0 with the
    # full aperture radius instead lands off-axis fields of a mid-stop
    # system at the wrong pupil position and reports wrong WFE.
    ep_z = float(getattr(fod, 'ep_z', 0.0))
    ep_r = float(getattr(fod, 'ep_radius', semi))
    if not np.isfinite(ep_r) or ep_r <= 0:
        ep_r = semi

    if infinite_object:
        # Collimated launch on a plane wavefront normal to the field
        # direction.  Every ray gets the SAME direction and its own foot
        # on that wavefront, so the launch carries the object-side OPL
        # difference between pupil points exactly (a tilted plane wave is
        # NOT in phase across a plane of constant z) while the piston --
        # the only part that depends on how far back the plane sits --
        # cancels in the chief-relative OPD below.  All path lengths are
        # O(aperture), so none of the object-side ray-sphere cancellation
        # of a large finite object_distance arises.
        # Sign: ``field`` scales an OBJECT POSITION at both conjugates, so
        # the direction cosines are NEGATIVE sines.  ``field = (0, +1)``
        # is an object above the axis in both branches -- at a finite
        # distance a point at ``y = +Hy*field_max_m``, at infinity a
        # direction ``+Hy*field_max_rad`` above the axis -- and in both
        # the chief therefore travels towards ``-y``.  Without the sign
        # the same ``field`` would name opposite field points at the two
        # conjugates (measured: up to 59 % of the map's span at 2 deg),
        # and PV / RMS are sign-blind, so nothing downstream would notice.
        # Relation to ``raytrace.ray_fan``, which takes a ray DIRECTION
        # angle (``M = +sin(field_angle)``): this ``field`` is its
        # negative, ``field_angle = -Hy*field_max_rad``.
        th_x = Hx * field_max_rad if Hx != 0.0 else 0.0
        th_y = Hy * field_max_rad if Hy != 0.0 else 0.0
        # Gate on the ANGLE, not on the direction cosine: ``sin`` is not
        # monotonic past pi/2, so ``Nd0 = sqrt(1 - sin^2)`` stays positive
        # for any angle and a 114.6 deg field would fold back silently to
        # 65.4 deg instead of being refused.
        if abs(th_x) >= 0.5 * np.pi or abs(th_y) >= 0.5 * np.pi:
            raise ValueError(
                f"eval_image_plane_wfe: field direction "
                f"({th_x:g}, {th_y:g}) rad is at or beyond 90 deg from "
                f"the axis; reduce field_max_rad.")
        Ld0 = -float(np.sin(th_x))
        Md0 = -float(np.sin(th_y))
        Nd0 = float(np.sqrt(max(1.0 - Ld0 ** 2 - Md0 ** 2, 0.0)))
        if Nd0 <= 0.0:
            raise ValueError(
                f"eval_image_plane_wfe: field direction "
                f"({th_x:g}, {th_y:g}) rad leaves no axial component; "
                f"reduce field_max_rad.")
        L = np.full_like(px, Ld0)
        M = np.full_like(px, Md0)
        tx = px * ep_r
        ty = py * ep_r
        tz = ep_z
        # Signed distance of each entrance-pupil target along the field
        # direction; the launch plane sits `back` behind the nearest one.
        proj = tx * Ld0 + ty * Md0 + tz * Nd0
        back = 2.0 * aperture_m + abs(ep_z) + 1e-3
        s = proj - (float(np.min(proj)) - back)
        bundle = _make_bundle(
            x=tx - s * Ld0, y=ty - s * Md0,
            L=L, M=M, wavelength=wavelength)
        bundle.z = tz - s * Nd0
        # OPL measured from the incident WAVEFRONT, not from the plane the
        # bundle happens to be launched on.  ``_make_bundle``'s own
        # ``opd_seed='eikonal'`` is the ``z = 0`` form and this bundle is
        # not on ``z = 0``, so use the functional helper, which carries the
        # ``N*z`` term (R2, AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11).  It
        # is a constant here by construction -- every launch point is the
        # foot of its own ray on one common wavefront -- and the constant
        # cancels in the chief-relative OPD below; seeding it anyway keeps
        # the launch correct if the construction ever changes.
        seed_entrance_eikonal(bundle)
    else:
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
        # NO entrance eikonal here: every ray leaves the SAME object
        # point, so the incident wavefront is a sphere of zero radius and
        # every ray starts with zero optical path.  ``_make_bundle``'s
        # default ``opd_seed='plane'`` is exactly that; the ``'eikonal'``
        # seed is for a COLLIMATED bundle and would add a spurious
        # ``L*src_x + M*src_y`` across the pupil for an off-axis source.
        _warn_object_distance_precision(
            surfaces, obj_d_m, wavelength, fod)

    res = trace(bundle, surfaces, wavelength, output_filter='last',
                **_way_back_kwargs(renormalize, sphere_normal))
    f = res.image_rays
    alive = np.asarray(f.alive, dtype=bool)
    opl = np.asarray(f.opd)

    # Identify chief = ALIVE ray closest to (0, 0) in pupil coords.  A
    # dead vignetted on-axis ray -- rare, but possible for systems where
    # the chief is geometrically blocked -- would NaN-poison every
    # downstream OPL calculation via ``opl[chief] = NaN``.  Fall back to
    # the unconstrained nearest if no rays survived (the caller sees the
    # all-NaN result anyway).
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
    # Leaving the sphere radius at the axial ``img_d_m`` while the
    # chief-image landing carries ``1/N_chief`` makes the sphere miss the
    # chief for off-axis fields, and the resulting quadratic shape error
    # is absorbed as phantom defocus by ``best_rms``.  On-axis (N = 1)
    # the factor is 1 and the distinction vanishes.
    # W4d (F2): FOLDED FRAME.  Everything below is done in the ALONG-THE-RAY
    # axial frame -- the one ``img_d_m`` (from ``bfl``/``pp_image_z``, both
    # unfolded per W4b/S11-1) is already expressed in, and the only frame in
    # which "downstream of the last surface" means the same thing for a
    # folded and an unfolded system.  After an ODD number of mirrors the
    # traced chief runs against global +z, so ``N_chief < 0``; the along-ray
    # z-cosine is ``_fold_sign * N_chief``, and every place a global-z
    # quantity meets an along-ray axial distance carries ``_fold_sign``
    # (``cz``, ``t_advance``, and ``xp_z``, which W3-T2/W4 define in GLOBAL
    # z).  Without ``_fold_sign``, on an air singlet + flat fold,
    # ``1/N_chief`` inverts every arc-length factor and the sphere is
    # centred on the wrong side: ``r_sphere_m`` comes back NEGATIVE
    # (measured -4.288895e-02 m) and the WFE reads 321.00 waves PV /
    # 100.17 waves RMS for a system that is a few waves unfolded.
    # ``_fold_sign`` is exactly +1 for every unfolded system (and every
    # even mirror count), so all of this is an IEEE no-op there.
    _fold_sign = _mirror_parity_sign(surfaces)
    if alive[chief]:
        _N_chief = float(Nd[chief])
        if abs(_N_chief) < 1e-12:
            N_chief_for_R = 1.0
        else:
            N_chief_for_R = _fold_sign * _N_chief
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
        # W4d: ``xp_z`` is a GLOBAL-z coordinate (W3-T2 / W4) while ``d``
        # is along-the-ray, so it is mapped into that frame first.
        return (d - _fold_sign * fod.xp_z) / N_chief_for_R

    R = _radius_for(img_d_m)
    cz = z_chief + _fold_sign * img_d_m

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
        # W4d: ``d`` is an ALONG-THE-RAY axial distance, so the arc length
        # uses the along-ray z-cosine ``_fold_sign * N_chief`` (identical
        # to ``N_chief`` for every unfolded system).  Without this a folded
        # chief advanced BACKWARDS to its "image".
        t_advance = d / (_fold_sign * N_chief)  # geometric path length
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
                # AN-5: the physical radius at which the normalised px=1 maps
                # is the EXIT-beam (marginal-ray) height at the reference-sphere
                # tangent plane, NOT the entrance-pupil semi-aperture -- they
                # differ by the pupil magnification, so ``semi`` leaves residual
                # defocus for strongly telephoto/retrofocus systems (exact only
                # at mag ~ 1).  Recover it from the traced ray positions
                # relative to the chief: r_phys = a_exit * sqrt(r2), so
                # a_exit = median(r_phys / sqrt(r2)) over the rim rays.
                r_pup = semi  # entrance-pupil semi-aperture [m] (fallback)
                _cx0, _cy0 = float(s2x[chief]), float(s2y[chief])
                _rphys = np.sqrt((s2x - _cx0) ** 2 + (s2y - _cy0) ** 2)
                _rmax = float(np.nanmax(r2[mask])) if mask.any() else 0.0
                _rim = mask & (r2 > 0.25 * _rmax) & (r2 > 1e-12)
                if _rim.any():
                    _scale = _rphys[_rim] / np.sqrt(r2[_rim])
                    _a_exit = float(np.median(_scale[np.isfinite(_scale)]))
                    if np.isfinite(_a_exit) and _a_exit > 0:
                        r_pup = _a_exit
                # The closed-form best-RMS uses the SPHERE radius (which
                # depends on ``sphere_tangent``), not ``img_d_m``
                # directly: for 'exit_pupil' the sphere radius is
                # ``img_d_m - fod.xp_z``, and ``1/img_d_m`` is wrong on
                # that branch.
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
                    # W4d: ``xp_z`` is GLOBAL z, ``img_d_m`` along-the-ray.
                    if sphere_tangent == 'exit_pupil':
                        img_d_m = (R_new * N_chief_for_R
                                   + _fold_sign
                                   * float(getattr(fod, 'xp_z', 0.0)))
                    else:
                        img_d_m = R_new * N_chief_for_R
        elif image_plane == 'best_pv':
            # 1-D numerical search over the longitudinal shift dz
            # that minimises PV.  Uses scipy.optimize when available;
            # falls back to a coarse + bisect search.
            def _pv_at(d_test):
                R_t = _radius_for(d_test)
                cz_t = z_chief + _fold_sign * d_test   # W4d: along-the-ray
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
            except (ImportError, ValueError, RuntimeError, AttributeError):
                # ImportError: scipy unavailable.  ValueError / RuntimeError:
                # bracket invalid or Brent failed to converge.
                # 21-point coarse scan as fallback.
                shifts = np.linspace(-dz_max, dz_max, 21)
                pvs = [_pv_at(img_d_m + dz) for dz in shifts]
                img_d_m = float(img_d_m + shifts[int(np.argmin(pvs))])
        # Re-evaluate with the shifted image distance
        R = _radius_for(img_d_m)
        cz = z_chief + _fold_sign * img_d_m       # W4d: along-the-ray
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
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # Per-field WFE eval can fail at extreme field
                # corners where the chief ray clips out -- leave the
                # entry at NaN and continue building the grid.
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
