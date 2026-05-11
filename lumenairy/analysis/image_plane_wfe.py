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
        Image distance used for the reference sphere [m].
    wavelength_m : float
        Wavelength used [m].
    aperture_m : float
        Entrance-pupil diameter used [m].
    """
    px: np.ndarray
    py: np.ndarray
    opd_w: np.ndarray
    alive: np.ndarray
    chief_idx: int
    img_d_m: float
    wavelength_m: float
    aperture_m: float

    @property
    def pv_waves(self) -> float:
        """Peak-to-valley OPD over alive rays, in waves."""
        v = self.opd_w[self.alive]
        v = v[np.isfinite(v)]
        return float(v.max() - v.min()) if v.size else float('nan')

    @property
    def rms_waves(self) -> float:
        """RMS OPD over alive rays, in waves."""
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


def eval_image_plane_wfe(
    prescription: dict,
    wavelength: float,
    field: Tuple[float, float] = (0.0, 0.0),
    n_pupil: int = 31,
    img_d_m: Optional[float] = None,
) -> ImagePlaneWFE:
    """Compute image-plane reference-sphere wavefront error.

    For each pupil-grid ray traced from the object plane through the
    lens, finds the exact intersection with the reference sphere
    centered on the chief ray's image-plane intersect, with radius
    equal to the chief lens-exit-to-image distance.  The chief is
    on the sphere by construction (zero OPD); marginal-ray OPDs
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
        Currently only the on-axis case ``(0, 0)`` is supported;
        off-axis fields require chief-ray offset handling that
        will be added in a later release.  A non-zero field raises
        ``NotImplementedError``.
    n_pupil : int, default 31
        Pupil samples per axis (a square ``n_pupil x n_pupil``
        Cartesian grid is generated and clipped to the unit disk).
    img_d_m : float, optional
        Override the paraxial image distance from the last lens
        surface [m].  Defaults to
        :func:`lumenairy.raytrace.find_paraxial_focus` of the
        prescription's surfaces.  Negative values are supported
        for diverging systems with virtual images.

    Returns
    -------
    ImagePlaneWFE
        Per-ray pupil coordinates, OPD in waves, alive mask, and
        aggregated PV / RMS / Strehl summaries.

    Raises
    ------
    NotImplementedError
        For non-zero ``field``.
    ValueError
        For invalid or empty prescriptions.

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
    formula remains continuous through ``t_chief = 0`` for both
    converging (``img_d_m > 0``) and diverging (``img_d_m < 0``)
    systems.  The OPL adjustment is ``t * n_air`` (with
    ``n_air = 1``), added to the chief-relative lens-exit OPL,
    sign-flipped to match the rayoptics / Optiland convention,
    then re-zeroed on the chief value.
    """
    if field != (0.0, 0.0):
        raise NotImplementedError(
            'eval_image_plane_wfe: off-axis fields (Hx,Hy != 0,0) '
            'are not yet supported.  Use field=(0, 0).')
    if not prescription.get('surfaces'):
        raise ValueError(
            'eval_image_plane_wfe: prescription has no "surfaces".')

    surfaces = surfaces_from_prescription(prescription)

    obj_d_m = float(prescription.get('object_distance', 0.0))
    if obj_d_m <= 0:
        raise ValueError(
            f'eval_image_plane_wfe: prescription object_distance must '
            f'be > 0 (got {obj_d_m:g} m).')

    if img_d_m is None:
        # Compute the paraxial image distance for the actual finite
        # object conjugate (not the BFL, which is the infinity-
        # conjugate special case).  Use the Gauss imaging equation
        # at the principal planes:
        #     1/v - 1/u = 1/f,  with u = -obj_d_m (object before lens)
        # then offset by the image-side principal-plane position.
        fod = first_order_data(surfaces, wavelength)
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

    aperture_m = float(prescription.get('aperture_diameter', 0.0))
    if aperture_m <= 0:
        raise ValueError(
            f'eval_image_plane_wfe: aperture_diameter must be > 0 '
            f'(got {aperture_m:g} m).')
    semi = aperture_m / 2.0

    # Square pupil grid clipped to the unit disk
    p1 = np.linspace(-1.0, 1.0, n_pupil)
    PX, PY = np.meshgrid(p1, p1)
    px = PX.ravel(); py = PY.ravel()
    inside = (px ** 2 + py ** 2) <= 1.0
    px = px[inside]; py = py[inside]

    # On-axis object: rays launch from (0,0,-obj_d) aimed at the
    # pupil position (px*semi, py*semi, 0).  Direction cosines:
    h_x = px * semi
    h_y = py * semi
    L = h_x / np.sqrt(h_x ** 2 + h_y ** 2 + obj_d_m ** 2)
    M = h_y / np.sqrt(h_x ** 2 + h_y ** 2 + obj_d_m ** 2)

    bundle = _make_bundle(x=np.zeros_like(px), y=np.zeros_like(px),
                            L=L, M=M, wavelength=wavelength)
    bundle.z = np.full(px.size, -obj_d_m)

    res = trace(bundle, surfaces, wavelength, output_filter='last')
    f = res.image_rays
    alive = np.asarray(f.alive, dtype=bool)
    opl = np.asarray(f.opd)

    # Identify chief = ray closest to (0, 0) in pupil coords
    chief = int(np.argmin(px ** 2 + py ** 2))

    # Conv-A: chief-relative OPL at the last lens surface (in WAVES)
    opl_chief = opl[chief] if alive[chief] else float(
        np.median(opl[alive]) if alive.any() else 0.0)
    opd_a_w = (opl - opl_chief) / wavelength

    # Exact ray-sphere intersection -- sphere centered on chief
    # image intersect with radius = img_d_m (signed)
    s2x = np.asarray(f.x)
    s2y = np.asarray(f.y)
    s2z = np.asarray(f.z)
    Ld = np.asarray(f.L)
    Md = np.asarray(f.M)
    Nd = np.asarray(f.N)
    z_chief = s2z[chief]
    cz = z_chief + img_d_m
    R = img_d_m
    b = 2.0 * (s2x * Ld + s2y * Md + (s2z - cz) * Nd)
    c = s2x ** 2 + s2y ** 2 + (s2z - cz) ** 2 - R ** 2
    disc = b ** 2 - 4.0 * c
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    t1 = (-b - sqrt_disc) / 2.0
    t2 = (-b + sqrt_disc) / 2.0
    # Smallest-|t| root: works for both converging (img_d_m > 0)
    # and diverging (img_d_m < 0) systems.
    t = np.where(np.abs(t1) < np.abs(t2), t1, t2)

    rs_w = -(opd_a_w + t / wavelength)  # sign-flip to match rayoptics
    if np.isfinite(rs_w[chief]):
        rs_w = rs_w - rs_w[chief]
    rs_w = np.where(alive, rs_w, np.nan)

    return ImagePlaneWFE(
        px=px, py=py, opd_w=rs_w, alive=alive,
        chief_idx=chief, img_d_m=img_d_m,
        wavelength_m=wavelength, aperture_m=aperture_m,
    )


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
