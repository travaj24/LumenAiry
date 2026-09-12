"""
lumenairy.raytrace.paraxial -- small paraxial-design helpers.

A handful of one-liner utilities that get re-implemented across many
projects:

* :func:`field_of_view` -- maximum field angle / height a prescription
  can image given its entrance pupil and (optionally) a sensor size.
* :func:`optical_invariant` -- Lagrange invariant / etendue.
* :func:`f_number` -- f/D ratio from prescription + wavelength.
* :func:`defocus_waves_to_zernike` /
  :func:`astigmatism_waves_to_zernike` -- Born-Wolf / OSA conversions
  for common geometric aberration -> Zernike-coefficient mappings.

All functions are deliberately small, well-documented one-liners.
They sit on top of the existing :mod:`lumenairy.raytrace` machinery
(``surfaces_from_prescription``, ``system_abcd``, ``first_order_data``)
and don't duplicate any physics already implemented elsewhere.

Author: Andrew Traverso
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def field_of_view(prescription: dict, wavelength: float,
                  sensor_half_height_m: Optional[float] = None
                  ) -> Tuple[float, float]:
    """Return ``(half_FoV_radians, half_FoV_object_m)`` for a prescription.

    Two views of the field of view:

    * The **angular half-FoV** in object space.  For a finite-conjugate
      system this is ``arctan(h_obj_max / object_distance)`` with
      ``h_obj_max`` the object height that fills the sensor.  For an
      infinite-conjugate (collimated) source the angular FoV is set by
      the sensor: ``arctan(sensor_half_height / EFL)``.
    * The **object-space half-FoV in metres** -- the linear field height
      at the object plane corresponding to the angular FoV.

    A field of view is set by the SENSOR (the detector's angular
    subtense as seen from the object), which is why
    ``sensor_half_height_m`` drives both branches.

    Parameters
    ----------
    prescription : dict
        Lumenairy lens prescription with ``surfaces``, ``thicknesses``,
        and ``aperture_diameter``.  ``object_distance`` is read if
        present (finite conjugate); otherwise the system is treated
        as object-at-infinity.
    wavelength : float
        Vacuum wavelength [m] -- needed for the EFL computation.
    sensor_half_height_m : float, optional
        Sensor half-extent at the image plane [m].  Needed for BOTH
        conjugates (R7, see Notes).  When omitted on a finite-conjugate
        system the function falls back to the historical
        aperture-half-angle proxy and warns.

    Returns
    -------
    (theta_max_rad, h_obj_max_m) : (float, float)
        Maximum half-field angle [rad] and the corresponding object-
        space linear half-FoV [m].  For a true infinite conjugate
        ``h_obj_max_m`` is ``np.inf``.

    Notes
    -----
    R7 (AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11): the finite-conjugate
    branch must NOT return ``arctan((aperture/2) / object_distance)``:
    that is the object-space APERTURE half-angle, independent of the
    sensor and therefore not a field of view at all (it is the
    marginal-ray cone, i.e. the numerical aperture; see
    docs/history/lumenairy.raytrace.paraxial.md).  It is computed
    from the transverse magnification,
    ``m = -image_distance / object_distance`` via the Newtonian /
    Gaussian imaging relation, giving
    ``h_obj_max = sensor_half_height / |m|`` and
    ``theta_max = arctan(h_obj_max / object_distance)``.  The aperture
    expression survives only as the explicitly-warned fallback when the
    caller supplies no sensor size, because removing it outright would
    break callers that relied on the (mislabelled) return value.

    Examples
    --------
    >>> import lumenairy as la
    >>> p = la.thorlabs_lens('AC254-100-C')
    >>> p['object_distance'] = 0.5
    >>> theta_max, h_max = la.field_of_view(p, 1.31e-6,
    ...                                     sensor_half_height_m=4.8e-3)
    """
    import warnings

    from .core import surfaces_from_prescription, system_abcd
    surfs = surfaces_from_prescription(prescription)
    _, efl, _, _ = system_abcd(surfs, float(wavelength))

    obj_d = prescription.get('object_distance', 0.0)
    if obj_d is not None and obj_d > 0:
        obj_d = float(obj_d)
        if (sensor_half_height_m is not None
                and np.isfinite(efl) and efl != 0
                and obj_d != float(efl)):
            # Gaussian imaging about the principal planes: 1/s' - 1/s =
            # 1/f with s = -obj_d gives s' = f * obj_d / (obj_d - f),
            # and the transverse magnification is m = -s'/obj_d
            # = -f / (obj_d - f).  The FIELD is what the sensor sees:
            # h_obj = h_img / |m|.
            m_t = -float(efl) / (obj_d - float(efl))
            if np.isfinite(m_t) and m_t != 0.0:
                h_max = abs(float(sensor_half_height_m) / m_t)
                return float(np.arctan(h_max / obj_d)), float(h_max)
        # Fallback: the historical aperture-half-angle proxy.  This is
        # NOT a field of view -- see Notes.
        ap = float(prescription.get('aperture_diameter', 0.0))
        h_max = ap / 2.0
        warnings.warn(
            "field_of_view: no usable sensor_half_height_m for this "
            "finite-conjugate prescription, so the returned value is the "
            "object-space APERTURE half-angle "
            "arctan((aperture/2)/object_distance), which is a numerical "
            "aperture, not a field of view (it does not depend on the "
            "sensor at all).  Pass sensor_half_height_m to get the real "
            "field of view.",
            RuntimeWarning, stacklevel=2)
        theta_max = float(np.arctan(h_max / obj_d))
        return theta_max, float(h_max)

    # Infinite conjugate: need a sensor extent to scale.
    if sensor_half_height_m is None:
        return float('inf'), float('inf')
    if not np.isfinite(efl) or efl == 0:
        return float('inf'), float('inf')
    theta_max = float(np.arctan(float(sensor_half_height_m) / float(efl)))
    return theta_max, float('inf')


def optical_invariant(efl: float, f_number_val: float,
                       pupil_diameter_m: Optional[float] = None,
                       field_height_m: float = 0.0) -> float:
    """Lagrange / optical invariant: ``H = n * y * u``.

    For a single-medium-air system, ``n = 1``.  The standard
    paraxial formula:

    .. math::
        H = y \\cdot u
          = y_{pupil} \\cdot u_{chief}
          = h \\cdot u_{marginal}

    where ``y_pupil`` is the marginal-ray height at the entrance pupil
    and ``u_chief`` the chief-ray angle there.  For an entrance pupil of
    diameter ``D``, EFL ``f``, and image-side field-height ``h``, the
    invariant is the half-aperture times the chief-ray field-angle

    .. math::
        H = \\frac{D}{2} \\cdot \\frac{h}{f}.

    .. note:: the formula is ``H = (D/2) * (h/efl)``; an
        ``h*D/(4*(f/#)*f)`` form carries a spurious extra division by
        ``2*(f/#))``.

    Parameters
    ----------
    efl : float
        Effective focal length [m].
    f_number_val : float
        F-number (``f/D``).
    pupil_diameter_m : float, optional
        Entrance pupil diameter [m].  Derived from ``efl /
        f_number_val`` if omitted.
    field_height_m : float, default 0
        Image-side field height [m] at which the invariant is
        evaluated.

    Returns
    -------
    H : float
        Lagrange invariant [m·rad].

    Notes
    -----
    The Lagrange invariant is conserved through any rotationally-
    symmetric paraxial system, so this is a sanity-check helper for
    designs where ``f/#``, EFL, and FoV are specified independently.
    """
    if pupil_diameter_m is None:
        if f_number_val == 0:
            raise ValueError("f-number is zero; provide pupil_diameter_m.")
        pupil_diameter_m = float(efl) / float(f_number_val)
    # y_marginal at the entrance pupil = D/2
    y_marg = float(pupil_diameter_m) / 2.0
    # u_chief = field_height / efl (small-angle / paraxial)
    if efl == 0:
        return 0.0
    u_chief = float(field_height_m) / float(efl)
    return y_marg * u_chief


def f_number(prescription: dict, wavelength: float) -> float:
    """Paraxial f-number ``abs(EFL) / D_pupil`` for a prescription.

    Parameters
    ----------
    prescription : dict
        Lumenairy prescription with ``aperture_diameter`` and
        ``surfaces`` / ``thicknesses``.
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    float
        f-number, always NON-NEGATIVE.  Returns ``inf`` for a
        degenerate prescription (zero aperture or zero EFL).

    Notes
    -----
    R-11 (AUDIT_ADVERSARIAL_CODEBASE_2026_07_25): the returned f/# is
    ``abs(EFL) / D``.  A SIGNED ratio makes a diverging prescription read
    ``f/-9.97`` while all three siblings that compute the same quantity
    -- ``raytrace.layout`` (``abs(efl) / ap``),
    ``optimize.merit_terms.MaxFNumberMerit`` (``abs(ctx.efl) / ap``) and
    ``seidel.compute_pupils`` -- report
    ``+9.97``.  f/# is a cone-angle magnitude by definition (``1 / (2
    NA)``); read ``EFL`` itself if you need the conjugate sign.  No
    consumer relied on the sign (grep-verified: every call site either
    takes a converging prescription or abs()es already).
    """
    from .core import surfaces_from_prescription, system_abcd
    surfs = surfaces_from_prescription(prescription)
    _, efl, _, _ = system_abcd(surfs, float(wavelength))
    ap = float(prescription.get('aperture_diameter', 0.0))
    if ap <= 0 or not np.isfinite(efl) or efl == 0:
        return float('inf')
    return float(abs(efl) / ap)


def defocus_waves_to_zernike(defocus_waves: float) -> float:
    """Convert "geometric defocus in waves" to the OSA Z(2, 0) coefficient.

    Geometric defocus over the unit disk is the parabolic phase
    ``W(rho) = defocus_waves * rho^2``.  In the OSA normalisation
    where ``Z(2, 0) = sqrt(3) * (2*rho^2 - 1)``, the same defocus
    decomposes as

    .. math::
        c_{(2,0)} = \\frac{\\text{defocus\\_waves}}{2 \\sqrt{3}}.

    Parameters
    ----------
    defocus_waves : float
        Peak-to-edge defocus expressed in waves.

    Returns
    -------
    float
        OSA Zernike coefficient for ``(n, m) = (2, 0)`` (defocus mode),
        in the same units (waves).

    Notes
    -----
    Inverse: the OSA c_{(2,0)} = a in waves corresponds to a Marechal
    Strehl drop of ``exp(-(2 pi a)^2)`` for small a.
    """
    return float(defocus_waves) / (2.0 * np.sqrt(3.0))


def astigmatism_waves_to_zernike(astig_waves: float) -> float:
    """Convert "geometric astigmatism in waves" to the OSA Z(2, ±2) coefficient.

    Geometric astigmatism over the unit disk along, say, the
    x-axis is ``W(rho, theta) = astig_waves * rho^2 * cos(2*theta)``.
    In the OSA normalisation where ``Z(2, 2) = sqrt(6) * rho^2 * cos(2 theta)``,
    the same astigmatism decomposes as

    .. math::
        c_{(2, 2)} = \\frac{\\text{astig\\_waves}}{\\sqrt{6}}.

    Parameters
    ----------
    astig_waves : float
        Peak-to-edge astigmatism magnitude in waves.

    Returns
    -------
    float
        OSA Zernike coefficient for ``(n, m) = (2, 2)``.

    Notes
    -----
    The same formula applies to ``(n, m) = (2, -2)`` by symmetry.
    """
    return float(astig_waves) / np.sqrt(6.0)


__all__ = [
    'field_of_view',
    'optical_invariant',
    'f_number',
    'defocus_waves_to_zernike',
    'astigmatism_waves_to_zernike',
]
