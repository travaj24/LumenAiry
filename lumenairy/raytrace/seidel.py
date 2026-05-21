"""
v5.1.0 split: paraxial / ABCD / pupil / first-order / Seidel analytics.

Extracted from ``lumenairy/raytrace/core.py`` as part of the v5.1.0
6-file split (ROADMAP Agent B).  The ROADMAP topology labels this
slice "seidel", but the actual content covers the whole paraxial-
aberration-analysis cluster that builds up to the Seidel sums:

* :func:`system_abcd`, :func:`system_abcd_prescription` -- system
  ABCD matrix from paraxial ray tracing.
* :class:`LensInfo`, :func:`lens_abcd`, :func:`find_lenses` -- per-
  element paraxial characterisation.
* :class:`PupilInfo`, :func:`compute_pupils` -- entrance / exit pupil
  positions and radii.
* :class:`FirstOrderData`, :func:`first_order_data` -- combined
  paraxial first-order report.
* :func:`seidel_coefficients`, :func:`seidel_prescription` -- Buchdahl-
  Hopkins third-order aberration coefficients.
* :func:`find_paraxial_focus` -- one-liner BFL accessor.

These pieces form an unbreakable dependency chain: every analytic
here is built on ``system_abcd``'s paraxial ABCD matrix and the
Welford reduced-coordinate paraxial trace.  Splitting them further
would require duplicating the paraxial trace helpers.

Every public name here is re-exported from
``lumenairy.raytrace.core`` so existing imports continue to resolve.

No physics change: contents are bit-for-bit copies of the original
implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from ..glass import get_glass_index
from .surface import Surface, _surface_copy_with
from .trace import find_stop, surfaces_from_prescription

# ============================================================================
# Paraxial ray trace and ABCD matrix
# ============================================================================

def _paraxial_trace(surfaces, wavelength, y_in=0.0, u_in=0.0):
    """Trace a single paraxial ray (y, u) through the system.

    Uses the exact paraxial recursion:
        y' = y + u * t          (transfer)
        u' = u - y * phi / n'   (refraction, phi = (n'-n)/R)

    Returns lists of (y, u) at each surface (after refraction).
    """
    y_hist = []
    u_hist = []

    y = float(y_in)
    u = float(u_in)

    for i, surf in enumerate(surfaces):
        # 3.7.0: Coord-breaks have no power; just apply their
        # transfer thickness and record the unchanged state.
        if surf.is_coordbrk:
            y_hist.append(y)
            u_hist.append(u)
            if i < len(surfaces) - 1:
                y = y + u * surf.thickness
            continue

        n1 = get_glass_index(surf.glass_before, wavelength)
        n2 = get_glass_index(surf.glass_after, wavelength)
        R = surf.radius

        # Refraction (power).  Paraxial refraction equation:
        #    n2 * u2  =  n1 * u1  -  y * (n2 - n1) / R
        # So the correct update for u ( = u' in the new medium) is:
        #    u  <-  (n1 * u_prev - y * phi) / n2
        # Historical equivalent form below (u prior to update is still
        # n1-normalised):  u <- u - y * phi / n2 .  Both agree because
        # u_prev on the right-hand side already satisfies u_prev = u
        # at this point in the loop (no intervening rewrite).
        if surf.is_mirror:
            phi = 2.0 * n1 / R if np.isfinite(R) else 0.0
            u = u - y * phi / n1
            n2 = n1  # medium doesn't change for mirrors
        else:
            phi = (n2 - n1) / R if np.isfinite(R) else 0.0
            u = u - y * phi / n2

        y_hist.append(y)
        u_hist.append(u)

        # Transfer to next surface
        if i < len(surfaces) - 1:
            t = surf.thickness
            y = y + u * t

    return y_hist, u_hist


def _paraxial_refract(y, n1_u, R, n1, n2):
    """Paraxial refraction: returns n2*u2 given n1*u1."""
    if np.isinf(R):
        return n1_u  # flat surface, no power
    return n1_u - y * (n2 - n1) / R


def _paraxial_transfer(y, n_u, t, n):
    """Paraxial transfer: y2 = y1 + u1 * t, n*u unchanged."""
    u = n_u / n
    return y + u * t, n_u


def system_abcd(
    surfaces: List['Surface'],
    wavelength: float,
) -> Tuple[np.ndarray, float, float, float]:
    """Compute the system ABCD matrix using paraxial ray tracing.

    Traces a marginal ray (y=1, u=0) and an axial ray (y=0, u=1)
    through the surface list and constructs the 2x2 system matrix.

    Parameters
    ----------
    surfaces : list of Surface
        Surface list (from :func:`surfaces_from_prescription`).
    wavelength : float
        Vacuum wavelength [m].

    Returns
    -------
    abcd : ndarray, shape (2, 2)
        System matrix ``[[A, B], [C, D]]``.
    efl : float
        Effective focal length ``-1/C`` [m].
    bfl : float
        Back focal length (distance from last surface to rear focus) [m].
    ffl : float
        Front focal length (distance from first surface to front focus) [m].
    """
    # Build system matrix by multiplying surface-by-surface.
    #
    # Mirror sign convention (v4.11.2)
    # ---------------------------------
    # We adopt the Welford convention in which a mirror is treated as
    # a refracting surface with ``n2 = -n1``.  In this convention the
    # post-mirror "index" carries a sign flip; after the k-th mirror
    # the effective index in every subsequent leg is multiplied by
    # ``(-1)**mirror_parity`` (``mirror_parity = mirror_count % 2``).
    # This brings ``system_abcd`` into agreement with
    # ``seidel_coefficients`` -- pre-4.11.2 the two used opposite
    # mirror-power signs, so a single concave mirror (R=-100mm) gave
    # EFL=-0.05m from ``system_abcd`` and the equivalent of +0.05m
    # from the seidel branch.  +0.05m is the conventional answer
    # (concave mirror has positive focal length) and is the convention
    # both codepaths now agree on.
    M = np.eye(2)
    mirror_parity = 0  # 0 = unflipped, 1 = post-odd-mirror (n -> -n)

    for i, surf in enumerate(surfaces):
        sign = 1.0 if mirror_parity == 0 else -1.0
        # 3.7.0: Skip the refraction matrix for coord-breaks (they're
        # frame transforms, not optical surfaces with power) but
        # still apply their thickness as an air-gap transfer so the
        # cumulative axial separation matches the trace.
        if surf.is_coordbrk:
            if i < len(surfaces) - 1:
                t = surf.thickness
                # n_after is the medium of the post-cb leg; for a
                # coord-break that's the same as the pre-cb medium.
                n_after = sign * get_glass_index(surf.glass_after, wavelength)
                T_mat = np.array([[1.0, t / n_after],
                                  [0.0, 1.0]])
                M = T_mat @ M
            continue

        # Apply current mirror-parity sign so the post-odd-mirror
        # medium correctly carries n -> -n through subsequent surfaces.
        n1 = sign * get_glass_index(surf.glass_before, wavelength)
        n2 = sign * get_glass_index(surf.glass_after, wavelength)
        R = surf.radius

        # Refraction matrix
        if np.isfinite(R) and not surf.is_mirror:
            phi = (n2 - n1) / R
            R_mat = np.array([[1.0, 0.0],
                              [-phi, 1.0]])
        elif surf.is_mirror and np.isfinite(R):
            # Welford mirror: n_after = -n_before.  Power is
            # phi = (n2 - n1)/R = -2*n1/R.  This is the OPPOSITE sign
            # from the pre-4.11.2 formula (phi = +2*n1/R) and matches
            # seidel_coefficients's mirror branch.
            n2 = -n1
            phi = (n2 - n1) / R   # = -2 * n1 / R
            R_mat = np.array([[1.0, 0.0],
                              [-phi, 1.0]])
            # Flip parity for the post-mirror legs.
            mirror_parity ^= 1
        else:
            R_mat = np.eye(2)

        M = R_mat @ M

        # Transfer matrix to next surface
        if i < len(surfaces) - 1:
            t = surf.thickness
            n_after = n2
            T_mat = np.array([[1.0, t / n_after],
                              [0.0, 1.0]])
            M = T_mat @ M

    A, _B = M[0, 0], M[0, 1]
    C, D = M[1, 0], M[1, 1]

    efl = -1.0 / C if abs(C) > 1e-30 else np.inf
    bfl = -A / C if abs(C) > 1e-30 else np.inf
    ffl = -D / C if abs(C) > 1e-30 else np.inf

    return M, efl, bfl, ffl


def system_abcd_prescription(
    prescription: Dict[str, Any],
    wavelength: float,
) -> Tuple[np.ndarray, float, float, float]:
    """Compute the ABCD matrix from a lens prescription dict.

    Convenience wrapper around :func:`system_abcd`.
    """
    surfaces = surfaces_from_prescription(prescription)
    return system_abcd(surfaces, wavelength)


# ============================================================================
# Per-lens ABCD helpers (3.1.9)
# ============================================================================

@dataclass
class LensInfo:
    """Paraxial characterisation of a single lens element.

    Returned by :func:`lens_abcd` and :func:`find_lenses`.  Captures
    everything you'd typically quote from a datasheet: effective focal
    length, back/front focal lengths, principal-plane positions, and
    the underlying ABCD matrix so callers can compose it with other
    transfer matrices.

    All lengths in metres.  ``principal_planes`` returns ``(H, H')``
    where H is measured from the first surface (positive = rearward)
    and H' is measured from the last surface (positive = forward).
    This follows the usual Welford / Hecht convention; Zemax
    reports H and H' with the opposite sign convention on
    ``radius < 0`` systems -- cross-check signs before comparing.
    """
    abcd: np.ndarray               # (2, 2) air-to-air
    efl: float
    bfl: float
    ffl: float
    principal_planes: tuple        # (H, H')
    thickness: float               # total center thickness
    start_index: Optional[int] = None
    end_index: Optional[int] = None
    label: str = ''


def lens_abcd(
    lens: Union[Dict[str, Any], List['Surface'], 'Surface', 'LensInfo'],
    wavelength: float,
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    label: Optional[str] = None,
    surfaces: Optional[List['Surface']] = None,
) -> 'LensInfo':
    """Compute paraxial ABCD + EFL/BFL/FFL for a single lens element.

    Accepts any of the following forms of ``lens``:

    - **Prescription dict** (has ``'surfaces'`` + ``'thicknesses'``):
      treats the whole prescription as one lens (entire air-to-air
      section).  Same format consumed by
      :func:`apply_real_lens` / :func:`apply_real_lens_traced`.
    - **List of `Surface`**: treated as the lens in its entirety.
      Pass ``start`` / ``end`` to select a contiguous slice
      (inclusive both ends) -- useful when you have a full system's
      surface list and know surfaces ``[s:e+1]`` constitute one
      physical element (cemented doublet, etc.).
    - **Single `Surface`** (e.g. a mirror treated as a "lens"):
      characterised as a one-surface optic.
    - **`LensInfo`** (as returned by :func:`find_lenses`): requires the
      ``surfaces`` kwarg to be the original surface list the LensInfo
      was derived from.  Useful for re-analyzing a detected lens at a
      different ``wavelength``.  The returned LensInfo preserves the
      original ``start_index``, ``end_index``, and ``label``.

    The trailing-thickness gap on the last surface is stripped before
    the ABCD computation so the returned ABCD is the lens alone
    (air-to-air at the last vertex), not "lens plus whatever
    propagation came after."  If you need lens + downstream gap,
    build a ``T(d)`` transfer matrix separately and compose.

    Note on the polymorphic dispatch
    --------------------------------
    The 2026-04 roadmap proposed a larger API accepting
    :mod:`system.py`-style element dicts (``{'type': 'lens', ...}``)
    as well.  That dispatch is intentionally omitted: callers with a
    system-element dict have a one-line unwrap path
    (``surfaces_from_elements([elem], wavelength)``), and the
    narrower API here is easier to reason about and type-check.

    Parameters
    ----------
    lens : dict | list[Surface] | Surface | LensInfo
    wavelength : float
        Vacuum wavelength [m].  ABCD is wavelength-dependent through
        glass dispersion, so no silent default.
    start, end : int, optional
        Inclusive surface-index range when ``lens`` is a list of
        surfaces; ignored otherwise.
    label : str, optional
        Override the auto-generated label on the returned LensInfo.
    surfaces : list[Surface], optional
        Required when ``lens`` is a ``LensInfo``: the original surface
        list the LensInfo was derived from.  Ignored otherwise.

    Returns
    -------
    LensInfo

    Examples
    --------
    >>> rx = make_doublet(...)
    >>> info = lens_abcd(rx, 1.31e-6)
    >>> print(f"EFL = {info.efl*1e3:.2f} mm, H' = {info.principal_planes[1]*1e3:.2f} mm")

    >>> # Auto-detect all lenses in a compound system
    >>> surfaces = surfaces_from_prescription(full_rx)
    >>> for L in find_lenses(surfaces, 1.31e-6):
    ...     print(f"{L.label}: EFL = {L.efl*1e3:.2f} mm")

    >>> # Re-analyze a detected lens at a different wavelength
    >>> lenses = find_lenses(surfaces, 1.31e-6)
    >>> L_vis = lens_abcd(lenses[0], 0.55e-6, surfaces=surfaces)
    """
    # ---- Dispatch input to a surface slice -----------------------
    if isinstance(lens, LensInfo):
        if surfaces is None:
            raise ValueError(
                "lens_abcd: LensInfo input requires the 'surfaces' kwarg "
                "(the original surface list the LensInfo was derived from).")
        if not isinstance(surfaces, list) or \
                not all(isinstance(s, Surface) for s in surfaces):
            raise TypeError(
                "lens_abcd: 'surfaces' must be a list of Surface objects.")
        if lens.start_index is None or lens.end_index is None:
            raise ValueError(
                "lens_abcd: LensInfo has no start_index/end_index -- it "
                "was not produced by find_lenses and cannot be sliced.")
        s_idx = int(lens.start_index)
        e_idx = int(lens.end_index)
        if not (0 <= s_idx <= e_idx < len(surfaces)):
            raise ValueError(
                f"lens_abcd: LensInfo indices [{s_idx}..{e_idx}] out of "
                f"range for surfaces list of length {len(surfaces)}.")
        sub = list(surfaces[s_idx:e_idx + 1])
        auto_label = lens.label or f'Lens@surfaces[{s_idx}..{e_idx}]'
    elif isinstance(lens, Surface):
        sub = [lens]
        auto_label = lens.label or 'Lens'
        s_idx, e_idx = None, None
    elif isinstance(lens, dict):
        if 'surfaces' not in lens:
            raise ValueError(
                "lens_abcd: dict input must be a prescription with "
                "a 'surfaces' key.  Got keys: "
                f"{sorted(lens.keys())!r}.")
        sub = surfaces_from_prescription(lens)
        auto_label = lens.get('name', 'Lens')
        s_idx, e_idx = None, None
    elif isinstance(lens, list):
        if not lens:
            raise ValueError("lens_abcd: empty surface list.")
        if not all(isinstance(s, Surface) for s in lens):
            raise TypeError(
                "lens_abcd: list input must contain Surface objects.")
        s_idx = 0 if start is None else int(start)
        e_idx = (len(lens) - 1) if end is None else int(end)
        if not (0 <= s_idx <= e_idx < len(lens)):
            raise ValueError(
                f"lens_abcd: start={start} / end={end} out of range "
                f"for surface list of length {len(lens)}.")
        sub = list(lens[s_idx:e_idx + 1])
        auto_label = (sub[0].label
                       if (sub[0].label and len(sub) == 1)
                       else f'Lens@surfaces[{s_idx}..{e_idx}]')
    else:
        raise TypeError(
            f"lens_abcd: unsupported lens type {type(lens).__name__}.  "
            f"Pass a prescription dict, list of Surface, a single "
            f"Surface, or a LensInfo (with the 'surfaces' kwarg).")

    # ---- Strip trailing thickness so ABCD is air-to-air ----------
    last = len(sub) - 1
    sub = [
        _surface_copy_with(s, thickness=(0.0 if i == last else s.thickness))
        for i, s in enumerate(sub)
    ]

    # ---- ABCD + paraxial focal quantities ------------------------
    M, efl, bfl, ffl = system_abcd(sub, wavelength)
    A, _B, C, D = float(M[0, 0]), float(M[0, 1]), \
                  float(M[1, 0]), float(M[1, 1])
    if abs(C) > 1e-30:
        # Welford convention: H = (D-1)/C (from front vertex),
        # H' = (1-A)/C (from rear vertex).
        H  = (D - 1.0) / C
        Hp = (1.0 - A) / C
    else:
        H, Hp = float('inf'), float('inf')

    thickness = sum(sub[k].thickness for k in range(len(sub) - 1))

    return LensInfo(
        abcd=np.asarray(M, dtype=np.float64),
        efl=float(efl),
        bfl=float(bfl),
        ffl=float(ffl),
        principal_planes=(H, Hp),
        thickness=float(thickness),
        start_index=s_idx,
        end_index=e_idx,
        label=(label if label is not None else auto_label),
    )


@dataclass
class PupilInfo:
    """Paraxial entrance and exit pupil characterisation.

    Both EP and XP positions are given relative to the lens's own
    reference surfaces:
    * ``ep_z``: axial distance from ``surfaces[0]`` to EP.
      Negative = EP is to the left (object side) of the first surface.
    * ``xp_z``: axial distance from ``surfaces[-1]`` to XP.
      Positive = XP is to the right (image side) of the last surface.
    """
    ep_z: float
    ep_radius: float
    xp_z: float
    xp_radius: float
    stop_index: int


@dataclass
class FirstOrderData:
    """Comprehensive paraxial first-order characterisation of a system.

    Combines :class:`PupilInfo`, the system ABCD matrix, and the
    common focal-length / focus-distance / principal-plane summaries
    that downstream analysis (image-plane wavefront error,
    diffraction-limited PSF / MTF estimates, tolerance budgets)
    needs.

    All distances are in metres unless noted; positions follow the
    same sign convention as :class:`PupilInfo` -- ``ep_z`` is
    measured from ``surfaces[0]`` (negative = object-side of the
    first surface), ``xp_z`` is measured from ``surfaces[-1]``
    (positive = image-side of the last surface).

    Attributes
    ----------
    efl : float
        Effective focal length [m].  ``-1 / C`` where ``C`` is the
        bottom-left ABCD element.
    bfl : float
        Back focal length: distance from the LAST surface vertex to
        the rear focal point [m].  Positive = on image side.
    ffl : float
        Front focal length: distance from the FIRST surface vertex to
        the front focal point [m].  Negative typically (focal point
        on object side).
    ep_z, ep_radius : float
        Entrance pupil axial position and radius (see
        :class:`PupilInfo`).
    xp_z, xp_radius : float
        Exit pupil axial position and radius (see
        :class:`PupilInfo`).
    pp_object_z : float
        Object-space principal plane H, measured from
        ``surfaces[0]``.  Positive = image-side of the first
        vertex.  Computed as ``efl - ffl``: front focal point is at
        ``-ffl`` from the first vertex (sign convention here treats
        ``ffl`` as the unsigned magnitude returned by
        :func:`system_abcd`), and H is one EFL forward of it.
        For a thin lens, ``efl == ffl`` and H collapses to the
        vertex (zero).
    pp_image_z : float
        Image-space principal plane H', measured from
        ``surfaces[-1]``.  Positive = image-side.
        ``pp_image_z = bfl - efl`` (rear focal point measured from
        last vertex, walked backward by efl).
    fnum : float
        Working f-number (``efl / (2 * ep_radius)``) when EP radius
        is finite, else ``inf``.
    abcd : ndarray, shape (2, 2)
        Full system ABCD matrix from
        :func:`system_abcd`.
    stop_index : int
        Surface index of the aperture stop used for pupil imaging.
    """
    efl: float
    bfl: float
    ffl: float
    ep_z: float
    ep_radius: float
    xp_z: float
    xp_radius: float
    pp_object_z: float
    pp_image_z: float
    fnum: float
    abcd: 'np.ndarray'
    stop_index: int

    def summary(self, units: str = 'mm') -> str:
        """One-page text summary, useful for GUI or stdout printouts."""
        s = {'m': 1.0, 'mm': 1e3, 'um': 1e6, 'µm': 1e6}.get(units, 1e3)
        u = 'mm' if units == 'm' else units  # display label
        if units == 'm':
            s = 1.0
            u = 'm'
        lines = [
            'First-order data:',
            f'  EFL = {self.efl*s:+.4f} {u}',
            f'  BFL = {self.bfl*s:+.4f} {u}   (last vertex -> rear focus)',
            f'  FFL = {self.ffl*s:+.4f} {u}   (first vertex -> front focus)',
            f'  f/# = {self.fnum:.3f}',
            f'  EP  z={self.ep_z*s:+.4f} {u}  radius={self.ep_radius*s:.4f} {u}',
            f'  XP  z={self.xp_z*s:+.4f} {u}  radius={self.xp_radius*s:.4f} {u}',
            f'  H   (object PP, from surf 0)        '
            f'= {self.pp_object_z*s:+.4f} {u}',
            f"  H'  (image  PP, from last surface) "
            f'= {self.pp_image_z*s:+.4f} {u}',
            f'  Stop surface index = {self.stop_index}',
        ]
        return '\n'.join(lines)


def first_order_data(
    surfaces_or_prescription: Union[List['Surface'], Dict[str, Any]],
    wavelength: float,
    stop_index: Optional[int] = None,
) -> 'FirstOrderData':
    """Compute a comprehensive paraxial first-order summary of a system.

    Combines :func:`system_abcd`, :func:`compute_pupils`, and the
    standard focal-length / principal-plane geometry into a single
    :class:`FirstOrderData` record.

    Parameters
    ----------
    surfaces_or_prescription : list of Surface OR prescription dict
        Accepts either a pre-built surface list (as from
        :func:`surfaces_from_prescription`) or a raw prescription
        dictionary; the latter is converted internally.
    wavelength : float
        Vacuum wavelength [m].
    stop_index : int, optional
        Override the aperture-stop surface index.  Defaults to the
        result of :func:`find_stop`.

    Returns
    -------
    FirstOrderData
        Single record with EFL, BFL, FFL, EP/XP positions and radii,
        principal-plane locations, f-number, and the underlying ABCD
        matrix.

    Notes
    -----
    Principal planes are computed from the focal lengths via the
    standard relations:

        H   = ffl + efl       (object-side, from first surface)
        H'  = bfl - efl       (image-side,  from last surface)

    These are the conjugate points where the system acts as a
    "thin lens" of focal length ``efl``.  For a thin lens both
    collapse to zero; for thick or compound systems they offset
    by the principal-plane separation.
    """
    if isinstance(surfaces_or_prescription, dict):
        surfaces = surfaces_from_prescription(surfaces_or_prescription)
    else:
        surfaces = surfaces_or_prescription
    abcd, efl, bfl, ffl = system_abcd(surfaces, wavelength)
    pupils = compute_pupils(surfaces, wavelength, stop_index=stop_index)
    pp_object_z = efl - ffl       # H from first vertex
    pp_image_z = bfl - efl        # H' from last vertex
    if np.isfinite(pupils.ep_radius) and pupils.ep_radius > 0:
        fnum = abs(efl) / (2.0 * pupils.ep_radius)
    else:
        fnum = float('inf')
    return FirstOrderData(
        efl=float(efl),
        bfl=float(bfl),
        ffl=float(ffl),
        ep_z=float(pupils.ep_z),
        ep_radius=float(pupils.ep_radius),
        xp_z=float(pupils.xp_z),
        xp_radius=float(pupils.xp_radius),
        pp_object_z=float(pp_object_z),
        pp_image_z=float(pp_image_z),
        fnum=float(fnum),
        abcd=abcd,
        stop_index=int(pupils.stop_index),
    )


def compute_pupils(
    surfaces: List['Surface'],
    wavelength: float,
    stop_index: Optional[int] = None,
) -> 'PupilInfo':
    """Paraxial entrance and exit pupil positions and radii.

    Images the aperture stop backward through the pre-stop optics to
    find the entrance pupil, and forward through the post-stop optics
    to find the exit pupil.  Both are computed from the sub-system
    ABCD matrices; no ray tracing needed.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
        Vacuum wavelength [m].
    stop_index : int, optional
        Explicit stop surface index.  Defaults to the result of
        :func:`find_stop` (i.e. the surface flagged ``is_stop=True``,
        or the first finite-semi-diameter surface, or 0).

    Returns
    -------
    PupilInfo

    Notes
    -----
    For the EP, we seek the object-space conjugate of the stop:
    image distance from surface 0 at which an object placed there
    would image onto the stop plane.  Equivalently, treat the stop
    as the "source" and propagate in reverse through the pre-stop
    sub-system.  For a reversed sub-system M_rev = T(-t1)
    L1^{-1} T(-t2) L2^{-1} ..., but the cleanest implementation is
    the imaging condition on the forward sub-system's ABCD:
    if M_pre = [[A, B], [C, D]] maps (y_obj, u_obj) at surface 0
    to (y_stop, u_stop) at the stop, then the object-space position
    z_ep (measured from surface 0, negative = to the left) that
    images to the stop satisfies A + B / (z_ep * ... ) = 0 after
    prepending T(|z_ep|).  Equivalently, solve B_new = 0 for the
    prepended distance:  B + z_ep * A = 0  =>  z_ep = -B / A.

    For the XP: same logic on the post-stop sub-system in the
    forward direction, with the stop as the object.
    """
    if not surfaces:
        raise ValueError("compute_pupils: empty surface list.")
    if stop_index is None:
        stop_index = find_stop(surfaces)
    if not (0 <= stop_index < len(surfaces)):
        raise ValueError(
            f"compute_pupils: stop_index={stop_index} out of range "
            f"[0, {len(surfaces)})")

    stop_surf = surfaces[stop_index]
    # Stop radius from the surface's semi_diameter (fall back to a
    # reasonable default when infinite, with a warning -- infinite
    # semi-diameter means "no stop was really declared here").
    stop_radius = stop_surf.semi_diameter
    if not np.isfinite(stop_radius):
        import warnings
        warnings.warn(
            f"compute_pupils: stop surface at index {stop_index} has "
            f"infinite semi_diameter; pupil radii will be reported "
            f"as NaN.  Declare a finite semi_diameter to get "
            f"meaningful pupil sizes.",
            UserWarning, stacklevel=2)
        stop_radius = float('nan')

    # ---- Entrance pupil -------------------------------------------
    # Pre-stop sub-system: surfaces[0 .. stop_index-1], ending with
    # the thickness from the last pre-stop surface to the stop's
    # vertex (i.e. include the propagation gap up to the stop).
    if stop_index == 0:
        # Stop is at the first surface; EP coincides with it.
        ep_z = 0.0
        ep_radius = stop_radius
    else:
        pre = [_surface_copy_with(s) for s in surfaces[:stop_index]]
        # Append the propagation leg to the stop as a trailing
        # thickness on the last pre-surface.  system_abcd walks
        # thicknesses between surfaces, so we need to insert an extra
        # "transfer only" leg.  Easiest: append a dummy flat air
        # surface at the stop vertex (zero power, just for the
        # transfer) -- its ABCD contribution is identity; the
        # thickness accumulates from pre[-1].thickness which is the
        # pre->stop gap.
        # Actually the pre-stop sub-system already walks
        # thicknesses[0..stop_index-1] which includes the gap from
        # surface stop_index-1 to stop (since s.thickness = distance
        # to NEXT surface).  So no dummy needed; system_abcd(pre)
        # already lands the ray at the stop vertex.
        M_pre, _, _, _ = system_abcd(pre, wavelength)
        A_pre, B_pre = float(M_pre[0, 0]), float(M_pre[0, 1])
        _C_pre, _D_pre = float(M_pre[1, 0]), float(M_pre[1, 1])
        # Imaging condition: prepend T(z_obj) so B_total = 0.
        # B + z * A = 0 ?  No: T(z) applied on the RIGHT gives
        # M . T(z) = [[A, A*z+B], [C, C*z+D]].  So B_total = A*z + B.
        # z_ep = -B / A (distance from surface 0 back to EP).
        # Magnification through pre: m_pre = A_pre when B=0.
        if abs(A_pre) > 1e-30:
            -B_pre / A_pre
            # Radius: EP is the reverse image of the stop with
            # magnification 1/A_pre (because the forward sub-system
            # maps object height to stop height with factor A when
            # B=0 -> object height = stop / A).
            ep_radius = abs(stop_radius / A_pre) if np.isfinite(stop_radius) else float('nan')
        else:
            float('inf')
            ep_radius = float('inf')

    # ---- Exit pupil -----------------------------------------------
    if stop_index == len(surfaces) - 1:
        xp_z = 0.0
        xp_radius = stop_radius
    else:
        post = [_surface_copy_with(s) for s in surfaces[stop_index + 1:]]
        # Propagation from stop to first post-surface is the
        # thickness attribute on surfaces[stop_index], which lives on
        # the stop surface itself.  To include it in the post
        # sub-system we prepend a dummy air surface with that
        # thickness.  Cleanest: pass a fake Surface at stop_index with
        # zero power and the correct thickness.
        stop_to_first_post = surfaces[stop_index].thickness
        dummy = Surface(
            radius=np.inf, conic=0.0,
            semi_diameter=np.inf,
            glass_before='air', glass_after='air',
            is_mirror=False, is_stop=False,
            thickness=stop_to_first_post,
            label='(stop->XP transfer)')
        post_full = [dummy] + post
        M_post, _, _, _ = system_abcd(post_full, wavelength)
        A_post, B_post = float(M_post[0, 0]), float(M_post[0, 1])
        C_post, D_post = float(M_post[1, 0]), float(M_post[1, 1])
        # XP is the image-space conjugate of the stop.  Append
        # T(z_img) on the LEFT (image side) so B_total = 0:
        # T(z) . M = [[A + z*C, B + z*D], [C, D]].  B_new = B + z*D = 0
        # => z_xp = -B_post / D_post.
        if abs(D_post) > 1e-30:
            xp_z = -B_post / D_post
            # After prepending T(z_xp) on the image side to enforce
            # B+z·D = 0, the new matrix is [[A+z_xp·C, 0], [C, D]].  Its
            # transverse magnification at imaging is m = det(M)/D =
            # (AD−BC)/D = 1/D for air-to-air systems (det M = 1).  Pre-
            # 4.10 used `stop_radius * D_post` (the angular magnification,
            # not transverse) — every XP-radius downstream consumer
            # (vignetting, f/#, Seidel) was wrong by 1/D² for non-trivial
            # post-stop systems.
            det_post = float(A_post * D_post - B_post * C_post)
            xp_radius = (abs(det_post * stop_radius / D_post)
                         if np.isfinite(stop_radius) else float('nan'))
        else:
            xp_z = float('inf')
            xp_radius = float('inf')

    return PupilInfo(
        ep_z=float(ep_z), ep_radius=float(ep_radius),
        xp_z=float(xp_z), xp_radius=float(xp_radius),
        stop_index=int(stop_index),
    )


def find_lenses(
    surfaces: List['Surface'],
    wavelength: float,
) -> List['LensInfo']:
    """Auto-detect individual lens elements in a surface list.

    Scans for air -> glass -> air blocks; each block becomes one
    ``LensInfo``.  Cemented multi-element lenses (glass -> glass
    interfaces in the middle) stay grouped.  Mirrors are treated
    as their own single-surface elements.  Air-only runs (gaps
    between lenses, dummy COORDBRK surfaces) are skipped.

    Limitations
    -----------
    * Pure air -> air surfaces (DOE phase masks represented as
      air-to-air elements, COORDBRK carriers, dummy reference
      planes) are not detected as "lenses" and are silently
      skipped.  That's usually the right thing -- they contribute
      no power -- but a phase-grating element imparts real optical
      power at non-zero diffraction orders that ``find_lenses``
      won't see.
    * A system that ends inside glass (last surface's
      ``glass_after != 'air'``) is malformed for this purpose.  The
      straggling partial block is omitted from the result.
    """
    lenses = []
    n_surf = len(surfaces)
    i = 0
    while i < n_surf:
        s = surfaces[i]
        n_b = get_glass_index(s.glass_before, wavelength)
        n_a = get_glass_index(s.glass_after, wavelength)
        # Detect air -> glass transition (entry of a lens element)
        if abs(n_b - 1.0) < 1e-6 and abs(n_a - 1.0) > 1e-6:
            start = i
            j = i
            end = None
            while j < n_surf:
                sj = surfaces[j]
                nb_j = get_glass_index(sj.glass_before, wavelength)
                na_j = get_glass_index(sj.glass_after, wavelength)
                if abs(nb_j - 1.0) > 1e-6 and abs(na_j - 1.0) < 1e-6:
                    # glass -> air (exit of lens block)
                    end = j
                    break
                j += 1
            if end is None:
                # Malformed: entered glass but never exited.  Skip.
                break
            lenses.append(lens_abcd(surfaces, wavelength,
                                     start=start, end=end))
            i = end + 1
        # Handle a free-standing mirror as its own element
        elif s.is_mirror:
            lenses.append(lens_abcd(surfaces, wavelength, start=i, end=i))
            i += 1
        else:
            i += 1
    return lenses


# ============================================================================
# Seidel aberration coefficients
# ============================================================================

def seidel_coefficients(
    surfaces: List['Surface'],
    wavelength: float,
    object_distance: float = np.inf,
    stop_index: Optional[int] = None,
    field_angle: float = 0.01,
    *,
    field_angle_deg: Optional[float] = None,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute the five Seidel (third-order) aberration coefficients.

    Uses the Buchdahl-Hopkins formulation based on paraxial marginal
    and chief ray data at each surface.  **Stop-aware** (3.1.11): the
    chief ray is constrained to pass through the centre of the
    declared aperture stop (``y_c = 0`` at the stop surface), and the
    marginal ray fills the stop (``y_m = r_stop`` at the stop).  The
    initial conditions at surface 0 are derived from the pre-stop
    ABCD so both ray constraints are satisfied automatically.

    When the stop is at surface 0 (the legacy assumption), behaviour
    is bit-for-bit backward compatible with 3.1.10.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float
        Vacuum wavelength [m].
    object_distance : float
        Object distance from the first surface [m].  ``np.inf`` for
        an object at infinity (collimated input).
    stop_index : int, optional
        Explicit stop surface index.  Defaults to :func:`find_stop`
        -- i.e. the surface flagged ``is_stop=True``, or the first
        surface with a finite ``semi_diameter``, or 0.  When the
        resolved stop is somewhere other than surface 0, the chief
        ray initial conditions are back-propagated through the
        pre-stop ABCD so that ``y_c = 0`` at the stop.
    field_angle : float, default 0.01
        Unreduced field half-angle [rad] for the chief ray.  Only
        the shape of the Seidel sums is reported; absolute
        magnitudes scale linearly with this value (and quadratically
        for astigmatism/Petzval).  0.01 rad (~0.57 deg) is the
        conventional small-angle normalisation.
    field_angle_deg : float, optional
        Same as ``field_angle`` but expressed in degrees.  When
        provided, takes precedence over the radian form.  4.7+ the
        library is converging on ``_deg`` as the canonical
        user-facing angle unit.

    Returns
    -------
    seidel : dict
        Keys: ``'S1'`` (spherical), ``'S2'`` (coma), ``'S3'``
        (astigmatism), ``'S4'`` (Petzval), ``'S5'`` (distortion).
        Each value is a 1-D per-surface array.  Also contains:

        * ``'total'`` : dict with the sums.
        * ``'labels'`` : dict with human-readable names.
        * ``'y_marginal'`` / ``'y_chief'`` : per-surface ray heights.
        * ``'stop_index'`` : the stop index used (for diagnostics).
    abcd : ndarray
        System ABCD matrix.
    """
    if field_angle_deg is not None:
        field_angle = float(np.radians(field_angle_deg))
    n_surf = len(surfaces)
    n_first = get_glass_index(surfaces[0].glass_before, wavelength)

    # ---- Resolve the stop surface and its radius ------------------
    if stop_index is None:
        stop_index = find_stop(surfaces)
    if not (0 <= stop_index < n_surf):
        raise ValueError(
            f"seidel_coefficients: stop_index={stop_index} out of "
            f"range [0, {n_surf})")
    r_stop = surfaces[stop_index].semi_diameter
    if not np.isfinite(r_stop):
        # No explicit stop radius declared; fall back to surface 0's
        # semi-diameter for normalisation (legacy behaviour).  The
        # absolute magnitude of the Seidel sums then depends on this
        # default; callers who care should declare a proper stop.
        r_stop = surfaces[0].semi_diameter
        if not np.isfinite(r_stop):
            r_stop = 12.7e-3  # last-resort fallback

    # ---- Pre-stop ABCD (surface 0 -> stop vertex) -----------------
    # system_abcd walks surfaces but only applies the transfer matrix
    # between SURFACES (``if i < len(surfaces) - 1``).  For the
    # pre-stop sub-system we need one additional transfer: from the
    # last pre-stop surface's vertex to the stop's vertex, using the
    # stop-ward glass index.  Build it explicitly.
    if stop_index == 0:
        A_pre, B_pre = 1.0, 0.0
    else:
        M_pre, _, _, _ = system_abcd(surfaces[:stop_index], wavelength)
        # Transfer from surface (stop_index-1) to the stop vertex in
        # the medium on its image side.
        t_last = float(surfaces[stop_index - 1].thickness)
        n_last = get_glass_index(
            surfaces[stop_index - 1].glass_after, wavelength)
        T_last = np.array([[1.0, t_last / n_last],
                            [0.0, 1.0]])
        M_pre = T_last @ M_pre
        A_pre = float(M_pre[0, 0])
        B_pre = float(M_pre[0, 1])

    # ---- Initial conditions at surface 0 --------------------------
    # Marginal ray: on-axis object (u_0 = 0, nu_0 = 0), filling the
    # stop (y_stop = r_stop).  In reduced-coord matrix form:
    #    y_stop = A_pre * y_0 + B_pre * 0  =>  y_0 = r_stop / A_pre.
    # For a finite object the marginal ray is launched from the
    # axial object point; we keep the legacy u-driven form in that
    # case (stop-awareness is a collimated-input concept primarily).
    if np.isinf(object_distance):
        y_m_init = (r_stop / A_pre) if abs(A_pre) > 1e-30 else r_stop
        nu_m_init = 0.0
    else:
        # Finite-conjugate marginal ray: launches from on-axis object
        # point at z = -object_distance with some slope u_obj such that
        # the ray fills the stop after traversing T(d) and M_pre.  In
        # reduced coords T(d) is [[1, d/n_first],[0,1]], so
        #   stop_y = (M_pre @ T(d) @ [0, n_first u_obj]^T)_y
        #          = u_obj * (A_pre*d + B_pre*n_first)
        # which we set equal to r_stop.  Pre-4.10 this branch hard-coded
        # y_m_init = 0, which made the Lagrange invariant H identically
        # zero for any finite-conjugate stop-at-front system (since
        # y_c_init is also 0 at a front-stop), zeroing the Petzval
        # contribution to seidel_wfe.
        lever = A_pre * object_distance + B_pre * n_first
        if abs(lever) > 1e-30:
            u_obj = r_stop / lever
        else:
            u_obj = r_stop / object_distance
        y_m_init = u_obj * object_distance
        nu_m_init = n_first * u_obj

    # Chief ray: from edge of field (angle = field_angle), through
    # centre of stop (y_stop = 0).  system_abcd works in reduced
    # coordinates (y, nu) where nu = n*u, so the transfer is
    #    y_stop = A_pre * y_0 + B_pre * nu_0
    # Setting y_stop = 0 with nu_0 = n_first * field_angle gives
    #    y_0 = -B_pre * nu_0 / A_pre = -B_pre * n_first * field_angle / A_pre.
    u_0_c = float(field_angle)
    nu_c_init = n_first * u_0_c
    if abs(A_pre) > 1e-30:
        y_c_init = -B_pre * nu_c_init / A_pre
    else:
        y_c_init = 0.0

    y_m = np.zeros(n_surf)
    nu_m = np.zeros(n_surf)  # n*u product

    y_c = np.zeros(n_surf)
    nu_c = np.zeros(n_surf)  # chief ray

    # Trace marginal and chief rays
    y_val_m = y_m_init
    nu_val_m = nu_m_init
    y_val_c = y_c_init
    nu_val_c = nu_c_init

    # Lagrange invariant H = y_c · nu_m - y_m · nu_c (system-wide
    # constant).  Computed up-front so the S5 Schwarzschild relation
    # below can fold in the H² that S_IV requires for unit consistency
    # with S_III (audit fix #4.7: pre-4.9 added S3 [m] to S4 [1/m]
    # directly, mixing dimensions and inheriting both bugs).
    H_lagrange = float(y_c_init * nu_m_init - y_m_init * nu_c_init)
    H_sq = H_lagrange ** 2

    # Per-surface Seidel contributions
    S1 = np.zeros(n_surf)
    S2 = np.zeros(n_surf)
    S3 = np.zeros(n_surf)
    S4 = np.zeros(n_surf)
    S5 = np.zeros(n_surf)

    # 4.11.2: Track Welford mirror parity so multi-mirror systems
    # carry the correct effective index sign through every surface
    # downstream of an odd number of mirrors.  Pre-4.11.2 the v4.10
    # mirror fix only applied ``n2 = -n1`` at the mirror surface
    # itself; the next surface re-queried ``glass_before='air'`` and
    # got ``n=+1`` instead of ``n=-1``, producing wrong Seidel sums
    # beyond the first mirror in any catadioptric / Cassegrain
    # / Schwarzschild design.
    mirror_parity = 0  # 0 = unflipped, 1 = post-odd-mirror (n -> -n)

    for i, surf in enumerate(surfaces):
        sign = 1.0 if mirror_parity == 0 else -1.0
        # 3.7.0: Coord-breaks contribute no Seidel power; transfer
        # the rays through their air-gap thickness and skip.
        if surf.is_coordbrk:
            y_m[i] = y_val_m
            nu_m[i] = nu_val_m
            y_c[i] = y_val_c
            nu_c[i] = nu_val_c
            if i < n_surf - 1:
                t = float(surf.thickness)
                n_after = sign * get_glass_index(surf.glass_after, wavelength)
                if abs(n_after) > 0:
                    y_val_m = y_val_m + (nu_val_m / n_after) * t
                    y_val_c = y_val_c + (nu_val_c / n_after) * t
            continue

        n1 = sign * get_glass_index(surf.glass_before, wavelength)
        n2 = sign * get_glass_index(surf.glass_after, wavelength)
        R = surf.radius

        # Store ray heights at this surface
        y_m[i] = y_val_m
        nu_m[i] = nu_val_m
        y_c[i] = y_val_c
        nu_c[i] = nu_val_c

        if np.isfinite(R) and not surf.is_mirror:
            c = 1.0 / R  # curvature

            # Incidence angle (paraxial): i = c*y + u = c*y + nu/n
            u_m = nu_val_m / n1
            u_c = nu_val_c / n1

            i_m = c * y_val_m + u_m
            i_c = c * y_val_c + u_c

            # Refract
            nu_m_after = nu_val_m - y_val_m * (n2 - n1) * c
            nu_c_after = nu_val_c - y_val_c * (n2 - n1) * c

            u_m_after = nu_m_after / n2
            nu_c_after / n2

            # Abbe invariant
            A_m = n1 * i_m  # = n2 * i_m_after (Snell)
            A_c = n1 * i_c

            # Welford / Hopkins per-surface Seidel sums (the Lagrange
            # invariant H is factored out of the sum by convention, so
            # each term carries A_m / A_c factors but no separate H
            # multiplier).  Pre-4.9 used ``delta_un = 1/n2 - 1/n1``
            # which is ``Δ(1/n)``, not ``Δ(u/n)`` -- the two differ by
            # a factor that depends on the incidence angle and index
            # (the multiplier ``(n1*i_before + u_before*n2)/n2``), so
            # the buggy code reported magnitudes off by anywhere from
            # 1.5× to >5× depending on surface geometry.  The fix
            # below uses the standard Δ(u/n) = u_after/n_after −
            # u_before/n_before formulation; cross-check against
            # Welford ch. 8 Eqs. (8.46)–(8.50).
            h = y_val_m
            delta_un = (u_m_after / n2) - (u_m / n1)

            S1[i] = -(A_m ** 2) * h * delta_un
            S2[i] = -(A_m * A_c) * h * delta_un
            S3[i] = -(A_c ** 2) * h * delta_un
            S4[i] = -(1.0 / (n2 * n1)) * c * (n2 - n1)
            # S_V Schwarzschild: S_V_k = -(A_c/A) · (S_III_k + H²·S_IV_k).
            # Pre-4.9 omitted H² on S4, mixing m and 1/m dimensions.
            S5[i] = -(A_c / A_m) * (S3[i] + H_sq * S4[i]) if abs(A_m) > 1e-30 else 0.0

            nu_val_m = nu_m_after
            nu_val_c = nu_c_after

        elif surf.is_mirror and np.isfinite(R):
            # Mirror in the Welford paraxial convention: treat as a
            # refracting surface with n2 = -n1, so the same Welford
            # Seidel sums apply.  Pre-4.10 the mirror branch only
            # updated ray heights and never wrote S1..S5 -- every
            # catadioptric / reflective design silently reported zero
            # spherical, coma, astigmatism, Petzval, distortion.
            #
            # 4.11.2: ``n1`` already carries the running mirror-parity
            # sign (from the ``sign`` multiplier above), so ``n2 = -n1``
            # composes correctly across chained mirrors -- after two
            # mirrors n1 returns to +1 (Cassegrain final leg), after
            # three it's -1, etc.
            c = 1.0 / R
            n2 = -n1
            u_m = nu_val_m / n1
            u_c = nu_val_c / n1

            i_m = c * y_val_m + u_m
            i_c = c * y_val_c + u_c

            # Refract: phi = (n2 - n1) c = -2 n1 c
            nu_m_after = nu_val_m - y_val_m * (n2 - n1) * c
            nu_c_after = nu_val_c - y_val_c * (n2 - n1) * c

            u_m_after = nu_m_after / n2
            nu_c_after / n2

            A_m = n1 * i_m   # = n2 * i_after (Snell)
            A_c = n1 * i_c

            h = y_val_m
            delta_un = (u_m_after / n2) - (u_m / n1)

            S1[i] = -(A_m ** 2) * h * delta_un
            S2[i] = -(A_m * A_c) * h * delta_un
            S3[i] = -(A_c ** 2) * h * delta_un
            S4[i] = -(1.0 / (n2 * n1)) * c * (n2 - n1)
            S5[i] = -(A_c / A_m) * (S3[i] + H_sq * S4[i]) if abs(A_m) > 1e-30 else 0.0

            nu_val_m = nu_m_after
            nu_val_c = nu_c_after

            # Flip parity for the post-mirror legs so subsequent
            # ``glass_before``/``glass_after`` lookups (which return
            # positive indices) get sign-corrected via ``sign``.
            mirror_parity ^= 1
        else:
            # Flat refracting surface: c=0 but Δ(u/n) is still nonzero
            # for non-normal incidence (Snell's law: n1·u_m = n2·u_m_after,
            # so u_after/n2 = n1·u_m/n2² which is ≠ u_m/n1 unless n1=n2).
            # Pre-4.9 zeroed S1/S2/S3 here -- but a flat surface inside
            # a stack contributes to spherical / coma / astigmatism
            # exactly as the audit's plano-convex hand calc showed:
            # the R2=∞ surface of a plano-convex singlet has a real
            # S1 contribution that the old branch dropped silently.
            # Compute the full S1..S5 here too, with c=0 baked in.
            u_m = nu_val_m / n1
            u_c = nu_val_c / n1
            i_m = u_m         # c=0, so i = u
            i_c = u_c

            # Refract at flat surface: Snell in the paraxial limit gives
            # n1·u_before = n2·u_after, so n·u is invariant.  Curvature
            # shift is zero (c=0); only u changes (= nu/n with new n2).
            nu_m_after = nu_val_m
            nu_c_after = nu_val_c
            u_m_after = nu_m_after / n2
            nu_c_after / n2

            A_m = n1 * i_m
            A_c = n1 * i_c
            h = y_val_m
            delta_un = (u_m_after / n2) - (u_m / n1)

            S1[i] = -(A_m ** 2) * h * delta_un
            S2[i] = -(A_m * A_c) * h * delta_un
            S3[i] = -(A_c ** 2) * h * delta_un
            S4[i] = 0.0           # flat: Petzval contribution exactly zero (c=0)
            # S_V Schwarzschild: S_V_k = -(A_c/A) · (S_III_k + H²·S_IV_k).
            # Pre-4.9 omitted H² on S4, mixing m and 1/m dimensions.
            S5[i] = -(A_c / A_m) * (S3[i] + H_sq * S4[i]) if abs(A_m) > 1e-30 else 0.0

            nu_val_m = nu_m_after
            nu_val_c = nu_c_after

        # Transfer to next surface
        if i < len(surfaces) - 1:
            t = surf.thickness
            n_after = n2
            u_m_t = nu_val_m / n_after
            u_c_t = nu_val_c / n_after
            y_val_m = y_val_m + u_m_t * t
            y_val_c = y_val_c + u_c_t * t

    abcd, _, _, _ = system_abcd(surfaces, wavelength)

    return {
        'S1': S1, 'S2': S2, 'S3': S3, 'S4': S4, 'S5': S5,
        'total': {
            'S1': np.sum(S1), 'S2': np.sum(S2), 'S3': np.sum(S3),
            'S4': np.sum(S4), 'S5': np.sum(S5),
        },
        'labels': {
            'S1': 'Spherical', 'S2': 'Coma', 'S3': 'Astigmatism',
            'S4': 'Petzval', 'S5': 'Distortion',
        },
        'y_marginal': y_m,
        'y_chief': y_c,
        'stop_index': stop_index,
        # 4.3.0: expose the field_angle used so seidel_wfe can apply
        # the correct scaling to the Hopkins-S_IV (Petzval) term
        # when reconstructing the wavefront expansion.
        'field_angle': float(field_angle),
        # 4.9 fix: explicit Lagrange invariant for the seidel_wfe
        # Petzval (S4·H²·ρ²) and distortion (S5·H³·ρ·cos) terms.
        'lagrange_invariant': H_lagrange,
    }, abcd


def seidel_prescription(
    prescription: Dict[str, Any],
    wavelength: float,
    object_distance: float = np.inf,
    stop_index: Optional[int] = None,
    field_angle: float = 0.01,
) -> Tuple[Dict[str, Any], np.ndarray]:
    """Compute Seidel coefficients from a lens prescription dict.

    Passes ``stop_index`` and ``field_angle`` through to
    :func:`seidel_coefficients`; the prescription dict's own
    ``stop_index`` key (used by the wave-optics pipeline) is
    propagated onto the surface list automatically via
    :func:`surfaces_from_prescription` and picked up by
    :func:`find_stop` when ``stop_index`` is left ``None`` here.
    """
    surfaces = surfaces_from_prescription(prescription)
    return seidel_coefficients(
        surfaces, wavelength, object_distance=object_distance,
        stop_index=stop_index, field_angle=field_angle)


# ============================================================================
# Analysis: find paraxial focus
# ============================================================================

def find_paraxial_focus(
    surfaces: List['Surface'],
    wavelength: float,
) -> float:
    """Find the paraxial image distance from the last surface.

    Parameters
    ----------
    surfaces : list of Surface
    wavelength : float

    Returns
    -------
    image_distance : float
        Axial distance from the last surface vertex to paraxial focus [m].
    """
    _, _, bfl, _ = system_abcd(surfaces, wavelength)
    return bfl


__all__ = [
    # Private paraxial helpers (kept underscore-private to signal
    # the API contract: callers should not rely on the exact form
    # of the paraxial recursion -- ``system_abcd`` is the supported
    # entry point).
    '_paraxial_trace', '_paraxial_refract', '_paraxial_transfer',
    # ABCD
    'system_abcd', 'system_abcd_prescription',
    # Per-lens characterisation
    'LensInfo', 'lens_abcd', 'find_lenses',
    # Pupils / first-order
    'PupilInfo', 'FirstOrderData', 'first_order_data', 'compute_pupils',
    # Seidel
    'seidel_coefficients', 'seidel_prescription',
    # Paraxial focus
    'find_paraxial_focus',
]
