"""
lumenairy.optimize.merit_terms -- individual merit-term classes.

v5.1.0 split (Agent E): extracted from ``lumenairy/optimize/core.py``.
Hosts the leaf-level merit-term classes consumed by ``design_optimize``:
focal-length / Strehl / wavefront / spot-size / OPD / Zernike /
Laguerre-Gauss aberration penalties, the geometric-constraint merits
(min/max thickness, min BFL, max f/#), and the chromatic focal-shift
merit.  Composite + Callable merit terms also live here for cohesion.

Wrapper merits (``MultiWavelengthMerit`` / ``MultiFieldMerit`` /
``ToleranceAwareMerit``) and JAX merits (``JaxMeritTerm`` /
``make_lg_aberration_merit_jax``) live in their own submodules.  All
public names are re-exported from ``optimize/core.py``; no public-API
behaviour changes.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from .context import MeritTerm, ctx_is_valid

# =========================================================================
# Geometric merits (fast, no wave leg)
# =========================================================================

class FocalLengthMerit(MeritTerm):
    """Penalise deviation from target focal length.

    ``contribution = weight * (efl - target)^2 / target^2``

    When ``target == 0`` (afocal / collimator), the normalised-error
    formula is ill-defined; the merit falls back to a penalty on the
    *optical power* ``1/efl`` (in m^-1, equivalently dioptres).  This
    drives the merit to zero as the system becomes truly afocal
    (``efl -> infinity``) and grows without bound as ``efl -> 0``
    (point-image collapse), which is the right gradient direction for
    a collimator design.

    Use ``target > 0`` for a finite-EFL target; use ``target == 0``
    for an afocal / collimator target.  There is no use case for
    ``target < 0`` (a virtual-image lens) -- pass the positive EFL
    instead and rely on ABCD sign conventions in the prescription.
    """

    needs_wave = False
    name = 'FocalLength'

    def __init__(self, target: float, weight: float = 1.0) -> None:
        self.target = float(target)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        efl = getattr(ctx, 'efl', float('nan'))
        if not ctx_is_valid(ctx, 'efl'):
            return self.weight  # graceful large-but-finite penalty
        if self.target == 0.0:
            # Afocal / collimator: minimise 1/|efl| so merit -> 0 as
            # efl -> infinity.  Clamp efl below to keep the merit
            # finite when the optimiser walks through near-zero EFL
            # during the search.
            efl_clamped = max(abs(efl), 1e-12)
            power = 1.0 / efl_clamped  # dioptres
            return self.weight * power * power
        err = (efl - self.target) / self.target
        return self.weight * err * err


class BackFocalLengthMerit(MeritTerm):
    """Penalise deviation from target back focal length.

    Same zero-target behaviour as :class:`FocalLengthMerit`: BFL ->
    infinity drives the merit toward zero, BFL -> 0 explodes it.
    """

    needs_wave = False
    name = 'BackFocalLength'

    def __init__(self, target: float, weight: float = 1.0) -> None:
        self.target = float(target)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        bfl = getattr(ctx, 'bfl', float('nan'))
        if not ctx_is_valid(ctx, 'bfl'):
            return self.weight
        if self.target == 0.0:
            bfl_clamped = max(abs(bfl), 1e-12)
            power = 1.0 / bfl_clamped
            return self.weight * power * power
        err = (bfl - self.target) / self.target
        return self.weight * err * err


class SphericalSeidelMerit(MeritTerm):
    """Minimise Seidel spherical aberration coefficient S_I.

    Fast, geometric-only term.
    """

    needs_wave = False
    name = 'SphericalSeidel'

    def __init__(self, weight: float = 1.0) -> None:
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        # 4.10.2: guard against the NaN-filled sentinel that
        # aberration_summary returns on failed Seidel computation
        # (4.10.1).  Pre-4.10.2 read ctx.seidel[0] blindly and
        # returned NaN, which scipy.optimize then refused as an
        # invalid objective.  Treat invalid Seidel as a moderate
        # penalty so the optimiser sees a finite value and can
        # step away from the bad region.
        s = ctx.seidel
        if s is None or len(s) == 0 or not np.isfinite(s[0]):
            return self.weight
        return self.weight * float(s[0]) ** 2


class StrehlMerit(MeritTerm):
    """Penalise Strehl ratio below ``min_strehl``.

    ``contribution = weight * max(0, min_strehl - best_strehl)^2``
    """

    needs_wave = True
    name = 'Strehl'

    def __init__(self, min_strehl: float = 0.8, weight: float = 1.0) -> None:
        self.min_strehl = float(min_strehl)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        # v4.15.3 (P1-NEW-F1-3): coerce ``ctx.strehl_best`` via
        # ``float()`` so the
        # ``_FAILED_SCAN_STREHL_SENTINEL_OBJ`` singleton written by
        # ``MultiFieldMerit`` / ``ToleranceAwareMerit`` collapses to
        # its scalar fallback (0.0) for arithmetic.  Identity-check
        # is available before the cast for callers that want to
        # distinguish a real-zero Strehl from a failed-scan zero.
        strehl_best = float(ctx.strehl_best)
        deficit = max(0.0, self.min_strehl - strehl_best)
        return self.weight * deficit * deficit


class RMSWavefrontMerit(MeritTerm):
    """Penalise RMS wavefront error above a target (waves).

    Uses Zernike decomposition to exclude the first
    ``exclude_low_order`` OSA/ANSI-ordered modes.

    v5.4.6 (audit F-4): the default ``exclude_low_order=4`` drops OSA
    indices [0:4] = piston + 2 tilts + OBLIQUE ASTIGMATISM, and KEEPS
    defocus (OSA index 4).  In OSA/ANSI order the low modes are
    0=piston, 1=tilt-Y, 2=tilt-X, 3=oblique-astig, 4=defocus,
    5=vertical-astig, so contiguous slicing of ``coeffs[N:]`` cannot
    remove defocus while keeping the two astigmatism orientations:
    defocus (index 4) sits between them.  Removing defocus by raising
    the cut to 5 would also remove oblique astigmatism, and removing
    BOTH astigmatism orientations needs index 6.  A genuinely
    defocus-insensitive, astigmatism-sensitive RMS therefore requires a
    future (n, m)-explicit mode exclusion rather than a leading-count
    slice; that is the real defocus-insensitive fix.  ``exclude_low_order``
    is documented here as a leading-OSA-index count, not a
    'piston/tilt/defocus' selector.
    """

    needs_wave = True
    name = 'RMSWavefront'

    def __init__(self, max_rms_waves: float = 0.07,
                 n_modes: int = 21,
                 exclude_low_order: int = 4,
                 weight: float = 1.0) -> None:
        self.max_rms_waves = float(max_rms_waves)
        self.n_modes = int(n_modes)
        self.exclude_low_order = int(exclude_low_order)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        rms_waves = ctx.rms_wavefront_waves(
            n_modes=self.n_modes,
            exclude_low_order=self.exclude_low_order)
        excess = max(0.0, rms_waves - self.max_rms_waves)
        return self.weight * excess * excess


class SpotSizeMerit(MeritTerm):
    """Penalise RMS spot radius at best focus above a target."""

    needs_wave = True
    name = 'SpotSize'

    def __init__(self, max_rms_radius: float, weight: float = 1.0) -> None:
        self.max_rms_radius = float(max_rms_radius)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        r = ctx.rms_radius_best
        # S4-5 (AUDIT_V5_24_2): ``rms_radius_best`` defaults to
        # ``np.inf`` and STAYS inf when the through-focus scan is
        # skipped (|BFL| out of range early-return in the driver) or
        # produced no finite slice.  A raw ``inf`` excess injects
        # ``inf`` into the scipy merit sum and stalls the L-BFGS-B
        # line search.  Coerce a non-finite radius to a large-but-
        # FINITE quadratic penalty (``excess = 10 * max_rms_radius``)
        # so a failed wave leg reads as a very-bad-but-usable design
        # rather than crashing the optimiser.
        if not np.isfinite(r):
            excess = 10.0 * self.max_rms_radius
            return self.weight * excess * excess
        excess = max(0.0, r - self.max_rms_radius)
        return self.weight * excess * excess


class MatchIdealThinLensMerit(MeritTerm):
    """Penalise deviation of the actual exit-pupil OPD from an
    idealised thin-lens wavefront with the same target focal length.

    This is the merit term you want when you're asking "make this real
    lens behave as much like an ideal thin lens of focal length f as
    possible".  At each evaluation:

    1. The actual exit-pupil OPD is extracted from the wave-optics
       output (using `wave_opd_2d`, with reference-sphere subtraction
       for numerical stability).
    2. An ideal thin-lens OPD ``OPD_ideal(r) = -r^2 / (2*f_target)`` is
       computed on the same grid.
    3. Their difference -- the *aberration wavefront* -- is masked to
       the pupil and its RMS computed.
    4. ``contribution = weight * RMS_diff^2 / wavelength^2`` (in
       wavelength-squared units, so weights of 1.0 produce
       ``waves^2`` of penalty).

    The result of optimizing against this merit is a wavefront whose
    departure from a perfect spherical converging wave is minimised
    -- which is the formal definition of "diffraction-limited" up to
    a tolerable RMS wavefront error.

    Parameters
    ----------
    target_focal_length : float
        Focal length [m] of the ideal thin lens to match.
    weight : float, default 1.0
    exclude_low_order : int, default 1
        Number of Zernike modes to exclude from the RMS (default 1 =
        piston only).  Set to 3 to also exclude tilts (so a lateral
        decenter doesn't dominate the merit), or 4 to also exclude
        defocus (so the merit becomes "match the ideal except for an
        arbitrary focus shift").
    n_modes : int, default 21
        How many Zernike modes the OPD is decomposed into to compute
        the high-order RMS.
    """

    needs_wave = True
    name = 'MatchIdealThinLens'

    def __init__(self, target_focal_length: float, weight: float = 1.0,
                 exclude_low_order: int = 1, n_modes: int = 21) -> None:
        self.target_focal_length = float(target_focal_length)
        self.weight = float(weight)
        self.exclude_low_order = int(exclude_low_order)
        self.n_modes = int(n_modes)

    def evaluate(self, ctx: Any) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        # Compute the ideal thin-lens OPD on the pupil grid
        Ny, Nx = ctx.opd_map.shape
        x = (np.arange(Nx) - Nx / 2) * ctx.dx
        y = (np.arange(Ny) - Ny / 2) * ctx.dx
        X, Y = np.meshgrid(x, y)
        opd_ideal = -(X ** 2 + Y ** 2) / (2.0 * self.target_focal_length)
        diff = ctx.opd_map - opd_ideal
        # Decompose into Zernikes; high-order RMS = aberration RMS
        # (excluding the requested low-order modes which represent
        # piston / tilt / defocus -- usually NOT what you want to
        # penalise unless you've fixed alignment).
        from ..analysis import zernike_decompose as _zd
        finite = np.isfinite(diff)
        # Replace NaN with 0 in the masked region; decompose handles it
        diff_clean = np.where(finite, diff, 0.0)
        try:
            coeffs, _ = _zd(diff_clean, ctx.dx, ap, n_modes=self.n_modes)
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                ZeroDivisionError):
            # zernike_decompose can fail on under-sampled pupils
            # (ValueError) or singular bases (LinAlgError); the
            # convention is to return a 0 merit so the optimizer
            # doesn't get derailed.
            return 0.0
        higher = coeffs[self.exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        rms_waves = rms_m / ctx.wavelength
        return self.weight * rms_waves * rms_waves


# =========================================================================
# Full-system "match this ideal thin-lens architecture" merit
# =========================================================================

class MatchIdealSystemMerit(MeritTerm):
    """Match the real system's output field to that of an idealised
    thin-lens reference system.

    Unlike :class:`MatchIdealThinLensMerit` -- which operates on the
    exit-pupil OPD of a single lens and compares it to a bare
    converging sphere -- this merit propagates a reference source
    through BOTH an ideal thin-lens element list AND the real
    prescription (wrapped in an optional pre/post-propagation envelope)
    and then compares the resulting **complex output fields**.

    Use cases:

    * Replace a paraxial thin lens with a singlet / doublet / aspheric
      while preserving the output radiation pattern + relative phase.
    * Replace a Keplerian / Galilean telescope's thin-lens pair with
      real achromats.
    * Replace a 4f imaging system's two thin lenses with two real
      lenses, jointly optimised.
    * Any architecture expressible as
      ``propagate_through_system(source, ideal_elements)``.

    The merit is designed for the common situation where the real
    system is **slightly longer** than the ideal (because real lenses
    have nonzero thickness) but otherwise functionally equivalent.
    Approximately preserving aperture and inter-element distances is
    the user's responsibility; the merit will drive whatever free
    variables it can to make the output fields match.

    Parameters
    ----------
    ideal_elements : list of dict
        Element list for the ideal reference system, in the same
        format as :func:`propagate_through_system`.  Typically a mix
        of ``{'type': 'lens', 'f': ...}`` and
        ``{'type': 'propagate', 'z': ...}`` elements, optionally with
        apertures / mirrors / masks.
    real_elements : list of dict, optional
        Element list for the real system.  Dicts with
        ``type='_prescription_'`` are replaced at evaluation time
        with the current ``ctx.prescription`` wrapped as a
        ``'real_lens'`` element (or ``'real_lens_traced'`` if
        ``use_traced_lens=True``).  Default: a single-lens drop-in,
        ``[{'type': '_prescription_'}]``, which is correct when the
        ideal is a single thin lens + propagate pair and the real
        prescription replaces that thin lens.
    source_fn : callable or None
        Factory returning the complex input field at the first plane.
        Signature ``source_fn(N, dx, wavelength) -> ndarray``.  The
        merit uses ``ctx.N``, ``ctx.dx``, ``ctx.wavelength``.  Default
        (``None``) is a uniform plane wave of unit amplitude inside
        the prescription's aperture (if specified) or the full grid
        (otherwise).
    match : ``'field_overlap'`` | ``'field_mse'`` | ``'intensity_mse'`` | ``'intensity_overlap'``
        Similarity metric:

        * ``'field_overlap'`` (**default**, recommended): coupling
          efficiency
          ``|<E_ideal | E_real>|^2 / (||E_ideal||^2 ||E_real||^2)``.
          Bounded in [0, 1], invariant to a global phase and to an
          overall amplitude scaling.  Merit = ``weight * (1 - overlap)``;
          drops to zero when the real field differs from the ideal only
          by a global phase factor and overall amplitude.  **This is
          the right choice for "match the radiation pattern + relative
          phase"** as requested -- global phase is not a physical
          observable.
        * ``'field_mse'``: power-normalised + phase-aligned MSE of the
          field difference.  Merit is roughly the squared "fraction"
          of energy in the difference.  More sensitive to absolute
          phase shape than ``field_overlap``.
        * ``'intensity_mse'``: MSE of ``|E|^2``, phase-blind.  Use
          this when only the radiation pattern (not phase) matters
          (e.g. matching a target irradiance profile).
        * ``'intensity_overlap'``: correlation of ``|E|^2`` patterns,
          phase-blind.
    aperture_mask : ndarray or None
        Optional boolean / real mask applied to BOTH output fields
        before the comparison.  Use it to restrict the match to a
        region of interest (e.g. the intended image area) and avoid
        letting low-intensity grid edges dominate.
    use_traced_lens : bool, default False
        If True, propagate the real prescription via
        ``apply_real_lens_traced`` (sub-nm OPD agreement with the
        ray trace, 10-30x slower) rather than ``apply_real_lens``.
    ray_subsample : int, default 4
        Passed to ``apply_real_lens_traced`` when used.
    focus_search : bool, default False
        If True, scan a small range of axial offsets on the real
        system's output plane and report the BEST (lowest-penalty)
        match.  Decouples "correct focal plane" from "aberration
        quality" so a small BFL shift caused by real-lens thickness
        doesn't dominate the penalty.  Not valid for
        ``match='intensity_mse'`` (no unique optimum under
        translation); enable it with any of the other three metrics.
    focus_search_range : tuple (z_lo, z_hi) or None
        Axial-offset bracket for the focus search, relative to the
        nominal output plane [m].  Default (None): +/- f/20 computed
        from ``ctx.bfl`` or ``ctx.efl``, falling back to +/- 5 mm.
    focus_search_n : int, default 9
        Number of samples in the z-offset scan.
    wavelengths : list of float, optional
        If given, evaluate the merit at each wavelength and average
        the results.  Drives the glass-index dispersion through
        ``apply_real_lens`` + ``propagate_through_system``.  Useful
        for broadband / chromatic-matching optimisation without
        needing a separate ``MultiWavelengthMerit`` wrapper.
    field_angles : list of (theta_x, theta_y) tuples, optional
        Off-axis tilts (radians) to evaluate.  Each field angle adds
        a carrier phase to the source so the merit penalises the
        real lens's output for multiple input beam directions
        simultaneously.  Combines Cartesian-product-wise with
        ``wavelengths``.
    weight : float

    Notes
    -----
    * The ideal system is propagated with ``propagate_through_system``;
      each thin lens applies the paraxial phase screen
      ``exp(-i k r^2 / 2f)``.  This is exact in the small-angle limit
      and close to correct for f/2 or slower systems.  For systems
      with non-paraxial focusing, consider using
      ``{'type': 'lens', 'f': ..., 'lens_model': 'nonparaxial'}`` in
      ``ideal_elements``.
    * ``field_overlap`` is the physical "coupling efficiency" metric
      used in fiber / optical-mode matching.  It's bounded and
      dimensionless, which makes it numerically well-behaved for the
      optimizer and meaningful as an absolute number (0.99 = nearly
      perfect; 0.50 = significant mismatch).
    * Multi-lens architectures where two or more prescriptions are
      independently varied require multiple
      :class:`DesignParameterization` -- not yet supported by the
      single-template design.  Open a PR if you hit this.
    """

    needs_wave = True
    name = 'MatchIdealSystem'

    def __init__(self, ideal_elements: Sequence[Dict[str, Any]],
                 real_elements: Optional[Sequence[Dict[str, Any]]] = None,
                 source_fn: Optional[Callable] = None,
                 match: str = 'field_overlap',
                 aperture_mask: Optional[np.ndarray] = None,
                 use_traced_lens: bool = False,
                 ray_subsample: int = 4,
                 focus_search: bool = False,
                 focus_search_range: Optional[Tuple[float, float]] = None,
                 focus_search_n: int = 9,
                 wavelengths: Optional[Sequence[float]] = None,
                 field_angles: Optional[Sequence[float]] = None,
                 weight: float = 1.0) -> None:
        self.ideal_elements = list(ideal_elements)
        self.real_elements = (list(real_elements)
                              if real_elements is not None
                              else [{'type': '_prescription_'}])
        self.source_fn = source_fn
        self.match = str(match)
        self.aperture_mask = aperture_mask
        self.use_traced_lens = bool(use_traced_lens)
        self.ray_subsample = int(ray_subsample)
        self.weight = float(weight)
        self.focus_search = bool(focus_search)
        self.focus_search_range = focus_search_range
        self.focus_search_n = int(focus_search_n)
        # ``wavelengths`` and ``field_angles`` drive built-in sweeps
        # (averaged penalty).  Both default to None = single
        # wavelength / on-axis.
        self.wavelengths = (list(wavelengths)
                            if wavelengths is not None else None)
        self.field_angles = (list(field_angles)
                             if field_angles is not None else None)
        valid = ('field_overlap', 'field_mse',
                 'intensity_mse', 'intensity_overlap')
        if self.match not in valid:
            raise ValueError(
                f"match must be one of {valid}; got {self.match!r}")
        if self.focus_search and self.match not in (
                'field_overlap', 'field_mse', 'intensity_overlap'):
            raise ValueError(
                f"focus_search requires match in "
                f"('field_overlap', 'field_mse', 'intensity_overlap'); "
                f"got {self.match!r}.  intensity_mse doesn't have a "
                f"unique optimum under axial translation.")

    # -- Helpers -----------------------------------------------------

    def _make_source(self, ctx, wavelength, field_angle=(0.0, 0.0)):
        """Build the reference input field at the first plane.

        Parameters
        ----------
        ctx : EvaluationContext
            Provides ``N``, ``dx``, ``prescription`` (for default
            aperture clipping).
        wavelength : float
            Wavelength [m] used both by ``source_fn`` (if supplied)
            and for the field-angle carrier phase.
        field_angle : (float, float)
            Off-axis tilt ``(theta_x, theta_y)`` in radians.  A linear
            phase ``exp(i * k_x X + i * k_y Y)`` is applied on top of
            whatever source the factory produced, with
            ``k_x = 2 pi sin(theta_x) / wavelength``.
        """
        from ..propagators.propagation import get_default_complex_dtype
        cdtype = get_default_complex_dtype()  # 4.10.2: honour precision knob
        if self.source_fn is not None:
            E = self.source_fn(ctx.N, ctx.dx, wavelength)
            E = np.asarray(E, dtype=cdtype)
        else:
            # v4.14.2 (P1-NEW-1): three branches matching the canonical
            # _ZERO_APERTURE_MASK semantics shared by
            # ``MultiWavelengthMerit.evaluate``,
            # ``MultiFieldMerit.evaluate``, and
            # ``ToleranceAwareMerit.evaluate``:
            #   * ``ap`` finite and > 0   -> circular boolean mask.
            #   * ``ap`` finite and <= 0  -> deliberate-zero aperture;
            #     block all light by zeroing E entirely.  Pre-v4.14.2
            #     the ``ap > 0`` check silently fell through to the
            #     ``else`` branch and produced a full-grid plane wave,
            #     which apply_real_lens would then propagate as a
            #     bright on-axis "source" -- the exact bug v4.14.1
            #     fixed in the wrapper-merit cache but missed at this
            #     pre-existing site.
            #   * ``ap`` is None or non-finite -> no aperture specified;
            #     full-grid plane wave (unchanged behaviour).
            ap = (ctx.prescription.get('aperture_diameter')
                  if ctx.prescription else None)
            if ap is not None and np.isfinite(ap) and ap > 0:
                E = np.ones((ctx.N, ctx.N), dtype=cdtype)
                x = (np.arange(ctx.N) - ctx.N / 2) * ctx.dx
                X, Y = np.meshgrid(x, x)
                mask = (X * X + Y * Y) <= (ap / 2.0) ** 2
                # v4.14.2 (P1-NEW-4): dtype-aware zero so a complex64
                # cdtype is not silently upcast to complex128 by the
                # 0.0+0.0j literal.  Mirrors the v4.13.2 sweep at
                # apply_aperture / apply_mirror / _lens_thin /
                # _lens_real.
                E = np.where(mask, E, np.zeros((), dtype=cdtype))
            elif ap is not None and np.isfinite(ap) and ap <= 0:
                E = np.zeros((ctx.N, ctx.N), dtype=cdtype)
            else:
                E = np.ones((ctx.N, ctx.N), dtype=cdtype)

        # Field-angle tilt: apply a linear phase ramp for off-axis
        # illumination.  Identity for on-axis (0, 0).
        tx, ty = field_angle
        if tx or ty:
            x = (np.arange(ctx.N) - ctx.N / 2) * ctx.dx
            X, Y = np.meshgrid(x, x)
            k0 = 2.0 * np.pi / wavelength
            E = E * np.exp(1j * (k0 * np.sin(tx) * X
                                   + k0 * np.sin(ty) * Y))
        return E

    def _build_real_elements(self, ctx):
        """Expand ``_prescription_`` sentinels into real-lens elements.

        The sentinel supports an optional ``'index'`` key that selects
        from ``ctx.prescriptions`` (populated when using
        :class:`MultiPrescriptionParameterization`).  With no index
        supplied, falls back to ``ctx.prescription`` -- the
        single-prescription (backward-compatible) case.
        """
        lens_type = ('real_lens_traced' if self.use_traced_lens
                     else 'real_lens')
        extras = {}
        if self.use_traced_lens:
            extras['ray_subsample'] = self.ray_subsample

        prescriptions = ctx.prescriptions
        if prescriptions is None:
            prescriptions = [ctx.prescription]

        expanded = []
        for elem in self.real_elements:
            if elem.get('type') == '_prescription_':
                idx = int(elem.get('index', 0))
                if not (0 <= idx < len(prescriptions)):
                    raise IndexError(
                        f"'_prescription_' placeholder index {idx} out "
                        f"of range (ctx has {len(prescriptions)} "
                        f"prescriptions)")
                expanded.append({
                    'type': lens_type,
                    'prescription': prescriptions[idx],
                    **extras,
                    # Preserve any user-specified bandlimit / per-
                    # element overrides passed through the sentinel,
                    # except for the meta keys we've already consumed.
                    **{k: v for k, v in elem.items()
                       if k not in ('type', 'index')},
                })
            else:
                expanded.append(dict(elem))
        return expanded

    def _propagate(self, elements, E_in, ctx, wavelength):
        from ..propagators.system import propagate_through_system
        E_out, _ = propagate_through_system(
            E_in, elements, wavelength, ctx.dx)
        return E_out

    # -- Main evaluate ----------------------------------------------

    def evaluate(self, ctx: Any) -> float:
        if ctx.prescription is None:
            return self.weight

        wavelengths = self.wavelengths or [ctx.wavelength]
        field_angles = self.field_angles or [(0.0, 0.0)]

        penalties = []
        for wl in wavelengths:
            for fa in field_angles:
                try:
                    p = self._evaluate_one(ctx, wavelength=float(wl),
                                            field_angle=tuple(fa))
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # Per-(wavelength, field) evaluation can fail at
                    # extreme corners; substitute the worst-case
                    # weight so the optimizer steers away rather than
                    # crashing.
                    p = self.weight
                penalties.append(p)
        # Arithmetic mean across all (wavelength, field) combinations.
        return float(np.mean(penalties))

    def _evaluate_one(self, ctx, wavelength, field_angle):
        """Compute the merit for a single wavelength + field-angle pair."""
        E_in = self._make_source(ctx, wavelength, field_angle)
        E_ideal = self._propagate(self.ideal_elements, E_in, ctx, wavelength)
        real_elems = self._build_real_elements(ctx)
        E_real = self._propagate(real_elems, E_in, ctx, wavelength)

        if E_ideal.shape != E_real.shape:
            return self.weight

        mask = self.aperture_mask
        if mask is not None:
            E_ideal = E_ideal * mask
            E_real = E_real * mask

        # Optional axial focus search: find the z-offset where the
        # real field best matches the ideal's radiation pattern.  This
        # decouples "correct focal plane" from "aberration quality" so
        # a small BFL shift introduced by lens thickness doesn't
        # dominate the penalty.
        if self.focus_search:
            return self._focus_search_penalty(
                E_ideal, E_real, ctx, wavelength)

        return self._compute_penalty(E_ideal, E_real)

    def _focus_search_penalty(self, E_ideal, E_real, ctx, wavelength):
        """Propagate E_real through a small range of z offsets, pick
        the one that minimises the penalty (i.e. maximises overlap),
        and return that value.  Uses ASM (fast, exact, preserves dx).
        """
        from ..propagators.propagation import angular_spectrum_propagate
        # Default range: +-f/20 where f ~= ctx.efl or ctx.bfl; fall
        # back to +-5 mm if neither is available.
        if self.focus_search_range is not None:
            z_lo, z_hi = self.focus_search_range
        else:
            ref = ctx.bfl if (ctx.bfl and np.isfinite(ctx.bfl)
                                and abs(ctx.bfl) < 10) else ctx.efl
            if ref and np.isfinite(ref) and abs(ref) < 10:
                half = max(abs(ref) / 20.0, 1e-4)
            else:
                half = 5e-3
            z_lo, z_hi = -half, +half
        zs = np.linspace(z_lo, z_hi, max(3, self.focus_search_n))
        best = self.weight  # worst-case sentinel
        for dz in zs:
            E_shifted = (E_real if dz == 0.0
                         else angular_spectrum_propagate(
                             E_real, float(dz), wavelength, ctx.dx,
                             bandlimit=True))
            p = self._compute_penalty(E_ideal, E_shifted)
            if p < best:
                best = p
        return best

    def _compute_penalty(self, E_ideal, E_real):
        if self.match == 'field_overlap':
            return self._field_overlap_penalty(E_ideal, E_real)
        if self.match == 'field_mse':
            return self._field_mse_penalty(E_ideal, E_real)
        if self.match == 'intensity_mse':
            return self._intensity_mse_penalty(E_ideal, E_real)
        # intensity_overlap
        return self._intensity_overlap_penalty(E_ideal, E_real)

    # -- Metric kernels ---------------------------------------------

    @staticmethod
    def _field_overlap_penalty_raw(E_ideal, E_real):
        """Return (1 - coupling_efficiency).  Returns 1.0 if either
        field is zero (worst case)."""
        num = abs(np.vdot(E_ideal.ravel(), E_real.ravel())) ** 2
        p_i = float(np.sum(np.abs(E_ideal) ** 2))
        p_r = float(np.sum(np.abs(E_real) ** 2))
        den = p_i * p_r
        if den < 1e-60:
            return 1.0
        overlap = float(num / den)
        return 1.0 - overlap

    def _field_overlap_penalty(self, E_ideal, E_real):
        return self.weight * self._field_overlap_penalty_raw(E_ideal, E_real)

    def _field_mse_penalty(self, E_ideal, E_real):
        """Power-normalised, global-phase-aligned, squared L2 of the
        field residual.  Roughly the "fraction of energy in the
        difference" when amplitude-normalised."""
        p_i = float(np.sum(np.abs(E_ideal) ** 2))
        p_r = float(np.sum(np.abs(E_real) ** 2))
        if p_i < 1e-30 or p_r < 1e-30:
            return self.weight
        scale = np.sqrt(p_i / p_r)
        inner = np.vdot(E_ideal.ravel(), E_real.ravel())
        phase_align = (np.conj(inner) / abs(inner)) if abs(inner) > 1e-30 else 1.0 + 0.0j
        E_real_aligned = E_real * scale * phase_align
        mse = float(np.sum(np.abs(E_ideal - E_real_aligned) ** 2) / p_i)
        return self.weight * mse

    def _intensity_mse_penalty(self, E_ideal, E_real):
        """Phase-blind: compares |E|^2 patterns, normalised to equal
        total power."""
        I_i = np.abs(E_ideal) ** 2
        I_r = np.abs(E_real) ** 2
        p_i = float(np.sum(I_i))
        p_r = float(np.sum(I_r))
        if p_i < 1e-30 or p_r < 1e-30:
            return self.weight
        I_r_norm = I_r * (p_i / p_r)
        return self.weight * float(np.sum((I_i - I_r_norm) ** 2) / (p_i ** 2))

    def _intensity_overlap_penalty(self, E_ideal, E_real):
        I_i = np.abs(E_ideal) ** 2
        I_r = np.abs(E_real) ** 2
        num = float(np.sum(I_i * I_r))
        den = np.sqrt(float(np.sum(I_i ** 2)) * float(np.sum(I_r ** 2)))
        if den < 1e-30:
            return self.weight
        return self.weight * (1.0 - num / den)

    # -- Convenience -------------------------------------------------

    @classmethod
    def single_lens(cls, focal_length, post_distance=None, **kwargs):
        """Shortcut for the single-lens drop-in replacement case.

        Generates ``ideal_elements`` = [thin_lens(f), propagate(z=f
        or post_distance)].  Equivalent real_elements is the default
        single-``_prescription_`` drop-in.

        Parameters
        ----------
        focal_length : float
            Ideal thin-lens focal length [m].
        post_distance : float, optional
            Propagation distance from the lens to the output plane
            [m].  Default: ``focal_length`` (i.e. evaluate at the
            paraxial focus).
        kwargs : forwarded to :meth:`__init__`.
        """
        post = float(focal_length) if post_distance is None else float(post_distance)
        ideal = [
            {'type': 'lens', 'f': float(focal_length)},
            {'type': 'propagate', 'z': post},
        ]
        real_elems = kwargs.pop('real_elements', None)
        if real_elems is None:
            real_elems = [
                {'type': '_prescription_'},
                {'type': 'propagate', 'z': post},
            ]
        return cls(ideal_elements=ideal,
                   real_elements=real_elems,
                   **kwargs)


class MatchTargetOPDMerit(MeritTerm):
    """Penalise deviation of the actual exit-pupil OPD from a
    user-supplied target OPD map (or callable returning one).

    Use this when you have a desired wavefront -- not necessarily a
    perfect sphere -- that the lens should produce at its exit pupil.
    Examples: matching a measured wavefront, copying an existing
    well-corrected design, or shaping a beam with a target phase
    profile.

    Parameters
    ----------
    target_opd : ndarray or callable
        - If ndarray (shape (Ny, Nx) matching the simulation grid):
          used directly.
        - If callable: called as ``target_opd(X, Y, prescription)``
          and expected to return an ndarray of OPD [m] over the
          pupil grid.  Useful when the target depends on the current
          prescription (e.g. "match a target with the same EFL").
    weight : float, default 1.0
    exclude_low_order : int, default 1
        Number of Zernike modes to remove from the residual before
        computing RMS.
    n_modes : int, default 21
    """

    needs_wave = True
    name = 'MatchTargetOPD'

    def __init__(self, target_opd: Union[np.ndarray, Callable],
                 weight: float = 1.0,
                 exclude_low_order: int = 1,
                 n_modes: int = 21) -> None:
        self.target_opd = target_opd
        self.weight = float(weight)
        self.exclude_low_order = int(exclude_low_order)
        self.n_modes = int(n_modes)

    def evaluate(self, ctx: Any) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        Ny, Nx = ctx.opd_map.shape
        x = (np.arange(Nx) - Nx / 2) * ctx.dx
        y = (np.arange(Ny) - Ny / 2) * ctx.dx
        X, Y = np.meshgrid(x, y)
        if callable(self.target_opd):
            target = np.asarray(self.target_opd(X, Y, ctx.prescription))
        else:
            target = np.asarray(self.target_opd)
        if target.shape != ctx.opd_map.shape:
            raise ValueError(
                f'target_opd shape {target.shape} does not match '
                f'opd_map shape {ctx.opd_map.shape}')
        diff = ctx.opd_map - target
        from ..analysis import zernike_decompose as _zd
        finite = np.isfinite(diff)
        diff_clean = np.where(finite, diff, 0.0)
        try:
            coeffs, _ = _zd(diff_clean, ctx.dx, ap, n_modes=self.n_modes)
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                ZeroDivisionError):
            return 0.0
        higher = coeffs[self.exclude_low_order:]
        rms_m = float(np.sqrt(np.sum(higher ** 2)))
        rms_waves = rms_m / ctx.wavelength
        return self.weight * rms_waves * rms_waves


class ZernikeCoefficientMerit(MeritTerm):
    """Penalise (or target) specific Zernike-mode coefficients of the
    actual exit-pupil OPD.

    Lets you express design intents like:
    - "Eliminate spherical aberration" -- target mode 12 (Z_4^0) = 0
    - "Allow some defocus but no tilt or coma" -- target tilts (1,2),
      vertical/horizontal coma (7,8), and trefoils (6,9) all = 0
    - "Match a measured aberration profile mode-by-mode"

    Parameters
    ----------
    targets : dict of {int: float}
        Map from OSA Zernike index to target coefficient [m].  Modes
        not in this dict are unconstrained.
    weight : float, default 1.0
    n_modes : int, default 21
        Number of modes to fit (must exceed max key in ``targets``).
    """

    needs_wave = True
    name = 'ZernikeCoefficient'

    def __init__(self, targets: Dict[int, float],
                 weight: float = 1.0, n_modes: int = 21) -> None:
        self.targets = {int(j): float(v) for j, v in targets.items()}
        self.weight = float(weight)
        self.n_modes = max(int(n_modes), max(self.targets) + 1
                           if self.targets else int(n_modes))

    def evaluate(self, ctx: Any) -> float:
        if ctx.opd_map is None:
            return 0.0
        ap = ctx.prescription.get('aperture_diameter')
        if ap is None:
            return 0.0
        from ..analysis import zernike_decompose as _zd
        finite = np.isfinite(ctx.opd_map)
        opd_clean = np.where(finite, ctx.opd_map, 0.0)
        try:
            coeffs, _ = _zd(opd_clean, ctx.dx, ap, n_modes=self.n_modes)
        except (ValueError, RuntimeError, np.linalg.LinAlgError,
                ZeroDivisionError):
            return 0.0
        total = 0.0
        for j, target in self.targets.items():
            err_waves = (coeffs[j] - target) / ctx.wavelength
            total = total + err_waves * err_waves
        return self.weight * total


class LGAberrationMerit(MeritTerm):
    """Penalise specified Laguerre-Gaussian aberration-tensor channels
    via the closed-form modal asymptotic propagator (
    Section 7).

    Each entry ``L_{(p, ell), n}(s_2^img)`` of the LG aberration tensor
    is the projection of the system's leading-order asymptotic
    image-plane field onto a named classical aberration channel:
    ``(0, 0)`` is piston/Strehl, ``(1, 0)`` is defocus, ``(2, 0)`` is
    primary spherical, ``(0, +-1)`` is tilt, ``(1, +-1)`` is coma,
    ``(0, +-2)`` is astigmatism, ``(0, +-3)`` is trefoil.  Driving a
    given ``|L_{(p, ell), 0}|^2`` to zero suppresses that aberration
    in the merit-function loop without invoking the wave leg.

    The merit is computed from a Chebyshev tensor-product fit of the
    prescription's canonical map ``Phi(s2, v2), s1(s2, v2)``
    -- a single fit drives all targeted
    aberration channels at all chosen field points.

    Parameters
    ----------
    targets : dict
        Map from output LG index ``(p, ell)`` to a float weight.  Each
        listed channel contributes ``|L_{(p, ell), 0}(s_2^img)|^2``
        times the entry weight.  Channels not listed are unconstrained.
        Common targets:

            {(2, 0): 1.0, (1, 1): 1.0, (1, -1): 1.0, (0, 2): 1.0, (0, -2): 1.0}

        suppresses primary spherical, both coma orientations, and
        both astigmatism orientations.
    field_points : list of (float, float), optional
        Source-plane points [m] at which to evaluate the tensor.
        Default: a single on-axis point ``[(0.0, 0.0)]``.  Each field
        point's contribution is summed.
    image_points : list of (float, float), optional
        Image-plane evaluation points (one per field point).  If None,
        defaults to the chief-ray landing of each source point (which
        the merit evaluator finds via Newton).
    w_s, w_p : float
        Source-plane and pupil-plane Gaussian waists [m and direction
        cosine].  Defaults: ``w_s = 50e-6`` (50 um), ``w_p = 0.05``
        (50 mrad).
    w_o : float, optional
        Output Gaussian waist [m].  Default: derived per-pixel from
        the local complex beam matrix.
    fit_kwargs : dict, optional
        Additional keyword arguments passed to
        ``fit_canonical_polynomials``:  ``poly_order``,
        ``source_box_half``, ``pupil_box_half``, ``n_field``,
        ``n_pupil``, ``extract_linear_phase``, ``object_distance``.
    weight : float, default 1.0
    name : str, optional

    See Also
    --------
    lumenairy.asymptotic.aberration_tensor :  raw tensor evaluator.
    SphericalSeidelMerit :  Seidel-coefficient-based primary-spherical
        merit (uses paraxial coefficients; LGAberrationMerit's full
        non-paraxial generalisation is preferred for high-NA work).
    """

    needs_wave = False
    name = 'LGAberration'

    def __init__(self, targets: Dict[Any, float],
                 field_points: Optional[Sequence[Any]] = None,
                 image_points: Optional[Sequence[Any]] = None,
                 w_s: float = 50e-6, w_p: float = 0.05,
                 w_o: Optional[float] = None,
                 fit_kwargs: Optional[Dict[str, Any]] = None,
                 weight: float = 1.0,
                 name: Optional[str] = None) -> None:
        if not targets:
            raise ValueError("LGAberrationMerit: targets dict is empty")
        self.targets = {tuple(k): float(v) for k, v in targets.items()}
        if field_points is None:
            field_points = [(0.0, 0.0)]
        self.field_points = [tuple(p) for p in field_points]
        if image_points is None:
            self.image_points = None
        else:
            ips = list(image_points)
            if len(ips) != len(self.field_points):
                raise ValueError(
                    f"LGAberrationMerit: image_points length "
                    f"{len(ips)} must match field_points length "
                    f"{len(self.field_points)}")
            self.image_points = [tuple(p) for p in ips]
        self.w_s = float(w_s)
        self.w_p = float(w_p)
        self.w_o = None if w_o is None else float(w_o)
        self.fit_kwargs = dict(fit_kwargs) if fit_kwargs else {}
        self.weight = float(weight)
        if name is not None:
            self.name = str(name)

    def evaluate(self, ctx: Any) -> float:
        # Lazy import to avoid bootstrap cycles.
        from ..propagators.asymptotic import aberration_tensor, fit_canonical_polynomials
        # Canonical-fit cache: when a CompositeMerit contains several
        # LGAberrationMerit terms with the same fit_kwargs (typical:
        # one term per emitter class -- centre / edge / corner -- all
        # using the same source/pupil sampling box), build the fit
        # once per merit eval and reuse across terms.  Cache lives
        # only for the lifetime of one ctx; the next merit_fn(x) call
        # gets a fresh context.  Audit perf #2 (3.5.5).
        try:
            cache = ctx._canonical_fit_cache
        except AttributeError:
            cache = None
        # Hash key: wavelength + repr of sorted fit_kwargs items.
        # repr() handles nested dicts (e.g. surface_diffraction).
        # Different field_points with the SAME fit_kwargs share the
        # fit -- the fit is purely a property of (prescription,
        # wavelength, fit_kwargs); field_points are evaluation points
        # within the fit's domain.
        cache_key = None
        if cache is not None:
            try:
                cache_key = (float(ctx.wavelength),
                             repr(sorted(self.fit_kwargs.items(),
                                         key=lambda kv: kv[0])))
            except (TypeError, ValueError, AttributeError):
                # Unhashable / unsortable fit_kwargs (rare -- user
                # passed a non-stringifiable value).  Skip caching.
                cache_key = None
        if cache_key is not None and cache_key in cache:
            fit = cache[cache_key]
            if fit is None:
                # Previous attempt at this fit failed; short-circuit
                # to the same penalty.
                return 1e20
        else:
            try:
                fit = fit_canonical_polynomials(
                    ctx.prescription,
                    wavelength=ctx.wavelength,
                    **self.fit_kwargs,
                )
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # If the fit can't be built (e.g., aperture clipping
                # kills too many rays for the current prescription),
                # assign a large penalty so the optimiser steers away.
                if cache_key is not None and cache is not None:
                    cache[cache_key] = None
                return 1e20
            if cache_key is not None and cache is not None:
                cache[cache_key] = fit

        # The output modes are exactly the target keys, plus (0, 0)
        # for piston (always useful diagnostically).
        target_keys = list(self.targets.keys())
        output_modes = list(set([(0, 0)] + target_keys))

        total = 0.0
        for ifp, src in enumerate(self.field_points):
            if self.image_points is None:
                # Use the nominal chief-ray landing computed from the
                # paraxial back-map at v2_centre = 0.  fit's s2_centre/
                # halfrange box is centered on the actual landing
                # distribution, so s2_centre is a good chief estimate.
                s2_img = (fit.s2x_centre, fit.s2y_centre)
            else:
                s2_img = self.image_points[ifp]
            try:
                tensor = aberration_tensor(
                    fit,
                    s2_image=s2_img,
                    source_point=src,
                    source_modes=[(0, 0)],
                    pupil_modes=[(0, 0)],
                    output_modes=output_modes,
                    w_s=self.w_s, w_p=self.w_p, w_o=self.w_o,
                )
            except (ValueError, RuntimeError, ZeroDivisionError, KeyError,
                    np.linalg.LinAlgError, IndexError, AttributeError,
                    TypeError):
                # Aberration tensor evaluation can diverge on
                # singular pupil sums or out-of-domain field points;
                # max-penalty so the optimizer steers away.
                return 1e20
            # Index of each target in output_modes
            idx_map = {m: i for i, m in enumerate(output_modes)}
            for (p, ell), wgt in self.targets.items():
                try:
                    i = idx_map[(p, ell)]
                except KeyError:
                    continue
                val = complex(tensor.L[i, 0])
                total = total + wgt * (val.real * val.real
                                       + val.imag * val.imag)

        return self.weight * total


# =========================================================================
# Composite / Callable / Chromatic helpers
# =========================================================================

class CompositeMerit(MeritTerm):
    """Combine multiple sub-merits into one weighted sum.

    Useful for composing a complex objective from simpler pieces,
    or for grouping merits that share an expensive intermediate
    (e.g., the Zernike decomposition of the exit-pupil OPD).
    """

    name = 'Composite'

    def __init__(self, sub_merits: Sequence[MeritTerm],
                 weight: float = 1.0) -> None:
        self.sub_merits = list(sub_merits)
        self.weight = float(weight)
        self.needs_wave = any(m.needs_wave for m in self.sub_merits)

    def evaluate(self, ctx: Any) -> float:
        s = 0.0
        for m in self.sub_merits:
            s = s + m.evaluate(ctx)
        return self.weight * s


class CallableMerit(MeritTerm):
    """Generic merit term that delegates to a user-supplied callable.

    Use for one-off custom objectives that don't fit the prebuilt
    classes:

        def my_merit(ctx):
            # ctx.efl, ctx.bfl, ctx.seidel, ctx.E_exit, ctx.opd_map,
            # ctx.strehl_best, ctx.rms_radius_best, ctx.prescription, ...
            return some_scalar

        merit = CallableMerit(my_merit, weight=1.0, needs_wave=True)
    """

    name = 'Callable'

    def __init__(self, fn: Callable, weight: float = 1.0,
                 needs_wave: bool = False,
                 name: Optional[str] = None) -> None:
        self.fn = fn
        self.weight = float(weight)
        self.needs_wave = bool(needs_wave)
        if name is not None:
            self.name = name

    def evaluate(self, ctx: Any) -> float:
        return self.weight * float(self.fn(ctx))


class ChromaticFocalShiftMerit(MeritTerm):
    """Penalise focal-length variation across wavelengths.

    4.10.2: this term is now self-contained.  Pass the wavelengths
    explicitly at construction.  Pre-4.10.2 it depended on
    ``ctx.efls_per_wavelength`` being populated as a SIDE EFFECT of
    a prior ``MultiWavelengthMerit.evaluate()`` call earlier in the
    merit-term list -- if the ordering put this term first, the
    constraint silently disabled.

    Parameters
    ----------
    wavelengths : sequence of float, optional
        Wavelengths [m] to evaluate the EFL at.  When ``None`` falls
        back to the pre-4.10.2 behaviour of reading
        ``ctx.efls_per_wavelength`` (which requires a
        ``MultiWavelengthMerit`` to populate it earlier in the term
        list).
    """

    needs_wave = False
    name = 'ChromaticFocalShift'

    def __init__(self, weight: float = 1.0,
                 wavelengths: Optional[Sequence[float]] = None) -> None:
        self.weight = float(weight)
        self.wavelengths = ([float(w) for w in wavelengths]
                            if wavelengths is not None else None)

    def evaluate(self, ctx: Any) -> float:
        if self.wavelengths is not None:
            # 4.10.2: self-contained per-wavelength EFL evaluation.
            from ..raytrace import surfaces_from_prescription, system_abcd
            efls = []
            for wl in self.wavelengths:
                try:
                    surfs = surfaces_from_prescription(ctx.prescription)
                    _, efl, _, _ = system_abcd(surfs, wl)
                    if np.isfinite(efl):
                        efls.append(float(efl))
                except (ValueError, RuntimeError, ZeroDivisionError,
                        KeyError, np.linalg.LinAlgError, IndexError,
                        AttributeError, TypeError):
                    # EFL unobtainable at this wavelength (degenerate
                    # ABCD / missing glass entry); drop from the
                    # spread accumulator.
                    pass
            if len(efls) < 2:
                return 0.0
            pv = max(efls) - min(efls)
            return self.weight * pv * pv
        # Legacy fallback: read context-attached EFLs.
        if (getattr(ctx, 'efls_per_wavelength', None) is None
                or len(ctx.efls_per_wavelength) < 2):
            return 0.0
        pv = (np.max(ctx.efls_per_wavelength)
              - np.min(ctx.efls_per_wavelength))
        return self.weight * pv * pv


# =========================================================================
# Constraint-style merits (geometric, fast)
# =========================================================================

class MinThicknessMerit(MeritTerm):
    """Penalise any GLASS thickness below a minimum.

    ``contribution = weight * sum_glass_thicknesses max(0, min_t - t_i)^2``

    4.10: only glass thicknesses count; air gaps are skipped.  Pre-4.10
    iterated every entry in ``prescription['thicknesses']`` which
    included air gaps that legitimately need to be small (e.g. cemented
    interfaces, near-zero gap between two surfaces).

    Parameters
    ----------
    min_thickness : float
        Minimum acceptable GLASS thickness [m].  Use
        ``MinAirGapMerit`` (if added) for air-gap constraints.
    weight : float
    include_air : bool, optional
        Set True to restore the pre-4.10 behaviour and also penalise
        small air gaps.  Default False.
    """

    needs_wave = False
    name = 'MinThickness'

    def __init__(self, min_thickness: float = 1e-3,
                 weight: float = 1.0,
                 include_air: bool = False) -> None:
        self.min_thickness = float(min_thickness)
        self.weight = float(weight)
        self.include_air = bool(include_air)

    def evaluate(self, ctx: Any) -> float:
        thicknesses = ctx.prescription.get('thicknesses', [])
        surfaces = ctx.prescription.get('surfaces', [])
        total = 0.0
        for i, t in enumerate(thicknesses):
            if not self.include_air:
                # Determine if this thickness sits between two glass
                # interfaces: use the glass_after of the i-th surface
                # (which is the material the i-th thickness sits in).
                glass = 'air'
                if i < len(surfaces):
                    surf = surfaces[i]
                    glass = (surf.get('glass_after', 'air')
                              if isinstance(surf, dict)
                              else getattr(surf, 'glass_after', 'air'))
                if isinstance(glass, str) and glass.lower() in (
                        'air', 'vacuum', '', None):
                    continue
            deficit = max(0.0, self.min_thickness - float(t))
            total = total + deficit * deficit
        return self.weight * total


class MaxThicknessMerit(MeritTerm):
    """Penalise any glass thickness above a maximum."""

    needs_wave = False
    name = 'MaxThickness'

    def __init__(self, max_thickness: float = 20e-3,
                 weight: float = 1.0) -> None:
        self.max_thickness = float(max_thickness)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        thicknesses = ctx.prescription.get('thicknesses', [])
        total = 0.0
        for t in thicknesses:
            excess = max(0.0, float(t) - self.max_thickness)
            total = total + excess * excess
        return self.weight * total


class MinBackFocalLengthMerit(MeritTerm):
    """Penalise BFL below a minimum (e.g. to keep clearance for
    a sensor package)."""

    needs_wave = False
    name = 'MinBFL'

    def __init__(self, min_bfl: float = 5e-3,
                 weight: float = 1.0) -> None:
        self.min_bfl = float(min_bfl)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        # 4.10: ctx.bfl is set to the sentinel 1e9 when the ray leg
        # fails.  Pre-4.10 deficit = max(0, min_bfl - 1e9) = 0, so
        # invalid prescriptions silently scored as "satisfies clearance".
        # Penalise them with a large finite value instead.
        if not ctx_is_valid(ctx, 'bfl'):
            return self.weight * (self.min_bfl ** 2)
        deficit = max(0.0, self.min_bfl - ctx.bfl)
        return self.weight * deficit * deficit


class MaxFNumberMerit(MeritTerm):
    """Penalise an f/# above a maximum (force faster lens)."""

    needs_wave = False
    name = 'MaxFNumber'

    def __init__(self, max_f_number: float = 8.0,
                 weight: float = 1.0) -> None:
        self.max_f_number = float(max_f_number)
        self.weight = float(weight)

    def evaluate(self, ctx: Any) -> float:
        # 4.10: guard against the ctx.efl = 1e9 sentinel.  Pre-4.10 a
        # failed ray leg produced fnum = 1e9 / aperture ~ 1e12, squared
        # to ~ 1e24 -- swamping every other merit term in the sum so the
        # optimizer "saw" only this penalty when the ray leg failed.
        if not ctx_is_valid(ctx, 'efl'):
            return self.weight
        ap = ctx.prescription.get('aperture_diameter', 1e-3)
        # v5.17.x (AUDIT_V5_17_0 P3-48): ``aperture_diameter`` present
        # with value ``None`` is a legal "no aperture" prescription
        # state (every sibling merit None-guards it); dict.get's
        # default does NOT fire for an existing None, and ``None > 0``
        # raised TypeError out of scipy.minimize, aborting the run.
        # Route None to the existing ap <= 0 fallback penalty below.
        if ap is None:
            ap = 0.0
        fnum = abs(ctx.efl) / ap if ap > 0 else 1e9
        excess = max(0.0, fnum - self.max_f_number)
        return self.weight * excess * excess
