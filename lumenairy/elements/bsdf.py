"""BSDF (Bidirectional Scattering Distribution Function) models for
surface microroughness and stray-light analysis.

A BSDF describes the angular distribution of radiance scattered by an
optical surface into directions other than the nominal reflected/
transmitted direction.  Attaching one to a :class:`raytrace.Surface`
via its ``bsdf`` field enables three workflows:

1. **Evaluation**  ``evaluate(incident_dir, scattered_dir)`` returns the
   BSDF value (units of 1/sr).  Useful for computing scattered
   irradiance integrals.

2. **Sampling**    ``sample(incident_dir, n, rng)`` draws ``n`` scattered
   directions according to the BSDF lobe, for Monte Carlo stray-light
   propagation.  The returned direction cosines can be fed into a
   fresh :class:`raytrace.RayBundle` and traced through the rest of
   the system.

3. **TIS**         ``total_integrated_scatter()`` returns the fraction of
   incident power scattered out of the specular direction.  Matches
   the standard spec used in coating/mirror datasheets.

Three models are supplied:

* :class:`LambertianBSDF`  -- uniform angular distribution (baseline
  for matte black surfaces, roughened metals).
* :class:`GaussianBSDF`    -- small-angle Gaussian lobe around the
  specular direction (typical polished-optic microroughness model for
  smooth surfaces).
* :class:`HarveyShackBSDF` -- three-parameter Harvey-Shack ABC model,
  a physically-motivated fit to the power-spectral-density of
  surface height variations.  Standard reference for mirror
  scatter at 633 nm and similar.

All three expose the same interface, so the ``Surface.bsdf`` attribute
is polymorphic.  The kind flag in the ``dict`` form
(``{'kind': 'lambertian', ...}``) is used by helpers that need to
serialize a BSDF (e.g. the CODE V / Zemax writers).

References
----------
[1] Harvey, J.E. (1976).  "Light-scattering characteristics of optical
    surfaces".  Ph.D. dissertation, University of Arizona.
[2] Harvey, J.E., Choi, N., Krywonos, A. (2009).  "Scattering from
    smooth-surface optics: A unified approach to surface roughness
    scatter".  Proc. SPIE 7426.
[3] Bass, M. (ed.). *Handbook of Optics, Vol. I* (3rd ed.), Ch. 8.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np

# Innermost node of the default TIS quadrature, in u = sin(theta).  Any lobe
# is flat well inside its own shoulder, so the u < _TIS_U_MIN disc is added
# analytically; 1e-7 is four decades below the narrowest shoulder a polished
# optic exhibits (l ~ 1e-3) and contributes ~pi*B(0)*1e-14 to the integral.
_TIS_U_MIN = 1e-7


# =============================================================================
# Base class
# =============================================================================


class BSDFModel(ABC):
    """Abstract base.  Subclasses implement ``evaluate`` and ``sample``.

    ``BSDFModel`` is an explicit ``abc.ABC``; attempting to
    instantiate it directly raises ``TypeError`` at construction
    rather than waiting for a downstream method call to fail with
    ``NotImplementedError``.  Subclass and implement the
    ``@abstractmethod`` slots ``evaluate`` and ``sample`` to make
    the class concrete -- see :class:`LambertianBSDF`,
    :class:`GaussianBSDF`, and :class:`HarveyShackBSDF` for canonical
    examples.

    Conventions
    -----------
    * Incident direction ``(Li, Mi, Ni)`` and scattered ``(Ls, Ms, Ns)``
      are **unit direction cosines** in the local surface frame where
      the outward surface normal points along ``+z`` (``Ni < 0`` for
      a ray arriving, ``Ns > 0`` for a ray scattered into the outgoing
      hemisphere).
    * Returned BSDF values are **per steradian**, as conventional.
    * Sample(): draws into the outgoing hemisphere only; rejection
      sampling is acceptable.
    """
    kind: str = 'abstract'

    @abstractmethod
    def evaluate(self, incident_dir: np.ndarray, scattered_dir: np.ndarray) -> np.ndarray:
        """BSDF value (1/sr) for the requested scattering direction."""

    @abstractmethod
    def sample(
        self,
        incident_dir: np.ndarray,
        n_samples: int,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        """Draw ``n_samples`` outgoing direction cosines from the
        BSDF lobe."""

    # True when the lobe is defined about the SPECULAR direction (so a
    # local-frame draw has to be rotated into it); False when it is defined
    # about the surface normal and the local draw is already in the surface
    # frame (Lambertian).  Read by :func:`sample_scatter_rays`, which draws
    # every ray's local sample in one call and rotates them as a batch.
    _lobe_about_specular: bool = True

    def _sample_local(
        self,
        n_samples: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Draw ``n_samples`` direction cosines in the LOBE-LOCAL frame
        (lobe axis along ``+z``), before any rotation onto the specular
        direction.  Incidence-independent by construction, which is what
        lets a whole ray bundle be sampled in one call."""
        raise NotImplementedError(
            f"{type(self).__name__}._sample_local: subclasses of BSDFModel "
            f"must implement _sample_local(n_samples, rng) -> (n, 3) local "
            f"direction cosines so that sample_scatter_rays can draw a whole "
            f"bundle at once.")

    def total_integrated_scatter(self) -> float:
        """TIS = integral over outgoing hemisphere of BSDF * cos(theta) dOmega.

        Subclasses override when a closed form is available; the
        default falls back to a numerical integration which is always
        correct but slow.

        The quadrature variable is ``u = sin(theta)``, not ``theta``:
        ``BSDF cos(theta) dOmega = BSDF(u) u du dphi`` exactly, so the
        integrand loses the cos/sin weights and -- more importantly -- the
        samples can be placed where a scatter lobe actually lives.  Real
        microroughness lobes have shoulders at ``u ~ 1e-3..1e-2``; a linear
        grid in theta with a few hundred points does not resolve them and
        under-reads the integral by double-digit percentages.  The nodes are
        therefore two-point Gauss-Legendre inside GEOMETRIC cells in
        ``ln u``, from ``_TIS_U_MIN`` to 1, with the remaining
        ``[0, u_min]`` disc added analytically as ``B(u_min) u_min**2 / 2``
        (the lobe is flat there by construction).

        Measured against a 2e6-point reference at the same node count
        (256 x 128), relative error across Lambertian, Gaussian
        (sigma = 1e-2 and 0.3) and Harvey-Shack (l = 1e-1 ... 1e-4,
        s = 1.5 / 2 / 2.5): worst 1.4e-6, typical 1e-9.  The linear-theta
        grid it replaces read -32 % on Gaussian(sigma = 1e-2), -18 % on
        Harvey-Shack(l = 1e-3) and -32 % at l = 1e-4.  The three shipped
        models all override this method with a closed form, so the grid
        matters for user subclasses.

        That 1.4e-6 is the worst of THOSE lobes, not a bound.  The rule is
        exact through cubic order in ``v = ln u``, so the residual is the
        quartic term ``dv**4 * 2**4 / 4320`` -- with ``dv = ln(1e7)/128``
        that is 9.3e-7 for a FLAT lobe (whose integrand ``u**2 = exp(2 v)``
        is not a cubic), and it was measured at 9.29e-7 for ``B = const``
        and 1.3e-5 for ``B = 1 - u**2`` (VERIFY-A8, 2026-09-12).  So
        budget ~1e-5 for a broad smooth lobe -- still four to five decades
        better than the grid this replaces.  A lobe narrower than about
        ``10 * _TIS_U_MIN`` loses accuracy to the analytic inner disc
        instead (6.4e-6 at l = 1e-6).

        v4.13.0 (Tier-2 perf, audit group alpha): the integrand is now
        evaluated as one fully-vectorised meshgrid call rather than a
        per-(theta, phi) Python loop, yielding ~2-3 orders of magnitude
        speedup at the default 256x128 quadrature.
        """
        n_cells = 128
        n_theta = 2 * n_cells       # two Gauss nodes per cell
        n_phi = 128
        # Geometric cells in v = ln u (u = sin theta), two-point
        # Gauss-Legendre inside each: with du = u dv the hemisphere integral
        # is int f(u) u**2 dv, whose integrand is smooth in v for any lobe
        # (a power-law tail is a straight line there).  Two nodes per cell
        # integrate that exactly through cubic order, so the residual is
        # O(dv**4) uniformly across lobe widths instead of the linear-theta
        # grid's "resolve the shoulder or lose it".
        ln_lo = np.log(_TIS_U_MIN)
        v_edges = np.linspace(ln_lo, 0.0, n_cells + 1)
        dv = v_edges[1] - v_edges[0]
        v_mid = 0.5 * (v_edges[:-1] + v_edges[1:])
        off = dv / (2.0 * np.sqrt(3.0))
        v = np.concatenate([v_mid - off, v_mid + off])
        order = np.argsort(v)
        v = v[order]
        u = np.exp(v)
        w_u = 0.5 * dv * u ** 2
        theta = np.arcsin(np.clip(u, 0.0, 1.0))
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        W, _ = np.meshgrid(w_u, phi, indexing='ij')
        # Build the full (n_theta, n_phi, 3) scattered-direction grid
        # and evaluate the BSDF in one call.  ``inc`` is broadcast
        # against the (..., 3) scattered direction shape.
        sin_T = np.sin(T)
        S = np.empty(T.shape + (3,), dtype=np.float64)
        S[..., 0] = sin_T * np.cos(P)
        S[..., 1] = sin_T * np.sin(P)
        S[..., 2] = np.cos(T)
        inc = np.array([0.0, 0.0, -1.0])
        B = np.asarray(self.evaluate(inc, S), dtype=np.float64)
        # v4.14 (audit P2 #17): require the subclass evaluator to
        # return a result whose shape matches the integration grid.
        # Pre-v4.14 a silent ``np.broadcast_to(B, T.shape)`` masked
        # subclass bugs (returning a scalar / rank-1 / wrong-axis
        # result), producing a TIS value that quietly drifted from
        # what the implementor intended.  Surface the mismatch as a
        # clear ValueError so the subclass is fixed at definition
        # time rather than during downstream physics analysis.
        if B.shape != T.shape:
            raise ValueError(
                f"BSDFModel.evaluate returned shape {B.shape!r}, but "
                f"total_integrated_scatter expected shape "
                f"{T.shape!r} (the integration grid: n_theta="
                f"{n_theta}, n_phi={n_phi}).  Subclass evaluators "
                f"must broadcast against the (n_theta, n_phi, 3) "
                f"scattered-direction grid and return a (n_theta, "
                f"n_phi) array of BSDF values.  Either fix the "
                f"evaluator's broadcasting, or override "
                f"total_integrated_scatter() with a closed-form "
                f"expression.")
        dph = phi[1] - phi[0]
        # Hemisphere integral in u = sin(theta): BSDF * u du dphi.
        tis = float((B * W).sum() * dph)
        # Analytic completion of the u < _TIS_U_MIN disc.
        tis += float(B[0].mean() * 2 * np.pi * _TIS_U_MIN ** 2 / 2)
        return tis


# =============================================================================
# Lambertian
# =============================================================================


@dataclass
class LambertianBSDF(BSDFModel):
    """Uniform diffuse scatter.

    BSDF(theta_in, theta_out, phi) = rho / pi   (constant).

    ``rho`` is the **diffuse reflectance** (0 = perfect absorber,
    1 = perfect diffuser).  The TIS of a Lambertian surface equals
    ``rho`` exactly.

    Frame assumption
    ----------------
    The ``incident_dir`` and ``scattered_dir`` arguments are assumed
    to be expressed in the **surface-local frame** where ``+z`` is
    the outward surface normal.  The "hemisphere" check used by
    :meth:`evaluate` is a sign test on the z-component of
    ``scattered_dir`` -- a non-local frame (e.g. world-coordinates
    with the surface tilted away from z) will mis-classify rays.
    Callers responsible for the global -> surface-local rotation
    BEFORE calling :meth:`evaluate`.

    v4.15.1 (P3-3 / Agent E): :meth:`evaluate` now emits a
    ``RuntimeWarning`` if the ``incident_dir`` has a non-trivial
    x/y component (|x|+|y| > 0.3) but its z-component is close to
    +1 -- a signature that the caller is likely passing a
    world-frame direction without applying the surface-frame
    rotation.  The warning is advisory; the result is still
    returned because Monte-Carlo paths that DO pre-rotate produce
    consistent z-aligned ``incident_dir`` and won't trip it.
    """
    rho: float = 1.0
    kind: str = 'lambertian'

    def evaluate(self, incident_dir: np.ndarray, scattered_dir: np.ndarray) -> np.ndarray:
        # v4.15.1 (P3-3 / Agent E): frame-assumption advisory.  The
        # Lambertian BSDF is invariant under azimuth in the
        # surface-local frame, but the hemisphere check (``sd[z] > 0``)
        # IS frame-sensitive.  Catch the common mistake of passing
        # an off-axis incident direction without applying the
        # surface-frame rotation.
        inc = np.asarray(incident_dir, dtype=float)
        if inc.ndim == 1 and inc.size == 3:
            ix, iy, iz = float(inc[0]), float(inc[1]), float(inc[2])
            if abs(ix) + abs(iy) > 0.3 and iz > 0.7:
                import warnings as _w
                _w.warn(
                    "LambertianBSDF.evaluate: incident_dir "
                    f"({ix:.3f}, {iy:.3f}, {iz:.3f}) has a large "
                    f"transverse component (|x|+|y|={abs(ix)+abs(iy):.3f}) "
                    "while z>+0.7; this is a signature of a world-frame "
                    "direction being passed without applying the "
                    "surface-frame rotation (which would yield "
                    "incident_dir~=(0,0,+/-1)).  The Lambertian BSDF "
                    "assumes the surface frame has +z = outward "
                    "normal; mis-framed inputs mis-classify the "
                    "hemisphere check.  See the class docstring for "
                    "the canonical frame convention.",
                    RuntimeWarning, stacklevel=2,
                )
        sd = np.asarray(scattered_dir)
        if sd.ndim == 1:
            in_hemi = sd[2] > 0
        else:
            in_hemi = sd[..., 2] > 0
        return self.rho / np.pi * in_hemi

    def sample(
        self,
        incident_dir: np.ndarray,
        n_samples: int,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        return self._sample_local(n_samples, _get_rng(rng))

    # The Lambertian lobe is about the surface NORMAL, so the local draw is
    # already in the surface frame and needs no specular rotation.
    _lobe_about_specular = False

    def _sample_local(self, n_samples, rng):
        # Cosine-weighted hemisphere sample (Lambertian importance).
        xi1 = rng.random(n_samples)
        xi2 = rng.random(n_samples)
        theta = np.arcsin(np.sqrt(xi1))
        phi = 2 * np.pi * xi2
        L = np.sin(theta) * np.cos(phi)
        M = np.sin(theta) * np.sin(phi)
        N = np.cos(theta)
        return np.stack([L, M, N], axis=-1)

    def total_integrated_scatter(self) -> float:
        return float(self.rho)


# =============================================================================
# Gaussian lobe (polished-surface microroughness approximation)
# =============================================================================


@dataclass
class GaussianBSDF(BSDFModel):
    """Small-angle Gaussian lobe centred on the specular direction.

    BSDF(theta_s) = A * exp( -(theta_s - theta_spec)^2 / (2 sigma^2) )

    where ``theta_s`` is the scattered-direction polar angle relative
    to the specular direction.  The normalization ``A`` is computed so
    that the TIS equals the caller-specified ``scattered_fraction``
    (typical polished optics: ~0.001 -- 0.01, i.e. 0.1 % -- 1 %).

    v5.17 (audit P3-15): the normalization is incidence-aware --
    ``evaluate`` divides the closed-form ``A`` by
    ``|cos(theta_spec)|`` so the projected-solid-angle integral
    ``int BSDF * cos(theta_s) dOmega`` equals ``scattered_fraction``
    at ALL incidence angles, consistent with
    :meth:`total_integrated_scatter`.  (Previously the closed form
    assumed a normal-sitting lobe, so oblique hemisphere integrals of
    ``evaluate`` under-reported by a factor ``cos(theta_i)``.)

    Parameters
    ----------
    sigma_rad : float
        1/e half-width of the scatter lobe [rad].  Smaller = more
        specular-like; typical polished optics: 1-10 mrad.
    scattered_fraction : float, default 0.01
        Fraction of incident power going into the scatter lobe
        (the TIS).  The remaining 1-f goes into the specular
        direction.
    """
    sigma_rad: float = 0.01
    scattered_fraction: float = 0.01
    kind: str = 'gaussian'

    def _normalization(self) -> float:
        # Integrate  f(theta)*cos(theta)*sin(theta) dtheta dphi  =
        # A * 2pi * int_0^{pi/2} exp(-theta^2/(2 sigma^2)) cos*sin dtheta
        # For sigma << 1 rad the integrand is well-approximated by
        # sin(theta) ~ theta, cos(theta) ~ 1:
        #   ~ A * 2pi * sigma^2 * (1 - exp(-pi^2/(8 sigma^2)))  [Gaussian]
        # We enforce TIS = scattered_fraction.
        s = self.sigma_rad
        # Closed-form in the small-angle limit:
        return self.scattered_fraction / (2 * np.pi * s ** 2)

    def evaluate(self, incident_dir: np.ndarray, scattered_dir: np.ndarray) -> np.ndarray:
        inc = np.asarray(incident_dir, dtype=float)
        sd = np.asarray(scattered_dir, dtype=float)
        # Specular direction = (L_i, M_i, -N_i) if we flip z;
        # here we assume incident has N<0 and scattered has N>0.
        # v5.4.6 (audit F-22): stack along the LAST axis so the batched-
        # incidence path produces (..., 3) matching scattered_dir, not
        # the transposed (3, ...) that crashed broadcasting in (M,3)*(3,M).
        specular = np.stack([inc[..., 0], inc[..., 1], -inc[..., 2]], axis=-1)
        # cos(theta_s) between scattered and specular
        cos_theta = np.clip(np.sum(sd * specular, axis=-1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        A = self._normalization()
        # v5.17 audit (P3-15): the closed-form A assumes the lobe sits
        # at the surface normal (cos(theta_s) ~ 1 over the lobe).  For
        # an oblique specular direction the cos(theta_s) weight in
        # TIS = int BSDF * cos(theta_s) dOmega is ~|cos(theta_spec)|,
        # so divide it out to keep TIS == scattered_fraction at all
        # incidence angles (consistent with total_integrated_scatter).
        cos_spec = np.abs(inc[..., 2]) if inc.ndim > 1 else abs(inc[2])
        A = A / np.where(cos_spec > 1e-12, cos_spec, 1.0)
        in_hemi = sd[..., 2] > 0 if sd.ndim > 1 else sd[2] > 0
        return (A * np.exp(-theta ** 2 / (2 * self.sigma_rad ** 2))
                * in_hemi)

    def sample(
        self,
        incident_dir: np.ndarray,
        n_samples: int,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        local = self._sample_local(n_samples, _get_rng(rng))
        return _rotate_local_to_specular(local, incident_dir)

    def _sample_local(self, n_samples, rng):
        # BSDF-1 (AUDIT_BSDF_SEGMENT_GEOMETRY): draw the offset angle from a
        # RAYLEIGH law, not a half-normal.  To reproduce the Gaussian lobe as a
        # Monte-Carlo DIRECTION distribution the per-theta density must carry
        # the solid-angle + projected-power weight
        # ``BSDF(theta)*cos*sin ~ theta*exp(-theta^2/2 sigma^2)`` (small angle)
        # -- exactly Rayleigh(sigma), via inverse-CDF ``theta =
        # sigma*sqrt(-2 ln(1-xi))``.  The old half-normal ``|N(0,sigma)|``
        # dropped the ``sin(theta)~theta`` factor, over-concentrating samples
        # toward specular (drawn mean 0.80*sigma vs the true lobe's
        # 1.25*sigma) -> a ~35%-too-narrow stray-light cone.  ``evaluate`` /
        # ``total_integrated_scatter`` (closed-form) were unaffected; this is
        # isolated to the MC draw, and matches the HarveyShack/Lambertian
        # ``~ BSDF*cos`` sampling convention the Gaussian path alone violated.
        xi = rng.random(n_samples)
        theta = self.sigma_rad * np.sqrt(-2.0 * np.log1p(-xi))
        theta = np.minimum(theta, np.pi / 2 - 1e-6)
        phi = 2 * np.pi * rng.random(n_samples)
        # Local frame: specular = +z
        L_loc = np.sin(theta) * np.cos(phi)
        M_loc = np.sin(theta) * np.sin(phi)
        N_loc = np.cos(theta)
        return np.stack([L_loc, M_loc, N_loc], axis=-1)

    def total_integrated_scatter(self) -> float:
        return float(self.scattered_fraction)


# =============================================================================
# Harvey-Shack ABC (three-parameter PSD-based model)
# =============================================================================


@dataclass
class HarveyShackBSDF(BSDFModel):
    """Harvey-Shack ABC model for smooth-surface microroughness.

    BSDF(theta_s) = b0 / ( 1 + (sin(theta_s) / l)^2 )^(s/2)

    where ``theta_s`` is the angle between scattered and specular
    directions.  Parameters:

    * ``b0`` (``A`` in some references): on-axis BSDF value [1/sr]
    * ``l``  (``B``): shoulder angle (transition from flat to rolloff)
    * ``s``  (``C``): high-angle rolloff exponent (typical: 1.5-2.5)

    This is the most common analytic form used in stray-light
    simulators for polished optics at visible and NIR wavelengths.
    The model is surface-side only (no wavelength scaling); to scale
    BSDF with wavelength use the optional ``wavelength_ref`` /
    ``wavelength`` pair which follows the 1/lambda^2 Rayleigh-like
    smooth-surface scaling.

    Parameters
    ----------
    b0 : float
        On-axis BSDF amplitude [1/sr].
    l : float
        Shoulder angle parameter [sin of angle, so unitless].
    s : float
        High-angle rolloff exponent (must be > 1 for finite TIS).
    wavelength_ref : float, optional
        Reference wavelength [m] for scatter scaling.
    wavelength : float, optional
        Current wavelength [m].  If both ``wavelength_ref`` and
        ``wavelength`` are set, amplitude scales as
        ``(wavelength_ref / wavelength) ** 2`` (smooth-surface limit).
    """
    b0: float = 1.0
    l: float = 0.01
    s: float = 2.0
    wavelength_ref: Optional[float] = None
    wavelength: Optional[float] = None
    kind: str = 'harvey_shack'

    def _amplitude(self) -> float:
        if self.wavelength_ref and self.wavelength:
            return (self.b0
                    * (self.wavelength_ref / self.wavelength) ** 2)
        return self.b0

    def evaluate(self, incident_dir: np.ndarray, scattered_dir: np.ndarray) -> np.ndarray:
        inc = np.asarray(incident_dir, dtype=float)
        sd = np.asarray(scattered_dir, dtype=float)
        # BSDF-nit (AUDIT_BSDF_SEGMENT_GEOMETRY): mirror GaussianBSDF's F-22
        # batch-safe form -- stack the specular along the LAST axis and reduce
        # with axis=-1 so a batched-incidence ``(..., 3)`` call broadcasts
        # instead of crashing on the single-incidence ``inc[0]`` indexing.
        specular = np.stack([inc[..., 0], inc[..., 1], -inc[..., 2]], axis=-1)
        specular = specular / np.linalg.norm(specular, axis=-1, keepdims=True)
        cos_theta = np.clip(np.sum(sd * specular, axis=-1), -1.0, 1.0)
        in_hemi = sd[..., 2] > 0
        sin_theta = np.sqrt(np.maximum(1 - cos_theta ** 2, 0.0))
        amp = self._amplitude()
        return amp / (1 + (sin_theta / self.l) ** 2) ** (self.s / 2) * in_hemi

    def total_integrated_scatter(self) -> float:
        """Closed-form hemisphere integral of the ABC lobe.

        With ``u = sin(theta)`` the hemisphere integral separates exactly::

            TIS = 2 pi b0 int_0^1 u du / (1 + (u/l)^2)^(s/2)

        and the substitution ``t = 1 + (u/l)^2`` integrates it in closed
        form::

            TIS = pi b0 l^2 [(1 + 1/l^2)^(1 - s/2) - 1] / (1 - s/2)   s != 2
            TIS = pi b0 l^2 ln(1 + 1/l^2)                             s == 2

        Exact and O(1).  The inherited quadrature is not used because the
        lobe is concentrated at ``u ~ l``, which the default grid must
        resolve; at ``l = 1e-3`` the pre-override linear-theta grid
        under-read TIS by 18 %.
        """
        amp = self._amplitude()
        l_sq = float(self.l) ** 2
        s = float(self.s)
        ratio = 1.0 + 1.0 / l_sq
        if abs(s - 2.0) < 1e-12:
            return float(np.pi * amp * l_sq * np.log(ratio))
        p = 1.0 - s / 2.0
        return float(np.pi * amp * l_sq * (ratio ** p - 1.0) / p)

    def sample(
        self,
        incident_dir: np.ndarray,
        n_samples: int,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ) -> np.ndarray:
        local = self._sample_local(n_samples, _get_rng(rng))
        return _rotate_local_to_specular(local, incident_dir)

    def _sample_local(self, n_samples, rng):
        # Exact inverse-CDF draw of u = sin(theta) from the power-weighted
        # radial density  p(u) ~ u / (1 + (u/l)^2)^(s/2)  on [0, 1].
        # Substituting t = 1 + (u/l)^2 makes the CDF elementary (the same
        # substitution that closes total_integrated_scatter):
        #     s == 2 : t = T**xi                       with T = 1 + 1/l^2
        #     s != 2 : t = (1 + xi (T**p - 1))**(1/p)  with p = 1 - s/2
        #     u = l * sqrt(t - 1)
        # This replaces a rejection sampler whose acceptance was ~9 % at
        # l = 1e-2 and ~1 % at l = 1e-3 (and which grew a Python list), so
        # the draw is now one RNG call of known length -- which is also what
        # lets sample_scatter_rays draw a whole bundle at once.
        l = float(self.l)
        s = float(self.s)
        xi = rng.random(n_samples)
        big_t = 1.0 + 1.0 / (l * l)
        if abs(s - 2.0) < 1e-12:
            t = big_t ** xi
        else:
            p = 1.0 - s / 2.0
            t = (1.0 + xi * (big_t ** p - 1.0)) ** (1.0 / p)
        sin_theta = np.clip(l * np.sqrt(np.maximum(t - 1.0, 0.0)), 0.0, 1.0)
        phi = 2 * np.pi * rng.random(n_samples)
        cos_theta = np.sqrt(np.maximum(1 - sin_theta ** 2, 0.0))
        L_loc = sin_theta * np.cos(phi)
        M_loc = sin_theta * np.sin(phi)
        N_loc = cos_theta
        return np.stack([L_loc, M_loc, N_loc], axis=-1)


# =============================================================================
# Builders and utilities
# =============================================================================


def _get_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def _rotate_local_to_specular(local: np.ndarray,
                              incident_dir: np.ndarray) -> np.ndarray:
    """Rotate lobe-local direction cosines onto the specular direction.

    ``local`` is ``(n, 3)`` with the lobe axis along ``+z``; ``incident_dir``
    is either one ``(3,)`` direction (applied to every sample) or ``(n, 3)``,
    one per sample -- the batched form is what lets a whole ray bundle be
    rotated in a single call instead of one Python call per ray.

    The specular direction of an arriving ``(Li, Mi, Ni)`` is
    ``(Li, Mi, -Ni)``; the orthonormal frame around it uses ``+z`` as the
    reference up-vector, falling back to ``+x`` when the specular direction
    is within 2.6 deg of ``+-z`` (where the cross product degenerates).
    Samples landing below the surface are folded back into the outgoing
    hemisphere.
    """
    local = np.asarray(local, dtype=float)
    inc = np.atleast_2d(np.asarray(incident_dir, dtype=float))
    spec = np.stack([inc[:, 0], inc[:, 1], -inc[:, 2]], axis=-1)
    spec = spec / np.linalg.norm(spec, axis=-1, keepdims=True)
    near_pole = np.abs(spec[:, 2]) >= 0.999
    up = np.where(near_pole[:, None],
                  np.array([1.0, 0.0, 0.0]),
                  np.array([0.0, 0.0, 1.0]))
    tangent = np.cross(up, spec)
    tangent = tangent / np.linalg.norm(tangent, axis=-1, keepdims=True)
    bitangent = np.cross(spec, tangent)
    dirs = (local[:, 0:1] * tangent
            + local[:, 1:2] * bitangent
            + local[:, 2:3] * spec)
    flip = dirs[:, 2] < 0
    dirs[flip] *= -1
    return dirs


def make_bsdf(spec: Optional[Union[BSDFModel, Dict[str, Any]]]) -> Optional[BSDFModel]:
    """Construct a BSDFModel from a dict spec, a BSDFModel, or None.

    Accepted forms
    --------------
    * ``BSDFModel`` instance  -- returned as-is
    * ``None``                -- returns ``None``
    * ``{'kind': 'lambertian', 'rho': 0.1}``
    * ``{'kind': 'gaussian', 'sigma_rad': 0.005,
         'scattered_fraction': 0.005}``
    * ``{'kind': 'harvey_shack', 'b0': 0.01, 'l': 0.01, 's': 2.0,
         'wavelength_ref': 633e-9, 'wavelength': 1310e-9}``

    The Harvey-Shack literature aliases ``A`` / ``B`` / ``C`` are accepted
    for ``b0`` / ``l`` / ``s`` (the :class:`HarveyShackBSDF` docstring
    teaches them), but not both spellings of the same parameter at once.

    Any unrecognised ``'kind'`` raises ``ValueError``, and so does any key
    the chosen kind does not consume -- a silently-ignored ``'sigma'`` or
    ``'scatter_fraction'`` builds a default lobe that can be 10x wider and
    50x weaker than the caller asked for.
    """
    if spec is None or isinstance(spec, BSDFModel):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(
            f"make_bsdf: expected dict, BSDFModel, or None -- got "
            f"{type(spec).__name__}")
    kind = spec.get('kind', '').lower()
    if kind == 'lambertian':
        _check_bsdf_keys(spec, kind, ('rho',))
        return LambertianBSDF(rho=spec.get('rho', 1.0))
    if kind == 'gaussian':
        _check_bsdf_keys(spec, kind, ('sigma_rad', 'scattered_fraction'))
        return GaussianBSDF(
            sigma_rad=spec.get('sigma_rad', 0.01),
            scattered_fraction=spec.get('scattered_fraction', 0.01))
    if kind == 'harvey_shack':
        aliases = {'A': 'b0', 'B': 'l', 'C': 's'}
        _check_bsdf_keys(
            spec, kind,
            ('b0', 'l', 's', 'wavelength_ref', 'wavelength'), aliases)
        resolved = {canon: spec[alias]
                    for alias, canon in aliases.items() if alias in spec}
        return HarveyShackBSDF(
            b0=spec.get('b0', resolved.get('b0', 1.0)),
            l=spec.get('l', resolved.get('l', 0.01)),
            s=spec.get('s', resolved.get('s', 2.0)),
            wavelength_ref=spec.get('wavelength_ref'),
            wavelength=spec.get('wavelength'))
    raise ValueError(
        f"make_bsdf: unknown kind {kind!r}. "
        f"Supported: 'lambertian', 'gaussian', 'harvey_shack'.")


def _check_bsdf_keys(spec, kind, accepted, aliases=None):
    """Reject dict keys the chosen BSDF kind would silently ignore.

    A spec key that no constructor argument consumes used to leave the
    corresponding parameter at its default with no diagnostic, so a
    mis-remembered name ('sigma' for 'sigma_rad') produced a plausible but
    wrong lobe.  ``aliases`` maps accepted alternative spellings to their
    canonical parameter; supplying both spellings of one parameter is an
    error rather than a silent precedence rule.
    """
    aliases = aliases or {}
    allowed = {'kind', *accepted, *aliases}
    unknown = sorted(set(spec) - allowed)
    if unknown:
        alias_note = ''
        if aliases:
            alias_note = (" Aliases: "
                          + ", ".join(f"{a} -> {c}"
                                      for a, c in sorted(aliases.items()))
                          + ".")
        raise ValueError(
            f"make_bsdf: unknown key(s) {unknown} for kind {kind!r}.  "
            f"Accepted: {sorted(accepted)}.{alias_note}  A key that is not "
            f"consumed would leave that parameter at its default "
            f"silently, so it is rejected instead.")
    for alias, canon in aliases.items():
        if alias in spec and canon in spec:
            raise ValueError(
                f"make_bsdf: {kind!r} spec sets both {alias!r} and its "
                f"canonical name {canon!r} "
                f"({spec[alias]!r} vs {spec[canon]!r}).  Pass one.")


def sample_scatter_rays(
    surface: Any,
    incident_rays: Any,
    n_per_ray: int = 1,
    rng: Optional[Union[int, np.random.Generator]] = None,
) -> Any:
    """Spawn scattered rays from a surface carrying a BSDF.

    Parameters
    ----------
    surface : :class:`raytrace.Surface`
        Must have a ``bsdf`` field (BSDFModel instance or dict spec).
    incident_rays : :class:`raytrace.RayBundle`
        Rays arriving at ``surface``; must already have been propagated
        to the surface (positions at the surface sag).
    n_per_ray : int, default 1
        Number of scattered rays per incident ray.
    rng : np.random.Generator or int, optional
        Seed or RNG for reproducibility.

    Returns
    -------
    scattered_rays : :class:`raytrace.RayBundle`
        A new bundle of ``incident_rays.x.size * n_per_ray`` rays
        starting at the incident positions with scattered direction
        cosines drawn from the BSDF.  Opd carried forward unchanged
        (the scatter happens at a point; no extra path length).

    Notes
    -----
    The returned rays can be fed directly into
    :func:`raytrace.trace` to propagate them through the remainder of
    the system for a stray-light analysis.  Use ``n_per_ray > 1`` for
    Monte Carlo stray-light integration.

    The whole bundle is drawn in ONE call: the lobe-local sample is
    incidence-independent (:meth:`BSDFModel._sample_local`) and the rotation
    onto each ray's specular direction is a batched 3x3 (see
    :func:`_rotate_local_to_specular`).  A consequence is that which random
    numbers land on which ray differs from a per-ray loop, so seeded output
    is not comparable ray-by-ray with pre-vectorisation runs; the sampled
    DISTRIBUTION is unchanged.

    A subclass that implements only the abstract :meth:`BSDFModel.sample`
    (the sole draw method the published ABC requires) keeps working: it has
    no batched hook, so this function falls back to the per-ray loop for it.
    All three shipped models override :meth:`BSDFModel._sample_local` and
    take the vectorised path.
    """
    from .. import raytrace as rt
    bsdf = make_bsdf(
        surface.bsdf if hasattr(surface, 'bsdf') else None)
    if bsdf is None:
        raise ValueError(
            "sample_scatter_rays: surface has no BSDF attached.")

    n_rays = incident_rays.x.size
    total = n_rays * n_per_ray
    rng = _get_rng(rng)
    if type(bsdf)._sample_local is BSDFModel._sample_local:
        # ``BSDFModel`` is a published extension point whose only required
        # draw method is ``sample``; a subclass written against that ABC has
        # no batched ``_sample_local``.  Vectorisation must not silently
        # become a compatibility requirement, so such a model takes the
        # per-ray loop this function used before it was vectorised.  The
        # three shipped models all override ``_sample_local`` and never
        # reach here.
        out_dirs = np.empty((total, 3), dtype=np.float64)
        for i in range(n_rays):
            inc_i = np.array([incident_rays.L[i],
                              incident_rays.M[i],
                              incident_rays.N[i]])
            out_dirs[i * n_per_ray:(i + 1) * n_per_ray] = bsdf.sample(
                inc_i, n_per_ray, rng=rng)
    else:
        local = bsdf._sample_local(total, rng)
        if bsdf._lobe_about_specular:
            inc = np.stack([np.repeat(incident_rays.L, n_per_ray),
                            np.repeat(incident_rays.M, n_per_ray),
                            np.repeat(incident_rays.N, n_per_ray)], axis=-1)
            out_dirs = _rotate_local_to_specular(local, inc)
        else:
            # Lobe defined about the surface normal: the local draw already
            # is the surface-frame direction (LambertianBSDF).
            out_dirs = np.asarray(local, dtype=np.float64)
    x = np.repeat(incident_rays.x, n_per_ray)
    y = np.repeat(incident_rays.y, n_per_ray)
    z = np.repeat(incident_rays.z, n_per_ray)
    opd = np.repeat(incident_rays.opd, n_per_ray)
    alive = np.ones(total, dtype=bool)
    return rt.RayBundle(
        x=x, y=y, z=z,
        L=out_dirs[:, 0], M=out_dirs[:, 1], N=out_dirs[:, 2],
        wavelength=incident_rays.wavelength,
        alive=alive, opd=opd)
