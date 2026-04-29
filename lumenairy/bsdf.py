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

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# =============================================================================
# Base class
# =============================================================================


class BSDFModel:
    """Abstract base.  Subclasses implement ``evaluate`` and ``sample``.

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

    def evaluate(self, incident_dir, scattered_dir):
        raise NotImplementedError

    def sample(self, incident_dir, n_samples, rng=None):
        raise NotImplementedError

    def total_integrated_scatter(self) -> float:
        """TIS = integral over outgoing hemisphere of BSDF * cos(theta) dOmega.

        Subclasses override when a closed form is available; the
        default falls back to a uniform-grid numerical integration
        which is always correct but slow.
        """
        n_theta = 256
        n_phi = 128
        theta = np.linspace(1e-6, np.pi / 2, n_theta)
        phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
        T, P = np.meshgrid(theta, phi, indexing='ij')
        inc = np.array([0.0, 0.0, -1.0])
        tot = 0.0
        dth = theta[1] - theta[0]
        dph = phi[1] - phi[0]
        for i in range(n_theta):
            for j in range(n_phi):
                s = np.array([
                    np.sin(theta[i]) * np.cos(phi[j]),
                    np.sin(theta[i]) * np.sin(phi[j]),
                    np.cos(theta[i]),
                ])
                b = float(self.evaluate(inc, s))
                tot += b * np.cos(theta[i]) * np.sin(theta[i]) * dth * dph
        return tot


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
    """
    rho: float = 1.0
    kind: str = 'lambertian'

    def evaluate(self, incident_dir, scattered_dir):
        sd = np.asarray(scattered_dir)
        if sd.ndim == 1:
            in_hemi = sd[2] > 0
        else:
            in_hemi = sd[..., 2] > 0
        return self.rho / np.pi * in_hemi

    def sample(self, incident_dir, n_samples, rng=None):
        rng = _get_rng(rng)
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

    def evaluate(self, incident_dir, scattered_dir):
        inc = np.asarray(incident_dir, dtype=float)
        sd = np.asarray(scattered_dir, dtype=float)
        # Specular direction = (L_i, M_i, -N_i) if we flip z;
        # here we assume incident has N<0 and scattered has N>0.
        specular = np.array([inc[..., 0] if inc.ndim else inc[0],
                             inc[..., 1] if inc.ndim else inc[1],
                             -inc[..., 2] if inc.ndim else -inc[2]])
        # cos(theta_s) between scattered and specular
        cos_theta = np.clip(np.sum(sd * specular, axis=-1), -1.0, 1.0)
        theta = np.arccos(cos_theta)
        A = self._normalization()
        in_hemi = sd[..., 2] > 0 if sd.ndim > 1 else sd[2] > 0
        return (A * np.exp(-theta ** 2 / (2 * self.sigma_rad ** 2))
                * in_hemi)

    def sample(self, incident_dir, n_samples, rng=None):
        rng = _get_rng(rng)
        # Sample offset angle theta ~ Gaussian truncated to [0, pi/2]
        # in the specular-frame, azimuth uniform.
        theta = np.abs(rng.normal(0, self.sigma_rad, n_samples))
        theta = np.minimum(theta, np.pi / 2 - 1e-6)
        phi = 2 * np.pi * rng.random(n_samples)
        # Local frame: specular = +z
        L_loc = np.sin(theta) * np.cos(phi)
        M_loc = np.sin(theta) * np.sin(phi)
        N_loc = np.cos(theta)
        # Rotate into surface frame.  Specular direction is
        # (Li, Mi, -Ni).  Build orthonormal basis around it.
        inc = np.asarray(incident_dir, dtype=float)
        spec = np.array([inc[0], inc[1], -inc[2]])
        spec = spec / np.linalg.norm(spec)
        if abs(spec[2]) < 0.999:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(up, spec); tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(spec, tangent)
        dirs = (L_loc[:, None] * tangent
                + M_loc[:, None] * bitangent
                + N_loc[:, None] * spec)
        # Force into outgoing hemisphere (z > 0)
        flip = dirs[:, 2] < 0
        dirs[flip] *= -1
        return dirs

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

    def evaluate(self, incident_dir, scattered_dir):
        inc = np.asarray(incident_dir, dtype=float)
        sd = np.asarray(scattered_dir, dtype=float)
        specular = np.array([inc[0], inc[1], -inc[2]])
        specular = specular / np.linalg.norm(specular)
        if sd.ndim == 1:
            cos_theta = np.clip(float(np.dot(sd, specular)), -1.0, 1.0)
            in_hemi = sd[2] > 0
        else:
            cos_theta = np.clip(np.sum(sd * specular, axis=-1), -1.0, 1.0)
            in_hemi = sd[..., 2] > 0
        sin_theta = np.sqrt(np.maximum(1 - cos_theta ** 2, 0.0))
        amp = self._amplitude()
        return amp / (1 + (sin_theta / self.l) ** 2) ** (self.s / 2) * in_hemi

    def sample(self, incident_dir, n_samples, rng=None):
        rng = _get_rng(rng)
        # Sample sin(theta) from the ABC radial profile via inverse-CDF.
        # PDF_radial ~ 1/(1 + (u/l)^2)^(s/2) * u  where u=sin(theta)
        # CDF from 0 to 1 (sin limit); use rejection sampling in the
        # interest of simplicity and robustness.
        accepted = []
        # Upper bound of radial weight (sin*BSDF) is achieved near sin ~ l
        peak_u = self.l / np.sqrt(max(self.s - 1, 1e-6))
        peak_val = peak_u / (1 + (peak_u / self.l) ** 2) ** (self.s / 2)
        while len(accepted) < n_samples:
            batch = min(4 * (n_samples - len(accepted)), 10000)
            u = rng.random(batch)  # candidate sin(theta) in [0,1]
            w = u / (1 + (u / self.l) ** 2) ** (self.s / 2)
            p = rng.random(batch) * peak_val
            sel = p < w
            accepted.extend(u[sel].tolist())
        sin_theta = np.array(accepted[:n_samples])
        phi = 2 * np.pi * rng.random(n_samples)
        cos_theta = np.sqrt(np.maximum(1 - sin_theta ** 2, 0.0))
        L_loc = sin_theta * np.cos(phi)
        M_loc = sin_theta * np.sin(phi)
        N_loc = cos_theta
        # Rotate into surface frame (same as Gaussian)
        inc = np.asarray(incident_dir, dtype=float)
        spec = np.array([inc[0], inc[1], -inc[2]])
        spec = spec / np.linalg.norm(spec)
        if abs(spec[2]) < 0.999:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([1.0, 0.0, 0.0])
        tangent = np.cross(up, spec); tangent /= np.linalg.norm(tangent)
        bitangent = np.cross(spec, tangent)
        dirs = (L_loc[:, None] * tangent
                + M_loc[:, None] * bitangent
                + N_loc[:, None] * spec)
        flip = dirs[:, 2] < 0
        dirs[flip] *= -1
        return dirs


# =============================================================================
# Builders and utilities
# =============================================================================


def _get_rng(rng):
    if rng is None:
        return np.random.default_rng()
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    return rng


def make_bsdf(spec) -> BSDFModel:
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

    Any unrecognised ``'kind'`` raises ``ValueError``.
    """
    if spec is None or isinstance(spec, BSDFModel):
        return spec
    if not isinstance(spec, dict):
        raise TypeError(
            f"make_bsdf: expected dict, BSDFModel, or None -- got "
            f"{type(spec).__name__}")
    kind = spec.get('kind', '').lower()
    if kind == 'lambertian':
        return LambertianBSDF(rho=spec.get('rho', 1.0))
    if kind == 'gaussian':
        return GaussianBSDF(
            sigma_rad=spec.get('sigma_rad', 0.01),
            scattered_fraction=spec.get('scattered_fraction', 0.01))
    if kind == 'harvey_shack':
        return HarveyShackBSDF(
            b0=spec.get('b0', 1.0),
            l=spec.get('l', 0.01),
            s=spec.get('s', 2.0),
            wavelength_ref=spec.get('wavelength_ref'),
            wavelength=spec.get('wavelength'))
    raise ValueError(
        f"make_bsdf: unknown kind {kind!r}. "
        f"Supported: 'lambertian', 'gaussian', 'harvey_shack'.")


def sample_scatter_rays(surface, incident_rays, n_per_ray=1, rng=None):
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
    """
    from . import raytrace as rt
    bsdf = make_bsdf(
        surface.bsdf if hasattr(surface, 'bsdf') else None)
    if bsdf is None:
        raise ValueError(
            "sample_scatter_rays: surface has no BSDF attached.")

    n_rays = incident_rays.x.size
    total = n_rays * n_per_ray
    rng = _get_rng(rng)
    out_dirs = np.empty((total, 3), dtype=np.float64)
    for i in range(n_rays):
        inc = np.array([incident_rays.L[i],
                        incident_rays.M[i],
                        incident_rays.N[i]])
        out_dirs[i * n_per_ray:(i + 1) * n_per_ray] = bsdf.sample(
            inc, n_per_ray, rng=rng)
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
