"""S3-10: backend-agnostic shared core for conic ray tracing.

Extracts the two primitives that the audit (S3-10, roadmap A3) found
implemented independently across the ray-trace strategies -- the conic
sag ``z(r)`` (plus its analytic radial derivatives) and the vector
Snell refraction / mirror reflection law -- into pure, no-in-place-write
functions callable from every backend:

* the NumPy exact Newton solver (``raytrace/intersection.py``),
* the differential ADRT dual / JAX path (``raytrace/differential.py``),
* the JAX-traceable kernels (``raytrace/jax_trace.py``).

Design constraints
------------------

* **Backend-agnostic.**  Array operations are *injected* by the caller
  -- ``xp`` for the sag; ``sqrt`` / ``where`` / ``val`` / ``tir_guard``
  for the Snell law -- so the identical arithmetic runs over NumPy
  arrays, the ADRT forward-mode ``_AdrtDual`` duals, and JAX tracers
  without the core importing any backend itself.
* **No in-place writes.**  Every function is pure and returns fresh
  values, so it is safe under ``jax.jit`` / ``jax.grad`` (which forbid
  mutation) and reusable by the dual operator overloads.
* **Byte-identical per call site.**  Each site keeps its OWN outer
  strategy (exact Newton / differential / traceable) and its own
  numerically load-bearing choices.  The two details that legitimately
  differ across sites are passed IN, so routing changes no bits:

  - the ``eta**2`` term -- ``intersection.py`` / ``jax_trace.py`` use
    the scalar power ``mu**2`` while ``differential.py`` uses ``eta*eta``,
    and IEEE-754 gives ``x**2 != x*x`` for ~0.05% of ratios -- is taken
    as the explicit ``eta_sq`` argument;
  - the total-internal-reflection ``sqrt`` guard -- the NumPy / ADRT
    paths clamp the negative radicand inside their own ``sqrt`` op,
    while the JAX kernels need the gradient-safe *double-where*
    (``where(tir, 1, disc)`` before the sqrt so ``jax.grad`` does not
    NaN-poison at the TIR boundary) -- is the injected ``tir_guard``.

  Verified against an independent hand oracle (forward bits AND
  ``jax.grad``) in ``tests/unit/test_s3_10_conic_core_shared.py``.

The paraxial site (``raytrace/seidel.py``) is intentionally NOT routed:
it carries no conic sag ``z(r)`` and no vector Snell law -- only the
first-order linearised refraction ``nu' = nu - y*(n2 - n1)*c`` in
reduced coordinates -- so there is no sag / Snell evaluation to share
without converting paraxial -> exact, which the roadmap explicitly
forbids ("must not impose the exact-solver's Newton loop on the
paraxial site").  The differential ADRT intersection likewise keeps its
implicit-conic quadratic (``F = c*r^2 - 2z + (1+k)*c*z^2 = 0``); only
its Snell / reflection step routes through the shared core here.
"""
from __future__ import annotations


def _identity(a):
    """Default ``val`` op: arrays already support the ``>`` / ``<``
    comparisons the Snell law needs; only the ADRT dual overrides this
    (to read ``.v`` off the dual)."""
    return a


def _identity_tir_guard(tir, disc_r):
    """Default ``tir_guard`` op: a no-op.

    The NumPy exact solver and the ADRT dual clamp the (possibly
    negative) radicand at zero *inside* their own ``sqrt`` op, so no
    pre-sqrt masking is required.  Only the JAX kernels override this
    with the gradient-safe double-where.
    """
    return disc_r


def refract_snell(L, M, N, nx, ny, nz, eta, eta_sq,
                  *, sqrt, where, val=_identity, tir_guard=_identity_tir_guard):
    """Vector Snell refraction with the surface normal oriented against
    the incoming ray.

    Parameters
    ----------
    L, M, N :
        Incident unit direction cosines.
    nx, ny, nz :
        Surface unit normal at the intersection point (either
        orientation; it is flipped here to point into the incident
        medium).
    eta :
        Index ratio ``n1 / n2``.
    eta_sq :
        ``eta`` squared, computed by the caller in whatever way its
        former inline code did (scalar ``mu**2`` power vs ``eta*eta``
        product) so the routing is bit-for-bit -- see the module
        docstring.
    sqrt :
        Backend square-root that clamps a negative argument at zero
        (``lambda a: sqrt(maximum(a, 0))`` for NumPy / JAX; the ADRT
        ``_dual_sqrt`` clamps its value channel).
    where, val, tir_guard :
        Injected backend ops (see module docstring).

    Returns
    -------
    tuple
        ``(Lp, Mp, Np, nx, ny, nz, cos_i, disc_r, tir)`` -- refracted
        direction cosines, the oriented normal, the (non-negative)
        cosine of incidence, the refraction radicand
        ``1 - eta^2 (1 - cos_i^2)``, and the TIR mask
        ``disc_r < 0``.  The caller applies its own TIR / vignette /
        renormalisation policy.
    """
    dn = L * nx + M * ny + N * nz
    fl = where(val(dn) > 0.0, -1.0, 1.0)
    nx = nx * fl
    ny = ny * fl
    nz = nz * fl
    cos_i = 0.0 - (L * nx + M * ny + N * nz)
    disc_r = 1.0 - eta_sq * (1.0 - cos_i * cos_i)
    tir = val(disc_r) < 0.0
    root = sqrt(tir_guard(tir, disc_r))
    # The ``(eta*cos_i - root)`` coefficient is written inline per axis
    # (NOT factored into one variable) so the reverse-mode autodiff graph
    # matches the JAX site's former inline block exactly -- factoring it
    # is bit-identical on the forward pass but reassociates the gradient
    # accumulation by ~1 ULP (see the shared-core test's grad check).
    Lp = eta * L + (eta * cos_i - root) * nx
    Mp = eta * M + (eta * cos_i - root) * ny
    Np = eta * N + (eta * cos_i - root) * nz
    return Lp, Mp, Np, nx, ny, nz, cos_i, disc_r, tir


def reflect_mirror(L, M, N, nx, ny, nz, *, where, val=_identity):
    """Vector mirror reflection with the surface normal oriented against
    the incoming ray.

    Returns
    -------
    tuple
        ``(Lp, Mp, Np, nx, ny, nz, cos_i)`` -- reflected direction
        cosines, the oriented normal, and the cosine of incidence.
    """
    dn = L * nx + M * ny + N * nz
    fl = where(val(dn) > 0.0, -1.0, 1.0)
    nx = nx * fl
    ny = ny * fl
    nz = nz * fl
    cos_i = 0.0 - (L * nx + M * ny + N * nz)
    two_ci = 2.0 * cos_i
    Lp = L + two_ci * nx
    Mp = M + two_ci * ny
    Np = N + two_ci * nz
    return Lp, Mp, Np, nx, ny, nz, cos_i


def conic_sag(x, y, R, conic, asph_items, *, xp):
    """Rotationally-symmetric conic + even-aspheric surface sag ``z(r)``.

    ``z = h^2 / (R (1 + sqrt(1 - (1+k) h^2 / R^2))) + sum a_p h^p`` with
    ``h^2 = x^2 + y^2``.  Outside the conic domain (``(1+k) h^2/R^2 >=
    0.9999``) and for a flat surface (``R`` infinite or ``|R| > 1e15``)
    the conic term is zeroed -- the finite, gradient-safe convention the
    JAX backends use (the NumPy wave-optics ``surface_sag_general`` fills
    NaN instead; the two are intentionally distinct, so they are NOT
    merged).

    Parameters
    ----------
    x, y :
        Transverse coordinates on the surface.
    R :
        Radius of curvature (may be ``inf`` or a JAX tracer).
    conic :
        Conic constant ``k``.
    asph_items :
        Iterable of ``(power, coeff)`` even-aspheric pairs; ``coeff``
        may be a JAX scalar (so ``jax.grad`` flows through it).
    xp :
        Array module exposing ``where`` / ``isinf`` / ``abs`` /
        ``sqrt`` / ``zeros_like`` (``numpy`` or ``jax.numpy``).
    """
    h_sq = x * x + y * y
    R_finite = xp.where(xp.isinf(R), 1e30, R)
    R_finite = xp.where(xp.abs(R_finite) < 1e-30, 1e-30, R_finite)
    norm = (1.0 + conic) * h_sq / (R_finite * R_finite)
    valid = norm < 0.9999
    denom_arg = xp.where(valid, 1.0 - norm, 0.01)
    conic_term = xp.where(
        valid, h_sq / (R_finite * (1.0 + xp.sqrt(denom_arg))), 0.0)
    is_flat = xp.isinf(R) | (xp.abs(R) > 1e15)
    sag = xp.where(is_flat, xp.zeros_like(h_sq), conic_term)
    for power, coeff in asph_items:
        sag = sag + coeff * h_sq ** (power // 2)
    return sag


def conic_sag_derivs(x, y, R, conic, asph_items, *, xp):
    """Analytic transverse derivatives ``(dz/dx, dz/dy)`` of
    :func:`conic_sag`.

    ``dz/dh = h / (R sqrt(1 - (1+k) h^2/R^2))`` distributed onto x / y,
    with the sign following ``sign(R)`` (positive for convex, negative
    for concave), plus the aspheric contributions.  Flat / infinite
    ``R`` gives a zero conic gradient.  ``xp`` and ``asph_items`` are as
    for :func:`conic_sag`.
    """
    h_sq = x * x + y * y
    R_finite = xp.where(xp.isinf(R), 1e30, R)
    R_finite = xp.where(xp.abs(R_finite) < 1e-30, 1e-30, R_finite)
    denom = R_finite * R_finite - (1.0 + conic) * h_sq
    denom_safe = xp.where(denom > 1e-30, denom, 1e-30)
    sd = xp.sqrt(denom_safe)
    R_sign = xp.where(R >= 0.0, 1.0, -1.0)
    zx_conic = R_sign * x / sd
    zy_conic = R_sign * y / sd
    is_flat = xp.isinf(R) | (xp.abs(R) > 1e15)
    zx = xp.where(is_flat, xp.zeros_like(x), zx_conic)
    zy = xp.where(is_flat, xp.zeros_like(y), zy_conic)
    for power, coeff in asph_items:
        if power == 2:
            zx = zx + 2.0 * coeff * x
            zy = zy + 2.0 * coeff * y
        else:
            scale = coeff * power * h_sq ** ((power - 2) // 2)
            zx = zx + scale * x
            zy = zy + scale * y
    return zx, zy


__all__ = ['refract_snell', 'reflect_mirror', 'conic_sag', 'conic_sag_derivs']
