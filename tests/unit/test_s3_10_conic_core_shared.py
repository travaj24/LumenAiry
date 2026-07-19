"""S3-10 (roadmap A3): backend-agnostic shared conic-sag + vector-Snell core.

The audit found the conic sag ``z(r)`` and the vector Snell / reflection
law implemented independently across the four ray-trace strategies
(``intersection.py`` exact Newton, ``differential.py`` ADRT,
``seidel.py`` paraxial, ``jax_trace.py`` traceable).  The remediation
extracts a small backend-agnostic shared core
(``raytrace._conic_core``) that each *non-paraxial* site routes through
while keeping its own outer strategy.

This test enforces two independent guarantees:

* **Physics correctness** -- the shared core matches an INDEPENDENT
  hand oracle (Snell's law via tangential/longitudinal decomposition;
  reflection via ``d - 2 (d.n) n``; the conic sag via the *implicit*
  conic quadratic root; sag derivatives via finite differences).  This
  proves the routing did not byte-match a latent bug.

* **Byte-identity per call site** -- the shared core reproduces each
  site's FORMER inline formula (hand-copied here from the pre-S3-10
  code) to the bit, including the TIR mask, on representative and edge
  inputs (TIR, grazing, backward post-mirror rays, flat / spherical /
  conic surfaces).  The two details that legitimately differ across
  sites -- the ``eta**2`` term (scalar power vs product) and the JAX
  gradient-safe double-where -- are passed in by each caller, so the
  routing changes no bits.  ``jax.grad`` through the shared refract is
  additionally shown finite and bit-identical to the former block.
"""
import numpy as np
import pytest

from lumenairy.raytrace._conic_core import (
    conic_sag,
    conic_sag_derivs,
    reflect_mirror,
    refract_snell,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _bits_equal(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    assert a.shape == b.shape, (a.shape, b.shape)
    va = a.view(np.uint64)
    vb = b.view(np.uint64)
    eq = (va == vb) | (np.isnan(a) & np.isnan(b))
    if not eq.all():
        idx = np.where(~eq)[0][0]
        raise AssertionError(
            f"bit mismatch at {idx}: {a.ravel()[idx]!r} vs {b.ravel()[idx]!r}")


def _rays(seed, n=4000, backward=False, grazing=False):
    rng = np.random.default_rng(seed)
    if grazing:
        L = rng.uniform(-0.96, 0.96, n)
        M = rng.uniform(-0.15, 0.15, n)
    else:
        L = rng.uniform(-0.6, 0.6, n)
        M = rng.uniform(-0.6, 0.6, n)
    N = np.sqrt(np.maximum(1.0 - L * L - M * M, 1e-9))
    if backward:
        N = -N
    nx = rng.uniform(-0.45, 0.45, n)
    ny = rng.uniform(-0.45, 0.45, n)
    nz = np.sqrt(np.maximum(1.0 - nx * nx - ny * ny, 1e-9))
    mag = np.sqrt(nx * nx + ny * ny + nz * nz)
    return L, M, N, nx / mag, ny / mag, nz / mag


_NP_SQRT = lambda a: np.sqrt(np.maximum(a, 0.0))  # noqa: E731


# ---------------------------------------------------------------------------
# INDEPENDENT physics oracles
# ---------------------------------------------------------------------------
def _oracle_refract(L, M, N, nx, ny, nz, n1, n2):
    """Snell's law by tangential/longitudinal decomposition (independent
    of the shared core's ``eta*d + (eta*cos_i - cos_t)*n`` form)."""
    d = np.stack([L, M, N], axis=-1)
    nrm = np.stack([nx, ny, nz], axis=-1)
    # orient normal against the ray
    dot = np.sum(d * nrm, axis=-1)
    n_or = np.where((dot > 0)[:, None], -nrm, nrm)
    cos_i = -np.sum(d * n_or, axis=-1)          # >= 0
    eta = n1 / n2
    dt = d + cos_i[:, None] * n_or              # tangential comp, |dt| = sin_i
    sin_i = np.linalg.norm(dt, axis=-1)
    sin_t = eta * sin_i
    tir = sin_t > 1.0
    cos_t = np.sqrt(np.maximum(1.0 - sin_t * sin_t, 0.0))
    with np.errstate(invalid="ignore", divide="ignore"):
        that = np.where(sin_i[:, None] > 1e-12, dt / sin_i[:, None], 0.0)
    d_refr = sin_t[:, None] * that - cos_t[:, None] * n_or
    return d_refr, tir


def _oracle_reflect(L, M, N, nx, ny, nz):
    """Reflection via ``d - 2 (d.n) n`` (orientation-independent)."""
    d = np.stack([L, M, N], axis=-1)
    nrm = np.stack([nx, ny, nz], axis=-1)
    dot = np.sum(d * nrm, axis=-1)[:, None]
    return d - 2.0 * dot * nrm


def _oracle_sag(x, y, R, conic, asph):
    """Conic sag from the IMPLICIT conic quadratic root
    ``(1+k) c z^2 - 2 z + c r^2 = 0`` (a different algebraic route than
    the shared core's rationalised ``c r^2 / (1 + sqrt(...))`` form),
    plus aspheric terms."""
    r2 = x * x + y * y
    if np.isinf(R):
        z = np.zeros_like(r2)
    else:
        c = 1.0 / R
        A = (1.0 + conic) * c
        disc = 1.0 - (1.0 + conic) * c * c * r2
        disc = np.maximum(disc, 0.0)
        # near-vertex root of A z^2 - 2 z + c r^2 = 0
        if abs(A) < 1e-30:
            z = 0.5 * c * r2                      # parabola limit (k = -1)
        else:
            z = (1.0 - np.sqrt(disc)) / A
    for p, a in asph:
        z = z + a * r2 ** (p // 2)
    return z


# ---------------------------------------------------------------------------
# FORMER inline formulas (hand-copied from the pre-S3-10 code)
# ---------------------------------------------------------------------------
def _former_intersection_refract(L, M, N, nx, ny, nz, n1, n2):
    cos_i = L * nx + M * ny + N * nz
    flip = cos_i > 0
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    nz = np.where(flip, -nz, nz)
    cos_i = np.abs(cos_i)
    mu = n1 / n2
    sin2_t = mu ** 2 * (1.0 - cos_i ** 2)
    tir = sin2_t > 1.0
    cos_t = np.sqrt(np.maximum(1.0 - sin2_t, 0.0))
    factor = mu * cos_i - cos_t
    return mu * L + factor * nx, mu * M + factor * ny, mu * N + factor * nz, tir


def _former_intersection_reflect(L, M, N, nx, ny, nz):
    cos_i = L * nx + M * ny + N * nz
    flip = cos_i > 0
    nx = np.where(flip, -nx, nx)
    ny = np.where(flip, -ny, ny)
    nz = np.where(flip, -nz, nz)
    cos_i = np.abs(cos_i)
    return L + 2.0 * cos_i * nx, M + 2.0 * cos_i * ny, N + 2.0 * cos_i * nz


def _former_differential_refract(L, M, N, nx, ny, nz, n1, n2):
    dn = L * nx + M * ny + N * nz
    fl = np.where(dn > 0.0, -1.0, 1.0)
    nx = nx * fl
    ny = ny * fl
    nz = nz * fl
    cos_i = 0.0 - (L * nx + M * ny + N * nz)
    eta = n1 / n2
    disc_r = 1.0 - (eta * eta) * (1.0 - cos_i * cos_i)
    root = np.sqrt(np.maximum(disc_r, 0.0))
    coef = eta * cos_i - root
    return (eta * L + coef * nx, eta * M + coef * ny, eta * N + coef * nz,
            disc_r < 0.0)


# ---------------------------------------------------------------------------
# 1. shared core vs INDEPENDENT physics oracle
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{}, {"backward": True}, {"grazing": True}])
def test_refract_matches_independent_oracle(kw):
    L, M, N, nx, ny, nz = _rays(1, **kw)
    n1, n2 = (1.6, 1.0) if kw.get("grazing") else (1.0, 1.5168)
    mu = n1 / n2
    Lp, Mp, Np, *_, tir = refract_snell(
        L, M, N, nx, ny, nz, mu, mu ** 2, sqrt=_NP_SQRT, where=np.where)
    d_or, tir_or = _oracle_refract(L, M, N, nx, ny, nz, n1, n2)
    assert np.array_equal(tir, tir_or)
    live = ~tir
    got = np.stack([Lp, Mp, Np], axis=-1)[live]
    assert np.allclose(got, d_or[live], atol=1e-13, rtol=0)
    # refracted rays are unit vectors
    assert np.allclose(np.linalg.norm(got, axis=-1), 1.0, atol=1e-12)


@pytest.mark.parametrize("kw", [{}, {"backward": True}])
def test_reflect_matches_independent_oracle(kw):
    L, M, N, nx, ny, nz = _rays(2, **kw)
    Lp, Mp, Np, *_ = reflect_mirror(L, M, N, nx, ny, nz, where=np.where)
    d_or = _oracle_reflect(L, M, N, nx, ny, nz)
    got = np.stack([Lp, Mp, Np], axis=-1)
    assert np.allclose(got, d_or, atol=1e-14, rtol=0)
    assert np.allclose(np.linalg.norm(got, axis=-1), 1.0, atol=1e-13)


@pytest.mark.parametrize("R", [0.05, -0.08, 0.5, np.inf])
@pytest.mark.parametrize("conic", [0.0, -1.0, -0.6, 2.0])
def test_conic_sag_matches_independent_oracle(R, conic):
    rng = np.random.default_rng(3)
    x = rng.uniform(-0.01, 0.01, 3000)
    y = rng.uniform(-0.01, 0.01, 3000)
    asph = ((4, 1.5e3), (6, -2.0e5))
    got = conic_sag(x, y, R, conic, asph, xp=np)
    ref = _oracle_sag(x, y, R, conic, asph)
    assert np.allclose(got, ref, atol=1e-12, rtol=1e-10)


@pytest.mark.parametrize("R", [0.05, -0.08])
def test_conic_sag_derivs_match_finite_difference(R):
    rng = np.random.default_rng(4)
    x = rng.uniform(-0.008, 0.008, 2000)
    y = rng.uniform(-0.008, 0.008, 2000)
    conic = -0.4
    asph = ((4, 8.0e2),)
    zx, zy = conic_sag_derivs(x, y, R, conic, asph, xp=np)
    h = 1e-7
    fd_x = (conic_sag(x + h, y, R, conic, asph, xp=np)
            - conic_sag(x - h, y, R, conic, asph, xp=np)) / (2 * h)
    fd_y = (conic_sag(x, y + h, R, conic, asph, xp=np)
            - conic_sag(x, y - h, R, conic, asph, xp=np)) / (2 * h)
    assert np.allclose(zx, fd_x, atol=1e-5, rtol=1e-5)
    assert np.allclose(zy, fd_y, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# 2. byte-identity vs each site's FORMER inline formula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [{}, {"backward": True}, {"grazing": True}])
def test_bit_identical_intersection_refract(kw):
    L, M, N, nx, ny, nz = _rays(10, **kw)
    n1, n2 = (1.7, 1.0) if kw.get("grazing") else (1.0, 1.5168)
    mu = n1 / n2
    Lp, Mp, Np, *_, tir = refract_snell(
        L, M, N, nx, ny, nz, mu, mu ** 2, sqrt=_NP_SQRT, where=np.where)
    fL, fM, fN, ftir = _former_intersection_refract(L, M, N, nx, ny, nz, n1, n2)
    # the site discards the (garbage) refracted dir on TIR rays via
    # ``where(alive, ...)``; compare the surviving rays bit-for-bit
    live = ~tir
    assert np.array_equal(tir, ftir)
    _bits_equal(Lp[live], fL[live])
    _bits_equal(Mp[live], fM[live])
    _bits_equal(Np[live], fN[live])


@pytest.mark.parametrize("kw", [{}, {"backward": True}])
def test_bit_identical_intersection_reflect(kw):
    L, M, N, nx, ny, nz = _rays(11, **kw)
    Lp, Mp, Np, *_ = reflect_mirror(L, M, N, nx, ny, nz, where=np.where)
    fL, fM, fN = _former_intersection_reflect(L, M, N, nx, ny, nz)
    _bits_equal(Lp, fL)
    _bits_equal(Mp, fM)
    _bits_equal(Np, fN)


@pytest.mark.parametrize("kw", [{}, {"backward": True}, {"grazing": True}])
def test_bit_identical_differential_refract(kw):
    L, M, N, nx, ny, nz = _rays(12, **kw)
    n1, n2 = (1.7, 1.0) if kw.get("grazing") else (1.0, 1.5168)
    eta = n1 / n2
    Lp, Mp, Np, *_, disc_r, tir = refract_snell(
        L, M, N, nx, ny, nz, eta, eta * eta, sqrt=_NP_SQRT, where=np.where)
    fL, fM, fN, ftir = _former_differential_refract(L, M, N, nx, ny, nz, n1, n2)
    # the differential site produces (and later masks) the refracted dir
    # for ALL rays -- byte-identical everywhere, not just the live subset
    _bits_equal(Lp, fL)
    _bits_equal(Mp, fM)
    _bits_equal(Np, fN)
    assert np.array_equal(tir, ftir)


def test_eta_sq_form_matters():
    """Guard the reason ``eta_sq`` is caller-supplied: the scalar power
    and product forms genuinely disagree for some index ratios, so a
    shared core hard-coding one would break byte-identity at the other
    site."""
    # find an index ratio where mu**2 != mu*mu
    rng = np.random.default_rng(99)
    found = None
    for _ in range(500000):
        x = rng.uniform(0.3, 0.9)
        if (x ** 2) != (x * x):
            found = x
            break
    assert found is not None, "expected some ratio with x**2 != x*x"


# ---------------------------------------------------------------------------
# 3. JAX: forward bit-identity + gradient preservation
# ---------------------------------------------------------------------------
def test_jax_refract_forward_and_grad():
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def former_jax(L, M, N, nx, ny, nz, n1, n2):
        dot = L * nx + M * ny + N * nz
        flip = dot > 0
        nx = jnp.where(flip, -nx, nx)
        ny = jnp.where(flip, -ny, ny)
        nz = jnp.where(flip, -nz, nz)
        eta = n1 / n2
        cos_i = -(L * nx + M * ny + N * nz)
        sin2_t = eta ** 2 * (1.0 - cos_i ** 2)
        tir = sin2_t > 1.0
        sin2_t_safe = jnp.where(tir, 0.0, sin2_t)
        cos_t = jnp.sqrt(jnp.maximum(1.0 - sin2_t_safe, 0.0))
        Lt_r = eta * L + (eta * cos_i - cos_t) * nx
        Mt_r = eta * M + (eta * cos_i - cos_t) * ny
        Nt_r = eta * N + (eta * cos_i - cos_t) * nz
        Lt = jnp.where(tir, L, Lt_r)
        Mt = jnp.where(tir, M, Mt_r)
        Nt = jnp.where(tir, N, Nt_r)
        return Lt, Mt, Nt, tir

    def routed_jax(L, M, N, nx, ny, nz, n1, n2):
        eta = n1 / n2
        Lt_r, Mt_r, Nt_r, *_ignore, tir = refract_snell(
            L, M, N, nx, ny, nz, eta, eta ** 2,
            sqrt=lambda z: jnp.sqrt(jnp.maximum(z, 0.0)),
            where=jnp.where,
            tir_guard=lambda t, d: jnp.where(t, 1.0, d))
        Lt = jnp.where(tir, L, Lt_r)
        Mt = jnp.where(tir, M, Mt_r)
        Nt = jnp.where(tir, N, Nt_r)
        return Lt, Mt, Nt, tir

    # grazing / dense-to-rare so some rays TIR (exercise the double-where)
    L, M, N, nx, ny, nz = _rays(20, n=800, grazing=True)
    args = tuple(jnp.asarray(a) for a in (L, M, N, nx, ny, nz))
    n1, n2 = 1.6, 1.0
    fL, fM, fN, ft = former_jax(*args, n1, n2)
    rL, rM, rN, rt = routed_jax(*args, n1, n2)
    assert np.array_equal(np.asarray(ft), np.asarray(rt))
    _bits_equal(np.asarray(fL), np.asarray(rL))
    _bits_equal(np.asarray(fM), np.asarray(rM))
    _bits_equal(np.asarray(fN), np.asarray(rN))

    # gradient of a scalar loss wrt the incident L must be finite AND
    # bit-identical (the double-where guard is the reason it stays finite
    # at the TIR boundary)
    def loss_former(Lv):
        o = former_jax(Lv, args[1], args[2], args[3], args[4], args[5], n1, n2)
        return jnp.sum(o[0] + o[1] + o[2])

    def loss_routed(Lv):
        o = routed_jax(Lv, args[1], args[2], args[3], args[4], args[5], n1, n2)
        return jnp.sum(o[0] + o[1] + o[2])

    gf = np.asarray(jax.grad(loss_former)(args[0]))
    gr = np.asarray(jax.grad(loss_routed)(args[0]))
    assert np.isfinite(gf).all() and np.isfinite(gr).all()
    _bits_equal(gf, gr)


# ---------------------------------------------------------------------------
# 4. end-to-end: NumPy trace and trace_jax still agree (routed both sides)
# ---------------------------------------------------------------------------
def test_end_to_end_numpy_vs_jax_parity():
    pytest.importorskip("jax")
    import jax
    jax.config.update("jax_enable_x64", True)

    from lumenairy.raytrace import surfaces_from_prescription, trace
    from lumenairy.raytrace.jax_trace import (
        make_jax_ray_state,
        trace_jax,
    )
    from lumenairy.raytrace.trace import _make_bundle

    rx = {
        "surfaces": [
            {"radius": 0.05, "conic": -0.3, "aspheric_coeffs": {4: 1.0e3},
             "glass_before": "air", "glass_after": "N-BK7"},
            {"radius": -0.06, "conic": 0.0,
             "glass_before": "N-BK7", "glass_after": "air"},
        ],
        "thicknesses": [0.004],
        "aperture_diameter": 0.02,
    }
    wl = 1.31e-6
    surfs = surfaces_from_prescription(rx)
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.006, 0.006, 64)
    y = rng.uniform(-0.006, 0.006, 64)
    z = np.zeros_like(x)
    L = np.zeros_like(x)
    M = np.zeros_like(x)
    Nn = np.ones_like(x)

    rb = _make_bundle(x, y, L, M, wl)
    res = trace(rb, surfs, wl)
    img = res.image_rays

    st = make_jax_ray_state(x, y, z, L, M, Nn)
    jres = trace_jax(st, rx, wl)

    live = img.alive & np.asarray(jres.alive)
    assert live.sum() > 40
    assert np.allclose(np.asarray(jres.x)[live], img.x[live], atol=1e-9)
    assert np.allclose(np.asarray(jres.y)[live], img.y[live], atol=1e-9)
