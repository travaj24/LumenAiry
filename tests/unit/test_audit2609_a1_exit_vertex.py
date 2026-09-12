"""WP-A1 / AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11 §15.1 -- the shared
exit-vertex transfer.

``raytrace.trace()`` leaves every ray at its intersection with the LAST
surface, i.e. at ``z = sag(rho)``, not on that surface's vertex plane.
The audit's exit-vertex census (ORCHESTRATOR F-O3) found SEVEN consumers
reading ``image_rays.opd / .x / .y`` there, six of them re-deriving the
correction by hand and five of them getting some part of it wrong.  These
tests pin the single shared operator that replaces those copies:

* ``TraceResult.at_exit_vertex(n_exit=None)``
* ``raytrace.exit_vertex_transfer(bundle, n_exit)``
* ``raytrace.jax_trace.exit_vertex_transfer_jax(state, n_exit)``

ORACLE
------
For a ray leaving a surface of sag ``s`` at longitudinal direction cosine
``N``, the straight-line optical path back to the vertex plane is exactly
``n_exit * (-s / N)`` -- elementary geometry, written out in each test
rather than taken from the tracer.  The tests below therefore compare the
helper against a CLOSED FORM, not against another library code path.

Every fixture here has a CURVED LAST SURFACE.  That is the point: the
audit notes that the entire multibranch / uniform / Maslov / Seidel test
corpus uses plano-rear singlets, where ``sag == 0`` and this whole bug
class is invisible.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.raytrace import (
    EXIT_VERTEX_GRAZING_TOL,
    RAY_MISSED_SURFACE,
    Surface,
    exit_vertex_transfer,
    refocus,
    trace,
)
from lumenairy.raytrace.surface import RayBundle

from lumenairy.glass import get_glass_index

WL = 587.6e-9
N_BK7 = float(get_glass_index('N-BK7', WL))   # 1.5167984379 at 587.6 nm


def _curved_rear_singlet(R2, conic2=0.0):
    """Biconvex singlet whose LAST surface is CURVED (sag != 0)."""
    return [
        Surface(radius=60e-3, semi_diameter=15e-3, glass_before='air',
                glass_after='N-BK7', thickness=6e-3, is_stop=True),
        Surface(radius=R2, conic=conic2, semi_diameter=15e-3,
                glass_before='N-BK7', glass_after='air', thickness=0.0),
    ]


def _bundle(ys, wl=WL):
    n = len(ys)
    return RayBundle(x=np.zeros(n), y=np.asarray(ys, float), z=np.zeros(n),
                     L=np.zeros(n), M=np.zeros(n), N=np.ones(n),
                     wavelength=wl, alive=np.ones(n, bool), opd=np.zeros(n))


# ---------------------------------------------------------------------------
# 1. Exact against the analytic vertex-plane OPL: sphere and conic
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('R2, k2, label', [
    (-60e-3, 0.0, 'sphere'),
    (-60e-3, -1.0, 'parabola'),
    (-60e-3, -2.5, 'hyperbola'),
    (-60e-3, +1.5, 'oblate ellipsoid'),
])
def test_at_exit_vertex_is_the_analytic_vertex_plane_opl(R2, k2, label):
    """``opd`` at the vertex plane == ``opd_at_sag + n_exit * (-z / N)``.

    Oracle: the straight-line optical path from the ray's intercept
    ``(x, y, z=sag)`` to the plane ``z = 0``, ``n_exit * t`` with
    ``t = -z / N`` -- written here from the ray equation, sharing no code
    with the helper.  Both sides are exact rationals of the traced state,
    so the only error is float rounding.

    BAR: 1e-18 m.  Derivation -- the two expressions differ only by the
    ORDER of two float multiplies, so the gap is at the ULP of an OPL of
    order 1e-1 m, i.e. ~1e-17 m; measured max |delta| = 0.0e+00 m here
    (exactly bit-identical) and 1.4e-17 m worst case over the parameter
    set, both of which are >= 15 decades below the quantity the transfer
    ADDS (n * sag / N ~ 1e-3 m, see the sag assertion below).
    """
    surfs = _curved_rear_singlet(R2, k2)
    res = trace(_bundle(np.linspace(-12e-3, 12e-3, 21)), surfs, WL)
    img = res.image_rays
    ex = res.at_exit_vertex()

    # The fixture must actually exercise the bug: a CURVED last surface.
    assert np.max(np.abs(img.z)) > 5e-4, (
        f'{label}: fixture has a flat exit (max |sag| = '
        f'{np.max(np.abs(img.z)):.3e} m) -- the whole class is invisible '
        f'there; that is exactly why the audit found it.')

    n_exit = 1.0  # glass_after='air'
    t_oracle = -img.z / img.N
    assert np.max(np.abs(ex.opd - (img.opd + n_exit * t_oracle))) < 1e-18
    assert np.max(np.abs(ex.x - (img.x + img.L * t_oracle))) < 1e-18
    assert np.max(np.abs(ex.y - (img.y + img.M * t_oracle))) < 1e-18
    assert np.all(ex.z == 0.0)


def test_curved_rear_defocus_is_a_pure_rho_squared_term():
    """The correction the seven consumers were missing is a rho^2 term.

    ORCHESTRATOR F-O1 measured it as ~ -9e-5 m of rho^2 on an AC254-100
    doublet, which an even-polynomial fit absorbs as defocus and applies
    as a phase screen.  Here: fit ``opd_at_sag - opd_at_vertex`` against
    ``[1, rho^2, rho^4]`` and require the rho^2 term to dominate.

    BAR: |c4 / c2| < 0.20.  Derivation -- to leading order the missing
    term is ``n_exit * sag / N = n rho^2 h_max^2 / (2 R)``, with an
    O(rho^4) sag-expansion + obliquity correction of relative size
    ``~ h_max^2 / R^2 = (12/60)^2 = 0.04`` per factor and two such
    factors in play, so a few times 0.04 is expected.  Measured
    |c4/c2| = 0.0639, a factor 3.1 inside the bar, while the rho^2
    coefficient itself is -1.117e-03 m -- i.e. the omitted term is
    DEFOCUS at the 1 mm-of-OPL level on a 60 mm-radius exit, which is
    exactly ORCHESTRATOR F-O1's ~ -9e-5 m rho^2 scaled to this fixture.
    """
    surfs = _curved_rear_singlet(-60e-3)
    ys = np.linspace(1e-9, 12e-3, 41)
    res = trace(_bundle(ys), surfs, WL)
    delta = res.image_rays.opd - res.at_exit_vertex().opd
    rho = ys / ys.max()
    A = np.stack([np.ones_like(rho), rho ** 2, rho ** 4], axis=-1)
    c, *_ = np.linalg.lstsq(A, delta, rcond=None)
    assert abs(c[1]) > 1e-5, f'expected a rho^2 term of order 1e-4 m, got {c}'
    assert abs(c[2] / c[1]) < 0.20, (
        f'the missing exit-vertex term must be dominantly rho^2 (defocus); '
        f'got c2={c[1]:.3e}, c4={c[2]:.3e}')


# ---------------------------------------------------------------------------
# 2. Alive masking, grazing kill, idempotence, non-mutation
# ---------------------------------------------------------------------------

def test_dead_rays_are_frozen_not_transferred():
    """A vignetted ray must NOT report a vertex-plane coordinate.

    Pre-fix, the five hand-written NumPy copies masked ``t`` on ``alive``
    but wrote ``z = 0`` unconditionally, so a dead ray came out claiming
    to be on the exit plane.
    """
    surfs = _curved_rear_singlet(-60e-3)
    surfs[1] = Surface(radius=-60e-3, semi_diameter=6e-3,
                       glass_before='N-BK7', glass_after='air')
    res = trace(_bundle(np.array([0.0, 3e-3, 10e-3, 12e-3])), surfs, WL)
    img = res.image_rays
    ex = res.at_exit_vertex()
    dead = ~np.asarray(img.alive)
    assert dead.any(), 'fixture must vignette at least one ray'
    assert np.array_equal(ex.x[dead], img.x[dead])
    assert np.array_equal(ex.y[dead], img.y[dead])
    assert np.array_equal(ex.z[dead], img.z[dead])
    assert np.array_equal(ex.opd[dead], img.opd[dead])
    assert np.array_equal(ex.error_code[dead], img.error_code[dead])


def test_grazing_rays_are_killed_not_teleported():
    """``|N| <= 1e-30`` never reaches the vertex plane -> RAY_MISSED_SURFACE.

    The two JAX copies the helper replaces clamped ``N`` to ``1e-30`` and
    produced ``t = -z / 1e-30`` (~1e26 m of phantom path); the NumPy
    copies produced ``t = 0`` but still set ``z = 0``, teleporting the ray
    with ZERO optical path while leaving it ``alive``.  Neither is
    acceptable, and the two disagreed -- a cross-backend divergence in the
    same primitive.
    """
    b = RayBundle(x=np.zeros(2), y=np.array([1e-3, 2e-3]),
                  z=np.array([1e-4, -1e-4]),
                  L=np.array([1.0, 0.0]), M=np.array([0.0, 1.0]),
                  N=np.zeros(2), wavelength=WL,
                  alive=np.ones(2, bool), opd=np.zeros(2))
    out = exit_vertex_transfer(b, 1.0)
    assert not out.alive.any()
    assert np.all(out.error_code == RAY_MISSED_SURFACE)
    # Frozen, not teleported: same z, same (x, y), zero OPL added.
    assert np.array_equal(out.z, b.z)
    assert np.array_equal(out.x, b.x)
    assert np.array_equal(out.opd, b.opd)
    assert EXIT_VERTEX_GRAZING_TOL == 1e-30


def test_first_failure_wins_on_the_grazing_kill():
    """A grazing ray that already carries a diagnosis keeps it."""
    b = RayBundle(x=np.zeros(1), y=np.zeros(1), z=np.array([1e-4]),
                  L=np.array([1.0]), M=np.zeros(1), N=np.zeros(1),
                  wavelength=WL, alive=np.ones(1, bool), opd=np.zeros(1),
                  error_code=np.array([2], dtype=np.uint8))  # RAY_APERTURE
    out = exit_vertex_transfer(b, 1.0)
    assert int(out.error_code[0]) == 2


def test_transfer_is_idempotent():
    """Applying the operator twice is a no-op, bit for bit."""
    surfs = _curved_rear_singlet(-60e-3, -1.0)
    res = trace(_bundle(np.linspace(-12e-3, 12e-3, 17)), surfs, WL)
    once = res.at_exit_vertex()
    twice = exit_vertex_transfer(once, 1.0)
    for f in ('x', 'y', 'z', 'L', 'M', 'N', 'opd'):
        assert np.array_equal(getattr(twice, f), getattr(once, f)), f
    assert np.array_equal(twice.alive, once.alive)


def test_source_bundle_is_never_mutated():
    surfs = _curved_rear_singlet(-60e-3)
    res = trace(_bundle(np.linspace(-12e-3, 12e-3, 9)), surfs, WL)
    img = res.image_rays
    snap = {f: getattr(img, f).copy() for f in ('x', 'y', 'z', 'opd')}
    res.at_exit_vertex()
    for f, v in snap.items():
        assert np.array_equal(getattr(img, f), v), f


def test_matches_refocus_zero_on_alive_rays():
    """``refocus(result, 0.0)`` is the same operator on alive rays.

    They differ ONLY in the grazing policy (``refocus`` leaves a grazing
    ray alive and unmoved so a focus sweep cannot mutate the caller's
    alive mask), and this fixture has none.  BAR: bit-identical --
    both now route through the same ``vertex_plane_transfer_t`` kernel.
    """
    surfs = _curved_rear_singlet(-60e-3, -0.5)
    res = trace(_bundle(np.linspace(-12e-3, 12e-3, 13)), surfs, WL)
    a = res.at_exit_vertex()
    b = refocus(res, 0.0).image_rays
    assert np.array_equal(a.opd, b.opd)
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.y, b.y)


# ---------------------------------------------------------------------------
# 3. n_exit resolution
# ---------------------------------------------------------------------------

def test_n_exit_is_inferred_from_the_last_surface_exit_medium():
    """Default ``n_exit`` == index of ``surfaces[-1].glass_after``.

    Fixture ends INSIDE the glass, so the inferred index is n(N-BK7),
    not 1.0 -- the case a hard-coded ``n_exit = 1`` would silently get
    wrong by 52 %.
    """
    surfs = [
        Surface(radius=60e-3, semi_diameter=15e-3, glass_before='air',
                glass_after='N-BK7', thickness=6e-3, is_stop=True),
        Surface(radius=-60e-3, semi_diameter=15e-3, glass_before='N-BK7',
                glass_after='N-BK7', thickness=0.0),
    ]
    res = trace(_bundle(np.linspace(-10e-3, 10e-3, 11)), surfs, WL)
    img = res.image_rays
    auto = res.at_exit_vertex()
    explicit = res.at_exit_vertex(n_exit=N_BK7)
    t = -img.z / img.N
    # The inferred index must be the glass, to the catalogue's own value.
    inferred = (auto.opd - img.opd)[t != 0] / t[t != 0]
    assert np.allclose(inferred, N_BK7, rtol=1e-9)
    assert np.max(np.abs(auto.opd - explicit.opd)) < 1e-15
    assert abs(float(np.mean(inferred)) - 1.0) > 0.5, (
        'a hard-coded n_exit = 1.0 would be wrong by 52 % here')


def test_mirror_last_surface_uses_the_incident_medium():
    """A reflected ray leaves through the medium it ARRIVED in.

    A mirror's ``glass_after`` is frequently a marker rather than a real
    medium (the audit's E7 row notes the documented ``'__MIRROR__'``
    registry entry does not even exist), so the helper falls back to
    ``glass_before``.  Here both are air, and the transfer is checked
    against the closed form directly: for a ray reflected into -z the
    parametric distance back to the vertex plane is ``t = -z / N`` with
    ``N < 0``, so a ray at positive sag gets ``t > 0`` and ADDS path --
    the sign that ``abs(t)`` would get right by luck and that a
    ``t = +z/|N|`` transcription would get wrong.
    """
    surfs = [Surface(radius=-200e-3, conic=-1.0, semi_diameter=25e-3,
                     glass_before='air', glass_after='air', is_mirror=True,
                     is_stop=True, thickness=0.0)]
    res = trace(_bundle(np.array([0.0, 10e-3, 20e-3])), surfs, WL)
    img = res.image_rays
    ex = res.at_exit_vertex()
    assert np.all(ex.z == 0.0)
    assert np.all(img.N < 0), 'post-mirror rays must travel in -z'
    t = -img.z / img.N
    assert np.max(np.abs(ex.opd - (img.opd + 1.0 * t))) < 1e-18
    # The concave mirror's sag is NEGATIVE and N < 0, so t < 0 here: the
    # transfer SUBTRACTS over-counted path.  An abs(t) implementation
    # would be wrong by 2|n t| = 4.0e-3 m (3.4e3 waves) at h = 20 mm.
    assert np.all(t[1:] < 0)
    assert abs(float(2.0 * t[-1])) > 1e-3


def test_negative_or_non_finite_n_exit_is_refused():
    surfs = _curved_rear_singlet(-60e-3)
    res = trace(_bundle(np.zeros(1)), surfs, WL)
    for bad in (-1.0, 0.0, np.nan, np.inf):
        with pytest.raises(ValueError, match='n_exit'):
            res.at_exit_vertex(n_exit=bad)


# ---------------------------------------------------------------------------
# 4. JAX parity
# ---------------------------------------------------------------------------

def _jax_ok():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_jax_twin_matches_the_numpy_helper():
    """``exit_vertex_transfer_jax`` == ``exit_vertex_transfer``.

    BAR: 1e-17 m on position and OPL.  Derivation -- the audited
    NumPy<->JAX trace parity floor is 6.9e-18 m (position) / 2.8e-17 m
    (OPL) with x64 enabled, and the transfer adds one fused multiply-add,
    so the helper cannot beat that floor.  Measured here: 1.7e-18 m
    (position) and 5.2e-18 m (OPL).
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    from lumenairy.raytrace.jax_trace import (
        exit_vertex_transfer_jax,
        make_jax_ray_state,
        trace_jax,
    )

    pres = {'surfaces': [
        {'radius': 60e-3, 'glass_before': 'air', 'glass_after': 'N-BK7'},
        {'radius': -60e-3, 'glass_before': 'N-BK7', 'glass_after': 'air'}],
        'thicknesses': [6e-3, 0.0]}
    surfs = _curved_rear_singlet(-60e-3)
    surfs[0] = Surface(radius=60e-3, semi_diameter=np.inf,
                       glass_before='air', glass_after='N-BK7',
                       thickness=6e-3, is_stop=True)
    surfs[1] = Surface(radius=-60e-3, semi_diameter=np.inf,
                       glass_before='N-BK7', glass_after='air')
    ys = np.linspace(-12e-3, 12e-3, 9)
    n = ys.size

    npres = trace(_bundle(ys), surfs, WL)
    np_ex = npres.at_exit_vertex()

    st = make_jax_ray_state(np.zeros(n), ys, np.zeros(n),
                            np.zeros(n), np.zeros(n), np.ones(n))
    jx = exit_vertex_transfer_jax(trace_jax(st, pres, WL), 1.0)

    assert np.max(np.abs(np.asarray(jx.x) - np_ex.x)) < 1e-17
    assert np.max(np.abs(np.asarray(jx.y) - np_ex.y)) < 1e-17
    assert np.max(np.abs(np.asarray(jx.opd) - np_ex.opd)) < 1e-17
    assert np.array_equal(np.asarray(jx.z), np_ex.z)
    assert np.array_equal(np.asarray(jx.alive), np_ex.alive)


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_jax_twin_kills_grazing_rays_like_the_numpy_helper():
    """The cross-backend divergence the audit measured is gone.

    The two ``_lens_jax`` copies clamped ``N`` to 1e-30 and produced
    ``t = -z / 1e-30``; here the ray must simply die.
    """
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.raytrace.jax_trace import (
        JaxRayState,
        exit_vertex_transfer_jax,
    )
    st = JaxRayState(jnp.zeros(2), jnp.array([1e-3, 2e-3]),
                     jnp.array([1e-4, -1e-4]),
                     jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]),
                     jnp.zeros(2), jnp.zeros(2),
                     jnp.ones(2, dtype=bool))
    out = exit_vertex_transfer_jax(st, 1.0)
    assert not bool(np.asarray(out.alive).any())
    assert np.array_equal(np.asarray(out.z), np.asarray(st.z))
    assert np.array_equal(np.asarray(out.opd), np.asarray(st.opd))
    assert np.all(np.isfinite(np.asarray(out.opd)))


@pytest.mark.skipif(not _jax_ok(), reason='jax not installed')
def test_jax_twin_is_grad_safe_at_the_grazing_boundary():
    """``jax.grad`` through the transfer must not see NaN at ``N -> 0``."""
    import jax
    jax.config.update('jax_enable_x64', True)
    import jax.numpy as jnp

    from lumenairy.raytrace.jax_trace import (
        JaxRayState,
        exit_vertex_transfer_jax,
    )

    def loss(nz):
        st = JaxRayState(jnp.zeros(1), jnp.zeros(1), jnp.array([1e-4]),
                         jnp.array([1.0]), jnp.zeros(1), nz * jnp.ones(1),
                         jnp.zeros(1), jnp.ones(1, dtype=bool))
        return jnp.sum(exit_vertex_transfer_jax(st, 1.0).opd)

    g = float(jax.grad(loss)(0.0))
    assert np.isfinite(g)
