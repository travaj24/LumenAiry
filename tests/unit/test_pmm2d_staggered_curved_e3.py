"""CURVED-CELL MAP, Phase E3, for the PURE staggered 2-D PMM: the JAX TWIN of
the shared-grid cascade, differentiable in the materials, the thicknesses,
the half-space indices and the SHAPE PARAMETERS -- gates E3-1 .. E3-9 of
``docs/audits/BUILD_PMM2D_CURVED_E3_2026_10_03.md`` (plan
``docs/audits/PLAN_PMM2D_CURVED_CELLS_2026_09_26.md`` section 4.5).

Words.  The TWIN (``lumenairy/elements/pmm/_jax_twod_staggered.py``,
``PMM2DStackPure(backend='jax')``) is built from a CONCRETE stack -- the
REFERENCE, at which every discrete decision (wall grid and topology,
quadrature node counts, corner cells) is frozen -- and evaluates the cascade
on a parameter dictionary whose leaves (and shape objects) may be JAX
values.  AD = its reverse-mode gradient; FD(twin) = central differences of
its own forward, Richardson-extrapolated from the h / P = 3e-4 and 1e-4
rungs (an h^2 truncation, h ratio 3).  A FAIL-BEFORE / MUTATION arm is a
deliberately broken variant that must fail the bar.

Fixture: the planning P3 cell (period 1.2, lambda 1, air over n = 1.45),
a circular pillar r = 0.36 (eps 4, depth 0.5) or a rectangular one
(0.5 x 0.4, depth 0.4), 3 x 3 grids, n_orders = 2, M = 3 / 4 (unit-test
sizes; the build doc carries M = 4 .. 6).  EVERY BAR is derived from a
measurement of this build on 2026-10-03 (Windows 11, CPython 3.14.6, numpy
2.4.4, scipy 1.17.1, jax 0.11.0; probe JSON under
``validation/probe_pmm2d_curved/build_e3/``, the unit-size readings in
``e3_unit_readings.json``), stated next to the assertion with its gap on
both sides.  JAX compiles are slow (10-40 s each here), so the compiled
functions are cached per module.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ast  # noqa: E402
import functools  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.backend import JAX_AVAILABLE  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    FilletRect,
    PMM2DStackPure,
    Rect,
    SinusoidalWall,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import shapes2d as SH  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed"),
    pytest.mark.filterwarnings("ignore:.*energy closure.*"),
    pytest.mark.filterwarnings("ignore:.*Rayleigh cutoff.*"),
]

_P, _WL = 1.2, 1.0
_LC = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.55)
_MU_G = np.array([[1.6, 0.4j, 0], [-0.4j, 1.6, 0], [0, 0, 1.2]], complex)
_EYE = np.eye(3, dtype=complex)


@pytest.fixture(autouse=True, scope="module")
def _jax_x64_for_this_module():
    """Hold ``jax_enable_x64`` on for this file and RESTORE it (the JAX x64
    isolation contract: the flag must not leak to the next module)."""
    if not JAX_AVAILABLE:
        yield
        return
    import jax
    previous = bool(getattr(jax.config, "jax_enable_x64", False))
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def _stack(layers, M=4, backend="jax", n_orders=2, theta=0.0, phi=0.0,
           cmap=None):
    kw = {} if cmap is None else {"cmap": cmap}
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=n_orders, backend=backend, **kw)
    for L in layers:
        st.add_layer(**L)
    st.set_source(_WL, theta=theta, phi=phi)
    return st


def _circle_layers(r=0.36, eps=4.0, cx=0.6, **kw):
    return [dict(thickness=0.5, shapes=[Circle(cx, 0.6, r, eps, **kw)],
                 background_eps=1.0)]


def _amax(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


@functools.lru_cache(maxsize=None)
def _circle_fns(M, quantity="T"):
    """(stack, twin, jit f(r), jit grad f) for the centred circle -- built
    and compiled ONCE per module (the per-test budget cannot pay a compile
    each)."""
    import jax
    import jax.numpy as jnp
    st = _stack(_circle_layers(), M=M)
    tw = st.jax_twin()
    p0 = tw.p0

    def f(r):
        p = tw.params()
        p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, 4.0)]
        _o, R, T, J = st.solve(params=p)
        return (T if quantity == "T" else R)[0, p0]
    return st, tw, jax.jit(f), jax.jit(jax.grad(f)), f


def _fd(fj, x0, scale=_P):
    rows = []
    for hs in (3e-4, 1e-4):
        h = hs * scale
        rows.append((float(fj(x0 + h)) - float(fj(x0 - h))) / (2 * h))
    return (9.0 * rows[1] - rows[0]) / 8.0, abs(rows[1] - rows[0])


# =========================================================================== #
# E3-1 -- no JAX call = today's bytes
# =========================================================================== #
def test_e3_1_numpy_solves_never_reach_the_twin_branches(monkeypatch):
    """No JAX call = today's NumPy bytes: ``e1_compare.json`` -- 156 / 156
    SHA-256 equal against ``eae470d9`` (operators, region modes, the mapped
    geometric cache, the cofactor far field, the incident decomposition and
    R / T / Jones / absorption of every dispatch branch, every shape
    primitive, the tensor and magnetic mapped solvers and a two-layer
    merged stack).  The unit restatement: with every twin-only branch
    booby-trapped, an unmapped, a mapped (circle), a tensor and a shapes
    stack all solve through the NumPy path, and ``_xset`` writes IN PLACE
    on NumPy (the shipped statement, the same object)."""
    def boom(*a, **k):
        raise AssertionError("a twin-only branch was reached")
    for name in ("_stag_map_weights_traced", "_stag_quad_embedded",
                 "_prefetch_frozen", "_stag_map_node_jacobian"):
        monkeypatch.setattr(TS, name, boom)
    monkeypatch.setattr(CM.TransfiniteMap, "_traced", boom)
    monkeypatch.setattr(CM.TransfiniteMap, "prefetch", boom)
    _stack(_circle_layers(), M=3, backend="numpy").solve()
    _stack([dict(thickness=0.4, shapes=[FilletRect(0.6, 0.6, 0.6, 0.5, 0.1,
                                                   4.0)],
                 background_eps=1.0)], M=3, backend="numpy").solve()
    cell = np.broadcast_to(_EYE, (2, 2, 3, 3)).copy()
    cell[0, 0] = _LC
    pmm_jones_2d_staggered(_P, _P, cell, 1.45, 1.0, 0.4, _WL, degree=3,
                           n_orders=2)
    A = np.zeros((3, 3), complex)
    out = TS._xset(np, A, slice(0, 1), slice(0, 1), 1.0)
    assert out is A and A[0, 0] == 1.0


# =========================================================================== #
# E3-2 -- forward parity
# =========================================================================== #
#: Fixture set of the unit parity gate (the build doc's f2 set, 23 fixtures,
#: at M = 4, is the full one).
_PAR = {
    "scalar_pillar": dict(layers=[dict(thickness=0.4, eps_cell=np.array(
        [[4.0, 1.0], [1.0, 1.0]], complex))]),
    "scalar_conical_lossy": dict(layers=[dict(thickness=0.4, eps_cell=np.array(
        [[4.0 + 0.5j, 1.0], [1.0, 1.0]]))], theta=0.25, phi=0.4),
    "tensor_lc": dict(layers=[dict(thickness=0.4, eps_cell=np.where(
        np.arange(4).reshape(2, 2, 1, 1) == 0, _LC, 2.0 * _EYE))]),
    "magnetic_tensor": dict(layers=[dict(
        thickness=0.4, eps_cell=np.array([[4.0, 1.0], [1.0, 1.0]], complex),
        mu_cell=np.where(np.arange(4).reshape(2, 2, 1, 1) == 0, _MU_G,
                         _EYE))]),
    "multilayer": dict(layers=[dict(thickness=0.2, eps=2.1),
                               dict(thickness=0.3, eps_cell=np.array(
                                   [[4.0 + 0.3j, 1.0], [1.0, 1.0]])),
                               dict(thickness=0.15, eps=_LC)],
                       theta=0.15, phi=0.2),
    "circle": dict(layers=_circle_layers()),
    "circle_conical": dict(layers=_circle_layers(), theta=0.3, phi=0.4),
    "fillet": dict(layers=[dict(thickness=0.4, shapes=[FilletRect(
        0.6, 0.6, 0.6, 0.5, 0.1, 4.0)], background_eps=1.0)]),
    "sinusoid": dict(layers=[dict(thickness=0.4, shapes=[SinusoidalWall(
        "x", 0.6, 0.08, eps=2.25)], background_eps=1.0)]),
    "circle_tensor_magnetic_two_layer": dict(layers=[
        dict(thickness=0.3, shapes=[Circle(0.6, 0.6, 0.36, _LC)],
             background_eps=2.0),
        dict(thickness=0.2, shapes=[Circle(0.6, 0.6, 0.36, 4.0, mu=_MU_G)],
             background_eps=1.0)]),
}


@pytest.mark.parametrize("name", sorted(_PAR))
def test_e3_2_forward_parity_twin_vs_numpy(name):
    """The twin's R / T / Jones equal the NumPy stack's to round-off.

    Measured 2026-10-03 (``f2_parity_M3.json`` / ``_M4.json``, 22 fixtures):
    max |twin - numpy| 3.4e-14 at M = 3 (5.3e-13 at M = 4, the fillet); the
    round-off REACH of the eig stage alone -- the NumPy stack with its QZ
    replaced by the standard eig of G^-1 L, an equally exact algorithm --
    reaches 5.1e-14 (M = 3) / 6.2e-13 (M = 4) over the same set, and the
    twin sits within 2.9x of it fixture by fixture; the assembled operators
    agree to <= 6.1e-14 relative.  Bar 5e-12: two decades above the M = 3
    maximum and 8x the M = 4 reach; the upper gap: the cofactor mutation of
    E3-9 moves the mapped T by 6.6e-2 (``f9_mutations_M4.json``), ten
    decades above."""
    fx = _PAR[name]
    kw = {k: v for k, v in fx.items() if k != "layers"}
    o, R, T, J = _stack(fx["layers"], M=3, backend="numpy", **kw).solve()
    _o, Rj, Tj, Jj = _stack(fx["layers"], M=3, **kw).solve()
    for a, b in ((Rj, R), (Tj, T), (Jj, J)):
        assert _amax(a, b) <= 5e-12, (name, _amax(a, b))


# =========================================================================== #
# E3-3 -- gradients vs converged central differences
# =========================================================================== #
def test_e3_3_circle_radius_gradient_matches_fd():
    """d T00 / d r of the circle (the first curved gradient of the library)
    through traced Circle objects.  Measured 2026-10-03 (unit size M = 3,
    ``e3_unit_readings.json``): AD vs the Richardson FD(twin) 6.7e-12
    relative; the two FD rungs differ by 8.9e-7 relative (their own h^2
    term, which the extrapolation removes -- the premise, asserted below
    1e-5: the FD is in its asymptotic range).  Build doc (``f1``, ``f3``):
    2.3e-8 at M = 6 against the plain last rung, 7.8e-8 against the
    shipped NumPy solve's FD.  Bar 1e-6: five decades above the reading;
    an over-broadened eig derivative (E3-5 arm) is off by 1.8e-6."""
    _st, _tw, fj, gj, _f = _circle_fns(3)
    g = float(gj(0.36))
    fd, ch = _fd(fj, 0.36)
    assert ch / abs(fd) < 1e-5                       # premise: FD converged
    assert abs(g - fd) / abs(fd) < 1e-6, (g, fd)


def test_e3_3_rectangle_width_gradient_matches_numpy_fd():
    """d R00 / d w of the rectangular pillar: the width enters as a traced
    ``Rect`` whose walls are the IMAGES of the frozen grid (affine cells), so
    the twin at ``w`` IS the NumPy solve at ``w`` (3.5e-15) and its AD meets
    the SHIPPED solver's FD.  Measured: 1.0e-11 relative at M = 3
    (``e3_unit_readings.json``; FD rungs 5.4e-7 apart), 1.2e-9 .. 2.4e-9
    against the unextrapolated FD at M = 4 .. 6 (``f1_rect_M*.json``); bar
    1e-6, five decades above."""
    import jax
    import jax.numpy as jnp
    lay = [dict(thickness=0.4, shapes=[Rect(0.6, 0.6, 0.5, 0.4, 4.0)],
                background_eps=1.0)]
    st = _stack(lay, M=3)
    tw = st.jax_twin()
    p0 = tw.p0

    def f(w):
        p = tw.params()
        p["layers"][0]["shapes"] = [Rect(0.6, 0.6, w, 0.4, 4.0)]
        return st.solve(params=p)[1][0, p0]
    g = float(jax.jit(jax.grad(f))(jnp.asarray(0.5)))

    def fn(w):
        return float(_stack([dict(thickness=0.4, shapes=[Rect(
            0.6, 0.6, w, 0.4, 4.0)], background_eps=1.0)], M=3,
            backend="numpy").solve()[1][0, p0])
    fd, ch = _fd(fn, 0.5)
    assert ch / abs(fd) < 1e-5
    assert abs(g - fd) / abs(fd) < 1e-6, (g, fd)


def test_e3_3_material_and_depth_gradients_match_fd():
    """d T00 / d Re(eps), d Im(eps) and d depth of the circle through the
    params dictionary.  Measured at M = 3 (``e3_unit_readings.json``):
    5.4e-10, 6.2e-10, 4.6e-13 relative (FD rungs 3e-9 .. 1.6e-6 apart);
    M = 4 / 5, both builds (``f3_eps_*``, ``f3_depth_*``): <= 1.9e-8.  Bar
    1e-6 relative."""
    import jax
    import jax.numpy as jnp
    st = _stack(_circle_layers(eps=4.0 + 0.2j), M=3)
    tw = st.jax_twin()
    p0 = tw.p0

    def f(v):
        p = tw.params()
        p["layers"][0]["shapes"] = [Circle(0.6, 0.6, 0.36,
                                           v[0] + 1j * v[1])]
        p["layers"][0]["thickness"] = v[2]
        return st.solve(params=p)[2][0, p0]
    x0 = np.array([4.0, 0.2, 0.5])
    fj = jax.jit(f)
    g = np.asarray(jax.jit(jax.grad(f))(jnp.asarray(x0)))
    for k in range(3):
        def fk(x, k=k):
            e = np.zeros(3)
            e[k] = x - x0[k]
            return float(fj(jnp.asarray(x0 + e)))
        fd, ch = _fd(fk, x0[k], scale=1.0)
        assert ch / abs(fd) < 1e-5
        assert abs(g[k] - fd) / abs(fd) < 1e-6, (k, g[k], fd)


# =========================================================================== #
# E3-4 -- non-differentiable events: refused or NaN, and a smooth path
# =========================================================================== #
def test_e3_4_a_fold_inside_the_trace_is_nan_and_a_path_inside_is_finite():
    """The frozen topology holds only while the map does not fold: a
    circle bulging past the cell edge (r = 0.62, c + r > P) folds the outer
    cells (det J <= 0 at nodes) and the jitted twin returns NaN (value and
    gradient) rather than a silently wrong number; r = 0.58 stays inside
    and is finite (``f4_events_M4.json``: fold NaN / NaN, control finite)."""
    _st, _tw, fj, gj, _f = _circle_fns(3)
    assert np.isnan(float(fj(0.62))) and np.isnan(float(gj(0.62)))
    assert np.isfinite(float(fj(0.58))) and np.isfinite(float(gj(0.58)))


def test_e3_4_a_topology_change_with_concrete_values_is_refused():
    """A CONCRETE shape parameter is checked against the frozen topology by
    the NumPy merge itself: a rectangle whose outer segment falls below the
    sliver contract (w = 1.199 in a 1.2 cell) and a fillet radius of 0 (the
    four singular vertices -- the Duffy cells -- vanish) are REFUSED, a
    width / radius inside the topology is accepted; traced, the same values
    return NaN (``f4_events_M4.json``)."""
    st = _stack([dict(thickness=0.4, shapes=[Rect(0.6, 0.6, 0.5, 0.4, 4.0)],
                      background_eps=1.0)], M=3)
    p = st.jax_params()
    p["layers"][0]["shapes"] = [Rect(0.6, 0.6, 1.199, 0.4, 4.0)]
    with pytest.raises(ValueError, match="TOPOLOGY|refused"):
        st.solve(params=p)
    p["layers"][0]["shapes"] = [Rect(0.6, 0.6, 1.19, 0.4, 4.0)]
    assert np.all(np.isfinite(np.asarray(st.solve(params=p)[1])))
    st = _stack([dict(thickness=0.4, shapes=[FilletRect(0.6, 0.6, 0.6, 0.5,
                                                        0.1, 4.0)],
                      background_eps=1.0)], M=3)
    p = st.jax_params()
    p["layers"][0]["shapes"] = [FilletRect(0.6, 0.6, 0.6, 0.5, 0.0, 4.0)]
    with pytest.raises(ValueError, match="TOPOLOGY|refused|structure"):
        st.solve(params=p)


# =========================================================================== #
# E3-5 -- degenerate eigenvalues
# =========================================================================== #
@functools.lru_cache(maxsize=None)
def _circle_rcx():
    """f(r, cx) -> T00 of the circle at M = 3, its jitted FD function along r
    reused from :func:`_circle_fns`."""
    st = _stack(_circle_layers(), M=3)
    tw = st.jax_twin()
    p0 = tw.p0

    def f(v):
        p = tw.params()
        p["layers"][0]["shapes"] = [Circle(v[1], 0.6, v[0], 4.0)]
        return st.solve(params=p)[2][0, p0]
    import jax
    vg = jax.jit(jax.value_and_grad(f))      # ONE compile: values for the FD
    return f, vg


def _fd_r(vg):
    """Richardson FD of T00 along r from the compiled value-and-grad."""
    import jax.numpy as jnp
    return _fd(lambda r: float(vg(jnp.asarray([r, 0.6]))[0]), 0.36)


def test_e3_5_gradients_through_degenerate_modes():
    """The CENTRED circle is four-fold symmetric: its layer pencil carries 37
    and the half-spaces' geometric pencil 41 exactly degenerate pairs at
    M = 4 (``f5_degenerate_M4.json``).  Through ``_jax_eig_stable``:

    * d T00 / d cx (a centre shift SPLITS the pairs) is zero by the
      x-mirror; measured 2.9e-15 at M = 3 (``e3_unit_readings.json``),
      -7e-14 .. 3e-15 at M = 4 for every regularisation 1e-14 .. 1e-6;
      bar 1e-9 (|d T00 / d r| ~ 1.4, nine decades below the scale);
    * d T00 / d r (symmetry-preserving) matches the Richardson FD to
      6.7e-12 at M = 3 (the FD's own round-off is ~1e-12 at h = 1e-4 P);
      bar 1e-10.  The broadening's own effect, measured: harmless up to
      tau_rel = 1e-8, an O(tau) bias at 1e-6 (2.9e-9 at M = 3, 1.8e-6 /
      2.4e-6 at M = 4) -- the over-broadened arm of the next test."""
    import jax.numpy as jnp
    _f, vg = _circle_rcx()
    g = np.asarray(vg(jnp.asarray([0.36, 0.6]))[1])
    fd, _ch = _fd_r(vg)
    assert abs(g[1]) <= 1e-9, g
    assert abs(g[0] - fd) / abs(fd) <= 1e-10, (g, fd)


def test_e3_5_mutations_of_the_eig_derivative_are_caught():
    """Fail-before arms of E3-5.  (1) An OVER-BROADENED eigenvector VJP
    (``tau_rel = 1e-6``) biases d T00 / d r past the 1e-10 bar (measured
    2.9e-9 at M = 3).  (2) ``jnp.linalg.eig`` in place of
    ``_jax_eig_stable``: JAX refuses eigenvector derivatives of a
    non-symmetric matrix outright (``NotImplementedError``) -- caught.
    (Measured, not gated: JAX's opt-in UNREGULARISED eigenvector derivative
    returns the same gradients here -- these outputs are gauge-invariant
    functions of the operator and LAPACK splits the degenerate pairs at
    round-off -- so the regularisation is a safety margin on this fixture,
    not a correction; ``f9_mutations_M4.json``.)"""
    import jax
    import jax.numpy as jnp

    import lumenairy.elements.pmm._jax_twod_staggered as JT
    f, vg = _circle_rcx()
    fd, _ch = _fd_r(vg)
    x0 = jnp.asarray([0.36, 0.6])
    JT._E3_EIG_TAU_REL = 1e-6
    try:
        gb = np.asarray(jax.jit(jax.grad(lambda v: f(v)))(x0))
    finally:
        JT._E3_EIG_TAU_REL = None
    assert abs(gb[0] - fd) / abs(fd) > 1e-10, (gb, fd)
    orig = JT._stag_geneig_jax
    JT._stag_geneig_jax = lambda L, G, tau_rel=None: jnp.linalg.eig(
        jnp.linalg.solve(G, L))
    try:
        with pytest.raises(NotImplementedError):
            jax.jit(jax.grad(lambda v: f(v)))(x0)
    finally:
        JT._stag_geneig_jax = orig


# =========================================================================== #
# E3-6 -- jit: compile once
# =========================================================================== #
def test_e3_6_jit_compiles_once_for_every_radius():
    """``jax.jit`` of a full solve traces ONCE and re-runs at new radii
    without recompiling (all array shapes frozen by the template); a
    Python-side trace counter and jit's cache size both read 1.  Timings in
    the build doc (``f6_jit_M*.json``)."""
    import jax
    st = _stack(_circle_layers(), M=3)
    tw = st.jax_twin()
    p0 = tw.p0
    n = {"traces": 0}

    def f(r):
        n["traces"] += 1
        p = tw.params()
        p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, 4.0)]
        return st.solve(params=p)[2][0, p0]
    fj = jax.jit(f)
    vals = [float(fj(r)) for r in (0.33, 0.36, 0.39)]
    assert n["traces"] == 1 and fj._cache_size() == 1
    assert len(set(vals)) == 3


# =========================================================================== #
# E3-7 -- a 1-D sanity: two independent differentiable solvers
# =========================================================================== #
def test_e3_7_stripe_gradients_match_the_1d_pmm_twin():
    """A y-uniform stripe -- ``Rect(0.9, 0.6, 0.6, P)``: the ridge [0.6, 1.2]
    spanning the period in y, a 2 x 2 grid -- is a 1-D lamellar grating
    (incident E_x = its TM case, E_y = TE).  d R00 / d eps and d R00 / d depth
    from the 2-D twin against the AD of the 1-D PMM's JAX twin
    (``pmm_efficiency_1d`` with JAX inputs, degree 40): two independent
    differentiable solvers.  The difference is the 2-D DISCRETISATION level
    (the twin equals the NumPy 2-D solve to round-off, E3-2), measured at
    this M = 7 in ``e3_unit_readings.json`` / ``f7_stripe_M*.json``; the bar
    is set from it.  (d / d width: neither 1-D twin traces a wall; the build
    doc compares it with the converged FD of the NumPy 1-D PMM.)"""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import pmm_efficiency_1d
    st = _stack([dict(thickness=0.4, shapes=[Rect(0.9, 0.6, 0.6, _P, 4.0)],
                      background_eps=1.0)], M=7)
    tw = st.jax_twin()
    p0 = tw.p0

    def f2(v):
        p = tw.params()
        p["layers"][0]["shapes"] = [Rect(0.9, 0.6, 0.6, _P, v[0])]
        p["layers"][0]["thickness"] = v[1]
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, p0], R[1, p0]])
    g2 = np.asarray(jax.jit(jax.jacrev(f2))(jnp.asarray([4.0, 0.4])))
    for k, pol in enumerate(("tm", "te")):
        def f1(v, pol=pol):
            o, R, T = pmm_efficiency_1d(_P, jnp.sqrt(v[0]), 1.0, 1.45, 1.0,
                                        v[1], 0.5, _WL, polarization=pol,
                                        degree=40, stabilize=False)
            return R[int(np.where(np.asarray(o) == 0)[0][0])]
        g1 = np.asarray(jax.grad(f1)(jnp.asarray([4.0, 0.4])))
        rel = np.abs(g2[k] - g1) / np.abs(g1)
        assert np.all(rel < _E37_BAR[pol]), (pol, g2[k], g1, rel)


_E37_BAR = {"tm": 1e-1, "te": 1e-2}


# =========================================================================== #
# E3-8 -- the census: every kernel the twin uses is the NumPy object
# =========================================================================== #
_SHARED = {
    TS: ("_stag_quad_weighted", "_stag_quad_axis_factor", "_stag_map_weights",
         "_stag_map_eff", "_stag_map_node_jacobian", "_region_modes_from_eig",
         "_homog_geom_from_eig", "_homog_region_modes", "_far_projector_mapped",
         "_stag_incident_load_mapped", "_stag_incident_coeffs_mapped",
         "_pmm2d_project_orders", "_pmm2d_order_kz", "_forward_branch_flip",
         "_inv_lam", "_kz_forward2", "_xset"),
    SH: ("_curve_piece", "_fill_mask"),
}


def test_e3_8_census_every_shared_kernel_is_the_numpy_object(monkeypatch):
    """The ONE-kernel rule in its JAX form, measured two ways.  (1) Spies
    on every shared kernel of ``twod_staggered`` / ``shapes2d`` /
    ``rcwa._core`` / the assembly method / the Gordon-Hall formula: one
    traced-shape twin solve (a circle in a tensor host, a lossy oblique
    source) must call EVERY one of them through the NumPy module.  (2) The
    twin module defines no function whose name is one of them (no private
    copy to drift)."""
    import jax.numpy as jnp

    import lumenairy.elements.pmm._jax_twod_staggered as JT
    import lumenairy.elements.rcwa as RC
    import lumenairy.elements.rcwa._core as RCC
    hits = {}

    def spy(mod, name):
        orig = getattr(mod, name)

        def wrapped(*a, **k):
            hits[name] = hits.get(name, 0) + 1
            return orig(*a, **k)
        monkeypatch.setattr(mod, name, wrapped)
    for mod, names in _SHARED.items():
        for nm in names:
            spy(mod, nm)
    for nm in ("_interface_smatrix", "_redheffer_star", "_propagation_smatrix",
               "_project_efficiency"):
        spy(RCC, nm)
    spy(RC, "_jax_eig_stable")
    spy(TS.Granet2DTransverseE, "_assemble")
    gh = CM.TransfiniteMap._gh

    def gh_spy(*a, **k):
        hits["_gh"] = hits.get("_gh", 0) + 1
        return gh(*a, **k)
    monkeypatch.setattr(CM.TransfiniteMap, "_gh", staticmethod(gh_spy))
    st = _stack([dict(thickness=0.5, shapes=[Circle(0.6, 0.6, 0.36, 4.0)],
                      background_eps=1.0),
                 dict(thickness=0.2, eps=_LC)], M=3, theta=0.2)
    p = st.jax_params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, jnp.asarray(0.35), 4.0)]
    st.solve(params=p)
    missing = [nm for names in _SHARED.values() for nm in names
               if nm not in hits]
    missing += [nm for nm in ("_interface_smatrix", "_redheffer_star",
                              "_propagation_smatrix", "_project_efficiency",
                              "_jax_eig_stable", "_assemble", "_gh")
                if nm not in hits]
    assert not missing, missing
    with open(JT.__file__, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    defined = {n.name for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    shared = {nm for names in _SHARED.values() for nm in names} | {
        "_assemble", "_gh", "_interface_smatrix", "_redheffer_star",
        "_propagation_smatrix", "_project_efficiency", "_jax_eig_stable",
        "_blend", "_layout", "_merge", "_curve_point"}
    assert not (defined & shared), defined & shared


# =========================================================================== #
# E3-9 -- mutation matrix
# =========================================================================== #
def test_e3_9_dropping_the_cofactor_in_the_jax_far_field_is_caught(
        monkeypatch):
    """Mutation: the JAX far field without the cofactor (the covariant
    coefficients projected as if Cartesian) -- in the TRACED path only, so
    the NumPy answer is untouched.  Caught by the E3-2 bar (5e-12) on the
    mapped circle by many decades (``f9_mutations.json``)."""
    import jax.numpy as jnp
    orig = TS._far_projector_mapped

    class NoCof:
        def __init__(self, m):
            self.m = m

        def __getattr__(self, k):
            return getattr(self.m, k)

        def geom(self, sx, sy, U, V):
            X, Y, xu, xv, yu, yv = self.m.geom(sx, sy, U, V)
            one = jnp.ones_like(xu)
            return X, Y, one, 0 * xv, 0 * yu, one

    def mutated(bx, by, ox, oy, a0x, a0y, cmap, xp=np, **kw):
        if xp is not np:
            cmap = NoCof(cmap)
        return orig(bx, by, ox, oy, a0x, a0y, cmap, xp=xp, **kw)
    monkeypatch.setattr(TS, "_far_projector_mapped", mutated)
    lay = _circle_layers()
    o, R, T, J = _stack(lay, M=3, backend="numpy").solve()
    st = _stack(lay, M=3)
    p = st.jax_params()
    p["layers"][0]["shapes"] = [Circle(0.6, 0.6, jnp.asarray(0.36), 4.0)]
    _o, Rj, Tj, Jj = st.solve(params=p)
    assert max(_amax(Rj, R), _amax(Tj, T)) > 1e-3


def test_e3_9_reading_the_frozen_decisions_inside_the_trace_fails_loudly(
        monkeypatch):
    """Two mutations of the FROZEN discrete decisions, both caught by the
    jit gate as a concretization error (not a silently different answer):
    (a) the far-field node count re-read inside the trace (``nq_cells``
    dropped -> the sizing calls ``float(ptp(X))`` on a traced map); (b) the
    forward-branch gauge decided on the host from traced data (the NumPy
    ``_forward_branch_flip`` concretises the scale with ``float``)."""
    import jax

    _st, _tw, _fj, _gj, f = _circle_fns(3)
    orig = TS._far_projector_mapped

    def reread(*a, **k):
        k.pop("nq_cells", None)
        return orig(*a, **k)
    monkeypatch.setattr(TS, "_far_projector_mapped", reread)
    # (a FRESH function object each time: jit caches compiled code per
    # function, and a cached executable would never retrace the mutation)
    with pytest.raises((jax.errors.ConcretizationTypeError,
                        jax.errors.TracerArrayConversionError)):
        jax.jit(lambda r: f(r))(0.36)
    monkeypatch.setattr(TS, "_far_projector_mapped", orig)
    flip = TS._forward_branch_flip
    monkeypatch.setattr(TS, "_forward_branch_flip",
                        lambda q, xp=np: flip(q))
    with pytest.raises((jax.errors.ConcretizationTypeError,
                        jax.errors.TracerArrayConversionError)):
        jax.jit(lambda r: f(r))(0.36)


# =========================================================================== #
# API: the convenience entry, the refusals, the switches
# =========================================================================== #
def test_e3_convenience_entry_matches_numpy_and_traces_eps():
    """``pmm_jones_2d_staggered(..., backend='jax')`` returns the NumPy
    entry's answer (bar 5e-12, E3-2) and traces an ``eps_cell`` VALUE and the
    depth; traced SHAPES need ``reference_shapes=``."""
    import jax
    import jax.numpy as jnp
    cell = np.array([[4.0, 1.0], [1.0, 1.0]], complex)
    ref = pmm_jones_2d_staggered(_P, _P, cell, 1.45, 1.0, 0.4, _WL,
                                 degree=3, n_orders=2)
    out = pmm_jones_2d_staggered(_P, _P, cell, 1.45, 1.0, 0.4, _WL,
                                 degree=3, n_orders=2, backend="jax")
    for a, b in zip(out[1:], ref[1:]):
        assert _amax(a, b) <= 5e-12

    def f(e):
        c = jnp.asarray(cell).at[0, 0].set(e)
        return pmm_jones_2d_staggered(_P, _P, c, 1.45, 1.0, 0.4, _WL,
                                      degree=3, n_orders=2,
                                      backend="jax")[2][0, 12]
    assert np.isfinite(float(jax.grad(f)(jnp.asarray(4.0 + 0j).real)))
    with pytest.raises(ValueError, match="reference_shapes"):
        pmm_jones_2d_staggered(_P, _P, None, 1.45, 1.0, 0.5, _WL, n_modes=3,
                               n_orders=2, backend="jax",
                               shapes=[Circle(0.6, 0.6, jnp.asarray(0.36),
                                              4.0)], background_eps=1.0)


def test_e3_refusals_name_the_follow_up():
    """Out of the twin's scope, each naming its follow-up: an OUT-OF-PLANE
    tensor and ``slant`` (Phase E1), ``layer_grids='per-layer'`` (Phase E2),
    ``retain_internal`` (NumPy-only), a traced superstrate index at oblique
    incidence (it sets the frozen Bloch glue)."""
    oop = uniaxial_tensor(1.5, 1.8, 0.6, phi=0.3)
    cell = np.broadcast_to(2.0 * _EYE, (2, 2, 3, 3)).copy()
    cell[0, 0] = oop
    st = _stack([dict(thickness=0.4, eps_cell=cell)], M=3)
    with pytest.raises(NotImplementedError, match="E1"):
        st.solve()
    st = _stack([dict(thickness=0.4, eps_cell=np.array([[4.0, 1], [1, 1]],
                                                       complex),
                      slant=(0.2, 0.0))], M=3)
    with pytest.raises(NotImplementedError, match="E1"):
        st.solve()
    st = PMM2DStackPure(_P, _P, n_modes=3, n_orders=2, backend="jax",
                        layer_grids="per-layer")
    st.add_layer(0.4, eps_cell=np.array([[4.0, 1], [1, 1]], complex))
    st.set_source(_WL)
    with pytest.raises(NotImplementedError, match="E2"):
        st.solve()
    st = _stack(_circle_layers(), M=3)
    with pytest.raises(NotImplementedError, match="retain_internal"):
        st.solve(retain_internal=True)
    import jax.numpy as jnp
    st = _stack(_circle_layers(), M=3, theta=0.2)
    p = st.jax_params()
    p["n_superstrate"] = jnp.asarray(1.0 + 0j)
    with pytest.raises(NotImplementedError, match="OBLIQUE"):
        st.solve(params=p)


def test_e3_disable_jax_switch_and_x64_are_honoured(monkeypatch):
    """``LUMENAIRY_DISABLE_JAX`` (read into ``backend.JAX_AVAILABLE``) turns
    the backend off with an ImportError naming the switch; without
    ``jax_enable_x64`` the twin RAISES (``_require_jax_x64``) instead of
    silently computing in complex64."""
    import jax

    import lumenairy.backend as B
    monkeypatch.setattr(B, "JAX_AVAILABLE", False)
    with pytest.raises(ImportError, match="LUMENAIRY_DISABLE_JAX"):
        PMM2DStackPure(_P, _P, n_modes=3, backend="jax")
    monkeypatch.setattr(B, "JAX_AVAILABLE", True)
    st = _stack(_circle_layers(), M=3)
    jax.config.update("jax_enable_x64", False)
    try:
        with pytest.raises(RuntimeError, match="x64"):
            st.solve()
    finally:
        jax.config.update("jax_enable_x64", True)


def test_e3_numpy_backend_rejects_params():
    """``solve(params=...)`` belongs to the twin; the NumPy stack refuses it
    rather than ignoring it."""
    st = _stack(_circle_layers(), M=3, backend="numpy")
    with pytest.raises(ValueError, match="backend='jax'"):
        st.solve(params={})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match="backend"):
            PMM2DStackPure(_P, _P, backend="torch")
