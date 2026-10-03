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
    Ellipse,
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


#: PER-FIXTURE parity bars (round 2 of the build, verifier observation on
#: E3-2): 30x that fixture's eig-stage round-off REACH -- the NumPy stack with
#: its QZ replaced by the standard eig of G^-1 L (the twin's reduction, an
#: equally exact algorithm) minus the shipped stack, max over R, T, J -- the
#: larger of the two builds, rounded up (``r7_parity_reach_{win,wsl}.json``).
#: The twin's own difference sits at 0.25 .. 2.9x the reach on every fixture
#: (both builds), so the factor 30 leaves >= 10x; the upper gap is the
#: cofactor mutation of E3-9 (6.6e-2) and V-E3-2's oblique-rectangle route
#: difference (2.5e-4), nine decades above.  Readings (reach win / wsl;
#: twin - numpy win / wsl) next to each bar.
_PAR_BAR = {
    "circle": 3e-13,  # reach 8.3e-15 / 7.3e-15; twin 6.4e-15 / 6.7e-15
    "circle_conical": 4e-13,  # reach 8.8e-15 / 1.2e-14; twin 7.0e-15 / 1.4e-14
    "circle_tensor_magnetic_two_layer": 3e-13,  # 7.7e-15 / 6.7e-15; 6.2e-15 / 5.3e-15
    "fillet": 3e-12,  # reach 5.1e-14 / 7.1e-14; twin 2.8e-14 / 4.9e-14
    "magnetic_tensor": 1e-13,  # reach 3.1e-15 / 2.3e-15; twin 2.9e-15 / 2.1e-15
    "multilayer": 6e-13,  # reach 1.3e-14 / 1.9e-14; twin 1.1e-14 / 1.9e-14
    "scalar_conical_lossy": 1e-13,  # 3.3e-15 / 2.7e-15; 2.8e-15 / 1.4e-15
    "scalar_pillar": 1e-13,  # reach 1.2e-15 / 3.1e-15; twin 1.6e-15 / 2.0e-15
    "sinusoid": 4e-14,  # reach 1.1e-15 / 1.0e-15; twin 3.2e-15 / 2.7e-15
    "tensor_lc": 5e-14,  # reach 1.4e-15 / 1.3e-15; twin 7.8e-16 / 3.3e-16
}


@pytest.mark.parametrize("name", sorted(_PAR))
def test_e3_2_forward_parity_twin_vs_numpy(name):
    """The twin's R / T / Jones equal the NumPy stack's to round-off.

    Bar PER FIXTURE (``_PAR_BAR``, round 2): 30x the fixture's own eig-stage
    round-off reach.  (The E3 build gated every fixture at one global 5e-12
    -- two decades above the M = 3 maximum; its 22-fixture f2 set reads
    3.4e-14 at M = 3 and 5.3e-13 at M = 4.)  The rectangles-only shape stack
    at OBLIQUE incidence is not in this set: it runs a different incident
    decomposition in the twin (V-E3-2, gated by
    ``test_e3r2_oblique_rectangles_converge_at_the_measured_rate``)."""
    fx = _PAR[name]
    kw = {k: v for k, v in fx.items() if k != "layers"}
    o, R, T, J = _stack(fx["layers"], M=3, backend="numpy", **kw).solve()
    _o, Rj, Tj, Jj = _stack(fx["layers"], M=3, **kw).solve()
    for a, b in ((Rj, R), (Tj, T), (Jj, J)):
        assert _amax(a, b) <= _PAR_BAR[name], (name, _amax(a, b))


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
    the backend off with an ImportError naming the switch (and, for the
    convenience entry, naming the entry); without
    ``jax_enable_x64`` the twin RAISES (``_require_jax_x64``) instead of
    silently computing in complex64."""
    import jax

    import lumenairy.backend as B
    monkeypatch.setattr(B, "JAX_AVAILABLE", False)
    with pytest.raises(ImportError, match="LUMENAIRY_DISABLE_JAX"):
        PMM2DStackPure(_P, _P, n_modes=3, backend="jax")
    # the convenience entry names ITSELF (round 2, verifier V-E3-4)
    with pytest.raises(ImportError, match=r"pmm_jones_2d_staggered\(backend"):
        pmm_jones_2d_staggered(_P, _P, np.ones((2, 2), complex), 1.45, 1.0,
                               0.4, _WL, degree=3, n_orders=2, backend="jax")
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


# =========================================================================== #
# ROUND 2 (VERIFY-E3, 2026-10-03) -- degenerate CLUSTERS in the reverse pass
# =========================================================================== #
# Readings: ``validation/probe_pmm2d_curved/build_e3r2/`` (``_win`` / ``_wsl``:
# Windows jax 0.11.0, WSL jax 0.10.2), every bar derived there and stated
# below with both builds' readings.
def test_e3r2_no_eig_level_rule_can_see_the_in_cluster_block():
    """WHY the fix wraps eig AND its consumer.  ``A0 = diag(1, 1, 2)`` (an
    exact pair), ``L_X(A) = Re tr(expm(A) X)`` evaluated through the
    eigenpairs.  For ``X1 = E_12`` (an in-cluster off-diagonal) and
    ``X2 = 0`` the consumer hands eig the SAME cotangent (zero, to round-off)
    -- yet the true gradients (``jax.scipy.linalg.expm``) differ by ``e``.
    No rule at the eig boundary can return both; the cluster rule
    (``rcwa._jax_eig_cluster_adjoint``) recovers both, and on a random
    similarity of ``diag(1, 1, 2, 3)`` it matches the expm oracle where the
    plain eig VJP does not.  Measured on the similarity case
    (``r9_matrix_oracle_{win,wsl}.json``): the rule 2.2e-9 / 1.6e-10
    relative, the plain VJP 0.30 / 1.01 (win / wsl -- build-dependent, as
    V-E3-1 was); bars 1e-7 / > 1e-2."""
    import jax
    import jax.numpy as jnp
    from jax.scipy.linalg import expm

    from lumenairy.elements.rcwa._core import (
        _jax_eig_cluster_adjoint,
        _jax_eig_stable,
    )

    def eig_fn(L, G):
        return _jax_eig_stable()(L)

    def via_eig(A, X, gap):
        def consumer(eigs):
            lam, V = eigs[0]
            M = V @ jnp.diag(jnp.exp(lam)) @ jnp.linalg.inv(V)
            return jnp.real(jnp.trace(M @ X))
        return _jax_eig_cluster_adjoint(eig_fn, [(A, None)], consumer,
                                        gap_rel=gap)

    def oracle(A, X):
        return jnp.real(jnp.trace(expm(A) @ X))

    A0 = jnp.diag(jnp.asarray([1.0, 1.0, 2.0], dtype=complex))
    X1 = jnp.zeros((3, 3), complex).at[0, 1].set(1.0)
    X2 = jnp.zeros((3, 3), complex)
    # the consumer's cotangent at the exact eigenpairs: identical (zero)
    lam, V = _jax_eig_stable()(A0)
    for X in (X1, X2):
        def cons(lv, X=X):
            lam_, V_ = lv
            M = V_ @ jnp.diag(jnp.exp(lam_)) @ jnp.linalg.inv(V_)
            return jnp.real(jnp.trace(M @ X))
        _v, vj = jax.vjp(cons, (lam, V))
        lb, Vb = vj(1.0)[0]
        assert float(jnp.max(jnp.abs(lb))) + float(jnp.max(jnp.abs(Vb))) \
            < 1e-14
    g1 = jax.grad(lambda A: oracle(A, X1), holomorphic=False)(
        A0.real)
    assert abs(float(g1[1, 0]) - np.e) < 1e-12     # the true gradient
    for X in (X1, X2):
        gt = np.asarray(jax.grad(lambda A, X=X: oracle(A, X))(A0.real))
        gr = np.asarray(jax.grad(lambda A, X=X: via_eig(
            A.astype(complex), X, None))(A0.real))
        assert np.max(np.abs(gr - gt)) <= 1e-7 * max(1.0, np.max(
            np.abs(gt)))
    # a random similarity of diag(1, 1, 2, 3), a random direction
    rng = np.random.default_rng(7)
    Q = jnp.asarray(rng.standard_normal((4, 4)) + 1j * rng.standard_normal(
        (4, 4)))
    D = jnp.diag(jnp.asarray([1.0, 1.0, 2.0, 3.0], dtype=complex))
    A1 = Q @ D @ jnp.linalg.inv(Q)
    B = jnp.asarray(rng.standard_normal((4, 4)) + 1j * rng.standard_normal(
        (4, 4)))
    X = jnp.asarray(rng.standard_normal((4, 4)) + 1j * rng.standard_normal(
        (4, 4)))
    gt = float(jax.grad(lambda t: oracle(A1 + t * B, X))(0.0))
    gr = float(jax.grad(lambda t: via_eig(A1 + t * B, X, None))(0.0))
    gp = float(jax.grad(lambda t: via_eig(A1 + t * B, X, 0.0))(0.0))
    assert abs(gr - gt) <= 1e-7 * abs(gt), (gr, gt)
    assert abs(gp - gt) > 1e-2 * abs(gt), (gp, gt)


_SYM = {
    "square_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.5, 3.5)]),
    "ellipse_a": (0.33, lambda x: [Ellipse(0.6, 0.6, x, 0.33, 3.5)]),
    "fillet_sq_w": (0.6, lambda x: [FilletRect(0.6, 0.6, x, 0.6, 0.1,
                                               3.5)]),
}


@functools.lru_cache(maxsize=None)
def _sym_fns(case, M=3):
    """(x0, f, jit f) of R00, T00 E_x, T00 E_y for a symmetric reference
    whose parameter BREAKS the four-fold symmetry (V-E3-1)."""
    import jax
    import jax.numpy as jnp
    x0, shp = _SYM[case]
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend="jax")
    st.add_layer(0.45, shapes=shp(x0), background_eps=1.0)
    st.set_source(_WL)
    tw = st.jax_twin()

    def f(x):
        p = tw.params()
        p["layers"][0]["shapes"] = shp(x)
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, tw.p0], T[0, tw.p0], T[1, tw.p0]])
    return x0, f, jax.jit(f)


def _jac(f, x0, gap=None):
    """jit(jacrev) of ``f`` traced with the cluster rule's ``gap_rel``
    (None = default, 0 = the rule off: the E3 build's adjoint)."""
    import jax

    import lumenairy.elements.pmm._jax_twod_staggered as JT
    JT._E3_EIG_CLUSTER_GAP_REL = gap
    try:
        return np.asarray(jax.jit(jax.jacrev(lambda x: f(x)))(x0))
    finally:
        JT._E3_EIG_CLUSTER_GAP_REL = None


def _fdv(fj, x0, scale=_P):
    """Richardson FD (h / P = 3e-4, 1e-4) of a vector function."""
    rows = []
    for hs in (3e-4, 1e-4):
        h = hs * scale
        rows.append((np.asarray(fj(x0 + h)) - np.asarray(fj(x0 - h)))
                    / (2 * h))
    return (9.0 * rows[1] - rows[0]) / 8.0


@pytest.mark.parametrize("case", sorted(_SYM))
def test_e3r2_symmetry_breaking_gradient_at_a_symmetric_cell(case):
    """V-E3-1 FIXED.  A parameter that breaks the four-fold symmetry of the
    evaluated cell (a square pillar's width alone, a circle deformed into an
    ellipse, a square fillet's width) at M = 3, AD vs the twin's own
    Richardson FD (``r2_dev_<case>_M3_final_{win,wsl}.json``):

    ==========  =====================  =====================
    case        rule on (win / wsl)    rule off (win / wsl)
    ==========  =====================  =====================
    square_w    1.9e-10 / 4.4e-10      3.1e-3 / 3.8e-2
    ellipse_a   8.6e-11 / 1.8e-11      6.5e-3 / 2.8e-2
    fillet_sq_w 4.1e-9 / 4.0e-9        2.9e-1 / 7.5e-1
    ==========  =====================  =====================

    (the square vs FD(numpy) is the verifier's own arm; for the curved cells
    FD(numpy) differs from FD(twin) by the documented frozen-grid offset,
    5e-4 / 2e-5 here.)  Bar 1e-7: 25x above the fillet's reading, four
    decades below the smallest rule-off error; the fail-before arm (the rule
    off) must exceed 1e-3."""
    x0, f, fj = _sym_fns(case)
    fd = _fdv(fj, x0)
    sc = np.max(np.abs(fd))
    on = _jac(f, x0)
    off = _jac(f, x0, gap=0.0)
    assert np.max(np.abs(on - fd)) / sc < 1e-7, (on, fd)
    assert np.max(np.abs(off - fd)) / sc > 1e-3, (off, fd)


def test_e3r2_near_symmetric_cells_are_inside_the_rule():
    """The NEAR-degenerate zone: an ellipse 1e-13 (relative) off the circle,
    d / d a.  The rule: 6.7e-11 / 5.5e-11 (win / wsl) here and <= 1.2e-10
    over the whole sweep 0 .. 1e-4 (``r6_offsym_M3_{win,wsl}.json``); bar
    1e-7.  The E3 build's adjoint (the fail-before arm) is wrong by an
    ARBITRARY amount here -- it depends on the basis LAPACK lands on inside
    the near-degenerate pair, which is the defect: 5.3e-2 (win probe),
    5.6e-3 (wsl probe), 1.1e-4 (wsl, this test's graph, 2026-10-03); bar
    > 1e-5, a decade under the smallest reading and two above the rule's
    bar."""
    import jax
    import jax.numpy as jnp
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=3, n_orders=2, backend="jax")
    st.add_layer(0.45, shapes=[Ellipse(0.6, 0.6, 0.33, 0.33, 3.5)],
                 background_eps=1.0)
    st.set_source(_WL)
    tw = st.jax_twin()
    b = 0.33 * (1.0 + 1e-13)

    def g(a):
        p = tw.params()
        p["layers"][0]["shapes"] = [Ellipse(0.6, 0.6, a, b, 3.5)]
        _o, R, T, J = st.solve(params=p)
        return jnp.stack([R[0, tw.p0], T[0, tw.p0], T[1, tw.p0]])
    fd = _fdv(jax.jit(g), 0.33)
    sc = np.max(np.abs(fd))
    assert np.max(np.abs(_jac(g, 0.33) - fd)) / sc < 1e-7
    assert np.max(np.abs(_jac(g, 0.33, gap=0.0) - fd)) / sc > 1e-5


def _rotating_eig(orig, seed):
    """``_stag_geneig_jax`` whose basis inside every EXACT cluster
    (gap <= 1e-12 max|lam|) is replaced by a random unitary rotation of it,
    and every mode by a random phase -- an equally valid eigensolver."""
    import jax
    import jax.numpy as jnp

    def eig(L, G, tau_rel=None):
        lam, V = orig(L, G, tau_rel)
        n = lam.shape[0]
        s = jnp.max(jnp.abs(lam))
        K = (jnp.abs(lam[:, None] - lam[None, :]) <= 1e-12 * s) | jnp.eye(
            n, dtype=bool)
        rng = np.random.default_rng(seed * 1000 + n)
        X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
        Y = jnp.where(K, X, 0.0)
        w, Q = jnp.linalg.eigh(jnp.conj(Y).T @ Y)
        U = Y @ ((Q * (1.0 / jnp.sqrt(w))[None, :]) @ jnp.conj(Q).T)
        return lam, V @ jax.lax.stop_gradient(U)
    return eig


def test_e3r2_gradient_is_invariant_under_a_rotation_inside_each_cluster(
        monkeypatch):
    """THE GAUGE TEST.  LAPACK's basis inside a degenerate cluster is
    arbitrary; replacing it by a random unitary rotation (and every mode by a
    random phase) must not move anything the twin returns.  Square pillar,
    d / d w, M = 3 (``r5_gauge_square_w_M3_{win,wsl}.json``): the forward
    values move by 2.6e-15 (round-off), the gradient with the cluster rule
    by 1.6e-10 / 4.3e-10 (win / wsl), the E3 build's adjoint (rule off) by
    0.23 / 0.24 -- V-E3-1 WAS this dependence.  Bars: values 1e-12, rule on
    1e-8, rule off > 1e-3."""
    import jax

    import lumenairy.elements.pmm._jax_twod_staggered as JT
    x0, f, _fj = _sym_fns("square_w")
    orig = JT._stag_geneig_jax
    base = {g: _jac(f, x0, gap=g) for g in (None, 0.0)}
    v0 = np.asarray(jax.jit(lambda x: f(x))(x0))
    monkeypatch.setattr(JT, "_stag_geneig_jax", _rotating_eig(orig, 1))
    v1 = np.asarray(jax.jit(lambda x: f(x))(x0))
    rot = {g: _jac(f, x0, gap=g) for g in (None, 0.0)}
    sc = np.max(np.abs(base[None]))
    assert np.max(np.abs(v1 - v0)) < 1e-12
    assert np.max(np.abs(rot[None] - base[None])) / sc < 1e-8
    assert np.max(np.abs(rot[0.0] - base[0.0])) / sc > 1e-3


def test_e3r2_the_rule_leaves_every_forward_value_byte_identical():
    """The rule acts in the reverse pass only: R, T and the Jones matrix
    with the rule on and off are BYTE-identical (eager and under jit; the
    probe ``r3_fwd_bytes`` compares 84 / 84 SHA-256 against the E3 build
    ``d4e92eb5`` on 14 fixtures on both builds)."""
    import jax

    import lumenairy.elements.pmm._jax_twod_staggered as JT
    x0, f, _fj = _sym_fns("square_w")
    vals = {}
    for g in (None, 0.0):
        JT._E3_EIG_CLUSTER_GAP_REL = g
        try:
            vals[g] = (np.asarray(f(x0 + 0.01)),
                       np.asarray(jax.jit(lambda x: f(x))(x0 + 0.01)))
        finally:
            JT._E3_EIG_CLUSTER_GAP_REL = None
    for a, b in zip(vals[None], vals[0.0]):
        assert np.array_equal(a, b)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MAINTAINER ITEM (round 2 of the E3 build, pre-existing, not E3): the "
    "RCWA JAX path's gradient is wrong for a SYMMETRY-BREAKING parameter at "
    "a four-fold symmetric cell -- the V-E3-1 class (the shared "
    "_jax_eig_stable VJP at a degenerate pair): 23 % (TE) / 39 % (TM) "
    "relative on Windows, 28 % / 47 % on WSL (build-dependent); the "
    "symmetry-KEEPING control is exact (<= 6.9e-10) "
    "(r4_other_twins_rcwa2d_{win,wsl}.json).  Fix: route its eigs through "
    "rcwa._jax_eig_cluster_adjoint; remove the marker with the fix."))
def test_e3r2_rcwa_jax_symmetry_breaking_gradient_at_a_symmetric_cell():
    """``rcwa_efficiency_2d`` (JAX eps_cell), a 15 x 15 pixel cell (centre
    block eps 4, side blocks 1.5, corners 1), t added to the two x-side
    blocks only, d R / d t and d T / d t of the (0, 0), (+-1, 0) orders, TE,
    vs the NumPy Richardson FD.  Bar 1e-6."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.rcwa import rcwa_efficiency_2d
    base3 = np.array([[1.0, 1.5, 1.0], [1.5, 4.0, 1.5], [1.0, 1.5, 1.0]])
    xs3 = np.zeros((3, 3))
    xs3[0, 1] = xs3[2, 1] = 1.0
    base, dirn = (np.kron(a, np.ones((5, 5))) for a in (base3, xs3))
    o, _R, _T = rcwa_efficiency_2d(_P, _P, base.astype(complex), 1.45, 1.0,
                                   0.45, _WL, n_orders_x=3, n_orders_y=3)
    o = np.asarray(o)
    idx = [int(np.nonzero((o[:, 0] == a) & (o[:, 1] == b))[0][0])
           for a, b in ((0, 0), (1, 0), (-1, 0))]

    def f(t, xp):
        eps = (xp.asarray(base) + t * xp.asarray(dirn)).astype(complex)
        _o, R, T = rcwa_efficiency_2d(_P, _P, eps, 1.45, 1.0, 0.45, _WL,
                                      n_orders_x=3, n_orders_y=3)
        return xp.concatenate([xp.stack([R[i] for i in idx]),
                               xp.stack([T[i] for i in idx])])
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    fd = _fdv(lambda t: f(t, np), 0.0, scale=1.0)
    assert np.max(np.abs(g - fd)) / np.max(np.abs(fd)) < 1e-6


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=(
    "MAINTAINER ITEM (pre-existing, documented in the W9 note of "
    "rcwa._core as 'exactly 0.0 stays unrecoverable'): the 1-D PMM twin's "
    "d / d(angle) AT EXACTLY normal incidence on a symmetric grating -- the "
    "angle splits the half-spaces' +-m pairs -- is wrong: 28 % (TE) / 590 % "
    "(TM) relative on the +-1 orders, both builds "
    "(r4_other_twins_pmm1d_{win,wsl}.json).  Fix: "
    "the cluster rule; remove the marker with the fix."))
def test_e3r2_pmm1d_jax_angle_gradient_at_normal_incidence():
    """``pmm_efficiency_1d`` (JAX), period 1.2, ridge n 2 / groove 1, duty
    0.5, depth 0.45, degree 12, TE: d R_{+-1} / d angle and d T_{+-1} /
    d angle at angle 0 vs the NumPy Richardson FD.  Bar 1e-6."""
    import jax
    import jax.numpy as jnp

    from lumenairy.elements.pmm import pmm_efficiency_1d
    o, _R, _T = pmm_efficiency_1d(_P, 2.0, 1.0, 1.45, 1.0, 0.45, 0.5, _WL,
                                  degree=12, stabilize=False)
    o = np.asarray(o)
    idx = [int(np.nonzero(o == m)[0][0]) for m in (1, -1)]

    def f(t, xp):
        _o, R, T = pmm_efficiency_1d(_P, xp.asarray(2.0 + 0j), 1.0, 1.45,
                                     1.0, 0.45, 0.5, _WL, angle=t,
                                     degree=12, stabilize=False)
        return xp.concatenate([xp.stack([R[i] for i in idx]),
                               xp.stack([T[i] for i in idx])])
    g = np.asarray(jax.jit(jax.jacrev(lambda t: f(t, jnp)))(0.0))
    fd = _fdv(lambda t: f(t, np), 0.0, scale=1.0)
    assert np.max(np.abs(g - fd)) / np.max(np.abs(fd)) < 1e-6


def test_e3r2_an_ellipse_inside_the_sliver_margin_is_poisoned_when_traced():
    """V-E3-3 for the primitive whose bounding box needed the round-2
    trace-safe ``Ellipse.bbox``: frozen at a = b = 0.5 (c = 0.6), a = 0.5995
    leaves 5e-4 P to the cell edge (sliver 1.2e-3 P) -> NaN value and
    gradient when traced; a = 0.59 finite."""
    import jax
    st = _stack([dict(thickness=0.4, shapes=[Ellipse(0.6, 0.6, 0.5, 0.5,
                                                     3.5)],
                      background_eps=1.0)], M=3)
    tw = st.jax_twin()

    def f(a):
        p = tw.params()
        p["layers"][0]["shapes"] = [Ellipse(0.6, 0.6, a, 0.5, 3.5)]
        return st.solve(params=p)[2][0, tw.p0]
    fj = jax.jit(f)
    assert np.isfinite(float(fj(0.59)))
    assert np.isnan(float(fj(0.5995)))
    assert np.isnan(float(jax.jit(jax.grad(f))(0.5995)))


def test_e3r2_oblique_rectangles_converge_at_the_measured_rate():
    """V-E3-2 (documented, not changed): a rectangles-only ``shapes=`` stack
    at OBLIQUE incidence runs the twin's identity-map route (L2 incident
    projection) and NumPy's unmapped route (least-squares overlap); they
    differ by the incident representation error, which must CONVERGE at the
    measured rate.  T, theta 0.3, the verifier's fixture: 2.5e-4 / 3.8e-5 /
    6.5e-7 at M = 3 / 4 / 5 (``v4b_rect_conical_M*_win.json``; this
    probe's ``r8_oblique_rate_{win,wsl}.json``).  Pinned: M = 3 inside
    [1e-5, 1e-3]; each step at least 3x (M 3 -> 4, measured 6.6x) and 10x
    (M 4 -> 5, measured 58x) smaller; ``geometry='static'`` reproduces the
    NumPy route to 1e-12 (8.5e-15)."""
    from lumenairy.elements.pmm._jax_twod_staggered import StagJaxTwin
    shp = [Rect(0.6, 0.55, 0.47, 0.42, 3.6)]
    d = {}
    for M in (3, 4, 5):
        _o, _R, T, _J = _stack([dict(thickness=0.4, shapes=shp,
                                     background_eps=1.0)], M=M,
                               theta=0.3).solve()
        _o, _R, Tn, _J = _stack([dict(thickness=0.4, shapes=shp,
                                      background_eps=1.0)], M=M, theta=0.3,
                                backend="numpy").solve()
        d[M] = float(np.max(np.abs(np.asarray(T) - Tn)))
        if M == 3:
            tws = StagJaxTwin(_stack([dict(thickness=0.4, shapes=shp,
                                           background_eps=1.0)], M=3,
                                     theta=0.3, backend="numpy"),
                              geometry="static")
            _o, _R, Ts, _J = tws.solve()
            assert np.max(np.abs(np.asarray(Ts) - Tn)) < 1e-12
    assert 1e-5 <= d[3] <= 1e-3, d
    assert d[4] <= d[3] / 3.0, d
    assert d[5] <= d[4] / 10.0, d


def test_e3r2_a_chained_cluster_far_from_the_reference_stays_finite():
    """REGRESSION of the rule's first draft (found by re-running the
    verifier's V-E3-3 events): a fillet frozen at r = 0.05 and evaluated at
    r = 0.00192 (1.6e-3 P, near its sliver limit) has CHAINS of
    near-degenerate eigenvalues (95 and 76 of 200 in a chain at gap 1e-6);
    a pairwise cluster mask then made a masked Gram block indefinite (min
    eigenvalue -6.3e-4) and the gradient NaN.  The clusters are now the
    connected components (transitive closure) and a non-finite lift is
    dropped.  The cell is not symmetric here, so the rule must agree with
    the plain VJP: measured 1.0e-9 relative (both 1.3e-6 from an FD that
    is outside its h^2 range, premise 16 instead of 11.4 --
    ``r10_fillet_far_from_reference_win.json``).  Bars: finite, 1e-7."""
    import jax
    st = _stack([dict(thickness=0.4, shapes=[FilletRect(0.6, 0.6, 0.6, 0.5,
                                                        0.05, 4.0)],
                      background_eps=1.0)], M=3)
    tw = st.jax_twin()

    def f(r):
        p = tw.params()
        p["layers"][0]["shapes"] = [FilletRect(0.6, 0.6, 0.6, 0.5, r, 4.0)]
        return st.solve(params=p)[2][0, tw.p0]
    on = float(_jac(f, 0.00192))
    off = float(_jac(f, 0.00192, gap=0.0))
    assert np.isfinite(on)
    assert abs(on - off) <= 1e-7 * abs(off), (on, off)
