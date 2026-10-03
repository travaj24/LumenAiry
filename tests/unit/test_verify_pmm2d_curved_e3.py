"""INDEPENDENT VERIFICATION of curved-cell Phase E3 (the JAX twin of the pure
staggered 2-D PMM): decision tests for the gaps the verifier closed
(``docs/audits/VERIFY_PMM2D_CURVED_E3_2026_10_03.md``; probes and JSON under
``validation/probe_pmm2d_curved/verify_e3/``).

Words.  TWIN = ``PMM2DStackPure(backend='jax')`` (frozen at its concrete
REFERENCE geometry); AD = its reverse-mode gradient; FD(twin) / FD(numpy) =
central differences of the twin's own forward / of the shipped NumPy solve
(whose grid moves with the parameter), Richardson-extrapolated from the
h / P = 3e-4 and 1e-4 rungs (h^2 truncation, ratio 3).  Every bar is derived
from a measurement of 2026-10-03 (Windows 11, CPython 3.14.6, numpy 2.4.4,
jax 0.11.0; WSL CPython 3.12.3, jax 0.10.2), stated next to the assertion.

Two arms were ``xfail(strict=True)`` pins of DEFECTS the verifier found
(V-E3-1, V-E3-3); round 2 of the build fixed both and removed the markers
(``docs/audits/BUILD_PMM2D_CURVED_E3_2026_10_03.md``, "Round 2").  The
regularisation arm was restated for the degenerate-cluster rule there.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import ast  # noqa: E402
import functools  # noqa: E402
import inspect  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import textwrap  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from lumenairy.backend import JAX_AVAILABLE  # noqa: E402
from lumenairy.elements.pmm import (  # noqa: E402
    Circle,
    PMM2DStackPure,
    Rect,
    SinusoidalWall,
)
from lumenairy.elements.pmm import _curvemap as CM  # noqa: E402
from lumenairy.elements.pmm import shapes2d as SH  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402

pytestmark = [
    pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed"),
    pytest.mark.filterwarnings("ignore:.*energy closure.*"),
    pytest.mark.filterwarnings("ignore:.*Rayleigh cutoff.*"),
]

_P, _WL = 1.2, 1.0


@pytest.fixture(autouse=True, scope="module")
def _jax_x64_for_this_module():
    """Hold ``jax_enable_x64`` on for this file and restore it after."""
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


def _stack(shapes, M=3, backend="jax", theta=0.0, phi=0.0, depth=0.45):
    st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2, backend=backend)
    st.add_layer(depth, shapes=shapes, background_eps=1.0)
    st.set_source(_WL, theta=theta, phi=phi)
    return st


def _q(R, T, xp):
    return xp.stack([R[0, 12], T[0, 12], T[1, 12]])


def _rich(fun, x0, scale=_P):
    rows = []
    for hs in (3e-4, 1e-4):
        h = hs * scale
        rows.append((np.asarray(fun(x0 + h)) - np.asarray(fun(x0 - h)))
                    / (2 * h))
    return (9.0 * rows[1] - rows[0]) / 8.0


@functools.lru_cache(maxsize=None)
def _twin_fns(case):
    """(jit f, jit jacrev f, numpy f) for a one-parameter shape family."""
    import jax
    import jax.numpy as jnp
    fam = {
        "square_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.5, 3.5)]),
        "rect_w": (0.5, lambda x: [Rect(0.6, 0.6, x, 0.4, 3.5)]),
        "sine_x0": (0.6, lambda x: [SinusoidalWall("x", x, 0.1, eps=3.0)]),
    }
    x0, shp = fam[case]
    st = _stack(shp(x0))
    tw = st.jax_twin()

    def f(x):
        p = tw.params()
        p["layers"][0]["shapes"] = shp(x)
        _o, R, T, J = st.solve(params=p)
        return _q(R, T, jnp)

    def fn(x):
        _o, R, T, J = _stack(shp(x), backend="numpy").solve()
        return _q(R, T, np)
    return x0, jax.jit(f), jax.jit(jax.jacrev(f)), fn, f


# =========================================================================== #
# E3-1 -- a NumPy solve never imports the twin (import-blocked subprocess)
# =========================================================================== #
def test_ve3_numpy_solves_never_import_the_twin_module():
    """With an import hook that makes importing ``_jax_twod_staggered``
    RAISE, NumPy solves of a curved, a tensor, a magnetic and an oblique
    rectangle stack still run, and the module is never imported
    (``v1_compare_{win,wsl}.json``: 164 / 164 SHA-256 equal to eae470d9 with
    the hook installed, and with jax itself blocked)."""
    code = textwrap.dedent("""
        import importlib.abc, sys
        class B(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "lumenairy.elements.pmm._jax_twod_staggered":
                    raise ImportError("blocked")
        sys.meta_path.insert(0, B())
        import numpy as np
        from lumenairy.elements.pmm import Circle, Ellipse, PMM2DStackPure, Rect
        from lumenairy.elements.rcwa._core import uniaxial_tensor
        lc = uniaxial_tensor(1.5, 1.8, np.pi / 2, phi=0.3)
        for shp, src in (([Circle(0.6, 0.6, 0.33, 3.5)], {}),
                         ([Circle(0.6, 0.6, 0.33, lc)], dict(theta=0.2)),
                         ([Ellipse(0.6, 0.6, 0.3, 0.2, 2.5, mu=1.3)], {}),
                         ([Rect(0.6, 0.55, 0.47, 0.42, 3.6)],
                          dict(theta=0.3, phi=0.45))):
            st = PMM2DStackPure(1.2, 1.2, n_substrate=1.45, n_modes=3,
                                n_orders=2)
            st.add_layer(0.4, shapes=shp, background_eps=1.0)
            st.set_source(1.0, **src)
            st.solve()
        assert "lumenairy.elements.pmm._jax_twod_staggered" not in sys.modules
        print("OK")
    """)
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, timeout=600)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), (
        r.stdout[-500:], r.stderr[-2000:])


# =========================================================================== #
# E3-5 -- a SYMMETRY-BREAKING gradient at an exactly symmetric cell (V-E3-1)
# =========================================================================== #
def test_ve3_symmetric_control_rect_width_gradient_matches_numpy():
    """The CONTROL of the arm below: a NON-square pillar (no four-fold
    degeneracy).  d / d w vs FD(numpy): 3.0e-11 relative at M = 3
    (``v6b_symbreak_rect_nonsq_w_M3_win.json``).  Bar 1e-7."""
    x0, fj, gj, fn, _f = _twin_fns("rect_w")
    g = np.asarray(gj(x0))
    fd = _rich(fn, x0)
    assert np.max(np.abs(g - fd)) / np.max(np.abs(fd)) < 1e-7


def test_ve3_symmetry_breaking_gradient_at_a_square_pillar():
    """d / d w of a SQUARE pillar (w = h = 0.5; w moved alone breaks the
    four-fold symmetry and splits the degenerate pairs to first order) vs
    FD(numpy), whose ladder satisfies the h^2 premise (rung ratios 11.375)
    and agrees with FD(twin) to 1e-10.  Bar 1e-6 (the control above: 3e-11;
    the prototype split fix of the report: 2.4e-7)."""
    x0, fj, gj, fn, _f = _twin_fns("square_w")
    g = np.asarray(gj(x0))
    fd = _rich(fn, x0)
    assert np.max(np.abs(g - fd)) / np.max(np.abs(fd)) < 1e-6


def test_ve3_the_eig_regularisation_is_load_bearing():
    """MUTATION that survived the build suite (27 / 27 pass): the eigenvector
    VJP's broadening set to ZERO.  On the non-square pillar the round-off-
    split pairs of the half-space geometric eig then carry 1 / dlam ~ 1e16
    and the width gradient is off by 0.117 relative at M = 3
    (``v6b_symbreak_rect_nonsq_w_M3_win.json``; default 3e-11).

    RESTATED in round 2 of the build (the degenerate-cluster rule): those
    round-off-split pairs are now CLUSTERS whose adjoint is evaluated at a
    lifted, resolved point, so the broadening is no longer what carries them
    -- the rule is.  Mutant = the rule off AND tau 0: must be off by > 1e-3
    (the 0.117 above); the rule on with tau 0: within 1e-7 (measured 3.0e-11,
    the FD's floor), the default within 1e-7."""
    import jax

    import lumenairy.elements.pmm._jax_twod_staggered as JT
    x0, fj, gj, fn, f = _twin_fns("rect_w")
    fd = _rich(fn, x0)
    sc = np.max(np.abs(fd))
    out = {}
    for name, gap in (("mutant", 0.0), ("rule_tau0", None)):
        JT._E3_EIG_TAU_REL = 0.0
        JT._E3_EIG_CLUSTER_GAP_REL = gap
        try:
            out[name] = np.asarray(jax.jit(jax.jacrev(lambda x: f(x)))(x0))
        finally:
            JT._E3_EIG_TAU_REL = None
            JT._E3_EIG_CLUSTER_GAP_REL = None
    assert np.max(np.abs(out["mutant"] - fd)) / sc > 1e-3
    assert np.max(np.abs(out["rule_tau0"] - fd)) / sc < 1e-7
    assert np.max(np.abs(np.asarray(gj(x0)) - fd)) / sc < 1e-7


# =========================================================================== #
# E3-3 -- a CURVED-cell gradient against FD(numpy) with NO frozen-grid offset
# =========================================================================== #
def test_ve3_sinusoid_wall_position_gradient_matches_numpy_fd():
    """The frozen grid re-parametrises a curved shape, so a curved gradient
    differs from FD(numpy) by the incident representation error -- EXCEPT
    where the non-polynomial dependence lies along an axis whose walls do
    not move: a sinusoidal wall's POSITION moves the u-wall, its wiggle runs
    along v, and the two discretisations coincide (AD vs FD(numpy)
    1.2e-10 .. 9.3e-10 at M = 3 .. 7, ``v2_analysis_win.json``).  This is the
    gate that sees a frozen-GEOMETRY mistake in the curved path that every
    AD-vs-FD(twin) gate misses: the cofactor's determinant taken from the
    REFERENCE map (forward at the reference unchanged, AD = FD(twin)) is
    off by 1.7 here (``v2_frozen_sinex_M4_tn_mutdetref_win.json``).  Bar
    1e-7."""
    x0, fj, gj, fn, _f = _twin_fns("sine_x0")
    g = np.asarray(gj(x0))
    fd = _rich(fn, x0)
    assert np.max(np.abs(g - fd)) / np.max(np.abs(fd)) < 1e-7


# =========================================================================== #
# E3-2 / F-E3-1 -- rectangles at OBLIQUE incidence: the twin is the mapped
# route (V-E3-2, a documented difference, pinned here)
# =========================================================================== #
def test_ve3_rectangles_at_oblique_incidence_run_the_mapped_route():
    """A rectangles-only shape stack runs UNMAPPED in NumPy (least-squares
    incident decomposition) but its twin (geometry='auto') runs the identity
    transfinite map (the L2 incident projection).  At normal incidence they
    agree to round-off (3e-15); at oblique incidence they differ by the
    incident representation error (2.5e-4 at M = 3, 3.8e-5 / 6.5e-7 /
    6.9e-9 at M = 4 / 5 / 6, ``v4b_rect_conical_M*_win.json``).  Pinned:
    the twin equals the NumPy IDENTITY-MAP route to round-off, the static
    twin equals the NumPy unmapped route to round-off, and the auto twin
    differs from the unmapped NumPy answer by more than 1e-6 at M = 3."""
    import jax.numpy as jnp  # noqa: F401
    from lumenairy.elements.pmm._jax_twod_staggered import StagJaxTwin
    shp = [Rect(0.6, 0.55, 0.47, 0.42, 3.6)]
    st = _stack(shp, theta=0.3, depth=0.4)
    _o, R, T, J = (np.asarray(a) for a in st.solve())
    _o, Rn, Tn, Jn = _stack(shp, theta=0.3, depth=0.4,
                            backend="numpy").solve()
    tws = StagJaxTwin(_stack(shp, theta=0.3, depth=0.4, backend="numpy"),
                      geometry="static")
    _o, Rs, Ts, Js = (np.asarray(a) for a in tws.solve())
    xw = np.array([0.0, 0.6 - 0.235, 0.6 + 0.235, _P])
    yw = np.array([0.0, 0.55 - 0.21, 0.55 + 0.21, _P])
    si = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=3, n_orders=2, cmap=CM.TransfiniteMap(xw, yw))
    c = np.ones((3, 3), complex)
    c[1, 1] = 3.6
    si.add_layer(0.4, eps_cell=c)
    si.set_source(_WL, theta=0.3)
    _o, Ri, Ti, Ji = si.solve()
    assert np.max(np.abs(T - Ti)) < 1e-12              # measured 8.1e-15
    assert np.max(np.abs(Ts - Tn)) < 1e-12             # measured 8.5e-15
    assert np.max(np.abs(T - Tn)) > 1e-6               # measured 2.5e-4


# =========================================================================== #
# E3-4 -- the merge's inside-the-cell contract inside the trace (V-E3-3)
# =========================================================================== #
def test_ve3_inside_the_cell_contract_is_poisoned_when_traced():
    """Twin frozen at r = 0.5 (c = 0.6, P = 1.2): r = 0.5995 leaves 5e-4 P
    between the outline and the cell edge (sliver 1.2e-3 P).  Concrete:
    refused by the merge (also by the NumPy stack).  Traced: must be NaN
    (value and gradient); r = 0.59 stays finite."""
    import jax
    st = _stack([Circle(0.6, 0.6, 0.5, 3.5)], depth=0.4)
    tw = st.jax_twin()

    def f(r):
        p = tw.params()
        p["layers"][0]["shapes"] = [Circle(0.6, 0.6, r, 3.5)]
        return st.solve(params=p)[2][0, tw.p0]
    fj = jax.jit(f)
    assert np.isfinite(float(fj(0.59)))
    assert np.isnan(float(fj(0.5995)))
    assert np.isnan(float(jax.jit(jax.grad(f))(0.5995)))


# =========================================================================== #
# E3-8 -- the census also refuses a RENAMED verbatim copy
# =========================================================================== #
def _bodies(mod):
    tree = ast.parse(inspect.getsource(mod))
    out = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.FunctionDef) and len(n.body) > 2:
            body = n.body
            if (isinstance(body[0], ast.Expr)
                    and isinstance(getattr(body[0], "value", None),
                                   ast.Constant)):
                body = body[1:]          # the docstring may differ
            out[n.name] = ast.dump(ast.Module(body=body, type_ignores=[]))
    return out


def test_ve3_twin_module_holds_no_renamed_copy_of_a_shared_kernel():
    """The build census (E3-8) spies that every shared kernel is REACHED and
    that no twin function carries a shared kernel's NAME; a verbatim copy
    under ANOTHER name used at one of two call sites passes both (measured:
    ``_pmm2d_project_orders`` copied as ``_proj_local`` for ``Hsub`` only --
    the build census and the library census tests stay green).  This arm
    compares function BODIES (AST, docstring dropped) of the twin module
    against every function of the modules it shares kernels with."""
    import lumenairy.elements.pmm._jax_twod_staggered as JT
    import lumenairy.elements.rcwa._core as RCC
    twin = _bodies(JT)
    shared = {}
    for mod in (TS, SH, CM, RCC):
        for k, v in _bodies(mod).items():
            shared.setdefault(v, []).append(f"{mod.__name__}.{k}")
    dup = {k: shared[v] for k, v in twin.items() if v in shared}
    assert not dup, dup


# =========================================================================== #
# API limits -- reverse mode only
# =========================================================================== #
def test_ve3_forward_mode_and_second_derivatives_raise():
    """The pencil eig is a custom-VJP: forward mode (``jax.jvp`` /
    ``jacfwd`` / ``hessian``) raises ``TypeError`` and a nested ``grad``
    raises ``NotImplementedError`` (JAX refuses non-symmetric eigenvector
    derivatives in the VJP's own primal) -- measured on both builds
    (``v4_second_M3_*.json``, ``v4_jac_M3_*.json``).  A user-facing limit the
    CHANGELOG must list (V-E3-4)."""
    import jax
    x0, fj, gj, fn, f = _twin_fns("rect_w")
    with pytest.raises(TypeError):
        jax.jvp(f, (x0,), (1.0,))
    with pytest.raises(NotImplementedError):
        jax.grad(lambda x: jax.grad(lambda y: f(y)[1])(x))(x0)


# =========================================================================== #
# F-E3-5 -- the paired convergence steps are a PARITY selection rule
# =========================================================================== #
def test_ve3_stripe_pairs_are_a_parity_selection_rule():
    """F-E3-5 explained: on the stripe Rect(0.9, 0.6, 0.6, P) both cells are
    centred on mirror planes of the structure; at normal incidence the
    excited fields are EVEN about them, and the degree step M -> M + 1 adds
    one polynomial per cell whose parity alternates -- an odd one leaves the
    answer unchanged.  TE (E_y): M = 5 -> 6 identical to 0.0e+00; TM (E_x):
    6 -> 7 to 1.6e-14.  Oblique incidence in x breaks the mirror symmetry and
    the pairing (5 -> 6 TE: 1.9e-2) -- ``v8_stripe_ladder_win.json``."""
    def r(M, theta=0.0):
        st = PMM2DStackPure(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=2)
        st.add_layer(0.4, shapes=[Rect(0.9, 0.6, 0.6, _P, 4.0)],
                     background_eps=1.0)
        st.set_source(_WL, theta=theta)
        o, R, T, J = st.solve()
        return R[:, 12]
    assert abs(r(5)[1] - r(6)[1]) < 1e-12              # TE pair
    assert abs(r(6)[0] - r(7)[0]) < 1e-12              # TM pair
    assert abs(r(5, 0.2)[1] - r(6, 0.2)[1]) > 1e-4     # oblique: no pair
