"""v5.12.0 PMM 1-D differentiability (Phase 0 + Phase 1 de-risking spike).

``pmm_efficiency_1d`` gains a JAX (differentiable) twin for the SIMPLEST
surface: a BINARY grating at NORMAL incidence, real lossless permittivity, TE
and TM, fixed geometry (``duty_cycle`` held fixed -- the moving-boundary
gradient is a later phase).  The path is opt-in: it is invoked ONLY when the
index / geometry inputs are JAX arrays (mirroring rcwa's backend detection), so
the NumPy path stays byte-identical.

This pins:

* the JAX path reproduces the NumPy path's per-order R, T to eig precision
  (the generalized modal pencil ``A x = q^2 B x`` is folded to the standard
  ``eig(B^-1 A)`` so the differentiable custom-VJP eig from rcwa applies);
* ``jax.grad`` of ``sum(T)`` / ``sum(R)`` matches a central finite difference
  for ``eps_ridge`` / ``eps_groove`` (real), ``depth`` and ``wavelength``,
  TE and TM.

REQUIRES JAX double precision: the modal eigenproblem is ill-conditioned in
single precision and JAX silently truncates complex128 -> complex64 unless
``jax_enable_x64`` is set (the solver warns; see ``_warn_if_jax_f32``).  The
JAX path requires ``stabilize=False`` (the degree-scan returns a discrete
degree via host control flow, which is non-differentiable).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.pmm import pmm_efficiency_1d  # noqa: E402

_CJ = jnp.complex128

# Three binary / normal-incidence cells, each clean (R+T == 1) at its degree
# with stabilize=False (no resonance at that fixed degree -- the JAX path takes
# a single fixed degree, no degree-scan).
CELLS = {
    "cell1": dict(period=0.8e-6, n_ridge=2.0, n_groove=1.0, n_substrate=1.45,
                  n_superstrate=1.0, depth=0.3e-6, duty_cycle=0.6,
                  wavelength=0.55e-6, degree=16),
    "cell2": dict(period=1.5e-6, n_ridge=1.8, n_groove=1.3, n_substrate=1.0,
                  n_superstrate=1.0, depth=0.5e-6, duty_cycle=0.4,
                  wavelength=0.7e-6, degree=16),
    "cell0": dict(period=1.2e-6, n_ridge=1.5, n_groove=1.0, n_substrate=1.5,
                  n_superstrate=1.0, depth=0.4e-6, duty_cycle=0.5,
                  wavelength=0.633e-6, degree=12),
}


def _solve_numpy(c, pol):
    return pmm_efficiency_1d(
        c["period"], c["n_ridge"], c["n_groove"], c["n_substrate"],
        c["n_superstrate"], c["depth"], c["duty_cycle"], c["wavelength"],
        angle=0.0, polarization=pol, degree=c["degree"], stabilize=False)


def _solve_jax(c, pol, *, n_ridge=None, n_groove=None, depth=None,
               wavelength=None, duty_cycle=None, grade=True):
    nr = c["n_ridge"] if n_ridge is None else n_ridge
    ng = c["n_groove"] if n_groove is None else n_groove
    dep = c["depth"] if depth is None else depth
    wl = c["wavelength"] if wavelength is None else wavelength
    duty = c["duty_cycle"] if duty_cycle is None else duty_cycle
    return pmm_efficiency_1d(
        c["period"], jnp.asarray(nr, _CJ), jnp.asarray(ng, _CJ),
        jnp.asarray(c["n_substrate"], _CJ), jnp.asarray(c["n_superstrate"], _CJ),
        jnp.asarray(dep), duty, jnp.asarray(wl),
        angle=0.0, polarization=pol, degree=c["degree"], stabilize=False,
        grade=grade)


# ---------------------------------------------------------------------------
# GATE-NUMPY-BITID: a plain NumPy call still hits the original code path; the
# oblique JAX plumbing must not perturb the NumPy result by a single bit.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", [0.0, 0.1, 0.4363, 0.6981])
@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", ["cell1", "cell2"])
def test_numpy_path_unchanged_normal_and_oblique(cname, pol, angle):
    """The NumPy path (no JAX inputs) is independent of the JAX twin; this pins
    that it runs + conserves at normal AND oblique (the byte-identical anchor is
    enforced out-of-band against a pre-edit snapshot in the workflow gate)."""
    c = CELLS[cname]
    o, R, T = pmm_efficiency_1d(
        c["period"], c["n_ridge"], c["n_groove"], c["n_substrate"],
        c["n_superstrate"], c["depth"], c["duty_cycle"], c["wavelength"],
        angle=angle, polarization=pol, degree=c["degree"], stabilize=False)
    assert isinstance(R, np.ndarray) and not isinstance(R, jnp.ndarray)
    assert abs(float(np.sum(R) + np.sum(T)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# GATE-FWD: the JAX twin reproduces the NumPy solve to eig precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cname", list(CELLS))
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_jax_forward_matches_numpy(cname, pol):
    c = CELLS[cname]
    o_np, R_np, T_np = _solve_numpy(c, pol)
    o_j, R_j, T_j = _solve_jax(c, pol)
    assert np.array_equal(np.asarray(o_j), o_np)
    assert np.max(np.abs(np.asarray(R_j) - R_np)) < 1e-9
    assert np.max(np.abs(np.asarray(T_j) - T_np)) < 1e-9
    # lossless: power is conserved on the JAX path too
    assert abs(float(np.sum(R_j) + np.sum(T_j)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# GATE-GRAD: jax.grad of sum(T) / sum(R) vs central finite difference
# ---------------------------------------------------------------------------

def _central_fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_eps_ridge_matches_fd(cname, pol):
    c = CELLS[cname]

    def sumT(eps):
        _, _, T = _solve_jax(c, pol, n_ridge=jnp.sqrt(eps))
        return jnp.sum(T)

    def sumR(eps):
        _, R, _ = _solve_jax(c, pol, n_ridge=jnp.sqrt(eps))
        return jnp.sum(R)

    x0 = c["n_ridge"] ** 2
    h = 1e-5
    gT = float(jax.grad(sumT)(x0))
    gR = float(jax.grad(sumR)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    fdR = float(_central_fd(lambda x: float(sumR(x)), x0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-9
    assert abs(gR - fdR) <= 1e-4 * abs(fdR) + 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_grad_wrt_eps_groove_matches_fd(pol):
    c = CELLS["cell1"]

    def sumT(eps):
        _, _, T = _solve_jax(c, pol, n_groove=jnp.sqrt(eps))
        return jnp.sum(T)

    x0 = c["n_groove"] ** 2
    h = 1e-5
    gT = float(jax.grad(sumT)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    assert np.isfinite(gT)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_depth_matches_fd(cname, pol):
    c = CELLS[cname]

    def sumT(d):
        _, _, T = _solve_jax(c, pol, depth=d)
        return jnp.sum(T)

    def sumR(d):
        _, R, _ = _solve_jax(c, pol, depth=d)
        return jnp.sum(R)

    x0 = c["depth"]
    # h small enough that the central-FD truncation error (the depth gradient
    # magnitude is ~1e6 per metre) is below the rtol; verified to converge
    # quadratically toward the autodiff value as h shrinks.
    h = 1e-10
    gT = float(jax.grad(sumT)(x0))
    gR = float(jax.grad(sumR)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    fdR = float(_central_fd(lambda x: float(sumR(x)), x0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-3
    assert abs(gR - fdR) <= 1e-4 * abs(fdR) + 1e-3


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_wavelength_matches_fd(cname, pol):
    c = CELLS[cname]

    def sumT(wl):
        _, _, T = _solve_jax(c, pol, wavelength=wl)
        return jnp.sum(T)

    def sumR(wl):
        _, R, _ = _solve_jax(c, pol, wavelength=wl)
        return jnp.sum(R)

    x0 = c["wavelength"]
    h = 1e-12
    gT = float(jax.grad(sumT)(x0))
    gR = float(jax.grad(sumR)(x0))
    fdT = float(_central_fd(lambda x: float(sumT(x)), x0, h))
    fdR = float(_central_fd(lambda x: float(sumR(x)), x0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1.0
    assert abs(gR - fdR) <= 1e-4 * abs(fdR) + 1.0


# ---------------------------------------------------------------------------
# Phase 2: d/d(duty_cycle) -- the moving-boundary (fixed-topology) gradient
# ---------------------------------------------------------------------------
# The grating wall sits EXACTLY on an element boundary, so duty is a smooth
# fixed-topology moving-mesh parameter (no remeshing, no Gibbs / blurred-step
# non-smoothness).  Route B: the d_wall-dependent geometry (element boundaries,
# Jacobians, masses, stiffnesses, projection phases) is rebuilt in jnp when
# duty is traced, so the moving Jacobians + projection phases carry the
# gradient; the element COUNT stays fixed (jit-safe).


@pytest.mark.parametrize("grade", [False, True])
@pytest.mark.parametrize("cname", list(CELLS))
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_jax_forward_matches_numpy_duty_traced(cname, pol, grade):
    """GATE-FWD with duty_cycle as a JAX input: the duty-traced jnp twin
    (moving-mesh masses + projection) still reproduces the NumPy solve."""
    c = CELLS[cname]
    o_np, R_np, T_np = pmm_efficiency_1d(
        c["period"], c["n_ridge"], c["n_groove"], c["n_substrate"],
        c["n_superstrate"], c["depth"], c["duty_cycle"], c["wavelength"],
        angle=0.0, polarization=pol, degree=c["degree"], stabilize=False,
        grade=grade)
    o_j, R_j, T_j = _solve_jax(
        c, pol, duty_cycle=jnp.asarray(c["duty_cycle"]), grade=grade)
    assert np.array_equal(np.asarray(o_j), o_np)
    assert np.max(np.abs(np.asarray(R_j) - R_np)) < 1e-9
    assert np.max(np.abs(np.asarray(T_j) - T_np)) < 1e-9
    assert abs(float(np.sum(R_j) + np.sum(T_j)) - 1.0) < 1e-6


@pytest.mark.parametrize("grade", [False, True])
@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_duty_cycle_matches_fd(cname, pol, grade):
    """GATE-DUTY-GRAD: jax.grad of sum(T) / sum(R) w.r.t. duty_cycle vs a
    central FD that PHYSICALLY moves the wall (h ~ 1e-4 in duty).  TE + TM, all
    cells, grade=False AND grade=True."""
    c = CELLS[cname]

    def sumT(duty):
        _, _, T = _solve_jax(c, pol, duty_cycle=duty, grade=grade)
        return jnp.sum(T)

    def sumR(duty):
        _, R, _ = _solve_jax(c, pol, duty_cycle=duty, grade=grade)
        return jnp.sum(R)

    x0 = c["duty_cycle"]
    h = 1e-4
    gT = float(jax.grad(sumT)(jnp.asarray(x0)))
    gR = float(jax.grad(sumR)(jnp.asarray(x0)))
    fdT = float(_central_fd(lambda x: float(sumT(jnp.asarray(x))), x0, h))
    fdR = float(_central_fd(lambda x: float(sumR(jnp.asarray(x))), x0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-3 * abs(fdT) + 1e-9
    assert abs(gR - fdR) <= 1e-3 * abs(fdR) + 1e-9


def test_grad_wrt_duty_does_not_disturb_phase1():
    """GATE-PH1-PRESERVED: with duty_cycle a plain python float (NOT traced),
    the eps_ridge / depth / wavelength gradients still flow + match FD."""
    c = CELLS["cell1"]
    h_eps, h_dep, h_wl = 1e-5, 1e-10, 1e-12

    def sumT_eps(eps):
        _, _, T = _solve_jax(c, "tm", n_ridge=jnp.sqrt(eps))
        return jnp.sum(T)

    def sumT_dep(dep):
        _, _, T = _solve_jax(c, "te", depth=dep)
        return jnp.sum(T)

    def sumT_wl(wl):
        _, _, T = _solve_jax(c, "te", wavelength=wl)
        return jnp.sum(T)

    x = c["n_ridge"] ** 2
    g = float(jax.grad(sumT_eps)(x))
    fd = float(_central_fd(lambda v: float(sumT_eps(v)), x, h_eps))
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-9

    x = c["depth"]
    g = float(jax.grad(sumT_dep)(x))
    fd = float(_central_fd(lambda v: float(sumT_dep(v)), x, h_dep))
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-3

    x = c["wavelength"]
    g = float(jax.grad(sumT_wl)(x))
    fd = float(_central_fd(lambda v: float(sumT_wl(v)), x, h_wl))
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1.0


@pytest.mark.parametrize("duty", [0.05, 0.95])
def test_duty_near_edges_no_nan(duty):
    """GATE-DUTY-GUARD: duty near 0 / 1 (but interior) stays finite -- forward
    AND gradient."""
    c = CELLS["cell0"]
    _, R, T = _solve_jax(c, "te", duty_cycle=jnp.asarray(duty))
    assert np.all(np.isfinite(np.asarray(T)))
    assert np.all(np.isfinite(np.asarray(R)))

    def sumT(d):
        _, _, T = _solve_jax(c, "te", duty_cycle=d)
        return jnp.sum(T)

    g = float(jax.grad(sumT)(jnp.asarray(duty)))
    assert np.isfinite(g)


@pytest.mark.parametrize("duty", [0.0, 1.0])
def test_degenerate_duty_raises(duty):
    """GATE-DUTY-GUARD: a zero-width ridge / groove collapses an element
    (singular Jacobian); the JAX path raises rather than NaN-ing."""
    c = CELLS["cell0"]
    with pytest.raises(ValueError):
        _solve_jax(c, "te", duty_cycle=jnp.asarray(duty))


def test_jit_grad_wrt_duty():
    """GATE-JIT: jax.jit(jax.grad(...)) w.r.t. duty_cycle compiles + is
    finite (the inverse-design loop wraps the solve in jit)."""
    c = CELLS["cell1"]

    def loss(duty):
        _, _, T = pmm_efficiency_1d(
            c["period"], jnp.asarray(c["n_ridge"], _CJ),
            jnp.asarray(c["n_groove"], _CJ),
            jnp.asarray(c["n_substrate"], _CJ),
            jnp.asarray(c["n_superstrate"], _CJ), jnp.asarray(c["depth"]),
            duty, jnp.asarray(c["wavelength"]), angle=0.0, polarization="te",
            degree=c["degree"], stabilize=False)
        return jnp.sum(T)

    g = jax.jit(jax.grad(loss))(jnp.asarray(c["duty_cycle"]))
    assert np.isfinite(float(g))


# ---------------------------------------------------------------------------
# Guard rails: the JAX path raises a precise error outside its validated scope
# ---------------------------------------------------------------------------

def test_jax_stabilize_true_raises():
    c = CELLS["cell1"]
    with pytest.raises(ValueError):
        pmm_efficiency_1d(
            c["period"], jnp.asarray(c["n_ridge"], _CJ), c["n_groove"],
            c["n_substrate"], c["n_superstrate"], c["depth"], c["duty_cycle"],
            c["wavelength"], angle=0.0, polarization="te", degree=c["degree"],
            stabilize=True)


def test_jax_multi_element_raises():
    c = CELLS["cell1"]
    with pytest.raises(NotImplementedError):
        pmm_efficiency_1d(
            c["period"], jnp.asarray(c["n_ridge"], _CJ), c["n_groove"],
            c["n_substrate"], c["n_superstrate"], c["depth"], c["duty_cycle"],
            c["wavelength"], angle=0.0, polarization="te", degree=c["degree"],
            elements_per_region=2, stabilize=False)


def test_jax_grad_is_jittable():
    """A real inverse-design loop wraps the solve in jax.jit; the single-level
    jax.jit(jax.grad(...)) path must compile and run."""
    c = CELLS["cell1"]

    def loss(n_ridge, depth):
        _, _, T = pmm_efficiency_1d(
            c["period"], n_ridge, jnp.asarray(c["n_groove"], _CJ),
            jnp.asarray(c["n_substrate"], _CJ),
            jnp.asarray(c["n_superstrate"], _CJ), depth, c["duty_cycle"],
            jnp.asarray(c["wavelength"]), angle=0.0, polarization="te",
            degree=c["degree"], stabilize=False)
        return jnp.sum(T)

    g = jax.jit(jax.grad(loss, argnums=(0, 1)))(
        jnp.asarray(c["n_ridge"], _CJ), jnp.asarray(c["depth"]))
    assert np.isfinite(complex(g[0]))
    assert np.isfinite(float(jnp.real(g[1])))


# ===========================================================================
# Phase 3: OBLIQUE incidence + COMPLEX/LOSSY eps gradients
# ===========================================================================
# The JAX twin lifts the angle==0 guard: kx0 = Re(n_sup) sin(angle) k0 is a
# TRACED jnp scalar, so d/d(angle) and d/d(n_superstrate) flow through the modal
# operator's Bloch convection (-1j kx0 (C - C^T) + kx0^2 mass), the per-order
# wavenumbers (kx0 + mG)/k0, and the oblique incident-flux normaliser.  This is
# a faithful transcription of the already-correct NumPy oblique solve
# (_pmm_solve_core / _sem_modes), NOT a re-derivation.  Lossy (Im(eps) > 0) eps
# breaks energy conservation, so the lossy gate validates the per-order R/T AND
# the absorbed fraction A = 1 - sumR - sumT against the NumPy oracle (the
# lossless-trap killer), NOT an energy check.

_ANGLES = [0.1, 0.4363, 0.6981]                # ~5.7 / 25 / 40 degrees


def _solve_jax_oblique(c, pol, angle, *, n_ridge=None, n_superstrate=None,
                       trace_angle=True):
    """Library JAX solve at oblique.  ``angle`` is fed as a jnp array (so the
    JAX path triggers even when no index input is traced) unless
    trace_angle=False."""
    nr = c["n_ridge"] if n_ridge is None else n_ridge
    nsup = c["n_superstrate"] if n_superstrate is None else n_superstrate
    ang = jnp.asarray(angle) if trace_angle else angle
    return pmm_efficiency_1d(
        c["period"], jnp.asarray(nr, _CJ), jnp.asarray(c["n_groove"], _CJ),
        jnp.asarray(c["n_substrate"], _CJ), jnp.asarray(nsup, _CJ),
        jnp.asarray(c["depth"]), c["duty_cycle"], jnp.asarray(c["wavelength"]),
        angle=ang, polarization=pol, degree=c["degree"], stabilize=False)


@pytest.mark.parametrize("angle", _ANGLES)
@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_jax_forward_oblique_matches_numpy(cname, pol, angle):
    """GATE-FWD-OBLIQUE: the jnp twin reproduces the NumPy oblique solve
    per-order (TE + TM) to eig precision, across angles."""
    c = CELLS[cname]
    o_np, R_np, T_np = pmm_efficiency_1d(
        c["period"], c["n_ridge"], c["n_groove"], c["n_substrate"],
        c["n_superstrate"], c["depth"], c["duty_cycle"], c["wavelength"],
        angle=angle, polarization=pol, degree=c["degree"], stabilize=False)
    o_j, R_j, T_j = _solve_jax_oblique(c, pol, angle)
    assert np.array_equal(np.asarray(o_j), o_np)
    assert np.max(np.abs(np.asarray(R_j) - R_np)) < 1e-10
    assert np.max(np.abs(np.asarray(T_j) - T_np)) < 1e-10
    # these cells are lossless -> power still conserved at oblique
    assert abs(float(np.sum(R_j) + np.sum(T_j)) - 1.0) < 1e-6


@pytest.mark.parametrize("pol", ["te", "tm"])
@pytest.mark.parametrize("cname", list(CELLS))
def test_grad_wrt_angle_matches_fd(cname, pol):
    """GATE-GRAD-ANGLE: jax.grad of sum(T) / sum(R) w.r.t. the incident angle
    vs central FD, TE + TM."""
    c = CELLS[cname]
    a0 = 0.4363

    def sumT(a):
        _, _, T = _solve_jax_oblique(c, pol, a)
        return jnp.sum(T)

    def sumR(a):
        _, R, _ = _solve_jax_oblique(c, pol, a)
        return jnp.sum(R)

    h = 1e-6
    gT = float(jax.grad(sumT)(jnp.asarray(a0)))
    gR = float(jax.grad(sumR)(jnp.asarray(a0)))
    fdT = float(_central_fd(lambda x: float(sumT(jnp.asarray(x))), a0, h))
    fdR = float(_central_fd(lambda x: float(sumR(jnp.asarray(x))), a0, h))
    assert np.isfinite(gT) and np.isfinite(gR)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-9
    assert abs(gR - fdR) <= 1e-4 * abs(fdR) + 1e-9


@pytest.mark.parametrize("pol", ["te", "tm"])
def test_grad_wrt_n_superstrate_oblique_matches_fd(pol):
    """GATE-GRAD-ANGLE (n_sup half): at oblique, n_superstrate enters BOTH the
    Bloch kx0 (via Re(n_sup)) and the incident flux; jax.grad vs FD."""
    # superstrate index > 1 so it actually drives kx0 / reflection.
    c = dict(CELLS["cell1"], n_superstrate=1.3)
    a0 = 0.4363

    def sumT(ns):
        nsc = jnp.asarray(ns, _CJ)
        _, _, T = _solve_jax_oblique(c, pol, a0, n_superstrate=nsc,
                                     trace_angle=False)
        return jnp.sum(T)

    x0 = float(c["n_superstrate"])
    h = 1e-6
    gT = float(jax.grad(sumT)(jnp.asarray(x0)))
    fdT = float(_central_fd(lambda x: float(sumT(jnp.asarray(x))), x0, h))
    assert np.isfinite(gT)
    assert abs(gT - fdT) <= 1e-4 * abs(fdT) + 1e-9


# ---- COMPLEX / LOSSY eps gradients (holomorphic=False) --------------------

# A lossy ridge (Im(n) > 0): n = 2.0 + 0.3j -> eps = 3.91 + 1.2j.  The cell
# absorbs, so R + T < 1; the absorbed fraction A = 1 - sumR - sumT is the
# physically-meaningful loss objective.
_LOSSY_N_RIDGE = 2.0 + 0.3j


def _solve_jax_lossy(c, pol, angle, eps_re, eps_im):
    """Library JAX oblique solve with a lossy ridge eps = eps_re + i eps_im,
    re / im fed as SEPARATE real tracers (holomorphic=False differentiation)."""
    n_ridge = jnp.sqrt(eps_re + 1j * eps_im)
    return pmm_efficiency_1d(
        c["period"], n_ridge, jnp.asarray(c["n_groove"], _CJ),
        jnp.asarray(c["n_substrate"], _CJ),
        jnp.asarray(c["n_superstrate"], _CJ),
        jnp.asarray(c["depth"]), c["duty_cycle"], jnp.asarray(c["wavelength"]),
        angle=jnp.asarray(angle), polarization=pol, degree=c["degree"],
        stabilize=False)


@pytest.mark.parametrize("angle", [0.1, 0.4363])
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_grad_wrt_complex_eps_matches_fd(pol, angle):
    """GATE-GRAD-COMPLEX: d(absorbed)/d(Re eps) and d(absorbed)/d(Im eps) of a
    LOSSY ridge (holomorphic=False, real + imag parts fed as separate real
    tracers) vs central FD."""
    c = CELLS["cell0"]
    eps0 = _LOSSY_N_RIDGE ** 2
    re0, im0 = float(np.real(eps0)), float(np.imag(eps0))

    def absorbed(re, im):
        _, R, T = _solve_jax_lossy(c, pol, angle, re, im)
        return 1.0 - jnp.sum(R) - jnp.sum(T)

    h = 1e-5
    gre = float(jax.grad(absorbed, argnums=0)(jnp.asarray(re0),
                                              jnp.asarray(im0)))
    gim = float(jax.grad(absorbed, argnums=1)(jnp.asarray(re0),
                                              jnp.asarray(im0)))
    fdre = float(_central_fd(
        lambda x: float(absorbed(jnp.asarray(x), jnp.asarray(im0))), re0, h))
    fdim = float(_central_fd(
        lambda x: float(absorbed(jnp.asarray(re0), jnp.asarray(x))), im0, h))
    assert np.isfinite(gre) and np.isfinite(gim)
    assert abs(gre - fdre) <= 1e-3 * abs(fdre) + 1e-9
    assert abs(gim - fdim) <= 1e-3 * abs(fdim) + 1e-9


@pytest.mark.parametrize("angle", [0.1, 0.4363])
@pytest.mark.parametrize("pol", ["te", "tm"])
def test_lossy_oracle_per_order_and_absorbed(pol, angle):
    """GATE-LOSSY-ORACLE (the lossless-trap killer): a LOSSY cell at oblique --
    per-order R / T AND the absorbed fraction A = 1 - sumR - sumT match the
    NumPy oracle (NOT an energy check; a lossy cell does not conserve power)."""
    c = CELLS["cell0"]
    o_np, R_np, T_np = pmm_efficiency_1d(
        c["period"], _LOSSY_N_RIDGE, c["n_groove"], c["n_substrate"],
        c["n_superstrate"], c["depth"], c["duty_cycle"], c["wavelength"],
        angle=angle, polarization=pol, degree=c["degree"], stabilize=False)
    A_np = 1.0 - float(np.sum(R_np)) - float(np.sum(T_np))
    eps0 = _LOSSY_N_RIDGE ** 2
    o_j, R_j, T_j = _solve_jax_lossy(c, pol, angle, float(np.real(eps0)),
                                     float(np.imag(eps0)))
    A_j = 1.0 - float(np.sum(R_j)) - float(np.sum(T_j))
    assert np.array_equal(np.asarray(o_j), o_np)
    # the cell genuinely absorbs (defeats the lossless trap -- a real absorbed
    # fraction the oracle must reproduce, not an auto-balanced unit total).
    assert A_np > 0.1
    assert np.max(np.abs(np.asarray(R_j) - R_np)) < 1e-6
    assert np.max(np.abs(np.asarray(T_j) - T_np)) < 1e-6
    assert abs(A_j - A_np) < 1e-6


# ---- jit + regression guards ----------------------------------------------

def test_jit_grad_wrt_angle_and_eps():
    """GATE-JIT: jax.jit(jax.grad) w.r.t. angle AND w.r.t. (complex) eps
    compiles + is finite at oblique."""
    c = CELLS["cell1"]

    def loss_angle(a):
        _, _, T = _solve_jax_oblique(c, "tm", a)
        return jnp.sum(T)

    ga = jax.jit(jax.grad(loss_angle))(jnp.asarray(0.4363))
    assert np.isfinite(float(ga))

    def loss_eps(re, im):
        _, R, T = _solve_jax_lossy(c, "te", 0.4363, re, im)
        return 1.0 - jnp.sum(R) - jnp.sum(T)

    eps0 = _LOSSY_N_RIDGE ** 2
    g = jax.jit(jax.grad(loss_eps, argnums=(0, 1)))(
        jnp.asarray(float(np.real(eps0))), jnp.asarray(float(np.imag(eps0))))
    assert np.isfinite(float(g[0])) and np.isfinite(float(g[1]))


def test_oblique_does_not_disturb_normal_incidence():
    """GATE-PH012-PRESERVED: the Phase 0-2 normal-incidence gradients
    (eps_ridge, depth, wavelength, duty) still match FD after the oblique
    plumbing -- the angle==0 path is byte-equal (python-literal kx0 = 0.0
    skips the convection)."""
    c = CELLS["cell1"]

    # eps_ridge (TM), normal incidence
    def sumT_eps(eps):
        _, _, T = _solve_jax(c, "tm", n_ridge=jnp.sqrt(eps))
        return jnp.sum(T)

    x = c["n_ridge"] ** 2
    g = float(jax.grad(sumT_eps)(x))
    fd = float(_central_fd(lambda v: float(sumT_eps(v)), x, 1e-5))
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-9

    # depth (TE), normal incidence
    def sumT_dep(dep):
        _, _, T = _solve_jax(c, "te", depth=dep)
        return jnp.sum(T)

    x = c["depth"]
    g = float(jax.grad(sumT_dep)(x))
    fd = float(_central_fd(lambda v: float(sumT_dep(v)), x, 1e-10))
    assert abs(g - fd) <= 1e-4 * abs(fd) + 1e-3
