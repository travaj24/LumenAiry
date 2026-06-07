"""v5.12.0 PMM Jones differentiability (Phase 4).

``pmm_jones_1d`` (the IN-PLANE anisotropic 2x2-tensor binary grating, full Jones
response) gains a JAX (differentiable) twin: the 2x2 complex Jones reflection
matrix and the (2, M) ``R_eff`` / ``T_eff`` are ``jax.grad``-able w.r.t. the
in-plane tensor entries (``exx, exy, eyx, eyy, ezz`` -- including the
off-diagonal ``exy`` / ``eyx`` cross-pol coupling), ``depth``, ``wavelength``,
``angle`` and the half-space indices, for the VERTICAL (slant == 0) wall at
NORMAL or OBLIQUE incidence, lossless or lossy eps.

Unlike the scalar generalized pencil, the Jones modal solver is a STANDARD eig
of the 2n x 2n coupled ``[Ex; Ey]`` block ``Mbig = -P@Q``, so the rcwa
custom-VJP eig applies directly (no generalized fold).  The path is opt-in:
invoked ONLY when a tensor / geometry / angle input is a JAX array (mirroring
the scalar ``pmm_efficiency_1d`` and rcwa), so the NumPy path stays
byte-identical.

This pins:

* GATE-FWD-JONES: the JAX twin reproduces the NumPy ``R_eff`` / ``T_eff`` AND
  the 2x2 complex ``jones`` (real + imag) to eig precision, normal + oblique,
  for a diagonal, an off-diagonal (cross-pol) and a lossy tensor;
* GATE-GRAD-JONES: ``jax.grad`` of a Jones FOM (``|jones[0,1]|^2`` cross-pol and
  ``sum(T)``) matches a central finite difference for the real tensor entries
  ``exx / exy / eyx / eyy`` (and the imaginary part of a lossy entry), ``depth``
  and the oblique ``angle``; a TE-equivalent diagonal tensor reduces to the
  scalar TE grad;
* GATE-LOSSY-JONES: a lossy tensor's per-order R/T AND absorbed fraction match
  the NumPy oracle (NOT an energy tripwire -- the lossless-trap killer);
* GATE-JIT: ``jax.jit`` of the grad of a Jones FOM compiles and is finite.

AUTODIFF HAZARD (documented): at NORMAL incidence ``np.linalg.eig`` mixes the
exactly-degenerate ``Ex`` / ``Ey``-dominant modes in the degenerate subspace.
The Jones OBSERVABLES (|J|^2, efficiencies) are GAUGE-INVARIANT so their grads
w.r.t. the tensor / depth are clean there; only ``d/d(angle)`` AT angle == 0 is
the symmetry-protected ZERO direction (sum(R), |J|^2 are even in angle), where a
tiny ~1e-5 AD artifact sits against a true-zero FD -- so the angle grad is
validated at OBLIQUE incidence (clean to ~1e-8) per the documented caveat.

REQUIRES JAX double precision (``jax_enable_x64``); the modal eig is
ill-conditioned in single precision.  The JAX path requires ``stabilize=False``
(the degree-scan is non-differentiable host control flow) and a degree where the
NumPy solve already conserves energy (the PMM has isolated-degree resonances).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from lumenairy.elements.pmm import (  # noqa: E402
    pmm_jones_1d, pmm_efficiency_1d)

_CJ = jnp.complex128


def _uniaxial(eps_o, eps_e, theta):
    """A (3, 3) IN-PLANE permittivity tensor (uniaxial director rotated by
    ``theta`` in the x-y plane) -- gives nonzero off-diagonal ``exy`` / ``eyx``
    (the cross-pol coupling) for ``theta`` not a multiple of pi/2."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    D = np.diag([eps_e, eps_o, eps_o]).astype(complex)
    return R @ D @ R.T


# Three binary in-plane-tensor cells, each ENERGY-PASSIVE at degree 9 with
# stabilize=False at BOTH normal and the oblique test angle (degree 9 is
# resonance-free for all three -- the JAX path takes a single fixed degree).
DIAG_ER = np.diag([2.5, 2.1, 2.3]).astype(complex)
DIAG_EG = np.eye(3).astype(complex)
OFF_ER = _uniaxial(2.0, 2.8, 0.5)               # cross-pol (exy = eyx != 0)
OFF_EG = np.diag([1.2, 1.2, 1.2]).astype(complex)
LOSSY_ER = _uniaxial(2.0 + 0.1j, 2.6 + 0.05j, 0.4)
LOSSY_EG = np.eye(3).astype(complex)

CELLS = {
    "diag": dict(period=1.0e-6, er=DIAG_ER, eg=DIAG_EG, n_sub=1.45, n_sup=1.0,
                 depth=0.35e-6, duty=0.5, wl=0.633e-6, degree=9),
    "offdiag": dict(period=1.0e-6, er=OFF_ER, eg=OFF_EG, n_sub=1.5, n_sup=1.0,
                    depth=0.4e-6, duty=0.55, wl=0.6e-6, degree=9),
    "lossy": dict(period=0.9e-6, er=LOSSY_ER, eg=LOSSY_EG, n_sub=1.45,
                  n_sup=1.0, depth=0.3e-6, duty=0.5, wl=0.55e-6, degree=9),
}
ANGLES = [0.0, 0.35]


def _solve_numpy(c, angle):
    return pmm_jones_1d(
        c["period"], c["er"], c["eg"], c["n_sub"], c["n_sup"], c["depth"],
        c["duty"], c["wl"], angle=angle, degree=c["degree"], stabilize=False)


def _solve_jax(c, angle, *, er=None, eg=None, depth=None, wl=None,
               n_sub=None, n_sup=None, angle_arr=None):
    erj = jnp.asarray(c["er"] if er is None else er, _CJ)
    egj = jnp.asarray(c["eg"] if eg is None else eg, _CJ)
    dep = jnp.asarray(c["depth"] if depth is None else depth)
    wlj = jnp.asarray(c["wl"] if wl is None else wl)
    nsub = jnp.asarray(c["n_sub"] if n_sub is None else n_sub, _CJ)
    nsup = jnp.asarray(c["n_sup"] if n_sup is None else n_sup, _CJ)
    ang = (angle_arr if angle_arr is not None
           else (jnp.asarray(angle) if angle != 0.0 else 0.0))
    return pmm_jones_1d(
        c["period"], erj, egj, nsub, nsup, dep, c["duty"], wlj,
        angle=ang, degree=c["degree"], stabilize=False)


# ---------------------------------------------------------------------------
# GATE-NUMPY-BITID: a plain NumPy call still hits the original code path; the
# JAX Jones plumbing must not perturb the NumPy result (the byte-identical
# anchor is enforced out-of-band against a pre-edit snapshot in the workflow).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cname", list(CELLS))
@pytest.mark.parametrize("angle", ANGLES)
def test_numpy_path_unchanged(cname, angle):
    """The NumPy path (no JAX inputs) runs + is well-formed at normal AND
    oblique; returns plain NumPy arrays (not JAX)."""
    c = CELLS[cname]
    o, R, T, J = _solve_numpy(c, angle)
    assert isinstance(R, np.ndarray) and not isinstance(R, jnp.ndarray)
    assert J.shape == (2, 2) and R.shape[0] == 2 and T.shape[0] == 2
    # passive (lossless cells exactly, lossy <= 1 per incident pol)
    for col in range(2):
        assert float(np.real(R[col].sum() + T[col].sum())) <= 1.0 + 1e-6


# ---------------------------------------------------------------------------
# GATE-FWD-JONES: the JAX twin reproduces R_eff, T_eff AND the 2x2 jones
# (real + imag) to eig precision, normal + oblique.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cname", list(CELLS))
@pytest.mark.parametrize("angle", ANGLES)
def test_jax_forward_matches_numpy(cname, angle):
    c = CELLS[cname]
    o_np, R_np, T_np, J_np = _solve_numpy(c, angle)
    o_j, R_j, T_j, J_j = _solve_jax(c, angle)
    assert np.array_equal(np.asarray(o_j), o_np)
    assert np.max(np.abs(np.asarray(R_j) - R_np)) < 1e-10
    assert np.max(np.abs(np.asarray(T_j) - T_np)) < 1e-10
    # the full COMPLEX Jones matrix (real + imag), incl off-diagonal cross-pol
    assert np.max(np.abs(np.asarray(J_j) - J_np)) < 1e-10


# ---------------------------------------------------------------------------
# GATE-LOSSY-JONES (the lossless-trap killer): per-order R/T + absorbed
# fraction vs the NumPy oracle, NOT an energy tripwire.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("angle", ANGLES)
def test_lossy_per_order_and_absorbed_fraction(angle):
    c = CELLS["lossy"]
    o_np, R_np, T_np, _ = _solve_numpy(c, angle)
    o_j, R_j, T_j, _ = _solve_jax(c, angle)
    # per-order R/T match the oracle
    assert np.max(np.abs(np.asarray(R_j) - R_np)) < 1e-6
    assert np.max(np.abs(np.asarray(T_j) - T_np)) < 1e-6
    # absorbed fraction (per incident polarization) matches, and is NONZERO
    abs_np = 1.0 - (np.asarray(R_np).sum(1) + np.asarray(T_np).sum(1))
    abs_j = 1.0 - (np.asarray(R_j).sum(1) + np.asarray(T_j).sum(1))
    assert np.max(np.abs(abs_j - abs_np)) < 1e-6
    assert np.all(abs_np > 1e-3)             # genuinely lossy (not the trap)


# ---------------------------------------------------------------------------
# GATE-GRAD-JONES: jax.grad of a Jones FOM vs central finite difference.
# ---------------------------------------------------------------------------

def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


def _grad_tensor_entry(fom_of_er, er0, i, j, *, imag=False, h=1e-5):
    """(AD, FD) of ``fom_of_er`` w.r.t. eps_ridge[i, j] (real or imag part).

    JAX ``holomorphic=False`` grad returns ``g`` with ``df = Re(conj(g) dz)`` so
    ``df/d Re(z) = Re(g)`` and ``df/d Im(z) = -Im(g)``."""
    erj = jnp.asarray(er0, _CJ)
    g = jax.grad(fom_of_er, holomorphic=False)(erj)
    ad = float(-jnp.imag(g[i, j])) if imag else float(jnp.real(g[i, j]))
    step = (1j * h) if imag else h
    fp = float(fom_of_er(erj.at[i, j].add(step)))
    fm = float(fom_of_er(erj.at[i, j].add(-step)))
    return ad, (fp - fm) / (2.0 * h)


@pytest.mark.parametrize("entry", [("exx", 0, 0), ("exy", 0, 1),
                                   ("eyx", 1, 0), ("eyy", 1, 1)])
def test_grad_wrt_tensor_entries_crosspol_fom(entry):
    """d|jones[0,1]|^2 / d(eps_ridge[i,j]) (real part) vs FD -- the cross-pol
    FOM at oblique incidence (the off-diagonal exy / eyx couplings carry the
    cross-pol gradient)."""
    _, i, j = entry
    c = CELLS["offdiag"]

    def fom(er):
        _, _, _, J = _solve_jax(c, 0.35, er=er)
        return jnp.abs(J[0, 1]) ** 2

    ad, fd = _grad_tensor_entry(fom, c["er"], i, j)
    assert np.isfinite(ad)
    assert abs(ad - fd) <= 1e-3 * abs(fd) + 1e-10


@pytest.mark.parametrize("entry", [("exx", 0, 0), ("exy", 0, 1),
                                   ("eyx", 1, 0), ("eyy", 1, 1)])
def test_grad_wrt_tensor_entries_sumT_fom(entry):
    """d sum(T) / d(eps_ridge[i,j]) (real part) vs FD, oblique."""
    _, i, j = entry
    c = CELLS["offdiag"]

    def fom(er):
        _, _, T, _ = _solve_jax(c, 0.35, er=er)
        return jnp.sum(T)

    ad, fd = _grad_tensor_entry(fom, c["er"], i, j)
    assert np.isfinite(ad)
    assert abs(ad - fd) <= 1e-3 * abs(fd) + 1e-10


def test_grad_wrt_tensor_entries_normal_incidence():
    """The tensor / cross-pol grads are CLEAN at NORMAL incidence too (the
    Jones observables are gauge-invariant, so the degenerate-subspace eig mixing
    does not corrupt them)."""
    c = CELLS["offdiag"]

    def fom(er):
        _, R, _, J = _solve_jax(c, 0.0, er=er)
        return jnp.abs(J[0, 1]) ** 2 + jnp.sum(R)

    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ad, fd = _grad_tensor_entry(fom, c["er"], i, j)
        assert np.isfinite(ad)
        assert abs(ad - fd) <= 1e-3 * abs(fd) + 1e-10


def test_grad_wrt_lossy_imag_part():
    """d|jones[0,1]|^2 / d(Im eps_ridge[0,0]) vs FD -- the lossy-entry imaginary
    gradient (the absorption channel)."""
    c = CELLS["lossy"]

    def fom(er):
        _, _, _, J = _solve_jax(c, 0.35, er=er)
        return jnp.abs(J[0, 1]) ** 2

    ad, fd = _grad_tensor_entry(fom, c["er"], 0, 0, imag=True)
    assert np.isfinite(ad)
    assert abs(ad - fd) <= 1e-3 * abs(fd) + 1e-10


def test_grad_wrt_depth():
    c = CELLS["offdiag"]

    def fom(depth):
        _, _, T, _ = _solve_jax(c, 0.35, depth=depth)
        return jnp.sum(T)

    x0 = jnp.asarray(c["depth"])
    ad = float(jax.grad(fom)(x0))
    fd = float(_fd(lambda x: float(fom(x)), x0, c["depth"] * 1e-4))
    assert np.isfinite(ad)
    assert abs(ad - fd) <= 1e-3 * abs(fd) + 1e-3


def test_grad_wrt_wavelength():
    c = CELLS["offdiag"]

    def fom(wl):
        _, R, _, _ = _solve_jax(c, 0.35, wl=wl)
        return jnp.sum(R)

    x0 = jnp.asarray(c["wl"])
    ad = float(jax.grad(fom)(x0))
    fd = float(_fd(lambda x: float(fom(x)), x0, c["wl"] * 1e-5))
    assert np.isfinite(ad)
    assert abs(ad - fd) <= 1e-3 * abs(fd) + 1.0   # large scale (1/wl ~ 1e6)


@pytest.mark.parametrize("angle0", [0.05, 0.2, 0.35])
def test_grad_wrt_angle_oblique(angle0):
    """d sum(R) / d(angle) vs FD at OBLIQUE incidence (clean; the angle grad at
    EXACTLY normal is the symmetry-protected zero -- see the module docstring
    and test_grad_wrt_angle_normal_is_symmetry_zero)."""
    c = CELLS["offdiag"]

    def fom(angle):
        _, R, _, _ = _solve_jax(c, angle0, angle_arr=angle)
        return jnp.sum(R)

    x0 = jnp.asarray(angle0)
    ad = float(jax.grad(fom)(x0))
    fd = float(_fd(lambda x: float(fom(x)), x0, 1e-5))
    assert np.isfinite(ad)
    assert abs(ad - fd) <= 1e-3 * abs(fd) + 1e-6


def test_grad_wrt_angle_normal_is_symmetry_zero():
    """At EXACTLY normal incidence d/d(angle) of sum(R) is the symmetry-
    protected ZERO (sum(R) is even in angle); the FD confirms ~0 and the AD is
    a tiny degenerate-eig artifact -- documented, harmless (the true value is
    zero)."""
    c = CELLS["offdiag"]

    def fom(angle):
        _, R, _, _ = _solve_jax(c, 0.0, angle_arr=angle)
        return jnp.sum(R)

    x0 = jnp.asarray(0.0)
    ad = float(jax.grad(fom)(x0))
    fd = float(_fd(lambda x: float(fom(x)), x0, 1e-5))
    assert np.isfinite(ad)
    assert abs(fd) < 1e-8                # true derivative is the symmetry zero
    assert abs(ad) < 1e-3               # AD artifact is tiny (near the zero)


# ---------------------------------------------------------------------------
# Diagonal-tensor -> scalar grad reduction: a TE-equivalent diagonal in-plane
# tensor's Ey (incident E_y) channel reproduces the scalar TE
# pmm_efficiency_1d grad w.r.t. eps_yy.
# ---------------------------------------------------------------------------

def test_diagonal_tensor_reduces_to_scalar_te_grad():
    period, n_sub, n_sup = 1.0e-6, 1.5, 1.0
    depth, duty, wl, degree, ang = 0.4e-6, 0.55, 0.6e-6, 9, 0.3
    nr_x, ng = 1.7, 1.2

    def jones_sumT_Ey(eyy):
        er = jnp.diag(jnp.asarray([nr_x ** 2, eyy, nr_x ** 2], _CJ))
        eg = jnp.diag(jnp.asarray([ng ** 2, ng ** 2, ng ** 2], _CJ))
        _, _, T, _ = pmm_jones_1d(
            period, er, eg, jnp.asarray(n_sub, _CJ), jnp.asarray(n_sup, _CJ),
            jnp.asarray(depth), duty, jnp.asarray(wl), angle=ang, degree=degree,
            stabilize=False)
        return jnp.sum(T[1])             # incident E_y (row 1) transmitted total

    def scalar_te_sumT(eps_r):
        _, _, T = pmm_efficiency_1d(
            period, jnp.sqrt(eps_r), jnp.asarray(ng, _CJ),
            jnp.asarray(n_sub, _CJ), jnp.asarray(n_sup, _CJ), jnp.asarray(depth),
            duty, jnp.asarray(wl), angle=ang, polarization="te", degree=degree,
            stabilize=False)
        return jnp.sum(T)

    eyy0 = jnp.asarray(nr_x ** 2 * 1.05)
    # forward equality
    assert abs(float(jones_sumT_Ey(eyy0)) - float(scalar_te_sumT(eyy0))) < 1e-9
    gJ = float(jax.grad(jones_sumT_Ey)(eyy0))
    gS = float(jax.grad(scalar_te_sumT)(eyy0))
    assert np.isfinite(gJ)
    assert abs(gJ - gS) <= 1e-6 * abs(gS) + 1e-9


# ---------------------------------------------------------------------------
# GATE-JIT: jax.jit of the grad of a Jones FOM compiles + is finite.
# ---------------------------------------------------------------------------

def test_jit_grad_of_jones_fom():
    c = CELLS["offdiag"]
    exy0 = float(np.real(c["er"][0, 1]))

    def fom(exy_re):
        er = jnp.asarray(c["er"], _CJ).at[0, 1].set(exy_re + 0j) \
            .at[1, 0].set(exy_re + 0j)
        _, _, _, J = _solve_jax(c, 0.35, er=er)
        return jnp.abs(J[0, 1]) ** 2

    g_eager = float(jax.grad(fom)(jnp.asarray(exy0)))
    g_jit = float(jax.jit(jax.grad(fom))(jnp.asarray(exy0)))
    assert np.isfinite(g_jit)
    assert abs(g_jit - g_eager) <= 1e-9 * abs(g_eager) + 1e-12


# ---------------------------------------------------------------------------
# Guard rails: the JAX path raises a precise error outside the supported
# surface (stabilize, out-of-plane tensor), never silently wrong.
# ---------------------------------------------------------------------------

def test_jax_path_requires_stabilize_false():
    c = CELLS["diag"]
    with pytest.raises(ValueError, match="stabilize=False"):
        pmm_jones_1d(
            c["period"], jnp.asarray(c["er"], _CJ), jnp.asarray(c["eg"], _CJ),
            jnp.asarray(c["n_sub"], _CJ), jnp.asarray(c["n_sup"], _CJ),
            jnp.asarray(c["depth"]), c["duty"], jnp.asarray(c["wl"]),
            angle=0.0, degree=c["degree"], stabilize=True)


def test_jax_path_out_of_plane_raises():
    """An out-of-plane tensor (exz / ezx != 0) is NumPy-only on the JAX path
    (the native metric generator is not differentiable yet)."""
    er = DIAG_ER.copy()
    er[0, 2] = 0.3                      # exz != 0 -> out of plane
    er[2, 0] = 0.3
    c = CELLS["diag"]
    with pytest.raises((NotImplementedError, ValueError)):
        pmm_jones_1d(
            c["period"], jnp.asarray(er, _CJ), jnp.asarray(c["eg"], _CJ),
            jnp.asarray(c["n_sub"], _CJ), jnp.asarray(c["n_sup"], _CJ),
            jnp.asarray(c["depth"]), c["duty"], jnp.asarray(c["wl"]),
            angle=0.0, degree=c["degree"], stabilize=False)
