"""2026-08-17 -- the OUT-OF-PLANE 4Nf generator's parity-sign block reduction.

``docs/audits/EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md``.  The generator
``G = [[A, P], [Q, B]]`` has ``A``, ``B`` (and the slant convection) LINEAR in
``Kx, Ky`` while ``P``, ``Q`` are quadratic-or-K-free, so at NORMAL incidence
the signed permutation ``R = diag(I,I,-I,-I) . (I4 (x) J)`` -- order flip TIMES
the E/H sign flip -- anti-commutes with ``G``.  ``G`` is then block-ANTI-
DIAGONAL in ``R``'s eigenbasis and ONE ``2Nf`` eig yields all ``4Nf``
eigenpairs.

Every bar here is derived from the running build (machine epsilon and the
problem size, or the dense solve's own residual on the very same matrix) --
nothing pins a measured number, per docs/TESTING_STANDARDS.md.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

# The out-of-plane fixtures below sit at a modest truncation and trip the
# lossless-closure warning (sum R+T - 1 ~ 8e-03 at degree 7 / n_orders 3,
# 3.4e-05 at degree 9 / n_orders 4).  That is a PRE-EXISTING property of the
# 4Nf path at these truncations, not of the reduction under test: verified
# 2026-08-17 against the 5.38.1 mount, which raises the identical warning on
# the identical cell.  Both arms of every comparison here run the same
# fixture, and nothing in this module asserts unity (the module docstring of
# test_v5_14_0_pmm2d_oop.py records why that would be a trap for the
# non-reciprocal cells).
pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

from lumenairy.elements.pmm import pmm_jones_2d  # noqa: E402
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid  # noqa: E402
from lumenairy.elements.rcwa import _core as C  # noqa: E402
from lumenairy.elements.rcwa import uniaxial_tensor  # noqa: E402

_P = 0.6e-6
_WL = 0.55e-6
_DEP = 0.25e-6
_NORD = 3
_DEG = 7


# --------------------------------------------------------------------------
# fixtures -- built through the public API, never hand-assembled operators
# --------------------------------------------------------------------------
def _tilted_pillar(s=6, theta=0.6, phi=0.4, centred=True):
    """A tilted-uniaxial (LC) pillar in an isotropic ambient -- the common
    metasurface out-of-plane cell.  ``centred=False`` puts the pillar on a
    wall layout that is NOT mirror-symmetric, which is how this test
    ENGINEERS a structure-violating cell rather than hoping for one."""
    tc = np.zeros((s, s, 3, 3), complex)
    tc[:, :] = np.eye(3)
    lc = uniaxial_tensor(1.5, 1.8, theta, phi=phi)
    if centred:
        tc[s // 2 - 1:s // 2 + 1, s // 2 - 1:s // 2 + 1] = lc
    else:
        tc[0:2, 1:3] = lc
    return tc


def _uniform_tilted(s=4, theta=0.6, phi=0.4):
    tc = np.zeros((s, s, 3, 3), complex)
    tc[:, :] = uniaxial_tensor(1.5, 1.8, theta, phi=phi)
    return tc


def _general_eps(s=6):
    """Fully general 3x3: biaxial, NON-reciprocal (e_xz != e_zx), lossy --
    the adversarial case for a structure claimed only for uniaxial."""
    e = np.array([[2.9 + 0.05j, 0.31, 0.44],
                  [-0.17, 2.4, 0.28 + 0.02j],
                  [0.52, -0.36, 2.1]], dtype=complex)
    tc = np.zeros((s, s, 3, 3), complex)
    tc[:, :] = np.eye(3)
    tc[s // 2 - 1:s // 2 + 1, s // 2 - 1:s // 2 + 1] = e
    return tc


def _inplane_cell(s=6):
    tc = np.zeros((s, s, 3, 3), complex)
    tc[:, :] = np.eye(3)
    tc[s // 2 - 1:s // 2 + 1, s // 2 - 1:s // 2 + 1] = (
        np.diag([6.0, 5.0, 4.0])
        + np.array([[0, 0.4, 0], [0.4, 0, 0], [0, 0, 0]]))
    return tc


class _Spy:
    """Counts how many 4N generators took the reduction vs the dense solve,
    and keeps the last generator + gauge for the eigensolve-level tests."""

    def __init__(self):
        self.factored = 0
        self.dense = 0
        self.last = None

    def install(self, mp):
        orig_block = C._generator_block_eig
        orig_modes = C._generator_modes

        def block(G, N, gauge, **kw):
            self.last = (np.array(G), N, gauge)
            out = orig_block(G, N, gauge, **kw)
            if out is None:
                self.dense += 1
            else:
                self.factored += 1
            return out

        def modes(G, Kx, xp, sym_gauge=None):
            if sym_gauge is None:
                self.dense += 1
            return orig_modes(G, Kx, xp, sym_gauge=sym_gauge)

        mp.setattr(C, "_generator_block_eig", block)
        mp.setattr(C, "_generator_modes", modes)
        return self


def _solve(tc, *, symmetry, n_orders=_NORD, **kw):
    return pmm_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL, degree=_DEG,
                        n_orders=n_orders, symmetry=symmetry, **kw)


def _algebra_bar(n_orders):
    """Derived agreement bar for two floating-point algorithms solving the
    SAME 4N eigenproblem and feeding the same cascade: the normwise backward
    error of a dense eig scales as ``c * n * eps``, so ``1e3 * 4Nf * eps``
    carries decades of margin over that.

    GAP, both sides, measured 2026-08-17 on this build (py3.14.6 / numpy
    2.4.4 / scipy-openblas 0.3.31).  BELOW: the two paths agree to 8.4e-15 ..
    2.8e-14 relative across the fixture set, i.e. 3.3 decades under the
    4.4e-11 this returns at n_orders=3.  ABOVE: the smallest WRONG answer the
    mechanism can produce -- the reduction taken with a mis-built involution
    (the order flip replaced by the identity, i.e. the E/H sign flip alone,
    forced past the structural gate) -- lands 3.2e-02 off in the spectrum,
    8.9 decades above the bar.  The bar is derived at runtime from the
    problem size and machine epsilon, so it tracks the library, and it is
    never a reading of one build's residual."""
    return 1e3 * 4 * (2 * n_orders + 1) ** 2 * float(np.finfo(float).eps)


def _agree(a, b, bar, what):
    d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
    assert d <= bar, f"{what}: {d:.3e} > {bar:.3e}"
    return d


# --------------------------------------------------------------------------
# 1.  it engages, and it agrees with the dense generator
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,cell,kw", [
    ("uniform tilted uniaxial", _uniform_tilted(), {}),
    ("patterned tilted uniaxial", _tilted_pillar(), {}),
    ("general biaxial non-reciprocal lossy", _general_eps(), {}),
    ("in-plane slanted tensor", _inplane_cell(), dict(slant=(0.2, 0.1))),
])
def test_block_reduction_engages_and_matches_the_dense_generator(
        name, cell, kw, monkeypatch):
    with monkeypatch.context() as mp:
        spy = _Spy().install(mp)
        fast = _solve(cell, symmetry=True, **kw)
    assert spy.factored >= 1 and spy.dense == 0, (
        f"{name}: the reduction did not engage ({spy.factored} factored / "
        f"{spy.dense} dense) -- the rest of this test would be vacuous")
    dense = _solve(cell, symmetry=False, **kw)
    bar = _algebra_bar(_NORD)
    assert np.array_equal(fast[0], dense[0])
    _agree(fast[1], dense[1], bar, f"{name} R")
    _agree(fast[2], dense[2], bar, f"{name} T")
    _agree(fast[3], dense[3], bar, f"{name} Jones")


# --------------------------------------------------------------------------
# 2.  the gate: engaged when the structure is there, REFUSED when it is not,
#     and the refusal is the dense path bit-for-bit
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,cell,kw", [
    # oblique incidence: the order set is not closed under (m,n)->(-m,-n), so
    # A and B are no longer J-odd -- engineered from a cell that DOES carry
    # the structure at normal incidence (the arm above proves that)
    ("oblique incidence", _tilted_pillar(), dict(theta=0.25, phi=0.4)),
    # a centro-symmetric permittivity whose spectral-element WALL layout is
    # not mirror-symmetric: the structure is a property of the assembled
    # operators, so this must refuse even though eps alone looks symmetric
    ("off-centre pillar, unmirrored walls", _tilted_pillar(centred=False), {}),
])
def test_gate_refuses_and_falls_back_bit_for_bit(name, cell, kw, monkeypatch):
    with monkeypatch.context() as mp:
        spy = _Spy().install(mp)
        got = _solve(cell, symmetry=True, **kw)
    assert spy.factored == 0, f"{name}: the reduction engaged but must not"
    assert spy.dense >= 1, f"{name}: no 4N generator was solved at all"
    ref = _solve(cell, symmetry=False, **kw)
    # a refusal runs the SAME dense code, so this is exact, not a tolerance
    for k, what in ((1, "R"), (2, "T"), (3, "Jones")):
        assert np.array_equal(np.asarray(got[k]), np.asarray(ref[k])), \
            f"{name}: {what} differs after a refusal"


def test_symmetry_false_never_engages_the_reduction(monkeypatch):
    """The documented escape hatch: symmetry=False forces the dense path."""
    with monkeypatch.context() as mp:
        spy = _Spy().install(mp)
        _solve(_tilted_pillar(), symmetry=False)
    assert spy.factored == 0 and spy.dense >= 1


# --------------------------------------------------------------------------
# 3.  the gate's bar has a gap on BOTH sides, measured through the shipped
#     gate itself (a tolerance ladder), not re-derived here
# --------------------------------------------------------------------------
def _engages_at(tol, cell, monkeypatch, **kw):
    with monkeypatch.context() as mp:
        mp.setattr(C, "_OOP_BLOCK_TOL", tol)
        spy = _Spy().install(mp)
        _solve(cell, symmetry=True, **kw)
    return spy.factored >= 1


def test_structural_bar_has_decades_of_gap_on_both_sides(monkeypatch):
    """Walk the shipped gate's own tolerance and show the bar is not sitting
    inside anything's spread.  Measured defects 2026-08-17: 2.6e-15 for the
    carrying cell, 2.5e-02 for the violating one, against a 1e-10 bar."""
    default = C._OOP_BLOCK_TOL
    holds = _tilted_pillar()
    fails = _tilted_pillar(centred=False)
    # BELOW: a carrying cell still passes a bar 100x tighter than shipped
    # (its defect is 2.6e-15, i.e. ~2.6 decades under that 1e-12 -- room for
    # any LAPACK's contribution to an O(n^2) operator-assembly residual)
    assert _engages_at(default * 1e-2, holds, monkeypatch)
    # ABOVE: a violating cell is still refused at a bar 1000x looser, so no
    # plausible re-derivation of the bar admits it
    assert not _engages_at(default * 1e3, fails, monkeypatch)
    # ... and the STRUCTURAL TEST is what refuses it -- open the tolerance all
    # the way and the same cell engages, so the refusal above is this gate and
    # not some other precondition quietly returning None
    assert _engages_at(1.0, fails, monkeypatch)


# --------------------------------------------------------------------------
# 4.  the eigensolve itself: residual bar derived from the DENSE solve's own
#     backward error on the very same matrix
# --------------------------------------------------------------------------
def _residual(G, gam, V):
    R = G @ V - V * gam[None, :]
    return float(np.max(np.linalg.norm(R, axis=0)
                        / (np.linalg.norm(G, 2)
                           * np.linalg.norm(V, axis=0))))


def _match(g1, g2):
    t = np.array(g2, dtype=complex)
    used = np.zeros(t.shape[0], dtype=bool)
    worst = 0.0
    for g in g1:
        dd = np.abs(t - g)
        dd[used] = np.inf
        k = int(np.argmin(dd))
        used[k] = True
        worst = max(worst, float(dd[k]))
    return worst / max(float(np.max(np.abs(g1))), 1e-300)


def test_factored_eigenpairs_meet_the_dense_solves_own_backward_error(
        monkeypatch):
    with monkeypatch.context() as mp:
        spy = _Spy().install(mp)
        _solve(_tilted_pillar(), symmetry=True)
    assert spy.factored >= 1 and spy.last is not None
    G, N, gauge = spy.last
    gam_f, V_f = C._generator_block_eig(G, N, gauge)
    gam_d, V_d = np.linalg.eig(G)
    r_dense = _residual(G, gam_d, V_d)
    # the same shape the eig-recycle experiment used: 1e3 x the dense solve's
    # OWN residual on THIS matrix, re-measured every run
    assert r_dense > 0.0
    assert _residual(G, gam_f, V_f) <= 1e3 * r_dense
    assert _match(gam_f, gam_d) <= 1e3 * r_dense


def test_flux_mode_selector_classifies_identically(monkeypatch):
    """The forward/backward split reads the eigenvectors' z-flux; the factored
    vectors must preserve it (they are unit-2-norm, as zgeev's are, because
    the selector's noise ceilings are RELATIVE to the largest flux)."""
    with monkeypatch.context() as mp:
        spy = _Spy().install(mp)
        _solve(_general_eps(), symmetry=True)
    G, N, gauge = spy.last
    gam_f, V_f = C._generator_block_eig(G, N, gauge)
    gam_d, V_d = np.linalg.eig(G)
    f_f = np.asarray(C._select_forward_flux(gam_f, V_f, N))
    f_d = np.asarray(C._select_forward_flux(gam_d, V_d, N))
    assert f_f.shape[0] == 2 * N and f_d.shape[0] == 2 * N
    assert _match(np.sort_complex(gam_f[f_f]), np.sort_complex(gam_d[f_d])) \
        <= 1e3 * _residual(G, gam_d, V_d)


# --------------------------------------------------------------------------
# 5.  the stack path (cascade + the priced eig cache)
# --------------------------------------------------------------------------
def _stack(symmetry, n_orders=_NORD):
    st = PMM2DStackHybrid(_P, _P, n_substrate=1.5, degree=_DEG,
                          n_orders=n_orders, symmetry=symmetry)
    st.add_layer(0.20e-6, eps_tensor_cell=_tilted_pillar(theta=0.6))
    st.add_layer(0.15e-6, eps_tensor_cell=_tilted_pillar(theta=0.9))
    st.add_layer(0.10e-6, eps=2.25)
    st.add_layer(0.20e-6, eps_tensor_cell=_tilted_pillar(theta=0.6))
    st.set_source(_WL, theta=0.0, phi=0.0)
    return st


def test_stack_cascade_agrees_and_the_cache_is_unaffected(monkeypatch):
    with monkeypatch.context() as mp:
        spy = _Spy().install(mp)
        st = _stack(True)
        fast = st.solve()
        again = st.solve()                      # served from the eig cache
    assert spy.factored >= 1 and spy.dense == 0
    for k in (1, 2, 3):
        assert np.array_equal(np.asarray(fast[k]), np.asarray(again[k])), \
            "the cached re-solve must reproduce its own result exactly"
    dense = _stack(False).solve()
    bar = _algebra_bar(_NORD)
    for k, what in ((1, "R"), (2, "T"), (3, "Jones")):
        _agree(fast[k], dense[k], bar, f"stack {what}")
