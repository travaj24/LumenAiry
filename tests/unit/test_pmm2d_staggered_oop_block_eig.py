"""The normal-incidence PARITY-sign block reduction for the PURE staggered
OUT-OF-PLANE region solve: ONE ``2 q^2`` eig instead of the ``4 q^2`` one.

Structure, derivation and every measured number:
``docs/audits/BUILD_PMM2D_STAGGERED_OOP_BLOCK_EIG_2026_09_10.md``.  The Fourier
twin of the same reduction (and the verify-then-use discipline copied here) is
``docs/audits/EXPERIMENT_PMM2D_OOP_BLOCK_EIG_2026_08_17.md`` /
``rcwa._core._generator_block_eig``.

EVERY bar below is DERIVED from a measurement made during this build on
2026-09-09/10 (Windows 11, py3.14.6, numpy 2.4.4, scipy 1.17.1, OMP=1), stated
in the assertion's comment with the build-doc table it comes from.  Nothing
pins a cross-build value: every comparison is TWO-ARM and same-build, and each
bar has decades of gap on both sides.

The reduction is an ACCELERATOR, so the shape of the suite is:

* engagement is asserted as a DECISION before any agreement is checked, so no
  arm can pass vacuously by quietly refusing;
* the gate is asserted TWO-SIDED -- engaged on cells that carry the structure,
  refused on ENGINEERED violations, and a refusal is asserted BIT-IDENTICAL to
  ``symmetry=False`` (exact, not a tolerance: a refusal literally runs the same
  code);
* the bar's own gap is measured THROUGH the shipped gate by walking a
  tolerance ladder, and the FAIL-BEFORE forces the reduction onto a cell that
  does not carry the structure to show the runtime verification is
  load-bearing.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.linalg as sla  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as TS  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    Granet2DTransverseE,
    _region_modes_oop,
    _stag_block_eig,
    _stag_parity_1d,
    _stag_parity_gauge,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import uniaxial_tensor  # noqa: E402

# --------------------------------------------------------------------------- #
# fixtures -- grids <= (3,3), M <= 8 (the plan's test-cost rule)
# --------------------------------------------------------------------------- #
PX = PY = 1.10e-6
WL = 0.68e-6
DEP = 0.34e-6
NSUB, NSUP = 1.50, 1.0
K0 = 2.0 * np.pi / WL

#: a tilted-director liquid crystal: the device case for the whole path
TIL = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0), phi=np.deg2rad(25.0))
#: NON-reciprocal (e13 = conj(e31), Hermitian, still lossless) -- the reduction
#: must not be a symmetric-tensor trick
NREC = TIL.copy()
NREC[0, 2] = TIL[0, 2] + 0.30j
NREC[2, 0] = TIL[2, 0] - 0.30j
LOSSY = uniaxial_tensor(1.5 + 0.06j, 1.7 + 0.03j, np.deg2rad(35.0),
                        phi=np.deg2rad(25.0))
AIR = np.eye(3, dtype=complex)
ISO = 2.25 * np.eye(3, dtype=complex)


def _tile(t, n):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[:, :] = t
    return c


def centro(t, n):
    """A cell that IS its own parity image: ``eps[i, j] == eps[n-1-i, n-1-j]``
    in every tensor component, on the mirror-symmetric uniform wall layout."""
    c = _tile(AIR, n)
    if n == 2:
        c[0, 0] = c[1, 1] = t
    else:
        c[1, 1] = t                       # the parity-FIXED centre pixel
        c[0, 0] = c[2, 2] = ISO           # a mirror PAIR
    return c


def offcentre(t, n):
    """Its own parity image nowhere: one corner pixel."""
    c = _tile(AIR, n)
    c[0, 0] = t
    return c


def broken(n):
    """Parity-symmetric PATTERN, parity-BREAKING tensor grid -- the two mirror
    pixels carry DIFFERENT out-of-plane tensors, so ``eps`` alone would look
    'symmetric' to a shape test while the operators are not."""
    c = _tile(AIR, n)
    if n == 2:
        c[0, 0] = TIL
        c[1, 1] = uniaxial_tensor(1.5, 1.7, np.deg2rad(35.0),
                                  phi=np.deg2rad(-70.0))
    else:
        c[1, 1] = c[0, 0] = TIL
        c[2, 2] = uniaxial_tensor(1.58, 1.82, 1.11, phi=2.05)
    return c


def solver(cell, M, theta=0.0, phi=0.0):
    """``Granet2DTransverseE`` built exactly as ``PMM2DStackPure.solve``
    builds it (physical periods, ``k0 = 2 pi / wl``)."""
    nre = float(np.real(NSUP))
    a0x = nre * np.sin(theta) * np.cos(phi) * K0
    a0y = nre * np.sin(theta) * np.sin(phi) * K0
    n = cell.shape[0]
    return Granet2DTransverseE(PX, PY, n, n, M, cell, alpha0x=a0x,
                               alpha0y=a0y, k0=K0)


def struct_dA(sol):
    """``max|R A R + A| / max|A|`` -- exactly what the shipped gate tests."""
    g = _stag_parity_gauge(sol)
    if g is None:
        return None
    perm, r = g
    A = sol.Agen
    rr = r[:, None] * r[None, :]
    return (float(np.max(np.abs(rr * A[np.ix_(perm, perm)] + A)))
            / float(np.max(np.abs(A))))


def jones_call(cell, M, sym, theta=0.0, phi=0.0, n_orders=5):
    return pmm_jones_2d_staggered(PX, PY, cell, NSUB, NSUP, DEP, WL,
                                  degree=M, n_orders=n_orders, theta=theta,
                                  phi=phi, symmetry=sym)


def rt_hash(res):
    h = hashlib.sha256()
    for a in res:
        h.update(np.ascontiguousarray(np.asarray(a)).tobytes())
    return h.hexdigest()


def dmax(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def hausdorff(a, b):
    """Symmetric set distance -- a plain sort would pair a near-degenerate
    doublet wrongly and manufacture a mismatch out of nothing."""
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    d = np.abs(a[:, None] - b[None, :])
    return max(float(np.max(np.min(d, axis=1))),
               float(np.max(np.min(d, axis=0))))


def backward_error(A, B, qv, X):
    """Normwise backward error of the pencil eigenpairs,
    ``|A x - q B x| / ((|A| + |q||B|) |x|)``."""
    na, nb = float(np.linalg.norm(A, 2)), float(np.linalg.norm(B, 2))
    res = A @ X - (B @ X) * qv[None, :]
    den = (na + np.abs(qv) * nb) * np.linalg.norm(X, axis=0)
    return float(np.max(np.linalg.norm(res, axis=0) / den))


def dense_eig(sol):
    """The dense arm's own ``(q, X)`` on the very same pencil."""
    A, B = sol.Agen, sol.Bgen
    Lc = np.linalg.cholesky(B)
    Ah = sla.solve_triangular(Lc, A, lower=True)
    Ah = sla.solve_triangular(Lc, Ah.conj().T, lower=True).conj().T
    qd, Y = np.linalg.eig(Ah)
    return qd, sla.solve_triangular(Lc.conj().T, Y, lower=False)


# --------------------------------------------------------------------------- #
# 1. the parity operator itself (build doc table B0)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Nx", [2, 3, 4])
@pytest.mark.parametrize("M", [5, 6, 8])
def test_parity_map_is_an_exact_involution_and_maps_the_eps_free_operators(
        Nx, M):
    """``J^2 = I`` EXACTLY, the eps-free MASSES are parity-INVARIANT and the
    transverse-derivative bracket is parity-ODD -- the three facts the
    involution ``R A R = -A`` is assembled from.

    The two mass/derivative bars are DERIVED, not assumed: the operators are
    Gauss-Legendre quadratures of polynomials, so their parity image differs
    from themselves only by the summation order.  MEASURED here in the same
    metric (relative to the operator's own max entry), 12 (Nx, M) combinations
    on 2026-09-09: masses 8.0e-17 .. 1.8e-16, derivative 6.6e-17 .. 1.4e-16
    (build doc table B0).  The 1e-12 bar is 4 decades above that envelope and
    ~10 decades below the 6.2e-02 the assembled generator reads when the
    parity genuinely fails (table B2) -- the quantity this feeds.
    """
    b = solver(_tile(TIL, Nx), M).bx
    pt, st, pb, sb = _stag_parity_1d(b)
    Pt = np.zeros((pt.size, pt.size))
    Pt[pt, np.arange(pt.size)] = st
    Pb = np.zeros((pb.size, pb.size))
    Pb[pb, np.arange(pb.size)] = sb
    # a permutation composed with itself, signs squared -> EXACT, not a bar
    assert np.array_equal(Pt @ Pt, np.eye(pt.size))
    assert np.array_equal(Pb @ Pb, np.eye(pb.size))
    Mtt = b.mass(b.Btilde, b.Btilde)
    Mbb = b.mass(b.B, b.B)
    Cbt = b.mixed(b.B, b.Btilde)
    assert (np.max(np.abs(Pt.T @ Mtt @ Pt - Mtt))
            <= 1e-12 * np.max(np.abs(Mtt)))
    assert (np.max(np.abs(Pb.T @ Mbb @ Pb - Mbb))
            <= 1e-12 * np.max(np.abs(Mbb)))
    # ODD: d/dx -> -d/dx under x -> d - x.  The +Cbt (not -Cbt) is the claim.
    assert (np.max(np.abs(Pb.T @ Cbt @ Pt + Cbt))
            <= 1e-12 * np.max(np.abs(Cbt)))
    # and the WRONG sign is not accidentally satisfied (two-sided)
    assert (np.max(np.abs(Pb.T @ Cbt @ Pt - Cbt))
            > 1e-3 * np.max(np.abs(Cbt)))


def test_parity_gauge_is_refused_at_oblique_and_conical_incidence():
    """The Bloch glue ``tau != 1`` breaks the hat permutation, so the gauge is
    a free necessary-condition gate BEFORE the pencil is ever touched."""
    cell = centro(TIL, 2)
    assert _stag_parity_gauge(solver(cell, 6)) is not None
    assert _stag_parity_gauge(solver(cell, 6, theta=np.deg2rad(25.0))) is None
    assert _stag_parity_gauge(
        solver(cell, 6, theta=np.deg2rad(25.0), phi=np.deg2rad(40.0))) is None


# --------------------------------------------------------------------------- #
# 2. engagement, sectors, agreement (build doc tables B1, B2)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("Nx,name", [(2, "centro pillar"),
                                     (3, "centro pillar"),
                                     (2, "uniform"), (3, "uniform")])
def test_reduction_engages_and_splits_into_two_equal_sectors(Nx, name):
    """ENGAGEMENT AS A DECISION, asserted before any agreement claim, plus the
    two structural facts that make the reduction exist: the involution's two
    eigenspaces are each exactly ``2 q^2``-dimensional, and the assembled
    residual is at roundoff."""
    cell = _tile(TIL, Nx) if name == "uniform" else centro(TIL, Nx)
    sol = solver(cell, 6)
    g = _stag_parity_gauge(sol)
    assert g is not None
    perm, r = g
    qq = sol.q * sol.q
    # tr R = 0 exactly -> dim(+) == dim(-) == 2 q^2 (the sectors the reduction
    # projects onto); computed from the signed permutation, not from an eig
    fixed = perm == np.arange(4 * qq)
    assert float(np.sum(r[fixed])) == 0.0
    pairs = int(np.sum(~fixed)) // 2          # each 2-cycle gives one of each
    assert pairs + int(np.sum(fixed & (r > 0))) == 2 * qq
    assert pairs + int(np.sum(fixed & (r < 0))) == 2 * qq
    # engagement, and the structure residual it is decided on
    assert _stag_block_eig(sol.Agen, sol.Bgen, qq, g) is not None
    # MEASURED 2026-09-09 (build doc table B2): carrying cells 5.8e-16..1.5e-14
    assert struct_dA(sol) <= 1e-10


@pytest.mark.parametrize("Nx", [2, 3])
@pytest.mark.parametrize("tensor,label", [(TIL, "tilted uniaxial"),
                                          (NREC, "non-reciprocal"),
                                          (LOSSY, "lossy")])
def test_reduction_matches_the_dense_path_on_R_T_and_jones(Nx, tensor, label):
    """The whole point: same answer, cheaper.  ON vs OFF through the SHIPPED
    entry point, both incident polarizations, per order.

    Bar DERIVED from this build's own measurement over 24 (grid, M, tensor)
    combinations on 2026-09-09 (build doc table B1): dR <= 4.9e-15,
    dT <= 1.6e-13, dJones <= 2.6e-14.  ``1e-11`` is ~2 decades above the worst
    of those and ~9 decades below the 1.5e-03 .. 2.2e-01 a FORCED reduction on
    a cell that does not carry the structure produces (table B5b, and the
    fail-before below) -- decades of gap on both sides.
    """
    cell = centro(tensor, Nx)
    # engagement first, so the agreement claim cannot pass by refusing
    sol = solver(cell, 6)
    assert _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q,
                           _stag_parity_gauge(sol)) is not None
    on = jones_call(cell, 6, True)
    off = jones_call(cell, 6, False)
    assert np.array_equal(np.asarray(on[0]), np.asarray(off[0]))
    assert dmax(on[1], off[1]) <= 1e-11
    assert dmax(on[2], off[2]) <= 1e-11
    assert dmax(on[3], off[3]) <= 1e-11


@pytest.mark.parametrize("Nx", [2, 3])
def test_reduction_reproduces_the_dense_eigenvalue_set_and_flux_split(Nx):
    """The eigenvalue SET (not the order -- the generalized cascade is
    invariant to the modal permutation) and the forward/backward split.

    The set bar is DERIVED: MEASURED 2026-09-09 over the same 24 combinations,
    ``d(eig set) <= 6.9e-12`` (build doc table B1), while FORCING the
    reduction on a structure-violating cell moves the set by 5.1e-02 .. 4.3e-01
    (table B5).  ``1e-9`` sits ~2 decades above the measured envelope and ~7
    below the smallest real violation.  The split is an exact count.
    """
    cell = centro(NREC, Nx)
    sol = solver(cell, 6)
    qq = sol.q * sol.q
    on = _region_modes_oop(sol, symmetry=True)
    off = _region_modes_oop(sol, symmetry=False)
    assert on[2].size == off[2].size == 2 * qq        # forward count, exact
    assert on[5].size == off[5].size == 2 * qq        # backward count, exact
    assert hausdorff(np.concatenate([on[2], on[5]]),
                     np.concatenate([off[2], off[5]])) <= 1e-9


@pytest.mark.parametrize("Nx", [2, 3])
def test_factored_eigenpair_residual_is_at_the_dense_solve_own_scale(Nx):
    """The accuracy claim measured where it lives -- the backward error of the
    eigenpairs on the very same pencil, against the DENSE solve's own backward
    error on that pencil (nothing pinned from elsewhere).

    MEASURED 2026-09-09 over 18 combinations (build doc table B6): dense
    1.1e-15 .. 5.3e-15, factored 4.5e-16 .. 2.1e-14, ratio 0.28 .. 7.20.  The
    reduction squares the spectrum, so unlike the hybrid's Fourier twin it is
    NOT uniformly better than dense -- it is within one decade, which is what
    is asserted (1e2 x, ~1.1 decades of headroom above the worst measured
    ratio).
    """
    sol = solver(centro(TIL, Nx), 6)
    g = _stag_parity_gauge(sol)
    fac = _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q, g)
    assert fac is not None
    qd, Xd = dense_eig(sol)
    bd = backward_error(sol.Agen, sol.Bgen, qd, Xd)
    bf = backward_error(sol.Agen, sol.Bgen, fac[0], fac[1])
    assert bf <= 1e2 * bd
    # and the dense arm is itself at roundoff, so the comparison means
    # something: 4 q^2 * eps * 1e3 is the conventional envelope
    assert bd <= 1e3 * 4 * sol.q * sol.q * np.finfo(float).eps


# --------------------------------------------------------------------------- #
# 3. the gate is TWO-SIDED, and a refusal is BIT-IDENTICAL
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,cellf,Nx,theta,phi", [
    ("off-centre pillar", lambda: offcentre(TIL, 2), 2, 0.0, 0.0),
    ("off-centre pillar", lambda: offcentre(TIL, 3), 3, 0.0, 0.0),
    ("parity-breaking tensor", lambda: broken(2), 2, 0.0, 0.0),
    ("parity-breaking tensor", lambda: broken(3), 3, 0.0, 0.0),
    ("oblique 25", lambda: centro(TIL, 2), 2, np.deg2rad(25.0), 0.0),
    ("conical 25/40", lambda: centro(TIL, 3), 3, np.deg2rad(25.0),
     np.deg2rad(40.0)),
])
def test_a_refusal_is_bit_identical_to_symmetry_false(label, cellf, Nx, theta,
                                                      phi):
    """A refusal literally runs the dense branch, so this is EXACT (sha256),
    not a tolerance.  Both halves are asserted: the gate REFUSES (a decision)
    and the bytes match."""
    cell = cellf()
    sol = solver(cell, 6, theta, phi)
    g = _stag_parity_gauge(sol)
    refused = (g is None
               or _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q, g) is None)
    assert refused, f"{label} was expected to be refused by the gate"
    on = jones_call(cell, 6, True, theta, phi)
    off = jones_call(cell, 6, False, theta, phi)
    assert rt_hash(on[1:]) == rt_hash(off[1:])


def test_structure_bar_has_a_gap_on_both_sides_measured_through_the_gate():
    """The bar's own two-sided gap, walked THROUGH the shipped gate (``tol`` is
    read at call time precisely so a test can do this) rather than re-derived
    in the test:

    * a CARRYING cell still engages at ``default / 1e2``;
    * a VIOLATING cell is still refused at ``default * 1e3``;
    * that same violating cell DOES engage once ``tol`` is opened to 1.0 --
      which proves the refusal is THIS structural test and not some other
      precondition quietly returning ``None``.
    """
    tol = TS._STAG_BLOCK_TOL
    ok = solver(centro(TIL, 3), 6)
    bad = solver(broken(3), 6)
    gok, gbad = _stag_parity_gauge(ok), _stag_parity_gauge(bad)
    assert gok is not None and gbad is not None
    n_ok, n_bad = ok.q * ok.q, bad.q * bad.q
    assert _stag_block_eig(ok.Agen, ok.Bgen, n_ok, gok, tol=tol / 1e2) is not None
    assert _stag_block_eig(bad.Agen, bad.Bgen, n_bad, gbad,
                           tol=tol * 1e3) is None
    assert _stag_block_eig(bad.Agen, bad.Bgen, n_bad, gbad,
                           tol=1.0) is not None
    # and the two residuals themselves straddle the bar by decades
    assert struct_dA(ok) < tol < struct_dA(bad)


def near_miss(d):
    """A (2, 2) cell whose two mirror pixels differ ONLY by a relative ``d`` on
    the out-of-plane entries, so the structure residual can be walked across
    ``_STAG_BLOCK_TOL`` instead of jumped over by eight decades."""
    t2 = TIL.copy()
    t2[0, 2] = TIL[0, 2] * (1.0 + d)
    t2[2, 0] = TIL[2, 0] * (1.0 + d)
    c = _tile(AIR, 2)
    c[0, 0] = TIL
    c[1, 1] = t2
    return c


def test_the_bar_is_walked_across_and_everything_it_accepts_is_still_exact():
    """The gate's bar tested where it actually sits, not eight decades away.

    ``test_structure_bar_has_a_gap_on_both_sides...`` above moves the BAR past
    fixtures whose residual is 6e-02 .. 7e-01.  This test instead moves the
    FIXTURE across the bar: a family of cells whose two mirror pixels differ by
    a relative ``d`` walks ``dA`` from ~1e-13 to ~1e-07, i.e. through
    ``_STAG_BLOCK_TOL = 1e-10``, and every quantity below is measured at
    runtime -- nothing is pinned.

    Three claims, all two-sided:

    * the family really does SPAN the bar (asserted, not hoped: at least one
      rung a factor 2 under it and one a factor 2 over it);
    * the gate's decision follows its own stated predicate on both sides of
      that guard band;
    * and -- the claim that makes the bar SAFE rather than merely tidy --
      every cell the gate ACCEPTS still reproduces the dense path to the same
      1e-11 the exact-symmetry tests assert.  MEASURED on this build
      2026-09-10: the largest accepted residual is dA = 5.1e-11 and the
      observable error there is 1.6e-15, because the reduction's error is
      QUADRATIC in ``dA`` (fitted 0.56 * dA^2 over the sweep), so the accepted
      error at the bar itself is ~1e-20 -- four decades under the 1e-11 bar at
      the worst rung measured, and the guard band's factor 2 is ~5 decades
      above ``dA``'s own build spread (a deterministic assembly quantity whose
      roundoff floor is ~5e-15 relative).
    """
    tol = TS._STAG_BLOCK_TOL
    rungs = []
    for d in (1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6):
        cell = near_miss(d)
        sol = solver(cell, 5)
        g = _stag_parity_gauge(sol)
        assert g is not None, "normal incidence: the gauge must be available"
        dA = struct_dA(sol)
        accepted = _stag_block_eig(sol.Agen, sol.Bgen, sol.q * sol.q,
                                   g) is not None
        rungs.append((d, cell, dA, accepted))
    # the ladder must actually straddle the bar, with a guard band
    assert any(dA <= tol / 2 for _d, _c, dA, _a in rungs), rungs
    assert any(dA >= 2 * tol for _d, _c, dA, _a in rungs), rungs
    # the gate follows its own predicate outside the guard band
    for d, cell, dA, accepted in rungs:
        if dA <= tol / 2:
            assert accepted, (d, dA)
        elif dA >= 2 * tol:
            assert not accepted, (d, dA)
    # and everything it ACCEPTS is still exact
    acc = [(d, cell, dA) for d, cell, dA, a in rungs if a]
    assert acc, rungs
    for d, cell, dA in acc:
        on = jones_call(cell, 5, True)
        off = jones_call(cell, 5, False)
        assert max(dmax(on[1], off[1]), dmax(on[2], off[2]),
                   dmax(on[3], off[3])) <= 1e-11, (d, dA)
    # and everything it REFUSES runs the dense branch bit-for-bit
    ref = [(d, cell) for d, cell, _dA, a in rungs if not a]
    assert ref, rungs
    d, cell = ref[0]
    assert rt_hash(jones_call(cell, 5, True)[1:])         == rt_hash(jones_call(cell, 5, False)[1:])


@pytest.mark.parametrize("label,cellf,Nx", [
    ("off-centre pillar", lambda: offcentre(TIL, 2), 2),
    ("parity-breaking tensor", lambda: broken(3), 3),
])
def test_forcing_the_reduction_without_the_structure_is_wrong_by_decades(
        label, cellf, Nx):
    """FAIL-BEFORE.  Disarm the runtime verification (``_STAG_BLOCK_TOL`` to
    1.0, the one knob that decides) and run the SHIPPED entry point on a cell
    that does not carry the structure: the answer moves by decades, which is
    what makes the verification load-bearing rather than decorative.

    MEASURED 2026-09-09 (build doc table B5b): dR 1.5e-03 .. 9.4e-03,
    dT 7.4e-03 .. 2.2e-01, dJones 7.4e-03 .. 6.3e-02, against the 1e-11
    agreement the same comparison shows when the structure IS there.  The
    ``1e-4`` bar below sits ~1.2 decades under the smallest measured error and
    7 decades above the accepted agreement.
    """
    cell = cellf()
    ref = jones_call(cell, 6, False)
    saved = TS._STAG_BLOCK_TOL
    try:
        # np.inf, not 1.0: the structure residual dA is bounded by 2, so a
        # 1.0 disarm still REFUSES some cells (an off-centre metal pillar reads
        # dA = 1.033 -- verify D4, 2026-09-10) and the fail-before would then
        # fail loudly for the wrong reason instead of forcing the reduction.
        TS._STAG_BLOCK_TOL = np.inf
        forced = jones_call(cell, 6, True)
    finally:
        TS._STAG_BLOCK_TOL = saved
    assert TS._STAG_BLOCK_TOL == saved
    assert max(dmax(forced[1], ref[1]), dmax(forced[2], ref[2]),
               dmax(forced[3], ref[3])) > 1e-4
    # the shipped tolerance refuses exactly this cell, so the shipped answer is
    # the reference one (the other half of the two-sided claim)
    assert rt_hash(jones_call(cell, 6, True)[1:]) == rt_hash(ref[1:])


# --------------------------------------------------------------------------- #
# 4. nothing else moves
# --------------------------------------------------------------------------- #
def test_scalar_and_in_plane_cells_are_untouched_by_the_keyword():
    """The reduction lives in the OUT-OF-PLANE region solve only, so a scalar
    cell and an in-plane tensor cell must be BIT-IDENTICAL under both settings
    -- the 'the default change moves nothing else' claim, asserted rather than
    assumed."""
    scal = np.array([[2.25, 1.0], [1.0, 2.25]], dtype=complex)
    inpl = _tile(uniaxial_tensor(1.5, 1.7, np.pi / 2, phi=0.4), 2)
    inpl[0, 0] = AIR
    inpl[1, 1] = AIR
    for cell in (scal, inpl):
        a = jones_call(cell, 6, True)
        b = jones_call(cell, 6, False)
        assert rt_hash(a[1:]) == rt_hash(b[1:])


def test_symmetry_keyword_is_validated_and_recorded():
    stack = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                           n_modes=5, n_orders=3)
    assert stack.symmetry is True                     # 'auto' default
    assert PMM2DStackPure(PX, PY, symmetry=False).symmetry is False
    assert PMM2DStackPure(PX, PY, symmetry=True).symmetry is True
    with pytest.raises(ValueError, match="symmetry must be"):
        PMM2DStackPure(PX, PY, symmetry="bogus")
    with pytest.raises(ValueError, match="symmetry must be"):
        pmm_jones_2d_staggered(PX, PY, centro(TIL, 2), NSUB, NSUP, DEP, WL,
                               degree=5, n_orders=3, symmetry="bogus")


def test_multilayer_out_of_plane_stack_agrees_and_engages_per_layer():
    """The cascade, not just one region: three DISTINCT out-of-plane layers (so
    the eig cache cannot collapse them), ON vs OFF.  Same bar as the
    single-layer agreement test, and the engagement is asserted first."""
    cells = []
    for i in range(3):
        c = centro(TIL, 2).copy()
        c[0, 0] = c[0, 0] * (1.0 + 0.02 * i)
        c[1, 1] = c[1, 1] * (1.0 + 0.02 * i)
        cells.append(c)
    for c in cells:
        s = solver(c, 5)
        assert _stag_block_eig(s.Agen, s.Bgen, s.q * s.q,
                               _stag_parity_gauge(s)) is not None
    out = []
    for sym in (True, False):
        st = PMM2DStackPure(PX, PY, n_superstrate=NSUP, n_substrate=NSUB,
                            n_modes=5, n_orders=5, symmetry=sym)
        for c in cells:
            st.add_layer(DEP / 3, eps_cell=c)
        st.set_source(WL)
        out.append(st.solve(jones=True))
    assert dmax(out[0][1], out[1][1]) <= 1e-11
    assert dmax(out[0][2], out[1][2]) <= 1e-11
    assert dmax(out[0][3], out[1][3]) <= 1e-11
