"""P2T -- the 2-D PMM cascade restructure (byte-identical) and the opt-in
fused / tree-parallel folds.

Build doc: ``docs/audits/BUILD_PMM2D_TREE_CASCADE_2026_08_17.md``.

What is under test
------------------
Three levers, pinned SEPARATELY because they have three different identity
contracts:

1. **The byte-identical restructure** (default, every cascade mode).
   ``_redheffer_star`` gained explicit zero-block branches that drop the ten
   zgemms it used to run against literal identity / zero blocks, plus common
   subexpression elimination that leaves the ASSOCIATION ORDER untouched;
   ``_interface_smatrix`` / ``_interface_smatrix_general`` hoist the ``S22``
   product they already computed twice.  Every one of these is asserted BYTE
   IDENTICAL here against a reference written out inline in this file -- the
   pre-P2T algebra, so the claim needs no other worktree to be true.

2. **``cascade='fused'``** (opt-in) swaps the star against a layer-propagation
   S-matrix for the diagonal-aware row/column scaling the rest of the library
   already uses.  Mathematically identical, NOT bit-for-bit (the zgemm path
   rounds its complex products with FMA and the scaling does not), so it is
   bounded by a bar DERIVED from the running build's own conditioning.

3. **``cascade='tree'``** (opt-in) reassociates the whole fold as a balanced
   binary tree so ``max_workers`` can parallelise the cascade itself.  Same
   derived bar; plus unconditional physics oracles (per-polarization energy
   closure, reciprocity, the thick-lossy overflow regime) that hold whatever
   the association order.

Every bar below is DERIVED at runtime from the running build's own measured
quantities (TESTING_STANDARDS rules 2 and 5): the maximum intermediate
S-matrix norm ``G`` and the maximum star-denominator amplification ``kappa``
-- the M1 census's dominant conditioning site -- both read off THE FOLD UNDER
TEST, because a tree's interior sub-chain products are measurably
worse-conditioned than the sequential fold's superstrate-anchored prefixes.
No test here pins a number this campaign happened to read; the shipped P2C
merge bar's shape (``1e3 (2n+2) r``) was scored against this envelope and
MISSES it by 1.9e+03, which is exactly why it is not reused.
"""
import numpy as np
import pytest

import lumenairy.elements.pmm.stack2d as _s2d
from lumenairy.elements.pmm._core import (
    _guarded_inverse,
    _interface_smatrix,
    _propagation_smatrix,
    _redheffer_star,
)
from lumenairy.elements.pmm.stack2d import (
    PMM2DStackHybrid,
    _tree_peak_extra_bytes,
)
from lumenairy.elements.rcwa._core import (
    _interface_smatrix_general,
    _modes_to_M,
)

WL = 1.55e-6
PX = 0.9e-6
_C = np.complex128


# ===================================================================== #
# helpers
# ===================================================================== #
def _cell(hi, lo=2.1, s=6, r=2):
    c = np.full((s, s), complex(lo))
    c[r:s - r, r:s - r] = complex(hi)
    return c


def _tensor_cell(hi, lo=2.1, s=6, r=2, oop=False):
    t = np.zeros((s, s, 3, 3), dtype=_C)
    for ix in range(s):
        for iy in range(s):
            v = hi if (r <= ix < s - r and r <= iy < s - r) else lo
            t[ix, iy] = np.diag([v, v * 1.06, v * 0.94]).astype(_C)
    if oop:
        t[r:s - r, r:s - r, 0, 2] = 0.35
        t[r:s - r, r:s - r, 2, 0] = 0.35
    return t


def _stack(layers, *, theta=0.0, phi=0.0, degree=7, n_orders=4,
           cascade="fast", **kw):
    st = PMM2DStackHybrid(PX, n_superstrate=1.0, n_substrate=1.45,
                          degree=degree, n_orders=n_orders, symmetry=False,
                          cascade=cascade, **kw)
    for t, cell in layers:
        if np.isscalar(cell):
            st.add_layer(t, eps=cell)
        elif np.asarray(cell).ndim == 4:
            st.add_layer(t, eps_tensor_cell=cell)
        else:
            st.add_layer(t, eps_cell=cell)
    st.set_source(WL, theta=theta, phi=phi)
    return st


def _maxdiff(a, b):
    return max(float(np.max(np.abs(np.asarray(x) - np.asarray(y))))
               for x, y in zip(a[1:], b[1:]))


def _bytes_equal(a, b):
    """Byte identity, not closeness: compares the float64 VIEW, so a
    ``-0.0`` / ``+0.0`` flip fails here even though ``==`` would pass."""
    return all(np.array_equal(np.asarray(x).view(np.float64),
                              np.asarray(y).view(np.float64))
               for x, y in zip(a, b))


def _modes_of(st):
    """``(modes, mkeys, k0, Wsup, Vsup)`` for ``st`` at its own source."""
    from lumenairy.elements.pmm.twod import _homogeneous_modes
    src = st._src
    wl = float(src["wavelength"])
    k0 = 2.0 * np.pi / wl
    ox = np.arange(-st.n_orders, st.n_orders + 1)
    order_x = np.tile(ox, len(ox))
    order_y = np.repeat(ox, len(ox))
    nre = float(np.real(np.sqrt(np.conj(complex(st.n_sup) ** 2))))
    kx0 = nre * np.sin(src["theta"]) * np.cos(src["phi"])
    ky0 = nre * np.sin(src["theta"]) * np.sin(src["phi"])
    kxv = kx0 + order_x * (wl / st.period_x)
    kyv = ky0 + order_y * (wl / st.period_y)
    modes, mkeys = st._layer_mode_sets(kxv, kyv, ox, ox, kx0, ky0, k0, wl)
    Wsup, Vsup, _l, _kz = _homogeneous_modes(
        kxv, kyv, np.conj(_C(st.n_sup) ** 2))
    return modes, mkeys, k0, Wsup, Vsup


def _fold_conditioning(st, **solve_kw):
    """``(G, kappa)`` measured over the fold ``st`` ACTUALLY runs.

    * ``kappa`` -- max ``||inv(I - B11 A22)||_inf`` over the fold's stars.
      The star denominators are the cascade's dominant conditioning site (the
      M1 census: they reached ``cond`` 2.4e31 where the interface behind them
      read 3.1e16, and they -- not the interface -- separated every solve that
      broke closure from every clean one).
    * ``G`` -- max ``||.||_inf`` over every intermediate S-matrix block.

    Measured on the fold in force, which is the point: Redheffer stability is
    a PREFIX property (every sequential partial product is anchored to the
    superstrate) and a TREE forms interior sub-chain products that are
    anchored to nothing.  Measured 2026-08-17 on a 40-layer graded scalar
    taper at conical incidence: the sequential fold ran at ``kappa`` 7.1 /
    ``G`` 4.3e+01, the tree at ``kappa`` 4.4e+02 / ``G`` 1.4e+04.  A bar built
    on the SEQUENTIAL fold's conditioning would therefore be ~300x too tight
    for the tree; this reads whichever fold is under test.
    """
    peak = {"kap": 1.0, "G": 1.0}
    orig = _s2d._redheffer_star

    def _spy(SA, SB):
        A22, B11 = np.asarray(SA[3]), np.asarray(SB[0])
        if bool(A22.any()) and bool(B11.any()):
            n = A22.shape[0]
            M = np.eye(n, dtype=_C) - B11 @ A22
            try:
                peak["kap"] = max(peak["kap"],
                                  float(np.linalg.norm(np.linalg.inv(M),
                                                       np.inf)))
            except np.linalg.LinAlgError:                # pragma: no cover
                peak["kap"] = np.inf
        out = orig(SA, SB)
        peak["G"] = max(peak["G"],
                        max(float(np.linalg.norm(np.asarray(x), np.inf))
                            for x in out))
        return out

    _s2d._redheffer_star = _spy
    try:
        st.solve(**solve_kw)
    finally:
        _s2d._redheffer_star = orig
    return peak["G"], peak["kap"]


def _reassoc_bar(st, n_layers, **solve_kw):
    """Derived agreement bar for a REASSOCIATED fold ('fused' / 'tree').

    A reassociation perturbs each intermediate S-matrix at the rounding level,
    ``eps_mach * G``; the answer passes through at most ``2 n + 2`` stars, each
    amplified by its star denominator, whose worst measured norm on THIS fold
    is ``kappa``; ``1e4`` is the decade headroom.

    Measured envelope 2026-08-17 (tesla-ryzen, py3.14.6, numpy 2.4.4,
    scipy-openblas 0.3.31, BLAS pinned to 1) over layers {5, 10, 20, 40} x
    {normal, oblique, conical} x {all-distinct, graded taper}: worst
    observed/bar = **3.9e-3**, i.e. a 259x margin BELOW, and the smallest real
    signal (reversing an asymmetric stack) sits ~1e6x ABOVE.  Both sides are
    checked in :func:`test_p2t_reassoc_bar_has_a_gap_on_both_sides`.
    """
    G, kappa = _fold_conditioning(st, **solve_kw)
    return (1.0e4 * (2 * n_layers + 2) * np.finfo(np.float64).eps
            * G * max(1.0, kappa))


def _taper(nlay, base=8.0, span=0.20):
    """A design-121-like eps staircase: ``span`` grade over the whole stack."""
    return [(0.16e-6, _cell(base * (1.0 + span * i / max(1, nlay - 1))))
            for i in range(nlay)]


def _taper_oop(nlay, base=8.0, span=0.20):
    return [(0.16e-6,
             _tensor_cell(base * (1.0 + span * i / max(1, nlay - 1)), oop=True))
            for i in range(nlay)]


# ===================================================================== #
# 1.  the byte-identical restructure, against inline pre-P2T references
# ===================================================================== #
def _star_reference(SA, SB):
    """The PRE-P2T ``_redheffer_star``, written out here so the byte-identity
    claim is self-contained: literal identity substituted for D/F on a zero
    block, and every product left-associated exactly as ``@`` chains them."""
    A11, A12, A21, A22 = SA
    B11, B12, B21, B22 = SB
    n = A11.shape[0]
    I = np.eye(n, dtype=_C)
    if not bool(A22.any()) or not bool(B11.any()):
        D = I
        F = I
    else:
        D = _guarded_inverse(I - B11 @ A22, "ref")
        F = _guarded_inverse(I - A22 @ B11, "ref")
    C11 = A11 + A12 @ D @ B11 @ A21
    C12 = A12 @ D @ B12
    C21 = B21 @ F @ A21
    C22 = B22 + B21 @ F @ A22 @ B12
    return (C11, C12, C21, C22)


def _ifc_reference(Wa, Va, Wb, Vb):
    """The PRE-P2T ``_interface_smatrix``."""
    a = np.linalg.solve(Wb, Wa)
    b = np.linalg.solve(Vb, Va)
    apb, amb = a + b, a - b
    iapb = _guarded_inverse(apb, "ref")
    return (-iapb @ amb, 2.0 * iapb,
            0.5 * (apb - amb @ iapb @ amb), amb @ iapb)


def _ifc_gen_reference(Ma, Mb):
    """The PRE-P2T ``_interface_smatrix_general``."""
    n2 = Ma.shape[0] // 2
    T = np.linalg.solve(Mb, Ma)
    T11, T12, T21, T22 = T[:n2, :n2], T[:n2, n2:], T[n2:, :n2], T[n2:, n2:]
    iT22 = _guarded_inverse(T22, "ref")
    return (-iT22 @ T21, iT22, T11 - T12 @ iT22 @ T21, T12 @ iT22)


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.30, 0.70)])
def test_p2t_redheffer_star_is_byte_identical_to_the_pre_p2t_algebra(theta,
                                                                    phi):
    """Both the zero-block branches and the CSE must move ZERO bits.

    Three shapes are driven, because they take three different branches:
    ``star(ifc, prop)`` (B11 == 0, the half of every stack solve's stars that
    the propagation S-matrix feeds), ``star(prop, ifc)`` (A22 == 0), and
    ``star(ifc, ifc)`` (the general two-inverse path).
    """
    st = _stack([(0.16e-6, _cell(8.0)), (0.13e-6, _cell(11.0))],
                theta=theta, phi=phi)
    modes, _mk, k0, Wsup, Vsup = _modes_of(st)
    W0, V0, l0 = modes[0][1], modes[0][2], modes[0][3]
    W1, V1 = modes[1][1], modes[1][2]
    S_a = _interface_smatrix(Wsup, Vsup, W0, V0)
    S_b = _interface_smatrix(W0, V0, W1, V1)
    S_p = _propagation_smatrix(l0, k0 * 0.16e-6)
    for name, SA, SB in (("star(ifc, prop)", S_a, S_p),
                         ("star(prop, ifc)", S_p, S_b),
                         ("star(ifc, ifc)", S_a, S_b)):
        got = _redheffer_star(SA, SB)
        ref = _star_reference(SA, SB)
        assert _bytes_equal(ref, got), (
            f"{name}: max|d| = "
            f"{max(float(np.max(np.abs(x - y))) for x, y in zip(ref, got)):.3e}")


@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.30, 0.70)])
def test_p2t_interface_smatrix_cse_is_byte_identical(theta, phi):
    """Hoisting ``S22 = amb @ iapb`` out of ``S21`` must move ZERO bits: the
    two are the SAME product at the SAME association order."""
    st = _stack([(0.16e-6, _cell(8.0)), (0.13e-6, _cell(11.0))],
                theta=theta, phi=phi)
    modes, _mk, _k0, Wsup, Vsup = _modes_of(st)
    W0, V0 = modes[0][1], modes[0][2]
    W1, V1 = modes[1][1], modes[1][2]
    for Wa, Va, Wb, Vb in ((Wsup, Vsup, W0, V0), (W0, V0, W1, V1)):
        got = _interface_smatrix(Wa, Va, Wb, Vb)
        ref = _ifc_reference(Wa, Va, Wb, Vb)
        assert _bytes_equal(ref, got)


def test_p2t_generalized_interface_cse_is_byte_identical():
    """The same CSE on the GENERALIZED (out-of-plane) interface."""
    st = _stack([(0.16e-6, _tensor_cell(8.0, oop=True)),
                 (0.13e-6, _tensor_cell(11.0, oop=True))], theta=0.30,
                phi=0.70)
    modes, _mk, _k0, Wsup, Vsup = _modes_of(st)
    Ms = _modes_to_M(Wsup, Vsup, Wsup, -Vsup)
    for m in modes:
        assert m[0] == "gen", "the OOP cell must promote the cascade"
        Ml = _modes_to_M(m[1], m[2], m[4], m[5])
        got = _interface_smatrix_general(Ms, Ml)
        ref = _ifc_gen_reference(Ms, Ml)
        assert _bytes_equal(ref, got)
        Ms = Ml


def test_p2t_propagation_star_is_the_zero_block_branch_of_the_star():
    """``cascade='fused'`` swaps a REASSOCIATED form in -- pin that this is a
    real change of arithmetic and not a no-op, and that it is small.

    The fail-before half: if the fused form were byte-identical there would be
    nothing to gate, and the opt-in would be pointless.  The claim is that it
    differs at the ROUNDING level only, so the difference is asserted to be
    non-zero AND below a residual derived from the operands themselves.
    """
    from lumenairy.elements.rcwa._core import _propagation_star
    st = _stack([(0.16e-6, _cell(8.0)), (0.13e-6, _cell(11.0))],
                theta=0.30, phi=0.70)
    modes, _mk, k0, Wsup, Vsup = _modes_of(st)
    W0, V0, l0 = modes[0][1], modes[0][2], modes[0][3]
    S = _interface_smatrix(Wsup, Vsup, W0, V0)
    a = _redheffer_star(S, _propagation_smatrix(l0, k0 * 0.16e-6))
    b = _propagation_star(S, l0, k0 * 0.16e-6)
    d = max(float(np.max(np.abs(x - y))) for x, y in zip(a, b))
    scale = max(float(np.max(np.abs(x))) for x in a)
    # rounding floor of ONE complex product chain at this scale, x1e3 decades
    assert d <= 1.0e3 * scale * np.finfo(np.float64).eps, (
        f"fused star departs by {d:.3e} at scale {scale:.3e}")


# ===================================================================== #
# 2.  fused / tree vs the shipped sequential fold, at a DERIVED bar
# ===================================================================== #
@pytest.mark.parametrize("cascade", ["fused", "tree"])
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.30, 0.0), (0.30, 0.70)])
@pytest.mark.parametrize("shape", ["distinct", "repeated", "taper"])
def test_p2t_reassociated_fold_matches_sequential_within_derived_bar(
        cascade, theta, phi, shape):
    layers = {"distinct": [(0.16e-6, _cell(8.0 + 0.7 * i)) for i in range(6)],
              "repeated": [(0.16e-6, _cell(8.0)) for _ in range(6)],
              "taper": _taper(6)}[shape]
    ref = _stack(layers, theta=theta, phi=phi).solve()
    got = _stack(layers, theta=theta, phi=phi, cascade=cascade).solve()
    bar = _reassoc_bar(_stack(layers, theta=theta, phi=phi,
                              cascade=cascade), len(layers))
    d = _maxdiff(ref, got)
    assert d < bar, f"{cascade} {shape}: {d:.3e} vs derived bar {bar:.3e}"


@pytest.mark.parametrize("nlay", [5, 10, 20, 40])
def test_p2t_tree_agrees_across_layer_counts(nlay):
    """The tree's depth (and so its reassociation) grows with the stack; the
    bar grows with it too, and must keep containing the difference."""
    layers = _taper(nlay)
    ref = _stack(layers, theta=0.30, phi=0.70).solve()
    got = _stack(layers, theta=0.30, phi=0.70, cascade="tree").solve()
    bar = _reassoc_bar(_stack(layers, theta=0.30, phi=0.70,
                              cascade='tree'), nlay)
    d = _maxdiff(ref, got)
    assert d < bar, f"nlay={nlay}: {d:.3e} vs derived bar {bar:.3e}"


def test_p2t_tree_bar_holds_on_the_worst_measured_stack():
    """The deepest, most strongly graded stack in the 2026-08-17 envelope: 40
    layers, pillar eps DOUBLING across the stack, conical incidence.

    This is where the tree's loss of prefix anchoring is largest -- measured
    on this build, the tree fold ran at ``G`` ~2.6e+04 / ``kappa`` ~4.5e+02
    against the sequential fold's ~4.3e+01 / ~7.1, and tree-vs-sequential read
    2.5e-06 where ``'fused'`` on the SAME stack read 7.0e-11.  It is the arm
    that decides whether the derived bar is a real bound or an accident of
    shallow stacks, so it is asserted explicitly rather than left to the
    parametrized sweep (whose tapers are gentler).
    """
    layers = _taper(40, span=1.0)
    ref = _stack(layers, theta=0.30, phi=0.70).solve()
    tree = _stack(layers, theta=0.30, phi=0.70, cascade="tree").solve()
    fused = _stack(layers, theta=0.30, phi=0.70, cascade="fused").solve()
    bar = _reassoc_bar(_stack(layers, theta=0.30, phi=0.70, cascade="tree"),
                       len(layers))
    d_tree = _maxdiff(ref, tree)
    d_fused = _maxdiff(ref, fused)
    assert d_tree < bar, f"tree {d_tree:.3e} vs derived bar {bar:.3e}"
    assert d_fused < bar, f"fused {d_fused:.3e} vs derived bar {bar:.3e}"
    # the recorded ordering: reassociating ONE star costs far less than
    # reassociating the fold.  Stated as a relation, not as either reading.
    assert d_fused < d_tree, (
        f"fused {d_fused:.3e} is not below tree {d_tree:.3e} on the stack "
        f"where the fold's association is supposed to matter most")


def test_p2t_oop_tree_agrees_within_derived_bar():
    """The GENERALIZED (out-of-plane) cascade has its own tree; same bar."""
    layers = _taper_oop(6)
    ref = _stack(layers, theta=0.30, phi=0.70).solve()
    for cascade in ("fused", "tree"):
        got = _stack(layers, theta=0.30, phi=0.70, cascade=cascade).solve()
        bar = _reassoc_bar(_stack(layers, theta=0.30, phi=0.70,
                                  cascade=cascade), len(layers))
        d = _maxdiff(ref, got)
        assert d < bar, f"OOP {cascade}: {d:.3e} vs derived bar {bar:.3e}"


def test_p2t_reassoc_bar_has_a_gap_on_both_sides():
    """Rule 5: measured reassociation envelope below, smallest REAL signal
    above, decades to each.

    The real signal is a physical change the cascade must resolve: reversing
    an asymmetric stack.  It has to sit far ABOVE the bar, or the bar would be
    testing noise.
    """
    layers = _taper(6)
    # priced on the TREE's own fold, which is the worse-conditioned of the two
    bar = _reassoc_bar(_stack(layers, theta=0.30, phi=0.70, cascade="tree"),
                       len(layers))
    seq = _stack(layers, theta=0.30, phi=0.70).solve()
    obs = max(_maxdiff(seq, _stack(layers, theta=0.30, phi=0.70,
                                   cascade=c).solve())
              for c in ("fused", "tree"))
    rev = _stack(list(reversed(layers)), theta=0.30, phi=0.70).solve()
    signal = _maxdiff(seq, rev)
    assert obs < bar, f"observed {obs:.3e} above bar {bar:.3e}"
    assert signal > 1.0e2 * bar, (
        f"stack reversal moves only {signal:.3e}; bar {bar:.3e} has no gap "
        f"above it")


# ===================================================================== #
# 3.  the threaded contract
# ===================================================================== #
@pytest.mark.parametrize("cascade", ["fused", "tree"])
def test_p2t_worker_counts_are_byte_identical_to_each_other(cascade):
    """Inside the threaded contract (``max_workers`` an INT) the answer must
    not depend on the worker count: one process-wide BLAS cap is entered
    around every fan-out, so 1 / 2 / 4 / 8 workers see identical BLAS
    reduction orders and the tree's pairing is fixed by INPUT order, not by
    scheduling.  (Bit-equality is asserted only WITHIN the threaded contract;
    against the ``max_workers=None`` default only the derived bar holds -- the
    default enters no cap, which is the S3 environment-dependent shape.)"""
    layers = _taper(9)
    base = _stack(layers, theta=0.30, phi=0.70, cascade=cascade).solve(
        max_workers=1)
    for mw in (2, 4, 8):
        got = _stack(layers, theta=0.30, phi=0.70, cascade=cascade).solve(
            max_workers=mw)
        assert _bytes_equal(base[1:], got[1:]), f"max_workers={mw} moved bits"


def test_p2t_tree_serial_and_threaded_agree_at_the_bar():
    layers = _taper(9)
    st = _stack(layers, theta=0.30, phi=0.70)
    bar = _reassoc_bar(st, len(layers))
    a = _stack(layers, theta=0.30, phi=0.70, cascade="tree").solve()
    b = _stack(layers, theta=0.30, phi=0.70, cascade="tree").solve(
        max_workers=4)
    assert _maxdiff(a, b) < bar


# ===================================================================== #
# 4.  the tree is WORK-OPTIMAL and LOGARITHMIC
# ===================================================================== #
def _count_stars(st, **solve_kw):
    """Number of ``_redheffer_star`` calls, and the reduction DEPTH the tree
    reports, for one solve."""
    n = [0]
    orig = _s2d._redheffer_star

    def _spy(SA, SB):
        n[0] += 1
        return orig(SA, SB)

    _s2d._redheffer_star = _spy
    try:
        st.solve(**solve_kw)
    finally:
        _s2d._redheffer_star = orig
    return n[0]


def test_p2t_tree_does_the_same_number_of_stars_as_the_sequential_fold():
    """A reassociation must not cost extra work.

    The sequential fold does one star per layer against the propagation
    S-matrix and one per interface; the tree pairs each interface with its own
    layer's propagation at level 1 and then joins ``N`` times.  Both are
    ``2 N`` stars for ``N`` layers -- the tree buys ``O(log N)`` DEPTH at
    equal work, which is the whole claim.
    """
    layers = _taper(8)
    seq = _count_stars(_stack(layers, theta=0.30, phi=0.70, cascade="fused"))
    tree = _count_stars(_stack(layers, theta=0.30, phi=0.70, cascade="tree"))
    assert tree == seq == len(layers), (
        f"tree {tree} stars, sequential {seq}, {len(layers)} layers")


def test_p2t_tree_actually_reassociates():
    """The tree must be a DIFFERENT bracketing of the same chain.

    Fail-before: a ``'tree'`` that quietly folded left to right would use the
    same primitives as ``'fused'`` and come back BYTE-IDENTICAL to it, and
    every bar test above would still pass.  On a stack deep enough for the
    association to matter the two must differ -- and still agree at the
    derived bar.
    """
    layers = _taper(12)
    fused = _stack(layers, theta=0.30, phi=0.70, cascade="fused").solve()
    tree = _stack(layers, theta=0.30, phi=0.70, cascade="tree").solve()
    assert not _bytes_equal(fused[1:], tree[1:]), (
        "cascade='tree' is byte-identical to the sequential fused fold -- "
        "then it is not reassociating anything")
    bar = _reassoc_bar(_stack(layers, theta=0.30, phi=0.70, cascade="tree"),
                       len(layers))
    assert _maxdiff(fused, tree) < bar


@pytest.mark.parametrize("nlay", [4, 8, 16])
def test_p2t_tree_depth_is_logarithmic(nlay):
    st = _stack(_taper(nlay), theta=0.30, phi=0.70, cascade="tree")
    st.solve()
    d = st.cascade_stats()["tree"]["depth"]
    assert d == int(np.ceil(np.log2(nlay + 1))), (
        f"nlay={nlay}: depth {d}")
    assert d < nlay, "an O(log N) reduction must be shallower than the chain"


# ===================================================================== #
# 5.  unconditional physics oracles on the reassociated path
# ===================================================================== #
@pytest.mark.parametrize("cascade", ["fast", "fused", "tree"])
@pytest.mark.parametrize("theta,phi", [(0.0, 0.0), (0.30, 0.70)])
def test_p2t_per_polarization_energy_closure(cascade, theta, phi):
    """House convention: sum orders WITHIN a polarization, MAX over
    polarizations -- never sum the two.  A lossless stack closes; the bar is
    the SEQUENTIAL path's own closure on the same stack (the hybrid's
    ``n_orders`` Fourier floor, which no cascade strategy touches), times
    decades."""
    layers = _taper(8)
    _o, R0, T0, _J0 = _stack(layers, theta=theta, phi=phi).solve()
    floor = max(abs(float(R0[p].sum() + T0[p].sum()) - 1.0) for p in (0, 1))
    _o, R, T, _J = _stack(layers, theta=theta, phi=phi,
                          cascade=cascade).solve()
    for p in (0, 1):
        tot = float(R[p].sum() + T[p].sum())
        assert abs(tot - 1.0) <= max(10.0 * floor, 1e-9), (
            f"{cascade} pol {p}: R+T = {tot:.12f}, sequential floor "
            f"{floor:.3e}")


@pytest.mark.parametrize("cascade", ["fused", "tree"])
def test_p2t_reciprocity_survives_reassociation(cascade):
    """A real-SYMMETRIC permittivity tensor is reciprocal, so the specular
    reflection Jones matrix is symmetric (``J01 == J10``) at normal incidence
    -- the library's own validated reciprocity signature.  The reassociated
    fold must preserve it no worse than the sequential fold does."""
    layers = _taper_oop(6)
    _o, _R, _T, J0 = _stack(layers).solve()
    _o, _R, _T, J = _stack(layers, cascade=cascade).solve()
    base = abs(J0[0, 1] - J0[1, 0])
    got = abs(J[0, 1] - J[1, 0])
    scale = float(np.max(np.abs(J)))
    assert got <= max(10.0 * base, 1e3 * scale * np.finfo(np.float64).eps), (
        f"{cascade}: |J01-J10| = {got:.3e}, sequential {base:.3e}")


@pytest.mark.parametrize("cascade", ["fused", "tree"])
def test_p2t_thick_lossy_overflow_survives_reassociation(cascade):
    """Redheffer stability must survive the reassociation.

    The thickness is derived from the layer's OWN measured eigenvalues so that
    ``exp(+lam k0 t)`` provably overflows float64 ON THIS BUILD (asserted
    non-finite), then the S-cascade is shown finite, per-polarization
    ``R + T <= 1``, ``T`` extinguished, and agreeing with the sequential fold.
    """
    cell = _cell(9.0 + 3.0j, lo=2.1)
    probe = _stack([(0.2e-6, cell)], theta=0.20)
    modes, _mk, k0, _Ws, _Vs = _modes_of(probe)
    lam = np.asarray(modes[0][3])
    lam_max = float(np.max(np.real(lam)))
    assert lam_max > 0.0
    # smallest thickness whose forward exponent provably overflows float64
    thick = 1.05 * np.log(np.finfo(np.float64).max) / (lam_max * k0)
    with np.errstate(over="ignore"):
        assert not np.all(np.isfinite(np.exp(lam * k0 * thick))), (
            "the constructed thickness does not overflow on this build")
    layers = [(thick, cell), (0.15e-6, _cell(8.0)), (thick, cell)]
    ref = _stack(layers, theta=0.20).solve()
    got = _stack(layers, theta=0.20, cascade=cascade).solve()
    _o, R, T, J = got
    assert np.all(np.isfinite(R)) and np.all(np.isfinite(T))
    assert np.all(np.isfinite(J))
    for p in (0, 1):
        assert float(R[p].sum() + T[p].sum()) <= 1.0 + 1e-9
        assert float(T[p].sum()) < 1e-12
    bar = _reassoc_bar(_stack(layers, theta=0.20, cascade=cascade),
                       len(layers))
    assert _maxdiff(ref, got) < bar


def test_p2t_tree_matches_one_thick_layer_oracle():
    """INDEPENDENT oracle: a run of identical adjacent layers is one thicker
    layer, and a ONE-layer stack has no tree to reduce -- so this checks the
    tree against a path that exercises neither the tree nor the merge."""
    cell = _cell(12.0)
    t1, t2, t3 = 0.22e-6, 0.17e-6, 0.11e-6
    many = _stack([(t1, cell), (t2, cell), (t3, cell)], theta=0.30,
                  phi=0.70, cascade="tree").solve()
    one = _stack([(t1 + t2 + t3, cell)], theta=0.30, phi=0.70).solve()
    bar = _reassoc_bar(_stack([(t1, cell), (t2, cell), (t3, cell)],
                              theta=0.30, phi=0.70), 3)
    assert _maxdiff(many, one) < bar


# ===================================================================== #
# 6.  the tree's memory gate -- two-sided and FORCED
# ===================================================================== #
def test_p2t_tree_memory_gate_is_two_sided_and_forced():
    """Rule 4: force the precondition, assert the gate separately.

    Priced IN, the tree engages.  Priced OUT (``tree_max_bytes=1``), it
    REFUSES and the sequential FUSED fold runs instead -- which is a real
    fold, so the refused answer must be BYTE-IDENTICAL to ``cascade='fused'``,
    not merely close.  No ``pytest.skip`` and no dependence on how big the box
    is.
    """
    layers = _taper(8)
    rich = _stack(layers, theta=0.30, phi=0.70, cascade="tree",
                  tree_max_bytes=4 * 1024 ** 3)
    r_rich = rich.solve()
    s_rich = rich.cascade_stats()["tree"]
    assert s_rich["requested"] and s_rich["engaged"]
    assert s_rich["peak_bytes"] > 0 and s_rich["leaves"] == len(layers) + 1

    poor = _stack(layers, theta=0.30, phi=0.70, cascade="tree",
                  tree_max_bytes=1)
    r_poor = poor.solve()
    s_poor = poor.cascade_stats()["tree"]
    assert s_poor["requested"] and not s_poor["engaged"]

    fused = _stack(layers, theta=0.30, phi=0.70, cascade="fused").solve()
    assert _bytes_equal(r_poor[1:], fused[1:]), (
        "a refused tree must BE the sequential fused fold, bit for bit")
    assert _maxdiff(r_rich, r_poor) < _reassoc_bar(
        _stack(layers, theta=0.30, phi=0.70), len(layers))


def test_p2t_tree_peak_extra_bytes_is_the_measured_shape():
    """The projected peak is ``ceil(leaves/2)`` S-matrices of four
    ``block x block`` complex128 blocks -- stated as a RELATION between the
    knobs, not as an MB reading of one box."""
    for leaves, block in ((9, 162), (21, 242), (41, 578)):
        want = ((leaves + 1) // 2) * 4 * block ** 2 * 16
        assert _tree_peak_extra_bytes(leaves, block) == want
    # doubling the layer count doubles the extra set; doubling the basis
    # quadruples it
    assert (_tree_peak_extra_bytes(41, 242)
            == pytest.approx(2 * _tree_peak_extra_bytes(21, 242), rel=0.06))
    assert (_tree_peak_extra_bytes(21, 484)
            == 4 * _tree_peak_extra_bytes(21, 242))


def test_p2t_tree_budget_tracks_set_max_ram_at_query_time():
    """The budget is priced when the solve asks, not frozen at construction
    -- the LayerCache precedent."""
    import lumenairy
    st = _stack(_taper(4), cascade="tree")
    prev = None
    try:
        lumenairy.set_max_ram(64)
        big = st.tree_budget()
        lumenairy.set_max_ram(1024 ** 3 // 1)     # 1 GB in bytes
        small = st.tree_budget()
    finally:
        lumenairy.set_max_ram(prev)
    assert big > small > 0


# ===================================================================== #
# 7.  the contract split: what the DEFAULT still is
# ===================================================================== #
def test_p2t_default_cascade_is_still_fast_and_still_sequential():
    """The reassociated folds are OPT-IN.  A stack built with no ``cascade=``
    must not reassociate anything: its answer is BYTE-IDENTICAL to
    ``cascade='fast'`` asked for explicitly, and DIFFERENT from 'fused' /
    'tree' (which is what makes them worth gating)."""
    layers = _taper(6)
    st = _stack(layers, theta=0.30, phi=0.70)
    assert st.cascade == "fast"
    d = st.solve()
    explicit = _stack(layers, theta=0.30, phi=0.70, cascade="fast").solve()
    assert _bytes_equal(d[1:], explicit[1:])
    st.solve()
    assert st.cascade_stats()["tree"]["requested"] is False
    for c in ("fused", "tree"):
        other = _stack(layers, theta=0.30, phi=0.70, cascade=c).solve()
        assert not _bytes_equal(d[1:], other[1:]), (
            f"cascade={c!r} is byte-identical to the default -- then it is "
            f"not a reassociation and needs no opt-in")


def test_p2t_monolithic_is_still_the_escape_hatch():
    """``monolithic`` must remain the per-layer build-and-star cascade: no
    dedup, no merge, no fusion, no tree.  A repeated stack is where the
    difference shows."""
    layers = [(0.16e-6, _cell(8.0)) for _ in range(4)]
    mono = _stack(layers, cascade="monolithic")
    n_mono = _count_stars(mono)
    fast = _stack(layers, cascade="fast")
    n_fast = _count_stars(fast)
    assert n_mono == 2 * len(layers), (
        f"monolithic did {n_mono} stars for {len(layers)} layers")
    assert n_fast < n_mono, "the merge must remove stars on a repeated stack"


def test_p2t_unknown_cascade_raises():
    with pytest.raises(ValueError, match="cascade must be one of"):
        PMM2DStackHybrid(PX, cascade="balanced")


def test_p2t_retain_internal_forces_the_sequential_fold():
    """``retain_internal`` indexes the partial cascades per LAYER, so the tree
    stands down; the fusion still applies, and the internal fields must still
    reconstruct."""
    layers = _taper(4)
    st = _stack(layers, theta=0.20, cascade="tree")
    st.solve(retain_internal=True)
    assert st.cascade_stats()["tree"]["requested"] is False
    f = st.internal_field(0.1e-6, nx=8, component="E")
    assert np.all(np.isfinite(f["Ex"]))
    assert len(st._internal["S_above"]) == len(layers)
