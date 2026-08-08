"""M3 -- "Per-layer, efficiently" (roadmap N-3) + the promoted T3-4 guard.

See ``docs/audits/PMM_M3_EFFICIENCY_2026_08_04.md``.

Four efficiency changes and one guard, each pinned to the standard the campaign
plan set for it:

* **hoist** (N-3.1) and **vectorise** (N-3.2) -- BIT-IDENTICAL, asserted by
  tolerance at 0.0 on the max absolute difference, never ``array_equal``;
* **de-kron** (N-3.3) -- an EXACT factorisation whose floating-point result
  moves in the last bits; asserted against the exact-arithmetic identity, the
  fail-before switch, and a measured envelope;
* **inv -> solve** (N-3.4) -- shipped at the mortar, REFUSED at the star by a
  shipped cross-path identity; both directions pinned here;
* **T3-4** -- the modal classification guard, with its calibration's decisive
  rows pinned in both directions (it fires on the silent-wrong cells; it is
  silent on the false-positive control that a naive bar refuses).
"""
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import _core as PC
from lumenairy.elements.rcwa import _core as _rc

# ---------------------------------------------------------------------------
# the M2 audit-class device (a coated 2 deg taper on a 700 nm pitch), rebuilt
# from the parameters in AUDIT_PMM_OBLIQUE_INPLANE_UNION_GRID_2026_07_28 S4.4
# ---------------------------------------------------------------------------
NM = 1e-9
PERIOD, WL = 700 * NM, 1310 * NM
SIDEWALL, H1, COAT, W_TOP = np.deg2rad(2.0), 310 * NM, 5.0 * NM, 340 * NM
EPS_CORE, EPS_COAT, EPS_GROOVE = (3.48 + 0j) ** 2, (1.76 + 0j) ** 2, 1.0 + 0j
N_SUP = N_SUB = 1.5
THETA = np.deg2rad(8.0)


def _segments(w_core, coat=COAT, eps_core=EPS_CORE):
    w_out = 0.5 * (w_core + 2.0 * coat) / PERIOD
    w_in = 0.5 * w_core / PERIOD
    g, c = 0.5 - w_out, w_out - w_in
    return [(g, EPS_GROOVE), (c, EPS_COAT), (2 * w_in, eps_core),
            (c, EPS_COAT), (g, EPS_GROOVE)]


def _taper(ns):
    dz = H1 / ns
    return [(dz, _segments(2.0 * (0.5 * W_TOP
                                  - ((k + 0.5) / ns) * H1
                                  * np.tan(SIDEWALL))))
            for k in range(ns)]


def build(ns, degree=8, grids="per-layer", min_feature=None, ffo=11):
    kw = {} if min_feature is None else {"min_feature": min_feature}
    st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP,
                  degree=degree, elements_per_region=1, far_field_orders=ffo,
                  layer_grids=grids, **kw)
    for t, segs in _taper(ns):
        st.add_layer(t, segments=segs)
    return st


def solve0(st, theta=THETA, quiet=True):
    """``(R0, |R+T-1|)`` -- order-0 reflectance for incident pol 0.

    ``quiet=False`` lets warnings through: the T3-4 tests need to SEE them,
    and a helper that swallows the thing under test is a useless helper."""
    with warnings.catch_warnings():
        if quiet:
            warnings.simplefilter("ignore")
        else:                      # keep only the noisy geometry diagnostic out
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message=".*_pmm_union_grid.*")
        o, R, T, _J = st.set_source(WL, theta=theta).solve()
    m0 = int(np.where(np.asarray(o) == 0)[0][0])
    R, T = np.asarray(R), np.asarray(T)
    return (float(np.real(R[0, m0])),
            float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0))))


def full(st, theta=THETA):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.set_source(WL, theta=theta).solve()
    return np.asarray(o), np.asarray(R), np.asarray(T), np.asarray(J)


def maxdiff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape
    return float(np.max(np.abs(a.astype(np.complex128)
                               - b.astype(np.complex128)))) if a.size else 0.0


@pytest.fixture
def separable_off():
    """Fail-before switch for N-3 items 3 + 4 (the ONE switch the plan asks
    for), restored on the way out."""
    old = PC.PMM_MORTAR_SEPARABLE
    PC.PMM_MORTAR_SEPARABLE = False
    try:
        yield
    finally:
        PC.PMM_MORTAR_SEPARABLE = old


@pytest.fixture
def t34_armed():
    """The T3-4 guard ships DISARMED (its bar was refuted on the conical
    family -- see the block comment in ``_core.py`` and the audit's S5.5).
    These tests arm it deliberately, on the family it IS calibrated on."""
    old = PC.PMM_MODE_CUT_GUARD
    PC.PMM_MODE_CUT_GUARD = True
    try:
        yield
    finally:
        PC.PMM_MODE_CUT_GUARD = old


# ===========================================================================
# N-3 item 2 -- vectorised ``_lagrange_eval``: BIT-IDENTICAL
# ===========================================================================

def _lagrange_eval_pre_m3(ref_nodes, xi):
    """The PRE-M3 implementation, VERBATIM.  Do not tidy: this is the
    reference the vectorised form is pinned against, and a paraphrase would
    silently weaken the pin."""
    ref = np.asarray(ref_nodes, dtype=float)
    p1 = ref.size
    wb = np.ones(p1)
    for j in range(p1):
        for k in range(p1):
            if k != j:
                wb[j] /= (ref[j] - ref[k])
    xi = np.atleast_1d(np.asarray(xi, dtype=float))
    out = np.zeros((xi.size, p1))
    for r, x in enumerate(xi):
        diff = x - ref
        hit = np.abs(diff) < 1e-14
        if hit.any():
            out[r, int(np.argmax(hit))] = 1.0
        else:
            num = wb / diff
            out[r] = num / num.sum()
    return out


@pytest.mark.parametrize("degree", [1, 2, 4, 6, 8, 10, 12, 16, 20])
def test_lagrange_eval_is_bit_identical_to_the_pre_m3_loop(degree):
    """Tolerance AT 0.0, on every input shape the library actually produces
    plus the awkward ones: exact node hits (the unit-row branch), a point
    just inside the 1e-14 hit window, one just outside, and a scalar."""
    ref, _w = PC._gll_nodes_weights(degree)
    rng = np.random.default_rng(11 + degree)
    cases = [np.polynomial.legendre.leggauss(degree + 2)[0],
             np.polynomial.legendre.leggauss(degree + 4)[0],
             rng.uniform(-1.0, 1.0, 41),
             np.asarray(ref),                                  # every node
             np.concatenate([np.asarray(ref), rng.uniform(-1, 1, 3)]),
             np.array([float(ref[degree // 2]) + 5e-15]),      # inside 1e-14
             np.array([float(ref[0]) - 1e-13]),                # outside
             np.array([0.0])]
    for xi in cases:
        got = PC._lagrange_eval(ref, xi)
        want = _lagrange_eval_pre_m3(ref, xi)
        assert got.shape == want.shape
        assert maxdiff(got, want) == 0.0, (degree, xi[:4])
    # scalar contract
    assert PC._lagrange_eval(ref, 0.3).shape == (1, degree + 1)
    # and it is still a partition of unity (the null floor: what "right" costs)
    row = PC._lagrange_eval(ref, np.array([0.137]))
    assert abs(float(row.sum()) - 1.0) < 1e-12


def test_lagrange_barycentric_weights_are_cached_frozen_and_recompute_equal():
    """The memo must be a pure memo: same bytes on a hit, same bytes after the
    registry drains it, and read-only so a caller cannot poison it."""
    ref, _w = PC._gll_nodes_weights(9)
    PC._clear_pmm_caches()
    first = PC._lagrange_bary_weights(np.asarray(ref, dtype=float))
    second = PC._lagrange_bary_weights(np.asarray(ref, dtype=float))
    assert second is first                       # a hit, by identity
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[0] = 1.0
    kept = np.array(first)
    PC._clear_pmm_caches()
    again = PC._lagrange_bary_weights(np.asarray(ref, dtype=float))
    assert again is not first
    assert maxdiff(again, kept) == 0.0


# ===========================================================================
# N-3 item 1 -- the geometry cache: complete key, bit-identical, actually hit
# ===========================================================================

def _grids(ns=6, degree=8, min_feature=3.0 * NM):
    gof = PC._perlayer_window_grids([L[1] for L in build(ns)._layers],
                                    min_feature / PERIOD, 1)
    return [PC._build_sem_tensor_segments(
        PERIOD, w, [PC._tensor3_dict(e) for e in row], degree, 1, 1.0)
        for w, row in gof]


def test_geometry_fingerprint_covers_exactly_what_the_two_builders_read():
    """The cache key is derived from the CONTENT the builders consume, so this
    pins both directions: a change in any consumed field must change the key,
    and a change in ``eps`` -- which neither builder reads -- must NOT."""
    mats = _grids()
    a, b = mats[0], mats[1]
    assert PC._geo_fingerprint(a) == PC._geo_fingerprint(a)
    assert PC._geo_fingerprint(a) != PC._geo_fingerprint(b)

    # eps is NOT consumed: same walls, different material -> same key AND the
    # same mass / cross-mass bytes.  (If this ever fails, the cache is unsound.)
    gof = PC._perlayer_window_grids([L[1] for L in build(6)._layers],
                                    3.0 * NM / PERIOD, 1)
    w0, row0 = gof[0]
    m_air = PC._build_sem_tensor_segments(
        PERIOD, w0, [PC._tensor3_dict(1.0 + 0j)] * len(row0), 8, 1, 1.0)
    m_real = PC._build_sem_tensor_segments(
        PERIOD, w0, [PC._tensor3_dict(e) for e in row0], 8, 1, 1.0)
    assert PC._geo_fingerprint(m_air) == PC._geo_fingerprint(m_real)
    assert maxdiff(PC._sem_mass_exact(m_air), PC._sem_mass_exact(m_real)) == 0.0
    assert maxdiff(PC._sem_cross_mass(m_air, mats[1]),
                   PC._sem_cross_mass(m_real, mats[1])) == 0.0

    # every consumed field, perturbed one at a time
    import copy
    for field, mutate in (
            ("degree", lambda d: d.__setitem__("degree", d["degree"] + 1)),
            ("n_glob", lambda d: d.__setitem__("n_glob", d["n_glob"] + 1)),
            ("ref_nodes",
             lambda d: d.__setitem__("ref_nodes",
                                     np.asarray(d["ref_nodes"]) + 1e-15)),
            ("elem_bnds",
             lambda d: d.__setitem__(
                 "elem_bnds",
                 [(b0[0], b0[1] + 1e-15) + tuple(b0[2:])
                  for b0 in d["elem_bnds"]])),
            ("l2g", lambda d: d.__setitem__("l2g",
                                            np.asarray(d["l2g"]) + 1))):
        d = copy.copy(a)
        d["elem_bnds"] = list(a["elem_bnds"])
        mutate(d)
        assert PC._geo_fingerprint(d) != PC._geo_fingerprint(a), field


def test_geometry_cache_returns_the_uncached_bytes_and_is_hit():
    mats = _grids()
    PC._clear_pmm_caches()
    ref_m = PC._sem_mass_exact(mats[0])
    ref_c = PC._sem_cross_mass(mats[0], mats[1])
    h0 = int(PC._PERLAYER_GEO_CACHE.hits)
    got_m = PC._sem_mass_exact_cached(mats[0])          # miss
    got_c = PC._sem_cross_mass_cached(mats[0], mats[1])  # miss
    assert maxdiff(got_m, ref_m) == 0.0
    assert maxdiff(got_c, ref_c) == 0.0
    assert PC._sem_mass_exact_cached(mats[0]) is got_m   # hit, by identity
    assert PC._sem_cross_mass_cached(mats[0], mats[1]) is got_c
    assert int(PC._PERLAYER_GEO_CACHE.hits) > h0
    # frozen against poisoning (the W7 A13 policy)
    assert not got_m.flags.writeable and not got_c.flags.writeable
    with pytest.raises(ValueError):
        got_c[0, 0] = 0.0
    # and the registry drains it
    PC._clear_pmm_caches()
    assert PC._sem_mass_exact_cached(mats[0]) is not got_m


@pytest.mark.parametrize("grids", ["per-layer", "shared"])
def test_repeat_solves_are_bit_identical_cold_and_warm(grids):
    """The hoist's whole safety claim in one line: the SECOND solve of a stack
    -- served entirely from the geometry cache -- must return the FIRST one's
    bits, and a solve after the cache is drained must return them too."""
    PC._clear_pmm_caches()
    st = build(6, degree=8, grids=grids, min_feature=3.0 * NM)
    a = full(st)
    b = full(build(6, degree=8, grids=grids, min_feature=3.0 * NM))
    PC._clear_pmm_caches()
    c = full(build(6, degree=8, grids=grids, min_feature=3.0 * NM))
    for i in (1, 2, 3):
        assert maxdiff(a[i], b[i]) == 0.0, ("warm", i)
        assert maxdiff(a[i], c[i]) == 0.0, ("cold", i)


# ===========================================================================
# N-3 item 3 -- the mortar is SEPARABLE, and that is an identity
# ===========================================================================

def test_kron2_apply_is_the_exact_factorisation_not_an_approximation():
    """Two claims, and they are different claims.

    1. EXACTNESS: on operands where every product and sum is representable,
       the blockwise form equals the dense ``kron(I2, M) @ X`` at tolerance
       0.0 -- because the terms it drops are STRUCTURAL zeros, not small
       numbers.
    2. FLOATING POINT: on general operands BLAS accumulates the length-2h dot
       product differently from the length-h one, so the last bits move.  The
       envelope is machine-eps-relative, and this pins that it IS only that."""
    rng = np.random.default_rng(5)
    # (1) exactly representable integers -> tolerance 0.0
    M = rng.integers(-8, 9, (6, 6)).astype(float)
    X = rng.integers(-8, 9, (12, 5)).astype(float) \
        + 1j * rng.integers(-8, 9, (12, 5))
    assert maxdiff(PC._kron2_apply(M, X), np.kron(np.eye(2), M) @ X) == 0.0
    # rectangular M (the cross-mass shape) too
    M2 = rng.integers(-8, 9, (7, 6)).astype(float)
    assert maxdiff(PC._kron2_apply(M2, X), np.kron(np.eye(2), M2) @ X) == 0.0

    # (2) general float operands: last bits only
    for n in (8, 40):
        Mf = rng.standard_normal((n, n))
        Xf = (rng.standard_normal((2 * n, 2 * n))
              + 1j * rng.standard_normal((2 * n, 2 * n)))
        dense = np.kron(np.eye(2), Mf) @ Xf
        sep = PC._kron2_apply(Mf, Xf)
        scale = float(np.max(np.abs(dense)))
        assert maxdiff(sep, dense) < 64.0 * n * np.finfo(float).eps * scale

    # and the operator is never materialised: the separable form allocates
    # nothing of the 2n x 2n shape
    assert PC._kron2_apply(M, X).shape == (12, 5)


def test_the_separable_mortar_builds_no_kron_operator_at_all():
    """The MEMORY claim, measured rather than argued: with the switch on, a
    per-layer solve calls ``np.kron`` for zero bytes of mortar operator; with
    it off, it builds four ``kron(I2, .)`` per non-conforming interface."""
    def kron_bytes(sep):
        old_flag, old_kron = PC.PMM_MORTAR_SEPARABLE, np.kron
        tot = [0]

        def spy(a, b):
            r = old_kron(a, b)
            tot[0] += int(np.asarray(r).nbytes)
            return r
        PC.PMM_MORTAR_SEPARABLE = sep
        np.kron = spy
        try:
            PC._clear_pmm_caches()
            solve0(build(6, degree=8, min_feature=3.0 * NM))
        finally:
            np.kron = old_kron
            PC.PMM_MORTAR_SEPARABLE = old_flag
        return tot[0]

    off, on = kron_bytes(False), kron_bytes(True)
    assert off > 0, "the fail-before arm must still build them"
    assert on == 0, f"separable arm still allocated {on} kron bytes"


# ===========================================================================
# N-3 items 3 + 4 -- the envelope, and the ONE fail-before switch
# ===========================================================================

@pytest.mark.parametrize("ns,degree", [(2, 8), (6, 8), (6, 10), (12, 8)])
def test_separable_and_solve_move_only_the_last_bits(ns, degree,
                                                     separable_off):
    """Per CONFIGURATION, not in aggregate (the campaign's rule 8).

    The fixture turns the switch OFF for the reference solve; the envelope is
    measured against it with the switch back ON."""
    st_off = build(ns, degree=degree, min_feature=3.0 * NM)
    o0, R0, T0, J0 = full(st_off)
    close_off = float(np.max(np.abs(R0.sum(1) + T0.sum(1) - 1.0)))
    PC.PMM_MORTAR_SEPARABLE = True
    st_on = build(ns, degree=degree, min_feature=3.0 * NM)
    o1, R1, T1, J1 = full(st_on)
    close_on = float(np.max(np.abs(R1.sum(1) + T1.sum(1) - 1.0)))

    assert np.array_equal(o0, o1)
    scale = max(float(np.max(np.abs(R0))), 1e-3)
    assert maxdiff(R0, R1) < 1e-10 * scale
    assert maxdiff(T0, T1) < 1e-10 * max(float(np.max(np.abs(T0))), 1e-3)
    assert maxdiff(J0, J1) < 1e-10 * max(float(np.max(np.abs(J0))), 1e-3)
    # conservation must not get WORSE (comparative, not an absolute bar)
    assert close_on <= max(10.0 * close_off, 1e-9)


def test_fail_before_switch_reproduces_the_pre_m3_answer(separable_off):
    """With the switch off the answer must be REPRODUCIBLE bit for bit across
    repeats -- the control the envelope above is measured against."""
    a = full(build(6, degree=8, min_feature=3.0 * NM))
    b = full(build(6, degree=8, min_feature=3.0 * NM))
    for i in (1, 2, 3):
        assert maxdiff(a[i], b[i]) == 0.0


def test_the_mortar_still_reduces_to_the_plain_interface_on_identical_grids():
    """The identity pin the plan names as the de-kron's oracle: on identical
    grids ``Ma = Mb = Cab`` and the mortar blocks must equal
    :func:`_interface_smatrix`'s.  Re-asserted HERE (not only in the per-layer
    suite) because it is what says the separable form did not change physics."""
    mats = _grids()
    m = mats[0]
    k0 = 2.0 * np.pi / WL
    W, V, _lam, _q = PC._sem_modes_tensor(m, k0, 0.3 * k0, True)
    Mass = PC._sem_mass_exact_cached(m)
    Cab = PC._sem_cross_mass_cached(m, m)
    plain = PC._interface_smatrix(W, V, W, V)
    mort = PC._interface_smatrix_mortar(W, V, W, V, Mass, Mass, Cab)
    for k in range(4):
        s = max(float(np.max(np.abs(plain[k]))), 1.0)
        assert maxdiff(plain[k], mort[k]) < 1e-9 * s, k


# ===========================================================================
# N-3 item 4 -- shipped at the mortar, REFUSED at the star.  Both pinned.
# ===========================================================================

def test_guarded_solve_carries_the_same_census_as_the_guarded_inverse():
    """``_guarded_solve`` introduces no threshold and no policy: it records the
    SAME instruments, under the SAME site name, and returns the backward-stable
    answer.  Default path: nothing recorded, nothing extra computed."""
    rng = np.random.default_rng(3)
    n = 24
    Q, _ = np.linalg.qr(rng.standard_normal((n, n))
                        + 1j * rng.standard_normal((n, n)))
    A = (Q * np.logspace(0, -20, n)) @ Q.conj().T        # cond ~1e20
    B = rng.standard_normal((n, 3)) + 0j

    assert _rc._INV_CENSUS is None
    X = PC._guarded_solve(A, B, "probe")                 # must NOT raise
    assert maxdiff(X, np.linalg.solve(A, B)) == 0.0

    _rc._INV_CENSUS = []
    try:
        PC._guarded_solve(A, B, "probe")
        assert len(_rc._INV_CENSUS) == 1
        site, dim, rcond_eq, resid_eq, refused = _rc._INV_CENSUS[0]
        assert site == "probe" and dim == n and refused is False
        assert rcond_eq < _rc._INV_RCOND_SCREEN
        assert resid_eq is not None and resid_eq > _rc._INV_RESID_REFUSE
    finally:
        _rc._INV_CENSUS = None

    # the RIGHT solve computes X @ inv(A), to solver round-off
    Xr = PC._guarded_solve_right(np.eye(n) + 0.1 * Q, B[:, :1].T.repeat(n, 0),
                                 "probe")
    want = B[:, :1].T.repeat(n, 0) @ np.linalg.inv(np.eye(n) + 0.1 * Q)
    assert maxdiff(Xr, want) < 1e-10 * max(float(np.max(np.abs(want))), 1.0)


def test_the_star_keeps_its_inverse_because_a_solve_breaks_a_shipped_identity():
    """PROVE the refusal recorded in ``_redheffer_star_rect``'s docstring
    rather than assert it.

    A conforming per-layer stack must equal the shared-grid stack BIT FOR BIT
    (five suites pin it).  Rebuilding the star with the backward-stable RIGHT
    solve -- which is algebraically identical -- can move the answer off those
    bits, and where it does the conversion is refused until the shared twin
    moves with it.

    WHICH BLOCK IT MOVES, AND WHETHER IT MOVES ONE AT ALL, IS A PER-BUILD
    FACT, and the first draft asserted it as a universal one
    (``assert moved > 0.0`` on the R block alone).  Measured 2026-08-05, max
    |difference| against the shared twin:

        build / BLAS threads        R block      Jones block
        Windows OpenBLAS-Haswell N  5.551e-17    2.001e-16
        Windows OpenBLAS-Haswell 1  5.551e-17    8.327e-17
        WSL OpenBLAS-SkylakeX    N  0.0          5.551e-17   <- the failure
        WSL OpenBLAS-SkylakeX    1  1.388e-17    5.551e-17

    -- so the substitution IS separable from the inverse on every build; the
    R block merely happened to coincide on one of them.  This reads the max
    over both blocks for that reason, and still treats an exact coincidence as
    a PASS: where ``solve`` emits the inverse's bits the identity survives,
    which is not a failure of anything.  Pinned on every build: the shipped
    identity holds; the experiment really ran; the substitution can only touch
    the last bits; and where it does touch them, byte-identity -- the thing
    five suites assert at tolerance 0.0 -- is broken, which is the refusal's
    whole justification."""
    calls = [0]

    def star_solve(SA, SB):
        calls[0] += 1
        A11, A12, A21, A22 = SA
        B11, B12, B21, B22 = SB
        n = A22.shape[0]
        I = np.eye(n, dtype=PC._C)
        if not bool(A22.any()) or not bool(B11.any()):
            AD, BF = A12 @ I, B21 @ I
        else:
            AD = PC._guarded_solve_right(I - B11 @ A22, A12, "probe")
            BF = PC._guarded_solve_right(I - A22 @ B11, B21, "probe")
        return (A11 + AD @ B11 @ A21, AD @ B12, BF @ A21,
                B22 + BF @ A22 @ B12)

    # a CONFORMING per-layer stack: identical walls on every layer
    def conforming(grids):
        st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP,
                      degree=8, far_field_orders=11, layer_grids=grids)
        for _ in range(4):
            st.add_layer(80 * NM, segments=_segments(320 * NM))
        return st

    base_pl = full(conforming("per-layer"))
    base_sh = full(conforming("shared"))
    assert maxdiff(base_pl[1], base_sh[1]) == 0.0, \
        "the shipped identity must hold before the experiment"

    import lumenairy.elements.pmm.stack as PS
    old = PS._redheffer_star_rect
    PS._redheffer_star_rect = star_solve
    try:
        alt = full(conforming("per-layer"))
    finally:
        PS._redheffer_star_rect = old
    # the experiment was actually PERFORMED.  This is the half a bits-moved
    # count cannot stand in for: a substitution that was never reached would
    # also "move nothing", and would look like a pass.
    assert calls[0] >= 3, (
        f"the substituted star ran {calls[0]} times on a 4-layer stack: the "
        f"fail-before switch did not take, so nothing was demonstrated")
    moved_R = maxdiff(alt[1], base_sh[1])
    moved_J = maxdiff(alt[3], base_sh[3])
    moved = max(moved_R, moved_J)
    assert moved < 1e-9, (
        f"the solve moved more than the last bits (R {moved_R:.3e}, J "
        f"{moved_J:.3e}) -- if this is large the refusal reasoning is wrong "
        f"and the star is buggy")
    if moved > 0.0:
        # this build separates the two paths: the byte-identity that five
        # suites assert at tolerance 0.0 is BROKEN by the substitution, which
        # is exactly the recorded reason for keeping the explicit inverse.
        assert (moved_R, moved_J) != (0.0, 0.0)
        assert moved <= 64.0 * np.finfo(float).eps, (
            f"R {moved_R:.3e}, J {moved_J:.3e}: an algebraically identical "
            f"substitution must not move more than solver round-off")
    else:
        # this build does not: `solve` returned the inverse's bits everywhere,
        # so the identity survives the substitution untouched.
        assert moved_R == 0.0 and moved_J == 0.0


# ===========================================================================
# T3-4 -- the promoted guard.  Both directions.
# ===========================================================================

def _t34_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn()
    return out, [w for w in caught
                 if isinstance(w.message, PC._ModeClassificationWarning)]


#: The candidate family the wrong-cell scan below partitions.  The DEVICE and
#: the DEFECT are build-independent -- M2's threshold rule says the library
#: default leaves a 0.4127 nm (ns = 2) / 1.8042 nm (ns = 6) cross-layer sliver,
#: and the degree ladder collapses somewhere above degree 8 on both -- but
#: WHICH degree collapses is not: it is decided by whether a round-off-flux
#: mode lands above or below the classification cut, i.e. by BLAS reduction
#: order.  So the cells are SCANNED and partitioned per build, never listed.
_T34_FAMILY = [(ns, deg) for ns in (2, 6) for deg in (10, 12, 14, 16)]
_T34_WIDEN = [(ns, deg) for ns in (2, 6) for deg in (18, 20)]
_T34_REF = {2: 0.1100920, 6: 0.1111090}             # RCWA, 141 orders
_T34_WRONG_REL = 0.05                               # 5 %: see the docstring


def _t34_scan(cells):
    """``[(ns, degree, rel_error, closure, fired)]`` over ``cells``."""
    out = []
    for ns, degree in cells:
        (r0, close), warns = _t34_warnings(
            lambda: solve0(build(ns, degree=degree), quiet=False))
        out.append((ns, degree, abs(r0 / _T34_REF[ns] - 1.0), close,
                    bool(warns), warns))
    return out


def test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build(t34_armed):
    """The cells M2 S5 measured as UNITARY BUT WRONG -- SCANNED, not listed
    (PMM_FOURNAME_ADJUDICATION_2026_08_05 S3).

    The previous form hard-coded ``(ns, degree, rel_min)`` triples.  Two of
    the three then failed on a second build for OPPOSITE reasons: ``(2, 12)``
    classifies CORRECTLY at N BLAS threads (rel 0.0047, so "the cell is
    supposed to be WRONG" is false there), and ``(6, 10)`` is wrong on both
    (0.5683670, 411 %) but the guard's stack-level ``spread`` half reads 1 at
    one thread and 0 at N -- the SAME wrong answer, with the guard speaking on
    one mount and silent on the other.  A list of wrong cells is therefore a
    per-build fact asserted as a universal one.

    What IS universal: this family contains silent-wrong cells on every build,
    and the guard must speak on all of them.  So the scan partitions the
    family on THIS build by the RCWA-anchored reference and asserts

      (a) the family really does contain a wrong cell here (widening the scan
          if the primary range does not -- never skipping);
      (b) conservation is blind to every one of them;
      (c) the guard fires on every one of them.

    The 5 % partition bar is not a calibration: the two populations are three
    decades apart (correct cells 0.03-0.9 %, wrong cells 42-466 %) on both
    builds and at 1 and N BLAS threads."""
    rows = _t34_scan(_T34_FAMILY)
    wrong = [r for r in rows if r[2] > _T34_WRONG_REL]
    if not wrong:                        # (a) WIDEN, do not skip
        rows += _t34_scan(_T34_WIDEN)
        wrong = [r for r in rows if r[2] > _T34_WRONG_REL]
    assert wrong, (
        "no silent-wrong cell in the scanned family on this build -- the "
        "sliver defect the T3-4 guard exists to find has stopped reproducing "
        "on the audit-class device.  If it was FIXED, re-pin this against the "
        "fix; if the family merely moved, widen _T34_WIDEN further.  "
        f"scanned: {[(r[0], r[1], round(r[2], 4)) for r in rows]}")
    # the partition really is a partition, not a bar sitting inside one cloud
    right = [r for r in rows if r[2] <= _T34_WRONG_REL]
    assert min(r[2] for r in wrong) > 20.0 * max(
        (r[2] for r in right), default=0.0), (
        f"the 5% partition is inside the population, not between two: "
        f"wrong {[round(r[2], 4) for r in wrong]} vs right "
        f"{[round(r[2], 4) for r in right]}")
    for ns, degree, rel, close, fired, warns in wrong:
        assert close < 1e-5, (                       # (b)
            f"ns={ns} deg={degree}: rel={rel:.4g} wrong but |R+T-1|="
            f"{close:.3e} -- conservation is supposed to be blind to this")
        assert fired, (                              # (c)
            f"ns={ns} deg={degree}: rel={rel:.4g} WRONG and the T3-4 guard "
            f"stayed silent -- it must speak where conservation cannot")
        msg = str(warns[0].message)
        assert "UNITARY BUT WRONG" in msg and "min_feature" in msg


def test_t34_guard_is_silent_on_the_cured_ladder(t34_armed):
    """The FALSE-POSITIVE control for the scan above, and the one control that
    is build-independent by construction: raise ``min_feature`` above M2's
    threshold ``min(off, |coat - off|)`` and there is no sliver left to
    misclassify, so every rung is right and the guard must say nothing at any
    degree.

      ns = 2: off = 5.4127 nm, coat 5 nm -> threshold 0.4127 nm, cured at 0.5
      ns = 6: off = 1.8042 nm, coat 5 nm -> threshold 1.8042 nm, cured at 3.0

    (0.5 nm is deliberately NOT used at ns = 6: it is BELOW that threshold, so
    it cures nothing and those cells belong to the scan above, not here.)"""
    for ns, mf in ((2, 0.5 * NM), (6, 3.0 * NM)):
        for degree in (6, 8, 10, 12, 14, 16):
            (r0, close), warns = _t34_warnings(
                lambda: solve0(build(ns, degree=degree, min_feature=mf),
                               quiet=False))
            rel = abs(r0 / _T34_REF[ns] - 1.0)
            assert rel < 0.02, (
                f"ns={ns} deg={degree} mf={mf / NM} nm: rel={rel:.4g} -- the "
                f"cured ladder is supposed to be RIGHT, so this control has "
                f"stopped being a control")
            assert not warns, (
                f"ns={ns} deg={degree} mf={mf / NM} nm: FALSE POSITIVE on a "
                f"cured cell (rel={rel:.4g}) -- {warns[0].message}")


#: "RIGHT" for the false-positive control below.  The populations it separates
#: are 0.03-0.9 % and 42-466 %, so nothing between 0.01 and 0.4 changes a
#: verdict.
_T34_RIGHT_REL = 0.02

#: Degrees the control may fall back to when a pinned rung has migrated,
#: coarsest last.  6 is the floor: below it the device is not resolved at all.
_T34_DEGREE_LADDER = (16, 14, 12, 10, 8, 6)

_T34_CELLS = {}


def _t34_cell(ns, degree, mf):
    """``dict(ns, degree, mf, r0, rel, close, fired, warns)``, cached."""
    key = (ns, degree, mf)
    if key not in _T34_CELLS:
        old, PC.PMM_MODE_CUT_GUARD = PC.PMM_MODE_CUT_GUARD, True
        try:
            (r0, close), warns = _t34_warnings(
                lambda: solve0(build(ns, degree=degree, min_feature=mf),
                               quiet=False))
        finally:
            PC.PMM_MODE_CUT_GUARD = old
        _T34_CELLS[key] = dict(ns=ns, degree=degree, mf=mf, r0=r0, close=close,
                               rel=abs(r0 / _T34_REF[ns] - 1.0),
                               fired=bool(warns), warns=warns)
    return _T34_CELLS[key]


@pytest.mark.parametrize("ns,degree,mf", [(2, 8, None), (2, 6, None),
                                          (2, 12, 0.5 * NM),
                                          (2, 16, 0.5 * NM),
                                          (6, 8, 3.0 * NM),
                                          (6, 16, 3.0 * NM)])
def test_t34_guard_is_silent_where_the_answer_is_right(ns, degree, mf,
                                                      t34_armed):
    """The false-positive control on the SAME device: a guard that refused
    these would be worse than the defect (the campaign's R-1b precedent).

    **v5.33.1 -- "THE CELL IS SUPPOSED TO BE RIGHT" IS A PRECONDITION, AND ON
    THE UNCURED LADDER IT IS A PER-BUILD ONE.**  ``(2, 8, None)`` reads
    ``rel`` = 0.0057 on Windows and WSL at 1, 2 and N BLAS threads and on the
    ubuntu CI images carrying numpy 2.4.6, and ``rel`` = 4.664 on the ubuntu
    image carrying numpy 2.2.6 / scipy 1.15.3 -- i.e. there the uncured
    ladder's collapse starts at degree 8 instead of degree 12/14, and the cell
    has migrated OUT of this control's population and INTO the silent-wrong
    family that ``test_t34_guard_fires_on_every_silent_wrong_cell_of_this_build``
    owns.  4.664 is exactly the collapse signature this file measures at
    ``(2, 14)`` and ``(2, 16)`` elsewhere; nothing about the device changed.

    A precondition that fails is RE-ESTABLISHED, not asserted: the control
    falls back down the degree ladder to the coarsest rung that IS right on
    this build and makes its claim there.  The migrated rung is not silently
    dropped -- its conservation is still asserted blind (the silent-wrong
    signature) and its verdict is printed.

    (The four CURED parametrizations are build-independent by construction --
    ``min_feature`` above M2's threshold leaves no sliver -- and are measured
    right and quiet in every cell; the fallback is inert on them.)"""
    cell = _t34_cell(ns, degree, mf)
    if cell["rel"] >= _T34_RIGHT_REL:
        # the pinned rung has migrated into the wrong family on this build.
        lower = [d for d in _T34_DEGREE_LADDER if d < degree]
        alt = next((c for c in (_t34_cell(ns, d, mf) for d in lower)
                    if c["rel"] < _T34_RIGHT_REL), None)
        assert alt is not None, (
            f"ns={ns} mf={mf}: degree {degree} reads rel={cell['rel']:.4g} "
            f"and no coarser rung down to degree {min(_T34_DEGREE_LADDER)} is "
            f"right either, so this device has no correct rung on this build "
            f"and there is nothing left to control.  ladder: "
            + str([(c['degree'], round(c['rel'], 4))
                   for c in (_t34_cell(ns, d, mf)
                             for d in (degree,) + tuple(lower))]))
        # the migration IS the silent-wrong defect and not a different device:
        # conservation stays blind to it, which is the whole T3-4 premise.
        assert cell["close"] < 1e-5, (
            f"ns={ns} deg={degree}: rel={cell['rel']:.4g} wrong AND "
            f"|R+T-1|={cell['close']:.3e} -- energy closure would have caught "
            f"this, so it is not the silent-wrong family")
        print(f"\nT3-4 FP control: ns={ns} degree={degree} mf={mf} has "
              f"migrated (rel={cell['rel']:.4g}, |R+T-1|={cell['close']:.3e}, "
              f"guard {'FIRES' if cell['fired'] else 'is QUIET'}); the control "
              f"falls back to degree {alt['degree']} (rel={alt['rel']:.4g}).")
        cell = alt
    assert cell["rel"] < _T34_RIGHT_REL, "the cell is supposed to be RIGHT"
    assert not cell["fired"], (
        f"false positive on ns={cell['ns']} degree={cell['degree']} "
        f"mf={cell['mf']} (rel={cell['rel']:.4g}): "
        f"{cell['warns'] and cell['warns'][0].message}")


def test_t34_guard_is_silent_when_layers_LEGITIMATELY_differ(t34_armed):
    """The instrument that a naive bar gets wrong, pinned as a control.

    A stack of alternating n = 3.48 / 1.45 gratings genuinely supports a
    DIFFERENT number of propagating modes per layer -- the propagating-count
    spread is 2 and the answer is right.  Only the CONJUNCTION with a
    load-bearing cut may fire, and here the cut is nowhere near load-bearing."""
    hi, lo = (3.48 + 0j) ** 2, (1.45 + 0j) ** 2
    st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP, degree=8,
                  far_field_orders=11, layer_grids="per-layer")
    for k in range(4):
        e = hi if k % 2 == 0 else lo
        w = (340.0 - 30.0 * k) * NM
        st.add_layer(120 * NM, segments=[(0.5 - 0.5 * w / PERIOD, 1.0 + 0j),
                                         (w / PERIOD, e),
                                         (0.5 - 0.5 * w / PERIOD, 1.0 + 0j)])
    PC._MODE_CUT_CENSUS = []
    try:
        (_r0, close), warns = _t34_warnings(
            lambda: solve0(st, quiet=False))
        rows = [c for c in PC._MODE_CUT_CENSUS if c["patterned"]]
    finally:
        PC._MODE_CUT_CENSUS = None
    counts = [c["n_prop"] for c in rows]
    assert max(counts) - min(counts) >= 2, \
        "the control must actually exercise a legitimate count spread"
    assert close < 1e-5
    assert not warns, "a spread-only bar would refuse this correct stack"


def test_t34_guard_ships_disarmed_and_arming_it_moves_no_number():
    """Two claims.

    1. The SHIPPED default is silence: a user who does nothing sees no T3-4
       warning even on a cell the guard would flag.
    2. Arming it changes nothing but the warning -- the returned number is
       ``==``, not merely close.  The guard only speaks."""
    assert PC.PMM_MODE_CUT_GUARD is False,         "T3-4 ships DISARMED; see the refutation in _core.py"
    (r_off, _c), warns_off = _t34_warnings(
        lambda: solve0(build(2, degree=12), quiet=False))
    assert not warns_off
    PC.PMM_MODE_CUT_GUARD = True
    try:
        (r_on, _c2), warns_on = _t34_warnings(
            lambda: solve0(build(2, degree=12), quiet=False))
    finally:
        PC.PMM_MODE_CUT_GUARD = False
    assert warns_on
    assert r_on == r_off


#: The conical (phi != 0) false-positive family.  ``0.133532`` is the
#: stationary plateau the whole ladder sits on (measured 0.133475 .. 0.133579
#: over degrees 10..18 on every build, i.e. 0.06 % wide).
_CONICAL_PLATEAU = 0.133532
_CONICAL_FAMILY = (10, 12, 14)
_CONICAL_WIDEN = (16, 18)
_CONICAL_SCAN = {}


def _conical_cell(degree):
    """One rung of the conical false-positive family, ARMED, cached.

    ``dict(degree, r0, close, fired, n_grow, spread, n_risky, margin)`` -- both
    verdict channels read straight off ``_MODE_CUT_CENSUS`` so the adjudication
    below is a measurement and not an inference."""
    if degree in _CONICAL_SCAN:
        return _CONICAL_SCAN[degree]
    st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP,
                  degree=degree, elements_per_region=1, far_field_orders=7,
                  layer_grids="per-layer", min_feature=3.0 * NM)
    for t, segs in _taper(4):
        st.add_layer(t, segments=segs)
    old_guard, PC.PMM_MODE_CUT_GUARD = PC.PMM_MODE_CUT_GUARD, True
    old_census, PC._MODE_CUT_CENSUS = PC._MODE_CUT_CENSUS, []
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message=".*_pmm_union_grid.*")
            o, R, T, _J = st.set_source(WL, theta=THETA,
                                        phi=np.deg2rad(20.0)).solve()
        rows = list(PC._MODE_CUT_CENSUS)
    finally:
        PC._MODE_CUT_CENSUS = old_census
        PC.PMM_MODE_CUT_GUARD = old_guard
    m0 = int(np.where(np.asarray(o) == 0)[0][0])
    pat = [c for c in rows if c["patterned"]]
    counts = [int(c["n_prop"]) for c in pat]
    risky = [c for c in pat
             if c["n_risk"] > 0 and c["margin"] < PC._MODE_CUT_MARGIN_WARN]
    cell = dict(
        degree=degree,
        r0=float(np.real(np.asarray(R)[0, m0])),
        close=float(np.max(np.abs(np.asarray(R).sum(axis=1)
                                  + np.asarray(T).sum(axis=1) - 1.0))),
        fired=any(isinstance(w.message, PC._ModeClassificationWarning)
                  for w in caught),
        n_grow=sum(int(c["n_grow"]) for c in rows),          # channel A
        spread=(max(counts) - min(counts)) if len(counts) >= 2 else 0,
        n_risky=len(risky),                                  # channel B
        margin=min([c["margin"] for c in risky], default=float("inf")))
    _CONICAL_SCAN[degree] = cell
    return cell


def _conical_table(degrees):
    return "\n    ".join(
        "deg %2d  r0=%.6f rel=%+.2e close=%.1e  A:n_grow=%d  "
        "B:spread=%d n_risky=%d margin=%.3g  -> %s"
        % (c["degree"], c["r0"], c["r0"] / _CONICAL_PLATEAU - 1.0, c["close"],
           c["n_grow"], c["spread"], c["n_risky"], c["margin"],
           "FIRES" if c["fired"] else "quiet")
        for c in (_conical_cell(d) for d in degrees))


@pytest.mark.parametrize("degree", [10, 12, 14])
def test_t34_bar_is_REFUTED_on_the_conical_family(degree, t34_armed):
    """THE REFUTATION, pinned -- the reason the guard ships disarmed.

    On the CONICAL (phi != 0) cascade at ``min_feature`` = 3 nm the degree
    ladder is stationary to 0.52 % and the answers are right, yet the
    conjunction fires at degree 10/12/14: those cells read spread 1-2,
    ``n_risk`` 2 and margin 1.06-1.40, i.e. INSIDE the band the classical
    family's broken cells occupy.  No bar on these instruments survives both
    mounts.

    A future fix should make this test FAIL.

    **v5.33.1 -- WHICH RUNG CARRIES THE FALSE POSITIVE IS A PER-BUILD FACT,
    AND THE TEST SAID SO ITSELF.**  Degree 10 fires on Windows and WSL at 1, 2
    and N BLAS threads (channel A ``n_grow`` = 1-2 on the third patterned row,
    channel B ``spread`` = 1 with ``margin`` = 1.075) and on ubuntu CI with
    numpy 2.2.6; it goes QUIET on the ubuntu CI images carrying numpy 2.4.6,
    where degrees 12 and 14 still fire.  That is the same round-off-flux mode
    landing on the other side of the cut that the four-name adjudication (S3,
    S4) measured moving the classical family's wrong-cell list.

    So the rung is a STARTING POINT and the refutation is a FAMILY claim: this
    rung must be RIGHT (all of them are, on the plateau), and the false
    positive must reproduce SOMEWHERE in the family -- widened before it is
    ever given up.  If it reproduces nowhere, the test FAILS with the
    adjudication printed, because the FP is the entire reason the guard ships
    disarmed and its disappearance is a headline event, not a pass: the
    message separates "channel B's numerics shifted" (rungs still read
    at-risk modes inside the cut's decade, or growing forward modes, but the
    verdict stayed quiet) from "the conical family is clean on this build"
    (every rung reads ``n_grow`` = 0 and no at-risk mode at all)."""
    cell = _conical_cell(degree)
    # the cell IS correct: it sits on the stationary plateau (0.1335 +/- 0.1%)
    assert abs(cell["r0"] / _CONICAL_PLATEAU - 1.0) < 0.01, cell["r0"]
    # ... and the plateau is measured here, not asserted from the audit: the
    # whole family agrees with this rung far better than the 1 % bar above.
    fam = [_conical_cell(d) for d in _CONICAL_FAMILY]
    assert max(abs(c["r0"] / cell["r0"] - 1.0) for c in fam) < 2e-3, (
        "the conical ladder is no longer stationary, so 'the answers are "
        "right' is not established on this build.\n    "
        + _conical_table(_CONICAL_FAMILY))

    if cell["fired"]:
        return                              # the FP reproduces on this rung
    scanned = list(_CONICAL_FAMILY) + list(_CONICAL_WIDEN)
    elsewhere = [c for c in (_conical_cell(d) for d in scanned) if c["fired"]]
    if elsewhere:
        print(f"\nT3-4 conical FP: degree {degree} is QUIET on this build "
              f"(n_grow={cell['n_grow']}, spread={cell['spread']}, "
              f"n_risky={cell['n_risky']}, margin={cell['margin']:.3g}); the "
              f"refutation reproduces at degree(s) "
              f"{[c['degree'] for c in elsewhere]} instead.\n    "
              + _conical_table(scanned))
        return
    # NOWHERE.  Adjudicate before failing -- the two cases need different fixes.
    near = [c for c in (_conical_cell(d) for d in scanned)
            if c["n_grow"] > 0 or c["n_risky"] > 0]
    why = ("channel B's NUMERICS SHIFTED: rungs "
           f"{[c['degree'] for c in near]} still carry growing-forward or "
           "at-risk modes inside the cut's decade, but the verdict no longer "
           "fires on them -- re-pin the refutation against whatever changed "
           "the verdict") if near else (
        "the CONICAL FAMILY IS CLEAN on this build: every scanned rung reads "
        "n_grow = 0 with no at-risk mode at all, so the false positive that "
        "disarmed the guard is gone here and the DISARMED default should be "
        "re-examined against this build")
    assert False, (
        "the conical FALSE POSITIVE stopped reproducing -- if the instrument "
        "was improved, re-pin this against the fix, do not delete it.  "
        + why + ".\n    " + _conical_table(scanned))


def test_t34_guard_warnings_come_out_in_WAVELENGTH_ORDER_on_the_sweep(
        t34_armed):
    """``solve_vs_wavelength`` promises warnings in wavelength order for ANY
    worker count -- that is a shipped contract of the method, and the T3-4
    verdict is taken inside a WORKER THREAD.  It is therefore DEFERRED and
    replayed in index order.

    Pinned: the guard fires; the messages are wavelength-ordered; and the
    serial and 4-worker runs produce the SAME messages in the SAME order and
    the SAME numbers.  (The first draft of this test forgot ``theta=`` on the
    sweep -- ``solve_vs_wavelength`` takes its own, it does NOT inherit
    ``set_source``'s -- and silently measured normal incidence.)"""
    wl = np.linspace(1250.0, 1370.0, 6) * NM

    def run(workers):
        st = build(2, degree=12)                  # a device that trips T3-4
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warnings.filterwarnings("ignore", message=".*_pmm_union_grid.*")
            st.set_source(float(wl[0]), theta=THETA)
            o, R, T = st.solve_vs_wavelength(wl, theta=THETA,
                                             max_workers=workers)
        return ([str(w.message) for w in caught
                 if isinstance(w.message, PC._ModeClassificationWarning)],
                np.asarray(R), np.asarray(T))

    m1, R1, T1 = run(1)
    m4, R4, T4 = run(4)
    assert m1, "the fixture must actually trip the guard"
    got = [float(m.split("wl=")[1].split(")")[0]) for m in m1]
    assert got == sorted(got), f"warnings out of wavelength order: {got}"
    assert m4 == m1, "worker count changed the warning stream"
    assert maxdiff(R1, R4) == 0.0 and maxdiff(T1, T4) == 0.0


def test_t34_instrument_is_free_and_the_default_path_is_unarmed():
    """The census is off by default and the accumulator is not allocated
    unless something asked for it."""
    assert PC._MODE_CUT_CENSUS is None
    # DISARMED and no census: the scope is a no-op and no rows are allocated
    with PC._mode_cut_scope("probe"):
        assert getattr(PC._MODE_CUT_TLS, "rows", None) is None
    # census armed: the scope opens and closes cleanly
    PC._MODE_CUT_CENSUS = []
    try:
        with PC._mode_cut_scope("probe"):
            assert getattr(PC._MODE_CUT_TLS, "rows", None) is not None
        assert getattr(PC._MODE_CUT_TLS, "rows", None) is None
    finally:
        PC._MODE_CUT_CENSUS = None
    # margin is +inf when no mode's two direction criteria disagree
    flux = np.array([1.0, -1.0, 2.0])
    q = np.array([1j, -1j, 2j])           # signs agree with flux everywhere
    n_risk, margin = PC._mode_cut_margin(flux, q, 1e-3)
    assert n_risk == 0 and margin == float("inf")
    # and finite when they do
    q2 = np.array([1j, 1j, 2j])           # mode 1 now disagrees
    n_risk2, margin2 = PC._mode_cut_margin(flux, q2, 1.0)
    assert n_risk2 == 1 and np.isfinite(margin2)


def test_mass_flux_cut_is_unchanged_by_the_threshold_split():
    """``_mass_flux_cut`` is the only policy owner; splitting the threshold out
    for the instrument must not have moved it."""
    rng = np.random.default_rng(2)
    for n in (4, 16):
        flux = rng.standard_normal(2 * n) * 10.0 ** rng.integers(-12, 2, 2 * n)
        W2 = rng.standard_normal((2 * n, 2 * n))
        SVt = rng.standard_normal((n, 2 * n))
        SVb = rng.standard_normal((n, 2 * n))
        thr = PC._mass_flux_threshold(flux, W2, SVt, SVb, n)
        assert np.array_equal(PC._mass_flux_cut(flux, W2, SVt, SVb, n),
                              np.abs(flux) > thr)


# ===========================================================================
# M4 REFERRAL -- the per-worker BLAS-cap defect at the three PMM sites
# ===========================================================================
# M4 measured, on the RCWA twin, that threadpoolctl's cap is PROCESS-GLOBAL on
# OpenBLAS while only the REQUEST is thread-local -- so entering
# ``_blas_threads_quiet(blas_per_worker)`` inside each worker puts N
# concurrent enter/exit pairs on one global setting and the shipped
# "BYTE-IDENTICAL for any worker count" contract was false.  Three source
# comments in the library asserted thread-locality; all three were wrong.  The
# identical pattern was present at ``PMMStack.solve_vs_wavelength`` (both grid
# paths) and ``PMM2DStackHybrid.solve_vs_wavelength``; M4's fix transfers
# verbatim and is applied here.  See PMM_M4_HYGIENE_2026_08_04.md S2 and S2.6.

def _pmm_sweep_stack(grids):
    st = PMMStack(PERIOD, n_substrate=N_SUB, n_superstrate=N_SUP, degree=6,
                  far_field_orders=5, layer_grids=grids,
                  min_feature=(3.0 * NM if grids == "per-layer" else None))
    for w in (340.0, 300.0):
        hw = 0.5 * w / 700.0
        st.add_layer(90 * NM, segments=[(0.5 - hw, 1.0 + 0j),
                                        (2 * hw, 12.1 + 0j),
                                        (0.5 - hw, 1.0 + 0j)])
    return st


@pytest.mark.parametrize("grids", ["shared", "per-layer"])
def test_pmm_sweep_applies_exactly_one_blas_cap(grids):
    """The BUILD-INDEPENDENT pin on the referred fix, mirroring M4's.

    The race needs BOTH a >1-thread environment pool AND ``threadpoolctl``, so
    a byte-identity assertion is green on the 2-core CI runner and on the WSL
    build whether or not the bug is present.  COUNTING cap applications works
    on any machine.

    Exactly 1 on the SERIAL branch too is the load-bearing half: the serial
    loop runs on the caller's thread, where the sweep-level request is set, so
    without the explicit ``_blas_threads_quiet(None)`` inside ``_solve_one`` it
    nests one enter/exit per wavelength inside the sweep-level one.

    The counting stand-in DELEGATES to the real controller (M4's lesson: an
    instrument that switches off the thing it instruments let every worker run
    BLAS at the full environment pool and took down a pytest-xdist worker)."""
    import contextlib

    from lumenairy.elements.rcwa import _core as _rc_core
    real = _rc_core._get_blas_controller()
    calls = []

    class _CountingController:
        def limit(self, **kw):
            calls.append(kw)
            return real.limit(**kw) if real is not None \
                else contextlib.nullcontext()

    wls = np.linspace(1290.0, 1330.0, 3) * NM
    prev = _rc_core._get_blas_controller
    try:
        _rc_core._get_blas_controller = lambda: _CountingController()
        for nw in (1, 2, 4):
            calls.clear()
            st = _pmm_sweep_stack(grids)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.set_source(float(wls[0]), theta=THETA)
                st.solve_vs_wavelength(wls, theta=THETA, max_workers=nw)
            assert len(calls) == 1, (
                f"{grids}, max_workers={nw}: BLAS cap applied {len(calls)} "
                f"times, expected exactly 1 (once around the whole sweep).  "
                f"More than one means concurrent enter/exit pairs on a "
                f"PROCESS-GLOBAL setting -- the byte-identity race.")
            assert calls[0]["user_api"] == "blas"
    finally:
        _rc_core._get_blas_controller = prev


def test_pmm2d_hybrid_sweep_applies_exactly_one_blas_cap():
    """The third referred site, ``PMM2DStackHybrid.solve_vs_wavelength``."""
    import contextlib

    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    from lumenairy.elements.rcwa import _core as _rc_core
    real = _rc_core._get_blas_controller()
    calls = []

    class _CountingController:
        def limit(self, **kw):
            calls.append(kw)
            return real.limit(**kw) if real is not None \
                else contextlib.nullcontext()

    n = 12
    x = (np.arange(n) + 0.5) / n
    xx, yy = np.meshgrid(x, x, indexing="ij")
    cell = np.where((np.abs(xx - 0.5) < 0.22) & (np.abs(yy - 0.5) < 0.22),
                    6.25 + 0j, 1.0 + 0j)
    wls = np.linspace(1290.0, 1330.0, 3) * NM
    prev = _rc_core._get_blas_controller
    try:
        _rc_core._get_blas_controller = lambda: _CountingController()
        for nw in (1, 2, 4):
            calls.clear()
            st = PMM2DStackHybrid(PERIOD, PERIOD, n_substrate=1.5,
                                  n_superstrate=1.0, n_orders=1, degree=5)
            st.add_layer(100 * NM, eps_cell=cell)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                st.solve_vs_wavelength(wls, theta=0.0, max_workers=nw)
            assert len(calls) == 1, (
                f"2-D hybrid, max_workers={nw}: BLAS cap applied "
                f"{len(calls)} times, expected exactly 1.")
    finally:
        _rc_core._get_blas_controller = prev


@pytest.mark.parametrize("grids", ["shared", "per-layer"])
def test_pmm_sweep_is_byte_identical_across_worker_counts_repeatedly(grids):
    """The null control, run REPEATEDLY: the pre-fix race fired
    nondeterministically (M4 measured 4 failures in 6 runs on the RCWA twin),
    so one comparison is not evidence.  Tolerance at 0.0."""
    wls = np.linspace(1290.0, 1330.0, 6) * NM

    def run(nw):
        st = _pmm_sweep_stack(grids)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st.set_source(float(wls[0]), theta=THETA)
            return st.solve_vs_wavelength(wls, theta=THETA, jones=True,
                                          max_workers=nw)
    ref = run(1)
    for _rep in range(3):
        for nw in (2, 4, 8):
            got = run(nw)
            for k in (1, 2, 3):
                assert maxdiff(ref[k], got[k]) == 0.0, (grids, nw, k)


# ===========================================================================
# M4 REFERRAL (2) -- pmm_jones_2d manufactures energy on a LOSSLESS cell
# ===========================================================================
# M4 measured that the NumPy in-plane route of ``pmm_jones_2d`` draws a
# BLAS-thread-count-dependent answer and violates lossless closure by up to
# 1.5 % at degree 11, invisible to every per-order and Jones bar in the suite,
# and referred it here with "M1's census is one file short of covering it".
#
# MEASURED HERE, AND THE REFERRAL'S PREMISE IS WRONG IN THE USEFUL DIRECTION:
# the census already covers this route -- the solve routes FOUR explicit
# inverses through ``_guarded_inverse`` (two ``pmm interface mode-match (a+b)``
# and the two ``rcwa Redheffer star`` denominators) -- and its instruments
# track the defect precisely.  At degree 11 on this cell:
#
#   threads   R+T (target 2)   worst equilibrated rcond   residuals paid
#   1         2.0125077        2.754e-09                  9.825e-10
#   24        2.0307687        3.831e-13                  9.709e-06, 2.241e-08
#
# M1's screen is 1e-8 and its (withdrawn) refusal bar is 1e-8 on the
# equilibrated residual.  On WINDOWS the 1-thread solve is screened in and
# PASSES the residual while the 24-thread solve, which manufactures a further
# 1.8 % of energy, FAILS it by three orders -- the instrument separates those
# two exactly.  On WSL the SAME degree-11 solve returns the SAME wrong answer
# (2.0124976 against 2.0125077) with rcond 1.776e-06, i.e. ABOVE the screen.
#
# So the defect is build-reproducible and the instrument's READING is not.
# That is a stronger statement of M1's own conclusion: there is no bar here,
# and this document's job is to record the reproducer, not to invent one.
# This is the X-1 class, still open, now with a SECOND reproducer and the
# first one where the wrong answer is a LOSSLESS-energy violation rather than
# a per-order error.
#
# A future fix should make this test FAIL.

_J2D_P, _J2D_WL, _J2D_DEP = 0.5e-6, 0.55e-6, 0.40e-6


def _j2d_cell():
    from lumenairy.elements.pmm import pmm_jones_2d  # noqa: F401
    lay = np.zeros((4, 4), dtype=np.int64)
    lay[:2, :2] = 1
    eo, ee = 1.5 ** 2, 1.9 ** 2
    D = np.diag(np.asarray([ee, eo, eo], dtype=np.complex128))
    ca, sa = np.cos(np.deg2rad(25.0)), np.sin(np.deg2rad(25.0))
    Rz = np.asarray([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.complex128)
    cell = np.zeros(lay.shape + (3, 3), dtype=np.complex128)
    cell[lay == 0] = np.diag(np.asarray([1.5 ** 2] * 3, dtype=np.complex128))
    cell[lay == 1] = Rz @ D @ Rz.T
    return cell


def test_m4_referral_pmm_jones_2d_energy_defect_IS_seen_by_the_m1_census():
    """The defect reproduces, and M1's census covers the route."""
    from lumenairy.elements.pmm import pmm_jones_2d

    cell = _j2d_cell()
    _rc._INV_CENSUS = []
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o, R, T, _J = pmm_jones_2d(
                _J2D_P, _J2D_P, cell, 1.5, 1.0, _J2D_DEP, _J2D_WL,
                theta=np.deg2rad(30.0), phi=np.deg2rad(40.0), degree=11,
                n_orders=3)
        rows = list(_rc._INV_CENSUS)
    finally:
        _rc._INV_CENSUS = None

    total = float(np.asarray(R).sum() + np.asarray(T).sum())
    # (1) the DEFECT: a lossless cell must close at 2.0 (one per polarization)
    assert abs(total - 2.0) > 5e-3, (
        f"R+T = {total!r}: the referred energy defect stopped reproducing.  "
        f"If it was FIXED, re-pin this against the fix -- do not delete it.")

    # (2) the census SEES this route (the referral said it did not)
    sites = [r[0] for r in rows]
    assert sum(1 for s in sites if s == "pmm interface mode-match (a+b)") == 2
    assert any("Redheffer star" in s for s in sites)

    # (3) the instrument is LIVE on every one of those sites.
    #
    # NO BAR IS ASSERTED ON ITS VALUE, and that is the finding, not a
    # weakening.  Measured on this exact cell at degree 11: the DEFECT is
    # build-reproducible to five digits (Windows 2.0125077 at 1 BLAS thread,
    # WSL 2.0124976) while the equilibrated rcond of the same operator reads
    # 2.754e-09 on Windows and 1.776e-06 on WSL -- THREE ORDERS apart, on
    # opposite sides of M1's 1e-8 screen, for the same wrong answer.  (At
    # degree 9 WSL raises _EnergyError where Windows returns a number.)
    #
    # So this is not merely "correct solves read inside the broken band"
    # (M1's reason for withdrawing the refusal) -- the SAME wrong solve reads
    # on opposite sides of the bar on two builds.  A first draft of this test
    # asserted `worst < _INV_RCOND_SCREEN`; it passed on Windows and failed on
    # WSL, which is exactly the absolute-bar-on-a-BLAS-dependent-magnitude
    # trap the campaign's rule 6 names.  What is pinned instead is that the
    # instrument RUNS and returns finite readings.
    rc = [r[2] for r in rows]
    assert all(np.isfinite(x) for x in rc), rc
    assert all(0.0 < x < 1.0 for x in rc), rc

    # (4) nothing was refused: X-1 is OPEN, not closed (M1's withdrawal)
    assert all(r[4] is False for r in rows)
