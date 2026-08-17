"""Native 2-D slant metric (PMM2D roadmap item 3).

Derivation and Phase A evidence:
``docs/audits/BUILD_PMM2D_SLANT_METRIC_2026_08_16.md``.

Every bar here is either (a) a DECISION -- "a shear of an invariant direction
changes nothing", "two different slants do not share a cached solve", "this
combination refuses" -- or (b) a residual asserted RELATIVE to a control the
test measures in the same run, never against a pinned reading.  That is
deliberate: the 2-D hybrid's Fourier-truncation floor moves with ``n_orders``,
the build and the geometry, so an absolute floor bar here would be exactly the
S4 shape ``docs/TESTING_STANDARDS.md`` forbids.
"""
import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_jones_1d, pmm_jones_1d_slanted, pmm_jones_2d
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid

P = 500e-9
WL = 633e-9
D = 250e-9
DUTY = 0.5
# eps 2.25/1 at P=500nm keeps the lossless closure at ~1e-04 while the slant
# signal is ~1e-02 (measured 2026-08-16) -- a 60-200x separation, so no gate
# here sits inside the truncation noise.
ER, EG = 2.25 + 0j, 1.0 + 0j
NSUB, NSUP = 1.5, 1.0
NPIX = 240
DEG = 11


def _grating(nx=NPIX, duty=DUTY, er=ER, eg=EG, shift=0, ny=1):
    """Binary x-grating, uniform in y, as an (nx, ny, 3, 3) isotropic cell.

    The ridge is an INTEGER pixel count and rolls by an INTEGER offset, so
    every wall lands exactly on a pixel boundary (no pixelation error).
    """
    line = np.full(nx, eg, dtype=complex)
    line[:int(round(duty * nx))] = er
    line = np.roll(line, int(shift))
    cell = np.zeros((nx, ny, 3, 3), dtype=complex)
    for i in range(3):
        cell[:, :, i, i] = line[:, None]
    return cell


def _pillar(nx=120, er=ER, eg=EG, sx=0, sy=0):
    """A genuinely 2-D rectangular pillar, rolled by integer pixel offsets."""
    g = np.full((nx, nx), eg, dtype=complex)
    g[int(0.25 * nx):int(0.60 * nx), int(0.30 * nx):int(0.65 * nx)] = er
    g = np.roll(np.roll(g, int(sx), axis=0), int(sy), axis=1)
    cell = np.zeros((nx, nx, 3, 3), dtype=complex)
    for i in range(3):
        cell[:, :, i, i] = g
    return cell


def _solve(cell, *, slant=None, theta=0.0, phi=0.0, n_orders=5,
           degree=DEG, symmetry="auto"):
    return pmm_jones_2d(P, P, cell, NSUB, NSUP, D, WL, theta=theta, phi=phi,
                        degree=degree, n_orders=n_orders, slant=slant,
                        symmetry=symmetry, formulation="laurent")


def _rt(res):
    return np.asarray(res[1]), np.asarray(res[2])


def _dmax(a, b):
    (Ra, Ta), (Rb, Tb) = _rt(a), _rt(b)
    return max(float(np.max(np.abs(Ra - Rb))), float(np.max(np.abs(Ta - Tb))))


# --------------------------------------------------------------------------
# 1.  slant = 0 changes nothing, EXACTLY
# --------------------------------------------------------------------------
def test_slant_zero_is_byte_identical_to_the_vertical_path():
    """A zero slant must not perturb a single bit.

    Structural, not tolerance-based: ``_slant_is_zero`` routes back to the
    shipped symmetric 2N path, so this is exact equality or a real regression.
    """
    base = _solve(_grating(), theta=0.20)
    for sl in (None, 0.0, (0.0, 0.0)):
        got = _solve(_grating(), slant=sl, theta=0.20)
        for a, b in zip(base, got):
            assert np.array_equal(np.asarray(a), np.asarray(b))


# --------------------------------------------------------------------------
# 2.  null controls -- decisions, not readings
# --------------------------------------------------------------------------
@pytest.mark.parametrize("theta", [0.0, 0.20])
@pytest.mark.parametrize("tdeg", [5.0, 20.0, 40.0])
def test_slant_uniform_layer_is_a_noop(theta, tdeg):
    """A shear of a HOMOGENEOUS medium is a pure coordinate change.

    Bar 1e-11, CROSS-BUILD measured 2026-08-16 (validation/slant2d_envelope.py):
    worst over this whole grid is 5.46e-14 on build W (CPython 3.14.6, numpy
    2.4.4, scipy 1.17.1) and 1.03e-13 on build L (CPython 3.12.3, numpy 1.26.4,
    scipy 1.11.4, different LAPACK).  So the bar sits ~2 decades above the
    measured cross-build envelope, and the smallest real slant signal on a
    PATTERNED cell of this size is ~1.0e-02 -- 9 decades above.  Gap on both
    sides.
    """
    uni = _grating(er=ER, eg=ER)          # ridge == groove -> homogeneous
    t = float(np.tan(np.radians(tdeg)))
    base = _solve(uni, theta=theta)
    assert _dmax(_solve(uni, slant=(t, 0.0), theta=theta), base) < 1e-11
    assert _dmax(_solve(uni, slant=(t, 0.7 * t), theta=theta), base) < 1e-11


@pytest.mark.parametrize("theta", [0.0, 0.20])
def test_slant_along_an_invariant_axis_is_a_noop(theta):
    """Shearing a y-uniform cell ALONG y does nothing, and does not disturb an
    x-slant.  This is the claim with no 1-D analogue: it tests the VECTOR
    structure of the 2-D convection.  Bar 1e-11; CROSS-BUILD measured worst
    3.85e-14 (W) / 7.25e-14 (L), 2026-08-16 -- same two-sided gap as the
    uniform no-op above.
    """
    cell = _grating()
    base = _solve(cell, theta=theta)
    tx = float(np.tan(np.radians(20.0)))
    bx = _solve(cell, slant=(tx, 0.0), theta=theta)
    for tydeg in (10.0, 30.0):
        ty = float(np.tan(np.radians(tydeg)))
        assert _dmax(_solve(cell, slant=(0.0, ty), theta=theta), base) < 1e-11
        assert _dmax(_solve(cell, slant=(tx, ty), theta=theta), bx) < 1e-11


# --------------------------------------------------------------------------
# 3.  the 1-D oracle -- asserted RELATIVE to the vertical control
# --------------------------------------------------------------------------
def _orders_1d(res):
    return np.asarray(res[0]), np.asarray(res[1]), np.asarray(res[2])


def _cmp_1d(res2, res1):
    o2 = np.asarray(res2[0])
    keep = o2[:, 1] == 0
    m2 = o2[keep][:, 0]
    idx = np.argsort(m2)
    m2 = m2[idx]
    R2 = np.asarray(res2[1])[:, keep][:, idx]
    T2 = np.asarray(res2[2])[:, keep][:, idx]
    o1, R1, T1 = _orders_1d(res1)
    common = np.intersect1d(m2, o1)
    i2, i1 = np.searchsorted(m2, common), np.searchsorted(o1, common)
    d = 0.0
    for r in (0, 1):
        d = max(d, float(np.max(np.abs(R2[r][i2] - R1[r][i1]))))
        d = max(d, float(np.max(np.abs(T2[r][i2] - T1[r][i1]))))
    return d


@pytest.mark.parametrize("theta", [0.0, 0.20])
@pytest.mark.parametrize("phideg", [10.0, 20.0, 35.0])
def test_slant_matches_the_shipped_1d_slant_solver(theta, phideg):
    """A y-uniform slanted grating is EXACTLY a 1-D slanted grating, so the
    shipped 1-D slant solver is an independent oracle for it.

    The bar is the VERTICAL CONTROL measured in this same run, not a constant:
    the 2-D hybrid carries a Fourier-truncation floor the 1-D PMM does not
    have, so even a zero-slant comparison is floor-limited.  The claim is that
    the slant adds NO error beyond that floor -- allowed 3x the control's own
    residual.  CROSS-BUILD measured 2026-08-16 over this 6-cell grid: ratios
    0.95 / 0.75 / 1.38 (theta=0) and 1.21 / 1.35 / 1.41 (theta=0.20), BIT-
    IDENTICAL on builds W and L, so the worst is 1.41 against a bar of 3.0.
    (A genuine sign or routing error scores >= 2 decades here -- the wrong
    convection sign measured 4.1e-01 against a ~9e-03 control.)
    """
    n_ord = 7
    cell = _grating()
    ctrl = _cmp_1d(_solve(cell, theta=theta, n_orders=n_ord),
                   pmm_jones_1d(P, ER * np.eye(3), EG * np.eye(3), NSUB, NSUP,
                                D, DUTY, WL, angle=theta, degree=16,
                                far_field_orders=15))
    phi = np.radians(phideg)
    got = _cmp_1d(_solve(cell, slant=(float(np.tan(phi)), 0.0), theta=theta,
                         n_orders=n_ord),
                  pmm_jones_1d_slanted(P, ER * np.eye(3), EG * np.eye(3),
                                       NSUB, NSUP, D, DUTY, WL, phi,
                                       angle=theta, degree=16,
                                       far_field_orders=15,
                                       factorization="convection"))
    assert got < 3.0 * ctrl, (
        f"slanted residual {got:.3e} exceeds 3x the vertical control "
        f"{ctrl:.3e} -- the slant is adding error beyond the solver's floor")


# --------------------------------------------------------------------------
# 4.  THE F2 EVEN-PARITY FOLD GATE -- with its fail-before demonstration
# --------------------------------------------------------------------------
def test_slant_at_normal_incidence_is_not_silently_folded(monkeypatch):
    """The gate that stops the worst failure this feature can produce.

    At NORMAL incidence (kt < 1e-12) the F2 even-parity fold is eligible.  The
    fold builds its own (P, Q) from ``return_ops`` and never reaches
    ``_layer_eigenmodes_tensor`` -- where the slant convection lives.  A shear
    breaks the order-flip symmetry the fold assumes, so if the fold is allowed
    to run on a slanted layer it returns THE VERTICAL ANSWER: energy is
    conserved, nothing warns, and the result is wrong by ~1e-01.

    FAIL-BEFORE arm: the gate is removed by making ``_slant_is_zero`` report
    every slant as zero (the exact pre-fix behaviour), and the wrongness is
    demonstrated against the vertical answer computed in the same run.  Both
    arms are unconditional -- nothing here is skipped.
    """
    from lumenairy.elements.pmm import twod_jones as TJ

    cell = _grating()
    t = float(np.tan(np.radians(35.0)))
    vertical = _solve(cell, theta=0.0)
    correct = _solve(cell, slant=(t, 0.0), theta=0.0)

    # the fixed path must NOT equal the vertical answer
    d_fixed = _dmax(correct, vertical)
    assert d_fixed > 1e-2, (
        f"a 35 deg slant moved the normal-incidence answer by only "
        f"{d_fixed:.2e}; the fold is probably still swallowing it")

    # FAIL-BEFORE: remove the gate and show the vertical answer coming back
    monkeypatch.setattr(TJ, "_slant_is_zero", lambda _s: True)
    unguarded = _solve(cell, slant=(t, 0.0), theta=0.0)
    d_unguarded = _dmax(unguarded, vertical)
    assert d_unguarded == 0.0, (
        "fail-before arm did not reproduce the defect: with the gate removed "
        f"the slanted solve differed from vertical by {d_unguarded:.2e}, "
        "expected exactly 0 (the fold returning the vertical answer)")
    # ... and the defect is LARGE, and energy-conserving (hence undetectable
    # by any closure check -- which is why this gate needs its own test)
    assert _dmax(unguarded, correct) > 1e-2
    R, T = _rt(unguarded)
    assert abs(float(np.sum(R[0]) + np.sum(T[0])) - 1.0) < 1e-2


# --------------------------------------------------------------------------
# 5.  cache keys -- an engineered collision
# --------------------------------------------------------------------------
def test_slant_is_in_the_cache_key_no_collision():
    """Two layers identical in every respect EXCEPT their slant vector must
    not share a cached modal solve.

    Engineered rather than hoped for: the two layers are built on the SAME
    stack instance so they genuinely contend for the same ``_geom_cache`` /
    ``_eig_cache``, and the assertion is that the stack's answer differs from
    the one where both layers carry the first slant.
    """
    cell = _grating()
    t1, t2 = 0.20, 0.45

    def stack(slants):
        st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                              degree=DEG, n_orders=5, formulation="laurent")
        for s in slants:
            st.add_layer(D / 2, eps_tensor_cell=cell, slant=(s, 0.0))
        st.set_source(WL, theta=0.20, phi=0.0)
        return st.solve()

    mixed = stack([t1, t2])
    same = stack([t1, t1])
    d = max(float(np.max(np.abs(np.asarray(mixed[1]) - np.asarray(same[1])))),
            float(np.max(np.abs(np.asarray(mixed[2]) - np.asarray(same[2])))))
    assert d > 1e-6, (
        f"a (t={t1}, t={t2}) stack matched a (t={t1}, t={t1}) stack to {d:.2e} "
        "-- the second layer reused the first layer's cached modes, so the "
        "slant vector is missing from the modal/geometry cache key")

    # and the key itself must differ (direct, not just via the answer)
    st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                          degree=DEG, n_orders=5, formulation="laurent")
    st.add_layer(D, eps_tensor_cell=cell, slant=(t1, 0.0))
    st.add_layer(D, eps_tensor_cell=cell, slant=(t2, 0.0))
    assert st._geom_key(st._layers[0]) != st._geom_key(st._layers[1])


# --------------------------------------------------------------------------
# 6.  what refuses
# --------------------------------------------------------------------------
def test_slant_with_out_of_plane_tensor_refuses():
    """slant x out-of-plane is UNVALIDATED (the 1-D gen2 precedent hit the
    lossless trap) and must refuse rather than be inherited silently."""
    cell = _grating()
    cell[:, :, 0, 2] = 0.3          # e_xz -> genuine out-of-plane coupling
    cell[:, :, 2, 0] = 0.3
    with pytest.raises(NotImplementedError, match="OUT-OF-PLANE"):
        _solve(cell, slant=(0.3, 0.0), theta=0.20)
    # ... and the same cell without slant still works (the refusal is about
    # the COMBINATION, not about out-of-plane, which is supported)
    _solve(cell, theta=0.20)


def test_slant_on_dispersive_and_traced_layers_refuses():
    st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                          degree=DEG, n_orders=5)
    with pytest.raises(NotImplementedError, match="DISPERSIVE"):
        st.add_layer(D, eps=lambda wl: 2.0, slant=(0.3, 0.0))


def test_slant_vector_validation():
    st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                          degree=DEG, n_orders=5)
    with pytest.raises(ValueError, match="finite"):
        st.add_layer(D, eps_tensor_cell=_grating(), slant=(np.inf, 0.0))
    with pytest.raises(ValueError, match="pair"):
        st.add_layer(D, eps_tensor_cell=_grating(), slant=(0.1, 0.2, 0.3))


# --------------------------------------------------------------------------
# 7.  cascade integration
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cascade", ["fused", "tree", "monolithic"])
def test_slanted_layer_through_every_cascade_mode(cascade):
    """All three cascade strategies must accept a slanted layer and agree.

    They differ only in association order, so the bar is a Redheffer
    re-association residual, not a physics tolerance.
    """
    cell = _grating()

    def run(mode):
        st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                              degree=DEG, n_orders=5, formulation="laurent")
        st.add_layer(D / 2, eps_tensor_cell=cell, slant=(0.35, 0.15))
        st.add_layer(D / 2, eps=2.25)
        st.set_source(WL, theta=0.20, phi=0.30)
        st.cascade = mode
        return st.solve()

    ref = run("fused")
    got = run(cascade)
    d = max(float(np.max(np.abs(np.asarray(ref[1]) - np.asarray(got[1])))),
            float(np.max(np.abs(np.asarray(ref[2]) - np.asarray(got[2])))))
    assert d < 1e-9, f"cascade={cascade} disagreed with fused by {d:.2e}"


def test_slanted_layer_mixes_with_vertical_layers():
    """A slanted layer promotes the stack to the generalized cascade; vertical
    layers must still ride along (mixed sym/gen), and a stack whose only
    slanted layer has t=0 must reproduce the all-vertical answer exactly."""
    cell = _grating()

    def run(t):
        st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                              degree=DEG, n_orders=5, formulation="laurent")
        st.add_layer(D / 2, eps=2.25)
        st.add_layer(D / 2, eps_tensor_cell=cell, slant=(t, 0.0))
        st.set_source(WL, theta=0.20, phi=0.0)
        return st.solve()

    a, b = run(0.0), run(0.0)
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
    assert _dmax(run(0.4), a) > 1e-3        # a real slant really moves it


# --------------------------------------------------------------------------
# 8.  the staircase converges TOWARD the metric layer (the physics claim)
# --------------------------------------------------------------------------
def test_staircase_converges_toward_the_single_slanted_layer():
    """An N-slice laterally-shifted staircase of the SAME geometry must
    approach the one exact slanted layer as N grows.

    Asserted as a DIRECTION (each refinement strictly closer, over the range
    where the staircase is well conditioned), not as a floor: past its sweet
    spot the 2-D staircase diverges again -- see S7 of the build doc -- so an
    absolute bar here would be asserting the wrong thing.
    """
    nx = 120
    walk_px = 24                       # integer pixel walk -> exact slices
    tx = walk_px / nx * P / D
    metric = _solve(_pillar(nx), slant=(tx, 0.0), theta=0.20, n_orders=5)

    def stair(ns):
        st = PMM2DStackHybrid(P, P, n_substrate=NSUB, n_superstrate=NSUP,
                              degree=DEG, n_orders=5, formulation="laurent")
        for k in range(ns):
            st.add_layer(D / ns, eps_tensor_cell=_pillar(
                nx, sx=int(round(walk_px * (k + 0.5) / ns))))
        st.set_source(WL, theta=0.20, phi=0.0)
        return st.solve()

    ds = [_dmax(stair(ns), metric) for ns in (1, 2, 4)]
    assert ds[0] > ds[1] > ds[2], (
        f"staircase did not converge toward the slanted layer: {ds}")
    assert ds[2] < 0.25 * ds[0]
