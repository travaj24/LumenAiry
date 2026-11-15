"""The PMM modal selector and magnetic partner at the eigensolve's round-off.

Two defects, one root: the eigenvalues of the spectral-element modal operator
carry an ABSOLUTE round-off of order ``eps * max|q^2|``, and the spectral
radius ``max|q^2|`` grows like ``(degree^2 / (k0 h))^2`` on an element of width
``h``.

1. THE SELECTOR.  ``_forward_growth_flip``'s passive widening counted a
   forward mode as GROWING when ``Im(q) < -1e-6 |q|`` and then took its
   direction from the sign of ``Im(q)`` instead of from its flux.  On grids
   with an element below ~1.6e-5 of the wavelength the round-off imaginary part
   of a genuinely PROPAGATING half-space order crossed that bar, so a
   propagating order carrying unambiguous power was sent to the backward
   branch by the sign of round-off: O(0.1 .. 1) wrong, super-unity answers on an
   absorbing strip (R + T up to 108), a discrete wrong branch (the same number
   at every element size and degree that flips the same order),
   BLAS-dependent, never at normal incidence (the geometric operator is real
   there).  The bar now never undercuts the eigensolve's round-off
   (``_core._growth_mask``); ``_core._PMM_EIG_ROUNDOFF_BAR = False`` restores
   the pre-fix test, which is how the pre-fix readings quoted below were
   taken.

2. THE PARTNER.  ``V_top = (Cyx W_x + (Cyy - Kx2) W_y) / lam`` loses every
   digit for a TE-like mode at cut-off (catastrophic cancellation, divided by
   ``lam -> 0``); the exact relation ``V_top = W_bot q^2 / lam`` does not
   (``_core._hx_partner_rows``).

Fixture replaced by a generic one on 2026-11-15; every physics bar carries its
measurement, re-taken on it (Windows/MKL).
"""
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm._core as C
import lumenairy.elements.pmm.stack as S
from lumenairy.elements.pmm import PMMStack, pmm_jones_1d_segments

#: The strip layer of the strip / spacer / mirror stack (a generic fixture):
#: a 40 nm metal strip (eps -90 + 7i) filling 0.36 of the period, air
#: between the strips and above, n = 1.5 below, 1500 nm.
WL = 1500e-9
E_STRIP = -90.0 + 7.0j
E_MIRROR = -130.0 + 20.0j
N_SUB = 1.5
DUTY = 0.36


def _ratio_for(h, width, n_el=12):
    """Geometric ratio whose wall element is ``h`` in a region ``width``."""
    return (h / (0.5 * width)) ** (1.0 / (n_el // 2 - 1))


def _strip(P, theta, degree, h_over_wl, duty=DUTY):
    """One 40 nm metal strip layer on n = 1.5, air above, geometric layout of
    12 elements per region whose wall element is ``h_over_wl`` wavelengths."""
    W = duty * P
    g = ("geometric", _ratio_for(h_over_wl * WL, W), 1e-30)
    st = PMMStack(P, n_substrate=N_SUB, n_superstrate=1.0, degree=degree,
                  elements_per_region=12, grade=g, allow_small_elements=True)
    st.add_layer(40e-9, segments=[(duty, E_STRIP), (1.0 - duty, 1.0)])
    st.set_source(WL, theta=theta)
    return st


def _solve(st):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve()
    return np.asarray(J), R.sum(axis=1) + T.sum(axis=1)


# --------------------------------------------------------------------------- #
# 1. the growth mask (pure arithmetic)
# --------------------------------------------------------------------------- #
def test_growth_mask_does_not_undercut_the_eig_roundoff():
    """The measured offending mode (fixture replaced by a generic one;
    re-measured 2026-11-15, Windows/MKL): the -1 order in air of the 1200 nm
    strip at theta 0.3, degree 8, wall element 1.25e-5 wavelengths, in a
    modal set whose largest ``|q|`` is 4.63e5.  ``|Im q| / |q|`` = 2.1e-6 is
    past the old bar and 250x inside the round-off bound -- it is not
    growth.  (The other three entries: the largest evanescent ``|q|``, the
    +1 order in air put on the forward branch with ``Im q < 0``, and the
    zeroth order in air.)"""
    q_all = np.array([0.298275 - 6.26e-7j, 4.626e5j, -1.1784j, 0.955336])
    qf = q_all.copy()
    grow = C._growth_mask(qf, q_all)
    assert not grow[0]                   # round-off on a propagating order
    assert grow[2]                       # an evanescent mode put forward grows
    assert not grow[1] and not grow[3]
    # the historical bar alone flags the propagating order
    assert qf[0].imag < -C._MODE_GROWTH_REL * abs(qf[0])


def test_growth_mask_is_the_historical_one_where_roundoff_is_small():
    """On a modest spectrum (largest |q| 70, a default layout) the round-off
    bound sits below ``_MODE_GROWTH_REL |q|`` for every mode with |q| above
    ~1e-2, so the mask is the historical one bit for bit."""
    rng = np.random.default_rng(7)
    q = (rng.uniform(0.05, 70, 400)
         * np.exp(1j * rng.uniform(-np.pi, np.pi, 400)))
    q[:50] = q[:50].real - 1j * 1e-6 * np.abs(q[:50]) * rng.uniform(0, 3, 50)
    old = q.imag < -C._MODE_GROWTH_REL * np.abs(q)
    assert np.array_equal(C._growth_mask(q, q), old)


def test_growth_mask_traces_under_jax():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    q_all = np.array([0.298275 - 6.26e-7j, 4.626e5j, -1.1784j, 0.0 + 0.0j])
    f = jax.jit(lambda q: C._growth_mask(q, q, jnp))
    assert np.array_equal(np.asarray(f(jnp.asarray(q_all))),
                          C._growth_mask(q_all, q_all))


# --------------------------------------------------------------------------- #
# 2. the selector on the measured failing grid
# --------------------------------------------------------------------------- #
def test_power_carrying_halfspace_orders_stay_forward_on_tiny_elements():
    """Half-space modes on the failing grid (1200 nm, theta 0.3, degree 8,
    wall element 1.25e-5 wavelengths): every propagating order must be put on
    the FORWARD branch (``Re q > 0``) whatever the sign of its round-off
    ``Im q``.  With the pre-fix selector restored the -1 order in air comes
    back as ``q = -0.298275 + 6.3e-7i`` on Windows/MKL (re-measured
    2026-11-15 on the generic fixture)."""
    P = 1200e-9
    W = DUTY * P
    g = C._normalize_grade(("geometric", _ratio_for(1.25e-5 * WL, W), 1e-30),
                           period=P, where="test")
    tens = [C._tensor3_dict(np.eye(3))] * 2
    mats = C._build_sem_tensor_segments(P, [DUTY, 1.0 - DUTY], tens, 8, 12, g)
    k0 = 2 * np.pi / WL
    kx0 = np.sin(0.3) * k0
    geo = C._uniform_geo_eig(mats, k0, kx0)
    for eps in (1.0, N_SUB ** 2):
        _W, _V, _lam, q = C._sem_modes_uniform(mats, k0, kx0, eps, geo)
        prop = np.abs(q.imag) < 1e-3 * np.abs(q)
        assert prop.sum() >= 2
        assert np.all(q[prop].real > 0.0), q[prop]


#: |J - J(wall element 1e-4 wavelengths)| at the formerly failing
#: configurations (same degree).  Fixture replaced by a generic one;
#: RE-MEASURED 2026-11-15 (Windows/MKL): <= 3.4e-6 after the fix (the
#: round-off floor of these element sizes); with the pre-fix selector
#: restored (``_core._PMM_EIG_ROUNDOFF_BAR = False``, energy backstop off)
#: the rows read 1.27 / 1.27 / 0.225 / 0.168 / 3.04 / 4.87 / 7.27 / 4.11 off
#: with R + T = 20.6 / 20.6 / 1.62 / 1.39 / 19.4 / 47.6 / 108 / 84.2.
#: Bar 5e-5: 15x above the readings, 3.5 decades below the smallest pre-fix
#: error.
FORMER_FAILURE_BAR = 5e-5

#: (period, theta, degree, wall element / wavelength): rows that return
#: super-unity wrong answers with the pre-fix selector restored -- 8 of the
#: 81 such rows of a 720-solve oblique sweep (periods 250 / 500 / 800 /
#: 1200 nm, theta 0.15-0.7, degrees 6-20, wall 2e-6 .. 3e-5 wavelengths;
#: none at a wall of 1.6e-5 wavelengths or above).  Which rows fail is a
#: lottery over code, degree and build (the sign of round-off).
_FORMER_FAILURES = [
    (1200e-9, 0.3, 8, 1.25e-5), (1200e-9, 0.3, 14, 5.5e-6),
    (1200e-9, 0.15, 6, 1e-5), (800e-9, 0.5, 20, 5.5e-6),
    (800e-9, 0.7, 10, 4e-6), (500e-9, 0.7, 10, 4e-6),
    (250e-9, 0.7, 6, 4e-6), (1200e-9, 0.5, 6, 4e-6),
]


@pytest.mark.parametrize("P,theta,deg,hw", _FORMER_FAILURES)
def test_former_failures_return_the_reference(P, theta, deg, hw):
    J, tot = _solve(_strip(P, theta, deg, hw))
    Jr, tot_r = _solve(_strip(P, theta, deg, 1e-4))
    assert float(np.max(tot)) <= 1.0, tot          # absorbing: R + T <= 1
    d = float(np.max(np.abs(J - Jr)))
    assert d < FORMER_FAILURE_BAR, d


def test_no_super_unity_in_a_representative_scan():
    """The strip absorbs, so ``R + T <= 1`` is a theorem.  A grid across the
    whole former failure band (wall element 3e-6 .. 2e-5 wavelengths, oblique,
    two periods, two degrees).  RE-MEASURED 2026-11-15 on the generic
    fixture: max R + T over the full post-fix sweep (900 Windows/MKL solves,
    periods 250 / 500 / 800 / 1200 nm, theta 0-0.7, degrees 6-20, wall
    2e-6 .. 3e-5 wavelengths) 0.99824, every row sub-unity; this grid reads
    0.99297."""
    worst = 0.0
    for P in (800e-9, 1200e-9):
        for theta in (0.15, 0.5):
            for deg in (8, 14):
                for hw in (3e-6, 1.6e-5):
                    _J, tot = _solve(_strip(P, theta, deg, hw))
                    worst = max(worst, float(np.max(tot)))
    assert worst <= 1.0, worst


# --------------------------------------------------------------------------- #
# 3. the single-layer tensor entry point shares the selector
# --------------------------------------------------------------------------- #
def test_single_layer_jones_on_tiny_elements():
    """``pmm_jones_1d_segments`` (the tensor path with the shared-eig
    half-spaces) on a formerly failing grid matches its own 1e-4-wavelength
    layout.  Bar as ``FORMER_FAILURE_BAR``.  RE-MEASURED 2026-11-15
    (Windows/MKL): 5.7e-7; 1.27 off with R + T = 20.6 with the pre-fix
    selector restored."""
    P, theta, deg = 1200e-9, 0.3, 8
    W = DUTY * P
    segs = [(DUTY, E_STRIP), (1.0 - DUTY, 1.0)]

    def run(hw):
        g = ("geometric", _ratio_for(hw * WL, W), 1e-30)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = pmm_jones_1d_segments(
                P, segs, N_SUB, 1.0, 40e-9, WL, theta=theta, degree=deg,
                elements_per_region=12, grade=g, stabilize=False)
        return np.asarray(J), R.sum(axis=1) + T.sum(axis=1)
    J, tot = run(1.25e-5)
    Jr, _ = run(1e-4)
    assert float(np.max(tot)) <= 1.0, tot
    assert float(np.max(np.abs(J - Jr))) < FORMER_FAILURE_BAR


# --------------------------------------------------------------------------- #
# 4. the exact H_x partner
# --------------------------------------------------------------------------- #
def test_hx_partner_is_the_q_route_where_that_route_is_accurate():
    """On a default (Chebyshev, 3 elements) patterned layer the new ``H_x``
    rows equal the ``Q``-route rows to round-off for every mode away from
    cut-off -- the same quantity by an exact identity, not a model change.
    RE-MEASURED 2026-11-15 on the generic strip: max relative difference
    7.0e-12 (|q| > 0.1, Windows/MKL) -- the Q-route's own cancellation error
    on this spectrum."""
    P = 500e-9
    tens = [C._tensor3_dict(E_STRIP * np.eye(3)), C._tensor3_dict(np.eye(3))]
    mats = C._build_sem_tensor_segments(P, [DUTY, 1.0 - DUTY], tens, 10, 3,
                                        True)
    k0 = 2 * np.pi / WL
    kx0 = np.sin(0.3) * k0
    W2, V2, lam, q = C._sem_modes_tensor(mats, k0, kx0, True)
    n = mats["n_glob"]
    # rebuild the Q-route top rows from the same operators
    S0 = mats["S0"]
    iS0 = np.diag(1.0 / np.real(np.diag(S0)))
    mass, stiff, conv = mats["mass"], mats["stiff"], mats["conv"]
    Cinv_xx = iS0 @ mass["inv_xx"]
    Cxx = np.linalg.inv(Cinv_xx)
    Cyx = (iS0 @ mass["eyx_xx"]) @ Cxx
    Cyy = iS0 @ mass["schur"] + (iS0 @ mass["eyx_xx"]) @ Cxx @ (iS0 @ mass["exy_xx"])
    Cw = conv["one"]
    Kx2 = (iS0 @ (stiff["one"] - 1j * kx0 * (Cw - Cw.T)
                  + kx0 ** 2 * mass["one"])) / k0 ** 2
    Vq = (Cyx @ W2[:n] + (Cyy - Kx2) @ W2[n:]) / lam[None, :]
    far = np.abs(q) > 0.1
    num = np.max(np.abs(V2[:n, far] - Vq[:, far]), axis=0)
    den = np.max(np.abs(V2[:n, far]), axis=0) + np.max(np.abs(V2[n:, far]),
                                                       axis=0)
    assert float(np.max(num / den)) < 1e-10


def _spacer_stack(degree, theta):
    """A uniform air spacer whose -1 order is exactly at cut-off inside it:
    n_sup 1.2, 1000 nm period, sin(theta) = (wl / P - 1) / 1.2."""
    st = PMMStack(1000e-9, n_substrate=N_SUB, n_superstrate=1.2,
                  degree=degree, elements_per_region=2)
    st.add_layer(40e-9, segments=[(0.4, E_STRIP), (0.6, 1.0)])
    st.add_layer(120e-9, eps=1.0)
    st.add_layer(30e-9, eps=2.25)
    st.set_source(WL, theta=theta)
    return st


#: Dense per-layer eig vs the structured uniform-layer solve, the -1 order at
#: cut-off inside a uniform spacer.  Fixture replaced by a generic one;
#: RE-MEASURED 2026-11-15 (Windows/MKL): max |dJ| 2.5e-13 at degree 10 and
#: 14; the code before the fix (the package at the parent of the fix,
#: same build) reads TE R + T = 1.365 / 1.332 on the dense path at degree
#: 10 / 14, 0.10 / 0.14 off the structured solve.  Bar 1e-7: 4e5 x above
#: the reading, 6 decades below the pre-fix error.
CUTOFF_DENSE_BAR = 1e-7


@pytest.mark.parametrize("deg", [10, 14])
def test_dense_path_at_a_cutoff_matches_the_structured_solve(deg, monkeypatch):
    th = float(np.arcsin((WL / 1000e-9 - 1.0) / 1.2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", True)
        fast = _spacer_stack(deg, th).solve()
        monkeypatch.setattr(S, "_UNIFORM_FAST_PATH", False)
        dense = _spacer_stack(deg, th).solve()
    tot = dense[1].sum(axis=1) + dense[2].sum(axis=1)
    assert float(np.max(tot)) <= 1.0, tot
    d = float(np.max(np.abs(np.asarray(fast[3]) - np.asarray(dense[3]))))
    assert d < CUTOFF_DENSE_BAR, d


# --------------------------------------------------------------------------- #
# 5. geometric grading on multi-layer and lossless oblique stacks, against an
#    independent (single high-degree element) reference
# --------------------------------------------------------------------------- #
def _two_walls(degree, epr, grade, w2=120.0, **kw):
    """Two patterned layers with DIFFERENT walls (a 180 nm metal strip over a
    ``w2`` nm dielectric strip, 500 nm period) on a metal mirror, theta 0.2."""
    P = 500e-9
    st = PMMStack(P, n_substrate=N_SUB, n_superstrate=1.0, degree=degree,
                  elements_per_region=epr, grade=grade, **kw)
    st.add_layer(40e-9, segments=[(180 / 500, E_STRIP), (320 / 500, 1.0)])
    st.add_layer(30e-9, segments=[(w2 / 500, 2.25), (1 - w2 / 500, 1.0)])
    st.add_layer(80e-9, eps=E_MIRROR)
    st.set_source(WL, theta=0.2)
    return st


def _lamellar(degree, epr, grade, theta, P=500e-9, **kw):
    """A lossless lamellar grating (eps 4 / air, duty 0.36, 150 nm) on
    n = 1.5 at oblique incidence."""
    st = PMMStack(P, n_substrate=1.5, n_superstrate=1.0, degree=degree,
                  elements_per_region=epr, grade=grade, **kw)
    st.add_layer(150e-9, segments=[(DUTY, 4.0), (1.0 - DUTY, 1.0)])
    st.set_source(WL, theta=theta)
    return st


#: Geometric layouts vs ONE Chebyshev element at degree 60.  Fixture replaced
#: by a generic one; RE-MEASURED 2026-11-15 (Windows/MKL): two-wall stack,
#: default floor, degrees 12 / 16 / 20: <= 6.0e-6; with NO floor (wall
#: elements down to 4.3 pm) <= 9.9e-6.  The reference itself is the limit:
#: against the adaptive-resolution RCWAStack (independent; 150 and 200
#: orders agree to 3e-9) the degree-60 element is 4.5e-6 off and geometric
#: degree 12 / 16 / 20 / 24 are 0.8e-6 / 1.6e-6 / 0.4e-6 / 1.7e-6 off.  (This
#: 500 nm stack has no propagating diffraction order at theta 0.2, so the
#: pre-fix selector reads the same here; the selector rows are above.)  Bar
#: 1e-4: 10x above the readings.
MULTI_LAYER_BAR = 1e-4


@pytest.mark.parametrize("deg", [12, 16, 20])
def test_geometric_two_patterned_layers_with_different_walls(deg):
    Jr, _ = _solve(_two_walls(60, 1, True))
    J, tot = _solve(_two_walls(deg, 12, "geometric"))
    assert float(np.max(tot)) <= 1.0, tot
    assert float(np.max(np.abs(J - Jr))) < MULTI_LAYER_BAR
    Jn, totn = _solve(_two_walls(deg, 12, ("geometric", 0.17, 1e-30),
                                 allow_small_elements=True))
    assert float(np.max(totn)) <= 1.0, totn
    assert float(np.max(np.abs(Jn - Jr))) < MULTI_LAYER_BAR


def test_geometric_near_coincident_walls():
    """Walls 1 nm apart (180 / 181 nm): the union grid carries a 1 nm cell.
    With the default floor the cell's 12 elements fall back to the uniform
    layout and the answer agrees with the reference (RE-MEASURED 2026-11-15
    on the generic fixture: 3.0e-6 at degree 20, Windows/MKL).  With the
    floor disabled the cell's wall element is 7e-14 m -- far inside the
    round-off band -- and the opted-out solve must say so: it reads 0.099
    off with R + T = 0.974, a sub-unity error no energy check sees."""
    Jr, _ = _solve(_two_walls(60, 1, True, w2=181.0))
    J, tot = _solve(_two_walls(20, 12, "geometric", w2=181.0))
    assert float(np.max(np.abs(J - Jr))) < MULTI_LAYER_BAR
    st = _two_walls(20, 12, ("geometric", 0.17, 1e-30), w2=181.0)
    with pytest.raises(ValueError, match="below the measured safe size"):
        st.solve()
    st = _two_walls(20, 12, ("geometric", 0.17, 1e-30), w2=181.0,
                    allow_small_elements=True)
    with pytest.warns(UserWarning, match="allow_small_elements=True"):
        st.solve()


#: Lossless lamellar grating, geometric (default floor and no floor) vs ONE
#: Chebyshev element at degree 60, theta 0.2 and 0.4.  RE-MEASURED
#: 2026-11-15 on the generic grating (Windows/MKL): <= 6.2e-7 at degrees
#: 16 / 24, |R + T - 1| <= 2.9e-7 (and the degree-60 element agrees with the
#: adaptive-resolution RCWAStack to 1.2e-7 at theta 0.4).  Bar 2e-5: 32x
#: above.
LAMELLAR_BAR = 2e-5


@pytest.mark.parametrize("theta", [0.2, 0.4])
def test_geometric_lossless_oblique_grating(theta):
    Jr, _ = _solve(_lamellar(60, 1, True, theta))
    for deg in (16, 24):
        for g, kw in (("geometric", {}),
                      (("geometric", 0.17, 1e-30),
                       {"allow_small_elements": True})):
            J, tot = _solve(_lamellar(deg, 12, g, theta, **kw))
            assert float(np.max(np.abs(tot - 1.0))) < 1e-5, tot
            assert float(np.max(np.abs(J - Jr))) < LAMELLAR_BAR


#: Lossless rows that return WRONG answers with the pre-fix selector
#: restored (``_core._PMM_EIG_ROUNDOFF_BAR = False``, energy backstop off;
#: generic grating, re-measured 2026-11-15, Windows/MKL): (800 nm, theta
#: 0.4, degree 12, wall 8e-6 wavelengths) 0.049 off the degree-60 reference
#: with R + T = 1.103; (1000 nm, theta 0.6, degree 16, wall 4e-6) 0.152 off,
#: R + T = 2.64; (500 nm, theta 0.4, degree 16, wall 4e-6) 7.5 off, R + T =
#: 112.  (On this grating every wrong row of a 108-row sweep was
#: super-unity; an earlier fixture also produced an energy-conserving wrong
#: row -- the lossless trap -- which is why R + T alone is no witness.)
#: After the fix: <= 1.2e-6, |R + T - 1| <= 1.0e-6 (bar ``LAMELLAR_BAR``).
_LOSSLESS_FORMER = [(800e-9, 0.4, 12, 8e-6), (1000e-9, 0.6, 16, 4e-6),
                    (500e-9, 0.4, 16, 4e-6)]


@pytest.mark.parametrize("P,theta,deg,hw", _LOSSLESS_FORMER)
def test_lossless_former_failures_match_the_reference(P, theta, deg, hw):
    g = ("geometric", _ratio_for(hw * WL, DUTY * P), 1e-30)
    J, tot = _solve(_lamellar(deg, 12, g, theta, P=P,
                              allow_small_elements=True))
    Jr, _ = _solve(_lamellar(60, 1, True, theta, P=P))
    assert float(np.max(np.abs(tot - 1.0))) < 1e-5, tot
    assert float(np.max(np.abs(J - Jr))) < LAMELLAR_BAR


# --------------------------------------------------------------------------- #
# 6. the near-coincident-wall (sliver) hazard band was the same defect
# --------------------------------------------------------------------------- #
# The O-11 fixture of tests/unit/test_fix_pmmstack_sliver_walls.py: two slices
# of a taper whose walls differ by ``delta`` of the period, so the union grid
# carries a sliver element of that width.  That file draws its "hazard band"
# (rows whose answer is > 100 x the physical shift ``delta`` away from the
# delta -> 0 reference) at run time; before the selector fix the band was
# non-empty on every kernel measured.
_SL_P, _SL_WL, _SL_TH, _SL_DZ = 1.2e-6, 0.85e-6, 0.15, 0.32e-6 / 4
_SL_A0, _SL_B0 = 0.27865, 0.62505


def _sliver(delta, degree):
    st = PMMStack(_SL_P, n_superstrate=1.0, n_substrate=1.0, degree=degree,
                  min_feature=_SL_P * 1e-5)
    for a, b in ((_SL_A0, _SL_B0), (_SL_A0 - delta, _SL_B0 + delta)):
        st.add_layer(_SL_DZ, segments=[(a, 2.25), (b - a, 9.0), (1.0 - b, 2.25)])
    st.set_source(_SL_WL, theta=_SL_TH)
    was = S.PMM_SLIVER_GUARD
    S.PMM_SLIVER_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
    finally:
        S.PMM_SLIVER_GUARD = was
    i = np.argsort(np.asarray(o).ravel())
    return np.asarray(R)[:, i], np.asarray(T)[:, i]


def test_the_sliver_hazard_band_is_empty_with_the_fix():
    """Every row of the sliver ladder (degrees 12 / 14 / 20, delta 3e-4 ..
    1.5e-5 of the period) tracks the physical shift.  MEASURED 2026-11-14:
    0 wrong rows on Windows/MKL and on WSL/OpenBLAS with the fix (11 sliver
    tests that need the wrong population now restore the pre-fix selector
    through ``_core._PMM_EIG_ROUNDOFF_BAR``; with it restored the band is
    non-empty again and those tests pass)."""
    wrong = []
    for deg in (12, 14, 20):
        Rr, Tr = _sliver(0.0, deg)
        for d in np.geomspace(3e-4, 1.5e-5, 12):
            R, T = _sliver(float(d), deg)
            e = max(float(np.max(np.abs(R - Rr))), float(np.max(np.abs(T - Tr))))
            if e > 100.0 * d:
                wrong.append((deg, float(d), e))
    assert wrong == [], wrong


def test_the_sliver_guard_does_no_harm_in_the_production_configuration():
    """The sliver refusal is kept as defence in depth after the selector fix;
    its own test modules exercise it with the pre-fix selector restored.
    Here it runs as shipped (fix on, guard on): hazard-band rows of the
    ladder above must solve without a refusal and track the physical
    shift: a guard kept as defence in depth must not refuse right answers."""
    assert S.PMM_SLIVER_GUARD
    for deg in (12, 20):
        st0 = PMMStack(_SL_P, n_superstrate=1.0, n_substrate=1.0,
                       degree=deg, min_feature=_SL_P * 1e-5)
        for a, b in ((_SL_A0, _SL_B0), (_SL_A0, _SL_B0)):
            st0.add_layer(_SL_DZ, segments=[(a, 2.25), (b - a, 9.0),
                                            (1.0 - b, 2.25)])
        st0.set_source(_SL_WL, theta=_SL_TH)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o0, R0, T0, _J0 = st0.solve()
        for d in (1e-4, 3e-5):
            st = PMMStack(_SL_P, n_superstrate=1.0, n_substrate=1.0,
                          degree=deg, min_feature=_SL_P * 1e-5)
            for a, b in ((_SL_A0, _SL_B0), (_SL_A0 - d, _SL_B0 + d)):
                st.add_layer(_SL_DZ, segments=[(a, 2.25), (b - a, 9.0),
                                               (1.0 - b, 2.25)])
            st.set_source(_SL_WL, theta=_SL_TH)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, R, T, _J = st.solve()            # no refusal
            e = max(float(np.max(np.abs(R - R0))),
                    float(np.max(np.abs(T - T0))))
            assert e <= 100.0 * d, (deg, d, e)


def test_selector_switch_is_private_and_single_sourced():
    """``_core._PMM_EIG_ROUNDOFF_BAR`` is private and not re-exported (a
    re-exported copy would be dead: setting it would change nothing).  The
    modules that flip it never import JAX, so no traced computation can
    capture the pre-fix mask (the switch is read at trace time)."""
    import pathlib

    import lumenairy.elements.pmm as pmm
    import lumenairy.elements.pmm._core as core
    assert not hasattr(pmm, "PMM_EIG_ROUNDOFF_BAR")
    assert not hasattr(pmm, "_PMM_EIG_ROUNDOFF_BAR")
    assert not hasattr(core, "PMM_EIG_ROUNDOFF_BAR")
    assert core._PMM_EIG_ROUNDOFF_BAR is True
    here = pathlib.Path(__file__).parent
    for p in here.glob("test_*.py"):
        if p.name == pathlib.Path(__file__).name:
            continue
        txt = p.read_text(encoding="utf-8")
        if "_PMM_EIG_ROUNDOFF_BAR = False" in txt:
            assert "import jax" not in txt and "from jax" not in txt, p.name
