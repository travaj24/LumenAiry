"""Geometric element grading through PMMStack, and the small-element guard.

Fixture A: period 370 nm, a 175 nm wide / 50 nm thick metal strip (eps -74+6i)
in air over 1 / 7 / 5 / 7 / 1 nm dielectric and lossy-oxide films over a 70 nm
metal mirror (eps -160+30i), air above, n = 1.447 below, 1300 nm, TM = E_x.
r_ref = 0.0571572 + 0.1312207i (external reference, uncertainty ~1e-5).

Every physics bar carries its measurement (date, both BLAS builds) in a comment;
the layout-arithmetic tests are exact.
"""
import ast
import pathlib
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm._core as C
import lumenairy.elements.pmm.stack as S
from lumenairy.elements.pmm import PMMStack, pmm_jones_1d_conical

WL, P, W = 1300e-9, 370e-9, 175e-9
F = W / P
E_AU, E_AL = -74.0 + 6.0j, -160.0 + 30.0j
E_AO, E_HF, E_OX = 3.05 + 0j, 3.53 + 0j, 2.5 + 0.5j
N_SUB = 1.447
R_REF = 0.0571572 + 0.1312207j
LAYERS = [(50e-9, E_AU, 1.0 + 0j), (1e-9, E_AO, E_AO), (7e-9, E_HF, E_HF),
          (5e-9, E_OX, E_OX), (7e-9, E_HF, E_HF), (1e-9, E_AO, E_AO),
          (70e-9, E_AL, E_AL)]


def _fixture(degree, epr, grade, theta=0.0, phi=0.0, layers=LAYERS, **kw):
    st = PMMStack(P, n_substrate=N_SUB, n_superstrate=1.0, degree=degree,
                  elements_per_region=epr, grade=grade, **kw)
    for t, es, eg in layers:
        if es == eg:
            st.add_layer(t, eps=es)
        else:
            st.add_layer(t, segments=[(F, es), (1.0 - F, eg)])
    st.set_source(WL, theta=theta, phi=phi)
    return st


def _J(st):
    return np.asarray(st.solve()[3])


# --------------------------------------------------------------------------- #
# wiring: the stored spec, its validation, the clones
# --------------------------------------------------------------------------- #
def test_grade_is_normalized_and_validated():
    st = PMMStack(P, elements_per_region=12, grade="geometric")
    assert st.grade == ("geometric", C.GEOMETRIC_DEFAULT_RATIO,
                        C.GEOMETRIC_DEFAULT_MIN_ELEMENT_FRAC * P)
    assert PMMStack(P, grade="chebyshev").grade is True
    assert PMMStack(P, grade=0).grade is False
    st2 = PMMStack(P, elements_per_region=4, grade=("geometric", 0.2))
    assert st2.grade == ("geometric", 0.2, C.GEOMETRIC_DEFAULT_MIN_ELEMENT_FRAC * P)
    spec = ("geometric", 0.25, 2e-11)
    assert PMMStack(P, elements_per_region=4, grade=spec).grade == spec
    with pytest.raises(ValueError, match="unknown grade"):
        PMMStack(P, grade="geometirc")
    with pytest.raises(ValueError, match="elements_per_region >= 3"):
        PMMStack(P, elements_per_region=2, grade="geometric")
    with pytest.raises(ValueError, match="0 < ratio < 1"):
        PMMStack(P, elements_per_region=4, grade=("geometric", 1.2))
    # assignment goes through the same validation and a refusal changes nothing
    st.grade = "chebyshev"
    assert st.grade is True
    with pytest.raises(ValueError):
        st.grade = "nope"
    assert st.grade is True


def test_clones_carry_the_spec_and_the_opt_out():
    st = _fixture(6, 4, ("geometric", 0.2, 3e-11), allow_small_elements=True)
    c = st._min_feature_clone(1e-10)
    assert c.grade == st.grade and c.allow_small_elements is True
    st2 = PMMStack(P, elements_per_region=4, grade="geometric")
    st2.add_tapered_grating(60e-9, eps_ridge=2.25, eps_groove=1.0,
                            duty_bottom=0.5, duty_top=0.4, n_slices=3)
    c2 = st2._resliced_clone(1)
    assert c2.grade == st2.grade


def test_no_bool_grade_left_in_stack_module():
    path = pathlib.Path(S.__file__)
    hits = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "bool" and len(node.args) == 1):
            a = node.args[0]
            if ((isinstance(a, ast.Name) and a.id == "grade")
                    or (isinstance(a, ast.Attribute) and a.attr == "grade")):
                hits.append(node.lineno)
    assert hits == []


def test_chebyshev_alias_is_bit_identical_to_true():
    a = _J(_fixture(8, 3, True))
    b = _J(_fixture(8, 3, "chebyshev"))
    assert np.array_equal(a, b)


# --------------------------------------------------------------------------- #
# physics on fixture A
# --------------------------------------------------------------------------- #
#: Geometric (default floor: max(1e-4 x period, 4e-5 x wavelength) = 52 pm
#: here) degree 8 x 12 el vs the same layout at degree 20.  MEASURED
#: 2026-11-14: 5.90e-6 on Windows/MKL and WSL/OpenBLAS (identical to 3
#: digits; the degree-20 value is itself 7.1e-7 from r_ref).  Bar 2e-5: 3.4x
#: above the reading, 50x below the Chebyshev-at-equal-cost error (1.03e-3)
#: it must separate from.
GEO_DEG_BAR = 2e-5


def test_geometric_beats_chebyshev_at_equal_cost():
    J8 = _J(_fixture(8, 12, "geometric"))
    J20 = _J(_fixture(20, 12, "geometric"))
    Jc = _J(_fixture(10, 8, True))                 # Chebyshev, DOF 160 vs 192
    e8 = abs(J8[0, 0] - J20[0, 0])
    assert e8 < GEO_DEG_BAR, e8
    # against the external reference: geometric 6.6e-6, Chebyshev 1.03e-3
    # (MEASURED 2026-11-14, both builds) -- a >= 50x decision with 3x to
    # spare
    assert abs(Jc[0, 0] - R_REF) > 50.0 * abs(J8[0, 0] - R_REF)


#: theta = 0.3: degree 8 vs 20 at 12 geometric elements.  MEASURED
#: 2026-11-14: 5.90e-6, both builds.  Bar as for normal incidence.
GEO_OBLIQUE_BAR = 2e-5


def test_geometric_oblique_converges():
    J8 = _J(_fixture(8, 12, "geometric", theta=0.3))
    J20 = _J(_fixture(20, 12, "geometric", theta=0.3))
    d = abs(J8[0, 0] - J20[0, 0])
    assert d < GEO_OBLIQUE_BAR, d


#: TE (E_y, spectrally converging -- a correct mesh change cannot move it) of
#: geometric degree 8 x 12 el vs single-element degree 60.  MEASURED
#: 2026-11-14: 1.7e-8, both builds -- 59x below the bar, which itself sits
#: 13x below what an unclamped tiny wall does to TE (1.3e-5 at ratio 0.10,
#: single-layer measurement).
GEO_TE_BAR = 1e-6


def test_geometric_leaves_te_untouched():
    Jg = _J(_fixture(8, 12, "geometric"))
    Jr = _J(_fixture(60, 1, True))
    d = abs(Jg[1, 1] - Jr[1, 1])
    assert d < GEO_TE_BAR, d


def test_the_floor_turns_a_bad_ratio_into_a_good_layout():
    """('geometric', 0.05) at 16 elements is clamped by the default floor
    (52 pm) to a working layout: its error is within 3x of the recommended
    0.17 / 12-element one.  MEASURED 2026-11-14, both builds: 6.56e-6 against
    6.56e-6 -- both layouts end in the same 52 pm wall element, which is what
    sets the corner error.  Unclamped, ratio 0.05 at 16 elements puts a
    1e-17 m wall element on the grid."""
    e_bad = abs(_J(_fixture(8, 16, ("geometric", 0.05)))[0, 0] - R_REF)
    e_ok = abs(_J(_fixture(8, 12, "geometric"))[0, 0] - R_REF)
    assert e_bad < 3.0 * e_ok, (e_bad, e_ok)


# --------------------------------------------------------------------------- #
# the JAX twin and the conical path pick the spec up
# --------------------------------------------------------------------------- #
#: NumPy (uniform layers from the shared geometric eig) vs the JAX twin (full
#: eig for every layer -- decision D10 keeps the fast path out of the trace).
#: MEASURED 2026-11-14 on fixture A, geometric degree 6 x 4 el, theta 0.2,
#: max over R / T / J: 2.96e-13 (Windows/MKL), 3.11e-13 (WSL/OpenBLAS);
#: against the DENSE NumPy path the same twin reads 1.43e-13 / 1.33e-13, so
#: the fast path's gauge adds ~2e-13.  Bar = 100 x the cross-build envelope,
#: 11 decades below |J| ~ 0.3.
JAX_PARITY_BAR = 3e-11


def test_jax_twin_forward_matches_numpy_with_geometric():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def build(traced):
        st = PMMStack(P, n_substrate=N_SUB, degree=6, elements_per_region=4,
                      grade="geometric")
        for t, es, eg in LAYERS:
            if es == eg:
                st.add_layer(t, eps=es)
            else:
                e = jnp.asarray(es) if traced else es
                st.add_layer(t, segments=[(F, e), (1 - F, eg)])
        st.set_source(WL, theta=0.2)
        return st
    out_np = build(False).solve()
    out_jx = build(True).solve()
    for a, b in zip(out_np[1:], out_jx[1:]):
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        assert d < JAX_PARITY_BAR, d


_CONICAL_LAYERS = [(50e-9, E_AU, 1.0 + 0j), (21e-9, E_HF, E_HF),
                   (70e-9, E_AL, E_AL)]


def test_conical_incidence_refuses_geometric():
    """Geometric grading is refused at conical incidence (phi != 0).

    MEASURED 2026-11-14 (both builds) on the strip / 21 nm spacer / mirror
    stack, theta 0.2, phi 0.3, 12 geometric elements at the default floor
    (37 pm): the cross-polarized J_xy drifts 0.00596 + 0.00742i ->
    0.00544 + 0.00833i over degree 8 .. 24 while the planar solve of the same
    layout converges by ~1e-6 per step, and the ky0 = 0 conical cascade
    reproduces the classical solve only to 5e-8 (Chebyshev 3 el: 4.5e-13)."""
    st = _fixture(8, 12, "geometric", theta=0.2, phi=0.3,
                  layers=_CONICAL_LAYERS)
    with pytest.raises(NotImplementedError, match="conical incidence"):
        st.solve()
    args = (P, E_AU, 1.0, N_SUB, 1.0, 50e-9, F, WL)
    with pytest.raises(NotImplementedError, match="conical incidence"):
        pmm_jones_1d_conical(*args, theta=0.2, phi=0.3, degree=8,
                             elements_per_region=12, grade="geometric")
    from lumenairy.elements.pmm import pmm_jones_1d_conical_tensor
    cell = np.array([E_AU * np.eye(3)] * 5 + [np.eye(3)] * 6)
    with pytest.raises(NotImplementedError, match="conical incidence"):
        pmm_jones_1d_conical_tensor(P, cell, N_SUB, 1.0, 50e-9, WL, theta=0.2,
                                    phi=0.3, degree=8, elements_per_region=12,
                                    grade="geometric")
    # the normalizer still runs first: an invalid value is a ValueError
    with pytest.raises(ValueError, match="unknown grade"):
        pmm_jones_1d_conical(*args, theta=0.2, phi=0.3, degree=8,
                             elements_per_region=4, grade="geometirc")
    # and Chebyshev / its alias keep working on the conical path, identically
    Jt = _J(_fixture(8, 3, True, theta=0.2, phi=0.3, layers=_CONICAL_LAYERS))
    Jc = _J(_fixture(8, 3, "chebyshev", theta=0.2, phi=0.3,
                     layers=_CONICAL_LAYERS))
    assert np.array_equal(Jt, Jc)
    J1 = np.asarray(pmm_jones_1d_conical(*args, theta=0.2, phi=0.3, degree=8,
                                         elements_per_region=3, grade=True)[3])
    J2 = np.asarray(pmm_jones_1d_conical(*args, theta=0.2, phi=0.3, degree=8,
                                         elements_per_region=3,
                                         grade="chebyshev")[3])
    assert np.array_equal(J1, J2)


def test_pmm_jones_2d_refuses_geometric_on_both_backends():
    from lumenairy.elements.pmm import pmm_jones_2d
    cell = np.ones((4, 4, 3, 3)) * np.eye(3)
    cell[:2, :2] = 4.0 * np.eye(3)
    args = (400e-9, 400e-9, cell, 1.5, 1.0, 100e-9, 1000e-9)
    with pytest.raises(NotImplementedError, match="1-D PMM family"):
        pmm_jones_2d(*args, degree=5, n_orders=3, grade="geometric")
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    with pytest.raises(NotImplementedError, match="1-D PMM family"):
        pmm_jones_2d(400e-9, 400e-9, jnp.asarray(cell), 1.5, 1.0, 100e-9,
                     1000e-9, degree=5, n_orders=3, grade="geometric")


# --------------------------------------------------------------------------- #
# the small-element guard
# --------------------------------------------------------------------------- #
def test_guard_refuses_a_layout_made_tiny_element():
    st = _fixture(8, 12, ("geometric", 0.08, 1e-15))   # wall element 0.29 pm
    with pytest.raises(ValueError, match="below the measured safe size") as e:
        st.solve()
    msg = str(e.value)
    assert "allow_small_elements=True" in msg and "explicit floor" in msg
    # the sweep and the prepared path read the same guard
    with pytest.raises(ValueError, match="below the measured safe size"):
        st.solve_vs_wavelength([1250e-9, 1300e-9])
    with pytest.raises(ValueError, match="below the measured safe size"):
        st.prepare().solve(wavelength=WL)


def test_guard_threshold_follows_the_wavelength():
    """The same layout is refused at a long wavelength and accepted at a
    short one: the safe size is a fraction of the wavelength."""
    h_wall = 0.5 * 0.17 ** 5 * W                       # 12.4 pm
    st = _fixture(6, 12, ("geometric", 0.17, 1e-15))
    long_wl = 2.0 * h_wall / S._SMALL_ELEMENT_FRAC     # h = frac / 2
    short_wl = 0.5 * h_wall / S._SMALL_ELEMENT_FRAC    # h = 2 x frac
    st.set_source(long_wl)
    with pytest.raises(ValueError, match="below the measured safe size"):
        st.solve()
    st.set_source(short_wl)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()


def test_default_geometric_floor_follows_the_wavelength():
    """Without an explicit floor PMMStack uses max(1e-4 x period,
    4e-5 x wavelength), resolved at solve time; an explicit floor is kept."""
    st = _fixture(8, 12, "geometric")
    stored = st.grade
    assert stored[2] == C.GEOMETRIC_DEFAULT_MIN_ELEMENT_FRAC * P      # 37 pm
    assert st._effective_grade(WL) == ("geometric", stored[1],
                                       S._SMALL_ELEMENT_SUGGEST_FRAC * WL)
    assert st._effective_grade(100e-9) == stored          # period part wins
    assert st._effective_grade(None) == stored
    seen = []
    orig = S._build_sem_tensor_segments

    def spy(*a, **k):
        seen.append(a[5])
        return orig(*a, **k)
    import unittest.mock as um
    with um.patch.object(S, "_build_sem_tensor_segments", spy):
        st.solve()
    assert set(seen) == {("geometric", stored[1],
                          S._SMALL_ELEMENT_SUGGEST_FRAC * WL)}
    assert st.grade == stored                        # restored after the solve
    explicit = ("geometric", 0.2, 6e-11)
    st2 = _fixture(8, 12, explicit)
    assert st2._effective_grade(WL) == explicit
    st3 = _fixture(8, 12, ("geometric", 0.2))         # ratio only: default floor
    assert st3._effective_grade(WL)[2] == S._SMALL_ELEMENT_SUGGEST_FRAC * WL


def test_guard_does_not_fire_on_default_layouts():
    """The defaults (Chebyshev, 1-3 elements) and the default geometric floor
    on fixture A must solve without tripping the guard or the backstop."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for grade, epr in ((True, 1), (True, 3), ("geometric", 12)):
            st = _fixture(8, epr, grade, theta=0.3)
            st.solve()
            assert st._small_element_note is None or \
                st._small_element_note[0] >= S._SMALL_ELEMENT_FRAC * WL


def test_guard_ignores_geometry_made_thin_cells():
    """A thin feature the GEOMETRY makes (an owned liner far below the safe
    size, one element) is the territory of the sliver / within-layer
    machinery, not of this guard: it is never refused here, and it does not
    arm the backstop."""
    s = 0.25 * S._SMALL_ELEMENT_FRAC * WL / P          # liner, period fraction
    st = PMMStack(P, n_substrate=N_SUB, degree=8)
    st.add_layer(50e-9, segments=[(F, E_AU), (s, 2.25), (1 - F - s, 1.0)])
    st.set_source(WL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st.solve()
    assert st._small_element_note is None


def test_opt_out_solves_and_the_backstop_refuses_super_unity():
    st = _fixture(6, 12, ("geometric", 0.08, 1e-15), allow_small_elements=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            out = st.solve()
        except ValueError as exc:              # the backstop may refuse it
            assert "energy not conserved" in str(exc)
        else:
            tot = out[1].sum(axis=1) + out[2].sum(axis=1)
            assert float(np.max(tot)) <= 1.0 + S._SLIVER_TRIGGER_BAR
    assert st._small_element_note is not None
    # the backstop itself, on an engineered super-unity answer
    R = np.array([[0.7, 0.0], [0.6, 0.0]])
    T = np.array([[0.4, 0.0], [0.3, 0.0]])            # max R+T = 1.1
    with pytest.raises(ValueError, match="energy not conserved"):
        S._warn_stack_energy(R, T, stack=st)
    st._small_element_note = None                     # not armed -> a warning
    with pytest.warns(UserWarning, match="energy not conserved"):
        S._warn_stack_energy(R, T, stack=st)


#: The configurations of the measured failure band that returned O(1) wrong,
#: super-unity answers on the Windows/MKL build with the guard opted out and
#: the backstop off (2026-11-14 sweep; which ones fail is build-dependent).
#: (period, theta, degree, wall element / wavelength)
_GARBAGE_PRONE = [
    (370e-9, 0.3, 10, 3e-6), (370e-9, 0.5, 8, 5.5e-6),
    (200e-9, 0.3, 6, 3e-6), (1300e-9, 0.3, 6, 8e-6),
]


@pytest.mark.parametrize("Pq,theta,deg,hw", _GARBAGE_PRONE)
def test_opted_out_solves_raise_or_are_clean(Pq, theta, deg, hw):
    """RAISE-OR-CLEAN, on every build: with the guard opted out, a solve in
    the failure band either is refused by the energy backstop or returns
    R + T <= 1 + 1e-3 (the strip absorbs, so the theorem is R + T <= 1; every
    measured wrong answer read at least 1.028)."""
    Wq = 0.47 * Pq
    r = (hw * WL / (0.5 * Wq)) ** 0.2
    st = PMMStack(Pq, n_substrate=N_SUB, degree=deg, elements_per_region=12,
                  grade=("geometric", r, 1e-15), allow_small_elements=True)
    st.add_layer(50e-9, segments=[(0.47, E_AU), (0.53, 1.0)])
    st.set_source(WL, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            o, R, T, J = st.solve()
        except ValueError as exc:
            assert "energy not conserved" in str(exc)
            return
    assert float(np.max(R.sum(axis=1) + T.sum(axis=1))) <= \
        1.0 + S._SLIVER_TRIGGER_BAR
