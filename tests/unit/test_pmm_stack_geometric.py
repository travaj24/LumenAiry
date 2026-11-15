"""Geometric element grading through PMMStack, and the small-element guard.

The strip / spacer / mirror stack (a generic fixture): period 500 nm, a 180 nm
wide / 40 nm thick metal strip (eps -90+7i) in air over 2 / 6 / 3 / 6 / 2 nm
dielectric and lossy films (eps 2.8 / 4.2 / 2.2+0.6i / 4.2 / 2.8) over an
80 nm metal mirror (eps -130+20i), air above, n = 1.5 below, 1500 nm,
TM = E_x.  TM sits on the gap resonance (|r| = 0.20).  r_ref: the converged
PMM value (see ``R_REF``).

Every physics bar carries its measurement (date, BLAS build) in a comment;
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

WL, P, W = 1500e-9, 500e-9, 180e-9
F = W / P
E_STRIP, E_MIRROR = -90.0 + 7.0j, -130.0 + 20.0j
E_LOW, E_HIGH, E_LOSSY = 2.8 + 0j, 4.2 + 0j, 2.2 + 0.6j
N_SUB = 1.5
#: TM reference: PMMStack geometric degree 24 x 12 elements (fixture replaced
#: by a generic one; converged 2026-11-15, Windows/MKL: degree 16 / 20 / 28
#: x 12 and degree 20 / 24 x 16 agree with it to <= 3.7e-7, and the
#: independent RCWAStack (Li rules, adaptive resolution, 200 orders) to
#: 1.5e-7 -- an uncertainty of ~4e-7).
R_REF = -0.10122424808 - 0.17630843363j
LAYERS = [(40e-9, E_STRIP, 1.0 + 0j), (2e-9, E_LOW, E_LOW),
          (6e-9, E_HIGH, E_HIGH), (3e-9, E_LOSSY, E_LOSSY),
          (6e-9, E_HIGH, E_HIGH), (2e-9, E_LOW, E_LOW),
          (80e-9, E_MIRROR, E_MIRROR)]


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
    # the DEFAULT floor is stored unresolved (None): each solve resolves it
    # from its own wavelength, and the stored spec is a fixed point
    assert st.grade == ("geometric", C.GEOMETRIC_DEFAULT_RATIO, None)
    assert PMMStack(P, grade="chebyshev").grade is True
    assert PMMStack(P, grade=0).grade is False
    st2 = PMMStack(P, elements_per_region=4, grade=("geometric", 0.2))
    assert st2.grade == ("geometric", 0.2, None)
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
# physics on the strip / spacer / mirror stack
# --------------------------------------------------------------------------- #
#: Geometric (default floor: max(3e-5 x period, 1.25e-5 x wavelength) =
#: 18.75 pm here) degree 8 x 12 el vs the same layout at degree 20.  Fixture
#: replaced by a generic one; RE-MEASURED 2026-11-15: 1.9e-6 (Windows/MKL;
#: the degree-20 value is itself 1.3e-7 from r_ref).  Bar 2e-5 kept: 10x
#: above the reading, 55x below the Chebyshev-at-equal-cost error (1.1e-3).
GEO_DEG_BAR = 2e-5


def test_geometric_beats_chebyshev_at_equal_cost():
    J8 = _J(_fixture(8, 12, "geometric"))
    J20 = _J(_fixture(20, 12, "geometric"))
    Jc = _J(_fixture(10, 8, True))                 # Chebyshev, DOF 160 vs 192
    e8 = abs(J8[0, 0] - J20[0, 0])
    assert e8 < GEO_DEG_BAR, e8
    # against the converged reference: geometric 1.9e-6, Chebyshev 1.1e-3
    # (RE-MEASURED 2026-11-15, Windows/MKL) -- a >= 50x decision with 12x to
    # spare
    assert abs(Jc[0, 0] - R_REF) > 50.0 * abs(J8[0, 0] - R_REF)


#: theta = 0.3: degree 8 vs 20 at 12 geometric elements.  RE-MEASURED
#: 2026-11-15 at the 18.75 pm floor: 1.7e-6 (Windows/MKL).  Bar as for
#: normal incidence (12x above).
GEO_OBLIQUE_BAR = 2e-5


def test_geometric_oblique_converges():
    J8 = _J(_fixture(8, 12, "geometric", theta=0.3))
    J20 = _J(_fixture(20, 12, "geometric", theta=0.3))
    d = abs(J8[0, 0] - J20[0, 0])
    assert d < GEO_OBLIQUE_BAR, d


#: TE (E_y, spectrally converging -- a correct mesh change cannot move it) of
#: geometric degree 8 x 12 el vs single-element degree 60.  RE-MEASURED
#: 2026-11-15 at the 18.75 pm floor: 5.8e-8 (Windows/MKL) -- the eig
#: round-off of a 1.25e-5-wavelength element; the bar sits 14x below what an
#: unclamped tiny wall does to TE (1.4e-5 .. 2.2e-5 at ratio 0.10,
#: single-layer measurement).
GEO_TE_BAR = 1e-6


def test_geometric_leaves_te_untouched():
    Jg = _J(_fixture(8, 12, "geometric"))
    Jr = _J(_fixture(60, 1, True))
    d = abs(Jg[1, 1] - Jr[1, 1])
    assert d < GEO_TE_BAR, d


def test_the_floor_turns_a_bad_ratio_into_a_good_layout():
    """('geometric', 0.05) at 16 elements is clamped by the default floor
    (18.75 pm) to a working layout.  Unclamped, ratio 0.05 at 16 elements puts
    a 7e-17 m wall element on the grid.  RE-MEASURED 2026-11-15 on the
    generic fixture: 1.1e-6 from r_ref (Windows/MKL) against 1.9e-6 for the
    0.17 / 12-element layout.  The decision is stated against the geometric
    bar: 19x below it, and 1000x below the Chebyshev error it must separate
    from."""
    e_bad = abs(_J(_fixture(8, 16, ("geometric", 0.05)))[0, 0] - R_REF)
    assert e_bad < GEO_DEG_BAR, e_bad


# --------------------------------------------------------------------------- #
# the JAX twin and the conical path pick the spec up
# --------------------------------------------------------------------------- #
#: NumPy (uniform layers from the shared geometric eig) vs the JAX twin (the
#: same fast path since 2026-11-14; this bar was derived when the twin ran the
#: full eig for every layer and stays as an upper bound).
#: RE-MEASURED 2026-11-15 on the strip / spacer / mirror stack, geometric
#: degree 6 x 4 el, theta 0.2, max over R / T / J: 4.2e-13 (Windows/MKL),
#: and the same against the DENSE NumPy path.  Bar 70x above the reading,
#: 11 decades below |J_xx| ~ 0.2.
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
                e = jnp.asarray(es) if traced else es  # the strip, traced
                st.add_layer(t, segments=[(F, e), (1 - F, eg)])
        st.set_source(WL, theta=0.2)
        return st
    out_np = build(False).solve()
    out_jx = build(True).solve()
    for a, b in zip(out_np[1:], out_jx[1:]):
        d = float(np.max(np.abs(np.asarray(a) - np.asarray(b))))
        assert d < JAX_PARITY_BAR, d


#: strip / single-layer spacer (19 nm, eps 3.6) / mirror
_CONICAL_LAYERS = [(40e-9, E_STRIP, 1.0 + 0j), (19e-9, 3.6 + 0j, 3.6 + 0j),
                   (80e-9, E_MIRROR, E_MIRROR)]


def test_conical_incidence_refuses_geometric(monkeypatch):
    """Geometric grading at conical incidence (phi != 0) is refused only with
    the structured conical solves switched off
    (``conical.PMM_CONICAL_STRUCTURED = False``, the fail-before switch): the
    dense conical eig it restores cannot carry the small elements.  With the
    structured solves (the default since 2026-11-14) it is accepted and
    validated in ``test_pmm_conical_structured.py``.

    RE-MEASURED 2026-11-15 (Windows/MKL) with the dense eig, on the strip /
    19 nm spacer / mirror stack, theta 0.2, phi 0.3, 12 geometric elements at
    the default floor (18.75 pm): the cross-polarized J_xy wanders from
    0.00872 + 0.00596i (degree 8) to 0.0192 - 0.0108i (degree 24), 0.022 at
    most, while the planar solve of the same layout converges by <= 3.4e-6
    per step, and the ky0 = 0 conical cascade reproduces the classical solve
    only to 4.0e-7 (Chebyshev 3 el: 2.7e-13)."""
    import lumenairy.elements.pmm.conical as CO
    monkeypatch.setattr(CO, "PMM_CONICAL_STRUCTURED", False)
    st = _fixture(8, 12, "geometric", theta=0.2, phi=0.3,
                  layers=_CONICAL_LAYERS)
    with pytest.raises(NotImplementedError, match="conical incidence"):
        st.solve()
    args = (P, E_STRIP, 1.0, N_SUB, 1.0, 40e-9, F, WL)
    with pytest.raises(NotImplementedError, match="conical incidence"):
        pmm_jones_1d_conical(*args, theta=0.2, phi=0.3, degree=8,
                             elements_per_region=12, grade="geometric")
    from lumenairy.elements.pmm import pmm_jones_1d_conical_tensor
    cell = np.array([E_STRIP * np.eye(3)] * 4 + [np.eye(3)] * 7)
    with pytest.raises(NotImplementedError, match="conical incidence"):
        pmm_jones_1d_conical_tensor(P, cell, N_SUB, 1.0, 40e-9, WL, theta=0.2,
                                    phi=0.3, degree=8, elements_per_region=12,
                                    grade="geometric")
    monkeypatch.setattr(CO, "PMM_CONICAL_STRUCTURED", True)
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
        st.solve_vs_wavelength([1450e-9, 1500e-9])
    with pytest.raises(ValueError, match="below the measured safe size"):
        st.prepare().solve(wavelength=WL)


def test_guard_threshold_follows_the_wavelength():
    """The same layout is refused at a long wavelength and accepted at a
    short one: the safe size is a fraction of the wavelength."""
    h_wall = 0.5 * 0.17 ** 5 * W                       # 12.8 pm
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
    """Without an explicit floor PMMStack uses the 1-D default
    max(3e-5 x period, 1.25e-5 x wavelength), resolved at solve time; an
    explicit floor is kept."""
    st = _fixture(8, 12, "geometric")
    stored = st.grade
    assert stored[2] is None                          # the default floor
    assert st._effective_grade(WL) == ("geometric", stored[1],
                                       S._SMALL_ELEMENT_SUGGEST_FRAC * WL)
    assert st._effective_grade(100e-9) == (                # period part wins
        "geometric", stored[1], C.GEOMETRIC_DEFAULT_MIN_ELEMENT_FRAC * P)
    with pytest.raises(ValueError, match="concrete positive number"):
        st._effective_grade(None)                     # no wavelength: refused
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
    on the strip / spacer / mirror stack must solve without tripping the
    guard or the backstop."""
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
    st.add_layer(40e-9, segments=[(F, E_STRIP), (s, 2.25), (1 - F - s, 1.0)])
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
    # not armed (the note is derived from the geometry: a stack without a
    # small layout-made element) -> a warning
    st_ok = _fixture(6, 3, True)
    assert st_ok._small_element_watch(WL) is None
    with pytest.warns(UserWarning, match="energy not conserved"):
        S._warn_stack_energy(R, T, stack=st_ok)


#: Configurations (the strip layer of the generic fixture) that return O(1)
#: wrong, super-unity answers with the pre-fix selector restored
#: (the private switch ``_core._PMM_EIG_ROUNDOFF_BAR`` set False, in a
#: measurement script -- this module imports JAX), the guard opted out and the
#: backstop off (re-measured 2026-11-15, Windows/MKL; which rows fail is
#: build-dependent).  (period, theta, degree, wall element / wavelength)
_GARBAGE_PRONE = [
    (800e-9, 0.3, 8, 4e-6), (1200e-9, 0.7, 6, 8e-6),
    (250e-9, 0.7, 8, 4e-6), (1200e-9, 0.15, 20, 8e-6),
]


@pytest.mark.parametrize("Pq,theta,deg,hw", _GARBAGE_PRONE)
def test_opted_out_solves_return_the_reference(Pq, theta, deg, hw):
    """These rows returned super-unity garbage before the selector fix
    (``_core._growth_mask``).  With the guard opted out they must now be
    passive (the strip absorbs: R + T <= 1) AND agree with the same degree at
    a 1e-4-wavelength wall element.  Fixture replaced by a generic one;
    RE-MEASURED 2026-11-15 (Windows/MKL): |dJ| <= 1.5e-6 (the round-off floor
    of 4e-6 .. 8e-6 wavelength elements); bar 5e-5, against pre-fix errors
    of 2.19 / 1.77 / 7.27 / 0.225 with R + T = 9.84 / 8.17 / 108 / 1.62
    (pre-fix selector restored, backstop off).  WHICH rows fail is a
    round-off sign lottery over code, degree and build;
    ``test_pmm_eig_roundoff_selector.py`` carries more of them."""
    def run(h):
        Wq = F * Pq
        r = (h * WL / (0.5 * Wq)) ** 0.2
        st = PMMStack(Pq, n_substrate=N_SUB, degree=deg,
                      elements_per_region=12,
                      grade=("geometric", r, 1e-15), allow_small_elements=True)
        st.add_layer(40e-9, segments=[(F, E_STRIP), (1.0 - F, 1.0)])
        st.set_source(WL, theta=theta)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
        return np.asarray(J), R.sum(axis=1) + T.sum(axis=1)
    J, tot = run(hw)
    Jr, _ = run(1e-4)
    assert float(np.max(tot)) <= 1.0, tot
    assert float(np.max(np.abs(J - Jr))) < 5e-5
