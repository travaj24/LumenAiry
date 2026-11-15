"""Geometric element grading for the 1-D PMM family (the grading kernel and
its single-layer documentation).

What is pinned here
-------------------
* The ONE grading rule (:func:`lumenairy.elements.pmm._core._graded_fractions`
  and its wrapper ``_graded_boundaries``): the uniform and Chebyshev layouts are
  bit-identical to a frozen copy of the previous implementation; the geometric
  layout is mirror-symmetric for even AND odd element counts, has the stated
  wall element, and honours an absolute floor on the smallest element.
* The spec normalizer ``_normalize_grade``: every accepted form round-trips to
  ``False`` / ``True`` / ``('geometric', ratio, min_element_m)``; everything
  else raises (no value is read through ``bool()`` any more); geometric is
  refused where it is inert (fewer than 3 elements per region) and on the 2-D
  entry points, where it is not validated.
* A census: no ``bool(grade)`` coercion left in the files that own the
  grading argument, exactly one definition of the rule, exactly one copy of
  the Chebyshev formula.
* Physics on the single-layer metal-strip fixture (``pmm_jones_1d_segments``,
  ``stabilize=False`` so the requested degree is the solved degree): the
  geometric layout's accuracy against a high-degree reference and against
  the Chebyshev layout at equal nodal count, the round-off floor (read in the
  spectrally converging TE channel), and that the default call is unchanged.

Bars are measured, with a gap on both sides (docs/TESTING_STANDARDS.md
restatement 5); every bar names its measurement.  The fixture was replaced by
a generic one on 2026-11-15 and every physics reading re-measured on it
(Windows/MKL).
"""
import ast
import os
import pathlib
import re

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

import lumenairy as lum
from lumenairy.elements.pmm import _core as C

PKG = pathlib.Path(lum.__file__).resolve().parent
PMM_DIR = PKG / "elements" / "pmm"

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
# Single-layer metal-strip grating (the strip layer of the strip / spacer /
# mirror stack): a 40 nm strip (eps -90 + 7i) of width W in a period P, air
# between the strips, air above, n = 1.5 below.  (Fixture replaced by a
# generic one; every reading below re-measured on it.)
GEOM = {
    # period ~ wavelength / 3
    "P500": dict(P=500e-9, W=180e-9, WL=1500e-9),
    # period ~ the wavelength (off the Rayleigh anomaly by 0.8 %)
    "P1488": dict(P=1488e-9, W=540e-9, WL=1500e-9),
    # period ~ wavelength / 6.5
    "P230": dict(P=230e-9, W=83e-9, WL=1500e-9),
}
E_STRIP = -90.0 + 7.0j
DEPTH = 40e-9
N_SUB = 1.5

_CACHE = {}


def _jones(grade, degree, epr, theta=0.0, geom="P500"):
    """Zeroth-order reflection Jones of the fixture, solved at EXACTLY
    ``degree`` (``stabilize=False``), memoized per process."""
    key = (repr(grade), degree, epr, theta, geom)
    if key not in _CACHE:
        g = GEOM[geom]
        f = g["W"] / g["P"]
        out = lum.pmm_jones_1d_segments(
            g["P"], [(f, E_STRIP), (1.0 - f, 1.0)], N_SUB, 1.0, DEPTH,
            g["WL"], angle=theta, degree=degree, elements_per_region=epr,
            grade=grade, stabilize=False)
        _CACHE[key] = np.asarray(out[3])
    return _CACHE[key]


def _tm(*a, **k):
    return complex(_jones(*a, **k)[0, 0])


def _te(*a, **k):
    return complex(_jones(*a, **k)[1, 1])


# A geometric spec with the floor disabled (smallest positive double-ish).
def _nofloor(ratio):
    return ("geometric", ratio, 1e-300)


# --------------------------------------------------------------------------- #
# 1. Legacy layouts frozen
# --------------------------------------------------------------------------- #
def _graded_boundaries_5_50_2(a, b, n_el, grade):
    """Frozen copy of the grading rule as shipped before the geometric mode."""
    if n_el <= 1:
        return np.array([a, b])
    if not grade:
        return np.linspace(a, b, n_el + 1)
    i = np.arange(n_el + 1)
    s = 0.5 * (1.0 - np.cos(np.pi * i / n_el))
    return a + (b - a) * s


def _graded_fractions_5_50_2(n_el, grade):
    """Frozen copy of the JAX topology's d_wall-independent fractions."""
    if n_el <= 1:
        return np.array([0.0, 1.0])
    if not grade:
        return np.linspace(0.0, 1.0, n_el + 1)
    i = np.arange(n_el + 1)
    return 0.5 * (1.0 - np.cos(np.pi * i / n_el))


_INTERVALS = [(0.0, 180e-9), (180e-9, 500e-9), (0.3, 1.7)]
_BOOL_LIKE = [True, False, None, 0, 1, np.True_, np.False_]


def test_legacy_layouts_are_bit_identical():
    for n_el in range(1, 31):
        for g in _BOOL_LIKE:
            for a, b in _INTERVALS:
                old = _graded_boundaries_5_50_2(a, b, n_el, g)
                assert np.array_equal(C._graded_boundaries(a, b, n_el, g), old)
                # the normalized spec gives the same layout as the raw value
                ng = C._normalize_grade(g, period=500e-9, where="t")
                assert ng is bool(g)
                assert np.array_equal(C._graded_boundaries(a, b, n_el, ng), old)
            assert np.array_equal(C._graded_fractions(n_el, g),
                                  _graded_fractions_5_50_2(n_el, g))


# --------------------------------------------------------------------------- #
# 2. Geometric layout
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n_el", range(3, 17))
def test_geometric_layout_shape(n_el):
    r, width = 0.17, 180e-9
    s = C._graded_fractions(n_el, _nofloor(r), width=width)
    assert s.shape == (n_el + 1,)
    assert s[0] == 0.0 and s[-1] == 1.0
    assert np.all(np.diff(s) > 0.0)
    # exact mirror symmetry (to rounding of 1 - v)
    assert np.max(np.abs(s + s[::-1] - 1.0)) <= 1e-15
    m = n_el // 2
    c = 0.5 if n_el % 2 == 0 else r / (1.0 + r)
    wall = c * r ** (m - 1)
    assert abs(np.diff(s)[0] - wall) <= 1e-12 * wall
    # the right half is 1 - (left half): exact up to the rounding of 1 - v
    assert abs(np.diff(s)[-1] - wall) <= 1e-15
    # sizes grow from each wall to the centre
    h = np.diff(s)
    half = h[: (n_el + 1) // 2]
    assert np.all(np.diff(half) > 0.0)
    if n_el % 2:
        # the central element continues the geometric sequence of the layers
        # (its neighbour is the layer c (1 - r) once there is more than one)
        assert abs(h[m] - (1.0 - r) / (1.0 + r)) <= 1e-15
        if m >= 2:
            assert abs(h[m - 1] / h[m] - r) <= 1e-12


def test_geometric_two_elements_is_the_midpoint_split():
    assert np.array_equal(C._graded_fractions(2, _nofloor(0.17), width=1.0),
                          np.array([0.0, 0.5, 1.0]))
    assert np.array_equal(C._graded_fractions(1, _nofloor(0.17)),
                          np.array([0.0, 1.0]))


def test_geometric_boundaries_land_on_the_walls():
    for a, b in _INTERVALS:
        for n_el in (3, 8, 12, 13):
            out = C._graded_boundaries(a, b, n_el, _nofloor(0.17))
            assert out[0] == a and out[-1] == b


def test_geometric_needs_the_width_for_three_or_more_elements():
    with pytest.raises(ValueError, match="width"):
        C._graded_fractions(4, _nofloor(0.17))


# --------------------------------------------------------------------------- #
# 3. Floor
# --------------------------------------------------------------------------- #
def test_floor_clamps_a_bad_ratio():
    floor, width = 15e-12, 180e-9
    b = C._graded_boundaries(0.0, width, 16, ("geometric", 0.05, floor))
    h = np.diff(b)
    assert len(h) == 16
    assert np.all(h >= floor * (1.0 - 1e-12))
    assert abs(h.min() / floor - 1.0) <= 1e-9
    # mirror symmetric: distance from the left wall == distance from the right
    assert np.max(np.abs((b - b[0]) - (b[-1] - b[::-1]))) <= 1e-21


def test_floor_only_acts_below_it():
    width = 180e-9
    free = C._graded_boundaries(0.0, width, 12, _nofloor(0.17))
    # ratio 0.17 at 12 elements has a 12.8 pm wall element: a 12 pm floor
    # leaves the layout bit-identical ...
    kept = C._graded_boundaries(0.0, width, 12, ("geometric", 0.17, 12e-12))
    assert np.array_equal(kept, free)
    # ... while the default floor of a 500 nm period at 1500 nm (18.75 pm, the
    # wavelength part) raises the ratio until the wall element is exactly the
    # floor (about 0.184 here).
    spec = C._normalize_grade("geometric", period=500e-9, where="t",
                              wavelength=1500e-9)
    assert spec[2] == C.GEOMETRIC_DEFAULT_MIN_ELEMENT_WL_FRAC * 1500e-9
    b = C._graded_boundaries(0.0, width, 12, spec)
    h = np.diff(b)
    assert abs(h[0] / spec[2] - 1.0) <= 1e-9 and abs(h[-1] / spec[2] - 1.0) <= 1e-9
    assert np.all(h >= spec[2] * (1.0 - 1e-12))
    r_eff = (h[0] + h[1]) / (h[0] + h[1] + h[2])
    assert 0.180 < r_eff < 0.188


def test_floor_falls_back_to_uniform_when_unreachable():
    # region of 0.3 nm, floor 15 pm, 12 elements: no graded layout with the
    # wall element smallest can keep every element >= 15 pm -> uniform.
    b = C._graded_boundaries(0.0, 0.3e-9, 12, ("geometric", 0.17, 15e-12))
    assert np.allclose(np.diff(b), 0.3e-9 / 12, rtol=1e-12, atol=0.0)
    # a requested ratio >= 1/2 whose interior element violates the floor
    # also falls back to uniform (above 1/2 the smallest element is interior)
    # (ratio 0.9, 12 elements: smallest element 0.5 * 0.9^4 * 0.1 = 0.033)
    s = C._graded_fractions(12, ("geometric", 0.9, 0.05), width=1.0)
    assert np.allclose(np.diff(s), 1.0 / 12, rtol=1e-12, atol=0.0)
    # ... and is kept when it does not
    s = C._graded_fractions(12, ("geometric", 0.6, 1e-3), width=1.0)
    assert not np.allclose(np.diff(s), 1.0 / 12)
    assert np.min(np.diff(s)) >= 1e-3


@pytest.mark.parametrize("n_el", [3, 5, 9, 12, 13, 16])
def test_floor_holds_for_every_element_count(n_el):
    floor = 15e-12
    for width in (2e-9, 20e-9, 180e-9, 1e-6):
        b = C._graded_boundaries(0.0, width, n_el, ("geometric", 0.17, floor))
        h = np.diff(b)
        assert len(h) == n_el
        if width / n_el >= floor:          # achievable at all
            assert h.min() >= floor * (1.0 - 1e-12)


# --------------------------------------------------------------------------- #
# 4. Normalizer
# --------------------------------------------------------------------------- #
def test_normalizer_accepted_forms_round_trip():
    P = 500e-9
    fl = C.GEOMETRIC_DEFAULT_MIN_ELEMENT_FRAC * P
    # a list, not a dict: True / 1 / np.True_ hash alike and would collapse
    cases = [
        (True, True), (False, False), (None, False), (0, False), (1, True),
        (np.True_, True), (np.False_, False), (np.int64(1), True),
        (np.uint8(0), False), ("chebyshev", True), ("Chebyshev", True),
        ("geometric", ("geometric", C.GEOMETRIC_DEFAULT_RATIO, fl)),
        (("geometric", 0.2), ("geometric", 0.2, fl)),
        (("geometric", 0.2, 5e-12), ("geometric", 0.2, 5e-12)),
        (["geometric", 0.25], ("geometric", 0.25, fl)),
        ((" Geometric ", np.float32(0.25)), ("geometric", 0.25, fl)),
    ]
    for raw, want in cases:
        got = C._normalize_grade(raw, period=P, where="t")
        assert got == want and type(got) is type(want)
        if isinstance(want, tuple):
            assert all(type(v) is float for v in got[1:])
        # a normalized spec is a fixed point
        assert C._normalize_grade(got, period=P, where="t") == got
        hash(got)
    assert C.GEOMETRIC_DEFAULT_RATIO == 0.17
    assert C.GEOMETRIC_DEFAULT_MIN_ELEMENT_FRAC == 3e-5
    assert C.GEOMETRIC_DEFAULT_MIN_ELEMENT_WL_FRAC == 1.25e-5


def test_default_floor_is_completed_by_the_wavelength():
    """max(3e-5 x period, 1.25e-5 x wavelength); the longest wavelength of an
    array binds; no wavelength -> the period part alone; a non-numeric,
    non-finite or non-positive one raises (dropping the wavelength part
    silently would change the layout without a word); an explicit floor is
    never touched."""
    P = 500e-9
    f = C._normalize_grade("geometric", period=P, where="t",
                           wavelength=1500e-9)[2]
    assert f == 1.25e-5 * 1500e-9
    f = C._normalize_grade("geometric", period=P, where="t",
                           wavelength=[1200e-9, 1500e-9, 1350e-9])[2]
    assert f == 1.25e-5 * 1500e-9
    f = C._normalize_grade("geometric", period=P, where="t",
                           wavelength=200e-9)[2]             # period part wins
    assert f == 3e-5 * P
    assert C._geometric_default_floor(P, None) == 3e-5 * P
    for wl in ("x", float("nan"), -1.0, 0.0):
        with pytest.raises(ValueError, match="wavelength"):
            C._geometric_default_floor(P, wl)
    spec = ("geometric", 0.2, 5e-12)
    assert C._normalize_grade(spec, period=P, where="t",
                              wavelength=1500e-9) == spec


@pytest.mark.parametrize("bad", [
    "geometirc", "yes", "", 2, -1, 0.5, 1.0, ("geometric", 1.2),
    ("geometric", 0.0), ("geometric", -0.1), ("geometric", 0.17, -1),
    ("geometric", 0.17, 0.0), ("geometric", float("nan")),
    ("geometric", "0.17"), ("geometric", True), ("chebyshev", 0.17),
    ("geometric",), ("geometric", 0.17, 1e-11, 3), object(), [1, 2]])
def test_normalizer_rejects(bad):
    with pytest.raises(ValueError, match="t: "):
        C._normalize_grade(bad, period=500e-9, where="t")


def test_normalizer_geometric_needs_three_elements():
    for epr in (1, 2):
        with pytest.raises(ValueError, match="elements_per_region >= 3"):
            C._normalize_grade("geometric", period=1.0, where="t",
                               elements_per_region=epr)
    assert C._normalize_grade("geometric", period=1.0, where="t",
                              elements_per_region=3)[0] == "geometric"
    # bool specs are unaffected by the element count
    assert C._normalize_grade(True, period=1.0, where="t",
                              elements_per_region=1) is True


def test_normalizer_refuses_geometric_where_not_validated():
    with pytest.raises(NotImplementedError, match="1-D PMM family"):
        C._normalize_grade("geometric", period=1.0, where="t",
                           allow_geometric=False)
    assert C._normalize_grade("chebyshev", period=1.0, where="t",
                              allow_geometric=False) is True


_SEG_ARGS = (500e-9, [(0.36, E_STRIP), (0.64, 1.0)], N_SUB, 1.0, DEPTH,
             1500e-9)


@pytest.mark.parametrize("fn_name", [
    "pmm_jones_1d_segments", "pmm_efficiency_1d_segments"])
@pytest.mark.parametrize("epr", [1, 2])
def test_public_1d_geometric_with_one_or_two_elements_raises(fn_name, epr):
    fn = getattr(lum, fn_name)
    with pytest.raises(ValueError, match="elements_per_region >= 3"):
        fn(*_SEG_ARGS, degree=4, elements_per_region=epr, grade="geometric")


def test_public_1d_rejects_an_unknown_grade_instead_of_reading_it_as_true():
    for fn_name in ("pmm_jones_1d_segments", "pmm_efficiency_1d_segments"):
        with pytest.raises(ValueError, match="unknown grade"):
            getattr(lum, fn_name)(*_SEG_ARGS, degree=4, elements_per_region=3,
                                  grade="geometirc")
    eye = np.eye(3)
    with pytest.raises(ValueError, match="unknown grade"):
        lum.pmm_jones_1d(500e-9, E_STRIP * eye, eye, N_SUB, 1.0, DEPTH, 0.36,
                         1500e-9, degree=4, grade="yes")
    with pytest.raises(ValueError, match="unknown grade"):
        lum.pmm_efficiency_1d(500e-9, 2.0, 1.0, N_SUB, 1.0, DEPTH, 0.36,
                              1500e-9, degree=4, grade="yes")


def test_2d_entry_points_refuse_geometric():
    from lumenairy.elements.pmm import twod
    P = 400e-9
    common = dict(n_substrate=1.5, n_superstrate=1.0, depth=100e-9,
                  wavelength=1000e-9)
    calls = [
        lambda g: twod.pmm_efficiency_2d(
            P, P, 4.0, 1.0, (100e-9, 300e-9), (100e-9, 300e-9), grade=g,
            **common),
        lambda g: twod.pmm_efficiency_2d_cell(
            P, P, np.ones((4, 4)), grade=g, **common),
        lambda g: twod.prepare_pmm_2d(
            P, P, 4.0, 1.0, (100e-9, 300e-9), (100e-9, 300e-9), 1.5, 1.0,
            100e-9, grade=g),
        lambda g: twod.prepare_pmm_2d_cell(
            P, P, np.ones((4, 4)), 1.5, 1.0, 100e-9, grade=g),
        lambda g: twod.pmm_efficiency_2d_vs_wavelength(
            P, P, 4.0, 1.0, (100e-9, 300e-9), (100e-9, 300e-9), 1.5, 1.0,
            100e-9, [1000e-9], grade=g),
        lambda g: twod.pmm_efficiency_2d_cell_vs_wavelength(
            P, P, np.ones((4, 4)), 1.5, 1.0, 100e-9, [1000e-9], grade=g),
    ]
    for call in calls:
        with pytest.raises(NotImplementedError, match="1-D PMM family"):
            call("geometric")
        with pytest.raises(NotImplementedError, match="1-D PMM family"):
            call(("geometric", 0.2, 1e-11))
        with pytest.raises(ValueError, match="unknown grade"):
            call("geometirc")


def test_pmm2dstack_refuses_geometric_at_construction_and_on_assignment():
    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    with pytest.raises(NotImplementedError, match="1-D PMM family"):
        PMM2DStackHybrid(400e-9, grade="geometric")
    st = PMM2DStackHybrid(400e-9, grade="chebyshev")
    assert st.grade is True
    with pytest.raises(NotImplementedError, match="1-D PMM family"):
        st.grade = ("geometric", 0.17, 1e-11)
    with pytest.raises(ValueError, match="unknown grade"):
        st.grade = "geometirc"
    assert st.grade is True               # a refused assignment changes nothing
    st.grade = 0
    assert st.grade is False
    with pytest.warns(DeprecationWarning):
        from lumenairy.elements.pmm.stack2d import PMM2DStack
        with pytest.raises(NotImplementedError, match="1-D PMM family"):
            PMM2DStack(400e-9, grade="geometric")


def test_unnormalized_string_reaching_the_kernel_raises():
    # a caller that hands ``grade`` to the layout builder without normalizing
    # it must not have a string read as truthy: the builder raises.
    for bad in ("geometric", "chebyshev", ("geometric", 0.17)):
        with pytest.raises(ValueError, match="un-normalized"):
            C._graded_boundaries(0.0, 1.0, 4, bad)
        with pytest.raises(ValueError, match="un-normalized"):
            C._graded_boundaries(0.0, 1.0, 1, bad)
    with pytest.raises(ValueError, match="invalid geometric spec"):
        C._graded_boundaries(0.0, 1.0, 4, ("geometric", 1.5, 1e-12))
    # the conical 1-D entry and pmm_jones_2d normalize ``grade`` at entry.
    # Conical accepts the geometric spec since the structured conical solves
    # (tests/unit/test_pmm_conical_structured.py) and refuses it only with
    # their fail-before switch off; 2-D refuses it (not validated).
    import lumenairy.elements.pmm.conical as CO
    lum.elements.pmm.pmm_jones_1d_conical(
        500e-9, E_STRIP, 1.0, N_SUB, 1.0, DEPTH, 0.36, 1500e-9,
        theta=0.1, phi=0.2, degree=6, elements_per_region=4,
        grade="geometric")
    CO.PMM_CONICAL_STRUCTURED = False
    try:
        with pytest.raises(NotImplementedError, match="conical incidence"):
            lum.elements.pmm.pmm_jones_1d_conical(
                500e-9, E_STRIP, 1.0, N_SUB, 1.0, DEPTH, 0.36, 1500e-9,
                theta=0.1, phi=0.2, degree=6, elements_per_region=4,
                grade="geometric")
    finally:
        CO.PMM_CONICAL_STRUCTURED = True
    cell = np.ones((4, 4, 3, 3)) * np.eye(3)
    cell[:2, :2] = 4.0 * np.eye(3)
    with pytest.raises(NotImplementedError, match="1-D PMM family"):
        lum.elements.pmm.pmm_jones_2d(
            400e-9, 400e-9, cell, 1.5, 1.0, 100e-9, 1000e-9, degree=5, n_orders=3,
            grade="geometric")


def test_sem_modes_uniform_label_is_recorded_and_inert():
    mats = C._build_sem_tensor_segments(
        500e-9, [0.36, 0.64], [C._tensor3_dict(2.25 * np.eye(3))] * 2, 6, 1,
        True)
    k0 = 2.0 * np.pi / 1500e-9
    base = C._sem_modes_uniform(mats, k0, 0.0, 2.25)
    old = C._MODE_CUT_CENSUS
    try:
        C._MODE_CUT_CENSUS = []
        lab = C._sem_modes_uniform(mats, k0, 0.0, 2.25, label="my site")
        sites = [row["site"] for row in C._MODE_CUT_CENSUS]
        C._MODE_CUT_CENSUS = []
        C._sem_modes_uniform(mats, k0, 0.0, 2.25)
        default_sites = [row["site"] for row in C._MODE_CUT_CENSUS]
    finally:
        C._MODE_CUT_CENSUS = old
    for x, y in zip(base, lab):
        assert np.array_equal(x, y)
    assert sites == ["my site"]
    assert default_sites == ["PMM modal solve (uniform half-space)"]


# --------------------------------------------------------------------------- #
# 5. Census (decision tests)
# --------------------------------------------------------------------------- #
# Files whose grading argument this module's rule owns.  Every one must route
# ``grade`` through the normalizer; ``bool(grade)`` is allowed only inside the
# normalizer itself (on values it has already validated as bool-like).
_OWNED = ["_core.py", "oned.py", "twod.py", "stack2d.py"]
# Elsewhere under pmm/ a ``bool(grade)`` may remain only where the value is
# already normalized upstream or the file is wired separately:
#   stack.py           -- PMMStack's wiring of the spec;
#   _jax_twod.py       -- reached only from twod.py entry points (normalized);
#   _jax_twod_jones.py -- reached from twod_jones.pmm_jones_2d's JAX branch.
_ALLOWED_ELSEWHERE = {"stack.py", "_jax_twod.py", "_jax_twod_jones.py"}


def _bool_grade_calls(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hits = []

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            f = func
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                f = child.name
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
                    and child.func.id == "bool" and len(child.args) == 1):
                arg = child.args[0]
                is_grade = ((isinstance(arg, ast.Name) and arg.id == "grade")
                            or (isinstance(arg, ast.Attribute)
                                and arg.attr == "grade"))
                if is_grade and f != "_normalize_grade":
                    hits.append((path.name, child.lineno))
            visit(child, f)

    visit(tree, None)
    return hits


def test_census_no_bool_grade_coercion():
    owned_hits = []
    for name in _OWNED:
        owned_hits += _bool_grade_calls(PMM_DIR / name)
    assert owned_hits == []
    other = {h[0] for p in PMM_DIR.glob("*.py") if p.name not in _OWNED
             for h in _bool_grade_calls(p)}
    assert other <= _ALLOWED_ELSEWHERE, other - _ALLOWED_ELSEWHERE


def test_census_one_grading_rule():
    defs = []
    cheb = 0
    for p in PKG.rglob("*.py"):
        src = p.read_text(encoding="utf-8")
        defs += [p.name for _ in re.finditer(r"^\s*def _graded_fractions\(", src,
                                             re.M)]
        cheb += src.count("np.cos(np.pi * i / n_el)")
    assert defs == ["_core.py"]
    assert cheb == 1
    # the PMM layout builder delegates to it (no private copy of the rule)
    src = (PMM_DIR / "_core.py").read_text(encoding="utf-8")
    body = src[src.index("def _graded_boundaries("):]
    body = body[: body.index("\ndef ", 10)]
    assert "_graded_fractions(" in body
    assert "np.cos" not in body


# --------------------------------------------------------------------------- #
# 6. Physics, single layer (stabilize=False: the requested degree is solved)
# --------------------------------------------------------------------------- #
# ONE bar separates "converged to the round-off floor" from "a real
# discretisation or floor error".  Fixture replaced by a generic one and
# RE-MEASURED 2026-11-15 (Windows/MKL) at the default floor
# max(3e-5 x period, 1.25e-5 x wavelength):
#   * below: every reading of the default geometric spec at 12 elements --
#     the 500 / 1488 / 230 nm periods; theta 0 and 0.3; degrees 8 / 10 / 14 /
#     20; TE against a one-element degree-60 solve and TM against degree
#     20 x 12 -- is <= 5.6e-7 (largest: TM, 1488 nm period, degree 10,
#     theta 0);
#   * above: the two signals the bar must see -- Chebyshev grading at the
#     same nodal count (TM 2.5e-5) and an UNCLAMPED ratio-0.10 layout (TE
#     1.4e-5 at theta 0, 2.2e-5 at theta 0.3, round-off of its 0.9 pm wall
#     element) -- are >= 1.4e-5.
# 2e-6 sits ~3.6x above the first and ~7x below the second.
_BAR = 2e-6


@pytest.fixture(scope="module")
def ref_tm():
    """TM reference: the default geometric spec at degree 20 x 12 elements."""
    return _tm("geometric", 20, 12)


def _te_ref(theta=0.0, geom="P500"):
    """TE reference: ONE element at degree 60.  TE (E along the strips) has
    no wall singularity and converges spectrally, so a single high-degree
    element is the cleanest reference (degree 80 agrees with it to 2.4e-11 or
    better on the 500 and 1488 nm periods, theta 0 and 0.3); a correct element
    layout cannot move TE, so any TE deviation of a graded layout is
    round-off."""
    return _te(True, 60, 1, theta, geom)


def test_geometric_beats_chebyshev_at_equal_nodal_count(ref_tm):
    # DOF = 2 * elements * degree = 160 for both: measured 1.8e-7 (geometric
    # 8 x 10) against 2.5e-5 (Chebyshev 10 x 8).
    geo = abs(_tm("geometric", 8, 10) - ref_tm)
    cheb = abs(_tm(True, 10, 8) - ref_tm)
    assert geo < _BAR < cheb
    # the recommended 12-element layout (measured 6.5e-8)
    assert abs(_tm("geometric", 8, 12) - ref_tm) < _BAR


def test_geometric_converges_at_oblique_incidence():
    # theta 0.3: degree 8 vs 20 at 12 elements, measured 1.5e-7.
    d8 = _tm("geometric", 8, 12, theta=0.3)
    d20 = _tm("geometric", 20, 12, theta=0.3)
    assert abs(d8 - d20) < _BAR


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_te_is_untouched_by_the_geometric_layout(theta):
    # measured 6.1e-8 / 1.0e-7 (degree 8) and 1.9e-8 / 2.0e-8 (degree 20) at
    # theta 0 / 0.3.
    ref = _te_ref(theta)
    for degree in (8, 20):
        assert abs(_te("geometric", degree, 12, theta) - ref) < _BAR


@pytest.mark.parametrize("theta", [0.0, 0.3])
def test_the_floor_turns_a_bad_ratio_into_a_good_layout(theta):
    """Ratio 0.10 at 12 elements puts a 0.9 pm element at each wall.  The
    default floor clamps it (measured TE 4.6e-8 / 4.6e-8 at theta 0 / 0.3);
    the same ratio with the floor disabled is the signal the bar must see
    (measured 1.4e-5 / 2.2e-5 of round-off)."""
    ref = _te_ref(theta)
    clamped = abs(_te(("geometric", 0.10), 8, 12, theta) - ref)
    unclamped = abs(_te(_nofloor(0.10), 8, 12, theta) - ref)
    assert clamped < _BAR < unclamped


def test_floor_rescues_ratio_005_at_16_elements(ref_tm):
    # measured TM 1.3e-7, TE 1.6e-7
    spec = ("geometric", 0.05)
    assert abs(_tm(spec, 8, 16) - ref_tm) < _BAR
    assert abs(_te(spec, 8, 16) - _te_ref()) < _BAR


# --------------------------------------------------------------------------- #
# 7. Default unchanged
# --------------------------------------------------------------------------- #
def _solve_set():
    g = GEOM["P500"]
    f = g["W"] / g["P"]
    seg = [(f, E_STRIP), (1.0 - f, 1.0)]
    out = []
    for kw in (dict(degree=6, elements_per_region=3, grade=True,
                    stabilize=False),
               dict(degree=6, elements_per_region=3, grade=False,
                    stabilize=False),
               dict(degree=6, elements_per_region=3, grade=True,
                    stabilize=False, angle=0.3),
               dict()):
        out.append(lum.pmm_jones_1d_segments(g["P"], seg, N_SUB, 1.0, DEPTH,
                                             g["WL"], **kw))
    from lumenairy.elements.pmm import twod
    out.append(tuple(twod.pmm_efficiency_2d(
        400e-9, 400e-9, 4.0, 1.0, (100e-9, 300e-9), (100e-9, 300e-9), 1.5,
        1.0, 100e-9, 1000e-9, degree=3, elements_per_strip=3, grade=True,
        n_orders=2)))
    return out


def test_default_results_are_bit_identical_to_the_previous_rule(monkeypatch):
    """Same build, same code path, only the layout rule swapped for its frozen
    copy: every array bit-identical (docs/TOLERANCE_POLICY.md rule 2)."""
    from lumenairy.elements.pmm import twod
    C._clear_pmm_caches()
    new = _solve_set()
    monkeypatch.setattr(C, "_graded_boundaries", _graded_boundaries_5_50_2)
    monkeypatch.setattr(twod, "_graded_boundaries", _graded_boundaries_5_50_2)
    C._clear_pmm_caches()
    old = _solve_set()
    C._clear_pmm_caches()
    for a, b in zip(new, old):
        for x, y in zip(a, b):
            assert np.array_equal(np.asarray(x), np.asarray(y))


# 5.50.2 readings of the same calls (J[0, 0], J[1, 1]) on Windows/MKL.  The
# fixture was replaced by a generic one on 2026-11-15 and these were
# re-recorded by running the calls on the 5.50.2 package itself (the current
# code reproduces them to <= 9e-15 on the same build).
_FROZEN_5_50_2 = [
    ((-0.2521125375489535 + 0.11526573310966924j),
     (-0.5806052442983239 - 0.4797216126784955j)),
    ((-0.25193215952876585 + 0.11501407091278934j),
     (-0.5806061561921603 - 0.47972151402164603j)),
    ((-0.23831230878150023 + 0.11572032664174278j),
     (-0.5994866037830147 - 0.470757725927181j)),
    ((-0.25230840819701517 + 0.11553734353974077j),
     (-0.5806048463606573 - 0.4797216547151939j)),
]


def test_default_results_match_the_previous_release():
    # 1e-12: ~100x the measured release-to-release round-off (<= 9e-15), and
    # far below what any change of layout produces (Chebyshev vs uniform at
    # 3 elements: 3.1e-4).
    for res, (jxx, jyy) in zip(_solve_set()[:4], _FROZEN_5_50_2):
        J = np.asarray(res[3])
        assert abs(J[0, 0] - jxx) < 1e-12
        assert abs(J[1, 1] - jyy) < 1e-12


# --------------------------------------------------------------------------- #
# 8. Floor vs degree and geometry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("geom, theta, degrees", [
    ("P1488", 0.0, (8, 20)),       # period ~ wavelength
    ("P230", 0.3, (8, 10, 20)),    # period ~ wavelength / 6.5, oblique
])
def test_default_floor_stays_clear_of_the_round_off_band(geom, theta, degrees):
    """The round-off danger scale is set by the WAVELENGTH (outright failures
    on small wall elements were a mode-selector defect, fixed 2026-11-14),
    and the default floor carries a wavelength part.  The decision measured
    here, on the geometries the floor is most and least generous on: TE
    stays at round-off at every degree (measured <= 2.1e-8 on the 1488 nm
    period and <= 1.6e-7 on the 230 nm one, Windows/MKL), so the floor does
    not need to grow with degree."""
    ref = _te_ref(theta, geom)
    for degree in degrees:
        assert abs(_te("geometric", degree, 12, theta, geom) - ref) < _BAR


# --------------------------------------------------------------------------- #
# 9. Single-layer documentation (the grading knob docs no longer contradict
#    the measurements)
# --------------------------------------------------------------------------- #
def test_single_layer_docs_state_what_the_grading_knob_does():
    from lumenairy.elements.pmm import oned
    eff = oned.pmm_efficiency_1d.__doc__
    jon = oned.pmm_jones_1d.__doc__
    src = (PMM_DIR / "oned.py").read_text(encoding="utf-8")
    # the two sentences that sent users the wrong way are gone
    assert "recovers the rate" not in src
    assert "Raise with ``grade=True``" not in src
    assert "grade=True to recover" not in src
    # what replaces them
    assert "INERT at 1 and 2 elements" in eff
    assert "'geometric'" in eff and "3e-5 * period" in eff
    assert "1.25e-5 * wavelength" in eff
    assert "ROUND-OFF" in eff
    assert "grade='geometric'" in jon
