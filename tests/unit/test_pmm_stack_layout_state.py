"""PMMStack's element-layout state: no per-solve state on the shared stack,
a fixed-point ``grade``, prepared caches keyed and guarded by the layout, and
the small-element guard's scope.

Every test here fails on the code before the fix (numbers in each docstring).
"""
import threading
import warnings

import numpy as np
import pytest

import lumenairy.elements.pmm.stack as ps
from lumenairy.elements.pmm import PMMStack

P = 500e-9
E_STRIP, E_MIR = -90 + 7j, -130 + 20j


def _strip(grade="geometric", n_el=12, degree=8, metal=E_STRIP):
    st = PMMStack(P, n_superstrate=1.0, n_substrate=np.sqrt(E_MIR),
                  degree=degree, elements_per_region=n_el, grade=grade)
    st.add_layer(40e-9, segments=[(0.36, metal), (0.64, 1.0)])
    st.add_layer(19e-9, eps=2.1)
    return st


def test_overlapping_prepared_solves_at_two_wavelengths(monkeypatch):
    """Two threads solve one prepared stack at 1500 / 1800 nm (default
    geometric floor: the wavelength part binds, so the two layouts differ),
    choreographed A enters, B enters, A exits, B resumes.  Before the fix
    the solve wrote its layout onto the shared stack: both answers came
    back wrong at the layout-change level and the prepared cache kept the
    wrong modes (a later single-threaded solve was still wrong).  Now:
    identical to the references bit for bit."""
    wl_a, wl_b = 1500e-9, 1800e-9
    ref_a = _strip().prepare().solve(wavelength=wl_a, angle=0.3)[3]
    ref_b = _strip().prepare().solve(wavelength=wl_b, angle=0.3)[3]
    pr = _strip().prepare()
    a_in, b_in, go_a, go_b = (threading.Event() for _ in range(4))
    orig = ps._PreparedPMMStack._layer_eig

    def hooked(self, i, *a, **k):
        name = threading.current_thread().name
        if i == 0 and name == "A":
            a_in.set()
            go_a.wait(60)
        if i == 0 and name == "B":
            b_in.set()
            go_b.wait(60)
        return orig(self, i, *a, **k)
    monkeypatch.setattr(ps._PreparedPMMStack, "_layer_eig", hooked)
    out, err = {}, []

    def run(key, wl):
        try:
            out[key] = pr.solve(wavelength=wl, angle=0.3)[3]
        except Exception as exc:                 # pragma: no cover - report
            err.append(exc)
    A = threading.Thread(target=run, args=("a", wl_a), name="A")
    B = threading.Thread(target=run, args=("b", wl_b), name="B")
    A.start()
    assert a_in.wait(60)
    B.start()
    assert b_in.wait(60)
    go_a.set()
    A.join(120)
    go_b.set()
    B.join(120)
    assert not err, err
    assert np.array_equal(out["a"], ref_a)
    assert np.array_equal(out["b"], ref_b)
    monkeypatch.setattr(ps._PreparedPMMStack, "_layer_eig", orig)
    assert np.array_equal(pr.solve(wavelength=wl_b, angle=0.3)[3], ref_b)


def test_stored_grade_is_a_fixed_point():
    """``PMMStack(..., grade=st.grade)`` and ``st.grade = st.grade`` must
    reproduce the layout.  Before the fix the stored spec carried the period
    part of the default floor as an EXPLICIT floor, so both raised the
    small-element error wherever that period part lies below the bar."""
    st = _strip()
    assert st.grade == ("geometric", 0.17, None)
    J0 = st.prepare().solve(wavelength=1500e-9, angle=0.2)[3]
    st2 = _strip(grade=st.grade)
    assert st2.grade == st.grade
    assert np.array_equal(st2.prepare().solve(wavelength=1500e-9,
                                              angle=0.2)[3], J0)
    st.grade = st.grade
    assert np.array_equal(st.prepare().solve(wavelength=1500e-9,
                                             angle=0.2)[3], J0)
    # the explicit-floor form round-trips too
    st3 = _strip(grade=("geometric", 0.2, 2e-11))
    assert _strip(grade=st3.grade).grade == ("geometric", 0.2, 2e-11)


def test_layout_change_after_prepare_is_refused():
    """``st.grade = True`` after ``prepare()``: before the fix the next
    prepared solve mixed the cached geometric half-space operators with
    Chebyshev layer operators and returned an O(1) wrong answer.  Now any layout change is refused."""
    st = _strip(metal="metal")
    pr = st.prepare()
    pr.solve(wavelength=1500e-9, materials={"metal": E_STRIP}, angle=0.3)
    st.grade = True
    with pytest.raises(ValueError, match="changed after prepare"):
        pr.solve(wavelength=1500e-9, materials={"metal": -60 + 5j}, angle=0.3)
    st.grade = "geometric"                       # back to the prepared layout
    pr.solve(wavelength=1500e-9, materials={"metal": -60 + 5j}, angle=0.3)
    st.degree = 10
    with pytest.raises(ValueError, match="changed after prepare"):
        pr.solve(wavelength=1500e-9, materials={"metal": -60 + 5j}, angle=0.3)


def _wall_layer(n_el, degree=10):
    """A dielectric ridge (eps 6, 45 % of an 800 nm period) with a 1 nm
    lossy wall layer (eps -8+4i) on one side, 1200 nm, theta 0.2."""
    st = PMMStack(800e-9, n_superstrate=1.0, n_substrate=1.5, degree=degree,
                  elements_per_region=n_el)
    lf = 1e-9 / 800e-9
    st.add_layer(200e-9, segments=[(0.45, 6.0), (lf, -8 + 4j),
                                    (0.55 - lf, 1.0)])
    return st.set_source(1200e-9, angle=0.2)


def test_default_grading_thin_wall_layer():
    """A 1 nm wall layer (a cell the geometry makes thin), Chebyshev
    grading.  16 / 24 elements per region (wall element 8.0e-6 / 3.6e-6
    wavelengths) solve silently, measured 5.1e-7 / 2.9e-6 from a degree-24 x
    6 reference at degree 10; 40 elements (1.3e-6, where this cell's
    round-off was measured at 1.5e-5) solve with the band's warning; 64
    (5.0e-7) are refused with the Chebyshev advice."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = np.asarray(_wall_layer(6, degree=24).solve()[3])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for n in (16, 24):
            J = np.asarray(_wall_layer(n).solve()[3])
            assert np.abs(J - ref).max() < 1e-5
    with pytest.warns(UserWarning, match="round-off"):
        _wall_layer(40, degree=4).solve()
    with pytest.raises(ValueError, match="fewer elements_per_region") as e:
        _wall_layer(64).solve()
    assert "geometric" not in str(e.value).split("grade=True.")[-1]


def test_traced_wavelength_with_default_geometric_floor_is_refused():
    """The default floor needs a concrete wavelength; before the fix a
    traced one fell back to the period-only floor (a different layout from
    the NumPy solve at the same wavelength)."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    st = _strip()

    def f(wl):
        return st._effective_grade(wl)[2]
    with pytest.raises(ValueError, match="concrete positive number"):
        jax.grad(lambda wl: f(wl) * wl)(1.5e-6)
    st_e = _strip(grade=("geometric", 0.17, 2e-11))
    assert st_e._effective_grade(None) == ("geometric", 0.17, 2e-11)
