"""``RCWAStack.add_layer(segments=[(width_fraction, eps), ...])``.

A lamellar layer of lines along y, laid out left to right from ``x = 0`` in
the PUBLIC frame (``segments[0]`` on ``[0, w0 P)``) -- the layout of
``PMMStack.add_layer(segments=)`` and ``rcwa_jones_1d_segments``.  It is stored
as the equivalent analytic ``shapes`` layer: background ``segments[-1]``, every
other segment a rectangle spanning the full ``period_y``.  The rectangles share
edges exactly, which the shapes validation accepts (its overlap test excludes
tangency).

Fixture A (see ``test_rcwa_stack_auto_formulation.py``): the strip layer via
``segments`` is the analytic ``shapes`` strip, bit for bit, and under the
default ``'li'`` errs 1.06e-2 at N = 100 against the converged reference.
The layout check is sign-sensitive: an asymmetric three-segment cell at
oblique incidence diffracts very differently into +1 and -1, and RCWAStack and
PMMStack must agree order by order.
"""
import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.rcwa import RCWAStack
from lumenairy.elements.rcwa._core import _validate_shapes

WL, P, W = 1300e-9, 370e-9, 175e-9
AU, AL, AO, HF, OX, NS = -74 + 6j, -160 + 30j, 3.05, 3.53, 2.5 + 0.5j, 1.447
LAYERS = [(50e-9, AU, 1.0), (1e-9, AO, AO), (7e-9, HF, HF), (5e-9, OX, OX),
          (7e-9, HF, HF), (1e-9, AO, AO), (70e-9, AL, AL)]
R_REF = 0.0571572 + 0.1312207j
F = W / P


def _fix_a(N, how, **kw):
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=N)
    # the equivalent shapes, written with the SAME wall arithmetic segments
    # uses (walls = cumsum / total, centre = mid-wall, size = wall gap)
    xw = np.concatenate([[0.0], np.cumsum([F, 1.0 - F])])
    xw = xw / xw[-1]
    strip = [dict(shape="rectangle", eps=AU,
                  center=(float(0.5 * (xw[0] + xw[1]) * P), 0.5 * P),
                  size=(float((xw[1] - xw[0]) * P), P))]
    for t, es, eg in LAYERS:
        if es == eg:
            st.add_layer(t, eps=es)
        elif how == "segments":
            st.add_layer(t, segments=[(F, es), (1.0 - F, eg)], **kw)
        else:
            st.add_layer(t, shapes=strip, eps_background=eg, **kw)
    return st.set_source(WL, theta=0.0)


def _bits(res):
    o, R, T = res.efficiencies()
    return tuple(np.asarray(a).tobytes() for a in
                 (R, T, res.jones_reflection(), res.jones_transmission()))


def test_touching_rectangles_pass_the_shapes_validation():
    """The premise of the mapping: rectangles that share an edge are legal
    (only a genuine overlap is refused)."""
    shapes = [dict(shape="rectangle", eps=4.0, center=(0.05 * P, P / 2),
                   size=(0.1 * P, P)),
              dict(shape="rectangle", eps=2.25, center=(0.2 * P, P / 2),
                   size=(0.2 * P, P)),
              dict(shape="rectangle", eps=-20 + 1j, center=(0.6 * P, P / 2),
                   size=(0.6 * P, P))]
    _validate_shapes("t", shapes, P, P, n_orders_y=None)


def test_segments_strip_is_the_shapes_strip_bit_for_bit():
    st = _fix_a(30, "segments")
    L = st.layers[0]
    assert L.kind == "shapes"
    assert (L.formulation, L.formulation_requested) == ("li", "auto")
    bg, shapes = L.data
    assert complex(bg) == 1.0 and len(shapes) == 1
    assert _bits(st.solve()) == _bits(_fix_a(30, "shapes").solve())
    for form in ("laurent", "li"):
        a = _fix_a(30, "segments", formulation=form).solve()
        b = _fix_a(30, "shapes", formulation=form).solve()
        assert _bits(a) == _bits(b)


def test_segments_strip_converges_like_the_shapes_strip():
    r = complex(np.asarray(_fix_a(100, "segments").solve()
                           .jones_reflection())[0, 0])
    assert abs(r - R_REF) < 1.2e-2           # measured 1.06e-2


# An asymmetric dielectric cell at oblique incidence: R(+1) and R(-1) differ
# by 1.6e-2 in TE, so a reversed or mirrored layout cannot hide.
_LP, _LWL, _TH, _D = 1.5e-6, 1.0e-6, 0.3, 0.3e-6
_SEG = [(0.2, 4.0), (0.3, 2.25), (0.5, 1.0)]


def _lam_rcwa(segs, N=40):
    st = RCWAStack(_LP, n_superstrate=1.0, n_substrate=1.5, n_orders=N)
    st.add_layer(_D, segments=segs)
    o, R, _T = st.set_source(_LWL, theta=_TH).solve().efficiencies()
    return np.asarray(o), np.asarray(R)


def _lam_pmm(segs, degree=40):
    st = PMMStack(_LP, n_substrate=1.5, n_superstrate=1.0, degree=degree)
    st.add_layer(_D, segments=segs)
    o, R, _T, _J = st.set_source(_LWL, theta=_TH).solve()
    return np.asarray(o), np.asarray(R)


def _pm1(o, R):
    return (R[:, int(np.where(o == 1)[0][0])],
            R[:, int(np.where(o == -1)[0][0])])


def test_layout_matches_pmmstack_order_by_order():
    """Measured 2026-11-14 at N = 40 vs PMM degree 40 (itself converged to
    1.6e-8 against degree 60): |dR| 4.88e-6 (TM) / 3.67e-7 (TE) on the +/-1
    orders, identical on Windows/MKL and WSL/OpenBLAS; bar 2e-5.
    R_TE(+1) - R_TE(-1) = -1.6e-2, 800x the bar."""
    bar = 2e-5
    rp, rm = _pm1(*_lam_rcwa(_SEG))
    pp, pm = _pm1(*_lam_pmm(_SEG))
    assert np.max(np.abs(rp - pp)) < bar
    assert np.max(np.abs(rm - pm)) < bar
    assert abs(pp[1] - pm[1]) > 100 * bar      # the check is sign-sensitive
    # ... and a reversed layout is a different structure
    qp, qm = _pm1(*_lam_rcwa(_SEG[::-1]))
    assert np.max(np.abs(qp - pp)) > 100 * bar


def test_segments_on_a_2d_stack_are_lines_along_y():
    """On a 2-D stack the same segments are y-invariant lines: with the same
    explicit formulation the 2-D solve reproduces the 1-D one (measured
    2.2e-16 in R, 2.5e-15 in the Jones matrix under 'li')."""
    def solve(two_d, form):
        kw = dict(period_y=_LP, n_orders_y=1) if two_d else {}
        st = RCWAStack(_LP, n_superstrate=1.0, n_substrate=1.5, n_orders=10,
                       **kw)
        st.add_layer(_D, segments=_SEG, formulation=form)
        res = st.set_source(_LWL, theta=_TH).solve()
        o, R, _T = res.efficiencies()
        o, R = np.asarray(o), np.asarray(R)
        if o.ndim == 2:
            R = R[:, o[:, 1] == 0]
        return R, np.asarray(res.jones_reflection())
    for form in ("laurent", "li"):
        (R1, J1), (R2, J2) = solve(False, form), solve(True, form)
        assert np.max(np.abs(R1 - R2)) < 1e-12
        assert np.max(np.abs(J1 - J2)) < 1e-12


def test_one_segment_is_a_uniform_layer():
    st = RCWAStack(1e-6, n_orders=3)
    st.add_layer(0.1e-6, segments=[(1.0, 2.25)], formulation="li")
    L = st.layers[0]
    assert (L.kind, L.formulation, L.formulation_requested) == (
        "uniform", "laurent", "li")
    assert complex(L.data) == 2.25


def test_dispersive_segments_pass_through():
    st = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=10)
    st.add_layer(50e-9, segments=[(F, lambda wl: AU), (1 - F, 1.0)])
    st.add_layer(70e-9, eps=AL)
    st.set_source(WL, theta=0.0)
    _o, R, _T, J = st.solve_vs_wavelength([WL])
    ref = RCWAStack(P, n_superstrate=1.0, n_substrate=NS, n_orders=10)
    ref.add_layer(50e-9, segments=[(F, AU), (1 - F, 1.0)])
    ref.add_layer(70e-9, eps=AL)
    res = ref.set_source(WL, theta=0.0).solve()
    assert np.max(np.abs(np.asarray(J)[0]
                         - np.asarray(res.jones_reflection()))) < 1e-12


@pytest.mark.parametrize("segs, match", [
    ([(0.3, 4.0), (0.6, 1.0)], "sum to 1"),
    ([(0.0, 4.0), (1.0, 1.0)], "> 0"),
    ([(-0.2, 4.0), (1.2, 1.0)], "> 0"),
    ([(0.4e-6, 4.0), (0.6e-6, 1.0)], "sum to 1"),       # metres, not fractions
    ([], "at least one"),
    ([(0.5, np.eye(3)), (0.5, 1.0)], "scalar"),
    ([(0.5,), (0.5, 1.0)], "pair"),
])
def test_segments_validation(segs, match):
    with pytest.raises(ValueError, match=match):
        RCWAStack(1e-6, n_orders=3).add_layer(0.1e-6, segments=segs)


def test_segments_is_exclusive_with_other_specs():
    cell = np.ones(16, dtype=complex)
    for kw in (dict(eps_cell=cell), dict(eps=2.0), dict(eps_background=1.0),
               dict(shapes=[])):
        with pytest.raises(ValueError, match="segments"):
            RCWAStack(1e-6, n_orders=3).add_layer(
                0.1e-6, segments=[(0.5, 4.0), (0.5, 1.0)], **kw)
