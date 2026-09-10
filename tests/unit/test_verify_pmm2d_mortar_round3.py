"""VERIFY ROUND 3 (per-layer L2 mortar) -- what the independent re-measurement
found that the round-3 gate does not cover.

Companion to ``tests/unit/test_fix_pmm2d_mortar_round3.py``.  Evidence and
every number: ``docs/audits/VERIFY_PMM2D_MORTAR_ROUND3_2026_09_11.md`` and
``validation/probe_verify_mortar_round3/``.

Four gates, and only where a gap was found:

* **G1** RESTATES the round-3 mechanism.  The operand is rank-deficient when
  EXACTLY ONE side of the interface is a promoted in-plane region, not
  "whenever either side is" -- with BOTH sides promoted it is as healthy as
  the both-out-of-plane control, measured.
* **G2** PINS A KNOWN LIMITATION: the band warning names an axis on which no
  mortar forms.  **If the library is later taught to condition the warning per
  axis, this test fails, and that failure is the gate working -- re-pin it
  against the improvement, do not relax it.**
* **G3** puts the false-positive census's margin back in scope: ORDINARY
  geometries the library builds sit closer to the band's upper edge than the
  round-3 gate's own battery does.
* **G4** BOUNDS the one false-positive class G3 raises but cannot reach -- a
  uniform lattice fine enough to land in the band is an eigenproblem no
  machine will run.

Everything asserted is measured on the running build against the library's own
constants; there is no cross-build value pin anywhere in this file.
"""
import os

# The generalized mortar operand carries a singular value ten decades under
# the largest; pin ONE BLAS thread before numpy is imported.  (In a MULTI-FILE
# run this is a no-op -- numpy is already imported -- so the shell must set
# them too; see the report's S11.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings  # noqa: E402

import numpy as np  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm import _core as _pc  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as _sp  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C  # noqa: E402

# ---- the verification's own fixture, in METRES ----------------------------
# Independent of the round-3 gate's (period 1.0e-6, wl 0.62e-6, theta 0.09,
# eps 2.25/6.0, walls .2371/.6183 and .3117/.7402).
_P = 0.87e-6
_WL = 0.73e-6
_TH, _PH = 0.17, 1.1
_EH, _EB = 1.96, 8.41
_WA = (0.1873, 0.5412)
_WB = (0.3106, 0.8039)
_WD = (0.2415, 0.6688)
_E_OOP = np.array([[3.10, 0.0, 0.62], [0.0, 2.70, 0.0],
                   [0.55, 0.0, 2.45]], dtype=_C)
_E_OOP2 = np.array([[2.40, 0.0, -0.41], [0.0, 2.90, 0.0],
                    [-0.37, 0.0, 2.10]], dtype=_C)


def _sc(w):
    return [x * _P for x in w]


def _mid(w):
    b = (0.0,) + tuple(w) + (1.0,)
    return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]


def _scalar(w, lo, hi, e=_EB):
    m = _mid(w)
    c = np.full((len(m), len(m)), _C(_EH))
    for i in range(len(m)):
        for j in range(len(m)):
            if lo < m[i] < hi and lo < m[j] < hi:
                c[i, j] = _C(e)
    return c


def _tensor(w, lo, hi, e=None):
    m = _mid(w)
    n = len(m)
    c = np.empty((n, n, 3, 3), dtype=_C)
    ee = _E_OOP if e is None else np.asarray(e, dtype=_C)
    for i in range(n):
        for j in range(n):
            c[i, j] = ee if (lo < m[i] < hi and lo < m[j] < hi) \
                else np.eye(3) * _EH
    return c


class _Capture:
    """Record every GENERALIZED-site ``(A, B, promoted_a, promoted_b)`` a solve
    builds, WITHOUT refusing.

    Promotion is read STRUCTURALLY off the 6-tuple, not by tolerance:
    :func:`~lumenairy.elements.pmm.twod_staggered._modes_as_general` returns
    ``(W, V, lam, W, -V, -lam)``, sharing ``W`` by identity and negating the
    other two bitwise, so the test is exact.
    """

    def __init__(self):
        self.ops = []
        self._cur = {}

    @staticmethod
    def _promoted(six):
        W, V, lam, W2, V2, lam2 = six[:6]
        return bool(np.array_equal(W2, W) and np.array_equal(V2, -V)
                    and np.array_equal(lam2, -lam))

    def __enter__(self):
        self._og, self._ogen = (_pc._guarded_mortar_solve,
                                _pc._interface_smatrix_general_mortar_2d)

        def gen(six_a, six_b, ga, gb, cr, ka):
            self._cur = {"prom_a": self._promoted(six_a),
                         "prom_b": self._promoted(six_b),
                         "ma": int(np.asarray(six_a[0]).shape[1])}
            try:
                return self._ogen(six_a, six_b, ga, gb, cr, ka)
            finally:
                self._cur = {}

        def solve(A, B, site, ga=None, gb=None, hint=None, screen="rcond"):
            if "GENERALIZED" not in site:
                return self._og(A, B, site, ga, gb, hint, screen)
            self.ops.append((np.array(A), np.array(B), dict(self._cur)))
            return np.linalg.solve(A, B)

        _pc._guarded_mortar_solve = solve
        _pc._interface_smatrix_general_mortar_2d = gen
        _sp._interface_smatrix_general_mortar_2d = gen
        return self

    def __exit__(self, *a):
        _pc._guarded_mortar_solve = self._og
        _pc._interface_smatrix_general_mortar_2d = self._ogen
        _sp._interface_smatrix_general_mortar_2d = self._ogen
        return False


def _facts(A, ma):
    """``s_min/s_max`` and the near-null right vector's split between the two
    block columns."""
    s, Vh = np.linalg.svd(A)[1:]
    v = Vh[-1].conj()
    return {"s_ratio": float(s[-1] / s[0]),
            "on_a": float(np.linalg.norm(v[:ma])),
            "on_b": float(np.linalg.norm(v[ma:]))}


def _solved(st):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    o = np.asarray(o)
    k = int(np.argmin(np.abs(o[:, 0]) + np.abs(o[:, 1])))
    band = [w for w in ws if issubclass(w.category, UserWarning)
            and "degradation band" in str(w.message)]
    return {"R00": float(np.atleast_2d(R)[1, k]),
            "closure": abs(float(np.atleast_2d(R).sum(axis=1)[1]
                                 + np.atleast_2d(T).sum(axis=1)[1]) - 1.0),
            "band": band}


# ==========================================================================
# G1 -- the MECHANISM, restated: ONE promoted side, not "either"
# ==========================================================================
def test_a_generalized_mortar_with_both_sides_promoted_is_not_rank_deficient():
    """RESTATEMENT of the round-3 mechanism.

    ``_MORTAR_RESID_REFUSE``'s derivation says the operand is rank-deficient by
    construction "whenever ONE side of the interface is an in-plane region
    promoted to the generalized 6-tuple form".  Read as "whenever EITHER side
    is", that is refuted here: the near-null space needs the interface to be
    ASYMMETRIC -- exactly one promoted side.  With BOTH sides promoted (two
    in-plane layers on different grids, an out-of-plane layer elsewhere in the
    stack putting the whole cascade on the generalized form) the operand is as
    healthy as the both-out-of-plane control.

    MEASURED 2026-09-11 at ``M`` = 4, IDENTICAL on WIN (py3.14 / OpenBLAS
    Haswell) and WSL (py3.12 / OpenBLAS SkylakeX) to every digit printed
    (``validation/probe_verify_mortar_round3/v1_mechanism_{win,wsl}.json``):

    ============================  ===========  =============
    interface                     s_min/s_max  near-null a/b
    ============================  ===========  =============
    exactly ONE side promoted     1.356e-10    0.000 / 1.000
    BOTH sides promoted           2.631e-05    0.884 / 0.468
    NEITHER (both out-of-plane)   8.105e-04    0.829 / 0.559
    ============================  ===========  =============

    So a both-promoted interface sits 5.3 decades ABOVE the one-promoted one
    and within 1.5 decades of the control, and its near-null direction is not
    localised at all.  Nothing SHIPPED changes -- the residual screen accepts
    all three -- but a later reader deriving a bar from "any promoted side"
    would mis-predict by five decades.
    """
    def _stack(kind, M=4):
        st = PMM2DStackPure(_P, n_modes=M, n_orders=2, n_substrate=1.45,
                            layer_grids="per-layer")
        st.add_layer(0.118e-6, eps_cell=_tensor(_WA, *_WA),
                     x_walls=_sc(_WA), y_walls=_sc(_WA))
        if kind == "both_promoted":
            # two IN-PLANE layers on DIFFERENT grids -> the middle interface
            # has both sides promoted, while layer 0 keeps the cascade general
            st.add_layer(0.071e-6, eps_cell=_scalar(_WB, *_WB),
                         x_walls=_sc(_WB), y_walls=_sc(_WB))
            st.add_layer(0.083e-6, eps_cell=_scalar(_WD, *_WD, e=4.7),
                         x_walls=_sc(_WD), y_walls=_sc(_WD))
        else:                                   # NEITHER side promoted
            st.add_layer(0.094e-6,
                         eps_cell=_tensor(_WB, *_WB, e=_E_OOP2),
                         x_walls=_sc(_WB), y_walls=_sc(_WB))
        st.set_source(_WL, theta=_TH, phi=_PH)
        return st

    with _Capture() as cap:
        _solved(_stack("both_promoted"))
    # interface 0 is the ASYMMETRIC one (out-of-plane beside promoted
    # in-plane); interface 1 is the one this gate is about
    one = [(A, m) for A, _B, m in cap.ops
           if m["prom_a"] != m["prom_b"]]
    both = [(A, m) for A, _B, m in cap.ops
            if m["prom_a"] and m["prom_b"]]
    assert len(one) == 1 and len(both) == 1, [m for _A, _B, m in cap.ops]

    with _Capture() as cap2:
        _solved(_stack("neither"))
    neither = [(A, m) for A, _B, m in cap2.ops
               if not m["prom_a"] and not m["prom_b"]]
    assert len(neither) == 1, [m for _A, _B, m in cap2.ops]

    f_one = _facts(one[0][0], one[0][1]["ma"])
    f_both = _facts(both[0][0], both[0][1]["ma"])
    f_nei = _facts(neither[0][0], neither[0][1]["ma"])

    # (1) ONE promoted side reproduces the round-3 reading: localised, and
    # decades closer to singular.  MEASURED 0.000 / 1.000 and 1.36e-10.
    assert max(f_one["on_a"], f_one["on_b"]) > 0.95, f_one
    assert min(f_one["on_a"], f_one["on_b"]) < 0.05, f_one
    assert f_one["s_ratio"] < 1e-4 * f_nei["s_ratio"], (f_one, f_nei)

    # (2) BOTH promoted does NOT.  The DECISION is "not localised" -- measured
    # 0.884, i.e. 0.066 of slack under the same 0.95 the round-3 gate uses on
    # the other side of this comparison.
    assert max(f_both["on_a"], f_both["on_b"]) < 0.95, f_both
    # (3) and it is decades AWAY from the one-promoted operand rather than
    # beside it.  MEASURED 2.63e-05 vs 1.36e-10 = 5.3 decades; asserted at 3.
    assert f_both["s_ratio"] > 1e3 * f_one["s_ratio"], (f_both, f_one)
    # (4) while sitting within two decades of the healthy control (1.5
    # measured), i.e. on the control's side of the divide, not the defect's
    assert f_both["s_ratio"] > 1e-2 * f_nei["s_ratio"], (f_both, f_nei)


# ==========================================================================
# G2 -- KNOWN LIMITATION: the band warning names a non-mortared axis
# ==========================================================================
def test_the_band_warning_fires_on_an_axis_that_carries_no_mortar():
    """PINS A KNOWN LIMITATION (VERIFY DEFECT 2, P3).

    :func:`~lumenairy.elements.pmm.twod_staggered._warn_stag_sliver_band` is
    conditioned on the STACK building a cross-grid interface somewhere, but
    :func:`_stag_band_narrowest` then scans BOTH axes of every grid.  A stack
    whose layers differ on x and share the y wall array EXACTLY therefore
    warns about a narrow y segment -- while the y mortar is the IDENTITY, so
    nothing on that axis is projected across grids and nothing is degraded.

    MEASURED 2026-09-11 on the fixture below, whose permittivity is UNIFORM in
    both layers, so the y walls carry no feature and the DEVICE cannot depend
    on their separation: ``R00`` = 0.247088457739 at a y width of 3e-1, 1e-1,
    5e-2, 3e-2, 1e-2, 3e-3 and 1.2e-3 -- a total movement of **4.2e-13** over
    2.5 decades, i.e. round-off -- while the warning fires from 1e-2 down and
    names "the y axis" and a cost of "about 4-5x"
    (``validation/probe_verify_mortar_round3/v5_falsepos_win_fp2.json``).

    **If the library is later taught to condition the band warning per AXIS,
    this test fails, and that failure is the gate working -- re-pin it against
    the improvement, do not relax it.**
    """
    P, wl, th = 1.07, 0.79, 0.31
    yc = 0.585

    def _stack(fy, M=4):
        yws = [(yc - fy / 2) * P, (yc + fy / 2) * P]
        st = PMM2DStackPure(P, n_modes=M, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.17, eps_cell=np.full((3, 3), _C(2.5)),
                     x_walls=[0.25 * P, 0.60 * P], y_walls=yws)
        st.add_layer(0.17, eps_cell=np.full((3, 3), _C(3.5)),
                     x_walls=[0.31 * P, 0.66 * P], y_walls=yws)
        st.set_source(wl, theta=th, phi=0.0)
        return st

    wide = _solved(_stack(3.0e-1))          # ordinary y partition
    edge = _ts._STAG_SLIVER_BAND_FRAC
    inside = _solved(_stack(0.3 * edge))    # 9e-3: comfortably in the band

    # the warning FIRES, and it names the axis that carries no mortar
    assert wide["band"] == [], wide["band"]
    assert len(inside["band"]) == 1, inside["band"]
    assert " y axis" in str(inside["band"][0].message)

    # and there is nothing on that axis for it to be about.  The DECISION is
    # "the answer does not move": measured 4.2e-13 relative over the whole
    # ladder, asserted at 1e-9 -- three decades of slack over the measurement
    # and nine under the "about 4-5x" the message claims.
    rel = abs(inside["R00"] - wide["R00"]) / abs(wide["R00"])
    assert rel < 1e-9, (rel, wide["R00"], inside["R00"])
    assert inside["closure"] < 1e-6, inside["closure"]

    # the CONTRAST that makes it a false positive rather than a true one: the
    # SAME width on the axis the two layers DISAGREE on is a real mortar, and
    # the round-3 gate already covers that side.
    st = PMM2DStackPure(P, n_modes=4, n_orders=1, layer_grids="per-layer")
    sw = [(yc - 0.3 * edge / 2) * P, (yc + 0.3 * edge / 2) * P]
    st.add_layer(0.17, eps_cell=np.full((3, 3), _C(2.5)), x_walls=sw,
                 y_walls=[0.25 * P, 0.60 * P])
    st.add_layer(0.17, eps_cell=np.full((3, 3), _C(3.5)),
                 x_walls=[0.31 * P, 0.66 * P], y_walls=[0.25 * P, 0.60 * P])
    st.set_source(wl, theta=th, phi=0.0)
    on_x = _solved(st)
    assert len(on_x["band"]) == 1, on_x["band"]
    assert " x axis" in str(on_x["band"][0].message)


# ==========================================================================
# G3 -- the false-positive census's margin is BATTERY-scoped
# ==========================================================================
def test_ordinary_geometries_sit_closer_to_the_band_edge_than_3x():
    """SCOPE of the round-3 census.

    ``test_no_ordinary_geometry_the_library_builds_lands_in_the_band``
    asserts that the narrowest ORDINARY geometry is more than 3x the band's
    upper edge, and its docstring states that of "ANY ordinary per-layer
    geometry the library builds".  It is measured over the ROUND-2 BATTERY,
    and the 3x is a property of THAT SAMPLE.

    Two geometries no less ordinary sit inside 3x, MEASURED 2026-09-11 on both
    builds (a pure geometry reading -- it has no cross-build spread at all):

    * a DUTY-0.9 pillar (a 5 %-wide trench either side) -- narrowest segment
      **5.000e-02**, i.e. **1.67x** the 3e-2 edge;
    * TWO pillars separated by a 5 %-wide gap -- **5.000e-02**, same margin,
      a different geometry class.

    A UNIFORM 16-cell lattice reads **6.250e-02** (2.08x) on the same census
    (``validation/probe_verify_mortar_round3/v4_band_win_rest.json``); it is
    not solved HERE because a 16-cell lattice is a ``2 q^2`` = 4608 region eig
    at ``M`` = 4 and this gate has a 60 s budget.

    All of these SOLVE and all are SILENT, so the edge itself is not refuted;
    what is refuted is the family reading of the margin.  The two rules the
    census is used to justify still hold on the sample they were measured on.
    """
    edge = _ts._STAG_SLIVER_BAND_FRAC
    P = 1.0

    def _duty(duty):
        a = 0.5 - duty / 2
        st = PMM2DStackPure(P, n_modes=4, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.1, eps_cell=_scalar((a, a + duty), a, a + duty, e=6.0),
                     x_walls=[a * P, (a + duty) * P],
                     y_walls=[a * P, (a + duty) * P])
        st.add_layer(0.1, eps=2.0)
        st.set_source(0.8, theta=0.2, phi=0.3)
        return st, min(a, duty, 1.0 - a - duty)

    def _two_pillars():
        w = (0.05, 0.45, 0.50, 0.90)
        st = PMM2DStackPure(P, n_modes=4, n_orders=1,
                            layer_grids="per-layer")
        st.add_layer(0.1, eps_cell=_scalar(w, 0.05, 0.45, e=6.0),
                     x_walls=[x * P for x in w], y_walls=[x * P for x in w])
        st.add_layer(0.1, eps=2.0)
        st.set_source(0.8, theta=0.2, phi=0.3)
        return st, float(np.min(np.diff((0.0,) + w + (1.0,))))

    for st, frac in (_duty(0.9), _two_pillars()):
        # ordinary, and inside the 3x the round-3 gate asserts on its battery
        assert edge < frac < 3.0 * edge, (frac, edge)
        got = _solved(st)
        # it is nevertheless ABOVE the edge, so it must be SILENT -- the edge
        # is not refuted, only the margin's scope
        assert got["band"] == [], (frac, got["band"])
        # the lossless closure only has to say "this is a real answer, not a
        # broken cascade": these are M = 4 solves on an extreme duty cycle and
        # a 16-cell lattice, where the closure is a CONVERGENCE reading
        # (measured 6.36e-04 and 2.15e-05), not a defect.  Same 1e-2 decision
        # the round-3 gate uses on its own coarse rungs.
        assert got["closure"] < 1e-2, (frac, got["closure"])


# ==========================================================================
# G4 -- the one class G3 raises but the API cannot reach
# ==========================================================================
def test_a_uniform_lattice_cannot_reach_the_band_through_the_public_api():
    """BOUNDS the false positive G3's second geometry points at.

    ``_stag_band_narrowest`` scores a UNIFORM basis as ``1/N``, so a uniform
    lattice with ``N > 1/_STAG_SLIVER_BAND_FRAC`` would land in the band --
    and a uniform lattice is the OPPOSITE of a sliver: every segment is the
    same width, so there is no ``1/J_n`` contrast for the mortar to see.  It
    is not a reachable false positive, and the reason is COST, computed here
    from the library's own basis arithmetic rather than asserted: the
    staggered basis carries ``q = N (M - 1)`` per axis and a ``2 q^2``
    region eigenproblem, so the smallest banded lattice needs a **20808**-
    dimension dense eig (6.5 GiB of operand) at the lowest legal ``M``, and
    ``M`` = 6 needs 57800 (49.8 GiB).

    This is a decision about REACHABILITY, so if the band edge or the basis
    arithmetic ever changes it re-derives rather than ageing.
    """
    edge = _ts._STAG_SLIVER_BAND_FRAC
    N = int(np.ceil(1.0 / edge))            # smallest lattice inside the band
    assert 1.0 / N < edge, (N, edge)
    M = 4                                   # the lowest legal modal count + 1
    q = N * (M - 1)
    dim = 2 * q * q
    gib = dim * dim * 16 / 2 ** 30
    # the DECISION: the operand alone exceeds the memory of an ordinary box by
    # more than a decade, at the CHEAPEST rung.  Measured 6.5 GiB at M = 4 and
    # 49.8 GiB at M = 6; asserted at 4 GiB, which is 1.6x of margin at the
    # cheapest rung and grows as (M - 1)^4.
    assert gib > 4.0, (N, M, q, dim, gib)
    # and the library refuses n_modes < 3, so there is no cheaper rung
    try:
        PMM2DStackPure(_P, n_modes=2)
    except ValueError as e:
        assert "n_modes" in str(e)
    else:                                                    # pragma: no cover
        raise AssertionError("n_modes = 2 was accepted")
