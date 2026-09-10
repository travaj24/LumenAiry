"""INDEPENDENT verification of ROUND 4 of the pure staggered 2-D PMM
per-layer mortar -- gates for the TWO gaps the verification found.

Verification: ``docs/audits/VERIFY_PMM2D_MORTAR_ROUND4_2026_09_11.md``.
Fix: ``docs/audits/FIX_PMM2D_MORTAR_ROUND4_2026_09_11.md``.

Round 4's own decisions are gated in ``tests/unit/test_fix_pmm2d_mortar_
round4.py`` and were re-measured in this verification without a defect.  Only
the two things that file does NOT gate are added here.

* **G1 -- the axis distinction is never measured against the DAMAGE.**  The
  fix silences the band warning on a CONFORMING axis and keeps it on a
  MORTARED one, and its evidence for "harmless" is that the answer does not
  move on a two-layer stack of UNIFORM permittivity (2.5 beside 3.5) with
  fictitious walls.  Nothing in the shipped suite asks the two-sided question
  on ONE device family: does a band-width segment on the axis the rule KEEPS
  cost accuracy, and does the same width on the axis it SKIPS cost no more
  than a stack with no mortar at all?  Measured here on a genuinely patterned
  stack whose swept wall is still a pure PARTITION choice, in BOTH axis
  orientations -- the mirror (y mortared, a narrow x segment conforming) is
  the direction the fix never measured at all.

* **G2 -- the restated ladder gate asserts its ratio at a rung its new
  precondition does not certify.**  ``test_the_band_the_warning_names_carries
  _a_measurable_cost`` now runs ``M`` = 5 AND 6 and requires
  ``e_band > 1.15 e_ord`` at both, with a precondition that the ORDINARY arm
  is still falling -- ``e_ord(6) < 0.5 e_ord(5)``.  That precondition is read
  AT rung 6; it says nothing about rung 5.  This gate exhibits an independent
  fixture that SATISFIES the precondition by 8.4x and whose ``M`` = 5 ratio is
  **0.473**, i.e. 2.4x BELOW the bar asserted there.  The round-3 gate does
  not fail -- it runs its own fixture -- but the restatement does not
  generalise, which is what S14 asked round 4 to fix.

  **If the gate is later given a per-rung precondition (or drops the ``M`` = 5
  rung), this test fails, and that failure is the gate working: re-pin it
  against the improvement, do not relax it.**

TERMS.  A **MORTARED AXIS** is one on which some adjacent pair of layers
carries different wall arrays; on a CONFORMING axis the cross-mass is the
layer's own mass matrix and the projection is the identity.  A **PURE
PARTITION** wall separates two segments of the SAME permittivity: the device
cannot depend on where it sits, so ANY movement of the answer as its segment
narrows is numerical damage rather than physics.

EVERY BAR BELOW IS DERIVED FROM A MEASUREMENT MADE ON **TWO BUILDS**
(2026-09-11), both readings stated in the assertion's comment:

  * WIN -- Windows 11, CPython 3.14.6, numpy 2.4.4, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS/MKL = 1;
  * WSL -- Ubuntu, CPython 3.12.3, numpy 2.4.6, scipy 1.17.1
    (scipy-openblas), OMP/OPENBLAS = 1.

Both link scipy-openblas, so every cross-build spread quoted is a LOWER bound
-- the caveat rounds 2, 3 and 4 and both verifications carry.  Full numbers:
``validation/probe_verify_mortar_round4/v3_ladder_{win,wsl}.json`` (G1) and
``v2_bars_{win,wsl}.json`` (G2).
"""
import os

# The comparisons below read twelve figures of a reflectance; pin ONE BLAS
# thread before numpy is imported.  (In a MULTI-FILE run this is a no-op --
# numpy is already imported -- so the shell must set them too.)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import contextlib  # noqa: E402
import warnings  # noqa: E402

import numpy as np  # noqa: E402

from lumenairy.elements.pmm import PMM2DStackPure, PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack2d_pure as _sp  # noqa: E402
from lumenairy.elements.pmm import twod_staggered as _ts  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import _C  # noqa: E402

# ---- THIS verification's own geometry, in METRES --------------------------
# Disjoint from the fix's fixtures and from the round-3 verification's:
# different period, wavelength, incidence, substrate, contrast and walls.
# Driving a metre-scale geometry with a period-1 wavelength (or the reverse)
# makes k0 wrong by 1e6 and asks for a ~1e5-node Gauss-Legendre rule.
_P, _WL, _TH, _PHI = 1.24e-6, 0.905e-6, 0.29, 0.71
_EPS_H, _EPS_P1, _EPS_P2 = 2.31, 6.76, 5.29
_XW0, _XW2, _XORD = (0.17, 0.61), (0.29, 0.74), (0.335, 0.485)
_FIXED = (0.31, 0.68)          # the ORDINARY wall pair on the untouched axis
_CEN, _CEN_CONF = 0.41, 0.52   # centres the swept pair straddles
_TH_LAYERS = (0.128e-6, 0.111e-6, 0.097e-6)

#: the widest and the narrowest rung of the swept width, as fractions of the
#: period.  1.2e-3 is INSIDE the band (above the 1e-3 width contract); 3e-1 is
#: an entirely ordinary partition.
_WIDE, _NARROW = 3.0e-1, 1.2e-3


def _walls(fracs):
    return [float(f) * _P for f in fracs]


def _pair(g, centre):
    return _walls((centre - 0.5 * g, centre + 0.5 * g))


def _tile(axis, which):
    """A 3x3 tile that depends on ONE axis only, so every wall on the OTHER
    axis is a pure PARTITION choice."""
    c = np.full((3, 3), _C(_EPS_H))
    if which:
        e = _C(_EPS_P1 if which == 1 else _EPS_P2)
        if axis == "x":
            c[1, :] = e
        else:
            c[:, 1] = e
    return c


def _stack(kind, g):
    """Three layers, the middle one UNIFORM.

    ``Xmort``    pattern x-only, layers on DIFFERENT x walls, the swept pair
                 is the middle layer's own x walls -- the swept axis IS
                 mortared.
    ``Xconf_y``  pattern x-only, layers on different x walls, the swept pair
                 is the y wall array SHARED by all three -- the swept axis
                 CONFORMS while the other carries the mortar.
    ``Xnomort``  pattern x-only, ALL layers on the SAME x walls (no mortar
                 anywhere), the swept pair is the shared y wall array.
    ``Ymort`` / ``Yconf_x``  the transposes.
    """
    st = PMM2DStackPure(_P, n_modes=4, n_orders=1, n_superstrate=1.0,
                        n_substrate=1.45, layer_grids="per-layer")
    if kind.startswith("X"):
        ax = "x"
        other = (_pair(g, _CEN_CONF) if kind in ("Xconf_y", "Xnomort")
                 else _walls(_FIXED))
        x0, x2 = _walls(_XW0), _walls(_XW2)
        x1 = _pair(g, _CEN) if kind == "Xmort" else _walls(_XORD)
        if kind == "Xnomort":
            x0 = x1 = x2 = _walls(_XW0)
        walls = ((x0, other), (x1, other), (x2, other))
    else:
        ax = "y"
        other = _pair(g, _CEN_CONF) if kind == "Yconf_x" else _walls(_FIXED)
        y1 = _pair(g, _CEN) if kind == "Ymort" else _walls(_XORD)
        walls = ((other, _walls(_XW0)), (other, y1), (other, _walls(_XW2)))
    for t, (xw, yw), which in zip(_TH_LAYERS, walls, (1, 0, 2)):
        st.add_layer(t, eps_cell=_tile(ax, which), x_walls=xw, y_walls=yw)
    st.set_source(_WL, theta=_TH, phi=_PHI)
    return st


@contextlib.contextmanager
def _round3_rule():
    """FAIL-BEFORE arm: round 3's condition -- ask the STACK, then scan BOTH
    axes -- restored in process, so both arms run in one interpreter with
    everything but the change held fixed.  ``StagGridOps.key`` is the
    fingerprint PAIR, so ``len({key}) > 1`` is exactly ``any(per-axis pair)``,
    which is what makes the collapsed form reproduce round 3 EXACTLY."""
    orig = _ts._stag_mortared_axes

    def collapsed(grids, force=False):
        m = bool(force) or len({g.key() for g in grids}) > 1
        return (m, m)

    _ts._stag_mortared_axes = collapsed
    _sp._stag_mortared_axes = collapsed
    try:
        yield
    finally:
        _ts._stag_mortared_axes = orig
        _sp._stag_mortared_axes = orig


def _solve(st):
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")
        o, R, T = st.solve(jones=False)
    band = [str(w.message) for w in ws if "degradation band" in str(w.message)]
    axes = ["x" if " on the x axis" in m else
            "y" if " on the y axis" in m else None for m in band]
    return (np.atleast_2d(R), np.atleast_2d(T), len(band), axes)


def _move(kind, g=_NARROW):
    """How far the answer moves from the WIDEST rung when the swept segment is
    ``g`` of the period.  The swept wall is a pure partition choice, so the
    exact answer is the same at both rungs and the whole move is damage."""
    Rw, Tw, n_w, ax_w = _solve(_stack(kind, _WIDE))
    Rn, Tn, n_n, ax_n = _solve(_stack(kind, g))
    move = float(max(np.max(np.abs(Rn - Rw)), np.max(np.abs(Tn - Tw))))
    return {"move": move, "n_warn_wide": n_w, "n_warn": n_n, "axes": ax_n}


def test_the_axis_the_rule_skips_is_harmless_and_the_one_it_keeps_is_not():
    """G1.  The per-axis rule's DECISION, measured against the DAMAGE, on one
    device family and in BOTH axis orientations.

    The swept wall separates two segments of the SAME permittivity, so the
    exact answer cannot depend on it and the answer's move is entirely
    numerical.  MEASURED 2026-09-11, ``M`` = 4, the narrowest rung 1.2e-3 of
    the period (``validation/probe_verify_mortar_round4/v3_ladder_*.json``);
    WIN and WSL agree to four significant figures on every entry:

    ==========  ===========  ===========  ==============================
    shape       WIN move     WSL move     what the rule does
    ==========  ===========  ===========  ==============================
    ``Xmort``   7.610e-03    7.610e-03    warns, names **x**
    ``Xconf_y`` 2.370e-07    2.370e-07    silent
    ``Xnomort`` 1.543e-07    1.543e-07    silent (no mortar at all)
    ``Ymort``   7.228e-03    7.228e-03    warns, names **y**
    ``Yconf_x`` 1.108e-07    1.108e-07    silent
    ==========  ===========  ===========  ==============================

    So the axis the rule KEEPS costs 7.6e-03 / 7.2e-03 -- four decades over
    the floor -- while the axis it SKIPS costs 2.4e-07 / 1.1e-07, which is
    within 1.6x of a stack that builds NO mortar anywhere.  At ``M`` = 6 the
    separation widens rather than narrows (1.731e-02 against 4.391e-10, the
    mortared arm GROWING with the modal count and the conforming arm FALLING
    with it), which is the signature of a floor beside a discretisation.

    This is the two-sided form the fix's own evidence lacks: its DEFECT-2
    fixture is a UNIFORM slab pair, on which the mortared axis moves only
    5.6e-07 (``M`` = 5) and the conforming one 9.8e-13 -- the right decision
    on a device that barely exercises it.
    """
    mort_x, conf_y = _move("Xmort"), _move("Xconf_y")
    nomort = _move("Xnomort")
    mort_y, conf_x = _move("Ymort"), _move("Yconf_x")

    # (1) the DECISIONS.  An ordinary (3e-1) partition never warns; the band
    # width warns exactly on the MORTARED axis and names it.
    for r in (mort_x, conf_y, nomort, mort_y, conf_x):
        assert r["n_warn_wide"] == 0, r
    assert (mort_x["n_warn"], mort_x["axes"]) == (1, ["x"]), mort_x
    assert (mort_y["n_warn"], mort_y["axes"]) == (1, ["y"]), mort_y
    for r in (conf_y, nomort, conf_x):
        assert r["n_warn"] == 0, r

    # (2) the axis the rule KEEPS really is damaged.  MEASURED 7.610e-03 (x)
    # and 7.228e-03 (y) on both builds; asserted at 1e-04, i.e. 72x of margin
    # on a quantity whose cross-build spread is below 1e-03 RELATIVE.
    assert mort_x["move"] > 1.0e-4, mort_x
    assert mort_y["move"] > 1.0e-4, mort_y

    # (3) the axis it SKIPS is not.  Stated as a RATIO against the mortared
    # arm of the SAME family, so both sides are re-measured every run:
    # MEASURED 2.370e-07 / 7.610e-03 = 3.1e-05 and 1.108e-07 / 7.228e-03 =
    # 1.5e-05; asserted at 1e-03, i.e. 32x of margin on the worse of the two.
    assert conf_y["move"] < 1.0e-3 * mort_x["move"], (conf_y, mort_x)
    assert conf_x["move"] < 1.0e-3 * mort_y["move"], (conf_x, mort_y)

    # (4) and it is not merely small -- it is what a stack with NO mortar
    # anywhere costs.  MEASURED 2.370e-07 against the no-mortar control's
    # 1.543e-07 = 1.54x; asserted at 5x, i.e. 3.2x of margin.  This is the
    # comparison that separates "the rule skips a harmless axis" from "this
    # device is harmless", and it is the one the fix's fixture cannot make.
    assert conf_y["move"] < 5.0 * nomort["move"], (conf_y, nomort)

    # (5) FAIL-BEFORE: under round 3's rule both conforming arms DO warn, and
    # they name the conforming axis -- the false positive this fix removes.
    with _round3_rule():
        r3_conf_y, r3_conf_x = _solve(_stack("Xconf_y", _NARROW)), _solve(
            _stack("Yconf_x", _NARROW))
    assert (r3_conf_y[2], r3_conf_y[3]) == (1, ["y"]), r3_conf_y
    assert (r3_conf_x[2], r3_conf_x[3]) == (1, ["x"]), r3_conf_x


# ==========================================================================
# G2 -- the ladder gate's new precondition does not certify its new rung
# ==========================================================================
_LP, _LWL, _LTH = 1.19e-6, 0.83e-6, 0.21
_LEPS_H, _LEPS_P = 2.25, 6.76
_LW0, _LW2, _LYW, _LT, _LXC = (0.17, 0.61), (0.29, 0.74), (0.31, 0.68), \
    0.128e-6, 0.41


def _oracle(deg):
    """The EXACT 1-D ``PMMStack`` -- a different assembly with no mortar and
    no element-grid partition.  The 2-D device below is INVARIANT IN Y and
    driven at ``phi`` = 0, so this is an INDEPENDENT truth for it, and the
    middle layer is uniform host, so that truth does not move with the swept
    width."""
    st = PMMStack(_LP, degree=deg, far_field_orders=5)
    st.add_layer(_LT, segments=[(_LW0[0], _LEPS_H),
                                (_LW0[1] - _LW0[0], _LEPS_P),
                                (1.0 - _LW0[1], _LEPS_H)])
    st.add_layer(_LT, segments=[(1.0, _LEPS_H)])
    st.add_layer(_LT, segments=[(_LW2[0], _LEPS_H),
                                (_LW2[1] - _LW2[0], _LEPS_P),
                                (1.0 - _LW2[1], _LEPS_H)])
    st.set_source(_LWL, theta=_LTH)
    o, R, T = st.solve(stabilize=None)[:3]
    return np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)


def _ltile(pillar):
    c = np.full((3, 3), _C(_LEPS_H))
    if pillar:
        c[1, :] = _C(_LEPS_P)
    return c


def _lerr(frac, M, o_ref, R_ref, T_ref):
    st = PMM2DStackPure(_LP, n_modes=M, n_orders=1, n_superstrate=1.0,
                        n_substrate=1.0, layer_grids="per-layer")
    yws = [_LYW[0] * _LP, _LYW[1] * _LP]
    st.add_layer(_LT, eps_cell=_ltile(True),
                 x_walls=[_LW0[0] * _LP, _LW0[1] * _LP], y_walls=yws)
    st.add_layer(_LT, eps_cell=_ltile(False),
                 x_walls=[(_LXC - frac / 2) * _LP, (_LXC + frac / 2) * _LP],
                 y_walls=yws)
    st.add_layer(_LT, eps_cell=_ltile(True),
                 x_walls=[_LW2[0] * _LP, _LW2[1] * _LP], y_walls=yws)
    st.set_source(_LWL, theta=_LTH, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T = st.solve(jones=False)
    o, R, T = np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)
    worst = 0.0
    for m in (-1, 0, 1):
        sel = int(np.where((o[:, 0] == m) & (o[:, 1] == 0))[0][0])
        j = int(np.where(np.asarray(o_ref) == m)[0][0])
        worst = max(worst, abs(float(R[1, sel]) - float(R_ref[1, j])),
                    abs(float(T[1, sel]) - float(T_ref[1, j])))
    return worst


def test_the_ordinary_arm_precondition_does_not_certify_the_m5_rung():
    """G2.  A stack that SATISFIES the restated gate's new precondition and
    still reads a ratio far below the bar the gate asserts at ``M`` = 5.

    ``tests/unit/test_fix_pmm2d_mortar_round3.py::test_the_band_the_warning_
    names_carries_a_measurable_cost`` was restated in round 4 to run ``M`` = 5
    AND 6 and require ``e_band > 1.15 e_ord`` at BOTH, guarded by a NEW
    precondition ``e_ord(6) < 0.5 e_ord(5)`` -- "the ordinary arm is still
    falling, so the baseline is a measurement and not a floor".  That
    precondition is read AT rung 6.  It certifies nothing about rung 5, and
    rung 5 is the one the restatement ADDED.

    MEASURED 2026-09-11 on this file's own fixture (period 1.19e-6, wavelength
    0.83e-6, a y-invariant three-layer grating whose middle layer is uniform
    host, oracle = the exact 1-D ``PMMStack`` at degree 12 and 14, self-gap
    1.1930e-06).  WIN and WSL agree to THIRTEEN significant figures -- it is a
    pure discretisation quantity:

    ====  ============  ============  ======
    ``M``  ``e_ord``     ``e_band``    ratio
    ====  ============  ============  ======
    5     1.647715e-01  7.799080e-02  0.4733
    6     9.780604e-03  1.456122e-02  1.4888
    7     2.499662e-03  7.225423e-03  2.8906
    ====  ============  ============  ======

    The ordinary arm falls 16.85x from ``M`` = 5 to 6, so the precondition
    passes with **8.4x** of margin -- and the ``M`` = 5 ratio is **0.4733**,
    2.4x BELOW the 1.15 the gate asserts there.  The round-3 gate does not
    fail: it runs its own fixture, where ``M`` = 5 happens to read 1.611.
    What is refuted is that adding a second rung made the bar durable; the
    added rung is the LESS converged one and carries no precondition of its
    own.  ``M`` = 7 is not run here (121 s a rung in the probe).

    **If the gate is later given a per-rung precondition, or drops the
    ``M`` = 5 rung, this test fails -- and that failure is the gate working.
    Re-pin it against the improvement, do not relax it.**
    """
    o14, R14, T14 = _oracle(14)
    o12, R12, T12 = _oracle(12)
    keep = np.abs(np.asarray(o14)) <= 1
    self_gap = float(max(np.max(np.abs(R12[:, keep] - R14[:, keep])),
                         np.max(np.abs(T12[:, keep] - T14[:, keep]))))
    e_ord5 = _lerr(3.0e-1, 5, o14, R14, T14)
    e_band5 = _lerr(3.0e-3, 5, o14, R14, T14)
    e_ord6 = _lerr(3.0e-1, 6, o14, R14, T14)

    # (1) the ORACLE is far better than the difference being read.  MEASURED
    # self-gap 1.1930e-06 against |e_band(5) - e_ord(5)| = 8.678e-02 = 7.3e+04
    # x; asserted at 20x, the same shape the round-3 gate uses.
    assert self_gap < 0.05 * abs(e_band5 - e_ord5), (self_gap, e_ord5, e_band5)

    # (2) this fixture SATISFIES the gate's new precondition -- and by a wide
    # margin.  MEASURED e_ord(6) / e_ord(5) = 5.9359e-02 on both builds
    # against the shipped 0.5; asserted at 0.25, i.e. 4.2x of margin.
    assert e_ord6 < 0.25 * e_ord5, (e_ord6, e_ord5)

    # (3) and yet the rung the restatement ADDED reads far below its bar.
    # MEASURED e_band(5) / e_ord(5) = 0.4733 on both builds against the
    # shipped 1.15; asserted at 0.9, i.e. 1.9x of margin.  A precondition read
    # at rung 6 does not certify rung 5.
    assert e_band5 < 0.9 * e_ord5, (e_band5, e_ord5, e_ord6, self_gap)
