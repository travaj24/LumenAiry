"""V1 / V2 / O2 -- the three defects
``docs/audits/VERIFY_HYBRID_SLANT_TRANSMISSION_ANCHOR_2026_09_11.md`` left open,
and their fixes.

  V1  ``pmm_jones_2d(..., slant=...)`` returned the VERTICAL answer, silently,
      on every one of the SEVEN traced inputs that route it to the jnp twin.
  V2  ``PMMStack``'s TRANSMITTED amplitudes on a sheared stack were
      FRAME-referenced -- the 2-D hybrid's frame-anchor defect, one dimension
      down.
  O2  ``_interface_smatrix_general``'s explicit ``T22`` inverse had no armed
      screen, so ``sum R + T`` up to 6.3e+30 came back with a warning
      indistinguishable from the 17 ordinary 1.03..1.08 truncation warnings.

Evidence: ``docs/audits/FIX_SLANT_ANCHOR_V1_V2_O2_2026_09_11.md``; probes in
``validation/probe_fix_slant_anchor_v1v2o2/`` (both builds, pre- and post-fix).

EVERY numeric bar below is a DECISION with measured room on both sides, per
``docs/TESTING_STANDARDS.md``: the readings quoted in each docstring were
re-measured on Windows py3.14 / numpy 2.4.4 / scipy-openblas AND on WSL
py3.12 / numpy 2.4.6 / scipy-openblas on 2026-09-11, and the bar sits decades
(or, where stated, a stated factor) from both populations.
"""
from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm.stack import (
    PMMStack,
    _anchor_transmission_to_lab_1d,
    _layer_enters_slant_frame_1d,
    _slant_frame_walk_1d,
)
from lumenairy.elements.pmm.twod_jones import (
    _slanted_cell_is_a_frame_noop,
    pmm_jones_2d,
)
from lumenairy.elements.rcwa import _core as _rc

# --------------------------------------------------------------------- V1
V1_PX = V1_PY = 0.85e-6
V1_WL = 0.58e-6
V1_DEPTH = 0.42e-6
V1_NSUP, V1_NSUB = 1.0, 1.5
V1_TSL = 0.5
V1_TH = math.radians(25.0)
V1_NORD, V1_DEG = 3, 5

_V1_BASE = np.array([
    [2.10, 2.10, 1.30, 1.30],
    [3.05, 2.60, 1.30, 1.72],
    [1.30, 1.30, 1.30, 1.30],
    [1.30, 2.44, 2.44, 1.30],
    [1.30, 1.30, 1.95, 1.95],
    [1.30, 1.30, 1.30, 1.30],
], dtype=float)
_V1_LAYOUT = np.arange(_V1_BASE.size).reshape(_V1_BASE.shape)

#: The seven members of ``pmm_jones_2d``'s JAX dispatch tuple, in its own order.
V1_ROUTES = ("eps_tensor_cell", "n_substrate", "n_superstrate", "depth",
             "wavelength", "theta", "phi")


def _tensorify(scal):
    T = np.zeros(np.shape(scal) + (3, 3), dtype=complex)
    for i in range(3):
        T[..., i, i] = scal
    return T


_V1_CELL = _tensorify(_V1_BASE)
_V1_CONST = _tensorify(np.full(_V1_BASE.shape, 2.20))


def _v1_call(**kw):
    a = dict(cell=_V1_CELL, n_sub=V1_NSUB, n_sup=V1_NSUP, depth=V1_DEPTH,
             wl=V1_WL, theta=V1_TH, phi=0.0, slant=(V1_TSL, 0.0))
    a.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pmm_jones_2d(V1_PX, V1_PY, a["cell"], a["n_sub"], a["n_sup"],
                            a["depth"], a["wl"], theta=a["theta"],
                            phi=a["phi"], n_orders=V1_NORD, degree=V1_DEG,
                            slant=a["slant"], region_layout=_V1_LAYOUT)


def _v1_traced_kw(route, jnp):
    return {"eps_tensor_cell": dict(cell=jnp.asarray(_V1_CELL)),
            "n_substrate": dict(n_sub=jnp.asarray(V1_NSUB)),
            "n_superstrate": dict(n_sup=jnp.asarray(V1_NSUP)),
            "depth": dict(depth=jnp.asarray(V1_DEPTH)),
            "wavelength": dict(wl=jnp.asarray(V1_WL)),
            "theta": dict(theta=jnp.asarray(V1_TH)),
            "phi": dict(phi=jnp.asarray(0.0))}[route]


def _jnp():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jax, jnp


def _dmax(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def _sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()


def test_v1_all_seven_traced_routes_refuse_a_slanted_patterned_cell():
    """FAIL-BEFORE, executed here as a two-sided statement rather than quoted.

    ``pmm_jones_2d``'s JAX branch never read ``slant``: it handed off to
    ``_pmm_jones_2d_cell_jax`` without it, and ``_norm_slant_pair`` was not
    reached until AFTER the branch.  MEASURED on the pre-fix tree, all SEVEN
    members of the dispatch tuple, every one reading identically and with ZERO
    warnings:

        vs the NumPy VERTICAL call   dR 1.34e-15 / dT 4.24e-15 / dJones 1.50e-14
        vs the NumPy SLANTED call    dR 3.670e-03 / dT 5.514e-02 / dJones 1.546e-02
        the two NumPy calls          dR 3.670e-03 / dT 5.514e-02 / dJones 1.546e-02

    (WSL: 3.670092143479e-03 / 5.513540231119e-02 / 1.545725310244e-02, the
    same to twelve significant figures.)  The last two rows agreeing is what
    makes it "the vertical answer" rather than "close to it".

    The magnitude the defect had is re-derived HERE, from the two NumPy calls,
    so the test cannot go vacuous if the fixture ever moves: the slant must
    actually change the answer by decades more than the solver's own noise.
    """
    _jax, jnp = _jnp()
    np_sl = _v1_call()
    np_vt = _v1_call(slant=None)
    # the defect had a magnitude, measured on THIS build's own bytes
    gap = _dmax(np_sl[3], np_vt[3])
    assert gap > 1e-3, (
        "the fixture no longer distinguishes slanted from vertical (dJones = "
        f"{gap:.3g}); this test would be vacuous")
    for route in V1_ROUTES:
        with pytest.raises(NotImplementedError, match="SLANTED"):
            _v1_call(**_v1_traced_kw(route, jnp))


def test_v1_the_refusal_names_the_traced_input_and_the_remedy():
    """A refusal that does not say WHICH input routed the call leaves the
    caller to bisect seven arguments.  Two different routes must produce two
    different messages, and both must name the NumPy alternative."""
    _jax, jnp = _jnp()
    msgs = {}
    for route in ("theta", "wavelength"):
        with pytest.raises(NotImplementedError) as ei:
            _v1_call(**_v1_traced_kw(route, jnp))
        msgs[route] = str(ei.value)
        assert route in msgs[route]
        assert "NumPy" in msgs[route]
        assert "z-staircase" in msgs[route]
    assert msgs["theta"] != msgs["wavelength"]
    assert "wavelength" not in msgs["theta"]


def test_v1_the_vertical_control_still_solves_and_matches_numpy():
    """The refusal is about the SHEAR, not about the API: the same traced
    ``depth`` on a VERTICAL cell still solves, and still agrees with the NumPy
    forward.  MEASURED dR 1.34e-15 / dT 4.24e-15 / dJones 1.50e-14 (WIN),
    2.33e-15 / 2.95e-14 / 1.67e-14 (WSL) -- bounded here at 1e-11, three
    decades above the worst reading and eight below the 3.7e-03 defect."""
    _jax, jnp = _jnp()
    got = _v1_call(depth=jnp.asarray(V1_DEPTH), slant=None)
    ref = _v1_call(slant=None)
    assert _dmax(got[1], ref[1]) < 1e-11
    assert _dmax(got[2], ref[2]) < 1e-11
    assert _dmax(got[3], ref[3]) < 1e-11


def test_v1_a_constant_tile_slanted_cell_is_a_measured_no_op_and_still_solves():
    """The decision is "does this cell get SOLVED IN A SHEARED FRAME", not
    "does it carry a slant keyword".  A CONSTANT-valued cell is a shear of a
    uniform medium -- a pure coordinate change -- so the slanted and vertical
    NumPy calls must agree, and the traced call must still be allowed through.

    MEASURED, constant tile at ``slant = (0.5, 0)`` vs ``slant=None``:
    ``dR 4.86e-17 / dT 2.08e-14 / dJones 5.36e-16`` (WIN) and
    ``8.33e-17 / 2.36e-14 / 2.07e-16`` (WSL).  Bounded at 1e-11 -- three
    decades above the worst reading, and nine below the 5.5e-02 ``dT`` the
    PATTERNED cell moves by."""
    _jax, jnp = _jnp()
    assert _slanted_cell_is_a_frame_noop(_V1_CONST) is True
    assert _slanted_cell_is_a_frame_noop(_V1_CELL) is False
    c_sl = _v1_call(cell=_V1_CONST)
    c_vt = _v1_call(cell=_V1_CONST, slant=None)
    assert _dmax(c_sl[1], c_vt[1]) < 1e-11
    assert _dmax(c_sl[2], c_vt[2]) < 1e-11
    assert _dmax(c_sl[3], c_vt[3]) < 1e-11
    got = _v1_call(cell=_V1_CONST, depth=jnp.asarray(V1_DEPTH))
    assert np.all(np.isfinite(np.asarray(got[1])))


def test_v1_the_vertical_traced_solve_still_differentiates():
    """A refusal that also broke the gradient would be a regression.  AD
    against a CENTRAL finite difference on the vertical traced depth: measured
    ``rel 1.52e-08`` (WIN) and ``1.53e-08`` (WSL).  The bar is derived from the
    FD's own error floor rather than from that reading -- with
    ``h = 1e-11 m`` on a 0.42 um depth the truncation + cancellation floor is
    ~``eps * |f| / h`` ~ 1e-6 relative, so 1e-5 is one decade above the oracle's
    own floor and three above the measurement."""
    jax, jnp = _jnp()

    def f(d):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o = pmm_jones_2d(V1_PX, V1_PY, _V1_CELL, V1_NSUB, V1_NSUP, d,
                             V1_WL, theta=V1_TH, phi=0.0, n_orders=V1_NORD,
                             degree=V1_DEG, slant=None,
                             region_layout=_V1_LAYOUT)
        return o[2].sum()

    g = float(jax.grad(f)(jnp.asarray(V1_DEPTH)))
    h = 1e-11
    fd = float((f(jnp.asarray(V1_DEPTH + h)) - f(jnp.asarray(V1_DEPTH - h)))
               / (2 * h))
    assert abs(g) > 1.0, "the gradient is not exercising anything"
    assert abs(g - fd) / abs(fd) < 1e-5


# --------------------------------------------------------------------- V2
V2_P = 0.80e-6
V2_WL = 0.55e-6
V2_D = 0.40e-6
V2_ER, V2_EG = 4.20, 1.45
V2_DUTY = 0.45
V2_SHEAR = 0.30                    # a 0.30-PERIOD walk: neither the half- nor
V2_TH = math.radians(25.0)         # the quarter-period degeneracy
V2_DEG, V2_NORD = 8, 5
V2_W = V2_SHEAR * V2_P


def _v2_stack(**kw):
    kw.setdefault("factorization", "convection")
    kw.setdefault("degree", V2_DEG)
    kw.setdefault("n_orders", V2_NORD)
    return PMMStack(V2_P, n_superstrate=1.0, n_substrate=1.6, **kw)


def _v2_solve(st, theta=V2_TH):
    st.set_source(V2_WL, theta=theta)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st, st.solve()


def _v2_sheared(theta=V2_TH, shear=V2_SHEAR, **kw):
    st = _v2_stack(**kw)
    st.add_sheared_grating(V2_D, eps_ridge=V2_ER, eps_groove=V2_EG,
                           duty=V2_DUTY, shear=shear)
    return _v2_solve(st, theta)


def _v2_stair(ns, theta=V2_TH, shear=V2_SHEAR):
    st = _v2_stack()
    st.add_tapered_grating(V2_D, eps_ridge=V2_ER, eps_groove=V2_EG,
                           duty_bottom=V2_DUTY, duty_top=V2_DUTY,
                           shear=shear, n_slices=ns)
    return _v2_solve(st, theta)


def _amps(st, port="transmission"):
    a = st.per_order_amplitudes(port)
    return (np.asarray(a["orders"]),
            np.stack([np.asarray(a["Ex"]), np.asarray(a["Ey"])], axis=0),
            np.asarray(np.real(a["kx"])))


def _resid(A, B):
    A, B = np.asarray(A), np.asarray(B)
    return float(np.linalg.norm(A - B) / np.linalg.norm(B))


def _align(oa, A, ob, B):
    m = {int(o): j for j, o in enumerate(np.asarray(ob))}
    num = den = 0.0
    for i, o in enumerate(np.asarray(oa)):
        j = m.get(int(o))
        if j is None:
            continue
        num += float(np.sum(np.abs(A[..., i] - B[..., j]) ** 2))
        den += float(np.sum(np.abs(B[..., j]) ** 2))
    return math.sqrt(num) / math.sqrt(den)


def _P(kx, walk, wl=V2_WL):
    return np.exp(1j * (2.0 * np.pi / wl) * np.asarray(kx) * walk)


def test_v2_the_walk_helpers_decide_by_the_generator_not_by_the_keyword():
    """``_layer_enters_slant_frame_1d`` mirrors the routing literal
    ``abs(slant_angle) > 1e-12`` that ``PMMStack.solve`` uses to send a layer
    to the metric generator, so the anchor and the routing cannot drift.

    The 1-D rule DIFFERS from the 2-D one and that is asserted here: a UNIFORM
    (unpatterned) layer DOES enter a frame in 1-D, because
    ``_build_generator_metric`` adds ``tan_conv * Dopx`` with no constant-tile
    short-circuit.  The walk is exact zero on a vertical stack -- an exact
    ``0.0``, which is what short-circuits the anchor -- and the walks ADD, with
    opposite slants cancelling."""
    tan = V2_W / V2_D
    phi = math.atan(tan)
    assert _layer_enters_slant_frame_1d((V2_D, [], 0.0)) is False
    assert _layer_enters_slant_frame_1d((V2_D, [], 1e-13)) is False
    assert _layer_enters_slant_frame_1d((V2_D, [], phi)) is True
    assert _slant_frame_walk_1d([(V2_D, [], 0.0), (0.1e-6, [], 0.0)]) == 0.0
    one = _slant_frame_walk_1d([(V2_D, [], phi)])
    assert abs(one - V2_W) < 1e-18
    two = _slant_frame_walk_1d([(V2_D, [], phi), (V2_D / 2, [], phi)])
    assert abs(two - 1.5 * V2_W) < 1e-18
    assert _slant_frame_walk_1d([(V2_D, [], phi), (V2_D, [], -phi)]) == 0.0


def test_v2_the_anchor_is_the_identity_at_zero_walk_and_is_unimodular():
    """Two structural facts the bit-identity of every non-sheared fixture
    rests on: at ``walk = 0.0`` the anchor returns the dict with the SAME
    array objects (not equal copies -- the same bytes cannot be perturbed if
    they are never touched), and the factor it applies is unimodular for every
    order, so no efficiency can move.  MEASURED ``max ||P_m| - 1|`` over a
    nine-order solve: ``1.11e-16``."""
    tx = np.arange(6, dtype=complex).reshape(2, 3)
    modal = dict(kx=np.array([-0.7, 0.0, 0.9]), tx=tx, ty=tx.copy())
    out = _anchor_transmission_to_lab_1d(modal, V2_WL, 0.0)
    assert out["tx"] is tx
    ph = _P(np.array([-0.7, 0.0, 0.9]), V2_W)
    assert float(np.max(np.abs(np.abs(ph) - 1.0))) < 1e-14
    assert complex(ph[1]) == 1.0 + 0.0j


def test_v2_a_uniform_slanted_film_is_the_vertical_film_of_the_same_eps():
    """THE EXACT ORACLE.  A shear of a HOMOGENEOUS medium is a pure coordinate
    change, so a uniform slanted layer and the vertical film of the same
    ``eps`` are the same physical solid and every lab-referenced observable
    must agree to machine precision.  That makes this an ANALYTIC no-op oracle
    for the anchor rather than a convergence argument.

    MEASURED, ``eps = 2.60``, ``d = 0.40 um``, ``tan(phi) = 0.60``, oblique 25,
    transmission Jones against the vertical film:

        as returned (the FIX)   2.87e-14
        with the anchor removed 1.0950      <- the pre-fix reading
        conjugated              1.8326

    and the REFLECTION Jones reads 1.51e-14 as returned on BOTH trees, which is
    the derivation's statement that the reflection side carries no anchor.
    """
    def film(tan_phi):
        st = _v2_stack()
        st.add_layer(V2_D, eps=2.60, slant_angle=math.atan(tan_phi))
        return _v2_solve(st)

    su, ou = film(V2_W / V2_D)
    sv, ov = film(0.0)
    Ju, Jv = np.asarray(su.jones_transmission()), \
        np.asarray(sv.jones_transmission())
    o_u, A_u, kx = _amps(su)
    o_v, A_v, _ = _amps(sv)
    ph = _P(kx, V2_W)
    shipped = _resid(Ju, Jv)
    unanchored = _resid(Ju / complex(ph[int(np.where(o_u == 0)[0][0])]), Jv)
    assert shipped < 1e-11, f"the no-op oracle reads {shipped:.3g}"
    assert _align(o_u, A_u, o_v, A_v) < 1e-11
    # two-sided: removing the anchor must break it by decades
    assert unanchored / max(shipped, 1e-300) > 1e6
    assert unanchored > 0.5
    # the reflection side never carried it
    assert _resid(np.asarray(ou[3]), np.asarray(ov[3])) < 1e-11
    o_ur, A_ur, _ku = _amps(su, "reflection")
    o_vr, A_vr, _kv = _amps(sv, "reflection")
    assert _align(o_ur, A_ur, o_vr, A_vr) < 1e-9


def _v2_ladder():
    """The shipped / un-anchored / conjugate arms against the engine's OWN
    z-staircase of the identical parallelogram, at three rungs.  Cached at
    module scope: five solves, ~12 s, shared by three tests."""
    if _v2_ladder._cache is None:
        sh, osh = _v2_sheared()
        J = np.asarray(sh.jones_transmission())
        o_s, A_s, kx = _amps(sh)
        ph = _P(kx, V2_W)
        p0 = complex(ph[int(np.where(o_s == 0)[0][0])])
        rows = {}
        prev = None
        for ns in (4, 8, 12):
            stc, oc = _v2_stair(ns)
            Jc = np.asarray(stc.jones_transmission())
            o_c, A_c, _k = _amps(stc)
            rows[ns] = dict(
                J=_resid(J, Jc), J_none=_resid(J / p0, Jc),
                J_conj=_resid(J * np.conj(p0) / p0, Jc),
                amps=_align(o_s, A_s, o_c, A_c),
                amps_none=_align(o_s, A_s / ph, o_c, A_c),
                refl=_resid(np.asarray(osh[3]), np.asarray(oc[3])),
                step=(None if prev is None else _resid(prev, Jc)))
            prev = Jc
        _v2_ladder._cache = rows
    return _v2_ladder._cache


_v2_ladder._cache = None


def test_v2_the_transmission_converges_to_the_lab_referenced_staircase():
    """The oracle is the SAME engine's z-staircase of the identical
    parallelogram, which is LAB-referenced by construction (vertical rungs
    only).  MEASURED, ``n_slices`` 4 / 8 / 12 at degree 8, ``n_orders`` 5:

        transmission Jones, shipped   6.674e-02  2.044e-02  1.101e-02
        per-order amps,     shipped   6.206e-02  2.064e-02  1.089e-02
        the oracle's own last step         --    4.929e-02  1.127e-02

    -- the shipped arm improves 6.06x over the ladder and lands 1.02x BELOW the
    oracle's own last step.  The bars are the DECISION (converging, and inside
    the oracle's own resolution with a factor of two of room), not those
    readings."""
    r = _v2_ladder()
    assert r[4]["J"] / r[12]["J"] > 3.0
    assert r[4]["amps"] / r[12]["amps"] > 3.0
    assert r[12]["J"] < 2.0 * r[12]["step"]
    assert r[12]["amps"] < 2.0 * r[12]["step"]


def test_v2_the_unanchored_and_conjugate_arms_do_not_converge():
    """Two-sided.  Removing the anchor leaves the residual FLAT -- measured
    ``1.108 / 1.093 / 1.096`` over a 3x refinement of the oracle, a spread of
    1.4% -- and CONJUGATING it is worse than applying nothing at all
    (``1.834 / 1.825 / 1.829``).  A phase that were merely an accuracy gap
    would follow the oracle; a FRAME does not.

    Bars: the un-anchored first/last inside ``(0.8, 1.25)`` (measured 1.011,
    i.e. 1.1% of a 25% allowance); ``none / shipped > 20`` (measured 99.5x);
    ``conj / shipped > 20`` (measured 166x); ``conj / none > 1.3`` (measured
    1.669, and the two builds agree on it to better than 1e-7 relative)."""
    r = _v2_ladder()
    assert 0.8 < r[4]["J_none"] / r[12]["J_none"] < 1.25
    assert 0.8 < r[4]["amps_none"] / r[12]["amps_none"] < 1.25
    assert r[12]["J_none"] / r[12]["J"] > 20.0
    assert r[12]["amps_none"] / r[12]["amps"] > 20.0
    assert r[12]["J_conj"] / r[12]["J"] > 20.0
    assert r[12]["J_conj"] / r[12]["J_none"] > 1.3


def test_v2_the_reflection_side_carries_no_anchor():
    """The frame is anchored at the stack's TOP, which is where the reflected
    wave leaves it, so the reflection converges AS RETURNED on both trees:
    measured ``3.06e-02 / 1.45e-02 / 9.74e-03`` over the same ladder, a 3.14x
    improvement with no factor applied at all."""
    r = _v2_ladder()
    assert r[4]["refl"] / r[12]["refl"] > 2.0
    assert r[12]["refl"] < 2.0 * r[12]["step"]


def test_v2_the_zeroth_order_is_exempt_at_normal_incidence():
    """``alpha_0 = 0`` at normal incidence, so ``P_0 = exp(0) = 1`` EXACTLY --
    for any walk whatsoever.  ``jones_transmission`` is therefore untouched
    there (an exact identity, asserted as one), while ``per_order_amplitudes``
    is not: every ``m != 0`` order carries a real ``alpha_m``.  MEASURED
    against an ``n_slices = 8`` staircase at normal incidence: the per-order
    arm reads ``as returned`` far above ``x P``, while the two Jones arms are
    the same object."""
    sh, _o = _v2_sheared(theta=0.0)
    o_s, A_s, kx = _amps(sh)
    i0 = int(np.where(o_s == 0)[0][0])
    assert kx[i0] == 0.0
    ph = _P(kx, V2_W)
    assert complex(ph[i0]) == 1.0 + 0.0j
    J = np.asarray(sh.jones_transmission())
    assert _sha(J) == _sha(J * complex(ph[i0]))
    # ... but the other orders are NOT exempt
    assert int(len(np.unique(np.round(np.angle(ph), 9)))) >= 5
    stc, _oc = _v2_stair(8, theta=0.0)
    o_c, A_c, _k = _amps(stc)
    shipped = _align(o_s, A_s, o_c, A_c)
    none = _align(o_s, A_s / ph, o_c, A_c)
    assert none / shipped > 20.0


def test_v2_the_walks_add_over_two_sheared_layers():
    """Two SHEARED layers at DIFFERENT shears, thicknesses, duties and
    permittivities (``d 0.24 um`` at shear 0.25 over ``d 0.16 um`` at shear
    0.10; total walk 0.35 P), against a staircase that places the LOWER layer
    at the ACCUMULATED walk -- the statement that the frame CONTINUES.

    MEASURED at ``K = 6`` rungs per layer, per-order transmitted amplitudes:

        the FULL sum (shipped)   1.420e-02      the oracle's own K4->K6 step
        only layer 1's walk      5.539e-01        1.545e-02
        only layer 2's walk      1.1658
        no sum at all            1.3601
        the CONJUGATE sum        1.6300

    -- 39x / 82x / 96x / 115x worse than the shipped arm, which itself sits
    1.09x below the oracle's own step.  Every wrong-sum arm is bounded at 10x
    here, 3.9x .. 11.5x of room."""
    st = _v2_stack()
    st.add_sheared_grating(0.24e-6, eps_ridge=4.20, eps_groove=1.45,
                           duty=0.45, shear=0.25, centre=0.5)
    st.add_sheared_grating(0.16e-6, eps_ridge=2.90, eps_groove=1.60,
                           duty=0.35, shear=0.10, centre=0.5)
    st, _o = _v2_solve(st)
    o_s, A_s, kx = _amps(st)
    w1, w2 = 0.25 * V2_P, 0.10 * V2_P
    Pw, P1, P2 = _P(kx, w1 + w2), _P(kx, w1), _P(kx, w2)

    prev = None
    rows = {}
    for K in (4, 6):
        sc = _v2_stack()
        sc.add_tapered_ridges(0.24e-6,
                              ridges=[(0.5 * V2_P, 0.45 * V2_P,
                                       0.45 * V2_P, 4.20)],
                              eps_groove=1.45, n_slices=K, shear=0.25)
        sc.add_tapered_ridges(0.16e-6,
                              ridges=[(0.75 * V2_P, 0.35 * V2_P,
                                       0.35 * V2_P, 2.90)],
                              eps_groove=1.60, n_slices=K, shear=0.10)
        sc, _oc = _v2_solve(sc)
        o_c, A_c, _k = _amps(sc)
        rows[K] = dict(full=_align(o_s, A_s, o_c, A_c),
                       only1=_align(o_s, A_s / Pw * P1, o_c, A_c),
                       only2=_align(o_s, A_s / Pw * P2, o_c, A_c),
                       none=_align(o_s, A_s / Pw, o_c, A_c),
                       conj=_align(o_s, A_s / Pw * np.conj(Pw), o_c, A_c),
                       step=(None if prev is None
                             else _align(prev[0], prev[1], o_c, A_c)))
        prev = (o_c, A_c)
    r = rows[6]
    assert r["full"] < 2.0 * r["step"]
    for arm in ("only1", "only2", "none", "conj"):
        assert r[arm] / r["full"] > 10.0, arm
    assert rows[4]["full"] / r["full"] > 1.5


def test_v2_the_layer_split_is_exact_and_blind_to_a_wrong_sum():
    """In the FRAME the structure is z-invariant, so one sheared layer of
    ``d`` at shear ``s`` is the same solid as two halves of ``d/2`` at shear
    ``s/2`` (same ``tan(phi)``) whose own-frame ridge centre is the original
    TOP-face centre.  MEASURED: the two agree to ``6.45e-16`` on the per-order
    transmitted amplitudes -- and, against the staircase, they read the SAME
    number under the RIGHT sum (1.0367e-02 both) AND the same number under a
    HALF sum (9.1697e-01 both) and under NO sum (1.3762e+00 both).

    That last row is the point: the split identity ALONE can never detect a
    mis-summed walk, because both arms move together.  What detects it is the
    lab-referenced staircase, against which the half-sum arm is 88x and the
    no-sum arm 133x worse."""
    def one():
        st = _v2_stack()
        st.add_sheared_grating(0.24e-6, eps_ridge=4.20, eps_groove=1.45,
                               duty=0.45, shear=0.25, centre=0.5)
        return _v2_solve(st)

    def halves():
        st = _v2_stack()
        for _ in range(2):
            st.add_sheared_grating(0.12e-6, eps_ridge=4.20, eps_groove=1.45,
                                   duty=0.45, shear=0.125,
                                   centre=0.5 - 0.0625)
        return _v2_solve(st)

    s1, _a = one()
    s2, _b = halves()
    o1, A1, kx = _amps(s1)
    o2, A2, _k = _amps(s2)
    assert _align(o1, A1, o2, A2) < 1e-12

    sc = _v2_stack()
    sc.add_tapered_ridges(0.24e-6,
                          ridges=[(0.5 * V2_P, 0.45 * V2_P,
                                   0.45 * V2_P, 4.20)],
                          eps_groove=1.45, n_slices=8, shear=0.25)
    sc, _oc = _v2_solve(sc)
    o_c, A_c, _kk = _amps(sc)
    w = 0.25 * V2_P
    Pw, Ph = _P(kx, w), _P(kx, w / 2)
    full1 = _align(o1, A1, o_c, A_c)
    full2 = _align(o2, A2, o_c, A_c)
    half1 = _align(o1, A1 / Pw * Ph, o_c, A_c)
    half2 = _align(o2, A2 / Pw * Ph, o_c, A_c)
    none1 = _align(o1, A1 / Pw, o_c, A_c)
    assert abs(full1 - full2) / full1 < 1e-6
    assert abs(half1 - half2) / half1 < 1e-6         # the BLINDNESS, to the digit
    assert half1 / full1 > 10.0
    assert none1 / full1 > 10.0


def test_v2_the_cross_engine_arm_agrees_with_the_independent_pure_engine():
    """The staircase is the same engine as the thing under test.  The 2-D PURE
    staggered engine is not: a no-floor nodal engine whose transmitted
    amplitudes are lab-referenced by its own ``solve``-level walk sum.  Handed
    the SAME parallelogram as a y-uniform cell it must agree with the anchored
    1-D answer on the ZEROTH-order transmitted Jones -- which is invariant
    under a lateral translation of the whole structure, so the two engines'
    ``centre`` conventions cannot contaminate it.

    MEASURED (period 0.80 um, wl 0.55 um, d 0.40 um, duty 0.50, shear 0.25,
    oblique 25), the 1-D answer against ``PMM2DStackPure``:

        n_modes   shipped     un-anchored          conjugated       cost
          4       6.721e-02   9.690e-01  (14.4x)   1.667  (24.8x)    1.1 s
          5       1.566e-02   9.380e-01  (59.9x)   1.650 (105.4x)    8.1 s
          6       9.009e-03   9.357e-01 (103.9x)   1.650 (183.2x)   30.7 s

    -- the shipped arm converges with the pure engine's own ``n_modes`` while
    the other two stand still, which is itself the frame-vs-accuracy decision.
    The file runs ``n_modes = 5`` for the budget; the pure arm is converged
    there to 6.6e-03 against ``n_modes = 6`` and 6.06e-04 between 6 and 8."""
    from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
    duty, shear, centre = 0.50, 0.25, 0.625      # top-face ridge centre 0.500
    tx, walk = shear * V2_P / V2_D, shear * V2_P
    st = _v2_stack(degree=12, n_orders=7)
    st.add_sheared_grating(V2_D, eps_ridge=V2_ER, eps_groove=V2_EG, duty=duty,
                           shear=shear, centre=centre)
    st, _o = _v2_solve(st)
    J = np.asarray(st.jones_transmission())

    nx = 4
    cell = np.full((nx, nx), V2_EG, dtype=complex)
    cell[int(round((0.5 - duty / 2) * nx)):
         int(round((0.5 + duty / 2) * nx)), :] = V2_ER
    pu = PMM2DStackPure(V2_P, V2_P, n_superstrate=1.0, n_substrate=1.6,
                        n_modes=5, n_orders=3)
    pu.add_layer(V2_D, eps_cell=cell, slant=(tx, 0.0))
    pu.set_source(V2_WL, theta=V2_TH, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pu.solve()
    R = np.asarray(pu.jones_transmission())

    p0 = complex(np.exp(1j * (2.0 * np.pi / V2_WL)
                        * math.sin(V2_TH) * walk))
    shipped = _resid(J, R)
    none = _resid(J / p0, R)
    conj = _resid(J * np.conj(p0) / p0, R)
    assert none / shipped > 20.0
    assert conj / shipped > 20.0
    assert shipped < 0.05


# --------------------------------------------------------------------- O2
O2_WL = 0.68e-6
O2_PX = O2_PY = 1.20e-6
O2_D, O2_DF, O2_FEPS = 0.50e-6, 0.25e-6, 3.6
O2_XPROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])


def _o2_cell(ground=1.0):
    c = np.full((6, 6), ground, dtype=complex)
    c[:, :3] = O2_XPROF[:, None] + (ground - 1.0)
    return c


def _o2_hybrid(*, M=5, slant=0.5, degree=11, theta=25.0, oop=False,
               ground=1.0):
    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    st = PMM2DStackHybrid(O2_PX, O2_PY, n_superstrate=1.0, n_substrate=1.5,
                          n_orders=M, degree=degree)
    c = _o2_cell(ground)
    sl = None if slant == 0 else (slant, 0.0)
    if oop:
        T = np.zeros(c.shape + (3, 3), dtype=complex)
        for i in range(3):
            T[..., i, i] = c
        T[..., 0, 2] = T[..., 2, 0] = 0.35 * (c - 1.0)
        st.add_layer(O2_D, eps_tensor_cell=T, slant=sl)
    else:
        st.add_layer(O2_D, eps_cell=c, slant=sl)
    st.set_source(O2_WL, theta=math.radians(theta), phi=0.0)
    return st


def _rt(res):
    R, T = np.asarray(res[1]), np.asarray(res[2])
    return float(np.max(R.sum(axis=-1) + T.sum(axis=-1)))


def test_o2_the_generalized_cascade_blow_up_is_refused():
    """FAIL-BEFORE, restated as the measurement it came from.  On the pre-fix
    tree these two rows RETURNED, with a ``UserWarning`` textually identical to
    the one 17 ordinary 1.03..1.08 truncation rows raise:

        slanted patterned, n_orders 5   sum R + T = 1.257e+30
        OUT-OF-PLANE vertical, M = 5    sum R + T = 2.766e+26

    -- so the defect is NOT slant-specific; what carries it is the GENERALIZED
    (4N) cascade, which slanted AND out-of-plane layers both reach.  Both now
    raise ``_ConditioningError`` from the ``T22`` guard."""
    for kw in (dict(M=5, slant=0.5), dict(M=5, slant=0.0, oop=True)):
        with pytest.raises(_rc._ConditioningError, match="generalized"):
            _o2_hybrid(**kw).solve()


def test_o2_the_refusal_names_both_instruments_and_the_measured_remedies():
    """A refusal is only useful if it says what to do.  The message must carry
    both instruments (so the reading can be reproduced), the block size, and
    the remedies that were MEASURED to work -- and must NOT repeat the
    library's generic detune advice, which was measured NOT to work on this
    failure class (1e-6 and 1e-4 detunes both left the blow-up 28 decades
    out)."""
    with pytest.raises(_rc._ConditioningError) as ei:
        _o2_hybrid(M=5, slant=0.5).solve()
    msg = str(ei.value)
    assert "generalized interface (T22)" in msg
    assert "rcond" in msg and "A X = I" in msg
    for remedy in ("n_orders", "degree", "shear", "PURE"):
        assert remedy in msg, remedy
    assert "DETUNING" in msg and "NOT" in msg
    assert isinstance(ei.value, _rc._EnergyError)      # stabilize= ladders


def test_o2_the_measured_cures_still_solve():
    """Every clause of the remedy line is a MEASURED cure on the exact fixture
    that produced the refusal.  Re-measured here, so the message can never
    promise something the library stopped doing:

        degree = 7           sum R + T = 1.0442
        slant  = 0.25        sum R + T = 1.0436   (a 14.0 deg wall tilt)
        n_orders = 9         sum R + T = 1.0065

    against 1.26e+30 at the shipped defaults."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert _rt(_o2_hybrid(M=5, degree=7).solve()) < 1.10
        assert _rt(_o2_hybrid(M=5, slant=0.25).solve()) < 1.10
        assert _rt(_o2_hybrid(M=9).solve()) < 1.10


def test_o2_the_benign_energy_warning_rows_are_not_refused():
    """The 17 false-alarm rows.  ``PMMStack.solve``'s energy tripwire warns at
    ``sum R + T = 1.03 .. 1.08`` on ordinary low-``n_orders`` truncations --
    vertical rows included -- and it is exactly that noise that made a 1e+30
    invisible.  Those rows must keep SOLVING and keep WARNING.

    MEASURED on the same fixture, three mounts: normal 1.0350, oblique 40
    1.0726, conical 25-40 1.0461.  Their ``T22`` reads equilibrated ``rcond``
    4.76e-04 .. 2.69e-02 -- 6 to 8 decades on the safe side of the 1e-10 bar."""
    from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
    for theta in (0.0, 40.0):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tot = _rt(_o2_hybrid(M=5, theta=theta).solve())
        assert 1.0 < tot < 1.10
        assert any("energy not conserved" in str(x.message) for x in w)
    st = PMM2DStackHybrid(O2_PX, O2_PY, n_superstrate=1.0, n_substrate=1.5,
                          n_orders=5)
    st.add_layer(O2_D, eps_cell=_o2_cell(), slant=(0.5, 0.0))
    st.set_source(O2_WL, theta=math.radians(25.0), phi=math.radians(40.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert 1.0 < _rt(st.solve()) < 1.10


def test_o2_the_threshold_sits_in_the_measured_gap_on_both_sides():
    """The bar is DERIVED, and re-derived here rather than quoted.  With the
    census armed, ``_guarded_inverse`` records ``(site, n, rcond, resid,
    refused)`` for every ``T22``.  The two populations must stay decades apart
    on BOTH instruments, or the refusal is testing noise:

        BROKEN  (the blow-up fixture)   rcond 3.90e-17   resid 8.08e-02
        HEALTHY (n_orders 9, degree 7,
                 slant 0.25, normal)    rcond >= 2.97e-05

    ``_INV_T22_RCOND_REFUSE`` = 1e-10 is the geometric middle of the measured
    11.10-decade gap (8.4e-11).  The bars below give each side five decades.

    RESTATED 2026-09-11 (verification, durability flag): the broken-side bar
    was 1e-15 -- a SAMPLE property of this one fixture (3.90e-17).  The
    verification's wider population of 134 solves / 229 interfaces reads the
    broken rcond up to 4.78e-16 (0.32 decades under 1e-15, i.e. inside
    build noise), so the bar is now 1e-13: 2.3 decades above the family's
    broken envelope and 3.0 decades below the refusal bar 1e-10, with the
    healthy side (>= 2.08e-05) untouched."""
    rows = []
    _rc._INV_CENSUS = rows
    try:
        with pytest.raises(_rc._ConditioningError):
            _o2_hybrid(M=5, slant=0.5).solve()
        broken = [r for r in rows if r[4]]
        assert broken, "the census recorded no refused row"
        assert max(r[2] for r in broken) < 1e-13
        assert min(r[3] for r in broken) > 1e-3
        rows.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _o2_hybrid(M=5, degree=7).solve()
            _o2_hybrid(M=5, slant=0.25).solve()
            _o2_hybrid(M=5, theta=0.0).solve()
        healthy = [r for r in rows
                   if r[0] == "rcwa generalized interface (T22)"]
        assert len(healthy) >= 4
        assert not any(r[4] for r in healthy)
        assert min(r[2] for r in healthy) > 1e-5
    finally:
        _rc._INV_CENSUS = None
    assert _rc._INV_T22_RCOND_REFUSE == 1e-10
    assert min(r[2] for r in healthy) / _rc._INV_T22_RCOND_REFUSE > 1e4
    assert _rc._INV_T22_RCOND_REFUSE / max(r[2] for r in broken) > 1e4


def test_o2_the_guard_is_site_scoped_and_default_off():
    """M1 withdrew the GLOBAL inverse refusal because no bar separated healthy
    from broken across every explicit inverse in the library, and that stands.
    What is armed is ONE call site.  Proved two ways: an unarmed
    ``_guarded_inverse`` on a matrix that WOULD be refused returns
    ``np.linalg.inv`` BIT-FOR-BIT and does not raise, and the default census
    hook is still ``None`` so the default path pays nothing."""
    assert _rc._INV_CENSUS is None
    rng = np.random.default_rng(20260911)
    n = 12
    U, _s, Vh = np.linalg.svd(rng.normal(size=(n, n))
                              + 1j * rng.normal(size=(n, n)))
    s = np.geomspace(1.0, 1e-18, n)
    A = (U * s) @ Vh
    plain = np.linalg.inv(A)
    got = _rc._guarded_inverse(A, "unit test (unarmed)")
    assert _sha(got) == _sha(plain)
    with pytest.raises(_rc._ConditioningError):
        _rc._guarded_inverse(A, "unit test (armed)",
                             rcond_refuse=_rc._INV_T22_RCOND_REFUSE)


def test_o2_the_mortar_interfaces_take_no_explicit_inverse():
    """The brief asks whether the same screen must guard the mortar.  It must
    not, and the reason is structural rather than a threshold: BOTH
    ``_interface_smatrix_general_mortar`` and its 2-D twin end in
    ``np.linalg.solve(A, B)``, so there is no explicit inverse for
    ``_guarded_inverse`` to screen.  MEASURED on the blow-up fixture CLASS (an
    OUT-OF-PLANE pure stack on per-layer grids whose two layers carry different
    grids -- the only route to the 2-D general mortar): ``cond(A) = 3.14e+03``
    on a 936 x 936 solve at ``sum R + T = 0.9997``, i.e. twelve decades below
    the singular ``T22`` readings the refusal exists for.

    RESTATED 2026-09-11 at the merge with mortar round 2 (D2): the 2-D twin
    no longer spells its solve ``np.linalg.solve`` -- it goes through
    ``_guarded_mortar_solve``, which is ``lu_factor`` + ``lu_solve`` (the
    same ``getrf`` + ``getrs`` pair, bit-identical, measured 60/60 solves by
    the round-2 verification).  The DECISION this test pins is unchanged:
    every mortar interface ends in a SOLVE and never forms an explicit
    inverse, so there is nothing for ``_guarded_inverse`` to screen.  The
    assertion now names both solve spellings and forbids both inverse
    spellings, on the mortar functions AND on the guarded solve itself."""
    from lumenairy.elements.pmm import _core as _pc
    solve_spellings = ("np.linalg.solve", "_guarded_mortar_solve")
    inverse_spellings = ("_guarded_inverse", "np.linalg.inv", "linalg.inv(")
    for name in ("_interface_smatrix_general_mortar_2d",
                 "_interface_smatrix_general_mortar"):
        src = _mortar_source(_pc, name)
        assert any(sp in src for sp in solve_spellings), name
        assert not any(sp in src for sp in inverse_spellings), name
    guarded = _mortar_source(_pc, "_guarded_mortar_solve")
    assert "lu_factor" in guarded and "lu_solve" in guarded
    assert not any(sp in guarded for sp in inverse_spellings)


def _mortar_source(mod, name):
    import inspect
    return inspect.getsource(getattr(mod, name))
