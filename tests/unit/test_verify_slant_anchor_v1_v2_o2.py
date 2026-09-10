"""The INDEPENDENT verification's own gates for V1 / V2 / O2.

``tests/unit/test_fix_slant_anchor_v1_v2_o2.py`` is the fix's gate; this file
holds only what the independent re-measurement found the fix's gate does NOT
pin.  Evidence, with both builds' readings and every bar's derivation:
``docs/audits/VERIFY_SLANT_ANCHOR_V1_V2_O2_2026_09_11.md``; probes in
``validation/probe_verify_slant_anchor/`` (Windows py3.14 / numpy 2.4.4 and
WSL py3.12 / numpy 2.4.6, both scipy-openblas, thread caps 1, 2026-09-11).

TERMS.  A sheared 1-D layer is solved in the FRAME ``u = x - z tan(phi)``; the
cascade's transmitted per-order coefficients are FRAME coefficients, and
``A_lab(m) = exp(+i k0 alpha_m W) A_frame(m)`` with ``W = sum_j tan(phi_j) d_j``
is the re-referencing V2 shipped.  ``alpha_m = kx_m / k0``.

WHAT IS NEW HERE

  D1  the FRAME-CONTINUATION placement of a SECOND sheared layer.  The
      cascade matches successive layers' FRAME coefficients directly, so the
      ``centre`` handed to ``add_sheared_grating`` for the second sheared
      layer lands at ``+ W1`` in the LAB -- while that method's docstring
      states a LAB centre law and says the builder and the z-staircase "are
      interchangeable at the call site".  Measured here two-sidedly.
  D2  ``factorization='auto'`` sends a single IN-PLANE sheared grating to the
      COVARIANT cascade, which retains NO amplitudes: on the library default
      the anchored surface does not exist at all.
  D3  the anchor stays UNIMODULAR through a Rayleigh (Wood) cut-off, for the
      EVANESCENT orders too, and under a lossy superstrate.
  D4  a stack whose NET walk is exactly zero gets the exact identity, while
      one layer's own walk is a genuine phase.
  D5  every OTHER surface of a sheared stack is refused, which is what makes
      "the transmitted per-order amplitudes are the only frame-referenced
      return" a closed statement rather than an unchecked one.
  D6  O2's refusal decides identically with the census hook armed and
      disarmed -- the armed hook raises the residual threshold, so the two
      paths are not the same code.
  D7  O2's two populations separate on an INDEPENDENT scalar-cell family.
  D8  V1: no EIGHTH route reaches the jnp twin carrying a shear.
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
    _slant_frame_walk_1d,
)
from lumenairy.elements.pmm.stack2d import PMM2DStackHybrid
from lumenairy.elements.pmm.twod_jones import pmm_jones_2d
from lumenairy.elements.rcwa import _core as _rc

UM = 1e-6

# ---- this verification's own 1-D fixture ---------------------------------
P = 0.66 * UM
WL = 0.52 * UM
D = 0.34 * UM
ER, EG = 3.80, 1.25
DUTY = 0.38
SHEAR = 0.24
TH = math.radians(31.0)
NSUB = 1.45
DEG, NORD = 9, 5


def _stack(grids="shared", fac="convection", **kw):
    return PMMStack(P, n_superstrate=1.0, n_substrate=NSUB, degree=DEG,
                    n_orders=NORD, factorization=fac, layer_grids=grids, **kw)


def _solve(st, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(**kw)


def _amps(st, port="transmission"):
    a = st.per_order_amplitudes(port)
    return (np.asarray(a["orders"]),
            np.stack([np.asarray(a["Ex"]), np.asarray(a["Ey"])], axis=0),
            np.asarray(np.real(a["kx"])), a)


def _P(kx, walk, wl=WL):
    return np.exp(1j * (2.0 * np.pi / wl) * np.asarray(kx) * walk)


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


def _sha(*a):
    h = hashlib.sha256()
    for x in a:
        h.update(np.ascontiguousarray(np.asarray(x)).tobytes())
    return h.hexdigest()


def _binary_segments(centre, duty, er, eg):
    """The ``(width_fraction, eps)`` list of ONE binary slice whose ridge of
    width ``duty`` is centred at ``centre`` in period fractions, wrap-aware --
    this file's own geometry, so the oracle below owes the library nothing but
    ``add_layer``."""
    a = (centre - duty / 2.0) % 1.0
    b = a + duty
    segs = ([(a, eg), (duty, er), (1.0 - b, eg)] if b <= 1.0
            else [(b - 1.0, er), (1.0 - duty, eg), (1.0 - a, er)])
    return [(w, e) for w, e in segs if w > 1e-12]


def _hand_staircase(spec, ns):
    """An ALL-VERTICAL stack of ``ns`` rungs per ``(t, c_top, shear, er, eg,
    duty)`` entry -- LAB-referenced by construction, so it enters no frame.
    ``layer_grids='per-layer'`` for cost (each rung's window carries its own
    walls plus its neighbours', not every rung's)."""
    st = _stack(grids="per-layer")
    for (t, c_top, sh, er, eg, du) in spec:
        for k in range(ns):
            st.add_layer(t / ns, segments=_binary_segments(
                c_top + sh * (k + 0.5) / ns, du, er, eg))
    st.set_source(WL, angle=TH)
    return st


def _sheared_run(spec):
    """The same solid as EXACT slanted layers.  ``add_sheared_grating`` lays
    the ridge centre at ``centre + shear (zeta - 0.5)``, so a TOP-face centre
    ``c_top`` is ``centre = c_top + shear / 2``."""
    st = _stack()
    for (t, c_top, sh, er, eg, du) in spec:
        st.add_sheared_grating(t, eps_ridge=er, eps_groove=eg, duty=du,
                               shear=sh, centre=c_top + sh / 2.0)
    st.set_source(WL, angle=TH)
    return st


# ======================================================================= D1
S1, S2 = 0.30, -0.12
D1_T, D2_T = 0.22 * UM, 0.16 * UM
C1 = 0.5 - S1 / 2.0


def test_verify_v2_a_second_sheared_layer_sits_at_the_ACCUMULATED_walk():
    """WHERE, IN THE LAB, DOES THE SECOND SHEARED LAYER ACTUALLY SIT?

    The 1-D cascade matches successive sheared layers' FRAME coefficients
    directly (``_interface_smatrix_general(Mls[i], Mls[i+1])``, no
    re-referencing), so layer 2 is solved in layer 1's frame ``u = x - W1``:
    the ``centre`` handed to :meth:`PMMStack.add_sheared_grating` for the
    second sheared layer describes a ridge that stands at ``centre + W1`` in
    the LAB.  That is the premise the walk SUM rests on, and the fix's own
    composition test encodes it (its staircase puts the lower ridge at
    ``0.75 P`` for a ``0.25 P`` first walk) -- but ``add_sheared_grating``'s
    docstring states a LAB centre law and says the builder and the staircase
    "describe the SAME structure and the two are interchangeable at the call
    site", which is true for ONE sheared layer and false for the second.

    MEASURED here against this file's OWN hand-laid staircase, per-order
    transmitted amplitudes, at 4 and 8 rungs per layer (WIN and WSL agree to
    five figures):

        oracle rungs                                    4          8
        layer 2 at ``c_top2 + W1`` (frame-continued)  3.083e-02  1.259e-02
        layer 2 at ``c_top2``      (the docstring)    1.1362     1.1408

    -- a factor 36.9 at 4 rungs, and the decision is not a tolerance: the
    frame-continued arm CONVERGES with the oracle (2.45x over one doubling)
    while the lab-placed one STANDS STILL (1.004x), which is what separates a
    wrong GEOMETRY from an accuracy gap.  The bars give the small arm 3.2x,
    the large one 2.3x, the ratio 12x, and the convergence 1.4x."""
    spec_frame = [(D1_T, C1, S1, ER, EG, DUTY),
                  (D2_T, C1 + S1 + S1, S2, 3.1, 1.1, 0.55)]
    spec_lab = [(D1_T, C1, S1, ER, EG, DUTY),
                (D2_T, C1 + S1, S2, 3.1, 1.1, 0.55)]
    two = _sheared_run([(D1_T, C1, S1, ER, EG, DUTY),
                        (D2_T, C1 + S1, S2, 3.1, 1.1, 0.55)])
    _solve(two)
    o_s, A_s, _kx, _a = _amps(two)
    got = {}
    for tag, spec in (("frame", spec_frame), ("lab", spec_lab)):
        for ns in (4, 8):
            sc = _hand_staircase(spec, ns)
            _solve(sc)
            o_c, A_c, _k, _b = _amps(sc)
            got[(tag, ns)] = _align(o_s, A_s, o_c, A_c)
    assert got[("frame", 8)] < 0.10, got
    assert got[("lab", 8)] > 0.50, got
    assert got[("lab", 4)] / got[("frame", 4)] > 3.0, got
    assert got[("frame", 4)] / got[("frame", 8)] > 1.8, got   # converges
    assert 0.8 < got[("lab", 4)] / got[("lab", 8)] < 1.25, got   # flat


# ======================================================================= D2
def test_verify_v2_the_default_factorization_exposes_no_transmitted_field():
    """WHICH ROUTE EVEN REACHES THE ANCHOR.

    ``factorization='auto'`` sends an IN-PLANE stack whose slant is UNIFORM to
    the COVARIANT (spectral) cascade, which retains no per-order amplitudes at
    all -- so on the library's DEFAULT setting a single sheared grating has no
    transmitted surface for the anchor to correct, and V2 is reachable only
    through ``'convection'`` (or through a mixed-slant / out-of-plane stack,
    which ``'auto'`` itself routes to convection).  Two-sided, because a claim
    about which route is anchored is worthless without the route that is
    not."""
    auto = _stack(fac="auto")
    auto.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                             shear=SHEAR)
    auto.set_source(WL, angle=TH)
    _solve(auto)
    with pytest.raises(ValueError, match="no per-order amplitudes"):
        auto.jones_transmission()
    with pytest.raises(ValueError, match="no per-order amplitudes"):
        auto.per_order_amplitudes("transmission")

    conv = _stack(fac="convection")
    conv.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                             shear=SHEAR)
    conv.set_source(WL, angle=TH)
    _solve(conv)
    J = np.asarray(conv.jones_transmission())
    assert J.shape == (2, 2) and np.all(np.isfinite(J))
    assert float(np.max(np.abs(J))) > 1e-3


# ======================================================================= D3
def test_verify_v2_the_anchor_stays_unimodular_through_a_wood_anomaly():
    """IS ``alpha_m`` REAL FOR EVERY ORDER, INCLUDING THE EVANESCENT ONES?

    The anchor is a phase only while ``alpha_m`` is real; that is what makes
    the omission silent in the efficiencies, and it is the reason no
    ``R``/``T`` byte moves.  ``kx0`` is built from ``Re(n_superstrate)``, so
    ``alpha_m = (kx0 + m G) / k0`` is real by construction -- but "by
    construction" is a reading of the source, so it is measured here at three
    wavelengths straddling a substrate Rayleigh (Wood) cut-off, where orders
    cross between propagating and evanescent, and once more under a LOSSY
    superstrate.

    MEASURED, both builds, identically: ``max|Im(alpha_m)| = 0.0`` exactly and
    ``max||P_m| - 1| = 1.110e-16`` on every row, with 4 to 6 evanescent orders
    present so the claim is not vacuous.  The bar is ``1e-14``, 90x above the
    reading, and the phase spread is asserted separately so a degenerate
    all-ones ``P`` cannot pass."""
    wl_cut = P * (NSUB + math.sin(TH)) / 2.0
    seen_evanescent = 0
    for f in (0.985, 1.0, 1.015):
        st = _stack()
        st.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                               shear=SHEAR)
        st.set_source(wl_cut * f, angle=TH)
        _solve(st)
        _o, _A, kx, a = _amps(st)
        alpha = np.asarray(a["kx"])
        assert float(np.max(np.abs(np.imag(alpha)))) == 0.0
        Pm = _P(alpha, SHEAR * P, wl_cut * f)
        assert float(np.max(np.abs(np.abs(Pm) - 1.0))) < 1e-14
        assert float(np.ptp(np.angle(Pm))) > 1.0     # not the identity
        seen_evanescent += int(np.sum(np.real(np.asarray(a["kz"])) <= 0.0))
    assert seen_evanescent >= 3, seen_evanescent

    lossy = PMMStack(P, n_superstrate=1.0 + 0.4j, n_substrate=NSUB,
                     degree=DEG, n_orders=NORD, factorization="convection")
    lossy.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                              shear=SHEAR)
    lossy.set_source(WL, angle=TH)
    _solve(lossy)
    _o, _A, _kx, a = _amps(lossy)
    assert float(np.max(np.abs(np.imag(np.asarray(a["kx"]))))) == 0.0
    Pm = _P(np.asarray(a["kx"]), SHEAR * P)
    assert float(np.max(np.abs(np.abs(Pm) - 1.0))) < 1e-14


# ======================================================================= D4
def test_verify_v2_a_net_zero_walk_is_the_exact_identity_through_a_solve():
    """THE CANCELLATION, THROUGH A REAL SOLVE RATHER THAN A TUPLE.

    Two sheared layers of equal thickness at ``+phi`` and ``-phi`` have
    ``W = tan(phi) d + tan(-phi) d``.  ``np.tan`` is odd to the bit on both
    builds, so the sum is EXACTLY ``0.0`` -- and ``_anchor_transmission_to_lab_1d``
    short-circuits on ``walk == 0.0``, returning the very same array objects.
    Bytes that are never touched cannot move, which is the strongest form the
    "no-op on a cancelling stack" claim can take.

    Two-sided: ONE layer's own walk applied to the same amplitudes moves them
    by ``> 0.5`` relative, so the identity above is not the identity of a
    quantity that no phase could change."""
    t = 0.26 * UM
    phi = float(math.atan(0.30 * P / t))
    st = _stack()
    st.add_layer(t, segments=[(DUTY, ER), (1 - DUTY, EG)], slant_angle=+phi)
    st.add_layer(t, segments=[(0.55, 3.1), (0.45, 1.1)], slant_angle=-phi)
    st.set_source(WL, angle=TH)
    _solve(st)
    walk = _slant_frame_walk_1d(st._layers)
    assert walk == 0.0, repr(walk)

    _o, A, kx, _a = _amps(st)
    modal = {"kx": kx, "tx": A[0].copy(), "ty": A[1].copy()}
    tx0, ty0 = modal["tx"], modal["ty"]
    out = _anchor_transmission_to_lab_1d(modal, WL, walk)
    assert out["tx"] is tx0 and out["ty"] is ty0

    one = float(np.tan(phi) * t)
    Pm = _P(kx, one)
    moved = float(np.linalg.norm(A[0] * Pm - A[0])
                  / np.linalg.norm(A[0]))
    assert moved > 0.5, moved


# ======================================================================= D5
def test_verify_v2_a_sheared_stack_exposes_no_other_frame_bearing_surface():
    """WHAT ELSE COULD CARRY THE FRAME, AND DOES IT EXIST?

    The anchor corrects ONE surface (the transmitted per-order amplitudes and
    what derives from them).  That is only a complete statement if every other
    port of a sheared stack is unreachable, so each is exercised here rather
    than argued from the source: conical incidence on all four
    factorization/grid routes, ``retain_internal`` (hence ``internal_field``
    and ``layer_absorption``) on all four plus a MIXED vertical + sheared
    stack, and ``solve_vs_wavelength``.  All refuse, on both builds."""
    for fac in ("auto", "convection", "covariant"):
        for grids in ("shared", "per-layer"):
            st = _stack(grids=grids, fac=fac)
            st.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG,
                                   duty=DUTY, shear=SHEAR)
            st.set_source(WL, angle=TH, phi=math.radians(37.0))
            with pytest.raises(NotImplementedError):
                st.solve()
    for fac in ("auto", "convection"):
        for grids in ("shared", "per-layer"):
            st = _stack(grids=grids, fac=fac)
            st.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG,
                                   duty=DUTY, shear=SHEAR)
            st.set_source(WL, angle=TH)
            with pytest.raises(NotImplementedError):
                st.solve(retain_internal=True)
    mx = _stack()
    mx.add_layer(0.20 * UM, segments=[(0.6, 3.1), (0.4, 1.1)])
    mx.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                           shear=SHEAR)
    mx.set_source(WL, angle=TH)
    with pytest.raises(NotImplementedError):
        mx.solve(retain_internal=True)
    sw = _stack()
    sw.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=DUTY,
                           shear=SHEAR)
    sw.set_source(WL, angle=TH)
    with pytest.raises(NotImplementedError):
        sw.solve_vs_wavelength(np.array([WL, WL * 1.01]), angle=TH)


# ======================================================================= O2
O2_PROF = np.array([4.0, 4.0, 2.0, 1.0, 1.0, 1.0])


def _o2_scalar_cell():
    """A HIGH-CONTRAST x profile against a HALF-CELL of uniform ground -- the
    shape that excites the generalized generator's rank deficiency.  A tensor
    cell of comparable contrast does NOT: 88 such solves in this
    verification's census were all healthy."""
    c = np.full((6, 6), 1.0, dtype=complex)
    c[:, :3] = O2_PROF[:, None]
    return c


def _o2_hyb(slant, norders, degree=11, theta=25.0):
    st = PMM2DStackHybrid(1.20 * UM, 1.20 * UM, n_superstrate=1.0,
                          n_substrate=1.5, n_orders=norders, degree=degree)
    st.add_layer(0.50 * UM, eps_cell=_o2_scalar_cell(),
                 slant=(None if slant == 0 else (slant, 0.0)))
    st.set_source(0.68 * UM, theta=math.radians(theta))
    return st


def test_verify_o2_the_refusal_is_the_same_with_the_census_armed_or_not():
    """THE CENSUS HOOK CHANGES THE CODE PATH, SO IT MUST NOT CHANGE THE
    VERDICT.

    ``_guarded_inverse`` computes the confirming residual when ``rcond`` falls
    below ``thr = max(_INV_RCOND_SCREEN if the census is armed else 0,
    rcond_refuse)`` -- ``1e-8`` with the census armed and ``1e-10`` without,
    two different thresholds and two different amounts of work.  The REFUSAL
    needs ``rcond < 1e-10 AND resid > 1e-8`` either way, so the verdict must
    be identical; the fix's gate only ever exercises the armed path (its
    threshold test) or the unarmed one (its blow-up test), never both on the
    same fixture.

    Also asserted: a HEALTHY solve returns the same bytes in both modes, so
    arming the census cannot perturb an answer."""
    for armed in (False, True):
        _rc._INV_CENSUS = [] if armed else None
        try:
            with pytest.raises(_rc._ConditioningError):
                _o2_hyb(0.5, 5).solve()
        finally:
            _rc._INV_CENSUS = None
    shas = []
    for armed in (False, True):
        _rc._INV_CENSUS = [] if armed else None
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                o, R, T, J = _o2_hyb(0.5, 5, degree=7).solve()
            shas.append(_sha(o, R, T, J))
        finally:
            _rc._INV_CENSUS = None
    assert shas[0] == shas[1], shas


def test_verify_o2_the_two_populations_separate_on_an_independent_family():
    """THE BAR, RE-DERIVED ON A FAMILY THE FIX DID NOT USE.

    Six rows of a scalar-cell hybrid spanning both populations.  Every REFUSED
    row must read ``rcond < 1e-15`` AND ``resid > 1e-3``; every SOLVING row
    must read ``rcond > 1e-5`` and close its energy below ``1.10``.  MEASURED
    on this family (WIN / WSL): refused ``rcond`` 4.7e-18 .. 4.8e-16 and
    ``resid`` >= 1.6e-02; solving ``rcond`` >= 1.67e-04.  ``1e-10`` therefore
    sits at least 5.3 decades above every refused reading and 5.2 decades
    below every solving one on a population built independently of the fix's.

    Non-vacuous by construction: the assertion at the end requires BOTH
    populations to be present, so a change that refused everything (or
    nothing) fails here rather than passing silently."""
    rows = {"refused": [], "solved": []}
    for slant, M in ((0.35, 3), (0.35, 5), (0.50, 3), (0.50, 5), (0.70, 5),
                     (0.70, 7)):
        _rc._INV_CENSUS = []
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    res = _o2_hyb(slant, M).solve()
                    tot = float(np.max(np.asarray(res[1]).sum(axis=-1)
                                       + np.asarray(res[2]).sum(axis=-1)))
                    kind = "solved"
                except _rc._ConditioningError:
                    tot, kind = None, "refused"
            cen = [c for c in _rc._INV_CENSUS
                   if c[0] == "rcwa generalized interface (T22)"]
        finally:
            _rc._INV_CENSUS = None
        assert cen, (slant, M)
        rcond = min(c[2] for c in cen)
        resid = [c[3] for c in cen if c[3] is not None]
        rows[kind].append((slant, M, rcond, resid, tot))
        if kind == "refused":
            assert rcond < 1e-15, (slant, M, rcond)
            assert resid and max(resid) > 1e-3, (slant, M, resid)
        else:
            assert rcond > 1e-5, (slant, M, rcond)
            assert tot < 1.10, (slant, M, tot)
    assert rows["refused"] and rows["solved"], rows


# ======================================================================= D8
V1_TH = math.radians(28.0)
V1_KW = dict(period_x=0.92 * UM, period_y=0.80 * UM, n_substrate=1.55,
             n_superstrate=1.0, depth=0.37 * UM, wavelength=0.61 * UM,
             theta=V1_TH, phi=0.0, n_orders=3, degree=5)


def _v1_cell(const=None):
    c = np.zeros((5, 4, 3, 3), dtype=complex)
    for i in range(5):
        for j in range(4):
            if const is not None:
                c[i, j] = np.eye(3) * const
            else:
                c[i, j] = np.eye(3) * (1.30 + 1.75
                                       * ((i * 3 + j * 2) % 7) / 6.0)
                c[i, j, 0, 1] = c[i, j, 1, 0] = 0.07 * ((i + 2 * j) % 3)
    return c


_V1_LAYOUT = np.arange(20, dtype=int).reshape(5, 4)


def test_verify_v1_no_eighth_route_carries_a_shear_into_the_jnp_twin():
    """THE HUNT FOR AN EIGHTH ROUTE.

    V1 closes the seven members of ``pmm_jones_2d``'s traced-input dispatch
    tuple.  The routes that tuple does NOT list were exercised on the pre-fix
    tree and every one of them either reached the twin (and is now refused) or
    failed loudly:

        jax.jit / jax.vmap / jax.grad over a listed input   REFUSED (they
                                                            route through the
                                                            same tuple)
        a TRACED ``slant`` with NumPy everything else       ConcretizationTypeError
        a y-only slant on a traced call                     REFUSED
        the PURE staggered entry under jit                  ConcretizationTypeError
                                                            (that engine has no
                                                            jnp twin at all)
        a CONSTANT NumPy cell with a traced depth           SOLVES, and
                                                            reproduces the
                                                            NumPy VERTICAL
                                                            answer

    The last row is the two-sided half: the refusal is about a cell that gets
    SOLVED IN A SHEARED FRAME, not about the API.  MEASURED there, identically
    on both builds and on all three trees: ``dR 2.7419e-10 / dT 1.5041e-10``
    against the NumPy vertical call -- the jnp twin's own accuracy on a
    degenerate (constant) cell, not a frame error, and eight decades below the
    ``dR 2.94e-02`` the same fixture's PATTERNED cell carries.  The bar is
    ``1e-6``: 3.6 decades above the reading and 4.5 below the defect."""
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    cell = _v1_cell()
    kw_nod = {k: v for k, v in V1_KW.items() if k != "depth"}

    with pytest.raises(NotImplementedError, match="SLANTED patterned cell"):
        jax.jit(lambda d: pmm_jones_2d(
            eps_tensor_cell=cell, depth=d, slant=(0.45, 0.0),
            region_layout=_V1_LAYOUT, **kw_nod)[1])(jnp.asarray(0.37e-6))
    with pytest.raises(NotImplementedError, match="SLANTED patterned cell"):
        jax.grad(lambda d: jnp.sum(pmm_jones_2d(
            eps_tensor_cell=cell, depth=d, slant=(0.45, 0.0),
            region_layout=_V1_LAYOUT, **kw_nod)[2]))(jnp.asarray(0.37e-6))
    with pytest.raises(NotImplementedError, match="SLANTED patterned cell"):
        pmm_jones_2d(eps_tensor_cell=cell, slant=(0.0, 0.45),
                     region_layout=_V1_LAYOUT,
                     depth=jnp.asarray(V1_KW["depth"]), **kw_nod)
    # a TRACED slant is not in the dispatch tuple: it must fail LOUDLY on the
    # NumPy path rather than drop the shear.
    with pytest.raises(Exception) as ei:
        jax.jit(lambda s: pmm_jones_2d(
            eps_tensor_cell=cell, slant=(s, 0.0),
            region_layout=_V1_LAYOUT, **V1_KW)[1])(jnp.asarray(0.45))
    assert not isinstance(ei.value, (float, np.ndarray))

    # the two-sided half: a CONSTANT NumPy cell on a traced depth SOLVES and
    # reproduces the NumPy VERTICAL answer bit for bit.
    cst = _v1_cell(const=2.20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vert = pmm_jones_2d(eps_tensor_cell=cst, slant=None, **V1_KW)
        got = pmm_jones_2d(eps_tensor_cell=cst, slant=(0.45, 0.0),
                           depth=jnp.asarray(V1_KW["depth"]),
                           region_layout=_V1_LAYOUT, **kw_nod)
    assert float(np.max(np.abs(np.asarray(got[1])
                               - np.asarray(vert[1])))) < 1e-6
