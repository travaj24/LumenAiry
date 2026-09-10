"""TASK 2 -- V2: the 1-D frame anchor's SIGN, by four independent means.

TERMS.  A sheared 1-D layer is solved in the FRAME coordinate
``u = x - z tan(phi)``, in which the structure is z-invariant.  The cascade's
transmitted per-order coefficients are therefore FRAME coefficients;
``A_lab(m) = P_m A_frame(m)`` with ``P_m = exp(+i k0 alpha_m W)`` and
``W = sum_j tan(phi_j) d_j`` over the layers that enter a frame is the
re-referencing under test.  ``alpha_m = kx_m / k0`` is the order's
dimensionless x-wavevector.  Every candidate arm below is reconstructed FROM
the tree's own returned answer, so the identical script is legible on a
pre-fix tree (where "as returned" IS ``A_frame``).

THE ORACLE IS BUILT HERE, BY HAND.  ``ladder()`` does not call
``add_tapered_grating``: it lays each slice of the parallelogram down with
``add_layer(segments=...)`` from segment widths this file computes, so the
reference is an ALL-VERTICAL stack whose lab geometry is under this probe's
own control and which enters no frame at all.  ``add_tapered_grating`` is run
beside it only as a second opinion.

The four means: (a) that hand-built z-staircase ladder, both signs;
(b) the ANALYTIC oracle -- a UNIFORM slanted film against the VERTICAL film of
the same eps; (c) the CROSS-ENGINE arm against the independent
``PMM2DStackPure`` staggered engine, whose own SIGN is adjudicated by the
anchor-free REFLECTION before the transmission is read; (d) NEGATIVE and
OPPOSITE shears, including a stack whose NET walk is exactly zero.

Plus the break attempts: conical incidence, a Wood-anomaly wavelength sweep
(is ``alpha_m`` real for the EVANESCENT orders too?), a lossy superstrate,
``solve_vs_wavelength``, ``retain_internal`` and the routing literal.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

if os.environ.get("LUM_ARM_TREE"):
    sys.path.insert(0, os.environ["LUM_ARM_TREE"])
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))

import _lib  # noqa: E402
import numpy as np  # noqa: E402

UM = 1e-6

# ---- MY fixture (none of these numbers is the fix's) ----------------------
P = 0.66 * UM
WL = 0.52 * UM
D = 0.34 * UM
E_R, E_G = 3.80, 1.25
DUTY = 0.38
SHEAR = 0.24                    # a 0.24-PERIOD walk: neither half nor quarter
THETA = float(np.deg2rad(31.0))
N_SUB, N_SUP = 1.45, 1.0
DEG, NORD = 11, 7
LADDER = (4, 8, 16)


def _stack(**kw):
    from lumenairy.elements.pmm import PMMStack
    kw.setdefault("period", P)
    kw.setdefault("n_substrate", N_SUB)
    kw.setdefault("n_superstrate", N_SUP)
    kw.setdefault("degree", DEG)
    kw.setdefault("n_orders", NORD)
    return PMMStack(**kw)


def _solve(st, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(**kw)


def _amps(st, port="transmission"):
    a = st.per_order_amplitudes(port)
    return np.asarray(a["Ex"]), np.asarray(a["Ey"]), np.asarray(a["kx"])


def _phase(alpha, walk, wl):
    return np.exp(1j * (2.0 * np.pi / wl) * np.asarray(alpha) * walk)


def binary_segments(centre, duty, eps_ridge, eps_groove):
    """The ``(width_fraction, eps)`` list of ONE binary grating slice whose
    ridge of width ``duty`` is centred at ``centre`` (period fractions),
    wrap-aware.  This is the oracle's geometry, written here."""
    a = (centre - duty / 2.0) % 1.0
    b = a + duty
    if b <= 1.0:
        segs = [(a, eps_groove), (duty, eps_ridge), (1.0 - b, eps_groove)]
    else:
        segs = [(b - 1.0, eps_ridge), (1.0 - duty, eps_groove),
                (1.0 - a, eps_ridge)]
    return [(w, e) for w, e in segs if w > 1e-12]


def hand_staircase(spec, ns, theta=THETA, grids="per-layer", **skw):
    """An ALL-VERTICAL stack of ``ns`` slices per entry of ``spec``, each entry
    ``(thickness, c_top, shear, eps_ridge, eps_groove, duty)`` -- the LAB
    geometry of the same parallelogram run, laid down by this file.

    ``layer_grids='per-layer'`` by default, for COST: on the shared union grid
    every slice's two walls enter every other slice's element partition, so a
    16-rung staircase carries a 32-element grid in each of its 16 eigensolves.
    ``ladder()`` measures the two routes against each other at the coarsest
    rung, so the choice is bounded rather than assumed."""
    st = _stack(layer_grids=grids, **skw)
    for (t, c_top, sh, er, eg, du) in spec:
        for k in range(ns):
            zeta = (k + 0.5) / ns
            st.add_layer(t / ns,
                         segments=binary_segments(c_top + sh * zeta, du,
                                                  er, eg))
    st.set_source(WL, angle=theta)
    return st


def sheared_run(spec, theta=THETA, **skw):
    """The SAME solid as EXACT slanted layers.  ``add_sheared_grating`` lays
    the LAB ridge centre at ``centre + shear (zeta - 0.5)``, so the top-face
    centre ``c_top`` means ``centre = c_top + shear / 2``."""
    st = _stack(factorization="convection", **skw)
    for (t, c_top, sh, er, eg, du) in spec:
        st.add_sheared_grating(t, eps_ridge=er, eps_groove=eg, duty=du,
                               shear=sh, centre=c_top + sh / 2.0)
    st.set_source(WL, angle=theta)
    return st


ONE = [(D, 0.5 - SHEAR / 2.0, SHEAR, E_R, E_G, DUTY)]


def _arms(Ex, Ey, alpha, walk, wl):
    """Every candidate re-referencing, built FROM the tree's own answer."""
    Pm = _phase(alpha, walk, wl)
    fx, fy = Ex / Pm, Ey / Pm
    return {
        "shipped": (Ex, Ey),
        "none": (fx, fy),
        "conj": (fx * np.conj(Pm), fy * np.conj(Pm)),
        "half": (fx * _phase(alpha, 0.5 * walk, wl),
                 fy * _phase(alpha, 0.5 * walk, wl)),
        "double": (fx * _phase(alpha, 2.0 * walk, wl),
                   fy * _phase(alpha, 2.0 * walk, wl)),
    }


def _cmp(arms, rx, ry):
    ref = np.concatenate([np.ravel(rx), np.ravel(ry)])
    return {k: _lib.rel(np.concatenate([np.ravel(a), np.ravel(b)]), ref)
            for k, (a, b) in arms.items()}


def _walk_helpers():
    """``(_layer_enters_slant_frame_1d, _slant_frame_walk_1d)``.

    The PRE-FIX trees do not define them (they SHIPPED with V2), so this probe
    carries the same two one-liners itself and says which it used -- otherwise
    the fail-before arm could not run at all."""
    try:
        from lumenairy.elements.pmm.stack import _layer_enters_slant_frame_1d, _slant_frame_walk_1d
        return _layer_enters_slant_frame_1d, _slant_frame_walk_1d, "library"
    except ImportError:
        def _enters(layer):
            return abs(float(layer[2])) > 1e-12

        def _walk(layers):
            return float(sum(np.tan(float(sl)) * float(t)
                             for t, _s, sl in layers if _enters((t, _s, sl))))
        return _enters, _walk, "probe-local"


def _armsJ(J, a0, walk, wl):
    """The candidate arms on a WHOLE 2x2 zeroth-order Jones, whose anchor is
    the single scalar ``P_0``."""
    p0 = complex(_phase(np.asarray([a0]), walk, wl)[0])
    F = np.asarray(J) / p0
    return {"shipped": np.asarray(J), "none": F,
            "conj": F * np.conj(p0),
            "half": F * complex(_phase(np.asarray([a0]), 0.5 * walk, wl)[0]),
            "double": F * complex(_phase(np.asarray([a0]), 2.0 * walk,
                                         wl)[0])}


def _cmpJ(arms, ref):
    return {k: _lib.rel(v, ref) for k, v in arms.items()}


def _read(st):
    Ex, Ey, alpha = _amps(st)
    Jt = np.asarray(st.jones_transmission())
    p0 = len(alpha) // 2
    return Ex, Ey, alpha, Jt, float(alpha[p0])


# ---------------------------------------------------------------------------


def ladder(out):
    """(a) MY OWN hand-built z-staircase ladder, both signs."""
    res = {}
    for tag, shear in (("pos", SHEAR), ("neg", -SHEAR)):
        spec = [(D, 0.5 - shear / 2.0, shear, E_R, E_G, DUTY)]
        walk = shear * P
        st = sheared_run(spec)
        _solve(st)
        Ex, Ey, alpha, Jt, a0 = _read(st)
        rEx, rEy, _ = _amps(st, "reflection")
        rows, prev, prevJ = {}, None, None
        for ns in LADDER:
            sc = hand_staircase(spec, ns)
            _solve(sc)
            sEx, sEy, _ = _amps(sc)
            sJt = np.asarray(sc.jones_transmission())
            srEx, srEy, _ = _amps(sc, "reflection")
            rows[str(ns)] = dict(
                per_order=_cmp(_arms(Ex, Ey, alpha, walk, WL), sEx, sEy),
                jones0=_cmp(_arms(Jt[0:1], Jt[1:2], np.asarray([a0]),
                                  walk, WL), sJt[0:1], sJt[1:2]),
                reflection_as_returned=_lib.rel(
                    np.concatenate([rEx.ravel(), rEy.ravel()]),
                    np.concatenate([srEx.ravel(), srEy.ravel()])),
                oracle_own_step=(None if prev is None else _lib.rel(
                    np.concatenate([sEx.ravel(), sEy.ravel()]),
                    np.concatenate([prev[0].ravel(), prev[1].ravel()]))),
                oracle_own_step_jones=(None if prevJ is None
                                       else _lib.rel(sJt, prevJ)))
            prev, prevJ = (sEx, sEy), sJt
        # the two GRID ROUTES of the oracle itself, at the coarsest rung
        sc_sh = hand_staircase(spec, LADDER[0], grids="shared")
        _solve(sc_sh)
        shEx, shEy, _ = _amps(sc_sh)
        sc_pl = hand_staircase(spec, LADDER[0])
        _solve(sc_pl)
        plEx, plEy, _ = _amps(sc_pl)
        rows["oracle_shared_vs_perlayer"] = _lib.rel(
            np.concatenate([shEx.ravel(), shEy.ravel()]),
            np.concatenate([plEx.ravel(), plEy.ravel()]))
        rows["shipped_vs_shared_grid_oracle"] = _cmp(
            _arms(Ex, Ey, alpha, walk, WL), shEx, shEy)
        # the LIBRARY's own taper builder, as a second opinion
        lib_sc = _stack()
        lib_sc.add_tapered_grating(D, eps_ridge=E_R, eps_groove=E_G,
                                   duty_bottom=DUTY, duty_top=DUTY,
                                   shear=shear, n_slices=LADDER[-1])
        lib_sc.set_source(WL, angle=THETA)
        _solve(lib_sc)
        lEx, lEy, _ = _amps(lib_sc)
        rows["library_taper_ns%d" % LADDER[-1]] = _cmp(
            _arms(Ex, Ey, alpha, walk, WL), lEx, lEy)
        res[tag] = rows
        print(f"  ladder_{tag}: " + " ".join(
            f"ns{ns}={rows[str(ns)]['per_order']['shipped']:.3e}"
            for ns in LADDER))
    out["ladder"] = res


def uniform_film(out):
    """(b) the ANALYTIC oracle -- a uniform slanted film IS the vertical film
    of the same eps (a shear of a homogeneous medium is a coordinate
    change)."""
    res = {}
    vt = _stack()
    vt.add_layer(D, eps=2.60)
    vt.set_source(WL, angle=THETA)
    o2, R2, T2, J2 = _solve(vt)
    vEx, vEy, _ = _amps(vt)
    vJt = np.asarray(vt.jones_transmission())
    vrEx, vrEy, _ = _amps(vt, "reflection")
    for tan_phi in (0.60, -0.60, 1.35):
        walk = tan_phi * D
        sl = _stack(factorization="convection")
        sl.add_layer(D, eps=2.60, slant_angle=float(np.arctan(tan_phi)))
        sl.set_source(WL, angle=THETA)
        o1, R1, T1, J1 = _solve(sl)
        Ex, Ey, alpha, Jt, a0 = _read(sl)
        rEx, rEy, _ = _amps(sl, "reflection")
        res[f"tan{tan_phi}"] = dict(
            per_order=_cmp(_arms(Ex, Ey, alpha, walk, WL), vEx, vEy),
            jones0=_cmp(_arms(Jt[0:1], Jt[1:2], np.asarray([a0]), walk, WL),
                        vJt[0:1], vJt[1:2]),
            reflection_as_returned=_lib.rel(
                np.concatenate([rEx.ravel(), rEy.ravel()]),
                np.concatenate([vrEx.ravel(), vrEy.ravel()])),
            dR=_lib.rel(R1, R2), dT=_lib.rel(T1, T2), dJrefl=_lib.rel(J1, J2),
            orders_same=bool(np.array_equal(o1, o2)), walk_m=walk)
        print(f"  uniform_film tan={tan_phi}: shipped="
              f"{res[f'tan{tan_phi}']['per_order']['shipped']:.3e} none="
              f"{res[f'tan{tan_phi}']['per_order']['none']:.3e}")
    out["uniform_film"] = res


def cross_engine(out):
    """(c) the INDEPENDENT pure staggered engine.  Its slant SIGN is
    adjudicated first by the ANCHOR-FREE reflection, then the transmission is
    read against whichever sign that picked."""
    from lumenairy.elements.pmm import PMM2DStackPure
    res = {}
    shear = SHEAR
    c_top = 0.5 - shear / 2.0
    spec = [(D, c_top, shear, E_R, E_G, DUTY)]
    walk = shear * P
    st = sheared_run(spec)
    _o, _R, _T, J1d = _solve(st)
    Ex, Ey, alpha, Jt, a0 = _read(st)
    lo, hi = (c_top - DUTY / 2.0) * P, (c_top + DUTY / 2.0) * P
    assert 0.0 < lo < hi < P, (lo, hi)

    def pure(t_x, nm):
        ps = PMM2DStackPure(P, P, n_substrate=N_SUB, n_superstrate=N_SUP,
                            n_modes=nm, n_orders=3, layer_grids="per-layer")
        # the staggered basis needs a SQUARE tile, so the y axis carries three
        # EQUAL segments of the identical material -- a y-uniform cell.
        cell = np.array([[E_G] * 3, [E_R] * 3, [E_G] * 3], dtype=float)
        ps.add_layer(D, eps_cell=cell, x_walls=np.array([lo, hi]),
                     y_walls=np.array([P / 3.0, 2.0 * P / 3.0]),
                     slant=(t_x, 0.0))
        ps.set_source(WL, theta=THETA)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = ps.solve()
            return (np.asarray(ps.jones_transmission()), np.asarray(J),
                    float(np.sum(R) + np.sum(T)))

    t_nom = float(shear * P / D)
    for nm in (4, 6):
        for sgn, t_x in (("plus", +t_nom), ("minus", -t_nom)):
            try:
                pJt, pJr, en = pure(t_x, nm)
            except Exception as exc:                        # noqa: BLE001
                res[f"nm{nm}_{sgn}"] = dict(outcome=type(exc).__name__,
                                            message=str(exc)[:200])
                continue
            res[f"nm{nm}_{sgn}"] = dict(
                outcome="SOLVED", energy=en,
                # the anchor-free adjudicator: the REFLECTION Jones
                refl_vs_1d=_lib.rel(pJr[:, 0], np.asarray(J1d)[:, 0]),
                jones0=_cmpJ(_armsJ(Jt, a0, walk, WL), pJt))
            print(f"  cross nm{nm} {sgn}: refl="
                  f"{res[f'nm{nm}_{sgn}']['refl_vs_1d']:.3e} "
                  f"jones0={res[f'nm{nm}_{sgn}']['jones0']}")
    # the pure engine's own resolution step (nm 6 -> 8, the winning sign)
    try:
        a6, _, _ = pure(+t_nom, 6)
        a8, _, _ = pure(+t_nom, 8)
        res["pure_own_step_6_to_8"] = _lib.rel(a6, a8)
        res["shipped_vs_pure_nm8"] = _cmpJ(_armsJ(Jt, a0, walk, WL), a8)
    except Exception as exc:                                # noqa: BLE001
        res["pure_own_step_6_to_8"] = f"raise:{type(exc).__name__}"
    out["cross_engine"] = res


def opposite(out):
    """(d) NEGATIVE and OPPOSITE shears; the NET-ZERO walk identity; and the
    composition claim against a hand-built two-run staircase."""
    _enters, _slant_frame_walk_1d, src = _walk_helpers()
    res = {"walk_helper_source": src}

    # -- the NET-ZERO walk ------------------------------------------------
    t = 0.26 * UM
    phi = float(np.arctan(0.30 * P / t))
    st = _stack(factorization="convection")
    st.add_layer(t, segments=[(DUTY, E_R), (1 - DUTY, E_G)], slant_angle=+phi)
    st.add_layer(t, segments=[(0.55, 3.1), (0.45, 1.1)], slant_angle=-phi)
    st.set_source(WL, angle=THETA)
    _solve(st)
    _Ex, _Ey, alpha = _amps(st)
    w_net = _slant_frame_walk_1d(st._layers)
    w_one = float(np.tan(phi) * t)
    res["net_zero"] = dict(
        walk_net=w_net, walk_net_is_exactly_zero=(w_net == 0.0),
        walk_one_layer=w_one,
        P_net_max_dev=float(np.max(np.abs(_phase(alpha, w_net, WL) - 1.0))),
        P_one_ptp_arg=float(np.ptp(np.angle(_phase(alpha, w_one, WL)))),
        P_one_max_dev=float(np.max(np.abs(_phase(alpha, w_one, WL) - 1.0))))
    print(f"  net_zero walk={w_net!r} exact0={w_net == 0.0} "
          f"one-layer ptp(arg)={res['net_zero']['P_one_ptp_arg']:.4f}")

    # -- an OPPOSITE-shear pair with a non-zero net walk ------------------
    #
    # THE FRAME-CONTINUATION SUBTLETY, measured here rather than assumed.
    # The cascade matches successive sheared layers' FRAME coefficients
    # directly, so layer 2 is solved in layer 1's frame: the ``centre``
    # handed to ``add_sheared_grating`` for the SECOND sheared layer is
    # interpreted at ``+ W1`` in the LAB.  Both placements of the oracle's
    # layer 2 are therefore built and compared -- ``naive`` (the docstring's
    # own lab reading of ``centre``) and ``frame`` (shifted by W1) -- and the
    # composition claim is read off whichever the cascade actually models.
    s1, s2 = 0.30, -0.12
    d1, d2 = 0.22 * UM, 0.16 * UM
    c1 = 0.5 - s1 / 2.0
    spec = [(d1, c1, s1, E_R, E_G, DUTY),
            (d2, c1 + s1, s2, 3.1, 1.1, 0.55)]
    spec_frame = [(d1, c1, s1, E_R, E_G, DUTY),
                  (d2, c1 + s1 + s1, s2, 3.1, 1.1, 0.55)]
    two = sheared_run(spec)
    _solve(two)
    tEx, tEy, talpha = _amps(two)
    W = s1 * P + s2 * P
    naive = {}
    for ns in (6, 12):
        sc = hand_staircase(spec, ns)
        _solve(sc)
        sEx, sEy, _ = _amps(sc)
        naive[str(ns)] = _cmp(_arms(tEx, tEy, talpha, W, WL), sEx, sEy)
    res["oracle_layer2_at_lab_centre"] = naive
    rows = {}
    prev = None
    for ns in (6, 12):
        sc = hand_staircase(spec_frame, ns)
        _solve(sc)
        sEx, sEy, _ = _amps(sc)
        fx = tEx / _phase(talpha, W, WL)
        fy = tEy / _phase(talpha, W, WL)
        arms = {
            "full": (tEx, tEy),
            "only_layer1": (fx * _phase(talpha, s1 * P, WL),
                            fy * _phase(talpha, s1 * P, WL)),
            "only_layer2": (fx * _phase(talpha, s2 * P, WL),
                            fy * _phase(talpha, s2 * P, WL)),
            "no_sum": (fx, fy),
            "conj_sum": (fx * np.conj(_phase(talpha, W, WL)),
                         fy * np.conj(_phase(talpha, W, WL))),
        }
        rows[str(ns)] = _cmp(arms, sEx, sEy)
        rows[str(ns)]["oracle_own_step"] = (
            None if prev is None else _lib.rel(
                np.concatenate([sEx.ravel(), sEy.ravel()]),
                np.concatenate([prev[0].ravel(), prev[1].ravel()])))
        prev = (sEx, sEy)
    res["opposite_pair"] = dict(walk_shipped=_slant_frame_walk_1d(
        two._layers), walk_expected=W, rows=rows,
        note="rows use the FRAME-CONTINUED oracle (layer 2 at +W1 in the "
             "lab); oracle_layer2_at_lab_centre is the same comparison "
             "against the docstring's own lab reading of `centre`.")
    print(f"  opposite_pair W={W:.4e} shipped="
          f"{res['opposite_pair']['walk_shipped']:.4e} {rows}")
    out["opposite"] = res


def breakage(out):                                           # noqa: C901
    """The break attempts."""
    (_layer_enters_slant_frame_1d, _slant_frame_walk_1d,
     src) = _walk_helpers()
    res = {"walk_helper_source": src}

    con = {}
    for fac in ("auto", "convection", "covariant"):
        for grids in ("shared", "per-layer"):
            st = _stack(factorization=fac, layer_grids=grids)
            st.add_sheared_grating(D, eps_ridge=E_R, eps_groove=E_G,
                                   duty=DUTY, shear=SHEAR)
            st.set_source(WL, angle=THETA, phi=float(np.deg2rad(37.0)))
            try:
                _solve(st)
                con[f"{fac}/{grids}"] = "SOLVED"
            except Exception as exc:                        # noqa: BLE001
                con[f"{fac}/{grids}"] = \
                    f"{type(exc).__name__}: {str(exc)[:110]}"
    vc = _stack()
    vc.add_layer(D, segments=[(DUTY, E_R), (1 - DUTY, E_G)])
    vc.set_source(WL, angle=THETA, phi=float(np.deg2rad(37.0)))
    _solve(vc)
    a = vc.per_order_amplitudes("transmission")
    con["vertical_conical_walk"] = _slant_frame_walk_1d(vc._layers)
    con["vertical_conical_max_abs_ky"] = float(np.max(np.abs(a["ky"])))
    res["conical"] = con
    print(f"  conical: {con}")

    wood = {}
    for f in (0.985, 0.997, 1.0, 1.003, 1.02):
        wl = -P * (N_SUB + np.sin(THETA)) / (-2.0) * f
        st = sheared_run(ONE)
        st.set_source(wl, angle=THETA)
        _solve(st)
        a = st.per_order_amplitudes("transmission")
        alpha, kz = np.asarray(a["kx"]), np.asarray(a["kz"])
        Pm = _phase(alpha, SHEAR * P, wl)
        wood[f"{f:.3f}"] = dict(
            max_abs_imag_alpha=float(np.max(np.abs(np.imag(alpha)))),
            alpha_dtype=str(alpha.dtype),
            n_evanescent=int(np.sum(np.real(kz) <= 0.0)),
            max_unimodular_dev=float(np.max(np.abs(np.abs(Pm) - 1.0))),
            min_abs_kz=float(np.min(np.abs(kz))),
            ptp_arg_P=float(np.ptp(np.angle(Pm))),
            n_distinct_P=int(len(np.unique(np.round(np.angle(Pm), 10)))))
    res["wood"] = wood
    print("  wood: " + " ".join(
        f"{k}:dev={v['max_unimodular_dev']:.2e}/ev={v['n_evanescent']}"
        for k, v in wood.items()))

    lossy = {}
    for nsup in (1.0, 1.0 + 0.02j, 1.0 + 0.4j):
        try:
            st = sheared_run(ONE, n_superstrate=nsup)
            _solve(st)
            a = st.per_order_amplitudes("transmission")
            alpha = np.asarray(a["kx"])
            lossy[str(nsup)] = dict(
                outcome="SOLVED",
                max_abs_imag_alpha=float(np.max(np.abs(np.imag(alpha)))),
                max_unimodular_dev=float(np.max(np.abs(
                    np.abs(_phase(alpha, SHEAR * P, WL)) - 1.0))))
        except Exception as exc:                            # noqa: BLE001
            lossy[str(nsup)] = dict(outcome=type(exc).__name__,
                                    message=str(exc)[:150])
    res["lossy_superstrate"] = lossy
    print(f"  lossy_superstrate: {lossy}")

    ri = {}
    for grids in ("shared", "per-layer"):
        for fac in ("auto", "convection"):
            st = _stack(factorization=fac, layer_grids=grids)
            st.add_sheared_grating(D, eps_ridge=E_R, eps_groove=E_G,
                                   duty=DUTY, shear=SHEAR)
            st.set_source(WL, angle=THETA)
            try:
                _solve(st, retain_internal=True)
                ri[f"{fac}/{grids}"] = "SOLVED"
                try:
                    st.internal_field(np.array([0.1e-6]))
                    ri[f"{fac}/{grids}"] += " + internal_field SOLVED"
                except Exception as exc:                    # noqa: BLE001
                    ri[f"{fac}/{grids}"] += \
                        f" + internal_field {type(exc).__name__}"
            except Exception as exc:                        # noqa: BLE001
                ri[f"{fac}/{grids}"] = \
                    f"{type(exc).__name__}: {str(exc)[:110]}"
    mx = _stack(factorization="convection")
    mx.add_layer(0.20 * UM, segments=[(0.6, 3.1), (0.4, 1.1)])
    mx.add_sheared_grating(D, eps_ridge=E_R, eps_groove=E_G, duty=DUTY,
                           shear=SHEAR)
    mx.set_source(WL, angle=THETA)
    try:
        _solve(mx, retain_internal=True)
        ri["mixed_vertical_plus_sheared"] = "SOLVED"
    except Exception as exc:                                # noqa: BLE001
        ri["mixed_vertical_plus_sheared"] = \
            f"{type(exc).__name__}: {str(exc)[:110]}"
    res["retain_internal"] = ri
    print(f"  retain_internal: {ri}")

    sw = {}
    st = sheared_run(ONE)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = st.solve_vs_wavelength(np.array([WL, WL * 1.01]), angle=THETA)
        sw["outcome"] = "SOLVED"
        sw["n_returns"] = len(r) if isinstance(r, tuple) else 1
        try:
            st.per_order_amplitudes("transmission")
            sw["amps_after_sweep"] = "AVAILABLE"
        except Exception as exc:                            # noqa: BLE001
            sw["amps_after_sweep"] = type(exc).__name__
    except Exception as exc:                                # noqa: BLE001
        sw["outcome"] = f"{type(exc).__name__}: {str(exc)[:150]}"
    res["solve_vs_wavelength"] = sw
    print(f"  solve_vs_wavelength: {sw}")

    rt = {}
    for sl in (0.0, 5e-13, 1e-12, 1.0000001e-12, 2e-12, 1e-9):
        st = _stack(factorization="convection")
        st.add_layer(D, segments=[(DUTY, E_R), (1 - DUTY, E_G)],
                     slant_angle=sl)
        st.set_source(WL, angle=THETA)
        rt[f"{sl:.7e}"] = dict(
            enters_frame=bool(_layer_enters_slant_frame_1d(st._layers[0])),
            walk=_slant_frame_walk_1d(st._layers))
    res["routing_literal"] = rt
    print(f"  routing_literal: {rt}")

    order = {}
    for tag, first in (("shear_on_top", True), ("shear_below", False)):
        st = _stack(factorization="convection")
        if first:
            st.add_sheared_grating(D, eps_ridge=E_R, eps_groove=E_G,
                                   duty=DUTY, shear=SHEAR)
            st.add_layer(0.2 * UM, segments=[(0.6, 3.1), (0.4, 1.1)])
        else:
            st.add_layer(0.2 * UM, segments=[(0.6, 3.1), (0.4, 1.1)])
            st.add_sheared_grating(D, eps_ridge=E_R, eps_groove=E_G,
                                   duty=DUTY, shear=SHEAR)
        st.set_source(WL, angle=THETA)
        _solve(st)
        order[tag] = dict(walk=_slant_frame_walk_1d(st._layers),
                          expected=float(SHEAR * P))
    res["layer_order_walk"] = order
    print(f"  layer_order_walk: {order}")

    out["breakage"] = res


def main():
    t0 = time.time()
    out = {"fixture": dict(period=P, wavelength=WL, thickness=D,
                           eps_ridge=E_R, eps_groove=E_G, duty=DUTY,
                           shear=SHEAR, theta_deg=31.0, n_sub=N_SUB,
                           n_sup=N_SUP, degree=DEG, n_orders=NORD,
                           ladder=list(LADDER))}
    ladder(out)
    uniform_film(out)
    cross_engine(out)
    opposite(out)
    breakage(out)
    out["total_secs"] = round(time.time() - t0, 1)
    _lib.save("t2_v2_sign", out)


if __name__ == "__main__":
    main()
