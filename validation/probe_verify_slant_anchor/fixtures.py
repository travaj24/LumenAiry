"""The verification's OWN fixture set -- 34 ``PMMStack`` mounts plus the five
2-D / single-layer entries -- and the surface hasher that reads every
observable each one exposes.

None of these fixtures is copied from the fix's own probe directory: the
periods, wavelengths, permittivities, duties, shears, thicknesses, mounts,
degrees and order counts are all chosen here, and the sheared members
deliberately span BOTH signs, OPPOSITE-shear pairs, a net-zero walk, a uniform
slanted film, a slanted layer BELOW a vertical patterned one, and both grid
routes.

Every builder returns ``(stack, solve_kwargs)``; ``run_fixture`` drives it and
returns a dict of sha256 digests, one per SURFACE, with the string
``"raise:<ExceptionName>"`` where a surface is unreachable.  A digest that
moves between two trees is a byte that moved.
"""
from __future__ import annotations

import warnings

import numpy as np
from _lib import sha

UM = 1e-6

# ---------------------------------------------------------------------------
# 1-D PMMStack fixtures
# ---------------------------------------------------------------------------


def _S(**kw):
    from lumenairy.elements.pmm import PMMStack
    kw.setdefault("period", 0.72 * UM)
    kw.setdefault("n_substrate", 1.62)
    kw.setdefault("n_superstrate", 1.0)
    kw.setdefault("degree", 10)
    kw.setdefault("n_orders", 7)
    return PMMStack(**kw)


def _binary(duty=0.42, er=4.6, eg=1.35):
    return [(duty, er), (1.0 - duty, eg)]


def _tensor_inplane(exx=3.4, eyy=2.1, exy=0.35):
    m = np.eye(3, dtype=complex) * 1.0
    m[0, 0], m[1, 1], m[2, 2] = exx, eyy, 2.7
    m[0, 1] = m[1, 0] = exy
    return m


def _tensor_oop(exx=3.4, ezz=2.9, exz=0.55):
    m = np.eye(3, dtype=complex)
    m[0, 0], m[1, 1], m[2, 2] = exx, 2.2, ezz
    m[0, 2] = m[2, 0] = exz
    return m


# ---- vertical -------------------------------------------------------------

def f_vert_binary_ob25():
    s = _S()
    s.add_layer(0.38 * UM, segments=_binary())
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_binary_norm():
    s = _S()
    s.add_layer(0.38 * UM, segments=_binary())
    s.set_source(0.56 * UM, angle=0.0)
    return s, {}


def f_vert_binary_ob40():
    s = _S()
    s.add_layer(0.38 * UM, segments=_binary())
    s.set_source(0.56 * UM, angle=np.deg2rad(40.0))
    return s, {}


def f_vert_two_layer():
    s = _S()
    s.add_layer(0.22 * UM, segments=_binary(0.55, 5.1, 1.0))
    s.add_layer(0.30 * UM, segments=_binary(0.33, 2.9, 1.9))
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_film():
    s = _S()
    s.add_layer(0.40 * UM, eps=2.60)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_tensor_inplane():
    s = _S()
    s.add_layer(0.30 * UM, segments=[(0.45, _tensor_inplane()), (0.55, 1.4)])
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_tensor_oop_shared():
    s = _S()
    s.add_layer(0.30 * UM, segments=[(0.45, _tensor_oop()), (0.55, 1.4)])
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_tensor_oop_perlayer():
    s = _S(layer_grids="per-layer")
    s.add_layer(0.30 * UM, segments=[(0.45, _tensor_oop()), (0.55, 1.4)])
    s.add_layer(0.18 * UM, segments=_binary(0.61, 3.1, 1.1))
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_perlayer():
    s = _S(layer_grids="per-layer")
    s.add_layer(0.22 * UM, segments=_binary(0.55, 5.1, 1.0))
    s.add_layer(0.30 * UM, segments=_binary(0.33, 2.9, 1.9))
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_retain_internal():
    s = _S()
    s.add_layer(0.38 * UM, segments=_binary())
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {"retain_internal": True}


def f_vert_lossy():
    s = _S()
    s.add_layer(0.30 * UM, segments=_binary(0.42, 4.6 + 0.35j, 1.35))
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {"retain_internal": True}


def f_vert_conical():
    s = _S()
    s.add_layer(0.38 * UM, segments=_binary())
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0), phi=np.deg2rad(33.0))
    return s, {}


def f_vert_stabilize_slices():
    s = _S()
    s.add_tapered_grating(0.36 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty_bottom=0.52, duty_top=0.30, n_slices=3)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {"stabilize": "slices"}


def f_vert_taper_staircase():
    s = _S()
    s.add_tapered_grating(0.36 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty_bottom=0.52, duty_top=0.30, n_slices=4)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_tapered_ridges():
    s = _S()
    P = 0.72 * UM
    s.add_tapered_ridges(0.30 * UM,
                         ridges=[(0.28 * P, 0.14 * P, 0.22 * P, 4.6),
                                 (0.70 * P, 0.18 * P, 0.18 * P, 3.0)],
                         eps_groove=1.2, n_slices=3)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_vert_three_layer_bragg():
    s = _S()
    for _ in range(3):
        s.add_layer(0.14 * UM, segments=_binary(0.5, 4.0, 1.0))
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


# ---- sheared --------------------------------------------------------------

def f_shear_pos_ob25():
    s = _S()
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_neg_ob25():
    s = _S()
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=-0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_pos_ob40():
    s = _S()
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(40.0))
    return s, {}


def f_shear_pos_norm():
    s = _S()
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=0.0)
    return s, {}


def f_shear_perlayer():
    s = _S(layer_grids="per-layer")
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_big():
    s = _S()
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.62)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_lossy():
    s = _S()
    s.add_sheared_grating(0.30 * UM, eps_ridge=4.6 + 0.35j, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_two_same_sign():
    s = _S()
    s.add_sheared_grating(0.24 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.22)
    s.add_sheared_grating(0.18 * UM, eps_ridge=3.1, eps_groove=1.1,
                          duty=0.55, shear=0.13)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_two_opposite():
    """OPPOSITE shears, DIFFERENT thicknesses -- a non-zero net walk built
    from two walks that partly cancel."""
    s = _S()
    s.add_sheared_grating(0.24 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.30)
    s.add_sheared_grating(0.18 * UM, eps_ridge=3.1, eps_groove=1.1,
                          duty=0.55, shear=-0.13)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_net_zero_walk():
    """Two sheared layers whose walks CANCEL EXACTLY: the layer thicknesses
    and the slant angles are set so ``tan(phi_1) d_1 = -tan(phi_2) d_2``."""
    s = _S()
    t = 0.26 * UM
    phi = np.arctan(0.30 * 0.72 * UM / t)
    s.add_layer(t, segments=_binary(0.42, 4.6, 1.35), slant_angle=+phi)
    s.add_layer(t, segments=_binary(0.55, 3.1, 1.10), slant_angle=-phi)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_over_vertical_pattern():
    """A VERTICAL patterned layer on top of a SHEARED one -- so the walk is
    contributed by the LOWER layer only, and the frame does not start at the
    stack's top face."""
    s = _S()
    s.add_layer(0.20 * UM, segments=_binary(0.61, 3.1, 1.1))
    s.add_sheared_grating(0.28 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    return_ = s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    del return_
    return s, {}


def f_shear_under_vertical_pattern():
    """The SHEARED layer on top, a VERTICAL patterned layer beneath it."""
    s = _S()
    s.add_sheared_grating(0.28 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.add_layer(0.20 * UM, segments=_binary(0.61, 3.1, 1.1))
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_uniform_slanted_film():
    """A UNIFORM slanted layer -- the analytic oracle: a shear of a
    homogeneous medium is a pure coordinate change."""
    s = _S()
    t = 0.40 * UM
    s.add_layer(t, eps=2.60, slant_angle=float(np.arctan(0.60)))
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_with_oop():
    """A SHEARED layer stacked with an OUT-OF-PLANE tensor layer."""
    s = _S()
    s.add_sheared_grating(0.24 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.add_layer(0.18 * UM, segments=[(0.45, _tensor_oop()), (0.55, 1.4)])
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_covariant():
    """``factorization='covariant'`` on a UNIFORM-slant stack -- the cascade
    that retains NO amplitudes at all."""
    s = _S(factorization="covariant")
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_conical_refused():
    """Conical incidence on a SHEARED stack -- the documented refusal."""
    s = _S()
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0), phi=np.deg2rad(33.0))
    return s, {}


def f_shear_retain_internal_refused():
    s = _S()
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {"retain_internal": True}


def f_shear_tiny_below_bar():
    """``slant_angle = 5e-13`` -- BELOW the ``1e-12`` routing literal, so the
    stack must take the all-vertical symmetric cascade and the walk must be
    exactly ``0.0``."""
    s = _S()
    s.add_layer(0.38 * UM, segments=_binary(), slant_angle=5e-13)
    s.set_source(0.56 * UM, angle=np.deg2rad(25.0))
    return s, {}


def f_shear_wl_near_wood():
    """A wavelength placed where a diffracted order sits within 0.3 % of its
    Rayleigh (Wood) cut-off in the substrate."""
    s = _S(n_substrate=1.62, n_orders=9)
    s.add_sheared_grating(0.38 * UM, eps_ridge=4.6, eps_groove=1.35,
                          duty=0.42, shear=0.28)
    # order m = -2 grazes in the substrate when |kx0 + m K| = k0 n_sub.
    # period 0.72, n_sub 1.62, theta 25 deg  ->  wl solved below.
    P, nsub, th = 0.72 * UM, 1.62, np.deg2rad(25.0)
    # (sin th + m wl / P) = -nsub   ->   wl = -P (nsub + sin th) / m , m = -2
    wl = -P * (nsub + np.sin(th)) / (-2.0)
    s.set_source(wl * 0.997, angle=th)
    return s, {}


FIXTURES_1D = {
    "vert_binary_ob25": f_vert_binary_ob25,
    "vert_binary_norm": f_vert_binary_norm,
    "vert_binary_ob40": f_vert_binary_ob40,
    "vert_two_layer": f_vert_two_layer,
    "vert_film": f_vert_film,
    "vert_tensor_inplane": f_vert_tensor_inplane,
    "vert_tensor_oop_shared": f_vert_tensor_oop_shared,
    "vert_tensor_oop_perlayer": f_vert_tensor_oop_perlayer,
    "vert_perlayer": f_vert_perlayer,
    "vert_retain_internal": f_vert_retain_internal,
    "vert_lossy": f_vert_lossy,
    "vert_conical": f_vert_conical,
    "vert_stabilize_slices": f_vert_stabilize_slices,
    "vert_taper_staircase": f_vert_taper_staircase,
    "vert_tapered_ridges": f_vert_tapered_ridges,
    "vert_three_layer_bragg": f_vert_three_layer_bragg,
    "shear_pos_ob25": f_shear_pos_ob25,
    "shear_neg_ob25": f_shear_neg_ob25,
    "shear_pos_ob40": f_shear_pos_ob40,
    "shear_pos_norm": f_shear_pos_norm,
    "shear_perlayer": f_shear_perlayer,
    "shear_big": f_shear_big,
    "shear_lossy": f_shear_lossy,
    "shear_two_same_sign": f_shear_two_same_sign,
    "shear_two_opposite": f_shear_two_opposite,
    "shear_net_zero_walk": f_shear_net_zero_walk,
    "shear_over_vertical_pattern": f_shear_over_vertical_pattern,
    "shear_under_vertical_pattern": f_shear_under_vertical_pattern,
    "uniform_slanted_film": f_uniform_slanted_film,
    "shear_with_oop": f_shear_with_oop,
    "shear_covariant": f_shear_covariant,
    "shear_conical_refused": f_shear_conical_refused,
    "shear_retain_internal_refused": f_shear_retain_internal_refused,
    "shear_tiny_below_bar": f_shear_tiny_below_bar,
    "shear_wl_near_wood": f_shear_wl_near_wood,
}



def _convection(fn):
    """A twin of ``fn`` forced onto ``factorization='convection'``.

    WHY THESE EXIST.  ``factorization='auto'`` sends an IN-PLANE UNIFORM-slant
    stack to the COVARIANT (spectral) cascade, which retains NO per-order
    amplitudes at all -- so on the library's DEFAULT setting a single sheared
    grating exposes no transmitted field and the anchor is unreachable.  Only
    the CONVECTION cascade retains amplitudes, and it is reached by default
    only for a MIXED-slant (or slanted out-of-plane) stack.  Every
    single-sheared-layer fixture therefore carries a convection twin, so the
    anchor is exercised on both signs, both mounts, both grid routes, the
    lossy row and the near-Wood row -- not only on the multi-layer members.
    """
    def g():
        st, kw = fn()
        st.factorization = "convection"
        return st, kw
    g.__name__ = fn.__name__ + "_conv"
    return g


for _base in ("shear_pos_ob25", "shear_neg_ob25", "shear_pos_ob40",
              "shear_pos_norm", "shear_perlayer", "shear_big", "shear_lossy",
              "uniform_slanted_film", "shear_wl_near_wood",
              "shear_conical_refused", "shear_retain_internal_refused"):
    FIXTURES_1D[_base + "_conv"] = _convection(FIXTURES_1D[_base])
del _base

SHEARED = {k for k in FIXTURES_1D if k.startswith("shear")
           or k.startswith("uniform_slanted")}

SURFACES = ("orders", "R", "T", "Jrefl", "Jtrans", "perT_Ex", "perT_Ey",
            "perR_Ex", "perR_Ey", "bridgeT", "bridgeR", "internal",
            "absorb")
TRANSMISSION_SURFACES = ("Jtrans", "perT_Ex", "perT_Ey", "bridgeT")


def run_fixture(name):
    """Every surface the fixture exposes, as sha256 digests."""
    from lumenairy.elements.polarization import jones_field_from_orders
    build, kw = FIXTURES_1D[name]()
    out = {k: None for k in SURFACES}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            orders, R, T, J = build.solve(**kw)
        except Exception as exc:                            # noqa: BLE001
            for k in SURFACES:
                out[k] = f"raise:{type(exc).__name__}"
            out["_solve_error"] = str(exc)[:220]
            return out
        out["orders"] = sha(np.asarray(orders))
        out["R"] = sha(np.asarray(R))
        out["T"] = sha(np.asarray(T))
        out["Jrefl"] = sha(np.asarray(J))
        out["_RT"] = float(np.max(np.sum(np.asarray(R), axis=1)
                                  + np.sum(np.asarray(T), axis=1)))
        try:
            out["Jtrans"] = sha(np.asarray(build.jones_transmission()))
        except Exception as exc:                            # noqa: BLE001
            out["Jtrans"] = f"raise:{type(exc).__name__}"
        for port, tag in (("transmission", "perT"), ("reflection", "perR")):
            try:
                a = build.per_order_amplitudes(port)
            except Exception as exc:                        # noqa: BLE001
                out[tag + "_Ex"] = out[tag + "_Ey"] = \
                    f"raise:{type(exc).__name__}"
                out["bridge" + tag[-1].upper()] = f"raise:{type(exc).__name__}"
                continue
            out[tag + "_Ex"] = sha(np.asarray(a["Ex"]))
            out[tag + "_Ey"] = sha(np.asarray(a["Ey"]))
            try:
                f = jones_field_from_orders(a, 12, 1, build.period / 12.0)
                out["bridge" + tag[-1].upper()] = sha(
                    np.asarray(f.Ex), np.asarray(f.Ey))
            except Exception as exc:                        # noqa: BLE001
                out["bridge" + tag[-1].upper()] = \
                    f"raise:{type(exc).__name__}"
        if kw.get("retain_internal"):
            try:
                z = np.linspace(0.02e-6, 0.30e-6, 5)
                fld = build.internal_field(z, incident=(1.0, 0.0))
                out["internal"] = sha(*[np.asarray(v) for _k, v in
                                        sorted(fld.items())
                                        if isinstance(v, np.ndarray)])
            except Exception as exc:                        # noqa: BLE001
                out["internal"] = f"raise:{type(exc).__name__}"
            try:
                out["absorb"] = sha(np.asarray(build.layer_absorption()))
            except Exception as exc:                        # noqa: BLE001
                out["absorb"] = f"raise:{type(exc).__name__}"
    return out
