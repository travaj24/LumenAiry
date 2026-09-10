"""V2 -- the CENSUS: every output surface of ``PMMStack``, hashed.

The anchor must move the TRANSMISSION-side surfaces of a stack that holds a
SHEARED layer and NOTHING else, on any fixture whatsoever.  This probe hashes
eleven surfaces on twenty-four fixtures -- eighteen with no shear anywhere (the
bit-identity set) and six sheared ones -- so the blast radius is a measured
set, not an argument.

Surfaces (sha256 of the raw bytes):

  ``orders`` ``R`` ``T`` ``Jrefl``   -- from ``solve()``; anchor-free
  ``Jtrans``                         -- ``jones_transmission()``
  ``perT_Ex`` ``perT_Ey``            -- ``per_order_amplitudes('transmission')``
  ``perR_Ex`` ``perR_Ey``            -- ``per_order_amplitudes('reflection')``
  ``bridgeT`` ``bridgeR``            -- ``jones_field_from_orders`` on each
                                        port's dict (the propagator bridge)
  ``internal`` ``absorb``            -- ``internal_field`` / ``layer_absorption``
                                        where ``retain_internal`` is available
"""
from __future__ import annotations

import math
import sys
import time
import warnings

import _lib as L
import numpy as np

P = 0.80e-6
WL = 0.55e-6
D = 0.40e-6
ER, EG = 4.20, 1.45
NSUP, NSUB = 1.0, 1.6
DEG, NORD = 12, 7
TH0, TH25, TH40 = 0.0, math.radians(25.0), math.radians(40.0)

_INPLANE = np.diag([3.1, 2.2, 2.6]).astype(complex)
_OOP = np.array([[3.1, 0.0, 0.7], [0.0, 2.2, 0.0], [0.7, 0.0, 2.6]],
                dtype=complex)


def _stack(**kw):
    from lumenairy.elements.pmm.stack import PMMStack
    kw.setdefault("factorization", "convection")
    return PMMStack(P, n_superstrate=NSUP, n_substrate=NSUB, degree=DEG,
                    n_orders=NORD, **kw)


# ---------------------------------------------------------------- fixtures
def f_vertical(theta=TH25, **kw):
    def go():
        st = _stack(**kw)
        st.add_layer(D, segments=[(0.45, ER), (0.55, EG)])
        st.set_source(WL, theta=theta)
        return st
    return go


def f_three_layer():
    def go():
        st = _stack()
        st.add_layer(0.12e-6, eps=2.10)
        st.add_layer(D, segments=[(0.45, ER), (0.55, EG)])
        st.add_layer(0.09e-6, segments=[(0.3, 2.9), (0.7, 1.6)])
        st.set_source(WL, theta=TH25)
        return st
    return go


def f_films():
    def go():
        st = _stack()
        st.add_layer(0.16e-6, eps=2.10)
        st.add_layer(0.11e-6, eps=3.40)
        st.set_source(WL, theta=TH25)
        return st
    return go


def f_tapered(ns=6):
    def go():
        st = _stack()
        st.add_tapered_grating(D, eps_ridge=ER, eps_groove=EG,
                               duty_bottom=0.30, duty_top=0.62, n_slices=ns)
        st.set_source(WL, theta=TH25)
        return st
    return go


def f_ridges():
    def go():
        st = _stack()
        st.add_tapered_ridges(D, ridges=[(0.30 * P, 0.18 * P, 0.18 * P, ER),
                                         (0.72 * P, 0.14 * P, 0.14 * P, 2.6)],
                              eps_groove=EG, n_slices=4)
        st.set_source(WL, theta=TH25)
        return st
    return go


def f_tensor(mat=_INPLANE, theta=TH25, **kw):
    def go():
        st = _stack(**kw)
        st.add_layer(D, segments=[(0.45, mat), (0.55, np.eye(3) * EG)])
        st.set_source(WL, theta=theta)
        return st
    return go


def f_conical():
    def go():
        st = _stack()
        st.add_layer(D, segments=[(0.45, ER), (0.55, EG)])
        st.set_source(WL, theta=TH25, phi=math.radians(35.0))
        return st
    return go


def f_lossy():
    def go():
        st = _stack()
        st.add_layer(D, segments=[(0.45, ER + 0.35j), (0.55, EG)])
        st.set_source(WL, theta=TH25)
        return st
    return go


def f_sheared(*, shear=0.30, theta=TH25, film=None, below=None, **kw):
    def go():
        st = _stack(**kw)
        st.add_sheared_grating(D, eps_ridge=ER, eps_groove=EG, duty=0.45,
                               shear=shear)
        if film is not None:
            st.add_layer(film, eps=3.40)
        if below is not None:
            st.add_layer(below, segments=[(0.35, 2.9), (0.65, 1.6)])
        st.set_source(WL, theta=theta)
        return st
    return go


def f_two_sheared():
    def go():
        st = _stack()
        st.add_sheared_grating(0.30e-6, eps_ridge=ER, eps_groove=EG,
                               duty=0.45, shear=0.25)
        st.add_sheared_grating(0.15e-6, eps_ridge=2.9, eps_groove=1.6,
                               duty=0.55, shear=-0.10)
        st.set_source(WL, theta=TH25)
        return st
    return go


def f_uniform_slanted():
    def go():
        st = _stack()
        st.add_layer(D, eps=2.60, slant_angle=math.atan(0.30 * P / D))
        st.set_source(WL, theta=TH25)
        return st
    return go


FIXTURES = {
    # ---- eighteen with NO shear anywhere: the bit-identity set ------------
    "vert_normal": (f_vertical(TH0), {}),
    "vert_ob25": (f_vertical(TH25), {}),
    "vert_ob40": (f_vertical(TH40), {}),
    "vert_ob25_retain": (f_vertical(TH25), dict(retain_internal=True)),
    "vert_ob25_slices": (f_vertical(TH25), dict(stabilize="slices")),
    "vert_perlayer": (f_vertical(TH25, layer_grids="per-layer"), {}),
    "three_layer": (f_three_layer(), {}),
    "three_layer_retain": (f_three_layer(), dict(retain_internal=True)),
    "films": (f_films(), {}),
    "films_retain": (f_films(), dict(retain_internal=True)),
    "tapered_ns6": (f_tapered(6), {}),
    "tapered_ns12": (f_tapered(12), {}),
    "ridges": (f_ridges(), {}),
    "tensor_inplane": (f_tensor(), {}),
    "tensor_oop": (f_tensor(_OOP), {}),
    "tensor_oop_perlayer": (f_tensor(_OOP, layer_grids="per-layer"), {}),
    "conical": (f_conical(), {}),
    "lossy": (f_lossy(), {}),
    # ---- six SHEARED ------------------------------------------------------
    "shear_normal": (f_sheared(theta=TH0), {}),
    "shear_ob25": (f_sheared(), {}),
    "shear_neg_ob25": (f_sheared(shear=-0.30), {}),
    "shear_perlayer_ob25": (f_sheared(layer_grids="per-layer"), {}),
    "shear_over_film": (f_sheared(film=0.12e-6), {}),
    "shear_over_pattern": (f_sheared(below=0.10e-6), {}),
    "two_sheared": (f_two_sheared(), {}),
    "uniform_slanted": (f_uniform_slanted(), {}),
    # the COVARIANT route, which retains no amplitudes at all -- included so
    # "not affected" is a measured row rather than a claim
    "shear_covariant_ob25": (f_sheared(factorization="covariant"), {}),
}


def surfaces(build, solve_kw):
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = build()
        try:
            o, R, T, J = st.solve(**solve_kw)
        except Exception as e:                            # noqa: BLE001
            return dict(_solve="RAISE %s: %s" % (type(e).__name__,
                                                 str(e)[:120]))
        out["orders"] = L.sha(o)
        out["R"] = L.sha(R)
        out["T"] = L.sha(T)
        out["Jrefl"] = L.sha(J)
        try:
            out["Jtrans"] = L.sha(st.jones_transmission())
        except Exception as e:                            # noqa: BLE001
            out["Jtrans"] = "NA:" + type(e).__name__
        from lumenairy.elements.polarization import jones_field_from_orders
        for port, tag in (("transmission", "T"), ("reflection", "R")):
            try:
                a = st.per_order_amplitudes(port)
            except Exception as e:                        # noqa: BLE001
                out["per%s_Ex" % tag] = "NA:" + type(e).__name__
                out["per%s_Ey" % tag] = "NA:" + type(e).__name__
                out["bridge%s" % tag] = "NA:" + type(e).__name__
                continue
            out["per%s_Ex" % tag] = L.sha(a["Ex"])
            out["per%s_Ey" % tag] = L.sha(a["Ey"])
            try:
                jf = jones_field_from_orders(a, 24, 4, P / 24.0)
                out["bridge%s" % tag] = L.sha(np.stack(
                    [np.asarray(jf.Ex), np.asarray(jf.Ey)]))
            except Exception as e:                        # noqa: BLE001
                out["bridge%s" % tag] = "NA:" + type(e).__name__
        if solve_kw.get("retain_internal"):
            try:
                f = st.internal_field(np.linspace(1e-9, D - 1e-9, 5))
                out["internal"] = L.sha(np.stack(
                    [np.asarray(f[k]) for k in ("Ex", "Ey", "Ez")]))
            except Exception as e:                        # noqa: BLE001
                out["internal"] = "NA:" + type(e).__name__
            try:
                out["absorb"] = L.sha(st.layer_absorption())
            except Exception as e:                        # noqa: BLE001
                out["absorb"] = "NA:" + type(e).__name__
    return out


def main():
    t0 = time.time()
    res = {}
    for name, (build, kw) in FIXTURES.items():
        res[name] = surfaces(build, kw)
        print("%-22s %s" % (name, sorted(res[name])))
    res["_seconds"] = round(time.time() - t0, 1)
    L.dump("v2_census", res, suffix=(sys.argv[1] if len(sys.argv) > 1 else ""))


if __name__ == "__main__":
    main()
