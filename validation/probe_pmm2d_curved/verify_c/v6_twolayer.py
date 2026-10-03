"""V6 -- C6 on the verifier's own two-layer merged stacks.

  python v6_twolayer.py <scenario> <M>
scenario: annulus (r 0.25 in layer 1 / r 0.45 in layer 2), circle_sine (a
sinusoidal half-plane wall beside a circle's transition cell in layer 1, the
circle in layer 2), fillets (fillet 0.5 / r 0.05 in layer 1 nested in fillet
0.9 / r 0.12 in layer 2), phaseb (the r 0.36 circle in layer 1, layer 2
painted VACUUM with the same circle -- the Phase B map exactly).

Quantities:
  closure       : lossless, max |1 - sum R - sum T| (both inputs);
  abs_lossless  : layer 2 lossy (eps + 0.4i), |absorption| of the LOSSLESS
                  layer 1 (must be round-off);
  abs_balance   : |sum layer_absorption - (1 - sum R - sum T)|;
  vacuum_id     : layer 2's shapes painted eps = background = 1 against a
                  UNIFORM eps = 1 layer 2 on the SAME merged map (layer 2's
                  shapes kept as no-op paint in layer 1): max |R, T diff|;
  phaseb        : (scenario phaseb) the stack against Phase B's explicit
                  _circle_map_3x3 + eps_cell + a uniform vacuum layer 2:
                  max |R, T diff| (same map -> round-off expected).
"""
import sys
import time
import warnings

import numpy as np
from _vc import BUILD, dump

from lumenairy.elements.pmm import (
    Circle,
    FilletRect,
    PMM2DStackPure,
    SinusoidalWall,
    _curvemap as CM,
)

warnings.simplefilter("ignore")
P, WL = 1.2, 1.0
T1, T2 = 0.3, 0.25


def scen(name, e2):
    c = 0.6
    if name == "annulus":
        return [Circle(c, c, 0.25, 4.0)], [Circle(c, c, 0.45, e2)]
    if name == "circle_sine":
        return ([SinusoidalWall("x", 0.15, 0.05, eps=2.25)],
                [Circle(c, c, 0.3, e2)])
    if name == "fillets":
        return ([FilletRect(c, c, 0.5, 0.5, 0.05, 4.0)],
                [FilletRect(c, c, 0.9, 0.9, 0.12, e2)])
    if name == "phaseb":
        return [Circle(c, c, 0.36, 4.0)], [Circle(c, c, 0.36, e2)]
    raise SystemExit(name)


def solve(l1, l2, M, uniform2=None, theta=0.15, phi=0.3, retain=False):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2)
    st.add_layer(T1, shapes=l1, background_eps=1.0)
    if uniform2 is None:
        st.add_layer(T2, shapes=l2, background_eps=1.0)
    else:
        st.add_layer(T2, eps=uniform2)
    st.set_source(WL, theta=theta, phi=phi)
    o, R, T, J = st.solve(retain_internal=retain)
    ab = st.layer_absorption() if retain else None
    return st, np.asarray(R), np.asarray(T), ab


def main(name, M):
    res = {"scenario": name, "M": M}
    t0 = time.time()
    l1, l2 = scen(name, 2.25)
    st, R, T, _ = solve(l1, l2, M)
    res["grid"] = list(st._grid)
    res["closure"] = float(np.max(np.abs(1 - R.sum(1) - T.sum(1))))
    l1, l2 = scen(name, 2.25 + 0.4j)
    st, R, T, ab = solve(l1, l2, M, retain=True)
    ab = np.asarray(ab)
    res["absorption"] = ab.tolist() if ab.ndim else float(ab)
    res["abs_lossless_layer1"] = float(np.max(np.abs(ab[0])))
    res["abs_balance"] = float(np.max(np.abs(np.sum(ab, 0)
                                             - (1 - R.sum(1) - T.sum(1)))))
    # vacuum identity on the same merged map
    l1, l2 = scen(name, 1.0)
    _st, Ra, Ta, _ = solve(l1, l2, M)
    l1b = list(l2) + list(l1)     # layer 2's shapes as NO-OP paint first
    _st2, Rb, Tb, _ = solve(l1b if name != "phaseb" else l1, None, M,
                            uniform2=1.0)
    same_map = _st.cmap.fingerprint == _st2.cmap.fingerprint
    res["vacuum_id_same_map"] = bool(same_map)
    res["vacuum_id"] = float(max(np.max(np.abs(Ra - Rb)),
                                 np.max(np.abs(Ta - Tb))))
    if name == "phaseb":
        cm, _w = CM._circle_map_3x3(P, 0.36)
        e = np.ones((3, 3), complex)
        e[1, 1] = 4.0
        st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=2, cmap=cm)
        st.add_layer(T1, eps_cell=e)
        st.add_layer(T2, eps=1.0)
        st.set_source(WL, theta=0.15, phi=0.3)
        o, Rc, Tc, J = st.solve()
        res["phaseb_fingerprint_equal"] = cm.fingerprint == \
            _st.cmap.fingerprint
        res["phaseb"] = float(max(np.max(np.abs(Ra - np.asarray(Rc))),
                                  np.max(np.abs(Ta - np.asarray(Tc)))))
    res["wall_s"] = time.time() - t0
    print(res, flush=True)
    dump(f"v6_{name}_M{M}_{BUILD}.json", res)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
