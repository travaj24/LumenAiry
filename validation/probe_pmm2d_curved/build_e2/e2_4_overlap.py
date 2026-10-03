"""E2-4: the circle over a sinusoidal wall whose curved regions OVERLAP (the
wall runs through the disk; the Phase C stack-wide merge refuses), through
the curved mortar.  Modes (arg 1), rung M (arg 2):

  closure  -- lossless |R + T - 1| per input polarization
  absorb   -- the disk lossy (eps 4 + 0.3i): sum of layer_absorption against
              1 - R - T, and the fail-before with -R (not the plain Gram) as
              the flux form of every layer
  vacuum   -- a VACUUM-painted sinusoid layer on its own map ON TOP of the
              circle (air above: a physical no-op), vs the circle alone and
              vs the same spacer on the circle map (shared path)
  merged   -- the NON-overlapping sinusoid (x0 = 0.12, A = 0.05): the stack
              solved both ways -- the merged map (fast path) and per-layer
              maps through the curved mortar (forced)
"""
import sys
import time
import types

import numpy as np
from _common import D1, D2, EPS_P, EPS_W, R_CIRC, TS, dump, shapes_stack, solve

from lumenairy.elements.pmm.shapes2d import Circle, SinusoidalWall

mode, M = sys.argv[1], int(sys.argv[2])
out = {"mode": mode, "M": M}


def circ(eps=EPS_P):
    return [Circle(0.6, 0.6, R_CIRC, eps)]


def sinw(x0=0.6, A=0.12, eps=EPS_W):
    return [SinusoidalWall("x", x0, A, eps=eps)]


def minus_r_flux(st, M):
    """Replace the stack flux form by -R (= C[chi_t]C under a map) per layer:
    the Phase A trap, as a fail-before arm."""
    Rm = []
    for L in st._layers:
        own = L["own"]
        cm = own["cmap"]
        if cm is None:
            raise RuntimeError("expects mapped shape layers")
        s = TS.Granet2DTransverseE(st.period_x, st.period_y, cm.u_walls,
                                   cm.v_walls, M, own["cell"],
                                   k0=2 * np.pi, cmap=cm)
        Rm.append(-s.Rmat)

    def flux_minusR(self, i, z_frac, amps, _Rm=Rm):
        dd = self._internal
        Wf, Vf, lam_f, Wb, Vb, lam_b, t = dd["modes"][i]
        c_fwd, c_bwd = amps[i]
        k0 = dd["k0"]
        qq = dd["qq_of"][i]
        Pz = np.exp(-lam_f * k0 * (z_frac * t))[:, None]
        Qz = np.exp(lam_b * k0 * ((1.0 - z_frac) * t))[:, None]
        E = Wf @ (Pz * c_fwd) + Wb @ (Qz * c_bwd)
        H = Vf @ (Pz * c_fwd) + Vb @ (Qz * c_bwd)
        G = _Rm[i]
        G1E = G[:qq, :qq] @ E[:qq]
        G2E = G[qq:, qq:] @ E[qq:]
        val = (np.sum(np.conj(H[qq:]) * G1E, axis=0)
               - np.sum(np.conj(H[:qq]) * G2E, axis=0))
        return np.real(val)
    st._flux_at = types.MethodType(flux_minusR, st)


t0 = time.perf_counter()
if mode == "closure":
    st = shapes_stack([(D1, circ(), 1.0), (D2, sinw(), 1.0)], M)
    assert not st._perlayer_fast_ok()
    o, R, T, J = solve(st)
    out.update(closure=np.abs(R.sum(1) + T.sum(1) - 1.0), R=R, T=T,
               orders=o)
elif mode == "absorb":
    st = shapes_stack([(D1, circ(4.0 + 0.3j), 1.0), (D2, sinw(), 1.0)], M)
    o, R, T, J = solve(st, retain=True)
    A = st.layer_absorption()
    budget = 1.0 - R.sum(1) - T.sum(1)
    out.update(absorption=A, budget=budget,
               mismatch=np.abs(A.sum(0) - budget),
               lossless_layer2=np.abs(A[1]))
    minus_r_flux(st, M)
    Af = st.layer_absorption()
    out.update(failbefore_minusR_mismatch=np.abs(Af.sum(0) - budget),
               failbefore_minusR_layer2=np.abs(Af[1]))
elif mode == "vacuum":
    # a VACUUM-painted sinusoid layer (homogeneous: it RIDES the circle
    # layer's map) ON TOP of the circle (air above: a physical no-op), vs
    # the same spacer on the circle map through the shared path (round-off)
    # and vs the circle alone (the Phase C spacer effect, gate C9); the
    # no-ride arm keeps the vacuum layer on its OWN sinusoid map (a genuine
    # curved mortar between two maps under the rim of the pillar)
    st = shapes_stack([(D2, sinw(eps=1.0), 1.0), (D1, circ(), 1.0)], M)
    o, R, T, J = solve(st)
    st2 = shapes_stack([(D2, None, 1.0), (D1, circ(), 1.0)], M,
                       per_layer=False)
    o2, R2, T2, J2 = solve(st2)
    st3 = shapes_stack([(D1, circ(), 1.0)], M, per_layer=False)
    o3, R3, T3, J3 = solve(st3)
    st4 = shapes_stack([(D2, sinw(eps=1.0), 1.0), (D1, circ(), 1.0)], M)
    st4._e2_no_ride = True
    o4, R4, T4, J4 = solve(st4)
    out.update(perlayer_vs_shared_spacer=float(max(np.abs(R - R2).max(),
                                                   np.abs(T - T2).max())),
               perlayer_bytes_equal_shared=bool(np.array_equal(R, R2)
                                                and np.array_equal(T, T2)),
               shared_spacer_vs_alone=float(max(np.abs(R2 - R3).max(),
                                                np.abs(T2 - T3).max())),
               noride_vs_alone=float(max(np.abs(R4 - R3).max(),
                                         np.abs(T4 - T3).max())),
               noride_closure=np.abs(R4.sum(1) + T4.sum(1) - 1.0),
               closure=np.abs(R.sum(1) + T.sum(1) - 1.0), R=R, T=T,
               R_alone=R3, T_alone=T3)
elif mode == "merged":
    lay = [(D1, circ(), 1.0), (D2, sinw(0.12, 0.05), 1.0)]
    st = shapes_stack(lay, M)
    assert st._perlayer_fast_ok()
    o, R, T, J = solve(st)
    stf = shapes_stack(lay, M)
    stf._e2_per_layer_maps = True
    o2, R2, T2, J2 = solve(stf)
    sts = shapes_stack(lay, M, per_layer=False)
    o3, R3, T3, J3 = solve(sts)
    out.update(fastpath_bytes_equal_shared=bool(
        np.array_equal(R, R3) and np.array_equal(T, T3)
        and np.array_equal(J, J3)),
        perlayer_maps_vs_merged=float(max(np.abs(R2 - R).max(),
                                          np.abs(T2 - T).max())),
        closure_merged=np.abs(R.sum(1) + T.sum(1) - 1.0),
        closure_perlayer=np.abs(R2.sum(1) + T2.sum(1) - 1.0), R=R, T=T,
        R_pl=R2, T_pl=T2)
out["wall"] = time.perf_counter() - t0
print({k: v for k, v in out.items() if k not in ("R", "T", "R_shared",
                                                  "T_shared", "R_pl", "T_pl",
                                                  "R_alone", "T_alone",
                                                  "orders")})
dump(f"e2_4_{mode}_M{M}.json", out)
