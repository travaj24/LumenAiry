"""V2 -- COMPOSITION: the walks ADD, and the split is blind to the sum.

Two SHEARED layers at DIFFERENT shears, thicknesses, duties and permittivities,
against a LAB-REFERENCED z-staircase of the same solid.  The oracle places the
LOWER layer at the ACCUMULATED walk, because the frame CONTINUES: the cascade
matches successive layers' frame coefficients directly, with no re-referencing.

Arms scored against that oracle, all built from the SHIPPED answer so the probe
reads the same way on either tree (``P(W) = exp(+i k0 alpha_m W)``):

    full sum          the shipped answer   (pre-fix: multiplied by P(W) here)
    only layer 1      / P(W) * P(W1)
    only layer 2      / P(W) * P(W2)
    no sum            / P(W)
    conjugate sum     / P(W) * conj(P(W))

Plus the LAYER-SPLIT identity: in the frame the structure is z-invariant, so
one sheared layer of ``d`` at shear ``s`` is the same solid as TWO sheared
halves of ``d/2`` at shear ``s/2`` (same ``tan(phi)``) whose own-frame ridge
centre is the ORIGINAL top-face centre.  Both must read the same -- and both
must read the same WRONG number under a wrong sum, which is why the split alone
can never detect a mis-summed walk.
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
NSUP, NSUB = 1.0, 1.6
TH = math.radians(25.0)
DEG, NORD = 8, 5

D1, S1, DUTY1, ER1, EG1 = 0.24e-6, 0.25, 0.45, 4.20, 1.45
D2, S2, DUTY2, ER2, EG2 = 0.16e-6, 0.10, 0.35, 2.90, 1.60
W1, W2 = S1 * P, S2 * P
WTOT = W1 + W2


def _stack(**kw):
    from lumenairy.elements.pmm.stack import PMMStack
    return PMMStack(P, n_superstrate=NSUP, n_substrate=NSUB, degree=DEG,
                    n_orders=NORD, factorization="convection", **kw)


def _solved(st):
    st.set_source(WL, theta=TH)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = st.solve()
    return st, out


def two_sheared():
    st = _stack()
    st.add_sheared_grating(D1, eps_ridge=ER1, eps_groove=EG1, duty=DUTY1,
                           shear=S1, centre=0.5)
    st.add_sheared_grating(D2, eps_ridge=ER2, eps_groove=EG2, duty=DUTY2,
                           shear=S2, centre=0.5)
    return _solved(st)


def staircase(K):
    """The same solid, vertical rungs only -- LAB-referenced by construction.
    The LOWER layer's ridge centre carries the ACCUMULATED walk ``S1`` (in
    period fractions), which is the statement that the frame continues."""
    st = _stack()
    st.add_tapered_ridges(D1, ridges=[(0.5 * P, DUTY1 * P, DUTY1 * P, ER1)],
                          eps_groove=EG1, n_slices=K, shear=S1)
    st.add_tapered_ridges(D2, ridges=[((0.5 + S1) * P, DUTY2 * P, DUTY2 * P, ER2)],
                          eps_groove=EG2, n_slices=K, shear=S2)
    return _solved(st)


def split_halves():
    """One sheared layer of ``D1`` at ``S1`` expressed as TWO halves."""
    st = _stack()
    for _ in range(2):
        st.add_sheared_grating(D1 / 2, eps_ridge=ER1, eps_groove=EG1,
                               duty=DUTY1, shear=S1 / 2, centre=0.5 - S1 / 4)
    return _solved(st)


def one_layer():
    st = _stack()
    st.add_sheared_grating(D1, eps_ridge=ER1, eps_groove=EG1, duty=DUTY1,
                           shear=S1, centre=0.5)
    return _solved(st)


def stair_one(K):
    st = _stack()
    st.add_tapered_ridges(D1, ridges=[(0.5 * P, DUTY1 * P, DUTY1 * P, ER1)],
                          eps_groove=EG1, n_slices=K, shear=S1)
    return _solved(st)


def amps(st, port="transmission"):
    a = st.per_order_amplitudes(port)
    return (np.asarray(a["orders"]),
            np.stack([np.asarray(a["Ex"]), np.asarray(a["Ey"])], axis=0),
            np.asarray(np.real(a["kx"])))


def align(oa, A, ob, B):
    bmap = {int(o): j for j, o in enumerate(np.asarray(ob))}
    num = den = 0.0
    for i, o in enumerate(np.asarray(oa)):
        j = bmap.get(int(o))
        if j is None:
            continue
        num += float(np.sum(np.abs(A[..., i] - B[..., j]) ** 2))
        den += float(np.sum(np.abs(B[..., j]) ** 2))
    return math.sqrt(num) / math.sqrt(den) if den > 0 else math.sqrt(num)


def main():
    t0 = time.time()
    res = dict(fixture=dict(period_um=P * 1e6, wl_um=WL * 1e6,
                            layer1=(D1 * 1e6, S1, DUTY1, ER1, EG1),
                            layer2=(D2 * 1e6, S2, DUTY2, ER2, EG2),
                            W1_um=W1 * 1e6, W2_um=W2 * 1e6,
                            W_total_um=WTOT * 1e6, theta_deg=25.0))
    st, _o = two_sheared()
    J = L.jt(st)
    o_s, A_s, kx = amps(st)
    Pw = L.P_of(kx, WTOT, WL)
    P1 = L.P_of(kx, W1, WL)
    P2 = L.P_of(kx, W2, WL)
    i0 = int(np.where(np.asarray(o_s) == 0)[0][0])
    ARMS = {"full_sum": A_s,
            "only_layer1": A_s / Pw * P1,
            "only_layer2": A_s / Pw * P2,
            "no_sum": A_s / Pw,
            "conjugate_sum": A_s / Pw * np.conj(Pw)}
    rows, prev = {}, None
    for K in (4, 6, 8):
        stc, _oc = staircase(K)
        Jc = L.jt(stc)
        o_c, A_c, _k = amps(stc)
        row = {("amps_" + k): align(o_s, V, o_c, A_c) for k, V in ARMS.items()}
        row["J_full_sum"] = L.resid(J, Jc)
        row["J_no_sum"] = L.resid(J / complex(Pw[i0]), Jc)
        row["J_only_layer1"] = L.resid(J / complex(Pw[i0]) * complex(P1[i0]),
                                       Jc)
        row["J_conjugate_sum"] = L.resid(
            J / complex(Pw[i0]) * np.conj(complex(Pw[i0])), Jc)
        row["oracle_step_amps"] = (None if prev is None
                                   else align(prev[0], prev[1], o_c, A_c))
        row["oracle_step_J"] = (None if prev is None
                                else L.resid(prev[2], Jc))
        rows["K%d" % K] = row
        prev = (o_c, A_c, Jc)
    res["two_sheared_vs_staircase"] = rows

    # ---- the LAYER-SPLIT identity ------------------------------------------
    s1, _ = one_layer()
    s2, _ = split_halves()
    J1, J2 = L.jt(s1), L.jt(s2)
    o1, A1, kx1 = amps(s1)
    o2, A2, _ = amps(s2)
    Pw1 = L.P_of(kx1, W1, WL)
    split = dict(sha_J_equal=(L.sha(J1) == L.sha(J2)),
                 sha_perT_equal=(L.sha(A1) == L.sha(A2)),
                 dJ=L.resid(J1, J2), damps=align(o1, A1, o2, A2))
    stc, _ = stair_one(8)
    Jc1 = L.jt(stc)
    o_c1, A_c1, _ = amps(stc)
    split["one_layer_vs_staircase"] = align(o1, A1, o_c1, A_c1)
    split["two_halves_vs_staircase"] = align(o2, A2, o_c1, A_c1)
    split["one_layer_HALF_sum"] = align(o1, A1 / Pw1 * L.P_of(kx1, W1 / 2, WL),
                                        o_c1, A_c1)
    split["two_halves_HALF_sum"] = align(o2, A2 / Pw1 * L.P_of(kx1, W1 / 2, WL),
                                         o_c1, A_c1)
    split["one_layer_NO_sum"] = align(o1, A1 / Pw1, o_c1, A_c1)
    split["two_halves_NO_sum"] = align(o2, A2 / Pw1, o_c1, A_c1)
    split["J_one_vs_stair"] = L.resid(J1, Jc1)
    split["J_two_vs_stair"] = L.resid(J2, Jc1)
    res["layer_split"] = split

    res["seconds"] = round(time.time() - t0, 1)
    for k, v in rows.items():
        print("==", k)
        for kk, vv in sorted(v.items()):
            print("   %-22s %s" % (kk, ("%.5e" % vv) if isinstance(vv, float)
                                   else vv))
    print("== layer split")
    for kk, vv in sorted(split.items()):
        print("   %-24s %s" % (kk, ("%.5e" % vv) if isinstance(vv, float)
                               else vv))
    L.dump("v2_compose", res, suffix=(sys.argv[1] if len(sys.argv) > 1 else ""))


if __name__ == "__main__":
    main()
