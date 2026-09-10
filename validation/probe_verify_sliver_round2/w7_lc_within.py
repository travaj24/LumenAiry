"""W7 -- the LIQUID-CRYSTAL class (V-5) and the WITHIN-LAYER arm (V-6).

  A  the three anisotropic tensor classes carrying the sliver -- an IN-PLANE
     45-degree director, an OUT-OF-PLANE 30-degree director, a gyrotropic
     layer -- each against its own SLIVER-FREE control, plus the
     non-Hermitian negative control that must keep the behaviour it had.
     Records whether each tensor is EXACTLY Hermitian on this build.
  B  the POL-0 evidence: the per-polarization error of the out-of-plane
     refusal, and the per-polarization MOVE the arbiter would read -- so the
     claim "a pol-1-only statistic would have called that solve right" is
     re-measured rather than read.
  C  the within-layer liner ladder 1e-2 .. 1e-7: err, R+T, the q-excess, and
     whether the library WARNS -- two-sided (silent on benign).
  D  can a WITHIN-LAYER liner be REFUSED by mistake through the CROSS-LAYER
     path?  Three geometries: the liner alone; the liner plus an ordinary
     non-conforming wall from another layer; two layers whose liners are
     offset.

    python w7_lc_within.py [out.json]
"""
import json
import os
import sys
import time
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from w_fixtures import (  # noqa: E402
    gyrotropic,
    non_hermitian,
    prescribed,
    shared_move,
    snapped,
    unguarded,
    uniaxial,
)

import lumenairy  # noqa: E402
from lumenairy.elements.pmm import PMMStack  # noqa: E402
from lumenairy.elements.pmm import stack as ps  # noqa: E402

P, WL, TH = 1.2e-6, 0.85e-6, 0.15
A0, B0 = 0.27865, 0.62505
DZ = 0.32e-6 / 4
NO_SNAP = P * 1e-12

TENSORS = {
    "in_plane_45": uniaxial(1.51, 1.72, 0.0, azim=np.pi / 4),
    "out_of_plane_30": uniaxial(1.51, 1.72, np.pi / 6),
    "gyrotropic": gyrotropic(4.0, 0.35),
    "lossy_director": uniaxial(1.51 + 0.02j, 1.72 + 0.03j, 0.4, azim=0.9),
    "non_hermitian": non_hermitian(4.0, 0.2),
}


def lc_stack(d, deg, tensor, nl=2):
    st = PMMStack(P, n_substrate=1.0, degree=deg, far_field_orders=31,
                  min_feature=NO_SNAP)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        st.add_layer(DZ, segments=[(A0 - dd, 2.25),
                                   (B0 + dd - (A0 - dd), tensor),
                                   (1.0 - (B0 + dd), 2.25)])
    st.set_source(WL, theta=TH)
    return st


def guarded(st):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            st.solve()
        except ValueError as exc:
            return ("REFUSED" if "NEAR-COINCIDENT-WALL SLIVER" in str(exc)
                    else f"raised: {str(exc)[:80]}"), \
                [str(w.message) for w in rec]
    return "returned", [str(w.message) for w in rec]


def arm_A():
    out = {}
    for name, M in TENSORS.items():
        M = np.asarray(M, dtype=complex)
        rec = dict(
            exactly_hermitian=bool(np.array_equal(M, np.conjugate(M).T)),
            segment_passive=bool(ps._segment_passive(M)),
            antiherm_min_eig=float(np.min(np.linalg.eigvalsh(
                (M - np.conjugate(M).T) / 2.0j))),
            rows={})
        for d in (3e-3, 1e-4, 3e-5):
            st = lc_stack(d, 14, M)
            cur = unguarded(st)
            ref = unguarded(lc_stack(0.0, 14, M))
            e = shared_move(cur, ref, pol=1)
            v, warns = guarded(lc_stack(d, 14, M))
            row = dict(verdict=v, worst=cur["worst"], err=e,
                       err_over_d=e / d,
                       passive_stack=ps._stack_provably_passive(st),
                       screen=ps._sliver_screen(st) is not None,
                       n_warn=len(warns))
            pre = prescribed(st)
            if pre is not None and cur["worst"] - 1.0 > 1e-3:
                snp = snapped(st, pre["mf"])
                row["su_snap"] = max(snp["worst"] - 1.0, 0.0)
                row["move_ratio"] = shared_move(cur, snp) / pre["w_wide"]
                row["move_ratio_p1"] = (shared_move(cur, snp, pol=1)
                                        / pre["w_wide"])
                row["move_ratio_p0"] = (shared_move(cur, snp, pol=0)
                                        / pre["w_wide"])
            rec["rows"][f"{d:g}"] = row
        # the SLIVER-FREE control at the same tensor: identical walls
        stc = lc_stack(0.0, 14, M)
        curc = unguarded(stc)
        vc, wc = guarded(lc_stack(0.0, 14, M))
        rec["control_sliver_free"] = dict(verdict=vc, worst=curc["worst"],
                                          n_warn=len(wc),
                                          screen=ps._sliver_screen(
                                              stc) is not None)
        out[name] = rec
    return out


def arm_B():
    """The pol-0 evidence, per polarization, on the out-of-plane director."""
    M = TENSORS["out_of_plane_30"]
    out = {}
    for d in (1e-4, 3e-5, 1e-5):
        ref = unguarded(lc_stack(0.0, 14, M))
        st = lc_stack(d, 14, M)
        cur = unguarded(st)
        rec = dict(err_p0=shared_move(cur, ref, pol=0),
                   err_p1=shared_move(cur, ref, pol=1),
                   err_both=shared_move(cur, ref), worst=cur["worst"])
        rec["err_p0_over_d"] = rec["err_p0"] / d
        rec["err_p1_over_d"] = rec["err_p1"] / d
        pre = prescribed(st)
        if pre is not None:
            snp = snapped(st, pre["mf"])
            rec["move_p0"] = shared_move(cur, snp, pol=0) / pre["w_wide"]
            rec["move_p1"] = shared_move(cur, snp, pol=1) / pre["w_wide"]
            rec["move_both"] = shared_move(cur, snp) / pre["w_wide"]
            rec["su_snap"] = max(snp["worst"] - 1.0, 0.0)
            rec["would_refuse_both_pol"] = bool(
                rec["su_snap"] <= 1e-5 and rec["move_both"] > 100.0)
            rec["would_refuse_pol1_only"] = bool(
                rec["su_snap"] <= 1e-5 and rec["move_p1"] > 100.0)
        rec["verdict"] = guarded(lc_stack(d, 14, M))[0]
        out[f"{d:g}"] = rec
    return out


def liner_stack(w, deg, nl=2, extra_wall=None):
    st = PMMStack(P, n_substrate=1.0, degree=deg, far_field_orders=21,
                  min_feature=NO_SNAP)
    for k in range(nl):
        segs = [(0.30, 2.25), (w, 9.0), (0.70 - w, 2.25)]
        if extra_wall is not None and k == nl - 1:
            x = extra_wall
            segs = [(x, 2.25), (0.30 - x, 2.25), (w, 9.0), (0.70 - w, 2.25)]
        st.add_layer(0.08e-6, segments=segs)
    st.set_source(WL, theta=TH)
    return st


def arm_C():
    out = {}
    for w in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
        row = {}
        for deg in (8, 12, 14, 16):
            st = liner_stack(w, deg)
            ref = PMMStack(P, n_substrate=1.0, degree=deg,
                           far_field_orders=21, min_feature=NO_SNAP)
            for _k in range(2):
                ref.add_layer(0.08e-6, segments=[(0.30, 2.25), (0.70, 2.25)])
            ref.set_source(WL, theta=TH)
            cur, r0 = unguarded(st), unguarded(ref)
            e = shared_move(cur, r0, pol=1)
            v, warns = guarded(liner_stack(w, deg))
            row[deg] = dict(err=e, err_over_w=e / w, worst=cur["worst"],
                            verdict=v, n_warn=len(warns),
                            within_layer_warn=any("WITHIN-LAYER" in x
                                                  for x in warns),
                            hazard=(ps._within_layer_hazard(st, None)
                                    is not None),
                            q_excess=(ps._sliver_q_predictor(
                                w, P, deg, WL) / 3.0))
        out[f"{w:g}"] = row
    return out


def arm_D():
    """Can an OWNED liner be REFUSED through the cross-layer path?"""
    out = {}
    for w in (1e-3, 1e-5, 1e-6, 1e-7):
        # (1) the liner alone, in EVERY layer -- owned, must never be refused
        st1 = liner_stack(w, 14)
        v1, w1 = guarded(liner_stack(w, 14))
        hit1 = ps._cross_layer_sliver([L[1] for L in st1._layers], 1e-12)
        # (2) the liner PLUS one ordinary non-conforming wall in the last
        #     layer, placed a real 1 % of a period away from the liner
        st2 = liner_stack(w, 14, extra_wall=0.29)
        v2, w2 = guarded(liner_stack(w, 14, extra_wall=0.29))
        hit2 = ps._cross_layer_sliver([L[1] for L in st2._layers], 1e-12)
        # (3) two layers whose LINERS are offset by w/2 -- the union then
        #     manufactures cells out of what each layer owns
        st3 = PMMStack(P, n_substrate=1.0, degree=14, far_field_orders=21,
                       min_feature=NO_SNAP)
        for k in (0, 1):
            x = 0.30 + 0.5 * w * k
            st3.add_layer(0.08e-6, segments=[(x, 2.25), (w, 9.0),
                                             (1.0 - x - w, 2.25)])
        st3.set_source(WL, theta=TH)
        cur3 = unguarded(st3)
        v3, w3 = guarded(st3)
        hit3 = ps._cross_layer_sliver([L[1] for L in st3._layers], 1e-12)
        out[f"{w:g}"] = dict(
            liner_only=dict(verdict=v1, cross_layer_hit=hit1 is not None,
                            within_layer_warn=any("WITHIN-LAYER" in x
                                                  for x in w1),
                            worst=unguarded(st1)["worst"]),
            liner_plus_wall=dict(verdict=v2,
                                 cross_layer_hit=hit2 is not None,
                                 ratio=(hit2[4] / hit2[0]) if hit2 else None,
                                 within_layer_warn=any("WITHIN-LAYER" in x
                                                       for x in w2),
                                 worst=unguarded(st2)["worst"]),
            offset_liners=dict(verdict=v3, cross_layer_hit=hit3 is not None,
                               ratio=(hit3[4] / hit3[0]) if hit3 else None,
                               within_layer_warn=any("WITHIN-LAYER" in x
                                                     for x in w3),
                               worst=cur3["worst"]))
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w7_lc_within.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    out = dict(lumenairy=lib, python=sys.version.split()[0],
               numpy=np.__version__)
    for name, fn in (("A_tensor_classes", arm_A), ("B_pol0", arm_B),
                     ("C_liner_ladder", arm_C), ("D_owned_vs_refused",
                                                 arm_D)):
        t = time.time()
        out[name] = fn()
        print(f"== {name} ({time.time() - t:.1f} s)")
        print(json.dumps(out[name], indent=1, default=str))
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
