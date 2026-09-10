"""W8 -- the round-2 report's OWN anisotropic numbers, re-measured.

Arm A re-measures the S3.5 table on the fix's OWN director convention
(``diag(ne^2, no^2, no^2)`` rotated about z or y, no = 1.50, ne = 1.72) so
the stated ``move`` = 38.9 / 290.4 / 22.2 / 316.5 and the pol-0 evidence
(pol 1 = 0.01x the shift, pol 0 = 316x) are checked as numbers, not read.

Arm B asks the question the round-2 report does not: what happens when a
CROSS-LAYER sliver and a broken WITHIN-LAYER liner are present on the SAME
stack?  The arbiter attributes to ONE cause, and the liner's super-unity
cannot be snapped away -- so the prediction is verdict ``truncation`` and a
RETURN, where round 1 refused.

    python w8_lc_exact.py [out.json]
"""
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import lumenairy                                              # noqa: E402
from lumenairy.elements.pmm import PMMStack                   # noqa: E402
from lumenairy.elements.pmm import stack as ps                # noqa: E402
from w_fixtures import prescribed, shared_move, snapped       # noqa: E402
from w_fixtures import unguarded                              # noqa: E402

P, WL, TH = 1.2e-6, 0.85e-6, 0.15
A0, B0, EH = 0.27865, 0.62505, 2.25
DZ = 0.32e-6 / 4
NO_SNAP = P * 1e-12


def fix_uniaxial(theta, axis="xy", no=1.50, ne=1.72, kappa=0.0):
    """The round-2 TEST FILE's own director, copied so the numbers compare."""
    d = np.diag([(ne + 1j * kappa) ** 2, (no + 1j * kappa) ** 2,
                 (no + 1j * kappa) ** 2]).astype(complex)
    c, s = np.cos(theta), np.sin(theta)
    R = (np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) if axis == "xy"
         else np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]))
    return R @ d @ R.T


def stack(d, deg, eps, nl=2, liner=None):
    st = PMMStack(P, n_substrate=1.0, degree=deg, far_field_orders=31,
                  min_feature=NO_SNAP)
    for k in range(nl):
        dd = d * k / max(nl - 1, 1)
        a, b = A0 - dd, B0 + dd
        if liner is None:
            segs = [(a, EH), (b - a, eps), (1.0 - b, EH)]
        else:
            segs = [(a, EH), (b - a, eps), (liner, 9.0),
                    (1.0 - b - liner, EH)]
        st.add_layer(DZ, segments=segs)
    st.set_source(WL, theta=TH)
    return st


def guarded(st):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            st.solve()
        except ValueError as exc:
            return ("REFUSED" if "NEAR-COINCIDENT-WALL SLIVER" in str(exc)
                    else "raised"), [str(w.message) for w in rec]
    return "returned", [str(w.message) for w in rec]


def arm_A():
    classes = {"lc_in_plane_45": fix_uniaxial(np.pi / 4.0, "xy"),
               "lc_out_of_plane_30": fix_uniaxial(np.pi / 6.0, "xz"),
               "gyrotropic": None, "non_hermitian": None,
               "lossy_in_plane": fix_uniaxial(np.pi / 4.0, "xy", kappa=0.2)}
    G = np.eye(3, dtype=complex) * 4.0
    G[0, 1], G[1, 0] = 0.35j, -0.35j
    classes["gyrotropic"] = G
    N = np.eye(3, dtype=complex) * 4.0
    N[0, 1] = 0.2
    classes["non_hermitian"] = N
    out = {}
    for name, M in classes.items():
        M = np.asarray(M, dtype=complex)
        rec = dict(exactly_hermitian=bool(np.array_equal(M,
                                                         np.conjugate(M).T)),
                   antiherm_min_eig_over_scale=float(
                       np.min(np.linalg.eigvalsh((M - np.conjugate(M).T)
                                                 / 2.0j)))
                   / max(float(np.max(np.abs(M))), 1.0),
                   segment_passive=bool(ps._segment_passive(M)), rows={})
        ref = unguarded(stack(0.0, 14, M))
        for d in (3e-3, 1e-4, 3e-5):
            st = stack(d, 14, M)
            cur = unguarded(st)
            row = dict(worst=cur["worst"],
                       err_p1_over_d=shared_move(cur, ref, pol=1) / d,
                       err_p0_over_d=shared_move(cur, ref, pol=0) / d,
                       verdict=guarded(stack(d, 14, M))[0],
                       passive=ps._stack_provably_passive(st))
            pre = prescribed(st)
            if pre is not None and cur["worst"] - 1.0 > 1e-3:
                snp = snapped(st, pre["mf"])
                row["su_snap"] = max(snp["worst"] - 1.0, 0.0)
                row["move"] = shared_move(cur, snp) / pre["w_wide"]
                row["move_p1"] = shared_move(cur, snp, pol=1) / pre["w_wide"]
                row["move_p0"] = shared_move(cur, snp, pol=0) / pre["w_wide"]
            rec["rows"][f"{d:g}"] = row
        out[name] = rec
    return out


def arm_B():
    """A cross-layer sliver AND a broken owned liner on the same stack."""
    out = {}
    for liner in (None, 1e-6, 1e-7):
        for d in (1e-4, 3e-5):
            for deg in (12, 14):
                st = stack(d, deg, 9.0, liner=liner)
                cur = unguarded(st)
                pre = prescribed(st)
                rec = dict(worst=cur["worst"], screen=pre is not None,
                           w_wide=pre["w_wide"] if pre else None,
                           own=pre["own"] if pre else None)
                if pre is not None and cur["worst"] - 1.0 > 1e-3:
                    snp = snapped(st, pre["mf"])
                    rec["su_snap"] = max(snp["worst"] - 1.0, 0.0)
                    rec["move"] = shared_move(cur, snp) / pre["w_wide"]
                v, warns = guarded(stack(d, deg, 9.0, liner=liner))
                rec["verdict"] = v
                rec["truncation_note"] = any("is NOT what moved" in w
                                             for w in warns)
                rec["within_layer_warn"] = any("WITHIN-LAYER" in w
                                               for w in warns)
                rec["round1_would_refuse"] = bool(
                    pre is not None
                    and cur["worst"] - 1.0 > ps._STACK_SUPERUNITY_BAR)
                out[f"liner={liner}_d={d:g}_deg={deg}"] = rec
    return out


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "w8_lc_exact.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib)
    out = dict(lumenairy=lib, python=sys.version.split()[0],
               numpy=np.__version__, A_fix_directors=arm_A(),
               B_cooccurring=arm_B())
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1, default=str)
    for name, rec in out["A_fix_directors"].items():
        print(f"== {name}: hermitian={rec['exactly_hermitian']} "
              f"passive={rec['segment_passive']} "
              f"lam/scale={rec['antiherm_min_eig_over_scale']:.3e}")
        for d, r in rec["rows"].items():
            print(f"   d={d:8s} {r['verdict']:9s} R+T={r['worst']:.6g} "
                  f"err/d p1={r['err_p1_over_d']:.4g} "
                  f"p0={r['err_p0_over_d']:.4g} "
                  f"move={r.get('move')} move_p1={r.get('move_p1')} "
                  f"su_snap={r.get('su_snap')}")
    print("== B: co-occurring cross-layer sliver + owned liner")
    for k, r in out["B_cooccurring"].items():
        print(f"   {k:32s} R+T={r['worst']:.6g} {r['verdict']:9s} "
              f"trunc_note={r['truncation_note']} "
              f"within={r['within_layer_warn']} "
              f"r1_would_refuse={r['round1_would_refuse']} "
              f"su_snap={r.get('su_snap')} move={r.get('move')}")
    print("->", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
