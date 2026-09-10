"""VERIFY -- open item E: the ONSET on the CONICAL and SLANT cascades.

The fix wires the guard into all nine ``PMMStack`` call sites but mapped the
onset only on the classical vertical cascade, and says so (open item E).  This
maps it on the other two, each against ITS OWN exact ``delta -> 0`` limit, so
the conjunction's transfer is measured rather than argued.

    python validation/probe_verify_sliver/v10_paths.py [out.json]
"""
import json
import os
import sys
import warnings

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm import stack as ps

HERE = os.path.dirname(os.path.abspath(__file__))
P, WL, TH = 1.2e-6, 0.85e-6, 0.15
EH, EP = 2.25, 9.0
A0, B0 = 0.27865, 0.62505
DZ = 0.32e-6 / 4
NO = P * 1e-12


def run(d, deg, mode, guard=False):
    st = PMMStack(P, n_superstrate=1.0, n_substrate=1.0, degree=deg,
                  min_feature=NO)
    if mode == "slant":
        st.add_layer(DZ, segments=[(A0, EH), (B0 - A0, EP), (1 - B0, EH)],
                     slant_angle=0.10)
        st.add_layer(DZ, segments=[(A0 - d, EH), (B0 + d - (A0 - d), EP),
                                   (1 - (B0 + d), EH)])
        st.set_source(WL, theta=TH)
    else:
        for (a, b) in ((A0, B0), (A0 - d, B0 + d)):
            st.add_layer(DZ, segments=[(a, EH), (b - a, EP), (1 - b, EH)])
        st.set_source(WL, theta=TH, phi=(0.5 if mode == "conical" else 0.0))
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = guard
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, _J = st.solve()
        o = np.asarray(o).ravel()
        i = np.argsort(o)
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        return o[i], np.asarray(R)[1][i], np.asarray(T)[1][i], float(np.max(tot))
    except ValueError:
        return None
    finally:
        ps.PMM_SLIVER_GUARD = was


def gap(a, b):
    c = np.intersect1d(a[0], b[0])
    ia, ib = np.searchsorted(a[0], c), np.searchsorted(b[0], c)
    return float(max(np.abs(a[1][ia] - b[1][ib]).max(),
                     np.abs(a[2][ia] - b[2][ib]).max()))


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v10_paths.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    assert "lum_vsliver" in lib.replace("\\", "/"), lib
    out = []
    for mode in ("classical", "conical", "slant"):
        for deg in (10, 14):
            ref = run(0.0, deg, mode)
            assert ref is not None and abs(ref[3] - 1.0) < 1e-9, (mode, deg)
            rows, misses = [], []
            for d in np.geomspace(3e-3, 1e-6, 40):
                d = float(d)
                r = run(d, deg, mode)
                if r is None:
                    continue
                e = gap(r, ref)
                refused = run(d, deg, mode, guard=True) is None
                kind = ("wrong" if e > 100.0 * d else
                        "right" if e <= 10.0 * d else "grey")
                rows.append(dict(delta=d, err=e, err_over_delta=e / d,
                                 RplusT=r[3], kind=kind, refused=refused))
                if kind == "wrong" and not refused:
                    misses.append(rows[-1])
                if kind == "right" and refused:
                    misses.append(dict(rows[-1], note="FALSE POSITIVE"))
            wrong = [r for r in rows if r["kind"] == "wrong"]
            right = [r for r in rows if r["kind"] == "right"]
            out.append(dict(
                mode=mode, degree=deg, n_rows=len(rows), n_wrong=len(wrong),
                n_right=len(right),
                n_refused=sum(r["refused"] for r in rows),
                refused_wrong=sum(r["refused"] for r in wrong),
                refused_right=sum(r["refused"] for r in right),
                misses=misses,
                worst_wrong_err=max((r["err"] for r in wrong), default=0.0),
                worst_wrong_RT=max((r["RplusT"] for r in wrong), default=0.0),
                max_absRT1_right=max((abs(r["RplusT"] - 1.0) for r in right),
                                     default=0.0),
                min_RT1_wrong=min((r["RplusT"] - 1.0 for r in wrong),
                                  default=0.0)))
            o = out[-1]
            print(f"  {mode:10s} deg {deg}: rows {o['n_rows']}  right "
                  f"{o['n_right']} wrong {o['n_wrong']}; refused "
                  f"{o['refused_wrong']}/{o['n_wrong']} wrong, "
                  f"{o['refused_right']}/{o['n_right']} right; "
                  f"worst wrong err {o['worst_wrong_err']:.3e} "
                  f"(R+T {o['worst_wrong_RT']:.5g}); "
                  f"max |R+T-1| right {o['max_absRT1_right']:.3e}; "
                  f"min R+T-1 wrong {o['min_RT1_wrong']:.3e}; "
                  f"misses {len(o['misses'])}")
            for mm in o["misses"]:
                print(f"      MISS delta {mm['delta']:.4e} err {mm['err']:.4e} "
                      f"({mm['err_over_delta']:.0f}x) R+T-1 "
                      f"{mm['RplusT'] - 1.0:+.3e} {mm.get('note', '')}")
    with open(out_path, "w") as f:
        json.dump(dict(meta=dict(lumenairy=lib, python=sys.version.split()[0],
                                 numpy=np.__version__), rows=out), f,
                  indent=1, default=str)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
