"""VERIFY task 7 -- durability of ``tests/unit/test_fix_pmmstack_sliver_walls.py``.

Re-measures, on the running build, the quantity behind EVERY numeric bar in
that file and prints the gap to the bar on both sides, so the audit table can
say what each constant's headroom actually is rather than that it passed.

    python validation/probe_verify_sliver/v7_durability.py [out.json]
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
from lumenairy.elements.pmm._core import (
    _build_sem_tensor_segments,
    _pmm_union_grid,
    _sem_modes_tensor,
)

HERE = os.path.dirname(os.path.abspath(__file__))
_P, _WL, _THETA = 1.2e-6, 0.85e-6, 0.15
_DZ = 0.32e-6 / 4
_EH, _EP = 2.25, 9.0
_A0, _B0 = 0.27865, 0.62505
_LADDER = ([(14, d) for d in (3e-3, 1e-3, 3e-4, 1e-4, 5e-5, 3e-5, 1e-5)]
           + [(12, d) for d in (3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5)])


def _solve(delta, degree=14, *, guard=True, min_feature=None, per_layer=False):
    kw = dict(layer_grids="per-layer") if per_layer else {}
    if min_feature is not None:
        kw["min_feature"] = min_feature
    st = PMMStack(_P, n_superstrate=1.0, n_substrate=1.0, degree=degree, **kw)
    for (a, b) in [(_A0, _B0), (_A0 - delta, _B0 + delta)]:
        st.add_layer(_DZ, segments=[(a, _EH), (b - a, _EP), (1.0 - b, _EH)])
    st.set_source(_WL, theta=_THETA)
    was = ps.PMM_SLIVER_GUARD
    ps.PMM_SLIVER_GUARD = bool(guard)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            o, R, T, J = st.solve()
    finally:
        ps.PMM_SLIVER_GUARD = was
    i = np.argsort(np.asarray(o).ravel())
    return np.asarray(o).ravel()[i], R[1][i], T[1][i], np.asarray(J)


def _err(a, b):
    return float(max(np.abs(a[1] - b[1]).max(), np.abs(a[2] - b[2]).max()))


def _total(res):
    return float(res[1].sum() + res[2].sum())


def _segs(walls, eps=(_EH, _EP, _EH)):
    out, prev = [], 0.0
    for w, e in zip(list(walls) + [1.0], eps + (eps[-1],)):
        out.append((w - prev, e))
        prev = w
    return [s for s in out if s[0] > 0.0]


def main():
    out_path = (sys.argv[1] if len(sys.argv) > 1
                else os.path.join(HERE, "v7_durability.json"))
    lib = os.path.abspath(lumenairy.__file__)
    print("lumenairy:", lib, lumenairy.__version__)
    m = {}

    # ---- the delta -> 0 reference's own closure (bar: < 1e-6) -------------
    ref14 = _solve(0.0, 14, guard=False)
    ref12 = _solve(0.0, 12, guard=False)
    m["ref_closure_deg14"] = abs(_total(ref14) - 1.0)
    m["ref_closure_deg12"] = abs(_total(ref12) - 1.0)

    # ---- the continuity slope (bars: 0.5 < s < 2.0) -----------------------
    slopes = [_err(_solve(d, 14, guard=False), ref14) / d
              for d in (3e-3, 1e-3, 3e-4)]
    m["slopes_deg14"] = slopes

    # ---- the fail-before magnitude (bars: e > 1000*max(slope)*1e-4, e > 0.1,
    #      total-1 > 1.0) --------------------------------------------------
    bad = _solve(1e-4, 14, guard=False)
    m["failbefore_err"] = _err(bad, ref14)
    m["failbefore_bar_from_slope"] = 1000.0 * max(slopes) * 1e-4
    m["failbefore_total_minus_1"] = _total(bad) - 1.0

    # ---- the LADDER's own populations -------------------------------------
    refs = {12: ref12, 14: ref14}
    right, wrong, grey, rows = [], [], [], []
    for deg, d in _LADDER:
        res = _solve(d, deg, guard=False)
        e = _err(res, refs[deg])
        kind = ("wrong" if e > 100.0 * d else
                "right" if e <= 10.0 * d else "grey")
        tot = _total(res)
        try:
            _solve(d, deg, guard=True)
            refused = False
        except ValueError:
            refused = True
        rows.append(dict(degree=deg, delta=d, err=e, err_over_delta=e / d,
                         total=tot, kind=kind, refused=refused))
        (right if kind == "right" else wrong if kind == "wrong"
         else grey).append(rows[-1])
    m["ladder_rows"] = rows
    m["ladder_n_right"] = len(right)
    m["ladder_n_wrong"] = len(wrong)
    m["ladder_n_grey"] = len(grey)
    m["ladder_max_absRT1_right"] = max(abs(r["total"] - 1.0) for r in right)
    m["ladder_min_RT1_wrong"] = min(r["total"] - 1.0 for r in wrong)
    m["ladder_max_eod_right"] = max(r["err_over_delta"] for r in right)
    m["ladder_min_eod_wrong"] = min(r["err_over_delta"] for r in wrong)
    bar = ps._STACK_SUPERUNITY_BAR
    m["bar_b"] = bar
    m["gap_below_bar_x"] = (bar / 100.0) / m["ladder_max_absRT1_right"]
    m["gap_above_bar_x"] = m["ladder_min_RT1_wrong"] / (bar * 30.0)

    # ---- the predictor constants (bars: 0.52 < c < 0.78, spread < 1.10) ----
    def _t3(e):
        return dict(exx=complex(e), exy=0.0, eyx=0.0, eyy=complex(e),
                    ezz=complex(e))
    k0 = 2.0 * np.pi / _WL
    kx0 = np.sin(_THETA) * k0
    segs = [_segs([_A0, _B0]), _segs([_A0 - 1e-4, _B0 + 1e-4])]
    uw, leps = _pmm_union_grid(segs, 1e-9)
    J = 0.5 * float(np.min(uw)) * _P
    consts = []
    for deg in (8, 12, 14, 16, 20):
        mm = _build_sem_tensor_segments(_P, uw, [_t3(e) for e in leps[0]],
                                        deg, 1, True)
        _W, _V, _lam, q = _sem_modes_tensor(mm, k0, kx0, True)
        consts.append(float(np.abs(q).max()) * k0 * J / (deg * (deg + 1) / 4.0))
    m["predictor_consts"] = consts
    m["predictor_spread"] = max(consts) / min(consts)

    # ---- the quoted |q| vs the measured one (bar: 0.9 < r < 1.1) ----------
    try:
        _solve(1e-4, 14, guard=True)
        m["quoted_over_measured"] = None
    except ValueError as exc:
        quoted = float(str(exc).split("|q| ~ ")[1].split(" ")[0])
        mm = _build_sem_tensor_segments(_P, uw, [_t3(e) for e in leps[1]],
                                        14, 1, True)
        _W, _V, _lam, q = _sem_modes_tensor(mm, k0, kx0, True)
        m["quoted"] = quoted
        m["measured_qmax"] = float(np.abs(q).max())
        m["quoted_over_measured"] = quoted / float(np.abs(q).max())
        m["prescribed_mf"] = float(str(exc).split("min_feature=")[1].split(" ")[0])
        m["prescribed_mf_bar"] = 1e-4 * _P

    # ---- the remedy bar (err <= 2 delta, closure < 1e-6) ------------------
    rem = []
    for deg in (12, 14, 16):
        r0 = _solve(0.0, deg, guard=False)
        for d in (1e-4, 5e-5, 3e-5, 1e-5):
            try:
                _solve(d, deg, guard=True)
                continue
            except ValueError as exc:
                mf = float(str(exc).split("min_feature=")[1].split(" ")[0])
            fixed = _solve(d, deg, guard=True, min_feature=mf)
            rem.append(dict(degree=deg, delta=d, err=_err(fixed, r0),
                            err_over_delta=_err(fixed, r0) / d,
                            closure=abs(_total(fixed) - 1.0)))
    m["remedy_rows"] = rem
    m["remedy_max_eod"] = max(r["err_over_delta"] for r in rem)
    m["remedy_max_closure"] = max(r["closure"] for r in rem)

    # ---- the per-layer caveat (bar: err < 1e-12) --------------------------
    m["per_layer_gap"] = {}
    for d in (1e-4, 3e-5):
        a = _solve(d, 14, guard=False)
        b = _solve(d, 14, guard=False, per_layer=True)
        m["per_layer_gap"][f"{d:g}"] = _err(a, b)

    print(json.dumps({k: v for k, v in m.items()
                      if k not in ("ladder_rows", "remedy_rows")},
                     indent=1, default=str))
    m["meta"] = dict(lumenairy=lib, python=sys.version.split()[0],
                     numpy=np.__version__)
    with open(out_path, "w") as f:
        json.dump(m, f, indent=1, default=str)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
