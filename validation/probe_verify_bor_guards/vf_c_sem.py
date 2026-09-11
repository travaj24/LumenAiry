"""TASK F / probe C -- RE-MEASURE the CHEAP STEP-4/5 quantities of
tests/unit/test_fix_bor_multilayer_guards.py: the delta ladder (one degree),
the separated control, the wall-free spacer, the within-layer liner, the FD
bit-identity claim, and the explicit-inverse census population.

Usage:  python vf_c_sem.py <pre|post> <tag> [degree]
"""
from __future__ import annotations

import math
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _vh  # noqa: E402

BUILD, TAG = sys.argv[1], sys.argv[2]
DEG = int(sys.argv[3]) if len(sys.argv) > 3 else 8
_vh.require_tree(BUILD)
import lumenairy  # noqa: E402

print("lumenairy.__file__ =", lumenairy.__file__)
_WANT = "lum_vbor" if BUILD == "post" else "lum_vbor_pre"
assert pathlib.Path(lumenairy.__file__).resolve().parents[1].name.lower() == _WANT

from lumenairy.elements.bor import BORStack  # noqa: E402
from lumenairy.elements.bor import _sem_contract as _sc  # noqa: E402


def _stack(Rbig=24.0, m=1, k0=2.0, degree=8, N=160,
           elements_per_segment=1, grade=False, basis="sem"):
    st = BORStack(Rbig, m, basis=basis, degree=degree, N=N,
                  n_superstrate=1.0, n_substrate=1.5,
                  elements_per_segment=elements_per_segment, grade=grade)
    st.set_source(k0=k0)
    return st


def _measure(st):
    prev = _sc.BOR_SEM_MESH_GUARD
    _sc.BOR_SEM_MESH_GUARD = False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = st.solve()
        recs = list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        _sc.BOR_SEM_MESH_GUARD = prev
    return recs, res


out = dict(arm=_vh.arm(), build=BUILD, tag=TAG, degree=DEG,
           SLIVER_BAND_FRAC=_sc._BOR_SLIVER_BAND_FRAC,
           MIN_ELEM_FRAC=_sc._BOR_MIN_ELEM_FRAC,
           Q_EXCESS=_sc._BOR_Q_EXCESS)


def _ulps(x, bar):
    """Signed distance from x to bar in ULPs of bar (how close to a knife edge)."""
    if not np.isfinite(x) or bar == 0:
        return None
    n, s = 0, float(bar)
    if x == bar:
        return 0
    up = x > bar
    while n < 4000:
        s = math.nextafter(s, math.inf if up else -math.inf)
        n += 1
        if (up and s >= x) or ((not up) and s <= x):
            return n if up else -n
    return (1 if up else -1) * 4000


# --- the DELTA LADDER -----------------------------------------------------
lad = []
for dl in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7):
    d = dl * 24.0
    st = _stack(degree=DEG, N=200)
    st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
    st.add_layer(0.5, segments=[(6.0 + d, 2.0), (24.0, 6.0)])
    err = None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            st.solve()
        except BaseException as exc:                     # noqa: BLE001
            err = type(exc).__name__
    recs = list(getattr(st, "_sem_mesh_report", []) or [])
    v = sorted({_sc.verdict(r) for r in recs}) if recs else []
    got = ("refuse" if "refuse" in v
           else "warn" if any(x.startswith("warn") for x in v) else "ok")
    fu = min((r["w_min_union_frac"] for r in recs), default=float("inf"))
    qq = max((r["q_excess"] for r in recs
              if np.isfinite(r["q_excess"])), default=float("nan"))
    lad.append(dict(
        delta_frac=dl, verdict=got, verdicts=v, raised=err,
        warned=any("MANUFACTURED" in str(x.message) for x in w),
        w_min_union_frac=float(fu), q_excess=float(qq),
        fu_over_sliver=float(fu) / _sc._BOR_SLIVER_BAND_FRAC,
        fu_over_minelem=float(fu) / _sc._BOR_MIN_ELEM_FRAC,
        ulps_to_sliver=_ulps(fu, _sc._BOR_SLIVER_BAND_FRAC),
        ulps_to_minelem=_ulps(fu, _sc._BOR_MIN_ELEM_FRAC),
        q_over_bar=float(qq) / _sc._BOR_Q_EXCESS))
out["delta_ladder"] = lad

# --- the SEPARATED control ------------------------------------------------
sep = []
for dfr in (1e-2, 1e-4, 1e-6, 1e-7):
    d = dfr * 24.0
    st = _stack(degree=8, N=200)
    st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
    st.add_layer(0.3, eps=2.0)
    st.add_layer(0.3, eps=2.0)
    st.add_layer(0.5, segments=[(6.0 + d, 2.0), (24.0, 6.0)])
    recs, _ = _measure(st)
    sep.append(dict(delta_frac=dfr,
                    verdicts=sorted({_sc.verdict(r) for r in recs}),
                    all_union_inf=all(not np.isfinite(r["w_min_union_frac"])
                                      for r in recs),
                    union=[float(r["w_min_union_frac"]) for r in recs]))
out["separated_control"] = sep

# --- the WALL-FREE SPACER -------------------------------------------------
d = 1e-6 * 24.0
st = _stack(degree=8, N=200)
st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
st.add_layer(0.3, eps=2.0)
st.add_layer(0.5, segments=[(6.0 + d, 2.0), (24.0, 6.0)])
recs, _ = _measure(st)
out["wall_free_spacer"] = dict(
    spacer_union_frac=float(recs[1]["w_min_union_frac"]),
    finite=bool(np.isfinite(recs[1]["w_min_union_frac"])),
    bar=1e-5, all_union=[float(r["w_min_union_frac"]) for r in recs])

# --- the WITHIN-LAYER LINER ----------------------------------------------
lin = []
for wf in (1e-5, 1e-6, 1e-7):
    w = wf * 24.0
    st = _stack(degree=8, N=200)
    st.add_layer(0.5, segments=[(6.0, 6.0), (6.0 + w, 2.0), (24.0, 2.0)])
    recs, _ = _measure(st)
    v = sorted({_sc.verdict(r) for r in recs})
    q = max(r["q_excess"] for r in recs if np.isfinite(r["q_excess"]))
    narrow = min(r["w_min_frac"] for r in recs)
    lin.append(dict(w_frac=wf, verdicts=v, q_excess=float(q),
                    narrow=float(narrow),
                    branch_hot=bool(q > _sc._BOR_Q_EXCESS
                                    and narrow < _sc._BOR_MIN_ELEM_FRAC),
                    ulps_narrow_to_minelem=_ulps(narrow,
                                                 _sc._BOR_MIN_ELEM_FRAC),
                    all_union_inf=all(not np.isfinite(r["w_min_union_frac"])
                                      for r in recs)))
out["liner"] = lin


# --- FD structural immunity / bit identity --------------------------------
def _fd(delta):
    st = _stack(degree=8, N=200, basis="fd")
    st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
    st.add_layer(0.5, segments=[(6.0 + delta, 2.0), (24.0, 6.0)])
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = st.solve()
    return (np.asarray(res["R"]), [str(x.message) for x in w],
            getattr(st, "_sem_mesh_report", None))


base, wb, rep_b = _fd(0.0)
fd_rows = [dict(delta_frac=0.0, hash=_vh.hash_arrays(base), n=int(base.size),
                report_is_none=rep_b is None)]
for dl in (1e-4, 1e-6, 1e-7):
    r, w, rep = _fd(dl * 24.0)
    fd_rows.append(dict(delta_frac=dl, hash=_vh.hash_arrays(r), n=int(r.size),
                        report_is_none=rep is None,
                        bit_identical=bool(r.shape == base.shape
                                           and np.array_equal(r, base)),
                        max_dR=(float(np.max(np.abs(r - base)))
                                if r.shape == base.shape else None),
                        warned=any("MANUFACTURED" in m for m in w)))
out["fd_immunity"] = fd_rows

# --- the explicit-inverse CENSUS -----------------------------------------
from lumenairy.elements.bor import _inv_census as _ic  # noqa: E402

rec = []
prev = _ic._BOR_INV_CENSUS
_ic._BOR_INV_CENSUS = rec
try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for basis in ("fd", "sem"):
            for m in (0, 1, 2):
                st = _stack(m=m, degree=8, basis=basis)
                st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
                st.add_layer(0.4, segments=[(9.0, 2.25), (24.0, 4.0)])
                st.solve()
            st = _stack(degree=8, basis=basis)
            st.add_layer(0.5, eps=1.0)
            st.add_layer(0.4, segments=[(6.0, 4.0), (24.0, 2.25)])
            st.solve()
finally:
    _ic._BOR_INV_CENSUS = prev
rconds = [r[2] for r in rec if np.isfinite(r[2])]
resids = [r[3] for r in rec if np.isfinite(r[3])]
worst_site = min(rec, key=lambda r: r[2] if np.isfinite(r[2]) else np.inf)
worst_res = max(rec, key=lambda r: r[3] if np.isfinite(r[3]) else -np.inf)
out["inv_census"] = dict(
    n=len(rec), n_finite_rcond=len(rconds), n_finite_resid=len(resids),
    sites=sorted({r[0] for r in rec}),
    min_rcond=float(min(rconds)), max_resid=float(max(resids)),
    min_rcond_site=worst_site[0], max_resid_site=worst_res[0],
    bar_rcond=1e-9, bar_resid=1e-9,
    census_default_is_none=(_ic._BOR_INV_CENSUS is None))

_vh.dump(pathlib.Path(__file__).with_name(
    "vf_c_sem_%s_%s_d%d.json" % (BUILD, TAG, DEG)), out)
for r in lad:
    print("delta %.0e -> %-6s fu=%.6e (%.4gx sliver, %.4gx minelem, ulps %s/%s)"
          " q=%.6g (%.4gx)"
          % (r["delta_frac"], r["verdict"], r["w_min_union_frac"],
             r["fu_over_sliver"], r["fu_over_minelem"],
             r["ulps_to_sliver"], r["ulps_to_minelem"],
             r["q_excess"], r["q_over_bar"]))
print("spacer union frac %.4e" % (out["wall_free_spacer"]["spacer_union_frac"],))
for r in lin:
    print("liner %.0e: v=%s q=%.5g narrow=%.5e hot=%s"
          % (r["w_frac"], r["verdicts"], r["q_excess"], r["narrow"],
             r["branch_hot"]))
print("FD bit-identical:", [r.get("bit_identical") for r in fd_rows[1:]])
print("census n=%d min_rcond=%.4e (%s) max_resid=%.4e (%s)"
      % (len(rec), min(rconds), worst_site[0], max(resids), worst_res[0]))
