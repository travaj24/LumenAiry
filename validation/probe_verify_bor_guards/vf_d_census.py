"""TASK F / probe D -- RE-MEASURE the EXPENSIVE STEP-4 quantities: the ordinary
geometry census (3 groups x 3 degrees) and the taper staircase.

Usage:  python vf_d_census.py <pre|post> <tag> [groups] [degrees] [tapers]
  groups  comma list from base,mesh,sweep     (default all three)
  degrees comma list                          (default 6,8,12)
  tapers  comma list of "n:deg:k0"            (default the gate's six arms)
"""
from __future__ import annotations

import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _vh  # noqa: E402

BUILD, TAG = sys.argv[1], sys.argv[2]
GROUPS = (sys.argv[3].split(",") if len(sys.argv) > 3
          else ["base", "mesh", "sweep"])
DEGREES = ([int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4
           else [6, 8, 12])
TAPERS = (sys.argv[5] if len(sys.argv) > 5
          else "8:8:2.0,16:8:2.0,32:8:2.0,64:8:2.0,128:6:0.8,256:6:0.8")
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
            st.solve()
        return list(getattr(st, "_sem_mesh_report", []) or [])
    finally:
        _sc.BOR_SEM_MESH_GUARD = prev


def _families(degrees, group):
    out = []
    base = group in ("all", "base")
    mesh = group in ("all", "mesh")
    sweep = group in ("all", "sweep")
    for deg in degrees:
        if base:
            st = _stack(degree=deg); st.add_layer(0.5, eps=2.25)
            out.append(("uniform|d%d" % deg, st))
            st = _stack(degree=deg)
            st.add_layer(0.5, eps_tensor=(2.25, 2.25, 3.24))
            out.append(("uniaxial|d%d" % deg, st))
            st = _stack(degree=deg)
            st.add_layer(0.5, segments=[(6.0, 6.0), (12.0, 2.25), (24.0, 2.0)])
            out.append(("segments3|d%d" % deg, st))
            st = _stack(degree=deg)
            st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
            st.add_layer(0.4, segments=[(6.0, 2.25), (24.0, 4.0)])
            out.append(("coincident_walls|d%d" % deg, st))
        if mesh:
            for period, duty in ((3.0, 0.5), (1.5, 0.3), (0.8, 0.5)):
                st = _stack(degree=deg)
                st.add_layer(0.5, rings=(period, duty, 2.449, 1.414))
                out.append(("grating_p%g_d%g|d%d" % (period, duty, deg), st))
            st = _stack(degree=deg, elements_per_segment=3)
            st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
            out.append(("hp3|d%d" % deg, st))
            st = _stack(degree=deg, elements_per_segment=3, grade=True)
            st.add_layer(0.5, segments=[(6.0, 6.0), (24.0, 2.0)])
            out.append(("hp3_graded|d%d" % deg, st))
            st = _stack(degree=deg)
            st.add_layer(0.4, segments=[(6.0, 4.0), (24.0, 2.25)])
            st.add_layer(0.3, eps=2.25)
            st.add_layer(0.4, segments=[(9.0, 2.25), (24.0, 4.0)])
            out.append(("spacer_between_rings|d%d" % deg, st))
        if sweep:
            for mm in (0, 2, 5):
                st = _stack(m=mm, degree=deg)
                st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
                out.append(("ring_m%d|d%d" % (mm, deg), st))
            for kk in (0.8, 3.5, 8.0):
                st = _stack(k0=kk, degree=deg)
                st.add_layer(0.5, segments=[(6.0, 4.0), (24.0, 2.25)])
                out.append(("ring_k%g|d%d" % (kk, deg), st))
            sc = 1e-6
            st = _stack(Rbig=20.0 * sc, k0=2.0 * np.pi / (1.55 * sc),
                        degree=deg)
            st.add_layer(0.5 * sc, rings=(3.0 * sc, 0.5, 2.45, 1.41))
            out.append(("nm_units|d%d" % deg, st))
            st = _stack(degree=deg)
            st.add_layer(0.5, segments=[(6.0, -20.0 + 2.0j), (24.0, 2.25)])
            out.append(("lossy_metal_ring|d%d" % deg, st))
    return out


def _taper(n_slices, degree=8, k0=2.0, r_top=8.0, r_bot=2.0, H=1.2):
    st = _stack(degree=degree, k0=k0)
    for i in range(n_slices):
        r = r_top + (r_bot - r_top) * (i + 0.5) / n_slices
        st.add_layer(H / n_slices, segments=[(r, 6.0), (24.0, 2.0)])
    return st


out = dict(arm=_vh.arm(), build=BUILD, tag=TAG,
           SLIVER_BAND_FRAC=_sc._BOR_SLIVER_BAND_FRAC,
           Q_EXCESS=_sc._BOR_Q_EXCESS, groups=GROUPS, degrees=DEGREES)

cen = []
for group in GROUPS:
    for deg in DEGREES:
        fams = _families([deg], group)
        narrowest, worst_q, bad, per = np.inf, 0.0, [], []
        for label, st in fams:
            try:
                recs = _measure(st)
            except BaseException as exc:                  # noqa: BLE001
                bad.append("%s: raised %s" % (label, type(exc).__name__))
                continue
            v = sorted({_sc.verdict(r) for r in recs})
            fu = min((r["w_min_union_frac"] for r in recs
                      if np.isfinite(r["w_min_union_frac"])),
                     default=float("inf"))
            qq = max((r["q_excess"] for r in recs
                      if np.isfinite(r["q_excess"])), default=0.0)
            narrowest = min(narrowest, fu)
            worst_q = max(worst_q, qq)
            per.append(dict(label=label, verdicts=v,
                            w_min_union_frac=float(fu), q_excess=float(qq)))
            if v != ["ok"]:
                bad.append("%s -> %s" % (label, v))
        cen.append(dict(
            group=group, degree=deg, n_families=len(fams), bad=bad,
            narrowest=float(narrowest), worst_q=float(worst_q),
            narrowest_over_bar=(float(narrowest) / _sc._BOR_SLIVER_BAND_FRAC),
            worst_q_over_bar=(float(worst_q) / _sc._BOR_Q_EXCESS),
            assert_narrow_bar=3.0 * _sc._BOR_SLIVER_BAND_FRAC,
            assert_q_bar=_sc._BOR_Q_EXCESS / 10.0,
            per=per))
        print("census %s d%d: fams=%d narrowest=%.4e (%.4gx) worst_q=%.6g "
              "(assert < %.4g) bad=%s"
              % (group, deg, len(fams), narrowest,
                 narrowest / _sc._BOR_SLIVER_BAND_FRAC, worst_q,
                 _sc._BOR_Q_EXCESS / 10.0, bad), flush=True)
out["ordinary_census"] = cen

tap = []
for spec in [s for s in TAPERS.split(",") if s]:
    n, dg, kk = spec.split(":")
    n, dg, kk = int(n), int(dg), float(kk)
    recs = _measure(_taper(n, degree=dg, k0=kk))
    v = sorted({_sc.verdict(r) for r in recs})
    w = min(r["w_min_union_frac"] for r in recs)
    q = max(r["q_excess"] for r in recs if np.isfinite(r["q_excess"]))
    tap.append(dict(n_slices=n, degree=dg, k0=kk, verdicts=v,
                    w=float(w), q=float(q),
                    w_over_bar=float(w) / _sc._BOR_SLIVER_BAND_FRAC,
                    assert_w_bar=3.0 * _sc._BOR_SLIVER_BAND_FRAC,
                    q_over_bar=float(q) / _sc._BOR_Q_EXCESS,
                    assert_q_bar=_sc._BOR_Q_EXCESS / 3.0))
    print("taper n=%d d=%d k0=%g: v=%s w=%.6e (%.4gx bar, assert>%.3e) "
          "q=%.6g (assert<%.5g)"
          % (n, dg, kk, v, w, w / _sc._BOR_SLIVER_BAND_FRAC,
             3.0 * _sc._BOR_SLIVER_BAND_FRAC, q, _sc._BOR_Q_EXCESS / 3.0),
          flush=True)
out["taper"] = tap

_vh.dump(pathlib.Path(__file__).with_name(
    "vf_d_census_%s_%s.json" % (BUILD, TAG)), out)
