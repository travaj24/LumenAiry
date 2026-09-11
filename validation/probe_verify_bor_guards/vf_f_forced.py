"""TASK F / probe F -- FORCED-PREMISE probes: reproduce the body of the most
fragile gates with the smallest perturbation that removes the premise, and
report whether the gate's assertion would still hold.

Perturbations (argv[3], default 'all'):
  ncut_N      the near-cutoff band's NOISE side at finer radial grids
              (the gate's own comment says "a much finer radial grid would
              eat the decade" -- this measures whether it does)
  ncount      the near-cutoff CHANNEL COUNT over a longer ladder and at
              neighbouring N / m (is "one number" a library property or a
              property of exactly this 13-rung, N=120, m=0 sample?)
  taper512    the taper staircase one doubling past the gate's deepest arm
  nodalN      the nodal five-layer violation vs N and m (does the refusal's
              premise -- the blow-up -- survive off the fixture?)
  floormsg    the '1.02882' message pin: how far does max(R+T) move under
              small fixture perturbations?

Usage:  python vf_f_forced.py <pre|post> <tag> [which]
"""
from __future__ import annotations

import pathlib
import sys
import warnings as _w

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _vh  # noqa: E402

BUILD, TAG = sys.argv[1], sys.argv[2]
WHICH = sys.argv[3] if len(sys.argv) > 3 else "all"
_vh.require_tree(BUILD)
import lumenairy  # noqa: E402

print("lumenairy.__file__ =", lumenairy.__file__)
_WANT = "lum_vbor" if BUILD == "post" else "lum_vbor_pre"
assert pathlib.Path(lumenairy.__file__).resolve().parents[1].name.lower() == _WANT

from lumenairy.elements.bor import BORStack  # noqa: E402
from lumenairy.elements.bor import _orient as _or  # noqa: E402
from lumenairy.elements.bor import _sem_contract as _sc  # noqa: E402
from lumenairy.elements.bor import bor_solve as _bs  # noqa: E402
from lumenairy.elements.bor.bor_solve import build_layer, solve  # noqa: E402
from lumenairy.elements.bor.zcascade import layer_modes  # noqa: E402

_RBIG, _NREF = 24.0, 1.41
_EPS = _NREF ** 2
out = dict(arm=_vh.arm(), build=BUILD, tag=TAG, which=WHICH)


def _fd_modes(m, k0, eps=_EPS, N=120):
    return layer_modes(m, _RBIG, N,
                       lambda r: np.full_like(r, eps, dtype=complex),
                       float(k0), staggered=True)


def _sigma(L, k0):
    q = np.asarray(L["q"])
    scale = max(float(np.max(np.abs(q))) if q.size else 0.0, float(k0))
    return q, np.abs(q.imag) / scale


def _gamma_of(m, N=120, idx=2):
    L = _fd_modes(m, 2.0, N=N)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * _EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def _cutoff_stack(m, k0, N=120):
    s = BORStack(_RBIG, m, n_substrate=_NREF, n_superstrate=_NREF, N=N,
                 basis="fd")
    s.add_layer(0.4, eps=_EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=_EPS)
    s.set_source(k0=float(k0))
    return s.solve()


# ---- ncut_N: the BINDING noise bar at finer radial grids ----------------
if WHICH in ("all", "ncut_N"):
    band = _or._BOR_CUT_BAND_REL
    rows = []
    for N in (120, 180, 240, 360):
        worst, n = 0.0, 0
        for m in (0, 1, 2):
            g = _gamma_of(m, N=N)
            for e_ in range(4, 27, 4):
                d_ = 10.0 ** (-e_)
                kk = g / (_NREF * np.sqrt(1.0 - d_))
                L = _fd_modes(m, kk, N=N)
                q, sig = _sigma(L, kk)
                phys = np.abs(q.real) > 10.0 * np.abs(q.imag)
                if phys.any():
                    worst = max(worst, float(np.max(sig[phys])))
                    n += 1
        rows.append(dict(N=N, n=n, worst=worst, bar=band / 2.0,
                         margin=(band / 2.0) / worst if worst else None,
                         would_pass=bool(worst < band / 2.0 and n >= 18)))
        print("ncut_N  N=%3d  worst=%.4e  bar=%.1e  margin=%.3gx  pass=%s"
              % (N, worst, band / 2.0,
                 (band / 2.0) / worst if worst else float("inf"),
                 worst < band / 2.0), flush=True)
    out["ncut_N"] = rows

# ---- ncount: is the single channel count a library property? ------------
if WHICH in ("all", "ncount"):
    rows = []
    for (m, N, lo, hi) in ((0, 120, 8, 21), (0, 120, 2, 27), (0, 100, 8, 21),
                           (0, 140, 8, 21), (1, 120, 8, 21), (2, 120, 8, 21)):
        g = _gamma_of(m, N=N)
        counts, worst = {}, 0.0
        for e_ in range(lo, hi):
            d_ = 10.0 ** (-e_ / 2.0)
            kk = g / (_NREF * np.sqrt(1.0 - d_))
            r = _cutoff_stack(m, kk, N=N)
            c = int(np.size(r["R"]))
            counts[c] = counts.get(c, 0) + 1
            en = np.asarray(r["energy"])
            if en.size:
                worst = max(worst, float(np.max(np.abs(en - 1.0))))
        rows.append(dict(m=m, N=N, lo=lo, hi=hi,
                         counts={str(k): v for k, v in counts.items()},
                         n_distinct=len(counts), worst=worst,
                         would_pass=bool(len(counts) == 1 and worst < 1e-6)))
        print("ncount m=%d N=%d rungs %d..%d -> counts %s worst=%.4e pass=%s"
              % (m, N, lo, hi, counts, worst,
                 len(counts) == 1 and worst < 1e-6), flush=True)
    out["ncount"] = rows

# ---- taper512 -----------------------------------------------------------
if WHICH in ("all", "taper512"):
    def _stack(Rbig=24.0, m=1, k0=2.0, degree=8, N=160):
        st = BORStack(Rbig, m, basis="sem", degree=degree, N=N,
                      n_superstrate=1.0, n_substrate=1.5)
        st.set_source(k0=k0)
        return st

    rows = []
    for n_sl, deg, kk in ((256, 6, 0.8), (512, 6, 0.8)):
        st = _stack(degree=deg, k0=kk)
        for i in range(n_sl):
            r = 8.0 + (2.0 - 8.0) * (i + 0.5) / n_sl
            st.add_layer(1.2 / n_sl, segments=[(r, 6.0), (24.0, 2.0)])
        prev = _sc.BOR_SEM_MESH_GUARD
        _sc.BOR_SEM_MESH_GUARD = False
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                st.solve()
            recs = list(getattr(st, "_sem_mesh_report", []) or [])
        finally:
            _sc.BOR_SEM_MESH_GUARD = prev
        v = sorted({_sc.verdict(r) for r in recs})
        w = min(r["w_min_union_frac"] for r in recs)
        q = max(r["q_excess"] for r in recs if np.isfinite(r["q_excess"]))
        rows.append(dict(n_slices=n_sl, degree=deg, k0=kk, verdicts=v,
                         w=float(w), q=float(q),
                         would_pass=bool(v == ["ok"]
                                         and w > 3.0 * _sc._BOR_SLIVER_BAND_FRAC
                                         and q < _sc._BOR_Q_EXCESS / 3.0)))
        print("taper n=%d: v=%s w=%.4e (%.3gx) q=%.6g (%.3gx of 3333) pass=%s"
              % (n_sl, v, w, w / _sc._BOR_SLIVER_BAND_FRAC, q,
                 q / (_sc._BOR_Q_EXCESS / 3.0), rows[-1]["would_pass"]),
              flush=True)
    out["taper512"] = rows

# ---- nodalN: does the refusal's premise survive off the fixture? --------
if WHICH in ("all", "nodalN"):
    def _nuni(v):
        return lambda r: np.full_like(r, v, dtype=complex)

    def _nring(period, lo, hi, duty=0.5):
        def f(r):
            e = np.full_like(r, lo, dtype=complex)
            e[(r % period) < duty * period] = hi
            return e
        return f

    rows = []
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        for nl in (1, 2, 4, 8, 14):
            for N in (100, 140, 180):
                R = nl * 2.0 * np.pi / 2.0
                with _w.catch_warnings():
                    _w.simplefilter("ignore")
                    layers = [
                        build_layer(1, R, N, _nuni(2.0), 2.0, basis="nodal"),
                        build_layer(1, R, N, _nring(0.8, 2.0, 6.0), 2.0,
                                    thickness=0.5, basis="nodal"),
                        build_layer(1, R, N, _nuni(2.0), 2.0, thickness=0.3,
                                    basis="nodal"),
                        build_layer(1, R, N, _nring(1.2, 6.0, 2.0), 2.0,
                                    thickness=0.4, basis="nodal"),
                        build_layer(1, R, N, _nuni(2.0), 2.0, basis="nodal")]
                    e = np.asarray(solve(layers, 2.0)["energy"])
                v = float(np.max(e)) - 1.0
                rows.append(dict(n_lambda=nl, N=N, violation=v,
                                 refused=bool(v > _bs._BOR_NODAL_SUPERUNITY_BAR)))
                print("nodal nl=%2d N=%3d violation=%.6g refused=%s"
                      % (nl, N, v, v > _bs._BOR_NODAL_SUPERUNITY_BAR),
                      flush=True)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev
    out["nodalN"] = rows

# ---- floormsg: how stable is the 5-digit '1.02882' message pin? ---------
if WHICH in ("all", "floormsg"):
    def _nuni(v):
        return lambda r: np.full_like(r, v, dtype=complex)

    def _nring(period, lo, hi, duty=0.5):
        def f(r):
            e = np.full_like(r, lo, dtype=complex)
            e[(r % period) < duty * period] = hi
            return e
        return f

    rows = []
    prev = _bs.BOR_NODAL_PASSIVITY_GUARD
    _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        for N in (198, 199, 200, 201, 202):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                layers = [build_layer(1, 4.0, N, _nuni(2.0), 2.0,
                                      basis="nodal"),
                          build_layer(1, 4.0, N, _nring(0.8, 2.0, 6.0), 2.0,
                                      thickness=0.5, basis="nodal"),
                          build_layer(1, 4.0, N, _nuni(2.0), 2.0,
                                      basis="nodal")]
                e = np.asarray(solve(layers, 2.0)["energy"])
            s = "%.6g" % (float(np.max(e)),)
            rows.append(dict(N=N, max_RT=float(np.max(e)), printed=s,
                             has_1_0288=("1.0288" in s),
                             maxdev=float(np.max(np.abs(e - 1.0)))))
            print("floormsg N=%d  max(R+T)=%s  has '1.0288'=%s  maxdev=%.5g"
                  % (N, s, "1.0288" in s, rows[-1]["maxdev"]), flush=True)
    finally:
        _bs.BOR_NODAL_PASSIVITY_GUARD = prev
    out["floormsg"] = rows

_vh.dump(pathlib.Path(__file__).with_name(
    "vf_f_forced_%s_%s_%s.json" % (BUILD, TAG, WHICH)), out)
