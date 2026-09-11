"""TASK F / probe B -- RE-MEASURE every asserted quantity in STEP 3 of
tests/unit/test_fix_bor_multilayer_guards.py (the nodal passivity refusal) and
the three changed gates in tests/unit/test_bor_solve.py.

Usage:  python vf_b_nodal.py <pre|post> <tag>
"""
from __future__ import annotations

import pathlib
import sys
import warnings as _w

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _vh  # noqa: E402

BUILD, TAG = sys.argv[1], sys.argv[2]
_vh.require_tree(BUILD)
import lumenairy  # noqa: E402

print("lumenairy.__file__ =", lumenairy.__file__)
_WANT = "lum_vbor" if BUILD == "post" else "lum_vbor_pre"
assert pathlib.Path(lumenairy.__file__).resolve().parents[1].name.lower() == _WANT

from lumenairy.elements.bor import bor_solve as _bs  # noqa: E402
from lumenairy.elements.bor.bor_solve import build_layer, solve  # noqa: E402

POST = BUILD == "post"
BAR = getattr(_bs, "_BOR_NODAL_SUPERUNITY_BAR", None)
WARN = getattr(_bs, "_BOR_NODAL_SUPERUNITY_WARN", None)


def _nuni(v):
    return lambda r: np.full_like(r, v, dtype=complex)


def _nring(period, lo, hi, duty=0.5):
    def f(r):
        e = np.full_like(r, lo, dtype=complex)
        e[(r % period) < duty * period] = hi
        return e
    return f


def _five(basis, n_lambda, N=140, m=1, k0=2.0):
    R = n_lambda * 2.0 * np.pi / k0
    return [build_layer(m, R, N, _nuni(2.0), k0, basis=basis),
            build_layer(m, R, N, _nring(0.8, 2.0, 6.0), k0, thickness=0.5,
                        basis=basis),
            build_layer(m, R, N, _nuni(2.0), k0, thickness=0.3, basis=basis),
            build_layer(m, R, N, _nring(1.2, 6.0, 2.0), k0, thickness=0.4,
                        basis=basis),
            build_layer(m, R, N, _nuni(2.0), k0, basis=basis)]


def _disarmed(fn):
    prev = getattr(_bs, "BOR_NODAL_PASSIVITY_GUARD", None)
    if prev is not None:
        _bs.BOR_NODAL_PASSIVITY_GUARD = False
    try:
        return fn()
    finally:
        if prev is not None:
            _bs.BOR_NODAL_PASSIVITY_GUARD = prev


out = dict(arm=_vh.arm(), build=BUILD, tag=TAG, bar=BAR, warn_edge=WARN)

# --- the five-layer nodal ladder: the violation the refusal keys on -------
rows = []
for nl in (1, 2, 4, 8, 14):
    def go(nl=nl):
        with _w.catch_warnings():
            _w.simplefilter("ignore")
            return np.asarray(solve(_five("nodal", nl), 2.0)["energy"])
    e = _disarmed(go)
    rows.append(dict(n_lambda=nl, max_RT=float(np.max(e)),
                     violation=float(np.max(e)) - 1.0, n=int(e.size)))
out["nodal_five_layer_disarmed"] = rows

# --- ARMED: does the refusal actually fire at every radius? ---------------
armed = []
for nl in (1, 2, 4, 8, 14):
    with _w.catch_warnings(record=True) as w:
        _w.simplefilter("always")
        layers = _five("nodal", nl)
        proxy = any("vacuum wavelengths" in str(x.message) for x in w)
        try:
            solve(layers, 2.0)
            armed.append(dict(n_lambda=nl, raised=None, proxy_warned=proxy,
                              warns=[str(x.message)[:120] for x in w]))
        except BaseException as exc:                     # noqa: BLE001
            armed.append(dict(n_lambda=nl, raised=type(exc).__name__,
                              proxy_warned=proxy, msg=str(exc)[:400]))
out["nodal_five_layer_armed"] = armed

# --- the staggered twin closure (bar 1e-9) --------------------------------
tw = []
for nl in (1, 2, 4, 8, 14):
    e = np.asarray(solve(_five("staggered", nl), 2.0)["energy"])
    tw.append(dict(n_lambda=nl, closure=float(np.max(np.abs(e - 1.0))),
                   n=int(e.size)))
out["staggered_twin"] = dict(rows=tw, bar=1e-9,
                             worst=max(r["closure"] for r in tw))

# --- HEALTHY side of the two-sided bar gate (bar = BAR/1e2 = 1e-5) --------
ok_rows, worst_ok, n_ok = [], 0.0, 0
for m in (0, 1, 2):
    for rl in (0.5, 1.0, 2.0):
        R = rl * 2.0 * np.pi / 2.0

        def go(m=m, R=R):
            with _w.catch_warnings():
                _w.simplefilter("ignore")
                layers = [build_layer(m, R, 120, _nuni(2.0), 2.0,
                                      basis="nodal"),
                          build_layer(m, R, 120, _nuni(2.5), 2.0,
                                      thickness=0.5, basis="nodal"),
                          build_layer(m, R, 120, _nuni(2.0), 2.0,
                                      basis="nodal")]
                return np.asarray(solve(layers, 2.0)["energy"])
        e = _disarmed(go)
        if e.size:
            v = float(np.max(e)) - 1.0
            worst_ok = max(worst_ok, v)
            n_ok += 1
            ok_rows.append(dict(m=m, rl=rl, violation=v, n=int(e.size)))
out["healthy_uniform_small_cell"] = dict(n=n_ok, worst=worst_ok,
                                         bar=(BAR / 1e2) if BAR else None,
                                         rows=ok_rows)


# --- BROKEN side: the mildest row (bar = 10*BAR = 1e-2) -------------------
def _mild():
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        layers = [build_layer(1, 4.0, 200, _nuni(2.0), 2.0, basis="nodal"),
                  build_layer(1, 4.0, 200, _nring(0.8, 2.0, 6.0), 2.0,
                              thickness=0.5, basis="nodal"),
                  build_layer(1, 4.0, 200, _nuni(2.0), 2.0, basis="nodal")]
        return np.asarray(solve(layers, 2.0)["energy"])


e = _disarmed(_mild)
out["mildest_broken_row"] = dict(max_RT=float(np.max(e)),
                                 violation=float(np.max(e)) - 1.0,
                                 bar=(10.0 * BAR) if BAR else None,
                                 max_abs_dev=float(np.max(np.abs(e - 1.0))),
                                 n_inc=int(e.size))

# --- test_bor_solve's three changed gates ---------------------------------
#   the message pin  '1.02882' / '1.0288'  and the pre-fix bar  < 0.05
def _floor_stack():
    m, R, N, k0 = 1, 4.0, 200, 2.0
    return k0, [build_layer(m, R, N, _nuni(2.0), k0, basis="nodal"),
                build_layer(m, R, N, _nring(0.8, 2.0, 6.0), k0, thickness=0.5,
                            basis="nodal"),
                build_layer(m, R, N, _nuni(2.0), k0, basis="nodal")]


k0, layers = _floor_stack()
msg = None
if POST:
    try:
        solve(layers, k0)
    except BaseException as exc:                        # noqa: BLE001
        msg = str(exc)
res = _disarmed(lambda: solve(_floor_stack()[1], 2.0))
en = np.asarray(res["energy"])
out["bor_solve_floor_stack"] = dict(
    raised_msg_head=(msg[:260] if msg else None),
    msg_contains_1_02882=(("1.02882" in msg) if msg else None),
    msg_contains_1_0288=(("1.0288" in msg) if msg else None),
    max_RT_printed="%.6g" % (float(np.max(en)),),
    max_RT=float(np.max(en)),
    max_abs_dev=float(np.max(np.abs(en - 1.0))),
    pre_fix_bar=0.05, n_inc=int(len(res["inc"])))

# --- the staggered twin of that stack (bar 1e-9) --------------------------
m, R, N = 1, 4.0, 200
lay = [build_layer(m, R, N, _nuni(2.0), 2.0, basis="staggered"),
       build_layer(m, R, N, _nring(0.8, 2.0, 6.0), 2.0, thickness=0.5,
                   basis="staggered"),
       build_layer(m, R, N, _nuni(2.0), 2.0, basis="staggered")]
r = solve(lay, 2.0)
en = np.asarray(r["energy"])
out["bor_solve_staggered_twin"] = dict(
    closure=float(np.max(np.abs(en - 1.0))), n_inc=int(len(r["inc"])),
    bar=1e-9)

_vh.dump(pathlib.Path(__file__).with_name(
    "vf_b_nodal_%s_%s.json" % (BUILD, TAG)), out)
print("five-layer nodal violations:",
      ["%.5g" % r["violation"] for r in out["nodal_five_layer_disarmed"]])
print("armed raises:", [r["raised"] for r in armed],
      " proxy:", [r["proxy_warned"] for r in armed])
print("staggered twin worst %.4e (bar 1e-9)" % (out["staggered_twin"]["worst"],))
print("healthy worst %.4e (bar %s)" % (worst_ok, out["healthy_uniform_small_cell"]["bar"]))
print("mildest broken %.4e (bar %s)" % (out["mildest_broken_row"]["violation"],
                                        out["mildest_broken_row"]["bar"]))
print("floor stack max(R+T)=%s  msg has 1.0288=%s  maxdev=%.5g (bar 0.05)"
      % (out["bor_solve_floor_stack"]["max_RT_printed"],
         out["bor_solve_floor_stack"]["msg_contains_1_0288"],
         out["bor_solve_floor_stack"]["max_abs_dev"]))
