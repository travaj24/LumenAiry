"""TASK F / probe G -- the near-cutoff ladder, PER RUNG, over a window WIDER
than the gate's ``range(8, 21)``, so we can say whether the gate's
``len(counts) == 1`` is a library property or a property of the window it
picked, and whether a count of 2 at an extended rung is PHYSICS (a genuinely
different number of propagating orders at that k0) or the pathology.

Usage:  python vf_g_rungs.py <pre|post> <tag> [m]
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _vh  # noqa: E402

BUILD, TAG = sys.argv[1], sys.argv[2]
MS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["0"])]
_vh.require_tree(BUILD)
import lumenairy  # noqa: E402

print("lumenairy.__file__ =", lumenairy.__file__)
_WANT = "lum_vbor" if BUILD == "post" else "lum_vbor_pre"
assert pathlib.Path(lumenairy.__file__).resolve().parents[1].name.lower() == _WANT

from lumenairy.elements.bor import BORStack  # noqa: E402
from lumenairy.elements.bor.zcascade import layer_modes  # noqa: E402

_RBIG, _NFD, _NREF = 24.0, 120, 1.41
_EPS = _NREF ** 2


def _gamma_of(m, idx=2):
    L = layer_modes(m, _RBIG, _NFD,
                    lambda r: np.full_like(r, _EPS, dtype=complex), 2.0,
                    staggered=True)
    q = np.asarray(L["q"])
    g = np.sqrt(2.0 ** 2 * _EPS - q ** 2)
    g = np.real(g[np.abs(g.imag) < 1e-9 * np.maximum(np.abs(g.real), 1e-300)])
    return float(np.sort(g[g > 1e-6])[idx])


def _stack(m, k0):
    s = BORStack(_RBIG, m, n_substrate=_NREF, n_superstrate=_NREF, N=_NFD,
                 basis="fd")
    s.add_layer(0.4, eps=_EPS)
    s.add_layer(0.5, rings=(3.0, 0.5, 2.45, 1.41))
    s.add_layer(0.4, eps=_EPS)
    s.set_source(k0=float(k0))
    return s.solve()


out = dict(arm=_vh.arm(), build=BUILD, tag=TAG, rows=[])
for m in MS:
    g = _gamma_of(m)
    for e_ in range(2, 27):
        dl = 10.0 ** (-e_ / 2.0)
        k0 = g / (_NREF * np.sqrt(1.0 - dl))
        r = _stack(m, k0)
        en = np.asarray(r["energy"])
        row = dict(m=m, e=e_, in_gate_window=bool(8 <= e_ < 21),
                   qn=float(_NREF * np.sqrt(dl)), k0=float(k0),
                   n=int(np.size(r["R"])),
                   closure=float(np.max(np.abs(en - 1.0))) if en.size
                   else float("nan"))
        out["rows"].append(row)
        print("m=%d e=%2d %s qn=%.4e k0=%.10f n=%d closure=%.4e"
              % (m, e_, "GATE" if row["in_gate_window"] else "    ",
                 row["qn"], row["k0"], row["n"], row["closure"]), flush=True)

_vh.dump(pathlib.Path(__file__).with_name(
    "vf_g_rungs_%s_%s.json" % (BUILD, TAG)), out)
for m in MS:
    sel = [r for r in out["rows"] if r["m"] == m]
    ing = [r for r in sel if r["in_gate_window"]]
    print("m=%d: gate window counts %s worst %.4e | full counts %s worst %.4e"
          % (m, sorted({r["n"] for r in ing}),
             max(r["closure"] for r in ing),
             sorted({r["n"] for r in sel}),
             max(r["closure"] for r in sel)))
