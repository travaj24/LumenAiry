"""V2b -- the cut rule at TANGENCIES and GRAZING crossings, kernel level
(Phase E2 verifier, item 2).  For each pair, the kernel's cross-mass at node
counts n = 6 .. 48 against its own adaptive result, three ways: shipped;
the square-root substitution at tangency ends OFF; tangencies NOT searched.
Also counts the tangency-flagged interval ends the shipped rule produced.
The brute-force physical oracle of the same pairs is v2_brute.py (case
names shared).

Usage: python v2b_tangent_ladder.py <case> [M]
"""
import sys
import time
import warnings

import numpy as np
from _ve import dump
from v2_brute import circle_map, grid, sin_map

from lumenairy.elements.pmm import _curvemortar as CMOR

case = sys.argv[1]
M = int(sys.argv[2]) if len(sys.argv) > 2 else 4
if case == "circ_sin":
    ga, gb = grid(circle_map(), M), grid(sin_map(), M)
elif case.startswith("sx_"):
    # NO singular vertices: the sinusoid x-wall map (crest x = 0.67 at
    # y = 0.3) over an UNMAPPED grid whose x-wall touches / grazes the crest
    off = {"sx_tan": 0.0, "sx_graze3": 1e-3, "sx_graze6": 1e-6,
           "sx_graze9": 1e-9}[case]
    ga = grid(sin_map(0.55, 0.12), M)
    gb = grid(None, M, walls=([0.0, 0.67 - off, 1.2], [0.0, 0.5, 1.2]))
elif case == "sinx_siny":
    ga = grid(sin_map(0.55, 0.12), M)
    gb = grid(sin_map(0.62, 0.10, axis="y", phase=0.4), M)
elif case == "circ_circ":
    ga = grid(circle_map(0.30, (0.5, 0.55)), M)
    gb = grid(circle_map(0.26, (0.74, 0.66)), M)
else:
    off = {"tangent": 0.0, "graze3": 1e-3, "graze6": 1e-6,
           "graze9": 1e-9}.get(case)
    if off is not None:
        ga = grid(circle_map(), M)
        gb = grid(None, M, walls=([0.0, 0.45, 1.2], [0.0, 0.24 + off,
                                                     1.2]))
    elif case == "tan_x":
        ga = grid(circle_map(), M)
        gb = grid(None, M, walls=([0.0, 0.96, 1.2], [0.0, 0.6, 1.2]))
    elif case == "sin_tan":
        # a sinusoid wall map over a circle whose rightmost point touches
        # the sinusoid's crest: x0 + A = 0.6 + 0.36 -> x0 = 0.84, A = 0.12
        # crest at y = 0.3 (sin = 1) -- the circle's 0-deg point is at
        # y = 0.6, so use a phase putting the crest at y = 0.6
        ga = grid(sin_map(0.84, 0.12, phase=-np.pi / 2 + 0.0), M)
        gb = grid(circle_map(), M)
    else:
        raise SystemExit(case)

orig_outer = CMOR._outer_rule
orig_tang = CMOR._tangencies
NSQ = [0]


def counting(o0, o1, sq0, sq1, n):
    NSQ[0] += int(bool(sq0)) + int(bool(sq1))
    return orig_outer(o0, o1, sq0, sq1, n)


def km(n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return CMOR.curved_cross_mass(ga, gb, n)


CMOR._outer_rule = counting
t0 = time.perf_counter()
with warnings.catch_warnings(record=True) as wl:
    warnings.simplefilter("always")
    Xad, nad, chg = CMOR.curved_cross_mass_adaptive(ga, gb)
t_ad = time.perf_counter() - t0
nsq_ad = NSQ[0]
CMOR._outer_rule = orig_outer
sc = float(np.max(np.abs(Xad)))
ns = [6, 9, 12, 18, 27, 40, 60]
out = {"case": case, "M": M, "adaptive_n": nad, "adaptive_change": chg,
       "adaptive_wall": t_ad, "adaptive_warnings": [str(w.message)[:160]
                                                   for w in wl],
       "tangency_ends_adaptive_run": nsq_ad, "n": ns}
for arm in ("shipped", "sqrt_off", "tang_missed"):
    if arm == "sqrt_off":
        CMOR._outer_rule = lambda o0, o1, a, b, n: orig_outer(o0, o1, False,
                                                              False, n)
    elif arm == "tang_missed":
        CMOR._tangencies = lambda *a, **k: []
    errs = []
    for n in ns:
        try:
            errs.append(float(np.max(np.abs(km(n) - Xad)) / sc))
        except Exception as ex:      # noqa: BLE001 -- recorded
            errs.append(f"{type(ex).__name__}: {str(ex)[:120]}")
    CMOR._outer_rule = orig_outer
    CMOR._tangencies = orig_tang
    out[arm] = errs
    print(case, arm, ["%.1e" % e if isinstance(e, float) else e
                      for e in errs])
print("adaptive n", nad, "change %.1e" % chg, "tangency ends", nsq_ad,
      "warnings", len(wl))
dump(f"v2b_tangent_{case}_M{M}", out)
