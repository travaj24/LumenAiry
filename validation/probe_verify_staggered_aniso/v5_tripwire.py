"""V5 -- audit of the new lossless-closure tripwire (_STAG_CLOSURE_TOL = 5e-2).

Three questions:
  1. is the build's fail-before margin (9.08e-02 against the 5e-02 window, a
     factor 1.82) real, and how does it move with the fixture?
  2. is there a fail-before with DECADES of margin instead of 1.8x, so the
     test can satisfy TESTING_STANDARDS rule 5 (a gap on BOTH sides)?
  3. does the guard stay silent when nothing is broken (the other side)?

For (2) the state is ENGINEERED rather than hoped for: the region solver's
H-partner block ``V`` is rescaled by ``1 + s`` through a monkeypatch of the
module-level ``_region_modes`` that ``stack2d_pure`` calls.  That is a
deliberately WRONG modal impedance -- the interface S-matrix then mismatches
and the cascade genuinely stops conserving energy -- while every permittivity
in the stack stays exactly Hermitian, so the guard's own lossless predicate
still says "unity is exact".  ``s`` is a continuous knob, so the violation can
be put decades outside the window instead of 1.8x.
"""
import json
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

import lumenairy.elements.pmm.stack2d_pure as SP  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_jones_2d_staggered,
)

# the build's own fail-before fixture, verbatim from the test file
_P, _WL, _DEP = 0.70e-6, 0.55e-6, 0.28e-6
_GYRO = np.array([[2.25, 0.5j, 0.0], [-0.5j, 2.25, 0.0], [0.0, 0.0, 2.0]],
                 dtype=complex)


def cell(host, pillar, n=2):
    c = np.empty((n, n, 3, 3), dtype=complex)
    c[:] = host
    c[0, 0] = pillar
    return c


def solve(M, pillar_eps=100.0, n_orders=4):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _o, R, T, _J = pmm_jones_2d_staggered(
            _P, _P, cell(_GYRO, pillar_eps * np.eye(3, dtype=complex)), 1.5,
            1.0, _DEP, _WL, degree=M, n_orders=n_orders)
    fired = any("closure violated" in str(w.message) for w in rec)
    dev = float(np.max(np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)))
    return dev, fired


OUT = {"tol": SP._STAG_CLOSURE_TOL}

# ---- 1. the build's claimed 1.82x margin -------------------------------
lad = {}
for eps_p in (4.0, 16.0, 36.0, 100.0, 400.0, 1600.0):
    for M in (3, 4, 5, 8):
        dev, fired = solve(M, eps_p)
        lad[f"eps{eps_p:g}_M{M}"] = dict(dev=dev, fired=fired,
                                         margin=dev / SP._STAG_CLOSURE_TOL)
OUT["ladder_eps_M"] = lad
OUT["best_available_graceful_violation"] = max(
    v["dev"] for v in lad.values())

# ---- 2. ENGINEERED break: a wrong modal impedance ----------------------
_rm = SP._region_modes
eng = {}
for s in (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3):
    def broken(solver, _s=s):
        W, V, lam, g2 = _rm(solver)
        return W, V * (1.0 + _s), lam, g2
    SP._region_modes = broken
    try:
        dev, fired = solve(8, 100.0)
    finally:
        SP._region_modes = _rm
    eng[f"s{s:g}"] = dict(dev=dev, fired=fired,
                          margin=dev / SP._STAG_CLOSURE_TOL)
OUT["engineered_V_rescale_M8"] = eng

# a well-resolved LOSSLESS cell of modest contrast, same knob
def solve_lc(M, s):
    def broken(solver):
        W, V, lam, g2 = _rm(solver)
        return W, V * (1.0 + s), lam, g2
    SP._region_modes = broken if s else _rm
    try:
        return solve(M, 4.0)
    finally:
        SP._region_modes = _rm


OUT["engineered_on_mild_cell_M8"] = {
    f"s{s:g}": dict(zip(("dev", "fired"), solve_lc(8, s)))
    for s in (0.0, 1e-3, 1e-2, 0.1, 0.3)}

fn = ("C:/tmp/lum_aniso/validation/probe_verify_staggered_aniso/"
      "out_v5_tripwire.json")
json.dump(OUT, open(fn, "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
print("written", fn)
