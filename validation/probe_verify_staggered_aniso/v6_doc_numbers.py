"""V6 -- re-measure the BUILD DOC's own headline numbers on the build's own
fixtures, and probe the headroom of the bars that derive from them.

Task 2 of the verification brief is re-measurement on FRESH fixtures (v2);
this file is the complementary half: "right conclusion, wrong numbers" is a
defect, so the doc's tables themselves are re-run.  It imports the shipped
test module's helpers so the fixtures are byte-identical to the ones the bars
were derived from.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

sys.path.insert(0, os.path.join(_ROOT, "tests", "unit"))
import test_pmm2d_staggered_anisotropic as TT  # noqa: E402

OUT = {}

# ---- T3 (G3): the six (tensor, incidence) combinations at M = 5 and 7 ----
g3 = {}
for name, t in (("lc", TT._LC), ("gyro", TT._GYRO)):
    for th, ph in ((0.0, 0.0), (25 * np.pi / 180, 0.0),
                   (25 * np.pi / 180, 40 * np.pi / 180)):
        for M in (5, 7):
            g3[f"{name}_th{int(round(np.degrees(th)))}_"
               f"ph{int(round(np.degrees(ph)))}_M{M}"] = TT._g3_residual(
                   t, 2, M, th, ph)
g3["lc_multiseg_33_M7"] = TT._g3_residual(TT._LC, 3, 7, 25 * np.pi / 180,
                                          40 * np.pi / 180)
OUT["T3_g3"] = g3

# ---- T4 (G4): the M ladder --------------------------------------------
oracle = TT.pmm_jones_1d(TT._G4_P, TT._G4_RIDGE, TT._G4_GROOVE, 1.5, 1.0,
                         TT._G4_DEP, 0.5, TT._G4_WL, angle=0.22, degree=16,
                         stabilize=False)
OUT["T4_g4"] = {f"M{M}": dict(zip(("dRT", "dJ", "yforb"),
                                  TT._g4_residual(M, oracle)))
                for M in (5, 6, 7, 8)}

# ---- T8b (G8): the uniform-tensor multilayer ladder, extended to M = 3/4
OUT["T8b_g8"] = {f"M{M}": TT._g8b_residual(M) for M in (3, 4, 5, 6, 7)}

# ---- T7 (G7): the transpose / swapped-placement controls ---------------
g7 = {}
A = TT._cell(TT._LC, TT._ISO)
g7["transpose_lc"] = TT._transpose_residual(A, TT._transpose_cell(A))
G = TT._cell(TT._GYRO, TT._ISO)
Gt = TT._transpose_cell(G)
g7["transpose_gyro"] = TT._transpose_residual(G, Gt)
Gw = G.copy()
Gw[..., 0, 1], Gw[..., 1, 0] = G[..., 1, 0].copy(), G[..., 0, 1].copy()
g7["swapped_gyro"] = TT._transpose_residual(Gw, Gt)
Aw = A.copy()
Aw[..., 0, 1], Aw[..., 1, 0] = A[..., 1, 0].copy(), A[..., 0, 1].copy()
g7["swapped_lc_is_a_noop"] = TT._transpose_residual(Aw,
                                                    TT._transpose_cell(A))
OUT["T7_g7"] = {k: list(v) for k, v in g7.items()}

# ---- T5 (G5): the no-floor two-sided numbers ---------------------------
t0 = time.time()
c = TT._g5_cell()
stag, hyb = {}, {}
import warnings  # noqa: E402

for nor in (4, 8):
    o, R, T, _J = TT.pmm_jones_2d_staggered(TT._P, TT._P, c, 1.5, 1.0,
                                            TT._DEP, TT._WL, degree=7,
                                            n_orders=nor)
    stag[nor] = np.concatenate([TT._order0(o, R), TT._order0(o, T)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oh, Rh, Th, _ = TT.pmm_jones_2d(TT._P, TT._P, c, 1.5, 1.0, TT._DEP,
                                        TT._WL, degree=9, n_orders=nor)
    hyb[nor] = np.concatenate([TT._order0(oh, Rh), TT._order0(oh, Th)])
OUT["T5_nofloor"] = dict(staggered=float(np.max(np.abs(stag[4] - stag[8]))),
                         hybrid=float(np.max(np.abs(hyb[4] - hyb[8]))),
                         wall=round(time.time() - t0, 2))

# ---- T9 (G9): the absorption ladder ------------------------------------
def g9(M):
    st = TT.PMM2DStackPure(TT._P, TT._P, n_superstrate=1.0, n_substrate=1.5,
                           n_modes=M, n_orders=3)
    st.add_layer(0.12e-6, eps_cell=TT._cell(TT._LC, TT._ISO))
    st.add_layer(TT._DEP,
                 eps_cell=TT._cell(TT._LC, TT._ISO + 0.8j * np.eye(3)))
    st.add_layer(0.09e-6, eps=TT._GYRO)
    st.set_source(TT._WL, theta=0.12, phi=0.3)
    _o, R, T, _J = st.solve(retain_internal=True)
    A = st.layer_absorption()
    return [abs(float(A[:, c].sum()) - float(1.0 - R[c].sum() - T[c].sum()))
            for c in (0, 1)], float(np.max(np.abs(A[[0, 2]])))


OUT["T9_g9"] = {f"M{M}": dict(zip(("dev", "A_lossless_max"), g9(M)))
                for M in (6, 7, 8)}

# ---- T6 (G6) closure ladder --------------------------------------------
kw = dict(period_x=TT._P, period_y=TT._P, n_substrate=1.5, n_superstrate=1.0,
          depth=TT._DEP, wavelength=TT._WL, n_orders=4)
t6 = {}
for nm, host in (("lc", TT._LC), ("gyro", TT._GYRO)):
    for M in (6, 8):
        _o, R, T, _J = TT.pmm_jones_2d_staggered(
            eps_cell=TT._cell(host, TT._ISO), degree=M, **kw)
        t6[f"{nm}_M{M}"] = float(np.max(np.abs(R.sum(axis=1)
                                               + T.sum(axis=1) - 1.0)))
OUT["T6_g6"] = t6

fn = ("C:/tmp/lum_aniso/validation/probe_verify_staggered_aniso/"
      "out_v6_doc_numbers.json")
json.dump(OUT, open(fn, "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
print("written", fn)
