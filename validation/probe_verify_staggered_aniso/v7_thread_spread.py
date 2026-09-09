"""V7 -- how much do the test's bar quantities move when the BLAS kernel path
changes?

TESTING_STANDARDS rule 5 asks for a measured envelope under the last-bit
differences a legitimate build is entitled to produce.  A second LAPACK wheel
is not available here, but OpenBLAS's threaded kernels use different blocking
and a different reduction ORDER from the single-threaded ones, so running the
same quantities at ``OPENBLAS_NUM_THREADS`` 1 vs 4 vs 8 is a real (if partial)
probe of that envelope on one box.

Run it three times with different thread caps and diff the JSON:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python v7_thread_spread.py 1
    OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python v7_thread_spread.py 4
"""
import json
import os
import sys
import warnings

import numpy as np

import lumenairy

_ROOT = os.path.abspath("C:/tmp/lum_aniso")
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

sys.path.insert(0, os.path.join(_ROOT, "tests", "unit"))
import test_pmm2d_staggered_anisotropic as TT  # noqa: E402

tag = sys.argv[1]
OUT = {"threads_env": {k: os.environ.get(k) for k in
                       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "MKL_NUM_THREADS")}}

# G3 (bar 1e-11) -- the worst of the six parametrized combinations
OUT["g3_worst_M7"] = max(
    TT._g3_residual(t, 2, 7, th, ph)
    for t in (TT._LC, TT._GYRO)
    for th, ph in ((0.0, 0.0), (25 * np.pi / 180, 0.0),
                   (25 * np.pi / 180, 40 * np.pi / 180)))

# G4 (bars 1.3e-5 / 2.7e-5) -- the M = 8 residual
oracle = TT.pmm_jones_1d(TT._G4_P, TT._G4_RIDGE, TT._G4_GROOVE, 1.5, 1.0,
                         TT._G4_DEP, 0.5, TT._G4_WL, angle=0.22, degree=16,
                         stabilize=False)
d8 = TT._g4_residual(8, oracle)
d5 = TT._g4_residual(5, oracle)
OUT["g4_M8_dRT"], OUT["g4_M8_dJ"], OUT["g4_M8_yforb"] = d8
OUT["g4_ladder_ratio_d5_over_d8"] = d5[0] / d8[0]

# G6 (bar 1e-5) -- the M = 8 closure of both Hermitian cells
kw = dict(period_x=TT._P, period_y=TT._P, n_substrate=1.5, n_superstrate=1.0,
          depth=TT._DEP, wavelength=TT._WL, n_orders=4)
for nm, host in (("lc", TT._LC), ("gyro", TT._GYRO)):
    _o, R, T, _J = TT.pmm_jones_2d_staggered(eps_cell=TT._cell(host, TT._ISO),
                                             degree=8, **kw)
    OUT[f"g6_{nm}_M8"] = float(np.max(np.abs(R.sum(axis=1)
                                             + T.sum(axis=1) - 1.0)))

# G7 (bars 1e-11 / 1e-7)
A = TT._cell(TT._LC, TT._ISO)
OUT["g7_transpose_lc"] = list(TT._transpose_residual(A, TT._transpose_cell(A)))
G = TT._cell(TT._GYRO, TT._ISO)
Gt = TT._transpose_cell(G)
OUT["g7_transpose_gyro"] = list(TT._transpose_residual(G, Gt))
Gw = G.copy()
Gw[..., 0, 1], Gw[..., 1, 0] = G[..., 1, 0].copy(), G[..., 0, 1].copy()
OUT["g7_swapped_gyro"] = list(TT._transpose_residual(Gw, Gt))

# G8 (bars 1e-12 / 1e-11 / 10x)
c = TT._cell(TT._LC, TT._ISO)
st = TT.PMM2DStackPure(TT._P, TT._P, n_superstrate=1.0, n_substrate=1.5,
                       n_modes=6, n_orders=3)
st.add_layer(TT._DEP, eps_cell=c).add_layer(TT._DEP, eps_cell=c)
st.set_source(TT._WL, theta=0.15, phi=0.4)
_o2, R2, T2, J2 = st.solve()
_o1, R1, T1, J1 = TT.pmm_jones_2d_staggered(TT._P, TT._P, c, 1.5, 1.0,
                                            2 * TT._DEP, TT._WL, degree=6,
                                            n_orders=3, theta=0.15, phi=0.4)
OUT["g8a_split"] = float(max(np.max(np.abs(R2 - R1)), np.max(np.abs(T2 - T1)),
                             np.max(np.abs(J2 - J1))))
OUT["g8b_M3"] = TT._g8b_residual(3)
OUT["g8b_M5"] = TT._g8b_residual(5)
OUT["g8b_M7"] = TT._g8b_residual(7)

# G5 no-floor (bars 1e-10 / 1e-4)
cc = TT._g5_cell()
stag, hyb = {}, {}
for nor in (4, 8):
    o, R, T, _J = TT.pmm_jones_2d_staggered(TT._P, TT._P, cc, 1.5, 1.0,
                                            TT._DEP, TT._WL, degree=7,
                                            n_orders=nor)
    stag[nor] = np.concatenate([TT._order0(o, R), TT._order0(o, T)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        oh, Rh, Th, _ = TT.pmm_jones_2d(TT._P, TT._P, cc, 1.5, 1.0, TT._DEP,
                                        TT._WL, degree=9, n_orders=nor)
    hyb[nor] = np.concatenate([TT._order0(oh, Rh), TT._order0(oh, Th)])
OUT["g5_nofloor_staggered"] = float(np.max(np.abs(stag[4] - stag[8])))
OUT["g5_nofloor_hybrid"] = float(np.max(np.abs(hyb[4] - hyb[8])))

# tripwire fail-before margin (bar 5e-2)
hot = TT._cell(TT._GYRO, 100.0 * np.eye(3, dtype=complex))
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    _o, R, T, _J = TT.pmm_jones_2d_staggered(TT._P, TT._P, hot, 1.5, 1.0,
                                             TT._DEP, TT._WL, degree=3,
                                             n_orders=4)
OUT["tripwire_M3_dev"] = abs(float(R[0].sum() + T[0].sum()) - 1.0)

fn = ("C:/tmp/lum_aniso/validation/probe_verify_staggered_aniso/"
      f"out_v7_threads_{tag}.json")
json.dump(OUT, open(fn, "w"), indent=1, default=float)
print(json.dumps(OUT, indent=1, default=float))
