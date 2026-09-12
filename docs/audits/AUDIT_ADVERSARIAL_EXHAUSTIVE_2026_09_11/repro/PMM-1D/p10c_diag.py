"""PROBE 10c (contention-robust): the SEM nodal masses are EXACTLY diagonal,
yet _safe_inv / _safe_solve invert them densely and every iS0 @ M is a dense
gemm.  A/B the end-to-end solve with a diagonal-aware _safe_inv, interleaved,
min-of-N (min is robust to CPU contention).  Also assert bit-agreement.
"""
import sys, time, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc
from lumenairy.elements.pmm import PMMStack

_orig_inv = pc._safe_inv
_orig_solve = pc._safe_solve
N_DIAG = [0, 0]          # [diagonal calls, total calls]
SIZES = []


def _counting_inv(A):
    N_DIAG[1] += 1
    off = A - np.diag(np.diag(A))
    if not off.any():
        N_DIAG[0] += 1
        SIZES.append(A.shape[0])
    return _orig_inv(A)


def _fast_inv(A):
    d = np.diag(A)
    if not (A - np.diag(d)).any():
        return np.diag(1.0 / d)
    return _orig_inv(A)


def _fast_solve(A, B):
    d = np.diag(A)
    if not (A - np.diag(d)).any():
        return B / d[:, None]
    return _orig_solve(A, B)


def build(degree, nlay, nseg):
    st = PMMStack(1.0e-6, n_substrate=1.444, n_superstrate=1.0, degree=degree,
                  far_field_orders=21)
    for i in range(nlay):
        w = np.diff(np.linspace(0, 1, nseg + 1)) * (1 + 0.0 * i)
        segs = [(float(w[j]), (3.48 ** 2 if (i + j) % 2 else 1.444 ** 2))
                for j in range(nseg)]
        segs[-1] = (1.0 - sum(x for x, _ in segs[:-1]), segs[-1][1])
        st.add_layer(0.05e-6, segments=segs)
    st.set_source(1.55e-6, angle=np.deg2rad(13.0))
    return st


print("=== how many _safe_inv calls are on EXACTLY DIAGONAL matrices? ===")
for degree, nlay, nseg in ((16, 6, 4), (24, 8, 6)):
    N_DIAG[0] = N_DIAG[1] = 0
    SIZES.clear()
    pc._safe_inv = _counting_inv
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            build(degree, nlay, nseg).solve()
    finally:
        pc._safe_inv = _orig_inv
    print(f"  degree={degree} nlay={nlay} nseg={nseg}: "
          f"{N_DIAG[0]} of {N_DIAG[1]} _safe_inv calls are diagonal, "
          f"sizes {sorted(set(SIZES))}")

print()
print("=== A/B end-to-end (interleaved, min of 7) ===")
for degree, nlay, nseg in ((16, 6, 4), (20, 8, 6), (24, 8, 6)):
    st = build(degree, nlay, nseg)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = st.solve()
    ta, tb = [], []
    for _ in range(7):
        for tag in ("A", "B"):
            if tag == "B":
                pc._safe_inv, pc._safe_solve = _fast_inv, _fast_solve
            try:
                st = build(degree, nlay, nseg)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t = time.perf_counter()
                    out = st.solve()
                    dt = time.perf_counter() - t
            finally:
                pc._safe_inv, pc._safe_solve = _orig_inv, _orig_solve
            (ta if tag == "A" else tb).append(dt)
            if tag == "B":
                d = max(np.max(np.abs(np.asarray(out[i]) - np.asarray(ref[i])))
                        for i in (1, 2, 3))
    a, b = min(ta), min(tb)
    print(f"  degree={degree} nlay={nlay} nseg={nseg}: shipped {a:.3f}s  "
          f"diag-aware {b:.3f}s  speed-up {a/b:.2f}x   max|d| vs shipped "
          f"= {d:.2e}")
