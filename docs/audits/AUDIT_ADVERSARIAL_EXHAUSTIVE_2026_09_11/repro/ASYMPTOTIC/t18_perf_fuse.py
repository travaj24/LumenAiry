"""T18: quantify the redundant-basis-build cost in eval_s1_with_v2_grad /
eval_phi_with_v2_grad / _phi_v2_hessian_batch."""
import sys, time
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy._math.chebyshev import (chebyshev_vandermonde as cv,
                                       chebyshev_derivative_vandermonde as cdv)
from lumenairy.elements.lenses import (_multi_indices_total_degree,
                                       _evaluate_polynomial_4d_and_grad34)

order = 6
mi = _multi_indices_total_degree(4, order)
K = np.asarray(mi, dtype=np.int64)
K1, K2, K3, K4 = K[:,0], K[:,1], K[:,2], K[:,3]
M = len(mi)
rng = np.random.default_rng(1)
cx = rng.normal(size=M); cy = rng.normal(size=M); cp = rng.normal(size=M)

def bench(f, reps=5):
    f(); t = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter()-t)/reps

for N in (1024, 4096, 16384, 65536):
    u1, u2, u3, u4 = [rng.uniform(-1, 1, N) for _ in range(4)]
    # -- as shipped: two independent calls for s1x/s1y + one for phi
    def shipped():
        _evaluate_polynomial_4d_and_grad34(cx, mi, u1, u2, u3, u4, order)
        _evaluate_polynomial_4d_and_grad34(cy, mi, u1, u2, u3, u4, order)
        _evaluate_polynomial_4d_and_grad34(cp, mi, u1, u2, u3, u4, order)
    # -- fused: one basis build, three tensordots
    def fused():
        T1 = cv(u1, order); T2 = cv(u2, order); T3 = cv(u3, order); T4 = cv(u4, order)
        dT3 = cdv(u3, order); dT4 = cdv(u4, order)
        T12 = T1[K1]*T2[K2]
        bf = T12*T3[K3]*T4[K4]; b3 = T12*dT3[K3]*T4[K4]; b4 = T12*T3[K3]*dT4[K4]
        C = np.stack([cx, cy, cp])
        np.tensordot(C, bf, axes=([1],[0]))
        np.tensordot(C, b3, axes=([1],[0]))
        np.tensordot(C, b4, axes=([1],[0]))
    # -- fused + T12 hoisted (Newton-loop case: u1,u2 fixed)
    T1c = cv(u1, order); T2c = cv(u2, order); T12c = T1c[K1]*T2c[K2]
    def fused_hoist():
        T3 = cv(u3, order); T4 = cv(u4, order)
        dT3 = cdv(u3, order); dT4 = cdv(u4, order)
        bf = T12c*T3[K3]*T4[K4]; b3 = T12c*dT3[K3]*T4[K4]; b4 = T12c*T3[K3]*dT4[K4]
        C = np.stack([cx, cy, cp])
        np.tensordot(C, bf, axes=([1],[0]))
        np.tensordot(C, b3, axes=([1],[0]))
        np.tensordot(C, b4, axes=([1],[0]))
    a = bench(shipped); b = bench(fused); c = bench(fused_hoist)
    print(f"N={N:6d}  shipped {a*1e3:8.2f} ms | fused {b*1e3:8.2f} ms ({a/b:.2f}x) "
          f"| fused+T12 hoisted {c*1e3:8.2f} ms ({a/c:.2f}x)   "
          f"[basis tensor (M={M}) = {M*N*8/1e6:.1f} MB each]")
