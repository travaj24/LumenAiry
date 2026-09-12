"""PROBE 10f: contention-robust cost SPLIT of a 1-D PMM stack solve.
Interleaved, min-of-N timings of each kernel in the same process.
Tests the module docstring's claim that the dense eig is "~85% of runtime".
"""
import sys, time, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc

per, wl = 1.0e-6, 1.55e-6
k0 = 2 * np.pi / wl
kx0 = np.sin(np.deg2rad(13.0)) * k0
t3 = lambda e: pc._tensor3_dict(e * np.eye(3))


def mk(degree, nseg):
    w = list(np.diff(np.linspace(0, 1, nseg + 1)))
    eps = [3.48 ** 2 if i % 2 else 1.444 ** 2 for i in range(nseg)]
    return pc._build_sem_tensor_segments(per, w, [t3(e) for e in eps],
                                         degree, 1, True)


def best(f, n=9):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        f()
        ts.append(time.perf_counter() - t)
    return min(ts)


print(f"{'n_glob':>7} {'2N':>5} {'modes(eig+asm)':>15} {'bare eig(2N)':>13} "
      f"{'interface':>11} {'star':>9} {'inv(2N)':>9} {'solve(2N)':>10}")
for degree, nseg in ((12, 6), (16, 6), (20, 8), (24, 8)):
    m = mk(degree, nseg)
    n = m["n_glob"]
    N2 = 2 * n
    W, V, lam, q = pc._sem_modes_tensor(m, k0, kx0)
    m2 = mk(degree, nseg)
    W2, V2, lam2, q2 = pc._sem_modes_tensor(m2, k0 * 1.01, kx0)
    Mrand = (np.random.rand(N2, N2) + 1j * np.random.rand(N2, N2))
    A = (np.random.rand(N2, N2) + 1j * np.random.rand(N2, N2)
         + N2 * np.eye(N2))
    B = np.random.rand(N2, N2) + 1j * np.random.rand(N2, N2)
    S = pc._interface_smatrix(W, V, W2, V2)
    fns = [
        ("modes", lambda: pc._sem_modes_tensor(m, k0, kx0)),
        ("eig", lambda: np.linalg.eig(Mrand)),
        ("ifc", lambda: pc._interface_smatrix(W, V, W2, V2)),
        ("star", lambda: pc._redheffer_star(S, S)),
        ("inv", lambda: np.linalg.inv(A)),
        ("solve", lambda: np.linalg.solve(A, B)),
    ]
    r = {}
    # interleave: one pass over every function, repeated
    times = {k: [] for k, _ in fns}
    for _ in range(9):
        for k, f in fns:
            t = time.perf_counter()
            f()
            times[k].append(time.perf_counter() - t)
    for k in times:
        r[k] = min(times[k])
    print(f"{n:7d} {N2:5d} {r['modes']*1e3:15.2f} {r['eig']*1e3:13.2f} "
          f"{r['ifc']*1e3:11.2f} {r['star']*1e3:9.2f} {r['inv']*1e3:9.2f} "
          f"{r['solve']*1e3:10.2f}   (ms)")
    # A stack of L layers costs ~ L*modes + (L+1)*ifc + L*star (+2 uniform)
    for L in (8, 40):
        tot = L * r['modes'] + (L + 1) * r['ifc'] + L * r['star']
        eig_share = L * r['eig'] / tot
        print(f"          model {L:2d}-layer stack: total {tot*1e3:8.1f} ms, "
              f"bare-eig share {100*eig_share:5.1f}%, "
              f"modes {100*L*r['modes']/tot:5.1f}%, "
              f"ifc {100*(L+1)*r['ifc']/tot:5.1f}%, "
              f"star {100*L*r['star']/tot:5.1f}%")
