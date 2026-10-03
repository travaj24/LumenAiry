"""O-1 (pre-existing, NumPy): is the UNMAPPED route's incident decomposition
(minimum-norm least squares on an underdetermined Rayleigh system) invariant
under a per-mode rescaling of the geometric eigenvectors W0?  Emulated by
solving (H D) c' = delta and returning D c' (D a random complex diagonal).
Also: the round-off noise of T under a 1e-12 change of a rectangle's width.

    python o1_gauge.py          (run it in the PRE tree too: LUM_TREE=...)
"""
from _ve3 import WL, P, PMM2DStackPure, dump, np

import lumenairy.elements.pmm.stack2d_pure as SP
from lumenairy.elements.pmm import Rect

orig = SP._guarded_lstsq
D = None
SHAPES = {}


def scaled(A, b, site, hint=None):
    SHAPES[site] = list(np.shape(A))
    if D is None:
        return orig(A, b, site, hint)
    d = D[:A.shape[1]]
    return d * orig(A * d[None, :], b, site, hint)


SP._guarded_lstsq = scaled


def solve(M, th, phi=0.0, w=0.47):
    st = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=M, n_orders=2)
    st.add_layer(0.4, shapes=[Rect(0.6, 0.55, w, 0.42, 3.6)],
                 background_eps=1.0)
    st.set_source(WL, theta=th, phi=phi)
    o, R, T, J = st.solve()
    return np.concatenate([R.ravel(), T.ravel()])


rng = np.random.default_rng(3)
out = {"rows": []}
for M in (3, 4, 5):
    for th, ph in ((0.0, 0.0), (0.3, 0.0), (0.3, 0.45)):
        D = None
        a = solve(M, th, ph)
        dev = []
        for _k in range(3):
            D = np.exp(rng.normal(size=4000) * 0.7
                       + 1j * rng.uniform(0, 2 * np.pi, 4000))
            dev.append(float(np.max(np.abs(solve(M, th, ph) - a))))
        D = None
        noise = max(float(np.max(np.abs(solve(M, th, ph, 0.47 + k * 1e-12)
                                        - a))) for k in range(1, 6))
        row = {"M": M, "theta": th, "phi": ph, "rescaling_dev": max(dev),
               "noise_1e-12_width": noise, "lstsq_shapes": dict(SHAPES)}
        out["rows"].append(row)
        print(row, flush=True)
print(dump("o1_gauge.json", out))
