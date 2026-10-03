"""V3 (E3-2): forward parity on the VERIFIER's fixtures, with the eig-stage
round-off reach measured PER FIXTURE.

    python v3_parity.py M

Per fixture three solves: numpy (QZ), numpy_se (the same stack with the
pencil eig replaced by the standard eig of G^-1 L -- the twin's reduction --
the eig stage's own round-off reach on THIS fixture), and the twin (jit).
Recorded: max |d| over every order of R, T and over the Jones matrix for
twin-numpy, numpy_se-numpy, twin-numpy_se; the per-fixture ratio
max(twin-numpy) / max(numpy_se-numpy); the closure of the NumPy solve.
"""
import sys

import scipy.linalg as _sla
from _ve3 import TS, WL, P, PMM2DStackPure, amax, dump, jax, lc, np, tic

from lumenairy.elements.pmm import Circle, Ellipse, FilletRect, Rect, SinusoidalWall

M = int(sys.argv[1])
ONLY = set(sys.argv[2:])
MUG = np.array([[1.3, 0.2j, 0], [-0.2j, 1.3, 0], [0, 0, 1.1]], complex)
GY = np.array([[2.9, 0.35j, 0], [-0.35j, 2.9, 0], [0, 0, 2.4]], complex)


def F(layers, theta=0.0, phi=0.0, n_sub=1.52):
    def build(backend="numpy"):
        s = PMM2DStackPure(P, P, n_superstrate=1.0, n_substrate=n_sub,
                           n_modes=M, n_orders=2, backend=backend)
        for L in layers:
            s.add_layer(**L)
        s.set_source(WL, theta=theta, phi=phi)
        return s
    return build


def sl(t, shapes, bg=1.0, **kw):
    return dict(thickness=t, shapes=shapes, background_eps=bg, **kw)


cellm = np.array([[1.0, 2.6 + 0.1j], [3.1, 1.0]], complex)
FIX = {
    "ellipse_mag_lossy_conical": F([sl(0.42, [Ellipse(
        0.55, 0.6, 0.38, 0.24, 2.7 + 0.08j, mu=MUG)])], 0.35, 0.6),
    "ellipse_rot_mag_lossy_conical": F([sl(0.42, [Ellipse(
        0.55, 0.6, 0.33, 0.22, 2.7 + 0.08j, mu=MUG, angle=0.12)])],
        0.3, -0.5),
    "two_layer_merged_map": F([
        sl(0.3, [Circle(0.55, 0.6, 0.3, 3.3 + 0.05j)]),
        dict(thickness=0.15, eps=1.9),
        sl(0.2, [Rect(0.55, 0.6, 0.82, 0.78, 2.0)], 1.2)], 0.2, 0.1),
    "two_circles_5x5": F([sl(0.4, [Circle(0.32, 0.3, 0.16, 3.0),
                                   Circle(0.85, 0.85, 0.2, 2.2)])], 0.15),
    "circle_core_5x5_conical": F([sl(0.4, [Circle(0.55, 0.55, 0.42, 3.3,
                                                  core=0.4)])], 0.2, 0.7),
    "fillet_lc_conical": F([sl(0.35, [FilletRect(0.55, 0.53, 0.58, 0.44,
                                                 0.09, lc(0.4))], 1.7)],
                           0.25, 0.35),
    "sine_ridge_oblique": F([sl(0.4, [SinusoidalWall(
        "y", 0.3, 0.06, eps=2.4 + 0.02j, width=0.45, phase=0.4)])], 0.3),
    "circle_gyro_mag": F([sl(0.4, [Circle(0.6, 0.55, 0.33, GY,
                                          mu=1.4 + 0.0j)])], 0.2, 0.3),
    "multilayer_magnetic_tensor": F([
        dict(thickness=0.2, eps=lc(0.9)),
        dict(thickness=0.3, eps_cell=cellm, mu_cell=np.array(
            [[1.0, 1.0], [1.5 + 0.03j, 1.0]], complex)),
        dict(thickness=0.1, eps=2.2 + 0.01j)], 0.33, 0.25),
}


class _SE:
    def __getattr__(self, k):
        return getattr(_sla, k)

    @staticmethod
    def eig(a, b=None, **kw):
        if b is None:
            return _sla.eig(a, **kw)
        return np.linalg.eig(np.linalg.solve(b, a))


out = {"M": M, "fixtures": {}}
for name, build in FIX.items():
    if ONLY and name not in ONLY:
        continue
    t = tic()
    o, R, T, J = build().solve()
    t_np = tic() - t
    TS.sla = _SE()
    try:
        _o, R2, T2, J2 = build().solve()
    finally:
        TS.sla = _sla
    st = build("jax")
    tw = st.jax_twin()
    t = tic()
    Rj, Tj, Jj = (np.asarray(a) for a in jax.jit(
        lambda: tw.solve()[1:])())
    t_tw = tic() - t
    rec = {"twin_numpy": [amax(Rj, R), amax(Tj, T), amax(Jj, J)],
           "numpy_se_numpy": [amax(R2, R), amax(T2, T), amax(J2, J)],
           "twin_numpy_se": [amax(Rj, R2), amax(Tj, T2), amax(Jj, J2)],
           "closure": float(np.max(np.abs(R.sum(1) + T.sum(1) - 1.0))),
           "grid": [int(tw.Nx), int(tw.Ny)], "mapped": bool(tw.mapped),
           "t_numpy_s": t_np, "t_twin_jit_s": t_tw}
    rec["ratio_fixture"] = max(rec["twin_numpy"]) / max(
        rec["numpy_se_numpy"])
    rec["ratio_per_quantity"] = [a / b for a, b in zip(
        rec["twin_numpy"], rec["numpy_se_numpy"])]
    out["fixtures"][name] = rec
    print(f"{name:32s} grid {rec['grid']} twin-np {max(rec['twin_numpy']):.1e}"
          f" se-np {max(rec['numpy_se_numpy']):.1e} ratio "
          f"{rec['ratio_fixture']:.2f}", flush=True)
fx = out["fixtures"].values()
out["max_twin_numpy"] = max(max(v["twin_numpy"]) for v in fx)
out["max_ratio_fixture"] = max(v["ratio_fixture"] for v in fx)
out["max_ratio_quantity"] = max(max(v["ratio_per_quantity"]) for v in fx)
print(dump(f"v3_parity_M{M}.json", out))
