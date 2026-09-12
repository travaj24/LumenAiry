"""PROBE 3c: the 1-D MORTAR (per-layer element grids).

(i) identical-grid reduction: _interface_smatrix_mortar must equal
    _interface_smatrix algebraically;
(ii) non-conforming convergence: per-layer answer -> shared-grid answer as
     degree rises;
(iii) energy on the per-layer path.
"""
import sys, warnings
import numpy as np
sys.path.insert(0, r"D:/Metacept/Neurophos/Python_Test_Scripts/Free_Space_Optics/Lumenairy")
from lumenairy.elements.pmm import _core as pc
from lumenairy.elements.pmm import PMMStack

print("=== 3c-i: identical grids -> mortar == plain interface ===")
per, wl = 1.0e-6, 1.55e-6
k0 = 2 * np.pi / wl
kx0 = np.sin(np.deg2rad(13.0)) * k0
t = lambda e: pc._tensor3_dict(e * np.eye(3))
w = [0.37, 0.63]
ma = pc._build_sem_tensor_segments(per, w, [t(3.48**2), t(1.444**2)], 12, 1,
                                   True)
mb = pc._build_sem_tensor_segments(per, w, [t(2.0**2), t(1.0)], 12, 1, True)
Wa, Va, la, qa = pc._sem_modes_tensor(ma, k0, kx0)
Wb, Vb, lb, qb = pc._sem_modes_tensor(mb, k0, kx0)
Ma = pc._sem_mass_exact(ma)
Mb = pc._sem_mass_exact(mb)
Cab = pc._sem_cross_mass(ma, mb)
print("  |Ma - Mb|max      =", np.max(np.abs(Ma - Mb)))
print("  |Cab - Ma|max     =", np.max(np.abs(Cab - Ma)))
S_plain = pc._interface_smatrix(Wa, Va, Wb, Vb)
S_mort = pc._interface_smatrix_mortar(Wa, Va, Wb, Vb, Ma, Mb, Cab)
for i, nm in enumerate(("S11", "S12", "S21", "S22")):
    d = np.max(np.abs(S_plain[i] - S_mort[i]))
    r = d / max(np.max(np.abs(S_plain[i])), 1e-300)
    print(f"  {nm}: max|d| = {d:.3e}  rel = {r:.3e}")

print()
print("=== 3c-ii: per-layer vs shared, 3-layer stack, degree ladder ===")
eps = [3.48 ** 2, 2.0 ** 2, 1.444 ** 2]


def stack(mode, degree, hw=1):
    st = PMMStack(per, n_substrate=1.444, n_superstrate=1.0, degree=degree,
                  far_field_orders=11, layer_grids=mode,
                  **({"window_halfwidth": hw} if mode == "per-layer" else {}))
    st.add_layer(0.2e-6, segments=[(0.31, eps[0]), (0.69, eps[2])])
    st.add_layer(0.2e-6, segments=[(0.47, eps[1]), (0.53, eps[2])])
    st.add_layer(0.2e-6, segments=[(0.63, eps[0]), (0.37, eps[1])])
    st.set_source(wl, angle=np.deg2rad(13.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve()


for degree in (10, 14, 18, 24, 30):
    o1, R1, T1, J1 = stack("shared", degree)
    o2, R2, T2, J2 = stack("per-layer", degree)
    k = min(len(o1), len(o2))
    ia, ib = (len(o1) - k) // 2, (len(o2) - k) // 2
    dR = np.max(np.abs(R1[:, ia:ia + k] - R2[:, ib:ib + k]))
    dT = np.max(np.abs(T1[:, ia:ia + k] - T2[:, ib:ib + k]))
    print(f"  deg={degree:3d}: max|dR|={dR:.3e} max|dT|={dT:.3e} "
          f"max|dJ|={np.max(np.abs(J1-J2)):.3e}  "
          f"tot_shared={R1.sum(1)+T1.sum(1)}  tot_pl={R2.sum(1)+T2.sum(1)}")

print()
print("=== 3c-iii: window_halfwidth >= nlayers-1 must reproduce SHARED ===")
for hw in (1, 2, 3, 4):
    o1, R1, T1, J1 = stack("shared", 18)
    o2, R2, T2, J2 = stack("per-layer", 18, hw)
    print(f"  hw={hw}: max|dJ| vs shared = {np.max(np.abs(J1-J2)):.3e}")
