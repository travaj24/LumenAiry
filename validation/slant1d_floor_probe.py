import time
import numpy as np
import lumenairy
assert 'lum_sl' in lumenairy.__file__, lumenairy.__file__
from lumenairy.elements.pmm import PMMStack

P, WL, D = 1.0e-6, 633e-9, 300e-9
EPSR, EPSG, DUTY, THETA = 4.0+0j, 1.0+0j, 0.45, 0.17

def vec(res):
    o, R, T, J = res
    keep = np.abs(o) <= 2
    return np.concatenate([np.asarray(R)[:, keep].ravel(),
                           np.asarray(T)[:, keep].ravel()]), (R, T)

def sheared(shear, degree):
    s = PMMStack(P, n_substrate=1.5, degree=degree)
    s.add_sheared_grating(D, eps_ridge=EPSR, eps_groove=EPSG, duty=DUTY, shear=shear)
    s.set_source(WL, theta=THETA); return s.solve()

def stair(shear, degree, ns):
    s = PMMStack(P, n_substrate=1.5, degree=degree)
    s.add_tapered_grating(D, eps_ridge=EPSR, eps_groove=EPSG, duty_bottom=DUTY,
                          duty_top=DUTY, n_slices=ns, shear=shear)
    s.set_source(WL, theta=THETA); return s.solve()

for shear in (0.05, 0.10, 0.20, 0.35):
    tilt = np.degrees(np.arctan(shear * P / D))
    ref, _ = vec(stair(shear, 10, 20))
    parts = []
    for deg in (8, 12, 16, 20):
        t0 = time.time(); res = sheared(shear, deg); dt = time.time() - t0
        v, (R, T) = vec(res)
        e = np.max(np.abs(v - ref))
        clos = np.max(np.abs(np.asarray(R).sum(axis=1) + np.asarray(T).sum(axis=1) - 1.0))
        parts.append(f"d{deg}: {e:.2e}/{clos:.1e}/{dt:.2f}s")
    print(f"shear={shear:<5} tilt={tilt:5.1f}deg  " + "  ".join(parts), flush=True)
