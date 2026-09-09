"""Gate 1 probe -- hash R/T + the nudged wavelength for scalar staggered
fixtures.  Run with argv[1] = the lumenairy root the arm MUST come from."""
import hashlib
import json
import os
import sys
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np  # noqa: E402

import lumenairy  # noqa: E402

_ROOT = os.path.abspath(sys.argv[1]).replace("\\", "/").lower()
_HERE = os.path.abspath(lumenairy.__file__).replace("\\", "/").lower()
assert _HERE.startswith(_ROOT), (_HERE, _ROOT)

import lumenairy.elements.rcwa._core as _rc  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)

_ORIG = _rc._grazing_safe_wavelength
_SEEN = []


def _spy(wavelength, *a, **kw):
    wl = _ORIG(wavelength, *a, **kw)
    _SEEN.append((float(wavelength), float(wl)))
    return wl


_rc._grazing_safe_wavelength = _spy


def _h(x):
    a = np.ascontiguousarray(np.asarray(x, dtype=float))
    return hashlib.sha256(a.tobytes()).hexdigest()[:24]


PILLAR = np.array([[2.25, 1.0], [1.0, 1.0]], dtype=complex)
LOSSY = np.array([[2.25 + 0.1j, 1.0], [1.0, 4.0 + 0.3j]], dtype=complex)
CELL_B = np.array([[1.0, 3.0], [3.0, 1.0]], dtype=complex)
G = dict(period_x=0.8e-6, period_y=0.8e-6, depth=0.3e-6, wavelength=0.633e-6)


def promote(m):
    t = np.zeros(np.shape(m) + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


def eff(**kw):
    k = dict(G)
    k.update(kw)
    o, R, T = pmm_efficiency_2d_staggered(
        n_substrate=1.45, n_superstrate=1.0, degree=5, n_orders=3, **k)
    return R, T


def stack(layers, *, theta=0.0, phi=0.0, jones=False, wl=0.633e-6, M=5,
          n_orders=3, n_sup=1.0, n_sub=1.45, px=0.8e-6):
    s = PMM2DStackPure(px, px, n_superstrate=n_sup, n_substrate=n_sub,
                       n_modes=M, n_orders=n_orders)
    for t, kind, v in layers:
        if kind == "u":
            s.add_layer(t, eps=v)
        else:
            s.add_layer(t, eps_cell=v)
    s.set_source(wl, theta=theta, phi=phi)
    out = s.solve(jones=jones)
    return out[1], out[2]


FIX = {}
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    FIX["eff_te_normal"] = eff(eps_cell=PILLAR, polarization="te")
    FIX["eff_tm_oblique"] = eff(eps_cell=PILLAR, polarization="tm", theta=0.2)
    FIX["eff_conical"] = eff(eps_cell=PILLAR, polarization="te", theta=0.25,
                             phi=0.4)
    FIX["eff_lossy"] = eff(eps_cell=LOSSY, polarization="tm", theta=0.1)
    FIX["eff_uniform_cell"] = eff(eps_cell=np.full((2, 2), 4.0 + 0j),
                                  polarization="te")
    FIX["stack_u_p"] = stack([(0.2e-6, "u", 2.1), (0.3e-6, "c", PILLAR)])
    FIX["stack_ab_oblique"] = stack(
        [(0.2e-6, "c", PILLAR), (0.25e-6, "c", CELL_B)], theta=0.15, phi=0.3,
        jones=True)
    FIX["stack_scalar_jones"] = stack([(0.3e-6, "c", PILLAR)], jones=True)
    FIX["stack_lossy_conical"] = stack([(0.22e-6, "c", LOSSY)], theta=0.22,
                                       phi=0.9, jones=True)
    # tensor CONTROL (must not move): real in-plane tensor cell
    _t = promote(PILLAR)
    _t[0, 0, 0, 1] = _t[0, 0, 1, 0] = 0.3
    _o, _R, _T, _J = pmm_jones_2d_staggered(
        0.8e-6, 0.8e-6, _t, 1.45, 1.0, 0.3e-6, 0.633e-6, degree=5, n_orders=3)
    FIX["jones_tensor_ctrl"] = (_R, _T)
    # tensor promotion of a scalar cell (the G1 identity partner)
    _o, _R, _T, _J = pmm_jones_2d_staggered(
        0.8e-6, 0.8e-6, promote(PILLAR), 1.45, 1.0, 0.3e-6, 0.633e-6,
        degree=5, n_orders=3)
    FIX["jones_promoted_pillar"] = (_R, _T)

out = {k: dict(R=_h(v[0]), T=_h(v[1])) for k, v in FIX.items()}
out["_wl_nudges"] = ["%.17e->%.17e" % p for p in _SEEN]
print(json.dumps(out, indent=1, sort_keys=True))
