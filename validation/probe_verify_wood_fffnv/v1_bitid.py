"""V1 -- INDEPENDENT bit-identity battery for the Wood-list change (Task G.a).

Twelve fixtures of my own construction (NOT the builder's `g1_hash.py` set):
single-layer TE / TM / oblique / conical / lossy, rectangular periods, the
scalar->`e*I` promotion pair, an in-plane tensor control, and five
`PMM2DStackPure` stacks (uniform+patterned, A|B oblique, uniform-tensor mix,
lossy conical, scalar jones).  For every fixture it hashes the RAW float64
bytes of R, T and (where produced) the order-0 Jones, and records EVERY
`_grazing_safe_wavelength` call the solve made -- its input wavelength, its
returned wavelength and the length of the permittivity list it was handed.

Run the SAME file on the post-fix worktree and on the pre-fix main clone and
`diff` the JSON.

    python v1_bitid.py <lumenairy-root> <out.json>
"""
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

_ROOT = os.path.abspath(sys.argv[1])
assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), lumenairy.__file__

import lumenairy.elements.rcwa._core as _rc  # noqa: E402
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)

_CALLS = []
_REAL_GUARD = _rc._grazing_safe_wavelength


def _spy(wavelength, kx0, ky0, m_orders, n_orders, px, py, eps_reals,
         max_iter=8):
    out = _REAL_GUARD(wavelength, kx0, ky0, m_orders, n_orders, px, py,
                      eps_reals, max_iter=max_iter)
    _CALLS.append({"in": repr(float(wavelength)), "out": repr(float(out)),
                   "n_eps": len(list(eps_reals))})
    return out


_rc._grazing_safe_wavelength = _spy


def _h(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:24]


def promote(m):
    m = np.asarray(m, dtype=complex)
    t = np.zeros(m.shape + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


def rot(deg, no, ne):
    """In-plane rotated uniaxial director as a BLOCK-FORM (3, 3) tensor."""
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R @ d @ R.T


# --- my fixtures ----------------------------------------------------------
# a 3x3 scalar cell: an L-shaped pillar (deliberately NOT a 2x2 half-fill)
CELL3 = np.array([[6.25, 6.25, 1.0],
                  [6.25, 2.10, 1.0],
                  [1.00, 1.00, 1.0]], dtype=complex)
CELL2 = np.array([[5.0, 1.0], [1.0, 3.0]], dtype=complex)
CELL_LOSSY = np.array([[6.25 + 0.35j, 1.0], [1.0, 2.5 + 0.05j]], dtype=complex)

PX, PY = 0.62e-6, 0.62e-6
WL = 0.85e-6
D = 0.31e-6
M, NO = 5, 3

RES = {}


def rec(name, R, T, J=None, extra=None):
    _CALLS.clear()
    return None


def run(name, fn):
    _CALLS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = fn()
    RES[name] = dict(out)
    RES[name]["guard_calls"] = list(_CALLS)


def f_eff(pol, theta, phi, cell=CELL3, px=PX, py=PY, wl=WL):
    def go():
        _o, R, T = pmm_efficiency_2d_staggered(
            px, py, cell, 1.45, 1.0, D, wl, degree=M, n_orders=NO,
            polarization=pol, theta=theta, phi=phi)
        return {"R": _h(R), "T": _h(T)}
    return go


def f_jones(cell, theta=0.0, phi=0.0, px=PX, py=PY, wl=WL):
    def go():
        _o, R, T, J = pmm_jones_2d_staggered(
            px, py, cell, 1.45, 1.0, D, wl, degree=M, n_orders=NO,
            theta=theta, phi=phi)
        return {"R": _h(R), "T": _h(T), "J": _h(J)}
    return go


def f_stack(build, jones, theta=0.0, phi=0.0, wl=WL):
    def go():
        st = PMM2DStackPure(PX, PY, n_superstrate=1.0, n_substrate=1.45,
                            n_modes=M, n_orders=NO)
        build(st)
        st.set_source(wl, theta=theta, phi=phi)
        if jones:
            _o, R, T, J = st.solve(jones=True)
            return {"R": _h(R), "T": _h(T), "J": _h(J)}
        _o, R, T = st.solve(jones=False)
        return {"R": _h(R), "T": _h(T)}
    return go


run("f01_eff_te_normal", f_eff("te", 0.0, 0.0))
run("f02_eff_tm_oblique", f_eff("tm", 0.27, 0.0))
run("f03_eff_conical", f_eff("te", 0.18, 0.77))
run("f04_eff_lossy_tm", f_eff("tm", 0.12, 0.0, cell=CELL_LOSSY))
run("f05_eff_rect_periods", f_eff("te", 0.0, 0.0, cell=CELL2,
                                  px=0.62e-6, py=0.47e-6))
run("f06_jones_promoted_cell3", f_jones(promote(CELL3)))
run("f07_jones_tensor_rot40", f_jones(np.array(
    [[rot(40.0, 1.5, 2.3), np.diag([2.10] * 3).astype(complex)],
     [np.diag([2.10] * 3).astype(complex), rot(40.0, 1.5, 2.3)]],
    dtype=complex)))
run("f08_jones_promoted_oblique", f_jones(promote(CELL2), theta=0.21,
                                          phi=0.44))
run("f09_stack_uniform_patterned",
    f_stack(lambda s: (s.add_layer(0.14e-6, eps=2.10),
                       s.add_layer(D, eps_cell=CELL3)), jones=False))
run("f10_stack_ab_oblique",
    f_stack(lambda s: (s.add_layer(0.19e-6, eps_cell=CELL3),
                       s.add_layer(0.13e-6, eps_cell=CELL2 * np.ones((3, 3)) if
                                   False else np.array(
                                       [[3.0, 1.0, 3.0],
                                        [1.0, 3.0, 1.0],
                                        [3.0, 1.0, 3.0]], dtype=complex))),
            jones=True, theta=0.20, phi=0.50))
run("f11_stack_uniform_tensor_mix",
    f_stack(lambda s: (s.add_layer(0.12e-6, eps=rot(40.0, 1.5, 2.3)),
                       s.add_layer(0.18e-6, eps_cell=CELL3)), jones=True))
run("f12_stack_lossy_conical",
    f_stack(lambda s: (s.add_layer(0.09e-6, eps=2.10),
                       s.add_layer(0.16e-6, eps_cell=np.array(
                           [[6.25 + 0.35j, 1.0, 1.0],
                            [1.0, 2.5 + 0.05j, 1.0],
                            [1.0, 1.0, 1.0]], dtype=complex))),
            jones=True, theta=0.23, phi=0.91))
run("f13_stack_scalar_jones",
    f_stack(lambda s: s.add_layer(D, eps_cell=CELL3), jones=True))
run("f14_stack_promoted_jones",
    f_stack(lambda s: s.add_layer(D, eps_cell=promote(CELL3)), jones=True))

out_path = sys.argv[2]
with open(out_path, "w", encoding="cp1252") as fh:
    json.dump(RES, fh, indent=1, sort_keys=True)
for k in sorted(RES):
    v = RES[k]
    print(f"{k:32s} R={v['R']} T={v['T']} "
          f"J={v.get('J', '-'):24s} guard={v['guard_calls']}")
