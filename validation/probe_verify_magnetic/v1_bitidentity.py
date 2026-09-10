"""V1 -- BIT-IDENTITY of the NONMAGNETIC paths (scalar / in-plane tensor /
out-of-plane) between the MAIN clone (fb3fd93) and the verify worktree
(5c384e5 = the magnetic build merged with the Wood-anomaly-list fix).

Run one ARM per interpreter and diff the two JSON dumps::

    python v1_bitidentity.py main > main.json      # PYTHONPATH=D:/.../Lumenairy
    python v1_bitidentity.py wt   > wt.json        # PYTHONPATH=/c/tmp/lum_vmag
    python v1_bitidentity.py diff main.json wt.json

Every quantity is hashed from the RAW BYTES of a C-contiguous complex128 (or
float64) array, so the comparison is EXACT -- not a tolerance.
"""
import hashlib
import json
import sys

import numpy as np

import lumenairy
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    Granet2DTransverseE,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)

_ARMS = {"main": "D:", "wt": "C:\\tmp\\lum_vmag"}


def _h(*arrays):
    m = hashlib.sha256()
    for a in arrays:
        if a is None:
            m.update(b"<None>")
            continue
        arr = np.ascontiguousarray(a)
        m.update(str(arr.shape).encode())
        m.update(str(arr.dtype).encode())
        m.update(arr.tobytes())
    return m.hexdigest()[:32]


# ------------------------------------------------------------------ fixtures
def _lc(no=1.50, ne=1.72, psi=0.4):
    """A rotated (in-plane) uniaxial tensor -- block form, e12 == e21."""
    c, s = np.cos(psi), np.sin(psi)
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rz @ d @ rz.T


def _tilt(no=1.50, ne=1.72, psi=0.35, tilt=0.30):
    """A TILTED director -- carries e13 / e31 (the out-of-plane path)."""
    d = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    cz, sz = np.cos(psi), np.sin(psi)
    cy, sy = np.cos(tilt), np.sin(tilt)
    rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    q = rz @ ry
    return q @ d @ q.T


def _scalar_cell(n=2):
    c = np.full((n, n), 2.10 + 0.0j)
    c[0, 0] = 6.25 + 0.15j
    if n == 3:
        c[1, 2] = 4.0
        c[2, 1] = 3.1 + 0.02j
    return c


def _tensor_cell(n=2):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[...] = np.eye(3) * (2.10 + 0.0j)
    c[0, 0] = _lc()
    if n == 3:
        c[1, 2] = np.eye(3) * (4.0 + 0.05j)
    return c


def _oop_cell(n=2):
    c = np.zeros((n, n, 3, 3), dtype=complex)
    c[...] = np.eye(3) * (2.10 + 0.0j)
    c[0, 0] = _tilt()
    return c


def _solver_ops(sol):
    return {
        "Lmat": _h(sol.Lmat), "Rmat": _h(sol.Rmat), "Stt": _h(sol.Stt),
        "Schur": _h(sol.Schur),
        "Et": _h(*(sol.Et_blocks if sol.Et_blocks is not None else (None,))),
        "Et_off": _h(*(sol.Et_offdiag if sol.Et_offdiag is not None
                       else (None,))),
        "Agen": _h(getattr(sol, "Agen", None)),
        "Bgen": _h(getattr(sol, "Bgen", None)),
        "dimtot": int(sol.dimtot),
    }


def collect():
    out = {}
    k0 = 2.0 * np.pi / 0.55e-6
    px = py = 0.90e-6

    # F1-F3 -- the retained OPERATORS, straight off the assembler
    for tag, cell, m in (("F1_scalar_ops", _scalar_cell(2), 6),
                         ("F2_tensor_ops", _tensor_cell(2), 6),
                         ("F3_oop_ops", _oop_cell(2), 5)):
        sol = Granet2DTransverseE(px, py, cell.shape[0], cell.shape[1], m, cell,
                                  alpha0x=0.31 * k0, alpha0y=0.17 * k0, k0=k0)
        out[tag] = _solver_ops(sol)
        out[tag]["magnetic"] = bool(getattr(sol, "magnetic", False))
        out[tag]["Ggram"] = _h(*(getattr(sol, "Ggram_blocks", None)
                                 or (None,)))

    # F4-F7 -- the public single-layer Jones entry
    jobs = [
        ("F4_jones_scalar_normal", _scalar_cell(2), 6, 0.0, 0.0),
        ("F5_jones_scalar_oblique", _scalar_cell(3), 5, 0.30, 0.0),
        ("F6_jones_tensor_conical", _tensor_cell(2), 6, 0.30, 0.70),
        ("F7_jones_oop_conical", _oop_cell(2), 5, 0.20, 0.40),
    ]
    for tag, cell, m, th, ph in jobs:
        o, r, t, j = pmm_jones_2d_staggered(px, py, cell, 1.45, 1.0, 0.30e-6,
                                            0.55e-6, n_modes=m, n_orders=5,
                                            theta=th, phi=ph)
        out[tag] = {"orders": _h(np.asarray(o)), "R": _h(r), "T": _h(t),
                    "J": _h(j), "sumR": repr(float(np.sum(r))),
                    "sumT": repr(float(np.sum(t)))}

    # F8 -- the SCALAR efficiency entry (TE + TM), the other Wood call site
    for pol in ("te", "tm"):
        eo, er, et = pmm_efficiency_2d_staggered(
            px, py, _scalar_cell(2), 1.45, 1.0, 0.30e-6, 0.55e-6, n_modes=6,
            n_orders=5, polarization=pol, theta=0.30, phi=0.25)[:3]
        out["F8_eff_" + pol] = {
            "R": _h(np.asarray(er)), "T": _h(np.asarray(et)),
            "orders": _h(np.asarray(eo)),
        }

    # F9 -- a 3-layer PURE stack (uniform scalar | patterned scalar | uniform
    #       tensor): the in-plane cascade + the Wood list's uniform/patterned
    #       scalar and uniform-tensor branches, all in one solve
    st = PMM2DStackPure(px, py, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=5, n_orders=5)
    st.add_layer(0.12e-6, eps=2.25)
    st.add_layer(0.20e-6, eps_cell=_scalar_cell(2))
    st.add_layer(0.10e-6, eps=_lc())
    st.set_source(0.55e-6, theta=0.30, phi=0.70)
    o, r, t, j = st.solve(jones=True)
    out["F9_stack_inplane"] = {"orders": _h(np.asarray(o)), "R": _h(r),
                               "T": _h(t), "J": _h(j)}

    # F10 -- a GENERALIZED cascade (one out-of-plane layer + one in-plane)
    st = PMM2DStackPure(px, py, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=5, n_orders=5)
    st.add_layer(0.15e-6, eps_cell=_oop_cell(2))
    st.add_layer(0.15e-6, eps_cell=_tensor_cell(2))
    st.set_source(0.55e-6, theta=0.20, phi=0.40)
    o, r, t, j = st.solve(jones=True)
    out["F10_stack_generalized"] = {"orders": _h(np.asarray(o)), "R": _h(r),
                                    "T": _h(t), "J": _h(j)}

    # F11 -- retain_internal + layer_absorption (a lossy scalar cell)
    st = PMM2DStackPure(px, py, n_superstrate=1.0, n_substrate=1.45,
                        n_modes=5, n_orders=5)
    st.add_layer(0.20e-6, eps_cell=_scalar_cell(2))
    st.add_layer(0.10e-6, eps=2.25)
    st.set_source(0.55e-6, theta=0.20, phi=0.40)
    o, r, t, j = st.solve(jones=True, retain_internal=True)
    a = st.layer_absorption()
    out["F11_absorption"] = {"R": _h(r), "T": _h(t), "J": _h(j),
                             "A": _h(np.asarray(a))}
    return out


def main():
    if sys.argv[1] == "diff":
        with open(sys.argv[2]) as fh:
            a = json.load(fh)
        with open(sys.argv[3]) as fh:
            b = json.load(fh)
        bad = 0
        for k in sorted(set(a) | set(b)):
            fa, fb = a.get(k, {}), b.get(k, {})
            for f in sorted(set(fa) | set(fb)):
                if fa.get(f) != fb.get(f):
                    bad += 1
                    print(f"DIFF {k}.{f}: {fa.get(f)!r} != {fb.get(f)!r}")
        n = sum(len(v) for v in a.values())
        print(f"{len(a)} fixtures, {n} hashed quantities, {bad} differences")
        print("BIT-IDENTICAL" if bad == 0 else "NOT IDENTICAL")
        return
    arm = sys.argv[1]
    assert lumenairy.__file__.startswith(_ARMS[arm]), (
        f"arm {arm}: lumenairy is {lumenairy.__file__}")
    print(json.dumps(collect(), indent=1))


if __name__ == "__main__":
    main()
