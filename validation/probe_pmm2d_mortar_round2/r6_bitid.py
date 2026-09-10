"""R6 -- BIT-IDENTITY of the round-2 changes against the PRE-CHANGE library.

``R6_TAG=with  PYTHONPATH=<this tree>   python r6_bitid.py``
``R6_TAG=pre   PYTHONPATH=<pristine>    R6_EXPECT_ROOT=<pristine> python r6_bitid.py``
``python r6_compare.py with pre``

ONE file, run twice against two trees, hashing raw IEEE bytes (dtype + shape +
``tobytes``).  Every arm asserts which ``lumenairy`` it imported.

Two families:

* **PROJECTOR** -- ``_stag_fourier_projection`` on INTEGER-``N`` grids, which
  D3's per-segment quadrature order must leave untouched bit for bit, and on
  explicitly-passed UNIFORM ARRAYS, where the rule must also return the shipped
  ``2 M + 8`` (that is what keeps gate N2's ULP claim true).
* **STACK** -- solves drawn from every shipped suite this branch can reach:
  the shared union path, per-layer conforming / nested / non-conforming, the
  taper, slant, out-of-plane, magnetic, tensor, and the 1-D ``PMMStack``.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import hashlib
import json
import sys
import time
import warnings

import numpy as np

import lumenairy
from lumenairy.elements.pmm import PMM2DStackPure, PMMStack
from lumenairy.elements.pmm.twod_staggered import (
    Basis1D,
    _stag_fourier_projection,
)

HERE = os.path.dirname(os.path.abspath(__file__))
_EXPECT = os.environ.get("R6_EXPECT_ROOT")
if _EXPECT:
    assert os.path.abspath(lumenairy.__file__).startswith(
        os.path.abspath(_EXPECT)), lumenairy.__file__
else:
    _ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
    assert os.path.abspath(lumenairy.__file__).startswith(_ROOT), \
        lumenairy.__file__
print(f"[arm] lumenairy = {lumenairy.__file__}", flush=True)

_C = complex
H = {}
T0 = time.time()


def _log(m):
    print(f"[{time.time() - T0:7.1f}s] {m}", flush=True)


def _h(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(
        (str(a.dtype) + str(a.shape)).encode() + a.tobytes()).hexdigest()


def _rec(name, *arrays):
    H[name] = [_h(a) for a in arrays]


# ------------------------------------------------------------- projectors
def sec_proj():
    for d, N, M, tau, a0, mo in (
            (1.2, 3, 5, 1.0 + 0.0j, 0.0, 4),
            (0.9, 4, 4, np.exp(-0.41j), 2.08, 5),
            (1.4, 6, 6, np.exp(0.77j), -3.1, 7),
            (0.7, 2, 8, np.exp(1.13j), 1.7, 6),
            (1.0, 12, 3, np.exp(-0.2j), 0.4, 8),
            (1.0, 1, 7, np.exp(-0.9j), 5.5, 2)):
        b = Basis1D(d, N, M, tau)
        orders = np.arange(-mo, mo + 1)
        asm = _stag_fourier_projection(b, orders, a0)
        _rec(f"projINT_d{d}_N{N}_M{M}_m{mo}", asm(b.B), asm(b.Btilde))
        # the same lattice spelled as an explicit ARRAY
        ba = Basis1D(d, np.linspace(0.0, d, N + 1), M, tau)
        asa = _stag_fourier_projection(ba, orders, a0)
        _rec(f"projARR_d{d}_N{N}_M{M}_m{mo}", asa(ba.B), asa(ba.Btilde))
    _log(f"projectors: {len([k for k in H if k.startswith('proj')])} entries")


# ------------------------------------------------------------------ stacks
def _s(st, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return st.solve(**kw)


def _tile(n, ep, eh):
    t = np.full((n, n), _C(eh))
    t[n // 2, n // 2] = _C(ep)
    return t


def sec_stack():
    P, WL = 1.2, 0.85
    EP, EH = 9.0, 2.25
    tl = _tile(3, EP, EH)

    # --- 1. SHARED union path (must be untouched) ---------------------------
    for name, (th, ph, M, N) in {
            "shared_normal": (0.0, 0.0, 5, 4),
            "shared_conical": (0.21, 0.37, 5, 4),
            "shared_M6N3": (0.15, 0.35, 6, 3)}.items():
        st = PMM2DStackPure(P, n_modes=M, n_orders=1)
        st.add_layer(0.20, eps_cell=_tile(N, EP, EH))
        st.add_layer(0.10, eps=EH)
        st.set_source(WL, theta=th, phi=ph)
        _rec(name, *_s(st, jones=False))

    # --- 2. PER-LAYER conforming / nested / non-conforming ------------------
    def _pl(layers, M=5, th=0.15, ph=0.35, n_orders=1):
        st = PMM2DStackPure(P, n_modes=M, n_orders=n_orders,
                            layer_grids="per-layer")
        for t, cell, xw, yw in layers:
            st.add_layer(t, eps_cell=cell, x_walls=xw, y_walls=yw)
        st.set_source(WL, theta=th, phi=ph)
        return st

    w1 = [0.2371 * P, 0.6183 * P]
    w2 = [0.3117 * P, 0.7402 * P]
    _rec("pl_conforming", *_s(_pl([(0.10, tl, w1, w1), (0.10, tl, w1, w1)]),
                              jones=False))
    _rec("pl_nonconf", *_s(_pl([(0.10, tl, w1, w1), (0.10, tl, w2, w2)]),
                           jones=False))
    _rec("pl_nested", *_s(_pl([
        (0.10, _tile(3, EP, EH), [0.25 * P, 0.75 * P], [0.25 * P, 0.75 * P]),
        (0.10, np.full((5, 5), _C(EH)),
         [0.125 * P, 0.25 * P, 0.75 * P, 0.875 * P],
         [0.125 * P, 0.25 * P, 0.75 * P, 0.875 * P])]), jones=False))
    _rec("pl_normal", *_s(_pl([(0.10, tl, w1, w1), (0.10, tl, w2, w2)],
                              th=0.0, ph=0.0), jones=False))
    _rec("pl_M6", *_s(_pl([(0.10, tl, w1, w1), (0.10, tl, w2, w2)], M=6),
                      jones=False))
    _rec("pl_jones", *_s(_pl([(0.10, tl, w1, w1), (0.10, tl, w2, w2)])))
    # mixed uniform / patterned / grid-int layers
    st = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.08, eps=EH, grid=2)
    st.add_layer(0.10, eps_cell=tl, x_walls=w1, y_walls=w1)
    st.add_layer(0.08, eps_cell=_tile(4, EP, EH))
    st.set_source(WL, theta=0.15, phi=0.35)
    _rec("pl_mixed", *_s(st, jones=False))

    # --- 3. the TAPER (both builders) ---------------------------------------
    st = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_tapered_pillar(0.24, eps_pillar=EP, eps_host=EH,
                          x_bounds_bottom=[0.1873 * P, 0.7241 * P],
                          y_bounds_bottom=[0.1873 * P, 0.7241 * P],
                          x_bounds_top=[0.2917 * P, 0.6109 * P],
                          y_bounds_top=[0.2917 * P, 0.6109 * P], n_slices=4)
    st.set_source(WL, theta=0.15, phi=0.35)
    _rec("taper_pillar", *_s(st, jones=False))
    st = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_tapered_pillars(
        0.20, eps_host=EH, n_slices=3,
        pillars=[((0.3 * P, 0.3 * P), (0.18 * P, 0.18 * P),
                  (0.26 * P, 0.26 * P), 9.0)])
    st.set_source(WL, theta=0.15, phi=0.35)
    _rec("taper_pillars", *_s(st, jones=False))

    # --- 4. SLANT, OOP tensor, MAGNETIC, in-plane tensor ---------------------
    st = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.12, eps_cell=tl, x_walls=w1, y_walls=w1,
                 slant=(0.20, 0.10))
    st.add_layer(0.12, eps_cell=tl, x_walls=w2, y_walls=w2,
                 slant=(0.20, 0.10))
    st.set_source(WL, theta=0.12, phi=0.30)
    _rec("pl_slant", *_s(st, jones=False))

    t33 = np.zeros((3, 3, 3, 3), dtype=_C)
    for i in range(3):
        for j in range(3):
            t33[i, j] = np.diag([EH, EH, EH])
    t33[1, 1] = np.array([[EP, 0.4, 0.0], [0.4, EP, 0.0], [0.0, 0.0, EP]],
                         dtype=_C)
    st = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.12, eps_cell=t33, x_walls=w1, y_walls=w1)
    st.add_layer(0.10, eps_cell=tl, x_walls=w2, y_walls=w2)
    st.set_source(WL, theta=0.15, phi=0.35)
    _rec("pl_tensor_inplane", *_s(st, jones=False))

    o33 = t33.copy()
    o33[1, 1] = np.array([[EP, 0.0, 0.5], [0.0, EP, 0.0], [0.5, 0.0, EP]],
                         dtype=_C)
    st = PMM2DStackPure(P, n_modes=4, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.12, eps_cell=o33, x_walls=w1, y_walls=w1)
    st.add_layer(0.10, eps_cell=tl, x_walls=w2, y_walls=w2)
    st.set_source(WL, theta=0.15, phi=0.35)
    _rec("pl_tensor_oop", *_s(st, jones=False))

    mu = np.full((3, 3), _C(1.0))
    mu[1, 1] = _C(1.4)
    st = PMM2DStackPure(P, n_modes=5, n_orders=1, layer_grids="per-layer")
    st.add_layer(0.12, eps_cell=tl, mu_cell=mu, x_walls=w1, y_walls=w1)
    st.add_layer(0.10, eps_cell=tl, x_walls=w2, y_walls=w2)
    st.set_source(WL, theta=0.15, phi=0.35)
    _rec("pl_magnetic", *_s(st, jones=False))

    # --- 5. retain_internal / layer_absorption / per-order -------------------
    st = _pl([(0.10, tl, w1, w1), (0.10, tl, w2, w2)])
    o, R, T, J = _s(st, retain_internal=True)
    _rec("pl_retain", o, R, T, J)
    _rec("pl_absorption", np.asarray(st.layer_absorption()))

    # --- 6. the 1-D PMMStack (the ~1850 interface's own fixtures) -----------
    for name, kw in (("st1d_shared", {}),
                     ("st1d_perlayer", {"layer_grids": "per-layer"})):
        st = PMMStack(P, degree=8, far_field_orders=5, **kw)
        st.add_layer(0.15, segments=[(0.3, EH), (0.35, EP), (0.35, EH)])
        st.add_layer(0.12, segments=[(0.45, EP), (0.55, EH)])
        st.set_source(WL, theta=0.18)
        _rec(name, *_s(st)[:3])
    st = PMMStack(P, degree=8, far_field_orders=5)
    st.add_layer(0.15, segments=[(0.3, EH), (0.35, EP), (0.35, EH)])
    st.set_source(WL, theta=0.18, phi=0.4)
    _rec("st1d_conical", *_s(st)[:3])
    _log(f"stacks: {len([k for k in H if not k.startswith('proj')])} entries")


SECTIONS = {"proj": sec_proj, "stack": sec_stack}


def main():
    for w in (sys.argv[1:] or list(SECTIONS)):
        SECTIONS[w]()
    tag = os.environ.get("R6_TAG", "with")
    path = os.path.join(HERE, f"r6_bitid_{tag}.json")
    with open(path, "w") as fh:
        json.dump(H, fh, indent=1, sort_keys=True)
    _log(f"wrote {path} ({len(H)} entries)")


if __name__ == "__main__":
    main()
