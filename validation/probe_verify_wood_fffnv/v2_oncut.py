"""V2 -- the ON-CUT-OFF reproducers, mine, derived through the public API
(Task G.b, G.c, G.e).

Three cut-off fixtures, each CONSTRUCTED from the geometry rather than
recorded:

  A. LAYER cut-off, patterned cell -- order m = 1 sits exactly at kz = 0 of a
     cell value that NEITHER half-space carries (eps_layer = 6.25,
     n_sup = 1.0, n_sub = 1.45).  Compared: the scalar cell through
     `PMM2DStackPure` against its `e * I` promotion through the SAME entry
     point, and the scalar single-layer entry `pmm_efficiency_2d_staggered`
     against `pmm_jones_2d_staggered` on the promotion.
  B. HALF-SPACE cut-off -- order m = 1 exactly at the SUBSTRATE cut-off
     (eps_sub = 2.1025, no cell value near it).  Must be unchanged pre/post:
     both lists always carried the half-spaces.
  C. UNIFORM SCALAR layer (`add_layer(eps=...)`) on its own cut-off, against
     the same layer passed as `eps * I` (`kind="uniform_tensor"`) -- claim
     G.4's "1.4e-15 by design" residual.

Also measures the guard itself: the nudge factor and the trigger window.

    python v2_oncut.py <lumenairy-root>
"""
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
_REAL = _rc._grazing_safe_wavelength


def _spy(wavelength, kx0, ky0, m_orders, n_orders, px, py, eps_reals,
         max_iter=8):
    out = _REAL(wavelength, kx0, ky0, m_orders, n_orders, px, py, eps_reals,
                max_iter=max_iter)
    _CALLS.append((float(wavelength), float(out), len(list(eps_reals))))
    return out


_rc._grazing_safe_wavelength = _spy


def promote(m):
    m = np.asarray(m, dtype=complex)
    t = np.zeros(m.shape + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


PX = 0.54e-6
N_SUP, N_SUB = 1.0, 1.45
DEPTH = 0.23e-6
M, NO = 5, 3

# --- construct the LAYER cut-off exactly, through the geometry -------------
EPS_HI = 6.25
WL_LAYER = PX * float(np.sqrt(EPS_HI))          # order m=1: kt = wl/px = 2.5
print(f"[construct] px = {PX!r}  eps_layer = {EPS_HI!r}")
print(f"[construct] wl_layer = {WL_LAYER!r}   "
      f"(wl/px)**2 - eps = {(WL_LAYER / PX) ** 2 - EPS_HI!r}")
EPS_SUB = float(np.real(complex(N_SUB) ** 2))
WL_HALF = PX * float(np.sqrt(EPS_SUB))
print(f"[construct] eps_sub = {EPS_SUB!r}  wl_half = {WL_HALF!r}   "
      f"(wl/px)**2 - eps_sub = {(WL_HALF / PX) ** 2 - EPS_SUB!r}")

CELL = np.array([[EPS_HI, EPS_HI, 1.0],
                 [EPS_HI, 1.0, 1.0],
                 [1.0, 1.0, 1.0]], dtype=complex)


def stack_solve(layer_spec, wl):
    _CALLS.clear()
    st = PMM2DStackPure(PX, PX, n_superstrate=N_SUP, n_substrate=N_SUB,
                        n_modes=M, n_orders=NO)
    layer_spec(st)
    st.set_source(wl, theta=0.0, phi=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        o, R, T, J = st.solve(jones=True)
    return R, T, J, list(_CALLS)


def report(tag, a, b):
    Ra, Ta, Ja, ca = a
    Rb, Tb, Jb, cb = b
    print(f"  {tag}")
    print(f"    max|dR| = {np.max(np.abs(Ra - Rb)):.4e}   "
          f"max|dT| = {np.max(np.abs(Ta - Tb)):.4e}   "
          f"max|dJ| = {np.max(np.abs(Ja - Jb)):.4e}")
    print(f"    bit-identical R/T/J = {np.array_equal(Ra, Rb)} "
          f"{np.array_equal(Ta, Tb)} {np.array_equal(Ja, Jb)}")
    for nm, c in (("scalar", ca), ("promoted", cb)):
        for win, wout, n in c:
            rel = (wout - win) / win
            print(f"    {nm:9s} n_eps={n:3d}  {win!r} -> {wout!r}  "
                  f"rel {rel:+.4e}")


print()
for note, wl in (("A. ON the LAYER cut-off", WL_LAYER),
                 ("A'. 1e-9 relative below it", WL_LAYER * (1 - 1e-9)),
                 ("A''. far from any cut-off", 0.81e-6)):
    print(f"{note}   wl = {wl!r}")
    sc = stack_solve(lambda s: s.add_layer(DEPTH, eps_cell=CELL), wl)
    pr = stack_solve(lambda s: s.add_layer(DEPTH, eps_cell=promote(CELL)), wl)
    report("PMM2DStackPure: scalar cell vs e*I", sc, pr)
    print()

print(f"B. ON the SUBSTRATE (half-space) cut-off   wl = {WL_HALF!r}")
sc = stack_solve(lambda s: s.add_layer(DEPTH, eps_cell=CELL), WL_HALF)
pr = stack_solve(lambda s: s.add_layer(DEPTH, eps_cell=promote(CELL)), WL_HALF)
report("PMM2DStackPure: scalar cell vs e*I", sc, pr)
print()

# --- the two single-layer ENTRY POINTS on the layer cut-off ----------------
print("A2. entry points: pmm_efficiency_2d_staggered vs pmm_jones_2d_staggered")
for note, wl in (("ON the layer cut-off", WL_LAYER),
                 ("1e-9 below it", WL_LAYER * (1 - 1e-9))):
    _CALLS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R1, T1 = pmm_efficiency_2d_staggered(
            PX, PX, CELL, N_SUB, N_SUP, DEPTH, wl, degree=M, n_orders=NO,
            polarization="tm")
    c1 = list(_CALLS)
    _CALLS.clear()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _o, R2, T2, _J = pmm_jones_2d_staggered(
            PX, PX, promote(CELL), N_SUB, N_SUP, DEPTH, wl, degree=M,
            n_orders=NO)
    c2 = list(_CALLS)
    print(f"  {note}: max|dR| = {np.max(np.abs(R1 - R2[0])):.4e}  "
          f"max|dT| = {np.max(np.abs(T1 - T2[0])):.4e}  "
          f"bit-identical = {np.array_equal(R1, R2[0])}")
    print(f"    eff  {c1}")
    print(f"    jones{c2}")
print()

# --- C: uniform SCALAR layer vs uniform TENSOR layer ----------------------
print(f"C. UNIFORM layer eps = {EPS_HI} on its own cut-off  wl = {WL_LAYER!r}")
for note, wl in (("ON the layer cut-off", WL_LAYER),
                 ("far off", 0.81e-6)):
    us = stack_solve(lambda s: s.add_layer(DEPTH, eps=EPS_HI), wl)
    ut = stack_solve(
        lambda s: s.add_layer(DEPTH, eps=np.diag([EPS_HI] * 3).astype(complex)),
        wl)
    report(f"uniform scalar vs uniform tensor -- {note}", us, ut)
print()

# --- the guard itself: nudge factor and trigger window --------------------
print("D. the guard, called directly")
mo = np.arange(-NO, NO + 1)
mx = np.tile(mo, len(mo))
my = np.repeat(mo, len(mo))
lst = [1.0, EPS_SUB, EPS_HI]
out = _REAL(WL_LAYER, 0.0, 0.0, mx, my, PX, PX, lst)
print(f"  on cut-off: {WL_LAYER!r} -> {out!r}   "
      f"rel {(out - WL_LAYER) / WL_LAYER:.6e}   "
      f"exact wl*(1+1e-7)? {out == WL_LAYER * (1.0 + 1e-7)}")
print("  trigger window scan (relative offset r, wl = WL*(1+r)):")
for r in (0.0, 1e-13, 1e-11, 2e-11, 1e-10, 1.24e-10, 1.26e-10, 5e-10, 1e-9,
          1e-8, -1e-10, -1.24e-10, -1.26e-10, -5e-10):
    wl = WL_LAYER * (1.0 + r)
    kt2 = (mx * (wl / PX)) ** 2 + (my * (wl / PX)) ** 2
    gap = min(float(np.min(np.abs(e - kt2))) for e in lst)
    o = _REAL(wl, 0.0, 0.0, mx, my, PX, PX, lst)
    print(f"    r = {r:+.3e}  |eps - kt2|_min = {gap:.4e}  "
          f"nudged = {o != wl}  rel_out = {(o - wl) / wl:+.4e}")
