"""Probe 11 -- derive the lossless-closure tripwire window from the SHIPPED
scalar fixtures of the pure staggered engine (they must never fire it).

Reproduces every lossless scalar configuration the shipped suites exercise on
`pmm_efficiency_2d_staggered` / `PMM2DStackPure` and reports |sum R+T - 1|.
"""
import numpy as np

import lumenairy

assert lumenairy.__file__.replace("\\", "/").startswith("C:/tmp/lum_aniso"), \
    lumenairy.__file__

from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure  # noqa: E402
from lumenairy.elements.pmm.twod_staggered import (  # noqa: E402
    pmm_efficiency_2d_staggered,
)

rows = []


def rec(tag, tot):
    tot = np.atleast_1d(np.asarray(tot, dtype=float))
    dev = float(np.max(np.abs(tot - 1.0)))
    rows.append((dev, tag))


# --- test_v5_12_0_pmm2d_staggered fixtures --------------------------------
G = dict(period_x=0.8e-6, period_y=0.8e-6, depth=0.3e-6, wavelength=0.633e-6)
PILLAR = np.array([[2.25, 1.0], [1.0, 1.0]], dtype=complex)
for no in (3, 5, 7):
    o, R, T = pmm_efficiency_2d_staggered(eps_cell=np.ones((2, 2)),
                                          n_substrate=1.0, n_superstrate=1.0,
                                          degree=6, n_orders=no, **G)
    rec(f"vacuum deg6 no{no}", R.sum() + T.sum())
for deg in (4, 5, 6, 7, 8):
    for no in (3, 5, 7):
        o, R, T = pmm_efficiency_2d_staggered(
            eps_cell=PILLAR, n_substrate=1.0, n_superstrate=1.0, degree=deg,
            n_orders=no, **G)
        rec(f"pillar deg{deg} no{no}", R.sum() + T.sum())
for pol in ("te", "tm"):
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=PILLAR, n_substrate=1.0, n_superstrate=1.0, degree=6,
        n_orders=5, polarization=pol, **G)
    rec(f"pillar deg6 {pol}", R.sum() + T.sum())
# uniform slab Fabry-Perot + position invariance
for shift in range(4):
    c = np.ones((2, 2), dtype=complex)
    c[shift // 2, shift % 2] = 2.25
    o, R, T = pmm_efficiency_2d_staggered(
        eps_cell=c, n_substrate=1.0, n_superstrate=1.0, degree=6, n_orders=5,
        **G)
    rec(f"pillar shift{shift}", R.sum() + T.sum())

# --- test_v5_21 oblique fixtures ------------------------------------------
PX = PY = 0.9e-6
ASYM = np.array([[2.0] * 3, [4.0] * 3, [9.0] * 3], np.complex128)
for cell, tag in ((np.array([[2.0, 2.0], [6.0, 6.0]], np.complex128), "stripe"),
                  (np.array([[6.0, 2.0], [2.0, 2.0]], np.complex128), "pillar"),
                  (ASYM, "asym3")):
    for th in (0.0, 0.10, 0.20):
        o, R, T = pmm_efficiency_2d_staggered(
            PX, PY, cell, 1.0, 1.0, 0.30e-6, 0.60e-6, degree=8, n_orders=6,
            theta=th)
        rec(f"oblique {tag} th{th}", R.sum() + T.sum())
# the shipped multilayer A|B pure-stack fixture
stk = PMM2DStackPure(PX, PY, n_modes=8, n_orders=6)
stk.add_layer(0.30e-6, eps_cell=np.array([[2.0] * 3, [4.0] * 3, [9.0] * 3],
                                         np.complex128))
stk.add_layer(0.22e-6, eps_cell=np.array([[9.0] * 3, [2.0] * 3, [4.0] * 3],
                                         np.complex128))
for th in (0.0, 0.10, 0.20):
    stk.set_source(0.60e-6, theta=th)
    o, R, T, J = stk.solve()
    rec(f"pure stack A|B th{th}", R.sum(axis=1) + T.sum(axis=1))

# --- lossless PMM2DStackPure configurations from the niche/w7 suite --------
for n_sub in (1.0, 1.5):
    for th in (0.0, 0.25):
        st = PMM2DStackPure(0.9e-6, 0.9e-6, n_substrate=n_sub, n_modes=6,
                            n_orders=5)
        st.add_layer(0.25e-6, eps_cell=np.array([[6.25, 1.0], [1.0, 1.0]],
                                                dtype=complex))
        st.add_layer(0.15e-6, eps=2.25)
        st.set_source(0.55e-6, theta=th)
        o, R, T, J = st.solve()
        rec(f"stack pillar+uniform nsub{n_sub} th{th}",
            R.sum(axis=1) + T.sum(axis=1))

rows.sort(reverse=True)
print(f"{'|R+T-1|':>10s}  fixture")
for dev, tag in rows[:14]:
    print(f"{dev:10.3e}  {tag}")
print(f"\nWORST over {len(rows)} lossless scalar fixtures: {rows[0][0]:.3e}"
      f"   ({rows[0][1]})")
