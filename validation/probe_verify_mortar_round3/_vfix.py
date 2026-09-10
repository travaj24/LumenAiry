"""Fixtures for the INDEPENDENT verification of round 3 of the pure staggered
2-D PMM per-layer mortar.

Deliberately DISJOINT from ``validation/probe_fix_mortar_round3/_fixtures.py``
and from ``tests/unit/test_fix_pmm2d_mortar_round3.py``: different period,
wavelength, incidence, host/pillar contrast, out-of-plane tensor, wall
positions and slant.  Nothing here is copied from the fix; every number the
verification reports is measured through this file.

UNITS.  Everything is METRES.  The period is 0.87e-6 and the wavelength
0.73e-6; wall FRACTIONS are multiplied by the period at the call site
(``_sc``).  Driving a metre-scale geometry with a period-1 wavelength (or the
reverse) makes ``k0`` wrong by 1e6 and the per-segment quadrature asks for a
~1e5-node Gauss-Legendre rule.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import _C

# ------------------------------------------------------------------ geometry
P = 0.87e-6
WL = 0.73e-6
THETA, PHI = 0.17, 1.1
EPS_H = 1.96                      # host
EPS_B = 8.41                      # in-plane pillar
EPS_SPACER = 2.56

#: an OUT-OF-PLANE cell -- e_xz / e_zx well above the 1e-12 relative floor of
#: ``_tile_needs_oop``, so the layer routes to the 4 q^2 first-order generator
E_OOP = np.array([[3.10, 0.0, 0.62],
                  [0.0, 2.70, 0.0],
                  [0.55, 0.0, 2.45]], dtype=_C)
E_OOP2 = np.array([[2.40, 0.0, -0.41],
                   [0.0, 2.90, 0.0],
                   [-0.37, 0.0, 2.10]], dtype=_C)

WA = (0.1873, 0.5412)             # the out-of-plane layer's pillar
WB = (0.3106, 0.8039)             # a neighbour on OTHER walls
WD = (0.2415, 0.6688)             # a third, for the both-promoted middle
WAB = (0.1873, 0.3106, 0.5412, 0.8039)          # common refinement of WA/WB

MU_IN = np.array([[1.35, 0.0, 0.0],
                  [0.0, 1.20, 0.0],
                  [0.0, 0.0, 1.10]], dtype=_C)


def sc(walls, period=P):
    return [float(w) * period for w in walls]


def _mid(walls):
    b = (0.0,) + tuple(walls) + (1.0,)
    return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]


def tensor_cell(walls, lo, hi, eps_in=None, eps_host=EPS_H):
    m = _mid(walls)
    n = len(m)
    c = np.empty((n, n, 3, 3), dtype=_C)
    e = E_OOP if eps_in is None else np.asarray(eps_in, dtype=_C)
    for i in range(n):
        for j in range(n):
            c[i, j] = e if (lo < m[i] < hi and lo < m[j] < hi) \
                else np.eye(3) * eps_host
    return c


def scalar_cell(walls, lo, hi, eps_in=EPS_B, eps_host=EPS_H):
    m = _mid(walls)
    n = len(m)
    c = np.full((n, n), _C(eps_host))
    for i in range(n):
        for j in range(n):
            if lo < m[i] < hi and lo < m[j] < hi:
                c[i, j] = _C(eps_in)
    return c


def mu_cell(walls, lo, hi, mu_in=None, mu_host=1.0):
    m = _mid(walls)
    n = len(m)
    c = np.empty((n, n, 3, 3), dtype=_C)
    mi = MU_IN if mu_in is None else np.asarray(mu_in, dtype=_C)
    for i in range(n):
        for j in range(n):
            c[i, j] = mi if (lo < m[i] < hi and lo < m[j] < hi) \
                else np.eye(3) * mu_host
    return c


def new_stack(M, *, n_orders=2, period=P, layer_grids="per-layer"):
    return PMM2DStackPure(period, n_modes=M, n_orders=n_orders,
                          n_superstrate=1.0, n_substrate=1.45,
                          layer_grids=layer_grids)


# --------------------------------------------------------------- the stacks
#: Every entry is (label, builder(M, n_orders) -> stack).  ``kind`` tags say
#: what the interface's two sides are, which is what claim (a) is about.
def build(kind, M, n_orders=2, *, period=P, wl=WL, theta=THETA, phi=PHI):
    st = new_stack(M, n_orders=n_orders, period=period)

    def oopA(**kw):
        st.add_layer(0.118e-6, eps_cell=tensor_cell(WA, *WA),
                     x_walls=sc(WA, period), y_walls=sc(WA, period), **kw)

    if kind == "mix_spacer":
        # OUT-OF-PLANE patterned + a plain UNIFORM spacer on its own grid
        oopA()
        st.add_layer(0.094e-6, eps=EPS_SPACER)
    elif kind == "mix_spacer_conf":
        # the SAME DEVICE with the spacer on the out-of-plane layer's own wall
        # array -> grids coincide, identical-grid bypass, NO mortar anywhere
        oopA()
        st.add_layer(0.094e-6, eps=EPS_SPACER,
                     x_walls=sc(WA, period), y_walls=sc(WA, period))
    elif kind == "mix_pattern":
        # OUT-OF-PLANE patterned + an IN-PLANE patterned neighbour, other walls
        oopA()
        st.add_layer(0.094e-6, eps_cell=scalar_cell(WB, *WB),
                     x_walls=sc(WB, period), y_walls=sc(WB, period))
    elif kind == "mix_pattern_union":
        # the same device on the COMMON REFINEMENT -> conforming, no mortar
        st.add_layer(0.118e-6, eps_cell=tensor_cell(WAB, *WA),
                     x_walls=sc(WAB, period), y_walls=sc(WAB, period))
        st.add_layer(0.094e-6, eps_cell=scalar_cell(WAB, *WB),
                     x_walls=sc(WAB, period), y_walls=sc(WAB, period))
    elif kind == "mix_magnetic":
        # OUT-OF-PLANE patterned + an IN-PLANE MAGNETIC patterned neighbour
        oopA()
        st.add_layer(0.094e-6, eps_cell=scalar_cell(WB, *WB),
                     mu_cell=mu_cell(WB, *WB),
                     x_walls=sc(WB, period), y_walls=sc(WB, period))
    elif kind == "mix_slant_inplane":
        # a SLANTED patterned layer + an IN-PLANE UNIFORM neighbour on its own
        # grid.  A slanted patterned layer beside an IN-PLANE PATTERNED one is
        # REFUSED by _check_stack_slant (mixed slants between patterned
        # layers), so the uniform spacer is the reachable form of "slanted
        # next to in-plane".
        st.add_layer(0.118e-6, eps_cell=scalar_cell(WA, *WA),
                     x_walls=sc(WA, period), y_walls=sc(WA, period),
                     slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps=EPS_SPACER, grid=3)
    elif kind == "mix_slant_inplane_conf":
        # the SAME device with the spacer on the slanted layer's own walls
        st.add_layer(0.118e-6, eps_cell=scalar_cell(WA, *WA),
                     x_walls=sc(WA, period), y_walls=sc(WA, period),
                     slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps=EPS_SPACER,
                     x_walls=sc(WA, period), y_walls=sc(WA, period))
    elif kind == "ctrl_oop_both":
        oopA()
        st.add_layer(0.094e-6, eps_cell=tensor_cell(WB, *WB, eps_in=E_OOP2),
                     x_walls=sc(WB, period), y_walls=sc(WB, period))
    elif kind == "ctrl_slant_both":
        st.add_layer(0.118e-6, eps_cell=scalar_cell(WA, *WA),
                     x_walls=sc(WA, period), y_walls=sc(WA, period),
                     slant=(0.11, 0.05))
        st.add_layer(0.094e-6, eps_cell=scalar_cell(WB, *WB, eps_in=5.3),
                     x_walls=sc(WB, period), y_walls=sc(WB, period),
                     slant=(0.11, 0.05))
    elif kind == "ctrl_inplane_both":
        # NO out-of-plane layer anywhere: this stack does NOT reach the
        # generalized site at all -- it takes the two IN-PLANE mortar sites
        st.add_layer(0.118e-6, eps_cell=scalar_cell(WA, *WA),
                     x_walls=sc(WA, period), y_walls=sc(WA, period))
        st.add_layer(0.094e-6, eps_cell=scalar_cell(WB, *WB, eps_in=5.3),
                     x_walls=sc(WB, period), y_walls=sc(WB, period))
    elif kind == "both_promoted":
        # THREE layers: the MIDDLE interface has BOTH sides promoted (two
        # in-plane regions on different grids), while an out-of-plane layer
        # elsewhere routes the whole cascade to the generalized form
        oopA()
        st.add_layer(0.071e-6, eps_cell=scalar_cell(WB, *WB),
                     x_walls=sc(WB, period), y_walls=sc(WB, period))
        st.add_layer(0.083e-6, eps_cell=scalar_cell(WD, *WD, eps_in=4.7),
                     x_walls=sc(WD, period), y_walls=sc(WD, period))
    elif kind == "adv_near_inplane_oop":
        # ADVERSARIAL CONTROL.  Neither side is PROMOTED -- both run the
        # out-of-plane generator -- but side b's tensor is in-plane to 1e-9,
        # i.e. it carries the same near-symmetry the promotion does.  If the
        # near-null vector localises here too, "by construction FROM THE
        # PROMOTION" is the wrong statement of the mechanism.
        e_tiny = np.array([[3.10, 0.0, 3.10e-9],
                           [0.0, 2.70, 0.0],
                           [3.10e-9, 0.0, 2.45]], dtype=_C)
        oopA()
        st.add_layer(0.094e-6, eps_cell=tensor_cell(WB, *WB, eps_in=e_tiny),
                     x_walls=sc(WB, period), y_walls=sc(WB, period))
    elif kind == "adv_tiny_slant_inplane":
        # ADVERSARIAL MIXED.  Side a is SLANTED (not promoted) by 1e-8 rad,
        # side b is a promoted in-plane region.  Both sides are near-symmetric.
        st.add_layer(0.118e-6, eps_cell=scalar_cell(WA, *WA),
                     x_walls=sc(WA, period), y_walls=sc(WA, period),
                     slant=(1e-8, 0.0))
        st.add_layer(0.094e-6, eps=EPS_SPACER, grid=3)
    elif kind == "adv_strong_oop_next_to_inplane":
        # ADVERSARIAL MIXED.  A VERY strongly out-of-plane side a (e_xz of the
        # same size as the diagonal) beside a promoted in-plane side b: if the
        # localisation is a property of the PROMOTION it must survive this.
        e_str = np.array([[3.10, 0.0, 1.55],
                          [0.0, 2.70, 0.0],
                          [1.48, 0.0, 2.45]], dtype=_C)
        st.add_layer(0.118e-6, eps_cell=tensor_cell(WA, *WA, eps_in=e_str),
                     x_walls=sc(WA, period), y_walls=sc(WA, period))
        st.add_layer(0.094e-6, eps_cell=scalar_cell(WB, *WB, eps_in=5.3),
                     x_walls=sc(WB, period), y_walls=sc(WB, period))
    else:
        raise ValueError(kind)
    st.set_source(wl, theta=theta, phi=phi)
    return st


MIXED = ("mix_spacer", "mix_pattern", "mix_magnetic", "mix_slant_inplane",
         "adv_strong_oop_next_to_inplane", "adv_tiny_slant_inplane")
CONTROLS = ("ctrl_oop_both", "ctrl_slant_both", "adv_near_inplane_oop")


def R00(orders, R, row=1):
    o = np.asarray(orders)
    k = int(np.argmin(np.abs(o[:, 0]) + np.abs(o[:, 1])))
    assert o[k, 0] == 0 and o[k, 1] == 0
    return float(np.atleast_2d(R)[row, k])


def env():
    import platform  # noqa: I001

    import lumenairy
    import scipy
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "lumenairy": lumenairy.__version__,
        "lumenairy_file": lumenairy.__file__,
        "platform": platform.platform(),
    }
