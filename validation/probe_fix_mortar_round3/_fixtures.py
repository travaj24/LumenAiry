"""Shared fixtures for the round-3 mortar probes.

Every stack here is built through the PUBLIC ``PMM2DStackPure`` surface.  The
geometry is the VERIFY audit's DEFECT V1 reproducer
(``validation/probe_verify_mortar_round2/v8_oop_regression.py``) plus the
neighbour variants its S6.3 table scopes: an out-of-plane patterned layer next
to a plain UNIFORM SPACER, next to a PATTERNED SCALAR layer on other walls,
and the healthy both-out-of-plane / both-slanted controls.
"""
import numpy as np

from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure

_C = complex

P = 1.0e-6
WL = 0.62e-6
THETA, PHI = 0.09, 0.3

WA = (0.2371, 0.6183)       # layer A's pillar, as fractions of the period
WB = (0.3117, 0.7402)       # layer B's pillar
UNION = (0.2371, 0.3117, 0.6183, 0.7402)
EPS_H = 2.25
EPS_B = 6.0
EPS_SPACER = 2.1
E_OOP = np.array([[4.0, 0.0, 0.8], [0.0, 3.4, 0.0], [0.75, 0.0, 3.2]],
                 dtype=_C)


def mid(walls):
    b = (0.0,) + tuple(walls) + (1.0,)
    return [0.5 * (b[i] + b[i + 1]) for i in range(len(b) - 1)]


def tensor_cell(walls, lo, hi, eps_in, eps_host):
    m = mid(walls)
    n = len(m)
    c = np.empty((n, n, 3, 3), dtype=_C)
    for i in range(n):
        for j in range(n):
            inside = (lo < m[i] < hi) and (lo < m[j] < hi)
            c[i, j] = eps_in if inside else np.eye(3) * eps_host
    return c


def scalar_cell(walls, lo, hi, eps_in, eps_host):
    m = mid(walls)
    n = len(m)
    c = np.full((n, n), _C(eps_host))
    for i in range(n):
        for j in range(n):
            if (lo < m[i] < hi) and (lo < m[j] < hi):
                c[i, j] = _C(eps_in)
    return c


def _scaled(walls):
    return [w * P for w in walls]


def build(kind, M, n_orders=2, walls_b=WB):
    """One fixture.  ``kind``:

      ``oop_scalar``  OOP patterned layer + PATTERNED SCALAR layer (v8)
      ``oop_spacer``  OOP patterned layer + plain UNIFORM SPACER (grid=1)
      ``oop_spacer_conf``  the same device with the spacer on the OOP layer's
                      OWN walls -- CONFORMING, so the mortar is bypassed: the
                      mortar-free oracle for ``oop_spacer``
      ``oop_both``    both layers out-of-plane          (healthy control)
      ``slant_both``  both layers slanted               (healthy control)
      ``union``       the v8 device on the COMMON REFINEMENT (no mortar)
    """
    s = PMM2DStackPure(P, n_modes=M, n_orders=n_orders, n_substrate=1.5,
                       layer_grids="per-layer")
    if kind == "oop_scalar":
        s.add_layer(0.13e-6, eps_cell=tensor_cell(WA, WA[0], WA[1], E_OOP,
                                                  EPS_H),
                    x_walls=_scaled(WA), y_walls=_scaled(WA))
        s.add_layer(0.10e-6, eps_cell=scalar_cell(walls_b, walls_b[0],
                                                  walls_b[1], EPS_B, EPS_H),
                    x_walls=_scaled(walls_b), y_walls=_scaled(walls_b))
    elif kind in ("oop_spacer", "oop_spacer_conf"):
        s.add_layer(0.13e-6, eps_cell=tensor_cell(WA, WA[0], WA[1], E_OOP,
                                                  EPS_H),
                    x_walls=_scaled(WA), y_walls=_scaled(WA))
        if kind == "oop_spacer":
            s.add_layer(0.10e-6, eps=EPS_SPACER)
        else:
            s.add_layer(0.10e-6, eps=EPS_SPACER,
                        x_walls=_scaled(WA), y_walls=_scaled(WA))
    elif kind == "oop_both":
        s.add_layer(0.13e-6, eps_cell=tensor_cell(WA, WA[0], WA[1], E_OOP,
                                                  EPS_H),
                    x_walls=_scaled(WA), y_walls=_scaled(WA))
        s.add_layer(0.10e-6, eps_cell=tensor_cell(walls_b, walls_b[0],
                                                  walls_b[1], 0.7 * E_OOP,
                                                  EPS_H),
                    x_walls=_scaled(walls_b), y_walls=_scaled(walls_b))
    elif kind == "slant_both":
        s.add_layer(0.13e-6, eps_cell=scalar_cell(WA, WA[0], WA[1], EPS_B,
                                                  EPS_H),
                    x_walls=_scaled(WA), y_walls=_scaled(WA),
                    slant=(0.08, 0.03))
        s.add_layer(0.10e-6, eps_cell=scalar_cell(walls_b, walls_b[0],
                                                  walls_b[1], 4.0, EPS_H),
                    x_walls=_scaled(walls_b), y_walls=_scaled(walls_b),
                    slant=(0.08, 0.03))
    elif kind == "union":
        s.add_layer(0.13e-6, eps_cell=tensor_cell(UNION, WA[0], WA[1], E_OOP,
                                                  EPS_H),
                    x_walls=_scaled(UNION), y_walls=_scaled(UNION))
        s.add_layer(0.10e-6, eps_cell=scalar_cell(UNION, WB[0], WB[1], EPS_B,
                                                  EPS_H),
                    x_walls=_scaled(UNION), y_walls=_scaled(UNION))
    else:
        raise ValueError(kind)
    s.set_source(WL, theta=THETA, phi=PHI)
    return s


def zeroth_R(orders, R):
    """``R`` of the zeroth order for the second incident polarisation -- the
    scalar the VERIFY audit's v8 table reports."""
    o = np.asarray(orders)
    R = np.atleast_2d(R)
    k = int(np.argmin(np.abs(o[:, 0]) + np.abs(o[:, 1])))
    assert o[k, 0] == 0 and o[k, 1] == 0, o[k]
    return float(R[1, k])
