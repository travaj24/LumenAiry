"""Fixtures for the INDEPENDENT verification of ROUND 4 of the pure staggered
2-D PMM per-layer mortar (the per-AXIS band warning and the exactly-one-
promoted-side wording).

Deliberately DISJOINT from ``validation/probe_fix_mortar_round4/`` and from
``validation/probe_verify_mortar_round3/_vfix.py``: different period,
wavelength, incidence, substrate, host/pillar contrast, out-of-plane tensor,
magnetic tensor, wall positions, slant and layer thicknesses.  Nothing here is
copied from the fix; every number this verification reports is measured
through this file.

UNITS.  Everything is METRES.  The period is 1.24e-6 and the wavelength
0.905e-6; wall FRACTIONS are multiplied by the period at the call site
(``sc``).  Driving a metre-scale geometry with a period-1 wavelength (or the
reverse) makes ``k0`` wrong by 1e6 and the per-segment quadrature asks for a
~1e5-node Gauss-Legendre rule.

THE SHAPE EVERY BAND CLAIM RESTS ON.  A wall that separates two segments of
the SAME permittivity is a pure PARTITION choice: the device cannot depend on
where it sits, so ANY movement of the answer as that wall's segment narrows is
numerical damage rather than physics.  :func:`ladder` builds exactly that --
uniform permittivity across the swept wall, everything else pinned.
"""
from __future__ import annotations

import numpy as np

from lumenairy.elements.pmm import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import _C

# ------------------------------------------------------------------ geometry
P = 1.24e-6
WL = 0.905e-6
THETA, PHI = 0.29, 0.71
N_SUB = 1.45

EPS_H = 2.31                      # host
EPS_B = 7.29                      # in-plane pillar
EPS_B2 = 5.76                     # a second pillar
EPS_S = 3.24                      # uniform spacer

#: OUT-OF-PLANE cells -- e_xz / e_zx far above the 1e-12 relative floor of
#: ``_tile_needs_oop``, so the layer routes to the 4 q^2 first-order generator
E_OOP_A = np.array([[3.41, 0.0, 0.71],
                    [0.0, 2.93, 0.0],
                    [0.66, 0.0, 2.58]], dtype=_C)
E_OOP_B = np.array([[2.62, 0.0, -0.48],
                    [0.0, 3.14, 0.0],
                    [-0.44, 0.0, 2.21]], dtype=_C)
MU_IN = np.array([[1.28, 0.0, 0.0],
                  [0.0, 1.17, 0.0],
                  [0.0, 0.0, 1.06]], dtype=_C)

# ordinary wall arrays, as FRACTIONS of the period
WXA = (0.1934, 0.5623)
WXB = (0.2871, 0.7145)
WXC = (0.3462, 0.6809)
WYA = (0.2417, 0.6382)
WYB = (0.3355, 0.7708)
WYC = (0.1826, 0.5271)

# FOUR-segment partners.  ``eps_cell`` must be SQUARE (Nx == Ny -- the
# staggered tensor-product basis needs Nx*(M-1) == Ny*(M-1)), so a fixture
# that puts THREE interior walls on one axis must put three on the other too.
WX4A = (0.1934, 0.5623, 0.8200)
WX4B = (0.2871, 0.7145, 0.8800)
WY4 = (0.1826, 0.5271, 0.7800)

XC, YC = 0.4400, 0.5850           # centres the narrow segments straddle


def sc(fracs, period=P):
    """Interior walls, FRACTIONS -> metres."""
    return [float(f) * period for f in fracs]


def narrow(g, centre, period=P):
    """A two-wall array whose MIDDLE segment is ``g`` of the period."""
    return sc((centre - 0.5 * g, centre + 0.5 * g), period)


def _n(walls):
    """Segment count implied by an interior-wall array."""
    return len(walls) + 1


def cell_scalar(nx, ny, *, eps_in=EPS_B, eps_host=EPS_H, box=None):
    """``(nx, ny)`` scalar tile.  ``box`` = ``(i0, i1, j0, j1)`` index window
    carrying ``eps_in``; ``None`` fills the whole tile with the host, which is
    what makes every wall a PURE PARTITION choice."""
    c = np.full((nx, ny), _C(eps_host))
    if box is not None:
        i0, i1, j0, j1 = box
        c[i0:i1, j0:j1] = _C(eps_in)
    return c


def cell_tensor(nx, ny, e=E_OOP_A, *, eps_host=EPS_H, box=None):
    c = np.empty((nx, ny, 3, 3), dtype=_C)
    c[...] = np.eye(3) * _C(eps_host)
    if box is None:
        c[...] = np.asarray(e, dtype=_C)
    else:
        i0, i1, j0, j1 = box
        c[i0:i1, j0:j1] = np.asarray(e, dtype=_C)
    return c


def cell_mu(nx, ny, *, box=None, mu_host=1.0):
    c = np.empty((nx, ny, 3, 3), dtype=_C)
    c[...] = np.eye(3) * _C(mu_host)
    if box is None:
        c[...] = MU_IN
    else:
        i0, i1, j0, j1 = box
        c[i0:i1, j0:j1] = MU_IN
    return c


def new_stack(M, *, n_orders=1, period=P):
    return PMM2DStackPure(period, n_modes=M, n_orders=n_orders,
                          n_superstrate=1.0, n_substrate=N_SUB,
                          layer_grids="per-layer")


def _add(st, t, xw, yw, *, kind="pillar", eps_in=EPS_B, e=E_OOP_A,
         slant=None, magnetic=False, box=(1, 2, 1, 2)):
    nx, ny = _n(xw), _n(yw)
    kw = {}
    if slant is not None:
        kw["slant"] = slant
    if kind == "pillar":
        cell = cell_scalar(nx, ny, eps_in=eps_in, box=box)
    elif kind == "host":                       # uniform: wall = PARTITION only
        cell = cell_scalar(nx, ny, box=None)
    elif kind == "oop":
        cell = cell_tensor(nx, ny, e, box=box)
    else:
        raise ValueError(kind)
    if magnetic:
        kw["mu_cell"] = cell_mu(nx, ny, box=box)
    st.add_layer(t, eps_cell=cell, x_walls=xw, y_walls=yw, **kw)


# =========================================================================
# the FIXTURE FAMILY -- part A (bit-identity) and part B (adversarial)
# =========================================================================
def build(name, M=4):                                    # noqa: C901, PLR0912
    st = new_stack(M)
    xa, xb, xc = sc(WXA), sc(WXB), sc(WXC)
    ya, yb, yc = sc(WYA), sc(WYB), sc(WYC)

    # ---- CONFORMING: no mortar anywhere -------------------------------
    if name == "conf2":
        _add(st, 0.131e-6, xa, ya)
        _add(st, 0.107e-6, xa, ya, kind="host")
    elif name == "conf3":
        _add(st, 0.131e-6, xa, ya)
        _add(st, 0.107e-6, xa, ya, kind="host")
        _add(st, 0.093e-6, xa, ya, eps_in=EPS_B2)
    elif name == "conf_narrow_y":
        yn = narrow(3.0e-3, YC)
        _add(st, 0.131e-6, xa, yn)
        _add(st, 0.107e-6, xa, yn, kind="host")
    elif name == "conf_narrow_x":
        xn = narrow(3.0e-3, XC)
        _add(st, 0.131e-6, xn, ya)
        _add(st, 0.107e-6, xn, ya, kind="host")

    # ---- NON-CONFORMING ON X ONLY -------------------------------------
    elif name == "x_only_ord":
        _add(st, 0.131e-6, xa, ya)
        _add(st, 0.107e-6, xb, ya, eps_in=EPS_B2)
    elif name == "x_only_narrow_x":
        _add(st, 0.131e-6, narrow(5.0e-3, XC), ya)
        _add(st, 0.107e-6, xb, ya, eps_in=EPS_B2)
    elif name == "x_only_narrow_y":                # the DEFECT-2 shape
        yn = narrow(3.0e-3, YC)
        _add(st, 0.131e-6, xa, yn)
        _add(st, 0.107e-6, xb, yn, eps_in=EPS_B2)
    elif name == "x_only_mixed":                   # x banded, y NARROWER
        yn = narrow(3.0e-3, YC)
        _add(st, 0.131e-6, narrow(9.0e-3, XC), yn)
        _add(st, 0.107e-6, xb, yn, eps_in=EPS_B2)

    # ---- NON-CONFORMING ON Y ONLY (the MIRROR of the defect) ----------
    elif name == "y_only_ord":
        _add(st, 0.131e-6, xa, ya)
        _add(st, 0.107e-6, xa, yb, eps_in=EPS_B2)
    elif name == "y_only_narrow_y":
        _add(st, 0.131e-6, xa, narrow(5.0e-3, YC))
        _add(st, 0.107e-6, xa, yb, eps_in=EPS_B2)
    elif name == "y_only_narrow_x":                # MIRROR of DEFECT 2
        xn = narrow(3.0e-3, XC)
        _add(st, 0.131e-6, xn, ya)
        _add(st, 0.107e-6, xn, yb, eps_in=EPS_B2)

    # ---- NON-CONFORMING ON BOTH ---------------------------------------
    elif name == "both_ord":
        _add(st, 0.131e-6, xa, ya)
        _add(st, 0.107e-6, xb, yb, eps_in=EPS_B2)
    elif name == "both_narrow_x":
        _add(st, 0.131e-6, narrow(4.0e-3, XC), ya)
        _add(st, 0.107e-6, xb, yb, eps_in=EPS_B2)
    elif name == "both_narrow_y":
        _add(st, 0.131e-6, xa, narrow(4.0e-3, YC))
        _add(st, 0.107e-6, xb, yb, eps_in=EPS_B2)

    # ---- MIXED in-plane / OUT-OF-PLANE (the GENERALIZED cascade) ------
    elif name == "oop_x_only":
        _add(st, 0.118e-6, xa, ya, kind="oop")
        _add(st, 0.094e-6, xb, ya, eps_in=EPS_B2)
    elif name == "oop_both":
        _add(st, 0.118e-6, xa, ya, kind="oop")
        _add(st, 0.094e-6, xb, yb, eps_in=EPS_B2)
    elif name == "oop_narrow_y_conf":
        yn = narrow(3.0e-3, YC)
        _add(st, 0.118e-6, xa, yn, kind="oop")
        _add(st, 0.094e-6, xb, yn, eps_in=EPS_B2)
    elif name == "oop_spacer":
        _add(st, 0.118e-6, xa, ya, kind="oop")
        st.add_layer(0.094e-6, eps=EPS_S)
    elif name == "oop_both_sides":                 # BOTH sides out-of-plane
        _add(st, 0.118e-6, xa, ya, kind="oop")
        _add(st, 0.094e-6, xb, yb, kind="oop", e=E_OOP_B)

    # ---- MAGNETIC ------------------------------------------------------
    elif name == "magnetic_x_only":
        _add(st, 0.118e-6, xa, ya, kind="oop")
        _add(st, 0.094e-6, xb, ya, eps_in=EPS_B2, magnetic=True)
    elif name == "magnetic_narrow_y_conf":
        yn = narrow(3.0e-3, YC)
        _add(st, 0.118e-6, xa, yn, kind="oop")
        _add(st, 0.094e-6, xb, yn, eps_in=EPS_B2, magnetic=True)
    elif name == "magnetic_conf":                  # magnetic, NO mortar
        _add(st, 0.118e-6, xa, ya, kind="oop")
        _add(st, 0.094e-6, xa, ya, eps_in=EPS_B2, magnetic=True)

    # ---- SLANT ---------------------------------------------------------
    elif name == "slant_spacer":
        _add(st, 0.118e-6, xa, ya, slant=(0.13, 0.06))
        st.add_layer(0.094e-6, eps=EPS_S, grid=3)
    elif name == "slant_spacer_narrow_y":
        yn = narrow(3.0e-3, YC)
        _add(st, 0.118e-6, xa, yn, slant=(0.13, 0.06))
        st.add_layer(0.094e-6, eps=EPS_S, grid=3)

    # ---- TAPERS (each slice its OWN x grid, y SHARED and ordinary) -----
    elif name.startswith("taper"):
        n_sl = int(name[5:])
        w0 = 0.52
        for i in range(n_sl):
            w = w0 * (1.0 - (i + 0.5) / n_sl)
            xw = sc((0.5 - w / 2, 0.5 + w / 2))
            _add(st, 0.036e-6, xw, ya)

    # ---- ADVERSARIAL (task B) -----------------------------------------
    elif name == "advB_i":
        # x DIFFERS by a wall that is NOT the narrow one; the narrow x segment
        # is IDENTICAL in both layers and sits elsewhere on x.  The axis
        # carries a mortar, so the rule must still warn -- and name x.
        g = 6.0e-3
        x0 = sc((0.1400, 0.7000 - g, 0.7000))
        x1 = sc((0.2600, 0.7000 - g, 0.7000))
        y4 = sc(WY4)
        _add(st, 0.131e-6, x0, y4)
        _add(st, 0.107e-6, x1, y4, eps_in=EPS_B2)
    elif name == "advB_ii":
        # THREE layers.  A and B share the y wall array and differ on x;
        # B and C differ on y.  The narrow y segment is common to all three,
        # so it is projected across the B-C interface's y mortar.
        g = 4.0e-3
        y_ab = sc((0.2100, 0.6800 - g, 0.6800))
        y_c = sc((0.3300, 0.6800 - g, 0.6800))
        x4a, x4b = sc(WX4A), sc(WX4B)
        _add(st, 0.111e-6, x4a, y_ab)
        _add(st, 0.097e-6, x4b, y_ab, eps_in=EPS_B2)
        _add(st, 0.089e-6, x4b, y_c, eps_in=EPS_B2)
    elif name == "advB_iii":
        # FOUR layers, ALTERNATING conformity: L0/L1 share x differ y,
        # L1/L2 differ x share y, L2/L3 share BOTH.  Both axes are live.
        _add(st, 0.101e-6, xa, ya)
        _add(st, 0.089e-6, xa, yb, eps_in=EPS_B2)
        _add(st, 0.083e-6, xc, yb)
        _add(st, 0.077e-6, xc, yb, kind="host")
    elif name == "advB_far":
        # The narrow x segment is in L0, whose neighbours CONFORM on x; the
        # only x mortar is at the L2-L3 interface, far away.  Round 4 warns
        # (it asks the STACK, per axis) -- an OVER-warning, the safe
        # direction, recorded rather than argued.
        xn = narrow(5.0e-3, XC)
        _add(st, 0.101e-6, xn, ya)
        _add(st, 0.089e-6, xn, yb, eps_in=EPS_B2)
        _add(st, 0.083e-6, xn, yc)
        _add(st, 0.077e-6, xc, yc, kind="host")
    elif name == "advB_ymortar_narrow_x":
        # THE MISS CANDIDATE.  y carries the mortar; x CONFORMS and carries a
        # band-width segment whose two sides are the SAME permittivity, so the
        # x wall is a pure PARTITION choice.  Round 3 warns x; round 4 is
        # silent.  ``ladder('x_conforming')`` measures whether it should be.
        xn = narrow(3.0e-3, XC)
        _add(st, 0.131e-6, xn, ya, kind="host")
        _add(st, 0.107e-6, xn, yb, kind="host")
    # ---- the PROMOTION population (task C, the spread bars) ------------
    elif name == "bars_both_promoted":
        # THREE layers: the MIDDLE interface has BOTH sides promoted (two
        # in-plane regions on different grids) while the out-of-plane layer
        # above puts the WHOLE cascade on the generalized form.
        _add(st, 0.118e-6, xa, ya, kind="oop")
        _add(st, 0.081e-6, xb, yb, eps_in=EPS_B2)
        _add(st, 0.093e-6, xc, yc, eps_in=EPS_B)
    elif name == "bars_slant_both":                # NEITHER promoted
        _add(st, 0.118e-6, xa, ya, slant=(0.13, 0.06))
        _add(st, 0.094e-6, xb, yb, eps_in=EPS_B2, slant=(0.13, 0.06))
    elif name == "bars_near_inplane_oop":          # NEITHER promoted
        # side b's tensor is in-plane to 1e-9: it carries the same near-
        # symmetry the promotion does but is NOT promoted.  If the near-null
        # vector localises here too, the mechanism is mis-stated.
        e_tiny = np.array([[3.41, 0.0, 3.41e-9],
                           [0.0, 2.93, 0.0],
                           [3.41e-9, 0.0, 2.58]], dtype=_C)
        _add(st, 0.118e-6, xa, ya, kind="oop")
        _add(st, 0.094e-6, xb, yb, kind="oop", e=e_tiny)
    elif name == "bars_strong_oop":                # EXACTLY ONE promoted
        e_str = np.array([[3.41, 0.0, 1.70],
                          [0.0, 2.93, 0.0],
                          [1.62, 0.0, 2.58]], dtype=_C)
        _add(st, 0.118e-6, xa, ya, kind="oop", e=e_str)
        _add(st, 0.094e-6, xb, yb, eps_in=EPS_B2)
    elif name == "bars_tiny_slant":                # EXACTLY ONE promoted
        _add(st, 0.118e-6, xa, ya, slant=(1e-8, 0.0))
        st.add_layer(0.094e-6, eps=EPS_S, grid=3)
    else:
        raise ValueError(name)
    st.set_source(WL, theta=THETA, phi=PHI)
    return st


#: Task C's promotion population -- every stack that reaches the GENERALIZED
#: mortar site, so its operands can be classified one-promoted / both-promoted
#: / neither-promoted.
BARS_FAMILY = ("oop_x_only", "oop_both", "oop_spacer", "oop_narrow_y_conf",
               "magnetic_x_only", "slant_spacer", "bars_strong_oop",
               "bars_tiny_slant", "bars_both_promoted", "oop_both_sides",
               "bars_slant_both", "bars_near_inplane_oop")


#: Part A's family.  Every entry is solved on BOTH trees and its answer hashed.
FAMILY = (
    "conf2", "conf3", "conf_narrow_y", "conf_narrow_x",
    "x_only_ord", "x_only_narrow_x", "x_only_narrow_y", "x_only_mixed",
    "y_only_ord", "y_only_narrow_y", "y_only_narrow_x",
    "both_ord", "both_narrow_x", "both_narrow_y",
    "oop_x_only", "oop_both", "oop_narrow_y_conf", "oop_spacer",
    "oop_both_sides",
    "magnetic_x_only", "magnetic_narrow_y_conf", "magnetic_conf",
    "slant_spacer", "slant_spacer_narrow_y",
    "taper8", "taper9", "taper16",
    "advB_i", "advB_ii", "advB_iii", "advB_far", "advB_ymortar_narrow_x",
)

#: Fixtures ALSO solved with ``force_mortar=True`` (the test instrument that
#: drives the mortar algebra where the grids coincide -- BOTH axes live).
FORCED = ("conf2", "conf_narrow_y", "conf_narrow_x")


# =========================================================================
# the LADDERS -- is a narrow segment on THIS axis actually damaging?
# =========================================================================
def ladder(kind, g, M=4):
    """A stack whose swept wall is a PURE PARTITION choice: uniform
    permittivity everywhere, so the device CANNOT depend on ``g``.  Any
    movement of the answer as ``g`` shrinks is numerical damage.

    ``kind``:
      ``x_mortared``    the swept (x) axis carries the mortar -- POSITIVE ctrl
      ``y_conforming``  x carries the mortar, the swept y axis CONFORMS
      ``x_conforming``  y carries the mortar, the swept x axis CONFORMS
      ``no_mortar``     neither axis carries a mortar        -- NEGATIVE ctrl
      ``advB_i``        x mortared by a DIFFERENT wall, swept segment SHARED
      ``advB_ii``       three layers, the swept y segment shared, y mortared
                        only at the B-C interface
    """
    st = new_stack(M)
    xa, xb = sc(WXA), sc(WXB)
    ya, yb = sc(WYA), sc(WYB)
    if kind == "x_mortared":
        _add(st, 0.131e-6, narrow(g, XC), ya, kind="host")
        _add(st, 0.107e-6, xb, ya, kind="host")
    elif kind == "y_conforming":
        yn = narrow(g, YC)
        _add(st, 0.131e-6, xa, yn, kind="host")
        _add(st, 0.107e-6, xb, yn, kind="host")
    elif kind == "x_conforming":
        xn = narrow(g, XC)
        _add(st, 0.131e-6, xn, ya, kind="host")
        _add(st, 0.107e-6, xn, yb, kind="host")
    elif kind == "no_mortar":
        xn = narrow(g, XC)
        _add(st, 0.131e-6, xn, ya, kind="host")
        _add(st, 0.107e-6, xn, ya, kind="host")
    elif kind == "advB_i":
        x0 = sc((0.1400,)) + narrow(g, 0.7000 - g / 2)
        x1 = sc((0.2600,)) + narrow(g, 0.7000 - g / 2)
        y4 = sc(WY4)
        _add(st, 0.131e-6, x0, y4, kind="host")
        _add(st, 0.107e-6, x1, y4, kind="host")
    elif kind == "advB_ii":
        y_ab = sc((0.2100,)) + narrow(g, 0.6800 - g / 2)
        y_c = sc((0.3300,)) + narrow(g, 0.6800 - g / 2)
        x4a, x4b = sc(WX4A), sc(WX4B)
        _add(st, 0.111e-6, x4a, y_ab, kind="host")
        _add(st, 0.097e-6, x4b, y_ab, kind="host")
        _add(st, 0.089e-6, x4b, y_c, kind="host")
    else:
        raise ValueError(kind)
    st.set_source(WL, theta=THETA, phi=PHI)
    return st


LADDERS = ("x_mortared", "y_conforming", "x_conforming", "no_mortar",
           "advB_i", "advB_ii")
LADDER_G = (3.0e-1, 1.0e-1, 3.0e-2, 1.0e-2, 3.0e-3, 1.2e-3)


# =========================================================================
# the PATTERNED ladders -- the discriminating form
# =========================================================================
# :func:`ladder` above puts a UNIFORM slab pair either side of the swept wall,
# which is what the fix's own DEFECT-2 fixture does.  On such a device the
# modal content is nearly plane-wave and NOTHING moves anywhere -- including
# on the mortared axis -- so it cannot tell "the axis is harmless" from "the
# device is harmless".  These ladders keep the swept wall a PURE PARTITION
# choice but make the stack genuinely patterned, so the mortar carries real
# modal content and the POSITIVE control actually moves.
#
# Family X: the permittivity depends on x ONLY, so EVERY y wall is a pure
# partition choice.  Family Y is its transpose.
PXW0, PXW2 = (0.17, 0.61), (0.29, 0.74)
PORD = (0.335, 0.485)             # an ORDINARY middle-layer wall pair
PC = 0.41                         # the swept pair's centre in the mortar arms
PC2 = 0.52                        # ... and in the conforming arms
PT = (0.128e-6, 0.111e-6, 0.097e-6)
PEPS_P1, PEPS_P2 = 6.76, 5.29


def _ptile(nx, ny, axis, which):
    """A tile that depends on ONE axis only.  ``which`` = 0 (host), 1 or 2
    (the two pillars)."""
    c = np.full((nx, ny), _C(EPS_H))
    if which:
        e = _C(PEPS_P1 if which == 1 else PEPS_P2)
        if axis == "x":
            c[1, :] = e
        else:
            c[:, 1] = e
    return c


def _padd(st, t, xw, yw, axis, which):
    st.add_layer(t, eps_cell=_ptile(_n(xw), _n(yw), axis, which),
                 x_walls=xw, y_walls=yw)


def ladder_p(kind, g, M=4):                              # noqa: C901
    """The DISCRIMINATING ladder.  ``kind``:

      ``Xmort``    pattern x-only; layers differ on x; the SWEPT pair is the
                   middle layer's own x walls -- the axis IS mortared.
                   POSITIVE control.
      ``Xconf_y``  pattern x-only; layers differ on x; the swept pair is the
                   y wall array SHARED by all three layers.  The DEFECT-2
                   shape: round 3 warns y, round 4 is silent.
      ``Xnomort``  pattern x-only; ALL layers on the same x walls (no mortar
                   anywhere); the swept pair is the shared y wall array.
                   NEGATIVE control.
      ``Ymort``    the transpose of ``Xmort``.
      ``Yconf_x``  the transpose of ``Xconf_y`` -- round 3 warns x, round 4
                   is silent.  The MIRROR the fix never measured.
    """
    st = new_stack(M)
    if kind.startswith("X"):
        ax = "x"
        ywall = narrow(g, PC2) if kind in ("Xconf_y", "Xnomort") else sc(
            (0.31, 0.68))
        x0, x2 = sc(PXW0), sc(PXW2)
        if kind == "Xmort":
            x1 = narrow(g, PC)
        elif kind == "Xconf_y":
            x1 = sc(PORD)
        else:                                   # Xnomort
            x0 = x1 = x2 = sc(PXW0)
        _padd(st, PT[0], x0, ywall, ax, 1)
        _padd(st, PT[1], x1, ywall, ax, 0)
        _padd(st, PT[2], x2, ywall, ax, 2)
    elif kind.startswith("Y"):
        ax = "y"
        xwall = narrow(g, PC2) if kind == "Yconf_x" else sc((0.31, 0.68))
        y0, y2 = sc(PXW0), sc(PXW2)
        y1 = narrow(g, PC) if kind == "Ymort" else sc(PORD)
        _padd(st, PT[0], xwall, y0, ax, 1)
        _padd(st, PT[1], xwall, y1, ax, 0)
        _padd(st, PT[2], xwall, y2, ax, 2)
    else:
        raise ValueError(kind)
    st.set_source(WL, theta=THETA, phi=PHI)
    return st


LADDERS_P = ("Xmort", "Xconf_y", "Xnomort", "Ymort", "Yconf_x")


# =========================================================================
# the FIX's OWN DEFECT-2 device, swept on BOTH axes
# =========================================================================
# ``validation/probe_fix_mortar_round4/p1_axis_band.py`` part (a) sweeps the
# SHARED y wall pair of a two-layer stack of UNIFORM permittivity (2.5 beside
# 3.5) whose layers differ on x, and reports that the answer does not move.
# It never sweeps the MORTARED axis of the same device, which is the arm that
# says whether the device can move AT ALL.  ``fixdev('x', ...)`` is that arm.
# Knobs reproduced exactly; the length unit is metres here (period 1.07e-6,
# wavelength 0.79e-6 -- the same ratio, so the physics is identical).
FDP, FDWL, FDTH = 1.07e-6, 0.79e-6, 0.31
FDX0, FDX1 = (0.25, 0.60), (0.31, 0.66)
FDYC, FDXC = 0.585, 0.425


def fixdev(axis, g, M=5):
    """The fix's DEFECT-2 device.  ``axis='y'`` sweeps the SHARED y pair (the
    fix's own arm); ``axis='x'`` sweeps layer 0's own x pair, which IS the
    mortared axis."""
    st = PMM2DStackPure(FDP, n_modes=M, n_orders=1, layer_grids="per-layer")
    if axis == "y":
        yws = narrow(g, FDYC, FDP)
        x0, x1 = sc(FDX0, FDP), sc(FDX1, FDP)
    elif axis == "x":
        yws = sc((FDYC - 0.15, FDYC + 0.15), FDP)
        x0, x1 = narrow(g, FDXC, FDP), sc(FDX1, FDP)
    else:
        raise ValueError(axis)
    st.add_layer(0.17e-6, eps_cell=np.full((3, 3), _C(2.5)),
                 x_walls=x0, y_walls=yws)
    st.add_layer(0.17e-6, eps_cell=np.full((3, 3), _C(3.5)),
                 x_walls=x1, y_walls=yws)
    st.set_source(FDWL, theta=FDTH, phi=0.0)
    return st


FIXDEV = ("x", "y")


# =========================================================================
# the DEGRADATION LADDER against an EXACT 1-D ORACLE (task C, bar 2)
# =========================================================================
# The device is INVARIANT IN Y and driven at ``phi`` = 0, so the exact 1-D
# ``PMMStack`` -- a completely different assembly with no mortar and no
# element-grid partition -- is an INDEPENDENT oracle for it.  The middle
# layer is UNIFORM HOST, so its two walls are a pure PARTITION choice and the
# 1-D truth is the same at every swept width.
LP, LWL, LTH = 1.19e-6, 0.83e-6, 0.21
LEPS_H, LEPS_P = 2.25, 6.76
LW0 = (0.17, 0.61)
LW2 = (0.29, 0.74)
LYW = (0.31, 0.68)
LT = 0.128e-6
LXC = 0.41


def ladder_oracle(deg, far=5):
    from lumenairy.elements.pmm import PMMStack  # noqa: PLC0415
    st = PMMStack(LP, degree=deg, far_field_orders=far)
    st.add_layer(LT, segments=[(LW0[0], LEPS_H),
                               (LW0[1] - LW0[0], LEPS_P),
                               (1.0 - LW0[1], LEPS_H)])
    st.add_layer(LT, segments=[(1.0, LEPS_H)])
    st.add_layer(LT, segments=[(LW2[0], LEPS_H),
                               (LW2[1] - LW2[0], LEPS_P),
                               (1.0 - LW2[1], LEPS_H)])
    st.set_source(LWL, theta=LTH)
    o, R, T = st.solve(stabilize=None)[:3]
    return np.asarray(o), np.atleast_2d(R), np.atleast_2d(T)


def _ltile(pillar):
    """3x3 tile, INVARIANT IN Y: column ``i`` only."""
    c = np.full((3, 3), _C(LEPS_H))
    if pillar:
        c[1, :] = _C(LEPS_P)
    return c


def ladder_2d(frac, M):
    """The same device on the 2-D pure staggered per-layer stack, with the
    MIDDLE layer's two x walls ``frac`` of the period apart.  x carries the
    mortar; the y wall array is SHARED and ordinary."""
    st = PMM2DStackPure(LP, n_modes=M, n_orders=1, n_superstrate=1.0,
                        n_substrate=1.0, layer_grids="per-layer")
    yws = [LYW[0] * LP, LYW[1] * LP]
    st.add_layer(LT, eps_cell=_ltile(True),
                 x_walls=[LW0[0] * LP, LW0[1] * LP], y_walls=yws)
    st.add_layer(LT, eps_cell=_ltile(False),
                 x_walls=[(LXC - frac / 2) * LP, (LXC + frac / 2) * LP],
                 y_walls=yws)
    st.add_layer(LT, eps_cell=_ltile(True),
                 x_walls=[LW2[0] * LP, LW2[1] * LP], y_walls=yws)
    st.set_source(LWL, theta=LTH, phi=0.0)
    return st


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
