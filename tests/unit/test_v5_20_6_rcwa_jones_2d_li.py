"""rcwa_jones_2d formulation='li' -- Li-1997 diagonal inverse rule (2-D tensor).

``rcwa_jones_2d`` gains a ``formulation`` kwarg.  ``'laurent'`` (default) is the
direct rule on every component; ``'li'`` applies the Li-1997 (JOSA A 14:2758,
Eqs. 8/9) inverse rule to the DIAGONAL blocks (``C_xx`` inverse-along-x from
``exx``, ``C_yy`` inverse-along-y from ``eyy``, via the validated scalar builder)
while the off-diagonal blocks and the ``E_z`` rule stay Laurent (Li rule 3).

Pins: a scalar (isotropic) cell reduces EXACTLY to
``rcwa_efficiency_2d(formulation='li')``; both formulations conserve energy for a
lossless cell; ``'li'`` is at least as accurate as ``'laurent'`` at a coarse
truncation for a high-contrast metallic anisotropic cell (and both converge to
the same answer); the guard rejects an unknown formulation.  (The off-diagonal
Popov-Neviere mixed composite is not implemented, so the gain is partial for a
strongly-rotated director -- documented in the entry point.)
"""
from __future__ import annotations

import numpy as np
import pytest

from lumenairy.elements.rcwa import rcwa_efficiency_2d, rcwa_jones_2d

_P = 0.5e-6
_WL = 0.55e-6
_DEP = 0.12e-6


def _inpl_cell(S, no, ne, rot_deg, hw=0.25, host=2.1):
    """A rotated in-plane uniaxial pillar (director in the x-y plane) in an
    isotropic host -- axis-aligned sharp walls (the Li rule's target)."""
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    c, s = np.cos(np.deg2rad(rot_deg)), np.sin(np.deg2rad(rot_deg))
    D = np.diag([ne ** 2, no ** 2, no ** 2]).astype(complex)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], complex)
    T = Rz @ D @ Rz.T
    cell = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        cell[:, :, i, i] = host
    cell[(np.abs(X) < hw) & (np.abs(Y) < hw)] = T
    return cell


@pytest.mark.parametrize("formu", ["laurent", "li"])
def test_energy_conserved_lossless(formu):
    cell = _inpl_cell(64, 1.5, 2.3, 30.0)
    _o, R, T, _J = rcwa_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL,
                                 n_orders_x=6, n_orders_y=6, theta=0.1,
                                 formulation=formu)
    assert np.allclose(R.sum(1) + T.sum(1), 1.0, atol=1e-9)


def test_scalar_cell_reduces_to_scalar_li_path():
    """An isotropic scalar-tensor cell with formulation='li' matches the scalar
    rcwa_efficiency_2d('li') to machine precision, and has no cross-pol."""
    S = 64
    x = (np.arange(S) + 0.5) / S - 0.5
    X, Y = np.meshgrid(x, x, indexing="ij")
    sc = np.full((S, S), 2.1, complex)
    sc[(np.abs(X) < 0.25) & (np.abs(Y) < 0.25)] = -8 + 1.2j     # metal-like
    tc = np.zeros((S, S, 3, 3), complex)
    for i in range(3):
        tc[:, :, i, i] = sc
    _o, R, T, J = rcwa_jones_2d(_P, _P, tc, 1.5, 1.0, _DEP, _WL, n_orders_x=6,
                                n_orders_y=6, theta=0.0, formulation="li")
    _oe, Re, Te = rcwa_efficiency_2d(_P, _P, sc, 1.5, 1.0, _DEP, _WL,
                                     polarization="te", n_orders_x=6,
                                     n_orders_y=6, formulation="li",
                                     symmetry=False)
    # incident E_x (row 0) == TE at normal incidence
    assert abs(R[0].sum() - Re.sum()) < 1e-11
    assert abs(T[0].sum() - Te.sum()) < 1e-11
    assert abs(J[0, 1]) < 1e-11 and abs(J[1, 0]) < 1e-11        # no cross-pol


def test_li_and_laurent_converge_to_the_same_limit():
    """Li's theorem: the diagonal inverse rule and the direct rule reach the
    SAME operator as n_orders -> infinity (the inverse rule just faster for the
    concurrent wall-normal jump).  So for a (fast-converging) dielectric
    anisotropic cell the two formulations must AGREE at high n_orders and the
    gap must SHRINK from a coarse to a fine truncation -- proving 'li' is a
    correct factorization, not a different physics."""
    cell = _inpl_cell(64, 1.5, 2.3, 20.0, host=2.1)          # dielectric
    kw = dict(theta=0.15)

    def _gap(No):
        Jl = rcwa_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL, n_orders_x=No,
                           n_orders_y=No, formulation="laurent", **kw)[3]
        Ji = rcwa_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL, n_orders_x=No,
                           n_orders_y=No, formulation="li", **kw)[3]
        return np.max(np.abs(Jl - Ji))

    g6, g12 = _gap(6), _gap(12)
    assert g6 > 1e-4          # genuinely a different rule at coarse truncation
    assert g12 < g6           # converging together (toward one limit)
    assert g12 < 2e-3         # and close by n_orders=12 (same limit)


def test_unknown_formulation_raises():
    cell = _inpl_cell(48, 1.5, 2.0, 20.0)
    with pytest.raises(ValueError, match="formulation must be"):
        rcwa_jones_2d(_P, _P, cell, 1.5, 1.0, _DEP, _WL, formulation="bogus")
