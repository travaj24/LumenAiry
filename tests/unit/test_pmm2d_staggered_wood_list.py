"""The Wood-anomaly permittivity list of the PURE staggered 2-D PMM is ONE
rule for the scalar and the tensor path (2026-09-10).

``_grazing_safe_wavelength`` nudges the wavelength off an EXACT Rayleigh
coincidence in any medium whose real permittivity is on its list.  Until this
change the TENSOR path (``pmm_jones_2d_staggered`` / a tensor layer of
``PMM2DStackPure``) listed the layer tensors' diagonals while the SCALAR path
(``pmm_efficiency_2d_staggered`` / a scalar layer) listed only the two
half-spaces -- so a scalar cell and its ``e * I`` promotion, the SAME
discretization everywhere else, took different nudges and gave different
answers when an order sat exactly on a LAYER's own cut-off
(``docs/audits/VERIFY_PMM2D_STAGGERED_ANISOTROPIC_2026_09_09.md`` limitation 2,
measured 4.591e-08; build-doc open item 6).  The staggered solver degrades like
``~1/sqrt(distance)`` near a cut-off INSIDE the layer too, so listing the layer
permittivity is the more robust convention, and it is now what BOTH paths do.

Everything below is asserted as a DECISION (nudged / not nudged, bit-identical
/ not) or against a bar derived from the run's own numbers -- see
``docs/TESTING_STANDARDS.md``.  The on-cut-off state is CONSTRUCTED from the
running build's geometry (``wl = px * sqrt(eps_layer)`` puts order ``m = 1``
exactly at ``kz = 0`` inside the layer), never hoped for.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pytest

from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import (
    _wood_eps_reals,
    pmm_efficiency_2d_staggered,
    pmm_jones_2d_staggered,
)
from lumenairy.elements.rcwa._core import _grazing_safe_wavelength

# --- the constructed on-cut-off geometry -----------------------------------
# Normal incidence, so kt(m, n) = sqrt(m^2 + n^2) * wl / px; wl = px*sqrt(EPS_L)
# puts (m, n) = (+/-1, 0) and (0, +/-1) exactly at kt^2 = EPS_L, i.e. grazing
# INSIDE the layer.  The half-spaces (eps 1.0 and 2.25) are nowhere near it, so
# the OLD (half-spaces-only) rule leaves this wavelength alone -- which is what
# makes the fixture discriminating.
_PX = 0.5e-6
_EPS_L = 4.0
_N_SUP, _N_SUB = 1.0, 1.5
_WL_CUT = _PX * np.sqrt(_EPS_L)
_DEPTH = 0.28e-6
_M, _NO = 5, 3

_UNIFORM_CELL = np.full((2, 2), _EPS_L + 0j)
_PATTERNED_CELL = np.array([[_EPS_L, 1.0], [1.0, 1.0]], dtype=complex)

# ordinary (off-cut-off) fixtures: the geometries the shipped staggered suites
# use -- (px, py, wl, cell, n_sup, n_sub, theta, phi)
_ORDINARY = [
    (0.8e-6, 0.8e-6, 0.633e-6, np.array([[2.25, 1.0], [1.0, 1.0]]), 1.0, 1.45,
     0.0, 0.0),
    (0.8e-6, 0.8e-6, 0.633e-6, np.array([[2.25, 1.0], [1.0, 1.0]]), 1.0, 1.45,
     0.25, 0.4),
    (0.8e-6, 0.6e-6, 0.55e-6, np.array([[4.0, 1.0], [1.0, 2.25]]), 1.0, 1.0,
     0.2, 0.0),
    (1.2e-6, 1.2e-6, 1.55e-6, np.array([[12.0, 1.0], [1.0, 1.0]]), 1.0, 3.48,
     0.1, 0.9),
    (0.7e-6, 0.7e-6, 0.55e-6, np.full((2, 2), 4.0), 1.0, 1.5, 0.0, 0.0),
    (0.5e-6, 0.5e-6, 0.633e-6, np.array([[2.25 + 0.1j, 1.0], [1.0, 4.0]]),
     1.0, 1.5, 0.15, 0.3),
]


def _orders(n_orders):
    mo = np.arange(-int(n_orders), int(n_orders) + 1)
    return np.tile(mo, len(mo)), np.repeat(mo, len(mo))


def _nudge(wl, eps_list, *, px=_PX, py=None, n_sup=_N_SUP, theta=0.0, phi=0.0,
           n_orders=_NO):
    """The library's own guard, driven with an EXPLICIT permittivity list --
    this is how the two rules (with / without the layer eps) are compared
    without reaching inside a solver."""
    mx, my = _orders(n_orders)
    kx0 = float(np.real(n_sup)) * np.sin(theta) * np.cos(phi)
    ky0 = float(np.real(n_sup)) * np.sin(theta) * np.sin(phi)
    return _grazing_safe_wavelength(float(wl), kx0, ky0, mx, my, px,
                                    px if py is None else py, list(eps_list))


def _promote(m):
    t = np.zeros(np.shape(m) + (3, 3), dtype=complex)
    for i in range(3):
        t[..., i, i] = m
    return t


def _stack_solve(wl, *, cell=None, uniform=None, theta=0.0, phi=0.0,
                 px=_PX, py=None, n_sup=_N_SUP, n_sub=_N_SUB, depth=_DEPTH):
    s = PMM2DStackPure(px, px if py is None else py, n_superstrate=n_sup,
                       n_substrate=n_sub, n_modes=_M, n_orders=_NO)
    if uniform is not None:
        s.add_layer(depth, eps=uniform)
    if cell is not None:
        s.add_layer(depth, eps_cell=cell)
    s.set_source(float(wl), theta=theta, phi=phi)
    _o, R, T, _J = s.solve(jones=True)
    return R, T


def _eff_solve(wl, cell, *, px=_PX, polarization="tm", theta=0.0, phi=0.0):
    _o, R, T = pmm_efficiency_2d_staggered(
        px, px, cell, _N_SUB, _N_SUP, _DEPTH, float(wl), degree=_M,
        n_orders=_NO, polarization=polarization, theta=theta, phi=phi)
    return R, T


# ===================================================================
# 1.  The fixture really is on a LAYER cut-off, and the two RULES differ there
# ===================================================================
def test_the_two_rules_are_distinguishable_on_a_layer_cutoff():
    """FAIL-BEFORE, stated as a decision about the rules themselves.

    The OLD rule (half-spaces only) returns the on-cut-off wavelength
    UNCHANGED -- it cannot see a grazing layer mode at all -- while the rule
    that includes the layer permittivity moves it.  Both arms are the library's
    own guard called with the two lists, so neither references pre-fix code nor
    any recorded number.
    """
    eps_half = [_N_SUP ** 2, _N_SUB ** 2]
    # the state is exactly on the cut-off, by construction and exactly in
    # float64: kt^2 - eps_layer == 0.0, not "small"
    assert (_WL_CUT / _PX) ** 2 - _EPS_L == 0.0

    old = _nudge(_WL_CUT, eps_half)
    new = _nudge(_WL_CUT, eps_half + [_EPS_L])
    assert old == _WL_CUT, "the half-spaces-only rule must not move this one"
    assert new != _WL_CUT, "the unified rule must move an exact layer grazing"
    # and the two half-spaces are genuinely far from the coincidence, so the
    # fixture isolates the LAYER contribution
    kt2 = np.array([1.0, 2.0]) * (_WL_CUT / _PX) ** 2
    assert min(abs(e - k) for e in eps_half for k in kt2) > 1e-3


# ===================================================================
# 2.  Every SCALAR site now takes that nudge -- observed through the API
# ===================================================================
@pytest.mark.parametrize("kind", ["patterned", "uniform", "single_layer"])
def test_every_scalar_site_takes_the_layer_cutoff_nudge(kind):
    """A solve AT the exact cut-off is BIT-IDENTICAL to the same solve at the
    nudged wavelength -- the public-surface proof that the guard fired and
    landed exactly where the unified rule says.

    The target wavelength is DERIVED by calling the guard (never by pinning its
    internal nudge constant), so this tracks any legitimate change of the
    nudge.
    """
    eps_list = [_N_SUP ** 2, _N_SUB ** 2, _EPS_L]
    wl_nudged = _nudge(_WL_CUT, eps_list)
    assert wl_nudged != _WL_CUT      # non-vacuous: there IS something to fire

    if kind == "patterned":
        at_cut = _stack_solve(_WL_CUT, cell=_PATTERNED_CELL)
        at_nudged = _stack_solve(wl_nudged, cell=_PATTERNED_CELL)
    elif kind == "uniform":
        # a UNIFORM SCALAR layer (kind="uniform"): eps = 4 differs from both
        # half-spaces, so only the layer can put this order on a cut-off
        at_cut = _stack_solve(_WL_CUT, uniform=_EPS_L + 0j)
        at_nudged = _stack_solve(wl_nudged, uniform=_EPS_L + 0j)
    else:
        at_cut = _eff_solve(_WL_CUT, _UNIFORM_CELL)
        at_nudged = _eff_solve(wl_nudged, _UNIFORM_CELL)

    assert np.array_equal(at_cut[0], at_nudged[0])
    assert np.array_equal(at_cut[1], at_nudged[1])


# ===================================================================
# 3.  The scalar cell and its e*I promotion agree ON the cut-off
# ===================================================================
@pytest.mark.parametrize("wl_rel,label", [(1.0, "on the layer cut-off"),
                                          (1 - 1e-9, "1e-9 below it"),
                                          (0.55, "far from any cut-off")])
def test_scalar_cell_equals_its_tensor_promotion(wl_rel, label):
    """G1's promotion identity, now unconditional in the wavelength.

    ``PMM2DStackPure`` with a scalar ``(Nx, Ny)`` cell and with its
    ``e * I`` promotion are the SAME discretization -- bit-identical, not
    merely close.  Before the rule was unified this failed at ``wl_rel = 1``
    (measured 7.561e-09 on this fixture, 4.591e-08 on the uniform one) while
    passing on the other two.
    """
    wl = _WL_CUT * wl_rel
    Rs, Ts = _stack_solve(wl, cell=_PATTERNED_CELL)
    Rt, Tt = _stack_solve(wl, cell=_promote(_PATTERNED_CELL))
    assert np.array_equal(Rs, Rt), f"{label}: max|dR| = {np.max(np.abs(Rs - Rt)):.3e}"
    assert np.array_equal(Ts, Tt), f"{label}: max|dT| = {np.max(np.abs(Ts - Tt)):.3e}"


def test_scalar_entry_equals_the_tensor_entry_on_a_layer_cutoff():
    """The verification report's own reproducer, across the two ENTRY POINTS:
    ``pmm_efficiency_2d_staggered`` (TM) vs ``pmm_jones_2d_staggered`` on the
    promoted cell, uniform ``eps = 4`` layer, order ``m = 1`` exactly at its
    cut-off.  The report measured max|dR| = max|dT| = 4.591e-08 there; both are
    exactly zero once the two paths take the same nudge."""
    R1, T1 = _eff_solve(_WL_CUT, _UNIFORM_CELL, polarization="tm")
    _o, R2, T2, _J = pmm_jones_2d_staggered(
        _PX, _PX, _promote(_UNIFORM_CELL), _N_SUB, _N_SUP, _DEPTH, _WL_CUT,
        degree=_M, n_orders=_NO)
    assert np.array_equal(R1, R2[0])
    assert np.array_equal(T1, T2[0])


# ===================================================================
# 4.  The nudge is consequential -- the rule difference is not noise
# ===================================================================
def test_the_layer_cutoff_nudge_is_consequential():
    """Two-sided sizing: the wavelength the two rules disagree about changes
    the ANSWER by decades more than this solve's arithmetic floor, so unifying
    the rule is a real decision and not a cosmetic one.

    Bar DERIVED from the run: ``1e3 * max|R| * eps_machine`` -- a thousand ULP
    of the largest reflectance the same solve produces (2.6e-12 here).  The
    signal measured 2026-09-10 (Win py3.14 / np2.4.4, OpenBLAS 1 thread) is
    7.6e-09, i.e. ~2.9e3x above that floor; the comparison wavelength is
    ``WL_CUT*(1 - 1e-9)``, which the guard leaves alone (its trigger band is
    ``|eps - kt^2| <= 1e-9``, ~1.2e-10 in relative wavelength), so this is the
    un-nudged answer the half-spaces-only rule used to return.
    """
    wl_nudged = _nudge(_WL_CUT, [_N_SUP ** 2, _N_SUB ** 2, _EPS_L])
    wl_below = _WL_CUT * (1 - 1e-9)
    assert _nudge(wl_below, [_N_SUP ** 2, _N_SUB ** 2, _EPS_L]) == wl_below

    Rn, _Tn = _stack_solve(wl_nudged, cell=_PATTERNED_CELL)
    Rb, _Tb = _stack_solve(wl_below, cell=_PATTERNED_CELL)
    signal = float(np.max(np.abs(Rn - Rb)))
    floor = 1e3 * float(np.max(np.abs(Rn))) * float(np.finfo(float).eps)
    assert signal > 100.0 * floor, (signal, floor)


# ===================================================================
# 5.  Nothing moves OFF a cut-off
# ===================================================================
@pytest.mark.parametrize("fx", _ORDINARY,
                         ids=[f"ord{i}" for i in range(len(_ORDINARY))])
def test_the_layer_eps_rule_moves_nothing_off_a_cutoff(fx):
    """The regression contract for every shipped scalar result: on an ordinary
    geometry the guard returns the wavelength UNCHANGED under the unified rule,
    exactly as it did under the half-spaces-only one.  An identical wavelength
    makes every downstream byte identical -- the claim is therefore about the
    rule, not about one build's R/T (which were separately confirmed
    bit-identical against the pre-fix clone on 2026-09-10, 11 fixtures)."""
    px, py, wl, cell, n_sup, n_sub, theta, phi = fx
    eps_half = [n_sup ** 2, n_sub ** 2]
    eps_all = _wood_eps_reals(n_sup ** 2, n_sub ** 2, cell)
    old = _nudge(wl, eps_half, px=px, py=py, n_sup=n_sup, theta=theta, phi=phi)
    new = _nudge(wl, eps_all, px=px, py=py, n_sup=n_sup, theta=theta, phi=phi)
    assert old == wl          # the shipped rule did nothing here ...
    assert new == wl          # ... and neither does the unified one


def test_scalar_and_promoted_agree_on_the_ordinary_fixtures():
    """The promotion identity on the ordinary geometries -- the property that
    would break first if the unified list were applied inconsistently."""
    for px, py, wl, cell, n_sup, n_sub, theta, phi in _ORDINARY[:3]:
        Rs, Ts = _stack_solve(wl, cell=np.asarray(cell, dtype=complex),
                              px=px, py=py, n_sup=n_sup, n_sub=n_sub,
                              theta=theta, phi=phi)
        Rt, Tt = _stack_solve(wl, cell=_promote(np.asarray(cell,
                                                           dtype=complex)),
                              px=px, py=py, n_sup=n_sup, n_sub=n_sub,
                              theta=theta, phi=phi)
        assert np.array_equal(Rs, Rt)
        assert np.array_equal(Ts, Tt)


# ===================================================================
# 6.  The cut-off WARNING keys on the half-spaces, unchanged
# ===================================================================
def test_cutoff_warning_still_keys_on_the_half_spaces_only():
    """Two-sided.  The warning is about the RAYLEIGH orders a caller can see in
    the far field, so it keys on the half-spaces -- a LAYER-only coincidence is
    silent (the nudge has already handled it) while a half-space one still
    fires.  Unifying the nudge list did not move this boundary."""
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _eff_solve(_WL_CUT, _UNIFORM_CELL)          # LAYER cut-off only
    assert [x for x in w if "Rayleigh cut" in str(x.message)] == []

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # wl = px puts m = 1 on the eps = 1 SUPERSTRATE cut-off
        pmm_efficiency_2d_staggered(0.8e-6, 0.8e-6, _PATTERNED_CELL, 1.0, 1.0,
                                    0.3e-6, 0.8e-6, degree=_M, n_orders=_NO,
                                    polarization="te")
    assert len([x for x in w if "Rayleigh cut" in str(x.message)]) == 1


# ===================================================================
# 7.  The list builder itself
# ===================================================================
def test_wood_eps_reals_is_numerically_inert_and_bounded():
    """Deduplication may not change the nudge (the guard takes a MIN over the
    list) and must keep the list O(materials): a fine cell would otherwise put
    3*Nx*Ny entries through a Python-level min per candidate wavelength
    (measured 2026-09-10: 46.9 ms/call at 64x64 raw vs 0.02 ms deduped)."""
    rng = np.random.default_rng(0)
    cell = np.where(rng.random((16, 16)) > 0.5, 4.0 + 0.2j, 1.0 + 0j)
    tensor = _promote(cell)
    raw = list(np.real(tensor[..., [0, 1, 2], [0, 1, 2]].ravel()))
    deduped = _wood_eps_reals(_N_SUP ** 2, _N_SUB ** 2,
                              tensor[..., [0, 1, 2], [0, 1, 2]])
    # {1.0 (superstrate AND the cell's background), 2.25 (substrate), 4.0}
    assert len(deduped) == 3
    assert len(raw) == 3 * 16 * 16
    for wl in (_WL_CUT, _WL_CUT * (1 - 1e-9), 0.633e-6, 0.8e-6):
        assert _nudge(wl, [_N_SUP ** 2, _N_SUB ** 2] + raw) == _nudge(wl,
                                                                      deduped)
    # a SCALAR (Nx, Ny) cell and its tensor promotion give the same list --
    # the property the two paths' agreement rests on
    assert _wood_eps_reals(cell) == _wood_eps_reals(
        tensor[..., [0, 1, 2], [0, 1, 2]])
    assert _wood_eps_reals() == []
