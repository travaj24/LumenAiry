"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 Territory M -- PMM findings M2, M3, M9.

M2 (HIGH, silent-wrong-input)
    ``PMMStack.add_layer(segments=...)`` never validated the width FRACTIONS.
    ``_pmm_union_grid`` normalises with ``cw[-1] = 1.0``, so a bad list was
    silently CLIPPED rather than rejected.  Measured before the fix
    (P = 0.5 um, degree 8, far_field_orders 5, eps 6.25 / 1.0, 0.2 um thick):

    * ``segments=[(0.7, r), (0.7, g)]`` (sum 1.4) solved BIT-IDENTICALLY to the
      clipped ``[(0.7, r), (0.3, g)]`` structure -- same ``R``, ``T`` and
      ``jones`` arrays (``J00 = -0.3732351345+0.215887036j`` for both), while
      the intended 50/50 cell gives ``J00 = -0.1174198748+0.186957736j``;
    * METRE-valued widths ``[(0.25e-6, r), (0.25e-6, g)]`` on a 0.5e-6 period
      solved as a physically unrelated near-uniform slab
      (``J00 = 0.1337615236+0.1473863644j``) with ``R+T = [0.9988, 1.0012]``:
      energy-clean, so no tripwire could see it.

    The 1-D sibling ``pmm_jones_1d_segments`` already raised on the same input
    via ``_segment_walls`` ("width fractions must sum to 1").

M3 (HIGH, silent gain / NaN)
    The classical (``phi = 0``) ``PMMStack`` cascade was the ONE PMM solve path
    without the ``_require_propagating_incidence`` guard the other 26 sites
    carry.  Measured before the fix, same stack:

    * ``n_superstrate = 1 - 1e-9j`` (infinitesimal GAIN) -> ``R+T =
      [-0.951, -0.816]`` with NO warning (``_kz_forward`` takes its ``Re < 0``
      root, so ``kz_inc < 0`` negates every efficiency);
    * ``n_superstrate = 1 - 0.01j`` -> ``R+T = [-0.943, -0.822]``, silent;
    * ``n_substrate = nan`` -> ``R+T = [nan, nan]``, silent (only numpy
      RuntimeWarnings);
    * ``n_superstrate = 0.2 + 6j`` (non-propagating) -> ``R+T = [32.5, 55.4]``
      (that one did warn);

    while the CONICAL (``phi != 0``) path already raised on the gain case.
    ``_warn_stack_energy`` was one-sided (``> 1``) and NaN-blind, where RCWA's
    ``_check_energy`` raises on non-finite and on negative totals.

M9 (MEDIUM, ownership)
    ``_cached_geo_eig`` handed its cached eigen-arrays out BY IDENTITY and
    WRITABLE, so an in-place write on a returned array poisoned the cache for
    every later solve on the same geometry (measured: ``w[0, 0] += 1`` changed
    the value the next two cache hits saw).  The module's own ``_readonly``
    guard was already applied at its other cache sites.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import PMMStack
from lumenairy.elements.pmm._core import (
    _build_sem_tensor_segments,
    _cached_geo_eig,
    _clear_geo_eig_cache,
    _tensor3_dict,
    _uniform_geo_eig,
)
from lumenairy.elements.pmm.oned import pmm_jones_1d_segments

P = 0.5e-6
WL = 0.633e-6
DEG = 8
FFO = 5
EPS_R = 6.25 + 0j
EPS_G = 1.0 + 0j
THK = 0.2e-6


def _stack(*, n_sub=1.5, n_sup=1.0):
    return PMMStack(P, n_substrate=n_sub, n_superstrate=n_sup, degree=DEG,
                    far_field_orders=FFO)


# ===========================================================================
# M2 -- segment width fractions are validated (the 1-D sibling's check)
# ===========================================================================

def test_m2_segment_fractions_must_sum_to_one():
    """The sum-1.4 list that used to be silently CLIPPED now raises, with the
    same 1e-6 tolerance as the 1-D sibling ``_segment_walls``."""
    with pytest.raises(ValueError, match="must sum to 1"):
        _stack().add_layer(THK, segments=[(0.7, EPS_R), (0.7, EPS_G)])
    # the 1-D sibling raises on the identical input (the pattern being matched)
    with pytest.raises(ValueError, match="sum to 1"):
        pmm_jones_1d_segments(P, [(0.7, 2.5), (0.7, 1.0)], 1.5, 1.0, THK, WL,
                              angle=0.0, degree=DEG, far_field_orders=FFO)


def test_m2_metre_valued_widths_raise_and_name_the_misuse():
    """Absolute (metre-valued) widths -- the audited misuse -- raise, and the
    message says they must be fractions of the period."""
    with pytest.raises(ValueError, match="must sum to 1") as ei:
        _stack().add_layer(THK, segments=[(0.25e-6, EPS_R), (0.25e-6, EPS_G)])
    msg = str(ei.value)
    assert "fractions of the period" in msg
    assert "NOT" in msg and "absolute widths" in msg


def test_m2_nonpositive_and_nonfinite_widths_raise():
    for bad in ([(0.0, EPS_R), (1.0, EPS_G)],
                [(-0.2, EPS_R), (1.2, EPS_G)],
                [(float("nan"), EPS_R), (1.0, EPS_G)]):
        with pytest.raises(ValueError, match="width"):
            _stack().add_layer(THK, segments=bad)


def test_m2_valid_fractions_unchanged_bit_for_bit():
    """A VALID fraction list is untouched: the frozen pre-fix values (recorded
    on this exact configuration) must still come out unchanged.  Held to rel
    1e-10, not exact equality: the solve is bit-reproducible on one machine
    but drifts ~4e-14 relative across platforms/BLAS builds (measured: CI
    Linux vs the Windows box the values were frozen on); a validation gate
    with any numeric side effect would move these by far more than 1e-10."""
    st = _stack()
    st.add_layer(THK, segments=[(0.5, EPS_R), (0.5, EPS_G)])
    orders, R, T, J = st.set_source(WL, angle=0.0).solve()
    assert complex(J[0, 0]) == pytest.approx(
        complex(-0.11741987478962615, 0.18695773604145147), rel=1e-10)
    assert complex(J[1, 1]) == pytest.approx(
        complex(-0.41864855426106673, -0.09161162775657859), rel=1e-10)
    assert float(R.sum()) == pytest.approx(0.23239992438644771, rel=1e-10)
    assert float(T.sum()) == pytest.approx(1.7676000851170701, rel=1e-10)
    # a 3-segment list whose float sum is 1 only to round-off still passes
    edge = 0.5 * (1.0 - 0.4)
    st2 = _stack()
    st2.add_layer(THK, segments=[(edge, EPS_G), (0.4, EPS_R), (edge, EPS_G)])
    assert len(st2._layers) == 1


def test_m2_internal_builders_still_pass_the_new_check():
    """``add_tapered_grating`` / ``add_tapered_ridges`` build their own segment
    lists -- their float sums must stay inside the 1e-6 tolerance."""
    st = _stack()
    st.add_tapered_grating(0.3e-6, eps_ridge=EPS_R, eps_groove=EPS_G,
                           duty_bottom=0.7, duty_top=0.3, n_slices=5)
    assert len(st._layers) == 5
    st2 = _stack()
    st2.add_tapered_ridges(
        0.3e-6, ridges=[(0.12e-6, 0.10e-6, 0.14e-6, EPS_R),
                        (0.36e-6, 0.08e-6, 0.06e-6, EPS_R)],
        eps_groove=EPS_G, n_slices=3)
    assert len(st2._layers) == 3


# ===========================================================================
# M3 -- gain / non-propagating / NaN guards on the CLASSICAL path
# ===========================================================================

def _solve_classical(*, n_sub=1.5, n_sup=1.0, phi=None, angle=0.0):
    st = _stack(n_sub=n_sub, n_sup=n_sup)
    st.add_layer(THK, segments=[(0.5, EPS_R), (0.5, EPS_G)])
    kw = {} if phi is None else dict(phi=phi)
    st.set_source(WL, angle=angle, **kw)
    return st.solve()


@pytest.mark.parametrize("n_sup", [1 - 1e-9j, 1 - 0.01j])
def test_m3_classical_gain_superstrate_raises(n_sup):
    """Both audited gain cases (infinitesimal and finite) now raise instead of
    silently returning a NEGATIVE total."""
    with pytest.raises(ValueError, match="gain incidence medium"):
        _solve_classical(n_sup=n_sup)


def test_m3_classical_nonpropagating_superstrate_raises():
    with pytest.raises(ValueError, match="non-propagating"):
        _solve_classical(n_sup=0.2 + 6.0j)


def test_m3_classical_nan_substrate_raises():
    """The NaN substrate that used to return ``R+T = [nan, nan]`` silently is
    caught by the now NaN-aware energy tripwire."""
    with np.errstate(invalid="ignore", divide="ignore"):
        with pytest.raises(ValueError, match="non-finite total efficiency"):
            _solve_classical(n_sub=float("nan"))


def test_m3_legitimate_lossy_halfspaces_still_solve_silently():
    """The guards must not fire on physical LOSS (the continuity requirement):
    a lossy substrate and a lossy superstrate both still solve, silently, with
    sane totals."""
    for kw in (dict(n_sub=1.5 + 0.05j), dict(n_sup=1.0 + 0.01j)):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            _o, R, T, _J = _solve_classical(**kw)
        assert not wl, [str(w.message) for w in wl]
        tot = np.real(R).sum(axis=-1) + np.real(T).sum(axis=-1)
        assert np.all(tot > 0.9) and np.all(tot < 1.01 + 1e-9)


def test_m3_classical_and_conical_agree_on_the_gain_rejection():
    """Consistency: the conical path already raised; the classical path now
    raises the same way (the audited 'guard at 26 of 27 sites' pattern)."""
    with pytest.raises(ValueError, match="gain incidence medium"):
        _solve_classical(n_sup=1 - 1e-9j)
    with pytest.raises(ValueError, match="gain incidence medium"):
        _solve_classical(n_sup=1 - 1e-9j, phi=0.3, angle=0.2)


def test_m3_energy_tripwire_is_two_sided_and_nan_aware():
    """Unit-level pin on ``_warn_stack_energy``'s three severities (RCWA's
    ``_check_energy`` semantics, replicated not imported)."""
    from lumenairy.elements.pmm.stack import _warn_stack_energy

    ok = np.array([[0.4, 0.1], [0.3, 0.2]])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        _warn_stack_energy(ok, ok)                 # tot = 1.0 -> silent
    assert not wl

    with pytest.raises(ValueError, match="non-finite total efficiency"):
        _warn_stack_energy(np.array([[np.nan, 0.0]]), np.array([[0.5, 0.0]]))
    with pytest.raises(ValueError, match="NEGATIVE total efficiency"):
        _warn_stack_energy(np.array([[-0.6, 0.0]]), np.array([[0.1, 0.0]]))
    with pytest.warns(UserWarning, match="energy not conserved"):
        _warn_stack_energy(np.array([[1.2, 0.0]]), np.array([[0.5, 0.0]]))


# ===========================================================================
# M9 -- the geometric-eig cache hands out READ-ONLY arrays
# ===========================================================================

def _geo_mats():
    t = _tensor3_dict(np.eye(3) * (1.0 + 0j))
    return _build_sem_tensor_segments(P, [0.5, 0.5], [t, t], 6, 1, True)


def test_m9_geo_eig_cache_values_are_readonly():
    _clear_geo_eig_cache()
    try:
        mats = _geo_mats()
        k0 = 2.0 * np.pi / WL
        mu1, w1, _K1 = _uniform_geo_eig(mats, k0, 0.0)
        mu2, w2, _K2 = _uniform_geo_eig(mats, k0, 0.0)      # cache HIT
        assert w1 is w2, "the cache is expected to hand values out by identity"
        assert not w1.flags.writeable
        with pytest.raises(ValueError):
            w1[0, 0] += 1.0
        # the derived (scaled) outputs stay writable -- they are fresh arrays
        assert mu1.flags.writeable and mu2.flags.writeable
    finally:
        _clear_geo_eig_cache()


def test_m9_cached_geo_eig_marks_every_ndarray_member():
    _clear_geo_eig_cache()
    try:
        calls = []

        def compute():
            calls.append(1)
            return (np.arange(4.0), np.eye(3), 7)

        a = _cached_geo_eig(b"m9-probe", compute)
        b = _cached_geo_eig(b"m9-probe", compute)
        assert len(calls) == 1 and a is b
        assert not a[0].flags.writeable and not a[1].flags.writeable
        assert a[2] == 7                    # non-arrays pass through untouched
    finally:
        _clear_geo_eig_cache()


def test_m9_repeat_solve_bit_identical_with_the_readonly_cache():
    _clear_geo_eig_cache()
    try:
        outs = []
        for _ in range(2):
            st = _stack()
            st.add_layer(THK, segments=[(0.5, EPS_R), (0.5, EPS_G)])
            outs.append(st.set_source(WL, angle=0.0).solve())
        assert np.array_equal(outs[0][1], outs[1][1])
        assert np.array_equal(outs[0][2], outs[1][2])
        assert np.array_equal(outs[0][3], outs[1][3])
    finally:
        _clear_geo_eig_cache()
