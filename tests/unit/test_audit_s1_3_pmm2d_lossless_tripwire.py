"""Audit S1-3 [P2][physics]: the hybrid 2-D PMM must not SILENTLY return a
non-physical energy total on the default ``stabilize=False`` fast path.

``_validate_cell_orders`` / the pillar order gate raise only when
``2*n_orders+1 > per-axis nodes``; at or near equality (and, more commonly,
at a too-low truncation for a high-contrast cell) the Fourier-projection
pseudo-inverse is ill-conditioned and the per-order efficiencies drift by a
few percent.  With ``stabilize=False`` (the default) the degree-scan passive
consensus in ``_stabilize_scalar`` never runs, so before the fix the solve
returned ``sum(R)+sum(T)`` well away from 1 for a PROVABLY LOSSLESS cell with
no warning at all.

The fix adds ``_warn_lossless_energy_2d`` -- a lossless per-order closure
tripwire on the ``stabilize=False`` return of both ``pmm_efficiency_2d`` and
``pmm_efficiency_2d_cell`` -- mirroring the RCWA sibling (S1-2,
``_check_energy(..., lossless=)``), the two-sided passive gate in
``_stabilize_scalar`` (P2-09), and ``PMMStack._warn_stack_energy``.  It fires
ONLY when every permittivity is exactly real (provably lossless) AND the
measured closure leaves the two-sided ``_PASSIVE_TOL_2D`` window; a lossy cell
(``R+T < 1`` is physical) is silent.

Independent oracle: the warned per-order zeroth transmission is cross-checked
against ``rcwa_efficiency_2d`` (Li rule, a wholly different solver) -- the
warned under-resolved solve is off by ~0.14, while the well-resolved PMM solve
that does NOT warn agrees with RCWA to ~0.02.  So the tripwire flags a
genuinely-wrong answer, not a bookkeeping blip, and stays silent on the
trustworthy one.

Measured (WSL venv, degree/orders as below):
    solve                                   E=sum(R+T)   T0(0,0)    warns
    pmm_efficiency_2d deg=7  n_orders=4      1.1473       0.2633     YES
    pmm_efficiency_2d deg=11 n_orders=12     1.0000       0.3872     no
    rcwa_efficiency_2d (Li) n_orders=12      1.0000       0.4073     --
    pmm_efficiency_2d deg=7  n_orders=4 lossy 0.8094      --         no
    pmm_efficiency_2d_cell deg=7 n_orders=6  1.0800       --         YES
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import pmm_efficiency_2d, pmm_efficiency_2d_cell
from lumenairy.elements.pmm.twod import _PASSIVE_TOL_2D
from lumenairy.elements.rcwa import rcwa_efficiency_2d

# shared provably-lossless, high-contrast separable pillar (real eps everywhere)
_P = 1.5e-6
_WL = 0.55e-6
_DEP = 0.5e-6
_EPS_PILLAR = 16.0            # real -> lossless
_EPS_HOST = 2.0              # real -> lossless
_XB = (0.25 * _P, 0.75 * _P)  # pillar fraction 0.5 -> 3 strips per axis
_MATCH = "closure violated"


def _t0(o, A):
    return float(np.asarray(A)[(o[:, 0] == 0) & (o[:, 1] == 0)][0])


def _solve_pillar(degree, n_orders, eps_pillar=_EPS_PILLAR):
    return pmm_efficiency_2d(_P, _P, eps_pillar, _EPS_HOST, _XB, _XB, 1.5, 1.0,
                             _DEP, _WL, degree=degree, n_orders=n_orders)


def _warned(record):
    return any(_MATCH in str(w.message) for w in record)


@pytest.fixture(scope="module")
def rcwa_t0_reference():
    """Independent zeroth-order transmission from ``rcwa_efficiency_2d`` (Li
    rule) on the SAME geometry, upsampled to an exact pixel cell -- a wholly
    different solver used as the oracle for per-order correctness."""
    S = 60
    up = np.full((S, S), _EPS_HOST + 0j)
    lo, hi = int(0.25 * S), int(0.75 * S)   # exact 0.25..0.75 wall alignment
    up[lo:hi, lo:hi] = _EPS_PILLAR
    o, R, T = rcwa_efficiency_2d(_P, _P, up, 1.5, 1.0, _DEP, _WL,
                                 n_orders_x=12, n_orders_y=12, formulation="li")
    assert abs(float(R.sum() + T.sum()) - 1.0) < 1e-3   # lossless RCWA closes
    return _t0(o, T)


# --------------------------------------------------------------------------- #
# (1) the tripwire fires on the under-resolved lossless pillar AND the warned
#     answer is genuinely wrong per the independent RCWA oracle.
# --------------------------------------------------------------------------- #
def test_pillar_lossless_tripwire_fires_and_answer_is_wrong(rcwa_t0_reference):
    with pytest.warns(UserWarning, match=_MATCH):
        o, R, T = _solve_pillar(degree=7, n_orders=4)
    # closure genuinely violated beyond the two-sided passive window
    E = float(R.sum() + T.sum())
    assert abs(E - 1.0) > _PASSIVE_TOL_2D, E
    # ... and the flagged per-order answer disagrees with the RCWA oracle by
    # far more than the well-resolved PMM solve does (below) -- an INDEPENDENT
    # confirmation that the warning marks a wrong result, not a bookkeeping blip.
    assert abs(_t0(o, T) - rcwa_t0_reference) > 0.08


# --------------------------------------------------------------------------- #
# (2) the SAME lossless geometry, well resolved, is silent AND trustworthy
#     (agrees with the independent RCWA oracle).
# --------------------------------------------------------------------------- #
def test_pillar_lossless_wellresolved_is_silent_and_correct(rcwa_t0_reference):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        o, R, T = _solve_pillar(degree=11, n_orders=12)
    assert not _warned(rec)
    E = float(R.sum() + T.sum())
    assert abs(E - 1.0) < _PASSIVE_TOL_2D, E
    # trustworthy per-order answer: matches the independent RCWA oracle
    assert abs(_t0(o, T) - rcwa_t0_reference) < 0.05


# --------------------------------------------------------------------------- #
# (3) NO false positive: the IDENTICAL under-resolved truncation on a LOSSY
#     cell (complex eps) leaves the closure short of 1 by ~0.19, yet the
#     tripwire is silent -- the guard keys on provable losslessness, not merely
#     on a closure deviation (this is what makes the test non-tautological).
# --------------------------------------------------------------------------- #
def test_pillar_lossy_same_undersampling_not_flagged():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        o, R, T = _solve_pillar(degree=7, n_orders=4,
                                eps_pillar=_EPS_PILLAR + 2.0j)
    assert not _warned(rec)
    E = float(R.sum() + T.sum())
    assert E < 1.0 - _PASSIVE_TOL_2D, E     # genuinely absorbs (deviates) ...
    # ... but is not provably lossless, so no warning was emitted.


# --------------------------------------------------------------------------- #
# (4) the cell entry point carries the same tripwire (single-pillar 3x3 cell).
# --------------------------------------------------------------------------- #
def _cell(eps_pillar=_EPS_PILLAR):
    cell = np.full((3, 3), _EPS_HOST + 0j)
    cell[1, 1] = eps_pillar
    return cell


def test_cell_lossless_tripwire_fires():
    with pytest.warns(UserWarning, match=_MATCH):
        o, R, T = pmm_efficiency_2d_cell(_P, _P, _cell(), 1.5, 1.0, _DEP, _WL,
                                         degree=7, n_orders=6)
    assert abs(float(R.sum() + T.sum()) - 1.0) > _PASSIVE_TOL_2D


def test_cell_lossy_not_flagged():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        o, R, T = pmm_efficiency_2d_cell(_P, _P, _cell(_EPS_PILLAR + 2.0j),
                                         1.5, 1.0, _DEP, _WL, degree=7,
                                         n_orders=6)
    assert not _warned(rec)
    assert float(R.sum() + T.sum()) < 1.0     # lossy -> short of 1, silent


# --------------------------------------------------------------------------- #
# (5) the stabilize=True path is unchanged: it routes through the degree-scan
#     consensus (its OWN passive gate), NOT the stabilize=False return, so the
#     new tripwire never fires there -- whether the consensus converges, emits
#     its own message, or raises on a hopeless truncation.
# --------------------------------------------------------------------------- #
def test_stabilize_true_does_not_emit_the_stabilize_false_tripwire():
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        try:
            pmm_efficiency_2d(_P, _P, _EPS_PILLAR, _EPS_HOST, _XB, _XB, 1.5,
                              1.0, _DEP, _WL, degree=7, n_orders=4,
                              stabilize=True)
        except RuntimeError:
            pass                     # hopeless truncation -> loud raise is fine
    assert not _warned(rec)          # the "closure violated" tripwire is F-path only
