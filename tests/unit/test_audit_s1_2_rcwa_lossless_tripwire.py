"""Audit S1-2 [P2][physics]: the provably-lossless per-order energy tripwire
(``_check_energy(..., lossless=)``) must reach ``RCWAStack.solve`` and the
``rcwa_jones_1d`` / ``rcwa_jones_1d_segments`` core.

Before the fix these two entry points called ``_check_energy(R, T)`` with the
flag defaulting ``False``, so the audited "silent window" guard (a lossless
solve violating ``sum(R)+sum(T) = 1`` by 1e-6..0.05, carrying per-order errors
of a few percent) was DEAD there -- unlike ``rcwa_efficiency_1d/2d``,
``rcwa_jones_2d`` and the shapes path, which all pass ``lossless=``.

The tests below verify (1) the ``_stack_lossless`` / ``_cell_lossless``
predicate classifies every layer-kind + region slot correctly, (2) the flag is
now actually forwarded to ``_check_energy`` (an independent probe that captures
the argument -- fails before the fix, passes after, on any BLAS build), and (3)
the end-to-end contract that a lossless jones solve is never silently wrong at
an unstable truncation (clean OR warns OR raises).
"""
import warnings

import numpy as np
import pytest

import lumenairy.elements.rcwa._core as _core
import lumenairy.elements.rcwa.oned as _onedmod
import lumenairy.elements.rcwa.stack as _stackmod
from lumenairy.elements.rcwa import (
    RCWAStack,
    rcwa_jones_1d,
    rcwa_jones_1d_segments,
)
from lumenairy.elements.rcwa._core import _EnergyWarning


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _iso_cell(n_lo=1.5, n_hi=2.5, size=24):
    """A small lossless isotropic patterned cell (real eps everywhere; sized
    above the 4*n_orders+1 Fourier-sampling floor for n_orders<=5)."""
    cell = np.full((size, size), n_lo ** 2 + 0j)
    q = size // 4
    cell[q:-q, q:-q] = n_hi ** 2
    return cell


def _tensor_cell(eps=2.25, size=24):
    """A small lossless in-plane isotropic tensor cell (eps * I3 everywhere)."""
    return (eps + 0j) * np.broadcast_to(
        np.eye(3, dtype=complex), (size, size, 3, 3)).copy()


def _capture_lossless(monkeypatch, module):
    """Patch ``module._check_energy`` to record the ``lossless`` kwarg it is
    handed (then call through so the real guard still runs)."""
    seen = {}
    orig = module._check_energy

    def _rec(fn_name, R, T, lossless=False):
        seen["lossless"] = lossless
        return orig(fn_name, R, T, lossless=lossless)

    monkeypatch.setattr(module, "_check_energy", _rec)
    return seen


# --------------------------------------------------------------------------- #
# (1) predicate: RCWAStack._stack_lossless enumerates every kind + region slot
# --------------------------------------------------------------------------- #
def test_stack_lossless_predicate_all_kinds_and_regions():
    # A provably-lossless mixed stack: uniform + iso + tensor + analytic shapes,
    # real region indices -> _stack_lossless() is True.
    def _base():
        st = RCWAStack(period=1.0e-6, period_y=1.0e-6, n_superstrate=1.0,
                       n_substrate=1.5, n_orders=5, n_orders_y=5)
        st.add_layer(0.05e-6, eps=2.25)
        st.add_layer(0.10e-6, eps_cell=_iso_cell())
        st.add_layer(0.08e-6, eps_tensor_cell=_tensor_cell())
        st.add_layer(0.06e-6, eps_background=1.0, shapes=[
            dict(shape="rectangle", eps=4.0, size=(0.2e-6, 0.2e-6),
                 center=(0.0, 0.0))])
        return st

    assert _base()._stack_lossless() is True

    # Flip ONE slot complex at a time -> the predicate must catch each branch
    # of the enumeration (superstrate, substrate, uniform, iso, tensor, shapes
    # background, shape eps).  A miss in any branch would leave a lossy input
    # mis-classified as lossless.
    st = _base()
    st.n_superstrate = 1.0 + 0.01j
    assert st._stack_lossless() is False
    st = _base()
    st.n_substrate = 1.5 + 0.01j
    assert st._stack_lossless() is False

    st = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=5)
    st.add_layer(0.05e-6, eps=2.25 + 0.1j)                 # lossy uniform
    assert st._stack_lossless() is False

    lossy_cell = _iso_cell()
    lossy_cell[8, 8] += 0.1j                                # lossy iso pixel
    st = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=5)
    st.add_layer(0.10e-6, eps_cell=lossy_cell)
    assert st._stack_lossless() is False

    lossy_t = _tensor_cell()
    lossy_t[..., 0, 0] += 0.1j                              # lossy tensor comp
    st = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=5)
    st.add_layer(0.08e-6, eps_tensor_cell=lossy_t)
    assert st._stack_lossless() is False

    st = RCWAStack(period=1.0e-6, period_y=1.0e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=5, n_orders_y=5)
    st.add_layer(0.06e-6, eps_background=1.0 + 0.05j, shapes=[   # lossy bg
        dict(shape="rectangle", eps=4.0, size=(0.2e-6, 0.2e-6),
             center=(0.0, 0.0))])
    assert st._stack_lossless() is False

    st = RCWAStack(period=1.0e-6, period_y=1.0e-6, n_superstrate=1.0,
                   n_substrate=1.5, n_orders=5, n_orders_y=5)
    st.add_layer(0.06e-6, eps_background=1.0, shapes=[
        dict(shape="rectangle", eps=4.0 + 0.05j, size=(0.2e-6, 0.2e-6),
             center=(0.0, 0.0))])                                # lossy shape
    assert st._stack_lossless() is False


# --------------------------------------------------------------------------- #
# (2) wiring: the flag actually reaches _check_energy (fail-before/pass-after)
# --------------------------------------------------------------------------- #
def test_rcwastack_solve_forwards_lossless_flag(monkeypatch):
    seen = _capture_lossless(monkeypatch, _stackmod)

    # provably lossless -> lossless=True must be forwarded
    st = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5,
                   n_orders=5)
    st.add_layer(0.10e-6, eps_cell=_iso_cell())
    st.set_source(0.633e-6, theta=0.1).solve()
    assert seen["lossless"] is True

    # a lossy (complex) substrate -> lossless=False (R+T<1 is physical, silent)
    seen.clear()
    st = RCWAStack(period=1.0e-6, n_superstrate=1.0, n_substrate=1.5 + 0.05j,
                   n_orders=5)
    st.add_layer(0.10e-6, eps_cell=_iso_cell())
    st.set_source(0.633e-6, theta=0.1).solve()
    assert seen["lossless"] is False


def test_rcwa_jones_1d_forwards_lossless_flag(monkeypatch):
    seen = _capture_lossless(monkeypatch, _onedmod)
    eps_lo = (1.0 + 0j) ** 2 * np.eye(3)
    eps_hi = (2.0 + 0j) ** 2 * np.eye(3)

    # binary rcwa_jones_1d, provably lossless
    rcwa_jones_1d(0.8e-6, eps_hi, eps_lo, 1.5, 1.0, 0.5e-6, 0.5, 0.55e-6,
                  angle=0.1, n_orders=7)
    assert seen["lossless"] is True

    # lossy ridge tensor -> False
    seen.clear()
    rcwa_jones_1d(0.8e-6, eps_hi + 0.1j * np.eye(3), eps_lo, 1.5, 1.0,
                  0.5e-6, 0.5, 0.55e-6, angle=0.1, n_orders=7)
    assert seen["lossless"] is False

    # multi-segment path shares the same core; lossless -> True
    seen.clear()
    rcwa_jones_1d_segments(
        0.8e-6, [(0.3, eps_hi), (0.3, eps_lo), (0.4, eps_hi)],
        1.5, 1.0, 0.5e-6, 0.55e-6, angle=0.1, n_orders=7)
    assert seen["lossless"] is True

    # ... and a lossy segment -> False
    seen.clear()
    rcwa_jones_1d_segments(
        0.8e-6, [(0.5, eps_hi), (0.5, eps_lo + 0.05j * np.eye(3))],
        1.5, 1.0, 0.5e-6, 0.55e-6, angle=0.1, n_orders=7)
    assert seen["lossless"] is False


# --------------------------------------------------------------------------- #
# (3) end-to-end contract: a lossless jones solve is never SILENTLY wrong at an
#     unstable truncation (mirrors test_v5_14_1's rcwa_efficiency_1d contract;
#     the instability at a large-period / low-contrast coincidence is
#     BLAS-build dependent, so the assertion is clean OR warns OR raises).
# --------------------------------------------------------------------------- #
def test_lossless_jones_never_silently_violates_closure():
    # P = 8 um, low index contrast in vacuum super / n=1.5 sub -- the audited
    # regime where a stray (period, n_orders) coincidence can go unstable.
    eps_ridge = (1.5 + 0j) ** 2 * np.eye(3)
    eps_groove = (1.45 + 0j) ** 2 * np.eye(3)
    for M in (18, 19, 20, 21, 22):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            try:
                _o, R, T, _J = rcwa_jones_1d(
                    8.0e-6, eps_ridge, eps_groove, 1.5, 1.0, 1.0e-6, 0.5,
                    0.6e-6, angle=0.0, n_orders=M)
            except Exception:            # a loud raise is acceptable behaviour
                continue
            # per incident polarization the lossless closure is either clean
            # or the tripwire fired -- never a SILENT violation beyond the
            # tripwire's OWN tolerance.  The tolerance must match the police
            # exactly: ``_check_energy`` fires when
            # ``abs(sum(R+T) - n_states) > 1e-6 * n_states`` (_core.py:348), so
            # the "clean" bound per incident state is ``1e-6 * n_states`` -- a
            # tighter per-state 1e-6 spuriously fails on benign finite-order
            # convergence residuals the police deliberately ignores (e.g.
            # n_orders=22 lands at ~1.06e-6, six orders below the audit's
            # 1e-6..0.05 per-order-wrong pathology).  Tie the assertion to the
            # tripwire constant so the test polices the real contract, not an
            # independently-chosen (and inconsistent) line.
            closure = np.abs(R.sum(axis=1) + T.sum(axis=1) - 1.0)
            n_states = R.shape[0]
            clean_tol = 1e-6 * n_states           # == _check_energy's threshold
            warned = any(issubclass(w.category, _EnergyWarning) for w in wl)
            assert bool(np.all(closure < clean_tol)) or warned, (
                f"n_orders={M}: silent lossless closure violation "
                f"{closure} (tol {clean_tol:.1e}) with no _EnergyWarning")


# --------------------------------------------------------------------------- #
# (4) NO false positive: a real but ASYMMETRIC (non-reciprocal, non-Hermitian)
#     tensor is physically NOT lossless (R+T != 1 grows with obliquity), so it
#     must not be flagged -- the predicate needs a symmetry clause, not just
#     Im == 0.
# --------------------------------------------------------------------------- #
def test_nonreciprocal_real_asymmetric_tensor_not_lossless():
    asym = np.array([[2.25, 0.0, 0.5],
                     [0.0, 2.25, 0.0],
                     [0.1, 0.0, 2.40]], dtype=complex)   # eps_xz=0.5 != eps_zx=0.1
    sym = 0.5 * (asym + asym.T)                          # its reciprocal sibling
    # predicate: real-asymmetric -> not lossless; real-symmetric -> lossless
    assert _core._cell_lossless(1.0, 1.5, asym, asym) is False
    assert _core._cell_lossless(1.0, 1.5, sym, sym) is True
    # end-to-end: the asymmetric tensor must NOT raise the lossless tripwire at
    # oblique incidence, where its physical closure deviation (~3e-3) would
    # otherwise be mistaken for numerical instability.
    with warnings.catch_warnings():
        warnings.simplefilter("error", _EnergyWarning)
        rcwa_jones_1d(0.8e-6, asym, asym, 1.5, 1.0, 0.5e-6, 0.5, 0.55e-6,
                      angle=np.deg2rad(35.0), n_orders=3)
