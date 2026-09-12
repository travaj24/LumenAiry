"""VERIFY-A13 -- four gaps found while re-verifying WP-A13 (audit 2026-09-11
G9 / G10), plus one raised by VERIFY-A14; all five fixed in the PMM 2-D files
and pinned here.

All five are GUARD / SIGNAL gaps: none of them moves a physical number, and each
disarms or omits something a sibling surface already provides.

V1  ``max_pencil_dof=`` was accepted by ``PMM2DStackPure.add_layer`` and then
    DROPPED on the magnetic (``mu`` / ``mu_cell``) branch, so the SEGMENT-grid
    refusal could not be lifted there at all -- a documented escape hatch that
    silently did nothing.  MEASURED before: a 30x30 cell at M = 8 (pencil
    88 200) with ``max_pencil_dof=1e7`` still raised "above
    max_pencil_dof=12000" on the magnetic branch while the identical
    nonmagnetic call was accepted.
V2  the same branch priced the guard at the STACK's ``M`` rather than the
    LAYER's ``n_modes=``, so a ``layer_grids='per-layer'`` stack at ``M = 3``
    with a layer at ``n_modes = 8`` was accepted at a priced 7 200 when the
    real pencil is 88 200 -- an under-pricing of (7/2)^3 = 43x in QZ time.
V3  a MISMATCHED ``eps_cell`` / ``mu_cell`` pair reached the new merge scan
    before the union-grid check and died on an internal
    ``IndexError: index 5 is out of bounds for axis 0 with size 4`` instead of
    the class's own "all patterned layers must share ONE common (Nx, Ny) grid"
    ValueError.
V4  ``PMM2DStackHybrid``'s eig-cache refusal warning is documented (and tested)
    as ONCE PER INSTANCE, but ``solve_vs_wavelength`` hands each wavelength a
    ``copy.copy`` clone, and the flag was a plain bool in ``__dict__`` -- so the
    clone always started unwarned.  MEASURED before: 6 warnings on a 6-point
    sweep at ``max_workers`` 1 AND 3, with the parent's flag still ``False``.
    A wide sweep is the exact case the warning exists to describe.
V6  the PMM 2-D ``Efficiency2D`` never carried ``wl_eff`` -- the wavelength the
    solve ACTUALLY ran at after the Wood-anomaly nudge -- although the 1-D and
    RCWA results have carried it since audit H2, so a caller could not tell a
    nudged 2-D solve from an exact one without parsing the warning.  (Raised by
    VERIFY-A14 sec 6 V8.)

Ten of the twelve tests fail on the pre-fix code.  The two that do not are the
``magnetic=False`` arms of V1 / V2: they pin the PLAIN branch, which was already
correct, and exist as the side-by-side control that isolates the magnetic branch
as the defect.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import warnings

import numpy as np
import pytest

from lumenairy.elements.pmm import (
    PMM2DStackHybrid,
    pmm_efficiency_2d,
    pmm_efficiency_2d_cell,
    pmm_efficiency_2d_staggered,
    prepare_pmm_2d,
    prepare_pmm_2d_cell,
)
from lumenairy.elements.pmm.stack2d_pure import PMM2DStackPure
from lumenairy.elements.pmm.twod_staggered import _MAX_STAG_PENCIL_DOF

_WL = 1.0e-6
_P = 0.9e-6


def _half_fill(N, lo, hi, e=12.25):
    c = np.full((N, N), 1.0 + 0j)
    c[lo:hi, lo:hi] = e
    return c


def _uniform_mu(N):
    return np.full((N, N), 1.0 + 0j)


# --------------------------------------------------------------------------- #
# V1 / V2 -- the SEGMENT-grid cost guard on the MAGNETIC branch
# --------------------------------------------------------------------------- #

# A 30x30 SEGMENT grid whose walls sit at 7 and 23: gcd(30, 7, 23) = 1, so it is
# already the minimal uniform lattice (no redundancy warning) and the ONLY thing
# that can stop it is the absolute cap.  At M = 8 the pencil is
# 2 * (30*7)^2 = 88 200, seven times the 12 000 default -- ~1 391 GB projected.
_BIG = _half_fill(30, 7, 23)
_BIG_DOF_M8 = 2 * (30 * 7) ** 2


@pytest.mark.parametrize("magnetic", [False, True])
def test_v1_max_pencil_dof_reaches_both_add_layer_branches(magnetic):
    """The refusal must fire at the default cap and must be liftable by the
    documented keyword -- on the magnetic branch exactly as on the plain one.

    No tolerance: both arms are exact decisions (raise / no raise) on an
    integer comparison, ``2*Nx*(M-1)*Ny*(M-1)`` vs ``max_pencil_dof``.
    """
    assert _BIG_DOF_M8 > _MAX_STAG_PENCIL_DOF          # the fixture is in range
    kw = dict(eps_cell=_BIG)
    if magnetic:
        kw["mu_cell"] = _uniform_mu(30)
    st = PMM2DStackPure(_P, _P, n_modes=8, n_orders=3)
    with pytest.raises(ValueError, match=r"max_pencil_dof=12000"):
        st.add_layer(0.3e-6, **kw)
    st2 = PMM2DStackPure(_P, _P, n_modes=8, n_orders=3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st2.add_layer(0.3e-6, max_pencil_dof=10 ** 7, **kw)   # must NOT raise
    assert len(st2._layers) == 1


@pytest.mark.parametrize("magnetic", [False, True])
def test_v2_the_guard_prices_the_layers_own_modal_count(magnetic):
    """``add_layer(n_modes=)`` sets THIS layer's modal count, so the pencil the
    guard prices is ``2*(Nx*(n_modes-1))^2`` -- not the stack default.

    Exact decision again: at stack ``M = 3`` the fixture prices to
    ``2*(30*2)^2 = 7 200`` (accepted) and at the layer's ``n_modes = 8`` to
    88 200 (refused), so a branch that reads the wrong one is caught by the
    raise/no-raise boundary rather than by a tolerance.
    """
    kw = dict(eps_cell=_BIG)
    if magnetic:
        kw["mu_cell"] = _uniform_mu(30)
    st = PMM2DStackPure(_P, _P, n_modes=3, n_orders=3, layer_grids="per-layer")
    with pytest.raises(ValueError, match=r"M=8"):
        st.add_layer(0.3e-6, n_modes=8, **kw)
    # the stack's own M = 3 (pencil 7 200) is under the cap and accepted
    st2 = PMM2DStackPure(_P, _P, n_modes=3, n_orders=3,
                         layer_grids="per-layer")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st2.add_layer(0.3e-6, **kw)
    assert 2 * (30 * 2) ** 2 < _MAX_STAG_PENCIL_DOF < _BIG_DOF_M8


def test_v3_mismatched_eps_and_mu_grids_keep_the_union_grid_message():
    """The cost guard runs BEFORE the union-grid check, so it must not scan a
    pair it cannot merge: the caller's error has to survive."""
    st = PMM2DStackPure(_P, _P, n_modes=5, n_orders=3)
    with pytest.raises(ValueError, match="share ONE common"):
        st.add_layer(0.3e-6, eps_cell=_half_fill(6, 2, 4),
                     mu_cell=_uniform_mu(4))


def test_v3_the_merge_scan_itself_drops_a_mismatched_array():
    """Unit-level: the helper must not IndexError on ragged input."""
    from lumenairy.elements.pmm.twod_staggered import (
        _stag_merged_segments,
        _stag_minimal_uniform_segments,
    )
    a = _half_fill(6, 2, 4)
    b = _uniform_mu(4)
    assert _stag_merged_segments(a, b) == _stag_merged_segments(a)
    assert (_stag_minimal_uniform_segments(a, b)
            == _stag_minimal_uniform_segments(a))


# --------------------------------------------------------------------------- #
# V4 -- the eig-cache refusal signal across sweep clones
# --------------------------------------------------------------------------- #

def _budget_stack(cache_max_bytes=None):
    cell = _half_fill(8, 2, 6)
    kw = {} if cache_max_bytes is None else dict(cache_max_bytes=cache_max_bytes)
    st = PMM2DStackHybrid(_P, _P, n_superstrate=1.0, n_substrate=1.45,
                          degree=7, n_orders=3, **kw)
    st.add_layer(0.3e-6, eps_cell=cell)
    # OBLIQUE: at normal incidence on a centro-symmetric cell the even-parity
    # fold bypasses the per-layer modal cache, so there is nothing to refuse.
    st.set_source(_WL, theta=0.2, phi=0.3)
    return st


def _refusals(rec):
    return [x for x in rec if "modal (eig) cache REFUSED" in str(x.message)]


@pytest.mark.parametrize("max_workers", [1, 3])
def test_v4_sweep_clones_share_the_once_per_stack_refusal_signal(max_workers):
    """``solve_vs_wavelength`` runs each point on a ``copy.copy`` clone that
    SHARES the cache object; the warn-once state must be shared with it.

    Exact count, not a tolerance: ONE warning for the whole sweep, at any
    worker count, while ``cache_stats()['eig']['refused']`` keeps counting all
    six.  Pre-fix this read 6 warnings at both worker counts.
    """
    st = _budget_stack(cache_max_bytes=1)
    wls = list(np.linspace(0.9e-6, 1.1e-6, 6))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st.solve_vs_wavelength(wls, theta=0.2, phi=0.3,
                               max_workers=max_workers)
    assert len(_refusals(w)) == 1, [str(x.message) for x in _refusals(w)]
    assert st.cache_stats()["eig"]["refused"] >= len(wls)


def test_v4_repeated_solves_and_a_fresh_budget_are_unchanged():
    """The contract WP-A13 already pinned must still hold: once per instance
    over repeated ``solve()``, re-armed by ``add_layer`` (fresh caches), silent
    at the default budget -- and the ANSWER identical either way
    (refuse-never-degrade).  ``np.array_equal`` because the budget must not
    touch a single bit."""
    st = _budget_stack(cache_max_bytes=1)
    n = 0
    for _ in range(4):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            a = st.solve()
        n += len(_refusals(w))
    assert n == 1
    st.add_layer(0.1e-6, eps=2.0)                  # fresh caches -> re-armed
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        st.solve()
    assert len(_refusals(w)) == 1
    st_ok = _budget_stack()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        c = st_ok.solve()
    assert st_ok.cache_stats()["eig"]["refused"] == 0
    assert not _refusals(w)
    assert np.array_equal(np.asarray(a[2]), np.asarray(c[2]))


# --------------------------------------------------------------------------- #
# V6 -- wl_eff on the PMM 2-D results
# --------------------------------------------------------------------------- #

# EXACT Wood anomaly, built rather than hoped for: at normal incidence into
# n_sup = 1 the m = +/-1 order sits exactly at cut-off when Lambda = lambda
# (kx = m*lambda/Lambda = 1 = n_sup).  _grazing_safe_wavelength then substitutes
# lambda*(1 + 1e-7).  _P_OFF is the same geometry at a period that puts every
# order far from any cut-off.
_P_ON, _P_OFF = 1.0e-6, 0.47e-6
_NUDGE_REL = 1.0e-7


def _pillar_cell():
    c = np.full((6, 6), 1.0 + 0j)
    c[2:4, 2:4] = 12.25
    return c


def _stag_cell():
    return np.array([[12.25, 12.25, 1.0], [12.25, 1.0, 1.0],
                     [1.0, 1.0, 1.0]], dtype=complex)


def _all_entries(P):
    """Every PMM 2-D entry that returns an ``Efficiency2D``, at period ``P``."""
    xb = (0.25 * P, 0.75 * P)
    cell = _pillar_cell()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield "pmm_efficiency_2d", pmm_efficiency_2d(
            P, P, 12.25, 1.0, xb, xb, 1.45, 1.0, 0.3e-6, _WL, degree=7,
            n_orders=3)
        yield "pmm_efficiency_2d(stabilize)", pmm_efficiency_2d(
            P, P, 12.25, 1.0, xb, xb, 1.45, 1.0, 0.3e-6, _WL, degree=7,
            n_orders=3, stabilize=True)
        yield "pmm_efficiency_2d_cell", pmm_efficiency_2d_cell(
            P, P, cell, 1.45, 1.0, 0.3e-6, _WL, degree=7, n_orders=3)
        yield "pmm_efficiency_2d_cell(stabilize)", pmm_efficiency_2d_cell(
            P, P, cell, 1.45, 1.0, 0.3e-6, _WL, degree=7, n_orders=3,
            stabilize=True)
        yield "prepare_pmm_2d.solve", prepare_pmm_2d(
            P, P, 12.25, 1.0, xb, xb, 1.45, 1.0, 0.3e-6, degree=7,
            n_orders=3).solve(_WL)
        yield "prepare_pmm_2d_cell.solve", prepare_pmm_2d_cell(
            P, P, cell, 1.45, 1.0, 0.3e-6, degree=7, n_orders=3).solve(_WL)
        yield "pmm_efficiency_2d_staggered", pmm_efficiency_2d_staggered(
            P, P, _stag_cell(), 1.45, 1.0, 0.3e-6, _WL, degree=5, n_orders=2)


def test_v6_wl_eff_is_the_requested_wavelength_off_any_anomaly():
    """Two-sided arm 1: OFF the anomaly the nudge is the identity, so
    ``wl_eff`` must be the requested wavelength EXACTLY -- an equality, not a
    tolerance, because ``_grazing_safe_wavelength`` returns its input
    unchanged when no order sits in the cut-off band."""
    for name, res in _all_entries(_P_OFF):
        assert res.wl_eff is not None, name
        assert float(res.wl_eff) == _WL, (name, res.wl_eff)


def test_v6_wl_eff_reports_the_wood_nudge_on_an_exact_anomaly():
    """Two-sided arm 2: ON an exact anomaly every entry must report the
    SUBSTITUTED wavelength.  The substitution is a fixed relative +1e-7 in
    ``_grazing_safe_wavelength``, so the bar is that relative shift to 1e-12
    relative -- 5 decades below the shift itself and far above float64 noise on
    a 1e-6 m quantity (~2e-16 relative)."""
    for name, res in _all_entries(_P_ON):
        assert res.wl_eff is not None, name
        rel = float(res.wl_eff) / _WL - 1.0
        assert abs(rel - _NUDGE_REL) < 1e-12 * _NUDGE_REL / 1e-7 + 1e-18, (
            name, res.wl_eff, rel)
        assert float(res.wl_eff) != _WL, name


def test_v6_solving_at_the_reported_wl_eff_reproduces_the_nudged_answer():
    """The claim ``wl_eff`` makes is that this IS the wavelength solved, and
    the way to test it is to solve there explicitly: the re-solve sits off the
    anomaly, takes no nudge of its own, and must be BIT-IDENTICAL.  A `wl_eff`
    that merely echoed the request would fail arm 2; one that reported a
    wavelength the solver did not use would fail here."""
    P = _P_ON
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = pmm_efficiency_2d_cell(P, P, _pillar_cell(), 1.45, 1.0, 0.3e-6,
                                   _WL, degree=7, n_orders=3)
        b = pmm_efficiency_2d_cell(P, P, _pillar_cell(), 1.45, 1.0, 0.3e-6,
                                   a.wl_eff, degree=7, n_orders=3)
        c = pmm_efficiency_2d_staggered(P, P, _stag_cell(), 1.45, 1.0, 0.3e-6,
                                        _WL, degree=5, n_orders=2)
        d = pmm_efficiency_2d_staggered(P, P, _stag_cell(), 1.45, 1.0, 0.3e-6,
                                        c.wl_eff, degree=5, n_orders=2)
    assert float(b.wl_eff) == float(a.wl_eff)      # no second nudge
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))
    assert np.array_equal(np.asarray(a[2]), np.asarray(b[2]))
    assert float(d.wl_eff) == float(c.wl_eff)
    assert np.array_equal(np.asarray(c[2]), np.asarray(d[2]))
    # the released contract is untouched: still unpacks as (orders, R, T)
    o, R, T = a
    assert np.asarray(R).shape == np.asarray(T).shape
    assert a.dof == 2 * len(o)
