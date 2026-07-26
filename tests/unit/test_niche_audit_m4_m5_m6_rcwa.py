"""AUDIT_ADVERSARIAL_CODEBASE_2026_07_25 Territory M -- RCWA findings M4, M5, M6.

M4 (HIGH-physics)
    ``fff_nv``'s non-separability gate keyed on CURVATURE
    (``_nv_curved_wall_fraction``), but the documented-unvalidated cross term
    ``Cxy = -Delta @ [[Nx Ny]]`` is driven by ``max|Nx Ny|``.  Measured
    discriminators (this repo, 2026-07-25)::

        cell            curved_frac   max|Nx*Ny|
        metal stripe         0.0000       0.0000   <- the VALIDATED case
        metal square         0.0041       0.5000   <- admitted by the old gate
        diel  square         0.0042       0.5000
        metal disk           0.1755       0.5000   <- already gated (curvature)

    On the audit's axis-aligned Ag square pillar (P = 0.9 um, n_sub = 1.5,
    depth 0.4 um, WL 0.55 um, theta = 0.2, phi = 0.3, TM,
    eps = (0.05+3.3j)^2, duty 0.5, S = 121) ``fff_nv`` was INSIDE the
    documented scope and:

    * ``n_orders = 3`` -> ``R+T = 2.135`` (hard energy failure),
    * ``n_orders = 6, 7`` -> ``R+T = 1.059, 1.122`` (hard failures),
    * the two surviving truncations wandered ``A = 0.0805`` (n=4) /
      ``0.1129`` (n=5) against a converged ``'li'`` value of ``0.0787``,

    while ``'li'`` and ``'laurent'`` were energy-clean at every truncation on
    the same cell.  The gate now ALSO trips on ``max|Nx Ny|`` when the cell is
    METALLIC, which is exactly the square/stripe discriminator; lossless
    dielectric cornered cells (closure 1e-15, tracks ``'li'``) stay admitted.

M5 (HIGH)
    ``formulation='fff_nv'`` + ``stabilize=True`` ALWAYS hard-raised on a
    lossless cell.  ``fff_nv``'s in-plane operator is non-Hermitian, so its
    closure error is INHERENT, not an instability: measured per-rung closure on
    lossless dielectric pillars at conical incidence was ``+1.9e-2`` (S=13),
    ``+2.8e-2`` (S=17), ``-5.9e-2`` (S=25) -- an ``_EnergyWarning`` on EVERY
    rung, which the ladder counted as a failed attempt, so the whole ladder
    burned and ``_EnergyError`` was raised (measured 264 s / 278 s on a S=41
    cell; 1162 s / 922 s in the audit's own configuration), on the most
    accurate of the three formulations, with the tripwire's own advice being
    "Pass stabilize=True".  ``'li'`` held ``1e-15`` closure on the same cells,
    so the exemption must NOT extend to it.

M6 (MEDIUM)
    ``set_blas_threads`` is silently inert without ``threadpoolctl`` (an
    OPTIONAL dependency, absent in this environment) while
    ``_get_blas_threads()`` keeps reporting the cap -- and two ``# pragma``
    comments claimed threadpoolctl "ships with numpy".
"""
from __future__ import annotations

import contextlib
import warnings

import numpy as np
import pytest

from lumenairy.elements.rcwa import _core as _rcore
from lumenairy.elements.rcwa import rcwa_efficiency_2d
from lumenairy.elements.rcwa import twod as _twod
from lumenairy.elements.rcwa._core import _EnergyError, _EnergyWarning
from lumenairy.elements.rcwa.twod import (
    _nv_curved_wall_fraction,
    _nv_field_2d,
    _nv_nonseparable_guard,
)

# ``_nv_metallic_cell`` / ``_stabilize_closure_failure`` are the symbols the fix
# ADDS, so they are imported inside the tests that need them -- that keeps this
# module importable on the pre-fix tree, where each pin then fails
# individually instead of the whole file erroring at collection.

_C = np.complex128
EPS_AG = (0.05 + 3.3j) ** 2          # the audit's Ag pillar, -10.8875+0.33j
EPS_DIEL = 6.25 + 0j

# the audit's metal-square configuration (P, n_sub, n_sup, depth, wl, th, phi)
AUDIT = dict(period_x=0.9e-6, period_y=0.9e-6, n_substrate=1.5,
             n_superstrate=1.0, depth=0.4e-6, wavelength=0.55e-6,
             theta=0.2, phi=0.3)


def _sq(S, half, eps_in, eps_out=1.0):
    u = (np.arange(S) + .5) / S
    xx, yy = np.meshgrid(u, u, indexing="ij")
    return np.where((np.abs(xx - .5) < half) & (np.abs(yy - .5) < half),
                    eps_in, eps_out).astype(_C)


def _stripe(S, half, eps_in, eps_out=1.0):
    u = (np.arange(S) + .5) / S
    xx, _ = np.meshgrid(u, u, indexing="ij")
    return np.where(np.abs(xx - .5) < half, eps_in, eps_out).astype(_C)


def _disk(S, rf, eps_in, eps_out=1.0):
    u = (np.arange(S) + .5) / S
    xx, yy = np.meshgrid(u, u, indexing="ij")
    return np.where((xx - .5) ** 2 + (yy - .5) ** 2 < rf ** 2,
                    eps_in, eps_out).astype(_C)


def _audit_solve(cell, nord, form, **kw):
    return rcwa_efficiency_2d(
        AUDIT["period_x"], AUDIT["period_y"], cell, AUDIT["n_substrate"],
        AUDIT["n_superstrate"], AUDIT["depth"], AUDIT["wavelength"],
        theta=AUDIT["theta"], phi=AUDIT["phi"], polarization="tm",
        n_orders_x=nord, n_orders_y=nord, formulation=form, **kw)


# ===========================================================================
# M4 -- the discriminator (unit level: no solve, so this is milliseconds)
# ===========================================================================

def test_m4_measured_discriminators_separate_stripe_from_square():
    """The two discriminators on the four audited geometries.  CURVATURE does
    NOT separate the metal square from the validated stripe; ``max|Nx Ny|``
    does."""
    got = {}
    for name, cell in (("stripe", _stripe(121, .25, EPS_AG)),
                       ("square", _sq(121, .25, EPS_AG)),
                       ("disk", _disk(121, .30, EPS_AG))):
        ci = np.conj(cell)                 # the internal cell the solver sees
        Nx, Ny = _nv_field_2d(ci, 1.0, 1.0)
        got[name] = (_nv_curved_wall_fraction(ci),
                     float(np.abs(Nx * Ny).max()))
    # curvature: square is indistinguishable from the stripe (both << 0.06)
    assert got["stripe"][0] < 0.01 and got["square"][0] < 0.01
    assert got["disk"][0] > 0.10
    # max|Nx*Ny|: 0 for the stripe, ~0.5 for BOTH cornered cells
    assert got["stripe"][1] < 1e-6
    assert got["square"][1] > 0.4 and got["disk"][1] > 0.4


def test_m4_metallic_cell_detector():
    from lumenairy.elements.rcwa.twod import _nv_metallic_cell
    assert _nv_metallic_cell(_sq(41, .25, EPS_AG))          # Re<0 and Im!=0
    assert _nv_metallic_cell(_sq(41, .25, -3.0 + 4.0j))
    assert _nv_metallic_cell(_sq(41, .25, 2.25 + 0.5j))     # lossy dielectric
    assert not _nv_metallic_cell(_sq(41, .25, EPS_DIEL))    # lossless
    assert not _nv_metallic_cell(np.full((9, 9), EPS_DIEL, dtype=_C))


@pytest.mark.parametrize("cell,tripped,why", [
    (_sq(61, .25, EPS_AG), True, "METALLIC"),
    (_sq(61, .25, -3.0 + 4.0j), True, "METALLIC"),
    (_disk(61, .30, EPS_AG), True, "curved"),
    (_disk(61, .30, EPS_DIEL), True, "curved"),
    (_sq(61, .25, EPS_DIEL), False, ""),                  # lossless corners OK
    (_stripe(61, .25, EPS_AG), False, ""),                # the validated case
    (_stripe(61, .25, EPS_DIEL), False, ""),
    (np.full((61, 61), EPS_AG, dtype=_C), False, ""),     # uniform: no walls
])
def test_m4_guard_gates_the_measured_geometries(cell, tripped, why):
    ci = np.conj(cell)
    if tripped:
        with pytest.raises(ValueError, match="NON-SEPARABLE") as ei:
            _nv_nonseparable_guard("probe", ci, False)
        assert why in str(ei.value)
        # the opt-out still downgrades to a warning
        with pytest.warns(UserWarning, match="NON-SEPARABLE"):
            _nv_nonseparable_guard("probe", ci, True)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _nv_nonseparable_guard("probe", ci, False)     # silent, no raise


def test_m4_metal_square_raises_end_to_end():
    """End-to-end on the audit's configuration: the cell that measured
    ``R+T = 2.135`` now never reaches the solve."""
    with pytest.raises(ValueError, match="NON-SEPARABLE") as ei:
        _audit_solve(_sq(61, .25, EPS_AG), 3, "fff_nv")
    assert "METALLIC" in str(ei.value)
    assert "max|Nx*Ny|" in str(ei.value)


def test_m4_admitted_cells_still_solve_end_to_end():
    """A lossless dielectric square and a metal stripe -- the two VALIDATED
    fff_nv cases -- still solve, energy-clean, with no gate warning."""
    for cell in (_sq(25, .25, EPS_DIEL), _stripe(25, .25, EPS_AG)):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            o, R, T = rcwa_efficiency_2d(
                0.5e-6, 0.5e-6, cell, 1.0, 1.0, 0.18e-6, 0.633e-6,
                theta=np.deg2rad(1e-3), phi=0.0, polarization="tm",
                n_orders_x=2, n_orders_y=2, formulation="fff_nv")
        assert not any("NON-SEPARABLE" in str(w.message) for w in wl)
        assert float(R.sum() + T.sum()) <= 1.0 + 1e-6


@pytest.mark.parametrize("form,r00,t00,tot", [
    ("li", 0.07888199021470943, 0.19293324003178738, 0.9180022542327394),
    ("laurent", 0.1057721321786429, 0.21321614900516705, 0.9023691566743703),
])
def test_m4_li_and_laurent_untouched_bit_for_bit(form, r00, t00, tot):
    """The gate must not perturb the rigorous formulations.  Per-order values
    frozen at 5c9f7c3 (pre-fix) on the audit's metal square, S=25,
    n_orders=3.  Held to rel 1e-10, not exact equality: the eigensolve is
    bit-reproducible on one machine but drifts ~4e-14 relative across
    platforms/BLAS builds (measured: CI Linux vs the Windows box the values
    were frozen on), while any actual gate perturbation of these
    formulations would move them by far more than 1e-10."""
    o, R, T = _audit_solve(_sq(25, .25, EPS_AG), 3, form)
    p0 = int(np.where((o[:, 0] == 0) & (o[:, 1] == 0))[0][0])
    assert float(R[p0]) == pytest.approx(r00, rel=1e-10)
    assert float(T[p0]) == pytest.approx(t00, rel=1e-10)
    assert float(R.sum() + T.sum()) == pytest.approx(tot, abs=1e-10)


# ===========================================================================
# M5 -- the stabilize ladder's lossless-closure accounting
# ===========================================================================

class _RecordedWarning:
    """The minimal duck-type of a ``warnings.catch_warnings(record=True)``
    entry that ``_stabilize_closure_failure`` consumes."""

    def __init__(self, category, message):
        self.category = category
        self.message = category(message)
        self.filename = __file__
        self.lineno = 1


@pytest.mark.parametrize("formulation,expect_failed_rung", [
    ("laurent", True),          # closure IS a valid instability signal here
    ("li", True),
    ("auto", True),
    (None, True),
    ("fff_nv", False),          # inherent, non-Hermitian -> exempt (audit M5)
])
def test_m5_closure_accounting_exempts_only_fff_nv(formulation,
                                                   expect_failed_rung):
    from lumenairy.elements.rcwa._core import _stabilize_closure_failure
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        got = _stabilize_closure_failure(
            [_RecordedWarning(_EnergyWarning, "lossless energy closure "
                                              "violated (probe)"),
             _RecordedWarning(UserWarning, "unrelated")], formulation)
    assert (got is not None) is expect_failed_rung
    cats = [w.category.__name__ for w in wl]
    assert "UserWarning" in cats                 # unrelated always re-emitted
    # the exempt path must still SHOW the closure warning to the caller
    assert ("_EnergyWarning" in cats) is (not expect_failed_rung)


def test_m5_fff_nv_stabilize_completes_on_a_lossless_cell():
    """The audited hard-raise: ``fff_nv`` + ``stabilize=True`` on a LOSSLESS
    cell.  Pre-fix this raised ``_EnergyError`` after burning the whole ladder;
    it must now return, and still surface the inherent closure warning."""
    cd = _sq(13, .25, 3.5 ** 2)              # tiny grid -> the ladder is short
    for nord in (1, 2):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            o, R, T = _audit_solve(cd, nord, "fff_nv", stabilize=True)
        closure = abs(float(R.sum() + T.sum()) - 1.0)
        assert 1e-3 < closure < 0.05, closure      # the inherent fff_nv level
        assert any(issubclass(w.category, _EnergyWarning) for w in wl), (
            "the exempted closure warning must still reach the caller")


def test_m5_laurent_stabilize_still_burns_the_ladder_on_closure(monkeypatch):
    """The tripwire must NOT be weakened for the rigorous formulations: with an
    injected closure warning on every rung, ``laurent`` + ``stabilize=True``
    still raises while ``fff_nv`` completes."""
    calls = []

    def _fake_check_energy(fn_name, R, T, lossless=False):
        calls.append(fn_name)
        warnings.warn(_EnergyWarning(
            f"{fn_name}: lossless energy closure violated (injected probe)"))

    monkeypatch.setattr(_twod, "_check_energy", _fake_check_energy)
    cd = _sq(13, .25, 3.5 ** 2)
    with pytest.raises(_EnergyError, match="closure violated"):
        _audit_solve(cd, 1, "laurent", stabilize=True)
    assert len(calls) >= 2, "the ladder should have tried several rungs"
    calls.clear()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        _audit_solve(cd, 1, "fff_nv", stabilize=True)     # must NOT raise
    assert len(calls) == 1, "fff_nv should stand on its first rung"
    assert any(issubclass(w.category, _EnergyWarning) for w in wl)


def test_m5_hard_energy_failures_still_advance_the_ladder():
    """The exemption is closure-ONLY: a HARD ``_EnergyError`` (the 5% tripwire)
    must still move fff_nv to the next truncation.  On the S=13 cell the first
    two rungs fail hard (measured R+T well past 1.05) and the ladder lands on
    the third, so the returned order count differs from the request."""
    cd = _sq(13, .25, 3.5 ** 2)
    with pytest.raises(_EnergyError):
        _audit_solve(cd, 1, "fff_nv", stabilize=False)    # rung 1 fails hard
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", _EnergyWarning)
        o, _R, _T = _audit_solve(cd, 1, "fff_nv", stabilize=True)
    assert len(o) > (2 * 1 + 1) ** 2, len(o)              # laddered upward


# ===========================================================================
# M6 -- set_blas_threads warns when it cannot take effect
# ===========================================================================

def test_m6_cap_without_threadpoolctl_warns_once(monkeypatch):
    monkeypatch.setattr(_rcore, "_threadpoolctl_available", lambda: False)
    monkeypatch.setattr(_rcore, "_BLAS_WARNED_UNCONTROLLABLE", False)
    try:
        with pytest.warns(UserWarning, match="threadpoolctl"):
            _rcore.set_blas_threads(2)
        assert _rcore._get_blas_threads() == 2       # still reported
        # ... and only ONCE per process
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            _rcore.set_blas_threads(3)
        assert not wl, [str(w.message) for w in wl]
    finally:
        _rcore.set_blas_threads(None)


def test_m6_no_warning_when_a_controller_is_available(monkeypatch):
    monkeypatch.setattr(_rcore, "_threadpoolctl_available", lambda: True)
    monkeypatch.setattr(_rcore, "_BLAS_WARNED_UNCONTROLLABLE", False)
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always")
            _rcore.set_blas_threads(2)
        assert not wl, [str(w.message) for w in wl]
    finally:
        _rcore.set_blas_threads(None)


def test_m6_clearing_the_cap_never_warns(monkeypatch):
    monkeypatch.setattr(_rcore, "_threadpoolctl_available", lambda: False)
    monkeypatch.setattr(_rcore, "_BLAS_WARNED_UNCONTROLLABLE", False)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        _rcore.set_blas_threads(None)
    assert not wl


def test_m6_context_manager_shares_the_warning(monkeypatch):
    monkeypatch.setattr(_rcore, "_threadpoolctl_available", lambda: False)
    monkeypatch.setattr(_rcore, "_BLAS_WARNED_UNCONTROLLABLE", False)
    with pytest.warns(UserWarning, match="threadpoolctl"):
        with _rcore.rcwa_blas_threads(2):
            assert _rcore._get_blas_threads() == 2
    assert _rcore._get_blas_threads() is None            # restored


def test_m6_library_internal_per_worker_cap_stays_quiet(monkeypatch):
    """The warning is for a cap the USER asked for.  The library's own
    per-worker caps inside threaded sweeps go through ``_blas_threads_quiet``,
    so an ordinary sweep does not report "your cap is inert"."""
    monkeypatch.setattr(_rcore, "_threadpoolctl_available", lambda: False)
    monkeypatch.setattr(_rcore, "_BLAS_WARNED_UNCONTROLLABLE", False)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        with _rcore._blas_threads_quiet(1):
            assert _rcore._get_blas_threads() == 1
    assert not wl, [str(w.message) for w in wl]
    assert _rcore._get_blas_threads() is None            # restored


def test_m6_pmm_threaded_sweep_emits_no_blas_warning(monkeypatch):
    """End-to-end on the real consumer: ``solve_vs_wavelength`` caps BLAS per
    worker internally (``blas_per_worker=1`` by default), and must stay
    silent."""
    monkeypatch.setattr(_rcore, "_threadpoolctl_available", lambda: False)
    monkeypatch.setattr(_rcore, "_BLAS_WARNED_UNCONTROLLABLE", False)
    from lumenairy.elements.pmm import PMMStack
    st = PMMStack(0.5e-6, n_substrate=1.5, degree=8, far_field_orders=5)
    st.add_layer(0.2e-6, segments=[(0.5, 6.25 + 0j), (0.5, 1.0 + 0j)])
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        st.solve_vs_wavelength([0.60e-6, 0.633e-6], angle=0.0)
    assert not any("threadpoolctl" in str(w.message) for w in wl), (
        [str(w.message) for w in wl])


def test_m6_the_cap_really_is_inert_here():
    """Documents WHY the warning exists: with no controller installed the cap
    resolves to a no-op context, so the solve runs at the environment's
    threading while ``_get_blas_threads()`` reports the requested cap."""
    try:
        import threadpoolctl  # noqa: F401
        pytest.skip("threadpoolctl installed: the cap is effective here")
    except ImportError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _rcore.set_blas_threads(2)
        assert _rcore._get_blas_threads() == 2
        assert isinstance(_rcore._blas_limit(), contextlib.nullcontext)
    finally:
        _rcore.set_blas_threads(None)
