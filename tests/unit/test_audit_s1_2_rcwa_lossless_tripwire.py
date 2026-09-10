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


# --------------------------------------------------------------------------- #
# (5) the guard's message must name the EXACT-INDEX coincidence and its remedy
#     (2026-09-10, VERIFY_WOOD_LIST_AND_FFFNV_2026_09_10 follow-up B).
# --------------------------------------------------------------------------- #
#: Bar on ``|sum R + sum T - 2|`` for the exact-index-coincidence fixture.
#:
#: The structure is provably lossless, so ``sum R + sum T == 2`` is EXACT at
#: every truncation under the Laurent rule (2 rather than 1: two incident
#: polarizations).  That conservation law is the reference -- there is no
#: second solve to compare against and no prior reading to reproduce, and its
#: error floor is the arithmetic.
#:
#: Measured 2026-09-11 over ``n_orders`` 11..41 (16 truncations), on
#: (Windows py3.14 / numpy 2.4.4) x {HASWELL, PRESCOTT->Katmai, SANDYBRIDGE}
#: and (WSL py3.12 / numpy 2.4.6) x {HASWELL, PRESCOTT->Katmai}, one BLAS
#: thread, ``OPENBLAS_CORETYPE`` pinned on the command line:
#:
#:   arm / build-coretype        worst |closure|   warnings
#:   POST coincident  WIN-HASWELL      1.643e-13     0/16
#:   POST coincident  WIN-PRESCOTT     3.579e-13     0/16
#:   POST coincident  WIN-SANDYBRIDGE  3.477e-13     0/16
#:   POST coincident  WSL-HASWELL      1.386e-13     0/16
#:   POST coincident  WSL-PRESCOTT     3.682e-13     0/16
#:   POST detuned     (all five)     <=6.772e-13     0/16
#:   PRE  coincident  WIN-HASWELL      2.761e-02    16/16
#:   PRE  coincident  WIN-PRESCOTT     7.289e-04    11/16
#:   PRE  coincident  WIN-SANDYBRIDGE  4.690e-02    14/16
#:   PRE  coincident  WSL-HASWELL      5.001e-02    16/16
#:   PRE  coincident  WSL-PRESCOTT     1.584e-03    11/16
#:   PRE  detuned     (all five)     <=5.840e-13     0/16
#:
#: ``SKYLAKEX`` is not in the table because it is not runnable on the
#: measuring hardware (a Ryzen 9 5950X has no AVX-512: forcing that kernel
#: aborts numpy with SIGILL on both builds), and the bundled OpenBLAS carries
#: no Zen target at all, so ``OPENBLAS_CORETYPE=ZEN`` resolves to Haswell --
#: which is also what CI's AMD EPYC runners select.
#:
#: The bar sits 3.17 decades above the POST envelope (6.772e-13) and 5.86
#: decades below the smallest PRE reading that manifests (7.289e-04), inside a
#: gap the measurement leaves entirely empty.
_S1_2_CLOSURE_BAR = 1e-9


def _s1_2_fixture(detune=0.0):
    """The exact-index coincidence, built through the public API: a rotated
    director whose ordinary ``no^2`` is 2.25, a groove of 2.25, and
    ``n_substrate`` = 1.5, so the LAYER carries modes exactly degenerate with
    the substrate's.  ``detune`` walks ONLY the groove index off the
    substrate's, leaving the director, the period and the truncation exactly
    where they were -- the control that shows the coincidence is the cause."""
    th = np.deg2rad(35.0)
    c, s = np.cos(th), np.sin(th)
    rot = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    er = rot @ np.diag([2.3 ** 2, 1.5 ** 2, 1.5 ** 2]).astype(complex) @ rot.T
    eg = np.diag([(1.5 * (1.0 + detune)) ** 2] * 3).astype(complex)
    return er, eg


def _s1_2_closure(er, eg, n_orders):
    """``(|sum R + sum T - 2|, an _EnergyWarning fired)`` at one truncation."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out = rcwa_jones_1d_segments(0.7e-6, [(0.5, er), (0.5, eg)], 1.5, 1.0,
                                     0.5e-6, 1.0e-6, angle=0.0,
                                     n_orders=n_orders)
    fired = [w for w in rec if isinstance(w.message, _EnergyWarning)]
    return (abs(float(np.sum(out[1]) + np.sum(out[2])) - 2.0), fired)


def _s1_2_pre_round1_sqrt_decay(x, xp=None, band=1e-8):
    """The pre-round-1 branch body: the EXACT ``Re(r) == 0`` pin.  An ``eig``
    output never satisfies it -- its real part is the eigensolver's backward
    error, ~1e-16 -- so the pin fires only for the REGION modes, built in exact
    arithmetic, and never for a structured LAYER's.  Reinstating it is how the
    fail-before below is ENGINEERED rather than hoped for from a build."""
    from lumenairy.backend.array import array_namespace
    if xp is None:
        xp = array_namespace(x)
    x = xp.asarray(x).astype(complex)
    r = xp.sqrt(x)
    return xp.where((r.real == 0) & (r.imag < 0), -r, r)


class _s1_2_pre_arm:
    """Reinstate the pre-round-1 branch body at every module that binds the
    shared selector, for the duration of a ``with`` block."""

    _MODULES = ("lumenairy.elements.rcwa._core", "lumenairy.elements.rcwa.oned",
                "lumenairy.elements.rcwa.stack", "lumenairy.elements.pmm.twod",
                "lumenairy.elements.berreman")

    def __enter__(self):
        import importlib
        self._saved = []
        for name in self._MODULES:
            mod = importlib.import_module(name)
            if hasattr(mod, "_sqrt_decay"):
                self._saved.append((mod, mod._sqrt_decay))
                mod._sqrt_decay = _s1_2_pre_round1_sqrt_decay
        return self

    def __exit__(self, *a):
        for mod, fn in self._saved:
            mod._sqrt_decay = fn
        return False


def test_the_exact_index_coincidence_now_closes_and_the_warning_is_silent():
    """The exact-index coincidence is REPAIRED, so the guard is silent on it --
    and that silence is asserted as an ANSWER, not as an absence.

    RESTATED 2026-09-11 (branch-cut ROUND 3).  Until this date the test asserted
    the guard's MESSAGE TEXT, requiring the words ``EXACTLY EQUAL`` and
    ``DETUNE`` in a warning it expected the fixture to raise.  Both halves of
    that had expired: round 2 dropped the "detune by ~1e-6" advice (measured
    NOT to cure this class -- a relative 1e-6 leaves 8.1e-05 on Windows and
    2.3e-04 on WSL), and the branch-cut fix CURED the fixture, so no warning
    fires at all.  ``DID NOT WARN`` was the failure on every CI shard.

    What replaces it is the contract the repair actually established, stated
    against the conservation law rather than against the message:

      (a) the coincident fixture CLOSES to :data:`_S1_2_CLOSURE_BAR` at every
          truncation on the ladder -- the answer is right, which is the reason
          the guard is quiet;
      (b) it agrees with the DETUNED control, which has no coincidence at all,
          to the same bar -- so the coincidence has stopped being a distinct
          numerical regime rather than merely stopped being loud;
      (c) NO ``_EnergyWarning`` fires anywhere on the ladder.  A decision on a
          count, which no build can move.

    The fail-before is the sibling test below, and it is where the message text
    is still asserted -- on the arm where the message is still reachable, which
    is better coverage than the old form had.
    """
    er, eg = _s1_2_fixture()
    _, eg_off = _s1_2_fixture(detune=1e-3)
    worst_c = worst_d = 0.0
    warned = 0
    for n in range(11, 42, 2):
        c, fired = _s1_2_closure(er, eg, n)
        d, _ = _s1_2_closure(er, eg_off, n)
        worst_c = max(worst_c, c)
        worst_d = max(worst_d, d)
        warned += len(fired)
    assert worst_c < _S1_2_CLOSURE_BAR, (
        f"the exact-index coincidence misses lossless closure by {worst_c:.3e} "
        f"somewhere on n_orders 11..41: the modal branch cut has REOPENED and "
        f"a propagating layer mode is carrying the incoming root again")
    assert worst_d < _S1_2_CLOSURE_BAR, (
        f"the DETUNED control misses closure by {worst_d:.3e}, so it cannot "
        f"show that the coincidence is no longer a distinct regime")
    assert abs(worst_c - worst_d) < _S1_2_CLOSURE_BAR
    assert warned == 0, (
        f"{warned} lossless-closure warnings fired on a fixture that closes "
        f"at {worst_c:.3e}: the guard is warning about a right answer")


def test_the_pre_round_one_branch_reopens_it_and_the_message_names_the_cause():
    """The fail-before, ENGINEERED and two-sided -- and the surviving home of
    the message-text claim.

    With the pre-round-1 branch body reinstated the coincident fixture misses
    closure by orders somewhere on the ladder while the DETUNED control stays
    at the arithmetic floor everywhere on it, so the coincidence is
    demonstrably the cause.  WHICH truncations manifest is a per-build fact
    (11 to 16 of 16 warned across the five build x core-type samples in
    :data:`_S1_2_CLOSURE_BAR`), so the assertion is on the WORST over the
    ladder, never on a named truncation.

    The message the guard emits there must still name the coincidence and its
    real status: that this class was REPAIRED, so meeting it again is a
    regression to report rather than a modelling choice to work around.  That
    is the round-2 rewrite of the remedy text, and this is the only arm on
    which it can be read, since the shipped tree no longer reaches it.
    """
    er, eg = _s1_2_fixture()
    _, eg_off = _s1_2_fixture(detune=1e-3)
    ladder = range(11, 42, 2)
    with _s1_2_pre_arm():
        bad = {n: _s1_2_closure(er, eg, n) for n in ladder}
        good = {n: _s1_2_closure(er, eg_off, n)[0] for n in ladder}
    worst_bad = max(v[0] for v in bad.values())
    worst_good = max(good.values())
    assert worst_bad > 1e3 * max(worst_good, 1e-15), (
        "the pre-round-1 arm does not reopen the coincidence at any truncation "
        "on this build: coincident worst %.3e against detuned worst %.3e"
        % (worst_bad, worst_good))
    assert worst_good < _S1_2_CLOSURE_BAR, (
        "the DETUNED control is not clean on the pre-round-1 arm (%.3e), so "
        "it cannot show that the coincidence is what breaks it" % worst_good)
    text = " ".join(str(w.message)
                    for v in bad.values() for w in v[1])
    assert text, "the pre-round-1 arm reopened the defect but warned about none"
    assert "EXACTLY EQUAL" in text, text
    assert "UNIFORM LAYER" in text, text
    assert "2026-09-11" in text and "branch cut" in text, text
    assert "report" in text, text
