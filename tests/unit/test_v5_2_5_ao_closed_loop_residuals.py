"""Tests for the v5.2.5 (AUDIT_V5_2_3 V9 + P3-F1-2) refinements to
``ao_closed_loop``.

Scope: pin the new ``leak`` and ``tol`` kwargs, the loosened ``gain``
range (``[0, 2]`` -- ``gain=0`` is the open-loop fallback), the
``wfs(...)`` returning ``None`` -> skip-update path, and the
``CONVENTIONS.md`` Section 7 docstring pointer.

Mirrors the v5.2.3 test idioms (small grids, fixed seeds, tight
tolerances, parametrised diagnostics) so the two test files compose
as a single pin set.

Author: Andrew Traverso -- v5.2.5 / AUDIT_V5_2_3 V9 + P3-F1-2.
"""
from __future__ import annotations

import numpy as np
import pytest

import lumenairy as la


# --------------------------------------------------------------------------- #
# Shared helpers (kept independent of the v5.2.3 test file -- copying the
# small builder is cheaper than cross-file imports).
# --------------------------------------------------------------------------- #


def _build_dm_and_grid(N=64, extent=20e-3, n_act=9, aperture=8e-3):
    """Helper: build a (DM, dx) pair matching the v5.2.3 unit-test
    geometry so the residual-magnitude pins are comparable.
    """
    dx = extent / N
    pitch = aperture / (n_act - 1)
    dm = la.DeformableMirror(
        n_actuators=n_act, pitch=pitch, dx=dx, N=N,
        inter_actuator_coupling=0.2,
    )
    return dm, dx


def _representable_disturbance(dm):
    """Render a DM-representable low-order disturbance onto the pupil
    grid using a scratch DM with the same geometry.

    Sampling the disturbance from the DM's own basis sidesteps the
    geometric mismatch a sinusoid-on-full-grid would create (the IFs
    decay outside the actuator footprint, so off-DM portions of a
    full-grid sinusoid are uncorrectable).
    """
    n = dm.n_actuators
    ai = (np.arange(n) - (n - 1) / 2) / ((n - 1) / 2)
    AX, _ = np.meshgrid(ai, ai)
    truth_cmd = np.sin(np.pi * AX)
    scratch = la.DeformableMirror(
        n_actuators=dm.n_actuators, pitch=dm.pitch, dx=dm.dx, N=dm.N,
        inter_actuator_coupling=dm.inter_actuator_coupling)
    scratch.command = truth_cmd
    return scratch.phase()


def _rms(a):
    a = np.asarray(a, dtype=np.float64)
    return float(np.sqrt(np.mean(a * a)))


# --------------------------------------------------------------------------- #
# leak kwarg pins
# --------------------------------------------------------------------------- #


def test_leak_zero_matches_v5_2_3_baseline():
    """Pin: with ``leak=0.0`` (default), behavior is bit-for-bit
    identical to v5.2.3.  Compares two independent runs (one default,
    one explicit ``leak=0.0``) on the same disturbance + DM geometry.
    """
    dm_a, dx = _build_dm_and_grid()
    dm_b, _ = _build_dm_and_grid()
    phase = _representable_disturbance(dm_a)

    # Run A: v5.2.3 call surface (no leak kwarg -> default 0.0).
    res_a, hist_a = la.ao_closed_loop(
        phase, dm=dm_a, n_iterations=8, gain=0.5, dx=dx,
        return_history=True)
    # Run B: explicit leak=0.0 should be byte-identical.
    res_b, hist_b = la.ao_closed_loop(
        phase, dm=dm_b, n_iterations=8, gain=0.5, leak=0.0, dx=dx,
        return_history=True)

    assert np.array_equal(res_a, res_b), (
        "leak=0.0 must reproduce v5.2.3 pure-integrator residual "
        "bit-for-bit.")
    assert np.array_equal(hist_a['rms_per_iter'], hist_b['rms_per_iter']), (
        "leak=0.0 must reproduce v5.2.3 rms_per_iter bit-for-bit.")
    assert np.array_equal(dm_a.command, dm_b.command), (
        "leak=0.0 must reproduce v5.2.3 accumulated DM command "
        "bit-for-bit.")


def test_leak_nonzero_decays_command():
    """Pin: with ``leak=0.5`` and a steady-state zero disturbance, the
    DM command relaxes toward zero across iterations (verifies the
    ``(1 - leak) * cmd_k`` term is wired in).

    Construction: pre-load the DM with a non-zero command, then call
    the loop on a zero phase disturbance.  Each iteration the WFS
    measures the (negative) DM phase as residual; with leak the
    command must shrink each step.
    """
    dm, dx = _build_dm_and_grid()
    # Pre-load a non-trivial DM command.
    n = dm.n_actuators
    rng = np.random.default_rng(7)
    dm.command = rng.standard_normal((n, n))
    cmd_norm_initial = float(np.linalg.norm(dm.command))
    assert cmd_norm_initial > 0.0  # sanity

    # gain=0 isolates the leak term -- the command updates only via
    # ``(1 - leak) * cmd_k``.  After K iterations the command should
    # be exactly ``(1 - leak)**K * cmd_initial``.
    K = 6
    leak = 0.5
    cmd_initial = dm.command.copy()
    la.ao_closed_loop(
        np.zeros((dm.N, dm.N), dtype=np.float64),
        dm=dm, n_iterations=K, gain=0.0, leak=leak, dx=dx)

    expected = (1.0 - leak) ** K * cmd_initial
    assert np.allclose(dm.command, expected, atol=1e-12), (
        f"leak={leak} did not decay the DM command as expected.  "
        f"||cmd_final||={np.linalg.norm(dm.command):.4g}, "
        f"expected ||{(1-leak)**K * cmd_norm_initial:.4g}||.")


def test_leak_validation_range():
    """Pin: leak must be in [0, 1]; out-of-range raises ValueError."""
    dm, dx = _build_dm_and_grid()
    phase = np.zeros((dm.N, dm.N), dtype=np.float64)
    with pytest.raises(ValueError, match='leak'):
        la.ao_closed_loop(
            phase, dm=dm, n_iterations=1, gain=0.5, leak=-0.1, dx=dx)
    with pytest.raises(ValueError, match='leak'):
        la.ao_closed_loop(
            phase, dm=dm, n_iterations=1, gain=0.5, leak=1.1, dx=dx)


# --------------------------------------------------------------------------- #
# gain=0 (open-loop fallback) pin
# --------------------------------------------------------------------------- #


def test_gain_zero_returns_disturbance_unchanged():
    """Pin: with ``gain=0`` the loop never updates the DM command; the
    residual equals ``phase_disturbance - dm.phase()`` (the entry
    state) at every iteration.
    """
    dm, dx = _build_dm_and_grid()
    phase = _representable_disturbance(dm)
    initial_cmd = dm.command.copy()

    residual, history = la.ao_closed_loop(
        phase, dm=dm, n_iterations=5, gain=0.0, dx=dx,
        return_history=True)

    # DM command is unchanged.
    assert np.array_equal(dm.command, initial_cmd), (
        "gain=0 must not update the DM command (open-loop fallback).")
    # Residual equals the initial residual exactly.
    expected_residual = phase - dm.phase()
    assert np.array_equal(residual, expected_residual), (
        "gain=0 residual must equal phase_disturbance - dm.phase() "
        "exactly.")
    # rms_per_iter should be flat (every entry equals the initial RMS).
    rms0 = _rms(expected_residual)
    assert np.allclose(history['rms_per_iter'], rms0, atol=1e-12), (
        f"gain=0 rms_per_iter should be flat at {rms0:.4g}; got "
        f"{history['rms_per_iter']}")


# --------------------------------------------------------------------------- #
# wfs returning None pin
# --------------------------------------------------------------------------- #


def test_wfs_returns_none_skips_update():
    """Pin: a WFS callable that returns ``None`` does not crash; the
    DM command is unchanged (with default ``leak=0``) that iteration.
    """
    dm, dx = _build_dm_and_grid()
    phase = _representable_disturbance(dm)
    initial_cmd = dm.command.copy()

    def _missed_wfs(_residual):
        return None  # sensor dropped this frame

    residual = la.ao_closed_loop(
        phase, dm=dm, n_iterations=4, gain=0.5, dx=dx,
        wfs=_missed_wfs)

    # With leak=0 (default), command is bit-for-bit unchanged.
    assert np.array_equal(dm.command, initial_cmd), (
        "wfs returning None with leak=0 must leave the DM command "
        "unchanged.")
    expected_residual = phase - dm.phase()
    assert np.array_equal(residual, expected_residual)


def test_wfs_non_callable_raises_typeerror():
    """Pin: non-None non-callable ``wfs`` raises TypeError early."""
    dm, dx = _build_dm_and_grid()
    phase = np.zeros((dm.N, dm.N), dtype=np.float64)
    with pytest.raises(TypeError, match='wfs'):
        la.ao_closed_loop(
            phase, dm=dm, n_iterations=1, gain=0.5, dx=dx,
            wfs='not callable')


# --------------------------------------------------------------------------- #
# tol early-stop pin
# --------------------------------------------------------------------------- #


def test_tol_early_stop():
    """Pin: with a small disturbance and a ``tol`` threshold larger
    than the steady-state residual, the loop stops before
    ``n_iterations`` and the returned history is sliced accordingly.
    """
    dm, dx = _build_dm_and_grid()
    # DM-representable disturbance so convergence is fast + deep.
    phase = _representable_disturbance(dm)
    n_iter_max = 50
    tol = 0.05  # rad RMS -- generous so the loop trips early

    residual, history = la.ao_closed_loop(
        phase, dm=dm, n_iterations=n_iter_max, gain=0.5, dx=dx,
        tol=tol, return_history=True)

    rms_arr = history['rms_per_iter']
    # Strictly fewer entries than the no-tol upper bound + 1.
    assert len(rms_arr) < n_iter_max + 1, (
        f"tol early-stop did not fire: rms_per_iter has "
        f"{len(rms_arr)} entries (== n_iterations + 1 = "
        f"{n_iter_max + 1}).")
    # Final entry is below tol (the trip condition).
    assert rms_arr[-1] < tol, (
        f"Loop stopped but final RMS {rms_arr[-1]:.4g} is not < "
        f"tol={tol}.")
    # And matches the returned residual array.
    assert np.isclose(rms_arr[-1], _rms(residual), rtol=1e-9)


# --------------------------------------------------------------------------- #
# Docstring pointer pin
# --------------------------------------------------------------------------- #


def test_conventions_doc_pointer_in_docstring():
    """Pin: the ``ao_closed_loop`` docstring cites ``CONVENTIONS.md``
    (Section 7) so future audits can find the sign-convention table
    from the call site.
    """
    doc = la.ao_closed_loop.__doc__ or ''
    assert 'CONVENTIONS' in doc, (
        "ao_closed_loop docstring must cite CONVENTIONS.md "
        "(Section 7) per v5.2.5 AUDIT_V5_2_3 V9 + P3-F1-2 "
        "doc-cohesion finding.")
