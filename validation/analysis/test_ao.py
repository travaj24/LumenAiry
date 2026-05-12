"""Tests for the adaptive-optics primitives (4.0+):
DeformableMirror, zernike_modal_basis, slope_to_modal,
LeakyIntegrator.
"""
from __future__ import annotations

import sys
import pathlib as _pathlib
import numpy as np

_sys_path = _pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_sys_path))
from _harness import Harness

import lumenairy as la


H = Harness('ao')


# ---------------------------------------------------------------------
H.section('DeformableMirror')


def t_dm_zero_command_gives_zero_phase():
    dm = la.DeformableMirror(n_actuators=5, pitch=8e-3, dx=1e-3, N=64)
    phi = dm.phase()
    return phi.shape == (64, 64) and float(np.abs(phi).max()) == 0, \
        f'max|phi|={float(np.abs(phi).max()):.3e}'


H.run('DM: zero command -> zero phase', t_dm_zero_command_gives_zero_phase)


def t_dm_single_actuator_poke_peaks_at_centre():
    """Poking the centre actuator with amplitude 1.0 should give a
    Gaussian profile peaked at the grid centre."""
    N, dx = 64, 1e-3
    dm = la.DeformableMirror(n_actuators=5, pitch=8e-3, dx=dx, N=N)
    cmd = np.zeros((5, 5))
    cmd[2, 2] = 1.0
    dm.set_command(cmd)
    phi = dm.phase()
    pk_centre = float(phi[N // 2, N // 2])
    pk_edge = float(phi[0, 0])
    return abs(pk_centre - 1.0) < 1e-9 and pk_edge < 0.01, \
        f'centre={pk_centre:.4f} (expect 1), edge={pk_edge:.4e}'


H.run('DM: centre poke gives peak phase at grid centre',
      t_dm_single_actuator_poke_peaks_at_centre)


def t_dm_apply_preserves_amplitude():
    """The DM is a pure phase element; |E_out| should equal |E_in|."""
    N, dx = 64, 1e-3
    dm = la.DeformableMirror(n_actuators=5, pitch=8e-3, dx=dx, N=N)
    dm.set_command(np.array([
        [0.0, 0.0, 0.5, 0.0, 0.0],
        [0.0, 0.3, 0.0, 0.3, 0.0],
        [0.5, 0.0, 0.0, 0.0, 0.5],
        [0.0, 0.3, 0.0, 0.3, 0.0],
        [0.0, 0.0, 0.5, 0.0, 0.0],
    ]))
    rng = np.random.default_rng(0)
    E = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))).astype(np.complex128)
    E_out = dm.apply(E)
    err = float(np.max(np.abs(np.abs(E_out) - np.abs(E))))
    return err < 1e-12, f'|E_out| - |E| max = {err:.2e}'


H.run('DM.apply: amplitude preserved', t_dm_apply_preserves_amplitude)


def t_dm_set_command_validates_size():
    dm = la.DeformableMirror(n_actuators=5, pitch=8e-3, dx=1e-3, N=64)
    try:
        dm.set_command(np.zeros(24))  # wrong size
        return False, 'no exception raised'
    except ValueError as e:
        return '25' in str(e), str(e)


H.run('DM.set_command: rejects wrong size', t_dm_set_command_validates_size)


# ---------------------------------------------------------------------
H.section('Slope-to-modal reconstructor')


def t_basis_shape_and_invertibility():
    basis = la.zernike_modal_basis(n_modes=6, n_lenslets=10,
                                     semi_aperture=10e-3, first_mode=1)
    R = basis['reconstructor']
    n_lens = basis['lenslet_xy'][0].size
    shape_ok = R.shape == (6, 2 * n_lens)
    # R * M should approximate identity (n_modes x n_modes).
    M = basis['influence_matrix']
    eye = R @ M
    err = float(np.max(np.abs(eye - np.eye(6))))
    return shape_ok and err < 1e-9, \
        f'R shape={R.shape}, max|R*M - I|={err:.2e}'


H.run('modal basis: R has correct shape + R*M ~ I',
      t_basis_shape_and_invertibility)


def t_single_mode_injection_recovery():
    """Inject a known single-mode slope vector; recover the mode with
    coefficient matching the injection to machine precision."""
    basis = la.zernike_modal_basis(n_modes=6, n_lenslets=10,
                                     semi_aperture=10e-3, first_mode=1)
    for k in range(6):
        true_modes = np.zeros(6)
        true_modes[k] = 0.5
        synth_slopes = basis['influence_matrix'] @ true_modes
        recovered = la.slope_to_modal(synth_slopes, basis)
        err = float(np.max(np.abs(recovered - true_modes)))
        if err > 1e-9:
            return False, f'mode {k}: err={err:.2e}'
    return True, 'all 6 modes recovered to <1e-9'


H.run('slope_to_modal: single-mode injections recovered to <1e-9',
      t_single_mode_injection_recovery)


def t_slope_to_modal_accepts_both_shapes():
    """slopes argument accepts both (N, 2) and (2*N,)."""
    basis = la.zernike_modal_basis(n_modes=3, n_lenslets=8,
                                     semi_aperture=10e-3)
    n_lens = basis['lenslet_xy'][0].size
    flat = np.linspace(0, 1, 2 * n_lens)
    pair = flat.reshape(2, n_lens).T  # (n_lens, 2)
    a = la.slope_to_modal(flat, basis)
    b = la.slope_to_modal(pair, basis)
    err = float(np.max(np.abs(a - b)))
    return err < 1e-12, f'flat vs (N,2) err={err:.2e}'


H.run('slope_to_modal: flat and (N, 2) shapes agree',
      t_slope_to_modal_accepts_both_shapes)


# ---------------------------------------------------------------------
H.section('LeakyIntegrator')


def t_integrator_first_step():
    integ = la.LeakyIntegrator(gain=0.5, leak=0.0, n_modes=4)
    cmd = integ.update(np.array([2.0, 0.0, 0.0, 0.0]))
    return abs(cmd[0] - 1.0) < 1e-15, f'cmd[0]={cmd[0]}'


H.run('integrator: c[1] = gain * err (pure integrator)',
      t_integrator_first_step)


def t_integrator_leak_decays_steady_command():
    """With error = 0 and leak > 0, the command decays."""
    integ = la.LeakyIntegrator(gain=0.5, leak=0.1, n_modes=2)
    integ.command = np.array([1.0, 0.0])
    cmd = integ.update(np.zeros(2))
    return abs(cmd[0] - 0.9) < 1e-15, f'cmd[0]={cmd[0]}'


H.run('integrator: leak decays command toward zero',
      t_integrator_leak_decays_steady_command)


def t_integrator_steady_state_geometric_series():
    """With constant unit error, integrator steady state c_inf =
    gain * sum_{k=0..inf} (1 - leak)^k = gain / leak."""
    gain = 0.4
    leak = 0.05
    integ = la.LeakyIntegrator(gain=gain, leak=leak, n_modes=1)
    err = np.array([1.0])
    for _ in range(1000):
        integ.update(err)
    expected = gain / leak  # 0.4 / 0.05 = 8
    rel = abs(float(integ.command[0]) - expected) / expected
    return rel < 1e-3, f'c_inf={float(integ.command[0]):.4f}, expect={expected}'


H.run('integrator: steady state matches gain/leak (1e-3 rel)',
      t_integrator_steady_state_geometric_series)


def t_integrator_rejects_wrong_error_size():
    integ = la.LeakyIntegrator(gain=0.3, n_modes=5)
    try:
        integ.update(np.zeros(3))
        return False, 'no exception'
    except ValueError as e:
        return '5' in str(e), str(e)


H.run('integrator.update: rejects wrong-size error vector',
      t_integrator_rejects_wrong_error_size)


# ---------------------------------------------------------------------
H.section('End-to-end AO loop sanity')


def t_end_to_end_zero_aberration_steady_zero():
    """With no input wavefront aberration the closed loop should
    settle to ~zero DM command."""
    # Build a minimal AO pipeline: identity slopes from the WFS
    # would feed zero error into the integrator, so command stays zero.
    basis = la.zernike_modal_basis(n_modes=5, n_lenslets=6,
                                     semi_aperture=10e-3)
    integ = la.LeakyIntegrator(gain=0.4, leak=0.01, n_modes=5)
    for _ in range(20):
        zero_slopes = np.zeros(2 * basis['lenslet_xy'][0].size)
        modes = la.slope_to_modal(zero_slopes, basis)
        integ.update(modes)
    return float(np.abs(integ.command).max()) < 1e-12, \
        f'final cmd max = {float(np.abs(integ.command).max()):.2e}'


H.run('AO loop: zero-aberration input gives zero steady-state cmd',
      t_end_to_end_zero_aberration_steady_zero)


# ---------------------------------------------------------------------
def main():
    return H.summary()


if __name__ == '__main__':
    sys.exit(main())
