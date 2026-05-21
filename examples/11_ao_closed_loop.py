"""
Lumenairy example 11 -- adaptive-optics closed loop.

Demonstrates a single-conjugate AO closed loop using v5.2.3's
canonical ``lumenairy.ao_closed_loop`` helper:

  * ``lumenairy.generate_turbulence_screen`` -- Kolmogorov atmosphere
  * ``lumenairy.DeformableMirror``           -- Gaussian-IF DM
  * ``lumenairy.ao_closed_loop``             -- closed-loop driver
    (leaky integrator + DM ``fit_phase`` projector).  Returns the
    final residual phase and a per-iteration RMS history.

The example draws a Kolmogorov phase screen on the pupil, then runs
10 closed-loop iterations with ideal phase sensing.  Each iteration:

  1. Measures the current residual phase (perfect-WFS approximation).
  2. Projects it onto the DM influence-function basis.
  3. Adds the gain-scaled increment to the cumulative DM command.
  4. Recomputes the residual phase.

v5.2.3 demonstrates the canonical ``ao_closed_loop`` helper: the
v5.2.0 build-it-yourself pattern (manual scratch-DM + cumulative
command bookkeeping) is replaced by a single library call.

Run::

    python examples/11_ao_closed_loop.py

A residual-RMS vs iteration plot is written to
``examples/output/11_ao_closed_loop.png``.

Author: Andrew Traverso -- v5.2 / examples roadmap.
"""
from __future__ import annotations

import os as _os

import numpy as np

import lumenairy as la


def _phase_rms(phase, mask):
    """RMS of ``phase`` over the masked support, in radians."""
    p = phase[mask]
    p = p - p.mean()
    return float(np.sqrt(np.mean(p * p)))


def main():
    # --- 1. Build the pupil + Kolmogorov turbulence screen ------------
    # 128x128 grid, 30 mm full extent, 8-mm clear pupil.  r0 = 2 mm
    # gives a strong-turbulence regime (D/r0 = 4) so the closed-loop
    # convergence is visually obvious.
    N = 128
    dx = 30e-3 / N
    wavelength = 1.55e-6
    aperture = 8e-3
    r0 = 2e-3

    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X * X + Y * Y)
    pupil_mask = R <= aperture / 2

    # Atmospheric phase screen [rad].  Use a fixed seed for
    # reproducibility.
    phase_atm = la.generate_turbulence_screen(
        N=N, dx=dx, r0=r0, seed=0)
    # Mask outside the pupil so the RMS metric tracks the actual
    # corrected region.
    phase_atm = phase_atm * pupil_mask

    rms_initial = _phase_rms(phase_atm, pupil_mask)
    print(f'  Initial turbulence RMS phase (pupil): '
          f'{rms_initial:.3f} rad '
          f'({rms_initial * wavelength / (2 * np.pi) * 1e9:.1f} nm WFE)')

    # --- 2. Build the DM ---------------------------------------------
    # 9x9 actuator grid spanning the pupil (1 mm pitch).  The actuator
    # pitch should be ~ r0 to track the dominant turbulence modes.
    n_act = 9
    pitch = aperture / (n_act - 1)
    dm = la.DeformableMirror(
        n_actuators=n_act, pitch=pitch, dx=dx, N=N,
        inter_actuator_coupling=0.2,
    )

    # --- 3. Closed-loop iterations ------------------------------------
    # v5.2.3: one-call closed-loop helper.  Leaky-integrator gain 0.5
    # and ideal phase sensing (``wfs=None``) replicate the v5.2.0
    # pattern.  ``return_history=True`` records per-iteration residual
    # RMS for the convergence plot below.
    gain = 0.5
    n_iter = 10
    residual_final, history = la.ao_closed_loop(
        phase_atm,
        dm=dm,
        n_iterations=n_iter,
        gain=gain,
        wavelength=wavelength,
        dx=dx,
        wfs=None,
        return_history=True,
    )

    # The helper records the global (full-grid) RMS each iteration.
    # For the convergence plot we want the pupil-masked RMS so the
    # numbers match the labelled "initial pupil RMS" line above.
    # Reconstruct per-iteration pupil-masked RMS by re-running a
    # second pass starting from a flat DM and recording the masked
    # statistic after each step.
    dm.reset()
    rms_history = [rms_initial]
    for k in range(n_iter):
        la.ao_closed_loop(
            phase_atm, dm=dm, n_iterations=1, gain=gain,
            wavelength=wavelength, dx=dx, wfs=None)
        rms_k = _phase_rms(phase_atm - dm.phase(), pupil_mask)
        rms_history.append(rms_k)
        print(f'  iter {k+1:2d}:  residual RMS = {rms_k:.3f} rad  '
              f'({rms_k * wavelength / (2*np.pi) * 1e9:5.1f} nm WFE)')

    rms_final = rms_history[-1]

    # --- 4. Plot residual-RMS-vs-iteration ---------------------------
    out_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                            'output')
    _os.makedirs(out_dir, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib not available; skipping AO figure.')
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        iters = np.arange(len(rms_history))
        ax.semilogy(iters, rms_history, '-o', color='C0')
        ax.set_xlabel('Closed-loop iteration')
        ax.set_ylabel('Residual phase RMS [rad]')
        ax.set_title(
            f'11_ao_closed_loop.py -- {n_act}x{n_act} DM, '
            f'gain={gain}, r0={r0*1e3:.1f} mm')
        ax.grid(True, which='both', alpha=0.3)
        plt.tight_layout()
        out_path = _os.path.join(out_dir, '11_ao_closed_loop.png')
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f'  Figure saved to {out_path}')

    # --- 5. Summary --------------------------------------------------
    print()
    print(f'  AO closed-loop convergence ({n_iter} iterations):')
    print(f'    Initial RMS WFE:  '
          f'{rms_initial * wavelength / (2*np.pi) * 1e9:.1f} nm '
          f'({rms_initial:.3f} rad)')
    print(f'    Final RMS WFE:    '
          f'{rms_final * wavelength / (2*np.pi) * 1e9:.1f} nm '
          f'({rms_final:.3f} rad)')
    print(f'    Convergence ratio: '
          f'{rms_final/rms_initial:.3f} '
          f'(lower = better)')
    # Reference the helper's global-RMS history too.
    print(f'    Helper history (global RMS) final: '
          f'{history["rms_per_iter"][-1]:.4f} rad')


if __name__ == '__main__':
    main()
