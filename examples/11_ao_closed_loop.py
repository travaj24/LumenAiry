"""
Lumenairy example 11 -- adaptive-optics closed loop.

Demonstrates an AO closed loop built from the v5.2 primitives:

  * ``lumenairy.generate_turbulence_screen`` -- Kolmogorov atmosphere
  * ``lumenairy.DeformableMirror``           -- Gaussian-IF DM
  * The DM ``fit_phase`` projector closes the loop with a single
    least-squares step per iteration (no separate Shack-Hartmann
    measurement -- this is a clairvoyant / ideal-WFS controller for
    pedagogy).

The example draws a Kolmogorov phase screen on the pupil, then runs
10 closed-loop iterations where each iteration:

  1. Measures the current residual phase (perfect-WFS approximation).
  2. Projects it onto the DM influence-function basis.
  3. Adds the negated correction to the cumulative DM command (leaky
     integrator with gain g).
  4. Recomputes the residual phase.

Run::

    python examples/11_ao_closed_loop.py

A residual-RMS vs iteration plot is written to
``examples/output/11_ao_closed_loop.png``.

Note: this uses the v5.2 DM primitive plus a simple closed-loop
control law assembled by hand.  A higher-level "ao_closed_loop"
helper is not yet in the library -- the build-your-own pattern
here is the canonical v5.2 idiom.

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
    # Leaky-integrator control law:
    #   cmd_{k+1} = (1 - leak) * cmd_k + gain * fit(residual)
    # with gain = 0.5 and leak = 0.0 (pure integrator) for a quick
    # demonstration.  In practice gain ~ 0.3 - 0.5 keeps the loop
    # stable across temporal bandwidths.
    gain = 0.5
    leak = 0.0
    n_iter = 10
    rms_history = [rms_initial]

    for k in range(n_iter):
        # Residual = atmospheric phase + current DM phase (DM imparts
        # a phase via dm.phase()).  Note we ADD the DM phase because
        # the DM is conjugate to the pupil; the controller learns the
        # negative of the residual through fit_phase + cumulative
        # commands.
        residual = phase_atm + dm.phase()
        # Project the *negative* residual onto the DM basis -- the
        # incremental command needed to push the residual to zero.
        # fit_phase overwrites dm.command with the fit; instead use
        # a scratch DM to compute the increment, then accumulate.
        scratch = la.DeformableMirror(
            n_actuators=n_act, pitch=pitch, dx=dx, N=N,
            inter_actuator_coupling=0.2,
        )
        scratch.fit_phase(-residual)
        delta_cmd = scratch.command
        # Apply leaky-integrator update.
        dm.command = (1.0 - leak) * dm.command + gain * delta_cmd
        rms_k = _phase_rms(phase_atm + dm.phase(), pupil_mask)
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


if __name__ == '__main__':
    main()
