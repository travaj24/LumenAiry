"""H2-2 (item 18) -- the NumPy Collins chain, archive-to-archive.

Run as a CHILD process bound to ONE tree:

    python probe_collins_bitid.py <tree> <out.json>

The claim being proved: threading ``(xp, is_jax, bld)`` through
``_collins_transport`` and its helper chain moves NOTHING on the NumPy path.
Every carrier-chain fixture that reaches ``_collins_transport`` is driven, and
the whole record is digested -- returned arrays, the returned carrier and
pitch, the class of the returned object, and every warning in EMISSION order
(the Kelly K1/K2/K3 guard speaks through warnings, so a reordered or reworded
guard is a moved key here).

Which fixtures REACH the transport, found by grepping the tree rather than
assumed: the direct entry with ``transport='collins'``, the focus readout, the
exact focus readout (whose readout leg is the same chirp-Z), the multi-leg
chain, and the two existing validation probes' own collins cells
(``validation/probe_known_reds/probe_carrier_attribution.py`` and
``validation/probe_verify_b14/probe_v5_two_caller.py``), restated here so this
campaign's claim rests on this campaign's probe.

``C-jax-*`` and ``C-trace-*`` keys exist only on the branch -- they are the new
behaviour, and the driver is told to expect them as branch-only.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import hlib  # noqa: E402

import numpy as np  # noqa: E402

hlib.anchor(_TREE)

import lumenairy.propagators.carrier as CA  # noqa: E402

WL = 633e-9


def _gauss(n, dx, w, dtype=np.complex128):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / (w * w)).astype(dtype)


def section_public(p):
    E = _gauss(64, 8e-6, 60e-6)
    E32 = _gauss(64, 8e-6, 60e-6, dtype=np.complex64)
    base = dict(wavelength=WL, dx=8e-6)
    for tag, R, z in (('short', -0.05, 5e-3), ('long', -0.02, 0.05),
                      ('back', -0.05, -3e-3), ('near', -0.02, 0.0199)):
        for gk in ('auto', 'fresnel', 'exact'):
            p.call(f"C-direct-{tag}-{gk}", CA.propagate_carrier_referenced,
                   E, R, z, **dict(base, transport='collins', gap_kernel=gk,
                                   on_collins_sampling='warn'))
            p.call(f"C-direct-dxout-{tag}-{gk}",
                   CA.propagate_carrier_referenced, E, R, z,
                   **dict(base, transport='collins', gap_kernel=gk,
                          dx_out=8e-6 / 64.0, on_collins_sampling='warn'))
            p.call(f"C-direct-flatref-{tag}-{gk}",
                   CA.propagate_carrier_referenced, E, R, z,
                   **dict(base, transport='collins', gap_kernel=gk,
                          dx_out=8e-6 / 8.0, carrier_out=float('inf'),
                          on_collins_sampling='warn'))
    # astigmatic carrier -- the per-axis arithmetic arm
    for gk in ('auto', 'fresnel'):
        p.call(f"C-astig-{gk}", CA.propagate_carrier_referenced,
               E, (-0.05, -0.08), 5e-3,
               **dict(base, transport='collins', gap_kernel=gk,
                      on_collins_sampling='warn'))
    p.call("C-astig-exact-refused", CA.propagate_carrier_referenced,
           E, (-0.05, -0.08), 5e-3,
           **dict(base, transport='collins', gap_kernel='exact',
                  on_collins_sampling='warn'))
    # complex64 -- the no-silent-upcast contract
    p.call("C-c64", CA.propagate_carrier_referenced, E32, -0.05, 5e-3,
           **dict(base, transport='collins', on_collins_sampling='warn'))
    # tilt, which only the exact kernel can carry
    p.call("C-tilt-exact", CA.propagate_carrier_referenced, E, -0.05, 5e-3,
           **dict(base, transport='collins', gap_kernel='exact',
                  tilt=(0.12, 0.03), on_collins_sampling='warn'))
    p.call("C-tilt-fresnel-inert", CA.propagate_carrier_referenced,
           E, -0.05, 5e-3,
           **dict(base, transport='collins', gap_kernel='fresnel',
                  tilt=(0.12, 0.0), on_collins_sampling='warn'))
    # the guard's three dispositions, on a leg that violates it
    for action in ('ignore', 'warn', 'error'):
        p.call(f"C-guard-{action}", CA.propagate_carrier_referenced,
               _gauss(64, 40e-6, 500e-6), -0.30, 0.29,
               wavelength=1.55e-6, dx=40e-6, transport='collins',
               dx_out=40e-6 * 16, on_collins_sampling=action)
    # sziklas alongside, to prove the OTHER transport did not move either
    for tag, R, z in (('short', -0.05, 5e-3), ('long', -0.02, 0.05)):
        for gk in ('auto', 'fresnel'):
            p.call(f"C-sziklas-{tag}-{gk}",
                   CA.propagate_carrier_referenced, E, R, z,
                   **dict(base, transport='sziklas', gap_kernel=gk))
    # coarse grid, a second wavelength
    p.call("C-coarse", CA.propagate_carrier_referenced,
           _gauss(32, 40e-6, 120e-6), -0.30, 0.20,
           wavelength=1.55e-6, dx=40e-6, transport='collins',
           on_collins_sampling='warn')


def section_readouts(p):
    E = _gauss(64, 8e-6, 60e-6)
    big = _gauss(128, 4e-6, 120e-6)
    p.call("C-focus-readout", CA.carrier_referenced_focus_readout,
           E, -0.05, 5e-3, wavelength=WL, dx=8e-6, dx_out=8e-6, N_out=32)
    p.call("C-focus-readout-replica", CA.carrier_referenced_focus_readout,
           E, -0.05, 5e-3, wavelength=WL, dx=8e-6, dx_out=8e-6 * 8,
           N_out=256, on_replica='warn', on_focus_containment='warn')
    p.call("C-exact-focus-readout",
           CA.carrier_referenced_exact_focus_readout, big, -0.05, 4.9e-2,
           wavelength=WL, dx=4e-6, dx_out=4e-6 * 40, N_out=256,
           on_readout_window='warn', on_replica='warn',
           on_n_fine_cap='warn', n_fine_cap=64)


def section_helpers(p):
    """The helper chain directly, which is where the ``xp`` went in."""
    env = _gauss(64, 8e-6, 60e-6)
    from lumenairy.propagators.fft_infra import _fft2
    S = _fft2(np.ascontiguousarray(env, dtype=np.complex128))
    p.call("C-h-angle-support", CA._collins_angle_support, S, 8e-6, 8e-6,
           WL, 1e-4)
    p.call("C-h-space-support", CA._collins_space_support, env, 8e-6, 8e-6,
           1e-4)
    p.call("C-h-input-box", CA._collins_input_box, env, 8e-6, 8e-6, WL, 1e-4)
    p.call("C-h-power-marginals", CA._collins_power_marginals, env)
    for R in (1e-3, -1e-3, 5e-2):
        for off in (0.0, 1.3e-5):
            for dt in (np.complex128, np.complex64):
                p.call(f"C-h-axis-chirp-{R}-{off}-{np.dtype(dt).name}",
                       CA._collins_axis_chirp, 64, 8e-6, WL, R, off, dt)
    for z_eff in (1e-3, -5e-2, 12.0):
        for tilt in ((0.0, 0.0), (0.12, 0.03)):
            p.call(f"C-h-exact-corr-{z_eff}-{tilt[0]}",
                   CA._collins_exact_kernel_correction, S, z_eff, WL,
                   8e-6, 8e-6, tilt)
    p.call("C-h-exact-corr-evanescent-tilt",
           CA._collins_exact_kernel_correction, S, 1e-3, WL, 8e-6, 8e-6,
           (0.9, 0.9))
    p.call("C-h-sampling-stats", CA._collins_sampling_stats,
           0.3, 5e-3, -1.0, 2.0, 8e-6, 8e-6, 1.7e-4, 1.7e-4, 2e-3, 2e-3,
           1e-6, 1e-6, 64, 64, (0.0, 0.0), WL)
    p.call("C-h-kernel-wrap-ratio", CA._collins_kernel_wrap_ratio,
           12.0, 0.02, 5.12e-4)
    p.call("C-h-envelope-abcd", CA._collins_envelope_abcd, -0.05, 5e-3,
           -0.045)
    # the private transport, with every keyword the callers use
    p.call("C-h-transport", CA._collins_transport, env, -0.05, 5e-3, WL,
           8e-6, 8e-6, dx_out=8e-6, dy_out=8e-6, N_out_x=64, N_out_y=64,
           R_ref=-0.045, on_collins_sampling='warn')
    p.call("C-h-transport-flat", CA._collins_transport, env, -0.05, 5e-3,
           WL, 8e-6, 8e-6, dx_out=1e-6, dy_out=1e-6, N_out_x=48,
           N_out_y=48, R_ref=float('inf'), on_collins_sampling='warn',
           centre_out=(1.1e-5, -3e-6), check_period=True)
    p.call("C-h-transport-B0", CA._collins_transport, env, -0.05, 0.0, WL,
           8e-6, 8e-6, dx_out=8e-6, dy_out=8e-6, N_out_x=64, N_out_y=64,
           R_ref=-0.05)
    st = {}
    p.call("C-h-transport-stats", CA._collins_transport, env, -0.05, 5e-3,
           WL, 8e-6, 8e-6, dx_out=8e-6, dy_out=8e-6, N_out_x=64, N_out_y=64,
           R_ref=-0.045, stats_out=st, on_collins_sampling='warn')
    p.add("C-h-transport-stats-dict", {k: st[k] for k in sorted(st)})


def main():
    out_path = sys.argv[2]
    p = hlib.Probe()
    section_public(p)
    section_readouts(p)
    section_helpers(p)
    p.write(out_path)
    print(f"[probe_collins_bitid] {len(p.out)} keys, "
          f"build={hlib.build_tag()}")


if __name__ == '__main__':
    main()
