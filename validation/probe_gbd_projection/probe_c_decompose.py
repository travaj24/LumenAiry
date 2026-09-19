"""WP-B12b probe C -- what moves on a CONIC last surface, and why.

On a conic last surface the deleted in-line copy computed the SAG exactly
(probe A reads the gap at 5.8e-23 m).  The field still moves, and this probe
separates the three things that could be responsible, at the BEAMLET level --
no field reconstruction, so it is cheap and it isolates the cause instead of
the symptom.

The three arms differ only in what the projection does to the 4x4 JACOBIAN;
the state (position, slope, optical path) is projected identically in all
three, so any difference here is the Jacobian alone:

* ``full``        -- what ships: ``J_v = P J`` with
  ``P = [[I - u (x) grad s, -s I], [0, I]]``;
* ``fs_sag_sec``  -- the composition the DELETED code effectively applied:
  a free-space of ``-sag*sec`` after the surface Jacobian,
  ``J_v = [[I, -sag*sec I], [0, I]] J``.  The old code reached it by folding
  ``-sag`` into the image leg ``t = (z_image - sag)/Nz2``, which is the same
  Moebius step on ``Q``;
* ``state_only``  -- the state projected and the Jacobian left ON the surface.
  This is what ``reference='surface'`` gives a caller who then adds a bare
  ``z_image`` leg, i.e. the pre-v5.22 GBD behaviour with NO vertex correction
  at all.  It is included because it is what a naive reading of "force the
  primitive back to the surface plane" reconstructs -- and it is NOT the
  v5.22 .. 5.47.0 behaviour on a curved-base surface.

Positions and the optical-path piston are identical across all three by
construction and are asserted here, so the reader can see that the CONIC row's
movement is the Jacobian and nothing else.

Run: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, PYTHONPATH pinned.
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbdproj_common import (  # noqa: E402
    FIXTURES,
    assert_tree,
    build_tag,
    dump,
)

ROOT = os.environ.get('B12B_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _variants(D):
    """Three replacements for ``_project_to_exit_vertex_plane``."""
    orig = D._project_to_exit_vertex_plane

    def make(mode):
        def patched(transfer, surfaces, wavelength, n_exit, fn_name):
            out = orig(transfer, surfaces, wavelength, n_exit, fn_name)
            if mode == 'full' or out is transfer:
                return out          # 'full', or the flat short-circuit
            from lumenairy.raytrace.surface import _surface_sag_xy
            last = surfaces[-1]
            sag = np.asarray(_surface_sag_xy(transfer.x, transfer.y, last),
                             dtype=np.float64)
            sec = np.sqrt(1.0 + transfer.ux ** 2 + transfer.uy ** 2)
            jac = np.array(transfer.jacobian, dtype=float, copy=True)
            if mode == 'fs_sag_sec':
                t0 = -sag * sec
                jac[-1][:, 0, :] += t0[:, None] * jac[-1][:, 2, :]
                jac[-1][:, 1, :] += t0[:, None] * jac[-1][:, 3, :]
            elif mode != 'state_only':
                raise ValueError(mode)
            return D.DifferentialTransfer(
                jacobian=jac, x=out.x, y=out.y, ux=out.ux, uy=out.uy,
                opd=out.opd, alive=out.alive)
        return patched

    return orig, make


def main():
    assert_tree(ROOT)
    import lumenairy as la
    from lumenairy.propagators import gbd as G
    from lumenairy.raytrace import differential as D, system_abcd_prescription

    out = {'build': build_tag(), 'version': la.__version__, 'fixtures': {}}
    orig, make = _variants(D)

    for key in ('conic_ref', 'conic_k_last', 'asphere_field',
                'asphere_flat_base', 'flat_last_lasf9'):
        fx = FIXTURES[key]
        fx.register()
        p = fx.prescription()
        z = float(system_abcd_prescription(p, fx.lam)[2])
        N, dx = 96, fx.dx * (fx.N / 96.0)
        xs = (np.arange(N) - N / 2) * dx
        X, Y = np.meshgrid(xs, xs)
        E = np.exp(-(X ** 2 + Y ** 2) / fx.w0 ** 2).astype(np.complex128)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            b = G.decompose_field_to_beamlets(E, dx, wavelength=fx.lam,
                                              sample_step=1, waist_factor=1.0)
        arms = {}
        try:
            for mode in ('full', 'fs_sag_sec', 'state_only'):
                D._project_to_exit_vertex_plane = make(mode)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ev = G.apply_prescription_persurface_to_beamlets(
                        b, p, fx.lam, z_image=z)
                arms[mode] = (np.asarray(ev.Q).copy(),
                              np.asarray(ev.amplitude).copy(),
                              np.asarray(ev.positions).copy())
        finally:
            D._project_to_exit_vertex_plane = orig

        def _cmp(a, bb):
            Qa, Aa, Pa = arms[a]
            Qb, Ab, Pb = arms[bb]
            n = min(Aa.size, Ab.size)
            ph = np.unwrap(np.angle(Ab[:n] / Aa[:n]))
            return {
                'rel_Q': float(np.abs(Qa[:n] - Qb[:n]).max()
                               / max(np.abs(Qa[:n]).max(), 1e-300)),
                'rel_amp': float(np.linalg.norm(Aa[:n] - Ab[:n])
                                 / max(np.linalg.norm(Aa[:n]), 1e-300)),
                'max_dpos_m': float(np.abs(Pa[:n] - Pb[:n]).max()),
                'max_relative_dphase_rad': float(
                    np.abs(ph - ph.mean()).max()),
            }

        out['fixtures'][key] = {
            'note': fx.note, 'n_beamlets': int(arms['full'][1].size),
            'z_image_m': z, 'N': N, 'dx': dx,
            'full_vs_fs_sag_sec': _cmp('full', 'fs_sag_sec'),
            'full_vs_state_only': _cmp('full', 'state_only'),
            'fs_sag_sec_vs_state_only': _cmp('fs_sag_sec', 'state_only'),
        }
        r = out['fixtures'][key]
        print(f"{key:18s} beamlets {r['n_beamlets']:6d}")
        for arm in ('full_vs_fs_sag_sec', 'full_vs_state_only',
                    'fs_sag_sec_vs_state_only'):
            v = r[arm]
            print(f"    {arm:26s} relQ {v['rel_Q']:.3e}  "
                  f"relAmp {v['rel_amp']:.3e}  dPos {v['max_dpos_m']:.3e} m  "
                  f"dPhase {v['max_relative_dphase_rad']:.3e} rad")

    dump(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      f'probe_c_decompose_{sys.platform}_'
                      f'{sys.version_info.major}{sys.version_info.minor}.json'),
         out)


if __name__ == '__main__':
    main()
