"""VERIFY-WP-B12b probe W3 -- the two open items the builder recorded, and
the ``world_output_plane`` branch's reference-plane DECISION, measured.

A. The MIRROR through the LOCAL branch.  ``Nz2 = 1/sec`` is positive whatever
   the true ``N``, so after a mirror the returned direction points the wrong
   way and the leg ``t = z_image/Nz2`` walks along an axis the light is no
   longer travelling.  Measured against my own 3-D trace: the transverse
   positions, the returned direction, and the base-ray PHASE, for
   ``z_image = +f`` and ``z_image = -f``, against the ``world_output_plane``
   branch on the same prescription.

B. The IMMERSED exit.  GBD's leg is ``exp(i k0 * z_image * sec)`` with no
   index, while the projection resolves ``n_exit``.  Measured: is the path
   REACHABLE (does a glass-exit prescription run at all?), and by how much
   does the returned base-ray phase miss my own tracer's optical path?

C. The ``world_output_plane`` branch's plane.  The shipped branch keeps
   ``reference='surface'``; the mutation ``world_exit_vertex`` asks for the
   other one.  Both are scored against my own diffraction oracle on the same
   world plane, so "would it double-count the sag?" is a measurement.

Author: VERIFY-WP-B12b
"""
from __future__ import annotations

import inspect
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vb12b_common import (  # noqa: E402
    assert_tree,
    dump,
    env_block,
    fidelity,
    fixtures,
    rel_l2,
    sha,
    to_vertex,
    trace3d,
)

FRAME = dict(sample_step=4, waist_factor=4.0)
_NR, _NAZ = 17, 8


def _fan(semi):
    r = np.linspace(semi / (2 * _NR), semi * 0.98, _NR)
    az = (np.arange(_NAZ) + 0.5) * (np.pi / _NAZ)
    R, A = np.meshgrid(r, az, indexing='ij')
    return (R * np.cos(A)).ravel(), (R * np.sin(A)).ravel()


def _bundle(h, y, lam, w0b):
    from lumenairy.propagators.gbd import BeamletBundle
    n = h.size
    z_R = np.pi * w0b ** 2 / lam
    return BeamletBundle(
        positions=np.stack([h, y, np.zeros_like(h)], axis=-1),
        directions=np.stack([np.zeros_like(h), np.zeros_like(h),
                             np.ones_like(h)], axis=-1),
        Q=np.full(n, -1j / z_R, dtype=np.complex128),
        amplitude=np.ones(n, dtype=np.complex128),
        waist0=np.full(n, float(w0b)))


def _my_state(fx, h, y):
    osurfs = fx.oracle_surfaces()
    z0 = np.zeros_like(h)
    st = trace3d(h, y, z0, z0, z0, np.ones_like(h), z0, osurfs)
    mv = to_vertex(st, fx.n_exit(), osurfs[-1]['zv'])
    return st, mv


def part_a_mirror(out):
    """The mirror through the LOCAL branch vs the WORLD branch."""
    from lumenairy.propagators import gbd as G
    fx = fixtures()['mirror']
    h, y = _fan(fx.semi)
    st, mv = _my_state(fx, h, y)
    # the geometric focus of a concave mirror R: f = R/2, BEHIND the vertex
    f = abs(fx.R2) / 2.0
    k0 = 2.0 * np.pi / fx.lam
    rows = []
    for zi in (f, -f):
        b = _bundle(h, y, fx.lam, 4.0 * fx.dx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = G.apply_prescription_persurface_to_beamlets(
                b, fx.prescription(), fx.lam, z_image=float(zi))
        pos = np.asarray(r.positions)
        drc = np.asarray(r.directions)
        amp = np.asarray(r.amplitude)
        # MY truth: the ray at the plane z = -f (the light goes toward -z)
        t_true = (-f - 0.0) / st['N']
        xt = mv['x'] + t_true * st['L']
        yt = mv['y'] + t_true * st['M']
        opl_t = mv['opl'] + fx.n_exit() * t_true
        # the reference arm: the same beamlets with z_image = 0, so the
        # phase difference isolates the LEG
        b0 = _bundle(h, y, fx.lam, 4.0 * fx.dx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r0 = G.apply_prescription_persurface_to_beamlets(
                b0, fx.prescription(), fx.lam, z_image=0.0)
        amp0 = np.asarray(r0.amplitude)
        leg_meas = np.angle(amp * np.conj(amp0))        # the leg's own phase
        leg_true = k0 * (opl_t - mv['opl'])
        d = np.angle(np.exp(1j * (leg_meas - leg_true)))
        rows.append(dict(
            z_image=float(zi),
            pos_err_vs_truth=float(np.nanmax(np.hypot(pos[:, 0] - xt,
                                                      pos[:, 1] - yt))),
            spot_rms=float(np.sqrt(np.mean((pos[:, 0] - pos[:, 0].mean()) ** 2
                                           + (pos[:, 1]
                                              - pos[:, 1].mean()) ** 2))),
            truth_spot_rms=float(np.sqrt(np.mean(
                (xt - xt.mean()) ** 2 + (yt - yt.mean()) ** 2))),
            dir_z_sign=float(np.sign(np.median(drc[:, 2]))),
            true_N_sign=float(np.sign(np.median(st['N']))),
            leg_phase_resid_waves=float(np.nanmax(np.abs(
                np.angle(np.exp(1j * (d - np.angle(
                    np.mean(np.exp(1j * d))))))))/ (2 * np.pi)),
            leg_piston_waves=float(np.angle(np.mean(np.exp(1j * d)))
                                   / (2 * np.pi)),
            leg_true_waves=float(np.nanmedian(leg_true) / (2 * np.pi)),
            leg_meas_waves_mod1=float(np.nanmedian(
                np.angle(np.exp(1j * leg_meas))) / (2 * np.pi)),
        ))
        print(f"  mirror LOCAL z_image {zi * 1e3:+.3f} mm: pos err "
              f"{rows[-1]['pos_err_vs_truth']:.4e} m, spot rms "
              f"{rows[-1]['spot_rms']:.4e} (truth "
              f"{rows[-1]['truth_spot_rms']:.4e}), returned N sign "
              f"{rows[-1]['dir_z_sign']:+.0f} vs true "
              f"{rows[-1]['true_N_sign']:+.0f}, leg phase piston "
              f"{rows[-1]['leg_piston_waves']:+.4f} waves "
              f"(resid {rows[-1]['leg_phase_resid_waves']:.2e})", flush=True)

    # the WORLD branch on the same prescription and the same plane
    world = None
    try:
        b = _bundle(h, y, fx.lam, 4.0 * fx.dx)
        from lumenairy.raytrace.world import world_surfaces_from_prescription
        lw = world_surfaces_from_prescription(fx.prescription())[-1]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            w = G.apply_prescription_persurface_to_beamlets(
                b, fx.prescription(), fx.lam,
                world_output_plane=(lw.world_origin
                                    - f * lw.world_R[:, 2], lw.world_R.copy()))
        wp = np.asarray(w.positions)
        t_true = (-f - 0.0) / st['N']
        xt = mv['x'] + t_true * st['L']
        yt = mv['y'] + t_true * st['M']
        n = min(wp.shape[0], xt.size)
        world = dict(n_returned=int(wp.shape[0]), n_rays=int(h.size),
                     pos_err_vs_truth=float(np.nanmax(np.hypot(
                         wp[:n, 0] - xt[:n], wp[:n, 1] - yt[:n]))),
                     spot_rms=float(np.sqrt(np.mean(
                         (wp[:, 0] - wp[:, 0].mean()) ** 2
                         + (wp[:, 1] - wp[:, 1].mean()) ** 2))),
                     dir_z_sign=float(np.sign(np.median(
                         np.asarray(w.directions)[:, 2]))))
        print(f"  mirror WORLD at z = -f: pos err "
              f"{world['pos_err_vs_truth']:.4e} m, spot rms "
              f"{world['spot_rms']:.4e}, returned N sign "
              f"{world['dir_z_sign']:+.0f}", flush=True)
    except Exception as e:                                  # noqa: BLE001
        world = dict(error=f'{type(e).__name__}: {e}')
        print('  mirror WORLD raised', world['error'], flush=True)
    out['mirror'] = dict(local=rows, world=world, f=float(f),
                         truth_spot_rms=rows[0]['truth_spot_rms'])


def part_b_immersed(out):
    """The vacuum-exit assumption in GBD's own image leg."""
    from lumenairy.propagators import gbd as G
    fx = fixtures()['immersed']
    h, y = _fan(fx.semi)
    st, mv = _my_state(fx, h, y)
    k0 = 2.0 * np.pi / fx.lam
    n_exit = fx.n_exit()
    z_img = 2.0e-3
    res = dict(n_exit=n_exit, z_image=z_img, reachable=None)
    try:
        b0 = _bundle(h, y, fx.lam, 4.0 * fx.dx)
        b1 = _bundle(h, y, fx.lam, 4.0 * fx.dx)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r0 = G.apply_prescription_persurface_to_beamlets(
                b0, fx.prescription(), fx.lam, z_image=0.0)
            r1 = G.apply_prescription_persurface_to_beamlets(
                b1, fx.prescription(), fx.lam, z_image=z_img)
        res['reachable'] = True
        a0 = np.asarray(r0.amplitude)
        a1 = np.asarray(r1.amplitude)
        ux = st['L'] / st['N']
        uy = st['M'] / st['N']
        sec = np.sqrt(1.0 + ux ** 2 + uy ** 2)
        leg_meas = np.angle(a1 * np.conj(a0))
        leg_true = k0 * n_exit * z_img * sec
        leg_vac = k0 * 1.0 * z_img * sec
        for nm, pred in (('with_n_exit', leg_true), ('vacuum', leg_vac)):
            d = np.angle(np.exp(1j * (leg_meas - pred)))
            piston = np.angle(np.mean(np.exp(1j * d)))
            res[f'resid_{nm}_waves'] = float(np.nanmax(np.abs(
                np.angle(np.exp(1j * (d - piston))))) / (2 * np.pi))
        res['missing_waves'] = float(np.nanmax(
            (n_exit - 1.0) * z_img * sec) / fx.lam)
        # the library's own guards, if any
        res['has_fga_immersed_guard'] = bool(
            '_require_non_immersed_exit'
            in inspect.getsource(sys.modules['lumenairy.propagators.fga']))
        import lumenairy as la
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                la.apply_real_lens_gbd(
                    fx.E_in(), prescription=fx.prescription(),
                    wavelength=fx.lam, dx=fx.dx,
                    output_plane_distance=float(z_img), sample_step=16,
                    waist_factor=16.0)
            res['apply_real_lens_gbd_immersed'] = 'served'
        except Exception as e:                              # noqa: BLE001
            res['apply_real_lens_gbd_immersed'] = f'{type(e).__name__}: {e}'
        print(f"  immersed: reachable={res['reachable']}, missing "
              f"{res['missing_waves']:.3f} waves; residual against the "
              f"VACUUM leg {res['resid_vacuum_waves']:.3e} wv, against the "
              f"n_exit leg {res['resid_with_n_exit_waves']:.3e} wv; "
              f"apply_real_lens_gbd -> {res['apply_real_lens_gbd_immersed']}",
              flush=True)
    except Exception as e:                                  # noqa: BLE001
        res['reachable'] = False
        res['error'] = f'{type(e).__name__}: {e}'
        print('  immersed raised', res['error'], flush=True)
    out['immersed'] = res


def part_c_world_plane(out, mutation=''):
    """Score the WORLD branch against my own oracle on an explicit world
    plane, under whichever reference plane this process's mutation selects."""
    from lumenairy.propagators import gbd as G
    res = []
    for key in ('conic', 'asph'):
        fx = fixtures()[key]
        zf = fx.best_focus()
        O, _i = fx.oracle_field(zf)
        E = fx.E_in()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            b = G.decompose_field_to_beamlets(E, fx.dx, wavelength=fx.lam,
                                              **FRAME)
            from lumenairy.raytrace.world import (
                world_surfaces_from_prescription,
            )
            lw = world_surfaces_from_prescription(fx.prescription())[-1]
            w = G.apply_prescription_persurface_to_beamlets(
                b, fx.prescription(), fx.lam,
                world_output_plane=(lw.world_origin
                                    + float(zf) * lw.world_R[:, 2],
                                    lw.world_R.copy()))
            F = np.asarray(G.reconstruct_field_from_beamlets(
                w, Ny=fx.N, Nx=fx.N, dx=fx.dx, wavelength=fx.lam,
                window=5.0))
        r = dict(key=key, mutation=mutation, z=float(zf),
                 fidelity=fidelity(F, O), rel_l2=rel_l2(F, O),
                 peak=float(np.abs(F).max()), sha=sha(F)[:24])
        res.append(r)
        print(f"  world-branch [{mutation or 'shipped'}] {key:8s} fid "
              f"{r['fidelity']:.8f} relL2 {r['rel_l2']:.5f} sha {r['sha']}",
              flush=True)
    out.setdefault('world_plane', []).extend(res)


def main():
    assert_tree()
    import lumenairy.propagators.fga  # noqa: F401  (for the guard probe)
    from lumenairy.propagators.gbd import (
        apply_prescription_persurface_to_beamlets as F_,
    )
    mutation = os.environ.get('VB12B_MUTATION', '')
    if mutation:
        import vb12b_mutate
        vb12b_mutate.apply_named(mutation)
    arm = 'pre' if '_Rl = float(' in inspect.getsource(F_) else 'post'
    print(f'ARM {arm}  MUTATION {mutation or "(none)"}')
    out = dict(env=env_block(), arm=arm, mutation=mutation)
    part_a_mirror(out)
    part_b_immersed(out)
    part_c_world_plane(out, mutation)
    dump(out, f'probe_w3_openitems_{arm}{("_" + mutation) if mutation else ""}')


if __name__ == '__main__':
    main()
