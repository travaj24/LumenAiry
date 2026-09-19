"""WP-B12b probe B -- the oracle ladder: the per-surface GBD field before and
after the in-line sag copy is replaced by the shared exit-vertex projection.

Run the SAME script in two trees and compare row by row:

    pre  = ``git archive <the WP-B12 head> lumenairy``  (the in-line copy)
    post = this package's tree                          (the shared projection)

The arm is DETECTED from the library, never asserted from the command line: a
tree whose ``apply_prescription_persurface_to_beamlets`` still defines ``_Rl``
is ``pre_b12b``, one that does not is ``post_b12b``.  The tag, the resolved
``lumenairy.__file__`` and the build are written into the JSON, and the output
file is named after the arm, so the two runs cannot overwrite each other.

What is scored
--------------
* Rotationally-symmetric fixtures (``gbdproj_common.FIXTURES``): the field
  ``apply_real_lens_gbd`` returns, against the independent
  Rayleigh-Sommerfeld-I oracle imported from
  ``validation/probe_wp_b12/b12_common.py``, at the exit vertex and at the
  fixture's own traced best focus -- fidelity, relative L2 after the best
  complex scale, power ratio, and the SHA-256 of the returned field's exact
  bytes.
* Non-rotationally-symmetric fixtures (biconic / freeform / field-frame /
  mirror): SHA-256, power and peak only.  The imported oracle is a
  rotationally-symmetric trace and cannot represent them, so their rows report
  MOVEMENT, never accuracy; their sag defect is measured at the ray level in
  probe A.
* The other public entries onto the same code path:
  ``propagate_gbd_through_prescription(per_surface=True)``,
  ``apply_real_lens_universal(method='gbd')`` and the ``world_output_plane``
  branch -- which keeps ``reference='surface'`` and must be byte-identical
  between the two arms.

Run: OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1, PYTHONPATH pinned.
"""
from __future__ import annotations

import hashlib
import inspect
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gbdproj_common import (  # noqa: E402
    FIXTURES,
    NONSYM,
    assert_tree,
    build_tag,
    dump,
    fidelity,
    nonsym_beam,
    rel_l2,
)

ROOT = os.environ.get('B12B_TREE', os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(a))).hexdigest()[:24]


def _arm():
    from lumenairy.propagators.gbd import (
        apply_prescription_persurface_to_beamlets as f,
    )
    inline = '_Rl = float(' in inspect.getsource(f)
    return ('pre_b12b' if inline else 'post_b12b'), inline


def _gbd(E, presc, lam, dx, z):
    from lumenairy.elements import apply_real_lens_gbd
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return np.asarray(apply_real_lens_gbd(
            E, prescription=presc, wavelength=lam, dx=dx,
            output_plane_distance=float(z)))


def _score(fx, z, label):
    t0 = time.time()
    f = _gbd(fx.beam(), fx.prescription(), fx.lam, fx.dx, z)
    t_gbd = time.time() - t0
    t0 = time.time()
    o = fx.oracle_field(float(z))
    t_or = time.time() - t0
    pg = float((np.abs(f) ** 2).sum())
    po = float((np.abs(o) ** 2).sum())
    return {
        'plane': label,
        'z_m': float(z),
        'fidelity': fidelity(f, o),
        'rel_l2': rel_l2(f, o),
        'power_ratio_gbd_over_oracle': (pg / po) if po > 0 else float('nan'),
        'sha_field': _sha(f),
        'gbd_power': pg,
        'oracle_power': po,
        'seconds_gbd': t_gbd,
        'seconds_oracle': t_or,
    }


def main():
    assert_tree(ROOT)
    import lumenairy as la
    arm, inline = _arm()
    print(f'arm = {arm}  (the in-line conic-sag copy is present: {inline})',
          flush=True)

    out = {'build': build_tag(), 'version': la.__version__, 'arm': arm,
           'inline_copy_present': inline,
           'lumenairy_file': os.path.abspath(la.__file__),
           'symmetric': {}, 'nonsym': {}, 'entry_points': {}}
    # Dump after EVERY row, not once at the end: these runs take an hour or
    # more on a loaded box, and a crash in a later section must not throw away
    # the rows already measured (it did, once, on a stale
    # ``reconstruct_field_from_beamlets`` signature).
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'probe_b_ladder_{arm}_{sys.platform}_'
        f'{sys.version_info.major}{sys.version_info.minor}.json')

    # ---- 1. the rotationally-symmetric ladder, against the oracle ---------
    for key, fx in FIXTURES.items():
        fx.register()
        zf = fx.best_focus()
        rows = [_score(fx, 0.0, 'exit_vertex'), _score(fx, zf, 'focus')]
        out['symmetric'][key] = {
            'note': fx.note, 'N': fx.N, 'dx': fx.dx, 'lam': fx.lam,
            'best_focus_m': zf, 'airy_radius_m': fx.airy_radius(),
            'dx_over_airy': fx.dx / fx.airy_radius(),
            'rows': rows,
        }
        dump(out_path, out)
        for r in rows:
            print(f"  {key:18s} {r['plane']:11s} z={r['z_m'] * 1e3:8.4f} mm  "
                  f"fid={r['fidelity']:.6f}  "
                  f"P={r['power_ratio_gbd_over_oracle']:.4f}  "
                  f"sha={r['sha_field']}  ({r['seconds_gbd']:.1f}s gbd / "
                  f"{r['seconds_oracle']:.1f}s oracle)", flush=True)

    # ---- 2. the non-symmetric fixtures: movement only ---------------------
    from lumenairy.raytrace import system_abcd_prescription
    for key, spec in NONSYM.items():
        p = spec['presc']()
        E = nonsym_beam(spec)
        try:
            z = float(system_abcd_prescription(p, spec['lam'])[2])
        except Exception as exc:                       # noqa: BLE001
            z = 0.0
            print(f'  {key}: BFL unavailable ({type(exc).__name__}), z = 0',
                  flush=True)
        rows = {}
        for lab, zz in (('exit_vertex', 0.0), ('bfl', z)):
            try:
                t0 = time.time()
                f = _gbd(E, p, spec['lam'], spec['dx'], zz)
                rows[lab] = {'z_m': zz, 'sha_field': _sha(f),
                             'power': float((np.abs(f) ** 2).sum()),
                             'peak': float(np.abs(f).max()),
                             'seconds': time.time() - t0}
            except Exception as exc:                   # noqa: BLE001
                rows[lab] = {'z_m': zz,
                             'error': f'{type(exc).__name__}: {exc}'}
        out['nonsym'][key] = {'note': spec['note'], 'rows': rows}
        dump(out_path, out)
        for lab, r in rows.items():
            print(f"  {key:12s} {lab:11s} " + (
                r['error'] if 'error' in r else
                f"sha={r['sha_field']} P={r['power']:.8e} "
                f"peak={r['peak']:.8e}"), flush=True)

    # ---- 3. the other entry points onto the same path --------------------
    fx = FIXTURES['asphere_flat_base']
    fx.register()
    p, E = fx.prescription(), fx.beam()
    zf = fx.best_focus()
    ep = {}

    from lumenairy.propagators.gbd import propagate_gbd_through_prescription
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f = np.asarray(propagate_gbd_through_prescription(
            E, fx.dx, p, wavelength=fx.lam, output_shape=(fx.N, fx.N),
            output_dx=fx.dx, per_surface=True, z_image=float(zf)))
    o = fx.oracle_field(float(zf))
    ep['propagate_gbd_through_prescription_per_surface'] = {
        'sha_field': _sha(f), 'fidelity': fidelity(f, o),
        'rel_l2': rel_l2(f, o)}

    from lumenairy.propagators.fga import apply_real_lens_universal
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        f = np.asarray(apply_real_lens_universal(
            E, prescription=p, wavelength=fx.lam, dx=fx.dx, method='gbd',
            output_plane_distance=float(zf)))
    ep['apply_real_lens_universal_method_gbd'] = {
        'sha_field': _sha(f), 'fidelity': fidelity(f, o),
        'rel_l2': rel_l2(f, o)}

    # the WORLD-frame branch keeps reference='surface' and must NOT move.
    from lumenairy.propagators.gbd import (
        apply_prescription_persurface_to_beamlets,
        decompose_field_to_beamlets,
        reconstruct_field_from_beamlets,
    )
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        b = decompose_field_to_beamlets(E, fx.dx, wavelength=fx.lam,
                                        sample_step=8, waist_factor=8.0)
        ev = apply_prescription_persurface_to_beamlets(
            b, p, fx.lam, world_output_plane='auto')
        fw = np.asarray(reconstruct_field_from_beamlets(
            ev, Ny=fx.N, Nx=fx.N, dx=fx.dx, wavelength=fx.lam,
            window=5.0))
    ep['world_output_plane_branch'] = {
        'sha_field': _sha(fw), 'power': float((np.abs(fw) ** 2).sum()),
        'n_beamlets': int(len(ev))}

    out['entry_points'] = ep
    for k, v in ep.items():
        print(f"  entry {k:48s} sha={v['sha_field']}"
              + (f" fid={v['fidelity']:.6f}" if 'fidelity' in v else ''),
              flush=True)

    dump(out_path, out)


if __name__ == '__main__':
    main()
