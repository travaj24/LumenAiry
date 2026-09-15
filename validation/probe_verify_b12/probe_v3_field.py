"""VERIFY-WP-B12 probe V3 -- the FGA field against MY diffraction oracle.

``apply_real_lens_fga`` is scored at the exit vertex and at each fixture's own
geometric best focus, in two arms of the same process:

* ``native``          -- the tree as it stands;
* ``forced_surface``  -- both differential primitives wrapped so that the
  ``reference`` keyword is DISCARDED, i.e. exactly the pre-WP-B12 behaviour.
  (``probe_v5_archive.py`` re-runs the same two planes in a separate child
  process against the real ``96cb2096`` tree, so this arm is checked rather
  than asserted.)

The oracle is the band-limited angular spectrum of MY geometrical-optics exit
field (``vb12_common``), which shares no code with the library; its own floor
is measured per reading by halving the source spacing and the output
refinement.

FLAT-last-surface fixtures must come back BIT-IDENTICAL between the two arms
(``np.array_equal`` and the same SHA-256), because the projection is a
structural no-op there.

``--only`` limits the run to a comma-separated list of fixture keys.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402


def _install_forced_surface():
    """Wrap both primitives so ``reference`` is discarded -- the pre-WP-B12
    contract.  The wrappers are created ONCE so ``fga._pick_ray_transfer``'s
    ``is not _rtja`` backend test still compares the same objects."""
    import lumenairy.raytrace.differential as D
    if getattr(D, '_vb12_forced', False):
        return
    orig_fd = D.ray_transfer_jacobian
    orig_an = D.ray_transfer_jacobian_analytic

    def mk(fn):
        def w(*a, **kw):
            kw.pop('reference', None)
            kw.pop('n_exit', None)
            return fn(*a, **kw)
        w.__name__ = fn.__name__
        return w

    D.ray_transfer_jacobian = mk(orig_fd)
    D.ray_transfer_jacobian_analytic = mk(orig_an)
    D._vb12_forced = True
    D._vb12_originals = (orig_fd, orig_an)


def _remove_forced_surface():
    import lumenairy.raytrace.differential as D
    if not getattr(D, '_vb12_forced', False):
        return
    D.ray_transfer_jacobian, D.ray_transfer_jacobian_analytic = \
        D._vb12_originals
    D._vb12_forced = False


def run(fx, planes, arm):
    from lumenairy.propagators.fga import _caustic_zone, apply_real_lens_fga
    E_in = fx.E_in()
    out = {}
    for name, z in planes.items():
        t0 = time.time()
        E = apply_real_lens_fga(E_in, prescription=fx.prescription(),
                                wavelength=fx.lam, dx=fx.dx,
                                output_plane_distance=z)
        dt = time.time() - t0
        out[name] = dict(z=z, sha=C.sha(E), seconds=dt,
                         field_l2=float(np.linalg.norm(E)))
        out[name]['_field'] = E
    zone = _caustic_zone(E_in, fx.dx, fx.prescription(), fx.lam)
    out['caustic_zone'] = [float(zone[0]), float(zone[1])]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default='')
    ap.add_argument('--refine', type=int, default=4)
    args = ap.parse_args()

    import lumenairy as la
    print('lumenairy.__file__ =', os.path.abspath(la.__file__), flush=True)
    keys = [k for k in args.only.split(',') if k] or None
    res = {'env': C.env_block(), 'fixtures': {}}

    for fx in C.fixtures():
        if fx.mirror_last:
            continue                    # FGA's image leg is +z: not an input
        if keys and fx.key not in keys:
            continue
        zf = fx.best_focus()
        planes = {'vertex': 0.0, 'focus': zf}

        # -- oracle, once per plane, with its own floor -------------------
        oracle = {}
        for nm, z in planes.items():
            O, info = fx.oracle_field(z, refine=args.refine)
            cv = fx.oracle_converge(z, refine=args.refine)
            oracle[nm] = dict(field=O, info=info, converge=cv)

        _remove_forced_surface()
        native = run(fx, planes, 'native')
        _install_forced_surface()
        forced = run(fx, planes, 'forced_surface')
        _remove_forced_surface()

        rec = dict(note=fx.note, flat_last=fx.flat_last, lam=fx.lam,
                   best_focus_m=zf, planes={}, caustic_zone=dict(
                       native=native['caustic_zone'],
                       forced_surface=forced['caustic_zone']))
        for nm in planes:
            O = oracle[nm]['field']
            En = native[nm]['_field']
            Ef = forced[nm]['_field']
            rec['planes'][nm] = dict(
                z=planes[nm],
                oracle_floor=oracle[nm]['converge'],
                oracle_mode=oracle[nm]['info'],
                native=dict(fidelity=C.fidelity(En, O),
                            power=C.power_ratio(En, O),
                            sha=native[nm]['sha'],
                            seconds=native[nm]['seconds']),
                forced_surface=dict(fidelity=C.fidelity(Ef, O),
                                    power=C.power_ratio(Ef, O),
                                    sha=forced[nm]['sha'],
                                    seconds=forced[nm]['seconds']),
                bit_identical=bool(np.array_equal(En, Ef)),
                max_abs_diff=float(np.max(np.abs(En - Ef))),
            )
            r = rec['planes'][nm]
            print(f"[{fx.key:17s}/{nm:6s}] pre={r['forced_surface']['fidelity']:.4f}"
                  f" (P {r['forced_surface']['power']:.3f})  ->  post="
                  f"{r['native']['fidelity']:.4f} (P "
                  f"{r['native']['power']:.3f})  bit-identical="
                  f"{r['bit_identical']}  oracle floor "
                  f"{1 - r['oracle_floor']['grid_halved_fidelity']:.1e}",
                  flush=True)
        z_n = rec['caustic_zone']['native']
        z_f = rec['caustic_zone']['forced_surface']
        print(f"[{fx.key:17s}/zone  ] pre=[{z_f[0]*1e6:.3f}, {z_f[1]*1e6:.3f}]"
              f" um -> post=[{z_n[0]*1e6:.3f}, {z_n[1]*1e6:.3f}] um  "
              f"(near edge moves {(z_n[0]-z_f[0])/zf*100:+.3f} % of f)",
              flush=True)
        res['fixtures'][fx.key] = rec
    C.dump(res, 'probe_v3_field')


if __name__ == '__main__':
    main()
