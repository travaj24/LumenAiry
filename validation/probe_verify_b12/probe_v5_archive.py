"""VERIFY-WP-B12 probe V5 -- tree-to-tree byte identity, in a child process.

Run once in a worktree of ``96cb2096`` (the parent) and once in the WP-B12
tree, each time with ``cwd`` and ``PYTHONPATH`` pinned to that tree, and with
``lumenairy.__file__`` asserted to live under it.  Nothing here goes through
pytest, and neither run can see the other tree's ``lumenairy``.

Outputs, per fixture and per plane: the SHA-256 of ``apply_real_lens_fga``'s
exact field bytes and the ``_caustic_zone`` edges printed to 17 digits.  A
FLAT-last-surface fixture must hash the same on both trees; a curved one must
not.  This is the check that makes probe V3's in-process ``forced_surface``
arm an assertion rather than an assumption.

**Pin the memory budget.**  ``apply_real_lens_fga``'s returned BYTES depend on
the momentum and position-lattice chunking, and the chunking on
``mem_budget_mb``, which defaults to a fraction of the AVAILABLE RAM at call
time -- measured, nine distinct digests for one field over a ``chunk`` sweep
(VERIFY-B12 defect D-4).  Export ``LUMENAIRY_MEM_BUDGET_MB`` to the same value
in both child processes, as ``run_rest_win.sh`` does; without it a run on a
busy box can read a FALSE regression on the flat controls.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vb12_common as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tree', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--only', default='')
    args = ap.parse_args()

    import lumenairy as la
    path = os.path.abspath(la.__file__)
    tree = os.path.abspath(args.tree)
    assert path.lower().startswith(tree.lower()), (
        f'lumenairy resolved to {path}, not under {tree}')
    print(f'TREE {args.tag}: lumenairy.__file__ = {path}', flush=True)
    from lumenairy.propagators.fga import _caustic_zone, apply_real_lens_fga
    from lumenairy.raytrace import differential as D
    has_ref = 'reference' in getattr(
        D.ray_transfer_jacobian, '__doc__', '') or ''
    out = {'tag': args.tag, 'lumenairy_file': path,
           'version': la.__version__,
           'primitive_has_reference_keyword': bool(has_ref),
           'fixtures': {}}
    keys = [k for k in args.only.split(',') if k] or None
    for fx in C.fixtures():
        if fx.mirror_last or (keys and fx.key not in keys):
            continue
        zf = fx.best_focus()
        E_in = fx.E_in()
        rec = {'flat_last': fx.flat_last, 'best_focus_m': repr(zf)}
        for nm, z in (('vertex', 0.0), ('focus', zf)):
            F = apply_real_lens_fga(E_in, prescription=fx.prescription(),
                                    wavelength=fx.lam, dx=fx.dx,
                                    output_plane_distance=z)
            rec[nm] = dict(sha=C.sha(F), l2=repr(float(np.linalg.norm(F))))
        zone = _caustic_zone(E_in, fx.dx, fx.prescription(), fx.lam)
        rec['caustic_zone'] = [repr(float(zone[0])), repr(float(zone[1]))]
        out['fixtures'][fx.key] = rec
        print(f"  [{fx.key:17s}] flat={fx.flat_last} vertex={rec['vertex']['sha'][:16]}"
              f" focus={rec['focus']['sha'][:16]} zone={rec['caustic_zone']}",
              flush=True)
    with open(args.out, 'w', encoding='cp1252') as fh:
        json.dump(out, fh, indent=1)
    print(f'WROTE {args.out}', flush=True)


if __name__ == '__main__':
    main()
