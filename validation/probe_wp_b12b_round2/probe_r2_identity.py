"""R2 -- the shipped defaults are byte-identical where no guard fires.

Run in BOTH trees (PRE = my own ``git archive`` of the integration tip
76019ede into ``C:\\tmp\\lum_gbd2_pre``, POST = this branch) and on BOTH
builds, with ``LUMENAIRY_MEM_BUDGET_MB`` PINNED on the command line
(VERIFY-WP-B12b D-6: a GBD field's SHA-256 depends on the budget through the
public entry, because ``_reconstruct_windowed`` chunks the coherent sum to
stay under it).  The arm is DETECTED from the library.

The fixture set is every AIR-terminated GBD fixture the two packages own --
VERIFY-WP-B12b's eight (one optic, last surface varied: conic, conic + k,
even asphere, flat-base asphere, biconic, freeform, field-frame decentre and
a flat last surface) plus WP-B12b's own six -- and my own singlet, at TWO
planes each (the exit vertex and the fixture's own traced best focus), and
through all THREE local public entry points on one fixture.  Every digest
must be identical PRE to POST: the guards refuse classes nothing here
contains, so nothing here may move by one bit.

Author: Andrew Traverso
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import r2_common as C

_GBDPROJ = Path(__file__).resolve().parent.parent / 'probe_gbd_projection'
if str(_GBDPROJ) not in sys.path:
    sys.path.insert(0, str(_GBDPROJ))
import gbdproj_common as GP  # noqa: E402


def _rows():
    """``(key, prescription, E, dx, lam, [planes])`` for every air fixture."""
    C.register_r2_media()
    out = []
    for key, fx in sorted(C.VB.fixtures().items()):
        if key in ('mirror', 'immersed'):
            continue                      # the two classes the guards refuse
        out.append((f'vb12b:{key}', fx.prescription(), fx.E_in(), fx.dx,
                    fx.lam, [0.0, fx.best_focus()]))
    for key, fx in sorted(GP.FIXTURES.items()):
        out.append((f'b12b:{key}', fx.prescription(), fx.beam(), fx.dx,
                    fx.lam, [0.0, float(fx.best_focus())]))
    out.append(('mine:air_singlet', C.r2_prescription('air'),
                C.r2_input_field(), C.R2_DX, C.R2_LAM, [0.0, 2.0e-3]))
    return out


def main():
    C.assert_tree()
    import lumenairy as la
    from lumenairy.propagators import gbd as G

    arm, tokens = C.detect_arm()
    out = dict(env=C.env_block(), arm=arm, arm_tokens=tokens, fields={})
    print(f'arm = {arm}  {tokens}')

    for key, presc, E, dx, lam, planes in _rows():
        for z in planes:
            F = C.gbd_field(presc, E, dx, lam, z)
            d = C.sha(F)
            out['fields'][f'{key}@{z:.6e}'] = dict(
                digest=d, shape=list(F.shape),
                peak=float(np.abs(F).max()),
                energy=float((np.abs(F) ** 2).sum()))
            print(f'  {key:28s} z={z: .6e}  {d}')

    # the three local entry points on one fixture, so "the dispatcher
    # dispatches" is re-read on both arms rather than assumed
    presc = C.r2_prescription('air')
    E, dx, lam, z = C.r2_input_field(), C.R2_DX, C.R2_LAM, 2.0e-3
    ep = {}
    ep['apply_real_lens_gbd'] = C.sha(C.gbd_field(presc, E, dx, lam, z))
    ep['apply_real_lens_universal_gbd'] = C.sha(np.asarray(
        la.apply_real_lens_universal(
            E, prescription=presc, wavelength=lam, dx=dx,
            output_plane_distance=z, method='gbd',
            method_kwargs={'gbd': dict(C.FRAME)})))
    ep['propagate_gbd_through_prescription'] = C.sha(np.asarray(
        G.propagate_gbd_through_prescription(
            E, dx, presc, wavelength=lam, per_surface=True, z_image=z,
            **C.FRAME)))
    out['entry_points'] = ep
    for k, v in ep.items():
        print(f'  entry {k:38s} {v}')

    C.dump(out, 'probe_r2_identity_' + arm)


if __name__ == '__main__':
    main()
