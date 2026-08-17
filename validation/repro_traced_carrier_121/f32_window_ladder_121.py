"""Design-121 float32-sag validator LADDER (D121 audit S11 item 1, lib side).

The audit's S7.3 table ran ``lens_sag_float32_opd_error`` per group at
``field_check_n=512, field_check_dx=0.90 um`` and called it "the production
sampling".  It is the production PITCH but not the production WINDOW: 512 x
0.90 um is a 0.46 mm window on lens groups whose clear apertures are several
mm, so the field-level A/B never sees the pupil edge -- which is exactly where
the sag (and therefore the float32 sag error) is largest.

This walks field_check_n up at the SAME production pitch until the window
covers the clear aperture, and reports the whole ladder so the N-dependence is
visible rather than assumed.
"""
import json
import os
import sys
import warnings

# IMPORT ORDER IS LOAD-BEARING.  _d121_common does its own
# sys.path.insert(0, <D121_ROOT>/Lumenairy), which would shadow this
# worktree's checkout with the main repo's.  Binding lumenairy FIRST puts it
# in sys.modules, so that insert can no longer change which library is under
# test -- and the assert below proves it did not.
_REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import lumenairy as la  # noqa: E402

print('lumenairy.__file__ =', la.__file__, flush=True)
print('lumenairy.__version__ =', la.__version__, flush=True)
assert os.path.normcase(la.__file__).startswith(
    os.path.normcase(r'C:\tmp\lum_au')), la.__file__

sys.path.insert(0, os.path.join(
    r'C:\tmp\lum_au', 'validation', 'repro_traced_carrier_121'))
import _d121_common as D  # noqa: E402

assert os.path.normcase(la.__file__).startswith(
    os.path.normcase(r'C:\tmp\lum_au')), la.__file__

LAM = D.LAM
# The production pitch of the N=32768 analytic run of record: exp31 ran
# N=16384 over a 29.583945393700297 mm extent, so N=32768 is half that pitch.
DX_PROD = 29.583945393700297e-3 / 32768.0

pre, post, gap, period = D.geometry()
groups = [g['prescription'] for g in (pre + post)]
print(f'{len(groups)} lens groups; production dx = {DX_PROD*1e6:.4f} um',
      flush=True)

LADDER = [int(t) for t in (sys.argv[1] if len(sys.argv) > 1
                           else '512,1024,2048,4096').split(',')]

out = {'lumenairy': la.__version__, 'dx_prod_m': DX_PROD, 'ladder': LADDER,
       'groups': []}
for gi, presc in enumerate(groups):
    ap = presc.get('aperture_diameter')
    name = presc.get('name') or f'group{gi}'
    row = {'index': gi, 'name': name, 'aperture_m': ap,
           'n_surfaces': len(presc['surfaces']), 'rungs': {}}
    print(f"\n[{gi}] {name}  aperture {ap*1e3:.4f} mm  "
          f"({len(presc['surfaces'])} surfaces)", flush=True)
    print(f"    {'N':>6s} {'window_mm':>10s} {'cover':>7s} "
          f"{'maxOPD_waves':>14s} {'field_rel':>12s}  ok", flush=True)
    for n in LADDER:
        win = n * DX_PROD
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            r = la.lens_sag_float32_opd_error(
                presc, LAM, field_check_n=n, field_check_dx=DX_PROD)
        r['window_m'] = win
        r['aperture_cover'] = win / ap if ap else None
        row['rungs'][str(n)] = r
        print(f"    {n:6d} {win*1e3:10.4f} {win/ap:7.3f} "
              f"{r['max_opd_error_waves']:14.4e} "
              f"{r['max_field_rel_error']:12.4e}  {r['ok']}", flush=True)
    out['groups'].append(row)

dest = sys.argv[2] if len(sys.argv) > 2 else 'probe_f32_ladder.json'
with open(dest, 'w') as fh:
    json.dump(out, fh, indent=1)
print('\nWROTE', dest, flush=True)
