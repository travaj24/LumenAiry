"""Build validation/pipeline/specs/_measured_nyquist.json from the exp29
run of record, and report what each shipped spec's dx_common / n_common must
become.
"""
import json
import os

_REPO = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import lumenairy as la
from lumenairy.propagators.carrier_field import (
    _BAND_HEADROOM,
    CarrierSpec,
    carrier_difference_nyquist,
)

assert os.path.normcase(la.__file__).startswith(
    os.path.normcase(r'C:\tmp\lum_au')), la.__file__

EXP = (r'd:\Metacept\Neurophos\Python_Test_Scripts\Free_Space_Optics'
       r'\Reverse_Symmetric_ASM\output_tx_design\design121'
       r'\exp29_121_traced_pipeline_32order_v5.35.1_nfc8192_rs1_legfull')
SPECS = r'C:\tmp\lum_au\validation\pipeline\specs'
LAM = 1.31e-6

led = json.load(open(os.path.join(EXP, 'aggregate', 'ledger.json')))
key, pay = led['_key'], led['payload']
R_c, dx_used = pay['R_c'], pay['dx_common']
rows = {r['beam']: r for r in pay['rows']}
dst = CarrierSpec(R=R_c, centre=(0.0, 0.0), tilt=(0.0, 0.0), piston=0.0)

beams = {}
for beam, row in rows.items():
    ch = json.load(open(os.path.join(EXP, f'{beam}.json')))['payload']
    src = CarrierSpec(R=ch['R_out'], centre=tuple(ch['chief_exit']),
                      tilt=tuple(ch['tilt_exit']), piston=0.0)
    sr = row['support_radius']
    dx_binding = row['nyquist_margin'] * dx_used
    rep0 = carrier_difference_nyquist(src, dst, LAM, sr, env_band=0.0)
    bare = (rep0.na_src_max if row['binding_term'] == 'reconstruct'
            else rep0.ramp_max)
    env_band = (LAM / (2.0 * dx_binding) - bare) / _BAND_HEADROOM
    rep = carrier_difference_nyquist(src, dst, LAM, sr, dx_target=dx_used,
                                     env_band=env_band)
    assert rep.binding_term == row['binding_term']
    assert abs(rep.margin / row['nyquist_margin'] - 1.0) < 1e-12, beam
    beams[beam] = {
        'R_out': ch['R_out'],
        'chief_exit': [float(v) for v in ch['chief_exit']],
        'tilt_exit': [float(v) for v in ch['tilt_exit']],
        'support_radius': sr,
        'env_band': env_band,
        'measured_dx_binding': dx_binding,
        'measured_binding_term': row['binding_term'],
    }

witness = {
 'schema': 1,
 'what': (
  "MEASURED inputs to the aggregate stage's band-aware Nyquist guard "
  "(lumenairy.propagators.carrier_field.carrier_difference_nyquist), one "
  "entry per design-121 order chain.  Everything here except env_band is "
  "read straight out of the run of record's own artifacts; env_band is "
  "recovered by inverting the guard at the pitch that run used, which is "
  "exact because it is the ONE free quantity in that arithmetic.  All 32 "
  "beams reproduce their recorded nyquist_margin to better than 1e-12 "
  "relative under lumenairy 5.38.1."),
 'why': (
  "So tests/unit/test_pipeline_spec_guard_validity.py can re-evaluate the "
  "LIVE guard against every shipped spec without the 6.7 GB of chain "
  "fields.  A future tightening of the guard's arithmetic then strands a "
  "spec in CI instead of at hour seven of a run -- which is the defect this "
  "file exists to prevent (audit S9.1 #2: the guard was tightened in "
  "0f46efb and every shipped spec had been written before it)."),
 'provenance': {
  'run': 'exp29_121_traced_pipeline_32order_v5.35.1_nfc8192_rs1_legfull',
  'ledger': 'aggregate/ledger.json',
  'ledger_written': led['_written'],
  'measured_with_lumenairy': key['lumenairy_source_sha256'][:12],
  'measured_at_lumenairy_version': key['lumenairy_version'],
  'reproduced_at_lumenairy_version': la.__version__,
  'reproduced_on': '2026-08-17',
  'dx_common_used_by_that_run': dx_used,
  'n_common_used_by_that_run': pay['n_common'],
  'worst_nyquist_margin_recorded': pay['worst_nyquist_margin'],
  'worst_containment_margin_recorded': pay['worst_containment_margin'],
 },
 'chain_signature': {
  'kind': 'traced', 'plane': 'fine_retrace_exit', 'ray_subsample': 1,
  'n_fine_cap': 8192, 'window_factor': 4.0,
  'note': ("the chain fields that determine a beam's exit carrier, support "
           "radius and envelope band.  A spec whose chain block differs in "
           "any of these does NOT share this witness."),
 },
 'wavelength': LAM,
 'common_carrier': {'R': R_c, 'centre': [0.0, 0.0], 'tilt': [0.0, 0.0],
                    'piston': 0.0},
 'order_key_for': {'(0,0)': 'p0_p0', '(-2,0)': 'm2_p0', '(-4,-2)': 'm4_m2'},
 'beams': dict(sorted(beams.items())),
}
dest = os.path.join(SPECS, '_measured_nyquist.json')
with open(dest, 'w') as fh:
    json.dump(witness, fh, indent=1)
    fh.write('\n')
print('WROTE', dest, f'({len(beams)} beams)')

# ---- what each shipped spec needs ----------------------------------------
SUB = {'d121_3order_ab_rcwa': ['p0_p0', 'm2_p0', 'm4_m2'],
       'd121_3order_ab_scalar': ['p0_p0', 'm2_p0', 'm4_m2'],
       'd121_3order_probe': ['p0_p0', 'm2_p0', 'm4_m2']}
print(f"\n{'spec':<24s} {'dx_bind_um':>11s} {'reach_mm':>9s} "
      f"{'n@1.0um':>8s}")
for name in ('d121_32order', 'd121_3order_ab_rcwa', 'd121_3order_ab_scalar',
             'd121_3order_probe'):
    keys = SUB.get(name, list(beams))
    dxb = min(beams[k]['measured_dx_binding'] for k in keys)
    reach = max(max(abs(beams[k]['chief_exit'][0]),
                    abs(beams[k]['chief_exit'][1]))
                + beams[k]['support_radius'] for k in keys)
    n_min = 2 * reach / 1.0e-6
    print(f'{name:<24s} {dxb*1e6:11.6f} {reach*1e3:9.4f} {n_min:8.0f}')
