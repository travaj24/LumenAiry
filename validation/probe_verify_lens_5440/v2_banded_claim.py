"""V2 -- THE BANDED CLAIM, on fixtures independent of the shipped test file.

Band heights {0 (whole-grid), 7, 32, 128, N} compared PAIRWISE by
``np.array_equal`` on the field AND on every diagnostic the call reports
(``n_out_of_domain``, ``gate_open`` / ``engaged``, and the FULL warning list --
the fold-caustic warning, the three ray-density self-checks and the niche-D9
origin verdict), on BOTH inversion routes.

``--forced`` drives the three ray-density self-checks and the origin verdict
OVER their thresholds (module tolerances set to ~0 for the duration), so the
warning-equality claim is measured on warnings that actually fire rather than
on two empty lists.

Usage:  python v2_banded_claim.py <out.json> [--forced]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _fix                                                    # noqa: E402

N, DX, W, SUB = 384, 13e-6, 1.0e-3, 4
ROWS = [0, 7, 32, 128, N]


def _base(**over):
    kw = dict(prescription=_fix.presc_singlet(ap=4.4e-3), wavelength=_fix.WL,
              dx=DX, ray_subsample=SUB, n_workers=1, on_undersample='silent',
              on_noncollimated='off', on_aperture_beam='silent',
              parallel_amp=False)
    kw.update(over)
    return kw


def _sph(n, dx, w, R, x0=0.0, y0=0.0):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    r2 = (X - x0) ** 2 + (Y - y0) ** 2
    k = 2 * np.pi / _fix.WL
    return (np.exp(-r2 / w ** 2) * np.exp(1j * k * r2 / (2.0 * R))
            ).astype(np.complex128)


def fixtures():
    E = lambda: _fix.gauss(N, DX, W)                           # noqa: E731
    Edec = lambda: _fix.gauss(N, DX, W, x0=0.30e-3, y0=-0.22e-3)  # noqa: E731
    Esph = lambda: _sph(N, DX, W, 0.055)                       # noqa: E731
    return [
        ('g01_rd_preserve', E, _base(amplitude_model='ray_density')),
        ('g02_rd_remap_lattice', E,
         _base(amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='lattice')),
        ('g03_rd_remap_full', E,
         _base(amplitude_model='ray_density',
               preserve_input_phase='remap', remap_sampling='full')),
        ('g04_rd_origin', Edec,
         _base(amplitude_model='ray_density', preserve_input_phase='remap',
               remap_sampling='full', origin=(0.30e-3, -0.22e-3))),
        ('g05_rd_carrier_sph', Esph,
         _base(carrier=0.055, amplitude_model='ray_density')),
        ('g06_rd_tilted', Edec,
         _base(amplitude_model='ray_density', tilt_aware_rays=True,
               preserve_input_phase='remap', remap_sampling='lattice',
               beam_centre=(0.30e-3, -0.22e-3))),
        ('g07_screen', E, _base()),
        ('g08_screen_carrier', Esph, _base(carrier=0.055)),
        # caustic-bearing: the fold warning fires; the evaluator's G2 guard
        # refuses, so this is the coarse-Newton banded route
        ('g09_rd_caustic', lambda: _fix.gauss(N, DX, 1.7e-3),
         _base(prescription=_fix.presc_strong(ap=4.4e-3),
               amplitude_model='ray_density')),
    ]


def _force(LT):
    """Drive every ray-density self-check and the origin verdict over its
    threshold, so the band/whole-grid warning comparison is made on warnings
    that FIRE.  Same monkeypatch on both arms and every band height."""
    LT._RD_ENERGY_GAIN_TOL = -1.0
    LT._RD_ENERGY_DEFICIT_BASE = -1.0
    LT._RD_ENERGY_DEFICIT_PER_SUB = 0.0
    LT._RD_HALO_AMAX_TOL = 0.0
    LT._SUPPORT_BAND_PEAK_RATIO_TOL = 0.0
    LT._ORIGIN_AMP_SUPPORT_TOL = -1.0
    LT.ORIGIN_AMP_SUPPORT_CHECK = 'warn'


def main():
    la = _fix.banner()
    forced = '--forced' in sys.argv
    from lumenairy.elements import _lens_traced as LT
    if forced:
        _force(LT)
        print('# FORCED: self-check tolerances driven to ~0', flush=True)
    medbite = '--medianbite' in sys.argv
    if medbite:
        # The census MEDIAN, put straight into every pixel: a floor at 1.5 x
        # the median exceeds max |det J| on these fixtures, so the capped
        # amplitude is ``|E_in| / sqrt(1.5 * median)`` EVERYWHERE.  Any
        # band-order difference in the median is then a field difference,
        # not merely a warning difference.
        LT._RAY_DENSITY_CAUSTIC_FLOOR_REL = 1.5
        LT._RAY_DENSITY_CAUSTIC_MAXMIN = 1.0000001
        ROWS[:] = [0, 1, 3, 7, 32, 128, N]
        print('# MEDIANBITE: floor 1.5 x median (caps every pixel)', flush=True)
    fold = '--fold' in sys.argv
    if fold:
        # Drive the CAUSTIC CENSUS itself: a floor at 0.999 x the median makes
        # the |det J| cap BITE, so the census median reaches the FIELD (not
        # only a warning), and max/min > 1+1e-7 fires the fold warning on the
        # evaluator route the G2 guard otherwise keeps fold-free (|det J| at the
        # exit FACE is ~1 everywhere, so a natural fold needs a folded map,
        # which the evaluator refuses -- hence the forced thresholds).
        LT._RAY_DENSITY_CAUSTIC_FLOOR_REL = 0.999
        LT._RAY_DENSITY_CAUSTIC_MAXMIN = 1.0000001
        ROWS[:] = [0, 1, 3, 7, 32, 128, N]
        print('# FOLD: caustic floor 0.999 x median, maxmin 1+1e-7, rows', ROWS,
              flush=True)
    out = {'lumenairy_file': la.__file__, 'version': la.__version__,
           'forced': forced, 'fold': fold, 'medianbite': medbite, 'N': N, 'rows': list(ROWS),
           'cases': {}}
    for name, mkE, kw0 in fixtures():
        E = mkE()
        for inv, itag in ((None, 'default'), (False, 'noinv')):
            kw = dict(kw0)
            if inv is not None:
                kw['inverse_map'] = inv
            recs = {}
            for rows in ROWS:
                t = time.time()
                f, rec, msgs = _fix.run(la, E, kw, rows)
                recs[rows] = {
                    'hash': _fix.h(f),
                    'sum_abs2': float(np.sum(np.abs(f) ** 2)),
                    'engaged': bool(rec.get('engaged', False)),
                    'gate_open': bool(rec.get('gate_open', False)),
                    'refused': rec.get('refused'),
                    'n_out_of_domain': rec.get('n_out_of_domain'),
                    'nwarn': len(msgs), 'warnings': msgs,
                    'secs': round(time.time() - t, 2)}
            ref = recs[0]
            eq = {r: (recs[r]['hash'] == ref['hash']) for r in ROWS}
            diag_eq = {r: (recs[r]['n_out_of_domain'] == ref['n_out_of_domain']
                           and recs[r]['engaged'] == ref['engaged']
                           and recs[r]['gate_open'] == ref['gate_open']
                           and recs[r]['warnings'] == ref['warnings'])
                       for r in ROWS}
            key = f'{name}.{itag}'
            out['cases'][key] = {'bands': recs, 'field_equal': eq,
                                 'diag_equal': diag_eq}
            print(f"{key:32s} eng={ref['engaged']} nw={ref['nwarn']} "
                  f"field_eq={[eq[r] for r in ROWS]} "
                  f"diag_eq={[diag_eq[r] for r in ROWS]}", flush=True)
    with open(sys.argv[1], 'w') as fh:
        json.dump(out, fh, indent=1)
    print('wrote', sys.argv[1])


if __name__ == '__main__':
    main()
