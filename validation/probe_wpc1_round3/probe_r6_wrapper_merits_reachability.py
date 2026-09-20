"""WP-C1 ROUND 3 -- R6 decision probe: what an ndarray ``aperture_diameter``
does on the in-library path through ``wrapper_merits.py:492``.

VERIFY_WP-C1_ROUND2.md R6 measures the boolean cast at
``wrapper_merits.py:266`` (a grey mask dilates 12281 -> 12449 px, +1.3680 %;
the power integrated over the CAST mask overshoots the correctly WEIGHTED grey
mask by 8.9051e-03) and asks the round-3 builder to DECIDE, with a
measurement, whether ``MultiWavelengthMerit.evaluate``'s unfloated
``ctx.prescription['aperture_diameter']`` at ``:492`` should weight by the
mask instead of casting it.

The decision turns on one fact the verifier did not measure: whether an
ndarray in ``prescription['aperture_diameter']`` can reach that line at all,
and survive it.  ``MultiWavelengthMerit.evaluate`` calls
``surfaces_from_prescription(ctx.prescription)`` at the TOP of its
per-wavelength loop -- before ``:492`` -- and hands the SAME prescription
object to ``apply_real_lens`` on the statement AFTER it.  Both read
``prescription['aperture_diameter']`` with ``float()`` / ``np.isfinite()``.

This probe drives those three steps in order, with the documented scalar as
the control and with hard / grey mask arrays as the R6 shapes, and records
what each returns or raises.  It also re-measures R6's own cast numbers on
the verifier's fixture so the two rounds are comparable.

Run:  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      PYTHONPATH=<tree> python probe_r6_wrapper_merits_reachability.py <out.json>
"""
import json
import sys

import numpy as np

import lumenairy  # noqa: F401
from lumenairy.elements.elements import apply_aperture

N, DX, D = 256, 4e-6, 0.5e-3
SEP = chr(92)


def _rx(aperture_diameter):
    """A VALID builder-shape prescription (``la.make_singlet``), so the only
    thing separating the control arm from the ndarray arms is the type of
    ``aperture_diameter`` itself."""
    import lumenairy as la
    rx = la.make_singlet(R1=0.032, R2=-0.075, d=4e-4, glass='N-BK7',
                         aperture=D)
    rx['aperture_diameter'] = aperture_diameter
    return rx


def _attempt(fn):
    try:
        return [False, '', repr(fn())[:160]]
    except BaseException as e:                      # noqa: BLE001 -- census
        return [True, '{0}: {1}'.format(type(e).__name__, str(e)[:200]), None]


def main(out_path):
    from lumenairy.optimize import core as _core
    from lumenairy.optimize.wrapper_merits import (
        _clear_wrapper_merit_cache,
        _get_wrapper_merit_cache,
    )

    E = np.ones((N, N), dtype=complex)
    hard = np.real(apply_aperture(E, DX, 'circular', {'diameter': D},
                                  edge='hard'))
    grey = np.real(apply_aperture(E, DX, 'circular', {'diameter': D}))
    n_hard = int(np.count_nonzero(hard.astype(bool)))
    n_grey = int(np.count_nonzero(grey.astype(bool)))

    yy, xx = np.mgrid[0:N, 0:N]
    rr = np.sqrt(((xx - (N - 1) / 2.0) * DX) ** 2
                 + ((yy - (N - 1) / 2.0) * DX) ** 2)
    inten = np.exp(-((rr / (D / 2)) ** 2) * 0.35) ** 2
    p_cast = float(inten[grey.astype(bool)].sum())
    p_weight = float((inten * grey).sum())
    p_hard = float(inten[hard.astype(bool)].sum())

    cdtype = _core.get_default_complex_dtype()
    steps = {}
    for label, ap in (('scalar', float(D)), ('ndarray_hard', hard),
                      ('ndarray_grey', grey)):
        rx = _rx(ap)
        row = {}
        # step 1 -- the call at the TOP of MultiWavelengthMerit.evaluate's
        # per-wavelength loop, BEFORE the cache call at :492.
        row['surfaces_from_prescription'] = _attempt(
            lambda r=rx: len(_core.surfaces_from_prescription(r)))
        # step 2 -- the cache call at :492 itself (the array branch at :266).
        _clear_wrapper_merit_cache()
        row['get_wrapper_merit_cache'] = _attempt(
            lambda a=ap: type(_get_wrapper_merit_cache(
                N, DX, a, cdtype)['mask']).__name__)
        # step 3 -- the statement AFTER :492, same prescription object.
        def _real_lens(r=rx, a=ap):
            m = _get_wrapper_merit_cache(N, DX, a, cdtype)['mask']
            E_in = (m.astype(cdtype) if isinstance(m, np.ndarray)
                    else np.ones((N, N), dtype=cdtype))
            out = _core.apply_real_lens(E_in, prescription=r,
                                        wavelength=633e-9, dx=DX)
            return np.asarray(out[0] if isinstance(out, tuple) else out).shape
        row['apply_real_lens'] = _attempt(_real_lens)
        steps[label] = row
    _clear_wrapper_merit_cache()

    out = {
        'lumenairy_file': lumenairy.__file__,
        'python': sys.version.split()[0],
        'numpy': np.__version__,
        'grid': {'N': N, 'dx': DX, 'D': D},
        'cast': {'n_hard': n_hard, 'n_grey': n_grey,
                 'dilation_px': n_grey - n_hard,
                 'dilation_rel': (n_grey - n_hard) / n_hard,
                 'analytic_disc_px': float(np.pi * (D / 2) ** 2 / DX ** 2)},
        'power': {'cast_over_grey_mask': repr(p_cast),
                  'weighted_grey_mask': repr(p_weight),
                  'cast_over_hard_mask': repr(p_hard),
                  'cast_vs_weight_rel': repr((p_cast - p_weight) / p_weight),
                  'cast_vs_hard_rel': repr((p_cast - p_hard) / p_hard)},
        'in_library_path': steps,
    }
    with open(out_path, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1, sort_keys=True, default=repr)
    print('lumenairy:', lumenairy.__file__)
    print('cast:', out['cast'])
    print('power:', out['power'])
    for label, row in steps.items():
        print(' ', label)
        for step, v in row.items():
            print('    {0:28s} raised={1!s:5s} {2}'.format(
                step, v[0], v[1] if v[0] else '-> ' + str(v[2])))


if __name__ == '__main__':
    main(sys.argv[1])
