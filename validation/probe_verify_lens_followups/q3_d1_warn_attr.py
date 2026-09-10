"""Q3 (D1) -- the ray-density self-check warnings' ATTRIBUTION.

Independent re-measurement of the 6-of-9 -> 0-of-9 claim, on my own fixture
(a decentred beam on the meniscus at 1.55 um) rather than the builder's.
Five ray-density notices are driven over their thresholds by lowering the
module-level tolerances, at band heights 0 / 32 / 7, and ``w.filename`` is
recorded for each.  The FIELD is hashed alongside so the attribution change
can be shown value-free.

Second, INDEPENDENT teeth for the part of the defect that is not the printed
location: the default warning filter's per-location dedup registry is keyed on
the reporting module, so with ``simplefilter('default')`` a second call from a
DIFFERENT caller module must re-warn.  Two caller modules are synthesised
(``exec`` of the same source under two filenames) and the number of distinct
notices counted.

Usage: python q3_d1_warn_attr.py <out.json> --tree <arm tree>
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _vf  # noqa: E402

N, DX, W, SUB = 352, 12e-6, 0.95e-3, 4
THIS = os.path.basename(__file__)

_CALLER_SRC = """
def call(la, E, kw, rows):
    return la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
"""


def _kw():
    return dict(prescription=_vf.presc_meniscus(ap=4.4e-3),
                wavelength=_vf.WL, dx=DX, ray_subsample=SUB, n_workers=1,
                on_undersample='silent', on_noncollimated='off',
                on_aperture_beam='silent', parallel_amp=False,
                amplitude_model='ray_density',
                preserve_input_phase='remap', remap_sampling='full',
                origin=(0.28e-3, -0.20e-3))


def _force(LT):
    """Drive all five ray-density notices over threshold."""
    LT._RD_ENERGY_GAIN_TOL = -1.0
    LT._RD_ENERGY_DEFICIT_BASE = -1.0
    LT._RD_ENERGY_DEFICIT_PER_SUB = 0.0
    LT._RD_HALO_AMAX_TOL = 0.0
    LT._SUPPORT_BAND_PEAK_RATIO_TOL = 0.0
    LT._ORIGIN_AMP_SUPPORT_TOL = -1.0
    LT.ORIGIN_AMP_SUPPORT_CHECK = 'warn'
    LT.RAY_DENSITY_HALO_CHECK = 'warn'
    LT.SUPPORT_BAND_CHECK = 'warn'


def _call(la, E, kw, rows):
    from lumenairy.elements import _lens_imap as IM
    IM.inverse_map_cache_clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        f = la.apply_real_lens_traced(E, sag_chunk_rows=rows, **kw)
    IM.inverse_map_cache_clear()
    return np.asarray(f), [{'msg': str(w.message)[:90],
                            'file': os.path.basename(str(w.filename)),
                            'line': int(w.lineno)} for w in caught]


def _dedup_probe(la, E, kw, rows):
    """Two DIFFERENT caller modules, default (per-location dedup) filter.

    The registry the default filter consults is ``__warningregistry__`` of the
    module the warning is ATTRIBUTED to.  Attributed to the caller, each of
    the two synthetic callers warns once (2 notices); attributed to the
    library, the second caller is silenced by the first caller's entry (1)."""
    from lumenairy.elements import _lens_imap as IM
    import lumenairy.elements._lens_traced as LT
    mods = []
    for i in (1, 2):
        g = {'__name__': f'q3_caller{i}', '__file__': f'q3_caller{i}.py'}
        code = compile(_CALLER_SRC, f'q3_caller{i}.py', 'exec')
        exec(code, g)
        mods.append(g)
    LT.__warningregistry__ = {}
    seen = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.simplefilter('default')
        for g in mods:
            IM.inverse_map_cache_clear()
            g['call'](la, E, kw, rows)
            IM.inverse_map_cache_clear()
            seen.append(len(caught))
    msgs = [str(w.message)[:60] for w in caught]
    return {'cumulative_notices_after_each_caller': seen,
            'n_total': len(caught),
            'n_distinct_msgs': len(set(msgs)),
            'files': sorted({os.path.basename(str(w.filename))
                             for w in caught})}


def main():
    args = _vf.argp(__doc__).parse_args()
    la = _vf.banner(args.tree)
    from lumenairy.elements import _lens_traced as LT
    _force(LT)
    E = _vf.gauss(N, DX, W, x0=0.28e-3, y0=-0.20e-3)
    kw = _kw()
    out, n_lib, n_tot = {}, 0, 0
    for rows in (0, 32, 7):
        f, ws = _call(la, E, kw, rows)
        lib = sum(w['file'] != THIS for w in ws)
        out[f'rows={rows}'] = {'hash': _vf.h(f), 'warnings': ws,
                               'n_from_caller': len(ws) - lib,
                               'n_from_library': lib, 'n_total': len(ws)}
        n_lib += lib
        n_tot += len(ws)
        print(f"rows={rows:<4d} hash={_vf.h(f)}  "
              f"{len(ws) - lib}/{len(ws)} from the CALLER", flush=True)
        for w in ws:
            print(f"    {w['file']:<26s}:{w['line']:<6d} {w['msg'][:64]}"
                  f"{'' if w['file'] == THIS else '   <-- LIBRARY'}",
                  flush=True)
    out['_totals'] = {'attributed_to_library': n_lib, 'total': n_tot}
    print(f"TOTAL attributed to the library: {n_lib} of {n_tot}", flush=True)

    out['_dedup'] = _dedup_probe(la, E, kw, 32)
    print(f"dedup probe (2 caller modules, default filter): "
          f"{out['_dedup']}", flush=True)
    _vf.dump(args, {'cases': out, 'N': N, 'DX': DX, 'SUB': SUB,
                    'free_gb': _vf.free_gb()})


if __name__ == '__main__':
    main()
