# G8 PROBE RE-ARCHITECTURE, on design 121's own last post-DOE group.
#
# LOCAL-ONLY.  No library edit here.  Three arms per order, all with
# ``TRACED_INVERSE_MAP`` forced on, at the retrace configuration
# ``_fine_trace_group_exit`` drives:
#
#   ship    the shipped fit domain and the shipped exit degree -- must ENGAGE,
#           and its G8 numbers are the acceptance measured where exit pixels
#           actually fall;
#   pre     BUILD_INVERSE_MAP S6.5b's PRE-RESTRICTION model, reproduced at the
#           BUILDER: the element keeps its shipped fit domain (so the
#           incumbent, the census region and the probe points are the shipped
#           ones) and only the MODEL is refitted with ``weights=None`` -- the
#           whole launch square, unweighted.  ``census_amp`` is the element's
#           own weights, which is what keeps the scoring region from widening
#           with the fit (the second S6.5b finding: a census inherited from the
#           fit weights can be widened by widening the fit, and a
#           total-degree-FOUR model then passes).  S6.5b measured 4.5258e-01
#           waves against a restricted 1.9965e-05; the re-architected guard
#           must still REFUSE it.
#           NOTE it is expressible only on the DECENTRED branch, where the
#           restriction is 1e-8 WEIGHTS: on a concentric NaN-masked branch the
#           pre-restriction landings never reach the builder at all.
#   deg8    the exit-degree underfit -- must still be REFUSED.
#
# Run:  python g8_probe_121.py                  (orders 0,0 and -4,-2)
#       G8_NFC=4096 G8_ORDERS="0,0" python g8_probe_121.py
import json
import os
import sys

import numpy as np

_HERE_DIR = os.path.dirname(os.path.abspath(__file__))
# THE REPOSITORY ROOT GOES FIRST, AND ``lumenairy`` IS IMPORTED BEFORE ANY
# ``_d121_common`` CONSUMER.  Neither half is decoration.
#
# A driver under ``validation/`` gets its OWN directory as ``sys.path[0]`` and
# this repo is installed, so a bare ``import lumenairy`` scores the INSTALLED
# tree.  Inserting the root is not enough on its own: ``_d121_common`` puts
# ``$D121_ROOT/Lumenairy`` at ``sys.path[0]`` (the dev-box checkout, by design
# -- it is how the runners find the design's own assets), so ANY module that
# reaches it first -- ``imap_cost_121`` does -- wins the resolution.  Importing
# ``lumenairy`` here puts the WORKING tree in ``sys.modules`` before that can
# happen, which is the same order ``imap_banner_arm.py`` already relies on.
#
# It cost a full round of design-121 arms during this fix.  The tell was a
# ``None``: ``n_probe_requested`` came back empty in the JSON, because that key
# does not exist in the installed tree -- while every parity number still
# looked plausible and both fail-befores still refused.
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE_DIR)))
sys.path.insert(1, _HERE_DIR)
import lumenairy  # noqa: E402,F401  (FIRST -- see above)
import lumenairy.elements as EL  # noqa: E402
import lumenairy.elements._lens_imap as IM  # noqa: E402

import imap_cost_121 as IC  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, '_g8_probe.json')

NFC = int(os.environ.get('G8_NFC', '4096'))


def _orders():
    raw = os.environ.get('G8_ORDERS', '0,0;-4,-2')
    out = []
    for chunk in raw.split(';'):
        a, b = chunk.split(',')
        out.append((int(a), int(b)))
    return out


_KEYS = ('refused', 'detail', 'n_alive', 'n_fit_samples', 'n_detj_census',
         'det_j_range', 'n_probe_requested', 'n_probe_traced', 'n_parity',
         'parity_map_opl_waves', 'parity_incumbent_opl_waves',
         'parity_ratio_opl', 'parity_map_opl_rms_waves',
         'parity_incumbent_opl_rms_waves', 'parity_ratio_opl_rms',
         'parity_map_pos_m', 'parity_incumbent_pos_m', 'parity_ratio_pos',
         'fit_resid_opl_waves', 'build_seconds')


def _unrestrict(rec):
    """Wrap the builder so the model is refitted UNWEIGHTED while everything
    else -- the incumbent, the census region, the probe points -- stays the
    element's shipped configuration.  Fills ``rec`` with THAT build's guards
    and returns the shipped object to the element unchanged."""
    orig = IM.build_inverse_map

    def wrapper(xs_in, XO, YO, OP, *a, **kw):
        w = kw.get('weights')
        alt = dict(kw)
        alt['weights'] = None
        alt['census_amp'] = w        # the shipped scoring region, verbatim
        alt['cache'] = False
        alt['guard_record'] = rec
        if w is None:
            rec['detail'] = ('this branch restricts by NaN MASK, so the '
                             'pre-restriction landings never reach the '
                             'builder -- S6.5b is not expressible here')
            rec['expressible'] = False
        else:
            rec['expressible'] = True
            orig(xs_in, XO, YO, OP, *a, **alt)
        return orig(xs_in, XO, YO, OP, *a, **kw)

    return wrapper, orig


def arm(tag, order, degree=None, unweighted=False, **kwover):
    E, kw, meta = IC._retrace_call_args(n_fine=NFC, order=order)
    kw.update(kwover)
    rec = {}
    ship_rec = {}
    old = IM._IMAP_EXIT_DEGREE
    old_build = IM.build_inverse_map
    IM.TRACED_INVERSE_MAP = True
    IM.inverse_map_cache_clear()
    try:
        if degree is not None:
            IM._IMAP_EXIT_DEGREE = int(degree)
        if unweighted:
            IM.build_inverse_map, _o = _unrestrict(rec)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            EL.apply_real_lens_traced(E, _imap_out=ship_rec, **kw)
    finally:
        IM.build_inverse_map = old_build
        IM._IMAP_EXIT_DEGREE = old
        IM.inverse_map_cache_clear()
    del E
    if not unweighted:
        rec = ship_rec
    rec['engaged'] = (rec.get('refused') is None
                      and rec.get('expressible', True))
    row = {'arm': tag, 'order': list(order), 'n_fine': meta['n_fine'],
           'engaged': bool(rec.get('engaged'))}
    row.update({k: rec.get(k) for k in _KEYS})
    print('  %-6s %-8s %-24s a_opl %.4e  b_opl %.4e (%.4gx)   '
          'a_pos %.4e  b_pos %.4e (%.4gx)'
          % (tag, str(tuple(order)),
             'ENGAGE' if row['engaged'] else 'REFUSE ' + str(row['refused']),
             row.get('parity_map_opl_waves') or np.nan,
             row.get('parity_incumbent_opl_waves') or np.nan,
             row.get('parity_ratio_opl') or np.nan,
             row.get('parity_map_pos_m') or np.nan,
             row.get('parity_incumbent_pos_m') or np.nan,
             row.get('parity_ratio_pos') or np.nan), flush=True)
    if row['refused'] or not row['engaged']:
        print('         %s' % row.get('detail'), flush=True)
    return row


if __name__ == '__main__':
    rows = []
    for order in _orders():
        rows.append(arm('ship', order))
        rows.append(arm('pre', order, unweighted=True))
        rows.append(arm('deg8', order, degree=8))
        with open(OUT, 'w', encoding='ascii') as fh:
            json.dump(rows, fh, indent=1, default=str)
    print('wrote %s' % OUT)
