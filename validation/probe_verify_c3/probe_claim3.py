"""VERIFY-WP-C3 CLAIM 3 -- the oracle ladder, on MY oracle and MY fixture, and
the FLOOR claim tested by widening the window.

    python probe_claim3.py <tree> <out.json>

Two fixtures are run:

* THEIRS, to check the printed digits: lambda = 1.064 um, w = 0.30 mm,
  N = 512 at six 1/e radii (dx = 3.5156 um), R = -40 mm;
* MINE, to check the QUALITATIVE table on parameters they never used:
  lambda = 0.633 um, w = 0.22 mm, N = 768 at six 1/e radii (dx = 1.71875 um),
  R = -25 mm.

and each is run at THREE window widths on the SAME pitch -- six, nine and
twelve 1/e radii (N, 1.5 N, 2 N).  That is the decisive test of the report's
"the Collins column IS the grid-truncation floor": the floor falls from
6.5e-05 to 7.1e-10 to 9.1e-17 across those three, so a column that is the
floor must fall with it and a column that is the transport's own error must
not.
"""
from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_TREE = os.path.abspath(sys.argv[1])
sys.path.insert(0, _TREE)

import numpy as np                                     # noqa: E402
import vclib as V                                      # noqa: E402

V.anchor(_TREE)

import lumenairy.propagators.carrier as CA             # noqa: E402

FIXTURES = {
    'theirs': dict(lam=1.064e-6, w=0.30e-3, N=512, R=-40.0e-3),
    'mine': dict(lam=0.633e-6, w=0.22e-3, N=768, R=-25.0e-3),
}


def cells(R):
    aR = abs(R)
    return [
        ('diverging_A+1.125', +aR, 0.125 * aR),
        ('converging_A+0.5', R, 0.5 * aR),
        ('focus_A0', R, 1.0 * aR),
        ('just_past_A-0.025', R, 1.025 * aR),
        ('well_past_A-0.5', R, 1.5 * aR),
        ('astigmatic', (R, R * 1.375), 0.5 * aR),
    ]


rows = []
for fname, fx in FIXTURES.items():
    lam, w, N0, Rc = fx['lam'], fx['w'], fx['N'], fx['R']
    dx = 6.0 * w / N0
    for nmul, radii in ((1.0, 6), (1.5, 9), (2.0, 12)):
        N = int(round(N0 * nmul))
        x = V.axis(N, dx)
        env = V.gauss_env(x, x, w)
        floor = V.truncation_floor(w, -(N / 2.0) * dx, (N / 2.0 - 1.0) * dx)
        for tag, R, z in cells(Rc):
            A = (1.0 + z / (R[0] if isinstance(R, tuple) else R))
            for tr in ('sziklas', 'collins'):
                for gk in ('auto', 'fresnel'):
                    with warnings.catch_warnings(record=True) as rec:
                        warnings.simplefilter('always')
                        try:
                            o = CA.propagate_carrier_referenced(
                                env, R, z, lam, dx, transport=tr,
                                gap_kernel=gk, on_collins_sampling='warn')
                            exc = None
                        except Exception as e:          # noqa: BLE001
                            o, exc = None, f'{type(e).__name__}: {e}'
                    row = {'fixture': fname, 'radii': radii, 'N': N,
                           'dx': dx, 'cell': tag, 'transport': tr,
                           'gap_kernel': gk, 'A': A, 'z': z,
                           'R': repr(R), 'raised': exc,
                           'floor': floor['converged'],
                           'n_kelly': sum('under-sampled' in str(wi.message)
                                          for wi in rec)}
                    if o is not None:
                        dxo, dyo = V.pitch2(o.dx)
                        got = V.refield(o.env, o.R, dxo, dyo, lam)
                        xo = V.axis(np.shape(got)[-1], dxo)
                        yo = V.axis(np.shape(got)[-2], dyo)
                        ref = V.gauss_field_at(xo, yo, w, R, lam, z)
                        c = (np.shape(got)[-2] // 2, np.shape(got)[-1] // 2)
                        row.update({
                            'dx_out': dxo, 'dy_out': dyo,
                            'R_out': repr(o.R),
                            'rel_l2': V.rel_l2(got, ref),
                            'rel_l2_abs': V.rel_l2(np.abs(got), np.abs(ref)),
                            'centre_ratio_abs': float(abs(got[c] / ref[c])),
                            'centre_ratio_arg': float(
                                np.angle(got[c] / ref[c])),
                            'rel_l2_over_floor': (
                                V.rel_l2(got, ref) / floor['converged']
                                if floor['converged'] else None),
                        })
                    rows.append(row)

out = {'rows': rows,
       'floors': {f'{fn}_{r}radii': V.truncation_floor(
           FIXTURES[fn]['w'],
           -(int(round(FIXTURES[fn]['N'] * m)) / 2.0)
           * (6.0 * FIXTURES[fn]['w'] / FIXTURES[fn]['N']),
           (int(round(FIXTURES[fn]['N'] * m)) / 2.0 - 1.0)
           * (6.0 * FIXTURES[fn]['w'] / FIXTURES[fn]['N']))
           for fn in FIXTURES for m, r in ((1.0, 6), (1.5, 9), (2.0, 12))}}

# ---- the ratio column, sziklas / collins, per (fixture, radii, cell, kernel)
ratios = []
by = {}
for r in rows:
    by[(r['fixture'], r['radii'], r['cell'], r['gap_kernel'],
        r['transport'])] = r
for (fn, ra, ce, gk, tr), r in list(by.items()):
    if tr != 'sziklas':
        continue
    co = by.get((fn, ra, ce, gk, 'collins'))
    if co is None:
        continue
    ratios.append({
        'fixture': fn, 'radii': ra, 'cell': ce, 'gap_kernel': gk,
        'sziklas': r.get('rel_l2'), 'sziklas_raised': r.get('raised'),
        'collins': co.get('rel_l2'), 'collins_raised': co.get('raised'),
        'ratio': (r['rel_l2'] / co['rel_l2']
                  if r.get('rel_l2') and co.get('rel_l2') else None),
        'collins_over_floor': co.get('rel_l2_over_floor'),
        'collins_worse': bool(r.get('rel_l2') is not None
                              and co.get('rel_l2') is not None
                              and co['rel_l2'] > r['rel_l2'] * 1.0000001),
    })
out['ratios'] = ratios

V.write_json(sys.argv[2], out)

for fn in FIXTURES:
    print(f'=== fixture {fn} ===')
    for ra in (6, 9, 12):
        fl = [r['floor'] for r in rows
              if r['fixture'] == fn and r['radii'] == ra][0]
        print(f'  -- {ra} radii (N={[r["N"] for r in rows if r["fixture"]==fn and r["radii"]==ra][0]}), '
              f'truncation floor {fl:.4e}')
        for rt in ratios:
            if rt['fixture'] != fn or rt['radii'] != ra:
                continue
            s = (f"{rt['sziklas']:.4e}" if rt['sziklas'] is not None
                 else f"RAISED {str(rt['sziklas_raised'])[:34]}")
            c = (f"{rt['collins']:.4e}" if rt['collins'] is not None
                 else f"RAISED {str(rt['collins_raised'])[:34]}")
            rr = (f"{rt['ratio']:.4f}" if rt['ratio'] else '--')
            of = (f"{rt['collins_over_floor']:.3f}"
                  if rt['collins_over_floor'] else '--')
            print(f"     {rt['cell']:20s} {rt['gap_kernel']:8s} "
                  f"szik={s:<44s} coll={c:<44s} ratio={rr:<12s} "
                  f"coll/floor={of}")
