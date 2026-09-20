"""VERIFY-WP-C3 -- are the OTHER ``focus_readout`` keys accepted-and-ignored on
the COLLINS readout route?

The route block builds ``_par_kw`` from one of two LITERAL key tuples:

    collins route : ('dx_out','N_out','centre_out','on_replica','replica_fill')
    sziklas route : ('dx_out','N_out','standoff','centre_out','bandlimit',
                     'on_replica','replica_fill','on_focus_containment')

``standoff`` and ``on_focus_containment`` now SELECT the sziklas route, so they
are never dropped.  ``bandlimit`` is in the sziklas set and NOT in the collins
set, and nothing refuses it -- so on the collins route it is silently dropped.
That is the accept-and-ignore shape the vocabulary gates exist to remove, and
the flip is what makes it the DEFAULT.

This probe (1) finds a chain configuration whose readout actually RESOLVES to
the collins route, (2) shows ``bandlimit`` changes the answer on 'sziklas' and
is bit-inert on the collins route, and (3) enumerates every
``_FOCUS_READOUT_KEYS`` member by its disposition on each route.
"""
import json
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
import lumenairy  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402

TREE = os.path.abspath(os.environ.get('VC3_TREE', os.getcwd()))
assert os.path.abspath(lumenairy.__file__).startswith(TREE)

LAM = 1.31e-6


def gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


def run(env, groups, dx, tkw, fd, fr, transport, r_in):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        res = C.propagate_traced_carrier_chain(
            env, groups, LAM, dx, r_in=r_in, ray_subsample=16, n_workers=1,
            traced_kwargs=tkw, final_leg='paraxial', final_distance=fd,
            focus_readout=fr, transport=transport)
    import hashlib
    h = hashlib.sha256(
        np.ascontiguousarray(res.field, dtype=np.complex128).tobytes()
    ).hexdigest()
    return res, h, [str(x.message)[:90] for x in w]


def main():
    from tests.unit.test_audit2609_b4_collins_transport import (
        _singlet, _CHAIN_TKW)
    out = {'lumenairy': lumenairy.__file__, 'python': sys.version.split()[0],
           'numpy': np.__version__}

    # --- 1. find a chain whose readout resolves to the COLLINS route -------
    #     a SMALL exit beam and a LONG final leg -- the regime the docstring
    #     names ("the WP-A6 fixture reads K1 = 0.16").
    n, dx, w, r_in = 256, 60e-6, 1.2e-3, np.inf
    presc = _singlet(300e-3, -300e-3, 3e-3, 'N-BK7', 6e-3, 'p')
    groups = [{'prescription': presc, 'gap_before': 10e-3}]
    env = gauss(n, dx, w)
    fr0 = dict(dx_out=0.5e-6, N_out=64)
    found = None
    for fd in (0.05, 0.1, 0.2, 0.4, 0.8, 1.6):
        try:
            res, h, _ = run(env, groups, dx, _CHAIN_TKW, fd, fr0,
                            'collins', r_in)
        except Exception as exc:                        # noqa: BLE001
            out.setdefault('search_errors', []).append(
                dict(fd=fd, err=f'{type(exc).__name__}: {exc}'[:160]))
            continue
        st = res.stages[-1]
        rec = dict(final_distance=fd, route=st.get('readout_route'),
                   k1=st.get('readout_route_k1'),
                   reason=st.get('readout_route_reason'))
        out.setdefault('route_search', []).append(rec)
        if st.get('readout_route') == 'collins' and found is None:
            found = fd
    out['collins_route_final_distance'] = found

    # --- 2. bandlimit: does it move the answer on each route? -------------
    rows = []
    for fd, label in ([(found, 'collins-route')] if found else []) + \
                     [(8e-3, 'sziklas-route')]:
        for transport in ('sziklas', 'collins'):
            try:
                _r0, h0, _ = run(env, groups, dx, _CHAIN_TKW, fd,
                                 dict(fr0), transport, r_in)
                r1, h1, _ = run(env, groups, dx, _CHAIN_TKW, fd,
                                dict(fr0, bandlimit=0.25), transport, r_in)
                rows.append(dict(
                    label=label, final_distance=fd, transport=transport,
                    route=r1.stages[-1].get('readout_route'),
                    sha_without_bandlimit=h0, sha_with_bandlimit=h1,
                    bandlimit_changes_the_answer=(h0 != h1)))
            except Exception as exc:                    # noqa: BLE001
                rows.append(dict(label=label, final_distance=fd,
                                 transport=transport,
                                 err=f'{type(exc).__name__}: {exc}'[:200]))
    out['bandlimit'] = rows

    # --- 3. every focus_readout key, by disposition on each route ---------
    collins_kw = ('dx_out', 'N_out', 'centre_out', 'on_replica',
                  'replica_fill')
    sziklas_kw = ('dx_out', 'N_out', 'standoff', 'centre_out', 'bandlimit',
                  'on_replica', 'replica_fill', 'on_focus_containment')
    disp = []
    for k in sorted(C._FOCUS_READOUT_KEYS):
        sel = k in C._FOCUS_READOUT_STOP_PLANE_KEYS
        disp.append(dict(key=k,
                         in_collins_par_kw=(k in collins_kw),
                         in_sziklas_par_kw=(k in sziklas_kw),
                         selects_sziklas_route=sel,
                         dropped_on_collins_route=(k not in collins_kw
                                                   and not sel)))
    out['key_disposition'] = disp
    out['dropped_on_collins_route'] = sorted(
        d['key'] for d in disp if d['dropped_on_collins_route'])

    tag = os.environ.get('VC3_TAG', 'x')
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     f'dropped_keys_{tag}.json')
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))
    print('WROTE', p)


if __name__ == '__main__':
    main()
