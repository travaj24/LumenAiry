"""Compare two :mod:`v4_entry` digest maps key by key and PRINT the
entry-point way-back table the report carries.

    python v4_entry_compare.py BASE.json BRANCH.json [OUT.json]
"""
from __future__ import annotations

import json
import sys

# entry point -> (does its OWN signature carry a route keyword?, what it is)
WAY_BACK = {
    'asm_mft': 'method=',
    'fresnel_mft': 'method=',
    'fraunhofer_mft': 'method=',
    'asm_propagate': 'method= (through **method_kwargs)',
    'compute_psf': None,
    'resample_field': None,
    'propagate_asm': None,
    'propagate_fresnel': None,
    'carrier_focus_readout': None,
    'carrier_exact_focus_readout': None,
    'propagate_through_system': None,
    're_reference': None,
}


def main(base_p, branch_p, out_p=None):
    base = json.load(open(base_p, encoding='cp1252'))
    branch = json.load(open(branch_p, encoding='cp1252'))
    bk, rk = base['keys'], branch['keys']
    assert set(bk) == set(rk), (set(bk) ^ set(rk))
    rows = {}
    for k in sorted(bk):
        parts = k.split('/')
        ep = parts[0]
        shp = parts[1] if len(parts) > 1 else '-'
        variant = parts[2] if len(parts) > 2 else '-'
        same = bk[k] == rk[k]
        rows.setdefault(ep, {})[(shp, variant)] = same
    out = {'base': base_p, 'branch': branch_p, 'build': branch['build'],
           'rows': [], 'defects': []}
    print(f"{'entry point':32s} {'captured moves':15s} {'refused stays':14s} "
          f"{'one-keyword way back':22s}")
    for ep in sorted(rows):
        cells = rows[ep]
        cap_nokw = cells.get(('captured', 'nokw'))
        ref_nokw = cells.get(('refused', 'nokw'))
        moves = (cap_nokw is False)
        stays = (ref_nokw is True) if ref_nokw is not None else None
        wb = WAY_BACK.get(ep, '?')
        # a way back is only real if the KEYWORD call is byte-identical to
        # the base's no-keyword call -- which is what these keys measure
        wb_ok = None
        if wb:
            got = [v for (s, var), v in cells.items()
                   if s == 'captured' and var.startswith('kw_')]
            wb_ok = bool(got) and all(got)
        rec = {'entry_point': ep, 'captured_moves': moves,
               'refused_stays': stays, 'way_back': wb,
               'way_back_reproduces_base_bytes': wb_ok}
        out['rows'].append(rec)
        print(f"{ep:32s} {str(moves):15s} {str(stays):14s} "
              f"{(wb or 'NONE'):22s} {'' if wb_ok is None else ('ok' if wb_ok else 'BROKEN')}")
        if moves and not wb:
            out['defects'].append(ep)
        if moves and wb and wb_ok is False:
            out['defects'].append(ep + ' (keyword does not restore bytes)')
    print()
    print("MOVED WITHOUT A ONE-KEYWORD WAY BACK:", out['defects'])
    if out_p:
        with open(out_p, 'w', encoding='cp1252', errors='replace') as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
        print('[wrote]', out_p)


if __name__ == '__main__':
    main(*sys.argv[1:])
