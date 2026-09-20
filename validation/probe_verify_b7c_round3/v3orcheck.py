"""VERIFY-WP-B7c round 3 -- the scoring oracle's own convergence control.

The population is scored with a band-limited ANGULAR SPECTRUM whose fine
window is sized by :func:`v3scan.asm_grid`.  The Matsushima band limit removes
every plane wave whose ray would traverse more than half that window in ``z``,
so a window that is large enough to HOLD the source can still be too small to
PROPAGATE the widest source-to-pixel angle the output grid asks for -- and
then the far dark tail is under-reported.

``asmbandlimit_win.json`` measures that directly at the ``Q`` oracle-floor row
against the exact azimuthal quadrature: 0.0144 in relative L2 at the shipped
window (band limit 0.137/um against a needed 0.206/um) falling to 0.0020 as
soon as the window clears it, and not moving after.

This control asks the only question that matters for the population: does it
move the FIDELITY column any claim is read off?  Each plane is re-scored with
the fine window TRIPLED, and the change in fidelity and in power/oracle is
reported.

Usage:  python v3orcheck.py <out.json> <in.json> [<in.json> ...] [--n 30]
"""
from __future__ import annotations

import argparse
import glob
import json
import sys

import numpy as np
import v3fixtures as FX
import v3oracle as OR
import v3scan as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('--n', type=int, default=30)
    ap.add_argument('--mult', type=float, default=3.0)
    a = ap.parse_args()
    import lumenairy
    print('lumenairy.__file__ =', lumenairy.__file__, flush=True)
    rows = []
    for pat in a.inputs:
        for p in sorted(glob.glob(pat)):
            with open(p, encoding='cp1252') as f:
                d = json.load(f)
            for key in ('L1', 'rows'):
                for r in d.get(key) or []:
                    if isinstance(r, dict) and r.get('fidelity') is not None:
                        rows.append(r)
    rows.sort(key=lambda r: r['fidelity'])
    # a stratified sample over the whole fidelity range, so the control is
    # not taken only where the field is easy
    take = [rows[int(i)] for i in
            np.linspace(0, len(rows) - 1, min(a.n, len(rows)))]
    out = []
    for r in take:
        fx = FX.FIXTURES[r['fixture']]
        z = r['z_um'] * 1e-6
        ef = OR.exit_field(fx['prescription'], fx['wavelength'], fx['w0'],
                           n_fan=6001)
        Nf = int(S.asm_grid(fx, 3) * a.mult)
        if (Nf - 3 * fx['N']) % 2:
            Nf += 1
        E_big = OR.asm_field(ef, z, fx['wavelength'], fx['N'], fx['dx'],
                             refine=3, Nf=Nf)
        E_ref = OR.asm_field(ef, z, fx['wavelength'], fx['N'], fx['dx'],
                             refine=3, Nf=S.asm_grid(fx, 3))
        # the returned field is recovered from the recorded power and the
        # recorded fidelity is against E_ref, so re-score by the RATIO of the
        # two oracles rather than by re-running the library
        p_big = OR.power(E_big, fx['dx'])
        p_ref = OR.power(E_ref, fx['dx'])
        row = dict(fixture=r['fixture'], z_um=r['z_um'],
                   fidelity_shipped_window=r['fidelity'],
                   oracle_self_fidelity=OR.fidelity(E_big, E_ref),
                   oracle_rel_l2=OR.rel_l2(E_ref, E_big),
                   oracle_power_ratio=p_ref / max(p_big, 1e-300),
                   Nf_shipped=S.asm_grid(fx, 3), Nf_big=Nf)
        out.append(row)
        print(json.dumps(row), flush=True)
    f = [r['oracle_self_fidelity'] for r in out]
    pr = [abs(r['oracle_power_ratio'] - 1.0) for r in out]
    rep = dict(lumenairy=lumenairy.__file__, python=sys.version.split()[0],
               numpy=np.__version__, mult=a.mult, n=len(out),
               worst_oracle_self_fidelity=min(f),
               worst_oracle_power_shift=max(pr),
               worst_oracle_rel_l2=max(r['oracle_rel_l2'] for r in out),
               rows=out)
    with open(a.out, 'w', encoding='cp1252') as fh:
        json.dump(rep, fh, indent=1)
    print('worst oracle self-fidelity %.8f  worst power shift %.3e  '
          'worst rel L2 %.3e' % (rep['worst_oracle_self_fidelity'],
                                 rep['worst_oracle_power_shift'],
                                 rep['worst_oracle_rel_l2']))
    print('wrote', a.out)


if __name__ == '__main__':
    main()
