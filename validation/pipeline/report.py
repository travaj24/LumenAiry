# validation/pipeline/report.py -- read a finished workdir and print what it
# measured.  Generic: it knows the artifact layout, not the design.
#
#   python report.py <workdir> [--tag <sum-tag>] [--mode crop|full]
#                              [--energy-bar 4e-5] [--ee-bar 0.1]
#
# WHY A SEPARATE READER.  The stage that produced these numbers may have run
# hours ago, in another process, possibly on another box; a report that can
# only be printed by the run that produced it is a report nobody reads.  Every
# number here comes from the JSON artifacts, so this is also the check that
# those artifacts actually carry what the run claimed.
#
# THE BARS are the design-121 campaign's own (PROBE_SUM_AT_APERTURE S9):
# 4e-05 on energy, 0.1 points on encircled energy, FWHM identical to the
# printed digits.  They are arguments, not constants, because a different
# design has different ones.
#
# cp1252-safe ASCII only.
from __future__ import annotations

import argparse
import glob
import json
import os
import sys


def _read(path):
    with open(path, 'r', encoding='cp1252', errors='strict') as fh:
        return json.load(fh)['payload']


def _f(v):
    """A JSON scalar that may be an encoded non-finite float."""
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return float('nan')
    return float('nan') if v is None else float(v)


def find_readouts(workdir):
    out = []
    for p in sorted(glob.glob(os.path.join(workdir, 'readout', '*',
                                           'metrics.json'))):
        out.append((os.path.basename(os.path.dirname(p)), p))
    return out


def report(workdir, tag=None, mode=None, energy_bar=4e-5, ee_bar=0.1,
           stream=sys.stdout):
    def say(s=''):
        print(s, file=stream)

    man_p = os.path.join(workdir, '_manifest.json')
    man = {}
    if os.path.exists(man_p):
        with open(man_p, 'r', encoding='cp1252') as fh:
            man = json.load(fh)
    spec = _read(os.path.join(workdir, 'spec.json')) \
        if os.path.exists(os.path.join(workdir, 'spec.json')) else {}

    say('=' * 78)
    say(f"PIPELINE REPORT -- {spec.get('name', '?')}")
    say(f"  workdir {workdir}")
    say('=' * 78)

    # ---- stage ledger ---------------------------------------------------
    say('\nSTAGES')
    say('  %-11s %10s %12s  %s' % ('stage', 'wall/s', 'digest', 'detail'))
    tot = 0.0
    for st, ent in [(k, man.get('stages', {}).get(k, {}))
                    for k in ('decompose', 'chains', 'aggregate', 'leg',
                              'readout')]:
        if not ent:
            continue
        w = _f(ent.get('wall'))
        tot += 0.0 if w != w else w
        detail = ', '.join(f"{k}={v}" for k, v in sorted(ent.items())
                           if k not in ('wall', 'digest', 'finished_at'))
        say('  %-11s %10.1f %12s  %s' % (st, w, ent.get('digest', '')[:10],
                                         detail))
    run = man.get('run', {})
    say('  %-11s %10.1f' % ('TOTAL', tot))
    if run:
        say(f"  last process: wall {_f(run.get('wall')):.1f}s   peak working "
            f"set {_f(run.get('peak_wset_gb')):.2f} GB   peak commit "
            f"{_f(run.get('peak_commit_gb')):.2f} GB")

    # ---- per-beam chain ledger ------------------------------------------
    ch = sorted(glob.glob(os.path.join(workdir, 'chains', '*.json')))
    if ch:
        say(f"\nCHAINS -- {len(ch)} beam(s) at the summable plane")
        say('  %-9s %10s %6s %13s %11s %11s %10s %9s'
            % ('beam', 'dx_ap/um', 'N', 'R_out/mm', 'origin_x/mm',
               'origin_y/mm', 'P/P_exit', 'wall/s'))
        walls, peaks = [], []
        for p in ch:
            m = _read(p)
            walls.append(_f(m.get('stage_wall')))
            peaks.append(_f(m.get('peak_wset_gb')))
            say('  %-9s %10.4f %6d %13.6f %11.4f %11.4f %10s %9.1f'
                % (os.path.splitext(os.path.basename(p))[0],
                   _f(m['dx_ap']) * 1e6, int(m['n_ap']),
                   _f(m['R_out']) * 1e3, _f(m['grid_origin'][0]) * 1e3,
                   _f(m['grid_origin'][1]) * 1e3,
                   ('%.9f' % _f(m['power_ratio_field_over_exit'])
                    if 'power_ratio_field_over_exit' in m else '-'),
                   _f(m.get('stage_wall'))))
        fin = [w for w in walls if w == w]
        if fin:
            say('  per-beam wall: mean %.1f s, min %.1f, max %.1f  '
                '(sum %.1f s = %.2f h)'
                % (sum(fin) / len(fin), min(fin), max(fin), sum(fin),
                   sum(fin) / 3600.0))
        fp = [p for p in peaks if p == p]
        if fp:
            say('  peak working set over the chains stage: %.2f GB' % max(fp))

    # ---- aggregate ledgers ----------------------------------------------
    for p in sorted(glob.glob(os.path.join(workdir, 'aggregate', '*',
                                           'ledger.json'))):
        led = _read(p)
        if tag and led['tag'] != tag:
            continue
        say(f"\nAGGREGATE -- {led['tag']}: {len(led['include'])} beam(s) onto "
            f"{led['n_common']}^2 at dx {_f(led['dx_common']) * 1e6:.4f} um "
            f"(window {led['n_common'] * _f(led['dx_common']) * 1e3:.4f} mm)")
        say(f"  common carrier R = {_f(led['R_c']) * 1e3:.6f} mm from beam "
            f"{led['reference_beam']!r}; R_out spread across the fan "
            f"{_f(led['R_out_spread_rel']):.3e} relative")
        say('  %-9s %14s %14s %12s %11s %12s %9s %s'
            % ('beam', 'P_source', 'P_common', 'frac_out', 'support/mm',
               'contain/mm', 'nyquist', 'binds'))
        for r in led['rows']:
            say('  %-9s %14.6e %14.6e %12.3e %11.4f %12.4f %9.3f %s'
                % (r['beam'], _f(r['power_source']), _f(r['power_on_common']),
                   _f(r['frac_out_of_window']), _f(r['support_radius']) * 1e3,
                   _f(r['containment_margin']) * 1e3,
                   _f(r['nyquist_margin']), r['binding_term']))
        say('  totals: P_source %.6e, on the common grid %.6e, out of window '
            '%.3e (%.3e of the total)'
            % (_f(led['power_source_total']), _f(led['power_on_common_total']),
               _f(led['power_out_of_window_total']),
               _f(led['frac_out_of_window_total'])))
        say('  worst Nyquist margin %.3fx; worst containment %+.4f mm; '
            '%d batch(es) of %d; %.1f s'
            % (_f(led['worst_nyquist_margin']),
               _f(led['worst_containment_margin']) * 1e3,
               int(led['n_batches']), int(led['batch_size']),
               _f(led['wall'])))

    # ---- readouts --------------------------------------------------------
    for name, p in find_readouts(workdir):
        ro = _read(p)
        if tag and ro['tag'] != tag:
            continue
        if mode and ro['mode'] != mode:
            continue
        say(f"\nREADOUT -- {ro['tag']} on the {ro['mode']!r} leg: "
            f"{len(ro['frames'])} frame(s), summed {len(ro['included'])}")
        adm = ro.get('admissibility', {})
        if adm:
            valid = ('VALID' if adm.get('piston_columns_valid')
                     else 'PRE-FIX, FLAGGED')
            say(f"  piston fix: {adm.get('piston_fix')!r} -- piston columns "
                f"{valid}")
        eek = sorted((k for k in next(iter(ro['rows'].values()))['spot']
                      if k.startswith('ee')),
                     key=lambda s: float(s[2:].replace('p', '.')))
        say('  %-9s %8s %s %14s %11s %11s %11s %9s'
            % ('frame', 'FWHM/um', ' '.join('%8s' % k.upper() for k in eek),
               'window power', 'P/P_shipped', 'relL2', 'piston/rad',
               'leg/s'))
        worst_e = worst_ee = 0.0
        fwhm_bad = []
        for k in ro['frames']:
            r = ro['rows'][k]
            s = r['spot']
            ref = r.get('reference_spot')
            vr = r.get('vs_reference', {})
            say('  %-9s %8.3f %s %14.7e %11s %11s %11s %9.1f'
                % (k, _f(s['fwhm']) * 1e6,
                   ' '.join('%8.4f' % (_f(s[q]) * 100.0) for q in eek),
                   _f(s['power']),
                   ('%.7f' % _f(r['power_ratio'])) if 'power_ratio' in r
                   else '-',
                   ('%.4e' % _f(vr['rel_l2'])) if vr else '-',
                   ('%+.3e' % _f(vr['piston'])) if vr else '-',
                   _f(r['t_leg'])))
            if ref and k in ro['included']:
                worst_e = max(worst_e, abs(_f(r['power_ratio']) - 1.0))
                for q in eek:
                    worst_ee = max(worst_ee,
                                   abs(_f(s[q]) - _f(ref[q])) * 100.0)
                if '%.3f' % (_f(s['fwhm']) * 1e6) != \
                        '%.3f' % (_f(ref['fwhm']) * 1e6):
                    fwhm_bad.append(k)
        say(f"\n  VS THE SHIPPED PER-ORDER PATH, over the {len(ro['included'])}"
            f" summed frame(s), against the campaign's own bars:")
        say(f"    worst |P/P_shipped - 1| = {worst_e:.3e}   (bar "
            f"{energy_bar:.0e})   {'ok' if worst_e <= energy_bar else 'FAIL'}")
        say(f"    worst |dEE|             = {worst_ee:.4f} points   (bar "
            f"{ee_bar})   {'ok' if worst_ee <= ee_bar else 'FAIL'}")
        say(f"    FWHM identical on every frame: "
            f"{'yes' if not fwhm_bad else 'NO -- ' + ', '.join(fwhm_bad)}")
        say(f"    readout stage {_f(ro['wall']):.1f} s; peak working set "
            f"{_f(ro['peak_wset_gb']):.2f} GB")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('workdir')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--mode', default=None)
    ap.add_argument('--energy-bar', type=float, default=4e-5)
    ap.add_argument('--ee-bar', type=float, default=0.1)
    a = ap.parse_args(argv)
    return report(a.workdir, a.tag, a.mode, a.energy_bar, a.ee_bar)


if __name__ == '__main__':
    raise SystemExit(main())
