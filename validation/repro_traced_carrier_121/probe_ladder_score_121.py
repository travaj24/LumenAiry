# SCORER for the artifacts ``probe_ladder_run_121.py`` writes.
# Prints the tables of docs/audits/PROBE_CHAIN_LADDER_PISTON_2026_08_11.md.
#
# Two runs are compared ONLY on co-located samples, never on interpolated ones:
#
#   * PITCH arm (window fixed, dx0 halved).  The co-moving magnification is a
#     property of the R-chain, not of the grid, so every rung lands on the SAME
#     aperture window with dx_ap halved.  Fine index ``i = nF/2 + s*(ic - nC/2)``
#     with ``s = dx_C/dx_F`` hits the coarse rung's own sample points EXACTLY.
#   * WINDOW arm (dx0 fixed, N doubled).  Same dx_ap, so the coarse rung is the
#     CENTRAL BLOCK of the fine one -- again exact, ``s = 1``.
#
# So "envelope rel L2 on the common support" is a difference of the SAME
# physical samples; no resampling error is folded into the score.
#
# METRIC -> BAR MAPPING (the campaign bars the task names):
#   energy 4e-5     -> |dP/P| between rungs (aperture power)
#   EE-class 1e-3   -> 1 - Strehl_equiv, where Strehl_equiv = exp(-sigma^2) and
#                      sigma is the intensity-weighted, PISTON-REMOVED rms phase
#                      difference between rungs (rad).  An aperture-plane
#                      wavefront difference of sigma costs a focal-plane
#                      encircled-energy/Strehl fraction of ~sigma^2 (Marechal),
#                      so this is the aperture-plane statement of the same bar,
#                      and it does not need the 15-min final leg to evaluate.
#                      Reported alongside R90/R99 (encircled-energy radii of the
#                      aperture irradiance), which are the direct EE analogue.
#   piston lambda/100 -> 2*pi/100 = 6.2832e-02 rad
import glob
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get('PL_OUT', os.path.join(_HERE, '_probe_ladder'))
TWO_PI = 2.0 * np.pi
BAR_ENERGY = 4e-5
BAR_EE = 1e-3
BAR_PISTON = TWO_PI / 100.0


def load(tag):
    with open(os.path.join(OUT, '%s.json' % tag), encoding='cp1252') as fh:
        d = json.load(fh)
    z = np.load(os.path.join(OUT, '_%s_env.npz' % tag))
    d['env'] = z['env']
    return d


def colocate(a, b):
    """``(envA_sub, envB_sub, n)`` on the co-located sample set."""
    eA, dA = a['env'], a['chain_b']['dx_ap']
    eB, dB = b['env'], b['chain_b']['dx_ap']
    nA, nB = eA.shape[0], eB.shape[0]
    if dA < dB:                      # make A the COARSER one
        eA, dA, eB, dB, nA, nB = eB, dB, eA, dA, nB, nA
        flip = True
    else:
        flip = False
    s = dA / dB
    si = int(round(s))
    if abs(si - s) > 1e-9:
        raise ValueError('pitch ratio %.12g is not an integer' % s)
    idx = nB // 2 + si * (np.arange(nA) - nA // 2)
    ok = (idx >= 0) & (idx < nB)
    ia = np.where(ok)[0]
    ib = idx[ok]
    sA = eA[np.ix_(ia, ia)]
    sB = eB[np.ix_(ib, ib)]
    return (sB, sA, len(ia)) if flip else (sA, sB, len(ia))


def score(a, b, core_frac=1e-3):
    """Rung-to-rung score.  ``a`` is the reference (the coarser rung)."""
    A, B, n = colocate(a, b)
    aa, ab = np.abs(A), np.abs(B)
    nrm = float(np.sqrt((aa * aa).sum()))
    amp_l2 = float(np.sqrt(((ab - aa) ** 2).sum())) / nrm
    raw_l2 = float(np.sqrt((np.abs(B - A) ** 2).sum())) / nrm
    ov = complex(np.vdot(A, B))                     # sum conj(A) * B
    dphi = float(np.angle(ov))
    al_l2 = float(np.sqrt(
        (np.abs(B * np.exp(-1j * dphi) - A) ** 2).sum())) / nrm
    m = aa >= core_frac * aa.max()
    w = (aa[m] ** 2)
    dp = np.angle(B[m] * np.conj(A[m]))             # already wrapped
    mean = float(np.angle(np.sum(w * np.exp(1j * dp))))
    dpz = np.angle(np.exp(1j * (dp - mean)))
    sig = float(np.sqrt(np.sum(w * dpz ** 2) / w.sum()))
    sig_raw = float(np.sqrt(np.sum(w * dp ** 2) / w.sum()))
    return {'n': n, 'amp_relL2': amp_l2, 'cplx_relL2_raw': raw_l2,
            'cplx_relL2_pistonfree': al_l2, 'overlap_phase_rad': dphi,
            'overlap_abs': abs(ov) / (float(np.linalg.norm(A))
                                      * float(np.linalg.norm(B))),
            'phase_rms_rad': sig, 'phase_rms_raw_rad': sig_raw,
            'phase_rms_waves': sig / TWO_PI, 'strehl_equiv': float(
                np.exp(-sig ** 2)), 'core_px': int(m.sum())}


def ee_radii(d, fracs=(0.90, 0.99, 0.999)):
    """Encircled-energy radii of the APERTURE irradiance about the chief ray
    (= the grid centre), by exact radial accumulation."""
    e = d['env']
    dx = d['chain_b']['dx_ap']
    n = e.shape[0]
    x = (np.arange(n) - n / 2) * dx
    r = np.hypot(x[None, :], x[:, None]).ravel()
    I = (np.abs(e) ** 2).ravel()
    o = np.argsort(r)
    c = np.cumsum(I[o])
    c /= c[-1]
    return [float(r[o][int(np.searchsorted(c, f))]) for f in fracs]


def halo_outside(d, half_m):
    """Power fraction OUTSIDE the centred square of half-width ``half_m``."""
    e = d['env']
    dx = d['chain_b']['dx_ap']
    n = e.shape[0]
    x = np.abs((np.arange(n) - n / 2) * dx)
    m = (x[None, :] <= half_m) & (x[:, None] <= half_m)
    I = np.abs(e) ** 2
    return float(1.0 - I[m].sum() / I.sum())


def wn(d, leg, key):
    return d[leg]['warn_counts']['unique'].get(key, 0)


def _fmt(v, f='%.3e'):
    return 'n/a' if v is None else (f % v)


if __name__ == '__main__':
    tags = sorted(os.path.basename(p)[:-5]
                  for p in glob.glob(os.path.join(OUT, '*.json')))
    runs = {}
    for t in tags:
        try:
            runs[t] = load(t)
        except OSError as exc:
            print('SKIP %s (%s)' % (t, exc))
    print('loaded %d runs: %s\n' % (len(runs), ', '.join(sorted(runs))))

    def key(N, dx0_um, order, suffix='none'):
        return 'n%d_dx%s_o%d_%d_%s' % (
            N, ('%g' % dx0_um).replace('.', 'p'), order[0], order[1], suffix)

    ORDERS = [(0, 0), (-4, -2)]
    PITCH = [(1024, 2.0), (2048, 1.0), (4096, 0.5), (8192, 0.25)]
    WINDOW = [(1024, 2.0), (2048, 2.0), (4096, 2.0), (8192, 2.0)]

    # ---------------------------------------------------------------- 1. census
    print('=' * 108)
    print('TABLE 1 -- PER-RUNG CENSUS (aperture = last post-DOE group exit '
          'vertex; warnings UN-SUPPRESSED, unique count)')
    print('=' * 108)
    # chain A is ORDER-INDEPENDENT and CACHED, so its warnings are observable
    # only on the run that built it cold.  Index them by (N, dx0) and reuse.
    cold = {}
    for t, d in runs.items():
        if d['chain_a']['cold']:
            cold.setdefault((d['N'], round(d['dx0'] * 1e6, 6)), d['chain_a'])
    print('%-24s %5s %7s %8s %8s %9s %9s %11s | %s | %s' % (
        'run', 'N', 'dx0um', 'dx_ap_um', 'win_mm', 'P_doe/Pin', 'P_ap/Pin',
        'piston_c', 'chainA: grp/nwt%/usmp', 'chainB: grp/nwt%/usmp'))
    for order in ORDERS:
        for N, dx0 in sorted(set(PITCH + WINDOW)):
            t = key(N, dx0, order)
            if t not in runs:
                continue
            d = runs[t]
            nb, na = d['chain_b'], d['chain_a']
            ca = cold.get((N, round(dx0, 6)), na)

            def _leg(x):
                us = [r['ratio'] for r in x['undersample'] if 'ratio' in r]
                nw = x['newton']
                return '%d/%5.1f/%5.2fx' % (
                    len(nw), max([r['pct'] for r in nw] or [0.0]),
                    max(us or [0.0]))
            print('%-24s %5d %7.3f %8.4f %8.4f %9.6f %9.6f %11.6f | %s | %s'
                  % (t, d['N'], d['dx0'] * 1e6, nb['dx_ap'] * 1e6,
                     nb['n_ap'] * nb['dx_ap'] * 1e3,
                     na['P_doe'] / na['P_in'],
                     d['aperture']['power'] / na['P_in'],
                     d['aperture']['piston_c'],
                     ('%-20s' % _leg(ca)) + ('' if ca is na else ' (cold run)'),
                     _leg(nb)))
    print()
    print('  grp = groups that warned Newton non-convergence (unique '
          'messages); nwt% = worst per-group non-converged pixel fraction; '
          'usmp = worst exit-wavefront')
    print('  under-sampling ratio dx_have/dx_needed (1.00x = Nyquist on the '
          'exit sphere).  chain A is cached and order-independent, so its '
          'column is read from the COLD build of that rung.')
    print()

    # ------------------------------------------------------------- 2. ladders
    for armname, arm in (('PITCH  (window fixed 2.048 mm, dx0 halved)', PITCH),
                         ('WINDOW (dx0 fixed 2.0 um, N doubled)', WINDOW)):
        print('=' * 108)
        print('TABLE 2%s -- %s' % ('a' if arm is PITCH else 'b', armname))
        print('=' * 108)
        for order in ORDERS:
            have = [(N, dx) for N, dx in arm if key(N, dx, order) in runs]
            if len(have) < 2:
                continue
            print('  order (%+d,%+d)' % order)
            print('   %-14s %11s %11s %11s %11s %11s %11s %11s' % (
                'rung pair', 'amp relL2', 'cplx relL2', 'piston-free',
                'd piston', 'phase rms', 'Strehl-1', '|overlap|'))
            for i in range(len(have) - 1):
                a = runs[key(have[i][0], have[i][1], order)]
                b = runs[key(have[i + 1][0], have[i + 1][1], order)]
                s = score(a, b)
                print('   %-14s %11.3e %11.3e %11.3e %11.3e %11.3e %11.3e '
                      '%11.8f' % (
                          '%d->%d' % (have[i][0], have[i + 1][0]),
                          s['amp_relL2'], s['cplx_relL2_raw'],
                          s['cplx_relL2_pistonfree'], s['overlap_phase_rad'],
                          s['phase_rms_rad'], 1.0 - s['strehl_equiv'],
                          s['overlap_abs']))
            # energy / EE-radius / halo columns
            ref = runs[key(1024, 2.0, order)] if key(1024, 2.0, order) in runs \
                else runs[key(*have[0], order)]
            half = 0.5 * ref['chain_b']['n_ap'] * ref['chain_b']['dx_ap']
            print('   %-14s %13s %13s %13s %13s %13s' % (
                'rung', 'P/P_in', 'dP/P vs prev', 'R90(mm)', 'R99(mm)',
                'halo>1024win'))
            prev = None
            for N, dx in have:
                d = runs[key(N, dx, order)]
                P = d['aperture']['power'] / d['chain_a']['P_in']
                r = ee_radii(d)
                print('   %-14s %13.9f %13.3e %13.6f %13.6f %13.3e' % (
                    '%d' % N, P, (abs(P / prev - 1.0) if prev else 0.0),
                    r[0] * 1e3, r[1] * 1e3, halo_outside(d, half)))
                prev = P
            print()

    # ------------------------------------------------------------- 3. pistons
    print('=' * 108)
    print('TABLE 3 -- PISTON (rad).  piston_c = phase at the chief ray; '
          'piston_w = intensity-weighted circular mean.')
    print('   bar: lambda/100 = %.6e rad' % BAR_PISTON)
    print('=' * 108)
    print('%-34s %14s %14s %10s %s' % ('run', 'piston_c', 'piston_w',
                                       'P/P_in', 'field sha256[:16]'))
    for t in sorted(runs):
        d = runs[t]
        print('%-34s %14.9f %14.9f %10.6f %s' % (
            t, d['aperture']['piston_c'], d['aperture']['piston_w'],
            d['aperture']['power'] / d['chain_a']['P_in'],
            d['field_sha256'][:16]))
    print()

    # (a) fresh-process repeats -------------------------------------------
    print('-- 3a  SAME order, SAME rung, FRESH PROCESS ---------------------')
    print('%-34s %-34s %12s %12s %14s %s' % (
        'run', 'repeat', 'd piston_c', 'd piston_w', 'max|dE|', 'sha equal'))
    for t in sorted(runs):
        for r in sorted(runs):
            if r == t or not r.startswith(t + '_rep'):
                continue
            a, b = runs[t], runs[r]
            de = float(np.abs(a['env'] - b['env']).max()) \
                if a['env'].shape == b['env'].shape else float('nan')
            print('%-34s %-34s %12.3e %12.3e %14.6e %s' % (
                t, r,
                a['aperture']['piston_c'] - b['aperture']['piston_c'],
                a['aperture']['piston_w'] - b['aperture']['piston_w'],
                de, a['field_sha256'] == b['field_sha256']))
    print()

    # (c) inter-order piston difference ------------------------------------
    print('-- 3c  TWO ORDERS, piston DIFFERENCE, per (independent) process --')
    print('%-14s %16s %16s %16s %16s' % (
        'suffix', 'piston_c (0,0)', 'piston_c (-4,-2)', 'difference',
        'diff drift'))
    base = None
    for suf in ('none', 'none_rep1', 'none_rep2'):
        ta = key(1024, 2.0, (0, 0), suf)
        tb = key(1024, 2.0, (-4, -2), suf)
        if ta not in runs or tb not in runs:
            continue
        pa = runs[ta]['aperture']['piston_c']
        pb = runs[tb]['aperture']['piston_c']
        diff = float(np.angle(np.exp(1j * (pa - pb))))
        if base is None:
            base = diff
        print('%-14s %16.9f %16.9f %16.9f %16.3e' % (
            suf, pa, pb, diff, diff - base))
    print()

    # (d) 1e-12 perturbation gain -----------------------------------------
    print('-- 3d  1e-12 RELATIVE UPSTREAM PERTURBATION -> PISTON GAIN -------')
    print('%-30s %-14s %12s %12s %12s %12s' % (
        'clean run', 'perturbation', 'd piston_c', 'd piston_w', 'relL2 dE',
        'gain (/eps)'))
    for order in ORDERS:
        for N, dx0 in ((1024, 2.0), (2048, 1.0)):
            t0 = key(N, dx0, order)
            if t0 not in runs:
                continue
            a = runs[t0]
            for suf in sorted(runs):
                if not suf.startswith(t0[:-4]) or suf == t0:
                    continue
                tail = suf[len(t0) - 4:]
                # ONLY the 1e-12 perturbation arms belong here.  The lever arms
                # (g*, tg*, rs1, frbf*, poly10, nit40) are DIFFERENT PHYSICS
                # or a different stop plane, not a perturbation of this run,
                # and reporting them as a "gain per eps" would be nonsense.
                if not (tail.startswith('field_') or tail.startswith('rdoe')):
                    continue
                b = runs[suf]
                eps = b['perturb']['eps']
                dpc = float(np.angle(np.exp(
                    1j * (b['aperture']['piston_c']
                          - a['aperture']['piston_c']))))
                dpw = float(np.angle(np.exp(
                    1j * (b['aperture']['piston_w']
                          - a['aperture']['piston_w']))))
                rel = float(np.linalg.norm(b['env'] - a['env'])
                            / np.linalg.norm(a['env']))
                print('%-30s %-14s %12.3e %12.3e %12.3e %12.3e' % (
                    t0, tail, dpc, dpw, rel, abs(dpc) / eps))
    print()

    # ------------------------------------------------- 3e. the LEVER arms
    # Everything the chain's own docstring calls dx-INVISIBLE, plus the
    # ray-subsample lattice.  Read against the SAME-order baseline run.
    print('-- 3e  LEVERS at N=1024 dx0=2.0 (full chain), vs the baseline arm --')
    print('%-9s %-9s %13s %13s %13s %10s' % (
        'order', 'arm', 'd piston_c', 'P_ap/P_in', 'dP/P vs base', 'wall_s'))
    for order in ((0, 0), (-1, 0), (-4, -2)):
        base = runs.get(key(1024, 2.0, order))
        if base is None:
            continue
        Pb = base['aperture']['power'] / base['chain_a']['P_in']
        for arm in ('none', 'frbf1p5', 'frbf3', 'poly10', 'nit40', 'rs1'):
            t = key(1024, 2.0, order, arm)
            if t not in runs:
                continue
            d = runs[t]
            P = d['aperture']['power'] / d['chain_a']['P_in']
            print('%-9s %-9s %13.3e %13.9f %13.3e %10.1f' % (
                '(%+d,%+d)' % order, arm,
                float(np.angle(np.exp(1j * (d['aperture']['piston_c']
                                            - base['aperture']['piston_c'])))),
                P, P / Pb - 1.0,
                d['chain_a']['wall_s'] + d['chain_b']['wall_s']))
    print()

    # ------------------------------------------------------- 4. C6 diagnostics
    print('=' * 108)
    print('TABLE 4 -- C6 STATIONARY-PHASE LAUNCH on this path '
          '(FIX_C13_BUILD_SPREAD_2026_08_06 S3.2 measured 1e12 gain on a '
          'MULTIPLEXED input)')
    print('=' * 108)
    print('%-34s %6s %8s %8s %14s %14s %10s' % (
        'run', 'calls', 'engaged', 'degree', 'max|grad a|', 'resid/data rms',
        'admissible'))
    for t in sorted(runs):
        d = runs[t]
        c6 = d['chain_b']['c6'] + d['chain_a']['c6']
        if not c6:
            continue
        g = [float(x['grad_a_fit_max_launch']) for x in c6
             if x.get('grad_a_fit_max_launch') not in (None, 'None')]
        rr, ra = [], []
        for x in c6:
            if x.get('grad_a_residual_rms') not in (None, 'None') \
                    and x.get('grad_a_rms') not in (None, 'None'):
                rr.append(float(x['grad_a_residual_rms']))
                ra.append(float(x['grad_a_rms']))
        deg = sorted({str(x.get('degree')) for x in c6})
        print('%-34s %6d %8d %8s %14.4e %14.4f %10s' % (
            t, len(c6), sum(1 for x in c6 if x.get('engaged') in (True, 'True')),
            ','.join(deg), max(g) if g else float('nan'),
            max((b / a) for a, b in zip(ra, rr) if a > 0) if ra else float('nan'),
            'YES' if (g and max(g) <= 1.0) else 'NO'))
    print()

    # ------------------------------------- 5. piston vs an exact OPL oracle
    # STABILITY (3a/3c) is not CORRECTNESS.  The inter-order piston difference
    # is the quantity a sum-at-aperture consumer actually needs, so it is
    # scored against an INDEPENDENT oracle: an exact skew ray trace of the
    # chief ray from the DOE plane to the same exit vertex, whose accumulated
    # ``opd`` is the geometric OPL.  Both congruences enter at (0,0) on the DOE
    # plane with the same field value, both are referenced to the same exit
    # sphere at their own chief ray (so the reference contributes 0 at the
    # sample point), and the chain's own transverse chief-ray prediction
    # agrees with this trace to < 1 nm -- so the only thing left between
    # ``k0 * dOPL`` and the measured ``d piston_c`` is the chain's own optical
    # -path bookkeeping.
    if os.environ.get('PL_ORACLE', '1') == '1':
        sys.path.insert(0, _HERE)
        import _d121_common as d121  # noqa: E402

        from lumenairy.raytrace import Surface, make_ray, trace  # noqa: E402
        _pre, _post, _gap, _per = d121.geometry()
        k0 = 2.0 * np.pi / d121.LAM

        def _surfs_for(ng):
            """Exact-ray surfaces truncated to the SAME stop plane the chain
            used at ``PL_NGROUPS = ng`` -- ng groups, ending at group ng-1's
            exit vertex (ng = 0 is the DOE -> first-group free leg alone)."""
            if ng <= 0:
                inf = float('inf')
                return [Surface(radius=inf, conic=0.0, semi_diameter=inf,
                                glass_before='air', glass_after='air',
                                is_mirror=False,
                                thickness=float(_post[0]['gap_before']),
                                label='doe_plane'),
                        Surface(radius=inf, conic=0.0, semi_diameter=inf,
                                glass_before='air', glass_after='air',
                                is_mirror=False, thickness=0.0, label='img')]
            return d121.post_surfaces(_post[:ng], trailing=0.0)

        _SC = {}

        def _opl(ng, L, M):
            if ng not in _SC:
                _SC[ng] = _surfs_for(ng)
            r = trace(make_ray(0.0, 0.0, L, M, wavelength=d121.LAM),
                      _SC[ng], d121.LAM)
            return (float(r.image_rays.opd[0]), float(r.image_rays.x[0]),
                    float(r.image_rays.y[0]))
        _surfs = _surfs_for(len(_post))
        print('=' * 108)
        print('TABLE 5 -- INTER-ORDER PISTON vs an EXACT CHIEF-RAY OPL ORACLE '
              '(reference order (0,0); bar lambda/100 = %.4e rad)'
              % BAR_PISTON)
        print('=' * 108)
        def _variant(d):
            """Tag tail after the (N, dx0, order) prefix -- the ARM label."""
            pre = 'n%d_dx%s_o%d_%d_' % (
                d['N'], ('%g' % (d['dx0'] * 1e6)).replace('.', 'p'),
                d['order'][0], d['order'][1])
            return d['tag'][len(pre):] if d['tag'].startswith(pre) else '?'

        def _armfam(v):
            """'tg3' (the forced-tilted-branch reference) shares a family with
            'g3' so the two (0,0) references sit in the same group."""
            v = re.sub(r'^t?g6$', 'none', v)      # ng = 6 IS the full run
            v = re.sub(r'^(g\d+)ts.*$', r'\1', v)  # tilt-scale arm of that ng
            v = re.sub(r'^ts.*$', 'none', v)       # tilt-scale, full chain
            return re.sub(r'^tg(\d+)$', r'g\1', v)

        groups = {}
        for t, d in runs.items():
            v = _variant(d)
            groups.setdefault(
                (d['N'], round(d['dx0'] * 1e6, 6), int(d.get('n_groups_b', 6)),
                 _armfam(v)), []).append(d)
        resid = {}
        for gk in sorted(groups):
            g = groups[gk]
            ng = gk[2]
            r_sca = [d for d in g if tuple(d['order']) == (0, 0)
                     and not d.get('x0')]
            r_til = [d for d in g if tuple(d['order']) == (0, 0)
                     and d.get('x0')]
            if not r_sca and not r_til:
                continue
            opl0 = _opl(ng, 0.0, 0.0)[0]
            print('  N=%d dx0=%.2f um  groups=%d  arm=%-10s traced_kwargs=%r'
                  % (gk[0], gk[1], ng, gk[3], g[0].get('traced_kwargs', {})))
            print('   %-13s %10s %15s %15s %14s %10s | %14s %10s | %11s' % (
                'order', 'tilt(mrad)', 'oracle k0*dOPL', 'measured dpist',
                'resid vs (0,0)', 'waves', 'vs TILTED(0,0)', 'waves',
                'UNWRAPPED'))
            for d in sorted(g, key=lambda q: (q['order'][0], q['order'][1],
                                              q.get('x0', 0.0))):
                o = tuple(d['order'])
                L = float(d.get('tilt_L', o[0] * d121.LAM / _per))
                M = float(d.get('tilt_M', o[1] * d121.LAM / _per))
                _o, _x, _y = _opl(ng, L, M)
                orc_raw = float(k0 * (_o - opl0))
                orc = float(np.angle(np.exp(1j * orc_raw)))
                out = []
                for ref in (r_sca, r_til):
                    if not ref:
                        out.extend([float('nan'), float('nan')])
                        continue
                    mes = float(np.angle(np.exp(
                        1j * (d['aperture']['piston_c']
                              - ref[0]['aperture']['piston_c']))))
                    rs = float(np.angle(np.exp(1j * (mes - orc))))
                    out.extend([mes, rs])
                resid[(gk[0], gk[1], ng, gk[3], o,
                       bool(d.get('x0')))] = (out[3], out[1])
                _o_lbl = '(%+d,%+d)%s%s' % (
                    o[0], o[1], '*' if d.get('x0') else '',
                    ('x%g' % d['tilt_scale']
                     if abs(d.get('tilt_scale', 1.0) - 1.0) > 1e-12 else ''))
                # UNWRAPPED ratio: lift the measured difference onto the
                # oracle's own branch (n = nearest winding) and divide.  This
                # is the number the mechanism shows up in -- a CONSTANT
                # fraction across four decades of tilt is a missing TERM, not
                # a discretisation error.
                _nw = round((orc_raw - out[0]) / TWO_PI)
                _rat = ((out[0] + _nw * TWO_PI) / orc_raw
                        if abs(orc_raw) > 1e-9 else float('nan'))
                print('   %-13s %10.4f %15.9f %15.9f %14.9f %10.5f | '
                      '%14.9f %10.5f | %11.7f'
                      % (_o_lbl,
                         1e3 * float(np.hypot(L, M)), orc,
                         out[0], out[1], out[1] / TWO_PI,
                         out[3], out[3] / TWO_PI, _rat))
            print()
        print('  * = the (0,0) order forced down the TILTED branch by '
              'x0 = 1 pm (PL_X0).  "resid vs (0,0)" uses the SCALAR-branch '
              'on-axis run as the phase reference; "vs TILTED(0,0)" uses the '
              'forced-tilted one, which cancels any scalar-vs-tilted branch '
              'offset and leaves only the genuinely TILT-dependent error.')
        print()

        # per-group increment: which group introduces the error
        print('  PER-GROUP INCREMENT of the residual (rad), N=1024 dx0=2.00, '
              'reference = TILTED (0,0) where available else scalar')
        print('   %-9s %10s %10s %10s %10s %10s %10s' % (
            'order', 'g=1', 'g=2', 'g=3', 'g=4', 'g=5', 'g=6'))
        _ords = sorted({k[4] for k in resid if k[4] != (0, 0)},
                       key=lambda q: (q[0], q[1]))
        for o in _ords:
            row = []
            for ng in range(1, 7):
                fam = 'g%d' % ng if ng < 6 else 'none'
                v = resid.get((1024, 2.0, ng, fam, o, False))
                if v is None:
                    row.append('    --   ')
                else:
                    row.append('%10.6f' % (v[0] if np.isfinite(v[0])
                                           else v[1]))
            print('   %-9s %s' % ('(%+d,%+d)' % o, ' '.join(row)))
        print()
    sys.stdout.flush()
