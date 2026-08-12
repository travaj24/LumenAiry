# ONE RUN of the design-121 chain to the BACK APERTURE of the last post-DOE
# group -- the shared instrument for BOTH probes in
# docs/audits/PROBE_CHAIN_LADDER_PISTON_2026_08_11.md:
#
#   * MEASUREMENT 1 (chain-grid convergence ladder) -- vary (N, dx0) and score
#     the aperture field between rungs.
#   * MEASUREMENT 2 (inter-chain piston consistency) -- re-run the SAME rung in
#     a FRESH PROCESS, at two rungs, at two orders, and under a 1e-12 relative
#     upstream perturbation, and score the piston.
#
# WHY THE BACK APERTURE AND NOT THE IMAGE PLANE.  The final leg (TRAILING =
# 7.7058 mm) is the exact high-NA Bluestein readout: it is 90 % of the wall
# clock (CAPSTONE_D121_2026_08_06 S6.5: 911 s/order, of which chain B's coarse
# legs are a small part) and it introduces its OWN grid (n_fine_cap,
# window_factor, dx_out) whose convergence is a different question from the
# COARSE CHAIN GRID this probe is about.  Stopping at ``final_distance=0.0``
# with ``focus_readout=None`` lands exactly on the last group's exit vertex --
# the plane an aggregating consumer would sum at -- and leaves ``final_leg``
# unread (carrier.py reads it only inside ``if is_final and focus_readout is
# not None``, proved byte-identical in CAPSTONE S2.5).
#
# WARNINGS ARE THE POINT, so this script installs NO blanket filter.  It cannot
# avoid ``_d121_common``'s three targeted ``filterwarnings('ignore', message=)``
# calls, which run at import -- so every chain call here is wrapped in
# ``catch_warnings(record=True) + simplefilter('always')``, which REPLACES the
# filter list for the duration and therefore records the 'under-sampled' and
# 'prescription aperture' warnings that import silenced.  That is the
# suppressed evidence CAPSTONE S4.4 / S9.2 left open.
#
# Env:
#   PL_N        coarse grid N              (default 1024)
#   PL_DX0      launch pitch, MICRONS      (default 2048.0/N, i.e. a fixed
#                                           2.048 mm window -- the PITCH arm)
#   PL_ORDER    "mx,my"                    (default "0,0")
#   PL_RS       ray_subsample              (default 4)
#   PL_NW       Newton workers             (default 1 -- see CAPSTONE S5: the
#                                           shipped 8 drove this box to 0.0 GB
#                                           free; the pool is a speed knob only)
#   PL_PERTURB  none | field | rdoe        (default none)
#   PL_EPS      relative perturbation      (default 1e-12)
#   PL_SEED     jitter seed                (default 1)
#   PL_TAG      artifact basename          (default auto)
#   PL_OUT      artifact dir               (default <here>/_probe_ladder)
#   PL_NOCACHE  1 = force a COLD chain A   (default 0; a warm chain A emits no
#                                           warnings, so the chain-A warning
#                                           census needs one cold pass)
#   PL_TSCALE   multiply the order's (L, M) by this factor (default 1).  A
#               CONTINUOUS tilt axis: it makes the piston-vs-oracle residual a
#               function of angle rather than of order index, which is what
#               separates a smooth missing TERM (residual -> 0 as tilt -> 0)
#               from a BRANCH switch (residual jumps to a constant the moment
#               ``tilted`` becomes True).  The exact-ray oracle is driven from
#               the SAME scaled (L, M), which the artifact records.
#   PL_X0       chief-ray x0 handed to TiltedCarrier, METRES (default 0).  Its
#               ONLY use is 1e-12: ``_parse_chain_carrier`` sets
#               ``tilted = bool(L or M or x0 or y0)``, so a 1 pm offset routes
#               the ON-AXIS order (0,0) down the TILTED branch at a physically
#               inert displacement (1e-12 m against a 33 um pitch).  That
#               separates a scalar-vs-tilted BRANCH piston offset from a
#               genuinely tilt-dependent one -- the two are indistinguishable
#               while the on-axis reference is the only scalar-branch run.
#   PL_NGROUPS  how many POST-DOE groups chain B runs (default all 6).  0 =
#               the DOE->first-group free leg alone.  This is the LOCALISER:
#               the same run at NGROUPS = 0..6 says WHICH stage a piston defect
#               enters at, and the exact-ray oracle can be truncated to match.
#   PL_TKW      JSON dict -> chain-B traced_kwargs (default {}).  The knobs the
#               chain's own docstring names as dx-INVISIBLE -- most of all
#               ``fit_radius_beam_factor`` (default 2.0), which it says "can
#               leave ~1 rad of exit-wavefront error ... and it does not move
#               with N".  Applied to chain B ONLY, so chain A stays the cached,
#               bit-identical field every arm shares.
import hashlib
import json
import os
import re
import sys
import time
import warnings

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)


# --- warning classification ------------------------------------------------
# Every message the 121 chain is known to raise, plus an '<other>' bucket so a
# NEW warning cannot be absorbed silently into an existing count.
_WCLASS = (
    ('undersampled_exit', re.compile(
        r'exit wavefront needs dx <=|beyond-Nyquist annulus')),
    ('newton_nonconv', re.compile(r'Newton inversion:.*did not converge')),
    ('prescription_aperture', re.compile(r'prescription aperture')),
    ('residual_transverse', re.compile(r'residual transverse')),
    ('undersampled_other', re.compile(r'under-sampled')),
    ('sphere_parab_bandlimit', re.compile(r'_sphere_parab_conversion')),
    ('zmx_dgrating', re.compile(r'is a DGRATING')),
    ('ram_cap', re.compile(
        r'_memory_bounded_n_fine|ram_budget|RAM budget|free memory')),
    ('na_proximity', re.compile(r'na_exact_threshold|NA proximity')),
    ('decentred_fit', re.compile(r'decentr')),
    ('gap_paraxial', re.compile(r'gap.*paraxial|paraxial.*gap')),
)
_RE_BANDLIMIT = re.compile(
    r'r_safe=([\d.eE+-]+) mm .*? at R=([\d.eE+-]+) mm, dx=([\d.eE+-]+)')
_RE_NEWTON = re.compile(
    r'(\d+)/(\d+) pixels \(([\d.]+)%\) did not converge to '
    r'tol=([\d.eE+-]+) m within (\d+) iterations')
_RE_UNDER = re.compile(
    r'NA_exit=([\d.eE+-]+).*?dx <= .*?= ([\d.eE+-]+) um but the grid has '
    r'dx = ([\d.eE+-]+) um', re.S)


def _classify(recs):
    """``(counts, newton_rows, undersample_rows, messages)`` from a recorded
    warning list.  Nothing is dropped: an unmatched message lands in
    ``<other>`` AND is kept verbatim.

    Counts are reported BOTH raw and de-duplicated by message text.  The
    element emits its Newton non-convergence warning twice per group (once
    from the forward map, once from the inverse), so a raw count reads 2x the
    number of AFFECTED GROUPS -- a distinction the verdict needs."""
    counts, ucounts, newton, under, msgs = {}, {}, [], [], []
    seen = set()
    for w in recs:
        s = str(w.message)
        msgs.append({'cat': w.category.__name__, 'msg': s[:400]})
        hit = None
        for name, rx in _WCLASS:
            if rx.search(s):
                hit = name
                break
        hit = hit or '<other>'
        counts[hit] = counts.get(hit, 0) + 1
        fresh = s not in seen
        seen.add(s)
        if fresh:
            ucounts[hit] = ucounts.get(hit, 0) + 1
        m = _RE_NEWTON.search(s)
        if m and fresh:
            newton.append({'bad': int(m.group(1)), 'tot': int(m.group(2)),
                           'pct': float(m.group(3)), 'tol': float(m.group(4)),
                           'iters': int(m.group(5))})
        m = _RE_UNDER.search(s)
        if m and fresh:
            under.append({'na_exit': float(m.group(1)),
                          'dx_need_um': float(m.group(2)),
                          'dx_have_um': float(m.group(3)),
                          'ratio': float(m.group(3)) / float(m.group(2))})
        m = _RE_BANDLIMIT.search(s)
        if m and fresh:
            under.append({'band_r_safe_mm': float(m.group(1)),
                          'band_R_mm': float(m.group(2))})
    return (counts, ucounts), newton, under, msgs


def _stats(env, dx):
    """Power, amplitude radius, peak, and the beam-weighted piston observables.

    ``piston_w``  -- intensity-weighted circular mean of the envelope phase,
                     ``arg(sum |E| E)``.  Beam-weighted by construction.
    ``piston_c``  -- the phase at the CHIEF RAY (the grid centre pixel).  The
                     accumulated OPL along the chief ray, i.e. the quantity an
                     aggregating consumer's inter-beam phase is built from.
    Both are read on the envelope with the sphere AND the tilt ramp divided
    out, so neither is contaminated by the carrier's own (huge) phase.
    """
    a = np.abs(env)
    p = float((a * a).sum()) * dx * dx
    tot = float((a * a).sum())
    n = env.shape[0]
    x = (np.arange(n, dtype=np.float64) - n / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    w = float(np.sqrt(2.0 * (a * a * r2).sum() / max(tot, 1e-300)))
    acc = complex(np.sum(a * env))
    return {'power': p, 'w_amp': w, 'peak': float(a.max()),
            'piston_w': float(np.angle(acc)), 'piston_w_absacc': abs(acc),
            'piston_c': float(np.angle(env[n // 2, n // 2])),
            'amp_c': float(a[n // 2, n // 2])}


if __name__ == '__main__':
    t_all = time.time()
    N = int(os.environ.get('PL_N', '1024'))
    DX0 = float(os.environ.get('PL_DX0', str(2048.0 / N))) * 1e-6
    ORD = tuple(int(v) for v in os.environ.get('PL_ORDER', '0,0').split(','))
    RS = int(os.environ.get('PL_RS', '4'))
    NW = int(os.environ.get('PL_NW', '1'))
    PERT = os.environ.get('PL_PERTURB', 'none')
    EPS = float(os.environ.get('PL_EPS', '1e-12'))
    SEED = int(os.environ.get('PL_SEED', '1'))
    OUT = os.environ.get('PL_OUT', os.path.join(_HERE, '_probe_ladder'))
    NOCACHE = os.environ.get('PL_NOCACHE', '0') == '1'
    TKW = json.loads(os.environ.get('PL_TKW', '{}'))
    NG = os.environ.get('PL_NGROUPS')
    X0 = float(os.environ.get('PL_X0', '0.0'))
    TSC = float(os.environ.get('PL_TSCALE', '1.0'))
    TAG = os.environ.get('PL_TAG', 'n%d_dx%s_o%d_%d_%s%s' % (
        N, ('%g' % (DX0 * 1e6)).replace('.', 'p'), ORD[0], ORD[1], PERT,
        ('' if PERT == 'none' else '_s%d' % SEED)))
    os.makedirs(OUT, exist_ok=True)

    print('=' * 78, flush=True)
    print('PROBE LADDER %s : N=%d dx0=%.4f um order=(%+d,%+d) rs=%d nw=%d '
          'perturb=%s eps=%g seed=%d tkw=%r'
          % (TAG, N, DX0 * 1e6, ORD[0], ORD[1], RS, NW, PERT, EPS, SEED, TKW),
          flush=True)
    print('=' * 78, flush=True)

    import _d121_common as d121  # noqa: E402

    import lumenairy as la  # noqa: E402
    import lumenairy.elements as _el  # noqa: E402
    from lumenairy.propagators.carrier import carrier_referenced_envelope  # noqa: E402

    # --- C6 stationary-phase-launch spy (script-side; no library edit) -----
    # FIX_C13_BUILD_SPREAD_2026_08_06 measured a 1e12 piston gain through this
    # launch ON A MULTIPLEXED (multi-valued) input.  This chain carries ONE
    # congruence, so the question is whether the same amplifier is in the path
    # here -- which the fit's own diagnostics answer directly.
    _C6 = []
    _orig_arlt = _el.apply_real_lens_traced

    def _arlt_spy(*a, **kw):
        d = {}
        kw['_remap_launch_out'] = d
        out = _orig_arlt(*a, **kw)
        _C6.append(d)
        return out

    _el.apply_real_lens_traced = _arlt_spy

    pre, post, gap_to_doe, period = d121.geometry()
    mx, my, amps = d121.order_table(period)
    sel = [i for i in range(len(mx)) if (int(mx[i]), int(my[i])) == ORD]
    if not sel:
        raise SystemExit('order %r not in the design table %r'
                         % (ORD, sorted(zip(mx.tolist(), my.tolist()))))
    kk = sel[0]
    amp = complex(amps[kk])
    L = TSC * float(ORD[0]) * d121.LAM / period
    M = TSC * float(ORD[1]) * d121.LAM / period
    print('DOE period %.4f um; order (%+d,%+d): a=%.6e%+.6ej  |a|^2=%.6e  '
          'tilt (L,M)=(%+.6e,%+.6e) rad'
          % (period * 1e6, ORD[0], ORD[1], amp.real, amp.imag,
             abs(amp) ** 2, L, M), flush=True)

    # ---------------- chain A: source -> DOE plane -------------------------
    _ck = d121._chain_a_key(N, DX0, RS, NW, 'exact')[1]
    _cf = os.path.join(_HERE, '_chainA_v%d_n%d_rs%d_%s.npz'
                       % (d121._CHAIN_A_SCHEMA, N, RS, _ck[:12]))
    cold = NOCACHE or not os.path.exists(_cf)
    _C6.clear()
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wA:
        warnings.simplefilter('always')
        env_doe, R_doe, dx_doe, P_in = d121.chain_a(
            n=N, dx0=DX0, rs=RS, nw=NW, cache=not NOCACHE, final_leg='exact')
    tA = time.time() - t0
    cA, nA, uA, mA = _classify(wA)
    c6A = list(_C6)
    P_doe = float(np.sum(np.abs(env_doe) ** 2)) * dx_doe * dx_doe
    print('chain A %s %.1f s: R_doe=%.6f mm dx=%.6f um  P=%.6f %% of input  '
          'warnings %d %r' % ('COLD' if cold else 'warm', tA, R_doe * 1e3,
                              dx_doe * 1e6, P_doe / P_in * 100, len(wA),
                              cA[0]),
          flush=True)

    # ---------------- the 1e-12 upstream perturbation ----------------------
    # Applied to the chain-B INPUT, i.e. strictly downstream of the cache, so
    # the perturbed and clean arms share a bit-identical chain A.
    pert = {'kind': PERT, 'eps': EPS, 'seed': SEED}
    if PERT == 'field':
        rng = np.random.default_rng(SEED)
        g = (rng.standard_normal(env_doe.shape)
             + 1j * rng.standard_normal(env_doe.shape))
        env_doe = env_doe * (1.0 + EPS * g)
        pert['applied_rel_rms'] = float(
            np.sqrt(np.mean(np.abs(EPS * g) ** 2)))
    elif PERT == 'rdoe':
        R_doe = R_doe * (1.0 + EPS)
        pert['applied_rel_rms'] = EPS
    elif PERT != 'none':
        raise SystemExit('PL_PERTURB must be none|field|rdoe, got %r' % PERT)

    # ---------------- chain B: DOE -> last group's BACK APERTURE -----------
    NG = len(post) if NG is None else int(NG)
    groups_b = post[:NG]
    fd_b = 0.0 if NG > 0 else float(post[0]['gap_before'])
    _C6.clear()
    t0 = time.time()
    with warnings.catch_warnings(record=True) as wB:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain(
            env_doe, groups_b, d121.LAM, dx_doe,
            r_in=la.TiltedCarrier(R_doe, L, M, X0, 0.0),
            ray_subsample=RS, n_workers=NW,
            traced_kwargs=(TKW or None),
            final_distance=fd_b)         # 0.0 = stop AT the back aperture
    tB = time.time() - t0
    cB, nB, uB, mB = _classify(wB)
    c6B = list(_C6)
    _el.apply_real_lens_traced = _orig_arlt

    field = np.asarray(res.field)
    R_ap, dx_ap = float(res.R), float(res.dx)
    tgt = [s for s in res.stages if s.get('target')]
    Lf = float(tgt[0]['L']) if tgt else 0.0
    Mf = float(tgt[0]['M']) if tgt else 0.0
    xc = float(tgt[0]['x_c']) if tgt else 0.0
    yc = float(tgt[0]['y_c']) if tgt else 0.0

    # envelope: divide out the sphere AND the tilt ramp (carrier.py builds the
    # returned field as reconstruct(env, R) * _tilt_ramp(..., 0, 0, +1), i.e.
    # the ramp is referenced to the grid centre = the chief ray).
    env = carrier_referenced_envelope(field, R_ap, d121.LAM, dx_ap)
    if Lf or Mf:
        n = field.shape[0]
        u = (np.arange(n, dtype=np.float64) - n / 2) * dx_ap
        k0 = 2.0 * np.pi / d121.LAM
        env = env * np.exp(-1j * k0 * (Lf * u[None, :] + Mf * u[:, None]))
    st = _stats(env, dx_ap)

    print('chain B %.1f s: R_ap=%.6f mm dx_ap=%.6f um  window=%.4f mm  '
          'w_amp=%.4f mm' % (tB, R_ap * 1e3, dx_ap * 1e6,
                             field.shape[0] * dx_ap * 1e3, st['w_amp'] * 1e3),
          flush=True)
    print('  chief ray at target: (%+.6f, %+.6f) mm   tilt (L,M)='
          '(%+.6e, %+.6e)' % (xc * 1e3, yc * 1e3, Lf, Mf), flush=True)
    print('  aperture power %.9e  (%.6f %% of chain-A input, %.6f %% of DOE)'
          % (st['power'], st['power'] / P_in * 100,
             st['power'] / P_doe * 100), flush=True)
    print('  piston_w %+.9f rad  piston_c %+.9f rad' % (st['piston_w'],
                                                        st['piston_c']),
          flush=True)
    print('  chain B warnings %d raw %r / unique %r'
          % (len(wB), cB[0], cB[1]), flush=True)
    for r in nB:
        print('    newton non-convergence %d/%d (%.1f %%) tol=%.3e it=%d'
              % (r['bad'], r['tot'], r['pct'], r['tol'], r['iters']),
              flush=True)
    for r in uB:
        if 'ratio' in r:
            print('    exit under-sampling NA=%.4f need dx<=%.3f um have '
                  '%.3f um (%.2fx)' % (r['na_exit'], r['dx_need_um'],
                                       r['dx_have_um'], r['ratio']),
                  flush=True)
        else:
            print('    sphere/parabola band limit r_safe=%.3f mm at R=%.3f mm'
                  % (r['band_r_safe_mm'], r['band_R_mm']), flush=True)
    if c6B:
        _g = [d.get('grad_a_fit_max_launch') for d in c6B
              if d.get('grad_a_fit_max_launch') is not None]
        _rr = [d.get('grad_a_residual_rms') for d in c6B
               if d.get('grad_a_residual_rms') is not None]
        _ra = [d.get('grad_a_rms') for d in c6B
               if d.get('grad_a_rms') is not None]
        print('  C6 launch: %d calls, engaged %d; max|grad a| %s; '
              'resid/data rms %s'
              % (len(c6B), sum(1 for d in c6B if d.get('engaged')),
                 ('%.3e' % max(_g)) if _g else 'n/a',
                 ('%.4f' % max(b / a for a, b in zip(_ra, _rr) if a > 0))
                 if (_ra and _rr) else 'n/a'), flush=True)

    # ---------------- artifacts -------------------------------------------
    doc = {
        'tag': TAG, 'N': N, 'dx0': DX0, 'order': list(ORD),
        'rs': RS, 'nw': NW, 'perturb': pert, 'traced_kwargs': TKW,
        'n_groups_b': NG, 'final_distance_b': fd_b, 'x0': X0,
        'lumenairy_version': str(la.__version__),
        'window_launch_m': N * DX0,
        'doe_period_m': period, 'amp_re': amp.real, 'amp_im': amp.imag,
        'tilt_L': L, 'tilt_M': M, 'tilt_scale': TSC,
        'chain_a': {'cold': bool(cold), 'wall_s': round(tA, 2),
                    'R_doe': R_doe, 'dx_doe': dx_doe, 'P_in': P_in,
                    'P_doe': P_doe, 'frac_in': P_doe / P_in,
                    'warn_n': len(wA),
                    'warn_counts': {'raw': cA[0], 'unique': cA[1]},
                    'newton': nA,
                    'undersample': uA, 'messages': mA,
                    'c6_calls': len(c6A),
                    'c6': [{k: (v if np.isscalar(v) or isinstance(v, (bool, str))
                                else str(v)) for k, v in d.items()}
                           for d in c6A]},
        'chain_b': {'wall_s': round(tB, 2), 'R_ap': R_ap, 'dx_ap': dx_ap,
                    'n_ap': int(field.shape[0]),
                    'window_ap_m': field.shape[0] * dx_ap,
                    'L': Lf, 'M': Mf, 'x_c': xc, 'y_c': yc,
                    'warn_n': len(wB),
                    'warn_counts': {'raw': cB[0], 'unique': cB[1]},
                    'newton': nB,
                    'undersample': uB, 'messages': mB,
                    'c6_calls': len(c6B),
                    'c6': [{k: (v if np.isscalar(v) or isinstance(v, (bool, str))
                                else str(v)) for k, v in d.items()}
                           for d in c6B]},
        'aperture': st,
        'field_sha256': hashlib.sha256(
            np.ascontiguousarray(field).tobytes()).hexdigest(),
        'env_sha256': hashlib.sha256(
            np.ascontiguousarray(env).tobytes()).hexdigest(),
        'stages': [{k: (float(v) if isinstance(v, (int, float, np.floating))
                        else str(v)) for k, v in s.items()}
                   for s in res.stages],
        'wall_s': round(time.time() - t_all, 2),
        'pid': os.getpid(),
    }
    with open(os.path.join(OUT, '%s.json' % TAG), 'w', encoding='cp1252',
              errors='replace') as fh:
        json.dump(doc, fh, indent=1, default=str)
    np.savez(os.path.join(OUT, '_%s_env.npz' % TAG),
             env=np.ascontiguousarray(env),
             meta=np.array(json.dumps(
                 {'N': N, 'dx0': DX0, 'dx_ap': dx_ap, 'R_ap': R_ap,
                  'L': Lf, 'M': Mf, 'x_c': xc, 'y_c': yc,
                  'order': list(ORD)})))
    print('WROTE %s.json (+ _%s_env.npz)  total wall %.1f s'
          % (TAG, TAG, doc['wall_s']), flush=True)
