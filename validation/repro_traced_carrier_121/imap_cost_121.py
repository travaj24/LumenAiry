# INVERSE (MIXED) CHARACTERISTIC prototype, step 2: THE COST.
#
# LOCAL-ONLY.  NO library edit -- the only instrumentation is a logging handler
# attached to ``lumenairy.elements._lens_traced``'s own INFO stream (the module
# already emits one record per Newton iteration, carrying the point count) and
# ``cProfile``.  Neither changes any argument or any result.
#
# THE PREMISE UNDER TEST.  PROTO_HAMILTON_MAP_2026_08_11 sec 6.3 wrote the
# Newton pull-back row of its cost table as
#
#     "retrace: Newton pull-back, 8192^2 = 6.71e+07 px -- dominates the 419.2 s"
#
# and sec 8.3's redirect is sized off that number.  It was never MEASURED: the
# 419.2 s was the whole element, and the row is an attribution.  It is also
# WRONG -- ``sub = ray_subsample``, the inversion runs on ``X[::sub, ::sub]``,
# and at n_fine = 8192 / sub = 87 that is a 95 x 95 = 9 025-point lattice which
# is then INTERPOLATED to the wave grid.  This script measures all of it, at
# the shipped retrace configuration:
#
#   newtonsite -- one real ``apply_real_lens_traced`` at n_fine = 8192 with the
#                 chain's own kwargs, timed, with the Newton point count read
#                 out of the library's own per-iteration INFO log records.
#   profile    -- the same call under cProfile: where the seconds actually are.
#   mapcoords  -- the ``map_coordinates`` census: which full-grid interpolations
#                 an exact per-pixel inverse would replace, and what they cost.
#   fitorder   -- which forward-fit ORDER the element actually resolves (the
#                 niche-C11 arbiter can raise 6 -> _DECENTRED_FIT_POLY_ORDER),
#                 which decides which row of the incumbent ladder is the bar.
#   evalcost   -- one total-degree Chebyshev evaluation at 6.71e+07 points,
#                 MEASURED (numpy chunked and a numba kernel of the same shape
#                 as the shipped ``_cheb2d_val_grad_numba``), against the
#                 interpolation the shipped path uses instead.
#   newtonpp   -- the shipped Newton's own us/point on the real forward fit,
#                 scaled to 6.71e+07 -- what the redirect's premise ASSUMED.
#   build      -- G build cost and storage, from the measured lattice trace.
#
# Run:  python imap_cost_121.py newtonsite      (~2 min, ~21 GB)
#       python imap_cost_121.py profile         (~3 min, ~21 GB)
#       python imap_cost_121.py mapcoords       (~3 min, ~21 GB)
#       python imap_cost_121.py fitorder        (~2 min, ~21 GB)
#       python imap_cost_121.py evalcost        (~3 min, ~7 GB)
#       python imap_cost_121.py newtonpp        (~1 min)
#       python imap_cost_121.py build           (~1 min)
import json
import logging
import os
import re
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _d121_common as C  # noqa: E402
import imap_probe_121 as I  # noqa: E402
import hmap_probe_121 as H  # noqa: E402

import lumenairy as la  # noqa: E402
import lumenairy.elements as EL  # noqa: E402
import lumenairy.elements._lens_traced as LT  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(_HERE, '_imap_cost.json')

NFC = int(os.environ.get('NFC', '8192'))
ORDER = (int(os.environ.get('HM_M', '-4')), int(os.environ.get('HM_N', '-2')))


def _peak_gb():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return float(getattr(mi, 'peak_wset', mi.rss)) / 1e9
    except Exception:
        return float('nan')


# ===========================================================================
# a retrace-shaped call to apply_real_lens_traced
# ===========================================================================
def _retrace_call_args(n_fine=NFC, order=ORDER):
    """The last post-DOE group, driven exactly as ``_fine_trace_group_exit``
    drives it at ``n_fine_cap = 8192``: the same window, the same
    pitch-preserving ``ray_subsample``, and the chain's own ``base_kw``
    (``propagate_traced_carrier_chain``, ``amplitude_model='ray_density'`` +
    ``preserve_input_phase='remap'`` + ``remap_sampling='full'`` +
    ``fit_radius_beam_factor``)."""
    S = I.setup()
    presc = S['groups_post'][-1]['prescription']
    win = 2.0 * H.R_PUPIL_ORDER              # the shipped retrace window
    dx = win / n_fine
    rs = max(1, int(round(4 * 33.2112e-6 / dx)))
    # the extreme order's own congruence at this group's entrance
    row = None
    for r in S['rows']:
        if tuple(r['order']) == tuple(order):
            row = r
    if row is None:
        row = S['rows'][0]
    car = la.TiltedCarrier(float(S['R_ent']), float(row['L']), float(row['M']),
                           float(row['x_c']), float(row['y_c']))
    x = (np.arange(n_fine) - n_fine // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    w = 0.25 * win
    E = np.exp(-((X - row['x_c']) ** 2 + (Y - row['y_c']) ** 2) / w ** 2)
    E = E.astype(np.complex128)
    del X, Y
    from lumenairy.elements._lens_traced import _FIT_RADIUS_BEAM_FACTOR_DEFAULT
    kw = dict(prescription=presc, wavelength=C.LAM, dx=dx, carrier=car,
              ray_subsample=rs, n_workers=1, on_undersample='silent',
              amplitude_model='ray_density', preserve_input_phase='remap',
              fit_radius_beam_factor=_FIT_RADIUS_BEAM_FACTOR_DEFAULT,
              remap_sampling='full')
    return E, kw, {'n_fine': n_fine, 'dx': dx, 'ray_subsample': rs,
                   'window_mm': win * 1e3, 'order': list(order),
                   'n_launch': max(8, int(2 * 0.75 * float(
                       presc['aperture_diameter']) / (dx * rs))),
                   'n_pixels': int(n_fine) ** 2,
                   'coarse_grid': int(np.ceil(n_fine / rs)),
                   'coarse_points': int(np.ceil(n_fine / rs)) ** 2}


class _NewtonLogSpy(logging.Handler):
    """Read the Newton point count and the Newton wall time out of the
    library's OWN per-iteration INFO records.  Zero interference: the module
    emits these unconditionally; attaching a handler only stops them being
    discarded."""
    _RE = re.compile(r'newton iter (\d+)/(\d+) residual_max=([-\d.e+naif]+) m '
                     r'remaining=(\d+)/(\d+)')

    def __init__(self):
        logging.Handler.__init__(self, level=logging.INFO)
        self.events = []

    def emit(self, record):
        try:
            msg = record.getMessage()
        except Exception:
            return
        m = self._RE.search(msg)
        if m:
            self.events.append({'t': time.perf_counter(),
                                'iter': int(m.group(1)),
                                'remaining': int(m.group(4)),
                                'n_total': int(m.group(5))})
        elif 'converged' in msg and 'newton' in msg:
            self.events.append({'t': time.perf_counter(), 'iter': -1,
                                'remaining': 0, 'n_total': None})

    def summary(self):
        """One entry per Newton CALL (a call is a run of records whose
        ``iter`` restarts at 1)."""
        calls = []
        cur = None
        for e in self.events:
            if e['iter'] == 1 or cur is None:
                if cur is not None:
                    calls.append(cur)
                cur = {'t0': e['t'], 't1': e['t'], 'iters': 0,
                       'n_total': e['n_total'], 'remaining_last': e['remaining']}
            cur['t1'] = e['t']
            cur['iters'] = max(cur['iters'], e['iter'])
            cur['remaining_last'] = e['remaining']
            if e['n_total']:
                cur['n_total'] = e['n_total']
        if cur is not None:
            calls.append(cur)
        for c in calls:
            c['seconds_between_records'] = c['t1'] - c['t0']
        return calls


def newtonsite(verbose=True):
    """One real element call at the retrace configuration, timed, with the
    Newton point count read from the library's own log."""
    E, kw, meta = _retrace_call_args()
    spy = _NewtonLogSpy()
    lg = logging.getLogger('lumenairy.elements._lens_traced')
    old_level, old_prop = lg.level, lg.propagate
    lg.addHandler(spy)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    t0 = time.perf_counter()
    try:
        out = EL.apply_real_lens_traced(E, **kw)
    finally:
        lg.removeHandler(spy)
        lg.setLevel(old_level)
        lg.propagate = old_prop
    dt = time.perf_counter() - t0
    calls = spy.summary()
    n_newton_pts = sum(c['n_total'] or 0 for c in calls)
    t_newton = sum(c['seconds_between_records'] for c in calls)
    res = {**meta, 'element_seconds': dt, 'newton_calls': calls,
           'newton_points_total': int(n_newton_pts),
           'newton_seconds_between_first_and_last_record': t_newton,
           'peak_wset_gb': _peak_gb(),
           'field_shape': list(np.asarray(out).shape),
           'lumenairy_version': str(la.__version__)}
    if verbose:
        print("apply_real_lens_traced at the SHIPPED retrace configuration")
        print("  n_fine %d  dx %.4f um  window %.4f mm  ray_subsample %d  "
              "order %s" % (meta['n_fine'], meta['dx'] * 1e6,
                            meta['window_mm'], meta['ray_subsample'],
                            tuple(meta['order'])))
        print("  n_launch %d (%d rays);  exit pixels %.3e;  COARSE Newton "
              "lattice %d x %d = %d points"
              % (meta['n_launch'], meta['n_launch'] ** 2, meta['n_pixels'],
                 meta['coarse_grid'], meta['coarse_grid'],
                 meta['coarse_points']))
        print("  element wall %.1f s   peak working set %.1f GB"
              % (dt, res['peak_wset_gb']))
        print("  Newton CALLS seen in the library's own log: %d" % len(calls))
        for c in calls:
            print("     %8s points, %2d iterations, %.3f s between first and "
                  "last record, %d still active at the cap"
                  % (c['n_total'], c['iters'], c['seconds_between_records'],
                     c['remaining_last']))
        print("  TOTAL Newton points %d = %.4f %% of the %d exit pixels"
              % (n_newton_pts, 100.0 * n_newton_pts / meta['n_pixels'],
                 meta['n_pixels']))
        print("  TOTAL Newton seconds %.3f = %.3f %% of the element"
              % (t_newton, 100.0 * t_newton / dt))
    return res


def profile(top=45, verbose=True):
    """The same call under cProfile -- where the element's seconds actually
    are.  Overhead inflates the wall; the SHARES are the result."""
    import cProfile
    import pstats
    E, kw, meta = _retrace_call_args()
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    EL.apply_real_lens_traced(E, **kw)
    pr.disable()
    dt = time.perf_counter() - t0
    st = pstats.Stats(pr)
    st.sort_stats('tottime')
    rows = []
    for func, (cc, nc, tt, ct, _cal) in st.stats.items():
        rows.append({'file': os.path.basename(func[0]), 'line': func[1],
                     'name': func[2], 'ncalls': int(nc), 'tottime': tt,
                     'cumtime': ct})
    rows.sort(key=lambda r: -r['tottime'])
    rows = rows[:top]
    if verbose:
        print("cProfile of apply_real_lens_traced at the retrace configuration "
              "(wall %.1f s WITH profiler overhead)" % dt)
        print("  %-9s %-9s %-34s %8s %10s %10s"
              % ('file', 'line', 'name', 'ncalls', 'tottime', 'cumtime'))
        for r in rows:
            print("  %-9s %-9d %-34s %8d %10.3f %10.3f"
                  % (r['file'][:9], r['line'], r['name'][:34], r['ncalls'],
                     r['tottime'], r['cumtime']))
    return {**meta, 'profiled_wall': dt, 'top': rows,
            'peak_wset_gb': _peak_gb()}


def fitorder(verbose=True):
    """WHICH forward-fit order does the shipped element actually resolve?

    ``newton_poly_order`` defaults to 6, but ``_decentred_fit_restriction``
    raises it to ``_DECENTRED_FIT_POLY_ORDER = 10`` when the niche-C11 arbiter
    picks the OFF-CENTRE disc -- and the retrace's launch lattice is
    axis-centred while the beam sits at the chief ray, so which branch wins is
    a property of THIS call, not of the defaults.  The incumbent's accuracy
    ladder is steep in that order, so the answer decides which row of it is the
    bar an inverse map has to clear.  Recorded by wrapping the module-global
    ``_Cheb2DEvaluator`` (the element resolves it by module attribute lookup).
    """
    E, kw, meta = _retrace_call_args()
    seen = []
    Base = LT._Cheb2DEvaluator

    class _Spy(Base):
        def __init__(self, xs, ys, values, order=6, **kk):
            seen.append({'order': int(order),
                         'n_samples': int(np.size(values)),
                         'weighted': kk.get('weights') is not None})
            Base.__init__(self, xs, ys, values, order=order, **kk)

    LT._Cheb2DEvaluator = _Spy
    t0 = time.perf_counter()
    try:
        EL.apply_real_lens_traced(E, **kw)
    finally:
        LT._Cheb2DEvaluator = Base
    dt = time.perf_counter() - t0
    if verbose:
        print("forward Chebyshev fits built inside one element call "
              "(%.1f s wall)" % dt)
        for i, s in enumerate(seen):
            print("  %d  order %2d  (%d total-degree terms)  %d samples  "
                  "weighted=%s" % (i, s['order'],
                                   (s['order'] + 1) * (s['order'] + 2) // 2,
                                   s['n_samples'], s['weighted']))
        if seen:
            print("  -> the order actually in force for the Newton inversion: "
                  "%d" % max(s['order'] for s in seen))
    return {**meta, 'element_seconds': dt, 'fits': seen,
            'resolved_order': (max(s['order'] for s in seen) if seen else None)}


def mapcoords(verbose=True):
    """Which of the element's ``map_coordinates`` calls could an exact
    per-pixel inverse replace, and what do they cost?

    The profile puts ``scipy.ndimage``'s C entry points at the top of the
    element.  They are NOT one thing: some UPSAMPLE the coarse Newton result
    (the OPL, its NaN mask, the ray-density amplitude, the entrance
    coordinates) -- exactly what G would produce exactly, per pixel, instead --
    and some SAMPLE the input field at the entrance points, which G does not
    touch.  Spy on the scipy entry point (the module imports it inside the
    function, so patching ``scipy.ndimage`` reaches the call site) and record
    shape + order + seconds per call.
    """
    import scipy.ndimage as _nd
    E, kw, meta = _retrace_call_args()
    calls = []
    orig = _nd.map_coordinates

    def _spy(inp, coords, *a, **k):
        t0 = time.perf_counter()
        out = orig(inp, coords, *a, **k)
        calls.append({'in_shape': list(np.shape(inp)),
                      'out_points': int(np.size(coords) // 2),
                      'order': int(k.get('order', a[1] if len(a) > 1 else 3)),
                      'seconds': time.perf_counter() - t0})
        return out

    _nd.map_coordinates = _spy
    t0 = time.perf_counter()
    try:
        EL.apply_real_lens_traced(E, **kw)
    finally:
        _nd.map_coordinates = orig
    dt = time.perf_counter() - t0
    tot = sum(c['seconds'] for c in calls)
    full = [c for c in calls if c['out_points'] >= meta['n_pixels'] // 2]
    if verbose:
        print("map_coordinates census inside one element call (%.1f s wall)"
              % dt)
        for i, c in enumerate(calls):
            print("  %2d  in %-14s -> %10d pts  order %d  %7.3f s"
                  % (i, str(tuple(c['in_shape'])), c['out_points'],
                     c['order'], c['seconds']))
        print("  TOTAL %.3f s = %.2f %% of the element; FULL-GRID calls %d "
              "totalling %.3f s" % (tot, 100 * tot / dt, len(full),
                                    sum(c['seconds'] for c in full)))
    return {**meta, 'element_seconds': dt, 'calls': calls,
            'total_seconds': tot,
            'full_grid_seconds': sum(c['seconds'] for c in full),
            'full_grid_calls': len(full), 'peak_wset_gb': _peak_gb()}


# ===========================================================================
# the per-pixel evaluation cost
# ===========================================================================
def _numba_poly_value():
    """A VALUE-ONLY total-degree Chebyshev kernel of exactly the same shape as
    the shipped ``_lens_traced._cheb2d_val_grad_numba`` (3-term recurrence on
    the stack, ``prange`` over samples, ``fastmath``), so the number it
    produces is comparable with the shipped Newton's own evaluator."""
    try:
        from numba import njit, prange
    except ImportError:
        return None

    @njit(cache=True, parallel=True, fastmath=True)
    def _kern(coeffs, K1, K2, u, v, max_order, out):
        N = u.shape[0]
        M = coeffs.shape[0]
        nch = coeffs.shape[1]
        for i in prange(N):
            uu = u[i]
            vv = v[i]
            Tu = np.empty(max_order + 1)
            Tv = np.empty(max_order + 1)
            Tu[0] = 1.0
            Tv[0] = 1.0
            if max_order >= 1:
                Tu[1] = uu
                Tv[1] = vv
            for n in range(2, max_order + 1):
                Tu[n] = 2.0 * uu * Tu[n - 1] - Tu[n - 2]
                Tv[n] = 2.0 * vv * Tv[n - 1] - Tv[n - 2]
            for c in range(nch):
                acc = 0.0
                for m in range(M):
                    acc += coeffs[m, c] * Tu[K1[m]] * Tv[K2[m]]
                out[i, c] = acc
    return _kern


def evalcost(n_pix=None, deg=10, n_chan=4, reps=3, verbose=True):
    """MEASURE one polynomial evaluation over a full ``n_fine**2`` exit grid.

    Three arms, all on the same points:
      * numba value-only kernel (the production shape);
      * numpy chunked design-matrix contraction (the shipped ``_invert_fit``
        shape, chunked because a (6.71e7, 66) float64 design matrix is 35 GB);
      * ``scipy.ndimage.map_coordinates`` order-3, which is what the SHIPPED
        path spends per pixel instead (the coarse Newton is upsampled).
    """
    n_pix = int(n_pix if n_pix is not None else NFC ** 2)
    terms = I.td_terms(deg)
    P = terms.shape[0]
    rng = np.random.default_rng(7)
    coef = rng.standard_normal((P, n_chan))
    K1 = np.ascontiguousarray(terms[:, 0].astype(np.int64))
    K2 = np.ascontiguousarray(terms[:, 1].astype(np.int64))
    side = int(round(np.sqrt(n_pix)))
    ax = np.linspace(-1.0, 1.0, side)
    u = np.repeat(ax, side)
    v = np.tile(ax, side)
    out = {'n_pixels': n_pix, 'deg': deg, 'n_terms': int(P),
           'n_channels': int(n_chan), 'reps': int(reps)}

    kern = _numba_poly_value()
    if kern is not None:
        buf = np.empty((n_pix, n_chan))
        kern(coef[:8], K1[:8], K2[:8], u[:16], v[:16], deg, buf[:16])  # warm
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            kern(coef, K1, K2, u, v, deg, buf)
            ts.append(time.perf_counter() - t0)
        out['numba_seconds'] = float(min(ts))
        out['numba_all'] = ts
        del buf
        if verbose:
            print("  numba value-only, %d terms x %d channels over %.3e px: "
                  "%.3f s (best of %d)"
                  % (P, n_chan, n_pix, out['numba_seconds'], reps))

    # chunked numpy, the shipped _invert_fit shape
    CH = 4_000_000
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        acc = np.empty((n_pix, n_chan))
        for s in range(0, n_pix, CH):
            e = min(s + CH, n_pix)
            A = I.td_design(u[s:e], v[s:e], deg, terms)
            acc[s:e] = A @ coef
        ts.append(time.perf_counter() - t0)
        del acc, A
    out['numpy_chunked_seconds'] = float(min(ts))
    if verbose:
        print("  numpy chunked design@coef  over %.3e px: %.3f s (best of %d)"
              % (n_pix, out['numpy_chunked_seconds'], reps))

    # what the shipped path spends per pixel INSTEAD of a per-pixel Newton
    from scipy.ndimage import map_coordinates
    sub = 87
    Ns = int(np.ceil(side / sub))
    coarse = rng.standard_normal((Ns, Ns))
    idx = np.arange(side, dtype=np.float64) / sub
    coords = np.empty((2, side, side))
    coords[0] = idx[:, None]
    coords[1] = idx[None, :]
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        map_coordinates(coarse, coords, order=3, mode='nearest',
                        prefilter=True)
        ts.append(time.perf_counter() - t0)
    out['map_coordinates_order3_seconds'] = float(min(ts))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        map_coordinates(coarse, coords, order=1, mode='nearest',
                        prefilter=False)
        ts.append(time.perf_counter() - t0)
    out['map_coordinates_order1_seconds'] = float(min(ts))
    del coords, coarse
    if verbose:
        print("  scipy map_coordinates order-3 (the SHIPPED upsample of the "
              "coarse Newton) over %.3e px: %.3f s"
              % (n_pix, out['map_coordinates_order3_seconds']))
        print("  scipy map_coordinates order-1                              "
              "  over %.3e px: %.3f s"
              % (n_pix, out['map_coordinates_order1_seconds']))
    del u, v
    return out


def newton_perpixel_cost(n_pts=(9025, 100000, 1000000), verbose=True):
    """The shipped Newton's own per-point cost on design 121's real forward
    fit, at the retrace's tolerance and iteration cap.  Scaling the measured
    us/point to 6.71e+07 gives what the redirect's premise ASSUMED the shipped
    path pays."""
    S = I.setup()
    src = S['src']
    k = 0
    for i, r in enumerate(S['rows']):
        if tuple(r['order']) == (-4, -2):
            k = i
    F = I.newton_forward_fit(S, src[k, 0], src[k, 1])
    tt = I.order_test_points(S, S['rows'][k], n_pup=61)
    rows = []
    for n in n_pts:
        reps = max(1, int(np.ceil(n / max(tt['n'], 1))))
        xq = np.tile(tt['x_out'], reps)[:n]
        yq = np.tile(tt['y_out'], reps)[:n]
        t0 = time.perf_counter()
        nr = I.newton_invert(S, F, xq, yq)
        dt = time.perf_counter() - t0
        rows.append({'n': int(xq.size), 'seconds': dt,
                     'us_per_point': 1e6 * dt / xq.size,
                     'iters_mean': float(nr['iters'].mean())})
        if verbose:
            print("  Newton over %9d points: %8.3f s = %6.3f us/pt "
                  "(mean %.2f iterations)"
                  % (xq.size, dt, 1e6 * dt / xq.size, nr['iters'].mean()))
    best = min(r['us_per_point'] for r in rows)
    if verbose:
        print("  -> scaled to 6.71e+07 exit pixels: %.1f s"
              % (best * 67.1e6 / 1e6))
    return {'rows': rows, 'us_per_point_best': best,
            'projected_seconds_at_67e6': best * 67.1e6 / 1e6}


# ===========================================================================
def build(nlist=((6, 5), (7, 6), (8, 7)), degs=(8, 10, 12), verbose=True):
    """G build cost and storage, from the measured lattice trace."""
    S = I.setup()
    cache = {}
    rows = []
    for (nx, ny) in nlist:
        for deg in degs:
            t0 = time.perf_counter()
            G = I.build_G(S, nx, ny, deg, node_cache=cache)
            rows.append({'nx': nx, 'ny': ny, 'nodes': nx * ny, 'deg': deg,
                         'n_terms': G['n_terms'], 'bytes': G['bytes'],
                         'trace_seconds': G['t_trace'],
                         'fit_seconds': G['t_fit'],
                         'total_seconds': time.perf_counter() - t0,
                         'rays_traced': G['n_rays_traced'],
                         'fit_samples_per_node': G['n_fit_samples']})
            if verbose:
                r = rows[-1]
                print("  %dx%d nodes, deg %2d: trace %6.2f s + fit %6.2f s "
                      "(cached traces reused), %d terms, %.1f kB"
                      % (nx, ny, deg, r['trace_seconds'], r['fit_seconds'],
                         r['n_terms'], r['bytes'] / 1e3))
    # ONE cold build, uncached, for the honest number
    t0 = time.perf_counter()
    G = I.build_G(S, 6, 5, 10, node_cache=None)
    cold = time.perf_counter() - t0
    if verbose:
        print("  COLD build 6x5 nodes deg 10 (30 lattice traces, no cache): "
              "%.2f s  (%d rays)  -> %.1f kB"
              % (cold, G['n_rays_traced'], G['bytes'] / 1e3))
    return {'rows': rows, 'cold_seconds_6x5_deg10': cold,
            'cold_bytes': G['bytes'], 'cold_rays': G['n_rays_traced']}


def main(argv):
    mode = argv[1] if len(argv) > 1 else 'newtonsite'
    blob = {}
    if os.path.exists(OUT):
        try:
            blob = json.load(open(OUT, encoding='ascii'))
        except (ValueError, OSError):
            blob = {}
    fn = {'newtonsite': newtonsite, 'profile': profile, 'evalcost': evalcost,
          'build': build, 'newtonpp': newton_perpixel_cost,
          'mapcoords': mapcoords, 'fitorder': fitorder}[mode]
    blob[mode] = fn()
    with open(OUT, 'w', encoding='ascii') as fh:
        json.dump(blob, fh, indent=1, sort_keys=True, default=float)
    print("\nwrote %s" % os.path.basename(OUT))


if __name__ == '__main__':
    main(sys.argv)
