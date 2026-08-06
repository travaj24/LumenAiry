# CAPSTONE runner for design 121 -- pre-flight assertions, timed chain-A build,
# and the Sziklas-Siegman residual tabulation.  ASCII only.
#
# LOCAL-ONLY (needs the 121 .zmx + the design-study runner), like every other
# script in this directory.  It adds NO physics: every number it prints is read
# out of the library or measured by running the library's own entry points.
#
# Modes:
#   preflight  -- the six assertions that must hold BEFORE any long stage.
#                 Exits non-zero on the first failure.
#   stageA     -- timed chain-A build + warm round-trip identity check.
#   stageD     -- Sziklas-Siegman per-leg magnification table for the REAL
#                 121 chain, mapped onto the measured D11 residual table.
#
# Stages B (focus_scan_121.py) and C (fan_multi_121.py) are the existing
# runners and are invoked directly, not from here.
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import _d121_common as D  # noqa: E402
import lumenairy as la  # noqa: E402
from lumenairy.propagators import carrier as C  # noqa: E402
from lumenairy.elements import _lens_traced as LT  # noqa: E402

_FAILS = []


def _chk(name, ok, detail=''):
    tag = 'PASS' if ok else 'FAIL'
    print(f"  [{tag}] {name}" + (f"\n         {detail}" if detail else ''),
          flush=True)
    if not ok:
        _FAILS.append(name)
    return ok


# ===========================================================================
# PRE-FLIGHT
# ===========================================================================
def preflight():
    print("=" * 74)
    print("CAPSTONE PRE-FLIGHT -- design 121")
    print("=" * 74)
    print(f"python           {sys.version.split()[0]}")
    print(f"numpy            {np.__version__}")
    print(f"lumenairy        {la.__version__}  {os.path.dirname(la.__file__)}")
    print()

    # ---- (a) chain-A cache key -------------------------------------------
    print("(a) chain-A cache key")
    key, digest = D._chain_a_key(1024, 2.0e-6, 4, 8, 'exact')
    print(f"      fields ({len(key)}): {sorted(key)}")
    _chk('key has 21 fields', len(key) == 21, f"len = {len(key)}")
    _chk("'numba_available' is a key field", 'numba_available' in key)
    gate = bool(LT._NUMBA_AVAILABLE)
    _chk('numba_available is read from the LIBRARY gate '
         '(_lens_traced._NUMBA_AVAILABLE)',
         key['numba_available'] == gate,
         f"key={key['numba_available']!r}  library gate={gate!r}")
    try:
        import importlib.util
        _spec = importlib.util.find_spec('numba') is not None
    except Exception:                                     # pragma: no cover
        _spec = None
    print(f"      (independent find_spec('numba') = {_spec!r}; the gate is "
          f"what selects the code path)")
    print(f"      digest[:12] = {digest[:12]}   schema = "
          f"{D._CHAIN_A_SCHEMA}")
    print(f"      lumenairy_source_sha256[:16] = "
          f"{key['lumenairy_source_sha256'][:16]}")
    print()

    # ---- (b) no pre-existing cache matches --------------------------------
    print("(b) chain-A cache state (fresh build expected)")
    existing = sorted(f for f in os.listdir(_HERE)
                      if f.startswith('_chainA') and f.endswith('.npz'))
    for f in existing:
        sz = os.path.getsize(os.path.join(_HERE, f)) / 1e6
        print(f"      present: {f}  ({sz:.1f} MB)")
    if not existing:
        print("      present: <none>")
    want = f'_chainA_v{D._CHAIN_A_SCHEMA}_n1024_rs4_{digest[:12]}.npz'
    print(f"      wanted : {want}")
    _v2 = [f for f in existing
           if f.startswith(f'_chainA_v{D._CHAIN_A_SCHEMA}_')]
    # ONE-SHOT PRECONDITION.  These two are true only BEFORE stage A has run;
    # stage A's whole point is to build the cache, so a re-run of preflight
    # after it legitimately reports them as failed.  Say which files are the
    # capstone's own so a later reader is not left guessing.
    if _v2:
        print(f"      NOTE: {len(_v2)} schema-v{D._CHAIN_A_SCHEMA} cache "
              f"file(s) already present.  If this capstone's stage A (or its")
        print(f"      'leginert' mode) has already run in this tree, they are "
              f"ITS output and these two")
        print(f"      checks are EXPECTED to fail on a re-run -- they are a "
              f"pre-stage-A precondition,")
        print(f"      not an invariant.  Delete them to restore the cold "
              f"state: {_v2}")
    _chk('no cache file matches this configuration (cold build expected)',
         want not in existing)
    _chk(f'no schema-v{D._CHAIN_A_SCHEMA} cache file exists at all',
         not _v2,
         'the legacy _chainA_*_{dx}nm_rs*.npz names are schema-v1 and are '
         'unreachable by the current key')
    print()

    # ---- shared: build a ONE-GROUP real 121 chain and spy on it -----------
    # Probes (c) and (e) are answered from the REAL chain code path, not from
    # a docstring: the first pre-DOE group of the actual prescription, at a
    # small grid so it costs seconds.
    print("(c)+(e) probing the REAL chain path (first 121 group, N=256)")
    pre, _post, gap_to_doe, _per = D.geometry()
    kernels = []
    _orig_step = C._envelope_tf_step

    def _spy_step(E_env, z_eff, wavelength, dx, dy, tilt, gap_kernel,
                  xp, is_jax, bld):
        kernels.append(str(gap_kernel))
        return _orig_step(E_env, z_eff, wavelength, dx, dy, tilt, gap_kernel,
                          xp, is_jax, bld)

    # The chain does ``from ..elements import apply_real_lens_traced`` INSIDE
    # the function body, so the attribute is looked up on the package at call
    # time -- patch it there, not on the carrier module.
    import lumenairy.elements as LE
    lens_calls = []
    _orig_lens = LE.apply_real_lens_traced

    def _spy_lens(*a, **kw):
        lens_calls.append(dict(kw))
        return _orig_lens(*a, **kw)

    fitinfo = []
    _orig_cheb = LT._Cheb2DEvaluator

    class _SpyCheb(_orig_cheb):
        def __init__(self, *a, **kw):
            fr = sys._getframe(1).f_locals
            fitinfo.append({
                k: repr(fr.get(k, '<absent>')) for k in (
                    'newton_fit', '_newton_fit_requested',
                    'fit_radius_beam_factor', '_frbf', '_beam_fit_radius',
                    '_fit_r_max', '_fit_r_geom', '_fit_domain_basis_ok',
                    '_fit_poly_order', '_fit_why', '_beam_decentred')})
            super().__init__(*a, **kw)

    spline_calls = []
    try:
        import scipy.interpolate as _si
        _orig_rbs = _si.RectBivariateSpline

        def _spy_rbs(*a, **kw):
            spline_calls.append(1)
            return _orig_rbs(*a, **kw)
        _si.RectBivariateSpline = _spy_rbs
    except ImportError:                                   # pragma: no cover
        _orig_rbs = None

    C._envelope_tf_step = _spy_step
    LE.apply_real_lens_traced = _spy_lens
    LT._Cheb2DEvaluator = _SpyCheb
    try:
        n, dx0 = 256, 8.0e-6
        zR = np.pi * D.W0 * D.W0 / D.LAM
        w_z1 = D.W0 * np.sqrt(1 + (D.Z1 / zR) ** 2)
        R1 = D.Z1 * (1 + (zR / D.Z1) ** 2)
        x = (np.arange(n) - n // 2) * dx0
        e0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                    / w_z1 ** 2).astype(np.complex128)
        t = time.perf_counter()
        _r = la.propagate_traced_carrier_chain(
            e0, pre[:1], D.LAM, dx0, r_in=R1, ray_subsample=4, n_workers=4,
            final_distance=1e-3)
        dt = time.perf_counter() - t
    finally:
        C._envelope_tf_step = _orig_step
        LE.apply_real_lens_traced = _orig_lens
        LT._Cheb2DEvaluator = _orig_cheb
        if _orig_rbs is not None:
            _si.RectBivariateSpline = _orig_rbs
    print(f"      one-group probe chain ran in {dt:.2f} s, "
          f"field finite = {bool(np.isfinite(_r.field).all())}")

    # (c) gap kernel
    print("(c) gap_kernel resolution ON THE CHAIN PATH")
    print(f"      chain signature default: gap_kernel="
          f"{_sig_default(la.propagate_traced_carrier_chain, 'gap_kernel')!r}")
    print(f"      resolved kernels observed at the transfer-function site: "
          f"{sorted(set(kernels))}  ({len(kernels)} leg step(s))")
    _chk('every chain leg resolved to the EXACT gap kernel',
         len(kernels) > 0 and set(kernels) == {'exact'},
         f"observed {sorted(set(kernels))!r}")
    # and the resolver itself, at the single funnel site
    kernels.clear()
    C._envelope_tf_step = _spy_step
    try:
        C._carrier_step_fast(np.ones((8, 8), complex), -0.05, 1e-3, D.LAM,
                             1e-5, 1e-5, gap_kernel='auto')
    finally:
        C._envelope_tf_step = _orig_step
    _chk("_carrier_step_fast('auto') resolves to 'exact'",
         kernels == ['exact'], f"observed {kernels!r}")

    # (e) newton fit + fit radius
    print("(e) newton_fit / fit_radius_beam_factor ON THE CHAIN PATH")
    print(f"      apply_real_lens_traced signature default: newton_fit="
          f"{_sig_default(la.apply_real_lens_traced, 'newton_fit')!r}, "
          f"fit_radius_beam_factor="
          f"{_sig_default(la.apply_real_lens_traced, 'fit_radius_beam_factor')!r}")
    frbf_passed = [kw.get('fit_radius_beam_factor', '<absent>')
                   for kw in lens_calls]
    nf_passed = [kw.get('newton_fit', '<absent>') for kw in lens_calls]
    print(f"      chain passed to apply_real_lens_traced: "
          f"fit_radius_beam_factor={frbf_passed}, newton_fit={nf_passed}")
    print(f"      LIBRARY default constant _FIT_RADIUS_BEAM_FACTOR_DEFAULT = "
          f"{LT._FIT_RADIUS_BEAM_FACTOR_DEFAULT}")
    for i, f in enumerate(fitinfo):
        print(f"      fit-site locals [{i}]: " + json.dumps(f, indent=None))
    _chk('polynomial (Chebyshev) evaluator was built',
         len(fitinfo) > 0, f"{len(fitinfo)} evaluator(s)")
    _chk('SciPy RectBivariateSpline was NOT used (spline path not taken)',
         not spline_calls, f"{len(spline_calls)} spline build(s)")
    _chk("newton_fit resolved to 'polynomial'",
         all(f['newton_fit'] == "'polynomial'" for f in fitinfo),
         str([f['newton_fit'] for f in fitinfo]))
    _chk('fit_radius_beam_factor reached the element as 2.0',
         all(v == LT._FIT_RADIUS_BEAM_FACTOR_DEFAULT for v in frbf_passed),
         str(frbf_passed))
    _chk('the beam-relative ray-fit disc is ACTIVE at the fit site '
         '(_beam_fit_radius set, _fit_r_max finite, basis ok)',
         all(f['_beam_fit_radius'] not in ("None", "'<absent>'")
             and f['_fit_r_max'] not in ("None", "'<absent>'")
             and f['_fit_domain_basis_ok'] == 'True' for f in fitinfo),
         str([(f['_beam_fit_radius'], f['_fit_r_max'],
               f['_fit_domain_basis_ok']) for f in fitinfo]))
    print()

    # ---- (d) final_leg ----------------------------------------------------
    print("(d) final_leg")
    thr = _sig_default(la.propagate_traced_carrier_chain,
                       'na_exact_threshold')
    fl = _sig_default(la.propagate_traced_carrier_chain, 'final_leg')
    print(f"      chain signature defaults: final_leg={fl!r}, "
          f"na_exact_threshold={thr!r}")
    na121 = 0.405
    resolves = 'exact' if na121 > float(thr) else 'paraxial'
    print(f"      resolver rule (carrier.py: do_exact = final_leg=='exact' or "
          f"(final_leg=='auto' and na_exit > na_exact_threshold))")
    print(f"      design 121 na_exit = {na121} > {thr} -> 'auto' routes "
          f"{resolves.upper()}")
    _chk("'auto' routes EXACT at 121's na_exit=0.405", resolves == 'exact')
    _chk("chain_a() default is final_leg='exact' (harness does NOT override "
         "to paraxial)",
         _sig_default(D.chain_a, 'final_leg') == 'exact',
         f"_d121_common.chain_a default = "
         f"{_sig_default(D.chain_a, 'final_leg')!r}")
    print("      static scan of the three harness files for final_leg:")
    remnants = []
    for fn in ('_d121_common.py', 'focus_scan_121.py', 'fan_multi_121.py'):
        p = os.path.join(_HERE, fn)
        for ln, line in enumerate(open(p, encoding='utf-8',
                                       errors='ignore').read().splitlines(), 1):
            s = line.strip()
            if s.startswith('#'):
                continue
            if 'final_leg=' in s or "LEG = os.environ" in s:
                print(f"        {fn}:{ln}: {s}")
                if "final_leg='paraxial'" in s or "'LEG', 'paraxial'" in s:
                    remnants.append(f"{fn}:{ln}")
    _chk('no ACTIVE paraxial final_leg remnant on the acceptance path '
         '(focus_scan_121.py)',
         not any(r.startswith('focus_scan') for r in remnants),
         f"remnants found: {remnants}")
    if remnants:
        print(f"      NOTE -- paraxial spellings present in: {remnants}")
        print("        fan_multi_121.py:218 is chain A, which passes NO "
              "focus_readout; the chain reads final_leg only inside")
        print("        'if is_final and focus_readout is not None' "
              "(carrier.py:7516/7785), so it is INERT there.")
        print("        fan_multi_121.py:305 LEG default IS live for chain B "
              "-- the capstone runs it with LEG=auto.")
    print()

    # ---- (f) replica guard ------------------------------------------------
    print("(f) replica guard on the readout path")
    n = 512
    dxr = 2.0e-6
    w = 4.0e-4
    xg = (np.arange(n) - n // 2) * dxr
    env = np.exp(-(xg[None, :] ** 2 + xg[:, None] ** 2)
                 / w ** 2).astype(np.complex128)
    per = {}
    ok_small = False
    try:
        C.carrier_referenced_focus_readout(
            env, -0.05, 0.05 - 5e-4, D.LAM, dxr,
            dx_out=0.05e-6, N_out=64, _period_out=per)
        ok_small = True
    except Exception as exc:                              # pragma: no cover
        print(f"      in-period readout unexpectedly raised: {exc}")
    pm = per.get('period')
    print(f"      Bluestein period of this readout: {pm}")
    print(f"      in-period window (64 x 0.05 um = 3.2 um) accepted: "
          f"{ok_small}")
    # Size the OFF-period window from the measured period, so the probe cannot
    # silently test an in-period window (it did on the first cut: 819.2 um
    # against a 1024 um period, and the guard was RIGHT to accept it).
    p_um = float(np.min(pm)) * 1e6
    dxo_bad, n_bad = 1.0e-6, int(2 ** np.ceil(np.log2(1.5 * p_um)))
    print(f"      off-period probe window: {n_bad} x 1.000 um = "
          f"{n_bad * 1.0:.1f} um against a {p_um:.1f} um period "
          f"({n_bad * 1.0 / p_um:.3f}x)")
    refused = None
    try:
        C.carrier_referenced_focus_readout(
            env, -0.05, 0.05 - 5e-4, D.LAM, dxr,
            dx_out=dxo_bad, N_out=n_bad)
        refused = False
    except RuntimeError as exc:
        refused = True
        print(f"      REFUSED: {str(exc)[:160]}...")
    _chk('an off-period readout is REFUSED by default (on_replica=error)',
         refused is True)
    ign = None
    try:
        C.carrier_referenced_focus_readout(
            env, -0.05, 0.05 - 5e-4, D.LAM, dxr,
            dx_out=dxo_bad, N_out=n_bad, on_replica='ignore')
        ign = True
    except Exception as exc:
        ign = False
        print(f"      on_replica='ignore' raised: {exc}")
    _chk("the refusal comes from the replica guard "
         "(on_replica='ignore' lets the same call through)", ign is True)
    print(f"      multi-orchestrator signature default: on_replica="
          f"{_sig_default(la.propagate_traced_carrier_chain_multi, 'on_replica')!r}")
    _chk("chain_multi default on_replica='error'",
         _sig_default(la.propagate_traced_carrier_chain_multi,
                      'on_replica') == 'error')
    print()

    print("=" * 74)
    if _FAILS:
        print(f"PRE-FLIGHT FAILED: {len(_FAILS)} assertion(s): {_FAILS}")
        return 1
    print("PRE-FLIGHT: ALL ASSERTIONS PASS")
    return 0


def _sig_default(fn, name):
    import inspect
    return inspect.signature(fn).parameters[name].default


# ===========================================================================
# STAGE A -- timed chain-A build + warm round trip
# ===========================================================================
def stage_a():
    import psutil
    proc = psutil.Process()
    n = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    nw = int(os.environ.get('NW', '8'))
    leg = os.environ.get('LEG', 'exact')
    print(f"STAGE A -- chain A (source -> DOE), N={n} rs={rs} nw={nw} "
          f"final_leg={leg!r}")
    key, digest = D._chain_a_key(n, 1.0e-6 * 2048 / n, rs, nw, leg)
    fn = f'_chainA_v{D._CHAIN_A_SCHEMA}_n{n}_rs{rs}_{digest[:12]}.npz'
    print(f"  cache file: {fn}   exists before: "
          f"{os.path.exists(os.path.join(_HERE, fn))}")
    kernels = []
    _orig = C._envelope_tf_step

    def _spy(E, z, wl, dx, dy, tilt, gk, xp, ij, bld):
        kernels.append(str(gk))
        return _orig(E, z, wl, dx, dy, tilt, gk, xp, ij, bld)
    C._envelope_tf_step = _spy
    try:
        t = time.perf_counter()
        env, R, dx, P_in = D.chain_a(n=n, rs=rs, nw=nw, final_leg=leg)
        t_cold = time.perf_counter() - t
    finally:
        C._envelope_tf_step = _orig
    rss = proc.memory_info().rss / 2 ** 30
    print(f"  COLD build : {t_cold:.2f} s   peak-ish RSS {rss:.3f} GB")
    print(f"  gap kernels resolved on the real chain-A legs: "
          f"{sorted(set(kernels))}  ({len(kernels)} steps)")
    print(f"  R_doe = {R * 1e3:.6f} mm   dx = {dx * 1e6:.6f} um   "
          f"P_in = {P_in:.9e}")
    print(f"  env: shape {env.shape} dtype {env.dtype} finite "
          f"{bool(np.isfinite(env).all())}  "
          f"power {float(np.sum(np.abs(env) ** 2)) * dx * dx / P_in * 100:.4f}"
          f" % of input")
    print(f"  cache written: {os.path.exists(os.path.join(_HERE, fn))} "
          f"({os.path.getsize(os.path.join(_HERE, fn)) / 1e6:.1f} MB)")
    t = time.perf_counter()
    env2, R2, dx2, P2 = D.chain_a(n=n, rs=rs, nw=nw, final_leg=leg)
    t_warm = time.perf_counter() - t
    ident = bool(np.array_equal(env, env2)) and R == R2 and dx == dx2 \
        and P_in == P2
    print(f"  WARM load  : {t_warm:.2f} s   round-trip BYTE-IDENTICAL: "
          f"{ident}")
    print(f"  speedup    : {t_cold / max(t_warm, 1e-9):.1f}x")
    print(f"STAGE_A_RESULT {json.dumps({'t_cold': t_cold, 't_warm': t_warm, 'identical': ident, 'rss_gb': rss, 'kernels': sorted(set(kernels)), 'R': R, 'dx': dx})}")
    return 0 if ident else 1


# ===========================================================================
# STAGE D -- Sziklas-Siegman per-leg magnification on the REAL 121 chain
# ===========================================================================
#
# Every carrier leg runs on the co-moving (Sziklas-Siegman) grid, whose
# magnification is m = R_out / R_in = (R_in + z) / R_in.  The SS frame is the
# one structural paraxial element left on the 121 path (handoff sec 2), and
# FIX_D4_D6_D7_2026_08_06.md sec D11 measured, against an independent exact
# ASM oracle of the FULL field, how far the exact kernel sits from truth
# INSIDE a scaled step at m = 0.700 / 0.750 / 0.850 / 1.150.  This stage reads
# each real 121 leg's m out of the chain's own stage diagnostics and maps it
# onto that table.  No new oracle is run.
_D11 = [
    # (m, w_um, NA_env, R_mm, exact_1leg, exact_2leg, fresnel, ratio)
    (0.700, 5.0, 0.0834, -1.00, 4.793e-3, 8.267e-3, 1.058e-2, 0.453),
    (0.700, 5.0, 0.0834, -0.50, 2.723e-3, 4.345e-3, 5.418e-3, 0.503),
    (0.750, 5.0, 0.0834, -2.00, 6.037e-3, 1.026e-2, 1.752e-2, 0.345),
    (0.850, 5.0, 0.0834, -1.00, 1.185e-3, 1.652e-3, 5.303e-3, 0.224),
    (0.850, 4.0, 0.1042, -2.00, 4.783e-3, 7.411e-3, 2.575e-2, 0.186),
    (1.150, 5.0, 0.0834, +2.00, 1.554e-3, 1.995e-3, 1.052e-2, 0.148),
]


#: The single largest 1-leg exact-vs-oracle residual in the D11 table.  Used
#: as the CONSERVATIVE envelope for any leg whose |m - 1| is inside the
#: measured span, because the table is NOT monotone in |m - 1| alone (see
#: :func:`_ss_bound`) and an interpolant through non-monotone data is an
#: estimate, not a bound.
_D11_MAX = max(r[4] for r in _D11)


def _ss_bound(m):
    """ESTIMATE the D11-measured 1-leg exact-vs-oracle residual at this |m-1|.

    The six D11 rows are the only direct measurements of the SS-frame residual
    inside a scaled step.  They are NOT a clean function of |m - 1|: grouping
    them gives 2.51e-3 at |m-1| = 0.150 (three rows, 1.19e-3 / 4.78e-3 /
    1.55e-3), 6.04e-3 at 0.250 and 3.76e-3 at 0.300 -- the spread WITHIN a
    |m-1| is as large as the variation between them, because the residual also
    depends on the envelope NA and on R.  So this returns a linear interpolant
    as an ESTIMATE and says so; the conservative figure to quote is
    ``_D11_MAX`` (6.04e-3) per leg inside the measured span.

    ``m == 1`` exactly is the one regime the exact kernel is genuinely exact
    in (D11: 1e-12 against the ASM oracle, and it composes across splits)."""
    d = abs(m - 1.0)
    xs = sorted({abs(r[0] - 1.0) for r in _D11})
    ys = [float(np.mean([r[4] for r in _D11 if abs(abs(r[0] - 1.0) - x) < 1e-9]))
          for x in xs]
    if d <= 1e-12:
        return 0.0, 'exact (m=1; D11 measured 1e-12)'
    if d > xs[-1] * 1.35:
        # REFUSE.  Linearly extrapolating a six-point, NON-monotone table far
        # past its span produces a number with no measurement behind it, and
        # this capstone's whole subject is numbers that look like measurements
        # and are not.  Report it as uncalibrated and exclude it from totals.
        return None, (f'NOT CALIBRATED -- |m-1| is {d / xs[-1]:.0f}x the D11 '
                      f'span (max {xs[-1]:.2f}); no D11 row bounds this')
    if d < xs[0]:
        return (ys[0] * d / xs[0],
                f'est., scaled below the measured span (|m-1| < {xs[0]:.2f})')
    if d > xs[-1]:
        return ys[-1] * d / xs[-1], 'est., just past the span (< 1.35x)'
    # Directly on a measured row?  Say so -- three of 121's post-DOE legs are.
    for x, y in zip(xs, ys):
        if abs(d - x) < 0.02:
            lo = min(r[4] for r in _D11 if abs(abs(r[0] - 1.0) - x) < 1e-9)
            hi = max(r[4] for r in _D11 if abs(abs(r[0] - 1.0) - x) < 1e-9)
            return y, (f'MEASURED row |m-1|={x:.2f} '
                       f'(D11 range {lo:.2e}..{hi:.2e})')
    return float(np.interp(d, xs, ys)), 'est., interpolated between measured rows'


def stage_d():
    print("STAGE D -- Sziklas-Siegman frame residual on the REAL 121 chain")
    print()
    print("D11 measured table (FIX_D4_D6_D7_2026_08_06.md, exact-ASM oracle "
          "of the FULL field):")
    print("     m     w(um)  NA_env  R(mm) | 1-leg exact  2-leg exact  "
          "fresnel   exact/fres")
    for m, w, na, R, e1, e2, fr, ra in _D11:
        print(f"   {m:5.3f}  {w:5.1f}  {na:.4f} {R:+6.2f} |   {e1:.3e}    "
              f"{e2:.3e}   {fr:.3e}     {ra:.3f}")
    print()

    rows = _ss_steps()

    print()
    print("EVERY SZIKLAS-SIEGMAN STEP THE REAL 121 CHAIN PERFORMS")
    print("  (recorded by spying on carrier._carrier_step_fast, the single")
    print("   site that applies the m = R_out/R_in co-moving rescale; the")
    print("   EXACT final leg does not appear because it does not use the SS")
    print("   frame at all -- it references the exact sphere and finishes with")
    print("   the band-limited ASM Bluestein zoom.)")
    print()
    print("  #  chain  R_in(mm)      z(mm)   R_out(mm)        m    |m-1|"
          "     resid  basis")
    tot = 0.0
    sq = 0.0
    n_counted = 0
    n_uncal = 0
    for i, (tag, R, z, Ro, m) in enumerate(rows):
        b, why = _ss_bound(m)
        last = (i == len(rows) - 1)
        if last:
            b, why = None, ('NOT TAKEN in production -- final_leg=exact '
                            'replaces this SS step with the exact-sphere + '
                            'ASM Bluestein readout')
        elif b is None:
            n_uncal += 1
        else:
            tot += b
            sq += b * b
            if abs(m - 1.0) > 1e-12:
                n_counted += 1
        print(f"  {i:<2d} {tag:<5s} {R * 1e3:10.4f} {z * 1e3:10.4f} "
              f"{Ro * 1e3:11.4f} {m:8.5f} {abs(m - 1):8.5f} "
              f"{'   --    ' if b is None else f'{b:9.2e}'}  {why}")
    print(f"  {'':2s} {'TOTAL':<5s} {'':10s} {'':10s} {'':11s} {'':8s} "
          f"{'':8s} {tot:9.2e}  linear sum over the {n_counted} CALIBRATED "
          f"scaled step(s) (coherent worst case)")
    print(f"  {'':2s} {'':5s} {'':10s} {'':10s} {'':11s} {'':8s} "
          f"{'':8s} {np.sqrt(sq):9.2e}  rss of the same (independent errors)")
    print(f"  {'':2s} {'':5s} {'':10s} {'':10s} {'':11s} {'':8s} "
          f"{'':8s} {_D11_MAX * n_counted:9.2e}  CONSERVATIVE ENVELOPE: "
          f"{n_counted} x the largest measured D11 row ({_D11_MAX:.2e})")
    if n_uncal:
        print()
        print(f"  !! {n_uncal} step(s) are NOT COVERED by any total above: "
              f"their |m-1| lies outside")
        print(f"     the D11 calibration and this capstone refuses to "
              f"extrapolate a six-point,")
        print(f"     non-monotone table past its span.  Closing it needs an "
              f"exact-ASM oracle run")
        print(f"     AT THAT GEOMETRY -- the same measurement D11 made, at "
              f"the missing m.")
    print()
    print("READING.  'resid' is the D11-measured relative L2 of the EXACT "
          "kernel")
    print("against an independent exact-ASM oracle of the FULL field, "
          "interpolated")
    print("in |m-1|.  It is the SS FRAME's own error, not the kernel's: the "
          "same")
    print("rows put the PARAXIAL kernel 2.0x-6.8x further from the same "
          "oracle.")
    print(f"STAGE_D_RESULT {json.dumps({'steps': [[t, R, z, Ro, m] for t, R, z, Ro, m in rows], 'total_linear': tot, 'total_rss': float(np.sqrt(sq)), 'n_calibrated': n_counted, 'n_uncalibrated': n_uncal})}")
    return 0


def _ss_steps():
    """Run the real 121 chain (A then B, on-axis) with a spy on the single
    Sziklas-Siegman step site, and return every (R_in, z, R_out, m)."""
    steps = []
    _orig = C._carrier_step_fast

    def _spy(E_env, R, z, wavelength, dx, dy, gap_kernel='auto',
             tilt=(0.0, 0.0)):
        steps.append((float(R), float(z)))
        return _orig(E_env, R, z, wavelength, dx, dy, gap_kernel=gap_kernel,
                     tilt=tilt)

    pre, post, gap_to_doe, _per = D.geometry()
    n, dx0 = 512, 4.0e-6
    zR = np.pi * D.W0 * D.W0 / D.LAM
    w_z1 = D.W0 * np.sqrt(1 + (D.Z1 / zR) ** 2)
    R1 = D.Z1 * (1 + (zR / D.Z1) ** 2)
    x = (np.arange(n) - n // 2) * dx0
    e0 = np.exp(-(x[None, :] ** 2 + x[:, None] ** 2)
                / w_z1 ** 2).astype(np.complex128)
    C._carrier_step_fast = _spy
    try:
        resA = la.propagate_traced_carrier_chain(
            e0, pre, D.LAM, dx0, r_in=R1, ray_subsample=2, n_workers=8,
            final_distance=gap_to_doe)
        nA = len(steps)
        from lumenairy.propagators.carrier import carrier_referenced_envelope
        envA = carrier_referenced_envelope(resA.field, resA.R, D.LAM, resA.dx)
        # NO focus_readout here, deliberately.  The magnification of every
        # SS step is a property of the carrier and the gap, not of how the
        # target plane is read, so this enumerates the same legs at a
        # fraction of the cost.  The consequence is that the FINAL leg
        # appears as an SS step -- in production it is NOT one: with
        # final_leg='exact' (what 121's na_exit=0.405 selects) that leg is
        # replaced by the fine retrace + exact-sphere + band-limited ASM
        # Bluestein zoom, which never enters the co-moving frame.  The last
        # row is flagged and excluded from the totals for exactly that
        # reason.
        la.propagate_traced_carrier_chain(
            envA, post, D.LAM, resA.dx, r_in=resA.R, ray_subsample=2,
            n_workers=1, final_distance=D.TRAILING)
    finally:
        C._carrier_step_fast = _orig
    rows = []
    for i, (R, z) in enumerate(steps):
        rows.append(('A' if i < nA else 'B', R, z, R + z, (R + z) / R))
    return rows


# ===========================================================================
# final_leg INERTNESS on chain A -- the empirical form of pre-flight (d)
# ===========================================================================
def leg_inert():
    """Prove BY BYTE IDENTITY that ``final_leg`` is inert on a chain-A call.

    Pre-flight (d) argues this from the code: the chain reads ``final_leg``
    only inside ``if is_final and focus_readout is not None``
    (carrier.py:7516 / :7785), and chain A passes no ``focus_readout``.  This
    runs both spellings and compares the fields, because an argument that
    reads exactly like the D6 defect deserves better than a code reading.

    Note the two calls CANNOT share a cache: ``final_leg`` is part of the
    chain-A cache key, so each builds its own file.  That is the point -- the
    key is deliberately conservative about an argument that is provably inert
    here, and this measures the cost of that caution (one extra cold build)
    against its benefit (D6 cannot recur)."""
    n = int(os.environ.get('RN', '1024'))
    rs = int(os.environ.get('RS', '4'))
    print("final_leg INERTNESS on chain A (no focus_readout) -- "
          f"N={n} rs={rs}")
    out = {}
    for leg in ('exact', 'paraxial'):
        t = time.perf_counter()
        env, R, dx, P = D.chain_a(n=n, rs=rs, final_leg=leg)
        dt = time.perf_counter() - t
        _k, dig = D._chain_a_key(n, 1.0e-6 * 2048 / n, rs, 8, leg)
        out[leg] = (env, R, dx, P)
        print(f"  final_leg={leg!r:10s} {dt:7.2f} s  cache "
              f"_chainA_v{D._CHAIN_A_SCHEMA}_n{n}_rs{rs}_{dig[:12]}.npz  "
              f"R={R!r} dx={dx!r}")
    a, b = out['exact'], out['paraxial']
    same = (np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]
            and a[3] == b[3])
    d = float(np.max(np.abs(a[0] - b[0]))) if not same else 0.0
    print(f"  fields byte-identical: {same}   max|diff| = {d:.3e}")
    print(f"  => fan_multi_121.py:218's final_leg='paraxial' is "
          f"{'INERT (dead spelling)' if same else 'LIVE -- INVESTIGATE'}")
    print(f"  NOTE the two builds still write DIFFERENT cache files: "
          f"final_leg is keyed even though it is inert here.")
    return 0 if same else 1


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'preflight'
    sys.exit({'preflight': preflight, 'stageA': stage_a,
              'stageD': stage_d, 'leginert': leg_inert}[mode]())
