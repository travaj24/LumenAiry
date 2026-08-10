"""P2 DAILY-DRIVER GUARDS (audit
``docs/audits/AUDIT_TRACED_PRODUCTION_READINESS_2026_07_24.md`` §4/§5-P2 and §6:
"no silent cliffs; guards fire, memory degrades gracefully, non-converged
results are flagged").

Three mechanisms, one file:

1. **Memory-bounded fine grid** -- the exact focus readout's internal
   resolution used to be sized from the physics alone and could demand a single
   32768^2 complex128 array (16 GiB), killing the process mid-propagation.  It
   is now capped to the RAM budget with a ``RuntimeWarning`` that names the
   un-degraded requirement, so a memory-limited number is never returned as if
   it were the requested one.
2. **dx convergence self-check** (``self_check='dx'``) -- opt-in second run at
   ``dx/sqrt(2)``, warning when the focal metrics are not dx-stable.
3. The aperture:beam cliff guard's own behaviour is pinned in
   ``test_niche_e4_corrected_relay_oracle.py`` (cliff + recovery + no
   vignetting) and ``test_niche_p2_design_battery.py`` (across the envelope);
   here we only pin its ARGUMENT VALIDATION and the escape hatches.
"""
from __future__ import annotations

import contextlib
import logging
import re
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.carrier import (
    _FINE_GRID_MIN,
    _FINE_GRID_WORK_ARRAYS,
    _memory_bounded_n_fine,
)

_WL = 1.31e-6

# ---------------------------------------------------------------------------
# The cap ladder is ARITHMETIC ON ``_FINE_GRID_WORK_ARRAYS``, so it moves every
# time that constant is re-measured:
#
#     cap(budget) = 2 ** floor(log2(floor(sqrt(
#                       _FINE_GRID_RAM_FRAC * budget
#                       / (_FINE_GRID_WORK_ARRAYS * 16)))))
#
# It has gone stale once already, silently: v5.33.2 re-measured the constant
# from 4 to 16 complex128-equivalents (docs/audits/
# FIX_PERF_CACHES_BLUESTEIN_2026_08_09.md sec 2.1) and left the ladder below
# at its 4-array values, so this file was RED on the branch that shipped it
# and said nothing about the guard it exists to pin.  The count is asserted
# FIRST now, with the formula in the message, so the next re-measurement fails
# with instructions instead of an unexplained 512 != 1024.
_LADDER_WORK_ARRAYS = 20
_LADDER = ((0.25, 512), (1, 1024), (4, 2048), (16, 4096), (34, 4096),
           (136, 8192))
_LADDER_WHY = (
    "the cap ladder in this file was hand-computed at "
    "_FINE_GRID_WORK_ARRAYS = %d and the shipped constant is now %d.  "
    "Re-derive it: cap = 2**floor(log2(floor(sqrt(0.5 * budget / "
    "(n_work * 16))))), and update _LADDER_WORK_ARRAYS with it."
    % (_LADDER_WORK_ARRAYS, _FINE_GRID_WORK_ARRAYS))


def _singlet(R1, R2, d, glass, ap, name='s'):
    return {'name': name, 'aperture_diameter': ap, 'thicknesses': [d],
            'surfaces': [
                {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
                {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
                 'conic': 0.0, 'radius_y': None, 'conic_y': None,
                 'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]}


def _converging_field(N=1024, dx=2.0e-6, w=120e-6, R=-260e-6):
    """A strongly-converging (NA ~ 0.46) exact-sphere-referenced full field --
    the regime the exact readout exists for."""
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    S = -(np.sqrt(X ** 2 + Y ** 2 + R * R) - abs(R))
    return env * np.exp(1j * (2 * np.pi / _WL) * S), dx, R


# ===========================================================================
# 1. Memory budget arithmetic.
# ===========================================================================
def test_memory_cap_is_monotone_power_of_two_and_floored():
    """The cap grows with the budget, stays a power of two, never exceeds the
    request, and never degrades below the floor.  The ladder is hand-computed
    at the SHIPPED work-array count (see ``_LADDER`` above for the formula and
    for why it is asserted rather than assumed)."""
    assert _FINE_GRID_WORK_ARRAYS == _LADDER_WORK_ARRAYS, _LADDER_WHY
    prev = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for gb, expect in _LADDER:
            n = _memory_bounded_n_fine(1 << 20, 'probe',
                                       ram_budget=gb * 1024 ** 3)
            assert n == expect, (gb, n, expect)
            assert n >= prev
            prev = n
        # request below the cap is returned untouched, and never below floor
        assert _memory_bounded_n_fine(256, 'probe',
                                      ram_budget=136 * 1024 ** 3) == 256
        assert _memory_bounded_n_fine(1 << 20, 'probe', ram_budget=1) == \
            _FINE_GRID_MIN
        # inf disables the cap entirely
        assert _memory_bounded_n_fine(1 << 20, 'probe',
                                      ram_budget=float('inf')) == 1 << 20


def test_memory_cap_warning_names_the_undegraded_requirement():
    """When the cap binds the warning must state (a) that the grid is
    memory-limited, (b) the capped size, (c) the un-degraded requirement, and
    (d) that the result is resolution-limited / non-converged -- the whole
    point being that the degraded number is never silent."""
    assert _FINE_GRID_WORK_ARRAYS == _LADDER_WORK_ARRAYS, _LADDER_WHY
    with pytest.warns(RuntimeWarning) as rec:
        n = _memory_bounded_n_fine(8192, 'probe', ram_budget=64 * 1024 ** 2)
    assert n == 256
    msg = ' '.join(str(rec[0].message).split())
    for frag in ('MEMORY-LIMITED', '256x256', '8192x8192',
                 'RESOLUTION-LIMITED', 'set_max_ram'):
        assert frag in msg, (frag, msg)


# ===========================================================================
# 2. The exact readout honours the budget (kwarg, set_max_ram, and inf).
# ===========================================================================
def test_exact_readout_degrades_gracefully_under_a_small_budget():
    """Same call, three budgets: generous -> full resolution and full power;
    tiny -> a WARNED, coarser readout that visibly loses power (the honest
    signature of a resolution-limited result).  Measured 2026-07-25 (NA 0.46,
    window_factor=7): EE within 1 um 85.3% and P/P_in 1.000 un-degraded vs
    59.7% / 0.887 at a 64 MB budget."""
    E, dx, R = _converging_field()
    P_in = float((np.abs(E) ** 2).sum()) * dx * dx
    out = {}
    for tag, budget in (('full', float('inf')), ('tight', 64 * 1024 ** 2)):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            f = la.carrier_referenced_exact_focus_readout(
                E, R, -R, _WL, dx, dx_out=0.05e-6, N_out=512,
                window_factor=7.0, ram_budget=budget)
        capped = [w for w in wl if 'MEMORY-LIMITED' in str(w.message)]
        I = np.abs(f) ** 2
        out[tag] = (float(I.sum()) * 0.05e-6 ** 2 / P_in, float(I.max()),
                    len(capped))
    assert out['full'][2] == 0, 'a generous budget must not warn'
    assert out['tight'][2] == 1, 'a tight budget must warn exactly once'
    assert out['full'][0] > 0.98, out['full']
    assert out['tight'][0] < out['full'][0] - 0.05, out
    assert out['tight'][1] < out['full'][1]


def test_exact_readout_honours_set_max_ram():
    """No ``ram_budget`` kwarg: the library-wide :func:`lumenairy.set_max_ram`
    budget must drive the cap (the machinery the audit asked us to respect)."""
    E, dx, R = _converging_field()
    la.set_max_ram(32 * 1024 ** 2)
    try:
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            f = la.carrier_referenced_exact_focus_readout(
                E, R, -R, _WL, dx, dx_out=0.05e-6, N_out=256,
                window_factor=7.0)
        assert [w for w in wl if 'MEMORY-LIMITED' in str(w.message)]
        assert f.shape == (256, 256)
    finally:
        la.set_max_ram(None)


def test_chain_threads_ram_budget_into_the_focus_readout():
    """``focus_readout={'ram_budget': ...}`` must reach BOTH memory-bounded
    grids inside the chain (the fine re-trace and the exact readout)."""
    N, dx, w0 = 512, 4e-6, 0.6e-3
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    groups = [{'prescription': _singlet(6e-3, -6e-3, 3e-3, 'N-BK7', 3e-3),
               'gap_before': 0.0}]
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, r_in=np.inf, ray_subsample=4, n_workers=1,
            traced_kwargs=dict(parallel_amp=False, on_undersample='silent'),
            final_distance=5.4e-3, final_leg='exact',
            focus_readout=dict(dx_out=0.1e-6, N_out=256,
                               ram_budget=16 * 1024 ** 2))
    assert res.field.shape == (256, 256)
    capped = [w for w in wl if 'MEMORY-LIMITED' in str(w.message)]
    assert capped, [str(w.message)[:80] for w in wl]


# ===========================================================================
# 3. dx convergence self-check.
# ===========================================================================
def _chain_kw(**over):
    kw = dict(r_in=np.inf, ray_subsample=4, n_workers=1,
              traced_kwargs=dict(parallel_amp=False, on_undersample='silent'),
              final_distance=118.3e-3,
              focus_readout=dict(dx_out=0.2e-6, N_out=256))
    kw.update(over)
    return kw


def _slow_singlet_chain(N=512, dx=6e-6, w0=0.9e-3, **over):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    env0 = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    groups = [{'prescription': _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 4e-3),
               'gap_before': 0.0}]
    return env0, groups, dx, over


# ---------------------------------------------------------------------------
# Margin diagnostics for the two tests that DEMAND the 'NOT dx-STABLE' warning.
#
# ``pytest.warns`` is a BINARY instrument: on a miss it reports DID NOT WARN
# plus the bystander warnings, and never the MARGIN -- which is the one number
# that separates "the guard stopped detecting" from "the guard fired and the
# capture lost it".  CI hit exactly that on 5af1edf and adjudicating it took two
# studies, because the failure carried no margin (see
# ``docs/audits/P2_DIDNOTWARN_DIAGNOSIS_2026_08_03.md``, which excludes the
# python/BLAS/numpy, RAM-budget, thread-count, glass-source, warning-registry
# and dispatch-global axes by measurement, and leaves the margin as the only
# open discriminator).
#
# The guard already LOGS its own ``m1`` / ``m2`` at INFO before deciding.
# Capture that and attach it to the failure -- the same self-reporting pattern
# niche D4 adopted -- so the NEXT occurrence names its own cause instead of
# starting another argument.
# ---------------------------------------------------------------------------
_METRIC_RE = re.compile(r"'(\w+)':\s*(?:np\.float64\()?(-?[0-9.eE+-]+)")


@contextlib.contextmanager
def _dx_selfcheck_margin():
    """Collect ``_run_chain_dx_self_check``'s own INFO line (which carries the
    two metric dicts it compares)."""
    lines: list[str] = []

    class _Sink(logging.Handler):
        def emit(self, record):
            try:
                msg = record.getMessage()
            except Exception:                          # pragma: no cover
                return
            if "self_check='dx'" in msg:
                lines.append(msg)

    log = logging.getLogger('lumenairy.propagators.carrier')
    sink, prev = _Sink(), log.level
    log.addHandler(sink)
    log.setLevel(logging.INFO)
    try:
        yield lines
    finally:
        log.removeHandler(sink)
        log.setLevel(prev)


def _margin_report(lines, tol, state_dump=None):
    """Render the guard's OWN comparison as a per-metric margin table."""
    if not lines:
        return ('DIAGNOSIS: the guard logged NO self-check line -- it returned '
                'before comparing anything (empty primary metrics), so the '
                'chain never reached the warn.  That is a guard-side silent '
                'pass, not a capture failure.')
    m = re.search(r'metrics (\{.*\}) vs (\{.*\})\s*$', lines[-1])
    if m is None:                                      # pragma: no cover
        return 'guard log line (unparsed): ' + lines[-1]
    a, b = dict(_METRIC_RE.findall(m.group(1))), dict(_METRIC_RE.findall(m.group(2)))
    keys = sorted(set(a) & set(b))
    rows = ["guard self_check='dx' margin (its OWN m1 vs m2, tol %.4g%%):"
            % (100.0 * tol)]
    if not keys:
        rows.append('  NO SHARED METRIC KEYS -- m1 %r vs m2 %r.  The guard '
                    'compares nothing and returns SILENTLY (a refined run that '
                    'degenerated reads as dx-stable).' % (sorted(a), sorted(b)))
    for k in keys:
        x, y = float(a[k]), float(b[k])
        rel = abs(x - y) / max(abs(x), abs(y), 1e-300)
        rows.append('  %-6s %.6g vs %.6g -> %9.4f %%  (%.2fx tol)  %s'
                    % (k, x, y, 100.0 * rel, (rel / tol) if tol else float('inf'),
                       'TRIPS' if rel > tol else 'inside'))
    rows.append('  If every metric TRIPS the guard DID fire and the capture '
                'lost it (look for a thread leaving warnings.catch_warnings '
                'mid-block); if none does, the physics moved.')
    rows.append('  full guard log: ' + lines[-1])
    if state_dump is not None:
        try:
            rows.append(state_dump())
        except Exception as exc:                       # pragma: no cover
            rows.append('(process-state dump failed: %r)' % (exc,))
    return '\n'.join(rows)


def _expect_dx_warning(call, tol, state_dump):
    """``pytest.warns(... 'NOT dx-STABLE')`` with the margin attached to a
    miss.  Keeps the contract identical; only the failure text grows.

    Only pytest's own ``Failed`` (the DID-NOT-WARN outcome) is decorated -- a
    genuine exception from the chain propagates untouched, with its traceback,
    because converting it here would cost more debuggability than the margin
    buys."""
    with _dx_selfcheck_margin() as lines:
        try:
            with pytest.warns(RuntimeWarning, match='NOT dx-STABLE'):
                call()
        except pytest.fail.Exception as exc:
            raise AssertionError(
                '%s\n\n%s' % (exc, _margin_report(lines, tol, state_dump))
            ) from None


def test_self_check_rejects_unknown_modes():
    env0, groups, dx, _ = _slow_singlet_chain(N=64, dx=6e-6)
    with pytest.raises(ValueError, match='self_check'):
        la.propagate_traced_carrier_chain(env0, groups, _WL, dx,
                                          self_check='sqrt2')


def test_self_check_off_is_the_default_and_silent():
    """No self-check unless asked (it costs ~3x), and 'off'/None behave
    identically to omitting it.

    Compared with ``allclose(rtol=1e-12)``, not bit-equality: the chain is
    reproducible only to the FP floor run-to-run (measured 1.2e-15 relative on
    two identical calls -- threaded FFT / parallel reductions), independent of
    this kwarg."""
    env0, groups, dx, _ = _slow_singlet_chain()
    outs = []
    for sc in (None, 'off'):
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            res = la.propagate_traced_carrier_chain(
                env0, groups, _WL, dx, self_check=sc, **_chain_kw())
        assert not [w for w in wl if 'self_check' in str(w.message)]
        outs.append(np.asarray(res.field))
    assert np.allclose(outs[0], outs[1], rtol=1e-12, atol=0.0)


def test_self_check_dx_passes_on_a_dx_stable_chain():
    """A well-sampled chain is dx-stable and must NOT warn: refining to
    N*sqrt(2) moves the readout power / peak / r50 by <0.1% (measured
    2026-07-25: 0.06% peak, 0.06% power, 0.000% r50) -- and the returned field
    is exactly the primary (coarse-grid) result, unchanged by the check."""
    env0, groups, dx, _ = _slow_singlet_chain()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        res_plain = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, **_chain_kw())
        res_check = la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, self_check='dx', **_chain_kw())
    assert not [w for w in wl if 'self_check' in str(w.message)], \
        [str(w.message)[:120] for w in wl if 'self_check' in str(w.message)]
    assert np.allclose(np.asarray(res_plain.field),
                       np.asarray(res_check.field), rtol=1e-12, atol=0.0)


def test_self_check_dx_flags_a_non_convergent_chain(process_state_dump):
    """The point of the flag: a chain whose INPUT CARRIER is beyond the grid
    Nyquist (a steep diverging conjugate on a coarse grid) is genuinely not
    dx-stable -- refining moves the readout power by ~50% -- and the self-check
    says so instead of returning the number silently.  Measured 2026-07-25 at
    N=768 / dx=4 um / r_in=+3 mm: power 52.5%, peak 50.0% and r50 3.6% apart
    between dx and dx/sqrt(2); re-measured 2026-08-03 as 52.4574 / 50.0264 /
    3.5842 % on Windows/MKL py3.14 and Linux/OpenBLAS py3.13 alike.

    ``final_leg`` is PINNED, and that is load-bearing (see
    ``docs/audits/P2_DIDNOTWARN_DIAGNOSIS_2026_08_03.md`` S3).  Under the
    ``'auto'`` default this fixture's measured exit NA is **0.14870 against
    ``na_exact_threshold`` = 0.15** -- 0.87 % below a routing cliff the library
    itself warns about ("final_leg='auto' flips between the exact and the
    PARAXIAL focus readout at that threshold with no other symptom ... a
    beam-size change of 0.9 % would flip it").  The 52.5 % headline is the
    PARAXIAL branch's number.  On the EXACT branch the same fixture reads
    power 8.884 %, peak 4.357 %, r50 0.421 % -- **1.78x** tolerance with the
    peak metric ALREADY inside it, against 10.49x here.  Both branches still
    fire on both builds, so nothing is being papered over: the pin removes a
    coin toss from a test that is not about routing, and it is exactly the
    remedy the guard's own message recommends.  Pinning also drops the
    NA-proximity bystander warning from the recorder."""
    env0, groups, dx, _ = _slow_singlet_chain(N=768, dx=4e-6)
    _expect_dx_warning(
        lambda: la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, self_check='dx', final_leg='paraxial',
            **_chain_kw(r_in=3e-3)),
        0.05, process_state_dump)


def test_self_check_tolerance_is_honoured(process_state_dump):
    """``self_check_tol`` sets the flag threshold: the dx-stable chain above
    (0.06% drift) trips a 0.01% tolerance and passes a 5% one.

    Re-measured 2026-08-03 on both builds: peak 0.1052 %, power 0.1019 %,
    r50 0.0008 % -- two of the three metrics trip at 10.5x and ``r50`` does
    not.  Unlike its sibling this fixture is clear of the ``final_leg='auto'``
    cliff (it emits no NA-proximity warning at all), so it needs no routing
    pin."""
    env0, groups, dx, _ = _slow_singlet_chain()
    _expect_dx_warning(
        lambda: la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, self_check='dx', self_check_tol=1e-4,
            **_chain_kw()),
        1e-4, process_state_dump)


# ===========================================================================
# 4. Cliff-guard argument validation / escape hatches.
# ===========================================================================
def test_fit_radius_beam_factor_validation():
    E = np.ones((64, 64), dtype=np.complex128)
    presc = _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 2e-3)
    common = dict(prescription=presc, wavelength=_WL, dx=4e-6, ray_subsample=4,
                  n_workers=1, parallel_amp=False, on_undersample='silent',
                  on_noncollimated='off')
    with pytest.raises(ValueError, match='fit_radius_beam_factor'):
        la.apply_real_lens_traced(E, fit_radius_beam_factor=0.0, **common)
    with pytest.raises(ValueError, match='fit_radius_beam_factor'):
        la.apply_real_lens_traced(E, fit_radius_beam_factor=-1.0, **common)
    with pytest.raises(ValueError, match='on_aperture_beam'):
        la.apply_real_lens_traced(E, on_aperture_beam='shout', **common)


def test_fit_radius_too_small_falls_back_and_says_so():
    """A beam-relative disc too small to constrain the order-6 fit must ABANDON
    the restriction (rather than fit noise) and warn that the guard is not
    active on that call."""
    N, dx, w0 = 256, 4e-6, 60e-6
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    E = np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)
    presc = _singlet(60e-3, -60e-3, 4e-3, 'N-BK7', 0.8e-3)
    with pytest.warns(RuntimeWarning, match='ABANDONED'):
        la.apply_real_lens_traced(
            E, prescription=presc, wavelength=_WL, dx=dx, ray_subsample=8,
            n_workers=1, parallel_amp=False, on_undersample='silent',
            on_noncollimated='off', fit_radius_beam_factor=0.5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
