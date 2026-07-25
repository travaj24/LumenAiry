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

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators.carrier import (_FINE_GRID_MIN,
                                           _memory_bounded_n_fine)

_WL = 1.31e-6


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
    request, and never degrades below the floor.  Measured 2026-07-25 at 4
    complex128 working arrays: 0.25 GB -> 1024, 1 GB -> 2048, 4 GB -> 4096,
    16 GB -> 8192, 34 GB -> 16384 (the audit box, where the physics wanted
    32768 = 64 GiB peak and the process died), 136 GB -> 32768."""
    prev = 0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for gb, expect in ((0.25, 1024), (1, 2048), (4, 4096), (16, 8192),
                           (34, 16384), (136, 32768)):
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
    with pytest.warns(RuntimeWarning) as rec:
        n = _memory_bounded_n_fine(8192, 'probe', ram_budget=64 * 1024 ** 2)
    assert n == 512
    msg = ' '.join(str(rec[0].message).split())
    for frag in ('MEMORY-LIMITED', '512x512', '8192x8192',
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


def test_self_check_dx_flags_a_non_convergent_chain():
    """The point of the flag: a chain whose INPUT CARRIER is beyond the grid
    Nyquist (a steep diverging conjugate on a coarse grid) is genuinely not
    dx-stable -- refining moves the readout power by ~50% -- and the self-check
    says so instead of returning the number silently.  Measured 2026-07-25 at
    N=768 / dx=4 um / r_in=+3 mm: power 52.5%, peak 50.0% and r50 3.6% apart
    between dx and dx/sqrt(2)."""
    env0, groups, dx, _ = _slow_singlet_chain(N=768, dx=4e-6)
    with pytest.warns(RuntimeWarning, match='NOT dx-STABLE'):
        la.propagate_traced_carrier_chain(env0, groups, _WL, dx,
                                          self_check='dx',
                                          **_chain_kw(r_in=3e-3))


def test_self_check_tolerance_is_honoured():
    """``self_check_tol`` sets the flag threshold: the dx-stable chain above
    (0.06% drift) trips a 0.01% tolerance and passes a 5% one."""
    env0, groups, dx, _ = _slow_singlet_chain()
    with pytest.warns(RuntimeWarning, match='NOT dx-STABLE'):
        la.propagate_traced_carrier_chain(
            env0, groups, _WL, dx, self_check='dx', self_check_tol=1e-4,
            **_chain_kw())


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
