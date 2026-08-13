"""Hammer-audit H3 (2026-07-19): traced exit-NA Nyquist guard.

``apply_real_lens_traced`` documents the critical-sampling rule
``dx <= lambda*f/aperture`` but never enforced it.  Violating it is
SILENT and insidious: the exit converging wavefront exceeds grid
Nyquist beyond a radius, the aliased annulus folds far-halo energy to
wrong positions, and r^2-weighted metrics (r2m) read LOW while
EE50/EE80 stay plausible.  Dual-oracle f/5 case (R=+/-51.68mm, t=5mm,
n=1.5168, lambda=1.31um, w0=5mm): traced r2m 40.9 vs oracle 65.0 um at
dx=6um (2.24x the limit), fully recovered to 64.77 (99.7%) at dx=3um.

The v5.25.0 guard computes the exit NA from the already-traced ray
direction cosines (amplitude-aware: rays with >= e^-4 of peak input
amplitude) and warns when dx > lambda/(2*NA_exit).  It deliberately
never raises (core metrics remain valid); ``on_undersample='silent'``
suppresses it.

RESOURCE NOTICES ARE NOT PHYSICS GUARDS (v5.32.3, FIX_CI_POOL)
--------------------------------------------------------------
The two suppression tests below assert that a policy leaves NO
``RuntimeWarning`` at all, which is the strong form and the one worth
keeping: it also catches a NEW guard that fires when it was told not to.
That form only survives if every warning this call can emit is routed
through a policy knob -- and when the Newton pool's memory clamp shipped,
its cap notice was not.  On CI's ~12 GB runners the clamp legitimately
answered a 4-worker request with 2 and said so, and a physics-guard
suppression test failed on a resource notice that had nothing to do with
H3 (all four python lanes, shard 1).  On the 128-256 GB dev boxes it
never fired, so the same test passed locally.

The remedy is the knob, not a weaker assertion: ``on_pool_memory='silent'``
below suppresses the cap notice through its OWN policy surface, exactly
as ``on_undersample`` and ``on_aperture_beam`` suppress theirs.  That
makes these tests deterministic across box RAM -- they pass on a 12 GB
runner and on a 256 GB workstation for the SAME reason, rather than
passing on one by luck.  ``test_the_pool_cap_notice_is_silenced_by_its_own
_knob`` pins that with the box's free memory frozen at 12 GB, so the CI
condition is reproduced here rather than waited for.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6

# Model glass for THIS module only: registered and removed by
# tests/conftest.py::_module_glass_registry_guard.
MODULE_GLASSES = {'_H3_FIX_GLASS': lambda wl: 1.5168}


def _singlet_f5():
    return {
        'wavelength': _WL,
        'aperture_diameter': 24e-3,
        'surfaces': [
            {'radius': 51.68e-3, 'thickness': 5e-3,
             'glass_before': 'air', 'glass_after': '_H3_FIX_GLASS',
             'semi_diameter': 12e-3},
            {'radius': -51.68e-3, 'thickness': 0.0,
             'glass_before': '_H3_FIX_GLASS', 'glass_after': 'air',
             'semi_diameter': 12e-3},
        ],
        'thicknesses': [5e-3],
        'stop_index': 0,
    }


def _gauss(N, dx, w0):
    x = (np.arange(N) - N / 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


def test_h3_guard_fires_on_undersampled_fast_beam():
    """f/5 beam (w0=5mm through the f~50mm singlet, NA_exit ~ 0.2) on a
    dx=12um grid violates dx <= lambda/(2*NA_exit) ~ 3um by 4x -> the
    guard must warn, naming NA_exit and the required dx."""
    E0 = _gauss(1024, 12e-6, 5e-3)
    with pytest.warns(RuntimeWarning, match='NA_exit'):
        la.apply_real_lens_traced(E0, prescription=_singlet_f5(),
                                  wavelength=_WL, dx=12e-6)


def test_h3_guard_silent_on_benign_slow_beam():
    """Benign w0=0.5mm beam through the same lens: significant rays only
    reach h ~ 1.4w0 = 0.7mm -> NA_exit ~ 0.014 -> dx_need ~ 47um >> the
    8um grid.  The guard must stay silent (amplitude-aware: the huge
    zero-energy aperture must NOT over-fire it)."""
    E0 = _gauss(1024, 8e-6, 0.5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # v5.29 (P2): this fixture is ALSO in the aperture:beam cliff regime
        # (24 mm aperture, 1 mm beam = 24x) and the new warn-only cliff flag
        # legitimately fires there.  This test is about the H3 NA_exit guard,
        # so silence the unrelated flag rather than weaken the 'error' filter.
        # ``on_pool_memory``: same reasoning, one layer down -- see the module
        # docstring.  Whether the Newton pool's memory clamp binds is a fact
        # about the RUNNER, not about this lens.
        la.apply_real_lens_traced(E0, prescription=_singlet_f5(),
                                  wavelength=_WL, dx=8e-6,
                                  on_aperture_beam='silent',
                                  on_pool_memory='silent')


def test_h3_guard_suppressed_by_silent_policy():
    E0 = _gauss(1024, 12e-6, 5e-3)
    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        # ``on_aperture_beam='silent'``: see the note in
        # test_h3_guard_silent_on_benign_slow_beam (this fixture's 24 mm
        # aperture is 2.5x the 5 mm beam, so the v5.29 cliff flag also fires).
        # ``on_pool_memory='silent'``: the Newton pool's memory clamp is a
        # RESOURCE notice whose firing depends on the box's free RAM (measured:
        # it fires on CI's 12 GB runners and never on the 128 GB dev box), so a
        # blanket zero-warning assertion is only deterministic once it is
        # routed through its own knob.  See the module docstring.
        la.apply_real_lens_traced(E0, prescription=_singlet_f5(),
                                  wavelength=_WL, dx=12e-6,
                                  on_undersample='silent',
                                  on_aperture_beam='silent',
                                  on_pool_memory='silent')


def _pin_available_ram(monkeypatch, free_b):
    """Freeze ``psutil.virtual_memory().available`` (and the library RAM
    budget the clamp mins against) so the Newton pool's memory cap decision is
    arithmetic on a pinned box rather than a race with this one.

    Pinned-snapshot idiom, as in ``test_fix_newton_pool_memory.py`` /
    ``test_fga_h4_h5.py::test_c2_env_budget_override``.
    """
    import psutil

    from lumenairy import memory as _mem
    vm = psutil.virtual_memory()
    monkeypatch.setattr(psutil, 'virtual_memory',
                        lambda: vm._replace(available=int(free_b)))
    monkeypatch.setattr(_mem, 'get_ram_budget', lambda: float(free_b))


def test_the_pool_cap_notice_is_silenced_by_its_own_knob(monkeypatch):
    """FAIL-BEFORE for the CI break, reproduced on THIS box.

    CI's ubuntu runners have ~12 GB available, where a 4-worker Newton
    dispatch against this fixture's ray-fit grid does not fit the budget and
    the clamp correctly runs 2 workers instead -- emitting a ``RuntimeWarning``
    that broke ``test_h3_guard_suppressed_by_silent_policy`` on all four python
    lanes.  Before the ``on_pool_memory`` knob there was no way to suppress it
    from the call, so the suppression test's contract held only on a big box.

    Both halves are asserted, because the knob is only worth having if the
    notice it silences would otherwise fire:

      1. at the pinned 12 GB the DEFAULT policy still announces the cap (if
         this stops holding, the emulation has gone vacuous and half 2 proves
         nothing);
      2. ``on_pool_memory='silent'`` leaves the very same call warning-free.

    The clamp itself is unchanged in both halves -- it is a resource decision
    on a path documented and tested to be bit-identical to serial, so the knob
    moves the REPORT and never the numbers.

    2026-08-13 -- BOTH HALVES ARE SCORED WITH ``inverse_map=False``, AND THE
    RE-DERIVATION THE ASSERTION MESSAGE BELOW ASKS FOR WAS TRIED FIRST AND
    CANNOT WORK.  The clamp prices a PER-PIXEL NEWTON INVERSION.  With the
    inverse-characteristic model engaged (``TRACED_INVERSE_MAP``, shipped
    ``True`` since ``FIX_G8_PROBE_2026_08_12``) there is no per-pixel Newton
    to dispatch -- the model evaluates the exit polynomial directly -- so the
    pool decision is never taken and ``_newton_worker_bytes`` is never called.
    Measured on this fixture over a pinned-RAM ladder 12 / 8 / 6 / 4 / 3 / 2 /
    1 GB:

        arm                clamp priced a worker?   cap warnings
        inverse_map=False  yes, every rung          1 at every rung
                           (1.871 GB per worker, 4096 Newton points/chunk,
                            140625-point ray-fit grid)
        SHIPPED default    NEVER, at any rung       0 at every rung

    So no value of the pinned box can make the notice fire on the shipped
    path, and lowering the pin would only make the FORWARD arm's emulation of
    CI less faithful.  The 12 GB pin is kept exactly as CI's runners have it,
    on the arm where the pool exists.  The shipped path is not left unstated:
    the third block below ASSERTS that it prices no worker at all, which is
    both the reason for the scoping and a real contract -- if a future build
    reintroduces a per-pixel Newton dispatch behind the model, this fails.
    """
    _pin_available_ram(monkeypatch, 12.0e9)
    LT.close_worker_pool()
    E0 = _gauss(1024, 12e-6, 5e-3)
    kw = dict(prescription=_singlet_f5(), wavelength=_WL, dx=12e-6,
              n_workers=4, inverse_map=False,
              on_undersample='silent', on_aperture_beam='silent')

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        la.apply_real_lens_traced(E0, **kw)
    caps = [w for w in rec
            if issubclass(w.category, RuntimeWarning)
            and 'Newton process pool asked for' in str(w.message)]
    assert caps, (
        'the pinned 12 GB box did not make the Newton pool memory clamp bind, '
        'so this test cannot show that the knob silences anything.  Re-derive '
        'the pin from _newton_worker_bytes rather than deleting the assertion')

    with warnings.catch_warnings():
        warnings.simplefilter('error', RuntimeWarning)
        la.apply_real_lens_traced(E0, on_pool_memory='silent', **kw)
    LT.close_worker_pool()

    # ... and on the SHIPPED path the same call at the same pinned 12 GB never
    # reaches the pool decision at all, because the model replaces the
    # per-pixel Newton the clamp exists to price.
    priced = []
    _orig_wb = LT._newton_worker_bytes

    def _spy(chunk_points, fit_points):
        priced.append((float(chunk_points), float(fit_points)))
        return _orig_wb(chunk_points, fit_points)

    monkeypatch.setattr(LT, '_newton_worker_bytes', _spy)
    kw_map = {k: v for k, v in kw.items() if k != 'inverse_map'}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        la.apply_real_lens_traced(E0, **kw_map)
    assert not priced, (
        'the shipped path priced a Newton pool worker, so it now HAS a '
        'per-pixel Newton dispatch and the two halves above should be scored '
        'on it rather than on inverse_map=False', priced)
    assert not [w for w in rec
                if issubclass(w.category, RuntimeWarning)
                and 'Newton process pool asked for' in str(w.message)]
    LT.close_worker_pool()


def test_the_cap_knob_refuses_junk_at_entry():
    """House rule (finding V4 / D5): a string mode knob on this signature
    refuses an unknown value at ENTRY, rather than falling through to whichever
    branch the equality test happens to miss.  Gated at entry specifically --
    the cap only binds on a small box, so a gate inside the warning branch
    would validate this knob on CI and not on a workstation."""
    with pytest.raises(ValueError, match='on_pool_memory'):
        la.apply_real_lens_traced(_gauss(64, 12e-6, 5e-3),
                                  prescription=_singlet_f5(),
                                  wavelength=_WL, dx=12e-6,
                                  on_pool_memory='zzz_not_a_policy')
    # ...and the carrier-house spelling of suppression is honoured, not
    # accepted-and-inert (the collision on_fit_domain_basis resolves the same
    # way): 'ignore' / 'off' mean 'silent'.
    for alias in ('ignore', 'off', 'silent'):
        assert LT._pool_memory_policy(alias) == 'silent'
    assert LT._pool_memory_policy('warn') == 'warn'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
