"""W9 dispatcher audit -- pins for the three measured auto-routing defects.

Audit: "is the auto-dispatcher properly updated for the traced and other
propagator upgrades?" (2026-07-27).  Every pin here FAILED before the fix in
the same change; each names the measurement that produced it.

The three defects share one root cause: :mod:`lumenairy.propagators.dispatch`
carries **two** auto-selectors -- :func:`_auto_select_method` (behind
``propagate(method='auto')``) and :func:`_select_asm_variant` (behind
:func:`which_propagator` / :func:`asm_propagate`) -- and the 4.12 audit
round-4 hardening (B1-6 "never route the user into a hard-raise from a kernel
they did not pick by name", B1-8 "never silently drop the caller's
output-grid request") was applied to the first and not the second, nor to the
first's interaction with ``output_grid`` / ``output_dx``.
"""
import warnings

import numpy as np
import pytest

from lumenairy.propagators.dispatch import (
    _auto_select_method,
    _select_asm_variant,
    asm_propagate,
    propagate,
    which_propagator,
)
from lumenairy.propagators.result import PropagationResult

LAM = 633e-9


def _gauss(N, dx, w0):
    x = (np.arange(N) - N // 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / w0 ** 2).astype(np.complex128)


# ===========================================================================
# W9-1 -- method='auto' + output_dx/output_grid must not land on a kernel
#         that cannot honour the request.
# ===========================================================================
#
# MEASURED pre-fix (N=64, dx=2 um, lambda=633 nm, z=1e-3 -> N_F=6.47, Q=2.47):
#
#   propagate(E, z=1e-3, wavelength=..., dx=2e-6, output_dx=3e-6)
#     -> ValueError: propagate(method='sas', output_grid/output_dx=...):
#        SAS does not support arbitrary output-grid sampling.
#
# The caller never wrote 'sas'; the selector picked it from z alone and then
# the bare-grid router raised on the caller's own (supported, documented)
# output-grid request.  z=1e-4 (asm) and z=5 (fraunhofer) both worked -- both
# have an MFT promotion -- so the failure was a pure function of z.  This is
# the B1-6 rule applied to B1-8's feature: when an output grid is requested,
# the selector must choose from the kernels that can deliver one.

_AUTO_OUTPUT_GRID_GEOM = dict(N=64, dx=2.0e-6, w0=20e-6)


@pytest.mark.parametrize('z', [6e-4, 1e-3, 3e-3, 1e-2])
def test_auto_with_output_dx_never_routes_to_a_no_mft_kernel(z):
    """In the Q>1 band the selector used to choose 'sas', which has no
    output-grid path.  With an output grid requested it must choose a kernel
    that has one."""
    g = _AUTO_OUTPUT_GRID_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    # Sanity: this z really is in the band that selects 'sas' when no output
    # grid is asked for (that routing is unchanged).
    assert _auto_select_method(
        E, z=z, wavelength=LAM, dx=g['dx'], prescription=None) == 'sas'
    chosen = _auto_select_method(
        E, z=z, wavelength=LAM, dx=g['dx'], prescription=None,
        output_requested=True)
    assert chosen != 'sas', (
        f"W9-1: method='auto' with an output-grid request selected {chosen!r} "
        f"at z={z}, which has no MFT/output-grid path -- the dispatcher would "
        f"raise a ValueError naming a kernel the caller never chose.")


@pytest.mark.parametrize('z', [6e-4, 1e-3, 3e-3, 1e-2])
def test_auto_with_output_dx_runs_and_honours_the_requested_pitch(z):
    """End-to-end: the documented output-grid feature must work under
    ``method='auto'`` for every z, not only where the selector happens to
    land on a kernel with an MFT variant."""
    g = _AUTO_OUTPUT_GRID_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    dx_req = 3.0e-6
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, z=z, wavelength=LAM, dx=g['dx'],
                        output_dx=dx_req)
    assert isinstance(res, PropagationResult)
    assert res.field is not None and res.field.shape == (g['N'], g['N'])
    assert res.dx == pytest.approx(dx_req, rel=1e-12)


def test_auto_with_output_dx_agrees_with_the_exact_mft_oracle():
    """The re-routed call must be RIGHT, not merely non-raising: compare it
    against the exact ASM-MFT kernel on the same requested grid.

    Oracle: :func:`angular_spectrum_propagate_mft` is the exact band-limited
    ASM evaluated on an arbitrary output grid -- the propagator the SAS error
    message itself points callers at.  Budget: 1e-9 on the complex overlap
    deficit; MEASURED deficit 0.0 (same kernel reached through the router), so
    this has >1e6x headroom and is a wiring pin, not a numerics pin.
    """
    from lumenairy.propagators.mft import angular_spectrum_propagate_mft
    g = _AUTO_OUTPUT_GRID_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    z, dx_req = 1e-3, 3.0e-6
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, z=z, wavelength=LAM, dx=g['dx'], output_dx=dx_req)
        ref = angular_spectrum_propagate_mft(
            E, z, LAM, g['dx'], dx_req, g['N'])
    a = np.asarray(res.field).ravel()
    b = np.asarray(ref).ravel()
    deficit = 1.0 - abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert deficit < 1e-9, (
        f"W9-1: auto+output_dx must deliver the exact ASM-MFT answer; "
        f"overlap deficit {deficit:.3e}")


def test_explicit_sas_plus_output_dx_still_raises():
    """The re-route is for ``method='auto'`` ONLY.  A caller who NAMES 'sas'
    still gets the v4.12 B1-8 ValueError (pinned in
    ``test_audit_propagation.py::test_sas_with_output_dx_raises``); nothing
    about that path changes."""
    g = _AUTO_OUTPUT_GRID_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with pytest.raises(ValueError, match='SAS does not support'):
        propagate(E, z=1e-3, wavelength=LAM, dx=g['dx'], method='sas',
                  output_dx=3e-6)


def test_auto_routing_without_an_output_grid_is_unchanged():
    """Regression fence: the documented no-output-grid routing table must be
    bit-for-bit the same (asm / sas / fraunhofer by z)."""
    g = _AUTO_OUTPUT_GRID_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    table = {1e-5: 'asm', 1e-4: 'asm', 1e-3: 'sas', 1e-2: 'sas',
             0.1: 'fraunhofer', 5.0: 'fraunhofer', -1e-3: 'asm',
             0.0: 'asm', None: 'asm'}
    for z, want in table.items():
        got = _auto_select_method(E, z=z, wavelength=LAM, dx=g['dx'],
                                 prescription=None)
        assert got == want, f'z={z!r}: expected {want!r}, got {got!r}'


# ===========================================================================
# W9-2 -- which_propagator / asm_propagate must not route back-propagation
#         into a forward-only kernel.
# ===========================================================================
#
# MEASURED pre-fix (N=64, dx=2 um, lambda=633 nm; threshold L^2/(N*lam) =
# 4.0442e-4 m):
#
#   asm_propagate(E, z=-1.2133e-3, ...)  -> which_propagator says 'sas'
#     -> ValueError: scalable_angular_spectrum_propagate: z must be > 0
#   asm_propagate(E, z=-1.2133e-2, ...)  -> which_propagator says 'fraunhofer'
#     -> ValueError: fraunhofer_propagate: z must be > 0
#
# ``_auto_select_method`` has carried the guard since 4.12 (audit round-4
# B1-6, "restrict the regime check to the back-propagating methods"); its twin
# in the same module never got it, so the ASM-family advisor recommends -- and
# ``asm_propagate`` then runs -- a kernel that cannot accept the sign of z.

_BACKPROP_GEOM = dict(N=64, dx=2.0e-6, w0=20e-6)


def _threshold(N, dx):
    L = N * dx
    return (L * L) / (N * LAM)


@pytest.mark.parametrize('mult', [3.0, 30.0])
def test_which_propagator_never_recommends_a_forward_only_kernel_for_z_lt_0(mult):
    g = _BACKPROP_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    z = -mult * _threshold(g['N'], g['dx'])
    # Sanity: the same |z| forward really is in the sas / fraunhofer band.
    assert _select_asm_variant(E, -z, LAM, g['dx']) in ('sas', 'fraunhofer')
    m = _select_asm_variant(E, z, LAM, g['dx'])
    assert m not in ('sas', 'fresnel', 'fraunhofer', 'rs'), (
        f"W9-2: _select_asm_variant recommended the forward-only {m!r} for "
        f"z={z:.6g} (< 0); every ASM-family member accepts either sign.")
    assert which_propagator(E, z, LAM, g['dx'])['method'] == m


@pytest.mark.parametrize('mult', [3.0, 30.0])
def test_asm_propagate_back_propagates_instead_of_raising(mult):
    g = _BACKPROP_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    z = -mult * _threshold(g['N'], g['dx'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = asm_propagate(E, z, LAM, g['dx'])
    arr = np.asarray(out[0] if isinstance(out, tuple) else out)
    assert arr.shape == E.shape
    assert np.all(np.isfinite(arr))


@pytest.mark.parametrize('mult', [1.0, 3.0])
def test_asm_propagate_back_prop_round_trip_returns_the_input(mult):
    """Physics oracle for the re-route: the back-propagating leg must be a
    CORRECT ASM back-propagation, not merely non-raising.

    The forward leg is taken with the explicit, pitch-preserving
    :func:`angular_spectrum_propagate` (naming it keeps the oracle independent
    of whatever the forward selector chooses -- at these ``|z|`` it picks the
    pitch-CHANGING sas / fraunhofer, which cannot round-trip on one grid).  The
    back leg goes through :func:`asm_propagate`, i.e. through the selector under
    test.  ASM is unitary, so the round trip must return the input.

    ``mult`` stops at 3 on purpose: the round trip is only unitary while the
    propagated beam still FITS the window.  MEASURED deficit vs distance on this
    probe (with the edge-energy fraction that explains it): 1.5e-13 @ 1x
    (5e-9 edge), 1.8e-11 @ 3x (4e-7), 1.2e-10 @ 5x (4e-5), 3.6e-4 @ 10x (1e-2),
    1.2e-1 @ 30x (9e-2).  Past ~5x the loss is real grid truncation, not the
    router, so pinning there would pin the window rather than the fix; the
    NON-raising behaviour is pinned out to 30x by the test above.

    Budget 1e-6 on the complex-overlap deficit; MEASURED 1.8e-11 at the loosest
    pinned point -- ~5e4x headroom, deliberately generous for the CI Linux /
    BLAS envelope.
    """
    from lumenairy.propagators.propagation import angular_spectrum_propagate
    g = _BACKPROP_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    z = mult * _threshold(g['N'], g['dx'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fwd = np.asarray(angular_spectrum_propagate(E, z, LAM, g['dx']))
        back = asm_propagate(fwd, -z, LAM, g['dx'])
        back = np.asarray(back[0] if isinstance(back, tuple) else back)
    assert back.shape == E.shape
    a, b = back.ravel(), E.ravel()
    deficit = 1.0 - abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert deficit < 1e-6, f'ASM round-trip overlap deficit {deficit:.3e}'


def test_forward_asm_family_routing_is_unchanged():
    """Regression fence: for z > 0 the ASM-family selector's decision table
    must be untouched."""
    g = _BACKPROP_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    thr = _threshold(g['N'], g['dx'])
    assert _select_asm_variant(E, 0.5 * thr, LAM, g['dx']) == 'asm'
    assert _select_asm_variant(E, 3.0 * thr, LAM, g['dx']) == 'sas'
    assert _select_asm_variant(E, 30.0 * thr, LAM, g['dx']) == 'fraunhofer'
    assert _select_asm_variant(E, 3.0 * thr, LAM, g['dx'],
                              tilt_x=0.05) == 'asm_tilted'
    assert _select_asm_variant(E, 3.0 * thr, LAM, g['dx'],
                              output_dx=3e-6) == 'asm_mft'


# ===========================================================================
# W9-3 -- tilt + output_dx must not silently discard the tilt.
# ===========================================================================
#
# MEASURED pre-fix (N=64, dx=2 um, lambda=633 nm, z=5e-4):
#
#   asm_propagate(E, z, lam, dx, tilt_x=0.05, output_dx=3e-6, output_N=64)
#   asm_propagate(E, z, lam, dx, tilt_x=0.0,  output_dx=3e-6, output_N=64)
#     -> max|difference| = 0.0  (BIT-IDENTICAL)
#
# ``_select_asm_variant`` puts the output-grid branch ABOVE the tilt branch
# and ``angular_spectrum_propagate_mft`` has no ``tilt_x`` / ``tilt_y``
# parameter, so a 50 mrad carrier vanished with no diagnostic.  v5.30 gave the
# sibling case -- the legacy ``'propagate_tilted'`` element ignoring
# ``elem['method']`` -- a ``UserWarning`` rather than a raise (see
# ``propagate_through_system``); this pin holds the ASM-family advisor to the
# same standard.


def test_tilt_plus_output_dx_warns_that_the_tilt_is_dropped():
    g = _BACKPROP_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        which_propagator(E, 5e-4, LAM, g['dx'], tilt_x=0.05, output_dx=3e-6)
    msgs = [str(r.message) for r in rec
            if issubclass(r.category, UserWarning)]
    assert any('tilt' in m and 'asm_mft' in m for m in msgs), (
        'W9-3: an asm_mft route that drops a non-zero tilt must say so; '
        f'got {msgs!r}')


def test_asm_propagate_tilt_plus_output_dx_warns():
    g = _BACKPROP_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        asm_propagate(E, 5e-4, LAM, g['dx'], tilt_x=0.05, output_dx=3e-6,
                      output_N=g['N'])
    assert any('tilt' in str(r.message) for r in rec
               if issubclass(r.category, UserWarning)), (
        'W9-3: asm_propagate must surface the dropped tilt too')


def test_no_tilt_or_no_output_dx_stays_silent():
    """The diagnostic must fire only on the genuine collision."""
    g = _BACKPROP_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    for kw in ({'output_dx': 3e-6},
               {'tilt_x': 0.05},
               {'tilt_x': 0.0, 'output_dx': 3e-6},
               {'tilt_x': 0.05, 'output_dx': g['dx']}):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter('always')
            which_propagator(E, 5e-4, LAM, g['dx'], **kw)
        tilt_warns = [str(r.message) for r in rec
                      if issubclass(r.category, UserWarning)
                      and 'tilt' in str(r.message)]
        assert not tilt_warns, f'{kw!r} should be silent, got {tilt_warns!r}'


# ===========================================================================
# W9-4 -- maslov / asymptotic / mhs must not silently swallow an output-grid
#         request (and must never mislabel the pitch they did not apply).
# ===========================================================================
#
# MEASURED pre-fix (N=64, dx=40 um, lambda=1.31 um, make_singlet R=+-50 mm
# N-BK7 d=3 mm aperture 2 mm; ``maslov`` is what method='auto' picks for this
# prescription):
#
#   propagate(E, prescription=rx, output_dx=80e-6)
#     -> field BIT-IDENTICAL to the no-request call (still 40 um sampling)
#        but PropagationResult.dx == 8e-05          <-- wrong metadata
#   propagate(E, prescription=rx, output_grid=(96, 80e-6))
#     -> shape (64, 64), dx 4e-05: the whole request vanished, silently
#
# gbd / hf / hfpi all honour both forms (measured shape 64->96, dx 40->80 um),
# so the diagnostic names them.  This is the 4.12 B1-8 treatment ('never
# silently drop the caller's output-grid request') finally applied to the
# prescription members it skipped.


def _singlet():
    import lumenairy as la
    return la.make_singlet(R1=0.05, R2=-0.05, d=3e-3, glass='N-BK7',
                           aperture=2e-3)


_PRESCRIPTION_GEOM = dict(N=64, dx=40e-6, w0=5e-4, wl=1.31e-6)


@pytest.mark.parametrize('kw', [{'output_dx': 80e-6},
                                {'output_grid': (96, 80e-6)},
                                {'output_grid': {'N': 96, 'dx': 80e-6}}])
def test_maslov_with_an_output_grid_request_raises_instead_of_dropping_it(kw):
    g = _PRESCRIPTION_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with pytest.raises(ValueError) as info:
        propagate(E, wavelength=g['wl'], dx=g['dx'], prescription=_singlet(),
                  method='maslov', **kw)
    msg = str(info.value)
    assert 'maslov' in msg and 'gbd' in msg, (
        f'W9-4: the diagnostic must name the offending method and the members '
        f'that honour the request; got {msg!r}')


def test_auto_prescription_with_an_output_grid_request_is_not_silent():
    """``method='auto'`` on a plain prescription selects ``maslov``; the
    output-grid request must not vanish there either.  The message has to
    explain the ``auto`` connection, because the caller never wrote
    'maslov'."""
    g = _PRESCRIPTION_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with pytest.raises(ValueError) as info:
        propagate(E, wavelength=g['wl'], dx=g['dx'], prescription=_singlet(),
                  output_dx=80e-6)
    msg = str(info.value)
    assert "method='auto'" in msg, (
        f"W9-4: an auto-selected maslov must explain why the caller landed on "
        f"a method they did not name; got {msg!r}")


def test_prescription_without_an_output_grid_request_is_unchanged():
    """Regression fence: the ordinary prescription call must be untouched."""
    g = _PRESCRIPTION_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, wavelength=g['wl'], dx=g['dx'],
                        prescription=_singlet())
    assert res.method == 'maslov'
    assert res.field.shape == (g['N'], g['N'])
    assert res.dx == pytest.approx(g['dx'], rel=1e-12)


# ===========================================================================
# W9-5 -- output_grid=(N_out, dx_out) must be reported on the result.
# ===========================================================================
#
# MEASURED pre-fix (same singlet probe): ``output_grid=(96, 80e-6)`` returned a
# field that was GENUINELY resampled to 80 um -- bit-identical to additionally
# passing ``output_dx=80e-6`` -- while ``PropagationResult.dx`` reported the
# 40 um INPUT pitch.  True for gbd / hf / hfpi (which forward the request) and
# for asm / fresnel / fraunhofer (via the MFT promotion).  On the post-P5-flip
# DEFAULT contract every consumer that builds coordinates from ``result.dx``
# was therefore off by the resampling ratio.


@pytest.mark.parametrize('form', ['tuple', 'dict'])
def test_output_grid_pitch_is_reported_on_the_result_bare_grid(form):
    g = _PRESCRIPTION_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    dx_req = 80e-6
    og = (96, dx_req) if form == 'tuple' else {'N': 96, 'dx': dx_req}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, z=2e-3, wavelength=g['wl'], dx=g['dx'],
                        method='asm', output_grid=og)
    assert res.field.shape == (96, 96)
    assert res.dx == pytest.approx(dx_req, rel=1e-12), (
        f'W9-5: output_grid pitch not reported (got {res.dx!r})')
    assert res.dy == pytest.approx(dx_req, rel=1e-12)


@pytest.mark.parametrize('method', ['gbd', 'hf'])
def test_output_grid_pitch_is_reported_on_the_result_prescription(method):
    g = _PRESCRIPTION_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    dx_req = 80e-6
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, wavelength=g['wl'], dx=g['dx'],
                        prescription=_singlet(), method=method,
                        output_grid=(96, dx_req))
        ref = propagate(E, wavelength=g['wl'], dx=g['dx'],
                        prescription=_singlet(), method=method,
                        output_grid=(96, dx_req), output_dx=dx_req)
    assert res.field.shape == (96, 96)
    assert res.dx == pytest.approx(dx_req, rel=1e-12), (
        f'W9-5: {method} output_grid pitch not reported (got {res.dx!r})')
    # The two spellings of the same request must be the same propagation --
    # this is what makes the pre-fix label provably wrong rather than a
    # different (coarser) call.
    assert np.array_equal(np.asarray(res.field), np.asarray(ref.field))


def test_output_dx_shortcut_still_wins_over_output_grid():
    """Precedence fence: ``output_dx`` overrides ``output_grid``'s pitch in the
    two resolvers, so the reported pitch must follow ``output_dx``."""
    g = _PRESCRIPTION_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, z=2e-3, wavelength=g['wl'], dx=g['dx'],
                        method='asm', output_grid=(96, 80e-6),
                        output_dx=120e-6)
    assert res.dx == pytest.approx(120e-6, rel=1e-12)


def test_no_output_request_reports_the_input_pitch():
    """Regression fence: with no request the wrapper still reports the input
    pitch for pitch-preserving kernels."""
    g = _PRESCRIPTION_GEOM
    E = _gauss(g['N'], g['dx'], g['w0'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = propagate(E, z=2e-3, wavelength=g['wl'], dx=g['dx'],
                        method='asm')
    assert res.dx == pytest.approx(g['dx'], rel=1e-12)


# ===========================================================================
# W9-6 -- the flipped default contract must never hand back field=None.
# ===========================================================================
#
# MEASURED pre-fix (N=32, dx=2 um, lambda=633 nm, one ASM subdomain):
#
#   propagate(E, method='mhs', subdomains=[...])
#     -> PropagationResult, field (32, 32)                     OK
#   propagate(E, method='mhs', subdomains=[...], return_intermediate=True)
#     -> PropagationResult, field is None, NO warning          <-- silent null
#
# MhsPipeline.run's native return there is a ``list`` of
# ``(HuygensSurface, ndarray)`` pairs; ``_coerce_field`` cannot read that shape
# and falls through to its ``(None, None, None)`` sentinel, which the wrapper
# then published.  ``return_intermediate=True`` is ``run``'s OWN default (only
# the dispatcher defaults it to False), so this is the natural thing to write.
# The P5 flip's stated guarantee is that ``.field`` is defined "whichever kernel
# ran" -- so the wrapper must raise rather than publish a null field.


def _mhs_subdomains(N, dx, wl):
    from lumenairy.propagators.mhs import HuygensSurface, asm_subdomain
    s0 = HuygensSurface(z=0.0, Ny=N, Nx=N, dx=dx, label='in')
    s1 = HuygensSurface(z=5e-5, Ny=N, Nx=N, dx=dx, label='out')
    return [asm_subdomain(s0, s1, wavelength=wl)]


def test_wrapper_never_publishes_a_null_field():
    N, dx = 32, 2.0e-6
    E = _gauss(N, dx, 10e-6)
    subs = _mhs_subdomains(N, dx, LAM)
    with pytest.raises(ValueError) as info:
        propagate(E, wavelength=LAM, dx=dx, method='mhs', subdomains=subs,
                  return_intermediate=True)
    msg = str(info.value)
    assert 'return_result=False' in msg, (
        f'W9-6: the diagnostic must name the escape hatch; got {msg!r}')


def test_mhs_history_is_reachable_through_the_legacy_contract():
    """The raise must come with a working alternative: ``return_result=False``
    hands back the native per-surface history untouched."""
    N, dx = 32, 2.0e-6
    E = _gauss(N, dx, 10e-6)
    subs = _mhs_subdomains(N, dx, LAM)
    out = propagate(E, wavelength=LAM, dx=dx, method='mhs', subdomains=subs,
                    return_intermediate=True, return_result=False)
    assert isinstance(out, list) and len(out) == 2
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in out)
    assert np.asarray(out[-1][1]).shape == (N, N)


def test_mhs_single_plane_default_still_wraps():
    """Regression fence: the dispatcher's own default
    (``return_intermediate=False``) is a single plane and must keep wrapping."""
    N, dx = 32, 2.0e-6
    E = _gauss(N, dx, 10e-6)
    subs = _mhs_subdomains(N, dx, LAM)
    res = propagate(E, wavelength=LAM, dx=dx, method='mhs', subdomains=subs)
    assert isinstance(res, PropagationResult)
    assert res.field is not None and res.field.shape == (N, N)
    assert res.method == 'mhs'
