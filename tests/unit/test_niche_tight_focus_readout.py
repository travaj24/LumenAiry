"""A 3 um spot does NOT require a 32k propagation grid.

WHY THIS FILE EXISTS
--------------------
A tight focus looks like it forces an enormous grid.  The reasoning goes: a 3 um
spot needs NA 0.278, Nyquist then demands ``dx <= lambda/(2 NA) = 2.36 um`` on
the converging wavefront, and because the Sziklas-Siegman co-moving frame
EXPANDS the pixel pitch on the way to a focus (measured 25.6x on design 121's
pre-DOE leg) you would have to pre-compensate by that factor -- landing at
N ~ 32768, which needs ~105 GB and is unreachable.

That reasoning is wrong for a CARRIER-REFERENCED field, and this file pins why.
The steep converging phase is carried ANALYTICALLY by the carrier
``exp(i k S(r; R))``, not stored on the grid.  What the grid holds is the
slowly-varying ENVELOPE, which for a clean converging beam is just a smooth
Gaussian with no steep phase at all -- so ``dx <= lambda/(2 NA)`` simply does
not apply to it.  Measured below: the readout of a 3 um focus is IDENTICAL
(relL2 = 0) whether the envelope is carried on N=1024 or N=4096, including grids
that violate the pupil Nyquist limit by 2.5x.

What actually governs accuracy at a tight focus is not transverse sampling of
the propagation grid but:

  1. LONGITUDINAL placement of the readout plane -- the dominant term.  At
     NA 0.278 the Rayleigh range is only 5.4 um, so a 2.3 um error in where you
     read (0.43 z_R) costs 8.7% in measured waist.  This is a far tighter
     tolerance than anything transverse, and it is the thing to get right.
  2. the READOUT WINDOW, but only for wing-weighted metrics.  The core 1/e^2
     width is robust: it moves <3% between a 0.05 um and a 0.4 um readout
     pitch.  What breaks is ``N_out * dx_out`` exceeding the Bluestein/MFT
     transform period ``N_in * d_in`` -- the outer window then fills with
     PERIODIC REPLICAS, so second-moment / r^2-weighted / large-radius
     encircled-energy metrics read wildly wrong (2 sigma = 88 um against a
     1.53 um waist) while the core still looks perfect.  Since fix D3 the
     readout REFUSES that window by default (``on_replica='error'``); the
     test below asserts the refusal AND, with the refusal explicitly waived,
     that the corruption it prevents is real.
  3. the GRID EXTENT in beam radii, which is what sets the leg length in (1).
     Fix D2 (2026-08-06) replaced a constant standoff with the shortest leg
     whose hand-off plane still contains the beam; the test at the bottom of
     this file pins that against BOTH constants it replaced, at two extents.

Ground truth is analytic, derived independently of the library: for a Gaussian
of radius ``w_in`` with wavefront radius ``-Z``, the complex-q law
``1/q = 1/R - i lambda/(pi w^2)``, ``q_out = q_in + Z`` gives a pure-imaginary
``q_out = i z_R`` and hence ``w0 = lambda Z / (pi w_in)`` exactly.  Where the
pupil is truncated hard enough for that to stop being the right truth (the
half-extent 2.0 case), the reference is instead an EXACT DISCRETE PARAXIAL
focal-plane quadrature of the same sampled pupil, in plain numpy -- see
``_oracle_focal_line``.
"""
import numpy as np
import pytest

from lumenairy.propagators.carrier import (
    _default_focus_standoff,
    carrier_referenced_focus_readout,
)

_WL = 1.31e-6
_W0_TARGET = 1.5e-6                       # 3.0 um spot DIAMETER
_W_IN = 1.0e-3
_Z = np.pi * _W_IN * _W0_TARGET / _WL     # pupil -> waist distance
_ZR = np.pi * _W0_TARGET ** 2 / _WL       # 5.4 um
_NA = _W_IN / _Z                          # 0.278
# The readout works in two legs -- a carrier step to ``z - standoff`` then a
# fine Bluestein zoom over ``standoff`` -- and its accuracy is governed by the
# LENGTH of that second leg, not by the propagation grid.  Measured waist error
# at the nominal plane vs standoff (analytic truth 1.5000 um):
#   0.5 zR  8.74% | 1.0 zR  0.60% | 1.5 zR  1.12% | 3 zR  3.33%
#   6.0 zR  8.74%  <-- the original _BRIDGE_ZR_FACTOR default
#  10.0 zR 15.58%
# ... at THIS grid extent (half-extent 3 beam radii).  The optimum MOVES with
# the extent -- that is defect D2 and the reason the shipped default is no
# longer a constant.  ``_STANDOFF`` is kept only where a test deliberately
# pins one leg length; everything else runs the SHIPPED default.
_STANDOFF = _ZR
_DZ_BEST = -0.50e-6
# Grid half-extent in beam radii for the default fixture.  Named because the
# whole D2 finding is that this number, not NA, controls the leg.
_EXT = 3.0


def _pupil(n, ext=_EXT):
    dx = 2.0 * ext * _W_IN / n
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X * X + Y * Y) / _W_IN ** 2).astype(np.complex128), dx


def _default_standoff(n=2048, ext=_EXT, dz=0.0):
    """The leg the SHIPPED resolver picks for this fixture."""
    env, dx = _pupil(n, ext)
    return _default_focus_standoff(env, -_Z, _Z + dz, _WL, dx)


def _period(n=2048, ext=_EXT, dz=0.0, standoff=None):
    """One Bluestein period of the readout, in metres.

    Derived rather than probed: the carrier leg stops at ``z_stop = z -
    standoff``, where the residual carrier radius is ``R + z_stop`` and the
    co-moving pitch is therefore ``dx * |R + z_stop| / |R|``; the period is
    ``N`` times it.  (With the target ``dz`` PAST the focus the resolved
    standoff already spans that overshoot, so the pitch is set by
    ``standoff - dz``, not by the standoff alone.)
    """
    _, dx = _pupil(n, ext)
    s = _default_standoff(n, ext, dz) if standoff is None else float(standoff)
    z_stop = (_Z + dz) - s
    return n * dx * abs(-_Z + z_stop) / _Z


def _n_out_for(window, dx_out):
    """Largest even ``N_out`` whose window fits inside ``window``."""
    return max(2, 2 * int(np.floor(window / dx_out / 2.0)))


def _read(n, dz=0.0, dx_out=0.05e-6, n_out=None, standoff=None, ext=_EXT,
          on_replica='error'):
    """Read out on the SHIPPED default leg unless a test pins one.

    ``n_out`` defaults to the largest window that fits inside one Bluestein
    period at the leg actually used -- i.e. the test asks the propagator for
    what it can deliver instead of quietly reading replicas.  A 1/e^2 radius
    of a 1.5 um waist needs ~4 waists of window; the period supplies 5.2.
    """
    env, dx = _pupil(n, ext)
    if n_out is None:
        n_out = _n_out_for(_period(n, ext, dz, standoff), dx_out)
    return np.asarray(carrier_referenced_focus_readout(
        env, -_Z, _Z + dz, _WL, dx, dx_out=dx_out, N_out=int(n_out),
        standoff=standoff, on_replica=on_replica))


def _oracle_focal_line(n, ext, xo):
    """|E| on the focal plane at abscissae ``xo``, by EXACT discrete paraxial
    quadrature of the same sampled pupil -- plain numpy, no lumenairy.

    For a pupil ``A(r) exp(-i k r^2 / 2Z)`` the two quadratic phases cancel
    identically in the Fresnel integral, so the focal field is the plain
    Fourier transform of the sampled AMPLITUDE.  The pupil is separable, so
    one 1-D chirp-z gives the whole 2-D answer as an outer product.  This is
    the truth the analytic complex-q Gaussian STOPS being once the square
    grid truncates the beam appreciably (half-extent 2.0: 1.79 um FWHM
    against the untruncated 1.77).
    """
    dx = 2.0 * ext * _W_IN / n
    x = (np.arange(n) - n // 2) * dx
    a = np.exp(-(x * x) / _W_IN ** 2)
    return np.abs(a @ np.exp(-2j * np.pi * np.outer(x, xo) / (_WL * _Z)))


def _e2_radius(I, dxo):
    """1/e^2 intensity radius by interpolation on the central profile."""
    c = np.unravel_index(np.argmax(I), I.shape)
    p = I[c[0], c[1]:].astype(float)
    p = p / p[0]
    tgt = np.exp(-2.0)
    k = int(np.argmax(p < tgt))
    if k == 0:
        return np.nan
    f = (p[k - 1] - tgt) / (p[k - 1] - p[k])
    return (k - 1 + f) * dxo


def test_the_analytic_ground_truth_is_self_consistent():
    """Verify the reference by the complex-q law before trusting it to judge
    the library: q_out must come out pure-imaginary with Im = z_R."""
    q_in = 1.0 / (-1.0 / _Z - 1j * _WL / (np.pi * _W_IN ** 2))
    q_out = q_in + _Z
    assert abs(q_out.real) < 0.03e-6, (
        f'waist is not at Z: Re(q_out) = {q_out.real:.3e} m')
    w0 = np.sqrt(_WL * q_out.imag / np.pi)
    assert abs(w0 - _W0_TARGET) / _W0_TARGET < 1e-3, (
        f'complex-q waist {w0 * 1e6:.4f} um != target {_W0_TARGET * 1e6} um')
    assert abs(q_out.imag - _ZR) / _ZR < 1e-3


def test_three_micron_spot_is_recovered_from_a_tractable_grid():
    """The headline: N=2048 (0.5 GB working set) resolves a 3 um focus to
    better than 3%.  No 32k grid, no 105 GB."""
    I = np.abs(_read(2048, dz=_DZ_BEST)) ** 2
    w = _e2_radius(I, 0.05e-6)
    assert abs(w - _W0_TARGET) / _W0_TARGET < 0.01, (
        f'recovered waist {w * 1e6:.4f} um vs analytic '
        f'{_W0_TARGET * 1e6:.4f} um')


@pytest.mark.parametrize('n', [1024, 2048, 4096])
def test_focal_accuracy_does_not_depend_on_the_propagation_grid(n):
    """The carrier holds the steep phase, so refining the envelope grid buys
    nothing -- including at N=1024, whose dx = 5.86 um violates the pupil
    Nyquist limit (2.36 um) by 2.5x and which is nevertheless correct."""
    I = np.abs(_read(n, dz=_DZ_BEST)) ** 2
    w = _e2_radius(I, 0.05e-6)
    assert abs(w - _W0_TARGET) / _W0_TARGET < 0.03, (
        f'N={n}: waist {w * 1e6:.4f} um')


def test_pupil_nyquist_really_is_violated_at_the_coarsest_grid():
    """Guard the premise of the test above: if the grid ever got fine enough to
    satisfy lambda/(2 NA), it would stop demonstrating anything."""
    _, dx = _pupil(1024)
    assert dx > _WL / (2.0 * _NA), (
        f'dx={dx * 1e6:.3f} um no longer violates the pupil Nyquist limit '
        f'{_WL / (2 * _NA) * 1e6:.3f} um')


def _second_moment_radius(I, dxo):
    """Wing-sensitive width: 2*sigma of the central profile.  For a clean
    Gaussian this equals the 1/e^2 radius; the two DIVERGE exactly when the
    outer window is contaminated."""
    c = np.unravel_index(np.argmax(I), I.shape)
    prof = I[c[0], :].astype(float)
    x = (np.arange(I.shape[1]) - c[1]) * dxo
    return 2.0 * np.sqrt(np.sum(prof * x * x) / np.sum(prof))


def test_the_core_waist_is_robust_to_the_readout_pitch():
    """Good news, and worth pinning: the 1/e^2 CORE width barely moves between
    a 0.05 um and a 0.4 um readout pitch (7.5 samples across the spot), so a
    core-width answer does not silently depend on how finely you read out."""
    w_fine = _e2_radius(np.abs(_read(2048, dz=_DZ_BEST,
                                     dx_out=0.05e-6)) ** 2, 0.05e-6)
    for dxo in (0.10e-6, 0.40e-6):
        w = _e2_radius(np.abs(_read(2048, dz=_DZ_BEST, dx_out=dxo)) ** 2, dxo)
        assert abs(w - w_fine) / w_fine < 0.03, (
            f'dx_out={dxo * 1e6:.2f} um core waist {w * 1e6:.4f} um vs '
            f'{w_fine * 1e6:.4f} um at 0.05 um')


def test_an_oversized_readout_window_is_refused_by_default():
    """FIX D3.  The readout REFUSES a window wider than one Bluestein period.

    ``carrier_referenced_focus_readout`` reaches its output grid through a
    Bluestein/MFT step whose transform PERIOD is ``N_in * d_in`` of the
    CO-MOVING grid at the stop plane.  Asking for ``N_out * dx_out`` beyond
    that period does not produce new information -- it produces periodic
    REPLICAS of the field in the outer window.

    Until 2026-08-06 the only thing that fired was a downstream
    ``UserWarning`` from ``angular_spectrum_propagate_mft``, which any
    upstream ``filterwarnings('ignore')`` removes -- and this file's own
    fixtures had been reading a 25.6 um window on a ~9.8 um period for their
    whole life, which is where the "even the good 2*sigma is 16.1 um for a
    1.5 um waist" note below came from.  It was replicas, not wings.
    """
    dxo, n_out = 0.40e-6, 512                       # 204.8 um: ~13 periods
    # RuntimeError, the same type the multi-congruence chain's own replica
    # guard raises -- one fault, one exception type, two scopes.
    with pytest.raises(RuntimeError, match='Bluestein period'):
        _read(2048, dz=_DZ_BEST, dx_out=dxo, n_out=n_out)
    # the refusal is not blanket: a window INSIDE one period is accepted, and
    # it is the same call in every other respect.
    _read(2048, dz=_DZ_BEST, dx_out=dxo,
          n_out=_n_out_for(_period(2048, dz=_DZ_BEST), dxo))


def test_the_refused_window_really_would_have_been_corrupt():
    """... and the refusal is worth having: with it explicitly waived, the
    core still reads correctly while the wing metric does not.

    That asymmetry is the whole reason this is an error rather than a
    warning -- a spot budget that checks a width or a peak cannot detect the
    failure, so a silent one is a plausible-looking wrong answer.
    """
    dxo = 0.40e-6
    I_bad = np.abs(_read(2048, dz=_DZ_BEST, dx_out=dxo, n_out=512,
                         on_replica='ignore')) ** 2
    I_good = np.abs(_read(2048, dz=_DZ_BEST, dx_out=0.05e-6)) ** 2
    core_bad = _e2_radius(I_bad, dxo)
    wing_bad = _second_moment_radius(I_bad, dxo)
    wing_good = _second_moment_radius(I_good, 0.05e-6)
    # the core survives ...
    assert abs(core_bad - _W0_TARGET) / _W0_TARGET < 0.05
    # ... while the wing metric does not, by a wide margin.  Both 2*sigma
    # readings exceed the 1.5 um waist because a second moment is dominated by
    # whatever sits in the wings; what separates them is that the accepted
    # window's wings are the beam's own halo and the refused window's are
    # copies of the spot.
    assert wing_bad > 5.0 * wing_good, (
        f'expected the oversized window to wreck the wing metric: '
        f'2sigma {wing_bad * 1e6:.2f} um vs {wing_good * 1e6:.2f} um')


def test_the_exact_readout_guards_the_same_way_on_its_own_period():
    """The EXACT high-NA readout has the same Bluestein periodicity, but its
    period is the FINE CROP WINDOW (``window_factor`` beam radii), not a
    function of a standoff -- so the guard must be there too and its remedy
    must name the right knob.  Leaving one of the two public readouts
    unguarded is the asymmetry that let the paraxial one ship without a guard
    in the first place.
    """
    from lumenairy import carrier_referenced_exact_focus_readout as _EX
    n, dx, w, R = 512, 0.5e-6, 30e-6, -0.2e-3
    x = (np.arange(n) - n // 2) * dx
    r2 = x[:, None] ** 2 + x[None, :] ** 2
    S = np.sign(R) * (np.sqrt(r2 + R * R) - abs(R))
    E = (np.exp(-r2 / w ** 2)
         * np.exp(1j * 2.0 * np.pi / _WL * S)).astype(np.complex128)
    kw = dict(dx_out=0.05e-6, window_factor=4.0)
    # 25.6 um fits inside the 4 w = 120 um crop window ...
    _EX(E, R, -R, _WL, dx, N_out=512, **kw)
    # ... 204.8 um does not, and the message names window_factor, not standoff
    with pytest.raises(RuntimeError, match='window_factor'):
        _EX(E, R, -R, _WL, dx, N_out=4096, **kw)
    _EX(E, R, -R, _WL, dx, N_out=4096, on_replica='ignore', **kw)


def test_on_replica_is_validated_and_not_a_silent_fall_through():
    """A typo in the disposition must not quietly disable the guard -- the
    same defect class as the ``on_readout_windo`` / ``gap_kernel`` fall-
    throughs this campaign already fixed."""
    with pytest.raises(ValueError, match='on_replica'):
        _read(2048, dx_out=0.05e-6, n_out=64, on_replica='erorr')


def test_longitudinal_placement_dominates_the_error_budget():
    """At NA 0.278 the Rayleigh range is 5.4 um, so depth placement -- not grid
    size -- is what a tight-focus budget must control.  A 3 um defocus (0.56 zR)
    widens the measured spot ~30%, which dwarfs every transverse effect measured
    in this file.

    Near focus the growth follows the Gaussian depth-of-focus law
    ``w(dz) = w0 sqrt(1 + (dz/zR)^2)``.  Further out the readout's own standoff
    bookkeeping adds a few percent on top, so the law is only asserted where it
    is clean.

    BAR RE-BASELINED 1% -> 3% (fix D2, 2026-08-06), and it is an EROSION, not
    a tightening.  This arm used to run a PINNED ``standoff = 1.0 zR``; it now
    runs the SHIPPED default, which at THIS grid extent (3 beam radii) is
    1.73 zR.  Measured deviation from the law: 0.98% at dz = 0 and 1.82% at
    dz = 1 um, against 0.17% / 0.52% for the retired 1.0 zR pin.  That cost is
    real and is the honest price of a resolver that follows the grid extent
    instead of one constant -- at half-extent 2 beam radii the same resolver is
    2.0x BETTER than either retired constant (see
    ``test_the_default_leg_beats_both_constants_at_a_truncated_pupil``), and it
    is 2.4x better than both at dz = 3 um here.  Stated rather than hidden.
    """
    w_ref = _e2_radius(np.abs(_read(2048, dz=_DZ_BEST)) ** 2, 0.05e-6)
    # 1. the law holds near focus
    for dzu in (0.0, 1.0):
        dz = dzu * 1e-6
        w = _e2_radius(np.abs(_read(2048, dz=dz)) ** 2, 0.05e-6)
        pred = _W0_TARGET * np.sqrt(1.0 + ((dz - _DZ_BEST) / _ZR) ** 2)
        assert abs(w - pred) / pred < 0.03, (
            f'dz={dzu} um: waist {w * 1e6:.4f} um vs depth-of-focus law '
            f'{pred * 1e6:.4f} um')
        # ... and the pinned 1.0 zR leg still reaches the historical 1%, so
        # the loosening above is attributed to the DEFAULT and not to a
        # regression in the readout itself.
        w1 = _e2_radius(np.abs(_read(2048, dz=dz, standoff=1.0 * _ZR)
                               ) ** 2, 0.05e-6)
        assert abs(w1 - pred) / pred < 0.01, (
            f'dz={dzu} um at standoff=1.0 zR: waist {w1 * 1e6:.4f} um vs '
            f'{pred * 1e6:.4f} um -- the readout itself has regressed, not '
            f'just the resolved default')
    # 2. and the penalty is large enough to dominate the budget -- an order
    #    of magnitude above the <3% the readout PITCH moves the same width.
    #
    #    SCORED AGAINST THE LAW, not against a bare number (fix D2,
    #    2026-08-06).  The retired ">25%" bar was itself a readout artefact:
    #    the depth-of-focus law predicts 19.2% here, and the pinned 1.0 zR leg
    #    read 30% only because it over-reads a DEFOCUSED spot by 14% (measured
    #    1.9650 um against an exact discrete paraxial oracle's 1.7178 um at
    #    dz = 3 um; the shipped default reads 1.8173 um, i.e. 2.4x closer).
    #    So the bar moved DOWN because the readout got more accurate, and it
    #    is now anchored to the physics rather than to the old error.
    w_3um = _e2_radius(np.abs(_read(2048, dz=3.0e-6)) ** 2, 0.05e-6)
    pred_ratio = (np.sqrt(1.0 + ((3.0e-6 - _DZ_BEST) / _ZR) ** 2)
                  / np.sqrt(1.0 + ((_DZ_BEST - _DZ_BEST) / _ZR) ** 2))
    assert pred_ratio > 1.15, 'fixture no longer defocuses appreciably'
    assert abs((w_3um / w_ref) / pred_ratio - 1.0) < 0.05, (
        f'a 3 um defocus should widen the spot by the depth-of-focus law '
        f'{100 * (pred_ratio - 1):.1f}%, measured '
        f'{100 * (w_3um / w_ref - 1):.1f}%')
    assert w_3um / w_ref > 1.15, (
        f'... and the effect must dwarf the <3% the readout pitch moves the '
        f'same width: got {100 * (w_3um / w_ref - 1):.1f}%')


def test_readout_accuracy_is_governed_by_the_bluestein_leg_length():
    """The dominant error term at a tight focus, and the one most likely to be
    mistaken for physics.

    ``standoff`` is documented as a robustness knob (keep the co-moving grid
    from collapsing at the focus), so it is easy to assume it does not affect
    the ANSWER.  It does, strongly: AT THIS GRID EXTENT (half-extent 3 beam
    radii) the measured waist at the nominal plane runs 0.60% wrong at
    ``standoff = zR`` and 8.74% wrong at ``6 zR`` -- a 14x accuracy
    difference from a knob that reads as purely defensive.

    The sensitivity is what is asserted here; WHICH leg is best is a function
    of the grid extent, which is defect D2 and is pinned separately below.
    """
    def waist_at_nominal(standoff):
        F = _read(2048, standoff=standoff, dx_out=0.05e-6)
        return _e2_radius(np.abs(F) ** 2, 0.05e-6)

    w_tuned = waist_at_nominal(1.0 * _ZR)
    w_default = waist_at_nominal(6.0 * _ZR)
    e_tuned = abs(w_tuned - _W0_TARGET) / _W0_TARGET
    e_default = abs(w_default - _W0_TARGET) / _W0_TARGET
    assert e_tuned < 0.015, (
        f'standoff=zR should be near-exact, got {w_tuned * 1e6:.4f} um '
        f'({e_tuned * 100:.2f}%)')
    assert e_default > 4.0 * e_tuned, (
        f'expected 6*zR to be materially worse than standoff=zR: '
        f'{e_default * 100:.2f}% vs {e_tuned * 100:.2f}%')


# ===========================================================================
# FIX D2 -- the leg length is set by the GRID EXTENT, not by a constant
# ===========================================================================
def _fwhm(I, dxo):
    c = np.unravel_index(np.argmax(I), I.shape)
    p = I[c[0], c[1]:].astype(float) / I[c[0], c[1]]
    k = int(np.argmax(p < 0.5))
    if k == 0:
        return np.nan
    f = (p[k - 1] - 0.5) / (p[k - 1] - p[k])
    return 2.0 * (k - 1 + f) * dxo


def _score_against_oracle(ext, standoff, n=2048, dx_out=0.02e-6):
    """FWHM error (fraction) and relative L2 of ``|F|``, scored on a window
    that fits inside the SMALLEST period of the arms being compared, so no
    arm is judged on replicas and all three see the same window."""
    env, dx = _pupil(n, ext)
    legs = [_default_standoff(n, ext), 0.8 * _ZR, 6.0 * _ZR]
    win = min(_period(n, ext, standoff=s) for s in legs)
    n_out = _n_out_for(win, dx_out)
    xo = (np.arange(n_out) - n_out // 2) * dx_out
    line = _oracle_focal_line(n, ext, xo)
    O = np.outer(line, line)
    F = np.abs(np.asarray(carrier_referenced_focus_readout(
        env, -_Z, _Z, _WL, dx, dx_out=dx_out, N_out=n_out,
        standoff=standoff)))
    f_o = _fwhm(O ** 2, dx_out)
    a, b = F / F.max(), O / O.max()
    return (abs(_fwhm(F ** 2, dx_out) - f_o) / f_o,
            float(np.linalg.norm(a - b) / np.linalg.norm(b)))


def test_the_default_leg_beats_both_constants_at_a_truncated_pupil():
    """THE D2 COUNTEREXAMPLE, pinned.

    At a grid half-extent of 2 beam radii -- a truncated but entirely
    ordinary pupil -- the retired 0.8 zR constant was measurably WORSE than
    the 6.0 zR one it replaced, and both were ~2x off the reachable optimum.
    The resolved default must beat BOTH.

    The reference is the exact discrete paraxial focal-plane quadrature of
    the same sampled pupil (``_oracle_focal_line``), NOT the analytic
    complex-q Gaussian: at this extent the square grid truncates the beam
    hard enough that the untruncated Gaussian is itself 1.5% away from the
    right answer, which is how the original measurement mistook the two.
    """
    ext = 2.0
    e_def, r_def = _score_against_oracle(ext, _default_standoff(2048, ext))
    e_08, r_08 = _score_against_oracle(ext, 0.8 * _ZR)
    e_60, r_60 = _score_against_oracle(ext, 6.0 * _ZR)
    # measured 2026-08-06: FWHM 4.475% vs 8.945% (0.8 zR) and 8.501% (6.0 zR);
    # relL2 4.81e-2 vs 9.02e-2 and 8.70e-2.  Bars are RATIOS with ~1.5x of
    # headroom against a 1.9-2.0x measurement, so a partial regression shows.
    assert e_def < 0.7 * min(e_08, e_60), (
        f'ext={ext}: the resolved default must beat both retired constants on '
        f'FWHM error: {e_def:.4%} vs 0.8 zR {e_08:.4%} / 6.0 zR {e_60:.4%}')
    assert r_def < 0.7 * min(r_08, r_60), (
        f'ext={ext}: ... and on the field: relL2 {r_def:.3e} vs '
        f'{r_08:.3e} / {r_60:.3e}')


def test_the_resolved_leg_holds_the_beam_at_the_hand_off_plane():
    """The INVARIANT the resolver is built on, as an identity rather than a
    numeric pin: at every grid extent the co-moving half-width at the stop
    plane must be ``_FOCUS_STANDOFF_MARGIN`` beam radii (or, on a grid too
    narrow to reach that, the capped fraction of what it can reach).

    This is the gate against reverting to a constant multiple of ``zR``: a
    constant satisfies it at exactly one extent.
    """
    from lumenairy.propagators.carrier import _FOCUS_STANDOFF_MARGIN, _FOCUS_STANDOFF_WAIST_GROWTH
    f_cap = np.sqrt(_FOCUS_STANDOFF_WAIST_GROWTH ** 2 - 1.0)
    sat = f_cap / np.sqrt(1.0 + f_cap ** 2)
    seen = []
    for ext in (1.5, 2.0, 3.0, 4.0, 6.0, 10.0):
        n = 2048
        env, dx = _pupil(n, ext)
        w_env = np.sqrt(2.0) * np.sqrt(
            (np.abs(env) ** 2 * (((np.arange(n) - n / 2) * dx)[:, None] ** 2
                                 + ((np.arange(n) - n / 2) * dx)[None, :] ** 2)
             ).sum() / (np.abs(env) ** 2).sum())
        ext_meas = 0.5 * n * dx / w_env
        # zR from the MEASURED envelope radius, exactly as the resolver does
        # -- at a hard truncation w_env is not w_in and the nominal _ZR is the
        # wrong denominator by ~1 %.
        w0_meas = _WL * _Z / (np.pi * w_env)
        zr_meas = np.pi * w0_meas ** 2 / _WL
        f = _default_standoff(n, ext) / zr_meas
        # half-width at the stop plane, in beam radii there
        margin = ext_meas * f / np.sqrt(1.0 + f * f)
        want = min(_FOCUS_STANDOFF_MARGIN, sat * ext_meas)
        assert abs(margin - want) < 1e-6 * max(want, 1.0), (
            f'ext={ext} (measured {ext_meas:.4f}): hand-off containment '
            f'margin {margin:.6f} != the required {want:.6f}')
        seen.append(f)
    # ... and it is genuinely NOT a constant: the resolved factor spans the
    # cap at a narrow grid down to a short leg at a wide one.
    assert max(seen) / min(seen) > 4.0, (
        f'the resolved standoff factors {seen} barely move with the grid '
        f'extent -- the resolver has collapsed back to a constant')
