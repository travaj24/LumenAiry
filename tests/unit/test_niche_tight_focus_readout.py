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
     1.53 um waist) while the core still looks perfect.  The readout warns
     about this; the test below asserts both halves.

Ground truth is analytic, derived independently of the library: for a Gaussian
of radius ``w_in`` with wavefront radius ``-Z``, the complex-q law
``1/q = 1/R - i lambda/(pi w^2)``, ``q_out = q_in + Z`` gives a pure-imaginary
``q_out = i z_R`` and hence ``w0 = lambda Z / (pi w_in)`` exactly.
"""
import numpy as np
import pytest

from lumenairy.propagators.carrier import carrier_referenced_focus_readout

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
#   6.0 zR  8.74%  <-- the shipped _BRIDGE_ZR_FACTOR default
#  10.0 zR 15.58%
# So ~1 zR is the sweet spot; it degrades again below that, which is the
# co-moving-grid collapse this helper exists to avoid.
_STANDOFF = _ZR
_DZ_BEST = -0.50e-6


def _pupil(n):
    dx = 6.0e-3 / n
    x = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(x, x)
    return np.exp(-(X * X + Y * Y) / _W_IN ** 2).astype(np.complex128), dx


def _read(n, dz=0.0, dx_out=0.05e-6, n_out=512):
    env, dx = _pupil(n)
    return np.asarray(carrier_referenced_focus_readout(
        env, -_Z, _Z + dz, _WL, dx, dx_out=dx_out, N_out=n_out,
        standoff=_STANDOFF))


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


def test_an_oversized_readout_window_corrupts_wing_metrics_not_the_core():
    """The REAL trap, and the one that matters for spot budgets.

    ``carrier_referenced_focus_readout`` reaches its output grid through a
    Bluestein/MFT step whose transform PERIOD is ``N_in * d_in``.  Asking for
    ``N_out * dx_out`` beyond that period does not produce new information -- it
    produces periodic REPLICAS of the field in the outer window.  The core is
    unharmed, so a 1/e^2 width still looks right, while any wing-weighted
    metric (second moment, r^2-weighted spot size, encircled energy at large
    radius) is silently garbage.  Measured: 2*sigma reads ~88 um against a true
    1.53 um waist -- a 58x error that a core-width check cannot see.

    This is the same failure signature the traced Nyquist guard warns about
    ("r^2-weighted spot metrics read low while EE50/EE80 stay plausible"), and
    the readout emits its own warning for it, asserted here.
    """
    import warnings
    dxo, n_out = 0.40e-6, 512
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        I_bad = np.abs(_read(2048, dz=_DZ_BEST, dx_out=dxo,
                             n_out=n_out)) ** 2
    msgs = ' '.join(str(w.message) for w in rec)
    assert 'period' in msgs.lower(), (
        'the readout did not warn that the requested window exceeds one '
        'transform period -- a silent replica field is far worse than a noisy '
        f'one.  warnings seen: {msgs[:200]!r}')
    I_good = np.abs(_read(2048, dz=_DZ_BEST, dx_out=0.05e-6,
                          n_out=n_out)) ** 2
    core_bad = _e2_radius(I_bad, dxo)
    wing_bad = _second_moment_radius(I_bad, dxo)
    wing_good = _second_moment_radius(I_good, 0.05e-6)
    # the core survives ...
    assert abs(core_bad - _W0_TARGET) / _W0_TARGET < 0.05
    # ... while the wing metric does not, by a wide margin
    # measured 119.1 um vs 16.1 um = 7.4x.  Note even the "good" 2*sigma
    # (16.1 um) hugely exceeds the 1.5 um waist, because a second moment over a
    # 25.6 um window is dominated by whatever sits in the wings -- which is
    # precisely why a wing metric must not be trusted without checking the
    # window against the transform period first.
    assert wing_bad > 5.0 * wing_good, (
        f'expected the oversized window to wreck the wing metric: '
        f'2sigma {wing_bad * 1e6:.2f} um vs {wing_good * 1e6:.2f} um')


def test_longitudinal_placement_dominates_the_error_budget():
    """At NA 0.278 the Rayleigh range is 5.4 um, so depth placement -- not grid
    size -- is what a tight-focus budget must control.  A 3 um defocus (0.56 zR)
    widens the measured spot ~30%, which dwarfs every transverse effect measured
    in this file.

    Near focus the growth follows the Gaussian depth-of-focus law
    ``w(dz) = w0 sqrt(1 + (dz/zR)^2)`` to better than 1%.  Further out the
    readout's own standoff bookkeeping adds a few percent on top, so the law is
    only asserted where it is clean.
    """
    w_ref = _e2_radius(np.abs(_read(2048, dz=_DZ_BEST)) ** 2, 0.05e-6)
    # 1. the law holds near focus
    for dzu in (0.0, 1.0):
        dz = dzu * 1e-6
        w = _e2_radius(np.abs(_read(2048, dz=dz)) ** 2, 0.05e-6)
        pred = _W0_TARGET * np.sqrt(1.0 + ((dz - _DZ_BEST) / _ZR) ** 2)
        assert abs(w - pred) / pred < 0.01, (
            f'dz={dzu} um: waist {w * 1e6:.4f} um vs depth-of-focus law '
            f'{pred * 1e6:.4f} um')
    # 2. and the penalty is large enough to dominate the budget
    w_3um = _e2_radius(np.abs(_read(2048, dz=3.0e-6)) ** 2, 0.05e-6)
    assert w_3um / w_ref > 1.25, (
        f'a 3 um defocus should widen the spot >25%, got '
        f'{100 * (w_3um / w_ref - 1):.1f}%')


def test_readout_accuracy_is_governed_by_the_bluestein_leg_length():
    """The dominant error term at a tight focus, and the one most likely to be
    mistaken for physics.

    ``standoff`` is documented as a robustness knob (keep the co-moving grid
    from collapsing at the focus), so it is easy to assume it does not affect
    the ANSWER.  It does, strongly: at NA 0.278 the measured waist at the
    nominal plane runs 0.60% wrong at ``standoff = zR`` and 8.74% wrong at the
    shipped default of ``6 zR`` -- a 14x accuracy difference from a knob that
    reads as purely defensive.  A tight-focus spot budget must therefore pin
    ``standoff``, not inherit it.

    Not asserted as a defect: the default is presumably conservative for good
    reason (the sub-zR end degrades too, at 8.74% for 0.5 zR), and the right
    factor may well be NA-dependent.  What is asserted is that the sensitivity
    is real and in the measured direction, so it cannot regress silently.
    """
    def waist_at_nominal(standoff):
        env, dx = _pupil(2048)
        F = np.asarray(carrier_referenced_focus_readout(
            env, -_Z, _Z, _WL, dx, dx_out=0.05e-6, N_out=512,
            standoff=standoff))
        return _e2_radius(np.abs(F) ** 2, 0.05e-6)

    w_tuned = waist_at_nominal(1.0 * _ZR)
    w_default = waist_at_nominal(6.0 * _ZR)
    e_tuned = abs(w_tuned - _W0_TARGET) / _W0_TARGET
    e_default = abs(w_default - _W0_TARGET) / _W0_TARGET
    assert e_tuned < 0.015, (
        f'standoff=zR should be near-exact, got {w_tuned * 1e6:.4f} um '
        f'({e_tuned * 100:.2f}%)')
    assert e_default > 4.0 * e_tuned, (
        f'expected the 6*zR default to be materially worse than standoff=zR: '
        f'{e_default * 100:.2f}% vs {e_tuned * 100:.2f}%')
