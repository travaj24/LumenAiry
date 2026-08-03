"""``REMAP_STATIONARY_PHASE_FIT_GUARD`` -- the opt-in remedy for the niche-C6
ON-AXIS GHOST, and the CONTRACT that makes it safe to leave in the tree.

WHAT THE GHOST IS (measured on design 121, not reproducible from unit-test
assets -- see docs/audits/APPROXIMATION_AUDIT_POST_C6_2026_07_31.md S3).  The
C6 stationary-phase launch augments every ray direction by ``grad(a_fit)`` of a
general, NON-RADIAL polynomial.  ``_FIT_DISC_OUTSIDE_WEIGHT_REL`` records that
the concentric hard NaN mask on the entrance->exit ray fit is safe only while
"the unconstrained directions of the fit inherit the map's RADIAL SYMMETRY", so
C6 destroys that precondition and the D1 fold returns on the one branch D1 left
alone.  On design 121's on-axis last-group call it costs a spurious annulus at
6.298-7.216 mm exit radius carrying 1.03e-03 of the input power at 33 % of peak
amplitude -- invisible to every encircled-energy metric, which is why it is
pinned with halo metrics there and with a pure CONTRACT here.

WHY THE FLAG IS OPT-IN.  The knob is a conditioning lottery, not a cure: on
design 121 on axis the weighted branch is exactly clean and the mask ghosts, on
a SYNTHETIC singlet at the same scale the mask is exactly clean and the
weighted branch ghosts at 4.5e-03, and the same reversal appears along
``fit_radius_beam_factor`` and ``_FIT_DISC_OUTSIDE_WEIGHT_REL``.  So what is
pinned here is not "the guard is better" -- it is:

  1. with the flag at its DEFAULT the library is byte-identical to the tree
     before it existed (tests 1-2 cover the two ways that can be reached);
  2. the flag is INERT wherever it must be -- C6 disengaged, no engaged
     carrier, a non-``remap`` mode, a DECENTRED beam, no fit-radius
     restriction, and the D1 constant at 0;
  3. when it does act, it acts by routing the CONCENTRIC fit through exactly
     the off-centre branch's restriction -- pinned by construction against a
     run that reaches that branch the other way (a forced null decentre), which
     is what makes it a re-use of D1/D7 rather than a new code path.

CORRECTION 2026-07-31 (docs/audits/C6_FIT_GUARD_DECISION_2026_07_31.md).  An
earlier version of this docstring read the decentre inertness above as "which
is every tilted order".  It is not.  The gate is per GROUP, not per order, and
the reach was measured at chain level: on design 121 the guard moves all six
groups of (0,0), the first TWO of (-1,0), the first of (-2,0) and (-3,0), and
none of (-4,0) or (-4,-2) -- because a group is on the concentric branch
whenever its own beam decentre is <= ``_DECENTRE_GATE_W_FRAC`` (0.05 w), and
the early groups of a mildly tilted order are.  Byte-identity of the CHAIN
therefore holds on (-4,0) and (-4,-2) only.  The element-level pin below
(``test_inert_on_a_decentred_beam``) is unaffected and is what it always
claimed to be: a statement about a decentred BEAM."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_K0 = 2.0 * np.pi / _WL
_N, _DX, _W = 384, 25e-6, 1.5e-3
_RC = -0.06                      # carrier conjugate (converging), metres
_Z = 15e-3
_AP = 12e-3
_ALPHA = 2.0                     # r^4 residual: phase in RADIANS at r = w
_RS = 4


def _flat():
    return {'radius': np.inf, 'glass_before': 'air', 'glass_after': 'air',
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _free_leg(z=_Z, ap=_AP):
    return {'name': 'c6g_freeleg', 'aperture_diameter': ap,
            'surfaces': [_flat(), _flat()], 'thicknesses': [z]}


def _carrier_WLM(x, y, s=_RC):
    sgn = 1.0 if s > 0 else -1.0
    rho = np.sqrt(x * x + y * y + s * s)
    return sgn * (rho - abs(s)), sgn * x / rho, sgn * y / rho


def _field(alpha=_ALPHA, n=_N, dx=_DX, w=_W, cx=0.0, cy=0.0):
    """Gaussian on the carrier sphere plus an ``alpha (r/w)^4`` residual,
    optionally DECENTRED by ``(cx, cy)`` so the off-centre fit branch is
    selected the ordinary way."""
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    W, _l, _m = _carrier_WLM(X, Y)
    r2 = (X - cx) ** 2 + (Y - cy) ** 2
    a = (alpha / _K0) * (r2 / (w * w)) ** 2
    amp = np.exp(-r2 / (w * w))
    return (amp * np.exp(1j * _K0 * (W + a))).astype(np.complex128), X, Y


_BASE_KW = dict(wavelength=_WL, dx=_DX, amplitude_model='ray_density',
                preserve_input_phase='remap', remap_sampling='full',
                parallel_amp=False, on_undersample='silent',
                on_noncollimated='silent', on_aperture_beam='silent',
                ray_subsample=_RS, fit_radius_beam_factor=2.0)


def _run(E, guard, launch=True, carrier=_RC, presc=None, gates=None, **over):
    """One element call with the C6 launch flag, the fit guard and (optionally)
    the niche-C1 decentre gates all controlled and restored."""
    old = (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
           LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC)
    LT.REMAP_STATIONARY_PHASE_LAUNCH = bool(launch)
    LT.REMAP_STATIONARY_PHASE_FIT_GUARD = bool(guard)
    if gates is not None:
        LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC = gates
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            kw = dict(_BASE_KW)
            kw.update(over)
            return np.asarray(la.apply_real_lens_traced(
                E, prescription=presc or _free_leg(), carrier=carrier, **kw))
    finally:
        (LT.REMAP_STATIONARY_PHASE_LAUNCH, LT.REMAP_STATIONARY_PHASE_FIT_GUARD,
         LT._DECENTRE_GATE_PIXELS, LT._DECENTRE_GATE_W_FRAC) = old


@pytest.fixture(scope='module')
def _warm():
    """Push every byte-identity pair past the traced pipeline's warm-up
    boundary (the W9 determinism calibration: in a fresh process one fixed
    input's traced output can change ONCE, at ulp scale, after the first
    calls).  Both members of a compared pair must sit on the same side."""
    E, _X, _Y = _field()
    for _ in range(2):
        _run(E, False)
    return True


# ---------------------------------------------------------------------------
# 1. THE DEFAULT.  Nothing ships turned on.
# ---------------------------------------------------------------------------
def test_flag_defaults_off():
    """The remedy is a lever, not a fix: see the flag's own note for the
    fixture-dependent reversal that keeps it opt-in."""
    assert LT.REMAP_STATIONARY_PHASE_FIT_GUARD is False


def test_default_run_is_the_hard_mask_path(_warm):
    """Reading the module default and passing ``False`` explicitly must be the
    same run -- i.e. the default really is the historical branch."""
    E, _X, _Y = _field()
    a = _run(E, LT.REMAP_STATIONARY_PHASE_FIT_GUARD)
    b = _run(E, False)
    assert np.array_equal(a, b), float(np.abs(a - b).max())


# ---------------------------------------------------------------------------
# 2. INERTNESS -- every place the guard must not reach
# ---------------------------------------------------------------------------
def test_inert_when_the_c6_launch_is_disengaged(_warm):
    """The guard is gated on ``_resid_eik is not None``.  With the C6 launch
    off there is no residual model, so the whole legacy path -- untilted
    included -- must be byte-identical."""
    E, _X, _Y = _field()
    a = _run(E, False, launch=False)
    b = _run(E, True, launch=False)
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_inert_without_an_engaged_carrier(_warm):
    """No carrier -> the de-chirp is the identity -> C6 does not engage -> the
    guard cannot either."""
    E, _X, _Y = _field()
    a = _run(E, False, carrier=None)
    b = _run(E, True, carrier=None)
    assert np.array_equal(a, b), float(np.abs(a - b).max())


@pytest.mark.parametrize('pip', [True, False])
def test_inert_outside_remap(_warm, pip):
    """Only ``preserve_input_phase='remap'`` transports the residual
    geometrically, so only there does C6 engage -- and only there may the fit
    branch move."""
    E, _X, _Y = _field()
    a = _run(E, False, preserve_input_phase=pip)
    b = _run(E, True, preserve_input_phase=pip)
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_inert_on_a_decentred_beam(_warm):
    """THE PIN THAT PROTECTS C6's RECOVERY.  Every tilted order of design 121
    is decentred, so it already takes the weighted branch and the guard has
    nothing to add.  Verified here on a beam decentred by 0.8 w, which clears
    the niche-C1 null-decentre gates."""
    E, _X, _Y = _field(cx=0.8 * _W, cy=-0.5 * _W)
    a = _run(E, False, beam_centre=(0.8 * _W, -0.5 * _W))
    b = _run(E, True, beam_centre=(0.8 * _W, -0.5 * _W))
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_inert_without_a_fit_radius_restriction(_warm):
    """The guard rides on the same ``_beam_fit_radius is not None`` condition
    the D1 branch does -- with no ``fit_radius_beam_factor`` there is no
    beam-tied disc to restrict and nothing changes."""
    E, _X, _Y = _field()
    a = _run(E, False, fit_radius_beam_factor=None)
    b = _run(E, True, fit_radius_beam_factor=None)
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_inert_when_the_d1_weight_is_zero(_warm):
    """``_FIT_DISC_OUTSIDE_WEIGHT_REL = 0`` is documented as the exact
    restoration of the historical hard NaN mask.  It must dominate this flag,
    not race it."""
    E, _X, _Y = _field()
    old = LT._FIT_DISC_OUTSIDE_WEIGHT_REL
    LT._FIT_DISC_OUTSIDE_WEIGHT_REL = 0.0
    try:
        a = _run(E, False)
        b = _run(E, True)
    finally:
        LT._FIT_DISC_OUTSIDE_WEIGHT_REL = old
    assert np.array_equal(a, b), float(np.abs(a - b).max())


# ---------------------------------------------------------------------------
# 3. WHAT IT DOES WHEN IT ACTS
# ---------------------------------------------------------------------------
def test_guard_acts_on_a_concentric_c6_call(_warm):
    """It must not be a no-op where it is meant to bite."""
    E, _X, _Y = _field()
    a = _run(E, False)
    b = _run(E, True)
    assert not np.array_equal(a, b)
    assert float(np.abs(a - b).max()) > 0.0


def test_guard_reproduces_the_offcentre_branch_exactly(_warm):
    """THE STRUCTURAL PIN.  The guard's whole content is "route a C6-engaged
    concentric fit through the branch a DECENTRED beam already takes".  Reach
    that branch the other way -- drive the niche-C1 decentre gates negative so
    a NULL decentre counts as off centre, which leaves the fit DISC identical
    (it is built about ``(0, 0)``) and changes only the restriction method and
    the order -- and the two runs must agree BIT FOR BIT.

    If this ever fails, the guard has grown a code path of its own and the
    D1/D7 evidence no longer covers it.

    Niche C11 (2026-08-03) added ONE pinned attribute to arm ``b`` and nothing
    else.  Driving the gates negative now also wakes the C11 ARBITER, which
    compares both candidates and -- at this NULL decentre, where the two discs
    are identical and the guard is off in that arm -- correctly prefers the
    plain concentric one, so the device stops reaching the branch it names.
    Arm ``a`` is untouched: the guard path has ``_beam_decentred = False``, so
    the arbiter never runs there.  The assertion is the original."""
    E, _X, _Y = _field()
    a = _run(E, True)
    _arb = LT.DECENTRED_FIT_ARBITER
    LT.DECENTRED_FIT_ARBITER = False
    try:
        b = _run(E, False, gates=(-1.0, -1.0))
    finally:
        LT.DECENTRED_FIT_ARBITER = _arb
    assert np.array_equal(a, b), float(np.abs(a - b).max())


def test_guard_raises_the_fit_order_like_d7(_warm):
    """The remedy is the weighted restriction AND D7's order raise; pinning
    only the first would silently ship a different (and measurably worse)
    configuration.  ``decentred_fit_poly_order`` is what carries the raise, so
    forcing it back to ``newton_poly_order`` must change the result."""
    E, _X, _Y = _field()
    a = _run(E, True)
    b = _run(E, True, decentred_fit_poly_order=6)
    assert not np.array_equal(a, b)


def test_guard_does_not_manufacture_energy_on_this_fixture(_warm):
    """A free leg's ray map cannot fold, so neither branch may gain power here.
    This is the null side of the design-121 halo measurement: the guard is
    allowed to move the wavefront, never the energy budget."""
    E, X, Y = _field()
    p_in = float((np.abs(E) ** 2).sum())
    R = np.hypot(X, Y)
    for guard in (False, True):
        F = _run(E, guard)
        pw = np.abs(F) ** 2
        assert 0.99 < float(pw.sum()) / p_in < 1.001, guard
        assert float(pw[R > 3.0 * _W].sum()) / p_in < 1e-6, guard
