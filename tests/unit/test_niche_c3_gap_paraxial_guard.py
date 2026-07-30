"""The high-NA-gap guard -- niche C3 (roadmap
ROADMAP_DESIGN121_FULL_CONFIGURATION_2026_07_27 P7).

WHAT P7 ASKED FOR.  "Inter-group transport is still paraxial
(Sziklas-Siegman) ... there is no high-NA-gap guard, so the next design finds
the edge the hard way", with the QUARTIC SAG PHASE
``phi_sag = k w^4/(8|R|^3)`` proposed as the metric (~7 rad entering design
121's final gap).

WHAT THE CALIBRATION FOUND, and why this module pins what it pins.  Scoring
the shipped :func:`propagate_carrier_referenced` against an INDEPENDENT
band-limited angular-spectrum oracle produced two results that decide the
guard's shape, and both are re-measured here rather than asserted:

* **A leg does not carry ``phi_sag``; it drops the CHANGE in it.**  ``w`` and
  ``|R|`` both scale by ``m = R_out/R_in`` across a Sziklas-Siegman leg, so
  the gap NA is invariant and ``phi_sag`` scales by ``m``.  The difference,

      ``phi_drop = |phi_sag(exit) - phi_sag(entry)| = k z NA^4 / 8``,

  is exactly the Fresnel kernel's own defect ``k z (sqrt(1-a^2) - 1 + a^2/2)``
  at ``a = NA``.  ``test_the_leg_drops_the_change_in_phi_sag_not_phi_sag``
  pins the identity; the full calibration measured -2.1 to -65 EE points at a
  FIXED ``phi_sag`` of 8 rad as the leg length ran 0.02 |R| -> 0.9 |R|.
* **Under the shipping ``carrier_reference='sphere'`` that drop CANCELS
  EXACTLY.**  The chain converts parabola -> exact sphere entering a group and
  back leaving it, and those two conversions differ by
  ``(R_in - R_out) x (parabola - S) = -z x (parabola - S)``; adding the Fresnel
  leg's own ``z(1 + t^2/2)`` gives ``z sqrt(1+t^2)``, the exact tilted-ray path,
  to ALL orders in ``t = r/|R|``.
  ``test_the_sphere_conversions_cancel_the_dropped_quartic`` is the MEASURED
  form of that, and its ``'parabola'`` half is the FAIL-BEFORE: the same leg,
  same grid, same oracle, reads three orders of magnitude worse when the
  conversions are removed.

So the guard trips on the DROPPED quartic (only reachable under the legacy
``'parabola'``) and on the GAP NA (where the diffractive residual the
cancellation leaves first comes off the floor), and merely REPORTS
``phi_sag``.  ``test_phi_sag_alone_does_not_predict_the_cost`` is the
fail-before for that choice: at fixed NA the measured disagreement FALLS as
``phi_sag`` rises, so a guard tripping on large ``phi_sag`` would fire on the
safe configurations and stay silent on the risky ones -- strictly worse than
no guard, which is the failure mode this whole module exists to avoid.

THE 121 MARGIN is pinned leg by leg from the design's own paraxial q-trace
(``test_the_shipped_121_leg_table_stays_silent``): on the shipping defaults
the worst leg sits 4.1x below the gap-NA trip and the dropped-quartic trip
does not apply at all.  Under the LEGACY ``'parabola'`` the guard DOES fire on
two of its legs -- correctly: the library's own audit measured that legacy
triple costing design 121 best-focus EE6 79.7 % against 99.3 %.

Everything here is SELF-CONTAINED: synthetic N-BK7 singlets built inline, no
prescription asset, no ``.zmx``, and the exact oracle is ~15 lines of plain
NumPy FFT written in this file (no lumenairy propagator is used to check a
lumenairy propagator).
"""
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.propagators import carrier as _carrier
from lumenairy.propagators.carrier import (
    _GAP_NA_TOL,
    _GAP_SAG_TOL_DEFAULT,
    _check_gap_paraxial,
    propagate_carrier_referenced,
)

_WL = 1.31e-6
_K = 2.0 * np.pi / _WL


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _sphere_sag(r2, R):
    """EXACT sphere phase-sag matching ``exp(+i k r^2/(2R))`` to 2nd order."""
    aR = abs(R)
    return np.sign(R) * (np.sqrt(aR * aR + r2) - aR)


def _asm(E, z, dx):
    """INDEPENDENT band-limited angular-spectrum step (Matsushima 2009).

    Plain NumPy FFT, deliberately not routed through any lumenairy
    propagator -- the point is to check one implementation against a
    different one."""
    n = E.shape[0]
    fx = np.fft.fftfreq(n, d=dx)
    lf2 = (_WL * fx) ** 2
    arg = 1.0 - (lf2[None, :] + lf2[:, None])
    prop = arg > 0.0
    kz = np.zeros_like(arg)
    np.sqrt(arg, out=kz, where=prop)
    H = np.exp((2j * np.pi * z / _WL) * kz) * prop
    df = 1.0 / (n * dx)
    f_lim = 1.0 / (_WL * np.sqrt((2.0 * df * z) ** 2 + 1.0))
    keep = (np.abs(fx)[None, :] <= f_lim) & (np.abs(fx)[:, None] <= f_lim)
    return np.fft.ifft2(np.fft.fft2(E) * H * keep)


def _leg_disagreement(phi_sag, na, *, z_frac=0.5, reference='sphere',
                      a_ny=0.88, half_widths=2.4, taper=(1.9, 2.3)):
    """Core rms phase disagreement (rad) between the shipped Sziklas-Siegman
    step and the independent ASM oracle, for a converging leg with the given
    entering ``phi_sag`` and gap NA.

    ``reference`` selects the carrier convention the way the chain does:
    ``'sphere'`` stores the residual against the EXACT sphere (the shipping
    default), ``'parabola'`` against the paraxial parabola (the legacy escape
    hatch).  Also returns the oracle's SAMPLING-ADEQUACY figure -- the input
    power fraction above the grid Nyquist -- because an ASM oracle on a grid
    that does not resolve the beam measures itself, not the physics."""
    w0 = 8.0 * phi_sag / (_K * na ** 3)
    R0 = -w0 / na
    z = z_frac * abs(R0)
    m = (R0 + z) / R0
    R_out = R0 + z
    dx = _WL / (2.0 * a_ny)
    n = int(2 * np.ceil(np.ceil(2.0 * half_widths * w0 / dx) / 2))
    while True:                                    # nudge to an even 7-smooth
        t = n
        for p in (2, 3, 5, 7):
            while t % p == 0:
                t //= p
        if t == 1 and n % 2 == 0:
            break
        n += 2
    x = (np.arange(n, dtype=np.float64) - n / 2) * dx
    r2 = x[None, :] ** 2 + x[:, None] ** 2
    amp = np.exp(-r2 / (w0 * w0))
    ta, tb = (taper[0] * w0) ** 2, (taper[1] * w0) ** 2
    amp = amp * np.cos(0.5 * np.pi * np.clip((r2 - ta) / (tb - ta),
                                             0.0, 1.0)) ** 2
    E_in = amp * np.exp(1j * _K * _sphere_sag(r2, R0))

    # sampling adequacy of the oracle grid
    A = np.abs(np.fft.fft2(E_in)) ** 2
    fx = np.fft.fftfreq(n, d=dx)
    a2 = (_WL * fx) ** 2
    a_ny_cos = _WL / (2.0 * dx)
    above = float(A[(a2[None, :] + a2[:, None]) > a_ny_cos ** 2].sum()
                  / A.sum())

    E_ref = _asm(E_in, z, dx)
    if reference == 'sphere':
        env = E_in * np.exp(-1j * _K * _sphere_sag(r2, R0))
    else:
        env = E_in * np.exp(-1j * _K * r2 / (2.0 * R0))
    cr = propagate_carrier_referenced(env, R0, z, _WL, dx)
    c = n // 2
    r_s2 = ((np.arange(c, dtype=np.float64) * cr.dx) ** 2)
    if reference == 'sphere':
        u_ss = np.asarray(cr.env)[c, c:]
    else:
        u_ss = (np.asarray(cr.env)[c, c:]
                * np.exp(1j * _K * (r_s2 / (2.0 * R_out)
                                    - _sphere_sag(r_s2, R_out))))
    r_a2 = ((np.arange(c, dtype=np.float64) * dx) ** 2)
    u_ref = E_ref[c, c:] * np.exp(-1j * _K * _sphere_sag(r_a2, R_out))

    # compare on the SS grid's radii (m < 1, so they are the finer set);
    # both residuals are smooth, so a linear interpolation is exact enough
    r_s = np.sqrt(r_s2)
    keep = r_s <= abs(m) * w0
    ur = (np.interp(r_s[keep], np.sqrt(r_a2), u_ref.real)
          + 1j * np.interp(r_s[keep], np.sqrt(r_a2), u_ref.imag))
    us = u_ss[keep]
    wgt = np.abs(ur) ** 2 * r_s[keep]
    d = np.angle(us * np.conj(ur))
    piston = np.angle(np.sum(wgt * np.exp(1j * d)) / wgt.sum())
    d = np.angle(np.exp(1j * (d - piston)))
    return float(np.sqrt(np.sum(wgt * d ** 2) / wgt.sum())), above, n


def _fire(w, R, z, *, action='warn', tol=_GAP_SAG_TOL_DEFAULT, sphere=True,
          where='leg'):
    """Run the guard and return ``(stats, [messages])``."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        stats = _check_gap_paraxial(w, R, z, (R + z) / R, _WL, where, action,
                                    tol, sphere)
    return stats, [str(r.message) for r in rec
                   if issubclass(r.category, RuntimeWarning)]


# design 121's own legs, from the paraxial q-trace off the real prescription:
# (name, w entering the leg, R entering the leg, leg length), all metres.
_D121_LEGS = (
    ('source -> S3-S4', 0.2085e-3, 2.0007e-3, 45.906284e-3),
    ('S3-S4 -> S5-S7', 5.6256e-3, 143.3724e-3, 10.0e-3),
    ('S18-S20 -> S21-S22', 5.9805e-3, -263.1942e-3, 32.47866e-3),
    ('S21-S22 -> S23-S24', 4.8120e-3, -60.1480e-3, 8.677925e-3),
    ('S23-S24 -> S25-S27', 3.6004e-3, -24.4625e-3, 3.323294e-3),
)


# ===========================================================================
# the metric itself
# ===========================================================================
def test_the_leg_drops_the_change_in_phi_sag_not_phi_sag():
    """``phi_drop = k z NA^4/8`` IS ``|phi_sag(exit) - phi_sag(entry)|``.

    This is the identity the guard is built on, and it is why a guard on
    ``phi_sag`` alone cannot work: at a FIXED ``phi_sag`` the drop is set by
    the leg LENGTH, and runs to zero for a short leg."""
    w, R = 3.6004e-3, -24.4625e-3
    drops = []
    for z_frac in (0.02, 0.1, 0.5, 0.9):
        z = z_frac * abs(R)
        st, _ = _fire(w, R, z, action='ignore')
        assert st['gap_phi_sag_in'] == pytest.approx(6.882, rel=2e-3)
        # the entering phi_sag is the SAME on every row ...
        assert st['gap_phi_drop'] == pytest.approx(
            abs(st['gap_phi_sag_out'] - st['gap_phi_sag_in']), rel=1e-9)
        assert st['gap_phi_drop'] == pytest.approx(
            _K * z * st['gap_na'] ** 4 / 8.0, rel=1e-9)
        drops.append(st['gap_phi_drop'])
    # ... while the DROP spans two orders of magnitude across the same rows
    assert drops[0] == pytest.approx(0.1376, rel=2e-3)
    assert drops[-1] == pytest.approx(6.194, rel=2e-3)
    assert drops[-1] / drops[0] == pytest.approx(45.0, rel=1e-2)


def test_the_gap_na_is_invariant_along_the_leg():
    """``w`` and ``|R|`` both scale by ``m``, so ``phi_sag`` scales by ``m``
    and ``NA`` does not.  Reported ``gap_na`` is therefore the leg's, not an
    end's."""
    w, R, z = 3.6004e-3, -24.4625e-3, 3.323294e-3
    st, _ = _fire(w, R, z, action='ignore')
    m = (R + z) / R
    assert st['gap_na'] == pytest.approx(w / abs(R), rel=1e-12)
    assert st['gap_phi_sag_out'] == pytest.approx(
        abs(m) * st['gap_phi_sag_in'], rel=1e-12)


# ===========================================================================
# the MEASURED physics the threshold rests on
# ===========================================================================
@pytest.mark.slow
def test_the_sphere_conversions_cancel_the_dropped_quartic():
    """The shipping ``'sphere'`` reference cancels the leg's dropped quartic;
    the legacy ``'parabola'`` does not.

    Same leg, same grid, same independent ASM oracle -- only the carrier
    convention differs.  The ``'parabola'`` row is the FAIL-BEFORE for this
    whole guard: it is the configuration whose cost P7 was worried about, and
    it is three orders of magnitude worse than the shipping one."""
    phi, na = 2.0, 0.35                      # drop = 1.0 rad at z = |R|/2
    rms_sphere, above_s, n = _leg_disagreement(phi, na, reference='sphere')
    rms_parab, above_p, _ = _leg_disagreement(phi, na, reference='parabola')
    # SAMPLING ADEQUACY: the oracle's Nyquist direction cosine is
    # 0.88 / (0.35/sqrt(1+0.35^2)) = 2.66x the beam's marginal-ray sine, and
    # the measured input power above that Nyquist is ~1e-11 -- eight decades
    # below the smaller of the two numbers compared.  Without this the rows
    # would be measuring the grid.
    assert above_s < 1e-8 and above_p < 1e-8
    assert n >= 256
    assert rms_sphere < 5.0e-3, rms_sphere        # measured 8.8e-4 rad
    assert rms_parab > 0.10, rms_parab            # measured 1.9e-1 rad
    assert rms_parab / rms_sphere > 50.0


@pytest.mark.slow
def test_phi_sag_alone_does_not_predict_the_cost():
    """FAIL-BEFORE for the guard's SHAPE.

    Under the shipping convention the measured disagreement FALLS as
    ``phi_sag`` rises at fixed NA (a larger ``phi_sag`` at fixed NA is a
    larger, more geometric beam).  A guard tripping on large ``phi_sag``
    would therefore fire hardest on the SAFEST rows -- which is why this one
    does not."""
    rms = []
    for phi in (2.0, 4.0, 8.0):
        r, above, _ = _leg_disagreement(phi, 0.45, reference='sphere')
        assert above < 1e-8
        rms.append(r)
    assert rms[0] > rms[1] > rms[2], rms
    # ~1/phi_sag, i.e. strictly ANTI-correlated with the proposed metric
    assert rms[0] / rms[2] > 3.0, rms


# ===========================================================================
# design 121 -- nothing accepted becomes refused
# ===========================================================================
def test_the_shipped_121_leg_table_stays_silent():
    """Every leg of design 121, on the SHIPPING defaults, with margin."""
    worst_na = 0.0
    worst_phi = 0.0
    for name, w, R, z in _D121_LEGS:
        st, msgs = _fire(w, R, z, where=name)
        assert msgs == [], (name, msgs)
        worst_na = max(worst_na, st['gap_na'])
        worst_phi = max(worst_phi, st['gap_phi_sag_in'],
                        st['gap_phi_sag_out'])
    # the worst leg is the final gap's NA, 4.1x below the gap-NA trip
    assert worst_na == pytest.approx(0.1472, rel=2e-3)
    assert _GAP_NA_TOL / worst_na > 4.0
    # and P7's "~7 rad" is that leg's ENTERING phi_sag -- reported, not tripped
    assert worst_phi == pytest.approx(6.882, rel=2e-3)


def test_the_p7_seven_radian_leg_is_the_final_gap():
    """Pin P7's own number to the leg it came from, so the record and the
    guard cannot drift apart."""
    _n, w, R, z = _D121_LEGS[-1]
    st, msgs = _fire(w, R, z)
    assert st['gap_phi_sag_in'] == pytest.approx(6.882, rel=2e-3)
    assert st['gap_phi_sag_out'] == pytest.approx(5.947, rel=2e-3)
    assert st['gap_phi_drop'] == pytest.approx(0.935, rel=3e-3)
    assert msgs == []


def test_the_legacy_parabola_convention_does_fire_on_121():
    """Not a regression: the legacy triple really does cost design 121
    best-focus EE6 79.7 % against the shipping 99.3 %, and two of its legs
    carry more dropped quartic than the whole tolerance."""
    fired = []
    for name, w, R, z in _D121_LEGS:
        _st, msgs = _fire(w, R, z, sphere=False, where=name)
        if msgs:
            fired.append((name, msgs[0]))
    assert [f[0] for f in fired] == ['source -> S3-S4',
                                     'S23-S24 -> S25-S27']
    assert 'DROPS 3.246 rad of quartic sag' in fired[0][1]
    assert "carrier_reference='parabola' does not put it back" in fired[0][1]


# ===========================================================================
# the trips themselves
# ===========================================================================
def test_the_dropped_quartic_threshold_boundary():
    """Straddle ``gap_sag_tol`` with a leg whose drop is set analytically."""
    na, w = 0.20, 300e-6
    R = -w / na
    # drop = k z NA^4 / 8  ->  z for a target drop
    def _z(drop):
        return drop * 8.0 / (_K * na ** 4)
    lo = _fire(w, R, _z(0.29), sphere=False)
    hi = _fire(w, R, _z(0.31), sphere=False)
    assert lo[0]['gap_phi_drop'] == pytest.approx(0.29, rel=1e-9)
    assert hi[0]['gap_phi_drop'] == pytest.approx(0.31, rel=1e-9)
    assert lo[1] == []
    assert len(hi[1]) == 1
    assert 'gap_sag_tol=0.3' in hi[1][0]
    # and the tolerance is a knob, not a constant
    assert _fire(w, R, _z(0.31), sphere=False, tol=0.5)[1] == []
    assert len(_fire(w, R, _z(0.29), sphere=False, tol=0.1)[1]) == 1
    # tol == 0 disables that trip entirely
    assert _fire(w, R, _z(50.0), sphere=False, tol=0.0)[1] == []


def test_a_genuinely_high_na_gap_fires_even_on_the_shipping_default():
    """The second trip: past NA 0.60 the diffractive residual the sphere
    cancellation leaves comes off the floor (-0.008 EE points at 0.60,
    -0.121 at 0.75, -0.504 at 0.90 in the calibration)."""
    for na, expect in ((0.55, False), (0.65, True), (0.90, True)):
        w = 60e-6
        R = -w / na
        _st, msgs = _fire(w, R, 0.5 * abs(R))
        assert bool(msgs) is expect, (na, msgs)
    _st, msgs = _fire(60e-6, -60e-6 / 0.75, 0.5 * 60e-6 / 0.75)
    assert 'gap NA 0.7500' in msgs[0]
    assert f'above the calibrated envelope {_GAP_NA_TOL}' in msgs[0]
    assert 'phi_sag = k w^4/(8|R|^3) runs' in msgs[0]
    assert 'Design 121' in msgs[0]


def test_the_message_names_the_geometry_and_the_remedies():
    _st, msgs = _fire(60e-6, -80e-6, 40e-6)
    m = msgs[0]
    for token in ('w = 0.0600 mm', '|R| = 0.0800 mm', 'gap NA 0.7500',
                  'leg 0.0400 mm', 'REMEDIES', "on_gap_paraxial='error'",
                  "'ignore' silences it"):
        assert token in m, token


def test_error_and_ignore():
    w, R, z = 60e-6, -80e-6, 40e-6
    with pytest.raises(RuntimeError, match='gap NA'):
        _check_gap_paraxial(w, R, z, (R + z) / R, _WL, 'leg', 'error',
                            _GAP_SAG_TOL_DEFAULT, True)
    st, msgs = _fire(w, R, z, action='ignore')
    assert msgs == []
    # 'ignore' still returns the diagnostic -- it silences, it does not blind
    assert st['gap_na'] == pytest.approx(0.75, rel=1e-12)
    assert st['gap_phi_drop'] > 0.0


def test_the_fail_before_switch():
    """Disabling BOTH trips makes the fired rows go silent, which is what the
    chain did before niche C3.  This is the switch that proves the tests
    above are measuring this guard and not some pre-existing diagnostic."""
    hot = ((60e-6, -80e-6, 40e-6, True),                    # NA 0.75
           (300e-6, -1.5e-3, 1.125e-3, False))              # drop 1.079 rad
    for w, R, z, sphere in hot:
        assert _fire(w, R, z, sphere=sphere)[1] != []
    saved = _carrier._GAP_NA_TOL
    try:
        _carrier._GAP_NA_TOL = np.inf
        for w, R, z, sphere in hot:
            assert _fire(w, R, z, sphere=sphere, tol=np.inf)[1] == []
    finally:
        _carrier._GAP_NA_TOL = saved


# ===========================================================================
# chain wiring: validation, stage diagnostics, byte identity
# ===========================================================================
def _singlet(R1, R2, d, glass, ap, name='s'):
    surfaces = [
        {'radius': R1, 'glass_before': 'air', 'glass_after': glass,
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
        {'radius': R2, 'glass_before': glass, 'glass_after': 'air',
         'conic': 0.0, 'radius_y': None, 'conic_y': None,
         'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': name, 'aperture_diameter': ap,
            'surfaces': surfaces, 'thicknesses': [d]}


_N, _DX, _W = 512, 4.0e-6, 300e-6
_TKW = dict(on_undersample='silent', on_noncollimated='silent')


@pytest.fixture(scope='module')
def _field():
    x = (np.arange(_N) - _N // 2) * _DX
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / _W ** 2).astype(np.complex128)


def _relay(gap=25e-3):
    return [{'prescription': _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 10e-3,
                                      'gA'), 'gap_before': 0.0},
            {'prescription': _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 10e-3,
                                      'gB'), 'gap_before': gap}]


def _chain(field, **kw):
    kw.setdefault('ray_subsample', 8)
    kw.setdefault('n_workers', 4)
    kw.setdefault('traced_kwargs', _TKW)
    kw.setdefault('final_leg', 'paraxial')
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter('always')
        res = la.propagate_traced_carrier_chain(
            field, kw.pop('groups', None) or _relay(), _WL, _DX, **kw)
    return res, [str(r.message) for r in rec
                 if issubclass(r.category, RuntimeWarning)]


def _gap_msgs(msgs):
    return [m for m in msgs
            if 'quartic sag' in m or 'above the calibrated envelope' in m]


def test_the_chain_validates_the_new_knobs(_field):
    with pytest.raises(ValueError, match='on_gap_paraxial'):
        la.propagate_traced_carrier_chain(_field, _relay(), _WL, _DX,
                                          on_gap_paraxial='shout')
    with pytest.raises(ValueError, match='gap_sag_tol'):
        la.propagate_traced_carrier_chain(_field, _relay(), _WL, _DX,
                                          gap_sag_tol=-1.0)
    with pytest.raises(ValueError, match='gap_sag_tol'):
        la.propagate_traced_carrier_chain(_field, _relay(), _WL, _DX,
                                          gap_sag_tol=np.nan)


def test_the_chain_reports_the_leg_per_stage_and_stays_silent(_field):
    """A benign collimated relay: the guard is silent and the numbers are
    still readable off ``stages`` without catching a warning (the same
    contract niche D3 gave ``na_exit``)."""
    res, msgs = _chain(_field)
    assert _gap_msgs(msgs) == []
    legged = [s for s in res.stages if 'gap_na' in s]
    assert len(legged) == 1                      # only gB has a gap_before
    st = legged[0]
    assert set(('gap_phi_sag_in', 'gap_phi_sag_out', 'gap_phi_drop',
                'gap_na')) <= set(st)
    assert st['gap_na'] < 0.01
    assert st['gap_phi_drop'] < 1e-4


def test_the_chain_fires_on_a_legacy_parabola_high_drop_leg(_field):
    """End-to-end wiring: a converging entry carrier at NA 0.20 spending
    0.75 |R| of gap drops 1.08 rad, which the legacy convention keeps."""
    groups = [{'prescription': _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 10e-3,
                                        'g0'), 'gap_before': 1.125e-3}]
    kw = dict(groups=groups, r_in=-1.5e-3, carrier_reference='parabola',
              traced_kwargs=dict(_TKW, amplitude_model='screen',
                                 preserve_input_phase=True))
    _res, msgs = _chain(_field, **kw)
    hot = _gap_msgs(msgs)
    assert len(hot) == 1, msgs
    assert 'DROPS 1.0' in hot[0]
    assert 'groups[0] (g0)' in hot[0]
    # and 'error' promotes the same leg to fatal
    with pytest.raises(RuntimeError, match='quartic sag'):
        _chain(_field, on_gap_paraxial='error', **kw)
    # while 'ignore' silences it
    assert _gap_msgs(_chain(_field, on_gap_paraxial='ignore', **kw)[1]) == []


def test_the_guard_is_diagnostic_only_byte_identical(_field):
    """DEFAULT-PATH BYTE IDENTITY: nothing the guard does may touch a number,
    on a silent leg OR on a firing one."""
    base = _chain(_field)[0]
    for kw in (dict(on_gap_paraxial='ignore'),
               dict(on_gap_paraxial='error'),
               dict(gap_sag_tol=0.0),
               dict(gap_sag_tol=1e6)):
        other = _chain(_field, **kw)[0]
        assert np.array_equal(np.asarray(base.field), np.asarray(other.field))
        assert base.dx == other.dx
        assert base.R == other.R
    # ... and on the leg that DOES fire
    groups = [{'prescription': _singlet(60e-3, -60e-3, 3e-3, 'N-BK7', 10e-3,
                                        'g0'), 'gap_before': 1.125e-3}]
    kw = dict(groups=groups, r_in=-1.5e-3, carrier_reference='parabola',
              traced_kwargs=dict(_TKW, amplitude_model='screen',
                                 preserve_input_phase=True))
    a = _chain(_field, **kw)[0]
    b = _chain(_field, on_gap_paraxial='ignore', **kw)[0]
    assert np.array_equal(np.asarray(a.field), np.asarray(b.field))


def test_byte_identity_against_the_pre_guard_code_path(_field, monkeypatch):
    """The strongest form: neutralise the guard's ENTIRE added code path
    (the per-leg envelope measurement and the check itself) and confirm the
    field is still bit-for-bit what the shipping default produces.  That is
    the code the chain ran before niche C3."""
    base = _chain(_field)[0]
    monkeypatch.setattr(_carrier, '_gap_amp_radius', lambda env, dx: 0.0)
    monkeypatch.setattr(_carrier, '_check_gap_paraxial',
                        lambda *a, **k: {})
    other, msgs = _chain(_field)
    assert _gap_msgs(msgs) == []
    assert not any(k.startswith('gap_') for s in other.stages for k in s)
    assert np.array_equal(np.asarray(base.field), np.asarray(other.field))
    assert base.dx == other.dx and base.R == other.R


def test_the_multi_orchestrator_takes_and_forwards_the_knobs():
    import inspect
    sig = inspect.signature(la.propagate_traced_carrier_chain_multi)
    assert sig.parameters['on_gap_paraxial'].default == 'warn'
    assert sig.parameters['gap_sag_tol'].default == _GAP_SAG_TOL_DEFAULT
    with pytest.raises(ValueError, match='on_gap_paraxial'):
        la.propagate_traced_carrier_chain_multi(
            [{'env': np.ones((8, 8), dtype=np.complex128)}], [], _WL, 1e-6,
            output_grid=dict(dx_out=1e-6, N_out=8), on_gap_paraxial='nope')
    with pytest.raises(ValueError, match='gap_sag_tol'):
        la.propagate_traced_carrier_chain_multi(
            [{'env': np.ones((8, 8), dtype=np.complex128)}], [], _WL, 1e-6,
            output_grid=dict(dx_out=1e-6, N_out=8), gap_sag_tol=-1.0)
