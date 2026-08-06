"""Fixes V1 / V2 / V3 / V6 / V8 from ``docs/audits/VERIFY_D1_D11_2026_08_06.md``.

Four independent defects, all in ``lumenairy/propagators/carrier.py``, each
pinned here with the shape of its own failure:

* **V3 (HIGH)** -- the D3 replica guard compared ``N_out*dx_out`` against the
  period and never looked at ``centre_out``.  Because the Bluestein
  reconstruction is periodic in the ABSOLUTE output coordinate, the faithful
  zone is centred on the FIELD's origin, so a narrow window placed one whole
  period off axis returned a bit-identical FULL-AMPLITUDE GHOST of the spot
  with no warning and no refusal.  The condition is per axis
  ``2|centre_out| + N_out*dx_out <= period``.  Pinned on the direct API, on
  the CHAIN (where the offset that reaches the transform is the chief-ray
  RESIDUAL, not the caller's argument) and on the exact readout.
* **V6 (LOW)** -- the bar sits at exactly one period where the measured error
  is still 4-significant-figures correct.  It STAYS a hard refusal (one period
  is the only bar that is correct-by-construction: past it some returned
  samples are literal aliases), but the message must now quote the measured
  margin so a caller can judge it.
* **V1 / V2 (MED)** -- below ``ext = 3.695`` the "extent-following" standoff
  law returned ``f = sqrt(3)`` EXACTLY: a constant in disguise, sitting on a
  contiguous 9-cell band where it lost to BOTH constants it replaced.  The
  sub-threshold branch now trades residual halo clipping against hand-off leg
  error, so ``f`` follows the extent there too.
* **V8 (LOW)** -- ``tilt`` was accepted and silently discarded under
  ``gap_kernel='fresnel'`` (and on the astigmatic path, which forces the
  paraxial kernel).  It now announces.

Everything here is self-contained: synthetic Gaussians on small grids, one
synthetic N-BK7 singlet for the chain, no prescription asset.  Every V3
assertion is a refusal / non-refusal or a comparative geometry check, so no
tolerance on a field value is load-bearing.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from lumenairy.propagators import carrier as C

_WL = 1.0e-6
_RMAG = 5.0e-3                      # converging carrier, focus at +5 mm
_N = 256


# ---------------------------------------------------------------------------
# fixtures / helpers -- the verifier's own geometry
# ---------------------------------------------------------------------------
def _build(NA, ext, n=_N):
    """The verifier's fixture: a Gaussian of 1/e amplitude radius ``NA*|R|``
    on a grid whose half-width is ``ext`` beam radii."""
    w = NA * _RMAG
    half = ext * w
    dx = 2.0 * half / n
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    env = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    w0 = _WL * _RMAG / (np.pi * w)
    return env, dx, w0


def _period(env, dx):
    po = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        C.carrier_referenced_focus_readout(
            env, -_RMAG, _RMAG, _WL, dx, dx_out=1e-7, N_out=8,
            on_replica='ignore', _period_out=po)
    return min(po['period'])


def _read(env, dx, dx_out, N_out, centre_out=(0.0, 0.0), on_replica='error'):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return C.carrier_referenced_focus_readout(
            env, -_RMAG, _RMAG, _WL, dx, dx_out=dx_out, N_out=N_out,
            centre_out=centre_out, on_replica=on_replica)


def _refuses(fn, *a, **kw):
    try:
        fn(*a, **kw)
    except RuntimeError as exc:
        return str(exc)
    return None


# ===========================================================================
# V3 -- the guard must see centre_out
# ===========================================================================
class TestV3ReplicaGuardSeesCentreOut:

    def test_the_verifiers_own_ghost_case_is_refused(self):
        """The exact configuration VERIFY_D1_D11 S2.2 measured: window 0.77
        periods (well under the old bar, so the old guard was silent BY
        CONSTRUCTION) placed one whole period off axis.  With the guard
        waived the readout still returns a bit-identical copy of the on-axis
        peak -- that is the ghost, reproduced here as the fail-before -- and
        the fixed guard must REFUSE it."""
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        dx_out, N_out = w0 / 12.0, 96
        win = N_out * dx_out
        assert win < per, "fixture must sit UNDER the old width-only bar"
        assert win / per < 0.85

        # fail-before, with the guard explicitly waived: a full-amplitude ghost
        on_axis = _read(env, dx, dx_out, N_out, (0.0, 0.0), 'ignore')
        ghost = _read(env, dx, dx_out, N_out, (per, 0.0), 'ignore')
        assert np.max(np.abs(ghost)) == pytest.approx(
            np.max(np.abs(on_axis)), rel=1e-12), (
            "one period off axis must reproduce the on-axis peak exactly -- "
            "that is what makes this a GHOST and not a small error")

        # after: refused
        msg = _refuses(_read, env, dx, dx_out, N_out, (per, 0.0))
        assert msg is not None, "V3: the ghost window must be REFUSED"
        assert 'centre_out' in msg and 'ALIASES' in msg

    def test_the_same_window_on_axis_still_passes(self):
        """The refusal must be about the OFFSET, not a blanket tightening:
        the identical window at ``centre_out = 0`` still returns."""
        env, dx, w0 = _build(0.10, 3.0)
        F = _read(env, dx, w0 / 12.0, 96, (0.0, 0.0), 'error')
        assert np.isfinite(F).all() and np.max(np.abs(F)) > 0.0

    def test_the_message_no_longer_claims_the_zone_follows_centre_out(self):
        """The shipped message asserted 'everything beyond +/-period/2 of
        centre_out is filled with PERIODIC REPLICAS', i.e. that the
        replica-free zone TRAVELS with the window.  It does not."""
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        msg = _refuses(_read, env, dx, w0 / 12.0, 96, (per, 0.0))
        assert msg is not None
        assert 'period/2 of centre_out' not in msg
        assert "FIELD'S OWN ORIGIN" in msg

    def _boundary(self, env, dx, dx_out, N_out, per):
        """Bisect the largest |centre_out| the guard still accepts."""
        lo, hi = 0.0, per
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            if _refuses(_read, env, dx, dx_out, N_out, (mid, 0.0)) is None:
                lo = mid
            else:
                hi = mid
        return lo

    def test_the_refusal_boundary_tracks_the_formula_comparatively(self):
        """The boundary is ``|centre_out| = (period - window)/2``.  Asserted
        COMPARATIVELY -- how the measured boundary MOVES when the window
        changes -- so the test pins the guard's geometry rather than one
        arithmetic evaluation of it.  Halving the window must move the
        boundary out by exactly half the window it gave back."""
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        dx_out = w0 / 12.0
        b = {n: self._boundary(env, dx, dx_out, n, per) for n in (96, 48, 24)}
        for n, meas in b.items():
            pred = 0.5 * (per - n * dx_out)
            assert meas == pytest.approx(pred, rel=1e-6), (
                f"N_out={n}: boundary {meas!r} vs formula {pred!r}")
        # comparative: the MOVEMENT must equal half the window given back
        assert (b[48] - b[96]) == pytest.approx(
            0.5 * (96 - 48) * dx_out, rel=1e-5)
        assert (b[24] - b[48]) == pytest.approx(
            0.5 * (48 - 24) * dx_out, rel=1e-5)
        # ... and the boundary is a strictly DECREASING function of the window
        assert b[24] > b[48] > b[96] > 0.0

    def test_the_offset_costs_twice_its_own_length(self):
        """``2|centre_out|``, not ``|centre_out|``: the window is symmetric
        about the centre, so an offset consumes half-period on the near side
        AND pushes the far edge out by the same amount.  Measured as the
        exchange rate between offset and window."""
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        dx_out = w0 / 12.0
        b96 = self._boundary(env, dx, dx_out, 96, per)
        b48 = self._boundary(env, dx, dx_out, 48, per)
        # d(boundary)/d(window) = -1/2  <=>  the offset enters with weight 2
        rate = (b48 - b96) / ((96 - 48) * dx_out)
        assert rate == pytest.approx(0.5, rel=1e-4)

    def test_a_huge_offset_says_no_window_is_safe(self):
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        msg = _refuses(_read, env, dx, w0 / 12.0, 8, (3.0 * per, 0.0))
        assert msg is not None
        assert 'NO window is faithful at this offset' in msg

    def test_ignore_and_warn_still_escape(self):
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        F = _read(env, dx, w0 / 12.0, 96, (per, 0.0), 'ignore')
        assert np.isfinite(F).all()
        with pytest.warns(RuntimeWarning, match='ALIASES'):
            with warnings.catch_warnings():
                warnings.simplefilter('always')
                C.carrier_referenced_focus_readout(
                    env, -_RMAG, _RMAG, _WL, dx, dx_out=w0 / 12.0, N_out=96,
                    centre_out=(per, 0.0), on_replica='warn')

    def test_the_y_axis_is_guarded_too(self):
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        assert _refuses(_read, env, dx, w0 / 12.0, 96, (0.0, per)) is not None


# ===========================================================================
# V3 -- the CHAIN path.  The chain re-references centre_out to the chief ray,
# so the offset that reaches the transform is the RESIDUAL.  That residual is
# NOT structurally zero: with the DEFAULT centre_out=(0,0) it is the whole
# chief-ray walk.
# ===========================================================================
_CHAIN_WL = 1.31e-6


def _singlet():
    sf = [{'radius': 60e-3, 'glass_before': 'air', 'glass_after': 'N-BK7',
           'conic': 0.0, 'radius_y': None, 'conic_y': None,
           'aspheric_coeffs': None, 'aspheric_coeffs_y': None},
          {'radius': -60e-3, 'glass_before': 'N-BK7', 'glass_after': 'air',
           'conic': 0.0, 'radius_y': None, 'conic_y': None,
           'aspheric_coeffs': None, 'aspheric_coeffs_y': None}]
    return {'name': 'gA', 'aperture_diameter': 10e-3, 'surfaces': sf,
            'thicknesses': [3.0e-3]}


@pytest.fixture(scope='module')
def chain_fixture():
    from lumenairy import TiltedCarrier
    from lumenairy.propagators.carrier import _group_abcd
    g = _singlet()
    A, _B, Cc, _D = _group_abcd(g, _CHAIN_WL)
    fd = -(A / Cc)                      # collimated in -> paraxial back focus
    n, dxg, w = 512, 12e-6, 1.0e-3
    x = (np.arange(n) - n / 2) * dxg
    X, Y = np.meshgrid(x, x, indexing='xy')
    E = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    return dict(g=g, fd=fd, E=E, dx=dxg, TC=TiltedCarrier,
                tkw=dict(on_undersample='silent', on_noncollimated='silent'))


def _chain(fx, L, centre_out, on_replica='error', n_out=128):
    fr = {'dx_out': 0.5e-6, 'N_out': n_out, 'centre_out': centre_out,
          'on_replica': on_replica}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return C.propagate_traced_carrier_chain(
            fx['E'], [fx['g']], r_in=fx['TC'](np.inf, L, 0.0, 0.0, 0.0),
            wavelength=_CHAIN_WL, dx=fx['dx'], final_distance=fx['fd'],
            focus_readout=fr, final_leg='paraxial', traced_kwargs=fx['tkw'])


def _chain_state(fx, L):
    res = _chain(fx, L, (0.0, 0.0), 'ignore', n_out=16)
    st = res.stages[-1]
    return float(st['x_c']), min(st['readout_period'])


class TestV3ChainScope:

    def test_the_chain_residual_is_not_structurally_zero(self, chain_fixture):
        """VERIFY_D1_D11 left this open ('I did NOT construct a chain
        configuration that trips it').  It is trippable: the re-referencing is
        ``centre_out - chief``, which is ZERO only if the caller happens to
        centre the window on the chief ray.  At the DEFAULT ``(0, 0)`` the
        residual IS the walk, here 2.8 mm = 3.3 periods."""
        x_c, per = _chain_state(chain_fixture, 0.046)
        assert abs(x_c) > 2.0e-3
        assert abs(x_c) / per > 3.0, (
            "fixture must walk several periods for this to be a real hole")

    def test_a_walking_chief_ray_gives_a_full_amplitude_ghost(
            self, chain_fixture):
        """Fail-before on the chain, guard waived: a window one WHOLE period
        from the chief ray returns a bit-identical copy of the real spot."""
        x_c, per = _chain_state(chain_fixture, 0.046)
        true_ = _chain(chain_fixture, 0.046, (x_c, 0.0), 'ignore')
        ghost = _chain(chain_fixture, 0.046, (x_c + per, 0.0), 'ignore')
        assert np.max(np.abs(ghost.field)) == pytest.approx(
            np.max(np.abs(true_.field)), rel=1e-12)

    def test_the_chain_now_refuses_the_ghost_and_keeps_the_real_window(
            self, chain_fixture):
        x_c, per = _chain_state(chain_fixture, 0.046)
        # on the chief ray: residual 0, faithful, returns
        good = _chain(chain_fixture, 0.046, (x_c, 0.0), 'error')
        assert np.max(np.abs(good.field)) > 0.0
        # one period away, and at the default on-axis window: refused
        for cen in ((x_c + per, 0.0), (0.0, 0.0)):
            with pytest.raises(RuntimeError, match='ALIASES'):
                _chain(chain_fixture, 0.046, cen, 'error')

    def test_an_untilted_chain_is_untouched(self, chain_fixture):
        """L = 0 puts the chief ray on axis, so the residual is 0 and the
        historical default window still returns -- the guard cannot have been
        bought by making every chain refuse."""
        out = _chain(chain_fixture, 0.0, (0.0, 0.0), 'error')
        assert np.max(np.abs(out.field)) > 0.0


# ===========================================================================
# V3 -- the EXACT readout weighs the CHIEF-RAY RESIDUAL, not the argument
# ===========================================================================
class TestV3ExactReadout:

    @staticmethod
    def _env(n=256, dx=4e-6, w=60e-6, cx=0.2e-3):
        x = (np.arange(n) - n / 2) * dx
        X, Y = np.meshgrid(x, x, indexing='xy')
        return np.exp(-((X - cx) ** 2 + Y ** 2) / w ** 2).astype(
            np.complex128), dx, cx

    def _call(self, env, dx, cx, centre_out, on_replica='error', n_out=64):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return C.carrier_referenced_exact_focus_readout(
                env, -20e-3, 20e-3, _WL, dx, dx_out=0.3e-6, N_out=n_out,
                centre=(cx, 0.0), centre_out=centre_out,
                on_replica=on_replica)

    def test_on_the_chief_ray_costs_no_period_however_far_off_axis(self):
        env, dx, cx = self._env()
        F = self._call(env, dx, cx, (cx, 0.0), 'error')
        assert np.max(np.abs(F)) > 0.0

    def test_one_period_off_the_chief_ray_is_refused(self):
        env, dx, cx = self._env()
        po = {}
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            C.carrier_referenced_exact_focus_readout(
                env, -20e-3, 20e-3, _WL, dx, dx_out=0.3e-6, N_out=16,
                centre=(cx, 0.0), centre_out=(cx, 0.0),
                on_replica='ignore', _period_out=po)
        per = min(po['period'])
        with pytest.raises(RuntimeError) as ei:
            self._call(env, dx, cx, (cx + per, 0.0), 'error')
        assert 'RESIDUAL' in str(ei.value)


# ===========================================================================
# V6 -- the bar stays hard, but the message must carry the measured margin
# ===========================================================================
class TestV6TheBarIsHardAndTheMarginIsQuoted:

    def test_the_guard_still_raises_just_past_one_period(self):
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        n = 256
        assert _refuses(_read, env, dx, per * 1.0000001 / n, n) is not None
        assert _read(env, dx, per * 0.98 / n, n) is not None

    def test_the_refusal_quotes_the_measured_overshoot(self):
        env, dx, w0 = _build(0.10, 3.0)
        per = _period(env, dx)
        n = 256
        msg = _refuses(_read, env, dx, per * 1.10 / n, n)
        assert msg is not None
        assert 'm over' in msg and 'sample(s) per edge' in msg
        # the quoted alias count must GROW with the overshoot -- a real
        # measurement, not a fixed sentence
        def n_alias(ratio):
            m = _refuses(_read, env, dx, per * ratio / n, n)
            return int(m.split('about ')[1].split(' sample')[0])
        assert n_alias(1.02) < n_alias(1.10) < n_alias(1.50)

    def test_the_disposition_is_still_error_by_default(self):
        """V6 must NOT have been closed by downgrading the guard."""
        import inspect
        for fn in (C.carrier_referenced_focus_readout,
                   C.carrier_referenced_exact_focus_readout):
            assert inspect.signature(fn).parameters[
                'on_replica'].default == 'error'


# ===========================================================================
# V1 / V2 -- the small-extent standoff branch
# ===========================================================================
def _resolved_f(NA, ext):
    """``standoff / zR`` in the RESOLVER's own zR, as the library picks it."""
    env, dx, _w0 = _build(NA, ext)
    w_env = C._envelope_amp_radius(env, dx, dx)
    w0r = _WL * _RMAG / (math.pi * w_env)
    zRr = math.pi * w0r * w0r / _WL
    s = C._default_focus_standoff(env, -_RMAG, _RMAG, _WL, dx)
    return s / zRr, (0.5 * _N * dx) / w_env


_F_CAP = math.sqrt(C._FOCUS_STANDOFF_WAIST_GROWTH ** 2 - 1.0)
_SAT = _F_CAP / math.sqrt(1.0 + _F_CAP * _F_CAP)
_EXT_THRESHOLD = C._FOCUS_STANDOFF_MARGIN / _SAT


class TestV1V2SmallExtentStandoff:

    def test_the_old_law_was_a_constant_below_the_threshold(self):
        """Fail-before, evaluated in-line from the module's own constants:
        the SHIPPED closed form ``f = m_req/sqrt(ext^2 - m_req^2)`` with
        ``m_req = min(M, sat*ext)`` collapses to EXACTLY ``f_cap`` for every
        extent under 3.695 -- the 'derived, extent-following' law was one more
        constant multiple of the Rayleigh range down there."""
        vals = []
        for ext in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 3.69):
            _f, ext_eff = _resolved_f(0.15, ext)
            m_req = min(C._FOCUS_STANDOFF_MARGIN, _SAT * ext_eff)
            vals.append(m_req / math.sqrt(ext_eff ** 2 - m_req ** 2))
        assert ext_eff < _EXT_THRESHOLD
        assert all(v == pytest.approx(_F_CAP, rel=1e-12) for v in vals), (
            "the fail-before witness itself is wrong if this is not constant")

    def test_f_actually_varies_with_extent_below_the_threshold(self):
        """The point of the fix: NO CONSTANT IN DISGUISE.  ``f`` must follow
        the extent in the sub-threshold band, and by a wide margin -- not by a
        rounding error."""
        fs = {}
        for ext in (1.8, 2.0, 2.2, 2.5, 3.0, 3.5):
            f, ext_eff = _resolved_f(0.15, ext)
            assert ext_eff < _EXT_THRESHOLD
            fs[ext] = f
        seq = [fs[e] for e in (1.8, 2.0, 2.2, 2.5, 3.0, 3.5)]
        # (the two widest cells both sit on the f_cap floor, so allow a ulp)
        assert all(a >= b * (1.0 - 1e-12) for a, b in zip(seq, seq[1:])), (
            f"f must not increase with extent: {seq}")
        assert fs[1.8] / fs[3.5] > 2.0, (
            f"f must SPREAD with the extent, got {fs}")
        assert len(set(round(v, 6) for v in seq)) >= 4

    def test_the_leg_is_never_shortened_relative_to_the_old_law(self):
        """The trade may only LENGTHEN: a shorter leg would walk into the
        near-focus bridge and into a Bluestein period too short for the
        caller's window."""
        for NA in (0.05, 0.10, 0.15, 0.278, 0.35):
            for ext in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5):
                f, _ee = _resolved_f(NA, ext)
                assert f >= _F_CAP * (1.0 - 1e-9), (NA, ext, f)

    def test_the_reachable_margin_branch_is_untouched(self):
        """Above the threshold the shipped closed form must be returned
        EXACTLY -- the fix is confined to the branch that was constant."""
        for NA in (0.05, 0.15, 0.35):
            for ext in (4.0, 6.0, 10.0):
                f, ext_eff = _resolved_f(NA, ext)
                assert ext_eff >= _EXT_THRESHOLD
                m = C._FOCUS_STANDOFF_MARGIN
                assert f == pytest.approx(
                    m / math.sqrt(ext_eff ** 2 - m * m), rel=1e-9)

    def test_the_leg_stays_inside_the_reachable_margin_asymptote(self):
        """``f`` is capped where the containment margin has reached
        ``_FOCUS_STANDOFF_ASYMPTOTE_FRAC`` of ``ext``; past that no leg buys
        containment and only the hand-off error grows."""
        frac = C._FOCUS_STANDOFF_ASYMPTOTE_FRAC
        f_max = frac / math.sqrt(1.0 - frac * frac)
        for NA in (0.05, 0.10, 0.35):
            for ext in (1.2, 1.5, 1.8, 2.0, 2.5, 3.0):
                f, _ee = _resolved_f(NA, ext)
                assert f <= f_max * (1.0 + 1e-9)

    def test_the_near_focus_bridge_route_is_left_alone(self):
        """When the leg at ``f_cap`` already routes through the through-waist
        ASM bridge, the clipping-vs-leg trade does not describe it and the
        resolver must not extend: measured, extending those cells cost 1.9x."""
        env, dx, _w0 = _build(0.15, 1.5)
        w_env = C._envelope_amp_radius(env, dx, dx)
        w0r = _WL * _RMAG / (math.pi * w_env)
        zRr = math.pi * w0r * w0r / _WL
        # the fixture must genuinely be a bridge cell, or the test is vacuous
        assert C._near_focus_needs_bridge(
            env, -_RMAG, -(_F_CAP * zRr), _WL, dx, dx)
        f, _ee = _resolved_f(0.15, 1.5)
        assert f == pytest.approx(_F_CAP, rel=1e-9)

    def test_the_law_is_still_independent_of_the_requested_window(self):
        """The resolver sets ACCURACY from the beam geometry; the window is
        the replica guard's job.  A field that depended on how wide a window
        it was viewed through would break the K=1 chain contract."""
        env, dx, w0 = _build(0.15, 2.0)
        # end to end: the Bluestein PERIOD is linear in the standoff, so an
        # identical period across a 6x change of requested window proves the
        # resolver never looked at the window.
        pers = []
        for n_out in (16, 48, 96):
            po = {}
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                C.carrier_referenced_focus_readout(
                    env, -_RMAG, _RMAG, _WL, dx, dx_out=w0 / 12.0,
                    N_out=n_out, on_replica='ignore', _period_out=po)
            pers.append(po['period'])
        assert pers[0] == pers[1] == pers[2]


# ===========================================================================
# V8 -- tilt under a paraxial gap kernel
# ===========================================================================
def _tenv(n=128, dx=2e-6, w=40e-6):
    x = (np.arange(n) - n / 2) * dx
    X, Y = np.meshgrid(x, x, indexing='xy')
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128), dx


def _calls(R, gap_kernel, tilt):
    env, dx = _tenv()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter('always')
        out = C.propagate_carrier_referenced(
            env, R, 5e-3, _WL, dx, gap_kernel=gap_kernel, tilt=tilt)
    return out, [str(w.message) for w in wl
                 if issubclass(w.category, RuntimeWarning)]


class TestV8TiltUnderAParaxialKernel:

    @pytest.mark.parametrize('R', [np.inf, -0.2, 0.5])
    def test_fail_before_the_tilt_really_is_inert(self, R):
        """The drop itself is real and unchanged -- the fix ANNOUNCES it, it
        does not secretly start honouring it (which would silently change the
        legacy kernel)."""
        a, _ = _calls(R, 'fresnel', (0.06, -0.04))
        b, _ = _calls(R, 'fresnel', (0.0, 0.0))
        assert np.array_equal(np.asarray(a.env), np.asarray(b.env))

    @pytest.mark.parametrize('R', [np.inf, -0.2, 0.5])
    def test_fresnel_plus_tilt_now_announces_once(self, R):
        _out, wl = _calls(R, 'fresnel', (0.06, -0.04))
        assert len(wl) == 1, wl
        msg = wl[0]
        for token in ("gap_kernel='fresnel'", 'tilt=', 'INERT',
                      "gap_kernel='auto'"):
            assert token in msg, (token, msg)

    @pytest.mark.parametrize('R', [np.inf, -0.2, 0.5])
    def test_zero_tilt_under_fresnel_is_silent(self, R):
        _out, wl = _calls(R, 'fresnel', (0.0, 0.0))
        assert wl == []

    @pytest.mark.parametrize('kernel', ['auto', 'exact'])
    def test_the_exact_kernel_is_silent_and_honours_the_tilt(self, kernel):
        _out, wl = _calls(-0.2, kernel, (0.06, -0.04))
        assert wl == []
        a, _ = _calls(-0.2, kernel, (0.06, -0.04))
        b, _ = _calls(-0.2, kernel, (0.0, 0.0))
        assert not np.array_equal(np.asarray(a.env), np.asarray(b.env))

    def test_the_astigmatic_path_announces_too(self):
        """It forces the separable PARAXIAL kernel whatever ``gap_kernel``
        says ('exact' is already refused there) and never receives ``tilt``
        at all."""
        env, dx = _tenv()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            C.propagate_carrier_referenced(
                env, (-0.2, -0.3), 5e-3, _WL, dx, gap_kernel='auto',
                tilt=(0.06, -0.04))
        msgs = [str(w.message) for w in wl
                if issubclass(w.category, RuntimeWarning)
                and 'INERT' in str(w.message)]
        assert len(msgs) == 1, msgs
        assert 'ASTIGMATIC' in msgs[0]

    def test_an_equal_radii_tuple_routes_to_the_scalar_path(self):
        """(R, R) is the scalar path, so the exact kernel applies and there is
        nothing to announce."""
        env, dx = _tenv()
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter('always')
            C.propagate_carrier_referenced(
                env, (-0.2, -0.2), 5e-3, _WL, dx, gap_kernel='auto',
                tilt=(0.06, -0.04))
        assert [w for w in wl if 'INERT' in str(w.message)] == []
