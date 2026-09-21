"""VERIFY-WP-C3 -- the decisions this verification had to take before it could
accept ``transport='collins'`` as the carrier chain's default.

WP-C3 makes the chain's focus readout RESOLVE its quadrature on
``_collins_readout_k1 <= 1``.  Re-measuring that resolution (report
``docs/audits/AUDIT_ADVERSARIAL_EXHAUSTIVE_2026_09_11/fixes/VERIFY_WP-C3.md``)
turned up three properties of the routing CONDITION that nothing in the
shipped suite states, and two contract gaps.  Everything here is measured on
the running build; no number is copied from a report.

THE CONDITION IS NOT CONTINUOUS.  ``_collins_readout_k1`` is

    K1 = space_term + angle_term
    space_term = 2 dx |A| r / (lambda |B|)
    angle_term = 2 dx theta / lambda

and BOTH ``r`` and ``theta`` come from ``_collins_containment_radius``, which
returns ``d[order[i]]`` -- a GRID COORDINATE, with no interpolation between
samples.  ``theta`` is therefore read off ``np.fft.fftfreq(N, d=dx)``, whose
outermost bin for even ``N`` is exactly ``1/(2 dx)``, so

    angle_term in {2j/N : j = 0 .. N/2},   max exactly 1.0.

Two consequences the shipped documentation does not draw:

* the route is a threshold on a STAIRCASE whose step is ``2/N``, so a margin
  quoted below ``2/N`` is not a margin -- ``docs/TESTING_STANDARDS.md`` shape
  S4/S5, one level up;
* ``angle_term == 1.0`` exactly whenever the envelope's angular support is
  GRID-CLIPPED (``_collins_containment_radius`` saturates at the outermost
  sample by design), and then ``K1 = 1 + space_term > 1`` for every
  non-degenerate leg -- the one-step readout is unreachable on such a field at
  ANY grid and ANY final distance, which refining ``dx`` does not change.

The last class carried the contract gaps this verification FILED as STRICT
xfails, the repository's own instrument for a gap a verification finds and
does not fix (``test_v4_14_0_dispatcher_pin_apply_lens.py``: "the xfail
markers will turn into ``passed`` results without any test change", and
``xfail_strict`` is on, so closing the gap turns the marker itself red and
forces its removal).  WP-C3 ROUND 2 (2026-09-20) closed all five of them --
D1, D2, D4, D5 and D6 -- so every marker is GONE and each id now runs as an
ordinary assertion, with a dated CLOSED-BY comment above it naming what
changed.  The class keeps its name because the ids are still the verifier's
claims, asserted the same way; nothing in it is skipped or xfailed.
"""

from __future__ import annotations

import hashlib
import warnings

import numpy as np
import pytest

from lumenairy.propagators import carrier as C

LAM = 1.31e-6

#: The routing bar itself.  Not a tolerance -- it is the Nyquist statement
#: ``_collins_readout_k1``'s docstring is written about, and the number the
#: route block compares against.
_K1_BAR = 1.0


def _gauss(n, dx, w):
    x = (np.arange(n) - n // 2) * dx
    xx, yy = np.meshgrid(x, x, indexing='ij')
    return np.exp(-(xx ** 2 + yy ** 2) / w ** 2).astype(np.complex128)


def _decompose(env, R, z, dx, lam=LAM):
    """``_collins_readout_k1`` split into the two terms it is the sum of,
    measured through the library's own helpers so the split cannot drift from
    what the library computes."""
    r_x, _r_y, th_x, _th_y = C._collins_input_box(
        env, dx, dx, lam, C._COLLINS_TAIL_FRAC)
    rx, _ry, _ = C._parse_carrier(R, '_decompose')
    a, b, _c, _d = C._collins_envelope_abcd(rx, z, np.inf)
    angle = 2.0 * float(dx) * th_x / lam
    space = 2.0 * float(dx) * abs(a) * r_x / (abs(b) * lam)
    return angle, space, angle + space


def _relay_readout_input(n, monkeypatch, z=8e-3):
    """The array WP-B4's two-group relay ACTUALLY hands
    ``_collins_readout_k1``, captured from the chain's own call, together with
    the K1 the stage publishes.

    IT IS NOT the field a ``final_distance=0`` run returns.  MEASURED
    2026-09-20 on this relay: the two arrays differ at relative L2 1.4149 on
    every grid, and although the published K1 and the reconstructed one agree
    at N = 256 and N = 512 (both saturate), at N = 1024 they read
    **21.4224837152085** and **21.455686840208497** -- apart by exactly 17
    angle quanta of ``2/N``.  Reconstructing the readout's input from a
    ``final_distance=0`` run is therefore not a way to read this condition,
    and this helper exists so nothing here does it.
    """
    from tests.unit.test_audit2609_b4_collins_transport import (
        _CHAIN_TKW, _chain_fixture)
    env0, dx0, r_in, groups = _chain_fixture()
    dx = dx0 * env0.shape[0] / n
    seen = {}
    real = C._collins_readout_k1

    def _spy(env, R, zz, wl, dxx, dyy):
        out = real(env, R, zz, wl, dxx, dyy)
        seen.update(env=np.array(env), R=R, z=zz, dx=float(dxx), k1=out)
        return out

    monkeypatch.setattr(C, '_collins_readout_k1', _spy)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = C.propagate_traced_carrier_chain(
            _gauss(n, dx, 4.5e-3), groups, LAM, dx, r_in=r_in,
            ray_subsample=16, n_workers=1, traced_kwargs=_CHAIN_TKW,
            final_leg='paraxial', final_distance=z,
            focus_readout=dict(dx_out=0.5e-6, N_out=64), transport='collins')
    assert seen, 'the chain never consulted the routing condition'
    return seen, res.stages[-1]


def _oversampled_chain(n=512, dx=8e-6, w=0.30e-3, f=300e-3):
    """A chain whose exit envelope is NOT grid-clipped, so its readout can
    actually take the one-step Collins route.  Returns the call's fixed
    arguments; the caller supplies ``final_distance`` and ``focus_readout``."""
    from tests.unit.test_audit2609_b4_collins_transport import (
        _CHAIN_TKW, _singlet)
    presc = _singlet(2 * f, -2 * f, 3e-3, 'N-BK7', 6e-3, 'p')
    return dict(
        env=_gauss(n, dx, w), groups=[{'prescription': presc,
                                       'gap_before': 10e-3}],
        dx=dx, kw=dict(r_in=np.inf, ray_subsample=16, n_workers=1,
                       traced_kwargs=_CHAIN_TKW, final_leg='paraxial'))


def _run(cfg, final_distance, focus_readout, transport):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return C.propagate_traced_carrier_chain(
            cfg['env'], cfg['groups'], LAM, cfg['dx'],
            final_distance=final_distance, focus_readout=focus_readout,
            transport=transport, **cfg['kw'])


def _sha(a):
    return hashlib.sha256(
        np.ascontiguousarray(a, dtype=np.complex128).tobytes()).hexdigest()


class TestTheRouteConditionIsAStaircase:
    """``_collins_readout_k1 <= 1`` is a threshold on a QUANTISED quantity.

    Both halves are derived, not fitted: the angle half is read off
    ``fftfreq``'s own bins and the containment helper returns a bin coordinate,
    so the assertions below are exact rational statements about the grid and
    carry no cross-build spread at all.
    """

    def test_the_angle_term_is_bounded_by_one_and_quantised_in_two_over_N(
            self):
        """``angle_term = 2 dx theta/lambda`` lives in ``{2j/N}`` and never
        exceeds 1.

        WHY IT MATTERS: it is one of the two summands of the routing
        condition, and it is the one that saturates.  A future change that
        interpolated the containment radius between bins, or that measured
        theta on a padded spectrum, would break both statements -- and would
        silently turn a discrete route decision into a continuous one without
        any value test noticing.
        """
        n, dx = 256, 4.0e-6
        step = 2.0 / n
        seen = set()
        for w_um in (25.0, 40.0, 60.0, 90.0, 140.0, 200.0, 300.0, 420.0):
            env = _gauss(n, dx, w_um * 1e-6)
            angle, _space, _k1 = _decompose(env, -40e-3, 6e-3, dx)
            assert angle <= 1.0, (
                f'the angular support exceeded the grid Nyquist: '
                f'angle_term={angle!r} at w={w_um} um.  fftfreq(N, dx)\'s '
                f'outermost bin is exactly 1/(2 dx), so this is impossible '
                f'unless the measurement stopped coming off the grid.')
            j = angle / step
            assert abs(j - round(j)) < 1e-9, (
                f'angle_term={angle!r} is not an integer multiple of '
                f'2/N={step!r} (j={j!r}) -- the containment radius stopped '
                f'being a grid coordinate.')
            seen.add(round(j))
        assert len(seen) >= 4, (
            f'the ladder did not move the angular support at all '
            f'(bins seen: {sorted(seen)}); it cannot say anything about '
            f'quantisation.')

    def test_the_smallest_step_the_condition_can_take_is_two_over_N(self):
        """The minimum non-zero increment of the angle term over a dense scan
        IS ``2/N``.

        This is what makes a quoted route margin readable: a margin below one
        step is not a margin, because the condition cannot take a value in
        between.  WP-C3 reports the design-121 N = 1024 route as sitting
        ``4.2e-04`` below the bar; one step there is ``2/1024 = 1.95e-03``,
        4.7x larger, so that margin is inside the quantum and the stability
        evidence has to be the containment INDEX matching, not the K1 value.
        """
        n, dx = 256, 4.0e-6
        step = 2.0 / n
        angles = sorted({round(_decompose(_gauss(n, dx, w), -40e-3, 6e-3,
                                          dx)[0] / step)
                         for w in np.linspace(20e-6, 500e-6, 97)})
        gaps = np.diff(np.asarray(angles, dtype=np.float64))
        assert angles, 'no readings'
        assert gaps.size, f'only one distinct reading: {angles}'
        assert float(gaps.min()) == 1.0, (
            f'the angle term moved by {float(gaps.min())} bins at its '
            f'smallest -- it is meant to move by exactly one, which is what '
            f'makes 2/N the quantum of the routing condition.  Bins seen: '
            f'{angles}')


class TestAGridClippedExitCannotReachTheOneStepReadout:
    """When the exit envelope's angular support is GRID-CLIPPED the one-step
    Collins readout is unreachable -- at any grid, and at any final distance.

    ``_collins_containment_radius`` "saturates at the outermost sample when the
    grid itself already clipped the tail", its own docstring says, so
    ``angle_term`` is then exactly 1.0 and ``K1 = 1 + space_term``.  Since
    ``space_term > 0`` for every finite leg with power on the grid,
    ``K1 <= 1`` is unsatisfiable.

    Where the support is NOT clipped the space term decides and does fall as
    ``1/dx``, so the shipped reading that K1 "falls only as ``1/dx``"
    (``_collins_readout_k1.__doc__``) is right about the mechanism and wrong
    about the constant -- see VERIFY_WP-C3.md defect D3.  What it is missing
    is the floor: there is no grid at which a CLIPPED envelope's readout
    becomes representable.

    Every reading here is taken from the array the chain ITSELF hands the
    condition, never from a ``final_distance=0`` reconstruction (see
    :func:`_relay_readout_input`).
    """

    @pytest.mark.parametrize('n', [256, 512])
    def test_the_relay_saturates_its_angular_support_and_stays_over_the_bar(
            self, n, monkeypatch):
        seen, stage = _relay_readout_input(n, monkeypatch)
        angle, space, k1 = _decompose(seen['env'], seen['R'], seen['z'],
                                      seen['dx'])
        assert angle == 1.0, (
            f'expected the WP-B4 relay exit envelope to be angularly '
            f'grid-clipped at N={n} (a 4.5 mm beam on a 15.4 mm window, '
            f'truncated at 1.7 w), so the containment radius saturates at the '
            f'outermost bin and angle_term reads exactly 1.0; got {angle!r}.')
        assert space > 0.0
        assert k1 > _K1_BAR, (
            f'K1={k1!r} at N={n}: with the angle term saturated the sum '
            f'cannot reach the bar, whatever the leg.')
        assert stage['readout_route'] == 'sziklas'
        assert stage['readout_route_reason'] == 'k1'
        # DERIVED bar, not fitted: the helper forms
        # ``2 dx (|A| r/|B| + theta)/lambda`` with ONE division while the
        # split above forms the two terms separately and adds, so the two
        # differ by a handful of roundings and by nothing else.  Eight ULPs of
        # the reading is the smallest bar two orderings of the same six
        # float64 operations can be held to; the measured difference on this
        # build is under one.
        assert stage['readout_route_k1'] == pytest.approx(
            k1, rel=8.0 * float(np.finfo(float).eps), abs=0.0), (
            f"the hand decomposition {k1!r} and the published "
            f"{stage['readout_route_k1']!r} disagree by more than a "
            f're-association of the same operations -- the split above is no '
            f'longer what the library computes.')

    def test_a_saturated_angle_term_forces_the_route_whatever_the_leg(
            self, monkeypatch):
        """With the angular support clipped the route is decided BEFORE the
        leg is looked at, because ``K1 >= angle_term = 1`` identically.

        That is a stronger statement than "K1 came out above 1 here", and it
        is the one worth gating: it says no final distance and no refinement
        of the SPACE term can reach the bar on such a field.  Asserted by
        sweeping the leg over three decades on the captured envelope.
        """
        seen, _stage = _relay_readout_input(256, monkeypatch)
        env, R, dxe = seen['env'], seen['R'], seen['dx']
        angle, _space, _k1 = _decompose(env, R, 8e-3, dxe)
        assert angle == 1.0
        for z in (1e-3, 8e-3, 50e-3, 0.5, 5.0):
            k1 = C._collins_readout_k1(env, R, z, LAM, dxe, dxe)
            assert k1 > _K1_BAR, (
                f'K1={k1!r} at final_distance={z}: the angle term alone is '
                f'1.0, so the sum cannot be <= 1 at any leg length.')


class TestTheResolvedRouteIsTwoSided:
    """The resolution's two halves, each asserted against the other spelling
    rather than against a recorded number."""

    def test_the_k1_fallback_is_the_sziklas_readout_to_the_bit(self):
        """Where the one-step form is not representable the default must
        return the SAME BYTES as ``transport='sziklas'`` -- that is the whole
        of "flipping the default moves nothing it cannot represent"."""
        cfg = _oversampled_chain()
        fr = dict(dx_out=0.5e-6, N_out=64)
        got = _run(cfg, 8e-3, dict(fr), 'collins')
        ref = _run(cfg, 8e-3, dict(fr), 'sziklas')
        st = got.stages[-1]
        assert st['readout_route'] == 'sziklas'
        assert st['readout_route_reason'] == 'k1'
        assert st['readout_route_k1'] > _K1_BAR
        assert _sha(got.field) == _sha(ref.field), (
            'the k1 fallback did not reproduce the named-sziklas readout '
            'bit for bit')

    def test_the_representable_route_really_takes_the_other_quadrature(self):
        """The complement: where the one-step form IS representable the
        default must NOT be the Sziklas answer, or the resolution is dead and
        every bit-identity above is vacuous."""
        cfg = _oversampled_chain()
        fr = dict(dx_out=0.5e-6, N_out=64)
        got = _run(cfg, 50e-3, dict(fr), 'collins')
        ref = _run(cfg, 50e-3, dict(fr), 'sziklas')
        st = got.stages[-1]
        assert st['readout_route'] == 'collins'
        assert st['readout_route_reason'] == 'representable'
        assert st['readout_route_k1'] <= _K1_BAR
        assert _sha(got.field) != _sha(ref.field), (
            'the readout resolved to the one-step Collins form and still '
            'returned the Sziklas bytes -- the route is published but not '
            'taken')

    def test_the_stop_plane_key_selects_and_k1_is_not_even_computed(self):
        cfg = _oversampled_chain()
        fr = dict(dx_out=0.5e-6, N_out=64)
        for key, val in (('standoff', 2e-3),
                         ('on_focus_containment', 'ignore')):
            got = _run(cfg, 50e-3, dict(fr, **{key: val}), 'collins')
            st = got.stages[-1]
            assert st['readout_route'] == 'sziklas', key
            assert st['readout_route_reason'] == 'sziklas_only_key', key
            assert st['readout_route_k1'] is None, (
                f'{key}: K1 was published for a route the keyword had '
                f'already decided -- the reason string and the reading '
                f'disagree about what took the decision')

    def test_the_three_route_keys_are_absent_on_the_sziklas_spelling(self):
        """The ``'sziklas'`` ``stages`` list is a bit-identity key, so the new
        keys must not appear there -- on EITHER route the condition would have
        chosen."""
        cfg = _oversampled_chain()
        fr = dict(dx_out=0.5e-6, N_out=64)
        for fd in (8e-3, 50e-3):
            st = _run(cfg, fd, dict(fr), 'sziklas').stages[-1]
            for k in ('readout_route', 'readout_route_k1',
                      'readout_route_reason'):
                assert k not in st, (
                    f'{k} leaked onto the sziklas stage at '
                    f'final_distance={fd}')


class TestOpenDefectsFiledByVerifyWpC3:
    """Gaps this verification FILED and did not fix.  Strict xfail is the
    repository's instrument for that (``xfail_strict`` is on), so closing one
    turns its marker red and forces the marker's removal with the fix."""

    # CLOSED by WP-C3 ROUND 2 (2026-09-20): ``not flat`` was dropped from
    # ``tf_available``, so a flat-resolving leg whose chirp-Z is not
    # representable falls back to the Sziklas transport like every other
    # unrepresentable leg.  MEASURED after the fix, same ladder, same window:
    # the default reads 1295.3594 / 1278.4007 / 1274.1529 um at
    # N = 256/512/1024 -- bit-identical to ``transport='sziklas'`` on every
    # rung -- against the oracle's 1413.30 / 1281.02 / 1157.11, i.e. 0.9165x /
    # 0.9980x / 1.1011x, and the Kelly warning this leg used to emit is gone.
    # The strict xfail marker was removed with the fix, which is what the
    # marker exists to force.
    def test_a_flat_reference_leg_agrees_with_the_moment_law_or_refuses(self):
        """The exact free-space second-moment law, which needs no propagator:

            <r^2>(z) = <r^2>(0) + 2 z <r.theta>(0) + z^2 <theta^2>(0)

        Read off the chain's OWN exit field -- which both transports share,
        the exit plane being before the final leg -- so the prediction is
        common to both arms and arbitrates between them.  The bar is 1.5x,
        derived: the oracle's own reading moves 22 % across N = 256..1024
        because a second moment is tail-sensitive on a truncated grid, and
        `'sziklas'` sits inside that spread (0.92 / 1.00 / 1.10), so 1.5x is
        a decade of room above the oracle's own uncertainty and a factor of
        two below the 2.7x-3.5x this leg actually reads.
        """
        from tests.unit.test_audit2609_b4_collins_transport import (
            _CHAIN_TKW, _singlet)
        p = _singlet(120e-3, -120e-3, 6e-3, 'N-BK7', 25.4e-3, 'p')
        groups = [{'prescription': p, 'gap_before': 20e-3}]
        n, w, z = 512, 2e-3, 10e-3
        dx = 10.24e-3 / n
        base = dict(r_in=np.inf, ray_subsample=16, n_workers=1,
                    traced_kwargs=_CHAIN_TKW, final_leg='paraxial')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ex = C.propagate_traced_carrier_chain(
                _gauss(n, dx, w), groups, LAM, dx, final_distance=0.0,
                transport='sziklas', **base)
        dxe = float(ex.dx[0]) if isinstance(ex.dx, tuple) else float(ex.dx)
        r_x, _r_y, _ = C._parse_carrier(ex.R, 'moment')
        ax = (np.arange(n) - n // 2) * dxe
        xx, yy = np.meshgrid(ax, ax, indexing='ij')
        kk = 2.0 * np.pi / LAM
        phys = np.asarray(ex.field) * np.exp(
            1j * kk * (xx ** 2 + yy ** 2) / (2.0 * r_x))
        inten = np.abs(phys) ** 2
        r2 = float(((xx ** 2 + yy ** 2) * inten).sum() / inten.sum())
        spec = np.fft.fft2(phys)
        ps = np.abs(spec) ** 2
        fr = np.fft.fftfreq(n, d=dxe)
        fx, fy = np.meshgrid(fr, fr, indexing='ij')
        th2 = float((((LAM * fx) ** 2 + (LAM * fy) ** 2) * ps).sum()
                    / ps.sum())
        mix = float(np.imag(np.sum(np.conj(phys) * (
            xx * np.fft.ifft2(spec * (2j * np.pi * fx))
            + yy * np.fft.ifft2(spec * (2j * np.pi * fy)))))
            / inten.sum() / kk)
        oracle = float(np.sqrt(r2 + 2.0 * z * mix + z * z * th2))

        def _r2m(field, d):
            a = (np.arange(field.shape[0]) - field.shape[0] // 2) * d
            gx, gy = np.meshgrid(a, a, indexing='ij')
            ii = np.abs(field) ** 2
            return float(np.sqrt(((gx ** 2 + gy ** 2) * ii).sum() / ii.sum()))

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ref = C.propagate_traced_carrier_chain(
                _gauss(n, dx, w), groups, LAM, dx, final_distance=z,
                transport='sziklas', **base)
        d_ref = float(ref.dx[0]) if isinstance(ref.dx, tuple) else float(ref.dx)
        assert 1 / 1.5 < _r2m(np.asarray(ref.field), d_ref) / oracle < 1.5, (
            'the fixture no longer exercises the law: even the co-moving '
            'step disagrees with the oracle, so it cannot arbitrate')
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                got = C.propagate_traced_carrier_chain(
                    _gauss(n, dx, w), groups, LAM, dx, final_distance=z,
                    **base)
        except (ValueError, RuntimeError):
            return          # refusing is an acceptable answer; aliasing is not
        d_got = float(got.dx[0]) if isinstance(got.dx, tuple) else float(got.dx)
        ratio = _r2m(np.asarray(got.field), d_got) / oracle
        assert 1 / 1.5 < ratio < 1.5, (
            f'the DEFAULT lands {ratio:.4f}x of the exact second-moment law '
            f'on a leg that is nowhere near a focus, while the co-moving step '
            f'agrees with it')

    # CLOSED by WP-C3 ROUND 2 (2026-09-20): ``bandlimit`` joined
    # ``_FOCUS_READOUT_SZIKLAS_ONLY_KEYS`` (the renamed
    # ``_FOCUS_READOUT_STOP_PLANE_KEYS``), so naming it on the default
    # SELECTS the Sziklas readout exactly as ``standoff`` and
    # ``on_focus_containment`` do, and the published reason string became
    # ``'sziklas_only_key'`` because ``bandlimit`` is not a stop-plane key.
    def test_bandlimit_is_not_accepted_and_ignored_on_the_collins_route(self):
        cfg = _oversampled_chain()
        fr = dict(dx_out=0.5e-6, N_out=64)
        # it BITES on the transport that owns it ...
        s_on = _sha(_run(cfg, 50e-3, dict(fr), 'sziklas').field)
        s_off = _sha(_run(cfg, 50e-3, dict(fr, bandlimit=False),
                          'sziklas').field)
        assert s_on != s_off, (
            'the fixture no longer exercises bandlimit at all on the Sziklas '
            'readout, so it cannot show the key being dropped elsewhere')
        # ... and on the default it must not be silently dropped.
        got = _run(cfg, 50e-3, dict(fr, bandlimit=False), 'collins')
        assert got.stages[-1]['readout_route'] == 'sziklas', (
            'naming a Sziklas-only key on the default neither selected the '
            'readout that has it nor was refused')

    # CLOSED by WP-C3 ROUND 2 (2026-09-20), and by the SAME edit as D6 above:
    # the raise was never the readout's.  This relay's SECOND gap leg resolved
    # a flat output reference, had no fallback, and ran the chirp-Z at Kelly
    # K1 = K3 = 9.566; the exit envelope came back with an amplitude radius of
    # 3830.25 um against the co-moving step's 1048.97 (3.65x) and 4.899e-04 of
    # envelope power against 8.074e-06 (60.7x).  The Sziklas focus readout's
    # containment guard then refused that lattice -- correctly.  With the leg
    # falling back, the 192-cell ordinary-chain census reads 192 IDENTICAL,
    # 0 MOVED, 0 OK->RAISED and 0 Kelly warnings against 49ddf4bd, where
    # before the fix it read 118 / 52 / 22 / 74.
    def test_no_ordinary_chain_that_returned_on_the_old_default_now_raises(
            self):
        from tests.unit.test_audit2609_b4_collins_transport import (
            _CHAIN_TKW, _singlet)
        # ROUND 2: the sixth argument (the group NAME) was missing when
        # this id was written, so the strict xfail was green on a
        # TypeError and never reached the claim it is about.
        p = _singlet(120e-3, -120e-3, 6e-3, 'N-BK7', 25.4e-3, 'p')
        groups = [{'prescription': p, 'gap_before': 20e-3},
                  {'prescription': p, 'gap_before': 15e-3}]
        n, dx, w = 512, 20e-6, 2.0e-3
        kw = dict(r_in=np.inf, ray_subsample=16, n_workers=1,
                  traced_kwargs=_CHAIN_TKW, final_leg='paraxial',
                  final_distance=8e-3,
                  focus_readout=dict(dx_out=0.5e-6, N_out=64))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            # the pre-flip arithmetic still returns -- so this is a property
            # of the DEFAULT, not of the fixture
            ref = C.propagate_traced_carrier_chain(
                _gauss(n, dx, w), groups, LAM, dx, transport='sziklas', **kw)
            assert np.all(np.isfinite(ref.field))
            got = C.propagate_traced_carrier_chain(
                _gauss(n, dx, w), groups, LAM, dx, **kw)
        assert np.all(np.isfinite(got.field))

    # CLOSED by WP-C3 ROUND 2 (2026-09-20): the bullet now says the keys
    # SELECT the Sziklas readout and publish
    # ``readout_route_reason='sziklas_only_key'``.  The two comments the
    # branch wrote in ``test_niche_d2_chain_multi.py`` were rewritten with
    # it, and ``_SZIKLAS_STANDOFF``'s docstring now describes all nine of
    # its uses rather than the six that pass a ``standoff``.
    def test_the_chain_docstring_does_not_still_say_the_keys_are_refused(self):
        doc = C.propagate_traced_carrier_chain.__doc__ or ''
        assert 'refused, not ignored' not in doc, (
            'the public docstring of the function whose contract changed '
            'still describes the pre-change contract')

    # CLOSED by WP-C3 ROUND 2 (2026-09-20).  Re-measured on the array the
    # chain itself hands the condition: exit pitch 76.5444 um, exit support
    # radius 6.7359 mm, K1 = 82.36047 / 41.44910 / 21.42248 / 10.94410 at
    # N = 256/512/1024/2048.  The docstring carries those numbers now, plus
    # the staircase and the slope, and the '5.76 mm / K1 = 56.0' clause and
    # the unreproducible free-leg 1.2359 are gone.
    def test_the_readout_k1_docstring_quotes_the_measured_reading(
            self, monkeypatch):
        doc = C._collins_readout_k1.__doc__ or ''
        _seen, stage = _relay_readout_input(256, monkeypatch)
        assert f"{stage['readout_route_k1']:.2f}".startswith('82.3'), stage
        assert 'K1 = 56.0' not in doc, (
            'the docstring states a K1 for this fixture that the fixture '
            'does not produce')
