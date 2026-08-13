"""Niche C15 (2026-08-11) -- the INVERSE CHARACTERISTIC per-pixel evaluator.

WHAT THIS FILE IS FOR, in three sentences.  The feature replaces the traced
element's coarse-Newton-lattice-plus-``map_coordinates``-upsample chain with one
exact polynomial evaluation per exit pixel.  It ships ON and SCOPED (see that
flag's own note: the exact-trace oracle decided the accuracy case in its
favour, and the one guard that blocked the default -- G8, whose probe sat on
the launch lattice -- was re-architected onto off-lattice points rather than
weakened), so the tests that matter are the ones that pin the FAIL-BEFORE
(``TRACED_INVERSE_MAP = False``
must leave the call indistinguishable from the pre-feature library, and must
not even attempt a build) and the REFUSAL (a guard that fires must keep the
shipped path, not a degraded one).  The accuracy tests are comparative
by construction -- the bar is the incumbent Newton path at the same
OFF-LATTICE probe points against the same exact ray truth, never an absolute
wave tolerance,
because the incumbent on design 121's last group is 90x tighter than
``lambda/100`` and an absolute bar at that tolerance would have admitted a 24x
regression.  The probe points are off the lattice because at the lattice an
``s = 0`` spline incumbent is exact by construction and the bar is decided by
which interpolant the caller asked for.  The rest pin the two optimisations
the path needed to be worth
having: the radially-screened hull tests, which must be BIT-identical to the
dense ones they replace, and the numba/NumPy evaluator pair.

Sources: ``docs/audits/BUILD_INVERSE_MAP_2026_08_11.md``,
         ``docs/audits/FIX_G8_PROBE_2026_08_12.md`` (the off-lattice probe,
         the default flip, and the fail-befores that keep the guard a guard),
         ``docs/audits/PROTO_INVERSE_MAP_2026_08_11.md`` (S3.1 the exit-degree
         ladder, S3.4 the parity framing, S4.3 the upsample census, S5.1 the
         1.1e+04-wave extrapolation outside the landing hull),
         ``docs/audits/PROTO_HAMILTON_MAP_2026_08_11.md`` (S5 guards G1-G5).
"""
from __future__ import annotations

import threading
import warnings

import numpy as np
import pytest

import lumenairy as la
from lumenairy.elements import _lens_imap as IM
from lumenairy.elements import _lens_traced as LT

_WL = 1.31e-6
_N, _DX, _W = 384, 16e-6, 1.4e-3
_SUB = 4
_RC = -0.06                      # carrier conjugate (converging), metres


def _surf(radius, gb, ga):
    return {'radius': radius, 'glass_before': gb, 'glass_after': ga,
            'conic': 0.0, 'radius_y': None, 'conic_y': None,
            'aspheric_coeffs': None, 'aspheric_coeffs_y': None}


def _presc():
    return {'name': 'c15_singlet', 'aperture_diameter': 9e-3,
            'surfaces': [_surf(0.030, 'air', 'N-BK7'),
                         _surf(-0.030, 'N-BK7', 'air')],
            'thicknesses': [3e-3]}


def _field(n=_N, dx=_DX, w=_W):
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    return np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)


_RD_KW = dict(amplitude_model='ray_density', carrier=_RC,
              preserve_input_phase='remap', remap_sampling='full')


def _call(flag=None, rd=True, **over):
    """One element call at a stated flag value, with the module state restored
    afterwards.  Returns ``(field, guard_record, warnings)``."""
    kw = dict(prescription=_presc(), wavelength=_WL, dx=_DX,
              ray_subsample=_SUB, n_workers=1, on_undersample='silent')
    if rd:
        kw.update(_RD_KW)
    kw.update(over)
    rec = {}
    old = IM.TRACED_INVERSE_MAP
    IM.inverse_map_cache_clear()
    try:
        if flag is not None:
            IM.TRACED_INVERSE_MAP = flag
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            out = la.apply_real_lens_traced(_field(), _imap_out=rec, **kw)
    finally:
        IM.TRACED_INVERSE_MAP = old
        IM.inverse_map_cache_clear()
    return out, rec, [str(w.message) for w in caught]


@pytest.fixture(scope='module')
def _on():
    return _call(flag=True)


@pytest.fixture(scope='module')
def _off():
    return _call(flag=False)


# ===========================================================================
# 1.  THE FAIL-BEFORE
# ===========================================================================
def test_the_fail_before_switch_restores_pre_c15_bits(_off):
    """``TRACED_INVERSE_MAP = False`` must not merely disable the map, it must
    leave the call indistinguishable from the library before the feature: no
    build attempted, nothing consulted, and the same bits twice."""
    E0, rec, _w = _off
    assert rec['engaged'] is False
    assert rec.get('refused') is None, (
        'the fail-before must not even ATTEMPT a build, so it can have no '
        'refusal to report')
    E1, _r1, _w1 = _call(flag=False)
    assert np.array_equal(E0, E1), 'the fail-before path is not deterministic'


def test_the_fail_before_never_reaches_the_module(monkeypatch):
    """The switch is not a filter on the map's OUTPUT, it is a gate before its
    INPUT.  Make the builder fatal and prove the flag-off call never calls it
    -- the difference between "we did the work and discarded it" and "we did
    not do the work", which is what the timing claim rests on."""
    def _boom(*_a, **_k):                          # pragma: no cover - fatal
        raise AssertionError('build_inverse_map called with the flag off')
    monkeypatch.setattr(IM, 'build_inverse_map', _boom)
    E, rec, _w = _call(flag=False)
    assert np.isfinite(np.abs(E)).all()
    assert rec['engaged'] is False


def test_the_per_call_kwarg_overrides_the_module_flag():
    """``inverse_map=`` is the per-call override; ``None`` follows the flag."""
    on_by_kw, rec_on, _ = _call(flag=False, inverse_map=True)
    off_by_kw, rec_off, _ = _call(flag=True, inverse_map=False)
    assert rec_on['engaged'] is True
    assert rec_off['engaged'] is False
    assert not np.array_equal(on_by_kw, off_by_kw)


@pytest.mark.parametrize('bad', ['yes', 1.5, 'auto', []])
def test_a_junk_inverse_map_value_is_refused_at_entry(bad):
    """The house rule the D5 gate test enforces: validate at the top of the
    call, so a junk value cannot behave as the default on every call where the
    map never engages and only surface on the one where it does."""
    with pytest.raises(ValueError, match='inverse_map='):
        la.apply_real_lens_traced(
            _field(), prescription=_presc(), wavelength=_WL, dx=_DX,
            ray_subsample=_SUB, n_workers=1, inverse_map=bad)


def test_the_switch_is_live(_on, _off):
    """A fail-before that changes nothing is not a fail-before -- it is a
    feature that never engaged."""
    assert _on[1]['engaged'] is True
    assert not np.array_equal(_on[0], _off[0])


# ===========================================================================
# 2.  G8 -- THE COMPARATIVE BAR, WHICH IS THE ONE THAT MATTERS
# ===========================================================================
def test_the_map_beats_the_incumbent_at_off_lattice_probe_points(_on):
    """The bar is PARITY with the Newton path the map replaces, measured where
    the evaluation actually happens.

    ``lambda/100`` is not the bar: the incumbent delivers lambda/9 000 on
    design 121's last group, so an absolute bar at lambda/100 would have
    admitted a 24x regression (proto S3.4).

    AND THE PROBE POINTS ARE NOT LAUNCH NODES, which is the correction
    ``FIX_G8_PROBE_2026_08_12`` made.  Scored at nodes, an ``s = 0``
    ``RectBivariateSpline`` incumbent is exact by construction -- it
    reproduces every node it was built on, including the ones held out of the
    MODEL -- so its measured "error" there was the Newton loop's leftover
    residual and the guard read it as 34.5x more accurate than it is anywhere
    an exit pixel can land.  Truth at the probe points comes from the element's
    own trace THERE, which neither arm has seen.
    """
    g = _on[1]
    assert g['n_parity'] >= IM._IMAP_PROBE_MIN
    assert g['n_probe_traced'] >= IM._IMAP_PROBE_MIN
    assert g['parity_map_opl_waves'] <= g['parity_incumbent_opl_waves']
    assert g['parity_map_opl_rms_waves'] <= g['parity_incumbent_opl_rms_waves']
    assert g['parity_map_pos_m'] <= g['parity_incumbent_pos_m']


def test_a_parity_failure_refuses_and_keeps_the_shipped_bits(_off):
    """Refuse, never degrade.  Drive G8's bar to something no model can meet
    and the call must return the FAIL-BEFORE field exactly, with a report."""
    old = IM._IMAP_PARITY_FACTOR
    try:
        IM._IMAP_PARITY_FACTOR = 0.0
        E, rec, msgs = _call(flag=True)
    finally:
        IM._IMAP_PARITY_FACTOR = old
    assert rec['refused'] == 'G8'
    assert rec['engaged'] is False
    assert np.array_equal(E, _off[0]), (
        'a refused build must keep the shipped path bit for bit')
    assert any('inverse-characteristic' in m and 'G8' in m for m in msgs)


@pytest.mark.parametrize('degree', [4, 6])
def test_a_degree_too_low_to_reach_parity_refuses(_off, degree):
    """G7 -- exit-degree adequacy, read straight off the least-squares
    residual.  Degree 4 is 2.15 waves on the proto's ladder; it must not be
    allowed to ship a plausible-looking answer.

    DEGREE 6 IS THE ONE THAT MATTERS, and it is here because the re-architected
    probe has to be shown firing at the KNEE and not only on absurdities.  On
    this fixture the ladder reads (off-lattice probe, 910 points)::

        degree   map OPL      map position   verdict
         4       1.14e-03 w   3.82e-08 m     REFUSE  (57x / 118x)
         6       1.20e-05 w   4.80e-10 m     REFUSE  (0.60x / 1.49x)
         8       1.48e-07 w   6.51e-12 m     ENGAGE
        14       6.08e-12 w   3.39e-17 m     ENGAGE  (shipped)

    Degree 6 already WINS the OPL channel at 0.60x and is still refused,
    because it loses the ENTRANCE POSITION channel at 1.49x -- which is the
    envelope rule doing its job: the ray-density amplitude and the C8 taper
    both read the entrance coordinates, so a model may not buy OPL with them.
    (The knee sits lower here than on design 121 because this fixture is a
    short singlet whose exit map a degree-8 polynomial spans easily; on design
    121's decentred order (-4,-2) degree 8 is refused at 2.17x.)
    """
    old = IM._IMAP_EXIT_DEGREE
    try:
        IM._IMAP_EXIT_DEGREE = degree
        E, rec, _m = _call(flag=True)
    finally:
        IM._IMAP_EXIT_DEGREE = old
    assert rec['refused'] in ('G7', 'G8')
    assert np.array_equal(E, _off[0])


# ===========================================================================
# 2A. THE PROBE -- WHERE G8 ASKS ITS QUESTION (FIX_G8_PROBE_2026_08_12)
#
# The guard's QUESTION was always right ("is this model at least as good as
# the path it replaces?") and its PROBE POINTS were wrong: launch nodes, where
# an interpolating incumbent is exact by construction.  These pin the three
# properties the replacement rests on -- the points are off-lattice, the truth
# is the element's own congruence, and without that truth the guard refuses
# rather than falling back to the degenerate probe.
# ===========================================================================
def _capture_build(**over):
    """One flag-on element call, with everything ``build_inverse_map`` was
    handed captured verbatim -- including the ``probe_trace`` closure, which is
    the object under test in two of the cases below."""
    cap = {}
    orig = IM.build_inverse_map

    def wrapper(xs_in, XO, YO, OP, *a, **kw):
        cap['args'] = (xs_in, XO, YO, OP) + a
        cap['kw'] = dict(kw)
        cap['xs'] = np.asarray(xs_in).copy()
        cap['XO'] = np.asarray(XO).copy()
        cap['YO'] = np.asarray(YO).copy()
        cap['OP'] = np.asarray(OP).copy()
        cap['imap'] = orig(xs_in, XO, YO, OP, *a, **kw)
        return cap['imap']

    IM.build_inverse_map = wrapper
    try:
        cap['out'], cap['rec'], cap['msgs'] = _call(flag=True, **over)
    finally:
        IM.build_inverse_map = orig
    return cap


@pytest.fixture(scope='module')
def _cap():
    return _capture_build()


def test_the_probe_points_are_off_the_launch_lattice():
    """The whole correction, as a property of the point chooser.

    Every probe sits strictly INSIDE a launch cell -- never on a node, never
    on a cell edge -- because a node is exactly where an ``s = 0`` spline
    incumbent has no error to measure.  The cell must also be wholly inside
    the census region, so no probe is scored where one of the arms is
    extrapolating.
    """
    n = 41
    xs = np.linspace(-1.0, 1.0, n)
    census = (xs[:, None] ** 2 + xs[None, :] ** 2) <= 0.55 ** 2
    px, py = IM._probe_entrance_points(xs, census)
    assert px.size >= IM._IMAP_PROBE_MIN
    assert px.size == py.size
    h = float(xs[1] - xs[0])
    for a in (px, py):
        # distance to the nearest lattice node, in cells
        frac = np.abs((a - xs[0]) / h - np.rint((a - xs[0]) / h))
        assert frac.min() > 0.0, (
            'a probe landed ON a launch node, which is the degenerate point '
            'the off-lattice probe exists to avoid')
    # ...and inside the census, by construction: the enclosing cell's four
    # corners must all be census nodes.
    ci = np.floor((px - xs[0]) / h).astype(int)
    cj = np.floor((py - xs[0]) / h).astype(int)
    assert census[ci, cj].all() and census[ci + 1, cj + 1].all()
    assert census[ci + 1, cj].all() and census[ci, cj + 1].all()
    # deterministic: no seed, no randomness, same answer twice
    qx, qy = IM._probe_entrance_points(xs, census)
    assert np.array_equal(px, qx) and np.array_equal(py, qy)


def test_the_probe_budget_is_a_stride_not_a_truncation():
    """A budget that took the FIRST N cells would probe one corner of the
    beam.  The stride keeps the probe set spread over the whole census region
    while bounding the extra rays -- so the count lands under the budget and
    the extent does not shrink."""
    n = 301
    xs = np.linspace(-1.0, 1.0, n)
    census = np.ones((n, n), dtype=bool)
    px, py = IM._probe_entrance_points(xs, census)
    assert (n - 1) ** 2 > IM._IMAP_PROBE_MAX, 'fixture no longer over budget'
    assert IM._IMAP_PROBE_MIN <= px.size <= IM._IMAP_PROBE_MAX
    span = float(xs[-1] - xs[0])
    for a in (px, py):
        assert float(a.max() - a.min()) > 0.90 * span, (
            'the probe budget truncated the probe set instead of striding it')
        # ...and it is CENTRED: the leftover cells are split between the two
        # edges, not all left at the far one.
        assert abs(float(a.min() - xs[0]) - float(xs[-1] - a.max())) < 4.0 * (
            span / (n - 1))


def test_the_probe_trace_is_the_element_s_own_lattice_trace(_cap):
    """G8's ground truth must be the SAME congruence that is on trial.

    Hand the probe trace the launch lattice itself and it has to reproduce the
    element's own ``x_out_grid`` / ``y_out_grid`` / ``opl_grid`` -- the launch
    directions (``grad(W + a_fit)`` where niche C6 is engaged), the
    exit-vertex correction, the H6 carrier eikonal and the on-axis OPL
    reference all included.  Measured BIT-IDENTICAL, which is the assertion
    here: a probe trace that merely agrees closely would be a second model of
    the element rather than the element.
    """
    probe = _cap['kw'].get('probe_trace')
    assert callable(probe), 'the element stopped supplying G8 its ground truth'
    xs = _cap['xs']
    X, Y = np.meshgrid(xs, xs, indexing='ij')
    qx, qy, qo = probe(X.ravel(), Y.ravel())
    m = np.isfinite(_cap['XO'].ravel())
    assert m.sum() > 100
    for got, want in ((qx, _cap['XO']), (qy, _cap['YO']), (qo, _cap['OP'])):
        assert np.array_equal(got[m], want.ravel()[m]), (
            'the probe trace no longer reproduces the element\'s own lattice '
            'landings, so G8 would be scoring both arms against a DIFFERENT '
            'congruence from the one they model')


def test_the_guard_refuses_without_off_lattice_ground_truth(_cap):
    """A comparative bar with no honest truth to compare against is not a bar.

    The old probe's fallback was the launch lattice, and that is exactly what
    must NOT happen: on an interpolating incumbent it decides the bar in the
    incumbent's favour by construction.  So a build with no ``probe_trace``
    REFUSES -- the same posture as a build with no ``parity_invert``.
    """
    for drop in ('probe_trace', 'parity_invert'):
        rec = {}
        kw = dict(_cap['kw'])
        kw[drop] = None
        kw['cache'] = False
        kw['guard_record'] = rec
        assert IM.build_inverse_map(*_cap['args'], **kw) is None
        assert rec['refused'] == 'G8', (drop, rec)


def test_the_guard_scores_the_coefficients_that_ship(_cap):
    """Arm A is the OBJECT THE CALLER GETS, re-derived from outside the guard.

    The old probe could not say this: it had to refit the model without the
    nodes it then scored it at, so the arm on trial was a different polynomial
    from the one that shipped.  An off-lattice probe withholds nothing, so the
    number in the record must be reproducible by evaluating the RETURNED model
    at the probe points -- which is what this recomputes, independently, from
    the point chooser and the element's own trace.
    """
    imap = _cap['imap']
    assert imap is not None and _cap['rec']['engaged'] is True
    # the census the guard scores on, rebuilt from what the builder was handed
    xs, XO, YO, OP = _cap['args'][:4]
    good = np.isfinite(XO) & np.isfinite(YO) & np.isfinite(OP)
    good &= np.isfinite(IM._detj_landing_stencil(xs, XO, YO))
    W = _cap['kw'].get('weights')
    if W is not None:
        sub = good & (W >= 0.5 * float(W[good].max()))
        if int(sub.sum()) >= 16:
            good = sub
    px, py = IM._probe_entrance_points(xs, good)
    qx, qy, qo = _cap['kw']['probe_trace'](px, py)
    ch = [np.empty(qx.shape) for _ in range(3)]
    imap.eval_into(qx, qy, ch,
                   channels=[IM.InverseCharacteristic.CH_X_IN,
                             IM.InverseCharacteristic.CH_Y_IN,
                             IM.InverseCharacteristic.CH_OPL])
    e_pos = float(np.nanmax(np.maximum(np.abs(ch[0] - px),
                                       np.abs(ch[1] - py))))
    e_opl = float(np.nanmax(np.abs(ch[2] - qo)) / _WL)
    assert e_pos == pytest.approx(_cap['rec']['parity_map_pos_m'], rel=1e-9)
    assert e_opl == pytest.approx(_cap['rec']['parity_map_opl_waves'],
                                  rel=1e-9)


def test_a_folded_jacobian_refuses(_off):
    """G2.  A ``det J`` that changes sign has no single-valued inverse, so
    there is nothing for an inverse characteristic to model."""
    old = IM._IMAP_DETJ_MAXMIN
    try:
        IM._IMAP_DETJ_MAXMIN = 1.0 + 1e-12
        E, rec, _m = _call(flag=True)
    finally:
        IM._IMAP_DETJ_MAXMIN = old
    assert rec['refused'] == 'G2'
    assert np.array_equal(E, _off[0])


def test_the_guard_action_is_validated_when_a_refusal_happens():
    """Validated at the reporting site rather than at import, because that is
    where a junk value could otherwise behave as 'warn' forever."""
    old_a, old_p = IM.INVERSE_MAP_GUARD, IM._IMAP_PARITY_FACTOR
    try:
        IM.INVERSE_MAP_GUARD = 'shout'
        IM._IMAP_PARITY_FACTOR = 0.0
        with pytest.raises(ValueError, match='INVERSE_MAP_GUARD'):
            _call(flag=True)
    finally:
        IM.INVERSE_MAP_GUARD, IM._IMAP_PARITY_FACTOR = old_a, old_p


def test_the_guard_report_is_reporting_only(_off):
    """'silent' must change what is SAID, never what is returned."""
    old_a, old_p = IM.INVERSE_MAP_GUARD, IM._IMAP_PARITY_FACTOR
    try:
        IM._IMAP_PARITY_FACTOR = 0.0
        IM.INVERSE_MAP_GUARD = 'warn'
        E_w, _r, msgs_w = _call(flag=True)
        IM.INVERSE_MAP_GUARD = 'silent'
        E_s, _r2, msgs_s = _call(flag=True)
    finally:
        IM.INVERSE_MAP_GUARD, IM._IMAP_PARITY_FACTOR = old_a, old_p
    assert np.array_equal(E_w, E_s)
    assert np.array_equal(E_w, _off[0])
    assert any('G8' in m for m in msgs_w)
    assert not any('inverse-characteristic' in m for m in msgs_s)


# ===========================================================================
# 3.  THE DOMAIN -- G6, and why it is not optional
# ===========================================================================
def test_outside_the_landing_hull_is_refused_not_extrapolated(_on):
    """A degree-14 model one plateau outside the hull is 1.1e+04 waves wrong
    (proto S5.1).  Every exit pixel the map answers for must be inside the hull
    of the rays that were actually traced."""
    E, rec, _w = _on
    assert rec['n_out_of_domain'] >= 0
    assert np.isfinite(np.abs(E)).all(), 'the domain mask leaked a non-finite'


def test_the_screened_hull_mask_is_bit_identical_to_the_dense_test():
    """The radial screens are what make a hull test affordable on the wave
    grid (6.7e+07 pixels x ~150 facets is a 10^10-MAC pass).  They are strict
    bounds, so the answer must be IDENTICAL, not merely close."""
    rng = np.random.default_rng(11)
    p = rng.standard_normal((400, 2)) * np.array([2e-3, 1.5e-3])
    hull = LT._TracedExitSupport.half_planes(p[:, 0], p[:, 1], strict=True)
    imap = IM.InverseCharacteristic(
        coef=np.zeros((3, 4)), terms=IM._td_terms(1), degree=1,
        exit_c=(0.0, 0.0), exit_h=(1.0, 1.0), hull=hull,
        hull_c=(float(p[:, 0].mean()), float(p[:, 1].mean())),
        hull_rmax=float(np.hypot(p[:, 0] - p[:, 0].mean(),
                                 p[:, 1] - p[:, 1].mean()).max()),
        launch_radius=1.0, wavelength=_WL, n_samples=400,
        residual=np.zeros(4), det_j_range=1.0, det_j_sign=1.0, guards={},
        key=None, build_seconds=0.0)
    xg = np.linspace(-4e-3, 4e-3, 231)
    yg = np.linspace(-3e-3, 3e-3, 197)
    fast = imap.hull_mask_grid(xg, yg)
    Xg, Yg = np.meshgrid(xg, yg)
    dense = LT._TracedExitSupport.signed_distance(
        hull[0], hull[1], Xg, Yg) <= 1e-12
    assert np.array_equal(fast, dense)


def test_the_screened_taper_is_bit_identical_to_the_dense_taper():
    """Same argument for niche C8's exit-support taper, which the inverse path
    asks for on the WAVE grid rather than on the coarse lattice."""
    rng = np.random.default_rng(3)
    p = rng.standard_normal((500, 2)) * np.array([2e-3, 1.6e-3])
    amp = np.ones(64)
    sup = LT._TracedExitSupport.from_landings(
        p[:64, 0].reshape(8, 8), p[:64, 1].reshape(8, 8), amp.reshape(8, 8),
        np.linspace(-1e-3, 1e-3, 8), 4e-3, 1e-5, 4,
        want_halo=False, want_bound=True)
    if sup.bound is None:                              # pragma: no cover
        pytest.skip('the fixture produced no bound to compare')
    xg = np.linspace(-4e-3, 4e-3, 151)
    yg = np.linspace(-3e-3, 3e-3, 131)
    Xg, Yg = np.meshgrid(xg, yg)
    plateau = 5e-5
    assert np.array_equal(sup.taper_grid(xg, yg, plateau),
                          sup.taper(Xg, Yg, plateau))


# ===========================================================================
# 4.  THE EVALUATOR
# ===========================================================================
def test_the_numba_kernel_and_the_numpy_fallback_agree():
    """Two implementations of one formula, differing only in the summation
    order over the basis -- the same contract ``_Cheb2DEvaluator`` states for
    its own pair, and the same reason: whichever branch a process resolves is
    the branch its answer comes from."""
    kern = IM._get_imap_eval_numba()
    if kern is None:                                   # pragma: no cover
        pytest.skip('numba unavailable; there is only one branch to compare')
    rng = np.random.default_rng(5)
    terms = IM._td_terms(9)
    imap = IM.InverseCharacteristic(
        coef=rng.standard_normal((terms.shape[0], 4)), terms=terms, degree=9,
        exit_c=(1e-4, -2e-4), exit_h=(3e-3, 2e-3), hull=None, hull_c=None,
        hull_rmax=None, launch_radius=1.0, wavelength=_WL, n_samples=1,
        residual=np.zeros(4), det_j_range=1.0, det_j_sign=1.0, guards={},
        key=None, build_seconds=0.0)
    X = rng.uniform(-3e-3, 3e-3, (37, 41))
    Y = rng.uniform(-2e-3, 2e-3, (37, 41))
    got = imap.eval(X, Y)
    IM._NUMBA_KERNELS['imap_eval'] = None
    try:
        ref = imap.eval(X, Y)
    finally:
        del IM._NUMBA_KERNELS['imap_eval']
    for a, b in zip(got, ref):
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())


def test_a_channel_subset_matches_the_full_evaluation():
    """The screen path does not want ``det J``, and on a 6.7e+07-pixel grid a
    channel it does not want is a 537 MB array it does not allocate."""
    rng = np.random.default_rng(7)
    terms = IM._td_terms(6)
    imap = IM.InverseCharacteristic(
        coef=rng.standard_normal((terms.shape[0], 4)), terms=terms, degree=6,
        exit_c=(0.0, 0.0), exit_h=(1e-3, 1e-3), hull=None, hull_c=None,
        hull_rmax=None, launch_radius=1.0, wavelength=_WL, n_samples=1,
        residual=np.zeros(4), det_j_range=1.0, det_j_sign=1.0, guards={},
        key=None, build_seconds=0.0)
    X = rng.uniform(-1e-3, 1e-3, (23, 19))
    Y = rng.uniform(-1e-3, 1e-3, (23, 19))
    full = imap.eval(X, Y)
    sub = imap.eval(X, Y, channels=(2, 0))
    assert np.array_equal(sub[0], full[2])
    assert np.array_equal(sub[1], full[0])


def test_the_evaluation_is_chunk_invariant():
    """Chunking bounds the scratch; it must not be visible in the answer."""
    rng = np.random.default_rng(13)
    terms = IM._td_terms(5)
    imap = IM.InverseCharacteristic(
        coef=rng.standard_normal((terms.shape[0], 4)), terms=terms, degree=5,
        exit_c=(0.0, 0.0), exit_h=(1.0, 1.0), hull=None, hull_c=None,
        hull_rmax=None, launch_radius=1.0, wavelength=_WL, n_samples=1,
        residual=np.zeros(4), det_j_range=1.0, det_j_sign=1.0, guards={},
        key=None, build_seconds=0.0)
    X = rng.uniform(-1, 1, (40, 40))
    Y = rng.uniform(-1, 1, (40, 40))
    for a, b in zip(imap.eval(X, Y, chunk=7), imap.eval(X, Y, chunk=10 ** 6)):
        assert np.array_equal(a, b)


# ===========================================================================
# 5.  THE CACHE -- chain-A key discipline
# ===========================================================================
def test_the_cache_key_moves_when_any_input_moves():
    """The chain-A lesson (FIX_D4_D6_D7 D6) applied to an in-process cache: a
    key that names the CONFIGURATION and not the CONTENT is how a cache
    silently becomes a cache of something else.  Every array that enters the
    fit is hashed, so a single moved ray moves the key."""
    rng = np.random.default_rng(2)
    xs = np.linspace(-1e-3, 1e-3, 9)
    a = [xs] + [rng.standard_normal((9, 9)) for _ in range(4)]
    k0 = IM._imap_key(*a, None, 14, 1e-3, _WL)
    assert k0 == IM._imap_key(*a, None, 14, 1e-3, _WL)
    a2 = [x.copy() for x in a]
    # ONE ULP -- the smallest change a traced landing can actually carry.  A
    # smaller perturbation than this is not a different ray, it is the same
    # float64, and a key that moved on it would be testing the test.
    a2[1][4, 4] = np.nextafter(a2[1][4, 4], np.inf)
    assert IM._imap_key(*a2, None, 14, 1e-3, _WL) != k0
    assert IM._imap_key(*a, None, 12, 1e-3, _WL) != k0
    assert IM._imap_key(*a, np.ones((9, 9)), 14, 1e-3, _WL) != k0
    old = IM.TRACED_INVERSE_MAP
    try:
        IM.TRACED_INVERSE_MAP = not old
        assert IM._imap_key(*a, None, 14, 1e-3, _WL) != k0
    finally:
        IM.TRACED_INVERSE_MAP = old


def test_a_repeated_identical_call_hits_the_cache():
    IM.inverse_map_cache_clear()
    assert IM.inverse_map_cache_info()['size'] == 0
    kw = dict(prescription=_presc(), wavelength=_WL, dx=_DX,
              ray_subsample=_SUB, n_workers=1, on_undersample='silent',
              **_RD_KW)
    old = IM.TRACED_INVERSE_MAP
    try:
        IM.TRACED_INVERSE_MAP = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            e0 = la.apply_real_lens_traced(_field(), **kw)
            info0 = IM.inverse_map_cache_info()
            e1 = la.apply_real_lens_traced(_field(), **kw)
            info1 = IM.inverse_map_cache_info()
    finally:
        IM.TRACED_INVERSE_MAP = old
        IM.inverse_map_cache_clear()
    assert info0['size'] == 1 and info0['hits'] == 0
    assert info1['hits'] == 1
    assert np.array_equal(e0, e1), 'a cache hit changed the answer'


def test_the_cache_containers_stay_consistent_under_threads():
    """The defect the companion lock closes, exercised rather than asserted.

    ``_cache_get`` and ``_cache_put`` do read-modify-write sequences ACROSS
    three containers -- the map dict, the LRU order list, the counters -- so
    an unlocked interleave can leave a key in one and not the other: past the
    capacity bound one way, a dropped live entry the other.  Hammer both from
    several threads and demand the invariant that the lock exists to hold."""
    IM.inverse_map_cache_clear()
    n_thread, n_op = 8, 400
    keys = ['k%02d' % i for i in range(3 * max(1, IM._IMAP_CACHE_SIZE))]

    def worker(seed):
        rng = np.random.default_rng(seed)
        for _ in range(n_op):
            k = keys[int(rng.integers(len(keys)))]
            if IM._cache_get(k) is None:
                IM._cache_put(k, k)

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(n_thread)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    with IM._IMAP_LOCK:
        assert set(IM._IMAP_CACHE) == set(IM._IMAP_CACHE_ORDER), (
            'the dict and the LRU order list disagree -- the exact torn '
            'read-modify-write the lock exists to prevent')
        assert len(IM._IMAP_CACHE_ORDER) == len(set(IM._IMAP_CACHE_ORDER))
        assert len(IM._IMAP_CACHE) <= IM._IMAP_CACHE_SIZE, (
            'the capacity bound leaked')
    info = IM.inverse_map_cache_info()
    assert info['hits'] + info['misses'] == n_thread * n_op, (
        'the counters are pinned EXACT under contention -- they live inside '
        'the same critical section the container invariant already needs')
    IM.inverse_map_cache_clear()


def test_the_cache_is_enrolled_with_the_central_registry():
    """v4.16.0's contract: a module owning a cache hands its clearer to the
    registry, so the library-wide drain reaches it.  Before this fix
    ``inverse_map_cache_clear`` existed and was never registered -- the exact
    sibling gap the registry was written to retire."""
    from lumenairy import _cache_registry as REG
    assert 'inverse_map' in REG.list_registered_cache_clearers()
    IM.inverse_map_cache_clear()
    with IM._IMAP_LOCK:
        IM._IMAP_CACHE['sentinel'] = 'sentinel'
        IM._IMAP_CACHE_ORDER.append('sentinel')
        IM._IMAP_CACHE_STATS['hits'] = 11
    REG.clear_all_registered_caches()
    assert IM.inverse_map_cache_info() == {
        'size': 0, 'capacity': IM._IMAP_CACHE_SIZE, 'hits': 0, 'misses': 0}


# ===========================================================================
# 6.  THE GATE
# ===========================================================================
@pytest.mark.parametrize('over, why', [
    ({'ray_subsample': 1}, 'no coarse lattice exists at sub == 1'),
    ({'inversion_method': 'fit'}, 'the fit path is already per-pixel'),
])
def test_the_gate_closes_where_there_is_nothing_to_replace(over, why):
    _E, rec, _w = _call(flag=True, rd=False, **over)
    assert rec['gate_open'] is False, why
    assert rec['engaged'] is False


def test_an_ordinary_chain_leg_is_scoped_out_even_on_the_shipped_default():
    """S5 SCOPING, and it is what makes the default flip a ONE-LEG change.

    ``carrier.py``'s ordinary chain-leg call site passes ``inverse_map=False``
    through ``setdefault``, so every INTERMEDIATE leg keeps the shipped path
    whatever ``TRACED_INVERSE_MAP`` holds.  That is structural, not numeric:
    an intermediate's output is re-fitted downstream (the C6 residual eikonal,
    the C11 arbiter, the beam radius that sizes the ray-fit disc, the carrier
    reference), and the evaluator's evidence is about one element's returned
    map, not about what a skirt-level change to an intermediate does to six
    downstream fits.

    With the flag defaulting to ``True`` this stops being a note and becomes
    the difference between changing one leg and changing the whole chain, so
    it is asserted at the call site: every element call the chain makes must
    carry ``inverse_map=False``, explicitly.
    """
    from lumenairy import elements as EL
    seen = []
    real = EL.apply_real_lens_traced

    def spy(*a, **kw):
        seen.append(kw.get('inverse_map', '<absent>'))
        return real(*a, **kw)

    presc = {'name': 'c15_relay', 'aperture_diameter': 9e-3,
             'surfaces': [_surf(0.060, 'air', 'N-BK7'),
                          _surf(-0.060, 'N-BK7', 'air')],
             'thicknesses': [3e-3]}
    groups = [{'prescription': presc, 'gap_before': 20e-3},
              {'prescription': presc, 'gap_before': 10e-3}]
    n, dx, w = 256, 30e-6, 2.2e-3
    ax = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(ax, ax)
    env = np.exp(-(X ** 2 + Y ** 2) / w ** 2).astype(np.complex128)
    old = IM.TRACED_INVERSE_MAP
    EL.apply_real_lens_traced = spy
    try:
        IM.TRACED_INVERSE_MAP = True
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            la.propagate_traced_carrier_chain(
                env, groups, _WL, dx, r_in=60e-3, ray_subsample=8,
                n_workers=1, final_distance=8e-3, final_leg='paraxial',
                traced_kwargs=dict(on_undersample='silent',
                                   on_noncollimated='silent'))
    finally:
        EL.apply_real_lens_traced = real
        IM.TRACED_INVERSE_MAP = old
    assert len(seen) >= 2, seen
    assert all(v is False for v in seen), (
        'an ordinary chain leg no longer scopes the inverse-characteristic '
        f'evaluator out: {seen}')


def test_the_map_is_registered_as_a_traced_layer_flag():
    """A switch that changes a returned bit and is not in the registry is a
    switch with no discoverable fail-before (niche C14)."""
    from lumenairy.elements import _traced_flags as TF
    names = {n for (_m, n) in TF.FLAGS}
    assert {'TRACED_INVERSE_MAP', 'INVERSE_MAP_GUARD'} <= names
    with TF.traced_era('v5.31'):
        assert IM.TRACED_INVERSE_MAP is False
<<<<<<< HEAD
    assert IM.TRACED_INVERSE_MAP is True, (
        'the evaluator SHIPS ON.  The exact-trace oracle decided the ACCURACY '
        'case in its favour (BUILD_INVERSE_MAP S6) and the one guard that '
        'blocked the flip -- G8, whose held-out probe sat ON the launch '
        'lattice where an s=0 spline incumbent is exact by construction -- '
        'was re-architected onto off-lattice probe points rather than '
        'weakened (FIX_G8_PROBE_2026_08_12).  The era table must move with '
        'the default or every era arm silently means something else.')
=======
    assert IM.TRACED_INVERSE_MAP is False, (
        'the evaluator SHIPS OFF -- the exact-trace oracle decided the '
        'ACCURACY case in its favour (S6), but the default flip is blocked '
        'by one shipped guard (S6.6)')


# ===========================================================================
# THE CACHE KEY COVERS THE ACCEPTANCE DECISION, NOT ONLY THE FIT
# ===========================================================================
# G8 is a COMPARATIVE bar -- the map is accepted only if it beats the
# INCUMBENT on held-out samples -- and the incumbent was the half of that bar
# the key did not hash.  So a map built against a weak incumbent was served to
# a call with a strong one, bypassing the single guard that decides whether
# the map may be used at all.  Production-reachable on this very element:
# ``newton_max_iters`` 2 -> 40, ``newton_poly_order`` 6 -> 8 and
# ``newton_fit`` 'polynomial' -> 'spline' each change ``_invert_newton``, and
# none of them changed the key (VERIFY_ARCHITECTURE P0-5).


def _cache_probe(**over):
    """One flag-ON call WITHOUT clearing the cache, so hits are observable."""
    kw = dict(prescription=_presc(), wavelength=_WL, dx=_DX,
              ray_subsample=_SUB, n_workers=1, on_undersample='silent',
              **_RD_KW)
    kw.update(over)
    rec = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out = la.apply_real_lens_traced(_field(), _imap_out=rec, **kw)
    return out, rec, IM.inverse_map_cache_info()


def _cold_incumbent_parity(**over):
    """G8's measured incumbent error for one configuration, on a COLD cache
    and with the flag on (otherwise no map builds and there is no record)."""
    old = IM.TRACED_INVERSE_MAP
    IM.inverse_map_cache_clear()
    try:
        IM.TRACED_INVERSE_MAP = True
        _e, rec, _i = _cache_probe(**over)
    finally:
        IM.TRACED_INVERSE_MAP = old
        IM.inverse_map_cache_clear()
    return rec.get('parity_incumbent_opl_waves'), rec.get('refused')


@pytest.mark.parametrize('knob,value', [
    ('newton_max_iters', 40),
    ('newton_poly_order', 8),
    ('newton_fit', 'spline'),
])
def test_the_cache_key_tracks_the_incumbent_by_content(knob, value):
    """THE CONTRACT, stated as content rather than as a knob list: the cache
    may serve a stored map ACROSS a configuration change IF AND ONLY IF the
    incumbent's answers did not move.

    Each knob here changes ``_invert_newton``, i.e. G8's arm B -- but on this
    element they do not all change its ANSWERS.  Measured (2026-08-12):

    .. code-block:: text

        newton_max_iters 2 -> 40    incumbent 1.996204432768027e-05, both
                                    -- Newton converges inside 2 steps here,
                                       so a hit is CORRECT
        newton_poly_order 6 -> 8    1.9962e-05 -> 2.6992e-07  (moves)
        newton_fit poly -> spline   1.9962e-05 -> 3.2540e-07  (moves, and
                                    G8 flips from accept to REFUSE)

    That last row is P0-5 itself: pre-fix the cache served the poly arm's map
    to the spline arm, engaging a map the shipped comparative bar refuses.
    So this test does not assert "these three must miss" -- it asks the
    incumbent what it does, and pins the key to that.
    """
    base_parity, _base_refused = _cold_incumbent_parity()
    mod_parity, _mod_refused = _cold_incumbent_parity(**{knob: value})
    incumbent_moved = (base_parity != mod_parity)

    old = IM.TRACED_INVERSE_MAP
    IM.inverse_map_cache_clear()
    try:
        IM.TRACED_INVERSE_MAP = True
        _e0, _r0, i0 = _cache_probe()
        assert i0['size'] >= 1, 'the map did not build, so nothing is cached'
        # control: the very same call HITS
        _e1, r1, i1 = _cache_probe()
        assert i1['hits'] == i0['hits'] + 1, (
            'an identical repeat must hit -- the cache has to work before '
            'its key can be tested')
        assert r1.get('cached') is True, (
            "rec['cached'] must report the hit; it is the channel every "
            "probe of this cache reads")
        _e2, _r2, i2 = _cache_probe(**{knob: value})
        if incumbent_moved:
            assert i2['hits'] == i1['hits'], (
                f"changing {knob} to {value!r} moved the incumbent G8 scores "
                f"against ({base_parity!r} -> {mod_parity!r}), but the cache "
                f"still served the old map (hits {i1['hits']} -> "
                f"{i2['hits']}) -- G8 has been bypassed")
            assert i2['misses'] == i1['misses'] + 1
        else:
            assert i2['hits'] == i1['hits'] + 1, (
                f"changing {knob} to {value!r} left the incumbent "
                f"bit-identical ({base_parity!r}), so G8's verdict cannot "
                f"differ and the stored map is still the right answer -- "
                f"missing here would be a false invalidation")
    finally:
        IM.TRACED_INVERSE_MAP = old
        IM.inverse_map_cache_clear()


def test_a_map_the_guard_would_refuse_is_not_served_from_the_cache():
    """P0-5 END TO END, on the one knob that flips the verdict.

    ``newton_fit='spline'`` gives an incumbent 61x tighter than the
    polynomial arm's, and against it G8 REFUSES the map.  Pre-fix the poly
    arm's accepted map was served to the spline arm from the cache, so the
    element ran with a map its own comparative bar rejects."""
    poly_parity, poly_refused = _cold_incumbent_parity()
    spl_parity, spl_refused = _cold_incumbent_parity(newton_fit='spline')
    assert poly_refused is None, (
        'the polynomial arm must ACCEPT the map, or there is nothing to '
        'wrongly serve')
    assert spl_refused == 'G8', (
        f'the spline arm must REFUSE on G8 (got {spl_refused!r}); its '
        f'incumbent is {spl_parity!r} against the poly arm\'s {poly_parity!r}')

    old = IM.TRACED_INVERSE_MAP
    IM.inverse_map_cache_clear()
    try:
        IM.TRACED_INVERSE_MAP = True
        _cache_probe()                                   # cache the poly map
        _e, rec, _i = _cache_probe(newton_fit='spline')
        assert rec.get('cached') is not True, (
            'the spline arm was served the polynomial arm\'s cached map')
        assert rec.get('refused') == 'G8', (
            f"the spline arm must still be refused by G8 with the cache warm "
            f"(got refused={rec.get('refused')!r}, cached="
            f"{rec.get('cached')!r})")
    finally:
        IM.TRACED_INVERSE_MAP = old
        IM.inverse_map_cache_clear()


def test_the_incumbent_fingerprint_is_by_evaluation_not_by_parameter_name():
    """The fingerprint must move when the incumbent's ANSWERS move, whatever
    made them move -- the module's own doctrine that a key naming the
    CONFIGURATION rather than the CONTENT is how a cache becomes a cache of
    something else.  A closure over the element's forward fits has no named
    knob at all."""
    XO, YO = np.meshgrid(np.linspace(-1e-3, 1e-3, 9),
                         np.linspace(-1e-3, 1e-3, 9))

    def inc(scale):
        def f(xq, yq):
            xq = np.asarray(xq, dtype=np.float64).ravel()
            yq = np.asarray(yq, dtype=np.float64).ravel()
            return xq * scale, yq * scale, (xq ** 2 + yq ** 2) * scale
        return f

    a = IM._incumbent_fingerprint(inc(1.0), XO, YO)
    b = IM._incumbent_fingerprint(inc(1.0), XO, YO)
    c = IM._incumbent_fingerprint(inc(1.0 + 1e-12), XO, YO)
    assert a == b, 'the fingerprint is not deterministic'
    assert a != c, 'a changed incumbent produced the same fingerprint'
    # no incumbent, and one that REFUSES these points, are DISTINCT states,
    # not crashes.  ``ValueError`` is the refusal scipy's
    # ``RectBivariateSpline.ev`` raises off its knot rectangle, i.e. the real
    # way an incumbent declines a strided probe point.
    assert IM._incumbent_fingerprint(None, XO, YO) not in (a, c)

    def boom(xq, yq):
        raise ValueError('outside the knot rectangle')

    assert IM._incumbent_fingerprint(boom, XO, YO) not in (a, c)

    # ... and the probe is NARROW (AUDIT_V4_12_1 L5): a DEFECT is not an
    # incumbent identity and must not be hashed into the cache key as one.
    # It costs nothing to let it out -- the shipped G8 arm calls
    # ``parity_invert`` unguarded, so a swallowed bug surfaces there anyway,
    # after the cache has been keyed on ``<raised:...>``.
    def bug(xq, yq):
        raise AttributeError("'NoneType' object has no attribute 'ev'")

    with pytest.raises(AttributeError):
        IM._incumbent_fingerprint(bug, XO, YO)

    # and it reaches the key
    args = [np.linspace(-1e-3, 1e-3, 9)] + [np.zeros((9, 9))] * 4 + [None]
    k_a = IM._imap_key(*args, 14, 1e-3, _WL, incumbent_fp=a)
    k_c = IM._imap_key(*args, 14, 1e-3, _WL, incumbent_fp=c)
    assert k_a != k_c


def test_the_cache_key_moves_with_the_det_j_source_and_the_census():
    """``_IMAP_DETJ_SOURCE`` selects the Jacobian the AMPLITUDE channel
    consumes (a stale hit served a det J 3.12e-03 relative wrong), and
    ``census_amp`` names where G2/G7/G8 are scored (a stale hit reported
    ``n_detj_census`` 1681 where a cold build reads 553)."""
    args = [np.linspace(-1e-3, 1e-3, 9)] + [np.zeros((9, 9))] * 4 + [None]
    k0 = IM._imap_key(*args, 14, 1e-3, _WL)
    old = IM._IMAP_DETJ_SOURCE
    try:
        other = [s for s in IM._IMAP_DETJ_SOURCES if s != old]
        assert other, 'there is only one det J source to choose from'
        IM._IMAP_DETJ_SOURCE = other[0]
        assert IM._imap_key(*args, 14, 1e-3, _WL) != k0
    finally:
        IM._IMAP_DETJ_SOURCE = old
    assert IM._imap_key(*args, 14, 1e-3, _WL,
                        census_amp=np.ones((9, 9))) != k0
>>>>>>> origin/main
